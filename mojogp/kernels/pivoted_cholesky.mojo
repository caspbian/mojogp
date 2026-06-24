"""Pivoted Cholesky Preconditioner for GP Training.

This module implements the low-rank Pivoted Cholesky preconditioner from the
GPyTorch paper (arXiv:1809.11165v6). This is critical for achieving fast
CG convergence with RBF kernels.

Key insight from the paper:
- Jacobi preconditioning has NO effect for stationary kernels (RBF, Matern)
- Pivoted Cholesky with rank k=5-10 dramatically reduces CG iterations
- The paper proves exponential convergence for RBF kernels (Theorem 1)

Algorithm:
1. Build low-rank approximation: K ≈ L @ L^T where L is n × k
2. Preconditioner: P = L @ L^T + noise * I
3. Apply P^{-1} using Woodbury identity in O(n * k²) time:
   P^{-1} = (1/noise) * (I - L @ (L^T @ L + noise * I)^{-1} @ L^T)

Expected speedup: 2-10x fewer CG iterations
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from gpu.primitives.warp import sum as warp_sum
from gpu.sync import barrier
from gpu.memory import AddressSpace, external_memory
from gpu.globals import WARP_SIZE
from memory import UnsafePointer, stack_allocation
from math import sqrt, log, atan2, cos, sin
from time import perf_counter_ns
from buffer import NDBuffer
from linalg.matmul import matmul as max_matmul
from collections import Optional
from random import randn, seed as random_seed
from buffer import DimList

from gpu.profiler import ProfileBlock

from .matvec_provider import MatvecProvider
from .gradient_provider import ForwardProvider, GradientProvider
from .preconditioner_trait import Preconditioner
from .cg_solver import kernel_dot_batched, kernel_copy
from .native_numerics import matrix_inv_native, compute_slogdet_native
from .constants import PROFILING
from memory.unsafe_pointer import alloc


# =============================================================================
# GPU Random Number Generation (replaces CPU randn + H2D copy)
# =============================================================================

fn kernel_gpu_gaussian(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    seed: UInt64,
) -> None:
    """GPU Gaussian random number generator using Philox hash + Box-Muller.
    
    Generates approximately Gaussian distributed values with zero mean and unit
    variance. Uses a hash-based counter RNG (Philox-style) for uniform randoms,
    then Box-Muller transform for Gaussian conversion.
    
    This replaces CPU randn() + H2D copy, which takes 15ms for 100K floats.
    GPU generation takes <0.01ms.
    """
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if idx >= n:
        return
    
    # Philox-style hash for uniform random in (0, 1)
    var state1 = (UInt64(idx) + UInt64(1)) * UInt64(6364136223846793005) + seed
    state1 = state1 ^ (state1 >> 33)
    state1 = state1 * UInt64(0xff51afd7ed558ccd)
    state1 = state1 ^ (state1 >> 33)
    
    var state2 = (UInt64(idx) + UInt64(1)) * UInt64(1442695040888963407) + seed + UInt64(0xBEEF)
    state2 = state2 ^ (state2 >> 33)
    state2 = state2 * UInt64(0xc4ceb9fe1a85ec53)
    state2 = state2 ^ (state2 >> 33)
    
    # Convert to uniform (0, 1) — avoid exact 0 for log()
    var u1 = (Float32(Int64(state1 & UInt64(0x7FFFFF))) + Float32(1.0)) / Float32(0x800000)
    var u2 = Float32(Int64(state2 & UInt64(0xFFFFFF))) / Float32(0x1000000)
    
    # Box-Muller transform: z = sqrt(-2 ln(u1)) * cos(2π u2)
    var z = sqrt(Float32(-2.0) * log(u1)) * cos(Float32(2.0) * Float32(3.14159265) * u2)
    out_ptr[idx] = z

alias float_dtype = DType.float32
alias PI = Float32(3.14159265358979323846)


# =============================================================================
# Data Structures
# =============================================================================

struct PivotedCholeskyPrecond(Preconditioner, Copyable, Movable):
    """Low-rank Pivoted Cholesky preconditioner.
    
    Stores L such that K ≈ L @ L^T, where L is n × rank.
    The preconditioner is P = L @ L^T + noise * I.
    
    Implements the Preconditioner trait for use in generic CG solvers.
    
    Fields:
        L: Low-rank factor [n × rank] column-major
        noise: Noise variance
        rank: Rank of the approximation
        n: Number of data points
        LTL_plus_noise_inv: (L^T @ L + noise * I)^{-1} [rank × rank] for Woodbury (host)
        LTL_plus_noise_inv_device: Same as above but cached on GPU for fast preconditioner application
        max_num_cols: Maximum number of columns for work buffers
        w_work: Cached work buffer [rank × max_num_cols] for L^T @ v
        u_work: Cached work buffer [rank × max_num_cols] for LTL_inv @ w
        z_work: Cached work buffer [n × max_num_cols] for L @ u
    """
    var L: DeviceBuffer[float_dtype]
    var noise: Float32
    var rank: Int
    var n: Int
    var LTL_plus_noise_inv: HostBuffer[float_dtype]  # Small matrix on host (for CPU version)
    var LTL_plus_noise_inv_device: DeviceBuffer[float_dtype]  # Cached on GPU (for fast GPU version)
    var max_num_cols: Int  # Maximum number of columns for work buffers
    var w_work: DeviceBuffer[float_dtype]  # Cached work buffer [rank × max_num_cols]
    var u_work: DeviceBuffer[float_dtype]  # Cached work buffer [rank × max_num_cols]
    var z_work: DeviceBuffer[float_dtype]  # Cached work buffer [n × max_num_cols]
    var cached_log_det: Float32  # Cached log|P| computed once at construction
    var noise_mode: Int  # 0=scalar noise, 1=fixed vector noise
    var noise_vec_ptr: UnsafePointer[Float32, MutAnyOrigin]
    
    fn __init__(out self, var L: DeviceBuffer[float_dtype], noise: Float32, rank: Int, n: Int,
                var LTL_plus_noise_inv: HostBuffer[float_dtype],
                var LTL_plus_noise_inv_device: DeviceBuffer[float_dtype],
                 max_num_cols: Int,
                 var w_work: DeviceBuffer[float_dtype],
                 var u_work: DeviceBuffer[float_dtype],
                 var z_work: DeviceBuffer[float_dtype],
                 cached_log_det: Float32 = 0.0,
                 noise_mode: Int = 0,
                 noise_vec_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin]()):
        self.L = L^
        self.noise = noise
        self.rank = rank
        self.n = n
        self.LTL_plus_noise_inv = LTL_plus_noise_inv^
        self.LTL_plus_noise_inv_device = LTL_plus_noise_inv_device^
        self.max_num_cols = max_num_cols
        self.w_work = w_work^
        self.u_work = u_work^
        self.z_work = z_work^
        self.cached_log_det = cached_log_det
        self.noise_mode = noise_mode
        self.noise_vec_ptr = noise_vec_ptr
    
    fn apply_precond(
        self,
        ctx: DeviceContext,
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        num_cols: Int,
        sync: Bool,
    ) raises:
        """Apply P^{-1} @ v using Woodbury identity on GPU.
        
        Delegates to apply_pivoted_cholesky_precond_gpu.
        """
        apply_pivoted_cholesky_precond_gpu(ctx, self, v_ptr, out_ptr, n, num_cols, sync)
    
    fn sample_probes(
        self,
        ctx: DeviceContext,
        out_device: DeviceBuffer[float_dtype],
        num_probes: Int,
        seed_val: UInt64,
    ) raises:
        """Sample probe vectors from N(0, P) where P = L L^T + noise I.
        
        Allocates temporary buffers internally and writes into out_device.
        Uses kernel_copy for exact element count to handle pool buffers
        that may have headroom (out_device larger than n * num_probes).
        """
        var result = sample_from_preconditioner_gpu(ctx, self, num_probes, seed_val)
        # Copy exactly n * num_probes elements (out_device may be larger due to pool headroom)
        var total = self.n * num_probes
        ctx.enqueue_function[kernel_copy](
            out_device.unsafe_ptr(), result.unsafe_ptr(), total,
            grid_dim=((total + 255) // 256,), block_dim=(256,)
        )
        ctx.synchronize()
    
    fn log_det(self, ctx: DeviceContext) raises -> Float32:
        """Return cached log|P| = log|L L^T + noise I|.
        
        The value is computed once at construction time and cached.
        It only changes when the preconditioner is rebuilt.
        """
        return self.cached_log_det


# =============================================================================
# GPU Kernels for Pivoted Cholesky
# =============================================================================

fn kernel_compute_diagonal_residual(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    current_rank: Int,
) -> None:
    """Compute diagonal residual: diag[i] -= L[i, current_rank-1]^2.
    
    This updates the diagonal after adding a new column to L.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    # L is column-major: L[i, j] = L_ptr[j * n + i]
    var L_val = L_ptr[UInt(current_rank - 1) * UInt(n) + idx]
    diag_ptr[idx] -= L_val * L_val


fn kernel_compute_L_column(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    Kv_ptr: UnsafePointer[Float32, MutAnyOrigin],
    pivot_idx: Int,
    n: Int,
    col_idx: Int,
    pivot_val_sqrt: Float32,
) -> None:
    """Compute new column of L: L[:, col_idx] = K[:, pivot] / sqrt(diag[pivot]).
    
    Then orthogonalize against previous columns.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    # L[:, col_idx] = Kv / sqrt(diag[pivot])
    # L is column-major: L[i, j] = L_ptr[j * n + i]
    L_ptr[UInt(col_idx) * UInt(n) + idx] = Kv_ptr[idx] / pivot_val_sqrt


fn kernel_orthogonalize_L_column(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_idx: Int,
    prev_col_idx: Int,
    coeff: Float32,
) -> None:
    """Orthogonalize L[:, col_idx] against L[:, prev_col_idx].
    
    L[:, col_idx] -= coeff * L[:, prev_col_idx]
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    # L is column-major
    var curr_offset = UInt(col_idx) * UInt(n) + idx
    var prev_offset = UInt(prev_col_idx) * UInt(n) + idx
    L_ptr[curr_offset] -= coeff * L_ptr[prev_offset]


fn kernel_apply_L_transpose(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    rank: Int,
) -> None:
    """Compute L^T @ v where L is n × rank and v is n × 1.
    
    Result is rank × 1.
    """
    var col = block_idx.x * block_dim.x + thread_idx.x
    
    if col >= UInt(rank):
        return
    
    # L^T @ v: out[col] = sum_i L[i, col] * v[i]
    var sum_val = Float32(0.0)
    for i in range(n):
        # L is column-major: L[i, col] = L_ptr[col * n + i]
        sum_val += L_ptr[UInt(col) * UInt(n) + UInt(i)] * v_ptr[i]
    
    out_ptr[col] = sum_val


fn kernel_apply_L(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    rank: Int,
) -> None:
    """Compute L @ v where L is n × rank and v is rank × 1.
    
    Result is n × 1.
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    
    if row >= UInt(n):
        return
    
    # L @ v: out[row] = sum_j L[row, j] * v[j]
    var sum_val = Float32(0.0)
    for j in range(rank):
        # L is column-major: L[row, j] = L_ptr[j * n + row]
        sum_val += L_ptr[UInt(j) * UInt(n) + row] * v_ptr[j]
    
    out_ptr[row] = sum_val


fn kernel_scale_and_subtract(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    sub_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scale: Float32,
    n: Int,
) -> None:
    """Compute out = scale * v - sub."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    out_ptr[idx] = scale * v_ptr[idx] - sub_ptr[idx]


fn kernel_small_matmul(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [rank × num_cols] column-major
    A_ptr: UnsafePointer[Float32, MutAnyOrigin],    # [rank × rank] row-major
    w_ptr: UnsafePointer[Float32, MutAnyOrigin],    # [rank × num_cols] column-major
    rank: Int,
    num_cols: Int,
) -> None:
    """Compute out = A @ w where A is [rank × rank] and w is [rank × num_cols].
    
    For small rank (5-15), this is efficient with one thread per output element.
    
    Args:
        out_ptr: Output buffer [rank, num_cols] column-major
        A_ptr: Matrix A [rank, rank] row-major
        w_ptr: Matrix w [rank, num_cols] column-major
        rank: Rank dimension
        num_cols: Number of columns
    """
    var i = block_idx.x * block_dim.x + thread_idx.x  # row (0..rank)
    var col = block_idx.y * block_dim.y + thread_idx.y  # column (0..num_cols)
    
    if i >= UInt(rank) or col >= UInt(num_cols):
        return
    
    var sum_val = Float32(0.0)
    for j in range(rank):
        # A row-major: A[i, j] = A_ptr[i * rank + j]
        # w column-major: w[j, col] = w_ptr[col * rank + j]
        sum_val += A_ptr[UInt(i) * UInt(rank) + UInt(j)] * w_ptr[UInt(col) * UInt(rank) + UInt(j)]
    
    # out column-major: out[i, col] = out_ptr[col * rank + i]
    out_ptr[UInt(col) * UInt(rank) + UInt(i)] = sum_val


fn kernel_woodbury_final(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n × num_cols] column-major
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],    # [n × num_cols] column-major
    z_ptr: UnsafePointer[Float32, MutAnyOrigin],    # [n × num_cols] column-major
    noise_inv: Float32,
    n: Int,
    num_cols: Int,
) -> None:
    """Compute out = (1/noise) * (v - z).
    
    This is the final step of the Woodbury identity application.
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major
        v_ptr: Input v [n, num_cols] column-major
        z_ptr: Input z [n, num_cols] column-major
        noise_inv: 1/noise
        n: Vector dimension
        num_cols: Number of columns
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    
    if i >= UInt(n) or col >= UInt(num_cols):
        return
    
    var idx = UInt(col) * UInt(n) + UInt(i)
    out_ptr[idx] = noise_inv * (v_ptr[idx] - z_ptr[idx])


fn kernel_woodbury_final_vector(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    z_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Compute out = D^-1 @ (v - z) for vector diagonal D."""
    var i = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    if i >= UInt(n) or col >= UInt(num_cols):
        return
    var idx = UInt(col) * UInt(n) + UInt(i)
    var noise = noise_ptr[i]
    out_ptr[idx] = (v_ptr[idx] - z_ptr[idx]) / noise


fn kernel_row_scale_by_inv_noise(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Compute out[row, col] = in[row, col] / noise[row]."""
    var i = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    if i >= UInt(n) or col >= UInt(num_cols):
        return
    var idx = UInt(col) * UInt(n) + UInt(i)
    out_ptr[idx] = in_ptr[idx] / noise_ptr[i]


# =============================================================================
# GPU Kernels for Preconditioner Construction
# =============================================================================

fn kernel_init_diagonal(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    value: Float32,
) -> None:
    """Initialize diagonal to a constant value."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    diag_ptr[idx] = value


fn kernel_create_unit_vector(
    e_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    pivot_idx: Int,
) -> None:
    """Create unit vector with 1.0 at pivot_idx, 0.0 elsewhere."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    e_ptr[idx] = Float32(1.0) if Int(idx) == pivot_idx else Float32(0.0)


fn kernel_subtract_at_index(
    ptr: UnsafePointer[Float32, MutAnyOrigin],
    idx: Int,
    value: Float32,
) -> None:
    """Subtract value from ptr[idx]. Single thread kernel."""
    if thread_idx.x == 0 and block_idx.x == 0:
        ptr[idx] -= value


fn kernel_subtract_vector_at_index(
    ptr: UnsafePointer[Float32, MutAnyOrigin],
    idx: Int,
    values: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Subtract values[idx] from ptr[idx]. Single thread kernel."""
    if thread_idx.x == 0 and block_idx.x == 0:
        ptr[idx] -= values[idx]


fn kernel_set_value_at_index(
    ptr: UnsafePointer[Float32, MutAnyOrigin],
    idx: Int,
    value: Float32,
) -> None:
    """Set ptr[idx] = value. Single thread kernel."""
    if thread_idx.x == 0 and block_idx.x == 0:
        ptr[idx] = value


fn kernel_extract_single_value(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    idx: Int,
) -> None:
    """Extract single value: out[0] = in[idx]. Single thread kernel."""
    if thread_idx.x == 0 and block_idx.x == 0:
        out_ptr[0] = in_ptr[idx]


fn kernel_masked_argmax_and_error(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mask_ptr: UnsafePointer[Float32, MutAnyOrigin],
    result_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """GPU-side masked argmax + absolute error sum over diagonal.

    Single block of 256 threads. Each thread scans n/256 elements, finding
    local max+index and summing |diag[i]| for unmasked entries (mask[i] > 0.5).

    Output (result_ptr): [pivot_idx_as_float, max_value, sum_abs_error]

    Launch: grid_dim=(1,), block_dim=(256,)
    """
    var tid = Int(thread_idx.x)
    var num_threads = Int(block_dim.x)  # 256
    var warp_id = tid // WARP_SIZE      # 0-7
    var lane = tid % WARP_SIZE          # 0-31

    alias NUM_WARPS = 8

    # Shared memory: [warp_max_val, warp_max_idx, warp_abs_sum] per warp = 3 * 8 = 24 floats
    var shared = stack_allocation[24, Float32, address_space = AddressSpace.SHARED]()
    # Layout: shared[0..7] = max_val, shared[8..15] = max_idx, shared[16..23] = abs_sum

    # Each thread scans its assigned portion.
    var local_max_val = Float32(-1e30)
    var local_max_idx = Int(-1)
    var local_abs_sum = Float32(0.0)

    var i = tid
    while i < n:
        var m = mask_ptr[UInt(i)]
        if m > Float32(0.5):
            # This index is available (not used as pivot)
            var d = diag_ptr[UInt(i)]
            if d > local_max_val or (
                d == local_max_val and (local_max_idx < 0 or i < local_max_idx)
            ):
                local_max_val = d
                local_max_idx = i
            # Accumulate |diag[i]|
            var abs_d = d
            if abs_d < Float32(0.0):
                abs_d = -abs_d
            local_abs_sum += abs_d
        i += num_threads

    # Reduce abs_sum within each warp.
    var warp_abs = warp_sum(local_abs_sum)

    # Store warp abs_sum results (lane 0 only)
    if lane == 0:
        shared[16 + warp_id] = warp_abs

    # Reduce argmax across warps through shared memory.
    # Each thread writes its val/idx, then tree reduction across all 256 threads
    var warp_argmax_val = stack_allocation[NUM_WARPS * WARP_SIZE, Float32, address_space = AddressSpace.SHARED]()
    var warp_argmax_idx = stack_allocation[NUM_WARPS * WARP_SIZE, Float32, address_space = AddressSpace.SHARED]()

    warp_argmax_val[tid] = local_max_val
    warp_argmax_idx[tid] = Float32(local_max_idx)
    barrier()

    # Tree reduction: 5 rounds for 32 lanes within each warp, then cross-warp
    var stride = 16
    while stride >= 1:
        if lane < stride:
            var other_val = warp_argmax_val[tid + stride]
            var other_idx = warp_argmax_idx[tid + stride]
            var current_idx = Int(warp_argmax_idx[tid])
            var other_idx_int = Int(other_idx)
            if other_val > warp_argmax_val[tid] or (
                other_val == warp_argmax_val[tid]
                and other_idx_int >= 0
                and (current_idx < 0 or other_idx_int < current_idx)
            ):
                warp_argmax_val[tid] = other_val
                warp_argmax_idx[tid] = other_idx
        barrier()
        stride //= 2

    # Lane 0 of each warp now holds the warp's best; write to cross-warp shared region
    if lane == 0:
        shared[warp_id] = warp_argmax_val[tid]
        shared[8 + warp_id] = warp_argmax_idx[tid]

    barrier()

    # Thread 0 reduces across 8 warps for final argmax and total abs sum
    if tid == 0:
        var best_val = shared[0]
        var best_idx = shared[8]
        var total_abs = shared[16]
        for w in range(1, NUM_WARPS):
            var other_idx = Int(shared[8 + w])
            var best_idx_int = Int(best_idx)
            if shared[w] > best_val or (
                shared[w] == best_val
                and other_idx >= 0
                and (best_idx_int < 0 or other_idx < best_idx_int)
            ):
                best_val = shared[w]
                best_idx = shared[8 + w]
            total_abs += shared[16 + w]

        result_ptr[0] = best_idx   # pivot index as float
        result_ptr[1] = best_val   # max diagonal value
        result_ptr[2] = total_abs  # sum of |diag[i]| for unmasked entries


fn kernel_gather_L_coefficients(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    pivot_idx: Int,
    n: Int,
    num_coeffs: Int,
) -> None:
    """Gather L[j, pivot_idx] for j = 0..num_coeffs-1 into contiguous buffer.
    
    Batches coefficient gathering into one kernel launch.
    L is column-major: L[i, col] = L_ptr[col * n + i]
    So L[j, pivot_idx] is at L_ptr[j * n + pivot_idx]
    
    Args:
        out_ptr: Output buffer [num_coeffs]
        L_ptr: L matrix [n × rank] column-major
        pivot_idx: Current pivot index
        n: Matrix dimension
        num_coeffs: Number of coefficients to gather (= current rank iteration m)
    """
    var j = block_idx.x * block_dim.x + thread_idx.x
    if j >= UInt(num_coeffs):
        return
    # L[j, pivot_idx] = L_ptr[j * n + pivot_idx]
    out_ptr[j] = L_ptr[UInt(j) * UInt(n) + UInt(pivot_idx)]


fn kernel_orthogonalize_L_column_batched(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    coeffs_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    m: Int,
    num_prev_cols: Int,
) -> None:
    """Orthogonalize L[:, m] against all previous columns using pre-gathered coefficients.
    
    Applies L[:, m] -= sum_j coeffs[j] * L[:, j] in one kernel.
    
    Args:
        L_ptr: L matrix [n × rank] column-major
        coeffs_ptr: Pre-gathered coefficients [num_prev_cols] = L[j, pivot_idx] for j < m
        n: Matrix dimension
        m: Current column index
        num_prev_cols: Number of previous columns (= m)
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= UInt(n):
        return
    
    # L[i, m] -= sum_j coeffs[j] * L[i, j]
    var sum = Float32(0.0)
    for j in range(num_prev_cols):
        sum += coeffs_ptr[j] * L_ptr[UInt(j) * UInt(n) + i]
    
    L_ptr[UInt(m) * UInt(n) + i] -= sum


fn kernel_scale_and_store_column(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    Kv_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scale: Float32,
    n: Int,
    col_idx: Int,
) -> None:
    """Store scaled column: L[:, col_idx] = Kv * scale."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    # L is column-major: L[i, col] = L_ptr[col * n + i]
    L_ptr[UInt(col_idx) * UInt(n) + idx] = Kv_ptr[idx] * scale


fn kernel_scale_column_inplace(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_idx: Int,
    scale: Float32,
) -> None:
    """Scale column in place: L[:, col_idx] *= scale."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    # L is column-major: L[i, col] = L_ptr[col * n + i]
    var offset = UInt(col_idx) * UInt(n) + idx
    L_ptr[offset] = L_ptr[offset] * scale


fn kernel_update_diagonal_from_L(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_idx: Int,
) -> None:
    """Update diagonal: diag[i] -= L[i, col_idx]^2."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    # L is column-major: L[i, col] = L_ptr[col * n + i]
    var L_val = L_ptr[UInt(col_idx) * UInt(n) + idx]
    diag_ptr[idx] -= L_val * L_val


fn kernel_update_diagonal_from_L_masked(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mask_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_idx: Int,
) -> None:
    """Update diagonal only for available indices: diag[i] -= mask[i] * L[i, col_idx]^2.
    
    The mask should be 1.0 for indices that haven't been used as pivots yet,
    and 0.0 for indices that have been used as pivots.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    
    # Only update if this index is still available (mask > 0.5)
    if mask_ptr[idx] > Float32(0.5):
        # L is column-major: L[i, col] = L_ptr[col * n + i]
        var L_val = L_ptr[UInt(col_idx) * UInt(n) + idx]
        diag_ptr[idx] -= L_val * L_val


fn kernel_dot_columns_warp(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    result_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_j: Int,
    col_m: Int,
) -> None:
    """Compute dot product of two L columns using warp reduction.
    
    result[0] = L[:, col_j]^T @ L[:, col_m]
    """
    var lane = thread_idx.x
    var partial_sum = Float32(0.0)
    
    # Each lane processes multiple elements
    var idx = Int(lane)
    while idx < n:
        # L is column-major: L[i, col] = L_ptr[col * n + i]
        var L_j = L_ptr[UInt(col_j) * UInt(n) + UInt(idx)]
        var L_m = L_ptr[UInt(col_m) * UInt(n) + UInt(idx)]
        partial_sum += L_j * L_m
        idx += WARP_SIZE
    
    # Warp reduction
    var sum_val = warp_sum(partial_sum)
    
    # Lane 0 writes result
    if lane == 0:
        result_ptr[0] = sum_val


# =============================================================================
# Host Functions
# =============================================================================

fn build_pivoted_cholesky_precond_gpu[T: MatvecProvider](
    provider: T,
    rank: Int = 10,
    error_tol: Float32 = 1e-3,
    max_num_cols: Int = 16,
    precond_method: Int = 2,
    noise_for_adaptive: Float32 = 0.1,
    adaptive_tol: Float32 = 0.01,
    seed: UInt64 = 42,
) raises -> PivotedCholeskyPrecond:
    """Build low-rank Pivoted Cholesky preconditioner with GPU acceleration.
    
    This implementation minimizes CPU-GPU synchronization by:
    1. Keeping working data on GPU where possible
    2. Using GPU kernels for orthogonalization
    3. Only syncing when pivot info is needed on CPU
    4. Pre-allocating work buffers for preconditioner application
    
    Args:
        provider: MatvecProvider for kernel operations
        rank: Maximum rank of the approximation (default: 10)
        error_tol: Early stopping tolerance for greedy/rpcholesky (default: 1e-3)
        max_num_cols: Maximum number of columns for work buffers (default: 16)
        precond_method: Pivot selection method (default: 2)
            0 = greedy (deterministic argmax, GPyTorch-compatible)
            1 = rpcholesky (randomized proportional sampling, fixed rank)
            2 = nystrom (rpcholesky + adaptive rank based on noise floor)
        noise_for_adaptive: Noise variance for Nystrom adaptive stopping (default: 0.1)
        adaptive_tol: Nystrom stopping threshold (default: 0.01).
            Stops when residual_trace < adaptive_tol * n * noise.
        seed: Random seed for rpcholesky/nystrom pivot sampling (default: 42)
        
    Returns:
        PivotedCholeskyPrecond ready for use in CG
    """
    var t_start = perf_counter_ns()
    
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var noise = provider.get_noise()
    var outputscale = provider.get_outputscale()
    
    alias BLOCK_SIZE = 256
    var num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # =========================================================================
    # Allocate GPU buffers.
    # =========================================================================
    
    # L matrix [n × rank] column-major on GPU
    var L_device = ctx.enqueue_create_buffer[float_dtype](n * rank)
    
    # Initialize L to zero on GPU (no CPU loop, no H2D copy)
    from .cg_solver import kernel_fill_constant
    var fill_blocks = (n * rank + 255) // 256
    ctx.enqueue_function[kernel_fill_constant](
        L_device.unsafe_ptr(), n * rank, Float32(0.0),
        grid_dim=fill_blocks, block_dim=256)
    
    # Diagonal values on GPU
    var diag_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_function[kernel_init_diagonal](
        diag_device.unsafe_ptr(), n, outputscale,
        grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
    )
    
    # Mask for tracking which indices are still available (1.0 = available, 0.0 = used as pivot)
    var mask_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_function[kernel_init_diagonal](
        mask_device.unsafe_ptr(), n, Float32(1.0),
        grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
    )
    
    # GPU argmax result buffer: [pivot_idx_as_float, max_value, sum_abs_error]
    var argmax_result_device = ctx.enqueue_create_buffer[float_dtype](3)
    var argmax_result_host = ctx.enqueue_create_host_buffer[float_dtype](3)
    
    # Unit vector for matvec
    var e_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Result of K @ e
    var Ke_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Dot product result buffer
    var dot_result_device = ctx.enqueue_create_buffer[float_dtype](1)
    var dot_result_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    ctx.synchronize()  # SYNC 1: After initialization
    
    var t_init = perf_counter_ns()
    
    # =========================================================================
    # Construct the factor iteratively.
    # =========================================================================
    
    var actual_rank = rank
    var orig_error = outputscale
    
    # For rpcholesky/nystrom: need host-side diagonal for proportional sampling
    var diag_host_sampling = ctx.enqueue_create_host_buffer[float_dtype](n)
    # Track which indices are used as pivots (host-side, for sampling)
    var used_as_pivot = List[Bool]()
    for _i in range(n):
        used_as_pivot.append(False)
    # Mutable seed state for rpcholesky/nystrom
    var rng_state = seed
    
    for m in range(rank):
        # Step 1+2: GPU-side masked argmax + error sum (eliminates n-element D2H copy)
        ctx.enqueue_function[kernel_masked_argmax_and_error](
            diag_device.unsafe_ptr(), mask_device.unsafe_ptr(),
            argmax_result_device.unsafe_ptr(), n,
            grid_dim=(1,), block_dim=(256,),
        )
        ctx.enqueue_copy(dst_buf=argmax_result_host, src_buf=argmax_result_device)
        ctx.synchronize()  # SYNC: Need argmax result for pivot selection
        
        var greedy_pivot_idx = Int(argmax_result_host[0])
        var max_diag_value = argmax_result_host[1]
        var sum_abs_error = argmax_result_host[2]
        
        # Check early termination (common to all methods)
        if max_diag_value < Float32(1e-10) or greedy_pivot_idx < 0:
            actual_rank = m
            if actual_rank == 0:
                actual_rank = 1
            break
        
        # Stopping criterion depends on method
        if precond_method == 2:
            # Nystrom: adaptive rank based on noise floor
            # sum_abs_error from the GPU kernel is sum of |diag[i]| for unmasked entries
            # For positive residual diag, this approximates the residual trace
            var residual_trace = sum_abs_error
            var threshold = adaptive_tol * Float32(n) * noise_for_adaptive
            if residual_trace < threshold and m > 0:
                actual_rank = m
                if actual_rank == 0:
                    actual_rank = 1
                break
        else:
            # Greedy / RPCholesky: fixed relative trace threshold
            var current_error = sum_abs_error / orig_error
            if current_error < error_tol:
                actual_rank = m
                if actual_rank == 0:
                    actual_rank = 1
                break
        
        # Pivot selection depends on method
        var pivot_idx: Int
        if precond_method == 0:
            # Greedy: use GPU argmax result directly
            pivot_idx = greedy_pivot_idx
        else:
            # RPCholesky / Nystrom: proportional sampling on CPU
            # Copy diagonal to host for sampling
            ctx.enqueue_copy(dst_buf=diag_host_sampling, src_buf=diag_device)
            ctx.synchronize()
            
            # Sum residual diagonal for sampling probabilities
            var total = Float32(0.0)
            for i in range(n):
                if not used_as_pivot[i]:
                    var d = diag_host_sampling[i]
                    if d > Float32(0.0):
                        total += d
            
            if total < Float32(1e-10):
                # All residual gone, early terminate
                actual_rank = m
                if actual_rank == 0:
                    actual_rank = 1
                break
            
            # Generate random uniform using xorshift64
            rng_state ^= rng_state >> 12
            rng_state ^= rng_state << 25
            rng_state ^= rng_state >> 27
            rng_state *= UInt64(0x2545F4914F6CDD1D)
            # Convert to float in [0, 1)
            var u = Float32(Int(rng_state & UInt64(0x7FFFFFFF))) / Float32(2147483648.0)
            u = u * total
            
            # Sample pivot proportional to max(diag[i], 0)
            var cumsum = Float32(0.0)
            pivot_idx = greedy_pivot_idx  # fallback
            for i in range(n):
                if not used_as_pivot[i]:
                    var d = diag_host_sampling[i]
                    if d > Float32(0.0):
                        cumsum += d
                    if cumsum >= u:
                        pivot_idx = i
                        max_diag_value = diag_host_sampling[i]
                        break
        
        # Track used pivots (host-side)
        used_as_pivot[pivot_idx] = True
        
        # Mark pivot as used in GPU mask (set to 0.0)
        ctx.enqueue_function[kernel_set_value_at_index](
            mask_device.unsafe_ptr(), pivot_idx, Float32(0.0),
            grid_dim=(1,), block_dim=(1,)
        )
        
        # Step 3: Create unit vector and compute K @ e_pivot (GPU)
        ctx.enqueue_function[kernel_create_unit_vector](
            e_device.unsafe_ptr(), n, pivot_idx,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
        
        # Matvec: Ke = K @ e_pivot
        provider.forward_matvec(Ke_device.unsafe_ptr(), e_device.unsafe_ptr(), 1)
        
        # Subtract noise from diagonal element
        ctx.enqueue_function[kernel_subtract_at_index](
            Ke_device.unsafe_ptr(), pivot_idx, noise,
            grid_dim=(1,), block_dim=(1,)
        )
        
        # Store raw Ke as the next L column before orthogonalization and scaling.
        ctx.enqueue_function[kernel_scale_and_store_column](
            L_device.unsafe_ptr(), Ke_device.unsafe_ptr(), Float32(1.0), n, m,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
        
        # Step 5: Cholesky orthogonalization against previous columns (GPU)
        # L[:, m] -= sum_j L[j, pivot_m] * L[:, j]
        # Batch-gather all coefficients, then orthogonalize in one kernel.
        # This reduces syncs from m to 1 per rank iteration (45→10 total for rank=10)
        if m > 0:
            # Allocate buffers for coefficients (reused across iterations would be better, but this is simpler)
            var coeffs_device = ctx.enqueue_create_buffer[float_dtype](m)
            var coeffs_host = ctx.enqueue_create_host_buffer[float_dtype](m)
            
            # Gather all L[j, pivot_idx] for j = 0..m-1 in one kernel
            ctx.enqueue_function[kernel_gather_L_coefficients](
                coeffs_device.unsafe_ptr(), L_device.unsafe_ptr(), pivot_idx, n, m,
                grid_dim=((m + 255) // 256,), block_dim=(256,)
            )
            # No sync needed - orthogonalization kernel will wait for gather to complete
            
            # Orthogonalize L[:, m] against all previous columns in one kernel
            ctx.enqueue_function[kernel_orthogonalize_L_column_batched](
                L_device.unsafe_ptr(), coeffs_device.unsafe_ptr(), n, m, m,
                grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
            )
            # No sync needed - scale kernel will wait for orthogonalization to complete
        
        # Step 5b: Now scale the orthogonalized column by 1/sqrt(diag[pivot])
        # For rpcholesky, max_diag_value was updated to diag[pivot_idx] above
        if max_diag_value < Float32(1e-10):
            max_diag_value = Float32(1e-10)  # Safety floor
        var scale = Float32(1.0) / sqrt(max_diag_value)
        # Scale L[:, m] in place
        ctx.enqueue_function[kernel_scale_column_inplace](
            L_device.unsafe_ptr(), n, m, scale,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
        
        # Step 6: Update diagonal (GPU) - only for indices not yet used as pivots
        ctx.enqueue_function[kernel_update_diagonal_from_L_masked](
            diag_device.unsafe_ptr(), L_device.unsafe_ptr(), mask_device.unsafe_ptr(), n, m,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
        

    
    ctx.synchronize()  # SYNC: After all iterations
    
    var t_construct = perf_counter_ns()
    
    # =========================================================================
    # Compute LTL_plus_noise_inv.
    # =========================================================================
    
    # No L resize needed: L is column-major, first actual_rank columns occupy
    # indices [0, actual_rank*n). Consumers use actual_rank, not rank.
    
    var LTL_plus_noise_inv = _compute_LTL_plus_noise_inv(ctx, L_device, n, actual_rank, noise)
    
    # Cache LTL_plus_noise_inv on GPU for fast preconditioner application
    # This eliminates the per-iteration copy in apply_pivoted_cholesky_precond_gpu
    var LTL_plus_noise_inv_device = ctx.enqueue_create_buffer[float_dtype](actual_rank * actual_rank)
    ctx.enqueue_copy(dst_buf=LTL_plus_noise_inv_device, src_buf=LTL_plus_noise_inv)
    ctx.synchronize()
    
    var t_end = perf_counter_ns()
    
    # Print timing (disabled for production)
    # print("    [PRECOND] GPU Precond construction (n=", n, ", rank=", actual_rank, "):")
    # print("      Init:      ", Float64((t_init - t_start)) / 1e6, "ms")
    # print("      Construct: ", Float64((t_construct - t_init)) / 1e6, "ms")
    # print("      LTL_inv:   ", Float64((t_end - t_construct)) / 1e6, "ms")
    # print("      Total:     ", Float64((t_end - t_start)) / 1e6, "ms")
    
    # Allocate work buffers for preconditioner application
    # This eliminates per-iteration allocations in apply_pivoted_cholesky_precond_gpu
    var w_work = ctx.enqueue_create_buffer[float_dtype](actual_rank * max_num_cols)
    var u_work = ctx.enqueue_create_buffer[float_dtype](actual_rank * max_num_cols)
    var z_work = ctx.enqueue_create_buffer[float_dtype](n * max_num_cols)
    
    # Compute log|P| once at construction and cache it
    var precond_temp = PivotedCholeskyPrecond(L_device^, noise, actual_rank, n, LTL_plus_noise_inv^, LTL_plus_noise_inv_device^,
                                  max_num_cols, w_work^, u_work^, z_work^, Float32(0.0))
    var log_det_val = compute_precond_log_det(ctx, precond_temp)
    precond_temp.cached_log_det = log_det_val
    return precond_temp^


fn _compute_LTL_plus_noise_inv(
    ctx: DeviceContext,
    L_device: DeviceBuffer[float_dtype],
    n: Int,
    rank: Int,
    noise: Float32,
) raises -> HostBuffer[float_dtype]:
    """Compute (L^T @ L + noise * I)^{-1} using pure Mojo Cholesky decomposition.
    
    For small matrices (rank ≤ 20), this is faster than calling PyTorch because
    it avoids the Python/PyTorch import overhead on first call.
    
    When noise is zero or very small (e.g., in the Kronecker multi-output path
    where noise is handled per-task separately), a minimum jitter of 1e-6 is
    added to ensure numerical stability. This matches GPyTorch's cholesky_jitter
    default for float32. Without this, rank-deficient kernels (Linear, Periodic)
    produce singular L^T @ L matrices that cannot be inverted.
    
    Uses Cholesky decomposition since L^T @ L + noise * I is positive definite:
    1. Compute M = L^T @ L + max(noise, 1e-6) * I
    2. Compute Cholesky: M = C @ C^T
    3. Compute C^{-1} by forward substitution
    4. Compute M^{-1} = C^{-T} @ C^{-1}
    
    Args:
        ctx: GPU device context
        L_device: Pivoted Cholesky factor L [n × rank] on device (column-major)
        n: Original matrix size
        rank: Preconditioner rank
        noise: Noise variance
        
    Returns:
        (L^T @ L + noise * I)^{-1} as HostBuffer [rank × rank] (row-major)
    """
    # Allocate output buffer
    var LTL_inv = ctx.enqueue_create_host_buffer[float_dtype](rank * rank)
    
    # Step 1: Compute M = L^T @ L + effective_noise * I [rank × rank] on GPU
    # Add minimum jitter (1e-6) when noise is zero/tiny to prevent singularity.
    alias MIN_JITTER: Float32 = 1e-6
    var effective_noise = noise if noise > MIN_JITTER else MIN_JITTER
    
    # GPU SYRK: M[i,j] = (i==j) + inv_noise * sum_k L[k,i]*L[k,j]
    # Note: kernel_ltl_syrk computes I + (1/noise)*L^T@L, so we use inv_noise=1/effective_noise
    # Then M = effective_noise * (I + (1/eff_noise)*L^T@L) = L^T@L + eff_noise*I
    # But kernel_ltl_syrk already adds identity and scales by inv_noise, producing I+(1/n)*LTL
    # We need L^T@L + eff_noise*I directly. So use inv_noise=1 to get L^T@L, then add eff_noise to diag
    var M_device = ctx.enqueue_create_buffer[DType.float32](rank * rank)
    var num_warps_ltl = (256 + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)
    # Use inv_noise=1.0 to get just L^T@L (kernel adds identity scaled by inv_noise)
    # Actually kernel does: sum * inv_noise + (i==j). We want sum + eff_noise*(i==j).
    # So set inv_noise=1.0 to get L^T@L + I, then need to adjust diagonal from +1 to +eff_noise.
    # Simpler: use inv_noise = 1/eff_noise to get (1/eff_noise)*L^T@L + I, then multiply by eff_noise:
    # eff_noise * ((1/eff_noise)*L^T@L + I) = L^T@L + eff_noise*I. But that requires scaling output.
    # Simplest: just use the kernel as-is with inv_noise=1.0 which gives L^T@L + I (identity, not noise*I)
    # Then fix up the diagonal on host: M[i,i] += (eff_noise - 1.0) for each i
    ctx.enqueue_function[kernel_ltl_syrk](
        M_device.unsafe_ptr(), L_device.unsafe_ptr(),
        n, rank, Float32(1.0),  # inv_noise=1.0 → result = L^T@L + I
        grid_dim=rank * rank, block_dim=256,
        shared_mem_bytes=num_warps_ltl * 4,
    )
    var M_host = ctx.enqueue_create_host_buffer[DType.float32](rank * rank)
    ctx.enqueue_copy(dst_buf=M_host, src_buf=M_device)
    ctx.synchronize()
    
    # Fixup diagonal: kernel gave L^T@L + I, we want L^T@L + eff_noise*I
    var M = List[Float32](capacity=rank * rank)
    for idx in range(rank * rank):
        M.append(M_host.unsafe_ptr()[idx])
    for i in range(rank):
        M[i * rank + i] += effective_noise - Float32(1.0)  # +noise - 1 (kernel already added +1)
    
    # Step 2: Cholesky decomposition M = C @ C^T
    # C is lower triangular, stored row-major
    var C = List[Float32](capacity=rank * rank)
    for _ in range(rank * rank):
        C.append(Float32(0.0))
    
    for i in range(rank):
        for j in range(i + 1):
            var sum_val = M[i * rank + j]
            for k in range(j):
                sum_val -= C[i * rank + k] * C[j * rank + k]
            if i == j:
                if sum_val <= Float32(0.0):
                    # Matrix not positive definite, use native LU decomposition
                    # Copy M to a contiguous buffer for matrix_inv_native
                    var M_ptr = alloc[Float32](rank * rank)
                    for idx in range(rank * rank):
                        M_ptr[idx] = M[idx]
                    matrix_inv_native(M_ptr, rank, LTL_inv.unsafe_ptr())
                    M_ptr.free()
                    return LTL_inv^
                C[i * rank + j] = sqrt(sum_val)
            else:
                C[i * rank + j] = sum_val / C[j * rank + j]
    
    # Step 3: Compute C^{-1} by forward substitution
    # C^{-1} is also lower triangular
    var C_inv = List[Float32](capacity=rank * rank)
    for _ in range(rank * rank):
        C_inv.append(Float32(0.0))
    
    for i in range(rank):
        C_inv[i * rank + i] = Float32(1.0) / C[i * rank + i]
        for j in range(i):
            var sum_val = Float32(0.0)
            for k in range(j, i):
                sum_val += C[i * rank + k] * C_inv[k * rank + j]
            C_inv[i * rank + j] = -sum_val / C[i * rank + i]
    
    # Step 4: Compute M^{-1} = C^{-T} @ C^{-1}
    # M^{-1}[i,j] = sum_k C^{-1}[k,i] * C^{-1}[k,j]
    for i in range(rank):
        for j in range(rank):
            var sum_val = Float32(0.0)
            # C^{-1} is lower triangular, so C^{-1}[k,i] is non-zero only for k >= i
            # and C^{-1}[k,j] is non-zero only for k >= j
            var k_start = i if i > j else j
            for k in range(k_start, rank):
                sum_val += C_inv[k * rank + i] * C_inv[k * rank + j]
            LTL_inv[i * rank + j] = sum_val
    
    return LTL_inv^


fn _compute_LTL_plus_noise_inv_vector(
    ctx: DeviceContext,
    L_device: DeviceBuffer[float_dtype],
    noise_vec_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    rank: Int,
) raises -> HostBuffer[float_dtype]:
    """Compute (I + L^T D^-1 L)^-1 for vector diagonal D."""
    var M_device = ctx.enqueue_create_buffer[float_dtype](rank * rank)
    var threads = 256
    var num_warps = (threads + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)
    ctx.enqueue_function[kernel_ltl_syrk_vector_noise](
        M_device.unsafe_ptr(), L_device.unsafe_ptr(), noise_vec_ptr,
        n, rank,
        grid_dim=rank * rank, block_dim=threads,
        shared_mem_bytes=num_warps * 4,
    )
    var M_host = ctx.enqueue_create_host_buffer[float_dtype](rank * rank)
    ctx.enqueue_copy(dst_buf=M_host, src_buf=M_device)
    ctx.synchronize()

    var LTL_inv = ctx.enqueue_create_host_buffer[float_dtype](rank * rank)
    var M = List[Float32](capacity=rank * rank)
    for i in range(rank * rank):
        M.append(M_host.unsafe_ptr()[i])

    var C = List[Float32](capacity=rank * rank)
    for _ in range(rank * rank):
        C.append(Float32(0.0))

    for i in range(rank):
        for j in range(i + 1):
            var sum_val = M[i * rank + j]
            for k in range(j):
                sum_val -= C[i * rank + k] * C[j * rank + k]
            if i == j:
                if sum_val <= Float32(0.0):
                    var M_ptr = alloc[Float32](rank * rank)
                    for idx in range(rank * rank):
                        M_ptr[idx] = M[idx]
                    matrix_inv_native(M_ptr, rank, LTL_inv.unsafe_ptr())
                    M_ptr.free()
                    return LTL_inv^
                C[i * rank + j] = sqrt(sum_val)
            else:
                C[i * rank + j] = sum_val / C[j * rank + j]

    var C_inv = List[Float32](capacity=rank * rank)
    for _ in range(rank * rank):
        C_inv.append(Float32(0.0))
    for i in range(rank):
        C_inv[i * rank + i] = Float32(1.0) / C[i * rank + i]
        for j in range(i):
            var sum_val = Float32(0.0)
            for k in range(j, i):
                sum_val += C[i * rank + k] * C_inv[k * rank + j]
            C_inv[i * rank + j] = -sum_val / C[i * rank + i]

    for i in range(rank):
        for j in range(rank):
            var sum_val = Float32(0.0)
            var k_start = i if i > j else j
            for k in range(k_start, rank):
                sum_val += C_inv[k * rank + i] * C_inv[k * rank + j]
            LTL_inv[i * rank + j] = sum_val

    return LTL_inv^


# =============================================================================
# GPU kernel for L^T @ L (replaces CPU triple-nested loop)
# =============================================================================

fn kernel_ltl_syrk(
    M_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    rank: Int,
    inv_noise: Float32,
) -> None:
    """Compute M[i,j] = (i==j) + inv_noise * sum_k L[k,i]*L[k,j] on GPU.

    Each BLOCK computes one element M[i,j] using parallel reduction over n.
    L is column-major: L[k,i] = L_ptr[i * n + k].
    Output M is rank×rank, row-major.
    Launch with grid_dim=rank*rank, block_dim=256.
    Requires shared_mem_bytes = (block_dim / WARP_SIZE) * 4.
    """
    var element_idx = Int(block_idx.x)
    if element_idx >= rank * rank:
        return
    var i = element_idx // rank
    var j = element_idx % rank
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)

    # Each thread sums a strided portion of the n dimension
    var partial_sum = Float32(0.0)
    var base_i = i * n
    var base_j = j * n
    var k = tid
    while k + bs * 3 < n:
        partial_sum += L_ptr[base_i + k] * L_ptr[base_j + k]
        partial_sum += L_ptr[base_i + k + bs] * L_ptr[base_j + k + bs]
        partial_sum += L_ptr[base_i + k + bs * 2] * L_ptr[base_j + k + bs * 2]
        partial_sum += L_ptr[base_i + k + bs * 3] * L_ptr[base_j + k + bs * 3]
        k += bs * 4
    while k < n:
        partial_sum += L_ptr[base_i + k] * L_ptr[base_j + k]
        k += bs

    # Warp-level reduction
    partial_sum = warp_sum(partial_sum)

    # Inter-warp reduction via shared memory
    var smem = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=16
    ]()
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE
    var num_warps = (bs + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)

    if lane_id == 0:
        smem[warp_id] = partial_sum
    barrier()

    # First warp reduces across warps
    if warp_id == 0:
        var val = smem[lane_id] if lane_id < num_warps else Float32(0.0)
        val = warp_sum(val)
        if lane_id == 0:
            var result = val * inv_noise
            if i == j:
                result += Float32(1.0)
            M_ptr[element_idx] = result


fn kernel_ltl_syrk_vector_noise(
    M_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    rank: Int,
) -> None:
    """Compute M = I + L^T D^-1 L for vector diagonal D."""
    var element_idx = Int(block_idx.x)
    if element_idx >= rank * rank:
        return
    var i = element_idx // rank
    var j = element_idx % rank
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var partial_sum = Float32(0.0)
    var base_i = i * n
    var base_j = j * n
    var k = tid
    while k + bs * 3 < n:
        partial_sum += L_ptr[base_i + k] * L_ptr[base_j + k] / noise_ptr[k]
        partial_sum += L_ptr[base_i + k + bs] * L_ptr[base_j + k + bs] / noise_ptr[k + bs]
        partial_sum += L_ptr[base_i + k + bs * 2] * L_ptr[base_j + k + bs * 2] / noise_ptr[k + bs * 2]
        partial_sum += L_ptr[base_i + k + bs * 3] * L_ptr[base_j + k + bs * 3] / noise_ptr[k + bs * 3]
        k += bs * 4
    while k < n:
        partial_sum += L_ptr[base_i + k] * L_ptr[base_j + k] / noise_ptr[k]
        k += bs
    partial_sum = warp_sum(partial_sum)
    var smem = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=16
    ]()
    var warp_id = tid // WARP_SIZE
    var lane_id = tid % WARP_SIZE
    var num_warps = (bs + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)
    if lane_id == 0:
        smem[warp_id] = partial_sum
    barrier()
    if warp_id == 0:
        var val = smem[lane_id] if lane_id < num_warps else Float32(0.0)
        val = warp_sum(val)
        if lane_id == 0:
            var result = val
            if i == j:
                result += Float32(1.0)
            M_ptr[element_idx] = result


fn kernel_add_sqrt_noise_vector(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_sample_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Compute out[row, col] += sqrt(noise[row]) * eps[row, col]."""
    var i = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    if i >= UInt(n) or col >= UInt(num_cols):
        return
    var idx = UInt(col) * UInt(n) + UInt(i)
    out_ptr[idx] += sqrt(noise_ptr[i]) * noise_sample_ptr[idx]


fn compute_precond_log_det(
    ctx: DeviceContext,
    precond: PivotedCholeskyPrecond,
) raises -> Float32:
    """Compute log|M| where M = L @ L^T + noise * I is the preconditioner.

    Uses GPU SYRK kernel for L^T @ L (O(n*rank²) on GPU instead of CPU),
    then CPU Cholesky on the small rank×rank result.

    Uses Sylvester's determinant theorem:
    |L @ L^T + noise * I| = noise^n * |I + (1/noise) * L @ L^T|
                          = noise^n * |I + (1/noise) * L^T @ L|  (Sylvester)

    So: log|M| = n * log(noise) + log|I_r + (1/noise) * L^T @ L|

    Args:
        ctx: GPU device context
        precond: Pivoted Cholesky preconditioner

    Returns:
        log|M| = log|L @ L^T + noise * I|
    """
    var n = precond.n
    var rank = precond.rank
    var noise = precond.noise
    var use_vector_noise = precond.noise_mode != 0

    # Step 1: Compute M = I_r + (1/noise) * L^T @ L on GPU [rank × rank]
    # L is already on GPU (precond.L), no D2H copy needed.
    var M_device = ctx.enqueue_create_buffer[DType.float32](rank * rank)

    with ProfileBlock[PROFILING]("PRECOND_logdet_gpu_syrk"):
        var threads = 256
        var num_warps = (threads + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)
        if use_vector_noise:
            ctx.enqueue_function[kernel_ltl_syrk_vector_noise](
                M_device.unsafe_ptr(), precond.L.unsafe_ptr(), precond.noise_vec_ptr,
                n, rank,
                grid_dim=rank * rank, block_dim=threads,
                shared_mem_bytes=num_warps * 4,
            )
        else:
            var inv_noise = Float32(1.0) / noise
            ctx.enqueue_function[kernel_ltl_syrk](
                M_device.unsafe_ptr(), precond.L.unsafe_ptr(),
                n, rank, inv_noise,
                grid_dim=rank * rank, block_dim=threads,
                shared_mem_bytes=num_warps * 4,
            )

    # Copy small rank×rank result to host (225 floats = 900 bytes)
    var M_host = ctx.enqueue_create_host_buffer[DType.float32](rank * rank)
    var noise_host = ctx.enqueue_create_host_buffer[DType.float32](n) if use_vector_noise else ctx.enqueue_create_host_buffer[DType.float32](1)
    with ProfileBlock[PROFILING]("PRECOND_logdet_d2h"):
        ctx.enqueue_copy(dst_buf=M_host, src_buf=M_device)
        if use_vector_noise:
            var noise_buf = DeviceBuffer[DType.float32](ctx, precond.noise_vec_ptr, n, owning=False)
            ctx.enqueue_copy(dst_buf=noise_host, src_buf=noise_buf)
        ctx.synchronize()

    # Step 2: Cholesky decomposition M = C @ C^T on CPU (rank×rank is tiny)
    var log_det_P: Float32
    with ProfileBlock[PROFILING]("PRECOND_logdet_cpu_chol"):
        var C = List[Float32](capacity=rank * rank)
        for _ in range(rank * rank):
            C.append(Float32(0.0))

        for i in range(rank):
            for j in range(i + 1):
                var sum_val = M_host.unsafe_ptr()[i * rank + j]
                for k in range(j):
                    sum_val -= C[i * rank + k] * C[j * rank + k]
                if i == j:
                    if sum_val <= Float32(0.0):
                        var M_ptr = alloc[Float32](rank * rank)
                        for idx in range(rank * rank):
                            M_ptr[idx] = M_host.unsafe_ptr()[idx]
                        var log_det_M = compute_slogdet_native(M_ptr, rank)
                        M_ptr.free()
                        if use_vector_noise:
                            var log_det_D = Float32(0.0)
                            for row in range(n):
                                log_det_D += log(noise_host[row])
                            log_det_P = log_det_D + log_det_M
                        else:
                            log_det_P = Float32(n) * log(noise) + log_det_M
                        return log_det_P
                    C[i * rank + j] = sqrt(sum_val)
                else:
                    C[i * rank + j] = sum_val / C[j * rank + j]

        # Step 3: log|M| = 2 * sum(log(diag(C)))
        var log_det_M = Float32(0.0)
        for i in range(rank):
            log_det_M += log(C[i * rank + i])
        log_det_M *= Float32(2.0)

        # Step 4: Sylvester's theorem
        if use_vector_noise:
            var log_det_D = Float32(0.0)
            for row in range(n):
                log_det_D += log(noise_host[row])
            log_det_P = log_det_D + log_det_M
        else:
            log_det_P = Float32(n) * log(noise) + log_det_M

    return log_det_P


fn apply_pivoted_cholesky_precond(
    ctx: DeviceContext,
    precond: PivotedCholeskyPrecond,
    v_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    out_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) raises:
    """Apply Pivoted Cholesky preconditioner: out = P^{-1} @ v.
    
    Uses Woodbury identity:
    P^{-1} = (L @ L^T + noise * I)^{-1}
           = (1/noise) * (I - L @ (L^T @ L + noise * I)^{-1} @ L^T)
    
    Algorithm:
    1. Compute w = L^T @ v [rank × num_cols]
    2. Compute u = (L^T @ L + noise * I)^{-1} @ w [rank × num_cols]
    3. Compute z = L @ u [n × num_cols]
    4. out = (1/noise) * (v - z)
    
    Args:
        ctx: GPU device context
        precond: Pivoted Cholesky preconditioner
        v_device_ptr: Input vectors [n × num_cols] column-major
        out_device_ptr: Output vectors [n × num_cols] column-major
        n: Number of rows
        num_cols: Number of columns
    """
    var t_start = perf_counter_ns()
    
    var rank = precond.rank
    var noise = precond.noise
    var noise_inv = Float32(1.0) / noise
    
    # Allocate temporary buffers
    var w_device = ctx.enqueue_create_buffer[float_dtype](rank * num_cols)  # L^T @ v
    var u_device = ctx.enqueue_create_buffer[float_dtype](rank * num_cols)  # (L^T L + noise I)^{-1} @ w
    var z_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)     # L @ u
    
    var w_host = ctx.enqueue_create_host_buffer[float_dtype](rank * num_cols)
    var u_host = ctx.enqueue_create_host_buffer[float_dtype](rank * num_cols)
    
    # Copy L to host for matrix operations
    var L_host = ctx.enqueue_create_host_buffer[float_dtype](n * rank)
    ctx.enqueue_copy(dst_buf=L_host, src_buf=precond.L)
    
    # Copy v to host
    var v_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_cols)
    var v_device_buf = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    ctx.enqueue_function[kernel_copy](
        v_device_buf.unsafe_ptr(), v_device_ptr, n * num_cols,
        grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
    )
    ctx.enqueue_copy(dst_buf=v_host, src_buf=v_device_buf)
    ctx.synchronize()
    
    # Step 1: w = L^T @ v [rank × num_cols]
    var t1 = perf_counter_ns()
    for col in range(num_cols):
        for i in range(rank):
            var sum_val = Float32(0.0)
            for k in range(n):
                # L is column-major: L[k, i] = L_host[i * n + k]
                # v is column-major: v[k, col] = v_host[col * n + k]
                sum_val += L_host[i * n + k] * v_host[col * n + k]
            # w is column-major: w[i, col] = w_host[col * rank + i]
            w_host[col * rank + i] = sum_val
    var t1_end = perf_counter_ns()
    
    # Step 2: u = (L^T @ L + noise * I)^{-1} @ w [rank × num_cols]
    # LTL_plus_noise_inv is row-major [rank × rank]
    var t2 = perf_counter_ns()
    for col in range(num_cols):
        for i in range(rank):
            var sum_val = Float32(0.0)
            for j in range(rank):
                # LTL_inv is row-major: LTL_inv[i, j] = LTL_inv[i * rank + j]
                # w is column-major: w[j, col] = w_host[col * rank + j]
                sum_val += precond.LTL_plus_noise_inv[i * rank + j] * w_host[col * rank + j]
            # u is column-major: u[i, col] = u_host[col * rank + i]
            u_host[col * rank + i] = sum_val
    var t2_end = perf_counter_ns()
    
    # Step 3: z = L @ u [n × num_cols]
    var t3 = perf_counter_ns()
    var z_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_cols)
    for col in range(num_cols):
        for i in range(n):
            var sum_val = Float32(0.0)
            for j in range(rank):
                # L is column-major: L[i, j] = L_host[j * n + i]
                # u is column-major: u[j, col] = u_host[col * rank + j]
                sum_val += L_host[j * n + i] * u_host[col * rank + j]
            # z is column-major: z[i, col] = z_host[col * n + i]
            z_host[col * n + i] = sum_val
    var t3_end = perf_counter_ns()
    
    # Step 4: out = (1/noise) * (v - z)
    var t4 = perf_counter_ns()
    var out_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_cols)
    for i in range(n * num_cols):
        out_host[i] = noise_inv * (v_host[i] - z_host[i])
    
    # Copy result to device
    var out_device_buf = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    ctx.enqueue_copy(dst_buf=out_device_buf, src_buf=out_host)
    ctx.synchronize()
    
    # Copy to output pointer
    ctx.enqueue_function[kernel_copy](
        out_device_ptr, out_device_buf.unsafe_ptr(), n * num_cols,
        grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
    )
    ctx.synchronize()
    var t4_end = perf_counter_ns()
    
    # PROFILING: Print timing breakdown (disabled for benchmarks)
    # print("Precond apply breakdown (n=", n, ", rank=", rank, ", num_cols=", num_cols, "):")
    # print("  L^T @ v:     ", Float64((t1_end - t1)) / 1e6, "ms")
    # print("  LTL_inv @ w: ", Float64((t2_end - t2)) / 1e6, "ms")
    # print("  L @ u:       ", Float64((t3_end - t3)) / 1e6, "ms")
    # print("  Scale/sub:   ", Float64((t4_end - t4)) / 1e6, "ms")
    # print("  Total:       ", Float64((t4_end - t_start)) / 1e6, "ms")
    _ = t4_end  # Suppress unused variable warning


fn apply_pivoted_cholesky_precond_gpu(
    ctx: DeviceContext,
    precond: PivotedCholeskyPrecond,
    v_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    out_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    sync: Bool = True,
) raises:
    """Apply Pivoted Cholesky preconditioner entirely on GPU using MAX matmul.
    
    Uses Woodbury identity:
    P^{-1} = (1/noise) * (I - L @ (L^T @ L + noise * I)^{-1} @ L^T)
    
    Algorithm:
    1. w = L^T @ v [rank × num_cols] using MAX matmul
    2. u = LTL_inv @ w [rank × num_cols] using custom kernel (small matrix)
    3. z = L @ u [n × num_cols] using MAX matmul
    4. out = (1/noise) * (v - z) using custom kernel
    
    All operations on GPU, no CPU-GPU copies during application.
    Uses pre-allocated work buffers from preconditioner to avoid per-call allocations.
    
    Args:
        ctx: GPU device context
        precond: Pivoted Cholesky preconditioner (with pre-allocated work buffers)
        v_device_ptr: Input vectors [n × num_cols] column-major ON DEVICE
        out_device_ptr: Output vectors [n × num_cols] column-major ON DEVICE
        n: Number of rows
        num_cols: Number of columns
        sync: Whether to synchronize after kernel launches (default True).
              Set to False when called inside CG loop to avoid per-iteration syncs.
    """
    var rank = precond.rank
    var noise_inv = Float32(1.0) / precond.noise
    
    # Runtime check: work buffers must be large enough for num_cols
    if num_cols > precond.max_num_cols:
        raise Error(
            "Preconditioner work buffers too small: num_cols=" + String(num_cols) +
            " > max_num_cols=" + String(precond.max_num_cols) +
            ". Rebuild preconditioner with larger max_num_cols."
        )
    
    with ProfileBlock[False]("PRECOND_apply_total"):  # Disabled: called 200+ times per iter
        # Use pre-allocated work buffers from preconditioner (no allocation needed!)
        # This eliminates per-iteration buffer allocations in the CG loop
        # Note: num_cols must be <= precond.max_num_cols
        var w_ptr = precond.w_work.unsafe_ptr()
        var u_ptr = precond.u_work.unsafe_ptr()
        var z_ptr = precond.z_work.unsafe_ptr()
        alias BLOCK_SIZE = 16
        
        # Use cached LTL_plus_noise_inv on GPU (no copy needed!)
        # This eliminates the per-iteration sync that was the main bottleneck
        
        # Step 1: w = L^T @ v for scalar noise, or L^T @ D^-1 @ v for vector noise.
        var input_for_ltv_ptr = v_device_ptr
        if precond.noise_mode != 0:
            var grid_scale = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,
                              (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
            ctx.enqueue_function[kernel_row_scale_by_inv_noise](
                z_ptr, v_device_ptr, precond.noise_vec_ptr, n, num_cols,
                grid_dim=grid_scale, block_dim=(BLOCK_SIZE, BLOCK_SIZE)
            )
            input_for_ltv_ptr = z_ptr

        # L is column-major [n, rank], v is column-major [n, num_cols]
        # Reinterpret as row-major: v becomes [num_cols, n], L becomes [rank, n]
        # We want w = L^T @ v, which in row-major is: w^T = v^T @ L
        # So: w[num_cols, rank] = v[num_cols, n] @ L[rank, n]^T
        var v_ndbuf = NDBuffer[DType.float32, 2](input_for_ltv_ptr, (num_cols, n))
        var L_ndbuf = NDBuffer[DType.float32, 2](precond.L.unsafe_ptr(), (rank, n))
        var w_ndbuf = NDBuffer[DType.float32, 2](w_ptr, (num_cols, rank))
        
        var opt_ctx = Optional[DeviceContext](ctx)
        # CORRECT: Use transpose_b=True (verified by isolated test)
        max_matmul[transpose_b=True, target="gpu"](w_ndbuf, v_ndbuf, L_ndbuf, opt_ctx)
        
        # Step 2: u = LTL_inv @ w [rank × num_cols] using custom kernel
        # Use cached LTL_plus_noise_inv_device (no copy needed!)
        var grid_2 = ((rank + BLOCK_SIZE - 1) // BLOCK_SIZE,
                      (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
        ctx.enqueue_function[kernel_small_matmul](
            u_ptr,
            precond.LTL_plus_noise_inv_device.unsafe_ptr(),
            w_ptr,
            rank, num_cols,
            grid_dim=grid_2, block_dim=(BLOCK_SIZE, BLOCK_SIZE)
        )
        
        # Step 3: z = L @ u [n × num_cols] using MAX matmul
        # L is column-major [n, rank], u is column-major [rank, num_cols]
        # Reinterpret as row-major: u becomes [num_cols, rank], L becomes [rank, n]
        # We want z = L @ u, which in row-major is: z^T = u^T @ L^T
        # So: z[num_cols, n] = u[num_cols, rank] @ L[rank, n]^T
        var u_ndbuf = NDBuffer[DType.float32, 2](u_ptr, (num_cols, rank))
        var z_ndbuf = NDBuffer[DType.float32, 2](z_ptr, (num_cols, n))
        # CORRECT: NO transpose (verified by isolated test)
        max_matmul[target="gpu"](z_ndbuf, u_ndbuf, L_ndbuf, opt_ctx)
        
        # Step 4: out = (1/noise) * (v - z)
        var grid_4 = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,
                      (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if precond.noise_mode != 0:
            ctx.enqueue_function[kernel_woodbury_final_vector](
                out_device_ptr, v_device_ptr, z_ptr, precond.noise_vec_ptr,
                n, num_cols,
                grid_dim=grid_4, block_dim=(BLOCK_SIZE, BLOCK_SIZE)
            )
        else:
            ctx.enqueue_function[kernel_woodbury_final](
                out_device_ptr, v_device_ptr, z_ptr,
                noise_inv, n, num_cols,
                grid_dim=grid_4, block_dim=(BLOCK_SIZE, BLOCK_SIZE)
            )
        
        # Only sync if requested (default True for backward compatibility)
        # Set sync=False when called inside CG loop to avoid per-iteration syncs
        if sync:
            ctx.synchronize()


# =============================================================================
# GPU Probe Vector Generation for Preconditioned SLQ
# =============================================================================

fn kernel_generate_rademacher(
    output: UnsafePointer[Float32, MutAnyOrigin],
    seed: UInt64,
    size: Int,
) -> None:
    """Generate Rademacher random variables (±1) on GPU.
    
    Uses xorshift64 for better randomness than LCG.
    Each thread generates one random value based on its global index.
    
    Args:
        output: Output buffer [size]
        seed: Random seed
        size: Number of values to generate
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(size):
        return
    
    # Initialize state with seed and index (use golden ratio hash for mixing)
    var state = seed ^ (UInt64(idx) * UInt64(0x9E3779B97F4A7C15))
    
    # Xorshift64 (better randomness than LCG)
    state ^= state >> 12
    state ^= state << 25
    state ^= state >> 27
    state *= UInt64(0x2545F4914F6CDD1D)
    
    # Use bit 17 (middle bit) instead of bit 0 for better randomness
    output[idx] = Float32(1.0) if (state & UInt64(0x20000)) != UInt64(0) else Float32(-1.0)


fn kernel_axpy_scalar(
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha: Float32,
    size: Int,
) -> None:
    """Compute y += alpha * x (scalar alpha version)."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(size):
        return
    y_ptr[idx] += alpha * x_ptr[idx]


fn kernel_normalize_columns(
    data_ptr: UnsafePointer[Float32, MutAnyOrigin],
    norms_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Normalize each column of a matrix by its L2 norm.
    
    Grid: (num_cols,)
    Block: (256,) - multi-warp with shared memory reduction
    
    Also stores the norms for later use (needed for gradient computation).
    Requires shared_mem_bytes = (256/WARP_SIZE) * 4 at launch.
    """
    # Each block handles one column
    var col = block_idx.x
    if col >= UInt(num_cols):
        return
    
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    
    # Each thread computes a partial norm² over strided elements.
    var partial_sum = Float32(0.0)
    var i = tid
    while i < n:
        var val = data_ptr[UInt(col) * UInt(n) + UInt(i)]
        partial_sum += val * val
        i += bs
    
    # Reduce within each warp.
    partial_sum = warp_sum(partial_sum)
    
    # Reduce across warps through shared memory.
    var smem = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=16,
    ]()
    var warp_id = tid // Int(WARP_SIZE)
    var lane_id = tid % Int(WARP_SIZE)
    var num_warps = (bs + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)
    
    if lane_id == 0:
        smem[warp_id] = partial_sum
    barrier()
    
    var norm = Float32(0.0)
    if warp_id == 0:
        var val = smem[lane_id] if lane_id < num_warps else Float32(0.0)
        val = warp_sum(val)
        if lane_id == 0:
            norm = sqrt(val)
            norms_ptr[col] = norm
            smem[0] = norm  # broadcast to all threads
    barrier()
    norm = smem[0]
    
    # Normalize using all threads with strided access.
    i = tid
    while i < n:
        data_ptr[UInt(col) * UInt(n) + UInt(i)] /= norm
        i += bs


fn normalize_columns_gpu(
    ctx: DeviceContext,
    data: DeviceBuffer[float_dtype],
    norms_out: DeviceBuffer[float_dtype],
    n: Int,
    num_cols: Int,
    sync: Bool = True,
) raises:
    """Normalize columns of a matrix and store their norms.
    
    Args:
        ctx: Device context
        data: Column-major matrix [n × num_cols] to normalize in-place
        norms_out: Output buffer for column norms [num_cols]
        n: Number of rows
        num_cols: Number of columns
        sync: Whether to synchronize after the kernel (default True)
    """
    var num_warps_nc = (256 + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)
    ctx.enqueue_function[kernel_normalize_columns](
        data.unsafe_ptr(), norms_out.unsafe_ptr(), n, num_cols,
        grid_dim=(num_cols,), block_dim=(256,),
        shared_mem_bytes=num_warps_nc * 4,
    )
    if sync:
        ctx.synchronize()


fn sample_from_preconditioner_gpu(
    ctx: DeviceContext,
    precond: PivotedCholeskyPrecond,
    num_samples: Int,
    seed_val: UInt64 = 42,
) raises -> DeviceBuffer[float_dtype]:
    """Sample probe vectors from N(0, P) where P = L@L^T + noise*I.
    
    GPyTorch-aligned implementation (matches zero_mean_mvn_samples):
    1. Generate base vectors: z ~ N(0, I) using Mojo's built-in randn
    2. Compute samples = L @ z (transform to N(0, LL^T))
    3. Add noise component: samples += sqrt(noise) * z_noise
    
    This matches GPyTorch's zero_mean_mvn_samples() approach which uses
    torch.randn() for Gaussian base samples, NOT Rademacher (±1).
    
    Note: We generate on host and copy to GPU. This adds ~0.1ms overhead
    which is negligible (<1%) compared to iteration time. For n > 50K,
    consider implementing a GPU Box-Muller kernel.
    
    Args:
        ctx: Device context
        precond: Pivoted Cholesky preconditioner with L [n × rank]
        num_samples: Number of probe vectors to generate
        seed_val: Random seed for reproducibility
        
    Returns:
        Probe vectors [n × num_samples] on device (column-major)
    """
    var n = precond.n
    var rank = precond.rank
    
    # Seed the RNG for reproducibility
    random_seed(Int(seed_val))
    
    # Step 1: Generate rank-dimensional GAUSSIAN vectors on GPU directly
    var z_rank_device: DeviceBuffer[float_dtype]
    with ProfileBlock[PROFILING]("PRECOND_sample_gpu_randn"):
        z_rank_device = ctx.enqueue_create_buffer[float_dtype](rank * num_samples)
        var rk_grid = (rank * num_samples + 255) // 256
        ctx.enqueue_function[kernel_gpu_gaussian](
            z_rank_device.unsafe_ptr(), rank * num_samples, seed_val,
            grid_dim=(rk_grid,), block_dim=(256,))
    
    # Step 2: Generate n-dimensional GAUSSIAN noise vectors on GPU directly
    var z_noise_device: DeviceBuffer[float_dtype]
    with ProfileBlock[PROFILING]("PRECOND_sample_gpu_randn"):
        z_noise_device = ctx.enqueue_create_buffer[float_dtype](n * num_samples)
        var ns_grid = (n * num_samples + 255) // 256
        ctx.enqueue_function[kernel_gpu_gaussian](
            z_noise_device.unsafe_ptr(), n * num_samples, seed_val + UInt64(1000000),
            grid_dim=(ns_grid,), block_dim=(256,))
    
    # Step 3: Compute z = L @ z_rank using GPU matmul
    # L: [n × rank] column-major, z_rank: [rank × num_samples] → z: [n × num_samples]
    var z_device = ctx.enqueue_create_buffer[float_dtype](n * num_samples)
    
    # Use MAX matmul with correct signature (from gemm_matvec.mojo:182-190 pattern)
    # We want: z = L @ z_rank where all are column-major
    # L: [n, rank] column-major
    # z_rank: [rank, num_samples] column-major
    # z: [n, num_samples] column-major
    #
    # Reinterpret column-major as row-major (swap dimensions):
    # L [n, rank] col-major → [rank, n] row-major
    # z_rank [rank, num_samples] col-major → [num_samples, rank] row-major
    # z [n, num_samples] col-major → [num_samples, n] row-major
    #
    # For row-major matmul: z^T = z_rank^T @ L^T
    # Column-major L [n, rank] reinterpreted as row-major is [rank, n]
    # Column-major z_rank [rank, num_samples] reinterpreted as row-major is [num_samples, rank]
    # Column-major z [n, num_samples] reinterpreted as row-major is [num_samples, n]
    # So: C[num_samples, n] = A[num_samples, rank] @ B[rank, n] (NO transpose)
    # This matches the verified-correct pattern in apply_pivoted_cholesky_precond_gpu
    
    with ProfileBlock[PROFILING]("PRECOND_sample_L_matmul"):
        var z_rank_ndbuf = NDBuffer[DType.float32, 2](z_rank_device.unsafe_ptr(), (num_samples, rank))
        var L_ndbuf = NDBuffer[DType.float32, 2](precond.L.unsafe_ptr(), (rank, n))
        var z_ndbuf = NDBuffer[DType.float32, 2](z_device.unsafe_ptr(), (num_samples, n))
        
        # FIXED: No transpose - matches apply_pivoted_cholesky_precond_gpu (line 1341)
        var opt_ctx = Optional[DeviceContext](ctx)
        max_matmul[target="gpu"](z_ndbuf, z_rank_ndbuf, L_ndbuf, opt_ctx)
        ctx.synchronize()
    
    # Step 4: Add noise component: z += sqrt(noise) * z_noise or sqrt(D_i) * z_noise.
    with ProfileBlock[PROFILING]("PRECOND_sample_noise_add"):
        if precond.noise_mode != 0:
            ctx.enqueue_function[kernel_add_sqrt_noise_vector](
                z_device.unsafe_ptr(), z_noise_device.unsafe_ptr(), precond.noise_vec_ptr,
                n, num_samples,
                grid_dim=((n + 255) // 256, num_samples), block_dim=(256, 1)
            )
        else:
            var noise_scale = sqrt(precond.noise)
            ctx.enqueue_function[kernel_axpy_scalar](
                z_device.unsafe_ptr(), z_noise_device.unsafe_ptr(),
                noise_scale, n * num_samples,
                grid_dim=((n * num_samples + 255) // 256,), block_dim=(256,)
            )
        ctx.synchronize()
    
    return z_device^


fn sample_from_preconditioner_gpu_pooled(
    ctx: DeviceContext,
    precond: PivotedCholeskyPrecond,
    num_samples: Int,
    seed_val: UInt64,
    z_rank_host: HostBuffer[float_dtype],
    z_rank_device: DeviceBuffer[float_dtype],
    z_noise_host: HostBuffer[float_dtype],
    z_noise_device: DeviceBuffer[float_dtype],
    out_device: DeviceBuffer[float_dtype],
) raises:
    """Sample probe vectors from N(0, P) using pre-allocated pool buffers.
    
    OPTIMIZATION #5b: This version uses pre-allocated buffers from CGBufferPool
    instead of allocating 5 fresh buffers per call. This eliminates buffer
    allocation overhead in the training hot-path.
    
    OPTIMIZATION: Reduced from 4 syncs to 1 sync by batching H→D copies.
    
    GPyTorch-aligned implementation (matches zero_mean_mvn_samples):
    1. Generate base vectors: z ~ N(0, I) using Mojo's built-in randn
    2. Compute samples = L @ z (transform to N(0, LL^T))
    3. Add noise component: samples += sqrt(noise) * z_noise
    
    Args:
        ctx: Device context
        precond: Pivoted Cholesky preconditioner with L [n × rank]
        num_samples: Number of probe vectors to generate
        seed_val: Random seed for reproducibility
        z_rank_host: Pre-allocated host buffer [rank × num_samples]
        z_rank_device: Pre-allocated device buffer [rank × num_samples]
        z_noise_host: Pre-allocated host buffer [n × num_samples]
        z_noise_device: Pre-allocated device buffer [n × num_samples]
        out_device: Pre-allocated output buffer [n × num_samples]
    """
    var n = precond.n
    var rank = precond.rank
    
    # Step 1: Generate Gaussian vectors directly on GPU (no CPU randn, no H2D copy)
    with ProfileBlock[PROFILING]("PRECOND_sample_gpu_randn"):
        var rank_blocks = (rank * num_samples + 255) // 256
        ctx.enqueue_function[kernel_gpu_gaussian](
            z_rank_device.unsafe_ptr(), rank * num_samples,
            UInt64(seed_val), UInt64(rank * num_samples + 1),
            grid_dim=rank_blocks, block_dim=256,
        )
    with ProfileBlock[PROFILING]("PRECOND_sample_gpu_randn"):
        var noise_blocks = (n * num_samples + 255) // 256
        ctx.enqueue_function[kernel_gpu_gaussian](
            z_noise_device.unsafe_ptr(), n * num_samples,
            UInt64(seed_val + 12345), UInt64(n * num_samples + 1),
            grid_dim=noise_blocks, block_dim=256,
        )
    
    # Step 3: Compute z = L @ z_rank using GPU matmul
    # Column-major L [n, rank] reinterpreted as row-major is [rank, n]
    # Column-major z_rank [rank, num_samples] reinterpreted as row-major is [num_samples, rank]
    # Column-major out [n, num_samples] reinterpreted as row-major is [num_samples, n]
    # So: C[num_samples, n] = A[num_samples, rank] @ B[rank, n] (NO transpose)
    var z_rank_ndbuf = NDBuffer[DType.float32, 2](z_rank_device.unsafe_ptr(), (num_samples, rank))
    var L_ndbuf = NDBuffer[DType.float32, 2](precond.L.unsafe_ptr(), (rank, n))
    var z_ndbuf = NDBuffer[DType.float32, 2](out_device.unsafe_ptr(), (num_samples, n))
    
    # FIXED: No transpose - matches apply_pivoted_cholesky_precond_gpu (line 1341)
    # No sync needed - axpy kernel will wait for matmul
    var opt_ctx = Optional[DeviceContext](ctx)
    max_matmul[target="gpu"](z_ndbuf, z_rank_ndbuf, L_ndbuf, opt_ctx)
    
    # Step 4: Add noise component: out += sqrt(noise) * z_noise or sqrt(D_i) * z_noise
    # No sync needed - subsequent kernels will wait
    if precond.noise_mode != 0:
        ctx.enqueue_function[kernel_add_sqrt_noise_vector](
            out_device.unsafe_ptr(), z_noise_device.unsafe_ptr(), precond.noise_vec_ptr,
            n, num_samples,
            grid_dim=((n + 255) // 256, num_samples), block_dim=(256, 1)
        )
    else:
        var noise_scale = sqrt(precond.noise)
        ctx.enqueue_function[kernel_axpy_scalar](
            out_device.unsafe_ptr(), z_noise_device.unsafe_ptr(),
            noise_scale, n * num_samples,
            grid_dim=((n * num_samples + 255) // 256,), block_dim=(256,)
        )
    ctx.synchronize()


# =============================================================================
# Unified Preconditioner Builder with GradientProvider Trait
# =============================================================================

fn build_pivoted_cholesky_precond_unified[P: ForwardProvider](
    provider: P,
    rank: Int = 10,
    error_tol: Float32 = 1e-3,
    max_num_cols: Int = 16,
    precond_method: Int = 2,
    adaptive_tol: Float32 = 0.01,
    seed: UInt64 = 42,
    noise_mode: Int = 0,
    noise_vec_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
) raises -> PivotedCholeskyPrecond:
    """Build low-rank Pivoted Cholesky preconditioner with GPU acceleration.
    
    This is the unified version that works with any GradientProvider.
    It replaces the 3 hot-path preconditioner builders:
    - build_pivoted_cholesky_precond_gpu[T: MatvecProvider]
    - build_pivoted_cholesky_precond_gpu_composite[DIM, K]
    - build_pivoted_cholesky_precond_gpu_materialized_composite[DIM, K]
    
    Key differences from build_pivoted_cholesky_precond_gpu:
    - Uses provider.extract_diagonal() instead of provider.get_outputscale()
    - Works with any GradientProvider (isotropic, ARD, or composite)
    
    This implementation minimizes CPU-GPU synchronization by:
    1. Keeping working data on GPU where possible
    2. Using GPU kernels for orthogonalization
    3. Only syncing when pivot info is needed on CPU
    4. Pre-allocating work buffers for preconditioner application
    
    Args:
        provider: Any GradientProvider (isotropic, ARD, or composite)
        rank: Maximum rank of the approximation (default: 10)
        error_tol: Early stopping tolerance for greedy/rpcholesky (default: 1e-3)
        max_num_cols: Maximum number of columns for work buffers (default: 16)
        precond_method: Pivot selection method (default: 2)
            0 = greedy (deterministic argmax, GPyTorch-compatible)
            1 = rpcholesky (randomized proportional sampling, fixed rank)
            2 = nystrom (rpcholesky + adaptive rank based on noise floor)
        adaptive_tol: Nystrom stopping threshold (default: 0.01).
            Stops when residual_trace < adaptive_tol * n * noise.
        seed: Random seed for rpcholesky/nystrom pivot sampling (default: 42)
        
    Returns:
        PivotedCholeskyPrecond ready for use in CG
    """
    var t_start = perf_counter_ns()
    
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var noise = provider.get_noise()
    var diagonal_value = provider.get_diagonal_value()  # For error tracking

    # Zero-rank is the explicit no-preconditioner / fairness lane.
    # The caller still passes a Preconditioner object into BBMM/CG, but those
    # paths branch on `use_preconditioner` and will never touch these buffers.
    if rank <= 0:
        var dummy_L = ctx.enqueue_create_buffer[float_dtype](1)
        var dummy_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        dummy_host[0] = Float32(0.0)
        var dummy_ltl_device = ctx.enqueue_create_buffer[float_dtype](1)
        var dummy_w = ctx.enqueue_create_buffer[float_dtype](1)
        var dummy_u = ctx.enqueue_create_buffer[float_dtype](1)
        var dummy_z = ctx.enqueue_create_buffer[float_dtype](1)
        return PivotedCholeskyPrecond(
            dummy_L^,
            noise,
            0,
            n,
            dummy_host^,
            dummy_ltl_device^,
            1,
            dummy_w^,
            dummy_u^,
            dummy_z^,
            Float32(0.0),
            noise_mode,
            noise_vec_ptr,
        )

    alias BLOCK_SIZE = 256
    var num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # =========================================================================
    # Allocate GPU buffers.
    # =========================================================================
    
    # L matrix [n × rank] column-major on GPU
    var L_device = ctx.enqueue_create_buffer[float_dtype](n * rank)
    
    # Initialize L to zero on GPU (no CPU loop, no H2D copy)
    from .cg_solver import kernel_fill_constant
    var fill_blocks2 = (n * rank + 255) // 256
    ctx.enqueue_function[kernel_fill_constant](
        L_device.unsafe_ptr(), n * rank, Float32(0.0),
        grid_dim=fill_blocks2, block_dim=256)
    
    # Diagonal values on GPU - use extract_diagonal instead of constant initialization
    var diag_device = ctx.enqueue_create_buffer[float_dtype](n)
    provider.extract_diagonal(diag_device.unsafe_ptr())
    
    # Mask for tracking which indices are still available (1.0 = available, 0.0 = used as pivot)
    var mask_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_function[kernel_init_diagonal](
        mask_device.unsafe_ptr(), n, Float32(1.0),
        grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
    )
    
    # GPU argmax result buffer: [pivot_idx_as_float, max_value, sum_abs_error]
    var argmax_result_device = ctx.enqueue_create_buffer[float_dtype](3)
    var argmax_result_host = ctx.enqueue_create_host_buffer[float_dtype](3)
    
    # Unit vector for matvec
    var e_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Result of K @ e
    var Ke_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Dot product result buffer
    var dot_result_device = ctx.enqueue_create_buffer[float_dtype](1)
    var dot_result_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    ctx.synchronize()  # SYNC 1: After initialization
    
    var t_init = perf_counter_ns()
    
    # =========================================================================
    # Construct the factor iteratively.
    # =========================================================================
    
    var actual_rank = rank
    var orig_error = diagonal_value  # Use diagonal_value for error tracking
    
    # For rpcholesky/nystrom: need host-side diagonal for proportional sampling
    var diag_host_sampling = ctx.enqueue_create_host_buffer[float_dtype](n)
    # Track which indices are used as pivots (host-side, for sampling)
    var used_as_pivot = List[Bool]()
    for _i in range(n):
        used_as_pivot.append(False)
    # Mutable seed state for rpcholesky/nystrom
    var rng_state = seed
    
    for m in range(rank):
        # Step 1+2: GPU-side masked argmax + error sum (eliminates n-element D2H copy)
        ctx.enqueue_function[kernel_masked_argmax_and_error](
            diag_device.unsafe_ptr(), mask_device.unsafe_ptr(),
            argmax_result_device.unsafe_ptr(), n,
            grid_dim=(1,), block_dim=(256,),
        )
        ctx.enqueue_copy(dst_buf=argmax_result_host, src_buf=argmax_result_device)
        ctx.synchronize()  # SYNC: Need argmax result for pivot selection
        
        var greedy_pivot_idx = Int(argmax_result_host[0])
        var max_diag_value = argmax_result_host[1]
        var sum_abs_error = argmax_result_host[2]
        
        # Check early termination (common to all methods)
        if max_diag_value < Float32(1e-10) or greedy_pivot_idx < 0:
            actual_rank = m
            if actual_rank == 0:
                actual_rank = 1
            break
        
        # Stopping criterion depends on method
        if precond_method == 2:
            # Nystrom: adaptive rank based on noise floor
            # sum_abs_error from the GPU kernel is sum of |diag[i]| for unmasked entries
            # For positive residual diag, this approximates the residual trace
            var residual_trace = sum_abs_error
            var threshold = adaptive_tol * Float32(n) * noise
            if residual_trace < threshold and m > 0:
                actual_rank = m
                if actual_rank == 0:
                    actual_rank = 1
                break
        else:
            # Greedy / RPCholesky: fixed relative trace threshold
            var current_error = sum_abs_error / orig_error
            if current_error < error_tol:
                actual_rank = m
                if actual_rank == 0:
                    actual_rank = 1
                break
        
        # Pivot selection depends on method
        var pivot_idx: Int
        if precond_method == 0:
            # Greedy: use GPU argmax result directly
            pivot_idx = greedy_pivot_idx
        else:
            # RPCholesky / Nystrom: proportional sampling on CPU
            # Copy diagonal to host for sampling
            ctx.enqueue_copy(dst_buf=diag_host_sampling, src_buf=diag_device)
            ctx.synchronize()
            
            # Sum residual diagonal for sampling probabilities
            var total = Float32(0.0)
            for i in range(n):
                if not used_as_pivot[i]:
                    var d = diag_host_sampling[i]
                    if d > Float32(0.0):
                        total += d
            
            if total < Float32(1e-10):
                # All residual gone, early terminate
                actual_rank = m
                if actual_rank == 0:
                    actual_rank = 1
                break
            
            # Generate random uniform using xorshift64
            rng_state ^= rng_state >> 12
            rng_state ^= rng_state << 25
            rng_state ^= rng_state >> 27
            rng_state *= UInt64(0x2545F4914F6CDD1D)
            # Convert to float in [0, 1)
            var u = Float32(Int(rng_state & UInt64(0x7FFFFFFF))) / Float32(2147483648.0)
            u = u * total
            
            # Sample pivot proportional to max(diag[i], 0)
            var cumsum = Float32(0.0)
            pivot_idx = greedy_pivot_idx  # fallback
            for i in range(n):
                if not used_as_pivot[i]:
                    var d = diag_host_sampling[i]
                    if d > Float32(0.0):
                        cumsum += d
                    if cumsum >= u:
                        pivot_idx = i
                        max_diag_value = diag_host_sampling[i]
                        break
        
        # Track used pivots (host-side)
        used_as_pivot[pivot_idx] = True
        
        # Mark pivot as used in GPU mask (set to 0.0)
        ctx.enqueue_function[kernel_set_value_at_index](
            mask_device.unsafe_ptr(), pivot_idx, Float32(0.0),
            grid_dim=(1,), block_dim=(1,)
        )
        
        # Step 3: Create unit vector and compute K @ e_pivot (GPU)
        ctx.enqueue_function[kernel_create_unit_vector](
            e_device.unsafe_ptr(), n, pivot_idx,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
        
        # Matvec: Ke = K @ e_pivot (using GradientProvider's forward_matvec)
        provider.forward_matvec(Ke_device.unsafe_ptr(), e_device.unsafe_ptr(), 1)
        
        # Subtract the diagonal noise contribution added by forward_matvec.
        if noise_mode != 0:
            ctx.enqueue_function[kernel_subtract_vector_at_index](
                Ke_device.unsafe_ptr(), pivot_idx, noise_vec_ptr,
                grid_dim=(1,), block_dim=(1,)
            )
        else:
            ctx.enqueue_function[kernel_subtract_at_index](
                Ke_device.unsafe_ptr(), pivot_idx, noise,
                grid_dim=(1,), block_dim=(1,)
            )
        
        # Step 4: Store raw Ke as temporary L column (will orthogonalize then scale)
        ctx.enqueue_function[kernel_scale_and_store_column](
            L_device.unsafe_ptr(), Ke_device.unsafe_ptr(), Float32(1.0), n, m,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
        
        # Step 5: Cholesky orthogonalization against previous columns (GPU)
        # L[:, m] -= sum_j L[j, pivot_m] * L[:, j]
        # Batch-gather all coefficients, then orthogonalize in one kernel.
        if m > 0:
            # Allocate buffers for coefficients
            var coeffs_device = ctx.enqueue_create_buffer[float_dtype](m)
            var coeffs_host = ctx.enqueue_create_host_buffer[float_dtype](m)
            
            # Gather all L[j, pivot_idx] for j = 0..m-1 in one kernel
            ctx.enqueue_function[kernel_gather_L_coefficients](
                coeffs_device.unsafe_ptr(), L_device.unsafe_ptr(), pivot_idx, n, m,
                grid_dim=((m + 255) // 256,), block_dim=(256,)
            )
            # No sync needed - orthogonalization kernel will wait for gather to complete
            
            # Orthogonalize L[:, m] against all previous columns in one kernel
            ctx.enqueue_function[kernel_orthogonalize_L_column_batched](
                L_device.unsafe_ptr(), coeffs_device.unsafe_ptr(), n, m, m,
                grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
            )
            # No sync needed - scale kernel will wait for orthogonalization to complete
        
        # Step 5b: Now scale the orthogonalized column by 1/sqrt(diag[pivot])
        # For rpcholesky, max_diag_value was updated to diag[pivot_idx] above
        if max_diag_value < Float32(1e-10):
            max_diag_value = Float32(1e-10)  # Safety floor
        var scale = Float32(1.0) / sqrt(max_diag_value)
        # Scale L[:, m] in place
        ctx.enqueue_function[kernel_scale_column_inplace](
            L_device.unsafe_ptr(), n, m, scale,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
        
        # Step 6: Update diagonal (GPU) - only for indices not yet used as pivots
        ctx.enqueue_function[kernel_update_diagonal_from_L_masked](
            diag_device.unsafe_ptr(), L_device.unsafe_ptr(), mask_device.unsafe_ptr(), n, m,
            grid_dim=(num_blocks,), block_dim=(BLOCK_SIZE,)
        )
    
    ctx.synchronize()  # SYNC: After all iterations
    
    var t_construct = perf_counter_ns()
    
    # =========================================================================
    # Compute LTL_plus_noise_inv.
    # =========================================================================
    
    # No L resize needed: L is column-major, first actual_rank columns occupy
    # indices [0, actual_rank*n). Consumers use actual_rank, not rank.
    
    var LTL_plus_noise_inv: HostBuffer[float_dtype]
    if noise_mode != 0:
        LTL_plus_noise_inv = _compute_LTL_plus_noise_inv_vector(ctx, L_device, noise_vec_ptr, n, actual_rank)
    else:
        LTL_plus_noise_inv = _compute_LTL_plus_noise_inv(ctx, L_device, n, actual_rank, noise)
    
    # Cache LTL_plus_noise_inv on GPU for fast preconditioner application
    var LTL_plus_noise_inv_device = ctx.enqueue_create_buffer[float_dtype](actual_rank * actual_rank)
    ctx.enqueue_copy(dst_buf=LTL_plus_noise_inv_device, src_buf=LTL_plus_noise_inv)
    ctx.synchronize()
    
    var t_end = perf_counter_ns()
    
    # Allocate work buffers for preconditioner application
    var w_work = ctx.enqueue_create_buffer[float_dtype](actual_rank * max_num_cols)
    var u_work = ctx.enqueue_create_buffer[float_dtype](actual_rank * max_num_cols)
    var z_work = ctx.enqueue_create_buffer[float_dtype](n * max_num_cols)
    
    # Compute log|P| once at construction and cache it
    var precond_temp = PivotedCholeskyPrecond(L_device^, noise, actual_rank, n, LTL_plus_noise_inv^, LTL_plus_noise_inv_device^,
                                  max_num_cols, w_work^, u_work^, z_work^, Float32(0.0), noise_mode, noise_vec_ptr)
    var log_det_val = compute_precond_log_det(ctx, precond_temp)
    precond_temp.cached_log_det = log_det_val
    return precond_temp^
