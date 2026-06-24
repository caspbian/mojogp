"""Conjugate Gradient solver for MojoGP.

Provides preconditioned CG solver for linear systems (K + noise*I) @ x = b.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from gpu.primitives.warp import sum as warp_sum
from gpu.sync import barrier
from gpu.memory import AddressSpace, external_memory
from gpu.globals import WARP_SIZE
from memory import UnsafePointer, stack_allocation
from math import sqrt

alias float_dtype = DType.float32


# =============================================================================
# CG Utility Kernels
# =============================================================================

fn kernel_dot_batched(
    a_ptr: UnsafePointer[Float32, MutAnyOrigin],
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    result_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int
) -> None:
    """Batched dot products with multi-warp reduction: result[col] = a[:, col]^T @ b[:, col].
    
    Uses 256 threads per block (8 warps). Each thread accumulates n/256 elements,
    then warp_sum reduces within each warp. Per-warp results are written to shared
    memory and warp 0 performs the final reduction across 8 warp sums.
    
    Memory layout: COLUMN-MAJOR [num_cols × n] for coalesced access.
    Index: a[col, row] = a_ptr[col * n + row]
    
    Launch: grid_dim=(num_cols,), block_dim=(256,)
    """
    var col = block_idx.x
    var tid = Int(thread_idx.x)         # 0-255 within block
    var num_threads = Int(block_dim.x)  # 256
    var warp_id = tid // WARP_SIZE      # 0-7
    var lane = tid % WARP_SIZE          # 0-31 within warp
    
    if col >= UInt(num_cols):
        return
    
    # Shared memory for per-warp partial sums (8 warps max)
    alias NUM_WARPS = 8
    var warp_sums = stack_allocation[NUM_WARPS, Float32, address_space = AddressSpace.SHARED]()
    
    # Column-major offset for this column
    var col_offset = UInt(col) * UInt(n)
    
    # Each thread computes partial sum (stride by block size = 256)
    var sum_val = Float32(0.0)
    var idx = tid
    while idx < n:
        sum_val += a_ptr[col_offset + UInt(idx)] * b_ptr[col_offset + UInt(idx)]
        idx += num_threads
    
    # Intra-warp reduction (32 threads -> 1 value per warp)
    sum_val = warp_sum(sum_val)
    
    # Lane 0 of each warp writes its result to shared memory
    if lane == 0:
        warp_sums[warp_id] = sum_val
    
    # Synchronize to ensure all warp sums are written
    barrier()
    
    # Warp 0 reduces across the per-warp results
    if warp_id == 0:
        var final_val = Float32(0.0)
        if lane < NUM_WARPS:
            final_val = warp_sums[lane]
        # Warp-level reduction of the 8 values (lanes >= 8 contribute 0)
        final_val = warp_sum(final_val)
        
        # Thread 0 writes the final result
        if lane == 0:
            result_ptr[col] = final_val


fn kernel_axpy_batched(
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int
) -> None:
    """Batched AXPY: y[:, col] += alpha[col] * x[:, col].
    
    Memory layout: COLUMN-MAJOR [num_cols × n] for coalesced access.
    Index: x[col, row] = x_ptr[col * n + row]
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    
    if row >= UInt(n) or col >= UInt(num_cols):
        return
    
    var idx = UInt(col) * UInt(n) + UInt(row)
    y_ptr[idx] += alpha_ptr[col] * x_ptr[idx]


fn kernel_scale_add_batched(
    beta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    p_ptr: UnsafePointer[Float32, MutAnyOrigin],
    r_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int
) -> None:
    """Batched scale-add: p[:, col] = r[:, col] + beta[col] * p[:, col].
    
    Memory layout: COLUMN-MAJOR [num_cols × n] for coalesced access.
    Index: p[col, row] = p_ptr[col * n + row]
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    
    if row >= UInt(n) or col >= UInt(num_cols):
        return
    
    var idx = UInt(col) * UInt(n) + UInt(row)
    var beta_val = beta_ptr[col]
    if beta_val == Float32(0.0) or beta_val != beta_val:
        p_ptr[idx] = r_ptr[idx]
    else:
        var p_old = p_ptr[idx]
        if p_old != p_old:
            p_ptr[idx] = r_ptr[idx]
        else:
            p_ptr[idx] = r_ptr[idx] + beta_val * p_old


fn kernel_cg_update_fused(
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    p_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ap_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    r_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int
) -> None:
    """
    Fused CG update: x += alpha * p AND r -= alpha * Ap in one kernel.
    
    Replaces 3 separate kernels:
    - kernel_axpy_batched: x += alpha * p
    - kernel_negate: alpha = -alpha (wasteful!)
    - kernel_axpy_batched: r += (-alpha) * Ap
    
    This reduces kernel launch overhead and eliminates the negate operation.
    
    Memory layout: COLUMN-MAJOR [num_cols × n] for coalesced access.
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    
    if row >= UInt(n) or col >= UInt(num_cols):
        return
    
    var idx = UInt(col) * UInt(n) + UInt(row)
    var alpha_val = alpha_ptr[col]

    # Freeze broken columns instead of letting 0 * NaN or NaN * v corrupt x/r.
    if alpha_val == Float32(0.0) or alpha_val != alpha_val:
        return
    if p_ptr[idx] != p_ptr[idx] or ap_ptr[idx] != ap_ptr[idx]:
        return
    
    # x += alpha * p
    x_ptr[idx] += alpha_val * p_ptr[idx]
    
    # r -= alpha * Ap (note: subtract, not add with negated alpha)
    r_ptr[idx] -= alpha_val * ap_ptr[idx]


fn kernel_compute_alpha(
    rs_old_ptr: UnsafePointer[Float32, MutAnyOrigin],
    pap_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    num_cols: Int
) -> None:
    """Compute alpha = rs_old / pAp on GPU with GPyTorch-style safe division.
    
    When pAp is near-zero or negative (due to floating-point errors in 
    ill-conditioned matrices), we set alpha = 0 to "freeze" that vector.
    This prevents extreme alpha values that would corrupt the CG solution
    and tridiagonal matrices used for log-det estimation.
    
    Reference: GPyTorch linear_operator/utils/linear_cg.py lines 253-257
    """
    var col = block_idx.x * block_dim.x + thread_idx.x
    
    if col >= UInt(num_cols):
        return
    
    var rs_old = rs_old_ptr[col]
    var pap = pap_ptr[col]
    # GPyTorch-style safe division: if pAp < eps, set alpha = 0 to freeze this vector
    # This prevents extreme alpha values that corrupt tridiagonal matrices
    if rs_old != rs_old or pap != pap or pap < Float32(1e-10):
        alpha_ptr[col] = Float32(0.0)
    else:
        alpha_ptr[col] = rs_old / pap


fn kernel_beta_and_copy_fused(
    rz_old_ptr: UnsafePointer[Float32, MutAnyOrigin],
    rz_new_ptr: UnsafePointer[Float32, MutAnyOrigin],
    beta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    num_cols: Int
) -> None:
    """
    Fused beta computation and rz_old update: beta = rz_new / rz_old, rz_old = rz_new.
    
    Replaces 2 separate kernels:
    - kernel_compute_beta: beta = rz_new / rz_old
    - kernel_copy_scalars: rz_old = rz_new
    
    This reduces kernel launch overhead for tiny scalar operations.
    """
    var col = block_idx.x * block_dim.x + thread_idx.x
    
    if col >= UInt(num_cols):
        return
    
    # Compute beta = rz_new / rz_old
    # Clamp to be non-negative for numerical stability (negative beta causes NaN in sqrt)
    var rz_old_val = rz_old_ptr[col]
    var rz_new_val = rz_new_ptr[col]
    var beta_val = Float32(0.0)
    if (
        rz_old_val == rz_old_val
        and rz_new_val == rz_new_val
        and rz_old_val >= Float32(1e-20)
    ):
        beta_val = rz_new_val / (rz_old_val + Float32(1e-20))
        if beta_val < Float32(0.0) or beta_val != beta_val:
            beta_val = Float32(0.0)
    beta_ptr[col] = beta_val
    
    # Update rz_old = rz_new (for next iteration)
    if rz_new_val == rz_new_val:
        rz_old_ptr[col] = rz_new_val
    else:
        rz_old_ptr[col] = Float32(0.0)


# =============================================================================
# Preconditioning Kernels
# =============================================================================

fn kernel_init_zero_and_copy(
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    r_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int
) -> None:
    """Initialize x = 0 and r = y in one kernel."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(size):
        return
    
    x_ptr[idx] = Float32(0.0)
    r_ptr[idx] = y_ptr[idx]


fn kernel_copy(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int
) -> None:
    """Simple copy kernel: dst = src."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(size):
        return
    
    dst_ptr[idx] = src_ptr[idx]


fn kernel_fill_constant(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
    value: Float32
) -> None:
    """Fill buffer with constant value: dst[i] = value for all i."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(size):
        return
    
    dst_ptr[idx] = value


fn kernel_sum_reduce(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """Single-block parallel reduction: sum n floats to 1 scalar.

    Launch with grid_dim=1, block_dim=256, shared_mem_bytes=num_warps*4.
    Uses strided access + warp_sum + inter-warp reduction via shared memory.
    """
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)

    var partial_sum = Float32(0.0)
    var k = tid
    while k < n:
        partial_sum += in_ptr[k]
        k += bs

    partial_sum = warp_sum(partial_sum)

    var smem = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=16
    ]()
    var warp_id = tid // Int(WARP_SIZE)
    var lane_id = tid % Int(WARP_SIZE)
    var num_warps = (bs + Int(WARP_SIZE) - 1) // Int(WARP_SIZE)

    if lane_id == 0:
        smem[warp_id] = partial_sum
    barrier()

    if warp_id == 0:
        var val = smem[lane_id] if lane_id < num_warps else Float32(0.0)
        val = warp_sum(val)
        if lane_id == 0:
            out_ptr[0] = val


fn kernel_subtract_inplace(
    a_ptr: UnsafePointer[Float32, MutAnyOrigin],
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int
) -> None:
    """Element-wise in-place subtraction: a[i] -= b[i].
    
    Used for CG warm-start: r = rhs - A @ x_init.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(size):
        return
    
    a_ptr[idx] -= b_ptr[idx]


fn kernel_subtract_scalar(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scalar: Float32,
    n: Int,
) -> None:
    """Element-wise scalar subtraction on GPU: out[i] = in[i] - scalar.
    
    Used for ConstantMean y-centering: y_centered = y - mean.
    Replaces CPU loop + H2D copy with a single GPU kernel launch.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    out_ptr[idx] = in_ptr[idx] - scalar


fn kernel_divide_column_by_scalar(
    mat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scalar_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col: Int,
) -> None:
    """Divide a single column of a column-major matrix by a scalar.
    
    mat[:, col] /= scalar[col]
    
    Used for re-normalizing recycled alpha before warm-start CG.
    Column-major layout: element (row, col) is at mat_ptr[col * n + row].
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    
    if row >= UInt(n):
        return
    
    var s = scalar_ptr[col]
    if s != Float32(0.0):
        mat_ptr[UInt(col) * UInt(n) + row] /= s


fn kernel_compute_diagonal_nonstationary(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    d: Int,
    outputscale: Float32,
    kernel_type: Int,
    kernel_param1: Float32,
    kernel_param2: Float32,
) -> None:
    """Compute diagonal of kernel matrix for non-stationary kernels.
    
    For each data point i, computes K[i,i]:
    - Linear: outputscale * (||x_i||² + bias) where bias = kernel_param1
    - Polynomial: outputscale * (||x_i||² + offset)^degree where offset = kernel_param2, degree = kernel_param1
    
    Layout: x is row-major [n, d], diag is [n].
    """
    var i = Int(block_idx.x * block_dim.x + thread_idx.x)
    if i >= n:
        return
    
    alias KERNEL_TYPE_LINEAR = 6
    alias KERNEL_TYPE_POLYNOMIAL = 7
    
    var x_row = x_ptr + UInt(i) * UInt(d)
    
    var norm_sq = Float32(0.0)
    for j in range(d):
        var diff = x_row[j]
        norm_sq += diff * diff
    
    if kernel_type == KERNEL_TYPE_LINEAR:
        var bias = kernel_param1
        diag_ptr[i] = outputscale * (norm_sq + bias)
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        var offset = kernel_param2
        var degree = Int(kernel_param1)
        var base = norm_sq + offset
        var result = Float32(1.0)
        for _ in range(degree):
            result *= base
        diag_ptr[i] = outputscale * result
    else:
        diag_ptr[i] = outputscale




fn kernel_compute_residual_norms_sq(
    residual_norms_sq: UnsafePointer[Float32, MutAnyOrigin],
    r: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Compute ||r[:, col]||^2 for each column using multi-warp reduction.
    
    Each block handles one column. Uses 256 threads (8 warps) with shared memory
    for inter-warp reduction.
    Memory layout: COLUMN-MAJOR [num_cols × n].
    
    Launch: grid_dim=(num_cols,), block_dim=(256,)
    """
    var col = block_idx.x
    if col >= UInt(num_cols):
        return
    
    var tid = Int(thread_idx.x)         # 0-255
    var num_threads = Int(block_dim.x)  # 256
    var warp_id = tid // WARP_SIZE      # 0-7
    var lane = tid % WARP_SIZE          # 0-31
    
    # Shared memory for per-warp partial sums
    alias NUM_WARPS = 8
    var warp_sums = stack_allocation[NUM_WARPS, Float32, address_space = AddressSpace.SHARED]()
    
    var col_offset = UInt(col) * UInt(n)
    
    # Each thread computes partial sum (stride by block size)
    var sum_sq = Float32(0.0)
    var idx = tid
    while idx < n:
        var val = r[col_offset + UInt(idx)]
        sum_sq += val * val
        idx += num_threads
    
    # Intra-warp reduction
    sum_sq = warp_sum(sum_sq)
    
    # Lane 0 of each warp writes to shared memory
    if lane == 0:
        warp_sums[warp_id] = sum_sq
    
    barrier()
    
    # Warp 0 reduces across per-warp results
    if warp_id == 0:
        var final_val = Float32(0.0)
        if lane < NUM_WARPS:
            final_val = warp_sums[lane]
        final_val = warp_sum(final_val)
        if lane == 0:
            residual_norms_sq[col] = final_val


fn kernel_compute_max_residual_norm(
    max_residual: UnsafePointer[Float32, MutAnyOrigin],
    residual_norms_sq: UnsafePointer[Float32, MutAnyOrigin],
    num_cols: Int,
) -> None:
    """Compute max(sqrt(residual_norms_sq)) across all columns.
    
    Single-threaded kernel since num_cols is typically small (10-20).
    """
    var max_val = Float32(0.0)
    for col in range(num_cols):
        var norm = sqrt(residual_norms_sq[col])
        if norm > max_val:
            max_val = norm
    max_residual[0] = max_val


fn kernel_compute_mean_residual_norm(
    mean_residual: UnsafePointer[Float32, MutAnyOrigin],
    residual_norms_sq: UnsafePointer[Float32, MutAnyOrigin],
    num_cols: Int,
) -> None:
    """Compute mean(sqrt(residual_norms_sq)) across all columns.

    The BBMM path normalizes RHS columns before CG, so GPyTorch's convergence
    check is the mean residual norm across columns against the absolute
    tolerance. A single-thread kernel is sufficient because num_cols is small.
    """
    if num_cols <= 0:
        mean_residual[0] = Float32(0.0)
        return

    var sum_val = Float32(0.0)
    for col in range(num_cols):
        sum_val += sqrt(residual_norms_sq[col])
    mean_residual[0] = sum_val / Float32(num_cols)


# =============================================================================
# Result and Type Definitions
# =============================================================================

@fieldwise_init
struct CGResult(Copyable):
    """Result of CG solver.
    
    Fields:
        solution: Solution vector α where (K + σ²I) α = y
        num_iterations: Number of CG iterations performed
        final_residual: Final residual norm ||r||
        converged: Whether solver converged within tolerance
    
    Note:
        Currently Float32-only. Float64 support requires GPU kernel updates.
    """
    var solution: DeviceBuffer[DType.float32]
    var num_iterations: Int
    var final_residual: Float32
    var converged: Bool


@fieldwise_init
struct PreconditionerType(ImplicitlyCopyable):
    """Preconditioner type for the direct CG solver.
    
    Currently implemented:
    - NONE: No preconditioning (z = r)
    
    Note: Production training uses PivotedCholeskyPrecond via the Preconditioner
    trait in combined_inv_quad_logdet.mojo. This enum is only used by pcg_solve/cg_solve.
    """
    var _value: Int
    
    comptime NONE = PreconditionerType(0)
    
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value
    
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# =============================================================================
# Host Functions - CG Solver
# =============================================================================

from .constants import (
    KERNEL_TYPE_RBF,
    KERNEL_TYPE_MATERN12,
    KERNEL_TYPE_MATERN32,
    KERNEL_TYPE_MATERN52,
    KERNEL_TYPE_PERIODIC,
    KERNEL_TYPE_RQ,
    KERNEL_TYPE_LINEAR,
    KERNEL_TYPE_POLYNOMIAL,
)
from .dispatchers_forward import dispatch_forward_matvec as unified_dispatch_forward_matvec
from .kernel_params import KernelParams, make_rbf_params, make_matern_params, make_periodic_params, make_rq_params, make_linear_params, make_polynomial_params


fn dispatch_forward_matvec(
    ctx: DeviceContext,
    kernel_type: Int,
    use_ard: Bool,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin],  # Device pointer: single value for iso, d values for ARD
    lengthscale: Float32,  # For isotropic kernels (ignored if use_ard=True)
    outputscale: Float32,
    n: Int,
    d: Int,
    num_cols: Int,
    noise: Float32,
    kernel_param1: Float32 = 1.0,  # period/alpha/variance/degree
    kernel_param2: Float32 = 0.0,  # polynomial offset
    inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
) raises:
    """Runtime dispatch to correct kernel matvec.
    
    Computes: out = (K + noise*I) @ v
    
    Args:
        ctx: GPU device context.
        kernel_type: Kernel type constant (KERNEL_TYPE_RBF, etc.).
        use_ard: Whether to use ARD (per-dimension lengthscales).
        out_ptr: Output buffer [n, num_cols] column-major.
        x_ptr: Training data [n, d] row-major.
        v_ptr: Input vectors [n, num_cols] column-major.
        lengthscales_ptr: Device pointer to lengthscale(s).
        lengthscale: Lengthscale value for isotropic kernels.
        outputscale: Output scale parameter.
        n: Number of training points.
        d: Input dimension.
        num_cols: Number of RHS columns (batch size).
        noise: Noise variance σ².
        inv_ls_ptr: Precomputed 1/ls[d] on device (ARD only).
    
    Raises:
        Error if kernel_type is unknown.
    
    Note:
        Currently Float32-only. Float64 support requires GPU kernel updates.
    """
    # Create KernelParams based on kernel type
    var params: KernelParams
    
    if kernel_type == KERNEL_TYPE_RBF:
        params = make_rbf_params(outputscale, lengthscale, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN12:
        params = make_matern_params(outputscale, lengthscale, Float32(0.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN32:
        params = make_matern_params(outputscale, lengthscale, Float32(1.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN52:
        params = make_matern_params(outputscale, lengthscale, Float32(2.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_PERIODIC:
        params = make_periodic_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_RQ:
        params = make_rq_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_LINEAR:
        params = make_linear_params(outputscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        params = make_polynomial_params(outputscale, kernel_param1, kernel_param2, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    else:
        raise Error("Unknown kernel_type: " + String(kernel_type))
    
    # Call unified dispatcher
    unified_dispatch_forward_matvec(
        ctx, kernel_type, out_ptr, x_ptr, v_ptr,
        n, d, num_cols, params, noise
    )


fn pcg_solve(
    ctx: DeviceContext,
    kernel_type: Int,
    use_ard: Bool,
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    d: Int,
    num_cols: Int,
    noise: Float32,
    max_iter: Int = 100,
    tol: Float32 = 1e-2,
    preconditioner: PreconditionerType = PreconditionerType.NONE,
    kernel_param1: Float32 = 1.0,
    kernel_param2: Float32 = 0.0,
) raises -> CGResult:
    """Preconditioned Conjugate Gradient solver for (K + σ²I) α = y.
    
    Solves the linear system using the preconditioned CG algorithm with
    optional Jacobi preconditioning.
    
    Args:
        ctx: GPU device context
        kernel_type: Kernel type constant (KERNEL_TYPE_RBF, etc.)
        use_ard: Whether to use ARD (per-dimension lengthscales)
        x_ptr: Training data [n, d] row-major
        params_ptr: Kernel parameters (see dispatch_forward_matvec for layout)
        y_ptr: Right-hand side [n, num_cols] column-major
        n: Number of training points
        d: Input dimension
        num_cols: Number of RHS columns (batch size)
        noise: Noise variance σ²
        max_iter: Maximum CG iterations (default: 100)
        tol: Convergence tolerance on residual norm (default: 1e-6)
        preconditioner: Preconditioner type (default: NONE)
    
    Returns:
        CGResult with solution, iterations, residual, and convergence status
    
    Raises:
        Error if max_iter <= 0, tol <= 0, or kernel_type is unknown
    
    Algorithm (Preconditioned CG):
        1. r_0 = y (since x_0 = 0)
        2. z_0 = M^{-1} @ r_0 (apply preconditioner)
        3. p_0 = z_0
        4. For k = 0 to max_iter-1:
           a. Ap = (K + noise*I) @ p
           b. α = (r·z) / (p·Ap)
           c. x += α*p; r -= α*Ap (fused kernel)
           d. Check ||r|| < tol, break if converged
           e. z = M^{-1} @ r (apply preconditioner)
           f. β = (r_new·z_new) / (r_old·z_old)
           g. p = z + β*p
    
    Note:
        Currently Float32-only. Float64 support requires GPU kernel updates.
    """
    # Input validation
    if max_iter <= 0:
        raise Error("max_iter must be positive")
    if tol <= 0.0:
        raise Error("tol must be positive")
    
    # Extract kernel parameters from device to host
    # Create a temporary device buffer and copy to host to extract lengthscale/outputscale
    var param_size = d + 1 if use_ard else 2
    var params_device_temp = ctx.enqueue_create_buffer[DType.float32](param_size)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](param_size)
    
    # Copy from input params_ptr to temp buffer (device to device)
    var param_copy_threads = 256
    var param_copy_blocks = (param_size + param_copy_threads - 1) // param_copy_threads
    ctx.enqueue_function[kernel_copy](
        params_device_temp.unsafe_ptr(), params_ptr, param_size,
        grid_dim=param_copy_blocks, block_dim=param_copy_threads
    )
    
    # Copy to host
    ctx.enqueue_copy(dst_buf=params_host, src_buf=params_device_temp)
    ctx.synchronize()
    
    # Extract lengthscale and outputscale
    var lengthscale: Float32
    var outputscale: Float32
    if use_ard:
        # ARD: lengthscales are [0..d-1], outputscale is at [d]
        lengthscale = Float32(0.0)  # Not used for ARD
        outputscale = params_host[d]
    else:
        # Isotropic: lengthscale at [0], outputscale at [1]
        lengthscale = params_host[0]
        outputscale = params_host[1]
    
    # Allocate GPU buffers - keep DeviceBuffer for solution (returned to user)
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n * num_cols)  # Solution
    var x_dev = x_buf.unsafe_ptr()
    
    # Allocate raw device memory for temporary buffers
    var r_buf = ctx.enqueue_create_buffer[DType.float32](n * num_cols)  # Residual
    var z_buf = ctx.enqueue_create_buffer[DType.float32](n * num_cols)  # Preconditioned residual
    var p_buf = ctx.enqueue_create_buffer[DType.float32](n * num_cols)  # Search direction
    var ap_buf = ctx.enqueue_create_buffer[DType.float32](n * num_cols)  # Matrix-vector product
    
    var r_dev = r_buf.unsafe_ptr()
    var z_dev = z_buf.unsafe_ptr()
    var p_dev = p_buf.unsafe_ptr()
    var ap_dev = ap_buf.unsafe_ptr()
    
    # Scalar buffers for dot products and CG coefficients
    var rz_old_buf = ctx.enqueue_create_buffer[DType.float32](num_cols)
    var rz_new_buf = ctx.enqueue_create_buffer[DType.float32](num_cols)
    var pap_buf = ctx.enqueue_create_buffer[DType.float32](num_cols)
    var alpha_buf = ctx.enqueue_create_buffer[DType.float32](num_cols)
    var beta_buf = ctx.enqueue_create_buffer[DType.float32](num_cols)
    
    var rz_old_dev = rz_old_buf.unsafe_ptr()
    var rz_new_dev = rz_new_buf.unsafe_ptr()
    var pap_dev = pap_buf.unsafe_ptr()
    var alpha_dev = alpha_buf.unsafe_ptr()
    var beta_dev = beta_buf.unsafe_ptr()
    
    # Initialize: x = 0, r = y (copy from input pointer)
    # Use a simple copy kernel since we can't use memcpy between raw pointers easily
    var init_threads = 256
    var init_blocks = (n * num_cols + init_threads - 1) // init_threads
    ctx.enqueue_function[kernel_init_zero_and_copy](
        x_dev, r_dev, y_ptr, n * num_cols,
        grid_dim=init_blocks, block_dim=init_threads
    )
    
    # No preconditioning: z = r
    var copy_threads_init = 256
    var copy_blocks_init = (n * num_cols + copy_threads_init - 1) // copy_threads_init
    ctx.enqueue_function[kernel_copy](
        z_dev, r_dev, n * num_cols,
        grid_dim=copy_blocks_init, block_dim=copy_threads_init
    )
    
    # Initialize search direction: p = z
    var copy_threads = 256
    var copy_blocks = (n * num_cols + copy_threads - 1) // copy_threads
    ctx.enqueue_function[kernel_copy](
        p_dev, z_dev, n * num_cols,
        grid_dim=copy_blocks, block_dim=copy_threads
    )
    
    # Compute initial rz_old = r · z
    ctx.enqueue_function[kernel_dot_batched](
        r_dev, z_dev, rz_old_dev, n, num_cols,
        grid_dim=num_cols, block_dim=256  # 8 warps per column
    )
    
    # CG iteration loop
    var converged = False
    var final_residual = Float32(0.0)
    var num_iterations = 0
    var initial_residual_norm = Float32(0.0)
    
    for iter in range(max_iter):
        num_iterations = iter + 1
        
        # 1. Compute Ap = (K + noise*I) @ p
        dispatch_forward_matvec(
            ctx, kernel_type, use_ard,
            ap_dev, x_ptr, p_dev, params_ptr,
            lengthscale, outputscale,
            n, d, num_cols, noise,
            kernel_param1, kernel_param2
        )
        
        # 2. Compute p·Ap
        ctx.enqueue_function[kernel_dot_batched](
            p_dev, ap_dev, pap_dev, n, num_cols,
            grid_dim=num_cols, block_dim=256
        )
        
        # 3. Compute alpha = rz_old / (p·Ap)
        var threads_per_block_scalar = 256
        var num_blocks_scalar = (num_cols + threads_per_block_scalar - 1) // threads_per_block_scalar
        ctx.enqueue_function[kernel_compute_alpha](
            rz_old_dev, pap_dev, alpha_dev, num_cols,
            grid_dim=num_blocks_scalar, block_dim=threads_per_block_scalar
        )
        
        # 4. Update x and r: x += alpha*p, r -= alpha*Ap (fused)
        var block_dim_2d = (16, 16)
        var grid_dim_2d = ((n + 15) // 16, (num_cols + 15) // 16)
        ctx.enqueue_function[kernel_cg_update_fused](
            alpha_dev, p_dev, ap_dev, x_dev, r_dev, n, num_cols,
            grid_dim=grid_dim_2d, block_dim=block_dim_2d
        )
        
        # 5. Check convergence: compute ||r|| and compare to tol
        # Check every iteration; batching this check is a future performance optimization.
        # Copy residual to host to check norm (simplified - could do on GPU)
        var r_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
        ctx.enqueue_copy(dst_buf=r_host, src_buf=r_buf)
        ctx.synchronize()
        
        # Compute max residual norm across all columns
        var max_residual_norm = Float32(0.0)
        for col in range(num_cols):
            var col_norm_sq = Float32(0.0)
            for row in range(n):
                var val = r_host[col * n + row]
                col_norm_sq += val * val
            var col_norm = sqrt(col_norm_sq)
            if col_norm > max_residual_norm:
                max_residual_norm = col_norm
        
        final_residual = max_residual_norm
        
        # Track initial residual for relative convergence
        if initial_residual_norm == Float32(0.0):
            initial_residual_norm = max_residual_norm
            if initial_residual_norm < Float32(1e-30):
                initial_residual_norm = Float32(1.0)
        
        # Relative convergence check (matching GPyTorch)
        var rel_residual = max_residual_norm / initial_residual_norm
        if rel_residual < tol:
            converged = True
            break
        
        # 6. No preconditioning: z = r
        var copy_threads_loop = 256
        var copy_blocks_loop = (n * num_cols + copy_threads_loop - 1) // copy_threads_loop
        ctx.enqueue_function[kernel_copy](
            z_dev, r_dev, n * num_cols,
            grid_dim=copy_blocks_loop, block_dim=copy_threads_loop
        )
        
        # 7. Compute rz_new = r · z
        ctx.enqueue_function[kernel_dot_batched](
            r_dev, z_dev, rz_new_dev, n, num_cols,
            grid_dim=num_cols, block_dim=256
        )
        
        # 8. Compute beta = rz_new / rz_old and update rz_old (fused)
        ctx.enqueue_function[kernel_beta_and_copy_fused](
            rz_old_dev, rz_new_dev, beta_dev, num_cols,
            grid_dim=num_blocks_scalar, block_dim=threads_per_block_scalar
        )
        
        # 9. Update search direction: p = z + beta*p
        ctx.enqueue_function[kernel_scale_add_batched](
            beta_dev, p_dev, z_dev, n, num_cols,
            grid_dim=grid_dim_2d, block_dim=block_dim_2d
        )
    
    # Return result
    return CGResult(
        solution=x_buf,
        num_iterations=num_iterations,
        final_residual=final_residual,
        converged=converged
    )


fn cg_solve(
    ctx: DeviceContext,
    kernel_type: Int,
    use_ard: Bool,
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    d: Int,
    num_cols: Int,
    noise: Float32,
    max_iter: Int = 100,
    tol: Float32 = 1e-2,
    kernel_param1: Float32 = 1.0,
    kernel_param2: Float32 = 0.0,
) raises -> CGResult:
    """Conjugate Gradient solver without preconditioning.
    
    Convenience wrapper for pcg_solve with preconditioner=NONE.
    
    Args:
        ctx: GPU device context
        kernel_type: Kernel type constant (KERNEL_TYPE_RBF, etc.)
        use_ard: Whether to use ARD (per-dimension lengthscales)
        x_ptr: Training data [n, d] row-major
        params_ptr: Kernel parameters (see dispatch_forward_matvec for layout)
        y_ptr: Right-hand side [n, num_cols] column-major
        n: Number of training points
        d: Input dimension
        num_cols: Number of RHS columns (batch size)
        noise: Noise variance σ²
        max_iter: Maximum CG iterations (default: 100)
        tol: Convergence tolerance on residual norm (default: 1e-6)
    
    Returns:
        CGResult with solution, iterations, residual, and convergence status
    
    Raises:
        Error if max_iter <= 0, tol <= 0, or kernel_type is unknown
    
    Note:
        Currently Float32-only. Float64 support requires GPU kernel updates.
    """
    return pcg_solve(
        ctx, kernel_type, use_ard, x_ptr, params_ptr, y_ptr,
        n, d, num_cols, noise, max_iter, tol,
        preconditioner=PreconditionerType.NONE,
        kernel_param1=kernel_param1,
        kernel_param2=kernel_param2
    )
