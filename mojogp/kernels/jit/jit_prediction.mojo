"""JIT prediction functions using ErasedJITProvider.

Provides mean prediction, LOVE variance, and exact variance for the
fn-ptr JIT engine path. All operations use fn ptrs from the kernel .so.

Architecture:
    kernel .so provides: cross_matvec, extract_diagonal_test, forward_matvec
    engine .so provides: CG solver, preconditioner, Lanczos (this file)
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import sqrt, ceildiv
from os import getenv
from time import perf_counter_ns
from buffer import NDBuffer
from linalg.matmul.vendor.blas import matmul as blas_matmul

from kernels.jit.erased_provider import ErasedJITProvider
from kernels.gradient_provider import ForwardProvider
from kernels.combined_inv_quad_logdet import batched_cg_unified, CGBufferPool, CGResultWithTridiag
from kernels.cg_solver import kernel_copy, kernel_fill_constant
from kernels.preconditioner_trait import Preconditioner
from kernels.pivoted_cholesky import (
    build_pivoted_cholesky_precond_unified,
    PivotedCholeskyPrecond,
    kernel_gpu_gaussian,
    normalize_columns_gpu,
)
from kernels.bbmm_gpu_kernels import scale_columns_by_norms
from kernels.constants import float_dtype, PROFILING
from kernels.native_numerics import tridiagonal_eigh_native
from gpu.profiler import ProfileBlock


# =============================================================================
# Prediction Result
# =============================================================================


struct PredictionResultJIT(Movable):
    """Result from JIT prediction."""
    var mean: HostBuffer[float_dtype]       # [n_test]
    var variance: HostBuffer[float_dtype]   # [n_test] (zeros if method=0)
    var n_test: Int
    var has_variance: Bool
    var exact_block_cols: Int
    var exact_cross_mode: Int
    var exact_cg_block_count: Int
    var exact_cg_total_iterations: Int
    var exact_cg_max_iterations: Int
    var exact_alloc_time_ns: Int
    var exact_cross_time_ns: Int
    var exact_diag_time_ns: Int
    var exact_solve_time_ns: Int
    var exact_post_time_ns: Int
    var love_alloc_time_ns: Int
    var love_cross_time_ns: Int
    var love_diag_time_ns: Int
    var love_post_time_ns: Int
    var love_cross_strategy: Int
    var love_cross_chunk_width: Int
    var alpha_time_ns: Int
    var love_root_cache_used: Bool
    var mean_time_ns: Int
    var variance_time_ns: Int
    var total_time_ns: Int
    
    fn __init__(out self, var mean: HostBuffer[float_dtype],
                 var variance: HostBuffer[float_dtype],
                 n_test: Int, has_variance: Bool,
                 exact_block_cols: Int, exact_cross_mode: Int,
                 alpha_time_ns: Int, love_root_cache_used: Bool, mean_time_ns: Int,
                 variance_time_ns: Int, total_time_ns: Int,
                 exact_cg_block_count: Int = 0,
                 exact_cg_total_iterations: Int = 0,
                 exact_cg_max_iterations: Int = 0,
                 exact_alloc_time_ns: Int = 0,
                 exact_cross_time_ns: Int = 0,
                 exact_diag_time_ns: Int = 0,
                 exact_solve_time_ns: Int = 0,
                 exact_post_time_ns: Int = 0,
                 love_alloc_time_ns: Int = 0,
                 love_cross_time_ns: Int = 0,
                 love_diag_time_ns: Int = 0,
                 love_post_time_ns: Int = 0,
                 love_cross_strategy: Int = 0,
                 love_cross_chunk_width: Int = 0):
        self.mean = mean^
        self.variance = variance^
        self.n_test = n_test
        self.has_variance = has_variance
        self.exact_block_cols = exact_block_cols
        self.exact_cross_mode = exact_cross_mode
        self.exact_cg_block_count = exact_cg_block_count
        self.exact_cg_total_iterations = exact_cg_total_iterations
        self.exact_cg_max_iterations = exact_cg_max_iterations
        self.exact_alloc_time_ns = exact_alloc_time_ns
        self.exact_cross_time_ns = exact_cross_time_ns
        self.exact_diag_time_ns = exact_diag_time_ns
        self.exact_solve_time_ns = exact_solve_time_ns
        self.exact_post_time_ns = exact_post_time_ns
        self.love_alloc_time_ns = love_alloc_time_ns
        self.love_cross_time_ns = love_cross_time_ns
        self.love_diag_time_ns = love_diag_time_ns
        self.love_post_time_ns = love_post_time_ns
        self.love_cross_strategy = love_cross_strategy
        self.love_cross_chunk_width = love_cross_chunk_width
        self.alpha_time_ns = alpha_time_ns
        self.love_root_cache_used = love_root_cache_used
        self.mean_time_ns = mean_time_ns
        self.variance_time_ns = variance_time_ns
        self.total_time_ns = total_time_ns
    
    fn __moveinit__(out self, owned other: Self):
        self.mean = other.mean^
        self.variance = other.variance^
        self.n_test = other.n_test
        self.has_variance = other.has_variance
        self.exact_block_cols = other.exact_block_cols
        self.exact_cross_mode = other.exact_cross_mode
        self.exact_cg_block_count = other.exact_cg_block_count
        self.exact_cg_total_iterations = other.exact_cg_total_iterations
        self.exact_cg_max_iterations = other.exact_cg_max_iterations
        self.exact_alloc_time_ns = other.exact_alloc_time_ns
        self.exact_cross_time_ns = other.exact_cross_time_ns
        self.exact_diag_time_ns = other.exact_diag_time_ns
        self.exact_solve_time_ns = other.exact_solve_time_ns
        self.exact_post_time_ns = other.exact_post_time_ns
        self.love_alloc_time_ns = other.love_alloc_time_ns
        self.love_cross_time_ns = other.love_cross_time_ns
        self.love_diag_time_ns = other.love_diag_time_ns
        self.love_post_time_ns = other.love_post_time_ns
        self.love_cross_strategy = other.love_cross_strategy
        self.love_cross_chunk_width = other.love_cross_chunk_width
        self.alpha_time_ns = other.alpha_time_ns
        self.love_root_cache_used = other.love_root_cache_used
        self.mean_time_ns = other.mean_time_ns
        self.variance_time_ns = other.variance_time_ns
        self.total_time_ns = other.total_time_ns


@fieldwise_init
struct IdentityPreconditioner(Preconditioner, Copyable):
    """Allocation-free identity preconditioner for no-preconditioner solves."""

    var n: Int

    fn apply_precond(
        self,
        ctx: DeviceContext,
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        num_cols: Int,
        sync: Bool,
    ) raises:
        _ = self
        ctx.enqueue_function[kernel_copy](
            out_ptr,
            v_ptr,
            n * num_cols,
            grid_dim=((n * num_cols + 255) // 256,),
            block_dim=(256,),
        )
        if sync:
            ctx.synchronize()

    fn sample_probes(
        self,
        ctx: DeviceContext,
        out_device: DeviceBuffer[float_dtype],
        num_probes: Int,
        seed_val: UInt64,
    ) raises:
        if num_probes <= 0:
            return
        var ns_grid = (self.n * num_probes + 255) // 256
        ctx.enqueue_function[kernel_gpu_gaussian](
            out_device.unsafe_ptr(),
            self.n * num_probes,
            seed_val,
            grid_dim=(ns_grid,),
            block_dim=(256,),
        )

    fn log_det(self, ctx: DeviceContext) raises -> Float32:
        _ = self
        _ = ctx
        return Float32(0.0)

# =============================================================================
# GPU Kernels for prediction post-processing
# =============================================================================


fn kernel_compute_mean_from_cross[BLOCK: Int](
    mean_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cross_alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mean_offset: Float32,
    n_test: Int,
) -> None:
    """mean[i] = cross_alpha[i] + mean_offset."""
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if i >= n_test:
        return
    mean_ptr[i] = cross_alpha_ptr[i] + mean_offset


fn kernel_love_variance[BLOCK: Int](
    var_ptr: UnsafePointer[Float32, MutAnyOrigin],
    V_ptr: UnsafePointer[Float32, MutAnyOrigin],
    diag_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
    rank: Int,
) -> None:
    """var[j] = diag_test[j] - ||V[:,j]||^2, clamped to 1e-10.
    
    V is [rank × n_test] column-major: V[k,j] = V_ptr[k * n_test + j].
    """
    var j = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if j >= n_test:
        return
    var vnorm_sq = Float32(0)
    for k in range(rank):
        var v = V_ptr[k * n_test + j]
        vnorm_sq += v * v
    var variance = diag_test_ptr[j] - vnorm_sq
    if variance < Float32(1e-10):
        variance = Float32(1e-10)
    var_ptr[j] = variance


fn kernel_exact_variance[BLOCK: Int](
    var_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cross_ptr: UnsafePointer[Float32, MutAnyOrigin],
    solve_ptr: UnsafePointer[Float32, MutAnyOrigin],
    diag_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
) -> None:
    """var[j] = diag_test[j] - dot(cross[:,j], solve[:,j]), clamped to 1e-10.
    
    cross and solve are [n_train × n_test] column-major.
    """
    var j = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if j >= n_test:
        return
    var dot = Float32(0)
    for i in range(n_train):
        dot += cross_ptr[j * n_train + i] * solve_ptr[j * n_train + i]
    var variance = diag_test_ptr[j] - dot
    if variance < Float32(1e-10):
        variance = Float32(1e-10)
    var_ptr[j] = variance


alias LOVE_CROSS_STRATEGY_FUSED = 0
alias LOVE_CROSS_STRATEGY_MATERIALIZE_BLAS = 1


fn love_cross_strategy_from_env() -> Int:
    """Runtime experiment switch for LOVE cross-root multiply strategy."""
    var raw = String(getenv("MOJOGP_LOVE_CROSS_STRATEGY", "fused"))
    if raw == "materialize_blas" or raw == "blas":
        return LOVE_CROSS_STRATEGY_MATERIALIZE_BLAS
    return LOVE_CROSS_STRATEGY_FUSED


fn love_cross_chunk_width_from_env(rank: Int) -> Int:
    """Runtime experiment switch for grouping generated cross-matvec chunks."""
    var width: Int
    try:
        width = Int(getenv("MOJOGP_LOVE_CROSS_CHUNK_WIDTH", String(rank)))
    except:
        width = rank
    if width <= 0:
        width = rank
    if width > rank:
        width = rank
    if width <= 0:
        width = 1
    return width


fn love_reduced_sync_enabled() -> Bool:
    """Runtime experiment switch for coarser provider-call synchronization."""
    var raw = String(getenv("MOJOGP_LOVE_SYNC_MODE", "safe"))
    return raw == "reduced"


fn exact_blocked_blas_enabled() -> Bool:
    """Standard exact-prediction wide-RHS matvec, with env opt-out."""
    var raw = String(getenv("MOJOGP_EXACT_BLOCKED_BLAS_MATVEC", "1"))
    return not (raw == "0" or raw == "false" or raw == "no")


fn exact_blocked_blas_min_cols() -> Int:
    var threshold: Int
    try:
        threshold = Int(getenv("MOJOGP_EXACT_BLOCKED_BLAS_MIN_COLS", "64"))
    except:
        threshold = 64
    if threshold < 1:
        threshold = 1
    return threshold


fn exact_blocked_blas_tile_cols(n: Int) -> Int:
    var tile_cols: Int
    try:
        tile_cols = Int(getenv("MOJOGP_EXACT_BLOCKED_BLAS_TILE_COLS", "1024"))
    except:
        tile_cols = 1024
    if tile_cols < 1:
        tile_cols = 1
    if tile_cols > n:
        tile_cols = n
    return tile_cols


fn kernel_transpose_cross_covariance[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
) -> None:
    """Transpose K(X_test, X_train) layout into batched-CG RHS layout.

    Input layout from cross_matvec with `num_cols=n_train`:
        in_ptr[train_col * n_test + test_row] = K(x_test_row, x_train_col)

    Output layout required by batched CG with `num_cols=n_test`:
        out_ptr[test_col * n_train + train_row] = K(x_train_row, x_test_col)
    """
    var idx = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var total = n_train * n_test
    if idx >= total:
        return

    var test_col = idx // n_train
    var train_row = idx - test_col * n_train
    out_ptr[idx] = in_ptr[train_row * n_test + test_col]


fn kernel_transpose_cross_covariance_chunk[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    train_offset: Int,
    chunk_cols: Int,
) -> None:
    """Transpose a basis-column chunk into the full cross-covariance buffer.

    `in_ptr` stores `chunk_cols` cross-matvec outputs in the layout emitted by
    `cross_matvec`: `in_ptr[local_col * n_test + test_row]`.

    The destination keeps the full batched-CG RHS layout:
    `out_ptr[test_col * n_train + train_row]`.
    """
    var idx = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var total = chunk_cols * n_test
    if idx >= total:
        return

    var local_col = idx // n_test
    var test_row = idx - local_col * n_test
    out_ptr[test_row * n_train + train_offset + local_col] = in_ptr[idx]


fn kernel_pack_rhs_tile[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    train_start: Int,
    tile_cols: Int,
) -> None:
    """Pack strided RHS rows into row-major [num_cols x tile_cols]."""
    var idx = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var total = num_cols * tile_cols
    if idx >= total:
        return

    var col = idx // tile_cols
    var local_row = idx - col * tile_cols
    out_ptr[idx] = v_ptr[col * n + train_start + local_row]


fn kernel_add_inplace[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    add_ptr: UnsafePointer[Float32, MutAnyOrigin],
    total: Int,
) -> None:
    """out += add for column-major buffers reinterpreted as row-major."""
    var idx = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if idx >= total:
        return
    out_ptr[idx] += add_ptr[idx]


fn kernel_add_noise_inplace[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    total: Int,
    noise: Float32,
) -> None:
    """out += noise * v."""
    var idx = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if idx >= total:
        return
    out_ptr[idx] += noise * v_ptr[idx]


fn kernel_form_lanczos_inv_root[BLOCK: Int](
    s_ptr: UnsafePointer[Float32, MutAnyOrigin],
    q_ptr: UnsafePointer[Float32, MutAnyOrigin],
    t_inv_sqrt_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    r: Int,
    total: Int,
) -> None:
    """Form S = Q @ T^{-1/2} with column-major [rank, n] storage."""
    var idx = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if idx >= total:
        return

    var row = idx % n
    var col = idx // n
    var sum_val = Float32(0.0)
    for k in range(r):
        sum_val += q_ptr[k * n + row] * t_inv_sqrt_ptr[k * r + col]
    s_ptr[col * n + row] = sum_val


fn blocked_blas_forward_matvec_jit(
    provider: ErasedJITProvider,
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    dim: Int,
    tile_cols: Int,
) raises:
    """Compute `(K + noise I) @ V` using bounded K tiles and vendor BLAS.

    This is the standard exact-prediction path for wide RHS blocks. It keeps
    matrix-free storage bounded by `O(n * tile_cols + n * num_cols)` rather than
    materializing the full train-train kernel.
    """
    alias BLOCK = 256
    var total = n * num_cols
    ctx.enqueue_function[kernel_fill_constant](
        out_ptr,
        total,
        Float32(0.0),
        grid_dim=((total + BLOCK - 1) // BLOCK,),
        block_dim=(BLOCK,),
    )

    var K_tile = ctx.enqueue_create_buffer[float_dtype](n * tile_cols)
    var V_tile = ctx.enqueue_create_buffer[float_dtype](num_cols * tile_cols)
    var tile_out = ctx.enqueue_create_buffer[float_dtype](total)

    var train_start = 0
    while train_start < n:
        var active_tile_cols = tile_cols
        var remaining = n - train_start
        if remaining < active_tile_cols:
            active_tile_cols = remaining

        provider.fill_cross_covariance(
            K_tile.unsafe_ptr(),
            provider.get_x_ptr().offset(train_start * dim),
            active_tile_cols,
        )

        var packed_total = num_cols * active_tile_cols
        ctx.enqueue_function[kernel_pack_rhs_tile[BLOCK]](
            V_tile.unsafe_ptr(),
            v_ptr,
            n,
            num_cols,
            train_start,
            active_tile_cols,
            grid_dim=((packed_total + BLOCK - 1) // BLOCK,),
            block_dim=(BLOCK,),
        )

        var V_ndbuf = NDBuffer[DType.float32, 2](
            V_tile.unsafe_ptr(), (num_cols, active_tile_cols)
        )
        var K_ndbuf = NDBuffer[DType.float32, 2](
            K_tile.unsafe_ptr(), (active_tile_cols, n)
        )
        var tile_out_ndbuf = NDBuffer[DType.float32, 2](
            tile_out.unsafe_ptr(), (num_cols, n)
        )
        blas_matmul[use_tf32=False](
            ctx,
            tile_out_ndbuf,
            V_ndbuf,
            K_ndbuf,
            c_row_major=True,
            transpose_a=False,
            transpose_b=False,
        )
        ctx.enqueue_function[kernel_add_inplace[BLOCK]](
            out_ptr,
            tile_out.unsafe_ptr(),
            total,
            grid_dim=((total + BLOCK - 1) // BLOCK,),
            block_dim=(BLOCK,),
        )

        # The next K_tile fill is launched by the generated provider's own
        # DeviceContext, so sync this context before K_tile is reused.
        ctx.synchronize()
        train_start += active_tile_cols

    ctx.enqueue_function[kernel_add_noise_inplace[BLOCK]](
        out_ptr,
        v_ptr,
        total,
        provider.get_noise(),
        grid_dim=((total + BLOCK - 1) // BLOCK,),
        block_dim=(BLOCK,),
    )
    ctx.synchronize()

    _ = K_tile
    _ = V_tile
    _ = tile_out


@fieldwise_init
struct ExactPredictionBlockedBLASProvider(ForwardProvider, Copyable):
    """ForwardProvider wrapper for exact-prediction wide-RHS CG solves."""

    var base: ErasedJITProvider
    var ctx: DeviceContext
    var n: Int
    var dim: Int
    var tile_cols: Int

    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        blocked_blas_forward_matvec_jit(
            self.base,
            self.ctx,
            out_ptr,
            v_ptr,
            self.n,
            num_cols,
            self.dim,
            self.tile_cols,
        )

    fn get_n(self) -> Int:
        return self.base.get_n()

    fn get_ctx(self) -> DeviceContext:
        return self.base.get_ctx()

    fn get_noise(self) -> Float32:
        return self.base.get_noise()

    fn get_diagonal_value(self) -> Float32:
        return self.base.get_diagonal_value()

    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        self.base.extract_diagonal(diag_ptr)

    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.base.get_x_ptr()


fn build_cross_covariance_from_cross_matvec_fallback_jit(
    provider: ErasedJITProvider,
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
) raises:
    """Synthesize `K(X_train, X_test)` without allocating an `n_train x n_train` identity.

    The fallback path batches canonical basis columns in small chunks, runs the
    existing `cross_matvec` kernel, and transposes each chunk directly into the
    final `[n_train x n_test]` RHS layout used by batched CG.
    """
    var max_basis_cols = 16
    var basis_host = ctx.enqueue_create_host_buffer[float_dtype](
        max(n_train * max_basis_cols, 1)
    )
    var basis_device = ctx.enqueue_create_buffer[float_dtype](
        max(n_train * max_basis_cols, 1)
    )
    var cross_chunk = ctx.enqueue_create_buffer[float_dtype](
        max(n_test * max_basis_cols, 1)
    )

    alias BLOCK = 256
    var train_start = 0
    with ProfileBlock[PROFILING]("JIT_predict_exact_cross_chunked"):
        while train_start < n_train:
            var remaining = n_train - train_start
            var chunk_cols = 1
            if remaining >= 16:
                chunk_cols = 16
            elif remaining >= 11:
                chunk_cols = 11
            elif remaining >= 6:
                chunk_cols = 6

            var used = n_train * chunk_cols
            for i in range(used):
                basis_host.unsafe_ptr()[i] = Float32(0.0)
            for local_col in range(chunk_cols):
                basis_host.unsafe_ptr()[local_col * n_train + train_start + local_col] = Float32(1.0)

            ctx.enqueue_copy(basis_device, basis_host)
            provider.cross_matvec(
                cross_chunk.unsafe_ptr(),
                x_test_device_ptr,
                basis_device.unsafe_ptr(),
                n_test,
                chunk_cols,
            )
            ctx.enqueue_function[kernel_transpose_cross_covariance_chunk[BLOCK]](
                out_ptr,
                cross_chunk.unsafe_ptr(),
                n_train,
                n_test,
                train_start,
                chunk_cols,
                grid_dim=((chunk_cols * n_test + BLOCK - 1) // BLOCK,),
                block_dim=(BLOCK,),
            )
            ctx.synchronize()
            train_start += chunk_cols

    _ = basis_host
    _ = basis_device
    _ = cross_chunk


# =============================================================================
# GPU helper: subtract scalar
# =============================================================================


fn kernel_subtract_scalar_pred[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scalar: Float32,
    n: Int,
) -> None:
    """out[i] = in[i] - scalar."""
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if i >= n:
        return
    out_ptr[i] = in_ptr[i] - scalar


# =============================================================================
# Deterministic host-orchestrated CG for prediction-time solves
# =============================================================================


fn solve_single_rhs_deterministic_host_jit[P: ForwardProvider](
    provider: P,
    ctx: DeviceContext,
    rhs_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    max_cg_iter: Int,
    cg_tol: Float32,
) raises -> HostBuffer[float_dtype]:
    """Solve (K + sigma^2 I) x = rhs with deterministic host reductions.

    Prediction-time exact solves must be stable across repeated calls and
    save/load. The training CG path keeps reductions and preconditioning on the
    GPU for speed, but that introduces enough numerical drift to break these
    exact prediction invariants. This helper keeps the expensive matvec on the
    GPU while doing all scalar reductions and vector updates on the host in a
    fixed order.
    """
    var x_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var r_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var p_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var ap_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var p_device = ctx.enqueue_create_buffer[float_dtype](n)
    var ap_device = ctx.enqueue_create_buffer[float_dtype](n)

    var rr_old = Float32(0.0)
    for i in range(n):
        var rhs_i = rhs_host_ptr[i]
        x_host[i] = Float32(0.0)
        r_host[i] = rhs_i
        p_host[i] = rhs_i
        rr_old += rhs_i * rhs_i

    var initial_residual_norm = sqrt(rr_old)
    if initial_residual_norm < Float32(1e-20):
        initial_residual_norm = Float32(1.0)

    for _iter in range(max_cg_iter):
        ctx.enqueue_copy(p_device, p_host)
        provider.forward_matvec(ap_device.unsafe_ptr(), p_device.unsafe_ptr(), 1)
        ctx.enqueue_copy(ap_host, ap_device)
        ctx.synchronize()

        var pAp = Float32(0.0)
        for i in range(n):
            pAp += p_host[i] * ap_host[i]

        if pAp != pAp or pAp < Float32(1e-10):
            break

        var alpha = rr_old / pAp
        if alpha != alpha:
            break

        var rr_new = Float32(0.0)
        for i in range(n):
            x_host[i] += alpha * p_host[i]
            r_host[i] -= alpha * ap_host[i]
            rr_new += r_host[i] * r_host[i]

        var residual_norm = sqrt(rr_new)
        if residual_norm / initial_residual_norm < cg_tol:
            break

        var beta = Float32(0.0)
        if rr_old >= Float32(1e-20):
            beta = rr_new / (rr_old + Float32(1e-20))
            if beta < Float32(0.0) or beta != beta:
                beta = Float32(0.0)

        for i in range(n):
            p_host[i] = r_host[i] + beta * p_host[i]
        rr_old = rr_new

    _ = r_host
    _ = p_host
    _ = ap_host
    _ = p_device
    _ = ap_device
    return x_host^


# =============================================================================
# Compute alpha: (K + σ²I)^{-1} @ (y - mean)
# =============================================================================


fn compute_alpha_jit(
    provider: ErasedJITProvider,
    ctx: DeviceContext,
    y_centered_device: DeviceBuffer[float_dtype],
    n: Int,
    max_cg_iter: Int,
    cg_tol: Float32,
    precond_rank: Int,
    use_preconditioner: Bool,
) raises -> DeviceBuffer[float_dtype]:
    """Compute alpha = (K + σ²I)^{-1} @ (y - mean) using CG.

    Returns alpha as a DeviceBuffer [n].
    """
    # Prediction-time exact solves prioritize repeatability over preconditioned
    # GPU CG throughput. Keep matvecs on GPU but do reductions/updates on host.
    var alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
    var y_centered_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    with ProfileBlock[PROFILING]("JIT_predict_alpha_copy_in"):
        ctx.enqueue_copy(y_centered_host, y_centered_device)
        ctx.synchronize()

    var alpha_host = solve_single_rhs_deterministic_host_jit(
        provider,
        ctx,
        y_centered_host.unsafe_ptr(),
        n,
        max_cg_iter,
        cg_tol,
    )
    with ProfileBlock[PROFILING]("JIT_predict_alpha_solve"):
        ctx.enqueue_copy(alpha_device, alpha_host)
        ctx.synchronize()

    _ = y_centered_host
    _ = alpha_host
    _ = precond_rank
    _ = use_preconditioner

    return alpha_device^


fn center_targets_jit(
    ctx: DeviceContext,
    y_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    mean: Float32,
) raises -> DeviceBuffer[float_dtype]:
    """Center training targets on GPU: y_centered = y - mean."""
    var y_centered = ctx.enqueue_create_buffer[float_dtype](n_train)
    alias BLK = 256
    var grid_center = ceildiv(n_train, BLK)
    with ProfileBlock[PROFILING]("JIT_predict_center_y"):
        ctx.enqueue_function[kernel_subtract_scalar_pred[BLK]](
            y_centered.unsafe_ptr(), y_device_ptr, mean, n_train,
            grid_dim=grid_center, block_dim=BLK,
        )
        ctx.synchronize()
    return y_centered^


# =============================================================================
# Mean Prediction
# =============================================================================


fn predict_mean_jit(
    provider: ErasedJITProvider,
    ctx: DeviceContext,
    alpha_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    mean: Float32,
) raises -> DeviceBuffer[float_dtype]:
    """Compute mean_pred = K(X_test, X_train) @ alpha + mean.
    
    Uses cross_matvec fn-ptr from kernel .so.
    Returns mean as DeviceBuffer [n_test].
    """
    # cross_matvec: K(X_test, X_train) @ alpha → [n_test]
    var cross_alpha = ctx.enqueue_create_buffer[float_dtype](n_test)
    var mean_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    with ProfileBlock[PROFILING]("JIT_predict_mean_cross"):
        provider.cross_matvec(
            cross_alpha.unsafe_ptr(),
            x_test_device_ptr,
            alpha_device_ptr,
            n_test,
            1,  # num_cols = 1
        )
    
    # Add mean offset: mean_pred = cross_alpha + mean
    alias BLOCK = 256
    var grid = ceildiv(n_test, BLOCK)
    with ProfileBlock[PROFILING]("JIT_predict_mean_post"):
        ctx.enqueue_function[kernel_compute_mean_from_cross[BLOCK]](
            mean_device.unsafe_ptr(), cross_alpha.unsafe_ptr(), mean, n_test,
            grid_dim=grid, block_dim=BLOCK,
        )
        ctx.synchronize()
    
    _ = cross_alpha
    return mean_device^


fn cross_matvec_specialized_chunks_jit(
    provider: ErasedJITProvider,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    num_cols: Int,
    requested_chunk_width: Int = 16,
) raises:
    """Launch cross_matvec only with compiled NCOLS specializations.

    The generated provider has its own NCOLS fallback ladder. Passing a larger
    `num_cols` groups more of that ladder behind one provider callback and one
    provider-side synchronize, which is useful for LOVE root-multiply sweeps.
    """
    var max_chunk = requested_chunk_width
    if max_chunk <= 0:
        max_chunk = 16
    if max_chunk > num_cols:
        max_chunk = num_cols
    var col_start = 0
    while col_start < num_cols:
        var remaining = num_cols - col_start
        var chunk_cols = max_chunk
        if chunk_cols > remaining:
            chunk_cols = remaining

        provider.cross_matvec(
            out_ptr.offset(col_start * n_test),
            x_test_device_ptr,
            v_ptr.offset(col_start * n_train),
            n_test,
            chunk_cols,
        )
        col_start += chunk_cols


fn love_cross_root_materialize_blas_jit(
    provider: ErasedJITProvider,
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    inv_root_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    rank: Int,
) raises -> Bool:
    """Materialize K(X_test, X_train) and compute K_cross @ S with vendor BLAS.

    `fill_cross_covariance` writes test columns contiguously as
    `out[test * n_train + train]`. Reinterpreting that as row-major
    `[n_test x n_train]`, and reinterpreting the column-major inverse root and
    output as row-major transposes, lets BLAS compute
    `(K_cross @ S)^T = S^T @ K_cross^T` directly into the existing LOVE layout.
    """
    if not provider.has_fill_cross_covariance():
        return False

    var K_cross = ctx.enqueue_create_buffer[float_dtype](n_train * n_test)
    provider.fill_cross_covariance(
        K_cross.unsafe_ptr(),
        x_test_device_ptr,
        n_test,
    )

    var K_tensor = NDBuffer[DType.float32, 2](K_cross.unsafe_ptr(), (n_test, n_train))
    var S_tensor = NDBuffer[DType.float32, 2](inv_root_device_ptr, (rank, n_train))
    var out_tensor = NDBuffer[DType.float32, 2](out_ptr, (rank, n_test))
    blas_matmul[use_tf32=False](
        ctx, out_tensor, S_tensor, K_tensor,
        c_row_major=True, transpose_a=False, transpose_b=True
    )
    ctx.synchronize()

    _ = K_cross
    return True


fn populate_test_diagonal_jit(
    provider: ErasedJITProvider,
    diag_provider: ErasedJITProvider,
    use_diag_provider: Bool,
    ctx: DeviceContext,
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
) raises:
    """Populate predictive test diagonal using the safest available route."""
    if use_diag_provider:
        diag_provider.extract_diagonal(diag_ptr)
    else:
        provider.extract_diagonal_test(diag_ptr, x_test_device_ptr, n_test)
    ctx.synchronize()


# =============================================================================
# LOVE Variance
# =============================================================================


fn compute_lanczos_inv_root_jit[P: ForwardProvider](
    provider: P,
    lanczos_iter: Int,
) raises -> HostBuffer[float_dtype]:
    """Compute an inverse-root factor S with S S^T ~= (K + sigma^2 I)^-1.

    This is the quantity LOVE needs for predictive variance:
    var(x*) ~= k(x*, x*) - ||K(x*, X_train) @ S||^2.

    LOVE variance needs an inverse-root factor for the train covariance, not raw
    CG probe solves. Using the wrong operator can collapse predictive variances
    to the clamp floor.
    """
    var n = provider.get_n()
    var ctx = provider.get_ctx()
    var r = lanczos_iter

    var alpha = List[Float32](capacity=r)
    var beta = List[Float32](capacity=max(r - 1, 0))
    for _ in range(r):
        alpha.append(Float32(0.0))
    for _ in range(max(r - 1, 0)):
        beta.append(Float32(0.0))

    var Q_host = ctx.enqueue_create_host_buffer[float_dtype](n * r)
    var z_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    for i in range(n):
        var state = UInt64(0xC0FFEE1234567890) ^ (UInt64(i) * UInt64(0x9E3779B97F4A7C15)) ^ (UInt64(r) * UInt64(0xBF58476D1CE4E5B9))
        state ^= state >> 12
        state ^= state << 25
        state ^= state >> 27
        state *= UInt64(0x2545F4914F6CDD1D)
        z_host[i] = Float32(1.0) if (state & UInt64(0x20000)) != UInt64(0) else Float32(-1.0)

    var z_norm_sq = Float32(0.0)
    for i in range(n):
        z_norm_sq += z_host[i] * z_host[i]
    var z_norm = sqrt(z_norm_sq)
    for i in range(n):
        z_host[i] /= z_norm
        Q_host[i] = z_host[i]

    var v_curr_device = ctx.enqueue_create_buffer[float_dtype](n)
    var v_prev_device = ctx.enqueue_create_buffer[float_dtype](n)
    var w_device = ctx.enqueue_create_buffer[float_dtype](n)

    var v_curr_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var v_prev_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var w_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    for i in range(n):
        v_curr_host[i] = z_host[i]
        v_prev_host[i] = Float32(0.0)

    ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
    ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
    ctx.synchronize()

    var beta_prev = Float32(0.0)
    alias LANCZOS_BREAKDOWN_TOL = Float32(1e-10)

    for j in range(r):
        provider.forward_matvec(w_device.unsafe_ptr(), v_curr_device.unsafe_ptr(), 1)
        ctx.enqueue_copy(dst_buf=w_host, src_buf=w_device)
        ctx.synchronize()

        var alpha_j = Float32(0.0)
        for i in range(n):
            alpha_j += v_curr_host[i] * w_host[i]
        alpha[j] = alpha_j

        for i in range(n):
            w_host[i] -= alpha_j * v_curr_host[i]
        if j > 0:
            for i in range(n):
                w_host[i] -= beta_prev * v_prev_host[i]

        # Full reorthogonalization keeps the Lanczos basis stable enough for
        # inverse-root prediction. Without this, the JIT LOVE path drifts back
        # toward prior variance because the Krylov basis loses orthogonality.
        for _ in range(2):
            var max_inner = Float32(0.0)
            for k in range(j + 1):
                var inner = Float32(0.0)
                for i in range(n):
                    inner += Q_host[k * n + i] * w_host[i]
                var inner_mag = inner
                if inner_mag < Float32(0.0):
                    inner_mag = -inner_mag
                if inner_mag > max_inner:
                    max_inner = inner_mag
                for i in range(n):
                    w_host[i] -= inner * Q_host[k * n + i]
            if max_inner < Float32(1e-5):
                break

        var beta_j_sq = Float32(0.0)
        for i in range(n):
            beta_j_sq += w_host[i] * w_host[i]
        var beta_j = sqrt(beta_j_sq)

        if j < r - 1:
            beta[j] = beta_j

        if beta_j < LANCZOS_BREAKDOWN_TOL:
            for k in range(j + 1, r):
                for i in range(n):
                    Q_host[k * n + i] = Float32(0.0)
            break

        for i in range(n):
            v_prev_host[i] = v_curr_host[i]
            v_curr_host[i] = w_host[i] / beta_j

        if j + 1 < r:
            for i in range(n):
                Q_host[(j + 1) * n + i] = v_curr_host[i]

        ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
        ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
        ctx.synchronize()
        beta_prev = beta_j

    var eigenvalues = List[Float32](capacity=r)
    var eigenvectors = List[Float32](capacity=r * r)
    for _ in range(r):
        eigenvalues.append(Float32(0.0))
    for _ in range(r * r):
        eigenvectors.append(Float32(0.0))

    tridiagonal_eigh_native(
        alpha.unsafe_ptr(), beta.unsafe_ptr(), r,
        eigenvalues.unsafe_ptr(), eigenvectors.unsafe_ptr(),
    )

    var max_eig = Float32(1e-10)
    for i in range(r):
        if eigenvalues[i] > max_eig:
            max_eig = eigenvalues[i]
    var eig_clamp = max_eig * Float32(1e-6)

    var V_Linv_sqrt = List[Float32](capacity=r * r)
    for _ in range(r * r):
        V_Linv_sqrt.append(Float32(0.0))

    for i in range(r):
        for j in range(r):
            var lambda_j = eigenvalues[j]
            if lambda_j < eig_clamp:
                lambda_j = eig_clamp
            var lambda_inv_sqrt = Float32(1.0) / sqrt(lambda_j)
            V_Linv_sqrt[i * r + j] = eigenvectors[i * r + j] * lambda_inv_sqrt

    var T_inv_sqrt = List[Float32](capacity=r * r)
    for _ in range(r * r):
        T_inv_sqrt.append(Float32(0.0))

    for i in range(r):
        for j in range(r):
            var sum_val = Float32(0.0)
            for k in range(r):
                sum_val += V_Linv_sqrt[i * r + k] * eigenvectors[j * r + k]
            T_inv_sqrt[i * r + j] = sum_val

    var T_inv_sqrt_host = ctx.enqueue_create_host_buffer[float_dtype](r * r)
    for i in range(r * r):
        T_inv_sqrt_host[i] = T_inv_sqrt[i]

    var Q_device = ctx.enqueue_create_buffer[float_dtype](n * r)
    var T_inv_sqrt_device = ctx.enqueue_create_buffer[float_dtype](r * r)
    var S_device = ctx.enqueue_create_buffer[float_dtype](n * r)
    ctx.enqueue_copy(Q_device, Q_host)
    ctx.enqueue_copy(T_inv_sqrt_device, T_inv_sqrt_host)
    ctx.synchronize()

    alias ROOT_BLOCK = 256
    var root_total = n * r
    ctx.enqueue_function[kernel_form_lanczos_inv_root[ROOT_BLOCK]](
        S_device.unsafe_ptr(),
        Q_device.unsafe_ptr(),
        T_inv_sqrt_device.unsafe_ptr(),
        n,
        r,
        root_total,
        grid_dim=((root_total + ROOT_BLOCK - 1) // ROOT_BLOCK,),
        block_dim=(ROOT_BLOCK,),
    )

    var S_host = ctx.enqueue_create_host_buffer[float_dtype](n * r)
    ctx.enqueue_copy(S_host, S_device)
    ctx.synchronize()

    _ = v_curr_device
    _ = v_prev_device
    _ = w_device
    _ = Q_device
    _ = T_inv_sqrt_device
    _ = S_device
    _ = T_inv_sqrt_host
    _ = v_curr_host
    _ = v_prev_host
    _ = w_host
    _ = z_host
    _ = Q_host
    return S_host


fn predict_variance_love_jit(
    provider: ErasedJITProvider,
    diag_provider: ErasedJITProvider,
    use_diag_provider: Bool,
    ctx: DeviceContext,
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    max_cg_iter: Int,
    cg_tol: Float32,
    precond_rank: Int,
    lanczos_rank: Int,
    love_alloc_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_cross_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_diag_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_post_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_cross_strategy_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_cross_chunk_width_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
) raises -> DeviceBuffer[float_dtype]:
    """Compute LOVE variance prediction.

    Returns variance as DeviceBuffer [n_test].
    """
    var rank = lanczos_rank

    # Step 1: Compute the inverse-root factor S ~= (K + sigma^2 I)^-1/2.
    var inv_root_host = compute_lanczos_inv_root_jit(provider, rank)
    var inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * rank)
    with ProfileBlock[PROFILING]("JIT_predict_love_copy_root"):
        ctx.enqueue_copy(inv_root_device, inv_root_host)
        ctx.synchronize()

    var var_device = predict_variance_love_from_inv_root_jit(
        provider,
        diag_provider,
        use_diag_provider,
        ctx,
        x_test_device_ptr,
        n_train,
        n_test,
        inv_root_device.unsafe_ptr(),
        rank,
        love_alloc_time_ns_out,
        love_cross_time_ns_out,
        love_diag_time_ns_out,
        love_post_time_ns_out,
        love_cross_strategy_out,
        love_cross_chunk_width_out,
    )

    _ = inv_root_host
    _ = inv_root_device

    return var_device^


fn predict_variance_love_from_inv_root_jit(
    provider: ErasedJITProvider,
    diag_provider: ErasedJITProvider,
    use_diag_provider: Bool,
    ctx: DeviceContext,
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    inv_root_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    rank: Int,
    love_alloc_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_cross_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_diag_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_post_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_cross_strategy_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    love_cross_chunk_width_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
) raises -> DeviceBuffer[float_dtype]:
    """Compute LOVE variance from a precomputed inverse-root factor."""
    # Step 2: V = K(X_test, X_train) @ S → [rank × n_test]
    var love_alloc_time_ns = Int(0)
    var love_cross_time_ns = Int(0)
    var love_diag_time_ns = Int(0)
    var love_post_time_ns = Int(0)
    var love_cross_strategy = love_cross_strategy_from_env()
    var love_cross_chunk_width = love_cross_chunk_width_from_env(rank)
    if love_reduced_sync_enabled():
        love_cross_chunk_width = rank

    var alloc_start = perf_counter_ns()
    var V = ctx.enqueue_create_buffer[float_dtype](rank * n_test)
    ctx.synchronize()
    love_alloc_time_ns += Int(perf_counter_ns() - alloc_start)

    var cross_start = perf_counter_ns()
    if love_cross_strategy == LOVE_CROSS_STRATEGY_MATERIALIZE_BLAS:
        var used_blas = love_cross_root_materialize_blas_jit(
            provider,
            ctx,
            V.unsafe_ptr(),
            x_test_device_ptr,
            inv_root_device_ptr,
            n_train,
            n_test,
            rank,
        )
        if not used_blas:
            love_cross_strategy = LOVE_CROSS_STRATEGY_FUSED
    if love_cross_strategy == LOVE_CROSS_STRATEGY_FUSED:
        cross_matvec_specialized_chunks_jit(
            provider,
            V.unsafe_ptr(),
            x_test_device_ptr,
            inv_root_device_ptr,
            n_train,
            n_test,
            rank,
            love_cross_chunk_width,
        )
    love_cross_time_ns += Int(perf_counter_ns() - cross_start)
    
    # Step 3: diag_test[j] = k(x*_j, x*_j)
    alloc_start = perf_counter_ns()
    var diag_test = ctx.enqueue_create_buffer[float_dtype](n_test)
    ctx.synchronize()
    love_alloc_time_ns += Int(perf_counter_ns() - alloc_start)
    var diag_start = perf_counter_ns()
    populate_test_diagonal_jit(
        provider,
        diag_provider,
        use_diag_provider,
        ctx,
        diag_test.unsafe_ptr(),
        x_test_device_ptr,
        n_test,
    )
    love_diag_time_ns += Int(perf_counter_ns() - diag_start)
    
    # Step 4: var[j] = diag_test[j] - ||V[:,j]||^2
    alias BLOCK = 256
    var grid = ceildiv(n_test, BLOCK)
    alloc_start = perf_counter_ns()
    var var_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    love_alloc_time_ns += Int(perf_counter_ns() - alloc_start)
    var post_start = perf_counter_ns()
    ctx.enqueue_function[kernel_love_variance[BLOCK]](
        var_device.unsafe_ptr(), V.unsafe_ptr(), diag_test.unsafe_ptr(),
        n_test, rank,
        grid_dim=grid, block_dim=BLOCK,
    )
    ctx.synchronize()
    love_post_time_ns += Int(perf_counter_ns() - post_start)

    if Int(love_alloc_time_ns_out) != 0:
        love_alloc_time_ns_out[] = love_alloc_time_ns
    if Int(love_cross_time_ns_out) != 0:
        love_cross_time_ns_out[] = love_cross_time_ns
    if Int(love_diag_time_ns_out) != 0:
        love_diag_time_ns_out[] = love_diag_time_ns
    if Int(love_post_time_ns_out) != 0:
        love_post_time_ns_out[] = love_post_time_ns
    if Int(love_cross_strategy_out) != 0:
        love_cross_strategy_out[] = love_cross_strategy
    if Int(love_cross_chunk_width_out) != 0:
        love_cross_chunk_width_out[] = love_cross_chunk_width

    _ = V
    _ = diag_test

    return var_device^


# =============================================================================
# Exact Variance (CG-based)
# =============================================================================


alias PREDICT_EXACT_CROSS_DIRECT_FILL = 1
alias PREDICT_EXACT_CROSS_CHUNKED_FALLBACK = 2
alias PREDICT_EXACT_MATRIX_FREE_MAX_BLOCK_COLS = 512


fn choose_exact_prediction_block_cols(
    n_test: Int,
    prefer_materialized_multicol: Bool = False,
    exact_block_cols_override: Int = 0,
) -> Int:
    """Choose an exact-prediction block width.

    Exact prediction solves A @ V = K(X, X*) for RHS columns from the full
    train-test covariance. GPyTorch runs this as one multi-RHS solve in the
    non-LOVE exact covariance path. Materialized MojoGP follows that behavior
    because the route already accepts O(n^2) storage and uses dense BLAS for
    forward_matvec. Matrix-free exact keeps a fixed cap to preserve the
    O(n * m_block) workspace contract.
    """
    if n_test <= 0:
        return 1
    if exact_block_cols_override > 0:
        if exact_block_cols_override < n_test:
            return exact_block_cols_override
        return n_test
    if prefer_materialized_multicol:
        return n_test
    if n_test <= PREDICT_EXACT_MATRIX_FREE_MAX_BLOCK_COLS:
        return n_test
    return PREDICT_EXACT_MATRIX_FREE_MAX_BLOCK_COLS


fn fill_exact_cross_block_jit(
    provider: ErasedJITProvider,
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_block_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    block_cols: Int,
) raises -> Int:
    """Fill one `[n_train x block_cols]` exact-prediction cross block.

    Returns a small enum describing which cross-fill path was used so the Python
    wrapper can surface whether exact prediction stayed on the direct fill path
    or needed the chunked bounded fallback.
    """
    if provider.has_fill_cross_covariance():
        with ProfileBlock[PROFILING]("JIT_predict_exact_cross"):
            provider.fill_cross_covariance(
                out_ptr,
                x_test_block_ptr,
                block_cols,
            )
        return PREDICT_EXACT_CROSS_DIRECT_FILL

    build_cross_covariance_from_cross_matvec_fallback_jit(
        provider,
        ctx,
        out_ptr,
        x_test_block_ptr,
        n_train,
        block_cols,
    )
    return PREDICT_EXACT_CROSS_CHUNKED_FALLBACK


fn solve_exact_cross_block_jit[Q: Preconditioner](
    provider: ErasedJITProvider,
    cross_block_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    active_cols: Int,
    x_dim: Int,
    max_cg_iter: Int,
    cg_tol: Float32,
    precond: Q,
    mut pool: CGBufferPool,
    use_preconditioner: Bool,
) raises -> CGResultWithTridiag:
    var ctx = provider.get_ctx()
    var rhs_cg = pool.rhs_cg
    var rhs_norms = pool.rhs_norms

    ctx.enqueue_function[kernel_copy](
        rhs_cg.unsafe_ptr(), cross_block_ptr, n_train * active_cols,
        grid_dim=((n_train * active_cols + 255) // 256,), block_dim=(256,)
    )
    normalize_columns_gpu(ctx, rhs_cg, rhs_norms, n_train, active_cols, sync=True)

    var use_blocked_blas = exact_blocked_blas_enabled()
    if not provider.has_fill_cross_covariance():
        use_blocked_blas = False
    if active_cols < exact_blocked_blas_min_cols():
        use_blocked_blas = False

    if use_blocked_blas:
        var wide_provider = ExactPredictionBlockedBLASProvider(
            provider.copy(),
            ctx,
            n_train,
            x_dim,
            exact_blocked_blas_tile_cols(n_train),
        )
        var result = batched_cg_unified(
            wide_provider,
            rhs_cg.unsafe_ptr(),
            n_train,
            active_cols,
            max_cg_iter,
            0,
            cg_tol,
            precond,
            pool,
            use_warm_start=False,
            use_preconditioner=use_preconditioner,
        )
        scale_columns_by_norms(ctx, result.solution, rhs_norms, n_train, active_cols, sync=True)

        _ = rhs_cg
        _ = rhs_norms
        _ = wide_provider
        return result.copy()

    var result = batched_cg_unified(
        provider,
        rhs_cg.unsafe_ptr(),
        n_train,
        active_cols,
        max_cg_iter,
        0,
        cg_tol,
        precond,
        pool,
        use_warm_start=False,
        use_preconditioner=use_preconditioner,
    )
    scale_columns_by_norms(ctx, result.solution, rhs_norms, n_train, active_cols, sync=True)

    _ = rhs_cg
    _ = rhs_norms
    return result.copy()


fn run_exact_variance_blocks_jit[Q: Preconditioner](
    provider: ErasedJITProvider,
    diag_provider: ErasedJITProvider,
    use_diag_provider: Bool,
    ctx: DeviceContext,
    var_device: DeviceBuffer[float_dtype],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    x_test_dim: Int,
    block_cols: Int,
    max_cg_iter: Int,
    cg_tol: Float32,
    precond: Q,
    mut pool: CGBufferPool,
    use_preconditioner: Bool,
    exact_cross_mode_out: UnsafePointer[Int, MutAnyOrigin],
    exact_cg_block_count_out: UnsafePointer[Int, MutAnyOrigin],
    exact_cg_total_iterations_out: UnsafePointer[Int, MutAnyOrigin],
    exact_cg_max_iterations_out: UnsafePointer[Int, MutAnyOrigin],
    exact_alloc_time_ns_out: UnsafePointer[Int, MutAnyOrigin],
    exact_cross_time_ns_out: UnsafePointer[Int, MutAnyOrigin],
    exact_diag_time_ns_out: UnsafePointer[Int, MutAnyOrigin],
    exact_solve_time_ns_out: UnsafePointer[Int, MutAnyOrigin],
    exact_post_time_ns_out: UnsafePointer[Int, MutAnyOrigin],
) raises:
    alias VAR_BLOCK = 256
    var observed_cross_mode = Int(0)
    var exact_cg_block_count = Int(0)
    var exact_cg_total_iterations = Int(0)
    var exact_cg_max_iterations = Int(0)
    var exact_alloc_time_ns = Int(0)
    var exact_cross_time_ns = Int(0)
    var exact_diag_time_ns = Int(0)
    var exact_solve_time_ns = Int(0)
    var exact_post_time_ns = Int(0)
    var test_start = 0
    while test_start < n_test:
        var remaining = n_test - test_start
        var active_cols = block_cols
        if remaining < active_cols:
            active_cols = remaining

        var x_test_block_ptr = x_test_device_ptr.offset(test_start * x_test_dim)
        var alloc_start = perf_counter_ns()
        var K_cross_block = ctx.enqueue_create_buffer[float_dtype](n_train * active_cols)
        var diag_test_block = ctx.enqueue_create_buffer[float_dtype](active_cols)
        exact_alloc_time_ns += Int(perf_counter_ns() - alloc_start)

        var cross_start = perf_counter_ns()
        var cross_mode = fill_exact_cross_block_jit(
            provider,
            ctx,
            K_cross_block.unsafe_ptr(),
            x_test_block_ptr,
            n_train,
            active_cols,
        )
        exact_cross_time_ns += Int(perf_counter_ns() - cross_start)
        if observed_cross_mode == 0:
            observed_cross_mode = cross_mode

        var diag_start = perf_counter_ns()
        with ProfileBlock[PROFILING]("JIT_predict_exact_diag"):
            populate_test_diagonal_jit(
                provider,
                diag_provider,
                use_diag_provider,
                ctx,
                diag_test_block.unsafe_ptr(),
                x_test_block_ptr,
                active_cols,
            )
        exact_diag_time_ns += Int(perf_counter_ns() - diag_start)

        var solve_start = perf_counter_ns()
        var cg_result = solve_exact_cross_block_jit(
            provider,
            K_cross_block.unsafe_ptr(),
            n_train,
            active_cols,
            x_test_dim,
            max_cg_iter,
            cg_tol,
            precond,
            pool,
            use_preconditioner,
        )
        exact_solve_time_ns += Int(perf_counter_ns() - solve_start)

        exact_cg_block_count += 1
        exact_cg_total_iterations += cg_result.num_iterations
        if cg_result.num_iterations > exact_cg_max_iterations:
            exact_cg_max_iterations = cg_result.num_iterations

        var post_start = perf_counter_ns()
        with ProfileBlock[PROFILING]("JIT_predict_exact_post"):
            ctx.enqueue_function[kernel_exact_variance[VAR_BLOCK]](
                var_device.unsafe_ptr().offset(test_start),
                K_cross_block.unsafe_ptr(),
                cg_result.solution.unsafe_ptr(),
                diag_test_block.unsafe_ptr(),
                n_train,
                active_cols,
                grid_dim=((active_cols + VAR_BLOCK - 1) // VAR_BLOCK,),
                block_dim=(VAR_BLOCK,),
            )
            ctx.synchronize()
        exact_post_time_ns += Int(perf_counter_ns() - post_start)

        _ = K_cross_block
        _ = diag_test_block
        _ = cg_result
        test_start += active_cols

    exact_cross_mode_out[] = observed_cross_mode
    exact_cg_block_count_out[] = exact_cg_block_count
    exact_cg_total_iterations_out[] = exact_cg_total_iterations
    exact_cg_max_iterations_out[] = exact_cg_max_iterations
    exact_alloc_time_ns_out[] = exact_alloc_time_ns
    exact_cross_time_ns_out[] = exact_cross_time_ns
    exact_diag_time_ns_out[] = exact_diag_time_ns
    exact_solve_time_ns_out[] = exact_solve_time_ns
    exact_post_time_ns_out[] = exact_post_time_ns


fn predict_variance_exact_jit(
    provider: ErasedJITProvider,
    diag_provider: ErasedJITProvider,
    use_diag_provider: Bool,
    ctx: DeviceContext,
    alpha_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    x_test_dim: Int,
    max_cg_iter: Int,
    cg_tol: Float32,
    precond_rank: Int,
    use_preconditioner: Bool,
    prefer_materialized_multicol: Bool,
    exact_block_cols_override: Int,
    exact_block_cols_out: UnsafePointer[Int, MutAnyOrigin],
    exact_cross_mode_out: UnsafePointer[Int, MutAnyOrigin],
    exact_cg_block_count_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    exact_cg_total_iterations_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    exact_cg_max_iterations_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    exact_alloc_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    exact_cross_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    exact_diag_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    exact_solve_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
    exact_post_time_ns_out: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](),
) raises -> DeviceBuffer[float_dtype]:
    """Compute exact variance via CG.
    
    Steps:
    1. Process active test blocks of width `m_block`
    2. Build K_cross_block [n_train × m_block] directly or via chunked fallback
    3. Solve (K + σ²I) @ V_block = K_cross_block via batched CG
    4. diag_test_block[j] = k(x*_j, x*_j)
    5. var_block[j] = diag_test_block[j] - dot(K_cross_block[:,j], V_block[:,j])
    
    Returns:
        variance device buffer [n_test]
        exact block width used
        cross-fill mode enum
    """
    var var_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    var block_cols = choose_exact_prediction_block_cols(
        n_test,
        prefer_materialized_multicol,
        exact_block_cols_override,
    )
    var pool = CGBufferPool(ctx, n_train, block_cols, 1, 1)
    var use_exact_preconditioner = use_preconditioner and precond_rank > 0

    var exact_cross_mode_slot = alloc[Int](1)
    exact_cross_mode_slot[] = 0
    var exact_cg_block_count_slot = alloc[Int](1)
    var exact_cg_total_iterations_slot = alloc[Int](1)
    var exact_cg_max_iterations_slot = alloc[Int](1)
    var exact_alloc_time_ns_slot = alloc[Int](1)
    var exact_cross_time_ns_slot = alloc[Int](1)
    var exact_diag_time_ns_slot = alloc[Int](1)
    var exact_solve_time_ns_slot = alloc[Int](1)
    var exact_post_time_ns_slot = alloc[Int](1)
    exact_cg_block_count_slot[] = 0
    exact_cg_total_iterations_slot[] = 0
    exact_cg_max_iterations_slot[] = 0
    exact_alloc_time_ns_slot[] = 0
    exact_cross_time_ns_slot[] = 0
    exact_diag_time_ns_slot[] = 0
    exact_solve_time_ns_slot[] = 0
    exact_post_time_ns_slot[] = 0

    if use_exact_preconditioner:
        var precond = build_pivoted_cholesky_precond_unified(
            provider,
            precond_rank,
            max_num_cols=block_cols,
            noise_mode=provider.get_noise_mode(),
            noise_vec_ptr=provider.get_noise_vector_ptr(),
        )
        run_exact_variance_blocks_jit(
            provider,
            diag_provider,
            use_diag_provider,
            ctx,
            var_device,
            x_test_device_ptr,
            n_train,
            n_test,
            x_test_dim,
            block_cols,
            max_cg_iter,
            cg_tol,
            precond,
            pool,
            use_preconditioner=True,
            exact_cross_mode_out=exact_cross_mode_slot,
            exact_cg_block_count_out=exact_cg_block_count_slot,
            exact_cg_total_iterations_out=exact_cg_total_iterations_slot,
            exact_cg_max_iterations_out=exact_cg_max_iterations_slot,
            exact_alloc_time_ns_out=exact_alloc_time_ns_slot,
            exact_cross_time_ns_out=exact_cross_time_ns_slot,
            exact_diag_time_ns_out=exact_diag_time_ns_slot,
            exact_solve_time_ns_out=exact_solve_time_ns_slot,
            exact_post_time_ns_out=exact_post_time_ns_slot,
        )
        _ = precond
    else:
        var identity_precond = IdentityPreconditioner(n_train)
        run_exact_variance_blocks_jit(
            provider,
            diag_provider,
            use_diag_provider,
            ctx,
            var_device,
            x_test_device_ptr,
            n_train,
            n_test,
            x_test_dim,
            block_cols,
            max_cg_iter,
            cg_tol,
            identity_precond,
            pool,
            use_preconditioner=False,
            exact_cross_mode_out=exact_cross_mode_slot,
            exact_cg_block_count_out=exact_cg_block_count_slot,
            exact_cg_total_iterations_out=exact_cg_total_iterations_slot,
            exact_cg_max_iterations_out=exact_cg_max_iterations_slot,
            exact_alloc_time_ns_out=exact_alloc_time_ns_slot,
            exact_cross_time_ns_out=exact_cross_time_ns_slot,
            exact_diag_time_ns_out=exact_diag_time_ns_slot,
            exact_solve_time_ns_out=exact_solve_time_ns_slot,
            exact_post_time_ns_out=exact_post_time_ns_slot,
        )
        _ = identity_precond

    _ = alpha_device_ptr
    _ = pool
    
    exact_block_cols_out[] = block_cols
    exact_cross_mode_out[] = exact_cross_mode_slot[]
    if Int(exact_cg_block_count_out) != 0:
        exact_cg_block_count_out[] = exact_cg_block_count_slot[]
    if Int(exact_cg_total_iterations_out) != 0:
        exact_cg_total_iterations_out[] = exact_cg_total_iterations_slot[]
    if Int(exact_cg_max_iterations_out) != 0:
        exact_cg_max_iterations_out[] = exact_cg_max_iterations_slot[]
    if Int(exact_alloc_time_ns_out) != 0:
        exact_alloc_time_ns_out[] = exact_alloc_time_ns_slot[]
    if Int(exact_cross_time_ns_out) != 0:
        exact_cross_time_ns_out[] = exact_cross_time_ns_slot[]
    if Int(exact_diag_time_ns_out) != 0:
        exact_diag_time_ns_out[] = exact_diag_time_ns_slot[]
    if Int(exact_solve_time_ns_out) != 0:
        exact_solve_time_ns_out[] = exact_solve_time_ns_slot[]
    if Int(exact_post_time_ns_out) != 0:
        exact_post_time_ns_out[] = exact_post_time_ns_slot[]
    exact_cross_mode_slot.free()
    exact_cg_block_count_slot.free()
    exact_cg_total_iterations_slot.free()
    exact_cg_max_iterations_slot.free()
    exact_alloc_time_ns_slot.free()
    exact_cross_time_ns_slot.free()
    exact_diag_time_ns_slot.free()
    exact_solve_time_ns_slot.free()
    exact_post_time_ns_slot.free()
    return var_device^


# =============================================================================
# Main Prediction Entry Point
# =============================================================================

alias PREDICT_MEAN_ONLY = 0
alias PREDICT_LOVE = 1
alias PREDICT_EXACT = 2


fn predict_jit(
    provider: ErasedJITProvider,
    diag_provider: ErasedJITProvider,
    ctx: DeviceContext,
    y_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    x_test_dim: Int,
    noise: Float32,
    mean: Float32,
    variance_method: Int,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = Float32(1e-2),
    precond_rank: Int = 10,
    lanczos_rank: Int = 20,
    use_preconditioner: Bool = True,
    use_diag_provider: Bool = False,
    prefer_materialized_multicol: Bool = False,
    exact_block_cols_override: Int = 0,
) raises -> PredictionResultJIT:
    """Main JIT prediction entry point."""
    var total_start = Int(perf_counter_ns())

    var y_centered = center_targets_jit(ctx, y_device_ptr, n_train, mean)
    var alpha_start = perf_counter_ns()
    var alpha_device = compute_alpha_jit(
        provider, ctx, y_centered,
        n_train, max_cg_iter, cg_tol, precond_rank, use_preconditioner,
    )
    var alpha_time_ns = Int(perf_counter_ns() - alpha_start)

    var result = predict_from_alpha_jit(
        provider,
        diag_provider,
        ctx,
        alpha_device.unsafe_ptr(),
        x_test_device_ptr,
        n_train,
        n_test,
        x_test_dim,
        noise,
        mean,
        variance_method,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        lanczos_rank=lanczos_rank,
        use_preconditioner=use_preconditioner,
        use_diag_provider=use_diag_provider,
        prefer_materialized_multicol=prefer_materialized_multicol,
        exact_block_cols_override=exact_block_cols_override,
        alpha_time_ns=alpha_time_ns,
        total_start_ns=total_start,
    )

    _ = y_centered
    _ = alpha_device
    return result^


fn predict_from_alpha_jit(
    provider: ErasedJITProvider,
    diag_provider: ErasedJITProvider,
    ctx: DeviceContext,
    alpha_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    x_test_dim: Int,
    noise: Float32,
    mean: Float32,
    variance_method: Int,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = Float32(1e-2),
    precond_rank: Int = 10,
    lanczos_rank: Int = 20,
    use_preconditioner: Bool = True,
    use_diag_provider: Bool = False,
    prefer_materialized_multicol: Bool = False,
    exact_block_cols_override: Int = 0,
    cached_inv_root_device_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    cached_inv_root_rank: Int = 0,
    use_cached_inv_root: Bool = False,
    alpha_time_ns: Int = 0,
    total_start_ns: Int = 0,
) raises -> PredictionResultJIT:
    """Run prediction assuming alpha has already been prepared."""
    var effective_total_start = total_start_ns
    if effective_total_start <= 0:
        effective_total_start = Int(perf_counter_ns())
    var mean_start = perf_counter_ns()
    var mean_device = predict_mean_jit(
        provider, ctx, alpha_device_ptr,
        x_test_device_ptr, n_train, n_test, mean,
    )
    var mean_time_ns = Int(perf_counter_ns() - mean_start)
    
    # Step 3: Variance prediction (optional)
    var has_variance = (
        variance_method != PREDICT_MEAN_ONLY
    )
    var var_device: DeviceBuffer[float_dtype]
    var variance_time_ns = Int(0)
    var exact_block_cols = Int(0)
    var exact_cross_mode = Int(0)
    var exact_cg_block_count = Int(0)
    var exact_cg_total_iterations = Int(0)
    var exact_cg_max_iterations = Int(0)
    var exact_alloc_time_ns = Int(0)
    var exact_cross_time_ns = Int(0)
    var exact_diag_time_ns = Int(0)
    var exact_solve_time_ns = Int(0)
    var exact_post_time_ns = Int(0)
    var love_alloc_time_ns = Int(0)
    var love_cross_time_ns = Int(0)
    var love_diag_time_ns = Int(0)
    var love_post_time_ns = Int(0)
    var love_cross_strategy = Int(0)
    var love_cross_chunk_width = Int(0)
    var love_root_cache_used = False

    if variance_method == PREDICT_LOVE:
        var variance_start = perf_counter_ns()
        var love_alloc_time_ns_slot = alloc[Int](1)
        var love_cross_time_ns_slot = alloc[Int](1)
        var love_diag_time_ns_slot = alloc[Int](1)
        var love_post_time_ns_slot = alloc[Int](1)
        var love_cross_strategy_slot = alloc[Int](1)
        var love_cross_chunk_width_slot = alloc[Int](1)
        love_alloc_time_ns_slot[] = 0
        love_cross_time_ns_slot[] = 0
        love_diag_time_ns_slot[] = 0
        love_post_time_ns_slot[] = 0
        love_cross_strategy_slot[] = 0
        love_cross_chunk_width_slot[] = 0
        if cached_inv_root_rank > 0:
            var_device = predict_variance_love_from_inv_root_jit(
                provider,
                diag_provider,
                use_diag_provider,
                ctx,
                x_test_device_ptr,
                n_train,
                n_test,
                cached_inv_root_device_ptr,
                cached_inv_root_rank,
                love_alloc_time_ns_slot,
                love_cross_time_ns_slot,
                love_diag_time_ns_slot,
                love_post_time_ns_slot,
                love_cross_strategy_slot,
                love_cross_chunk_width_slot,
            )
            love_root_cache_used = use_cached_inv_root
        else:
            var_device = predict_variance_love_jit(
                provider, diag_provider, use_diag_provider, ctx, x_test_device_ptr,
                n_train, n_test, max_cg_iter, cg_tol, precond_rank, lanczos_rank,
                love_alloc_time_ns_slot,
                love_cross_time_ns_slot,
                love_diag_time_ns_slot,
                love_post_time_ns_slot,
                love_cross_strategy_slot,
                love_cross_chunk_width_slot,
            )
        love_alloc_time_ns = love_alloc_time_ns_slot[]
        love_cross_time_ns = love_cross_time_ns_slot[]
        love_diag_time_ns = love_diag_time_ns_slot[]
        love_post_time_ns = love_post_time_ns_slot[]
        love_cross_strategy = love_cross_strategy_slot[]
        love_cross_chunk_width = love_cross_chunk_width_slot[]
        love_alloc_time_ns_slot.free()
        love_cross_time_ns_slot.free()
        love_diag_time_ns_slot.free()
        love_post_time_ns_slot.free()
        love_cross_strategy_slot.free()
        love_cross_chunk_width_slot.free()
        variance_time_ns = Int(perf_counter_ns() - variance_start)
    elif variance_method == PREDICT_EXACT:
        var variance_start = perf_counter_ns()
        var exact_block_cols_slot = alloc[Int](1)
        var exact_cross_mode_slot = alloc[Int](1)
        var exact_cg_block_count_slot = alloc[Int](1)
        var exact_cg_total_iterations_slot = alloc[Int](1)
        var exact_cg_max_iterations_slot = alloc[Int](1)
        var exact_alloc_time_ns_slot = alloc[Int](1)
        var exact_cross_time_ns_slot = alloc[Int](1)
        var exact_diag_time_ns_slot = alloc[Int](1)
        var exact_solve_time_ns_slot = alloc[Int](1)
        var exact_post_time_ns_slot = alloc[Int](1)
        exact_block_cols_slot[] = 0
        exact_cross_mode_slot[] = 0
        exact_cg_block_count_slot[] = 0
        exact_cg_total_iterations_slot[] = 0
        exact_cg_max_iterations_slot[] = 0
        exact_alloc_time_ns_slot[] = 0
        exact_cross_time_ns_slot[] = 0
        exact_diag_time_ns_slot[] = 0
        exact_solve_time_ns_slot[] = 0
        exact_post_time_ns_slot[] = 0
        var_device = predict_variance_exact_jit(
            provider, diag_provider, use_diag_provider, ctx,
            alpha_device_ptr, x_test_device_ptr,
            n_train, n_test, x_test_dim, max_cg_iter, cg_tol, precond_rank,
            use_preconditioner,
            prefer_materialized_multicol,
            exact_block_cols_override,
            exact_block_cols_slot,
            exact_cross_mode_slot,
            exact_cg_block_count_slot,
            exact_cg_total_iterations_slot,
            exact_cg_max_iterations_slot,
            exact_alloc_time_ns_slot,
            exact_cross_time_ns_slot,
            exact_diag_time_ns_slot,
            exact_solve_time_ns_slot,
            exact_post_time_ns_slot,
        )
        exact_block_cols = exact_block_cols_slot[]
        exact_cross_mode = exact_cross_mode_slot[]
        exact_cg_block_count = exact_cg_block_count_slot[]
        exact_cg_total_iterations = exact_cg_total_iterations_slot[]
        exact_cg_max_iterations = exact_cg_max_iterations_slot[]
        exact_alloc_time_ns = exact_alloc_time_ns_slot[]
        exact_cross_time_ns = exact_cross_time_ns_slot[]
        exact_diag_time_ns = exact_diag_time_ns_slot[]
        exact_solve_time_ns = exact_solve_time_ns_slot[]
        exact_post_time_ns = exact_post_time_ns_slot[]
        exact_block_cols_slot.free()
        exact_cross_mode_slot.free()
        exact_cg_block_count_slot.free()
        exact_cg_total_iterations_slot.free()
        exact_cg_max_iterations_slot.free()
        exact_alloc_time_ns_slot.free()
        exact_cross_time_ns_slot.free()
        exact_diag_time_ns_slot.free()
        exact_solve_time_ns_slot.free()
        exact_post_time_ns_slot.free()
        variance_time_ns = Int(perf_counter_ns() - variance_start)
    else:
        var_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    
    # Copy results to host
    var mean_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    var var_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    with ProfileBlock[PROFILING]("JIT_predict_copy_back"):
        ctx.enqueue_copy(mean_host, mean_device)
        if has_variance:
            ctx.enqueue_copy(var_host, var_device)
        ctx.synchronize()

    if has_variance:
        with ProfileBlock[PROFILING]("JIT_predict_noise_add"):
            for i in range(n_test):
                var_host[i] += noise

    _ = alpha_device_ptr
    _ = mean_device
    _ = var_device

    var total_time_ns = Int(perf_counter_ns() - UInt(effective_total_start))
    return PredictionResultJIT(
        mean_host^,
        var_host^,
        n_test,
        has_variance,
        exact_block_cols,
        exact_cross_mode,
        alpha_time_ns,
        love_root_cache_used,
        mean_time_ns,
        variance_time_ns,
        total_time_ns,
        exact_cg_block_count,
        exact_cg_total_iterations,
        exact_cg_max_iterations,
        exact_alloc_time_ns,
        exact_cross_time_ns,
        exact_diag_time_ns,
        exact_solve_time_ns,
        exact_post_time_ns,
        love_alloc_time_ns,
        love_cross_time_ns,
        love_diag_time_ns,
        love_post_time_ns,
        love_cross_strategy,
        love_cross_chunk_width,
    )
