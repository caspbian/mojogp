"""Multi-output GP prediction using Kronecker decomposition.

This module implements prediction for multi-output GPs using the ICM model.
The predictive distribution is computed by combining predictions from T
independent sub-problems via the task covariance eigenvectors Q.

Key formulas:
- Mean: mu_t(x*) = sum_s Q[t,s] * s_s * k(x*, X)^T * alpha_tilde_s
- Variance: var_t(x*) = sum_s Q[t,s]^2 * var_rotated_s(x*)

where:
- Q is the T x T eigenvector matrix of B
- s_s = outputscale * lambda_s is the effective scale for sub-problem s
- alpha_tilde_s is the CG solution for sub-problem s in the rotated basis
- var_rotated_s(x*) is the single-output predictive variance for sub-problem s
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import sqrt

from .constants import float_dtype, PI, KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
from .matvec_provider import MaterializedProvider
from .cross_covariance_provider import cross_matvec_with_provider
from .love_provider import kernel_matmul_ST_K, kernel_variance_from_V, kernel_KcrossT_matmul_alpha, kernel_vnorm_sq_from_ST_Kcross
from .combined_inv_quad_logdet import batched_cg_unified, CGBufferPool
from .gradient_provider import IsotropicGradientAdapter, MaterializedCompositeGradientAdapter
from .pivoted_cholesky import PivotedCholeskyPrecond, build_pivoted_cholesky_precond_unified
from .love_provider import compute_cross_covariance_with_provider, compute_cross_covariance_device_with_provider
from .lanczos_with_provider import compute_lanczos_root_with_provider, LanczosRootResult
from .composable_kernel import ComposableKernel, ScaleKernel
from .composite_provider import CompositeProvider, MaterializedCompositeProvider
from .composite_prediction import (
    cg_solve_composite,
    predict_mean_composite,
    predict_variance_love_composite,
    compute_lanczos_root_composite,
    LanczosRootResultComposite,
)


# =============================================================================
# Result Structs
# =============================================================================

struct MultiOutputPredictionResult:
    """Result from multi-output GP prediction.
    
    Fields:
        mean: Predicted means [m x T], row-major (m test points, T tasks)
        variance: Predicted variances [m x T], row-major
        m: Number of test points
        num_tasks: Number of tasks T
        has_variance: Whether variance was computed
    """
    var mean: HostBuffer[float_dtype]
    var variance: HostBuffer[float_dtype]
    var m: Int
    var num_tasks: Int
    var has_variance: Bool
    
    fn __init__(
        out self,
        mean: HostBuffer[float_dtype],
        variance: HostBuffer[float_dtype],
        m: Int,
        num_tasks: Int,
        has_variance: Bool,
    ):
        self.mean = mean
        self.variance = variance
        self.m = m
        self.num_tasks = num_tasks
        self.has_variance = has_variance


# =============================================================================
# GPU Kernels
# =============================================================================

fn kernel_unrotate_predictions(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    rotated_ptr: UnsafePointer[Float32, MutAnyOrigin],
    Q_ptr: UnsafePointer[Float32, MutAnyOrigin],
    m: Int,
    T: Int,
) -> None:
    """Unrotate predictions from rotated basis to original task basis.
    
    out[i, t] = sum_s Q[t, s] * rotated[i, s]
    
    Args:
        out_ptr: Output [m x T], row-major
        rotated_ptr: Rotated predictions [m x T], row-major
        Q_ptr: Eigenvector matrix [T x T], row-major
        m: Number of test points
        T: Number of tasks
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = UInt(m * T)
    
    if idx >= total:
        return
    
    var i = Int(idx // UInt(T))  # Test point index
    var t = Int(idx % UInt(T))   # Task index
    
    var sum_val = Float32(0.0)
    for s in range(T):
        # Q[t, s] is at Q_ptr[t * T + s]
        # rotated[i, s] is at rotated_ptr[i * T + s]
        sum_val += Q_ptr[t * T + s] * rotated_ptr[i * T + s]
    
    out_ptr[idx] = sum_val


fn kernel_unrotate_variance(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    rotated_var_ptr: UnsafePointer[Float32, MutAnyOrigin],
    Q_ptr: UnsafePointer[Float32, MutAnyOrigin],
    m: Int,
    T: Int,
) -> None:
    """Unrotate variance predictions using Q^2 weighting.
    
    var_t(x*) = sum_s Q[t,s]^2 * var_rotated_s(x*)
    
    Args:
        out_ptr: Output variance [m x T], row-major
        rotated_var_ptr: Rotated variances [m x T], row-major
        Q_ptr: Eigenvector matrix [T x T], row-major
        m: Number of test points
        T: Number of tasks
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = UInt(m * T)
    
    if idx >= total:
        return
    
    var i = Int(idx // UInt(T))  # Test point index
    var t = Int(idx % UInt(T))   # Task index
    
    var sum_val = Float32(0.0)
    for s in range(T):
        var Q_ts = Q_ptr[t * T + s]
        sum_val += Q_ts * Q_ts * rotated_var_ptr[i * T + s]
    
    out_ptr[idx] = sum_val


fn kernel_write_column(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [m × T] row-major
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [m] source column
    col_idx: Int,     # which column to write
    m: Int,           # number of rows
    T: Int,           # total number of columns (stride)
) -> None:
    """Write src vector into column col_idx of a row-major [m × T] matrix on device.
    
    dst[i, col_idx] = src[i]  for i in 0..m-1
    
    Row-major layout: dst[i, col_idx] = dst_ptr[i * T + col_idx]
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(m):
        return
    dst_ptr[Int(idx) * T + col_idx] = src_ptr[idx]


fn kernel_scale_mean_by_effective_scale(
    mean_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scales_ptr: UnsafePointer[Float32, MutAnyOrigin],
    m: Int,
    T: Int,
) -> None:
    """Scale mean predictions by effective scale s_t.
    
    mean[i, t] *= scales[t]
    
    Args:
        mean_ptr: Mean predictions [m x T], row-major (modified in-place)
        scales_ptr: Effective scales [T]
        m: Number of test points
        T: Number of tasks
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = UInt(m * T)
    
    if idx >= total:
        return
    
    var t = Int(idx % UInt(T))  # Task index
    mean_ptr[idx] *= scales_ptr[t]


# =============================================================================
# Mean Prediction
# =============================================================================

fn predict_mean_multi_output(
    ctx: DeviceContext,
    x_train_device: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host: HostBuffer[float_dtype],
    alpha_rotated_host: HostBuffer[float_dtype],  # [n x T], row-major
    Q_host: HostBuffer[float_dtype],               # [T x T], row-major
    effective_scales_host: HostBuffer[float_dtype], # [T]
    n: Int,
    m: Int,
    dim: Int,
    num_tasks: Int,
    kernel_type: Int,
    lengthscale: Float32,
    outputscale: Float32,
    noise: Float32,
    use_ard: Bool = False,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
) raises -> HostBuffer[float_dtype]:
    """Predict mean for multi-output GP.
    
    For each test point x* and task t:
        mu_t(x*) = sum_s Q[t,s] * s_s * k(x*, X)^T * alpha_tilde_s
    
    Args:
        ctx: GPU device context
        x_train_device: Training data [n x dim] on device
        x_test_host: Test data [m x dim] on host
        alpha_rotated_host: CG solutions in rotated basis [n x T], row-major
        Q_host: Eigenvector matrix [T x T], row-major
        effective_scales_host: Effective scales s_t = outputscale * lambda_t [T]
        n: Number of training points
        m: Number of test points
        dim: Input dimension
        num_tasks: Number of tasks T
        kernel_type: Kernel type (0=RBF, 1=Matern32, 2=Matern52)
        lengthscale: Kernel lengthscale (ignored if use_ard=True)
        outputscale: Output scale (used for K_X computation)
        noise: Noise variance
        use_ard: Whether to use ARD (per-dimension lengthscales)
        lengthscales_ptr: Pointer to [dim] per-dimension lengthscales on host (required if use_ard=True)
        
    Returns:
        mean_host: Predicted means [m x T], row-major
    """
    var T = num_tasks
    
    # Copy test data to device
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](m * dim)
    ctx.enqueue_copy(dst_buf=x_test_device, src_buf=x_test_host)
    
    # Copy alpha_rotated to device
    var alpha_rotated_device = ctx.enqueue_create_buffer[float_dtype](n * T)
    ctx.enqueue_copy(dst_buf=alpha_rotated_device, src_buf=alpha_rotated_host)
    
    # Copy Q to device
    var Q_device = ctx.enqueue_create_buffer[float_dtype](T * T)
    ctx.enqueue_copy(dst_buf=Q_device, src_buf=Q_host)
    
    # Copy effective scales to device
    var scales_device = ctx.enqueue_create_buffer[float_dtype](T)
    ctx.enqueue_copy(dst_buf=scales_device, src_buf=effective_scales_host)
    
    ctx.synchronize()
    
    # Allocate output buffers
    var mean_rotated_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    var mean_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    
    # Create parameter buffer on device (for MaterializedProvider)
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = HostBuffer[float_dtype](ctx, 2)
    params_host_temp.unsafe_ptr()[0] = lengthscale
    params_host_temp.unsafe_ptr()[1] = Float32(1.0)  # outputscale=1 for K_X
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
    ctx.synchronize()
    
    # Create provider for cross-covariance computation (outputscale=1 for K_X)
    var provider = MaterializedProvider(
        ctx,
        x_train_device,
        params_device.unsafe_ptr(),
        n,
        dim,
        kernel_type,
        use_ard,
        lengthscale,
        Float32(1.0),  # outputscale=1 for K_X
        noise,
        Float32(0.0),
        Float32(0.0),
    )
    if use_ard:
        # Re-materialize with correct ARD lengthscales (constructor materialized
        # with uninitialized lengthscales_device buffer)
        provider.update_hyperparams_ard(lengthscales_ptr, Float32(1.0), noise)
    
    # For each sub-problem s, compute k(x*, X)^T @ alpha_tilde_s
    for s in range(T):
        # Extract alpha_tilde_s (column s of alpha_rotated)
        var alpha_s_device = ctx.enqueue_create_buffer[float_dtype](n)
        var alpha_s_host = HostBuffer[float_dtype](ctx, n)
        for i in range(n):
            alpha_s_host.unsafe_ptr()[i] = alpha_rotated_host.unsafe_ptr()[i * T + s]
        ctx.enqueue_copy(dst_buf=alpha_s_device, src_buf=alpha_s_host)
        ctx.synchronize()
        
        # Compute k(x*, X)^T @ alpha_s for all test points
        var mean_s_device = cross_matvec_with_provider(
            provider,
            x_test_device.unsafe_ptr(),
            alpha_s_device,
            m
        )
        
        # Write cross_result directly into column s of mean_rotated on device
        # (replaces D2H → update → H2D roundtrip with a single kernel launch)
        var write_block_size = 256
        var write_num_blocks = (m + write_block_size - 1) // write_block_size
        ctx.enqueue_function[kernel_write_column](
            mean_rotated_device.unsafe_ptr(),
            mean_s_device.unsafe_ptr(),
            s,
            m,
            T,
            grid_dim=(write_num_blocks,),
            block_dim=(write_block_size,),
        )
        ctx.synchronize()
    
    # Scale by effective scales: mean_rotated[i, s] *= s_s
    var block_size = 256
    var num_blocks = (m * T + block_size - 1) // block_size
    ctx.enqueue_function[kernel_scale_mean_by_effective_scale](
        mean_rotated_device.unsafe_ptr(),
        scales_device.unsafe_ptr(),
        m,
        T,
        grid_dim=(num_blocks,),
        block_dim=(block_size,),
    )
    ctx.synchronize()
    
    # Unrotate: mean[i, t] = sum_s Q[t, s] * mean_rotated[i, s]
    ctx.enqueue_function[kernel_unrotate_predictions](
        mean_device.unsafe_ptr(),
        mean_rotated_device.unsafe_ptr(),
        Q_device.unsafe_ptr(),
        m,
        T,
        grid_dim=(num_blocks,),
        block_dim=(block_size,),
    )
    ctx.synchronize()
    
    # Copy result to host
    var mean_host = HostBuffer[float_dtype](ctx, m * T)
    ctx.enqueue_copy(dst_buf=mean_host, src_buf=mean_device)
    ctx.synchronize()
    
    return mean_host


# =============================================================================
# Helper: Compute k(x*, x*) for a test point
# =============================================================================

fn compute_k_star_star(
    x_test_host: HostBuffer[float_dtype],
    j: Int,
    dim: Int,
    kernel_type: Int,
    outputscale: Float32,
    kernel_param1: Float32,
    kernel_param2: Float32,
) -> Float32:
    """Compute k(x*_j, x*_j) for a test point.
    
    For stationary kernels (RBF, Matern, Periodic, RQ): k(x,x) = outputscale
    For Linear: k(x,x) = outputscale * (||x||^2 + bias)
    For Polynomial: k(x,x) = outputscale * (||x||^2 + offset)^degree
    """
    if kernel_type == KERNEL_TYPE_LINEAR:
        var norm_sq = Float32(0.0)
        for dd in range(dim):
            var val = x_test_host[j * dim + dd]
            norm_sq += val * val
        return outputscale * (norm_sq + kernel_param1)
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        var norm_sq = Float32(0.0)
        for dd in range(dim):
            var val = x_test_host[j * dim + dd]
            norm_sq += val * val
        return outputscale * ((norm_sq + kernel_param2) ** kernel_param1)
    else:
        # Stationary kernels: k(x*, x*) = outputscale
        return outputscale


# =============================================================================
# Variance Prediction (Exact via CG)
# =============================================================================

fn predict_variance_exact_multi_output(
    ctx: DeviceContext,
    x_train_device: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host: HostBuffer[float_dtype],
    Q_host: HostBuffer[float_dtype],               # [T x T], row-major
    effective_scales_host: HostBuffer[float_dtype], # [T]
    n: Int,
    m: Int,
    dim: Int,
    num_tasks: Int,
    kernel_type: Int,
    lengthscale: Float32,
    outputscale: Float32,
    noise: Float32,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = Float32(1e-3),
    precond_rank: Int = 15,
    use_ard: Bool = False,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
) raises -> HostBuffer[float_dtype]:
    """Predict variance for multi-output GP using exact CG solve.
    
    For each sub-problem s (eigencomponent of task covariance B):
        K_s = s_s * K_X + noise * I
        Solve K_s @ V_s = K_cross_s  (m RHS columns)
        var_rotated_s(x*_j) = s_s * k(x*_j, x*_j) - dot(K_cross_s[:,j], V_s[:,j])
    
    Then unrotate: var_t(x*) = sum_s Q[t,s]^2 * var_rotated_s(x*)
    
    Args:
        ctx: GPU device context
        x_train_device: Training data [n x dim] on device (raw pointer)
        x_test_host: Test data [m x dim] on host
        Q_host: Eigenvector matrix [T x T], row-major
        effective_scales_host: Effective scales s_s = outputscale * lambda_s [T]
        n: Number of training points
        m: Number of test points
        dim: Input dimension
        num_tasks: Number of tasks T
        kernel_type: Kernel type
        lengthscale: Kernel lengthscale (ignored if use_ard=True)
        outputscale: Output scale
        noise: Noise variance
        max_cg_iter: Maximum CG iterations for variance solve
        cg_tol: CG convergence tolerance
        precond_rank: Pivoted Cholesky preconditioner rank
        use_ard: Whether to use ARD (per-dimension lengthscales)
        lengthscales_ptr: Pointer to [dim] per-dimension lengthscales on host (required if use_ard=True)
        
    Returns:
        variance_host: Predicted variances [m x T], row-major
    """
    var T = num_tasks
    
    # Allocate rotated variance buffer on host [m x T], row-major
    var var_rotated_host = HostBuffer[float_dtype](ctx, m * T)
    ctx.synchronize()
    
    # Create params buffer for MaterializedProvider
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = HostBuffer[float_dtype](ctx, 2)
    ctx.synchronize()
    
    # For each sub-problem s, solve K_s @ V_s = K_cross_s via CG
    for s in range(T):
        var s_s = effective_scales_host.unsafe_ptr()[s]
        
        # Update params buffer: [lengthscale, s_s]
        params_host_temp.unsafe_ptr()[0] = lengthscale
        params_host_temp.unsafe_ptr()[1] = s_s
        ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
        ctx.synchronize()
        
        # Create provider for sub-problem s: K_s = s_s * K_X + noise * I
        var provider = MaterializedProvider(
            ctx,
            x_train_device,
            params_device.unsafe_ptr(),
            n, dim, kernel_type,
            use_ard,
            lengthscale,
            s_s,    # outputscale = effective scale for this sub-problem
            noise,
            Float32(0.0),  # kernel_param1
            Float32(0.0),  # kernel_param2
        )
        if use_ard:
            # Re-materialize with correct ARD lengthscales
            provider.update_hyperparams_ard(lengthscales_ptr, s_s, noise)
        
        # Compute cross-covariance K_cross_s [n x m], column-major
        # K_cross_s[i,j] = s_s * k(x_train_i, x_test_j)
        var K_cross_host = compute_cross_covariance_with_provider(
            provider, x_test_host, m
        )
        
        # Copy K_cross to device as RHS for CG
        var K_cross_device = ctx.enqueue_create_buffer[float_dtype](n * m)
        ctx.enqueue_copy(dst_buf=K_cross_device, src_buf=K_cross_host)
        ctx.synchronize()
        
        # Wrap provider in adapter for CG
        var adapter = IsotropicGradientAdapter(provider^)
        
        # Build preconditioner for this sub-problem
        # max_num_cols must be >= m because CG solves m RHS columns simultaneously
        var precond = build_pivoted_cholesky_precond_unified(adapter, precond_rank, max_num_cols=m)
        
        # Create CG buffer pool
        var pool = CGBufferPool(ctx, n, m)
        
        # Solve K_s @ V_s = K_cross_s via batched CG (no tridiag needed)
        var cg_result = batched_cg_unified(
            adapter, K_cross_device.unsafe_ptr(), n, m,
            max_cg_iter,
            0,       # max_tridiag_iter=0, no log-det needed
            cg_tol,
            precond, pool
        )
        
        # Copy CG solution V_s [n x m] to host (column-major)
        var V_host = ctx.enqueue_create_host_buffer[float_dtype](n * m)
        ctx.enqueue_copy(dst_buf=V_host, src_buf=cg_result.solution)
        ctx.synchronize()
        
        # Compute var_rotated_s for each test point
        for j in range(m):
            # dot(K_cross_s[:,j], V_s[:,j])
            var dot_product = Float32(0.0)
            for i in range(n):
                # Column-major: column j at offset j * n
                dot_product += K_cross_host[j * n + i] * V_host[j * n + i]
            
            # k(x*_j, x*_j) with the sub-problem's effective scale
            var k_star_star = compute_k_star_star(
                x_test_host, j, dim, kernel_type, s_s, Float32(0.0), Float32(0.0)
            )
            
            # var_rotated_s(x*_j) = k_star_star - dot_product
            var variance = k_star_star - dot_product
            if variance < Float32(1e-10):
                variance = Float32(1e-10)
            
            # Store in rotated buffer: var_rotated[j, s] at row-major index j * T + s
            var_rotated_host.unsafe_ptr()[j * T + s] = variance
        
        # Keep buffers alive
        _ = K_cross_device
        _ = K_cross_host
        _ = V_host
    
    # Keep params buffer alive for all iterations
    _ = params_device
    _ = params_host_temp
    
    # Unrotate variance on GPU: var[j, t] = sum_s Q[t,s]^2 * var_rotated[j, s]
    var var_rotated_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    ctx.enqueue_copy(dst_buf=var_rotated_device, src_buf=var_rotated_host)
    
    var Q_device = ctx.enqueue_create_buffer[float_dtype](T * T)
    ctx.enqueue_copy(dst_buf=Q_device, src_buf=Q_host)
    
    var variance_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    var variance_host = HostBuffer[float_dtype](ctx, m * T)
    ctx.synchronize()
    
    var block_size = 256
    var num_blocks = (m * T + block_size - 1) // block_size
    ctx.enqueue_function[kernel_unrotate_variance](
        variance_device.unsafe_ptr(),
        var_rotated_device.unsafe_ptr(),
        Q_device.unsafe_ptr(),
        m,
        T,
        grid_dim=(num_blocks,),
        block_dim=(block_size,),
    )
    ctx.synchronize()
    
    # Copy result to host
    ctx.enqueue_copy(dst_buf=variance_host, src_buf=variance_device)
    ctx.synchronize()
    
    return variance_host


# =============================================================================
# Variance Prediction (LOVE — inline Lanczos + variance per sub-problem)
# =============================================================================

fn predict_variance_love_multi_output_inline(
    ctx: DeviceContext,
    x_train_device: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host: HostBuffer[float_dtype],
    Q_host: HostBuffer[float_dtype],               # [T x T], row-major
    effective_scales_host: HostBuffer[float_dtype], # [T]
    n: Int,
    m: Int,
    dim: Int,
    num_tasks: Int,
    kernel_type: Int,
    lengthscale: Float32,
    outputscale: Float32,
    noise: Float32,
    use_ard: Bool = False,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    lanczos_iter: Int = 20,
) raises -> HostBuffer[float_dtype]:
    """Predict variance for multi-output GP using LOVE (fast low-rank approximation).
    
    For each sub-problem s, computes Lanczos root inline and then LOVE variance:
        S_s = lanczos_root(K_s, rank=r)
        K_cross_s = s_s * k(X_test, X_train)
        V_s = S_s^T @ K_cross_s  [r x m]
        var_rotated_s[j] = s_s * k(x*,x*) - ||V_s[:,j]||^2
    
    Then unrotate: var_t(x*) = sum_s Q[t,s]^2 * var_rotated_s(x*)
    
    Args:
        ctx: GPU device context
        x_train_device: Training data [n x dim] on device (raw pointer)
        x_test_host: Test data [m x dim] on host
        Q_host: Eigenvector matrix [T x T], row-major
        effective_scales_host: Effective scales s_s = outputscale * lambda_s [T]
        n: Number of training points
        m: Number of test points
        dim: Input dimension
        num_tasks: Number of tasks T
        kernel_type: Kernel type
        lengthscale: Kernel lengthscale
        outputscale: Output scale
        noise: Noise variance
        use_ard: Whether to use ARD (per-dimension lengthscales)
        lengthscales_ptr: Pointer to [dim] per-dimension lengthscales on host (required if use_ard=True)
        lanczos_iter: Number of Lanczos iterations (rank r)
        
    Returns:
        variance_host: Predicted variances [m x T], row-major
    """
    var T = num_tasks
    
    # Allocate rotated variance buffer on host [m x T], row-major
    var var_rotated_host = HostBuffer[float_dtype](ctx, m * T)
    ctx.synchronize()
    
    # Create params buffer for MaterializedProvider
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = HostBuffer[float_dtype](ctx, 2)
    ctx.synchronize()
    
    for s in range(T):
        var s_s = effective_scales_host.unsafe_ptr()[s]
        
        # Update params buffer: [lengthscale, s_s]
        params_host_temp.unsafe_ptr()[0] = lengthscale
        params_host_temp.unsafe_ptr()[1] = s_s
        ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
        ctx.synchronize()
        
        # Create provider for sub-problem s
        var provider = MaterializedProvider(
            ctx,
            x_train_device,
            params_device.unsafe_ptr(),
            n, dim, kernel_type,
            use_ard,
            lengthscale,
            s_s,
            noise,
            Float32(0.0),
            Float32(0.0),
        )
        if use_ard:
            # Re-materialize with correct ARD lengthscales
            provider.update_hyperparams_ard(lengthscales_ptr, s_s, noise)
        
        # Compute Lanczos root for this sub-problem
        var lr = compute_lanczos_root_with_provider(provider, lanczos_iter)
        var r = lr.rank
        
        # Compute cross-covariance K_cross_s [n x m], column-major
        var K_cross_host = compute_cross_covariance_with_provider(
            provider, x_test_host, m
        )
        
        # Compute V_s = S_s^T @ K_cross_s  [r x m]
        # S_s is [n x r] column-major (from LanczosRootResult.root)
        # K_cross_s is [n x m] column-major
        # V_s[i,j] = sum_k S_s[k,i] * K_cross_s[k,j]
        for j in range(m):
            var v_norm_sq = Float32(0.0)
            for i in range(r):
                var dot = Float32(0.0)
                for k in range(n):
                    # S_s[k,i] in column-major: lr.root[i * n + k]
                    # K_cross_s[k,j] in column-major: K_cross_host[j * n + k]
                    dot += lr.root[i * n + k] * K_cross_host[j * n + k]
                v_norm_sq += dot * dot
            
            # k(x*_j, x*_j) with the sub-problem's effective scale
            var k_star_star = compute_k_star_star(
                x_test_host, j, dim, kernel_type, s_s, Float32(0.0), Float32(0.0)
            )
            
            # var_rotated_s[j] = k_star_star - ||V_s[:,j]||^2
            var variance = k_star_star - v_norm_sq
            if variance < Float32(1e-10):
                variance = Float32(1e-10)
            
            var_rotated_host.unsafe_ptr()[j * T + s] = variance
        
        _ = K_cross_host
    
    # Keep params buffer alive
    _ = params_device
    _ = params_host_temp
    
    # Unrotate variance on GPU
    var var_rotated_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    ctx.enqueue_copy(dst_buf=var_rotated_device, src_buf=var_rotated_host)
    
    var Q_device = ctx.enqueue_create_buffer[float_dtype](T * T)
    ctx.enqueue_copy(dst_buf=Q_device, src_buf=Q_host)
    
    var variance_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    var variance_host = HostBuffer[float_dtype](ctx, m * T)
    ctx.synchronize()
    
    var block_size = 256
    var num_blocks = (m * T + block_size - 1) // block_size
    ctx.enqueue_function[kernel_unrotate_variance](
        variance_device.unsafe_ptr(),
        var_rotated_device.unsafe_ptr(),
        Q_device.unsafe_ptr(),
        m,
        T,
        grid_dim=(num_blocks,),
        block_dim=(block_size,),
    )
    ctx.synchronize()
    
    ctx.enqueue_copy(dst_buf=variance_host, src_buf=variance_device)
    ctx.synchronize()
    
    return variance_host


# =============================================================================
# Combined Prediction
# =============================================================================

fn predict_multi_output(
    ctx: DeviceContext,
    x_train_device: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host: HostBuffer[float_dtype],
    alpha_rotated_host: HostBuffer[float_dtype],  # [n x T], row-major
    Q_host: HostBuffer[float_dtype],               # [T x T], row-major
    effective_scales_host: HostBuffer[float_dtype], # [T]
    n: Int,
    m: Int,
    dim: Int,
    num_tasks: Int,
    kernel_type: Int,
    lengthscale: Float32,
    outputscale: Float32,
    noise: Float32,
    compute_variance: Bool = True,
    use_ard: Bool = False,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    variance_method: Int = 0,
) raises -> MultiOutputPredictionResult:
    """Predict mean and variance for multi-output GP.
    
    Supports both LOVE (fast approximate) and exact CG variance methods.
    
    Args:
        ctx: GPU device context
        x_train_device: Training data [n x dim] on device
        x_test_host: Test data [m x dim] on host
        alpha_rotated_host: CG solutions in rotated basis [n x T], row-major
        Q_host: Eigenvector matrix [T x T], row-major
        effective_scales_host: Effective scales s_t = outputscale * lambda_t [T]
        n: Number of training points
        m: Number of test points
        dim: Input dimension
        num_tasks: Number of tasks T
        kernel_type: Kernel type
        lengthscale: Kernel lengthscale
        outputscale: Output scale
        noise: Noise variance
        compute_variance: Whether to compute variance
        use_ard: Whether to use ARD (per-dimension lengthscales)
        lengthscales_ptr: Pointer to [dim] per-dimension lengthscales on host (required if use_ard=True)
        variance_method: 0 = LOVE (fast approximate), 1 = exact CG variance
        
    Returns:
        MultiOutputPredictionResult with mean and optionally variance
    """
    # Compute mean
    var mean_host = predict_mean_multi_output(
        ctx,
        x_train_device,
        x_test_host,
        alpha_rotated_host,
        Q_host,
        effective_scales_host,
        n,
        m,
        dim,
        num_tasks,
        kernel_type,
        lengthscale,
        outputscale,
        noise,
        use_ard,
        lengthscales_ptr,
    )
    
    # Compute variance if requested
    var variance_host: HostBuffer[float_dtype]
    if compute_variance:
        if variance_method == 1:
            # Exact CG variance (preconditioned, more accurate but slower)
            variance_host = predict_variance_exact_multi_output(
                ctx,
                x_train_device,
                x_test_host,
                Q_host,
                effective_scales_host,
                n,
                m,
                dim,
                num_tasks,
                kernel_type,
                lengthscale,
                outputscale,
                noise,
                use_ard=use_ard,
                lengthscales_ptr=lengthscales_ptr,
            )
        else:
            # LOVE variance (default, fast approximate)
            variance_host = predict_variance_love_multi_output_inline(
                ctx,
                x_train_device,
                x_test_host,
                Q_host,
                effective_scales_host,
                n,
                m,
                dim,
                num_tasks,
                kernel_type,
                lengthscale,
                outputscale,
                noise,
                use_ard,
                lengthscales_ptr,
            )
    else:
        variance_host = HostBuffer[float_dtype](ctx, m * num_tasks)
    
    return MultiOutputPredictionResult(
        mean_host,
        variance_host,
        m,
        num_tasks,
        compute_variance,
    )


# =============================================================================
# LMC (Linear Model of Coregionalization) Prediction
# =============================================================================
#
# For LMC, the full kernel is K_full = sum_{s=1}^{R} (A_s ⊗ K_X_s) + D
# where each latent s has its own kernel type and lengthscale.
#
# Mean prediction:
#   mu_t(x*) = sum_s sum_{t'} A_s[t,t'] * k_s(x*, X) @ alpha_{t'}
#
# Variance prediction (block-diagonal approximation):
#   For each latent s, compute LOVE root S_s of (K_s + noise*I)
#   var_t(x*) ≈ sum_s A_s[t,t] * k_s(x*,x*) 
#             - sum_s sum_{t1,t2} A_s[t,t1] * A_s[t,t2] * ||S_s^T @ k_s(x*,X) @ alpha_{t'}||^2
#   This is exact when R=1 (reduces to ICM).
# =============================================================================


fn predict_mean_multi_output_lmc(
    ctx: DeviceContext,
    x_train_device: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host: HostBuffer[float_dtype],
    alpha_host: HostBuffer[float_dtype],           # [n x T], row-major (direct, NOT rotated)
    A_all_host: HostBuffer[float_dtype],           # [R * T * T] contiguous, A_s row-major per latent
    kernel_types_host: HostBuffer[float_dtype],    # [R] kernel type per latent (stored as float)
    lengthscales_host: HostBuffer[float_dtype],    # [R] or [R*d] lengthscale per latent (ARD: per-dim)
    outputscales_host: HostBuffer[float_dtype],    # [R] outputscale per latent
    kernel_params1_host: HostBuffer[float_dtype],  # [R] kernel_param1 per latent
    kernel_params2_host: HostBuffer[float_dtype],  # [R] kernel_param2 per latent
    n: Int,
    m: Int,
    dim: Int,
    num_tasks: Int,
    num_latents: Int,
    use_ard: Bool = False,
) raises -> HostBuffer[float_dtype]:
    """Predict mean for LMC multi-output GP.
    
    mu_t(x*) = sum_{s=1}^{R} sum_{t'=1}^{T} A_s[t,t'] * k_s(x*, X) @ alpha_{t'}
    
    For each latent s, we create a MaterializedProvider with that latent's kernel
    type and lengthscale, compute cross-matvec k_s(x*, X) @ alpha_{t'} for all T
    tasks, then accumulate with A_s weighting.
    
    Args:
        ctx: GPU device context
        x_train_device: Training data [n x dim] on device (raw pointer)
        x_test_host: Test data [m x dim] on host
        alpha_host: CG solution [n x T], row-major (direct alpha, not rotated)
        A_all_host: Coregionalization matrices [R * T * T], A_s[t,t'] at offset s*T*T + t*T + t'
        kernel_types_host: Kernel type per latent [R] (stored as float, cast to int)
        lengthscales_host: Lengthscale per latent [R]
        outputscales_host: Outputscale per latent [R]
        kernel_params1_host: kernel_param1 per latent [R]
        kernel_params2_host: kernel_param2 per latent [R]
        n: Number of training points
        m: Number of test points
        dim: Input dimension
        num_tasks: Number of tasks T
        num_latents: Number of latents R
        
    Returns:
        mean_host: Predicted means [m x T], row-major
    """
    var T = num_tasks
    var R = num_latents
    
    # Allocate output mean buffer on host, initialized to zero
    var mean_host = HostBuffer[float_dtype](ctx, m * T)
    for i in range(m * T):
        mean_host.unsafe_ptr()[i] = Float32(0.0)
    
    # Copy alpha [n x T] to device once (row-major, used for all latents)
    var alpha_device = ctx.enqueue_create_buffer[float_dtype](n * T)
    ctx.enqueue_copy(dst_buf=alpha_device, src_buf=alpha_host)
    
    # Create params buffer for MaterializedProvider
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = HostBuffer[float_dtype](ctx, 2)
    ctx.synchronize()
    
    # For each latent s, compute K_cross_s once and do batched matmul with all T alpha columns
    for s in range(R):
        var kernel_type = Int(kernel_types_host.unsafe_ptr()[s])
        # For ARD, use mean of per-dim lengthscales as the scalar; for isotropic, use directly
        var lengthscale: Float32
        if use_ard:
            var ls_sum = Float32(0.0)
            for d_idx in range(dim):
                ls_sum += lengthscales_host.unsafe_ptr()[s * dim + d_idx]
            lengthscale = ls_sum / Float32(dim)
        else:
            lengthscale = lengthscales_host.unsafe_ptr()[s]
        var outputscale = outputscales_host.unsafe_ptr()[s]
        var kp1 = kernel_params1_host.unsafe_ptr()[s]
        var kp2 = kernel_params2_host.unsafe_ptr()[s]
        
        # Update params buffer
        params_host_temp.unsafe_ptr()[0] = lengthscale
        params_host_temp.unsafe_ptr()[1] = outputscale
        ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
        ctx.synchronize()
        
        # Create provider for latent s
        var provider = MaterializedProvider(
            ctx,
            x_train_device,
            params_device.unsafe_ptr(),
            n, dim, kernel_type,
            use_ard,
            lengthscale,
            outputscale,
            Float32(0.0),  # noise=0 for cross-covariance (no noise in k_s)
            kp1,
            kp2,
        )
        if use_ard:
            # Set per-dimension lengthscales for this latent
            provider.update_hyperparams_ard(
                lengthscales_host.unsafe_ptr().offset(s * dim),
                outputscale,
                Float32(0.0),  # noise=0 for cross-covariance
            )
        
        # Compute K_cross_s [n x m] column-major on GPU ONCE for this latent
        # (same cross-covariance for all T tasks)
        var K_cross_device = compute_cross_covariance_device_with_provider(
            provider, x_test_host, m
        )
        
        # Batched GPU matmul: cross_result = K_cross_s^T @ alpha_all -> [m x T] row-major
        # This replaces T separate cross_matvec_with_provider calls with one GPU kernel
        var cross_result_device = ctx.enqueue_create_buffer[float_dtype](m * T)
        ctx.synchronize()
        
        alias BLOCK_SIZE = 256
        var total_elements = m * T
        var grid_dim = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        ctx.enqueue_function[kernel_KcrossT_matmul_alpha](
            cross_result_device.unsafe_ptr(),
            K_cross_device.unsafe_ptr(),
            alpha_device.unsafe_ptr(),
            n, m, T,
            grid_dim=(grid_dim,), block_dim=(BLOCK_SIZE,)
        )
        ctx.synchronize()
        
        # Copy batched result [m x T] to host
        var cross_result_host = HostBuffer[float_dtype](ctx, m * T)
        ctx.enqueue_copy(dst_buf=cross_result_host, src_buf=cross_result_device)
        ctx.synchronize()
        
        # Accumulate: mean[j, t] += sum_{t'} A_s[t, t'] * cross_result[j, t']
        # cross_result[j, t'] = K_cross_s^T[j, :] @ alpha[:, t'] = sum_i K_cross_s[i, j] * alpha[i, t']
        var A_s_offset = s * T * T
        for j in range(m):
            for t in range(T):
                var acc = Float32(0.0)
                for tp in range(T):
                    var A_s_t_tp = A_all_host.unsafe_ptr()[A_s_offset + t * T + tp]
                    acc += A_s_t_tp * cross_result_host.unsafe_ptr()[j * T + tp]
                mean_host.unsafe_ptr()[j * T + t] += acc
        
        # Keep buffers alive
        _ = K_cross_device
        _ = cross_result_device
        _ = cross_result_host
    
    # Keep buffers alive
    _ = alpha_device
    _ = params_device
    _ = params_host_temp
    
    return mean_host


fn predict_variance_lmc(
    ctx: DeviceContext,
    x_train_device: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host: HostBuffer[float_dtype],
    A_all_host: HostBuffer[float_dtype],           # [R * T * T] contiguous
    kernel_types_host: HostBuffer[float_dtype],    # [R] kernel type per latent
    lengthscales_host: HostBuffer[float_dtype],    # [R] or [R*d] lengthscale per latent (ARD: per-dim)
    outputscales_host: HostBuffer[float_dtype],    # [R] outputscale per latent
    kernel_params1_host: HostBuffer[float_dtype],  # [R] kernel_param1 per latent
    kernel_params2_host: HostBuffer[float_dtype],  # [R] kernel_param2 per latent
    noise: Float32,
    n: Int,
    m: Int,
    dim: Int,
    num_tasks: Int,
    num_latents: Int,
    lanczos_iter: Int = 20,
    use_ard: Bool = False,
) raises -> HostBuffer[float_dtype]:
    """Predict variance for LMC multi-output GP using per-latent LOVE.
    
    Uses a block-diagonal approximation: for each latent s, compute LOVE root
    S_s of (outputscale_s * K_X_s + noise * I), then:
    
    var_t(x*) = sum_s A_s[t,t] * k_s(x*,x*) 
              - sum_s (sum_{t'} A_s[t,t'] * S_s^T @ k_s(x*,X))^2  [summed over LOVE rank]
    
    More precisely, for each latent s with LOVE root S_s [n x r_s]:
      c_s[j] = k_s(x*_j, X)  [n-vector]
      For each task t:
        w_s_t[j] = sum_{t'} A_s[t,t'] * S_s^T @ c_s[j]  [r_s-vector]
        contribution_s_t[j] = ||w_s_t[j]||^2
    
    var_t(x*_j) = sum_s A_s[t,t] * k_s(x*_j, x*_j) - sum_s contribution_s_t[j]
    
    This is exact when R=1 (reduces to ICM LOVE variance).
    
    Args:
        ctx: GPU device context
        x_train_device: Training data [n x dim] on device (raw pointer)
        x_test_host: Test data [m x dim] on host
        A_all_host: Coregionalization matrices [R * T * T]
        kernel_types_host: Kernel type per latent [R]
        lengthscales_host: Lengthscale per latent [R]
        outputscales_host: Outputscale per latent [R]
        kernel_params1_host: kernel_param1 per latent [R]
        kernel_params2_host: kernel_param2 per latent [R]
        noise: Average noise variance (used for LOVE solve)
        n: Number of training points
        m: Number of test points
        dim: Input dimension
        num_tasks: Number of tasks T
        num_latents: Number of latents R
        lanczos_iter: Number of Lanczos iterations for LOVE
        
    Returns:
        variance_host: Predicted variances [m x T], row-major
    """
    var T = num_tasks
    var R = num_latents
    
    # Allocate output variance buffer, initialized to zero
    var variance_host = HostBuffer[float_dtype](ctx, m * T)
    for i in range(m * T):
        variance_host.unsafe_ptr()[i] = Float32(0.0)
    
    # Create params buffer for MaterializedProvider
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = HostBuffer[float_dtype](ctx, 2)
    ctx.synchronize()
    
    # For each latent s
    for s in range(R):
        var kernel_type = Int(kernel_types_host.unsafe_ptr()[s])
        # For ARD, use mean of per-dim lengthscales as the scalar; for isotropic, use directly
        var lengthscale: Float32
        if use_ard:
            var ls_sum = Float32(0.0)
            for d_idx in range(dim):
                ls_sum += lengthscales_host.unsafe_ptr()[s * dim + d_idx]
            lengthscale = ls_sum / Float32(dim)
        else:
            lengthscale = lengthscales_host.unsafe_ptr()[s]
        var outputscale = outputscales_host.unsafe_ptr()[s]
        var kp1 = kernel_params1_host.unsafe_ptr()[s]
        var kp2 = kernel_params2_host.unsafe_ptr()[s]
        
        # Update params buffer
        params_host_temp.unsafe_ptr()[0] = lengthscale
        params_host_temp.unsafe_ptr()[1] = outputscale
        ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
        ctx.synchronize()
        
        # Create provider for latent s (with noise for the LOVE solve)
        var provider = MaterializedProvider(
            ctx,
            x_train_device,
            params_device.unsafe_ptr(),
            n, dim, kernel_type,
            use_ard,
            lengthscale,
            outputscale,
            noise,  # Include noise for LOVE solve (K_s + noise*I)
            kp1,
            kp2,
        )
        if use_ard:
            # Set per-dimension lengthscales for this latent
            provider.update_hyperparams_ard(
                lengthscales_host.unsafe_ptr().offset(s * dim),
                outputscale,
                noise,
            )
        
        # Compute Lanczos root for this latent's kernel
        var lr = compute_lanczos_root_with_provider(provider, lanczos_iter)
        var r = lr.rank
        
        # Compute cross-covariance K_cross_s [n x m] column-major on GPU (stays on device)
        var K_cross_device = compute_cross_covariance_device_with_provider(
            provider, x_test_host, m
        )
        
        # GPU: V_s = S_s^T @ K_cross_s  [r x m] column-major
        # Copy Lanczos root S_s [n x r] from host to device
        var S_device = ctx.enqueue_create_buffer[float_dtype](n * r)
        ctx.enqueue_copy(dst_buf=S_device, src_buf=lr.root)
        
        var V_device = ctx.enqueue_create_buffer[float_dtype](r * m)
        ctx.synchronize()
        
        alias BLOCK_X = 16
        alias BLOCK_Y = 16
        var grid_x = (r + BLOCK_X - 1) // BLOCK_X
        var grid_y = (m + BLOCK_Y - 1) // BLOCK_Y
        
        ctx.enqueue_function[kernel_matmul_ST_K](
            S_device.unsafe_ptr(), K_cross_device.unsafe_ptr(), V_device.unsafe_ptr(),
            n, r, m,
            grid_dim=(grid_x, grid_y), block_dim=(BLOCK_X, BLOCK_Y)
        )
        
        # GPU: vnorm_sq[j] = ||V_s[:, j]||^2 for each test point j
        var vnorm_sq_device = ctx.enqueue_create_buffer[float_dtype](m)
        ctx.synchronize()
        
        alias BLOCK_VAR = 256
        var grid_var = (m + BLOCK_VAR - 1) // BLOCK_VAR
        
        ctx.enqueue_function[kernel_vnorm_sq_from_ST_Kcross](
            vnorm_sq_device.unsafe_ptr(), V_device.unsafe_ptr(),
            r, m,
            grid_dim=(grid_var,), block_dim=(BLOCK_VAR,)
        )
        ctx.synchronize()
        
        # Copy vnorm_sq [m] to host
        var vnorm_sq_host = HostBuffer[float_dtype](ctx, m)
        ctx.enqueue_copy(dst_buf=vnorm_sq_host, src_buf=vnorm_sq_device)
        ctx.synchronize()
        
        # Accumulate variance contribution from latent s for each task
        # Using diagonal approximation:
        #   prior_contribution = A_s[t,t] * k_star_star
        #   posterior_reduction = A_s[t,t] * vnorm_sq[j]
        #   net_contribution = A_s[t,t] * (k_star_star - vnorm_sq[j])
        var A_s_offset = s * T * T
        
        for j in range(m):
            var vnorm_sq_j = vnorm_sq_host.unsafe_ptr()[j]
            
            # k_s(x*_j, x*_j)
            var k_star_star = compute_k_star_star(
                x_test_host, j, dim, kernel_type, outputscale, kp1, kp2
            )
            
            for t in range(T):
                var A_s_tt = A_all_host.unsafe_ptr()[A_s_offset + t * T + t]
                variance_host.unsafe_ptr()[j * T + t] += A_s_tt * (k_star_star - vnorm_sq_j)
        
        # Keep buffers alive
        _ = K_cross_device
        _ = S_device
        _ = V_device
        _ = vnorm_sq_device
        _ = vnorm_sq_host
    
    # Clamp variance to be non-negative
    for i in range(m * T):
        if variance_host.unsafe_ptr()[i] < Float32(1e-10):
            variance_host.unsafe_ptr()[i] = Float32(1e-10)
    
    # Keep params buffer alive
    _ = params_device
    _ = params_host_temp
    
    return variance_host


fn predict_multi_output_lmc(
    ctx: DeviceContext,
    x_train_device: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host: HostBuffer[float_dtype],
    alpha_host: HostBuffer[float_dtype],           # [n x T], row-major (direct, NOT rotated)
    A_all_host: HostBuffer[float_dtype],           # [R * T * T] contiguous
    kernel_types_host: HostBuffer[float_dtype],    # [R] kernel type per latent
    lengthscales_host: HostBuffer[float_dtype],    # [R] or [R*d] lengthscale per latent (ARD: per-dim)
    outputscales_host: HostBuffer[float_dtype],    # [R] outputscale per latent
    kernel_params1_host: HostBuffer[float_dtype],  # [R] kernel_param1 per latent
    kernel_params2_host: HostBuffer[float_dtype],  # [R] kernel_param2 per latent
    noise: Float32,
    n: Int,
    m: Int,
    dim: Int,
    num_tasks: Int,
    num_latents: Int,
    compute_variance: Bool = True,
    use_ard: Bool = False,
) raises -> MultiOutputPredictionResult:
    """Predict mean and variance for LMC multi-output GP.
    
    Uses per-latent kernels and A_s matrices for correct LMC prediction.
    Variance uses per-latent LOVE with diagonal approximation.
    
    Args:
        ctx: GPU device context
        x_train_device: Training data [n x dim] on device
        x_test_host: Test data [m x dim] on host
        alpha_host: CG solution [n x T], row-major (direct alpha)
        A_all_host: Coregionalization matrices [R * T * T]
        kernel_types_host: Kernel type per latent [R]
        lengthscales_host: Lengthscale per latent [R]
        outputscales_host: Outputscale per latent [R]
        kernel_params1_host: kernel_param1 per latent [R]
        kernel_params2_host: kernel_param2 per latent [R]
        noise: Average noise variance
        n: Number of training points
        m: Number of test points
        dim: Input dimension
        num_tasks: Number of tasks T
        num_latents: Number of latents R
        compute_variance: Whether to compute variance
        
    Returns:
        MultiOutputPredictionResult with mean and optionally variance
    """
    # Compute mean
    var mean_host = predict_mean_multi_output_lmc(
        ctx, x_train_device, x_test_host,
        alpha_host, A_all_host,
        kernel_types_host, lengthscales_host, outputscales_host,
        kernel_params1_host, kernel_params2_host,
        n, m, dim, num_tasks, num_latents,
        use_ard=use_ard,
    )
    
    # Compute variance if requested
    var variance_host: HostBuffer[float_dtype]
    if compute_variance:
        variance_host = predict_variance_lmc(
            ctx, x_train_device, x_test_host,
            A_all_host,
            kernel_types_host, lengthscales_host, outputscales_host,
            kernel_params1_host, kernel_params2_host,
            noise,
            n, m, dim, num_tasks, num_latents,
            use_ard=use_ard,
        )
    else:
        variance_host = HostBuffer[float_dtype](ctx, m * num_tasks)
    
    return MultiOutputPredictionResult(
        mean_host,
        variance_host,
        m,
        num_tasks,
        compute_variance,
    )


# =============================================================================
# Composite Kernel Multi-Output Prediction
# =============================================================================

fn predict_multi_output_composite[DIM: Int, K: ComposableKernel](
    ctx: DeviceContext,
    x_train_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    m: Int,
    num_tasks: Int,
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [K.num_params()] kernel params
    noise: Float32,
    alpha_rotated_host: HostBuffer[float_dtype],  # [n x T], row-major
    Q_host: HostBuffer[float_dtype],               # [T x T], row-major
    Lambda_host: HostBuffer[float_dtype],           # [T] eigenvalues
    compute_variance: Bool = True,
) raises -> MultiOutputPredictionResult:
    """Predict mean and variance for multi-output GP with composite kernel.
    
    Uses the ICM decomposition with ScaleKernel[K] per sub-problem.
    For each sub-problem s:
        K_s = lambda_s * K_composite(params) + noise * I
    
    Mean: mu_t(x*) = sum_s Q[t,s] * K_cross_s(x*, X)^T @ alpha_s
    Variance: var_t(x*) = sum_s Q[t,s]^2 * var_rotated_s(x*)
    
    Args:
        ctx: GPU device context
        x_train_host_ptr: Training data [n, DIM], row-major float32
        x_test_host_ptr: Test data [m, DIM], row-major float32
        n: Number of training points
        m: Number of test points
        num_tasks: Number of tasks T
        params_ptr: Kernel parameters [K.num_params()], float32
        noise: Noise variance
        alpha_rotated_host: CG solutions in rotated basis [n x T], row-major
        Q_host: Eigenvector matrix [T x T], row-major
        Lambda_host: Eigenvalues [T]
        compute_variance: Whether to compute variance
        
    Returns:
        MultiOutputPredictionResult with mean and optionally variance
    """
    var T = num_tasks
    alias NUM_KERNEL_PARAMS = K.num_params()
    alias NUM_SCALED_PARAMS = ScaleKernel[K].num_params()
    
    # Copy Q to device for unrotation
    var Q_device = ctx.enqueue_create_buffer[float_dtype](T * T)
    ctx.enqueue_copy(dst_buf=Q_device, src_buf=Q_host)
    
    # Allocate rotated mean/variance buffers
    var mean_rotated_host = HostBuffer[float_dtype](ctx, m * T)
    var var_rotated_host = HostBuffer[float_dtype](ctx, m * T)
    
    # Host buffer for ScaleKernel[K] params: [scale | K params...]
    var scaled_params_host = ctx.enqueue_create_host_buffer[float_dtype](NUM_SCALED_PARAMS)
    
    # Copy X_train to host buffer for provider construction
    var x_train_host = ctx.enqueue_create_host_buffer[float_dtype](n * DIM)
    for i in range(n * DIM):
        x_train_host[i] = x_train_host_ptr[i]
    
    # Copy X_test to host buffer
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](m * DIM)
    for i in range(m * DIM):
        x_test_host[i] = x_test_host_ptr[i]
    
    ctx.synchronize()
    
    # For each sub-problem s, compute mean and optionally variance
    for s in range(T):
        var lambda_s = Lambda_host.unsafe_ptr()[s]
        
        # Build ScaleKernel[K] params: [lambda_s | K params...]
        scaled_params_host[0] = lambda_s
        for p in range(NUM_KERNEL_PARAMS):
            scaled_params_host[1 + p] = params_ptr[p]
        
        # Create provider for sub-problem s
        var provider = CompositeProvider[DIM, ScaleKernel[K]](
            ctx,
            x_train_host.unsafe_ptr(),
            scaled_params_host.unsafe_ptr(),
            n,
            noise,
        )
        
        # Extract alpha_s (column s of alpha_rotated)
        var alpha_s_host = ctx.enqueue_create_host_buffer[float_dtype](n)
        for i in range(n):
            alpha_s_host[i] = alpha_rotated_host.unsafe_ptr()[i * T + s]
        
        # Compute mean: K_cross_s(x*, X)^T @ alpha_s
        var mean_s_host = predict_mean_composite[DIM, ScaleKernel[K]](
            provider, x_test_host.unsafe_ptr(), m, alpha_s_host
        )
        
        # Store in mean_rotated[:, s]
        for i in range(m):
            mean_rotated_host.unsafe_ptr()[i * T + s] = mean_s_host[i]
        
        # Compute variance if requested
        if compute_variance:
            # Compute Lanczos root for this sub-problem
            var lanczos_root = compute_lanczos_root_composite[DIM, ScaleKernel[K]](
                provider, 20  # lanczos_iter - match non-composite default
            )
            
            # Compute LOVE variance
            var var_s_host = predict_variance_love_composite[DIM, ScaleKernel[K]](
                provider, x_test_host.unsafe_ptr(), m, lanczos_root
            )
            
            # Store in var_rotated[:, s]
            for i in range(m):
                var_rotated_host.unsafe_ptr()[i * T + s] = var_s_host[i]
    
    # Unrotate mean: mean[i, t] = sum_s Q[t, s] * mean_rotated[i, s]
    var mean_rotated_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    var mean_device = ctx.enqueue_create_buffer[float_dtype](m * T)
    ctx.enqueue_copy(dst_buf=mean_rotated_device, src_buf=mean_rotated_host)
    ctx.synchronize()
    
    var block_size = 256
    var num_blocks = (m * T + block_size - 1) // block_size
    ctx.enqueue_function[kernel_unrotate_predictions](
        mean_device.unsafe_ptr(),
        mean_rotated_device.unsafe_ptr(),
        Q_device.unsafe_ptr(),
        m,
        T,
        grid_dim=(num_blocks,),
        block_dim=(block_size,),
    )
    ctx.synchronize()
    
    var mean_host = HostBuffer[float_dtype](ctx, m * T)
    ctx.enqueue_copy(dst_buf=mean_host, src_buf=mean_device)
    ctx.synchronize()
    
    # Unrotate variance if computed
    var variance_host: HostBuffer[float_dtype]
    if compute_variance:
        var var_rotated_device = ctx.enqueue_create_buffer[float_dtype](m * T)
        var var_device = ctx.enqueue_create_buffer[float_dtype](m * T)
        ctx.enqueue_copy(dst_buf=var_rotated_device, src_buf=var_rotated_host)
        ctx.synchronize()
        
        ctx.enqueue_function[kernel_unrotate_variance](
            var_device.unsafe_ptr(),
            var_rotated_device.unsafe_ptr(),
            Q_device.unsafe_ptr(),
            m,
            T,
            grid_dim=(num_blocks,),
            block_dim=(block_size,),
        )
        ctx.synchronize()
        
        variance_host = HostBuffer[float_dtype](ctx, m * T)
        ctx.enqueue_copy(dst_buf=variance_host, src_buf=var_device)
        ctx.synchronize()
    else:
        variance_host = HostBuffer[float_dtype](ctx, m * T)
    
    # Keep buffers alive
    _ = scaled_params_host
    _ = x_train_host
    _ = x_test_host
    _ = Q_device
    
    return MultiOutputPredictionResult(
        mean_host,
        variance_host,
        m,
        T,
        compute_variance,
    )
