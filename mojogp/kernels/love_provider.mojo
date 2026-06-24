"""LOVE prediction with provider abstraction.

Provides fast GP prediction using providers for both matrix-free and materialized approaches.
Supports both LOVE (low-rank approximation) and exact variance computation.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import sqrt

from .matvec_provider import MatvecProvider
from .cross_covariance_provider import cross_matvec_with_provider
from .kernel_params import KernelParams
from .gradient_provider import ForwardProvider
from .pivoted_cholesky import PivotedCholeskyPrecond, build_pivoted_cholesky_precond_unified
from .combined_inv_quad_logdet import batched_cg_unified, CGBufferPool

alias float_dtype = DType.float32


# =============================================================================
# GPU Kernels for LOVE (moved from love.mojo)
# =============================================================================

fn kernel_matmul_ST_K(
    S_ptr: UnsafePointer[Float32, MutAnyOrigin],      # n × r (column-major)
    K_cross_ptr: UnsafePointer[Float32, MutAnyOrigin], # n × m (column-major)
    V_ptr: UnsafePointer[Float32, MutAnyOrigin],       # r × m output (column-major)
    n: Int,
    r: Int,
    m: Int
) -> None:
    """Compute V = S^T @ K_cross where S is n×r and K_cross is n×m.
    
    Output V is r×m. Each thread computes one element V[i, j].
    
    Memory layout (all column-major):
    - S[k, i] = S_ptr[i * n + k]
    - K_cross[k, j] = K_cross_ptr[j * n + k]
    - V[i, j] = V_ptr[j * r + i]
    """
    var i = block_idx.x * block_dim.x + thread_idx.x  # row in V (0..r-1)
    var j = block_idx.y * block_dim.y + thread_idx.y  # col in V (0..m-1)
    
    if i >= UInt(r) or j >= UInt(m):
        return
    
    var sum_val = Float32(0.0)
    for k in range(n):
        sum_val += S_ptr[UInt(i) * UInt(n) + UInt(k)] * K_cross_ptr[UInt(j) * UInt(n) + UInt(k)]
    
    V_ptr[UInt(j) * UInt(r) + UInt(i)] = sum_val


fn kernel_variance_from_V(
    var_ptr: UnsafePointer[Float32, MutAnyOrigin],    # m output
    V_ptr: UnsafePointer[Float32, MutAnyOrigin],      # r × m (column-major)
    r: Int,
    m: Int,
    outputscale: Float32
) -> None:
    """Compute variance = outputscale - ||V[:, j]||² for each test point j.
    
    V is r×m in column-major: V[i, j] = V_ptr[j * r + i]
    var[j] = outputscale - sum_i V[i, j]²
    """
    var j = block_idx.x * block_dim.x + thread_idx.x
    
    if j >= UInt(m):
        return
    
    var sum_sq = Float32(0.0)
    for i in range(r):
        var v_ij = V_ptr[UInt(j) * UInt(r) + UInt(i)]
        sum_sq += v_ij * v_ij
    
    var_ptr[j] = outputscale - sum_sq


fn kernel_KcrossT_matmul_alpha(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],     # m × T output (row-major)
    K_cross_ptr: UnsafePointer[Float32, MutAnyOrigin], # n × m (column-major)
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],   # n × T (row-major)
    n: Int,
    m: Int,
    T: Int
) -> None:
    """Compute out = K_cross^T @ alpha_all where K_cross is [n×m] col-major, alpha is [n×T] row-major.
    
    Output out is [m × T] row-major: out[j, t] = sum_i K_cross[i, j] * alpha[i, t]
    
    Each thread computes one element out[j, t].
    Grid should be (ceil(m*T / BLOCK_SIZE),).
    
    Memory layout:
    - K_cross[i, j] = K_cross_ptr[j * n + i]  (column-major)
    - alpha[i, t] = alpha_ptr[i * T + t]       (row-major)
    - out[j, t] = out_ptr[j * T + t]           (row-major)
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = UInt(m) * UInt(T)
    
    if idx >= total:
        return
    
    var j = Int(idx // UInt(T))  # test point index
    var t = Int(idx % UInt(T))   # task index
    
    var sum_val = Float32(0.0)
    for i in range(n):
        sum_val += K_cross_ptr[UInt(j) * UInt(n) + UInt(i)] * alpha_ptr[UInt(i) * UInt(T) + UInt(t)]
    
    out_ptr[idx] = sum_val


fn kernel_vnorm_sq_from_ST_Kcross(
    vnorm_sq_ptr: UnsafePointer[Float32, MutAnyOrigin],  # m output
    V_ptr: UnsafePointer[Float32, MutAnyOrigin],          # r × m (column-major)
    r: Int,
    m: Int,
) -> None:
    """Compute vnorm_sq[j] = ||V[:, j]||^2 for each test point j.
    
    V is r×m in column-major: V[i, j] = V_ptr[j * r + i]
    
    This is similar to kernel_variance_from_V but only computes the squared norm
    without subtracting from outputscale, for use in LMC variance where the 
    prior/reduction weighting is task-dependent.
    """
    var j = block_idx.x * block_dim.x + thread_idx.x
    
    if j >= UInt(m):
        return
    
    var sum_sq = Float32(0.0)
    for i in range(r):
        var v_ij = V_ptr[UInt(j) * UInt(r) + UInt(i)]
        sum_sq += v_ij * v_ij
    
    vnorm_sq_ptr[j] = sum_sq


fn kernel_compute_mean(
    mean_ptr: UnsafePointer[Float32, MutAnyOrigin],    # m output
    K_cross_ptr: UnsafePointer[Float32, MutAnyOrigin], # n × m (column-major)
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],   # n
    n: Int,
    m: Int
) -> None:
    """Compute mean[j] = K_cross[:, j]^T @ alpha.
    
    K_cross is n × m in column-major: K_cross[i, j] = K_cross_ptr[j * n + i]
    """
    var j = block_idx.x * block_dim.x + thread_idx.x
    
    if j >= UInt(m):
        return
    
    var sum_val = Float32(0.0)
    for i in range(n):
        sum_val += K_cross_ptr[UInt(j) * UInt(n) + UInt(i)] * alpha_ptr[i]
    
    mean_ptr[j] = sum_val


# =============================================================================
# Prediction Result
# =============================================================================

struct PredictionResult(Copyable):
    """Result from prediction.
    
    Fields:
        mean_host: Predicted means [m]
        var_host: Predicted variances [m]
    """
    var mean_host: HostBuffer[float_dtype]
    var var_host: HostBuffer[float_dtype]
    
    fn __init__(out self, mean_host: HostBuffer[float_dtype], var_host: HostBuffer[float_dtype]):
        self.mean_host = mean_host
        self.var_host = var_host


# =============================================================================
# Helper Functions
# =============================================================================

fn _build_kernel_params[T: MatvecProvider](
    provider: T,
) raises -> KernelParams:
    """Build KernelParams from provider settings. Shared by host and device cross-covariance helpers."""
    from .kernel_params import (
        make_rbf_params,
        make_matern_params,
        make_periodic_params,
        make_rq_params,
        make_linear_params,
        make_polynomial_params,
    )
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

    var kernel_type = provider.get_kernel_type()
    var lengthscale = provider.get_lengthscale()
    var outputscale = provider.get_outputscale()
    var kernel_param1 = provider.get_kernel_param1()
    var kernel_param2 = provider.get_kernel_param2()
    var use_ard = provider.get_use_ard()
    var lengthscales_ptr = provider.get_lengthscales_device_ptr()
    var inv_ls_ptr = provider.get_inv_ls_device_ptr()

    if kernel_type == KERNEL_TYPE_RBF:
        return make_rbf_params(outputscale, lengthscale, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN12:
        return make_matern_params(outputscale, lengthscale, Float32(0.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN32:
        return make_matern_params(outputscale, lengthscale, Float32(1.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN52:
        return make_matern_params(outputscale, lengthscale, Float32(2.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_PERIODIC:
        return make_periodic_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_RQ:
        return make_rq_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_LINEAR:
        return make_linear_params(outputscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        return make_polynomial_params(outputscale, kernel_param1, kernel_param2, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    else:
        raise Error("Unknown kernel type: " + String(kernel_type))


fn compute_cross_covariance_with_provider[T: MatvecProvider](
    provider: T,
    x_test_host: HostBuffer[float_dtype],
    m: Int,
) raises -> HostBuffer[float_dtype]:
    """Compute cross-covariance K_cross[i, j] = k(x_train[i], x_test[j]).
    
    Args:
        provider: Provider for kernel computations
        x_test_host: Test points [m, d], row-major
        m: Number of test points
        
    Returns:
        K_cross: Cross-covariance matrix [n × m], column-major
        
    Note:
        Uses a fused GPU kernel to compute all n × m elements in a single kernel
        launch and avoid per-test-column launch/synchronization overhead.
    """
    from .cross_covariance_provider import compute_cross_covariance_fused
    
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var d = provider.get_d()
    var kernel_type = provider.get_kernel_type()
    var x_train_ptr = provider.get_x_ptr()
    
    var K_cross_host = ctx.enqueue_create_host_buffer[float_dtype](n * m)
    
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](m * d)
    ctx.enqueue_copy(dst_buf=x_test_device, src_buf=x_test_host)
    
    var K_cross_device = ctx.enqueue_create_buffer[float_dtype](n * m)
    ctx.synchronize()
    
    var params = _build_kernel_params(provider)
    
    compute_cross_covariance_fused(
        ctx, K_cross_device,
        x_train_ptr,
        x_test_device.unsafe_ptr(),
        n, m, d, kernel_type, params
    )
    ctx.synchronize()
    
    ctx.enqueue_copy(dst_buf=K_cross_host, src_buf=K_cross_device)
    ctx.synchronize()
    
    _ = x_test_device
    _ = K_cross_device
    
    return K_cross_host


fn compute_cross_covariance_device_with_provider[T: MatvecProvider](
    provider: T,
    x_test_host: HostBuffer[float_dtype],
    m: Int,
) raises -> DeviceBuffer[float_dtype]:
    """Compute cross-covariance K_cross on GPU and keep result on device.
    
    Same as compute_cross_covariance_with_provider but returns a DeviceBuffer
    instead of copying to host. Used by GPU-accelerated LOVE variance.
    
    Args:
        provider: Provider for kernel computations
        x_test_host: Test points [m, d], row-major
        m: Number of test points
        
    Returns:
        K_cross_device: Cross-covariance matrix [n × m], column-major, on device
    """
    from .cross_covariance_provider import compute_cross_covariance_fused
    
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var d = provider.get_d()
    var kernel_type = provider.get_kernel_type()
    var x_train_ptr = provider.get_x_ptr()
    
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](m * d)
    ctx.enqueue_copy(dst_buf=x_test_device, src_buf=x_test_host)
    
    var K_cross_device = ctx.enqueue_create_buffer[float_dtype](n * m)
    ctx.synchronize()
    
    var params = _build_kernel_params(provider)
    
    compute_cross_covariance_fused(
        ctx, K_cross_device,
        x_train_ptr,
        x_test_device.unsafe_ptr(),
        n, m, d, kernel_type, params
    )
    ctx.synchronize()
    
    _ = x_test_device
    
    return K_cross_device


# =============================================================================
# Provider-Based LOVE Prediction
# =============================================================================

fn predict_mean_with_provider[T: MatvecProvider](
    provider: T,
    x_test_host: HostBuffer[float_dtype],
    alpha_device: DeviceBuffer[float_dtype],
    m: Int,
) raises -> HostBuffer[float_dtype]:
    """Predict mean using provider and precomputed alpha = K^{-1} @ y.
    
    Computes: mean(x*) = K_test_train @ alpha
    
    Args:
        provider: Provider for kernel computations
        x_test_host: Test points [m, d]
        alpha_device: Precomputed K^{-1} @ y [n]
        m: Number of test points
        
    Returns:
        mean_host: Predictive means [m]
    """
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var d = provider.get_d()
    
    # Copy test data to device
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](m * d)
    ctx.enqueue_copy(dst_buf=x_test_device, src_buf=x_test_host)
    ctx.synchronize()
    
    # Compute K_test_train @ alpha using provider
    var mean_device = cross_matvec_with_provider(
        provider,
        x_test_device.unsafe_ptr(),
        alpha_device,
        m
    )
    
    # Copy to host
    var mean_host = ctx.enqueue_create_host_buffer[float_dtype](m)
    ctx.enqueue_copy(dst_buf=mean_host, src_buf=mean_device)
    ctx.synchronize()
    
    return mean_host


fn predict_variance_love_with_provider[T: MatvecProvider](
    provider: T,
    x_test_host: HostBuffer[float_dtype],
    lanczos_root: HostBuffer[float_dtype],  # S [n × r], column-major
    lanczos_rank: Int,  # r
    m: Int,  # number of test points
) raises -> HostBuffer[float_dtype]:
    """Compute LOVE variance: var(x*) = k(x*, x*) - ||S^T @ k_cross||²
    
    This is the full LOVE (Low-rank Orthogonal decomposition for Variance Estimation)
    algorithm from the GPyTorch paper.
    
    Uses GPU kernels kernel_matmul_ST_K and kernel_variance_from_V for the heavy
    computation. Only the final m-element variance vector is copied back to host.
    
    Args:
        provider: Provider for kernel computations (any provider type)
        x_test_host: Test points [m, d], row-major
        lanczos_root: Cached Lanczos root S [n × r], column-major (on host)
        lanczos_rank: Rank r (number of Lanczos iterations)
        m: Number of test points
        
    Returns:
        var_host: Predictive variances [m]
        
    Algorithm:
        1. Compute cross-covariance K_cross [n × m] on GPU
        2. Copy S [n × r] to GPU (once per predict call, O(n*r))
        3. GPU: V = S^T @ K_cross [r × m]  (kernel_matmul_ST_K)
        4. GPU: var[j] = outputscale - ||V[:, j]||²  (kernel_variance_from_V)
        5. Copy final variance [m] to host
        
    For non-stationary kernels (Linear, Polynomial) where k(x*,x*) != outputscale,
    a CPU correction pass adjusts the per-point prior variance.
    """
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var outputscale = provider.get_outputscale()
    
    # Step 1: Compute cross-covariance K_cross [n × m] on GPU (stays on device)
    var K_cross_device = compute_cross_covariance_device_with_provider(
        provider, x_test_host, m
    )
    
    # Step 2: Copy Lanczos root S [n × r] from host to device
    var S_device = ctx.enqueue_create_buffer[float_dtype](n * lanczos_rank)
    ctx.enqueue_copy(dst_buf=S_device, src_buf=lanczos_root)
    
    # Step 3: GPU matmul V = S^T @ K_cross [r × m]
    var V_device = ctx.enqueue_create_buffer[float_dtype](lanczos_rank * m)
    ctx.synchronize()
    
    alias BLOCK_X = 16
    alias BLOCK_Y = 16
    var grid_x = (lanczos_rank + BLOCK_X - 1) // BLOCK_X
    var grid_y = (m + BLOCK_Y - 1) // BLOCK_Y
    
    ctx.enqueue_function[kernel_matmul_ST_K](
        S_device.unsafe_ptr(), K_cross_device.unsafe_ptr(), V_device.unsafe_ptr(),
        n, lanczos_rank, m,
        grid_dim=(grid_x, grid_y), block_dim=(BLOCK_X, BLOCK_Y)
    )
    
    # Step 4: GPU variance = outputscale - ||V[:, j]||² for each test point
    var var_device = ctx.enqueue_create_buffer[float_dtype](m)
    
    alias BLOCK_VAR = 256
    var grid_var = (m + BLOCK_VAR - 1) // BLOCK_VAR
    
    ctx.enqueue_function[kernel_variance_from_V](
        var_device.unsafe_ptr(), V_device.unsafe_ptr(),
        lanczos_rank, m, outputscale,
        grid_dim=(grid_var,), block_dim=(BLOCK_VAR,)
    )
    ctx.synchronize()
    
    # Step 5: Copy final variance [m] to host
    var var_host = ctx.enqueue_create_host_buffer[float_dtype](m)
    ctx.enqueue_copy(dst_buf=var_host, src_buf=var_device)
    ctx.synchronize()
    
    # Step 6: Handle non-stationary kernels where k(x*, x*) != outputscale
    # The GPU kernel used outputscale as k(x*, x*). For Linear and Polynomial
    # kernels, we correct each point: var[j] += (k(x*_j, x*_j) - outputscale)
    var kernel_type = provider.get_kernel_type()
    from .constants import KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
    
    if kernel_type == KERNEL_TYPE_LINEAR or kernel_type == KERNEL_TYPE_POLYNOMIAL:
        var d = provider.get_d()
        var kernel_param1 = provider.get_kernel_param1()
        var kernel_param2 = provider.get_kernel_param2()
        
        for j in range(m):
            var norm_sq = Float32(0.0)
            for dd in range(d):
                var val = x_test_host[j * d + dd]
                norm_sq += val * val
            
            var k_star_star: Float32
            if kernel_type == KERNEL_TYPE_LINEAR:
                k_star_star = outputscale * (norm_sq + kernel_param1)
            else:
                k_star_star = outputscale * ((norm_sq + kernel_param2) ** kernel_param1)
            
            # Correct: var[j] was computed as (outputscale - ||V[:,j]||²)
            # Should be (k_star_star - ||V[:,j]||²) = var[j] + (k_star_star - outputscale)
            var_host[j] = var_host[j] + (k_star_star - outputscale)
    
    # Clamp to small positive value (numerical stability)
    for j in range(m):
        if var_host[j] < Float32(1e-10):
            var_host[j] = Float32(1e-10)
    
    # Keep device buffers alive until all GPU work is done
    _ = K_cross_device
    _ = S_device
    _ = V_device
    _ = var_device
    
    return var_host


fn predict_variance_exact_with_provider[T: MatvecProvider, P: ForwardProvider](
    provider: T,
    adapter: P,
    x_test_host: HostBuffer[float_dtype],
    precond: PivotedCholeskyPrecond,
    mut pool: CGBufferPool,
    m: Int,
) raises -> HostBuffer[float_dtype]:
    """Compute exact predictive variance via CG solve.
    
    var(x*_j) = k(x*_j, x*_j) - K_cross[:,j]^T @ (K + sigma^2 I)^{-1} @ K_cross[:,j]
    
    This matches GPyTorch's default behavior (without fast_pred_var).
    Uses batched CG with Pivoted Cholesky preconditioning to solve
    (K + sigma^2 I) @ V = K_cross for all m test points simultaneously.
    
    Args:
        provider: MatvecProvider for cross-covariance computation
        adapter: ForwardProvider (adapter-wrapped provider) for CG solve
        x_test_host: Test points [m, d], row-major
        precond: Pre-built Pivoted Cholesky preconditioner
        pool: CG buffer pool (will be resized if needed)
        m: Number of test points
        
    Returns:
        var_host: Predictive variances [m]
    """
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var outputscale = provider.get_outputscale()
    
    # Step 1: Compute cross-covariance K_cross [n × m] on host (column-major)
    var K_cross_host = compute_cross_covariance_with_provider(provider, x_test_host, m)
    
    # Step 2: Copy K_cross to device as RHS for batched CG
    var K_cross_device = ctx.enqueue_create_buffer[float_dtype](n * m)
    ctx.enqueue_copy(dst_buf=K_cross_device, src_buf=K_cross_host)
    ctx.synchronize()
    
    # Step 3: Ensure CG buffer pool has capacity for m columns
    pool.ensure_capacity(ctx, n, m, 0, 0, rank=10)
    
    # Step 4: Solve (K + sigma^2 I) @ V = K_cross using batched CG
    var cg_result = batched_cg_unified(
        adapter, K_cross_device.unsafe_ptr(), n, m,
        100,  # max_iter
        0,    # max_tridiag_iter=0 (no log-det needed)
        Float32(1e-3),  # tol
        precond, pool
    )
    
    
    # Step 5: Copy solution V [n × m] to host
    var V_host = ctx.enqueue_create_host_buffer[float_dtype](n * m)
    ctx.enqueue_copy(dst_buf=V_host, src_buf=cg_result.solution)
    ctx.synchronize()
    
    
    # Step 6: Compute variance for each test point
    # var(x*_j) = k(x*_j, x*_j) - dot(K_cross[:,j], V[:,j])
    # For stationary kernels, k(x*, x*) = outputscale
    # For non-stationary kernels (Linear, Polynomial), it depends on x*
    
    # Get kernel type and dimension for computing k(x*, x*)
    var d = provider.get_d()
    var kernel_type = provider.get_kernel_type()
    var kernel_param1 = provider.get_kernel_param1()
    var kernel_param2 = provider.get_kernel_param2()
    
    from .constants import (
        KERNEL_TYPE_LINEAR,
        KERNEL_TYPE_POLYNOMIAL,
    )
    
    var var_host = ctx.enqueue_create_host_buffer[float_dtype](m)
    for j in range(m):
        var dot_product = Float32(0.0)
        for i in range(n):
            # K_cross and V are column-major: column j starts at j * n
            dot_product += K_cross_host[j * n + i] * V_host[j * n + i]
        
        # Compute k(x*_j, x*_j) based on kernel type
        var k_star_star: Float32
        
        if kernel_type == KERNEL_TYPE_LINEAR:
            # k(x*, x*) = outputscale * (||x*||^2 + bias)
            var norm_sq = Float32(0.0)
            for dd in range(d):
                var val = x_test_host[j * d + dd]
                norm_sq += val * val
            k_star_star = outputscale * (norm_sq + kernel_param1)
        
        elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
            # k(x*, x*) = outputscale * (||x*||^2 + offset)^degree
            var norm_sq = Float32(0.0)
            for dd in range(d):
                var val = x_test_host[j * d + dd]
                norm_sq += val * val
            k_star_star = outputscale * ((norm_sq + kernel_param2) ** kernel_param1)
        
        else:
            # Stationary kernels: k(x*, x*) = outputscale
            k_star_star = outputscale
        
        # var(x*_j) = k(x*_j, x*_j) - dot_product
        var variance = k_star_star - dot_product
        
        # Clamp to small positive value (numerical stability)
        if variance < Float32(1e-10):
            variance = Float32(1e-10)
        
        var_host[j] = variance
    
    return var_host


fn predict_with_provider[T: MatvecProvider](
    provider: T,
    x_test_host: HostBuffer[float_dtype],
    alpha_device: DeviceBuffer[float_dtype],
    lanczos_root: HostBuffer[float_dtype],  # NEW: Required for LOVE variance
    lanczos_rank: Int,  # NEW: Required
    m: Int,
) raises -> PredictionResult:
    """Predict both mean and variance using provider.
    
    Uses LOVE variance (not prior variance).
    
    Args:
        provider: Provider for kernel computations
        x_test_host: Test points [m, d]
        alpha_device: Precomputed K^{-1} @ y [n]
        lanczos_root: Cached Lanczos root S [n × r]
        lanczos_rank: Rank r
        m: Number of test points
        
    Returns:
        PredictionResult with mean_host and var_host
    """
    var mean_host = predict_mean_with_provider(
        provider, x_test_host, alpha_device, m
    )
    var var_host = predict_variance_love_with_provider(
        provider, x_test_host, lanczos_root, lanczos_rank, m
    )
    
    return PredictionResult(mean_host, var_host)
