"""Lanczos Tridiagonalization and SLQ Log-Determinant Estimation.

This module provides:
- Lanczos algorithm for tridiagonalizing kernel matrices
- Stochastic Lanczos Quadrature (SLQ) for log-determinant estimation
- Tridiagonal eigendecomposition (via native QR in native_numerics.mojo)

The Lanczos algorithm computes a tridiagonal matrix T such that V^T @ K @ V = T,
where V is an orthonormal basis. This is used for:
1. Log-determinant estimation via SLQ
2. Eigenvalue approximation
3. Matrix function approximation

Note: For training, log-det is now computed via CG tridiagonals (see
combined_inv_quad_logdet.mojo). The standalone Lanczos log-det functions
have been removed. This module retains Lanczos tridiagonalization for
LOVE variance computation and the SLQ formula for computing log-det
from pre-computed tridiagonal matrices.

Supports all kernel types (RBF, Matérn 1/2, 3/2, 5/2, Periodic, RQ, Linear, Polynomial)
with both isotropic and ARD parameterizations.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt, log
from random import random_float64

from .constants import (
    float_dtype,
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
from .native_numerics import tridiagonal_eigh_native, compute_logdet_from_tridiag_batched_native

# =============================================================================
# Constants
# =============================================================================

alias LANCZOS_BREAKDOWN_TOL = Float32(1e-10)  # Tolerance for Lanczos breakdown

fn compute_kernel_matvec_batched(
    out_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ctx: DeviceContext,
    x_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    dim: Int,
    num_cols: Int,
    lengthscale: Float32,
    noise: Float32,
    outputscale: Float32,
    kernel_type: Int,
    kernel_param1: Float32 = 1.0,
    kernel_param2: Float32 = 0.0,
) raises:
    """Dispatcher for kernel matvec (isotropic kernels).
    
    Supports all kernel types.
    """
    # Create KernelParams based on kernel type
    var params: KernelParams
    var lengthscales_ptr = UnsafePointer[Float32, MutAnyOrigin]()  # Null for isotropic
    
    if kernel_type == KERNEL_TYPE_RBF:
        params = make_rbf_params(outputscale, lengthscale, lengthscales_ptr, is_ard=False)
    elif kernel_type == KERNEL_TYPE_MATERN12:
        params = make_matern_params(outputscale, lengthscale, Float32(0.5), lengthscales_ptr, is_ard=False)
    elif kernel_type == KERNEL_TYPE_MATERN32:
        params = make_matern_params(outputscale, lengthscale, Float32(1.5), lengthscales_ptr, is_ard=False)
    elif kernel_type == KERNEL_TYPE_MATERN52:
        params = make_matern_params(outputscale, lengthscale, Float32(2.5), lengthscales_ptr, is_ard=False)
    elif kernel_type == KERNEL_TYPE_PERIODIC:
        params = make_periodic_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, is_ard=False)
    elif kernel_type == KERNEL_TYPE_RQ:
        params = make_rq_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, is_ard=False)
    elif kernel_type == KERNEL_TYPE_LINEAR:
        params = make_linear_params(outputscale, kernel_param1, lengthscales_ptr, is_ard=False)
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        params = make_polynomial_params(outputscale, kernel_param1, kernel_param2, lengthscales_ptr, is_ard=False)
    else:
        raise Error("Unsupported kernel type in Lanczos: " + String(kernel_type))
    
    # Call unified dispatcher
    unified_dispatch_forward_matvec(
        ctx, kernel_type, out_device_ptr, x_device_ptr, v_device_ptr,
        n, dim, num_cols, params, noise
    )

# =============================================================================
# Host Function: Tridiagonal Eigendecomposition (CPU)
# =============================================================================

fn tridiagonal_eigh(
    alpha: UnsafePointer[Float32, MutAnyOrigin],
    beta: UnsafePointer[Float32, MutAnyOrigin],
    m: Int,
    eigenvalues_out: UnsafePointer[Float32, MutAnyOrigin],
    eigenvectors_out: UnsafePointer[Float32, MutAnyOrigin]
) raises:
    """Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix T.
    
    Uses Float32 QR algorithm (tridiagonal_eigh_native). The inputs are Float32
    from Lanczos iteration so Float64 eigendecomp gives no measurable benefit.
    
    Args:
        alpha: Diagonal elements [m] (Float32)
        beta: Off-diagonal elements [m-1] (Float32)
        m: Size of matrix
        eigenvalues_out: Output eigenvalues [m] (sorted ascending, Float32)
        eigenvectors_out: Output eigenvector matrix [m x m] (row-major, Float32)
    """
    tridiagonal_eigh_native(alpha, beta, m, eigenvalues_out, eigenvectors_out)


# =============================================================================
# Host Function: Lanczos Tridiagonalization
# =============================================================================

fn lanczos_tridiagonalization(
    ctx: DeviceContext,
    x_device: DeviceBuffer[float_dtype],
    v0_host: HostBuffer[float_dtype],
    n: Int,
    dim: Int,
    lengthscale: Float32,
    noise: Float32,
    outputscale: Float32,
    lanczos_iter: Int,
    alpha_out: UnsafePointer[Float32, MutAnyOrigin],
    beta_out: UnsafePointer[Float32, MutAnyOrigin],
    kernel_type: Int,
    kernel_param1: Float32 = 1.0,
    kernel_param2: Float32 = 0.0
) raises:
    """Lanczos algorithm to compute tridiagonal matrix T such that V^T @ K @ V = T.
    
    This is the basic Lanczos algorithm for a single probe vector. For multiple
    probes, use the batched version for better GPU utilization.
    
    Args:
        ctx: GPU device context
        x_device: Training points on device [n × dim]
        v0_host: Initial vector (probe) on host [n]
        n: Number of points
        dim: Dimensionality
        lengthscale: Kernel lengthscale (isotropic kernels)
        noise: Observation noise
        outputscale: Output scale
        lanczos_iter: Number of Lanczos iterations
        alpha_out: Output diagonal elements [lanczos_iter]
        beta_out: Output off-diagonal elements [lanczos_iter-1]
        kernel_type: Kernel type constant (KERNEL_TYPE_RBF, etc.)
    
    Algorithm:
        1. Normalize initial vector v0
        2. For j = 0 to lanczos_iter-1:
           a. Compute w = K @ v_curr
           b. Compute alpha[j] = v_curr^T @ w
           c. Orthogonalize: w = w - alpha[j] * v_curr - beta[j-1] * v_prev
           d. Compute beta[j] = ||w||
           e. Update: v_prev = v_curr, v_curr = w / beta[j]
        3. Return tridiagonal matrix T with diag(alpha) and off-diag(beta)
    
    Notes:
        - If beta[j] < LANCZOS_BREAKDOWN_TOL, the algorithm has converged early
        - The Lanczos vectors V = [v0, v1, ..., v_{m-1}] satisfy V^T @ V = I
        - The tridiagonal matrix T satisfies V^T @ K @ V = T
    """
    
    # Normalize v0
    var v0_norm_sq = Float32(0.0)
    for i in range(n):
        v0_norm_sq += v0_host[i] * v0_host[i]
    var v0_norm = sqrt(v0_norm_sq)
    
    # Allocate device buffers for Lanczos vectors
    var v_curr_device = ctx.enqueue_create_buffer[float_dtype](n)
    var v_prev_device = ctx.enqueue_create_buffer[float_dtype](n)
    var w_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Allocate host buffers
    var v_curr_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var v_prev_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var w_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    
    ctx.synchronize()
    
    # Initialize v_curr = v0 / ||v0||
    for i in range(n):
        v_curr_host[i] = v0_host[i] / v0_norm
        v_prev_host[i] = Float32(0.0)
    
    ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
    ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
    ctx.synchronize()
    
    var beta_prev = Float32(0.0)
    
    for j in range(lanczos_iter):
        # Compute w = K @ v_curr (using template-dispatched kernel)
        compute_kernel_matvec_batched(
            w_device.unsafe_ptr(), ctx, x_device.unsafe_ptr(), v_curr_device.unsafe_ptr(),
            n, dim, 1, lengthscale, noise, outputscale, kernel_type,
            kernel_param1, kernel_param2
        )
        
        ctx.enqueue_copy(dst_buf=w_host, src_buf=w_device)
        ctx.synchronize()
        
        # Compute alpha[j] = v_curr^T @ w
        var alpha_j = Float32(0.0)
        for i in range(n):
            alpha_j += v_curr_host[i] * w_host[i]
        alpha_out[j] = alpha_j
        
        # w = w - alpha[j] * v_curr
        for i in range(n):
            w_host[i] -= alpha_j * v_curr_host[i]
        
        # w = w - beta[j-1] * v_prev (if j > 0)
        if j > 0:
            for i in range(n):
                w_host[i] -= beta_prev * v_prev_host[i]
        
        # Compute beta[j] = ||w||
        var beta_j_sq = Float32(0.0)
        for i in range(n):
            beta_j_sq += w_host[i] * w_host[i]
        var beta_j = sqrt(beta_j_sq)
        
        if j < lanczos_iter - 1:
            beta_out[j] = beta_j
        
        # Check for convergence (Lanczos breakdown)
        if beta_j < LANCZOS_BREAKDOWN_TOL:
            # Fill remaining entries with zeros
            for k in range(j + 1, lanczos_iter):
                alpha_out[k] = Float32(0.0)
                if k < lanczos_iter - 1:
                    beta_out[k] = Float32(0.0)
            break
        
        # Update vectors: v_prev = v_curr, v_curr = w / beta_j
        for i in range(n):
            v_prev_host[i] = v_curr_host[i]
            v_curr_host[i] = w_host[i] / beta_j
        
        ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
        ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
        ctx.synchronize()
        
        beta_prev = beta_j


# =============================================================================
# Host Function: SLQ Log-Determinant (Isotropic Kernels)
# =============================================================================

fn compute_lanczos_root(
    ctx: DeviceContext,
    x_device: DeviceBuffer[float_dtype],
    n: Int,
    dim: Int,
    lengthscale: Float32,
    noise: Float32,
    outputscale: Float32,
    lanczos_iter: Int,
    kernel_type: Int
) raises -> HostBuffer[float_dtype]:
    """Compute Lanczos root S = Q @ T^{-1/2} for LOVE variance.
    
    This function runs Lanczos decomposition to get Q and T, then computes
    the matrix square root T^{-1/2} via eigendecomposition, and finally
    computes S = Q @ T^{-1/2}.
    
    Args:
        ctx: GPU device context
        x_device: Training points on device [n × dim]
        n: Number of points
        dim: Dimensionality
        lengthscale: Kernel lengthscale
        noise: Observation noise
        outputscale: Output scale
        lanczos_iter: Number of Lanczos iterations (rank r)
        kernel_type: Kernel type constant
    
    Returns:
        S_host: Lanczos root [n × r] in column-major format
        
    Algorithm:
        1. Run Lanczos to get Q [n × r] and T [r × r]
        2. Compute eigendecomposition: T = V @ Λ @ V^T
        3. Compute T^{-1/2} = V @ Λ^{-1/2} @ V^T
        4. Compute S = Q @ T^{-1/2}
    
    Notes:
        - Uses a single probe vector (can be extended to multiple probes)
        - S is used for fast O(r) variance prediction
        - Typical r = 15-50 for good approximation
    """
    var r = lanczos_iter
    
    # Generate probe vector (Rademacher: random ±1)
    var z_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    for i in range(n):
        var rand_val = random_float64()
        z_host[i] = Float32(1.0) if rand_val > 0.5 else Float32(-1.0)
    
    # Allocate buffers for tridiagonal matrix
    var alpha = List[Float32](capacity=r)
    var beta = List[Float32](capacity=r - 1)
    for i in range(r):
        alpha.append(Float32(0.0))
    for i in range(r - 1):
        beta.append(Float32(0.0))
    
    # Allocate storage for Lanczos vectors Q [n × r] column-major
    var Q_host = ctx.enqueue_create_host_buffer[float_dtype](n * r)
    
    # Modified Lanczos that also stores Q vectors
    # Normalize v0
    var v0_norm_sq = Float32(0.0)
    for i in range(n):
        v0_norm_sq += z_host[i] * z_host[i]
    var v0_norm = sqrt(v0_norm_sq)
    
    # Allocate device buffers for Lanczos vectors
    var v_curr_device = ctx.enqueue_create_buffer[float_dtype](n)
    var v_prev_device = ctx.enqueue_create_buffer[float_dtype](n)
    var w_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Allocate host buffers
    var v_curr_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var v_prev_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var w_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    
    ctx.synchronize()
    
    # Initialize v_curr = v0 / ||v0||
    for i in range(n):
        v_curr_host[i] = z_host[i] / v0_norm
        v_prev_host[i] = Float32(0.0)
    
    # Store first Lanczos vector in Q[:, 0]
    for i in range(n):
        Q_host[0 * n + i] = v_curr_host[i]  # Column 0
    
    ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
    ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
    ctx.synchronize()
    
    var beta_prev = Float32(0.0)
    
    for j in range(r):
        # Compute w = K @ v_curr
        compute_kernel_matvec_batched(
            w_device.unsafe_ptr(), ctx, x_device.unsafe_ptr(), v_curr_device.unsafe_ptr(),
            n, dim, 1, lengthscale, noise, outputscale, kernel_type
        )
        
        ctx.enqueue_copy(dst_buf=w_host, src_buf=w_device)
        ctx.synchronize()
        
        # Compute alpha[j] = v_curr^T @ w
        var alpha_j = Float32(0.0)
        for i in range(n):
            alpha_j += v_curr_host[i] * w_host[i]
        alpha[j] = alpha_j
        
        # w = w - alpha[j] * v_curr
        for i in range(n):
            w_host[i] -= alpha_j * v_curr_host[i]
        
        # w = w - beta[j-1] * v_prev (if j > 0)
        if j > 0:
            for i in range(n):
                w_host[i] -= beta_prev * v_prev_host[i]
        
        # Compute beta[j] = ||w||
        var beta_j_sq = Float32(0.0)
        for i in range(n):
            beta_j_sq += w_host[i] * w_host[i]
        var beta_j = sqrt(beta_j_sq)
        
        if j < r - 1:
            beta[j] = beta_j
        
        # Check for convergence
        if beta_j < LANCZOS_BREAKDOWN_TOL:
            # Fill remaining Q columns with zeros
            for k in range(j + 1, r):
                for i in range(n):
                    Q_host[k * n + i] = Float32(0.0)
            break
        
        # Update vectors: v_prev = v_curr, v_curr = w / beta_j
        for i in range(n):
            v_prev_host[i] = v_curr_host[i]
            v_curr_host[i] = w_host[i] / beta_j
        
        # Store Lanczos vector in Q[:, j+1]
        if j + 1 < r:
            for i in range(n):
                Q_host[(j + 1) * n + i] = v_curr_host[i]
        
        ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
        ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
        ctx.synchronize()
        
        beta_prev = beta_j
    
    # Now compute T^{-1/2} via eigendecomposition (Float64 for numerical stability)
    var eigenvalues = List[Float32](capacity=r)
    var eigenvectors = List[Float32](capacity=r * r)
    for i in range(r):
        eigenvalues.append(Float32(0.0))
    for i in range(r * r):
        eigenvectors.append(Float32(0.0))
    
    # Compute eigendecomposition of tridiagonal T (Float32)
    tridiagonal_eigh_native(
        alpha.unsafe_ptr(), beta.unsafe_ptr(), r,
        eigenvalues.unsafe_ptr(), eigenvectors.unsafe_ptr()
    )
    
    # Compute T^{-1/2} = V @ Λ^{-1/2} @ V^T
    # Eigenvectors are row-major: V[i, j] = eigenvectors[i * r + j]
    # Relative eigenvalue clamping (matches GPyTorch): clamp at max_eig * 1e-6
    # instead of absolute 1e-10, so the threshold scales with the spectrum
    var max_eig = Float32(1e-10)  # floor to avoid zero threshold
    for i in range(r):
        if eigenvalues[i] > max_eig:
            max_eig = eigenvalues[i]
    var eig_clamp = max_eig * Float32(1e-6)

    # First compute V @ Λ^{-1/2}: multiply each column j by λ_j^{-1/2}
    var V_Linv_sqrt = List[Float32](capacity=r * r)
    for i in range(r * r):
        V_Linv_sqrt.append(Float32(0.0))
    
    for i in range(r):
        for j in range(r):
            var lambda_j = eigenvalues[j]
            # Clamp to avoid division by zero
            if lambda_j < eig_clamp:
                lambda_j = eig_clamp
            var lambda_inv_sqrt = Float32(1.0) / sqrt(lambda_j)
            # V @ Λ^{-1/2}: multiply column j by λ_j^{-1/2}
            # V[i, j] in row-major: eigenvectors[i * r + j]
            V_Linv_sqrt[i * r + j] = eigenvectors[i * r + j] * lambda_inv_sqrt
    
    # Compute T_inv_sqrt = (V @ Λ^{-1/2}) @ V^T
    # Result[i, j] = sum_k (V @ Λ^{-1/2})[i, k] * V[j, k]
    var T_inv_sqrt = List[Float32](capacity=r * r)
    for i in range(r * r):
        T_inv_sqrt.append(Float32(0.0))
    
    for i in range(r):
        for j in range(r):
            var sum_val = Float32(0.0)
            for k in range(r):
                # (V @ Λ^{-1/2})[i, k] = V_Linv_sqrt[i * r + k]
                # V^T[k, j] = V[j, k] = eigenvectors[j * r + k]
                sum_val += V_Linv_sqrt[i * r + k] * eigenvectors[j * r + k]
            T_inv_sqrt[i * r + j] = sum_val
    
    # Finally compute S = Q @ T^{-1/2} [n × r]
    var S_host = ctx.enqueue_create_host_buffer[float_dtype](n * r)
    
    for col in range(r):
        for row in range(n):
            var sum_val = Float32(0.0)
            for k in range(r):
                # Q[row, k] in column-major: Q_host[k * n + row]
                # T_inv_sqrt[k, col]: T_inv_sqrt[k * r + col]
                sum_val += Q_host[k * n + row] * T_inv_sqrt[k * r + col]
            # S[row, col] in column-major: S_host[col * n + row]
            S_host[col * n + row] = sum_val
    
    return S_host


fn slq_log_determinant(
    ctx: DeviceContext,
    x_device: DeviceBuffer[float_dtype],
    n: Int,
    dim: Int,
    lengthscale: Float32,
    noise: Float32,
    outputscale: Float32,
    num_probes: Int,
    lanczos_iter: Int,
    kernel_type: Int,
    kernel_param1: Float32 = 1.0,
    kernel_param2: Float32 = 0.0
) raises -> Float32:
    """Compute log|K| using Stochastic Lanczos Quadrature (SLQ).
    
    This implements the same algorithm as GPyTorch's linear_operator library:
    1. Run Lanczos to get tridiagonal matrix T
    2. Compute eigendecomposition of T: T = V @ diag(λ) @ V^T
    3. Use SLQ formula: log|K| ≈ n * sum_i (V[0,i])^2 * log(λ_i)
    
    Args:
        ctx: GPU device context
        x_device: Training points on device [n × dim]
        n: Number of points
        dim: Dimensionality
        lengthscale: Kernel lengthscale (isotropic kernels)
        noise: Observation noise
        outputscale: Output scale
        num_probes: Number of probe vectors (more probes = better accuracy)
        lanczos_iter: Number of Lanczos iterations (typically 10-50)
        kernel_type: Kernel type constant (KERNEL_TYPE_RBF, etc.)
    
    Returns:
        Approximation of log|K + noise*I|
    
    Notes:
        - Accuracy improves with more probes and iterations
        - Typical settings: num_probes=10, lanczos_iter=15
        - For n=1000, this is ~100x faster than Cholesky decomposition
        - Uses Rademacher probe vectors (random ±1)
    """
    
    var log_det_sum = Float32(0.0)
    
    # Allocate buffers for Lanczos output
    var alpha = List[Float32](capacity=lanczos_iter)
    var beta = List[Float32](capacity=lanczos_iter - 1)
    var eigenvalues = List[Float32](capacity=lanczos_iter)
    var eigenvectors = List[Float32](capacity=lanczos_iter * lanczos_iter)
    
    for i in range(lanczos_iter):
        alpha.append(Float32(0.0))
        eigenvalues.append(Float32(0.0))
    for i in range(lanczos_iter - 1):
        beta.append(Float32(0.0))
    for i in range(lanczos_iter * lanczos_iter):
        eigenvectors.append(Float32(0.0))
    
    for probe in range(num_probes):
        # Generate random probe vector (Rademacher: random ±1)
        var z_host = ctx.enqueue_create_host_buffer[float_dtype](n)
        for i in range(n):
            var rand_val = random_float64()
            z_host[i] = Float32(1.0) if rand_val > 0.5 else Float32(-1.0)
        
        # Run Lanczos tridiagonalization
        lanczos_tridiagonalization(
            ctx, x_device, z_host, n, dim,
            lengthscale, noise, outputscale,
            lanczos_iter, alpha.unsafe_ptr(), beta.unsafe_ptr(), kernel_type,
            kernel_param1, kernel_param2
        )
        
        # Compute eigenvalues and eigenvectors of tridiagonal matrix (CPU, Float64)
        # Compute eigendecomposition of tridiagonal matrix (Float32)
        tridiagonal_eigh_native(
            alpha.unsafe_ptr(), beta.unsafe_ptr(), lanczos_iter,
            eigenvalues.unsafe_ptr(), eigenvectors.unsafe_ptr()
        )
        
        # SLQ formula: log|K| ≈ n * sum_i (V[0,i])^2 * log(λ_i)
        # where V[0,i] is the first component of the i-th eigenvector
        # and λ_i is the i-th eigenvalue
        var log_det_probe = Float32(0.0)
        for i in range(lanczos_iter):
            # Get first component of i-th eigenvector: V[0, i] = eigenvectors[0 * m + i]
            var v0i = eigenvectors[i]  # First row, i-th column
            var v0i_sq = v0i * v0i
            
            # Clamp eigenvalue to be positive (numerical stability)
            var lambda_i = eigenvalues[i]
            if lambda_i < Float32(1e-10):
                lambda_i = Float32(1e-10)
            
            log_det_probe += v0i_sq * log(lambda_i)
        
        # Scale by n (matrix dimension)
        log_det_probe *= Float32(n)
        log_det_sum += log_det_probe
    
    # Average over all probes
    return log_det_sum / Float32(num_probes)



fn compute_logdet_from_tridiag_batched(
    ctx: DeviceContext,
    tridiag_diag_batch: List[List[Float32]],   # [num_probes][m] diagonals
    tridiag_offdiag_batch: List[List[Float32]], # [num_probes][m-1] off-diagonals
    m: Int,
    num_probes: Int,
    n: Int,  # Original matrix size for scaling
) raises -> Float32:
    """Compute log|K| from batched tridiagonal matrices using PyTorch.
    
    This uses PyTorch's batched eigendecomposition (torch.linalg.eigh) which is
    backed by LAPACK and provides accurate results. The overhead is ~1.3ms for
    typical settings (10 probes, m=28), which is <3% of iteration time for n>1000.
    
    Algorithm:
    1. Build batch of tridiagonal matrices [num_probes × m × m]
    2. Compute batched eigendecomposition using torch.linalg.eigh()
    3. Apply SLQ formula: log|K| ≈ n * mean(sum_i V[0,i]^2 * log(λ_i))
    
    Args:
        ctx: Device context (unused, kept for API compatibility)
        tridiag_diag_batch: Diagonal elements for each probe
        tridiag_offdiag_batch: Off-diagonal elements for each probe
        m: Tridiagonal size (number of CG iterations)
        num_probes: Number of probe vectors
        n: Original matrix size (for SLQ scaling)
        
    Returns:
        Estimated log|K|
    """
    # Use native Mojo QR algorithm for batched eigendecomposition
    return compute_logdet_from_tridiag_batched_native(
        tridiag_diag_batch, tridiag_offdiag_batch, m, num_probes, n
    )



