"""Lanczos algorithm with generic provider abstraction.

This module provides:
- Tridiagonal eigendecomposition (CPU Jacobi method)
- Lanczos root computation for LOVE variance (Q @ T^{-1/2})

Note: The standalone Lanczos log-det estimation (slq_log_determinant_with_provider)
has been removed. Log-det is now computed via CG tridiagonals in
combined_inv_quad_logdet.mojo.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt
from random import random_float64

from .matvec_provider import MatvecProvider
from .native_numerics import tridiagonal_eigh_native



alias float_dtype = DType.float32


# =============================================================================
# Result Structs
# =============================================================================

struct LanczosRootResult(Copyable):
    """Result from Lanczos root computation.
    
    Fields:
        root: S matrix [n × r] where S = Q @ T^{-1/2} (column-major)
        rank: r (number of Lanczos iterations)
        n: Number of rows (training points)
    """
    var root: HostBuffer[float_dtype]
    var rank: Int
    var n: Int
    
    fn __init__(out self, root: HostBuffer[float_dtype], rank: Int, n: Int):
        self.root = root
        self.rank = rank
        self.n = n


# =============================================================================
# Constants
# =============================================================================

alias LANCZOS_BREAKDOWN_TOL = Float32(1e-10)  # Tolerance for Lanczos breakdown

# NOTE: tridiagonal_eigh_cpu (Jacobi method) and its associated constants
# (MAX_SWEEPS, EIGEN_TOL, abs_f32) were removed as dead code.
# The codebase uses tridiagonal_eigh_f64 from native_numerics.mojo instead,
# which operates in Float64 for better numerical stability.



# =============================================================================
# Host Function: Compute Lanczos Root for LOVE
# =============================================================================

fn compute_lanczos_root_with_provider[T: MatvecProvider](
    provider: T,
    lanczos_iter: Int = 20,
) raises -> LanczosRootResult:
    """Compute Lanczos root S = Q @ T^{-1/2} for LOVE variance.
    
    This function runs Lanczos decomposition to get Q and T, then computes
    the matrix square root T^{-1/2} via eigendecomposition, and finally
    computes S = Q @ T^{-1/2}.
    
    Args:
        provider: Provider for kernel matvec operations
        lanczos_iter: Number of Lanczos iterations (rank r)
        
    Returns:
        LanczosRootResult with root S [n × r] in column-major format, rank r, and n
    """
    var n = provider.get_n()
    var ctx = provider.get_ctx()
    var r = lanczos_iter
    
    # Allocate buffers
    var alpha = List[Float32](capacity=r)
    var beta = List[Float32](capacity=r - 1)
    for i in range(r):
        alpha.append(Float32(0.0))
    for i in range(r - 1):
        beta.append(Float32(0.0))
    
    # Allocate Q matrix [n × r] in column-major format
    var Q_host = ctx.enqueue_create_host_buffer[float_dtype](n * r)
    
    # Generate random probe vector
    var z_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    for i in range(n):
        var rand_val = random_float64()
        z_host[i] = Float32(1.0) if rand_val > 0.5 else Float32(-1.0)
    
    # Normalize
    var z_norm_sq = Float32(0.0)
    for i in range(n):
        z_norm_sq += z_host[i] * z_host[i]
    var z_norm = sqrt(z_norm_sq)
    for i in range(n):
        z_host[i] /= z_norm
    
    # Store first Lanczos vector in Q[:, 0]
    for i in range(n):
        Q_host[i] = z_host[i]
    
    # Allocate device buffers
    var v_curr_device = ctx.enqueue_create_buffer[float_dtype](n)
    var v_prev_device = ctx.enqueue_create_buffer[float_dtype](n)
    var w_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    var v_curr_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var v_prev_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var w_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    
    # Initialize
    for i in range(n):
        v_curr_host[i] = z_host[i]
        v_prev_host[i] = Float32(0.0)
    
    ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
    ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
    ctx.synchronize()
    
    var beta_prev = Float32(0.0)
    
    # Run Lanczos and store Q vectors
    for j in range(r):
        # Compute w = K @ v_curr
        provider.forward_matvec(w_device.unsafe_ptr(), v_curr_device.unsafe_ptr(), 1)
        ctx.enqueue_copy(dst_buf=w_host, src_buf=w_device)
        ctx.synchronize()
        
        # Compute alpha[j]
        var alpha_j = Float32(0.0)
        for i in range(n):
            alpha_j += v_curr_host[i] * w_host[i]
        alpha[j] = alpha_j
        
        # Orthogonalize
        for i in range(n):
            w_host[i] -= alpha_j * v_curr_host[i]
        if j > 0:
            for i in range(n):
                w_host[i] -= beta_prev * v_prev_host[i]
        
        # Compute beta[j]
        var beta_j_sq = Float32(0.0)
        for i in range(n):
            beta_j_sq += w_host[i] * w_host[i]
        var beta_j = sqrt(beta_j_sq)
        
        if j < r - 1:
            beta[j] = beta_j
        
        # Check for breakdown
        if beta_j < LANCZOS_BREAKDOWN_TOL:
            # Fill remaining Q columns with zeros
            for k in range(j + 1, r):
                for i in range(n):
                    Q_host[k * n + i] = Float32(0.0)
            break
        
        # Update vectors
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
    
    # Compute T^{-1/2} via eigendecomposition (Float32)
    var eigenvalues = List[Float32](capacity=r)
    var eigenvectors = List[Float32](capacity=r * r)
    for i in range(r):
        eigenvalues.append(Float32(0.0))
    for i in range(r * r):
        eigenvectors.append(Float32(0.0))
    
    tridiagonal_eigh_native(
        alpha.unsafe_ptr(), beta.unsafe_ptr(), r,
        eigenvalues.unsafe_ptr(), eigenvectors.unsafe_ptr()
    )
    
    # Compute T^{-1/2} = V @ Λ^{-1/2} @ V^T
    # Relative eigenvalue clamping (matches GPyTorch): clamp at max_eig * 1e-6
    # instead of absolute 1e-10, so the threshold scales with the spectrum
    var max_eig = Float32(1e-10)  # floor to avoid zero threshold
    for i in range(r):
        if eigenvalues[i] > max_eig:
            max_eig = eigenvalues[i]
    var eig_clamp = max_eig * Float32(1e-6)

    var V_Linv_sqrt = List[Float32](capacity=r * r)
    for i in range(r * r):
        V_Linv_sqrt.append(Float32(0.0))
    
    for i in range(r):
        for j in range(r):
            var lambda_j = eigenvalues[j]
            if lambda_j < eig_clamp:
                lambda_j = eig_clamp
            var lambda_inv_sqrt = Float32(1.0) / sqrt(lambda_j)
            V_Linv_sqrt[i * r + j] = eigenvectors[i * r + j] * lambda_inv_sqrt
    
    # Compute T_inv_sqrt = (V @ Λ^{-1/2}) @ V^T
    var T_inv_sqrt = List[Float32](capacity=r * r)
    for i in range(r * r):
        T_inv_sqrt.append(Float32(0.0))
    
    for i in range(r):
        for j in range(r):
            var sum_val = Float32(0.0)
            for k in range(r):
                sum_val += V_Linv_sqrt[i * r + k] * eigenvectors[j * r + k]
            T_inv_sqrt[i * r + j] = sum_val
    
    # Compute S = Q @ T^{-1/2} [n × r]
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
    
    return LanczosRootResult(S_host, r, n)
