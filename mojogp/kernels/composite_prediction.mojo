"""Prediction functions for composite kernels.

This module provides prediction capabilities for composite kernels:
- compute_lanczos_root_composite: Lanczos root S = Q @ T^{-1/2} for LOVE variance
- predict_mean_composite: Mean prediction via CG solve
- predict_variance_love_composite: Variance prediction using LOVE approximation
- predict_composite: Combined mean and variance prediction

These functions work with CompositeProvider[DIM, K] instead of the MatvecProvider trait.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt, log
from random import random_float64
from collections import Optional

from .composable_kernel import ComposableKernel
from .composite_provider import CompositeProvider
from .composite_matvec import composite_cross_matvec_8x, composite_extract_diagonal, composite_cross_covariance_gpu
from .native_numerics import tridiagonal_eigh_native
from .pivoted_cholesky import PivotedCholeskyPrecond



alias float_dtype = DType.float32


# =============================================================================
# Result Structs
# =============================================================================

struct LanczosRootResultComposite(Copyable):
    """Result from Lanczos root computation for composite kernels.
    
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


struct PredictionResultComposite(Copyable):
    """Result from composite kernel prediction.
    
    Fields:
        mean: Predicted mean [n_test]
        variance: Predicted variance [n_test] (if computed)
        n_test: Number of test points
    """
    var mean: HostBuffer[float_dtype]
    var variance: HostBuffer[float_dtype]
    var n_test: Int
    var has_variance: Bool
    
    fn __init__(out self, mean: HostBuffer[float_dtype], variance: HostBuffer[float_dtype], 
                n_test: Int, has_variance: Bool):
        self.mean = mean
        self.variance = variance
        self.n_test = n_test
        self.has_variance = has_variance


# =============================================================================
# Constants
# =============================================================================

alias LANCZOS_BREAKDOWN_TOL = Float32(1e-10)  # Tolerance for Lanczos breakdown


# =============================================================================
# Lanczos Root for Composite Kernels
# =============================================================================

fn compute_lanczos_root_composite[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    lanczos_iter: Int = 20,
) raises -> LanczosRootResultComposite:
    """Compute Lanczos root S = Q @ T^{-1/2} for LOVE variance with composite kernels.
    
    This function runs Lanczos decomposition to get Q and T, then computes
    the matrix square root T^{-1/2} via eigendecomposition, and finally
    computes S = Q @ T^{-1/2}.
    
    The Lanczos algorithm computes:
        K ≈ Q @ T @ Q^T
    where Q is [n × r] orthonormal and T is [r × r] tridiagonal.
    
    For LOVE variance, we need:
        K^{-1} ≈ Q @ T^{-1} @ Q^T = (Q @ T^{-1/2}) @ (Q @ T^{-1/2})^T = S @ S^T
    
    Args:
        provider: CompositeProvider for kernel matvec operations
        lanczos_iter: Number of Lanczos iterations (rank r)
        
    Returns:
        LanczosRootResultComposite with root S [n × r] in column-major format, rank r, and n
    """
    var n = provider.get_n()
    var ctx = provider.get_ctx()
    var r = lanczos_iter
    
    # Allocate buffers for tridiagonal elements
    var alpha = List[Float32](capacity=r)
    var beta = List[Float32](capacity=r - 1)
    for i in range(r):
        alpha.append(Float32(0.0))
    for i in range(r - 1):
        beta.append(Float32(0.0))
    
    # Allocate Q matrix [n × r] in column-major format
    var Q_host = ctx.enqueue_create_host_buffer[float_dtype](n * r)
    
    # Generate random probe vector (Rademacher: random ±1)
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
        Q_host[i] = z_host[i]  # Column-major: Q[i, 0] = Q_host[0 * n + i]
    
    # Allocate device buffers for Lanczos vectors
    var v_curr_device = ctx.enqueue_create_buffer[float_dtype](n)
    var v_prev_device = ctx.enqueue_create_buffer[float_dtype](n)
    var w_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Allocate host buffers for intermediate computations
    var v_curr_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var v_prev_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var w_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    
    # Initialize v_curr = z (normalized), v_prev = 0
    for i in range(n):
        v_curr_host[i] = z_host[i]
        v_prev_host[i] = Float32(0.0)
    
    ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
    ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
    ctx.synchronize()
    
    var beta_prev = Float32(0.0)
    
    # Run Lanczos iterations and store Q vectors
    for j in range(r):
        # Compute w = (K + noise*I) @ v_curr using provider
        provider.forward_matvec(w_device.unsafe_ptr(), v_curr_device.unsafe_ptr(), 1)
        
        # Copy result to host
        ctx.enqueue_copy(dst_buf=w_host, src_buf=w_device)
        ctx.synchronize()
        
        # Compute alpha[j] = v_curr^T @ w
        var alpha_j = Float32(0.0)
        for i in range(n):
            alpha_j += v_curr_host[i] * w_host[i]
        alpha[j] = alpha_j
        
        # Orthogonalize: w = w - alpha[j] * v_curr
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
        
        # Check for Lanczos breakdown (early convergence)
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
                Q_host[(j + 1) * n + i] = v_curr_host[i]  # Column-major
        
        # Copy updated vectors to device for next iteration
        ctx.enqueue_copy(dst_buf=v_prev_device, src_buf=v_prev_host)
        ctx.enqueue_copy(dst_buf=v_curr_device, src_buf=v_curr_host)
        ctx.synchronize()
        
        beta_prev = beta_j
    
    # Compute T^{-1/2} via eigendecomposition of tridiagonal T
    # Use PyTorch's LAPACK-backed eigendecomposition for accuracy
    var eigenvalues = List[Float32](capacity=r)
    var eigenvectors = List[Float32](capacity=r * r)
    for i in range(r):
        eigenvalues.append(Float32(0.0))
    for i in range(r * r):
        eigenvectors.append(Float32(0.0))
    
    # Eigendecomposition (Float32 - inputs are Float32 from Lanczos)
    tridiagonal_eigh_native(
        alpha.unsafe_ptr(), beta.unsafe_ptr(), r,
        eigenvalues.unsafe_ptr(), eigenvectors.unsafe_ptr()
    )
    
    # Compute V @ Λ^{-1/2} where V is eigenvector matrix and Λ is diagonal eigenvalue matrix
    # T^{-1/2} = V @ Λ^{-1/2} @ V^T
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
            # Clamp eigenvalue to be positive (numerical stability)
            if lambda_j < eig_clamp:
                lambda_j = eig_clamp
            var lambda_inv_sqrt = Float32(1.0) / sqrt(lambda_j)
            # V_Linv_sqrt[i, j] = V[i, j] * λ_j^{-1/2}
            V_Linv_sqrt[i * r + j] = eigenvectors[i * r + j] * lambda_inv_sqrt
    
    # Compute T_inv_sqrt = (V @ Λ^{-1/2}) @ V^T
    var T_inv_sqrt = List[Float32](capacity=r * r)
    for i in range(r * r):
        T_inv_sqrt.append(Float32(0.0))
    
    for i in range(r):
        for j in range(r):
            var sum_val = Float32(0.0)
            for k in range(r):
                # (V @ Λ^{-1/2})[i, k] * V^T[k, j] = V_Linv_sqrt[i, k] * V[j, k]
                sum_val += V_Linv_sqrt[i * r + k] * eigenvectors[j * r + k]
            T_inv_sqrt[i * r + j] = sum_val
    
    # Compute S = Q @ T^{-1/2} [n × r]
    # Q is [n × r] column-major, T_inv_sqrt is [r × r] row-major
    # S[row, col] = sum_k Q[row, k] * T_inv_sqrt[k, col]
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
    
    return LanczosRootResultComposite(S_host, r, n)


# =============================================================================
# CG Solve for Prediction (alpha = K^{-1} @ y)
# =============================================================================

fn cg_solve_composite[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    rhs_host: HostBuffer[float_dtype],
    max_iter: Int = 100,
    tol: Float32 = 1e-3,
) raises -> HostBuffer[float_dtype]:
    """Solve (K + noise*I) @ x = rhs using Conjugate Gradient.
    
    This is a simple unpreconditioned CG for prediction.
    For training, use the batched CG in combined_inv_quad_logdet.mojo.
    
    Args:
        provider: CompositeProvider for kernel matvec
        rhs_host: Right-hand side vector [n] on host
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        
    Returns:
        Solution x [n] on host
    """
    var n = provider.get_n()
    var ctx = provider.get_ctx()
    
    # Allocate device buffers
    var x_device = ctx.enqueue_create_buffer[float_dtype](n)
    var r_device = ctx.enqueue_create_buffer[float_dtype](n)
    var p_device = ctx.enqueue_create_buffer[float_dtype](n)
    var Ap_device = ctx.enqueue_create_buffer[float_dtype](n)
    var rhs_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Allocate host buffers
    var x_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var r_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var p_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var Ap_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    
    # Initialize x = 0, r = rhs, p = rhs
    for i in range(n):
        x_host[i] = Float32(0.0)
        r_host[i] = rhs_host[i]
        p_host[i] = rhs_host[i]
    
    # Copy to device
    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=r_device, src_buf=r_host)
    ctx.enqueue_copy(dst_buf=p_device, src_buf=p_host)
    ctx.enqueue_copy(dst_buf=rhs_device, src_buf=rhs_host)
    ctx.synchronize()
    
    # Compute initial r^T @ r
    var rTr = Float32(0.0)
    for i in range(n):
        rTr += r_host[i] * r_host[i]
    
    var rhs_norm = Float32(0.0)
    for i in range(n):
        rhs_norm += rhs_host[i] * rhs_host[i]
    rhs_norm = sqrt(rhs_norm)
    
    for iter in range(max_iter):
        # Ap = (K + noise*I) @ p
        provider.forward_matvec(Ap_device.unsafe_ptr(), p_device.unsafe_ptr(), 1)
        ctx.enqueue_copy(dst_buf=Ap_host, src_buf=Ap_device)
        ctx.synchronize()
        
        # alpha = rTr / (p^T @ Ap)
        var pTAp = Float32(0.0)
        for i in range(n):
            pTAp += p_host[i] * Ap_host[i]
        
        if pTAp < Float32(1e-20):
            break
        
        var alpha = rTr / pTAp
        
        # x = x + alpha * p
        # r = r - alpha * Ap
        for i in range(n):
            x_host[i] += alpha * p_host[i]
            r_host[i] -= alpha * Ap_host[i]
        
        # Check convergence
        var rTr_new = Float32(0.0)
        for i in range(n):
            rTr_new += r_host[i] * r_host[i]
        
        var residual_norm = sqrt(rTr_new)
        if residual_norm / rhs_norm < tol:
            break
        
        # beta = rTr_new / rTr
        var beta = rTr_new / rTr
        
        # p = r + beta * p
        for i in range(n):
            p_host[i] = r_host[i] + beta * p_host[i]
        
        # Update for next iteration
        rTr = rTr_new
        
        # Copy updated p to device
        ctx.enqueue_copy(dst_buf=p_device, src_buf=p_host)
        ctx.synchronize()
    
    return x_host


fn cg_solve_composite_preconditioned[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    rhs_host: HostBuffer[float_dtype],
    precond: PivotedCholeskyPrecond,
    max_iter: Int = 100,
    tol: Float32 = 1e-3,
) raises -> HostBuffer[float_dtype]:
    """Solve (K + noise*I) @ x = rhs using Preconditioned Conjugate Gradient.
    
    Uses PivotedCholeskyPrecond to accelerate convergence, matching
    the approach used in training (batched_cg_unified).
    
    Args:
        provider: CompositeProvider for kernel matvec
        rhs_host: Right-hand side vector [n] on host
        precond: Pivoted Cholesky preconditioner
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        
    Returns:
        Solution x [n] on host
    """
    var n = provider.get_n()
    var ctx = provider.get_ctx()
    
    # Allocate device buffers
    var x_device = ctx.enqueue_create_buffer[float_dtype](n)
    var r_device = ctx.enqueue_create_buffer[float_dtype](n)
    var z_device = ctx.enqueue_create_buffer[float_dtype](n)
    var p_device = ctx.enqueue_create_buffer[float_dtype](n)
    var Ap_device = ctx.enqueue_create_buffer[float_dtype](n)
    var rhs_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Allocate host buffers for reading back
    var x_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var r_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var z_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var p_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var Ap_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    
    # Initialize x = 0, r = rhs
    for i in range(n):
        x_host[i] = Float32(0.0)
        r_host[i] = rhs_host[i]
    
    # Copy r to device for preconditioner
    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=r_device, src_buf=r_host)
    ctx.enqueue_copy(dst_buf=rhs_device, src_buf=rhs_host)
    ctx.synchronize()
    
    # z = P^{-1} @ r (preconditioned residual)
    precond.apply_precond(ctx, r_device.unsafe_ptr(), z_device.unsafe_ptr(), n, 1, sync=True)
    ctx.enqueue_copy(dst_buf=z_host, src_buf=z_device)
    ctx.synchronize()
    
    # p = z (initial search direction)
    for i in range(n):
        p_host[i] = z_host[i]
    ctx.enqueue_copy(dst_buf=p_device, src_buf=p_host)
    ctx.synchronize()
    
    # Compute initial r^T @ z
    var rTz = Float32(0.0)
    for i in range(n):
        rTz += r_host[i] * z_host[i]
    
    var rhs_norm = Float32(0.0)
    for i in range(n):
        rhs_norm += rhs_host[i] * rhs_host[i]
    rhs_norm = sqrt(rhs_norm)
    
    for iter_idx in range(max_iter):
        # Ap = (K + noise*I) @ p
        provider.forward_matvec(Ap_device.unsafe_ptr(), p_device.unsafe_ptr(), 1)
        ctx.enqueue_copy(dst_buf=Ap_host, src_buf=Ap_device)
        ctx.synchronize()
        
        # alpha = r^T z / (p^T @ Ap)
        var pTAp = Float32(0.0)
        for i in range(n):
            pTAp += p_host[i] * Ap_host[i]
        
        if pTAp < Float32(1e-20):
            break
        
        var alpha = rTz / pTAp
        
        # x = x + alpha * p
        # r = r - alpha * Ap
        for i in range(n):
            x_host[i] += alpha * p_host[i]
            r_host[i] -= alpha * Ap_host[i]
        
        # Check convergence
        var r_norm = Float32(0.0)
        for i in range(n):
            r_norm += r_host[i] * r_host[i]
        r_norm = sqrt(r_norm)
        
        if r_norm / rhs_norm < tol:
            break
        
        # z = P^{-1} @ r (preconditioned residual)
        ctx.enqueue_copy(dst_buf=r_device, src_buf=r_host)
        ctx.synchronize()
        precond.apply_precond(ctx, r_device.unsafe_ptr(), z_device.unsafe_ptr(), n, 1, sync=True)
        ctx.enqueue_copy(dst_buf=z_host, src_buf=z_device)
        ctx.synchronize()
        
        # beta = r^T z_new / r^T z_old
        var rTz_new = Float32(0.0)
        for i in range(n):
            rTz_new += r_host[i] * z_host[i]
        
        var beta = rTz_new / rTz
        
        # p = z + beta * p
        for i in range(n):
            p_host[i] = z_host[i] + beta * p_host[i]
        
        # Update for next iteration
        rTz = rTz_new
        
        # Copy updated p to device
        ctx.enqueue_copy(dst_buf=p_device, src_buf=p_host)
        ctx.synchronize()
    
    return x_host


# =============================================================================
# Mean Prediction
# =============================================================================

fn predict_mean_composite[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    x_test_host: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
    alpha_host: HostBuffer[float_dtype],
) raises -> HostBuffer[float_dtype]:
    """Compute predictive mean: μ* = K(X*, X) @ α where α = (K + σ²I)^{-1} @ y.
    
    Args:
        provider: CompositeProvider for kernel operations
        x_test_host: Test data [n_test, DIM] on host
        n_test: Number of test points
        alpha_host: Pre-computed α = K^{-1} @ y [n] on host
        
    Returns:
        Predictive mean [n_test] on host
    """
    var n = provider.get_n()
    var ctx = provider.get_ctx()
    
    # Allocate device buffers
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * DIM)
    var alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
    var mean_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    
    # Copy test data and alpha to device
    var x_test_host_buf = ctx.enqueue_create_host_buffer[float_dtype](n_test * DIM)
    for i in range(n_test * DIM):
        x_test_host_buf[i] = x_test_host[i]
    
    ctx.enqueue_copy(dst_buf=x_test_device, src_buf=x_test_host_buf)
    ctx.enqueue_copy(dst_buf=alpha_device, src_buf=alpha_host)
    ctx.synchronize()
    
    # Compute K(X*, X) @ α using cross_matvec
    provider.cross_matvec(
        mean_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        alpha_device.unsafe_ptr(),
        n_test,
        1,  # num_cols = 1
    )
    
    # Copy result to host
    var mean_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    ctx.enqueue_copy(dst_buf=mean_host, src_buf=mean_device)
    ctx.synchronize()
    
    return mean_host


# =============================================================================
# LOVE Variance Prediction
# =============================================================================

fn predict_variance_love_composite[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    x_test_host: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
    lanczos_root: LanczosRootResultComposite,
) raises -> HostBuffer[float_dtype]:
    """Compute predictive variance using LOVE approximation.
    
    LOVE (Low-rank Orthogonal decomposition for Variance Estimation) computes:
        Var(f*) = K(x*, x*) - K(x*, X) @ K^{-1} @ K(X, x*)
                ≈ K(x*, x*) - ||K(x*, X) @ S||²
    
    where S is the Lanczos root such that K^{-1} ≈ S @ S^T.
    
    Args:
        provider: CompositeProvider for kernel operations
        x_test_host: Test data [n_test, DIM] on host
        n_test: Number of test points
        lanczos_root: Pre-computed Lanczos root S [n × r]
        
    Returns:
        Predictive variance [n_test] on host
    """
    var n = provider.get_n()
    var r = lanczos_root.rank
    var ctx = provider.get_ctx()
    
    # Allocate device buffers
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * DIM)
    var S_device = ctx.enqueue_create_buffer[float_dtype](n * r)
    var KxS_device = ctx.enqueue_create_buffer[float_dtype](n_test * r)
    var diag_test_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    
    # Copy test data and Lanczos root to device
    var x_test_host_buf = ctx.enqueue_create_host_buffer[float_dtype](n_test * DIM)
    for i in range(n_test * DIM):
        x_test_host_buf[i] = x_test_host[i]
    
    ctx.enqueue_copy(dst_buf=x_test_device, src_buf=x_test_host_buf)
    ctx.enqueue_copy(dst_buf=S_device, src_buf=lanczos_root.root)
    ctx.synchronize()
    
    # Compute K(X*, X) @ S using cross_matvec
    # S is [n × r] column-major, so we treat it as r columns
    provider.cross_matvec(
        KxS_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        S_device.unsafe_ptr(),
        n_test,
        r,  # num_cols = r
    )
    
    # Copy KxS to host for variance computation
    var KxS_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * r)
    ctx.enqueue_copy(dst_buf=KxS_host, src_buf=KxS_device)
    ctx.synchronize()
    
    # Compute K(x*, x*) for each test point using the diagonal extraction kernel.
    var diag_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    
    # Get kernel parameters from provider
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](K.num_params())
    var params_device_copy = ctx.enqueue_create_buffer[float_dtype](K.num_params())
    ctx.enqueue_copy(dst_buf=params_device_copy, src_buf=provider._params_device)
    ctx.enqueue_copy(dst_buf=params_host, src_buf=params_device_copy)
    ctx.synchronize()
    
    # Compute diagonal K(x*, x*) = K(x_i, x_i) for each test point using GPU kernel
    var threads_per_block = 256
    var num_blocks = (n_test + threads_per_block - 1) // threads_per_block
    
    ctx.enqueue_function[composite_extract_diagonal[DIM, K]](
        diag_test_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        params_device_copy.unsafe_ptr(),
        n_test,
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )
    ctx.synchronize()
    
    # Copy diagonal to host
    ctx.enqueue_copy(dst_buf=diag_test_host, src_buf=diag_test_device)
    ctx.synchronize()
    
    # Compute variance: Var = K(x*, x*) - ||K(x*, X) @ S||²
    var variance_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    
    for i in range(n_test):
        # Compute ||K(x_i*, X) @ S||² = sum_j (KxS[i, j])²
        var norm_sq = Float32(0.0)
        for j in range(r):
            # KxS is [n_test × r] column-major: KxS[i, j] = KxS_host[j * n_test + i]
            var val = KxS_host[j * n_test + i]
            norm_sq += val * val
        
        # Var[i] = K(x_i*, x_i*) - norm_sq
        variance_host[i] = diag_test_host[i] - norm_sq
    
    return variance_host


# =============================================================================
# Exact CG Variance Prediction
# =============================================================================

fn predict_variance_exact_composite[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    x_test_host: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-3,
    precond: Optional[PivotedCholeskyPrecond] = None,
) raises -> HostBuffer[float_dtype]:
    """Compute exact predictive variance using CG solve.
    
    For each test point x*, computes:
        Var(f*) = K(x*, x*) - K(x*, X) @ (K + σ²I)^{-1} @ K(X, x*)
    
    The inner solve (K + σ²I)^{-1} @ K(X, x*) is done via CG.
    Test points are batched for efficiency. Uses preconditioner if provided.
    
    Args:
        provider: CompositeProvider for kernel operations
        x_test_host: Test data [n_test, DIM] on host
        n_test: Number of test points
        max_cg_iter: Maximum CG iterations
        cg_tol: CG convergence tolerance
        
    Returns:
        Predictive variance [n_test] on host
    """
    var n = provider.get_n()
    var ctx = provider.get_ctx()
    
    # --- Step 1: Compute K(X_train, X_test) columns on device ---
    # K_cross[i, j] = K(x_train_i, x_test_j), shape [n, n_test]
    # We get this by computing K(X_test, X_train)^T @ e_j for each j,
    # but more efficiently by using cross_matvec with identity columns.
    # Actually: cross_matvec computes K(X_test, X_train) @ v.
    # We need K(X_train, X_test) @ e_j = column j of K_cross.
    # Since K is symmetric: K(X_train, X_test)[:, j] = K(X_test, X_train)[j, :].T
    # The most direct approach: compute K_cross = K(X_test, X_train)^T 
    # by using cross_matvec with identity columns on the test side.
    #
    # Simpler approach: use cross_matvec to compute K(X_test, X_train) @ v
    # where v is the CG solution. This avoids materializing K_cross entirely.
    #
    # Algorithm per test point x*:
    #   1. rhs = K(X_train, x*) [n×1]  — one column of cross-covariance
    #   2. Solve (K+σ²I) @ v = rhs via CG  
    #   3. Var(x*) = K(x*,x*) - rhs^T @ v
    #
    # For batching: process BATCH_SIZE test points at a time.
    
    alias BATCH_SIZE = 32  # Process this many test points per CG batch
    
    # Allocate output variance on host
    var variance_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    
    # Copy test data to device
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * DIM)
    var x_test_host_buf = ctx.enqueue_create_host_buffer[float_dtype](n_test * DIM)
    for i in range(n_test * DIM):
        x_test_host_buf[i] = x_test_host[i]
    ctx.enqueue_copy(dst_buf=x_test_device, src_buf=x_test_host_buf)
    ctx.synchronize()
    
    # --- Step 2: Compute K(x*, x*) diagonal for all test points ---
    var diag_test_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    var threads_per_block = 256
    var num_blocks_diag = (n_test + threads_per_block - 1) // threads_per_block
    
    ctx.enqueue_function[composite_extract_diagonal[DIM, K]](
        diag_test_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        provider.params_ptr,
        n_test,
        grid_dim=num_blocks_diag,
        block_dim=threads_per_block,
    )
    ctx.synchronize()
    
    var diag_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    ctx.enqueue_copy(dst_buf=diag_test_host, src_buf=diag_test_device)
    ctx.synchronize()
    
    # Batched CG across all test points.
    # Materialize K_cross [n, n_test] on GPU, then solve (K+σ²I) @ V = K_cross
    # in a single batched CG, then compute variance from the solution.
    
    # Materialize K_cross [n_train, n_test] on GPU using dedicated kernel
    var k_cross_device = ctx.enqueue_create_buffer[float_dtype](n * n_test)
    var num_blocks_cross = (n + threads_per_block - 1) // threads_per_block
    
    ctx.enqueue_function[composite_cross_covariance_gpu[DIM, K]](
        k_cross_device.unsafe_ptr(),
        provider.x_ptr,              # x_train [n, DIM]
        x_test_device.unsafe_ptr(),  # x_test [n_test, DIM]
        provider.params_ptr,
        n,                           # n_train
        n_test,                      # n_test
        grid_dim=num_blocks_cross,
        block_dim=threads_per_block,
    )
    ctx.synchronize()
    
    # Copy K_cross to host as RHS for batched CG
    var k_cross_host = ctx.enqueue_create_host_buffer[float_dtype](n * n_test)
    ctx.enqueue_copy(dst_buf=k_cross_host, src_buf=k_cross_device)
    ctx.synchronize()
    
    # Batched CG: solve (K + σ²I) @ V = K_cross for all n_test columns at once.
    # V has shape [n, n_test].
    # We reuse the single-column CG implementation but extend to multiple columns.
    # Each column is independent, so we process them in parallel batches.
    
    # Allocate solution buffers
    var V_host = ctx.enqueue_create_host_buffer[float_dtype](n * n_test)
    var V_device = ctx.enqueue_create_buffer[float_dtype](n * n_test)
    
    # Process columns in batches that fit forward_matvec's multi-column support
    alias CG_BATCH = 32  # Process up to 32 CG columns at a time
    
    var col = 0
    while col < n_test:
        var batch_cols = min(CG_BATCH, n_test - col)
        
        # Set up host buffers for this CG batch
        var x_cg_host = ctx.enqueue_create_host_buffer[float_dtype](n * batch_cols)
        var r_cg_host = ctx.enqueue_create_host_buffer[float_dtype](n * batch_cols)
        var p_cg_host = ctx.enqueue_create_host_buffer[float_dtype](n * batch_cols)
        var Ap_cg_host = ctx.enqueue_create_host_buffer[float_dtype](n * batch_cols)
        
        # Initialize: x=0, r=rhs, p=rhs
        for c in range(batch_cols):
            var rhs_col_offset = (col + c) * n
            var cg_col_offset = c * n
            for i in range(n):
                x_cg_host[cg_col_offset + i] = Float32(0.0)
                r_cg_host[cg_col_offset + i] = k_cross_host[rhs_col_offset + i]
                p_cg_host[cg_col_offset + i] = k_cross_host[rhs_col_offset + i]
        
        # Compute initial per-column rTr and rhs_norm
        var rTr = List[Float32]()
        var rhs_norm = List[Float32]()
        for c in range(batch_cols):
            var cg_col_offset = c * n
            var s = Float32(0.0)
            for i in range(n):
                s += r_cg_host[cg_col_offset + i] * r_cg_host[cg_col_offset + i]
            rTr.append(s)
            rhs_norm.append(sqrt(s))
        
        # Device buffers for p and Ap
        var p_device = ctx.enqueue_create_buffer[float_dtype](n * batch_cols)
        var Ap_device = ctx.enqueue_create_buffer[float_dtype](n * batch_cols)
        
        ctx.enqueue_copy(dst_buf=p_device, src_buf=p_cg_host)
        ctx.synchronize()
        
        # CG iterations
        for cg_iter in range(max_cg_iter):
            # Ap = (K + noise*I) @ p  (batched multi-column matvec)
            provider.forward_matvec(Ap_device.unsafe_ptr(), p_device.unsafe_ptr(), batch_cols)
            ctx.enqueue_copy(dst_buf=Ap_cg_host, src_buf=Ap_device)
            ctx.synchronize()
            
            var all_converged = True
            for c in range(batch_cols):
                var cg_col_offset = c * n
                
                # alpha_c = rTr_c / (p_c^T @ Ap_c)
                var pTAp = Float32(0.0)
                for i in range(n):
                    pTAp += p_cg_host[cg_col_offset + i] * Ap_cg_host[cg_col_offset + i]
                
                if pTAp < Float32(1e-20):
                    continue
                
                var alpha = rTr[c] / pTAp
                
                # x_c += alpha * p_c;  r_c -= alpha * Ap_c
                var rTr_new = Float32(0.0)
                for i in range(n):
                    x_cg_host[cg_col_offset + i] += alpha * p_cg_host[cg_col_offset + i]
                    r_cg_host[cg_col_offset + i] -= alpha * Ap_cg_host[cg_col_offset + i]
                    rTr_new += r_cg_host[cg_col_offset + i] * r_cg_host[cg_col_offset + i]
                
                if sqrt(rTr_new) / rhs_norm[c] >= cg_tol:
                    all_converged = False
                
                # beta = rTr_new / rTr; p = r + beta * p
                var beta = rTr_new / max(rTr[c], Float32(1e-30))
                for i in range(n):
                    p_cg_host[cg_col_offset + i] = r_cg_host[cg_col_offset + i] + beta * p_cg_host[cg_col_offset + i]
                
                rTr[c] = rTr_new
            
            if all_converged:
                break
            
            # Copy updated p to device for next matvec
            ctx.enqueue_copy(dst_buf=p_device, src_buf=p_cg_host)
            ctx.synchronize()
        
        # Copy solution columns into V_host
        for c in range(batch_cols):
            var cg_col_offset = c * n
            var v_col_offset = (col + c) * n
            for i in range(n):
                V_host[v_col_offset + i] = x_cg_host[cg_col_offset + i]
        
        col += batch_cols
    
    # --- Step 4: Compute variance ---
    # Var[j] = K(x*_j, x*_j) - K_cross[:, j]^T @ V[:, j]
    for j in range(n_test):
        var dot_product = Float32(0.0)
        var col_offset = j * n
        for i in range(n):
            dot_product += k_cross_host[col_offset + i] * V_host[col_offset + i]
        variance_host[j] = diag_test_host[j] - dot_product
    
    _ = k_cross_device
    _ = x_test_device
    return variance_host


# =============================================================================
# Combined Prediction (Mean + Variance) with method selection
# =============================================================================

fn predict_composite_with_method[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    x_test_host: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
    y_host: HostBuffer[float_dtype],
    lanczos_root: LanczosRootResultComposite,
    variance_method: Int,  # 0 = love, 1 = exact
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-3,
    precond: Optional[PivotedCholeskyPrecond] = None,
) raises -> PredictionResultComposite:
    """Compute predictive mean and variance for composite kernels.
    
    Supports both LOVE (fast approximate) and exact (CG-based) variance.
    When a preconditioner is provided, CG convergence is significantly faster
    for ill-conditioned kernels.
    
    Args:
        provider: CompositeProvider for kernel operations
        x_test_host: Test data [n_test, DIM] on host
        n_test: Number of test points
        y_host: Training targets [n] on host
        lanczos_root: Pre-computed Lanczos root S [n × r] (used for LOVE)
        variance_method: 0 = LOVE variance, 1 = exact CG variance
        max_cg_iter: Maximum CG iterations
        cg_tol: CG convergence tolerance
        precond: Optional Pivoted Cholesky preconditioner for faster CG convergence
        
    Returns:
        PredictionResultComposite with mean and variance
    """
    # Step 1: Solve for α = K^{-1} @ y (preconditioned if available)
    var alpha_host: HostBuffer[float_dtype]
    if precond:
        alpha_host = cg_solve_composite_preconditioned(
            provider, y_host, precond.value(), max_cg_iter, cg_tol
        )
    else:
        alpha_host = cg_solve_composite(provider, y_host, max_cg_iter, cg_tol)
    
    # Step 2: Compute mean
    var mean_host = predict_mean_composite(provider, x_test_host, n_test, alpha_host)
    
    # Step 3: Compute variance using selected method
    var variance_host: HostBuffer[float_dtype]
    if variance_method == 1:
        # Exact CG variance (preconditioned if available)
        variance_host = predict_variance_exact_composite(
            provider, x_test_host, n_test, max_cg_iter, cg_tol, precond
        )
    else:
        # LOVE variance (default)
        variance_host = predict_variance_love_composite(
            provider, x_test_host, n_test, lanczos_root
        )
    
    return PredictionResultComposite(mean_host, variance_host, n_test, True)


# =============================================================================
# Combined Prediction (Mean + Variance) — LOVE only (original)
# =============================================================================

fn predict_composite[DIM: Int, K: ComposableKernel](
    provider: CompositeProvider[DIM, K],
    x_test_host: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
    y_host: HostBuffer[float_dtype],
    lanczos_root: LanczosRootResultComposite,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-3,
    precond: Optional[PivotedCholeskyPrecond] = None,
) raises -> PredictionResultComposite:
    """Compute predictive mean and variance for composite kernels.
    
    This function:
    1. Solves α = (K + σ²I)^{-1} @ y using CG (preconditioned if available)
    2. Computes mean μ* = K(X*, X) @ α
    3. Computes variance using LOVE: Var* = K(X*, X*) - ||K(X*, X) @ S||²
    
    Args:
        provider: CompositeProvider for kernel operations
        x_test_host: Test data [n_test, DIM] on host
        n_test: Number of test points
        y_host: Training targets [n] on host
        lanczos_root: Pre-computed Lanczos root S [n × r]
        max_cg_iter: Maximum CG iterations for solving α
        cg_tol: CG convergence tolerance
        precond: Optional Pivoted Cholesky preconditioner for faster CG convergence
        
    Returns:
        PredictionResultComposite with mean and variance
    """
    # Step 1: Solve for α = K^{-1} @ y (preconditioned if available)
    var alpha_host: HostBuffer[float_dtype]
    if precond:
        alpha_host = cg_solve_composite_preconditioned(
            provider, y_host, precond.value(), max_cg_iter, cg_tol
        )
    else:
        alpha_host = cg_solve_composite(provider, y_host, max_cg_iter, cg_tol)
    
    # Step 2: Compute mean
    var mean_host = predict_mean_composite(provider, x_test_host, n_test, alpha_host)
    
    # Step 3: Compute variance using LOVE
    var variance_host = predict_variance_love_composite(
        provider, x_test_host, n_test, lanczos_root
    )
    
    return PredictionResultComposite(mean_host, variance_host, n_test, True)
