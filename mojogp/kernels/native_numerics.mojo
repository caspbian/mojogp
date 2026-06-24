"""
Native Mojo Numerical Algorithms for MojoGP.

This module provides native Mojo implementations of critical linear algebra routines
to replace the PyTorch interop overhead.

Functions:
- cholesky_decompose(): Cholesky decomposition (L L^T)
- lu_decompose(): LU decomposition with partial pivoting
- matrix_inv_native(): Matrix inversion (Cholesky with LU fallback)
- compute_slogdet_native(): Log-determinant (Cholesky with LU fallback)
- tridiagonal_eigh_native(): QR algorithm with Wilkinson shifts for symmetric tridiagonal matrices
"""

from memory import UnsafePointer
from memory.unsafe_pointer import alloc
from math import sqrt, log, copysign
from builtin.math import abs, max

# =============================================================================
# Cholesky Decomposition
# =============================================================================

fn cholesky_decompose(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    L_out: UnsafePointer[Float32, MutAnyOrigin]
) -> Bool:
    """Compute Cholesky decomposition A = L L^T.
    
    Args:
        matrix: Input symmetric positive-definite matrix [n x n] (row-major)
        n: Matrix size
        L_out: Output lower triangular matrix [n x n] (row-major)
        
    Returns:
        True if successful, False if matrix is not positive definite.
    """
    # Initialize L_out to zeros
    for i in range(n * n):
        L_out[i] = 0.0
        
    for i in range(n):
        for j in range(i + 1):
            var sum_val = matrix[i * n + j]
            for k in range(j):
                sum_val -= L_out[i * n + k] * L_out[j * n + k]
                
            if i == j:
                if sum_val <= 0.0:
                    return False # Not positive definite
                L_out[i * n + i] = sqrt(sum_val)
            else:
                L_out[i * n + j] = sum_val / L_out[j * n + j]
                
    return True

# =============================================================================
# LU Decomposition with Partial Pivoting
# =============================================================================

fn lu_decompose(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    LU_out: UnsafePointer[Float32, MutAnyOrigin],
    P_out: UnsafePointer[Int, MutAnyOrigin]
) -> Int:
    """Compute LU decomposition with partial pivoting PA = LU.
    
    Args:
        matrix: Input matrix [n x n] (row-major)
        n: Matrix size
        LU_out: Output matrix containing L (strictly lower) and U (upper) [n x n]
        P_out: Output permutation array [n]
        
    Returns:
        Number of row swaps (useful for determinant sign), or -1 if singular.
    """
    # Copy matrix to LU_out and initialize P_out
    for i in range(n):
        P_out[i] = i
        for j in range(n):
            LU_out[i * n + j] = matrix[i * n + j]
            
    var swaps = 0
    
    for k in range(n):
        # Find pivot
        var pivot_val = Float32(0.0)
        var pivot_idx = k
        for i in range(k, n):
            var val = abs(LU_out[P_out[i] * n + k])
            if val > pivot_val:
                pivot_val = val
                pivot_idx = i
                
        if pivot_val == 0.0:
            return -1 # Singular matrix
            
        # Swap rows in permutation array
        if pivot_idx != k:
            var temp = P_out[k]
            P_out[k] = P_out[pivot_idx]
            P_out[pivot_idx] = temp
            swaps += 1
            
        var pk = P_out[k]
        var diag_val = LU_out[pk * n + k]
        
        # Eliminate
        for i in range(k + 1, n):
            var pi = P_out[i]
            var factor = LU_out[pi * n + k] / diag_val
            LU_out[pi * n + k] = factor
            for j in range(k + 1, n):
                LU_out[pi * n + j] -= factor * LU_out[pk * n + j]
                
    return swaps

# =============================================================================
# Matrix Inversion
# =============================================================================

fn matrix_inv_native(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    result_out: UnsafePointer[Float32, MutAnyOrigin]
) raises:
    """Compute matrix inverse using Cholesky with LU fallback.
    
    Args:
        matrix: Input matrix [n x n] (row-major)
        n: Matrix size
        result_out: Output inverse matrix [n x n] (row-major)
    """
    var L = alloc[Float32](n * n)
    var is_spd = cholesky_decompose(matrix, n, L)
    
    if is_spd:
        # Invert using Cholesky: A^-1 = L^-T L^-1
        # First find L^-1
        var L_inv = alloc[Float32](n * n)
        for i in range(n * n):
            L_inv[i] = 0.0
            
        for i in range(n):
            L_inv[i * n + i] = 1.0 / L[i * n + i]
            for j in range(i):
                var sum_val = Float32(0.0)
                for k in range(j, i):
                    sum_val -= L[i * n + k] * L_inv[k * n + j]
                L_inv[i * n + j] = sum_val / L[i * n + i]
                
        # Compute L^-T L^-1
        for i in range(n):
            for j in range(n):
                var sum_val = Float32(0.0)
                # L_inv is lower triangular, so L_inv^T is upper triangular
                # (L_inv^T)_{i,k} = L_inv_{k,i}
                # We want sum_k (L_inv^T)_{i,k} * L_inv_{k,j} = sum_k L_inv_{k,i} * L_inv_{k,j}
                # Since L_inv is lower triangular, k must be >= max(i, j)
                var start_k = i
                if j > i:
                    start_k = j
                for k in range(start_k, n):
                    sum_val += L_inv[k * n + i] * L_inv[k * n + j]
                result_out[i * n + j] = sum_val
                
        L_inv.free()
    else:
        # Fallback to LU decomposition
        var LU = alloc[Float32](n * n)
        var P = alloc[Int](n)
        var swaps = lu_decompose(matrix, n, LU, P)
        
        if swaps == -1:
            LU.free()
            P.free()
            L.free()
            raise Error("Matrix is singular and cannot be inverted")
            
        # Solve AX = I column by column
        var y = alloc[Float32](n)
        for col in range(n):
            # Forward substitution: Ly = P * e_col
            for i in range(n):
                var pi = P[i]
                var val = Float32(1.0) if pi == col else Float32(0.0)
                for j in range(i):
                    val -= LU[pi * n + j] * y[j]
                y[i] = val
                
            # Backward substitution: Ux = y
            for i in range(n - 1, -1, -1):
                var pi = P[i]
                var val = y[i]
                for j in range(i + 1, n):
                    val -= LU[pi * n + j] * result_out[j * n + col]
                result_out[i * n + col] = val / LU[pi * n + i]
                
        y.free()
        LU.free()
        P.free()
        
    L.free()

# =============================================================================
# Log-Determinant
# =============================================================================

fn compute_slogdet_native(
    matrix: UnsafePointer[Float32, MutAnyOrigin],
    n: Int
) raises -> Float32:
    """Compute log|det(matrix)| using Cholesky with LU fallback.
    
    Args:
        matrix: Input matrix [n x n] (row-major)
        n: Matrix size
        
    Returns:
        log|det(matrix)|
    """
    var L = alloc[Float32](n * n)
    var is_spd = cholesky_decompose(matrix, n, L)
    var logdet = Float32(0.0)
    
    if is_spd:
        for i in range(n):
            logdet += log(L[i * n + i])
        logdet *= 2.0
    else:
        var LU = alloc[Float32](n * n)
        var P = alloc[Int](n)
        var swaps = lu_decompose(matrix, n, LU, P)
        
        if swaps == -1:
            LU.free()
            P.free()
            L.free()
            # Determinant is 0, log|det| is -inf
            return Float32(-1e20) # Approximation of -inf
            
        for i in range(n):
            var pi = P[i]
            logdet += log(abs(LU[pi * n + i]))
            
        LU.free()
        P.free()
        
    L.free()
    return logdet

# =============================================================================
# Tridiagonal Eigendecomposition (QR Algorithm with Wilkinson Shifts)
# =============================================================================

fn pythag(a: Float32, b: Float32) -> Float32:
    """Computes sqrt(a^2 + b^2) without destructive underflow or overflow."""
    var absa = abs(a)
    var absb = abs(b)
    if absa > absb:
        var ratio = absb / absa
        return absa * sqrt(1.0 + ratio * ratio)
    elif absb > 0.0:
        var ratio = absa / absb
        return absb * sqrt(1.0 + ratio * ratio)
    else:
        return 0.0

fn tridiagonal_eigh_native(
    diag: UnsafePointer[Float32, MutAnyOrigin],
    offdiag: UnsafePointer[Float32, MutAnyOrigin],
    m: Int,
    eigenvalues_out: UnsafePointer[Float32, MutAnyOrigin],
    eigenvectors_out: UnsafePointer[Float32, MutAnyOrigin]
) raises:
    """Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix.
    
    Uses the QR algorithm with Wilkinson shifts.
    
    Args:
        diag: Diagonal elements [m]
        offdiag: Off-diagonal elements [m-1]
        m: Size of matrix
        eigenvalues_out: Output eigenvalues [m] (sorted ascending)
        eigenvectors_out: Output eigenvector matrix [m x m] (row-major)
    """
    # Initialize eigenvectors to identity
    for i in range(m):
        for j in range(m):
            eigenvectors_out[i * m + j] = Float32(1.0) if i == j else Float32(0.0)
            
    # Copy diag and offdiag to working arrays
    var d = alloc[Float32](m)
    var e = alloc[Float32](m)
    
    for i in range(m):
        d[i] = diag[i]
        if i < m - 1:
            e[i] = offdiag[i]
        else:
            e[i] = 0.0
            
    var max_iter = 200 * m  # Generous budget for strongly-coupled tridiagonals
    var eps = Float32(1e-6) # Tolerance
    
    for l in range(m):
        var iter = 0
        while True:
            # Look for a small sub-diagonal element to split the matrix
            # Use explicit relative tolerance (scales correctly for large-magnitude
            # tridiagonals from Polynomial/Linear kernels)
            var m_idx = l
            while m_idx < m - 1:
                var dd = abs(d[m_idx]) + abs(d[m_idx + 1])
                if abs(e[m_idx]) <= eps * dd or abs(e[m_idx]) < Float32(1e-30):
                    break
                m_idx += 1
                
            if m_idx == l:
                break # Eigenvalue found
                
            if iter == max_iter:
                # QR didn't fully converge. Use partially-converged diagonal as
                # eigenvalue estimates rather than crashing. This can happen with
                # small n and kernels that produce ill-conditioned tridiagonals
                # (e.g., Polynomial/Linear at n<2000).
                print("WARNING: QR eigendecomposition did not fully converge (l=", l, "/", m, "). Using approximate eigenvalues.")
                break
                
            iter += 1
            
            # Form shift
            var g = (d[l + 1] - d[l]) / (2.0 * e[l])
            var r = pythag(g, 1.0)
            g = d[m_idx] - d[l] + e[l] / (g + copysign(r, g))
            
            var s = Float32(1.0)
            var c = Float32(1.0)
            var p = Float32(0.0)
            
            var i = m_idx - 1
            while i >= l:
                var f = s * e[i]
                var b = c * e[i]
                r = pythag(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m_idx] = 0.0
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2.0 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                
                # Update eigenvectors
                for k in range(m):
                    var f_vec = eigenvectors_out[k * m + i + 1]
                    eigenvectors_out[k * m + i + 1] = s * eigenvectors_out[k * m + i] + c * f_vec
                    eigenvectors_out[k * m + i] = c * eigenvectors_out[k * m + i] - s * f_vec
                    
                i -= 1
                
            if r == 0.0 and i >= l:
                continue
                
            d[l] -= p
            e[l] = g
            e[m_idx] = 0.0
            
    # Sort eigenvalues and corresponding eigenvectors
    for i in range(m - 1):
        var k = i
        var p = d[i]
        for j in range(i + 1, m):
            if d[j] < p:
                k = j
                p = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p
            for j in range(m):
                var p_vec = eigenvectors_out[j * m + i]
                eigenvectors_out[j * m + i] = eigenvectors_out[j * m + k]
                eigenvectors_out[j * m + k] = p_vec
                
    # Copy eigenvalues to output
    for i in range(m):
        eigenvalues_out[i] = d[i]
        
    d.free()
    e.free()

# =============================================================================
# Batched SLQ Log-Determinant from Tridiagonals
# =============================================================================

fn compute_logdet_from_tridiag_batched_native(
    tridiag_diag_batch: List[List[Float32]],
    tridiag_offdiag_batch: List[List[Float32]],
    m: Int,
    num_probes: Int,
    n: Int,
) raises -> Float32:
    """Compute log|K| from batched tridiagonal matrices using native Mojo.
    
    This replaces the PyTorch implementation with native eigendecomposition.
    
    Algorithm:
    1. For each probe's tridiagonal matrix T:
       - Compute eigendecomposition using tridiagonal_eigh_native()
       - Apply SLQ formula: log|K| ≈ n * mean(sum_i V[0,i]² * log(λ_i))
    2. Handle negative eigenvalues exactly like GPyTorch (zero out contribution)
    
    Args:
        tridiag_diag_batch: Diagonal elements for each probe [num_probes][m]
        tridiag_offdiag_batch: Off-diagonal elements for each probe [num_probes][m-1]
        m: Tridiagonal size (number of CG iterations)
        num_probes: Number of probe vectors
        n: Original matrix size (for SLQ scaling)
        
    Returns:
        Estimated log|K|
    """
    if m == 0 or num_probes == 0:
        return Float32(0.0)
    
    # Allocate working arrays (Float64 for eigendecomposition, Float32 for I/O)
    var diag_f64 = alloc[Float64](m)
    var offdiag_f64 = alloc[Float64](m - 1)
    var eigenvalues_f64 = alloc[Float64](m)
    var eigenvectors_f64 = alloc[Float64](m * m)
    
    var total_slq = Float32(0.0)
    var valid_probes = 0
    
    for p in range(num_probes):
        # Copy tridiagonal data for this probe (upcast to Float64)
        var has_nan_or_inf = False
        for i in range(m):
            var val = tridiag_diag_batch[p][i]
            if val != val or val == Float32(1e20) or val == Float32(-1e20):  # NaN or Inf check
                has_nan_or_inf = True
                break
            diag_f64[i] = Float64(val)
            
        if has_nan_or_inf:
            continue  # Skip this probe
            
        for i in range(m - 1):
            var val = tridiag_offdiag_batch[p][i]
            if val != val or val == Float32(1e20) or val == Float32(-1e20):
                has_nan_or_inf = True
                break
            offdiag_f64[i] = Float64(val)
            
        if has_nan_or_inf:
            continue  # Skip this probe
        
        # Compute eigendecomposition in Float64
        try:
            tridiagonal_eigh_f64(diag_f64, offdiag_f64, m, eigenvalues_f64, eigenvectors_f64)
        except:
            # Eigendecomposition failed, skip this probe
            continue
        
        # Apply SLQ formula: sum_i V[0,i]² * log(λ_i)
        # V[0, i] is the first component of the i-th eigenvector
        # In row-major: eigenvectors_f64[0 * m + i] = eigenvectors_f64[i]
        
        # Relative eigenvalue clamping (matches GPyTorch): use max_eig * 1e-6
        # instead of absolute 1e-20, so the threshold scales with the spectrum
        var max_eig = Float32(1e-10)  # floor to avoid zero threshold
        for i in range(m):
            var eig_f32 = Float32(eigenvalues_f64[i])
            if eig_f32 > max_eig:
                max_eig = eig_f32
        var eig_clamp = max_eig * Float32(1e-6)

        var slq_term = Float32(0.0)
        for i in range(m):
            var lambda_i = Float32(eigenvalues_f64[i])
            var v0_i = Float32(eigenvectors_f64[i])  # First row, i-th column
            
            # Handle negative eigenvalues like GPyTorch:
            # Skip negative eigenvalues (equivalent to clamping to 1, so log(1)=0)
            if lambda_i < Float32(0.0):
                continue  # Skip this eigenvalue (contributes 0)
            
            # Skip eigenvalues below relative threshold to guard against
            # log(0) which gives -inf, and 0 * (-inf) = NaN
            if lambda_i < eig_clamp:
                continue  # Treat as zero contribution
            
            slq_term += v0_i * v0_i * log(lambda_i)
        
        total_slq += slq_term
        valid_probes += 1
    
    # Free working arrays
    diag_f64.free()
    offdiag_f64.free()
    eigenvalues_f64.free()
    eigenvectors_f64.free()
    
    # Average over valid probes and scale by n
    if valid_probes == 0:
        return Float32(0.0)
    
    var mean_slq = total_slq / Float32(valid_probes)
    var log_det = Float32(n) * mean_slq
    
    return log_det


# =============================================================================
# Float64 Symmetric Eigendecomposition
# =============================================================================
# 
# This section implements native Mojo eigendecomposition for small symmetric
# matrices (n <= 32), replacing torch.linalg.eigh which can fail on 
# ill-conditioned matrices.
#
# Algorithm:
# 1. Householder reduction: A -> Q T Q^T (tridiagonalize)
# 2. QR algorithm with Wilkinson shifts: T -> V_T Lambda V_T^T
# 3. Back-transform: V = Q @ V_T
#
# Reference: Golub & Van Loan, "Matrix Computations", 4th ed., Chapter 8
# =============================================================================


fn abs_f64(x: Float64) -> Float64:
    """Absolute value for Float64."""
    if x < 0:
        return -x
    return x


fn copysign_f64(magnitude: Float64, sign_val: Float64) -> Float64:
    """Return magnitude with sign of sign_val."""
    var abs_mag = abs_f64(magnitude)
    if sign_val >= 0:
        return abs_mag
    return -abs_mag


fn pythag_f64(a: Float64, b: Float64) -> Float64:
    """Compute sqrt(a^2 + b^2) without overflow/underflow.
    
    Uses the formula: |a| * sqrt(1 + (b/a)^2) when |a| > |b|
    """
    var absa = abs_f64(a)
    var absb = abs_f64(b)
    
    if absa > absb:
        if absb == 0.0:
            return absa
        var ratio = absb / absa
        return absa * sqrt(1.0 + ratio * ratio)
    elif absb > 0.0:
        var ratio = absa / absb
        return absb * sqrt(1.0 + ratio * ratio)
    else:
        return 0.0


fn matrix_multiply_f64(
    A: UnsafePointer[Float64, MutAnyOrigin],  # [m x k] row-major
    B: UnsafePointer[Float64, MutAnyOrigin],  # [k x n] row-major
    m: Int, 
    k: Int, 
    n: Int,
    C: UnsafePointer[Float64, MutAnyOrigin]   # [m x n] row-major output
):
    """Compute C = A @ B for Float64 matrices.
    
    Simple O(m*k*n) implementation suitable for small matrices.
    For large matrices, use vendor BLAS instead.
    
    Args:
        A: Input matrix [m x k] row-major
        B: Input matrix [k x n] row-major
        m, k, n: Matrix dimensions
        C: Output matrix [m x n] row-major
    """
    for i in range(m):
        for j in range(n):
            var sum_val = Float64(0.0)
            for l in range(k):
                sum_val += A[i * k + l] * B[l * n + j]
            C[i * n + j] = sum_val


fn householder_tridiagonalize_f64(
    A: UnsafePointer[Float64, MutAnyOrigin],      # Input symmetric [n x n], MODIFIED
    n: Int,
    Q: UnsafePointer[Float64, MutAnyOrigin],      # Output orthogonal [n x n]
    diag: UnsafePointer[Float64, MutAnyOrigin],   # Output diagonal [n]
    offdiag: UnsafePointer[Float64, MutAnyOrigin] # Output off-diagonal [n-1]
):
    """Reduce symmetric matrix A to tridiagonal form using Householder reflections.
    
    Computes A = Q T Q^T where T is tridiagonal and Q is orthogonal.
    
    This implementation follows the standard algorithm from Numerical Recipes
    and LAPACK's DSYTRD routine.
    
    Args:
        A: Input symmetric matrix [n x n] row-major. MODIFIED in place.
        n: Matrix size
        Q: Output orthogonal matrix [n x n] row-major
        diag: Output diagonal elements [n]
        offdiag: Output off-diagonal elements [n-1]
    """
    # Initialize Q to identity
    for i in range(n):
        for j in range(n):
            Q[i * n + j] = Float64(1.0) if i == j else Float64(0.0)
    
    # Handle small matrices directly
    if n == 1:
        diag[0] = A[0]
        return
    
    if n == 2:
        diag[0] = A[0]
        diag[1] = A[1 * n + 1]
        offdiag[0] = A[0 * n + 1]
        return
    
    # Allocate working vectors
    var e = alloc[Float64](n)  # Off-diagonal elements during reduction
    
    # Reduce to tridiagonal form from bottom-right to top-left
    # This is the standard "tred2" algorithm
    for i in range(n - 1, 0, -1):
        var l = i - 1
        var h = Float64(0.0)
        var scale = Float64(0.0)
        
        if l > 0:
            # Scale row to avoid overflow
            for k in range(l + 1):
                scale += abs_f64(A[i * n + k])
            
            if scale == 0.0:
                e[i] = A[i * n + l]
            else:
                # Scale the row
                for k in range(l + 1):
                    A[i * n + k] /= scale
                    h += A[i * n + k] * A[i * n + k]
                
                var f = A[i * n + l]
                var g = -copysign_f64(sqrt(h), f)
                e[i] = scale * g
                h -= f * g
                A[i * n + l] = f - g
                
                # Store u in ith row of A
                f = Float64(0.0)
                for j in range(l + 1):
                    # Store u/H in jth column of Q
                    Q[j * n + i] = A[i * n + j] / h
                    
                    # Form element of A*u in g
                    g = Float64(0.0)
                    for k in range(j + 1):
                        g += A[j * n + k] * A[i * n + k]
                    for k in range(j + 1, l + 1):
                        g += A[k * n + j] * A[i * n + k]
                    
                    # Form element of p in temporarily unused element of e
                    e[j] = g / h
                    f += e[j] * A[i * n + j]
                
                var hh = f / (h + h)
                
                # Form reduced A
                for j in range(l + 1):
                    f = A[i * n + j]
                    g = e[j] - hh * f
                    e[j] = g
                    for k in range(j + 1):
                        A[j * n + k] -= f * e[k] + g * A[i * n + k]
        else:
            e[i] = A[i * n + l]
        
        diag[i] = h
    
    diag[0] = Float64(0.0)
    e[0] = Float64(0.0)
    
    # Accumulate transformation matrices
    for i in range(n):
        var l = i - 1
        if diag[i] != 0.0:
            for j in range(l + 1):
                var g = Float64(0.0)
                for k in range(l + 1):
                    g += A[i * n + k] * Q[k * n + j]
                for k in range(l + 1):
                    Q[k * n + j] -= g * Q[k * n + i]
        
        diag[i] = A[i * n + i]
        A[i * n + i] = Float64(1.0)
        for j in range(l + 1):
            Q[j * n + i] = Float64(0.0)
            Q[i * n + j] = Float64(0.0)
    
    # Copy off-diagonal elements
    for i in range(n - 1):
        offdiag[i] = e[i + 1]
    
    e.free()


fn tridiagonal_eigh_f64(
    diag: UnsafePointer[Float64, MutAnyOrigin],
    offdiag: UnsafePointer[Float64, MutAnyOrigin],
    m: Int,
    eigenvalues_out: UnsafePointer[Float64, MutAnyOrigin],
    eigenvectors_out: UnsafePointer[Float64, MutAnyOrigin]
) raises:
    """Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix.
    
    Uses the QR algorithm with Wilkinson shifts (Float64 version).
    
    Args:
        diag: Diagonal elements [m]
        offdiag: Off-diagonal elements [m-1]
        m: Size of matrix
        eigenvalues_out: Output eigenvalues [m] (sorted ascending)
        eigenvectors_out: Output eigenvector matrix [m x m] (row-major)
    """
    # Initialize eigenvectors to identity
    for i in range(m):
        for j in range(m):
            eigenvectors_out[i * m + j] = Float64(1.0) if i == j else Float64(0.0)
            
    # Copy diag and offdiag to working arrays
    var d = alloc[Float64](m)
    var e = alloc[Float64](m)
    
    for i in range(m):
        d[i] = diag[i]
        if i < m - 1:
            e[i] = offdiag[i]
        else:
            e[i] = 0.0
            
    var max_iter = 1000 * m  # Large budget for strongly-coupled tridiagonals
    
    for l in range(m):
        var iter_count = 0
        while True:
            # Look for a small sub-diagonal element to split the matrix
            # Use explicit relative tolerance instead of floating-point equality test.
            # The classic criterion `abs(e[i]) + dd == dd` relies on machine epsilon
            # which fails when diagonal values are huge (e.g., Polynomial kernels where
            # k(x,x) ~ (x^Tx + offset)^degree can be 1e6+). An explicit tolerance of
            # 1e-14 * (|d[i]| + |d[i+1]|) is equivalent but works across all scales.
            var m_idx = l
            while m_idx < m - 1:
                var dd = abs_f64(d[m_idx]) + abs_f64(d[m_idx + 1])
                if abs_f64(e[m_idx]) <= Float64(1e-14) * dd:
                    break
                m_idx += 1
                
            if m_idx == l:
                break  # Eigenvalue found
                
            if iter_count == max_iter:
                d.free()
                e.free()
                raise Error("QR algorithm failed to converge in tridiagonal_eigh_f64")
                
            iter_count += 1
            
            # Form Wilkinson shift
            var g = (d[l + 1] - d[l]) / (Float64(2.0) * e[l])
            var r = pythag_f64(g, Float64(1.0))
            g = d[m_idx] - d[l] + e[l] / (g + copysign_f64(r, g))
            
            var s = Float64(1.0)
            var c = Float64(1.0)
            var p = Float64(0.0)
            
            var i = m_idx - 1
            while i >= l:
                var f = s * e[i]
                var b = c * e[i]
                r = pythag_f64(f, g)
                e[i + 1] = r
                if r == 0.0:
                    d[i + 1] -= p
                    e[m_idx] = 0.0
                    break
                s = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s + Float64(2.0) * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b
                
                # Update eigenvectors
                for k in range(m):
                    var f_vec = eigenvectors_out[k * m + i + 1]
                    eigenvectors_out[k * m + i + 1] = s * eigenvectors_out[k * m + i] + c * f_vec
                    eigenvectors_out[k * m + i] = c * eigenvectors_out[k * m + i] - s * f_vec
                    
                i -= 1
                
            if r == 0.0 and i >= l:
                continue
                
            d[l] -= p
            e[l] = g
            e[m_idx] = 0.0
            
    # Sort eigenvalues and corresponding eigenvectors (ascending order)
    for i in range(m - 1):
        var k = i
        var p_val = d[i]
        for j in range(i + 1, m):
            if d[j] < p_val:
                k = j
                p_val = d[j]
        if k != i:
            d[k] = d[i]
            d[i] = p_val
            for j in range(m):
                var temp = eigenvectors_out[j * m + i]
                eigenvectors_out[j * m + i] = eigenvectors_out[j * m + k]
                eigenvectors_out[j * m + k] = temp
                
    # Copy eigenvalues to output
    for i in range(m):
        eigenvalues_out[i] = d[i]
        
    d.free()
    e.free()


fn symmetric_eigh_native(
    A: UnsafePointer[Float64, MutAnyOrigin],  # Input symmetric [n x n], NOT modified
    n: Int,
    eigenvalues: UnsafePointer[Float64, MutAnyOrigin],   # Output [n]
    eigenvectors: UnsafePointer[Float64, MutAnyOrigin],  # Output [n x n]
    jitter: Float64 = 1e-12                              # Regularization
) raises:
    """Compute eigenvalues and eigenvectors of a symmetric matrix.
    
    Uses Householder reduction to tridiagonal form, then QR algorithm.
    This is a native Mojo implementation that doesn't rely on PyTorch/LAPACK.
    
    Args:
        A: Input symmetric matrix [n x n] row-major (NOT modified)
        n: Matrix size (should be small, n <= 32 recommended)
        eigenvalues: Output eigenvalues [n] (sorted ascending)
        eigenvectors: Output eigenvector matrix [n x n] (row-major, columns are eigenvectors)
        jitter: Small value added to diagonal for numerical stability
    """
    # Allocate working buffers
    var A_work = alloc[Float64](n * n)
    var Q = alloc[Float64](n * n)
    var diag = alloc[Float64](n)
    var offdiag = alloc[Float64](n)  # Allocate n for safety, only n-1 used
    var V_tridiag = alloc[Float64](n * n)
    
    # Copy A to working buffer and add jitter to diagonal
    for i in range(n * n):
        A_work[i] = A[i]
    for i in range(n):
        A_work[i * n + i] += jitter
    
    # Step 1: Householder reduction to tridiagonal form
    # A_work -> Q, diag, offdiag such that A = Q T Q^T
    householder_tridiagonalize_f64(A_work, n, Q, diag, offdiag)
    
    # Step 2: Tridiagonal eigendecomposition
    # T -> eigenvalues, V_tridiag such that T = V_tridiag Lambda V_tridiag^T
    tridiagonal_eigh_f64(diag, offdiag, n, eigenvalues, V_tridiag)
    
    # Step 3: Back-transform eigenvectors
    # eigenvectors = Q @ V_tridiag
    matrix_multiply_f64(Q, V_tridiag, n, n, n, eigenvectors)
    
    # Free working buffers
    A_work.free()
    Q.free()
    diag.free()
    offdiag.free()
    V_tridiag.free()
