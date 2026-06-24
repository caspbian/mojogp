"""Categorical kernel correlation matrix computation.

Implements all 5 categorical kernel variants from Saves et al. (2023):
  - GD  (Gower Distance): 1 param per variable, correlations in [0, 1]
  - CR  (Continuous Relaxation): L params per variable, correlations in [0, 1]
  - EHH (Exponential Homoscedastic Hypersphere): L(L-1)/2 params, [0, 1]
  - HH  (Homoscedastic Hypersphere): L(L-1)/2 params, [-1, 1]
  - FE  (Fully Exponential): L(L+1)/2 params, [0, 1]

Each categorical variable i has a small L_i x L_i correlation matrix R_i.
The overall categorical kernel is: k_cat(c_r, c_s) = prod_i R_i[c_r^i, c_s^i]

Reference: arXiv:2211.08262v4, Equations 7-15
"""

from math import exp as math_exp, sqrt, sin, cos, log
from memory import UnsafePointer
from .constants import (
    CAT_KERNEL_GD,
    CAT_KERNEL_CR,
    CAT_KERNEL_EHH,
    CAT_KERNEL_HH,
    CAT_KERNEL_FE,
    MAX_CAT_LEVELS,
    PI,
)


# ============================================================================
# GD (Gower Distance) Kernel
# ============================================================================

fn compute_gd_correlation(
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta: Float32,
    L: Int,
) -> None:
    """Compute GD correlation matrix for one categorical variable.
    
    R[l_r, l_s] = exp(-theta) if l_r != l_s, else 1.0
    
    This is the simplest variant: one scalar parameter theta >= 0 controls
    the correlation between all distinct levels equally.
    
    Args:
        R_ptr: Output buffer of size L*L (row-major).
        theta: Non-negative distance parameter.
        L: Number of levels for this variable.
    """
    var off_diag = math_exp(-theta)
    for i in range(L):
        for j in range(L):
            if i == j:
                R_ptr[i * L + j] = Float32(1.0)
            else:
                R_ptr[i * L + j] = off_diag


fn compute_gd_gradient(
    dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta: Float32,
    L: Int,
) -> None:
    """Compute dR/d(theta) for GD kernel.
    
    dR[l_r, l_s]/d(theta) = -exp(-theta) if l_r != l_s, else 0.0
    
    Args:
        dR_ptr: Output buffer of size L*L (row-major).
        theta: Non-negative distance parameter.
        L: Number of levels for this variable.
    """
    var neg_off_diag = -math_exp(-theta)
    for i in range(L):
        for j in range(L):
            if i == j:
                dR_ptr[i * L + j] = Float32(0.0)
            else:
                dR_ptr[i * L + j] = neg_off_diag


# ============================================================================
# CR (Continuous Relaxation) Kernel
# ============================================================================

fn compute_cr_correlation(
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
) -> None:
    """Compute CR correlation matrix for one categorical variable.
    
    R[l_r, l_s] = exp(-(theta[l_r] + theta[l_s])) if l_r != l_s, else 1.0
    
    Each level has its own parameter theta[l] >= 0.
    
    Args:
        R_ptr: Output buffer of size L*L (row-major).
        theta_ptr: L non-negative parameters.
        L: Number of levels.
    """
    for i in range(L):
        for j in range(L):
            if i == j:
                R_ptr[i * L + j] = Float32(1.0)
            else:
                R_ptr[i * L + j] = math_exp(-(theta_ptr[i] + theta_ptr[j]))


fn compute_cr_gradient(
    dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    param_index: Int,
) -> None:
    """Compute dR/d(theta[param_index]) for CR kernel.
    
    dR[l_r, l_s]/d(theta[k]) = -R[l_r, l_s] if (l_r == k or l_s == k) and l_r != l_s
                               = 0 otherwise
    
    Args:
        dR_ptr: Output buffer of size L*L (row-major).
        theta_ptr: L non-negative parameters.
        L: Number of levels.
        param_index: Which theta to differentiate with respect to.
    """
    var k = param_index
    for i in range(L):
        for j in range(L):
            if i == j:
                dR_ptr[i * L + j] = Float32(0.0)
            elif i == k or j == k:
                # dR/d(theta_k) = -exp(-(theta_i + theta_j)) when i==k or j==k
                dR_ptr[i * L + j] = -math_exp(-(theta_ptr[i] + theta_ptr[j]))
            else:
                dR_ptr[i * L + j] = Float32(0.0)


# ============================================================================
# Hypersphere Decomposition (shared by EHH, HH, FE)
# ============================================================================

fn compute_cholesky_factor(
    C_ptr: UnsafePointer[Float32, MutAnyOrigin],
    Theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
) -> None:
    """Compute lower-triangular Cholesky-like factor C from angle matrix Theta.
    
    Hypersphere decomposition (Eq. 13 in paper):
        C[1,1] = 1
        C[k,1] = cos(Theta[k,1])                                    for 2 <= k <= L
        C[k,k'] = cos(Theta[k,k']) * prod_{j=1}^{k'-1} sin(Theta[k,j])  for 2 <= k' < k
        C[k,k] = prod_{j=1}^{k-1} sin(Theta[k,j])                  for 2 <= k <= L
    
    Theta is stored as a flat array of the strictly lower-triangular elements,
    in row-major order: Theta[2,1], Theta[3,1], Theta[3,2], Theta[4,1], ...
    Total: L*(L-1)/2 elements.
    
    C is stored as L*L row-major (full matrix, upper triangle is zero).
    
    Args:
        C_ptr: Output L*L buffer (row-major).
        Theta_ptr: L*(L-1)/2 angle parameters in [0, pi].
        L: Number of levels.
    """
    # Zero out the matrix
    for i in range(L * L):
        C_ptr[i] = Float32(0.0)
    
    # C[0,0] = 1 (using 0-indexed)
    C_ptr[0] = Float32(1.0)
    
    # For each row k >= 1 (0-indexed)
    for k in range(1, L):
        # Compute product of sines for columns up to k
        # sin_products[j] = prod_{m=0}^{j-1} sin(Theta[k,m])
        # We build this incrementally
        var sin_prod = Float32(1.0)
        
        # Column 0: C[k,0] = cos(Theta[k,0])
        var theta_idx = _lower_tri_index(k, 0)
        var theta_val = Theta_ptr[theta_idx]
        C_ptr[k * L + 0] = cos(theta_val)
        sin_prod = sin(theta_val)
        
        # Columns 1 to k-1: C[k,j] = cos(Theta[k,j]) * prod_{m=0}^{j-1} sin(Theta[k,m])
        for j in range(1, k):
            theta_idx = _lower_tri_index(k, j)
            theta_val = Theta_ptr[theta_idx]
            C_ptr[k * L + j] = cos(theta_val) * sin_prod
            sin_prod *= sin(theta_val)
        
        # Diagonal: C[k,k] = prod_{j=0}^{k-1} sin(Theta[k,j])
        C_ptr[k * L + k] = sin_prod


@always_inline
fn _lower_tri_index(row: Int, col: Int) -> Int:
    """Convert (row, col) with row > col to flat index in lower-triangular storage.
    
    Storage order: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
    Index = row*(row-1)/2 + col
    """
    return row * (row - 1) // 2 + col


# ============================================================================
# EHH (Exponential Homoscedastic Hypersphere) Kernel
# ============================================================================

fn compute_ehh_correlation(
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute EHH correlation matrix.
    
    EHH uses hypersphere decomposition with zero diagonal in Theta:
    1. Compute C from angles Theta (off-diagonal only, diagonal of Theta is 0)
    2. Compute R_raw = C @ C^T (this is a valid correlation matrix)
    3. R[i,j] = exp(log_eps/2 * (1 - R_raw[i,j])) for i != j, R[i,i] = 1
    
    where log_eps = log(1e-12) ≈ -27.63 (ensures R[i,j] in [0, 1])
    Note: (1 - dot) not (dot - 1), so exponent is always <= 0 and R in (0, 1].
    
    Args:
        R_ptr: Output L*L buffer (row-major).
        theta_ptr: L*(L-1)/2 angle parameters in [0, pi].
        L: Number of levels.
        work_ptr: Workspace of at least L*L floats for C matrix.
    """
    # Step 1: Compute C from angles
    compute_cholesky_factor(work_ptr, theta_ptr, L)
    
    # Step 2: R_raw = C @ C^T
    var log_eps_half = Float32(-13.815510558)  # log(1e-12) / 2
    
    for i in range(L):
        for j in range(L):
            if i == j:
                R_ptr[i * L + j] = Float32(1.0)
            else:
                # Dot product of row i and row j of C
                var dot = Float32(0.0)
                var min_col = min(i, j) + 1  # C is lower triangular
                for c in range(min_col):
                    dot += work_ptr[i * L + c] * work_ptr[j * L + c]
                # Apply exponential mapping: (1 - dot) ensures exponent <= 0, so R in (0, 1]
                R_ptr[i * L + j] = math_exp(log_eps_half * (Float32(1.0) - dot))


fn compute_ehh_gradient(
    dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    param_index: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute dR/d(theta[param_index]) for EHH kernel.
    
    Chain rule: dR[i,j]/d(theta_k) = -R[i,j] * log_eps/2 * d(C@C^T)[i,j]/d(theta_k)
    Note: negative sign because R = exp(c * (1 - dot)), so dR/d(dot) = -c * R.
    
    where d(C@C^T)[i,j]/d(theta_k) = sum_c (dC[i,c]/d(theta_k) * C[j,c] + C[i,c] * dC[j,c]/d(theta_k))
    
    Args:
        dR_ptr: Output L*L buffer (row-major).
        theta_ptr: L*(L-1)/2 angle parameters.
        R_ptr: Pre-computed correlation matrix (from compute_ehh_correlation).
        L: Number of levels.
        param_index: Which angle parameter to differentiate.
        work_ptr: Workspace of at least 2*L*L floats (C and dC).
    """
    var log_eps_half = Float32(-13.815510558)  # log(1e-12) / 2
    
    # Compute C
    var C_ptr = work_ptr
    compute_cholesky_factor(C_ptr, theta_ptr, L)
    
    # Compute dC/d(theta[param_index])
    var dC_ptr = work_ptr + L * L
    _compute_dC_dtheta(dC_ptr, C_ptr, theta_ptr, L, param_index)
    
    # dR[i,j] = R[i,j] * log_eps/2 * d(C@C^T)[i,j]
    for i in range(L):
        for j in range(L):
            if i == j:
                dR_ptr[i * L + j] = Float32(0.0)
            else:
                # d(C@C^T)[i,j] = sum_c (dC[i,c]*C[j,c] + C[i,c]*dC[j,c])
                var d_dot = Float32(0.0)
                for c in range(L):
                    d_dot += dC_ptr[i * L + c] * C_ptr[j * L + c]
                    d_dot += C_ptr[i * L + c] * dC_ptr[j * L + c]
                # Negative sign: d/d(dot) exp(c*(1-dot)) = -c*R
                dR_ptr[i * L + j] = -R_ptr[i * L + j] * log_eps_half * d_dot


fn _compute_dC_dtheta(
    dC_ptr: UnsafePointer[Float32, MutAnyOrigin],
    C_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    param_index: Int,
) -> None:
    """Compute dC/d(theta[param_index]).
    
    The param_index maps to a specific (row, col) in the lower-triangular
    angle matrix via _lower_tri_index inverse.
    
    dC[k,j]/d(theta[r,c]) is non-zero only when k == r (same row of C).
    Within row r:
    - If j == c: dC[r,c]/d(theta[r,c]) = -sin(theta[r,c]) * prod_{m<c} sin(theta[r,m])
    - If j > c and j < r: dC[r,j]/d(theta[r,c]) = cos(theta[r,j]) * cos(theta[r,c]) / sin(theta[r,c]) * prod_{m<j} sin(theta[r,m])
      (but only if sin(theta[r,c]) != 0)
    - If j == r (diagonal): dC[r,r]/d(theta[r,c]) = cos(theta[r,c]) / sin(theta[r,c]) * prod_{m<r} sin(theta[r,m])
    
    Simplified: dC[r,j]/d(theta[r,c]) = C[r,j] * (-sin/cos)(theta[r,c]) for j==c
                                        = C[r,j] * (cos/sin)(theta[r,c]) for j>c
    """
    # Zero out dC
    for i in range(L * L):
        dC_ptr[i] = Float32(0.0)
    
    # Find which (row, col) this param_index corresponds to
    var target_row = 0
    var target_col = 0
    var idx = 0
    for r in range(1, L):
        for c in range(r):
            if idx == param_index:
                target_row = r
                target_col = c
            idx += 1
    
    var r = target_row
    var c_param = target_col
    
    # Only row r of C is affected
    var theta_rc = theta_ptr[_lower_tri_index(r, c_param)]
    var sin_rc = sin(theta_rc)
    var cos_rc = cos(theta_rc)
    
    # For numerical stability, handle sin_rc ≈ 0
    var eps = Float32(1e-10)
    
    # Recompute sin products for row r
    # sin_prod_before[j] = prod_{m=0}^{j-1} sin(theta[r,m])
    # We need these to compute dC entries
    
    # Column c_param: dC[r, c_param] = -sin(theta[r,c_param]) * prod_{m<c_param} sin(theta[r,m])
    var sin_prod = Float32(1.0)
    for m in range(c_param):
        var theta_rm = theta_ptr[_lower_tri_index(r, m)]
        sin_prod *= sin(theta_rm)
    dC_ptr[r * L + c_param] = -sin_rc * sin_prod
    
    # For columns j > c_param: the factor cos(theta[r,c_param]) in the product
    # changes to -sin(theta[r,c_param]) * d(theta[r,c_param])/d(theta[r,c_param])
    # Actually, for j > c_param, sin(theta[r,c_param]) appears in the product.
    # dC[r,j]/d(theta[r,c_param]) = C[r,j] * cos(theta[r,c_param]) / sin(theta[r,c_param])
    # (when sin(theta[r,c_param]) != 0)
    
    if sin_rc > eps or sin_rc < -eps:
        var ratio = cos_rc / sin_rc  # cot(theta[r,c_param])
        for j in range(c_param + 1, r):
            dC_ptr[r * L + j] = C_ptr[r * L + j] * ratio
        # Diagonal column r
        dC_ptr[r * L + r] = C_ptr[r * L + r] * ratio
    else:
        # sin ≈ 0 means theta ≈ 0 or pi
        # Use L'Hopital or finite difference approach
        # For theta ≈ 0: cos/sin → ∞, but C[r,j] → 0 for j > c_param
        # The product C[r,j] * cos/sin is finite
        # Recompute directly
        sin_prod = Float32(1.0)
        for m in range(c_param):
            var theta_rm = theta_ptr[_lower_tri_index(r, m)]
            sin_prod *= sin(theta_rm)
        
        # For j > c_param, C[r,j] contains sin(theta[r,c_param]) as a factor
        # dC[r,j]/d(theta[r,c_param]) replaces sin with cos in that factor
        var cos_factor = cos_rc * sin_prod
        
        for j in range(c_param + 1, r):
            # Rebuild C[r,j] but replace sin(theta[r,c_param]) with cos(theta[r,c_param])
            var val = cos_factor
            for m in range(c_param + 1, j):
                var theta_rm = theta_ptr[_lower_tri_index(r, m)]
                val *= sin(theta_rm)
            var theta_rj = theta_ptr[_lower_tri_index(r, j)]
            val *= cos(theta_rj)
            dC_ptr[r * L + j] = val
        
        # Diagonal: replace sin(theta[r,c_param]) with cos in the full product
        var diag_val = cos_factor
        for m in range(c_param + 1, r):
            var theta_rm = theta_ptr[_lower_tri_index(r, m)]
            diag_val *= sin(theta_rm)
        dC_ptr[r * L + r] = diag_val


# ============================================================================
# HH (Homoscedastic Hypersphere) Kernel
# ============================================================================

fn compute_hh_correlation(
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute HH correlation matrix: R = C @ C^T directly.
    
    Unlike EHH, HH does NOT apply the exponential mapping, so correlations
    can be negative (range [-1, 1]).
    
    Args:
        R_ptr: Output L*L buffer (row-major).
        theta_ptr: L*(L-1)/2 angle parameters in [0, pi].
        L: Number of levels.
        work_ptr: Workspace of at least L*L floats for C matrix.
    """
    compute_cholesky_factor(work_ptr, theta_ptr, L)
    
    # R = C @ C^T
    for i in range(L):
        for j in range(L):
            var dot = Float32(0.0)
            var max_col = min(i, j) + 1
            for c in range(max_col):
                dot += work_ptr[i * L + c] * work_ptr[j * L + c]
            R_ptr[i * L + j] = dot


fn compute_hh_gradient(
    dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    param_index: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute dR/d(theta[param_index]) for HH kernel.
    
    d(C@C^T)[i,j] = sum_c (dC[i,c]*C[j,c] + C[i,c]*dC[j,c])
    
    Args:
        dR_ptr: Output L*L buffer.
        theta_ptr: Angle parameters.
        L: Number of levels.
        param_index: Which parameter.
        work_ptr: Workspace of at least 2*L*L floats.
    """
    var C_ptr = work_ptr
    compute_cholesky_factor(C_ptr, theta_ptr, L)
    
    var dC_ptr = work_ptr + L * L
    _compute_dC_dtheta(dC_ptr, C_ptr, theta_ptr, L, param_index)
    
    for i in range(L):
        for j in range(L):
            var d_dot = Float32(0.0)
            for c in range(L):
                d_dot += dC_ptr[i * L + c] * C_ptr[j * L + c]
                d_dot += C_ptr[i * L + c] * dC_ptr[j * L + c]
            dR_ptr[i * L + j] = d_dot


# ============================================================================
# FE (Fully Exponential) Kernel
# ============================================================================

fn compute_fe_correlation(
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute FE correlation matrix.
    
    FE uses both diagonal and off-diagonal parameters:
    - First L*(L-1)/2 params: off-diagonal angles (same as EHH)
    - Next L params: diagonal parameters (non-negative)
    
    Phi[i,j] = theta_diag[i] + theta_diag[j] + log_eps/2 * (C@C^T[i,j] - 1)  for i != j
    R[i,j] = exp(-Phi[i,j]) for i != j, R[i,i] = 1
    
    Total params: L*(L-1)/2 + L = L*(L+1)/2
    
    Args:
        R_ptr: Output L*L buffer.
        theta_ptr: L*(L+1)/2 parameters. First L*(L-1)/2 are angles, next L are diagonal.
        L: Number of levels.
        work_ptr: Workspace of at least L*L floats.
    """
    var num_angles = L * (L - 1) // 2
    var angle_ptr = theta_ptr
    var diag_ptr = theta_ptr + num_angles
    
    # Compute C from angles
    compute_cholesky_factor(work_ptr, angle_ptr, L)
    
    var log_eps_half = Float32(-13.815510558)  # log(1e-12) / 2
    
    for i in range(L):
        for j in range(L):
            if i == j:
                R_ptr[i * L + j] = Float32(1.0)
            else:
                # C@C^T[i,j]
                var dot = Float32(0.0)
                var max_col = min(i, j) + 1
                for c in range(max_col):
                    dot += work_ptr[i * L + c] * work_ptr[j * L + c]
                
                var phi = diag_ptr[i] + diag_ptr[j] + log_eps_half * (dot - Float32(1.0))
                R_ptr[i * L + j] = math_exp(-phi)


fn compute_fe_gradient(
    dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    param_index: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute dR/d(theta[param_index]) for FE kernel.
    
    For angle parameters (param_index < L*(L-1)/2):
        dR[i,j] = -R[i,j] * log_eps/2 * d(C@C^T)[i,j]/d(theta_k)
    
    For diagonal parameters (param_index >= L*(L-1)/2):
        Let k = param_index - L*(L-1)/2 (the diagonal index)
        dR[i,j]/d(diag_k) = -R[i,j] if (i == k or j == k) and i != j
                            = 0 otherwise
    
    Args:
        dR_ptr: Output L*L buffer.
        theta_ptr: L*(L+1)/2 parameters.
        R_ptr: Pre-computed correlation matrix.
        L: Number of levels.
        param_index: Which parameter.
        work_ptr: Workspace of at least 2*L*L floats.
    """
    var num_angles = L * (L - 1) // 2
    var log_eps_half = Float32(-13.815510558)
    
    if param_index < num_angles:
        # Angle parameter — chain through C@C^T
        var C_ptr = work_ptr
        compute_cholesky_factor(C_ptr, theta_ptr, L)
        
        var dC_ptr = work_ptr + L * L
        _compute_dC_dtheta(dC_ptr, C_ptr, theta_ptr, L, param_index)
        
        for i in range(L):
            for j in range(L):
                if i == j:
                    dR_ptr[i * L + j] = Float32(0.0)
                else:
                    var d_dot = Float32(0.0)
                    for c in range(L):
                        d_dot += dC_ptr[i * L + c] * C_ptr[j * L + c]
                        d_dot += C_ptr[i * L + c] * dC_ptr[j * L + c]
                    # dR = -R * d(phi) = -R * log_eps/2 * d(C@C^T)
                    # But phi = diag_i + diag_j + log_eps/2 * (dot - 1)
                    # R = exp(-phi), so dR/d(angle) = -R * log_eps/2 * d_dot
                    dR_ptr[i * L + j] = -R_ptr[i * L + j] * log_eps_half * d_dot
    else:
        # Diagonal parameter
        var k = param_index - num_angles
        for i in range(L):
            for j in range(L):
                if i == j:
                    dR_ptr[i * L + j] = Float32(0.0)
                elif i == k or j == k:
                    # dR/d(diag_k) = -R[i,j] * 1 (since d(phi)/d(diag_k) = 1)
                    dR_ptr[i * L + j] = -R_ptr[i * L + j]
                else:
                    dR_ptr[i * L + j] = Float32(0.0)


# ============================================================================
# Unified Interface
# ============================================================================

fn compute_correlation_matrix(
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    cat_kernel_type: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute correlation matrix for any categorical kernel variant.
    
    Args:
        R_ptr: Output L*L buffer (row-major).
        theta_ptr: Parameters (count depends on variant).
        L: Number of levels.
        cat_kernel_type: One of CAT_KERNEL_GD, CR, EHH, HH, FE.
        work_ptr: Workspace (at least L*L floats, 2*L*L for EHH/HH/FE).
    """
    if cat_kernel_type == CAT_KERNEL_GD:
        compute_gd_correlation(R_ptr, theta_ptr[0], L)
    elif cat_kernel_type == CAT_KERNEL_CR:
        compute_cr_correlation(R_ptr, theta_ptr, L)
    elif cat_kernel_type == CAT_KERNEL_EHH:
        compute_ehh_correlation(R_ptr, theta_ptr, L, work_ptr)
    elif cat_kernel_type == CAT_KERNEL_HH:
        compute_hh_correlation(R_ptr, theta_ptr, L, work_ptr)
    elif cat_kernel_type == CAT_KERNEL_FE:
        compute_fe_correlation(R_ptr, theta_ptr, L, work_ptr)


fn compute_correlation_gradient(
    dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
    theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
    R_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L: Int,
    cat_kernel_type: Int,
    param_index: Int,
    work_ptr: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Compute gradient of correlation matrix for any variant.
    
    Args:
        dR_ptr: Output L*L buffer.
        theta_ptr: Parameters.
        R_ptr: Pre-computed correlation matrix.
        L: Number of levels.
        cat_kernel_type: Kernel variant.
        param_index: Which parameter to differentiate.
        work_ptr: Workspace (at least 2*L*L floats).
    """
    if cat_kernel_type == CAT_KERNEL_GD:
        compute_gd_gradient(dR_ptr, theta_ptr[0], L)
    elif cat_kernel_type == CAT_KERNEL_CR:
        compute_cr_gradient(dR_ptr, theta_ptr, L, param_index)
    elif cat_kernel_type == CAT_KERNEL_EHH:
        compute_ehh_gradient(dR_ptr, theta_ptr, R_ptr, L, param_index, work_ptr)
    elif cat_kernel_type == CAT_KERNEL_HH:
        compute_hh_gradient(dR_ptr, theta_ptr, L, param_index, work_ptr)
    elif cat_kernel_type == CAT_KERNEL_FE:
        compute_fe_gradient(dR_ptr, theta_ptr, R_ptr, L, param_index, work_ptr)


fn num_params_for_variant(L: Int, cat_kernel_type: Int) -> Int:
    """Return the number of parameters for a given variant and level count.
    
    Args:
        L: Number of levels.
        cat_kernel_type: Kernel variant.
    
    Returns:
        Number of parameters.
    """
    if cat_kernel_type == CAT_KERNEL_GD:
        return 1
    elif cat_kernel_type == CAT_KERNEL_CR:
        return L
    elif cat_kernel_type == CAT_KERNEL_EHH:
        return L * (L - 1) // 2
    elif cat_kernel_type == CAT_KERNEL_HH:
        return L * (L - 1) // 2
    elif cat_kernel_type == CAT_KERNEL_FE:
        return L * (L + 1) // 2
    else:
        return 0


# ============================================================================
# Categorical kernel evaluation (for use in GPU kernels)
# ============================================================================

@always_inline
fn categorical_kernel_value(
    c_i_ptr: UnsafePointer[Int32, MutAnyOrigin],
    c_j_ptr: UnsafePointer[Int32, MutAnyOrigin],
    num_cat_vars: Int,
    corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
) -> Float32:
    """Evaluate categorical kernel k_cat(c_i, c_j) = prod_v R_v[c_i^v, c_j^v].
    
    This is the inner-loop function called per (i,j) pair in GPU kernels.
    It looks up pre-computed correlation matrices.
    
    Args:
        c_i_ptr: Categorical indices for point i (num_cat_vars values).
        c_j_ptr: Categorical indices for point j (num_cat_vars values).
        num_cat_vars: Number of categorical variables.
        corr_flat_ptr: Flattened correlation matrices (all variables concatenated).
        corr_offsets_ptr: Offset into corr_flat for each variable.
        corr_levels_ptr: Number of levels for each variable.
    
    Returns:
        Product of correlation lookups across all categorical variables.
    """
    var result = Float32(1.0)
    for v in range(num_cat_vars):
        var l_i = Int(c_i_ptr[v])
        var l_j = Int(c_j_ptr[v])
        if l_i != l_j:
            var offset = Int(corr_offsets_ptr[v])
            var L = Int(corr_levels_ptr[v])
            result *= corr_flat_ptr[offset + l_i * L + l_j]
    return result
