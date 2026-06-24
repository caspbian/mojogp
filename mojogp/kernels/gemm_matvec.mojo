"""GEMM-based matrix-vector multiplication for materialized kernels.

Provides efficient K @ v computation using the materialized kernel matrix.
Uses vendor BLAS (cuBLAS/rocBLAS) with full float32 precision by default.

The MAX AI Kernels matmul uses TF32 tensor cores which truncate float32 mantissa
from 23 bits to 10 bits, causing ~0.03% error per matmul. This compounds in CG
iterations and causes divergence at low noise levels. The vendor BLAS with
use_tf32=False provides full float32 precision and is actually faster on most GPUs.
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from buffer import NDBuffer
from linalg.matmul.vendor.blas import matmul as blas_matmul


# =============================================================================
# Configuration
# =============================================================================

# Minimum size to use MAX matmul (overhead dominates for very small matrices)
alias MIN_SIZE_FOR_MAX = 200

# Runtime control for MAX matmul using environment variable
# Set MOJOGP_USE_MAX_MATMUL=0 to disable MAX matmul (defaults to enabled)
fn get_use_max_matmul() -> Bool:
    """Get whether MAX matmul is enabled (checks environment variable)."""
    try:
        from os import getenv
        var env_val = getenv("MOJOGP_USE_MAX_MATMUL", "1")
        return env_val != "0"
    except:
        return True  # Default to enabled if env var check fails


fn set_use_max_matmul(enable: Bool):
    """Set whether to use MAX matmul (True) or naive GEMM (False).
    
    Note: This sets the MOJOGP_USE_MAX_MATMUL environment variable.
    """
    try:
        from os import setenv
        setenv("MOJOGP_USE_MAX_MATMUL", "1" if enable else "0")
    except:
        pass  # Silently fail if setenv not available


# =============================================================================
# GEMM Kernels
# =============================================================================

fn kernel_gemm_matvec(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × num_cols output (column-major)
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × n kernel matrix (row-major)
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × num_cols input (column-major)
    n: Int,
    num_cols: Int,
) -> None:
    """Compute out = K @ v using simple GEMM.
    
    K is row-major: K[i, j] = K_ptr[i * n + j]
    v is column-major: v[i, col] = v_ptr[col * n + i]
    out is column-major: out[i, col] = out_ptr[col * n + i]
    
    Each thread computes one element out[i, col].
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major
        K_ptr: Kernel matrix [n, n] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Matrix dimension
        num_cols: Number of RHS columns
    """
    var i = block_idx.x * block_dim.x + thread_idx.x  # row in output
    var col = block_idx.y * block_dim.y + thread_idx.y  # column in output
    
    if i >= UInt(n) or col >= UInt(num_cols):
        return
    
    # Compute out[i, col] = sum_j K[i, j] * v[j, col]
    var sum_val = Float32(0.0)
    for j in range(n):
        # K[i, j] in row-major: K_ptr[i * n + j]
        # v[j, col] in column-major: v_ptr[col * n + j]
        sum_val += K_ptr[UInt(i) * UInt(n) + UInt(j)] * v_ptr[UInt(col) * UInt(n) + UInt(j)]
    
    # out[i, col] in column-major: out_ptr[col * n + i]
    out_ptr[UInt(col) * UInt(n) + UInt(i)] = sum_val


fn kernel_add_noise_diagonal(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    noise: Float32,
) -> None:
    """Add noise * v to out: out += noise * v.
    
    This completes the (K + noise*I) @ v computation.
    Both out and v are column-major: [num_cols × n]
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major (modified in-place)
        v_ptr: Input vectors [n, num_cols] column-major
        n: Vector dimension
        num_cols: Number of columns
        noise: Noise variance to add
    """
    var i = block_idx.x * block_dim.x + thread_idx.x  # row
    var col = block_idx.y * block_dim.y + thread_idx.y  # column
    
    if i >= UInt(n) or col >= UInt(num_cols):
        return
    
    var idx = UInt(col) * UInt(n) + UInt(i)
    out_ptr[idx] += noise * v_ptr[idx]


# =============================================================================
# Host Functions
# =============================================================================

fn gemm_matvec_naive(
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × num_cols output
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × n kernel matrix
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × num_cols input
    n: Int,
    num_cols: Int,
) raises:
    """Compute out = K @ v using naive GEMM kernel.
    
    This is the original implementation kept as a fallback.
    
    Args:
        ctx: GPU device context
        out_ptr: Output buffer [n, num_cols] column-major
        K_ptr: Kernel matrix [n, n] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Matrix dimension
        num_cols: Number of RHS columns
    """
    alias BLOCK_SIZE_X = 16
    alias BLOCK_SIZE_Y = 16
    
    var grid_dim_x = (n + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    var grid_dim_y = (num_cols + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    var grid_dim = (grid_dim_x, grid_dim_y)
    var block_dim = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
    
    ctx.enqueue_function[kernel_gemm_matvec](
        out_ptr, K_ptr, v_ptr, n, num_cols,
        grid_dim=grid_dim, block_dim=block_dim
    )


fn gemm_matvec_blas(
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × num_cols output
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × n kernel matrix
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × num_cols input
    n: Int,
    num_cols: Int,
) raises:
    """Compute out = K @ v using vendor BLAS (cuBLAS/rocBLAS) with full float32 precision.
    
    This function uses vendor BLAS with use_tf32=False to ensure full float32 precision.
    The MAX AI Kernels matmul uses TF32 tensor cores which cause ~0.03% error per matmul,
    leading to CG divergence at low noise levels. Vendor BLAS with full precision is
    actually faster on most GPUs and provides correct CG convergence.
    
    Memory layout:
    - K: n×n row-major
    - v: n×num_cols column-major
    - out: n×num_cols column-major
    
    We use the formula: Out[num_cols,n] = V[num_cols,n] @ K[n,n]^T
    where column-major [n, num_cols] is reinterpreted as row-major [num_cols, n].
    
    Args:
        ctx: GPU device context
        out_ptr: Output buffer [n, num_cols] column-major
        K_ptr: Kernel matrix [n, n] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Matrix dimension
        num_cols: Number of RHS columns
    """
    # Create NDBuffers with runtime dimensions
    # Reinterpret column-major [n, num_cols] as row-major [num_cols, n]
    var v_ndbuf = NDBuffer[DType.float32, 2](v_ptr, (num_cols, n))
    var K_ndbuf = NDBuffer[DType.float32, 2](K_ptr, (n, n))
    var out_ndbuf = NDBuffer[DType.float32, 2](out_ptr, (num_cols, n))
    
    # Call vendor BLAS matmul with full float32 precision (no TF32 tensor cores)
    # This fixes CG divergence at low noise caused by TF32 precision loss
    blas_matmul[use_tf32=False](
        ctx, out_ndbuf, v_ndbuf, K_ndbuf,
        c_row_major=True, transpose_a=False, transpose_b=True
    )


fn gemm_matvec(
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × num_cols output
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × n kernel matrix
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],    # n × num_cols input
    n: Int,
    num_cols: Int,
) raises:
    """Compute out = K @ v using best available GEMM implementation.
    
    Uses vendor BLAS (cuBLAS/rocBLAS) with full float32 precision for larger matrices,
    and naive GEMM for very small matrices where BLAS overhead dominates.
    
    The MOJOGP_USE_MAX_MATMUL environment variable controls which implementation to use:
    - MOJOGP_USE_MAX_MATMUL=1 (default): Use vendor BLAS with full float32 precision
    - MOJOGP_USE_MAX_MATMUL=0: Use naive GEMM (slower but always works)
    
    Note: The MAX AI Kernels matmul is NOT used because it uses TF32 tensor cores
    which cause ~0.03% error per matmul, leading to CG divergence at low noise levels.
    
    Args:
        ctx: GPU device context
        out_ptr: Output buffer [n, num_cols] column-major
        K_ptr: Kernel matrix [n, n] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Matrix dimension
        num_cols: Number of RHS columns
    """
    # Use vendor BLAS for larger matrices (5-10x faster than naive, full float32 precision)
    # Fall back to naive for very small matrices where BLAS overhead dominates
    if get_use_max_matmul() and n >= MIN_SIZE_FOR_MAX:
        gemm_matvec_blas(ctx, out_ptr, K_ptr, v_ptr, n, num_cols)
    else:
        gemm_matvec_naive(ctx, out_ptr, K_ptr, v_ptr, n, num_cols)


fn add_noise_diagonal(
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    noise: Float32,
) raises:
    """Add noise * v to out: out += noise * v.
    
    This completes the (K + noise*I) @ v computation.
    
    Args:
        ctx: GPU device context
        out_ptr: Output buffer [n, num_cols] column-major (modified in-place)
        v_ptr: Input vectors [n, num_cols] column-major
        n: Vector dimension
        num_cols: Number of columns
        noise: Noise variance to add
    """
    alias BLOCK_SIZE_X = 16
    alias BLOCK_SIZE_Y = 16
    
    var grid_dim_x = (n + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    var grid_dim_y = (num_cols + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    var grid_dim = (grid_dim_x, grid_dim_y)
    var block_dim = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
    
    ctx.enqueue_function[kernel_add_noise_diagonal](
        out_ptr, v_ptr, n, num_cols, noise,
        grid_dim=grid_dim, block_dim=block_dim
    )
