"""GPU utility kernels for MojoGP.

This module provides GPU-optimized utility functions to reduce CPU-GPU transfers
and improve memory utilization.

Key optimizations:
1. GPU-based Rademacher probe generation (avoids host allocation and transfer)
2. GPU-based column extraction (avoids D->H->D round trips)
3. GPU-based vector operations that stay on device
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

alias float_dtype = DType.float32


# =============================================================================
# GPU Random Number Generation (Hash-based)
# =============================================================================

fn kernel_generate_rademacher(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    seed: UInt64,
) -> None:
    """Generate Rademacher random vectors (+1 or -1) directly on GPU.
    
    Uses a simple hash-based PRNG (xorshift) for fast GPU random generation.
    This avoids the need to generate probes on host and copy to device.
    
    Args:
        out_ptr: Output buffer [n * num_cols] column-major
        n: Number of elements per column
        num_cols: Number of columns
        seed: Random seed
    
    Memory layout: Column-major [num_cols × n]
    Index: out[col, row] = out_ptr[col * n + row]
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y * block_dim.y + thread_idx.y
    
    if row >= UInt(n) or col >= UInt(num_cols):
        return
    
    var idx = UInt(col) * UInt(n) + UInt(row)
    
    # Simple hash-based random: combine seed, row, col
    # Using xorshift-style mixing
    var state = seed ^ (UInt64(row) * UInt64(2654435761)) ^ (UInt64(col) * UInt64(2246822519))
    state ^= state >> 17
    state ^= state << 31
    state ^= state >> 8
    
    # Use lowest bit to determine +1 or -1
    var rand_bit = state & UInt64(1)
    out_ptr[idx] = Float32(1.0) if rand_bit == 1 else Float32(-1.0)


fn kernel_generate_rademacher_single_col(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    seed: UInt64,
) -> None:
    """Generate a single Rademacher random vector (+1 or -1) directly on GPU.
    
    Args:
        out_ptr: Output buffer [n]
        n: Number of elements
        seed: Random seed
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    # Simple hash-based random
    var state = seed ^ (UInt64(idx) * UInt64(2654435761))
    state ^= state >> 17
    state ^= state << 31
    state ^= state >> 8
    
    var rand_bit = state & UInt64(1)
    out_ptr[idx] = Float32(1.0) if rand_bit == 1 else Float32(-1.0)


# =============================================================================
# GPU Column Operations
# =============================================================================

fn kernel_extract_column(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_idx: Int,
) -> None:
    """Extract a single column from a column-major matrix.
    
    Args:
        out_ptr: Output buffer [n]
        src_ptr: Source matrix [n * num_cols] column-major
        n: Number of rows
        col_idx: Column index to extract
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    out_ptr[idx] = src_ptr[UInt(col_idx) * UInt(n) + idx]


fn kernel_set_column(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_idx: Int,
) -> None:
    """Set a single column in a column-major matrix.
    
    Args:
        dst_ptr: Destination matrix [n * num_cols] column-major
        src_ptr: Source vector [n]
        n: Number of rows
        col_idx: Column index to set
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    dst_ptr[UInt(col_idx) * UInt(n) + idx] = src_ptr[idx]


fn kernel_copy_with_offset(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
    dst_offset: Int,
    src_offset: Int,
) -> None:
    """Copy with offsets: dst[dst_offset + i] = src[src_offset + i].
    
    Args:
        dst_ptr: Destination buffer
        src_ptr: Source buffer
        size: Number of elements to copy
        dst_offset: Offset in destination
        src_offset: Offset in source
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(size):
        return
    
    dst_ptr[UInt(dst_offset) + idx] = src_ptr[UInt(src_offset) + idx]


# =============================================================================
# GPU Vector Operations (Stay on Device)
# =============================================================================

fn kernel_normalize_vector(
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    norm_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """Normalize vector in-place: v = v / ||v||.
    
    Note: norm_ptr[0] must contain ||v|| computed beforehand.
    
    Args:
        v_ptr: Vector to normalize [n]
        norm_ptr: Pointer to norm value [1]
        n: Vector length
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    var norm = norm_ptr[0]
    if norm > Float32(1e-10):
        v_ptr[idx] /= norm


fn kernel_orthogonalize(
    w_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_curr_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_prev_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha: Float32,
    beta: Float32,
    n: Int,
) -> None:
    """Lanczos orthogonalization: w = w - alpha * v_curr - beta * v_prev.
    
    Args:
        w_ptr: Work vector [n] (modified in-place)
        v_curr_ptr: Current Lanczos vector [n]
        v_prev_ptr: Previous Lanczos vector [n]
        alpha: Diagonal coefficient
        beta: Off-diagonal coefficient
        n: Vector length
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    w_ptr[idx] -= alpha * v_curr_ptr[idx] + beta * v_prev_ptr[idx]


fn kernel_update_lanczos_vectors(
    v_curr_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_prev_ptr: UnsafePointer[Float32, MutAnyOrigin],
    w_ptr: UnsafePointer[Float32, MutAnyOrigin],
    beta_inv: Float32,
    n: Int,
) -> None:
    """Update Lanczos vectors: v_prev = v_curr, v_curr = w / beta.
    
    Args:
        v_curr_ptr: Current vector [n] (becomes new v_curr = w/beta)
        v_prev_ptr: Previous vector [n] (becomes old v_curr)
        w_ptr: Work vector [n]
        beta_inv: 1.0 / beta
        n: Vector length
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    var v_curr_old = v_curr_ptr[idx]
    v_prev_ptr[idx] = v_curr_old
    v_curr_ptr[idx] = w_ptr[idx] * beta_inv


fn kernel_store_lanczos_vector(
    Q_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    col_idx: Int,
) -> None:
    """Store Lanczos vector in Q matrix (column-major).
    
    Args:
        Q_ptr: Q matrix [n * r] column-major
        v_ptr: Lanczos vector [n]
        n: Number of rows
        col_idx: Column index
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    Q_ptr[UInt(col_idx) * UInt(n) + idx] = v_ptr[idx]


# =============================================================================
# Host Functions for GPU Utilities
# =============================================================================

fn generate_rademacher_probes_gpu(
    ctx: DeviceContext,
    n: Int,
    num_cols: Int,
    seed: Int = 42,
) raises -> DeviceBuffer[float_dtype]:
    """Generate Rademacher probe vectors directly on GPU.
    
    This avoids:
    1. Host buffer allocation
    2. CPU random number generation
    3. Host-to-device copy
    
    Args:
        ctx: GPU device context
        n: Number of elements per probe
        num_cols: Number of probe vectors
        seed: Random seed
        
    Returns:
        DeviceBuffer containing probe vectors [n * num_cols] column-major
    """
    var probes_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    
    var threads_x = 16
    var threads_y = 16
    var blocks_x = (n + threads_x - 1) // threads_x
    var blocks_y = (num_cols + threads_y - 1) // threads_y
    
    ctx.enqueue_function[kernel_generate_rademacher](
        probes_device.unsafe_ptr(), n, num_cols, UInt64(seed),
        grid_dim=(blocks_x, blocks_y), block_dim=(threads_x, threads_y)
    )
    ctx.synchronize()
    
    return probes_device


fn generate_single_rademacher_probe_gpu(
    ctx: DeviceContext,
    n: Int,
    seed: Int = 42,
) raises -> DeviceBuffer[float_dtype]:
    """Generate a single Rademacher probe vector directly on GPU.
    
    Args:
        ctx: GPU device context
        n: Number of elements
        seed: Random seed
        
    Returns:
        DeviceBuffer containing probe vector [n]
    """
    var probe_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    var threads = 256
    var blocks = (n + threads - 1) // threads
    
    ctx.enqueue_function[kernel_generate_rademacher_single_col](
        probe_device.unsafe_ptr(), n, UInt64(seed),
        grid_dim=(blocks,), block_dim=(threads,)
    )
    ctx.synchronize()
    
    return probe_device


fn extract_column_gpu(
    ctx: DeviceContext,
    src_device: DeviceBuffer[float_dtype],
    n: Int,
    col_idx: Int,
) raises -> DeviceBuffer[float_dtype]:
    """Extract a column from a matrix on GPU (no host transfer).
    
    Args:
        ctx: GPU device context
        src_device: Source matrix [n * num_cols] column-major
        n: Number of rows
        col_idx: Column index to extract
        
    Returns:
        DeviceBuffer containing extracted column [n]
    """
    var col_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    var threads = 256
    var blocks = (n + threads - 1) // threads
    
    ctx.enqueue_function[kernel_extract_column](
        col_device.unsafe_ptr(), src_device.unsafe_ptr(), n, col_idx,
        grid_dim=(blocks,), block_dim=(threads,)
    )
    ctx.synchronize()
    
    return col_device
