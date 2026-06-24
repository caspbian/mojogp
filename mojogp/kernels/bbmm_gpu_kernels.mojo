"""BBMM GPU Kernels.

This module contains GPU kernel functions used by the BBMM (Black-Box Matrix-Matrix)
CG solver for dot products, column operations, and scaling. Extracted from
combined_inv_quad_logdet.mojo as a pure refactoring — no logic changes.

Kernels:
- kernel_dot_single_vs_strided: Single vector dot product against strided output
- kernel_dot_batched_vs_strided: Batched column dot products against strided output
- kernel_copy_column: Copy a single column into a column-major matrix
- kernel_copy_columns: Copy multiple columns with offset
- kernel_extract_columns_range: Extract a range of columns
- kernel_scale_columns_by_norms: Scale each column by its norm (in-place)
- scale_columns_by_norms: Host wrapper for kernel_scale_columns_by_norms
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from gpu.primitives.warp import sum as warp_sum
from gpu.sync import barrier
from gpu.memory import AddressSpace
from gpu.globals import WARP_SIZE
from memory import UnsafePointer, stack_allocation

alias float_dtype = DType.float32


# =============================================================================
# Fused Gradient Dot Product Kernels
# =============================================================================

fn kernel_dot_single_vs_strided(
    a_ptr: UnsafePointer[Float32, MutAnyOrigin],
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    result_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_params: Int,
) -> None:
    """Compute result[p] = a^T @ b[p*n : (p+1)*n] for p = 0..num_params-1.
    
    Used for fused gradient data term: alpha^T @ (dK/dp @ alpha) for all params.
    Grid: (num_params,), Block: (256,) - 8 warps per param.
    
    Args:
        a_ptr: Single vector [n] (e.g., alpha)
        b_ptr: Strided output [num_params * n] from fused gradient kernel
        result_ptr: Output [num_params] dot products
        n: Vector length
        num_params: Number of parameters
    """
    var p = block_idx.x
    var tid = Int(thread_idx.x)         # 0-255
    var num_threads = Int(block_dim.x)  # 256
    var warp_id = tid // WARP_SIZE      # 0-7
    var lane = tid % WARP_SIZE          # 0-31
    
    if p >= UInt(num_params):
        return
    
    # Shared memory for per-warp partial sums
    alias NUM_WARPS = 8
    var warp_sums = stack_allocation[NUM_WARPS, Float32, address_space = AddressSpace.SHARED]()
    
    var b_offset = UInt(p) * UInt(n)
    
    # Each thread accumulates partial sum (stride by block size)
    var sum_val = Float32(0.0)
    var idx = tid
    while idx < n:
        sum_val += a_ptr[UInt(idx)] * b_ptr[b_offset + UInt(idx)]
        idx += num_threads
    
    # Intra-warp reduction
    sum_val = warp_sum(sum_val)
    
    # Lane 0 of each warp writes to shared memory
    if lane == 0:
        warp_sums[warp_id] = sum_val
    
    barrier()
    
    # Warp 0 reduces across per-warp results
    if warp_id == 0:
        var final_val = Float32(0.0)
        if lane < NUM_WARPS:
            final_val = warp_sums[lane]
        final_val = warp_sum(final_val)
        if lane == 0:
            result_ptr[p] = final_val


fn kernel_dot_batched_vs_strided(
    a_ptr: UnsafePointer[Float32, MutAnyOrigin],
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    result_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    num_params: Int,
) -> None:
    """Compute result[p * num_cols + col] = a[:, col]^T @ b[p*n*num_cols + col*n : ...].
    
    Used for fused gradient trace term: probe_solutions[:, j]^T @ (dK/dp @ right_factors[:, j]).
    Grid: (num_params * num_cols,), Block: (256,) - 8 warps per (param, col) pair.
    
    Args:
        a_ptr: Column-major matrix [n, num_cols] (e.g., probe_solutions)
        b_ptr: Fused gradient output [num_params * n * num_cols], layout [p][col][row]
        result_ptr: Output [num_params * num_cols] dot products
        n: Vector length
        num_cols: Number of columns
        num_params: Number of parameters
    """
    var idx = block_idx.x
    var tid = Int(thread_idx.x)         # 0-255
    var num_threads = Int(block_dim.x)  # 256
    var warp_id = tid // WARP_SIZE      # 0-7
    var lane = tid % WARP_SIZE          # 0-31
    
    if idx >= UInt(num_params * num_cols):
        return
    
    # Shared memory for per-warp partial sums
    alias NUM_WARPS = 8
    var warp_sums = stack_allocation[NUM_WARPS, Float32, address_space = AddressSpace.SHARED]()
    
    var p = Int(idx) // num_cols
    var col = Int(idx) % num_cols
    
    var a_offset = UInt(col) * UInt(n)
    var b_offset = UInt(p) * UInt(n * num_cols) + UInt(col) * UInt(n)
    
    # Each thread accumulates partial sum (stride by block size)
    var sum_val = Float32(0.0)
    var i = tid
    while i < n:
        sum_val += a_ptr[a_offset + UInt(i)] * b_ptr[b_offset + UInt(i)]
        i += num_threads
    
    # Intra-warp reduction
    sum_val = warp_sum(sum_val)
    
    # Lane 0 of each warp writes to shared memory
    if lane == 0:
        warp_sums[warp_id] = sum_val
    
    barrier()
    
    # Warp 0 reduces across per-warp results
    if warp_id == 0:
        var final_val = Float32(0.0)
        if lane < NUM_WARPS:
            final_val = warp_sums[lane]
        final_val = warp_sum(final_val)
        if lane == 0:
            result_ptr[idx] = final_val


# =============================================================================
# Helper Kernels for GPU-Optimized BBMM
# =============================================================================

fn kernel_copy_column(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    dest_col: Int,
) -> None:
    """Copy a single column from src to dst.
    
    dst[:, dest_col] = src[:]
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    
    dst_ptr[UInt(dest_col) * UInt(n) + idx] = src_ptr[idx]


fn kernel_copy_columns(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    dest_col_offset: Int,
) -> None:
    """Copy columns from src to dst with offset.
    
    dst[:, dest_col_offset:dest_col_offset+num_cols] = src[:, :]
    Both are column-major.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n * num_cols):
        return
    
    var col = Int(idx) // n
    var row = Int(idx) % n
    
    var src_idx = col * n + row
    var dst_idx = (col + dest_col_offset) * n + row
    
    dst_ptr[dst_idx] = src_ptr[src_idx]


fn kernel_extract_columns_range(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    src_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    src_col_offset: Int,
) -> None:
    """Extract a range of columns from src to dst.
    
    dst[:, 0:num_cols] = src[:, src_col_offset:src_col_offset+num_cols]
    Both are column-major.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n * num_cols):
        return
    
    var col = Int(idx) // n
    var row = Int(idx) % n
    
    var src_idx = (col + src_col_offset) * n + row
    var dst_idx = col * n + row
    
    dst_ptr[dst_idx] = src_ptr[src_idx]


fn kernel_scale_columns_by_norms(
    data_ptr: UnsafePointer[Float32, MutAnyOrigin],
    norms_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Scale each column by its corresponding norm (in-place).
    
    Used to unnormalize probe vectors for gradient computation.
    data[:, col] *= norms[col]
    
    Args:
        data_ptr: Column-major matrix [n × num_cols] to scale in-place
        norms_ptr: Vector of norms [num_cols]
        n: Number of rows
        num_cols: Number of columns
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n * num_cols):
        return
    
    var col = Int(idx) // n
    var norm = norms_ptr[col]
    
    data_ptr[idx] = data_ptr[idx] * norm


fn scale_columns_by_norms(
    ctx: DeviceContext,
    matrix: DeviceBuffer[float_dtype],
    norms: DeviceBuffer[float_dtype],
    n: Int,
    num_cols: Int,
    sync: Bool = True,
) raises:
    """Scale each column by its corresponding norm in-place.
    
    Computes: matrix[:, i] *= norms[i] for each column i.
    
    Args:
        ctx: GPU device context
        matrix: [n x num_cols] matrix to scale in-place (column-major)
        norms: [num_cols] scaling factors
        n: Number of rows
        num_cols: Number of columns
        sync: Whether to synchronize after the kernel (default True)
    """
    ctx.enqueue_function[kernel_scale_columns_by_norms](
        matrix.unsafe_ptr(), norms.unsafe_ptr(), n, num_cols,
        grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
    )
    if sync:
        ctx.synchronize()


# =============================================================================
# Batched Dot Product Matrix Kernel (for Multi-Output B Gradients)
# =============================================================================

fn kernel_dot_matrix(
    a_ptr: UnsafePointer[Float32, MutAnyOrigin],
    b_ptr: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    T1: Int,
    T2: Int,
) -> None:
    """Compute out[s, t] = a[:, s]^T @ b[:, t] for all (s, t) pairs in one launch.

    Replaces T1*T2 sequential kernel_dot_batched calls + D2H syncs with a
    single kernel launch. Each block computes one dot product using 256-thread
    reduction (8 warps).

    Memory layout: a and b are column-major [n × T1] and [n × T2].
    out is row-major [T1 × T2]: out[s * T2 + t].

    Launch: grid_dim=(T1 * T2,), block_dim=(256,)

    Args:
        a_ptr: Column-major matrix [n × T1] (e.g., alpha solutions per task)
        b_ptr: Column-major matrix [n × T2] (e.g., K_X @ alpha per task)
        out_ptr: Output matrix [T1 × T2] dot products (row-major)
        n: Vector length (number of data points)
        T1: Number of columns in A (source 1)
        T2: Number of columns in B (source 2)
    """
    var pair_idx = block_idx.x
    var tid = Int(thread_idx.x)         # 0-255
    var num_threads = Int(block_dim.x)  # 256
    var warp_id = tid // WARP_SIZE      # 0-7
    var lane = tid % WARP_SIZE          # 0-31

    if pair_idx >= UInt(T1 * T2):
        return

    alias NUM_WARPS = 8
    var warp_sums = stack_allocation[NUM_WARPS, Float32, address_space = AddressSpace.SHARED]()

    var s = Int(pair_idx) // T2
    var t = Int(pair_idx) % T2

    var a_offset = UInt(s) * UInt(n)
    var b_offset = UInt(t) * UInt(n)

    # Each thread accumulates partial sum (stride by block size)
    var sum_val = Float32(0.0)
    var i = tid
    while i < n:
        sum_val += a_ptr[a_offset + UInt(i)] * b_ptr[b_offset + UInt(i)]
        i += num_threads

    # Intra-warp reduction
    sum_val = warp_sum(sum_val)

    # Lane 0 of each warp writes to shared memory
    if lane == 0:
        warp_sums[warp_id] = sum_val

    barrier()

    # Warp 0 reduces across per-warp results
    if warp_id == 0:
        var final_val = Float32(0.0)
        if lane < NUM_WARPS:
            final_val = warp_sums[lane]
        final_val = warp_sum(final_val)
        if lane == 0:
            out_ptr[pair_idx] = final_val
