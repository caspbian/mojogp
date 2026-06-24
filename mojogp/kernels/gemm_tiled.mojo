"""Tiled GEMM with shared memory for batched matrix-vector operations.

This module implements a shared memory tiled GEMM that exploits BBMM batching.
The key optimization: load K tiles into shared memory once and reuse across all columns.

Expected speedup: 3-5x for batched operations (num_cols >= 4)
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu.id import block_dim, block_idx, thread_idx
from gpu.sync import barrier
from gpu.memory import AddressSpace
from memory import UnsafePointer, stack_allocation


# =============================================================================
# Tiled GEMM Kernel
# =============================================================================

fn kernel_gemm_tiled[TILE_M: Int, TILE_K: Int, MAX_COLS: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],
    V_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Tiled GEMM: Out = K @ V
    
    K: [n × n] row-major
    V: [n × num_cols] column-major
    Out: [n × num_cols] column-major
    
    Each block processes TILE_M rows of output.
    K tiles are loaded into shared memory and reused across all columns.
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major
        K_ptr: Kernel matrix [n, n] row-major
        V_ptr: Input vectors [n, num_cols] column-major
        n: Matrix dimension
        num_cols: Number of RHS columns (must be <= MAX_COLS)
    """
    # Block handles rows [block_row, block_row + TILE_M)
    var block_row = Int(block_idx.x) * TILE_M
    
    # Thread position within block
    var tx = Int(thread_idx.x)  # 0..TILE_M-1 (row within tile)
    var ty = Int(thread_idx.y)  # 0..TILE_K-1 (for loading)
    
    # Global row this thread is responsible for
    var row = block_row + tx
    
    # Early exit if row is out of bounds
    if row >= n:
        return
    
    # Shared memory for tiles
    var K_shared = stack_allocation[
        TILE_M * TILE_K, Float32,
        address_space = AddressSpace.SHARED
    ]()
    var V_shared = stack_allocation[
        TILE_K * MAX_COLS, Float32,
        address_space = AddressSpace.SHARED
    ]()
    
    # Accumulators in registers (one per output column)
    var acc = stack_allocation[MAX_COLS, Float32]()
    @parameter
    for c in range(MAX_COLS):
        acc[c] = Float32(0.0)
    
    # Number of tiles in K dimension
    var num_tiles = (n + TILE_K - 1) // TILE_K
    
    # Iterate over K in TILE_K chunks
    for tile_idx in range(num_tiles):
        var k_start = tile_idx * TILE_K
        
        # Load tiles into shared memory.
        
        # Collaboratively load K tile: K[block_row:block_row+TILE_M, k_start:k_start+TILE_K]
        # Thread (tx, ty) loads K[block_row + tx, k_start + ty]
        var k_row = block_row + tx
        var k_col = k_start + ty
        
        if k_row < n and k_col < n:
            K_shared[tx * TILE_K + ty] = K_ptr[UInt(k_row) * UInt(n) + UInt(k_col)]
        else:
            K_shared[tx * TILE_K + ty] = Float32(0.0)
        
        # Collaboratively load V tile: V[k_start:k_start+TILE_K, :]
        # Only threads with tx < TILE_K participate
        if tx < TILE_K:
            var v_row = k_start + tx
            if v_row < n:
                for c in range(num_cols):
                    V_shared[tx * MAX_COLS + c] = V_ptr[UInt(c) * UInt(n) + UInt(v_row)]
            else:
                for c in range(num_cols):
                    V_shared[tx * MAX_COLS + c] = Float32(0.0)
        
        barrier()
        
        # Accumulate tile products.
        
        # Each thread computes partial sums for its row across all columns
        @parameter
        for k in range(TILE_K):
            var k_val = K_shared[tx * TILE_K + k]
            for c in range(num_cols):
                acc[c] += k_val * V_shared[k * MAX_COLS + c]
        
        barrier()
    
    # Store accumulated outputs.
    
    if row < n:
        for c in range(num_cols):
            out_ptr[UInt(c) * UInt(n) + UInt(row)] = acc[c]


# =============================================================================
# Host Function
# =============================================================================

fn gemm_tiled(
    ctx: DeviceContext,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],
    V_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) raises:
    """Launch tiled GEMM kernel.
    
    Args:
        ctx: Device context
        out_ptr: Output buffer [n, num_cols] column-major
        K_ptr: Kernel matrix [n, n] row-major
        V_ptr: Input vectors [n, num_cols] column-major
        n: Matrix dimension
        num_cols: Number of RHS columns
    """
    alias TILE_M = 32
    alias TILE_K = 32
    alias MAX_COLS = 16  # Support up to 16 columns
    
    if num_cols > MAX_COLS:
        raise Error("num_cols exceeds MAX_COLS (16)")
    
    var num_blocks = (n + TILE_M - 1) // TILE_M
    
    ctx.enqueue_function[kernel_gemm_tiled[TILE_M, TILE_K, MAX_COLS]](
        out_ptr, K_ptr, V_ptr, n, num_cols,
        grid_dim=(num_blocks,),
        block_dim=(TILE_M, TILE_K)
    )
