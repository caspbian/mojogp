"""GPU kernel templates for composite kernels.

These templates are parameterized by ComposableKernel types, enabling
compile-time composition with zero runtime overhead.

The key difference from generic_matvec.mojo:
- Uses ComposableKernel trait instead of function pointers
- Parameters are in a flat UnsafePointer[Float32] instead of KernelParams struct
- All composition is resolved at compile time

Example usage:
    alias MyKernel = SumKernel[RBFComposable, LinearComposable]
    
    ctx.enqueue_function[composite_forward_matvec_8x[5, MyKernel]](
        out_ptr, x_ptr, v_ptr, params_ptr, n, num_cols, noise,
        grid_dim=num_blocks, block_dim=threads_per_block
    )
"""

from collections import InlineArray
from gpu.id import block_dim, block_idx, thread_idx
from gpu.sync import barrier
from gpu.memory import external_memory
from memory import UnsafePointer, AddressSpace

from .composable_kernel import ComposableKernel


# =============================================================================
# Forward Matvec: out = (K + noise*I) @ v
# =============================================================================

fn composite_forward_matvec_8x[
    DIM: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    noise: Float32,
) -> None:
    """Forward matvec for composite kernels: out = (K + noise*I) @ v.
    
    8x unrolled for instruction-level parallelism.
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        params_ptr: Flat parameter array for the composite kernel
        n: Number of data points
        num_cols: Number of RHS columns
        noise: Noise variance σ²
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    for col in range(num_cols):
        var col_offset = UInt(col) * UInt(n)
        var sum0 = Float32(0.0)
        var sum1 = Float32(0.0)
        var sum2 = Float32(0.0)
        var sum3 = Float32(0.0)
        var sum4 = Float32(0.0)
        var sum5 = Float32(0.0)
        var sum6 = Float32(0.0)
        var sum7 = Float32(0.0)
        
        # 8x unrolled main loop
        var j = 0
        while j + 7 < n:
            var v0 = v_ptr[col_offset + UInt(j)]
            var v1 = v_ptr[col_offset + UInt(j + 1)]
            var v2 = v_ptr[col_offset + UInt(j + 2)]
            var v3 = v_ptr[col_offset + UInt(j + 3)]
            var v4 = v_ptr[col_offset + UInt(j + 4)]
            var v5 = v_ptr[col_offset + UInt(j + 5)]
            var v6 = v_ptr[col_offset + UInt(j + 6)]
            var v7 = v_ptr[col_offset + UInt(j + 7)]
            
            var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j2 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j3 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j4 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j5 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j6 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j7 = InlineArray[Float32, DIM](uninitialized=True)
            
            @parameter
            for d in range(DIM):
                x_j0[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                x_j1[d] = x_ptr[UInt(j + 1) * UInt(DIM) + UInt(d)]
                x_j2[d] = x_ptr[UInt(j + 2) * UInt(DIM) + UInt(d)]
                x_j3[d] = x_ptr[UInt(j + 3) * UInt(DIM) + UInt(d)]
                x_j4[d] = x_ptr[UInt(j + 4) * UInt(DIM) + UInt(d)]
                x_j5[d] = x_ptr[UInt(j + 5) * UInt(DIM) + UInt(d)]
                x_j6[d] = x_ptr[UInt(j + 6) * UInt(DIM) + UInt(d)]
                x_j7[d] = x_ptr[UInt(j + 7) * UInt(DIM) + UInt(d)]
            
            sum0 += K.evaluate[DIM](x_row, x_j0, params_ptr) * v0
            sum1 += K.evaluate[DIM](x_row, x_j1, params_ptr) * v1
            sum2 += K.evaluate[DIM](x_row, x_j2, params_ptr) * v2
            sum3 += K.evaluate[DIM](x_row, x_j3, params_ptr) * v3
            sum4 += K.evaluate[DIM](x_row, x_j4, params_ptr) * v4
            sum5 += K.evaluate[DIM](x_row, x_j5, params_ptr) * v5
            sum6 += K.evaluate[DIM](x_row, x_j6, params_ptr) * v6
            sum7 += K.evaluate[DIM](x_row, x_j7, params_ptr) * v7
            
            j += 8
        
        # Handle remainder
        while j < n:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            
            sum0 += K.evaluate[DIM](x_row, x_j, params_ptr) * v_ptr[col_offset + UInt(j)]
            j += 1
        
        var total = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
        
        # Add noise to diagonal
        total += noise * v_ptr[col_offset + row]
        
        out_ptr[col_offset + row] = total


# =============================================================================
# Multi-Column Forward Matvec (4x Unrolled, Fused Columns)
# =============================================================================

fn composite_forward_matvec_multicol[
    DIM: Int,
    NCOLS: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    noise: Float32,
) -> None:
    """Multi-column fused forward matvec for composite kernels.
    
    Uses shared memory tiling: cooperatively loads x_j + v_j into shmem,
    then each thread computes K.evaluate(x_i, x_j) from shmem to reduce
    global-memory traffic.
    """
    alias DIMY = DIM + NCOLS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var i = Int(block_idx.x) * bs + tid

    var yj = external_memory[
        Float32,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    var valid = i < n

    # Cache x[i] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    if valid:
        var row_offset = UInt(i) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row[d] = x_ptr[row_offset + UInt(d)]

    # Per-column accumulators
    var acc = InlineArray[Float32, NCOLS](fill=Float32(0.0))

    # Tiled j-loop: cooperative load into shared memory
    var jstart = 0
    while jstart < n:
        var j = jstart + tid
        # Cooperative load: each thread loads ONE j-point (x_j + v_j)
        if j < n:
            var shared_base = tid * DIMY
            @parameter
            for d in range(DIM):
                yj[shared_base + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            @parameter
            for c in range(NCOLS):
                yj[shared_base + DIM + c] = v_ptr[UInt(c) * UInt(n) + UInt(j)]
        barrier()

        # Each valid thread computes over the tile
        if valid:
            var tile_end = bs
            if jstart + bs > n:
                tile_end = n - jstart
            for jrel in range(tile_end):
                var shared_base = jrel * DIMY
                var x_j = InlineArray[Float32, DIM](uninitialized=True)
                @parameter
                for d in range(DIM):
                    x_j[d] = yj[shared_base + d]

                var kval = K.evaluate[DIM](x_row, x_j, params_ptr)

                @parameter
                for c in range(NCOLS):
                    acc[c] += kval * yj[shared_base + DIM + c]
        barrier()
        jstart += bs

    # Write output + noise diagonal
    if valid:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            out_ptr[col_off + UInt(i)] = acc[c] + noise * v_ptr[col_off + UInt(i)]


# =============================================================================
# Gradient Matvec: out[p] = (dK/dtheta_p) @ v for each parameter p
# =============================================================================

fn composite_gradient_matvec_4x[
    DIM: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_params: Int,
) -> None:
    """Gradient matvec for composite kernels.
    
    Computes out[p] = (dK/dtheta_p) @ v for each parameter p.
    
    4x unrolled for balance between ILP and register pressure.
    
    Args:
        out_ptr: Output buffer [n, num_params] column-major
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vector [n]
        params_ptr: Flat parameter array
        n: Number of data points
        num_params: Number of kernel parameters (K.num_params())
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    # Accumulate gradients for each parameter
    # We use a simple approach: one pass per parameter
    # This could be optimized by computing all gradients in one pass
    for p in range(num_params):
        var col_offset = UInt(p) * UInt(n)
        var sum0 = Float32(0.0)
        var sum1 = Float32(0.0)
        var sum2 = Float32(0.0)
        var sum3 = Float32(0.0)
        
        # 4x unrolled main loop
        var j = 0
        while j + 3 < n:
            var v0 = v_ptr[UInt(j)]
            var v1 = v_ptr[UInt(j + 1)]
            var v2 = v_ptr[UInt(j + 2)]
            var v3 = v_ptr[UInt(j + 3)]
            
            var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j2 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j3 = InlineArray[Float32, DIM](uninitialized=True)
            
            @parameter
            for d in range(DIM):
                x_j0[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                x_j1[d] = x_ptr[UInt(j + 1) * UInt(DIM) + UInt(d)]
                x_j2[d] = x_ptr[UInt(j + 2) * UInt(DIM) + UInt(d)]
                x_j3[d] = x_ptr[UInt(j + 3) * UInt(DIM) + UInt(d)]
            
            sum0 += K.gradient[DIM](x_row, x_j0, params_ptr, p) * v0
            sum1 += K.gradient[DIM](x_row, x_j1, params_ptr, p) * v1
            sum2 += K.gradient[DIM](x_row, x_j2, params_ptr, p) * v2
            sum3 += K.gradient[DIM](x_row, x_j3, params_ptr, p) * v3
            
            j += 4
        
        # Handle remainder
        while j < n:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            
            sum0 += K.gradient[DIM](x_row, x_j, params_ptr, p) * v_ptr[UInt(j)]
            j += 1
        
        out_ptr[col_offset + row] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Batched Gradient Matvec: out[p, c] = (dK/dtheta_p) @ v[:, c]
# =============================================================================

fn composite_gradient_matvec_batched_4x[
    DIM: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    num_params: Int,
) -> None:
    """Batched gradient matvec for composite kernels.
    
    Computes out[p, c] = (dK/dtheta_p) @ v[:, c] for each parameter p and column c.
    
    Output layout: [n, num_params * num_cols] where the first num_cols columns
    are for parameter 0, next num_cols for parameter 1, etc.
    
    Args:
        out_ptr: Output buffer [n, num_params * num_cols] column-major
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        params_ptr: Flat parameter array
        n: Number of data points
        num_cols: Number of RHS columns
        num_params: Number of kernel parameters
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    # For each parameter
    for p in range(num_params):
        # For each column
        for col in range(num_cols):
            var v_col_offset = UInt(col) * UInt(n)
            var out_col_offset = UInt(p * num_cols + col) * UInt(n)
            
            var sum0 = Float32(0.0)
            var sum1 = Float32(0.0)
            var sum2 = Float32(0.0)
            var sum3 = Float32(0.0)
            
            # 4x unrolled main loop
            var j = 0
            while j + 3 < n:
                var v0 = v_ptr[v_col_offset + UInt(j)]
                var v1 = v_ptr[v_col_offset + UInt(j + 1)]
                var v2 = v_ptr[v_col_offset + UInt(j + 2)]
                var v3 = v_ptr[v_col_offset + UInt(j + 3)]
                
                var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
                var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
                var x_j2 = InlineArray[Float32, DIM](uninitialized=True)
                var x_j3 = InlineArray[Float32, DIM](uninitialized=True)
                
                @parameter
                for d in range(DIM):
                    x_j0[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                    x_j1[d] = x_ptr[UInt(j + 1) * UInt(DIM) + UInt(d)]
                    x_j2[d] = x_ptr[UInt(j + 2) * UInt(DIM) + UInt(d)]
                    x_j3[d] = x_ptr[UInt(j + 3) * UInt(DIM) + UInt(d)]
                
                sum0 += K.gradient[DIM](x_row, x_j0, params_ptr, p) * v0
                sum1 += K.gradient[DIM](x_row, x_j1, params_ptr, p) * v1
                sum2 += K.gradient[DIM](x_row, x_j2, params_ptr, p) * v2
                sum3 += K.gradient[DIM](x_row, x_j3, params_ptr, p) * v3
                
                j += 4
            
            # Handle remainder
            while j < n:
                var x_j = InlineArray[Float32, DIM](uninitialized=True)
                @parameter
                for d in range(DIM):
                    x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                
                sum0 += K.gradient[DIM](x_row, x_j, params_ptr, p) * v_ptr[v_col_offset + UInt(j)]
                j += 1
            
            out_ptr[out_col_offset + row] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Single-Parameter Gradient Matvec: out = (dK/dtheta_p) @ v for ONE parameter
# =============================================================================

fn composite_gradient_matvec_single_param_4x[
    DIM: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    param_index: Int,
) -> None:
    """Single-parameter gradient matvec for composite kernels.
    
    Computes out = (dK/dtheta_p) @ v for a SINGLE parameter p.
    This is used by BBMM which computes gradients one parameter at a time.
    
    4x unrolled for balance between ILP and register pressure.
    
    Args:
        out_ptr: Output buffer [n]
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vector [n]
        params_ptr: Flat parameter array
        n: Number of data points
        param_index: Which parameter to differentiate (0 to num_params-1)
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    var sum0 = Float32(0.0)
    var sum1 = Float32(0.0)
    var sum2 = Float32(0.0)
    var sum3 = Float32(0.0)
    
    # 4x unrolled main loop
    var j = 0
    while j + 3 < n:
        var v0 = v_ptr[UInt(j)]
        var v1 = v_ptr[UInt(j + 1)]
        var v2 = v_ptr[UInt(j + 2)]
        var v3 = v_ptr[UInt(j + 3)]
        
        var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
        var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
        var x_j2 = InlineArray[Float32, DIM](uninitialized=True)
        var x_j3 = InlineArray[Float32, DIM](uninitialized=True)
        
        @parameter
        for d in range(DIM):
            x_j0[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            x_j1[d] = x_ptr[UInt(j + 1) * UInt(DIM) + UInt(d)]
            x_j2[d] = x_ptr[UInt(j + 2) * UInt(DIM) + UInt(d)]
            x_j3[d] = x_ptr[UInt(j + 3) * UInt(DIM) + UInt(d)]
        
        sum0 += K.gradient[DIM](x_row, x_j0, params_ptr, param_index) * v0
        sum1 += K.gradient[DIM](x_row, x_j1, params_ptr, param_index) * v1
        sum2 += K.gradient[DIM](x_row, x_j2, params_ptr, param_index) * v2
        sum3 += K.gradient[DIM](x_row, x_j3, params_ptr, param_index) * v3
        
        j += 4
    
    # Handle remainder
    while j < n:
        var x_j = InlineArray[Float32, DIM](uninitialized=True)
        @parameter
        for d in range(DIM):
            x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
        
        sum0 += K.gradient[DIM](x_row, x_j, params_ptr, param_index) * v_ptr[UInt(j)]
        j += 1
    
    out_ptr[row] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Single-Parameter Batched Gradient Matvec: out[:, c] = (dK/dtheta_p) @ v[:, c]
# =============================================================================

fn composite_gradient_matvec_single_param_batched_4x[
    DIM: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    param_index: Int,
) -> None:
    """Single-parameter batched gradient matvec for composite kernels.
    
    Computes out[:, c] = (dK/dtheta_p) @ v[:, c] for a SINGLE parameter p
    across multiple RHS columns.
    
    This is used by BBMM for gradient trace estimation with probe vectors.
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        params_ptr: Flat parameter array
        n: Number of data points
        num_cols: Number of RHS columns
        param_index: Which parameter to differentiate (0 to num_params-1)
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    # For each column
    for col in range(num_cols):
        var v_col_offset = UInt(col) * UInt(n)
        var out_col_offset = UInt(col) * UInt(n)
        
        var sum0 = Float32(0.0)
        var sum1 = Float32(0.0)
        var sum2 = Float32(0.0)
        var sum3 = Float32(0.0)
        
        # 4x unrolled main loop
        var j = 0
        while j + 3 < n:
            var v0 = v_ptr[v_col_offset + UInt(j)]
            var v1 = v_ptr[v_col_offset + UInt(j + 1)]
            var v2 = v_ptr[v_col_offset + UInt(j + 2)]
            var v3 = v_ptr[v_col_offset + UInt(j + 3)]
            
            var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j2 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j3 = InlineArray[Float32, DIM](uninitialized=True)
            
            @parameter
            for d in range(DIM):
                x_j0[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                x_j1[d] = x_ptr[UInt(j + 1) * UInt(DIM) + UInt(d)]
                x_j2[d] = x_ptr[UInt(j + 2) * UInt(DIM) + UInt(d)]
                x_j3[d] = x_ptr[UInt(j + 3) * UInt(DIM) + UInt(d)]
            
            sum0 += K.gradient[DIM](x_row, x_j0, params_ptr, param_index) * v0
            sum1 += K.gradient[DIM](x_row, x_j1, params_ptr, param_index) * v1
            sum2 += K.gradient[DIM](x_row, x_j2, params_ptr, param_index) * v2
            sum3 += K.gradient[DIM](x_row, x_j3, params_ptr, param_index) * v3
            
            j += 4
        
        # Handle remainder
        while j < n:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            
            sum0 += K.gradient[DIM](x_row, x_j, params_ptr, param_index) * v_ptr[v_col_offset + UInt(j)]
            j += 1
        
        out_ptr[out_col_offset + row] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Cross-Covariance Matvec: out = K(X_test, X_train) @ v
# =============================================================================

fn composite_cross_matvec_8x[
    DIM: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_test: Int,
    n_train: Int,
    num_cols: Int,
) -> None:
    """Cross-covariance matvec for composite kernels.
    
    Computes out = K(X_test, X_train) @ v for prediction.
    
    Args:
        out_ptr: Output buffer [n_test, num_cols] column-major
        x_test_ptr: Test data [n_test, DIM] row-major
        x_train_ptr: Training data [n_train, DIM] row-major
        v_ptr: Input vectors [n_train, num_cols] column-major
        params_ptr: Flat parameter array
        n_test: Number of test points
        n_train: Number of training points
        num_cols: Number of RHS columns
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n_test):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x_test[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_test_ptr[row_offset + UInt(d)]
    
    for col in range(num_cols):
        var col_offset = UInt(col) * UInt(n_train)
        var out_col_offset = UInt(col) * UInt(n_test)
        
        var sum0 = Float32(0.0)
        var sum1 = Float32(0.0)
        var sum2 = Float32(0.0)
        var sum3 = Float32(0.0)
        var sum4 = Float32(0.0)
        var sum5 = Float32(0.0)
        var sum6 = Float32(0.0)
        var sum7 = Float32(0.0)
        
        # 8x unrolled main loop
        var j = 0
        while j + 7 < n_train:
            var v0 = v_ptr[col_offset + UInt(j)]
            var v1 = v_ptr[col_offset + UInt(j + 1)]
            var v2 = v_ptr[col_offset + UInt(j + 2)]
            var v3 = v_ptr[col_offset + UInt(j + 3)]
            var v4 = v_ptr[col_offset + UInt(j + 4)]
            var v5 = v_ptr[col_offset + UInt(j + 5)]
            var v6 = v_ptr[col_offset + UInt(j + 6)]
            var v7 = v_ptr[col_offset + UInt(j + 7)]
            
            var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j2 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j3 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j4 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j5 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j6 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j7 = InlineArray[Float32, DIM](uninitialized=True)
            
            @parameter
            for d in range(DIM):
                x_j0[d] = x_train_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                x_j1[d] = x_train_ptr[UInt(j + 1) * UInt(DIM) + UInt(d)]
                x_j2[d] = x_train_ptr[UInt(j + 2) * UInt(DIM) + UInt(d)]
                x_j3[d] = x_train_ptr[UInt(j + 3) * UInt(DIM) + UInt(d)]
                x_j4[d] = x_train_ptr[UInt(j + 4) * UInt(DIM) + UInt(d)]
                x_j5[d] = x_train_ptr[UInt(j + 5) * UInt(DIM) + UInt(d)]
                x_j6[d] = x_train_ptr[UInt(j + 6) * UInt(DIM) + UInt(d)]
                x_j7[d] = x_train_ptr[UInt(j + 7) * UInt(DIM) + UInt(d)]
            
            sum0 += K.evaluate[DIM](x_row, x_j0, params_ptr) * v0
            sum1 += K.evaluate[DIM](x_row, x_j1, params_ptr) * v1
            sum2 += K.evaluate[DIM](x_row, x_j2, params_ptr) * v2
            sum3 += K.evaluate[DIM](x_row, x_j3, params_ptr) * v3
            sum4 += K.evaluate[DIM](x_row, x_j4, params_ptr) * v4
            sum5 += K.evaluate[DIM](x_row, x_j5, params_ptr) * v5
            sum6 += K.evaluate[DIM](x_row, x_j6, params_ptr) * v6
            sum7 += K.evaluate[DIM](x_row, x_j7, params_ptr) * v7
            
            j += 8
        
        # Handle remainder
        while j < n_train:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_train_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            
            sum0 += K.evaluate[DIM](x_row, x_j, params_ptr) * v_ptr[col_offset + UInt(j)]
            j += 1
        
        out_ptr[out_col_offset + row] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7


# =============================================================================
# Diagonal Extraction: diag[i] = K(x_i, x_i)
# =============================================================================

fn composite_extract_diagonal[
    DIM: Int,
    K: ComposableKernel,
](
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """Extract diagonal of kernel matrix for composite kernels.
    
    Computes diag[i] = K(x_i, x_i) for variance computation.
    
    Args:
        diag_ptr: Output buffer [n]
        x_ptr: Data points [n, DIM] row-major
        params_ptr: Flat parameter array
        n: Number of data points
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Load x[row]
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    # K(x_i, x_i) - diagonal element
    diag_ptr[row] = K.evaluate[DIM](x_row, x_row, params_ptr)


# =============================================================================
# Fused Gradient Matvec: compute ALL (dK/dtheta_p @ v) in one pass
# =============================================================================

fn _call_all_gradients[DIM: Int, NUM_PARAMS: Int, K: ComposableKernel](
    x_i: InlineArray[Float32, DIM],
    x_j: InlineArray[Float32, DIM],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    grads_out: UnsafePointer[Float32, MutAnyOrigin],
) -> None:
    """Helper to call K.all_gradients with MutAnyOrigin pointers.
    
    This wrapper exists because GPU kernel functions cannot have unbound origin
    parameters. InlineArray.unsafe_ptr() returns a pointer with a local stack
    origin, which would introduce an unbound origin parameter into the GPU kernel.
    By routing through this helper (which accepts MutAnyOrigin), the compiler
    performs implicit origin conversion at the call site.
    """
    K.all_gradients[DIM](x_i, x_j, params_ptr, grads_out)


fn composite_fused_gradient_matvec_4x[DIM: Int, K: ComposableKernel](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
) -> None:
    """Fused gradient matvec for composite kernels using K.all_gradients().
    
    Computes all K.num_params() gradient matvecs in a single GPU kernel launch,
    calling K.all_gradients() once per pair to get all gradient values simultaneously.
    
    Output layout:
        out_ptr[p * n * num_cols + col * n + row] for parameter p
        Contiguous per-parameter blocks, each [n, num_cols] column-major.
    
    Uses 4x unrolling for ILP. Register usage per thread:
        - 4 * NUM_PARAMS gradient accumulators
        - For 20-param kernel: 4*20 = 80 registers (well within 255 limit)
    
    Args:
        out_ptr: Output for all gradient matvecs [K.num_params() * n * num_cols]
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        params_ptr: Flat parameter array
        n: Number of data points
        num_cols: Number of input columns
    """
    alias NUM_PARAMS = K.num_params()
    
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Load x[row] into registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    for col in range(num_cols):
        var col_offset = UInt(col) * UInt(n)
        
        # 4 accumulators per gradient param for ILP
        var grad_sums = InlineArray[Float32, NUM_PARAMS * 4](uninitialized=True)
        @parameter
        for p in range(NUM_PARAMS * 4):
            grad_sums[p] = Float32(0.0)
        
        # 4x unrolled main loop
        var j = 0
        while j + 3 < n:
            var v0 = v_ptr[col_offset + UInt(j)]
            var v1 = v_ptr[col_offset + UInt(j + 1)]
            var v2 = v_ptr[col_offset + UInt(j + 2)]
            var v3 = v_ptr[col_offset + UInt(j + 3)]
            
            var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j2 = InlineArray[Float32, DIM](uninitialized=True)
            var x_j3 = InlineArray[Float32, DIM](uninitialized=True)
            
            @parameter
            for d in range(DIM):
                var base = UInt(j) * UInt(DIM) + UInt(d)
                x_j0[d] = x_ptr[base]
                x_j1[d] = x_ptr[base + UInt(DIM)]
                x_j2[d] = x_ptr[base + UInt(2 * DIM)]
                x_j3[d] = x_ptr[base + UInt(3 * DIM)]
            
            # Compute all gradients for each pair
            var dk0 = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
            _call_all_gradients[DIM, NUM_PARAMS, K](x_row, x_j0, params_ptr, dk0.unsafe_ptr())
            @parameter
            for p in range(NUM_PARAMS):
                grad_sums[p * 4] += dk0[p] * v0
            
            var dk1 = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
            _call_all_gradients[DIM, NUM_PARAMS, K](x_row, x_j1, params_ptr, dk1.unsafe_ptr())
            @parameter
            for p in range(NUM_PARAMS):
                grad_sums[p * 4 + 1] += dk1[p] * v1
            
            var dk2 = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
            _call_all_gradients[DIM, NUM_PARAMS, K](x_row, x_j2, params_ptr, dk2.unsafe_ptr())
            @parameter
            for p in range(NUM_PARAMS):
                grad_sums[p * 4 + 2] += dk2[p] * v2
            
            var dk3 = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
            _call_all_gradients[DIM, NUM_PARAMS, K](x_row, x_j3, params_ptr, dk3.unsafe_ptr())
            @parameter
            for p in range(NUM_PARAMS):
                grad_sums[p * 4 + 3] += dk3[p] * v3
            
            j += 4
        
        # Remainder loop
        var grad_sums_rem = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
        @parameter
        for p in range(NUM_PARAMS):
            grad_sums_rem[p] = Float32(0.0)
        
        while j < n:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            
            var dk = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
            _call_all_gradients[DIM, NUM_PARAMS, K](x_row, x_j, params_ptr, dk.unsafe_ptr())
            
            var v_j = v_ptr[col_offset + UInt(j)]
            @parameter
            for p in range(NUM_PARAMS):
                grad_sums_rem[p] += dk[p] * v_j
            j += 1
        
        # Write results for all gradient params
        @parameter
        for p in range(NUM_PARAMS):
            var grad_total = grad_sums[p * 4] + grad_sums[p * 4 + 1] + grad_sums[p * 4 + 2] + grad_sums[p * 4 + 3] + grad_sums_rem[p]
            out_ptr[UInt(p) * UInt(n * num_cols) + col_offset + UInt(row)] = grad_total


# =============================================================================
# Fused Gradient Matvec with Shared Memory (Multi-Column)
# =============================================================================

fn composite_fused_gradient_matvec_shmem_multicol[
    DIM: Int,
    NCOLS: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """Fused gradient matvec with shared memory tiling.
    
    Computes all K.num_params() gradient matvecs in a single kernel launch.
    Uses shmem tiling for x_j + v_j, calling K.all_gradients() once per
    (i,j) pair and scattering all gradient values across NCOLS columns.
    
    Output layout:
        out_ptr[p * n * NCOLS + col * n + row] for parameter p.
    """
    alias NUM_PARAMS = K.num_params()
    alias DIMY = DIM + NCOLS
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var i = Int(block_idx.x) * bs + tid

    var yj = external_memory[
        Float32,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    var valid = i < n

    # Cache x[i] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    if valid:
        var row_offset = UInt(i) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row[d] = x_ptr[row_offset + UInt(d)]

    # Accumulators: NUM_PARAMS gradient outputs × NCOLS columns
    # Layout: grad_acc[p * NCOLS + c]
    var grad_acc = InlineArray[Float32, NUM_PARAMS * NCOLS](fill=Float32(0.0))

    # Tiled j-loop
    var jstart = 0
    while jstart < n:
        var j = jstart + tid
        if j < n:
            var shared_base = tid * DIMY
            @parameter
            for d in range(DIM):
                yj[shared_base + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            @parameter
            for c in range(NCOLS):
                yj[shared_base + DIM + c] = v_ptr[UInt(c) * UInt(n) + UInt(j)]
        barrier()

        if valid:
            var tile_end = bs
            if jstart + bs > n:
                tile_end = n - jstart
            for jrel in range(tile_end):
                var shared_base = jrel * DIMY
                var x_j = InlineArray[Float32, DIM](uninitialized=True)
                @parameter
                for d in range(DIM):
                    x_j[d] = yj[shared_base + d]

                # Compute all gradients ONCE per (i,j) pair
                var dk = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                _call_all_gradients[DIM, NUM_PARAMS, K](
                    x_row, x_j, params_ptr, dk.unsafe_ptr()
                )

                # Scatter each gradient against all NCOLS v_j columns
                @parameter
                for p in range(NUM_PARAMS):
                    var dk_p = dk[p]
                    @parameter
                    for c in range(NCOLS):
                        grad_acc[p * NCOLS + c] += dk_p * yj[shared_base + DIM + c]
        barrier()
        jstart += bs

    # Write output
    if valid:
        @parameter
        for p in range(NUM_PARAMS):
            @parameter
            for c in range(NCOLS):
                out_ptr[UInt(p) * UInt(n * NCOLS) + UInt(c) * UInt(n) + UInt(i)] = grad_acc[p * NCOLS + c]


# =============================================================================
# Cross-Covariance Matrix Materialization
# =============================================================================

fn composite_cross_covariance_gpu[
    DIM: Int,
    K: ComposableKernel,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
) -> None:
    """Materialize cross-covariance matrix K_cross[i, j] = K(x_train_i, x_test_j).

    Output is column-major [n_train, n_test]:
        out_ptr[j * n_train + i] = K(x_train_i, x_test_j)

    Each GPU thread computes one row i of K_cross (all n_test entries).

    Args:
        out_ptr: Output buffer [n_train * n_test] on device, column-major
        x_train_ptr: Training data [n_train, DIM] row-major
        x_test_ptr: Test data [n_test, DIM] row-major
        params_ptr: Flat kernel parameter array
        n_train: Number of training points
        n_test: Number of test points
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n_train):
        return

    var row_offset = UInt(row) * UInt(DIM)

    # Cache x_train[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_train_ptr[row_offset + UInt(d)]

    # Compute K(x_train_row, x_test_j) for each test point j
    for j in range(n_test):
        var j_offset = UInt(j) * UInt(DIM)
        var x_j = InlineArray[Float32, DIM](uninitialized=True)
        @parameter
        for d in range(DIM):
            x_j[d] = x_test_ptr[j_offset + UInt(d)]

        var kval = K.evaluate[DIM](x_row, x_j, params_ptr)
        # Column-major: element (row, j) at j * n_train + row
        out_ptr[UInt(j) * UInt(n_train) + UInt(row)] = kval
