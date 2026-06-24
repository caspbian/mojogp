"""Generic GPU kernel templates for matrix-vector operations.

These templates are parameterized by kernel functions, enabling code reuse
across all kernel types with zero runtime overhead.

The key insight is that Mojo's parametric functions are resolved at compile time,
so passing a kernel function as a parameter has zero runtime cost - the function
call is inlined directly into the generated GPU code.
"""

from gpu.id import block_dim, block_idx, thread_idx
from gpu.sync import barrier
from gpu.memory import external_memory, AddressSpace
from memory import UnsafePointer
from .kernel_params import KernelParams


# =============================================================================
# Generic Forward Matvec Template (8x Unrolled)
# =============================================================================

fn kernel_forward_matvec_8x[
    DIM: Int,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    params: KernelParams,
    noise: Float32,
) -> None:
    """Generic forward matvec: out = (K + noise*I) @ v
    
    This single template replaces 12+ separate kernel implementations.
    
    The template handles all the boilerplate:
    - Thread/block indexing
    - Row caching in registers
    - 8x loop unrolling for ILP
    - Remainder handling
    - Noise diagonal addition
    
    The only thing that varies is kernel_fn, which is inlined at compile time.
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Number of data points
        num_cols: Number of RHS columns (batch size)
        params: Kernel parameters (unified struct)
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
                var base = UInt(j) * UInt(DIM) + UInt(d)
                x_j0[d] = x_ptr[base]
                x_j1[d] = x_ptr[base + UInt(DIM)]
                x_j2[d] = x_ptr[base + UInt(2 * DIM)]
                x_j3[d] = x_ptr[base + UInt(3 * DIM)]
                x_j4[d] = x_ptr[base + UInt(4 * DIM)]
                x_j5[d] = x_ptr[base + UInt(5 * DIM)]
                x_j6[d] = x_ptr[base + UInt(6 * DIM)]
                x_j7[d] = x_ptr[base + UInt(7 * DIM)]
            
            # Kernel function calls - INLINED at compile time
            sum0 += kernel_fn[DIM](x_row, x_j0, params) * v0
            sum1 += kernel_fn[DIM](x_row, x_j1, params) * v1
            sum2 += kernel_fn[DIM](x_row, x_j2, params) * v2
            sum3 += kernel_fn[DIM](x_row, x_j3, params) * v3
            sum4 += kernel_fn[DIM](x_row, x_j4, params) * v4
            sum5 += kernel_fn[DIM](x_row, x_j5, params) * v5
            sum6 += kernel_fn[DIM](x_row, x_j6, params) * v6
            sum7 += kernel_fn[DIM](x_row, x_j7, params) * v7
            
            j += 8
        
        # Remainder loop
        while j < n:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            sum0 += kernel_fn[DIM](x_row, x_j, params) * v_ptr[col_offset + UInt(j)]
            j += 1
        
        var sum_val = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
        sum_val += noise * v_ptr[col_offset + UInt(row)]
        out_ptr[col_offset + UInt(row)] = sum_val


# =============================================================================
# Generic Multi-Column Forward Matvec Template (4x Unrolled, Fused Columns)
# =============================================================================

fn kernel_forward_matvec_multicol[
    DIM: Int,
    NCOLS: Int,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    params: KernelParams,
    noise: Float32,
) -> None:
    """KeOps-style shared-memory tiled forward matvec: out = (K + noise*I) @ V.

    Cooperatively loads BOTH x_j AND v_j into shared memory per tile, matching
    the KeOps GpuReduc1D architecture and reducing global-memory traffic across
    n=2000 to n=100000 benchmark envelopes.

    Shared memory layout per tile: BLOCK_SIZE * (DIM + NCOLS) floats.
    Each thread loads one j-point's x and v data cooperatively.
    Inner loop reads from shared memory (~5 cycle latency) instead of
    global memory/L2 (~30-100 cycle latency).

    Requires shared_mem_bytes = block_dim.x * (DIM + NCOLS) * 4 at launch.

    Args:
        out_ptr: Output buffer [n, NCOLS] column-major.
        x_ptr: Training data [n, DIM] row-major.
        v_ptr: Input vectors [n, NCOLS] column-major.
        n: Number of data points.
        params: Kernel parameters (unified struct).
        noise: Noise variance sigma^2.
    """
    alias DIMY = DIM + NCOLS  # floats per j-point in shared memory

    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)

    # TM=4: each block covers 4*bs consecutive output rows
    var base = Int(block_idx.x) * (bs * 4)
    var i0 = base + tid
    var i1 = base + tid + bs
    var i2 = base + tid + bs * 2
    var i3 = base + tid + bs * 3

    # Shared memory for j-point tiles: x_j AND v_j
    var yj = external_memory[
        Float32,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    var valid0 = i0 < n
    var valid1 = i1 < n
    var valid2 = i2 < n
    var valid3 = i3 < n

    # Cache x[i] in registers for all 4 rows
    var x_row0 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row1 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row2 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row3 = InlineArray[Float32, DIM](uninitialized=True)
    if valid0:
        var row_offset = UInt(i0) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row0[d] = x_ptr[row_offset + UInt(d)]
    if valid1:
        var row_offset = UInt(i1) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row1[d] = x_ptr[row_offset + UInt(d)]
    if valid2:
        var row_offset = UInt(i2) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row2[d] = x_ptr[row_offset + UInt(d)]
    if valid3:
        var row_offset = UInt(i3) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row3[d] = x_ptr[row_offset + UInt(d)]

    # Per-column accumulators for all 4 rows
    var acc0 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var acc1 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var acc2 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var acc3 = InlineArray[Float32, NCOLS](fill=Float32(0.0))

    # Tiled j-loop: cooperative load into shared memory (unchanged)
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

        # Compute: all 4 rows process the same shmem tile
        var tile_end = bs
        if jstart + bs > n:
            tile_end = n - jstart

        for jrel in range(tile_end):
            var shared_base = jrel * DIMY

            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = yj[shared_base + d]

            if valid0:
                var kval = kernel_fn[DIM](x_row0, x_j, params)
                @parameter
                for c in range(NCOLS):
                    acc0[c] += kval * yj[shared_base + DIM + c]
            if valid1:
                var kval = kernel_fn[DIM](x_row1, x_j, params)
                @parameter
                for c in range(NCOLS):
                    acc1[c] += kval * yj[shared_base + DIM + c]
            if valid2:
                var kval = kernel_fn[DIM](x_row2, x_j, params)
                @parameter
                for c in range(NCOLS):
                    acc2[c] += kval * yj[shared_base + DIM + c]
            if valid3:
                var kval = kernel_fn[DIM](x_row3, x_j, params)
                @parameter
                for c in range(NCOLS):
                    acc3[c] += kval * yj[shared_base + DIM + c]

        barrier()
        jstart += bs

    # Write output + noise diagonal for all 4 rows
    if valid0:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            out_ptr[col_off + UInt(i0)] = acc0[c] + noise * v_ptr[col_off + UInt(i0)]
    if valid1:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            out_ptr[col_off + UInt(i1)] = acc1[c] + noise * v_ptr[col_off + UInt(i1)]
    if valid2:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            out_ptr[col_off + UInt(i2)] = acc2[c] + noise * v_ptr[col_off + UInt(i2)]
    if valid3:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            out_ptr[col_off + UInt(i3)] = acc3[c] + noise * v_ptr[col_off + UInt(i3)]


# =============================================================================
# Generic Gradient Matvec Template (4x Unrolled)
# =============================================================================

fn kernel_gradient_matvec_4x[
    DIM: Int,
    gradient_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
        Int,  # grad_dim: -1 for scalar lengthscale, 0..DIM-1 for ARD
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    params: KernelParams,
    grad_dim: Int,
) -> None:
    """Generic gradient matvec: out = (∂K/∂θ) @ v
    
    This single template replaces 12+ separate gradient kernel implementations.
    
    Uses 4x unrolling (less than forward due to extra computation per element).
    No noise term (gradient of noise*I is zero for lengthscale).
    
    Args:
        out_ptr: Output buffer [n]
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vector [n]
        n: Number of data points
        params: Kernel parameters
        grad_dim: Which parameter to differentiate
            -1: scalar lengthscale (isotropic)
            0..DIM-1: per-dimension lengthscale (ARD)
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x[row]
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    var sum0 = Float32(0.0)
    var sum1 = Float32(0.0)
    var sum2 = Float32(0.0)
    var sum3 = Float32(0.0)
    
    # 4x unrolled loop
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
            var base = UInt(j) * UInt(DIM) + UInt(d)
            x_j0[d] = x_ptr[base]
            x_j1[d] = x_ptr[base + UInt(DIM)]
            x_j2[d] = x_ptr[base + UInt(2 * DIM)]
            x_j3[d] = x_ptr[base + UInt(3 * DIM)]
        
        # Gradient function calls - INLINED at compile time
        sum0 += gradient_fn[DIM](x_row, x_j0, params, grad_dim) * v0
        sum1 += gradient_fn[DIM](x_row, x_j1, params, grad_dim) * v1
        sum2 += gradient_fn[DIM](x_row, x_j2, params, grad_dim) * v2
        sum3 += gradient_fn[DIM](x_row, x_j3, params, grad_dim) * v3
        
        j += 4
    
    # Remainder
    while j < n:
        var x_j = InlineArray[Float32, DIM](uninitialized=True)
        @parameter
        for d in range(DIM):
            x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
        sum0 += gradient_fn[DIM](x_row, x_j, params, grad_dim) * v_ptr[UInt(j)]
        j += 1
    
    out_ptr[UInt(row)] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Generic Multi-Column Gradient Matvec with Shared Memory (KeOps-style)
# =============================================================================

fn kernel_gradient_matvec_shmem[
    DIM: Int,
    gradient_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
        Int,  # grad_dim
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    params: KernelParams,
    grad_dim: Int,
) -> None:
    """Single-column gradient matvec with shared memory for x_j tiling.

    Same as kernel_gradient_matvec_4x but loads x_j tiles into shared memory
    for better cache utilization. Single column only — the caller loops over
    columns. This avoids NCOLS compile-time specialization explosion while
    still getting the ~3x shared memory speedup per column.

    Requires shared_mem_bytes = block_dim.x * DIM * 4 at launch.
    """
    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)
    var i = Int(block_idx.x) * bs + tid

    var yj = external_memory[
        Float32,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    var valid = i < n

    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    if valid:
        var row_offset = UInt(i) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row[d] = x_ptr[row_offset + UInt(d)]

    var acc = Float32(0.0)

    var jstart = 0
    while jstart < n:
        var j = jstart + tid

        # Cooperative load: x_j only (single column, v is read from global)
        if j < n:
            var shared_base = tid * DIM
            @parameter
            for d in range(DIM):
                yj[shared_base + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]

        barrier()

        if valid:
            var tile_end = bs
            if jstart + bs > n:
                tile_end = n - jstart

            for jrel in range(tile_end):
                var shared_base = jrel * DIM

                var x_j = InlineArray[Float32, DIM](uninitialized=True)
                @parameter
                for d in range(DIM):
                    x_j[d] = yj[shared_base + d]

                var gval = gradient_fn[DIM](x_row, x_j, params, grad_dim)
                acc += gval * v_ptr[UInt(jstart + jrel)]

        barrier()
        jstart += bs

    if valid:
        out_ptr[UInt(i)] = acc


# =============================================================================
# Generic Multi-Column Gradient Matvec with Shared Memory (KeOps-style)
# Mirrors kernel_forward_matvec_multicol but for gradient functions.
# =============================================================================

fn kernel_gradient_matvec_shmem_multicol[
    DIM: Int,
    NCOLS: Int,
    gradient_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
        Int,  # grad_dim
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    params: KernelParams,
    grad_dim: Int,
) -> None:
    """KeOps-style shared-memory tiled gradient matvec: out = (dK/dtheta) @ V.

    Same architecture as kernel_forward_matvec_multicol but computes the
    gradient function instead of the kernel function. Cooperatively loads
    BOTH x_j AND v_j into shared memory per tile, processing all NCOLS
    columns in a single kernel launch.

    Batches all RHS columns in one shared-memory kernel launch instead of
    launching one gradient matvec per column.

    No noise term (gradient of noise*I is zero for kernel params).

    Requires shared_mem_bytes = block_dim.x * (DIM + NCOLS) * 4 at launch.

    Args:
        out_ptr: Output buffer [n, NCOLS] column-major.
        x_ptr: Training data [n, DIM] row-major.
        v_ptr: Input vectors [n, NCOLS] column-major.
        n: Number of data points.
        params: Kernel parameters (unified struct).
        grad_dim: Which parameter to differentiate
            -1: scalar lengthscale (isotropic)
            0..DIM-1: per-dimension lengthscale (ARD).
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

    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    if valid:
        var row_offset = UInt(i) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row[d] = x_ptr[row_offset + UInt(d)]

    var acc = InlineArray[Float32, NCOLS](fill=Float32(0.0))

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

                var gval = gradient_fn[DIM](x_row, x_j, params, grad_dim)

                @parameter
                for c in range(NCOLS):
                    acc[c] += gval * yj[shared_base + DIM + c]

        barrier()
        jstart += bs

    if valid:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            out_ptr[col_off + UInt(i)] = acc[c]


# =============================================================================
# Fused ls+os Multi-Column Gradient with Shared Memory (KeOps-style)
# Computes BOTH dK/dl@V and K@V in a single O(n²) pass.
# =============================================================================

fn kernel_fused_ls_os_gradient_shmem_multicol[
    DIM: Int,
    NCOLS: Int,
    gradient_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
        Int,
    ) -> Float32,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
    ) -> Float32,
](
    ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    params: KernelParams,
    grad_dim: Int,
    inv_outputscale: Float32,
) -> None:
    """Fused ls+os gradient matvec: computes BOTH dK/dl@V and (K/os)@V in one pass.

    Uses same KeOps-style shared memory architecture. For each (i,j) pair,
    evaluates both gradient_fn (ls) and kernel_fn (os) from the same shared
    memory tile. The os output is divided by outputscale to match the
    convention dK/d(os) = K/os (base kernel without outputscale factor).

    Registers: 2 × NCOLS accumulators + DIM x_row + temps ≈ 2*NCOLS + 10.
    For NCOLS=10: 30 registers — no spilling.

    Benchmarked at 1.5-1.7x faster than 2 separate shmem launches.
    Saves one full O(n²) pass over the data.

    Requires shared_mem_bytes = block_dim.x * (DIM + NCOLS) * 4 at launch.

    Args:
        ls_out_ptr: Output for ls gradient [n, NCOLS] column-major.
        os_out_ptr: Output for os gradient (K/os@V) [n, NCOLS] column-major.
        x_ptr: Training data [n, DIM] row-major.
        v_ptr: Input vectors [n, NCOLS] column-major.
        n: Number of data points.
        params: Kernel parameters (with full outputscale).
        grad_dim: -1 for isotropic, 0..DIM-1 for ARD per-dim.
        inv_outputscale: 1.0 / outputscale, to convert K to K/os for os gradient.
    """
    alias DIMY = DIM + NCOLS

    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)

    # TM=4: each block covers 4*bs consecutive output rows
    var base = Int(block_idx.x) * (bs * 4)
    var i0 = base + tid
    var i1 = base + tid + bs
    var i2 = base + tid + bs * 2
    var i3 = base + tid + bs * 3

    var yj = external_memory[
        Float32,
        address_space=AddressSpace.SHARED,
        alignment=16,
    ]()

    var valid0 = i0 < n
    var valid1 = i1 < n
    var valid2 = i2 < n
    var valid3 = i3 < n

    var x_row0 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row1 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row2 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row3 = InlineArray[Float32, DIM](uninitialized=True)
    if valid0:
        var row_offset = UInt(i0) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row0[d] = x_ptr[row_offset + UInt(d)]
    if valid1:
        var row_offset = UInt(i1) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row1[d] = x_ptr[row_offset + UInt(d)]
    if valid2:
        var row_offset = UInt(i2) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row2[d] = x_ptr[row_offset + UInt(d)]
    if valid3:
        var row_offset = UInt(i3) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row3[d] = x_ptr[row_offset + UInt(d)]

    # Two sets of accumulators per row: ls gradient + os gradient
    var ls_acc0 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc0 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var ls_acc1 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc1 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var ls_acc2 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc2 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var ls_acc3 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc3 = InlineArray[Float32, NCOLS](fill=Float32(0.0))

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

        var tile_end = bs
        if jstart + bs > n:
            tile_end = n - jstart

        for jrel in range(tile_end):
            var shared_base = jrel * DIMY

            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = yj[shared_base + d]

            if valid0:
                var gval = gradient_fn[DIM](x_row0, x_j, params, grad_dim)
                var os_val = kernel_fn[DIM](x_row0, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc0[c] += gval * vj
                    os_acc0[c] += os_val * vj
            if valid1:
                var gval = gradient_fn[DIM](x_row1, x_j, params, grad_dim)
                var os_val = kernel_fn[DIM](x_row1, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc1[c] += gval * vj
                    os_acc1[c] += os_val * vj
            if valid2:
                var gval = gradient_fn[DIM](x_row2, x_j, params, grad_dim)
                var os_val = kernel_fn[DIM](x_row2, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc2[c] += gval * vj
                    os_acc2[c] += os_val * vj
            if valid3:
                var gval = gradient_fn[DIM](x_row3, x_j, params, grad_dim)
                var os_val = kernel_fn[DIM](x_row3, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc3[c] += gval * vj
                    os_acc3[c] += os_val * vj

        barrier()
        jstart += bs

    if valid0:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i0)] = ls_acc0[c]
            os_out_ptr[col_off + UInt(i0)] = os_acc0[c]
    if valid1:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i1)] = ls_acc1[c]
            os_out_ptr[col_off + UInt(i1)] = os_acc1[c]
    if valid2:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i2)] = ls_acc2[c]
            os_out_ptr[col_off + UInt(i2)] = os_acc2[c]
    if valid3:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i3)] = ls_acc3[c]
            os_out_ptr[col_off + UInt(i3)] = os_acc3[c]


# =============================================================================
# Fused 3-param (ls+param1+os) Multi-Column Gradient with Shared Memory
# For Periodic (ls+period+os) and RQ (ls+alpha+os). 3×NCOLS accumulators.
# Proven 1.7x faster than 3 separate launches in isolated testing.
# =============================================================================

fn kernel_fused_3param_gradient_shmem_multicol[
    DIM: Int,
    NCOLS: Int,
    ls_gradient_fn: fn[D: Int](
        InlineArray[Float32, D], InlineArray[Float32, D], KernelParams, Int,
    ) -> Float32,
    param1_gradient_fn: fn[D: Int](
        InlineArray[Float32, D], InlineArray[Float32, D], KernelParams, Int,
    ) -> Float32,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D], InlineArray[Float32, D], KernelParams,
    ) -> Float32,
](
    ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    params: KernelParams,
    grad_dim: Int,
    inv_outputscale: Float32,
) -> None:
    """Fused 3-param gradient: dK/dl@V, dK/dp1@V, and (K/os)@V in one pass.

    Computes kernel value and all gradient values ONCE per (i,j) pair,
    eliminating 2 redundant O(n²) passes over the data. For compute-heavy
    kernels (Periodic with sin/cos/sqrt, RQ with power), this saves
    significant trig/math redundancy.

    Registers: 3 × NCOLS accumulators + DIM x_row + temps ≈ 3*NCOLS + 10.
    For NCOLS=10: 40 registers — no spilling.
    For NCOLS=11: 43 registers — no spilling.

    Requires shared_mem_bytes = block_dim.x * (DIM + NCOLS) * 4 at launch.
    """
    alias DIMY = DIM + NCOLS

    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)

    # TM=4: each block covers 4*bs consecutive output rows
    var base = Int(block_idx.x) * (bs * 4)
    var i0 = base + tid
    var i1 = base + tid + bs
    var i2 = base + tid + bs * 2
    var i3 = base + tid + bs * 3

    var yj = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=16,
    ]()

    var valid0 = i0 < n
    var valid1 = i1 < n
    var valid2 = i2 < n
    var valid3 = i3 < n

    var x_row0 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row1 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row2 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row3 = InlineArray[Float32, DIM](uninitialized=True)
    if valid0:
        var row_offset = UInt(i0) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row0[d] = x_ptr[row_offset + UInt(d)]
    if valid1:
        var row_offset = UInt(i1) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row1[d] = x_ptr[row_offset + UInt(d)]
    if valid2:
        var row_offset = UInt(i2) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row2[d] = x_ptr[row_offset + UInt(d)]
    if valid3:
        var row_offset = UInt(i3) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row3[d] = x_ptr[row_offset + UInt(d)]

    var ls_acc0 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var p1_acc0 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc0 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var ls_acc1 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var p1_acc1 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc1 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var ls_acc2 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var p1_acc2 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc2 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var ls_acc3 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var p1_acc3 = InlineArray[Float32, NCOLS](fill=Float32(0.0))
    var os_acc3 = InlineArray[Float32, NCOLS](fill=Float32(0.0))

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

        var tile_end = bs
        if jstart + bs > n:
            tile_end = n - jstart

        for jrel in range(tile_end):
            var shared_base = jrel * DIMY

            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = yj[shared_base + d]

            if valid0:
                var gval_ls = ls_gradient_fn[DIM](x_row0, x_j, params, grad_dim)
                var gval_p1 = param1_gradient_fn[DIM](x_row0, x_j, params, -1)
                var os_val = kernel_fn[DIM](x_row0, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc0[c] += gval_ls * vj
                    p1_acc0[c] += gval_p1 * vj
                    os_acc0[c] += os_val * vj
            if valid1:
                var gval_ls = ls_gradient_fn[DIM](x_row1, x_j, params, grad_dim)
                var gval_p1 = param1_gradient_fn[DIM](x_row1, x_j, params, -1)
                var os_val = kernel_fn[DIM](x_row1, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc1[c] += gval_ls * vj
                    p1_acc1[c] += gval_p1 * vj
                    os_acc1[c] += os_val * vj
            if valid2:
                var gval_ls = ls_gradient_fn[DIM](x_row2, x_j, params, grad_dim)
                var gval_p1 = param1_gradient_fn[DIM](x_row2, x_j, params, -1)
                var os_val = kernel_fn[DIM](x_row2, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc2[c] += gval_ls * vj
                    p1_acc2[c] += gval_p1 * vj
                    os_acc2[c] += os_val * vj
            if valid3:
                var gval_ls = ls_gradient_fn[DIM](x_row3, x_j, params, grad_dim)
                var gval_p1 = param1_gradient_fn[DIM](x_row3, x_j, params, -1)
                var os_val = kernel_fn[DIM](x_row3, x_j, params) * inv_outputscale
                @parameter
                for c in range(NCOLS):
                    var vj = yj[shared_base + DIM + c]
                    ls_acc3[c] += gval_ls * vj
                    p1_acc3[c] += gval_p1 * vj
                    os_acc3[c] += os_val * vj

        barrier()
        jstart += bs

    if valid0:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i0)] = ls_acc0[c]
            p1_out_ptr[col_off + UInt(i0)] = p1_acc0[c]
            os_out_ptr[col_off + UInt(i0)] = os_acc0[c]
    if valid1:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i1)] = ls_acc1[c]
            p1_out_ptr[col_off + UInt(i1)] = p1_acc1[c]
            os_out_ptr[col_off + UInt(i1)] = os_acc1[c]
    if valid2:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i2)] = ls_acc2[c]
            p1_out_ptr[col_off + UInt(i2)] = p1_acc2[c]
            os_out_ptr[col_off + UInt(i2)] = os_acc2[c]
    if valid3:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            ls_out_ptr[col_off + UInt(i3)] = ls_acc3[c]
            p1_out_ptr[col_off + UInt(i3)] = p1_acc3[c]
            os_out_ptr[col_off + UInt(i3)] = os_acc3[c]


# =============================================================================
# Generic 2D Gradient Matvec Template (4x Unrolled, Parallelized over rows AND columns)
# =============================================================================

fn kernel_gradient_matvec_2d_4x[
    DIM: Int,
    gradient_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
        Int,  # grad_dim: -1 for scalar lengthscale, 0..DIM-1 for ARD
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    params: KernelParams,
    grad_dim: Int,
) -> None:
    """Generic 2D gradient matvec: out = (∂K/∂θ) @ V
    
    Uses 2D grid parallelization: each thread handles one (row, col) pair.
    This eliminates the overhead of launching one kernel per column while
    maintaining the same work per thread as the single-column kernel.
    
    Grid dimensions: (num_blocks_rows, num_cols, 1)
    - block_idx.x * block_dim.x + thread_idx.x -> row index
    - block_idx.y -> column index
    
    Uses 4x unrolling (less than forward due to extra computation per element).
    No noise term (gradient of noise*I is zero for lengthscale).
    
    Args:
        out_ptr: Output buffer [n, num_cols] column-major
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Number of data points
        num_cols: Number of RHS columns (batch size)
        params: Kernel parameters
        grad_dim: Which parameter to differentiate
            -1: scalar lengthscale (isotropic)
            0..DIM-1: per-dimension lengthscale (ARD)
    """
    # 2D indexing: row from x-dimension, col from y-dimension
    var row = block_idx.x * block_dim.x + thread_idx.x
    var col = block_idx.y  # One column per y-block
    
    if row >= UInt(n) or col >= UInt(num_cols):
        return
    
    var col_offset = UInt(col) * UInt(n)
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    # Compute single (row, col) output with 4x unrolling
    var sum0 = Float32(0.0)
    var sum1 = Float32(0.0)
    var sum2 = Float32(0.0)
    var sum3 = Float32(0.0)
    
    # 4x unrolled loop
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
        
        # Gradient function calls - INLINED at compile time
        sum0 += gradient_fn[DIM](x_row, x_j0, params, grad_dim) * v0
        sum1 += gradient_fn[DIM](x_row, x_j1, params, grad_dim) * v1
        sum2 += gradient_fn[DIM](x_row, x_j2, params, grad_dim) * v2
        sum3 += gradient_fn[DIM](x_row, x_j3, params, grad_dim) * v3
        
        j += 4
    
    # Remainder
    while j < n:
        var x_j = InlineArray[Float32, DIM](uninitialized=True)
        @parameter
        for d in range(DIM):
            x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
        sum0 += gradient_fn[DIM](x_row, x_j, params, grad_dim) * v_ptr[col_offset + UInt(j)]
        j += 1
    
    out_ptr[col_offset + UInt(row)] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Generic Cross-Covariance Matvec Template (8x Unrolled)
# =============================================================================

fn kernel_cross_matvec_8x[
    DIM: Int,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    m: Int,  # Number of test points
    n: Int,  # Number of training points
    params: KernelParams,
) -> None:
    """Generic cross-covariance matvec: out = K(X_test, X_train) @ alpha
    
    This single template replaces 16+ separate cross-covariance implementations.
    
    Used for prediction (computing posterior mean).
    No noise term (rectangular matrix, no diagonal).
    
    Args:
        out_ptr: Output buffer [m]
        x_test_ptr: Test data [m, DIM] row-major
        x_train_ptr: Training data [n, DIM] row-major
        alpha_ptr: Alpha vector [n] (solution to K @ alpha = y)
        m: Number of test points
        n: Number of training points
        params: Kernel parameters
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(m):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    # Cache x_test[row]
    var x_test_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_test_row[d] = x_test_ptr[row_offset + UInt(d)]
    
    var sum0 = Float32(0.0)
    var sum1 = Float32(0.0)
    var sum2 = Float32(0.0)
    var sum3 = Float32(0.0)
    var sum4 = Float32(0.0)
    var sum5 = Float32(0.0)
    var sum6 = Float32(0.0)
    var sum7 = Float32(0.0)
    
    # 8x unrolled loop over training points
    var j = 0
    while j + 7 < n:
        var alpha0 = alpha_ptr[UInt(j)]
        var alpha1 = alpha_ptr[UInt(j + 1)]
        var alpha2 = alpha_ptr[UInt(j + 2)]
        var alpha3 = alpha_ptr[UInt(j + 3)]
        var alpha4 = alpha_ptr[UInt(j + 4)]
        var alpha5 = alpha_ptr[UInt(j + 5)]
        var alpha6 = alpha_ptr[UInt(j + 6)]
        var alpha7 = alpha_ptr[UInt(j + 7)]
        
        var x_train0 = InlineArray[Float32, DIM](uninitialized=True)
        var x_train1 = InlineArray[Float32, DIM](uninitialized=True)
        var x_train2 = InlineArray[Float32, DIM](uninitialized=True)
        var x_train3 = InlineArray[Float32, DIM](uninitialized=True)
        var x_train4 = InlineArray[Float32, DIM](uninitialized=True)
        var x_train5 = InlineArray[Float32, DIM](uninitialized=True)
        var x_train6 = InlineArray[Float32, DIM](uninitialized=True)
        var x_train7 = InlineArray[Float32, DIM](uninitialized=True)
        
        @parameter
        for d in range(DIM):
            var base = UInt(j) * UInt(DIM) + UInt(d)
            x_train0[d] = x_train_ptr[base]
            x_train1[d] = x_train_ptr[base + UInt(DIM)]
            x_train2[d] = x_train_ptr[base + UInt(2 * DIM)]
            x_train3[d] = x_train_ptr[base + UInt(3 * DIM)]
            x_train4[d] = x_train_ptr[base + UInt(4 * DIM)]
            x_train5[d] = x_train_ptr[base + UInt(5 * DIM)]
            x_train6[d] = x_train_ptr[base + UInt(6 * DIM)]
            x_train7[d] = x_train_ptr[base + UInt(7 * DIM)]
        
        # Kernel function calls - INLINED at compile time
        sum0 += kernel_fn[DIM](x_test_row, x_train0, params) * alpha0
        sum1 += kernel_fn[DIM](x_test_row, x_train1, params) * alpha1
        sum2 += kernel_fn[DIM](x_test_row, x_train2, params) * alpha2
        sum3 += kernel_fn[DIM](x_test_row, x_train3, params) * alpha3
        sum4 += kernel_fn[DIM](x_test_row, x_train4, params) * alpha4
        sum5 += kernel_fn[DIM](x_test_row, x_train5, params) * alpha5
        sum6 += kernel_fn[DIM](x_test_row, x_train6, params) * alpha6
        sum7 += kernel_fn[DIM](x_test_row, x_train7, params) * alpha7
        
        j += 8
    
    # Remainder
    while j < n:
        var x_train_j = InlineArray[Float32, DIM](uninitialized=True)
        @parameter
        for d in range(DIM):
            x_train_j[d] = x_train_ptr[UInt(j) * UInt(DIM) + UInt(d)]
        sum0 += kernel_fn[DIM](x_test_row, x_train_j, params) * alpha_ptr[UInt(j)]
        j += 1
    
    out_ptr[UInt(row)] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7


# =============================================================================
# Fused Gradient-Only Matvec Template (4x Unrolled)
# =============================================================================

fn kernel_fused_gradient_only_ard[
    DIM: Int,
    NUM_PARAMS: Int,
    fused_fn: fn[D: Int, P: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[Float32, MutAnyOrigin],
    ) -> None,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    params: KernelParams,
) -> None:
    """Fused gradient matvec: compute ALL (dK/dtheta_p @ v) in one pass.
    
    Computes k(x_i, x_j) ONCE per pair and extracts all NUM_PARAMS gradient
    values simultaneously, eliminating redundant kernel evaluations.
    
    Output layout:
        out_ptr: [NUM_PARAMS, n, num_cols] - out_ptr[p * n * num_cols + col * n + row]
        Contiguous per-parameter blocks, each [n, num_cols] column-major.
    
    Register usage (per thread):
        - DIM > 16: 2x unrolling, 2 * NUM_PARAMS accumulators
          (For ARD d=20: 2*21 = 42 registers — avoids register spill)
        - DIM <= 16: 4x unrolling, 4 * NUM_PARAMS accumulators
          (For ARD d=10: 4*11 = 44 registers — good ILP without spill)
    
    Args:
        out_ptr: Output for all gradient matvecs [NUM_PARAMS * n * num_cols]
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Number of data points
        num_cols: Number of input columns
        params: Kernel parameters (includes lengthscales, outputscale, etc.)
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_ptr[row_offset + UInt(d)]
    
    for col in range(num_cols):
        var col_offset = UInt(col) * UInt(n)
        
        @parameter
        if DIM > 16:
            # 2x unrolled variant for high-dimensional ARD (DIM > 16)
            # Reduces register pressure: 2 * NUM_PARAMS accumulators instead of 4 * NUM_PARAMS.
            # For DIM=20: 2*21 = 42 accumulators vs 4*21 = 84, avoiding register spill.
            var grad_sums = InlineArray[Float32, NUM_PARAMS * 2](uninitialized=True)
            @parameter
            for p in range(NUM_PARAMS * 2):
                grad_sums[p] = Float32(0.0)
            
            # 2x unrolled main loop
            var j = 0
            while j + 1 < n:
                var v0 = v_ptr[col_offset + UInt(j)]
                var v1 = v_ptr[col_offset + UInt(j + 1)]
                
                var x_j0 = InlineArray[Float32, DIM](uninitialized=True)
                var x_j1 = InlineArray[Float32, DIM](uninitialized=True)
                
                @parameter
                for d in range(DIM):
                    var base = UInt(j) * UInt(DIM) + UInt(d)
                    x_j0[d] = x_ptr[base]
                    x_j1[d] = x_ptr[base + UInt(DIM)]
                
                var k0_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk0_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j0, params, k0_buf.unsafe_ptr(), dk0_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums[p * 2] += dk0_buf[p] * v0
                
                var k1_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk1_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j1, params, k1_buf.unsafe_ptr(), dk1_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums[p * 2 + 1] += dk1_buf[p] * v1
                
                j += 2
            
            # Remainder loop (at most 1 element)
            var grad_sums_rem = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
            @parameter
            for p in range(NUM_PARAMS):
                grad_sums_rem[p] = Float32(0.0)
            
            while j < n:
                var x_j = InlineArray[Float32, DIM](uninitialized=True)
                @parameter
                for d in range(DIM):
                    x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                
                var k_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j, params, k_buf.unsafe_ptr(), dk_buf.unsafe_ptr())
                
                var v_j = v_ptr[col_offset + UInt(j)]
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums_rem[p] += dk_buf[p] * v_j
                j += 1
            
            # Write results for all gradient params
            @parameter
            for p in range(NUM_PARAMS):
                var grad_total = grad_sums[p * 2] + grad_sums[p * 2 + 1] + grad_sums_rem[p]
                out_ptr[UInt(p) * UInt(n * num_cols) + col_offset + UInt(row)] = grad_total
        else:
            # 4x unrolled variant for lower dimensions (DIM <= 16)
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
                
                var k0_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk0_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j0, params, k0_buf.unsafe_ptr(), dk0_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums[p * 4] += dk0_buf[p] * v0
                
                var k1_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk1_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j1, params, k1_buf.unsafe_ptr(), dk1_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums[p * 4 + 1] += dk1_buf[p] * v1
                
                var k2_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk2_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j2, params, k2_buf.unsafe_ptr(), dk2_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums[p * 4 + 2] += dk2_buf[p] * v2
                
                var k3_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk3_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j3, params, k3_buf.unsafe_ptr(), dk3_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums[p * 4 + 3] += dk3_buf[p] * v3
                
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
                
                var k_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row, x_j, params, k_buf.unsafe_ptr(), dk_buf.unsafe_ptr())
                
                var v_j = v_ptr[col_offset + UInt(j)]
                @parameter
                for p in range(NUM_PARAMS):
                    grad_sums_rem[p] += dk_buf[p] * v_j
                j += 1
            
            # Write results for all gradient params
            @parameter
            for p in range(NUM_PARAMS):
                var grad_total = grad_sums[p * 4] + grad_sums[p * 4 + 1] + grad_sums[p * 4 + 2] + grad_sums[p * 4 + 3] + grad_sums_rem[p]
                out_ptr[UInt(p) * UInt(n * num_cols) + col_offset + UInt(row)] = grad_total


# =============================================================================
# Column-Tiled Fused Gradient Matvec (compute kernel ONCE, scatter across TILE cols)
# =============================================================================

fn kernel_fused_gradient_multicol_ard[
    DIM: Int,
    NUM_PARAMS: Int,
    TILE: Int,
    fused_fn: fn[D: Int, P: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
        UnsafePointer[Float32, MutAnyOrigin],
        UnsafePointer[Float32, MutAnyOrigin],
    ) -> None,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    col_start: Int,
    params: KernelParams,
) -> None:
    """KeOps-style shared memory fused ARD gradient matvec.

    Cooperatively loads x_j AND v_j into shared memory per tile, then
    computes ALL NUM_PARAMS gradient values from the fused_fn ONCE per
    (i,j) pair. Scatters across TILE columns from shared memory reads.

    Shared memory reduces the global-memory bandwidth bottleneck while the fused
    path computes all gradient values for each input pair.

    Requires shared_mem_bytes = block_dim.x * (DIM + TILE) * 4 at launch.

    Args:
        out_ptr: Output for all gradient matvecs [NUM_PARAMS * n * num_cols]
        x_ptr: Training data [n, DIM] row-major
        v_ptr: Input vectors [n, num_cols] column-major
        n: Number of data points
        num_cols: Total number of columns (for output indexing)
        col_start: Starting column index for this tile
        params: Kernel parameters
    """
    alias DIMY = DIM + TILE

    var tid = Int(thread_idx.x)
    var bs = Int(block_dim.x)

    # TM=4: each block covers 4*bs consecutive output rows
    var base = Int(block_idx.x) * (bs * 4)
    var i0 = base + tid
    var i1 = base + tid + bs
    var i2 = base + tid + bs * 2
    var i3 = base + tid + bs * 3

    var yj = external_memory[
        Float32, address_space=AddressSpace.SHARED, alignment=16,
    ]()

    var valid0 = i0 < n
    var valid1 = i1 < n
    var valid2 = i2 < n
    var valid3 = i3 < n

    var x_row0 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row1 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row2 = InlineArray[Float32, DIM](uninitialized=True)
    var x_row3 = InlineArray[Float32, DIM](uninitialized=True)
    if valid0:
        var row_offset = UInt(i0) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row0[d] = x_ptr[row_offset + UInt(d)]
    if valid1:
        var row_offset = UInt(i1) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row1[d] = x_ptr[row_offset + UInt(d)]
    if valid2:
        var row_offset = UInt(i2) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row2[d] = x_ptr[row_offset + UInt(d)]
    if valid3:
        var row_offset = UInt(i3) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row3[d] = x_ptr[row_offset + UInt(d)]

    # Accumulators: [NUM_PARAMS × TILE] per row
    var grad_sums0 = InlineArray[Float32, NUM_PARAMS * TILE](fill=Float32(0.0))
    var grad_sums1 = InlineArray[Float32, NUM_PARAMS * TILE](fill=Float32(0.0))
    var grad_sums2 = InlineArray[Float32, NUM_PARAMS * TILE](fill=Float32(0.0))
    var grad_sums3 = InlineArray[Float32, NUM_PARAMS * TILE](fill=Float32(0.0))

    var jstart = 0
    while jstart < n:
        var j = jstart + tid

        # Cooperative load: x_j + v_j for TILE columns (unchanged)
        if j < n:
            var shared_base = tid * DIMY
            @parameter
            for d in range(DIM):
                yj[shared_base + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            @parameter
            for c in range(TILE):
                yj[shared_base + DIM + c] = v_ptr[UInt(col_start + c) * UInt(n) + UInt(j)]

        barrier()

        var tile_end = bs
        if jstart + bs > n:
            tile_end = n - jstart

        for jrel in range(tile_end):
            var shared_base = jrel * DIMY

            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = yj[shared_base + d]

            if valid0:
                var k_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row0, x_j, params, k_buf.unsafe_ptr(), dk_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    var dk_p = dk_buf[p]
                    @parameter
                    for c in range(TILE):
                        grad_sums0[p * TILE + c] += dk_p * yj[shared_base + DIM + c]
            if valid1:
                var k_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row1, x_j, params, k_buf.unsafe_ptr(), dk_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    var dk_p = dk_buf[p]
                    @parameter
                    for c in range(TILE):
                        grad_sums1[p * TILE + c] += dk_p * yj[shared_base + DIM + c]
            if valid2:
                var k_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row2, x_j, params, k_buf.unsafe_ptr(), dk_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    var dk_p = dk_buf[p]
                    @parameter
                    for c in range(TILE):
                        grad_sums2[p * TILE + c] += dk_p * yj[shared_base + DIM + c]
            if valid3:
                var k_buf = InlineArray[Float32, 1](uninitialized=True)
                var dk_buf = InlineArray[Float32, NUM_PARAMS](uninitialized=True)
                fused_fn[DIM, NUM_PARAMS](x_row3, x_j, params, k_buf.unsafe_ptr(), dk_buf.unsafe_ptr())
                @parameter
                for p in range(NUM_PARAMS):
                    var dk_p = dk_buf[p]
                    @parameter
                    for c in range(TILE):
                        grad_sums3[p * TILE + c] += dk_p * yj[shared_base + DIM + c]

        barrier()
        jstart += bs

    if valid0:
        @parameter
        for p in range(NUM_PARAMS):
            @parameter
            for c in range(TILE):
                out_ptr[UInt(p) * UInt(n * num_cols) + UInt(col_start + c) * UInt(n) + UInt(i0)] = grad_sums0[p * TILE + c]
    if valid1:
        @parameter
        for p in range(NUM_PARAMS):
            @parameter
            for c in range(TILE):
                out_ptr[UInt(p) * UInt(n * num_cols) + UInt(col_start + c) * UInt(n) + UInt(i1)] = grad_sums1[p * TILE + c]
    if valid2:
        @parameter
        for p in range(NUM_PARAMS):
            @parameter
            for c in range(TILE):
                out_ptr[UInt(p) * UInt(n * num_cols) + UInt(col_start + c) * UInt(n) + UInt(i2)] = grad_sums2[p * TILE + c]
    if valid3:
        @parameter
        for p in range(NUM_PARAMS):
            @parameter
            for c in range(TILE):
                out_ptr[UInt(p) * UInt(n * num_cols) + UInt(col_start + c) * UInt(n) + UInt(i3)] = grad_sums3[p * TILE + c]
