"""GPU kernel templates for mixed composite + categorical kernels.

These templates combine:
- ComposableKernel trait parameterization (compile-time kernel composition)
- Runtime categorical correlation lookup (from categorical_state)

All templates accept an `IS_PRODUCT: Bool` compile-time parameter that controls
how the continuous and categorical parts are combined:

- Product mode (IS_PRODUCT=True):
    k_total(i,j) = K.evaluate[DIM](x_i, x_j, params) * k_cat(c_i, c_j)

- Sum mode (IS_PRODUCT=False):
    k_total(i,j) = K.evaluate[DIM](x_i, x_j, params) + k_cat(c_i, c_j)

In both modes, k_cat is the product of per-variable correlation matrix lookups:
    k_cat(c_i, c_j) = prod_v R_v[c_i^v, c_j^v]

IS_PRODUCT only controls how K_cont and K_cat are combined (product vs sum).

where K is an arbitrary composite kernel (RBF+Matern52, ScaleKernel[RBF], etc.)
and k_cat is the product of per-variable correlation matrix lookups.

Templates provided:
- composite_mixed_forward_matvec_8x: (K_mixed + noise*I) @ v
- composite_mixed_forward_matvec_multicol: Multi-column fused forward matvec
- composite_mixed_gradient_cont_matvec_4x: (dK_cont/dtheta * K_cat) @ v (single param, batched)
- composite_mixed_gradient_cat_matvec_4x: (K_cont * dK_cat/dtheta) @ v (single param, batched)
- composite_mixed_materialize: Materialize full K_mixed matrix
- composite_mixed_cross_matvec_8x: K(X_test, X_train) @ v for prediction
- composite_mixed_cross_covariance_fused: K(X_train, X_test) for LOVE variance
- composite_mixed_extract_diagonal: diag[i] = K_mixed(x_i, x_i)
"""

from collections import InlineArray
from gpu.id import block_dim, block_idx, thread_idx
from gpu.sync import barrier
from gpu.memory import external_memory
from memory import UnsafePointer, AddressSpace

from .composable_kernel import ComposableKernel


# =============================================================================
# Inline Categorical Correlation Lookup
# =============================================================================
# This is inlined into every GPU kernel for performance (no function call overhead).
# For each pair (i, j), computes: prod_v R_v[c_i^v, c_j^v]
# When l_i == l_j, the contribution is 1.0 (skipped for efficiency).


# =============================================================================
# Forward Matvec: out = (K_cont * K_cat + noise*I) @ v
# =============================================================================

fn composite_mixed_forward_matvec_8x[
    DIM: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],       # [n, num_cols] column-major
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, DIM] row-major
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, num_cols] column-major
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    c_ptr: UnsafePointer[Int32, MutAnyOrigin],           # [n, num_cat_vars] row-major
    n: Int,
    num_cols: Int,
    num_cat_vars: Int,
    corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
    noise: Float32,
) -> None:
    """Matrix-free mixed composite forward matvec.
    
    Product mode (IS_PRODUCT=True):
        out[i] = sum_j K.evaluate(x_i, x_j) * k_cat(c_i, c_j) * v[j] + noise * v[i]
    Sum mode (IS_PRODUCT=False):
        out[i] = sum_j (K.evaluate(x_i, x_j) + k_cat(c_i, c_j)) * v[j] + noise * v[i]
    
    8x unrolled inner loop for instruction-level parallelism.
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    var cat_row_offset = UInt(row) * UInt(num_cat_vars)
    
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
            
            # Load x_j vectors
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
            
            # Compute continuous kernel values
            var k0 = K.evaluate[DIM](x_row, x_j0, params_ptr)
            var k1 = K.evaluate[DIM](x_row, x_j1, params_ptr)
            var k2 = K.evaluate[DIM](x_row, x_j2, params_ptr)
            var k3 = K.evaluate[DIM](x_row, x_j3, params_ptr)
            var k4 = K.evaluate[DIM](x_row, x_j4, params_ptr)
            var k5 = K.evaluate[DIM](x_row, x_j5, params_ptr)
            var k6 = K.evaluate[DIM](x_row, x_j6, params_ptr)
            var k7 = K.evaluate[DIM](x_row, x_j7, params_ptr)
            
            # Combine continuous and categorical parts
            @parameter
            if IS_PRODUCT:
                # Product mode: multiply categorical correlation into k_val
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    var cat_base = UInt(v)
                    
                    var l_j0 = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + cat_base])
                    var l_j1 = Int(c_ptr[UInt(j + 1) * UInt(num_cat_vars) + cat_base])
                    var l_j2 = Int(c_ptr[UInt(j + 2) * UInt(num_cat_vars) + cat_base])
                    var l_j3 = Int(c_ptr[UInt(j + 3) * UInt(num_cat_vars) + cat_base])
                    var l_j4 = Int(c_ptr[UInt(j + 4) * UInt(num_cat_vars) + cat_base])
                    var l_j5 = Int(c_ptr[UInt(j + 5) * UInt(num_cat_vars) + cat_base])
                    var l_j6 = Int(c_ptr[UInt(j + 6) * UInt(num_cat_vars) + cat_base])
                    var l_j7 = Int(c_ptr[UInt(j + 7) * UInt(num_cat_vars) + cat_base])
                    
                    if l_i != l_j0:
                        k0 *= corr_flat_ptr[offset + l_i * L + l_j0]
                    if l_i != l_j1:
                        k1 *= corr_flat_ptr[offset + l_i * L + l_j1]
                    if l_i != l_j2:
                        k2 *= corr_flat_ptr[offset + l_i * L + l_j2]
                    if l_i != l_j3:
                        k3 *= corr_flat_ptr[offset + l_i * L + l_j3]
                    if l_i != l_j4:
                        k4 *= corr_flat_ptr[offset + l_i * L + l_j4]
                    if l_i != l_j5:
                        k5 *= corr_flat_ptr[offset + l_i * L + l_j5]
                    if l_i != l_j6:
                        k6 *= corr_flat_ptr[offset + l_i * L + l_j6]
                    if l_i != l_j7:
                        k7 *= corr_flat_ptr[offset + l_i * L + l_j7]
            else:
                # Sum mode: compute k_cat separately, then add to k_cont
                var cat0 = Float32(1.0)
                var cat1 = Float32(1.0)
                var cat2 = Float32(1.0)
                var cat3 = Float32(1.0)
                var cat4 = Float32(1.0)
                var cat5 = Float32(1.0)
                var cat6 = Float32(1.0)
                var cat7 = Float32(1.0)
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    var cat_base = UInt(v)
                    
                    var l_j0 = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + cat_base])
                    var l_j1 = Int(c_ptr[UInt(j + 1) * UInt(num_cat_vars) + cat_base])
                    var l_j2 = Int(c_ptr[UInt(j + 2) * UInt(num_cat_vars) + cat_base])
                    var l_j3 = Int(c_ptr[UInt(j + 3) * UInt(num_cat_vars) + cat_base])
                    var l_j4 = Int(c_ptr[UInt(j + 4) * UInt(num_cat_vars) + cat_base])
                    var l_j5 = Int(c_ptr[UInt(j + 5) * UInt(num_cat_vars) + cat_base])
                    var l_j6 = Int(c_ptr[UInt(j + 6) * UInt(num_cat_vars) + cat_base])
                    var l_j7 = Int(c_ptr[UInt(j + 7) * UInt(num_cat_vars) + cat_base])
                    
                    if l_i != l_j0:
                        cat0 *= corr_flat_ptr[offset + l_i * L + l_j0]
                    if l_i != l_j1:
                        cat1 *= corr_flat_ptr[offset + l_i * L + l_j1]
                    if l_i != l_j2:
                        cat2 *= corr_flat_ptr[offset + l_i * L + l_j2]
                    if l_i != l_j3:
                        cat3 *= corr_flat_ptr[offset + l_i * L + l_j3]
                    if l_i != l_j4:
                        cat4 *= corr_flat_ptr[offset + l_i * L + l_j4]
                    if l_i != l_j5:
                        cat5 *= corr_flat_ptr[offset + l_i * L + l_j5]
                    if l_i != l_j6:
                        cat6 *= corr_flat_ptr[offset + l_i * L + l_j6]
                    if l_i != l_j7:
                        cat7 *= corr_flat_ptr[offset + l_i * L + l_j7]
                k0 = k0 + cat0
                k1 = k1 + cat1
                k2 = k2 + cat2
                k3 = k3 + cat3
                k4 = k4 + cat4
                k5 = k5 + cat5
                k6 = k6 + cat6
                k7 = k7 + cat7
            
            sum0 += k0 * v0
            sum1 += k1 * v1
            sum2 += k2 * v2
            sum3 += k3 * v3
            sum4 += k4 * v4
            sum5 += k5 * v5
            sum6 += k6 * v6
            sum7 += k7 * v7
            
            j += 8
        
        # Handle remainder
        while j < n:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            
            var k_val = K.evaluate[DIM](x_row, x_j, params_ptr)
            
            # Categorical correlation
            @parameter
            if IS_PRODUCT:
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var l_j = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + UInt(v)])
                    if l_i != l_j:
                        var offset = Int(corr_offsets_ptr[v])
                        var L = Int(corr_levels_ptr[v])
                        k_val *= corr_flat_ptr[offset + l_i * L + l_j]
            else:
                var k_cat = Float32(1.0)
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var l_j = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + UInt(v)])
                    if l_i != l_j:
                        var offset = Int(corr_offsets_ptr[v])
                        var L = Int(corr_levels_ptr[v])
                        k_cat *= corr_flat_ptr[offset + l_i * L + l_j]
                k_val = k_val + k_cat
            
            sum0 += k_val * v_ptr[col_offset + UInt(j)]
            j += 1
        
        var total = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
        
        # Add noise to diagonal
        total += noise * v_ptr[col_offset + row]
        
        out_ptr[col_offset + row] = total


# =============================================================================
# Multi-Column Forward Matvec (4x Unrolled, Fused Columns)
# =============================================================================

alias _MULTICOL_NCOLS = 11
alias _MULTICOL_NCOLS_6 = 6

fn composite_mixed_forward_matvec_multicol[
    DIM: Int,
    NCOLS: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],       # [n, NCOLS] column-major
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, DIM] row-major
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, NCOLS] column-major
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    c_ptr: UnsafePointer[Int32, MutAnyOrigin],           # [n, num_cat_vars] row-major
    n: Int,
    num_cat_vars: Int,
    corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
    noise: Float32,
) -> None:
    """Multi-column fused forward matvec for mixed composite kernels.
    
    Uses shared memory tiling for x_j + v_j. Categorical correlation
    indices are read from global memory (L1-cached, small).
    
    Product mode (IS_PRODUCT=True): k_total = K_cont * k_cat
    Sum mode (IS_PRODUCT=False): k_total = K_cont + k_cat
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
    var cat_row_offset = UInt(i) * UInt(num_cat_vars)

    # Cache x[i] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    if valid:
        var row_offset = UInt(i) * UInt(DIM)
        @parameter
        for d in range(DIM):
            x_row[d] = x_ptr[row_offset + UInt(d)]

    var acc = InlineArray[Float32, NCOLS](fill=Float32(0.0))

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

                # Compute continuous kernel from shmem
                var kval = K.evaluate[DIM](x_row, x_j, params_ptr)

                # Categorical correlation from global memory (L1-cached)
                var actual_j = jstart + jrel
                @parameter
                if IS_PRODUCT:
                    for v in range(num_cat_vars):
                        var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                        var l_j = Int(c_ptr[UInt(actual_j) * UInt(num_cat_vars) + UInt(v)])
                        if l_i != l_j:
                            var offset = Int(corr_offsets_ptr[v])
                            var L = Int(corr_levels_ptr[v])
                            kval *= corr_flat_ptr[offset + l_i * L + l_j]
                else:
                    var k_cat = Float32(1.0)
                    for v in range(num_cat_vars):
                        var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                        var l_j = Int(c_ptr[UInt(actual_j) * UInt(num_cat_vars) + UInt(v)])
                        if l_i != l_j:
                            var offset = Int(corr_offsets_ptr[v])
                            var L = Int(corr_levels_ptr[v])
                            k_cat *= corr_flat_ptr[offset + l_i * L + l_j]
                    kval = kval + k_cat

                @parameter
                for c in range(NCOLS):
                    acc[c] += kval * yj[shared_base + DIM + c]
        barrier()
        jstart += bs

    if valid:
        @parameter
        for c in range(NCOLS):
            var col_off = UInt(c) * UInt(n)
            out_ptr[col_off + UInt(i)] = acc[c] + noise * v_ptr[col_off + UInt(i)]


# =============================================================================
# Continuous Gradient Matvec: out = (dK_cont/dtheta_p * K_cat) @ v
# =============================================================================

fn composite_mixed_gradient_cont_matvec_4x[
    DIM: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],       # [n, num_cols] column-major
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, DIM] row-major
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, num_cols] column-major
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    c_ptr: UnsafePointer[Int32, MutAnyOrigin],           # [n, num_cat_vars] row-major
    n: Int,
    num_cols: Int,
    num_cat_vars: Int,
    corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
    param_index: Int,
) -> None:
    """Gradient matvec for a single continuous kernel parameter.
    
    Product mode (IS_PRODUCT=True):
        out[i,c] = sum_j dK_cont/dtheta(x_i, x_j) * k_cat(c_i, c_j) * v[j,c]
    Sum mode (IS_PRODUCT=False):
        out[i,c] = sum_j dK_cont/dtheta(x_i, x_j) * v[j,c]
        (categorical part is additive, so d(K_cont + K_cat)/d(theta_cont) = dK_cont/d(theta_cont))
    
    4x unrolled. No noise term (gradient of noise is handled by BBMM).
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    var cat_row_offset = UInt(row) * UInt(num_cat_vars)
    
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
            
            # Compute continuous gradient values
            var dk0 = K.gradient[DIM](x_row, x_j0, params_ptr, param_index)
            var dk1 = K.gradient[DIM](x_row, x_j1, params_ptr, param_index)
            var dk2 = K.gradient[DIM](x_row, x_j2, params_ptr, param_index)
            var dk3 = K.gradient[DIM](x_row, x_j3, params_ptr, param_index)
            
            # In product mode, multiply by categorical correlation
            # In sum mode, dk0..dk3 are already the correct gradients
            @parameter
            if IS_PRODUCT:
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    var cat_base = UInt(v)
                    
                    var l_j0 = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + cat_base])
                    var l_j1 = Int(c_ptr[UInt(j + 1) * UInt(num_cat_vars) + cat_base])
                    var l_j2 = Int(c_ptr[UInt(j + 2) * UInt(num_cat_vars) + cat_base])
                    var l_j3 = Int(c_ptr[UInt(j + 3) * UInt(num_cat_vars) + cat_base])
                    
                    if l_i != l_j0:
                        dk0 *= corr_flat_ptr[offset + l_i * L + l_j0]
                    if l_i != l_j1:
                        dk1 *= corr_flat_ptr[offset + l_i * L + l_j1]
                    if l_i != l_j2:
                        dk2 *= corr_flat_ptr[offset + l_i * L + l_j2]
                    if l_i != l_j3:
                        dk3 *= corr_flat_ptr[offset + l_i * L + l_j3]
            # else: sum mode - no categorical multiplication needed
            
            sum0 += dk0 * v0
            sum1 += dk1 * v1
            sum2 += dk2 * v2
            sum3 += dk3 * v3
            
            j += 4
        
        # Handle remainder
        while j < n:
            var x_j = InlineArray[Float32, DIM](uninitialized=True)
            @parameter
            for d in range(DIM):
                x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
            
            var dk_val = K.gradient[DIM](x_row, x_j, params_ptr, param_index)
            
            @parameter
            if IS_PRODUCT:
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var l_j = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + UInt(v)])
                    if l_i != l_j:
                        var offset = Int(corr_offsets_ptr[v])
                        var L = Int(corr_levels_ptr[v])
                        dk_val *= corr_flat_ptr[offset + l_i * L + l_j]
            # else: sum mode - no categorical multiplication needed
            
            sum0 += dk_val * v_ptr[col_offset + UInt(j)]
            j += 1
        
        out_ptr[col_offset + row] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Categorical Gradient Matvec: out = (K_cont * dK_cat/dtheta) @ v
# =============================================================================

fn composite_mixed_gradient_cat_matvec_4x[
    DIM: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],       # [n, num_cols] column-major
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, DIM] row-major
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, num_cols] column-major
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    c_ptr: UnsafePointer[Int32, MutAnyOrigin],           # [n, num_cat_vars] row-major
    n: Int,
    num_cols: Int,
    num_cat_vars: Int,
    grad_corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],  # gradient correlation matrices
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
) -> None:
    """Gradient matvec for a single categorical parameter.
    
    Product mode (IS_PRODUCT=True):
        out[i,c] = sum_j K.evaluate(x_i, x_j) * dk_cat/dtheta(c_i, c_j) * v[j,c]
    Sum mode (IS_PRODUCT=False):
        out[i,c] = sum_j dk_cat/dtheta(c_i, c_j) * v[j,c]
        (continuous part is additive, so d(K_cont + K_cat)/d(theta_cat) = dK_cat/d(theta_cat))
    
    Uses grad_corr_flat_ptr which contains dR_target for the target variable and R_w for others.
    Always looks up the matrix value (including diagonal) because dR[l,l] = 0 naturally.
    
    4x unrolled. No noise term.
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(n):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    var cat_row_offset = UInt(row) * UInt(num_cat_vars)
    
    # Cache x[row] in registers (only needed in product mode, but load unconditionally
    # to keep the parameter list uniform - compiler will optimize out if unused)
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
        
        # 4x unrolled main loop
        var j = 0
        while j + 3 < n:
            var v0 = v_ptr[col_offset + UInt(j)]
            var v1 = v_ptr[col_offset + UInt(j + 1)]
            var v2 = v_ptr[col_offset + UInt(j + 2)]
            var v3 = v_ptr[col_offset + UInt(j + 3)]
            
            @parameter
            if IS_PRODUCT:
                # Product mode: compute K_cont * dk_cat
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
                
                # Compute continuous kernel values
                var k0 = K.evaluate[DIM](x_row, x_j0, params_ptr)
                var k1 = K.evaluate[DIM](x_row, x_j1, params_ptr)
                var k2 = K.evaluate[DIM](x_row, x_j2, params_ptr)
                var k3 = K.evaluate[DIM](x_row, x_j3, params_ptr)
                
                # Multiply by gradient correlation (always look up, including diagonal)
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    var cat_base = UInt(v)
                    
                    var l_j0 = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + cat_base])
                    var l_j1 = Int(c_ptr[UInt(j + 1) * UInt(num_cat_vars) + cat_base])
                    var l_j2 = Int(c_ptr[UInt(j + 2) * UInt(num_cat_vars) + cat_base])
                    var l_j3 = Int(c_ptr[UInt(j + 3) * UInt(num_cat_vars) + cat_base])
                    
                    # Always look up (dR[l,l] = 0 for target var handles diagonal naturally)
                    k0 *= grad_corr_flat_ptr[offset + l_i * L + l_j0]
                    k1 *= grad_corr_flat_ptr[offset + l_i * L + l_j1]
                    k2 *= grad_corr_flat_ptr[offset + l_i * L + l_j2]
                    k3 *= grad_corr_flat_ptr[offset + l_i * L + l_j3]
                
                sum0 += k0 * v0
                sum1 += k1 * v1
                sum2 += k2 * v2
                sum3 += k3 * v3
            else:
                # Sum mode: just dk_cat, skip K.evaluate entirely
                var k0 = Float32(1.0)
                var k1 = Float32(1.0)
                var k2 = Float32(1.0)
                var k3 = Float32(1.0)
                
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    var cat_base = UInt(v)
                    
                    var l_j0 = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + cat_base])
                    var l_j1 = Int(c_ptr[UInt(j + 1) * UInt(num_cat_vars) + cat_base])
                    var l_j2 = Int(c_ptr[UInt(j + 2) * UInt(num_cat_vars) + cat_base])
                    var l_j3 = Int(c_ptr[UInt(j + 3) * UInt(num_cat_vars) + cat_base])
                    
                    k0 *= grad_corr_flat_ptr[offset + l_i * L + l_j0]
                    k1 *= grad_corr_flat_ptr[offset + l_i * L + l_j1]
                    k2 *= grad_corr_flat_ptr[offset + l_i * L + l_j2]
                    k3 *= grad_corr_flat_ptr[offset + l_i * L + l_j3]
                
                sum0 += k0 * v0
                sum1 += k1 * v1
                sum2 += k2 * v2
                sum3 += k3 * v3
            
            j += 4
        
        # Handle remainder
        while j < n:
            @parameter
            if IS_PRODUCT:
                var x_j = InlineArray[Float32, DIM](uninitialized=True)
                @parameter
                for d in range(DIM):
                    x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
                
                var k_val = K.evaluate[DIM](x_row, x_j, params_ptr)
                
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var l_j = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + UInt(v)])
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    k_val *= grad_corr_flat_ptr[offset + l_i * L + l_j]
                
                sum0 += k_val * v_ptr[col_offset + UInt(j)]
            else:
                var k_val = Float32(1.0)
                
                for v in range(num_cat_vars):
                    var l_i = Int(c_ptr[cat_row_offset + UInt(v)])
                    var l_j = Int(c_ptr[UInt(j) * UInt(num_cat_vars) + UInt(v)])
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    k_val *= grad_corr_flat_ptr[offset + l_i * L + l_j]
                
                sum0 += k_val * v_ptr[col_offset + UInt(j)]
            j += 1
        
        out_ptr[col_offset + row] = sum0 + sum1 + sum2 + sum3


# =============================================================================
# Materialize: K_mixed[i,j] = K.evaluate(x_i, x_j, params) * k_cat(c_i, c_j)
# =============================================================================

fn composite_mixed_materialize[
    DIM: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, n] output row-major
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, DIM] row-major
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    c_ptr: UnsafePointer[Int32, MutAnyOrigin],           # [n, num_cat_vars] row-major
    n: Int,
    num_cat_vars: Int,
    corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
) -> None:
    """Materialize the full mixed kernel matrix.
    
    Product mode (IS_PRODUCT=True):
        K_mixed[i,j] = K.evaluate(x_i, x_j) * k_cat(c_i, c_j)
    Sum mode (IS_PRODUCT=False):
        K_mixed[i,j] = K.evaluate(x_i, x_j) + k_cat(c_i, c_j)
    
    Grid: (ceil(n/16), ceil(n/16)), Block: (16, 16).
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y
    
    if i >= UInt(n) or j >= UInt(n):
        return
    
    # Load x_i and x_j
    var x_i = InlineArray[Float32, DIM](uninitialized=True)
    var x_j = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_i[d] = x_ptr[i * UInt(DIM) + UInt(d)]
        x_j[d] = x_ptr[j * UInt(DIM) + UInt(d)]
    
    # Continuous kernel
    var k_val = K.evaluate[DIM](x_i, x_j, params_ptr)
    
    # Categorical correlation
    @parameter
    if IS_PRODUCT:
        for v in range(num_cat_vars):
            var l_i = Int(c_ptr[i * UInt(num_cat_vars) + UInt(v)])
            var l_j = Int(c_ptr[j * UInt(num_cat_vars) + UInt(v)])
            if l_i != l_j:
                var offset = Int(corr_offsets_ptr[v])
                var L = Int(corr_levels_ptr[v])
                k_val *= corr_flat_ptr[offset + l_i * L + l_j]
    else:
        var k_cat = Float32(1.0)
        for v in range(num_cat_vars):
            var l_i = Int(c_ptr[i * UInt(num_cat_vars) + UInt(v)])
            var l_j = Int(c_ptr[j * UInt(num_cat_vars) + UInt(v)])
            if l_i != l_j:
                var offset = Int(corr_offsets_ptr[v])
                var L = Int(corr_levels_ptr[v])
                k_cat *= corr_flat_ptr[offset + l_i * L + l_j]
        k_val = k_val + k_cat
    
    K_ptr[i * UInt(n) + j] = k_val


# =============================================================================
# Cross-Matvec: out = K(X_test, X_train) @ v (for prediction mean)
# =============================================================================

fn composite_mixed_cross_matvec_8x[
    DIM: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],       # [m]
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],    # [m, DIM] row-major
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],   # [n, DIM] row-major
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n] (alpha)
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    c_test_ptr: UnsafePointer[Int32, MutAnyOrigin],      # [m, num_cat_vars] row-major
    c_train_ptr: UnsafePointer[Int32, MutAnyOrigin],     # [n, num_cat_vars] row-major
    m: Int,
    n: Int,
    num_cat_vars: Int,
    corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
) -> None:
    """Cross-matvec for prediction: out[i] = sum_j K_mixed(x_test_i, x_train_j) * alpha[j].
    
    Product mode (IS_PRODUCT=True): K_mixed = K_cont * k_cat
    Sum mode (IS_PRODUCT=False): K_mixed = K_cont + k_cat
    
    8x unrolled inner loop.
    """
    var row = block_idx.x * block_dim.x + thread_idx.x
    if row >= UInt(m):
        return
    
    var row_offset = UInt(row) * UInt(DIM)
    var cat_row_offset = UInt(row) * UInt(num_cat_vars)
    
    # Cache x_test[row] in registers
    var x_row = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_row[d] = x_test_ptr[row_offset + UInt(d)]
    
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
        var v0 = v_ptr[UInt(j)]
        var v1 = v_ptr[UInt(j + 1)]
        var v2 = v_ptr[UInt(j + 2)]
        var v3 = v_ptr[UInt(j + 3)]
        var v4 = v_ptr[UInt(j + 4)]
        var v5 = v_ptr[UInt(j + 5)]
        var v6 = v_ptr[UInt(j + 6)]
        var v7 = v_ptr[UInt(j + 7)]
        
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
            x_j0[d] = x_train_ptr[base]
            x_j1[d] = x_train_ptr[base + UInt(DIM)]
            x_j2[d] = x_train_ptr[base + UInt(2 * DIM)]
            x_j3[d] = x_train_ptr[base + UInt(3 * DIM)]
            x_j4[d] = x_train_ptr[base + UInt(4 * DIM)]
            x_j5[d] = x_train_ptr[base + UInt(5 * DIM)]
            x_j6[d] = x_train_ptr[base + UInt(6 * DIM)]
            x_j7[d] = x_train_ptr[base + UInt(7 * DIM)]
        
        var k0 = K.evaluate[DIM](x_row, x_j0, params_ptr)
        var k1 = K.evaluate[DIM](x_row, x_j1, params_ptr)
        var k2 = K.evaluate[DIM](x_row, x_j2, params_ptr)
        var k3 = K.evaluate[DIM](x_row, x_j3, params_ptr)
        var k4 = K.evaluate[DIM](x_row, x_j4, params_ptr)
        var k5 = K.evaluate[DIM](x_row, x_j5, params_ptr)
        var k6 = K.evaluate[DIM](x_row, x_j6, params_ptr)
        var k7 = K.evaluate[DIM](x_row, x_j7, params_ptr)
        
        # Categorical correlation (test vs train)
        @parameter
        if IS_PRODUCT:
            # Product mode: multiply categorical correlation into k_val
            for v in range(num_cat_vars):
                var l_i = Int(c_test_ptr[cat_row_offset + UInt(v)])
                var offset = Int(corr_offsets_ptr[v])
                var L = Int(corr_levels_ptr[v])
                var cat_base = UInt(v)
                
                var l_j0 = Int(c_train_ptr[UInt(j) * UInt(num_cat_vars) + cat_base])
                var l_j1 = Int(c_train_ptr[UInt(j + 1) * UInt(num_cat_vars) + cat_base])
                var l_j2 = Int(c_train_ptr[UInt(j + 2) * UInt(num_cat_vars) + cat_base])
                var l_j3 = Int(c_train_ptr[UInt(j + 3) * UInt(num_cat_vars) + cat_base])
                var l_j4 = Int(c_train_ptr[UInt(j + 4) * UInt(num_cat_vars) + cat_base])
                var l_j5 = Int(c_train_ptr[UInt(j + 5) * UInt(num_cat_vars) + cat_base])
                var l_j6 = Int(c_train_ptr[UInt(j + 6) * UInt(num_cat_vars) + cat_base])
                var l_j7 = Int(c_train_ptr[UInt(j + 7) * UInt(num_cat_vars) + cat_base])
                
                if l_i != l_j0:
                    k0 *= corr_flat_ptr[offset + l_i * L + l_j0]
                if l_i != l_j1:
                    k1 *= corr_flat_ptr[offset + l_i * L + l_j1]
                if l_i != l_j2:
                    k2 *= corr_flat_ptr[offset + l_i * L + l_j2]
                if l_i != l_j3:
                    k3 *= corr_flat_ptr[offset + l_i * L + l_j3]
                if l_i != l_j4:
                    k4 *= corr_flat_ptr[offset + l_i * L + l_j4]
                if l_i != l_j5:
                    k5 *= corr_flat_ptr[offset + l_i * L + l_j5]
                if l_i != l_j6:
                    k6 *= corr_flat_ptr[offset + l_i * L + l_j6]
                if l_i != l_j7:
                    k7 *= corr_flat_ptr[offset + l_i * L + l_j7]
        else:
            # Sum mode: compute k_cat separately, then add to k_cont
            var cat0 = Float32(1.0)
            var cat1 = Float32(1.0)
            var cat2 = Float32(1.0)
            var cat3 = Float32(1.0)
            var cat4 = Float32(1.0)
            var cat5 = Float32(1.0)
            var cat6 = Float32(1.0)
            var cat7 = Float32(1.0)
            for v in range(num_cat_vars):
                var l_i = Int(c_test_ptr[cat_row_offset + UInt(v)])
                var offset = Int(corr_offsets_ptr[v])
                var L = Int(corr_levels_ptr[v])
                var cat_base = UInt(v)
                
                var l_j0 = Int(c_train_ptr[UInt(j) * UInt(num_cat_vars) + cat_base])
                var l_j1 = Int(c_train_ptr[UInt(j + 1) * UInt(num_cat_vars) + cat_base])
                var l_j2 = Int(c_train_ptr[UInt(j + 2) * UInt(num_cat_vars) + cat_base])
                var l_j3 = Int(c_train_ptr[UInt(j + 3) * UInt(num_cat_vars) + cat_base])
                var l_j4 = Int(c_train_ptr[UInt(j + 4) * UInt(num_cat_vars) + cat_base])
                var l_j5 = Int(c_train_ptr[UInt(j + 5) * UInt(num_cat_vars) + cat_base])
                var l_j6 = Int(c_train_ptr[UInt(j + 6) * UInt(num_cat_vars) + cat_base])
                var l_j7 = Int(c_train_ptr[UInt(j + 7) * UInt(num_cat_vars) + cat_base])
                
                if l_i != l_j0:
                    cat0 *= corr_flat_ptr[offset + l_i * L + l_j0]
                if l_i != l_j1:
                    cat1 *= corr_flat_ptr[offset + l_i * L + l_j1]
                if l_i != l_j2:
                    cat2 *= corr_flat_ptr[offset + l_i * L + l_j2]
                if l_i != l_j3:
                    cat3 *= corr_flat_ptr[offset + l_i * L + l_j3]
                if l_i != l_j4:
                    cat4 *= corr_flat_ptr[offset + l_i * L + l_j4]
                if l_i != l_j5:
                    cat5 *= corr_flat_ptr[offset + l_i * L + l_j5]
                if l_i != l_j6:
                    cat6 *= corr_flat_ptr[offset + l_i * L + l_j6]
                if l_i != l_j7:
                    cat7 *= corr_flat_ptr[offset + l_i * L + l_j7]
            k0 = k0 + cat0
            k1 = k1 + cat1
            k2 = k2 + cat2
            k3 = k3 + cat3
            k4 = k4 + cat4
            k5 = k5 + cat5
            k6 = k6 + cat6
            k7 = k7 + cat7
        
        sum0 += k0 * v0
        sum1 += k1 * v1
        sum2 += k2 * v2
        sum3 += k3 * v3
        sum4 += k4 * v4
        sum5 += k5 * v5
        sum6 += k6 * v6
        sum7 += k7 * v7
        
        j += 8
    
    # Handle remainder
    while j < n:
        var x_j = InlineArray[Float32, DIM](uninitialized=True)
        @parameter
        for d in range(DIM):
            x_j[d] = x_train_ptr[UInt(j) * UInt(DIM) + UInt(d)]
        
        var k_val = K.evaluate[DIM](x_row, x_j, params_ptr)
        
        @parameter
        if IS_PRODUCT:
            for v in range(num_cat_vars):
                var l_i = Int(c_test_ptr[cat_row_offset + UInt(v)])
                var l_j = Int(c_train_ptr[UInt(j) * UInt(num_cat_vars) + UInt(v)])
                if l_i != l_j:
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    k_val *= corr_flat_ptr[offset + l_i * L + l_j]
        else:
            var k_cat = Float32(1.0)
            for v in range(num_cat_vars):
                var l_i = Int(c_test_ptr[cat_row_offset + UInt(v)])
                var l_j = Int(c_train_ptr[UInt(j) * UInt(num_cat_vars) + UInt(v)])
                if l_i != l_j:
                    var offset = Int(corr_offsets_ptr[v])
                    var L = Int(corr_levels_ptr[v])
                    k_cat *= corr_flat_ptr[offset + l_i * L + l_j]
            k_val = k_val + k_cat
        
        sum0 += k_val * v_ptr[UInt(j)]
        j += 1
    
    out_ptr[row] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7


# =============================================================================
# Cross-Covariance Fused: out[j,i] = K_mixed(x_train_j, x_test_i) for LOVE
# =============================================================================

fn composite_mixed_cross_covariance_fused[
    DIM: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],       # [n, m] column-major
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],   # [n, DIM] row-major
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],    # [m, DIM] row-major
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    c_train_ptr: UnsafePointer[Int32, MutAnyOrigin],     # [n, num_cat_vars] row-major
    c_test_ptr: UnsafePointer[Int32, MutAnyOrigin],      # [m, num_cat_vars] row-major
    n: Int,
    m: Int,
    num_cat_vars: Int,
    corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
    corr_offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
    corr_levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
) -> None:
    """Compute cross-covariance matrix K(X_train, X_test) for LOVE variance.
    
    Product mode (IS_PRODUCT=True):
        out[j + i*n] = K.evaluate(x_train_j, x_test_i) * k_cat(c_train_j, c_test_i)
    Sum mode (IS_PRODUCT=False):
        out[j + i*n] = K.evaluate(x_train_j, x_test_i) + k_cat(c_train_j, c_test_i)
    
    Output is column-major [n, m].
    Grid: (ceil(n/16), ceil(m/16)), Block: (16, 16).
    """
    var j_idx = block_idx.x * block_dim.x + thread_idx.x  # train index
    var i_idx = block_idx.y * block_dim.y + thread_idx.y  # test index
    
    if j_idx >= UInt(n) or i_idx >= UInt(m):
        return
    
    # Load x_train[j] and x_test[i]
    var x_train = InlineArray[Float32, DIM](uninitialized=True)
    var x_test = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_train[d] = x_train_ptr[j_idx * UInt(DIM) + UInt(d)]
        x_test[d] = x_test_ptr[i_idx * UInt(DIM) + UInt(d)]
    
    # Continuous kernel
    var k_val = K.evaluate[DIM](x_train, x_test, params_ptr)
    
    # Categorical correlation
    @parameter
    if IS_PRODUCT:
        for v in range(num_cat_vars):
            var l_train = Int(c_train_ptr[j_idx * UInt(num_cat_vars) + UInt(v)])
            var l_test = Int(c_test_ptr[i_idx * UInt(num_cat_vars) + UInt(v)])
            if l_train != l_test:
                var offset = Int(corr_offsets_ptr[v])
                var L = Int(corr_levels_ptr[v])
                k_val *= corr_flat_ptr[offset + l_train * L + l_test]
    else:
        var k_cat = Float32(1.0)
        for v in range(num_cat_vars):
            var l_train = Int(c_train_ptr[j_idx * UInt(num_cat_vars) + UInt(v)])
            var l_test = Int(c_test_ptr[i_idx * UInt(num_cat_vars) + UInt(v)])
            if l_train != l_test:
                var offset = Int(corr_offsets_ptr[v])
                var L = Int(corr_levels_ptr[v])
                k_cat *= corr_flat_ptr[offset + l_train * L + l_test]
        k_val = k_val + k_cat
    
    # Column-major: out[j + i*n]
    out_ptr[j_idx + i_idx * UInt(n)] = k_val


# =============================================================================
# Extract Diagonal: diag[i] = K_mixed(x_i, x_i)
# =============================================================================

fn composite_mixed_extract_diagonal[
    DIM: Int,
    K: ComposableKernel,
    IS_PRODUCT: Bool,
](
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],      # [n]
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [n, DIM] row-major
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],    # flat composite kernel params
    n: Int,
) -> None:
    """Extract diagonal of the mixed kernel matrix.
    
    Since k_cat(c_i, c_i) = 1.0 always (R[l,l] = 1 for all variants):
    
    Product mode (IS_PRODUCT=True):
        diag[i] = K.evaluate(x_i, x_i) * 1.0 = K.evaluate(x_i, x_i)
    Sum mode (IS_PRODUCT=False):
        diag[i] = K.evaluate(x_i, x_i) + 1.0
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= UInt(n):
        return
    
    var x_i = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_i[d] = x_ptr[i * UInt(DIM) + UInt(d)]
    
    @parameter
    if IS_PRODUCT:
        diag_ptr[i] = K.evaluate[DIM](x_i, x_i, params_ptr)
    else:
        diag_ptr[i] = K.evaluate[DIM](x_i, x_i, params_ptr) + Float32(1.0)
