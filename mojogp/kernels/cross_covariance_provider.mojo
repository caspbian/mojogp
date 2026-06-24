"""Cross-covariance computation with provider abstraction.

Provides K_test_train @ v computation for both matrix-free and materialized approaches.
Also provides fused cross-covariance matrix computation for prediction variance.
"""

from gpu.host import DeviceContext, DeviceBuffer
from memory import UnsafePointer
from .matvec_provider import MatvecProvider, MatrixFreeProvider, MaterializedProvider
from .kernel_functions import (
    rbf_kernel_unified,
    matern_kernel_unified,
    periodic_kernel_unified,
    rq_kernel_unified,
    linear_kernel_unified,
    polynomial_kernel_unified,
)
from .kernel_params import KernelParams
from .constants import MAX_SUPPORTED_DIM
from gpu.id import block_dim, block_idx, thread_idx

alias float_dtype = DType.float32


# =============================================================================
# Cross-Covariance Kernel (K_test_train @ v)
# =============================================================================

fn kernel_cross_matvec[
    DIM: Int,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [m] output
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [m, DIM] test points
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n, DIM] train points
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n] input vector
    m: Int,  # number of test points
    n: Int,  # number of train points
    params: KernelParams,
) -> None:
    """Compute out[i] = sum_j K(x_test[i], x_train[j]) * v[j].
    
    Each thread computes one output element.
    Grid should be (ceil(m/BLOCK_SIZE),).
    Block should be (BLOCK_SIZE,).
    
    Args:
        out_ptr: Output vector [m]
        x_test_ptr: Test points [m, DIM] row-major
        x_train_ptr: Train points [n, DIM] row-major
        v_ptr: Input vector [n]
        m: Number of test points
        n: Number of train points
        params: Kernel parameters
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    
    if i >= UInt(m):
        return
    
    # Load x_test[i]
    var x_i = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_i[d] = x_test_ptr[UInt(i) * UInt(DIM) + UInt(d)]
    
    # Compute sum_j K(x_test[i], x_train[j]) * v[j]
    var result: Float32 = 0.0
    for j in range(n):
        # Load x_train[j]
        var x_j = InlineArray[Float32, DIM](uninitialized=True)
        @parameter
        for d in range(DIM):
            x_j[d] = x_train_ptr[UInt(j) * UInt(DIM) + UInt(d)]
        
        # Compute kernel value
        var k_ij = kernel_fn[DIM](x_i, x_j, params)
        
        # Accumulate
        result += k_ij * v_ptr[UInt(j)]
    
    out_ptr[UInt(i)] = result


# =============================================================================
# Cross-Covariance Dispatcher
# =============================================================================

fn compute_cross_matvec(
    ctx: DeviceContext,
    out_device: DeviceBuffer[float_dtype],
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_device: DeviceBuffer[float_dtype],
    m: Int,
    n: Int,
    d: Int,
    kernel_type: Int,
    params: KernelParams,
) raises:
    """Compute K_test_train @ v using matrix-free approach.
    
    Args:
        ctx: GPU device context
        out_device: Output buffer [m]
        x_test_ptr: Test points [m, d] row-major
        x_train_ptr: Train points [n, d] row-major
        v_device: Input vector [n]
        m: Number of test points
        n: Number of train points
        d: Input dimension
        kernel_type: Kernel type constant
        params: Kernel parameters
    """
    from .constants import (
        KERNEL_TYPE_RBF,
        KERNEL_TYPE_MATERN12,
        KERNEL_TYPE_MATERN32,
        KERNEL_TYPE_MATERN52,
        KERNEL_TYPE_PERIODIC,
        KERNEL_TYPE_RQ,
        KERNEL_TYPE_LINEAR,
        KERNEL_TYPE_POLYNOMIAL,
    )
    
    alias BLOCK_SIZE = 256
    var grid_dim = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    var block_dim_val = BLOCK_SIZE
    
    var out_ptr = out_device.unsafe_ptr()
    var v_ptr = v_device.unsafe_ptr()
    
    # Dispatch based on kernel type and dimension
    if kernel_type == KERNEL_TYPE_RBF:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_matvec[D, rbf_kernel_unified]](
                    out_ptr, x_test_ptr, x_train_ptr, v_ptr, m, n, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_MATERN12 or kernel_type == KERNEL_TYPE_MATERN32 or kernel_type == KERNEL_TYPE_MATERN52:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_matvec[D, matern_kernel_unified]](
                    out_ptr, x_test_ptr, x_train_ptr, v_ptr, m, n, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_PERIODIC:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_matvec[D, periodic_kernel_unified]](
                    out_ptr, x_test_ptr, x_train_ptr, v_ptr, m, n, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_RQ:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_matvec[D, rq_kernel_unified]](
                    out_ptr, x_test_ptr, x_train_ptr, v_ptr, m, n, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_LINEAR:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_matvec[D, linear_kernel_unified]](
                    out_ptr, x_test_ptr, x_train_ptr, v_ptr, m, n, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_matvec[D, polynomial_kernel_unified]](
                    out_ptr, x_test_ptr, x_train_ptr, v_ptr, m, n, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    else:
        raise Error("Unknown kernel type: " + String(kernel_type))
    
    raise Error("Dimension exceeds MAX_SUPPORTED_DIM: " + String(d))


# =============================================================================
# Fused Cross-Covariance Matrix (K_train_test)
# =============================================================================

fn kernel_cross_covariance_fused[
    DIM: Int,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
    ) -> Float32,
](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n * m] output, column-major
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n, DIM] train points
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [m, DIM] test points
    n: Int,  # number of train points
    m: Int,  # number of test points
    params: KernelParams,
) -> None:
    """Compute K_train_test[i, j] = K(x_train[i], x_test[j]) for all i, j.
    
    Each thread computes one output element K_train_test[i, j].
    Grid should be (ceil(n*m/BLOCK_SIZE),).
    Output is stored in column-major order: out_ptr[j * n + i] = K_train_test[i, j].
    
    Args:
        out_ptr: Output matrix [n * m] column-major
        x_train_ptr: Train points [n, DIM] row-major
        x_test_ptr: Test points [m, DIM] row-major
        n: Number of train points
        m: Number of test points
        params: Kernel parameters
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = UInt(n) * UInt(m)
    
    if idx >= total:
        return
    
    var i = Int(idx % UInt(n))  # train point index (row)
    var j = Int(idx // UInt(n))  # test point index (column)
    
    var x_train_i = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_train_i[d] = x_train_ptr[UInt(i) * UInt(DIM) + UInt(d)]
    
    var x_test_j = InlineArray[Float32, DIM](uninitialized=True)
    @parameter
    for d in range(DIM):
        x_test_j[d] = x_test_ptr[UInt(j) * UInt(DIM) + UInt(d)]
    
    var k_ij = kernel_fn[DIM](x_train_i, x_test_j, params)
    
    out_ptr[UInt(j) * UInt(n) + UInt(i)] = k_ij


fn compute_cross_covariance_fused(
    ctx: DeviceContext,
    out_device: DeviceBuffer[float_dtype],
    x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    m: Int,
    d: Int,
    kernel_type: Int,
    params: KernelParams,
) raises:
    """Compute full cross-covariance matrix K_train_test [n × m] in one kernel launch.
    
    Args:
        ctx: GPU device context
        out_device: Output buffer [n * m] column-major
        x_train_ptr: Train points [n, d] row-major
        x_test_ptr: Test points [m, d] row-major
        n: Number of train points
        m: Number of test points
        d: Input dimension
        kernel_type: Kernel type constant
        params: Kernel parameters
    """
    from .constants import (
        KERNEL_TYPE_RBF,
        KERNEL_TYPE_MATERN12,
        KERNEL_TYPE_MATERN32,
        KERNEL_TYPE_MATERN52,
        KERNEL_TYPE_PERIODIC,
        KERNEL_TYPE_RQ,
        KERNEL_TYPE_LINEAR,
        KERNEL_TYPE_POLYNOMIAL,
    )
    
    alias BLOCK_SIZE = 256
    var total_elements = n * m
    var grid_dim = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    var block_dim_val = BLOCK_SIZE
    
    var out_ptr = out_device.unsafe_ptr()
    
    if kernel_type == KERNEL_TYPE_RBF:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_covariance_fused[D, rbf_kernel_unified]](
                    out_ptr, x_train_ptr, x_test_ptr, n, m, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_MATERN12 or kernel_type == KERNEL_TYPE_MATERN32 or kernel_type == KERNEL_TYPE_MATERN52:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_covariance_fused[D, matern_kernel_unified]](
                    out_ptr, x_train_ptr, x_test_ptr, n, m, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_PERIODIC:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_covariance_fused[D, periodic_kernel_unified]](
                    out_ptr, x_train_ptr, x_test_ptr, n, m, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_RQ:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_covariance_fused[D, rq_kernel_unified]](
                    out_ptr, x_train_ptr, x_test_ptr, n, m, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_LINEAR:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_covariance_fused[D, linear_kernel_unified]](
                    out_ptr, x_train_ptr, x_test_ptr, n, m, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_cross_covariance_fused[D, polynomial_kernel_unified]](
                    out_ptr, x_train_ptr, x_test_ptr, n, m, params,
                    grid_dim=(grid_dim,), block_dim=(block_dim_val,)
                )
                return
    
    else:
        raise Error("Unknown kernel type: " + String(kernel_type))
    
    raise Error("Dimension exceeds MAX_SUPPORTED_DIM: " + String(d))


# =============================================================================
# Provider-Based Cross-Covariance
# =============================================================================

fn cross_matvec_with_provider[T: MatvecProvider](
    provider: T,
    x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_device: DeviceBuffer[float_dtype],
    m: Int,
) raises -> DeviceBuffer[float_dtype]:
    """Compute K_test_train @ v using a provider.
    
    For MatrixFreeProvider: computes on-the-fly
    For MaterializedProvider: could materialize K_test_train or compute on-the-fly
    
    Args:
        provider: Provider for kernel computations
        x_test_ptr: Test points [m, d] row-major
        v_device: Input vector [n] on device
        m: Number of test points
        
    Returns:
        Output vector [m] on device
    """
    from .kernel_params import (
        make_rbf_params,
        make_matern_params,
        make_periodic_params,
        make_rq_params,
        make_linear_params,
        make_polynomial_params,
    )
    from .constants import (
        KERNEL_TYPE_RBF,
        KERNEL_TYPE_MATERN12,
        KERNEL_TYPE_MATERN32,
        KERNEL_TYPE_MATERN52,
        KERNEL_TYPE_PERIODIC,
        KERNEL_TYPE_RQ,
        KERNEL_TYPE_LINEAR,
        KERNEL_TYPE_POLYNOMIAL,
    )
    
    var ctx = provider.get_ctx()
    var n = provider.get_n()
    var d = provider.get_d()
    var kernel_type = provider.get_kernel_type()
    var x_train_ptr = provider.get_x_ptr()
    var lengthscale = provider.get_lengthscale()
    var outputscale = provider.get_outputscale()
    var kernel_param1 = provider.get_kernel_param1()
    var kernel_param2 = provider.get_kernel_param2()
    var use_ard = provider.get_use_ard()
    var lengthscales_ptr = provider.get_lengthscales_device_ptr()
    var inv_ls_ptr = provider.get_inv_ls_device_ptr()
    
    # Allocate output buffer
    var out_device = ctx.enqueue_create_buffer[float_dtype](m)
    
    # Create KernelParams based on kernel type
    var params: KernelParams
    
    if kernel_type == KERNEL_TYPE_RBF:
        params = make_rbf_params(outputscale, lengthscale, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN12:
        params = make_matern_params(outputscale, lengthscale, Float32(0.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN32:
        params = make_matern_params(outputscale, lengthscale, Float32(1.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_MATERN52:
        params = make_matern_params(outputscale, lengthscale, Float32(2.5), lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_PERIODIC:
        params = make_periodic_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_RQ:
        params = make_rq_params(outputscale, lengthscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_LINEAR:
        params = make_linear_params(outputscale, kernel_param1, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        params = make_polynomial_params(outputscale, kernel_param1, kernel_param2, lengthscales_ptr, inv_ls_ptr, is_ard=use_ard)
    else:
        raise Error("Unknown kernel type: " + String(kernel_type))
    
    # Compute K_test_train @ v using matrix-free approach
    compute_cross_matvec(
        ctx, out_device, x_test_ptr, x_train_ptr, v_device,
        m, n, d, kernel_type, params
    )
    ctx.synchronize()
    
    return out_device
