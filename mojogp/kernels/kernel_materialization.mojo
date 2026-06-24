"""GPU kernels for materializing kernel matrices.

Provides efficient GPU kernels to compute K[i,j] = k(x[i], x[j]) for all pairs.
Supports all 8 kernel types and ARD variants.
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

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


# =============================================================================
# Generic Kernel Materialization Template
# =============================================================================

fn kernel_materialize[
    DIM: Int,
    kernel_fn: fn[D: Int](
        InlineArray[Float32, D],
        InlineArray[Float32, D],
        KernelParams,
    ) -> Float32,
](
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × n output (row-major)
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × DIM input (row-major)
    n: Int,
    params: KernelParams,
) -> None:
    """Materialize kernel matrix K[i,j] = kernel_fn(x[i], x[j]).
    
    Each thread computes one element K[i,j].
    Grid should be (ceil(n/BLOCK_X), ceil(n/BLOCK_Y)).
    Block should be (BLOCK_X, BLOCK_Y).
    
    Output is row-major: K[i,j] = K_ptr[i * n + j]
    
    Args:
        K_ptr: Output kernel matrix [n, n] row-major
        x_ptr: Input data [n, DIM] row-major
        n: Number of data points
        params: Kernel parameters
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y
    
    if i >= UInt(n) or j >= UInt(n):
        return
    
    # Load x[i] and x[j]
    var x_i = InlineArray[Float32, DIM](uninitialized=True)
    var x_j = InlineArray[Float32, DIM](uninitialized=True)
    
    @parameter
    for d in range(DIM):
        x_i[d] = x_ptr[UInt(i) * UInt(DIM) + UInt(d)]
        x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
    
    # Compute kernel value
    var k_ij = kernel_fn[DIM](x_i, x_j, params)
    
    # Store result (row-major)
    K_ptr[UInt(i) * UInt(n) + UInt(j)] = k_ij


# =============================================================================
# Dispatcher for Kernel Materialization
# =============================================================================

fn materialize_kernel_matrix(
    ctx: DeviceContext,
    K_device: DeviceBuffer[DType.float32],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    d: Int,
    kernel_type: Int,
    params: KernelParams,
) raises:
    """Materialize the full kernel matrix K.
    
    Dispatches to the appropriate kernel based on kernel_type and dimension.
    
    Args:
        ctx: GPU device context
        K_device: Output buffer for kernel matrix [n, n]
        x_ptr: Input data [n, d] row-major
        n: Number of data points
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
    
    alias BLOCK_SIZE_X = 16  # 16x16 thread blocks
    alias BLOCK_SIZE_Y = 16
    
    var grid_dim_x = (n + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    var grid_dim_y = (n + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    var grid_dim = (grid_dim_x, grid_dim_y)
    var block_dim = (BLOCK_SIZE_X, BLOCK_SIZE_Y)
    
    var K_ptr = K_device.unsafe_ptr()
    
    # Dispatch based on kernel type and dimension
    if kernel_type == KERNEL_TYPE_RBF:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_materialize[D, rbf_kernel_unified]](
                    K_ptr, x_ptr, n, params,
                    grid_dim=grid_dim, block_dim=block_dim
                )
                return
    
    elif kernel_type == KERNEL_TYPE_MATERN12 or kernel_type == KERNEL_TYPE_MATERN32 or kernel_type == KERNEL_TYPE_MATERN52:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_materialize[D, matern_kernel_unified]](
                    K_ptr, x_ptr, n, params,
                    grid_dim=grid_dim, block_dim=block_dim
                )
                return
    
    elif kernel_type == KERNEL_TYPE_PERIODIC:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_materialize[D, periodic_kernel_unified]](
                    K_ptr, x_ptr, n, params,
                    grid_dim=grid_dim, block_dim=block_dim
                )
                return
    
    elif kernel_type == KERNEL_TYPE_RQ:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_materialize[D, rq_kernel_unified]](
                    K_ptr, x_ptr, n, params,
                    grid_dim=grid_dim, block_dim=block_dim
                )
                return
    
    elif kernel_type == KERNEL_TYPE_LINEAR:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_materialize[D, linear_kernel_unified]](
                    K_ptr, x_ptr, n, params,
                    grid_dim=grid_dim, block_dim=block_dim
                )
                return
    
    elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
        @parameter
        for D in range(1, MAX_SUPPORTED_DIM + 1):
            if d == D:
                ctx.enqueue_function[kernel_materialize[D, polynomial_kernel_unified]](
                    K_ptr, x_ptr, n, params,
                    grid_dim=grid_dim, block_dim=block_dim
                )
                return
    
    else:
        raise Error("Unknown kernel type: " + String(kernel_type))
    
    raise Error("Dimension exceeds MAX_SUPPORTED_DIM: " + String(d))


# =============================================================================
# Composite Kernel Materialization
# =============================================================================

from .composable_kernel import ComposableKernel
from collections import InlineArray


fn kernel_materialize_composite[DIM: Int, K: ComposableKernel](
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × n output (row-major)
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × DIM input (row-major)
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],  # Flat parameter array
    n: Int,
) -> None:
    """Materialize composite kernel matrix K[i,j] = K.evaluate[DIM](x[i], x[j], params).
    
    Each thread computes one element K[i,j].
    Grid: (ceil(n/16), ceil(n/16)), Block: (16, 16)
    
    Args:
        K_ptr: Output kernel matrix [n, n] row-major
        x_ptr: Input data [n, DIM] row-major
        params_ptr: Flat parameter array for the composite kernel
        n: Number of data points
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y
    
    if i >= UInt(n) or j >= UInt(n):
        return
    
    # Load x[i] and x[j] into registers
    var x_i = InlineArray[Float32, DIM](uninitialized=True)
    var x_j = InlineArray[Float32, DIM](uninitialized=True)
    
    @parameter
    for d in range(DIM):
        x_i[d] = x_ptr[UInt(i) * UInt(DIM) + UInt(d)]
        x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
    
    # Compute kernel value using composite kernel
    K_ptr[UInt(i) * UInt(n) + UInt(j)] = K.evaluate[DIM](x_i, x_j, params_ptr)


fn kernel_materialize_composite_gradient[DIM: Int, K: ComposableKernel](
    dK_ptr: UnsafePointer[Float32, MutAnyOrigin],  # n × n output (row-major)
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],   # n × DIM input (row-major)
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],  # Flat parameter array
    n: Int,
    param_idx: Int,  # Which parameter to differentiate
) -> None:
    """Materialize gradient matrix dK/dθ_p[i,j] = K.gradient[DIM](x[i], x[j], params, p).
    
    Each thread computes one element dK/dθ_p[i,j].
    Grid: (ceil(n/16), ceil(n/16)), Block: (16, 16)
    
    Args:
        dK_ptr: Output gradient matrix [n, n] row-major
        x_ptr: Input data [n, DIM] row-major
        params_ptr: Flat parameter array for the composite kernel
        n: Number of data points
        param_idx: Which parameter to differentiate (0 to K.num_params()-1)
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y
    
    if i >= UInt(n) or j >= UInt(n):
        return
    
    # Load x[i] and x[j] into registers
    var x_i = InlineArray[Float32, DIM](uninitialized=True)
    var x_j = InlineArray[Float32, DIM](uninitialized=True)
    
    @parameter
    for d in range(DIM):
        x_i[d] = x_ptr[UInt(i) * UInt(DIM) + UInt(d)]
        x_j[d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]
    
    # Compute gradient value using composite kernel
    dK_ptr[UInt(i) * UInt(n) + UInt(j)] = K.gradient[DIM](x_i, x_j, params_ptr, param_idx)


fn materialize_composite_kernel_matrix[DIM: Int, K: ComposableKernel](
    ctx: DeviceContext,
    K_device: DeviceBuffer[DType.float32],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # Device pointer
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],  # Device pointer
    n: Int,
) raises:
    """Materialize the full composite kernel matrix K on GPU.
    
    Args:
        ctx: GPU device context
        K_device: Output buffer for kernel matrix [n, n]
        x_ptr: Input data [n, DIM] row-major ON DEVICE
        params_ptr: Flat parameter array ON DEVICE
        n: Number of data points
    """
    alias BLOCK_SIZE = 16  # 16x16 thread blocks
    
    var grid_dim_x = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    var grid_dim_y = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    ctx.enqueue_function[kernel_materialize_composite[DIM, K]](
        K_device.unsafe_ptr(), x_ptr, params_ptr, n,
        grid_dim=(grid_dim_x, grid_dim_y), block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )
    ctx.synchronize()


fn materialize_composite_gradient_matrix[DIM: Int, K: ComposableKernel](
    ctx: DeviceContext,
    dK_device: DeviceBuffer[DType.float32],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],  # Device pointer
    params_ptr: UnsafePointer[Float32, MutAnyOrigin],  # Device pointer
    n: Int,
    param_idx: Int,
) raises:
    """Materialize a gradient matrix dK/dθ_p on GPU.
    
    Args:
        ctx: GPU device context
        dK_device: Output buffer for gradient matrix [n, n]
        x_ptr: Input data [n, DIM] row-major ON DEVICE
        params_ptr: Flat parameter array ON DEVICE
        n: Number of data points
        param_idx: Which parameter to differentiate
    """
    alias BLOCK_SIZE = 16  # 16x16 thread blocks
    
    var grid_dim_x = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    var grid_dim_y = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    ctx.enqueue_function[kernel_materialize_composite_gradient[DIM, K]](
        dK_device.unsafe_ptr(), x_ptr, params_ptr, n, param_idx,
        grid_dim=(grid_dim_x, grid_dim_y), block_dim=(BLOCK_SIZE, BLOCK_SIZE),
    )
    ctx.synchronize()


fn kernel_extract_diagonal_from_matrix(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n] output
    K_ptr: UnsafePointer[Float32, MutAnyOrigin],     # [n, n] input (row-major)
    n: Int,
) -> None:
    """Extract diagonal from materialized kernel matrix.
    
    diag[i] = K[i, i] = K_ptr[i * n + i]
    
    Args:
        diag_ptr: Output diagonal [n]
        K_ptr: Input kernel matrix [n, n] row-major
        n: Matrix dimension
    """
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i >= UInt(n):
        return
    
    diag_ptr[i] = K_ptr[UInt(i) * UInt(n) + UInt(i)]


fn extract_diagonal_from_matrix(
    ctx: DeviceContext,
    diag_device: DeviceBuffer[DType.float32],
    K_device: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """Extract diagonal from materialized kernel matrix.
    
    Args:
        ctx: GPU device context
        diag_device: Output buffer [n]
        K_device: Input kernel matrix [n, n]
        n: Matrix dimension
    """
    var threads_per_block = 256
    var num_blocks = (n + threads_per_block - 1) // threads_per_block
    
    ctx.enqueue_function[kernel_extract_diagonal_from_matrix](
        diag_device.unsafe_ptr(), K_device.unsafe_ptr(), n,
        grid_dim=num_blocks, block_dim=threads_per_block,
    )
    ctx.synchronize()
