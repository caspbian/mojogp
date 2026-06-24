"""Unified kernel parameter structures.

Provides a single parameter struct that can represent any kernel's parameters,
enabling generic templates that work across all kernel types.
"""

from memory import UnsafePointer


@fieldwise_init
struct KernelParams(Copyable, Movable):
    """Unified parameter struct for all kernel types.
    
    This struct contains all possible parameters needed by any kernel.
    Unused parameters are simply ignored by kernels that don't need them.
    
    Fields:
        outputscale: Output scale σ² (all kernels)
        inv_outputscale: Precomputed 1/outputscale (avoids division in fused gradients)
        lengthscale: Scalar lengthscale (isotropic mode)
        lengthscales_ptr: Per-dimension lengthscales (ARD mode), or null
        inv_ls_ptr: Precomputed 1/ls[d] per dimension (ARD mode), or null.
            Use inv_ls*inv_ls for 1/ls², inv_ls*inv_ls*inv_ls for 1/ls³.
        is_ard: Whether using ARD mode
        param1: Extra parameter 1:
            - Periodic: period
            - RQ: alpha (mixture parameter)
            - Linear: variance (bias term)
            - Polynomial: degree
            - Matern: nu (smoothness, 0.5/1.5/2.5)
        param2: Extra parameter 2:
            - Polynomial: offset
            - Others: unused (0.0)
    """
    var outputscale: Float32
    var inv_outputscale: Float32
    var lengthscale: Float32
    var lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var is_ard: Bool
    var param1: Float32
    var param2: Float32


fn make_rbf_params(
    outputscale: Float32,
    lengthscale: Float32,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    is_ard: Bool = False,
) -> KernelParams:
    """Create KernelParams for RBF kernel."""
    return KernelParams(
        outputscale=outputscale,
        inv_outputscale=Float32(1.0) / outputscale,
        lengthscale=lengthscale,
        lengthscales_ptr=lengthscales_ptr,
        inv_ls_ptr=inv_ls_ptr,
        is_ard=is_ard,
        param1=Float32(0.0),
        param2=Float32(0.0),
    )


fn make_matern_params(
    outputscale: Float32,
    lengthscale: Float32,
    nu: Float32,  # 0.5, 1.5, or 2.5
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    is_ard: Bool = False,
) -> KernelParams:
    """Create KernelParams for Matern kernel."""
    return KernelParams(
        outputscale=outputscale,
        inv_outputscale=Float32(1.0) / outputscale,
        lengthscale=lengthscale,
        lengthscales_ptr=lengthscales_ptr,
        inv_ls_ptr=inv_ls_ptr,
        is_ard=is_ard,
        param1=nu,
        param2=Float32(0.0),
    )


fn make_periodic_params(
    outputscale: Float32,
    lengthscale: Float32,
    period: Float32,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    is_ard: Bool = False,
) -> KernelParams:
    """Create KernelParams for Periodic kernel."""
    return KernelParams(
        outputscale=outputscale,
        inv_outputscale=Float32(1.0) / outputscale,
        lengthscale=lengthscale,
        lengthscales_ptr=lengthscales_ptr,
        inv_ls_ptr=inv_ls_ptr,
        is_ard=is_ard,
        param1=period,
        param2=Float32(0.0),
    )


fn make_rq_params(
    outputscale: Float32,
    lengthscale: Float32,
    alpha: Float32,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    is_ard: Bool = False,
) -> KernelParams:
    """Create KernelParams for Rational Quadratic kernel."""
    return KernelParams(
        outputscale=outputscale,
        inv_outputscale=Float32(1.0) / outputscale,
        lengthscale=lengthscale,
        lengthscales_ptr=lengthscales_ptr,
        inv_ls_ptr=inv_ls_ptr,
        is_ard=is_ard,
        param1=alpha,
        param2=Float32(0.0),
    )


fn make_linear_params(
    outputscale: Float32,
    variance: Float32,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    is_ard: Bool = False,
) -> KernelParams:
    """Create KernelParams for Linear kernel."""
    return KernelParams(
        outputscale=outputscale,
        inv_outputscale=Float32(1.0) / outputscale,
        lengthscale=Float32(1.0),  # Not used by Linear
        lengthscales_ptr=lengthscales_ptr,
        inv_ls_ptr=inv_ls_ptr,
        is_ard=is_ard,
        param1=variance,
        param2=Float32(0.0),
    )


fn make_polynomial_params(
    outputscale: Float32,
    degree: Float32,
    offset: Float32,
    lengthscales_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    inv_ls_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
    is_ard: Bool = False,
) -> KernelParams:
    """Create KernelParams for Polynomial kernel."""
    return KernelParams(
        outputscale=outputscale,
        inv_outputscale=Float32(1.0) / outputscale,
        lengthscale=Float32(1.0),  # Not used by Polynomial
        lengthscales_ptr=lengthscales_ptr,
        inv_ls_ptr=inv_ls_ptr,
        is_ard=is_ard,
        param1=degree,
        param2=offset,
    )
