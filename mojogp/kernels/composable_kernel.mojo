"""Composable kernel trait and implementations for compile-time kernel composition.

This module provides:
1. ComposableKernel trait - interface for kernels that can be composed
2. Base kernel implementations (RBF, Matern, Periodic, Linear, RQ)
3. Composition operators (SumKernel, ProductKernel, ScaleKernel)

Design principles:
- All kernels read parameters from a flat UnsafePointer[Float32] at offset 0
- Composition operators offset the pointer for sub-kernels
- Arbitrary nesting is supported: SumKernel[ProductKernel[RBF, Periodic], Linear]
- All gradient computations are explicit analytical derivatives (no autograd)

Example usage:
    alias LocallyPeriodic = ProductKernel[RBFComposable, PeriodicComposable]
    alias MyKernel = SumKernel[LocallyPeriodic, LinearComposable]
    
    var k_val = MyKernel.evaluate[5](x_i, x_j, params_ptr)
    var grad = MyKernel.gradient[5](x_i, x_j, params_ptr, param_idx)
"""

from collections import InlineArray
from math import exp as math_exp, sqrt, sin, cos, log
from memory import UnsafePointer

from .constants import PI, SQRT3, SQRT5


# =============================================================================
# ComposableKernel Trait
# =============================================================================

trait ComposableKernel:
    """Kernel that can participate in compile-time composition.
    
    All kernels read parameters from a flat pointer starting at offset 0.
    Composition operators offset the pointer for sub-kernels.
    
    Parameter layout is kernel-specific:
    - RBF: [lengthscale, outputscale]
    - Periodic: [lengthscale, period, outputscale]
    - etc.
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        """Compute k(x_i, x_j) given parameters at params_ptr."""
        ...
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        """Compute dk/d(theta[param_idx]) analytically."""
        ...
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Compute all dk/d(theta[p]) for p in [0, num_params()) and write to grads_out.
        
        grads_out must have space for num_params() Float32 values.
        
        Default implementations call gradient() per param. ARD kernels override
        with fused implementations that compute shared intermediates once.
        """
        ...
    
    @staticmethod
    fn num_params() -> Int:
        """Number of hyperparameters this kernel reads from params_ptr."""
        ...


# =============================================================================
# Helper: Compute squared Euclidean distance
# =============================================================================

@always_inline
fn compute_dist_sq[DIM: Int](
    x_i: InlineArray[Float32, DIM],
    x_j: InlineArray[Float32, DIM],
) -> Float32:
    """Compute ||x_i - x_j||^2."""
    var dist_sq = Float32(0.0)
    @parameter
    for d in range(DIM):
        var diff = x_i[d] - x_j[d]
        dist_sq += diff * diff
    return dist_sq


@always_inline
fn compute_dot_product[DIM: Int](
    x_i: InlineArray[Float32, DIM],
    x_j: InlineArray[Float32, DIM],
) -> Float32:
    """Compute x_i^T x_j."""
    var dot = Float32(0.0)
    @parameter
    for d in range(DIM):
        dot += x_i[d] * x_j[d]
    return dot


# =============================================================================
# Base Kernel: RBF (Squared Exponential)
# =============================================================================

struct RBFComposable(ComposableKernel):
    """RBF (Squared Exponential) kernel.
    
    k(x, x') = outputscale * exp(-||x-x'||^2 / (2 * lengthscale^2))
    
    Parameters: [lengthscale, outputscale]
    num_params() = 2
    
    Gradients:
    - param_idx=0: dk/d(lengthscale) = k * dist_sq / lengthscale^3
    - param_idx=1: dk/d(outputscale) = k / outputscale
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var inv_2ls2 = Float32(-0.5) / (lengthscale * lengthscale)
        return outputscale * math_exp(dist_sq * inv_2ls2)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var inv_2ls2 = Float32(-0.5) / (lengthscale * lengthscale)
        var k_val = outputscale * math_exp(dist_sq * inv_2ls2)
        
        if param_idx == 0:
            # dk/d(lengthscale) = k * dist_sq / lengthscale^3
            return k_val * dist_sq / (lengthscale * lengthscale * lengthscale)
        else:
            # dk/d(outputscale) = k / outputscale
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var inv_2ls2 = Float32(-0.5) / (lengthscale * lengthscale)
        var k_val = outputscale * math_exp(dist_sq * inv_2ls2)
        grads_out[0] = k_val * dist_sq / (lengthscale * lengthscale * lengthscale)
        grads_out[1] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return 2


# =============================================================================
# Base Kernel: Matern 1/2
# =============================================================================

struct Matern12Composable(ComposableKernel):
    """Matern 1/2 kernel (exponential kernel).
    
    k(x, x') = outputscale * exp(-r / lengthscale)
    where r = ||x - x'||
    
    Parameters: [lengthscale, outputscale]
    num_params() = 2
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq)
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        
        return outputscale * math_exp(-r / lengthscale)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq)
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        
        var k_val = outputscale * math_exp(-r / lengthscale)
        
        if param_idx == 0:
            # dk/d(lengthscale) = k * r / lengthscale^2
            return k_val * r / (lengthscale * lengthscale)
        else:
            # dk/d(outputscale) = k / outputscale
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq)
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        var k_val = outputscale * math_exp(-r / lengthscale)
        grads_out[0] = k_val * r / (lengthscale * lengthscale)
        grads_out[1] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return 2


# =============================================================================
# Base Kernel: Matern 3/2
# =============================================================================

struct Matern32Composable(ComposableKernel):
    """Matern 3/2 kernel.
    
    k(x, x') = outputscale * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)
    where r = ||x - x'||, l = lengthscale
    
    Parameters: [lengthscale, outputscale]
    num_params() = 2
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq) / lengthscale
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        
        var sqrt3_r = SQRT3 * r
        return outputscale * (Float32(1.0) + sqrt3_r) * math_exp(-sqrt3_r)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq) / lengthscale
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        
        var sqrt3_r = SQRT3 * r
        var k_val = outputscale * (Float32(1.0) + sqrt3_r) * math_exp(-sqrt3_r)
        
        if param_idx == 0:
            # dk/d(lengthscale) = 3 * outputscale * dist_sq * exp(-sqrt3*r) / lengthscale^3
            return Float32(3.0) * outputscale * dist_sq * math_exp(-sqrt3_r) / (lengthscale * lengthscale * lengthscale)
        else:
            # dk/d(outputscale) = k / outputscale
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq) / lengthscale
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        var sqrt3_r = SQRT3 * r
        var exp_val = math_exp(-sqrt3_r)
        var k_val = outputscale * (Float32(1.0) + sqrt3_r) * exp_val
        grads_out[0] = Float32(3.0) * outputscale * dist_sq * exp_val / (lengthscale * lengthscale * lengthscale)
        grads_out[1] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return 2


# =============================================================================
# Base Kernel: Matern 5/2
# =============================================================================

struct Matern52Composable(ComposableKernel):
    """Matern 5/2 kernel.
    
    k(x, x') = outputscale * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
    where r = ||x - x'||, l = lengthscale
    
    Parameters: [lengthscale, outputscale]
    num_params() = 2
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq) / lengthscale
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        
        var sqrt5_r = SQRT5 * r
        return outputscale * (Float32(1.0) + sqrt5_r + Float32(1.6666667) * r * r) * math_exp(-sqrt5_r)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq) / lengthscale
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        
        var sqrt5_r = SQRT5 * r
        var k_val = outputscale * (Float32(1.0) + sqrt5_r + Float32(1.6666667) * r * r) * math_exp(-sqrt5_r)
        
        if param_idx == 0:
            # dk/d(lengthscale) = (5/3) * outputscale * dist_sq * (1 + sqrt5*r) * exp(-sqrt5*r) / lengthscale^3
            var coeff = Float32(1.6666667) * outputscale / (lengthscale * lengthscale * lengthscale)
            return coeff * dist_sq * (Float32(1.0) + sqrt5_r) * math_exp(-sqrt5_r)
        else:
            # dk/d(outputscale) = k / outputscale
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var lengthscale = params_ptr[0]
        var outputscale = params_ptr[1]
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var r = sqrt(dist_sq) / lengthscale
        var eps = Float32(1e-10)
        r = r if r > eps else eps
        var sqrt5_r = SQRT5 * r
        var exp_val = math_exp(-sqrt5_r)
        var k_val = outputscale * (Float32(1.0) + sqrt5_r + Float32(1.6666667) * r * r) * exp_val
        var coeff = Float32(1.6666667) * outputscale / (lengthscale * lengthscale * lengthscale)
        grads_out[0] = coeff * dist_sq * (Float32(1.0) + sqrt5_r) * exp_val
        grads_out[1] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return 2


# =============================================================================
# Base Kernel: Periodic
# =============================================================================

struct PeriodicComposable(ComposableKernel):
    """Periodic kernel.
    
    GPyTorch formula: K = σ² exp(-2 * Σ_d sin²(π|x_d - x'_d|/period) / l)
    
    Parameters: [lengthscale, period, outputscale]
    num_params() = 3
    
    Gradients:
    - param_idx=0: dk/d(lengthscale) = k * 2 * sin_sq_sum / l²
    - param_idx=1: dk/d(period) = k * (4π / (l * period²)) * Σ_d |x_d - x'_d| * sin(u_d) * cos(u_d)
    - param_idx=2: dk/d(outputscale) = k / outputscale
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        alias EPS = Float32(1e-6)
        var lengthscale = max(params_ptr[0], EPS)
        var period = max(params_ptr[1], EPS)
        var outputscale = params_ptr[2]
        
        var sin_sq_sum = Float32(0.0)
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var sin_arg = PI * diff / period
            var sin_val = sin(sin_arg)
            sin_sq_sum += sin_val * sin_val
        
        var exp_arg = Float32(-2.0) * sin_sq_sum / lengthscale
        return outputscale * math_exp(exp_arg)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        alias EPS = Float32(1e-6)
        var lengthscale = max(params_ptr[0], EPS)
        var period = max(params_ptr[1], EPS)
        var outputscale = params_ptr[2]
        
        var sin_sq_sum = Float32(0.0)
        var sin_cos_sum = Float32(0.0)
        
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var sin_arg = PI * diff / period
            var sin_val = sin(sin_arg)
            var cos_val = cos(sin_arg)
            sin_sq_sum += sin_val * sin_val
            sin_cos_sum += diff * sin_val * cos_val
        
        var exp_arg = Float32(-2.0) * sin_sq_sum / lengthscale
        var k_val = outputscale * math_exp(exp_arg)
        
        if param_idx == 0:
            return k_val * Float32(2.0) * sin_sq_sum / (lengthscale * lengthscale)
        elif param_idx == 1:
            return k_val * Float32(4.0) * PI * sin_cos_sum / (period * period * lengthscale)
        else:
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        alias EPS = Float32(1e-6)
        var lengthscale = max(params_ptr[0], EPS)
        var period = max(params_ptr[1], EPS)
        var outputscale = params_ptr[2]
        
        var sin_sq_sum = Float32(0.0)
        var sin_cos_sum = Float32(0.0)
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var sin_arg = PI * diff / period
            var sin_val = sin(sin_arg)
            var cos_val = cos(sin_arg)
            sin_sq_sum += sin_val * sin_val
            sin_cos_sum += diff * sin_val * cos_val
        
        var exp_arg = Float32(-2.0) * sin_sq_sum / lengthscale
        var k_val = outputscale * math_exp(exp_arg)
        grads_out[0] = k_val * Float32(2.0) * sin_sq_sum / (lengthscale * lengthscale)
        grads_out[1] = k_val * Float32(4.0) * PI * sin_cos_sum / (period * period * lengthscale)
        grads_out[2] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return 3


# =============================================================================
# Base Kernel: Linear
# =============================================================================

struct LinearComposable(ComposableKernel):
    """Linear kernel matching GPyTorch's ScaleKernel(LinearKernel()).
    
    k(x, x') = outputscale * variance * x^T x'
    
    Parameters: [variance, outputscale]
    num_params() = 2
    
    GPyTorch's LinearKernel computes k = v * x^T x' where v is the variance
    parameter (a multiplicative scale on the dot product, NOT an additive bias).
    When wrapped in ScaleKernel, the full formula is outputscale * variance * dot.
    
    Gradients:
    - param_idx=0: dk/d(variance) = outputscale * x^T x'
    - param_idx=1: dk/d(outputscale) = variance * x^T x'
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var variance = params_ptr[0]
        var outputscale = params_ptr[1]
        
        var dot = compute_dot_product[DIM](x_i, x_j)
        return outputscale * variance * dot
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var variance = params_ptr[0]
        var outputscale = params_ptr[1]
        var dot = compute_dot_product[DIM](x_i, x_j)
        
        if param_idx == 0:
            # dk/d(variance) = outputscale * x^T x'
            return outputscale * dot
        else:
            # dk/d(outputscale) = variance * x^T x'
            return variance * dot
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var variance = params_ptr[0]
        var outputscale = params_ptr[1]
        var dot = compute_dot_product[DIM](x_i, x_j)
        grads_out[0] = outputscale * dot
        grads_out[1] = variance * dot
    
    @staticmethod
    fn num_params() -> Int:
        return 2


# =============================================================================
# Base Kernel: Linear ARD
# =============================================================================

struct LinearComposableARD[D: Int](ComposableKernel):
    """Linear kernel with per-dimension variance weights (ARD).
    
    k(x, x') = outputscale * Sigma_d v_d * x_d * x'_d
    
    Parameters: [v_0, v_1, ..., v_{D-1}, outputscale]
    num_params() = D + 1
    
    GPyTorch's LinearKernel with ard_num_dims=D has per-dimension variance
    weights that control the relevance of each input dimension.
    
    Gradients:
    - param_idx < D:  dk/dv_d = outputscale * x_d * x'_d
    - param_idx == D:  dk/d(os) = k / os
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var outputscale = params_ptr[D]
        var weighted_dot = Float32(0.0)
        @parameter
        for d in range(DIM):
            weighted_dot += params_ptr[d] * x_i[d] * x_j[d]
        return outputscale * weighted_dot
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var outputscale = params_ptr[D]
        
        if param_idx < D:
            # dk/dv_d = outputscale * x_d * x'_d
            return outputscale * x_i[param_idx] * x_j[param_idx]
        else:
            # dk/d(outputscale) = k / outputscale = weighted_dot
            var weighted_dot = Float32(0.0)
            @parameter
            for d in range(DIM):
                weighted_dot += params_ptr[d] * x_i[d] * x_j[d]
            return weighted_dot
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute weighted dot once, write all D+1 gradients."""
        var outputscale = params_ptr[D]
        var weighted_dot = Float32(0.0)
        @parameter
        for d in range(DIM):
            var prod = x_i[d] * x_j[d]
            grads_out[d] = outputscale * prod
            weighted_dot += params_ptr[d] * prod
        grads_out[D] = weighted_dot
    
    @staticmethod
    fn num_params() -> Int:
        return D + 1


# =============================================================================
# Base Kernel: Polynomial
# =============================================================================

struct PolynomialComposable(ComposableKernel):
    """Polynomial kernel.
    
    k(x, x') = outputscale * (x^T x' + offset)^degree
    
    Parameters: [degree, offset, outputscale]
    num_params() = 3
    
    Note: degree is treated as a continuous parameter for gradient computation.
    In practice, degree is typically fixed (e.g., 2 or 3) and not optimized,
    but the gradient is provided for completeness.
    
    Gradients:
    - param_idx=0: dk/d(degree) = k * log(base) where base = dot + offset
    - param_idx=1: dk/d(offset) = outputscale * degree * base^(degree-1)
    - param_idx=2: dk/d(outputscale) = k / outputscale
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var degree = params_ptr[0]
        var offset = params_ptr[1]
        var outputscale = params_ptr[2]
        
        var dot = compute_dot_product[DIM](x_i, x_j)
        var base = dot + offset
        # Floor base to avoid log(0) or negative base issues
        if base < Float32(1e-10):
            base = Float32(1e-10)
        return outputscale * (base ** degree)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var degree = params_ptr[0]
        var offset = params_ptr[1]
        var outputscale = params_ptr[2]
        
        var dot = compute_dot_product[DIM](x_i, x_j)
        var base = dot + offset
        # Floor base to avoid log(0) or negative base issues
        if base < Float32(1e-10):
            base = Float32(1e-10)
        var k_val = outputscale * (base ** degree)
        
        if param_idx == 0:
            # dk/d(degree) = k * log(base)
            return k_val * log(base)
        elif param_idx == 1:
            # dk/d(offset) = outputscale * degree * base^(degree-1)
            return outputscale * degree * (base ** (degree - Float32(1.0)))
        else:
            # dk/d(outputscale) = k / outputscale
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var degree = params_ptr[0]
        var offset = params_ptr[1]
        var outputscale = params_ptr[2]
        var dot = compute_dot_product[DIM](x_i, x_j)
        var base = dot + offset
        if base < Float32(1e-10):
            base = Float32(1e-10)
        var k_val = outputscale * (base ** degree)
        grads_out[0] = k_val * log(base)
        grads_out[1] = outputscale * degree * (base ** (degree - Float32(1.0)))
        grads_out[2] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return 3


struct PolynomialComposableARD[D: Int](ComposableKernel):
    """Polynomial kernel with per-dimension lengthscales (ARD).
    
    k(x, x') = outputscale * (Sigma_d x_d * x'_d / l_d^2 + offset)^degree
    
    Parameters: [ls_0, ls_1, ..., ls_{D-1}, degree, offset, outputscale]
    num_params() = D + 3
    
    Each dimension gets its own lengthscale l_d. Dimensions with small
    lengthscales contribute more to the polynomial interaction; dimensions
    with large lengthscales are effectively ignored.
    
    Gradients:
    - param_idx < D:    dk/dl_d = os * degree * base^(d-1) * (-2 * x_d * x'_d / l_d^3)
    - param_idx == D:   dk/d(degree) = k * log(base)
    - param_idx == D+1: dk/d(offset) = os * degree * base^(d-1)
    - param_idx == D+2: dk/d(outputscale) = base^degree
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var degree = params_ptr[D]
        var offset = params_ptr[D + 1]
        var outputscale = params_ptr[D + 2]
        var weighted_dot = Float32(0.0)
        @parameter
        for d in range(DIM):
            var l_d = params_ptr[d]
            weighted_dot += x_i[d] * x_j[d] / (l_d * l_d)
        var base = weighted_dot + offset
        if base < Float32(1e-10):
            base = Float32(1e-10)
        return outputscale * (base ** degree)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var degree = params_ptr[D]
        var offset = params_ptr[D + 1]
        var outputscale = params_ptr[D + 2]
        var weighted_dot = Float32(0.0)
        @parameter
        for d in range(DIM):
            var l_d = params_ptr[d]
            weighted_dot += x_i[d] * x_j[d] / (l_d * l_d)
        var base = weighted_dot + offset
        if base < Float32(1e-10):
            base = Float32(1e-10)
        
        if param_idx < D:
            # dk/dl_d = os * degree * base^(d-1) * (-2 * x_d * x'_d / l_d^3)
            var l_d = params_ptr[param_idx]
            var prod = x_i[param_idx] * x_j[param_idx]
            return outputscale * degree * (base ** (degree - Float32(1.0))) * (Float32(-2.0) * prod / (l_d * l_d * l_d))
        elif param_idx == D:
            # dk/d(degree) = k * log(base)
            return outputscale * (base ** degree) * log(base)
        elif param_idx == D + 1:
            # dk/d(offset) = os * degree * base^(d-1)
            return outputscale * degree * (base ** (degree - Float32(1.0)))
        else:
            # dk/d(outputscale) = base^degree
            return base ** degree
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute weighted dot and base once, write all D+3 gradients."""
        var degree = params_ptr[D]
        var offset = params_ptr[D + 1]
        var outputscale = params_ptr[D + 2]
        var weighted_dot = Float32(0.0)
        @parameter
        for d in range(DIM):
            var l_d = params_ptr[d]
            weighted_dot += x_i[d] * x_j[d] / (l_d * l_d)
        var base = weighted_dot + offset
        if base < Float32(1e-10):
            base = Float32(1e-10)
        var base_pow_dm1 = base ** (degree - Float32(1.0))
        var k_val = outputscale * base_pow_dm1 * base
        var os_deg_bpd = outputscale * degree * base_pow_dm1
        
        # Lengthscale gradients
        @parameter
        for d in range(DIM):
            var l_d = params_ptr[d]
            var prod = x_i[d] * x_j[d]
            grads_out[d] = os_deg_bpd * (Float32(-2.0) * prod / (l_d * l_d * l_d))
        
        # Degree gradient: dk/d(degree) = k * log(base)
        grads_out[D] = k_val * log(base)
        # Offset gradient: dk/d(offset) = os * degree * base^(d-1)
        grads_out[D + 1] = os_deg_bpd
        # Outputscale gradient: dk/d(os) = base^degree
        grads_out[D + 2] = base_pow_dm1 * base
    
    @staticmethod
    fn num_params() -> Int:
        return D + 3


# =============================================================================
# Base Kernel: Rational Quadratic
# =============================================================================

struct RQComposable(ComposableKernel):
    """Rational Quadratic kernel.
    
    k(x, x') = outputscale * (1 + ||x-x'||^2 / (2 * alpha * lengthscale^2))^(-alpha)
    
    Parameters: [lengthscale, alpha, outputscale]
    num_params() = 3
    
    Gradients:
    - param_idx=0: dk/d(lengthscale) = k * alpha * dist_sq / (lengthscale^3 * base)
    - param_idx=1: dk/d(alpha) = k * (dist_sq/(2*alpha*l^2*base) - log(base))
    - param_idx=2: dk/d(outputscale) = k / outputscale
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var alpha = params_ptr[1]
        var outputscale = params_ptr[2]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var inv_2alpha_l2 = Float32(1.0) / (Float32(2.0) * alpha * lengthscale * lengthscale)
        var base = Float32(1.0) + dist_sq * inv_2alpha_l2
        
        # Use ** operator (pow was removed from math module)
        return outputscale * (base ** (-alpha))
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var lengthscale = params_ptr[0]
        var alpha = params_ptr[1]
        var outputscale = params_ptr[2]
        
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var inv_2alpha_l2 = Float32(1.0) / (Float32(2.0) * alpha * lengthscale * lengthscale)
        var base = Float32(1.0) + dist_sq * inv_2alpha_l2
        # Use ** operator (pow was removed from math module)
        var k_val = outputscale * (base ** (-alpha))
        
        if param_idx == 0:
            # dk/d(lengthscale) = k * dist_sq / (lengthscale^3 * base)  [alpha cancels in chain rule]
            return k_val * dist_sq / (lengthscale * lengthscale * lengthscale * base)
        elif param_idx == 1:
            # dk/d(alpha) = k * (dist_sq/(2*alpha*l^2*base) - log(base))
            var term1 = dist_sq * inv_2alpha_l2 / base
            var term2 = log(base)
            return k_val * (term1 - term2)
        else:
            # dk/d(outputscale) = k / outputscale
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var lengthscale = params_ptr[0]
        var alpha = params_ptr[1]
        var outputscale = params_ptr[2]
        var dist_sq = compute_dist_sq[DIM](x_i, x_j)
        var inv_2alpha_l2 = Float32(1.0) / (Float32(2.0) * alpha * lengthscale * lengthscale)
        var base = Float32(1.0) + dist_sq * inv_2alpha_l2
        var k_val = outputscale * (base ** (-alpha))
        grads_out[0] = k_val * dist_sq / (lengthscale * lengthscale * lengthscale * base)
        var term1 = dist_sq * inv_2alpha_l2 / base
        grads_out[1] = k_val * (term1 - log(base))
        grads_out[2] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return 3


# =============================================================================
# Composition Operator: SumKernel
# =============================================================================

struct SumKernel[K1: ComposableKernel, K2: ComposableKernel](ComposableKernel):
    """Sum of two kernels: k(x,x') = k1(x,x') + k2(x,x').
    
    Parameters layout: [K1 params... | K2 params...]
    
    Gradient: d(k1+k2)/dtheta = dk1/dtheta if theta in K1, else dk2/dtheta
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var k1_val = K1.evaluate[DIM](x_i, x_j, params_ptr)
        var k2_val = K2.evaluate[DIM](x_i, x_j, params_ptr + K1.num_params())
        return k1_val + k2_val
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        if param_idx < K1.num_params():
            return K1.gradient[DIM](x_i, x_j, params_ptr, param_idx)
        else:
            return K2.gradient[DIM](x_i, x_j, params_ptr + K1.num_params(), param_idx - K1.num_params())
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        # Delegate to sub-kernels at their respective offsets
        K1.all_gradients[DIM](x_i, x_j, params_ptr, grads_out)
        K2.all_gradients[DIM](x_i, x_j, params_ptr + K1.num_params(), grads_out + K1.num_params())
    
    @staticmethod
    fn num_params() -> Int:
        return K1.num_params() + K2.num_params()


# =============================================================================
# Composition Operator: ProductKernel
# =============================================================================

struct ProductKernel[K1: ComposableKernel, K2: ComposableKernel](ComposableKernel):
    """Product of two kernels: k(x,x') = k1(x,x') * k2(x,x').
    
    Parameters layout: [K1 params... | K2 params...]
    
    Gradient (product rule):
    - d(k1*k2)/dtheta = k2 * dk1/dtheta  if theta in K1
    - d(k1*k2)/dtheta = k1 * dk2/dtheta  if theta in K2
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var k1_val = K1.evaluate[DIM](x_i, x_j, params_ptr)
        var k2_val = K2.evaluate[DIM](x_i, x_j, params_ptr + K1.num_params())
        return k1_val * k2_val
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var k1_val = K1.evaluate[DIM](x_i, x_j, params_ptr)
        var k2_val = K2.evaluate[DIM](x_i, x_j, params_ptr + K1.num_params())
        
        if param_idx < K1.num_params():
            # Product rule: k2 * dk1/dtheta
            var dk1 = K1.gradient[DIM](x_i, x_j, params_ptr, param_idx)
            return k2_val * dk1
        else:
            # Product rule: k1 * dk2/dtheta
            var dk2 = K2.gradient[DIM](x_i, x_j, params_ptr + K1.num_params(), param_idx - K1.num_params())
            return k1_val * dk2
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        # Product rule: d(k1*k2)/dtheta = k2*dk1/dtheta (K1 params) or k1*dk2/dtheta (K2 params)
        var k1_val = K1.evaluate[DIM](x_i, x_j, params_ptr)
        var k2_val = K2.evaluate[DIM](x_i, x_j, params_ptr + K1.num_params())
        
        # Get all K1 gradients, then scale by k2
        K1.all_gradients[DIM](x_i, x_j, params_ptr, grads_out)
        for p in range(K1.num_params()):
            grads_out[p] = grads_out[p] * k2_val
        
        # Get all K2 gradients, then scale by k1
        K2.all_gradients[DIM](x_i, x_j, params_ptr + K1.num_params(), grads_out + K1.num_params())
        for p in range(K2.num_params()):
            (grads_out + K1.num_params())[p] = (grads_out + K1.num_params())[p] * k1_val
    
    @staticmethod
    fn num_params() -> Int:
        return K1.num_params() + K2.num_params()


# =============================================================================
# Composition Operator: ScaleKernel
# =============================================================================

struct ScaleKernel[K: ComposableKernel](ComposableKernel):
    """Scaled kernel: k(x,x') = scale * k_base(x,x').
    
    Parameters layout: [scale | K params...]
    
    This is useful when you want to add an explicit outputscale to a kernel
    that doesn't have one, or to add an additional scaling factor.
    
    Gradient:
    - param_idx=0: dk/d(scale) = k_base(x,x')
    - param_idx>0: dk/dtheta = scale * dk_base/dtheta
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var scale = params_ptr[0]
        var k_base = K.evaluate[DIM](x_i, x_j, params_ptr + 1)
        return scale * k_base
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var scale = params_ptr[0]
        
        if param_idx == 0:
            # dk/d(scale) = k_base
            return K.evaluate[DIM](x_i, x_j, params_ptr + 1)
        else:
            # dk/dtheta = scale * dk_base/dtheta
            var dk_base = K.gradient[DIM](x_i, x_j, params_ptr + 1, param_idx - 1)
            return scale * dk_base
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var scale = params_ptr[0]
        # dk/d(scale) = k_base
        grads_out[0] = K.evaluate[DIM](x_i, x_j, params_ptr + 1)
        # dk/dtheta = scale * dk_base/dtheta
        K.all_gradients[DIM](x_i, x_j, params_ptr + 1, grads_out + 1)
        for p in range(K.num_params()):
            (grads_out + 1)[p] = (grads_out + 1)[p] * scale
    
    @staticmethod
    fn num_params() -> Int:
        return 1 + K.num_params()


# =============================================================================
# Composition Operator: DimSliceKernel
# =============================================================================

struct DimSliceKernel[K: ComposableKernel, START: Int, END: Int](ComposableKernel):
    """Apply kernel K to dimensions [START, END) of the input.

    The inner kernel K sees SLICE_DIM = END - START dimensions.
    Parameters are passed through unchanged (DimSlice adds no params).

    This enables dimension routing in composite kernels:
        alias SpatialRBF = DimSliceKernel[RBFComposable, 0, 3]
        alias TemporalMatern = DimSliceKernel[Matern52Composable, 3, 4]
        alias SpatioTemporal = ProductKernel[SpatialRBF, TemporalMatern]

    The slice is compile-time: @parameter for loops are fully unrolled,
    so the overhead is just register-to-register copies.
    """
    alias SLICE_DIM = END - START

    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var x_i_slice = InlineArray[Float32, Self.SLICE_DIM](unsafe_uninitialized=True)
        var x_j_slice = InlineArray[Float32, Self.SLICE_DIM](unsafe_uninitialized=True)
        @parameter
        for d in range(Self.SLICE_DIM):
            x_i_slice[d] = x_i[START + d]
            x_j_slice[d] = x_j[START + d]
        return K.evaluate[Self.SLICE_DIM](x_i_slice, x_j_slice, params_ptr)

    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var x_i_slice = InlineArray[Float32, Self.SLICE_DIM](unsafe_uninitialized=True)
        var x_j_slice = InlineArray[Float32, Self.SLICE_DIM](unsafe_uninitialized=True)
        @parameter
        for d in range(Self.SLICE_DIM):
            x_i_slice[d] = x_i[START + d]
            x_j_slice[d] = x_j[START + d]
        return K.gradient[Self.SLICE_DIM](x_i_slice, x_j_slice, params_ptr, param_idx)

    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        var x_i_slice = InlineArray[Float32, Self.SLICE_DIM](unsafe_uninitialized=True)
        var x_j_slice = InlineArray[Float32, Self.SLICE_DIM](unsafe_uninitialized=True)
        @parameter
        for d in range(Self.SLICE_DIM):
            x_i_slice[d] = x_i[START + d]
            x_j_slice[d] = x_j[START + d]
        K.all_gradients[Self.SLICE_DIM](x_i_slice, x_j_slice, params_ptr, grads_out)

    @staticmethod
    fn num_params() -> Int:
        return K.num_params()


# =============================================================================
# ARD (Automatic Relevance Determination) Kernel Variants
# =============================================================================
# 
# ARD kernels use per-dimension lengthscales: [l_0, l_1, ..., l_{D-1}, ...]
# The struct parameter D is the input dimension (number of features).
# These are used by the JIT codegen when ExactGP(ard=True) is set.


# Helper: Compute weighted squared distance for ARD kernels
@always_inline
fn compute_ard_dist_sq[DIM: Int](
    x_i: InlineArray[Float32, DIM],
    x_j: InlineArray[Float32, DIM],
    params_ptr: UnsafePointer[Float32],  # per-dim lengthscales at [0..DIM-1]
) -> Float32:
    """Compute Σ_d (x_i[d] - x_j[d])^2 / l_d^2."""
    var dist_sq = Float32(0.0)
    @parameter
    for d in range(DIM):
        var diff = x_i[d] - x_j[d]
        var l_d = params_ptr[d]
        dist_sq += (diff * diff) / (l_d * l_d)
    return dist_sq


# Helper: Compute weighted Euclidean distance for ARD kernels
@always_inline
fn compute_ard_dist[DIM: Int](
    x_i: InlineArray[Float32, DIM],
    x_j: InlineArray[Float32, DIM],
    params_ptr: UnsafePointer[Float32],
) -> Float32:
    """Compute sqrt(Σ_d (x_i[d] - x_j[d])^2 / l_d^2)."""
    var d_sq = compute_ard_dist_sq[DIM](x_i, x_j, params_ptr)
    var eps = Float32(1e-10)
    return sqrt(d_sq) if d_sq > eps else sqrt(eps)


struct RBFComposableARD[D: Int](ComposableKernel):
    """RBF kernel with per-dimension lengthscales (ARD).
    
    k(x, x') = os * exp(-0.5 * Σ_d (x_d - x'_d)^2 / l_d^2)
    
    Parameters: [l_0, l_1, ..., l_{D-1}, outputscale]
    num_params() = D + 1
    
    Gradients:
    - param_idx < D:  dk/dl_d = k * (x_d - x'_d)^2 / l_d^3
    - param_idx == D:  dk/d(os) = k / os
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var outputscale = params_ptr[D]
        var ard_dist_sq = compute_ard_dist_sq[DIM](x_i, x_j, params_ptr)
        return outputscale * math_exp(Float32(-0.5) * ard_dist_sq)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var outputscale = params_ptr[D]
        var ard_dist_sq = compute_ard_dist_sq[DIM](x_i, x_j, params_ptr)
        var k_val = outputscale * math_exp(Float32(-0.5) * ard_dist_sq)
        
        if param_idx < D:
            # dk/dl_d = k * (x_d - x'_d)^2 / l_d^3
            var diff = x_i[param_idx] - x_j[param_idx]
            var l_d = params_ptr[param_idx]
            return k_val * (diff * diff) / (l_d * l_d * l_d)
        else:
            # dk/d(outputscale) = k / outputscale
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute ard_dist_sq and k_val once, write all D+1 gradients."""
        var outputscale = params_ptr[D]
        var ard_dist_sq = compute_ard_dist_sq[DIM](x_i, x_j, params_ptr)
        var k_val = outputscale * math_exp(Float32(-0.5) * ard_dist_sq)
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var l_d = params_ptr[d]
            grads_out[d] = k_val * (diff * diff) / (l_d * l_d * l_d)
        grads_out[D] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return D + 1


struct Matern12ComposableARD[D: Int](ComposableKernel):
    """Matern 1/2 kernel with ARD.
    
    k(x, x') = os * exp(-r_ard)
    where r_ard = sqrt(Σ_d (x_d - x'_d)^2 / l_d^2)
    
    Parameters: [l_0, ..., l_{D-1}, outputscale]
    num_params() = D + 1
    
    Gradients:
    - param_idx < D:  dk/dl_d = k * (x_d - x'_d)^2 / (l_d^3 * r_ard)
    - param_idx == D:  dk/d(os) = k / os
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        return outputscale * math_exp(-r)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var k_val = outputscale * math_exp(-r)
        
        if param_idx < D:
            # dk/dl_d = k * (x_d - x'_d)^2 / (l_d^3 * r)
            var diff = x_i[param_idx] - x_j[param_idx]
            var l_d = params_ptr[param_idx]
            var eps = Float32(1e-10)
            var safe_r = r if r > eps else eps
            return k_val * (diff * diff) / (l_d * l_d * l_d * safe_r)
        else:
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute r and k_val once, write all D+1 gradients."""
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var k_val = outputscale * math_exp(-r)
        var eps = Float32(1e-10)
        var safe_r = r if r > eps else eps
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var l_d = params_ptr[d]
            grads_out[d] = k_val * (diff * diff) / (l_d * l_d * l_d * safe_r)
        grads_out[D] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return D + 1


struct Matern32ComposableARD[D: Int](ComposableKernel):
    """Matern 3/2 kernel with ARD.
    
    k(x, x') = os * (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
    where r = sqrt(Σ_d (x_d - x'_d)^2 / l_d^2)
    
    Parameters: [l_0, ..., l_{D-1}, outputscale]
    num_params() = D + 1
    
    Gradients:
    - param_idx < D:  dk/dl_d = os * 3 * (x_d - x'_d)^2 / (l_d^3) * exp(-sqrt(3)*r) / r
    - param_idx == D:  dk/d(os) = k / os
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var sqrt3_r = SQRT3 * r
        return outputscale * (Float32(1.0) + sqrt3_r) * math_exp(-sqrt3_r)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var sqrt3_r = SQRT3 * r
        var k_val = outputscale * (Float32(1.0) + sqrt3_r) * math_exp(-sqrt3_r)
        
        if param_idx < D:
            # dk/dl_d = os * 3 * (x_d-x'd)^2 / l_d^3 * exp(-sqrt3*r)
            # (chain rule: dk/dr * dr/dl_d, the r factors cancel for Matern 3/2)
            var diff = x_i[param_idx] - x_j[param_idx]
            var l_d = params_ptr[param_idx]
            return outputscale * Float32(3.0) * (diff * diff) / (l_d * l_d * l_d) * math_exp(-sqrt3_r)
        else:
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute r, exp, k_val once, write all D+1 gradients."""
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var sqrt3_r = SQRT3 * r
        var exp_val = math_exp(-sqrt3_r)
        var k_val = outputscale * (Float32(1.0) + sqrt3_r) * exp_val
        var coeff = outputscale * Float32(3.0) * exp_val
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var l_d = params_ptr[d]
            grads_out[d] = coeff * (diff * diff) / (l_d * l_d * l_d)
        grads_out[D] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return D + 1


struct Matern52ComposableARD[D: Int](ComposableKernel):
    """Matern 5/2 kernel with ARD.
    
    k(x, x') = os * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    where r = sqrt(Σ_d (x_d - x'_d)^2 / l_d^2)
    
    Parameters: [l_0, ..., l_{D-1}, outputscale]
    num_params() = D + 1
    
    Gradients:
    - param_idx < D:  dk/dl_d = os * (5/3) * (x_d-x'd)^2 / (l_d^3) * (1+sqrt5*r) * exp(-sqrt5*r) / r
    - param_idx == D:  dk/d(os) = k / os
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var sqrt5_r = SQRT5 * r
        return outputscale * (Float32(1.0) + sqrt5_r + Float32(1.6666667) * r * r) * math_exp(-sqrt5_r)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var sqrt5_r = SQRT5 * r
        var k_val = outputscale * (Float32(1.0) + sqrt5_r + Float32(1.6666667) * r * r) * math_exp(-sqrt5_r)
        
        if param_idx < D:
            # dk/dl_d = os * 5/3 * (x_d-x'd)^2 / l_d^3 * (1+sqrt5*r) * exp(-sqrt5*r)
            # (chain rule: dk/dr * dr/dl_d, the r factors cancel for Matern 5/2)
            var diff = x_i[param_idx] - x_j[param_idx]
            var l_d = params_ptr[param_idx]
            return outputscale * Float32(1.6666667) * (diff * diff) / (l_d * l_d * l_d) * (Float32(1.0) + sqrt5_r) * math_exp(-sqrt5_r)
        else:
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute r, exp, k_val once, write all D+1 gradients."""
        var outputscale = params_ptr[D]
        var r = compute_ard_dist[DIM](x_i, x_j, params_ptr)
        var sqrt5_r = SQRT5 * r
        var exp_val = math_exp(-sqrt5_r)
        var k_val = outputscale * (Float32(1.0) + sqrt5_r + Float32(1.6666667) * r * r) * exp_val
        var coeff = outputscale * Float32(1.6666667) * (Float32(1.0) + sqrt5_r) * exp_val
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var l_d = params_ptr[d]
            grads_out[d] = coeff * (diff * diff) / (l_d * l_d * l_d)
        grads_out[D] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return D + 1


struct PeriodicComposableARD[D: Int](ComposableKernel):
    """Periodic kernel with ARD.
    
    k(x, x') = os * exp(-2 * Σ_d sin²(π|x_d - x'_d|/period) / l_d)
    
    Note: period is shared across dimensions (standard GPyTorch convention).
    
    Parameters: [l_0, ..., l_{D-1}, period, outputscale]
    num_params() = D + 2
    
    Gradients:
    - param_idx < D:     dk/dl_d = k * 2 * sin²(π*diff_d/p) / l_d²
    - param_idx == D:    dk/d(period)
    - param_idx == D+1:  dk/d(os) = k / os
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        alias EPS = Float32(1e-6)
        var period = max(params_ptr[D], EPS)
        var outputscale = params_ptr[D + 1]
        
        var exp_arg = Float32(0.0)
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var sin_val = sin(PI * diff / period)
            var l_d = max(params_ptr[d], EPS)
            exp_arg += sin_val * sin_val / l_d
        
        return outputscale * math_exp(Float32(-2.0) * exp_arg)
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        alias EPS = Float32(1e-6)
        var period = max(params_ptr[D], EPS)
        var outputscale = params_ptr[D + 1]
        
        # Precompute per-dimension sin values and exp_arg
        var exp_arg = Float32(0.0)
        var sin_vals: InlineArray[Float32, DIM] = InlineArray[Float32, DIM](unsafe_uninitialized=True)
        var cos_vals: InlineArray[Float32, DIM] = InlineArray[Float32, DIM](unsafe_uninitialized=True)
        var diffs: InlineArray[Float32, DIM] = InlineArray[Float32, DIM](unsafe_uninitialized=True)
        
        @parameter
        for d in range(DIM):
            diffs[d] = x_i[d] - x_j[d]
            var u = PI * diffs[d] / period
            sin_vals[d] = sin(u)
            cos_vals[d] = cos(u)
            var l_d = max(params_ptr[d], EPS)
            exp_arg += sin_vals[d] * sin_vals[d] / l_d
        
        var k_val = outputscale * math_exp(Float32(-2.0) * exp_arg)
        
        if param_idx < D:
            # dk/dl_d = k * 2 * sin²(π*diff_d/p) / l_d²
            var l_d = max(params_ptr[param_idx], EPS)
            return k_val * Float32(2.0) * sin_vals[param_idx] * sin_vals[param_idx] / (l_d * l_d)
        elif param_idx == D:
            # dk/d(period) = k * (4π / (period²)) * Σ_d diff_d * sin_d * cos_d / l_d
            var deriv = Float32(0.0)
            @parameter
            for d in range(DIM):
                var l_d = max(params_ptr[d], EPS)
                deriv += diffs[d] * sin_vals[d] * cos_vals[d] / l_d
            return k_val * Float32(4.0) * PI * deriv / (period * period)
        else:
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute sin/cos/exp once, write all D+2 gradients."""
        alias EPS = Float32(1e-6)
        var period = max(params_ptr[D], EPS)
        var outputscale = params_ptr[D + 1]
        
        var exp_arg = Float32(0.0)
        var period_deriv = Float32(0.0)
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var u = PI * diff / period
            var sin_val = sin(u)
            var cos_val = cos(u)
            var l_d = max(params_ptr[d], EPS)
            var sin_sq = sin_val * sin_val
            exp_arg += sin_sq / l_d
            # Per-dim lengthscale gradient: dk/dl_d = k * 2 * sin^2 / l_d^2
            grads_out[d] = Float32(2.0) * sin_sq / (l_d * l_d)
            # Accumulate period derivative
            period_deriv += diff * sin_val * cos_val / l_d
        
        var k_val = outputscale * math_exp(Float32(-2.0) * exp_arg)
        
        # Scale per-dim gradients by k_val
        @parameter
        for d in range(DIM):
            grads_out[d] = grads_out[d] * k_val
        
        # Period gradient
        grads_out[D] = k_val * Float32(4.0) * PI * period_deriv / (period * period)
        # Outputscale gradient
        grads_out[D + 1] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return D + 2


struct RQComposableARD[D: Int](ComposableKernel):
    """Rational Quadratic kernel with ARD.
    
    k(x, x') = os * (1 + r_ard² / (2*alpha))^(-alpha)
    where r_ard² = Σ_d (x_d - x'_d)^2 / l_d^2
    
    Parameters: [l_0, ..., l_{D-1}, alpha, outputscale]
    num_params() = D + 2
    
    Gradients:
    - param_idx < D:     dk/dl_d = k * (x_d-x'd)^2 / (l_d^3 * base)
    - param_idx == D:    dk/d(alpha) = k * (r_ard²/(2*alpha*base) - log(base))
    - param_idx == D+1:  dk/d(os) = k / os
    """
    
    @staticmethod
    fn evaluate[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
    ) -> Float32:
        var alpha = params_ptr[D]
        var outputscale = params_ptr[D + 1]
        
        var ard_dist_sq = compute_ard_dist_sq[DIM](x_i, x_j, params_ptr)
        var base = Float32(1.0) + ard_dist_sq / (Float32(2.0) * alpha)
        return outputscale * (base ** (-alpha))
    
    @staticmethod
    fn gradient[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        param_idx: Int,
    ) -> Float32:
        var alpha = params_ptr[D]
        var outputscale = params_ptr[D + 1]
        
        var ard_dist_sq = compute_ard_dist_sq[DIM](x_i, x_j, params_ptr)
        var base = Float32(1.0) + ard_dist_sq / (Float32(2.0) * alpha)
        var k_val = outputscale * (base ** (-alpha))
        
        if param_idx < D:
            # dk/dl_d = k * alpha * (x_d-x'd)^2 / (l_d^3 * base * alpha) 
            #         = k * (x_d-x'd)^2 / (l_d^3 * base)
            # Actually: chain rule gives dk/dl_d = k * alpha * 2*(x_d-x'd)^2 / (2*alpha*l_d^3*base)
            #                                    = k * (x_d-x'd)^2 / (l_d^3 * base)
            var diff = x_i[param_idx] - x_j[param_idx]
            var l_d = params_ptr[param_idx]
            return k_val * (diff * diff) / (l_d * l_d * l_d * base)
        elif param_idx == D:
            # dk/d(alpha) = k * (ard_dist_sq/(2*alpha*base) - log(base))
            var term1 = ard_dist_sq / (Float32(2.0) * alpha * base)
            var term2 = log(base)
            return k_val * (term1 - term2)
        else:
            return k_val / outputscale
    
    @staticmethod
    fn all_gradients[DIM: Int](
        x_i: InlineArray[Float32, DIM],
        x_j: InlineArray[Float32, DIM],
        params_ptr: UnsafePointer[Float32],
        grads_out: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Fused: compute ard_dist_sq, base, k_val once, write all D+2 gradients."""
        var alpha = params_ptr[D]
        var outputscale = params_ptr[D + 1]
        var ard_dist_sq = compute_ard_dist_sq[DIM](x_i, x_j, params_ptr)
        var base = Float32(1.0) + ard_dist_sq / (Float32(2.0) * alpha)
        var k_val = outputscale * (base ** (-alpha))
        
        @parameter
        for d in range(DIM):
            var diff = x_i[d] - x_j[d]
            var l_d = params_ptr[d]
            grads_out[d] = k_val * (diff * diff) / (l_d * l_d * l_d * base)
        
        # Alpha gradient
        var inv_2alpha = Float32(1.0) / (Float32(2.0) * alpha)
        grads_out[D] = k_val * (ard_dist_sq * inv_2alpha / base - log(base))
        # Outputscale gradient
        grads_out[D + 1] = k_val / outputscale
    
    @staticmethod
    fn num_params() -> Int:
        return D + 2


# =============================================================================
# Pre-defined Composite Kernel Aliases
# =============================================================================

# Locally periodic: RBF envelope with periodic component
alias LocallyPeriodicKernel = ProductKernel[RBFComposable, PeriodicComposable]

# RBF plus linear trend
alias RBFPlusLinearKernel = SumKernel[RBFComposable, LinearComposable]

# Multi-scale RBF (sum of two RBFs with different lengthscales)
alias MultiScaleRBFKernel = SumKernel[RBFComposable, RBFComposable]

# Locally periodic plus linear trend
alias LocallyPeriodicPlusLinearKernel = SumKernel[ProductKernel[RBFComposable, PeriodicComposable], LinearComposable]

# Matern52 plus linear (common for time series with trend)
alias Matern52PlusLinearKernel = SumKernel[Matern52Composable, LinearComposable]
