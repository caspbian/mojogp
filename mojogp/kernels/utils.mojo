"""Utility functions for MojoGP training.

Provides parameter transformations (softplus, sigmoid), array utilities, and helper functions.
"""

from math import exp as math_exp, log, sqrt
from memory import UnsafePointer


# =============================================================================
# Parameter Transformations
# =============================================================================

fn softplus(x: Float32) -> Float32:
    """Apply softplus transformation: softplus(x) = log(1 + exp(x)).
    
    Ensures parameters are positive. Uses numerically stable implementation
    with clipping for extreme values.
    
    Args:
        x: Raw (unconstrained) parameter value
        
    Returns:
        Positive transformed value
        
    Note:
        - For x > 20: softplus(x) ≈ x (avoids overflow)
        - For x < -20: softplus(x) ≈ exp(x) (avoids underflow)
        - Otherwise: exact formula
    """
    if x > Float32(20.0):
        return x
    elif x < Float32(-20.0):
        return math_exp(x)
    else:
        return log(Float32(1.0) + math_exp(x))


fn inv_softplus(y: Float32) -> Float32:
    """Inverse softplus transformation: inv_softplus(y) = log(exp(y) - 1).
    
    Converts positive constrained parameter back to unconstrained space.
    
    Args:
        y: Positive constrained parameter value
        
    Returns:
        Unconstrained raw parameter value
        
    Note:
        - For y > 20: inv_softplus(y) ≈ y
        - For y < 1e-6: inv_softplus(y) ≈ log(y)
        - Otherwise: exact formula
    """
    if y > Float32(20.0):
        return y
    elif y < Float32(1e-6):
        return log(y)
    else:
        return log(math_exp(y) - Float32(1.0))


fn softplus_derivative(x: Float32) -> Float32:
    """Derivative of softplus: d/dx softplus(x) = 1 / (1 + exp(-x)).
    
    Used for chain rule when computing gradients in raw parameter space.
    
    Args:
        x: Raw (unconstrained) parameter value
        
    Returns:
        Derivative value (always in (0, 1))
        
    Note:
        - For x > 20: derivative ≈ 1
        - For x < -20: derivative ≈ exp(x)
        - Otherwise: exact formula (sigmoid function)
    """
    if x > Float32(20.0):
        return Float32(1.0)
    elif x < Float32(-20.0):
        return math_exp(x)
    else:
        return Float32(1.0) / (Float32(1.0) + math_exp(-x))


# =============================================================================
# Sigmoid Transformations (for angle parameters in [0, pi])
# =============================================================================

fn sigmoid(x: Float32) -> Float32:
    """Sigmoid function: sigma(x) = 1 / (1 + exp(-x)).

    Maps unconstrained x to (0, 1). Combined with pi scaling, maps to (0, pi)
    for angle parameters in EHH/HH/FE categorical kernels.

    Note: This is numerically identical to softplus_derivative(x).
    """
    if x > Float32(20.0):
        return Float32(1.0)
    elif x < Float32(-20.0):
        return math_exp(x)
    else:
        return Float32(1.0) / (Float32(1.0) + math_exp(-x))


fn inv_sigmoid(y: Float32) -> Float32:
    """Inverse sigmoid (logit): inv_sigmoid(y) = log(y / (1 - y)).

    Converts from (0, 1) back to unconstrained space.
    For angle params: raw = inv_sigmoid(angle / pi).

    Args:
        y: Value in (0, 1)

    Returns:
        Unconstrained value
    """
    # Clamp to avoid log(0) or division by zero
    var y_safe = y
    if y_safe < Float32(1e-7):
        y_safe = Float32(1e-7)
    elif y_safe > Float32(1.0) - Float32(1e-7):
        y_safe = Float32(1.0) - Float32(1e-7)
    return log(y_safe / (Float32(1.0) - y_safe))


fn sigmoid_derivative(x: Float32) -> Float32:
    """Derivative of sigmoid: sigma(x) * (1 - sigma(x)).

    Used for chain rule when angle params use sigmoid * pi mapping.
    The full chain rule for angle params is: sigmoid_derivative(raw) * pi.
    """
    var s = sigmoid(x)
    return s * (Float32(1.0) - s)


# =============================================================================
# Array Utilities
# =============================================================================

fn fill_array(ptr: UnsafePointer[Float32, MutAnyOrigin], value: Float32, size: Int):
    """Fill array with a constant value.
    
    Args:
        ptr: Pointer to array
        value: Value to fill with
        size: Number of elements
    """
    for i in range(size):
        ptr[i] = value


fn copy_array(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int
):
    """Copy array from source to destination.
    
    Args:
        src: Source array pointer
        dst: Destination array pointer
        size: Number of elements to copy
    """
    for i in range(size):
        dst[i] = src[i]


fn dot_product(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    size: Int
) -> Float32:
    """Compute dot product of two arrays.
    
    Args:
        a: First array pointer
        b: Second array pointer
        size: Number of elements
        
    Returns:
        Dot product a^T @ b
    """
    var result = Float32(0.0)
    for i in range(size):
        result += a[i] * b[i]
    return result


fn array_norm(ptr: UnsafePointer[Float32, MutAnyOrigin], size: Int) -> Float32:
    """Compute L2 norm of array.
    
    Args:
        ptr: Array pointer
        size: Number of elements
        
    Returns:
        L2 norm ||x||_2
    """
    var sum_sq = Float32(0.0)
    for i in range(size):
        sum_sq += ptr[i] * ptr[i]
    return sqrt(sum_sq)


# =============================================================================
# Parameter Initialization Helpers
# =============================================================================

fn compute_data_mean(
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    d: Int
) -> Float32:
    """Compute mean of all data values (for initialization).
    
    Args:
        x_ptr: Data pointer [n, d] in row-major layout
        n: Number of points
        d: Dimensionality
        
    Returns:
        Mean value across all elements
    """
    var sum = Float32(0.0)
    for i in range(n * d):
        sum += x_ptr[i]
    return sum / Float32(n * d)


fn compute_data_std(
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    d: Int,
    mean: Float32
) -> Float32:
    """Compute standard deviation of all data values.
    
    Args:
        x_ptr: Data pointer [n, d] in row-major layout
        n: Number of points
        d: Dimensionality
        mean: Pre-computed mean
        
    Returns:
        Standard deviation across all elements
    """
    var sum_sq = Float32(0.0)
    for i in range(n * d):
        var diff = x_ptr[i] - mean
        sum_sq += diff * diff
    return sqrt(sum_sq / Float32(n * d))


fn compute_target_variance(
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int
) -> Float32:
    """Compute variance of target values.
    
    Args:
        y_ptr: Target pointer [n]
        n: Number of points
        
    Returns:
        Variance of targets
    """
    # Compute mean
    var mean = Float32(0.0)
    for i in range(n):
        mean += y_ptr[i]
    mean /= Float32(n)
    
    # Compute variance
    var var_sum = Float32(0.0)
    for i in range(n):
        var diff = y_ptr[i] - mean
        var_sum += diff * diff
    
    return var_sum / Float32(n)


# =============================================================================
# Numerical Utilities
# =============================================================================

fn clip_value(value: Float32, min_val: Float32, max_val: Float32) -> Float32:
    """Clip value to range [min_val, max_val].
    
    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clipped value
    """
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value


fn is_finite(value: Float32) -> Bool:
    """Check if value is finite (not NaN or infinity).
    
    Args:
        value: Value to check
        
    Returns:
        True if finite, False otherwise
    """
    # A value is finite if it equals itself and is not infinite
    # NaN != NaN, so this catches NaN
    # For infinity, we check against a large threshold
    if value != value:  # NaN check
        return False
    if value > Float32(1e30) or value < Float32(-1e30):  # Infinity check
        return False
    return True


fn safe_divide(numerator: Float32, denominator: Float32, default: Float32 = 0.0) -> Float32:
    """Safely divide two numbers, returning default if denominator is too small.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if division is unsafe
        
    Returns:
        numerator / denominator if safe, otherwise default
    """
    if abs(denominator) < Float32(1e-10):
        return default
    return numerator / denominator
