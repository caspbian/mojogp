"""Utility functions for GP training.

Provides helper functions for Adam optimizer updates, gradient clipping,
and GPU warmup used by the main training loops.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt

from .training_types import (
    float_dtype,
    AdamState,
    AdamStateARD,
    AdamStateGeneric,
    AdamUpdateResult,
    AdamUpdateResultARD,
    AdamUpdateResultGeneric,
)
from .utils import softplus, inv_softplus, softplus_derivative


fn compute_cosine_lr(lr_max: Float32, iteration: Int, total_iterations: Int) -> Float32:
    """Compute cosine annealing learning rate.

    Implements the cosine annealing schedule:
        lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

    where lr_min = lr_max / 20. This smoothly decays the learning rate from
    lr_max to lr_min over total_iterations, following a cosine curve.

    The schedule starts at lr_max (iteration=0) and ends near lr_min
    (iteration=total_iterations). This is useful for GP training where
    large initial steps explore the parameter space, then smaller steps
    refine near the optimum.

    This function is designed to be called from the Python training layer,
    which passes the scheduled LR to Mojo training functions. No Mojo
    training function signatures need to change.

    Args:
        lr_max: Maximum (initial) learning rate.
        iteration: Current training iteration (0-indexed).
        total_iterations: Total number of training iterations.

    Returns:
        Scheduled learning rate for the given iteration.
    """
    from math import cos

    var lr_min = lr_max / Float32(20.0)
    alias pi = Float32(3.14159265358979)
    var t_frac = Float32(iteration) / Float32(max(total_iterations, 1))
    var cosine_val = cos(pi * t_frac)
    return lr_min + Float32(0.5) * (lr_max - lr_min) * (Float32(1.0) + cosine_val)


fn compute_dynamic_cg_tol(base_tol: Float32, iteration: Int, total_iterations: Int) -> Float32:
    """Compute dynamic CG tolerance: tighter early, looser late.

    Early training iterations benefit from more accurate CG solves because
    gradients are large and direction matters. Late iterations are near
    convergence, so a looser tolerance saves CG iterations without hurting
    final accuracy.

    Schedule:
        - iteration 0:              base_tol * 0.5  (2x tighter)
        - iteration total/2:        base_tol * 1.25 (slightly looser)
        - iteration total:          base_tol * 2.0  (2x looser)

    The scale interpolates linearly from 0.5 to 2.0 over training.

    This is a pure utility function. Training loops can optionally call it
    to compute a per-iteration CG tolerance without changing any function
    signatures.

    Args:
        base_tol: Base CG tolerance (e.g. 1e-2).
        iteration: Current training iteration (0-indexed).
        total_iterations: Total number of training iterations planned.

    Returns:
        Scaled CG tolerance for this iteration.
    """
    var progress = Float32(iteration) / Float32(max(total_iterations, 1))
    # Linear interpolation: scale goes from 0.5 (tight) to 2.0 (loose)
    var scale = Float32(0.5) + Float32(1.5) * progress
    return base_tol * scale


# Helper for pow since it's not in math module
fn pow_float32(base: Float32, exp: Int) -> Float32:
    """Compute base^exp for integer exponent using exponentiation by squaring.

    O(log exp) multiplications instead of O(exp).
    """
    if exp == 0:
        return Float32(1.0)
    if exp == 1:
        return base
    var result = Float32(1.0)
    var b = base
    var e = exp
    while e > 0:
        if e & 1 == 1:
            result *= b
        b *= b
        e >>= 1
    return result


# Helper for gradient clipping and NaN protection
fn clip_gradient(grad: Float32, max_grad: Float32 = 100.0) -> Float32:
    """Clip gradient to prevent explosion and handle NaN/Inf.
    
    Args:
        grad: Gradient value to clip
        max_grad: Maximum absolute gradient value (default 100.0, increased from 10.0
                  to allow larger gradient steps while still preventing explosion)
        
    Returns:
        Clipped gradient, or 0.0 if NaN/Inf
    """
    # NaN check: NaN != NaN
    if grad != grad:
        return Float32(0.0)
    
    # Inf check
    if grad > Float32(1e30) or grad < Float32(-1e30):
        return Float32(0.0)
    
    # Clip to [-max_grad, max_grad]
    if grad > max_grad:
        return max_grad
    if grad < -max_grad:
        return -max_grad
    
    return grad


fn adam_update_generic(
    read state: AdamStateGeneric,
    read gradients: List[Float32],  # [N+1] gradients (N kernel + noise)
    read raw_params: List[Float32],  # [N+1] raw parameters
    learning_rate: Float32 = 0.01,
    beta1: Float32 = 0.9,
    beta2: Float32 = 0.999,
    eps: Float32 = 1e-8
) -> AdamUpdateResultGeneric:
    """Perform Adam optimizer update step for N+1 parameters.
    
    Updates raw (unconstrained) parameters using Adam algorithm.
    All parameters go through softplus transform.
    
    Args:
        state: Adam optimizer state
        gradients: Gradients w.r.t. each parameter [N+1] (constrained space)
        raw_params: Raw parameter values [N+1]
        learning_rate: Adam learning rate
        beta1: Adam first moment decay
        beta2: Adam second moment decay
        eps: Adam epsilon for numerical stability
        
    Returns:
        AdamUpdateResultGeneric with updated state and raw parameters
    """
    var num_params = state.num_params
    
    # Create copies to modify
    var new_m = List[Float32]()
    var new_v = List[Float32]()
    var new_raw_params = List[Float32]()
    
    # Increment time step
    var t = Float32(state.t + 1)
    
    # Clamp bounds for raw parameters
    alias MAX_RAW = Float32(20.0)
    alias MIN_RAW = Float32(-20.0)
    
    # Update each parameter
    for p in range(num_params):
        # Clip gradient to prevent NaN/Inf propagation
        var clipped_grad = clip_gradient(gradients[p])
        
        # Chain rule: convert gradient to raw parameter space
        var grad_raw = clipped_grad * softplus_derivative(raw_params[p])
        
        # Update moments
        var m = beta1 * state.m[p] + (Float32(1.0) - beta1) * grad_raw
        var v = beta2 * state.v[p] + (Float32(1.0) - beta2) * grad_raw * grad_raw
        new_m.append(m)
        new_v.append(v)
        
        # Bias correction
        var m_hat = m / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat = v / (Float32(1.0) - pow_float32(beta2, Int(t)))
        
        # Update parameter with clamping
        var new_raw = raw_params[p] - learning_rate * m_hat / (sqrt(v_hat) + eps)
        new_raw = max(MIN_RAW, min(MAX_RAW, new_raw))
        new_raw_params.append(new_raw)
    
    # Create new state
    var new_state = AdamStateGeneric(num_params - 1)  # -1 because constructor adds 1 for noise
    new_state.m = new_m^
    new_state.v = new_v^
    new_state.t = Int(t)
    new_state.num_params = num_params
    
    return AdamUpdateResultGeneric(new_state^, new_raw_params^)


fn adam_update_state_inplace(
    var state: AdamStateGeneric,
    read gradients: List[Float32],
    read raw_params: List[Float32],
    learning_rate: Float32 = 0.01,
    beta1: Float32 = 0.9,
    beta2: Float32 = 0.999,
) -> AdamStateGeneric:
    """Update Adam state in-place and return it.
    
    This is a helper to avoid ownership issues with AdamUpdateResultGeneric.
    """
    var num_params = state.num_params
    var t = Float32(state.t + 1)
    
    # Update moments in place
    for p in range(num_params):
        var clipped_grad = clip_gradient(gradients[p])
        var grad_raw = clipped_grad * softplus_derivative(raw_params[p])
        
        state.m[p] = beta1 * state.m[p] + (Float32(1.0) - beta1) * grad_raw
        state.v[p] = beta2 * state.v[p] + (Float32(1.0) - beta2) * grad_raw * grad_raw
    
    state.t = Int(t)
    return state^


fn adam_update_state_inplace_custom(
    var state: AdamStateGeneric,
    read gradients: List[Float32],
    read raw_params: List[Float32],
    read chain_rule_deriv: List[Float32],
    learning_rate: Float32 = 0.01,
    beta1: Float32 = 0.9,
    beta2: Float32 = 0.999,
) -> AdamStateGeneric:
    """Update Adam state using pre-computed per-parameter chain rule derivatives.

    This variant accepts a list of chain rule derivatives instead of applying
    softplus_derivative uniformly. Needed for mixed categorical kernels where
    angle parameters use sigmoid*pi (derivative = sigmoid_derivative*pi) while
    distance parameters use softplus.

    Args:
        state: Current Adam state (consumed via transfer)
        gradients: Gradients in constrained parameter space
        raw_params: Current raw (unconstrained) parameter values
        chain_rule_deriv: Pre-computed d(constrained)/d(raw) for each parameter
        learning_rate: Learning rate (unused but kept for API consistency)
        beta1: First moment decay rate
        beta2: Second moment decay rate

    Returns:
        Updated Adam state
    """
    var num_params = state.num_params
    var t = Float32(state.t + 1)

    for p in range(num_params):
        var clipped_grad = clip_gradient(gradients[p])
        var grad_raw = clipped_grad * chain_rule_deriv[p]

        state.m[p] = beta1 * state.m[p] + (Float32(1.0) - beta1) * grad_raw
        state.v[p] = beta2 * state.v[p] + (Float32(1.0) - beta2) * grad_raw * grad_raw

    state.t = Int(t)
    return state^


fn adam_update_params(
    read state: AdamStateGeneric,
    read gradients: List[Float32],
    read raw_params: List[Float32],
    learning_rate: Float32 = 0.01,
    beta1: Float32 = 0.9,
    beta2: Float32 = 0.999,
    eps: Float32 = 1e-8
) -> List[Float32]:
    """Compute updated parameters using Adam.
    
    This is a helper to avoid ownership issues with AdamUpdateResultGeneric.
    Uses the ALREADY UPDATED state (after adam_update_state_inplace).
    """
    var num_params = state.num_params
    var t = Float32(state.t)  # Already incremented by adam_update_state_inplace
    
    alias MAX_RAW = Float32(20.0)
    alias MIN_RAW = Float32(-20.0)
    
    var new_raw_params = List[Float32]()
    
    for p in range(num_params):
        # Bias correction using updated moments
        var m_hat = state.m[p] / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat = state.v[p] / (Float32(1.0) - pow_float32(beta2, Int(t)))
        
        # Update parameter with clamping
        var new_raw = raw_params[p] - learning_rate * m_hat / (sqrt(v_hat) + eps)
        new_raw = max(MIN_RAW, min(MAX_RAW, new_raw))
        new_raw_params.append(new_raw)
    
    return new_raw_params^


fn adam_update(
    state: AdamState,
    grad_ls: Float32,
    grad_noise: Float32,
    grad_os: Float32,
    raw_ls: Float32,
    raw_noise: Float32,
    raw_os: Float32,
    learning_rate: Float32 = 0.01,
    beta1: Float32 = 0.9,
    beta2: Float32 = 0.999,
    eps: Float32 = 1e-8,
    grad_param1: Float32 = Float32(0.0),
    raw_param1: Float32 = Float32(0.0),
    has_param1: Bool = False,
    grad_param2: Float32 = Float32(0.0),
    raw_param2: Float32 = Float32(0.0),
    has_param2: Bool = False,
) -> AdamUpdateResult:
    """Perform Adam optimizer update step.
    
    Updates raw (unconstrained) parameters using Adam algorithm.
    
    Args:
        state: Adam optimizer state (will be copied and updated)
        grad_ls: Gradient w.r.t. lengthscale (constrained space)
        grad_noise: Gradient w.r.t. noise (constrained space)
        grad_os: Gradient w.r.t. output scale (constrained space)
        raw_ls: Raw lengthscale parameter
        raw_noise: Raw noise parameter
        raw_os: Raw output scale parameter
        learning_rate: Adam learning rate
        beta1: Adam first moment decay
        beta2: Adam second moment decay
        eps: Adam epsilon for numerical stability
        grad_param1: Gradient w.r.t. param1 (period/alpha) - only used for periodic/RQ/linear
        raw_param1: Raw param1 parameter - only used for periodic/RQ/linear
        has_param1: Whether to optimize param1
        grad_param2: Gradient w.r.t. param2 (offset) - only used for polynomial
        raw_param2: Raw param2 parameter - only used for polynomial
        has_param2: Whether to optimize param2
        
    Returns:
        AdamUpdateResult with updated state and raw parameters
    """
    # Create a copy of state to modify
    var new_state = state
    
    # Increment time step
    new_state.t += 1
    var t = Float32(new_state.t)
    
    # Chain rule: convert gradients to raw parameter space
    var grad_raw_ls = grad_ls * softplus_derivative(raw_ls)
    var grad_raw_noise = grad_noise * softplus_derivative(raw_noise)
    var grad_raw_os = grad_os * softplus_derivative(raw_os)
    
    # Update lengthscale
    new_state.m_ls = beta1 * new_state.m_ls + (Float32(1.0) - beta1) * grad_raw_ls
    new_state.v_ls = beta2 * new_state.v_ls + (Float32(1.0) - beta2) * grad_raw_ls * grad_raw_ls
    var m_hat_ls = new_state.m_ls / (Float32(1.0) - pow_float32(beta1, Int(t)))
    var v_hat_ls = new_state.v_ls / (Float32(1.0) - pow_float32(beta2, Int(t)))
    var new_raw_ls = raw_ls - learning_rate * m_hat_ls / (sqrt(v_hat_ls) + eps)
    
    # Update noise
    new_state.m_noise = beta1 * new_state.m_noise + (Float32(1.0) - beta1) * grad_raw_noise
    new_state.v_noise = beta2 * new_state.v_noise + (Float32(1.0) - beta2) * grad_raw_noise * grad_raw_noise
    var m_hat_noise = new_state.m_noise / (Float32(1.0) - pow_float32(beta1, Int(t)))
    var v_hat_noise = new_state.v_noise / (Float32(1.0) - pow_float32(beta2, Int(t)))
    var new_raw_noise = raw_noise - learning_rate * m_hat_noise / (sqrt(v_hat_noise) + eps)
    
    # Update output scale
    new_state.m_os = beta1 * new_state.m_os + (Float32(1.0) - beta1) * grad_raw_os
    new_state.v_os = beta2 * new_state.v_os + (Float32(1.0) - beta2) * grad_raw_os * grad_raw_os
    var m_hat_os = new_state.m_os / (Float32(1.0) - pow_float32(beta1, Int(t)))
    var v_hat_os = new_state.v_os / (Float32(1.0) - pow_float32(beta2, Int(t)))
    var new_raw_os = raw_os - learning_rate * m_hat_os / (sqrt(v_hat_os) + eps)
    
    # Update param1 (period/alpha) if applicable
    var new_raw_param1 = raw_param1
    if has_param1:
        var grad_raw_param1 = grad_param1 * softplus_derivative(raw_param1)
        new_state.m_param1 = beta1 * new_state.m_param1 + (Float32(1.0) - beta1) * grad_raw_param1
        new_state.v_param1 = beta2 * new_state.v_param1 + (Float32(1.0) - beta2) * grad_raw_param1 * grad_raw_param1
        var m_hat_param1 = new_state.m_param1 / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_param1 = new_state.v_param1 / (Float32(1.0) - pow_float32(beta2, Int(t)))
        new_raw_param1 = raw_param1 - learning_rate * m_hat_param1 / (sqrt(v_hat_param1) + eps)

    # Update param2 (offset for Polynomial) if applicable
    var new_raw_param2 = raw_param2
    if has_param2:
        var grad_raw_param2 = grad_param2 * softplus_derivative(raw_param2)
        new_state.m_param2 = beta1 * new_state.m_param2 + (Float32(1.0) - beta1) * grad_raw_param2
        new_state.v_param2 = beta2 * new_state.v_param2 + (Float32(1.0) - beta2) * grad_raw_param2 * grad_raw_param2
        var m_hat_param2 = new_state.m_param2 / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_param2 = new_state.v_param2 / (Float32(1.0) - pow_float32(beta2, Int(t)))
        new_raw_param2 = raw_param2 - learning_rate * m_hat_param2 / (sqrt(v_hat_param2) + eps)
    
    # Clamp raw parameters to prevent overflow (softplus of 20 ≈ 20, so this is safe)
    # This prevents parameters from exploding to Float32.max
    alias MAX_RAW = Float32(20.0)
    alias MIN_RAW = Float32(-20.0)
    new_raw_ls = max(MIN_RAW, min(MAX_RAW, new_raw_ls))
    new_raw_noise = max(MIN_RAW, min(MAX_RAW, new_raw_noise))
    new_raw_os = max(MIN_RAW, min(MAX_RAW, new_raw_os))
    if has_param1:
        new_raw_param1 = max(MIN_RAW, min(MAX_RAW, new_raw_param1))
    if has_param2:
        new_raw_param2 = max(MIN_RAW, min(MAX_RAW, new_raw_param2))
    
    return AdamUpdateResult(new_state, new_raw_ls, new_raw_noise, new_raw_os, new_raw_param1, new_raw_param2)


fn adam_update_ard(
    read state: AdamStateARD,
    read grad_lengthscales: List[Float32],  # [dim]
    grad_noise: Float32,
    grad_os: Float32,
    read raw_lengthscales: List[Float32],  # [dim]
    raw_noise: Float32,
    raw_os: Float32,
    learning_rate: Float32 = 0.01,
    beta1: Float32 = 0.9,
    beta2: Float32 = 0.999,
    eps: Float32 = 1e-8
) -> AdamUpdateResultARD:
    """Perform Adam optimizer update step for ARD.
    
    Updates raw (unconstrained) parameters using Adam algorithm.
    
    Args:
        state: Adam optimizer state for ARD
        grad_lengthscales: Gradients w.r.t. each lengthscale [dim]
        grad_noise: Gradient w.r.t. noise
        grad_os: Gradient w.r.t. output scale
        raw_lengthscales: Raw lengthscale parameters [dim]
        raw_noise: Raw noise parameter
        raw_os: Raw output scale parameter
        learning_rate: Adam learning rate
        beta1: Adam first moment decay
        beta2: Adam second moment decay
        eps: Adam epsilon for numerical stability
        
    Returns:
        AdamUpdateResultARD with updated state and raw parameters
    """
    var dim = state.dim
    
    # Create copies to modify
    var new_m_ls = List[Float32]()
    var new_v_ls = List[Float32]()
    var new_raw_ls = List[Float32]()
    
    # Increment time step
    var t = Float32(state.t + 1)
    
    # Clamp bounds for raw parameters
    alias MAX_RAW = Float32(20.0)
    alias MIN_RAW = Float32(-20.0)
    
    # Update each lengthscale dimension
    for d in range(dim):
        # Clip gradient to prevent NaN/Inf propagation
        var clipped_grad = clip_gradient(grad_lengthscales[d])
        
        # Chain rule: convert gradient to raw parameter space
        var grad_raw = clipped_grad * softplus_derivative(raw_lengthscales[d])
        
        # Update moments
        var m = beta1 * state.m_ls[d] + (Float32(1.0) - beta1) * grad_raw
        var v = beta2 * state.v_ls[d] + (Float32(1.0) - beta2) * grad_raw * grad_raw
        new_m_ls.append(m)
        new_v_ls.append(v)
        
        # Bias correction
        var m_hat = m / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat = v / (Float32(1.0) - pow_float32(beta2, Int(t)))
        
        # Update parameter with clamping
        var new_raw = raw_lengthscales[d] - learning_rate * m_hat / (sqrt(v_hat) + eps)
        new_raw = max(MIN_RAW, min(MAX_RAW, new_raw))
        new_raw_ls.append(new_raw)
    
    # Update noise with gradient clipping
    var clipped_grad_noise = clip_gradient(grad_noise)
    var grad_raw_noise = clipped_grad_noise * softplus_derivative(raw_noise)
    var new_m_noise = beta1 * state.m_noise + (Float32(1.0) - beta1) * grad_raw_noise
    var new_v_noise = beta2 * state.v_noise + (Float32(1.0) - beta2) * grad_raw_noise * grad_raw_noise
    var m_hat_noise = new_m_noise / (Float32(1.0) - pow_float32(beta1, Int(t)))
    var v_hat_noise = new_v_noise / (Float32(1.0) - pow_float32(beta2, Int(t)))
    var new_raw_noise = raw_noise - learning_rate * m_hat_noise / (sqrt(v_hat_noise) + eps)
    new_raw_noise = max(MIN_RAW, min(MAX_RAW, new_raw_noise))
    
    # Update output scale with gradient clipping
    var clipped_grad_os = clip_gradient(grad_os)
    var grad_raw_os = clipped_grad_os * softplus_derivative(raw_os)
    var new_m_os = beta1 * state.m_os + (Float32(1.0) - beta1) * grad_raw_os
    var new_v_os = beta2 * state.v_os + (Float32(1.0) - beta2) * grad_raw_os * grad_raw_os
    var m_hat_os = new_m_os / (Float32(1.0) - pow_float32(beta1, Int(t)))
    var v_hat_os = new_v_os / (Float32(1.0) - pow_float32(beta2, Int(t)))
    var new_raw_os = raw_os - learning_rate * m_hat_os / (sqrt(v_hat_os) + eps)
    new_raw_os = max(MIN_RAW, min(MAX_RAW, new_raw_os))
    
    # Create new state
    var new_state = AdamStateARD(dim)
    new_state.m_ls = new_m_ls^
    new_state.v_ls = new_v_ls^
    new_state.m_noise = new_m_noise
    new_state.v_noise = new_v_noise
    new_state.m_os = new_m_os
    new_state.v_os = new_v_os
    new_state.t = Int(t)
    
    return AdamUpdateResultARD(new_state^, new_raw_ls^, new_raw_noise, new_raw_os)


# =============================================================================
# Training Loop
# =============================================================================

fn _warmup_gpu_kernels(ctx: DeviceContext) raises:
    """Warmup GPU kernels to avoid cold start in first training iteration.
    
    This pre-loads:
    1. PyTorch (for eigendecomposition in log-det computation)
    2. MAX matmul (for probe sampling)
    
    Skips warmup if PyTorch is already imported (i.e., warmup already done).
    """
    from python import Python
    from buffer import NDBuffer
    from linalg.matmul import matmul as max_matmul
    from collections import Optional
    
    # Check if PyTorch is already imported (skip warmup if so)
    var sys = Python.import_module("sys")
    if "torch" in sys.modules:
        return
    
    # 1. Pre-load PyTorch
    var torch = Python.import_module("torch")
    var np = Python.import_module("numpy")
    
    # 2. Warmup MAX matmul with a small matrix
    var warmup_size = 10
    var A_host = ctx.enqueue_create_host_buffer[float_dtype](warmup_size * warmup_size)
    var B_host = ctx.enqueue_create_host_buffer[float_dtype](warmup_size * warmup_size)
    var C_host = ctx.enqueue_create_host_buffer[float_dtype](warmup_size * warmup_size)
    
    for i in range(warmup_size * warmup_size):
        A_host[i] = Float32(1.0)
        B_host[i] = Float32(1.0)
        C_host[i] = Float32(0.0)
    
    var A_device = ctx.enqueue_create_buffer[float_dtype](warmup_size * warmup_size)
    var B_device = ctx.enqueue_create_buffer[float_dtype](warmup_size * warmup_size)
    var C_device = ctx.enqueue_create_buffer[float_dtype](warmup_size * warmup_size)
    
    ctx.enqueue_copy(dst_buf=A_device, src_buf=A_host)
    ctx.enqueue_copy(dst_buf=B_device, src_buf=B_host)
    ctx.enqueue_copy(dst_buf=C_device, src_buf=C_host)
    ctx.synchronize()
    
    var A_ndbuf = NDBuffer[DType.float32, 2](A_device.unsafe_ptr(), (warmup_size, warmup_size))
    var B_ndbuf = NDBuffer[DType.float32, 2](B_device.unsafe_ptr(), (warmup_size, warmup_size))
    var C_ndbuf = NDBuffer[DType.float32, 2](C_device.unsafe_ptr(), (warmup_size, warmup_size))
    
    var opt_ctx = Optional[DeviceContext](ctx)
    max_matmul[transpose_b=True, target="gpu"](C_ndbuf, A_ndbuf, B_ndbuf, opt_ctx)
    ctx.synchronize()
