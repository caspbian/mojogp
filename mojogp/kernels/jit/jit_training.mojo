"""JIT training infrastructure for MojoGP.

Contains:
- JITGradientProvider trait: extends GradientProvider with update_params/update_noise
- train_jit_with_provider: copy of training.mojo:train_with_provider (lines 1662-1941)
  with JITGradientProvider constraint instead of GradientProvider

Copied from: mojogp/kernels/training.mojo (train_with_provider, lines 1662-1941)
Only change: trait constraint GradientProvider -> JITGradientProvider

See AGENTS.md "JIT Path Isolation Policy" for why this is a copy, not a modification.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import sqrt, log
from time import perf_counter_ns
from python import PythonObject

from kernels.gradient_provider import GradientProvider, ForwardProvider
from kernels.combined_inv_quad_logdet import bbmm_with_precond, CGBufferPool
from kernels.pivoted_cholesky import build_pivoted_cholesky_precond_unified
from kernels.training_types import TrainingResultGeneric, AdamStateGeneric
from kernels.training_utils import (
    adam_update_state_inplace,
    adam_update_params,
    compute_cosine_lr,
    clip_gradient,
    pow_float32,
)
from kernels.utils import softplus, inv_softplus, softplus_derivative
from kernels.cg_solver import kernel_subtract_scalar, kernel_fill_constant
from kernels.constants import float_dtype
from kernels.training_utils import _warmup_gpu_kernels
from gpu.profiler import ProfileBlock
from kernels.constants import PROFILING
from kernels.jit.jit_progress import emit_progress_event, progress_interval_should_emit


fn kernel_update_learned_noise_vector(
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    raw_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    noise_floor: Float32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < UInt(n):
        noise_ptr[i] = softplus(raw_noise_ptr[i]) + noise_floor


fn kernel_update_learned_group_noise_vector(
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    raw_group_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    group_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    n: Int,
    num_groups: Int,
    noise_floor: Float32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < UInt(n):
        var g = Int(group_ids_ptr[i])
        if g >= 0 and g < num_groups:
            noise_ptr[i] = softplus(raw_group_noise_ptr[g]) + noise_floor


fn kernel_update_learned_linear_noise_vector(
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    raw_noise_fn_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    dim: Int,
    noise_floor: Float32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < UInt(n):
        var raw_i = raw_noise_fn_ptr[0]
        for j in range(dim):
            raw_i += raw_noise_fn_ptr[j + 1] * x_ptr[i * UInt(dim) + UInt(j)]
        noise_ptr[i] = softplus(raw_i) + noise_floor


fn kernel_compute_learned_noise_gradient(
    grad_raw_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    probe_solutions_ptr: UnsafePointer[Float32, MutAnyOrigin],
    right_factors_ptr: UnsafePointer[Float32, MutAnyOrigin],
    raw_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_probes: Int,
    noise_floor: Float32,
    regularization: Float32,
    regularization_target: Float32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < UInt(n):
        var trace_est = Float32(0.0)
        for j in range(num_probes):
            var idx = UInt(j) * UInt(n) + i
            trace_est += probe_solutions_ptr[idx] * right_factors_ptr[idx]
        trace_est = trace_est / Float32(num_probes)
        var alpha_i = alpha_ptr[i]
        var grad_noise = Float32(0.5) * (trace_est - alpha_i * alpha_i) / Float32(n)
        var raw_i = raw_noise_ptr[i]
        var noise_i = softplus(raw_i) + noise_floor
        if regularization > Float32(0.0):
            var target = regularization_target
            if target < noise_floor:
                target = noise_floor
            grad_noise += regularization * log(noise_i / target) / (noise_i * Float32(n))
        grad_raw_ptr[i] = grad_noise * softplus_derivative(raw_i)


fn kernel_compute_learned_group_noise_gradient(
    grad_group_raw_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    probe_solutions_ptr: UnsafePointer[Float32, MutAnyOrigin],
    right_factors_ptr: UnsafePointer[Float32, MutAnyOrigin],
    raw_group_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    group_ids_ptr: UnsafePointer[Int32, MutAnyOrigin],
    n: Int,
    num_probes: Int,
    num_groups: Int,
    noise_floor: Float32,
    regularization: Float32,
    regularization_target: Float32,
):
    var g = Int(block_idx.x)
    if g < num_groups:
        var grad_noise_sum = Float32(0.0)
        for i in range(n):
            if Int(group_ids_ptr[i]) == g:
                var trace_est = Float32(0.0)
                for j in range(num_probes):
                    var idx = UInt(j) * UInt(n) + UInt(i)
                    trace_est += probe_solutions_ptr[idx] * right_factors_ptr[idx]
                trace_est = trace_est / Float32(num_probes)
                var alpha_i = alpha_ptr[i]
                grad_noise_sum += Float32(0.5) * (trace_est - alpha_i * alpha_i) / Float32(n)
        var raw_g = raw_group_noise_ptr[g]
        var noise_g = softplus(raw_g) + noise_floor
        if regularization > Float32(0.0):
            var target = regularization_target
            if target < noise_floor:
                target = noise_floor
            grad_noise_sum += regularization * log(noise_g / target) / (noise_g * Float32(num_groups))
        grad_group_raw_ptr[g] = grad_noise_sum * softplus_derivative(raw_g)


fn kernel_compute_learned_linear_noise_gradient(
    grad_noise_fn_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha_ptr: UnsafePointer[Float32, MutAnyOrigin],
    probe_solutions_ptr: UnsafePointer[Float32, MutAnyOrigin],
    right_factors_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    raw_noise_fn_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    dim: Int,
    num_probes: Int,
    noise_floor: Float32,
    regularization: Float32,
    regularization_target: Float32,
):
    var p = Int(block_idx.x)
    var num_params = dim + 1
    if p >= num_params:
        return
    var grad_sum = Float32(0.0)
    for i in range(n):
        var raw_i = raw_noise_fn_ptr[0]
        for j in range(dim):
            raw_i += raw_noise_fn_ptr[j + 1] * x_ptr[UInt(i) * UInt(dim) + UInt(j)]
        var trace_est = Float32(0.0)
        for probe in range(num_probes):
            var idx = UInt(probe) * UInt(n) + UInt(i)
            trace_est += probe_solutions_ptr[idx] * right_factors_ptr[idx]
        trace_est = trace_est / Float32(num_probes)
        var alpha_i = alpha_ptr[i]
        var grad_noise = Float32(0.5) * (trace_est - alpha_i * alpha_i) / Float32(n)
        var noise_i = softplus(raw_i) + noise_floor
        if regularization > Float32(0.0):
            var target = regularization_target
            if target < noise_floor:
                target = noise_floor
            grad_noise += regularization * log(noise_i / target) / (noise_i * Float32(n))
        var grad_raw = grad_noise * softplus_derivative(raw_i)
        if p == 0:
            grad_sum += grad_raw
        else:
            grad_sum += grad_raw * x_ptr[UInt(i) * UInt(dim) + UInt(p - 1)]
    if regularization > Float32(0.0) and p > 0:
        var denom = Float32(dim)
        if denom < Float32(1.0):
            denom = Float32(1.0)
        grad_sum += regularization * raw_noise_fn_ptr[p] / denom
    grad_noise_fn_ptr[p] = grad_sum


fn kernel_adam_update_vector(
    raw_ptr: UnsafePointer[Float32, MutAnyOrigin],
    m_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    grad_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    beta1_power: Float32,
    beta2_power: Float32,
    eps: Float32,
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < UInt(n):
        var g = grad_ptr[i]
        if g != g or g > Float32(1e6) or g < Float32(-1e6):
            g = Float32(0.0)
        var m_new = beta1 * m_ptr[i] + (Float32(1.0) - beta1) * g
        var v_new = beta2 * v_ptr[i] + (Float32(1.0) - beta2) * g * g
        m_ptr[i] = m_new
        v_ptr[i] = v_new
        var m_hat = m_new / (Float32(1.0) - beta1_power)
        var v_hat = v_new / (Float32(1.0) - beta2_power)
        raw_ptr[i] -= lr * m_hat / (sqrt(v_hat) + eps)


# =============================================================================
# JIT-specific trait: extends GradientProvider with param update methods
# =============================================================================

trait JITGradientProvider(GradientProvider):
    """GradientProvider with parameter update methods for JIT training loop.
    
    The AOT GradientProvider trait doesn't include update_params/update_noise,
    but train_with_provider needs them. Rather than modifying the AOT trait
    (which would trigger rebuilds of ALL modules), we create this JIT sub-trait.
    
    Any JIT adapter struct must implement all GradientProvider methods PLUS:
    - update_params: update kernel parameters from host buffer
    - update_noise: update noise variance
    """
    
    fn update_params(
        mut self,
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Update kernel parameters from host buffer.
        
        Args:
            params_host_ptr: New kernel parameters on host [num_kernel_params]
        """
        ...
    
    fn update_noise(mut self, noise: Float32):
        """Update noise variance.
        
        Args:
            noise: New noise variance sigma^2
        """
        ...

    fn get_noise_mode(self) -> Int:
        """Return 0 for scalar noise, 1 for fixed vector noise."""
        ...

    fn get_noise_vector_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return fixed vector noise pointer when get_noise_mode() == 1."""
        ...


# =============================================================================
# JIT training function (copy of training.mojo:train_with_provider)
# =============================================================================

fn train_jit_with_provider[P: JITGradientProvider](
    mut provider: P,
    ctx: DeviceContext,
    y_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_kernel_params: Int,
    initial_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    initial_noise: Float32,
    max_iterations: Int = 100,
    learning_rate: Float32 = 0.01,
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    enable_early_stopping: Bool = False,
    early_stop_patience: Int = 10,
    early_stop_tol: Float32 = 1e-4,
    verbose: Bool = False,
    init_mean: Float32 = 0.0,
    max_tridiag_iter: Int = 30,
    precond_rebuild_threshold: Float32 = 0.5,
    use_cosine_lr: Bool = True,
    use_preconditioner: Bool = True,
    precond_method: Int = 0,
    learn_noise: Bool = True,
    noise_floor: Float32 = 1e-6,
    noise_regularization: Float32 = 0.01,
    noise_group_ids_ptr: UnsafePointer[Int32, MutAnyOrigin] = UnsafePointer[Int32, MutAnyOrigin](),
    num_noise_groups: Int = 0,
    noise_function_dim: Int = 0,
    progress_callback: PythonObject = None,
    progress_interval: Int = 1,
    progress_enabled: Bool = False,
) raises -> TrainingResultGeneric:
    """Train a GP with any JITGradientProvider using BBMM.
    
    Copy of training.mojo:train_with_provider (lines 1662-1941) with
    JITGradientProvider constraint. This enables provider.update_params()
    and provider.update_noise() to resolve through the trait.
    
    Args:
        provider: A JITGradientProvider (JIT adapter wrapping CompositeProvider)
        ctx: GPU device context
        y_host_ptr: Training targets [n] on host
        n: Number of training points
        num_kernel_params: Number of kernel parameters
        initial_params_ptr: Initial kernel parameters [num_kernel_params] on host (constrained space)
        initial_noise: Initial noise variance
        max_iterations: Maximum training iterations
        learning_rate: Adam learning rate
        num_probes: Number of probes for log-det and gradient estimation
        max_cg_iter: Maximum CG iterations
        cg_tol: CG convergence tolerance
        precond_rank: Rank for Pivoted Cholesky preconditioner
        early_stop_patience: Iterations without improvement before stopping
        early_stop_tol: Minimum improvement to reset patience
        verbose: Print progress
        init_mean: Initial constant mean value
        max_tridiag_iter: Maximum tridiagonal iterations for log-det
        precond_rebuild_threshold: Relative param change threshold for preconditioner rebuild
        use_cosine_lr: Whether to use cosine LR schedule
        
    Returns:
        TrainingResultGeneric with optimized parameters
    """
    # NOTE: Skipping _warmup_gpu_kernels(ctx) — the AOT path calls this in
    # train_gp_with_method, but JIT modules don't need it since PyTorch
    # eigendecomposition is imported lazily on first BBMM iteration.
    
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var y_device = ctx.enqueue_create_buffer[float_dtype](n)
    var raw_mean = init_mean
    var y_host_centered = ctx.enqueue_create_host_buffer[float_dtype](n)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](n)
    var raw_params = List[Float32]()
    var adam_state = AdamStateGeneric(num_kernel_params)
    var pool = CGBufferPool(ctx, n, 1 + num_probes, num_probes, max_tridiag_iter)
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_kernel_params)
    var learned_noise_mode = provider.get_noise_mode() == 2
    var learned_group_noise_mode = provider.get_noise_mode() == 3
    var learned_linear_noise_mode = provider.get_noise_mode() == 4
    var noise_state_size = n
    if learned_group_noise_mode:
        noise_state_size = num_noise_groups
    if learned_linear_noise_mode:
        if noise_function_dim <= 0:
            raise Error("learned linear noise requires a positive noise_function_dim")
        noise_state_size = noise_function_dim + 1
    var raw_noise_device = ctx.enqueue_create_buffer[float_dtype](noise_state_size)
    var grad_noise_device = ctx.enqueue_create_buffer[float_dtype](noise_state_size)
    var noise_m_device = ctx.enqueue_create_buffer[float_dtype](noise_state_size)
    var noise_v_device = ctx.enqueue_create_buffer[float_dtype](noise_state_size)
    var raw_noise_init_host = ctx.enqueue_create_host_buffer[float_dtype](noise_state_size)
    var best_raw_noise_host = ctx.enqueue_create_host_buffer[float_dtype](noise_state_size)
    var best_noise_m_host = ctx.enqueue_create_host_buffer[float_dtype](noise_state_size)
    var best_noise_v_host = ctx.enqueue_create_host_buffer[float_dtype](noise_state_size)
    with ProfileBlock[PROFILING]("JIT_fit_setup"):
        # Copy y to device
        for i in range(n):
            y_host[i] = y_host_ptr[i]
        ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
        ctx.synchronize()
        
        # Initialize raw parameters (unconstrained space via inv_softplus)
        for p in range(num_kernel_params):
            raw_params.append(inv_softplus(initial_params_ptr[p]))
        raw_params.append(inv_softplus(initial_noise))  # Noise slot is last; fixed-noise mode leaves it unchanged.
        
        # Create params host buffer (reused each iteration)
        for p in range(num_kernel_params):
            params_host[p] = initial_params_ptr[p]
        var init_vector_noise = initial_noise - noise_floor
        if init_vector_noise < Float32(1e-8):
            init_vector_noise = Float32(1e-8)
        if learned_linear_noise_mode:
            raw_noise_init_host[0] = inv_softplus(init_vector_noise)
            best_raw_noise_host[0] = raw_noise_init_host[0]
            best_noise_m_host[0] = Float32(0.0)
            best_noise_v_host[0] = Float32(0.0)
            for i in range(1, noise_state_size):
                raw_noise_init_host[i] = Float32(0.0)
                best_raw_noise_host[i] = Float32(0.0)
                best_noise_m_host[i] = Float32(0.0)
                best_noise_v_host[i] = Float32(0.0)
        else:
            for i in range(noise_state_size):
                raw_noise_init_host[i] = inv_softplus(init_vector_noise)
                best_raw_noise_host[i] = raw_noise_init_host[i]
                best_noise_m_host[i] = Float32(0.0)
                best_noise_v_host[i] = Float32(0.0)
        ctx.enqueue_copy(dst_buf=raw_noise_device, src_buf=raw_noise_init_host)
        ctx.enqueue_function[kernel_fill_constant](
            noise_m_device.unsafe_ptr(), Float32(0.0), noise_state_size,
            grid_dim=((noise_state_size + 255) // 256,), block_dim=(256,)
        )
        ctx.enqueue_function[kernel_fill_constant](
            noise_v_device.unsafe_ptr(), Float32(0.0), noise_state_size,
            grid_dim=((noise_state_size + 255) // 256,), block_dim=(256,)
        )
        if learned_noise_mode:
            ctx.enqueue_function[kernel_update_learned_noise_vector](
                provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), n, noise_floor,
                grid_dim=((n + 255) // 256,), block_dim=(256,)
            )
            ctx.synchronize()
        if learned_group_noise_mode:
            ctx.enqueue_function[kernel_update_learned_group_noise_vector](
                provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), noise_group_ids_ptr,
                n, num_noise_groups, noise_floor,
                grid_dim=((n + 255) // 256,), block_dim=(256,)
            )
            ctx.synchronize()
        if learned_linear_noise_mode:
            ctx.enqueue_function[kernel_update_learned_linear_noise_vector](
                provider.get_noise_vector_ptr(), provider.get_x_ptr(), raw_noise_device.unsafe_ptr(),
                n, noise_function_dim, noise_floor,
                grid_dim=((n + 255) // 256,), block_dim=(256,)
            )
            ctx.synchronize()
    
    # Training loop variables
    var best_nll = Float32(1e30)
    var no_improve_count = 0
    var actual_iterations = 0
    var converged = False
    var last_nll = Float32(0.0)
    
    # Mutable copy of learning_rate (function params are immutable in Mojo)
    var lr = learning_rate
    
    # Track last params for preconditioner rebuild threshold
    var last_rebuild_params = List[Float32]()
    for p in range(num_kernel_params):
        last_rebuild_params.append(initial_params_ptr[p])
    var last_rebuild_noise = initial_noise
    
    # Best-param tracking: snapshot params at best NLL
    var best_raw_params = List[Float32]()
    var best_adam_m = List[Float32]()
    var best_adam_v = List[Float32]()
    for p in range(num_kernel_params + 1):  # +1 for noise
        best_raw_params.append(raw_params[p])
        best_adam_m.append(Float32(0.0))
        best_adam_v.append(Float32(0.0))
    var best_raw_mean = raw_mean
    var best_adam_m_mean = Float32(0.0)
    var best_adam_v_mean = Float32(0.0)
    var best_adam_t = 0
    var best_nll_seen = Float32(1e30)
    var best_alpha_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var best_alpha_valid = False
    var precond_build_count = 0
    var precond_build_total_ns = Int(0)
    var precond_rank_history = List[Int]()
    var precond_rebuild_steps = List[Int]()

    # Build initial preconditioner
    var num_cols = 1 + num_probes
    var num_kparams = provider.num_gradient_params()
    var input_dim_for_buffers = noise_function_dim
    if input_dim_for_buffers <= 0:
        input_dim_for_buffers = 1
    var precond_error_tol = Float32(1e-3)
    if precond_method == 0:
        precond_error_tol = Float32(0.0)

    pool.ensure_capacity(
        ctx, n, num_cols, num_probes, max_tridiag_iter, precond_rank,
        dim=input_dim_for_buffers, num_kernel_params=num_kparams,
    )
    var initial_precond_build_start = perf_counter_ns()
    var precond = build_pivoted_cholesky_precond_unified(
        provider,
        rank=precond_rank,
        error_tol=precond_error_tol,
        max_num_cols=num_cols,
        precond_method=precond_method,
        noise_mode=provider.get_noise_mode(),
        noise_vec_ptr=provider.get_noise_vector_ptr(),
    )
    precond_build_total_ns += Int(perf_counter_ns() - initial_precond_build_start)
    precond_build_count += 1

    
    # Per-iteration timing, NLL history, and realized CG iterations
    var iter_times_ns = List[Int]()
    var nll_history = List[Float32]()
    var cg_iterations_history = List[Int]()

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "single_output",
            "unknown",
            "start",
            0,
            max_iterations,
            precond_rank=precond.rank,
            precond_rebuild_count=max(0, precond_build_count - 1),
        )

    for iteration in range(max_iterations):
        var iter_start = perf_counter_ns()
        actual_iterations = iteration + 1
        
        # Convert raw params to constrained space
        var constrained_params = List[Float32]()
        for p in range(num_kernel_params):
            constrained_params.append(softplus(raw_params[p]))
        var noise = softplus(raw_params[num_kernel_params]) if learn_noise else Float32(0.0)
        
        # Update params host buffer
        for p in range(num_kernel_params):
            params_host[p] = constrained_params[p]
        
        with ProfileBlock[PROFILING]("JIT_iter_provider_update"):
            if learned_noise_mode:
                ctx.enqueue_function[kernel_update_learned_noise_vector](
                    provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), n, noise_floor,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
            if learned_group_noise_mode:
                ctx.enqueue_function[kernel_update_learned_group_noise_vector](
                    provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), noise_group_ids_ptr,
                    n, num_noise_groups, noise_floor,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
            if learned_linear_noise_mode:
                ctx.enqueue_function[kernel_update_learned_linear_noise_vector](
                    provider.get_noise_vector_ptr(), provider.get_x_ptr(), raw_noise_device.unsafe_ptr(),
                    n, noise_function_dim, noise_floor,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
            # Update provider with new parameters
            # NOTE: These resolve because JITGradientProvider declares them
            provider.update_params(params_host.unsafe_ptr())
            provider.update_noise(noise)
        
        # Adaptive preconditioner rebuild
        if iteration > 0:
            var max_rel = Float32(0.0)
            for p in range(num_kernel_params):
                var rel_p = abs(constrained_params[p] - last_rebuild_params[p]) / max(abs(last_rebuild_params[p]), Float32(1e-8))
                max_rel = max(max_rel, rel_p)
            if learn_noise:
                var rel_noise = abs(noise - last_rebuild_noise) / max(abs(last_rebuild_noise), Float32(1e-8))
                max_rel = max(max_rel, rel_noise)
            if learned_noise_mode or learned_group_noise_mode or learned_linear_noise_mode:
                max_rel = precond_rebuild_threshold + Float32(1.0)
            if max_rel > precond_rebuild_threshold:
                with ProfileBlock[PROFILING]("JIT_precond_rebuild"):
                    var precond_build_start = perf_counter_ns()
                    precond = build_pivoted_cholesky_precond_unified(
                        provider,
                        rank=precond_rank,
                        error_tol=precond_error_tol,
                        max_num_cols=num_cols,
                        precond_method=precond_method,
                        noise_mode=provider.get_noise_mode(),
                        noise_vec_ptr=provider.get_noise_vector_ptr(),
                    )
                    precond_build_total_ns += Int(perf_counter_ns() - precond_build_start)
                    precond_build_count += 1
                    precond_rebuild_steps.append(iteration)
                for p in range(num_kernel_params):
                    last_rebuild_params[p] = constrained_params[p]
                last_rebuild_noise = noise

        precond_rank_history.append(precond.rank)
        
        # ConstantMean: Center y by current mean on GPU (no CPU loop or H2D copy)
        with ProfileBlock[PROFILING]("JIT_iter_center_y"):
            ctx.enqueue_function[kernel_subtract_scalar](
                y_centered_device.unsafe_ptr(), y_device.unsafe_ptr(), raw_mean, n,
                grid_dim=((n + 255) // 256,), block_dim=(256,)
            )
        
        # Compute NLL and gradients with cached preconditioner
        var result = bbmm_with_precond(
            provider, precond, y_centered_device.unsafe_ptr(), n, pool,
            num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
            iteration=iteration,
            recycle_alpha=iteration > 0,
            use_preconditioner=use_preconditioner,
        )

        last_nll = result.nll
        cg_iterations_history.append(result.num_iterations)
        
        # NaN detection: if NLL is NaN, halve learning rate and skip this step
        if last_nll != last_nll:
            lr = lr * Float32(0.5)
            if verbose:
                print("  NaN NLL detected at iteration", iteration, "- halving LR to", lr)
            for p in range(num_kernel_params + 1):
                raw_params[p] = best_raw_params[p]
                adam_state.m[p] = best_adam_m[p]
                adam_state.v[p] = best_adam_v[p]
            adam_state.m_mean = best_adam_m_mean
            adam_state.v_mean = best_adam_v_mean
            adam_state.t = best_adam_t
            raw_mean = best_raw_mean
            for p in range(num_kernel_params):
                params_host[p] = softplus(raw_params[p])
                last_rebuild_params[p] = params_host[p]
            var restored_noise = softplus(raw_params[num_kernel_params]) if learn_noise else Float32(0.0)
            provider.update_params(params_host.unsafe_ptr())
            provider.update_noise(restored_noise)
            last_rebuild_noise = restored_noise
            if learned_noise_mode:
                ctx.enqueue_copy(dst_buf=raw_noise_device, src_buf=best_raw_noise_host)
                ctx.enqueue_copy(dst_buf=noise_m_device, src_buf=best_noise_m_host)
                ctx.enqueue_copy(dst_buf=noise_v_device, src_buf=best_noise_v_host)
                ctx.enqueue_function[kernel_update_learned_noise_vector](
                    provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), n, noise_floor,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
                ctx.synchronize()
            if learned_group_noise_mode:
                ctx.enqueue_copy(dst_buf=raw_noise_device, src_buf=best_raw_noise_host)
                ctx.enqueue_copy(dst_buf=noise_m_device, src_buf=best_noise_m_host)
                ctx.enqueue_copy(dst_buf=noise_v_device, src_buf=best_noise_v_host)
                ctx.enqueue_function[kernel_update_learned_group_noise_vector](
                    provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), noise_group_ids_ptr,
                    n, num_noise_groups, noise_floor,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
                ctx.synchronize()
            if learned_linear_noise_mode:
                ctx.enqueue_copy(dst_buf=raw_noise_device, src_buf=best_raw_noise_host)
                ctx.enqueue_copy(dst_buf=noise_m_device, src_buf=best_noise_m_host)
                ctx.enqueue_copy(dst_buf=noise_v_device, src_buf=best_noise_v_host)
                ctx.enqueue_function[kernel_update_learned_linear_noise_vector](
                    provider.get_noise_vector_ptr(), provider.get_x_ptr(), raw_noise_device.unsafe_ptr(),
                    n, noise_function_dim, noise_floor,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
                ctx.synchronize()
            var precond_build_start_nan = perf_counter_ns()
            precond = build_pivoted_cholesky_precond_unified(
                provider,
                rank=precond_rank,
                error_tol=precond_error_tol,
                max_num_cols=num_cols,
                precond_method=precond_method,
                noise_mode=provider.get_noise_mode(),
                noise_vec_ptr=provider.get_noise_vector_ptr(),
            )
            precond_build_total_ns += Int(perf_counter_ns() - precond_build_start_nan)
            precond_build_count += 1
            precond_rebuild_steps.append(iteration)
            if progress_enabled:
                var nan_iter_time = Int(perf_counter_ns() - iter_start)
                emit_progress_event(
                    progress_callback,
                    "train",
                    "single_output",
                    "unknown",
                    "nan",
                    actual_iterations,
                    max_iterations,
                    nll=last_nll,
                    best_nll=best_nll_seen,
                    cg_iter=result.num_iterations,
                    iter_time_ns=nan_iter_time,
                    noise=restored_noise,
                    mean=raw_mean,
                    precond_rank=precond.rank,
                    precond_rebuild_count=max(0, precond_build_count - 1),
                )
            continue
        
        # Best-param tracking: snapshot params at best NLL
        if last_nll < best_nll_seen:
            best_nll_seen = last_nll
            for p in range(num_kernel_params + 1):
                best_raw_params[p] = raw_params[p]
            best_raw_mean = raw_mean
            for p in range(num_kernel_params + 1):
                best_adam_m[p] = adam_state.m[p]
                best_adam_v[p] = adam_state.v[p]
            best_adam_m_mean = adam_state.m_mean
            best_adam_v_mean = adam_state.v_mean
            best_adam_t = adam_state.t
            if learned_noise_mode or learned_group_noise_mode or learned_linear_noise_mode:
                ctx.enqueue_copy(dst_buf=best_raw_noise_host, src_buf=raw_noise_device)
                ctx.enqueue_copy(dst_buf=best_noise_m_host, src_buf=noise_m_device)
                ctx.enqueue_copy(dst_buf=best_noise_v_host, src_buf=noise_v_device)
            ctx.enqueue_copy(dst_buf=best_alpha_host, src_buf=result.solution)
            ctx.synchronize()
            best_alpha_valid = True
        
        # Cosine LR decay
        var effective_lr = lr
        if use_cosine_lr:
            effective_lr = compute_cosine_lr(learning_rate, iteration, max_iterations)
        
        # Collect gradients: [kernel_gradients..., grad_noise]
        # Unified result: gradients[0..N-1] = kernel params, gradients[N] = noise
        var gradients = List[Float32]()
        for p in range(num_kernel_params):
            gradients.append(result.gradients[p])
        if learn_noise:
            gradients.append(result.gradients[num_kernel_params])  # Noise is last
        else:
            gradients.append(Float32(0.0))
        
        # Adam update for kernel params + noise
        with ProfileBlock[PROFILING]("JIT_iter_adam"):
            adam_state = adam_update_state_inplace(
                adam_state^, gradients, raw_params, effective_lr
            )
            raw_params = adam_update_params(
                adam_state, gradients, raw_params, effective_lr
            )
            if learned_noise_mode:
                ctx.enqueue_function[kernel_compute_learned_noise_gradient](
                    grad_noise_device.unsafe_ptr(),
                    result.solution.unsafe_ptr(),
                    result.probe_solutions.unsafe_ptr(),
                    result.right_factors.unsafe_ptr(),
                    raw_noise_device.unsafe_ptr(),
                    n,
                    num_probes,
                    noise_floor,
                    noise_regularization,
                    initial_noise,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
                ctx.enqueue_function[kernel_adam_update_vector](
                    raw_noise_device.unsafe_ptr(),
                    noise_m_device.unsafe_ptr(),
                    noise_v_device.unsafe_ptr(),
                    grad_noise_device.unsafe_ptr(),
                    n,
                    effective_lr,
                    Float32(0.9),
                    Float32(0.999),
                    pow_float32(Float32(0.9), adam_state.t),
                    pow_float32(Float32(0.999), adam_state.t),
                    Float32(1e-8),
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
                ctx.synchronize()
            if learned_group_noise_mode:
                ctx.enqueue_function[kernel_compute_learned_group_noise_gradient](
                    grad_noise_device.unsafe_ptr(),
                    result.solution.unsafe_ptr(),
                    result.probe_solutions.unsafe_ptr(),
                    result.right_factors.unsafe_ptr(),
                    raw_noise_device.unsafe_ptr(),
                    noise_group_ids_ptr,
                    n,
                    num_probes,
                    num_noise_groups,
                    noise_floor,
                    noise_regularization,
                    initial_noise,
                    grid_dim=(num_noise_groups,), block_dim=(1,)
                )
                ctx.enqueue_function[kernel_adam_update_vector](
                    raw_noise_device.unsafe_ptr(),
                    noise_m_device.unsafe_ptr(),
                    noise_v_device.unsafe_ptr(),
                    grad_noise_device.unsafe_ptr(),
                    num_noise_groups,
                    effective_lr,
                    Float32(0.9),
                    Float32(0.999),
                    pow_float32(Float32(0.9), adam_state.t),
                    pow_float32(Float32(0.999), adam_state.t),
                    Float32(1e-8),
                    grid_dim=((num_noise_groups + 255) // 256,), block_dim=(256,)
                )
                ctx.synchronize()
            if learned_linear_noise_mode:
                ctx.enqueue_function[kernel_compute_learned_linear_noise_gradient](
                    grad_noise_device.unsafe_ptr(),
                    result.solution.unsafe_ptr(),
                    result.probe_solutions.unsafe_ptr(),
                    result.right_factors.unsafe_ptr(),
                    provider.get_x_ptr(),
                    raw_noise_device.unsafe_ptr(),
                    n,
                    noise_function_dim,
                    num_probes,
                    noise_floor,
                    noise_regularization,
                    initial_noise,
                    grid_dim=(noise_state_size,), block_dim=(1,)
                )
                ctx.enqueue_function[kernel_adam_update_vector](
                    raw_noise_device.unsafe_ptr(),
                    noise_m_device.unsafe_ptr(),
                    noise_v_device.unsafe_ptr(),
                    grad_noise_device.unsafe_ptr(),
                    noise_state_size,
                    effective_lr,
                    Float32(0.9),
                    Float32(0.999),
                    pow_float32(Float32(0.9), adam_state.t),
                    pow_float32(Float32(0.999), adam_state.t),
                    Float32(1e-8),
                    grid_dim=((noise_state_size + 255) // 256,), block_dim=(256,)
                )
                ctx.synchronize()
        
        # ConstantMean: Compute mean gradient via GPU reduction
        with ProfileBlock[PROFILING]("JIT_iter_mean_update"):
            from kernels.cg_solver import kernel_sum_reduce
            var mean_sum_dev = ctx.enqueue_create_buffer[DType.float32](1)
            var mean_sum_host = ctx.enqueue_create_host_buffer[DType.float32](1)
            ctx.enqueue_function[kernel_sum_reduce](
                mean_sum_dev.unsafe_ptr(), result.solution.unsafe_ptr(), n,
                grid_dim=1, block_dim=256, shared_mem_bytes=8 * 4)
            ctx.enqueue_copy(dst_buf=mean_sum_host, src_buf=mean_sum_dev)
            ctx.synchronize()
            var mean_grad = -mean_sum_host.unsafe_ptr()[0] / Float32(n)
            mean_grad = clip_gradient(mean_grad)
            
            # Adam update for mean (unconstrained, no softplus chain rule)
            var beta1 = Float32(0.9)
            var beta2 = Float32(0.999)
            var eps = Float32(1e-8)
            var t_mean = Float32(adam_state.t)  # Already incremented by adam_update_state_inplace
            adam_state.m_mean = beta1 * adam_state.m_mean + (Float32(1.0) - beta1) * mean_grad
            adam_state.v_mean = beta2 * adam_state.v_mean + (Float32(1.0) - beta2) * mean_grad * mean_grad
            var m_hat_mean = adam_state.m_mean / (Float32(1.0) - pow_float32(beta1, Int(t_mean)))
            var v_hat_mean = adam_state.v_mean / (Float32(1.0) - pow_float32(beta2, Int(t_mean)))
            raw_mean -= effective_lr * m_hat_mean / (sqrt(v_hat_mean) + eps)
        
        # Safety check for mean
        if raw_mean != raw_mean:  # NaN check
            raw_mean = init_mean
        elif raw_mean > Float32(1000.0) or raw_mean < Float32(-1000.0):
            raw_mean = init_mean
        
        # Print progress
        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print("Iter", iteration, ": NLL =", last_nll, ", noise =", noise, ", mean =", raw_mean)
        
        # Record per-iteration time and NLL
        var iter_end = perf_counter_ns()
        var iter_time_ns = Int(iter_end - iter_start)
        iter_times_ns.append(iter_time_ns)
        nll_history.append(last_nll)
        if progress_enabled and progress_interval_should_emit(iteration, max_iterations, progress_interval):
            emit_progress_event(
                progress_callback,
                "train",
                "single_output",
                "unknown",
                "iteration",
                actual_iterations,
                max_iterations,
                nll=last_nll,
                best_nll=best_nll_seen,
                cg_iter=result.num_iterations,
                iter_time_ns=iter_time_ns,
                noise=noise,
                mean=raw_mean,
                precond_rank=precond.rank,
                precond_rebuild_count=max(0, precond_build_count - 1),
            )
        
        # Early stopping check (disabled by default)
        if enable_early_stopping:
            if last_nll < best_nll - early_stop_tol:
                best_nll = last_nll
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    converged = True
                    if verbose:
                        print("Early stopping: converged after", actual_iterations, "iterations")
                    if progress_enabled:
                        emit_progress_event(
                            progress_callback,
                            "train",
                            "single_output",
                            "unknown",
                            "early_stop",
                            actual_iterations,
                            max_iterations,
                            nll=last_nll,
                            best_nll=best_nll_seen,
                            cg_iter=result.num_iterations,
                            iter_time_ns=iter_time_ns,
                            noise=noise,
                            mean=raw_mean,
                            precond_rank=precond.rank,
                            precond_rebuild_count=max(0, precond_build_count - 1),
                            converged=converged,
                        )
                    break
    
    # Keepalive for cached preconditioner
    _ = precond

    # Restore best-seen params before computing final parameters
    for p in range(num_kernel_params + 1):
        raw_params[p] = best_raw_params[p]
    raw_mean = best_raw_mean
    if learned_noise_mode:
        ctx.enqueue_copy(dst_buf=raw_noise_device, src_buf=best_raw_noise_host)
        ctx.enqueue_function[kernel_update_learned_noise_vector](
            provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), n, noise_floor,
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        ctx.synchronize()
    if learned_group_noise_mode:
        ctx.enqueue_copy(dst_buf=raw_noise_device, src_buf=best_raw_noise_host)
        ctx.enqueue_function[kernel_update_learned_group_noise_vector](
            provider.get_noise_vector_ptr(), raw_noise_device.unsafe_ptr(), noise_group_ids_ptr,
            n, num_noise_groups, noise_floor,
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        ctx.synchronize()
    if learned_linear_noise_mode:
        ctx.enqueue_copy(dst_buf=raw_noise_device, src_buf=best_raw_noise_host)
        ctx.enqueue_function[kernel_update_learned_linear_noise_vector](
            provider.get_noise_vector_ptr(), provider.get_x_ptr(), raw_noise_device.unsafe_ptr(),
            n, noise_function_dim, noise_floor,
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        ctx.synchronize()

    # Extract final parameters
    var final_params = List[Float32]()
    for p in range(num_kernel_params):
        final_params.append(softplus(raw_params[p]))
    var final_noise = softplus(raw_params[num_kernel_params]) if learn_noise else Float32(0.0)
    var final_mean = raw_mean
    var final_noise_function_params = List[Float32]()
    if learned_linear_noise_mode:
        var noise_fn_params_host = ctx.enqueue_create_host_buffer[float_dtype](noise_state_size)
        ctx.enqueue_copy(dst_buf=noise_fn_params_host, src_buf=raw_noise_device)
        ctx.synchronize()
        for i in range(noise_state_size):
            final_noise_function_params.append(noise_fn_params_host[i])
    
    if verbose:
        print("Training complete:")
        print("  Final NLL =", best_nll_seen)
        print("  Final noise =", final_noise)
        print("  Final mean =", final_mean)
        print("  Final params = [", end="")
        for p in range(num_kernel_params):
            print(final_params[p], end="")
            if p < num_kernel_params - 1:
                print(", ", end="")
        print("]")

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "single_output",
            "unknown",
            "complete",
            actual_iterations,
            max_iterations,
            nll=best_nll_seen,
            best_nll=best_nll_seen,
            noise=final_noise,
            mean=final_mean,
            precond_rank=precond.rank,
            precond_rebuild_count=max(0, precond_build_count - 1),
            converged=converged,
        )
    
    # Lanczos root deferred to prediction time (predict recomputes it).
    var lanczos_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    # Keep buffers alive until all pointer users are done
    _ = y_device
    _ = y_centered_device
    
    with ProfileBlock[PROFILING]("JIT_fit_finalize"):
        return TrainingResultGeneric(
            final_params^, final_noise, final_mean, best_nll_seen,
            actual_iterations, converged,
            lanczos_dummy^, 0, n, num_kernel_params,
            best_alpha_host^, best_alpha_valid,
            iter_times_ns^,
            nll_history^,
            cg_iterations_history^,
            precond_build_count,
            precond_build_total_ns,
            precond_rank_history^,
            precond_rebuild_steps^,
            final_noise_function_params^,
        )
