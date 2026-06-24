"""GP training functions.

Provides the main training loops for single-output GP models:
- train_gp_with_method: Isotropic GP training (materialized or matrix-free)
- train_gp_ard: ARD GP training with per-dimension lengthscales
- train_gp_composite: Composite kernel GP training

Types and utility functions are defined in training_types.mojo and training_utils.mojo
and re-exported here for backward compatibility.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt

# Re-export types from training_types for backward compatibility
from .training_types import (
    float_dtype,
    NLLResult,
    GradientResult,
    TrainingResult,
    TrainingResultARD,
    AdamState,
    AdamStateARD,
    AdamStateGeneric,
    AdamUpdateResultGeneric,
    TrainingResultGeneric,
    AdamUpdateResult,
    AdamUpdateResultARD,
)

# Re-export utility functions from training_utils for backward compatibility
from .training_utils import (
    pow_float32,
    clip_gradient,
    adam_update_generic,
    adam_update_state_inplace,
    adam_update_state_inplace_custom,
    adam_update_params,
    adam_update,
    adam_update_ard,
    _warmup_gpu_kernels,
    compute_dynamic_cg_tol,
    compute_cosine_lr,
)

from .matvec_provider import MaterializedProvider, MatrixFreeProvider
from .combined_inv_quad_logdet import CGBufferPool, bbmm_with_precond, batched_cg_unified, build_pivoted_cholesky_precond_unified
from .gradient_provider import GradientProvider, IsotropicGradientAdapter, ARDGradientAdapter, CompositeGradientAdapter, MaterializedCompositeGradientAdapter
from .composable_kernel import ComposableKernel
from .composite_provider import CompositeProvider, MaterializedCompositeProvider
# compute_lanczos_root_composite import removed — Lanczos root deferred to prediction time
from .utils import softplus, inv_softplus, softplus_derivative
from .cg_solver import kernel_subtract_scalar
from gpu.profiler import ProfileBlock
from .constants import (
    PROFILING,
    KERNEL_TYPE_PERIODIC,
    KERNEL_TYPE_RQ,
    KERNEL_TYPE_LINEAR,
    KERNEL_TYPE_POLYNOMIAL,
)


fn train_gp_with_method(
    ctx: DeviceContext,
    x_host: HostBuffer[float_dtype],
    y_host: HostBuffer[float_dtype],
    n: Int,
    dim: Int,
    kernel_type: Int,
    method: String = "auto",
    num_iterations: Int = 100,
    learning_rate: Float32 = 0.05,
    init_lengthscale: Float32 = 1.0,
    init_noise: Float32 = 0.1,
    init_outputscale: Float32 = 1.0,
    kernel_param1: Float32 = 1.0,
    kernel_param2: Float32 = 0.0,
    early_stop_patience: Int = 15,
    early_stop_threshold: Float32 = 1e-4,
    min_iterations: Int = 20,
    verbose: Bool = True,
    use_preconditioner: Bool = True,
    precond_type_str: String = "pivoted_cholesky",
    init_mean: Float32 = 0.0,
    # CG parameters (configurable from Python via presets)
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    max_tridiag_iter: Int = 30,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    precond_rebuild_threshold: Float32 = 0.5,
    # Preconditioner method: 0=greedy, 1=rpcholesky, 2=nystrom (default)
    precond_method: Int = 2,
    # Learning rate schedule: True=cosine decay, False=constant
    use_cosine_lr: Bool = True,
) raises -> TrainingResult:
    """Train GP with strategy/method selection.
    
    This function implements the dual-mode architecture from the BBMM plan:
    - FAST mode: Uses BBMM (Blackbox Matrix-Matrix) inference with Pivoted Cholesky
      preconditioning. This is GPyTorch-competitive and should be 3-4x faster than
      the matrix-free approach for n < 10000.
    - MEMORY_EFFICIENT mode: Uses matrix-free approach with O(n) memory.
    
    Args:
        method: Strategy selection
            - "fast" or "materialized": Use BBMM with O(n²) memory (GPyTorch-style)
            - "memory_efficient" or "matrix_free": Use O(n) matrix-free approach
            - "auto": Choose based on n (fast if n < 10000, memory_efficient otherwise)
        use_preconditioner: (Deprecated) Use True for Pivoted Cholesky, False for none
        precond_type_str: Preconditioner type (overrides use_preconditioner)
            - "pivoted_cholesky" or "pivoted": Low-rank Pivoted Cholesky (recommended, default)
            - "jacobi": Diagonal Jacobi preconditioning (ineffective for RBF kernels)
            - "none": No preconditioning
            
    BBMM Algorithm (fast mode):
        1. Single batched mBCG solve produces alpha AND probe solutions
        2. log|K| extracted from CG coefficients (essentially FREE!)
        3. Gradients computed using probe solutions (NO EXTRA CG SOLVE!)
        4. Pivoted Cholesky preconditioning reduces CG iterations 2-10x
        
    Expected speedup: 3-4x faster than matrix-free for n < 10000
    """
    var use_materialized = False
    if method == "materialized" or method == "fast":
        use_materialized = True
    elif method == "auto":
        use_materialized = (n < 10000)
    # method == "matrix_free" or "memory_efficient" -> use_materialized = False
    
    if verbose:
        if use_materialized:
            print("Using BBMM fast mode (O(n²) memory, GPyTorch-competitive)")
        else:
            print("Using matrix-free mode (O(n) memory)")
    
    # Copy x to device (needed for both providers)
    var x_device = ctx.enqueue_create_buffer[float_dtype](n * dim)
    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    
    # Copy y to device
    var y_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.synchronize()
    
    # ConstantMean: Initialize mean parameter (unconstrained, no softplus)
    var raw_mean = init_mean
    
    # ConstantMean: Create y_centered buffers for centering y by current mean
    var y_host_centered = ctx.enqueue_create_host_buffer[float_dtype](n)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Initialize raw (unconstrained) parameters
    var raw_lengthscale = inv_softplus(init_lengthscale)
    var raw_noise = inv_softplus(init_noise)
    var raw_outputscale = inv_softplus(init_outputscale)
    
    # Initialize Adam state
    var adam_state = AdamState()
    
    # Early stopping state
    var best_nll = Float32(1e10)
    var no_improve_count = 0
    var converged = False
    var actual_iterations = 0
    
    # Create parameter buffer on device ONCE (reused across iterations)
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = ctx.enqueue_create_host_buffer[float_dtype](2)
    params_host_temp[0] = init_lengthscale
    params_host_temp[1] = init_outputscale
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
    ctx.synchronize()
    
    # Pre-load PyTorch and JIT-compile MAX matmul.
    # This is a ~2s one-time cost per process. Without it, the first bbmm_unified
    # iteration takes ~2.5s due to lazy PyTorch import.
    # TODO: Replace PyTorch dependency with pure Mojo eigendecomposition to eliminate this.
    _warmup_gpu_kernels(ctx)
    
    # Reuse GPU buffers across iterations; ensure_capacity resizes as needed.
    var bbmm_pool = CGBufferPool(ctx)
    
    # Track last NLL to avoid redundant final BBMM call
    var last_nll = Float32(0.0)
    
    # Track NLL history for diagnostics
    var nll_history = List[Float32]()
    
    # Mutable copy of learning_rate (function params are immutable in Mojo)
    var lr = learning_rate
    
    # Branch based on method: matrix-free vs materialized
    # Both use BBMM, but with different providers (O(n) vs O(n²) memory)
    if not use_materialized:
        # =========================================================================
        # MATRIX-FREE PATH: O(n) memory, recomputes K @ v on-the-fly
        # Uses MatrixFreeProvider with unified BBMM via IsotropicGradientAdapter
        # =========================================================================
        
        # Check if kernel has learnable param1 (periodic/RQ/linear) or param2 (polynomial offset)
        from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
        var has_learnable_param1_mf = (kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ or kernel_type == KERNEL_TYPE_LINEAR or kernel_type == KERNEL_TYPE_POLYNOMIAL)
        var has_learnable_param2_mf = (kernel_type == KERNEL_TYPE_POLYNOMIAL)
        
        # Initialize raw_param1 if applicable (for Polynomial, param1=degree is frozen but slot exists)
        var raw_param1_mf = inv_softplus(kernel_param1) if has_learnable_param1_mf else Float32(0.0)
        # Initialize raw_param2 (offset for Polynomial)
        var raw_param2_mf = inv_softplus(kernel_param2) if has_learnable_param2_mf else Float32(0.0)
        
        var mf_provider = MatrixFreeProvider(
            ctx, x_device.unsafe_ptr(), params_device.unsafe_ptr(),
            n, dim, kernel_type, False,  # use_ard=False
            init_lengthscale, init_outputscale, init_noise,
            kernel_param1, kernel_param2
        )
        
        # Wrap provider in IsotropicGradientAdapter for unified BBMM
        var adapter = IsotropicGradientAdapter(mf_provider^)
        
        # Preconditioner caching: build once, rebuild only when hyperparams change significantly
        var num_cols_total_mf = 1 + num_probes
        var num_kparams_mf = adapter.num_gradient_params()
        bbmm_pool.ensure_capacity(ctx, n, num_cols_total_mf, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kparams_mf)
        var precond_mf = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_mf, precond_method=precond_method)
        var last_rebuild_ls_mf = init_lengthscale
        var last_rebuild_os_mf = init_outputscale
        var last_rebuild_noise_mf = init_noise
        
        # Best-param tracking: snapshot params at best NLL
        var best_raw_ls = raw_lengthscale
        var best_raw_noise = raw_noise
        var best_raw_os = raw_outputscale
        var best_raw_param1 = raw_param1_mf
        var best_raw_param2 = raw_param2_mf
        var best_raw_mean = raw_mean
        var best_nll_seen = Float32(1e10)
        
        for step in range(num_iterations):
            # Transform to constrained space
            var lengthscale: Float32
            var noise: Float32
            var outputscale: Float32
            var param1: Float32
            with ProfileBlock[PROFILING]("ITER_update_hyperparams"):
                lengthscale = softplus(raw_lengthscale)
                noise = softplus(raw_noise)
                outputscale = softplus(raw_outputscale)
                param1 = softplus(raw_param1_mf) if has_learnable_param1_mf else kernel_param1
                
                # Update adapter hyperparameters (delegates to underlying provider)
                if has_learnable_param1_mf:
                    adapter.update_hyperparams_with_param1(lengthscale, outputscale, noise, param1)
                else:
                    adapter.update_hyperparams(lengthscale, outputscale, noise)
                if has_learnable_param2_mf:
                    adapter.update_param2(softplus(raw_param2_mf))
                
                # Adaptive preconditioner rebuild: only rebuild when hyperparams change significantly
                if step > 0:
                    var rel_ls = abs(lengthscale - last_rebuild_ls_mf) / max(abs(last_rebuild_ls_mf), Float32(1e-8))
                    var rel_os = abs(outputscale - last_rebuild_os_mf) / max(abs(last_rebuild_os_mf), Float32(1e-8))
                    var rel_noise = abs(noise - last_rebuild_noise_mf) / max(abs(last_rebuild_noise_mf), Float32(1e-8))
                    var max_rel = max(rel_ls, max(rel_os, rel_noise))
                    if max_rel > precond_rebuild_threshold:
                        precond_mf = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_mf, precond_method=precond_method)
                        last_rebuild_ls_mf = lengthscale
                        last_rebuild_os_mf = outputscale
                        last_rebuild_noise_mf = noise
                
                # ConstantMean: Center y by current mean on GPU (no CPU loop or H2D copy)
                ctx.enqueue_function[kernel_subtract_scalar](
                    y_centered_device.unsafe_ptr(), y_device.unsafe_ptr(), raw_mean, n,
                    grid_dim=((n + 255) // 256,), block_dim=(256,)
                )
                ctx.synchronize()
            
            # BBMM: Compute NLL AND gradients with cached preconditioner
            # ProfileBlock is inside bbmm_with_precond (BBMM_* blocks)
            var bbmm_result = bbmm_with_precond(
                    adapter, precond_mf, y_centered_device.unsafe_ptr(), n, bbmm_pool,
                    num_probes=num_probes,
                    max_iter=max_cg_iter,
                    max_tridiag_iter=max_tridiag_iter,
                    tol=cg_tol,
                    iteration=step,
                recycle_alpha=step > 0,
                use_preconditioner=use_preconditioner,
            )
            ctx.synchronize()
            
            var nll = bbmm_result.nll
            # Extract gradients based on kernel type:
            # - Standard kernels (RBF, Matern): [ls, os, noise]
            # - Param1 kernels (Periodic, RQ, Linear): [ls, os, param1, noise]
            # - Polynomial: [ls, os, param1(0), param2(offset), noise]
            var grad_lengthscale = bbmm_result.gradients[0]
            var grad_outputscale = bbmm_result.gradients[1]
            var grad_param1 = Float32(0.0)
            var grad_param2 = Float32(0.0)
            var grad_noise: Float32
            
            if has_learnable_param2_mf:
                # Polynomial: [ls, os, param1(0), param2(offset), noise]
                grad_param1 = bbmm_result.gradients[2]  # degree gradient (~0)
                grad_param2 = bbmm_result.gradients[3]  # offset gradient
                grad_noise = bbmm_result.gradients[4]
            elif has_learnable_param1_mf:
                grad_param1 = bbmm_result.gradients[2]
                grad_noise = bbmm_result.gradients[3]
            else:
                grad_noise = bbmm_result.gradients[2]
            
            # NaN detection: if NLL is NaN, halve learning rate and skip this step
            if nll != nll:
                lr = lr * Float32(0.5)
                if verbose:
                    print("  NaN NLL detected at step", step, "- halving LR to", lr)
                continue
            
            # Best-param tracking: snapshot params at best NLL
            if nll < best_nll_seen:
                best_nll_seen = nll
                best_raw_ls = raw_lengthscale
                best_raw_noise = raw_noise
                best_raw_os = raw_outputscale
                best_raw_param1 = raw_param1_mf
                best_raw_param2 = raw_param2_mf
                best_raw_mean = raw_mean
            
            with ProfileBlock[PROFILING]("ITER_adam_update"):
                # Clip gradients to prevent NaN/Inf propagation
                grad_lengthscale = clip_gradient(grad_lengthscale)
                grad_noise = clip_gradient(grad_noise)
                grad_outputscale = clip_gradient(grad_outputscale)
                if has_learnable_param1_mf:
                    grad_param1 = clip_gradient(grad_param1)
                if has_learnable_param2_mf:
                    grad_param2 = clip_gradient(grad_param2)
                
                last_nll = nll
                nll_history.append(nll)
                
                # Cosine LR decay: reduce LR smoothly over training
                var effective_lr = lr
                if use_cosine_lr:
                    effective_lr = compute_cosine_lr(learning_rate, step, num_iterations)
                
                # Adam update
                var adam_result = adam_update(
                    adam_state,
                    grad_lengthscale, grad_noise, grad_outputscale,
                    raw_lengthscale, raw_noise, raw_outputscale,
                    effective_lr,
                    0.9, 0.999, 1e-8,  # default Adam params
                    grad_param1, raw_param1_mf, has_learnable_param1_mf,
                    grad_param2, raw_param2_mf, has_learnable_param2_mf
                )
                adam_state = adam_result.state
                raw_lengthscale = adam_result.raw_ls
                raw_noise = adam_result.raw_noise
                raw_outputscale = adam_result.raw_os
                if has_learnable_param1_mf:
                    raw_param1_mf = adam_result.raw_param1
                if has_learnable_param2_mf:
                    raw_param2_mf = adam_result.raw_param2
                ctx.synchronize()
            
            # ConstantMean: Compute mean gradient and Adam update
            # d(NLL)/d(mean) = -sum(alpha) / n, where alpha = K^{-1}(y - mean)
            # alpha is bbmm_result.solution (CG solution for column 0 = y_centered)
            var mean_grad: Float32
            with ProfileBlock[PROFILING]("ITER_mean_gradient"):
                from .cg_solver import kernel_sum_reduce
                var mean_sum_dev = ctx.enqueue_create_buffer[DType.float32](1)
                var mean_sum_host = ctx.enqueue_create_host_buffer[DType.float32](1)
                ctx.enqueue_function[kernel_sum_reduce](
                    mean_sum_dev.unsafe_ptr(), bbmm_result.solution.unsafe_ptr(), n,
                    grid_dim=1, block_dim=256, shared_mem_bytes=8 * 4)
                ctx.enqueue_copy(dst_buf=mean_sum_host, src_buf=mean_sum_dev)
                ctx.synchronize()
                mean_grad = -mean_sum_host.unsafe_ptr()[0] / Float32(n)
                mean_grad = clip_gradient(mean_grad)
                
                # Adam update for mean (unconstrained, no softplus chain rule)
                var beta1 = Float32(0.9)
                var beta2 = Float32(0.999)
                var eps = Float32(1e-8)
                adam_state.m_mean = beta1 * adam_state.m_mean + (Float32(1.0) - beta1) * mean_grad
                adam_state.v_mean = beta2 * adam_state.v_mean + (Float32(1.0) - beta2) * mean_grad * mean_grad
                var step_f = Float32(step + 1)
                var m_hat_mean = adam_state.m_mean / (Float32(1.0) - pow(beta1, step_f))
                var v_hat_mean = adam_state.v_mean / (Float32(1.0) - pow(beta2, step_f))
                raw_mean -= lr * m_hat_mean / (sqrt(v_hat_mean) + eps)
                ctx.synchronize()
            
            # Safety check for mean (allow wider range than kernel params)
            if raw_mean != raw_mean:  # NaN check
                raw_mean = init_mean
            elif raw_mean > Float32(1000.0) or raw_mean < Float32(-1000.0):
                raw_mean = init_mean
            
            # Safety check for NaN/Inf
            if raw_lengthscale != raw_lengthscale or raw_lengthscale > Float32(20.0) or raw_lengthscale < Float32(-20.0):
                raw_lengthscale = Float32(0.0)
            if raw_noise != raw_noise or raw_noise > Float32(20.0) or raw_noise < Float32(-20.0):
                raw_noise = Float32(-2.0)
            if raw_outputscale != raw_outputscale or raw_outputscale > Float32(20.0) or raw_outputscale < Float32(-20.0):
                raw_outputscale = Float32(0.0)
            if has_learnable_param1_mf:
                if raw_param1_mf != raw_param1_mf or raw_param1_mf > Float32(20.0) or raw_param1_mf < Float32(-20.0):
                    raw_param1_mf = Float32(0.0)
            if has_learnable_param2_mf:
                if raw_param2_mf != raw_param2_mf or raw_param2_mf > Float32(20.0) or raw_param2_mf < Float32(-20.0):
                    raw_param2_mf = Float32(0.0)
            
            actual_iterations = step + 1
            
            if verbose and (step % 10 == 0 or step == num_iterations - 1):
                print("Iter", step, "NLL:", nll, 
                      "ℓ:", lengthscale, "σ²:", noise, "σ_f²:", outputscale,
                      "mean:", raw_mean)
                if has_learnable_param1_mf:
                    print("  param1:", param1)
                if has_learnable_param2_mf:
                    var param2_val = softplus(raw_param2_mf)
                    print("  param2 (offset):", param2_val)
            
            # Early stopping check
            if step >= min_iterations:
                if nll < best_nll - abs(best_nll) * early_stop_threshold:
                    best_nll = nll
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= early_stop_patience:
                        converged = True
                        if verbose:
                            print("Early stopping: converged after", actual_iterations, "iterations")
                        break
        
        # Final parameters — use best-seen params, not last-iteration params
        raw_lengthscale = best_raw_ls
        raw_noise = best_raw_noise
        raw_outputscale = best_raw_os
        raw_param1_mf = best_raw_param1
        raw_param2_mf = best_raw_param2
        raw_mean = best_raw_mean
        var final_lengthscale = softplus(raw_lengthscale)
        var final_noise = softplus(raw_noise)
        var final_outputscale = softplus(raw_outputscale)
        var final_param1_mf = softplus(raw_param1_mf) if has_learnable_param1_mf else kernel_param1
        var final_param2_mf = softplus(raw_param2_mf) if has_learnable_param2_mf else kernel_param2
        var final_nll = best_nll_seen
        var final_mean = raw_mean
        
        # Lanczos root deferred to prediction time (predict_with_method recomputes it).
        var lanczos_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # Alpha = K^{-1} @ (y - mean) is deferred to prediction time.
        # The prediction path computes it on demand when has_alpha=False.
        var alpha_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # Keep buffers alive until all pointer users are done
        _ = precond_mf
        _ = y_device
        _ = y_centered_device
        
        return TrainingResult(
            final_lengthscale, final_noise, final_outputscale,
            final_param1_mf, final_param2_mf, final_mean,
            final_nll, nll_history^, actual_iterations, converged,
            lanczos_dummy, 0, n, alpha_dummy, False
        )
    
    # =========================================================================
    # MATERIALIZED PATH: O(n²) memory, pre-computes K for GEMM-based matvecs
    # Uses MaterializedProvider with unified BBMM via IsotropicGradientAdapter
    # =========================================================================
    
    # Check if kernel has learnable param1 (periodic/RQ/linear) or param2 (polynomial offset)
    from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
    var has_learnable_param1 = (kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ or kernel_type == KERNEL_TYPE_LINEAR or kernel_type == KERNEL_TYPE_POLYNOMIAL)
    var has_learnable_param2 = (kernel_type == KERNEL_TYPE_POLYNOMIAL)
    
    # Initialize raw_param1 if applicable (for Polynomial, param1=degree is frozen but slot exists)
    var raw_param1 = inv_softplus(kernel_param1) if has_learnable_param1 else Float32(0.0)
    # Initialize raw_param2 (offset for Polynomial)
    var raw_param2 = inv_softplus(kernel_param2) if has_learnable_param2 else Float32(0.0)
    
    var provider = MaterializedProvider(
        ctx, x_device.unsafe_ptr(), params_device.unsafe_ptr(),
        n, dim, kernel_type, False,  # use_ard=False
        init_lengthscale, init_outputscale, init_noise,
        kernel_param1, kernel_param2
    )
    
    # Wrap provider in IsotropicGradientAdapter for unified BBMM
    var adapter = IsotropicGradientAdapter(provider^)
    
    # Preconditioner caching: build once, rebuild only when hyperparams change significantly
    var num_cols_total_mat = 1 + num_probes
    var num_kparams_mat = adapter.num_gradient_params()
    bbmm_pool.ensure_capacity(ctx, n, num_cols_total_mat, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kparams_mat)
    var precond_mat = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_mat, precond_method=precond_method)
    var last_rebuild_ls_mat = init_lengthscale
    var last_rebuild_os_mat = init_outputscale
    var last_rebuild_noise_mat = init_noise
    
    # Best-param tracking: snapshot params at best NLL
    var best_raw_ls_mat = raw_lengthscale
    var best_raw_noise_mat = raw_noise
    var best_raw_os_mat = raw_outputscale
    var best_raw_param1_mat = raw_param1
    var best_raw_param2_mat = raw_param2
    var best_raw_mean_mat = raw_mean
    var best_nll_seen_mat = Float32(1e10)
    
    for step in range(num_iterations):
        # Transform to constrained space
        var lengthscale: Float32
        var noise: Float32
        var outputscale: Float32
        var param1: Float32
        with ProfileBlock[PROFILING]("ITER_update_hyperparams"):
            lengthscale = softplus(raw_lengthscale)
            noise = softplus(raw_noise)
            outputscale = softplus(raw_outputscale)
            param1 = softplus(raw_param1) if has_learnable_param1 else kernel_param1
            
            # Update adapter hyperparameters (delegates to underlying provider, re-materializes K)
            if has_learnable_param1:
                adapter.update_hyperparams_with_param1(lengthscale, outputscale, noise, param1)
            else:
                adapter.update_hyperparams(lengthscale, outputscale, noise)
            if has_learnable_param2:
                adapter.update_param2(softplus(raw_param2))
            
            # Adaptive preconditioner rebuild: only rebuild when hyperparams change significantly
            if step > 0:
                var rel_ls = abs(lengthscale - last_rebuild_ls_mat) / max(abs(last_rebuild_ls_mat), Float32(1e-8))
                var rel_os = abs(outputscale - last_rebuild_os_mat) / max(abs(last_rebuild_os_mat), Float32(1e-8))
                var rel_noise = abs(noise - last_rebuild_noise_mat) / max(abs(last_rebuild_noise_mat), Float32(1e-8))
                var max_rel = max(rel_ls, max(rel_os, rel_noise))
                if max_rel > precond_rebuild_threshold:
                    precond_mat = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_mat, precond_method=precond_method)
                    last_rebuild_ls_mat = lengthscale
                    last_rebuild_os_mat = outputscale
                    last_rebuild_noise_mat = noise
            
            # ConstantMean: Center y by current mean on GPU (no CPU loop or H2D copy)
            ctx.enqueue_function[kernel_subtract_scalar](
                y_centered_device.unsafe_ptr(), y_device.unsafe_ptr(), raw_mean, n,
                grid_dim=((n + 255) // 256,), block_dim=(256,)
            )
            ctx.synchronize()
        
        # BBMM: Compute NLL AND gradients with cached preconditioner
        # ProfileBlock is inside bbmm_with_precond (BBMM_* blocks)
        var bbmm_result = bbmm_with_precond(
            adapter, precond_mat, y_centered_device.unsafe_ptr(), n, bbmm_pool,
            num_probes=num_probes,
            max_iter=max_cg_iter,
            max_tridiag_iter=max_tridiag_iter,
            tol=cg_tol,
            iteration=step,
            recycle_alpha=step > 0,
            use_preconditioner=use_preconditioner,
        )
        ctx.synchronize()
        
        var nll = bbmm_result.nll
        # Extract gradients based on kernel type:
        # - Standard kernels (RBF, Matern): [ls, os, noise]
        # - Param1 kernels (Periodic, RQ, Linear): [ls, os, param1, noise]
        # - Polynomial: [ls, os, param1(0), param2(offset), noise]
        var grad_lengthscale = bbmm_result.gradients[0]
        var grad_outputscale = bbmm_result.gradients[1]
        var grad_param1 = Float32(0.0)
        var grad_param2 = Float32(0.0)
        var grad_noise: Float32
        
        if has_learnable_param2:
            # Polynomial: [ls, os, param1(0), param2(offset), noise]
            grad_param1 = bbmm_result.gradients[2]  # degree gradient (~0)
            grad_param2 = bbmm_result.gradients[3]  # offset gradient
            grad_noise = bbmm_result.gradients[4]
        elif has_learnable_param1:
            grad_param1 = bbmm_result.gradients[2]
            grad_noise = bbmm_result.gradients[3]
        else:
            grad_noise = bbmm_result.gradients[2]
        
        # NaN detection: if NLL is NaN, halve learning rate and skip this step
        if nll != nll:
            lr = lr * Float32(0.5)
            if verbose:
                print("  NaN NLL detected at step", step, "- halving LR to", lr)
            continue
        
        # Best-param tracking: snapshot params at best NLL
        if nll < best_nll_seen_mat:
            best_nll_seen_mat = nll
            best_raw_ls_mat = raw_lengthscale
            best_raw_noise_mat = raw_noise
            best_raw_os_mat = raw_outputscale
            best_raw_param1_mat = raw_param1
            best_raw_param2_mat = raw_param2
            best_raw_mean_mat = raw_mean
        
        with ProfileBlock[PROFILING]("ITER_adam_update"):
            # Clip gradients to prevent NaN/Inf propagation
            grad_lengthscale = clip_gradient(grad_lengthscale)
            grad_noise = clip_gradient(grad_noise)
            grad_outputscale = clip_gradient(grad_outputscale)
            if has_learnable_param1:
                grad_param1 = clip_gradient(grad_param1)
            if has_learnable_param2:
                grad_param2 = clip_gradient(grad_param2)
            
            last_nll = nll
            nll_history.append(nll)
            
            # Cosine LR decay: reduce LR smoothly over training
            var effective_lr = lr
            if use_cosine_lr:
                effective_lr = compute_cosine_lr(learning_rate, step, num_iterations)
            
            # Adam update
            var adam_result = adam_update(
                adam_state,
                grad_lengthscale, grad_noise, grad_outputscale,
                raw_lengthscale, raw_noise, raw_outputscale,
                effective_lr,
                0.9, 0.999, 1e-8,  # default Adam params
                grad_param1, raw_param1, has_learnable_param1,
                grad_param2, raw_param2, has_learnable_param2
            )
            adam_state = adam_result.state
            raw_lengthscale = adam_result.raw_ls
            raw_noise = adam_result.raw_noise
            raw_outputscale = adam_result.raw_os
            if has_learnable_param1:
                raw_param1 = adam_result.raw_param1
            if has_learnable_param2:
                raw_param2 = adam_result.raw_param2
            ctx.synchronize()
        
        # ConstantMean: Compute mean gradient and Adam update
        var mean_grad: Float32
        with ProfileBlock[PROFILING]("ITER_mean_gradient"):
            from .cg_solver import kernel_sum_reduce
            var mean_sum_dev = ctx.enqueue_create_buffer[DType.float32](1)
            var mean_sum_host = ctx.enqueue_create_host_buffer[DType.float32](1)
            ctx.enqueue_function[kernel_sum_reduce](
                mean_sum_dev.unsafe_ptr(), bbmm_result.solution.unsafe_ptr(), n,
                grid_dim=1, block_dim=256, shared_mem_bytes=8 * 4)
            ctx.enqueue_copy(dst_buf=mean_sum_host, src_buf=mean_sum_dev)
            ctx.synchronize()
            mean_grad = -mean_sum_host.unsafe_ptr()[0] / Float32(n)
            mean_grad = clip_gradient(mean_grad)
            
            # Adam update for mean (unconstrained, no softplus chain rule)
            var beta1 = Float32(0.9)
            var beta2 = Float32(0.999)
            var eps = Float32(1e-8)
            adam_state.m_mean = beta1 * adam_state.m_mean + (Float32(1.0) - beta1) * mean_grad
            adam_state.v_mean = beta2 * adam_state.v_mean + (Float32(1.0) - beta2) * mean_grad * mean_grad
            var step_f = Float32(step + 1)
            var m_hat_mean = adam_state.m_mean / (Float32(1.0) - pow(beta1, step_f))
            var v_hat_mean = adam_state.v_mean / (Float32(1.0) - pow(beta2, step_f))
            raw_mean -= lr * m_hat_mean / (sqrt(v_hat_mean) + eps)
            ctx.synchronize()
        
        # Safety check for mean
        if raw_mean != raw_mean:
            raw_mean = init_mean
        elif raw_mean > Float32(1000.0) or raw_mean < Float32(-1000.0):
            raw_mean = init_mean
        
        # Safety check for NaN/Inf
        if raw_lengthscale != raw_lengthscale or raw_lengthscale > Float32(20.0) or raw_lengthscale < Float32(-20.0):
            raw_lengthscale = Float32(0.0)
        if raw_noise != raw_noise or raw_noise > Float32(20.0) or raw_noise < Float32(-20.0):
            raw_noise = Float32(-2.0)
        if raw_outputscale != raw_outputscale or raw_outputscale > Float32(20.0) or raw_outputscale < Float32(-20.0):
            raw_outputscale = Float32(0.0)
        if has_learnable_param1:
            if raw_param1 != raw_param1 or raw_param1 > Float32(20.0) or raw_param1 < Float32(-20.0):
                raw_param1 = Float32(0.0)
        if has_learnable_param2:
            if raw_param2 != raw_param2 or raw_param2 > Float32(20.0) or raw_param2 < Float32(-20.0):
                raw_param2 = Float32(0.0)
        
        actual_iterations = step + 1
        
        if verbose and (step % 10 == 0 or step == num_iterations - 1):
            print("Iter", step, "NLL:", nll, 
                  "ℓ:", lengthscale, "σ²:", noise, "σ_f²:", outputscale,
                  "mean:", raw_mean)
            if has_learnable_param1:
                print("  param1:", param1)
            if has_learnable_param2:
                var param2_val = softplus(raw_param2)
                print("  param2 (offset):", param2_val)
        
        # Early stopping check
        if step >= min_iterations:
            if nll < best_nll - abs(best_nll) * early_stop_threshold:
                best_nll = nll
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    converged = True
                    if verbose:
                        print("Early stopping: converged after", actual_iterations, "iterations")
                    break
    
    # Final parameters — use best-seen params, not last-iteration params
    raw_lengthscale = best_raw_ls_mat
    raw_noise = best_raw_noise_mat
    raw_outputscale = best_raw_os_mat
    raw_param1 = best_raw_param1_mat
    raw_param2 = best_raw_param2_mat
    raw_mean = best_raw_mean_mat
    var final_lengthscale = softplus(raw_lengthscale)
    var final_noise = softplus(raw_noise)
    var final_outputscale = softplus(raw_outputscale)
    var final_param1 = softplus(raw_param1) if has_learnable_param1 else kernel_param1
    var final_param2 = softplus(raw_param2) if has_learnable_param2 else kernel_param2
    var final_nll = best_nll_seen_mat
    var final_mean = raw_mean
    
    # Lanczos root deferred to prediction time (predict_with_method recomputes it).
    var lanczos_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    # Alpha = K^{-1} @ (y - mean) is deferred to prediction time.
    var alpha_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    # Keep buffers alive until all pointer users are done
    _ = precond_mat
    _ = y_device
    _ = y_centered_device
    
    return TrainingResult(
        final_lengthscale, final_noise, final_outputscale,
        final_param1, final_param2, final_mean,
        final_nll, nll_history^, actual_iterations, converged,
        lanczos_dummy, 0, n, alpha_dummy, False
    )


# =============================================================================
# ARD Training Function
# =============================================================================

fn train_gp_ard(
    ctx: DeviceContext,
    x_host: HostBuffer[float_dtype],
    y_host: HostBuffer[float_dtype],
    n: Int,
    dim: Int,
    kernel_type: Int,
    init_lengthscales_host: HostBuffer[float_dtype],  # [dim] initial lengthscales
    num_iterations: Int = 100,
    learning_rate: Float32 = 0.05,
    init_noise: Float32 = 0.1,
    init_outputscale: Float32 = 1.0,
    kernel_param1: Float32 = 1.0,
    kernel_param2: Float32 = 0.0,
    early_stop_patience: Int = 30,
    early_stop_threshold: Float32 = 1e-4,
    min_iterations: Int = 50,
    verbose: Bool = True,
    precond_type_str: String = "pivoted_cholesky",
    method: String = "auto",
    init_mean: Float32 = 0.0,
    # CG parameters (configurable from Python via presets)
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    max_tridiag_iter: Int = 30,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    precond_rebuild_threshold: Float32 = 0.5,
    # Preconditioner method: 0=greedy, 1=rpcholesky, 2=nystrom (default)
    precond_method: Int = 2,
    # Learning rate schedule: True=cosine decay, False=constant
    use_cosine_lr: Bool = True,
) raises -> TrainingResultARD:
    """Train GP with ARD (Automatic Relevance Determination).
    
    This function trains a GP with per-dimension lengthscales, allowing the model
    to automatically learn which input dimensions are most relevant for prediction.
    
    Uses BBMM (Blackbox Matrix-Matrix) inference with Pivoted Cholesky preconditioning.
    
    Args:
        ctx: GPU device context
        x_host: Training data [n, dim] row-major
        y_host: Training targets [n]
        n: Number of training points
        dim: Input dimension
        kernel_type: Kernel type constant
        init_lengthscales_host: Initial lengthscales [dim]
        num_iterations: Maximum training iterations
        learning_rate: Adam learning rate
        init_noise: Initial noise variance
        init_outputscale: Initial output scale
        kernel_param1: Extra kernel parameter 1
        kernel_param2: Extra kernel parameter 2
        early_stop_patience: Iterations without improvement before stopping
        early_stop_threshold: Minimum relative improvement
        min_iterations: Minimum iterations before early stopping
        verbose: Print progress
        precond_type_str: Preconditioner type ("pivoted_cholesky", "jacobi", "none")
        method: Training method ("auto", "materialized", "matrix_free")
                - "auto": Use materialized for n < 10000, matrix_free otherwise
                - "materialized": O(n²) memory, uses cached kernel matrix
                - "matrix_free": O(n) memory, computes kernel on-the-fly
        
    Returns:
        TrainingResultARD with learned per-dimension lengthscales
    """
    # Determine which method to use
    var use_materialized = False
    if method == "materialized" or method == "fast":
        use_materialized = True
    elif method == "auto":
        use_materialized = (n < 10000)
    # else: matrix_free
    
    if use_materialized:
        return _train_gp_ard_materialized(
            ctx, x_host, y_host, n, dim, kernel_type,
            init_lengthscales_host, num_iterations, learning_rate,
            init_noise, init_outputscale, kernel_param1, kernel_param2,
            early_stop_patience, early_stop_threshold, min_iterations,
            verbose, precond_type_str, init_mean,
            num_probes, max_cg_iter, max_tridiag_iter, cg_tol, precond_rank, precond_rebuild_threshold,
            precond_method, use_cosine_lr,
        )
    else:
        return _train_gp_ard_matrix_free(
            ctx, x_host, y_host, n, dim, kernel_type,
            init_lengthscales_host, num_iterations, learning_rate,
            init_noise, init_outputscale, kernel_param1, kernel_param2,
            early_stop_patience, early_stop_threshold, min_iterations,
            verbose, precond_type_str, init_mean,
            num_probes, max_cg_iter, max_tridiag_iter, cg_tol, precond_rank, precond_rebuild_threshold,
            precond_method, use_cosine_lr,
        )


fn _train_gp_ard_materialized(
    ctx: DeviceContext,
    x_host: HostBuffer[float_dtype],
    y_host: HostBuffer[float_dtype],
    n: Int,
    dim: Int,
    kernel_type: Int,
    init_lengthscales_host: HostBuffer[float_dtype],
    num_iterations: Int,
    learning_rate: Float32,
    init_noise: Float32,
    init_outputscale: Float32,
    kernel_param1: Float32,
    kernel_param2: Float32,
    early_stop_patience: Int,
    early_stop_threshold: Float32,
    min_iterations: Int,
    verbose: Bool,
    precond_type_str: String,
    init_mean: Float32 = 0.0,
    # CG parameters
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    max_tridiag_iter: Int = 30,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    precond_rebuild_threshold: Float32 = 0.5,
    # Preconditioner method: 0=greedy, 1=rpcholesky, 2=nystrom (default)
    precond_method: Int = 2,
    # Learning rate schedule: True=cosine decay, False=constant
    use_cosine_lr: Bool = True,
) raises -> TrainingResultARD:
    """Internal: ARD training with MaterializedProvider (O(n²) memory)."""
    # Copy x to device
    var x_device = ctx.enqueue_create_buffer[float_dtype](n * dim)
    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    
    # Copy y to device
    var y_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.synchronize()
    
    # ConstantMean: Initialize mean parameter (unconstrained, no softplus)
    var raw_mean = init_mean
    
    # ConstantMean: Create y_centered buffers for centering y by current mean
    var y_host_centered = ctx.enqueue_create_host_buffer[float_dtype](n)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Initialize raw (unconstrained) parameters
    var raw_lengthscales = List[Float32]()
    for d in range(dim):
        raw_lengthscales.append(inv_softplus(init_lengthscales_host[d]))
    var raw_noise = inv_softplus(init_noise)
    var raw_outputscale = inv_softplus(init_outputscale)
    var raw_param1 = inv_softplus(kernel_param1)  # For Periodic/RQ/Polynomial param1
    
    # Check if kernel needs param1 optimization.
    # NOTE: Linear excluded -- in ARD mode, per-dim variance weights replace param1.
    var has_param1 = kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ or kernel_type == KERNEL_TYPE_POLYNOMIAL
    # Check if kernel needs param2 optimization (Polynomial offset)
    var has_param2 = kernel_type == KERNEL_TYPE_POLYNOMIAL
    var raw_param2 = inv_softplus(kernel_param2) if has_param2 else Float32(0.0)
    
    # Initialize Adam state for ARD
    var adam_state = AdamStateARD(dim)
    
    # Early stopping state
    var best_nll = Float32(1e10)
    var no_improve_count = 0
    var converged = False
    var actual_iterations = 0
    
    # Track NLL history
    var nll_history_ard = List[Float32]()
    
    # Create parameter buffer on device (not used for ARD, but needed for provider)
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = ctx.enqueue_create_host_buffer[float_dtype](2)
    params_host_temp[0] = init_lengthscales_host[0]
    params_host_temp[1] = init_outputscale
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
    ctx.synchronize()
    
    # Create MaterializedProvider with ARD enabled
    var provider = MaterializedProvider(
        ctx, x_device.unsafe_ptr(), params_device.unsafe_ptr(),
        n, dim, kernel_type, True,  # use_ard=True
        init_lengthscales_host[0], init_outputscale, init_noise,
        kernel_param1, kernel_param2
    )
    
    # Set initial ARD lengthscales on device
    provider.set_lengthscales_device(init_lengthscales_host.unsafe_ptr())
    
    # Wrap provider in ARDGradientAdapter for unified BBMM
    var adapter = ARDGradientAdapter(provider^, dim)
    
    # Warmup GPU kernels
    if verbose:
        print("Warming up GPU kernels...")
    _warmup_gpu_kernels(ctx)
    if verbose:
        print("Warmup complete.")
    
    # Create buffer pool
    var bbmm_pool = CGBufferPool(ctx)
    
    # Track last NLL
    var last_nll = Float32(0.0)
    
    # Mutable copy of learning_rate (function params are immutable in Mojo)
    var lr = learning_rate
    
    # Preconditioner caching: build once, rebuild only when hyperparams change significantly
    var num_cols_total_ard = 1 + num_probes
    var num_kparams_ard = adapter.num_gradient_params()
    bbmm_pool.ensure_capacity(ctx, n, num_cols_total_ard, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kparams_ard)
    var precond_ard = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_ard, precond_method=precond_method)
    var last_rebuild_os_ard = init_outputscale
    var last_rebuild_noise_ard = init_noise
    # Track max lengthscale for rebuild threshold (use max of initial lengthscales)
    var last_rebuild_max_ls_ard = Float32(0.0)
    for d in range(dim):
        last_rebuild_max_ls_ard = max(last_rebuild_max_ls_ard, init_lengthscales_host[d])
    
    # Pre-allocate host buffers reused every iteration
    var ls_host = ctx.enqueue_create_host_buffer[float_dtype](dim)
    
    # Best-param tracking: snapshot params at best NLL
    var best_raw_ls_ard = List[Float32]()
    for d in range(dim):
        best_raw_ls_ard.append(raw_lengthscales[d])
    var best_raw_noise_ard = raw_noise
    var best_raw_os_ard = raw_outputscale
    var best_raw_param1_ard = raw_param1
    var best_raw_param2_ard = raw_param2
    var best_raw_mean_ard = raw_mean
    var best_nll_seen_ard = Float32(1e10)
    
    # Training loop (materialized ARD)
    for step in range(num_iterations):
        # Transform to constrained space
        var lengthscales = List[Float32]()
        for d in range(dim):
            lengthscales.append(softplus(raw_lengthscales[d]))
        var noise = softplus(raw_noise)
        var outputscale = softplus(raw_outputscale)
        var param1 = softplus(raw_param1) if has_param1 else kernel_param1
        
        # Update provider with current hyperparameters (including param1 for RQ/Periodic)
        for d in range(dim):
            ls_host[d] = lengthscales[d]
        adapter.provider.update_hyperparams_ard(ls_host.unsafe_ptr(), outputscale, noise)
        if has_param1:
            adapter.provider.update_param1(param1)
        if has_param2:
            adapter.provider.update_param2(softplus(raw_param2))
        
        # Adaptive preconditioner rebuild: only rebuild when hyperparams change significantly
        if step > 0:
            var rel_os = abs(outputscale - last_rebuild_os_ard) / max(abs(last_rebuild_os_ard), Float32(1e-8))
            var rel_noise = abs(noise - last_rebuild_noise_ard) / max(abs(last_rebuild_noise_ard), Float32(1e-8))
            var max_rel = max(rel_os, rel_noise)
            # Check max relative change across all lengthscales
            for d in range(dim):
                var cur_ls = lengthscales[d]
                # Compare against max ls at last rebuild (conservative: tracks overall scale change)
                var rel_ls_d = abs(cur_ls - last_rebuild_max_ls_ard) / max(abs(last_rebuild_max_ls_ard), Float32(1e-8))
                max_rel = max(max_rel, rel_ls_d)
            if max_rel > precond_rebuild_threshold:
                precond_ard = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_ard, precond_method=precond_method)
                last_rebuild_os_ard = outputscale
                last_rebuild_noise_ard = noise
                last_rebuild_max_ls_ard = Float32(0.0)
                for d in range(dim):
                    last_rebuild_max_ls_ard = max(last_rebuild_max_ls_ard, lengthscales[d])
        
        # ConstantMean: Center y by current mean on GPU (no CPU loop or H2D copy)
        ctx.enqueue_function[kernel_subtract_scalar](
            y_centered_device.unsafe_ptr(), y_device.unsafe_ptr(), raw_mean, n,
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        
        # BBMM with ARD gradients using cached preconditioner
        var bbmm_result = bbmm_with_precond(
            adapter, precond_ard, y_centered_device.unsafe_ptr(), n, bbmm_pool,
            num_probes=num_probes,
            max_iter=max_cg_iter,
            max_tridiag_iter=max_tridiag_iter,
            tol=cg_tol,
            iteration=step,
            recycle_alpha=step > 0,
            use_preconditioner=use_preconditioner,
        )
        
        var nll = bbmm_result.nll
        
        # Extract gradients from unified result
        # gradients[0..d-1] = per-dimension lengthscale gradients
        # gradients[d] = outputscale gradient
        # For Periodic/RQ/Linear: gradients[d+1] = param1, gradients[d+2] = noise
        # For Polynomial: gradients[d+1] = param1(0), gradients[d+2] = param2(offset), gradients[d+3] = noise
        # For RBF/Matern: gradients[d+1] = noise
        var grad_outputscale = bbmm_result.gradients[dim]
        var grad_noise: Float32
        var grad_param1: Float32 = Float32(0.0)
        var grad_param2: Float32 = Float32(0.0)
        if has_param2:
            # Polynomial: [ls..., os, param1(0), param2(offset), noise]
            grad_param1 = bbmm_result.gradients[dim + 1]  # degree gradient (~0)
            grad_param2 = bbmm_result.gradients[dim + 2]  # offset gradient
            grad_noise = bbmm_result.gradients[dim + 3]
        elif has_param1:
            grad_param1 = bbmm_result.gradients[dim + 1]
            grad_noise = bbmm_result.gradients[dim + 2]
        else:
            grad_noise = bbmm_result.gradients[dim + 1]
        
        # Save NLL for final result and history
        last_nll = nll
        nll_history_ard.append(nll)
        
        # NaN detection: if NLL is NaN, halve learning rate and skip this step
        if nll != nll:
            lr = lr * Float32(0.5)
            if verbose:
                print("  NaN NLL detected at step", step, "- halving LR to", lr)
            continue
        
        # Best-param tracking: snapshot params at best NLL
        if nll < best_nll_seen_ard:
            best_nll_seen_ard = nll
            for d in range(dim):
                best_raw_ls_ard[d] = raw_lengthscales[d]
            best_raw_noise_ard = raw_noise
            best_raw_os_ard = raw_outputscale
            best_raw_param1_ard = raw_param1
            best_raw_param2_ard = raw_param2
            best_raw_mean_ard = raw_mean
        
        # Cosine LR decay
        var effective_lr = lr
        if use_cosine_lr:
            effective_lr = compute_cosine_lr(learning_rate, step, num_iterations)
        
        # Adam update (ARD) - inline to avoid struct move issues
        var beta1 = Float32(0.9)
        var beta2 = Float32(0.999)
        var eps = Float32(1e-8)
        alias MAX_RAW = Float32(20.0)
        alias MIN_RAW = Float32(-20.0)
        adam_state.t += 1
        var t = Float32(adam_state.t)
        
        # Update each lengthscale dimension
        for d_idx in range(dim):
            var clipped_grad = clip_gradient(bbmm_result.gradients[d_idx])
            var grad_raw = clipped_grad * softplus_derivative(raw_lengthscales[d_idx])
            adam_state.m_ls[d_idx] = beta1 * adam_state.m_ls[d_idx] + (Float32(1.0) - beta1) * grad_raw
            adam_state.v_ls[d_idx] = beta2 * adam_state.v_ls[d_idx] + (Float32(1.0) - beta2) * grad_raw * grad_raw
            var m_hat = adam_state.m_ls[d_idx] / (Float32(1.0) - pow_float32(beta1, Int(t)))
            var v_hat = adam_state.v_ls[d_idx] / (Float32(1.0) - pow_float32(beta2, Int(t)))
            raw_lengthscales[d_idx] = raw_lengthscales[d_idx] - effective_lr * m_hat / (sqrt(v_hat) + eps)
            raw_lengthscales[d_idx] = max(MIN_RAW, min(MAX_RAW, raw_lengthscales[d_idx]))
        
        # Update noise
        var clipped_grad_noise = clip_gradient(grad_noise)
        var grad_raw_noise = clipped_grad_noise * softplus_derivative(raw_noise)
        adam_state.m_noise = beta1 * adam_state.m_noise + (Float32(1.0) - beta1) * grad_raw_noise
        adam_state.v_noise = beta2 * adam_state.v_noise + (Float32(1.0) - beta2) * grad_raw_noise * grad_raw_noise
        var m_hat_noise = adam_state.m_noise / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_noise = adam_state.v_noise / (Float32(1.0) - pow_float32(beta2, Int(t)))
        raw_noise = raw_noise - effective_lr * m_hat_noise / (sqrt(v_hat_noise) + eps)
        raw_noise = max(MIN_RAW, min(MAX_RAW, raw_noise))
        
        # Update output scale
        var clipped_grad_os = clip_gradient(grad_outputscale)
        var grad_raw_os = clipped_grad_os * softplus_derivative(raw_outputscale)
        adam_state.m_os = beta1 * adam_state.m_os + (Float32(1.0) - beta1) * grad_raw_os
        adam_state.v_os = beta2 * adam_state.v_os + (Float32(1.0) - beta2) * grad_raw_os * grad_raw_os
        var m_hat_os = adam_state.m_os / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_os = adam_state.v_os / (Float32(1.0) - pow_float32(beta2, Int(t)))
        raw_outputscale = raw_outputscale - effective_lr * m_hat_os / (sqrt(v_hat_os) + eps)
        raw_outputscale = max(MIN_RAW, min(MAX_RAW, raw_outputscale))
        
        # Update param1 (period/alpha) for Periodic/RQ kernels
        if has_param1:
            var clipped_grad_p1 = clip_gradient(grad_param1)
            var grad_raw_p1 = clipped_grad_p1 * softplus_derivative(raw_param1)
            adam_state.m_param1 = beta1 * adam_state.m_param1 + (Float32(1.0) - beta1) * grad_raw_p1
            adam_state.v_param1 = beta2 * adam_state.v_param1 + (Float32(1.0) - beta2) * grad_raw_p1 * grad_raw_p1
            var m_hat_p1 = adam_state.m_param1 / (Float32(1.0) - pow_float32(beta1, Int(t)))
            var v_hat_p1 = adam_state.v_param1 / (Float32(1.0) - pow_float32(beta2, Int(t)))
            raw_param1 = raw_param1 - effective_lr * m_hat_p1 / (sqrt(v_hat_p1) + eps)
            raw_param1 = max(MIN_RAW, min(MAX_RAW, raw_param1))
        
        # Update param2 (offset) for Polynomial kernel
        if has_param2:
            var clipped_grad_p2 = clip_gradient(grad_param2)
            var grad_raw_p2 = clipped_grad_p2 * softplus_derivative(raw_param2)
            adam_state.m_param2 = beta1 * adam_state.m_param2 + (Float32(1.0) - beta1) * grad_raw_p2
            adam_state.v_param2 = beta2 * adam_state.v_param2 + (Float32(1.0) - beta2) * grad_raw_p2 * grad_raw_p2
            var m_hat_p2 = adam_state.m_param2 / (Float32(1.0) - pow_float32(beta1, Int(t)))
            var v_hat_p2 = adam_state.v_param2 / (Float32(1.0) - pow_float32(beta2, Int(t)))
            raw_param2 = raw_param2 - effective_lr * m_hat_p2 / (sqrt(v_hat_p2) + eps)
            raw_param2 = max(MIN_RAW, min(MAX_RAW, raw_param2))
        
        # ConstantMean: Compute mean gradient and Adam update
        # d(NLL)/d(mean) = -sum(alpha) / n, where alpha = K^{-1}(y - mean)
        # alpha is bbmm_result.solution (CG solution for column 0 = y_centered)
        from .cg_solver import kernel_sum_reduce
        var mean_sum_dev = ctx.enqueue_create_buffer[DType.float32](1)
        var mean_sum_host = ctx.enqueue_create_host_buffer[DType.float32](1)
        ctx.enqueue_function[kernel_sum_reduce](
            mean_sum_dev.unsafe_ptr(), bbmm_result.solution.unsafe_ptr(), n,
            grid_dim=1, block_dim=256, shared_mem_bytes=8 * 4)
        ctx.enqueue_copy(dst_buf=mean_sum_host, src_buf=mean_sum_dev)
        ctx.synchronize()
        var mean_grad = -mean_sum_host.unsafe_ptr()[0] / Float32(n)
        mean_grad = clip_gradient(mean_grad)
        
        # Adam update for mean (unconstrained, no softplus chain rule)
        adam_state.m_mean = beta1 * adam_state.m_mean + (Float32(1.0) - beta1) * mean_grad
        adam_state.v_mean = beta2 * adam_state.v_mean + (Float32(1.0) - beta2) * mean_grad * mean_grad
        var m_hat_mean = adam_state.m_mean / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_mean = adam_state.v_mean / (Float32(1.0) - pow_float32(beta2, Int(t)))
        raw_mean -= effective_lr * m_hat_mean / (sqrt(v_hat_mean) + eps)
        
        # Safety check for mean (allow wider range than kernel params)
        if raw_mean != raw_mean:  # NaN check
            raw_mean = init_mean
        elif raw_mean > Float32(1000.0) or raw_mean < Float32(-1000.0):
            raw_mean = init_mean
        
        actual_iterations = step + 1
        
        # Print progress
        if verbose and (step % 10 == 0 or step == num_iterations - 1):
            print("Iter", step, "NLL:", nll, "σ²:", noise, "σ_f²:", outputscale, "mean:", raw_mean)
            if has_param1:
                print("  param1:", param1)
            if has_param2:
                var param2_val = softplus(raw_param2)
                print("  param2 (offset):", param2_val)
            # Print first few lengthscales
            if dim <= 5:
                for d in range(dim):
                    print("  ℓ[", d, "]:", lengthscales[d])
            else:
                for d in range(3):
                    print("  ℓ[", d, "]:", lengthscales[d])
                print("  ...")
        
        # Early stopping check
        if step >= min_iterations:
            if nll < best_nll - abs(best_nll) * early_stop_threshold:
                best_nll = nll
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    converged = True
                    if verbose:
                        print("Early stopping: converged after", actual_iterations, "iterations")
                    break
    
    # Restore best-seen params before computing final parameters
    for d in range(dim):
        raw_lengthscales[d] = best_raw_ls_ard[d]
    raw_noise = best_raw_noise_ard
    raw_outputscale = best_raw_os_ard
    raw_param1 = best_raw_param1_ard
    raw_param2 = best_raw_param2_ard
    raw_mean = best_raw_mean_ard
    
    # Final parameters
    var final_lengthscales = ctx.enqueue_create_host_buffer[float_dtype](dim)
    for d in range(dim):
        final_lengthscales[d] = softplus(raw_lengthscales[d])
    var final_noise = softplus(raw_noise)
    var final_outputscale = softplus(raw_outputscale)
    var final_param1 = softplus(raw_param1) if has_param1 else kernel_param1
    var final_param2 = softplus(raw_param2) if has_param2 else kernel_param2
    var final_mean = raw_mean
    var final_nll = best_nll_seen_ard
    
    # Lanczos root deferred to prediction time (predict_with_method recomputes it).
    var lanczos_dummy_ard = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    # Alpha = K^{-1} @ (y - mean) is deferred to prediction time.
    var alpha_dummy_ard = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    # Keep buffers alive until all pointer users are done
    _ = precond_ard
    _ = y_device
    _ = y_centered_device
    
    return TrainingResultARD(
        final_lengthscales, final_noise, final_outputscale,
        final_param1, final_param2, final_mean,
        final_nll, nll_history_ard^, actual_iterations, converged,
        lanczos_dummy_ard, 0, n, dim, alpha_dummy_ard, False
    )


fn _train_gp_ard_matrix_free(
    ctx: DeviceContext,
    x_host: HostBuffer[float_dtype],
    y_host: HostBuffer[float_dtype],
    n: Int,
    dim: Int,
    kernel_type: Int,
    init_lengthscales_host: HostBuffer[float_dtype],
    num_iterations: Int,
    learning_rate: Float32,
    init_noise: Float32,
    init_outputscale: Float32,
    kernel_param1: Float32,
    kernel_param2: Float32,
    early_stop_patience: Int,
    early_stop_threshold: Float32,
    min_iterations: Int,
    verbose: Bool,
    precond_type_str: String,
    init_mean: Float32 = 0.0,
    # CG parameters
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    max_tridiag_iter: Int = 30,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    precond_rebuild_threshold: Float32 = 0.5,
    # Preconditioner method: 0=greedy, 1=rpcholesky, 2=nystrom (default)
    precond_method: Int = 2,
    # Learning rate schedule: True=cosine decay, False=constant
    use_cosine_lr: Bool = True,
) raises -> TrainingResultARD:
    """Internal: ARD training with MatrixFreeProvider (O(n) memory)."""
    # Copy x to device
    var x_device = ctx.enqueue_create_buffer[float_dtype](n * dim)
    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    
    # Copy y to device
    var y_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.synchronize()
    
    # ConstantMean: Initialize mean parameter (unconstrained, no softplus)
    var raw_mean = init_mean
    
    # ConstantMean: Create y_centered buffers for centering y by current mean
    var y_host_centered = ctx.enqueue_create_host_buffer[float_dtype](n)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Initialize raw (unconstrained) parameters
    var raw_lengthscales = List[Float32]()
    for d in range(dim):
        raw_lengthscales.append(inv_softplus(init_lengthscales_host[d]))
    var raw_noise = inv_softplus(init_noise)
    var raw_outputscale = inv_softplus(init_outputscale)
    var raw_param1 = inv_softplus(kernel_param1)  # For Periodic/RQ/Polynomial param1
    
    # Check if kernel needs param1 optimization.
    # NOTE: Linear excluded -- in ARD mode, per-dim variance weights replace param1.
    var has_param1 = kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ or kernel_type == KERNEL_TYPE_POLYNOMIAL
    # Check if kernel needs param2 optimization (Polynomial offset)
    var has_param2 = kernel_type == KERNEL_TYPE_POLYNOMIAL
    var raw_param2 = inv_softplus(kernel_param2) if has_param2 else Float32(0.0)
    
    # Initialize Adam state for ARD
    var adam_state = AdamStateARD(dim)
    
    # Early stopping state
    var best_nll = Float32(1e10)
    var no_improve_count = 0
    var converged = False
    var actual_iterations = 0
    
    # Track NLL history
    var nll_history_ard_mf = List[Float32]()
    
    # Create parameter buffer on device (not used for ARD, but needed for provider)
    var params_device = ctx.enqueue_create_buffer[float_dtype](2)
    var params_host_temp = ctx.enqueue_create_host_buffer[float_dtype](2)
    params_host_temp[0] = init_lengthscales_host[0]
    params_host_temp[1] = init_outputscale
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host_temp)
    ctx.synchronize()
    
    # Create MatrixFreeProvider with ARD enabled
    var provider = MatrixFreeProvider(
        ctx, x_device.unsafe_ptr(), params_device.unsafe_ptr(),
        n, dim, kernel_type, True,  # use_ard=True
        init_lengthscales_host[0], init_outputscale, init_noise,
        kernel_param1, kernel_param2
    )
    
    # Set initial ARD lengthscales on device
    provider.set_lengthscales_device(init_lengthscales_host.unsafe_ptr())
    
    # Wrap provider in ARDGradientAdapter for unified BBMM
    var adapter = ARDGradientAdapter(provider^, dim)
    
    # Warmup GPU kernels
    if verbose:
        print("Warming up GPU kernels...")
    _warmup_gpu_kernels(ctx)
    if verbose:
        print("Warmup complete.")
    
    # Create buffer pool
    var bbmm_pool = CGBufferPool(ctx)
    
    # Track last NLL
    var last_nll = Float32(0.0)
    
    # Mutable copy of learning_rate (function params are immutable in Mojo)
    var lr = learning_rate
    
    # Preconditioner caching for ARD matrix-free path
    var num_cols_total_ard_mf = 1 + num_probes
    var num_kparams_ard_mf = adapter.num_gradient_params()
    bbmm_pool.ensure_capacity(ctx, n, num_cols_total_ard_mf, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kparams_ard_mf)
    var precond_ard_mf = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_ard_mf, precond_method=precond_method)
    var last_rebuild_os_ard_mf = init_outputscale
    var last_rebuild_noise_ard_mf = init_noise
    var last_rebuild_max_ls_ard_mf = Float32(0.0)
    for d in range(dim):
        last_rebuild_max_ls_ard_mf = max(last_rebuild_max_ls_ard_mf, init_lengthscales_host[d])
    
    # Pre-allocate host buffers reused every iteration
    var ls_host_mf = ctx.enqueue_create_host_buffer[float_dtype](dim)
    
    # Best-param tracking: snapshot params at best NLL
    var best_raw_ls_mf = List[Float32]()
    for d in range(dim):
        best_raw_ls_mf.append(raw_lengthscales[d])
    var best_raw_noise_mf = raw_noise
    var best_raw_os_mf = raw_outputscale
    var best_raw_param1_mf = raw_param1
    var best_raw_param2_mf = raw_param2
    var best_raw_mean_mf = raw_mean
    var best_nll_seen_mf = Float32(1e10)
    
    # Training loop
    for step in range(num_iterations):
        # Transform to constrained space
        var lengthscales = List[Float32]()
        for d in range(dim):
            lengthscales.append(softplus(raw_lengthscales[d]))
        var noise = softplus(raw_noise)
        var outputscale = softplus(raw_outputscale)
        var param1 = softplus(raw_param1) if has_param1 else kernel_param1
        
        # Update provider with current hyperparameters (including param1 for RQ/Periodic)
        for d in range(dim):
            ls_host_mf[d] = lengthscales[d]
        adapter.provider.update_hyperparams_ard(ls_host_mf.unsafe_ptr(), outputscale, noise)
        if has_param1:
            adapter.provider.update_param1(param1)
        if has_param2:
            adapter.provider.update_param2(softplus(raw_param2))
        
        # Adaptive preconditioner rebuild
        if step > 0:
            var rel_os = abs(outputscale - last_rebuild_os_ard_mf) / max(abs(last_rebuild_os_ard_mf), Float32(1e-8))
            var rel_noise = abs(noise - last_rebuild_noise_ard_mf) / max(abs(last_rebuild_noise_ard_mf), Float32(1e-8))
            var max_rel = max(rel_os, rel_noise)
            for d in range(dim):
                var rel_ls_d = abs(lengthscales[d] - last_rebuild_max_ls_ard_mf) / max(abs(last_rebuild_max_ls_ard_mf), Float32(1e-8))
                max_rel = max(max_rel, rel_ls_d)
            if max_rel > precond_rebuild_threshold:
                precond_ard_mf = build_pivoted_cholesky_precond_unified(adapter, rank=precond_rank, max_num_cols=num_cols_total_ard_mf, precond_method=precond_method)
                last_rebuild_os_ard_mf = outputscale
                last_rebuild_noise_ard_mf = noise
                last_rebuild_max_ls_ard_mf = Float32(0.0)
                for d in range(dim):
                    last_rebuild_max_ls_ard_mf = max(last_rebuild_max_ls_ard_mf, lengthscales[d])
        
        # ConstantMean: Center y by current mean on GPU (no CPU loop or H2D copy)
        ctx.enqueue_function[kernel_subtract_scalar](
            y_centered_device.unsafe_ptr(), y_device.unsafe_ptr(), raw_mean, n,
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        
        # BBMM with ARD gradients using cached preconditioner
        var bbmm_result = bbmm_with_precond(
            adapter, precond_ard_mf, y_centered_device.unsafe_ptr(), n, bbmm_pool,
            num_probes=num_probes,
            max_iter=max_cg_iter,
            max_tridiag_iter=max_tridiag_iter,
            tol=cg_tol,
            iteration=step,
            recycle_alpha=step > 0,
            use_preconditioner=use_preconditioner,
        )
        
        var nll = bbmm_result.nll
        # Extract gradients from unified result
        var grad_outputscale = bbmm_result.gradients[dim]
        var grad_noise: Float32
        var grad_param1: Float32 = Float32(0.0)
        var grad_param2: Float32 = Float32(0.0)
        if has_param2:
            # Polynomial: [ls..., os, param1(0), param2(offset), noise]
            grad_param1 = bbmm_result.gradients[dim + 1]  # degree gradient (~0)
            grad_param2 = bbmm_result.gradients[dim + 2]  # offset gradient
            grad_noise = bbmm_result.gradients[dim + 3]
        elif has_param1:
            grad_param1 = bbmm_result.gradients[dim + 1]
            grad_noise = bbmm_result.gradients[dim + 2]
        else:
            grad_noise = bbmm_result.gradients[dim + 1]
        
        # Save NLL for final result and history
        last_nll = nll
        nll_history_ard_mf.append(nll)
        
        # NaN detection: if NLL is NaN, halve learning rate and skip this step
        if nll != nll:
            lr = lr * Float32(0.5)
            if verbose:
                print("  NaN NLL detected at step", step, "- halving LR to", lr)
            continue
        
        # Best-param tracking: snapshot params at best NLL
        if nll < best_nll_seen_mf:
            best_nll_seen_mf = nll
            for d in range(dim):
                best_raw_ls_mf[d] = raw_lengthscales[d]
            best_raw_noise_mf = raw_noise
            best_raw_os_mf = raw_outputscale
            best_raw_param1_mf = raw_param1
            best_raw_param2_mf = raw_param2
            best_raw_mean_mf = raw_mean
        
        # Cosine LR decay
        var effective_lr = lr
        if use_cosine_lr:
            effective_lr = compute_cosine_lr(learning_rate, step, num_iterations)
        
        # Adam update (ARD) - inline to avoid struct move issues
        var beta1 = Float32(0.9)
        var beta2 = Float32(0.999)
        var eps = Float32(1e-8)
        alias MAX_RAW = Float32(20.0)
        alias MIN_RAW = Float32(-20.0)
        adam_state.t += 1
        var t = Float32(adam_state.t)
        
        # Update each lengthscale dimension
        # Unified result: gradients[0..d-1] = per-dimension lengthscale gradients
        for d_idx in range(dim):
            var clipped_grad = clip_gradient(bbmm_result.gradients[d_idx])
            var grad_raw = clipped_grad * softplus_derivative(raw_lengthscales[d_idx])
            adam_state.m_ls[d_idx] = beta1 * adam_state.m_ls[d_idx] + (Float32(1.0) - beta1) * grad_raw
            adam_state.v_ls[d_idx] = beta2 * adam_state.v_ls[d_idx] + (Float32(1.0) - beta2) * grad_raw * grad_raw
            var m_hat = adam_state.m_ls[d_idx] / (Float32(1.0) - pow_float32(beta1, Int(t)))
            var v_hat = adam_state.v_ls[d_idx] / (Float32(1.0) - pow_float32(beta2, Int(t)))
            raw_lengthscales[d_idx] = raw_lengthscales[d_idx] - effective_lr * m_hat / (sqrt(v_hat) + eps)
            raw_lengthscales[d_idx] = max(MIN_RAW, min(MAX_RAW, raw_lengthscales[d_idx]))
        
        # Update noise
        var clipped_grad_noise = clip_gradient(grad_noise)
        var grad_raw_noise = clipped_grad_noise * softplus_derivative(raw_noise)
        adam_state.m_noise = beta1 * adam_state.m_noise + (Float32(1.0) - beta1) * grad_raw_noise
        adam_state.v_noise = beta2 * adam_state.v_noise + (Float32(1.0) - beta2) * grad_raw_noise * grad_raw_noise
        var m_hat_noise = adam_state.m_noise / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_noise = adam_state.v_noise / (Float32(1.0) - pow_float32(beta2, Int(t)))
        raw_noise = raw_noise - effective_lr * m_hat_noise / (sqrt(v_hat_noise) + eps)
        raw_noise = max(MIN_RAW, min(MAX_RAW, raw_noise))
        
        # Update output scale
        var clipped_grad_os = clip_gradient(grad_outputscale)
        var grad_raw_os = clipped_grad_os * softplus_derivative(raw_outputscale)
        adam_state.m_os = beta1 * adam_state.m_os + (Float32(1.0) - beta1) * grad_raw_os
        adam_state.v_os = beta2 * adam_state.v_os + (Float32(1.0) - beta2) * grad_raw_os * grad_raw_os
        var m_hat_os = adam_state.m_os / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_os = adam_state.v_os / (Float32(1.0) - pow_float32(beta2, Int(t)))
        raw_outputscale = raw_outputscale - effective_lr * m_hat_os / (sqrt(v_hat_os) + eps)
        raw_outputscale = max(MIN_RAW, min(MAX_RAW, raw_outputscale))
        
        # Update param1 (period/alpha) for Periodic/RQ kernels
        if has_param1:
            var clipped_grad_p1 = clip_gradient(grad_param1)
            var grad_raw_p1 = clipped_grad_p1 * softplus_derivative(raw_param1)
            adam_state.m_param1 = beta1 * adam_state.m_param1 + (Float32(1.0) - beta1) * grad_raw_p1
            adam_state.v_param1 = beta2 * adam_state.v_param1 + (Float32(1.0) - beta2) * grad_raw_p1 * grad_raw_p1
            var m_hat_p1 = adam_state.m_param1 / (Float32(1.0) - pow_float32(beta1, Int(t)))
            var v_hat_p1 = adam_state.v_param1 / (Float32(1.0) - pow_float32(beta2, Int(t)))
            raw_param1 = raw_param1 - effective_lr * m_hat_p1 / (sqrt(v_hat_p1) + eps)
            raw_param1 = max(MIN_RAW, min(MAX_RAW, raw_param1))
        
        # Update param2 (offset) for Polynomial kernel
        if has_param2:
            var clipped_grad_p2 = clip_gradient(grad_param2)
            var grad_raw_p2 = clipped_grad_p2 * softplus_derivative(raw_param2)
            adam_state.m_param2 = beta1 * adam_state.m_param2 + (Float32(1.0) - beta1) * grad_raw_p2
            adam_state.v_param2 = beta2 * adam_state.v_param2 + (Float32(1.0) - beta2) * grad_raw_p2 * grad_raw_p2
            var m_hat_p2 = adam_state.m_param2 / (Float32(1.0) - pow_float32(beta1, Int(t)))
            var v_hat_p2 = adam_state.v_param2 / (Float32(1.0) - pow_float32(beta2, Int(t)))
            raw_param2 = raw_param2 - effective_lr * m_hat_p2 / (sqrt(v_hat_p2) + eps)
            raw_param2 = max(MIN_RAW, min(MAX_RAW, raw_param2))
        
        # ConstantMean: Compute mean gradient and Adam update
        # d(NLL)/d(mean) = -sum(alpha) / n, where alpha = K^{-1}(y - mean)
        # alpha is bbmm_result.solution (CG solution for column 0 = y_centered)
        from .cg_solver import kernel_sum_reduce
        var mean_sum_dev = ctx.enqueue_create_buffer[DType.float32](1)
        var mean_sum_host = ctx.enqueue_create_host_buffer[DType.float32](1)
        ctx.enqueue_function[kernel_sum_reduce](
            mean_sum_dev.unsafe_ptr(), bbmm_result.solution.unsafe_ptr(), n,
            grid_dim=1, block_dim=256, shared_mem_bytes=8 * 4)
        ctx.enqueue_copy(dst_buf=mean_sum_host, src_buf=mean_sum_dev)
        ctx.synchronize()
        var mean_grad = -mean_sum_host.unsafe_ptr()[0] / Float32(n)
        mean_grad = clip_gradient(mean_grad)
        
        # Adam update for mean (unconstrained, no softplus chain rule)
        adam_state.m_mean = beta1 * adam_state.m_mean + (Float32(1.0) - beta1) * mean_grad
        adam_state.v_mean = beta2 * adam_state.v_mean + (Float32(1.0) - beta2) * mean_grad * mean_grad
        var m_hat_mean = adam_state.m_mean / (Float32(1.0) - pow_float32(beta1, Int(t)))
        var v_hat_mean = adam_state.v_mean / (Float32(1.0) - pow_float32(beta2, Int(t)))
        raw_mean -= effective_lr * m_hat_mean / (sqrt(v_hat_mean) + eps)
        
        # Safety check for mean (allow wider range than kernel params)
        if raw_mean != raw_mean:  # NaN check
            raw_mean = init_mean
        elif raw_mean > Float32(1000.0) or raw_mean < Float32(-1000.0):
            raw_mean = init_mean
        
        actual_iterations = step + 1
        
        # Print progress
        if verbose and (step % 10 == 0 or step == num_iterations - 1):
            print("Iter", step, "NLL:", nll, "σ²:", noise, "σ_f²:", outputscale, "mean:", raw_mean)
            if has_param1:
                print("  param1:", param1)
            if has_param2:
                var param2_val = softplus(raw_param2)
                print("  param2 (offset):", param2_val)
            # Print first few lengthscales
            if dim <= 5:
                for d in range(dim):
                    print("  ℓ[", d, "]:", lengthscales[d])
            else:
                for d in range(3):
                    print("  ℓ[", d, "]:", lengthscales[d])
                print("  ...")
        
        # Early stopping check
        if step >= min_iterations:
            if nll < best_nll - abs(best_nll) * early_stop_threshold:
                best_nll = nll
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    converged = True
                    if verbose:
                        print("Early stopping: converged after", actual_iterations, "iterations")
                    break
    
    # Restore best-seen params before computing final parameters
    for d in range(dim):
        raw_lengthscales[d] = best_raw_ls_mf[d]
    raw_noise = best_raw_noise_mf
    raw_outputscale = best_raw_os_mf
    raw_param1 = best_raw_param1_mf
    raw_param2 = best_raw_param2_mf
    raw_mean = best_raw_mean_mf
    
    # Final parameters
    var final_lengthscales = ctx.enqueue_create_host_buffer[float_dtype](dim)
    for d in range(dim):
        final_lengthscales[d] = softplus(raw_lengthscales[d])
    var final_noise = softplus(raw_noise)
    var final_outputscale = softplus(raw_outputscale)
    var final_param1 = softplus(raw_param1) if has_param1 else kernel_param1
    var final_param2 = softplus(raw_param2) if has_param2 else kernel_param2
    var final_mean = raw_mean
    var final_nll = best_nll_seen_mf
    
    # Lanczos root deferred to prediction time (predict_with_method recomputes it).
    var lanczos_dummy_mf = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    # Alpha = K^{-1} @ (y - mean) is deferred to prediction time.
    var alpha_dummy_mf = ctx.enqueue_create_host_buffer[float_dtype](1)
    
    # Keep buffers alive until all pointer users are done
    _ = precond_ard_mf
    _ = y_device
    _ = y_centered_device
    
    return TrainingResultARD(
        final_lengthscales, final_noise, final_outputscale,
        final_param1, final_param2, final_mean,
        final_nll, nll_history_ard_mf^, actual_iterations, converged,
        lanczos_dummy_mf, 0, n, dim, alpha_dummy_mf, False
    )


# =============================================================================
# Generic Provider-Based Training Loop
# =============================================================================

fn train_with_provider[P: GradientProvider](
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
    early_stop_patience: Int = 10,
    early_stop_tol: Float32 = 1e-4,
    verbose: Bool = False,
    init_mean: Float32 = 0.0,
    max_tridiag_iter: Int = 30,
    precond_rebuild_threshold: Float32 = 0.5,
    use_cosine_lr: Bool = True,
    use_preconditioner: Bool = True,
) raises -> TrainingResultGeneric:
    """Train a GP with any GradientProvider using BBMM.
    
    Generic training loop that accepts any provider implementing the
    GradientProvider trait. This enables plugging in different provider types
    (matrix-free, materialized, fused codegen, etc.) without duplicating
    the training loop.
    
    Args:
        provider: A GradientProvider (e.g., CompositeGradientAdapter, FusedCompositeGradientAdapter)
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
    # Copy y to device
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    for i in range(n):
        y_host[i] = y_host_ptr[i]
    var y_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.synchronize()
    
    # ConstantMean: Initialize mean parameter (unconstrained, no softplus)
    var raw_mean = init_mean
    
    # ConstantMean: Create y_centered buffers for centering y by current mean
    var y_host_centered = ctx.enqueue_create_host_buffer[float_dtype](n)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # Initialize raw parameters (unconstrained space via inv_softplus)
    var raw_params = List[Float32]()
    for p in range(num_kernel_params):
        raw_params.append(inv_softplus(initial_params_ptr[p]))
    raw_params.append(inv_softplus(initial_noise))  # Noise is last
    
    # Initialize Adam state
    var adam_state = AdamStateGeneric(num_kernel_params)
    
    # Create buffer pool for BBMM
    var pool = CGBufferPool(ctx, n, 1 + num_probes, num_probes, max_tridiag_iter)
    
    # Create params host buffer (reused each iteration)
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_kernel_params)
    for p in range(num_kernel_params):
        params_host[p] = initial_params_ptr[p]
    
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
    for p in range(num_kernel_params + 1):  # +1 for noise
        best_raw_params.append(raw_params[p])
    var best_raw_mean = raw_mean
    var best_nll_seen = Float32(1e30)
    
    # Build initial preconditioner
    var num_cols = 1 + num_probes
    var num_kparams = provider.num_gradient_params()
    pool.ensure_capacity(ctx, n, num_cols, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kparams)
    var precond = build_pivoted_cholesky_precond_unified(provider, rank=precond_rank, max_num_cols=num_cols)
    
    for iteration in range(max_iterations):
        actual_iterations = iteration + 1
        
        # Convert raw params to constrained space
        var constrained_params = List[Float32]()
        for p in range(num_kernel_params):
            constrained_params.append(softplus(raw_params[p]))
        var noise = softplus(raw_params[num_kernel_params])
        
        # Update params host buffer
        for p in range(num_kernel_params):
            params_host[p] = constrained_params[p]
        
        # Update provider with new parameters
        provider.update_params(params_host.unsafe_ptr())
        provider.update_noise(noise)
        
        # Adaptive preconditioner rebuild
        if iteration > 0:
            var max_rel = Float32(0.0)
            for p in range(num_kernel_params):
                var rel_p = abs(constrained_params[p] - last_rebuild_params[p]) / max(abs(last_rebuild_params[p]), Float32(1e-8))
                max_rel = max(max_rel, rel_p)
            var rel_noise = abs(noise - last_rebuild_noise) / max(abs(last_rebuild_noise), Float32(1e-8))
            max_rel = max(max_rel, rel_noise)
            if max_rel > precond_rebuild_threshold:
                precond = build_pivoted_cholesky_precond_unified(provider, rank=precond_rank, max_num_cols=num_cols)
                for p in range(num_kernel_params):
                    last_rebuild_params[p] = constrained_params[p]
                last_rebuild_noise = noise
        
        # ConstantMean: Center y by current mean on GPU (no CPU loop or H2D copy)
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
        
        # NaN detection: if NLL is NaN, halve learning rate and skip this step
        if last_nll != last_nll:
            lr = lr * Float32(0.5)
            if verbose:
                print("  NaN NLL detected at iteration", iteration, "- halving LR to", lr)
            continue
        
        # Best-param tracking: snapshot params at best NLL
        if last_nll < best_nll_seen:
            best_nll_seen = last_nll
            for p in range(num_kernel_params + 1):
                best_raw_params[p] = raw_params[p]
            best_raw_mean = raw_mean
        
        # Cosine LR decay
        var effective_lr = lr
        if use_cosine_lr:
            effective_lr = compute_cosine_lr(learning_rate, iteration, max_iterations)
        
        # Collect gradients: [kernel_gradients..., grad_noise]
        # Unified result: gradients[0..N-1] = kernel params, gradients[N] = noise
        var gradients = List[Float32]()
        for p in range(num_kernel_params):
            gradients.append(result.gradients[p])
        gradients.append(result.gradients[num_kernel_params])  # Noise is last
        
        # Adam update for kernel params + noise
        adam_state = adam_update_state_inplace(
            adam_state^, gradients, raw_params, effective_lr
        )
        raw_params = adam_update_params(
            adam_state, gradients, raw_params, effective_lr
        )
        
        # ConstantMean: Compute mean gradient via GPU reduction
        from .cg_solver import kernel_sum_reduce
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
        
        # Early stopping check
        if last_nll < best_nll - early_stop_tol:
            best_nll = last_nll
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                converged = True
                if verbose:
                    print("Early stopping: converged after", actual_iterations, "iterations")
                break
    
    # Keepalive for cached preconditioner
    _ = precond
    
    # Restore best-seen params before computing final parameters
    for p in range(num_kernel_params + 1):
        raw_params[p] = best_raw_params[p]
    raw_mean = best_raw_mean
    
    # Extract final parameters
    var final_params = List[Float32]()
    for p in range(num_kernel_params):
        final_params.append(softplus(raw_params[p]))
    var final_noise = softplus(raw_params[num_kernel_params])
    var final_mean = raw_mean
    
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
    
    # Lanczos root deferred to prediction time (predict recomputes it).
    var lanczos_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
    var alpha_dummy = ctx.enqueue_create_host_buffer[float_dtype](n)
    
    # Keep buffers alive until all pointer users are done
    _ = y_device
    _ = y_centered_device
    
    return TrainingResultGeneric(
        final_params^, final_noise, final_mean, best_nll_seen,
        actual_iterations, converged,
        lanczos_dummy^, 0, n, num_kernel_params,
        alpha_dummy^, False
    )


# =============================================================================
# Composite Kernel Training (Generic N parameters)
# =============================================================================

fn train_gp_composite[DIM: Int, K: ComposableKernel](
    ctx: DeviceContext,
    x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    initial_params_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [N] initial kernel params
    initial_noise: Float32,
    max_iterations: Int = 100,
    learning_rate: Float32 = 0.01,
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    early_stop_patience: Int = 10,
    early_stop_tol: Float32 = 1e-4,
    verbose: Bool = False,
    method: Int = 0,  # 0=matrix_free, 1=materialized, 3=auto
    init_mean: Float32 = 0.0,
    max_tridiag_iter: Int = 30,
    precond_rebuild_threshold: Float32 = 0.5,
    # Learning rate schedule: True=cosine decay, False=constant
    use_cosine_lr: Bool = True,
) raises -> TrainingResultGeneric:
    """Train a GP with a composite kernel using BBMM.
    
    Delegates to train_with_provider after creating the appropriate provider
    for the selected method (matrix-free or materialized).
    
    Args:
        ctx: GPU device context
        x_host_ptr: Training data [n, DIM] on host (row-major)
        y_host_ptr: Training targets [n] on host
        n: Number of training points
        initial_params_ptr: Initial kernel parameters [N] on host
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
        method: Training method:
            0 = matrix_free (default): O(n) memory, recomputes K each matvec
            1 = materialized: O(n²) memory, materializes K for GEMM-based matvecs (~1.8x faster)
            3 = auto: Choose based on n (materialized for n≤15000, else matrix_free)
        
    Returns:
        TrainingResultGeneric with optimized parameters
    """
    var num_kernel_params = K.num_params()
    
    # Determine effective method based on auto selection
    var effective_method = method
    if method == 3:  # auto
        if n <= 15000:
            effective_method = 1  # materialized
        else:
            effective_method = 0  # matrix_free
    elif method == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    if effective_method != 0 and effective_method != 1:
        raise Error("unsupported training method")
    
    if verbose:
        var method_name: String
        if effective_method == 0:
            method_name = "matrix_free"
        else:
            method_name = "materialized"
        print("Training GP with composite kernel")
        print("  n =", n, ", DIM =", DIM, ", num_kernel_params =", num_kernel_params)
        print("  method =", method_name)
    
    if effective_method == 0:
        # Matrix-free path
        var provider = CompositeProvider[DIM, K](
            ctx, x_host_ptr, initial_params_ptr, n, initial_noise
        )
        var adapter = CompositeGradientAdapter[DIM, K](provider^)
        return train_with_provider(
            adapter, ctx, y_host_ptr, n, num_kernel_params, initial_params_ptr,
            initial_noise, max_iterations, learning_rate, num_probes, max_cg_iter,
            cg_tol, precond_rank, early_stop_patience, early_stop_tol, verbose,
            init_mean, max_tridiag_iter, precond_rebuild_threshold, use_cosine_lr,
        )
    else:
        # Materialized path
        var mat_provider = MaterializedCompositeProvider[DIM, K](
            ctx, x_host_ptr, initial_params_ptr, n, initial_noise, False
        )
        var mat_adapter = MaterializedCompositeGradientAdapter[DIM, K](mat_provider^)
        return train_with_provider(
            mat_adapter, ctx, y_host_ptr, n, num_kernel_params, initial_params_ptr,
            initial_noise, max_iterations, learning_rate, num_probes, max_cg_iter,
            cg_tol, precond_rank, early_stop_patience, early_stop_tol, verbose,
            init_mean, max_tridiag_iter, precond_rebuild_threshold, use_cosine_lr,
        )
