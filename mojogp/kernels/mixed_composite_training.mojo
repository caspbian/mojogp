"""Training loop for mixed composite + categorical GP models.

Implements train_gp_mixed_composite[DIM, K: ComposableKernel]() which combines:
- Compile-time composite kernel parameterization (arbitrary kernel compositions)
- Runtime categorical correlation optimization

The parameter vector layout:
  [0 .. K.num_params()-1]: raw composite kernel params (softplus)
  [K.num_params() .. K.num_params()+total_cat_params-1]: raw categorical params (softplus)
  [last]: raw noise (softplus)

The gradient ordering from bbmm_unified matches this because
MixedCompositeProvider.num_gradient_params() = K.num_params() + total_cat_params,
and noise is appended by bbmm_unified as the last element.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt, log

from .constants import (
    float_dtype,
    PI,
    CAT_KERNEL_GD,
    CAT_KERNEL_CR,
    CAT_KERNEL_EHH,
    CAT_KERNEL_HH,
    CAT_KERNEL_FE,
)
from .composable_kernel import ComposableKernel
from .categorical_state import CategoricalCorrelationState
from .mixed_composite_provider import (
    MixedCompositeProvider,
    MixedMaterializedCompositeProvider,
)
from .utils import softplus, inv_softplus, softplus_derivative, sigmoid, inv_sigmoid, sigmoid_derivative
from .training import (
    clip_gradient,
    AdamStateGeneric,
    adam_update_state_inplace,
    adam_update_state_inplace_custom,
    adam_update_params,
    TrainingResultGeneric,
    _warmup_gpu_kernels,
)
from .combined_inv_quad_logdet import (
    bbmm_unified,
    bbmm_with_precond,
    CGBufferPool,
    batched_cg_unified,
    build_pivoted_cholesky_precond_unified,
)
from .pivoted_cholesky import PivotedCholeskyPrecond


# =============================================================================
# Mixed Composite Training Result
# =============================================================================

struct MixedCompositeTrainingResult(Movable):
    """Result from mixed composite + categorical GP training.
    
    Stores both the composite kernel params and categorical params separately.
    """
    var final_params: List[Float32]       # [K.num_params()] optimized composite params
    var cat_params: HostBuffer[float_dtype]  # [total_cat_params] optimized categorical params
    var noise: Float32
    var mean: Float32                     # Learned constant mean function value
    var final_nll: Float32
    var iterations: Int
    var converged: Bool
    var n: Int
    var num_kernel_params: Int            # K.num_params()
    var num_cat_params: Int
    var has_alpha: Bool                   # Whether alpha contains valid data
    var alpha: HostBuffer[float_dtype]    # [n] K^{-1} @ (y - mean) for prediction (valid when has_alpha=True)
    
    fn __init__(
        out self,
        var final_params: List[Float32],
        var cat_params: HostBuffer[float_dtype],
        noise: Float32,
        mean: Float32,
        final_nll: Float32,
        iterations: Int,
        converged: Bool,
        n: Int,
        num_kernel_params: Int,
        num_cat_params: Int,
        var alpha: HostBuffer[float_dtype],
        has_alpha: Bool = True,
    ):
        self.final_params = final_params^
        self.cat_params = cat_params^
        self.noise = noise
        self.mean = mean
        self.final_nll = final_nll
        self.iterations = iterations
        self.converged = converged
        self.n = n
        self.num_kernel_params = num_kernel_params
        self.num_cat_params = num_cat_params
        self.has_alpha = has_alpha
        self.alpha = alpha^
    
    fn __moveinit__(out self, deinit other: Self):
        self.final_params = other.final_params^
        self.cat_params = other.cat_params^
        self.noise = other.noise
        self.mean = other.mean
        self.final_nll = other.final_nll
        self.iterations = other.iterations
        self.converged = other.converged
        self.n = other.n
        self.num_kernel_params = other.num_kernel_params
        self.num_cat_params = other.num_cat_params
        self.has_alpha = other.has_alpha
        self.alpha = other.alpha^


# =============================================================================
# Main Training Function
# =============================================================================

fn train_gp_mixed_composite[DIM: Int, K: ComposableKernel](
    ctx: DeviceContext,
    x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],     # [n, DIM] continuous data on host
    c_host: HostBuffer[DType.int32],                       # [n * num_cat_vars] categorical indices
    y_host_ptr: UnsafePointer[Float32, MutAnyOrigin],      # [n] targets on host
    n: Int,
    initial_params_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [K.num_params()] initial composite params
    initial_noise: Float32,
    cat_levels: List[Int],
    cat_kernel_types: List[Int],
    max_iterations: Int = 100,
    learning_rate: Float32 = 0.01,
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    max_tridiag_iter: Int = 30,
    precond_rebuild_threshold: Float32 = 0.5,
    early_stop_patience: Int = 10,
    early_stop_tol: Float32 = 1e-4,
    verbose: Bool = False,
    method: Int = 0,  # 0=matrix_free, 1=materialized, 3=auto
    init_mean: Float32 = 0.0,
) raises -> MixedCompositeTrainingResult:
    """Train a GP with composite kernel + categorical features using BBMM.
    
    Args:
        ctx: GPU device context
        x_host_ptr: Continuous training data [n, DIM] on host (row-major)
        c_host: Categorical indices [n * num_cat_vars] on host
        y_host_ptr: Training targets [n] on host
        n: Number of training points
        initial_params_ptr: Initial composite kernel parameters [K.num_params()]
        initial_noise: Initial noise variance
        cat_levels: Number of levels per categorical variable
        cat_kernel_types: Kernel type (CAT_KERNEL_*) per categorical variable
        max_iterations: Maximum training iterations
        learning_rate: Adam learning rate
        num_probes: Number of probes for log-det estimation
        max_cg_iter: Maximum CG iterations
        cg_tol: CG convergence tolerance
        precond_rank: Pivoted Cholesky preconditioner rank
        early_stop_patience: Iterations without improvement before stopping
        early_stop_tol: Minimum improvement to reset patience
        verbose: Print progress
        method: 0=matrix_free, 1=materialized, 3=auto
        init_mean: Initial constant mean value (default 0.0, auto-detected from data)
        
    Returns:
        MixedCompositeTrainingResult with optimized parameters
    """
    var num_comp_params = K.num_params()
    
    # Auto-select method
    var effective_method = method
    if method == 3:
        if n <= 10000:
            effective_method = 1  # materialized
        else:
            effective_method = 0  # matrix_free
    
    if verbose:
        print("Training mixed composite + categorical GP")
        print("  n =", n, ", DIM =", DIM, ", composite params =", num_comp_params)
        print("  num_cat_vars =", len(cat_levels))
        for v in range(len(cat_levels)):
            print("  cat var", v, ": levels =", cat_levels[v], "kernel =", cat_kernel_types[v])
        if effective_method == 0:
            print("  method = matrix_free")
        else:
            print("  method = materialized")
    
    # =========================================================================
    # Step 1: Upload data to GPU
    # =========================================================================
    
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    for i in range(n):
        y_host[i] = y_host_ptr[i]
    var y_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.synchronize()
    
    # ConstantMean: centered y buffers
    var raw_mean = init_mean
    var m_mean = Float32(0.0)  # Adam first moment for mean
    var v_mean = Float32(0.0)  # Adam second moment for mean
    var y_host_centered = ctx.enqueue_create_host_buffer[float_dtype](n)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](n)
    
    # =========================================================================
    # Step 2: Initialize categorical state
    # =========================================================================
    
    var cat_state = CategoricalCorrelationState(ctx, cat_levels.copy(), cat_kernel_types.copy(), n)
    cat_state.upload_categorical_data(c_host)
    
    var total_cat_params = cat_state.get_total_cat_params()
    var num_kernel_params = num_comp_params + total_cat_params  # excludes noise
    
    if verbose:
        print("  total categorical params:", total_cat_params)
        print("  total gradient params:", num_kernel_params)
    
    # =========================================================================
    # Step 3: Initialize parameters
    # =========================================================================
    
    # Build raw parameter vector: [comp_params..., cat_params..., noise]
    var raw_params = List[Float32]()
    
    # Composite kernel params
    for p in range(num_comp_params):
        raw_params.append(inv_softplus(initial_params_ptr[p]))
    
    # Categorical params — same initialization as train_gp_mixed
    for v in range(len(cat_levels)):
        var L = cat_levels[v]
        var kt = cat_kernel_types[v]
        var np = _num_params_for_variant(L, kt)
        for p in range(np):
            if kt == CAT_KERNEL_GD:
                raw_params.append(inv_softplus(Float32(0.5)))
            elif kt == CAT_KERNEL_CR:
                raw_params.append(inv_softplus(Float32(0.3)))
            elif kt == CAT_KERNEL_EHH or kt == CAT_KERNEL_HH:
                # Angles: sigmoid*pi, init to pi/4
                raw_params.append(inv_sigmoid(Float32(0.25)))
            elif kt == CAT_KERNEL_FE:
                var num_angles = L * (L - 1) // 2
                if p < num_angles:
                    raw_params.append(inv_sigmoid(Float32(0.25)))
                else:
                    raw_params.append(inv_softplus(Float32(0.3)))
    
    # Noise (last element)
    raw_params.append(inv_softplus(initial_noise))
    
    var total_params = num_kernel_params + 1  # +1 for noise
    
    # =========================================================================
    # Step 4: Initialize optimizer and buffers
    # =========================================================================
    
    var adam_state = AdamStateGeneric(num_kernel_params)
    var pool = CGBufferPool(ctx, n, 1 + num_probes, num_probes, max_tridiag_iter)
    
    # Composite params host buffer (reused each iteration)
    var comp_params_host = ctx.enqueue_create_host_buffer[float_dtype](num_comp_params)
    for p in range(num_comp_params):
        comp_params_host[p] = initial_params_ptr[p]
    
    # Categorical params host buffer
    var cat_params_host = HostBuffer[float_dtype](ctx, total_cat_params)
    _extract_cat_params(raw_params, num_comp_params, total_cat_params, cat_params_host, cat_levels, cat_kernel_types)
    cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())
    
    # =========================================================================
    # Step 5: Training loop
    # =========================================================================
    
    _warmup_gpu_kernels(ctx)
    
    var best_nll = Float32(1e30)
    var no_improve_count = 0
    var actual_iterations = 0
    var converged = False
    var last_nll = Float32(0.0)
    
    if effective_method == 0:
        # =================================================================
        # MATRIX-FREE PATH
        # =================================================================
        var provider = MixedCompositeProvider[DIM, K](
            ctx, x_host_ptr, comp_params_host.unsafe_ptr(), n, initial_noise, cat_state^,
        )
        provider.set_cat_params_ptr(cat_params_host.unsafe_ptr())
        
        # Build initial preconditioner for matrix-free path
        var num_cols_mc_mf = 1 + num_probes
        var num_kparams_mc_mf = provider.num_gradient_params()
        pool.ensure_capacity(ctx, n, num_cols_mc_mf, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kparams_mc_mf)
        var precond_mc_mf = build_pivoted_cholesky_precond_unified(provider, rank=precond_rank, max_num_cols=num_cols_mc_mf)
        
        # Track params at last rebuild for adaptive rebuild
        var last_rebuild_params_mc_mf = List[Float32]()
        for p in range(num_kernel_params):
            last_rebuild_params_mc_mf.append(softplus(raw_params[p]))
        var last_rebuild_noise_mc_mf = initial_noise
        
        for iteration in range(max_iterations):
            actual_iterations = iteration + 1
            
            # Convert raw params to constrained space
            for p in range(num_comp_params):
                comp_params_host[p] = softplus(raw_params[p])
            var noise = softplus(raw_params[total_params - 1])
            _extract_cat_params(raw_params, num_comp_params, total_cat_params, cat_params_host, cat_levels, cat_kernel_types)
            
            # Update provider
            provider.update_params(comp_params_host.unsafe_ptr())
            provider.update_noise(noise)
            provider.update_categorical_params(cat_params_host.unsafe_ptr())
            
            # Adaptive preconditioner rebuild
            if iteration > 0:
                var max_rel = Float32(0.0)
                for p in range(num_kernel_params):
                    var cur_val = softplus(raw_params[p])
                    var rel_p = abs(cur_val - last_rebuild_params_mc_mf[p]) / max(abs(last_rebuild_params_mc_mf[p]), Float32(1e-8))
                    max_rel = max(max_rel, rel_p)
                var rel_noise = abs(noise - last_rebuild_noise_mc_mf) / max(abs(last_rebuild_noise_mc_mf), Float32(1e-8))
                max_rel = max(max_rel, rel_noise)
                if max_rel > precond_rebuild_threshold:
                    precond_mc_mf = build_pivoted_cholesky_precond_unified(provider, rank=precond_rank, max_num_cols=num_cols_mc_mf)
                    for p in range(num_kernel_params):
                        last_rebuild_params_mc_mf[p] = softplus(raw_params[p])
                    last_rebuild_noise_mc_mf = noise
            
            # Center y by subtracting current mean
            for i in range(n):
                y_host_centered[i] = y_host[i] - raw_mean
            ctx.enqueue_copy(dst_buf=y_centered_device, src_buf=y_host_centered)
            ctx.synchronize()
            
            # Compute NLL and gradients with cached preconditioner
            var result = bbmm_with_precond(
                provider, precond_mc_mf, y_centered_device.unsafe_ptr(), n, pool,
                num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
                iteration=iteration,
                recycle_alpha=iteration > 0,
            )
            
            last_nll = result.nll
            
            # Compute mean gradient: d(NLL)/d(mean) = -sum(alpha) / n
            # where alpha = K^{-1}(y - mean) is in result.solution
            var alpha_host_tmp = ctx.enqueue_create_host_buffer[float_dtype](n)
            ctx.enqueue_copy(dst_buf=alpha_host_tmp, src_buf=result.solution)
            ctx.synchronize()
            var sum_alpha = Float32(0.0)
            for i in range(n):
                sum_alpha += alpha_host_tmp[i]
            var mean_grad = -sum_alpha / Float32(n)
            mean_grad = clip_gradient(mean_grad)
            
            # Adam update for mean (unconstrained, no softplus)
            var t_f = Float32(iteration + 1)
            m_mean = Float32(0.9) * m_mean + Float32(0.1) * mean_grad
            v_mean = Float32(0.999) * v_mean + Float32(0.001) * mean_grad * mean_grad
            var m_hat = m_mean / (Float32(1.0) - Float32(0.9) ** t_f)
            var v_hat = v_mean / (Float32(1.0) - Float32(0.999) ** t_f)
            raw_mean -= learning_rate * m_hat / (sqrt(v_hat) + Float32(1e-8))
            
            # Safety: NaN check and clamp
            if raw_mean != raw_mean:
                raw_mean = init_mean
            if raw_mean > Float32(1000.0):
                raw_mean = Float32(1000.0)
            elif raw_mean < Float32(-1000.0):
                raw_mean = Float32(-1000.0)
            
            # Collect gradients: [comp_grads..., cat_grads..., noise_grad]
            var gradients = List[Float32]()
            for p in range(num_kernel_params):
                gradients.append(clip_gradient(result.gradients[p]))
            gradients.append(clip_gradient(result.gradients[num_kernel_params]))
            
            # Adam update
            var chain_rule = _build_chain_rule_derivatives_composite(raw_params, num_comp_params, total_params, cat_levels, cat_kernel_types)
            adam_state = adam_update_state_inplace_custom(
                adam_state^, gradients, raw_params, chain_rule, learning_rate
            )
            raw_params = adam_update_params(
                adam_state, gradients, raw_params, learning_rate
            )
            
            # Clamp NaN/extreme raw params
            for p in range(total_params):
                var rp = raw_params[p]
                if rp != rp or rp > Float32(20.0) or rp < Float32(-20.0):
                    raw_params[p] = Float32(-2.0) if p == total_params - 1 else Float32(0.0)
            
            # Print progress
            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                print("Iter", iteration, ": NLL =", last_nll, ", noise =", noise, ", mean =", raw_mean)
            
            # Early stopping
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
        _ = precond_mc_mf
        
        # Finalize: compute alpha = K^{-1} @ (y - mean)
        # Update provider one last time with final params
        for p in range(num_comp_params):
            comp_params_host[p] = softplus(raw_params[p])
        var final_noise = softplus(raw_params[total_params - 1])
        _extract_cat_params(raw_params, num_comp_params, total_cat_params, cat_params_host, cat_levels, cat_kernel_types)
        
        provider.update_params(comp_params_host.unsafe_ptr())
        provider.update_noise(final_noise)
        provider.update_categorical_params(cat_params_host.unsafe_ptr())
        
        var final_params = List[Float32]()
        for p in range(num_comp_params):
            final_params.append(softplus(raw_params[p]))
        
        var final_cat_params = HostBuffer[float_dtype](ctx, total_cat_params)
        for p in range(total_cat_params):
            final_cat_params[p] = cat_params_host[p]
        
        # Alpha = K^{-1} @ (y - mean) is deferred to prediction time.
        var alpha_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        if verbose:
            print("Training complete: NLL =", last_nll, ", noise =", final_noise, ", mean =", raw_mean)
        
        _ = y_device
        _ = y_centered_device
        _ = cat_params_host
        _ = comp_params_host
        
        return MixedCompositeTrainingResult(
            final_params^, final_cat_params^, final_noise, raw_mean, last_nll,
            actual_iterations, converged, n, num_comp_params, total_cat_params,
            alpha_dummy^, False,
        )
    else:
        # =================================================================
        # MATERIALIZED PATH
        # =================================================================
        var mat_provider = MixedMaterializedCompositeProvider[DIM, K](
            ctx, x_host_ptr, comp_params_host.unsafe_ptr(), n, initial_noise, cat_state^,
        )
        mat_provider.set_cat_params_ptr(cat_params_host.unsafe_ptr())
        
        # Build initial preconditioner for materialized path
        var num_cols_mc_mat = 1 + num_probes
        var num_kparams_mc_mat = mat_provider.num_gradient_params()
        pool.ensure_capacity(ctx, n, num_cols_mc_mat, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kparams_mc_mat)
        var precond_mc_mat = build_pivoted_cholesky_precond_unified(mat_provider, rank=precond_rank, max_num_cols=num_cols_mc_mat)
        
        # Track params at last rebuild for adaptive rebuild
        var last_rebuild_params_mc_mat = List[Float32]()
        for p in range(num_kernel_params):
            last_rebuild_params_mc_mat.append(softplus(raw_params[p]))
        var last_rebuild_noise_mc_mat = initial_noise
        
        for iteration in range(max_iterations):
            actual_iterations = iteration + 1
            
            for p in range(num_comp_params):
                comp_params_host[p] = softplus(raw_params[p])
            var noise = softplus(raw_params[total_params - 1])
            _extract_cat_params(raw_params, num_comp_params, total_cat_params, cat_params_host, cat_levels, cat_kernel_types)
            
            mat_provider.update_params(comp_params_host.unsafe_ptr())
            mat_provider.update_noise(noise)
            mat_provider.update_categorical_params(cat_params_host.unsafe_ptr())
            
            # Adaptive preconditioner rebuild
            if iteration > 0:
                var max_rel = Float32(0.0)
                for p in range(num_kernel_params):
                    var cur_val = softplus(raw_params[p])
                    var rel_p = abs(cur_val - last_rebuild_params_mc_mat[p]) / max(abs(last_rebuild_params_mc_mat[p]), Float32(1e-8))
                    max_rel = max(max_rel, rel_p)
                var rel_noise = abs(noise - last_rebuild_noise_mc_mat) / max(abs(last_rebuild_noise_mc_mat), Float32(1e-8))
                max_rel = max(max_rel, rel_noise)
                if max_rel > precond_rebuild_threshold:
                    precond_mc_mat = build_pivoted_cholesky_precond_unified(mat_provider, rank=precond_rank, max_num_cols=num_cols_mc_mat)
                    for p in range(num_kernel_params):
                        last_rebuild_params_mc_mat[p] = softplus(raw_params[p])
                    last_rebuild_noise_mc_mat = noise
            
            # Center y by subtracting current mean
            for i in range(n):
                y_host_centered[i] = y_host[i] - raw_mean
            ctx.enqueue_copy(dst_buf=y_centered_device, src_buf=y_host_centered)
            ctx.synchronize()
            
            var result = bbmm_with_precond(
                mat_provider, precond_mc_mat, y_centered_device.unsafe_ptr(), n, pool,
                num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
                iteration=iteration,
                recycle_alpha=iteration > 0,
            )
            
            last_nll = result.nll
            
            # Compute mean gradient: d(NLL)/d(mean) = -sum(alpha) / n
            var alpha_host_tmp = ctx.enqueue_create_host_buffer[float_dtype](n)
            ctx.enqueue_copy(dst_buf=alpha_host_tmp, src_buf=result.solution)
            ctx.synchronize()
            var sum_alpha = Float32(0.0)
            for i in range(n):
                sum_alpha += alpha_host_tmp[i]
            var mean_grad = -sum_alpha / Float32(n)
            mean_grad = clip_gradient(mean_grad)
            
            # Adam update for mean (unconstrained, no softplus)
            var t_f = Float32(iteration + 1)
            m_mean = Float32(0.9) * m_mean + Float32(0.1) * mean_grad
            v_mean = Float32(0.999) * v_mean + Float32(0.001) * mean_grad * mean_grad
            var m_hat = m_mean / (Float32(1.0) - Float32(0.9) ** t_f)
            var v_hat = v_mean / (Float32(1.0) - Float32(0.999) ** t_f)
            raw_mean -= learning_rate * m_hat / (sqrt(v_hat) + Float32(1e-8))
            
            # Safety: NaN check and clamp
            if raw_mean != raw_mean:
                raw_mean = init_mean
            if raw_mean > Float32(1000.0):
                raw_mean = Float32(1000.0)
            elif raw_mean < Float32(-1000.0):
                raw_mean = Float32(-1000.0)
            
            var gradients = List[Float32]()
            for p in range(num_kernel_params):
                gradients.append(clip_gradient(result.gradients[p]))
            gradients.append(clip_gradient(result.gradients[num_kernel_params]))
            
            var chain_rule = _build_chain_rule_derivatives_composite(raw_params, num_comp_params, total_params, cat_levels, cat_kernel_types)
            adam_state = adam_update_state_inplace_custom(
                adam_state^, gradients, raw_params, chain_rule, learning_rate
            )
            raw_params = adam_update_params(
                adam_state, gradients, raw_params, learning_rate
            )
            
            for p in range(total_params):
                var rp = raw_params[p]
                if rp != rp or rp > Float32(20.0) or rp < Float32(-20.0):
                    raw_params[p] = Float32(-2.0) if p == total_params - 1 else Float32(0.0)
            
            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                print("Iter", iteration, ": NLL =", last_nll, ", noise =", noise, ", mean =", raw_mean)
            
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
        _ = precond_mc_mat
        
        # Finalize: compute alpha = K^{-1} @ (y - mean)
        for p in range(num_comp_params):
            comp_params_host[p] = softplus(raw_params[p])
        var final_noise = softplus(raw_params[total_params - 1])
        _extract_cat_params(raw_params, num_comp_params, total_cat_params, cat_params_host, cat_levels, cat_kernel_types)
        
        mat_provider.update_params(comp_params_host.unsafe_ptr())
        mat_provider.update_noise(final_noise)
        mat_provider.update_categorical_params(cat_params_host.unsafe_ptr())
        
        var final_params = List[Float32]()
        for p in range(num_comp_params):
            final_params.append(softplus(raw_params[p]))
        
        var final_cat_params = HostBuffer[float_dtype](ctx, total_cat_params)
        for p in range(total_cat_params):
            final_cat_params[p] = cat_params_host[p]
        
        # Alpha = K^{-1} @ (y - mean) is deferred to prediction time.
        var alpha_dummy = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        if verbose:
            print("Training complete: NLL =", last_nll, ", noise =", final_noise, ", mean =", raw_mean)
        
        _ = y_device
        _ = y_centered_device
        _ = cat_params_host
        _ = comp_params_host
        
        return MixedCompositeTrainingResult(
            final_params^, final_cat_params^, final_noise, raw_mean, last_nll,
            actual_iterations, converged, n, num_comp_params, total_cat_params,
            alpha_dummy^, False,
        )


# =============================================================================
# Helper Functions
# =============================================================================

fn _build_chain_rule_derivatives_composite(
    read raw_params: List[Float32],
    num_comp_params: Int,
    total_params: Int,
    read cat_levels: List[Int],
    read cat_kernel_types: List[Int],
) -> List[Float32]:
    """Build per-parameter chain rule derivatives for composite + categorical.

    Same logic as _build_chain_rule_derivatives in mixed_training.mojo but
    for the composite parameter layout.
    """
    var derivs = List[Float32]()

    # Composite params: softplus
    for p in range(num_comp_params):
        derivs.append(softplus_derivative(raw_params[p]))

    # Categorical params
    var cat_offset = num_comp_params
    for v in range(len(cat_levels)):
        var L = cat_levels[v]
        var kt = cat_kernel_types[v]
        var np = _num_params_for_variant(L, kt)
        for p in range(np):
            if kt == CAT_KERNEL_EHH or kt == CAT_KERNEL_HH:
                derivs.append(sigmoid_derivative(raw_params[cat_offset]) * Float32(PI))
            elif kt == CAT_KERNEL_FE:
                var num_angles = L * (L - 1) // 2
                if p < num_angles:
                    derivs.append(sigmoid_derivative(raw_params[cat_offset]) * Float32(PI))
                else:
                    derivs.append(softplus_derivative(raw_params[cat_offset]))
            else:
                derivs.append(softplus_derivative(raw_params[cat_offset]))
            cat_offset += 1

    # Noise: softplus
    derivs.append(softplus_derivative(raw_params[total_params - 1]))

    return derivs^


fn _extract_cat_params(
    read raw_params: List[Float32],
    num_comp_params: Int,
    total_cat_params: Int,
    cat_params_host: HostBuffer[float_dtype],
    read cat_levels: List[Int],
    read cat_kernel_types: List[Int],
) -> None:
    """Extract constrained categorical parameters from raw parameter vector.
    
    Applies the correct transformation per parameter type:
    - GD/CR: softplus (non-negative distance parameters)
    - EHH/HH: sigmoid*pi (angle parameters in [0, pi])
    - FE: sigmoid*pi for angle params, softplus for diagonal params
    """
    var offset = 0
    for v in range(len(cat_levels)):
        var L = cat_levels[v]
        var kt = cat_kernel_types[v]
        var np = _num_params_for_variant(L, kt)
        for p in range(np):
            var raw_idx = num_comp_params + offset
            if kt == CAT_KERNEL_EHH or kt == CAT_KERNEL_HH:
                cat_params_host[offset] = sigmoid(raw_params[raw_idx]) * Float32(PI)
            elif kt == CAT_KERNEL_FE:
                var num_angles = L * (L - 1) // 2
                if p < num_angles:
                    cat_params_host[offset] = sigmoid(raw_params[raw_idx]) * Float32(PI)
                else:
                    cat_params_host[offset] = softplus(raw_params[raw_idx])
            else:
                cat_params_host[offset] = softplus(raw_params[raw_idx])
            offset += 1


fn _num_params_for_variant(L: Int, cat_kernel_type: Int) -> Int:
    """Return the number of parameters for a given variant and level count."""
    if cat_kernel_type == CAT_KERNEL_GD:
        return 1
    elif cat_kernel_type == CAT_KERNEL_CR:
        return L
    elif cat_kernel_type == CAT_KERNEL_EHH:
        return L * (L - 1) // 2
    elif cat_kernel_type == CAT_KERNEL_HH:
        return L * (L - 1) // 2
    elif cat_kernel_type == CAT_KERNEL_FE:
        return L * (L + 1) // 2
    else:
        return 1
