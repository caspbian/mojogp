"""LMC BBMM Step - Separated to reduce compiler monomorphization burden.

This file contains the bbmm_with_precond call for LMC training, isolated
in its own module to prevent the Mojo compiler from stack-overflowing when
monomorphizing the full training function.

The key insight: bbmm_with_precond[MaterializedLMCGradientAdapter, LMCPreconditioner]
compiles fine in isolation (proven by test_lmc_compile.mojo), but crashes when
embedded in the large train_gp_multi_output_lmc() function body.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import isnan

from .constants import float_dtype
from .matvec_provider import MaterializedProvider
from .gradient_provider import IsotropicGradientAdapter
from .combined_inv_quad_logdet import bbmm_with_precond, CGBufferPool, UnifiedBBMMResult
from .pivoted_cholesky import build_pivoted_cholesky_precond_unified, PivotedCholeskyPrecond
from .lmc_provider import MaterializedLMCGradientAdapter
from .lmc_preconditioner import LMCPreconditioner
from .cg_solver import kernel_copy, kernel_dot_batched


# =============================================================================
# BBMM Step Result
# =============================================================================

struct LMCBBMMStepResult:
    """Result of a single BBMM step in LMC training."""
    var nll: Float32
    var num_iterations: Int
    var solution: DeviceBuffer[float_dtype]
    var grad_lengthscales: HostBuffer[float_dtype]
    var G_A_all: HostBuffer[float_dtype]
    var grad_noise_per_task: HostBuffer[float_dtype]
    var valid: Bool  # False if NLL was NaN
    
    fn __init__(
        out self,
        nll: Float32,
        num_iterations: Int,
        var solution: DeviceBuffer[float_dtype],
        var grad_lengthscales: HostBuffer[float_dtype],
        var G_A_all: HostBuffer[float_dtype],
        var grad_noise_per_task: HostBuffer[float_dtype],
        valid: Bool,
    ):
        self.nll = nll
        self.num_iterations = num_iterations
        self.solution = solution^
        self.grad_lengthscales = grad_lengthscales^
        self.G_A_all = G_A_all^
        self.grad_noise_per_task = grad_noise_per_task^
        self.valid = valid


# =============================================================================
# Preconditioner Rebuild
# =============================================================================

fn lmc_rebuild_preconditioner(
    ctx: DeviceContext,
    x_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    all_params_device: DeviceBuffer[float_dtype],
    all_params_host: HostBuffer[float_dtype],
    lengthscales: HostBuffer[float_dtype],
    kernel_types: HostBuffer[float_dtype],
    kernel_params1: HostBuffer[float_dtype],
    kernel_params2: HostBuffer[float_dtype],
    noise_per_task: HostBuffer[float_dtype],
    C_all: HostBuffer[float_dtype],
    pc_holders: UnsafePointer[PivotedCholeskyPrecond, MutAnyOrigin],
    mut lmc_precond_out: LMCPreconditioner,
    mut L_all_device_out: DeviceBuffer[float_dtype],
    mut actual_rank_out: Int,
    R: Int,
    T: Int,
    n: Int,
    dim: Int,
    precond_rank: Int,
    num_probes: Int,
    verbose: Bool,
    iteration: Int,
    use_ard: Bool = False,
) raises:
    """Rebuild the LMC preconditioner from current hyperparameters."""
    alias PARAMS_PER_LATENT = 2
    
    if verbose and iteration > 0:
        print("  Rebuilding LMC preconditioner at iter", iteration)
    ctx.synchronize()
    
    for s in range(R):
        var ls_scalar: Float32
        if use_ard:
            var ls_sum = Float32(0.0)
            for d_idx in range(dim):
                ls_sum += lengthscales.unsafe_ptr()[s * dim + d_idx]
            ls_scalar = ls_sum / Float32(dim)
        else:
            ls_scalar = lengthscales.unsafe_ptr()[s]
        var kt = Int(kernel_types.unsafe_ptr()[s])
        var precond_base = MaterializedProvider(
            ctx, x_device_ptr, all_params_device.unsafe_ptr().offset(s * PARAMS_PER_LATENT),
            n, dim, kt, use_ard,
            ls_scalar, Float32(1.0), Float32(0.0),
            kernel_params1.unsafe_ptr()[s], kernel_params2.unsafe_ptr()[s],
        )
        if use_ard:
            # Set per-dimension lengthscales (pointer to [d] for latent s)
            precond_base.update_hyperparams_ard(
                lengthscales.unsafe_ptr().offset(s * dim),
                Float32(1.0), Float32(0.0),
            )
        var precond_adapter = IsotropicGradientAdapter(precond_base^)
        var new_pc = build_pivoted_cholesky_precond_unified(
            precond_adapter, precond_rank, max_num_cols=1 + num_probes
        )
        (pc_holders + s).destroy_pointee()
        (pc_holders + s).init_pointee_move(new_pc^)
        _ = precond_adapter
    
    var actual_rank = pc_holders[0].rank
    var L_all_device = ctx.enqueue_create_buffer[float_dtype](R * n * actual_rank)
    for s in range(R):
        var L_host_temp = ctx.enqueue_create_host_buffer[float_dtype](n * actual_rank)
        ctx.enqueue_copy(dst_buf=L_host_temp, src_buf=pc_holders[s].L)
        ctx.synchronize()
        var L_all_host_temp = ctx.enqueue_create_host_buffer[float_dtype](R * n * actual_rank)
        ctx.enqueue_copy(dst_buf=L_all_host_temp, src_buf=L_all_device)
        ctx.synchronize()
        for i in range(n * actual_rank):
            L_all_host_temp[s * n * actual_rank + i] = L_host_temp[i]
        ctx.enqueue_copy(dst_buf=L_all_device, src_buf=L_all_host_temp)
        ctx.synchronize()
    
    var lmc_precond = LMCPreconditioner(
        ctx, L_all_device, C_all, noise_per_task,
        n, actual_rank, T, R,
        max_num_cols=1 + num_probes,
    )
    
    ctx.synchronize()
    lmc_precond_out = lmc_precond^
    L_all_device_out = L_all_device^
    actual_rank_out = actual_rank


# =============================================================================
# A_s and Noise Gradient Computation
# =============================================================================

fn _compute_A_and_noise_gradients(
    ctx: DeviceContext,
    x_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    all_params_device: DeviceBuffer[float_dtype],
    result: UnifiedBBMMResult,
    lengthscales: HostBuffer[float_dtype],
    kernel_types: HostBuffer[float_dtype],
    kernel_params1: HostBuffer[float_dtype],
    kernel_params2: HostBuffer[float_dtype],
    R: Int,
    T: Int,
    n: Int,
    dim: Int,
    nT: Int,
    num_probes: Int,
    mut G_A_out: HostBuffer[float_dtype],
    mut grad_noise_out: HostBuffer[float_dtype],
    use_ard: Bool = False,
) raises:
    """Compute A_s gradients and per-task noise gradients from CG solutions."""
    alias PARAMS_PER_LATENT = 2
    var TT = T * T
    
    # Workspace buffers
    var dot_result_device = ctx.enqueue_create_buffer[float_dtype](1)
    var dot_result_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    var temp_v = ctx.enqueue_create_buffer[float_dtype](n)
    var kx_alpha = ctx.enqueue_create_buffer[float_dtype](n * T)
    var kx_rf_j = ctx.enqueue_create_buffer[float_dtype](n)
    
    # ---- A_s gradients ----
    for s in range(R):
        var ls_scalar: Float32
        if use_ard:
            var ls_sum = Float32(0.0)
            for d_idx in range(dim):
                ls_sum += lengthscales.unsafe_ptr()[s * dim + d_idx]
            ls_scalar = ls_sum / Float32(dim)
        else:
            ls_scalar = lengthscales.unsafe_ptr()[s]
        var kt = Int(kernel_types.unsafe_ptr()[s])
        var grad_base = MaterializedProvider(
            ctx, x_device_ptr, all_params_device.unsafe_ptr().offset(s * PARAMS_PER_LATENT),
            n, dim, kt, use_ard,
            ls_scalar, Float32(1.0), Float32(0.0),
            kernel_params1.unsafe_ptr()[s], kernel_params2.unsafe_ptr()[s],
        )
        if use_ard:
            # Set per-dimension lengthscales (pointer to [d] for latent s)
            grad_base.update_hyperparams_ard(
                lengthscales.unsafe_ptr().offset(s * dim),
                Float32(1.0), Float32(0.0),
            )
        
        for t in range(T):
            ctx.enqueue_function[kernel_copy](
                temp_v.unsafe_ptr(),
                result.solution.unsafe_ptr().offset(t * n), n,
                grid_dim=((n + 255) // 256,), block_dim=(256,),
            )
            grad_base.forward_matvec(kx_alpha.unsafe_ptr().offset(t * n), temp_v.unsafe_ptr(), 1)
        
        for i in range(T):
            for j in range(T):
                ctx.enqueue_function[kernel_dot_batched](
                    result.solution.unsafe_ptr().offset(i * n),
                    kx_alpha.unsafe_ptr().offset(j * n),
                    dot_result_device.unsafe_ptr(), n, 1,
                    grid_dim=(1, 1), block_dim=(256, 1),
                )
                ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
                ctx.synchronize()
                G_A_out.unsafe_ptr()[s * TT + i * T + j] = -dot_result_host[0]
        
        for k in range(num_probes):
            for j in range(T):
                ctx.enqueue_function[kernel_copy](
                    temp_v.unsafe_ptr(),
                    result.right_factors.unsafe_ptr().offset(k * nT + j * n), n,
                    grid_dim=((n + 255) // 256,), block_dim=(256,),
                )
                grad_base.forward_matvec(kx_rf_j.unsafe_ptr(), temp_v.unsafe_ptr(), 1)
                
                for i in range(T):
                    ctx.enqueue_function[kernel_dot_batched](
                        result.probe_solutions.unsafe_ptr().offset(k * nT + i * n),
                        kx_rf_j.unsafe_ptr(),
                        dot_result_device.unsafe_ptr(), n, 1,
                        grid_dim=(1, 1), block_dim=(256, 1),
                    )
                    ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
                    ctx.synchronize()
                    G_A_out.unsafe_ptr()[s * TT + i * T + j] += dot_result_host[0] / Float32(num_probes)
        
        for idx in range(TT):
            G_A_out.unsafe_ptr()[s * TT + idx] = Float32(0.5) * G_A_out.unsafe_ptr()[s * TT + idx] / Float32(nT)
    
    # ---- Per-task noise gradients ----
    for t in range(T):
        ctx.enqueue_function[kernel_dot_batched](
            result.solution.unsafe_ptr().offset(t * n),
            result.solution.unsafe_ptr().offset(t * n),
            dot_result_device.unsafe_ptr(), n, 1,
            grid_dim=(1, 1), block_dim=(256, 1),
        )
        ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
        ctx.synchronize()
        var data_term = -dot_result_host[0]
        
        var trace_sum = Float32(0.0)
        for k in range(num_probes):
            ctx.enqueue_function[kernel_dot_batched](
                result.probe_solutions.unsafe_ptr().offset(k * nT + t * n),
                result.right_factors.unsafe_ptr().offset(k * nT + t * n),
                dot_result_device.unsafe_ptr(), n, 1,
                grid_dim=(1, 1), block_dim=(256, 1),
            )
            ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
            ctx.synchronize()
            trace_sum += dot_result_host[0]
        
        grad_noise_out.unsafe_ptr()[t] = Float32(0.5) * (data_term + trace_sum / Float32(num_probes)) / Float32(nT)
    
    _ = dot_result_device
    _ = temp_v
    _ = kx_alpha
    _ = kx_rf_j


# =============================================================================
# BBMM Step
# =============================================================================

fn lmc_bbmm_step(
    ctx: DeviceContext,
    lmc_adapter: MaterializedLMCGradientAdapter,
    lmc_precond: LMCPreconditioner,
    y_blocked_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    all_params_device: DeviceBuffer[float_dtype],
    lengthscales: HostBuffer[float_dtype],
    kernel_types: HostBuffer[float_dtype],
    kernel_params1: HostBuffer[float_dtype],
    kernel_params2: HostBuffer[float_dtype],
    mut cg_pool: CGBufferPool,
    nT: Int,
    n: Int,
    dim: Int,
    R: Int,
    T: Int,
    num_probes: Int,
    max_cg_iter: Int,
    max_tridiag_iter: Int,
    cg_tol: Float32,
    iteration: Int,
    should_rebuild: Bool,
    use_ard: Bool = False,
) raises -> LMCBBMMStepResult:
    """Run one BBMM step: CG solve + gradient computation.
    
    This function is in a separate file from the training loop to reduce
    the monomorphization burden on the Mojo compiler.
    """
    var num_cols_total = 1 + num_probes
    cg_pool.ensure_capacity(ctx, nT, num_cols_total, num_probes, max_tridiag_iter, 0)
    
    var result = bbmm_with_precond[MaterializedLMCGradientAdapter, LMCPreconditioner](
        lmc_adapter, lmc_precond,
        y_blocked_device_ptr, nT, cg_pool,
        num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
        iteration=iteration,
        recycle_alpha=iteration > 0 and not should_rebuild,
    )
    
    var nll = result.nll
    
    # Determine number of lengthscale params per latent
    var num_ls_per_latent = lmc_adapter.get_num_ls_params_per_latent()
    var total_ls_grads = R * num_ls_per_latent
    
    if isnan(nll):
        var empty_ls = HostBuffer[float_dtype](ctx, total_ls_grads)
        var empty_ga = HostBuffer[float_dtype](ctx, R * T * T)
        var empty_gn = HostBuffer[float_dtype](ctx, T)
        var empty_sol = ctx.enqueue_create_buffer[float_dtype](nT)
        return LMCBBMMStepResult(nll, result.num_iterations, empty_sol^, empty_ls^, empty_ga^, empty_gn^, False)
    
    # Extract lengthscale gradients: 1 per latent (isotropic) or dim per latent (ARD)
    var grad_lengthscales = HostBuffer[float_dtype](ctx, total_ls_grads)
    for s in range(R):
        var param_offset = lmc_adapter.get_param_offset(s)
        for d_idx in range(num_ls_per_latent):
            grad_lengthscales.unsafe_ptr()[s * num_ls_per_latent + d_idx] = result.gradients[param_offset + d_idx]
    
    var G_A_all = HostBuffer[float_dtype](ctx, R * T * T)
    var grad_noise = HostBuffer[float_dtype](ctx, T)
    _compute_A_and_noise_gradients(
        ctx, x_device_ptr, all_params_device, result,
        lengthscales, kernel_types, kernel_params1, kernel_params2,
        R, T, n, dim, nT, num_probes,
        G_A_all, grad_noise,
        use_ard=use_ard,
    )
    
    var solution_copy = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.enqueue_function[kernel_copy](
        solution_copy.unsafe_ptr(), result.solution.unsafe_ptr(), nT,
        grid_dim=((nT + 255) // 256,), block_dim=(256,),
    )
    ctx.synchronize()
    
    return LMCBBMMStepResult(
        nll, result.num_iterations, solution_copy^,
        grad_lengthscales^, G_A_all^, grad_noise^, True,
    )
