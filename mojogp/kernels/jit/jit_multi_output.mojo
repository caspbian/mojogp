"""JIT multi-output (Kronecker/ICM) training using ErasedJITProvider.

Adapts the AOT multi-output training loop for the fn-ptr JIT engine path.
The base kernel (K_X) is provided via fn ptrs from a kernel .so.
Kronecker structure (B matrix, per-task noise) is handled entirely in the engine.

Model: K_full = outputscale * (K_X ⊗ B) + D
where B = WW^T + diag(softplus(raw_v)), D = diag(noise_per_task ⊗ I_n)

Uses bbmm_with_precond for NLL + kernel param gradients.
B gradient and per-task noise gradient computed from CG solution + right factors.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import sqrt, ceildiv, exp as math_exp, log

from kernels.jit.erased_provider import ErasedJITProvider
from kernels.jit.jit_training import softplus, inv_softplus, pow_float32, compute_cosine_lr, clip_gradient
from kernels.kronecker_direct_provider import KroneckerDirectProvider
from kernels.jit.fused_kronecker_provider import FusedKroneckerProvider
from kernels.task_covariance import TaskCovariance
from kernels.combined_inv_quad_logdet import bbmm_with_precond, CGBufferPool
from kernels.pivoted_cholesky import build_pivoted_cholesky_precond_unified, PivotedCholeskyPrecond
from kernels.kronecker_preconditioner import KroneckerPreconditioner
from kernels.bbmm_gpu_kernels import kernel_dot_matrix
from kernels.cg_solver import kernel_dot_batched
from kernels.constants import float_dtype
from time import perf_counter_ns
from python import PythonObject
from kernels.jit.jit_progress import emit_progress_event, progress_interval_should_emit


# =============================================================================
# Multi-Output Result
# =============================================================================


fn kernel_subtract_task_means_blocked(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    mean_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
) -> None:
    var idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var nT = n * num_tasks
    if idx >= nT:
        return
    var task = idx // n
    out_ptr[UInt(idx)] = y_ptr[UInt(idx)] - mean_ptr[UInt(task)]


struct MultiOutputJITResult(Movable):
    """Result from JIT multi-output training."""
    var final_params: List[Float32]
    var num_kernel_params: Int
    var outputscale: Float32
    var noise_per_task: List[Float32]
    var B_flat: List[Float32]
    var mean_per_task: List[Float32]
    var alpha_blocked: List[Float32]
    var nll_history: List[Float32]
    var cg_iterations_history: List[Int]
    var iter_times_ns: List[Int]
    var final_nll: Float32
    var iterations: Int
    var converged: Bool
    var num_tasks: Int
    var task_rank: Int
    var n: Int
    var precond_rebuild_count: Int

    fn __init__(
        out self, var final_params: List[Float32], num_kernel_params: Int,
        outputscale: Float32, var noise_per_task: List[Float32],
        var B_flat: List[Float32], var mean_per_task: List[Float32], var alpha_blocked: List[Float32],
        var nll_history: List[Float32], var cg_iterations_history: List[Int],
        var iter_times_ns: List[Int],
        final_nll: Float32, iterations: Int, converged: Bool,
        num_tasks: Int, task_rank: Int, n: Int,
        precond_rebuild_count: Int,
    ):
        self.final_params = final_params^
        self.num_kernel_params = num_kernel_params
        self.outputscale = outputscale
        self.noise_per_task = noise_per_task^
        self.B_flat = B_flat^
        self.mean_per_task = mean_per_task^
        self.alpha_blocked = alpha_blocked^
        self.nll_history = nll_history^
        self.cg_iterations_history = cg_iterations_history^
        self.iter_times_ns = iter_times_ns^
        self.final_nll = final_nll
        self.iterations = iterations
        self.converged = converged
        self.num_tasks = num_tasks
        self.task_rank = task_rank
        self.n = n
        self.precond_rebuild_count = precond_rebuild_count

    fn __moveinit__(out self, owned other: Self):
        self.final_params = other.final_params^
        self.num_kernel_params = other.num_kernel_params
        self.outputscale = other.outputscale
        self.noise_per_task = other.noise_per_task^
        self.B_flat = other.B_flat^
        self.mean_per_task = other.mean_per_task^
        self.alpha_blocked = other.alpha_blocked^
        self.nll_history = other.nll_history^
        self.cg_iterations_history = other.cg_iterations_history^
        self.iter_times_ns = other.iter_times_ns^
        self.final_nll = other.final_nll
        self.iterations = other.iterations
        self.converged = other.converged
        self.num_tasks = other.num_tasks
        self.task_rank = other.task_rank
        self.n = other.n
        self.precond_rebuild_count = other.precond_rebuild_count


# =============================================================================
# Training
# =============================================================================


fn train_multi_output_jit(
    base_provider: ErasedJITProvider,
    ctx: DeviceContext,
    y_blocked_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
    num_kernel_params: Int,
    initial_params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    trainable_mask_ptr: UnsafePointer[Bool, MutAnyOrigin],
    initial_noise_per_task_ptr: UnsafePointer[Float32, MutAnyOrigin],
    initial_outputscale: Float32,
    initial_mean_per_task_ptr: UnsafePointer[Float32, MutAnyOrigin],
    fixed_noise_vector_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_mode: Int,
    max_iterations: Int = 100,
    learning_rate: Float32 = Float32(0.05),
    task_rank: Int = -1,
    num_probes: Int = 10,
    max_cg_iter: Int = 200,
    max_tridiag_iter: Int = 30,
    cg_tol: Float32 = Float32(1e-2),
    precond_rank: Int = 15,
    precond_method: Int = 0,
    precond_rebuild_threshold: Float32 = Float32(0.5),
    use_cosine_lr: Bool = False,
    early_stop_patience: Int = 15,
    early_stop_tol: Float32 = Float32(1e-4),
    verbose: Bool = True,
    progress_callback: PythonObject = None,
    progress_interval: Int = 1,
    progress_enabled: Bool = False,
) raises -> MultiOutputJITResult:
    """Train multi-output GP with Kronecker/ICM via JIT fn-ptr provider.

    Uses bbmm_with_precond for kernel param gradients.
    B and per-task noise gradients computed from CG solution + right factors.
    """
    var T = num_tasks
    var R = task_rank
    if R < 0:
        R = T
    var nT = n * T
    var fixed_vector_noise = noise_mode == 1

    # =========================================================================
    # 1. Initialize parameters
    # =========================================================================

    var raw_params = List[Float32]()
    for p in range(num_kernel_params):
        raw_params.append(inv_softplus(initial_params_host_ptr[p]))
    var raw_outputscale = inv_softplus(initial_outputscale)
    var raw_noise = List[Float32]()
    for t in range(T):
        raw_noise.append(inv_softplus(initial_noise_per_task_ptr[t]))
    var raw_mean = List[Float32]()
    for t in range(T):
        raw_mean.append(initial_mean_per_task_ptr[t])

    # Task covariance B = WW^T + diag(softplus(raw_v))
    var task_cov = TaskCovariance(ctx, T, R)

    # Adam state: kernel [N] + outputscale [1] + noise [T] + W [T*R] + v [T] + mean [T]
    var total_adam = num_kernel_params + 1 + T + T * R + T + T
    var m_adam = List[Float32]()
    var v_adam = List[Float32]()
    for _ in range(total_adam):
        m_adam.append(Float32(0))
        v_adam.append(Float32(0))

    var best_nll = Float32(1e30)
    var best_nll_seen = Float32(1e30)
    var has_valid_best = False
    var patience_counter = 0
    var converged = False
    var actual_iters = 0

    # Best params
    var best_raw_params = List[Float32]()
    for p in range(num_kernel_params):
        best_raw_params.append(raw_params[p])
    var best_raw_os = raw_outputscale
    var best_raw_noise = List[Float32]()
    for t in range(T):
        best_raw_noise.append(raw_noise[t])
    var best_raw_mean = List[Float32]()
    for t in range(T):
        best_raw_mean.append(raw_mean[t])
    var best_sol_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    var best_B_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    ctx.synchronize()
    for i in range(nT):
        best_sol_host.unsafe_ptr()[i] = Float32(0)
    for i in range(T * T):
        best_B_host.unsafe_ptr()[i] = Float32(0)

    # =========================================================================
    # 2. GPU buffers
    # =========================================================================

    var params_host_buf = ctx.enqueue_create_host_buffer[float_dtype](num_kernel_params)
    var mean_host_buf = ctx.enqueue_create_host_buffer[float_dtype](T)
    var mean_device_buf = ctx.enqueue_create_buffer[float_dtype](T)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.synchronize()

    var num_cols = 1 + num_probes
    var precond_error_tol = Float32(1e-3)
    var effective_precond_rank = precond_rank
    var use_preconditioner = effective_precond_rank > 0
    if fixed_vector_noise:
        use_preconditioner = False
        effective_precond_rank = 0
    if precond_method == 0:
        precond_error_tol = Float32(0.0)

    var pool = CGBufferPool(ctx, nT, num_cols, num_probes, max_tridiag_iter)

    # B gradient workspace
    var kx_alpha = ctx.enqueue_create_buffer[float_dtype](n * T)
    var kx_rf_batch = ctx.enqueue_create_buffer[float_dtype](n * T)
    var dot_matrix_device = ctx.enqueue_create_buffer[float_dtype](T * T)
    var dot_matrix_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    var dot_result_device = ctx.enqueue_create_buffer[float_dtype](1)
    var dot_result_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    var G_B = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    var fixed_noise_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for idx_noise in range(nT):
        fixed_noise_host.unsafe_ptr()[idx_noise] = fixed_noise_vector_ptr[idx_noise]

    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var eps = Float32(1e-8)
    var t_step = 1

    # =========================================================================
    # 3. Build initial preconditioner ONCE (rebuild only on threshold)
    # =========================================================================

    # Initial params for preconditioner
    for p in range(num_kernel_params):
        params_host_buf.unsafe_ptr()[p] = softplus(raw_params[p])
    var init_os = softplus(raw_outputscale)

    # Build base K_X preconditioner (n-dim, fast)
    var kx_precond_base = base_provider.clone()
    kx_precond_base.update_params(params_host_buf.unsafe_ptr())
    kx_precond_base.update_noise(Float32(0))
    var kx_precond_holder = build_pivoted_cholesky_precond_unified(
        kx_precond_base,
        rank=effective_precond_rank,
        error_tol=precond_error_tol,
        max_num_cols=num_cols,
        precond_method=precond_method,
    )

    # Build initial Kronecker preconditioner
    task_cov.update_B()
    var precond_B_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    var precond_noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    for s in range(T):
        for t_idx in range(T):
            precond_B_host.unsafe_ptr()[s * T + t_idx] = task_cov.B.unsafe_ptr()[s * T + t_idx]
        precond_noise_host.unsafe_ptr()[s] = softplus(raw_noise[s])

    var kron_precond = KroneckerPreconditioner(
        ctx, kx_precond_holder.L, n, kx_precond_holder.rank,
        T, init_os, precond_B_host, precond_noise_host,
        max_num_cols=num_cols,
    )
    for i in range(T * T):
        best_B_host.unsafe_ptr()[i] = task_cov.B.unsafe_ptr()[i]
    var precond_rebuild_count = 0
    var nll_history = List[Float32]()
    var cg_iterations_history = List[Int]()
    var iter_times_ns = List[Int]()
    var last_rebuild_params = ctx.enqueue_create_host_buffer[float_dtype](num_kernel_params)
    var last_rebuild_noise = ctx.enqueue_create_host_buffer[float_dtype](T)
    var last_rebuild_B = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    ctx.synchronize()
    for p in range(num_kernel_params):
        last_rebuild_params.unsafe_ptr()[p] = params_host_buf.unsafe_ptr()[p]
    var last_rebuild_outputscale = init_os
    for s in range(T):
        last_rebuild_noise.unsafe_ptr()[s] = precond_noise_host.unsafe_ptr()[s]
        for t_idx in range(T):
            last_rebuild_B.unsafe_ptr()[s * T + t_idx] = precond_B_host.unsafe_ptr()[s * T + t_idx]

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "multi_output_icm",
            "unknown",
            "start",
            0,
            max_iterations,
            precond_rank=kx_precond_holder.rank,
            precond_rebuild_count=precond_rebuild_count,
        )

    # =========================================================================
    # 4. Training loop
    # =========================================================================

    for iteration in range(max_iterations):
        var iter_start = perf_counter_ns()
        actual_iters = iteration + 1

        # --- Apply current params ---
        for p in range(num_kernel_params):
            params_host_buf.unsafe_ptr()[p] = softplus(raw_params[p])

        # Create provider for this iteration
        var kron_base = base_provider.clone()
        kron_base.update_params(params_host_buf.unsafe_ptr())
        kron_base.update_noise(Float32(0))

        var os_val = softplus(raw_outputscale)
        var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
        ctx.synchronize()
        for t in range(T):
            noise_host.unsafe_ptr()[t] = softplus(raw_noise[t])

        # Build Fused Kronecker provider (single kernel launch per CG iter)
        task_cov.update_B()
        var kron_provider = FusedKroneckerProvider(
            kron_base^, ctx, T, os_val, task_cov.B, noise_host,
            noise_mode, fixed_noise_host,
        )

        var should_rebuild = False
        if iteration > 0:
            var max_rel = Float32(0)
            for p in range(num_kernel_params):
                var prev = last_rebuild_params.unsafe_ptr()[p]
                var cur = params_host_buf.unsafe_ptr()[p]
                var rel = abs(cur - prev) / (abs(prev) + Float32(1e-8))
                if rel > max_rel:
                    max_rel = rel
            var os_rel = abs(os_val - last_rebuild_outputscale) / (abs(last_rebuild_outputscale) + Float32(1e-8))
            if os_rel > max_rel:
                max_rel = os_rel
            for t in range(T):
                var prev_noise = last_rebuild_noise.unsafe_ptr()[t]
                var cur_noise = noise_host.unsafe_ptr()[t]
                var rel_noise = abs(cur_noise - prev_noise) / (abs(prev_noise) + Float32(1e-8))
                if rel_noise > max_rel:
                    max_rel = rel_noise
            for i in range(T * T):
                var prev_b = last_rebuild_B.unsafe_ptr()[i]
                var cur_b = task_cov.B.unsafe_ptr()[i]
                var rel_b = abs(cur_b - prev_b) / (abs(prev_b) + Float32(1e-8))
                if rel_b > max_rel:
                    max_rel = rel_b
            should_rebuild = max_rel > precond_rebuild_threshold
        if should_rebuild:
            kx_precond_base.update_params(params_host_buf.unsafe_ptr())
            kx_precond_base.update_noise(Float32(0))
            kx_precond_holder = build_pivoted_cholesky_precond_unified(
                kx_precond_base,
                rank=effective_precond_rank,
                error_tol=precond_error_tol,
                max_num_cols=num_cols,
                precond_method=precond_method,
            )
            for s in range(T):
                for t_idx in range(T):
                    precond_B_host.unsafe_ptr()[s * T + t_idx] = task_cov.B.unsafe_ptr()[s * T + t_idx]
                precond_noise_host.unsafe_ptr()[s] = noise_host.unsafe_ptr()[s]
            kron_precond = KroneckerPreconditioner(
                ctx, kx_precond_holder.L, n, kx_precond_holder.rank,
                T, os_val, precond_B_host, precond_noise_host,
                max_num_cols=num_cols,
            )
            precond_rebuild_count += 1
            for p in range(num_kernel_params):
                last_rebuild_params.unsafe_ptr()[p] = params_host_buf.unsafe_ptr()[p]
            last_rebuild_outputscale = os_val
            for s in range(T):
                last_rebuild_noise.unsafe_ptr()[s] = noise_host.unsafe_ptr()[s]
                for t_idx in range(T):
                    last_rebuild_B.unsafe_ptr()[s * T + t_idx] = task_cov.B.unsafe_ptr()[s * T + t_idx]

        pool.ensure_capacity(ctx, nT, num_cols, num_probes, max_tridiag_iter, effective_precond_rank, num_kernel_params=kron_provider.num_gradient_params())

        for t in range(T):
            mean_host_buf.unsafe_ptr()[t] = raw_mean[t]
        ctx.enqueue_copy(mean_device_buf, mean_host_buf)
        ctx.enqueue_function[kernel_subtract_task_means_blocked](
            y_centered_device.unsafe_ptr(),
            y_blocked_device_ptr,
            mean_device_buf.unsafe_ptr(),
            n,
            T,
            grid_dim=((nT + 255) // 256,),
            block_dim=(256,),
        )

        # --- BBMM: NLL + kernel param gradients ---
        var bbmm_result = bbmm_with_precond(
            kron_provider, kron_precond, y_centered_device.unsafe_ptr(), nT, pool,
            num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
            iteration=iteration, recycle_alpha=iteration > 0 and not should_rebuild,
            use_preconditioner=use_preconditioner,
        )

        var cg_converged = bbmm_result.num_iterations < max_cg_iter
        if not cg_converged and use_preconditioner:
            if verbose:
                print("MO Iter", iteration, ": CG reached max iterations; rebuilding preconditioner and retrying")
            kx_precond_base.update_params(params_host_buf.unsafe_ptr())
            kx_precond_base.update_noise(Float32(0))
            kx_precond_holder = build_pivoted_cholesky_precond_unified(
                kx_precond_base,
                rank=effective_precond_rank,
                error_tol=precond_error_tol,
                max_num_cols=num_cols,
                precond_method=precond_method,
            )
            for s in range(T):
                for t_idx in range(T):
                    precond_B_host.unsafe_ptr()[s * T + t_idx] = task_cov.B.unsafe_ptr()[s * T + t_idx]
                precond_noise_host.unsafe_ptr()[s] = noise_host.unsafe_ptr()[s]
            kron_precond = KroneckerPreconditioner(
                ctx, kx_precond_holder.L, n, kx_precond_holder.rank,
                T, os_val, precond_B_host, precond_noise_host,
                max_num_cols=num_cols,
            )
            precond_rebuild_count += 1
            for p in range(num_kernel_params):
                last_rebuild_params.unsafe_ptr()[p] = params_host_buf.unsafe_ptr()[p]
            last_rebuild_outputscale = os_val
            for s in range(T):
                last_rebuild_noise.unsafe_ptr()[s] = noise_host.unsafe_ptr()[s]
                for t_idx in range(T):
                    last_rebuild_B.unsafe_ptr()[s * T + t_idx] = task_cov.B.unsafe_ptr()[s * T + t_idx]
            bbmm_result = bbmm_with_precond(
                kron_provider, kron_precond, y_centered_device.unsafe_ptr(), nT, pool,
                num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
                iteration=iteration, recycle_alpha=False,
                use_preconditioner=use_preconditioner,
            )
            cg_converged = bbmm_result.num_iterations < max_cg_iter

        var nll = bbmm_result.nll
        nll_history.append(nll)
        cg_iterations_history.append(bbmm_result.num_iterations)
        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print("MO Iter", iteration, ": NLL =", nll)

        if not cg_converged:
            if verbose:
                print("MO Iter", iteration, ": CG did not converge; stopping with best valid state")
            iter_times_ns.append(Int(perf_counter_ns() - iter_start))
            if not has_valid_best:
                best_nll_seen = nll
                for p in range(num_kernel_params):
                    best_raw_params[p] = raw_params[p]
                best_raw_os = raw_outputscale
                for t in range(T):
                    best_raw_noise[t] = raw_noise[t]
                    best_raw_mean[t] = raw_mean[t]
                for i in range(T * T):
                    best_B_host.unsafe_ptr()[i] = task_cov.B.unsafe_ptr()[i]
                ctx.enqueue_copy(best_sol_host, bbmm_result.solution)
                ctx.synchronize()
                has_valid_best = True
            break

        var sol_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
        ctx.enqueue_copy(sol_host, bbmm_result.solution)
        ctx.synchronize()

        # Save the exact state used for this NLL before Adam mutates parameters.
        if nll < best_nll_seen:
            best_nll_seen = nll
            for p in range(num_kernel_params):
                best_raw_params[p] = raw_params[p]
            best_raw_os = raw_outputscale
            for t in range(T):
                best_raw_noise[t] = raw_noise[t]
                best_raw_mean[t] = raw_mean[t]
            for i in range(T * T):
                best_B_host.unsafe_ptr()[i] = task_cov.B.unsafe_ptr()[i]
            for i in range(nT):
                best_sol_host.unsafe_ptr()[i] = sol_host.unsafe_ptr()[i]
            has_valid_best = True

        # --- B gradient from solution + right_factors ---


        # Use a fresh provider copy for pure K_X matvec
        var kx_provider = base_provider.clone()
        kx_provider.update_params(params_host_buf.unsafe_ptr())
        kx_provider.update_noise(Float32(0))


        # K_X @ alpha (all T task columns, each n elements)
        kx_provider.forward_matvec(kx_alpha.unsafe_ptr(), bbmm_result.solution.unsafe_ptr(), T)


        # Data term: G_B[s,t] = -alpha_s^T @ (K_X @ alpha_t)
        ctx.enqueue_function[kernel_dot_matrix](
            bbmm_result.solution.unsafe_ptr(), kx_alpha.unsafe_ptr(),
            dot_matrix_device.unsafe_ptr(), n, T, T,
            grid_dim=(T * T,), block_dim=(256,),
        )
        ctx.enqueue_copy(dot_matrix_host, dot_matrix_device)
        ctx.synchronize()

        for i in range(T * T):
            G_B.unsafe_ptr()[i] = -dot_matrix_host.unsafe_ptr()[i]

        # Trace term: per probe j

        for j in range(num_probes):
            kx_provider.forward_matvec(
                kx_rf_batch.unsafe_ptr(),
                bbmm_result.right_factors.unsafe_ptr().offset(j * nT), T)
            ctx.enqueue_function[kernel_dot_matrix](
                bbmm_result.probe_solutions.unsafe_ptr().offset(j * nT),
                kx_rf_batch.unsafe_ptr(),
                dot_matrix_device.unsafe_ptr(), n, T, T,
                grid_dim=(T * T,), block_dim=(256,),
            )
            ctx.enqueue_copy(dot_matrix_host, dot_matrix_device)
            ctx.synchronize()
            for i in range(T * T):
                G_B.unsafe_ptr()[i] += dot_matrix_host.unsafe_ptr()[i] / Float32(num_probes)

        # Scale G_B
        for i in range(T * T):
            G_B.unsafe_ptr()[i] = Float32(0.5) * os_val * G_B.unsafe_ptr()[i] / Float32(nT)
            # Clip for stability
            if G_B.unsafe_ptr()[i] > Float32(10):
                G_B.unsafe_ptr()[i] = Float32(10)
            elif G_B.unsafe_ptr()[i] < Float32(-10):
                G_B.unsafe_ptr()[i] = Float32(-10)


        # --- Per-task noise gradient ---
        var grad_noise = List[Float32]()
        if fixed_vector_noise:
            for t in range(T):
                grad_noise.append(Float32(0))
        else:
            for t in range(T):
                # Data term: -alpha_t^T @ alpha_t
                ctx.enqueue_function[kernel_dot_batched](
                    bbmm_result.solution.unsafe_ptr().offset(t * n),
                    bbmm_result.solution.unsafe_ptr().offset(t * n),
                    dot_result_device.unsafe_ptr(), n, 1,
                    grid_dim=(1, 1), block_dim=(256, 1),
                )
                ctx.enqueue_copy(dot_result_host, dot_result_device)
                ctx.synchronize()
                var data_term = -dot_result_host.unsafe_ptr()[0]

                # Trace term
                var trace_sum = Float32(0)
                for j in range(num_probes):
                    ctx.enqueue_function[kernel_dot_batched](
                        bbmm_result.probe_solutions.unsafe_ptr().offset(j * nT + t * n),
                        bbmm_result.right_factors.unsafe_ptr().offset(j * nT + t * n),
                        dot_result_device.unsafe_ptr(), n, 1,
                        grid_dim=(1, 1), block_dim=(256, 1),
                    )
                    ctx.enqueue_copy(dot_result_host, dot_result_device)
                    ctx.synchronize()
                    trace_sum += dot_result_host.unsafe_ptr()[0]

                grad_noise.append(Float32(0.5) * (data_term + trace_sum / Float32(num_probes)) / Float32(nT))


        # --- Adam updates ---
        var idx = 0
        var effective_lr = learning_rate
        if use_cosine_lr:
            effective_lr = compute_cosine_lr(learning_rate, iteration, max_iterations)

        # Kernel params (from BBMM gradients)
        for p in range(num_kernel_params):
            if trainable_mask_ptr[p]:
                var grad_p = bbmm_result.gradients[p]
                # Softplus chain rule
                var sigmoid = Float32(1.0) / (Float32(1.0) + math_exp(-raw_params[p]))
                grad_p = grad_p * sigmoid

                m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * grad_p
                v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * grad_p * grad_p
                var m_hat = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
                var v_hat = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
                raw_params[p] -= effective_lr * m_hat / (sqrt(v_hat) + eps)
            idx += 1


        # Outputscale
        var grad_os = bbmm_result.gradients[num_kernel_params]  # Last kernel gradient = outputscale
        var sigmoid_os = Float32(1.0) / (Float32(1.0) + math_exp(-raw_outputscale))
        grad_os = grad_os * sigmoid_os
        m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * grad_os
        v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * grad_os * grad_os
        var m_hat_os = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
        var v_hat_os = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
        raw_outputscale -= effective_lr * m_hat_os / (sqrt(v_hat_os) + eps)
        idx += 1


        # Per-task noise
        for t in range(T):
            if not fixed_vector_noise:
                var gn = grad_noise[t] * (Float32(1.0) / (Float32(1.0) + math_exp(-raw_noise[t])))
                m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * gn
                v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * gn * gn
                var mh = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
                var vh = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
                raw_noise[t] -= effective_lr * mh / (sqrt(vh) + eps)
            idx += 1


        # W gradients from G_B: grad_W[s,r] = 2 * sum_t G_B[s,t] * W[t,r]
        for s in range(T):
            for r in range(R):
                var gw = Float32(0)
                for t_idx in range(T):
                    gw += G_B.unsafe_ptr()[s * T + t_idx] * task_cov.W.unsafe_ptr()[t_idx * R + r]
                gw = Float32(2.0) * gw
                m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * gw
                v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * gw * gw
                var mh = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
                var vh = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
                task_cov.W.unsafe_ptr()[s * R + r] -= effective_lr * mh / (sqrt(vh) + eps)
                idx += 1


        # v gradients from G_B: grad_raw_v[t] = G_B[t,t] * softplus'(raw_v[t])
        for t in range(T):
            var gv = G_B.unsafe_ptr()[t * T + t] * (Float32(1.0) / (Float32(1.0) + math_exp(-task_cov.raw_v.unsafe_ptr()[t])))
            m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * gv
            v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * gv * gv
            var mh = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
            var vh = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
            task_cov.raw_v.unsafe_ptr()[t] -= effective_lr * mh / (sqrt(vh) + eps)
            idx += 1


        # Per-task mean (simple gradient from NLL)
        for t in range(T):
            # Mean gradient: -alpha_t sum (from solution)
            var mean_grad = Float32(0)
            for i in range(n):
                mean_grad -= sol_host.unsafe_ptr()[t * n + i]
            mean_grad = clip_gradient(mean_grad / Float32(nT))
            m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * mean_grad
            v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * mean_grad * mean_grad
            var mh = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
            var vh = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
            raw_mean[t] -= effective_lr * mh / (sqrt(vh) + eps)
            idx += 1


        t_step += 1
        var iter_time_ns = Int(perf_counter_ns() - iter_start)
        iter_times_ns.append(iter_time_ns)
        if progress_enabled and progress_interval_should_emit(iteration, max_iterations, progress_interval):
            emit_progress_event(
                progress_callback,
                "train",
                "multi_output_icm",
                "unknown",
                "iteration",
                actual_iters,
                max_iterations,
                nll=nll,
                best_nll=best_nll_seen,
                cg_iter=bbmm_result.num_iterations,
                iter_time_ns=iter_time_ns,
                noise=softplus(raw_noise[0]) if T > 0 else Float32(0.0),
                mean=raw_mean[0] if T > 0 else Float32(0.0),
                precond_rank=kx_precond_holder.rank,
                precond_rebuild_count=precond_rebuild_count,
            )

        if early_stop_tol > Float32(0.0):
            if nll < best_nll - early_stop_tol:
                best_nll = nll
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    converged = True
                    if verbose:
                        print("Early stopping at iteration", actual_iters)
                    if progress_enabled:
                        emit_progress_event(
                            progress_callback,
                            "train",
                            "multi_output_icm",
                            "unknown",
                            "early_stop",
                            actual_iters,
                            max_iterations,
                            nll=nll,
                            best_nll=best_nll_seen,
                            cg_iter=bbmm_result.num_iterations,
                            iter_time_ns=iter_time_ns,
                            noise=softplus(raw_noise[0]) if T > 0 else Float32(0.0),
                            mean=raw_mean[0] if T > 0 else Float32(0.0),
                            precond_rank=kx_precond_holder.rank,
                            precond_rebuild_count=precond_rebuild_count,
                            converged=converged,
                        )
                    break

        _ = kron_provider
        _ = bbmm_result
        _ = noise_host
        _ = sol_host
    _ = fixed_noise_host

    # =========================================================================
    # 4. Build result
    # =========================================================================

    var final_params = List[Float32]()
    for p in range(num_kernel_params):
        final_params.append(softplus(best_raw_params[p]))
    var final_os = softplus(best_raw_os)
    var final_noise = List[Float32]()
    for t in range(T):
        final_noise.append(softplus(best_raw_noise[t]))
    var final_mean = List[Float32]()
    for t in range(T):
        final_mean.append(best_raw_mean[t])

    # Get final B from the best-seen task covariance state
    var B_flat = List[Float32]()
    for i in range(T * T):
        B_flat.append(best_B_host.unsafe_ptr()[i])
    var alpha_blocked = List[Float32]()
    for i in range(nT):
        alpha_blocked.append(best_sol_host.unsafe_ptr()[i])

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "multi_output_icm",
            "unknown",
            "complete",
            actual_iters,
            max_iterations,
            nll=best_nll_seen,
            best_nll=best_nll_seen,
            noise=final_noise[0] if T > 0 else Float32(0.0),
            mean=final_mean[0] if T > 0 else Float32(0.0),
            precond_rank=kx_precond_holder.rank,
            precond_rebuild_count=precond_rebuild_count,
            converged=converged,
        )

    _ = pool
    _ = params_host_buf
    _ = mean_host_buf
    _ = mean_device_buf
    _ = y_centered_device
    _ = task_cov
    _ = kx_alpha
    _ = kx_rf_batch
    _ = dot_matrix_device
    _ = dot_matrix_host
    _ = dot_result_device
    _ = dot_result_host
    _ = G_B
    _ = best_sol_host
    _ = best_B_host
    _ = precond_B_host
    _ = precond_noise_host
    _ = kx_precond_holder

    return MultiOutputJITResult(
        final_params^, num_kernel_params, final_os,
        final_noise^, B_flat^, final_mean^, alpha_blocked^,
        nll_history^, cg_iterations_history^, iter_times_ns^,
        best_nll_seen, actual_iters, converged, T, R, n,
        precond_rebuild_count,
    )
