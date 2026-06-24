"""Mixed (continuous x categorical) GP training and prediction.

JIT path implementation following the JIT Path Isolation Policy:
- New code only; no modifications to AOT traits or functions.
- MixedJITGradientAdapter implements JITGradientProvider by routing
  forward/gradient matvecs through the mixed GPU kernels.
- Cat params updated via split Adam outside BBMM using post-BBMM probes.

Architecture:
    forward_matvec     -> provider.mixed_forward_matvec (GPU kernel)
    fused_gradient     -> provider.mixed_fused_gradient_matvec (cont grads)
    cat grad (inline)  -> mixed_forward_matvec with gradient correlations,
                         dot(probe_solutions_s, dK_k @ Z_s) computed on host

Cat gradient formula (NLL):
    dNLL/dtheta_k = -0.5 * alpha^T dK_k alpha + (0.5/S) sum_s (K^-1 Z_s)^T dK_k Z_s
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt, ceildiv
from time import perf_counter_ns
from python import PythonObject
from gpu import block_dim, block_idx, thread_idx

from kernels.gradient_provider import GradientProvider, ForwardProvider
from kernels.combined_inv_quad_logdet import (
    bbmm_with_precond,
    batched_cg_unified,
    CGBufferPool,
)
from kernels.pivoted_cholesky import build_pivoted_cholesky_precond_unified
from kernels.training_types import TrainingResultGeneric, AdamStateGeneric
from kernels.training_utils import (
    adam_update_state_inplace,
    adam_update_state_inplace_custom,
    adam_update_params,
    compute_cosine_lr,
    clip_gradient,
    pow_float32,
    softplus_derivative,
)
from kernels.utils import softplus, inv_softplus
from kernels.cg_solver import kernel_subtract_scalar, kernel_sum_reduce
from kernels.constants import float_dtype
from kernels.categorical_state import CategoricalCorrelationState
from kernels.jit.jit_training import JITGradientProvider
from kernels.jit.erased_provider import ErasedJITProvider
from kernels.jit.jit_categorical_params import (
    cat_chain_derivative_for_param,
    write_constrained_cat_params,
)
from kernels.jit.jit_prediction import (
    PredictionResultJIT,
    choose_exact_prediction_block_cols,
    compute_lanczos_inv_root_jit,
    kernel_compute_mean_from_cross,
    kernel_love_variance,
    kernel_exact_variance,
    solve_single_rhs_deterministic_host_jit,
    PREDICT_MEAN_ONLY,
    PREDICT_LOVE,
    PREDICT_EXACT,
)
from kernels.jit.jit_progress import emit_progress_event, progress_interval_should_emit


# =============================================================================
# GPU helper: subtract scalar (private to this module)
# =============================================================================

fn _mixed_sub_scalar[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scalar: Float32,
    n: Int,
) -> None:
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if i >= n:
        return
    out_ptr[i] = in_ptr[i] - scalar


fn _mixed_fill_scalar[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    val: Float32,
    n: Int,
) -> None:
    """Fill out_ptr[0:n] with val. Used for constant test kernel diagonal."""
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if i >= n:
        return
    out_ptr[i] = val


fn _copy_cat_test_block_variable_major[BLOCK: Int](
    out_ptr: UnsafePointer[Int32, MutAnyOrigin],
    in_ptr: UnsafePointer[Int32, MutAnyOrigin],
    num_cat_vars: Int,
    full_n_test: Int,
    block_start: Int,
    block_cols: Int,
) -> None:
    var idx = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    var total = num_cat_vars * block_cols
    if idx >= total:
        return
    var var_idx = idx // block_cols
    var local_col = idx - var_idx * block_cols
    out_ptr[idx] = in_ptr[var_idx * full_n_test + block_start + local_col]


fn build_mixed_cross_covariance_host_jit(
    provider: ErasedJITProvider,
    cat_state: CategoricalCorrelationState,
    ctx: DeviceContext,
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cat_test_device_ptr: UnsafePointer[Int32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
) raises -> HostBuffer[float_dtype]:
    """Build `K(X_train, X_test)` on host without allocating an `n_train x n_train` identity.

    The mixed prediction fallback still uses deterministic host CG for exact
    variances, so the cross-covariance only needs to be retained on host. Build
    it from chunked canonical basis blocks to keep the extra workspace
    `O(n_train * chunk_cols + n_test * chunk_cols)` instead of `O(n_train^2)`.
    """
    var max_basis_cols = 16
    var basis_host = ctx.enqueue_create_host_buffer[float_dtype](
        max(n_train * max_basis_cols, 1)
    )
    var basis_device = ctx.enqueue_create_buffer[float_dtype](
        max(n_train * max_basis_cols, 1)
    )
    var cross_chunk_device = ctx.enqueue_create_buffer[float_dtype](
        max(n_test * max_basis_cols, 1)
    )
    var cross_chunk_host = ctx.enqueue_create_host_buffer[float_dtype](
        max(n_test * max_basis_cols, 1)
    )
    var K_cross_host = ctx.enqueue_create_host_buffer[float_dtype](
        max(n_train * n_test, 1)
    )

    var train_start = 0
    while train_start < n_train:
        var remaining = n_train - train_start
        var chunk_cols = 1
        if remaining >= 16:
            chunk_cols = 16
        elif remaining >= 11:
            chunk_cols = 11
        elif remaining >= 6:
            chunk_cols = 6

        var used = n_train * chunk_cols
        for i in range(used):
            basis_host.unsafe_ptr()[i] = Float32(0.0)
        for local_col in range(chunk_cols):
            basis_host.unsafe_ptr()[local_col * n_train + train_start + local_col] = Float32(1.0)

        ctx.enqueue_copy(dst_buf=basis_device, src_buf=basis_host)
        provider.mixed_cross_matvec(
            cross_chunk_device.unsafe_ptr(),
            x_test_device_ptr,
            basis_device.unsafe_ptr(),
            cat_test_device_ptr,
            cat_state.get_c_device_ptr(),
            cat_state.get_corr_flat_device_ptr(),
            cat_state.get_offsets_device_ptr(),
            cat_state.get_levels_device_ptr(),
            cat_state.num_cat_vars,
            n_test,
            chunk_cols,
        )
        ctx.enqueue_copy(dst_buf=cross_chunk_host, src_buf=cross_chunk_device)
        ctx.synchronize()

        for local_col in range(chunk_cols):
            var train_idx = train_start + local_col
            for test_idx in range(n_test):
                K_cross_host.unsafe_ptr()[test_idx * n_train + train_idx] = cross_chunk_host.unsafe_ptr()[local_col * n_test + test_idx]

        train_start += chunk_cols

    _ = basis_host
    _ = basis_device
    _ = cross_chunk_device
    _ = cross_chunk_host
    return K_cross_host^


# =============================================================================
# MixedTrainingResult
# =============================================================================

struct MixedTrainingResult(Movable):
    """Training result for mixed continuous x categorical GP."""

    var final_params: List[Float32]   # Trained continuous kernel params (constrained)
    var noise: Float32
    var mean: Float32
    var final_nll: Float32
    var iterations: Int
    var converged: Bool
    var num_cont_params: Int
    var cat_params: List[Float32]     # Trained categorical params (unconstrained)
    var total_cat_params: Int
    var iter_times_ns: List[Int]
    var nll_history: List[Float32]  # NLL value at each iteration
    var cg_iterations_history: List[Int]  # Realized CG iterations per optimizer step

    fn __init__(
        out self,
        var final_params: List[Float32],
        noise: Float32,
        mean: Float32,
        final_nll: Float32,
        iterations: Int,
        converged: Bool,
        num_cont_params: Int,
        var cat_params: List[Float32],
        total_cat_params: Int,
        var iter_times_ns: List[Int],
        var nll_history: List[Float32] = List[Float32](),
        var cg_iterations_history: List[Int] = List[Int](),
    ):
        self.final_params = final_params^
        self.noise = noise
        self.mean = mean
        self.final_nll = final_nll
        self.iterations = iterations
        self.converged = converged
        self.num_cont_params = num_cont_params
        self.cat_params = cat_params^
        self.total_cat_params = total_cat_params
        self.iter_times_ns = iter_times_ns^
        self.nll_history = nll_history^
        self.cg_iterations_history = cg_iterations_history^

    fn __moveinit__(out self, owned other: Self):
        self.final_params = other.final_params^
        self.noise = other.noise
        self.mean = other.mean
        self.final_nll = other.final_nll
        self.iterations = other.iterations
        self.converged = other.converged
        self.num_cont_params = other.num_cont_params
        self.cat_params = other.cat_params^
        self.total_cat_params = other.total_cat_params
        self.iter_times_ns = other.iter_times_ns^
        self.nll_history = other.nll_history^
        self.cg_iterations_history = other.cg_iterations_history^


# =============================================================================
# MixedJITGradientAdapter
# =============================================================================

struct MixedJITGradientAdapter(JITGradientProvider, Movable):
    """JITGradientProvider for mixed continuous x categorical kernels.

    Owns ErasedJITProvider (continuous fn ptrs) + CategoricalCorrelationState
    (cat indices + correlation matrices on GPU).

    num_gradient_params() returns num_cont_params only. Categorical params are
    updated separately outside BBMM via inline cat gradient computation in
    train_mixed_jit.

    forward_matvec   -> mixed_forward_matvec  (K_cont x prod R_cv + noise*I) @ v
    fused_gradient   -> mixed_fused_gradient  (dK_cont/dtheta_k x prod R_cv) @ v
    extract_diagonal -> provider.extract_diagonal (K_cont diag + noise; valid
                        for stationary kernels since R_cv(c_i, c_i) = 1)
    """

    var provider: ErasedJITProvider
    var cat_state: CategoricalCorrelationState
    var num_cont_params: Int

    fn __init__(
        out self,
        owned provider: ErasedJITProvider,
        owned cat_state: CategoricalCorrelationState,
        num_cont_params: Int,
    ):
        self.provider = provider^
        self.cat_state = cat_state^
        self.num_cont_params = num_cont_params

    fn __moveinit__(out self, owned other: Self):
        self.provider = other.provider^
        self.cat_state = other.cat_state^
        self.num_cont_params = other.num_cont_params

    # --- ForwardProvider ---

    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        self.provider.mixed_forward_matvec(
            out_ptr, v_ptr,
            self.cat_state.get_c_device_ptr(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            self.cat_state.num_cat_vars,
            num_cols,
            self.provider.get_noise(),
        )

    fn get_n(self) -> Int:
        return self.provider.get_n()

    fn get_ctx(self) -> DeviceContext:
        return self.provider.get_ctx()

    fn get_noise(self) -> Float32:
        return self.provider.get_noise()

    fn get_diagonal_value(self) -> Float32:
        return self.provider.get_diagonal_value()

    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        # R_cv(c_i, c_i) = 1 for all correlation matrices, so
        # K_mixed(x_i, x_i) = K_cont(x_i, x_i). Delegate to cont provider.
        self.provider.extract_diagonal(diag_ptr)

    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.provider.get_x_ptr()

    # --- GradientProvider ---

    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        pass  # Not used: supports_fused_gradient() = True

    fn num_gradient_params(self) -> Int:
        return self.num_cont_params

    fn supports_fused_gradient(self) -> Bool:
        return True

    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        self.provider.mixed_fused_gradient_matvec(
            out_ptr, v_ptr,
            self.cat_state.get_c_device_ptr(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            self.cat_state.num_cat_vars,
            num_cols,
        )

    fn supports_fused_ls_os(self) -> Bool:
        return False

    fn supports_fused_3param(self) -> Bool:
        return False

    fn fused_ls_os_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        pass

    fn fused_3param_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        pass

    # --- JITGradientProvider ---

    fn update_params(
        mut self,
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        self.provider.update_params(params_host_ptr)

    fn update_noise(mut self, noise: Float32):
        self.provider.update_noise(noise)

    fn get_noise_mode(self) -> Int:
        return self.provider.get_noise_mode()

    fn get_noise_vector_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.provider.get_noise_vector_ptr()

    fn refresh_materialization(mut self) raises:
        self.provider.mixed_materialize(
            self.cat_state.get_c_device_ptr(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            self.cat_state.num_cat_vars,
        )


# =============================================================================
# train_mixed_jit
# =============================================================================

fn train_mixed_jit(
    owned provider: ErasedJITProvider,
    owned cat_state: CategoricalCorrelationState,
    ctx: DeviceContext,
    y_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cont_params: Int,
    initial_cont_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    initial_noise: Float32,
    initial_cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    total_cat_params: Int,
    max_iterations: Int = 100,
    learning_rate: Float32 = 0.01,
    num_probes: Int = 10,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    precond_method: Int = 0,
    enable_early_stopping: Bool = False,
    early_stop_patience: Int = 10,
    early_stop_tol: Float32 = 1e-4,
    verbose: Bool = False,
    init_mean: Float32 = 0.0,
    max_tridiag_iter: Int = 30,
    precond_rebuild_threshold: Float32 = 0.5,
    use_cosine_lr: Bool = True,
    use_preconditioner: Bool = True,
    use_materialized: Bool = False,
    progress_callback: PythonObject = None,
    progress_interval: Int = 1,
    progress_enabled: Bool = False,
) raises -> MixedTrainingResult:
    """Train a mixed continuous x categorical GP with split Adam.

    Continuous params (+ noise + mean) are optimised via BBMM gradients.
    Categorical params are optimised via post-BBMM gradient computation using
    pool.probes_device (Z) and result.probe_solutions (K^{-1}Z).

    Cat gradient:
        dNLL/dtheta_k = -0.5 * alpha^T dK_k alpha
                       + (0.5/S) * sum_s (K^{-1}Z_s)^T dK_k Z_s

    where dK_k uses gradient correlation matrices from
    CategoricalCorrelationState.compute_all_gradient_correlations().
    """
    var adapter = MixedJITGradientAdapter(provider^, cat_state^, num_cont_params)

    # Copy y to device
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    for i in range(n):
        y_host[i] = y_host_ptr[i]
    var y_device = ctx.enqueue_create_buffer[float_dtype](n)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.synchronize()

    var raw_mean = init_mean
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](n)

    # Continuous raw params (inv_softplus, noise appended)
    var raw_cont = List[Float32]()
    for p in range(num_cont_params):
        raw_cont.append(inv_softplus(initial_cont_params_ptr[p]))
    raw_cont.append(inv_softplus(initial_noise))

    # Categorical raw params (unconstrained; no softplus)
    var raw_cat = List[Float32]()
    for k in range(total_cat_params):
        raw_cat.append(initial_cat_params_ptr[k])

    # Adam states
    var adam_cont = AdamStateGeneric(num_cont_params)
    var adam_cat = AdamStateGeneric(total_cat_params)

    # Buffer pool and params host buffers
    var pool = CGBufferPool(ctx, n, 1 + num_probes, num_probes, max_tridiag_iter)
    var cont_params_host = ctx.enqueue_create_host_buffer[float_dtype](num_cont_params)
    for p in range(num_cont_params):
        cont_params_host[p] = initial_cont_params_ptr[p]

    var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    if total_cat_params > 0:
        write_constrained_cat_params(adapter.cat_state, raw_cat, cat_params_host.unsafe_ptr())
        adapter.cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())
    if use_materialized:
        adapter.refresh_materialization()

    # Best-param tracking
    var best_nll = Float32(1e30)
    var no_improve_count = 0
    var actual_iterations = 0
    var converged = False
    var last_nll = Float32(0.0)
    var lr = learning_rate

    var last_rebuild_cont = List[Float32]()
    for p in range(num_cont_params):
        last_rebuild_cont.append(initial_cont_params_ptr[p])
    var last_rebuild_noise = initial_noise

    var precond_error_tol = Float32(1e-3)
    if precond_method == 0:
        precond_error_tol = Float32(0.0)

    var best_raw_cont = List[Float32]()
    for p in range(num_cont_params + 1):
        best_raw_cont.append(raw_cont[p])
    var best_raw_cat = List[Float32]()
    for k in range(total_cat_params):
        best_raw_cat.append(raw_cat[k])
    var best_raw_mean = raw_mean
    var best_nll_seen = Float32(1e30)

    # Initial preconditioner
    var num_cols = 1 + num_probes
    pool.ensure_capacity(ctx, n, num_cols, num_probes, max_tridiag_iter,
                         precond_rank, num_kernel_params=num_cont_params)
    var precond = build_pivoted_cholesky_precond_unified(
        adapter,
        rank=precond_rank,
        error_tol=precond_error_tol,
        max_num_cols=num_cols,
        precond_method=precond_method,
    )

    # Cat gradient buffers (allocated once, reused every iteration)
    var corr_stride = adapter.cat_state.total_corr_size
    var grad_buf_size = max(total_cat_params * corr_stride, 1)
    var all_grad_corr_host = HostBuffer[float_dtype](ctx, grad_buf_size)
    var all_grad_corr_device = ctx.enqueue_create_buffer[float_dtype](grad_buf_size)
    var out_alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
    var out_probes_device = ctx.enqueue_create_buffer[float_dtype](max(n * num_probes, 1))
    var out_alpha_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var out_probes_host = ctx.enqueue_create_host_buffer[float_dtype](max(n * num_probes, 1))
    var alpha_host_buf = ctx.enqueue_create_host_buffer[float_dtype](n)
    var probe_sol_host_buf = ctx.enqueue_create_host_buffer[float_dtype](max(n * num_probes, 1))
    var probes_host_buf = ctx.enqueue_create_host_buffer[float_dtype](max(n * num_probes, 1))

    var iter_times_ns = List[Int]()
    var nll_history = List[Float32]()
    var cg_iterations_history = List[Int]()

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "single_output",
            "materialized" if use_materialized else "matrix_free",
            "start",
            0,
            max_iterations,
            precond_rank=precond.rank,
            precond_rebuild_count=0,
        )

    # =========================================================================
    # Training loop
    # =========================================================================
    for iteration in range(max_iterations):
        var iter_start = perf_counter_ns()
        actual_iterations = iteration + 1

        # Constrained cont params + noise
        var cont_c = List[Float32]()
        for p in range(num_cont_params):
            cont_c.append(softplus(raw_cont[p]))
        var noise = softplus(raw_cont[num_cont_params])

        for p in range(num_cont_params):
            cont_params_host[p] = cont_c[p]
        adapter.update_params(cont_params_host.unsafe_ptr())
        adapter.update_noise(noise)

        # Update correlation matrices from current cat params
        if total_cat_params > 0:
            write_constrained_cat_params(adapter.cat_state, raw_cat, cat_params_host.unsafe_ptr())
            adapter.cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())
        if use_materialized:
            adapter.refresh_materialization()

        # Adaptive preconditioner rebuild
        if iteration > 0:
            var max_rel = Float32(0.0)
            for p in range(num_cont_params):
                var d = abs(cont_c[p] - last_rebuild_cont[p])
                var rel = d / max(abs(last_rebuild_cont[p]), Float32(1e-8))
                if rel > max_rel:
                    max_rel = rel
            var rel_noise = abs(noise - last_rebuild_noise) / max(abs(last_rebuild_noise), Float32(1e-8))
            if rel_noise > max_rel:
                max_rel = rel_noise
            if max_rel > precond_rebuild_threshold:
                precond = build_pivoted_cholesky_precond_unified(
                    adapter,
                    rank=precond_rank,
                    error_tol=precond_error_tol,
                    max_num_cols=num_cols,
                    precond_method=precond_method,
                )
                for p in range(num_cont_params):
                    last_rebuild_cont[p] = cont_c[p]
                last_rebuild_noise = noise

        # Center y
        alias BLK = 256
        ctx.enqueue_function[_mixed_sub_scalar[BLK]](
            y_centered_device.unsafe_ptr(), y_device.unsafe_ptr(), raw_mean, n,
            grid_dim=ceildiv(n, BLK), block_dim=BLK,
        )

        # BBMM: NLL + cont gradients
        var result = bbmm_with_precond(
            adapter, precond, y_centered_device.unsafe_ptr(), n, pool,
            num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
            iteration=iteration,
            recycle_alpha=iteration > 0,
            use_preconditioner=use_preconditioner,
        )

        last_nll = result.nll
        cg_iterations_history.append(result.num_iterations)

        if last_nll != last_nll:  # NaN
            lr = lr * Float32(0.5)
            if verbose:
                print("  NaN NLL at iter", iteration, "- halving LR")
            if progress_enabled:
                emit_progress_event(
                    progress_callback,
                    "train",
                    "single_output",
                    "materialized" if use_materialized else "matrix_free",
                    "nan",
                    actual_iterations,
                    max_iterations,
                    nll=last_nll,
                    best_nll=best_nll_seen,
                    cg_iter=result.num_iterations,
                    iter_time_ns=Int(perf_counter_ns() - iter_start),
                    noise=noise,
                    mean=raw_mean,
                    precond_rank=precond.rank,
                )
            continue

        if last_nll < best_nll_seen:
            best_nll_seen = last_nll
            for p in range(num_cont_params + 1):
                best_raw_cont[p] = raw_cont[p]
            for k in range(total_cat_params):
                best_raw_cat[k] = raw_cat[k]
            best_raw_mean = raw_mean

        var effective_lr = lr
        if use_cosine_lr:
            effective_lr = compute_cosine_lr(learning_rate, iteration, max_iterations)

        # ------------------------------------------------------------------
        # Cont + noise Adam update
        # ------------------------------------------------------------------
        var cont_grads = List[Float32]()
        for p in range(num_cont_params):
            cont_grads.append(result.gradients[p])
        cont_grads.append(result.gradients[num_cont_params])

        adam_cont = adam_update_state_inplace(adam_cont^, cont_grads, raw_cont, effective_lr)
        raw_cont = adam_update_params(adam_cont, cont_grads, raw_cont, effective_lr)

        # ------------------------------------------------------------------
        # Constant mean update
        # ------------------------------------------------------------------
        var mean_sum_dev = ctx.enqueue_create_buffer[DType.float32](1)
        var mean_sum_host = ctx.enqueue_create_host_buffer[DType.float32](1)
        ctx.enqueue_function[kernel_sum_reduce](
            mean_sum_dev.unsafe_ptr(), result.solution.unsafe_ptr(), n,
            grid_dim=1, block_dim=256, shared_mem_bytes=8 * 4,
        )
        ctx.enqueue_copy(dst_buf=mean_sum_host, src_buf=mean_sum_dev)
        ctx.synchronize()
        var mean_grad = clip_gradient(-mean_sum_host.unsafe_ptr()[0] / Float32(n))

        var b1 = Float32(0.9)
        var b2 = Float32(0.999)
        var eps = Float32(1e-8)
        var tm = Float32(adam_cont.t)
        adam_cont.m_mean = b1 * adam_cont.m_mean + (Float32(1.0) - b1) * mean_grad
        adam_cont.v_mean = b2 * adam_cont.v_mean + (Float32(1.0) - b2) * mean_grad * mean_grad
        var mh = adam_cont.m_mean / (Float32(1.0) - pow_float32(b1, Int(tm)))
        var vh = adam_cont.v_mean / (Float32(1.0) - pow_float32(b2, Int(tm)))
        raw_mean -= effective_lr * mh / (sqrt(vh) + eps)
        if raw_mean != raw_mean or raw_mean > Float32(1000.0) or raw_mean < Float32(-1000.0):
            raw_mean = init_mean

        # ------------------------------------------------------------------
        # Categorical Adam update (post-BBMM using probe vectors)
        # ------------------------------------------------------------------
        if total_cat_params > 0:
            # Compute dR_k for all cat params at once (CPU, then upload)
            adapter.cat_state.compute_all_gradient_correlations(
                all_grad_corr_host, cat_params_host.unsafe_ptr()
            )
            ctx.enqueue_copy(dst_buf=all_grad_corr_device, src_buf=all_grad_corr_host)
            ctx.synchronize()

            # Copy alpha and probes to host (once per iteration)
            ctx.enqueue_copy(dst_buf=alpha_host_buf, src_buf=result.solution)
            ctx.enqueue_copy(dst_buf=probe_sol_host_buf, src_buf=result.probe_solutions)
            ctx.enqueue_copy(dst_buf=probes_host_buf, src_buf=pool.probes_device)
            ctx.synchronize()

            var inv_S = Float32(1.0) / Float32(num_probes)
            var cat_grads = List[Float32]()

            for k in range(total_cat_params):
                var dR_k = all_grad_corr_device.unsafe_ptr() + k * corr_stride

                # dK_k @ alpha  (ncols=1, noise=0 for gradient)
                adapter.provider.mixed_forward_matvec(
                    out_alpha_device.unsafe_ptr(),
                    result.solution.unsafe_ptr(),
                    adapter.cat_state.get_c_device_ptr(),
                    dR_k,
                    adapter.cat_state.get_offsets_device_ptr(),
                    adapter.cat_state.get_levels_device_ptr(),
                    adapter.cat_state.num_cat_vars,
                    1,
                    Float32(0.0),
                )
                ctx.enqueue_copy(dst_buf=out_alpha_host, src_buf=out_alpha_device)
                ctx.synchronize()

                var dot_alpha = Float32(0.0)
                for i in range(n):
                    dot_alpha += alpha_host_buf[i] * out_alpha_host[i]

                # dK_k @ Z  (ncols=num_probes, noise=0)
                adapter.provider.mixed_forward_matvec(
                    out_probes_device.unsafe_ptr(),
                    pool.probes_device.unsafe_ptr(),
                    adapter.cat_state.get_c_device_ptr(),
                    dR_k,
                    adapter.cat_state.get_offsets_device_ptr(),
                    adapter.cat_state.get_levels_device_ptr(),
                    adapter.cat_state.num_cat_vars,
                    num_probes,
                    Float32(0.0),
                )
                ctx.enqueue_copy(dst_buf=out_probes_host, src_buf=out_probes_device)
                ctx.synchronize()

                # sum_s dot(K^{-1}Z_s, dK_k @ Z_s)
                var trace_term = Float32(0.0)
                for s in range(num_probes):
                    var ds = Float32(0.0)
                    for i in range(n):
                        ds += probe_sol_host_buf[s * n + i] * out_probes_host[s * n + i]
                    trace_term += ds

                # dNLL/dtheta_k = -0.5*dot_alpha + (0.5/S)*trace_term
                cat_grads.append(clip_gradient(
                    Float32(-0.5) * dot_alpha + Float32(0.5) * inv_S * trace_term
                ))

            # Cat Adam updates raw params using the categorical transform chain rule.
            var chain_derivs = List[Float32]()
            for k in range(total_cat_params):
                chain_derivs.append(cat_chain_derivative_for_param(adapter.cat_state, raw_cat, k))
            adam_cat = adam_update_state_inplace_custom(
                adam_cat^, cat_grads, raw_cat, chain_derivs, effective_lr
            )
            raw_cat = adam_update_params(adam_cat, cat_grads, raw_cat, effective_lr)

        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print("Iter", iteration, ": NLL =", last_nll, ", noise =", noise)

        var iter_time_ns = Int(perf_counter_ns() - iter_start)
        iter_times_ns.append(iter_time_ns)
        nll_history.append(last_nll)
        if progress_enabled and progress_interval_should_emit(iteration, max_iterations, progress_interval):
            emit_progress_event(
                progress_callback,
                "train",
                "single_output",
                "materialized" if use_materialized else "matrix_free",
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
            )

        if enable_early_stopping:
            if last_nll < best_nll - early_stop_tol:
                best_nll = last_nll
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    converged = True
                    if verbose:
                        print("Early stopping after", actual_iterations, "iters")
                    if progress_enabled:
                        emit_progress_event(
                            progress_callback,
                            "train",
                            "single_output",
                            "materialized" if use_materialized else "matrix_free",
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
                            converged=converged,
                        )
                    break

    _ = precond

    # Restore best-seen params
    for p in range(num_cont_params + 1):
        raw_cont[p] = best_raw_cont[p]
    for k in range(total_cat_params):
        raw_cat[k] = best_raw_cat[k]
    raw_mean = best_raw_mean

    var final_cont = List[Float32]()
    for p in range(num_cont_params):
        final_cont.append(softplus(raw_cont[p]))
    var final_noise = softplus(raw_cont[num_cont_params])

    var final_cat = List[Float32]()
    for k in range(total_cat_params):
        final_cat.append(raw_cat[k])

    if verbose:
        print("Mixed GP done: NLL =", best_nll_seen, ", noise =", final_noise)

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "single_output",
            "materialized" if use_materialized else "matrix_free",
            "complete",
            actual_iterations,
            max_iterations,
            nll=best_nll_seen,
            best_nll=best_nll_seen,
            noise=final_noise,
            mean=raw_mean,
            precond_rank=precond.rank,
            converged=converged,
        )

    _ = y_device
    _ = y_centered_device
    _ = all_grad_corr_device
    _ = out_alpha_device
    _ = out_probes_device
    _ = cont_params_host
    _ = cat_params_host

    return MixedTrainingResult(
        final_cont^, final_noise, raw_mean, best_nll_seen,
        actual_iterations, converged, num_cont_params,
        final_cat^, total_cat_params, iter_times_ns^, nll_history^,
        cg_iterations_history^,
    )


# =============================================================================
# predict_mixed_jit
# =============================================================================

fn predict_mixed_jit(
    owned provider: ErasedJITProvider,
    owned cat_state: CategoricalCorrelationState,
    ctx: DeviceContext,
    y_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cat_test_device_ptr: UnsafePointer[Int32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    x_test_dim: Int,
    noise: Float32,
    mean: Float32,
    variance_method: Int,
    max_cg_iter: Int = 100,
    cg_tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    lanczos_rank: Int = 20,
    use_preconditioner: Bool = True,
    use_materialized: Bool = False,
) raises -> PredictionResultJIT:
    """Predict with a trained mixed GP.

    Args:
        provider: ErasedJITProvider with final trained cont params set.
        cat_state: CategoricalCorrelationState with correlation matrices
                   already updated from trained cat params.
        y_device_ptr: Training targets [n_train] on GPU.
        x_test_device_ptr: Test continuous inputs [n_test x dim] on GPU.
        cat_test_device_ptr: Test cat indices [num_cat_vars x n_test] on GPU
                             (variable-major: layout [cv * n_test + i]).
        variance_method: PREDICT_MEAN_ONLY=0, PREDICT_LOVE=1, PREDICT_EXACT=2.
    """
    # adapter with num_cont_params=0: only need forward matvec + extract_diagonal
    var adapter = MixedJITGradientAdapter(provider^, cat_state^, 0)
    if use_materialized:
        adapter.refresh_materialization()

    alias BLK = 256

    # Center y
    var y_centered = ctx.enqueue_create_buffer[float_dtype](n_train)
    ctx.enqueue_function[_mixed_sub_scalar[BLK]](
        y_centered.unsafe_ptr(), y_device_ptr, mean, n_train,
        grid_dim=ceildiv(n_train, BLK), block_dim=BLK,
    )
    ctx.synchronize()

    # Build alpha = (K_mixed + sigma*I)^{-1} @ (y - mean)
    # Mixed prediction must be deterministic across repeated calls and save/load.
    var y_centered_host = ctx.enqueue_create_host_buffer[float_dtype](n_train)
    ctx.enqueue_copy(dst_buf=y_centered_host, src_buf=y_centered)
    ctx.synchronize()
    var alpha_h = solve_single_rhs_deterministic_host_jit(
        adapter,
        ctx,
        y_centered_host.unsafe_ptr(),
        n_train,
        max_cg_iter,
        cg_tol,
    )
    var alpha_device = ctx.enqueue_create_buffer[float_dtype](n_train)
    ctx.enqueue_copy(dst_buf=alpha_device, src_buf=alpha_h)
    ctx.synchronize()
    _ = y_centered_host

    # Mean: K_mixed(X_test, X_train) @ alpha + mean
    var cross_alpha = ctx.enqueue_create_buffer[float_dtype](n_test)
    ctx.synchronize()
    adapter.provider.mixed_cross_matvec(
        cross_alpha.unsafe_ptr(),
        x_test_device_ptr,
        alpha_device.unsafe_ptr(),
        cat_test_device_ptr,
        adapter.cat_state.get_c_device_ptr(),
        adapter.cat_state.get_corr_flat_device_ptr(),
        adapter.cat_state.get_offsets_device_ptr(),
        adapter.cat_state.get_levels_device_ptr(),
        adapter.cat_state.num_cat_vars,
        n_test,
        1,
    )
    var mean_device = ctx.enqueue_create_buffer[float_dtype](n_test)
    ctx.enqueue_function[kernel_compute_mean_from_cross[BLK]](
        mean_device.unsafe_ptr(), cross_alpha.unsafe_ptr(), mean, n_test,
        grid_dim=ceildiv(n_test, BLK), block_dim=BLK,
    )
    ctx.synchronize()
    _ = cross_alpha

    # Variance
    var has_variance = variance_method != PREDICT_MEAN_ONLY
    var var_device = ctx.enqueue_create_buffer[float_dtype](n_test)

    if variance_method == PREDICT_LOVE:
        var rank = lanczos_rank
        var inv_root_host = compute_lanczos_inv_root_jit(adapter, rank)
        var inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * rank)
        ctx.enqueue_copy(dst_buf=inv_root_device, src_buf=inv_root_host)
        ctx.synchronize()

        var V = ctx.enqueue_create_buffer[float_dtype](rank * n_test)
        ctx.synchronize()
        adapter.provider.mixed_cross_matvec(
            V.unsafe_ptr(),
            x_test_device_ptr,
            inv_root_device.unsafe_ptr(),
            cat_test_device_ptr,
            adapter.cat_state.get_c_device_ptr(),
            adapter.cat_state.get_corr_flat_device_ptr(),
            adapter.cat_state.get_offsets_device_ptr(),
            adapter.cat_state.get_levels_device_ptr(),
            adapter.cat_state.num_cat_vars,
            n_test,
            rank,
        )

        # Diagonal of test kernel: K_cont(x*,x*) * prod R_cv(c*,c*) = K_cont(x*,x*)*1
        # For stationary kernels K_cont(x*,x*) is constant = get_diagonal_value().
        var diag_test = ctx.enqueue_create_buffer[float_dtype](n_test)
        ctx.synchronize()
        ctx.enqueue_function[_mixed_fill_scalar[BLK]](
            diag_test.unsafe_ptr(), adapter.provider.get_diagonal_value(), n_test,
            grid_dim=ceildiv(n_test, BLK), block_dim=BLK,
        )

        ctx.enqueue_function[kernel_love_variance[BLK]](
            var_device.unsafe_ptr(), V.unsafe_ptr(), diag_test.unsafe_ptr(),
            n_test, rank,
            grid_dim=ceildiv(n_test, BLK), block_dim=BLK,
        )
        ctx.synchronize()
        _ = inv_root_host
        _ = inv_root_device
        _ = V
        _ = diag_test

    elif variance_method == PREDICT_EXACT:
        var var_exact_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
        var diag_value = adapter.provider.get_diagonal_value()
        var block_cols = choose_exact_prediction_block_cols(n_test)
        var cat_block_device = ctx.enqueue_create_buffer[DType.int32](
            max(adapter.cat_state.num_cat_vars * block_cols, 1)
        )
        var test_start = 0
        while test_start < n_test:
            var active_cols = block_cols
            if n_test - test_start < active_cols:
                active_cols = n_test - test_start

            ctx.enqueue_function[_copy_cat_test_block_variable_major[BLK]](
                cat_block_device.unsafe_ptr(),
                cat_test_device_ptr,
                adapter.cat_state.num_cat_vars,
                n_test,
                test_start,
                active_cols,
                grid_dim=ceildiv(adapter.cat_state.num_cat_vars * active_cols, BLK),
                block_dim=BLK,
            )
            ctx.synchronize()

            var Kch = build_mixed_cross_covariance_host_jit(
                adapter.provider,
                adapter.cat_state,
                ctx,
                x_test_device_ptr.offset(test_start * x_test_dim),
                cat_block_device.unsafe_ptr(),
                n_train,
                active_cols,
            )
            for j in range(active_cols):
                var solve_host = solve_single_rhs_deterministic_host_jit(
                    adapter,
                    ctx,
                    Kch.unsafe_ptr().offset(j * n_train),
                    n_train,
                    max_cg_iter,
                    cg_tol,
                )
                var dot = Float32(0.0)
                for i in range(n_train):
                    dot += Kch.unsafe_ptr()[j * n_train + i] * solve_host.unsafe_ptr()[i]
                var variance = diag_value - dot
                if variance < Float32(1e-10):
                    variance = Float32(1e-10)
                var_exact_host.unsafe_ptr()[test_start + j] = variance
                _ = solve_host
            _ = Kch
            test_start += active_cols

        ctx.enqueue_copy(dst_buf=var_device, src_buf=var_exact_host)
        ctx.synchronize()
        _ = cat_block_device
        _ = var_exact_host

    # Copy results to host
    var mean_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    var var_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    ctx.enqueue_copy(dst_buf=mean_host, src_buf=mean_device)
    if has_variance:
        ctx.enqueue_copy(dst_buf=var_host, src_buf=var_device)
    ctx.synchronize()

    _ = y_centered
    _ = alpha_device
    _ = alpha_h
    _ = mean_device
    _ = var_device

    return PredictionResultJIT(
        mean_host^,
        var_host^,
        n_test,
        has_variance,
        choose_exact_prediction_block_cols(n_test) if variance_method == PREDICT_EXACT else 0,
        2 if variance_method == PREDICT_EXACT else 0,
        0,
        False,
        0,
        0,
        0,
    )
