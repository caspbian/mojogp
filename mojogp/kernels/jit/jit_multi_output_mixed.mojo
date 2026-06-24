"""JIT mixed multi-output (Kronecker/ICM) training and prediction helpers.

Implements mixed continuous+categorical multi-output support without modifying
the AOT Kronecker provider code. The base mixed kernel is exposed through a
copyable JIT provider view that stores raw categorical-state pointers while the
owning CategoricalCorrelationState stays alive in the binding/training scope.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from math import sqrt, ceildiv, exp as math_exp
from time import perf_counter_ns
from gpu import block_idx, thread_idx
from python import PythonObject

from kernels.constants import float_dtype
from kernels.categorical_state import CategoricalCorrelationState
from kernels.jit.erased_provider import ErasedJITProvider
from kernels.jit.jit_training import JITGradientProvider
from kernels.jit.jit_training import softplus, inv_softplus, pow_float32, compute_cosine_lr, clip_gradient
from kernels.jit.jit_categorical_params import (
    cat_chain_derivative_for_param,
    write_constrained_cat_params,
)
from kernels.jit.jit_prediction import compute_lanczos_inv_root_jit, choose_exact_prediction_block_cols, kernel_love_variance, kernel_exact_variance, solve_single_rhs_deterministic_host_jit
from kernels.jit.jit_multi_output import kernel_subtract_task_means_blocked
from kernels.task_covariance import TaskCovariance
from kernels.kronecker_direct_provider import KroneckerDirectProvider
from kernels.kronecker_gpu_kernels import kernel_kronecker_combine_batched
from kernels.combined_inv_quad_logdet import bbmm_with_precond, batched_cg_unified, CGBufferPool
from kernels.pivoted_cholesky import build_pivoted_cholesky_precond_unified
from kernels.bbmm_gpu_kernels import kernel_dot_matrix
from kernels.cg_solver import kernel_dot_batched
from kernels.jit.jit_progress import emit_progress_event, progress_interval_should_emit


fn _fill_scalar[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    val: Float32,
    n: Int,
) -> None:
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
    provider: MixedKroneckerBaseProviderView,
    ctx: DeviceContext,
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cat_test_device_ptr: UnsafePointer[Int32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
) raises -> HostBuffer[float_dtype]:
    """Build `K(X_train, X_test)` on host without allocating an `n_train x n_train` identity.

    Exact mixed variance is solved with deterministic host CG one RHS at a time,
    so keep the cross-covariance on host and synthesize it from chunked basis
    blocks rather than a full train-size identity buffer.
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


fn _copy_slice[BLOCK: Int](
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    in_ptr: UnsafePointer[Float32, MutAnyOrigin],
    offset: Int,
    n: Int,
) -> None:
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if i >= n:
        return
    out_ptr[i] = in_ptr[offset + i]


struct MixedMultiOutputJITResult(Movable):
    var final_params: List[Float32]
    var cat_params: List[Float32]
    var num_kernel_params: Int
    var outputscale: Float32
    var noise_per_task: List[Float32]
    var B_flat: List[Float32]
    var mean_per_task: List[Float32]
    var alpha_blocked: List[Float32]
    var nll_history: List[Float32]
    var cg_iterations_history: List[Int]
    var final_nll: Float32
    var iterations: Int
    var converged: Bool
    var num_tasks: Int
    var task_rank: Int
    var n: Int
    var precond_rebuild_count: Int

    fn __init__(
        out self,
        var final_params: List[Float32],
        var cat_params: List[Float32],
        num_kernel_params: Int,
        outputscale: Float32,
        var noise_per_task: List[Float32],
        var B_flat: List[Float32],
        var mean_per_task: List[Float32],
        var alpha_blocked: List[Float32],
        var nll_history: List[Float32],
        var cg_iterations_history: List[Int],
        final_nll: Float32,
        iterations: Int,
        converged: Bool,
        num_tasks: Int,
        task_rank: Int,
        n: Int,
        precond_rebuild_count: Int,
    ):
        self.final_params = final_params^
        self.cat_params = cat_params^
        self.num_kernel_params = num_kernel_params
        self.outputscale = outputscale
        self.noise_per_task = noise_per_task^
        self.B_flat = B_flat^
        self.mean_per_task = mean_per_task^
        self.alpha_blocked = alpha_blocked^
        self.nll_history = nll_history^
        self.cg_iterations_history = cg_iterations_history^
        self.final_nll = final_nll
        self.iterations = iterations
        self.converged = converged
        self.num_tasks = num_tasks
        self.task_rank = task_rank
        self.n = n
        self.precond_rebuild_count = precond_rebuild_count

    fn __moveinit__(out self, owned other: Self):
        self.final_params = other.final_params^
        self.cat_params = other.cat_params^
        self.num_kernel_params = other.num_kernel_params
        self.outputscale = other.outputscale
        self.noise_per_task = other.noise_per_task^
        self.B_flat = other.B_flat^
        self.mean_per_task = other.mean_per_task^
        self.alpha_blocked = other.alpha_blocked^
        self.nll_history = other.nll_history^
        self.cg_iterations_history = other.cg_iterations_history^
        self.final_nll = other.final_nll
        self.iterations = other.iterations
        self.converged = other.converged
        self.num_tasks = other.num_tasks
        self.task_rank = other.task_rank
        self.n = other.n
        self.precond_rebuild_count = other.precond_rebuild_count


struct MixedKroneckerBaseProviderView(JITGradientProvider, Copyable, Movable):
    var provider: ErasedJITProvider
    var cat_indices_ptr: UnsafePointer[Int32, MutAnyOrigin]
    var corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var offsets_ptr: UnsafePointer[Int32, MutAnyOrigin]
    var levels_ptr: UnsafePointer[Int32, MutAnyOrigin]
    var num_cat_vars: Int

    fn __init__(
        out self,
        owned provider: ErasedJITProvider,
        cat_state: CategoricalCorrelationState,
    ):
        self.provider = provider^
        self.cat_indices_ptr = cat_state.get_c_device_ptr()
        self.corr_flat_ptr = cat_state.get_corr_flat_device_ptr()
        self.offsets_ptr = cat_state.get_offsets_device_ptr()
        self.levels_ptr = cat_state.get_levels_device_ptr()
        self.num_cat_vars = cat_state.num_cat_vars

    fn __copyinit__(out self, existing: Self):
        self.provider = existing.provider.clone()
        self.cat_indices_ptr = existing.cat_indices_ptr
        self.corr_flat_ptr = existing.corr_flat_ptr
        self.offsets_ptr = existing.offsets_ptr
        self.levels_ptr = existing.levels_ptr
        self.num_cat_vars = existing.num_cat_vars

    fn __moveinit__(out self, owned other: Self):
        self.provider = other.provider^
        self.cat_indices_ptr = other.cat_indices_ptr
        self.corr_flat_ptr = other.corr_flat_ptr
        self.offsets_ptr = other.offsets_ptr
        self.levels_ptr = other.levels_ptr
        self.num_cat_vars = other.num_cat_vars

    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        self.provider.mixed_forward_matvec(
            out_ptr,
            v_ptr,
            self.cat_indices_ptr,
            self.corr_flat_ptr,
            self.offsets_ptr,
            self.levels_ptr,
            self.num_cat_vars,
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
        self.provider.extract_diagonal(diag_ptr)

    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.provider.get_x_ptr()

    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        var ctx = self.provider.get_ctx()
        var n = self.provider.get_n()
        var total = self.provider.num_gradient_params() * n * num_cols
        var temp = ctx.enqueue_create_buffer[float_dtype](max(total, 1))
        self.provider.mixed_fused_gradient_matvec(
            temp.unsafe_ptr(),
            v_ptr,
            self.cat_indices_ptr,
            self.corr_flat_ptr,
            self.offsets_ptr,
            self.levels_ptr,
            self.num_cat_vars,
            num_cols,
        )
        alias BLOCK = 256
        ctx.enqueue_function[_copy_slice[BLOCK]](
            out_ptr,
            temp.unsafe_ptr(),
            param_index * n * num_cols,
            n * num_cols,
            grid_dim=(ceildiv(n * num_cols, BLOCK),),
            block_dim=(BLOCK,),
        )
        # `temp` is a local DeviceBuffer. It must stay alive until every kernel
        # that reads it has finished, even when the caller requested deferred sync.
        ctx.synchronize()
        _ = temp

    fn num_gradient_params(self) -> Int:
        return self.provider.num_gradient_params()

    fn supports_fused_gradient(self) -> Bool:
        return False

    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        self.provider.mixed_fused_gradient_matvec(
            out_ptr,
            v_ptr,
            self.cat_indices_ptr,
            self.corr_flat_ptr,
            self.offsets_ptr,
            self.levels_ptr,
            self.num_cat_vars,
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
            self.cat_indices_ptr,
            self.corr_flat_ptr,
            self.offsets_ptr,
            self.levels_ptr,
            self.num_cat_vars,
        )

    fn mixed_cross_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cat_test_ptr: UnsafePointer[Int32, MutAnyOrigin],
        n_test: Int,
        num_cols: Int,
    ) raises:
        self.provider.mixed_cross_matvec(
            out_ptr,
            x_test_ptr,
            v_ptr,
            cat_test_ptr,
            self.cat_indices_ptr,
            self.corr_flat_ptr,
            self.offsets_ptr,
            self.levels_ptr,
            self.num_cat_vars,
            n_test,
            num_cols,
        )


fn _apply_cat_gradient_kronecker(
    mixed_provider: MixedKroneckerBaseProviderView,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    temp_base_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    B_ptr: UnsafePointer[Float32, MutAnyOrigin],
    zero_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    T: Int,
    num_cols: Int,
    outputscale: Float32,
    grad_corr_ptr: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    alias BLOCK = 256
    var nT = n * T
    mixed_provider.provider.mixed_forward_matvec(
        temp_base_ptr,
        v_ptr,
        mixed_provider.cat_indices_ptr,
        grad_corr_ptr,
        mixed_provider.offsets_ptr,
        mixed_provider.levels_ptr,
        mixed_provider.num_cat_vars,
        T * num_cols,
        Float32(0.0),
    )
    mixed_provider.provider.get_ctx().enqueue_function[kernel_kronecker_combine_batched](
        out_ptr,
        temp_base_ptr,
        v_ptr,
        B_ptr,
        zero_noise_ptr,
        n,
        T,
        num_cols,
        outputscale,
        grid_dim=(ceildiv(nT, BLOCK),),
        block_dim=(BLOCK,),
    )
    mixed_provider.provider.get_ctx().synchronize()


fn train_multi_output_mixed_jit(
    owned provider: ErasedJITProvider,
    owned cat_state: CategoricalCorrelationState,
    ctx: DeviceContext,
    y_blocked_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
    num_kernel_params: Int,
    initial_params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    initial_noise_per_task_ptr: UnsafePointer[Float32, MutAnyOrigin],
    initial_outputscale: Float32,
    initial_mean_per_task_ptr: UnsafePointer[Float32, MutAnyOrigin],
    initial_cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    total_cat_params: Int,
    max_iterations: Int = 100,
    learning_rate: Float32 = Float32(0.05),
    task_rank: Int = -1,
    num_probes: Int = 10,
    max_cg_iter: Int = 200,
    max_tridiag_iter: Int = 30,
    cg_tol: Float32 = Float32(1.0),
    precond_rank: Int = 15,
    precond_method: Int = 0,
    precond_rebuild_threshold: Float32 = Float32(0.5),
    use_cosine_lr: Bool = False,
    early_stop_patience: Int = 15,
    early_stop_tol: Float32 = Float32(1e-4),
    verbose: Bool = True,
    use_materialized: Bool = False,
    progress_callback: PythonObject = None,
    progress_interval: Int = 1,
    progress_enabled: Bool = False,
) raises -> MixedMultiOutputJITResult:
    var cat_state_local = cat_state^
    var T = num_tasks
    var R = task_rank
    if R < 0:
        R = T
    var nT = n * T

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
    var raw_cat = List[Float32]()
    for k in range(total_cat_params):
        raw_cat.append(initial_cat_params_ptr[k])

    var task_cov = TaskCovariance(ctx, T, R)

    var total_adam = num_kernel_params + 1 + T + T * R + T + T
    var m_adam = List[Float32]()
    var v_adam = List[Float32]()
    for _ in range(total_adam):
        m_adam.append(Float32(0))
        v_adam.append(Float32(0))
    var m_cat = List[Float32]()
    var v_cat = List[Float32]()
    for _ in range(total_cat_params):
        m_cat.append(Float32(0))
        v_cat.append(Float32(0))

    var best_nll = Float32(1e30)
    var best_nll_seen = Float32(1e30)
    var patience_counter = 0
    var converged = False
    var actual_iters = 0
    var nll_history = List[Float32]()
    var cg_iterations_history = List[Int]()

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
    var best_raw_cat = List[Float32]()
    for k in range(total_cat_params):
        best_raw_cat.append(raw_cat[k])
    var best_sol_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for i in range(nT):
        best_sol_host.unsafe_ptr()[i] = Float32(0)
    var best_B_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    ctx.synchronize()
    for i in range(T * T):
        best_B_host.unsafe_ptr()[i] = Float32(0)

    var params_host_buf = ctx.enqueue_create_host_buffer[float_dtype](max(num_kernel_params, 1))
    var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    var mean_host_buf = ctx.enqueue_create_host_buffer[float_dtype](T)
    var mean_device_buf = ctx.enqueue_create_buffer[float_dtype](T)
    var y_centered_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.synchronize()
    for p in range(num_kernel_params):
        params_host_buf.unsafe_ptr()[p] = initial_params_host_ptr[p]
    if total_cat_params > 0:
        write_constrained_cat_params(cat_state_local, raw_cat, cat_params_host.unsafe_ptr())
        cat_state_local.update_correlation_matrices(cat_params_host.unsafe_ptr())

    var mixed_base = MixedKroneckerBaseProviderView(provider^, cat_state_local)
    mixed_base.update_params(params_host_buf.unsafe_ptr())
    mixed_base.update_noise(Float32(0))
    if use_materialized:
        mixed_base.refresh_materialization()

    var precond_B_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    var precond_noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    task_cov.update_B()
    for s in range(T):
        for t_idx in range(T):
            precond_B_host.unsafe_ptr()[s * T + t_idx] = task_cov.B.unsafe_ptr()[s * T + t_idx]
        precond_noise_host.unsafe_ptr()[s] = softplus(raw_noise[s])
    for i in range(T * T):
        best_B_host.unsafe_ptr()[i] = task_cov.B.unsafe_ptr()[i]

    var os_val_init = softplus(raw_outputscale)
    var kron_provider = KroneckerDirectProvider(
        mixed_base^,
        ctx,
        T,
        os_val_init,
        precond_B_host,
        precond_noise_host,
    )

    var num_cols = 1 + num_probes
    var precond_error_tol = Float32(1e-3)
    var use_preconditioner = precond_rank > 0
    if precond_method == 0:
        precond_error_tol = Float32(0.0)

    var pool = CGBufferPool(ctx, nT, num_cols, num_probes, max_tridiag_iter)
    pool.ensure_capacity(ctx, nT, num_cols, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=kron_provider.num_gradient_params())
    var precond = build_pivoted_cholesky_precond_unified(
        kron_provider,
        rank=precond_rank,
        error_tol=precond_error_tol,
        max_num_cols=num_cols,
        precond_method=precond_method,
    )
    var precond_rebuild_count = 0
    var last_rebuild_params = ctx.enqueue_create_host_buffer[float_dtype](max(num_kernel_params, 1))
    var last_rebuild_cat = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    var last_rebuild_noise = ctx.enqueue_create_host_buffer[float_dtype](T)
    var last_rebuild_B = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    ctx.synchronize()
    for p in range(num_kernel_params):
        last_rebuild_params.unsafe_ptr()[p] = params_host_buf.unsafe_ptr()[p]
    for k in range(total_cat_params):
        last_rebuild_cat.unsafe_ptr()[k] = cat_params_host.unsafe_ptr()[k]
    var last_rebuild_outputscale = os_val_init
    for s in range(T):
        last_rebuild_noise.unsafe_ptr()[s] = precond_noise_host.unsafe_ptr()[s]
        for t_idx in range(T):
            last_rebuild_B.unsafe_ptr()[s * T + t_idx] = precond_B_host.unsafe_ptr()[s * T + t_idx]

    var kx_alpha = ctx.enqueue_create_buffer[float_dtype](n * T)
    var kx_rf_batch = ctx.enqueue_create_buffer[float_dtype](n * T)
    var dot_matrix_device = ctx.enqueue_create_buffer[float_dtype](T * T)
    var dot_matrix_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    var dot_result_device = ctx.enqueue_create_buffer[float_dtype](1)
    var dot_result_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    var G_B = ctx.enqueue_create_host_buffer[float_dtype](T * T)

    var corr_stride = cat_state_local.total_corr_size
    var grad_buf_size = max(total_cat_params * corr_stride, 1)
    var all_grad_corr_host = HostBuffer[float_dtype](ctx, grad_buf_size)
    var all_grad_corr_device = ctx.enqueue_create_buffer[float_dtype](grad_buf_size)
    var zero_noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    for t in range(T):
        zero_noise_host.unsafe_ptr()[t] = Float32(0)
    var zero_noise_device = ctx.enqueue_create_buffer[float_dtype](T)
    ctx.enqueue_copy(zero_noise_device, zero_noise_host)
    ctx.synchronize()
    var cat_temp_base = ctx.enqueue_create_buffer[float_dtype](max(nT * num_probes, nT))
    var out_alpha_device = ctx.enqueue_create_buffer[float_dtype](nT)
    var out_probes_device = ctx.enqueue_create_buffer[float_dtype](max(nT * num_probes, 1))
    var sol_host_buf = ctx.enqueue_create_host_buffer[float_dtype](nT)
    var probe_sol_host_buf = ctx.enqueue_create_host_buffer[float_dtype](max(nT * num_probes, 1))
    var out_alpha_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    var out_probes_host = ctx.enqueue_create_host_buffer[float_dtype](max(nT * num_probes, 1))

    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var eps = Float32(1e-8)
    var t_step = 1

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "multi_output_icm",
            "materialized" if use_materialized else "matrix_free",
            "start",
            0,
            max_iterations,
            precond_rank=precond.rank,
            precond_rebuild_count=precond_rebuild_count,
        )

    for iteration in range(max_iterations):
        var iter_start = perf_counter_ns()
        actual_iters = iteration + 1

        for p in range(num_kernel_params):
            params_host_buf.unsafe_ptr()[p] = softplus(raw_params[p])
        if total_cat_params > 0:
            write_constrained_cat_params(cat_state_local, raw_cat, cat_params_host.unsafe_ptr())
            cat_state_local.update_correlation_matrices(cat_params_host.unsafe_ptr())

        kron_provider.base_provider.update_params(params_host_buf.unsafe_ptr())
        kron_provider.base_provider.update_noise(Float32(0))
        if use_materialized:
            kron_provider.base_provider.refresh_materialization()

        var os_val = softplus(raw_outputscale)
        var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
        ctx.synchronize()
        for t in range(T):
            noise_host.unsafe_ptr()[t] = softplus(raw_noise[t])
        task_cov.update_B()
        kron_provider.update_B(task_cov.B)
        kron_provider.update_noise(noise_host)
        kron_provider.update_outputscale(os_val)

        var should_rebuild = False
        if iteration > 0:
            var max_rel = Float32(0)
            for p in range(num_kernel_params):
                var prev = last_rebuild_params.unsafe_ptr()[p]
                var cur = params_host_buf.unsafe_ptr()[p]
                var rel = abs(cur - prev) / (abs(prev) + Float32(1e-8))
                if rel > max_rel:
                    max_rel = rel
            for k in range(total_cat_params):
                var prev_cat = last_rebuild_cat.unsafe_ptr()[k]
                var cur_cat = cat_params_host.unsafe_ptr()[k]
                var rel_cat = abs(cur_cat - prev_cat) / (abs(prev_cat) + Float32(1e-8))
                if rel_cat > max_rel:
                    max_rel = rel_cat
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
            precond = build_pivoted_cholesky_precond_unified(
                kron_provider,
                rank=precond_rank,
                error_tol=precond_error_tol,
                max_num_cols=num_cols,
                precond_method=precond_method,
            )
            precond_rebuild_count += 1
            for p in range(num_kernel_params):
                last_rebuild_params.unsafe_ptr()[p] = params_host_buf.unsafe_ptr()[p]
            for k in range(total_cat_params):
                last_rebuild_cat.unsafe_ptr()[k] = cat_params_host.unsafe_ptr()[k]
            last_rebuild_outputscale = os_val
            for s in range(T):
                last_rebuild_noise.unsafe_ptr()[s] = noise_host.unsafe_ptr()[s]
                for t_idx in range(T):
                    last_rebuild_B.unsafe_ptr()[s * T + t_idx] = task_cov.B.unsafe_ptr()[s * T + t_idx]

        pool.ensure_capacity(ctx, nT, num_cols, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=kron_provider.num_gradient_params())

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

        var bbmm_result = bbmm_with_precond(
            kron_provider,
            precond,
            y_centered_device.unsafe_ptr(),
            nT,
            pool,
            num_probes,
            max_cg_iter,
            max_tridiag_iter,
            cg_tol,
            iteration=iteration,
            recycle_alpha=iteration > 0 and not should_rebuild,
            use_preconditioner=use_preconditioner,
        )

        var nll = bbmm_result.nll
        nll_history.append(nll)
        cg_iterations_history.append(bbmm_result.num_iterations)
        if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
            print("MO mixed iter", iteration, ": NLL =", nll)

        ctx.enqueue_copy(sol_host_buf, bbmm_result.solution)
        if total_cat_params > 0 and num_probes > 0:
            ctx.enqueue_copy(probe_sol_host_buf, bbmm_result.probe_solutions)
        ctx.synchronize()

        if nll < best_nll_seen:
            best_nll_seen = nll
            for p in range(num_kernel_params):
                best_raw_params[p] = raw_params[p]
            best_raw_os = raw_outputscale
            for t in range(T):
                best_raw_noise[t] = raw_noise[t]
                best_raw_mean[t] = raw_mean[t]
            for k in range(total_cat_params):
                best_raw_cat[k] = raw_cat[k]
            for i in range(T * T):
                best_B_host.unsafe_ptr()[i] = task_cov.B.unsafe_ptr()[i]
            for i in range(nT):
                best_sol_host.unsafe_ptr()[i] = sol_host_buf.unsafe_ptr()[i]

        kron_provider.base_provider.forward_matvec(
            kx_alpha.unsafe_ptr(), bbmm_result.solution.unsafe_ptr(), T
        )
        ctx.enqueue_function[kernel_dot_matrix](
            bbmm_result.solution.unsafe_ptr(),
            kx_alpha.unsafe_ptr(),
            dot_matrix_device.unsafe_ptr(),
            n,
            T,
            T,
            grid_dim=(T * T,),
            block_dim=(256,),
        )
        ctx.enqueue_copy(dot_matrix_host, dot_matrix_device)
        ctx.synchronize()
        for i in range(T * T):
            G_B.unsafe_ptr()[i] = -dot_matrix_host.unsafe_ptr()[i]

        for j in range(num_probes):
            kron_provider.base_provider.forward_matvec(
                kx_rf_batch.unsafe_ptr(),
                bbmm_result.right_factors.unsafe_ptr().offset(j * nT),
                T,
            )
            ctx.enqueue_function[kernel_dot_matrix](
                bbmm_result.probe_solutions.unsafe_ptr().offset(j * nT),
                kx_rf_batch.unsafe_ptr(),
                dot_matrix_device.unsafe_ptr(),
                n,
                T,
                T,
                grid_dim=(T * T,),
                block_dim=(256,),
            )
            ctx.enqueue_copy(dot_matrix_host, dot_matrix_device)
            ctx.synchronize()
            for i in range(T * T):
                G_B.unsafe_ptr()[i] += dot_matrix_host.unsafe_ptr()[i] / Float32(num_probes)

        for i in range(T * T):
            G_B.unsafe_ptr()[i] = Float32(0.5) * os_val * G_B.unsafe_ptr()[i] / Float32(nT)
            if G_B.unsafe_ptr()[i] > Float32(10):
                G_B.unsafe_ptr()[i] = Float32(10)
            elif G_B.unsafe_ptr()[i] < Float32(-10):
                G_B.unsafe_ptr()[i] = Float32(-10)

        var grad_noise = List[Float32]()
        for t in range(T):
            ctx.enqueue_function[kernel_dot_batched](
                bbmm_result.solution.unsafe_ptr().offset(t * n),
                bbmm_result.solution.unsafe_ptr().offset(t * n),
                dot_result_device.unsafe_ptr(),
                n,
                1,
                grid_dim=(1, 1),
                block_dim=(256, 1),
            )
            ctx.enqueue_copy(dot_result_host, dot_result_device)
            ctx.synchronize()
            var data_term = -dot_result_host.unsafe_ptr()[0]

            var trace_sum = Float32(0)
            for j in range(num_probes):
                ctx.enqueue_function[kernel_dot_batched](
                    bbmm_result.probe_solutions.unsafe_ptr().offset(j * nT + t * n),
                    bbmm_result.right_factors.unsafe_ptr().offset(j * nT + t * n),
                    dot_result_device.unsafe_ptr(),
                    n,
                    1,
                    grid_dim=(1, 1),
                    block_dim=(256, 1),
                )
                ctx.enqueue_copy(dot_result_host, dot_result_device)
                ctx.synchronize()
                trace_sum += dot_result_host.unsafe_ptr()[0]
            grad_noise.append(Float32(0.5) * (data_term + trace_sum / Float32(num_probes)) / Float32(nT))

        if total_cat_params > 0:
            var effective_lr = learning_rate
            if use_cosine_lr:
                effective_lr = compute_cosine_lr(learning_rate, iteration, max_iterations)
            cat_state_local.compute_all_gradient_correlations(
                all_grad_corr_host, cat_params_host.unsafe_ptr()
            )
            ctx.enqueue_copy(all_grad_corr_device, all_grad_corr_host)
            ctx.synchronize()

            var inv_S = Float32(1.0) / Float32(num_probes)
            for k in range(total_cat_params):
                var grad_corr_ptr = all_grad_corr_device.unsafe_ptr() + k * corr_stride
                _apply_cat_gradient_kronecker(
                    kron_provider.base_provider,
                    out_alpha_device.unsafe_ptr(),
                    cat_temp_base.unsafe_ptr(),
                    bbmm_result.solution.unsafe_ptr(),
                    kron_provider.B_device.unsafe_ptr(),
                    zero_noise_device.unsafe_ptr(),
                    n,
                    T,
                    1,
                    os_val,
                    grad_corr_ptr,
                )
                ctx.enqueue_copy(out_alpha_host, out_alpha_device)
                ctx.synchronize()
                var dot_alpha = Float32(0.0)
                for i in range(nT):
                    dot_alpha += sol_host_buf.unsafe_ptr()[i] * out_alpha_host.unsafe_ptr()[i]

                _apply_cat_gradient_kronecker(
                    kron_provider.base_provider,
                    out_probes_device.unsafe_ptr(),
                    cat_temp_base.unsafe_ptr(),
                    pool.probes_device.unsafe_ptr(),
                    kron_provider.B_device.unsafe_ptr(),
                    zero_noise_device.unsafe_ptr(),
                    n,
                    T,
                    num_probes,
                    os_val,
                    grad_corr_ptr,
                )
                ctx.enqueue_copy(out_probes_host, out_probes_device)
                ctx.synchronize()

                var trace_term = Float32(0.0)
                for s in range(num_probes):
                    var ds = Float32(0.0)
                    for i in range(nT):
                        ds += probe_sol_host_buf.unsafe_ptr()[s * nT + i] * out_probes_host.unsafe_ptr()[s * nT + i]
                    trace_term += ds

                var cat_grad = (
                    Float32(-0.5) * dot_alpha + Float32(0.5) * inv_S * trace_term
                ) * cat_chain_derivative_for_param(cat_state_local, raw_cat, k)
                m_cat[k] = beta1 * m_cat[k] + (Float32(1) - beta1) * cat_grad
                v_cat[k] = beta2 * v_cat[k] + (Float32(1) - beta2) * cat_grad * cat_grad
                var m_hat_cat = m_cat[k] / (Float32(1) - pow_float32(beta1, t_step))
                var v_hat_cat = v_cat[k] / (Float32(1) - pow_float32(beta2, t_step))
                raw_cat[k] -= effective_lr * m_hat_cat / (sqrt(v_hat_cat) + eps)

        var idx = 0
        var effective_lr = learning_rate
        if use_cosine_lr:
            effective_lr = compute_cosine_lr(learning_rate, iteration, max_iterations)
        for p in range(num_kernel_params):
            var grad_p = bbmm_result.gradients[p]
            var sigmoid = Float32(1.0) / (Float32(1.0) + math_exp(-raw_params[p]))
            grad_p = grad_p * sigmoid
            m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * grad_p
            v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * grad_p * grad_p
            var m_hat = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
            var v_hat = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
            raw_params[p] -= effective_lr * m_hat / (sqrt(v_hat) + eps)
            idx += 1

        var grad_os = bbmm_result.gradients[num_kernel_params]
        var sigmoid_os = Float32(1.0) / (Float32(1.0) + math_exp(-raw_outputscale))
        grad_os = grad_os * sigmoid_os
        m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * grad_os
        v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * grad_os * grad_os
        var m_hat_os = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
        var v_hat_os = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
        raw_outputscale -= effective_lr * m_hat_os / (sqrt(v_hat_os) + eps)
        idx += 1

        for t in range(T):
            var gn = grad_noise[t] * (Float32(1.0) / (Float32(1.0) + math_exp(-raw_noise[t])))
            m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * gn
            v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * gn * gn
            var mh = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
            var vh = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
            raw_noise[t] -= effective_lr * mh / (sqrt(vh) + eps)
            idx += 1

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

        for t in range(T):
            var gv = G_B.unsafe_ptr()[t * T + t] * (Float32(1.0) / (Float32(1.0) + math_exp(-task_cov.raw_v.unsafe_ptr()[t])))
            m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * gv
            v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * gv * gv
            var mh = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
            var vh = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
            task_cov.raw_v.unsafe_ptr()[t] -= effective_lr * mh / (sqrt(vh) + eps)
            idx += 1

        for t in range(T):
            var mean_grad = Float32(0)
            for i in range(n):
                mean_grad -= sol_host_buf.unsafe_ptr()[t * n + i]
            mean_grad = clip_gradient(mean_grad / Float32(nT))
            m_adam[idx] = beta1 * m_adam[idx] + (Float32(1) - beta1) * mean_grad
            v_adam[idx] = beta2 * v_adam[idx] + (Float32(1) - beta2) * mean_grad * mean_grad
            var mh = m_adam[idx] / (Float32(1) - pow_float32(beta1, t_step))
            var vh = v_adam[idx] / (Float32(1) - pow_float32(beta2, t_step))
            raw_mean[t] -= effective_lr * mh / (sqrt(vh) + eps)
            idx += 1

        t_step += 1

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
                            "materialized" if use_materialized else "matrix_free",
                            "early_stop",
                            actual_iters,
                            max_iterations,
                            nll=nll,
                            best_nll=best_nll_seen,
                            cg_iter=bbmm_result.num_iterations,
                            iter_time_ns=Int(perf_counter_ns() - iter_start),
                            noise=softplus(raw_noise[0]) if T > 0 else Float32(0.0),
                            mean=raw_mean[0] if T > 0 else Float32(0.0),
                            precond_rank=precond.rank,
                            precond_rebuild_count=precond_rebuild_count,
                            converged=converged,
                        )
                    break

        if progress_enabled and progress_interval_should_emit(iteration, max_iterations, progress_interval):
            emit_progress_event(
                progress_callback,
                "train",
                "multi_output_icm",
                "materialized" if use_materialized else "matrix_free",
                "iteration",
                actual_iters,
                max_iterations,
                nll=nll,
                best_nll=best_nll_seen,
                cg_iter=bbmm_result.num_iterations,
                iter_time_ns=Int(perf_counter_ns() - iter_start),
                noise=softplus(raw_noise[0]) if T > 0 else Float32(0.0),
                mean=raw_mean[0] if T > 0 else Float32(0.0),
                precond_rank=precond.rank,
                precond_rebuild_count=precond_rebuild_count,
            )

        _ = noise_host
        _ = bbmm_result

    var final_params = List[Float32]()
    for p in range(num_kernel_params):
        final_params.append(softplus(best_raw_params[p]))
    var final_cat = List[Float32]()
    for k in range(total_cat_params):
        final_cat.append(best_raw_cat[k])
    var final_os = softplus(best_raw_os)
    var final_noise = List[Float32]()
    for t in range(T):
        final_noise.append(softplus(best_raw_noise[t]))
    var final_mean = List[Float32]()
    for t in range(T):
        final_mean.append(best_raw_mean[t])

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
            "materialized" if use_materialized else "matrix_free",
            "complete",
            actual_iters,
            max_iterations,
            nll=best_nll_seen,
            best_nll=best_nll_seen,
            noise=final_noise[0] if T > 0 else Float32(0.0),
            mean=final_mean[0] if T > 0 else Float32(0.0),
            precond_rank=precond.rank,
            precond_rebuild_count=precond_rebuild_count,
            converged=converged,
        )

    _ = cat_state_local
    _ = mean_host_buf
    _ = mean_device_buf
    _ = y_centered_device

    return MixedMultiOutputJITResult(
        final_params^,
        final_cat^,
        num_kernel_params,
        final_os,
        final_noise^,
        B_flat^,
        final_mean^,
        alpha_blocked^,
        nll_history^,
        cg_iterations_history^,
        best_nll_seen,
        actual_iters,
        converged,
        T,
        R,
        n,
        precond_rebuild_count,
    )


fn predict_variance_mixed_jit(
    provider: MixedKroneckerBaseProviderView,
    ctx: DeviceContext,
    x_test_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    cat_test_device_ptr: UnsafePointer[Int32, MutAnyOrigin],
    n_train: Int,
    n_test: Int,
    x_test_dim: Int,
    variance_method: Int,
    max_cg_iter: Int,
    cg_tol: Float32,
    precond_rank: Int,
    lanczos_rank: Int,
    precond_method: Int = 0,
    use_materialized: Bool = False,
) raises -> DeviceBuffer[float_dtype]:
    alias BLK = 256
    var provider_local = provider.copy()
    if use_materialized:
        provider_local.refresh_materialization()

    if variance_method == 1:
        var rank = lanczos_rank
        var inv_root_host = compute_lanczos_inv_root_jit(provider_local, rank)
        var inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * rank)
        ctx.enqueue_copy(inv_root_device, inv_root_host)
        ctx.synchronize()

        var V = ctx.enqueue_create_buffer[float_dtype](rank * n_test)
        provider_local.mixed_cross_matvec(
            V.unsafe_ptr(),
            x_test_device_ptr,
            inv_root_device.unsafe_ptr(),
            cat_test_device_ptr,
            n_test,
            rank,
        )

        var diag_test = ctx.enqueue_create_buffer[float_dtype](n_test)
        ctx.enqueue_function[_fill_scalar[BLK]](
            diag_test.unsafe_ptr(),
            provider_local.get_diagonal_value(),
            n_test,
            grid_dim=(ceildiv(n_test, BLK),),
            block_dim=(BLK,),
        )
        var out = ctx.enqueue_create_buffer[float_dtype](n_test)
        ctx.enqueue_function[kernel_love_variance[BLK]](
            out.unsafe_ptr(),
            V.unsafe_ptr(),
            diag_test.unsafe_ptr(),
            n_test,
            rank,
            grid_dim=(ceildiv(n_test, BLK),),
            block_dim=(BLK,),
        )
        ctx.synchronize()
        _ = inv_root_host
        _ = inv_root_device
        _ = V
        _ = diag_test
        return out^

    var out_exact_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    var diag_value = provider_local.get_diagonal_value()
    var block_cols = choose_exact_prediction_block_cols(n_test)
    var cat_block_device = ctx.enqueue_create_buffer[DType.int32](
        max(provider_local.num_cat_vars * block_cols, 1)
    )
    var test_start = 0
    while test_start < n_test:
        var active_cols = block_cols
        if n_test - test_start < active_cols:
            active_cols = n_test - test_start

        ctx.enqueue_function[_copy_cat_test_block_variable_major[BLK]](
            cat_block_device.unsafe_ptr(),
            cat_test_device_ptr,
            provider_local.num_cat_vars,
            n_test,
            test_start,
            active_cols,
            grid_dim=(ceildiv(provider_local.num_cat_vars * active_cols, BLK),),
            block_dim=(BLK,),
        )
        ctx.synchronize()

        var K_cross_host = build_mixed_cross_covariance_host_jit(
            provider_local,
            ctx,
            x_test_device_ptr.offset(test_start * x_test_dim),
            cat_block_device.unsafe_ptr(),
            n_train,
            active_cols,
        )

        for j in range(active_cols):
            var solve_host = solve_single_rhs_deterministic_host_jit(
                provider_local,
                ctx,
                K_cross_host.unsafe_ptr().offset(j * n_train),
                n_train,
                max_cg_iter,
                cg_tol,
            )
            var dot = Float32(0.0)
            for i in range(n_train):
                dot += K_cross_host.unsafe_ptr()[j * n_train + i] * solve_host.unsafe_ptr()[i]
            var variance = diag_value - dot
            if variance < Float32(1e-10):
                variance = Float32(1e-10)
            out_exact_host.unsafe_ptr()[test_start + j] = variance
            _ = solve_host
        _ = K_cross_host
        test_start += active_cols

    var out_exact = ctx.enqueue_create_buffer[float_dtype](n_test)
    ctx.enqueue_copy(out_exact, out_exact_host)
    ctx.synchronize()

    _ = cat_block_device
    _ = out_exact_host
    return out_exact^
