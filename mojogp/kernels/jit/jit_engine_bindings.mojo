"""JIT Engine Python bindings.

Pre-compiled .so that receives function pointers from kernel .so modules
and runs GP training via ErasedJITProvider + train_jit_with_provider.

Build: mojo build mojogp/kernels/jit/jit_engine_bindings.mojo --emit shared-lib -I mojogp/ -o mojogp_jit_engine
Import: import mojogp_jit_engine

Usage from Python:
    import mojogp_jit_engine as engine
    import some_kernel_so  # JIT-compiled kernel module
    
    # Step 1: Initialize kernel provider (in kernel .so)
    provider_info = some_kernel_so.init_provider(X, params, noise)
    
    # Step 2: Train via engine .so
    result = engine.train(provider_info, y, max_iterations=100, ...)
"""

from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from gpu.host import DeviceContext, DeviceBuffer
from os import abort
from time import perf_counter_ns

from kernels.jit.jit_training import train_jit_with_provider
from kernels.jit.erased_provider import (
    ErasedJITProvider,
    _cvt_fwd, _cvt_grad, _cvt_fused, _cvt_lsos, _cvt_3p,
    _cvt_diag, _cvt_upd, _cvt_unoise, _cvt_getf, _cvt_geti, _cvt_getptr, _cvt_xptr, _cvt_i32ptr,
    _cvt_cross, _cvt_diagtest, _cvt_fill_cross,
    get_noop_cross, get_noop_diagtest, get_noop_fill_cross, get_noop_kron_fwd, get_noop_kron_grad,
    get_noop_noise_mode, get_noop_noise_vector_ptr,
    _cvt_mixed_fwd, _cvt_mixed_fused_grad, _cvt_mixed_cross, _cvt_mixed_diag, _cvt_mixed_mat,
)
from kernels.jit.jit_prediction import (
    center_targets_jit,
    compute_alpha_jit,
    predict_from_alpha_jit,
    predict_jit,
    PREDICT_LOVE,
    compute_lanczos_inv_root_jit,
)
from kernels.jit.jit_mixed import train_mixed_jit, predict_mixed_jit, MixedTrainingResult
from kernels.py_conversion import bulk_copy_to_host_buffer
from kernels.constants import float_dtype, CAT_KERNEL_GD, CAT_KERNEL_CR, CAT_KERNEL_EHH, CAT_KERNEL_HH, CAT_KERNEL_FE
from kernels.bbmm_gpu_kernels import kernel_dot_batched_vs_strided
from kernels.categorical_state import CategoricalCorrelationState
from kernels.jit.jit_engine_binding_helpers import (
    _info_bool,
    _info_materialization_mode,
    _resolve_exact_predict_lanczos_rank,
)
from kernels.jit.jit_engine_multi_output_bindings import (
    train_multi_output_python,
    predict_multi_output_python,
    sample_multi_output_pathwise_python,
    train_multi_output_mixed_python,
    sample_multi_output_mixed_pathwise_python,
    predict_multi_output_mixed_python,
)
from kernels.jit.jit_engine_lmc_bindings import (
    train_lmc_mixed_python,
    train_lmc_python,
    sample_lmc_pathwise_python,
    sample_lmc_mixed_pathwise_python,
    predict_lmc_python,
    predict_lmc_mixed_python,
)


struct PredictionCacheJIT(Movable):
    """Device-resident single-output prediction state.

    The cache owns train-side buffers only: alpha and, optionally, the LOVE
    inverse root. Test-side cross-covariances are still computed per prediction.
    """
    var ctx: DeviceContext
    var alpha_device: DeviceBuffer[float_dtype]
    var inv_root_device: DeviceBuffer[float_dtype]
    var n_train: Int
    var rank: Int
    var has_love_root: Bool

    fn __init__(
        out self,
        ctx: DeviceContext,
        var alpha_device: DeviceBuffer[float_dtype],
        var inv_root_device: DeviceBuffer[float_dtype],
        n_train: Int,
        rank: Int,
        has_love_root: Bool,
    ):
        self.ctx = ctx
        self.alpha_device = alpha_device^
        self.inv_root_device = inv_root_device^
        self.n_train = n_train
        self.rank = rank
        self.has_love_root = has_love_root

    fn __moveinit__(out self, owned other: Self):
        self.ctx = other.ctx
        self.alpha_device = other.alpha_device^
        self.inv_root_device = other.inv_root_device^
        self.n_train = other.n_train
        self.rank = other.rank
        self.has_love_root = other.has_love_root


@always_inline
fn _cache_ptr_from_handle(handle: Int) -> UnsafePointer[PredictionCacheJIT, MutAnyOrigin]:
    var s = alloc[Int](1)
    s[] = handle
    var r = s.bitcast[UnsafePointer[PredictionCacheJIT, MutAnyOrigin]]()[]
    s.free()
    return r


# =============================================================================
# Python binding: train with fn-ptr provider
# =============================================================================

fn train_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Train GP using function pointers from a kernel .so.
    
    Args (from Python):
        args[0]: provider_info dict from kernel .so's init_provider()
                 Keys: provider_ptr, n, num_gradient_params, supports_fused_gradient,
                       supports_fused_ls_os, supports_fused_3param, x_ptr,
                       forward_matvec, gradient_matvec, fused_gradient_matvec,
                       fused_ls_os_gradient_matvec, fused_3param_gradient_matvec,
                       extract_diagonal, update_params, update_noise,
                       get_noise, get_diagonal_value
        args[1]: y numpy array (n,) float32
        args[2]: initial_params numpy array [num_params] float32
        args[3]: initial_noise (float)
        args[4]: max_iterations (int, default=100)
        args[5]: learning_rate (float, default=0.01)
        args[6]: num_probes (int, default=10)
        args[7]: max_cg_iter (int, default=100)
        args[8]: cg_tol (float, default=1e-2)
        args[9]: precond_rank (int, default=10)
        args[10]: verbose (bool, default=False)
        args[11]: num_probes (int, default=10)
        args[12]: max_cg_iter (int, default=200)
        args[13]: cg_tol (float, default=1.0)
        args[14]: precond_rank (int, default=15)
        args[15]: max_tridiag_iter (int, default=30)
        args[11]: use_cosine_lr (bool, default=True)
        args[12]: use_preconditioner (bool, default=True)
        args[13]: max_tridiag_iter (int, default=30)
        args[14]: precond_rebuild_threshold (float, default=0.5)
        args[15]: precond_method (int, default=0=greedy)
        args[16]: init_mean (float)
        args[17]: enable_early_stopping (bool, default=False)
        args[18]: early_stop_patience (int, default=10)
        args[19]: early_stop_tol (float, default=1e-4)
    
    Returns:
        dict with final training metrics, iteration telemetry, and preconditioner
        diagnostics for the live JIT BBMM path.
    """
    if len(args) < 17 or len(args) > 25:
        raise Error("train() expects between 17 and 25 positional arguments")

    var np = Python.import_module("numpy")
    
    # Parse provider info dict
    var info = args[0]
    var y_np = args[1]
    var params_np = args[2]
    var init_noise = Float32(Float64(args[3]))
    
    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n = Int(info["n"].__int__())
    var dim = Int(info["dim"].__int__()) if Bool(info.__contains__("dim")) else 0
    var num_gradient_params = Int(info["num_gradient_params"].__int__())
    var supports_fused = Bool(info["supports_fused_gradient"].__bool__())
    var supports_ls_os = Bool(info["supports_fused_ls_os"].__bool__())
    var supports_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())
    
    # Extract fn ptr addresses
    var fwd_ptr = Int(info["forward_matvec"].__int__())
    var grad_ptr = Int(info["gradient_matvec"].__int__())
    var fused_ptr = Int(info["fused_gradient_matvec"].__int__())
    var lsos_ptr = Int(info["fused_ls_os_gradient_matvec"].__int__())
    var threep_ptr = Int(info["fused_3param_gradient_matvec"].__int__())
    var diag_ptr = Int(info["extract_diagonal"].__int__())
    var updp_ptr = Int(info["update_params"].__int__())
    var updn_ptr = Int(info["update_noise"].__int__())
    var get_noise_ptr = Int(info["get_noise"].__int__())
    var get_noise_mode_ptr = Int(info["get_noise_mode"].__int__()) if Bool(info.__contains__("get_noise_mode")) else 0
    var get_noise_vector_ptr = Int(info["get_noise_vector_ptr"].__int__()) if Bool(info.__contains__("get_noise_vector_ptr")) else 0
    var get_diag_val_ptr = Int(info["get_diagonal_value"].__int__())
    var noise_group_ptr_addr = Int(info["noise_group_ptr"].__int__()) if Bool(info.__contains__("noise_group_ptr")) else 0
    var num_noise_groups = Int(info["num_noise_groups"].__int__()) if Bool(info.__contains__("num_noise_groups")) else 0
    
    # Parse optional training args
    var max_iterations = 100
    var learning_rate = Float32(0.01)
    var num_probes = 10
    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var verbose = False
    max_iterations = Int(args[4].__int__())
    learning_rate = Float32(Float64(args[5]))
    num_probes = Int(args[6].__int__())
    max_cg_iter = Int(args[7].__int__())
    cg_tol = Float32(Float64(args[8]))
    precond_rank = Int(args[9].__int__())
    verbose = Bool(args[10].__bool__())
    var use_cosine_lr = True
    use_cosine_lr = Bool(args[11].__bool__())
    var use_preconditioner = True
    use_preconditioner = Bool(args[12].__bool__())
    var max_tridiag_iter = Int(args[13].__int__())
    var precond_rebuild_threshold = Float32(Float64(args[14]))
    var precond_method = Int(args[15].__int__())
    var init_mean = Float32(Float64(args[16]))
    var enable_early_stopping = False
    var early_stop_patience = 10
    var early_stop_tol = Float32(1e-4)
    if len(args) > 17:
        enable_early_stopping = Bool(args[17].__bool__())
    if len(args) > 18:
        early_stop_patience = Int(args[18].__int__())
    if len(args) > 19:
        early_stop_tol = Float32(Float64(args[19]))
    var learn_noise = True
    if len(args) > 20:
        learn_noise = Bool(args[20].__bool__())
    var noise_floor = Float32(1e-6)
    if len(args) > 21:
        noise_floor = Float32(Float64(args[21]))
    var noise_regularization = Float32(0.01)
    if len(args) > 22:
        noise_regularization = Float32(Float64(args[22]))
    var progress_callback: PythonObject = None
    var progress_interval = 1
    var progress_enabled = False
    if len(args) > 23:
        progress_callback = args[23]
        progress_enabled = True
    if len(args) > 24:
        progress_interval = Int(args[24].__int__())
    
    # Create DeviceContext and copy y to host buffer
    var ctx = DeviceContext()
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    ctx.synchronize()
    var y_c = np.ascontiguousarray(y_np, dtype=np.float32)
    bulk_copy_to_host_buffer(y_c, y_host, n)
    
    # Create initial params host buffer
    var params_c = np.ascontiguousarray(params_np, dtype=np.float32).flatten()
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_gradient_params)
    ctx.synchronize()
    bulk_copy_to_host_buffer(params_c, params_host, num_gradient_params)
    
    var effective_x_ptr = _cvt_xptr(x_ptr_addr)
    var engine_x_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    var engine_x_device = ctx.enqueue_create_buffer[float_dtype](1)
    if Bool(info.__contains__("x_host")):
        var x_c = np.ascontiguousarray(info["x_host"], dtype=np.float32).flatten()
        engine_x_host = ctx.enqueue_create_host_buffer[float_dtype](n * dim)
        bulk_copy_to_host_buffer(x_c, engine_x_host, n * dim)
        engine_x_device = ctx.enqueue_create_buffer[float_dtype](n * dim)
        ctx.enqueue_copy(dst_buf=engine_x_device, src_buf=engine_x_host)
        ctx.synchronize()
        effective_x_ptr = engine_x_device.unsafe_ptr()

    # Create ErasedJITProvider — convert Int → fn ptr types ONCE here
    var provider = ErasedJITProvider(
        provider_ptr=provider_ptr,
        ctx=ctx,
        n=n,
        x_ptr=effective_x_ptr,
        num_gradient_params=num_gradient_params,
        supports_fused_gradient=supports_fused,
        supports_fused_ls_os=supports_ls_os,
        supports_fused_3param=supports_3p,
        forward_matvec=_cvt_fwd(fwd_ptr),
        gradient_matvec=_cvt_grad(grad_ptr),
        fused_gradient_matvec=_cvt_fused(fused_ptr),
        fused_ls_os_gradient_matvec=_cvt_lsos(lsos_ptr),
        fused_3param_gradient_matvec=_cvt_3p(threep_ptr),
        extract_diagonal=_cvt_diag(diag_ptr),
        update_params=_cvt_upd(updp_ptr),
        update_noise=_cvt_unoise(updn_ptr),
        get_noise=_cvt_getf(get_noise_ptr),
        get_noise_mode=_cvt_geti(get_noise_mode_ptr) if get_noise_mode_ptr != 0 else get_noop_noise_mode(),
        get_noise_vector_ptr=_cvt_getptr(get_noise_vector_ptr) if get_noise_vector_ptr != 0 else get_noop_noise_vector_ptr(),
        get_diagonal_value=_cvt_getf(get_diag_val_ptr),
        cross_matvec=get_noop_cross(),
        extract_diagonal_test=get_noop_diagtest(),
        has_prediction=False,
        kronecker_forward_matvec=get_noop_kron_fwd(),
        kronecker_gradient_matvec=get_noop_kron_grad(),
        has_kronecker=False,
    )
    
    # Train
    var result = train_jit_with_provider(
        provider, ctx, y_host.unsafe_ptr(), n, num_gradient_params,
        params_host.unsafe_ptr(), init_noise,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        num_probes=num_probes,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        verbose=verbose,
        init_mean=init_mean,
        max_tridiag_iter=max_tridiag_iter,
        precond_rebuild_threshold=precond_rebuild_threshold,
        use_cosine_lr=use_cosine_lr,
        use_preconditioner=use_preconditioner,
        precond_method=precond_method,
        enable_early_stopping=enable_early_stopping,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
        learn_noise=learn_noise,
        noise_floor=noise_floor,
        noise_regularization=noise_regularization,
        noise_group_ids_ptr=_cvt_i32ptr(noise_group_ptr_addr),
        num_noise_groups=num_noise_groups,
        noise_function_dim=dim,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        progress_enabled=progress_enabled,
    )
    
    # Build return dict
    var out = Python.dict()
    out["final_nll"] = Float64(result.final_nll)
    out["noise"] = Float64(result.noise)
    out["mean"] = Float64(result.mean)
    out["iterations"] = result.iterations
    out["converged"] = result.converged
    out["training_route"] = (
        "materialized" if _info_materialization_mode(info) == 1 else "matrix_free"
    )
    out["materialization_mode"] = _info_materialization_mode(info)
    out["is_ard"] = _info_bool(info, "is_ard")
    out["precond_method"] = precond_method
    out["precond_rank"] = precond_rank
    out["max_tridiag_iter"] = max_tridiag_iter
    out["precond_rebuild_threshold"] = Float64(precond_rebuild_threshold)
    out["use_preconditioner"] = use_preconditioner
    out["learn_noise"] = learn_noise
    out["noise_floor"] = Float64(noise_floor)
    out["noise_regularization"] = Float64(noise_regularization)
    out["enable_early_stopping"] = enable_early_stopping
    out["early_stop_patience"] = early_stop_patience
    out["early_stop_tol"] = Float64(early_stop_tol)
    
    var params_list = Python.list()
    for p in range(result.num_kernel_params):
        params_list.append(Float64(result.final_params[p]))
    out["params"] = params_list
    
    # Per-iteration timing (nanoseconds → milliseconds)
    var iter_times_list = Python.list()
    for t in range(len(result.iter_times_ns)):
        iter_times_list.append(Float64(result.iter_times_ns[t]) / 1e6)
    out["iter_times_ms"] = iter_times_list

    # NLL history (one value per iteration)
    var nll_history_list = Python.list()
    for i in range(len(result.nll_history)):
        nll_history_list.append(Float64(result.nll_history[i]))
    out["nll_history"] = nll_history_list

    var cg_history_list = Python.list()
    for i in range(len(result.cg_iterations_history)):
        cg_history_list.append(result.cg_iterations_history[i])
    out["cg_iterations_history"] = cg_history_list

    var precond_rank_history_list = Python.list()
    for i in range(len(result.precond_rank_history)):
        precond_rank_history_list.append(result.precond_rank_history[i])
    out["precond_rank_history"] = precond_rank_history_list

    var precond_rebuild_steps_list = Python.list()
    for i in range(len(result.precond_rebuild_steps)):
        precond_rebuild_steps_list.append(result.precond_rebuild_steps[i])
    out["precond_rebuild_steps"] = precond_rebuild_steps_list
    out["precond_build_count"] = result.precond_build_count
    out["precond_rebuild_count"] = max(0, result.precond_build_count - 1)
    out["precond_build_total_ms"] = Float64(result.precond_build_total_ns) / 1e6
    if len(result.precond_rank_history) > 0:
        out["actual_precond_rank"] = result.precond_rank_history[len(result.precond_rank_history) - 1]
    else:
        out["actual_precond_rank"] = 0

    if result.has_alpha:
        var alpha_arr = np.zeros(n, dtype=np.float32)
        for i in range(n):
            alpha_arr[i] = Float64(result.alpha[i])
        out["cached_alpha"] = alpha_arr

    if len(result.noise_function_params) > 0:
        var noise_fn_params_arr = np.zeros(len(result.noise_function_params), dtype=np.float32)
        for i in range(len(result.noise_function_params)):
            noise_fn_params_arr[i] = Float64(result.noise_function_params[i])
        out["learned_noise_function_params"] = noise_fn_params_arr

    if provider.get_noise_mode() == 2 or provider.get_noise_mode() == 3 or provider.get_noise_mode() == 4:
        var learned_noise_host = ctx.enqueue_create_host_buffer[float_dtype](n)
        var learned_noise_buf = DeviceBuffer[float_dtype](ctx, provider.get_noise_vector_ptr(), n, owning=False)
        ctx.enqueue_copy(dst_buf=learned_noise_host, src_buf=learned_noise_buf)
        ctx.synchronize()
        var learned_noise_arr = np.zeros(n, dtype=np.float32)
        for i in range(n):
            learned_noise_arr[i] = Float64(learned_noise_host[i])
        out["learned_observation_noise"] = learned_noise_arr
        if provider.get_noise_mode() == 4:
            out["noise_mode"] = "learned_input_dependent"
        elif provider.get_noise_mode() == 3:
            out["noise_mode"] = "learned_grouped"
        else:
            out["noise_mode"] = "learned_vector"

    # Keepalives
    _ = y_host
    _ = params_host
    _ = engine_x_host
    _ = engine_x_device

    return out


# =============================================================================
# Python binding: isolated generated matvec benchmarks
# =============================================================================

fn benchmark_provider_ops_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Benchmark generated provider matvec operations in isolation.

    Args:
        args[0]: provider_info dict from kernel .so's init_provider()
        args[1]: rhs numpy array, flattened column-major as [num_cols, n]
        args[2]: num_cols
        args[3]: op_code: 0=forward, 1=fused_gradient, 2=fused_gradient_plus_dots, 3=dot_only_after_gradient
        args[4]: repeats
        args[5]: warmup

    Returns:
        dict with per-call timings in milliseconds.
    """
    if len(args) != 6:
        raise Error("benchmark_provider_ops() expects 6 positional arguments")

    var np = Python.import_module("numpy")
    var info = args[0]
    var rhs_np = args[1]
    var num_cols = Int(args[2].__int__())
    var op_code = Int(args[3].__int__())
    var repeats = Int(args[4].__int__())
    var warmup = Int(args[5].__int__())

    if num_cols <= 0:
        raise Error("num_cols must be positive")
    if repeats <= 0:
        raise Error("repeats must be positive")
    if warmup < 0:
        raise Error("warmup must be non-negative")
    if op_code < 0 or op_code > 3:
        raise Error("op_code must be one of 0, 1, 2, 3")

    var ctx = DeviceContext()
    var provider = _provider_from_info(info, ctx)
    var n = Int(info["n"].__int__())
    var num_params = Int(info["num_gradient_params"].__int__())
    var total = n * num_cols
    var rhs_c = np.ascontiguousarray(rhs_np, dtype=np.float32).flatten()
    if Int(rhs_c.size.__int__()) != total:
        raise Error("rhs size does not match n * num_cols")

    var rhs_host = ctx.enqueue_create_host_buffer[float_dtype](total)
    ctx.synchronize()
    bulk_copy_to_host_buffer(rhs_c, rhs_host, total)
    var rhs_device = ctx.enqueue_create_buffer[float_dtype](total)
    ctx.enqueue_copy(rhs_device, rhs_host)
    ctx.synchronize()

    var out_size = total if op_code == 0 else num_params * total
    var out_device = ctx.enqueue_create_buffer[float_dtype](out_size)
    var dots_device = ctx.enqueue_create_buffer[float_dtype](num_params * num_cols)
    ctx.synchronize()

    if op_code == 3:
        provider.fused_gradient_matvec(out_device.unsafe_ptr(), rhs_device.unsafe_ptr(), num_cols)
        ctx.synchronize()

    for _ in range(warmup):
        if op_code == 0:
            provider.forward_matvec(out_device.unsafe_ptr(), rhs_device.unsafe_ptr(), num_cols)
        elif op_code == 1:
            provider.fused_gradient_matvec(out_device.unsafe_ptr(), rhs_device.unsafe_ptr(), num_cols)
        elif op_code == 2:
            provider.fused_gradient_matvec(out_device.unsafe_ptr(), rhs_device.unsafe_ptr(), num_cols)
            ctx.enqueue_function[kernel_dot_batched_vs_strided](
                rhs_device.unsafe_ptr(), out_device.unsafe_ptr(), dots_device.unsafe_ptr(),
                n, num_cols, num_params,
                grid_dim=(num_params * num_cols,), block_dim=(256,)
            )
            ctx.synchronize()
        else:
            ctx.enqueue_function[kernel_dot_batched_vs_strided](
                rhs_device.unsafe_ptr(), out_device.unsafe_ptr(), dots_device.unsafe_ptr(),
                n, num_cols, num_params,
                grid_dim=(num_params * num_cols,), block_dim=(256,)
            )
            ctx.synchronize()

    var times = Python.list()
    for _ in range(repeats):
        ctx.synchronize()
        var start_ns = perf_counter_ns()
        if op_code == 0:
            provider.forward_matvec(out_device.unsafe_ptr(), rhs_device.unsafe_ptr(), num_cols)
        elif op_code == 1:
            provider.fused_gradient_matvec(out_device.unsafe_ptr(), rhs_device.unsafe_ptr(), num_cols)
        elif op_code == 2:
            provider.fused_gradient_matvec(out_device.unsafe_ptr(), rhs_device.unsafe_ptr(), num_cols)
            ctx.enqueue_function[kernel_dot_batched_vs_strided](
                rhs_device.unsafe_ptr(), out_device.unsafe_ptr(), dots_device.unsafe_ptr(),
                n, num_cols, num_params,
                grid_dim=(num_params * num_cols,), block_dim=(256,)
            )
            ctx.synchronize()
        else:
            ctx.enqueue_function[kernel_dot_batched_vs_strided](
                rhs_device.unsafe_ptr(), out_device.unsafe_ptr(), dots_device.unsafe_ptr(),
                n, num_cols, num_params,
                grid_dim=(num_params * num_cols,), block_dim=(256,)
            )
            ctx.synchronize()
        ctx.synchronize()
        times.append(Float64(perf_counter_ns() - start_ns) / 1e6)

    var out = Python.dict()
    out["n"] = n
    out["num_cols"] = num_cols
    out["num_params"] = num_params
    out["op_code"] = op_code
    out["repeats"] = repeats
    out["warmup"] = warmup
    out["times_ms"] = times
    if op_code == 0:
        var out_host = ctx.enqueue_create_host_buffer[float_dtype](total)
        ctx.enqueue_copy(out_host, out_device)
        ctx.synchronize()
        var out_arr = np.zeros(total, dtype=np.float32)
        for i in range(total):
            out_arr[i] = Float64(out_host[i])
        out["output_column_major"] = out_arr
        _ = out_host

    _ = rhs_host
    _ = rhs_device
    _ = out_device
    _ = dots_device
    return out


# =============================================================================
# Python binding: predict with fn-ptr provider
# =============================================================================

fn _provider_from_info(info: PythonObject, ctx: DeviceContext) raises -> ErasedJITProvider:
    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n = Int(info["n"].__int__())
    var num_gradient_params = Int(info["num_gradient_params"].__int__())
    var supports_fused = Bool(info["supports_fused_gradient"].__bool__())
    var supports_ls_os = Bool(info["supports_fused_ls_os"].__bool__())
    var supports_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())

    var fwd_ptr = Int(info["forward_matvec"].__int__())
    var grad_ptr = Int(info["gradient_matvec"].__int__())
    var fused_ptr = Int(info["fused_gradient_matvec"].__int__())
    var lsos_ptr = Int(info["fused_ls_os_gradient_matvec"].__int__())
    var threep_ptr = Int(info["fused_3param_gradient_matvec"].__int__())
    var diag_ptr = Int(info["extract_diagonal"].__int__())
    var updp_ptr = Int(info["update_params"].__int__())
    var updn_ptr = Int(info["update_noise"].__int__())
    var get_noise_ptr = Int(info["get_noise"].__int__())
    var get_noise_mode_ptr = Int(info["get_noise_mode"].__int__()) if Bool(info.__contains__("get_noise_mode")) else 0
    var get_noise_vector_ptr = Int(info["get_noise_vector_ptr"].__int__()) if Bool(info.__contains__("get_noise_vector_ptr")) else 0
    var get_diag_val_ptr = Int(info["get_diagonal_value"].__int__())

    var has_cross = Bool(info.__contains__("cross_matvec"))
    var cross_ptr = Int(info["cross_matvec"].__int__()) if has_cross else 0
    var has_diag_test = Bool(info.__contains__("extract_diagonal_test"))
    var diag_test_ptr = Int(info["extract_diagonal_test"].__int__()) if has_diag_test else 0
    var has_fill_cross = Bool(info.__contains__("fill_cross_covariance"))
    var fill_cross_ptr = Int(info["fill_cross_covariance"].__int__()) if has_fill_cross else 0
    var has_prediction = has_cross and has_diag_test

    return ErasedJITProvider(
        provider_ptr=provider_ptr,
        ctx=ctx,
        n=n,
        x_ptr=_cvt_xptr(x_ptr_addr),
        num_gradient_params=num_gradient_params,
        supports_fused_gradient=supports_fused,
        supports_fused_ls_os=supports_ls_os,
        supports_fused_3param=supports_3p,
        forward_matvec=_cvt_fwd(fwd_ptr),
        gradient_matvec=_cvt_grad(grad_ptr),
        fused_gradient_matvec=_cvt_fused(fused_ptr),
        fused_ls_os_gradient_matvec=_cvt_lsos(lsos_ptr),
        fused_3param_gradient_matvec=_cvt_3p(threep_ptr),
        extract_diagonal=_cvt_diag(diag_ptr),
        update_params=_cvt_upd(updp_ptr),
        update_noise=_cvt_unoise(updn_ptr),
        get_noise=_cvt_getf(get_noise_ptr),
        get_noise_mode=_cvt_geti(get_noise_mode_ptr) if get_noise_mode_ptr != 0 else get_noop_noise_mode(),
        get_noise_vector_ptr=_cvt_getptr(get_noise_vector_ptr) if get_noise_vector_ptr != 0 else get_noop_noise_vector_ptr(),
        get_diagonal_value=_cvt_getf(get_diag_val_ptr),
        cross_matvec=_cvt_cross(cross_ptr) if has_prediction else get_noop_cross(),
        extract_diagonal_test=_cvt_diagtest(diag_test_ptr) if has_prediction else get_noop_diagtest(),
        has_prediction=has_prediction,
        fill_cross_covariance=_cvt_fill_cross(fill_cross_ptr) if has_fill_cross else get_noop_fill_cross(),
        has_fill_cross_covariance=has_fill_cross,
        kronecker_forward_matvec=get_noop_kron_fwd(),
        kronecker_gradient_matvec=get_noop_kron_grad(),
        has_kronecker=False,
    )


fn prepare_prediction_cache_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Prepare device-resident alpha and optional LOVE root for prediction."""
    var num_args = len(args)
    if num_args < 11 or num_args > 13:
        raise Error("prepare_prediction_cache() expects 11 to 13 positional arguments")

    var np = Python.import_module("numpy")
    var total_start = perf_counter_ns()

    var info = args[0]
    var y_np = args[1]
    var params_np = args[2]
    var final_noise = Float32(Float64(args[3]))
    var final_mean = Float32(Float64(args[4]))
    var variance_method = Int(args[5].__int__())
    var max_cg_iter = Int(args[6].__int__())
    var cg_tol = Float32(Float64(args[7]))
    var precond_rank = Int(args[8].__int__())
    var lanczos_rank = Int(args[9].__int__())
    var provider_state_current = Bool(args[10].__bool__())
    var use_cached_alpha = num_args > 11
    var use_cached_love_root = num_args > 12

    var n_train = Int(info["n"].__int__())
    var num_gradient_params = Int(info["num_gradient_params"].__int__())
    var use_materialized = _info_materialization_mode(info) != 0
    lanczos_rank = _resolve_exact_predict_lanczos_rank(
        info, lanczos_rank, False, use_materialized,
    )

    var ctx = DeviceContext()
    var provider = _provider_from_info(info, ctx)
    if not provider_state_current:
        var params_c = np.ascontiguousarray(params_np, dtype=np.float32).flatten()
        var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_gradient_params)
        ctx.synchronize()
        bulk_copy_to_host_buffer(params_c, params_host, num_gradient_params)
        provider.update_params(params_host.unsafe_ptr())
        provider.update_noise(final_noise)
        _ = params_host

    var alpha_device: DeviceBuffer[float_dtype]
    var alpha_time_ns = Int(0)
    var alpha_cache_used = False
    if use_cached_alpha:
        var alpha_host = ctx.enqueue_create_host_buffer[float_dtype](n_train)
        ctx.synchronize()
        var alpha_c = np.ascontiguousarray(args[11], dtype=np.float32).flatten()
        var alpha_start = perf_counter_ns()
        bulk_copy_to_host_buffer(alpha_c, alpha_host, n_train)
        alpha_device = ctx.enqueue_create_buffer[float_dtype](n_train)
        ctx.enqueue_copy(alpha_device, alpha_host)
        ctx.synchronize()
        alpha_time_ns = Int(perf_counter_ns() - alpha_start)
        alpha_cache_used = True
        _ = alpha_host
    else:
        var y_host = ctx.enqueue_create_host_buffer[float_dtype](n_train)
        ctx.synchronize()
        var y_c = np.ascontiguousarray(y_np, dtype=np.float32)
        bulk_copy_to_host_buffer(y_c, y_host, n_train)
        var y_device = ctx.enqueue_create_buffer[float_dtype](n_train)
        ctx.enqueue_copy(y_device, y_host)
        ctx.synchronize()
        var y_centered = center_targets_jit(ctx, y_device.unsafe_ptr(), n_train, final_mean)
        var alpha_start = perf_counter_ns()
        alpha_device = compute_alpha_jit(
            provider,
            ctx,
            y_centered,
            n_train,
            max_cg_iter,
            cg_tol,
            precond_rank,
            True,
        )
        alpha_time_ns = Int(perf_counter_ns() - alpha_start)
        _ = y_host
        _ = y_device
        _ = y_centered

    var inv_root_device = ctx.enqueue_create_buffer[float_dtype](1)
    var love_root_time_ns = Int(0)
    var love_root_cache_used = False
    var has_love_root = False
    var cached_root_rank = 0
    if variance_method == PREDICT_LOVE:
        var root_start = perf_counter_ns()
        if use_cached_love_root:
            var inv_root_host = ctx.enqueue_create_host_buffer[float_dtype](n_train * lanczos_rank)
            ctx.synchronize()
            var inv_root_c = np.ascontiguousarray(args[12], dtype=np.float32).flatten()
            if len(inv_root_c) == n_train * lanczos_rank:
                bulk_copy_to_host_buffer(inv_root_c, inv_root_host, n_train * lanczos_rank)
                inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * lanczos_rank)
                ctx.enqueue_copy(inv_root_device, inv_root_host)
                ctx.synchronize()
                love_root_cache_used = True
                has_love_root = True
                cached_root_rank = lanczos_rank
            _ = inv_root_host
        if not has_love_root:
            var inv_root_host_out = compute_lanczos_inv_root_jit(provider, lanczos_rank)
            inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * lanczos_rank)
            ctx.enqueue_copy(inv_root_device, inv_root_host_out)
            ctx.synchronize()
            has_love_root = True
            cached_root_rank = lanczos_rank
            _ = inv_root_host_out
        love_root_time_ns = Int(perf_counter_ns() - root_start)

    var cache_ptr = alloc[PredictionCacheJIT](1)
    cache_ptr.init_pointee_move(PredictionCacheJIT(
        ctx,
        alpha_device^,
        inv_root_device^,
        n_train,
        cached_root_rank,
        has_love_root,
    ))

    var out = Python.dict()
    out["cache_handle"] = Int(cache_ptr)
    out["n_train"] = n_train
    out["rank"] = cached_root_rank
    out["has_love_root"] = has_love_root
    out["alpha_time_s"] = Float64(alpha_time_ns) / 1e9
    out["love_root_time_s"] = Float64(love_root_time_ns) / 1e9
    out["prepare_time_s"] = Float64(perf_counter_ns() - total_start) / 1e9
    out["alpha_cache_used"] = alpha_cache_used
    out["love_root_cache_used"] = love_root_cache_used
    return out


fn destroy_prediction_cache_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Destroy a prediction cache returned by prepare_prediction_cache()."""
    if len(args) != 1:
        raise Error("destroy_prediction_cache() expects 1 positional argument")
    var handle = Int(args[0].__int__())
    if handle != 0:
        var cache_ptr = _cache_ptr_from_handle(handle)
        cache_ptr.destroy_pointee()
        cache_ptr.free()
    return PythonObject(True)


fn predict_with_cache_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Predict using a device-resident PredictionCacheJIT handle."""
    var num_args = len(args)
    if num_args < 7 or num_args > 14:
        raise Error("predict_with_cache() expects 7 to 14 positional arguments")

    var np = Python.import_module("numpy")
    var total_start_ns = Int(perf_counter_ns())

    var info = args[0]
    var cache_handle = Int(args[1].__int__())
    if cache_handle == 0:
        raise Error("predict_with_cache() received null cache handle")
    var cache_ptr = _cache_ptr_from_handle(cache_handle)
    var x_test_np = args[2]
    var params_np = args[3]
    var final_noise = Float32(Float64(args[4]))
    var final_mean = Float32(Float64(args[5]))
    var variance_method = Int(args[6].__int__())

    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var lanczos_rank = 0
    var provider_state_current = False
    var exact_block_cols_override = 0
    if num_args > 7:
        max_cg_iter = Int(args[7].__int__())
    if num_args > 8:
        cg_tol = Float32(Float64(args[8]))
    if num_args > 9:
        precond_rank = Int(args[9].__int__())
    if num_args > 10:
        lanczos_rank = Int(args[10].__int__())
    if num_args > 11:
        provider_state_current = Bool(args[11].__bool__())
    if num_args > 12:
        exact_block_cols_override = Int(args[12].__int__())

    var n_train = Int(info["n"].__int__())
    if n_train != cache_ptr[].n_train:
        raise Error("prediction cache n_train does not match provider")
    var num_gradient_params = Int(info["num_gradient_params"].__int__())
    var use_materialized = _info_materialization_mode(info) != 0
    lanczos_rank = _resolve_exact_predict_lanczos_rank(
        info, lanczos_rank, False, use_materialized,
    )
    _ = lanczos_rank

    var ctx = cache_ptr[].ctx
    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])
    var provider = _provider_from_info(info, ctx)
    var diag_provider = provider.clone()
    var use_diag_provider = False
    if num_args > 13:
        diag_provider = _provider_from_info(args[13], ctx)
        use_diag_provider = True

    if not provider_state_current or use_diag_provider:
        var params_c = np.ascontiguousarray(params_np, dtype=np.float32).flatten()
        var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_gradient_params)
        ctx.synchronize()
        bulk_copy_to_host_buffer(params_c, params_host, num_gradient_params)
        if not provider_state_current:
            provider.update_params(params_host.unsafe_ptr())
            provider.update_noise(final_noise)
        if use_diag_provider:
            diag_provider.update_params(params_host.unsafe_ptr())
            diag_provider.update_noise(final_noise)
        _ = params_host

    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)
    ctx.synchronize()

    var cached_inv_root_device_ptr = UnsafePointer[Float32, MutAnyOrigin]()
    var cached_inv_root_rank = 0
    var use_cached_inv_root = False
    if variance_method == PREDICT_LOVE and cache_ptr[].has_love_root:
        cached_inv_root_device_ptr = cache_ptr[].inv_root_device.unsafe_ptr()
        cached_inv_root_rank = cache_ptr[].rank
        use_cached_inv_root = True

    var result = predict_from_alpha_jit(
        provider,
        diag_provider,
        ctx,
        cache_ptr[].alpha_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        n_train,
        n_test,
        dim,
        final_noise,
        final_mean,
        variance_method,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        lanczos_rank=cache_ptr[].rank,
        use_diag_provider=use_diag_provider,
        prefer_materialized_multicol=use_materialized,
        exact_block_cols_override=exact_block_cols_override,
        cached_inv_root_device_ptr=cached_inv_root_device_ptr,
        cached_inv_root_rank=cached_inv_root_rank,
        use_cached_inv_root=use_cached_inv_root,
        alpha_time_ns=0,
        total_start_ns=total_start_ns,
    )

    var output_copy_start = perf_counter_ns()
    var out = Python.dict()
    var mean_arr = np.zeros(n_test, dtype=np.float32)
    for i in range(n_test):
        mean_arr[i] = Float64(result.mean.unsafe_ptr()[i])
    out["mean"] = mean_arr
    if result.has_variance:
        var var_arr = np.zeros(n_test, dtype=np.float32)
        for i in range(n_test):
            var_arr[i] = Float64(result.variance.unsafe_ptr()[i])
        out["variance"] = var_arr
        out["std"] = np.sqrt(var_arr)
    else:
        out["variance"] = np.zeros(n_test, dtype=np.float32)
        out["std"] = np.zeros(n_test, dtype=np.float32)

    out["alpha_time_s"] = Float64(result.alpha_time_ns) / 1e9
    out["mean_time_s"] = Float64(result.mean_time_ns) / 1e9
    out["variance_time_s"] = Float64(result.variance_time_ns) / 1e9
    out["total_time_s"] = Float64(result.total_time_ns) / 1e9
    out["love_root_time_s"] = Float64(0.0)
    out["exact_block_cols"] = result.exact_block_cols
    out["exact_cross_mode"] = result.exact_cross_mode
    out["exact_cg_block_count"] = result.exact_cg_block_count
    out["exact_cg_total_iterations"] = result.exact_cg_total_iterations
    out["exact_cg_max_iterations"] = result.exact_cg_max_iterations
    out["exact_alloc_time_s"] = Float64(result.exact_alloc_time_ns) / 1e9
    out["exact_cross_time_s"] = Float64(result.exact_cross_time_ns) / 1e9
    out["exact_diag_time_s"] = Float64(result.exact_diag_time_ns) / 1e9
    out["exact_solve_time_s"] = Float64(result.exact_solve_time_ns) / 1e9
    out["exact_post_time_s"] = Float64(result.exact_post_time_ns) / 1e9
    out["love_alloc_time_s"] = Float64(result.love_alloc_time_ns) / 1e9
    out["love_cross_time_s"] = Float64(result.love_cross_time_ns) / 1e9
    out["love_diag_time_s"] = Float64(result.love_diag_time_ns) / 1e9
    out["love_post_time_s"] = Float64(result.love_post_time_ns) / 1e9
    out["love_cross_strategy"] = result.love_cross_strategy
    out["love_cross_chunk_width"] = result.love_cross_chunk_width
    out["provider_state_update_skipped"] = provider_state_current
    out["alpha_cache_used"] = True
    out["love_root_cache_used"] = result.love_root_cache_used
    out["prediction_cache_used"] = True
    out["output_copy_time_s"] = Float64(perf_counter_ns() - output_copy_start) / 1e9

    _ = x_test_host
    _ = x_test_device
    _ = cache_ptr
    return out

fn predict_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Predict using function pointers from a kernel .so.
    
    Args (from Python):
        args[0]: provider_info dict (must include cross_matvec, extract_diagonal_test)
        args[1]: y_train numpy array (n,) float32
        args[2]: x_test numpy array (m, d) float32
        args[3]: final_params numpy array [num_params] float32
        args[4]: final_noise (float)
        args[5]: final_mean (float)
        args[6]: variance_method (int): 0=mean_only, 1=love, 2=exact
        args[7]: max_cg_iter (int, default=100)
        args[8]: cg_tol (float, default=1e-2)
        args[9]: precond_rank (int, default=10)
        args[10]: lanczos_rank (int, optional, default auto)
        args[11]: provider_state_current (bool, optional, default False)
        args[12]: exact prediction block-column override (int, optional, default 0)
        args[13]: optional cached alpha numpy array [n_train] float32
        args[14]: optional cached LOVE inverse root numpy array [n_train * rank] float32
        args[15]: optional test-provider_info dict for safe test diagonal extraction
    
    Returns:
        dict with: mean, variance, std (numpy arrays)
    """
    var num_args = len(args)
    if num_args < 7 or num_args > 16:
        raise Error("predict() expects 7 to 16 positional arguments")

    var np = Python.import_module("numpy")
    
    # Parse args
    var info = args[0]
    var y_np = args[1]
    var x_test_np = args[2]
    var params_np = args[3]
    var final_noise = Float32(Float64(args[4]))
    var final_mean = Float32(Float64(args[5]))
    var variance_method = Int(args[6].__int__())
    
    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var lanczos_rank = 0
    var provider_state_current = False
    var exact_block_cols_override = 0
    var use_cached_alpha = False
    var use_cached_love_root = False
    var args_len = num_args
    if args_len > 7:
        max_cg_iter = Int(args[7].__int__())
    if args_len > 8:
        cg_tol = Float32(Float64(args[8]))
    if args_len > 9:
        precond_rank = Int(args[9].__int__())
    if args_len > 10:
        lanczos_rank = Int(args[10].__int__())
    if args_len > 11:
        provider_state_current = Bool(args[11].__bool__())
    if args_len > 12:
        exact_block_cols_override = Int(args[12].__int__())
    if args_len > 13:
        use_cached_alpha = True
    if args_len > 14:
        use_cached_love_root = True
    
    var n_train = Int(info["n"].__int__())
    var num_gradient_params = Int(info["num_gradient_params"].__int__())
    var use_materialized = _info_materialization_mode(info) != 0
    lanczos_rank = _resolve_exact_predict_lanczos_rank(
        info, lanczos_rank, False, use_materialized,
    )
    
    # Update provider with final trained params
    var ctx = DeviceContext()
    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])
    
    # Create ErasedJITProvider
    var provider = _provider_from_info(info, ctx)
    var diag_provider = provider.clone()
    var use_diag_provider = False
    if args_len > 15:
        diag_provider = _provider_from_info(args[15], ctx)
        use_diag_provider = True

    if not provider_state_current or use_diag_provider:
        var params_c = np.ascontiguousarray(params_np, dtype=np.float32).flatten()
        var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_gradient_params)
        ctx.synchronize()
        bulk_copy_to_host_buffer(params_c, params_host, num_gradient_params)

        # Reused training providers already hold the trained state.
        if not provider_state_current:
            provider.update_params(params_host.unsafe_ptr())
            provider.update_noise(final_noise)
        if use_diag_provider:
            diag_provider.update_params(params_host.unsafe_ptr())
            diag_provider.update_noise(final_noise)

        _ = params_host
    
    # Copy x_test to GPU
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)
    ctx.synchronize()

    var total_start_ns = Int(perf_counter_ns())
    var alpha_time_ns = Int(0)
    var love_root_time_ns = Int(0)
    var alpha_cache_used = False
    var love_root_cache_used = False
    var inv_root_device = ctx.enqueue_create_buffer[float_dtype](1)

    var alpha_device: DeviceBuffer[float_dtype]
    if use_cached_alpha:
        var alpha_host = ctx.enqueue_create_host_buffer[float_dtype](n_train)
        ctx.synchronize()
        var alpha_c = np.ascontiguousarray(args[13], dtype=np.float32).flatten()
        var alpha_start = perf_counter_ns()
        bulk_copy_to_host_buffer(alpha_c, alpha_host, n_train)
        alpha_device = ctx.enqueue_create_buffer[float_dtype](n_train)
        ctx.enqueue_copy(alpha_device, alpha_host)
        ctx.synchronize()
        alpha_time_ns = Int(perf_counter_ns() - alpha_start)
        alpha_cache_used = True
        _ = alpha_host
    else:
        var y_host = ctx.enqueue_create_host_buffer[float_dtype](n_train)
        ctx.synchronize()
        var y_c = np.ascontiguousarray(y_np, dtype=np.float32)
        bulk_copy_to_host_buffer(y_c, y_host, n_train)
        var y_device = ctx.enqueue_create_buffer[float_dtype](n_train)
        ctx.enqueue_copy(y_device, y_host)
        ctx.synchronize()

        var y_centered = center_targets_jit(ctx, y_device.unsafe_ptr(), n_train, final_mean)
        var alpha_start = perf_counter_ns()
        alpha_device = compute_alpha_jit(
            provider,
            ctx,
            y_centered,
            n_train,
            max_cg_iter,
            cg_tol,
            precond_rank,
            True,
        )
        alpha_time_ns = Int(perf_counter_ns() - alpha_start)
        _ = y_host
        _ = y_device
        _ = y_centered

    var cached_inv_root_device_ptr = UnsafePointer[Float32, MutAnyOrigin]()
    var cached_inv_root_rank = 0
    if variance_method == PREDICT_LOVE:
        if use_cached_love_root:
            var root_start = perf_counter_ns()
            var inv_root_host = ctx.enqueue_create_host_buffer[float_dtype](n_train * lanczos_rank)
            ctx.synchronize()
            var inv_root_c = np.ascontiguousarray(args[14], dtype=np.float32).flatten()
            if len(inv_root_c) == n_train * lanczos_rank:
                bulk_copy_to_host_buffer(inv_root_c, inv_root_host, n_train * lanczos_rank)
                inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * lanczos_rank)
                ctx.enqueue_copy(inv_root_device, inv_root_host)
                ctx.synchronize()
                cached_inv_root_device_ptr = inv_root_device.unsafe_ptr()
                cached_inv_root_rank = lanczos_rank
                love_root_cache_used = True
            else:
                var inv_root_host_out = compute_lanczos_inv_root_jit(provider, lanczos_rank)
                inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * lanczos_rank)
                ctx.enqueue_copy(inv_root_device, inv_root_host_out)
                ctx.synchronize()
                cached_inv_root_device_ptr = inv_root_device.unsafe_ptr()
                cached_inv_root_rank = lanczos_rank
                _ = inv_root_host_out
            love_root_time_ns = Int(perf_counter_ns() - root_start)
            _ = inv_root_host
        else:
            var root_start = perf_counter_ns()
            var inv_root_host_out = compute_lanczos_inv_root_jit(provider, lanczos_rank)
            inv_root_device = ctx.enqueue_create_buffer[float_dtype](n_train * lanczos_rank)
            ctx.enqueue_copy(inv_root_device, inv_root_host_out)
            ctx.synchronize()
            cached_inv_root_device_ptr = inv_root_device.unsafe_ptr()
            cached_inv_root_rank = lanczos_rank
            love_root_time_ns = Int(perf_counter_ns() - root_start)
            _ = inv_root_host_out

    var result = predict_from_alpha_jit(
        provider,
        diag_provider,
        ctx,
        alpha_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        n_train,
        n_test,
        dim,
        final_noise,
        final_mean,
        variance_method,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        lanczos_rank=lanczos_rank,
        use_diag_provider=use_diag_provider,
        prefer_materialized_multicol=use_materialized,
        exact_block_cols_override=exact_block_cols_override,
        cached_inv_root_device_ptr=cached_inv_root_device_ptr,
        cached_inv_root_rank=cached_inv_root_rank,
        use_cached_inv_root=love_root_cache_used,
        alpha_time_ns=alpha_time_ns,
        total_start_ns=total_start_ns,
    )

    # Build return dict with numpy arrays
    var output_copy_start = perf_counter_ns()
    var out = Python.dict()

    var mean_arr = np.zeros(n_test, dtype=np.float32)
    for i in range(n_test):
        mean_arr[i] = Float64(result.mean.unsafe_ptr()[i])
    out["mean"] = mean_arr

    if result.has_variance:
        var var_arr = np.zeros(n_test, dtype=np.float32)
        for i in range(n_test):
            var_arr[i] = Float64(result.variance.unsafe_ptr()[i])
        out["variance"] = var_arr
        out["std"] = np.sqrt(var_arr)
    else:
        out["variance"] = np.zeros(n_test, dtype=np.float32)
        out["std"] = np.zeros(n_test, dtype=np.float32)

    out["alpha_time_s"] = Float64(result.alpha_time_ns) / 1e9
    out["mean_time_s"] = Float64(result.mean_time_ns) / 1e9
    out["variance_time_s"] = Float64(result.variance_time_ns) / 1e9
    out["total_time_s"] = Float64(result.total_time_ns) / 1e9
    out["love_root_time_s"] = Float64(love_root_time_ns) / 1e9
    out["exact_block_cols"] = result.exact_block_cols
    out["exact_cross_mode"] = result.exact_cross_mode
    out["exact_cg_block_count"] = result.exact_cg_block_count
    out["exact_cg_total_iterations"] = result.exact_cg_total_iterations
    out["exact_cg_max_iterations"] = result.exact_cg_max_iterations
    out["exact_alloc_time_s"] = Float64(result.exact_alloc_time_ns) / 1e9
    out["exact_cross_time_s"] = Float64(result.exact_cross_time_ns) / 1e9
    out["exact_diag_time_s"] = Float64(result.exact_diag_time_ns) / 1e9
    out["exact_solve_time_s"] = Float64(result.exact_solve_time_ns) / 1e9
    out["exact_post_time_s"] = Float64(result.exact_post_time_ns) / 1e9
    out["love_alloc_time_s"] = Float64(result.love_alloc_time_ns) / 1e9
    out["love_cross_time_s"] = Float64(result.love_cross_time_ns) / 1e9
    out["love_diag_time_s"] = Float64(result.love_diag_time_ns) / 1e9
    out["love_post_time_s"] = Float64(result.love_post_time_ns) / 1e9
    out["love_cross_strategy"] = result.love_cross_strategy
    out["love_cross_chunk_width"] = result.love_cross_chunk_width
    out["provider_state_update_skipped"] = provider_state_current
    out["alpha_cache_used"] = alpha_cache_used
    out["love_root_cache_used"] = result.love_root_cache_used
    out["output_copy_time_s"] = Float64(perf_counter_ns() - output_copy_start) / 1e9

    if not alpha_cache_used:
        var alpha_host_out = ctx.enqueue_create_host_buffer[float_dtype](n_train)
        ctx.enqueue_copy(alpha_host_out, alpha_device)
        ctx.synchronize()
        var alpha_arr = np.zeros(n_train, dtype=np.float32)
        for i in range(n_train):
            alpha_arr[i] = Float64(alpha_host_out[i])
        out["cached_alpha"] = alpha_arr
        _ = alpha_host_out

    if variance_method == PREDICT_LOVE and not love_root_cache_used:
        var inv_root_host_out = ctx.enqueue_create_host_buffer[float_dtype](n_train * lanczos_rank)
        ctx.enqueue_copy(inv_root_host_out, inv_root_device)
        ctx.synchronize()
        var inv_root_arr = np.zeros(n_train * lanczos_rank, dtype=np.float32)
        for i in range(n_train * lanczos_rank):
            inv_root_arr[i] = Float64(inv_root_host_out[i])
        out["cached_lanczos_root"] = inv_root_arr
        _ = inv_root_host_out

    # Keepalives
    _ = alpha_device
    _ = inv_root_device
    _ = x_test_host
    _ = x_test_device

    return out


# =============================================================================
# Python binding: train_mixed (continuous + categorical)
# =============================================================================

fn train_mixed_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Train mixed continuous+categorical GP.
    
    Uses continuous fn-ptrs from kernel .so + categorical correlation matrices
    managed by the engine's CategoricalCorrelationState.
    
    Args:
        args[0]: provider_info dict
        args[1]: y numpy array (n,) float32
        args[2]: initial_cont_params numpy num_cont_params float32
        args[3]: initial_noise float
        args[4]: cat_data numpy [n, num_cat_vars] int32
        args[5]: cat_specs list of dicts {"levels": int, "kernel_type": str}
        args[6]: initial_cat_params numpy total_cat_params float32
        args[7]: max_iterations int
        args[8]: learning_rate float
        args[14]: method int (0=matrix_free, 1=materialized, 3=auto)
        args[15]: precond_method int (optional, default=0=greedy)
        args[16]: enable_early_stopping bool (optional, default=False)
        args[17]: early_stop_patience int (optional, default=10)
        args[18]: early_stop_tol float (optional, default=1e-4)
    """
    var np = Python.import_module("numpy")

    var info = args[0]
    var y_np = args[1]
    var cont_params_np = args[2]
    var init_noise = Float32(Float64(args[3]))
    var cat_data_np = args[4]
    var cat_specs_py = args[5]
    var cat_params_np = args[6]

    var max_iters = 100
    var lr = Float32(0.01)
    var num_probes = 10
    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var verbose = False
    var method_int = 0
    var args_len = len(args)
    if args_len > 7:
        max_iters = Int(args[7].__int__())
    if args_len > 8:
        lr = Float32(Float64(args[8]))
    if args_len > 9:
        num_probes = Int(args[9].__int__())
    if args_len > 10:
        max_cg_iter = Int(args[10].__int__())
    if args_len > 11:
        cg_tol = Float32(Float64(args[11]))
    if args_len > 12:
        precond_rank = Int(args[12].__int__())
    if args_len > 13:
        verbose = Bool(args[13].__bool__())
    var precond_method = 0
    if args_len > 14:
        method_int = Int(args[14].__int__())
    if args_len > 15:
        precond_method = Int(args[15].__int__())
    var enable_early_stopping = False
    var early_stop_patience = 10
    var early_stop_tol = Float32(1e-4)
    if args_len > 16:
        enable_early_stopping = Bool(args[16].__bool__())
    if args_len > 17:
        early_stop_patience = Int(args[17].__int__())
    if args_len > 18:
        early_stop_tol = Float32(Float64(args[18]))
    var progress_callback: PythonObject = None
    var progress_interval = 1
    var progress_enabled = False
    if args_len > 19:
        progress_callback = args[19]
        progress_enabled = True
    if args_len > 20:
        progress_interval = Int(args[20].__int__())

    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1

    var n = Int(y_np.shape[0])
    var num_cat_vars = Int(cat_data_np.shape[1])
    var ctx = DeviceContext()

    # Build level and kernel type lists
    var levels_list = List[Int]()
    var ktypes_list = List[Int]()
    for v in range(num_cat_vars):
        var spec = cat_specs_py[v]
        var levels = Int(spec["levels"].__int__())
        var ktype_str = String(spec["kernel_type"])
        var ktype = CAT_KERNEL_GD
        if ktype_str == "gd":
            ktype = CAT_KERNEL_GD
        elif ktype_str == "cr":
            ktype = CAT_KERNEL_CR
        elif ktype_str == "ehh":
            ktype = CAT_KERNEL_EHH
        elif ktype_str == "hh":
            ktype = CAT_KERNEL_HH
        elif ktype_str == "fe":
            ktype = CAT_KERNEL_FE
        levels_list.append(levels)
        ktypes_list.append(ktype)

    # Create CategoricalCorrelationState
    var cat_state = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n)
    var total_cat_params = cat_state.total_cat_params

    # Upload categorical data (variable-major: [cv * n + i])
    var cat_data_c = np.ascontiguousarray(cat_data_np, dtype=np.int32)
    var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](n * num_cat_vars)
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n):
            cat_data_host.unsafe_ptr()[v * n + i] = Int32(Int(cat_data_c[i][v].__int__()))
    cat_state.upload_categorical_data(cat_data_host)

    # Initialize categorical params
    var cat_params_c = np.ascontiguousarray(cat_params_np, dtype=np.float32).flatten()
    var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    ctx.synchronize()
    for i in range(total_cat_params):
        cat_params_host.unsafe_ptr()[i] = Float32(Float64(cat_params_c[i]))
    cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())

    # Extract provider info
    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n_prov = Int(info["n"].__int__())
    var num_gp = Int(info["num_gradient_params"].__int__())
    var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
    var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
    var sup_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())

    # Build provider with mixed fn-ptrs
    var cont_params_c = np.ascontiguousarray(cont_params_np, dtype=np.float32).flatten()
    var provider = ErasedJITProvider(
        provider_ptr=provider_ptr, ctx=ctx, n=n_prov,
        x_ptr=_cvt_xptr(x_ptr_addr),
        num_gradient_params=num_gp,
        supports_fused_gradient=sup_fused,
        supports_fused_ls_os=sup_lsos,
        supports_fused_3param=sup_3p,
        forward_matvec=_cvt_fwd(Int(info["forward_matvec"].__int__())),
        gradient_matvec=_cvt_grad(Int(info["gradient_matvec"].__int__())),
        fused_gradient_matvec=_cvt_fused(Int(info["fused_gradient_matvec"].__int__())),
        fused_ls_os_gradient_matvec=_cvt_lsos(Int(info["fused_ls_os_gradient_matvec"].__int__())),
        fused_3param_gradient_matvec=_cvt_3p(Int(info["fused_3param_gradient_matvec"].__int__())),
        extract_diagonal=_cvt_diag(Int(info["extract_diagonal"].__int__())),
        update_params=_cvt_upd(Int(info["update_params"].__int__())),
        update_noise=_cvt_unoise(Int(info["update_noise"].__int__())),
        get_noise=_cvt_getf(Int(info["get_noise"].__int__())),
        get_diagonal_value=_cvt_getf(Int(info["get_diagonal_value"].__int__())),
        cross_matvec=get_noop_cross(),
        extract_diagonal_test=get_noop_diagtest(),
        has_prediction=False,
        kronecker_forward_matvec=get_noop_kron_fwd(),
        kronecker_gradient_matvec=get_noop_kron_grad(),
        has_kronecker=False,
        mixed_forward_matvec=_cvt_mixed_fwd(Int(info["mixed_forward_matvec"].__int__())),
        mixed_fused_gradient_matvec=_cvt_mixed_fused_grad(Int(info["mixed_fused_gradient_matvec"].__int__())),
        mixed_cross_matvec=_cvt_mixed_cross(Int(info["mixed_cross_matvec"].__int__())),
        mixed_extract_diagonal=_cvt_mixed_diag(Int(info["mixed_extract_diagonal"].__int__())),
        mixed_materialize=_cvt_mixed_mat(Int(info["mixed_materialize"].__int__())),
        has_mixed=True,
    )

    # Set initial continuous params
    var cont_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(num_gp, 1))
    ctx.synchronize()
    for i in range(num_gp):
        cont_params_host.unsafe_ptr()[i] = Float32(Float64(cont_params_c[i]))
    provider.update_params(cont_params_host.unsafe_ptr())
    provider.update_noise(init_noise)

    # Build y host buffer
    var y_c = np.ascontiguousarray(y_np, dtype=np.float32)
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    ctx.synchronize()
    bulk_copy_to_host_buffer(y_c, y_host, n)

    # Train via train_mixed_jit (split Adam: cont params via BBMM, cat via post-BBMM)
    var result = train_mixed_jit(
        provider^, cat_state^, ctx,
        y_host.unsafe_ptr(), n,
        num_cont_params=num_gp,
        initial_cont_params_ptr=cont_params_host.unsafe_ptr(),
        initial_noise=init_noise,
        initial_cat_params_ptr=cat_params_host.unsafe_ptr(),
        total_cat_params=total_cat_params,
        max_iterations=max_iters,
        learning_rate=lr,
        num_probes=num_probes,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        precond_method=precond_method,
        verbose=verbose,
        use_materialized=use_materialized,
        enable_early_stopping=enable_early_stopping,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        progress_enabled=progress_enabled,
    )

    # Build return dict
    var out = Python.dict()
    out["status"] = "trained"
    out["num_cat_vars"] = num_cat_vars
    out["total_cat_params"] = total_cat_params
    out["num_cont_params"] = result.num_cont_params
    out["nll"] = Float64(result.final_nll)
    out["noise"] = Float64(result.noise)
    out["mean"] = Float64(result.mean)
    out["iterations"] = result.iterations
    out["converged"] = result.converged
    out["training_route"] = "materialized" if use_materialized else "matrix_free"
    out["materialization_mode"] = method_int
    out["precond_method"] = precond_method
    out["precond_rank"] = precond_rank
    out["enable_early_stopping"] = enable_early_stopping
    out["early_stop_patience"] = early_stop_patience
    out["early_stop_tol"] = Float64(early_stop_tol)

    var params_list = Python.list()
    for i in range(result.num_cont_params):
        params_list.append(Float64(result.final_params[i]))
    out["params"] = params_list

    # Trained categorical params (unconstrained)
    var cat_out = np.zeros(total_cat_params, dtype=np.float32)
    for k in range(total_cat_params):
        cat_out[k] = Float64(result.cat_params[k])
    out["cat_params"] = cat_out

    # Iteration timing
    var iter_times_list = Python.list()
    for i in range(len(result.iter_times_ns)):
        iter_times_list.append(Float64(result.iter_times_ns[i]) / 1e6)
    out["iter_times_ms"] = iter_times_list

    var iter_times_ns_list = Python.list()
    for i in range(len(result.iter_times_ns)):
        iter_times_ns_list.append(result.iter_times_ns[i])
    out["iter_times_ns"] = iter_times_ns_list

    # NLL history
    var nll_hist_list = Python.list()
    for i in range(len(result.nll_history)):
        nll_hist_list.append(Float64(result.nll_history[i]))
    out["nll_history"] = nll_hist_list

    var cg_hist_list = Python.list()
    for i in range(len(result.cg_iterations_history)):
        cg_hist_list.append(result.cg_iterations_history[i])
    out["cg_iterations_history"] = cg_hist_list

    _ = cat_data_host
    _ = cat_params_host
    _ = cont_params_host
    _ = y_host

    return out


# =============================================================================
# Python binding: predict_mixed (continuous + categorical)
# =============================================================================

fn predict_mixed_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Predict with a trained mixed continuous+categorical GP.

    Args:
        args[0]: provider_info dict (with mixed fn-ptrs from kernel .so)
        args[1]: y_train numpy [n_train] float32
        args[2]: x_test numpy [n_test x dim] float32
        args[3]: cat_train numpy [n_train x num_cat_vars] int32
        args[4]: cat_test numpy [n_test x num_cat_vars] int32
        args[5]: cont_params numpy [num_cont_params] float32 (trained)
        args[6]: noise float (trained)
        args[7]: mean float (trained)
        args[8]: cat_params numpy [total_cat_params] float32 (trained, unconstrained)
        args[9]: cat_specs list of dicts {"levels": int, "kernel_type": str}
        args[10]: variance_method int (0=mean_only, 1=love, 2=exact), default 1
        args[11]: max_cg_iter int, default 100
        args[12]: cg_tol float, default 1e-2
        args[13]: precond_rank int, default 10
        args[14]: lanczos_rank int, default auto
        args[15]: method int (0=matrix_free, 1=materialized, 3=auto)
    """
    var np = Python.import_module("numpy")

    var info = args[0]
    var y_np = args[1]
    var x_test_np = args[2]
    var cat_train_np = args[3]
    var cat_test_np = args[4]
    var cont_params_np = args[5]
    var noise = Float32(Float64(args[6]))
    var mean = Float32(Float64(args[7]))
    var cat_params_np = args[8]
    var cat_specs_py = args[9]

    var variance_method = PREDICT_LOVE
    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var lanczos_rank = 0
    var method_int = 0
    var args_len = len(args)
    if args_len > 10:
        variance_method = Int(args[10].__int__())
    if args_len > 11:
        max_cg_iter = Int(args[11].__int__())
    if args_len > 12:
        cg_tol = Float32(Float64(args[12]))
    if args_len > 13:
        precond_rank = Int(args[13].__int__())
    if args_len > 14:
        lanczos_rank = Int(args[14].__int__())
    if args_len > 15:
        method_int = Int(args[15].__int__())

    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1
    lanczos_rank = _resolve_exact_predict_lanczos_rank(
        info, lanczos_rank, True, use_materialized,
    )

    var n_train = Int(y_np.shape[0])
    var n_test = Int(x_test_np.shape[0])
    var num_cat_vars = Int(cat_train_np.shape[1])
    var ctx = DeviceContext()

    # Build level and kernel type lists
    var levels_list = List[Int]()
    var ktypes_list = List[Int]()
    for v in range(num_cat_vars):
        var spec = cat_specs_py[v]
        var levels = Int(spec["levels"].__int__())
        var ktype_str = String(spec["kernel_type"])
        var ktype = CAT_KERNEL_GD
        if ktype_str == "gd":
            ktype = CAT_KERNEL_GD
        elif ktype_str == "cr":
            ktype = CAT_KERNEL_CR
        elif ktype_str == "ehh":
            ktype = CAT_KERNEL_EHH
        elif ktype_str == "hh":
            ktype = CAT_KERNEL_HH
        elif ktype_str == "fe":
            ktype = CAT_KERNEL_FE
        levels_list.append(levels)
        ktypes_list.append(ktype)

    # Rebuild CategoricalCorrelationState with training data
    var cat_state = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n_train)
    var total_cat_params = cat_state.total_cat_params

    var cat_train_c = np.ascontiguousarray(cat_train_np, dtype=np.int32)
    var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](n_train * num_cat_vars)
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n_train):
            cat_data_host.unsafe_ptr()[v * n_train + i] = Int32(Int(cat_train_c[i][v].__int__()))
    cat_state.upload_categorical_data(cat_data_host)

    # Update correlation matrices from trained cat_params
    var cat_params_c = np.ascontiguousarray(cat_params_np, dtype=np.float32).flatten()
    var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    ctx.synchronize()
    for i in range(total_cat_params):
        cat_params_host.unsafe_ptr()[i] = Float32(Float64(cat_params_c[i]))
    cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())

    # Build provider with mixed fn-ptrs and trained cont params
    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n_prov = Int(info["n"].__int__())
    var num_gp = Int(info["num_gradient_params"].__int__())
    var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
    var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
    var sup_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())

    var cont_params_c = np.ascontiguousarray(cont_params_np, dtype=np.float32).flatten()
    var provider = ErasedJITProvider(
        provider_ptr=provider_ptr, ctx=ctx, n=n_prov,
        x_ptr=_cvt_xptr(x_ptr_addr),
        num_gradient_params=num_gp,
        supports_fused_gradient=sup_fused,
        supports_fused_ls_os=sup_lsos,
        supports_fused_3param=sup_3p,
        forward_matvec=_cvt_fwd(Int(info["forward_matvec"].__int__())),
        gradient_matvec=_cvt_grad(Int(info["gradient_matvec"].__int__())),
        fused_gradient_matvec=_cvt_fused(Int(info["fused_gradient_matvec"].__int__())),
        fused_ls_os_gradient_matvec=_cvt_lsos(Int(info["fused_ls_os_gradient_matvec"].__int__())),
        fused_3param_gradient_matvec=_cvt_3p(Int(info["fused_3param_gradient_matvec"].__int__())),
        extract_diagonal=_cvt_diag(Int(info["extract_diagonal"].__int__())),
        update_params=_cvt_upd(Int(info["update_params"].__int__())),
        update_noise=_cvt_unoise(Int(info["update_noise"].__int__())),
        get_noise=_cvt_getf(Int(info["get_noise"].__int__())),
        get_diagonal_value=_cvt_getf(Int(info["get_diagonal_value"].__int__())),
        cross_matvec=get_noop_cross(),
        extract_diagonal_test=get_noop_diagtest(),
        has_prediction=False,
        kronecker_forward_matvec=get_noop_kron_fwd(),
        kronecker_gradient_matvec=get_noop_kron_grad(),
        has_kronecker=False,
        mixed_forward_matvec=_cvt_mixed_fwd(Int(info["mixed_forward_matvec"].__int__())),
        mixed_fused_gradient_matvec=_cvt_mixed_fused_grad(Int(info["mixed_fused_gradient_matvec"].__int__())),
        mixed_cross_matvec=_cvt_mixed_cross(Int(info["mixed_cross_matvec"].__int__())),
        mixed_extract_diagonal=_cvt_mixed_diag(Int(info["mixed_extract_diagonal"].__int__())),
        mixed_materialize=_cvt_mixed_mat(Int(info["mixed_materialize"].__int__())),
        has_mixed=True,
    )

    # Set trained cont params and noise
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](max(num_gp, 1))
    ctx.synchronize()
    for i in range(num_gp):
        params_host.unsafe_ptr()[i] = Float32(Float64(cont_params_c[i]))
    provider.update_params(params_host.unsafe_ptr())
    provider.update_noise(noise)

    # Upload y_train to GPU
    var y_c = np.ascontiguousarray(y_np, dtype=np.float32)
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](n_train)
    ctx.synchronize()
    bulk_copy_to_host_buffer(y_c, y_host, n_train)
    var y_device = ctx.enqueue_create_buffer[float_dtype](n_train)
    ctx.enqueue_copy(y_device, y_host)

    # Upload x_test to GPU
    var dim = Int(x_test_np.shape[1])
    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)

    # Upload cat_test to GPU (variable-major: [cv * n_test + i])
    var cat_test_c = np.ascontiguousarray(cat_test_np, dtype=np.int32)
    var cat_test_host = ctx.enqueue_create_host_buffer[DType.int32](n_test * num_cat_vars)
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n_test):
            cat_test_host.unsafe_ptr()[v * n_test + i] = Int32(Int(cat_test_c[i][v].__int__()))
    var cat_test_device = ctx.enqueue_create_buffer[DType.int32](n_test * num_cat_vars)
    ctx.enqueue_copy(cat_test_device, cat_test_host)
    ctx.synchronize()

    # Predict
    var pred = predict_mixed_jit(
        provider^, cat_state^, ctx,
        y_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        cat_test_device.unsafe_ptr(),
        n_train, n_test, dim, noise, mean,
        variance_method,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        lanczos_rank=lanczos_rank,
        use_materialized=use_materialized,
    )

    # Build return dict
    var out = Python.dict()
    var mean_arr = np.zeros(n_test, dtype=np.float32)
    for i in range(n_test):
        mean_arr[i] = Float64(pred.mean.unsafe_ptr()[i])
    out["mean"] = mean_arr
    out["lanczos_rank_used"] = lanczos_rank

    if pred.has_variance:
        var var_arr = np.zeros(n_test, dtype=np.float32)
        for i in range(n_test):
            var_arr[i] = Float64(pred.variance.unsafe_ptr()[i])
        out["variance"] = var_arr
        out["std"] = np.sqrt(var_arr)
    else:
        out["variance"] = np.zeros(n_test, dtype=np.float32)
        out["std"] = np.zeros(n_test, dtype=np.float32)

    _ = cat_data_host
    _ = cat_params_host
    _ = params_host
    _ = y_host
    _ = y_device
    _ = x_test_host
    _ = x_test_device
    _ = cat_test_host
    _ = cat_test_device

    return out


# =============================================================================
# Module Initialization
# =============================================================================

@export
fn PyInit_mojogp_jit_engine() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojogp_jit_engine")
        m.def_py_function[train_python]("train", docstring="Train GP with JIT kernel fn ptrs")
        m.def_py_function[benchmark_provider_ops_python]("benchmark_provider_ops", docstring="Benchmark generated provider matvec ops in isolation")
        m.def_py_function[prepare_prediction_cache_python]("prepare_prediction_cache", docstring="Prepare device-resident single-output prediction cache")
        m.def_py_function[predict_with_cache_python]("predict_with_cache", docstring="Predict with a device-resident single-output prediction cache")
        m.def_py_function[destroy_prediction_cache_python]("destroy_prediction_cache", docstring="Destroy a device-resident single-output prediction cache")
        m.def_py_function[predict_python]("predict", docstring="Predict with JIT kernel fn ptrs")
        m.def_py_function[train_multi_output_python]("train_multi_output", docstring="Train multi-output GP with Kronecker/ICM")
        m.def_py_function[predict_multi_output_python]("predict_multi_output", docstring="Predict mean for multi-output GP with Kronecker/ICM")
        m.def_py_function[sample_multi_output_pathwise_python]("sample_multi_output_pathwise", docstring="Compute one backend correction sample for MultiOutputGP pathwise sampling")
        m.def_py_function[train_multi_output_mixed_python]("train_multi_output_mixed", docstring="Train mixed multi-output GP with Kronecker/ICM")
        m.def_py_function[sample_multi_output_mixed_pathwise_python]("sample_multi_output_mixed_pathwise", docstring="Compute one backend correction sample for mixed MultiOutputGP pathwise sampling")
        m.def_py_function[predict_multi_output_mixed_python]("predict_multi_output_mixed", docstring="Predict with trained mixed multi-output GP")
        m.def_py_function[train_mixed_python]("train_mixed", docstring="Train mixed continuous+categorical GP")
        m.def_py_function[predict_mixed_python]("predict_mixed", docstring="Predict with trained mixed continuous+categorical GP")
        m.def_py_function[train_lmc_mixed_python]("train_lmc_mixed", docstring="Train mixed LMC multi-output GP with per-latent categorical state")
        m.def_py_function[train_lmc_python]("train_lmc", docstring="Train LMC multi-output GP with R latent kernels")
        m.def_py_function[sample_lmc_pathwise_python]("sample_lmc_pathwise", docstring="Compute one backend correction sample for MultiOutputLMCGP pathwise sampling")
        m.def_py_function[sample_lmc_mixed_pathwise_python]("sample_lmc_mixed_pathwise", docstring="Compute one backend correction sample for mixed MultiOutputLMCGP pathwise sampling")
        m.def_py_function[predict_lmc_python]("predict_lmc", docstring="Predict mean for LMC multi-output GP using GPU cross-matvec per latent")
        m.def_py_function[predict_lmc_mixed_python]("predict_lmc_mixed", docstring="Predict with trained mixed LMC multi-output GP using GPU mixed cross-matvec per latent")
        return m.finalize()
    except e:
        return abort[PythonObject]("Failed to init mojogp_jit_engine: " + String(e))
