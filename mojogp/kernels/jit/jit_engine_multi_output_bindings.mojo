from python import PythonObject, Python
from gpu.host import DeviceContext, DeviceBuffer
from math import isnan, isinf

from kernels.jit.erased_provider import (
    ErasedJITProvider,
    _cvt_fwd,
    _cvt_grad,
    _cvt_fused,
    _cvt_lsos,
    _cvt_3p,
    _cvt_diag,
    _cvt_upd,
    _cvt_unoise,
    _cvt_getf,
    _cvt_xptr,
    _cvt_cross,
    _cvt_diagtest,
    _cvt_kron_fwd,
    _cvt_kron_grad,
    get_noop_cross,
    get_noop_diagtest,
    get_noop_kron_fwd,
    get_noop_kron_grad,
    _cvt_mixed_fwd,
    _cvt_mixed_fused_grad,
    _cvt_mixed_cross,
    _cvt_mixed_diag,
    _cvt_mixed_mat,
)
from kernels.jit.jit_prediction import (
    predict_variance_love_jit,
    predict_variance_exact_jit,
    solve_single_rhs_deterministic_host_jit,
    PREDICT_MEAN_ONLY,
    PREDICT_LOVE,
    PREDICT_EXACT,
)
from kernels.jit.jit_multi_output import train_multi_output_jit
from kernels.jit.fused_kronecker_provider import FusedKroneckerProvider
from kernels.kronecker_direct_provider import KroneckerDirectProvider
from kernels.jit.jit_multi_output_mixed import (
    train_multi_output_mixed_jit,
    MixedKroneckerBaseProviderView,
    predict_variance_mixed_jit,
)
from kernels.py_conversion import bulk_copy_to_host_buffer, py_to_f32
from kernels.constants import float_dtype, CAT_KERNEL_GD, CAT_KERNEL_CR, CAT_KERNEL_EHH, CAT_KERNEL_HH, CAT_KERNEL_FE
from kernels.categorical_state import CategoricalCorrelationState
from kernels.jit.jit_engine_binding_helpers import (
    _info_bool,
    _info_materialization_mode,
    _resolve_multi_predict_lanczos_rank,
)


fn train_multi_output_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Train multi-output GP with Kronecker/ICM structure.

    Args (from Python):
        args[0]: provider_info dict (base kernel fn ptrs)
        args[1]: Y numpy array (n, T) float32 — multi-output targets
        args[2]: initial_params numpy array [num_params] float32
        args[3]: trainable mask numpy array [num_params] bool
        args[4]: initial_noise_per_task numpy array [T] float32
        args[5]: initial_outputscale (float)
        args[6]: initial_mean_per_task numpy array [T] float32
        args[7]: num_tasks (int)
        args[8]: max_iterations (int, default=100)
        args[9]: learning_rate (float, default=0.05)
        args[10]: task_rank (int, default=-1 for full rank)
        args[11]: verbose (bool, default=False)
        args[12]: num_probes (int, default=10)
        args[13]: max_cg_iter (int, default=200)
        args[14]: cg_tol (float, default=1.0)
        args[15]: precond_rank (int, default=15)
        args[16]: precond_method (int, default=0)
        args[17]: precond_rebuild_threshold (float, default=0.5)
        args[18]: max_tridiag_iter (int, default=30)
        args[19]: early_stop_patience (int, default=15)
        args[20]: early_stop_tol (float, default=1e-4, 0.0 disables early stopping)
        args[21]: use_cosine_lr (bool, default=False)
        args[22]: optional fixed observation noise [n, T] float32
    """
    if len(args) < 21 or len(args) > 25:
        raise Error("train_multi_output() expects 21 to 25 positional arguments")

    var np = Python.import_module("numpy")

    var info = args[0]
    var Y_np = args[1]
    var params_np = args[2]
    var trainable_mask_np = args[3]
    var noise_per_task_np = args[4]
    var init_outputscale = py_to_f32(args[5])
    var mean_per_task_np = args[6]
    var num_tasks = Int(args[7].__int__())

    var max_iterations = 100
    var learning_rate = Float32(0.05)
    var task_rank = -1
    var verbose = False
    var num_probes = 10
    var max_cg_iter = 200
    var cg_tol = Float32(1e-2)
    var precond_rank = 15
    var early_stop_patience = 15
    var early_stop_tol = Float32(1e-4)
    max_iterations = Int(args[8].__int__())
    learning_rate = py_to_f32(args[9])
    task_rank = Int(args[10].__int__())
    verbose = Bool(args[11].__bool__())
    num_probes = Int(args[12].__int__())
    max_cg_iter = Int(args[13].__int__())
    cg_tol = py_to_f32(args[14])
    precond_rank = Int(args[15].__int__())
    var precond_method = Int(args[16].__int__())
    var precond_rebuild_threshold = py_to_f32(args[17])
    var max_tridiag_iter = Int(args[18].__int__())
    early_stop_patience = Int(args[19].__int__())
    early_stop_tol = py_to_f32(args[20])
    var use_cosine_lr = False
    if len(args) >= 22:
        use_cosine_lr = Bool(args[21].__bool__())
    var progress_callback: PythonObject = None
    var progress_interval = 1
    var progress_enabled = False
    if len(args) > 23:
        progress_callback = args[23]
        progress_enabled = True
    if len(args) > 24:
        progress_interval = Int(args[24].__int__())

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
    var get_diag_val_ptr = Int(info["get_diagonal_value"].__int__())

    var ctx = DeviceContext()
    var base_provider = ErasedJITProvider(
        provider_ptr=provider_ptr, ctx=ctx, n=n,
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
        get_diagonal_value=_cvt_getf(get_diag_val_ptr),
        cross_matvec=get_noop_cross(),
        extract_diagonal_test=get_noop_diagtest(),
        has_prediction=False,
        kronecker_forward_matvec=_cvt_kron_fwd(Int(info["kronecker_forward_matvec"].__int__())) if Bool(info.__contains__("kronecker_forward_matvec")) else get_noop_kron_fwd(),
            kronecker_gradient_matvec=_cvt_kron_grad(Int(info["kronecker_gradient_matvec"].__int__())) if Bool(info.__contains__("kronecker_gradient_matvec")) else get_noop_kron_grad(),
            has_kronecker=Bool(info.__contains__("kronecker_forward_matvec")),
    )

    var T = num_tasks
    var nT = n * T
    var Y_c = np.ascontiguousarray(Y_np, dtype=np.float32)
    var y_blocked_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n):
            y_blocked_host.unsafe_ptr()[t * n + i] = py_to_f32(Y_c[i][t])
    var y_blocked_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.enqueue_copy(y_blocked_device, y_blocked_host)
    ctx.synchronize()

    var params_c = np.ascontiguousarray(params_np, dtype=np.float32).flatten()
    var trainable_mask_c = np.ascontiguousarray(trainable_mask_np, dtype=np.bool_).flatten()
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_gradient_params)
    var trainable_mask = alloc[Bool](max(num_gradient_params, 1))
    ctx.synchronize()
    if Int(trainable_mask_c.shape[0]) != num_gradient_params:
        raise Error("train_multi_output() trainable mask length does not match provider gradient params")
    bulk_copy_to_host_buffer(params_c, params_host, num_gradient_params)
    for p in range(num_gradient_params):
        trainable_mask[p] = Bool(trainable_mask_c[p].__bool__())

    var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    if noise_per_task_np is None:
        for t in range(T):
            noise_host.unsafe_ptr()[t] = Float32(0.1)
    else:
        var noise_c = np.ascontiguousarray(noise_per_task_np, dtype=np.float32).flatten()
        bulk_copy_to_host_buffer(noise_c, noise_host, T)

    var mean_c = np.ascontiguousarray(mean_per_task_np, dtype=np.float32).flatten()
    var mean_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(mean_c, mean_host, T)

    var noise_mode = 0
    var fixed_noise_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for idx in range(nT):
        fixed_noise_host.unsafe_ptr()[idx] = Float32(0)
    if len(args) >= 23 and args[22] is not None:
        noise_mode = 1
        var fixed_noise_c = np.ascontiguousarray(args[22], dtype=np.float32).T.flatten()
        bulk_copy_to_host_buffer(fixed_noise_c, fixed_noise_host, nT)

    var result = train_multi_output_jit(
        base_provider, ctx,
        y_blocked_device.unsafe_ptr(),
        n, T, num_gradient_params,
        params_host.unsafe_ptr(),
        trainable_mask,
        noise_host.unsafe_ptr(),
        init_outputscale,
        mean_host.unsafe_ptr(),
        fixed_noise_host.unsafe_ptr(),
        noise_mode,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        task_rank=task_rank,
        num_probes=num_probes,
        max_cg_iter=max_cg_iter,
        max_tridiag_iter=max_tridiag_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        precond_method=precond_method,
        precond_rebuild_threshold=precond_rebuild_threshold,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
        verbose=verbose,
        use_cosine_lr=use_cosine_lr,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        progress_enabled=progress_enabled,
    )

    var out = Python.dict()
    out["final_nll"] = Float64(result.final_nll)
    out["outputscale"] = Float64(result.outputscale)
    out["iterations"] = result.iterations
    out["converged"] = result.converged
    out["num_tasks"] = result.num_tasks
    out["task_rank"] = result.task_rank
    out["max_tridiag_iter"] = max_tridiag_iter
    out["precond_rebuild_threshold"] = precond_rebuild_threshold
    out["precond_rebuild_count"] = result.precond_rebuild_count
    out["precond_rank"] = precond_rank
    out["precond_method"] = precond_method
    out["early_stop_patience"] = early_stop_patience
    out["early_stop_tol"] = Float64(early_stop_tol)
    out["use_cosine_lr"] = use_cosine_lr
    out["training_route"] = (
        "materialized" if _info_materialization_mode(info) == 1 else "matrix_free"
    )
    out["materialization_mode"] = _info_materialization_mode(info)
    out["noise_mode"] = "fixed_per_sample_task" if noise_mode == 1 else "per_task"
    out["has_observation_noise_vector"] = noise_mode == 1

    var cg_history = Python.list()
    for i in range(len(result.cg_iterations_history)):
        cg_history.append(result.cg_iterations_history[i])
    out["cg_iterations_history"] = cg_history

    var iter_times_list = Python.list()
    for i in range(len(result.iter_times_ns)):
        iter_times_list.append(Float64(result.iter_times_ns[i]) / 1e6)
    out["iter_times_ms"] = iter_times_list

    var nll_history = Python.list()
    for i in range(len(result.nll_history)):
        nll_history.append(Float64(result.nll_history[i]))
    out["nll_history"] = nll_history

    var params_list = Python.list()
    for p in range(result.num_kernel_params):
        params_list.append(Float64(result.final_params[p]))
    out["params"] = params_list

    var noise_list = Python.list()
    for t in range(T):
        noise_list.append(Float64(result.noise_per_task[t]))
    out["noise_per_task"] = noise_list

    var mean_list = Python.list()
    for t in range(T):
        mean_list.append(Float64(result.mean_per_task[t]))
    out["mean_per_task"] = mean_list

    var B_list = Python.list()
    for i in range(T * T):
        B_list.append(Float64(result.B_flat[i]))
    out["B_flat"] = B_list

    var alpha_np = np.zeros(nT, dtype=np.float32)
    for i in range(nT):
        alpha_np[i] = Float64(result.alpha_blocked[i])
    out["alpha"] = alpha_np

    _ = y_blocked_host
    _ = y_blocked_device
    _ = params_host
    trainable_mask.free()
    _ = noise_host
    _ = fixed_noise_host
    _ = mean_host

    return out


fn predict_multi_output_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Predict with Kronecker/ICM multi-output GP using rotated posterior state.

    Args:
        args[0]: provider_info dict
        args[1]: alpha_rotated numpy array [n, T] float32
        args[2]: Q numpy array [T, T] float32
        args[3]: effective_scales numpy array [T] float32
        args[4]: X_test numpy array [m, d] float32
        args[5]: params numpy array [num_params] float32
        args[6]: noise float
        args[7]: variance_method int (0=mean_only, 1=love, 2=exact)
        args[8]: max_cg_iter int (optional, default 100)
        args[9]: cg_tol float (optional, default 1e-2)
        args[10]: precond_rank int (optional, default 10)
        args[11]: lanczos_rank int (optional, default auto)

    Returns:
        dict with: mean [m, T] numpy float32, optional variance/std [m, T]
    """
    if len(args) < 8 or len(args) > 12:
        raise Error("predict_multi_output() expects 8-12 positional arguments")

    var np = Python.import_module("numpy")

    var info = args[0]
    var alpha_rotated_np = args[1]
    var Q_np = args[2]
    var effective_scales_np = args[3]
    var x_test_np = args[4]
    var params_np = args[5]
    var noise = py_to_f32(args[6])
    var variance_method = Int(args[7].__int__())

    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var lanczos_rank = 0
    var args_len = len(args)
    if args_len > 8:
        max_cg_iter = Int(args[8].__int__())
    if args_len > 9:
        cg_tol = py_to_f32(args[9])
    if args_len > 10:
        precond_rank = Int(args[10].__int__())
    if args_len > 11:
        lanczos_rank = Int(args[11].__int__())

    var n = Int(alpha_rotated_np.shape[0])
    var T = Int(alpha_rotated_np.shape[1])
    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])

    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n_prov = Int(info["n"].__int__())
    var num_gp = Int(info["num_gradient_params"].__int__())
    var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
    var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
    var sup_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())
    var has_cross = Bool(info.__contains__("cross_matvec"))
    var cross_ptr = Int(info["cross_matvec"].__int__()) if has_cross else 0
    var has_diagtest = Bool(info.__contains__("extract_diagonal_test"))
    var diagtest_ptr = Int(info["extract_diagonal_test"].__int__()) if has_diagtest else 0
    var can_compute_variance = variance_method != PREDICT_MEAN_ONLY and has_diagtest
    var use_materialized = _info_materialization_mode(info) != 0
    lanczos_rank = _resolve_multi_predict_lanczos_rank(
        lanczos_rank, False, use_materialized, _info_bool(info, "is_ard"),
    )

    var ctx = DeviceContext()
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
        cross_matvec=_cvt_cross(cross_ptr) if has_cross else get_noop_cross(),
        extract_diagonal_test=_cvt_diagtest(diagtest_ptr) if has_diagtest else get_noop_diagtest(),
        has_prediction=has_cross,
        kronecker_forward_matvec=get_noop_kron_fwd(),
        kronecker_gradient_matvec=get_noop_kron_grad(),
        has_kronecker=False,
    )

    var params_c = np.ascontiguousarray(params_np, dtype=np.float32).ravel()
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_gp)
    ctx.synchronize()
    bulk_copy_to_host_buffer(params_c, params_host, num_gp)
    provider.update_params(params_host.unsafe_ptr())
    provider.update_noise(noise)

    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)
    ctx.synchronize()

    var alpha_rotated_c = np.ascontiguousarray(alpha_rotated_np, dtype=np.float32).ravel()
    var Q_c = np.ascontiguousarray(Q_np, dtype=np.float32).ravel()
    var effective_scales_c = np.ascontiguousarray(effective_scales_np, dtype=np.float32).ravel()

    var mean_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var variance_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var alpha_s_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var alpha_s_device = ctx.enqueue_create_buffer[float_dtype](n)
    var cross_buffer = ctx.enqueue_create_buffer[float_dtype](n_test)
    var cross_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    var var_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    ctx.synchronize()

    for s in range(T):
        for i in range(n):
            alpha_s_host.unsafe_ptr()[i] = py_to_f32(alpha_rotated_c[i * T + s])
        ctx.enqueue_copy(alpha_s_device, alpha_s_host)
        ctx.synchronize()

        provider.cross_matvec(
            cross_buffer.unsafe_ptr(),
            x_test_device.unsafe_ptr(),
            alpha_s_device.unsafe_ptr(),
            n_test,
            1,
        )
        ctx.enqueue_copy(cross_host, cross_buffer)
        ctx.synchronize()

        var scale_s = py_to_f32(effective_scales_c[s])
        for t in range(T):
            var q_ts = py_to_f32(Q_c[t * T + s])
            for i in range(n_test):
                var current_mean = py_to_f32(mean_out_flat[i * T + t])
                mean_out_flat[i * T + t] = current_mean + q_ts * scale_s * cross_host.unsafe_ptr()[i]

        if can_compute_variance and scale_s > Float32(1e-8):
            provider.update_noise(noise / scale_s)

            var latent_var_device: DeviceBuffer[float_dtype]
            if variance_method == PREDICT_LOVE:
                latent_var_device = predict_variance_love_jit(
                    provider,
                    provider,
                    False,
                    ctx,
                    x_test_device.unsafe_ptr(),
                    n,
                    n_test,
                    max_cg_iter,
                    cg_tol,
                    precond_rank,
                    lanczos_rank,
                )
            else:
                var exact_block_cols_slot = alloc[Int](1)
                var exact_cross_mode_slot = alloc[Int](1)
                exact_block_cols_slot[] = 0
                exact_cross_mode_slot[] = 0
                latent_var_device = predict_variance_exact_jit(
                    provider,
                    provider,
                    False,
                    ctx,
                    alpha_s_device.unsafe_ptr(),
                    x_test_device.unsafe_ptr(),
                    n,
                    n_test,
                    dim,
                    max_cg_iter,
                    cg_tol,
                    precond_rank,
                    precond_rank > 0,
                    False,
                    0,
                    exact_block_cols_slot,
                    exact_cross_mode_slot,
                )
                exact_block_cols_slot.free()
                exact_cross_mode_slot.free()

            ctx.enqueue_copy(var_host, latent_var_device)
            ctx.synchronize()

            for t in range(T):
                var q_ts = py_to_f32(Q_c[t * T + s])
                var weight = q_ts * q_ts * scale_s
                for i in range(n_test):
                    var current_var = py_to_f32(variance_out_flat[i * T + t])
                    variance_out_flat[i * T + t] = current_var + weight * var_host.unsafe_ptr()[i]

            _ = latent_var_device

    var out = Python.dict()
    out["mean"] = mean_out_flat.reshape(n_test, T)
    out["lanczos_rank_used"] = lanczos_rank
    if can_compute_variance:
        out["variance"] = variance_out_flat.reshape(n_test, T)
        out["std"] = np.sqrt(out["variance"])

    _ = params_host
    _ = x_test_host
    _ = x_test_device
    _ = alpha_s_host
    _ = alpha_s_device
    _ = cross_buffer
    _ = cross_host
    _ = var_host

    return out


fn sample_multi_output_pathwise_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Compute one backend correction sample for MultiOutputGP pathwise sampling.

    Args:
        args[0]: provider_info dict
        args[1]: residual numpy array [n, T] float32
        args[2]: task covariance B numpy array [T, T] float32 (already scaled)
        args[3]: X_test numpy array [m, d] float32
        args[4]: params numpy array [num_params] float32
        args[5]: noise_per_task numpy array [T] float32
        args[6]: max_cg_iter int (optional, default 100)
        args[7]: cg_tol float (optional, default 1e-2)
    """
    if len(args) < 6 or len(args) > 8:
        raise Error("sample_multi_output_pathwise() expects 6-8 positional arguments")

    var np = Python.import_module("numpy")

    var info = args[0]
    var residual_np = args[1]
    var B_np = args[2]
    var x_test_np = args[3]
    var params_np = args[4]
    var noise_per_task_np = args[5]

    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var args_len = len(args)
    if args_len > 6:
        max_cg_iter = Int(args[6].__int__())
    if args_len > 7:
        cg_tol = py_to_f32(args[7])

    var n = Int(residual_np.shape[0])
    var T = Int(residual_np.shape[1])
    var nT = n * T
    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])

    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n_prov = Int(info["n"].__int__())
    var num_gp = Int(info["num_gradient_params"].__int__())
    var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
    var sup_ls_os = Bool(info["supports_fused_ls_os"].__bool__())
    var sup_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())
    var has_cross = Bool(info.__contains__("cross_matvec"))
    if not has_cross:
        raise Error("sample_multi_output_pathwise() requires cross_matvec support")

    var ctx = DeviceContext()
    var base_provider = ErasedJITProvider(
        provider_ptr=provider_ptr, ctx=ctx, n=n_prov,
        x_ptr=_cvt_xptr(x_ptr_addr),
        num_gradient_params=num_gp,
        supports_fused_gradient=sup_fused,
        supports_fused_ls_os=sup_ls_os,
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
        cross_matvec=_cvt_cross(Int(info["cross_matvec"].__int__())),
        extract_diagonal_test=_cvt_diagtest(Int(info["extract_diagonal_test"].__int__())) if Bool(info.__contains__("extract_diagonal_test")) else get_noop_diagtest(),
        has_prediction=True,
        kronecker_forward_matvec=_cvt_kron_fwd(Int(info["kronecker_forward_matvec"].__int__())) if Bool(info.__contains__("kronecker_forward_matvec")) else get_noop_kron_fwd(),
        kronecker_gradient_matvec=_cvt_kron_grad(Int(info["kronecker_gradient_matvec"].__int__())) if Bool(info.__contains__("kronecker_gradient_matvec")) else get_noop_kron_grad(),
        has_kronecker=Bool(info.__contains__("kronecker_forward_matvec")),
    )

    var params_c = np.ascontiguousarray(params_np, dtype=np.float32).ravel()
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_gp)
    ctx.synchronize()
    bulk_copy_to_host_buffer(params_c, params_host, num_gp)
    base_provider.update_params(params_host.unsafe_ptr())
    base_provider.update_noise(Float32(0.0))
    var cross_provider = base_provider.clone()

    var B_c = np.ascontiguousarray(B_np, dtype=np.float32).ravel()
    var B_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(B_c, B_host, T * T)

    var noise_c = np.ascontiguousarray(noise_per_task_np, dtype=np.float32).ravel()
    var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(noise_c, noise_host, T)
    var zero_noise_vector_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for idx in range(nT):
        zero_noise_vector_host.unsafe_ptr()[idx] = Float32(0)

    var kron_provider = FusedKroneckerProvider(
        base_provider^, ctx, T, Float32(1.0), B_host, noise_host,
        0, zero_noise_vector_host,
    )

    var residual_c = np.ascontiguousarray(residual_np, dtype=np.float32)
    var residual_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n):
            residual_host.unsafe_ptr()[t * n + i] = py_to_f32(residual_c[i][t])

    var beta_host = solve_single_rhs_deterministic_host_jit(
        kron_provider, ctx, residual_host.unsafe_ptr(), nT, max_cg_iter, cg_tol,
    )

    var beta_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.enqueue_copy(beta_device, beta_host)

    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)

    var cross_device = ctx.enqueue_create_buffer[float_dtype](n_test * T)
    var cross_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
    ctx.synchronize()
    cross_provider.cross_matvec(
        cross_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        beta_device.unsafe_ptr(),
        n_test,
        T,
    )
    ctx.enqueue_copy(cross_host, cross_device)
    ctx.synchronize()

    var correction_flat = np.zeros(n_test * T, dtype=np.float32)
    for i in range(n_test):
        for t in range(T):
            var val = Float32(0.0)
            for u in range(T):
                val += B_host.unsafe_ptr()[t * T + u] * cross_host.unsafe_ptr()[u * n_test + i]
            correction_flat[i * T + t] = Float64(val)

    var out = Python.dict()
    out["correction"] = correction_flat.reshape(n_test, T)

    _ = params_host
    _ = B_host
    _ = noise_host
    _ = zero_noise_vector_host
    _ = residual_host
    _ = beta_host
    _ = beta_device
    _ = x_test_host
    _ = x_test_device
    _ = cross_device
    _ = cross_host

    return out


fn sample_multi_output_mixed_pathwise_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Compute one backend correction sample for mixed MultiOutputGP pathwise sampling.

    Args:
        args[0]: provider_info dict
        args[1]: residual numpy array [n, T] float32
        args[2]: task covariance B numpy array [T, T] float32 (already scaled)
        args[3]: X_test numpy array [m, d] float32
        args[4]: params numpy array [num_params] float32
        args[5]: noise_per_task numpy array [T] float32
        args[6]: categorical train data [n, c] int32
        args[7]: categorical test data [m, c] int32
        args[8]: categorical params numpy array [num_cat_params] float32
        args[9]: categorical spec list
        args[10]: max_cg_iter int (optional, default 100)
        args[11]: cg_tol float (optional, default 1e-2)
        args[12]: materialization_mode int (optional, default 0)
    """
    if len(args) < 10 or len(args) > 13:
        raise Error("sample_multi_output_mixed_pathwise() expects 10-13 positional arguments")

    var np = Python.import_module("numpy")

    var info = args[0]
    var residual_np = args[1]
    var B_np = args[2]
    var x_test_np = args[3]
    var params_np = args[4]
    var noise_per_task_np = args[5]
    var cat_train_np = args[6]
    var cat_test_np = args[7]
    var cat_params_np = args[8]
    var cat_specs_py = args[9]

    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var method_int = 0
    if len(args) > 10:
        max_cg_iter = Int(args[10].__int__())
    if len(args) > 11:
        cg_tol = py_to_f32(args[11])
    if len(args) > 12:
        method_int = Int(args[12].__int__())

    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1

    var n = Int(residual_np.shape[0])
    var T = Int(residual_np.shape[1])
    var nT = n * T
    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])
    var num_cat_vars = Int(cat_train_np.shape[1])

    var ctx = DeviceContext()

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

    var cat_state = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n)
    var total_cat_params = cat_state.total_cat_params

    var cat_train_c = np.ascontiguousarray(cat_train_np, dtype=np.int32)
    var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](max(n * num_cat_vars, 1))
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n):
            cat_data_host.unsafe_ptr()[v * n + i] = Int32(Int(cat_train_c[i][v].__int__()))
    if num_cat_vars > 0:
        cat_state.upload_categorical_data(cat_data_host)

    var cat_params_c = np.ascontiguousarray(cat_params_np, dtype=np.float32).flatten()
    var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    ctx.synchronize()
    for i in range(total_cat_params):
        cat_params_host.unsafe_ptr()[i] = py_to_f32(cat_params_c[i])
    if total_cat_params > 0:
        cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())

    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n_prov = Int(info["n"].__int__())
    var num_gp = Int(info["num_gradient_params"].__int__())
    var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
    var sup_ls_os = Bool(info["supports_fused_ls_os"].__bool__())
    var sup_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())

    var provider = ErasedJITProvider(
        provider_ptr=provider_ptr, ctx=ctx, n=n_prov,
        x_ptr=_cvt_xptr(x_ptr_addr),
        num_gradient_params=num_gp,
        supports_fused_gradient=sup_fused,
        supports_fused_ls_os=sup_ls_os,
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

    var params_c = np.ascontiguousarray(params_np, dtype=np.float32).ravel()
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](max(num_gp, 1))
    ctx.synchronize()
    for i in range(num_gp):
        params_host.unsafe_ptr()[i] = py_to_f32(params_c[i])
    provider.update_params(params_host.unsafe_ptr())
    provider.update_noise(Float32(0.0))

    var mixed_provider = MixedKroneckerBaseProviderView(provider^, cat_state)
    if use_materialized:
        mixed_provider.refresh_materialization()

    var B_c = np.ascontiguousarray(B_np, dtype=np.float32).ravel()
    var B_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(B_c, B_host, T * T)

    var noise_c = np.ascontiguousarray(noise_per_task_np, dtype=np.float32).ravel()
    var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(noise_c, noise_host, T)

    var kron_provider = KroneckerDirectProvider(
        mixed_provider.copy(), ctx, T, Float32(1.0), B_host, noise_host,
    )

    var residual_c = np.ascontiguousarray(residual_np, dtype=np.float32)
    var residual_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n):
            residual_host.unsafe_ptr()[t * n + i] = py_to_f32(residual_c[i][t])

    var beta_host = solve_single_rhs_deterministic_host_jit(
        kron_provider, ctx, residual_host.unsafe_ptr(), nT, max_cg_iter, cg_tol,
    )
    var beta_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.enqueue_copy(beta_device, beta_host)
    ctx.synchronize()

    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](max(n_test * dim, 1))
    ctx.synchronize()
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](max(n_test * dim, 1))
    ctx.enqueue_copy(x_test_device, x_test_host)

    var cat_test_c = np.ascontiguousarray(cat_test_np, dtype=np.int32)
    var cat_test_host = ctx.enqueue_create_host_buffer[DType.int32](max(n_test * num_cat_vars, 1))
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n_test):
            cat_test_host.unsafe_ptr()[v * n_test + i] = Int32(Int(cat_test_c[i][v].__int__()))
    var cat_test_device = ctx.enqueue_create_buffer[DType.int32](max(n_test * num_cat_vars, 1))
    ctx.enqueue_copy(cat_test_device, cat_test_host)
    ctx.synchronize()

    if use_materialized:
        mixed_provider.refresh_materialization()
    var cross_device = ctx.enqueue_create_buffer[float_dtype](n_test * T)
    var cross_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
    mixed_provider.mixed_cross_matvec(
        cross_device.unsafe_ptr(),
        x_test_device.unsafe_ptr(),
        beta_device.unsafe_ptr(),
        cat_test_device.unsafe_ptr(),
        n_test,
        T,
    )
    ctx.enqueue_copy(cross_host, cross_device)
    ctx.synchronize()

    var correction_flat = np.zeros(n_test * T, dtype=np.float32)
    for i in range(n_test):
        for t in range(T):
            var val = Float32(0.0)
            for u in range(T):
                val += B_host.unsafe_ptr()[t * T + u] * cross_host.unsafe_ptr()[u * n_test + i]
            correction_flat[i * T + t] = Float64(val)

    var out = Python.dict()
    out["correction"] = correction_flat.reshape(n_test, T)

    _ = params_host
    _ = B_host
    _ = noise_host
    _ = residual_host
    _ = beta_host
    _ = beta_device
    _ = x_test_host
    _ = x_test_device
    _ = cat_test_host
    _ = cat_test_device
    _ = cross_device
    _ = cross_host
    _ = cat_data_host
    _ = cat_params_host
    _ = cat_state

    return out


fn train_multi_output_mixed_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Train mixed continuous+categorical multi-output GP."""
    if len(args) < 24 or len(args) > 27:
        raise Error("train_multi_output_mixed() expects 24 to 27 positional arguments")

    var np = Python.import_module("numpy")

    var info = args[0]
    var Y_np = args[1]
    var params_np = args[2]
    var noise_per_task_np = args[3]
    var init_outputscale = py_to_f32(args[4])
    var mean_per_task_np = args[5]
    var num_tasks = Int(args[6].__int__())
    var cat_data_np = args[7]
    var cat_specs_py = args[8]
    var cat_params_np = args[9]

    var max_iterations = 100
    var learning_rate = Float32(0.05)
    var task_rank = -1
    var verbose = False
    var num_probes = 10
    var max_cg_iter = 200
    var cg_tol = Float32(1e-2)
    var precond_rank = 15
    var method_int = 0
    var early_stop_patience = 15
    var early_stop_tol = Float32(1e-4)
    max_iterations = Int(args[10].__int__())
    learning_rate = py_to_f32(args[11])
    task_rank = Int(args[12].__int__())
    verbose = Bool(args[13].__bool__())
    num_probes = Int(args[14].__int__())
    max_cg_iter = Int(args[15].__int__())
    cg_tol = py_to_f32(args[16])
    precond_rank = Int(args[17].__int__())
    var precond_method = Int(args[18].__int__())
    var precond_rebuild_threshold = py_to_f32(args[19])
    method_int = Int(args[20].__int__())
    var max_tridiag_iter = Int(args[21].__int__())
    early_stop_patience = Int(args[22].__int__())
    early_stop_tol = py_to_f32(args[23])
    var use_cosine_lr = False
    if len(args) >= 25:
        use_cosine_lr = Bool(args[24].__bool__())
    var progress_callback: PythonObject = None
    var progress_interval = 1
    var progress_enabled = False
    if len(args) > 25:
        progress_callback = args[25]
        progress_enabled = True
    if len(args) > 26:
        progress_interval = Int(args[26].__int__())

    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1

    var T = num_tasks
    var n = Int(Y_np.shape[0])
    var num_cat_vars = Int(cat_data_np.shape[1])
    var ctx = DeviceContext()

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

    var cat_state = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n)
    var total_cat_params = cat_state.total_cat_params

    var cat_data_c = np.ascontiguousarray(cat_data_np, dtype=np.int32)
    var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](n * num_cat_vars)
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n):
            cat_data_host.unsafe_ptr()[v * n + i] = Int32(Int(cat_data_c[i][v].__int__()))
    cat_state.upload_categorical_data(cat_data_host)

    var cat_params_c = np.ascontiguousarray(cat_params_np, dtype=np.float32).flatten()
    var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    ctx.synchronize()
    for i in range(total_cat_params):
        cat_params_host.unsafe_ptr()[i] = py_to_f32(cat_params_c[i])
    if total_cat_params > 0:
        cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())

    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n_prov = Int(info["n"].__int__())
    var num_gp = Int(info["num_gradient_params"].__int__())
    var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
    var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
    var sup_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())

    var base_provider = ErasedJITProvider(
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

    var nT = n * T
    var Y_c = np.ascontiguousarray(Y_np, dtype=np.float32)
    var y_blocked_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n):
            y_blocked_host.unsafe_ptr()[t * n + i] = py_to_f32(Y_c[i][t])
    var y_blocked_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.enqueue_copy(y_blocked_device, y_blocked_host)
    ctx.synchronize()

    var params_c = np.ascontiguousarray(params_np, dtype=np.float32).flatten()
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](max(num_gp, 1))
    ctx.synchronize()
    for i in range(num_gp):
        params_host.unsafe_ptr()[i] = py_to_f32(params_c[i])

    var noise_c = np.ascontiguousarray(noise_per_task_np, dtype=np.float32).flatten()
    var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(noise_c, noise_host, T)

    var mean_c = np.ascontiguousarray(mean_per_task_np, dtype=np.float32).flatten()
    var mean_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(mean_c, mean_host, T)

    var result = train_multi_output_mixed_jit(
        base_provider^,
        cat_state^,
        ctx,
        y_blocked_device.unsafe_ptr(),
        n,
        T,
        num_gp,
        params_host.unsafe_ptr(),
        noise_host.unsafe_ptr(),
        init_outputscale,
        mean_host.unsafe_ptr(),
        cat_params_host.unsafe_ptr(),
        total_cat_params,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        task_rank=task_rank,
        num_probes=num_probes,
        max_cg_iter=max_cg_iter,
        max_tridiag_iter=max_tridiag_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        precond_method=precond_method,
        precond_rebuild_threshold=precond_rebuild_threshold,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
        verbose=verbose,
        use_materialized=use_materialized,
        use_cosine_lr=use_cosine_lr,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        progress_enabled=progress_enabled,
    )

    var out = Python.dict()
    out["final_nll"] = Float64(result.final_nll)
    out["outputscale"] = Float64(result.outputscale)
    out["iterations"] = result.iterations
    out["converged"] = result.converged
    out["num_tasks"] = result.num_tasks
    out["task_rank"] = result.task_rank
    out["max_tridiag_iter"] = max_tridiag_iter
    out["precond_rebuild_threshold"] = precond_rebuild_threshold
    out["precond_rebuild_count"] = result.precond_rebuild_count
    out["precond_rank"] = precond_rank
    out["precond_method"] = precond_method
    out["early_stop_patience"] = early_stop_patience
    out["early_stop_tol"] = Float64(early_stop_tol)
    out["use_cosine_lr"] = use_cosine_lr
    out["training_route"] = "materialized" if use_materialized else "matrix_free"
    out["materialization_mode"] = method_int

    var cg_history = Python.list()
    for i in range(len(result.cg_iterations_history)):
        cg_history.append(result.cg_iterations_history[i])
    out["cg_iterations_history"] = cg_history

    var nll_list = Python.list()
    for i in range(len(result.nll_history)):
        nll_list.append(Float64(result.nll_history[i]))
    out["nll_history"] = nll_list

    var params_list = Python.list()
    for p in range(result.num_kernel_params):
        params_list.append(Float64(result.final_params[p]))
    out["params"] = params_list

    var cat_params_out = np.zeros(max(total_cat_params, 1), dtype=np.float32)
    for k in range(total_cat_params):
        cat_params_out[k] = Float64(result.cat_params[k])
    out["cat_params"] = cat_params_out[:total_cat_params]

    var noise_list = Python.list()
    for t in range(T):
        noise_list.append(Float64(result.noise_per_task[t]))
    out["noise_per_task"] = noise_list

    var mean_list = Python.list()
    for t in range(T):
        mean_list.append(Float64(result.mean_per_task[t]))
    out["mean_per_task"] = mean_list

    var B_list = Python.list()
    for i in range(T * T):
        B_list.append(Float64(result.B_flat[i]))
    out["B_flat"] = B_list

    var alpha_np = np.zeros(nT, dtype=np.float32)
    for i in range(nT):
        alpha_np[i] = Float64(result.alpha_blocked[i])
    out["alpha"] = alpha_np

    _ = y_blocked_host
    _ = y_blocked_device
    _ = params_host
    _ = noise_host
    _ = mean_host
    _ = cat_data_host
    _ = cat_params_host

    return out


fn predict_multi_output_mixed_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Predict with a trained mixed multi-output GP."""
    var np = Python.import_module("numpy")

    var info = args[0]
    var alpha_rotated_np = args[1]
    var Q_np = args[2]
    var effective_scales_np = args[3]
    var x_test_np = args[4]
    var params_np = args[5]
    var noise = py_to_f32(args[6])
    var cat_train_np = args[7]
    var cat_test_np = args[8]
    var cat_params_np = args[9]
    var cat_specs_py = args[10]

    var variance_method = PREDICT_MEAN_ONLY
    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var lanczos_rank = 0
    var method_int = 0
    var args_len = len(args)
    if args_len > 11:
        variance_method = Int(args[11].__int__())
    if args_len > 12:
        max_cg_iter = Int(args[12].__int__())
    if args_len > 13:
        cg_tol = py_to_f32(args[13])
    if args_len > 14:
        precond_rank = Int(args[14].__int__())
    if args_len > 15:
        lanczos_rank = Int(args[15].__int__())
    if args_len > 16:
        method_int = Int(args[16].__int__())

    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1
    lanczos_rank = _resolve_multi_predict_lanczos_rank(
        lanczos_rank, True, use_materialized, _info_bool(info, "is_ard"),
    )

    var n = Int(alpha_rotated_np.shape[0])
    var T = Int(alpha_rotated_np.shape[1])
    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])
    var num_cat_vars = Int(cat_train_np.shape[1])
    var can_compute_variance = variance_method != PREDICT_MEAN_ONLY
    var ctx = DeviceContext()

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

    var cat_state = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n)
    var total_cat_params = cat_state.total_cat_params

    var cat_train_c = np.ascontiguousarray(cat_train_np, dtype=np.int32)
    var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](n * num_cat_vars)
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n):
            cat_data_host.unsafe_ptr()[v * n + i] = Int32(Int(cat_train_c[i][v].__int__()))
    cat_state.upload_categorical_data(cat_data_host)

    var cat_params_c = np.ascontiguousarray(cat_params_np, dtype=np.float32).flatten()
    var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
    ctx.synchronize()
    for i in range(total_cat_params):
        cat_params_host.unsafe_ptr()[i] = py_to_f32(cat_params_c[i])
    if total_cat_params > 0:
        cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())

    var provider_ptr = Int(info["provider_ptr"].__int__())
    var n_prov = Int(info["n"].__int__())
    var num_gp = Int(info["num_gradient_params"].__int__())
    var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
    var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
    var sup_3p = Bool(info["supports_fused_3param"].__bool__())
    var x_ptr_addr = Int(info["x_ptr"].__int__())

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

    var params_c = np.ascontiguousarray(params_np, dtype=np.float32).ravel()
    var params_host = ctx.enqueue_create_host_buffer[float_dtype](max(num_gp, 1))
    ctx.synchronize()
    for i in range(num_gp):
        params_host.unsafe_ptr()[i] = py_to_f32(params_c[i])
    provider.update_params(params_host.unsafe_ptr())
    provider.update_noise(noise)

    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)

    var cat_test_c = np.ascontiguousarray(cat_test_np, dtype=np.int32)
    var cat_test_host = ctx.enqueue_create_host_buffer[DType.int32](n_test * num_cat_vars)
    ctx.synchronize()
    for v in range(num_cat_vars):
        for i in range(n_test):
            cat_test_host.unsafe_ptr()[v * n_test + i] = Int32(Int(cat_test_c[i][v].__int__()))
    var cat_test_device = ctx.enqueue_create_buffer[DType.int32](n_test * num_cat_vars)
    ctx.enqueue_copy(cat_test_device, cat_test_host)
    ctx.synchronize()

    var x_test_single_host = ctx.enqueue_create_host_buffer[float_dtype](max(dim, 1))
    var x_test_single_device = ctx.enqueue_create_buffer[float_dtype](max(dim, 1))
    var cat_test_single_host = ctx.enqueue_create_host_buffer[DType.int32](max(num_cat_vars, 1))
    var cat_test_single_device = ctx.enqueue_create_buffer[DType.int32](max(num_cat_vars, 1))
    var var_single_host = ctx.enqueue_create_host_buffer[float_dtype](1)

    var mixed_provider = MixedKroneckerBaseProviderView(provider^, cat_state)

    var alpha_rotated_c = np.ascontiguousarray(alpha_rotated_np, dtype=np.float32).ravel()
    var Q_c = np.ascontiguousarray(Q_np, dtype=np.float32).ravel()
    var effective_scales_c = np.ascontiguousarray(effective_scales_np, dtype=np.float32).ravel()

    var mean_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var variance_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var alpha_s_host = ctx.enqueue_create_host_buffer[float_dtype](n)
    var alpha_s_device = ctx.enqueue_create_buffer[float_dtype](n)
    var cross_buffer = ctx.enqueue_create_buffer[float_dtype](n_test)
    var cross_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    var var_host = ctx.enqueue_create_host_buffer[float_dtype](n_test)
    ctx.synchronize()

    for s in range(T):
        for i in range(n):
            alpha_s_host.unsafe_ptr()[i] = py_to_f32(alpha_rotated_c[i * T + s])
        ctx.enqueue_copy(alpha_s_device, alpha_s_host)
        ctx.synchronize()

        mixed_provider.mixed_cross_matvec(
            cross_buffer.unsafe_ptr(),
            x_test_device.unsafe_ptr(),
            alpha_s_device.unsafe_ptr(),
            cat_test_device.unsafe_ptr(),
            n_test,
            1,
        )
        ctx.enqueue_copy(cross_host, cross_buffer)
        ctx.synchronize()

        var scale_s = py_to_f32(effective_scales_c[s])
        for t in range(T):
            var q_ts = py_to_f32(Q_c[t * T + s])
            for i in range(n_test):
                var current_mean = py_to_f32(mean_out_flat[i * T + t])
                mean_out_flat[i * T + t] = current_mean + q_ts * scale_s * cross_host.unsafe_ptr()[i]

        if can_compute_variance and scale_s > Float32(1e-8):
            mixed_provider.update_noise(noise / scale_s)
            if variance_method == PREDICT_EXACT:
                for i in range(n_test):
                    for d in range(dim):
                        x_test_single_host.unsafe_ptr()[d] = x_test_host.unsafe_ptr()[i * dim + d]
                    ctx.enqueue_copy(x_test_single_device, x_test_single_host)
                    for v in range(num_cat_vars):
                        cat_test_single_host.unsafe_ptr()[v] = cat_test_host.unsafe_ptr()[v * n_test + i]
                    ctx.enqueue_copy(cat_test_single_device, cat_test_single_host)
                    ctx.synchronize()

                    var latent_var_device = predict_variance_mixed_jit(
                        mixed_provider,
                        ctx,
                        x_test_single_device.unsafe_ptr(),
                        cat_test_single_device.unsafe_ptr(),
                        n,
                        1,
                        dim,
                        PREDICT_EXACT,
                        max_cg_iter,
                        cg_tol,
                        precond_rank,
                        lanczos_rank,
                        0,
                        use_materialized=use_materialized,
                    )
                    ctx.enqueue_copy(var_single_host, latent_var_device)
                    ctx.synchronize()
                    var_host.unsafe_ptr()[i] = var_single_host.unsafe_ptr()[0]
                    _ = latent_var_device
            else:
                var latent_var_device = predict_variance_mixed_jit(
                    mixed_provider,
                    ctx,
                    x_test_device.unsafe_ptr(),
                    cat_test_device.unsafe_ptr(),
                    n,
                    n_test,
                    dim,
                    variance_method,
                    max_cg_iter,
                    cg_tol,
                    precond_rank,
                    lanczos_rank,
                    0,
                    use_materialized=use_materialized,
                )
                ctx.enqueue_copy(var_host, latent_var_device)
                ctx.synchronize()

            if variance_method == PREDICT_LOVE:
                var need_exact_fallback = False
                for i in range(n_test):
                    var latent_var_i = var_host.unsafe_ptr()[i]
                    if isnan(latent_var_i) or isinf(latent_var_i):
                        need_exact_fallback = True
                        break

                if need_exact_fallback:
                    for i in range(n_test):
                        for d in range(dim):
                            x_test_single_host.unsafe_ptr()[d] = x_test_host.unsafe_ptr()[i * dim + d]
                        ctx.enqueue_copy(x_test_single_device, x_test_single_host)
                        for v in range(num_cat_vars):
                            cat_test_single_host.unsafe_ptr()[v] = cat_test_host.unsafe_ptr()[v * n_test + i]
                        ctx.enqueue_copy(cat_test_single_device, cat_test_single_host)
                        ctx.synchronize()

                        var latent_var_exact_device = predict_variance_mixed_jit(
                            mixed_provider,
                            ctx,
                            x_test_single_device.unsafe_ptr(),
                            cat_test_single_device.unsafe_ptr(),
                            n,
                            1,
                            dim,
                            PREDICT_EXACT,
                            max_cg_iter,
                            cg_tol,
                            precond_rank,
                            lanczos_rank,
                            0,
                            use_materialized=use_materialized,
                        )
                        ctx.enqueue_copy(var_single_host, latent_var_exact_device)
                        ctx.synchronize()
                        var_host.unsafe_ptr()[i] = var_single_host.unsafe_ptr()[0]
                        _ = latent_var_exact_device

            for t in range(T):
                var q_ts = py_to_f32(Q_c[t * T + s])
                var weight = q_ts * q_ts * scale_s
                for i in range(n_test):
                    var current_var = py_to_f32(variance_out_flat[i * T + t])
                    variance_out_flat[i * T + t] = current_var + weight * var_host.unsafe_ptr()[i]

    var out = Python.dict()
    out["mean"] = mean_out_flat.reshape(n_test, T)
    if can_compute_variance:
        out["variance"] = variance_out_flat.reshape(n_test, T)
        out["std"] = np.sqrt(out["variance"])

    _ = params_host
    _ = x_test_host
    _ = x_test_device
    _ = cat_test_host
    _ = cat_test_device
    _ = alpha_s_host
    _ = alpha_s_device
    _ = cross_buffer
    _ = cross_host
    _ = var_host
    _ = x_test_single_host
    _ = x_test_single_device
    _ = cat_test_single_host
    _ = cat_test_single_device
    _ = var_single_host
    _ = cat_data_host
    _ = cat_params_host
    _ = cat_state

    return out
