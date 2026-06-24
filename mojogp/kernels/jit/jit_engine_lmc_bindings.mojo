from python import PythonObject, Python
from gpu.host import DeviceContext, DeviceBuffer
from memory import UnsafePointer, alloc
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
from kernels.jit.jit_multi_output_mixed import (
    MixedKroneckerBaseProviderView,
    predict_variance_mixed_jit,
)
from kernels.jit.jit_lmc import train_lmc_jit, JITLMCGradientAdapter
from kernels.jit.jit_lmc_mixed import train_lmc_mixed_jit, JITLMCMixedGradientAdapter
from kernels.py_conversion import bulk_copy_to_host_buffer, py_to_f32
from kernels.constants import float_dtype, CAT_KERNEL_GD, CAT_KERNEL_CR, CAT_KERNEL_EHH, CAT_KERNEL_HH, CAT_KERNEL_FE
from kernels.categorical_state import CategoricalCorrelationState
from kernels.jit.jit_engine_binding_helpers import (
    _info_bool,
    _info_materialization_mode,
    _resolve_multi_predict_lanczos_rank,
)


fn train_lmc_mixed_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Train mixed LMC multi-output GP with per-latent categorical state."""
    if len(args) < 22 or len(args) > 24:
        raise Error("train_lmc_mixed() expects 22 to 24 positional arguments")

    var np = Python.import_module("numpy")

    var provider_infos = args[0]
    var init_params_per_latent_py = args[1]
    var Y_np = args[2]
    var num_tasks = Int(args[3].__int__())
    var latent_C_trains = args[4]
    var latent_cat_specs_py = args[5]
    var cat_init_params_per_latent_py = args[6]

    var max_iters = Int(args[7].__int__())
    var lr = py_to_f32(args[8])
    var verbose = Bool(args[9].__bool__())
    var num_probes = Int(args[10].__int__())
    var max_cg_iter = Int(args[11].__int__())
    var cg_tol = py_to_f32(args[12])
    var precond_rank = Int(args[13].__int__())
    var precond_method = Int(args[14].__int__())
    var precond_rebuild_threshold = py_to_f32(args[15])
    var max_tridiag_iter = Int(args[16].__int__())
    var init_noise_np = args[17]
    var fixed_noise_np_arg = args[18]
    var has_fixed_noise = Bool(args[19].__bool__())
    var init_mean_np = args[20]
    var method_int = Int(args[21].__int__())
    var progress_callback: PythonObject = None
    var progress_interval = 1
    var progress_enabled = False
    if len(args) > 22:
        progress_callback = args[22]
        progress_enabled = True
    if len(args) > 23:
        progress_interval = Int(args[23].__int__())

    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1

    var n = Int(Y_np.shape[0])
    var T = num_tasks
    var R = Int(len(provider_infos))
    var ctx = DeviceContext()

    var cat_states = alloc[CategoricalCorrelationState](R)
    var providers = alloc[MixedKroneckerBaseProviderView](R)
    var params_per_latent = alloc[UnsafePointer[Float32, MutAnyOrigin]](R)
    var num_params_per_latent = alloc[Int](R)
    var cat_params_per_latent = alloc[UnsafePointer[Float32, MutAnyOrigin]](R)
    var num_cat_params_per_latent = alloc[Int](R)

    for s in range(R):
        var cat_specs_s = latent_cat_specs_py[s]
        var C_np_s = latent_C_trains[s]
        var num_cat_vars_s = Int(len(cat_specs_s))

        var levels_list = List[Int]()
        var ktypes_list = List[Int]()
        for v in range(num_cat_vars_s):
            var spec = cat_specs_s[v]
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

        var cat_state_s = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n)
        cat_states.offset(s).init_pointee_move(cat_state_s^)

        if num_cat_vars_s > 0:
            var cat_data_c = np.ascontiguousarray(C_np_s, dtype=np.int32)
            var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](n * num_cat_vars_s)
            ctx.synchronize()
            for v in range(num_cat_vars_s):
                for i in range(n):
                    cat_data_host.unsafe_ptr()[v * n + i] = Int32(Int(cat_data_c[i][v].__int__()))
            cat_states.offset(s)[].upload_categorical_data(cat_data_host)
            _ = cat_data_host

        var cat_init_np_s = np.ascontiguousarray(cat_init_params_per_latent_py[s], dtype=np.float32).flatten()
        var total_cat_params_s = cat_states.offset(s)[].total_cat_params
        var cat_buf_s = alloc[Float32](max(total_cat_params_s, 1))
        for k in range(total_cat_params_s):
            cat_buf_s[k] = py_to_f32(cat_init_np_s[k])
        cat_params_per_latent[s] = cat_buf_s
        num_cat_params_per_latent[s] = total_cat_params_s
        if total_cat_params_s > 0:
            cat_states.offset(s)[].update_correlation_matrices(cat_buf_s)

        var info = provider_infos[s]
        var provider_ptr = Int(info["provider_ptr"].__int__())
        var n_prov = Int(info["n"].__int__())
        var num_gp = Int(info["num_gradient_params"].__int__())
        var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
        var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
        var sup_3p = Bool(info["supports_fused_3param"].__bool__())
        var x_ptr_addr = Int(info["x_ptr"].__int__())

        var init_params_np_s = np.ascontiguousarray(init_params_per_latent_py[s], dtype=np.float32).flatten()
        var cont_buf_s = alloc[Float32](max(num_gp, 1))
        for p in range(num_gp):
            cont_buf_s[p] = py_to_f32(init_params_np_s[p])
        params_per_latent[s] = cont_buf_s
        num_params_per_latent[s] = num_gp

        var provider_s = ErasedJITProvider(
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
        providers.offset(s).init_pointee_move(
            MixedKroneckerBaseProviderView(provider_s^, cat_states.offset(s)[])
        )

    var nT = n * T
    var Y_c = np.ascontiguousarray(Y_np, dtype=np.float32)
    var y_blocked_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n):
            y_blocked_host.unsafe_ptr()[t * n + i] = py_to_f32(Y_c[i][t])

    var init_noise = alloc[Float32](T)
    var noise_np = np.ascontiguousarray(init_noise_np, dtype=np.float32).flatten()
    for t in range(T):
        init_noise[t] = py_to_f32(noise_np[t])

    var fixed_noise = alloc[Float32](max(nT, 1))
    if has_fixed_noise:
        var fixed_noise_np = np.ascontiguousarray(fixed_noise_np_arg, dtype=np.float32)
        if Int(fixed_noise_np.shape[0]) != n or Int(fixed_noise_np.shape[1]) != T:
            raise Error("train_lmc_mixed() fixed observation noise must have shape [n, T]")
        for t in range(T):
            for i in range(n):
                fixed_noise[t * n + i] = py_to_f32(fixed_noise_np[i][t])
    else:
        for i in range(max(nT, 1)):
            fixed_noise[i] = Float32(0.0)

    var init_mean = alloc[Float32](T)
    var mean_np = np.ascontiguousarray(init_mean_np, dtype=np.float32).flatten()
    for t in range(T):
        init_mean[t] = py_to_f32(mean_np[t])

    var result = train_lmc_mixed_jit(
        providers,
        cat_states,
        ctx,
        y_blocked_host.unsafe_ptr(),
        n,
        T,
        R,
        params_per_latent,
        num_params_per_latent,
        cat_params_per_latent,
        num_cat_params_per_latent,
        init_noise,
        fixed_noise,
        has_fixed_noise,
        init_mean,
        max_iterations=max_iters,
        learning_rate=lr,
        num_probes=num_probes,
        max_cg_iter=max_cg_iter,
        max_tridiag_iter=max_tridiag_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        precond_method=precond_method,
        precond_rebuild_threshold=precond_rebuild_threshold,
        verbose=verbose,
        use_materialized=use_materialized,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        progress_enabled=progress_enabled,
    )

    var out = Python.dict()
    out["status"] = "trained"
    out["model_type"] = "lmc_mixed"
    out["num_latents"] = R
    out["num_tasks"] = T
    out["n"] = n
    out["final_nll"] = Float64(result.final_nll)
    out["iterations"] = result.iterations
    out["converged"] = result.converged
    out["max_tridiag_iter"] = max_tridiag_iter
    out["precond_rebuild_threshold"] = precond_rebuild_threshold
    out["precond_rebuild_count"] = result.precond_rebuild_count
    out["precond_rank"] = precond_rank
    out["precond_method"] = precond_method
    out["training_route"] = "materialized" if use_materialized else "matrix_free"
    out["materialization_mode"] = method_int

    var all_params = Python.list()
    for s in range(R):
        var latent_params = Python.list()
        for p in range(len(result.final_params_per_latent[s])):
            latent_params.append(Float64(result.final_params_per_latent[s][p]))
        all_params.append(latent_params)
    out["params_per_latent"] = all_params

    var all_cat_params = Python.list()
    for s in range(R):
        var latent_cat_params = Python.list()
        for k in range(len(result.final_cat_params_per_latent[s])):
            latent_cat_params.append(Float64(result.final_cat_params_per_latent[s][k]))
        all_cat_params.append(latent_cat_params)
    out["cat_params_per_latent"] = all_cat_params

    var noise_list = Python.list()
    for t in range(T):
        noise_list.append(Float64(result.noise_per_task[t]))
    out["noise_per_task"] = noise_list

    var mean_list = Python.list()
    for t in range(T):
        mean_list.append(Float64(result.mean_per_task[t]))
    out["mean_per_task"] = mean_list

    var A_flat_size = R * T * T
    var A_np = np.zeros(A_flat_size, dtype=np.float32)
    for i in range(A_flat_size):
        A_np[i] = Float64(result.A_matrices_flat[i])
    out["A_matrices"] = A_np.reshape(R, T, T)

    var L_np = np.zeros(A_flat_size, dtype=np.float32)
    for i in range(A_flat_size):
        L_np[i] = Float64(result.L_factors_flat[i])
    out["L_factors"] = L_np.reshape(R, T, T)

    var vd_size = R * T
    var vd_np = np.zeros(vd_size, dtype=np.float32)
    for i in range(vd_size):
        vd_np[i] = Float64(result.var_diag_flat[i])
    out["var_diag"] = vd_np.reshape(R, T)

    var alpha_size = n * T
    var alpha_np = np.zeros(alpha_size, dtype=np.float32)
    for i in range(alpha_size):
        alpha_np[i] = Float64(result.alpha_blocked[i])
    out["alpha"] = alpha_np

    var nll_list = Python.list()
    for i in range(len(result.nll_history)):
        nll_list.append(Float64(result.nll_history[i]))
    out["nll_history"] = nll_list

    var iter_times_list = Python.list()
    for i in range(len(result.iter_times_ns)):
        iter_times_list.append(Float64(result.iter_times_ns[i]) / 1e6)
    out["iter_times_ms"] = iter_times_list

    for s in range(R):
        params_per_latent[s].free()
        cat_params_per_latent[s].free()
        providers.offset(s).destroy_pointee()
        cat_states.offset(s).destroy_pointee()
    params_per_latent.free()
    num_params_per_latent.free()
    cat_params_per_latent.free()
    num_cat_params_per_latent.free()
    providers.free()
    cat_states.free()
    init_noise.free()
    fixed_noise.free()
    init_mean.free()

    _ = y_blocked_host

    return out


fn train_lmc_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Train LMC multi-output GP with R latent kernels via joint BBMM."""
    if len(args) < 19 or len(args) > 21:
        raise Error("train_lmc() expects 19 to 21 positional arguments")

    var np = Python.import_module("numpy")

    var provider_infos = args[0]
    var init_params_per_latent_py = args[1]
    var trainable_masks_per_latent_py = args[2]
    var Y_np = args[3]
    var num_tasks = Int(args[4].__int__())
    var max_iters = Int(args[5].__int__())
    var lr = py_to_f32(args[6])
    var verbose = Bool(args[7].__bool__())
    var num_probes = Int(args[8].__int__())
    var max_cg_iter = Int(args[9].__int__())
    var cg_tol = py_to_f32(args[10])
    var precond_rank = Int(args[11].__int__())
    var precond_method = Int(args[12].__int__())
    var precond_rebuild_threshold = py_to_f32(args[13])
    var max_tridiag_iter = Int(args[14].__int__())
    var progress_callback: PythonObject = None
    var progress_interval = 1
    var progress_enabled = False
    if len(args) > 19:
        progress_callback = args[19]
        progress_enabled = True
    if len(args) > 20:
        progress_interval = Int(args[20].__int__())

    var n = Int(Y_np.shape[0])
    var T = num_tasks
    var R = Int(len(provider_infos))

    var ctx = DeviceContext()
    var providers = alloc[ErasedJITProvider](R)

    for s in range(R):
        var info = provider_infos[s]
        var provider_ptr = Int(info["provider_ptr"].__int__())
        var n_prov = Int(info["n"].__int__())
        var num_gp = Int(info["num_gradient_params"].__int__())
        var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
        var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
        var sup_3p = Bool(info["supports_fused_3param"].__bool__())
        var x_ptr_addr = Int(info["x_ptr"].__int__())

        providers.offset(s).init_pointee_move(ErasedJITProvider(
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
        ))

    var nT = n * T
    var Y_c = np.ascontiguousarray(Y_np, dtype=np.float32)
    var y_blocked_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n):
            y_blocked_host.unsafe_ptr()[t * n + i] = py_to_f32(Y_c[i][t])

    var params_per_latent = alloc[UnsafePointer[Float32, MutAnyOrigin]](R)
    var num_params_per_latent = alloc[Int](R)
    var trainable_masks_per_latent = alloc[UnsafePointer[Bool, MutAnyOrigin]](R)
    for s in range(R):
        var info_s = provider_infos[s]
        var num_gp = Int(info_s["num_gradient_params"].__int__())
        var init_params_np_s = np.ascontiguousarray(init_params_per_latent_py[s], dtype=np.float32).flatten()
        var trainable_mask_np_s = np.ascontiguousarray(trainable_masks_per_latent_py[s], dtype=np.bool_).flatten()
        if Int(init_params_np_s.shape[0]) != num_gp:
            raise Error("train_lmc() initial params length does not match provider gradient params")
        if Int(trainable_mask_np_s.shape[0]) != num_gp:
            raise Error("train_lmc() trainable mask length does not match provider gradient params")
        num_params_per_latent[s] = num_gp
        var p_buf = alloc[Float32](max(num_gp, 1))
        var mask_buf = alloc[Bool](max(num_gp, 1))
        for i in range(num_gp):
            p_buf[i] = py_to_f32(init_params_np_s[i])
            mask_buf[i] = Bool(trainable_mask_np_s[i].__bool__())
        params_per_latent[s] = p_buf
        trainable_masks_per_latent[s] = mask_buf

    var init_noise = alloc[Float32](T)
    var noise_np = np.ascontiguousarray(args[15], dtype=np.float32)
    for t in range(T):
        init_noise[t] = py_to_f32(noise_np[t])

    var fixed_noise = alloc[Float32](max(nT, 1))
    var has_fixed_noise = Bool(args[17].__bool__())
    if has_fixed_noise:
        var fixed_noise_np = np.ascontiguousarray(args[16], dtype=np.float32)
        if Int(fixed_noise_np.shape[0]) != n or Int(fixed_noise_np.shape[1]) != T:
            raise Error("train_lmc() fixed observation noise must have shape [n, T]")
        for t in range(T):
            for i in range(n):
                fixed_noise[t * n + i] = py_to_f32(fixed_noise_np[i][t])
    else:
        for i in range(max(nT, 1)):
            fixed_noise[i] = Float32(0.0)

    var init_mean = alloc[Float32](T)
    var mean_np = np.ascontiguousarray(args[18], dtype=np.float32)
    for t in range(T):
        init_mean[t] = py_to_f32(mean_np[t])

    var result = train_lmc_jit(
        providers, ctx,
        y_blocked_host.unsafe_ptr(),
        n, T, R,
        params_per_latent, num_params_per_latent, trainable_masks_per_latent,
        init_noise, fixed_noise, has_fixed_noise, init_mean,
        max_iterations=max_iters,
        learning_rate=lr,
        num_probes=num_probes,
        max_cg_iter=max_cg_iter,
        max_tridiag_iter=max_tridiag_iter,
        cg_tol=cg_tol,
        precond_rank=precond_rank,
        precond_method=precond_method,
        precond_rebuild_threshold=precond_rebuild_threshold,
        verbose=verbose,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        progress_enabled=progress_enabled,
    )

    var out = Python.dict()
    out["status"] = "trained"
    out["model_type"] = "lmc"
    out["num_latents"] = R
    out["num_tasks"] = T
    out["n"] = n
    out["final_nll"] = Float64(result.final_nll)
    out["iterations"] = result.iterations
    out["converged"] = result.converged
    out["max_tridiag_iter"] = max_tridiag_iter
    out["precond_rebuild_threshold"] = precond_rebuild_threshold
    out["precond_rebuild_count"] = result.precond_rebuild_count
    out["precond_rank"] = precond_rank
    out["precond_method"] = precond_method
    out["uses_fixed_observation_noise"] = has_fixed_noise
    out["training_route"] = (
        "materialized" if _info_materialization_mode(provider_infos[0]) == 1 else "matrix_free"
    )
    out["materialization_mode"] = _info_materialization_mode(provider_infos[0])

    var all_params = Python.list()
    for s in range(R):
        var latent_params = Python.list()
        for p in range(len(result.final_params_per_latent[s])):
            latent_params.append(Float64(result.final_params_per_latent[s][p]))
        all_params.append(latent_params)
    out["params_per_latent"] = all_params

    var noise_list = Python.list()
    for t in range(T):
        noise_list.append(Float64(result.noise_per_task[t]))
    out["noise_per_task"] = noise_list

    var mean_list = Python.list()
    for t in range(T):
        mean_list.append(Float64(result.mean_per_task[t]))
    out["mean_per_task"] = mean_list

    var A_flat_size = R * T * T
    var A_np = np.zeros(A_flat_size, dtype=np.float32)
    for i in range(A_flat_size):
        A_np[i] = Float64(result.A_matrices_flat[i])
    out["A_matrices"] = A_np.reshape(R, T, T)

    var L_np = np.zeros(A_flat_size, dtype=np.float32)
    for i in range(A_flat_size):
        L_np[i] = Float64(result.L_factors_flat[i])
    out["L_factors"] = L_np.reshape(R, T, T)

    var vd_size = R * T
    var vd_np = np.zeros(vd_size, dtype=np.float32)
    for i in range(vd_size):
        vd_np[i] = Float64(result.var_diag_flat[i])
    out["var_diag"] = vd_np.reshape(R, T)

    var alpha_size = n * T
    var alpha_np = np.zeros(alpha_size, dtype=np.float32)
    for i in range(alpha_size):
        alpha_np[i] = Float64(result.alpha_blocked[i])
    out["alpha"] = alpha_np

    var nll_list = Python.list()
    for i in range(len(result.nll_history)):
        nll_list.append(Float64(result.nll_history[i]))
    out["nll_history"] = nll_list

    var iter_times_list = Python.list()
    for i in range(len(result.iter_times_ns)):
        iter_times_list.append(Float64(result.iter_times_ns[i]) / 1e6)
    out["iter_times_ms"] = iter_times_list

    for s in range(R):
        params_per_latent[s].free()
        trainable_masks_per_latent[s].free()
        providers.offset(s).destroy_pointee()
    params_per_latent.free()
    num_params_per_latent.free()
    trainable_masks_per_latent.free()
    providers.free()
    init_noise.free()
    fixed_noise.free()
    init_mean.free()

    _ = y_blocked_host

    return out


fn sample_lmc_pathwise_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Compute one backend correction sample for continuous MultiOutputLMCGP pathwise sampling.

    Args:
        args[0]: provider_infos list
        args[1]: residual numpy array [n, T] float32
        args[2]: A_matrices numpy array [R, T, T] float32
        args[3]: X_test numpy array [m, d] float32
        args[4]: params_per_latent list of R arrays [num_params_s] float32
        args[5]: noise_per_task numpy array [T] float32
        args[6]: max_cg_iter int (optional, default 100)
        args[7]: cg_tol float (optional, default 1e-2)
        args[8]: fixed observation noise [n, T] float32 (optional)
    """
    if len(args) < 6 or len(args) > 9:
        raise Error("sample_lmc_pathwise() expects 6-9 positional arguments")

    var np = Python.import_module("numpy")

    var provider_infos = args[0]
    var residual_np = args[1]
    var A_np = args[2]
    var x_test_np = args[3]
    var params_per_latent_py = args[4]
    var noise_per_task_np = args[5]

    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    if len(args) > 6:
        max_cg_iter = Int(args[6].__int__())
    if len(args) > 7:
        cg_tol = py_to_f32(args[7])

    var n = Int(residual_np.shape[0])
    var T = Int(residual_np.shape[1])
    var R = Int(len(provider_infos))
    var nT = n * T
    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])

    var ctx = DeviceContext()
    var providers = alloc[ErasedJITProvider](R)

    for s in range(R):
        var info = provider_infos[s]
        var provider_ptr = Int(info["provider_ptr"].__int__())
        var n_prov = Int(info["n"].__int__())
        var num_gp = Int(info["num_gradient_params"].__int__())
        var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
        var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
        var sup_3p = Bool(info["supports_fused_3param"].__bool__())
        var x_ptr_addr = Int(info["x_ptr"].__int__())
        var has_cross = Bool(info.__contains__("cross_matvec"))
        if not has_cross:
            raise Error("sample_lmc_pathwise() requires cross_matvec support")
        var cross_ptr = Int(info["cross_matvec"].__int__())

        providers.offset(s).init_pointee_move(ErasedJITProvider(
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
            cross_matvec=_cvt_cross(cross_ptr),
            extract_diagonal_test=get_noop_diagtest(),
            has_prediction=True,
            kronecker_forward_matvec=get_noop_kron_fwd(),
            kronecker_gradient_matvec=get_noop_kron_grad(),
            has_kronecker=False,
        ))

        var params_s_py = params_per_latent_py[s]
        var params_s_c = np.ascontiguousarray(params_s_py, dtype=np.float32).ravel()
        var params_s_host = ctx.enqueue_create_host_buffer[float_dtype](num_gp)
        ctx.synchronize()
        bulk_copy_to_host_buffer(params_s_c, params_s_host, num_gp)
        providers[s].update_params(params_s_host.unsafe_ptr())
        providers[s].update_noise(Float32(0.0))
        _ = params_s_host
    
    var residual_c = np.ascontiguousarray(residual_np, dtype=np.float32)
    var residual_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n):
            residual_host.unsafe_ptr()[t * n + i] = py_to_f32(residual_c[i][t])

    var A_c = np.ascontiguousarray(A_np, dtype=np.float32).ravel()
    var A_host = ctx.enqueue_create_host_buffer[float_dtype](R * T * T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(A_c, A_host, R * T * T)

    var noise_c = np.ascontiguousarray(noise_per_task_np, dtype=np.float32).ravel()
    var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(noise_c, noise_host, T)

    var fixed_noise_host = ctx.enqueue_create_host_buffer[float_dtype](max(nT, 1))
    var has_fixed_noise = False
    if len(args) > 8:
        var fixed_noise_np = np.ascontiguousarray(args[8], dtype=np.float32)
        has_fixed_noise = Int(fixed_noise_np.size) == nT
        if has_fixed_noise:
            for i in range(n):
                for t in range(T):
                    fixed_noise_host.unsafe_ptr()[t * n + i] = py_to_f32(fixed_noise_np[i][t])
    if not has_fixed_noise:
        for i in range(nT):
            fixed_noise_host.unsafe_ptr()[i] = Float32(0.0)

    var adapter = JITLMCGradientAdapter(
        ctx,
        providers,
        R,
        T,
        n,
        A_host.unsafe_ptr(),
        noise_host.unsafe_ptr(),
        fixed_noise_host.unsafe_ptr(),
        has_fixed_noise,
        1,
    )

    var beta_host = solve_single_rhs_deterministic_host_jit(
        adapter,
        ctx,
        residual_host.unsafe_ptr(),
        nT,
        max_cg_iter,
        cg_tol,
    )
    var beta_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.enqueue_copy(beta_device, beta_host)
    ctx.synchronize()

    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)
    ctx.synchronize()

    var correction_flat = np.zeros(n_test * T, dtype=np.float32)
    var cross_buffer = ctx.enqueue_create_buffer[float_dtype](n_test * T)

    for s in range(R):
        providers[s].cross_matvec(
            cross_buffer.unsafe_ptr(),
            x_test_device.unsafe_ptr(),
            beta_device.unsafe_ptr(),
            n_test,
            T,
        )
        var cross_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
        ctx.enqueue_copy(cross_host, cross_buffer)
        ctx.synchronize()

        for t in range(T):
            for tp in range(T):
                var a_val = A_host.unsafe_ptr()[s * T * T + t * T + tp]
                for i in range(n_test):
                    var idx = i * T + t
                    correction_flat[idx] = Float64(
                        py_to_f32(correction_flat[idx])
                        + a_val * cross_host.unsafe_ptr()[tp * n_test + i]
                    )
        _ = cross_host

    var out = Python.dict()
    out["correction"] = correction_flat.reshape(n_test, T)

    for s in range(R):
        providers.offset(s).destroy_pointee()
    providers.free()

    _ = residual_host
    _ = A_host
    _ = noise_host
    _ = fixed_noise_host
    _ = beta_host
    _ = beta_device
    _ = x_test_host
    _ = x_test_device
    _ = cross_buffer

    return out


fn sample_lmc_mixed_pathwise_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Compute one backend correction sample for mixed MultiOutputLMCGP pathwise sampling.

    Args:
        args[0]: provider_infos list
        args[1]: residual numpy array [n, T] float32
        args[2]: A_matrices numpy array [R, T, T] float32
        args[3]: list of R transformed continuous X_test arrays [m, d_s] float32
        args[4]: params_per_latent list of R arrays [num_params_s] float32
        args[5]: latent_is_mixed list of R bools
        args[6]: latent_C_trains list of R categorical train arrays [n, c_s] int32
        args[7]: latent_C_tests list of R categorical test arrays [m, c_s] int32
        args[8]: cat_params_per_latent list of R arrays [num_cat_params_s] float32
        args[9]: latent_cat_specs list of R categorical spec lists
        args[10]: noise_per_task numpy array [T] float32
        args[11]: max_cg_iter int (optional, default 100)
        args[12]: cg_tol float (optional, default 1e-2)
        args[13]: materialization_mode int (optional, default 0)
    """
    if len(args) < 11 or len(args) > 14:
        raise Error("sample_lmc_mixed_pathwise() expects 11-14 positional arguments")

    var np = Python.import_module("numpy")

    var provider_infos = args[0]
    var residual_np = args[1]
    var A_np = args[2]
    var latent_x_test_np = args[3]
    var params_per_latent_py = args[4]
    var latent_is_mixed_py = args[5]
    var latent_C_trains = args[6]
    var latent_C_tests = args[7]
    var cat_params_per_latent_py = args[8]
    var latent_cat_specs_py = args[9]
    var noise_per_task_np = args[10]

    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var method_int = 0
    if len(args) > 11:
        max_cg_iter = Int(args[11].__int__())
    if len(args) > 12:
        cg_tol = py_to_f32(args[12])
    if len(args) > 13:
        method_int = Int(args[13].__int__())

    var n_train = Int(residual_np.shape[0])
    var T = Int(residual_np.shape[1])
    var R = Int(len(provider_infos))
    var nT = n_train * T
    var first_x_test_np = latent_x_test_np[0]
    var n_test = Int(first_x_test_np.shape[0])
    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1

    var ctx = DeviceContext()
    var cat_states = alloc[CategoricalCorrelationState](R)
    var providers = alloc[MixedKroneckerBaseProviderView](R)

    for s in range(R):
        var info = provider_infos[s]
        var is_mixed_s = Bool(latent_is_mixed_py[s].__bool__())
        var provider_ptr = Int(info["provider_ptr"].__int__())
        var n_prov = Int(info["n"].__int__())
        var num_gp = Int(info["num_gradient_params"].__int__())
        var sup_fused = Bool(info["supports_fused_gradient"].__bool__())
        var sup_lsos = Bool(info["supports_fused_ls_os"].__bool__())
        var sup_3p = Bool(info["supports_fused_3param"].__bool__())
        var x_ptr_addr = Int(info["x_ptr"].__int__())
        var has_cross = Bool(info.__contains__("cross_matvec"))
        if not is_mixed_s and not has_cross:
            raise Error("sample_lmc_mixed_pathwise() requires cross_matvec support for continuous latents")
        var cross_ptr = Int(info["cross_matvec"].__int__()) if has_cross else 0

        var provider_s = ErasedJITProvider(
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
            extract_diagonal_test=get_noop_diagtest(),
            has_prediction=has_cross,
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

        var params_s_py = params_per_latent_py[s]
        var params_s_c = np.ascontiguousarray(params_s_py, dtype=np.float32).ravel()
        var params_s_host = ctx.enqueue_create_host_buffer[float_dtype](max(num_gp, 1))
        ctx.synchronize()
        bulk_copy_to_host_buffer(params_s_c, params_s_host, num_gp)
        provider_s.update_params(params_s_host.unsafe_ptr())
        provider_s.update_noise(Float32(0.0))

        var cat_state_s = CategoricalCorrelationState(ctx, List[Int](), List[Int](), n_train)
        if is_mixed_s:
            var cat_specs_s = latent_cat_specs_py[s]
            var cat_train_np_s = latent_C_trains[s]
            var cat_params_s_c = np.ascontiguousarray(cat_params_per_latent_py[s], dtype=np.float32).flatten()
            var num_cat_vars_s = Int(len(cat_specs_s))

            var levels_list = List[Int]()
            var ktypes_list = List[Int]()
            for v in range(num_cat_vars_s):
                var spec = cat_specs_s[v]
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

            cat_state_s = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n_train)

            if num_cat_vars_s > 0:
                var cat_train_c = np.ascontiguousarray(cat_train_np_s, dtype=np.int32)
                var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](n_train * num_cat_vars_s)
                ctx.synchronize()
                for v in range(num_cat_vars_s):
                    for i in range(n_train):
                        cat_data_host.unsafe_ptr()[v * n_train + i] = Int32(Int(cat_train_c[i][v].__int__()))
                cat_state_s.upload_categorical_data(cat_data_host)
                _ = cat_data_host

            var total_cat_params_s = cat_state_s.total_cat_params
            var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params_s, 1))
            ctx.synchronize()
            for i in range(total_cat_params_s):
                cat_params_host.unsafe_ptr()[i] = py_to_f32(cat_params_s_c[i])
            if total_cat_params_s > 0:
                cat_state_s.update_correlation_matrices(cat_params_host.unsafe_ptr())
            _ = cat_params_host

        cat_states.offset(s).init_pointee_move(cat_state_s^)
        providers.offset(s).init_pointee_move(
            MixedKroneckerBaseProviderView(provider_s^, cat_states.offset(s)[])
        )
        if is_mixed_s and use_materialized:
            providers.offset(s)[].refresh_materialization()
        _ = params_s_host

    var residual_c = np.ascontiguousarray(residual_np, dtype=np.float32)
    var residual_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for t in range(T):
        for i in range(n_train):
            residual_host.unsafe_ptr()[t * n_train + i] = py_to_f32(residual_c[i][t])

    var A_c = np.ascontiguousarray(A_np, dtype=np.float32).ravel()
    var A_host = ctx.enqueue_create_host_buffer[float_dtype](R * T * T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(A_c, A_host, R * T * T)

    var noise_c = np.ascontiguousarray(noise_per_task_np, dtype=np.float32).ravel()
    var noise_host = ctx.enqueue_create_host_buffer[float_dtype](T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(noise_c, noise_host, T)

    var adapter = JITLMCMixedGradientAdapter(
        ctx,
        providers,
        R,
        T,
        n_train,
        A_host.unsafe_ptr(),
        noise_host.unsafe_ptr(),
        UnsafePointer[Float32, MutAnyOrigin](),
        False,
        1,
    )

    var beta_host = solve_single_rhs_deterministic_host_jit(
        adapter,
        ctx,
        residual_host.unsafe_ptr(),
        nT,
        max_cg_iter,
        cg_tol,
    )
    var beta_device = ctx.enqueue_create_buffer[float_dtype](nT)
    ctx.enqueue_copy(beta_device, beta_host)
    ctx.synchronize()

    var correction_flat = np.zeros(n_test * T, dtype=np.float32)
    var cross_buffer = ctx.enqueue_create_buffer[float_dtype](n_test * T)

    for s in range(R):
        var is_mixed_s = Bool(latent_is_mixed_py[s].__bool__())
        var x_test_np_s = latent_x_test_np[s]
        var n_test_s = Int(x_test_np_s.shape[0])
        if n_test_s != n_test:
            raise Error("sample_lmc_mixed_pathwise() requires matching test row counts across latents")
        var dim_s = Int(x_test_np_s.shape[1])

        var x_test_c_s = np.ascontiguousarray(x_test_np_s, dtype=np.float32)
        var x_test_host_s = ctx.enqueue_create_host_buffer[float_dtype](max(n_test * dim_s, 1))
        ctx.synchronize()
        bulk_copy_to_host_buffer(x_test_c_s.ravel(), x_test_host_s, n_test * dim_s)
        var x_test_device_s = ctx.enqueue_create_buffer[float_dtype](max(n_test * dim_s, 1))
        ctx.enqueue_copy(x_test_device_s, x_test_host_s)
        ctx.synchronize()

        if is_mixed_s:
            var cat_test_np_s = latent_C_tests[s]
            var cat_specs_s = latent_cat_specs_py[s]
            var num_cat_vars_s = Int(len(cat_specs_s))
            var cat_test_c = np.ascontiguousarray(cat_test_np_s, dtype=np.int32)
            var cat_test_host = ctx.enqueue_create_host_buffer[DType.int32](max(n_test * num_cat_vars_s, 1))
            ctx.synchronize()
            for v in range(num_cat_vars_s):
                for i in range(n_test):
                    cat_test_host.unsafe_ptr()[v * n_test + i] = Int32(Int(cat_test_c[i][v].__int__()))
            var cat_test_device = ctx.enqueue_create_buffer[DType.int32](max(n_test * num_cat_vars_s, 1))
            ctx.enqueue_copy(cat_test_device, cat_test_host)
            ctx.synchronize()

            if use_materialized:
                providers.offset(s)[].refresh_materialization()
            providers.offset(s)[].mixed_cross_matvec(
                cross_buffer.unsafe_ptr(),
                x_test_device_s.unsafe_ptr(),
                beta_device.unsafe_ptr(),
                cat_test_device.unsafe_ptr(),
                n_test,
                T,
            )
            _ = cat_test_host
            _ = cat_test_device
        else:
            providers.offset(s)[].provider.cross_matvec(
                cross_buffer.unsafe_ptr(),
                x_test_device_s.unsafe_ptr(),
                beta_device.unsafe_ptr(),
                n_test,
                T,
            )

        var cross_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
        ctx.enqueue_copy(cross_host, cross_buffer)
        ctx.synchronize()

        for t in range(T):
            for tp in range(T):
                var a_val = A_host.unsafe_ptr()[s * T * T + t * T + tp]
                for i in range(n_test):
                    var idx = i * T + t
                    correction_flat[idx] = Float64(
                        py_to_f32(correction_flat[idx])
                        + a_val * cross_host.unsafe_ptr()[tp * n_test + i]
                    )

        _ = cross_host
        _ = x_test_host_s
        _ = x_test_device_s

    var out = Python.dict()
    out["correction"] = correction_flat.reshape(n_test, T)

    for s in range(R):
        providers.offset(s).destroy_pointee()
        cat_states.offset(s).destroy_pointee()
    providers.free()
    cat_states.free()

    _ = residual_host
    _ = A_host
    _ = noise_host
    _ = beta_host
    _ = beta_device
    _ = cross_buffer

    return out


fn predict_lmc_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Predict mean/variance for LMC multi-output GP using GPU cross-matvec per latent.

    Computes: mean_pred[m, t] = sum_s sum_{t'} A_s[t,t'] * K_s(X_test, X_train) @ alpha[n, t']

    Args (from Python):
        args[0]: list of R provider_info dicts (one per latent), must include cross_matvec
        args[1]: alpha numpy array [n*T] float32 — task-blocked alpha from training
        args[2]: A_matrices numpy array [R, T, T] float32 — LMC mixing matrices
        args[3]: X_test numpy array [m, d] float32 — test inputs
        args[4]: params_per_latent list of R arrays — trained params per latent
        args[5]: mean_per_task numpy array [T] float32 — per-task mean offset
        args[6]: n_train (int)
        args[7]: num_tasks (int)
        args[8]: noise_per_task numpy array [T] float32 (optional)
        args[9]: variance_method int (0=mean_only, 1=love, 2=exact, optional)
        args[10]: max_cg_iter int (optional, default 100)
        args[11]: cg_tol float (optional, default 1e-2)
        args[12]: precond_rank int (optional, default 10)
        args[13]: lanczos_rank int (optional, default auto)

    Returns:
        dict with: mean [m, T] numpy float32, optional variance/std [m, T]
    """
    var np = Python.import_module("numpy")

    var provider_infos = args[0]
    var alpha_np = args[1]
    var A_np = args[2]
    var x_test_np = args[3]
    var params_per_latent_py = args[4]
    var mean_per_task_np = args[5]
    var n_train = Int(args[6].__int__())
    var T = Int(args[7].__int__())
    var R = Int(len(provider_infos))

    var variance_method = PREDICT_MEAN_ONLY
    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var lanczos_rank = 0
    var has_noise_per_task = len(args) > 8
    if len(args) > 9:
        variance_method = Int(args[9].__int__())
    if len(args) > 10:
        max_cg_iter = Int(args[10].__int__())
    if len(args) > 11:
        cg_tol = py_to_f32(args[11])
    if len(args) > 12:
        precond_rank = Int(args[12].__int__())
    if len(args) > 13:
        lanczos_rank = Int(args[13].__int__())

    var n_test = Int(x_test_np.shape[0])
    var dim = Int(x_test_np.shape[1])

    var ctx = DeviceContext()

    var x_test_c = np.ascontiguousarray(x_test_np, dtype=np.float32)
    var x_test_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * dim)
    ctx.synchronize()
    bulk_copy_to_host_buffer(x_test_c.ravel(), x_test_host, n_test * dim)
    var x_test_device = ctx.enqueue_create_buffer[float_dtype](n_test * dim)
    ctx.enqueue_copy(x_test_device, x_test_host)

    var alpha_c = np.ascontiguousarray(alpha_np, dtype=np.float32).ravel()
    var alpha_host = ctx.enqueue_create_host_buffer[float_dtype](n_train * T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(alpha_c, alpha_host, n_train * T)
    var alpha_device = ctx.enqueue_create_buffer[float_dtype](n_train * T)
    ctx.enqueue_copy(alpha_device, alpha_host)
    ctx.synchronize()

    var mean_accum_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
    ctx.synchronize()
    for i in range(n_test * T):
        mean_accum_host.unsafe_ptr()[i] = Float32(0.0)
    var mean_device = ctx.enqueue_create_buffer[float_dtype](n_test * T)
    ctx.enqueue_copy(mean_device, mean_accum_host)
    ctx.synchronize()

    var variance_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var latent_variances_flat = np.zeros(max(R * n_test, 1), dtype=np.float32)
    var can_compute_variance = variance_method != PREDICT_MEAN_ONLY and has_noise_per_task
    var use_materialized = False
    if R > 0:
        use_materialized = _info_materialization_mode(provider_infos[0]) != 0
    var has_ard_latent = False
    for s in range(R):
        if _info_bool(provider_infos[s], "is_ard"):
            has_ard_latent = True
    lanczos_rank = _resolve_multi_predict_lanczos_rank(
        lanczos_rank, False, use_materialized, has_ard_latent,
    )
    var avg_noise = Float32(0.0)
    if can_compute_variance:
        var noise_per_task_c = np.ascontiguousarray(args[8], dtype=np.float32).ravel()
        for t in range(T):
            avg_noise += py_to_f32(noise_per_task_c[t])
        avg_noise /= Float32(T)

        for i in range(n_test * T):
            variance_out_flat[i] = Float32(0.0)
    var latent_var_host = ctx.enqueue_create_host_buffer[float_dtype](max(n_test, 1))
    ctx.synchronize()

    var cross_buffer = ctx.enqueue_create_buffer[float_dtype](n_test * T)
    for s in range(R):
        var info = provider_infos[s]

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

        if can_compute_variance and not has_diagtest:
            can_compute_variance = False

        var provider_s = ErasedJITProvider(
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

        var params_s_py = params_per_latent_py[s]
        var params_s_c = np.ascontiguousarray(params_s_py, dtype=np.float32).ravel()
        var params_s_host = ctx.enqueue_create_host_buffer[float_dtype](num_gp)
        ctx.synchronize()
        bulk_copy_to_host_buffer(params_s_c, params_s_host, num_gp)
        provider_s.update_params(params_s_host.unsafe_ptr())

        provider_s.cross_matvec(
            cross_buffer.unsafe_ptr(),
            x_test_device.unsafe_ptr(),
            alpha_device.unsafe_ptr(),
            n_test,
            T,
        )
        ctx.synchronize()

        var A_s_c = np.ascontiguousarray(A_np[s], dtype=np.float32)
        var A_s_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
        ctx.synchronize()
        for i in range(T * T):
            A_s_host.unsafe_ptr()[i] = py_to_f32(A_s_c.ravel()[i])

        var cross_s_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
        ctx.enqueue_copy(cross_s_host, cross_buffer)
        var mean_host_s = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
        ctx.enqueue_copy(mean_host_s, mean_device)
        ctx.synchronize()

        for t in range(T):
            for tp in range(T):
                var a_val = A_s_host.unsafe_ptr()[t * T + tp]
                for i in range(n_test):
                    var cross_val = cross_s_host.unsafe_ptr()[tp * n_test + i]
                    mean_host_s.unsafe_ptr()[t * n_test + i] += a_val * cross_val

        if can_compute_variance:
            provider_s.update_noise(avg_noise)
            var latent_var_device: DeviceBuffer[float_dtype]
            if variance_method == PREDICT_LOVE:
                latent_var_device = predict_variance_love_jit(
                    provider_s,
                    provider_s,
                    False,
                    ctx,
                    x_test_device.unsafe_ptr(),
                    n_train,
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
                    provider_s,
                    provider_s,
                    False,
                    ctx,
                    alpha_device.unsafe_ptr(),
                    x_test_device.unsafe_ptr(),
                    n_train,
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

            ctx.enqueue_copy(latent_var_host, latent_var_device)
            ctx.synchronize()

            for i in range(n_test):
                latent_variances_flat[s * n_test + i] = Float64(latent_var_host.unsafe_ptr()[i])

            for t in range(T):
                var row_weight = Float32(0.0)
                for tp in range(T):
                    var a_val = A_s_host.unsafe_ptr()[t * T + tp]
                    row_weight += a_val * a_val
                for i in range(n_test):
                    var current_var = py_to_f32(variance_out_flat[i * T + t])
                    variance_out_flat[i * T + t] = current_var + row_weight * latent_var_host.unsafe_ptr()[i]

            _ = latent_var_device

        ctx.enqueue_copy(mean_device, mean_host_s)
        ctx.synchronize()

        _ = params_s_host
        _ = A_s_host
        _ = cross_s_host
        _ = mean_host_s

    var mean_host_final = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
    ctx.enqueue_copy(mean_host_final, mean_device)
    ctx.synchronize()

    var mean_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var mean_per_task_c = np.ascontiguousarray(mean_per_task_np, dtype=np.float32)
    for t in range(T):
        var task_mean_offset = py_to_f32(mean_per_task_c[t])
        for i in range(n_test):
            mean_out_flat[i * T + t] = mean_host_final.unsafe_ptr()[t * n_test + i] + task_mean_offset
    var mean_out = mean_out_flat.reshape(n_test, T)

    var out = Python.dict()
    out["mean"] = mean_out
    out["lanczos_rank_used"] = lanczos_rank
    if can_compute_variance:
        var noise_per_task_c = np.ascontiguousarray(args[8], dtype=np.float32).ravel()
        for t in range(T):
            var noise_t = py_to_f32(noise_per_task_c[t])
            for i in range(n_test):
                var current_var = py_to_f32(variance_out_flat[i * T + t])
                variance_out_flat[i * T + t] = current_var + noise_t

        out["variance"] = variance_out_flat.reshape(n_test, T)
        out["std"] = np.sqrt(out["variance"])
        out["latent_variances"] = latent_variances_flat.reshape(R, n_test)

    _ = x_test_host
    _ = x_test_device
    _ = alpha_host
    _ = alpha_device
    _ = mean_accum_host
    _ = mean_device
    _ = cross_buffer
    _ = latent_var_host
    _ = mean_host_final

    return out


fn predict_lmc_mixed_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:
    """Predict mean/variance for LMC multi-output GP with per-latent mixed routing.

    Args (from Python):
        args[0]: list of R provider_info dicts (one per latent)
        args[1]: alpha numpy array [n*T] float32 — task-blocked alpha from training
        args[2]: A_matrices numpy array [R, T, T] float32 — LMC mixing matrices
        args[3]: list of R transformed continuous X_test arrays [m, d_s] float32
        args[4]: params_per_latent list of R arrays — trained params per latent
        args[5]: mean_per_task numpy array [T] float32 — per-task mean offset
        args[6]: n_train (int)
        args[7]: num_tasks (int)
        args[8]: latent_is_mixed list of R bools
        args[9]: latent_C_trains list of R categorical train arrays [n, c_s] int32
        args[10]: latent_C_tests list of R categorical test arrays [m, c_s] int32
        args[11]: cat_params_per_latent list of R arrays — trained categorical params
        args[12]: latent_cat_specs list of R categorical spec lists
        args[13]: noise_per_task numpy array [T] float32 (optional)
        args[14]: variance_method int (0=mean_only, 1=love, 2=exact, optional)
        args[15]: max_cg_iter int (optional, default 100)
        args[16]: cg_tol float (optional, default 1e-2)
        args[17]: precond_rank int (optional, default 10)
        args[18]: lanczos_rank int (optional, default auto)
        args[19]: materialization_mode int (optional, default 0)
        args[20]: precond_method int (optional, default 0=greedy)

    Returns:
        dict with: mean [m, T] numpy float32, optional variance/std [m, T]
    """
    var np = Python.import_module("numpy")

    var provider_infos = args[0]
    var alpha_np = args[1]
    var A_np = args[2]
    var latent_x_test_np = args[3]
    var params_per_latent_py = args[4]
    var mean_per_task_np = args[5]
    var n_train = Int(args[6].__int__())
    var T = Int(args[7].__int__())
    var latent_is_mixed_py = args[8]
    var latent_C_trains = args[9]
    var latent_C_tests = args[10]
    var cat_params_per_latent_py = args[11]
    var latent_cat_specs_py = args[12]
    var R = Int(len(provider_infos))

    var variance_method = PREDICT_MEAN_ONLY
    var max_cg_iter = 100
    var cg_tol = Float32(1e-2)
    var precond_rank = 10
    var lanczos_rank = 0
    var method_int = 0
    var precond_method = 0
    var has_noise_per_task = len(args) > 13
    if len(args) > 14:
        variance_method = Int(args[14].__int__())
    if len(args) > 15:
        max_cg_iter = Int(args[15].__int__())
    if len(args) > 16:
        cg_tol = py_to_f32(args[16])
    if len(args) > 17:
        precond_rank = Int(args[17].__int__())
    if len(args) > 18:
        lanczos_rank = Int(args[18].__int__())
    if len(args) > 19:
        method_int = Int(args[19].__int__())
    if len(args) > 20:
        precond_method = Int(args[20].__int__())

    if R == 0:
        raise Error("predict_lmc_mixed() requires at least one latent provider")

    var first_x_test_np = latent_x_test_np[0]
    var n_test = Int(first_x_test_np.shape[0])
    if method_int == 2:
        raise Error("method int 2 is not a public MojoGP training route")
    var use_materialized = method_int == 1
    var has_ard_latent = False
    for s in range(R):
        if _info_bool(provider_infos[s], "is_ard"):
            has_ard_latent = True
    lanczos_rank = _resolve_multi_predict_lanczos_rank(
        lanczos_rank, True, use_materialized, has_ard_latent,
    )

    var ctx = DeviceContext()

    var alpha_c = np.ascontiguousarray(alpha_np, dtype=np.float32).ravel()
    var alpha_host = ctx.enqueue_create_host_buffer[float_dtype](n_train * T)
    ctx.synchronize()
    bulk_copy_to_host_buffer(alpha_c, alpha_host, n_train * T)
    var alpha_device = ctx.enqueue_create_buffer[float_dtype](n_train * T)
    ctx.enqueue_copy(alpha_device, alpha_host)
    ctx.synchronize()

    var mean_accum_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
    ctx.synchronize()
    for i in range(n_test * T):
        mean_accum_host.unsafe_ptr()[i] = Float32(0.0)
    var mean_device = ctx.enqueue_create_buffer[float_dtype](n_test * T)
    ctx.enqueue_copy(mean_device, mean_accum_host)
    ctx.synchronize()

    var variance_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var latent_variances_flat = np.zeros(max(R * n_test, 1), dtype=np.float32)
    var can_compute_variance = variance_method != PREDICT_MEAN_ONLY and has_noise_per_task
    var avg_noise = Float32(0.0)
    if can_compute_variance:
        var noise_per_task_c = np.ascontiguousarray(args[13], dtype=np.float32).ravel()
        for t in range(T):
            avg_noise += py_to_f32(noise_per_task_c[t])
        avg_noise /= Float32(T)

        for i in range(n_test * T):
            variance_out_flat[i] = Float32(0.0)
    var latent_var_host = ctx.enqueue_create_host_buffer[float_dtype](max(n_test, 1))
    ctx.synchronize()

    var cross_buffer = ctx.enqueue_create_buffer[float_dtype](n_test * T)
    for s in range(R):
        var info = provider_infos[s]
        var is_mixed_s = Bool(latent_is_mixed_py[s].__bool__())
        var x_test_np_s = latent_x_test_np[s]
        var n_test_s = Int(x_test_np_s.shape[0])
        if n_test_s != n_test:
            raise Error("predict_lmc_mixed() requires matching test row counts across latents")

        var dim_s = Int(x_test_np_s.shape[1])
        var x_test_c_s = np.ascontiguousarray(x_test_np_s, dtype=np.float32)
        var x_test_host_s = ctx.enqueue_create_host_buffer[float_dtype](max(n_test * dim_s, 1))
        ctx.synchronize()
        bulk_copy_to_host_buffer(x_test_c_s.ravel(), x_test_host_s, n_test * dim_s)
        var x_test_device_s = ctx.enqueue_create_buffer[float_dtype](max(n_test * dim_s, 1))
        ctx.enqueue_copy(x_test_device_s, x_test_host_s)
        ctx.synchronize()

        var A_s_c = np.ascontiguousarray(A_np[s], dtype=np.float32)
        var A_s_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
        ctx.synchronize()
        for i in range(T * T):
            A_s_host.unsafe_ptr()[i] = py_to_f32(A_s_c.ravel()[i])

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

        if not is_mixed_s and can_compute_variance and not has_diagtest:
            can_compute_variance = False

        var provider_s = ErasedJITProvider(
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
            mixed_forward_matvec=_cvt_mixed_fwd(Int(info["mixed_forward_matvec"].__int__())),
            mixed_fused_gradient_matvec=_cvt_mixed_fused_grad(Int(info["mixed_fused_gradient_matvec"].__int__())),
            mixed_cross_matvec=_cvt_mixed_cross(Int(info["mixed_cross_matvec"].__int__())),
            mixed_extract_diagonal=_cvt_mixed_diag(Int(info["mixed_extract_diagonal"].__int__())),
            mixed_materialize=_cvt_mixed_mat(Int(info["mixed_materialize"].__int__())),
            has_mixed=True,
        )

        var params_s_py = params_per_latent_py[s]
        var params_s_c = np.ascontiguousarray(params_s_py, dtype=np.float32).ravel()
        var params_s_host = ctx.enqueue_create_host_buffer[float_dtype](max(num_gp, 1))
        ctx.synchronize()
        bulk_copy_to_host_buffer(params_s_c, params_s_host, num_gp)
        provider_s.update_params(params_s_host.unsafe_ptr())

        var cat_state = CategoricalCorrelationState(ctx, List[Int](), List[Int](), n_train)
        var total_cat_params = 0
        var num_cat_vars_s = 0
        var cat_test_host = ctx.enqueue_create_host_buffer[DType.int32](1)
        var cat_test_device = ctx.enqueue_create_buffer[DType.int32](1)
        var cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        var need_mixed_exact_fallback = False

        if is_mixed_s:
            var cat_train_np_s = latent_C_trains[s]
            var cat_test_np_s = latent_C_tests[s]
            var cat_specs_s = latent_cat_specs_py[s]
            var cat_params_c = np.ascontiguousarray(cat_params_per_latent_py[s], dtype=np.float32).flatten()
            num_cat_vars_s = Int(len(cat_specs_s))

            var levels_list = List[Int]()
            var ktypes_list = List[Int]()
            for v in range(num_cat_vars_s):
                var spec = cat_specs_s[v]
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

            cat_state = CategoricalCorrelationState(ctx, levels_list^, ktypes_list^, n_train)
            total_cat_params = cat_state.total_cat_params

            if num_cat_vars_s > 0:
                var cat_train_c = np.ascontiguousarray(cat_train_np_s, dtype=np.int32)
                var cat_data_host = ctx.enqueue_create_host_buffer[DType.int32](n_train * num_cat_vars_s)
                ctx.synchronize()
                for v in range(num_cat_vars_s):
                    for i in range(n_train):
                        cat_data_host.unsafe_ptr()[v * n_train + i] = Int32(Int(cat_train_c[i][v].__int__()))
                cat_state.upload_categorical_data(cat_data_host)
                _ = cat_data_host

            cat_params_host = ctx.enqueue_create_host_buffer[float_dtype](max(total_cat_params, 1))
            ctx.synchronize()
            for i in range(total_cat_params):
                cat_params_host.unsafe_ptr()[i] = py_to_f32(cat_params_c[i])
            if total_cat_params > 0:
                cat_state.update_correlation_matrices(cat_params_host.unsafe_ptr())

            var cat_test_c = np.ascontiguousarray(cat_test_np_s, dtype=np.int32)
            cat_test_host = ctx.enqueue_create_host_buffer[DType.int32](max(n_test * num_cat_vars_s, 1))
            ctx.synchronize()
            for v in range(num_cat_vars_s):
                for i in range(n_test):
                    cat_test_host.unsafe_ptr()[v * n_test + i] = Int32(Int(cat_test_c[i][v].__int__()))
            cat_test_device = ctx.enqueue_create_buffer[DType.int32](max(n_test * num_cat_vars_s, 1))
            ctx.enqueue_copy(cat_test_device, cat_test_host)
            ctx.synchronize()

            var x_test_single_host = ctx.enqueue_create_host_buffer[float_dtype](max(dim_s, 1))
            var x_test_single_device = ctx.enqueue_create_buffer[float_dtype](max(dim_s, 1))
            var cat_test_single_host = ctx.enqueue_create_host_buffer[DType.int32](max(num_cat_vars_s, 1))
            var cat_test_single_device = ctx.enqueue_create_buffer[DType.int32](max(num_cat_vars_s, 1))
            var latent_var_single_host = ctx.enqueue_create_host_buffer[float_dtype](1)

            var mixed_provider = MixedKroneckerBaseProviderView(provider_s^, cat_state)
            if can_compute_variance:
                mixed_provider.update_noise(avg_noise)
            if use_materialized:
                mixed_provider.refresh_materialization()

            mixed_provider.mixed_cross_matvec(
                cross_buffer.unsafe_ptr(),
                x_test_device_s.unsafe_ptr(),
                alpha_device.unsafe_ptr(),
                cat_test_device.unsafe_ptr(),
                n_test,
                T,
            )
            ctx.synchronize()

            if can_compute_variance:
                if variance_method == PREDICT_EXACT:
                    for i in range(n_test):
                        for d in range(dim_s):
                            x_test_single_host.unsafe_ptr()[d] = x_test_host_s.unsafe_ptr()[i * dim_s + d]
                        ctx.enqueue_copy(x_test_single_device, x_test_single_host)
                        for v in range(num_cat_vars_s):
                            cat_test_single_host.unsafe_ptr()[v] = cat_test_host.unsafe_ptr()[v * n_test + i]
                        ctx.enqueue_copy(cat_test_single_device, cat_test_single_host)
                        ctx.synchronize()

                        var latent_var_device = predict_variance_mixed_jit(
                            mixed_provider,
                            ctx,
                            x_test_single_device.unsafe_ptr(),
                            cat_test_single_device.unsafe_ptr(),
                            n_train,
                            1,
                            dim_s,
                            PREDICT_EXACT,
                            max_cg_iter,
                            cg_tol,
                            precond_rank,
                            lanczos_rank,
                            0,
                            use_materialized=use_materialized,
                        )
                        ctx.enqueue_copy(latent_var_single_host, latent_var_device)
                        ctx.synchronize()
                        latent_var_host.unsafe_ptr()[i] = latent_var_single_host.unsafe_ptr()[0]
                        _ = latent_var_device
                else:
                    var latent_var_device = predict_variance_mixed_jit(
                        mixed_provider,
                        ctx,
                        x_test_device_s.unsafe_ptr(),
                        cat_test_device.unsafe_ptr(),
                        n_train,
                        n_test,
                        dim_s,
                        variance_method,
                        max_cg_iter,
                        cg_tol,
                        precond_rank,
                        lanczos_rank,
                        0,
                        use_materialized=use_materialized,
                    )
                    ctx.enqueue_copy(latent_var_host, latent_var_device)
                    ctx.synchronize()

                if variance_method == PREDICT_LOVE:
                    for i in range(n_test):
                        var latent_var_i = latent_var_host.unsafe_ptr()[i]
                        if isnan(latent_var_i) or isinf(latent_var_i):
                            need_mixed_exact_fallback = True
                            break

                    if need_mixed_exact_fallback:
                        for i in range(n_test):
                            for d in range(dim_s):
                                x_test_single_host.unsafe_ptr()[d] = x_test_host_s.unsafe_ptr()[i * dim_s + d]
                            ctx.enqueue_copy(x_test_single_device, x_test_single_host)
                            for v in range(num_cat_vars_s):
                                cat_test_single_host.unsafe_ptr()[v] = cat_test_host.unsafe_ptr()[v * n_test + i]
                            ctx.enqueue_copy(cat_test_single_device, cat_test_single_host)
                            ctx.synchronize()

                            var latent_var_exact_device = predict_variance_mixed_jit(
                                mixed_provider,
                                ctx,
                                x_test_single_device.unsafe_ptr(),
                                cat_test_single_device.unsafe_ptr(),
                                n_train,
                                1,
                                dim_s,
                                PREDICT_EXACT,
                                max_cg_iter,
                                cg_tol,
                                precond_rank,
                                lanczos_rank,
                                0,
                                use_materialized=use_materialized,
                            )
                            ctx.enqueue_copy(latent_var_single_host, latent_var_exact_device)
                            ctx.synchronize()
                            latent_var_host.unsafe_ptr()[i] = latent_var_single_host.unsafe_ptr()[0]
                            _ = latent_var_exact_device

                for i in range(n_test):
                    latent_variances_flat[s * n_test + i] = Float64(latent_var_host.unsafe_ptr()[i])

                for t in range(T):
                    var row_weight = Float32(0.0)
                    for tp in range(T):
                        var a_val = A_s_host.unsafe_ptr()[t * T + tp]
                        row_weight += a_val * a_val
                    for i in range(n_test):
                        var current_var = py_to_f32(variance_out_flat[i * T + t])
                        variance_out_flat[i * T + t] = current_var + row_weight * latent_var_host.unsafe_ptr()[i]

                _ = x_test_single_host
                _ = x_test_single_device
                _ = cat_test_single_host
                _ = cat_test_single_device
                _ = latent_var_single_host
        else:
            provider_s.cross_matvec(
                cross_buffer.unsafe_ptr(),
                x_test_device_s.unsafe_ptr(),
                alpha_device.unsafe_ptr(),
                n_test,
                T,
            )
            ctx.synchronize()

            if can_compute_variance:
                provider_s.update_noise(avg_noise)
                if variance_method == PREDICT_LOVE:
                    var latent_var_device = predict_variance_love_jit(
                        provider_s,
                        provider_s,
                        False,
                        ctx,
                        x_test_device_s.unsafe_ptr(),
                        n_train,
                        n_test,
                        max_cg_iter,
                        cg_tol,
                        precond_rank,
                        lanczos_rank,
                    )
                    ctx.enqueue_copy(latent_var_host, latent_var_device)
                    ctx.synchronize()
                    _ = latent_var_device
                else:
                    var x_test_single_host = ctx.enqueue_create_host_buffer[float_dtype](max(dim_s, 1))
                    var x_test_single_device = ctx.enqueue_create_buffer[float_dtype](max(dim_s, 1))
                    var latent_var_single_host = ctx.enqueue_create_host_buffer[float_dtype](1)
                    for i in range(n_test):
                        for d in range(dim_s):
                            x_test_single_host.unsafe_ptr()[d] = x_test_host_s.unsafe_ptr()[i * dim_s + d]
                        ctx.enqueue_copy(x_test_single_device, x_test_single_host)
                        ctx.synchronize()
                        var exact_block_cols_slot = alloc[Int](1)
                        var exact_cross_mode_slot = alloc[Int](1)
                        exact_block_cols_slot[] = 0
                        exact_cross_mode_slot[] = 0
                        var latent_var_exact_device = predict_variance_exact_jit(
                            provider_s,
                            provider_s,
                            False,
                            ctx,
                            alpha_device.unsafe_ptr(),
                            x_test_single_device.unsafe_ptr(),
                            n_train,
                            1,
                            dim_s,
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
                        ctx.enqueue_copy(latent_var_single_host, latent_var_exact_device)
                        ctx.synchronize()
                        latent_var_host.unsafe_ptr()[i] = latent_var_single_host.unsafe_ptr()[0]
                        _ = latent_var_exact_device
                    _ = x_test_single_host
                    _ = x_test_single_device
                    _ = latent_var_single_host

                for i in range(n_test):
                    latent_variances_flat[s * n_test + i] = Float64(latent_var_host.unsafe_ptr()[i])

                for t in range(T):
                    var row_weight = Float32(0.0)
                    for tp in range(T):
                        var a_val = A_s_host.unsafe_ptr()[t * T + tp]
                        row_weight += a_val * a_val
                    for i in range(n_test):
                        var current_var = py_to_f32(variance_out_flat[i * T + t])
                        variance_out_flat[i * T + t] = current_var + row_weight * latent_var_host.unsafe_ptr()[i]

        ctx.synchronize()

        var cross_s_host = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
        ctx.enqueue_copy(cross_s_host, cross_buffer)
        var mean_host_s = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
        ctx.enqueue_copy(mean_host_s, mean_device)
        ctx.synchronize()

        for t in range(T):
            for tp in range(T):
                var a_val = A_s_host.unsafe_ptr()[t * T + tp]
                for i in range(n_test):
                    var cross_val = cross_s_host.unsafe_ptr()[tp * n_test + i]
                    mean_host_s.unsafe_ptr()[t * n_test + i] += a_val * cross_val

        ctx.enqueue_copy(mean_device, mean_host_s)
        ctx.synchronize()

        _ = params_s_host
        _ = A_s_host
        _ = cross_s_host
        _ = mean_host_s
        _ = x_test_host_s
        _ = x_test_device_s
        _ = cat_params_host
        _ = cat_test_host
        _ = cat_test_device
        _ = cat_state
    
    var mean_host_final = ctx.enqueue_create_host_buffer[float_dtype](n_test * T)
    ctx.enqueue_copy(mean_host_final, mean_device)
    ctx.synchronize()

    var mean_out_flat = np.zeros(n_test * T, dtype=np.float32)
    var mean_per_task_c = np.ascontiguousarray(mean_per_task_np, dtype=np.float32)
    for t in range(T):
        var task_mean_offset = py_to_f32(mean_per_task_c[t])
        for i in range(n_test):
            mean_out_flat[i * T + t] = mean_host_final.unsafe_ptr()[t * n_test + i] + task_mean_offset
    var mean_out = mean_out_flat.reshape(n_test, T)

    var out = Python.dict()
    out["mean"] = mean_out
    out["lanczos_rank_used"] = lanczos_rank
    if can_compute_variance:
        var noise_per_task_c = np.ascontiguousarray(args[13], dtype=np.float32).ravel()
        for t in range(T):
            var noise_t = py_to_f32(noise_per_task_c[t])
            for i in range(n_test):
                var current_var = py_to_f32(variance_out_flat[i * T + t])
                variance_out_flat[i * T + t] = current_var + noise_t

        out["variance"] = variance_out_flat.reshape(n_test, T)
        out["std"] = np.sqrt(out["variance"])
        out["latent_variances"] = latent_variances_flat.reshape(R, n_test)

    _ = alpha_host
    _ = alpha_device
    _ = mean_accum_host
    _ = mean_device
    _ = cross_buffer
    _ = latent_var_host
    _ = mean_host_final

    return out
