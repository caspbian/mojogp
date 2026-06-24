"""JIT mixed LMC (Linear Model of Coregionalization) training helpers.

Extends the continuous-only JIT LMC path to latents whose base kernels are
mixed continuous×categorical kernels. Prediction stays on the Python CPU
fallback path; this module focuses on exact training.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from memory.unsafe_pointer import alloc
from math import sqrt, exp as math_exp, isnan
from time import perf_counter_ns
from python import PythonObject

from kernels.constants import float_dtype
from kernels.constants import (
    CAT_KERNEL_EHH,
    CAT_KERNEL_FE,
    CAT_KERNEL_HH,
    PI,
)
from kernels.jit.jit_training import softplus, inv_softplus, pow_float32
from kernels.jit.jit_multi_output_mixed import MixedKroneckerBaseProviderView
from kernels.categorical_state import CategoricalCorrelationState
from kernels.combined_inv_quad_logdet import (
    bbmm_with_precond,
    CGBufferPool,
    UnifiedBBMMResult,
)
from kernels.pivoted_cholesky import (
    build_pivoted_cholesky_precond_unified,
    PivotedCholeskyPrecond,
)
from kernels.lmc_preconditioner import LMCPreconditioner
from kernels.lmc_provider import (
    kernel_lmc_accumulate_batched,
    kernel_lmc_add_noise_batched,
    kernel_lmc_add_fixed_noise_batched,
    kernel_zero_buffer,
    kernel_lmc_extract_diagonal,
)
from kernels.cg_solver import kernel_copy, kernel_dot_batched
from kernels.gradient_provider import GradientProvider
from kernels.native_numerics import cholesky_decompose
from kernels.jit.jit_progress import emit_progress_event, progress_interval_should_emit


struct LMCMixedJITResult(Movable):
    var final_params_per_latent: List[List[Float32]]
    var final_cat_params_per_latent: List[List[Float32]]
    var num_kernel_params_per_latent: List[Int]
    var A_matrices_flat: List[Float32]
    var L_factors_flat: List[Float32]
    var var_diag_flat: List[Float32]
    var noise_per_task: List[Float32]
    var mean_per_task: List[Float32]
    var alpha_blocked: List[Float32]
    var nll_history: List[Float32]
    var iter_times_ns: List[Int]
    var final_nll: Float32
    var iterations: Int
    var converged: Bool
    var num_tasks: Int
    var num_latents: Int
    var n: Int
    var precond_rebuild_count: Int

    fn __init__(
        out self,
        owned final_params_per_latent: List[List[Float32]],
        owned final_cat_params_per_latent: List[List[Float32]],
        owned num_kernel_params_per_latent: List[Int],
        owned A_matrices_flat: List[Float32],
        owned L_factors_flat: List[Float32],
        owned var_diag_flat: List[Float32],
        owned noise_per_task: List[Float32],
        owned mean_per_task: List[Float32],
        owned alpha_blocked: List[Float32],
        owned nll_history: List[Float32],
        owned iter_times_ns: List[Int],
        final_nll: Float32,
        iterations: Int,
        converged: Bool,
        num_tasks: Int,
        num_latents: Int,
        n: Int,
        precond_rebuild_count: Int,
    ):
        self.final_params_per_latent = final_params_per_latent^
        self.final_cat_params_per_latent = final_cat_params_per_latent^
        self.num_kernel_params_per_latent = num_kernel_params_per_latent^
        self.A_matrices_flat = A_matrices_flat^
        self.L_factors_flat = L_factors_flat^
        self.var_diag_flat = var_diag_flat^
        self.noise_per_task = noise_per_task^
        self.mean_per_task = mean_per_task^
        self.alpha_blocked = alpha_blocked^
        self.nll_history = nll_history^
        self.iter_times_ns = iter_times_ns^
        self.final_nll = final_nll
        self.iterations = iterations
        self.converged = converged
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.n = n
        self.precond_rebuild_count = precond_rebuild_count

    fn __moveinit__(out self, owned other: Self):
        self.final_params_per_latent = other.final_params_per_latent^
        self.final_cat_params_per_latent = other.final_cat_params_per_latent^
        self.num_kernel_params_per_latent = other.num_kernel_params_per_latent^
        self.A_matrices_flat = other.A_matrices_flat^
        self.L_factors_flat = other.L_factors_flat^
        self.var_diag_flat = other.var_diag_flat^
        self.noise_per_task = other.noise_per_task^
        self.mean_per_task = other.mean_per_task^
        self.alpha_blocked = other.alpha_blocked^
        self.nll_history = other.nll_history^
        self.iter_times_ns = other.iter_times_ns^
        self.final_nll = other.final_nll
        self.iterations = other.iterations
        self.converged = other.converged
        self.num_tasks = other.num_tasks
        self.num_latents = other.num_latents
        self.n = other.n
        self.precond_rebuild_count = other.precond_rebuild_count


struct JITLMCMixedGradientAdapter(GradientProvider, Movable):
    var _providers: UnsafePointer[MixedKroneckerBaseProviderView, MutAnyOrigin]
    var _num_latents: Int
    var _num_tasks: Int
    var _n_data: Int
    var _param_offsets: UnsafePointer[Int, MutAnyOrigin]
    var _total_gradient_params: Int
    var _A_host: HostBuffer[float_dtype]
    var _A_device: DeviceBuffer[float_dtype]
    var _noise_host: HostBuffer[float_dtype]
    var _noise_device: DeviceBuffer[float_dtype]
    var _has_fixed_noise: Bool
    var _fixed_noise_host: HostBuffer[float_dtype]
    var _fixed_noise_device: DeviceBuffer[float_dtype]
    var _temp_kx_v: DeviceBuffer[float_dtype]
    var _ctx: DeviceContext

    fn __init__(
        out self,
        ctx: DeviceContext,
        providers: UnsafePointer[MixedKroneckerBaseProviderView, MutAnyOrigin],
        num_latents: Int,
        num_tasks: Int,
        n_data: Int,
        A_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        noise_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        fixed_noise_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        has_fixed_noise: Bool,
        max_num_cols: Int,
    ) raises:
        self._ctx = ctx
        self._providers = providers
        self._num_latents = num_latents
        self._num_tasks = num_tasks
        self._n_data = n_data

        self._param_offsets = alloc[Int](num_latents + 1)
        self._param_offsets[0] = 0
        for s in range(num_latents):
            var n_params_s = providers[s].num_gradient_params()
            self._param_offsets[s + 1] = self._param_offsets[s] + n_params_s
        self._total_gradient_params = self._param_offsets[num_latents]

        var A_size = num_latents * num_tasks * num_tasks
        self._A_host = HostBuffer[float_dtype](ctx, A_size)
        self._A_device = ctx.enqueue_create_buffer[float_dtype](A_size)
        for i in range(A_size):
            self._A_host.unsafe_ptr()[i] = A_host_ptr[i]
        self._A_device.enqueue_copy_from(self._A_host)

        self._noise_host = HostBuffer[float_dtype](ctx, num_tasks)
        self._noise_device = ctx.enqueue_create_buffer[float_dtype](num_tasks)
        for t in range(num_tasks):
            self._noise_host.unsafe_ptr()[t] = noise_host_ptr[t]
        self._noise_device.enqueue_copy_from(self._noise_host)

        self._has_fixed_noise = has_fixed_noise
        var nT = n_data * num_tasks
        self._fixed_noise_host = HostBuffer[float_dtype](ctx, nT)
        self._fixed_noise_device = ctx.enqueue_create_buffer[float_dtype](nT)
        for i in range(nT):
            if has_fixed_noise:
                self._fixed_noise_host.unsafe_ptr()[i] = fixed_noise_host_ptr[i]
            else:
                self._fixed_noise_host.unsafe_ptr()[i] = Float32(0.0)
        self._fixed_noise_device.enqueue_copy_from(self._fixed_noise_host)

        self._temp_kx_v = ctx.enqueue_create_buffer[float_dtype](n_data * num_tasks * max_num_cols)
        ctx.synchronize()

    fn __moveinit__(out self, owned other: Self):
        self._ctx = other._ctx
        self._providers = other._providers
        self._num_latents = other._num_latents
        self._num_tasks = other._num_tasks
        self._n_data = other._n_data
        self._param_offsets = other._param_offsets
        self._total_gradient_params = other._total_gradient_params
        self._A_host = other._A_host^
        self._A_device = other._A_device^
        self._noise_host = other._noise_host^
        self._noise_device = other._noise_device^
        self._has_fixed_noise = other._has_fixed_noise
        self._fixed_noise_host = other._fixed_noise_host^
        self._fixed_noise_device = other._fixed_noise_device^
        self._temp_kx_v = other._temp_kx_v^
        other._param_offsets = UnsafePointer[Int, MutAnyOrigin]()
        other._num_latents = 0

    fn __del__(owned self):
        if self._param_offsets:
            self._param_offsets.free()

    fn _find_latent_for_param(self, param_index: Int) -> Tuple[Int, Int]:
        for s in range(self._num_latents):
            if param_index < self._param_offsets[s + 1]:
                return (s, param_index - self._param_offsets[s])
        return (self._num_latents - 1, 0)

    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        var n = self._n_data
        var T = self._num_tasks
        var nT = n * T
        alias BLOCK = 256
        var total = num_cols * nT
        var num_blocks = (total + BLOCK - 1) // BLOCK
        var TT = T * T

        self._ctx.enqueue_function[kernel_zero_buffer](
            out_ptr, total,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )

        for s in range(self._num_latents):
            self._providers[s].forward_matvec(
                self._temp_kx_v.unsafe_ptr(), v_ptr, T * num_cols
            )
            self._ctx.enqueue_function[kernel_lmc_accumulate_batched](
                out_ptr,
                self._temp_kx_v.unsafe_ptr(),
                self._A_device.unsafe_ptr().offset(s * TT),
                n,
                T,
                nT,
                num_cols,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK,),
            )

        self._ctx.enqueue_function[kernel_lmc_add_noise_batched](
            out_ptr,
            v_ptr,
            self._noise_device.unsafe_ptr(),
            n,
            T,
            nT,
            num_cols,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK,),
        )
        if self._has_fixed_noise:
            self._ctx.enqueue_function[kernel_lmc_add_fixed_noise_batched](
                out_ptr,
                v_ptr,
                self._fixed_noise_device.unsafe_ptr(),
                n,
                T,
                nT,
                num_cols,
                grid_dim=(num_blocks,),
                block_dim=(BLOCK,),
            )

    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        var n = self._n_data
        var T = self._num_tasks
        var nT = n * T
        alias BLOCK = 256
        var total = num_cols * nT
        var num_blocks = (total + BLOCK - 1) // BLOCK
        var latent_local = self._find_latent_for_param(param_index)
        var s = latent_local[0]
        var p_local = latent_local[1]
        var TT = T * T

        self._providers[s].gradient_matvec(
            self._temp_kx_v.unsafe_ptr(), v_ptr, T * num_cols, p_local, False
        )

        self._ctx.enqueue_function[kernel_zero_buffer](
            out_ptr, total,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        self._ctx.enqueue_function[kernel_lmc_accumulate_batched](
            out_ptr,
            self._temp_kx_v.unsafe_ptr(),
            self._A_device.unsafe_ptr().offset(s * TT),
            n,
            T,
            nT,
            num_cols,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK,),
        )

        if sync:
            self._ctx.synchronize()

    fn num_gradient_params(self) -> Int:
        return self._total_gradient_params

    fn get_n(self) -> Int:
        return self._n_data * self._num_tasks

    fn get_ctx(self) -> DeviceContext:
        return self._ctx

    fn get_noise(self) -> Float32:
        var min_noise = self._noise_host.unsafe_ptr()[0]
        for t in range(1, self._num_tasks):
            if self._noise_host.unsafe_ptr()[t] < min_noise:
                min_noise = self._noise_host.unsafe_ptr()[t]
        return min_noise

    fn get_diagonal_value(self) -> Float32:
        var TT = self._num_tasks * self._num_tasks
        var max_val = Float32(0.0)
        for t in range(self._num_tasks):
            var diag_t = Float32(0.0)
            for s in range(self._num_latents):
                diag_t += self._A_host.unsafe_ptr()[s * TT + t * self._num_tasks + t]
            if diag_t > max_val:
                max_val = diag_t
        return max_val

    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        var nT = self._n_data * self._num_tasks
        alias BLOCK = 256
        var num_blocks = (nT + BLOCK - 1) // BLOCK
        self._ctx.enqueue_function[kernel_lmc_extract_diagonal](
            diag_ptr,
            self._A_device.unsafe_ptr(),
            self._num_latents,
            self._n_data,
            self._num_tasks,
            nT,
            grid_dim=(num_blocks,),
            block_dim=(BLOCK,),
        )

    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self._providers[0].get_x_ptr()

    fn supports_fused_gradient(self) -> Bool:
        return False

    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        raise Error("fused_gradient_matvec not supported for mixed JIT LMC")

    fn supports_fused_ls_os(self) -> Bool:
        return False

    fn fused_ls_os_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        raise Error("fused_ls_os not supported for mixed JIT LMC")

    fn supports_fused_3param(self) -> Bool:
        return False

    fn fused_3param_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        raise Error("fused_3param not supported for mixed JIT LMC")

    fn get_param_offset(self, latent_index: Int) -> Int:
        return self._param_offsets[latent_index]


fn _compute_A_from_L(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    A_ptr: UnsafePointer[Float32, MutAnyOrigin],
    T: Int,
    var_diag_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
):
    for i in range(T):
        for j in range(T):
            var val = Float32(0.0)
            var max_k = i if i < j else j
            for k in range(max_k + 1):
                val += L_ptr[i * T + k] * L_ptr[j * T + k]
            if i == j and var_diag_ptr:
                val += var_diag_ptr[i]
            A_ptr[i * T + j] = val


fn _cholesky_of_A(
    A_ptr: UnsafePointer[Float32, MutAnyOrigin],
    C_ptr: UnsafePointer[Float32, MutAnyOrigin],
    T: Int,
) -> Bool:
    return cholesky_decompose(A_ptr, T, C_ptr)


fn _softplus_derivative(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + math_exp(-x))


fn _sigmoid_lmc_mixed(x: Float32) -> Float32:
    return _softplus_derivative(x)


fn _sigmoid_derivative_lmc_mixed(x: Float32) -> Float32:
    var s = _sigmoid_lmc_mixed(x)
    return s * (Float32(1.0) - s)


fn _constrain_cat_raw_lmc_mixed(
    raw: Float32,
    kernel_type: Int,
    levels: Int,
    local_param_index: Int,
) -> Float32:
    if kernel_type == CAT_KERNEL_EHH or kernel_type == CAT_KERNEL_HH:
        return _sigmoid_lmc_mixed(raw) * Float32(PI)
    elif kernel_type == CAT_KERNEL_FE:
        var num_angles = levels * (levels - 1) // 2
        if local_param_index < num_angles:
            return _sigmoid_lmc_mixed(raw) * Float32(PI)
        return softplus(raw)
    return softplus(raw)


fn _cat_chain_derivative_lmc_mixed(
    raw: Float32,
    kernel_type: Int,
    levels: Int,
    local_param_index: Int,
) -> Float32:
    if kernel_type == CAT_KERNEL_EHH or kernel_type == CAT_KERNEL_HH:
        return _sigmoid_derivative_lmc_mixed(raw) * Float32(PI)
    elif kernel_type == CAT_KERNEL_FE:
        var num_angles = levels * (levels - 1) // 2
        if local_param_index < num_angles:
            return _sigmoid_derivative_lmc_mixed(raw) * Float32(PI)
        return _softplus_derivative(raw)
    return _softplus_derivative(raw)


fn _cat_param_local_info_lmc_mixed(
    read cat_state: CategoricalCorrelationState,
    param_index: Int,
) -> Tuple[Int, Int, Int]:
    var local_index = param_index
    for var_idx in range(cat_state.num_cat_vars):
        var levels = cat_state.levels[var_idx]
        var n_params = cat_state.get_num_params_for_var(var_idx)
        if local_index < n_params:
            return (cat_state.kernel_types[var_idx], levels, local_index)
        local_index -= n_params
    return (CAT_KERNEL_EHH, 0, 0)


fn _constrained_cat_param_lmc_mixed(
    read cat_state: CategoricalCorrelationState,
    read raw_cat: List[Float32],
    param_index: Int,
) -> Float32:
    var info = _cat_param_local_info_lmc_mixed(cat_state, param_index)
    return _constrain_cat_raw_lmc_mixed(raw_cat[param_index], info[0], info[1], info[2])


fn _cat_chain_derivative_for_param_lmc_mixed(
    read cat_state: CategoricalCorrelationState,
    read raw_cat: List[Float32],
    param_index: Int,
) -> Float32:
    var info = _cat_param_local_info_lmc_mixed(cat_state, param_index)
    return _cat_chain_derivative_lmc_mixed(raw_cat[param_index], info[0], info[1], info[2])


fn _write_constrained_cat_params_lmc_mixed(
    read cat_state: CategoricalCorrelationState,
    read raw_cat: List[Float32],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    for k in range(len(raw_cat)):
        out_ptr[k] = _constrained_cat_param_lmc_mixed(cat_state, raw_cat, k)


fn _apply_cat_gradient_lmc(
    mixed_provider: MixedKroneckerBaseProviderView,
    A_s_ptr: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    temp_base_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    T: Int,
    num_cols: Int,
    grad_corr_ptr: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    alias BLOCK = 256
    var nT = n * T
    var total = nT * num_cols
    mixed_provider.provider.get_ctx().enqueue_function[kernel_zero_buffer](
        out_ptr,
        total,
        grid_dim=(((total + BLOCK - 1) // BLOCK),),
        block_dim=(BLOCK,),
    )
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
    mixed_provider.provider.get_ctx().enqueue_function[kernel_lmc_accumulate_batched](
        out_ptr,
        temp_base_ptr,
        A_s_ptr,
        n,
        T,
        nT,
        num_cols,
        grid_dim=(((total + BLOCK - 1) // BLOCK),),
        block_dim=(BLOCK,),
    )
    mixed_provider.provider.get_ctx().synchronize()


fn _compute_A_and_noise_gradients_mixed_jit(
    ctx: DeviceContext,
    providers: UnsafePointer[MixedKroneckerBaseProviderView, MutAnyOrigin],
    bbmm_result: UnifiedBBMMResult,
    R: Int,
    T: Int,
    n: Int,
    nT: Int,
    num_probes: Int,
    G_A_out: UnsafePointer[Float32, MutAnyOrigin],
    grad_noise_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var TT = T * T
    var dot_result_device = ctx.enqueue_create_buffer[float_dtype](1)
    var dot_result_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    var temp_v = ctx.enqueue_create_buffer[float_dtype](n)
    var kx_alpha = ctx.enqueue_create_buffer[float_dtype](n * T)
    var kx_rf_j = ctx.enqueue_create_buffer[float_dtype](n)

    for s in range(R):
        for t in range(T):
            ctx.enqueue_function[kernel_copy](
                temp_v.unsafe_ptr(),
                bbmm_result.solution.unsafe_ptr().offset(t * n),
                n,
                grid_dim=((n + 255) // 256,),
                block_dim=(256,),
            )
            providers[s].forward_matvec(kx_alpha.unsafe_ptr().offset(t * n), temp_v.unsafe_ptr(), 1)

        for i in range(T):
            for j in range(T):
                ctx.enqueue_function[kernel_dot_batched](
                    bbmm_result.solution.unsafe_ptr().offset(i * n),
                    kx_alpha.unsafe_ptr().offset(j * n),
                    dot_result_device.unsafe_ptr(),
                    n,
                    1,
                    grid_dim=(1, 1),
                    block_dim=(256, 1),
                )
                ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
                ctx.synchronize()
                G_A_out[s * TT + i * T + j] = -dot_result_host[0]

        for k in range(num_probes):
            for j in range(T):
                ctx.enqueue_function[kernel_copy](
                    temp_v.unsafe_ptr(),
                    bbmm_result.right_factors.unsafe_ptr().offset(k * nT + j * n),
                    n,
                    grid_dim=((n + 255) // 256,),
                    block_dim=(256,),
                )
                providers[s].forward_matvec(kx_rf_j.unsafe_ptr(), temp_v.unsafe_ptr(), 1)

                for i in range(T):
                    ctx.enqueue_function[kernel_dot_batched](
                        bbmm_result.probe_solutions.unsafe_ptr().offset(k * nT + i * n),
                        kx_rf_j.unsafe_ptr(),
                        dot_result_device.unsafe_ptr(),
                        n,
                        1,
                        grid_dim=(1, 1),
                        block_dim=(256, 1),
                    )
                    ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
                    ctx.synchronize()
                    G_A_out[s * TT + i * T + j] += dot_result_host[0] / Float32(num_probes)

        for idx in range(TT):
            G_A_out[s * TT + idx] = Float32(0.5) * G_A_out[s * TT + idx] / Float32(nT)

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
        ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
        ctx.synchronize()
        var data_term = -dot_result_host[0]

        var trace_sum = Float32(0.0)
        for k in range(num_probes):
            ctx.enqueue_function[kernel_dot_batched](
                bbmm_result.probe_solutions.unsafe_ptr().offset(k * nT + t * n),
                bbmm_result.right_factors.unsafe_ptr().offset(k * nT + t * n),
                dot_result_device.unsafe_ptr(),
                n,
                1,
                grid_dim=(1, 1),
                block_dim=(256, 1),
            )
            ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
            ctx.synchronize()
            trace_sum += dot_result_host[0]

        grad_noise_out[t] = Float32(0.5) * (data_term + trace_sum / Float32(num_probes)) / Float32(nT)

    _ = dot_result_device
    _ = dot_result_host
    _ = temp_v
    _ = kx_alpha
    _ = kx_rf_j


fn train_lmc_mixed_jit(
    providers: UnsafePointer[MixedKroneckerBaseProviderView, MutAnyOrigin],
    cat_states: UnsafePointer[CategoricalCorrelationState, MutAnyOrigin],
    ctx: DeviceContext,
    y_blocked_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
    num_latents: Int,
    params_per_latent: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin]],
    num_params_per_latent: UnsafePointer[Int],
    cat_params_per_latent: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin]],
    num_cat_params_per_latent: UnsafePointer[Int],
    initial_noise_per_task_ptr: UnsafePointer[Float32, MutAnyOrigin],
    fixed_observation_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    has_fixed_observation_noise: Bool,
    initial_mean_per_task_ptr: UnsafePointer[Float32, MutAnyOrigin],
    max_iterations: Int = 100,
    learning_rate: Float32 = Float32(0.05),
    num_probes: Int = 10,
    max_cg_iter: Int = 200,
    max_tridiag_iter: Int = 30,
    cg_tol: Float32 = Float32(1.0),
    precond_rank: Int = 15,
    precond_method: Int = 0,
    precond_rebuild_threshold: Float32 = Float32(0.5),
    early_stop_tol: Float32 = Float32(1e-4),
    early_stop_patience: Int = 15,
    verbose: Bool = True,
    use_materialized: Bool = False,
    progress_callback: PythonObject = None,
    progress_interval: Int = 1,
    progress_enabled: Bool = False,
) raises -> LMCMixedJITResult:
    var T = num_tasks
    var R = num_latents
    var nT = n * T
    var TT = T * T
    var num_cols_total = 1 + num_probes
    var precond_error_tol = Float32(1e-3)
    var use_preconditioner = precond_rank > 0
    if precond_method == 0:
        precond_error_tol = Float32(0.0)

    var raw_params_per_latent = List[List[Float32]]()
    for s in range(R):
        var raw_s = List[Float32]()
        for p in range(num_params_per_latent[s]):
            raw_s.append(inv_softplus(params_per_latent[s][p]))
        raw_params_per_latent.append(raw_s^)

    var raw_cat_per_latent = List[List[Float32]]()
    for s in range(R):
        var raw_cat_s = List[Float32]()
        for k in range(num_cat_params_per_latent[s]):
            raw_cat_s.append(cat_params_per_latent[s][k])
        raw_cat_per_latent.append(raw_cat_s^)

    var raw_noise = List[Float32]()
    for t in range(T):
        raw_noise.append(inv_softplus(initial_noise_per_task_ptr[t]))

    var raw_mean = List[Float32]()
    for t in range(T):
        raw_mean.append(initial_mean_per_task_ptr[t])

    var init_diag = sqrt(Float32(1.0) / Float32(R))
    var L_all = alloc[Float32](R * TT)
    var raw_L_diag = alloc[Float32](R * T)
    for s in range(R):
        for i in range(T):
            for j in range(T):
                if i == j:
                    L_all[s * TT + i * T + j] = init_diag
                elif j < i:
                    L_all[s * TT + i * T + j] = Float32(0.01) * Float32(s + 1) / Float32(R)
                else:
                    L_all[s * TT + i * T + j] = Float32(0.0)
            raw_L_diag[s * T + i] = inv_softplus(init_diag)

    var raw_var_diag = alloc[Float32](R * T)
    var init_var_diag_raw = inv_softplus(Float32(0.1))
    for i in range(R * T):
        raw_var_diag[i] = init_var_diag_raw

    var A_all = alloc[Float32](R * TT)
    var C_all = alloc[Float32](R * TT)
    var init_var_diag_buf = alloc[Float32](T)
    for i in range(T):
        init_var_diag_buf[i] = Float32(0.1)
    for s in range(R):
        _compute_A_from_L(L_all + s * TT, A_all + s * TT, T, init_var_diag_buf)
        var ok = _cholesky_of_A(A_all + s * TT, C_all + s * TT, T)
        if not ok:
            for i in range(T):
                A_all[s * TT + i * T + i] += Float32(1e-4)
            _ = _cholesky_of_A(A_all + s * TT, C_all + s * TT, T)
    init_var_diag_buf.free()

    var total_kernel_params = 0
    for s in range(R):
        total_kernel_params += num_params_per_latent[s]
    var total_adam = total_kernel_params + T + R * TT + R * T + T
    var m_adam = alloc[Float32](total_adam)
    var v_adam = alloc[Float32](total_adam)
    for i in range(total_adam):
        m_adam[i] = Float32(0.0)
        v_adam[i] = Float32(0.0)

    var m_cat = List[List[Float32]]()
    var v_cat = List[List[Float32]]()
    for s in range(R):
        var ms = List[Float32]()
        var vs = List[Float32]()
        for _ in range(num_cat_params_per_latent[s]):
            ms.append(Float32(0.0))
            vs.append(Float32(0.0))
        m_cat.append(ms^)
        v_cat.append(vs^)

    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var eps = Float32(1e-8)
    var grad_clip_val = Float32(10.0)
    var t_step = 0

    var nll_history = List[Float32]()
    var iter_times_ns = List[Int]()
    var patience_counter = 0
    var converged = False
    var actual_iters = 0
    var best_nll = Float32(1e30)
    var precond_rebuild_count = 0

    var best_raw_params = List[List[Float32]]()
    var best_raw_cat = List[List[Float32]]()
    for s in range(R):
        var bp = List[Float32]()
        for p in range(num_params_per_latent[s]):
            bp.append(raw_params_per_latent[s][p])
        best_raw_params.append(bp^)

        var bc = List[Float32]()
        for k in range(num_cat_params_per_latent[s]):
            bc.append(raw_cat_per_latent[s][k])
        best_raw_cat.append(bc^)

    var best_raw_noise = List[Float32]()
    var best_raw_mean = List[Float32]()
    for t in range(T):
        best_raw_noise.append(raw_noise[t])
        best_raw_mean.append(raw_mean[t])

    var best_L_all = alloc[Float32](R * TT)
    var best_raw_L_diag = alloc[Float32](R * T)
    var best_raw_var_diag = alloc[Float32](R * T)
    for i in range(R * TT):
        best_L_all[i] = L_all[i]
    for i in range(R * T):
        best_raw_L_diag[i] = raw_L_diag[i]
        best_raw_var_diag[i] = raw_var_diag[i]

    var best_sol_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    ctx.synchronize()
    for i in range(nT):
        best_sol_host[i] = Float32(0.0)

    var max_params = 0
    var max_cat_params = 0
    for s in range(R):
        if num_params_per_latent[s] > max_params:
            max_params = num_params_per_latent[s]
        if num_cat_params_per_latent[s] > max_cat_params:
            max_cat_params = num_cat_params_per_latent[s]
    var params_buf = ctx.enqueue_create_host_buffer[float_dtype](max(max_params, 1))
    var cat_params_buf = ctx.enqueue_create_host_buffer[float_dtype](max(max_cat_params, 1))
    ctx.synchronize()

    var cg_pool = CGBufferPool(ctx, nT, num_cols_total)

    var y_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    var y_original_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    for i in range(nT):
        y_host[i] = y_blocked_host_ptr[i]
        y_original_host[i] = y_blocked_host_ptr[i]
    var y_blocked_device = ctx.enqueue_create_buffer[float_dtype](nT)
    y_blocked_device.enqueue_copy_from(y_host)
    ctx.synchronize()

    var sol_host_buf = ctx.enqueue_create_host_buffer[float_dtype](nT)
    var probe_sol_host_buf = ctx.enqueue_create_host_buffer[float_dtype](max(nT * num_probes, 1))
    var out_alpha_device = ctx.enqueue_create_buffer[float_dtype](nT)
    var out_alpha_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    var out_probes_device = ctx.enqueue_create_buffer[float_dtype](max(nT * num_probes, 1))
    var out_probes_host = ctx.enqueue_create_host_buffer[float_dtype](max(nT * num_probes, 1))
    var cat_temp_base = ctx.enqueue_create_buffer[float_dtype](max(nT * num_probes, nT))

    fn _copy_pc_factors_to_device(
        pc_holders: UnsafePointer[PivotedCholeskyPrecond, MutAnyOrigin],
        L_pc_device: DeviceBuffer[float_dtype],
        actual_rank: Int,
    ) raises:
        for s in range(R):
            var L_host_temp = ctx.enqueue_create_host_buffer[float_dtype](n * actual_rank)
            ctx.enqueue_copy(dst_buf=L_host_temp, src_buf=pc_holders[s].L)
            ctx.synchronize()
            var L_all_host_temp = ctx.enqueue_create_host_buffer[float_dtype](R * n * actual_rank)
            ctx.enqueue_copy(dst_buf=L_all_host_temp, src_buf=L_pc_device)
            ctx.synchronize()
            for i in range(n * actual_rank):
                L_all_host_temp[s * n * actual_rank + i] = L_host_temp[i]
            ctx.enqueue_copy(dst_buf=L_pc_device, src_buf=L_all_host_temp)
            ctx.synchronize()

    for s in range(R):
        for p in range(num_params_per_latent[s]):
            params_buf[p] = softplus(raw_params_per_latent[s][p])
        providers.offset(s)[].update_params(params_buf.unsafe_ptr())
        providers.offset(s)[].update_noise(Float32(0.0))

        if num_cat_params_per_latent[s] > 0:
            _write_constrained_cat_params_lmc_mixed(
                cat_states.offset(s)[], raw_cat_per_latent[s], cat_params_buf.unsafe_ptr()
            )
            cat_states.offset(s)[].update_correlation_matrices(cat_params_buf.unsafe_ptr())

        if use_materialized:
            providers.offset(s)[].refresh_materialization()

    var pc_holders = alloc[PivotedCholeskyPrecond](R)
    for s in range(R):
        var pc_base = providers.offset(s)[].copy()
        pc_base.update_noise(Float32(0.0))
        if use_materialized:
            pc_base.refresh_materialization()
        var pc = build_pivoted_cholesky_precond_unified(
            pc_base,
            precond_rank,
            error_tol=precond_error_tol,
            max_num_cols=num_cols_total,
            precond_method=precond_method,
        )
        (pc_holders + s).init_pointee_move(pc^)
        _ = pc_base

    var actual_rank = pc_holders[0].rank
    var L_pc_device = ctx.enqueue_create_buffer[float_dtype](R * n * actual_rank)
    _copy_pc_factors_to_device(pc_holders, L_pc_device, actual_rank)

    var noise_host_precond = HostBuffer[float_dtype](ctx, T)
    for t in range(T):
        noise_host_precond.unsafe_ptr()[t] = initial_noise_per_task_ptr[t]

    var C_all_host = HostBuffer[float_dtype](ctx, R * TT)
    for i in range(R * TT):
        C_all_host.unsafe_ptr()[i] = C_all[i]

    var lmc_precond = LMCPreconditioner(
        ctx,
        L_pc_device,
        C_all_host,
        noise_host_precond,
        n,
        actual_rank,
        T,
        R,
        max_num_cols=num_cols_total,
    )

    var last_rebuild_kernel_params = alloc[Float32](max(total_kernel_params, 1))
    var kernel_offset = 0
    for s in range(R):
        for p in range(num_params_per_latent[s]):
            last_rebuild_kernel_params[kernel_offset + p] = params_per_latent[s][p]
        kernel_offset += num_params_per_latent[s]

    var last_rebuild_cat = List[List[Float32]]()
    for s in range(R):
        var last_cat_s = List[Float32]()
        for k in range(num_cat_params_per_latent[s]):
            last_cat_s.append(_constrained_cat_param_lmc_mixed(cat_states.offset(s)[], raw_cat_per_latent[s], k))
        last_rebuild_cat.append(last_cat_s^)

    var last_rebuild_A_all = alloc[Float32](R * TT)
    for i in range(R * TT):
        last_rebuild_A_all[i] = A_all[i]

    var last_rebuild_noise = alloc[Float32](T)
    for t in range(T):
        last_rebuild_noise[t] = initial_noise_per_task_ptr[t]

    fn _clip(val: Float32) -> Float32:
        if val > grad_clip_val:
            return grad_clip_val
        elif val < -grad_clip_val:
            return -grad_clip_val
        return val

    fn _adam_update(
        m_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        idx: Int,
        step: Int,
        g: Float32,
    ) -> Float32:
        var m = beta1 * m_ptr[idx] + (Float32(1.0) - beta1) * g
        var v = beta2 * v_ptr[idx] + (Float32(1.0) - beta2) * g * g
        m_ptr[idx] = m
        v_ptr[idx] = v
        var mh = m / (Float32(1.0) - pow_float32(beta1, step))
        var vh = v / (Float32(1.0) - pow_float32(beta2, step))
        return -learning_rate * mh / (sqrt(vh) + eps)

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "multi_output_lmc",
            "materialized" if use_materialized else "matrix_free",
            "start",
            0,
            max_iterations,
            precond_rank=actual_rank,
            precond_rebuild_count=precond_rebuild_count,
        )

    for iteration in range(max_iterations):
        actual_iters = iteration + 1
        t_step += 1
        ctx.synchronize()
        var iter_start = perf_counter_ns()

        for s in range(R):
            for p in range(num_params_per_latent[s]):
                params_buf[p] = softplus(raw_params_per_latent[s][p])
            providers.offset(s)[].update_params(params_buf.unsafe_ptr())
            providers.offset(s)[].update_noise(Float32(0.0))

            if num_cat_params_per_latent[s] > 0:
                _write_constrained_cat_params_lmc_mixed(
                    cat_states.offset(s)[], raw_cat_per_latent[s], cat_params_buf.unsafe_ptr()
                )
                cat_states.offset(s)[].update_correlation_matrices(cat_params_buf.unsafe_ptr())

            if use_materialized:
                providers.offset(s)[].refresh_materialization()

        var current_noise = alloc[Float32](T)
        for t in range(T):
            current_noise[t] = softplus(raw_noise[t])

        for t in range(T):
            for i in range(n):
                y_host[t * n + i] = y_original_host[t * n + i] - raw_mean[t]
        y_blocked_device.enqueue_copy_from(y_host)
        ctx.synchronize()

        var var_diag_cur = alloc[Float32](R * T)
        for si in range(R * T):
            var_diag_cur[si] = softplus(raw_var_diag[si])
        for s in range(R):
            for i in range(T):
                L_all[s * TT + i * T + i] = softplus(raw_L_diag[s * T + i])
            _compute_A_from_L(L_all + s * TT, A_all + s * TT, T, var_diag_cur + s * T)
            var ok = _cholesky_of_A(A_all + s * TT, C_all + s * TT, T)
            if not ok:
                for i in range(T):
                    A_all[s * TT + i * T + i] += Float32(1e-4)
                _ = _cholesky_of_A(A_all + s * TT, C_all + s * TT, T)
        var_diag_cur.free()

        var lmc_adapter = JITLMCMixedGradientAdapter(
            ctx, providers, R, T, n, A_all, current_noise,
            fixed_observation_noise_ptr, has_fixed_observation_noise,
            num_cols_total,
        )

        var should_rebuild = False
        var max_rel = Float32(0.0)
        var kernel_offset_cur = 0
        for s in range(R):
            for p in range(num_params_per_latent[s]):
                var cur_val = softplus(raw_params_per_latent[s][p])
                var prev_val = last_rebuild_kernel_params[kernel_offset_cur + p]
                var rel = abs(cur_val - prev_val) / (abs(prev_val) + Float32(1e-8))
                if rel > max_rel:
                    max_rel = rel
                if rel > precond_rebuild_threshold:
                    should_rebuild = True
            kernel_offset_cur += num_params_per_latent[s]

        if not should_rebuild:
            for s in range(R):
                for k in range(num_cat_params_per_latent[s]):
                    var prev_cat = last_rebuild_cat[s][k]
                    var cur_cat = _constrained_cat_param_lmc_mixed(
                        cat_states.offset(s)[], raw_cat_per_latent[s], k
                    )
                    var rel = abs(cur_cat - prev_cat) / (abs(prev_cat) + Float32(1e-8))
                    if rel > max_rel:
                        max_rel = rel
                    if rel > precond_rebuild_threshold:
                        should_rebuild = True
                        break
                if should_rebuild:
                    break

        if not should_rebuild:
            for i in range(R * TT):
                var prev_a = last_rebuild_A_all[i]
                var rel = abs(A_all[i] - prev_a) / (abs(prev_a) + Float32(1e-8))
                if rel > max_rel:
                    max_rel = rel
                if rel > precond_rebuild_threshold:
                    should_rebuild = True
                    break

        if not should_rebuild:
            for t in range(T):
                var prev_noise = last_rebuild_noise[t]
                var rel = abs(current_noise[t] - prev_noise) / (abs(prev_noise) + Float32(1e-8))
                if rel > max_rel:
                    max_rel = rel
                if rel > precond_rebuild_threshold:
                    should_rebuild = True
                    break

        if should_rebuild:
            precond_rebuild_count += 1
            if verbose:
                print(
                    "  Rebuilding mixed LMC preconditioner at iter",
                    iteration,
                    "max rel:",
                    max_rel,
                )

            for s in range(R):
                var pc_base = providers.offset(s)[].copy()
                pc_base.update_noise(Float32(0.0))
                if use_materialized:
                    pc_base.refresh_materialization()
                var pc = build_pivoted_cholesky_precond_unified(
                    pc_base,
                    precond_rank,
                    error_tol=precond_error_tol,
                    max_num_cols=num_cols_total,
                    precond_method=precond_method,
                )
                (pc_holders + s).destroy_pointee()
                (pc_holders + s).init_pointee_move(pc^)
                _ = pc_base

            actual_rank = pc_holders[0].rank
            L_pc_device = ctx.enqueue_create_buffer[float_dtype](R * n * actual_rank)
            _copy_pc_factors_to_device(pc_holders, L_pc_device, actual_rank)

            for i in range(R * TT):
                C_all_host.unsafe_ptr()[i] = C_all[i]
                last_rebuild_A_all[i] = A_all[i]

            for t in range(T):
                noise_host_precond.unsafe_ptr()[t] = current_noise[t]
                last_rebuild_noise[t] = current_noise[t]

            var kernel_offset_save = 0
            for s in range(R):
                for p in range(num_params_per_latent[s]):
                    last_rebuild_kernel_params[kernel_offset_save + p] = softplus(raw_params_per_latent[s][p])
                kernel_offset_save += num_params_per_latent[s]
                for k in range(num_cat_params_per_latent[s]):
                    last_rebuild_cat[s][k] = _constrained_cat_param_lmc_mixed(
                        cat_states.offset(s)[], raw_cat_per_latent[s], k
                    )

            lmc_precond = LMCPreconditioner(
                ctx,
                L_pc_device,
                C_all_host,
                noise_host_precond,
                n,
                actual_rank,
                T,
                R,
                max_num_cols=num_cols_total,
            )

        cg_pool.ensure_capacity(ctx, nT, num_cols_total, num_probes, max_tridiag_iter, 0)
        var bbmm_result = bbmm_with_precond[JITLMCMixedGradientAdapter, LMCPreconditioner](
            lmc_adapter,
            lmc_precond,
            y_blocked_device.unsafe_ptr(),
            nT,
            cg_pool,
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
        if verbose and (iteration < 5 or iteration % 10 == 0 or iteration == max_iterations - 1):
            print("  [mixed LMC iter", iteration, "] NLL:", nll, "CG:", bbmm_result.num_iterations)

        if isnan(nll):
            ctx.synchronize()
            var nan_iter_time_ns = Int(perf_counter_ns() - iter_start)
            iter_times_ns.append(nan_iter_time_ns)
            if progress_enabled:
                emit_progress_event(
                    progress_callback,
                    "train",
                    "multi_output_lmc",
                    "materialized" if use_materialized else "matrix_free",
                    "nan",
                    actual_iters,
                    max_iterations,
                    nll=nll,
                    best_nll=best_nll,
                    cg_iter=bbmm_result.num_iterations,
                    iter_time_ns=nan_iter_time_ns,
                    noise=current_noise[0] if T > 0 else Float32(0.0),
                    mean=raw_mean[0] if T > 0 else Float32(0.0),
                    precond_rank=actual_rank,
                    precond_rebuild_count=precond_rebuild_count,
                )
            current_noise.free()
            _ = lmc_adapter
            _ = bbmm_result
            break

        var G_A_all = alloc[Float32](R * TT)
        var grad_noise_per_task = alloc[Float32](T)
        _compute_A_and_noise_gradients_mixed_jit(
            ctx,
            providers,
            bbmm_result,
            R,
            T,
            n,
            nT,
            num_probes,
            G_A_all,
            grad_noise_per_task,
        )

        ctx.enqueue_copy(sol_host_buf, bbmm_result.solution)
        if num_probes > 0:
            ctx.enqueue_copy(probe_sol_host_buf, bbmm_result.probe_solutions)
        ctx.synchronize()

        if nll < best_nll:
            best_nll = nll
            patience_counter = 0
            for s in range(R):
                for p in range(num_params_per_latent[s]):
                    best_raw_params[s][p] = raw_params_per_latent[s][p]
                for k in range(num_cat_params_per_latent[s]):
                    best_raw_cat[s][k] = raw_cat_per_latent[s][k]
            for t in range(T):
                best_raw_noise[t] = raw_noise[t]
                best_raw_mean[t] = raw_mean[t]
            for i in range(R * TT):
                best_L_all[i] = L_all[i]
            for i in range(R * T):
                best_raw_L_diag[i] = raw_L_diag[i]
                best_raw_var_diag[i] = raw_var_diag[i]
            for i in range(nT):
                best_sol_host[i] = sol_host_buf[i]

        for s in range(R):
            if num_cat_params_per_latent[s] <= 0:
                continue

            _write_constrained_cat_params_lmc_mixed(
                cat_states.offset(s)[], raw_cat_per_latent[s], cat_params_buf.unsafe_ptr()
            )

            var corr_stride = cat_states.offset(s)[].total_corr_size
            var grad_buf_size = max(num_cat_params_per_latent[s] * corr_stride, 1)
            var all_grad_corr_host = HostBuffer[float_dtype](ctx, grad_buf_size)
            var all_grad_corr_device = ctx.enqueue_create_buffer[float_dtype](grad_buf_size)
            cat_states.offset(s)[].compute_all_gradient_correlations(
                all_grad_corr_host, cat_params_buf.unsafe_ptr()
            )
            ctx.enqueue_copy(all_grad_corr_device, all_grad_corr_host)
            ctx.synchronize()

            for k in range(num_cat_params_per_latent[s]):
                var grad_corr_ptr = all_grad_corr_device.unsafe_ptr() + k * corr_stride
                _apply_cat_gradient_lmc(
                    providers[s],
                    lmc_adapter._A_device.unsafe_ptr().offset(s * TT),
                    out_alpha_device.unsafe_ptr(),
                    cat_temp_base.unsafe_ptr(),
                    bbmm_result.solution.unsafe_ptr(),
                    n,
                    T,
                    1,
                    grad_corr_ptr,
                )
                ctx.enqueue_copy(out_alpha_host, out_alpha_device)
                ctx.synchronize()
                var dot_alpha = Float32(0.0)
                for i in range(nT):
                    dot_alpha += sol_host_buf[i] * out_alpha_host[i]

                _apply_cat_gradient_lmc(
                    providers[s],
                    lmc_adapter._A_device.unsafe_ptr().offset(s * TT),
                    out_probes_device.unsafe_ptr(),
                    cat_temp_base.unsafe_ptr(),
                    cg_pool.probes_device.unsafe_ptr(),
                    n,
                    T,
                    num_probes,
                    grad_corr_ptr,
                )
                ctx.enqueue_copy(out_probes_host, out_probes_device)
                ctx.synchronize()

                var trace_term = Float32(0.0)
                for probe_idx in range(num_probes):
                    var ds = Float32(0.0)
                    for i in range(nT):
                        ds += probe_sol_host_buf[probe_idx * nT + i] * out_probes_host[probe_idx * nT + i]
                    trace_term += ds

                var cat_grad = _clip(
                    (
                        Float32(-0.5) * dot_alpha
                        + Float32(0.5) * trace_term / Float32(num_probes)
                    ) / Float32(nT)
                ) * _cat_chain_derivative_for_param_lmc_mixed(
                    cat_states.offset(s)[], raw_cat_per_latent[s], k
                )
                m_cat[s][k] = beta1 * m_cat[s][k] + (Float32(1.0) - beta1) * cat_grad
                v_cat[s][k] = beta2 * v_cat[s][k] + (Float32(1.0) - beta2) * cat_grad * cat_grad
                var m_hat_cat = m_cat[s][k] / (Float32(1.0) - pow_float32(beta1, t_step))
                var v_hat_cat = v_cat[s][k] / (Float32(1.0) - pow_float32(beta2, t_step))
                raw_cat_per_latent[s][k] -= learning_rate * m_hat_cat / (sqrt(v_hat_cat) + eps)

            _ = all_grad_corr_host
            _ = all_grad_corr_device

        var adam_idx = 0
        for s in range(R):
            var p_offset = lmc_adapter.get_param_offset(s)
            for p in range(num_params_per_latent[s]):
                var grad_raw = _clip(bbmm_result.gradients[p_offset + p]) * _softplus_derivative(raw_params_per_latent[s][p])
                raw_params_per_latent[s][p] += _adam_update(m_adam, v_adam, adam_idx, t_step, grad_raw)
                adam_idx += 1

        for t in range(T):
            var gn = _clip(grad_noise_per_task[t]) * _softplus_derivative(raw_noise[t])
            raw_noise[t] += _adam_update(m_adam, v_adam, adam_idx, t_step, gn)
            adam_idx += 1

        for s in range(R):
            for i in range(T):
                for j in range(i + 1):
                    var grad_L_ij = Float32(0.0)
                    for b in range(T):
                        grad_L_ij += G_A_all[s * TT + i * T + b] * L_all[s * TT + b * T + j]
                    grad_L_ij = _clip(Float32(2.0) * grad_L_ij)
                    if i == j:
                        var grad_raw_diag = grad_L_ij * _softplus_derivative(raw_L_diag[s * T + i])
                        raw_L_diag[s * T + i] += _adam_update(m_adam, v_adam, adam_idx, t_step, grad_raw_diag)
                    else:
                        L_all[s * TT + i * T + j] += _adam_update(m_adam, v_adam, adam_idx, t_step, grad_L_ij)
                    adam_idx += 1
                for j in range(i + 1, T):
                    adam_idx += 1

        for s in range(R):
            for i in range(T):
                var g_diag = _clip(G_A_all[s * TT + i * T + i]) * _softplus_derivative(raw_var_diag[s * T + i])
                raw_var_diag[s * T + i] += _adam_update(m_adam, v_adam, adam_idx, t_step, g_diag)
                adam_idx += 1

        for t in range(T):
            var grad_mean_t = Float32(0.0)
            for i in range(n):
                grad_mean_t -= sol_host_buf[t * n + i]
            grad_mean_t = _clip(grad_mean_t / Float32(nT))
            raw_mean[t] += _adam_update(m_adam, v_adam, adam_idx, t_step, grad_mean_t)
            adam_idx += 1

        if nll >= best_nll and iteration > 20 and early_stop_tol > Float32(0.0):
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                converged = True
                ctx.synchronize()
                var stop_iter_time_ns = Int(perf_counter_ns() - iter_start)
                iter_times_ns.append(stop_iter_time_ns)
                if progress_enabled:
                    emit_progress_event(
                        progress_callback,
                        "train",
                        "multi_output_lmc",
                        "materialized" if use_materialized else "matrix_free",
                        "early_stop",
                        actual_iters,
                        max_iterations,
                        nll=nll,
                        best_nll=best_nll,
                        cg_iter=bbmm_result.num_iterations,
                        iter_time_ns=stop_iter_time_ns,
                        noise=softplus(raw_noise[0]) if T > 0 else Float32(0.0),
                        mean=raw_mean[0] if T > 0 else Float32(0.0),
                        precond_rank=actual_rank,
                        precond_rebuild_count=precond_rebuild_count,
                        converged=converged,
                    )
                current_noise.free()
                G_A_all.free()
                grad_noise_per_task.free()
                _ = lmc_adapter
                _ = bbmm_result
                break

        ctx.synchronize()
        var iter_time_ns = Int(perf_counter_ns() - iter_start)
        iter_times_ns.append(iter_time_ns)
        if progress_enabled and progress_interval_should_emit(iteration, max_iterations, progress_interval):
            emit_progress_event(
                progress_callback,
                "train",
                "multi_output_lmc",
                "materialized" if use_materialized else "matrix_free",
                "iteration",
                actual_iters,
                max_iterations,
                nll=nll,
                best_nll=best_nll,
                cg_iter=bbmm_result.num_iterations,
                iter_time_ns=iter_time_ns,
                noise=softplus(raw_noise[0]) if T > 0 else Float32(0.0),
                mean=raw_mean[0] if T > 0 else Float32(0.0),
                precond_rank=actual_rank,
                precond_rebuild_count=precond_rebuild_count,
            )
        current_noise.free()
        G_A_all.free()
        grad_noise_per_task.free()
        _ = lmc_adapter
        _ = bbmm_result

    var final_params_per_latent = List[List[Float32]]()
    var final_cat_params_per_latent = List[List[Float32]]()
    var final_num_params = List[Int]()
    for s in range(R):
        var ps = List[Float32]()
        for p in range(num_params_per_latent[s]):
            ps.append(softplus(best_raw_params[s][p]))
        final_params_per_latent.append(ps^)

        var cs = List[Float32]()
        for k in range(num_cat_params_per_latent[s]):
            cs.append(_constrained_cat_param_lmc_mixed(cat_states.offset(s)[], best_raw_cat[s], k))
        final_cat_params_per_latent.append(cs^)
        final_num_params.append(num_params_per_latent[s])

    var final_noise = List[Float32]()
    var final_mean = List[Float32]()
    for t in range(T):
        final_noise.append(softplus(best_raw_noise[t]))
        final_mean.append(best_raw_mean[t])

    for i in range(R * TT):
        L_all[i] = best_L_all[i]
    for i in range(R * T):
        raw_L_diag[i] = best_raw_L_diag[i]
        raw_var_diag[i] = best_raw_var_diag[i]

    var final_L_flat = List[Float32]()
    var final_A_flat = List[Float32]()
    var final_var_diag_flat = List[Float32]()
    var final_var_diag_buf = alloc[Float32](T)
    for s in range(R):
        for i in range(T):
            L_all[s * TT + i * T + i] = softplus(raw_L_diag[s * T + i])
        for idx in range(TT):
            final_L_flat.append(L_all[s * TT + idx])
        for i in range(T):
            var v_si = softplus(raw_var_diag[s * T + i])
            final_var_diag_flat.append(v_si)
            final_var_diag_buf[i] = v_si
        _compute_A_from_L(L_all + s * TT, A_all + s * TT, T, final_var_diag_buf)
        for idx in range(TT):
            final_A_flat.append(A_all[s * TT + idx])
    final_var_diag_buf.free()

    var alpha_list = List[Float32]()
    for i in range(nT):
        alpha_list.append(best_sol_host[i])

    var final_nll = best_nll if best_nll < Float32(1e29) else Float32(0.0)

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "multi_output_lmc",
            "materialized" if use_materialized else "matrix_free",
            "complete",
            actual_iters,
            max_iterations,
            nll=final_nll,
            best_nll=final_nll,
            noise=final_noise[0] if T > 0 else Float32(0.0),
            mean=final_mean[0] if T > 0 else Float32(0.0),
            precond_rank=actual_rank,
            precond_rebuild_count=precond_rebuild_count,
            converged=converged,
        )

    L_all.free()
    raw_L_diag.free()
    raw_var_diag.free()
    A_all.free()
    C_all.free()
    m_adam.free()
    v_adam.free()
    best_L_all.free()
    best_raw_L_diag.free()
    best_raw_var_diag.free()
    last_rebuild_kernel_params.free()
    last_rebuild_A_all.free()
    last_rebuild_noise.free()

    for s in range(R):
        (pc_holders + s).destroy_pointee()
    pc_holders.free()

    _ = params_buf
    _ = cat_params_buf
    _ = cg_pool
    _ = y_blocked_device
    _ = y_host
    _ = y_original_host
    _ = sol_host_buf
    _ = probe_sol_host_buf
    _ = out_alpha_device
    _ = out_alpha_host
    _ = out_probes_device
    _ = out_probes_host
    _ = cat_temp_base
    _ = L_pc_device
    _ = lmc_precond

    return LMCMixedJITResult(
        final_params_per_latent^,
        final_cat_params_per_latent^,
        final_num_params^,
        final_A_flat^,
        final_L_flat^,
        final_var_diag_flat^,
        final_noise^,
        final_mean^,
        alpha_list^,
        nll_history^,
        iter_times_ns^,
        final_nll,
        actual_iters,
        converged,
        T,
        R,
        n,
        precond_rebuild_count,
    )
