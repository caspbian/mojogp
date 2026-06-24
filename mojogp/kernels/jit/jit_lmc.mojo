"""JIT LMC (Linear Model of Coregionalization) training using ErasedJITProvider.

LMC model: K_full = sum_{s=1}^{R} (A_s ⊗ K_X_s) + D
Each latent s has its own kernel .so providing K_X_s via fn ptrs.

Architecture:
    R kernel .so modules: each provides K_X_s forward/gradient matvec
    engine .so: combines with A_s matrices, runs BBMM training loop

Training uses CG with Sum-of-Kronecker Woodbury preconditioner (LMCPreconditioner).
A_s and noise gradients are computed externally from CG solutions.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from memory.unsafe_pointer import alloc
from math import sqrt, ceildiv, exp as math_exp, log, isnan
from time import perf_counter_ns
from python import PythonObject

from kernels.jit.erased_provider import ErasedJITProvider
from kernels.jit.jit_training import softplus, inv_softplus, pow_float32
from kernels.combined_inv_quad_logdet import bbmm_with_precond, CGBufferPool, UnifiedBBMMResult
from kernels.pivoted_cholesky import build_pivoted_cholesky_precond_unified, PivotedCholeskyPrecond
from kernels.lmc_preconditioner import LMCPreconditioner
from kernels.lmc_provider import (
    kernel_lmc_accumulate,
    kernel_lmc_fused_grad_accumulate,
    kernel_lmc_add_noise,
    kernel_lmc_accumulate_batched,
    kernel_lmc_add_noise_batched,
    kernel_lmc_add_fixed_noise_batched,
    kernel_zero_buffer,
    kernel_lmc_extract_diagonal,
)
from kernels.cg_solver import kernel_copy, kernel_dot_batched, kernel_fill_constant
from kernels.constants import float_dtype
from kernels.gradient_provider import GradientProvider
from kernels.native_numerics import cholesky_decompose
from kernels.jit.jit_progress import emit_progress_event, progress_interval_should_emit


# =============================================================================
# LMC Result
# =============================================================================


struct LMCJITResult(Movable):
    """Result from JIT LMC training."""
    var final_params_per_latent: List[List[Float32]]  # R lists of kernel params
    var num_kernel_params_per_latent: List[Int]        # [R] param counts
    var A_matrices_flat: List[Float32]                  # R * T * T flattened
    var L_factors_flat: List[Float32]                   # R * T * T L_s factors
    var var_diag_flat: List[Float32]                    # R * T softplus-transformed
    var noise_per_task: List[Float32]                   # [T]
    var mean_per_task: List[Float32]                    # [T]
    var alpha_blocked: List[Float32]                    # [n * T] task-blocked CG solution
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


# =============================================================================
# JIT LMC GradientProvider
# =============================================================================


struct JITLMCGradientAdapter(GradientProvider, Movable):
    """LMC GradientProvider wrapping R ErasedJITProvider instances.

    Implements the full GradientProvider trait for:
        K_full = sum_{s=1}^{R} (A_s ⊗ K_X_s) + D

    Each K_X_s is computed via fn ptrs from an ErasedJITProvider (with noise=0).
    A_s and noise are managed by this adapter.

    Providers are NOT owned — the training loop owns them and passes a pointer.
    """
    var _providers: UnsafePointer[ErasedJITProvider, MutAnyOrigin]  # R providers, not owned
    var _num_latents: Int       # R
    var _num_tasks: Int         # T
    var _n_data: Int            # n (per-latent data points)
    var _param_offsets: UnsafePointer[Int, MutAnyOrigin]  # [R+1] cumulative param counts
    var _total_gradient_params: Int

    # A_s matrices: [R * T * T] on host + device
    var _A_host: HostBuffer[float_dtype]
    var _A_device: DeviceBuffer[float_dtype]

    # Per-task noise: [T] on host + device
    var _noise_host: HostBuffer[float_dtype]
    var _noise_device: DeviceBuffer[float_dtype]

    # Optional fixed per-observation noise: task-blocked [T * n]
    var _has_fixed_noise: Bool
    var _fixed_noise_host: HostBuffer[float_dtype]
    var _fixed_noise_device: DeviceBuffer[float_dtype]

    # Workspace for batched K_X_s @ v (T columns at once)
    var _temp_kx_v: DeviceBuffer[float_dtype]  # [n * T]

    var _ctx: DeviceContext

    fn __init__(
        out self,
        ctx: DeviceContext,
        providers: UnsafePointer[ErasedJITProvider, MutAnyOrigin],
        num_latents: Int,
        num_tasks: Int,
        n_data: Int,
        A_host_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [R * T * T]
        noise_host_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [T]
        fixed_noise_host_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [T * n] or null
        has_fixed_noise: Bool,
        max_num_cols: Int,
    ) raises:
        """Create LMC adapter from R pre-configured providers.

        Args:
            ctx: GPU device context.
            providers: Array of R ErasedJITProvider (noise already set to 0).
            num_latents: Number of latent kernels R.
            num_tasks: Number of tasks T.
            n_data: Number of training data points n.
            A_host_ptr: Flattened [R * T * T] A_s matrices on host.
            noise_host_ptr: Per-task noise [T] on host.
            max_num_cols: Maximum number of CG columns (for temp buffer sizing).
        """
        self._ctx = ctx
        self._providers = providers
        self._num_latents = num_latents
        self._num_tasks = num_tasks
        self._n_data = n_data

        # Compute param offsets from providers
        self._param_offsets = alloc[Int](num_latents + 1)
        self._param_offsets[0] = 0
        for s in range(num_latents):
            var n_params_s = providers[s].num_gradient_params()
            self._param_offsets[s + 1] = self._param_offsets[s] + n_params_s
        self._total_gradient_params = self._param_offsets[num_latents]

        # Copy A_s to host + device
        var A_size = num_latents * num_tasks * num_tasks
        self._A_host = HostBuffer[float_dtype](ctx, A_size)
        self._A_device = ctx.enqueue_create_buffer[float_dtype](A_size)
        for i in range(A_size):
            self._A_host.unsafe_ptr()[i] = A_host_ptr[i]
        self._A_device.enqueue_copy_from(self._A_host)

        # Copy noise to host + device
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

        # Workspace: sized for all CG columns at once (num_cols * T rows of length n)
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
        # Nullify source
        other._param_offsets = UnsafePointer[Int, MutAnyOrigin]()
        other._num_latents = 0

    fn __del__(owned self):
        if self._param_offsets:
            self._param_offsets.free()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    fn _find_latent_for_param(self, param_index: Int) -> Tuple[Int, Int]:
        """Map global param_index to (latent_index, local_param_index)."""
        for s in range(self._num_latents):
            if param_index < self._param_offsets[s + 1]:
                return (s, param_index - self._param_offsets[s])
        return (self._num_latents - 1, 0)

    # =========================================================================
    # GradientProvider trait implementation
    # =========================================================================

    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (sum_s A_s ⊗ K_X_s + D) @ v — all CG columns batched.

        Batches all num_cols columns into a single base kernel call per latent:
            1. Zero all output columns at once
            2. For each latent s: K_X_s @ v (T*num_cols columns) then batched accumulate
            3. Add per-task noise for all columns at once

        This reduces kernel launches from O(R*num_cols) to O(R).
        """
        var n = self._n_data
        var T = self._num_tasks
        var nT = n * T
        alias BLOCK = 256
        var total = num_cols * nT
        var num_blocks = (total + BLOCK - 1) // BLOCK
        var TT = T * T

        # Zero all output columns at once
        self._ctx.enqueue_function[kernel_zero_buffer](
            out_ptr, total,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )

        # For each latent, compute K_X_s @ v (all columns at once) then accumulate
        for s in range(self._num_latents):
            # Single call with T * num_cols columns — base kernel handles NCOLS dispatch
            self._providers[s].forward_matvec(
                self._temp_kx_v.unsafe_ptr(), v_ptr, T * num_cols
            )
            # Accumulate for all columns at once
            self._ctx.enqueue_function[kernel_lmc_accumulate_batched](
                out_ptr,
                self._temp_kx_v.unsafe_ptr(),
                self._A_device.unsafe_ptr().offset(s * TT),
                n, T, nT, num_cols,
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )

        # Add per-task noise for all columns at once
        self._ctx.enqueue_function[kernel_lmc_add_noise_batched](
            out_ptr, v_ptr, self._noise_device.unsafe_ptr(),
            n, T, nT, num_cols,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        if self._has_fixed_noise:
            self._ctx.enqueue_function[kernel_lmc_add_fixed_noise_batched](
                out_ptr, v_ptr, self._fixed_noise_device.unsafe_ptr(),
                n, T, nT, num_cols,
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )

    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute gradient matvec for kernel hyperparameters — all CG columns batched.

        Maps global param_index to (latent_s, local_param_index).
        Then: out = (A_s ⊗ dK_X_s/d(theta)) @ v

        Batches all num_cols columns into a single gradient kernel call,
        reducing kernel launches from O(num_cols) to O(1) per param.
        """
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

        # Single gradient call with T * num_cols columns — all CG columns at once
        self._providers[s].gradient_matvec(
            self._temp_kx_v.unsafe_ptr(), v_ptr, T * num_cols, p_local, False
        )

        # Zero output for all columns then accumulate with A_s
        self._ctx.enqueue_function[kernel_zero_buffer](
            out_ptr, total,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        self._ctx.enqueue_function[kernel_lmc_accumulate_batched](
            out_ptr,
            self._temp_kx_v.unsafe_ptr(),
            self._A_device.unsafe_ptr().offset(s * TT),
            n, T, nT, num_cols,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
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
        """Return minimum per-task noise."""
        var min_noise = self._noise_host.unsafe_ptr()[0]
        for t in range(1, self._num_tasks):
            if self._noise_host.unsafe_ptr()[t] < min_noise:
                min_noise = self._noise_host.unsafe_ptr()[t]
        return min_noise

    fn get_diagonal_value(self) -> Float32:
        """Return max diagonal value: max_t sum_s A_s[t,t]."""
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
        """Extract diagonal of sum_s (A_s ⊗ K_X_s).

        For stationary kernels K_X_s[i,i] = 1.0 (providers have outputscale=1).
        """
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
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )

    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return training data pointer from first provider."""
        return self._providers[0].get_x_ptr()

    fn supports_fused_gradient(self) -> Bool:
        for s in range(self._num_latents):
            if not self._providers[s].supports_fused_gradient():
                return False
        return True

    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute all LMC kernel-gradient matvecs through fn-ptr fused providers."""
        var n = self._n_data
        var T = self._num_tasks
        var nT = n * T
        var TT = T * T
        alias BLOCK = 256

        var max_params_per_latent = 0
        for s in range(self._num_latents):
            var ps = self._param_offsets[s + 1] - self._param_offsets[s]
            if ps > max_params_per_latent:
                max_params_per_latent = ps

        var temp_fused_grad = self._ctx.enqueue_create_buffer[float_dtype](
            max_params_per_latent * nT
        )

        var total_out = self._total_gradient_params * nT * num_cols
        self._ctx.enqueue_function[kernel_zero_buffer](
            out_ptr,
            total_out,
            grid_dim=((total_out + BLOCK - 1) // BLOCK,), block_dim=(BLOCK,),
        )

        for c in range(num_cols):
            var v_col = v_ptr.offset(c * nT)
            for s in range(self._num_latents):
                var num_params_s = self._param_offsets[s + 1] - self._param_offsets[s]
                var param_offset_s = self._param_offsets[s]
                self._providers[s].fused_gradient_matvec(
                    temp_fused_grad.unsafe_ptr(), v_col, T
                )
                var total_threads = num_params_s * nT
                self._ctx.enqueue_function[kernel_lmc_fused_grad_accumulate](
                    out_ptr,
                    temp_fused_grad.unsafe_ptr(),
                    self._A_device.unsafe_ptr().offset(s * TT),
                    n,
                    T,
                    nT,
                    c,
                    num_cols,
                    num_params_s,
                    param_offset_s,
                    grid_dim=((total_threads + BLOCK - 1) // BLOCK,),
                    block_dim=(BLOCK,),
                )

        self._ctx.synchronize()
        _ = temp_fused_grad

    fn supports_fused_ls_os(self) -> Bool:
        return False

    fn fused_ls_os_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        raise Error("fused_ls_os not supported for JIT LMC")

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
        raise Error("fused_3param not supported for JIT LMC")

    # =========================================================================
    # LMC-specific accessors
    # =========================================================================

    fn get_param_offset(self, latent_index: Int) -> Int:
        return self._param_offsets[latent_index]

    fn get_num_latents(self) -> Int:
        return self._num_latents

    fn get_num_tasks(self) -> Int:
        return self._num_tasks

    fn get_n_data(self) -> Int:
        return self._n_data


# =============================================================================
# Helper Functions
# =============================================================================


fn _compute_A_from_L(
    L_ptr: UnsafePointer[Float32, MutAnyOrigin],
    A_ptr: UnsafePointer[Float32, MutAnyOrigin],
    T: Int,
    var_diag_ptr: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
):
    """Compute A = L @ L^T + diag(v) from lower-triangular L."""
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
    """Compute Cholesky C such that A = C C^T. Returns True if successful."""
    return cholesky_decompose(A_ptr, T, C_ptr)


fn _softplus_derivative(x: Float32) -> Float32:
    """Derivative of softplus: sigmoid(x) = 1 / (1 + exp(-x))."""
    return Float32(1.0) / (Float32(1.0) + math_exp(-x))


# =============================================================================
# A_s and Noise Gradient Computation
# =============================================================================


fn _compute_A_and_noise_gradients_jit(
    ctx: DeviceContext,
    providers: UnsafePointer[ErasedJITProvider, MutAnyOrigin],
    bbmm_result: UnifiedBBMMResult,
    R: Int,
    T: Int,
    n: Int,
    nT: Int,
    num_probes: Int,
    G_A_out: UnsafePointer[Float32, MutAnyOrigin],  # [R * T * T] on host
    grad_noise_out: UnsafePointer[Float32, MutAnyOrigin],  # [T] on host
) raises:
    """Compute A_s gradients and per-task noise gradients from CG solutions.

    Uses ErasedJITProvider.forward_matvec for K_X_s @ v (instead of
    MaterializedProvider as in AOT).
    """
    var TT = T * T

    # Workspace buffers
    var dot_result_device = ctx.enqueue_create_buffer[float_dtype](1)
    var dot_result_host = ctx.enqueue_create_host_buffer[float_dtype](1)
    var temp_v = ctx.enqueue_create_buffer[float_dtype](n)
    var kx_alpha = ctx.enqueue_create_buffer[float_dtype](n * T)
    var kx_rf_j = ctx.enqueue_create_buffer[float_dtype](n)

    # ---- A_s gradients ----
    for s in range(R):
        # Compute K_X_s @ alpha_t for all tasks
        for t in range(T):
            ctx.enqueue_function[kernel_copy](
                temp_v.unsafe_ptr(),
                bbmm_result.solution.unsafe_ptr().offset(t * n), n,
                grid_dim=((n + 255) // 256,), block_dim=(256,),
            )
            providers[s].forward_matvec(
                kx_alpha.unsafe_ptr().offset(t * n), temp_v.unsafe_ptr(), 1
            )

        # Data term: G_A[s, i, j] = -alpha_i^T @ K_X_s @ alpha_j
        for i in range(T):
            for j in range(T):
                ctx.enqueue_function[kernel_dot_batched](
                    bbmm_result.solution.unsafe_ptr().offset(i * n),
                    kx_alpha.unsafe_ptr().offset(j * n),
                    dot_result_device.unsafe_ptr(), n, 1,
                    grid_dim=(1, 1), block_dim=(256, 1),
                )
                ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
                ctx.synchronize()
                G_A_out[s * TT + i * T + j] = -dot_result_host[0]

        # Trace term from probes
        for k in range(num_probes):
            for j in range(T):
                ctx.enqueue_function[kernel_copy](
                    temp_v.unsafe_ptr(),
                    bbmm_result.right_factors.unsafe_ptr().offset(k * nT + j * n), n,
                    grid_dim=((n + 255) // 256,), block_dim=(256,),
                )
                providers[s].forward_matvec(
                    kx_rf_j.unsafe_ptr(), temp_v.unsafe_ptr(), 1
                )

                for i in range(T):
                    ctx.enqueue_function[kernel_dot_batched](
                        bbmm_result.probe_solutions.unsafe_ptr().offset(k * nT + i * n),
                        kx_rf_j.unsafe_ptr(),
                        dot_result_device.unsafe_ptr(), n, 1,
                        grid_dim=(1, 1), block_dim=(256, 1),
                    )
                    ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
                    ctx.synchronize()
                    G_A_out[s * TT + i * T + j] += dot_result_host[0] / Float32(num_probes)

        # Scale
        for idx in range(TT):
            G_A_out[s * TT + idx] = Float32(0.5) * G_A_out[s * TT + idx] / Float32(nT)

    # ---- Per-task noise gradients ----
    for t in range(T):
        ctx.enqueue_function[kernel_dot_batched](
            bbmm_result.solution.unsafe_ptr().offset(t * n),
            bbmm_result.solution.unsafe_ptr().offset(t * n),
            dot_result_device.unsafe_ptr(), n, 1,
            grid_dim=(1, 1), block_dim=(256, 1),
        )
        ctx.enqueue_copy(dst_buf=dot_result_host, src_buf=dot_result_device)
        ctx.synchronize()
        var data_term = -dot_result_host[0]

        var trace_sum = Float32(0.0)
        for k in range(num_probes):
            ctx.enqueue_function[kernel_dot_batched](
                bbmm_result.probe_solutions.unsafe_ptr().offset(k * nT + t * n),
                bbmm_result.right_factors.unsafe_ptr().offset(k * nT + t * n),
                dot_result_device.unsafe_ptr(), n, 1,
                grid_dim=(1, 1), block_dim=(256, 1),
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


# =============================================================================
# JIT LMC Training
# =============================================================================


fn train_lmc_jit(
    providers: UnsafePointer[ErasedJITProvider, MutAnyOrigin],
    ctx: DeviceContext,
    y_blocked_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
    num_latents: Int,
    params_per_latent: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin]],
    num_params_per_latent: UnsafePointer[Int],
    trainable_masks_per_latent: UnsafePointer[UnsafePointer[Bool, MutAnyOrigin]],
    initial_noise_per_task_ptr: UnsafePointer[Float32, MutAnyOrigin],
    fixed_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    has_fixed_noise: Bool,
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
    progress_callback: PythonObject = None,
    progress_interval: Int = 1,
    progress_enabled: Bool = False,
) raises -> LMCJITResult:
    """Train LMC multi-output GP with R latent kernels via JIT fn-ptr providers.

    Full BBMM training with:
    - Joint CG solve for NLL + kernel hyperparameter gradients
    - External A_s and per-task noise gradient computation
    - Adam optimizer with softplus constraints
    - L_s L_s^T + diag(v_s) parameterization for A_s (PSD guarantee)
    - Adaptive preconditioner rebuild
    - NaN recovery with var_diag bumping
    - Per-task constant mean learning
    """
    var T = num_tasks
    var R = num_latents
    var nT = n * T
    var TT = T * T
    var num_cols_total = 1 + num_probes
    var precond_error_tol = Float32(1e-3)
    var use_preconditioner = precond_rank > 0
    if precond_method == 0:
        precond_error_tol = Float32(0.0)

    # =========================================================================
    # 1. Initialize parameters
    # =========================================================================
    

    # Per-latent kernel params (unconstrained)
    var raw_params_per_latent = List[List[Float32]]()
    for s in range(R):
        var raw_s = List[Float32]()
        var N_s = num_params_per_latent[s]
        for p in range(N_s):
            raw_s.append(inv_softplus(params_per_latent[s][p]))
        raw_params_per_latent.append(raw_s^)

    # Per-task noise (unconstrained)
    var raw_noise = List[Float32]()
    for t in range(T):
        raw_noise.append(inv_softplus(initial_noise_per_task_ptr[t]))

    # Per-task means
    var raw_mean = List[Float32]()
    for t in range(T):
        raw_mean.append(initial_mean_per_task_ptr[t])

    # L_s factors: Initialize as scaled identity (A_s ≈ (1/R) * I)
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

    # Diagonal variance: A_s = L_s L_s^T + diag(v_s)
    var raw_var_diag = alloc[Float32](R * T)
    var init_var_diag_raw = inv_softplus(Float32(0.1))
    for i in range(R * T):
        raw_var_diag[i] = init_var_diag_raw

    # Compute initial A_s matrices
    var A_all = alloc[Float32](R * TT)
    var C_all = alloc[Float32](R * TT)  # Cholesky of A_s for preconditioner
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

    # Adam state: flat array for all parameters
    # Layout: [R*N_s kernel params] + [T noise] + [R*TT L_factors] + [R*T var_diag] + [T mean]
    var total_kernel_params = 0
    for s in range(R):
        total_kernel_params += num_params_per_latent[s]
    var total_adam = total_kernel_params + T + R * TT + R * T + T
    var m_adam = alloc[Float32](total_adam)
    var v_adam = alloc[Float32](total_adam)
    for i in range(total_adam):
        m_adam[i] = Float32(0.0)
        v_adam[i] = Float32(0.0)

    var beta1 = Float32(0.9)
    var beta2 = Float32(0.999)
    var eps = Float32(1e-8)
    var grad_clip_val = Float32(10.0)
    var t_step = 0

    var nll_history = List[Float32]()
    var iter_times_ns = List[Int]()
    var nll_smoothed = Float32(1e10)
    var patience_counter = 0
    var converged = False
    var actual_iters = 0

    # Host buffer for provider param updates
    var max_params = 0
    for s in range(R):
        if num_params_per_latent[s] > max_params:
            max_params = num_params_per_latent[s]
    var params_buf = ctx.enqueue_create_host_buffer[float_dtype](max_params)
    ctx.synchronize()

    # CG buffer pool
    var cg_pool = CGBufferPool(ctx, nT, num_cols_total)

    # =========================================================================
    # 2. Build initial preconditioner
    # =========================================================================
    

    var pc_holders = alloc[PivotedCholeskyPrecond](R)
    for s in range(R):
        var pc_base = providers.offset(s)[].clone()
        pc_base.update_noise(Float32(0))
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

    # Build noise host for preconditioner
    var noise_host_precond = HostBuffer[float_dtype](ctx, T)
    for t in range(T):
        var fixed_mean_t = Float32(0.0)
        if has_fixed_noise:
            for i in range(n):
                fixed_mean_t += fixed_noise_ptr[t * n + i]
            fixed_mean_t = fixed_mean_t / Float32(n)
        noise_host_precond.unsafe_ptr()[t] = initial_noise_per_task_ptr[t] + fixed_mean_t

    # Build C_all as HostBuffer for LMCPreconditioner
    var C_all_host = HostBuffer[float_dtype](ctx, R * TT)
    for i in range(R * TT):
        C_all_host.unsafe_ptr()[i] = C_all[i]

    var lmc_precond = LMCPreconditioner(
        ctx, L_pc_device, C_all_host, noise_host_precond,
        n, actual_rank, T, R,
        max_num_cols=num_cols_total,
    )

    # Track params for adaptive rebuild
    var precond_rebuild_count = 0
    var last_rebuild_params = alloc[Float32](total_kernel_params)
    var param_offset = 0
    for s in range(R):
        for p in range(num_params_per_latent[s]):
            last_rebuild_params[param_offset + p] = params_per_latent[s][p]
        param_offset += num_params_per_latent[s]
    var last_rebuild_noise = alloc[Float32](T)
    for t in range(T):
        last_rebuild_noise[t] = initial_noise_per_task_ptr[t]

    # "Last good" parameters for NaN recovery
    var good_raw_params = List[List[Float32]]()
    for s in range(R):
        var gs = List[Float32]()
        for p in range(num_params_per_latent[s]):
            gs.append(raw_params_per_latent[s][p])
        good_raw_params.append(gs^)
    var good_raw_noise = List[Float32]()
    for t in range(T):
        good_raw_noise.append(raw_noise[t])
    var good_raw_mean = List[Float32]()
    for t in range(T):
        good_raw_mean.append(raw_mean[t])
    var good_L_all = alloc[Float32](R * TT)
    for i in range(R * TT):
        good_L_all[i] = L_all[i]
    var good_raw_L_diag = alloc[Float32](R * T)
    for i in range(R * T):
        good_raw_L_diag[i] = raw_L_diag[i]
    var good_raw_var_diag = alloc[Float32](R * T)
    for i in range(R * T):
        good_raw_var_diag[i] = raw_var_diag[i]

    var consecutive_nan_count: Int = 0
    var last_good_nll = Float32(1e10)
    var best_nll = Float32(1e30)
    var best_raw_params = List[List[Float32]]()
    for s in range(R):
        var bs = List[Float32]()
        for p in range(num_params_per_latent[s]):
            bs.append(raw_params_per_latent[s][p])
        best_raw_params.append(bs^)
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

    

    # y_host for mean centering — copy from host pointer passed by bindings
    var y_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    for i in range(nT):
        y_host.unsafe_ptr()[i] = y_blocked_host_ptr[i]
    var y_original_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
    for i in range(nT):
        y_original_host[i] = y_blocked_host_ptr[i]

    var y_blocked_device = ctx.enqueue_create_buffer[float_dtype](nT)
    y_blocked_device.enqueue_copy_from(y_host)
    ctx.synchronize()

    if progress_enabled:
        emit_progress_event(
            progress_callback,
            "train",
            "multi_output_lmc",
            "unknown",
            "start",
            0,
            max_iterations,
            precond_rank=actual_rank,
            precond_rebuild_count=precond_rebuild_count,
        )

    # =========================================================================
    # 3. Training loop
    # =========================================================================
    

    for iteration in range(max_iterations):
        actual_iters = iteration + 1
        t_step += 1
        ctx.synchronize()
        var iter_start = perf_counter_ns()

        # ==================================================================
        # 3a. Transform raw parameters to constrained
        # ==================================================================
        for s in range(R):
            var N_s = num_params_per_latent[s]
            for p in range(N_s):
                params_buf.unsafe_ptr()[p] = softplus(raw_params_per_latent[s][p])
            providers.offset(s)[].update_params(params_buf.unsafe_ptr())
            providers.offset(s)[].update_noise(Float32(0))  # Noise handled by adapter

        var current_noise = alloc[Float32](T)
        for t in range(T):
            current_noise[t] = softplus(raw_noise[t])

        # Re-center Y with current per-task mean
        for t in range(T):
            for i in range(n):
                y_host[t * n + i] = y_original_host[t * n + i] - raw_mean[t]
        y_blocked_device.enqueue_copy_from(y_host)
        ctx.synchronize()

        # Reconstruct L_s from raw_L_diag + off-diagonal, compute A_s
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

        # ==================================================================
        # 3b. Build LMC adapter
        # ==================================================================
        var lmc_adapter = JITLMCGradientAdapter(
            ctx, providers, R, T, n, A_all, current_noise,
            fixed_noise_ptr, has_fixed_noise, num_cols_total,
        )

        # ==================================================================
        # 3c. Adaptive preconditioner rebuild
        # ==================================================================
        var should_rebuild = iteration == 0
        if not should_rebuild:
            var max_rel = Float32(0.0)
            var po = 0
            for s in range(R):
                for p in range(num_params_per_latent[s]):
                    var cur_val = softplus(raw_params_per_latent[s][p])
                    var rel = abs(cur_val - last_rebuild_params[po]) / (abs(last_rebuild_params[po]) + Float32(1e-8))
                    if rel > max_rel:
                        max_rel = rel
                    po += 1
            for t in range(T):
                var rel_n = abs(current_noise[t] - last_rebuild_noise[t]) / (abs(last_rebuild_noise[t]) + Float32(1e-8))
                if rel_n > max_rel:
                    max_rel = rel_n
            should_rebuild = max_rel > precond_rebuild_threshold

        if should_rebuild:
            if iteration > 0:
                precond_rebuild_count += 1
            if verbose and iteration > 0:
                print("  Rebuilding LMC preconditioner at iter", iteration)
            for s in range(R):
                var pc_base = providers.offset(s)[].clone()
                pc_base.update_noise(Float32(0))
                var new_pc = build_pivoted_cholesky_precond_unified(
                    pc_base,
                    precond_rank,
                    error_tol=precond_error_tol,
                    max_num_cols=num_cols_total,
                    precond_method=precond_method,
                )
                (pc_holders + s).destroy_pointee()
                (pc_holders + s).init_pointee_move(new_pc^)
                _ = pc_base

            actual_rank = pc_holders[0].rank
            L_pc_device = ctx.enqueue_create_buffer[float_dtype](R * n * actual_rank)
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

            # Rebuild C_all_host from current C_all
            for i in range(R * TT):
                C_all_host.unsafe_ptr()[i] = C_all[i]

            noise_host_precond = HostBuffer[float_dtype](ctx, T)
            for t in range(T):
                var fixed_mean_t = Float32(0.0)
                if has_fixed_noise:
                    for i in range(n):
                        fixed_mean_t += fixed_noise_ptr[t * n + i]
                    fixed_mean_t = fixed_mean_t / Float32(n)
                noise_host_precond.unsafe_ptr()[t] = current_noise[t] + fixed_mean_t

            lmc_precond = LMCPreconditioner(
                ctx, L_pc_device, C_all_host, noise_host_precond,
                n, actual_rank, T, R,
                max_num_cols=num_cols_total,
            )

            # Save rebuild params
            var po2 = 0
            for s in range(R):
                for p in range(num_params_per_latent[s]):
                    last_rebuild_params[po2] = softplus(raw_params_per_latent[s][p])
                    po2 += 1
            for t in range(T):
                last_rebuild_noise[t] = current_noise[t]

        # ==================================================================
        # 3d. BBMM step: NLL + kernel param gradients
        # ==================================================================
        cg_pool.ensure_capacity(ctx, nT, num_cols_total, num_probes, max_tridiag_iter, 0, num_kernel_params=total_kernel_params)

        var bbmm_result = bbmm_with_precond[JITLMCGradientAdapter, LMCPreconditioner](
            lmc_adapter, lmc_precond,
            y_blocked_device.unsafe_ptr(), nT, cg_pool,
            num_probes, max_cg_iter, max_tridiag_iter, cg_tol,
            iteration=iteration,
            recycle_alpha=iteration > 0 and not should_rebuild,
            use_preconditioner=use_preconditioner,
        )

        var nll = bbmm_result.nll
        nll_history.append(nll)

        # ==================================================================
        # 3e. NaN guard
        # ==================================================================
        if isnan(nll):
            consecutive_nan_count += 1
            if verbose:
                print("  [LMC iter", iteration, "] NLL is NaN (count:", consecutive_nan_count, "), restoring last-good params")
            # Zero CG warm-start
            var pool_x_size = nT * num_cols_total
            ctx.enqueue_function[kernel_fill_constant](
                cg_pool.x.unsafe_ptr(), pool_x_size, Float32(0.0),
                grid_dim=((pool_x_size + 255) // 256,), block_dim=(256,),
            )
            ctx.synchronize()
            # Force precond rebuild
            for i in range(total_kernel_params):
                last_rebuild_params[i] = Float32(0.0)
            # Restore params
            for s in range(R):
                for p in range(num_params_per_latent[s]):
                    raw_params_per_latent[s][p] = good_raw_params[s][p]
            for t in range(T):
                raw_noise[t] = good_raw_noise[t]
                raw_mean[t] = good_raw_mean[t]
            for i in range(R * TT):
                L_all[i] = good_L_all[i]
            for i in range(R * T):
                raw_L_diag[i] = good_raw_L_diag[i]
            # Bump var_diag
            var bump = Float32(0.7) * Float32(consecutive_nan_count)
            for i in range(R * T):
                raw_var_diag[i] = good_raw_var_diag[i] + bump
                good_raw_var_diag[i] = raw_var_diag[i]

            ctx.synchronize()
            var nan_iter_time_ns = Int(perf_counter_ns() - iter_start)
            iter_times_ns.append(nan_iter_time_ns)
            if progress_enabled:
                emit_progress_event(
                    progress_callback,
                    "train",
                    "multi_output_lmc",
                    "unknown",
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
            continue

        # ==================================================================
        # 3f. Compute A_s and noise gradients externally
        # ==================================================================
        var G_A_all = alloc[Float32](R * TT)
        var grad_noise_per_task = alloc[Float32](T)
        _compute_A_and_noise_gradients_jit(
            ctx, providers, bbmm_result,
            R, T, n, nT, num_probes,
            G_A_all, grad_noise_per_task,
        )

        var sol_host = ctx.enqueue_create_host_buffer[float_dtype](nT)
        ctx.enqueue_copy(dst_buf=sol_host, src_buf=bbmm_result.solution)
        ctx.synchronize()

        if nll < best_nll:
            best_nll = nll
            for s in range(R):
                for p in range(num_params_per_latent[s]):
                    best_raw_params[s][p] = raw_params_per_latent[s][p]
            for t in range(T):
                best_raw_noise[t] = raw_noise[t]
                best_raw_mean[t] = raw_mean[t]
            for i in range(R * TT):
                best_L_all[i] = L_all[i]
            for i in range(R * T):
                best_raw_L_diag[i] = raw_L_diag[i]
                best_raw_var_diag[i] = raw_var_diag[i]
            for i in range(nT):
                best_sol_host[i] = sol_host[i]

        # ==================================================================
        # 3g. Adam updates for all parameters
        # ==================================================================
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
            g: Float32,
        ) -> Float32:
            var m = beta1 * m_ptr[idx] + (Float32(1.0) - beta1) * g
            var v = beta2 * v_ptr[idx] + (Float32(1.0) - beta2) * g * g
            m_ptr[idx] = m
            v_ptr[idx] = v
            var mh = m / (Float32(1.0) - pow_float32(beta1, t_step))
            var vh = v / (Float32(1.0) - pow_float32(beta2, t_step))
            return -learning_rate * mh / (sqrt(vh) + eps)

        var adam_idx = 0

        # Kernel param gradients (from BBMM) with softplus chain rule
        for s in range(R):
            var p_offset = lmc_adapter.get_param_offset(s)
            for p in range(num_params_per_latent[s]):
                if trainable_masks_per_latent[s][p]:
                    var grad_raw = _clip(bbmm_result.gradients[p_offset + p]) * _softplus_derivative(raw_params_per_latent[s][p])
                    raw_params_per_latent[s][p] += _adam_update(m_adam, v_adam, adam_idx, grad_raw)
                adam_idx += 1

        # Per-task noise with softplus chain rule
        for t in range(T):
            var gn = _clip(grad_noise_per_task[t]) * _softplus_derivative(raw_noise[t])
            raw_noise[t] += _adam_update(m_adam, v_adam, adam_idx, gn)
            adam_idx += 1

        # L_s off-diagonal (unconstrained) and diagonal (through softplus)
        # Gradient: dNLL/dL[i,j] = 2 * sum_b G_A[i,b] * L[b,j]
        for s in range(R):
            for i in range(T):
                for j in range(i + 1):  # j <= i (lower triangular)
                    var grad_L_ij = Float32(0.0)
                    for b in range(T):
                        grad_L_ij += G_A_all[s * TT + i * T + b] * L_all[s * TT + b * T + j]
                    grad_L_ij = _clip(Float32(2.0) * grad_L_ij)

                    if i == j:
                        # Diagonal: chain rule through softplus
                        var grad_raw_diag = grad_L_ij * _softplus_derivative(raw_L_diag[s * T + i])
                        raw_L_diag[s * T + i] += _adam_update(m_adam, v_adam, adam_idx, grad_raw_diag)
                    else:
                        # Off-diagonal: unconstrained
                        L_all[s * TT + i * T + j] += _adam_update(m_adam, v_adam, adam_idx, grad_L_ij)
                    adam_idx += 1
                # Skip upper triangular (zero gradient)
                for j in range(i + 1, T):
                    adam_idx += 1

        # var_diag gradients: dNLL/d(raw_v_s[i]) = G_A[s,i,i] * softplus'(raw_v[i])
        for s in range(R):
            for i in range(T):
                var g_diag = _clip(G_A_all[s * TT + i * T + i]) * _softplus_derivative(raw_var_diag[s * T + i])
                raw_var_diag[s * T + i] += _adam_update(m_adam, v_adam, adam_idx, g_diag)
                adam_idx += 1

        # Per-task mean gradient: d(NLL)/d(mean_t) = -sum_i alpha[t*n+i] / nT
        for t in range(T):
            var grad_mean_t = Float32(0.0)
            for i in range(n):
                grad_mean_t -= sol_host[t * n + i]
            grad_mean_t = _clip(grad_mean_t / Float32(nT))
            raw_mean[t] += _adam_update(m_adam, v_adam, adam_idx, grad_mean_t)
            adam_idx += 1

        # ==================================================================
        # 3h. Save "last good" parameters
        # ==================================================================
        var nll_drop = last_good_nll - nll
        var nll_is_reasonable = (nll_drop < Float32(2.0)) or (iteration < 3)
        if nll_is_reasonable:
            consecutive_nan_count = 0
            last_good_nll = nll
            for s in range(R):
                for p in range(num_params_per_latent[s]):
                    good_raw_params[s][p] = raw_params_per_latent[s][p]
            for t in range(T):
                good_raw_noise[t] = raw_noise[t]
                good_raw_mean[t] = raw_mean[t]
            for i in range(R * TT):
                good_L_all[i] = L_all[i]
            for i in range(R * T):
                good_raw_L_diag[i] = raw_L_diag[i]
                good_raw_var_diag[i] = raw_var_diag[i]

        # ==================================================================
        # 3i. Logging and early stopping
        # ==================================================================
        if verbose and (iteration < 5 or iteration % 10 == 0 or iteration == max_iterations - 1):
            print("  [LMC iter", iteration, "] NLL:", nll, "CG:", bbmm_result.num_iterations)

        var prev_smoothed = nll_smoothed
        if iteration == 0:
            nll_smoothed = nll
        else:
            nll_smoothed = Float32(0.9) * nll_smoothed + Float32(0.1) * nll

        if iteration > 20 and early_stop_tol > Float32(0.0):
            var rel_change = (prev_smoothed - nll_smoothed) / (abs(nll_smoothed) + Float32(1e-8))
            if rel_change < early_stop_tol and rel_change > Float32(-0.1):
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print("  Early stopping at iteration", iteration, "NLL:", nll)
                    converged = True
                    ctx.synchronize()
                    var stop_iter_time_ns = Int(perf_counter_ns() - iter_start)
                    iter_times_ns.append(stop_iter_time_ns)
                    if progress_enabled:
                        emit_progress_event(
                            progress_callback,
                            "train",
                            "multi_output_lmc",
                            "unknown",
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
            else:
                patience_counter = 0

        ctx.synchronize()
        var iter_time_ns = Int(perf_counter_ns() - iter_start)
        iter_times_ns.append(iter_time_ns)
        if progress_enabled and progress_interval_should_emit(iteration, max_iterations, progress_interval):
            emit_progress_event(
                progress_callback,
                "train",
                "multi_output_lmc",
                "unknown",
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

    # =========================================================================
    # 4. Package results
    # =========================================================================

    var final_params_per_latent = List[List[Float32]]()
    var final_num_params = List[Int]()
    for s in range(R):
        var ps = List[Float32]()
        for p in range(num_params_per_latent[s]):
            ps.append(softplus(best_raw_params[s][p]))
        final_params_per_latent.append(ps^)
        final_num_params.append(num_params_per_latent[s])

    var final_noise = List[Float32]()
    for t in range(T):
        final_noise.append(softplus(best_raw_noise[t]))

    var final_mean = List[Float32]()
    for t in range(T):
        final_mean.append(best_raw_mean[t])

    for i in range(R * TT):
        L_all[i] = best_L_all[i]
    for i in range(R * T):
        raw_L_diag[i] = best_raw_L_diag[i]
        raw_var_diag[i] = best_raw_var_diag[i]

    # Reconstruct final A_s with softplus on L diag and var_diag
    var final_L_flat = List[Float32]()
    var final_A_flat = List[Float32]()
    var final_var_diag_flat = List[Float32]()
    var final_var_diag_buf = alloc[Float32](T)

    for s in range(R):
        # Apply softplus to L diagonal
        for i in range(T):
            L_all[s * TT + i * T + i] = softplus(raw_L_diag[s * T + i])
        # Copy L_s
        for idx in range(TT):
            final_L_flat.append(L_all[s * TT + idx])
        # Compute var_diag
        for i in range(T):
            var v_si = softplus(raw_var_diag[s * T + i])
            final_var_diag_flat.append(v_si)
            final_var_diag_buf[i] = v_si
        # Compute A_s = L L^T + diag(v)
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
            "unknown",
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

    # Keep live providers synchronized with the packaged best state. Training may
    # finish on a later optimizer step than the best NLL, but wrapper prediction
    # reuses these providers immediately after fit.
    for s in range(R):
        for p in range(num_params_per_latent[s]):
            params_buf.unsafe_ptr()[p] = softplus(best_raw_params[s][p])
        providers.offset(s)[].update_params(params_buf.unsafe_ptr())
        providers.offset(s)[].update_noise(Float32(0))

    # Cleanup
    for s in range(R):
        (pc_holders + s).destroy_pointee()
    pc_holders.free()
    L_all.free()
    raw_L_diag.free()
    raw_var_diag.free()
    A_all.free()
    C_all.free()
    m_adam.free()
    v_adam.free()
    last_rebuild_params.free()
    last_rebuild_noise.free()
    good_L_all.free()
    good_raw_L_diag.free()
    good_raw_var_diag.free()
    best_L_all.free()
    best_raw_L_diag.free()
    best_raw_var_diag.free()

    _ = params_buf
    _ = cg_pool
    _ = L_pc_device
    _ = y_blocked_device
    _ = y_host
    _ = y_original_host

    return LMCJITResult(
        final_params_per_latent^, final_num_params^,
        final_A_flat^, final_L_flat^, final_var_diag_flat^,
        final_noise^, final_mean^, alpha_list^,
        nll_history^, iter_times_ns^, final_nll, actual_iters, converged, T, R, n,
        precond_rebuild_count,
    )
