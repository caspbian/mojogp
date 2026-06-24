"""LMC (Linear Model of Coregionalization) Provider for Multi-Output GP.

Implements a GradientProvider for the LMC kernel:
    K_full = sum_{s=1}^{R} (A_s ⊗ K_X_s) + D

where:
    R = number of latent kernels
    A_s = T×T coregionalization matrix for latent s (PSD)
    K_X_s = n×n kernel matrix for latent s
    D = diag(noise_1, ..., noise_T) ⊗ I_n

Vector layout: task-blocked ordering
    v = [v_task0; v_task1; ...; v_taskT]
    where each v_task_t is n-dimensional.

Matvec: (sum_s A_s ⊗ K_X_s + D) @ v
    For each task t_out:
        out[t_out] = sum_s sum_t (A_s[t_out,t] * K_X_s @ v_t) + noise[t_out] * v[t_out]

Gradient matvec:
    For param_index p belonging to latent s (local index p_local):
        out = (A_s ⊗ dK_X_s/d(theta_{p_local})) @ v

A_s and noise gradients are computed externally from CG solution, NOT through
gradient_matvec (same pattern as B/noise in KroneckerDirectProvider).

Reference: Alvarez et al. (2012), "Kernels for Vector-Valued Functions"
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from memory.unsafe_pointer import alloc

from .constants import float_dtype
from .gradient_provider import GradientProvider
from .matvec_provider import MaterializedProvider, MatrixFreeProvider
from .cg_solver import kernel_copy


# =============================================================================
# GPU Kernels for LMC Operations
# =============================================================================

fn kernel_lmc_accumulate(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_buf_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n * T] for this latent
    A_s_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [T × T] row-major
    n: Int,
    num_tasks: Int,
    nT: Int,
    col: Int,
) -> None:
    """Accumulate A_s ⊗ K_X_s @ v contribution to output.
    
    For each task t_out and data point i:
        out[t_out*n + i, col] += sum_t (A_s[t_out, t] * kx_v_buf[t*n + i])
    
    This ADDS to out_ptr (does not overwrite), enabling accumulation over latents.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(nT):
        return
    
    var i = Int(idx) % n        # data point index
    var t_out = Int(idx) // n   # output task index
    
    var val = Float32(0.0)
    for t in range(num_tasks):
        val += A_s_ptr[t_out * num_tasks + t] * kx_v_buf_ptr[t * n + i]
    
    var out_idx = col * nT + t_out * n + i
    out_ptr[out_idx] += val


fn kernel_lmc_add_noise(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
    nT: Int,
    col: Int,
) -> None:
    """Add noise[t] * v[t*n+i, col] to output.
    
    out[t*n + i, col] += noise[t] * v[t*n + i, col]
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(nT):
        return
    
    var i = Int(idx) % n
    var t = Int(idx) // n
    
    var v_idx = col * nT + t * n + i
    out_ptr[v_idx] += noise_ptr[t] * v_ptr[v_idx]


fn kernel_zero_buffer(
    ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """Zero a buffer."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    ptr[idx] = Float32(0.0)


fn kernel_lmc_accumulate_batched(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_buf_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [num_cols * n * T]
    A_s_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [T × T] row-major
    n: Int,
    num_tasks: Int,
    nT: Int,
    num_cols: Int,
) -> None:
    """Batched accumulate across all CG columns.

    kx_v_buf_ptr layout: [c * nT + t * n + i]  (col-major within each CG column)
    out_ptr layout:       [c * nT + t_out * n + i]

    This ADDS to out_ptr (does not overwrite).
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = num_cols * nT
    if idx >= total:
        return
    var c = idx // nT
    var rem = idx % nT
    var t_out = rem // n
    var i = rem % n

    var val = Float32(0.0)
    for t in range(num_tasks):
        val += A_s_ptr[t_out * num_tasks + t] * kx_v_buf_ptr[c * nT + t * n + i]

    out_ptr[c * nT + t_out * n + i] += val


fn kernel_lmc_add_noise_batched(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
    nT: Int,
    num_cols: Int,
) -> None:
    """Add per-task noise for all CG columns at once.

    out[c * nT + t * n + i] += noise[t] * v[c * nT + t * n + i]
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = num_cols * nT
    if idx >= total:
        return
    var c = idx // nT
    var rem = idx % nT
    var t = rem // n
    var i = rem % n

    var v_idx = c * nT + t * n + i
    out_ptr[v_idx] += noise_ptr[t] * v_ptr[v_idx]


fn kernel_lmc_add_fixed_noise_batched(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    fixed_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_tasks: Int,
    nT: Int,
    num_cols: Int,
) -> None:
    """Add per-observation fixed noise for all CG columns at once.

    fixed_noise_ptr is task-blocked [t * n + i], matching the native LMC
    training layout. The learned per-task noise is added separately.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = num_cols * nT
    if idx >= total:
        return
    var c = idx // nT
    var rem = idx % nT
    var v_idx = c * nT + rem
    out_ptr[v_idx] += fixed_noise_ptr[rem] * v_ptr[v_idx]


fn kernel_lmc_fused_grad_accumulate(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    fused_buf_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [P_s * n * T] from per-task fused grads
    A_s_ptr: UnsafePointer[Float32, MutAnyOrigin],         # [T × T] row-major
    n: Int,
    num_tasks: Int,
    nT: Int,
    col: Int,           # which outer column in the BBMM RHS
    num_cols: Int,       # total outer columns (for output stride)
    num_params_s: Int,   # P_s: number of params for this latent
    param_offset: Int,   # global param offset for this latent
) -> None:
    """Accumulate A_s-weighted fused gradient contributions to the global output.
    
    The fused gradient buffer holds per-task gradient results for all params of latent s:
        fused_buf[p * n * T + t * n + i] = (dK_X_s/dtheta_p @ v_t)[i]
    
    This kernel computes, for each param p and each output task t_out:
        out[(param_offset + p) * nT * num_cols + col * nT + t_out * n + i] +=
            sum_t A_s[t_out, t] * fused_buf[p * n * T + t * n + i]
    
    Threads indexed by (p, t_out, i) flattened: idx = p * nT + t_out * n + i
    Total threads: num_params_s * nT.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = UInt(num_params_s * nT)
    
    if idx >= total:
        return
    
    var flat_idx = Int(idx)
    var p = flat_idx // nT                    # param index within this latent
    var rem = flat_idx - p * nT
    var t_out = rem // n                       # output task
    var i = rem - t_out * n                    # data point
    
    # Accumulate A_s-weighted sum over source tasks
    var val = Float32(0.0)
    for t in range(num_tasks):
        val += A_s_ptr[t_out * num_tasks + t] * fused_buf_ptr[p * n * num_tasks + t * n + i]
    
    # Write to global output
    var global_p = param_offset + p
    var out_idx = global_p * nT * num_cols + col * nT + t_out * n + i
    out_ptr[out_idx] += val


fn kernel_lmc_extract_diagonal(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    A_all_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [R * T * T] contiguous
    num_latents: Int,
    n: Int,
    num_tasks: Int,
    nT: Int,
) -> None:
    """Extract diagonal of sum_s (A_s ⊗ K_X_s).
    
    For stationary kernels, K_X_s[i,i] = 1.0 (base outputscale=1).
    So diag[t*n+i] = sum_s A_s[t,t].
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(nT):
        return
    
    var t = Int(idx) // n  # task index
    var TT = num_tasks * num_tasks
    
    var val = Float32(0.0)
    for s in range(num_latents):
        val += A_all_ptr[s * TT + t * num_tasks + t]
    
    diag_ptr[idx] = val


struct MaterializedLMCGradientAdapter(GradientProvider, Movable):
    """Fully flat GradientProvider for LMC multi-output GP.
    
    This struct eliminates the deeply nested generic type hierarchy
    that causes Mojo compiler stack overflow during AOT compilation.
    
    Instead, this struct holds MaterializedProvider instances directly via
    UnsafePointer and inlines all the LMC matvec logic. bbmm_with_precond sees
    only the concrete type MaterializedLMCGradientAdapter (no generic params).
    
    The LMC kernel is:
        K_full = sum_{s=1}^{R} (A_s ⊗ K_X_s) + D
    
    where each K_X_s comes from a MaterializedProvider with outputscale=1, noise=0.
    
    Gradient parameter layout (per latent s):
      Isotropic (use_ard=False):
        param_offset[s] + 0 -> lengthscale gradient of K_X_s
        param_offset[s] + 1 -> outputscale gradient of K_X_s (fixed at 1.0 in LMC)
        (Periodic/RQ kernels add param_offset[s] + 2 -> param1 gradient)
      ARD (use_ard=True):
        param_offset[s] + 0..d-1 -> per-dimension lengthscale gradients
        param_offset[s] + d -> outputscale gradient (fixed at 1.0 in LMC)
        (Periodic/RQ kernels add param_offset[s] + d+1 -> param1 gradient)
    
    A_s and noise gradients are computed externally by the training loop.
    """
    # R base MaterializedProviders (one per latent)
    var _providers_ptr: UnsafePointer[MaterializedProvider, MutAnyOrigin]
    var num_latents: Int       # R
    var _param_offsets: UnsafePointer[Int, MutAnyOrigin]  # [R+1] cumulative param counts
    var _total_gradient_params: Int
    
    # A_s matrices: stored contiguously [R * T * T] row-major
    var A_all_host: HostBuffer[float_dtype]
    var A_all_device: DeviceBuffer[float_dtype]
    
    # Per-task noise
    var noise_host: HostBuffer[float_dtype]
    var noise_device: DeviceBuffer[float_dtype]
    
    # Dimensions
    var n_data: Int            # n (data points, NOT nT)
    var num_tasks: Int         # T
    var dim: Int               # input dimensionality (needed for ARD)
    var use_ard: Bool          # whether using per-dimension lengthscales
    
    # GPU context and workspaces
    var ctx: DeviceContext
    var temp_kx_v: DeviceBuffer[float_dtype]     # [n * T] workspace
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        num_latents: Int,
        num_tasks: Int,
        n_data: Int,
        noise_host: HostBuffer[float_dtype],
        dim: Int = 1,
        use_ard: Bool = False,
    ) raises:
        """Create adapter with empty provider slots.
        
        After construction, call add_latent() R times to add providers and A_s.
        
        Args:
            ctx: GPU device context.
            num_latents: Number of latent kernels R.
            num_tasks: Number of tasks T.
            n_data: Number of training data points n.
            noise_host: Per-task noise [T] on host.
            dim: Input dimensionality (needed for ARD param count).
            use_ard: Whether using per-dimension lengthscales.
        """
        self.ctx = ctx
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.n_data = n_data
        self.dim = dim
        self.use_ard = use_ard
        
        # Allocate provider array
        self._providers_ptr = alloc[MaterializedProvider](num_latents)
        
        # Allocate param offsets [R+1]
        self._param_offsets = alloc[Int](num_latents + 1)
        self._param_offsets[0] = 0
        self._total_gradient_params = 0
        
        # Allocate A_s storage [R * T * T]
        var A_size = num_latents * num_tasks * num_tasks
        self.A_all_host = HostBuffer[float_dtype](ctx, A_size)
        self.A_all_device = ctx.enqueue_create_buffer[float_dtype](A_size)
        for i in range(A_size):
            self.A_all_host.unsafe_ptr()[i] = Float32(0.0)
        
        # Copy noise to device
        self.noise_host = HostBuffer[float_dtype](ctx, num_tasks)
        self.noise_device = ctx.enqueue_create_buffer[float_dtype](num_tasks)
        for t in range(num_tasks):
            self.noise_host.unsafe_ptr()[t] = noise_host.unsafe_ptr()[t]
        self.noise_device.enqueue_copy_from(self.noise_host)
        
        # Allocate workspaces
        self.temp_kx_v = ctx.enqueue_create_buffer[float_dtype](n_data * num_tasks)
        
        ctx.synchronize()
    
    fn __moveinit__(out self, owned other: Self):
        """Move constructor: transfer ownership of all fields."""
        self._providers_ptr = other._providers_ptr
        self.num_latents = other.num_latents
        self._param_offsets = other._param_offsets
        self._total_gradient_params = other._total_gradient_params
        self.A_all_host = other.A_all_host^
        self.A_all_device = other.A_all_device^
        self.noise_host = other.noise_host^
        self.noise_device = other.noise_device^
        self.n_data = other.n_data
        self.num_tasks = other.num_tasks
        self.dim = other.dim
        self.use_ard = other.use_ard
        self.ctx = other.ctx^
        self.temp_kx_v = other.temp_kx_v^
        # Nullify source pointers so __del__ on moved-from is safe
        other._providers_ptr = UnsafePointer[MaterializedProvider, MutAnyOrigin]()
        other._param_offsets = UnsafePointer[Int, MutAnyOrigin]()
        other.num_latents = 0
    
    fn __del__(owned self):
        """Clean up manually allocated memory."""
        if self._providers_ptr:
            for s in range(self.num_latents):
                (self._providers_ptr + s).destroy_pointee()
            self._providers_ptr.free()
        if self._param_offsets:
            self._param_offsets.free()
    
    # =========================================================================
    # Construction helpers
    # =========================================================================
    
    fn add_latent(
        mut self,
        latent_index: Int,
        var provider: MaterializedProvider,
        A_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Add a latent provider and its A_s matrix.
        
        Must be called in order: 0, 1, ..., R-1.
        The provider should have outputscale=1, noise=0.
        
        Args:
            latent_index: Index s (0 to R-1).
            provider: MaterializedProvider for K_X_s (moved in).
            A_host_ptr: Coregionalization matrix A_s [T × T] row-major on host.
        """
        (self._providers_ptr + latent_index).init_pointee_move(provider^)
        
        # Compute num_gradient_params for this latent's kernel type
        # Isotropic: RBF/Matern -> 2 (ls, os), Periodic/RQ -> 3 (ls, os, param1)
        # ARD: RBF/Matern -> d+1 (d ls, os), Periodic/RQ -> d+2 (d ls, os, param1)
        from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_POLYNOMIAL
        var kernel_type = self._providers_ptr[latent_index].get_kernel_type()
        var num_params: Int
        if self.use_ard:
            # ARD: d per-dim lengthscales + outputscale + optional param1
            num_params = self.dim + 1
            if kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ:
                num_params = self.dim + 2
            elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
                num_params = self.dim + 3  # d ls + os + param1(degree) + param2(offset)
        else:
            num_params = 2
            if kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ:
                num_params = 3
            elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
                num_params = 4  # ls + os + param1(degree) + param2(offset)
        
        self._param_offsets[latent_index + 1] = self._param_offsets[latent_index] + num_params
        self._total_gradient_params = self._param_offsets[latent_index + 1]
        
        # Copy A_s to host buffer
        var TT = self.num_tasks * self.num_tasks
        var offset = latent_index * TT
        for i in range(TT):
            self.A_all_host.unsafe_ptr()[offset + i] = A_host_ptr[i]
    
    fn sync_A_to_device(self) raises:
        """Copy all A_s matrices from host to device. Call after all add_latent() calls."""
        self.A_all_device.enqueue_copy_from(self.A_all_host)
        self.ctx.synchronize()
    
    fn update_A(mut self, latent_index: Int, A_host_ptr: UnsafePointer[Float32, MutAnyOrigin]) raises:
        """Update A_s matrix for latent s and sync to device."""
        var TT = self.num_tasks * self.num_tasks
        var offset = latent_index * TT
        for i in range(TT):
            self.A_all_host.unsafe_ptr()[offset + i] = A_host_ptr[i]
        self.A_all_device.enqueue_copy_from(self.A_all_host)
        self.ctx.synchronize()
    
    fn update_noise(mut self, noise_host_buf: HostBuffer[float_dtype]) raises:
        """Update per-task noise and sync to device."""
        for t in range(self.num_tasks):
            self.noise_host.unsafe_ptr()[t] = noise_host_buf.unsafe_ptr()[t]
        self.noise_device.enqueue_copy_from(self.noise_host)
        self.ctx.synchronize()
    
    # =========================================================================
    # Internal helpers
    # =========================================================================
    
    fn _find_latent_for_param(self, param_index: Int) -> Tuple[Int, Int]:
        """Map global param_index to (latent_index, local_param_index).
        
        Isotropic local param_index mapping:
            0 -> lengthscale gradient
            1 -> outputscale gradient
            2 -> param1 gradient (Periodic/RQ only)
        
        ARD local param_index mapping:
            0..d-1 -> per-dimension lengthscale gradients
            d -> outputscale gradient
            d+1 -> param1 gradient (Periodic/RQ only)
        """
        for s in range(self.num_latents):
            if param_index < self._param_offsets[s + 1]:
                return (s, param_index - self._param_offsets[s])
        return (self.num_latents - 1, 0)
    
    fn get_num_ls_params_per_latent(self) -> Int:
        """Return number of lengthscale parameters per latent (1 for isotropic, dim for ARD)."""
        if self.use_ard:
            return self.dim
        return 1
    
    fn _compute_kx_v_all_tasks_for_latent(
        self,
        latent_index: Int,
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        col: Int,
        use_gradient: Bool = False,
        local_param_index: Int = 0,
    ) raises:
        """Compute K_X_s @ v_t (or dK_X_s/d(theta) @ v_t) for all tasks.
        
        Results stored in self.temp_kx_v: [task0: n | task1: n | ... | taskT-1: n]
        
        Uses batched provider call: instead of T separate calls with num_cols=1,
        makes a single call with num_cols=T. The task-blocked layout v[t*n + i]
        matches column-major [n × T] expected by the provider (col=t, row=i).
        
        This reduces kernel launches from T to 1 per latent per CG column.
        """
        var n = self.n_data
        var nT = n * self.num_tasks
        
        # Point directly into the column's task-blocked data
        var v_col_ptr = v_ptr.offset(col * nT)
        
        # Single batched call with num_cols=T
        if use_gradient:
            if self.use_ard and local_param_index < self.dim:
                # ARD per-dimension lengthscale gradient
                self._providers_ptr[latent_index].gradient_matvec_ard(
                    self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks, local_param_index, False
                )
            elif self.use_ard:
                # ARD non-lengthscale params: translate local index to provider index
                # local d -> outputscale (provider param_index 1)
                # local d+1 -> param1 (provider param_index 2)
                # local d+2 -> param2 (provider param_index 3)
                var provider_param_index = local_param_index - self.dim + 1
                self._providers_ptr[latent_index].gradient_matvec(
                    self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks, provider_param_index, False
                )
            else:
                # Isotropic: local_param_index maps directly to provider param_index
                self._providers_ptr[latent_index].gradient_matvec(
                    self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks, local_param_index, False
                )
        else:
            # Forward matvec: K_X_s @ v (provider has noise=0)
            self._providers_ptr[latent_index].forward_matvec(
                self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks
            )
    
    # =========================================================================
    # GradientProvider Trait Implementation
    # =========================================================================
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (sum_s A_s ⊗ K_X_s + D) @ v.
        
        For each CG column c:
            1. Zero output
            2. For each latent s:
                a. For each task t: compute K_X_s @ v_t
                b. Accumulate: out += A_s-weighted K_X_s matvec results
            3. Add noise: out[t*n+i] += noise[t] * v[t*n+i]
        """
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
        
        for c in range(num_cols):
            # Step 1: Zero the output for this column
            self.ctx.enqueue_function[kernel_zero_buffer](
                out_ptr.offset(c * nT),
                nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            
            # Step 2: For each latent, accumulate A_s ⊗ K_X_s @ v
            var TT = num_t * num_t
            for s in range(self.num_latents):
                self._compute_kx_v_all_tasks_for_latent(s, v_ptr, c)
                
                self.ctx.enqueue_function[kernel_lmc_accumulate](
                    out_ptr,
                    self.temp_kx_v.unsafe_ptr(),
                    self.A_all_device.unsafe_ptr().offset(s * TT),
                    n,
                    num_t,
                    nT,
                    c,
                    grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
                )
            
            # Step 3: Add noise
            self.ctx.enqueue_function[kernel_lmc_add_noise](
                out_ptr,
                v_ptr,
                self.noise_device.unsafe_ptr(),
                n,
                num_t,
                nT,
                c,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute gradient matvec for kernel hyperparameters.
        
        Maps global param_index to (latent_s, local_param_index) via _find_latent_for_param.
        
        Isotropic (local_param_index maps directly to MaterializedProvider.gradient_matvec):
            local 0 -> lengthscale gradient (provider param_index 0)
            local 1 -> outputscale gradient (provider param_index 1)
            local 2 -> param1 gradient (provider param_index 2, Periodic/RQ)
        
        ARD (local_param_index is translated before dispatch):
            local 0..d-1 -> per-dim lengthscale gradients (via provider.gradient_matvec_ard)
            local d -> outputscale gradient (provider param_index 1)
            local d+1 -> param1 gradient (provider param_index 2, Periodic/RQ)
        
        Then computes: out = (A_s ⊗ dK_X_s/d(theta)) @ v
        """
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
        
        var latent_local = self._find_latent_for_param(param_index)
        var s = latent_local[0]
        var p_local = latent_local[1]
        var TT = num_t * num_t
        
        for c in range(num_cols):
            # Compute dK_X_s/d(theta_p_local) @ v_t for all tasks
            self._compute_kx_v_all_tasks_for_latent(
                s, v_ptr, c, use_gradient=True, local_param_index=p_local
            )
            
            # Combine with A_s: out = A_s-weighted gradient results
            self.ctx.enqueue_function[kernel_zero_buffer](
                out_ptr.offset(c * nT),
                nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            self.ctx.enqueue_function[kernel_lmc_accumulate](
                out_ptr,
                self.temp_kx_v.unsafe_ptr(),
                self.A_all_device.unsafe_ptr().offset(s * TT),
                n,
                num_t,
                nT,
                c,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
        
        if sync:
            self.ctx.synchronize()
    
    fn num_gradient_params(self) -> Int:
        """Total kernel hyperparameters across all latents."""
        return self._total_gradient_params
    
    fn get_n(self) -> Int:
        """Return nT (full system dimension)."""
        return self.n_data * self.num_tasks
    
    fn get_ctx(self) -> DeviceContext:
        """Return GPU device context."""
        return self.ctx
    
    fn get_noise(self) -> Float32:
        """Return minimum per-task noise (for preconditioner construction)."""
        var min_noise = self.noise_host.unsafe_ptr()[0]
        for t in range(1, self.num_tasks):
            if self.noise_host.unsafe_ptr()[t] < min_noise:
                min_noise = self.noise_host.unsafe_ptr()[t]
        return min_noise
    
    fn get_diagonal_value(self) -> Float32:
        """Return max diagonal value (for preconditioner construction).
        
        For stationary kernels K_X_s[i,i] = outputscale = 1.0, so
        diag[t*n+i] = sum_s A_s[t,t]. Returns the maximum over tasks.
        """
        var TT = self.num_tasks * self.num_tasks
        var max_val = Float32(0.0)
        for t in range(self.num_tasks):
            var diag_t = Float32(0.0)
            for s in range(self.num_latents):
                diag_t += self.A_all_host.unsafe_ptr()[s * TT + t * self.num_tasks + t]
            if diag_t > max_val:
                max_val = diag_t
        return max_val
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract diagonal of sum_s (A_s ⊗ K_X_s) to device buffer."""
        var nT = self.n_data * self.num_tasks
        alias BLOCK = 256
        var num_blocks = (nT + BLOCK - 1) // BLOCK
        
        self.ctx.enqueue_function[kernel_lmc_extract_diagonal](
            diag_ptr,
            self.A_all_device.unsafe_ptr(),
            self.num_latents,
            self.n_data,
            self.num_tasks,
            nT,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data from first latent's provider."""
        return self._providers_ptr[0].get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """LMC supports fused gradients when ARD is enabled and all latent
        kernel types have fused gradient implementations.
        
        Supported kernel types: RBF, Matern12/32/52, Periodic, RQ.
        Not supported: Linear, Polynomial (no ARD fused gradient kernels).
        
        When enabled, replaces total_params * T per-param kernel launches with
        R fused launches per column (e.g. 126 → 6 for R=2, d=20, T=3).
        """
        if not self.use_ard:
            return False
        
        from .constants import (
            KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52, KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ,
        )
        for s in range(self.num_latents):
            var kt = self._providers_ptr[s].get_kernel_type()
            if not (kt == KERNEL_TYPE_RBF or kt == KERNEL_TYPE_MATERN12
                    or kt == KERNEL_TYPE_MATERN32 or kt == KERNEL_TYPE_MATERN52
                    or kt == KERNEL_TYPE_PERIODIC or kt == KERNEL_TYPE_RQ):
                return False
        return True
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs for ALL latents in fused kernel launches.
        
        For each outer column c and each latent s:
          1. Extract all T task vectors from v[:, c] into a [n, T] buffer
          2. Call dispatch_fused_gradient_matvec on n points with T columns
             → produces [P_s * n * T] fused gradient output
          3. Apply A_s Kronecker weighting to scatter results into the global
             output buffer at the correct param offsets
        
        Output layout: out[(param_offset[s] + p) * nT * num_cols + c * nT + t_out * n + i]
        
        Reduces kernel launches from (total_params * T * num_cols) to
        (R * num_cols) fused launches + (R * num_cols) accumulation launches.
        """
        from .dispatchers import dispatch_fused_gradient_matvec
        from .kernel_params import (
            KernelParams,
            make_rbf_params, make_matern_params,
            make_periodic_params, make_rq_params,
        )
        from .constants import (
            KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52, KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ,
        )
        
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        
        # Find max params per latent for buffer sizing
        var max_params_per_latent = 0
        for s in range(self.num_latents):
            var ps = self._param_offsets[s + 1] - self._param_offsets[s]
            if ps > max_params_per_latent:
                max_params_per_latent = ps
        
        # Allocate temp buffers:
        #   temp_multi_task_v: [n * T] - all task vectors extracted for one column
        #   temp_fused_grad:   [max_P * n * T] - fused gradient output for one latent
        var temp_multi_task_v = self.ctx.enqueue_create_buffer[float_dtype](n * num_t)
        var temp_fused_grad = self.ctx.enqueue_create_buffer[float_dtype](
            max_params_per_latent * n * num_t
        )
        
        # Zero the entire output buffer
        var total_out_size = self._total_gradient_params * nT * num_cols
        var num_blocks_out = (total_out_size + BLOCK - 1) // BLOCK
        self.ctx.enqueue_function[kernel_zero_buffer](
            out_ptr, total_out_size,
            grid_dim=(num_blocks_out,), block_dim=(BLOCK,),
        )
        
        for c in range(num_cols):
            # Extract all T task slices from v[:, c] into temp_multi_task_v
            # Layout: temp_multi_task_v[t * n + i] = v[c * nT + t * n + i]
            # This is a contiguous copy since v is already in task-blocked layout
            var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
            self.ctx.enqueue_function[kernel_copy](
                temp_multi_task_v.unsafe_ptr(),
                v_ptr.offset(c * nT),
                nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            
            for s in range(self.num_latents):
                var num_params_s = self._param_offsets[s + 1] - self._param_offsets[s]
                var param_offset_s = self._param_offsets[s]
                var provider_ref = self._providers_ptr.offset(s)
                var kt = provider_ref[].get_kernel_type()
                
                # Build KernelParams for this latent (ARD)
                var outputscale = provider_ref[].get_outputscale()
                var lengthscale = provider_ref[].get_lengthscale()
                var ls_ptr = provider_ref[].get_lengthscales_device_ptr()
                var x_ptr = provider_ref[].get_x_ptr()
                
                var params: KernelParams
                if kt == KERNEL_TYPE_RBF:
                    params = make_rbf_params(outputscale, lengthscale, ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_MATERN12:
                    params = make_matern_params(outputscale, lengthscale, Float32(0.5), ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_MATERN32:
                    params = make_matern_params(outputscale, lengthscale, Float32(1.5), ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_MATERN52:
                    params = make_matern_params(outputscale, lengthscale, Float32(2.5), ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_PERIODIC:
                    var period = provider_ref[].get_kernel_param1()
                    params = make_periodic_params(outputscale, lengthscale, period, ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_RQ:
                    var alpha = provider_ref[].get_kernel_param1()
                    params = make_rq_params(outputscale, lengthscale, alpha, ls_ptr, is_ard=True)
                else:
                    raise Error("Unsupported kernel type for fused LMC gradient: " + String(kt))
                
                # Fused gradient: [P_s * n * T] output
                # dispatch expects v as [n, num_cols] col-major: v[col * n + row]
                # temp_multi_task_v is [n * T] in task-blocked order: v[t * n + i]
                # These match: col=t, row=i → v[t * n + i] ✓
                dispatch_fused_gradient_matvec(
                    self.ctx, kt,
                    temp_fused_grad.unsafe_ptr(),
                    x_ptr,
                    temp_multi_task_v.unsafe_ptr(),
                    n, self.dim, num_t, params,
                )
                
                # Apply A_s Kronecker weighting and accumulate into global output
                var TT = num_t * num_t
                var total_threads = num_params_s * nT
                var num_blocks_accum = (total_threads + BLOCK - 1) // BLOCK
                self.ctx.enqueue_function[kernel_lmc_fused_grad_accumulate](
                    out_ptr,
                    temp_fused_grad.unsafe_ptr(),
                    self.A_all_device.unsafe_ptr().offset(s * TT),
                    n,
                    num_t,
                    nT,
                    c,
                    num_cols,
                    num_params_s,
                    param_offset_s,
                    grid_dim=(num_blocks_accum,), block_dim=(BLOCK,),
                )
        
        self.ctx.synchronize()
        
        # Keep temp buffers alive until sync completes
        _ = temp_multi_task_v
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
        raise Error("fused_ls_os not supported")
    
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
        raise Error("fused_3param not supported for LMC")
    
    # =========================================================================
    # LMC-specific accessors (used by training loop)
    # =========================================================================
    
    fn get_param_offset(self, latent_index: Int) -> Int:
        """Get the global param offset for a given latent."""
        return self._param_offsets[latent_index]
    
    fn get_base_provider_ptr(self, latent_index: Int) -> UnsafePointer[MaterializedProvider]:
        """Get pointer to a base MaterializedProvider (for computing K_X_s @ v).
        
        Used by the training loop for A_s gradient computation.
        """
        return self._providers_ptr + latent_index


# =============================================================================
# Matrix-Free LMC Gradient Adapter
# =============================================================================

struct MatrixFreeLMCGradientAdapter(GradientProvider, Movable):
    """Matrix-free GradientProvider for LMC multi-output GP.
    
    Identical to MaterializedLMCGradientAdapter but uses MatrixFreeProvider
    instead of MaterializedProvider. This avoids materializing the n×n kernel
    matrix for each latent, reducing memory from O(R*n²) to O(R*n).
    
    The LMC kernel is:
        K_full = sum_{s=1}^{R} (A_s ⊗ K_X_s) + D
    
    where each K_X_s is computed on-the-fly via MatrixFreeProvider.
    """
    var _providers_ptr: UnsafePointer[MatrixFreeProvider, MutAnyOrigin]
    var num_latents: Int
    var _param_offsets: UnsafePointer[Int, MutAnyOrigin]
    var _total_gradient_params: Int
    
    var A_all_host: HostBuffer[float_dtype]
    var A_all_device: DeviceBuffer[float_dtype]
    
    var noise_host: HostBuffer[float_dtype]
    var noise_device: DeviceBuffer[float_dtype]
    
    var n_data: Int
    var num_tasks: Int
    var dim: Int
    var use_ard: Bool
    
    var ctx: DeviceContext
    var temp_kx_v: DeviceBuffer[float_dtype]
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        num_latents: Int,
        num_tasks: Int,
        n_data: Int,
        noise_host: HostBuffer[float_dtype],
        dim: Int = 1,
        use_ard: Bool = False,
    ) raises:
        """Create adapter with empty provider slots.
        
        After construction, call add_latent() R times to add providers and A_s.
        """
        self.ctx = ctx
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.n_data = n_data
        self.dim = dim
        self.use_ard = use_ard
        
        self._providers_ptr = alloc[MatrixFreeProvider](num_latents)
        self._param_offsets = alloc[Int](num_latents + 1)
        self._param_offsets[0] = 0
        self._total_gradient_params = 0
        
        var A_size = num_latents * num_tasks * num_tasks
        self.A_all_host = HostBuffer[float_dtype](ctx, A_size)
        self.A_all_device = ctx.enqueue_create_buffer[float_dtype](A_size)
        for i in range(A_size):
            self.A_all_host.unsafe_ptr()[i] = Float32(0.0)
        
        self.noise_host = HostBuffer[float_dtype](ctx, num_tasks)
        self.noise_device = ctx.enqueue_create_buffer[float_dtype](num_tasks)
        for t in range(num_tasks):
            self.noise_host.unsafe_ptr()[t] = noise_host.unsafe_ptr()[t]
        self.noise_device.enqueue_copy_from(self.noise_host)
        
        self.temp_kx_v = ctx.enqueue_create_buffer[float_dtype](n_data * num_tasks)
        
        ctx.synchronize()
    
    fn __moveinit__(out self, owned other: Self):
        self._providers_ptr = other._providers_ptr
        self.num_latents = other.num_latents
        self._param_offsets = other._param_offsets
        self._total_gradient_params = other._total_gradient_params
        self.A_all_host = other.A_all_host^
        self.A_all_device = other.A_all_device^
        self.noise_host = other.noise_host^
        self.noise_device = other.noise_device^
        self.n_data = other.n_data
        self.num_tasks = other.num_tasks
        self.dim = other.dim
        self.use_ard = other.use_ard
        self.ctx = other.ctx^
        self.temp_kx_v = other.temp_kx_v^
        other._providers_ptr = UnsafePointer[MatrixFreeProvider, MutAnyOrigin]()
        other._param_offsets = UnsafePointer[Int, MutAnyOrigin]()
        other.num_latents = 0
    
    fn __del__(owned self):
        if self._providers_ptr:
            for s in range(self.num_latents):
                (self._providers_ptr + s).destroy_pointee()
            self._providers_ptr.free()
        if self._param_offsets:
            self._param_offsets.free()
    
    # =========================================================================
    # Construction helpers
    # =========================================================================
    
    fn add_latent(
        mut self,
        latent_index: Int,
        var provider: MatrixFreeProvider,
        A_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Add a latent provider and its A_s matrix."""
        (self._providers_ptr + latent_index).init_pointee_move(provider^)
        
        from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_POLYNOMIAL
        var kernel_type = self._providers_ptr[latent_index].get_kernel_type()
        var num_params: Int
        if self.use_ard:
            num_params = self.dim + 1
            if kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ:
                num_params = self.dim + 2
            elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
                num_params = self.dim + 3
        else:
            num_params = 2
            if kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ:
                num_params = 3
            elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
                num_params = 4
        
        self._param_offsets[latent_index + 1] = self._param_offsets[latent_index] + num_params
        self._total_gradient_params = self._param_offsets[latent_index + 1]
        
        var TT = self.num_tasks * self.num_tasks
        var offset = latent_index * TT
        for i in range(TT):
            self.A_all_host.unsafe_ptr()[offset + i] = A_host_ptr[i]
    
    fn sync_A_to_device(self) raises:
        self.A_all_device.enqueue_copy_from(self.A_all_host)
        self.ctx.synchronize()
    
    fn update_A(mut self, latent_index: Int, A_host_ptr: UnsafePointer[Float32, MutAnyOrigin]) raises:
        var TT = self.num_tasks * self.num_tasks
        var offset = latent_index * TT
        for i in range(TT):
            self.A_all_host.unsafe_ptr()[offset + i] = A_host_ptr[i]
        self.A_all_device.enqueue_copy_from(self.A_all_host)
        self.ctx.synchronize()
    
    fn update_noise(mut self, noise_host_buf: HostBuffer[float_dtype]) raises:
        for t in range(self.num_tasks):
            self.noise_host.unsafe_ptr()[t] = noise_host_buf.unsafe_ptr()[t]
        self.noise_device.enqueue_copy_from(self.noise_host)
        self.ctx.synchronize()
    
    # =========================================================================
    # Internal helpers
    # =========================================================================
    
    fn _find_latent_for_param(self, param_index: Int) -> Tuple[Int, Int]:
        for s in range(self.num_latents):
            if param_index < self._param_offsets[s + 1]:
                return (s, param_index - self._param_offsets[s])
        return (self.num_latents - 1, 0)
    
    fn get_num_ls_params_per_latent(self) -> Int:
        if self.use_ard:
            return self.dim
        return 1
    
    fn _compute_kx_v_all_tasks_for_latent(
        self,
        latent_index: Int,
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        col: Int,
        use_gradient: Bool = False,
        local_param_index: Int = 0,
    ) raises:
        """Compute K_X_s @ v_t (or dK_X_s/d(theta) @ v_t) for all tasks.
        
        Uses batched provider call with num_cols=T instead of T separate calls.
        The task-blocked layout v[t*n + i] matches column-major [n × T].
        """
        var n = self.n_data
        var nT = n * self.num_tasks
        
        # Point directly into the column's task-blocked data
        var v_col_ptr = v_ptr.offset(col * nT)
        
        # Single batched call with num_cols=T
        if use_gradient:
            if self.use_ard and local_param_index < self.dim:
                self._providers_ptr[latent_index].gradient_matvec_ard(
                    self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks, local_param_index, False
                )
            elif self.use_ard:
                var provider_param_index = local_param_index - self.dim + 1
                self._providers_ptr[latent_index].gradient_matvec(
                    self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks, provider_param_index, False
                )
            else:
                self._providers_ptr[latent_index].gradient_matvec(
                    self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks, local_param_index, False
                )
        else:
            self._providers_ptr[latent_index].forward_matvec(
                self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks
            )
    
    # =========================================================================
    # GradientProvider Trait Implementation
    # =========================================================================
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
        
        for c in range(num_cols):
            self.ctx.enqueue_function[kernel_zero_buffer](
                out_ptr.offset(c * nT), nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            
            var TT = num_t * num_t
            for s in range(self.num_latents):
                self._compute_kx_v_all_tasks_for_latent(s, v_ptr, c)
                self.ctx.enqueue_function[kernel_lmc_accumulate](
                    out_ptr, self.temp_kx_v.unsafe_ptr(),
                    self.A_all_device.unsafe_ptr().offset(s * TT),
                    n, num_t, nT, c,
                    grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
                )
            
            self.ctx.enqueue_function[kernel_lmc_add_noise](
                out_ptr, v_ptr, self.noise_device.unsafe_ptr(),
                n, num_t, nT, c,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
        
        var latent_local = self._find_latent_for_param(param_index)
        var s = latent_local[0]
        var p_local = latent_local[1]
        var TT = num_t * num_t
        
        for c in range(num_cols):
            self._compute_kx_v_all_tasks_for_latent(
                s, v_ptr, c, use_gradient=True, local_param_index=p_local
            )
            
            self.ctx.enqueue_function[kernel_zero_buffer](
                out_ptr.offset(c * nT), nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            self.ctx.enqueue_function[kernel_lmc_accumulate](
                out_ptr, self.temp_kx_v.unsafe_ptr(),
                self.A_all_device.unsafe_ptr().offset(s * TT),
                n, num_t, nT, c,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
        
        if sync:
            self.ctx.synchronize()
    
    fn num_gradient_params(self) -> Int:
        return self._total_gradient_params
    
    fn get_n(self) -> Int:
        return self.n_data * self.num_tasks
    
    fn get_ctx(self) -> DeviceContext:
        return self.ctx
    
    fn get_noise(self) -> Float32:
        var min_noise = self.noise_host.unsafe_ptr()[0]
        for t in range(1, self.num_tasks):
            if self.noise_host.unsafe_ptr()[t] < min_noise:
                min_noise = self.noise_host.unsafe_ptr()[t]
        return min_noise
    
    fn get_diagonal_value(self) -> Float32:
        var TT = self.num_tasks * self.num_tasks
        var max_val = Float32(0.0)
        for t in range(self.num_tasks):
            var diag_t = Float32(0.0)
            for s in range(self.num_latents):
                diag_t += self.A_all_host.unsafe_ptr()[s * TT + t * self.num_tasks + t]
            if diag_t > max_val:
                max_val = diag_t
        return max_val
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        var nT = self.n_data * self.num_tasks
        alias BLOCK = 256
        var num_blocks = (nT + BLOCK - 1) // BLOCK
        self.ctx.enqueue_function[kernel_lmc_extract_diagonal](
            diag_ptr, self.A_all_device.unsafe_ptr(),
            self.num_latents, self.n_data, self.num_tasks, nT,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self._providers_ptr[0].get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """Matrix-free LMC supports fused gradients under same conditions as materialized."""
        if not self.use_ard:
            return False
        
        from .constants import (
            KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52, KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ,
        )
        for s in range(self.num_latents):
            var kt = self._providers_ptr[s].get_kernel_type()
            if not (kt == KERNEL_TYPE_RBF or kt == KERNEL_TYPE_MATERN12
                    or kt == KERNEL_TYPE_MATERN32 or kt == KERNEL_TYPE_MATERN52
                    or kt == KERNEL_TYPE_PERIODIC or kt == KERNEL_TYPE_RQ):
                return False
        return True
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Fused gradient matvec for matrix-free LMC — same algorithm as materialized.
        
        dispatch_fused_gradient_matvec computes kernel values on-the-fly, so this
        works identically for both materialized and matrix-free providers.
        """
        from .dispatchers import dispatch_fused_gradient_matvec
        from .kernel_params import (
            KernelParams,
            make_rbf_params, make_matern_params,
            make_periodic_params, make_rq_params,
        )
        from .constants import (
            KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52, KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ,
        )
        
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        
        var max_params_per_latent = 0
        for s in range(self.num_latents):
            var ps = self._param_offsets[s + 1] - self._param_offsets[s]
            if ps > max_params_per_latent:
                max_params_per_latent = ps
        
        var temp_multi_task_v = self.ctx.enqueue_create_buffer[float_dtype](n * num_t)
        var temp_fused_grad = self.ctx.enqueue_create_buffer[float_dtype](
            max_params_per_latent * n * num_t
        )
        
        var total_out_size = self._total_gradient_params * nT * num_cols
        var num_blocks_out = (total_out_size + BLOCK - 1) // BLOCK
        self.ctx.enqueue_function[kernel_zero_buffer](
            out_ptr, total_out_size,
            grid_dim=(num_blocks_out,), block_dim=(BLOCK,),
        )
        
        for c in range(num_cols):
            var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
            self.ctx.enqueue_function[kernel_copy](
                temp_multi_task_v.unsafe_ptr(),
                v_ptr.offset(c * nT),
                nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            
            for s in range(self.num_latents):
                var num_params_s = self._param_offsets[s + 1] - self._param_offsets[s]
                var param_offset_s = self._param_offsets[s]
                var provider_ref = self._providers_ptr.offset(s)
                var kt = provider_ref[].get_kernel_type()
                
                var outputscale = provider_ref[].get_outputscale()
                var lengthscale = provider_ref[].get_lengthscale()
                var ls_ptr = provider_ref[].get_lengthscales_device_ptr()
                var x_ptr = provider_ref[].get_x_ptr()
                
                var params: KernelParams
                if kt == KERNEL_TYPE_RBF:
                    params = make_rbf_params(outputscale, lengthscale, ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_MATERN12:
                    params = make_matern_params(outputscale, lengthscale, Float32(0.5), ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_MATERN32:
                    params = make_matern_params(outputscale, lengthscale, Float32(1.5), ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_MATERN52:
                    params = make_matern_params(outputscale, lengthscale, Float32(2.5), ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_PERIODIC:
                    var period = provider_ref[].get_kernel_param1()
                    params = make_periodic_params(outputscale, lengthscale, period, ls_ptr, is_ard=True)
                elif kt == KERNEL_TYPE_RQ:
                    var alpha = provider_ref[].get_kernel_param1()
                    params = make_rq_params(outputscale, lengthscale, alpha, ls_ptr, is_ard=True)
                else:
                    raise Error("Unsupported kernel type for fused LMC gradient: " + String(kt))
                
                dispatch_fused_gradient_matvec(
                    self.ctx, kt,
                    temp_fused_grad.unsafe_ptr(),
                    x_ptr,
                    temp_multi_task_v.unsafe_ptr(),
                    n, self.dim, num_t, params,
                )
                
                var TT = num_t * num_t
                var total_threads = num_params_s * nT
                var num_blocks_accum = (total_threads + BLOCK - 1) // BLOCK
                self.ctx.enqueue_function[kernel_lmc_fused_grad_accumulate](
                    out_ptr,
                    temp_fused_grad.unsafe_ptr(),
                    self.A_all_device.unsafe_ptr().offset(s * TT),
                    n,
                    num_t,
                    nT,
                    c,
                    num_cols,
                    num_params_s,
                    param_offset_s,
                    grid_dim=(num_blocks_accum,), block_dim=(BLOCK,),
                )
        
        self.ctx.synchronize()
        _ = temp_multi_task_v
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
        raise Error("fused_ls_os not supported")
    
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
        raise Error("fused_3param not supported for LMC")
    
    # =========================================================================
    # LMC-specific accessors
    # =========================================================================
    
    fn get_param_offset(self, latent_index: Int) -> Int:
        return self._param_offsets[latent_index]
    
    fn get_base_provider_ptr(self, latent_index: Int) -> UnsafePointer[MatrixFreeProvider]:
        """Get pointer to a base MatrixFreeProvider (for computing K_X_s @ v)."""
        return self._providers_ptr + latent_index
