"""LMC Composite Provider - GradientProvider for LMC with composite kernels.

Implements MaterializedLMCCompositeGradientAdapter[DIM, K] which holds R
MaterializedCompositeProvider[DIM, K] instances for LMC multi-output GP
training with composite (sum/product) kernels.

All R latents share the same composite kernel STRUCTURE but with different
learned parameters. This is the practical use case (e.g., all latents use
RBF + Matern52 but each with its own lengthscales).

This file is parameterized by [DIM, K: ComposableKernel] and is imported
by JIT-generated code, NOT by static .so bindings.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from memory.unsafe_pointer import alloc

from .constants import float_dtype
from .gradient_provider import GradientProvider
from .composite_provider import MaterializedCompositeProvider
from .composable_kernel import ComposableKernel


# GPU kernels for LMC accumulation (reused from lmc_provider.mojo)
fn kernel_zero_buffer(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < UInt(n):
        buf[i] = Float32(0.0)


fn kernel_lmc_composite_accumulate(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    A_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    T: Int,
    s_offset: Int,  # offset into A_all for latent s: s * T * T
) -> None:
    """Accumulate A_s-weighted K_X_s matvec results into output.
    
    For each (t_out, i): out[t_out*n + i] += sum_t A_s[t_out,t] * kx_v[t*n + i]
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var nT = n * T
    if idx < UInt(nT):
        var t_out = Int(idx) // n
        var i = Int(idx) % n
        var acc = Float32(0.0)
        for t in range(T):
            acc += A_ptr[s_offset + t_out * T + t] * kx_v_ptr[t * n + i]
        out_ptr[t_out * n + i] += acc


fn kernel_lmc_composite_add_noise(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    T: Int,
) -> None:
    """Add per-task noise: out[t*n + i] += noise[t] * v[t*n + i]."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var nT = n * T
    if idx < UInt(nT):
        var t = Int(idx) // n
        out_ptr[Int(idx)] += noise_ptr[t] * v_ptr[Int(idx)]


fn kernel_lmc_composite_extract_diagonal(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    A_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    T: Int,
    R: Int,
) -> None:
    """Extract diagonal of full LMC kernel: diag[t*n+i] = sum_s A_s[t,t] * K_X_s[i,i] + noise[t]."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var nT = n * T
    if idx < UInt(nT):
        var t = Int(idx) // n
        var i = Int(idx) % n
        var val = noise_ptr[t]
        for s in range(R):
            val += A_ptr[s * T * T + t * T + t] * kx_diag_ptr[s * n + i]
        diag_ptr[Int(idx)] = val


struct MaterializedLMCCompositeGradientAdapter[DIM: Int, K: ComposableKernel](GradientProvider, Movable):
    """GradientProvider for LMC with composite kernels.
    
    Holds R MaterializedCompositeProvider[DIM, K] instances (one per latent).
    All latents share the same kernel structure K but with different parameters.
    
    The LMC kernel is:
        K_full = sum_{s=1}^{R} (A_s ⊗ K_X_s) + D
    
    where each K_X_s is a composite kernel evaluated by MaterializedCompositeProvider.
    
    Gradient parameter layout (per latent s):
        param_offset[s] + 0..K.num_params()-1 -> composite kernel parameter gradients
    
    A_s and noise gradients are computed externally by the training loop.
    """
    # R base MaterializedCompositeProviders (one per latent)
    var _providers_ptr: UnsafePointer[MaterializedCompositeProvider[DIM, K], MutAnyOrigin]
    var num_latents: Int       # R
    var _param_offsets: UnsafePointer[Int, MutAnyOrigin]  # [R+1] cumulative param counts
    var _total_gradient_params: Int
    
    # A_s matrices: stored contiguously [R * T * T] row-major
    var A_all_host: HostBuffer[float_dtype]
    var A_all_device: DeviceBuffer[float_dtype]
    
    # Per-task noise
    var noise_host: HostBuffer[float_dtype]
    var noise_device: DeviceBuffer[float_dtype]
    
    # Per-latent diagonal cache [R * n] for extract_diagonal
    var _kx_diag_device: DeviceBuffer[float_dtype]
    
    # Dimensions
    var n_data: Int            # n (data points, NOT nT)
    var num_tasks: Int         # T
    
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
    ) raises:
        """Create adapter with empty provider slots.
        
        After construction, call add_latent() R times to add providers and A_s.
        """
        self.ctx = ctx
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.n_data = n_data
        
        # Allocate provider array
        self._providers_ptr = alloc[MaterializedCompositeProvider[DIM, K]](num_latents)
        
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
        
        # Per-latent diagonal cache
        self._kx_diag_device = ctx.enqueue_create_buffer[float_dtype](num_latents * n_data)
        
        # Allocate workspaces
        self.temp_kx_v = ctx.enqueue_create_buffer[float_dtype](n_data * num_tasks)
        
        ctx.synchronize()
    
    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self._providers_ptr = other._providers_ptr
        self.num_latents = other.num_latents
        self._param_offsets = other._param_offsets
        self._total_gradient_params = other._total_gradient_params
        self.A_all_host = other.A_all_host^
        self.A_all_device = other.A_all_device^
        self.noise_host = other.noise_host^
        self.noise_device = other.noise_device^
        self._kx_diag_device = other._kx_diag_device^
        self.n_data = other.n_data
        self.num_tasks = other.num_tasks
        self.ctx = other.ctx^
        self.temp_kx_v = other.temp_kx_v^
        # Nullify source pointers
        other._providers_ptr = UnsafePointer[MaterializedCompositeProvider[DIM, K], MutAnyOrigin]()
        other._param_offsets = UnsafePointer[Int, MutAnyOrigin]()
        other.num_latents = 0
    
    fn __del__(deinit self):
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
        var provider: MaterializedCompositeProvider[DIM, K],
        A_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Add a latent provider and its A_s matrix.
        
        Must be called in order: 0, 1, ..., R-1.
        The provider should have noise=0 (noise is handled by the LMC adapter).
        """
        (self._providers_ptr + latent_index).init_pointee_move(provider^)
        
        # Each latent has K.num_params() gradient parameters
        alias num_params = K.num_params()
        self._param_offsets[latent_index + 1] = self._param_offsets[latent_index] + num_params
        self._total_gradient_params = self._param_offsets[latent_index + 1]
        
        # Copy A_s to host buffer
        var TT = self.num_tasks * self.num_tasks
        var offset = latent_index * TT
        for i in range(TT):
            self.A_all_host.unsafe_ptr()[offset + i] = A_host_ptr[i]
    
    fn sync_A_to_device(self) raises:
        """Copy all A_s matrices from host to device."""
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
        """Map global param_index to (latent_index, local_param_index)."""
        for s in range(self.num_latents):
            if param_index < self._param_offsets[s + 1]:
                return (s, param_index - self._param_offsets[s])
        return (self.num_latents - 1, 0)
    
    fn get_num_params_per_latent(self) -> Int:
        """Return number of gradient parameters per latent."""
        return K.num_params()
    
    fn get_param_offset(self, latent_index: Int) -> Int:
        """Return the global param offset for a given latent."""
        return self._param_offsets[latent_index]
    
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
        
        Uses batched provider call with num_cols=T instead of T separate calls.
        The task-blocked layout v[t*n + i] matches column-major [n × T].
        """
        var n = self.n_data
        var nT = n * self.num_tasks
        
        # Point directly into the column's task-blocked data
        var v_col_ptr = v_ptr.offset(col * nT)
        
        # Single batched call with num_cols=T
        if use_gradient:
            # Composite kernel: flat param_index maps directly
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
        """Compute (sum_s A_s ⊗ K_X_s + D) @ v."""
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
        
        for c in range(num_cols):
            # Zero output
            self.ctx.enqueue_function[kernel_zero_buffer](
                out_ptr.offset(c * nT), nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            
            # Accumulate sum_s (A_s ⊗ K_X_s) @ v
            for s in range(self.num_latents):
                self._compute_kx_v_all_tasks_for_latent(s, v_ptr, c)
                self.ctx.enqueue_function[kernel_lmc_composite_accumulate](
                    out_ptr.offset(c * nT),
                    self.temp_kx_v.unsafe_ptr(),
                    self.A_all_device.unsafe_ptr(),
                    n, num_t, s * num_t * num_t,
                    grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
                )
            
            # Add noise
            self.ctx.enqueue_function[kernel_lmc_composite_add_noise](
                out_ptr.offset(c * nT),
                v_ptr.offset(c * nT),
                self.noise_device.unsafe_ptr(),
                n, num_t,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
        
        self.ctx.synchronize()
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute (A_s ⊗ dK_X_s/d(theta_p)) @ v for the latent s that owns param_index."""
        var s_and_local = self._find_latent_for_param(param_index)
        var s = s_and_local[0]
        var local_p = s_and_local[1]
        
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
        
        for c in range(num_cols):
            # Zero output
            self.ctx.enqueue_function[kernel_zero_buffer](
                out_ptr.offset(c * nT), nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            
            # Compute dK_X_s/d(theta_p) @ v_t for all tasks
            self._compute_kx_v_all_tasks_for_latent(s, v_ptr, c, use_gradient=True, local_param_index=local_p)
            
            # Accumulate with A_s weighting
            self.ctx.enqueue_function[kernel_lmc_composite_accumulate](
                out_ptr.offset(c * nT),
                self.temp_kx_v.unsafe_ptr(),
                self.A_all_device.unsafe_ptr(),
                n, num_t, s * num_t * num_t,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
        
        if sync:
            self.ctx.synchronize()
    
    fn gradient_matvec_ard(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        dim_index: Int,
        sync: Bool = True,
    ) raises:
        """Not used for composite kernels — composite kernels use flat param_index."""
        # Composite kernels don't have a separate ARD path; all params go through gradient_matvec
        self.gradient_matvec(out_ptr, v_ptr, num_cols, dim_index, sync)
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract diagonal of full LMC kernel."""
        var n = self.n_data
        var num_t = self.num_tasks
        var nT = n * num_t
        alias BLOCK = 256
        
        # Get per-latent diagonals
        for s in range(self.num_latents):
            self._providers_ptr[s].extract_diagonal(
                self._kx_diag_device.unsafe_ptr().offset(s * n)
            )
        
        # Combine: diag[t*n+i] = sum_s A_s[t,t] * K_X_s[i,i] + noise[t]
        var num_blocks = (nT + BLOCK - 1) // BLOCK
        self.ctx.enqueue_function[kernel_lmc_composite_extract_diagonal](
            diag_ptr,
            self._kx_diag_device.unsafe_ptr(),
            self.A_all_device.unsafe_ptr(),
            self.noise_device.unsafe_ptr(),
            n, num_t, self.num_latents,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        self.ctx.synchronize()
    
    fn num_gradient_params(self) -> Int:
        """Total number of gradient parameters across all latents."""
        return self._total_gradient_params
    
    fn get_n(self) -> Int:
        """Return nT (total dimension of the system)."""
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
        
        For composite kernels, the diagonal depends on the kernel structure.
        Returns a conservative estimate based on A_s diagonal entries.
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
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data from first latent's provider."""
        return self._providers_ptr[0].get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """Composite LMC does not support fused gradients (composite kernels have
        their own gradient dispatch mechanism)."""
        return False
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Not supported for composite LMC."""
        raise Error("fused_gradient_matvec not supported for MaterializedLMCCompositeGradientAdapter")
    
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
    
    # =========================================================================
    # LMC-specific accessors (used by training loop)
    # =========================================================================
    
    fn get_base_provider_ptr(self, latent_index: Int) -> UnsafePointer[MaterializedCompositeProvider[DIM, K]]:
        """Get pointer to a base MaterializedCompositeProvider."""
        return self._providers_ptr + latent_index
