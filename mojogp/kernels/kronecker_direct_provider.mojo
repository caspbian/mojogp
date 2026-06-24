"""KroneckerDirectProvider for Direct Kronecker CG Multi-Output GP Training.

This module implements a GradientProvider that operates on the full nT-dimensional
Kronecker system (outputscale * K_X ⊗ B + D) directly, instead of decomposing
into T separate sub-problems.

Vector layout: task-blocked ordering
    v = [v_task0; v_task1; ...; v_taskT]
    where each v_task_s is n-dimensional.

Matvec: (outputscale * K_X ⊗ B + D) @ v
    For each task s:
        out[s*n..(s+1)*n] = outputscale * sum_t(B[s,t] * K_X @ v[t*n..(t+1)*n]) + noise[s] * v[s*n..(s+1)*n]

This requires T calls to the base K_X provider per CG iteration (same cost as
the old Rakitsch approach), but solves a single nT-dimensional system instead
of T separate n-dimensional systems.

Benefits:
- Single SLQ log-det estimate (no T× error accumulation)
- Direct noise/B gradients (no eigendecomposition chain rule)
- Works with any base GradientProvider (materialized, matrix-free, ARD, composite, mixed)

Generalized Design:
- The type parameter T is GradientProvider & Movable (not just MatvecProvider)
- This allows wrapping MixedMaterializedProvider, ARDGradientAdapter, etc.
- num_gradient_params() delegates to base_provider.num_gradient_params() + 1 (outputscale)
- gradient_matvec() delegates to base for param_index < base_params, handles outputscale last
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import ceildiv

from .constants import float_dtype
from .gradient_provider import GradientProvider
from .kronecker_gpu_kernels import kernel_reshuffle_to_flat, kernel_kronecker_combine_batched


# =============================================================================
# GPU Kernels for Kronecker Direct Operations
# =============================================================================

fn kernel_kronecker_combine(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_ptrs: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    B_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    outputscale: Float32,
    n: Int,
    num_tasks: Int,
    col: Int,
) -> None:
    """Combine K_X @ v_t results with B matrix and noise for one CG column.
    
    For each task s and data point i:
        out[s*n + i, col] = outputscale * sum_t(B[s,t] * (K_X @ v_t)[i]) + noise[s] * v[s*n + i, col]
    
    kx_v_ptrs[t] points to K_X @ v_t [n] for task t.
    B is [T x T] row-major.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = n * num_tasks
    
    if idx >= UInt(total):
        return
    
    var i = Int(idx) % n       # data point index
    var s = Int(idx) // n      # task index
    
    # Compute outputscale * sum_t(B[s,t] * (K_X @ v_t)[i])
    var val = Float32(0.0)
    for t in range(num_tasks):
        val += B_ptr[s * num_tasks + t] * kx_v_ptrs[t][i]
    val *= outputscale
    
    # Add noise: noise[s] * v[s*n + i, col]
    var v_idx = col * (n * num_tasks) + s * n + i  # column-major index
    val += noise_ptr[s] * v_ptr[v_idx]
    
    var out_idx = col * (n * num_tasks) + s * n + i  # column-major index
    out_ptr[out_idx] = val


fn kernel_kronecker_combine_simple(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_buf_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n * T] contiguous: kx_v for task 0, 1, ...
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    B_ptr: UnsafePointer[Float32, MutAnyOrigin],
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    outputscale: Float32,
    n: Int,
    num_tasks: Int,
    nT: Int,
    col: Int,
) -> None:
    """Combine K_X @ v_t results with B matrix and noise for one CG column.
    
    kx_v_buf_ptr layout: [task0: n floats | task1: n floats | ... | taskT-1: n floats]
    v_ptr is the full nT * num_cols column-major buffer.
    out_ptr is the full nT * num_cols column-major buffer.
    B is [T x T] row-major.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(nT):
        return
    
    var i = Int(idx) % n       # data point index
    var s = Int(idx) // n      # task index
    
    # Compute outputscale * sum_t(B[s,t] * (K_X @ v_t)[i])
    var val = Float32(0.0)
    for t in range(num_tasks):
        val += B_ptr[s * num_tasks + t] * kx_v_buf_ptr[t * n + i]
    val *= outputscale
    
    # Add noise: noise[s] * v[s*n + i, col]
    var v_idx = col * nT + s * n + i  # column-major index in full buffer
    val += noise_ptr[s] * v_ptr[v_idx]
    
    var out_idx = col * nT + s * n + i
    out_ptr[out_idx] = val


fn kernel_kronecker_combine_no_noise(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_buf_ptr: UnsafePointer[Float32, MutAnyOrigin],
    B_ptr: UnsafePointer[Float32, MutAnyOrigin],
    outputscale: Float32,
    n: Int,
    num_tasks: Int,
    nT: Int,
    col: Int,
) -> None:
    """Combine K_X @ v_t results with B matrix, NO noise term.
    
    Used for gradient matvecs where dK/d(theta) has no noise contribution.
    
    out[s*n+i, col] = outputscale * sum_t(B[s,t] * kx_v_buf[t*n + i])
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(nT):
        return
    
    var i = Int(idx) % n
    var s = Int(idx) // n
    
    var val = Float32(0.0)
    for t in range(num_tasks):
        val += B_ptr[s * num_tasks + t] * kx_v_buf_ptr[t * n + i]
    val *= outputscale
    
    var out_idx = col * nT + s * n + i
    out_ptr[out_idx] = val


fn kernel_extract_task_block(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    nT: Int,
    task: Int,
    col: Int,
) -> None:
    """Extract task block from nT-dimensional column-major vector.
    
    Copies v[task*n..(task+1)*n, col] to out[0..n].
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    var src_idx = col * nT + task * n + Int(idx)
    out_ptr[idx] = v_ptr[src_idx]


fn kernel_extract_diagonal_kronecker(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    base_diag: Float32,
    B_ptr: UnsafePointer[Float32, MutAnyOrigin],
    outputscale: Float32,
    n: Int,
    num_tasks: Int,
    nT: Int,
) -> None:
    """Extract diagonal of (outputscale * K_X ⊗ B).
    
    For stationary kernels, K_X[i,i] = base_diag (constant).
    So diagonal element at position s*n+i = outputscale * base_diag * B[s,s].
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(nT):
        return
    
    var i = Int(idx) % n
    var s = Int(idx) // n
    diag_ptr[idx] = outputscale * base_diag * B_ptr[s * num_tasks + s]


# =============================================================================
# KroneckerDirectProvider Struct
# =============================================================================

struct KroneckerDirectProvider[T: GradientProvider & Movable](GradientProvider):
    """GradientProvider for direct Kronecker CG on the full nT-dimensional system.
    
    Wraps a base single-output GradientProvider and computes:
        (outputscale * K_X ⊗ B + D) @ v
    where v is nT-dimensional in task-blocked ordering.
    
    The base provider should have outputscale=1 and noise=0 (pure K_X).
    
    Generalized gradient parameter layout:
        [0..base_params-1]: base provider gradient params (delegated)
        [base_params]: outputscale gradient — (K_X ⊗ B) @ v (no noise)
    
    This allows any base provider (IsotropicGradientAdapter, ARDGradientAdapter,
    MaterializedCompositeGradientAdapter, MixedMaterializedProvider, etc.) to be
    used for multi-output training with the full Kronecker CG approach.
    
    B and noise gradients are computed externally from the CG solution,
    not through gradient_matvec.
    
    Attributes:
        base_provider: Underlying single-output GradientProvider (outputscale=1, noise=0)
        B_host: Task covariance B [T x T] row-major on host
        B_device: Task covariance B [T x T] row-major on GPU
        noise_host: Per-task noise [T] on host
        noise_device: Per-task noise [T] on GPU
        n_data: Number of data points n
        num_tasks: Number of tasks T
        outputscale: Global output scale
        ctx: GPU device context
        temp_kx_v: Workspace [n * T] for K_X @ v_t results
    """
    var base_provider: T
    var B_host: HostBuffer[float_dtype]
    var B_device: DeviceBuffer[float_dtype]
    var noise_host: HostBuffer[float_dtype]
    var noise_device: DeviceBuffer[float_dtype]
    var n_data: Int
    var num_tasks: Int
    var outputscale: Float32
    var ctx: DeviceContext
    var temp_kx_v: DeviceBuffer[float_dtype]     # [n * T] workspace for K_X @ v_t results
    
    fn __init__(
        out self,
        owned base_provider: T,
        ctx: DeviceContext,
        num_tasks: Int,
        outputscale: Float32,
        B_host: HostBuffer[float_dtype],
        noise_host: HostBuffer[float_dtype],
    ) raises:
        """Create KroneckerDirectProvider.
        
        Args:
            base_provider: Underlying GradientProvider for K_X (outputscale=1, noise=0).
            ctx: GPU device context.
            num_tasks: Number of tasks T.
            outputscale: Global output scale.
            B_host: Task covariance B [T x T] row-major on host.
            noise_host: Per-task noise [T] on host.
        """
        self.n_data = base_provider.get_n()
        self.base_provider = base_provider^
        self.ctx = ctx
        self.num_tasks = num_tasks
        self.outputscale = outputscale
        
        # Copy B to device
        self.B_host = HostBuffer[float_dtype](ctx, num_tasks * num_tasks)
        self.B_device = ctx.enqueue_create_buffer[float_dtype](num_tasks * num_tasks)
        for i in range(num_tasks * num_tasks):
            self.B_host.unsafe_ptr()[i] = B_host.unsafe_ptr()[i]
        self.B_device.enqueue_copy_from(self.B_host)
        
        # Copy noise to device
        self.noise_host = HostBuffer[float_dtype](ctx, num_tasks)
        self.noise_device = ctx.enqueue_create_buffer[float_dtype](num_tasks)
        for t in range(num_tasks):
            self.noise_host.unsafe_ptr()[t] = noise_host.unsafe_ptr()[t]
        self.noise_device.enqueue_copy_from(self.noise_host)
        
        # Allocate workspaces
        self.temp_kx_v = ctx.enqueue_create_buffer[float_dtype](self.n_data * num_tasks)
        
        ctx.synchronize()
    
    fn _compute_kx_v_all_tasks(
        self,
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        col: Int,
        use_gradient: Bool = False,
        param_index: Int = 0,
    ) raises:
        """Compute K_X @ v_t (or dK_X/d(theta) @ v_t) for all tasks t, for one CG column.
        
        Results stored in self.temp_kx_v: [task0: n | task1: n | ... | taskT-1: n]
        
        Uses batched base provider call: instead of T separate calls with num_cols=1,
        makes a single call with num_cols=T. The task-blocked layout v[t*n + i] matches
        the column-major layout v[col*n + row] expected by the base provider (col=t, row=i).
        
        This reduces kernel launches from T to 1 per CG column.
        
        Args:
            v_ptr: Full nT * num_cols column-major buffer.
            col: Which CG column to process.
            use_gradient: If True, use gradient_matvec instead of forward_matvec.
            param_index: Which gradient parameter (only used if use_gradient=True).
        """
        var n = self.n_data
        var nT = n * self.num_tasks
        
        # Point directly into the column's task-blocked data: v[col*nT .. col*nT + nT]
        # Layout: [task0: n | task1: n | ... | taskT-1: n]
        # This matches column-major [n × T] with col=task, row=data_point
        var v_col_ptr = v_ptr.offset(col * nT)
        
        # Single batched call with num_cols=T instead of T calls with num_cols=1
        if use_gradient:
            self.base_provider.gradient_matvec(
                self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks, param_index, False
            )
        else:
            self.base_provider.forward_matvec(
                self.temp_kx_v.unsafe_ptr(), v_col_ptr, self.num_tasks
            )
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (outputscale * K_X ⊗ B + D) @ v.
        
        GPU-batched: reshuffles all columns, single base matvec with T*num_cols
        columns, then GPU B-matrix combine. 3 kernel launches total.
        
        Args:
            out_ptr: Output [nT * num_cols] column-major on device.
            v_ptr: Input [nT * num_cols] column-major on device.
            num_cols: Number of CG columns.
        """
        var n = self.n_data
        var num_tasks = self.num_tasks
        var nT = n * num_tasks
        var total_base_cols = num_tasks * num_cols
        alias BLOCK = 256

        var v_flat = self.ctx.enqueue_create_buffer[float_dtype](n * total_base_cols)
        var kx_flat = self.ctx.enqueue_create_buffer[float_dtype](n * total_base_cols)

        # GPU reshuffle: Kronecker layout → flat batch columns
        var grid_rs = ceildiv(n * total_base_cols, BLOCK)
        self.ctx.enqueue_function[kernel_reshuffle_to_flat](
            v_flat.unsafe_ptr(), v_ptr, n, num_tasks, num_cols,
            grid_dim=grid_rs, block_dim=BLOCK,
        )
        self.ctx.synchronize()

        # Single base forward_matvec with all T*num_cols columns
        self.base_provider.forward_matvec(kx_flat.unsafe_ptr(), v_flat.unsafe_ptr(), total_base_cols)

        # GPU B-matrix combine + noise
        var grid_kc = ceildiv(nT, BLOCK)
        self.ctx.enqueue_function[kernel_kronecker_combine_batched](
            out_ptr, kx_flat.unsafe_ptr(), v_ptr,
            self.B_device.unsafe_ptr(), self.noise_device.unsafe_ptr(),
            n, num_tasks, num_cols, self.outputscale,
            grid_dim=grid_kc, block_dim=BLOCK,
        )
        self.ctx.synchronize()

        _ = v_flat
        _ = kx_flat
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute gradient matvec for kernel hyperparameters.
        
        Generalized parameter layout:
            param_index 0..base_params-1: delegated to base_provider.gradient_matvec()
                Each computes: outputscale * (dK_X/d(theta_i) ⊗ B) @ v
            param_index == base_params: outputscale gradient
                Computes: (K_X ⊗ B) @ v (no noise, outputscale=1)
        
        B and noise gradients are NOT computed here — they are computed
        directly from the CG solution externally.
        
        Args:
            out_ptr: Output [nT * num_cols] column-major on device.
            v_ptr: Input [nT * num_cols] column-major on device.
            num_cols: Number of CG columns.
            param_index: Which parameter gradient to compute.
            sync: Whether to synchronize.
        """
        var n = self.n_data
        var num_tasks = self.num_tasks
        var nT = n * num_tasks
        var base_params = self.base_provider.num_gradient_params()
        alias BLOCK = 256
        
        if param_index < base_params:
            # GPU-batched: reshuffle, single base gradient call, B combine
            var total_base_cols = num_tasks * num_cols
            var v_flat = self.ctx.enqueue_create_buffer[float_dtype](n * total_base_cols)
            var grad_flat = self.ctx.enqueue_create_buffer[float_dtype](n * total_base_cols)
            self.ctx.synchronize()
            
            var grid_rs = ceildiv(n * total_base_cols, BLOCK)
            self.ctx.enqueue_function[kernel_reshuffle_to_flat](
                v_flat.unsafe_ptr(), v_ptr, n, num_tasks, num_cols,
                grid_dim=grid_rs, block_dim=BLOCK,
            )
            self.ctx.synchronize()
            
            self.base_provider.gradient_matvec(
                grad_flat.unsafe_ptr(), v_flat.unsafe_ptr(), total_base_cols, param_index, True
            )
            
            # Zero noise for gradient combine
            var zero_noise = self.ctx.enqueue_create_host_buffer[float_dtype](num_tasks)
            self.ctx.synchronize()
            for ti in range(num_tasks):
                zero_noise.unsafe_ptr()[ti] = Float32(0)
            var zero_noise_dev = self.ctx.enqueue_create_buffer[float_dtype](num_tasks)
            self.ctx.enqueue_copy(zero_noise_dev, zero_noise)
            self.ctx.synchronize()
            
            var grid_kc = ceildiv(nT, BLOCK)
            self.ctx.enqueue_function[kernel_kronecker_combine_batched](
                out_ptr, grad_flat.unsafe_ptr(), v_ptr,
                self.B_device.unsafe_ptr(), zero_noise_dev.unsafe_ptr(),
                n, num_tasks, num_cols, self.outputscale,
                grid_dim=grid_kc, block_dim=BLOCK,
            )
            if sync:
                self.ctx.synchronize()
            _ = v_flat
            _ = grad_flat
            _ = zero_noise
            _ = zero_noise_dev
        else:
            # Outputscale: dK/d(os) = (K_X ⊗ B) @ v (no noise, outputscale=1)
            var total_base_cols2 = num_tasks * num_cols
            var v_flat2 = self.ctx.enqueue_create_buffer[float_dtype](n * total_base_cols2)
            var kx_flat2 = self.ctx.enqueue_create_buffer[float_dtype](n * total_base_cols2)
            self.ctx.synchronize()
            
            var grid_rs2 = ceildiv(n * total_base_cols2, BLOCK)
            self.ctx.enqueue_function[kernel_reshuffle_to_flat](
                v_flat2.unsafe_ptr(), v_ptr, n, num_tasks, num_cols,
                grid_dim=grid_rs2, block_dim=BLOCK,
            )
            self.ctx.synchronize()
            self.base_provider.forward_matvec(kx_flat2.unsafe_ptr(), v_flat2.unsafe_ptr(), total_base_cols2)
            
            var zero_noise2 = self.ctx.enqueue_create_host_buffer[float_dtype](num_tasks)
            self.ctx.synchronize()
            for ti in range(num_tasks):
                zero_noise2.unsafe_ptr()[ti] = Float32(0)
            var zero_noise_dev2 = self.ctx.enqueue_create_buffer[float_dtype](num_tasks)
            self.ctx.enqueue_copy(zero_noise_dev2, zero_noise2)
            self.ctx.synchronize()
            
            var grid_kc2 = ceildiv(nT, BLOCK)
            self.ctx.enqueue_function[kernel_kronecker_combine_batched](
                out_ptr, kx_flat2.unsafe_ptr(), v_ptr,
                self.B_device.unsafe_ptr(), zero_noise_dev2.unsafe_ptr(),
                n, num_tasks, num_cols, Float32(1.0),
                grid_dim=grid_kc2, block_dim=BLOCK,
            )
            if sync:
                self.ctx.synchronize()
            _ = v_flat2
            _ = kx_flat2
            _ = zero_noise2
            _ = zero_noise_dev2
    
    fn num_gradient_params(self) -> Int:
        """Return base_provider.num_gradient_params() + 1 (for outputscale).
        
        B and noise gradients are computed separately externally.
        """
        return self.base_provider.num_gradient_params() + 1
    
    fn get_n(self) -> Int:
        """Return nT (full system dimension)."""
        return self.n_data * self.num_tasks
    
    fn get_ctx(self) -> DeviceContext:
        """Return GPU device context."""
        return self.ctx
    
    fn get_noise(self) -> Float32:
        """Return a representative noise value.
        
        For the Kronecker system, noise varies by task. This returns the
        minimum noise (used for preconditioner construction).
        """
        var min_noise = self.noise_host.unsafe_ptr()[0]
        for t in range(1, self.num_tasks):
            if self.noise_host.unsafe_ptr()[t] < min_noise:
                min_noise = self.noise_host.unsafe_ptr()[t]
        return min_noise
    
    fn get_diagonal_value(self) -> Float32:
        """Return max diagonal value (for preconditioner construction)."""
        var max_val = Float32(0.0)
        for s in range(self.num_tasks):
            var diag_s = self.outputscale * self.B_host.unsafe_ptr()[s * self.num_tasks + s]
            if diag_s > max_val:
                max_val = diag_s
        return max_val
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract diagonal of (outputscale * K_X ⊗ B) to device buffer.
        
        For stationary kernels, K_X[i,i] = 1.0 (base outputscale=1).
        So diag[s*n+i] = outputscale * B[s,s].
        """
        var nT = self.n_data * self.num_tasks
        alias BLOCK = 256
        var num_blocks = (nT + BLOCK - 1) // BLOCK
        
        self.ctx.enqueue_function[kernel_extract_diagonal_kronecker](
            diag_ptr,
            Float32(1.0),  # base K_X diagonal for stationary kernels
            self.B_device.unsafe_ptr(),
            self.outputscale,
            self.n_data,
            self.num_tasks,
            nT,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data on device."""
        return self.base_provider.get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """Kronecker direct provider doesn't support all-param fused gradients (ARD)."""
        return False
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Not supported."""
        raise Error("fused_gradient_matvec not supported for KroneckerDirectProvider")
    
    fn supports_fused_ls_os(self) -> Bool:
        """Delegate to base provider — if base supports fused ls+os, so do we.
        
        This enables the BBMM to compute ls and os gradient matvecs in a single
        O(n²×T) pass instead of 2 separate passes per CG column. 2.5x faster.
        """
        return self.base_provider.supports_fused_ls_os()
    
    fn fused_ls_os_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Fused ls + os gradient for Kronecker system.
        
        Batches ALL num_cols CG columns into a single base provider call with
        num_cols * T columns. The memory layout is compatible: v_ptr is
        [nT × num_cols] column-major = [n × (num_cols × T)] column-major
        since nT = n × T. This gives 1 kernel launch instead of num_cols launches.
        
        Then applies Kronecker combine kernels for each CG column.
        """
        var n = self.n_data
        var nT = n * self.num_tasks
        var total_cols = num_cols * self.num_tasks  # e.g. 10 probes × 3 tasks = 30
        alias BLOCK = 256
        
        # Single-pass: all num_cols*T columns in one fused kernel launch.
        # The dispatcher has specializations for NCOLS=20,30,50 (T=2,3,5 × J=10).
        # v_ptr layout [nT × num_cols] = [n × total_cols] — same memory, compatible.
        # Benchmarked 1.4-2.6x faster than multi-pass (see scripts/test_ncols_strategy.mojo).
        var temp_ls_kx_v = self.ctx.enqueue_create_buffer[float_dtype](n * total_cols)
        var temp_os_kx_v = self.ctx.enqueue_create_buffer[float_dtype](n * total_cols)
        
        self.base_provider.fused_ls_os_gradient_matvec(
            temp_ls_kx_v.unsafe_ptr(),
            temp_os_kx_v.unsafe_ptr(),
            v_ptr,
            total_cols,
        )
        
        # Apply Kronecker combine for each CG column.
        # temp_ls_kx_v is [n × total_cols] column-major (base provider output),
        # so column c starts at offset c * n (NOT c * nT).
        var num_blocks = (nT + BLOCK - 1) // BLOCK
        for c in range(num_cols):
            self.ctx.enqueue_function[kernel_kronecker_combine_no_noise](
                ls_out_ptr,
                temp_ls_kx_v.unsafe_ptr().offset(c * n),
                self.B_device.unsafe_ptr(),
                self.outputscale,
                n, self.num_tasks, nT, c,
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )
            self.ctx.enqueue_function[kernel_kronecker_combine_no_noise](
                os_out_ptr,
                temp_os_kx_v.unsafe_ptr().offset(c * n),
                self.B_device.unsafe_ptr(),
                Float32(1.0),
                n, self.num_tasks, nT, c,
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )
        
        _ = temp_ls_kx_v  # keepalive
        _ = temp_os_kx_v  # keepalive
    
    fn supports_fused_3param(self) -> Bool:
        """Delegate to base provider for 3-param (Periodic/RQ) fusion."""
        return self.base_provider.supports_fused_3param()
    
    fn fused_3param_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Fused 3-param gradient for Kronecker system (Periodic/RQ).
        
        Batches all num_cols CG columns into a single base provider call.
        """
        var n = self.n_data
        var nT = n * self.num_tasks
        var total_cols = num_cols * self.num_tasks
        alias BLOCK = 256
        
        # Single-pass: all num_cols*T columns in one fused kernel launch.
        var temp_ls_kx_v = self.ctx.enqueue_create_buffer[float_dtype](n * total_cols)
        var temp_p1_kx_v = self.ctx.enqueue_create_buffer[float_dtype](n * total_cols)
        var temp_os_kx_v = self.ctx.enqueue_create_buffer[float_dtype](n * total_cols)
        
        self.base_provider.fused_3param_gradient_matvec(
            temp_ls_kx_v.unsafe_ptr(),
            temp_p1_kx_v.unsafe_ptr(),
            temp_os_kx_v.unsafe_ptr(),
            v_ptr,
            total_cols,
        )
        
        # temp_*_kx_v are [n × total_cols] column-major (base provider output),
        # so column c starts at offset c * n (NOT c * nT).
        var num_blocks = (nT + BLOCK - 1) // BLOCK
        for c in range(num_cols):
            self.ctx.enqueue_function[kernel_kronecker_combine_no_noise](
                ls_out_ptr, temp_ls_kx_v.unsafe_ptr().offset(c * n),
                self.B_device.unsafe_ptr(), self.outputscale,
                n, self.num_tasks, nT, c,
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )
            self.ctx.enqueue_function[kernel_kronecker_combine_no_noise](
                p1_out_ptr, temp_p1_kx_v.unsafe_ptr().offset(c * n),
                self.B_device.unsafe_ptr(), self.outputscale,
                n, self.num_tasks, nT, c,
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )
            self.ctx.enqueue_function[kernel_kronecker_combine_no_noise](
                os_out_ptr, temp_os_kx_v.unsafe_ptr().offset(c * n),
                self.B_device.unsafe_ptr(), Float32(1.0),
                n, self.num_tasks, nT, c,
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )
        
        _ = temp_ls_kx_v  # keepalive
        _ = temp_p1_kx_v  # keepalive
        _ = temp_os_kx_v  # keepalive
    
    # =========================================================================
    # Update methods for training loop
    # =========================================================================
    
    fn update_B(mut self, B_host: HostBuffer[float_dtype]) raises:
        """Update task covariance B matrix.
        
        Args:
            B_host: New B [T x T] row-major on host.
        """
        for i in range(self.num_tasks * self.num_tasks):
            self.B_host.unsafe_ptr()[i] = B_host.unsafe_ptr()[i]
        self.B_device.enqueue_copy_from(self.B_host)
        self.ctx.synchronize()
    
    fn update_noise(mut self, noise_host: HostBuffer[float_dtype]) raises:
        """Update per-task noise.
        
        Args:
            noise_host: New per-task noise [T] on host.
        """
        for t in range(self.num_tasks):
            self.noise_host.unsafe_ptr()[t] = noise_host.unsafe_ptr()[t]
        self.noise_device.enqueue_copy_from(self.noise_host)
        self.ctx.synchronize()
    
    fn update_outputscale(mut self, outputscale: Float32):
        """Update global output scale."""
        self.outputscale = outputscale
