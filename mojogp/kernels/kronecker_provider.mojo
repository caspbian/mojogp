"""KroneckerBatchedProvider for multi-output GP training.

This module provides the KroneckerBatchedProvider struct that wraps a base
single-output MatvecProvider and adds per-sub-problem eigenvalue scaling for
batched Kronecker multi-output CG.

The Kronecker decomposition produces T sub-problems with different effective
scales s_t = outputscale * lambda_t. In the batched CG, all T sub-problems
are solved simultaneously with T * (1 + P) columns (where P = num_probes).

Column layout: T groups of (1 + P) columns each.
Columns [t*(1+P), (t+1)*(1+P)) belong to sub-problem t.

Gradient parameters exposed:
    0..K-1: kernel hyperparameters (lengthscale, etc.)
            gradient_matvec returns s_t * dK_X/d(theta) @ v (per-column scaling)
    K:      scale gradient
            gradient_matvec returns K_X @ v (no scaling, same for all columns)

The training loop post-processes the scale gradient to get:
    dNLL/d(lambda_t) = dNLL_t/d(s_t) * outputscale
    dNLL/d(outputscale) = sum_t dNLL_t/d(s_t) * lambda_t
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

from .constants import float_dtype
from .matvec_provider import MatvecProvider
from .gradient_provider import GradientProvider


# =============================================================================
# GPU Kernels for Per-Column Scaling
# =============================================================================

fn kernel_scale_columns_by_task(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_ptr: UnsafePointer[Float32, MutAnyOrigin],   # K_X @ V result (from base provider)
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],       # original V (for noise addition)
    scales_ptr: UnsafePointer[Float32, MutAnyOrigin],  # T effective scales on device
    noise: Float32,
    n: Int,
    cols_per_task: Int,
    num_tasks: Int,
) -> None:
    """For each element (i, col):
       out[i,col] = scales[task(col)] * kx_v[i,col] + noise * v[i,col]
    where task(col) = col // cols_per_task
    
    Memory layout: column-major [n * total_cols]
    Element (i, col) is at index col * n + i
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total_elements = n * num_tasks * cols_per_task
    
    if idx >= UInt(total_elements):
        return
    
    # Compute row and column from linear index (column-major)
    var col = Int(idx) // n
    var task = col // cols_per_task
    var scale = scales_ptr[task]
    
    out_ptr[idx] = scale * kx_v_ptr[idx] + noise * v_ptr[idx]


fn kernel_scale_columns_by_task_no_noise(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scales_ptr: UnsafePointer[Float32, MutAnyOrigin],  # T effective scales on device
    n: Int,
    cols_per_task: Int,
    num_tasks: Int,
) -> None:
    """In-place scaling: out[i,col] *= scales[task(col)]
    
    Memory layout: column-major [n * total_cols]
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total_elements = n * num_tasks * cols_per_task
    
    if idx >= UInt(total_elements):
        return
    
    var col = Int(idx) // n
    var task = col // cols_per_task
    out_ptr[idx] *= scales_ptr[task]


fn kernel_fill_diagonal_scaled(
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    base_diag_value: Float32,  # outputscale from base kernel (should be 1.0)
    scales_ptr: UnsafePointer[Float32, MutAnyOrigin],  # T effective scales
    n: Int,
    num_tasks: Int,
) -> None:
    """Fill diagonal for Kronecker provider.
    
    For multi-output, the diagonal varies by task. But since we're doing
    batched CG with all tasks together, we just use the max scale for
    the preconditioner diagonal estimate.
    
    This kernel fills diag[i] = base_diag_value (constant for stationary kernels).
    The actual scaling is handled by the preconditioner set.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(n):
        return
    
    diag_ptr[idx] = base_diag_value


# =============================================================================
# KroneckerBatchedProvider Struct
# =============================================================================

struct KroneckerBatchedProvider[T: MatvecProvider & Movable](GradientProvider):
    """GradientProvider for batched Kronecker multi-output CG.
    
    Wraps a base single-output provider and adds per-sub-problem eigenvalue scaling.
    The base provider should have outputscale=1 and compute K_X (the unscaled kernel).
    
    Column layout: T groups of (1 + P) columns each.
    Columns [t*(1+P), (t+1)*(1+P)) belong to sub-problem t.
    
    Gradient parameters exposed:
        0..K-1: kernel hyperparameters (lengthscale, etc.)
                gradient_matvec returns s_t * dK_X/d(theta) @ v (per-column scaling)
        K:      scale gradient
                gradient_matvec returns K_X @ v (no scaling, same for all columns)
    
    The training loop post-processes the scale gradient to get:
        dNLL/d(lambda_t) = dNLL_t/d(s_t) * outputscale
        dNLL/d(outputscale) = sum_t dNLL_t/d(s_t) * lambda_t
    
    Attributes:
        base_provider: Underlying single-output MatvecProvider (outputscale=1)
        num_tasks: Number of tasks T
        cols_per_task: Columns per task (1 + num_probes)
        effective_scales: T values: s_t = outputscale * lambda_t (on host)
        effective_scales_device: GPU copy of effective_scales
        noise: Observation noise variance
        outputscale: Global output scale
        eigenvalues: T eigenvalues lambda_t (on host)
        temp_buffer: Temporary buffer for intermediate results
    """
    var base_provider: T
    var num_tasks: Int
    var cols_per_task: Int
    var effective_scales: HostBuffer[float_dtype]
    var effective_scales_device: DeviceBuffer[float_dtype]
    var noise: Float32
    var outputscale: Float32
    var eigenvalues: HostBuffer[float_dtype]
    var temp_buffer: DeviceBuffer[float_dtype]  # For K_X @ v intermediate result
    var ctx: DeviceContext
    
    fn __init__(
        out self,
        owned base_provider: T,
        ctx: DeviceContext,
        num_tasks: Int,
        cols_per_task: Int,
        noise: Float32,
        outputscale: Float32,
        eigenvalues: HostBuffer[float_dtype],
    ) raises:
        """Create KroneckerBatchedProvider.
        
        Args:
            base_provider: Underlying MatvecProvider for K_X (outputscale should be 1)
            ctx: GPU device context
            num_tasks: Number of tasks T
            cols_per_task: Columns per task (1 + num_probes)
            noise: Observation noise variance
            outputscale: Global output scale
            eigenvalues: T eigenvalues lambda_t from TaskCovariance
        """
        self.base_provider = base_provider^
        self.ctx = ctx
        self.num_tasks = num_tasks
        self.cols_per_task = cols_per_task
        self.noise = noise
        self.outputscale = outputscale
        
        # Allocate eigenvalue storage
        self.eigenvalues = HostBuffer[float_dtype](ctx, num_tasks)
        self.effective_scales = HostBuffer[float_dtype](ctx, num_tasks)
        self.effective_scales_device = ctx.enqueue_create_buffer[float_dtype](num_tasks)
        
        # Copy eigenvalues and compute effective scales
        for t in range(num_tasks):
            self.eigenvalues.unsafe_ptr()[t] = eigenvalues.unsafe_ptr()[t]
            self.effective_scales.unsafe_ptr()[t] = outputscale * eigenvalues.unsafe_ptr()[t]
        
        # Copy effective scales to device
        self.effective_scales_device.enqueue_copy_from(self.effective_scales)
        
        # Allocate temp buffer for intermediate K_X @ v result
        var n = self.base_provider.get_n()
        var total_cols = num_tasks * cols_per_task
        self.temp_buffer = ctx.enqueue_create_buffer[float_dtype](n * total_cols)
        
        ctx.synchronize()
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute K_t @ v for each sub-problem t.
        
        Implementation:
        1. K_X @ V (single GEMM or matrix-free matvec, all columns at once)
           Note: base_provider has noise=0 internally, we add noise here
        2. Per-column scaling + noise: out[i,col] = s_{task(col)} * (K_X @ V)[i,col] + sigma^2 * v[i,col]
           where task(col) = col // cols_per_task
        
        Args:
            out_ptr: Output buffer [n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of columns (should be num_tasks * cols_per_task)
        """
        var n = self.base_provider.get_n()
        var ctx = self.base_provider.get_ctx()
        
        # Step 1: Compute K_X @ V (without noise, base provider should have noise=0)
        # We use the temp buffer to store the unscaled result
        self.base_provider.forward_matvec(self.temp_buffer.unsafe_ptr(), v_ptr, num_cols)
        
        # Step 2: Apply per-column scaling and add noise
        var total_elements = n * num_cols
        var block_size = 256
        var num_blocks = (total_elements + block_size - 1) // block_size
        
        ctx.enqueue_function[kernel_scale_columns_by_task](
            out_ptr,
            self.temp_buffer.unsafe_ptr(),
            v_ptr,
            self.effective_scales_device.unsafe_ptr(),
            self.noise,
            n,
            self.cols_per_task,
            self.num_tasks,
            grid_dim=(num_blocks,),
            block_dim=(block_size,),
        )
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute gradient matvec for parameter param_index.
        
        For param_index < num_base_kernel_params:
            Compute s_t * dK_X/d(theta) @ v for each sub-problem t.
            Implementation:
              1. base_provider.gradient_matvec(out, v, num_cols, param_index)
                 This gives dK_X/d(theta) @ v (unscaled)
              2. Scale each column group by s_t:
                 out[i,col] *= s_{task(col)}
            
        For param_index == num_base_kernel_params (scale gradient):
            Compute K_X @ v (unscaled, no per-column scaling, no noise).
            This gives dK_t/d(s_t) @ v = K_X @ v for all columns.
        
        Args:
            out_ptr: Output buffer [n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of columns
            param_index: Which parameter (0 to num_gradient_params()-1)
            sync: Whether to synchronize after kernel launch
        """
        var n = self.base_provider.get_n()
        var ctx = self.base_provider.get_ctx()
        var num_base_params = 2  # Isotropic: lengthscale, outputscale
        
        if param_index < num_base_params:
            # Kernel parameter gradient (lengthscale or outputscale)
            # Step 1: Get unscaled gradient matvec from base provider
            self.base_provider.gradient_matvec(out_ptr, v_ptr, num_cols, param_index, False)
            
            # Step 2: Scale each column group by s_t
            var total_elements = n * num_cols
            var block_size = 256
            var num_blocks = (total_elements + block_size - 1) // block_size
            
            ctx.enqueue_function[kernel_scale_columns_by_task_no_noise](
                out_ptr,
                self.effective_scales_device.unsafe_ptr(),
                n,
                self.cols_per_task,
                self.num_tasks,
                grid_dim=(num_blocks,),
                block_dim=(block_size,),
            )
        else:
            # Scale gradient (param_index == num_base_params)
            # Return K_X @ v (unscaled, no noise)
            # The base provider's forward_matvec includes noise, so we use gradient_matvec
            # with outputscale param (index 1) which gives K_X @ v / outputscale
            # But since base outputscale = 1, this is just K_X @ v
            # Actually, we need to compute K_X @ v directly without noise
            # Use the temp buffer trick: forward_matvec gives K_X @ v + noise * v
            # So K_X @ v = forward_matvec(v) - noise * v
            # But that's inefficient. Better: use gradient_matvec with outputscale
            # which gives dK/d(outputscale) @ v = K_X @ v / outputscale = K_X @ v (since outputscale=1)
            self.base_provider.gradient_matvec(out_ptr, v_ptr, num_cols, 1, False)  # outputscale gradient
        
        if sync:
            ctx.synchronize()
    
    fn num_gradient_params(self) -> Int:
        """num_base_kernel_params + 1 (the +1 is the scale gradient).
        
        For isotropic kernels: 2 (lengthscale, outputscale) + 1 (scale) = 3
        """
        return 2 + 1  # lengthscale, outputscale, scale
    
    fn get_n(self) -> Int:
        """Return number of data points."""
        return self.base_provider.get_n()
    
    fn get_ctx(self) -> DeviceContext:
        """Return GPU device context."""
        return self.ctx
    
    fn get_noise(self) -> Float32:
        """Return noise variance sigma^2."""
        return self.noise
    
    fn get_diagonal_value(self) -> Float32:
        """Return the diagonal value of K (without noise).
        
        For Kronecker provider, this is the max effective scale (for preconditioner).
        """
        var max_scale = self.effective_scales.unsafe_ptr()[0]
        for t in range(1, self.num_tasks):
            if self.effective_scales.unsafe_ptr()[t] > max_scale:
                max_scale = self.effective_scales.unsafe_ptr()[t]
        return max_scale
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract the full diagonal of K (without noise) to device buffer.
        
        For stationary kernels, diagonal is constant = base outputscale = 1.
        The actual scaling is handled by the preconditioner set.
        """
        var n = self.base_provider.get_n()
        var ctx = self.base_provider.get_ctx()
        var block_size = 256
        var num_blocks = (n + block_size - 1) // block_size
        
        ctx.enqueue_function[kernel_fill_diagonal_scaled](
            diag_ptr,
            Float32(1.0),  # base outputscale
            self.effective_scales_device.unsafe_ptr(),
            n,
            self.num_tasks,
            grid_dim=(num_blocks,),
            block_dim=(block_size,),
        )
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data on device."""
        return self.base_provider.get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """Kronecker provider doesn't support fused gradients."""
        return False
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Not supported for Kronecker provider."""
        raise Error("fused_gradient_matvec not supported for KroneckerBatchedProvider")
    
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
    
    fn update_scales(
        mut self,
        outputscale: Float32,
        eigenvalues: HostBuffer[float_dtype],
    ) raises:
        """Update effective scales when outputscale or eigenvalues change.
        
        Called at the start of each training iteration after eigendecomposition.
        
        Args:
            outputscale: New global output scale
            eigenvalues: New eigenvalues from TaskCovariance
        """
        self.outputscale = outputscale
        for t in range(self.num_tasks):
            self.eigenvalues.unsafe_ptr()[t] = eigenvalues.unsafe_ptr()[t]
            self.effective_scales.unsafe_ptr()[t] = outputscale * eigenvalues.unsafe_ptr()[t]
        
        # Copy to GPU
        self.effective_scales_device.enqueue_copy_from(self.effective_scales)
        self.ctx.synchronize()
    
    fn update_noise(mut self, noise: Float32):
        """Update noise variance.
        
        Args:
            noise: New noise variance
        """
        self.noise = noise
    
    fn get_effective_scale(self, task_index: Int) -> Float32:
        """Get effective scale s_t for a specific task.
        
        Args:
            task_index: Task index (0 to num_tasks-1)
            
        Returns:
            s_t = outputscale * lambda_t
        """
        return self.effective_scales.unsafe_ptr()[task_index]
    
    fn get_eigenvalue(self, task_index: Int) -> Float32:
        """Get eigenvalue lambda_t for a specific task.
        
        Args:
            task_index: Task index (0 to num_tasks-1)
            
        Returns:
            lambda_t
        """
        return self.eigenvalues.unsafe_ptr()[task_index]
