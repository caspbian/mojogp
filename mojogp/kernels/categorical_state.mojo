"""Categorical correlation state management.

Manages the per-variable correlation matrices, their GPU buffers,
and parameter-to-variable mapping for mixed-categorical GP training.

The state holds:
- Specification of categorical variables (level counts, kernel types)
- Pre-computed correlation matrices (CPU-side, uploaded to GPU)
- Flattened GPU buffers for correlation matrices and categorical indices
- Parameter offset mapping for gradient computation
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer
from .constants import (
    float_dtype,
    CAT_KERNEL_GD,
    CAT_KERNEL_CR,
    CAT_KERNEL_EHH,
    CAT_KERNEL_HH,
    CAT_KERNEL_FE,
    MAX_CAT_VARS,
    MAX_CAT_LEVELS,
    MAX_CAT_CORR_SIZE,
)
from .categorical_kernel import (
    compute_correlation_matrix,
    compute_correlation_gradient,
    num_params_for_variant,
)


struct CategoricalCorrelationState(Movable):
    """Manages categorical correlation matrices and GPU buffers.
    
    This struct is the central state for mixed-categorical kernels. It:
    1. Stores the specification (num vars, levels per var, kernel type per var)
    2. Computes correlation matrices from raw parameters (CPU-side)
    3. Uploads flattened correlation matrices to GPU
    4. Provides gradient correlation matrices for training
    
    GPU Memory Layout:
        c_device:         Int32[n * num_cat_vars]     - categorical indices
        corr_flat_device: Float32[sum(L_i^2)]         - flattened correlation matrices
        corr_offsets:     Int32[num_cat_vars]          - offsets into corr_flat
        corr_levels:      Int32[num_cat_vars]          - L_i per variable
    
    The correlation matrices are small (typically < 4KB total) and are
    recomputed on CPU then uploaded to GPU whenever categorical hyperparameters
    change during training.
    """
    var ctx: DeviceContext
    var num_cat_vars: Int
    var num_data: Int
    
    # Per-variable specification
    var levels: List[Int]           # L_i for each variable
    var kernel_types: List[Int]     # CAT_KERNEL_* for each variable
    var param_offsets: List[Int]    # Offset into flat param vector for each variable
    var total_cat_params: Int       # Total number of categorical parameters
    
    # Flattened correlation matrix storage
    var corr_offsets: List[Int]     # Offset into corr_flat for each variable's L_i*L_i block
    var total_corr_size: Int        # sum(L_i^2)
    
    # GPU buffers
    var c_device: DeviceBuffer[DType.int32]       # Categorical indices [n * num_cat_vars]
    var corr_flat_device: DeviceBuffer[float_dtype]  # Flattened correlation matrices
    var offsets_device: DeviceBuffer[DType.int32]     # Offsets into corr_flat
    var levels_device: DeviceBuffer[DType.int32]      # L_i per variable
    
    # CPU-side correlation matrices (for gradient computation)
    var corr_flat_host: HostBuffer[float_dtype]
    
    # Workspace for hypersphere decomposition (CPU-side)
    var work_host: HostBuffer[float_dtype]
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        var levels: List[Int],
        var kernel_types: List[Int],
        n: Int,
    ) raises:
        """Initialize categorical state.
        
        Args:
            ctx: GPU device context.
            levels: List of L_i (number of levels) for each categorical variable.
            kernel_types: List of CAT_KERNEL_* type for each variable.
            n: Number of data points.
        """
        self.ctx = ctx
        self.num_cat_vars = len(levels)
        self.num_data = n
        self.levels = levels^
        self.kernel_types = kernel_types^
        
        # Compute parameter offsets and total param count
        self.param_offsets = List[Int]()
        var offset = 0
        for v in range(self.num_cat_vars):
            self.param_offsets.append(offset)
            offset += num_params_for_variant(self.levels[v], self.kernel_types[v])
        self.total_cat_params = offset
        
        # Compute correlation matrix offsets and total size
        self.corr_offsets = List[Int]()
        var corr_offset = 0
        for v in range(self.num_cat_vars):
            self.corr_offsets.append(corr_offset)
            corr_offset += self.levels[v] * self.levels[v]
        self.total_corr_size = corr_offset
        
        # Allocate GPU buffers
        self.c_device = ctx.enqueue_create_buffer[DType.int32](n * self.num_cat_vars)
        self.corr_flat_device = ctx.enqueue_create_buffer[float_dtype](self.total_corr_size)
        self.offsets_device = ctx.enqueue_create_buffer[DType.int32](self.num_cat_vars)
        self.levels_device = ctx.enqueue_create_buffer[DType.int32](self.num_cat_vars)
        
        # Allocate CPU-side buffers
        self.corr_flat_host = HostBuffer[float_dtype](ctx, self.total_corr_size)
        
        # Workspace: need at least 2 * max(L_i)^2 for EHH/HH/FE gradient computation
        var max_L = 0
        for v in range(self.num_cat_vars):
            if self.levels[v] > max_L:
                max_L = self.levels[v]
        self.work_host = HostBuffer[float_dtype](ctx, 2 * max_L * max_L)
        
        # Upload offsets and levels (these don't change during training)
        var offsets_host = HostBuffer[DType.int32](ctx, self.num_cat_vars)
        var levels_host = HostBuffer[DType.int32](ctx, self.num_cat_vars)
        for v in range(self.num_cat_vars):
            offsets_host[v] = Int32(self.corr_offsets[v])
            levels_host[v] = Int32(self.levels[v])
        ctx.enqueue_copy(self.offsets_device, offsets_host)
        ctx.enqueue_copy(self.levels_device, levels_host)
        ctx.synchronize()
        
        # Keep temp buffers alive until copy completes
        _ = offsets_host
        _ = levels_host
    
    fn upload_categorical_data(
        mut self,
        c_host: HostBuffer[DType.int32],
    ) raises:
        """Upload categorical index data to GPU.
        
        Args:
            c_host: Int32 buffer of shape [n * num_cat_vars], row-major.
                    c_host[i * num_cat_vars + v] = level index for point i, variable v.
        """
        self.ctx.enqueue_copy(self.c_device, c_host)
        self.ctx.synchronize()
    
    fn update_correlation_matrices(
        mut self,
        cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Recompute all correlation matrices from current parameters and upload to GPU.
        
        This is called whenever categorical hyperparameters change during training.
        The computation is done on CPU (correlation matrices are small) and then
        uploaded to GPU.
        
        Args:
            cat_params_ptr: Flat array of all categorical parameters.
                           Layout: [params_var_0, params_var_1, ..., params_var_{l-1}]
        """
        var corr_ptr = self.corr_flat_host.unsafe_ptr()
        var work_ptr = self.work_host.unsafe_ptr()
        
        for v in range(self.num_cat_vars):
            var L = self.levels[v]
            var kt = self.kernel_types[v]
            var p_offset = self.param_offsets[v]
            var c_offset = self.corr_offsets[v]
            
            compute_correlation_matrix(
                corr_ptr + c_offset,
                cat_params_ptr + p_offset,
                L,
                kt,
                work_ptr,
            )
        
        # Upload to GPU
        self.ctx.enqueue_copy(self.corr_flat_device, self.corr_flat_host)
        self.ctx.synchronize()
    
    fn compute_gradient_correlation(
        self,
        dR_host: HostBuffer[float_dtype],
        cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cat_param_index: Int,
    ) raises:
        """Compute gradient correlation matrix for a specific categorical parameter.
        
        Returns the full flattened gradient correlation matrices where only the
        variable affected by cat_param_index has non-zero gradient, and all other
        variables have their original correlation values (for the product rule).
        
        For the product rule: d(prod_v R_v)/d(theta_k) = dR_v/d(theta_k) * prod_{w!=v} R_w
        
        This function computes dR_v/d(theta_k) for the affected variable v,
        and the caller handles the product with other variables' R_w values.
        
        Args:
            dR_host: Output buffer of size total_corr_size.
            cat_params_ptr: Flat array of all categorical parameters.
            cat_param_index: Index into the flat categorical parameter vector.
        """
        var dR_ptr = dR_host.unsafe_ptr()
        var corr_ptr = self.corr_flat_host.unsafe_ptr()
        var work_ptr = self.work_host.unsafe_ptr()
        
        # Find which variable this parameter belongs to
        var target_var = -1
        var local_param_index = cat_param_index
        for v in range(self.num_cat_vars):
            var np = num_params_for_variant(self.levels[v], self.kernel_types[v])
            if local_param_index < np:
                target_var = v
                break
            local_param_index -= np
        
        if target_var < 0:
            return  # Invalid parameter index
        
        # For non-target variables, copy the correlation matrix as-is
        # For the target variable, compute the gradient
        for v in range(self.num_cat_vars):
            var L = self.levels[v]
            var c_offset = self.corr_offsets[v]
            
            if v == target_var:
                var p_offset = self.param_offsets[v]
                compute_correlation_gradient(
                    dR_ptr + c_offset,
                    cat_params_ptr + p_offset,
                    corr_ptr + c_offset,
                    L,
                    self.kernel_types[v],
                    local_param_index,
                    work_ptr,
                )
            else:
                # Copy existing correlation matrix
                for idx in range(L * L):
                    dR_ptr[c_offset + idx] = corr_ptr[c_offset + idx]
    
    fn get_c_device_ptr(self) -> UnsafePointer[Int32, MutAnyOrigin]:
        """Get raw pointer to categorical indices on GPU."""
        return self.c_device.unsafe_ptr()
    
    fn get_corr_flat_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Get raw pointer to flattened correlation matrices on GPU."""
        return self.corr_flat_device.unsafe_ptr()
    
    fn get_offsets_device_ptr(self) -> UnsafePointer[Int32, MutAnyOrigin]:
        """Get raw pointer to correlation offsets on GPU."""
        return self.offsets_device.unsafe_ptr()
    
    fn get_levels_device_ptr(self) -> UnsafePointer[Int32, MutAnyOrigin]:
        """Get raw pointer to level counts on GPU."""
        return self.levels_device.unsafe_ptr()
    
    fn get_total_cat_params(self) -> Int:
        """Get total number of categorical parameters across all variables."""
        return self.total_cat_params
    
    fn get_num_cat_vars(self) -> Int:
        """Get number of categorical variables."""
        return self.num_cat_vars
    
    fn get_param_offset_for_var(self, var_index: Int) -> Int:
        """Get parameter offset for a specific variable."""
        return self.param_offsets[var_index]
    
    fn get_num_params_for_var(self, var_index: Int) -> Int:
        """Get number of parameters for a specific variable."""
        return num_params_for_variant(self.levels[var_index], self.kernel_types[var_index])

    fn get_corr_stride(self) -> Int:
        """Get the stride per parameter in the packed gradient buffer.
        
        This is the total flat correlation size: sum of L_v * L_v for all
        categorical variables. Each parameter's gradient correlation block
        is exactly this many floats.
        
        Returns:
            total_corr_size = sum(L_v^2) for v in 0..num_cat_vars.
        """
        return self.total_corr_size

    fn compute_all_gradient_correlations(
        self,
        all_grad_corr_host: HostBuffer[float_dtype],
        cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Compute gradient correlations for ALL categorical parameters at once.
        
        Packs results into a single contiguous buffer so the caller can perform
        one H2D copy instead of one per parameter.
        
        Output layout:
            all_grad_corr_host[param_idx * corr_stride + flat_offset]
        
        where:
            corr_stride = total_corr_size = sum(L_v^2) for all variables
            param_idx   = 0 .. total_cat_params - 1
            flat_offset = offset within the per-variable correlation blocks
        
        For each categorical parameter k (global index):
          - Find which variable v owns parameter k and compute local_param_index
          - For non-target variables w != v: copy R_w into the output block
          - For target variable v: compute dR_v/d(theta_k)
        
        This reuses the same logic as compute_gradient_correlation() but writes
        each result into a different slice of the contiguous output buffer.
        
        Args:
            all_grad_corr_host: Output buffer of size total_cat_params * total_corr_size.
            cat_params_ptr: Flat array of all categorical parameters.
        """
        var corr_stride = self.total_corr_size
        var out_ptr = all_grad_corr_host.unsafe_ptr()
        var corr_ptr = self.corr_flat_host.unsafe_ptr()
        var work_ptr = self.work_host.unsafe_ptr()
        
        # Iterate over each categorical parameter (global index k)
        var global_k = 0
        for v in range(self.num_cat_vars):
            var np_v = num_params_for_variant(self.levels[v], self.kernel_types[v])
            
            for local_p in range(np_v):
                # Destination slice for this parameter: out_ptr + global_k * corr_stride
                var dest = out_ptr + global_k * corr_stride
                
                # For each variable, either copy R_w or compute dR_v/d(theta_k)
                for w in range(self.num_cat_vars):
                    var L_w = self.levels[w]
                    var c_off_w = self.corr_offsets[w]
                    
                    if w == v:
                        # Target variable: compute gradient
                        var p_offset = self.param_offsets[v]
                        compute_correlation_gradient(
                            dest + c_off_w,
                            cat_params_ptr + p_offset,
                            corr_ptr + c_off_w,
                            L_w,
                            self.kernel_types[v],
                            local_p,
                            work_ptr,
                        )
                    else:
                        # Non-target variable: copy existing correlation matrix
                        for idx in range(L_w * L_w):
                            (dest + c_off_w)[idx] = (corr_ptr + c_off_w)[idx]
                
                global_k += 1
