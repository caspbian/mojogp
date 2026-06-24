"""Providers for mixed composite + categorical kernels.

Two providers:
1. MixedCompositeProvider[DIM, K] — matrix-free, O(n) memory
2. MixedMaterializedCompositeProvider[DIM, K] — materializes full K_mixed, GEMM-based

Both implement GradientProvider directly (no separate adapter needed).

The parameter layout for gradient_matvec is:
  [0 .. K.num_params()-1]  = continuous composite kernel params
  [K.num_params() .. K.num_params()+total_cat_params-1] = categorical params
  (noise gradient is appended by bbmm_unified, not part of num_gradient_params)
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer

from .composable_kernel import ComposableKernel
from .categorical_state import CategoricalCorrelationState
from .gradient_provider import GradientProvider
from .gemm_matvec import gemm_matvec, add_noise_diagonal
from .mixed_composite_matvec import (
    composite_mixed_forward_matvec_8x,
    composite_mixed_forward_matvec_multicol,
    _MULTICOL_NCOLS,
    _MULTICOL_NCOLS_6,
    composite_mixed_gradient_cont_matvec_4x,
    composite_mixed_gradient_cat_matvec_4x,
    composite_mixed_materialize,
    composite_mixed_cross_matvec_8x,
    composite_mixed_cross_covariance_fused,
    composite_mixed_extract_diagonal,
)

alias float_dtype = DType.float32


# =============================================================================
# Matrix-Free Provider
# =============================================================================

struct MixedCompositeProvider[DIM: Int, K: ComposableKernel, IS_PRODUCT: Bool = True](GradientProvider, Movable):
    """Matrix-free provider for mixed composite + categorical kernels.
    
    Computes (K_cont OP K_cat + noise*I) @ v on-the-fly without materializing
    the full kernel matrix. Memory: O(n).
    
    IS_PRODUCT=True: k_total = K_cont * K_cat (product composition)
    IS_PRODUCT=False: k_total = K_cont + K_cat (sum composition)
    
    The continuous kernel K is parameterized at compile time via ComposableKernel.
    The categorical correlation is computed at runtime from CategoricalCorrelationState.
    """
    var ctx: DeviceContext
    var x_ptr: UnsafePointer[Float32, MutAnyOrigin]      # device ptr to continuous X [n, DIM]
    var params_ptr: UnsafePointer[Float32, MutAnyOrigin]  # device ptr to composite params [K.num_params()]
    var n: Int
    var noise: Float32
    
    var _x_device: DeviceBuffer[float_dtype]              # owns GPU memory for X
    var _params_device: DeviceBuffer[float_dtype]         # owns GPU memory for params
    
    var cat_state: CategoricalCorrelationState             # owns categorical state
    var grad_corr_host: HostBuffer[float_dtype]            # CPU workspace for gradient correlations
    var grad_corr_device: DeviceBuffer[float_dtype]        # GPU buffer for gradient correlations
    var all_grad_corr_host: HostBuffer[float_dtype]        # batch gradient corr for fused path
    var all_grad_corr_device: DeviceBuffer[float_dtype]    # batch gradient corr GPU buffer
    var cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin]  # external, caller must keep alive
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],     # [n, DIM] on host
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin], # [K.num_params()] on host
        n: Int,
        noise: Float32,
        var cat_state: CategoricalCorrelationState,
    ) raises:
        """Create a matrix-free mixed composite provider.
        
        Args:
            ctx: GPU device context
            x_host_ptr: Continuous training data [n, DIM] on host (row-major)
            params_host_ptr: Composite kernel parameters [K.num_params()] on host
            n: Number of training points
            noise: Noise variance
            cat_state: Categorical correlation state (moved in)
        """
        self.ctx = ctx
        self.n = n
        self.noise = noise
        
        # Copy X to device
        var x_host = ctx.enqueue_create_host_buffer[float_dtype](n * DIM)
        for i in range(n * DIM):
            x_host[i] = x_host_ptr[i]
        self._x_device = ctx.enqueue_create_buffer[float_dtype](n * DIM)
        ctx.enqueue_copy(dst_buf=self._x_device, src_buf=x_host)
        self.x_ptr = self._x_device.unsafe_ptr()
        
        # Copy params to device
        var num_params = K.num_params()
        var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_params)
        for i in range(num_params):
            params_host[i] = params_host_ptr[i]
        self._params_device = ctx.enqueue_create_buffer[float_dtype](num_params)
        ctx.enqueue_copy(dst_buf=self._params_device, src_buf=params_host)
        self.params_ptr = self._params_device.unsafe_ptr()
        
        ctx.synchronize()
        
        # Categorical state
        self.cat_state = cat_state^
        
        # Gradient correlation workspace (single-param path)
        var total_corr_size = self.cat_state.total_corr_size
        self.grad_corr_host = ctx.enqueue_create_host_buffer[float_dtype](total_corr_size)
        self.grad_corr_device = ctx.enqueue_create_buffer[float_dtype](total_corr_size)
        
        # Batch gradient correlation buffers (all cat params at once, fused path)
        var total_cat = self.cat_state.get_total_cat_params()
        var batch_size = max(total_cat * total_corr_size, 1)
        self.all_grad_corr_host = HostBuffer[float_dtype](ctx, batch_size)
        self.all_grad_corr_device = ctx.enqueue_create_buffer[float_dtype](batch_size)
        
        self.cat_params_ptr = UnsafePointer[Float32, MutAnyOrigin]()
    
    # =========================================================================
    # GradientProvider trait implementation
    # =========================================================================
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (K_cont * K_cat + noise*I) @ v."""
        alias BLOCK_SIZE = 256
        var num_blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        if num_cols == _MULTICOL_NCOLS:
            self.ctx.enqueue_function[
                composite_mixed_forward_matvec_multicol[DIM, _MULTICOL_NCOLS, K, IS_PRODUCT]
            ](
                out_ptr, self.x_ptr, v_ptr, self.params_ptr,
                self.cat_state.get_c_device_ptr(),
                self.n, self.cat_state.get_num_cat_vars(),
                self.cat_state.get_corr_flat_device_ptr(),
                self.cat_state.get_offsets_device_ptr(),
                self.cat_state.get_levels_device_ptr(),
                self.noise,
                grid_dim=num_blocks, block_dim=BLOCK_SIZE,
                shared_mem_bytes=BLOCK_SIZE * (DIM + _MULTICOL_NCOLS) * 4,
            )
        elif num_cols == _MULTICOL_NCOLS_6:
            self.ctx.enqueue_function[
                composite_mixed_forward_matvec_multicol[DIM, _MULTICOL_NCOLS_6, K, IS_PRODUCT]
            ](
                out_ptr, self.x_ptr, v_ptr, self.params_ptr,
                self.cat_state.get_c_device_ptr(),
                self.n, self.cat_state.get_num_cat_vars(),
                self.cat_state.get_corr_flat_device_ptr(),
                self.cat_state.get_offsets_device_ptr(),
                self.cat_state.get_levels_device_ptr(),
                self.noise,
                grid_dim=num_blocks, block_dim=BLOCK_SIZE,
                shared_mem_bytes=BLOCK_SIZE * (DIM + _MULTICOL_NCOLS_6) * 4,
            )
        else:
            self.ctx.enqueue_function[composite_mixed_forward_matvec_8x[DIM, K, IS_PRODUCT]](
                out_ptr, self.x_ptr, v_ptr, self.params_ptr,
                self.cat_state.get_c_device_ptr(),
                self.n, num_cols, self.cat_state.get_num_cat_vars(),
                self.cat_state.get_corr_flat_device_ptr(),
                self.cat_state.get_offsets_device_ptr(),
                self.cat_state.get_levels_device_ptr(),
                self.noise,
                grid_dim=num_blocks, block_dim=BLOCK_SIZE,
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
        """Compute dK/dtheta_i @ v for parameter gradient.
        
        param_index routing:
          [0 .. K.num_params()-1] -> continuous composite gradient
          [K.num_params() .. ] -> categorical gradient
        """
        var num_cont_params = K.num_params()
        
        if param_index < num_cont_params:
            # Continuous composite kernel gradient
            self._gradient_matvec_continuous(out_ptr, v_ptr, num_cols, param_index)
        else:
            # Categorical gradient
            var cat_param_idx = param_index - num_cont_params
            self._gradient_matvec_categorical(out_ptr, v_ptr, num_cols, cat_param_idx)
        
        if sync:
            self.ctx.synchronize()
    
    fn num_gradient_params(self) -> Int:
        """Total gradient params = composite params + categorical params."""
        return K.num_params() + self.cat_state.get_total_cat_params()
    
    fn get_n(self) -> Int:
        return self.n
    
    fn get_ctx(self) -> DeviceContext:
        return self.ctx
    
    fn get_noise(self) -> Float32:
        return self.noise
    
    fn get_diagonal_value(self) -> Float32:
        """Default diagonal value. Real diagonal comes from extract_diagonal."""
        return Float32(1.0)
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract diagonal of K_mixed (without noise).
        
        Since k_cat(c_i, c_i) = 1.0 always, this is just K.evaluate(x_i, x_i, params).
        For sum mode: K.evaluate(x_i, x_i, params) + 1.0 (adding the categorical self-correlation).
        """
        alias BLOCK_SIZE = 256
        var num_blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_extract_diagonal[DIM, K, IS_PRODUCT]](
            diag_ptr, self.x_ptr, self.params_ptr, self.n,
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.x_ptr
    
    fn supports_fused_gradient(self) -> Bool:
        """Fused gradient is supported when we have categorical params within limit."""
        alias MAX_FUSED_CAT_PARAMS = 25
        var total_cat = self.cat_state.get_total_cat_params()
        return total_cat > 0 and total_cat <= MAX_FUSED_CAT_PARAMS
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs (continuous + categorical) in fused fashion.
        
        Output layout: out_ptr[p * n * num_cols + col * n + row] for parameter p.
        Total output size: num_gradient_params() * n * num_cols.
        
        Strategy:
          Part 1: Continuous params — per-param sequential gradient matvec
                  (only 2-4 params, fast enough, no fusion needed)
          Part 2: Categorical params — batch compute all gradient correlations
                  on CPU, single H2D upload, then per-param gradient matvec
                  using batch buffer (eliminates per-param CPU + H2D overhead)
        """
        var n = self.n
        var total_params = self.num_gradient_params()
        var num_cont = K.num_params()
        var total_cat = self.cat_state.get_total_cat_params()
        
        # Part 1: Continuous gradient params — per-param path
        for p in range(num_cont):
            var out_p = out_ptr.offset(p * n * num_cols)
            self._gradient_matvec_continuous(out_p, v_ptr, num_cols, p)
        
        # Part 2: Categorical gradient params — batch correlation + per-param matvec
        if total_cat > 0:
            # Batch compute ALL gradient correlations on CPU
            self.cat_state.compute_all_gradient_correlations(
                self.all_grad_corr_host,
                self.cat_params_ptr,
            )
            
            # Single H2D upload for all gradient correlations
            self.ctx.enqueue_copy(self.all_grad_corr_device, self.all_grad_corr_host)
            self.ctx.synchronize()
            
            # Per-param gradient matvec using the batch buffer
            var corr_stride = self.cat_state.get_corr_stride()
            alias BLOCK_SIZE = 256
            var num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            for cp in range(total_cat):
                var out_p = out_ptr.offset((num_cont + cp) * n * num_cols)
                
                # Use the pre-computed gradient correlation slice for this param
                var grad_corr_slice_ptr = self.all_grad_corr_device.unsafe_ptr().offset(
                    cp * corr_stride
                )
                
                self.ctx.enqueue_function[
                    composite_mixed_gradient_cat_matvec_4x[DIM, K, IS_PRODUCT]
                ](
                    out_p, self.x_ptr, v_ptr, self.params_ptr,
                    self.cat_state.get_c_device_ptr(),
                    n, num_cols, self.cat_state.get_num_cat_vars(),
                    grad_corr_slice_ptr,
                    self.cat_state.get_offsets_device_ptr(),
                    self.cat_state.get_levels_device_ptr(),
                    grid_dim=num_blocks, block_dim=BLOCK_SIZE,
                )
            self.ctx.synchronize()
    
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
    # Internal gradient methods
    # =========================================================================
    
    fn _gradient_matvec_continuous(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
    ) raises:
        """Compute (dK_cont/dtheta_p * K_cat) @ v for a continuous parameter."""
        alias BLOCK_SIZE = 256
        var num_blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_gradient_cont_matvec_4x[DIM, K, IS_PRODUCT]](
            out_ptr, self.x_ptr, v_ptr, self.params_ptr,
            self.cat_state.get_c_device_ptr(),
            self.n, num_cols, self.cat_state.get_num_cat_vars(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            param_index,
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    fn _gradient_matvec_categorical(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        cat_param_index: Int,
    ) raises:
        """Compute (K_cont * dK_cat/dtheta) @ v for a categorical parameter.
        
        Steps:
        1. CPU: compute gradient correlation matrices via cat_state
        2. Upload to GPU
        3. Launch gradient kernel
        """
        # Compute gradient correlations on CPU
        self.cat_state.compute_gradient_correlation(
            self.grad_corr_host, self.cat_params_ptr, cat_param_index
        )
        
        # Upload to GPU
        self.ctx.enqueue_copy(dst_buf=self.grad_corr_device, src_buf=self.grad_corr_host)
        self.ctx.synchronize()
        
        # Launch gradient kernel
        alias BLOCK_SIZE = 256
        var num_blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_gradient_cat_matvec_4x[DIM, K, IS_PRODUCT]](
            out_ptr, self.x_ptr, v_ptr, self.params_ptr,
            self.cat_state.get_c_device_ptr(),
            self.n, num_cols, self.cat_state.get_num_cat_vars(),
            self.grad_corr_device.unsafe_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    # =========================================================================
    # Cross-kernel methods (for prediction)
    # =========================================================================
    
    fn cross_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        c_test_ptr: UnsafePointer[Int32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_test: Int,
    ) raises:
        """Compute K_mixed(X_test, X_train) @ v for prediction mean."""
        alias BLOCK_SIZE = 256
        var num_blocks = (n_test + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_cross_matvec_8x[DIM, K, IS_PRODUCT]](
            out_ptr, x_test_ptr, self.x_ptr, v_ptr, self.params_ptr,
            c_test_ptr, self.cat_state.get_c_device_ptr(),
            n_test, self.n, self.cat_state.get_num_cat_vars(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    fn cross_covariance(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        c_test_ptr: UnsafePointer[Int32, MutAnyOrigin],
        n_test: Int,
    ) raises:
        """Compute K_mixed(X_train, X_test) for LOVE variance prediction.
        
        Output: [n_train, n_test] column-major.
        """
        alias BLOCK_SIZE = 16
        var grid_x = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        var grid_y = (n_test + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_cross_covariance_fused[DIM, K, IS_PRODUCT]](
            out_ptr, self.x_ptr, x_test_ptr, self.params_ptr,
            self.cat_state.get_c_device_ptr(), c_test_ptr,
            self.n, n_test, self.cat_state.get_num_cat_vars(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            grid_dim=(grid_x, grid_y), block_dim=(BLOCK_SIZE, BLOCK_SIZE),
        )
        self.ctx.synchronize()


# =============================================================================
# Materialized Provider
# =============================================================================

struct MixedMaterializedCompositeProvider[DIM: Int, K: ComposableKernel, IS_PRODUCT: Bool = True](GradientProvider, Movable):
    """Materialized provider for mixed composite + categorical kernels.
    
    Materializes the full K_mixed = K_cont * K_cat matrix and uses GEMM for
    forward matvec. Memory: O(n^2).
    
    Gradient matvecs for continuous params use matrix-free GPU kernels.
    Gradient matvecs for categorical params use CPU gradient correlation + matrix-free.
    """
    var ctx: DeviceContext
    var x_ptr: UnsafePointer[Float32, MutAnyOrigin]      # device ptr to continuous X [n, DIM]
    var params_ptr: UnsafePointer[Float32, MutAnyOrigin]  # device ptr to composite params
    var n: Int
    var noise: Float32
    
    var _x_device: DeviceBuffer[float_dtype]              # owns GPU memory for X
    var _params_device: DeviceBuffer[float_dtype]         # owns GPU memory for params
    var _K_device: DeviceBuffer[float_dtype]              # [n*n] materialized mixed kernel
    var K_ptr: UnsafePointer[Float32, MutAnyOrigin]
    
    var cat_state: CategoricalCorrelationState
    var grad_corr_host: HostBuffer[float_dtype]
    var grad_corr_device: DeviceBuffer[float_dtype]
    var all_grad_corr_host: HostBuffer[float_dtype]        # batch gradient corr for fused path
    var all_grad_corr_device: DeviceBuffer[float_dtype]    # batch gradient corr GPU buffer
    var cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin]
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        noise: Float32,
        var cat_state: CategoricalCorrelationState,
    ) raises:
        """Create a materialized mixed composite provider.
        
        Immediately materializes the full K_mixed matrix.
        """
        self.ctx = ctx
        self.n = n
        self.noise = noise
        
        # Copy X to device
        var x_host = ctx.enqueue_create_host_buffer[float_dtype](n * DIM)
        for i in range(n * DIM):
            x_host[i] = x_host_ptr[i]
        self._x_device = ctx.enqueue_create_buffer[float_dtype](n * DIM)
        ctx.enqueue_copy(dst_buf=self._x_device, src_buf=x_host)
        self.x_ptr = self._x_device.unsafe_ptr()
        
        # Copy params to device
        var num_params = K.num_params()
        var params_host = ctx.enqueue_create_host_buffer[float_dtype](num_params)
        for i in range(num_params):
            params_host[i] = params_host_ptr[i]
        self._params_device = ctx.enqueue_create_buffer[float_dtype](num_params)
        ctx.enqueue_copy(dst_buf=self._params_device, src_buf=params_host)
        self.params_ptr = self._params_device.unsafe_ptr()
        
        ctx.synchronize()
        
        # Categorical state
        self.cat_state = cat_state^
        
        # Allocate K matrix
        self._K_device = ctx.enqueue_create_buffer[float_dtype](n * n)
        self.K_ptr = self._K_device.unsafe_ptr()
        
        # Gradient correlation workspace (single-param path)
        var total_corr_size = self.cat_state.total_corr_size
        self.grad_corr_host = ctx.enqueue_create_host_buffer[float_dtype](total_corr_size)
        self.grad_corr_device = ctx.enqueue_create_buffer[float_dtype](total_corr_size)
        
        # Batch gradient correlation buffers (all cat params at once, fused path)
        var total_cat = self.cat_state.get_total_cat_params()
        var batch_size = max(total_cat * total_corr_size, 1)
        self.all_grad_corr_host = HostBuffer[float_dtype](ctx, batch_size)
        self.all_grad_corr_device = ctx.enqueue_create_buffer[float_dtype](batch_size)
        
        self.cat_params_ptr = UnsafePointer[Float32, MutAnyOrigin]()
        
        # Materialize K_mixed
        self._materialize()
    
    fn _materialize(self) raises:
        """Materialize the full K_mixed = K_cont * K_cat matrix."""
        alias BLOCK_SIZE = 16
        var grid_x = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        var grid_y = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_materialize[DIM, K, IS_PRODUCT]](
            self.K_ptr, self.x_ptr, self.params_ptr,
            self.cat_state.get_c_device_ptr(),
            self.n, self.cat_state.get_num_cat_vars(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            grid_dim=(grid_x, grid_y), block_dim=(BLOCK_SIZE, BLOCK_SIZE),
        )
        self.ctx.synchronize()
    
    # =========================================================================
    # GradientProvider trait implementation
    # =========================================================================
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (K_mixed + noise*I) @ v using GEMM on materialized matrix."""
        gemm_matvec(self.ctx, out_ptr, self.K_ptr, v_ptr, self.n, num_cols)
        add_noise_diagonal(self.ctx, out_ptr, v_ptr, self.n, num_cols, self.noise)
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute dK/dtheta_i @ v for parameter gradient.
        
        Both continuous and categorical gradients use matrix-free GPU kernels
        (we don't materialize gradient matrices to save memory).
        """
        var num_cont_params = K.num_params()
        
        if param_index < num_cont_params:
            self._gradient_matvec_continuous(out_ptr, v_ptr, num_cols, param_index)
        else:
            var cat_param_idx = param_index - num_cont_params
            self._gradient_matvec_categorical(out_ptr, v_ptr, num_cols, cat_param_idx)
        
        if sync:
            self.ctx.synchronize()
    
    fn num_gradient_params(self) -> Int:
        return K.num_params() + self.cat_state.get_total_cat_params()
    
    fn get_n(self) -> Int:
        return self.n
    
    fn get_ctx(self) -> DeviceContext:
        return self.ctx
    
    fn get_noise(self) -> Float32:
        return self.noise
    
    fn get_diagonal_value(self) -> Float32:
        return Float32(1.0)
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract diagonal from materialized K_mixed."""
        alias BLOCK_SIZE = 256
        var num_blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Since K_mixed is materialized, extract from it directly
        self.ctx.enqueue_function[composite_mixed_extract_diagonal[DIM, K, IS_PRODUCT]](
            diag_ptr, self.x_ptr, self.params_ptr, self.n,
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.x_ptr
    
    fn supports_fused_gradient(self) -> Bool:
        """Fused gradient is supported when we have categorical params within limit."""
        alias MAX_FUSED_CAT_PARAMS = 25
        var total_cat = self.cat_state.get_total_cat_params()
        return total_cat > 0 and total_cat <= MAX_FUSED_CAT_PARAMS
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs (continuous + categorical) in fused fashion.
        
        Same strategy as matrix-free provider:
          Part 1: Continuous params — per-param sequential
          Part 2: Categorical params — batch correlations + per-param matvec
        """
        var n = self.n
        var num_cont = K.num_params()
        var total_cat = self.cat_state.get_total_cat_params()
        
        # Part 1: Continuous gradient params — per-param path
        for p in range(num_cont):
            var out_p = out_ptr.offset(p * n * num_cols)
            self._gradient_matvec_continuous(out_p, v_ptr, num_cols, p)
        
        # Part 2: Categorical gradient params — batch correlation + per-param matvec
        if total_cat > 0:
            self.cat_state.compute_all_gradient_correlations(
                self.all_grad_corr_host,
                self.cat_params_ptr,
            )
            
            self.ctx.enqueue_copy(self.all_grad_corr_device, self.all_grad_corr_host)
            self.ctx.synchronize()
            
            var corr_stride = self.cat_state.get_corr_stride()
            alias BLOCK_SIZE = 256
            var num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            for cp in range(total_cat):
                var out_p = out_ptr.offset((num_cont + cp) * n * num_cols)
                var grad_corr_slice_ptr = self.all_grad_corr_device.unsafe_ptr().offset(
                    cp * corr_stride
                )
                
                self.ctx.enqueue_function[
                    composite_mixed_gradient_cat_matvec_4x[DIM, K, IS_PRODUCT]
                ](
                    out_p, self.x_ptr, v_ptr, self.params_ptr,
                    self.cat_state.get_c_device_ptr(),
                    n, num_cols, self.cat_state.get_num_cat_vars(),
                    grad_corr_slice_ptr,
                    self.cat_state.get_offsets_device_ptr(),
                    self.cat_state.get_levels_device_ptr(),
                    grid_dim=num_blocks, block_dim=BLOCK_SIZE,
                )
            self.ctx.synchronize()
    
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
    # Internal gradient methods (same as matrix-free, using GPU kernels)
    # =========================================================================
    
    fn _gradient_matvec_continuous(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
    ) raises:
        alias BLOCK_SIZE = 256
        var num_blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_gradient_cont_matvec_4x[DIM, K, IS_PRODUCT]](
            out_ptr, self.x_ptr, v_ptr, self.params_ptr,
            self.cat_state.get_c_device_ptr(),
            self.n, num_cols, self.cat_state.get_num_cat_vars(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            param_index,
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    fn _gradient_matvec_categorical(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        cat_param_index: Int,
    ) raises:
        self.cat_state.compute_gradient_correlation(
            self.grad_corr_host, self.cat_params_ptr, cat_param_index
        )
        
        self.ctx.enqueue_copy(dst_buf=self.grad_corr_device, src_buf=self.grad_corr_host)
        self.ctx.synchronize()
        
        alias BLOCK_SIZE = 256
        var num_blocks = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_gradient_cat_matvec_4x[DIM, K, IS_PRODUCT]](
            out_ptr, self.x_ptr, v_ptr, self.params_ptr,
            self.cat_state.get_c_device_ptr(),
            self.n, num_cols, self.cat_state.get_num_cat_vars(),
            self.grad_corr_device.unsafe_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    # =========================================================================
    # Parameter update methods
    # =========================================================================
    
    fn update_params(mut self, params_host_ptr: UnsafePointer[Float32, MutAnyOrigin]) raises:
        """Update composite kernel parameters and re-materialize K_mixed."""
        var num_params = K.num_params()
        var params_host = self.ctx.enqueue_create_host_buffer[float_dtype](num_params)
        for i in range(num_params):
            params_host[i] = params_host_ptr[i]
        self.ctx.enqueue_copy(dst_buf=self._params_device, src_buf=params_host)
        self.ctx.synchronize()
        self.params_ptr = self._params_device.unsafe_ptr()
        
        # Re-materialize
        self._materialize()
    
    fn update_noise(mut self, noise: Float32):
        self.noise = noise
    
    fn update_categorical_params(
        mut self,
        cat_params_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Update categorical parameters, recompute correlations, and re-materialize."""
        self.cat_params_ptr = cat_params_ptr
        self.cat_state.update_correlation_matrices(cat_params_ptr)
        # Re-materialize since K_cat changed
        self._materialize()
    
    fn set_cat_params_ptr(
        mut self,
        ptr: UnsafePointer[Float32, MutAnyOrigin],
    ):
        self.cat_params_ptr = ptr
    
    # =========================================================================
    # Prediction methods
    # =========================================================================
    
    fn cross_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        c_test_ptr: UnsafePointer[Int32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_test: Int,
    ) raises:
        alias BLOCK_SIZE = 256
        var num_blocks = (n_test + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_cross_matvec_8x[DIM, K, IS_PRODUCT]](
            out_ptr, x_test_ptr, self.x_ptr, v_ptr, self.params_ptr,
            c_test_ptr, self.cat_state.get_c_device_ptr(),
            n_test, self.n, self.cat_state.get_num_cat_vars(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            grid_dim=num_blocks, block_dim=BLOCK_SIZE,
        )
        self.ctx.synchronize()
    
    fn cross_covariance(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        c_test_ptr: UnsafePointer[Int32, MutAnyOrigin],
        n_test: Int,
    ) raises:
        alias BLOCK_SIZE = 16
        var grid_x = (self.n + BLOCK_SIZE - 1) // BLOCK_SIZE
        var grid_y = (n_test + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        self.ctx.enqueue_function[composite_mixed_cross_covariance_fused[DIM, K, IS_PRODUCT]](
            out_ptr, self.x_ptr, x_test_ptr, self.params_ptr,
            self.cat_state.get_c_device_ptr(), c_test_ptr,
            self.n, n_test, self.cat_state.get_num_cat_vars(),
            self.cat_state.get_corr_flat_device_ptr(),
            self.cat_state.get_offsets_device_ptr(),
            self.cat_state.get_levels_device_ptr(),
            grid_dim=(grid_x, grid_y), block_dim=(BLOCK_SIZE, BLOCK_SIZE),
        )
        self.ctx.synchronize()
