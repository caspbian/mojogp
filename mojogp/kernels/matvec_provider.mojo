"""MatvecProvider abstraction for O(n) and O(n²) memory strategies.

This module provides a unified interface for matrix-vector operations,
allowing the same CG/Lanczos/training code to work with both:
- Matrix-free (O(n) memory): compute K @ v on-the-fly
- Materialized (O(n²) memory): pre-compute K, use GEMM for K @ v
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu.profiler import ProfileBlock
from memory import UnsafePointer

from .kernel_params import KernelParams
from .constants import PROFILING

alias float_dtype = DType.float32


# =============================================================================
# MatvecProvider Trait
# =============================================================================

trait MatvecProvider:
    """Trait for matrix-vector product providers.
    
    Any type implementing this trait can be used with generic CG/Lanczos algorithms.
    """
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int
    ) raises:
        """Compute (K + noise*I) @ v for num_cols columns.
        
        Args:
            out_ptr: Output buffer [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns (for batched operations)
        """
        ...
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute ∂K/∂θ @ v for parameter gradient.
        
        Args:
            out_ptr: Output buffer [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
            param_index: Which parameter (0=lengthscale, 1=outputscale)
            sync: Whether to synchronize after kernel launch (default True).
                  Set to False when batching multiple gradient_matvec calls.
        
        Note: Noise gradient is just identity, handled separately.
        """
        ...
    
    fn get_n(self) -> Int:
        """Return the number of data points n."""
        ...
    
    fn get_ctx(self) -> DeviceContext:
        """Return the device context."""
        ...
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data."""
        ...
    
    fn get_d(self) -> Int:
        """Return input dimension."""
        ...
    
    fn get_kernel_type(self) -> Int:
        """Return kernel type constant."""
        ...
    
    fn get_lengthscale(self) -> Float32:
        """Return lengthscale."""
        ...
    
    fn get_outputscale(self) -> Float32:
        """Return output scale."""
        ...
    
    fn get_kernel_param1(self) -> Float32:
        """Return kernel parameter 1."""
        ...
    
    fn get_kernel_param2(self) -> Float32:
        """Return kernel parameter 2."""
        ...
    
    fn get_noise(self) -> Float32:
        """Return noise variance."""
        ...
    
    fn get_diagonal(self) -> Float32:
        """Return diagonal value (outputscale + noise for stationary kernels)."""
        ...
    
    fn update_hyperparams(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
    ) raises:
        """Update hyperparameters. May trigger re-materialization."""
        ...
    
    fn update_hyperparams_ard(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
        outputscale: Float32,
        noise: Float32,
    ) raises:
        """Update hyperparameters with ARD lengthscales.
        
        Args:
            lengthscales_host: Pointer to [d] lengthscales on HOST
            outputscale: New output scale
            noise: New noise variance
        """
        ...
    
    fn gradient_matvec_ard(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        dim_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute ∂K/∂l_d @ v for ARD lengthscale gradient (dimension d).
        
        Args:
            out_ptr: Output buffer [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
            dim_index: Which dimension (0..d-1)
            sync: Whether to synchronize after kernel launch.
        """
        ...
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract the full diagonal of K (without noise) to device buffer.
        
        Args:
            diag_ptr: Output buffer [n] on device to store diagonal values
        """
        ...
    
    fn get_use_ard(self) -> Bool:
        """Return whether ARD is enabled (per-dimension lengthscales)."""
        ...
    
    fn get_lengthscales_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to ARD lengthscales on device [d].
        
        Returns null pointer if use_ard=False.
        """
        ...

    fn get_inv_ls_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to precomputed 1/ls[d] on device.
        
        Returns null pointer if use_ard=False.
        """
        ...
    
    fn update_hyperparams_with_param1(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
        param1: Float32,
    ) raises:
        """Update hyperparameters including param1.
        
        Args:
            lengthscale: New lengthscale value
            outputscale: New outputscale value
            noise: New noise variance
            param1: New param1 value (period for Periodic, alpha for RQ)
        """
        ...
    
    fn update_param1(mut self, param1: Float32):
        """Update only param1 (period/alpha) without changing other hyperparameters.
        
        Args:
            param1: New param1 value
        """
        ...

    fn update_param2(mut self, param2: Float32):
        """Update only param2 (offset for Polynomial) without changing other hyperparameters.
        
        Args:
            param2: New param2 value
        """
        ...


# =============================================================================
# Provider Type Enum
# =============================================================================

@fieldwise_init
struct ProviderType(ImplicitlyCopyable):
    """Type of matvec provider."""
    var _value: Int
    
    alias MATRIX_FREE = ProviderType(0)
    alias MATERIALIZED = ProviderType(1)
    
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value
    
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# =============================================================================
# MatrixFreeProvider
# =============================================================================

struct MatrixFreeProvider(MatvecProvider, Movable):
    """O(n) memory provider - computes K @ v on-the-fly.
    
    This wraps the existing dispatch_forward_matvec functionality.
    """
    var ctx: DeviceContext
    var x_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var params_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var n: Int
    var d: Int
    var kernel_type: Int
    var use_ard: Bool
    var lengthscale: Float32
    var outputscale: Float32
    var noise: Float32
    var kernel_param1: Float32
    var kernel_param2: Float32
    var lengthscales_device: DeviceBuffer[float_dtype]  # [d] ARD lengthscales
    var inv_ls_device: DeviceBuffer[float_dtype]  # [d] precomputed 1/ls[d]
    var _ard_buffer_allocated: Bool  # Track if ARD buffer was allocated
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        params_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        d: Int,
        kernel_type: Int,
        use_ard: Bool,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
        kernel_param1: Float32 = 1.0,
        kernel_param2: Float32 = 0.0,
    ) raises:
        """Initialize matrix-free provider.
        
        Args:
            ctx: GPU device context
            x_ptr: Training data [n, d] row-major
            params_ptr: Kernel parameters on device
            n: Number of data points
            d: Input dimension
            kernel_type: Kernel type constant
            use_ard: Whether to use ARD
            lengthscale: Lengthscale (isotropic)
            outputscale: Output scale
            noise: Noise variance
            kernel_param1: Extra parameter 1 (period/alpha/variance/degree)
            kernel_param2: Extra parameter 2 (polynomial offset)
        """
        self.ctx = ctx
        self.x_ptr = x_ptr
        self.params_ptr = params_ptr
        self.n = n
        self.d = d
        self.kernel_type = kernel_type
        self.use_ard = use_ard
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.noise = noise
        self.kernel_param1 = kernel_param1
        self.kernel_param2 = kernel_param2
        
        # Allocate ARD lengthscales buffer if needed
        if use_ard:
            self.lengthscales_device = ctx.enqueue_create_buffer[float_dtype](d)
            self.inv_ls_device = ctx.enqueue_create_buffer[float_dtype](d)
            self._ard_buffer_allocated = True
            ctx.synchronize()
        else:
            # Create a dummy buffer (will not be used)
            self.lengthscales_device = ctx.enqueue_create_buffer[float_dtype](1)
            self.inv_ls_device = ctx.enqueue_create_buffer[float_dtype](1)
            self._ard_buffer_allocated = False
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int
    ) raises:
        """Compute (K + noise*I) @ v on-the-fly.
        
        Args:
            out_ptr: Output buffer [n, num_cols] column-major
            v_ptr: Input vectors [n, num_cols] column-major
            num_cols: Number of RHS columns
        """
        # Import here to avoid circular dependency
        from .cg_solver import dispatch_forward_matvec
        
        # BUG FIX: When use_ard=True, dispatch_forward_matvec reads [d]
        # per-dimension lengthscales from the pointer. self.params_ptr
        # points to all_params_device which only has [ls_scalar, 1.0],
        # NOT the d ARD lengthscales. Must use self.lengthscales_device
        # which was populated by update_hyperparams_ard().
        with ProfileBlock[False]("PROV_mf_forward_matvec"):  # Disabled: called 400+ times per iter
            if self.use_ard:
                dispatch_forward_matvec(
                    self.ctx,
                    self.kernel_type,
                    self.use_ard,
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.lengthscales_device.unsafe_ptr(),
                    self.lengthscale,
                    self.outputscale,
                    self.n,
                    self.d,
                    num_cols,
                    self.noise,
                    self.kernel_param1,
                    self.kernel_param2,
                    self.inv_ls_device.unsafe_ptr(),
                )
            else:
                dispatch_forward_matvec(
                    self.ctx,
                    self.kernel_type,
                    self.use_ard,
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.params_ptr,
                    self.lengthscale,
                    self.outputscale,
                    self.n,
                    self.d,
                    num_cols,
                    self.noise,
                    self.kernel_param1,
                    self.kernel_param2,
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
        """Compute ∂K/∂θ @ v for parameter gradient.
        
        Args:
            out_ptr: Output buffer [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
            param_index: Which parameter (0=lengthscale, 1=outputscale)
            sync: Whether to synchronize after kernel launch (default True).
                  Set to False when batching multiple gradient_matvec calls.
        """
        # Import here to avoid circular dependency
        from .dispatchers import unified_dispatch_gradient_matvec
        from .lanczos import compute_kernel_matvec_batched
        from .kernel_params import make_rbf_params, make_matern_params, make_periodic_params, make_rq_params, make_linear_params, make_polynomial_params
        from .constants import (
            KERNEL_TYPE_RBF,
            KERNEL_TYPE_MATERN12,
            KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52,
            KERNEL_TYPE_PERIODIC,
            KERNEL_TYPE_RQ,
            KERNEL_TYPE_LINEAR,
            KERNEL_TYPE_POLYNOMIAL,
        )
        
        if param_index == 0:  # lengthscale gradient
            # Create KernelParams for gradient computation
            var params: KernelParams
            if self.kernel_type == KERNEL_TYPE_RBF:
                params = make_rbf_params(self.outputscale, self.lengthscale, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_MATERN12:
                params = make_matern_params(self.outputscale, self.lengthscale, Float32(0.5), self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_MATERN32:
                params = make_matern_params(self.outputscale, self.lengthscale, Float32(1.5), self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_MATERN52:
                params = make_matern_params(self.outputscale, self.lengthscale, Float32(2.5), self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_PERIODIC:
                params = make_periodic_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_RQ:
                params = make_rq_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_LINEAR:
                params = make_linear_params(self.outputscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
                params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, self.params_ptr, is_ard=self.use_ard)
            else:
                raise Error("Unknown kernel_type: " + String(self.kernel_type))
            
            # Multi-column gradient in ONE kernel launch (shared memory)
            from .dispatchers import unified_dispatch_gradient_matvec_multicol
            with ProfileBlock[PROFILING]("PROV_mf_grad_ls_matvec"):
                unified_dispatch_gradient_matvec_multicol(
                    self.ctx,
                    self.kernel_type,
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.n,
                    self.d,
                    num_cols,
                    params,
                    -1,  # -1 for isotropic lengthscale gradient
                )
                self.ctx.synchronize()
            if sync:
                self.ctx.synchronize()
            
        elif param_index == 1:  # outputscale gradient
            # ∂K/∂σ_f² = K / σ_f² (unscaled kernel with outputscale=1.0)
            # Use dispatch_forward_matvec directly so outputscale gradients take
            # the same KeOps-style shared-memory route as forward matvecs.
            with ProfileBlock[PROFILING]("PROV_mf_grad_os_matvec"):
                from .cg_solver import dispatch_forward_matvec as dispatch_fwd
                dispatch_fwd(
                    self.ctx,
                    self.kernel_type,
                    False,  # use_ard
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.params_ptr,
                    self.lengthscale,
                    Float32(1.0),  # outputscale=1 (for ∂K/∂σ² = K/σ²)
                    self.n,
                    self.d,
                    num_cols,
                    Float32(0.0),  # noise=0
                    self.kernel_param1,
                    self.kernel_param2,
                )
                self.ctx.synchronize()
            if sync:
                self.ctx.synchronize()
        
        elif param_index == 2:  # param1 gradient (period/alpha/variance/degree)
            from .dispatchers import unified_dispatch_param1_gradient_matvec_multicol
            from .kernel_params import make_periodic_params, make_rq_params, make_linear_params, make_polynomial_params
            from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
            
            var params: KernelParams
            if self.kernel_type == KERNEL_TYPE_PERIODIC:
                params = make_periodic_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_RQ:
                params = make_rq_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_LINEAR:
                params = make_linear_params(self.outputscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
                params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, self.params_ptr, is_ard=self.use_ard)
            else:
                raise Error("param_index=2 (param1 gradient) only supported for PERIODIC, RQ, LINEAR, and POLYNOMIAL kernels")
            
            # Multi-column param1 gradient in one launch (shared memory)
            unified_dispatch_param1_gradient_matvec_multicol(
                self.ctx,
                self.kernel_type,
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.n,
                self.d,
                num_cols,
                params,
            )
            if sync:
                self.ctx.synchronize()

        elif param_index == 3:  # param2 gradient (offset for Polynomial)
            from .dispatchers import unified_dispatch_param2_gradient_matvec_multicol
            from .kernel_params import make_polynomial_params
            from .constants import KERNEL_TYPE_POLYNOMIAL

            if self.kernel_type != KERNEL_TYPE_POLYNOMIAL:
                raise Error("param_index=3 (param2 gradient) only supported for POLYNOMIAL kernel")

            var params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, self.params_ptr, is_ard=self.use_ard)

            # Multi-column param2 gradient in one launch (shared memory)
            unified_dispatch_param2_gradient_matvec_multicol(
                self.ctx,
                self.kernel_type,
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.n,
                self.d,
                num_cols,
                params,
            )
            if sync:
                self.ctx.synchronize()

        else:
            raise Error("Invalid param_index: " + String(param_index))
    
    fn update_hyperparams(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32
    ) raises:
        """Update hyperparameters (no re-computation needed).
        
        Args:
            lengthscale: New lengthscale
            outputscale: New output scale
            noise: New noise variance
        """
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.noise = noise
    
    fn update_hyperparams_with_param1(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
        param1: Float32,
    ) raises:
        """Update hyperparameters including param1 (period/alpha).
        
        Args:
            lengthscale: New lengthscale
            outputscale: New output scale
            noise: New noise variance
            param1: New param1 value (period for Periodic, alpha for RQ)
        """
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.noise = noise
        self.kernel_param1 = param1
    
    fn update_param1(mut self, param1: Float32):
        """Update only param1 (period/alpha) without changing other hyperparameters.
        
        Args:
            param1: New param1 value
        """
        self.kernel_param1 = param1

    fn update_param2(mut self, param2: Float32):
        """Update only param2 (offset for Polynomial) without changing other hyperparameters.
        
        Args:
            param2: New param2 value
        """
        self.kernel_param2 = param2
    
    fn get_noise(self) -> Float32:
        """Return noise variance."""
        return self.noise
    
    fn get_diagonal(self) -> Float32:
        """Return diagonal value (outputscale + noise for stationary kernels).
        
        Returns:
            Diagonal value of K + noise*I
        """
        return self.outputscale + self.noise
    
    fn get_n(self) -> Int:
        """Return the number of data points n."""
        return self.n
    
    fn get_ctx(self) -> DeviceContext:
        """Return the device context."""
        return self.ctx
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data."""
        return self.x_ptr
    
    fn get_d(self) -> Int:
        """Return input dimension."""
        return self.d
    
    fn get_kernel_type(self) -> Int:
        """Return kernel type constant."""
        return self.kernel_type
    
    fn get_lengthscale(self) -> Float32:
        """Return lengthscale."""
        return self.lengthscale
    
    fn get_outputscale(self) -> Float32:
        """Return output scale."""
        return self.outputscale
    
    fn get_kernel_param1(self) -> Float32:
        """Return kernel parameter 1."""
        return self.kernel_param1
    
    fn get_kernel_param2(self) -> Float32:
        """Return kernel parameter 2."""
        return self.kernel_param2
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract the full diagonal of K (without noise) to device buffer.
        
        For stationary kernels (RBF, Matern, Periodic, RQ): diagonal is constant = outputscale.
        For non-stationary kernels (Linear, Polynomial): diagonal varies per data point.
        """
        from .cg_solver import kernel_fill_constant, kernel_compute_diagonal_nonstationary
        from .constants import KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
        
        var grid_dim = (self.n + 255) // 256
        
        if self.kernel_type == KERNEL_TYPE_LINEAR or self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
            self.ctx.enqueue_function[kernel_compute_diagonal_nonstationary](
                diag_ptr, self.x_ptr, self.n, self.d, self.outputscale,
                self.kernel_type, self.kernel_param1, self.kernel_param2,
                grid_dim=(grid_dim,), block_dim=(256,)
            )
        else:
            self.ctx.enqueue_function[kernel_fill_constant](
                diag_ptr, self.n, self.outputscale,
                grid_dim=(grid_dim,), block_dim=(256,)
            )
        self.ctx.synchronize()
    
    # =========================================================================
    # ARD-specific methods
    # =========================================================================
    
    fn gradient_matvec_ard(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        dim_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute ∂K/∂l_d @ v for ARD lengthscale gradient (dimension d).
        
        This computes the gradient with respect to a single dimension's lengthscale.
        For full ARD gradient, call this for each dimension d in 0..self.d-1.
        
        Args:
            out_ptr: Output buffer [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
            dim_index: Which dimension (0..d-1)
            sync: Whether to synchronize after kernel launch (default True).
        """
        if not self.use_ard:
            raise Error("gradient_matvec_ard called but use_ard=False")
        if dim_index < 0 or dim_index >= self.d:
            raise Error("dim_index out of range: " + String(dim_index) + " (d=" + String(self.d) + ")")
        
        from .dispatchers import unified_dispatch_gradient_matvec
        from .kernel_params import make_rbf_params, make_matern_params, make_periodic_params, make_rq_params, make_linear_params, make_polynomial_params
        from .constants import (
            KERNEL_TYPE_RBF,
            KERNEL_TYPE_MATERN12,
            KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52,
            KERNEL_TYPE_PERIODIC,
            KERNEL_TYPE_RQ,
            KERNEL_TYPE_LINEAR,
            KERNEL_TYPE_POLYNOMIAL,
        )
        
        # Create KernelParams with ARD lengthscales + precomputed inv_ls
        var params: KernelParams
        var ls_p = self.lengthscales_device.unsafe_ptr()
        var inv_ls_p = self.inv_ls_device.unsafe_ptr()
        if self.kernel_type == KERNEL_TYPE_RBF:
            params = make_rbf_params(self.outputscale, self.lengthscale, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_MATERN12:
            params = make_matern_params(self.outputscale, self.lengthscale, Float32(0.5), ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_MATERN32:
            params = make_matern_params(self.outputscale, self.lengthscale, Float32(1.5), ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_MATERN52:
            params = make_matern_params(self.outputscale, self.lengthscale, Float32(2.5), ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_PERIODIC:
            params = make_periodic_params(self.outputscale, self.lengthscale, self.kernel_param1, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_RQ:
            params = make_rq_params(self.outputscale, self.lengthscale, self.kernel_param1, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_LINEAR:
            params = make_linear_params(self.outputscale, self.kernel_param1, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
            params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, ls_p, inv_ls_p, True)
        else:
            raise Error("Unknown kernel_type: " + String(self.kernel_type))
        
        # Multi-column ARD gradient in one launch (shared memory)
        from .dispatchers import unified_dispatch_gradient_matvec_multicol
        unified_dispatch_gradient_matvec_multicol(
            self.ctx,
            self.kernel_type,
            out_ptr,
            self.x_ptr,
            v_ptr,
            self.n,
            self.d,
            num_cols,
            params,
            dim_index,  # Per-dimension gradient
        )
        if sync:
            self.ctx.synchronize()
    
    fn update_hyperparams_ard(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
        outputscale: Float32,
        noise: Float32
    ) raises:
        """Update hyperparameters with ARD lengthscales.
        
        Unlike MaterializedProvider, this does NOT need to re-materialize K
        since we compute K @ v on-the-fly.
        
        Args:
            lengthscales_host: Pointer to [d] lengthscales on HOST
            outputscale: New output scale
            noise: New noise variance
        """
        if not self.use_ard:
            raise Error("update_hyperparams_ard called but use_ard=False")
        
        # Copy lengthscales from host to device
        self.set_lengthscales_device(lengthscales_host)
        self.outputscale = outputscale
        self.noise = noise
    
    fn set_lengthscales_device(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Copy lengthscales from host to device buffer and precompute inv_ls.
        
        This is used during initialization to set up ARD lengthscales.
        Also precomputes inv_ls[d] = 1.0 / ls[d] for fast kernel evaluation.
        
        Args:
            lengthscales_host: Pointer to [d] lengthscales on HOST
        """
        var ls_host_buf = self.ctx.enqueue_create_host_buffer[float_dtype](self.d)
        var inv_ls_host_buf = self.ctx.enqueue_create_host_buffer[float_dtype](self.d)
        for i in range(self.d):
            ls_host_buf[i] = lengthscales_host[i]
            inv_ls_host_buf[i] = Float32(1.0) / lengthscales_host[i]
        self.ctx.enqueue_copy(dst_buf=self.lengthscales_device, src_buf=ls_host_buf)
        self.ctx.enqueue_copy(dst_buf=self.inv_ls_device, src_buf=inv_ls_host_buf)
        self.ctx.synchronize()
    
    fn get_use_ard(self) -> Bool:
        """Return whether ARD is enabled."""
        return self.use_ard
    
    fn get_lengthscales_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to ARD lengthscales [d] on device."""
        return self.lengthscales_device.unsafe_ptr()

    fn get_inv_ls_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to precomputed 1/ls[d] on device."""
        return self.inv_ls_device.unsafe_ptr()


# =============================================================================
# MaterializedProvider
# =============================================================================

struct MaterializedProvider(MatvecProvider, Movable):
    """O(n²) memory provider - materializes K, uses GEMM for K @ v.
    
    The kernel matrix K is computed once and cached. When hyperparameters
    change, K is re-materialized.
    
    Supports both isotropic (single lengthscale) and ARD (per-dimension lengthscales).
    """
    var ctx: DeviceContext
    var x_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var K_device: DeviceBuffer[float_dtype]  # n × n kernel matrix (row-major)
    var n: Int
    var d: Int
    var kernel_type: Int
    var use_ard: Bool
    var lengthscale: Float32  # Used for isotropic mode
    var outputscale: Float32
    var noise: Float32
    var kernel_param1: Float32
    var kernel_param2: Float32
    var is_materialized: Bool
    var params_ptr: UnsafePointer[Float32, MutAnyOrigin]  # For creating KernelParams
    # ARD-specific: device buffer for per-dimension lengthscales [d]
    var lengthscales_device: DeviceBuffer[float_dtype]
    var inv_ls_device: DeviceBuffer[float_dtype]  # [d] precomputed 1/ls[d]
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        params_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        d: Int,
        kernel_type: Int,
        use_ard: Bool,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
        kernel_param1: Float32 = 1.0,
        kernel_param2: Float32 = 0.0,
    ) raises:
        """Initialize materialized provider and materialize K.
        
        Args:
            ctx: GPU device context
            x_ptr: Training data [n, d] row-major
            params_ptr: Kernel parameters on device
            n: Number of data points
            d: Input dimension
            kernel_type: Kernel type constant
            use_ard: Whether to use ARD
            lengthscale: Lengthscale (isotropic)
            outputscale: Output scale
            noise: Noise variance
            kernel_param1: Extra parameter 1 (period/alpha/variance/degree)
            kernel_param2: Extra parameter 2 (polynomial offset)
        """
        self.ctx = ctx
        self.x_ptr = x_ptr
        self.params_ptr = params_ptr
        self.n = n
        self.d = d
        self.kernel_type = kernel_type
        self.use_ard = use_ard
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.noise = noise
        self.kernel_param1 = kernel_param1
        self.kernel_param2 = kernel_param2
        self.is_materialized = False
        
        # Allocate kernel matrix buffer
        self.K_device = ctx.enqueue_create_buffer[float_dtype](n * n)
        
        # Allocate ARD lengthscales buffer (always allocate, even if not used)
        # This simplifies the struct layout and avoids optional handling
        self.lengthscales_device = ctx.enqueue_create_buffer[float_dtype](d)
        self.inv_ls_device = ctx.enqueue_create_buffer[float_dtype](d)
        
        # Materialize K
        self._materialize_kernel()
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int
    ) raises:
        """Compute (K + noise*I) @ v using GEMM.
        
        Args:
            out_ptr: Output buffer [n, num_cols] column-major
            v_ptr: Input vectors [n, num_cols] column-major
            num_cols: Number of RHS columns
        """
        from .gemm_matvec import gemm_matvec, add_noise_diagonal
        
        # DEBUG: Print noise value (disabled to reduce output)
        # print("DEBUG MaterializedProvider.forward_matvec: noise =", self.noise, ", n =", self.n, ", num_cols =", num_cols, ", outputscale =", self.outputscale, ", lengthscale =", self.lengthscale)
        
        with ProfileBlock[False]("PROV_mat_forward_matvec"):  # Disabled: called 400+ times per iter
            # 1. out = K @ v (using GEMM)
            gemm_matvec(
                self.ctx,
                out_ptr,
                self.K_device.unsafe_ptr(),
                v_ptr,
                self.n,
                num_cols,
            )
            
            # 2. out += noise * v (add noise diagonal)
            add_noise_diagonal(
                self.ctx,
                out_ptr,
                v_ptr,
                self.n,
                num_cols,
                self.noise,
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
        """Compute ∂K/∂θ @ v for parameter gradient.
        
        For materialized provider, we use matrix-free gradient computation
        since materializing gradient matrices would require O(n²) memory per parameter.
        
        Args:
            out_ptr: Output buffer [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
            param_index: Which parameter (0=lengthscale, 1=outputscale)
            sync: Whether to synchronize after kernel launch (default True).
                  Set to False when batching multiple gradient_matvec calls.
        """
        # Import here to avoid circular dependency
        from .dispatchers import unified_dispatch_gradient_matvec
        from .lanczos import compute_kernel_matvec_batched
        from .kernel_params import make_rbf_params, make_matern_params, make_periodic_params, make_rq_params, make_linear_params, make_polynomial_params
        from .constants import (
            KERNEL_TYPE_RBF,
            KERNEL_TYPE_MATERN12,
            KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52,
            KERNEL_TYPE_PERIODIC,
            KERNEL_TYPE_RQ,
            KERNEL_TYPE_LINEAR,
            KERNEL_TYPE_POLYNOMIAL,
        )
        
        if param_index == 0:  # lengthscale gradient
            # Create KernelParams for gradient computation
            var params: KernelParams
            if self.kernel_type == KERNEL_TYPE_RBF:
                params = make_rbf_params(self.outputscale, self.lengthscale, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_MATERN12:
                params = make_matern_params(self.outputscale, self.lengthscale, Float32(0.5), self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_MATERN32:
                params = make_matern_params(self.outputscale, self.lengthscale, Float32(1.5), self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_MATERN52:
                params = make_matern_params(self.outputscale, self.lengthscale, Float32(2.5), self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_PERIODIC:
                params = make_periodic_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_RQ:
                params = make_rq_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_LINEAR:
                params = make_linear_params(self.outputscale, self.kernel_param1, self.params_ptr, is_ard=self.use_ard)
            elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
                params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, self.params_ptr, is_ard=self.use_ard)
            else:
                raise Error("Unknown kernel_type: " + String(self.kernel_type))
            
            # Multi-column gradient in ONE kernel launch (shared memory)
            from .dispatchers import unified_dispatch_gradient_matvec_multicol
            with ProfileBlock[PROFILING]("PROV_mat_grad_ls_matvec"):
                unified_dispatch_gradient_matvec_multicol(
                    self.ctx,
                    self.kernel_type,
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.n,
                    self.d,
                    num_cols,
                    params,
                    -1,  # -1 for isotropic lengthscale gradient
                )
                self.ctx.synchronize()
            if sync:
                self.ctx.synchronize()
            
        elif param_index == 1:  # outputscale gradient
            # ∂K/∂σ_f² = K / σ_f² — use fast shared-memory forward_matvec
            with ProfileBlock[PROFILING]("PROV_mat_grad_os_matvec"):
                from .cg_solver import dispatch_forward_matvec as dispatch_fwd
                dispatch_fwd(
                    self.ctx,
                    self.kernel_type,
                    False,  # use_ard
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.params_ptr,
                    self.lengthscale,
                    Float32(1.0),  # outputscale=1
                    self.n,
                    self.d,
                    num_cols,
                    Float32(0.0),  # noise=0
                    self.kernel_param1,
                    self.kernel_param2,
                )
                self.ctx.synchronize()
            if sync:
                self.ctx.synchronize()
        
        elif param_index == 2:  # param1 gradient (period/alpha/variance)
            from .dispatchers import unified_dispatch_param1_gradient_matvec_multicol
            from .kernel_params import make_periodic_params, make_rq_params, make_linear_params, make_polynomial_params
            from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
            
            var params: KernelParams
            if self.use_ard:
                var ls_p2 = self.lengthscales_device.unsafe_ptr()
                var inv_ls_p2 = self.inv_ls_device.unsafe_ptr()
                if self.kernel_type == KERNEL_TYPE_PERIODIC:
                    params = make_periodic_params(self.outputscale, self.lengthscale, self.kernel_param1, ls_p2, inv_ls_p2, True)
                elif self.kernel_type == KERNEL_TYPE_RQ:
                    params = make_rq_params(self.outputscale, self.lengthscale, self.kernel_param1, ls_p2, inv_ls_p2, True)
                elif self.kernel_type == KERNEL_TYPE_LINEAR:
                    params = make_linear_params(self.outputscale, self.kernel_param1, ls_p2, inv_ls_p2, True)
                elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
                    params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, ls_p2, inv_ls_p2, True)
                else:
                    raise Error("param_index=2 (param1 gradient) only supported for PERIODIC, RQ, LINEAR, and POLYNOMIAL kernels")
            else:
                if self.kernel_type == KERNEL_TYPE_PERIODIC:
                    params = make_periodic_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=False)
                elif self.kernel_type == KERNEL_TYPE_RQ:
                    params = make_rq_params(self.outputscale, self.lengthscale, self.kernel_param1, self.params_ptr, is_ard=False)
                elif self.kernel_type == KERNEL_TYPE_LINEAR:
                    params = make_linear_params(self.outputscale, self.kernel_param1, self.params_ptr, is_ard=False)
                elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
                    params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, self.params_ptr, is_ard=False)
                else:
                    raise Error("param_index=2 (param1 gradient) only supported for PERIODIC, RQ, LINEAR, and POLYNOMIAL kernels")
            
            # Multi-column param1 gradient in one launch (shared memory)
            unified_dispatch_param1_gradient_matvec_multicol(
                self.ctx,
                self.kernel_type,
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.n,
                self.d,
                num_cols,
                params,
            )
            if sync:
                self.ctx.synchronize()

        elif param_index == 3:  # param2 gradient (offset for Polynomial)
            from .dispatchers import unified_dispatch_param2_gradient_matvec_multicol
            from .kernel_params import make_polynomial_params
            from .constants import KERNEL_TYPE_POLYNOMIAL

            if self.kernel_type != KERNEL_TYPE_POLYNOMIAL:
                raise Error("param_index=3 (param2 gradient) only supported for POLYNOMIAL kernel")

            var params: KernelParams
            if self.use_ard:
                params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, self.lengthscales_device.unsafe_ptr(), self.inv_ls_device.unsafe_ptr(), True)
            else:
                params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, self.params_ptr, is_ard=False)

            # Multi-column param2 gradient in one launch (shared memory)
            unified_dispatch_param2_gradient_matvec_multicol(
                self.ctx,
                self.kernel_type,
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.n,
                self.d,
                num_cols,
                params,
            )
            if sync:
                self.ctx.synchronize()

        else:
            raise Error("Invalid param_index: " + String(param_index))
    
    fn gradient_matvec_ard(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        dim_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute ∂K/∂l_d @ v for ARD lengthscale gradient (dimension d).
        
        This computes the gradient with respect to a single dimension's lengthscale.
        For full ARD gradient, call this for each dimension d in 0..self.d-1.
        
        Args:
            out_ptr: Output buffer [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
            dim_index: Which dimension (0..d-1)
            sync: Whether to synchronize after kernel launch (default True).
        """
        if not self.use_ard:
            raise Error("gradient_matvec_ard called but use_ard=False")
        if dim_index < 0 or dim_index >= self.d:
            raise Error("dim_index out of range: " + String(dim_index) + " (d=" + String(self.d) + ")")
        
        from .dispatchers import unified_dispatch_gradient_matvec
        from .kernel_params import make_rbf_params, make_matern_params, make_periodic_params, make_rq_params, make_linear_params, make_polynomial_params
        from .constants import (
            KERNEL_TYPE_RBF,
            KERNEL_TYPE_MATERN12,
            KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52,
            KERNEL_TYPE_PERIODIC,
            KERNEL_TYPE_RQ,
            KERNEL_TYPE_LINEAR,
            KERNEL_TYPE_POLYNOMIAL,
        )
        
        # Create KernelParams with ARD lengthscales + precomputed inv_ls
        var params: KernelParams
        var ls_p = self.lengthscales_device.unsafe_ptr()
        var inv_ls_p = self.inv_ls_device.unsafe_ptr()
        if self.kernel_type == KERNEL_TYPE_RBF:
            params = make_rbf_params(self.outputscale, self.lengthscale, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_MATERN12:
            params = make_matern_params(self.outputscale, self.lengthscale, Float32(0.5), ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_MATERN32:
            params = make_matern_params(self.outputscale, self.lengthscale, Float32(1.5), ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_MATERN52:
            params = make_matern_params(self.outputscale, self.lengthscale, Float32(2.5), ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_PERIODIC:
            params = make_periodic_params(self.outputscale, self.lengthscale, self.kernel_param1, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_RQ:
            params = make_rq_params(self.outputscale, self.lengthscale, self.kernel_param1, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_LINEAR:
            params = make_linear_params(self.outputscale, self.kernel_param1, ls_p, inv_ls_p, True)
        elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
            params = make_polynomial_params(self.outputscale, self.kernel_param1, self.kernel_param2, ls_p, inv_ls_p, True)
        else:
            raise Error("Unknown kernel_type: " + String(self.kernel_type))
        
        # Multi-column ARD gradient in one launch (shared memory)
        from .dispatchers import unified_dispatch_gradient_matvec_multicol
        unified_dispatch_gradient_matvec_multicol(
            self.ctx,
            self.kernel_type,
            out_ptr,
            self.x_ptr,
            v_ptr,
            self.n,
            self.d,
            num_cols,
            params,
            dim_index,  # Per-dimension gradient
        )
        if sync:
            self.ctx.synchronize()
    
    fn update_hyperparams(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32
    ) raises:
        """Update hyperparameters (isotropic) and re-materialize K.
        
        Args:
            lengthscale: New lengthscale
            outputscale: New output scale
            noise: New noise variance
        """
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.noise = noise
        self._materialize_kernel()
    
    fn update_hyperparams_with_param1(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
        param1: Float32,
    ) raises:
        """Update hyperparameters including param1 (period/alpha) and re-materialize K.
        
        Args:
            lengthscale: New lengthscale
            outputscale: New output scale
            noise: New noise variance
            param1: New param1 value (period for Periodic, alpha for RQ)
        """
        self.lengthscale = lengthscale
        self.outputscale = outputscale
        self.noise = noise
        self.kernel_param1 = param1
        self._materialize_kernel()
    
    fn update_param1(mut self, param1: Float32):
        """Update only param1 (period/alpha) and re-materialize K.
        
        Args:
            param1: New param1 value
        """
        self.kernel_param1 = param1

    fn update_param2(mut self, param2: Float32):
        """Update only param2 (offset for Polynomial) without changing other hyperparameters.
        
        Args:
            param2: New param2 value
        """
        self.kernel_param2 = param2
    
    fn update_hyperparams_ard(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
        outputscale: Float32,
        noise: Float32
    ) raises:
        """Update hyperparameters (ARD) and re-materialize K.
        
        Args:
            lengthscales_host: Pointer to [d] lengthscales on HOST
            outputscale: New output scale
            noise: New noise variance
        """
        if not self.use_ard:
            raise Error("update_hyperparams_ard called but use_ard=False")
        
        # Copy lengthscales and precomputed inv_ls from host to device
        var ls_host_buf = self.ctx.enqueue_create_host_buffer[float_dtype](self.d)
        var inv_ls_host_buf = self.ctx.enqueue_create_host_buffer[float_dtype](self.d)
        for i in range(self.d):
            ls_host_buf[i] = lengthscales_host[i]
            inv_ls_host_buf[i] = Float32(1.0) / lengthscales_host[i]
        self.ctx.enqueue_copy(dst_buf=self.lengthscales_device, src_buf=ls_host_buf)
        self.ctx.enqueue_copy(dst_buf=self.inv_ls_device, src_buf=inv_ls_host_buf)
        self.ctx.synchronize()
        
        self.outputscale = outputscale
        self.noise = noise
        self._materialize_kernel()
    
    fn set_lengthscales_device(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Copy lengthscales from host to device buffer and precompute inv_ls.
        
        Also precomputes inv_ls[d] = 1.0 / ls[d] for fast kernel evaluation.
        
        Args:
            lengthscales_host: Pointer to [d] lengthscales on HOST
        """
        var ls_host_buf = self.ctx.enqueue_create_host_buffer[float_dtype](self.d)
        var inv_ls_host_buf = self.ctx.enqueue_create_host_buffer[float_dtype](self.d)
        for i in range(self.d):
            ls_host_buf[i] = lengthscales_host[i]
            inv_ls_host_buf[i] = Float32(1.0) / lengthscales_host[i]
        self.ctx.enqueue_copy(dst_buf=self.lengthscales_device, src_buf=ls_host_buf)
        self.ctx.enqueue_copy(dst_buf=self.inv_ls_device, src_buf=inv_ls_host_buf)
        self.ctx.synchronize()
    
    fn get_use_ard(self) -> Bool:
        """Return whether ARD is enabled."""
        return self.use_ard
    
    fn get_lengthscales_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to ARD lengthscales [d] on device."""
        return self.lengthscales_device.unsafe_ptr()

    fn get_inv_ls_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to precomputed 1/ls[d] on device."""
        return self.inv_ls_device.unsafe_ptr()
    
    fn _materialize_kernel(mut self) raises:
        """Compute and cache K[i,j] = k(x[i], x[j]) for all i,j."""
        from .kernel_materialization import materialize_kernel_matrix
        from .kernel_params import (
            make_rbf_params,
            make_matern_params,
            make_periodic_params,
            make_rq_params,
            make_linear_params,
            make_polynomial_params,
        )
        from .constants import (
            KERNEL_TYPE_RBF,
            KERNEL_TYPE_MATERN12,
            KERNEL_TYPE_MATERN32,
            KERNEL_TYPE_MATERN52,
            KERNEL_TYPE_PERIODIC,
            KERNEL_TYPE_RQ,
            KERNEL_TYPE_LINEAR,
            KERNEL_TYPE_POLYNOMIAL,
        )
        
        # Create KernelParams based on kernel type
        # ARD stores lengthscales separately; isotropic kernels use the compact params buffer.
        var ls_ptr: UnsafePointer[Float32, MutAnyOrigin]
        if self.use_ard:
            ls_ptr = self.lengthscales_device.unsafe_ptr()
        else:
            ls_ptr = self.params_ptr
        var params: KernelParams
        
        var inv_ls_p_mat: UnsafePointer[Float32, MutAnyOrigin]
        if self.use_ard:
            inv_ls_p_mat = self.inv_ls_device.unsafe_ptr()
        else:
            inv_ls_p_mat = UnsafePointer[Float32, MutAnyOrigin]()
        if self.kernel_type == KERNEL_TYPE_RBF:
            params = make_rbf_params(
                self.outputscale, self.lengthscale, ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        elif self.kernel_type == KERNEL_TYPE_MATERN12:
            params = make_matern_params(
                self.outputscale, self.lengthscale, Float32(0.5), ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        elif self.kernel_type == KERNEL_TYPE_MATERN32:
            params = make_matern_params(
                self.outputscale, self.lengthscale, Float32(1.5), ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        elif self.kernel_type == KERNEL_TYPE_MATERN52:
            params = make_matern_params(
                self.outputscale, self.lengthscale, Float32(2.5), ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        elif self.kernel_type == KERNEL_TYPE_PERIODIC:
            params = make_periodic_params(
                self.outputscale, self.lengthscale, self.kernel_param1, ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        elif self.kernel_type == KERNEL_TYPE_RQ:
            params = make_rq_params(
                self.outputscale, self.lengthscale, self.kernel_param1, ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        elif self.kernel_type == KERNEL_TYPE_LINEAR:
            params = make_linear_params(
                self.outputscale, self.kernel_param1, ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        elif self.kernel_type == KERNEL_TYPE_POLYNOMIAL:
            params = make_polynomial_params(
                self.outputscale, self.kernel_param1, self.kernel_param2, ls_ptr, inv_ls_p_mat, is_ard=self.use_ard
            )
        else:
            raise Error("Unknown kernel_type: " + String(self.kernel_type))
        
        # Materialize the kernel matrix
        with ProfileBlock[PROFILING]("PROV_materialize_kernel"):
            materialize_kernel_matrix(
                self.ctx,
                self.K_device,
                self.x_ptr,
                self.n,
                self.d,
                self.kernel_type,
                params,
            )
            self.ctx.synchronize()
        
        self.is_materialized = True
    
    fn get_noise(self) -> Float32:
        """Return noise variance."""
        return self.noise
    
    fn get_diagonal(self) -> Float32:
        """Return diagonal value.
        
        Returns:
            Diagonal value of K + noise*I
        """
        return self.outputscale + self.noise
    
    fn get_kernel_matrix(self) -> DeviceBuffer[float_dtype]:
        """Return the materialized kernel matrix (for variance computation).
        
        Returns:
            Kernel matrix K [n, n] row-major
        """
        return self.K_device
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract the full diagonal of K (without noise) to device buffer.
        
        For MaterializedProvider, extracts actual diagonal from the materialized kernel matrix.
        This is correct for both stationary (RBF, Matern) and non-stationary (Linear, Polynomial) kernels.
        """
        from .kernel_materialization import kernel_extract_diagonal_from_matrix
        
        var grid_dim = (self.n + 255) // 256
        self.ctx.enqueue_function[kernel_extract_diagonal_from_matrix](
            diag_ptr, self.K_device.unsafe_ptr(), self.n,
            grid_dim=(grid_dim,), block_dim=(256,)
        )
        self.ctx.synchronize()
    
    fn get_n(self) -> Int:
        """Return the number of data points n."""
        return self.n
    
    fn get_ctx(self) -> DeviceContext:
        """Return the device context."""
        return self.ctx
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data."""
        return self.x_ptr
    
    fn get_d(self) -> Int:
        """Return input dimension."""
        return self.d
    
    fn get_kernel_type(self) -> Int:
        """Return kernel type constant."""
        return self.kernel_type
    
    fn get_lengthscale(self) -> Float32:
        """Return lengthscale."""
        return self.lengthscale
    
    fn get_outputscale(self) -> Float32:
        """Return output scale."""
        return self.outputscale
    
    fn get_kernel_param1(self) -> Float32:
        """Return kernel parameter 1."""
        return self.kernel_param1
    
    fn get_kernel_param2(self) -> Float32:
        """Return kernel parameter 2."""
        return self.kernel_param2
