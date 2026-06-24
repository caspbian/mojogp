"""GradientProvider trait for unified BBMM.

This module provides the GradientProvider trait and adapter structs that enable
a single unified BBMM function to work with all kernel types:
- Isotropic RBF/Matern (2 gradient params: lengthscale, outputscale)
- ARD kernels (d+1 gradient params: per-dim lengthscales + outputscale)
- Composite kernels (N gradient params as defined by the kernel)

The key insight is that BBMM only needs:
1. Forward matvec: (K + noise*I) @ v
2. Gradient matvecs: dK/dtheta_i @ v for each hyperparameter
3. Metadata: n, noise, ctx, num_gradient_params
4. Diagonal extraction for preconditioner construction

By abstracting these operations into a trait, we can write one BBMM function
that works with all kernel types via adapter structs.
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu.id import block_dim, block_idx, thread_idx

from .kernel_params import KernelParams
from memory import UnsafePointer

from .matvec_provider import MatvecProvider
from .composable_kernel import ComposableKernel
from .composite_provider import CompositeProvider, MaterializedCompositeProvider


# =============================================================================
# ForwardProvider Trait — forward matvec + metadata (no gradient methods)
# =============================================================================

trait ForwardProvider:
    """Minimal trait for CG solver and preconditioner: forward matvec only.
    
    This trait captures what the CG solver and preconditioner builder need:
    1. Forward matvec: (K + noise*I) @ v
    2. Metadata: n, noise, ctx
    3. Diagonal extraction for preconditioner construction
    
    Prediction code and CG solvers only need ForwardProvider, NOT GradientProvider.
    This enables prediction modules to compile without gradient dispatcher imports.
    """
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (K + noise*I) @ v for num_cols columns."""
        ...
    
    fn get_n(self) -> Int:
        """Return number of data points."""
        ...
    
    fn get_ctx(self) -> DeviceContext:
        """Return GPU device context."""
        ...
    
    fn get_noise(self) -> Float32:
        """Return noise variance sigma^2."""
        ...
    
    fn get_diagonal_value(self) -> Float32:
        """Return the diagonal value of K (without noise).
        
        For stationary kernels this is outputscale.
        Used by the preconditioner builder for initial diagonal.
        """
        ...
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract the full diagonal of K (without noise) to device buffer.
        
        For non-stationary kernels where diagonal varies.
        
        Args:
            diag_ptr: Output buffer [n] on device to store diagonal values
        """
        ...
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data on device.
        
        Needed by the preconditioner builder for column extraction.
        """
        ...


# =============================================================================
# GradientProvider Trait — extends ForwardProvider with gradient methods
# =============================================================================

trait GradientProvider(ForwardProvider):
    """Full trait for BBMM training: forward matvec + gradient matvecs.
    
    Extends ForwardProvider with gradient computation methods needed for training.
    Noise gradient is always computed by BBMM (dK/d(noise) = I), so it's not
    part of num_gradient_params().
    """
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute dK/dtheta_i @ v for parameter gradient.
        
        Args:
            out_ptr: Output buffer [n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of columns
            param_index: Which parameter (0 to num_gradient_params()-1)
            sync: Whether to synchronize after kernel launch
        """
        ...
    
    fn num_gradient_params(self) -> Int:
        """Number of kernel hyperparameters (excluding noise).
        
        Noise gradient is always computed by BBMM (dK/d(noise) = I).
        
        Examples:
        - Isotropic RBF: 2 (lengthscale, outputscale)
        - ARD RBF with d dims: d+1 (d lengthscales + outputscale)
        - Composite kernel: K.num_params()
        """
        ...
    
    fn supports_fused_gradient(self) -> Bool:
        """Whether this provider supports fused gradient computation.
        
        When True, fused_gradient_matvec() computes ALL gradient matvecs in a single
        kernel launch, avoiding redundant kernel evaluations. Supported for all ARD
        kernel types: RBF, Matern, Periodic, RQ, Linear, and Polynomial.
        
        Returns:
            True if fused_gradient_matvec() is available and efficient.
        """
        ...
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs in a single fused kernel launch.
        
        Output layout: out_ptr[p * n * num_cols + col * n + row] for parameter p.
        Total output size: num_gradient_params() * n * num_cols.
        
        Only call this when supports_fused_gradient() returns True.
        
        Args:
            out_ptr: Output buffer [num_gradient_params * n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of input columns
        """
        ...
    
    fn supports_fused_ls_os(self) -> Bool:
        """Whether this provider supports fused ls+os gradient computation.
        
        When True, fused_ls_os_gradient_matvec() computes BOTH the lengthscale
        gradient matvec and outputscale gradient matvec (K@V) in a single O(n²) pass
        using shared memory. 1.5-1.7x faster than 2 separate shmem launches.
        
        Supported for isotropic RBF and Matern kernels (2-param kernels).
        """
        ...
    
    fn fused_ls_os_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute BOTH dK/dl@V and K@V in a single fused shmem kernel launch.
        
        Only call when supports_fused_ls_os() returns True.
        
        Args:
            ls_out_ptr: Output for ls gradient [n * num_cols]
            os_out_ptr: Output for os gradient (K@V) [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
        """
        ...
    
    fn supports_fused_3param(self) -> Bool:
        """Whether this provider supports fused 3-param gradient computation.
        
        When True, fused_3param_gradient_matvec() computes ALL three gradient
        matvecs (ls + param1 + os) in a single O(n²) pass using shared memory.
        1.7x faster than 3 separate shmem launches.
        
        Supported for isotropic Periodic and RQ kernels (3-param kernels).
        """
        ...
    
    fn fused_3param_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL 3 gradient matvecs (ls, param1, os) in one fused kernel.
        
        Only call when supports_fused_3param() returns True.
        
        Args:
            ls_out_ptr: Output for ls gradient [n * num_cols]
            p1_out_ptr: Output for param1 gradient [n * num_cols]
            os_out_ptr: Output for os gradient [n * num_cols]
            v_ptr: Input vectors [n * num_cols]
            num_cols: Number of columns
        """
        ...


# =============================================================================
# GPU Kernel: Fill with constant value
# =============================================================================

fn kernel_fill_constant_gp(
    dst_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
    value: Float32,
) -> None:
    """Fill buffer with constant value: dst[i] = value for all i."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(size):
        return
    
    dst_ptr[idx] = value


# =============================================================================
# IsotropicGradientAdapter
# =============================================================================

struct IsotropicGradientAdapter[T: MatvecProvider & Movable](GradientProvider, Movable):
    """Adapts any MatvecProvider to GradientProvider with 2 gradient params.
    
    param_index mapping:
      0 -> lengthscale gradient (calls provider.gradient_matvec(..., 0))
      1 -> outputscale gradient (calls provider.gradient_matvec(..., 1))
    
    Noise gradient is handled by BBMM (universal: dK/d(noise) = I).
    """
    var provider: T
    
    fn __init__(out self, var provider: T):
        """Create adapter from a MatvecProvider.
        
        Args:
            provider: The underlying MatvecProvider (ownership transferred)
        """
        self.provider = provider^
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Delegate to underlying provider."""
        self.provider.forward_matvec(out_ptr, v_ptr, num_cols)
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Delegate to underlying provider.
        
        param_index 0 = lengthscale, 1 = outputscale, 2 = param1, 3 = param2
        
        Short-circuits known zero-gradient parameters for Linear/Polynomial kernels
        to avoid launching O(n^2) GPU kernels that would compute all zeros:
        - Linear: param 0 (lengthscale) is always zero
        - Polynomial: param 0 (lengthscale) and param 2 (degree, frozen) are always zero
        """
        from .constants import KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
        var kernel_type = self.provider.get_kernel_type()
        
        # Skip known zero-gradient params: fill output with zeros instead of
        # launching an O(n^2) GPU kernel that computes all zeros.
        var is_zero = False
        if kernel_type == KERNEL_TYPE_LINEAR and param_index == 0:
            is_zero = True  # Linear has no lengthscale
        elif kernel_type == KERNEL_TYPE_POLYNOMIAL and (param_index == 0 or param_index == 2):
            is_zero = True  # Polynomial has no lengthscale; degree is frozen
        
        if is_zero:
            var total_size = self.provider.get_n() * num_cols
            var ctx = self.provider.get_ctx()
            var threads = 256
            var blocks = (total_size + threads - 1) // threads
            ctx.enqueue_function[kernel_fill_constant_gp](
                out_ptr, total_size, Float32(0.0),
                grid_dim=blocks, block_dim=threads,
            )
            if sync:
                ctx.synchronize()
            return
        
        self.provider.gradient_matvec(out_ptr, v_ptr, num_cols, param_index, sync)
    
    fn num_gradient_params(self) -> Int:
        """Return number of gradient parameters based on kernel type.
        
        - RBF, Matern: 2 (lengthscale, outputscale)
        - Periodic, RQ, Linear: 3 (lengthscale, outputscale, param1)
        - Polynomial: 4 (lengthscale, outputscale, param1_dummy, param2_offset)
        
        For Polynomial, param_index=2 maps to param1 (degree, frozen — gradient is 0),
        and param_index=3 maps to param2 (offset, learnable).
        """
        from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
        var kernel_type = self.provider.get_kernel_type()
        if kernel_type == KERNEL_TYPE_POLYNOMIAL:
            return 4  # ls, os, param1 (degree, frozen), param2 (offset)
        if kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ or kernel_type == KERNEL_TYPE_LINEAR:
            return 3
        return 2
    
    fn get_n(self) -> Int:
        return self.provider.get_n()
    
    fn get_ctx(self) -> DeviceContext:
        return self.provider.get_ctx()
    
    fn get_noise(self) -> Float32:
        return self.provider.get_noise()
    
    fn get_diagonal_value(self) -> Float32:
        """For stationary kernels, diagonal = outputscale."""
        return self.provider.get_outputscale()
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Stationary kernels: diagonal is constant = outputscale."""
        self.provider.extract_diagonal(diag_ptr)
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.provider.get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """Isotropic kernels don't benefit from fused gradients (only 2-3 params)."""
        return False
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Not supported for isotropic kernels."""
        raise Error("fused_gradient_matvec not supported for IsotropicGradientAdapter")
    
    fn supports_fused_ls_os(self) -> Bool:
        """Isotropic RBF/Matern support fused ls+os (2-param kernels)."""
        from .constants import KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32, KERNEL_TYPE_MATERN52
        var kt = self.provider.get_kernel_type()
        return kt == KERNEL_TYPE_RBF or kt == KERNEL_TYPE_MATERN12 or kt == KERNEL_TYPE_MATERN32 or kt == KERNEL_TYPE_MATERN52
    
    fn fused_ls_os_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Fused ls+os gradient: compute both dK/dl@V and K@V in one pass."""
        from .dispatchers import dispatch_fused_ls_os_gradient_matvec_multicol
        from .kernel_params import make_rbf_params, make_matern_params
        from .constants import KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32, KERNEL_TYPE_MATERN52
        
        var kt = self.provider.get_kernel_type()
        var ls_ptr = self.provider.get_lengthscales_device_ptr()
        var use_ard = self.provider.get_use_ard()
        var params: KernelParams
        if kt == KERNEL_TYPE_RBF:
            params = make_rbf_params(self.provider.get_outputscale(), self.provider.get_lengthscale(), ls_ptr, is_ard=use_ard)
        elif kt == KERNEL_TYPE_MATERN12:
            params = make_matern_params(self.provider.get_outputscale(), self.provider.get_lengthscale(), Float32(0.5), ls_ptr, is_ard=use_ard)
        elif kt == KERNEL_TYPE_MATERN32:
            params = make_matern_params(self.provider.get_outputscale(), self.provider.get_lengthscale(), Float32(1.5), ls_ptr, is_ard=use_ard)
        elif kt == KERNEL_TYPE_MATERN52:
            params = make_matern_params(self.provider.get_outputscale(), self.provider.get_lengthscale(), Float32(2.5), ls_ptr, is_ard=use_ard)
        else:
            raise Error("Fused ls+os not supported for kernel type: " + String(kt))
        
        dispatch_fused_ls_os_gradient_matvec_multicol(
            self.provider.get_ctx(),
            kt,
            ls_out_ptr,
            os_out_ptr,
            self.provider.get_x_ptr(),
            v_ptr,
            self.provider.get_n(),
            self.provider.get_d(),
            num_cols,
            params,
            -1,  # isotropic
        )
    
    fn supports_fused_3param(self) -> Bool:
        """Isotropic Periodic/RQ support fused 3-param (ls + param1 + os)."""
        from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ
        var kt = self.provider.get_kernel_type()
        return kt == KERNEL_TYPE_PERIODIC or kt == KERNEL_TYPE_RQ
    
    fn fused_3param_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Fused 3-param gradient: ls + param1 + os in one O(n²) pass."""
        from .dispatchers import dispatch_fused_3param_gradient_matvec_multicol
        from .kernel_params import make_periodic_params, make_rq_params
        from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ
        
        var kt = self.provider.get_kernel_type()
        var ls_ptr = self.provider.get_lengthscales_device_ptr()
        var use_ard = self.provider.get_use_ard()
        var params: KernelParams
        if kt == KERNEL_TYPE_PERIODIC:
            params = make_periodic_params(
                self.provider.get_outputscale(), self.provider.get_lengthscale(),
                self.provider.get_kernel_param1(), ls_ptr, is_ard=use_ard)
        elif kt == KERNEL_TYPE_RQ:
            params = make_rq_params(
                self.provider.get_outputscale(), self.provider.get_lengthscale(),
                self.provider.get_kernel_param1(), ls_ptr, is_ard=use_ard)
        else:
            raise Error("Fused 3-param not supported for kernel type: " + String(kt))
        
        dispatch_fused_3param_gradient_matvec_multicol(
            self.provider.get_ctx(),
            kt,
            ls_out_ptr,
            p1_out_ptr,
            os_out_ptr,
            self.provider.get_x_ptr(),
            v_ptr,
            self.provider.get_n(),
            self.provider.get_d(),
            num_cols,
            params,
            -1,  # isotropic
        )
    
    fn update_hyperparams(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
    ) raises:
        """Update hyperparameters on the underlying provider.
        
        Args:
            lengthscale: New lengthscale value
            outputscale: New outputscale value
            noise: New noise variance
        """
        self.provider.update_hyperparams(lengthscale, outputscale, noise)
    
    fn update_hyperparams_with_param1(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
        param1: Float32,
    ) raises:
        """Update hyperparameters including param1 on the underlying provider.
        
        Args:
            lengthscale: New lengthscale value
            outputscale: New outputscale value
            noise: New noise variance
            param1: New param1 value (period for Periodic, alpha for RQ)
        """
        self.provider.update_hyperparams_with_param1(lengthscale, outputscale, noise, param1)
    
    fn update_param2(
        mut self,
        param2: Float32,
    ) raises:
        """Update param2 (offset for Polynomial) on the underlying provider.
        
        Args:
            param2: New param2 value
        """
        self.provider.update_param2(param2)


# =============================================================================
# ARDGradientAdapter
# =============================================================================

struct ARDGradientAdapter[T: MatvecProvider & Movable](GradientProvider, Movable):
    """Adapts a MatvecProvider (with ARD) to GradientProvider with d+1 gradient params.
    
    param_index mapping:
      0..d-1 -> per-dimension lengthscale gradients
      d      -> outputscale gradient
    
    Requires the underlying provider to support gradient_matvec_ard().
    This method is part of the MatvecProvider trait.
    """
    var provider: T
    var d: Int  # input dimension
    
    fn __init__(out self, var provider: T, d: Int):
        """Create ARD adapter from a MatvecProvider.
        
        Args:
            provider: The underlying MatvecProvider (ownership transferred)
            d: Input dimension (number of features)
        """
        self.provider = provider^
        self.d = d
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Delegate to underlying provider."""
        self.provider.forward_matvec(out_ptr, v_ptr, num_cols)
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute gradient matvec for ARD parameters.
        
        param_index 0..d-1 = per-dimension lengthscale gradients
        param_index d = outputscale gradient
        param_index d+1 = param1 gradient (period/alpha/variance) for Periodic/RQ/Linear
        param_index d+2 = param2 gradient (offset) for Polynomial
        """
        var num_base_params = self.d + 1  # d lengthscales + outputscale
        
        if param_index < self.d:
            # Per-dimension lengthscale gradient
            self.provider.gradient_matvec_ard(out_ptr, v_ptr, num_cols, param_index, sync)
        elif param_index == self.d:
            # Outputscale gradient (param_index == d)
            # In MatvecProvider, param_index 1 = outputscale
            self.provider.gradient_matvec(out_ptr, v_ptr, num_cols, 1, sync)
        elif param_index == self.d + 1:
            # Param1 gradient (period/alpha/variance) for Periodic/RQ/Linear kernels
            # In MatvecProvider, param_index 2 = param1
            self.provider.gradient_matvec(out_ptr, v_ptr, num_cols, 2, sync)
        elif param_index == self.d + 2:
            # Param2 gradient (offset) for Polynomial kernel
            # In MatvecProvider, param_index 3 = param2
            self.provider.gradient_matvec(out_ptr, v_ptr, num_cols, 3, sync)
        else:
            raise Error("Invalid param_index for ARDGradientAdapter: " + String(param_index))
    
    fn num_gradient_params(self) -> Int:
        """ARD kernels have d+1, d+2, or d+3 params depending on kernel type.
        
        - RBF, Matern, Linear: d per-dim params + outputscale = d+1
        - Periodic, RQ: d lengthscales + outputscale + param1 = d+2
        - Polynomial: d lengthscales + outputscale + param1(degree) + param2(offset) = d+3
        
        Note: Linear ARD has d per-dimension variance weights + outputscale = d+1.
        The scalar variance (param1) is replaced by per-dim weights, not added to them.
        
        For Polynomial, param1 (degree) gradient is zero (frozen), but we still
        include it in the gradient layout for consistency.
        """
        from .constants import KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_POLYNOMIAL
        var kernel_type = self.provider.get_kernel_type()
        if kernel_type == KERNEL_TYPE_POLYNOMIAL:
            return self.d + 3  # Include param1 (degree, frozen) + param2 (offset)
        if kernel_type == KERNEL_TYPE_PERIODIC or kernel_type == KERNEL_TYPE_RQ:
            return self.d + 2  # Include param1 (period/alpha)
        # RBF, Matern, Linear: d + 1
        return self.d + 1
    
    fn get_n(self) -> Int:
        return self.provider.get_n()
    
    fn get_ctx(self) -> DeviceContext:
        return self.provider.get_ctx()
    
    fn get_noise(self) -> Float32:
        return self.provider.get_noise()
    
    fn get_diagonal_value(self) -> Float32:
        """For stationary kernels, diagonal = outputscale."""
        return self.provider.get_outputscale()
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Stationary kernels: diagonal is constant = outputscale."""
        self.provider.extract_diagonal(diag_ptr)
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.provider.get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """ARD kernels with fused gradient implementations.
        
        RBF, Matern, Linear: d+1 params (d per-dim params + outputscale)
        Periodic, RQ: d+2 params (d lengthscales + param1 + outputscale)
        Polynomial: d+3 params (d lengthscales + degree + offset + outputscale)
        """
        from .constants import KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32, KERNEL_TYPE_MATERN52, KERNEL_TYPE_PERIODIC, KERNEL_TYPE_RQ, KERNEL_TYPE_LINEAR, KERNEL_TYPE_POLYNOMIAL
        var kt = self.provider.get_kernel_type()
        return (kt == KERNEL_TYPE_RBF or kt == KERNEL_TYPE_MATERN12 
                or kt == KERNEL_TYPE_MATERN32 or kt == KERNEL_TYPE_MATERN52
                or kt == KERNEL_TYPE_PERIODIC or kt == KERNEL_TYPE_RQ
                or kt == KERNEL_TYPE_LINEAR or kt == KERNEL_TYPE_POLYNOMIAL)
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL d+1 gradient matvecs in a single fused kernel launch.
        
        Output layout: out_ptr[p * n * num_cols + col * n + row]
        where p=0..d-1 are lengthscale grads and p=d is outputscale grad.
        
        Only valid when supports_fused_gradient() returns True.
        """
        from .dispatchers import dispatch_fused_gradient_matvec
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
        
        var kernel_type = self.provider.get_kernel_type()
        var n = self.provider.get_n()
        var d = self.d
        var x_ptr = self.provider.get_x_ptr()
        var outputscale = self.provider.get_outputscale()
        var lengthscale = self.provider.get_lengthscale()
        var ls_ptr = self.provider.get_lengthscales_device_ptr()
        var ctx = self.provider.get_ctx()
        
        # Build KernelParams with ARD parameters
        var params: KernelParams
        if kernel_type == KERNEL_TYPE_RBF:
            params = make_rbf_params(outputscale, lengthscale, ls_ptr, is_ard=True)
        elif kernel_type == KERNEL_TYPE_MATERN12:
            params = make_matern_params(outputscale, lengthscale, Float32(0.5), ls_ptr, is_ard=True)
        elif kernel_type == KERNEL_TYPE_MATERN32:
            params = make_matern_params(outputscale, lengthscale, Float32(1.5), ls_ptr, is_ard=True)
        elif kernel_type == KERNEL_TYPE_MATERN52:
            params = make_matern_params(outputscale, lengthscale, Float32(2.5), ls_ptr, is_ard=True)
        elif kernel_type == KERNEL_TYPE_PERIODIC:
            var period = self.provider.get_kernel_param1()
            params = make_periodic_params(outputscale, lengthscale, period, ls_ptr, is_ard=True)
        elif kernel_type == KERNEL_TYPE_RQ:
            var alpha = self.provider.get_kernel_param1()
            params = make_rq_params(outputscale, lengthscale, alpha, ls_ptr, is_ard=True)
        elif kernel_type == KERNEL_TYPE_LINEAR:
            # ARD: per-dim variance weights in ls_ptr, no scalar variance
            params = make_linear_params(outputscale, Float32(1.0), ls_ptr, is_ard=True)
        elif kernel_type == KERNEL_TYPE_POLYNOMIAL:
            var degree = self.provider.get_kernel_param1()
            var offset = self.provider.get_kernel_param2()
            params = make_polynomial_params(outputscale, degree, offset, ls_ptr, is_ard=True)
        else:
            raise Error("Fused gradient not supported for kernel type: " + String(kernel_type))
        
        dispatch_fused_gradient_matvec(
            ctx, kernel_type, out_ptr, x_ptr, v_ptr, n, d, num_cols, params
        )
    
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
        """ARD kernels use the all-params fused path instead."""
        return False
    
    fn fused_3param_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        raise Error("fused_3param not supported for ARD — use fused_gradient_matvec instead")
    
    fn update_hyperparams_ard(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
        outputscale: Float32,
        noise: Float32,
    ) raises:
        """Update ARD hyperparameters on the underlying provider.
        
        Args:
            lengthscales_host: Pointer to [d] lengthscales on HOST
            outputscale: New outputscale value
            noise: New noise variance
        """
        self.provider.update_hyperparams_ard(lengthscales_host, outputscale, noise)
    
    fn update_param2(
        mut self,
        param2: Float32,
    ) raises:
        """Update param2 (offset for Polynomial) on the underlying provider.
        
        Args:
            param2: New param2 value
        """
        self.provider.update_param2(param2)


# =============================================================================
# CompositeGradientAdapter
# =============================================================================

struct CompositeGradientAdapter[DIM: Int, K: ComposableKernel, EXTRA_NCOLS: Int = 0](GradientProvider, Movable):
    """Adapts a CompositeProvider to GradientProvider with N gradient params.
    
    param_index mapping:
      0..N-1 -> kernel parameter gradients (as defined by K.num_params())
    """
    var provider: CompositeProvider[DIM, K, EXTRA_NCOLS]
    
    fn __init__(out self, var provider: CompositeProvider[DIM, K, EXTRA_NCOLS]):
        """Create adapter from a CompositeProvider.
        
        Args:
            provider: The underlying CompositeProvider (ownership transferred)
        """
        self.provider = provider^
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Delegate to underlying provider."""
        self.provider.forward_matvec(out_ptr, v_ptr, num_cols)
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Delegate to underlying provider."""
        self.provider.gradient_matvec(out_ptr, v_ptr, num_cols, param_index, sync)
    
    fn num_gradient_params(self) -> Int:
        """Composite kernels have K.num_params() gradient parameters."""
        return K.num_params()
    
    fn get_n(self) -> Int:
        return self.provider.get_n()
    
    fn get_ctx(self) -> DeviceContext:
        return self.provider.get_ctx()
    
    fn get_noise(self) -> Float32:
        return self.provider.get_noise()
    
    fn get_diagonal_value(self) -> Float32:
        """Composite kernels may not have a constant diagonal.
        
        Return a reasonable default; extract_diagonal provides the full vector.
        """
        return Float32(1.0)
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Delegate to underlying provider."""
        self.provider.extract_diagonal(diag_ptr)
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.provider.get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """Composite kernels support fused gradients via K.all_gradients()."""
        return True
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs in a single fused kernel launch.
        
        Delegates to CompositeProvider.fused_gradient_matvec() which uses
        composite_fused_gradient_matvec_4x[DIM, K] GPU kernel.
        """
        self.provider.fused_gradient_matvec(out_ptr, v_ptr, num_cols)
    
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
        raise Error("fused_3param not supported for composite kernels")
    
    fn update_params(
        mut self,
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Update kernel parameters on the underlying provider.
        
        Args:
            params_host_ptr: New parameters on host [K.num_params()]
        """
        self.provider.update_params(params_host_ptr)
    
    fn update_noise(mut self, noise: Float32):
        """Update noise variance on the underlying provider."""
        self.provider.update_noise(noise)


# =============================================================================
# MaterializedCompositeGradientAdapter
# =============================================================================

struct MaterializedCompositeGradientAdapter[DIM: Int, K: ComposableKernel](GradientProvider, Movable):
    """Adapts a MaterializedCompositeProvider to GradientProvider.
    
    param_index mapping:
      0..N-1 -> kernel parameter gradients (as defined by K.num_params())
    """
    var provider: MaterializedCompositeProvider[DIM, K]
    
    fn __init__(out self, var provider: MaterializedCompositeProvider[DIM, K]):
        """Create adapter from a MaterializedCompositeProvider.
        
        Args:
            provider: The underlying MaterializedCompositeProvider (ownership transferred)
        """
        self.provider = provider^
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Delegate to underlying provider."""
        self.provider.forward_matvec(out_ptr, v_ptr, num_cols)
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Delegate to underlying provider."""
        self.provider.gradient_matvec(out_ptr, v_ptr, num_cols, param_index, sync)
    
    fn num_gradient_params(self) -> Int:
        """Composite kernels have K.num_params() gradient parameters."""
        return K.num_params()
    
    fn get_n(self) -> Int:
        return self.provider.get_n()
    
    fn get_ctx(self) -> DeviceContext:
        return self.provider.get_ctx()
    
    fn get_noise(self) -> Float32:
        return self.provider.get_noise()
    
    fn get_diagonal_value(self) -> Float32:
        """Composite kernels may not have a constant diagonal.
        
        Return a reasonable default; extract_diagonal provides the full vector.
        """
        return Float32(1.0)
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Delegate to underlying provider (uses cached diagonal)."""
        self.provider.extract_diagonal(diag_ptr)
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self.provider.get_x_ptr()
    
    fn supports_fused_gradient(self) -> Bool:
        """Materialized composite kernels support fused gradients via K.all_gradients()."""
        return True
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs in a single fused kernel launch.
        
        Delegates to MaterializedCompositeProvider.fused_gradient_matvec() which uses
        composite_fused_gradient_matvec_4x[DIM, K] GPU kernel.
        """
        self.provider.fused_gradient_matvec(out_ptr, v_ptr, num_cols)
    
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
        raise Error("fused_3param not supported for materialized composite kernels")
    
    fn update_params(
        mut self,
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Update kernel parameters on the underlying provider.
        
        Args:
            params_host_ptr: New parameters on host [K.num_params()]
        """
        self.provider.update_params(params_host_ptr)
    
    fn update_noise(mut self, noise: Float32):
        """Update noise variance on the underlying provider."""
        self.provider.update_noise(noise)
