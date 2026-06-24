"""CompositeProvider for composite kernel matrix-vector operations.

This module provides a MatvecProvider implementation for composite kernels,
enabling them to be used with the existing CG/BBMM infrastructure.

The key difference from MatrixFreeProvider:
- Parameterized by ComposableKernel type at compile time
- Uses flat parameter array instead of KernelParams struct
- Supports arbitrary kernel compositions

Example usage:
    alias MyKernel = SumKernel[RBFComposable, LinearComposable]
    var provider = CompositeProvider[5, MyKernel](ctx, x_ptr, params_ptr, n, noise)
    provider.forward_matvec(out_ptr, v_ptr, num_cols)
"""

from gpu.host import DeviceContext, DeviceBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer

from .composable_kernel import ComposableKernel
from .matvec_provider import MatvecProvider
from .composite_matvec import (
    composite_forward_matvec_8x,
    composite_forward_matvec_multicol,
    composite_gradient_matvec_4x,
    composite_gradient_matvec_batched_4x,
    composite_gradient_matvec_single_param_4x,
    composite_gradient_matvec_single_param_batched_4x,
    composite_cross_matvec_8x,
    composite_extract_diagonal,
)

alias _MULTICOL_NCOLS = 11
alias _MULTICOL_NCOLS_6 = 6
from .constants import float_dtype


# =============================================================================
# GPU Kernel for diagonal copy
# =============================================================================

fn _kernel_copy_diag(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """Copy n elements from src to dst."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < n:
        dst[idx] = src[idx]


# =============================================================================
# CompositeProvider
# =============================================================================

struct CompositeProvider[DIM: Int, K: ComposableKernel, EXTRA_NCOLS: Int = 0](Movable):
    """Provider for composite kernel matrix-vector operations.
    
    This provider is parameterized by:
    - DIM: Input dimension (compile-time constant)
    - K: ComposableKernel type (e.g., SumKernel[RBF, Linear])
    - EXTRA_NCOLS: Additional compile-time NCOLS specialization (default 0 = none).
      Used for multi-output where num_cols=T (num_tasks) is known at JIT time.
    
    Parameters are stored in a flat array on device memory.
    The layout is determined by the kernel composition.
    """
    var ctx: DeviceContext
    var x_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var params_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var n: Int
    var noise: Float32
    
    # Device buffers (owned by this provider)
    var _x_device: DeviceBuffer[DType.float32]
    var _params_device: DeviceBuffer[DType.float32]
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        noise: Float32,
    ) raises:
        """Create a CompositeProvider.
        
        Args:
            ctx: GPU device context
            x_host_ptr: Training data on host [n, DIM] row-major
            params_host_ptr: Kernel parameters on host [K.num_params()]
            n: Number of data points
            noise: Noise variance σ²
        """
        self.ctx = ctx
        self.n = n
        self.noise = noise
        
        # Allocate device buffers
        self._x_device = ctx.enqueue_create_buffer[DType.float32](n * DIM)
        self._params_device = ctx.enqueue_create_buffer[DType.float32](K.num_params())
        
        # Copy data to device
        # Note: We need to create host buffers for the copy
        var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * DIM)
        var params_host = ctx.enqueue_create_host_buffer[DType.float32](K.num_params())
        
        # Copy from input pointers to host buffers
        for i in range(n * DIM):
            x_host.unsafe_ptr()[i] = x_host_ptr[i]
        for i in range(K.num_params()):
            params_host.unsafe_ptr()[i] = params_host_ptr[i]
        
        # Copy to device
        ctx.enqueue_copy(dst_buf=self._x_device, src_buf=x_host)
        ctx.enqueue_copy(dst_buf=self._params_device, src_buf=params_host)
        ctx.synchronize()
        
        # Store device pointers
        self.x_ptr = self._x_device.unsafe_ptr()
        self.params_ptr = self._params_device.unsafe_ptr()
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (K + noise*I) @ v.
        
        Args:
            out_ptr: Output buffer [n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of columns
        """
        var threads_per_block = 256
        var num_blocks = (self.n + threads_per_block - 1) // threads_per_block
        
        if num_cols == _MULTICOL_NCOLS:
            self.ctx.enqueue_function[composite_forward_matvec_multicol[DIM, _MULTICOL_NCOLS, K]](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                self.noise,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
                shared_mem_bytes=threads_per_block * (DIM + _MULTICOL_NCOLS) * 4,
            )
        elif num_cols == _MULTICOL_NCOLS_6:
            self.ctx.enqueue_function[composite_forward_matvec_multicol[DIM, _MULTICOL_NCOLS_6, K]](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                self.noise,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
                shared_mem_bytes=threads_per_block * (DIM + _MULTICOL_NCOLS_6) * 4,
            )
        else:
            # EXTRA_NCOLS specialization for multi-output (num_cols=T baked in at JIT time)
            @parameter
            if EXTRA_NCOLS > 0 and EXTRA_NCOLS != _MULTICOL_NCOLS and EXTRA_NCOLS != _MULTICOL_NCOLS_6:
                if num_cols == EXTRA_NCOLS:
                    self.ctx.enqueue_function[composite_forward_matvec_multicol[DIM, EXTRA_NCOLS, K]](
                        out_ptr,
                        self.x_ptr,
                        v_ptr,
                        self.params_ptr,
                        self.n,
                        self.noise,
                        grid_dim=num_blocks,
                        block_dim=threads_per_block,
                        shared_mem_bytes=threads_per_block * (DIM + EXTRA_NCOLS) * 4,
                    )
                    self.ctx.synchronize()
                    return
            self.ctx.enqueue_function[composite_forward_matvec_8x[DIM, K]](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                num_cols,
                self.noise,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
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
        """Compute (∂K/∂θ_p) @ v for a single parameter.
        
        Args:
            out_ptr: Output buffer [n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of columns
            param_index: Which parameter to differentiate (0 to num_params-1)
            sync: Whether to synchronize after kernel launch
        """
        var threads_per_block = 256
        var num_blocks = (self.n + threads_per_block - 1) // threads_per_block
        
        # Use single-parameter kernels that respect param_index
        if num_cols == 1:
            self.ctx.enqueue_function[composite_gradient_matvec_single_param_4x[DIM, K]](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                param_index,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            # For batched case, use the batched single-param kernel
            self.ctx.enqueue_function[composite_gradient_matvec_single_param_batched_4x[DIM, K]](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                num_cols,
                param_index,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        
        if sync:
            self.ctx.synchronize()
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs in a single fused kernel launch.
        
        Output layout: out_ptr[p * n * num_cols + col * n + row] for parameter p.
        Total output size: K.num_params() * n * num_cols.
        
        Uses shmem-tiled multicol kernel for known NCOLS values (2-3x faster),
        falls back to 4x-unrolled kernel for other column counts.
        
        Args:
            out_ptr: Output buffer [K.num_params() * n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of input columns
        """
        from .composite_matvec import (
            composite_fused_gradient_matvec_4x,
            composite_fused_gradient_matvec_shmem_multicol,
        )
        
        var threads_per_block = 256
        var num_blocks = (self.n + threads_per_block - 1) // threads_per_block
        
        if num_cols == _MULTICOL_NCOLS:
            self.ctx.enqueue_function[
                composite_fused_gradient_matvec_shmem_multicol[DIM, _MULTICOL_NCOLS, K]
            ](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
                shared_mem_bytes=threads_per_block * (DIM + _MULTICOL_NCOLS) * 4,
            )
        elif num_cols == _MULTICOL_NCOLS_6:
            self.ctx.enqueue_function[
                composite_fused_gradient_matvec_shmem_multicol[DIM, _MULTICOL_NCOLS_6, K]
            ](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
                shared_mem_bytes=threads_per_block * (DIM + _MULTICOL_NCOLS_6) * 4,
            )
        else:
            # EXTRA_NCOLS specialization for multi-output (num_cols=T baked in at JIT time)
            @parameter
            if EXTRA_NCOLS > 0 and EXTRA_NCOLS != _MULTICOL_NCOLS and EXTRA_NCOLS != _MULTICOL_NCOLS_6:
                if num_cols == EXTRA_NCOLS:
                    self.ctx.enqueue_function[
                        composite_fused_gradient_matvec_shmem_multicol[DIM, EXTRA_NCOLS, K]
                    ](
                        out_ptr,
                        self.x_ptr,
                        v_ptr,
                        self.params_ptr,
                        self.n,
                        grid_dim=num_blocks,
                        block_dim=threads_per_block,
                        shared_mem_bytes=threads_per_block * (DIM + EXTRA_NCOLS) * 4,
                    )
                    return
            self.ctx.enqueue_function[composite_fused_gradient_matvec_4x[DIM, K]](
                out_ptr,
                self.x_ptr,
                v_ptr,
                self.params_ptr,
                self.n,
                num_cols,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
    
    fn cross_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_test: Int,
        num_cols: Int,
    ) raises:
        """Compute K(X_test, X_train) @ v for prediction.
        
        Args:
            out_ptr: Output buffer [n_test * num_cols] on device
            x_test_ptr: Test data [n_test, DIM] on device
            v_ptr: Input vectors [n_train * num_cols] on device
            n_test: Number of test points
            num_cols: Number of columns
        """
        var threads_per_block = 256
        var num_blocks = (n_test + threads_per_block - 1) // threads_per_block
        
        self.ctx.enqueue_function[composite_cross_matvec_8x[DIM, K]](
            out_ptr,
            x_test_ptr,
            self.x_ptr,
            v_ptr,
            self.params_ptr,
            n_test,
            self.n,
            num_cols,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        self.ctx.synchronize()
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract diagonal of kernel matrix: diag[i] = K(x_i, x_i).
        
        Args:
            diag_ptr: Output buffer [n] on device
        """
        var threads_per_block = 256
        var num_blocks = (self.n + threads_per_block - 1) // threads_per_block
        
        self.ctx.enqueue_function[composite_extract_diagonal[DIM, K]](
            diag_ptr,
            self.x_ptr,
            self.params_ptr,
            self.n,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        self.ctx.synchronize()
    
    fn update_params(
        mut self,
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Update kernel parameters.
        
        Args:
            params_host_ptr: New parameters on host [K.num_params()]
        """
        var params_host = self.ctx.enqueue_create_host_buffer[DType.float32](K.num_params())
        for i in range(K.num_params()):
            params_host.unsafe_ptr()[i] = params_host_ptr[i]
        
        self.ctx.enqueue_copy(dst_buf=self._params_device, src_buf=params_host)
        self.ctx.synchronize()
    
    fn update_noise(mut self, noise: Float32):
        """Update noise variance."""
        self.noise = noise
    
    fn get_n(self) -> Int:
        """Return number of data points."""
        return self.n
    
    fn get_num_params(self) -> Int:
        """Return number of kernel parameters."""
        return K.num_params()
    
    fn get_noise(self) -> Float32:
        """Return noise variance."""
        return self.noise
    
    fn get_ctx(self) -> DeviceContext:
        """Return device context."""
        return self.ctx
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data on device."""
        return self.x_ptr
    
    fn get_params_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to parameters on device."""
        return self.params_ptr


# =============================================================================
# Helper: Create provider from host data
# =============================================================================

fn create_composite_provider[DIM: Int, K: ComposableKernel](
    ctx: DeviceContext,
    x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    noise: Float32,
) raises -> CompositeProvider[DIM, K]:
    """Create a CompositeProvider from host data.
    
    This is a convenience function that handles the type parameters.
    
    Args:
        ctx: GPU device context
        x_host_ptr: Training data on host [n, DIM]
        params_host_ptr: Kernel parameters on host [K.num_params()]
        n: Number of data points
        noise: Noise variance
    
    Returns:
        CompositeProvider ready for use
    """
    return CompositeProvider[DIM, K](ctx, x_host_ptr, params_host_ptr, n, noise)


# =============================================================================
# MaterializedCompositeProvider
# =============================================================================

from .kernel_materialization import (
    materialize_composite_kernel_matrix,
    materialize_composite_gradient_matrix,
    extract_diagonal_from_matrix,
)
from .gemm_matvec import gemm_matvec, add_noise_diagonal


struct MaterializedCompositeProvider[DIM: Int, K: ComposableKernel](MatvecProvider, Movable):
    """Provider that materializes the composite kernel matrix for fast GEMM-based matvecs.
    
    This provider computes the full n×n kernel matrix K once (per parameter update)
    and uses GEMM for forward matvecs. This is much faster than the matrix-free
    CompositeProvider for training, at the cost of O(n²) memory.
    
    Optionally materializes gradient matrices dK/dθ_p for additional speedup.
    
    Implements MatvecProvider so it can be wrapped in MaterializedCompositeGradientAdapter
    and used as the base provider in KroneckerDirectProvider for multi-output Kronecker CG training.
    
    Memory usage:
    - K only: n² × 4 bytes (e.g., 400MB for n=10000)
    - K + gradients: (1 + N) × n² × 4 bytes (e.g., 2GB for n=10000, N=4)
    
    Example usage:
        alias MyKernel = SumKernel[RBFComposable, LinearComposable]
        var provider = MaterializedCompositeProvider[5, MyKernel](
            ctx, x_ptr, params_ptr, n, noise, materialize_gradients=True
        )
        provider.forward_matvec(out_ptr, v_ptr, num_cols)  # Uses GEMM
    """
    var ctx: DeviceContext
    var n: Int
    var noise: Float32
    var materialize_gradients: Bool
    var is_materialized: Bool
    
    # Device buffers
    var _x_device: DeviceBuffer[DType.float32]       # [n × DIM]
    var _params_device: DeviceBuffer[DType.float32]  # [K.num_params()]
    var _K_device: DeviceBuffer[DType.float32]       # [n × n] materialized kernel matrix
    var _diag_device: DeviceBuffer[DType.float32]    # [n] diagonal cache
    var _dK_devices: List[DeviceBuffer[DType.float32]]  # [N] × [n × n] gradient matrices (optional)
    
    # Device pointers for fast access
    var x_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var params_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var K_ptr: UnsafePointer[Float32, MutAnyOrigin]
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        noise: Float32,
        materialize_gradients: Bool = False,
    ) raises:
        """Create a MaterializedCompositeProvider.
        
        Args:
            ctx: GPU device context
            x_host_ptr: Training data on host [n, DIM] row-major
            params_host_ptr: Kernel parameters on host [K.num_params()]
            n: Number of data points
            noise: Noise variance σ²
            materialize_gradients: If True, also materialize dK/dθ matrices
        """
        self.ctx = ctx
        self.n = n
        self.noise = noise
        self.materialize_gradients = materialize_gradients
        self.is_materialized = False
        
        # Allocate device buffers for X, params
        self._x_device = ctx.enqueue_create_buffer[DType.float32](n * DIM)
        self._params_device = ctx.enqueue_create_buffer[DType.float32](K.num_params())
        
        # Allocate K matrix buffer (n × n)
        self._K_device = ctx.enqueue_create_buffer[DType.float32](n * n)
        
        # Allocate diagonal cache
        self._diag_device = ctx.enqueue_create_buffer[DType.float32](n)
        
        # Allocate gradient matrix buffers if requested
        self._dK_devices = List[DeviceBuffer[DType.float32]]()
        if materialize_gradients:
            for _ in range(K.num_params()):
                self._dK_devices.append(ctx.enqueue_create_buffer[DType.float32](n * n))
        
        # Copy X and params to device
        var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * DIM)
        var params_host = ctx.enqueue_create_host_buffer[DType.float32](K.num_params())
        
        for i in range(n * DIM):
            x_host.unsafe_ptr()[i] = x_host_ptr[i]
        for i in range(K.num_params()):
            params_host.unsafe_ptr()[i] = params_host_ptr[i]
        
        ctx.enqueue_copy(dst_buf=self._x_device, src_buf=x_host)
        ctx.enqueue_copy(dst_buf=self._params_device, src_buf=params_host)
        ctx.synchronize()
        
        # Store device pointers
        self.x_ptr = self._x_device.unsafe_ptr()
        self.params_ptr = self._params_device.unsafe_ptr()
        self.K_ptr = self._K_device.unsafe_ptr()
        
        # Materialize K (and optionally gradients)
        self._materialize()
    
    fn _materialize(mut self) raises:
        """Materialize the kernel matrix K (and optionally gradient matrices)."""
        # Materialize K
        materialize_composite_kernel_matrix[DIM, K](
            self.ctx, self._K_device, self.x_ptr, self.params_ptr, self.n
        )
        
        # Extract diagonal for preconditioner
        extract_diagonal_from_matrix(
            self.ctx, self._diag_device, self._K_device, self.n
        )
        
        # Materialize gradient matrices if requested
        if self.materialize_gradients:
            for p in range(K.num_params()):
                materialize_composite_gradient_matrix[DIM, K](
                    self.ctx, self._dK_devices[p], self.x_ptr, self.params_ptr, self.n, p
                )
        
        self.is_materialized = True
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute (K + noise*I) @ v using GEMM.
        
        Uses gemm_matvec() for K @ v, then add_noise_diagonal() for the noise term.
        
        Args:
            out_ptr: Output buffer [n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of columns
        """
        # GEMM: out = K @ v
        gemm_matvec(self.ctx, out_ptr, self.K_ptr, v_ptr, self.n, num_cols)
        
        # Add noise: out += noise * v
        add_noise_diagonal(self.ctx, out_ptr, v_ptr, self.n, num_cols, self.noise)
        
        self.ctx.synchronize()
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        """Compute (dK/dθ_p) @ v.
        
        If gradient matrices are materialized: uses GEMM (fast)
        Otherwise: falls back to matrix-free composite gradient matvec
        
        Args:
            out_ptr: Output buffer [n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of columns
            param_index: Which parameter to differentiate (0 to num_params-1)
            sync: Whether to synchronize after kernel launch
        """
        if self.materialize_gradients:
            # Use GEMM with materialized gradient matrix
            var dK_ptr = self._dK_devices[param_index].unsafe_ptr()
            gemm_matvec(self.ctx, out_ptr, dK_ptr, v_ptr, self.n, num_cols)
        else:
            # Fall back to matrix-free gradient computation
            var threads_per_block = 256
            var num_blocks = (self.n + threads_per_block - 1) // threads_per_block
            
            if num_cols == 1:
                self.ctx.enqueue_function[composite_gradient_matvec_single_param_4x[DIM, K]](
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.params_ptr,
                    self.n,
                    param_index,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                self.ctx.enqueue_function[composite_gradient_matvec_single_param_batched_4x[DIM, K]](
                    out_ptr,
                    self.x_ptr,
                    v_ptr,
                    self.params_ptr,
                    self.n,
                    num_cols,
                    param_index,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
        
        if sync:
            self.ctx.synchronize()
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        """Compute ALL gradient matvecs in a single fused kernel launch.
        
        Uses the matrix-free fused GPU kernel regardless of materialize_gradients
        setting, since the fused kernel avoids redundant kernel evaluations.
        
        Output layout: out_ptr[p * n * num_cols + col * n + row] for parameter p.
        Total output size: K.num_params() * n * num_cols.
        
        Args:
            out_ptr: Output buffer [K.num_params() * n * num_cols] on device
            v_ptr: Input vectors [n * num_cols] on device
            num_cols: Number of input columns
        """
        from .composite_matvec import composite_fused_gradient_matvec_4x
        
        var threads_per_block = 256
        var num_blocks = (self.n + threads_per_block - 1) // threads_per_block
        
        self.ctx.enqueue_function[composite_fused_gradient_matvec_4x[DIM, K]](
            out_ptr,
            self.x_ptr,
            v_ptr,
            self.params_ptr,
            self.n,
            num_cols,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
    
    fn cross_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_test: Int,
        num_cols: Int,
    ) raises:
        """Compute K(X_test, X_train) @ v for prediction.
        
        Cross-covariance is NOT materialized (would need n_test × n matrix).
        Falls back to matrix-free computation.
        
        Args:
            out_ptr: Output buffer [n_test * num_cols] on device
            x_test_ptr: Test data [n_test, DIM] on device
            v_ptr: Input vectors [n_train * num_cols] on device
            n_test: Number of test points
            num_cols: Number of columns
        """
        var threads_per_block = 256
        var num_blocks = (n_test + threads_per_block - 1) // threads_per_block
        
        self.ctx.enqueue_function[composite_cross_matvec_8x[DIM, K]](
            out_ptr,
            x_test_ptr,
            self.x_ptr,
            v_ptr,
            self.params_ptr,
            n_test,
            self.n,
            num_cols,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        self.ctx.synchronize()
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Extract diagonal from materialized K (fast, just copy from cache).
        
        Args:
            diag_ptr: Output buffer [n] on device
        """
        # Copy from cached diagonal
        var threads_per_block = 256
        var num_blocks = (self.n + threads_per_block - 1) // threads_per_block
        
        # Simple copy kernel
        fn kernel_copy_diag(
            dst: UnsafePointer[Float32, MutAnyOrigin],
            src: UnsafePointer[Float32, MutAnyOrigin],
            n: Int,
        ) -> None:
            var i = block_idx.x * block_dim.x + thread_idx.x
            if i < UInt(n):
                dst[i] = src[i]
        
        self.ctx.enqueue_function[kernel_copy_diag](
            diag_ptr, self._diag_device.unsafe_ptr(), self.n,
            grid_dim=num_blocks, block_dim=threads_per_block,
        )
        self.ctx.synchronize()
    
    fn update_params(
        mut self,
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Update kernel parameters and re-materialize K (and optionally gradients).
        
        Args:
            params_host_ptr: New parameters on host [K.num_params()]
        """
        # Copy new params to device
        var params_host = self.ctx.enqueue_create_host_buffer[DType.float32](K.num_params())
        for i in range(K.num_params()):
            params_host.unsafe_ptr()[i] = params_host_ptr[i]
        
        self.ctx.enqueue_copy(dst_buf=self._params_device, src_buf=params_host)
        self.ctx.synchronize()
        
        # Re-materialize K (and gradients if enabled)
        self._materialize()
    
    fn update_noise(mut self, noise: Float32):
        """Update noise variance (no re-materialization needed).
        
        Noise is not baked into K, just update the field.
        """
        self.noise = noise
    
    fn get_n(self) -> Int:
        """Return number of data points."""
        return self.n
    
    fn get_num_params(self) -> Int:
        """Return number of kernel parameters."""
        return K.num_params()
    
    fn get_noise(self) -> Float32:
        """Return noise variance."""
        return self.noise
    
    fn get_ctx(self) -> DeviceContext:
        """Return device context."""
        return self.ctx
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to training data on device."""
        return self.x_ptr
    
    fn get_params_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to parameters on device."""
        return self.params_ptr
    
    fn get_K_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to materialized kernel matrix on device."""
        return self.K_ptr
    
    fn get_diagonal_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to cached diagonal on device."""
        return self._diag_device.unsafe_ptr()
    
    # =========================================================================
    # MatvecProvider trait conformance
    # These methods enable MaterializedCompositeProvider to be used as the base
    # provider in KroneckerDirectProvider for multi-output Kronecker CG training.
    # =========================================================================
    
    fn get_d(self) -> Int:
        """Return input dimension."""
        return DIM
    
    fn get_kernel_type(self) -> Int:
        """Return kernel type constant.
        
        Composite kernels don't have a single kernel type. Returns -1 as sentinel.
        """
        return -1
    
    fn get_lengthscale(self) -> Float32:
        """Return lengthscale.
        
        Composite kernels don't have a single lengthscale. Returns 0.0 as sentinel.
        """
        return Float32(0.0)
    
    fn get_outputscale(self) -> Float32:
        """Return output scale.
        
        Composite kernels don't have a single outputscale. Returns 1.0 as default.
        """
        return Float32(1.0)
    
    fn get_kernel_param1(self) -> Float32:
        """Return kernel parameter 1. Returns 0.0 for composite kernels."""
        return Float32(0.0)
    
    fn get_kernel_param2(self) -> Float32:
        """Return kernel parameter 2. Returns 0.0 for composite kernels."""
        return Float32(0.0)
    
    fn get_diagonal(self) -> Float32:
        """Return diagonal value (max of cached diagonal + noise)."""
        # For composite kernels, the diagonal varies by point.
        # Return a representative value (first element + noise).
        return Float32(1.0) + self.noise
    
    fn get_use_ard(self) -> Bool:
        """Composite kernels don't use ARD in the traditional sense."""
        return False
    
    fn get_lengthscales_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Not applicable for composite kernels."""
        return UnsafePointer[Float32, MutAnyOrigin]()

    fn get_inv_ls_device_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Not applicable for composite kernels."""
        return UnsafePointer[Float32, MutAnyOrigin]()
    
    fn update_hyperparams(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
    ) raises:
        """Update hyperparameters. Not applicable for composite kernels.
        
        Use update_params() instead for composite kernel parameters.
        """
        self.noise = noise
    
    fn update_hyperparams_ard(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
        outputscale: Float32,
        noise: Float32,
    ) raises:
        """Not applicable for composite kernels."""
        self.noise = noise
    
    fn gradient_matvec_ard(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        dim_index: Int,
        sync: Bool = True,
    ) raises:
        """Not applicable for composite kernels."""
        raise Error("gradient_matvec_ard not supported for MaterializedCompositeProvider")
    
    fn set_lengthscales_device(
        mut self,
        lengthscales_host: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Not applicable for composite kernels."""
        pass
    
    fn update_hyperparams_with_param1(
        mut self,
        lengthscale: Float32,
        outputscale: Float32,
        noise: Float32,
        param1: Float32,
    ) raises:
        """Not applicable for composite kernels. Use update_params() instead."""
        self.noise = noise
    
    fn update_param1(mut self, param1: Float32):
        """Not applicable for composite kernels."""
        pass

    fn update_param2(mut self, param2: Float32):
        """Not applicable for composite kernels."""
        pass


fn create_materialized_composite_provider[DIM: Int, K: ComposableKernel](
    ctx: DeviceContext,
    x_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    noise: Float32,
    materialize_gradients: Bool = False,
) raises -> MaterializedCompositeProvider[DIM, K]:
    """Create a MaterializedCompositeProvider from host data.
    
    Args:
        ctx: GPU device context
        x_host_ptr: Training data on host [n, DIM]
        params_host_ptr: Kernel parameters on host [K.num_params()]
        n: Number of data points
        noise: Noise variance
        materialize_gradients: If True, also materialize gradient matrices
    
    Returns:
        MaterializedCompositeProvider ready for use
    """
    return MaterializedCompositeProvider[DIM, K](
        ctx, x_host_ptr, params_host_ptr, n, noise, materialize_gradients
    )
