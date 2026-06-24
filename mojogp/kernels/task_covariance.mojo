"""TaskCovariance: Inter-task covariance matrix for multi-output GPs.

This module implements the TaskCovariance struct that manages B = WW^T + diag(v),
its eigendecomposition, and analytical gradients for the Intrinsic Coregionalization
Model (ICM).

The key insight is that for multi-output GPs with shared inputs, the full covariance
K_full = K_X ⊗ B can be decomposed using B's eigendecomposition B = Q Λ Q^T into
T independent sub-problems, each with covariance λ_t * K_X + noise * I.

Key features:
- B = WW^T + diag(softplus(raw_v)) ensures positive semi-definiteness
- Eigendecomposition via numpy.linalg.eigh (LAPACK-backed, reliable for tiny T*T matrices)
- Full backward-through-eigh gradient computation in float64 for numerical stability
- GPU kernels for rotating targets (Y @ Q) and unrotating solutions (X @ Q^T)

Reference: Bonilla et al. (2007) "Multi-task Gaussian Process Prediction"
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from python import Python, PythonObject
from random import random_float64
from math import exp as math_exp, log, sqrt

from .constants import float_dtype
from .utils import softplus, softplus_derivative


fn _py_to_f32(py_obj: PythonObject) raises -> Float32:
    """Convert a Python object to Float32 via string parsing.
    
    This is a local copy to avoid circular imports with binding helpers.
    """
    var str_val = String(py_obj)
    return Float32(atof(str_val))


# =============================================================================
# Result Struct for B Gradient
# =============================================================================

struct BGradientResult:
    """Result from compute_B_gradient.
    
    Fields:
        grad_W: Gradient w.r.t. W [T x R]
        grad_raw_v: Gradient w.r.t. raw_v [T]
    """
    var grad_W: HostBuffer[float_dtype]
    var grad_raw_v: HostBuffer[float_dtype]
    
    fn __init__(out self, grad_W: HostBuffer[float_dtype], grad_raw_v: HostBuffer[float_dtype]):
        self.grad_W = grad_W
        self.grad_raw_v = grad_raw_v


# =============================================================================
# GPU Kernels for Target Rotation
# =============================================================================

fn kernel_rotate_targets(
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],        # Input: n x T (row-major)
    q_ptr: UnsafePointer[Float32, MutAnyOrigin],        # Q matrix: T x T (row-major)
    y_rotated_ptr: UnsafePointer[Float32, MutAnyOrigin], # Output: n x T (row-major)
    n: Int,
    T: Int,
) -> None:
    """Compute Y_rotated = Y @ Q on GPU.
    
    Each thread computes one element of the output matrix.
    Y_rotated[i, j] = sum_k Y[i, k] * Q[k, j]
    
    Memory layout (row-major):
    - Y[i, k] = y_ptr[i * T + k]
    - Q[k, j] = q_ptr[k * T + j]
    - Y_rotated[i, j] = y_rotated_ptr[i * T + j]
    """
    var i = block_idx.x * block_dim.x + thread_idx.x  # Row index (data point)
    var j = block_idx.y * block_dim.y + thread_idx.y  # Column index (task)
    
    if i >= UInt(n) or j >= UInt(T):
        return
    
    var sum = Float32(0.0)
    for k in range(T):
        # Y[i, k] * Q[k, j]
        sum += y_ptr[Int(i) * T + k] * q_ptr[k * T + Int(j)]
    
    y_rotated_ptr[Int(i) * T + Int(j)] = sum


fn kernel_unrotate_solution(
    x_rotated_ptr: UnsafePointer[Float32, MutAnyOrigin], # Input: n x T (row-major)
    q_ptr: UnsafePointer[Float32, MutAnyOrigin],         # Q matrix: T x T (row-major)
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],         # Output: n x T (row-major)
    n: Int,
    T: Int,
) -> None:
    """Compute X = X_rotated @ Q^T on GPU.
    
    Each thread computes one element of the output matrix.
    X[i, j] = sum_k X_rotated[i, k] * Q[j, k]  (note: Q^T[k, j] = Q[j, k])
    
    Memory layout (row-major):
    - X_rotated[i, k] = x_rotated_ptr[i * T + k]
    - Q[j, k] = q_ptr[j * T + k]
    - X[i, j] = x_ptr[i * T + j]
    """
    var i = block_idx.x * block_dim.x + thread_idx.x  # Row index (data point)
    var j = block_idx.y * block_dim.y + thread_idx.y  # Column index (task)
    
    if i >= UInt(n) or j >= UInt(T):
        return
    
    var sum = Float32(0.0)
    for k in range(T):
        # X_rotated[i, k] * Q^T[k, j] = X_rotated[i, k] * Q[j, k]
        sum += x_rotated_ptr[Int(i) * T + k] * q_ptr[Int(j) * T + k]
    
    x_ptr[Int(i) * T + Int(j)] = sum


# =============================================================================
# TaskCovariance Struct
# =============================================================================

struct TaskCovariance:
    """Inter-task covariance matrix B = WW^T + diag(softplus(raw_v)).
    
    This struct manages the task covariance matrix for multi-output GPs using
    the Intrinsic Coregionalization Model (ICM). It maintains:
    
    - W: T x R coregionalization matrix (learnable)
    - raw_v: T raw diagonal variances (learnable, transformed via softplus)
    - B: T x T positive semi-definite covariance matrix
    - Q, Lambda: Eigendecomposition B = Q @ diag(Lambda) @ Q^T
    
    The eigendecomposition enables decomposing the full nT x nT system into
    T independent n x n sub-problems, each with covariance λ_t * K_X + noise * I.
    
    Attributes:
        T: Number of tasks
        R: Rank of W (typically R = T for full rank)
        W: T x R coregionalization matrix (row-major, on host)
        raw_v: T raw diagonal variances (before softplus, on host)
        Q: T x T eigenvector matrix (row-major, on host)
        Lambda: T eigenvalues (on host)
        B: T x T covariance matrix (row-major, on host)
        eigendecomp_valid: Whether cached eigendecomposition is up-to-date
        Q_device: T x T eigenvector matrix (row-major, on GPU)
        Lambda_device: T eigenvalues (on GPU)
        ctx: GPU device context
    """
    var T: Int
    var R: Int
    var W: HostBuffer[float_dtype]
    var raw_v: HostBuffer[float_dtype]
    var Q: HostBuffer[float_dtype]
    var Lambda: HostBuffer[float_dtype]
    var B: HostBuffer[float_dtype]
    var eigendecomp_valid: Bool
    var Q_device: DeviceBuffer[float_dtype]
    var Lambda_device: DeviceBuffer[float_dtype]
    var ctx: DeviceContext
    
    fn __init__(out self, ctx: DeviceContext, T: Int, R: Int, seed: Int = 42) raises:
        """Initialize TaskCovariance with small random W and raw_v = 0.
        
        W is initialized with small random values (0.1 to 0.5) so that the
        outputscale can absorb signal variance early in training. raw_v is
        initialized to 0, which gives v = softplus(0) ≈ 0.693.
        
        Args:
            ctx: GPU device context
            T: Number of tasks
            R: Rank of W (typically R = T for full rank)
            seed: Random seed for W initialization
        """
        self.T = T
        self.R = R
        self.ctx = ctx
        self.eigendecomp_valid = False
        
        # Allocate host buffers
        self.W = HostBuffer[float_dtype](ctx, T * R)
        self.raw_v = HostBuffer[float_dtype](ctx, T)
        self.Q = HostBuffer[float_dtype](ctx, T * T)
        self.Lambda = HostBuffer[float_dtype](ctx, T)
        self.B = HostBuffer[float_dtype](ctx, T * T)
        
        # Allocate device buffers
        self.Q_device = ctx.enqueue_create_buffer[float_dtype](T * T)
        self.Lambda_device = ctx.enqueue_create_buffer[float_dtype](T)
        
        # Initialize W with small random values (0.1 to 0.5)
        # Using simple hash-based random for reproducibility
        for i in range(T * R):
            # Simple LCG-style random
            var state = UInt64(seed) ^ (UInt64(i) * UInt64(2654435761))
            state = (state * UInt64(1103515245) + UInt64(12345)) & UInt64(0x7FFFFFFF)
            var rand_val = Float32(state) / Float32(0x7FFFFFFF)  # [0, 1]
            self.W.unsafe_ptr()[i] = Float32(0.1) + rand_val * Float32(0.4)  # [0.1, 0.5]
        
        # Initialize raw_v to 0 (softplus(0) ≈ 0.693)
        for i in range(T):
            self.raw_v.unsafe_ptr()[i] = Float32(0.0)
        
        # Compute initial eigendecomposition
        self.update_eigendecomposition()
    
    fn update_eigendecomposition(mut self) raises:
        """Recompute B = WW^T + diag(softplus(raw_v)), then eigendecompose.
        
        This method should be called after W or raw_v are updated by the optimizer.
        Uses torch.linalg.eigh via Python interop for accurate LAPACK-backed
        eigendecomposition.
        
        The eigendecomposition is computed in float64 for numerical stability,
        then cast back to float32 for storage.
        """
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")
        
        var T = self.T
        var R = self.R
        
        # Build B = WW^T + diag(softplus(raw_v)) in numpy (float64 for accuracy)
        var shape_B = Python.tuple(PythonObject(T), PythonObject(T))
        var B_np = np.zeros(shape_B, dtype=np.float64)
        
        # Compute WW^T
        for i in range(T):
            for j in range(T):
                var sum = Float64(0.0)
                for k in range(R):
                    # W[i, k] * W[j, k]
                    var w_ik = Float64(self.W.unsafe_ptr()[i * R + k])
                    var w_jk = Float64(self.W.unsafe_ptr()[j * R + k])
                    sum += w_ik * w_jk
                B_np[i, j] = sum
        
        # Add diag(softplus(raw_v))
        for i in range(T):
            var v_i = Float64(softplus(self.raw_v.unsafe_ptr()[i]))
            B_np[i, i] = B_np[i, i].__add__(v_i)
        
        # Store B in host buffer (float32)
        for i in range(T):
            for j in range(T):
                self.B.unsafe_ptr()[i * T + j] = _py_to_f32(B_np[i, j])
        
        # Eigendecomposition using numpy (LAPACK-backed, always reliable)
        # These are tiny T*T matrices (T=2-5), Python interop overhead is negligible
        var result = np.linalg.eigh(B_np)
        var eigenvalues_np = result[0]
        var eigenvectors_np = result[1]
        
        for i in range(T):
            self.Lambda.unsafe_ptr()[i] = _py_to_f32(eigenvalues_np[i])
        
        for i in range(T):
            for j in range(T):
                self.Q.unsafe_ptr()[i * T + j] = _py_to_f32(eigenvectors_np[i, j])
        
        # Copy Q and Lambda to device
        self.Q_device.enqueue_copy_from(self.Q)
        self.Lambda_device.enqueue_copy_from(self.Lambda)
        self.ctx.synchronize()
        
        self.eigendecomp_valid = True
    
    fn update_B(mut self) raises:
        """Recompute B = WW^T + diag(softplus(raw_v)) without eigendecomposition.
        
        This method updates only the B matrix, useful when you need B but will
        apply a different transformation (e.g., Rakitsch symmetrization).
        """
        var np = Python.import_module("numpy")
        
        var T = self.T
        var R = self.R
        
        # Build B = WW^T + diag(softplus(raw_v)) in numpy (float64 for accuracy)
        var shape_B = Python.tuple(PythonObject(T), PythonObject(T))
        var B_np = np.zeros(shape_B, dtype=np.float64)
        
        # Compute WW^T
        for i in range(T):
            for j in range(T):
                var sum = Float64(0.0)
                for k in range(R):
                    var w_ik = Float64(self.W.unsafe_ptr()[i * R + k])
                    var w_jk = Float64(self.W.unsafe_ptr()[j * R + k])
                    sum += w_ik * w_jk
                B_np[i, j] = sum
        
        # Add diag(softplus(raw_v))
        for i in range(T):
            var v_i = Float64(softplus(self.raw_v.unsafe_ptr()[i]))
            B_np[i, i] = B_np[i, i].__add__(v_i)
        
        # Store B in host buffer (float32)
        for i in range(T):
            for j in range(T):
                self.B.unsafe_ptr()[i * T + j] = _py_to_f32(B_np[i, j])
    
    fn get_eigenvalue(self, t: Int) -> Float32:
        """Return eigenvalue λ_t.
        
        Args:
            t: Task index (0 to T-1)
            
        Returns:
            The t-th eigenvalue
        """
        return self.Lambda.unsafe_ptr()[t]
    
    fn get_eigenvector_matrix_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to Q matrix (T x T, row-major).
        
        Returns:
            Pointer to the eigenvector matrix on host
        """
        return self.Q.unsafe_ptr()
    
    fn get_W_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to W matrix (T x R, row-major).
        
        Returns:
            Pointer to the coregionalization matrix on host
        """
        return self.W.unsafe_ptr()
    
    fn get_raw_v_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        """Return pointer to raw_v vector (T).
        
        Returns:
            Pointer to the raw diagonal variances on host
        """
        return self.raw_v.unsafe_ptr()
    
    fn rotate_targets(
        self,
        y: DeviceBuffer[float_dtype],
        y_rotated: DeviceBuffer[float_dtype],
        n: Int,
    ) raises:
        """Compute Y_rotated = Y @ Q on GPU.
        
        Rotates the target matrix from the original task basis to the
        eigenvector basis. This enables decomposing the full system into
        T independent sub-problems.
        
        Args:
            y: Input targets [n x T] (row-major, on GPU)
            y_rotated: Output rotated targets [n x T] (row-major, on GPU)
            n: Number of data points
        """
        var T = self.T
        
        # Launch kernel with 2D grid
        var threads_x = 16
        var threads_y = 16
        var blocks_x = (n + threads_x - 1) // threads_x
        var blocks_y = (T + threads_y - 1) // threads_y
        
        self.ctx.enqueue_function[kernel_rotate_targets](
            y.unsafe_ptr(),
            self.Q_device.unsafe_ptr(),
            y_rotated.unsafe_ptr(),
            n,
            T,
            grid_dim=(blocks_x, blocks_y),
            block_dim=(threads_x, threads_y),
        )
    
    fn unrotate_solution(
        self,
        x_rotated: DeviceBuffer[float_dtype],
        x: DeviceBuffer[float_dtype],
        n: Int,
    ) raises:
        """Compute X = X_rotated @ Q^T on GPU.
        
        Unrotates the solution from the eigenvector basis back to the
        original task basis.
        
        Args:
            x_rotated: Input rotated solution [n x T] (row-major, on GPU)
            x: Output solution [n x T] (row-major, on GPU)
            n: Number of data points
        """
        var T = self.T
        
        # Launch kernel with 2D grid
        var threads_x = 16
        var threads_y = 16
        var blocks_x = (n + threads_x - 1) // threads_x
        var blocks_y = (T + threads_y - 1) // threads_y
        
        self.ctx.enqueue_function[kernel_unrotate_solution](
            x_rotated.unsafe_ptr(),
            self.Q_device.unsafe_ptr(),
            x.unsafe_ptr(),
            n,
            T,
            grid_dim=(blocks_x, blocks_y),
            block_dim=(threads_x, threads_y),
        )
    
    fn compute_B_gradient(
        self,
        eigenvalue_grads: HostBuffer[float_dtype],  # T-vector: dNLL/d(lambda_t)
        y_host: HostBuffer[float_dtype],            # n x T original targets (row-major)
        alpha_rotated_host: HostBuffer[float_dtype], # n x T rotated solve vectors (row-major)
        n: Int,
    ) raises -> BGradientResult:
        """Compute dNLL/dW (T x R) and dNLL/d(raw_v) (T) analytically.
        
        Uses the full eigendecomposition backward formula in float64 for
        numerical stability. The Löwdin F matrix F_{ij} = 1/(λ_i - λ_j)
        suffers from catastrophic cancellation in float32 when eigenvalues
        are close.
        
        Algorithm:
        1. G_Q = Y^T @ Alpha_rotated                    (T x T)
        2. M = diag(g) + F .* (Q^T @ G_Q)               (T x T)
        3. dNLL/dB = Q @ Sym(M) @ Q^T                   (T x T)
        4. dNLL/dW = 2 * (dNLL/dB) @ W                  (T x R)
        5. dNLL/d(raw_v) = diag(dNLL/dB) * sigmoid(raw_v)  (T)
        
        where F_{ij} = 1/(λ_i - λ_j) for i≠j, F_{ii} = 0
        and Sym(M) = (M + M^T) / 2
        
        Args:
            eigenvalue_grads: dNLL/d(λ_t) for each task [T]
            y_host: Original targets [n x T] (row-major, on host)
            alpha_rotated_host: Rotated solve vectors [n x T] (row-major, on host)
            n: Number of data points
            
        Returns:
            Tuple of (dNLL/dW [T x R], dNLL/d(raw_v) [T])
        """
        var torch = Python.import_module("torch")
        var np = Python.import_module("numpy")
        
        var T = self.T
        var R = self.R
        
        # =====================================================================
        # Step 1: Build matrices in float64 for numerical stability
        # =====================================================================
        
        # Build Y (n x T) in numpy
        var shape_Y = Python.tuple(PythonObject(n), PythonObject(T))
        var Y_np = np.zeros(shape_Y, dtype=np.float64)
        for i in range(n):
            for j in range(T):
                Y_np[i, j] = Float64(y_host.unsafe_ptr()[i * T + j])
        
        # Build Alpha_rotated (n x T) in numpy
        var Alpha_np = np.zeros(shape_Y, dtype=np.float64)
        for i in range(n):
            for j in range(T):
                Alpha_np[i, j] = Float64(alpha_rotated_host.unsafe_ptr()[i * T + j])
        
        # Build Q (T x T) in numpy
        var shape_Q = Python.tuple(PythonObject(T), PythonObject(T))
        var Q_np = np.zeros(shape_Q, dtype=np.float64)
        for i in range(T):
            for j in range(T):
                Q_np[i, j] = Float64(self.Q.unsafe_ptr()[i * T + j])
        
        # Build eigenvalue gradients (T) in numpy
        var g_np = np.zeros(T, dtype=np.float64)
        for i in range(T):
            g_np[i] = Float64(eigenvalue_grads.unsafe_ptr()[i])
        
        # Build eigenvalues (T) in numpy
        var Lambda_np = np.zeros(T, dtype=np.float64)
        for i in range(T):
            Lambda_np[i] = Float64(self.Lambda.unsafe_ptr()[i])
        
        # Build W (T x R) in numpy
        var shape_W = Python.tuple(PythonObject(T), PythonObject(R))
        var W_np = np.zeros(shape_W, dtype=np.float64)
        for i in range(T):
            for j in range(R):
                W_np[i, j] = Float64(self.W.unsafe_ptr()[i * R + j])
        
        # Build raw_v (T) in numpy
        var raw_v_np = np.zeros(T, dtype=np.float64)
        for i in range(T):
            raw_v_np[i] = Float64(self.raw_v.unsafe_ptr()[i])
        
        # =====================================================================
        # Step 2: Compute G_Q = Y^T @ Alpha_rotated (T x T)
        # =====================================================================
        var Y_torch = torch.tensor(Y_np, dtype=torch.float64)
        var Alpha_torch = torch.tensor(Alpha_np, dtype=torch.float64)
        var G_Q = torch.mm(Y_torch.T, Alpha_torch)  # [T, T]
        
        # =====================================================================
        # Step 3: Build Löwdin F matrix (T x T)
        # F_{ij} = 1/(λ_i - λ_j) for i≠j, F_{ii} = 0
        # With clamping for near-degenerate eigenvalues
        # =====================================================================
        var eps = Float64(1e-12)
        var F_np = np.zeros(shape_Q, dtype=np.float64)
        for i in range(T):
            for j in range(T):
                if i != j:
                    var diff = Float64(atof(String(Lambda_np[i]))) - Float64(atof(String(Lambda_np[j])))
                    if abs(diff) > eps:
                        F_np[i, j] = Float64(1.0) / diff
                    else:
                        # Near-degenerate: set to 0 (gradient is undefined)
                        F_np[i, j] = Float64(0.0)
                else:
                    F_np[i, j] = Float64(0.0)
        
        # =====================================================================
        # Step 4: Compute M = diag(g) + F .* (Q^T @ G_Q) (T x T)
        # =====================================================================
        var Q_torch = torch.tensor(Q_np, dtype=torch.float64)
        var F_torch = torch.tensor(F_np, dtype=torch.float64)
        var g_torch = torch.tensor(g_np, dtype=torch.float64)
        
        var QT_G_Q = torch.mm(Q_torch.T, G_Q)  # [T, T]
        var M = torch.diag(g_torch) + F_torch * QT_G_Q  # Element-wise multiply
        
        # =====================================================================
        # Step 5: Compute dNLL/dB = Q @ Sym(M) @ Q^T (T x T)
        # Sym(M) = (M + M^T) / 2
        # =====================================================================
        var M_sym = (M + M.T) / Float64(2.0)
        var dNLL_dB = torch.mm(torch.mm(Q_torch, M_sym), Q_torch.T)  # [T, T]
        
        # =====================================================================
        # Step 6: Compute dNLL/dW = 2 * (dNLL/dB) @ W (T x R)
        # =====================================================================
        var W_torch = torch.tensor(W_np, dtype=torch.float64)
        var dNLL_dW = Float64(2.0) * torch.mm(dNLL_dB, W_torch)  # [T, R]
        
        # =====================================================================
        # Step 7: Compute dNLL/d(raw_v) = diag(dNLL/dB) * sigmoid(raw_v) (T)
        # sigmoid(x) = 1 / (1 + exp(-x)) = softplus_derivative(x)
        # =====================================================================
        var raw_v_torch = torch.tensor(raw_v_np, dtype=torch.float64)
        var sigmoid_raw_v = torch.sigmoid(raw_v_torch)  # [T]
        var dNLL_dB_diag = torch.diag(dNLL_dB)  # [T]
        var dNLL_draw_v = dNLL_dB_diag * sigmoid_raw_v  # [T]
        
        # =====================================================================
        # Step 8: Convert results back to Float32 and return
        # =====================================================================
        var dW_np = dNLL_dW.numpy()
        var dv_np = dNLL_draw_v.numpy()
        
        # Allocate output buffers
        var grad_W = HostBuffer[float_dtype](self.ctx, T * R)
        var grad_raw_v = HostBuffer[float_dtype](self.ctx, T)
        
        # Copy dNLL/dW
        for i in range(T):
            for j in range(R):
                grad_W.unsafe_ptr()[i * R + j] = _py_to_f32(dW_np[i][j])
        
        # Copy dNLL/d(raw_v)
        for i in range(T):
            grad_raw_v.unsafe_ptr()[i] = _py_to_f32(dv_np[i])
        
        return BGradientResult(grad_W, grad_raw_v)
    
    fn set_W(mut self, W_new: HostBuffer[float_dtype]) raises:
        """Set W matrix and invalidate eigendecomposition cache.
        
        Args:
            W_new: New W matrix [T x R] (row-major)
        """
        for i in range(self.T * self.R):
            self.W.unsafe_ptr()[i] = W_new.unsafe_ptr()[i]
        self.eigendecomp_valid = False
    
    fn set_raw_v(mut self, raw_v_new: HostBuffer[float_dtype]) raises:
        """Set raw_v vector and invalidate eigendecomposition cache.
        
        Args:
            raw_v_new: New raw_v vector [T]
        """
        for i in range(self.T):
            self.raw_v.unsafe_ptr()[i] = raw_v_new.unsafe_ptr()[i]
        self.eigendecomp_valid = False
    
    fn update_W_element(mut self, i: Int, j: Int, value: Float32):
        """Update a single element of W and invalidate cache.
        
        Args:
            i: Row index
            j: Column index
            value: New value
        """
        self.W.unsafe_ptr()[i * self.R + j] = value
        self.eigendecomp_valid = False
    
    fn update_raw_v_element(mut self, i: Int, value: Float32):
        """Update a single element of raw_v and invalidate cache.
        
        Args:
            i: Index
            value: New value
        """
        self.raw_v.unsafe_ptr()[i] = value
        self.eigendecomp_valid = False
    
    fn get_v(self, i: Int) -> Float32:
        """Get the i-th diagonal variance (after softplus transform).
        
        Args:
            i: Index
            
        Returns:
            softplus(raw_v[i])
        """
        return softplus(self.raw_v.unsafe_ptr()[i])
    
    fn get_B_element(self, i: Int, j: Int) -> Float32:
        """Get element B[i, j].
        
        Args:
            i: Row index
            j: Column index
            
        Returns:
            B[i, j]
        """
        return self.B.unsafe_ptr()[i * self.T + j]
    
    fn num_params(self) -> Int:
        """Return total number of learnable parameters (W and raw_v).
        
        Returns:
            T * R + T
        """
        return self.T * self.R + self.T


# =============================================================================
# Per-Task Noise Support (Rakitsch Symmetrization)
# =============================================================================

struct PerTaskNoiseResult:
    """Result from symmetrize_for_per_task_noise.
    
    Contains the symmetrized eigendecomposition and transformed targets.
    
    Fields:
        Q_tilde: Eigenvectors of symmetrized B_tilde [T x T]
        Lambda_tilde: Eigenvalues of symmetrized B_tilde [T]
        y_transformed: Transformed targets [n x T] (D^{-1/2} @ y @ Q_tilde)
        log_det_D: n * sum(log(noise_per_task)) for NLL correction
        noise_per_task: Per-task noise variances [T]
        D_inv_sqrt: D^{-1/2} = diag(1/sqrt(noise_t)) [T]
    """
    var Q_tilde: HostBuffer[float_dtype]
    var Lambda_tilde: HostBuffer[float_dtype]
    var y_transformed: HostBuffer[float_dtype]
    var log_det_D: Float32
    var noise_per_task: HostBuffer[float_dtype]
    var D_inv_sqrt: HostBuffer[float_dtype]
    var T: Int
    var n: Int
    
    fn __init__(
        out self,
        Q_tilde: HostBuffer[float_dtype],
        Lambda_tilde: HostBuffer[float_dtype],
        y_transformed: HostBuffer[float_dtype],
        log_det_D: Float32,
        noise_per_task: HostBuffer[float_dtype],
        D_inv_sqrt: HostBuffer[float_dtype],
        T: Int,
        n: Int,
    ):
        self.Q_tilde = Q_tilde
        self.Lambda_tilde = Lambda_tilde
        self.y_transformed = y_transformed
        self.log_det_D = log_det_D
        self.noise_per_task = noise_per_task
        self.D_inv_sqrt = D_inv_sqrt
        self.T = T
        self.n = n


struct PerTaskNoiseGradientResult:
    """Result from compute_per_task_noise_gradients.
    
    Fields:
        grad_raw_noise_per_task: Gradient w.r.t. raw_noise_per_task [T]
        dNLL_dB_tilde: dNLL/dB_tilde [T x T] as PythonObject (torch.Tensor, float64)
            Computed using the CORRECT Q_tilde and Lambda_tilde from
            the current iteration's Rakitsch decomposition. Used by the
            training loop to compute dNLL/dW and dNLL/d(raw_v) via chain rule.
    """
    var grad_raw_noise_per_task: HostBuffer[float_dtype]
    var dNLL_dB_tilde: PythonObject
    
    fn __init__(out self, grad_raw_noise_per_task: HostBuffer[float_dtype],
                dNLL_dB_tilde: PythonObject):
        self.grad_raw_noise_per_task = grad_raw_noise_per_task
        self.dNLL_dB_tilde = dNLL_dB_tilde


fn symmetrize_for_per_task_noise(
    task_cov: TaskCovariance,
    raw_noise_per_task: HostBuffer[float_dtype],  # T-vector of raw noise (before softplus)
    y_host: HostBuffer[float_dtype],               # n x T targets (row-major)
    n: Int,
) raises -> PerTaskNoiseResult:
    """Apply Rakitsch symmetrization for per-task noise.
    
    Given per-task noise D = diag(sigma_1^2, ..., sigma_T^2), this function:
    1. Computes D^{-1/2} = diag(1/sigma_1, ..., 1/sigma_T)
    2. Symmetrizes: B_tilde = D^{-1/2} @ B @ D^{-1/2}
    3. Eigendecomposes: B_tilde = Q_tilde @ diag(Lambda_tilde) @ Q_tilde^T
    4. Transforms targets: y_tilde = D^{-1/2} @ y @ Q_tilde
    5. Computes log|D| correction for NLL
    
    After symmetrization, sub-problems have noise = 1.0:
        K_t = s_tilde_t * K_X + I
    where s_tilde_t = outputscale * lambda_tilde_t
    
    Args:
        task_cov: TaskCovariance struct (provides B matrix)
        raw_noise_per_task: Raw per-task noise [T] (before softplus)
        y_host: Original targets [n x T] (row-major)
        n: Number of data points
        
    Returns:
        PerTaskNoiseResult with symmetrized eigendecomposition and transformed targets
    """
    var torch = Python.import_module("torch")
    var np = Python.import_module("numpy")
    
    var T = task_cov.T
    var ctx = task_cov.ctx
    
    # =========================================================================
    # Step 1: Compute noise_per_task = softplus(raw_noise_per_task)
    # =========================================================================
    var noise_per_task = HostBuffer[float_dtype](ctx, T)
    var D_inv_sqrt_host = HostBuffer[float_dtype](ctx, T)
    var log_det_D = Float32(0.0)
    
    for t in range(T):
        var noise_t = softplus(raw_noise_per_task.unsafe_ptr()[t])
        noise_per_task.unsafe_ptr()[t] = noise_t
        D_inv_sqrt_host.unsafe_ptr()[t] = Float32(1.0) / sqrt(noise_t)
        log_det_D += log(noise_t)
    
    # Multiply by n (n copies of the T x T diagonal noise matrix)
    log_det_D = Float32(n) * log_det_D
    
    # Check for extreme noise ratios
    var max_noise = Float32(0.0)
    var min_noise = Float32(1e10)
    for t in range(T):
        var noise_t = noise_per_task.unsafe_ptr()[t]
        if noise_t > max_noise:
            max_noise = noise_t
        if noise_t < min_noise:
            min_noise = noise_t
    
    if max_noise / min_noise > Float32(100.0):
        print("WARNING: Per-task noise ratio exceeds 100x. Condition number of B_tilde may be poor.")
    
    # =========================================================================
    # Step 2: Build B_tilde = D^{-1/2} @ B @ D^{-1/2} in float64
    # =========================================================================
    var shape_B = Python.tuple(PythonObject(T), PythonObject(T))
    var B_tilde_np = np.zeros(shape_B, dtype=np.float64)
    
    for i in range(T):
        for j in range(T):
            var B_ij = Float64(task_cov.B.unsafe_ptr()[i * T + j])
            var d_i = Float64(D_inv_sqrt_host.unsafe_ptr()[i])
            var d_j = Float64(D_inv_sqrt_host.unsafe_ptr()[j])
            B_tilde_np[i, j] = d_i * B_ij * d_j
    
    # =========================================================================
    # Step 3: Eigendecompose B_tilde using native Mojo implementation
    # =========================================================================
    # Check for NaN values in B_tilde_np (indicates training instability)
    var has_nan = False
    for i in range(T):
        for j in range(T):
            var val = B_tilde_np[i, j]
            var builtins = Python.import_module("builtins")
            var math_mod = Python.import_module("math")
            if math_mod.isnan(val):
                has_nan = True
                break
        if has_nan:
            break
    
    if has_nan:
        raise Error("B_tilde contains NaN values - training has become unstable")
    
    # Eigendecomposition using numpy (LAPACK-backed, always reliable)
    # These are tiny T*T matrices (T=2-5), Python interop overhead is negligible
    var result = np.linalg.eigh(B_tilde_np)
    var eigenvalues_np = result[0]
    var eigenvectors_np = result[1]
    
    # Copy to host buffers
    var Q_tilde = HostBuffer[float_dtype](ctx, T * T)
    var Lambda_tilde = HostBuffer[float_dtype](ctx, T)
    
    for i in range(T):
        Lambda_tilde.unsafe_ptr()[i] = _py_to_f32(eigenvalues_np[i])
    
    for i in range(T):
        for j in range(T):
            Q_tilde.unsafe_ptr()[i * T + j] = _py_to_f32(eigenvectors_np[i, j])
    
    # =========================================================================
    # Step 4: Transform targets: y_tilde = y @ D^{-1/2} @ Q_tilde
    # (Each row: y[i, :] @ D^{-1/2} @ Q_tilde)
    # =========================================================================
    var y_transformed = HostBuffer[float_dtype](ctx, n * T)
    
    # First apply D^{-1/2}: y_scaled[i, t] = y[i, t] * D_inv_sqrt[t]
    # Then apply Q_tilde: y_transformed[i, s] = sum_t y_scaled[i, t] * Q_tilde[t, s]
    for i in range(n):
        for s in range(T):
            var sum_val = Float32(0.0)
            for t in range(T):
                var y_it = y_host.unsafe_ptr()[i * T + t]
                var d_t = D_inv_sqrt_host.unsafe_ptr()[t]
                var Q_ts = Q_tilde.unsafe_ptr()[t * T + s]
                sum_val += y_it * d_t * Q_ts
            y_transformed.unsafe_ptr()[i * T + s] = sum_val
    
    return PerTaskNoiseResult(
        Q_tilde,
        Lambda_tilde,
        y_transformed,
        log_det_D,
        noise_per_task,
        D_inv_sqrt_host,
        T,
        n,
    )


fn compute_per_task_noise_gradients(
    task_cov: TaskCovariance,
    per_task_result: PerTaskNoiseResult,
    raw_noise_per_task: HostBuffer[float_dtype],
    y_host: HostBuffer[float_dtype],              # n x T original targets
    alpha_rotated_host: HostBuffer[float_dtype],  # n x T rotated solve vectors
    eigenvalue_grads: HostBuffer[float_dtype],    # T-vector: dNLL/d(lambda_tilde_t)
    n: Int,
) raises -> PerTaskNoiseGradientResult:
    """Compute gradients w.r.t. per-task noise.
    
    The gradient dNLL/d(sigma_t^2) has three contributions:
    
    1. Direct log-det term: n / (2 * sigma_t^2)
       From 0.5 * n * sum_t log(sigma_t^2)
    
    2. Through B_tilde (flows through eigendecomposition):
       Requires backward through eigh of B_tilde
    
    3. Through y_tilde (flows through target transformation):
       dNLL/d(d_t) from y_tilde = D^{-1/2} @ y @ Q_tilde
    
    Combined with chain rule for softplus:
       dNLL/d(raw_noise_t) = dNLL/d(sigma_t^2) * softplus_derivative(raw_noise_t)
    
    Args:
        task_cov: TaskCovariance struct
        per_task_result: Result from symmetrize_for_per_task_noise
        raw_noise_per_task: Raw per-task noise [T]
        y_host: Original targets [n x T]
        alpha_rotated_host: Rotated solve vectors [n x T]
        eigenvalue_grads: dNLL/d(lambda_tilde_t) [T]
        n: Number of data points
        
    Returns:
        PerTaskNoiseGradientResult with gradients w.r.t. raw_noise_per_task
    """
    var torch = Python.import_module("torch")
    var np = Python.import_module("numpy")
    
    var T = task_cov.T
    var ctx = task_cov.ctx
    
    # =========================================================================
    # Build matrices in float64 for numerical stability
    # =========================================================================
    var shape_T = Python.tuple(PythonObject(T), PythonObject(T))
    
    # Build B (T x T)
    var B_np = np.zeros(shape_T, dtype=np.float64)
    for i in range(T):
        for j in range(T):
            B_np[i, j] = Float64(task_cov.B.unsafe_ptr()[i * T + j])
    
    # Build Q_tilde (T x T)
    var Q_tilde_np = np.zeros(shape_T, dtype=np.float64)
    for i in range(T):
        for j in range(T):
            Q_tilde_np[i, j] = Float64(per_task_result.Q_tilde.unsafe_ptr()[i * T + j])
    
    # Build Lambda_tilde (T)
    var Lambda_tilde_np = np.zeros(T, dtype=np.float64)
    for i in range(T):
        Lambda_tilde_np[i] = Float64(per_task_result.Lambda_tilde.unsafe_ptr()[i])
    
    # Build D_inv_sqrt (T)
    var D_inv_sqrt_np = np.zeros(T, dtype=np.float64)
    for i in range(T):
        D_inv_sqrt_np[i] = Float64(per_task_result.D_inv_sqrt.unsafe_ptr()[i])
    
    # Build noise_per_task (T)
    var noise_np = np.zeros(T, dtype=np.float64)
    for i in range(T):
        noise_np[i] = Float64(per_task_result.noise_per_task.unsafe_ptr()[i])
    
    # Build eigenvalue gradients (T)
    var g_np = np.zeros(T, dtype=np.float64)
    for i in range(T):
        g_np[i] = Float64(eigenvalue_grads.unsafe_ptr()[i])
    
    # Build Y (n x T) and Alpha_rotated (n x T)
    var shape_Y = Python.tuple(PythonObject(n), PythonObject(T))
    var Y_np = np.zeros(shape_Y, dtype=np.float64)
    var Alpha_np = np.zeros(shape_Y, dtype=np.float64)
    for i in range(n):
        for j in range(T):
            Y_np[i, j] = Float64(y_host.unsafe_ptr()[i * T + j])
            Alpha_np[i, j] = Float64(alpha_rotated_host.unsafe_ptr()[i * T + j])
    
    # =========================================================================
    # Contribution 1: Direct log-det term
    # dNLL/d(sigma_t^2)_direct = n / (2 * sigma_t^2)
    # =========================================================================
    var grad_direct_np = np.zeros(T, dtype=np.float64)
    for t in range(T):
        var sigma_sq = Float64(atof(String(noise_np[t])))
        grad_direct_np[t] = Float64(n) / (Float64(2.0) * sigma_sq)
    
    # =========================================================================
    # Contribution 2: Through B_tilde (backward through eigh)
    # This is complex - we need dNLL/dB_tilde, then chain rule to d_t
    # 
    # dNLL/dB_tilde = Q_tilde @ Sym(M) @ Q_tilde^T
    # where M = diag(g) + F .* (Q_tilde^T @ G_Q)
    # and G_Q = Y_transformed^T @ Alpha_rotated
    # 
    # Then: dNLL/d(d_t) from B_tilde = 2 * sum_j (dNLL/dB_tilde)_{tj} * B_{tj} * d_j
    # =========================================================================
    
    # Compute G_Q = Y_transformed^T @ Alpha_rotated
    var Y_transformed_np = np.zeros(shape_Y, dtype=np.float64)
    for i in range(n):
        for j in range(T):
            Y_transformed_np[i, j] = Float64(per_task_result.y_transformed.unsafe_ptr()[i * T + j])
    
    var Y_torch = torch.tensor(Y_transformed_np, dtype=torch.float64)
    var Alpha_torch = torch.tensor(Alpha_np, dtype=torch.float64)
    var G_Q = torch.mm(Y_torch.T, Alpha_torch)  # [T, T]
    
    # Build Löwdin F matrix
    var eps = Float64(1e-12)
    var F_np = np.zeros(shape_T, dtype=np.float64)
    for i in range(T):
        for j in range(T):
            if i != j:
                var diff = Float64(atof(String(Lambda_tilde_np[i]))) - Float64(atof(String(Lambda_tilde_np[j])))
                if abs(diff) > eps:
                    F_np[i, j] = Float64(1.0) / diff
    
    # Compute M = diag(g) + F .* G_Q
    # G_Q = Y_tilde^T @ Alpha = Q^T @ diag(d) @ Y^T @ Alpha = Q^T @ dL/dQ
    # So G_Q is ALREADY Q^T @ dL/dQ. The eigh backward formula needs:
    #   M = diag(dL/dΛ) + F ⊙ (Q^T dL/dQ) = diag(g) + F ⊙ G_Q
    # Do NOT multiply by Q^T again (that was a bug: extra Q^T rotation).
    var Q_tilde_torch = torch.tensor(Q_tilde_np, dtype=torch.float64)
    var F_torch = torch.tensor(F_np, dtype=torch.float64)
    var g_torch = torch.tensor(g_np, dtype=torch.float64)
    
    var M = torch.diag(g_torch) + F_torch * G_Q
    
    # Compute dNLL/dB_tilde = Q_tilde @ Sym(M) @ Q_tilde^T
    var M_sym = (M + M.T) / Float64(2.0)
    var dNLL_dB_tilde = torch.mm(torch.mm(Q_tilde_torch, M_sym), Q_tilde_torch.T)
    
    # Chain rule: dNLL/d(d_t) from B_tilde
    # B_tilde[i,j] = d_i * B[i,j] * d_j
    # dB_tilde[i,j]/d(d_t) = B[i,t] * d_j if i=t, + B[t,j] * d_i if j=t
    # = delta_{it} * B[i,j] * d_j + delta_{jt} * B[i,j] * d_i
    # 
    # dNLL/d(d_t) = sum_{i,j} dNLL/dB_tilde[i,j] * dB_tilde[i,j]/d(d_t)
    #             = sum_j dNLL/dB_tilde[t,j] * B[t,j] * d_j + sum_i dNLL/dB_tilde[i,t] * B[i,t] * d_i
    #             = 2 * sum_j dNLL/dB_tilde[t,j] * B[t,j] * d_j  (by symmetry)
    var B_torch = torch.tensor(B_np, dtype=torch.float64)
    var D_inv_sqrt_torch = torch.tensor(D_inv_sqrt_np, dtype=torch.float64)
    
    var grad_B_tilde_np = np.zeros(T, dtype=np.float64)
    for t in range(T):
        var sum_val = Float64(0.0)
        for j in range(T):
            var dNLL_dB_tilde_tj = dNLL_dB_tilde[t][j].item()
            var B_tj = Float64(atof(String(B_np[t, j])))
            var d_j = Float64(atof(String(D_inv_sqrt_np[j])))
            sum_val += Float64(atof(String(dNLL_dB_tilde_tj))) * B_tj * d_j
        grad_B_tilde_np[t] = Float64(2.0) * sum_val
    
    # =========================================================================
    # Contribution 3: Through y_tilde (target transformation)
    # y_tilde[i, s] = sum_t y[i, t] * d_t * Q_tilde[t, s]
    # dNLL/d(d_t) from y_tilde = sum_{i,s} dNLL/dy_tilde[i,s] * y[i,t] * Q_tilde[t,s]
    # 
    # dNLL/dy_tilde[i,s] = alpha_rotated[i,s] (from inv_quad term)
    # So: dNLL/d(d_t) = sum_i y[i,t] * (sum_s alpha_rotated[i,s] * Q_tilde[t,s])
    #                 = sum_i y[i,t] * (Alpha_rotated @ Q_tilde^T)[i,t]
    # =========================================================================
    var Y_orig_torch = torch.tensor(Y_np, dtype=torch.float64)
    var Alpha_QT = torch.mm(Alpha_torch, Q_tilde_torch.T)  # [n, T]
    
    var grad_y_tilde_np = np.zeros(T, dtype=np.float64)
    for t in range(T):
        var sum_val = Float64(0.0)
        for i in range(n):
            var y_it = Float64(atof(String(Y_np[i, t])))
            var alpha_qt_it = Alpha_QT[i][t].item()
            sum_val += y_it * Float64(atof(String(alpha_qt_it)))
        grad_y_tilde_np[t] = sum_val
    
    # =========================================================================
    # Combine contributions and apply chain rule
    # 
    # d_t = 1 / sqrt(sigma_t^2)
    # d(d_t)/d(sigma_t^2) = -1 / (2 * sigma_t^3)
    # 
    # dNLL/d(sigma_t^2) = grad_direct + (grad_B_tilde + grad_y_tilde) * (-1/(2*sigma_t^3))
    # 
    # Then apply softplus chain rule:
    # dNLL/d(raw_noise_t) = dNLL/d(sigma_t^2) * softplus_derivative(raw_noise_t)
    # =========================================================================
    var grad_raw_noise = HostBuffer[float_dtype](ctx, T)
    
    for t in range(T):
        var sigma_sq = Float64(atof(String(noise_np[t])))
        var sigma_cubed = sigma_sq * sqrt(sigma_sq)
        
        var grad_d_t = Float64(atof(String(grad_B_tilde_np[t]))) + Float64(atof(String(grad_y_tilde_np[t])))
        var grad_sigma_sq = Float64(atof(String(grad_direct_np[t]))) + grad_d_t * (Float64(-1.0) / (Float64(2.0) * sigma_cubed))
        
        # Apply softplus chain rule
        var raw_noise_t = raw_noise_per_task.unsafe_ptr()[t]
        var softplus_deriv = softplus_derivative(raw_noise_t)
        
        grad_raw_noise.unsafe_ptr()[t] = Float32(grad_sigma_sq) * softplus_deriv
    
    return PerTaskNoiseGradientResult(grad_raw_noise, dNLL_dB_tilde)
