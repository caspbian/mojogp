"""KroneckerPreconditioner for Direct Kronecker CG Multi-Output GP Training.

Implements the Preconditioner trait for the Kronecker system:
    P = outputscale * (B ⊗ L L^T) + D_task ⊗ I_n

where L is the pivoted Cholesky factor of K_X (rank r), B is the T×T task
covariance, and D_task = diag(noise_1, ..., noise_T).

The preconditioner uses the Woodbury identity on the Kronecker structure:
    P = (D^{1/2} ⊗ I) [os * (B_tilde ⊗ L L^T) + I] (D^{1/2} ⊗ I)
    
where B_tilde = D^{-1/2} B D^{-1/2} is the noise-normalized task covariance.

P^{-1} application cost: O(nrT) per application.
log|P| computation: O(rT) from precomputed eigenvalues.
Probe sampling from N(0, P): O(nrT) per sample.

Reference: Saatci (2011), "Scalable Inference for Structured Gaussian Process Models"
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from math import sqrt, log
from buffer import NDBuffer
from linalg.matmul import matmul as max_matmul
from collections import Optional

from .constants import float_dtype
from .preconditioner_trait import Preconditioner
from .cg_solver import kernel_copy
from .native_numerics import matrix_inv_native
from .pivoted_cholesky import kernel_gpu_gaussian


# =============================================================================
# GPU Kernels for Kronecker Preconditioner
# =============================================================================

fn kernel_scale_task_blocks(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scale_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [T] per-task scales
    n: Int,
    num_tasks: Int,
    nT: Int,
    num_cols: Int,
) -> None:
    """Scale each task block by a per-task factor.
    
    out[s*n+i, c] = scale[s] * v[s*n+i, c]
    
    Column-major layout: element (row, col) at index col * nT + row.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = nT * num_cols
    
    if idx >= UInt(total):
        return
    
    var col = Int(idx) // nT
    var row = Int(idx) % nT
    var s = row // n  # task index
    
    out_ptr[idx] = scale_ptr[s] * v_ptr[idx]


fn kernel_scale_diag_rT(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    diag_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [r * T] diagonal values
    rT: Int,
    num_cols: Int,
) -> None:
    """Scale each element by a diagonal: out[i, c] = diag[i] * v[i, c].
    
    Column-major layout.
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = rT * num_cols
    
    if idx >= UInt(total):
        return
    
    var col = Int(idx) // rT
    var row = Int(idx) % rT
    
    out_ptr[idx] = diag_ptr[row] * v_ptr[idx]


fn kernel_subtract_scaled(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],
    sub_ptr: UnsafePointer[Float32, MutAnyOrigin],
    scale: Float32,
    total: Int,
) -> None:
    """out = v - scale * sub."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    
    if idx >= UInt(total):
        return
    
    out_ptr[idx] = v_ptr[idx] - scale * sub_ptr[idx]


fn kernel_add_scaled_noise(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    L_result_ptr: UnsafePointer[Float32, MutAnyOrigin],
    z_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
    sqrt_noise_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [T] sqrt(noise_t)
    n: Int,
    num_tasks: Int,
    nT: Int,
    num_cols: Int,
) -> None:
    """Combine L @ z_rank + sqrt(noise_t) * z_noise for probe sampling.
    
    out[s*n+i, c] = L_result[s*n+i, c] + sqrt_noise[s] * z_noise[s*n+i, c]
    """
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var total = nT * num_cols
    
    if idx >= UInt(total):
        return
    
    var col = Int(idx) // nT
    var row = Int(idx) % nT
    var s = row // n
    
    out_ptr[idx] = L_result_ptr[idx] + sqrt_noise_ptr[s] * z_noise_ptr[idx]


# =============================================================================
# KroneckerPreconditioner Struct
# =============================================================================

struct KroneckerPreconditioner(Preconditioner, Copyable, Movable):
    """Woodbury-based preconditioner for the Kronecker system.
    
    Implements P^{-1} where P = outputscale * (B ⊗ L L^T) + D_task ⊗ I_n.
    
    The Woodbury identity gives:
        P^{-1} = (D^{-1/2} ⊗ I) M^{-1} (D^{-1/2} ⊗ I)
    where:
        M = outputscale * (B_tilde ⊗ L L^T) + I
        M^{-1} v = v - (I_T ⊗ L) C^{-1} (I_T ⊗ L^T) v  [os absorbed in C^{-1}]
        C eigenvalues: 1/(os * λ_j) + σ_i  in (Q_tilde ⊗ V) basis
    
    Precomputed:
        - L [n × r]: pivoted Cholesky factor of K_X
        - V [r × r]: eigenvectors of L^T L
        - Sigma [r]: eigenvalues of L^T L (σ_i)
        - Q_tilde [T × T]: eigenvectors of B_tilde
        - Lambda_tilde [T]: eigenvalues of B_tilde (λ_j)
        - C_inv_diag [T × r]: precomputed 1 / (1/(os*λ_j) + σ_i)
        - D_inv_sqrt [T]: 1/sqrt(noise_t)
        - D_sqrt [T]: sqrt(noise_t)
    """
    var L_device: DeviceBuffer[float_dtype]          # [n × r] pivoted Cholesky factor
    var V_device: DeviceBuffer[float_dtype]           # [r × r] eigenvectors of L^T L
    var Sigma_host: HostBuffer[float_dtype]           # [r] eigenvalues of L^T L
    var Q_tilde_device: DeviceBuffer[float_dtype]     # [T × T] eigenvectors of B_tilde
    var Lambda_tilde_host: HostBuffer[float_dtype]    # [T] eigenvalues of B_tilde
    var C_inv_diag_device: DeviceBuffer[float_dtype]  # [T * r] precomputed diagonal
    var D_inv_sqrt_device: DeviceBuffer[float_dtype]  # [T] 1/sqrt(noise_t)
    var D_sqrt_device: DeviceBuffer[float_dtype]      # [T] sqrt(noise_t)
    var outputscale: Float32
    var n: Int
    var rank: Int
    var num_tasks: Int
    var log_det_val: Float32                          # Precomputed log|P|
    
    # Work buffers for apply_precond
    var w_scaled: DeviceBuffer[float_dtype]           # [nT * max_cols] D^{-1/2} scaled input
    var w_Lt_v: DeviceBuffer[float_dtype]             # [r * T * max_cols] L^T @ v per task
    var w_rotated: DeviceBuffer[float_dtype]          # [r * T * max_cols] rotated by Q_tilde ⊗ V
    var w_L_result: DeviceBuffer[float_dtype]         # [nT * max_cols] L @ result per task
    var max_num_cols: Int
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        owned L_device: DeviceBuffer[float_dtype],
        n: Int,
        rank: Int,
        num_tasks: Int,
        outputscale: Float32,
        B_host: HostBuffer[float_dtype],       # [T × T] row-major
        noise_host: HostBuffer[float_dtype],   # [T]
        max_num_cols: Int = 16,
    ) raises:
        """Build KroneckerPreconditioner from pivoted Cholesky factor and task covariance.
        
        Steps:
        1. Compute L^T L [r × r] via GPU matmul, eigendecompose on CPU → V, Sigma
        2. Compute B_tilde = D^{-1/2} B D^{-1/2}, eigendecompose on CPU → Q_tilde, Lambda_tilde
        3. Precompute C^{-1} diagonal and log|P|
        4. Upload everything to GPU
        
        Args:
            ctx: GPU device context.
            L_device: Pivoted Cholesky factor [n × r] column-major on GPU.
            n: Number of data points.
            rank: Rank of pivoted Cholesky.
            num_tasks: Number of tasks T.
            outputscale: Global output scale.
            B_host: Task covariance B [T × T] row-major on host.
            noise_host: Per-task noise [T] on host.
            max_num_cols: Maximum CG columns for work buffers.
        """
        self.L_device = L_device^
        self.n = n
        self.rank = rank
        self.num_tasks = num_tasks
        self.outputscale = outputscale
        self.max_num_cols = max_num_cols
        var nT = n * num_tasks
        var T = num_tasks
        var r = rank
        
        # =====================================================================
        # Step 1: Compute L^T L and eigendecompose
        # =====================================================================
        
        # L^T L via GPU matmul: [r × r]
        var LtL_device = ctx.enqueue_create_buffer[float_dtype](r * r)
        # Column-major L [n, r] → row-major [r, n]
        # L^T L in column-major = (L^T @ L) 
        # Row-major interpretation: L is [r, n], L^T is [n, r]
        # We want C = L^T @ L [r, r]
        # In row-major: C[r,r] = L[r,n] @ L[r,n]^T → use transpose_b
        var L_ndbuf = NDBuffer[DType.float32, 2](self.L_device.unsafe_ptr(), (r, n))
        var LtL_ndbuf = NDBuffer[DType.float32, 2](LtL_device.unsafe_ptr(), (r, r))
        var opt_ctx = Optional[DeviceContext](ctx)
        max_matmul[transpose_b=True, target="gpu"](LtL_ndbuf, L_ndbuf, L_ndbuf, opt_ctx)
        
        # Copy L^T L to host for eigendecomposition
        var LtL_host = ctx.enqueue_create_host_buffer[float_dtype](r * r)
        ctx.enqueue_copy(dst_buf=LtL_host, src_buf=LtL_device)
        ctx.synchronize()
        
        # Eigendecompose L^T L on CPU (small r×r matrix)
        # Use Jacobi eigendecomposition
        self.Sigma_host = HostBuffer[float_dtype](ctx, r)
        var V_host = ctx.enqueue_create_host_buffer[float_dtype](r * r)
        _symmetric_eigen_cpu(LtL_host.unsafe_ptr(), V_host.unsafe_ptr(), 
                            self.Sigma_host.unsafe_ptr(), r)
        
        self.V_device = ctx.enqueue_create_buffer[float_dtype](r * r)
        ctx.enqueue_copy(dst_buf=self.V_device, src_buf=V_host)
        
        # =====================================================================
        # Step 2: Compute B_tilde and eigendecompose
        # =====================================================================
        
        # D^{-1/2} and D^{1/2}
        var D_inv_sqrt_host = ctx.enqueue_create_host_buffer[float_dtype](T)
        var D_sqrt_host = ctx.enqueue_create_host_buffer[float_dtype](T)
        for t in range(T):
            var noise_t = noise_host.unsafe_ptr()[t]
            D_inv_sqrt_host[t] = Float32(1.0) / sqrt(noise_t)
            D_sqrt_host[t] = sqrt(noise_t)
        
        self.D_inv_sqrt_device = ctx.enqueue_create_buffer[float_dtype](T)
        self.D_sqrt_device = ctx.enqueue_create_buffer[float_dtype](T)
        ctx.enqueue_copy(dst_buf=self.D_inv_sqrt_device, src_buf=D_inv_sqrt_host)
        ctx.enqueue_copy(dst_buf=self.D_sqrt_device, src_buf=D_sqrt_host)
        
        # B_tilde = D^{-1/2} B D^{-1/2} + jitter * I
        # Jitter stabilizes eigendecomposition for near-singular B matrices
        # (e.g. from periodic or linear kernels at small n)
        var B_tilde_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
        alias JITTER = Float32(1e-6)
        for s in range(T):
            for t in range(T):
                var val = D_inv_sqrt_host[s] * B_host.unsafe_ptr()[s * T + t] * D_inv_sqrt_host[t]
                if s == t:
                    val += JITTER
                B_tilde_host[s * T + t] = val
        
        # Eigendecompose B_tilde on CPU (small T×T matrix)
        self.Lambda_tilde_host = HostBuffer[float_dtype](ctx, T)
        var Q_tilde_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
        _symmetric_eigen_cpu(B_tilde_host.unsafe_ptr(), Q_tilde_host.unsafe_ptr(),
                            self.Lambda_tilde_host.unsafe_ptr(), T)
        
        self.Q_tilde_device = ctx.enqueue_create_buffer[float_dtype](T * T)
        ctx.enqueue_copy(dst_buf=self.Q_tilde_device, src_buf=Q_tilde_host)
        
        # =====================================================================
        # Step 3: Precompute C^{-1} diagonal and log|P|
        # =====================================================================
        
        # C eigenvalues: c_{j,i} = 1/(os * λ_j) + σ_i
        # C^{-1} diagonal: 1 / c_{j,i}
        # Layout: [T * r] with index j * r + i
        var C_inv_diag_host = ctx.enqueue_create_host_buffer[float_dtype](T * r)
        self.log_det_val = Float32(0.0)
        
        # log|P| = n * sum_t log(noise_t) + sum_{j,i} log(1 + os * λ_j * σ_i)
        for t in range(T):
            self.log_det_val += Float32(n) * log(noise_host.unsafe_ptr()[t])
        
        # Clamp near-zero eigenvalues to prevent division by zero / inf.
        # This is needed for kernels that produce near-singular matrices
        # (e.g. periodic, linear) where lambda_j or sigma_i can be ~0.
        alias EIGEN_CLAMP = Float32(1e-8)
        
        for j in range(T):
            var lambda_j = self.Lambda_tilde_host.unsafe_ptr()[j]
            if lambda_j < EIGEN_CLAMP:
                lambda_j = EIGEN_CLAMP
            for i in range(r):
                var sigma_i = self.Sigma_host.unsafe_ptr()[i]
                if sigma_i < EIGEN_CLAMP:
                    sigma_i = EIGEN_CLAMP
                var c_ji = Float32(1.0) / (outputscale * lambda_j) + sigma_i
                C_inv_diag_host[j * r + i] = Float32(1.0) / c_ji
                
                # log|P| contribution: log(1 + os * λ_j * σ_i)
                var log_arg = Float32(1.0) + outputscale * lambda_j * sigma_i
                if log_arg <= Float32(0.0):
                    log_arg = Float32(1e-10)  # Clamp to prevent NaN
                self.log_det_val += log(log_arg)
        
        self.C_inv_diag_device = ctx.enqueue_create_buffer[float_dtype](T * r)
        ctx.enqueue_copy(dst_buf=self.C_inv_diag_device, src_buf=C_inv_diag_host)
        
        # =====================================================================
        # Step 4: Allocate work buffers
        # =====================================================================
        self.w_scaled = ctx.enqueue_create_buffer[float_dtype](nT * max_num_cols)
        self.w_Lt_v = ctx.enqueue_create_buffer[float_dtype](r * T * max_num_cols)
        self.w_rotated = ctx.enqueue_create_buffer[float_dtype](r * T * max_num_cols)
        self.w_L_result = ctx.enqueue_create_buffer[float_dtype](nT * max_num_cols)
        
        ctx.synchronize()
    
    fn apply_precond(
        self,
        ctx: DeviceContext,
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_total: Int,  # Should be n * num_tasks
        num_cols: Int,
        sync: Bool,
    ) raises:
        """Apply P^{-1} @ v using Woodbury identity on Kronecker structure.
        
        Algorithm:
        1. w = D^{-1/2} @ v (per-task scaling)
        2. For each task t: compute L^T @ w_t → w_Lt [r × num_cols]
        3. Rotate by (Q_tilde^T ⊗ V^T): small matrix ops
        4. Scale by C^{-1} diagonal
        5. Rotate back by (Q_tilde ⊗ V)
        6. For each task t: compute L @ result_t → w_L [n × num_cols]
        7. out = D^{-1/2} @ (w - w_L)  [os already absorbed in C^{-1}]
        
        Total cost: O(nrT) per application.
        """
        var nn = self.n
        var T = self.num_tasks
        var r = self.rank
        var nT = nn * T
        var os = self.outputscale
        alias BLOCK = 256
        var opt_ctx = Optional[DeviceContext](ctx)
        
        # Step 1: w = D^{-1/2} @ v (per-task block scaling)
        var total_elements = nT * num_cols
        var num_blocks = (total_elements + BLOCK - 1) // BLOCK
        ctx.enqueue_function[kernel_scale_task_blocks](
            self.w_scaled.unsafe_ptr(), v_ptr,
            self.D_inv_sqrt_device.unsafe_ptr(),
            nn, T, nT, num_cols,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        
        # Step 2: For each task t, compute L^T @ w_t → w_Lt_v[t*r*num_cols..(t+1)*r*num_cols]
        # L is column-major [n, r] → row-major [r, n]
        # w_t is column-major [n, num_cols] → row-major [num_cols, n]
        # L^T @ w_t [r, num_cols] → row-major [num_cols, r]
        # So: result[num_cols, r] = w_t[num_cols, n] @ L[r, n]^T
        var L_ndbuf = NDBuffer[DType.float32, 2](self.L_device.unsafe_ptr(), (r, nn))
        
        for t in range(T):
            var w_t_ptr = self.w_scaled.unsafe_ptr().offset(t * nn)  # Start of task t in each column
            # For column-major with nT stride, we need to handle the stride properly.
            # Actually, w_scaled is [nT × num_cols] column-major. Task t's data for column c
            # is at offset c * nT + t * nn.
            # We can't directly use matmul with strided access. Instead, we need to
            # extract task blocks first, or handle it column by column.
            
            # For simplicity and correctness, process one column at a time:
            for c in range(num_cols):
                var src_offset = c * nT + t * nn
                var dst_offset = (t * num_cols + c) * r  # w_Lt_v layout: [T, num_cols, r] contiguous
                
                # Extract w_t[:,c] into a contiguous buffer (reuse part of w_L_result as temp)
                var temp_n_ptr = self.w_L_result.unsafe_ptr()  # Reuse as temp
                ctx.enqueue_function[kernel_copy](
                    temp_n_ptr, self.w_scaled.unsafe_ptr().offset(src_offset), nn,
                    grid_dim=((nn + 255) // 256,), block_dim=(256,),
                )
                
                # L^T @ w_t[:,c]: result[1, r] = w_t[1, n] @ L[r, n]^T
                var w_t_ndbuf = NDBuffer[DType.float32, 2](temp_n_ptr, (1, nn))
                var lt_v_ndbuf = NDBuffer[DType.float32, 2](
                    self.w_Lt_v.unsafe_ptr().offset(dst_offset), (1, r)
                )
                max_matmul[transpose_b=True, target="gpu"](lt_v_ndbuf, w_t_ndbuf, L_ndbuf, opt_ctx)
        
        # Step 3-5: Rotate by (Q_tilde^T ⊗ V^T), scale by C^{-1}, rotate back
        # This operates on the [T, r] space for each column.
        # w_Lt_v layout: [T, num_cols, r] → for each column c, we have T vectors of size r.
        # 
        # The rotation is:
        #   rotated[j, i] = sum_t Q_tilde[t, j] * sum_k V[k, i] * w_Lt_v[t, k]
        # Then scale: rotated[j, i] *= C_inv_diag[j * r + i]
        # Then rotate back:
        #   result[t, k] = sum_j Q_tilde[t, j] * sum_i V[k, i] * rotated[j, i]
        #
        # For small T and r (T=3, r=15), this is done on CPU for simplicity.
        # Copy w_Lt_v to host, do the rotation, copy back.
        
        var rT = r * T
        # Use max_num_cols for host buffer size to match self.w_Lt_v device buffer size
        # (self.w_Lt_v is allocated with r * T * max_num_cols)
        var w_Lt_v_host = ctx.enqueue_create_host_buffer[float_dtype](rT * self.max_num_cols)
        ctx.enqueue_copy(dst_buf=w_Lt_v_host, src_buf=self.w_Lt_v)
        
        var Q_tilde_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
        ctx.enqueue_copy(dst_buf=Q_tilde_host, src_buf=self.Q_tilde_device)
        
        var V_host = ctx.enqueue_create_host_buffer[float_dtype](r * r)
        ctx.enqueue_copy(dst_buf=V_host, src_buf=self.V_device)
        
        var C_inv_diag_host = ctx.enqueue_create_host_buffer[float_dtype](T * r)
        ctx.enqueue_copy(dst_buf=C_inv_diag_host, src_buf=self.C_inv_diag_device)
        ctx.synchronize()
        
        var w_rotated_host = ctx.enqueue_create_host_buffer[float_dtype](rT * self.max_num_cols)
        
        for c in range(num_cols):
            # Forward rotation: (Q_tilde^T ⊗ V^T) @ w_Lt_v[:, c]
            # w_Lt_v layout for column c: [t * num_cols * r + c * r + i] = w_Lt_v[t, c, i]
            # But we stored it as [T, num_cols, r] contiguous, so index = (t * num_cols + c) * r + i
            
            # Step 3: Rotate to (Q_tilde ⊗ V) basis
            for j in range(T):
                for i in range(r):
                    var val = Float32(0.0)
                    for t in range(T):
                        for k in range(r):
                            var w_idx = (t * num_cols + c) * r + k
                            # Q_tilde^T[j, t] = Q_tilde[t, j] (row-major)
                            # V^T[i, k] = V[k, i] (row-major)
                            val += Q_tilde_host[t * T + j] * V_host[k * r + i] * w_Lt_v_host[w_idx]
                    
                    # Step 4: Scale by C^{-1}
                    val *= C_inv_diag_host[j * r + i]
                    
                    # Store in rotated buffer (same layout)
                    var rot_idx = (j * num_cols + c) * r + i
                    w_rotated_host[rot_idx] = val
            
            # Step 5: Rotate back to original basis
            for t in range(T):
                for k in range(r):
                    var val = Float32(0.0)
                    for j in range(T):
                        for i in range(r):
                            var rot_idx = (j * num_cols + c) * r + i
                            # Q_tilde[t, j] * V[k, i]
                            val += Q_tilde_host[t * T + j] * V_host[k * r + i] * w_rotated_host[rot_idx]
                    
                    var out_idx = (t * num_cols + c) * r + k
                    w_Lt_v_host[out_idx] = val  # Reuse w_Lt_v_host for the result
        
        # Copy result back to GPU
        ctx.enqueue_copy(dst_buf=self.w_Lt_v, src_buf=w_Lt_v_host)
        ctx.synchronize()
        
        # Step 6: For each task t, compute L @ result_t → w_L_result
        for t in range(T):
            for c in range(num_cols):
                var src_offset = (t * num_cols + c) * r
                var dst_col_offset = c * nT + t * nn
                
                # L @ result_t[:,c]: result[1, n] = result[1, r] @ L[r, n]
                var result_ndbuf = NDBuffer[DType.float32, 2](
                    self.w_Lt_v.unsafe_ptr().offset(src_offset), (1, r)
                )
                var L_result_ndbuf = NDBuffer[DType.float32, 2](
                    self.w_L_result.unsafe_ptr().offset(dst_col_offset), (1, nn)
                )
                max_matmul[target="gpu"](L_result_ndbuf, result_ndbuf, L_ndbuf, opt_ctx)
        
        # Step 7: out = D^{-1/2} @ (w - L_result)
        # The Woodbury identity gives M^{-1} v = v - (I_T ⊗ L) C^{-1}_rot (I_T ⊗ L^T) v
        # where C^{-1} already absorbs the outputscale factor via
        # c_{j,i} = 1/(os * λ_j) + σ_i. No extra os multiplication needed here.
        # BUG FIX: Previously had `os *` which double-counted outputscale,
        # causing CG divergence whenever os != 1.0 (i.e., after any rebuild
        # where outputscale had changed from its initial value of 1.0).
        total_elements = nT * num_cols
        num_blocks = (total_elements + BLOCK - 1) // BLOCK
        ctx.enqueue_function[kernel_subtract_scaled](
            self.w_scaled.unsafe_ptr(),  # Reuse w_scaled as temp
            self.w_scaled.unsafe_ptr(),  # w (D^{-1/2} @ v from step 1)
            self.w_L_result.unsafe_ptr(),
            Float32(1.0),               # No extra os factor — already in C^{-1}
            total_elements,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        
        # Then: out = D^{-1/2} @ temp
        ctx.enqueue_function[kernel_scale_task_blocks](
            out_ptr, self.w_scaled.unsafe_ptr(),
            self.D_inv_sqrt_device.unsafe_ptr(),
            nn, T, nT, num_cols,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        
        if sync:
            ctx.synchronize()
    
    fn log_det(self, ctx: DeviceContext) raises -> Float32:
        """Return precomputed log|P|.
        
        log|P| = n * sum_t log(noise_t) + sum_{j,i} log(1 + os * λ_j * σ_i)
        """
        return self.log_det_val
    
    fn sample_probes(
        self,
        ctx: DeviceContext,
        out_device: DeviceBuffer[float_dtype],
        num_probes: Int,
        seed_val: UInt64,
    ) raises:
        """Sample probe vectors from N(0, P).
        
        P^{1/2} z where z ~ N(0, I_{nT}):
        
        The sampling uses the Kronecker structure:
        1. Generate z_noise ~ N(0, I_{nT}) 
        2. Generate z_rank ~ N(0, I_{rT})
        3. Compute L_part = (I_T ⊗ L) @ (C_sqrt ⊗ I) @ (Q_tilde ⊗ V)^T @ z_rank
        4. out = (D^{1/2} ⊗ I) @ (sqrt(os) * L_part + z_noise)
        
        For simplicity, we use a simpler but equivalent approach:
        out = D^{1/2} @ (L @ z_rank_scaled + z_noise)
        where z_rank_scaled accounts for the Kronecker structure.
        
        Actually, the simplest correct approach for N(0, P) where
        P = os * (B ⊗ L L^T) + D ⊗ I:
        
        1. z ~ N(0, I_{nT})
        2. For each task block: out[s*n..(s+1)*n] = sqrt(noise_s) * z[s*n..(s+1)*n]
        3. z_r ~ N(0, I_{rT})
        4. Compute Cholesky of os*B = C_B C_B^T (small T×T)
        5. For each task s: out[s*n..(s+1)*n] += sum_t C_B[s,t] * L @ z_r[t*r..(t+1)*r]
        
        This gives samples from N(0, os * (B ⊗ L L^T) + D ⊗ I).
        """
        var nn = self.n
        var T = self.num_tasks
        var r = self.rank
        var nT = nn * T
        var os = self.outputscale
        alias BLOCK = 256
        var opt_ctx = Optional[DeviceContext](ctx)
        
        # Generate deterministic Gaussian factors directly on GPU. Avoid the
        # process-global host RNG so repeated training is independent of prior
        # tests or models in the same Python process.
        var z_noise_device = ctx.enqueue_create_buffer[float_dtype](nT * num_probes)
        var z_rank_device = ctx.enqueue_create_buffer[float_dtype](r * T * num_probes)
        ctx.enqueue_function[kernel_gpu_gaussian](
            z_noise_device.unsafe_ptr(), nT * num_probes, seed_val + UInt64(12345),
            grid_dim=((nT * num_probes + BLOCK - 1) // BLOCK,), block_dim=(BLOCK,),
        )
        ctx.enqueue_function[kernel_gpu_gaussian](
            z_rank_device.unsafe_ptr(), r * T * num_probes, seed_val,
            grid_dim=((r * T * num_probes + BLOCK - 1) // BLOCK,), block_dim=(BLOCK,),
        )
        ctx.synchronize()
        
        # Compute Cholesky of os * B on CPU (small T×T)
        # First get B from B_tilde: B = D^{1/2} B_tilde D^{1/2}
        # Actually, we need the original B. Let's reconstruct it.
        # B_tilde = D^{-1/2} B D^{-1/2}, so B = D^{1/2} B_tilde D^{1/2}
        # B_tilde = Q_tilde Lambda_tilde Q_tilde^T
        # os * B = os * D^{1/2} Q_tilde Lambda_tilde Q_tilde^T D^{1/2}
        # Cholesky of os * B: C_B such that C_B C_B^T = os * B
        # C_B = sqrt(os) * D^{1/2} Q_tilde Lambda_tilde^{1/2}
        
        # Compute C_B [T × T] on host
        var Q_tilde_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
        ctx.enqueue_copy(dst_buf=Q_tilde_host, src_buf=self.Q_tilde_device)
        var D_sqrt_host = ctx.enqueue_create_host_buffer[float_dtype](T)
        ctx.enqueue_copy(dst_buf=D_sqrt_host, src_buf=self.D_sqrt_device)
        ctx.synchronize()
        
        var C_B_host = ctx.enqueue_create_host_buffer[float_dtype](T * T)
        var sqrt_os = sqrt(os)
        for s in range(T):
            for j in range(T):
                var lambda_j = self.Lambda_tilde_host.unsafe_ptr()[j]
                # Clamp negative eigenvalues to small positive value
                var sqrt_lambda_j = sqrt(max(lambda_j, Float32(1e-8)))
                C_B_host[s * T + j] = sqrt_os * D_sqrt_host[s] * Q_tilde_host[s * T + j] * sqrt_lambda_j
        
        # Now compute probes: for each probe p and task s:
        # out[s*n+i, p] = sqrt(noise_s) * z_noise[s*n+i, p] + sum_t C_B[s,t] * L @ z_rank[t*r..(t+1)*r, p]
        
        # Initialize output with sqrt(noise_s) * z_noise
        var out_ptr = out_device.unsafe_ptr()
        var num_blocks = (nT * num_probes + BLOCK - 1) // BLOCK
        ctx.enqueue_function[kernel_scale_task_blocks](
            out_ptr, z_noise_device.unsafe_ptr(),
            self.D_sqrt_device.unsafe_ptr(),
            nn, T, nT, num_probes,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        
        # Add C_B[s,t] * L @ z_rank[t] for each task pair
        var L_ndbuf = NDBuffer[DType.float32, 2](self.L_device.unsafe_ptr(), (r, nn))
        
        # Temp buffer for L @ z_rank_t_p [n]
        var temp_Lz = ctx.enqueue_create_buffer[float_dtype](nn)
        
        for p in range(num_probes):
            for t in range(T):
                # z_rank for task t, probe p: z_rank_device[(t * num_probes + p) * r ... + r]
                var z_r_offset = (t * num_probes + p) * r
                
                # L @ z_rank_t_p: [1, n] = [1, r] @ [r, n]
                var z_r_ndbuf = NDBuffer[DType.float32, 2](
                    z_rank_device.unsafe_ptr().offset(z_r_offset), (1, r)
                )
                var Lz_ndbuf = NDBuffer[DType.float32, 2](temp_Lz.unsafe_ptr(), (1, nn))
                max_matmul[target="gpu"](Lz_ndbuf, z_r_ndbuf, L_ndbuf, opt_ctx)
                
                # Add C_B[s,t] * L @ z_rank_t_p to out[s*n..(s+1)*n, p] for each task s
                for s in range(T):
                    var c_b_st = C_B_host[s * T + t]
                    if c_b_st != Float32(0.0):
                        var out_offset = p * nT + s * nn
                        # out[out_offset..out_offset+n] += c_b_st * temp_Lz[0..n]
                        var axpy_blocks = (nn + BLOCK - 1) // BLOCK
                        ctx.enqueue_function[_kernel_axpy](
                            out_ptr.offset(out_offset),
                            temp_Lz.unsafe_ptr(),
                            c_b_st,
                            nn,
                            grid_dim=(axpy_blocks,), block_dim=(BLOCK,),
                        )
        
        ctx.synchronize()
        
        # Keep buffers alive
        _ = z_noise_device
        _ = z_rank_device
        _ = temp_Lz
    
    fn get_rank(self) -> Int:
        """Return the rank of the pivoted Cholesky factor."""
        return self.rank


# =============================================================================
# Helper GPU Kernels
# =============================================================================

fn _kernel_axpy(
    y_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_ptr: UnsafePointer[Float32, MutAnyOrigin],
    alpha: Float32,
    n: Int,
) -> None:
    """y += alpha * x."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    y_ptr[idx] += alpha * x_ptr[idx]


# =============================================================================
# CPU Eigendecomposition for Small Matrices
# =============================================================================

fn _symmetric_eigen_cpu(
    A_ptr: UnsafePointer[Float32, MutAnyOrigin],
    V_ptr: UnsafePointer[Float32, MutAnyOrigin],
    eigenvalues_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) raises:
    """Eigendecompose a symmetric n×n matrix using Jacobi iteration in Float64.
    
    A = V @ diag(eigenvalues) @ V^T
    
    A is row-major [n × n] (input, not destroyed).
    V is row-major [n × n] (output, eigenvectors as columns).
    eigenvalues is [n] (output).
    
    Uses classical Jacobi eigenvalue algorithm for small matrices (n < 20).
    Internal computation in Float64 for numerical stability.
    """
    # Copy A to Float64 working matrix
    var W_storage = List[Float64](capacity=n * n)
    for i in range(n * n):
        W_storage.append(Float64(A_ptr[i]))
    var W = W_storage.unsafe_ptr()
    
    # Float64 eigenvector accumulator
    var V64_storage = List[Float64](capacity=n * n)
    for i in range(n):
        for j in range(n):
            if i == j:
                V64_storage.append(Float64(1.0))
            else:
                V64_storage.append(Float64(0.0))
    var V64 = V64_storage.unsafe_ptr()
    
    # Jacobi iteration in Float64
    var max_iter = 100 * n * n
    var tol = Float64(1e-12)
    
    for _ in range(max_iter):
        # Find largest off-diagonal element
        var max_val = Float64(0.0)
        var p = 0
        var q = 1
        for i in range(n):
            for j in range(i + 1, n):
                var abs_val = W[i * n + j]
                if abs_val < Float64(0.0):
                    abs_val = -abs_val
                if abs_val > max_val:
                    max_val = abs_val
                    p = i
                    q = j
        
        if max_val < tol:
            break
        
        # Compute rotation angle
        var app = W[p * n + p]
        var aqq = W[q * n + q]
        var apq = W[p * n + q]
        
        var tau = (aqq - app) / (Float64(2.0) * apq)
        var t: Float64
        if tau >= Float64(0.0):
            t = Float64(1.0) / (tau + sqrt(Float64(1.0) + tau * tau))
        else:
            t = Float64(-1.0) / (-tau + sqrt(Float64(1.0) + tau * tau))
        
        var c = Float64(1.0) / sqrt(Float64(1.0) + t * t)
        var s = t * c
        
        # Apply rotation to W
        for i in range(n):
            var wip = W[i * n + p]
            var wiq = W[i * n + q]
            W[i * n + p] = c * wip - s * wiq
            W[i * n + q] = s * wip + c * wiq
        for j in range(n):
            var wpj = W[p * n + j]
            var wqj = W[q * n + j]
            W[p * n + j] = c * wpj - s * wqj
            W[q * n + j] = s * wpj + c * wqj
        
        # Fix diagonal
        W[p * n + p] = app - t * apq
        W[q * n + q] = aqq + t * apq
        W[p * n + q] = Float64(0.0)
        W[q * n + p] = Float64(0.0)
        
        # Apply rotation to V64
        for i in range(n):
            var vip = V64[i * n + p]
            var viq = V64[i * n + q]
            V64[i * n + p] = c * vip - s * viq
            V64[i * n + q] = s * vip + c * viq
    
    # Extract eigenvalues and convert back to Float32
    var eig64_storage = List[Float64](capacity=n)
    for i in range(n):
        eig64_storage.append(W[i * n + i])
    var eig64 = eig64_storage.unsafe_ptr()
    
    # Sort eigenvalues in ascending order (and corresponding eigenvectors)
    for i in range(n):
        var min_idx = i
        for j in range(i + 1, n):
            if eig64[j] < eig64[min_idx]:
                min_idx = j
        if min_idx != i:
            # Swap eigenvalues
            var tmp = eig64[i]
            eig64[i] = eig64[min_idx]
            eig64[min_idx] = tmp
            # Swap eigenvector columns
            for k in range(n):
                var tmp_v = V64[k * n + i]
                V64[k * n + i] = V64[k * n + min_idx]
                V64[k * n + min_idx] = tmp_v
    
    # Write back to Float32 outputs
    for i in range(n):
        eigenvalues_ptr[i] = Float32(eig64[i])
    for i in range(n * n):
        V_ptr[i] = Float32(V64[i])
    
    # List storage freed when going out of scope
    _ = W_storage
    _ = V64_storage
    _ = eig64_storage
