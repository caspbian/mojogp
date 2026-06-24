"""LMC (Sum-of-Kronecker) Woodbury Preconditioner for Multi-Output GP.

Implements the Preconditioner trait for the LMC kernel:
    P = sum_{s=1}^{R} (A_s ⊗ L_s L_s^T) + D

where A_s = C_s C_s^T (Cholesky parameterization), L_s is the rank-r
pivoted Cholesky factor of K_X_s, and D = diag(noise_1,...,noise_T) ⊗ I_n.

Using the factorization:
    U = [C_1 ⊗ L_1 | C_2 ⊗ L_2 | ... | C_R ⊗ L_R]   (nT × K_total)
    P = U U^T + D

where K_total = R * r * T.

**P^{-1} via Woodbury identity:**
    P^{-1} v = D^{-1} v - D^{-1} U M^{-1} U^T D^{-1} v
    where M = I + U^T D^{-1} U   (K_total × K_total)

**log|P|:**
    log|P| = log|D| + log|M|
    = n * sum_t log(noise_t) + 2 * sum(log(diag(chol(M))))

**N(0, P) sampling:**
    z = D^{1/2} ε + U w,  ε ~ N(0, I_{nT}), w ~ N(0, I_{K_total})

M has Kronecker block structure:
    M_{(s1),(s2)} = δ_{s1=s2} I_{rT} + G_{s1,s2} ⊗ LtL_{s1,s2}
    where G_{s1,s2} = C_{s1}^T D^{-1} C_{s2}  (T × T)
          LtL_{s1,s2} = L_{s1}^T L_{s2}        (r × r)

Reference: Alvarez et al. (2012), "Kernels for Vector-Valued Functions"
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from memory.unsafe_pointer import alloc
from math import sqrt, log
from random import randn, seed as random_seed
from buffer import NDBuffer
from linalg.matmul import matmul as max_matmul
from collections import Optional

from .constants import float_dtype
from .preconditioner_trait import Preconditioner
from .cg_solver import kernel_copy
from .kronecker_preconditioner import (
    kernel_scale_task_blocks,
    kernel_subtract_scaled,
    _kernel_axpy,
)


# =============================================================================
# Float64 Cholesky and Solve Helpers (for the inner matrix M)
# =============================================================================

fn _cholesky_f64(
    A: UnsafePointer[Float64, MutAnyOrigin],  # Input [n × n] row-major (not modified)
    n: Int,
    L: UnsafePointer[Float64, MutAnyOrigin],  # Output lower-triangular [n × n] row-major
    jitter: Float64 = 1e-10,
) -> Bool:
    """Compute Cholesky decomposition A = L L^T in Float64.
    
    Returns True if successful. Adds jitter to diagonal for stability.
    """
    # Zero L
    for i in range(n * n):
        L[i] = Float64(0.0)
    
    for i in range(n):
        for j in range(i + 1):
            var s = A[i * n + j]
            for k in range(j):
                s -= L[i * n + k] * L[j * n + k]
            
            if i == j:
                s += jitter
                if s <= Float64(0.0):
                    return False
                L[i * n + i] = sqrt(s)
            else:
                L[i * n + j] = s / L[j * n + j]
    return True


fn _cholesky_solve_f64(
    L: UnsafePointer[Float64, MutAnyOrigin],  # Lower-triangular [n × n] row-major
    b: UnsafePointer[Float64, MutAnyOrigin],  # RHS [n] (not modified)
    x: UnsafePointer[Float64, MutAnyOrigin],  # Solution [n]
    n: Int,
):
    """Solve L L^T x = b via forward/back substitution in Float64."""
    # Forward: L y = b
    var y = alloc[Float64](n)
    for i in range(n):
        var s = b[i]
        for k in range(i):
            s -= L[i * n + k] * y[k]
        y[i] = s / L[i * n + i]
    
    # Backward: L^T x = y
    for i in range(n - 1, -1, -1):
        var s = y[i]
        for k in range(i + 1, n):
            s -= L[k * n + i] * x[k]  # L^T[i,k] = L[k,i]
        x[i] = s / L[i * n + i]
    
    y.free()


fn _cholesky_f32(
    A: UnsafePointer[Float32, MutAnyOrigin],  # Input [n × n] row-major
    n: Int,
    L: UnsafePointer[Float32, MutAnyOrigin],  # Output lower-triangular [n × n] row-major
) -> Bool:
    """Compute Cholesky decomposition A = L L^T in Float32.
    
    Returns True if successful.
    """
    for i in range(n * n):
        L[i] = Float32(0.0)
    
    for i in range(n):
        for j in range(i + 1):
            var s = A[i * n + j]
            for k in range(j):
                s -= L[i * n + k] * L[j * n + k]
            if i == j:
                if s <= Float32(0.0):
                    return False
                L[i * n + i] = sqrt(s)
            else:
                L[i * n + j] = s / L[j * n + j]
    return True


# =============================================================================
# LMCPreconditioner Struct
# =============================================================================

struct LMCPreconditioner(Preconditioner, Copyable, Movable):
    """Woodbury-based preconditioner for the LMC multi-output system.
    
    P = sum_s (A_s ⊗ L_s L_s^T) + D
    
    where A_s = C_s C_s^T, and L_s are rank-r pivoted Cholesky factors.
    
    Uses Woodbury identity with inner matrix M = I + U^T D^{-1} U.
    M is K_total × K_total where K_total = R * r * T (typically 60-180).
    Cholesky of M is precomputed in Float64 for numerical stability.
    """
    # Per-latent pivoted Cholesky factors stored contiguously [R * n * r]
    # Layout: L_s starts at offset s * n * r, column-major [n, r]
    var L_all_device: DeviceBuffer[float_dtype]
    
    # Per-latent Cholesky of A_s stored contiguously [R * T * T] on host
    # Layout: C_s starts at offset s * T * T, row-major [T, T]
    var C_all_host: HostBuffer[float_dtype]
    
    # Noise
    var D_inv_host: HostBuffer[float_dtype]    # [T] = 1/noise_t
    var D_inv_device: DeviceBuffer[float_dtype]
    var D_sqrt_host: HostBuffer[float_dtype]   # [T] = sqrt(noise_t)
    var D_sqrt_device: DeviceBuffer[float_dtype]
    
    # Inner matrix M Cholesky (Float64, on host)
    var M_chol: UnsafePointer[Float64, MutAnyOrigin]  # [K_total × K_total] lower-triangular
    var K_total: Int                     # R * r * T
    
    # Cross products L_s^T L_{s'} [R * R * r * r] on host
    # Layout: LtL_{s1,s2} at offset (s1 * R + s2) * r * r
    var LtL_all_host: UnsafePointer[Float32, MutAnyOrigin]
    
    # Precomputed
    var log_det_val: Float32
    
    # Dimensions
    var n: Int
    var rank: Int          # Same r for all latents
    var num_tasks: Int
    var num_latents: Int
    var max_num_cols: Int
    
    # Work buffers
    var w_dinv: DeviceBuffer[float_dtype]       # [nT * max_cols] D^{-1} scaled
    var temp_Lr: DeviceBuffer[float_dtype]      # [r] temp for L^T products
    var temp_Ln: DeviceBuffer[float_dtype]      # [n] temp for L products
    var w_result: DeviceBuffer[float_dtype]     # [nT * max_cols] accumulated U @ z
    
    fn __init__(
        out self,
        ctx: DeviceContext,
        L_all_device: DeviceBuffer[float_dtype],  # [R * n * r] contiguous
        C_all_host: HostBuffer[float_dtype],       # [R * T * T] contiguous
        noise_host: HostBuffer[float_dtype],       # [T]
        n: Int,
        rank: Int,
        num_tasks: Int,
        num_latents: Int,
        max_num_cols: Int = 16,
    ) raises:
        """Build LMCPreconditioner.
        
        Steps:
        1. Compute L_s^T L_{s'} for all pairs via GPU matmul
        2. Compute G_{s1,s2} = C_{s1}^T D^{-1} C_{s2} on CPU
        3. Assemble M = I + sum G ⊗ LtL block structure
        4. Cholesky of M in Float64
        5. Compute log|P| = log|D| + 2 * sum(log(diag(chol_M)))
        
        Args:
            ctx: GPU device context.
            L_all_device: All pivoted Cholesky factors [R * n * r] contiguous.
            C_all_host: All Cholesky of A_s [R * T * T] contiguous.
            noise_host: Per-task noise [T].
            n: Number of data points.
            rank: Pivoted Cholesky rank (same for all latents).
            num_tasks: Number of tasks T.
            num_latents: Number of latents R.
            max_num_cols: Max CG columns for work buffers.
        """
        self.L_all_device = L_all_device
        self.n = n
        self.rank = rank
        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.max_num_cols = max_num_cols
        var T = num_tasks
        var R = num_latents
        var r = rank
        var nT = n * T
        self.K_total = R * r * T
        
        # Copy C_all
        self.C_all_host = HostBuffer[float_dtype](ctx, R * T * T)
        for i in range(R * T * T):
            self.C_all_host.unsafe_ptr()[i] = C_all_host.unsafe_ptr()[i]
        
        # Compute D^{-1} and D^{1/2}
        self.D_inv_host = HostBuffer[float_dtype](ctx, T)
        self.D_sqrt_host = HostBuffer[float_dtype](ctx, T)
        for t in range(T):
            var noise_t = noise_host.unsafe_ptr()[t]
            self.D_inv_host.unsafe_ptr()[t] = Float32(1.0) / noise_t
            self.D_sqrt_host.unsafe_ptr()[t] = sqrt(noise_t)
        
        self.D_inv_device = ctx.enqueue_create_buffer[float_dtype](T)
        self.D_sqrt_device = ctx.enqueue_create_buffer[float_dtype](T)
        var D_inv_host_buf = ctx.enqueue_create_host_buffer[float_dtype](T)
        var D_sqrt_host_buf = ctx.enqueue_create_host_buffer[float_dtype](T)
        for t in range(T):
            D_inv_host_buf[t] = self.D_inv_host.unsafe_ptr()[t]
            D_sqrt_host_buf[t] = self.D_sqrt_host.unsafe_ptr()[t]
        ctx.enqueue_copy(dst_buf=self.D_inv_device, src_buf=D_inv_host_buf)
        ctx.enqueue_copy(dst_buf=self.D_sqrt_device, src_buf=D_sqrt_host_buf)
        
        # =====================================================================
        # Step 1: Compute L_s^T L_{s'} for all pairs via GPU matmul
        # =====================================================================
        self.LtL_all_host = alloc[Float32](R * R * r * r)
        
        var opt_ctx = Optional[DeviceContext](ctx)
        
        for s1 in range(R):
            for s2 in range(R):
                # L_s1^T @ L_s2: [r, n] @ [n, r] → [r, r]
                # In row-major: L_s1 is [r, n], L_s2 is [r, n]
                # Want C[r,r] = L_s1[r,n] @ L_s2[r,n]^T → transpose_b=True
                # But actually L_s is column-major [n, r], which in row-major memory is [r, n]
                # L_s1^T @ L_s2 = L_s1[r,n] @ L_s2[r,n]^T... 
                # No: column-major [n,r] means element (i,k) at index k*n+i
                # As NDBuffer with shape (r,n) row-major, this reads the transpose
                # L_s^T is row-major [r, n], so NDBuffer(ptr, (r, n)) IS L_s^T
                # L_s1^T @ L_s2 = NDBuffer[r,n] @ NDBuffer[r,n]^T = matmul[transpose_b=True]
                
                var LtL_device = ctx.enqueue_create_buffer[float_dtype](r * r)
                var L_s1_ndbuf = NDBuffer[DType.float32, 2](
                    self.L_all_device.unsafe_ptr().offset(s1 * n * r), (r, n)
                )
                var L_s2_ndbuf = NDBuffer[DType.float32, 2](
                    self.L_all_device.unsafe_ptr().offset(s2 * n * r), (r, n)
                )
                var LtL_ndbuf = NDBuffer[DType.float32, 2](LtL_device.unsafe_ptr(), (r, r))
                max_matmul[transpose_b=True, target="gpu"](LtL_ndbuf, L_s1_ndbuf, L_s2_ndbuf, opt_ctx)
                
                # Copy to host
                var LtL_host_buf = ctx.enqueue_create_host_buffer[float_dtype](r * r)
                ctx.enqueue_copy(dst_buf=LtL_host_buf, src_buf=LtL_device)
                ctx.synchronize()
                
                var offset = (s1 * R + s2) * r * r
                for i in range(r * r):
                    self.LtL_all_host[offset + i] = LtL_host_buf.unsafe_ptr()[i]
                
                _ = LtL_device
        
        # =====================================================================
        # Step 2: Compute G_{s1,s2} = C_{s1}^T D^{-1} C_{s2} on CPU
        # =====================================================================
        # G is [R, R, T, T] — for each pair (s1, s2), a T×T matrix
        var G_all = alloc[Float64](R * R * T * T)
        
        var C = self.C_all_host.unsafe_ptr()
        for s1 in range(R):
            for s2 in range(R):
                for j1 in range(T):
                    for j2 in range(T):
                        var val = Float64(0.0)
                        for t in range(T):
                            # C_{s1}^T[j1, t] = C_{s1}[t, j1]
                            # C_{s2}[t, j2]
                            val += Float64(C[s1 * T * T + t * T + j1]) * Float64(C[s2 * T * T + t * T + j2]) * Float64(self.D_inv_host.unsafe_ptr()[t])
                        G_all[(s1 * R + s2) * T * T + j1 * T + j2] = val
        
        # =====================================================================
        # Step 3: Assemble M = I + sum of G ⊗ LtL blocks
        # =====================================================================
        var Kt = self.K_total
        var M = alloc[Float64](Kt * Kt)
        
        # Zero M
        for i in range(Kt * Kt):
            M[i] = Float64(0.0)
        
        # Add identity
        for i in range(Kt):
            M[i * Kt + i] = Float64(1.0)
        
        # Add G ⊗ LtL blocks
        # Global index mapping: (s, j, k) → s * (r * T) + j * r + k
        for s1 in range(R):
            for s2 in range(R):
                var G_offset = (s1 * R + s2) * T * T
                var LtL_offset = (s1 * R + s2) * r * r
                
                for j1 in range(T):
                    for j2 in range(T):
                        var g_val = G_all[G_offset + j1 * T + j2]
                        
                        for k1 in range(r):
                            for k2 in range(r):
                                var ltl_val = Float64(self.LtL_all_host[LtL_offset + k1 * r + k2])
                                
                                var row = s1 * (r * T) + j1 * r + k1
                                var col = s2 * (r * T) + j2 * r + k2
                                M[row * Kt + col] += g_val * ltl_val
        
        # =====================================================================
        # Step 4: Cholesky of M in Float64
        # =====================================================================
        self.M_chol = alloc[Float64](Kt * Kt)
        var chol_ok = _cholesky_f64(M, Kt, self.M_chol)
        if not chol_ok:
            # Try with more jitter
            chol_ok = _cholesky_f64(M, Kt, self.M_chol, jitter=Float64(1e-6))
            if not chol_ok:
                print("WARNING: LMC preconditioner inner matrix Cholesky failed")
                # Fall back to identity (M_chol = I)
                for i in range(Kt * Kt):
                    self.M_chol[i] = Float64(0.0)
                for i in range(Kt):
                    self.M_chol[i * Kt + i] = Float64(1.0)
        
        # =====================================================================
        # Step 5: Compute log|P|
        # =====================================================================
        # log|P| = log|D| + log|M|
        # log|D| = n * sum_t log(noise_t)
        # log|M| = 2 * sum log(diag(chol_M))
        self.log_det_val = Float32(0.0)
        for t in range(T):
            self.log_det_val += Float32(n) * log(noise_host.unsafe_ptr()[t])
        
        for i in range(Kt):
            var diag_val = self.M_chol[i * Kt + i]
            if diag_val > Float64(0.0):
                self.log_det_val += Float32(Float64(2.0) * log(diag_val))
        
        # Clean up
        M.free()
        G_all.free()
        
        # =====================================================================
        # Step 6: Allocate work buffers
        # =====================================================================
        self.w_dinv = ctx.enqueue_create_buffer[float_dtype](nT * max_num_cols)
        self.temp_Lr = ctx.enqueue_create_buffer[float_dtype](r)
        self.temp_Ln = ctx.enqueue_create_buffer[float_dtype](n)
        self.w_result = ctx.enqueue_create_buffer[float_dtype](nT * max_num_cols)
        
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
        """Apply P^{-1} @ v using Woodbury identity.
        
        P^{-1} v = D^{-1} v - D^{-1} U M^{-1} U^T D^{-1} v
        
        Steps:
        1. w = D^{-1} v (per-task scaling on GPU)
        2. y = U^T w (on CPU via GPU L^T matmuls + CPU C^T mixing)
        3. z = M^{-1} y (Cholesky solve on CPU)
        4. result = U z (on GPU via L matmuls + C mixing)
        5. out = w - result (GPU)
        """
        var nn = self.n
        var T = self.num_tasks
        var R = self.num_latents
        var r = self.rank
        var nT = nn * T
        var Kt = self.K_total
        alias BLOCK = 256
        var opt_ctx = Optional[DeviceContext](ctx)
        
        # D^{-1} as per-task scale: 1/noise_t
        # Need a DeviceBuffer with 1/noise for the task scaling kernel
        # We already have D_inv_device
        
        for c in range(num_cols):
            # =================================================================
            # Step 1: w = D^{-1} v (per-task scaling)
            # =================================================================
            # Process this column: v[c*nT..(c+1)*nT] → w_dinv[c*nT..(c+1)*nT]
            var total_elements = nT
            var num_blocks = (total_elements + BLOCK - 1) // BLOCK
            ctx.enqueue_function[kernel_scale_task_blocks](
                self.w_dinv.unsafe_ptr().offset(c * nT), v_ptr.offset(c * nT),
                self.D_inv_device.unsafe_ptr(),
                nn, T, nT, 1,  # Process as 1-column block at correct offset
                grid_dim=(num_blocks,), block_dim=(BLOCK,),
            )
            
            # =================================================================
            # Step 2: y = U^T w on CPU
            # =================================================================
            # For each latent s and task t: compute L_s^T @ w_t → temp [r]
            # Then mix with C_s^T: y_s[j*r+k] = sum_t C_s[t,j] * (L_s^T @ w_t)[k]
            
            # First, compute L_s^T @ w_t for all (s, t) pairs via GPU matmul
            # Store results in a host buffer [R * T * r]
            var Lt_w_host = alloc[Float32](R * T * r)
            
            var L_all_ptr = self.L_all_device.unsafe_ptr()
            var w_dinv_ptr = self.w_dinv.unsafe_ptr()
            
            for s in range(R):
                var L_s_ndbuf = NDBuffer[DType.float32, 2](
                    L_all_ptr.offset(s * nn * r), (r, nn)
                )
                
                for t in range(T):
                    # Extract w_t for this column into temp_Ln
                    var w_t_offset = c * nT + t * nn
                    ctx.enqueue_function[kernel_copy](
                        self.temp_Ln.unsafe_ptr(), w_dinv_ptr.offset(w_t_offset), nn,
                        grid_dim=((nn + 255) // 256,), block_dim=(256,),
                    )
                    
                    # L_s^T @ w_t: [1, r] = [1, n] @ [r, n]^T = [1, n] @ L_s
                    # Actually: L_s is NDBuffer(ptr, (r, n)) which is L_s^T in row-major
                    # So L_s^T @ w_t = L_s_ndbuf @ w_t^T... no.
                    # L_s^T is [r, n], w_t is [n, 1]
                    # result [r, 1] = L_s^T [r, n] @ w_t [n, 1]
                    # In row-major NDBuffer: result [1, r] = w_t^T [1, n] @ L_s^T^T [n, r]
                    # = w_t^T [1, n] @ L_s_ndbuf^T [n, r] → transpose_b=True
                    var w_t_ndbuf = NDBuffer[DType.float32, 2](self.temp_Ln.unsafe_ptr(), (1, nn))
                    var result_ndbuf = NDBuffer[DType.float32, 2](self.temp_Lr.unsafe_ptr(), (1, r))
                    max_matmul[transpose_b=True, target="gpu"](result_ndbuf, w_t_ndbuf, L_s_ndbuf, opt_ctx)
                    
                    # Copy result to host
                    var temp_Lr_host = ctx.enqueue_create_host_buffer[float_dtype](r)
                    ctx.enqueue_copy(dst_buf=temp_Lr_host, src_buf=self.temp_Lr)
                    ctx.synchronize()
                    
                    var host_offset = (s * T + t) * r
                    for k in range(r):
                        Lt_w_host[host_offset + k] = temp_Lr_host.unsafe_ptr()[k]
            
            # Mix with C_s^T to get y = U^T w
            var y = alloc[Float64](Kt)
            var C = self.C_all_host.unsafe_ptr()
            
            for s in range(R):
                for j in range(T):
                    for k in range(r):
                        var val = Float64(0.0)
                        for t in range(T):
                            # C_s^T[j,t] = C_s[t,j]
                            val += Float64(C[s * T * T + t * T + j]) * Float64(Lt_w_host[(s * T + t) * r + k])
                        y[s * (r * T) + j * r + k] = val
            
            # =================================================================
            # Step 3: z = M^{-1} y (Cholesky solve on CPU)
            # =================================================================
            var z = alloc[Float64](Kt)
            _cholesky_solve_f64(self.M_chol, y, z, Kt)
            
            # =================================================================
            # Step 4: result = U z (on GPU)
            # =================================================================
            # For each task t: result_t = sum_s sum_j C_s[t,j] * L_s @ z_s[j*r..(j+1)*r]
            
            # Zero the result for this column
            var num_blocks_nT = (nT + BLOCK - 1) // BLOCK
            ctx.enqueue_function[_kernel_zero](
                self.w_result.unsafe_ptr().offset(c * nT), nT,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )
            
            for s in range(R):
                for j in range(T):
                    # Copy z_s[j*r..(j+1)*r] to GPU temp_Lr
                    var z_host_buf = ctx.enqueue_create_host_buffer[float_dtype](r)
                    for k in range(r):
                        z_host_buf[k] = Float32(z[s * (r * T) + j * r + k])
                    ctx.enqueue_copy(dst_buf=self.temp_Lr, src_buf=z_host_buf)
                    
                    # L_s @ z_sj: [1, n] = [1, r] @ [r, n]
                    var z_ndbuf = NDBuffer[DType.float32, 2](self.temp_Lr.unsafe_ptr(), (1, r))
                    var L_s_ndbuf = NDBuffer[DType.float32, 2](
                        L_all_ptr.offset(s * nn * r), (r, nn)
                    )
                    var Lz_ndbuf = NDBuffer[DType.float32, 2](self.temp_Ln.unsafe_ptr(), (1, nn))
                    max_matmul[target="gpu"](Lz_ndbuf, z_ndbuf, L_s_ndbuf, opt_ctx)
                    
                    # Accumulate C_s[t,j] * L_s @ z_sj into result for each task t
                    for t in range(T):
                        var c_val = C[s * T * T + t * T + j]
                        if c_val != Float32(0.0):
                            var result_offset = c * nT + t * nn
                            var axpy_blocks = (nn + BLOCK - 1) // BLOCK
                            ctx.enqueue_function[_kernel_axpy](
                                self.w_result.unsafe_ptr().offset(result_offset),
                                self.temp_Ln.unsafe_ptr(),
                                c_val,
                                nn,
                                grid_dim=(axpy_blocks,), block_dim=(BLOCK,),
                            )
            
            # =================================================================
            # Step 5: out = w - D^{-1} result
            # =================================================================
            # Woodbury gives P^{-1}v = D^{-1}v - D^{-1} U M^{-1} U^T D^{-1}v.
            # Step 4 computed only U M^{-1} U^T D^{-1}v, so apply the left
            # D^{-1} task scaling before subtracting. For T=1,R=1 this reduces
            # to the single-output pivoted-Cholesky preconditioner formula.
            ctx.enqueue_function[kernel_scale_task_blocks](
                self.w_result.unsafe_ptr().offset(c * nT),
                self.w_result.unsafe_ptr().offset(c * nT),
                self.D_inv_device.unsafe_ptr(),
                nn, T, nT, 1,
                grid_dim=(num_blocks_nT,), block_dim=(BLOCK,),
            )

            var total_col = nT
            var nb = (total_col + BLOCK - 1) // BLOCK
            ctx.enqueue_function[kernel_subtract_scaled](
                out_ptr.offset(c * nT),
                self.w_dinv.unsafe_ptr().offset(c * nT),
                self.w_result.unsafe_ptr().offset(c * nT),
                Float32(1.0),
                total_col,
                grid_dim=(nb,), block_dim=(BLOCK,),
            )
            
            # Clean up per-column allocations
            Lt_w_host.free()
            y.free()
            z.free()
        
        if sync:
            ctx.synchronize()
    
    fn log_det(self, ctx: DeviceContext) raises -> Float32:
        """Return precomputed log|P|.
        
        log|P| = n * sum_t log(noise_t) + 2 * sum(log(diag(chol_M)))
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
        
        P = U U^T + D, so z ~ N(0, P) can be sampled as:
        z = D^{1/2} ε + U w,  ε ~ N(0, I_{nT}), w ~ N(0, I_{K_total})
        
        U w = sum_s (C_s ⊗ L_s) w_s
        For each task t: (U w)_t = sum_s sum_j C_s[t,j] * L_s @ w_s[j*r+0..j*r+r]
        """
        var nn = self.n
        var T = self.num_tasks
        var R = self.num_latents
        var r = self.rank
        var nT = nn * T
        var Kt = self.K_total
        alias BLOCK = 256
        var opt_ctx = Optional[DeviceContext](ctx)
        
        random_seed(Int(seed_val))
        
        # Generate ε ~ N(0, I_{nT * num_probes}) on host
        var z_noise_host = ctx.enqueue_create_host_buffer[float_dtype](nT * num_probes)
        randn[DType.float32](z_noise_host.unsafe_ptr(), nT * num_probes)
        
        # Generate w ~ N(0, I_{K_total * num_probes}) on host
        var w_rank_host = ctx.enqueue_create_host_buffer[float_dtype](Kt * num_probes)
        randn[DType.float32](w_rank_host.unsafe_ptr(), Kt * num_probes)
        
        # Copy noise part to device
        var z_noise_device = ctx.enqueue_create_buffer[float_dtype](nT * num_probes)
        ctx.enqueue_copy(dst_buf=z_noise_device, src_buf=z_noise_host)
        ctx.synchronize()
        
        # Initialize output with D^{1/2} ε
        var out_ptr = out_device.unsafe_ptr()
        var num_blocks = (nT * num_probes + BLOCK - 1) // BLOCK
        ctx.enqueue_function[kernel_scale_task_blocks](
            out_ptr, z_noise_device.unsafe_ptr(),
            self.D_sqrt_device.unsafe_ptr(),
            nn, T, nT, num_probes,
            grid_dim=(num_blocks,), block_dim=(BLOCK,),
        )
        
        # Add U w contribution
        # For each probe p, latent s, task pair (j, t):
        #   out_t += C_s[t,j] * L_s @ w_s[j*r..(j+1)*r, p]
        
        var L_all_ptr = self.L_all_device.unsafe_ptr()
        var C = self.C_all_host.unsafe_ptr()
        var w_ptr = w_rank_host.unsafe_ptr()
        
        # Temp GPU buffer for L_s @ w piece
        var temp_Lw = ctx.enqueue_create_buffer[float_dtype](nn)
        var temp_w_r = ctx.enqueue_create_buffer[float_dtype](r)
        
        for p in range(num_probes):
            for s in range(R):
                for j in range(T):
                    # w_s[j*r..(j+1)*r, p]: from w_rank_host
                    # Layout: w for probe p, latent s, task j, rank k
                    # Linear index: (s * (r * T) + j * r + k) * num_probes + p
                    # Wait, w is [K_total * num_probes] column-major
                    # Element (i, p) at p * Kt + i
                    # i = s * (r * T) + j * r + k
                    
                    # Copy w_sj to GPU
                    var w_sj_host = ctx.enqueue_create_host_buffer[float_dtype](r)
                    for k in range(r):
                        var idx = p * Kt + s * (r * T) + j * r + k
                        w_sj_host[k] = w_ptr[idx]
                    ctx.enqueue_copy(dst_buf=temp_w_r, src_buf=w_sj_host)
                    
                    # L_s @ w_sj: [1, n] = [1, r] @ [r, n]
                    var w_ndbuf = NDBuffer[DType.float32, 2](temp_w_r.unsafe_ptr(), (1, r))
                    var L_s_ndbuf = NDBuffer[DType.float32, 2](
                        L_all_ptr.offset(s * nn * r), (r, nn)
                    )
                    var Lw_ndbuf = NDBuffer[DType.float32, 2](temp_Lw.unsafe_ptr(), (1, nn))
                    max_matmul[target="gpu"](Lw_ndbuf, w_ndbuf, L_s_ndbuf, opt_ctx)
                    
                    # Accumulate C_s[t,j] * L_s @ w_sj for each task t
                    for t in range(T):
                        var c_val = C[s * T * T + t * T + j]
                        if c_val != Float32(0.0):
                            var out_offset = p * nT + t * nn
                            var axpy_blocks = (nn + BLOCK - 1) // BLOCK
                            ctx.enqueue_function[_kernel_axpy](
                                out_ptr.offset(out_offset),
                                temp_Lw.unsafe_ptr(),
                                c_val,
                                nn,
                                grid_dim=(axpy_blocks,), block_dim=(BLOCK,),
                            )
        
        ctx.synchronize()
        
        # Keep buffers alive
        _ = z_noise_device
        _ = temp_Lw
        _ = temp_w_r
    
    fn get_rank(self) -> Int:
        """Return the rank of the pivoted Cholesky factors."""
        return self.rank
    
    fn __del__(deinit self):
        """Clean up manually allocated memory."""
        self.M_chol.free()
        self.LtL_all_host.free()


# =============================================================================
# Helper GPU Kernel
# =============================================================================

fn _kernel_zero(
    ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    """Zero a buffer."""
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx >= UInt(n):
        return
    ptr[idx] = Float32(0.0)
