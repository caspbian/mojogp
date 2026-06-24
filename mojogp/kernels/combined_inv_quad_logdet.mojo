"""Core BBMM Algorithm — Combined Inverse Quadratic Form and Log Determinant.

This module implements the core BBMM (Black-Box Matrix-Matrix) algorithm:
computing BOTH inv_quad (y^T K^{-1} y) AND log_det (log|K|) in a single pass,
plus all kernel parameter gradients.

Key insight: CG iterations ARE Lanczos iterations! The tridiagonal matrix
needed for log det estimation can be extracted from CG coefficients.

Functions:
- batched_cg_unified: Unified batched CG solver with tridiagonal tracking
- bbmm_with_precond: Generic BBMM with pre-built preconditioner
- bbmm_unified: Wrapper that builds PivotedCholesky preconditioner internally

Types and GPU kernels are in separate modules:
- bbmm_types: CGBufferPool, CGResultWithTridiag, UnifiedBBMMResult
- bbmm_gpu_kernels: GPU kernels for dot products, column ops, and scaling

Reference: GPyTorch's LinearCG implementation
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from gpu.primitives.warp import sum as warp_sum
from gpu.globals import WARP_SIZE
from memory import UnsafePointer
from math import sqrt, log, isnan, isinf
from random import random_float64


from .cg_solver import (
    kernel_dot_batched,
    kernel_axpy_batched,
    kernel_scale_add_batched,
    kernel_copy,
    kernel_init_zero_and_copy,
    kernel_fill_constant,
    kernel_subtract_inplace,
    kernel_compute_alpha,
    kernel_beta_and_copy_fused,
    kernel_cg_update_fused,
    kernel_compute_residual_norms_sq,
    kernel_compute_mean_residual_norm,
)
from .matvec_provider import MatvecProvider, MaterializedProvider, MatrixFreeProvider
from .composable_kernel import ComposableKernel
from .composite_provider import CompositeProvider, MaterializedCompositeProvider
from .gradient_provider import ForwardProvider, GradientProvider
from .preconditioner_trait import Preconditioner
from .lanczos import compute_logdet_from_tridiag_batched
from .pivoted_cholesky import PivotedCholeskyPrecond, build_pivoted_cholesky_precond_gpu, build_pivoted_cholesky_precond_unified, apply_pivoted_cholesky_precond, apply_pivoted_cholesky_precond_gpu, compute_precond_log_det, sample_from_preconditioner_gpu, sample_from_preconditioner_gpu_pooled, normalize_columns_gpu, kernel_generate_rademacher, kernel_gpu_gaussian
from .generic_matvec import kernel_fused_gradient_only_ard
from .kernel_functions import rbf_fused_gradient_ard, matern_fused_gradient_ard
from gpu.profiler import ProfileBlock
from time import perf_counter_ns
from .constants import PROFILING

# Re-export types from bbmm_types so existing importers don't break
from .bbmm_types import CGBufferPool, CGResultWithTridiag, UnifiedBBMMResult

# Re-export GPU kernels from bbmm_gpu_kernels so existing importers don't break
from .bbmm_gpu_kernels import (
    kernel_dot_single_vs_strided,
    kernel_dot_batched_vs_strided,
    kernel_copy_column,
    kernel_copy_columns,
    kernel_extract_columns_range,
    kernel_scale_columns_by_norms,
    scale_columns_by_norms,
)

alias float_dtype = DType.float32

alias PI = Float32(3.14159265358979323846)


# =============================================================================
# Unified CG Solver with GradientProvider Trait
# =============================================================================

fn batched_cg_unified[P: ForwardProvider, Q: Preconditioner](
    provider: P,
    rhs_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    max_iter: Int,
    max_tridiag_iter: Int,
    tol: Float32,
    precond: Q,
    mut pool: CGBufferPool,
    use_warm_start: Bool = False,
    use_preconditioner: Bool = True,
) raises -> CGResultWithTridiag:
    """Unified batched CG solver for any ForwardProvider with any Preconditioner.
    
    This function unifies the 3 hot-path CG solvers:
    - batched_cg_with_pivoted_cholesky[T: MatvecProvider]
    - batched_cg_with_pivoted_cholesky_composite[DIM, K]
    - batched_cg_with_pivoted_cholesky_materialized_composite[DIM, K]
    
    Generic over both the ForwardProvider (P) and the Preconditioner (Q),
    allowing use with PivotedCholeskyPrecond (single-output) or
    KroneckerPreconditioner (multi-output).
    
    Key features:
    - Always uses CGBufferPool (no fresh allocations)
    - Uses GPU-optimized sync pattern (minimal syncs)
    - Uses full GPyTorch-aligned tridiagonal builder (alpha=0 handling, early-stop)
    - Checks convergence every 10 iterations AND prevents early exit before max_tridiag_iter
    
    Args:
        provider: Any ForwardProvider (isotropic, ARD, or composite)
        rhs_ptr: Batched RHS [n * num_cols] ON DEVICE (column-major)
        n: Problem size
        num_cols: Number of RHS columns
        max_iter: Maximum CG iterations
        max_tridiag_iter: Max iterations to track for tridiagonal
        tol: Convergence tolerance
        precond: Any Preconditioner (PivotedCholeskyPrecond, KroneckerPreconditioner, etc.)
        pool: Reusable buffer pool
        
    Returns:
        CGResultWithTridiag containing solution and tridiagonal matrices
    """
    var ctx = provider.get_ctx()
    
    # Use pool buffers
    var x = pool.x
    var r = pool.r
    var z = pool.z
    var p = pool.p
    var Ap = pool.Ap
    var rz_old = pool.rz_old
    var rz_new = pool.rz_new
    var pAp = pool.pAp
    var cg_alpha = pool.cg_alpha
    var cg_beta = pool.cg_beta
    var residual_norms_sq = pool.residual_norms_sq
    var max_residual = pool.max_residual
    var max_residual_host = pool.max_residual_host
    
    # Tridiagonal tracking buffers
    var all_alphas = pool.all_alphas
    var all_betas = pool.all_betas
    var all_alphas_host = pool.all_alphas_host
    var all_betas_host = pool.all_betas_host
    
    # Initialize x and r
    var t_cg_init_start = perf_counter_ns()
    if use_warm_start:
        # Solution recycling: x already has warm-start values in pool.x
        # (column 0 = previous alpha re-normalized, columns 1..P = zeros)
        # Compute r = rhs - A @ x
        # First: r = rhs (copy)
        ctx.enqueue_function[kernel_copy](
            r.unsafe_ptr(), rhs_ptr, n * num_cols,
            grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
        )
        # Ap = A @ x (reuse Ap buffer as temp for A @ x_init)
        provider.forward_matvec(Ap.unsafe_ptr(), x.unsafe_ptr(), num_cols)
        # r -= Ap  (r = rhs - A @ x_init)
        ctx.enqueue_function[kernel_subtract_inplace](
            r.unsafe_ptr(), Ap.unsafe_ptr(), n * num_cols,
            grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
        )
    else:
        # Standard init: x = 0, r = rhs
        ctx.enqueue_function[kernel_init_zero_and_copy](
            x.unsafe_ptr(), r.unsafe_ptr(), rhs_ptr, n * num_cols,
            grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
        )
    ctx.synchronize()
    
    # Apply preconditioner: z = P^{-1} @ r.  In the no-preconditioner case,
    # use r directly instead of copying r -> z every CG iteration.
    if use_preconditioner:
        precond.apply_precond(ctx, r.unsafe_ptr(), z.unsafe_ptr(), n, num_cols, sync=False)
        ctx.synchronize()

        # p = z
        ctx.enqueue_function[kernel_copy](
            p.unsafe_ptr(), z.unsafe_ptr(), n * num_cols,
            grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
        )

        # rz_old = r^T @ z
        ctx.enqueue_function[kernel_dot_batched](
            r.unsafe_ptr(), z.unsafe_ptr(), rz_old.unsafe_ptr(),
            n, num_cols,
            grid_dim=(num_cols, 1), block_dim=(256, 1)
        )
    else:
        # p = r and rz_old = r^T @ r for ordinary CG.
        ctx.enqueue_function[kernel_copy](
            p.unsafe_ptr(), r.unsafe_ptr(), n * num_cols,
            grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
        )
        ctx.enqueue_function[kernel_dot_batched](
            r.unsafe_ptr(), r.unsafe_ptr(), rz_old.unsafe_ptr(),
            n, num_cols,
            grid_dim=(num_cols, 1), block_dim=(256, 1)
        )
    ctx.synchronize()
    
    if PROFILING:
        var t_cg_init_end = perf_counter_ns()
        print("@ CG_init_total", -Int(t_cg_init_end - t_cg_init_start))
    
    # CG iterations
    var num_iterations = 0
    var converged = False
    var tridiag_size = 0
    
    # Initialize tridiagonal lists
    var tridiag_diag = List[List[Float32]]()
    var tridiag_offdiag = List[List[Float32]]()
    if max_tridiag_iter > 0:
        for _ in range(num_cols):
            tridiag_diag.append(List[Float32]())
            tridiag_offdiag.append(List[Float32]())
    
    var t_cg_loop_start = perf_counter_ns()
    
    for iter in range(max_iter):
        num_iterations = iter + 1
        
        # Deep profiling for CG iterations 0 and 20 only (to avoid sync overhead)
        var profile_this_iter = (iter == 0 or iter == 20) and PROFILING
        
        # Ap = K @ p (using GradientProvider's forward_matvec)
        if profile_this_iter:
            ctx.synchronize()
        with ProfileBlock[False](""):
            # Dummy block — real profiling below
            pass
        
        var t_matvec_start = perf_counter_ns() if profile_this_iter else UInt(0)
        provider.forward_matvec(Ap.unsafe_ptr(), p.unsafe_ptr(), num_cols)
        if profile_this_iter:
            ctx.synchronize()
            var t_matvec_end = perf_counter_ns()
            print("@ CG_iter" + String(iter) + "_forward_matvec", -Int(t_matvec_end - t_matvec_start))
        
        # pAp = p^T @ Ap
        ctx.enqueue_function[kernel_dot_batched](
            p.unsafe_ptr(), Ap.unsafe_ptr(), pAp.unsafe_ptr(),
            n, num_cols,
            grid_dim=(num_cols, 1), block_dim=(256, 1)
        )
        
        # alpha = rz_old / pAp
        ctx.enqueue_function[kernel_compute_alpha](
            rz_old.unsafe_ptr(), pAp.unsafe_ptr(), cg_alpha.unsafe_ptr(), num_cols,
            grid_dim=((num_cols + 255) // 256,), block_dim=(256,)
        )
        
        # x = x + alpha * p, r = r - alpha * Ap (fused kernel)
        ctx.enqueue_function[kernel_cg_update_fused](
            cg_alpha.unsafe_ptr(), p.unsafe_ptr(), Ap.unsafe_ptr(), x.unsafe_ptr(),
            r.unsafe_ptr(), n, num_cols,
            grid_dim=((n + 15) // 16, (num_cols + 15) // 16), block_dim=(16, 16)
        )
        
        # z = P^{-1} @ r when preconditioning is enabled.  Otherwise keep the
        # no-preconditioner route on r directly to avoid a full-buffer copy.
        var t_precond_start = perf_counter_ns() if profile_this_iter else UInt(0)
        if use_preconditioner:
            precond.apply_precond(ctx, r.unsafe_ptr(), z.unsafe_ptr(), n, num_cols, sync=False)
        if profile_this_iter:
            ctx.synchronize()
            var t_precond_end = perf_counter_ns()
            print("@ CG_iter" + String(iter) + "_precond_apply", -Int(t_precond_end - t_precond_start))

        # rz_new = r^T @ z for PCG, or r^T @ r for ordinary CG.
        if use_preconditioner:
            ctx.enqueue_function[kernel_dot_batched](
                r.unsafe_ptr(), z.unsafe_ptr(), rz_new.unsafe_ptr(),
                n, num_cols,
                grid_dim=(num_cols, 1), block_dim=(256, 1)
            )
        else:
            ctx.enqueue_function[kernel_dot_batched](
                r.unsafe_ptr(), r.unsafe_ptr(), rz_new.unsafe_ptr(),
                n, num_cols,
                grid_dim=(num_cols, 1), block_dim=(256, 1)
            )
        
        # beta = rz_new / rz_old, copy rz_new to rz_old (fused kernel)
        ctx.enqueue_function[kernel_beta_and_copy_fused](
            rz_old.unsafe_ptr(), rz_new.unsafe_ptr(), cg_beta.unsafe_ptr(), num_cols,
            grid_dim=((num_cols + 31) // 32,), block_dim=(32,)
        )
        
        # p = z + beta * p for PCG, or r + beta * p for ordinary CG.
        if use_preconditioner:
            ctx.enqueue_function[kernel_scale_add_batched](
                cg_beta.unsafe_ptr(), p.unsafe_ptr(), z.unsafe_ptr(), n, num_cols,
                grid_dim=((n + 15) // 16, (num_cols + 15) // 16),
                block_dim=(16, 16)
            )
        else:
            ctx.enqueue_function[kernel_scale_add_batched](
                cg_beta.unsafe_ptr(), p.unsafe_ptr(), r.unsafe_ptr(), n, num_cols,
                grid_dim=((n + 15) // 16, (num_cols + 15) // 16),
                block_dim=(16, 16)
            )
        
        # Profile full CG iteration (dots + updates + tridiag) for selected iterations
        if profile_this_iter:
            ctx.synchronize()
            var t_rest_end = perf_counter_ns()
            print("@ CG_iter" + String(iter) + "_dots_updates", -Int(t_rest_end - (t_precond_start if t_precond_start > UInt(0) else t_matvec_start)))
        
        # Store alpha and beta for tridiagonal construction
        if iter < max_tridiag_iter:
            ctx.enqueue_function[kernel_copy](
                all_alphas.unsafe_ptr() + iter * num_cols, cg_alpha.unsafe_ptr(), num_cols,
                grid_dim=((num_cols + 255) // 256,), block_dim=(256,)
            )
            ctx.enqueue_function[kernel_copy](
                all_betas.unsafe_ptr() + iter * num_cols, cg_beta.unsafe_ptr(), num_cols,
                grid_dim=((num_cols + 255) // 256,), block_dim=(256,)
            )
            tridiag_size = iter + 1
        
        # Check convergence every 10 iterations after initial CG progress.
        # RHS columns were normalized before CG, so the GPyTorch-aligned stop
        # rule is the mean residual norm across columns against the absolute
        # tolerance, not the max residual relative to the initial residual.
        if iter >= 10 and (((iter + 1) % 10) == 0 or iter == max_iter - 1):
            ctx.enqueue_function[kernel_compute_residual_norms_sq](
                residual_norms_sq.unsafe_ptr(), r.unsafe_ptr(), n, num_cols,
                grid_dim=(num_cols,), block_dim=(256,)
            )
            ctx.enqueue_function[kernel_compute_mean_residual_norm](
                max_residual.unsafe_ptr(), residual_norms_sq.unsafe_ptr(), num_cols,
                grid_dim=(1,), block_dim=(1,)
            )
            ctx.enqueue_copy(dst_buf=max_residual_host, src_buf=max_residual)
            ctx.synchronize()

            if max_residual_host[0] < tol:
                # Don't exit early if still collecting tridiagonal entries (GPyTorch behavior)
                # Reference: GPyTorch linear_cg.py lines 302-308
                if iter >= max_tridiag_iter - 1:
                    converged = True
                    break
    
    ctx.synchronize()
    var t_cg_loop_end = perf_counter_ns()
    if PROFILING:
        print("@ CG_loop_only", -Int(t_cg_loop_end - t_cg_loop_start))
        print("@ CG_num_iterations", -Int(num_iterations))
    
    # Copy alpha/beta to host and build tridiagonals
    if tridiag_size > 0:
        ctx.enqueue_copy(dst_buf=all_alphas_host, src_buf=all_alphas)
        ctx.enqueue_copy(dst_buf=all_betas_host, src_buf=all_betas)
        ctx.synchronize()
        
        # Build tridiagonal matrices using GPyTorch formula
        # Handle alpha=0 case (frozen vectors) by setting 1/alpha = 1 (not 0)
        # This matches GPyTorch's approach: masked_fill alpha with 1, compute reciprocal, then restore
        # 
        # GPyTorch early-stop: if max(off_diag across ALL columns) < 1e-6, stop updating tridiagonal
        # This prevents late-stage CG instability from corrupting log-det estimate
        # Reference: linear_operator/utils/linear_cg.py lines 326-327
        var update_tridiag = True
        var actual_tridiag_size = 0
        
        for k in range(tridiag_size):
            if not update_tridiag:
                break  # Stop building tridiagonal for ALL columns
            
            var max_offdiag = Float32(0.0)  # Track max off-diagonal across all columns
            
            for col in range(num_cols):
                var alpha_k = all_alphas_host[k * num_cols + col]
                
                # Diagonal: T[k,k] = 1/alpha[k] + beta[k-1]/alpha[k-1]
                # If alpha_k is near-zero (frozen vector), use 1/alpha = 1 (GPyTorch convention)
                var alpha_reciprocal: Float32
                if alpha_k < Float32(1e-10):
                    alpha_reciprocal = Float32(1.0)  # GPyTorch: masked_fill with 1, then reciprocal
                else:
                    alpha_reciprocal = Float32(1.0) / alpha_k
                
                var diag_val = alpha_reciprocal
                if k > 0:
                    var beta_km1 = all_betas_host[(k - 1) * num_cols + col]
                    var alpha_km1 = all_alphas_host[(k - 1) * num_cols + col]
                    # Use same convention for prev_alpha_reciprocal
                    var prev_alpha_reciprocal: Float32
                    if alpha_km1 < Float32(1e-10):
                        prev_alpha_reciprocal = Float32(1.0)
                    else:
                        prev_alpha_reciprocal = Float32(1.0) / alpha_km1
                    diag_val = diag_val + beta_km1 * prev_alpha_reciprocal
                tridiag_diag[col].append(diag_val)
                
                # Off-diagonal: T[k,k-1] = sqrt(beta[k-1]) * prev_alpha_reciprocal
                if k > 0:
                    var beta_km1 = all_betas_host[(k - 1) * num_cols + col]
                    var alpha_km1 = all_alphas_host[(k - 1) * num_cols + col]
                    if beta_km1 < Float32(0.0):
                        beta_km1 = Float32(0.0)
                    var prev_alpha_reciprocal: Float32
                    if alpha_km1 < Float32(1e-10):
                        prev_alpha_reciprocal = Float32(1.0)
                    else:
                        prev_alpha_reciprocal = Float32(1.0) / alpha_km1
                    var offdiag_val = sqrt(beta_km1) * prev_alpha_reciprocal
                    tridiag_offdiag[col].append(offdiag_val)
                    
                    # Track max off-diagonal across all columns
                    if offdiag_val > max_offdiag:
                        max_offdiag = offdiag_val
            
            actual_tridiag_size = k + 1
            
            # GPyTorch early-stop: if max(off_diag across ALL columns) < 1e-6, stop updating
            if k > 0 and max_offdiag < Float32(1e-6):
                update_tridiag = False
    
    var t_tridiag_end = perf_counter_ns()
    if PROFILING:
        print("@ CG_tridiag_build_cpu", -Int(t_tridiag_end - t_cg_loop_end))
    
    # Create result
    var solution = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    ctx.enqueue_copy(dst_buf=solution, src_buf=x)
    ctx.synchronize()
    
    if PROFILING:
        var t_cg_exit = perf_counter_ns()
        print("@ CG_exit_copy_sync", -Int(t_cg_exit - t_tridiag_end))
    
    return CGResultWithTridiag(
        solution^,
        num_iterations,
        converged,
        tridiag_diag^,
        tridiag_offdiag^,
        tridiag_size
    )


# =============================================================================
# Generic BBMM with Pre-built Preconditioner
# =============================================================================

fn bbmm_with_precond[P: GradientProvider, Q: Preconditioner](
    provider: P,
    precond: Q,
    y_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    mut pool: CGBufferPool,
    num_probes: Int = 10,
    max_iter: Int = 100,
    max_tridiag_iter: Int = 30,
    tol: Float32 = 1e-2,
    iteration: Int = 0,
    recycle_alpha: Bool = False,
    use_preconditioner: Bool = True,
) raises -> UnifiedBBMMResult:
    """Generic BBMM inference with a pre-built preconditioner.
    
    Core BBMM implementation that works with ANY preconditioner implementing the
    Preconditioner trait. The caller builds the preconditioner and passes it in.
    
    Preconditioner types:
    - PivotedCholeskyPrecond: Standard single-output (used by bbmm_unified wrapper)
    - KroneckerPreconditioner: Multi-output Kronecker structure
    
    Algorithm:
    1. Sample probes from N(0, P) via precond.sample_probes()
    2. Create batched RHS: [y | z_1 | ... | z_t]
    3. Run ONE batched mBCG solve -> [alpha | u_1 | ... | u_t] + tridiagonal matrices
    4. Extract inv_quad = y^T @ alpha (data fit term)
    5. Extract log|K| from tridiagonals + precond.log_det() correction
    6. Compute gradients for all kernel parameters
    7. Compute noise gradient (dK/d(noise) = I)
    
    Args:
        provider: Any GradientProvider (isotropic, ARD, composite, or Kronecker)
        precond: Pre-built preconditioner implementing Preconditioner trait
        y_device_ptr: Training targets [n] ON DEVICE
        n: Number of training points (nT for Kronecker)
        pool: Reusable buffer pool (caller must call pool.ensure_capacity first)
        num_probes: Number of probes for log_det and gradient estimation
        max_iter: Maximum CG iterations
        max_tridiag_iter: Max iterations to track for tridiagonal
        tol: CG convergence tolerance
        iteration: Training iteration for probe seed randomization
        recycle_alpha: If True, warm-start CG column 0 from previous solution in pool.x
        
    Returns:
        UnifiedBBMMResult containing NLL, all gradients, and solution
    """
    var ctx = provider.get_ctx()
    var num_params = provider.num_gradient_params()
    

    # Total columns: 1 (for y) + num_probes (for log_det and gradients)
    var num_cols_total = 1 + num_probes
    
    # =========================================================================
    # Step 1: Sample probes from N(0, P) using pre-built preconditioner
    # =========================================================================
    var probes_device = pool.probes_device
    var probe_norms = pool.probe_norms
    var probe_seed = iteration * 104729 + 42
    
    with ProfileBlock[PROFILING]("BBMM_probe_sampling"):
        if use_preconditioner:
            # Sample probes from N(0, P) via preconditioner
            precond.sample_probes(ctx, pool.sampled_probes_device, num_probes, UInt64(probe_seed))
        else:
            # Sample probes from N(0, I) — standard normal (no preconditioner)
            var ns_grid = (n * num_probes + 255) // 256
            ctx.enqueue_function[kernel_gpu_gaussian](
                pool.sampled_probes_device.unsafe_ptr(), n * num_probes, UInt64(probe_seed),
                grid_dim=(ns_grid,), block_dim=(256,))
        ctx.enqueue_function[kernel_copy](
            probes_device.unsafe_ptr(), pool.sampled_probes_device.unsafe_ptr(), n * num_probes,
            grid_dim=((n * num_probes + 255) // 256,), block_dim=(256,)
        )
        normalize_columns_gpu(ctx, probes_device, probe_norms, n, num_probes, sync=False)
        ctx.synchronize()
    

    # =========================================================================
    # Step 2: Assemble RHS [y | Z_norm] and run CG
    # =========================================================================
    var rhs_cg = pool.rhs_cg
    var y_device_buf = pool.y_device_buf
    var do_warm_start = recycle_alpha and iteration > 0
    var rhs_norms = pool.rhs_norms
    
    with ProfileBlock[PROFILING]("BBMM_rhs_assembly"):
        # Copy y to y_device_buf (needed for inv_quad later)
        ctx.enqueue_function[kernel_copy](
            y_device_buf.unsafe_ptr(), y_device_ptr, n,
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        
        # Assemble RHS: [y | z_norm_1 | ... | z_norm_P]
        ctx.enqueue_function[kernel_copy_column](
            rhs_cg.unsafe_ptr(), y_device_buf.unsafe_ptr(), n, 0,
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        ctx.enqueue_function[kernel_copy_columns](
            rhs_cg.unsafe_ptr(), probes_device.unsafe_ptr(),
            n, num_probes, 1,
            grid_dim=((n * num_probes + 255) // 256,), block_dim=(256,)
        )
        
        # OPTIMIZATION: Normalize all RHS columns to balance CG convergence
        # This matches GPyTorch's approach in linear_cg.py:131-133
        # When ||y|| >> ||probe||, the probes converge slower due to tighter tolerance.
        # Normalizing all columns to unit norm balances convergence across columns.
        normalize_columns_gpu(ctx, rhs_cg, rhs_norms, n, num_cols_total, sync=True)
        
        # Solution recycling: prepare pool.x for warm-start.
        # pool.x is the CG work buffer and remains in the normalized-RHS scale;
        # only the returned solution copy is un-normalized below. Reuse column 0
        # as-is and zero probe columns because probes change each iteration.
        if do_warm_start:
            # Zero out probe columns (1..P) — column-major, so columns 1..P start at offset n
            if num_cols_total > 1:
                ctx.enqueue_function[kernel_fill_constant](
                    pool.x.unsafe_ptr() + n, n * (num_cols_total - 1), Float32(0.0),
                    grid_dim=((n * (num_cols_total - 1) + 255) // 256,), block_dim=(256,)
                )
            ctx.synchronize()
        ctx.synchronize()
    
    # Run unified CG solver with generic preconditioner
    var cg_result: CGResultWithTridiag
    with ProfileBlock[PROFILING]("BBMM_cg_solve"):
        cg_result = batched_cg_unified[P, Q](
            provider, rhs_cg.unsafe_ptr(), n, num_cols_total,
            max_iter, max_tridiag_iter, tol, precond, pool,
            use_warm_start=do_warm_start,
            use_preconditioner=use_preconditioner,
        )
        
        # OPTIMIZATION: Un-normalize solution columns
        # The solution was computed for normalized RHS, so we need to scale back.
        # Solution[i] *= ||rhs[i]|| for each column.
        scale_columns_by_norms(ctx, cg_result.solution, rhs_norms, n, num_cols_total, sync=False)
        ctx.synchronize()
    

    # =========================================================================
    # Step 3: Compute inv_quad = y^T @ alpha
    # =========================================================================
    var inv_quad_device = pool.inv_quad_device
    var inv_quad_host = pool.inv_quad_host
    var inv_quad: Float32
    
    with ProfileBlock[PROFILING]("BBMM_inv_quad"):
        ctx.enqueue_function[kernel_dot_batched](
            y_device_buf.unsafe_ptr(), cg_result.solution.unsafe_ptr(), inv_quad_device.unsafe_ptr(),
            n, 1,
            grid_dim=(1, 1), block_dim=(256, 1)
        )
        ctx.enqueue_copy(dst_buf=inv_quad_host, src_buf=inv_quad_device)
        ctx.synchronize()
    
    inv_quad = inv_quad_host[0]
    
    # =========================================================================
    # Step 4: Compute log|K| from CG tridiagonals
    # =========================================================================
    var log_det = Float32(0.0)
    
    if cg_result.tridiag_size > 0:
        # Extract probe tridiagonals (skip column 0 which is y)
        var probe_diags = List[List[Float32]]()
        var probe_offdiags = List[List[Float32]]()
        
        for probe in range(1, 1 + num_probes):
            var diag_copy = List[Float32]()
            var offdiag_copy = List[Float32]()
            for i in range(len(cg_result.tridiag_diag[probe])):
                diag_copy.append(cg_result.tridiag_diag[probe][i])
            for i in range(len(cg_result.tridiag_offdiag[probe])):
                offdiag_copy.append(cg_result.tridiag_offdiag[probe][i])
            probe_diags.append(diag_copy^)
            probe_offdiags.append(offdiag_copy^)
        
        # Compute log|P^{-1}K| from CG tridiagonals
        var log_det_pinv_k: Float32
        with ProfileBlock[PROFILING]("BBMM_logdet_tridiag"):
            log_det_pinv_k = compute_logdet_from_tridiag_batched(
                ctx,
                probe_diags,
                probe_offdiags,
                cg_result.tridiag_size,
                num_probes,
                n
            )
            ctx.synchronize()
        
        # Add log|P| correction via Preconditioner trait (0 if no preconditioner)
        var log_det_P: Float32
        with ProfileBlock[PROFILING]("BBMM_logdet_precond"):
            if use_preconditioner:
                log_det_P = precond.log_det(ctx)
            else:
                log_det_P = Float32(0.0)  # No correction: CG tridiags give log|K| directly
            ctx.synchronize()
        log_det = log_det_pinv_k + log_det_P
    
    # =========================================================================
    # Compute NLL
    # =========================================================================
    var constant = Float32(0.5) * Float32(n) * log(Float32(2.0) * PI)
    var nll_total = Float32(0.5) * inv_quad + Float32(0.5) * log_det + constant
    var nll = nll_total / Float32(n)  # Per-sample NLL
    

    # =========================================================================
    # Step 5: Compute gradients for all kernel parameters
    # =========================================================================
    # Extract alpha and probe_solutions
    var alpha_device = pool.alpha_device
    var probe_solutions_device = pool.probe_solutions_device
    var right_factors_device_buf = pool.grad_probes_device
    
    with ProfileBlock[PROFILING]("BBMM_gradient_setup"):
        ctx.enqueue_function[kernel_extract_columns_range](
            alpha_device.unsafe_ptr(), cg_result.solution.unsafe_ptr(),
            n, 1, 0,  # Extract 1 column starting at offset 0
            grid_dim=((n + 255) // 256,), block_dim=(256,)
        )
        
        ctx.enqueue_function[kernel_extract_columns_range](
            probe_solutions_device.unsafe_ptr(), cg_result.solution.unsafe_ptr(),
            n, num_probes, 1,  # Extract num_probes columns starting at offset 1
            grid_dim=((n * num_probes + 255) // 256,), block_dim=(256,)
        )
        
        # Scale probe_solutions by norms to match GPyTorch's variance-reduced estimator
        # See: docs/plan/fix_gradient_trace_norm_factor_10022026_1227.md
        ctx.enqueue_function[kernel_scale_columns_by_norms](
            probe_solutions_device.unsafe_ptr(), probe_norms.unsafe_ptr(), n, num_probes,
            grid_dim=((n * num_probes + 255) // 256,), block_dim=(256,)
        )
        
        # Compute right_factors = P^{-1}(Z_norm * norms)
        ctx.enqueue_function[kernel_copy](
            right_factors_device_buf.unsafe_ptr(), probes_device.unsafe_ptr(), n * num_probes,
            grid_dim=((n * num_probes + 255) // 256,), block_dim=(256,)
        )
        
        ctx.enqueue_function[kernel_scale_columns_by_norms](
            right_factors_device_buf.unsafe_ptr(), probe_norms.unsafe_ptr(), n, num_probes,
            grid_dim=((n * num_probes + 255) // 256,), block_dim=(256,)
        )
        
        if use_preconditioner:
            precond.apply_precond(
                ctx,
                right_factors_device_buf.unsafe_ptr(),
                right_factors_device_buf.unsafe_ptr(),
                n, num_probes,
                sync=False,
            )
        # When no preconditioner: right_factors = Z_norm * norms (already computed above)
        ctx.synchronize()
    
    # -------------------------------------------------------------------------
    # Compute gradients for each kernel parameter
    # -------------------------------------------------------------------------
    var gradients = List[Float32](capacity=num_params + 1)
    
    with ProfileBlock[PROFILING]("BBMM_gradient_computation"):
        if provider.supports_fused_gradient():
            # =================================================================
            # FUSED PATH: Compute ALL gradient matvecs in 2 kernel launches
            # instead of num_params * 2 launches with per-param sync.
            # =================================================================
            # SAFETY CHECK: Verify fused buffers are large enough for num_params.
            # The caller (bbmm_unified) must pass num_kernel_params to ensure_capacity.
            # If this assertion fires, the caller forgot to size the fused buffers.
            debug_assert(
                pool.capacity_num_params_fused >= num_params,
                "FUSED GRADIENT BUFFER TOO SMALL: capacity_num_params_fused="
                + String(pool.capacity_num_params_fused) + " but need " + String(num_params)
                + ". Caller must pass num_kernel_params to pool.ensure_capacity()."
            )
            # Buffer layout within gradient_out_device [num_params * n * (1 + num_probes)]:
            #   [0, num_params * n)                          : grad_alpha  [num_params, n]
            #   [num_params * n, num_params * n * (1 + num_probes)) : grad_probes [num_params, n, num_probes]
            var grad_alpha_ptr = pool.gradient_out_device.unsafe_ptr()
            var grad_probes_ptr = pool.gradient_out_device.unsafe_ptr().offset(num_params * n)
            
            # Launch 1: Compute dK/dtheta_p @ alpha for ALL params simultaneously
            with ProfileBlock[PROFILING]("GRAD_fused_alpha"):
                provider.fused_gradient_matvec(grad_alpha_ptr, alpha_device.unsafe_ptr(), 1)
                ctx.synchronize()
            
            # Launch 2: Compute dK/dtheta_p @ right_factors for ALL params simultaneously
            with ProfileBlock[PROFILING]("GRAD_fused_probes"):
                provider.fused_gradient_matvec(grad_probes_ptr, right_factors_device_buf.unsafe_ptr(), num_probes)
                ctx.synchronize()
            
            with ProfileBlock[PROFILING]("GRAD_fused_dots_sync"):
                # Batched dot products for data terms: data_term[p] = alpha^T @ grad_alpha[p*n:(p+1)*n]
                var data_terms_ptr = pool.gradient_dots_device.unsafe_ptr()
                ctx.enqueue_function[kernel_dot_single_vs_strided](
                    alpha_device.unsafe_ptr(), grad_alpha_ptr, data_terms_ptr,
                    n, num_params,
                    grid_dim=(num_params,), block_dim=(256,)
                )
                
                # Batched dot products for trace terms:
                # trace_term[p * num_probes + j] = probe_solutions[:, j]^T @ grad_probes[p*n*num_probes + j*n : ...]
                var trace_terms_ptr = pool.gradient_dots_device.unsafe_ptr().offset(num_params)
                ctx.enqueue_function[kernel_dot_batched_vs_strided](
                    probe_solutions_device.unsafe_ptr(), grad_probes_ptr, trace_terms_ptr,
                    n, num_probes, num_params,
                    grid_dim=(num_params * num_probes,), block_dim=(256,)
                )
                
                # Single host copy + sync for ALL params
                ctx.enqueue_copy(dst_buf=pool.gradient_dots_host, src_buf=pool.gradient_dots_device)
                ctx.synchronize()
            
            # Extract gradients from host buffer
            var dots_host = pool.gradient_dots_host.unsafe_ptr()
            for p in range(num_params):
                var data_term = Float32(-0.5) * dots_host[p]
                var trace_sum = Float32(0.0)
                for j in range(num_probes):
                    trace_sum += dots_host[num_params + p * num_probes + j]
                var complexity_term = Float32(0.5) * trace_sum / Float32(num_probes)
                var grad_p = (data_term + complexity_term) / Float32(n)
                
                if isnan(grad_p) or isinf(grad_p):
                    grad_p = Float32(0.0)
                
                gradients.append(grad_p)
        elif provider.supports_fused_ls_os() and num_params == 2:
            # =================================================================
            # FUSED LS+OS PATH: Both dK/dl@V and K@V in one O(n²) pass.
            # 1.5-1.7x faster than 2 separate shmem launches.
            # Only for isotropic RBF/Matern (2-param kernels).
            # =================================================================
            var dK_param_alpha_device = pool.dK_param_alpha_device   # reuse for ls alpha
            var dK_param_Z_device = pool.dK_param_Z_device           # reuse for ls probes
            var param_data_term_device = pool.param_data_term_device
            var param_trace_device = pool.param_trace_device
            var param_trace_host = pool.param_trace_host
            var param_data_term_host = pool.param_data_term_host
            
            # Use pooled buffers for os gradient results (avoids per-iteration allocation)
            var os_alpha_device = pool.os_alpha_device
            var os_probes_device = pool.os_probes_device
            
            # Fused launch 1: alpha (1 col) → both ls and os
            with ProfileBlock[PROFILING]("GRAD_fused_ls_os_alpha"):
                provider.fused_ls_os_gradient_matvec(
                    dK_param_alpha_device.unsafe_ptr(),  # ls result
                    os_alpha_device.unsafe_ptr(),         # os result
                    alpha_device.unsafe_ptr(),
                    1,
                )
                ctx.synchronize()
            
            # Fused launch 2: probes (num_probes cols) → both ls and os
            with ProfileBlock[PROFILING]("GRAD_fused_ls_os_probes"):
                provider.fused_ls_os_gradient_matvec(
                    dK_param_Z_device.unsafe_ptr(),  # ls result
                    os_probes_device.unsafe_ptr(),    # os result
                    right_factors_device_buf.unsafe_ptr(),
                    num_probes,
                )
                ctx.synchronize()
            
            # Compute dots for BOTH params (ls=0, os=1)
            with ProfileBlock[PROFILING]("GRAD_fused_ls_os_dots"):
                # -- ls (param 0) --
                ctx.enqueue_function[kernel_dot_batched](
                    alpha_device.unsafe_ptr(), dK_param_alpha_device.unsafe_ptr(), param_data_term_device.unsafe_ptr(),
                    n, 1,
                    grid_dim=(1, 1), block_dim=(256, 1)
                )
                ctx.enqueue_function[kernel_dot_batched](
                    probe_solutions_device.unsafe_ptr(), dK_param_Z_device.unsafe_ptr(), param_trace_device.unsafe_ptr(),
                    n, num_probes,
                    grid_dim=(num_probes, 1), block_dim=(256, 1)
                )
                ctx.enqueue_copy(dst_buf=param_data_term_host, src_buf=param_data_term_device)
                ctx.enqueue_copy(dst_buf=param_trace_host, src_buf=param_trace_device)
                ctx.synchronize()
            
            # ls gradient
            var ls_data = Float32(-0.5) * param_data_term_host[0]
            var ls_trace_sum = Float32(0.0)
            for i in range(num_probes):
                ls_trace_sum += param_trace_host[i]
            var ls_grad = (ls_data + Float32(0.5) * ls_trace_sum / Float32(num_probes)) / Float32(n)
            if isnan(ls_grad) or isinf(ls_grad):
                ls_grad = Float32(0.0)
            gradients.append(ls_grad)
            
            with ProfileBlock[PROFILING]("GRAD_fused_ls_os_dots_os"):
                # -- os (param 1) --
                ctx.enqueue_function[kernel_dot_batched](
                    alpha_device.unsafe_ptr(), os_alpha_device.unsafe_ptr(), param_data_term_device.unsafe_ptr(),
                    n, 1,
                    grid_dim=(1, 1), block_dim=(256, 1)
                )
                ctx.enqueue_function[kernel_dot_batched](
                    probe_solutions_device.unsafe_ptr(), os_probes_device.unsafe_ptr(), param_trace_device.unsafe_ptr(),
                    n, num_probes,
                    grid_dim=(num_probes, 1), block_dim=(256, 1)
                )
                ctx.enqueue_copy(dst_buf=param_data_term_host, src_buf=param_data_term_device)
                ctx.enqueue_copy(dst_buf=param_trace_host, src_buf=param_trace_device)
                ctx.synchronize()
            
            # os gradient
            var os_data = Float32(-0.5) * param_data_term_host[0]
            var os_trace_sum = Float32(0.0)
            for i in range(num_probes):
                os_trace_sum += param_trace_host[i]
            var os_grad = (os_data + Float32(0.5) * os_trace_sum / Float32(num_probes)) / Float32(n)
            if isnan(os_grad) or isinf(os_grad):
                os_grad = Float32(0.0)
            gradients.append(os_grad)
         elif num_params == 3 and not provider.supports_fused_gradient():
            # =================================================================
            # FUSED 3-PARAM PATH: ls + param1 + os in one O(n²) pass.
            # 1.7x faster than 3 separate launches for Periodic/RQ.
            # Uses the fused 3-param dispatcher when available (Periodic/RQ).
            # For providers with fused_ls_os (e.g. Kronecker+RBF where the 3rd
            # param is the Kronecker outputscale = same as os_base gradient),
            # use fused ls+os for params 0+1 and a separate call for param 2.
            # =================================================================
            var dK_param_alpha_device = pool.dK_param_alpha_device
            var dK_param_Z_device = pool.dK_param_Z_device
            var param_data_term_device = pool.param_data_term_device
            var param_trace_device = pool.param_trace_device
            var param_trace_host = pool.param_trace_host
            var param_data_term_host = pool.param_data_term_host
            var os_alpha_device = pool.os_alpha_device
            var os_probes_device = pool.os_probes_device
            
            # Use pooled p1 buffers
            var p1_alpha_device = pool.p1_alpha_device
            var p1_probes_device = pool.p1_probes_device
            
            if provider.supports_fused_3param():
                # Truly fused: 1 kernel launch per vector set (1.7x faster)
                # Used for Periodic/RQ kernels
                with ProfileBlock[PROFILING]("GRAD_fused_3param_alpha"):
                    provider.fused_3param_gradient_matvec(
                        dK_param_alpha_device.unsafe_ptr(),  # ls result
                        p1_alpha_device.unsafe_ptr(),         # p1 result
                        os_alpha_device.unsafe_ptr(),         # os result
                        alpha_device.unsafe_ptr(),
                        1,
                    )
                
                with ProfileBlock[PROFILING]("GRAD_fused_3param_probes"):
                    provider.fused_3param_gradient_matvec(
                        dK_param_Z_device.unsafe_ptr(),       # ls result
                        p1_probes_device.unsafe_ptr(),        # p1 result
                        os_probes_device.unsafe_ptr(),        # os result
                        right_factors_device_buf.unsafe_ptr(),
                        num_probes,
                    )
                ctx.synchronize()
            elif provider.supports_fused_ls_os():
                # Fused ls+os for params 0+1, separate call for param 2.
                # Used for Kronecker+RBF/Matern where num_params=3 but the
                # base kernel only has 2 params (ls + os_base) and param 2
                # is the Kronecker outputscale (also a forward matvec).
                with ProfileBlock[PROFILING]("GRAD_fused_ls_os_alpha"):
                    provider.fused_ls_os_gradient_matvec(
                        dK_param_alpha_device.unsafe_ptr(),  # ls result (param 0)
                        os_alpha_device.unsafe_ptr(),         # os result (param 1)
                        alpha_device.unsafe_ptr(),
                        1,
                    )
                    provider.gradient_matvec(p1_alpha_device.unsafe_ptr(), alpha_device.unsafe_ptr(), 1, 2, sync=False)
                    ctx.synchronize()
                
                with ProfileBlock[PROFILING]("GRAD_fused_ls_os_probes"):
                    provider.fused_ls_os_gradient_matvec(
                        dK_param_Z_device.unsafe_ptr(),   # ls result (param 0)
                        os_probes_device.unsafe_ptr(),     # os result (param 1)
                        right_factors_device_buf.unsafe_ptr(),
                        num_probes,
                    )
                    provider.gradient_matvec(p1_probes_device.unsafe_ptr(), right_factors_device_buf.unsafe_ptr(), num_probes, 2, sync=False)
                    ctx.synchronize()
            else:
                # Fallback: 3 sequential gradient_matvec calls per vector set
                with ProfileBlock[PROFILING]("GRAD_3param_seq_alpha"):
                    provider.gradient_matvec(dK_param_alpha_device.unsafe_ptr(), alpha_device.unsafe_ptr(), 1, 0, sync=False)
                    provider.gradient_matvec(p1_alpha_device.unsafe_ptr(), alpha_device.unsafe_ptr(), 1, 2, sync=False)
                    provider.gradient_matvec(os_alpha_device.unsafe_ptr(), alpha_device.unsafe_ptr(), 1, 1, sync=False)
                    ctx.synchronize()
                
                with ProfileBlock[PROFILING]("GRAD_3param_seq_probes"):
                    provider.gradient_matvec(dK_param_Z_device.unsafe_ptr(), right_factors_device_buf.unsafe_ptr(), num_probes, 0, sync=False)
                    provider.gradient_matvec(p1_probes_device.unsafe_ptr(), right_factors_device_buf.unsafe_ptr(), num_probes, 2, sync=False)
                    provider.gradient_matvec(os_probes_device.unsafe_ptr(), right_factors_device_buf.unsafe_ptr(), num_probes, 1, sync=False)
                    ctx.synchronize()
            
            # Compute dots for all 3 params: ls=0, param1=2, os=1
            with ProfileBlock[PROFILING]("GRAD_fused_3param_dots"):
                # -- ls (param 0) --
                ctx.enqueue_function[kernel_dot_batched](
                    alpha_device.unsafe_ptr(), dK_param_alpha_device.unsafe_ptr(), param_data_term_device.unsafe_ptr(),
                    n, 1, grid_dim=(1, 1), block_dim=(256, 1))
                ctx.enqueue_function[kernel_dot_batched](
                    probe_solutions_device.unsafe_ptr(), dK_param_Z_device.unsafe_ptr(), param_trace_device.unsafe_ptr(),
                    n, num_probes, grid_dim=(num_probes, 1), block_dim=(256, 1))
                ctx.enqueue_copy(dst_buf=param_data_term_host, src_buf=param_data_term_device)
                ctx.enqueue_copy(dst_buf=param_trace_host, src_buf=param_trace_device)
                ctx.synchronize()
            
            var ls_data = Float32(-0.5) * param_data_term_host[0]
            var ls_trace = Float32(0.0)
            for i in range(num_probes): ls_trace += param_trace_host[i]
            var ls_grad = (ls_data + Float32(0.5) * ls_trace / Float32(num_probes)) / Float32(n)
            if isnan(ls_grad) or isinf(ls_grad): ls_grad = Float32(0.0)
            gradients.append(ls_grad)
            
            with ProfileBlock[PROFILING]("GRAD_fused_3param_dots_os"):
                # -- os (param 1) --
                ctx.enqueue_function[kernel_dot_batched](
                    alpha_device.unsafe_ptr(), os_alpha_device.unsafe_ptr(), param_data_term_device.unsafe_ptr(),
                    n, 1, grid_dim=(1, 1), block_dim=(256, 1))
                ctx.enqueue_function[kernel_dot_batched](
                    probe_solutions_device.unsafe_ptr(), os_probes_device.unsafe_ptr(), param_trace_device.unsafe_ptr(),
                    n, num_probes, grid_dim=(num_probes, 1), block_dim=(256, 1))
                ctx.enqueue_copy(dst_buf=param_data_term_host, src_buf=param_data_term_device)
                ctx.enqueue_copy(dst_buf=param_trace_host, src_buf=param_trace_device)
                ctx.synchronize()
            
            var os_data = Float32(-0.5) * param_data_term_host[0]
            var os_trace = Float32(0.0)
            for i in range(num_probes): os_trace += param_trace_host[i]
            var os_grad = (os_data + Float32(0.5) * os_trace / Float32(num_probes)) / Float32(n)
            if isnan(os_grad) or isinf(os_grad): os_grad = Float32(0.0)
            gradients.append(os_grad)
            
            with ProfileBlock[PROFILING]("GRAD_fused_3param_dots_p1"):
                # -- param1 (param 2) --
                ctx.enqueue_function[kernel_dot_batched](
                    alpha_device.unsafe_ptr(), p1_alpha_device.unsafe_ptr(), param_data_term_device.unsafe_ptr(),
                    n, 1, grid_dim=(1, 1), block_dim=(256, 1))
                ctx.enqueue_function[kernel_dot_batched](
                    probe_solutions_device.unsafe_ptr(), p1_probes_device.unsafe_ptr(), param_trace_device.unsafe_ptr(),
                    n, num_probes, grid_dim=(num_probes, 1), block_dim=(256, 1))
                ctx.enqueue_copy(dst_buf=param_data_term_host, src_buf=param_data_term_device)
                ctx.enqueue_copy(dst_buf=param_trace_host, src_buf=param_trace_device)
                ctx.synchronize()
            
            var p1_data = Float32(-0.5) * param_data_term_host[0]
            var p1_trace = Float32(0.0)
            for i in range(num_probes): p1_trace += param_trace_host[i]
            var p1_grad = (p1_data + Float32(0.5) * p1_trace / Float32(num_probes)) / Float32(n)
            if isnan(p1_grad) or isinf(p1_grad): p1_grad = Float32(0.0)
            gradients.append(p1_grad)
        else:
            # =================================================================
            # SEQUENTIAL PATH: Per-parameter loop (for 4+ param kernels, composite)
            # =================================================================
            var dK_param_alpha_device = pool.dK_param_alpha_device
            var dK_param_Z_device = pool.dK_param_Z_device
            var param_data_term_device = pool.param_data_term_device
            var param_trace_device = pool.param_trace_device
            var param_trace_host = pool.param_trace_host
            var param_data_term_host = pool.param_data_term_host
            
            for p in range(num_params):
                # Compute dK/dtheta_p @ alpha
                with ProfileBlock[PROFILING]("GRAD_param_alpha"):
                    provider.gradient_matvec(dK_param_alpha_device.unsafe_ptr(), alpha_device.unsafe_ptr(), 1, p, sync=False)
                    ctx.synchronize()
                
                # Compute dK/dtheta_p @ right_factors
                with ProfileBlock[PROFILING]("GRAD_param_probes"):
                    provider.gradient_matvec(dK_param_Z_device.unsafe_ptr(), right_factors_device_buf.unsafe_ptr(), num_probes, p, sync=False)
                    ctx.synchronize()
                
                with ProfileBlock[PROFILING]("GRAD_param_dots_sync"):
                    # Data term: alpha^T @ (dK/dtheta_p @ alpha)
                    ctx.enqueue_function[kernel_dot_batched](
                        alpha_device.unsafe_ptr(), dK_param_alpha_device.unsafe_ptr(), param_data_term_device.unsafe_ptr(),
                        n, 1,
                        grid_dim=(1, 1), block_dim=(256, 1)
                    )
                    
                    # Trace term: probe_solutions^T @ (dK/dtheta_p @ right_factors)
                    ctx.enqueue_function[kernel_dot_batched](
                        probe_solutions_device.unsafe_ptr(), dK_param_Z_device.unsafe_ptr(), param_trace_device.unsafe_ptr(),
                        n, num_probes,
                        grid_dim=(num_probes, 1), block_dim=(256, 1)
                    )
                    
                    # Copy results to host
                    ctx.enqueue_copy(dst_buf=param_data_term_host, src_buf=param_data_term_device)
                    ctx.enqueue_copy(dst_buf=param_trace_host, src_buf=param_trace_device)
                    ctx.synchronize()
                
                # Compute gradient
                var data_term = Float32(-0.5) * param_data_term_host[0]
                var trace_sum = Float32(0.0)
                for i in range(num_probes):
                    trace_sum += param_trace_host[i]
                var complexity_term = Float32(0.5) * trace_sum / Float32(num_probes)
                var grad_p = (data_term + complexity_term) / Float32(n)  # Normalize by n
                
                # Check for NaN/Inf
                if isnan(grad_p) or isinf(grad_p):
                    grad_p = Float32(0.0)
                
                gradients.append(grad_p)
        ctx.synchronize()
    
    # -------------------------------------------------------------------------
    # Compute noise gradient (dK/d(noise) = I)
    # -------------------------------------------------------------------------
    var grad_noise: Float32
    with ProfileBlock[PROFILING]("BBMM_noise_gradient"):
        # Data term: alpha^T @ alpha = ||alpha||^2
        var alpha_norm_sq_device = pool.alpha_norm_sq_device
        ctx.enqueue_function[kernel_dot_batched](
            alpha_device.unsafe_ptr(), alpha_device.unsafe_ptr(), alpha_norm_sq_device.unsafe_ptr(),
            n, 1,
            grid_dim=(1, 1), block_dim=(256, 1)
        )
        
        # Trace term: probe_solutions^T @ right_factors (Tr(K^{-1}))
        var trace_noise_device = pool.trace_noise_device
        ctx.enqueue_function[kernel_dot_batched](
            probe_solutions_device.unsafe_ptr(), right_factors_device_buf.unsafe_ptr(), trace_noise_device.unsafe_ptr(),
            n, num_probes,
            grid_dim=(num_probes, 1), block_dim=(256, 1)
        )
        
        # Copy results to host
        var alpha_norm_sq_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        var trace_noise_host = ctx.enqueue_create_host_buffer[float_dtype](num_probes)
        ctx.enqueue_copy(dst_buf=alpha_norm_sq_host, src_buf=alpha_norm_sq_device)
        ctx.enqueue_copy(dst_buf=trace_noise_host, src_buf=trace_noise_device)
        ctx.synchronize()
        
        var noise_data_term = Float32(-0.5) * alpha_norm_sq_host[0]
        var noise_trace_sum = Float32(0.0)
        for i in range(num_probes):
            noise_trace_sum += trace_noise_host[i]
        var noise_complexity_term = Float32(0.5) * noise_trace_sum / Float32(num_probes)
        grad_noise = (noise_data_term + noise_complexity_term) / Float32(n)
        
        if isnan(grad_noise) or isinf(grad_noise):
            grad_noise = Float32(0.0)
        
        gradients.append(grad_noise)
        ctx.synchronize()
    

    # =========================================================================
    # Return result
    # =========================================================================
    var solution_copy: DeviceBuffer[float_dtype]
    var probe_solutions_copy: DeviceBuffer[float_dtype]
    var right_factors_copy: DeviceBuffer[float_dtype]
    with ProfileBlock[PROFILING]("BBMM_result_packaging"):
        solution_copy = ctx.enqueue_create_buffer[float_dtype](n)
        probe_solutions_copy = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        right_factors_copy = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        ctx.enqueue_copy(dst_buf=solution_copy, src_buf=alpha_device)
        ctx.enqueue_copy(dst_buf=probe_solutions_copy, src_buf=probe_solutions_device)
        ctx.enqueue_function[kernel_copy](
            right_factors_copy.unsafe_ptr(), right_factors_device_buf.unsafe_ptr(), n * num_probes,
            grid_dim=((n * num_probes + 255) // 256,), block_dim=(256,)
        )
        ctx.synchronize()
    
    return UnifiedBBMMResult(
        inv_quad, log_det, nll,
        solution_copy^, probe_solutions_copy^, right_factors_copy^,
        gradients^,
        cg_result.num_iterations, cg_result.converged, Float32(0.0),
        num_params
    )


# =============================================================================
# Unified BBMM Wrapper (builds PivotedCholesky preconditioner internally)
# =============================================================================

fn bbmm_unified[P: GradientProvider](
    provider: P,
    y_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    mut pool: CGBufferPool,
    num_probes: Int = 10,
    max_iter: Int = 100,
    max_tridiag_iter: Int = 30,
    tol: Float32 = 1e-2,
    precond_rank: Int = 10,
    iteration: Int = 0,
    recycle_alpha: Bool = False,
) raises -> UnifiedBBMMResult:
    """Unified BBMM inference for single-output GPs.
    
    Thin wrapper around bbmm_with_precond that builds a PivotedCholeskyPrecond
    internally. All existing callers use this function unchanged.
    
    For multi-output Kronecker GPs, call bbmm_with_precond directly with a
    KroneckerPreconditioner instead.
    
    Args:
        provider: Any GradientProvider (isotropic, ARD, or composite)
        y_device_ptr: Training targets [n] ON DEVICE
        n: Number of training points
        pool: Reusable buffer pool
        num_probes: Number of probes for log_det and gradient estimation
        max_iter: Maximum CG iterations
        max_tridiag_iter: Max iterations to track for tridiagonal
        tol: CG convergence tolerance
        precond_rank: Rank for Pivoted Cholesky preconditioner
        iteration: Training iteration for probe seed randomization
        recycle_alpha: If True, warm-start CG column 0 from previous solution in pool.x
        
    Returns:
        UnifiedBBMMResult containing NLL, all gradients, and solution
    """
    var ctx = provider.get_ctx()
    var num_cols_total = 1 + num_probes
    var num_kernel_params = provider.num_gradient_params()
    
    # Ensure buffer pool has sufficient capacity
    # CRITICAL: Must pass num_kernel_params for fused gradient path.
    # The fused path writes num_kernel_params * n * (1 + num_probes) elements
    # to gradient_out_device. Without this, the buffer defaults to 1 param
    # and the fused kernel writes out of bounds, corrupting GPU memory.
    pool.ensure_capacity(ctx, n, num_cols_total, num_probes, max_tridiag_iter, precond_rank, num_kernel_params=num_kernel_params)
    
    # Build PivotedCholesky preconditioner
    var precond = build_pivoted_cholesky_precond_unified[P](provider, precond_rank, max_num_cols=num_cols_total)
    
    # Delegate to generic bbmm_with_precond
    return bbmm_with_precond[P, PivotedCholeskyPrecond](
        provider, precond, y_device_ptr, n, pool,
        num_probes, max_iter, max_tridiag_iter, tol, iteration,
        recycle_alpha=recycle_alpha,
    )
