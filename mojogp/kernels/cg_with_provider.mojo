"""CG solver using provider abstraction."""

from gpu.host import DeviceContext, DeviceBuffer
from memory import UnsafePointer
from math import sqrt

from .cg_solver import (
    kernel_dot_batched,
    kernel_axpy_batched,
    kernel_scale_add_batched,
    kernel_copy,
    kernel_compute_alpha,
    kernel_cg_update_fused,
    kernel_beta_and_copy_fused,
    CGResult,
)
from .matvec_provider import MatvecProvider

alias float_dtype = DType.float32


fn cg_solve_with_provider[T: MatvecProvider](
    provider: T,
    y_device_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    num_cols: Int,
    max_iter: Int = 100,
    tol: Float32 = 1e-2,
    use_preconditioner: Bool = True,
    check_interval: Int = 5,
) raises -> CGResult:
    """Generic CG solver using any MatvecProvider.
    
    This function works with both MatrixFreeProvider and MaterializedProvider.
    No preconditioning (z = r). For preconditioned CG, use batched_cg_unified
    with a Preconditioner trait implementation.
    
    Args:
        provider: Any provider implementing MatvecProvider trait
        y_device_ptr: Right-hand side [n * num_cols] ON DEVICE
        n: Number of data points
        num_cols: Number of RHS columns
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        use_preconditioner: Ignored (kept for API compatibility)
        check_interval: Check convergence every N iterations (default: 5)
        
    Returns:
        CGResult with solution and convergence info
    """
    var ctx = provider.get_ctx()
    
    # Allocate device buffers
    var x_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    var r_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    var z_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)  # Preconditioned residual
    var p_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    var Ap_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
    var rz_old_device = ctx.enqueue_create_buffer[float_dtype](num_cols)
    var rz_new_device = ctx.enqueue_create_buffer[float_dtype](num_cols)
    var pAp_device = ctx.enqueue_create_buffer[float_dtype](num_cols)
    var alpha_device = ctx.enqueue_create_buffer[float_dtype](num_cols)
    var beta_device = ctx.enqueue_create_buffer[float_dtype](num_cols)
    
    var x_ptr = x_device.unsafe_ptr()
    var r_ptr = r_device.unsafe_ptr()
    var z_ptr = z_device.unsafe_ptr()
    var p_ptr = p_device.unsafe_ptr()
    var Ap_ptr = Ap_device.unsafe_ptr()
    
    # Initialize x = 0 using host buffer
    var zero_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_cols)
    for i in range(n * num_cols):
        zero_host[i] = Float32(0.0)
    ctx.enqueue_copy(dst_buf=x_device, src_buf=zero_host)
    
    # Initialize r = y using GPU copy kernel
    ctx.enqueue_function[kernel_copy](
        r_ptr, y_device_ptr, n * num_cols,
        grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
    )
    ctx.synchronize()
    
    # No preconditioning: z = r
    ctx.enqueue_function[kernel_copy](
        z_ptr, r_ptr, n * num_cols,
        grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
    )
    
    # Initialize search direction: p = z
    ctx.enqueue_function[kernel_copy](
        p_ptr, z_ptr, n * num_cols,
        grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
    )
    
    # Compute initial rz_old = r · z
    ctx.enqueue_function[kernel_dot_batched](
        r_ptr, z_ptr, rz_old_device.unsafe_ptr(), n, num_cols,
        grid_dim=(num_cols, 1), block_dim=(256, 1)
    )
    ctx.synchronize()
    
    var converged = False
    var num_iterations = 0
    var final_residual = Float32(0.0)
    
    for iter in range(max_iter):
        num_iterations = iter + 1
        
        # 1. Ap = (K + noise*I) @ p (using provider)
        provider.forward_matvec(Ap_ptr, p_ptr, num_cols)
        ctx.synchronize()
        
        # 2. pAp = p^T @ Ap
        ctx.enqueue_function[kernel_dot_batched](
            p_ptr, Ap_ptr, pAp_device.unsafe_ptr(), n, num_cols,
            grid_dim=(num_cols, 1), block_dim=(256, 1)
        )
        
        # 3. alpha = rz_old / pAp (compute on GPU)
        ctx.enqueue_function[kernel_compute_alpha](
            rz_old_device.unsafe_ptr(),
            pAp_device.unsafe_ptr(),
            alpha_device.unsafe_ptr(),
            num_cols,
            grid_dim=((num_cols + 255) // 256,),
            block_dim=(256,)
        )
        
        # 4-5. Fused: x += alpha * p AND r -= alpha * Ap
        ctx.enqueue_function[kernel_cg_update_fused](
            alpha_device.unsafe_ptr(),
            p_ptr,
            Ap_ptr,
            x_ptr,
            r_ptr,
            n,
            num_cols,
            grid_dim=((n + 15) // 16, (num_cols + 15) // 16),
            block_dim=(16, 16)
        )
        
        # 6. Check convergence every check_interval iterations (or on last iteration)
        if (iter + 1) % check_interval == 0 or iter == max_iter - 1:
            # Compute ||r||² on GPU using dot product: r · r
            var r_norm_sq_device = ctx.enqueue_create_buffer[float_dtype](num_cols)
            ctx.enqueue_function[kernel_dot_batched](
                r_ptr, r_ptr, r_norm_sq_device.unsafe_ptr(), n, num_cols,
                grid_dim=(num_cols, 1), block_dim=(256, 1)
            )
            
            # Copy only num_cols scalars (not n × num_cols!)
            var r_norm_sq_host = ctx.enqueue_create_host_buffer[float_dtype](num_cols)
            ctx.enqueue_copy(dst_buf=r_norm_sq_host, src_buf=r_norm_sq_device)
            ctx.synchronize()
            
            # Find max ||r|| across columns
            var max_residual = Float32(0.0)
            for col in range(num_cols):
                var col_norm = sqrt(r_norm_sq_host[col])
                if col_norm > max_residual:
                    max_residual = col_norm
            
            final_residual = max_residual
            
            if max_residual < tol:
                converged = True
                break
        
        # 7. No preconditioning: z = r
        ctx.enqueue_function[kernel_copy](
            z_ptr, r_ptr, n * num_cols,
            grid_dim=((n * num_cols + 255) // 256,), block_dim=(256,)
        )
        
        # 8. rz_new = r · z
        ctx.enqueue_function[kernel_dot_batched](
            r_ptr, z_ptr, rz_new_device.unsafe_ptr(), n, num_cols,
            grid_dim=(num_cols, 1), block_dim=(256, 1)
        )
        
        # 9-11. Fused: beta = rz_new / rz_old AND rz_old = rz_new
        ctx.enqueue_function[kernel_beta_and_copy_fused](
            rz_old_device.unsafe_ptr(),
            rz_new_device.unsafe_ptr(),
            beta_device.unsafe_ptr(),
            num_cols,
            grid_dim=((num_cols + 255) // 256,),
            block_dim=(256,)
        )
        
        # 10. p = z + beta * p
        ctx.enqueue_function[kernel_scale_add_batched](
            beta_device.unsafe_ptr(), p_ptr, z_ptr, n, num_cols,
            grid_dim=((n + 15) // 16, (num_cols + 15) // 16),
            block_dim=(16, 16)
        )
    
    return CGResult(x_device, num_iterations, final_residual, converged)
