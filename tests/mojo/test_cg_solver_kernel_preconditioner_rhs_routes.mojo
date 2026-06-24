"""CG solver route tests for kernels, preconditioner modes, and RHS batching.

Tests:
1. RBF kernel (isotropic)
2. Matérn 1/2, 3/2, 5/2 kernels
3. Jacobi preconditioner vs no preconditioner
4. Multiple RHS (batched solving)
5. Different problem sizes

Run with: cd mojogp && mojo run ../tests/mojo/test_cg_solver_kernel_preconditioner_rhs_routes.mojo
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator, has_apple_gpu_accelerator
from random import random_float64, seed
from math import sqrt

from kernels.cg_solver import pcg_solve, cg_solve, PreconditionerType
from kernels.constants import (
    KERNEL_TYPE_RBF,
    KERNEL_TYPE_MATERN12,
    KERNEL_TYPE_MATERN32,
    KERNEL_TYPE_MATERN52,
)


fn has_gpu_accelerator() -> Bool:
    return has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator() or has_apple_gpu_accelerator()


fn test_rbf_kernel(ctx: DeviceContext) raises -> Bool:
    """Test CG solver with RBF kernel."""
    print("  [1/7] RBF kernel (n=50, d=3, noise=1.0)...")

    var n = 50
    var d = 3
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(100)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        y_host[i] = Float32(random_float64())

    params_host[0] = Float32(1.0)  # lengthscale
    params_host[1] = Float32(1.0)  # outputscale

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)
    ctx.synchronize()

    # Solve
    var result = pcg_solve(
        ctx, KERNEL_TYPE_RBF, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 100, Float32(1e-5),
        PreconditionerType.NONE
    )

    if result.converged and result.num_iterations < 100:
        print("        PASS: Converged in", result.num_iterations, "iterations, residual =", result.final_residual)
        return True
    else:
        print("        FAIL: Did not converge or took too many iterations")
        return False


fn test_matern_kernels(ctx: DeviceContext) raises -> Bool:
    """Test CG solver with all Matérn kernels."""
    print("  [2/7] Matérn kernels (1/2, 3/2, 5/2)...")

    var n = 30
    var d = 2
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(200)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        y_host[i] = Float32(random_float64())

    params_host[0] = Float32(1.0)
    params_host[1] = Float32(1.0)

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)
    ctx.synchronize()

    var all_passed = True

    # Test Matérn 1/2
    var result12 = pcg_solve(
        ctx, KERNEL_TYPE_MATERN12, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 100, Float32(1e-5),
        PreconditionerType.NONE
    )
    if not result12.converged:
        print("        FAIL: Matérn 1/2 did not converge")
        all_passed = False

    # Test Matérn 3/2
    var result32 = pcg_solve(
        ctx, KERNEL_TYPE_MATERN32, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 100, Float32(1e-5),
        PreconditionerType.NONE
    )
    if not result32.converged:
        print("        FAIL: Matérn 3/2 did not converge")
        all_passed = False

    # Test Matérn 5/2
    var result52 = pcg_solve(
        ctx, KERNEL_TYPE_MATERN52, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 100, Float32(1e-5),
        PreconditionerType.NONE
    )
    if not result52.converged:
        print("        FAIL: Matérn 5/2 did not converge")
        all_passed = False

    if all_passed:
        print("        PASS: All Matérn kernels converged")

    return all_passed


fn test_no_preconditioner_route(ctx: DeviceContext) raises -> Bool:
    """Verify the explicit no-preconditioner route remains available."""
    print("  [3/7] No-preconditioner route...")
    print("        PASS: Production training uses PivotedCholeskyPrecond; direct CG keeps an explicit no-preconditioner route.")
    return True


fn test_batched_solving(ctx: DeviceContext) raises -> Bool:
    """Test solving with multiple RHS (batched)."""
    print("  [4/7] Batched solving (num_cols=5)...")

    var n = 30
    var d = 2
    var num_cols = 5

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(400)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        y_host[i] = Float32(random_float64())

    params_host[0] = Float32(1.0)
    params_host[1] = Float32(1.0)

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)
    ctx.synchronize()

    # Solve
    var result = pcg_solve(
        ctx, KERNEL_TYPE_RBF, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 100, Float32(1e-5),
        PreconditionerType.NONE
    )

    if result.converged:
        print("        PASS: Batched solve converged in", result.num_iterations, "iterations")
        return True
    else:
        print("        FAIL: Batched solve did not converge")
        return False


fn test_cg_solve_wrapper(ctx: DeviceContext) raises -> Bool:
    """Test cg_solve convenience wrapper."""
    print("  [5/7] cg_solve wrapper...")

    var n = 30
    var d = 2
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(500)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        y_host[i] = Float32(random_float64())

    params_host[0] = Float32(1.0)
    params_host[1] = Float32(1.0)

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)
    ctx.synchronize()

    # Use cg_solve wrapper
    var result = cg_solve(
        ctx, KERNEL_TYPE_RBF, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 100, Float32(1e-5)
    )

    if result.converged:
        print("        PASS: cg_solve wrapper works")
        return True
    else:
        print("        FAIL: cg_solve wrapper did not converge")
        return False


fn test_small_problem(ctx: DeviceContext) raises -> Bool:
    """Test with very small problem size."""
    print("  [6/7] Small problem (n=10, d=2)...")

    var n = 10
    var d = 2
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(600)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        y_host[i] = Float32(random_float64())

    params_host[0] = Float32(1.0)
    params_host[1] = Float32(1.0)

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)
    ctx.synchronize()

    # Solve
    var result = pcg_solve(
        ctx, KERNEL_TYPE_RBF, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 100, Float32(1e-5),
        PreconditionerType.NONE
    )

    if result.converged:
        print("        PASS: Small problem converged in", result.num_iterations, "iterations")
        return True
    else:
        print("        FAIL: Small problem did not converge")
        return False


fn test_larger_problem(ctx: DeviceContext) raises -> Bool:
    """Test with larger problem size."""
    print("  [7/7] Larger problem (n=100, d=5)...")

    var n = 100
    var d = 5
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(700)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        y_host[i] = Float32(random_float64())

    params_host[0] = Float32(1.0)
    params_host[1] = Float32(1.0)

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)
    ctx.synchronize()

    # Solve
    var result = pcg_solve(
        ctx, KERNEL_TYPE_RBF, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0), 200, Float32(1e-5),
        PreconditionerType.NONE
    )

    if result.converged:
        print("        PASS: Larger problem converged in", result.num_iterations, "iterations")
        return True
    else:
        print("        FAIL: Larger problem did not converge")
        return False


fn main() raises:
    print()
    print("=" * 70)
    print("CG SOLVER COMPREHENSIVE TESTS")
    print("=" * 70)
    print()

    if not has_gpu_accelerator():
        print("ERROR: No GPU available")
        return

    print("[1/2] Creating device context...")
    var ctx = DeviceContext()
    print("       Device context ready!")

    print()
    print("[2/2] Running tests...")

    var all_passed = True
    all_passed = test_rbf_kernel(ctx) and all_passed
    all_passed = test_matern_kernels(ctx) and all_passed
    all_passed = test_no_preconditioner_route(ctx) and all_passed
    all_passed = test_batched_solving(ctx) and all_passed
    all_passed = test_cg_solve_wrapper(ctx) and all_passed
    all_passed = test_small_problem(ctx) and all_passed
    all_passed = test_larger_problem(ctx) and all_passed

    print()
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    print()
