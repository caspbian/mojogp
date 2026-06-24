"""Unit tests for CG solver components.

Tests:
1. CGResult struct creation and access
2. PreconditionerType enum usage
3. dispatch_forward_matvec for each kernel type
4. Error handling (invalid kernel type, invalid parameters)
5. Basic CG solve functionality

Run with: mojo run tests/unit/test_cg_solver.mojo
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator, has_apple_gpu_accelerator
from math import sqrt
from random import random_float64, seed

# Import from refactored kernels - direct submodule imports
from kernels.cg_solver import pcg_solve, cg_solve, CGResult, dispatch_forward_matvec
from kernels.constants import KERNEL_TYPE_RBF, KERNEL_TYPE_MATERN12, KERNEL_TYPE_MATERN32, KERNEL_TYPE_MATERN52


fn has_gpu_accelerator() -> Bool:
    """Check if any supported GPU accelerator is available."""
    return has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator() or has_apple_gpu_accelerator()


fn test_cg_result_struct() raises -> Bool:
    """Test CGResult struct creation and field access."""
    print("Test 1: CGResult struct...")

    if not has_gpu_accelerator():
        print("  SKIP: No GPU available")
        return True

    var ctx = DeviceContext()
    var solution_buf = ctx.enqueue_create_buffer[DType.float32](100)

    var result = CGResult(
        solution=solution_buf,
        num_iterations=42,
        final_residual=Float32(1e-7),
        converged=True
    )

    var passed = True
    if result.num_iterations != 42:
        print("  FAIL: num_iterations expected 42, got", result.num_iterations)
        passed = False

    if result.final_residual != Float32(1e-7):
        print("  FAIL: final_residual expected 1e-7, got", result.final_residual)
        passed = False

    if not result.converged:
        print("  FAIL: converged expected True, got False")
        passed = False

    if passed:
        print("  PASS: CGResult struct works correctly")

    return passed


fn test_placeholder_enum() -> Bool:
    """Placeholder test (Jacobi enum was removed as dead code)."""
    print("Test 2: Placeholder (JACOBI enum removed)...")
    print("  PASS: No-op (Jacobi preconditioner was dead code, removed)")
    return True


fn test_dispatch_forward_matvec_rbf() raises -> Bool:
    """Test dispatch_forward_matvec with RBF kernel."""
    print("Test 3: dispatch_forward_matvec (RBF)...")

    if not has_gpu_accelerator():
        print("  SKIP: No GPU available")
        return True

    var ctx = DeviceContext()
    var n = 100
    var d = 5
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var v_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(42)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        v_host[i] = Float32(random_float64())

    # RBF isotropic params: [lengthscale, outputscale]
    params_host[0] = Float32(1.0)  # lengthscale
    params_host[1] = Float32(1.0)  # outputscale

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var v_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)
    var out_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=v_device, src_buf=v_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)

    # Call dispatch_forward_matvec
    dispatch_forward_matvec(
        ctx, KERNEL_TYPE_RBF, False,
        out_device.unsafe_ptr(), x_device.unsafe_ptr(), v_device.unsafe_ptr(),
        params_device.unsafe_ptr(), n, d, num_cols, Float32(0.01)
    )

    # Copy result back
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_device)
    ctx.synchronize()

    # Check that output is non-zero and reasonable
    var sum_val = Float32(0.0)
    for i in range(n * num_cols):
        sum_val += out_host[i]

    var passed = True
    if sum_val == Float32(0.0):
        print("  FAIL: Output is all zeros")
        passed = False

    if passed:
        print("  PASS: dispatch_forward_matvec (RBF) produces non-zero output")

    return passed


fn test_dispatch_forward_matvec_matern() raises -> Bool:
    """Test dispatch_forward_matvec with Matérn kernels."""
    print("Test 4: dispatch_forward_matvec (Matérn)...")

    if not has_gpu_accelerator():
        print("  SKIP: No GPU available")
        return True

    var ctx = DeviceContext()
    var n = 50
    var d = 3
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var v_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(123)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        v_host[i] = Float32(random_float64())

    params_host[0] = Float32(1.0)  # lengthscale
    params_host[1] = Float32(1.0)  # outputscale

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var v_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)
    var out_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=v_device, src_buf=v_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)

    var passed = True

    # Test Matérn 1/2
    dispatch_forward_matvec(
        ctx, KERNEL_TYPE_MATERN12, False,
        out_device.unsafe_ptr(), x_device.unsafe_ptr(), v_device.unsafe_ptr(),
        params_device.unsafe_ptr(), n, d, num_cols, Float32(0.01)
    )
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_device)
    ctx.synchronize()

    var sum_val = Float32(0.0)
    for i in range(n * num_cols):
        sum_val += out_host[i]
    if sum_val == Float32(0.0):
        print("  FAIL: Matérn 1/2 output is all zeros")
        passed = False

    # Test Matérn 3/2
    dispatch_forward_matvec(
        ctx, KERNEL_TYPE_MATERN32, False,
        out_device.unsafe_ptr(), x_device.unsafe_ptr(), v_device.unsafe_ptr(),
        params_device.unsafe_ptr(), n, d, num_cols, Float32(0.01)
    )
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_device)
    ctx.synchronize()

    sum_val = Float32(0.0)
    for i in range(n * num_cols):
        sum_val += out_host[i]
    if sum_val == Float32(0.0):
        print("  FAIL: Matérn 3/2 output is all zeros")
        passed = False

    # Test Matérn 5/2
    dispatch_forward_matvec(
        ctx, KERNEL_TYPE_MATERN52, False,
        out_device.unsafe_ptr(), x_device.unsafe_ptr(), v_device.unsafe_ptr(),
        params_device.unsafe_ptr(), n, d, num_cols, Float32(0.01)
    )
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_device)
    ctx.synchronize()

    sum_val = Float32(0.0)
    for i in range(n * num_cols):
        sum_val += out_host[i]
    if sum_val == Float32(0.0):
        print("  FAIL: Matérn 5/2 output is all zeros")
        passed = False

    if passed:
        print("  PASS: dispatch_forward_matvec (Matérn 1/2, 3/2, 5/2) all work")

    return passed


fn test_dispatch_unknown_kernel() raises -> Bool:
    """Test dispatch_forward_matvec with invalid kernel type."""
    print("Test 5: dispatch_forward_matvec (unknown kernel)...")

    if not has_gpu_accelerator():
        print("  SKIP: No GPU available")
        return True

    var ctx = DeviceContext()
    var n = 10
    var d = 2
    var num_cols = 1

    # Create minimal buffers
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var v_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)
    var out_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)

    var raised_error = False
    try:
        # Use invalid kernel type (999)
        dispatch_forward_matvec(
            ctx, 999, False,
            out_device.unsafe_ptr(), x_device.unsafe_ptr(), v_device.unsafe_ptr(),
            params_device.unsafe_ptr(), n, d, num_cols, Float32(0.01)
        )
    except:
        raised_error = True

    if raised_error:
        print("  PASS: Unknown kernel type raises error")
        return True
    else:
        print("  FAIL: Unknown kernel type did not raise error")
        return False


fn test_pcg_solve_basic() raises -> Bool:
    """Test basic pcg_solve functionality."""
    print("Test 6: pcg_solve (basic)...")

    if not has_gpu_accelerator():
        print("  SKIP: No GPU available")
        return True

    var ctx = DeviceContext()
    var n = 50
    var d = 3
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(456)
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

    # Solve with large noise for well-conditioned system
    var result = pcg_solve(
        ctx, KERNEL_TYPE_RBF, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0),
        100, Float32(1e-5),
        PreconditionerType.NONE
    )

    var passed = True

    # Check that it converged
    if not result.converged:
        print("  FAIL: CG did not converge (iterations:", result.num_iterations, ", residual:", result.final_residual, ")")
        passed = False

    # Check that iterations is reasonable
    if result.num_iterations <= 0 or result.num_iterations > 100:
        print("  FAIL: Unreasonable iteration count:", result.num_iterations)
        passed = False

    # Check that residual is small
    if result.final_residual >= Float32(1e-5):
        print("  FAIL: Residual too large:", result.final_residual)
        passed = False

    if passed:
        print("  PASS: pcg_solve converged in", result.num_iterations, "iterations, residual =", result.final_residual)

    return passed


fn test_cg_solve_wrapper() raises -> Bool:
    """Test cg_solve convenience wrapper."""
    print("Test 7: cg_solve wrapper...")

    if not has_gpu_accelerator():
        print("  SKIP: No GPU available")
        return True

    var ctx = DeviceContext()
    var n = 30
    var d = 2
    var num_cols = 1

    # Create test data
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    seed(789)
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
        n, d, num_cols, Float32(1.0),
        100, Float32(1e-5)
    )

    var passed = True

    if not result.converged:
        print("  FAIL: cg_solve did not converge")
        passed = False

    if passed:
        print("  PASS: cg_solve wrapper works correctly")

    return passed


fn main() raises:
    print()
    print("=" * 70)
    print("CG SOLVER UNIT TESTS")
    print("=" * 70)
    print()

    var all_passed = True

    # Run tests
    all_passed = test_cg_result_struct() and all_passed
    all_passed = test_placeholder_enum() and all_passed
    all_passed = test_dispatch_forward_matvec_rbf() and all_passed
    all_passed = test_dispatch_forward_matvec_matern() and all_passed
    all_passed = test_dispatch_unknown_kernel() and all_passed
    all_passed = test_pcg_solve_basic() and all_passed
    all_passed = test_cg_solve_wrapper() and all_passed

    print()
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)
    print()
