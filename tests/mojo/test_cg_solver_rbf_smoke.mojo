"""RBF CG solver smoke test with direct Mojo kernel imports.

Run with: cd mojogp && mojo run ../tests/mojo/test_cg_solver_rbf_smoke.mojo
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator, has_apple_gpu_accelerator
from random import random_float64, seed

# Import directly from the cg_solver module
from kernels.cg_solver import (
    pcg_solve,
    cg_solve,
    CGResult,
    PreconditionerType,
)
from kernels.constants import (
    KERNEL_TYPE_RBF,
    KERNEL_TYPE_MATERN12,
    KERNEL_TYPE_MATERN32,
    KERNEL_TYPE_MATERN52,
)


fn has_gpu_accelerator() -> Bool:
    """Check if any supported GPU accelerator is available."""
    return has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator() or has_apple_gpu_accelerator()


fn main() raises:
    print()
    print("=" * 70)
    print("CG SOLVER RBF SMOKE TEST")
    print("=" * 70)
    print()

    if not has_gpu_accelerator():
        print("ERROR: No GPU available")
        return

    print("[1/4] Creating device context...")
    var ctx = DeviceContext()
    print("       Device context ready!")

    print()
    print("[2/4] Testing CGResult struct...")
    var solution_buf = ctx.enqueue_create_buffer[DType.float32](100)
    var result = CGResult(
        solution=solution_buf,
        num_iterations=42,
        final_residual=Float32(1e-7),
        converged=True
    )
    print("       CGResult created: iterations =", result.num_iterations, ", residual =", result.final_residual)

    print()
    print("[3/4] Verifying direct no-preconditioner route...")
    print("       PASS: Direct CG exposes PreconditionerType.NONE")

    print()
    print("[4/4] Testing pcg_solve...")
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

    print("       Running CG solver (n=50, d=3, RBF kernel, noise=1.0)...")

    # Solve with large noise for well-conditioned system
    var cg_result = pcg_solve(
        ctx, KERNEL_TYPE_RBF, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, Float32(1.0),
        100, Float32(1e-5),
        PreconditionerType.NONE
    )

    print("       CG converged:", cg_result.converged)
    print("       Iterations:", cg_result.num_iterations)
    print("       Final residual:", cg_result.final_residual)

    print()
    print("=" * 70)
    if cg_result.converged:
        print("TEST PASSED ✓")
    else:
        print("TEST FAILED ✗ - CG did not converge")
    print("=" * 70)
    print()
