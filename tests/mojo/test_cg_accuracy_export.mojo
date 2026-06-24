"""Export MojoGP CG solver solution for accuracy comparison.

This test:
1. Generates test data with fixed seed
2. Solves with MojoGP CG solver
3. Exports data and solution to files for Python comparison

Run with: mojo run test_cg_accuracy_export.mojo
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator, has_apple_gpu_accelerator
from random import random_float64, seed

from kernels.cg_solver import pcg_solve, PreconditionerType
from kernels.constants import (
    KERNEL_TYPE_RBF,
    KERNEL_TYPE_MATERN12,
    KERNEL_TYPE_MATERN32,
    KERNEL_TYPE_MATERN52,
)


fn has_gpu_accelerator() -> Bool:
    return has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator() or has_apple_gpu_accelerator()


fn write_array_to_file(filename: String, data: HostBuffer[DType.float32], rows: Int, cols: Int) raises:
    """Write array to text file (one value per line)."""
    var file = open(filename, "w")
    for i in range(rows):
        for j in range(cols):
            var idx = i * cols + j
            var val = data[idx]
            _ = file.write(String(val))
            _ = file.write("\n")
    file.close()


fn solve_and_export_kernel(
    ctx: DeviceContext,
    kernel_type: Int,
    kernel_name: String,
    n: Int, d: Int, num_cols: Int,
    noise: Float32,
    lengthscale: Float32,
    outputscale: Float32,
    x_device: DeviceBuffer[DType.float32],
    y_device: DeviceBuffer[DType.float32],
    params_device: DeviceBuffer[DType.float32],
    x_host: HostBuffer[DType.float32],
    y_host: HostBuffer[DType.float32],
) raises:
    """Solve with a specific kernel and export results."""
    print("  [", kernel_name, "]")

    # Solve with CG
    var result = pcg_solve(
        ctx, kernel_type, False,
        x_device.unsafe_ptr(), params_device.unsafe_ptr(), y_device.unsafe_ptr(),
        n, d, num_cols, noise,
        100, Float32(1e-6),
        PreconditionerType.NONE
    )

    print("       Converged:", result.converged, "| Iterations:", result.num_iterations, "| Residual:", result.final_residual)

    if not result.converged:
        print("       ERROR: CG did not converge!")
        return

    # Copy solution to host
    var alpha_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    ctx.enqueue_copy(dst_buf=alpha_host, src_buf=result.solution)
    ctx.synchronize()

    # Compute solution norm
    var alpha_norm_sq = Float32(0.0)
    for i in range(n * num_cols):
        var val = alpha_host[i]
        alpha_norm_sq += val * val
    var alpha_norm = alpha_norm_sq ** 0.5
    print("       Solution norm:", alpha_norm)

    # Write solution to file
    var filename = "test_data/cg_accuracy_alpha_mojo_" + kernel_name + ".txt"
    write_array_to_file(filename, alpha_host, n, num_cols)
    print("       Wrote", filename)

    # Write metadata
    var meta_filename = "test_data/cg_accuracy_meta_" + kernel_name + ".txt"
    var meta_file = open(meta_filename, "w")
    _ = meta_file.write("kernel=" + kernel_name + "\n")
    _ = meta_file.write("n=" + String(n) + "\n")
    _ = meta_file.write("d=" + String(d) + "\n")
    _ = meta_file.write("noise=" + String(noise) + "\n")
    _ = meta_file.write("lengthscale=" + String(lengthscale) + "\n")
    _ = meta_file.write("outputscale=" + String(outputscale) + "\n")
    _ = meta_file.write("iterations=" + String(result.num_iterations) + "\n")
    _ = meta_file.write("residual=" + String(result.final_residual) + "\n")
    _ = meta_file.write("solution_norm=" + String(alpha_norm) + "\n")
    meta_file.close()


fn main() raises:
    print()
    print("=" * 70)
    print("MOJOGP CG SOLVER - ACCURACY TEST DATA EXPORT (ALL 4 KERNELS)")
    print("=" * 70)
    print()

    if not has_gpu_accelerator():
        print("ERROR: No GPU available")
        return

    # Test configuration (must match Python test)
    var n = 50
    var d = 3
    var num_cols = 1
    var noise = Float32(0.1)
    var lengthscale = Float32(1.0)
    var outputscale = Float32(1.0)
    var test_seed = 42

    print("Test configuration:")
    print("  n =", n, ", d =", d)
    print("  kernels = RBF, Matérn 1/2, Matérn 3/2, Matérn 5/2")
    print("  lengthscale =", lengthscale, ", outputscale =", outputscale)
    print("  noise =", noise)
    print("  seed =", test_seed)
    print()

    print("[1/4] Creating device context...")
    var ctx = DeviceContext()

    print("[2/4] Generating test data (seed =", test_seed, ")...")

    # Create host buffers
    var x_host = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var y_host = ctx.enqueue_create_host_buffer[DType.float32](n * num_cols)
    var params_host = ctx.enqueue_create_host_buffer[DType.float32](2)

    # Generate data with fixed seed
    seed(test_seed)
    for i in range(n * d):
        x_host[i] = Float32(random_float64())
    for i in range(n * num_cols):
        y_host[i] = Float32(random_float64())

    params_host[0] = lengthscale
    params_host[1] = outputscale

    # Copy to device
    var x_device = ctx.enqueue_create_buffer[DType.float32](n * d)
    var y_device = ctx.enqueue_create_buffer[DType.float32](n * num_cols)
    var params_device = ctx.enqueue_create_buffer[DType.float32](2)

    ctx.enqueue_copy(dst_buf=x_device, src_buf=x_host)
    ctx.enqueue_copy(dst_buf=y_device, src_buf=y_host)
    ctx.enqueue_copy(dst_buf=params_device, src_buf=params_host)
    ctx.synchronize()

    # Write shared data to files (used by all kernels)
    write_array_to_file("test_data/cg_accuracy_X.txt", x_host, n, d)
    print("       Wrote test_data/cg_accuracy_X.txt")

    write_array_to_file("test_data/cg_accuracy_y.txt", y_host, n, num_cols)
    print("       Wrote test_data/cg_accuracy_y.txt")

    print()
    print("[3/4] Solving with MojoGP CG solver (all 4 kernels)...")

    # Solve with each kernel type
    solve_and_export_kernel(ctx, KERNEL_TYPE_RBF, "rbf", n, d, num_cols, noise, lengthscale, outputscale, x_device, y_device, params_device, x_host, y_host)
    solve_and_export_kernel(ctx, KERNEL_TYPE_MATERN12, "matern12", n, d, num_cols, noise, lengthscale, outputscale, x_device, y_device, params_device, x_host, y_host)
    solve_and_export_kernel(ctx, KERNEL_TYPE_MATERN32, "matern32", n, d, num_cols, noise, lengthscale, outputscale, x_device, y_device, params_device, x_host, y_host)
    solve_and_export_kernel(ctx, KERNEL_TYPE_MATERN52, "matern52", n, d, num_cols, noise, lengthscale, outputscale, x_device, y_device, params_device, x_host, y_host)

    print()
    print("[4/4] Writing global metadata...")
    var meta_file = open("test_data/cg_accuracy_config.txt", "w")
    _ = meta_file.write("n=" + String(n) + "\n")
    _ = meta_file.write("d=" + String(d) + "\n")
    _ = meta_file.write("noise=" + String(noise) + "\n")
    _ = meta_file.write("lengthscale=" + String(lengthscale) + "\n")
    _ = meta_file.write("outputscale=" + String(outputscale) + "\n")
    _ = meta_file.write("seed=" + String(test_seed) + "\n")
    meta_file.close()
    print("       Wrote test_data/cg_accuracy_config.txt")

    print()
    print("=" * 70)
    print("EXPORT COMPLETE - 4 KERNELS TESTED")
    print("=" * 70)
    print()
    print("Next step: Run 'python test_cg_accuracy_compare.py' to compare with GPyTorch")
    print()
