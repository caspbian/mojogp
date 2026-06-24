"""Allocate a known amount of native Mojo GPU memory and report the delta.

Environment variables:
  MOJOGP_TEST_MOJO_GPU_INJECT_MB   size in MB (default 128)
  MOJOGP_TEST_MOJO_GPU_HOLD_S      hold duration in seconds (default 0.35)
"""

from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from os import getenv
from time import perf_counter_ns


fn kernel_touch_buffer(
    ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < UInt(n):
        ptr[tid] = Float32(tid % 1024) * Float32(0.001)


fn _get_inject_mb() -> Int:
    try:
        return Int(getenv("MOJOGP_TEST_MOJO_GPU_INJECT_MB", "128"))
    except:
        return 128


fn _get_hold_ns() -> Int:
    try:
        var raw = String(getenv("MOJOGP_TEST_MOJO_GPU_HOLD_S", "0.35"))
        if raw == "0.10":
            return 100_000_000
        if raw == "0.15":
            return 150_000_000
        if raw == "0.20":
            return 200_000_000
        if raw == "0.25":
            return 250_000_000
        if raw == "0.30":
            return 300_000_000
        if raw == "0.35":
            return 350_000_000
        if raw == "0.40":
            return 400_000_000
        if raw == "0.50":
            return 500_000_000
        if raw == "1.00":
            return 1_000_000_000
        if raw == "1.50":
            return 1_500_000_000
        if raw == "2.00":
            return 2_000_000_000
    except:
        pass
    return 350_000_000


fn main() raises:
    var inject_mb = _get_inject_mb()
    var hold_ns = _get_hold_ns()
    var numel = inject_mb * 1024 * 1024 // 4

    print("mojo_gpu_allocation_probe_mb=", inject_mb)
    var ctx = DeviceContext()
    var mem_before = ctx.get_memory_info()
    var free_before = mem_before[0]
    print("mojo_gpu_allocation_probe_free_before_bytes=", free_before)

    var buf = ctx.enqueue_create_buffer[DType.float32](max(numel, 1))
    buf.enqueue_fill(Float32(1.0))
    ctx.enqueue_function[kernel_touch_buffer](
        buf.unsafe_ptr(), max(numel, 1),
        grid_dim=((max(numel, 1) + 255) // 256,), block_dim=(256,),
    )
    ctx.synchronize()
    var mem_after = ctx.get_memory_info()
    var free_after = mem_after[0]
    var delta_bytes = UInt(0)
    if free_before > free_after:
        delta_bytes = free_before - free_after
    print("mojo_gpu_allocation_probe_free_after_bytes=", free_after)
    print("mojo_gpu_allocation_probe_delta_bytes=", delta_bytes)
    print("mojo_gpu_allocation_probe_ready=1")

    var start = perf_counter_ns()
    while perf_counter_ns() - start < UInt(hold_ns):
        pass

    _ = buf
    ctx.synchronize()
    print("mojo_gpu_allocation_probe_done=1")
