"""Hold a known GPU allocation in a separate process for tests."""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for gpu_memory_holder") from exc


def _alloc_cuda_mb(size_mb: float):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for gpu_memory_holder")

    numel = max(int(round(size_mb * 1024 * 1024 / 4.0)), 1)
    tensor = torch.empty(numel, dtype=torch.float32, device="cuda")
    tensor.fill_(1.0)
    torch.cuda.synchronize()
    return tensor


def main() -> int:
    size_mb = float(os.environ["MOJOGP_TEST_GPU_HOLDER_MB"])
    ready_path = Path(os.environ["MOJOGP_TEST_GPU_HOLDER_READY"])
    stop_path = Path(os.environ["MOJOGP_TEST_GPU_HOLDER_STOP"])
    timeout_s = float(os.environ.get("MOJOGP_TEST_GPU_HOLDER_TIMEOUT_S", "30"))

    holder = _alloc_cuda_mb(size_mb)
    ready_path.write_text("ready", encoding="utf-8")

    start = time.perf_counter()
    while not stop_path.exists():
        if time.perf_counter() - start > timeout_s:
            _ = holder
            return 2
        time.sleep(0.05)

    _ = holder
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
