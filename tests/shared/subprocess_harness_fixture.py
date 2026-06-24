"""Child fixture module for subprocess harness contract tests."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .subprocess_harness import run_child_main


def _alloc_cuda_mb(size_mb: float):
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for harness GPU-allocation fixtures")

    numel = max(int(round(size_mb * 1024 * 1024 / 4.0)), 1)
    tensor = torch.empty(numel, dtype=torch.float32, device="cuda")
    tensor.fill_(1.0)
    torch.cuda.synchronize()
    return tensor


_IMPORT_BASELINE_HOLDER = None
_IMPORT_BASELINE_MB = float(os.environ.get("MOJOGP_HARNESS_IMPORT_BASELINE_MB", "0"))
if _IMPORT_BASELINE_MB > 0.0:
    _IMPORT_BASELINE_HOLDER = _alloc_cuda_mb(_IMPORT_BASELINE_MB)


def _handle(payload, session):
    case = str(payload["case"])
    if case == "echo":
        return {
            "payload": {
                "message": str(payload["message"]),
                "child_flag": True,
            },
            "timing": {"case": case},
        }

    if case == "result_path":
        temp_dir = Path(tempfile.mkdtemp(prefix="mojogp_harness_fixture_"))
        result_path = temp_dir / "result.json"
        result_path.write_text(
            json.dumps({"message": str(payload["message"])}), encoding="utf-8"
        )
        return {
            "result_path": result_path,
            "timing": {"case": case},
        }

    if case == "failure":
        raise RuntimeError(str(payload.get("message", "fixture failure")))

    if case == "gpu_injection":
        inject_mb = float(payload["inject_mb"])
        hold_s = float(payload.get("hold_s", 0.3))
        tensor = _alloc_cuda_mb(inject_mb)
        time.sleep(hold_s)
        fit_snapshot = session.snapshot_gpu()
        return {
            "payload": {
                "requested_mb": inject_mb,
                "import_baseline_mb": _IMPORT_BASELINE_MB,
                "value": float(tensor[0].item()),
            },
            "memory_snapshots": [fit_snapshot],
            "timing": {"case": case},
        }

    fit_snapshot = session.snapshot_gpu()
    return {
        "payload": {"snapshots_recorded": fit_snapshot["method"] is not None},
        "memory_snapshots": [fit_snapshot],
        "timing": {"case": case},
    }


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
