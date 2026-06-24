"""Small child module for benchmark infrastructure tests."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from tests.shared.subprocess_harness import run_child_main


def _alloc_cuda_mb(size_mb: float):
    if torch is None or not torch.cuda.is_available():
        return None
    numel = max(int(round(size_mb * 1024 * 1024 / 4.0)), 1)
    tensor = torch.empty(numel, dtype=torch.float32, device="cuda")
    tensor.fill_(1.0)
    torch.cuda.synchronize()
    return tensor


def _handle(payload: dict[str, object], session):
    case = str(payload.get("case", "direct"))
    inject_mb = float(payload.get("inject_mb", 0.0))
    hold_s = float(payload.get("hold_s", 0.2))
    tensor = _alloc_cuda_mb(inject_mb)
    if tensor is not None:
        time.sleep(hold_s)
    snapshot = session.snapshot_gpu()
    result_payload = {
        "run_id": str(payload.get("run_id", "fixture-run")),
        "speed": {
            "training_time_s": 0.25,
            "prediction_mean_time_s": 0.1,
            "prediction_variance_time_s": 0.15,
            "end_to_end_time_s": 0.5,
            "iterations_run": 4,
            "max_iterations": 5,
            "early_stopped": True,
            "ms_per_iteration": 62.0,
            "iter_time_min_ms": 60.0,
            "iter_time_q25_ms": 61.0,
            "iter_time_mean_ms": 62.5,
            "iter_time_median_ms": 62.0,
            "iter_time_q75_ms": 64.0,
            "iter_time_max_ms": 66.0,
            "iter_time_p5_ms": 60.2,
            "iter_time_p95_ms": 65.8,
            "startup_compile_time_s": 0.3,
            "startup_warm_cache_hit_s": 0.05,
            "startup_prepare_time_s": 0.02,
        },
        "memory": {"gpu_max_mb": 1.0},
    }
    if case == "nested_benchmark":
        result_payload = {
            "benchmark": {
                "config": {
                    "framework": "mojogp",
                    "kernel": "rbf",
                    "model_type": "MultiOutputGP",
                    "training_method": "materialized",
                    "prediction_mode": "love",
                    "n": 150,
                    "d": 3,
                    "num_tasks": 2,
                    "comparison_class": "mojogp_only",
                    "baseline_backend": "none",
                    "fairness_note": "fixture nested benchmark",
                },
                "speed": {
                    "training_time_s": 0.75,
                    "prediction_mean_time_s": 0.1,
                    "prediction_variance_time_s": 0.15,
                    "end_to_end_time_s": 1.0,
                    "iterations_run": 6,
                    "max_iterations": 6,
                    "early_stopped": False,
                    "ms_per_iteration": 120.0,
                    "iter_time_min_ms": 115.0,
                    "iter_time_q25_ms": 118.0,
                    "iter_time_mean_ms": 120.5,
                    "iter_time_median_ms": 120.0,
                    "iter_time_q75_ms": 123.0,
                    "iter_time_max_ms": 126.0,
                    "iter_time_p5_ms": 116.0,
                    "iter_time_p95_ms": 125.0,
                    "startup_compile_time_s": 0.4,
                    "startup_warm_cache_hit_s": 0.07,
                    "startup_prepare_time_s": 0.03,
                },
                "memory": {"gpu_max_mb": 2.0},
            },
            "run_id": str(payload.get("run_id", "fixture-run")),
        }
    return {
        "payload": result_payload,
        "memory_snapshots": [snapshot],
        "timing": {"fixture": True},
    }


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
