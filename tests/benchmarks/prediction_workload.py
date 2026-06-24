"""Shared benchmark prediction workload sizes."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

BENCHMARK_PREDICTION_N_TEST = 5000
BENCHMARK_PATHWISE_SAMPLING_N_TEST = 24

PREDICTION_X_TEST_CORE_TARGETS = {
    "minimal": [1_000, BENCHMARK_PREDICTION_N_TEST],
    "standard": [1_000, BENCHMARK_PREDICTION_N_TEST, 10_000],
    "extensive": [1_000, BENCHMARK_PREDICTION_N_TEST, 10_000],
    # The dedicated matrix-free n-scaling suite varies n_train only. Keep the
    # post-fit prediction workload at the canonical n_test so it does not turn
    # into a second prediction-scaling benchmark.
    "matrix_free_n_scaling": [BENCHMARK_PREDICTION_N_TEST],
}

PREDICTION_X_TEST_ENVELOPE_TARGETS = {
    "xsmall": [],
    "small": [],
    "medium": [25_000],
    # A100 40GB large-tier rows preserve large n_train coverage. Large test-point
    # envelopes can exceed that VRAM budget for materialized exact prediction.
    "large": [],
    "xlarge": [100_000, 200_000],
}


def prediction_x_test_repeat_count() -> int:
    raw = os.environ.get("MOJOGP_SCALING_PREDICTION_REPEAT_CALLS", "3")
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def _dedupe_ordered(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def prediction_x_test_core_targets(variety: str) -> list[int]:
    targets = PREDICTION_X_TEST_CORE_TARGETS.get(variety)
    if targets is None:
        raise ValueError(
            "prediction X_test variety must be minimal, standard, extensive, or matrix_free_n_scaling; "
            f"got {variety!r}"
        )
    values = _dedupe_ordered(targets)
    if BENCHMARK_PREDICTION_N_TEST not in values:
        raise AssertionError("canonical benchmark n_test must remain in every core grid")
    return values


def prediction_x_test_envelope_targets(tier: str) -> list[int]:
    targets = PREDICTION_X_TEST_ENVELOPE_TARGETS.get(tier)
    if targets is None:
        raise ValueError(
            "prediction X_test tier must be xsmall, small, medium, large, or xlarge; "
            f"got {tier!r}"
        )
    return _dedupe_ordered(targets)


def prediction_x_test_target_specs(
    *,
    variety: str,
    tier: str,
    framework: str,
) -> list[dict[str, object]]:
    specs = [
        {
            "n_test": n_test,
            "size_role": "core",
            "comparison_class": "fair_match",
        }
        for n_test in prediction_x_test_core_targets(variety)
    ]
    if framework == "mojogp" and variety != "matrix_free_n_scaling":
        core_values = {int(spec["n_test"]) for spec in specs}
        specs.extend(
            {
                "n_test": n_test,
                "size_role": "envelope",
                "comparison_class": "mojogp_only_prediction_envelope",
            }
            for n_test in prediction_x_test_envelope_targets(tier)
            if n_test not in core_values
        )
    return specs


def prediction_time_quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "repeated_median_time_s": None,
            "repeated_p5_time_s": None,
            "repeated_p95_time_s": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "repeated_median_time_s": float(np.median(arr)),
        "repeated_p5_time_s": float(np.percentile(arr, 5)),
        "repeated_p95_time_s": float(np.percentile(arr, 95)),
    }


def prediction_x_test_failure_status(error: BaseException) -> str:
    message = str(error).lower()
    error_type = type(error).__name__.lower()
    if (
        "out of memory" in message
        or "outofmemory" in error_type
        or "memory allocation" in message
        or "cuda_error_out_of_memory" in message
    ):
        return "failed_oom"
    return "failed"


def prediction_x_test_should_record_failure(spec: dict[str, object]) -> bool:
    return spec.get("size_role") == "envelope"


def clear_prediction_x_test_failure_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def prediction_x_test_scaling_failure_entry(
    *,
    spec: dict[str, object],
    error: BaseException,
    failure_stage: str,
    timing_quality: str,
    cache_used: bool,
    prediction_peak_gpu_mb: float | None = None,
    prediction_delta_gpu_mb: float | None = None,
) -> dict[str, object]:
    error_message = str(error).replace("\n", " ").strip()
    if len(error_message) > 500:
        error_message = error_message[:497] + "..."
    status = prediction_x_test_failure_status(error)
    return {
        "n_test": int(spec["n_test"]),
        "size_role": spec.get("size_role"),
        "comparison_class": spec.get("comparison_class"),
        "status": status,
        "failure_reason": "oom" if status == "failed_oom" else "exception",
        "failure_stage": failure_stage,
        "error_type": type(error).__name__,
        "error_message": error_message,
        "timing_quality": timing_quality,
        "cache_used": bool(cache_used),
        "first_apply_time_s": None,
        "repeat_count": 0,
        "prediction_peak_gpu_mb": prediction_peak_gpu_mb,
        "prediction_delta_gpu_mb": prediction_delta_gpu_mb,
        "mean_time_s": None,
        "variance_time_s": None,
        **prediction_time_quantiles([]),
    }


def prediction_x_test_scaling_entry(
    *,
    spec: dict[str, object],
    timing_quality: str,
    cache_used: bool,
    first_apply_time_s: float,
    repeat_times_s: list[float],
    prediction_peak_gpu_mb: float | None = None,
    prediction_delta_gpu_mb: float | None = None,
    mean_time_s: float | None = None,
    variance_time_s: float | None = None,
) -> dict[str, object]:
    return {
        "n_test": int(spec["n_test"]),
        "size_role": spec.get("size_role"),
        "comparison_class": spec.get("comparison_class"),
        "status": "ok",
        "failure_reason": None,
        "failure_stage": None,
        "error_type": None,
        "error_message": None,
        "timing_quality": timing_quality,
        "cache_used": bool(cache_used),
        "first_apply_time_s": float(first_apply_time_s),
        "repeat_count": len(repeat_times_s),
        "prediction_peak_gpu_mb": prediction_peak_gpu_mb,
        "prediction_delta_gpu_mb": prediction_delta_gpu_mb,
        "mean_time_s": mean_time_s,
        "variance_time_s": variance_time_s,
        **prediction_time_quantiles(repeat_times_s),
    }
