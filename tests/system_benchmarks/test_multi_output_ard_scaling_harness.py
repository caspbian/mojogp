"""Multi-output ARD scaling benchmark surface."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.benchmarks.comparison_policy import policy_for
from tests.benchmarks.multi_output_ard_scaling import (
    run_multi_output_ard_scaling_subprocess,
)
from tests.shared.benchmarking.report import print_result, save_summary_report
from tests.shared.benchmarking.result_types import BenchmarkResult

from .conftest import assert_gpu_available, assert_gpu_was_used, get_vram_tier, requires_cuda
from .test_multi_output_scaling_harness import _assert_benchmark_methodology_fields
from .test_scaling_certification_harness import (
    _assert_matrix_free_prediction_memory_scaling,
    _size_role,
)


BENCHMARK_TARGETS = {
    "xsmall": {
        "materialized": [2000, 3000],
        "matrix_free": [5000, 10000],
        "num_tasks": 3,
        "d": 9,
        "relevant_dims": 3,
    },
    "small": {
        "materialized": [3000, 5000],
        "matrix_free": [10000, 25000],
        "num_tasks": 3,
        "d": 9,
        "relevant_dims": 3,
    },
    "medium": {
        "materialized": [5000, 8000],
        "matrix_free": [25000, 50000],
        "num_tasks": 4,
        "d": 17,
        "relevant_dims": 5,
    },
    "large": {
        "materialized": [3000, 8000, 12000],
        "matrix_free": [25000, 50000],
        "num_tasks": 4,
        "d": 17,
        "relevant_dims": 5,
    },
    "xlarge": {
        "materialized": [12000, 20000],
        "matrix_free": [50000, 75000],
        "num_tasks": 6,
        "d": 17,
        "relevant_dims": 5,
    },
}


GPYTORCH_MATERIALIZED_SHARED_SIZES = {
    "xsmall": [2000, 3000],
    "small": [3000, 5000],
    "medium": [5000, 8000],
    "large": [3000],
    "xlarge": [12000, 20000],
}


def _quick_enabled() -> bool:
    return os.environ.get("MOJOGP_BENCHMARK_QUICK", "0") == "1"


def _framework_sizes(*, framework: str, method: str, tier: str) -> list[int]:
    if framework == "gpytorch" and method != "materialized":
        return []
    if framework == "gpytorch":
        return list(GPYTORCH_MATERIALIZED_SHARED_SIZES[tier])
    return list(BENCHMARK_TARGETS[tier][method])


def _allow_multi_output_ard_recorded_failure(
    *, framework: str, method: str, size_role: str
) -> bool:
    return framework == "gpytorch" and method == "materialized" and size_role in {
        "anchor_1",
        "envelope",
    }


def _is_recordable_multi_output_ard_failure(exc: Exception) -> bool:
    error = str(exc).lower()
    return (
        "outofmemory" in error
        or "out of memory" in error
        or "cuda out of memory" in error
        or "timed out" in error
    )


def _run_case_capture(
    *,
    framework: str,
    prediction_mode: str,
    method: str,
    n_train: int,
    d: int,
    num_tasks: int,
    relevant_dims: int,
    tier: str,
    size_role: str,
    allow_recorded_failure: bool,
    results_dir,
) -> tuple[BenchmarkResult | None, dict[str, object] | None]:
    try:
        result = run_multi_output_ard_scaling_subprocess(
            framework=framework,
            prediction_mode=prediction_mode,
            method=method,
            n_train=n_train,
            d=d,
            num_tasks=num_tasks,
            relevant_dims=relevant_dims,
            tier=tier,
            specialization=None,
            results_dir=results_dir,
        )
        return result, None
    except Exception as exc:
        failure = {
            "framework": framework,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "benchmark_track": os.environ.get("MOJOGP_BENCHMARK_TRACK", "scaling"),
            "benchmark_route_tier": tier,
            "size_role": size_role,
            "n": n_train,
            "d": d,
            "num_tasks": num_tasks,
            "relevant_dims": relevant_dims,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if allow_recorded_failure and _is_recordable_multi_output_ard_failure(exc):
            return None, failure
        raise


@pytest.mark.minimal
@pytest.mark.ard
@pytest.mark.multi_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_multi_output_ard_scaling_benchmark(results_dir, method: str):
    assert_gpu_available()
    policy = policy_for("multi_output_ard_scaling")
    tier = get_vram_tier()
    targets = BENCHMARK_TARGETS[tier]
    d = int(targets["d"])
    num_tasks = int(targets["num_tasks"])
    relevant_dims = int(targets["relevant_dims"])
    frameworks = (
        ["mojogp", "gpytorch"]
        if method == "materialized" and policy.published_cross_framework
        else ["mojogp"]
    )

    results: list[BenchmarkResult] = []
    failures: list[dict[str, object]] = []
    prediction_modes = ["exact", "love"]
    for prediction_mode in prediction_modes:
        for framework in frameworks:
            framework_sizes = _framework_sizes(
                framework=framework,
                method=method,
                tier=tier,
            )
            if _quick_enabled():
                framework_sizes = framework_sizes[:1]
            for n_train in framework_sizes:
                size_role = _size_role(n_train, framework_sizes)
                allow_recorded_failure = _allow_multi_output_ard_recorded_failure(
                    framework=framework,
                    method=method,
                    size_role=size_role,
                )
                benchmark, failure = _run_case_capture(
                    framework=framework,
                    prediction_mode=prediction_mode,
                    method=method,
                    n_train=n_train,
                    d=d,
                    num_tasks=num_tasks,
                    relevant_dims=relevant_dims,
                    tier=tier,
                    size_role=size_role,
                    allow_recorded_failure=allow_recorded_failure,
                    results_dir=results_dir,
                )
                if failure is not None:
                    failures.append(failure)
                    continue
                assert benchmark is not None
                benchmark.config["size_role"] = size_role
                print_result(
                    benchmark,
                    title=(
                        f"Multi-output ARD scaling: {framework} {method} "
                        f"pred={prediction_mode} n={n_train} d={d} T={num_tasks} rel={relevant_dims}"
                    ),
                )
                assert_gpu_was_used(benchmark)
                assert np.isfinite(benchmark.accuracy.rmse)
                assert np.isfinite(benchmark.accuracy.r_squared)
                assert np.isfinite(benchmark.hyperparameters.final_nll)
                _assert_benchmark_methodology_fields(benchmark)
                if benchmark.config.get("framework") == "mojogp":
                    assert benchmark.config.get("pairwise_relevance_accuracy", 0.0) >= 0.5
                results.append(benchmark)

    if method == "matrix_free":
        _assert_matrix_free_prediction_memory_scaling(results)

    save_summary_report(
        results,
        results_dir,
        f"multi_output_ard_scaling_{method}",
        failures=failures,
    )
