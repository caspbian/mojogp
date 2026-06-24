"""Large-scale multi-output benchmark profile focused on n-scaling."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.benchmarks.comparison_policy import policy_for
from tests.benchmarks.multi_output_scaling import run_multi_output_scaling_subprocess
from tests.benchmarks.multi_output_timeout_policy import multi_output_scaling_timeout_s
from tests.shared.subprocess_harness import run_isolated_case

from .conftest import (
    assert_gpu_available,
    assert_gpu_was_used,
    get_vram_tier,
    requires_cuda,
)
from tests.shared.benchmarking.report import load_benchmark_result, print_result, save_summary_report
from tests.shared.benchmarking.result_types import BenchmarkResult


def _assert_benchmark_methodology_fields(benchmark: BenchmarkResult) -> None:
    assert benchmark.speed.iter_timing_quality == "direct_per_iteration"
    assert benchmark.speed.iter_times_ms
    assert benchmark.speed.iter_time_p5_ms is not None
    assert benchmark.speed.iter_time_p95_ms is not None
    assert benchmark.memory.training_peak_gpu_mb is not None
    assert benchmark.memory.prediction_peak_gpu_mb is not None
    assert benchmark.config["phase_memory_quality"] == "phase_specific"
    if benchmark.config["prediction_mode"] == "exact":
        assert benchmark.memory.exact_prediction_peak_gpu_mb is not None
    elif benchmark.config["prediction_mode"] == "love":
        assert benchmark.memory.love_prediction_peak_gpu_mb is not None


BENCHMARK_TARGETS = {
    "xsmall": {
        "materialized": [1500, 3000],
        "matrix_free": [5000, 10000],
        "dims": [5],
        "num_tasks": 3,
    },
    "small": {
        "materialized": [3000, 5000],
        "matrix_free": [25000, 50000],
        "dims": [5],
        "num_tasks": 3,
    },
    "medium": {
        "materialized": [5000, 8000],
        "matrix_free": [50000, 75000],
        "dims": [7],
        "num_tasks": 4,
    },
    "large": {
        "materialized": [2000, 12000],
        "matrix_free": [50000, 100000],
        "dims": [12],
        "num_tasks": 6,
    },
    "xlarge": {
        "materialized": [20000, 30000],
        "matrix_free": [100000, 150000],
        "dims": [16],
        "num_tasks": 8,
    },
}


MATRIX_FREE_EXACT_SIZE_CAPS = {
    "mojogp": {
        "xsmall": [10000],
        "small": [25000],
        "medium": [50000],
        "large": [50000],
        "xlarge": [100000],
    },
    "gpytorch": {
        "xsmall": [1500],
        "small": [25000],
        "medium": [50000],
        "large": [50000],
        "xlarge": [100000],
    },
}


def _targets(method: str) -> tuple[list[int], list[int], int, str]:
    tier = get_vram_tier()
    targets = BENCHMARK_TARGETS[tier]
    return list(targets[method]), list(targets["dims"]), int(targets["num_tasks"]), tier


def _framework_sizes(framework: str, method: str, tier: str) -> list[int]:
    targets = BENCHMARK_TARGETS[tier]
    if framework != "gpytorch":
        return list(targets[method])

    gpytorch_sizes = {
        "xsmall": [1500],
        "small": [3000, 5000],
        "medium": [5000, 8000],
        "large": [2000],
        "xlarge": [20000, 30000],
    }
    return list(gpytorch_sizes[tier])


def _benchmark_sizes(
    *, framework: str, method: str, prediction_mode: str, tier: str
) -> list[int]:
    if method == "matrix_free" and prediction_mode == "exact":
        return list(MATRIX_FREE_EXACT_SIZE_CAPS[framework][tier])
    return _framework_sizes(framework, method, tier)


def _quick_enabled() -> bool:
    return os.environ.get("MOJOGP_BENCHMARK_QUICK", "0") == "1"


def _case_timeout_s(*, framework: str, method: str, prediction_mode: str, tier: str) -> int:
    return multi_output_scaling_timeout_s(
        framework=framework,
        method=method,
        prediction_mode=prediction_mode,
        tier=tier,
    )


def _run_case_subprocess(
    *,
    framework: str,
    prediction_mode: str,
    method: str,
    n_train: int,
    d: int,
    num_tasks: int,
    tier: str,
    results_dir,
) -> BenchmarkResult:
    return run_multi_output_scaling_subprocess(
        framework=framework,
        prediction_mode=prediction_mode,
        method=method,
        n_train=n_train,
        d=d,
        num_tasks=num_tasks,
        tier=tier,
        specialization=None,
        results_dir=results_dir,
        timeout_s=_case_timeout_s(
            framework=framework,
            method=method,
            prediction_mode=prediction_mode,
            tier=tier,
        ),
    )


def _run_sizes_for_framework(
    *,
    framework: str,
    method: str,
    prediction_mode: str,
    tier: str,
) -> list[int]:
    return _benchmark_sizes(
        framework=framework,
        method=method,
        prediction_mode=prediction_mode,
        tier=tier,
    )


@pytest.mark.multi_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_multi_output_scaling_benchmark(results_dir, method: str):
    assert_gpu_available()
    policy = policy_for("multi_output_scaling")
    sizes, dims, num_tasks, tier = _targets(method)
    if _quick_enabled():
        dims = dims[:1]

    results = []
    frameworks = (
        ["mojogp", "gpytorch"]
        if method == "materialized" and policy.published_cross_framework
        else ["mojogp"]
    )
    prediction_modes = ["exact", "love"]
    for d in dims:
        for prediction_mode in prediction_modes:
            if method == "materialized":
                mojogp_sizes = _run_sizes_for_framework(
                    framework="mojogp",
                    method=method,
                    prediction_mode=prediction_mode,
                    tier=tier,
                )
                gpytorch_sizes = _run_sizes_for_framework(
                    framework="gpytorch",
                    method=method,
                    prediction_mode=prediction_mode,
                    tier=tier,
                )
                fair_sizes = sorted(set(mojogp_sizes).intersection(gpytorch_sizes))
                mojogp_only_sizes = [
                    n_train for n_train in mojogp_sizes if n_train not in fair_sizes
                ]
                if _quick_enabled():
                    fair_sizes = fair_sizes[:1]
                    mojogp_only_sizes = mojogp_only_sizes[:1]

                for framework in frameworks:
                    for n_train in fair_sizes:
                        benchmark = _run_case_subprocess(
                            framework=framework,
                            prediction_mode=prediction_mode,
                            method=method,
                            n_train=n_train,
                            d=d,
                            num_tasks=num_tasks,
                            tier=tier,
                            results_dir=results_dir,
                        )
                        print_result(
                            benchmark,
                            title=(
                                f"Multi-output scaling: {framework} {method} "
                                f"pred={prediction_mode} n={n_train} d={d} T={num_tasks}"
                            ),
                        )
                        assert_gpu_was_used(benchmark)
                        assert np.isfinite(benchmark.accuracy.rmse)
                        assert np.isfinite(benchmark.accuracy.r_squared)
                        assert np.isfinite(benchmark.hyperparameters.final_nll)
                        _assert_benchmark_methodology_fields(benchmark)
                        results.append(benchmark)

                for n_train in mojogp_only_sizes:
                    benchmark = _run_case_subprocess(
                        framework="mojogp",
                        prediction_mode=prediction_mode,
                        method=method,
                        n_train=n_train,
                        d=d,
                        num_tasks=num_tasks,
                        tier=tier,
                        results_dir=results_dir,
                    )
                    print_result(
                        benchmark,
                        title=(
                            f"Multi-output scaling: mojogp {method} "
                            f"pred={prediction_mode} n={n_train} d={d} T={num_tasks}"
                        ),
                    )
                    assert_gpu_was_used(benchmark)
                    assert np.isfinite(benchmark.accuracy.rmse)
                    assert np.isfinite(benchmark.accuracy.r_squared)
                    assert np.isfinite(benchmark.hyperparameters.final_nll)
                    _assert_benchmark_methodology_fields(benchmark)
                    results.append(benchmark)
                continue

            for framework in frameworks:
                framework_sizes = _run_sizes_for_framework(
                    framework=framework,
                    method=method,
                    prediction_mode=prediction_mode,
                    tier=tier,
                )
                if _quick_enabled():
                    framework_sizes = framework_sizes[:1]
                for n_train in framework_sizes:
                    benchmark = _run_case_subprocess(
                        framework=framework,
                        prediction_mode=prediction_mode,
                        method=method,
                        n_train=n_train,
                        d=d,
                        num_tasks=num_tasks,
                        tier=tier,
                        results_dir=results_dir,
                    )
                    print_result(
                        benchmark,
                        title=(
                            f"Multi-output scaling: {framework} {method} "
                            f"pred={prediction_mode} n={n_train} d={d} T={num_tasks}"
                        ),
                    )
                    assert_gpu_was_used(benchmark)
                    assert np.isfinite(benchmark.accuracy.rmse)
                    assert np.isfinite(benchmark.accuracy.r_squared)
                    assert np.isfinite(benchmark.hyperparameters.final_nll)
                    _assert_benchmark_methodology_fields(benchmark)
                    results.append(benchmark)

    save_summary_report(results, results_dir, f"multi_output_scaling_{method}")

    for d in dims:
        for framework in frameworks:
            for prediction_mode in prediction_modes:
                per_dim = [
                    result
                    for result in results
                    if result.config["d"] == d
                    and result.config["framework"] == framework
                    and result.config["prediction_mode"] == prediction_mode
                ]
                if len(per_dim) >= 2:
                    small, large = per_dim[0], per_dim[-1]
                    assert large.config["n"] > small.config["n"]
