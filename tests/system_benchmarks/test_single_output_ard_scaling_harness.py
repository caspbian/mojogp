"""Single-output ARD scaling benchmark surface."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.benchmarks.comparison_policy import policy_for
from tests.benchmarks.single_output_ard_scaling import (
    run_single_output_ard_scaling_subprocess,
)
from tests.shared.benchmarking.report import print_result, save_summary_report
from tests.shared.benchmarking.result_types import BenchmarkResult

from .conftest import (
    assert_gpu_available,
    assert_gpu_was_used,
    get_matrix_free_capability_tier,
    get_vram_tier,
    requires_cuda,
)
from .test_scaling_certification_harness import (
    _allow_recorded_failure,
    _assert_matrix_free_prediction_memory_scaling,
    _prediction_modes_for_framework,
    _size_role,
)


ARD_DIM_CONFIGS = {
    "minimal": [(9, 3)],
    "standard": [(9, 3), (17, 5)],
    "extensive": [(9, 3), (17, 5), (31, 7)],
}


MATERIALIZED_ARD_TARGETS = {
    "minimal": {
        "xsmall": [5000],
        "small": [8000],
        "medium": [12000],
        "large": [20000],
        "xlarge": [30000],
    },
    "standard": {
        "xsmall": [5000, 8000],
        "small": [8000, 12000],
        "medium": [12000, 20000],
        "large": [20000, 35000],
        "xlarge": [30000, 60000],
    },
    "extensive": {
        "xsmall": [5000, 8000],
        "small": [8000, 12000],
        "medium": [12000, 20000],
        "large": [20000, 35000],
        "xlarge": [30000, 60000],
    },
}


GPYTORCH_MATERIALIZED_ARD_TARGETS = {
    variety: {tier: list(sizes) for tier, sizes in tiers.items()}
    for variety, tiers in MATERIALIZED_ARD_TARGETS.items()
}
GPYTORCH_MATERIALIZED_ARD_TARGETS["standard"]["large"] = [20000]
GPYTORCH_MATERIALIZED_ARD_TARGETS["extensive"]["large"] = [20000]


MATRIX_FREE_ARD_TARGETS = {
    "minimal": {
        "xsmall": [5000],
        "small": [10000],
        "medium": [25000],
        "large": [25000],
        "xlarge": [50000],
    },
    "standard": {
        "xsmall": [5000, 10000],
        "small": [10000, 25000],
        "medium": [25000, 50000],
        "large": [25000, 50000],
        "xlarge": [50000, 75000],
    },
    "extensive": {
        "xsmall": [5000, 10000],
        "small": [10000, 25000],
        "medium": [25000, 50000],
        "large": [25000, 50000],
        "xlarge": [50000, 75000],
    },
}


def _benchmark_variety() -> str:
    variety = os.environ.get("MOJOGP_BENCHMARK_VARIETY")
    if variety is not None:
        normalized = variety.strip().lower()
        if normalized not in ARD_DIM_CONFIGS:
            raise ValueError(
                "MOJOGP_BENCHMARK_VARIETY must be one of "
                f"{sorted(ARD_DIM_CONFIGS)}, got '{variety}'"
            )
        return normalized
    if os.environ.get("MOJOGP_BENCHMARK_QUICK", "0") == "1":
        return "minimal"
    return "standard"


def _route_tier_and_policy(method: str) -> tuple[str, str]:
    if method == "materialized":
        return get_vram_tier(), "vram"
    return get_matrix_free_capability_tier(), "bandwidth"


def _framework_sizes(
    *,
    framework: str,
    method: str,
    prediction_mode: str,
    tier: str,
    benchmark_variety: str,
) -> list[int]:
    if framework == "gpytorch" and method == "matrix_free" and prediction_mode != "love":
        return []
    if method == "materialized":
        if framework == "gpytorch":
            return list(GPYTORCH_MATERIALIZED_ARD_TARGETS[benchmark_variety][tier])
        return list(MATERIALIZED_ARD_TARGETS[benchmark_variety][tier])
    return list(MATRIX_FREE_ARD_TARGETS[benchmark_variety][tier])


def _run_case_capture(
    *,
    method: str,
    n_train: int,
    d: int,
    relevant_dims: int,
    framework: str,
    prediction_mode: str,
    tier: str,
    benchmark_variety: str,
    n_selection_policy: str,
    size_role: str,
    allow_recorded_failure: bool,
    results_dir,
) -> tuple[BenchmarkResult | None, dict[str, object] | None]:
    try:
        result = run_single_output_ard_scaling_subprocess(
            method=method,
            n_train=n_train,
            d=d,
            relevant_dims=relevant_dims,
            framework=framework,
            prediction_mode=prediction_mode,
            tier=tier,
            benchmark_variety=benchmark_variety,
            benchmark_track="scaling",
            n_selection_policy=n_selection_policy,
            size_role=size_role,
            max_iterations=100,
            enable_early_stopping=False,
            results_dir=results_dir,
        )
        return result, None
    except Exception as exc:
        failure = {
            "framework": framework,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "benchmark_track": "scaling",
            "benchmark_variety": benchmark_variety,
            "benchmark_route_tier": tier,
            "n_selection_policy": n_selection_policy,
            "size_role": size_role,
            "n": n_train,
            "d": d,
            "relevant_dims": relevant_dims,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if allow_recorded_failure:
            return None, failure
        raise


@pytest.mark.minimal
@pytest.mark.ard
@pytest.mark.single_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_single_output_ard_scaling_benchmark(results_dir, method: str):
    assert_gpu_available()
    policy = policy_for("single_output_ard_scaling")
    benchmark_variety = _benchmark_variety()
    dim_configs = ARD_DIM_CONFIGS[benchmark_variety]
    tier, n_selection_policy = _route_tier_and_policy(method)
    frameworks = ["mojogp", "gpytorch"] if policy.published_cross_framework else ["mojogp"]

    results: list[BenchmarkResult] = []
    failures: list[dict[str, object]] = []
    for d, relevant_dims in dim_configs:
        for framework in frameworks:
            for prediction_mode in _prediction_modes_for_framework(
                framework=framework, method=method
            ):
                framework_sizes = _framework_sizes(
                    framework=framework,
                    method=method,
                    prediction_mode=prediction_mode,
                    tier=tier,
                    benchmark_variety=benchmark_variety,
                )
                for n_train in framework_sizes:
                    size_role = _size_role(n_train, framework_sizes)
                    allow_recorded_failure = _allow_recorded_failure(
                        framework=framework,
                        method=method,
                        prediction_mode=prediction_mode,
                        benchmark_variety=benchmark_variety,
                        size_role=size_role,
                    )
                    result, failure = _run_case_capture(
                        method=method,
                        n_train=n_train,
                        d=d,
                        relevant_dims=relevant_dims,
                        framework=framework,
                        prediction_mode=prediction_mode,
                        tier=tier,
                        benchmark_variety=benchmark_variety,
                        n_selection_policy=n_selection_policy,
                        size_role=size_role,
                        allow_recorded_failure=allow_recorded_failure,
                        results_dir=results_dir,
                    )
                    if failure is not None:
                        failures.append(failure)
                        continue
                    assert result is not None
                    print_result(
                        result,
                        title=(
                            f"Single-output ARD scaling: {framework} {method} "
                            f"pred={prediction_mode} n={n_train} d={d} rel={relevant_dims}"
                        ),
                    )
                    assert_gpu_was_used(result)
                    results.append(result)

    if method == "matrix_free":
        _assert_matrix_free_prediction_memory_scaling(results)

    save_summary_report(
        results,
        results_dir,
        f"single_output_ard_scaling_{benchmark_variety}_{method}",
        failures=failures,
    )
    assert results, f"No successful single-output ARD results collected for {method}"

    for result in results:
        if result.config.get("comparison_class") == "unsupported_comparator":
            continue
        assert np.isfinite(result.accuracy.rmse)
        assert np.isfinite(result.accuracy.r_squared)
        assert np.isfinite(result.hyperparameters.final_nll)
        if result.config.get("framework") == "mojogp":
            assert result.config.get("pairwise_relevance_accuracy", 0.0) >= 0.5
