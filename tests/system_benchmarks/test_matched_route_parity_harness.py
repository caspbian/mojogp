"""Matched-data route parity rows for MojoGP, GPyTorch CG, and KeOps."""

from __future__ import annotations

import numpy as np
import pytest

from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess
from tests.shared.benchmarking.gpytorch_models import is_keops_available

from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda


@pytest.mark.minimal
@requires_cuda
@pytest.mark.parametrize("framework,method", [("mojogp", "materialized"), ("gpytorch", "materialized")])
def test_matched_route_parity_materialized_exact_row(results_dir, framework: str, method: str):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_matched_route_parity_case",
        payload={
            "framework": framework,
            "method": method,
            "prediction_mode": "exact",
            "n": 2000,
            "d": 5,
        },
        suite_name="mojogp_route_parity",
        benchmark_name="matched_data_route_parity",
        framework="gpytorch" if framework.startswith("gpytorch") else framework,
        case_id=f"{framework}.matched_route_parity.{method}.exact.n2000.d5",
        benchmark_group_id=f"matched_route_parity.{method}.exact.d5",
        config={
            "suite_name": "mojogp_route_parity",
            "benchmark": "matched_data_route_parity",
            "framework": framework,
            "training_method": method,
            "prediction_mode": "exact",
            "n": 2000,
            "d": 5,
        },
        results_dir=results_dir,
        timeout=300,
    )

    assert result.config["suite_name"] == "mojogp_route_parity"
    assert result.config["benchmark"] == "matched_data_route_parity"
    assert np.isfinite(result.accuracy.rmse)
    assert_gpu_was_used(result)


@pytest.mark.moderate
@requires_cuda
@pytest.mark.parametrize("d", [5, 17, 31])
@pytest.mark.parametrize("framework,method", [("mojogp", "materialized"), ("mojogp", "matrix_free"), ("gpytorch", "materialized")])
def test_matched_route_parity_artifact_dimensions(results_dir, framework: str, method: str, d: int):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_matched_route_parity_case",
        payload={
            "framework": framework,
            "method": method,
            "prediction_mode": "exact",
            "n": 5000,
            "d": d,
        },
        suite_name="mojogp_route_parity",
        benchmark_name="matched_data_route_parity",
        framework="gpytorch" if framework.startswith("gpytorch") else framework,
        case_id=f"{framework}.matched_route_parity.{method}.exact.n5000.d{d}",
        benchmark_group_id=f"matched_route_parity.{method}.exact.d{d}",
        config={
            "suite_name": "mojogp_route_parity",
            "benchmark": "matched_data_route_parity",
            "framework": framework,
            "training_method": method,
            "prediction_mode": "exact",
            "n": 5000,
            "d": d,
        },
        results_dir=results_dir,
        timeout=600,
    )

    assert result.config["d"] == d
    assert np.isfinite(result.accuracy.rmse)
    assert_gpu_was_used(result)


@pytest.mark.moderate
@requires_cuda
@pytest.mark.skipif(not is_keops_available(), reason="pykeops not installed")
def test_matched_route_parity_keops_row(results_dir):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_matched_route_parity_case",
        payload={
            "framework": "gpytorch_keops",
            "method": "matrix_free",
            "prediction_mode": "love",
            "n": 5000,
            "d": 17,
        },
        suite_name="mojogp_route_parity",
        benchmark_name="matched_data_route_parity",
        framework="gpytorch",
        case_id="gpytorch_keops.matched_route_parity.matrix_free.love.n5000.d17",
        benchmark_group_id="matched_route_parity.matrix_free.love.d17",
        config={
            "suite_name": "mojogp_route_parity",
            "benchmark": "matched_data_route_parity",
            "framework": "gpytorch_keops",
            "training_method": "matrix_free",
            "prediction_mode": "love",
            "n": 5000,
            "d": 17,
        },
        results_dir=results_dir,
        timeout=600,
    )

    assert result.config["fairness_note"].startswith("N.B.")
    assert result.memory.prediction_peak_gpu_mb is not None
    assert_gpu_was_used(result)
