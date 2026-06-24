"""Benchmark harness that persists isolated VRAM measurements."""

from __future__ import annotations

import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess
from tests.shared.benchmarking.gpytorch_models import is_keops_available

from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda


@pytest.mark.minimal
@requires_cuda
@pytest.mark.parametrize(
    ("framework", "method", "prediction_mode"),
    [
        ("mojogp", "materialized", "exact"),
        ("mojogp", "matrix_free", "exact"),
        ("gpytorch", "materialized", "exact"),
    ],
)
def test_isolated_vram_measurement_rows_persist(
    results_dir,
    framework: str,
    method: str,
    prediction_mode: str,
):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_isolated_vram_measurement_case",
        payload={
            "framework": framework,
            "n": 2000,
            "d": 5,
            "method": method,
            "prediction_mode": prediction_mode,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "max_iterations": 1,
        },
        suite_name="isolated_vram_measurement",
        benchmark_name="isolated_vram_measurement",
        framework=framework,
        case_id=f"{framework}.isolated_vram.{method}.{prediction_mode}.n2000.d5",
        benchmark_group_id=f"{framework}.isolated_vram.{method}.{prediction_mode}",
        config={
            "benchmark": "isolated_vram_measurement",
            "framework": framework,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "n": 2000,
            "d": 5,
        },
        results_dir=results_dir,
        timeout=180,
    )

    assert result.config["suite_name"] == "isolated_vram_measurement"
    assert result.memory.gpu_max_mb >= result.memory.gpu_baseline_mb
    assert result.memory.gpu_delta_mb >= 0.0
    assert result.memory.training_delta_gpu_mb is not None
    assert result.memory.prediction_peak_gpu_mb is not None
    assert_gpu_was_used(result)


@pytest.mark.moderate
@requires_cuda
@pytest.mark.parametrize(
    ("framework", "method", "prediction_mode", "n"),
    [
        ("mojogp", "materialized", "exact", 5000),
        ("mojogp", "matrix_free", "exact", 8000),
        ("gpytorch", "materialized", "exact", 5000),
    ],
)
def test_isolated_vram_measurement_memory_law_rows_persist(
    results_dir,
    framework: str,
    method: str,
    prediction_mode: str,
    n: int,
):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_isolated_vram_measurement_case",
        payload={
            "framework": framework,
            "n": n,
            "d": 5,
            "method": method,
            "prediction_mode": prediction_mode,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "max_iterations": 1,
        },
        suite_name="isolated_vram_measurement",
        benchmark_name="isolated_vram_measurement",
        framework=framework,
        case_id=f"{framework}.isolated_vram.{method}.{prediction_mode}.n{n}.d5",
        benchmark_group_id=f"{framework}.isolated_vram.{method}.{prediction_mode}",
        config={
            "benchmark": "isolated_vram_measurement",
            "framework": framework,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "n": n,
            "d": 5,
        },
        results_dir=results_dir,
        timeout=240,
    )

    assert result.memory.prediction_peak_gpu_mb is not None
    assert result.memory.prediction_delta_gpu_mb is not None
    assert_gpu_was_used(result)


@pytest.mark.minimal
@requires_cuda
@pytest.mark.skipif(not is_keops_available(), reason="pykeops not installed")
def test_isolated_vram_measurement_keops_row_persists(results_dir):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_isolated_vram_measurement_case",
        payload={
            "framework": "gpytorch_keops",
            "n": 2000,
            "d": 5,
            "method": "matrix_free",
            "prediction_mode": "love",
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "max_iterations": 1,
        },
        suite_name="isolated_vram_measurement",
        benchmark_name="isolated_vram_measurement",
        framework="gpytorch",
        case_id="gpytorch_keops.isolated_vram.matrix_free.love.n2000.d5",
        benchmark_group_id="gpytorch_keops.isolated_vram.matrix_free.love",
        config={
            "benchmark": "isolated_vram_measurement",
            "framework": "gpytorch_keops",
            "training_method": "matrix_free",
            "prediction_mode": "love",
            "n": 2000,
            "d": 5,
        },
        results_dir=results_dir,
        timeout=180,
    )

    assert result.memory.love_prediction_peak_gpu_mb is not None
    assert result.memory.love_prediction_peak_gpu_mb >= 0.0
