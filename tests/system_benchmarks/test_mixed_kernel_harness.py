"""Harness benchmark for mixed-kernel ablations.

These tests verify that discrete-aware kernels improve modelling on datasets with
real categorical structure instead of merely exercising the mixed code path.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess

from .conftest import assert_gpu_available, requires_cuda
from tests.shared.benchmarking.report import print_result


@pytest.mark.minimal
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_mixed_kernel_beats_continuous_only_baseline(method: str, results_dir):
    assert_gpu_available()
    benchmark = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_mixed_kernel_case",
        payload={"method": method},
        suite_name="mixed_kernel_harness",
        benchmark_name="mixed_accuracy",
        framework="mojogp",
        case_id=f"mixed_accuracy.{method}",
        benchmark_group_id=f"mixed_accuracy.{method}",
        config={
            "benchmark": "mixed_accuracy",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf_x_ehh",
            "training_method": method,
            "prediction_mode": "love",
        },
        results_dir=results_dir,
    )
    print_result(benchmark)

    baseline_rmse = float(benchmark.config["baseline_rmse"])
    baseline_r2 = float(benchmark.config["baseline_r2"])
    assert benchmark.accuracy.rmse <= baseline_rmse * 0.95, (
        f"Mixed kernel did not improve enough over continuous-only baseline on {method}: "
        f"mixed_rmse={benchmark.accuracy.rmse:.4f}, baseline_rmse={baseline_rmse:.4f}"
    )
    assert benchmark.accuracy.r_squared >= baseline_r2 - 0.05, (
        f"Mixed kernel unexpectedly regressed in R^2 on {method}: mixed={benchmark.accuracy.r_squared:.4f}, baseline={baseline_r2:.4f}"
    )
