"""Harness benchmark for MultiOutputGP posterior sampling workflows."""

from __future__ import annotations

import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PATHWISE_SAMPLING_N_TEST
from tests.benchmarks.workflow_runner import run_workflow_benchmark_subprocess
from tests.shared.gpu_test_utils import assert_gpu_available, assert_gpu_was_used, requires_cuda


def _sampling_payload(training_method: str, sampling_method: str) -> dict[str, int | str]:
    if sampling_method == "pathwise":
        return {
            "case": "sampling",
            "training_method": training_method,
            "sampling_method": sampling_method,
            "n_train": 2000,
            "n_test": BENCHMARK_PATHWISE_SAMPLING_N_TEST,
            "d": 2,
            "num_tasks": 2,
            "n_samples": 4,
            "n_rff_features": 1024,
        }
    return {
        "case": "sampling",
        "training_method": training_method,
        "sampling_method": sampling_method,
        "n_train": 800 if training_method == "materialized" else 1200,
        "n_test": 96,
        "d": 5,
        "num_tasks": 3,
        "n_samples": 24,
        "n_rff_features": 512,
    }


@pytest.mark.minimal
@pytest.mark.multi_output
@requires_cuda
@pytest.mark.parametrize("training_method", ["materialized", "matrix_free"])
@pytest.mark.parametrize("sampling_method", ["diagonal", "pathwise"])
def test_multi_output_sampling_harness(training_method: str, sampling_method: str, results_dir):
    assert_gpu_available()
    payload = _sampling_payload(training_method, sampling_method)
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_multi_output_workflow_case",
        payload=payload,
        suite_name="multi_output_workflow",
        benchmark_name="multi_output_sampling_harness",
        framework="mojogp",
        case_id=f"multi_output.sampling.{training_method}.{sampling_method}",
        benchmark_group_id=f"multi_output.sampling.{training_method}.{sampling_method}",
        config={
            "framework": "mojogp",
            "training_method": training_method,
            "prediction_mode": sampling_method,
            "workflow": "sampling",
            "n": int(payload["n_train"]),
            "n_test": int(payload["n_test"]),
            "d": int(payload["d"]),
            "num_tasks": int(payload["num_tasks"]),
        },
        results_dir=results_dir,
    )
    assert_gpu_was_used(benchmark)
    assert "benchmark_contracts" not in benchmark.config
    assert benchmark.config["sampling_consistency_policy"] == "hard_assertion"
    assert benchmark.accuracy.rmse <= benchmark.config["sample_mean_rmse_threshold"]
    assert benchmark.accuracy.crps <= benchmark.config["sample_std_rmse_threshold"]
    expected_route = (
        "provider_pathwise"
        if sampling_method == "pathwise"
        else "diagonal_from_predictive_std"
    )
    assert benchmark.config["actual_sampling_route"] == expected_route
    assert benchmark.speed.startup_prepare_time_s is not None
    if benchmark.speed.startup_compile_time_s is not None:
        assert benchmark.speed.startup_warm_cache_hit_s is not None
