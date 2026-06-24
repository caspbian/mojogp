"""Harness benchmark for ExactGP save/load workflows."""

from __future__ import annotations

import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.benchmarks.workflow_runner import run_workflow_benchmark_subprocess
from tests.shared.gpu_test_utils import assert_gpu_available, assert_gpu_was_used, requires_cuda


@pytest.mark.minimal
@pytest.mark.single_output
@requires_cuda
@pytest.mark.parametrize("training_method", ["materialized", "matrix_free"])
def test_single_output_persistence_harness(training_method: str, results_dir):
    assert_gpu_available()
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_single_output_workflow_case",
        payload={
            "case": "persistence",
            "training_method": training_method,
            "n_train": 2000 if training_method == "materialized" else 3000,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "d": 5,
        },
        suite_name="single_output_workflow",
        benchmark_name="single_output_persistence_harness",
        framework="mojogp",
        case_id=f"single_output.persistence.{training_method}",
        benchmark_group_id=f"single_output.persistence.{training_method}",
        config={
            "framework": "mojogp",
            "training_method": training_method,
            "prediction_mode": "exact",
            "workflow": "persistence",
            "n": 2000 if training_method == "materialized" else 3000,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "d": 5,
        },
        results_dir=results_dir,
    )
    assert_gpu_was_used(benchmark)
    assert "benchmark_contracts" not in benchmark.config
    assert benchmark.config["variance_round_trip_policy"] == "hard_assertion"
    assert benchmark.config["max_abs_mean_diff"] <= benchmark.config["max_abs_mean_diff_threshold"]
    assert benchmark.config["max_abs_var_diff"] <= benchmark.config["max_abs_var_diff_threshold"]
    assert benchmark.speed.startup_prepare_time_s is not None
    if benchmark.speed.startup_compile_time_s is not None:
        assert benchmark.speed.startup_warm_cache_hit_s is not None
