"""Harness benchmark for MultiOutputLMCGP per-task-noise workflows."""

from __future__ import annotations

import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.benchmarks.workflow_runner import run_workflow_benchmark_subprocess
from tests.shared.benchmarking.report import print_result
from tests.shared.gpu_test_utils import assert_gpu_available, assert_gpu_was_used, requires_cuda


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_per_task_noise_recovery_orders_tasks(method: str, results_dir):
    """LMC should recover the relative ordering of learned task noises."""
    assert_gpu_available()
    n_train = 2000 if method == "materialized" else 2500
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_lmc_workflow_case",
        payload={
            "case": "per_task_noise",
            "method": method,
            "n_train": n_train,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "d": 5,
            "noise_per_task": [0.015, 0.06, 0.14],
            "task_correlation": "medium",
            "max_iterations": 20,
            "num_probes": 3,
            "max_cg_iterations": 30,
        },
        suite_name="lmc_workflow",
        benchmark_name="lmc_per_task_noise",
        framework="mojogp",
        case_id=f"lmc.per_task_noise.{method}",
        benchmark_group_id=f"lmc.per_task_noise.{method}",
        config={
            "framework": "mojogp",
            "training_method": method,
            "prediction_mode": "love",
            "workflow": "per_task_noise_recovery",
            "n": n_train,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "d": 5,
            "num_tasks": 3,
        },
        results_dir=results_dir,
    )
    print_result(benchmark)

    assert_gpu_was_used(benchmark)
    assert benchmark.speed.iter_timing_quality == "direct_per_iteration"
    assert not benchmark.speed.early_stopped
    assert benchmark.accuracy.rmse < 0.75
    assert benchmark.config["noise_recovery_claim"] == "ordering_not_absolute_magnitude"
    assert benchmark.config["noise_rank_correlation"] >= 0.5
    assert benchmark.config["backend_train_info"]["training_route"] == method
    assert benchmark.config["backend_predict_info"]["actual_variance_route"] == "predict_lmc"
    assert (
        benchmark.config["backend_predict_info"]["lmc_variance_exactness"]
        == "scalar_latent_approximation"
    )
