"""Harness benchmark for MultiOutputLMCGP posterior sampling workflows."""

from __future__ import annotations

import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PATHWISE_SAMPLING_N_TEST
from tests.benchmarks.workflow_runner import run_workflow_benchmark_subprocess
from tests.shared.gpu_test_utils import assert_gpu_available, assert_gpu_was_used, requires_cuda


@pytest.mark.minimal
@pytest.mark.multi_output
@requires_cuda
@pytest.mark.parametrize("sampling_method", ["diagonal", "pathwise"])
def test_lmc_sampling_harness(sampling_method: str, results_dir):
    assert_gpu_available()
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_lmc_workflow_case",
        payload={
            "case": "sampling",
            "sampling_method": sampling_method,
            "n_train": 700,
            "n_test": BENCHMARK_PATHWISE_SAMPLING_N_TEST if sampling_method == "pathwise" else 96,
            "n_samples": 24,
            "n_rff_features": 512,
        },
        suite_name="lmc_workflow",
        benchmark_name="lmc_sampling_harness",
        framework="mojogp",
        case_id=f"lmc.sampling.materialized.{sampling_method}",
        benchmark_group_id=f"lmc.sampling.materialized.{sampling_method}",
        config={
            "framework": "mojogp",
            "training_method": "materialized",
            "prediction_mode": sampling_method,
            "workflow": "sampling",
            "n": 700,
            "n_test": BENCHMARK_PATHWISE_SAMPLING_N_TEST if sampling_method == "pathwise" else 96,
            "d": 1,
            "num_tasks": 3,
        },
        results_dir=results_dir,
    )
    assert_gpu_was_used(benchmark)
    assert "benchmark_contracts" not in benchmark.config
    assert benchmark.config["sampling_consistency_policy"] == "hard_assertion"
    assert benchmark.accuracy.rmse <= benchmark.config["sample_mean_rmse_threshold"]
    assert benchmark.accuracy.crps <= benchmark.config["sample_std_rmse_threshold"]
    assert benchmark.speed.startup_prepare_time_s is not None
    if benchmark.speed.startup_compile_time_s is not None:
        assert benchmark.speed.startup_warm_cache_hit_s is not None


@pytest.mark.minimal
@pytest.mark.multi_output
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_ard_relevance_harness(method: str, results_dir):
    assert_gpu_available()
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_lmc_workflow_case",
        payload={
            "case": "ard_relevance",
            "method": method,
            "n_train": 2000,
            "d": 3,
            "num_tasks": 2,
            "seed": 70,
            "max_iterations": 120,
            "learning_rate": 0.003,
            "num_probes": 16,
            "max_cg_iterations": 100,
        },
        suite_name="lmc_workflow",
        benchmark_name="lmc_ard_relevance_harness",
        framework="mojogp",
        case_id=f"lmc.ard_relevance.{method}.rbf",
        benchmark_group_id=f"lmc.ard_relevance.{method}.rbf",
        config={
            "framework": "mojogp",
            "training_method": method,
            "prediction_mode": "none",
            "workflow": "ard_relevance",
            "n": 2000,
            "n_test": 0,
            "d": 3,
            "num_tasks": 2,
        },
        results_dir=results_dir,
    )
    assert_gpu_was_used(benchmark)
    assert benchmark.config["metadata_policy"] == "hard_assertion"
    assert benchmark.config["relevance_policy"] == "hard_assertion"
    assert benchmark.config["training_route"] == method
    assert benchmark.config["lengthscales_finite"] is True
    assert benchmark.config["lengthscales_positive"] is True
    assert benchmark.config["relevance_margin"] >= benchmark.config["relevance_margin_threshold"]
    assert benchmark.speed.ms_per_iteration is not None
