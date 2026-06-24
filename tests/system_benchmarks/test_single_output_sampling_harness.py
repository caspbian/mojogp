"""Harness benchmark for ExactGP posterior sampling workflows."""

from __future__ import annotations

import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PATHWISE_SAMPLING_N_TEST
from tests.benchmarks.workflow_runner import run_workflow_benchmark_subprocess
from tests.shared.gpu_test_utils import assert_gpu_available, assert_gpu_was_used, requires_cuda


@pytest.mark.minimal
@pytest.mark.single_output
@requires_cuda
@pytest.mark.parametrize("training_method", ["materialized", "matrix_free"])
@pytest.mark.parametrize("sampling_method", ["diagonal", "pathwise"])
def test_single_output_sampling_harness(training_method: str, sampling_method: str, results_dir):
    assert_gpu_available()
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_single_output_workflow_case",
        payload={
            "case": "sampling",
            "training_method": training_method,
            "sampling_method": sampling_method,
            "n_train": 2000 if training_method == "materialized" else 3000,
            "n_test": BENCHMARK_PATHWISE_SAMPLING_N_TEST if sampling_method == "pathwise" else 96,
            "d": 5,
            "n_samples": 32,
            "n_rff_features": 512,
        },
        suite_name="single_output_workflow",
        benchmark_name="single_output_sampling_harness",
        framework="mojogp",
        case_id=f"single_output.sampling.{training_method}.{sampling_method}",
        benchmark_group_id=f"single_output.sampling.{training_method}.{sampling_method}",
        config={
            "framework": "mojogp",
            "training_method": training_method,
            "prediction_mode": sampling_method,
            "workflow": "sampling",
            "n": 2000 if training_method == "materialized" else 3000,
            "n_test": BENCHMARK_PATHWISE_SAMPLING_N_TEST if sampling_method == "pathwise" else 96,
            "d": 5,
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


@pytest.mark.minimal
@pytest.mark.single_output
@requires_cuda
@pytest.mark.parametrize("training_method", ["materialized", "matrix_free"])
def test_single_output_pathwise_covariance_matches_dense_reference(
    training_method: str, results_dir
):
    assert_gpu_available()
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_single_output_workflow_case",
        payload={
            "case": "pathwise_covariance_sanity",
            "training_method": training_method,
            "n_train": 2000,
            "n_samples": 64,
            "n_rff_features": 2048,
            "max_iterations": 20,
        },
        suite_name="single_output_workflow",
        benchmark_name="single_output_pathwise_covariance_sanity",
        framework="mojogp",
        case_id=f"single_output.pathwise_covariance.{training_method}",
        benchmark_group_id=f"single_output.pathwise_covariance.{training_method}",
        config={
            "framework": "mojogp",
            "training_method": training_method,
            "prediction_mode": "pathwise",
            "workflow": "pathwise_covariance_sanity",
            "n": 2000,
            "n_test": 6,
            "d": 1,
        },
        results_dir=results_dir,
    )
    assert_gpu_was_used(benchmark)
    assert benchmark.config["sampling_consistency_policy"] == "dense_covariance_hard_assertion"
    assert benchmark.config["actual_sampling_route"] == "provider_pathwise"
    assert benchmark.accuracy.rmse <= benchmark.config["sample_mean_rmse_threshold"]
    assert benchmark.accuracy.crps <= benchmark.config["sample_var_rel_rmse_threshold"]
    assert benchmark.accuracy.calibration_error <= benchmark.config["sample_corr_rmse_threshold"]
    assert benchmark.config["empirical_close_corr"] >= benchmark.config["min_empirical_close_corr"]
    assert benchmark.config["close_far_corr_gap"] >= benchmark.config["close_far_corr_gap_threshold"]
