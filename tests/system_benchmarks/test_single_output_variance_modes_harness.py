"""Harness benchmark for single-output variance modes.

The suite uses the public ExactGP wrapper and shared GPyTorch helpers to compare
exact and LOVE variance behavior on the active benchmarking surface.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from tests.benchmarks.comparison_policy import policy_for
from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess
from mojogp import SingleOutputGP, RBF

from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda
from tests.shared.benchmarking.data_generators import generate_structured_function_data
from tests.shared.benchmarking.gpu_memory import GPUMemoryMonitor, get_torch_memory_stats, reset_torch_memory_stats
from tests.shared.benchmarking.gpytorch_models import predict_gpytorch_single_output, train_gpytorch_single_output
from tests.shared.benchmarking.metrics import compute_all_accuracy_metrics
from tests.shared.benchmarking.report import print_result
from tests.shared.benchmarking.result_types import AccuracyResult, BenchmarkResult, HyperparameterResult, MemoryResult, SpeedResult


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _build_love_dataset(seed: int = 42):
    return generate_structured_function_data(
        n_train=2000,
        n_test=BENCHMARK_PREDICTION_N_TEST,
        d=5,
        function_type="friedman1",
        noise_level="low",
        seed=seed,
    )


def _benchmark_from_prediction(
    *,
    dataset,
    method: str,
    prediction_mode: str,
    training_time_s: float,
    prediction_total_time_s: float,
    iterations_run: int,
    max_iterations: int,
    memory_stats: dict[str, float],
    mean: np.ndarray,
    variance: np.ndarray,
    learned_params: dict[str, float],
    final_nll: float,
    baseline_config: dict[str, object],
) -> BenchmarkResult:
    policy = policy_for("love_variance_comparison")
    std = np.sqrt(np.maximum(variance, 1e-10))
    accuracy_metrics = compute_all_accuracy_metrics(
        dataset.f_test,
        mean,
        std,
        y_train_mean=float(np.mean(dataset.y_train)),
        y_train_std=float(np.std(dataset.y_train)),
    )
    return BenchmarkResult(
        config={
            "benchmark": "love_variance_comparison",
            "route_group": "single_output",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": prediction_mode,
            "comparison_class": policy.comparator_type,
            "baseline_backend": "none" if not policy.published_cross_framework else "gpytorch_cg",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": (
                "N.B. MojoGP-only LOVE row: this benchmark compares LOVE against the exact "
                "MojoGP variance route on the same trained wrapper state."
            ),
            "fairness_axes": {
                "comparator_scope": _fairness_axis(
                    "mojogp_only",
                    "The baseline is MojoGP exact variance on the same trained wrapper state.",
                ),
                "sample_count_n": _fairness_axis(
                    "aligned",
                    "LOVE and exact predictions use the same train/test split and model state.",
                ),
                "optimizer": _fairness_axis(
                    "aligned",
                    "The comparison reuses the same trained model rather than retraining separate baselines.",
                ),
                "solver_budget": _fairness_axis(
                    "aligned",
                    "Both predictive modes use the same training-state CG/Lanczos configuration where applicable.",
                ),
                "preconditioner": _fairness_axis(
                    "aligned",
                    "Both predictive modes use the same trained model and preconditioner state.",
                ),
                "prediction_mode": _fairness_axis(
                    "varied",
                    "This suite explicitly contrasts MojoGP LOVE against MojoGP exact predictive variance.",
                ),
                "telemetry": _fairness_axis(
                    "observed",
                    "MojoGP telemetry is observed for the active prediction route.",
                ),
            },
            "n": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            **baseline_config,
        },
        accuracy=AccuracyResult(
            rmse=float(accuracy_metrics["rmse"]),
            mae=float(accuracy_metrics["mae"]),
            r_squared=float(accuracy_metrics["r_squared"]),
            crps=float(accuracy_metrics["crps"]),
            msll=float(accuracy_metrics["msll"]),
            calibration_coverage={
                0.5: float(accuracy_metrics["calibration_50"]),
                0.9: float(accuracy_metrics["calibration_90"]),
                0.95: float(accuracy_metrics["calibration_95"]),
                0.99: float(accuracy_metrics["calibration_99"]),
            },
            calibration_error=float(accuracy_metrics["calibration_error"]),
            sharpness=float(accuracy_metrics["sharpness"]),
            interval_width_95=float(accuracy_metrics["interval_width_95"]),
        ),
        speed=SpeedResult(
            training_time_s=float(training_time_s),
            prediction_mean_time_s=float(prediction_total_time_s),
            prediction_variance_time_s=float(prediction_total_time_s),
            end_to_end_time_s=float(training_time_s + prediction_total_time_s),
            iterations_run=int(iterations_run),
            max_iterations=int(max_iterations),
            early_stopped=int(iterations_run) < int(max_iterations),
            ms_per_iteration=float(training_time_s / max(int(iterations_run), 1) * 1000.0),
        ),
        memory=MemoryResult(
            gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
            gpu_min_mb=memory_stats.get("min_mb", 0.0),
            gpu_max_mb=memory_stats.get("max_mb", 0.0),
            gpu_var_mb=memory_stats.get("var_mb", 0.0),
            torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
            torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
            cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
            measurement_method=memory_stats.get("method", "none"),
            num_samples=int(memory_stats.get("samples", 0)),
        ),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(
                learned_params.get("lengthscale", learned_params.get("rbf_lengthscale", 1.0))
            ),
            learned_noise=float(learned_params["noise"]),
            learned_outputscale=float(
                learned_params.get("outputscale", learned_params.get("rbf_outputscale", 1.0))
            ),
            learned_mean=float(learned_params.get("mean", 0.0)),
            final_nll=float(final_nll),
        ),
    )


@pytest.mark.minimal
@pytest.mark.single_output
@pytest.mark.accuracy
@pytest.mark.speed
@pytest.mark.memory
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_love_matches_exact_route_reasonably(method: str, results_dir):
    assert_gpu_available()
    policy = policy_for("love_variance_comparison")
    assert policy.published_cross_framework is False
    benchmark = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_variance_modes_case",
        payload={"case": "love_vs_exact", "method": method},
        suite_name="single_output_variance_modes_harness",
        benchmark_name="love_variance_comparison",
        framework="mojogp",
        case_id=f"single_output.variance_modes.love_vs_exact.{method}",
        benchmark_group_id=f"single_output.variance_modes.love_vs_exact.{method}",
        config={
            "benchmark": "love_variance_comparison",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "prediction_mode": "love",
        },
        results_dir=results_dir,
    )
    print_result(benchmark, title=f"LOVE route benchmark: {method}")
    assert_gpu_was_used(benchmark)
    assert benchmark.config["mean_rmse_vs_exact"] < 0.15
    assert 0.2 < benchmark.config["variance_mean_ratio_vs_exact"] < 5.0
