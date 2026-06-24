"""Harness benchmark for MojoGP route parity.

These benchmarks compare MojoGP against itself at modest sizes where both
materialized and matrix-free routes fit comfortably on the local GPU.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest

from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess
from mojogp import SingleOutputGP, MultiOutputGP, RBF
from mojogp.gp import (
    _DEFAULT_EXACT_PREDICT_CG_TOL,
    _DEFAULT_EXACT_PREDICT_MAX_CG_ITER,
)

from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda
from tests.shared.benchmarking.data_generators import (
    generate_multi_output_data,
    generate_structured_function_data,
)
from tests.shared.benchmarking.gpu_memory import (
    GPUMemoryMonitor,
    get_torch_memory_stats,
    reset_torch_memory_stats,
)
from tests.shared.benchmarking.metrics import (
    calibration_coverage,
    calibration_error,
    crps_gaussian,
    mae,
    mean_standardized_log_loss,
    rmse,
    r_squared,
    sharpness,
    interval_width,
)
from tests.shared.benchmarking.mojogp_runners import normalize_single_output_benchmark_hparams
from tests.shared.benchmarking.report import print_result, save_summary_report
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)


PARITY_SOLVER = {
    "cg_tolerance": 1e-2,
    "max_cg_iterations": 100,
    "num_trace_samples": 10,
    "max_lanczos_quadrature_iterations": 20,
    "precond_rank": 0,
    "precond_method": 0,
    "precond": "auto",
    "use_preconditioner": False,
}


def _prediction_solver_metadata(prediction_mode: str) -> dict[str, float | int]:
    if prediction_mode == "exact":
        return {
            "cg_tolerance": _DEFAULT_EXACT_PREDICT_CG_TOL,
            "max_cg_iterations": _DEFAULT_EXACT_PREDICT_MAX_CG_ITER,
            "precond_rank": PARITY_SOLVER["precond_rank"],
            "precond_method": PARITY_SOLVER["precond_method"],
        }
    return {
        "cg_tolerance": PARITY_SOLVER["cg_tolerance"],
        "max_cg_iterations": PARITY_SOLVER["max_cg_iterations"],
        "precond_rank": PARITY_SOLVER["precond_rank"],
        "precond_method": PARITY_SOLVER["precond_method"],
    }


def _memory_stats() -> tuple[GPUMemoryMonitor, float]:
    reset_torch_memory_stats()
    monitor = GPUMemoryMonitor(interval=0.1)
    monitor.start()
    tracemalloc.start()
    return monitor, time.perf_counter()


def _finish_memory_stats(monitor: GPUMemoryMonitor) -> dict[str, float]:
    monitor.stop()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del current
    stats = monitor.get_stats()
    stats.update(get_torch_memory_stats())
    stats["cpu_peak_mb"] = peak / (1024 * 1024)
    return stats


def _build_single_output_result(
    *,
    dataset,
    method: str,
    prediction_mode: str,
    training_time_s: float,
    prediction_time_s: float,
    iterations_run: int,
    memory_stats: dict[str, float],
    mean: np.ndarray,
    variance: np.ndarray,
    params: dict[str, float],
    final_nll: float,
) -> BenchmarkResult:
    std = np.sqrt(np.maximum(variance, 1e-10))
    normalized_params = normalize_single_output_benchmark_hparams(params)
    prediction_solver = _prediction_solver_metadata(prediction_mode)
    return BenchmarkResult(
        config={
            "benchmark": "single_output_route_parity",
            "suite_name": "mojogp_route_parity",
            "route_group": "single_output",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": prediction_mode,
            "comparison_class": "intra_mojogp_parity",
            "baseline_backend": "none",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": (
                "N.B. Intra-MojoGP route-parity row: this benchmark compares MojoGP routes "
                "against each other and is not a cross-framework fairness claim."
            ),
            "fairness_axes": {
                "comparator_scope": {
                    "status": "intra_mojogp",
                    "note": "No GPyTorch comparator is intended for this row.",
                },
                "sample_count_n": {
                    "status": "aligned",
                    "note": "All compared MojoGP routes run on the same dataset size.",
                },
                "optimizer": {
                    "status": "aligned",
                    "note": "The same MojoGP optimization policy is used across the compared routes.",
                },
                "solver_budget": {
                    "status": "aligned",
                    "note": "The same CG and Lanczos budgets are used across the compared routes.",
                },
                "preconditioner": {
                    "status": "aligned",
                    "note": "The same pivoted-Cholesky budget is used across the compared routes.",
                },
                "prediction_mode": {
                    "status": "aligned",
                    "note": "Each row reports a specific route/prediction-mode combination at matched settings.",
                },
                "telemetry": {
                    "status": "observed",
                    "note": "MojoGP telemetry is observed for the active route.",
                },
            },
            "n": int(dataset.X_train.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "n_test": int(dataset.X_test.shape[0]),
            "training_solver_config": {
                "framework": "mojogp",
                "mode": method,
                "cg_tolerance": PARITY_SOLVER["cg_tolerance"],
                "max_cg_iterations": PARITY_SOLVER["max_cg_iterations"],
                "num_trace_samples": PARITY_SOLVER["num_trace_samples"],
                "max_tridiag_iter": PARITY_SOLVER["max_lanczos_quadrature_iterations"],
                "precond_rank": PARITY_SOLVER["precond_rank"],
                "precond_method": PARITY_SOLVER["precond_method"],
            },
            "prediction_solver_config": {
                "framework": "mojogp",
                "mode": method,
                "prediction_mode": prediction_mode,
                "cg_tolerance": prediction_solver["cg_tolerance"],
                "max_cg_iterations": prediction_solver["max_cg_iterations"],
                "precond_rank": prediction_solver["precond_rank"],
                "precond_method": prediction_solver["precond_method"],
            },
            "preconditioner_config": {
                "family": "pivoted_cholesky",
                "framework": "mojogp",
                "rank": PARITY_SOLVER["precond_rank"],
                "method": PARITY_SOLVER["precond_method"],
                "rebuild_threshold": None,
            },
            "cg_telemetry_quality": {
                "training": "observed",
                "prediction": "observed"
                if prediction_mode == "exact"
                else "not_applicable",
                "configured_for_cg": True,
                "observed_cg_calls": True,
            },
            "prediction_timing_quality": "total_only_combined_call",
            "iter_timing_quality": "derived_total_div_iterations",
        },
        accuracy=AccuracyResult(
            rmse=rmse(dataset.f_test, mean),
            mae=mae(dataset.f_test, mean),
            r_squared=r_squared(dataset.f_test, mean),
            crps=crps_gaussian(dataset.f_test, mean, std),
            msll=mean_standardized_log_loss(
                dataset.f_test,
                mean,
                std,
                y_train_mean=float(np.mean(dataset.y_train)),
                y_train_std=float(np.std(dataset.y_train)),
            ),
            calibration_coverage=calibration_coverage(dataset.f_test, mean, std),
            calibration_error=calibration_error(dataset.f_test, mean, std),
            sharpness=sharpness(std),
            interval_width_95=interval_width(mean, std),
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            # This route parity harness measures a single combined prediction call.
            prediction_mean_time_s=0.0,
            prediction_variance_time_s=prediction_time_s,
            end_to_end_time_s=training_time_s + prediction_time_s,
            iterations_run=iterations_run,
            max_iterations=8,
            early_stopped=iterations_run < 8,
            ms_per_iteration=(training_time_s / max(iterations_run, 1)) * 1000.0,
            iter_timing_quality="derived_total_div_iterations",
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
            learned_lengthscale=float(normalized_params["lengthscale"]),
            learned_noise=float(params.get("noise", 0.1)),
            learned_outputscale=float(normalized_params["outputscale"]),
            final_nll=final_nll,
        ),
    )


def _build_multi_output_result(
    *,
    dataset,
    method: str,
    prediction_mode: str,
    training_time_s: float,
    prediction_time_s: float,
    iterations_run: int,
    memory_stats: dict[str, float],
    mean: np.ndarray,
    variance: np.ndarray,
    train_result,
) -> BenchmarkResult:
    std = np.sqrt(np.maximum(variance, 1e-10))
    rmse_vals = [rmse(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
    mae_vals = [mae(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
    r2_vals = [
        r_squared(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])
    ]
    crps_vals = [
        crps_gaussian(dataset.Y_test[:, t], mean[:, t], std[:, t])
        for t in range(mean.shape[1])
    ]
    msll_vals = [
        mean_standardized_log_loss(
            dataset.Y_test[:, t],
            mean[:, t],
            std[:, t],
            y_train_mean=float(np.mean(dataset.Y_train[:, t])),
            y_train_std=float(np.std(dataset.Y_train[:, t])),
        )
        for t in range(mean.shape[1])
    ]
    coverage = [
        calibration_coverage(dataset.Y_test[:, t], mean[:, t], std[:, t])
        for t in range(mean.shape[1])
    ]
    avg_coverage = {
        level: float(np.mean([entry[level] for entry in coverage]))
        for level in coverage[0]
    }
    return BenchmarkResult(
        config={
            "benchmark": "multi_output_route_parity",
            "suite_name": "mojogp_route_parity",
            "route_group": "multi_output",
            "framework": "mojogp",
            "model_type": "MultiOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": prediction_mode,
            "comparison_class": "intra_mojogp_parity",
            "baseline_backend": "none",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": (
                "N.B. Intra-MojoGP route-parity row: this benchmark compares MojoGP routes "
                "against each other and is not a cross-framework fairness claim."
            ),
            "fairness_axes": {
                "comparator_scope": {
                    "status": "intra_mojogp",
                    "note": "No GPyTorch comparator is intended for this row.",
                },
                "sample_count_n": {
                    "status": "aligned",
                    "note": "All compared MojoGP routes run on the same dataset size.",
                },
                "optimizer": {
                    "status": "aligned",
                    "note": "The same MojoGP optimization policy is used across the compared routes.",
                },
                "solver_budget": {
                    "status": "aligned",
                    "note": "The same CG and Lanczos budgets are used across the compared routes.",
                },
                "preconditioner": {
                    "status": "aligned",
                    "note": "The same pivoted-Cholesky budget is used across the compared routes.",
                },
                "prediction_mode": {
                    "status": "aligned",
                    "note": "Each row reports a specific route/prediction-mode combination at matched settings.",
                },
                "telemetry": {
                    "status": "observed",
                    "note": "MojoGP telemetry is observed for the active route.",
                },
            },
            "n": int(dataset.X_train.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "n_test": int(dataset.X_test.shape[0]),
            "num_tasks": int(dataset.Y_train.shape[1]),
            "training_solver_config": {
                "framework": "mojogp",
                "mode": method,
                "cg_tolerance": PARITY_SOLVER["cg_tolerance"],
                "max_cg_iterations": PARITY_SOLVER["max_cg_iterations"],
                "num_trace_samples": PARITY_SOLVER["num_trace_samples"],
                "max_tridiag_iter": PARITY_SOLVER["max_lanczos_quadrature_iterations"],
                "precond_rank": PARITY_SOLVER["precond_rank"],
                "precond_method": PARITY_SOLVER["precond_method"],
            },
            "prediction_solver_config": {
                "framework": "mojogp",
                "mode": method,
                "prediction_mode": prediction_mode,
                "cg_tolerance": PARITY_SOLVER["cg_tolerance"],
                "max_cg_iterations": PARITY_SOLVER["max_cg_iterations"],
                "precond_rank": PARITY_SOLVER["precond_rank"],
                "precond_method": PARITY_SOLVER["precond_method"],
            },
            "preconditioner_config": {
                "family": "pivoted_cholesky",
                "framework": "mojogp",
                "rank": PARITY_SOLVER["precond_rank"],
                "method": PARITY_SOLVER["precond_method"],
                "rebuild_threshold": None,
            },
            "cg_telemetry_quality": {
                "training": "observed",
                "prediction": "observed"
                if prediction_mode == "exact"
                else "not_applicable",
                "configured_for_cg": True,
                "observed_cg_calls": True,
            },
            "prediction_timing_quality": "total_only_combined_call",
            "iter_timing_quality": "derived_total_div_iterations",
        },
        accuracy=AccuracyResult(
            rmse=float(np.mean(rmse_vals)),
            mae=float(np.mean(mae_vals)),
            r_squared=float(np.mean(r2_vals)),
            crps=float(np.mean(crps_vals)),
            msll=float(np.mean(msll_vals)),
            calibration_coverage=avg_coverage,
            calibration_error=float(
                np.mean(
                    [
                        calibration_error(dataset.Y_test[:, t], mean[:, t], std[:, t])
                        for t in range(mean.shape[1])
                    ]
                )
            ),
            sharpness=float(
                np.mean([sharpness(std[:, t]) for t in range(mean.shape[1])])
            ),
            interval_width_95=float(
                np.mean(
                    [
                        interval_width(mean[:, t], std[:, t])
                        for t in range(mean.shape[1])
                    ]
                )
            ),
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            # This route parity harness measures a single combined prediction call.
            prediction_mean_time_s=0.0,
            prediction_variance_time_s=prediction_time_s,
            end_to_end_time_s=training_time_s + prediction_time_s,
            iterations_run=iterations_run,
            max_iterations=6,
            early_stopped=iterations_run < 6,
            ms_per_iteration=(training_time_s / max(iterations_run, 1)) * 1000.0,
            iter_timing_quality="derived_total_div_iterations",
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
            learned_lengthscale=float(np.mean(np.asarray(train_result.params))),
            learned_noise=float(np.mean(train_result.noise_per_task)),
            learned_outputscale=float(np.mean(train_result.effective_scales)),
            final_nll=float(train_result.final_nll),
        ),
    )


@pytest.mark.minimal
@pytest.mark.single_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
@pytest.mark.parametrize("prediction_mode", ["exact", "love"])
def test_single_output_route_parity(results_dir, method: str, prediction_mode: str):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_mojogp_route_parity_case",
        payload={"case": "single_output", "method": method, "prediction_mode": prediction_mode},
        suite_name="mojogp_route_parity_harness",
        benchmark_name="single_output_route_parity",
        framework="mojogp",
        case_id=f"mojogp.route_parity.single_output.{method}.{prediction_mode}",
        benchmark_group_id=f"mojogp.route_parity.single_output.{method}.{prediction_mode}",
        config={
            "benchmark": "single_output_route_parity",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "prediction_mode": prediction_mode,
        },
        results_dir=results_dir,
    )
    print_result(
        result, title=f"Single-output route parity: {method} {prediction_mode}"
    )
    assert_gpu_was_used(result)
    assert np.isfinite(result.accuracy.rmse)


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
@pytest.mark.parametrize("prediction_mode", ["exact", "love"])
def test_multi_output_route_parity(results_dir, method: str, prediction_mode: str):
    assert_gpu_available()
    result = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_mojogp_route_parity_case",
        payload={"case": "multi_output", "method": method, "prediction_mode": prediction_mode},
        suite_name="mojogp_route_parity_harness",
        benchmark_name="multi_output_route_parity",
        framework="mojogp",
        case_id=f"mojogp.route_parity.multi_output.{method}.{prediction_mode}",
        benchmark_group_id=f"mojogp.route_parity.multi_output.{method}.{prediction_mode}",
        config={
            "benchmark": "multi_output_route_parity",
            "framework": "mojogp",
            "model_type": "MultiOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "prediction_mode": prediction_mode,
            "num_tasks": 3,
        },
        results_dir=results_dir,
    )
    print_result(result, title=f"Multi-output route parity: {method} {prediction_mode}")
    assert_gpu_was_used(result)
    assert np.isfinite(result.accuracy.rmse)


def test_mojogp_route_parity_summary(results_dir):
    # The summary test runs last within the file and packages the emitted parity rows.
    from tests.shared.benchmarking.report import load_all_results

    results = [
        r
        for r in load_all_results(results_dir, "*route_parity*.json")
        if r.config.get("benchmark") in {"single_output_route_parity", "multi_output_route_parity"}
    ]
    if results:
        save_summary_report(results, results_dir, "mojogp_route_parity")
