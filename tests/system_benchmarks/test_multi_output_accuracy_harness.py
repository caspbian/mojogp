"""Harness benchmark for multi-output GP accuracy vs synthetic truth.

This file is the active JIT-era synthetic accuracy surface for ``MultiOutputGP``.
It focuses on the questions that matter for practitioner confidence:

1. do joint multi-output models learn useful task structure?
2. do they match or beat independent-task baselines when correlation exists?
3. do they avoid inventing gains when tasks are effectively independent?
4. do both materialized and matrix-free routes remain numerically usable?
"""

from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np
import pytest

from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess
from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.benchmarks.mojogp.multi_output_independent_baseline import (
    run_independent_exactgp_baseline,
)
from mojogp import (
    MultiOutputGP,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    RQ,
    Linear,
    Polynomial,
)

from .conftest import (
    assert_gpu_available,
    assert_gpu_was_used,
    get_vram_info,
    requires_cuda,
    scale_n_for_vram,
)
from tests.shared.benchmarking.data_generators import MultiOutputDataset, generate_multi_output_data
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
from tests.shared.benchmarking.report import print_result
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)
# Format: (kernel, n_train, n_test, d, num_tasks, task_correlation, method)
MINIMAL_CONFIGS = [
    ("rbf", 500, BENCHMARK_PREDICTION_N_TEST, 5, 2, "medium", "materialized"),
    ("matern52", 500, BENCHMARK_PREDICTION_N_TEST, 5, 3, "medium", "materialized"),
    ("rbf", 2000, BENCHMARK_PREDICTION_N_TEST, 5, 2, "medium", "matrix_free"),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("rbf", 750, BENCHMARK_PREDICTION_N_TEST, 5, 2, "high", "materialized"),
    ("rbf", 750, BENCHMARK_PREDICTION_N_TEST, 5, 2, "independent", "materialized"),
    ("rbf", 3000, BENCHMARK_PREDICTION_N_TEST, 5, 4, "medium", "matrix_free"),
    ("matern32", 750, BENCHMARK_PREDICTION_N_TEST, 5, 3, "medium", "materialized"),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    ("rbf", 1000, BENCHMARK_PREDICTION_N_TEST, 8, 4, "medium", "materialized"),
    ("rbf", 4000, BENCHMARK_PREDICTION_N_TEST, 8, 4, "medium", "matrix_free"),
    ("matern12", 1000, BENCHMARK_PREDICTION_N_TEST, 5, 3, "medium", "materialized"),
    ("matern52", 4000, BENCHMARK_PREDICTION_N_TEST, 5, 4, "high", "matrix_free"),
    ("rbf", 1000, BENCHMARK_PREDICTION_N_TEST, 5, 8, "low", "materialized"),
]

SYNTHETIC_RMSE_MAX = 1.0
SYNTHETIC_R2_MIN = 0.15
JOINT_VS_INDEPENDENT_RATIO_MAX = 1.05
INDEPENDENT_JOINT_GAIN_ABS_MAX = 0.15
CORRELATED_TASK_COVARIANCE_CORR_MIN = 0.8


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _multi_output_accuracy_metadata(
    *,
    dataset: MultiOutputDataset,
    kernel: str,
    method: str,
    num_tasks: int,
    max_iterations: int,
    learning_rate: float,
) -> dict[str, object]:
    return {
        "benchmark": "multi_output_accuracy",
        "route_group": "multi_output",
        "framework": "mojogp",
        "model_type": "MultiOutputGP",
        "kernel": kernel,
        "training_method": method,
        "method": method,
        "prediction_mode": "love",
        "comparison_class": "mojogp_only",
        "baseline_backend": "none",
        "keops_supported": False,
        "keops_used": False,
        "fairness_note": (
            "N.B. MojoGP-only synthetic row: this benchmark measures joint multi-output "
            "accuracy against synthetic ground truth and an in-repo independent ExactGP "
            "baseline, not against a cross-framework comparator."
        ),
        "fairness_axes": {
            "comparator_scope": _fairness_axis(
                "mojogp_only",
                "The published comparator is the in-repo independent ExactGP baseline, not GPyTorch.",
            ),
            "sample_count_n": _fairness_axis(
                "aligned",
                "The joint and independent MojoGP baselines run on the same dataset size.",
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "Both the joint and independent MojoGP baselines use the same optimizer family and budget.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "The benchmark records one MojoGP training route with fixed CG/Lanczos settings.",
            ),
            "preconditioner": _fairness_axis(
                "aligned",
                "The active route uses its configured pivoted-Cholesky preconditioner budget.",
            ),
            "prediction_mode": _fairness_axis(
                "aligned",
                "This suite reports LOVE-based predictive metrics consistently across rows.",
            ),
            "telemetry": _fairness_axis(
                "observed",
                "MojoGP telemetry is observed for the active route.",
            ),
        },
        "n": int(dataset.X_train.shape[0]),
        "n_test": int(dataset.X_test.shape[0]),
        "d": int(dataset.X_train.shape[1]),
        "num_tasks": num_tasks,
        "task_correlation": dataset.true_params["task_correlation"],
        "training_solver_config": {
            "framework": "mojogp",
            "mode": method,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
        },
    }


def _safe_average(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _kernel_name_for_exactgp(kernel: str) -> str:
    if kernel not in {
        "rbf",
        "matern12",
        "matern32",
        "matern52",
        "periodic",
        "rq",
        "linear",
        "polynomial",
    }:
        raise ValueError(f"Unsupported ExactGP kernel '{kernel}'")
    return kernel


def _make_exactgp_kernel(kernel: str):
    mapping = {
        "rbf": RBF,
        "matern12": Matern12,
        "matern32": Matern32,
        "matern52": Matern52,
        "periodic": Periodic,
        "rq": RQ,
        "linear": Linear,
        "polynomial": Polynomial,
    }
    return mapping[_kernel_name_for_exactgp(kernel)]()


def _run_multi_output_accuracy_subprocess(
    *,
    kernel: str,
    n_train: int,
    n_test: int,
    d: int,
    num_tasks: int,
    correlation: str,
    method: str,
    results_dir,
    seed: int = 42,
) -> BenchmarkResult:
    return run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_multi_output_accuracy_case",
        payload={
            "kernel": kernel,
            "n_train": n_train,
            "n_test": n_test,
            "d": d,
            "num_tasks": num_tasks,
            "correlation": correlation,
            "method": method,
            "seed": seed,
        },
        suite_name="multi_output_accuracy_harness",
        benchmark_name="multi_output_accuracy",
        framework="mojogp",
        case_id=f"multi_output.accuracy.{kernel}.{method}.t{num_tasks}.corr_{correlation}.n{n_train}.d{d}",
        benchmark_group_id=f"multi_output.accuracy.{kernel}.{method}.t{num_tasks}.corr_{correlation}",
        config={
            "benchmark": "multi_output_accuracy",
            "framework": "mojogp",
            "model_type": "MultiOutputGP",
            "kernel": kernel,
            "training_method": method,
            "prediction_mode": "love",
            "n": n_train,
            "d": d,
            "num_tasks": num_tasks,
            "task_correlation": correlation,
        },
        results_dir=results_dir,
    )


class TestMultiOutputAccuracy:
    """Synthetic quantitative validation for MultiOutputGP."""

    def _fit_independent_exactgps(
        self,
        dataset: MultiOutputDataset,
        kernel: str,
        method: str,
        max_iterations: int,
        learning_rate: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        means = []
        variances = []
        fit_times = []

        for task_idx in range(dataset.Y_train.shape[1]):
            mean, variance, timing = run_independent_exactgp_baseline(
                x_train=dataset.X_train,
                y_train=dataset.Y_train[:, task_idx],
                x_test=dataset.X_test,
                kernel=kernel,
                method=method,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                task_idx=task_idx,
            )
            means.append(mean)
            variances.append(variance)
            fit_times.append(float(timing["training_time_s"]))

        return (
            np.stack(means, axis=1),
            np.stack(variances, axis=1),
            {"training_time_s": float(np.sum(fit_times))},
        )

    def _compute_accuracy(
        self,
        dataset: MultiOutputDataset,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> AccuracyResult:
        std = np.sqrt(np.maximum(variance, 1e-10))

        task_rmses = [
            rmse(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])
        ]
        task_maes = [
            mae(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])
        ]
        task_r2s = [
            r_squared(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])
        ]
        task_crps = [
            crps_gaussian(dataset.Y_test[:, t], mean[:, t], std[:, t])
            for t in range(mean.shape[1])
        ]
        task_msll = [
            mean_standardized_log_loss(
                dataset.Y_test[:, t],
                mean[:, t],
                std[:, t],
                y_train_mean=float(np.mean(dataset.Y_train[:, t])),
                y_train_std=float(np.std(dataset.Y_train[:, t])),
            )
            for t in range(mean.shape[1])
        ]
        coverage_levels = [0.5, 0.9, 0.95, 0.99]
        per_task_coverage = [
            calibration_coverage(
                dataset.Y_test[:, t], mean[:, t], std[:, t], levels=coverage_levels
            )
            for t in range(mean.shape[1])
        ]
        avg_coverage = {
            level: float(np.mean([cov[level] for cov in per_task_coverage]))
            for level in coverage_levels
        }
        task_calibration_errors = [
            calibration_error(dataset.Y_test[:, t], mean[:, t], std[:, t])
            for t in range(mean.shape[1])
        ]
        task_sharpness = [sharpness(std[:, t]) for t in range(mean.shape[1])]
        task_interval_width = [
            interval_width(mean[:, t], std[:, t]) for t in range(mean.shape[1])
        ]

        return AccuracyResult(
            rmse=_safe_average(task_rmses),
            mae=_safe_average(task_maes),
            r_squared=_safe_average(task_r2s),
            crps=_safe_average(task_crps),
            msll=_safe_average(task_msll),
            calibration_coverage=avg_coverage,
            calibration_error=_safe_average(task_calibration_errors),
            sharpness=_safe_average(task_sharpness),
            interval_width_95=_safe_average(task_interval_width),
        )

    def _train_multi_output_benchmark(
        self,
        dataset: MultiOutputDataset,
        kernel: str,
        method: str,
        num_tasks: int,
        learning_rate: float,
        max_iterations: int,
    ) -> tuple[BenchmarkResult, dict[str, float]]:
        reset_torch_memory_stats()
        gpu_monitor = GPUMemoryMonitor(interval=0.1)
        gpu_monitor.start()
        tracemalloc.start()

        gp = MultiOutputGP(
            kernel=kernel,
            num_probes=5 if method == "materialized" else 3,
            max_cg_iterations=50 if method == "materialized" else 30,
            max_tridiag_iterations=20 if method == "materialized" else 10,
            use_preconditioner=False,
        )

        fit_start = time.perf_counter()
        result = gp.fit(
            dataset.X_train,
            dataset.Y_train,
            method=method,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            verbose=False,
        )
        training_time_s = time.perf_counter() - fit_start

        mean_start = time.perf_counter()
        pred_mean = gp.predict(dataset.X_test)
        prediction_mean_time_s = time.perf_counter() - mean_start

        var_start = time.perf_counter()
        pred_var = gp.predict(dataset.X_test, return_var=True)
        prediction_variance_time_s = time.perf_counter() - var_start

        gpu_monitor.stop()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del current

        memory_stats = gpu_monitor.get_stats()
        memory_stats.update(get_torch_memory_stats())
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)

        mean = np.asarray(pred_mean.mean, dtype=np.float32)
        variance = np.asarray(pred_var[1], dtype=np.float32)
        accuracy = self._compute_accuracy(dataset, mean, variance)

        speed = SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=prediction_mean_time_s,
            prediction_variance_time_s=prediction_variance_time_s,
            end_to_end_time_s=(
                training_time_s + prediction_mean_time_s + prediction_variance_time_s
            ),
            iterations_run=int(result.iterations),
            max_iterations=max_iterations,
            early_stopped=int(result.iterations) < max_iterations,
            ms_per_iteration=(training_time_s / max(int(result.iterations), 1))
            * 1000.0,
        )

        memory = MemoryResult(
            gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
            gpu_min_mb=memory_stats.get("min_mb", 0.0),
            gpu_max_mb=memory_stats.get("max_mb", 0.0),
            gpu_var_mb=memory_stats.get("var_mb", 0.0),
            torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
            torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
            cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
            measurement_method=memory_stats.get("method", "none"),
            num_samples=int(memory_stats.get("samples", 0)),
        )

        task_cov = np.asarray(gp.task_covariance, dtype=np.float32)
        true_cov = np.asarray(dataset.true_params["B"], dtype=np.float32)
        cov_flat = task_cov.reshape(-1)
        true_flat = true_cov.reshape(-1)
        cov_corr = 0.0
        if np.std(cov_flat) > 1e-8 and np.std(true_flat) > 1e-8:
            cov_corr = float(np.corrcoef(cov_flat, true_flat)[0, 1])

        learned_lengthscales = getattr(result, "lengthscales", None)
        if learned_lengthscales is None:
            learned_lengthscales = np.array([getattr(result, "lengthscale", 1.0)])
        learned_outputscale = getattr(result, "outputscale", None)
        if learned_outputscale is None:
            learned_outputscale = float(np.mean(np.asarray(result.effective_scales)))

        hyperparameters = HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(learned_lengthscales))),
            learned_noise=float(np.mean(np.asarray(result.noise_per_task))),
            learned_outputscale=float(learned_outputscale),
            final_nll=float(result.final_nll),
        )

        benchmark = BenchmarkResult(
            config={
                **_multi_output_accuracy_metadata(
                    dataset=dataset,
                    kernel=kernel,
                    method=method,
                    num_tasks=num_tasks,
                    max_iterations=max_iterations,
                    learning_rate=learning_rate,
                ),
                "task_covariance_correlation": cov_corr,
                "task_covariance_fro_error": float(np.linalg.norm(task_cov - true_cov)),
            },
            accuracy=accuracy,
            speed=speed,
            memory=memory,
            hyperparameters=hyperparameters,
        )
        extra = {
            "mean": mean,
            "variance": variance,
            "gp": gp,
            "result": result,
            "covariance_correlation": cov_corr,
        }
        return benchmark, extra

    def _run_multi_output_test(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        correlation: str,
        method: str,
        seed: int = 42,
    ) -> BenchmarkResult:
        effective_n = scale_n_for_vram(n_train, method)
        dataset = generate_multi_output_data(
            n_train=effective_n,
            n_test=n_test,
            d=d,
            num_tasks=num_tasks,
            kernel_type=kernel,
            task_correlation=correlation,
            seed=seed,
        )
        max_iterations = 60 if method == "materialized" else 40
        learning_rate = 0.03 if method == "materialized" else 0.02

        benchmark, extra = self._train_multi_output_benchmark(
            dataset=dataset,
            kernel=kernel,
            method=method,
            num_tasks=num_tasks,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
        )

        independent_mean, independent_var, independent_speed = (
            self._fit_independent_exactgps(
                dataset,
                kernel=kernel,
                method=method,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
            )
        )
        independent_rmse = float(
            np.mean(
                [
                    rmse(dataset.F_test[:, t], independent_mean[:, t])
                    for t in range(num_tasks)
                ]
            )
        )
        benchmark.config.update(
            {
                "requested_n": int(n_train),
                "vram_scaled": effective_n < n_train,
                "independent_rmse": independent_rmse,
                "joint_gain_vs_independent": float(
                    independent_rmse - benchmark.accuracy.rmse
                ),
                "joint_vs_independent_rmse_ratio": float(
                    benchmark.accuracy.rmse / max(independent_rmse, 1e-6)
                ),
                "independent_training_time_s": independent_speed["training_time_s"],
                "mean_alignment_vs_independent": float(
                    np.mean(np.abs(extra["mean"] - independent_mean))
                ),
                "joint_vs_independent_policy": "hard_asserted_ratio",
                "joint_vs_independent_rmse_ratio_max": JOINT_VS_INDEPENDENT_RATIO_MAX,
            }
        )
        if correlation == "independent":
            benchmark.config["independent_joint_gain_abs_max"] = INDEPENDENT_JOINT_GAIN_ABS_MAX

        benchmark.config.update(
            {
                "synthetic_rmse_cap_status": "hard_asserted",
                "synthetic_rmse_max": SYNTHETIC_RMSE_MAX,
                "synthetic_r2_floor_status": "hard_asserted",
                "synthetic_r2_min": SYNTHETIC_R2_MIN,
                "task_covariance_correlation_status": "hard_asserted_for_medium_high_correlated_tasks",
                "task_covariance_correlation_min": CORRELATED_TASK_COVARIANCE_CORR_MIN,
            }
        )

        return benchmark

    def _assert_accuracy_contract(self, result: BenchmarkResult) -> None:
        assert np.isfinite(result.accuracy.rmse)
        assert np.isfinite(result.accuracy.r_squared)
        assert result.accuracy.rmse <= result.config["synthetic_rmse_max"], (
            f"Multi-output synthetic RMSE exceeded contract: {result.accuracy.rmse:.4f}"
        )
        assert result.accuracy.r_squared >= result.config["synthetic_r2_min"], (
            f"Multi-output synthetic R2 fell below contract: {result.accuracy.r_squared:.4f}"
        )
        ratio = float(result.config["joint_vs_independent_rmse_ratio"])
        assert ratio <= result.config["joint_vs_independent_rmse_ratio_max"], (
            f"Joint model regressed too far vs independent baseline: ratio={ratio:.4f}"
        )
        if result.config["task_correlation"] == "independent":
            gain = float(result.config["joint_gain_vs_independent"])
            assert abs(gain) <= result.config["independent_joint_gain_abs_max"], (
                f"Independent-task joint gain was unexpectedly large: gain={gain:.4f}"
            )
        elif result.config["task_correlation"] in {"medium", "high"}:
            cov_corr = float(result.config["task_covariance_correlation"])
            assert cov_corr >= result.config["task_covariance_correlation_min"], (
                f"Learned task covariance did not track synthetic task structure: corr={cov_corr:.4f}"
            )

    def _report_result(self, result: BenchmarkResult, results_dir) -> None:
        print_result(result)

    @pytest.mark.minimal
    @pytest.mark.multi_output
    @pytest.mark.accuracy
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,correlation,method", MINIMAL_CONFIGS
    )
    @requires_cuda
    def test_multi_output_accuracy_benchmark_core_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        correlation: str,
        method: str,
        results_dir,
        n_override,
    ):
        assert_gpu_available()
        if n_override is not None:
            n_train = n_override
        result = _run_multi_output_accuracy_subprocess(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            num_tasks=num_tasks,
            correlation=correlation,
            method=method,
            results_dir=results_dir,
        )
        self._report_result(result, results_dir)
        assert_gpu_was_used(result)
        self._assert_accuracy_contract(result)

    @pytest.mark.moderate
    @pytest.mark.multi_output
    @pytest.mark.accuracy
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,correlation,method", MODERATE_CONFIGS
    )
    @requires_cuda
    def test_multi_output_accuracy_benchmark_extended_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        correlation: str,
        method: str,
        results_dir,
        n_override,
    ):
        assert_gpu_available()
        if n_override is not None:
            n_train = n_override
        result = _run_multi_output_accuracy_subprocess(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            num_tasks=num_tasks,
            correlation=correlation,
            method=method,
            results_dir=results_dir,
        )
        self._report_result(result, results_dir)
        assert_gpu_was_used(result)
        self._assert_accuracy_contract(result)

    @pytest.mark.full
    @pytest.mark.multi_output
    @pytest.mark.accuracy
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,correlation,method", FULL_CONFIGS
    )
    @requires_cuda
    def test_multi_output_accuracy_benchmark_broad_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        correlation: str,
        method: str,
        results_dir,
        n_override,
    ):
        assert_gpu_available()
        if n_override is not None:
            n_train = n_override
        result = _run_multi_output_accuracy_subprocess(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            num_tasks=num_tasks,
            correlation=correlation,
            method=method,
            results_dir=results_dir,
        )
        self._report_result(result, results_dir)
        assert_gpu_was_used(result)
        self._assert_accuracy_contract(result)

    @pytest.mark.minimal
    @pytest.mark.multi_output
    @pytest.mark.accuracy
    @requires_cuda
    def test_joint_model_matches_independent_when_correlation_is_high(self, results_dir):
        assert_gpu_available()
        result = _run_multi_output_accuracy_subprocess(
            kernel="rbf",
            n_train=750,
            n_test=BENCHMARK_PREDICTION_N_TEST,
            d=5,
            num_tasks=3,
            correlation="high",
            method="materialized",
            results_dir=results_dir,
            seed=123,
        )
        self._report_result(result, results_dir)
        self._assert_accuracy_contract(result)
        assert result.config["joint_vs_independent_rmse_ratio"] <= 1.02, (
            "High-correlation synthetic data showed too large a joint-model regression vs independent baselines"
        )

    @pytest.mark.minimal
    @pytest.mark.multi_output
    @pytest.mark.accuracy
    @requires_cuda
    def test_materialized_and_matrix_free_are_in_same_ballpark(self, results_dir):
        assert_gpu_available()
        seed = 321
        dataset = generate_multi_output_data(
            n_train=scale_n_for_vram(800, "materialized"),
            n_test=BENCHMARK_PREDICTION_N_TEST,
            d=5,
            num_tasks=3,
            kernel_type="rbf",
            task_correlation="medium",
            seed=seed,
        )
        mat_result = _run_multi_output_accuracy_subprocess(
            kernel="rbf",
            n_train=int(dataset.X_train.shape[0]),
            n_test=int(dataset.X_test.shape[0]),
            d=5,
            num_tasks=3,
            correlation="medium",
            method="materialized",
            results_dir=results_dir,
            seed=seed,
        )
        mf_result = _run_multi_output_accuracy_subprocess(
            kernel="rbf",
            n_train=int(dataset.X_train.shape[0]),
            n_test=int(dataset.X_test.shape[0]),
            d=5,
            num_tasks=3,
            correlation="medium",
            method="matrix_free",
            results_dir=results_dir,
            seed=seed,
        )
        self._report_result(mat_result, results_dir)
        self._report_result(mf_result, results_dir)
        assert abs(mat_result.accuracy.rmse - mf_result.accuracy.rmse) < 0.35, (
            f"Materialized and matrix-free multi-output routes diverged too far: "
            f"mat={mat_result.accuracy.rmse:.4f}, mf={mf_result.accuracy.rmse:.4f}"
        )
