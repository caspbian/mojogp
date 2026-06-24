"""Harness benchmark for MultiOutputGP per-task-noise recovery.

This benchmark uses the public ``MultiOutputGP`` API and result plumbing. It
validates that MojoGP can:

1. fit multi-output models with distinct per-task observation noise
2. recover the relative ordering of task noise levels
3. remain numerically usable on both materialized and matrix-free routes
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest
import torch

from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess
from tests.benchmarks.prediction_workload import (
    BENCHMARK_PREDICTION_N_TEST,
    clear_prediction_x_test_failure_memory,
    prediction_x_test_repeat_count,
    prediction_x_test_scaling_entry,
    prediction_x_test_scaling_failure_entry,
    prediction_x_test_should_record_failure,
    prediction_x_test_target_specs,
)
from mojogp import MultiOutputGP

from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda, scale_n_for_vram
from tests.shared.benchmarking.data_generators import (
    MultiOutputDataset,
    generate_multi_output_per_task_noise_data,
    generate_multi_output_structured_per_task_noise_data,
)
from tests.shared.benchmarking.gpu_memory import (
    GPUMemoryMonitor,
    get_torch_memory_stats,
    measure_gpu_phase,
    reset_torch_memory_stats,
)
from tests.shared.benchmarking.metrics import (
    calibration_coverage,
    calibration_error,
    crps_gaussian,
    mae,
    mean_standardized_log_loss,
    param_relative_error,
    rmse,
    r_squared,
    sharpness,
    interval_width,
)
from tests.shared.benchmarking.report import print_result
from tests.shared.benchmarking.result_types import AccuracyResult, BenchmarkResult, HyperparameterResult, MemoryResult, SpeedResult


MINIMAL_CONFIGS = [
    ("rbf", 1000, BENCHMARK_PREDICTION_N_TEST, 5, 3, "medium", "mild", "zero", "materialized"),
    ("rbf", 2500, BENCHMARK_PREDICTION_N_TEST, 5, 3, "medium", "strong", "zero", "matrix_free"),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("matern52", 1200, BENCHMARK_PREDICTION_N_TEST, 5, 3, "high", "strong", "offset", "materialized"),
    ("rbf", 3000, BENCHMARK_PREDICTION_N_TEST, 5, 4, "medium", "strong", "offset", "matrix_free"),
]

PER_TASK_NOISE_RMSE_MAX = 2.0
PER_TASK_NOISE_R2_MIN = 0.3
NOISE_RANK_CORRELATION_MIN = 0.5


def _safe_average(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _timing_payload(iter_times_ms) -> dict[str, object]:
    if iter_times_ms is None:
        return {
            "iter_times_ms": None,
            "iter_time_min_ms": None,
            "iter_time_q25_ms": None,
            "iter_time_mean_ms": None,
            "iter_time_median_ms": None,
            "iter_time_q75_ms": None,
            "iter_time_max_ms": None,
            "iter_time_p5_ms": None,
            "iter_time_p95_ms": None,
        }
    values = np.asarray(iter_times_ms, dtype=np.float64)
    if values.size == 0:
        return {
            "iter_times_ms": [],
            "iter_time_min_ms": 0.0,
            "iter_time_q25_ms": 0.0,
            "iter_time_mean_ms": 0.0,
            "iter_time_median_ms": 0.0,
            "iter_time_q75_ms": 0.0,
            "iter_time_max_ms": 0.0,
            "iter_time_p5_ms": 0.0,
            "iter_time_p95_ms": 0.0,
        }
    return {
        "iter_times_ms": values.tolist(),
        "iter_time_min_ms": float(np.min(values)),
        "iter_time_q25_ms": float(np.percentile(values, 25)),
        "iter_time_mean_ms": float(np.mean(values)),
        "iter_time_median_ms": float(np.median(values)),
        "iter_time_q75_ms": float(np.percentile(values, 75)),
        "iter_time_max_ms": float(np.max(values)),
        "iter_time_p5_ms": float(np.percentile(values, 5)),
        "iter_time_p95_ms": float(np.percentile(values, 95)),
    }


def _merge_phase_memory(
    memory_stats: dict[str, float],
    fit_memory_stats: dict[str, float],
    pred_memory_stats: dict[str, float],
) -> dict[str, float]:
    merged = dict(memory_stats)
    merged.update(get_torch_memory_stats())
    fit_peak = float(fit_memory_stats.get("phase_peak_gpu_mb", 0.0))
    pred_peak = float(pred_memory_stats.get("phase_peak_gpu_mb", 0.0))
    merged["max_mb"] = max(float(merged.get("max_mb", 0.0)), fit_peak, pred_peak)
    merged["mean_mb"] = max(float(merged.get("mean_mb", 0.0)), float(merged["max_mb"]))
    merged["torch_peak_mb"] = max(
        float(merged.get("torch_peak_mb", 0.0)),
        float(fit_memory_stats.get("torch_peak_mb", 0.0)),
        float(pred_memory_stats.get("torch_peak_mb", 0.0)),
    )
    merged["torch_current_mb"] = float(
        pred_memory_stats.get("torch_current_mb", merged.get("torch_current_mb", 0.0))
    )
    merged["training_peak_gpu_mb"] = fit_peak
    merged["training_delta_gpu_mb"] = float(
        fit_memory_stats.get("phase_delta_gpu_mb", 0.0)
    )
    merged["prediction_peak_gpu_mb"] = pred_peak
    merged["prediction_delta_gpu_mb"] = float(
        pred_memory_stats.get("phase_delta_gpu_mb", 0.0)
    )
    merged["love_prediction_peak_gpu_mb"] = pred_peak
    merged["love_prediction_delta_gpu_mb"] = float(
        pred_memory_stats.get("phase_delta_gpu_mb", 0.0)
    )
    return merged


def _time_prediction_call(callable_obj):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_obj()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, float(time.perf_counter() - start)


def _per_task_prediction_inputs(
    *,
    n_test: int,
    d: int,
    dataset: MultiOutputDataset,
    seed: int,
) -> np.ndarray:
    if int(dataset.X_test.shape[0]) == int(n_test):
        return np.asarray(dataset.X_test, dtype=np.float32)
    rng = np.random.default_rng(seed + 1_000_003 + int(n_test))
    return rng.uniform(-3.0, 3.0, size=(int(n_test), d)).astype(np.float32)


def _measure_per_task_prediction_x_test_scaling(
    *,
    gp: MultiOutputGP,
    dataset: MultiOutputDataset,
    canonical_first_time_s: float,
    canonical_memory_stats: dict[str, float],
    seed: int,
    extra_config: dict[str, object] | None,
) -> list[dict[str, object]]:
    extra_config = dict(extra_config or {})
    rows: list[dict[str, object]] = []
    for spec in prediction_x_test_target_specs(
        variety=str(extra_config.get("benchmark_variety", "standard")),
        tier=str(extra_config.get("benchmark_route_tier", "xsmall")),
        framework="mojogp",
    ):
        n_test = int(spec["n_test"])
        try:
            if n_test == int(dataset.X_test.shape[0]):
                X_test = np.asarray(dataset.X_test, dtype=np.float32)
                first_time_s = canonical_first_time_s
                first_memory_stats = canonical_memory_stats
            else:
                X_test = _per_task_prediction_inputs(
                    n_test=n_test,
                    d=int(dataset.X_train.shape[1]),
                    dataset=dataset,
                    seed=seed,
                )

                def _predict_size():
                    return gp.predict(X_test, return_var=True)

                first_start = time.perf_counter()
                _, first_memory_stats = measure_gpu_phase(_predict_size, interval=0.02)
                first_time_s = float(time.perf_counter() - first_start)

            def _predict_repeat():
                return gp.predict(X_test, return_var=True)

            repeat_times: list[float] = []
            for _ in range(prediction_x_test_repeat_count()):
                _, repeat_time_s = _time_prediction_call(_predict_repeat)
                repeat_times.append(repeat_time_s)
            rows.append(
                prediction_x_test_scaling_entry(
                    spec=spec,
                    timing_quality="warm_repeated_prediction",
                    cache_used=False,
                    first_apply_time_s=first_time_s,
                    repeat_times_s=repeat_times,
                    prediction_peak_gpu_mb=first_memory_stats.get("phase_peak_gpu_mb"),
                    prediction_delta_gpu_mb=first_memory_stats.get("phase_delta_gpu_mb"),
                    mean_time_s=None,
                    variance_time_s=first_time_s,
                )
            )
        except Exception as exc:
            if not prediction_x_test_should_record_failure(spec):
                raise
            clear_prediction_x_test_failure_memory()
            rows.append(
                prediction_x_test_scaling_failure_entry(
                    spec=spec,
                    error=exc,
                    failure_stage="prediction_apply",
                    timing_quality="failed_prediction_envelope",
                    cache_used=False,
                )
            )
    return rows


def _noise_profile(num_tasks: int, profile: str) -> np.ndarray:
    if profile == "mild":
        base = np.array([0.03, 0.06, 0.10, 0.14, 0.18], dtype=np.float32)
    elif profile == "strong":
        base = np.array([0.01, 0.05, 0.15, 0.30, 0.50], dtype=np.float32)
    else:
        raise ValueError(f"Unknown noise profile '{profile}'")
    return base[:num_tasks].copy()


def _mean_profile(num_tasks: int, profile: str) -> np.ndarray:
    if profile == "zero":
        return np.zeros(num_tasks, dtype=np.float32)
    if profile == "offset":
        base = np.array([-1.25, -0.25, 0.75, 1.5, 2.0], dtype=np.float32)
        return base[:num_tasks].copy()
    raise ValueError(f"Unknown mean profile '{profile}'")


def _noise_rank_correlation(learned: np.ndarray, truth: np.ndarray) -> float:
    learned_rank = np.argsort(np.argsort(learned)).astype(np.float32)
    truth_rank = np.argsort(np.argsort(truth)).astype(np.float32)
    if np.std(learned_rank) < 1e-8 or np.std(truth_rank) < 1e-8:
        return 1.0
    return float(np.corrcoef(learned_rank, truth_rank)[0, 1])


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _metadata(
    *,
    dataset: MultiOutputDataset,
    kernel: str,
    method: str,
    max_iterations: int,
    learning_rate: float,
    num_probes: int,
    max_cg_iter: int,
    max_tridiag_iter: int,
    precond_rank: int,
) -> dict[str, object]:
    return {
        "benchmark": "multi_output_per_task_noise",
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
            "N.B. MojoGP-only per-task-noise row: this benchmark measures synthetic truth recovery "
            "for per-task means and observation noise. It is not a cross-framework fairness claim."
        ),
        "fairness_axes": {
            "comparator_scope": _fairness_axis(
                "mojogp_only",
                "This suite validates supported MojoGP per-task-noise behavior against synthetic truth.",
            ),
            "sample_count_n": _fairness_axis(
                "aligned",
                "Each row evaluates one route on one synthetic dataset with fixed train/test sizes.",
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "Rows use a fixed optimizer family, learning rate, and iteration budget within each route.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "Each row records the configured CG/Lanczos settings for the active route.",
            ),
            "preconditioner": _fairness_axis(
                "aligned",
                "The active route uses its configured pivoted-Cholesky preconditioner budget.",
            ),
            "prediction_mode": _fairness_axis(
                "aligned",
                "The suite reports LOVE-based predictive metrics consistently.",
            ),
            "telemetry": _fairness_axis(
                "observed",
                "MojoGP telemetry is observed for the active route.",
            ),
        },
        "n": int(dataset.X_train.shape[0]),
        "n_test": int(dataset.X_test.shape[0]),
        "d": int(dataset.X_train.shape[1]),
        "num_tasks": int(dataset.Y_train.shape[1]),
        "task_correlation": dataset.true_params["task_correlation"],
        "training_solver_config": {
            "framework": "mojogp",
            "mode": method,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "num_probes": num_probes,
            "max_cg_iter": max_cg_iter,
            "max_tridiag_iter": max_tridiag_iter,
            "precond_rank": precond_rank,
            "early_stop_tol": 0.0,
        },
    }


def _run_multi_output_per_task_noise_subprocess(
    *,
    kernel: str,
    n_train: int,
    n_test: int,
    d: int,
    num_tasks: int,
    task_correlation: str,
    noise_profile: str,
    mean_profile: str,
    method: str,
    results_dir,
) -> BenchmarkResult:
    return run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_multi_output_per_task_noise_case",
        payload={
            "kernel": kernel,
            "n_train": n_train,
            "n_test": n_test,
            "d": d,
            "num_tasks": num_tasks,
            "task_correlation": task_correlation,
            "noise_profile": noise_profile,
            "mean_profile": mean_profile,
            "method": method,
        },
        suite_name="multi_output_per_task_noise_harness",
        benchmark_name="multi_output_per_task_noise",
        framework="mojogp",
        case_id=(
            f"multi_output.per_task_noise.{kernel}.{method}.t{num_tasks}.corr_{task_correlation}."
            f"noise_{noise_profile}.mean_{mean_profile}.n{n_train}.d{d}"
        ),
        benchmark_group_id=(
            f"multi_output.per_task_noise.{kernel}.{method}.t{num_tasks}.corr_{task_correlation}."
            f"noise_{noise_profile}.mean_{mean_profile}"
        ),
        config={
            "benchmark": "multi_output_per_task_noise",
            "framework": "mojogp",
            "model_type": "MultiOutputGP",
            "kernel": kernel,
            "training_method": method,
            "prediction_mode": "love",
            "n": n_train,
            "d": d,
            "num_tasks": num_tasks,
            "task_correlation": task_correlation,
            "noise_profile": noise_profile,
            "mean_profile": mean_profile,
        },
        results_dir=results_dir,
    )


class TestMultiOutputPerTaskNoise:
    def _compute_accuracy(
        self,
        dataset: MultiOutputDataset,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> AccuracyResult:
        std = np.sqrt(np.maximum(variance, 1e-10))
        task_rmses = [rmse(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
        task_maes = [mae(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
        task_r2s = [r_squared(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
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
        coverages = [
            calibration_coverage(dataset.Y_test[:, t], mean[:, t], std[:, t])
            for t in range(mean.shape[1])
        ]
        avg_coverage = {
            level: float(np.mean([entry[level] for entry in coverages]))
            for level in coverages[0]
        }
        return AccuracyResult(
            rmse=_safe_average(task_rmses),
            mae=_safe_average(task_maes),
            r_squared=_safe_average(task_r2s),
            crps=_safe_average(task_crps),
            msll=_safe_average(task_msll),
            calibration_coverage=avg_coverage,
            calibration_error=_safe_average(
                [
                    calibration_error(dataset.Y_test[:, t], mean[:, t], std[:, t])
                    for t in range(mean.shape[1])
                ]
            ),
            sharpness=_safe_average([sharpness(std[:, t]) for t in range(mean.shape[1])]),
            interval_width_95=_safe_average(
                [interval_width(mean[:, t], std[:, t]) for t in range(mean.shape[1])]
            ),
        )

    def _run_case(
        self,
        *,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_correlation: str,
        noise_profile: str,
        mean_profile: str,
        method: str,
        dataset_family: str = "gp_prior",
        extra_config: dict[str, object] | None = None,
    ) -> BenchmarkResult:
        effective_n = (
            scale_n_for_vram(n_train, method)
            if dataset_family == "gp_prior"
            else n_train
        )
        if dataset_family == "structured":
            dataset = generate_multi_output_structured_per_task_noise_data(
                n_train=effective_n,
                n_test=n_test,
                d=d,
                num_tasks=num_tasks,
                noise_per_task=_noise_profile(num_tasks, noise_profile),
                mean_per_task=_mean_profile(num_tasks, mean_profile),
                task_correlation=task_correlation,
                seed=42,
            )
        elif dataset_family == "gp_prior":
            dataset = generate_multi_output_per_task_noise_data(
                n_train=effective_n,
                n_test=n_test,
                d=d,
                num_tasks=num_tasks,
                kernel_type=kernel,
                noise_per_task=_noise_profile(num_tasks, noise_profile),
                mean_per_task=_mean_profile(num_tasks, mean_profile),
                task_correlation=task_correlation,
                seed=42,
            )
        else:
            raise ValueError(f"Unknown per-task-noise dataset_family '{dataset_family}'")
        if method == "materialized":
            max_iterations = 60
            learning_rate = 0.03
            num_probes = 5
            max_cg_iter = 50
            max_tridiag_iter = 20
        else:
            # The strong per-task-noise matrix-free rows need a healthier BBMM budget
            # than the generic smoke settings to recover task ordering reliably.
            max_iterations = 60
            learning_rate = 0.02
            num_probes = 5
            max_cg_iter = 60
            max_tridiag_iter = 15

        reset_torch_memory_stats()
        gpu_monitor = GPUMemoryMonitor(interval=0.1)
        gpu_monitor.start()
        tracemalloc.start()

        gp = MultiOutputGP(
            kernel=kernel,
            num_probes=num_probes,
            max_cg_iterations=max_cg_iter,
            max_tridiag_iterations=max_tridiag_iter,
            use_preconditioner=False,
        )
        fit_start = time.perf_counter()
        result, fit_memory_stats = measure_gpu_phase(
            lambda: gp.fit(
                dataset.X_train,
                dataset.Y_train,
                method=method,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                initial_noise_per_task=np.full(num_tasks, 0.1, dtype=np.float32),
                early_stop_tol=0.0,
                verbose=False,
            ),
            interval=0.02,
        )
        training_time_s = time.perf_counter() - fit_start

        pred_mean_start = time.perf_counter()
        pred_mean = gp.predict(dataset.X_test)
        prediction_mean_time_s = time.perf_counter() - pred_mean_start

        pred_var_start = time.perf_counter()
        (_, pred_var), pred_memory_stats = measure_gpu_phase(
            lambda: gp.predict(dataset.X_test, return_var=True), interval=0.02
        )
        prediction_variance_time_s = time.perf_counter() - pred_var_start

        gpu_monitor.stop()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del current

        memory_stats = _merge_phase_memory(
            gpu_monitor.get_stats(), fit_memory_stats, pred_memory_stats
        )
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)
        prediction_x_test_scaling = _measure_per_task_prediction_x_test_scaling(
            gp=gp,
            dataset=dataset,
            canonical_first_time_s=prediction_variance_time_s,
            canonical_memory_stats=pred_memory_stats,
            seed=42,
            extra_config=extra_config,
        )

        mean = np.asarray(pred_mean.mean, dtype=np.float32)
        variance = np.asarray(pred_var, dtype=np.float32)
        accuracy = self._compute_accuracy(dataset, mean, variance)

        true_noise = np.asarray(dataset.true_params["noise_per_task"], dtype=np.float32)
        learned_noise = np.asarray(result.noise_per_task, dtype=np.float32)
        noise_rel_errors = [
            param_relative_error(float(learned_noise[t]), float(true_noise[t]))
            for t in range(num_tasks)
        ]

        true_mean = np.asarray(dataset.true_params["mean_per_task"], dtype=np.float32)
        learned_mean = np.asarray(result.mean_per_task, dtype=np.float32)
        mean_rel_errors = [
            param_relative_error(float(learned_mean[t]), float(true_mean[t]))
            for t in range(num_tasks)
        ]

        timing_payload = _timing_payload(getattr(result, "iter_times_ms", None))
        iter_timing_quality = (
            "direct_per_iteration"
            if timing_payload["iter_times_ms"] is not None
            else "derived_total_div_iterations"
        )
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
            ms_per_iteration=float(
                timing_payload["iter_time_median_ms"]
                if timing_payload["iter_time_median_ms"] is not None
                else (training_time_s / max(int(result.iterations), 1)) * 1000.0
            ),
            iter_time_min_ms=timing_payload["iter_time_min_ms"],
            iter_time_q25_ms=timing_payload["iter_time_q25_ms"],
            iter_time_mean_ms=timing_payload["iter_time_mean_ms"],
            iter_time_median_ms=timing_payload["iter_time_median_ms"],
            iter_time_q75_ms=timing_payload["iter_time_q75_ms"],
            iter_time_max_ms=timing_payload["iter_time_max_ms"],
            iter_time_p5_ms=timing_payload["iter_time_p5_ms"],
            iter_time_p95_ms=timing_payload["iter_time_p95_ms"],
            iter_times_ms=timing_payload["iter_times_ms"],
            iter_timing_quality=iter_timing_quality,
            prediction_x_test_scaling=prediction_x_test_scaling,
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
            training_peak_gpu_mb=memory_stats.get("training_peak_gpu_mb"),
            training_delta_gpu_mb=memory_stats.get("training_delta_gpu_mb"),
            prediction_peak_gpu_mb=memory_stats.get("prediction_peak_gpu_mb"),
            prediction_delta_gpu_mb=memory_stats.get("prediction_delta_gpu_mb"),
            love_prediction_peak_gpu_mb=memory_stats.get("love_prediction_peak_gpu_mb"),
            love_prediction_delta_gpu_mb=memory_stats.get("love_prediction_delta_gpu_mb"),
        )

        return BenchmarkResult(
            config={
                **_metadata(
                    dataset=dataset,
                    kernel=kernel,
                    method=method,
                    max_iterations=max_iterations,
                    learning_rate=learning_rate,
                    num_probes=num_probes,
                    max_cg_iter=max_cg_iter,
                    max_tridiag_iter=max_tridiag_iter,
                    precond_rank=gp.precond_rank,
                ),
                "requested_n": int(n_train),
                "vram_scaled": effective_n < n_train,
                "noise_profile": noise_profile,
                "mean_profile": mean_profile,
                "dataset_family": dataset.true_params.get("dataset_family", dataset_family),
                "true_noise_per_task": true_noise,
                "learned_noise_per_task": learned_noise,
                "noise_rank_correlation": _noise_rank_correlation(learned_noise, true_noise),
                "noise_rank_correlation_min": NOISE_RANK_CORRELATION_MIN,
                "noise_recovery_policy": "hard_asserted_rank_order_and_predictive_accuracy",
                "per_task_noise_rmse_max": PER_TASK_NOISE_RMSE_MAX,
                "per_task_noise_r2_min": PER_TASK_NOISE_R2_MIN,
                "noise_rel_error_status": "telemetry_only",
                "mean_per_task_rel_error": float(np.mean(mean_rel_errors)),
                "model_family": "MultiOutputGP",
                "feature_surface": "multi_output_per_task_noise",
                "feature_variant": f"per_task_noise_{noise_profile}_mean_{mean_profile}",
                "heteroskedastic_scope": "per_task_observation_noise",
                "unsupported_related_routes": [
                    {
                        "feature": "input_dependent_heteroskedastic_noise_gp",
                        "status": "unsupported_not_benchmarked",
                        "reason": "MojoGP benchmarks exact per-task diagonal noise here, not a latent noise GP approximation.",
                    }
                ],
                **dict(extra_config or {}),
            },
            accuracy=accuracy,
            speed=speed,
            memory=memory,
            hyperparameters=HyperparameterResult(
                learned_lengthscale=float(np.mean(np.asarray(result.params[:-1])))
                if len(result.params) > 1
                else float(result.params[0]),
                learned_noise=float(np.mean(learned_noise)),
                learned_outputscale=float(np.mean(np.asarray(result.effective_scales))),
                learned_mean=float(np.mean(learned_mean)),
                final_nll=float(result.final_nll),
                noise_rel_error=float(np.mean(noise_rel_errors)),
                mean_rel_error=float(np.mean(mean_rel_errors)),
            ),
        )

    def _report_result(self, result: BenchmarkResult, results_dir) -> None:
        print_result(result)

    def _assert_per_task_noise_contract(self, result: BenchmarkResult) -> None:
        assert not result.speed.early_stopped, "Per-task-noise benchmark should run its full budget"
        assert np.isfinite(result.accuracy.rmse)
        assert np.isfinite(result.accuracy.r_squared)
        assert result.accuracy.rmse < result.config["per_task_noise_rmse_max"], (
            "Per-task-noise predictive RMSE unexpectedly high"
        )
        assert result.accuracy.r_squared >= result.config["per_task_noise_r2_min"], (
            f"Per-task-noise predictive R2 unexpectedly low: {result.accuracy.r_squared:.3f}"
        )
        assert result.config["noise_rank_correlation"] >= result.config[
            "noise_rank_correlation_min"
        ], (
            f"Per-task noise ordering was not recovered well enough: {result.config['noise_rank_correlation']:.3f}"
        )

    @pytest.mark.minimal
    @pytest.mark.multi_output
    @pytest.mark.accuracy
    @requires_cuda
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,task_correlation,noise_profile,mean_profile,method",
        MINIMAL_CONFIGS,
    )
    def test_per_task_noise_recovery_benchmark_core_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_correlation: str,
        noise_profile: str,
        mean_profile: str,
        method: str,
        results_dir,
        n_override,
    ):
        assert_gpu_available()
        if n_override is not None:
            n_train = n_override
        result = _run_multi_output_per_task_noise_subprocess(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            num_tasks=num_tasks,
            task_correlation=task_correlation,
            noise_profile=noise_profile,
            mean_profile=mean_profile,
            method=method,
            results_dir=results_dir,
        )
        self._report_result(result, results_dir)
        assert_gpu_was_used(result)
        self._assert_per_task_noise_contract(result)

    @pytest.mark.moderate
    @pytest.mark.multi_output
    @pytest.mark.accuracy
    @requires_cuda
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,task_correlation,noise_profile,mean_profile,method",
        MODERATE_CONFIGS,
    )
    def test_per_task_noise_recovery_benchmark_extended_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_correlation: str,
        noise_profile: str,
        mean_profile: str,
        method: str,
        results_dir,
        n_override,
    ):
        assert_gpu_available()
        if n_override is not None:
            n_train = n_override
        result = _run_multi_output_per_task_noise_subprocess(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            num_tasks=num_tasks,
            task_correlation=task_correlation,
            noise_profile=noise_profile,
            mean_profile=mean_profile,
            method=method,
            results_dir=results_dir,
        )
        self._report_result(result, results_dir)
        assert_gpu_was_used(result)
        assert result.hyperparameters.noise_rel_error is not None
        self._assert_per_task_noise_contract(result)
