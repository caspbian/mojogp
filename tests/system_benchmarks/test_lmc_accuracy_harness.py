"""Harness benchmark for LMC accuracy and ablations."""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest
import torch

from mojogp import Kernel, MultiOutputGP, MultiOutputLMCGP
from tests.benchmarks.prediction_workload import (
    BENCHMARK_PREDICTION_N_TEST,
    clear_prediction_x_test_failure_memory,
    prediction_x_test_repeat_count,
    prediction_x_test_scaling_entry,
    prediction_x_test_scaling_failure_entry,
    prediction_x_test_should_record_failure,
    prediction_x_test_target_specs,
)
from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess

from .conftest import assert_gpu_available, requires_cuda
from tests.shared.benchmarking.data_generators import generate_multi_output_heterogeneous_latent_data
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
    rmse,
    r_squared,
    sharpness,
    interval_width,
)
from tests.shared.benchmarking.report import print_result
from tests.shared.benchmarking.result_types import AccuracyResult, BenchmarkResult, HyperparameterResult, MemoryResult, SpeedResult


LMC_VS_ICM_RMSE_RATIO_MAX = 0.8
LMC_VS_ICM_R2_DELTA_MIN = 0.05


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


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


def _lmc_prediction_inputs(*, n_test: int, dataset, seed: int) -> np.ndarray:
    if int(dataset.X_test.shape[0]) == int(n_test):
        return np.asarray(dataset.X_test, dtype=np.float32)
    rng = np.random.default_rng(seed + 1_000_003 + int(n_test))
    return rng.uniform(-2.5, 2.5, size=(int(n_test), 1)).astype(np.float32)


def _measure_lmc_prediction_x_test_scaling(
    *,
    lmc: MultiOutputLMCGP,
    dataset,
    seed: int,
    tier: str,
    variety: str,
    canonical_first_time_s: float,
    canonical_memory_stats: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in prediction_x_test_target_specs(
        variety=variety,
        tier=tier,
        framework="mojogp",
    ):
        n_test = int(spec["n_test"])
        try:
            if n_test == int(dataset.X_test.shape[0]):
                X_test = np.asarray(dataset.X_test, dtype=np.float32)
                first_time_s = canonical_first_time_s
                first_memory_stats = canonical_memory_stats
            else:
                X_test = _lmc_prediction_inputs(n_test=n_test, dataset=dataset, seed=seed)

                def _predict_size():
                    return lmc.predict(X_test, return_var=True)

                first_start = time.perf_counter()
                _, first_memory_stats = measure_gpu_phase(_predict_size, interval=0.02)
                first_time_s = float(time.perf_counter() - first_start)

            def _predict_repeat():
                return lmc.predict(X_test, return_var=True)

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


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def _run_lmc_accuracy_case(
    *,
    n_train: int = 2000,
    n_test: int = BENCHMARK_PREDICTION_N_TEST,
    method: str = "materialized",
    max_iterations: int = 40,
    learning_rate: float = 0.03,
    seed: int = 42,
    extra_config: dict[str, object] | None = None,
) -> BenchmarkResult:
    assert_gpu_available()
    dataset = generate_multi_output_heterogeneous_latent_data(
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )

    # The active LMC wrapper path is sensitive to random initialization.
    # Fix the optimizer start state so this ablation measures model capacity
    # rather than optimizer lottery.
    np.random.seed(42)
    icm = MultiOutputGP(
        kernel="rbf",
        task_rank=1,
        num_probes=5,
        max_cg_iterations=50,
        use_preconditioner=False,
    )
    np.random.seed(42)
    lmc = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=5,
        max_cg_iterations=50,
        use_preconditioner=False,
    )

    reset_torch_memory_stats()
    monitor = GPUMemoryMonitor(interval=0.1)
    monitor.start()
    tracemalloc.start()

    icm_fit_start = time.perf_counter()
    icm.fit(
        dataset.X_train,
        dataset.Y_train,
        method=method,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        verbose=False,
    )
    icm_training_time_s = time.perf_counter() - icm_fit_start

    lmc_fit_start = time.perf_counter()
    result, fit_memory_stats = measure_gpu_phase(
        lambda: lmc.fit(
            dataset.X_train,
            dataset.Y_train,
            method=method,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            verbose=False,
        ),
        interval=0.02,
    )
    lmc_training_time_s = time.perf_counter() - lmc_fit_start

    icm_mean, _ = icm.predict(dataset.X_test, return_var=True)
    pred_start = time.perf_counter()
    (lmc_mean, lmc_var), pred_memory_stats = measure_gpu_phase(
        lambda: lmc.predict(dataset.X_test, return_var=True), interval=0.02
    )
    prediction_time_s = time.perf_counter() - pred_start

    monitor.stop()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del current
    memory_stats = _merge_phase_memory(
        monitor.get_stats(), fit_memory_stats, pred_memory_stats
    )
    memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)
    prediction_x_test_scaling = _measure_lmc_prediction_x_test_scaling(
        lmc=lmc,
        dataset=dataset,
        seed=seed,
        tier=str((extra_config or {}).get("benchmark_route_tier", "xsmall")),
        variety=str((extra_config or {}).get("benchmark_variety", "standard")),
        canonical_first_time_s=prediction_time_s,
        canonical_memory_stats=pred_memory_stats,
    )

    icm_rmse = float(
        np.mean(
            [
                rmse(dataset.F_test[:, t], icm_mean[:, t])
                for t in range(icm_mean.shape[1])
            ]
        )
    )
    lmc_rmse = float(
        np.mean(
            [
                rmse(dataset.F_test[:, t], lmc_mean[:, t])
                for t in range(lmc_mean.shape[1])
            ]
        )
    )
    icm_r2 = float(
        np.mean(
            [
                r_squared(dataset.F_test[:, t], icm_mean[:, t])
                for t in range(icm_mean.shape[1])
            ]
        )
    )
    lmc_r2 = float(
        np.mean(
            [
                r_squared(dataset.F_test[:, t], lmc_mean[:, t])
                for t in range(lmc_mean.shape[1])
            ]
        )
    )
    lmc_std = np.sqrt(np.maximum(np.asarray(lmc_var, dtype=np.float32), 1e-10))
    lengthscales = np.asarray(result.lengthscales, dtype=np.float32)
    outputscales = np.asarray(result.outputscales, dtype=np.float32)
    timing_payload = _timing_payload(getattr(result, "iter_times_ms", None))
    iter_timing_quality = (
        "direct_per_iteration"
        if timing_payload["iter_times_ms"] is not None
        else "derived_total_div_iterations"
    )
    benchmark = BenchmarkResult(
        config={
            "benchmark": "lmc_ablation_accuracy",
            "route_group": "multi_output_lmc",
            "framework": "mojogp",
            "model_family": "MultiOutputLMCGP",
            "feature_surface": "multi_output_lmc",
            "feature_variant": "two_latent_rbf_plus_matern52",
            "model_type": "MultiOutputLMCGP",
            "kernel": "rbf_plus_matern52",
            "training_method": method,
            "method": method,
            "prediction_mode": "love",
            "comparison_class": "mojogp_only",
            "baseline_backend": "none",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": (
                "N.B. MojoGP-only LMC ablation row: this benchmark compares two-latent LMC against "
                "the in-repo ICM baseline on heterogeneous latent synthetic data."
            ),
            "fairness_axes": {
                "comparator_scope": _fairness_axis(
                    "mojogp_only",
                    "The baseline is the in-repo ICM route, not a cross-framework comparator.",
                ),
                "sample_count_n": _fairness_axis(
                    "aligned",
                    "LMC and ICM run on the same synthetic dataset split.",
                ),
                "optimizer": _fairness_axis(
                    "aligned",
                    "Both baselines use the same optimizer family, initialization seed, and iteration budget.",
                ),
                "solver_budget": _fairness_axis(
                    "aligned",
                    "LMC and ICM use the same route-level CG and Lanczos budgets.",
                ),
                "preconditioner": _fairness_axis(
                    "aligned",
                    "Both baselines use the same route-level preconditioner budget.",
                ),
                "prediction_mode": _fairness_axis(
                    "aligned",
                    "This suite reports LOVE-based predictive metrics for the LMC row.",
                ),
                "telemetry": _fairness_axis(
                    "observed",
                    "MojoGP telemetry is observed for the active LMC route.",
                ),
            },
            "n": int(dataset.X_train.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "n_test": int(dataset.X_test.shape[0]),
            "num_tasks": int(dataset.Y_train.shape[1]),
            "baseline_type": "icm",
            "baseline_rmse": icm_rmse,
            "baseline_r2": icm_r2,
            "baseline_training_time_s": icm_training_time_s,
            "joint_gain_vs_baseline": icm_rmse - lmc_rmse,
            "joint_vs_icm_policy": "hard_asserted_rmse_ratio_and_r2_delta",
            "joint_vs_icm_rmse_ratio": float(lmc_rmse / max(icm_rmse, 1e-6)),
            "joint_vs_icm_rmse_ratio_max": LMC_VS_ICM_RMSE_RATIO_MAX,
            "joint_vs_icm_r2_delta": float(lmc_r2 - icm_r2),
            "joint_vs_icm_r2_delta_min": LMC_VS_ICM_R2_DELTA_MIN,
            **dict(extra_config or {}),
        },
        accuracy=AccuracyResult(
            rmse=lmc_rmse,
            mae=float(mae(dataset.F_test, lmc_mean)),
            r_squared=lmc_r2,
            crps=crps_gaussian(dataset.Y_test, lmc_mean, lmc_std),
            msll=mean_standardized_log_loss(
                dataset.Y_test,
                lmc_mean,
                lmc_std,
                y_train_mean=float(np.mean(dataset.Y_train)),
                y_train_std=float(np.std(dataset.Y_train)),
            ),
            calibration_coverage=calibration_coverage(dataset.Y_test, lmc_mean, lmc_std),
            calibration_error=calibration_error(dataset.Y_test, lmc_mean, lmc_std),
            sharpness=sharpness(lmc_std),
            interval_width_95=interval_width(lmc_mean, lmc_std),
        ),
        speed=SpeedResult(
            training_time_s=lmc_training_time_s,
            prediction_mean_time_s=prediction_time_s,
            prediction_variance_time_s=prediction_time_s,
            end_to_end_time_s=lmc_training_time_s + prediction_time_s,
            iterations_run=int(result.iterations),
            max_iterations=max_iterations,
            early_stopped=int(result.iterations) < max_iterations,
            ms_per_iteration=float(
                timing_payload["iter_time_median_ms"]
                if timing_payload["iter_time_median_ms"] is not None
                else lmc_training_time_s / max(int(result.iterations), 1) * 1000.0
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
            training_peak_gpu_mb=memory_stats.get("training_peak_gpu_mb"),
            training_delta_gpu_mb=memory_stats.get("training_delta_gpu_mb"),
            prediction_peak_gpu_mb=memory_stats.get("prediction_peak_gpu_mb"),
            prediction_delta_gpu_mb=memory_stats.get("prediction_delta_gpu_mb"),
            love_prediction_peak_gpu_mb=memory_stats.get("love_prediction_peak_gpu_mb"),
            love_prediction_delta_gpu_mb=memory_stats.get("love_prediction_delta_gpu_mb"),
        ),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(lengthscales)),
            learned_noise=float(np.mean(np.asarray(result.noise_per_task))),
            learned_outputscale=float(np.mean(outputscales)),
            final_nll=float(result.final_nll),
            learned_mean=float(np.mean(np.asarray(result.mean_per_task))),
        ),
    )
    return benchmark


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_two_latent_lmc_beats_icm_on_heterogeneous_latent_data(results_dir):
    benchmark = run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_lmc_accuracy_case",
        payload={},
        suite_name="lmc_accuracy_harness",
        benchmark_name="lmc_ablation_accuracy",
        framework="mojogp",
        case_id="lmc.accuracy.two_latent_vs_icm",
        benchmark_group_id="lmc.accuracy.two_latent_vs_icm",
        config={
            "benchmark": "lmc_ablation_accuracy",
            "framework": "mojogp",
            "model_type": "MultiOutputLMCGP",
            "kernel": "rbf_plus_matern52",
            "training_method": "materialized",
            "prediction_mode": "love",
        },
        results_dir=results_dir,
    )
    print_result(benchmark)

    assert np.isfinite(benchmark.accuracy.rmse)
    assert benchmark.config["joint_vs_icm_rmse_ratio"] <= benchmark.config[
        "joint_vs_icm_rmse_ratio_max"
    ], (
        "LMC did not beat the ICM baseline strongly enough on heterogeneous latent data"
    )
    assert benchmark.config["joint_vs_icm_r2_delta"] >= benchmark.config[
        "joint_vs_icm_r2_delta_min"
    ], (
        "LMC R2 improvement over ICM fell below the heterogeneous-latent evidence contract"
    )
