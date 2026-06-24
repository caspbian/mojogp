"""Run one mixed-kernel benchmark case in isolation."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

from mojogp import SingleOutputGP, Kernel, RBF
from tests.benchmarks.prediction_workload import (
    BENCHMARK_PREDICTION_N_TEST,
    clear_prediction_x_test_failure_memory,
    prediction_x_test_repeat_count,
    prediction_x_test_scaling_entry,
    prediction_x_test_scaling_failure_entry,
    prediction_x_test_should_record_failure,
    prediction_x_test_target_specs,
)
from tests.shared.subprocess_harness import run_child_main

from tests.shared.benchmarking.data_generators import generate_mixed_categorical_data
from tests.shared.benchmarking.gpu_memory import (
    GPUMemoryMonitor,
    get_torch_memory_stats,
    measure_gpu_phase,
    reset_torch_memory_stats,
)
from tests.shared.benchmarking.metrics import calibration_coverage, calibration_error, crps_gaussian, mae, mean_standardized_log_loss, rmse, r_squared, sharpness, interval_width
from tests.shared.benchmarking.report import save_result_artifact
from tests.shared.benchmarking.result_types import AccuracyResult, BenchmarkResult, HyperparameterResult, MemoryResult, SpeedResult


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


def _mixed_prediction_inputs(
    *,
    n_test: int,
    dataset,
    cont_dim: int,
    cat_levels: list[int],
    seed: int,
) -> np.ndarray:
    if int(dataset.X_test.shape[0]) == int(n_test):
        return np.asarray(dataset.X_test, dtype=np.float32)
    rng = np.random.RandomState(seed + 1_000_003 + int(n_test))
    X_cont = rng.randn(int(n_test), cont_dim).astype(np.float32)
    C = np.column_stack(
        [rng.randint(0, levels, size=int(n_test)) for levels in cat_levels]
    ).astype(np.float32)
    return np.column_stack([X_cont, C]).astype(np.float32)


def _measure_mixed_prediction_x_test_scaling(
    *,
    gp: SingleOutputGP,
    dataset,
    cont_dim: int,
    cat_levels: list[int],
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
                X_test = _mixed_prediction_inputs(
                    n_test=n_test,
                    dataset=dataset,
                    cont_dim=cont_dim,
                    cat_levels=cat_levels,
                    seed=seed,
                )

                def _predict_size():
                    return gp.predict(X_test)

                first_start = time.perf_counter()
                _, first_memory_stats = measure_gpu_phase(_predict_size, interval=0.02)
                first_time_s = float(time.perf_counter() - first_start)

            def _predict_repeat():
                return gp.predict(X_test)

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


def _mixed_kernel(cont_dim: int, cat_levels: list[int]):
    kernel = Kernel.rbf(active_dims=list(range(cont_dim)))
    for cat_idx, levels in enumerate(cat_levels):
        kernel = kernel * Kernel.ehh(levels=int(levels), active_dims=[cont_dim + cat_idx])
    return kernel


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    method = str(payload["method"])
    extra_config = dict(payload.get("extra_config", {}))
    cont_dim = int(payload.get("cont_dim", 2))
    cat_levels = [int(level) for level in payload.get("cat_levels", [3])]
    n_train = int(payload.get("n_train", 1500 if method == "materialized" else 4000))
    n_test = int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST))
    max_iterations = int(payload.get("max_iterations", 80 if method == "materialized" else 50))
    learning_rate = float(payload.get("learning_rate", 0.03))
    seed = int(payload.get("seed", 42 if method == "materialized" else 123))
    dataset = generate_mixed_categorical_data(
        n_train=n_train,
        n_test=n_test,
        cont_dim=cont_dim,
        cat_levels=cat_levels,
        true_noise=float(payload.get("true_noise", 0.08)),
        seed=seed,
    )
    mixed_gp = SingleOutputGP(_mixed_kernel(cont_dim, cat_levels))
    baseline_gp = SingleOutputGP(RBF(active_dims=list(range(cont_dim))))
    reset_torch_memory_stats()
    monitor = GPUMemoryMonitor(interval=0.1)
    monitor.start()
    tracemalloc.start()
    fit_start = time.perf_counter()
    _, fit_memory_stats = measure_gpu_phase(
        lambda: mixed_gp.fit(
            dataset.X_train,
            dataset.y_train,
            method=method,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            verbose=False,
        ),
        interval=0.02,
    )
    mixed_train_time_s = time.perf_counter() - fit_start
    baseline_gp.fit(dataset.X_train, dataset.y_train, method=method, max_iterations=max_iterations, learning_rate=learning_rate, verbose=False)
    pred_start = time.perf_counter()
    mixed_pred, pred_memory_stats = measure_gpu_phase(
        lambda: mixed_gp.predict(dataset.X_test, return_std=True), interval=0.02
    )
    prediction_time_s = time.perf_counter() - pred_start
    baseline_pred = baseline_gp.predict(dataset.X_test)
    monitor.stop()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del current
    memory_stats = _merge_phase_memory(
        monitor.get_stats(), fit_memory_stats, pred_memory_stats
    )
    memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)
    prediction_x_test_scaling = _measure_mixed_prediction_x_test_scaling(
        gp=mixed_gp,
        dataset=dataset,
        cont_dim=cont_dim,
        cat_levels=cat_levels,
        seed=seed,
        tier=str(extra_config.get("benchmark_route_tier", "xsmall")),
        variety=str(extra_config.get("benchmark_variety", "standard")),
        canonical_first_time_s=prediction_time_s,
        canonical_memory_stats=pred_memory_stats,
    )
    mixed_mean, mixed_std = mixed_pred
    mixed_mean = np.asarray(mixed_mean, dtype=np.float32)
    mixed_std = np.asarray(mixed_std, dtype=np.float32)
    baseline_mean = np.asarray(baseline_pred.mean, dtype=np.float32)
    mixed_rmse = rmse(dataset.f_test, mixed_mean)
    baseline_rmse = rmse(dataset.f_test, baseline_mean)
    training_result = mixed_gp.training_result
    params = np.asarray(training_result.params, dtype=np.float32)
    learned_outputscale = float(params[-1]) if params.size > 1 else float(params[0])
    timing_payload = _timing_payload(getattr(training_result, "iter_times_ms", None))
    iter_timing_quality = (
        "direct_per_iteration"
        if timing_payload["iter_times_ms"] is not None
        else "derived_total_div_iterations"
    )
    kernel_label = "rbf_x_" + "_x_".join(f"ehh{levels}" for levels in cat_levels)
    benchmark = BenchmarkResult(
        config={
            "benchmark": "mixed_accuracy",
            "route_group": "mixed",
            "framework": "mojogp",
            "model_family": "SingleOutputGP",
            "feature_surface": "mixed_continuous_categorical",
            "feature_variant": f"cont{cont_dim}_cat{'x'.join(map(str, cat_levels))}",
            "model_type": "SingleOutputGP",
            "kernel": kernel_label,
            "training_method": method,
            "method": method,
            "prediction_mode": "love",
            "comparison_class": "mojogp_only",
            "baseline_backend": "none",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": "N.B. MojoGP-only mixed-kernel row: this benchmark compares a discrete-aware mixed kernel against a continuous-only in-repo baseline on the same synthetic dataset.",
            "fairness_axes": {
                "comparator_scope": _fairness_axis("mojogp_only", "The baseline is the in-repo continuous-only model, not a cross-framework comparator."),
                "sample_count_n": _fairness_axis("aligned", "Both baselines run on the same mixed synthetic dataset split."),
                "optimizer": _fairness_axis("aligned", "Both baselines use the same optimizer family and budget."),
                "solver_budget": _fairness_axis("aligned", "Both baselines use the same route-level CG/Lanczos settings."),
                "preconditioner": _fairness_axis("aligned", "Both baselines use the same route-level preconditioner settings."),
                "prediction_mode": _fairness_axis("aligned", "This suite reports LOVE-based predictive metrics consistently."),
                "telemetry": _fairness_axis("observed", "MojoGP telemetry is observed for the active mixed route."),
            },
            "n": int(dataset.X_train.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "n_test": int(dataset.X_test.shape[0]),
            "continuous_dim": cont_dim,
            "num_categorical": len(cat_levels),
            "categorical_levels": cat_levels,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": float(r_squared(dataset.f_test, baseline_mean)),
            "joint_gain_vs_baseline": baseline_rmse - mixed_rmse,
            "unsupported_related_routes": [
                {
                    "feature": "pure_categorical_exactgp",
                    "status": "unsupported_not_benchmarked",
                    "reason": "The active public mixed-kernel benchmark route requires at least one continuous kernel component.",
                }
            ],
            **extra_config,
        },
        accuracy=AccuracyResult(
            rmse=float(mixed_rmse),
            mae=float(mae(dataset.f_test, mixed_mean)),
            r_squared=float(r_squared(dataset.f_test, mixed_mean)),
            crps=float(crps_gaussian(dataset.y_test, mixed_mean, mixed_std)),
            msll=float(mean_standardized_log_loss(dataset.y_test, mixed_mean, mixed_std, y_train_mean=float(np.mean(dataset.y_train)), y_train_std=float(np.std(dataset.y_train)))),
            calibration_coverage=calibration_coverage(dataset.y_test, mixed_mean, mixed_std),
            calibration_error=float(calibration_error(dataset.y_test, mixed_mean, mixed_std)),
            sharpness=float(sharpness(mixed_std)),
            interval_width_95=float(interval_width(mixed_mean, mixed_std)),
        ),
        speed=SpeedResult(
            training_time_s=float(mixed_train_time_s),
            prediction_mean_time_s=float(prediction_time_s),
            prediction_variance_time_s=float(prediction_time_s),
            end_to_end_time_s=float(mixed_train_time_s + prediction_time_s),
            iterations_run=int(training_result.iterations),
            max_iterations=max_iterations,
            early_stopped=int(training_result.iterations) < max_iterations,
            ms_per_iteration=float(
                timing_payload["iter_time_median_ms"]
                if timing_payload["iter_time_median_ms"] is not None
                else mixed_train_time_s / max(int(training_result.iterations), 1) * 1000.0
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
            learned_lengthscale=float(params[0]),
            learned_noise=float(training_result.noise),
            learned_outputscale=learned_outputscale,
            final_nll=float(training_result.nll),
            learned_mean=float(training_result.mean),
        ),
    )
    result_path = save_result_artifact(benchmark, Path(str(payload["results_dir"])), "mixed_accuracy")
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
