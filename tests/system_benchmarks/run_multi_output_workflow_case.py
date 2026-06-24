"""Run one multi-output workflow benchmark case in isolation."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from mojogp import MultiOutputGP
from tests.benchmarks.prediction_workload import (
    BENCHMARK_PATHWISE_SAMPLING_N_TEST,
    BENCHMARK_PREDICTION_N_TEST,
)
from tests.shared.subprocess_harness import IsolatedGPUTestSession, run_child_main

from tests.shared.benchmarking.data_generators import generate_multi_output_data
from tests.shared.benchmarking.report import save_result_artifact
from tests.shared.benchmarking.result_types import AccuracyResult, BenchmarkResult, HyperparameterResult, MemoryResult, SpeedResult
from tests.shared.benchmarking.startup_timing import measure_startup_profile


def _memory_result(memory_stats: dict[str, float]) -> MemoryResult:
    return MemoryResult(
        gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
        gpu_min_mb=memory_stats.get("min_mb", 0.0),
        gpu_max_mb=memory_stats.get("max_mb", 0.0),
        gpu_var_mb=memory_stats.get("var_mb", 0.0),
        torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
        torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
        cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
        measurement_method=str(memory_stats.get("method", "none")),
        num_samples=int(memory_stats.get("samples", 0)),
        gpu_baseline_mb=memory_stats.get("baseline_gpu_mb"),
        gpu_current_mb=memory_stats.get("current_gpu_mb"),
        gpu_delta_mb=memory_stats.get("delta_gpu_mb"),
        gpu_isolated_peak_mb=memory_stats.get("isolated_peak_gpu_mb"),
        gpu_isolated_current_mb=memory_stats.get("isolated_current_gpu_mb"),
        torch_baseline_mb=memory_stats.get("torch_baseline_mb"),
        torch_peak_delta_mb=memory_stats.get("torch_peak_delta_mb"),
        torch_current_delta_mb=memory_stats.get("torch_current_delta_mb"),
        torch_reserved_mb=memory_stats.get("torch_reserved_mb"),
        torch_reserved_baseline_mb=memory_stats.get("torch_reserved_baseline_mb"),
        torch_reserved_delta_mb=memory_stats.get("torch_reserved_delta_mb"),
    )


def _make_gp(method: str) -> MultiOutputGP:
    return MultiOutputGP(
        kernel="rbf",
        num_probes=4,
        max_cg_iterations=60,
        use_preconditioner=False,
    )


def _probe_multi_output_compile(method: str, d: int, num_tasks: int) -> None:
    gp = _make_gp(method)
    gp._ensure_compiled(d, num_tasks=num_tasks)


def _persistence_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    method = str(payload["training_method"])
    dataset = generate_multi_output_data(
        n_train=int(payload.get("n_train", 800)),
        n_test=int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST)),
        d=int(payload.get("d", 5)),
        num_tasks=int(payload.get("num_tasks", 3)),
        kernel_type="rbf",
        seed=int(payload.get("seed", 11)),
    )
    gp, startup_profile = measure_startup_profile(
        prepare_factory=lambda: _make_gp(method),
        cold_start_probe=lambda: _probe_multi_output_compile(
            method,
            int(dataset.X_train.shape[1]),
            int(dataset.Y_train.shape[1]),
        ),
        cache_prefix="mojogp_multi_output_workflow_",
    )

    fit_start = time.perf_counter()
    result = gp.fit(dataset.X_train, dataset.Y_train, max_iterations=35, learning_rate=0.03, verbose=False)
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred_before = gp.predict(dataset.X_test, variance_method="exact")
    persist_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mojogp-multi-persist-") as tmpdir:
        model_path = Path(tmpdir) / "multi_output_gp"
        gp.save(str(model_path))
        loaded = MultiOutputGP.load(str(model_path))
        pred_after = loaded.predict(dataset.X_test, variance_method="exact")
    persistence_time_s = time.perf_counter() - persist_start
    persist_snapshot = session.snapshot_gpu()

    mean_diff = np.asarray(pred_before.mean - pred_after.mean, dtype=np.float32)
    var_diff = np.asarray(pred_before.variance - pred_after.variance, dtype=np.float32)
    max_abs_mean_diff = float(np.max(np.abs(mean_diff)))
    max_abs_var_diff = float(np.max(np.abs(var_diff)))
    mean_threshold = float(payload.get("max_abs_mean_diff_threshold", 1e-5))
    var_threshold = float(payload.get("max_abs_var_diff_threshold", 1e-5))
    if max_abs_mean_diff > mean_threshold or max_abs_var_diff > var_threshold:
        raise AssertionError(
            "Multi-output save/load prediction round trip changed outputs: "
            f"max_abs_mean_diff={max_abs_mean_diff:.3e}, "
            f"max_abs_var_diff={max_abs_var_diff:.3e}"
        )
    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, persist_snapshot])

    return BenchmarkResult(
        config={
            "benchmark": "multi_output_persistence_harness",
            "route_group": "multi_output",
            "framework": "mojogp",
            "model_type": "MultiOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": "exact",
            "workflow": "persistence",
            "n": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "num_tasks": int(dataset.Y_train.shape[1]),
            "max_abs_mean_diff": max_abs_mean_diff,
            "max_abs_var_diff": max_abs_var_diff,
            "max_abs_mean_diff_threshold": mean_threshold,
            "max_abs_var_diff_threshold": var_threshold,
            "variance_round_trip_policy": "hard_assertion",
            "comparison_class": "mojogp_only",
            "baseline_backend": "none",
        },
        accuracy=AccuracyResult(
            rmse=float(np.sqrt(np.mean(mean_diff**2))),
            mae=float(np.mean(np.abs(mean_diff))),
            r_squared=1.0,
            crps=max_abs_var_diff,
            msll=max_abs_mean_diff,
            calibration_coverage={},
            calibration_error=max_abs_var_diff,
            sharpness=float(np.mean(np.sqrt(np.maximum(pred_after.variance, 1e-10)))),
            interval_width_95=float(3.92 * np.mean(np.sqrt(np.maximum(pred_after.variance, 1e-10)))),
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=persistence_time_s,
            prediction_variance_time_s=persistence_time_s,
            end_to_end_time_s=training_time_s + persistence_time_s,
            iterations_run=int(result.iterations),
            max_iterations=35,
            early_stopped=int(result.iterations) < 35,
            ms_per_iteration=training_time_s / max(int(result.iterations), 1) * 1000.0,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(getattr(result, "lengthscale", 1.0)),
            learned_noise=float(getattr(result, "noise", 0.1)),
            learned_outputscale=float(getattr(result, "outputscale", 1.0)),
            learned_mean=float(np.mean(getattr(result, "mean_per_task", np.zeros(dataset.Y_train.shape[1], dtype=np.float32)))),
            final_nll=float(result.final_nll),
        ),
    )


def _sampling_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    method = str(payload["training_method"])
    sampling_method = str(payload["sampling_method"])
    default_n_test = BENCHMARK_PATHWISE_SAMPLING_N_TEST if sampling_method == "pathwise" else 96
    dataset = generate_multi_output_data(
        n_train=int(payload.get("n_train", 800)),
        n_test=int(payload.get("n_test", default_n_test)),
        d=int(payload.get("d", 5)),
        num_tasks=int(payload.get("num_tasks", 3)),
        kernel_type="rbf",
        seed=int(payload.get("seed", 22)),
    )
    gp, startup_profile = measure_startup_profile(
        prepare_factory=lambda: _make_gp(method),
        cold_start_probe=lambda: _probe_multi_output_compile(
            method,
            int(dataset.X_train.shape[1]),
            int(dataset.Y_train.shape[1]),
        ),
        cache_prefix="mojogp_multi_output_workflow_",
    )

    fit_start = time.perf_counter()
    result = gp.fit(dataset.X_train, dataset.Y_train, max_iterations=35, learning_rate=0.03, verbose=False)
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred = gp.predict(dataset.X_test)
    sample_start = time.perf_counter()
    samples = gp.sample_posterior(
        dataset.X_test,
        n_samples=int(payload.get("n_samples", 24)),
        method=sampling_method,
        n_rff_features=int(payload.get("n_rff_features", 512)),
        rng=np.random.default_rng(int(payload.get("sample_seed", 9))),
    )
    sample_time_s = time.perf_counter() - sample_start
    sample_snapshot = session.snapshot_gpu()
    sample_mean = np.mean(samples, axis=0).astype(np.float32)
    sample_std = np.std(samples, axis=0).astype(np.float32)
    pred_std = np.sqrt(np.maximum(pred.variance, 1e-10)).astype(np.float32)

    mean_rmse = float(np.sqrt(np.mean((sample_mean - pred.mean) ** 2)))
    std_rmse = float(np.sqrt(np.mean((sample_std - pred_std) ** 2)))
    mean_threshold = float(payload.get("sample_mean_rmse_threshold", 2.5))
    std_threshold = float(payload.get("sample_std_rmse_threshold", 2.5))
    sample_info = dict(gp.backend_sample_info or {})
    expected_route = (
        "provider_pathwise"
        if sampling_method == "pathwise"
        else "diagonal_from_predictive_std"
    )
    actual_route = str(sample_info.get("actual_sampling_route", "missing"))
    if actual_route != expected_route:
        raise AssertionError(
            f"Multi-output sampling used route {actual_route!r}, expected {expected_route!r}"
        )
    if not np.all(np.isfinite(samples)):
        raise AssertionError("Multi-output posterior samples contain NaN or Inf values")
    if float(np.mean(sample_std)) <= 0.0:
        raise AssertionError("Multi-output posterior samples have zero empirical spread")
    if mean_rmse > mean_threshold or std_rmse > std_threshold:
        raise AssertionError(
            "Multi-output sampling moments diverged from predictive moments: "
            f"sample_mean_rmse={mean_rmse:.3e} threshold={mean_threshold:.3e}, "
            f"sample_std_rmse={std_rmse:.3e} threshold={std_threshold:.3e}"
        )
    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, sample_snapshot])

    return BenchmarkResult(
        config={
            "benchmark": "multi_output_sampling_harness",
            "route_group": "multi_output",
            "framework": "mojogp",
            "model_type": "MultiOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": sampling_method,
            "workflow": "sampling",
            "sampling_method": sampling_method,
            "n": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "num_tasks": int(dataset.Y_train.shape[1]),
            "sample_mean_rmse": mean_rmse,
            "sample_std_rmse": std_rmse,
            "sample_mean_rmse_threshold": mean_threshold,
            "sample_std_rmse_threshold": std_threshold,
            "actual_sampling_route": actual_route,
            "sampling_consistency_policy": "hard_assertion",
            "comparison_class": "mojogp_only",
            "baseline_backend": "none",
        },
        accuracy=AccuracyResult(
            rmse=mean_rmse,
            mae=float(np.mean(np.abs(sample_mean - pred.mean))),
            r_squared=1.0,
            crps=std_rmse,
            msll=mean_rmse,
            calibration_coverage={},
            calibration_error=std_rmse,
            sharpness=float(np.mean(sample_std)),
            interval_width_95=float(3.92 * np.mean(sample_std)),
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=sample_time_s,
            prediction_variance_time_s=sample_time_s,
            end_to_end_time_s=training_time_s + sample_time_s,
            iterations_run=int(result.iterations),
            max_iterations=35,
            early_stopped=int(result.iterations) < 35,
            ms_per_iteration=training_time_s / max(int(result.iterations), 1) * 1000.0,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(getattr(result, "lengthscale", 1.0)),
            learned_noise=float(getattr(result, "noise", 0.1)),
            learned_outputscale=float(getattr(result, "outputscale", 1.0)),
            learned_mean=float(np.mean(getattr(result, "mean_per_task", np.zeros(dataset.Y_train.shape[1], dtype=np.float32)))),
            final_nll=float(result.final_nll),
        ),
    )


def _handle(payload: dict[str, object], session: IsolatedGPUTestSession) -> dict[str, object]:
    case = str(payload["case"])
    results_dir = Path(str(payload["results_dir"]))
    if case == "persistence":
        result = _persistence_case(payload, session)
        benchmark_name = "multi_output_persistence_harness"
    elif case == "sampling":
        result = _sampling_case(payload, session)
        benchmark_name = "multi_output_sampling_harness"
    else:
        raise ValueError(f"Unknown multi-output workflow case '{case}'")
    result_path = save_result_artifact(result, results_dir, benchmark_name)
    return {"result_path": result_path}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
