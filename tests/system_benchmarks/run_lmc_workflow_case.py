"""Run one LMC workflow benchmark case in isolation."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from mojogp import Kernel, MultiOutputLMCGP
from tests.benchmarks.prediction_workload import (
    BENCHMARK_PATHWISE_SAMPLING_N_TEST,
    BENCHMARK_PREDICTION_N_TEST,
)
from tests.shared.subprocess_harness import IsolatedGPUTestSession, run_child_main

from tests.shared.benchmarking.data_generators import (
    generate_multi_output_heterogeneous_latent_data,
    generate_multi_output_structured_per_task_noise_data,
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


def _median_iter_time_ms(iter_times_ms) -> float | None:
    if iter_times_ms is None:
        return None
    values = np.asarray(iter_times_ms, dtype=np.float64)
    if values.size == 0:
        return None
    return float(np.median(values))


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


def _noise_rank_correlation(learned: np.ndarray, true: np.ndarray) -> float:
    learned_rank = np.argsort(np.argsort(np.asarray(learned, dtype=np.float32)))
    true_rank = np.argsort(np.argsort(np.asarray(true, dtype=np.float32)))
    if np.std(learned_rank) == 0.0 or np.std(true_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(learned_rank, true_rank)[0, 1])


def _make_gp() -> MultiOutputLMCGP:
    return MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=60,
        use_preconditioner=False,
    )


def _persistence_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    dataset = generate_multi_output_heterogeneous_latent_data(
        n_train=int(payload.get("n_train", 700)),
        n_test=int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST)),
        seed=int(payload.get("seed", 33)),
    )
    # In-process LMC compile probes can destabilize the later real fit, so keep
    # workflow startup timing to wrapper construction here.
    gp, startup_profile = measure_startup_profile(prepare_factory=_make_gp)

    fit_start = time.perf_counter()
    result = gp.fit(dataset.X_train, dataset.Y_train, max_iterations=30, learning_rate=0.03, verbose=False)
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred_before = gp.predict(dataset.X_test)
    persist_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mojogp-lmc-persist-") as tmpdir:
        model_path = Path(tmpdir) / "lmc_gp"
        gp.save(str(model_path))
        loaded = MultiOutputLMCGP.load(str(model_path))
        pred_after = loaded.predict(dataset.X_test)
    persistence_time_s = time.perf_counter() - persist_start
    persist_snapshot = session.snapshot_gpu()

    mean_diff = np.asarray(pred_before.mean - pred_after.mean, dtype=np.float32)
    var_diff = np.asarray(pred_before.variance - pred_after.variance, dtype=np.float32)
    max_abs_mean_diff = float(np.max(np.abs(mean_diff)))
    max_abs_var_diff = float(np.max(np.abs(var_diff)))
    if max_abs_mean_diff > 1e-5 or max_abs_var_diff > 1e-5:
        raise AssertionError(
            "LMC save/load round trip changed predictions: "
            f"max_abs_mean_diff={max_abs_mean_diff:.3e}, "
            f"max_abs_var_diff={max_abs_var_diff:.3e}"
        )
    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, persist_snapshot])
    median_iter_ms = _median_iter_time_ms(getattr(result, "iter_times_ms", None))

    return BenchmarkResult(
        config={
            "benchmark": "lmc_persistence_harness",
            "route_group": "multi_output_lmc",
            "framework": "mojogp",
            "model_type": "MultiOutputLMCGP",
            "kernel": "rbf_plus_matern52",
            "training_method": "materialized",
            "method": "materialized",
            "prediction_mode": "love",
            "workflow": "persistence",
            "n": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "num_tasks": int(dataset.Y_train.shape[1]),
            "max_abs_mean_diff": max_abs_mean_diff,
            "max_abs_var_diff": max_abs_var_diff,
            "variance_round_trip_policy": "hard_assertion",
            "iteration_timing_policy": "direct_backend_median",
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
            max_iterations=30,
            early_stopped=int(result.iterations) < 30,
            ms_per_iteration=median_iter_ms,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(result.lengthscales))),
            learned_noise=float(np.mean(np.asarray(result.noise_per_task))),
            learned_outputscale=float(np.mean(np.asarray(result.outputscales))),
            learned_mean=float(np.mean(np.asarray(getattr(result, "mean_per_task", np.zeros(dataset.Y_train.shape[1]))))),
            final_nll=float(result.final_nll),
        ),
    )


def _sampling_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    sampling_method = str(payload["sampling_method"])
    default_n_test = BENCHMARK_PATHWISE_SAMPLING_N_TEST if sampling_method == "pathwise" else 96
    dataset = generate_multi_output_heterogeneous_latent_data(
        n_train=int(payload.get("n_train", 700)),
        n_test=int(payload.get("n_test", default_n_test)),
        seed=int(payload.get("seed", 44)),
    )
    # In-process LMC compile probes can destabilize the later real fit, so keep
    # workflow startup timing to wrapper construction here.
    gp, startup_profile = measure_startup_profile(prepare_factory=_make_gp)

    fit_start = time.perf_counter()
    result = gp.fit(dataset.X_train, dataset.Y_train, max_iterations=30, learning_rate=0.03, verbose=False)
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred = gp.predict(dataset.X_test)
    sample_start = time.perf_counter()
    samples = gp.sample_posterior(
        dataset.X_test,
        n_samples=int(payload.get("n_samples", 24)),
        method=sampling_method,
        n_rff_features=int(payload.get("n_rff_features", 512)),
        rng=np.random.default_rng(int(payload.get("sample_seed", 13))),
    )
    sample_time_s = time.perf_counter() - sample_start
    sample_snapshot = session.snapshot_gpu()
    sample_mean = np.mean(samples, axis=0).astype(np.float32)
    sample_std = np.std(samples, axis=0).astype(np.float32)
    pred_std = np.sqrt(np.maximum(pred.variance, 1e-10)).astype(np.float32)

    mean_rmse = float(np.sqrt(np.mean((sample_mean - pred.mean) ** 2)))
    std_rmse = float(np.sqrt(np.mean((sample_std - pred_std) ** 2)))
    mean_threshold = float(payload.get("sample_mean_rmse_threshold", 1.0))
    std_threshold = float(payload.get("sample_std_rmse_threshold", 1.5))
    if not np.all(np.isfinite(samples)):
        raise AssertionError("LMC posterior samples contain NaN or Inf values")
    if mean_rmse > mean_threshold or std_rmse > std_threshold:
        raise AssertionError(
            "LMC sampling moments diverged from predictive moments: "
            f"sample_mean_rmse={mean_rmse:.3e} threshold={mean_threshold:.3e}, "
            f"sample_std_rmse={std_rmse:.3e} threshold={std_threshold:.3e}"
        )
    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, sample_snapshot])
    median_iter_ms = _median_iter_time_ms(getattr(result, "iter_times_ms", None))
    return BenchmarkResult(
        config={
            "benchmark": "lmc_sampling_harness",
            "route_group": "multi_output_lmc",
            "framework": "mojogp",
            "model_type": "MultiOutputLMCGP",
            "kernel": "rbf_plus_matern52",
            "training_method": "materialized",
            "method": "materialized",
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
            "sampling_consistency_policy": "hard_assertion",
            "iteration_timing_policy": "direct_backend_median",
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
            max_iterations=30,
            early_stopped=int(result.iterations) < 30,
            ms_per_iteration=median_iter_ms,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(result.lengthscales))),
            learned_noise=float(np.mean(np.asarray(result.noise_per_task))),
            learned_outputscale=float(np.mean(np.asarray(result.outputscales))),
            learned_mean=float(np.mean(np.asarray(getattr(result, "mean_per_task", np.zeros(dataset.Y_train.shape[1]))))),
            final_nll=float(result.final_nll),
        ),
    )


def _ard_relevance_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    seed = int(payload.get("seed", 52))
    n_train = int(payload.get("n_train", 2000))
    d = int(payload.get("d", 3))
    num_tasks = int(payload.get("num_tasks", 2))
    max_iterations = int(payload.get("max_iterations", 120))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_train, d)).astype(np.float32)
    latent = np.sin(2.0 * X[:, 0])
    Y = np.zeros((n_train, num_tasks), dtype=np.float32)
    Y[:, 0] = latent + 0.05 * rng.standard_normal(n_train)
    Y[:, 1] = 1.5 * latent + 0.05 * rng.standard_normal(n_train)

    method = str(payload.get("method", "matrix_free"))
    gp, startup_profile = measure_startup_profile(
        prepare_factory=lambda: MultiOutputLMCGP(
            kernels=[Kernel.rbf()],
            ard=True,
            num_probes=int(payload.get("num_probes", 16)),
            max_cg_iterations=int(payload.get("max_cg_iterations", 100)),
            cg_tolerance=float(payload.get("cg_tolerance", 1e-4)),
            preconditioner_rank=int(payload.get("preconditioner_rank", 20)),
        )
    )

    fit_start = time.perf_counter()
    result = gp.fit(
        X,
        Y,
        method=method,
        max_iterations=max_iterations,
        learning_rate=float(payload.get("learning_rate", 0.003)),
        initial_noise_per_task=np.full(num_tasks, 0.05, dtype=np.float32),
        early_stop_tol=0.0,
        verbose=False,
    )
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    lengthscales = result.lengthscales_per_dim
    if lengthscales is None:
        raise AssertionError("LMC ARD training did not return per-dimension lengthscales")
    avg_lengthscales = np.asarray(lengthscales, dtype=np.float32).mean(axis=0)
    if not np.all(np.isfinite(avg_lengthscales)) or not np.all(avg_lengthscales > 0.0):
        raise AssertionError(
            "LMC ARD training returned invalid per-dimension lengthscales: "
            f"avg_lengthscales={avg_lengthscales.tolist()}"
        )
    relevance_margin = float(np.min(avg_lengthscales[1:]) - avg_lengthscales[0])
    threshold = float(payload.get("relevance_margin_threshold", 0.15))
    if relevance_margin < threshold:
        raise AssertionError(
            "LMC ARD relevance check failed: "
            f"avg_lengthscales={avg_lengthscales.tolist()}, "
            f"margin={relevance_margin}, threshold={threshold}"
        )
    train_info = dict(gp.backend_train_info or {})
    if train_info.get("training_route") != method:
        raise AssertionError(
            f"Expected training_route={method}, got {train_info.get('training_route')}"
        )

    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot])
    median_iter_ms = _median_iter_time_ms(getattr(result, "iter_times_ms", None))
    return BenchmarkResult(
        config={
            "benchmark": "lmc_ard_relevance_harness",
            "route_group": "multi_output_lmc",
            "framework": "mojogp",
            "model_type": "MultiOutputLMCGP",
            "kernel": "rbf_ard",
            "training_method": method,
            "method": method,
            "prediction_mode": "none",
            "workflow": "ard_relevance",
            "n": n_train,
            "n_test": 0,
            "d": d,
            "num_tasks": num_tasks,
            "seed": seed,
            "avg_lengthscales": avg_lengthscales.tolist(),
            "lengthscales_finite": True,
            "lengthscales_positive": True,
            "relevance_margin": relevance_margin,
            "relevance_margin_threshold": threshold,
            "relevance_policy": "hard_assertion",
            "metadata_policy": "hard_assertion",
            "training_route": train_info.get("training_route"),
            "iteration_timing_policy": "direct_backend_median",
            "comparison_class": "mojogp_only",
            "baseline_backend": "none",
        },
        accuracy=AccuracyResult(
            rmse=max(0.0, threshold - relevance_margin),
            mae=abs(relevance_margin),
            r_squared=1.0,
            crps=0.0,
            msll=float(result.final_nll),
            calibration_coverage={},
            calibration_error=max(0.0, threshold - relevance_margin),
            sharpness=float(np.mean(avg_lengthscales)),
            interval_width_95=0.0,
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=0.0,
            prediction_variance_time_s=0.0,
            end_to_end_time_s=training_time_s,
            iterations_run=int(result.iterations),
            max_iterations=max_iterations,
            early_stopped=int(result.iterations) < max_iterations,
            ms_per_iteration=median_iter_ms,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get("startup_warm_cache_hit_s"),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(result.lengthscales))),
            learned_noise=float(np.mean(np.asarray(result.noise_per_task))),
            learned_outputscale=float(np.mean(np.asarray(result.outputscales))),
            learned_mean=float(np.mean(np.asarray(result.mean_per_task))),
            final_nll=float(result.final_nll),
        ),
    )


def _per_task_noise_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    method = str(payload.get("method", "materialized"))
    true_noise = np.asarray(
        payload.get("noise_per_task", [0.015, 0.06, 0.14]), dtype=np.float32
    )
    dataset = generate_multi_output_structured_per_task_noise_data(
        n_train=int(payload.get("n_train", 2000)),
        n_test=int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST)),
        d=int(payload.get("d", 5)),
        num_tasks=int(true_noise.shape[0]),
        noise_per_task=true_noise,
        task_correlation=str(payload.get("task_correlation", "medium")),
        seed=int(payload.get("seed", 73)),
    )
    np.random.seed(int(payload.get("init_seed", 42)))
    gp, startup_profile = measure_startup_profile(
        prepare_factory=lambda: MultiOutputLMCGP(
            kernels=[Kernel.rbf(), Kernel.matern52()],
            num_probes=int(payload.get("num_probes", 3)),
            max_cg_iterations=int(payload.get("max_cg_iterations", 30)),
            use_preconditioner=False,
        )
    )

    max_iterations = int(payload.get("max_iterations", 20))
    fit_start = time.perf_counter()
    result = gp.fit(
        dataset.X_train,
        dataset.Y_train,
        method=method,
        max_iterations=max_iterations,
        learning_rate=float(payload.get("learning_rate", 0.03)),
        initial_noise_per_task=np.full(true_noise.shape[0], 0.05, dtype=np.float32),
        verbose=False,
        early_stop_tol=0.0,
    )
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred_start = time.perf_counter()
    pred_mean, pred_var = gp.predict(
        dataset.X_test, return_var=True, variance_method="love"
    )
    prediction_time_s = time.perf_counter() - pred_start
    pred_snapshot = session.snapshot_gpu()

    mean = np.asarray(pred_mean, dtype=np.float32)
    variance = np.asarray(pred_var, dtype=np.float32)
    std = np.sqrt(np.maximum(variance, 1e-10))
    learned_noise = np.asarray(result.noise_per_task, dtype=np.float32)
    noise_rel_errors = [
        param_relative_error(float(learned_noise[t]), float(true_noise[t]))
        for t in range(true_noise.shape[0])
    ]
    timing_payload = _timing_payload(getattr(result, "iter_times_ms", None))
    iter_timing_quality = (
        "direct_per_iteration"
        if timing_payload["iter_times_ms"] is not None
        else "missing_per_iteration"
    )
    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, pred_snapshot])

    return BenchmarkResult(
        config={
            "benchmark": "lmc_per_task_noise",
            "route_group": "multi_output_lmc",
            "framework": "mojogp",
            "model_family": "MultiOutputLMCGP",
            "model_type": "MultiOutputLMCGP",
            "feature_surface": "lmc_per_task_noise",
            "feature_variant": f"structured_per_task_noise_{method}",
            "training_method": method,
            "method": method,
            "prediction_mode": "love",
            "workflow": "per_task_noise_recovery",
            "kernel": "rbf_plus_matern52",
            "n": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "num_tasks": int(dataset.Y_train.shape[1]),
            "task_correlation": str(dataset.true_params["task_correlation"]),
            "true_noise_per_task": true_noise,
            "learned_noise_per_task": learned_noise,
            "noise_rank_correlation": _noise_rank_correlation(learned_noise, true_noise),
            "noise_recovery_claim": "ordering_not_absolute_magnitude",
            "comparison_class": "mojogp_only",
            "baseline_backend": "none",
            "fairness_note": (
                "N.B. MojoGP-only LMC per-task-noise row: this benchmark checks "
                "noise ordering and predictive usability on synthetic structured data."
            ),
            "backend_train_info": gp.backend_train_info,
            "backend_predict_info": gp.backend_predict_info,
        },
        accuracy=AccuracyResult(
            rmse=float(rmse(dataset.F_test, mean)),
            mae=float(mae(dataset.F_test, mean)),
            r_squared=float(r_squared(dataset.F_test, mean)),
            crps=crps_gaussian(dataset.Y_test, mean, std),
            msll=mean_standardized_log_loss(
                dataset.Y_test,
                mean,
                std,
                y_train_mean=float(np.mean(dataset.Y_train)),
                y_train_std=float(np.std(dataset.Y_train)),
            ),
            calibration_coverage=calibration_coverage(dataset.Y_test, mean, std),
            calibration_error=calibration_error(dataset.Y_test, mean, std),
            sharpness=sharpness(std),
            interval_width_95=interval_width(mean, std),
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=prediction_time_s,
            prediction_variance_time_s=prediction_time_s,
            end_to_end_time_s=training_time_s + prediction_time_s,
            iterations_run=int(result.iterations),
            max_iterations=max_iterations,
            early_stopped=int(result.iterations) < max_iterations,
            ms_per_iteration=float(timing_payload["iter_time_median_ms"] or 0.0),
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
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get("startup_warm_cache_hit_s"),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(result.lengthscales))),
            learned_noise=float(np.mean(learned_noise)),
            learned_outputscale=float(np.mean(np.asarray(result.outputscales))),
            learned_mean=float(np.mean(np.asarray(result.mean_per_task))),
            final_nll=float(result.final_nll),
            noise_rel_error=float(np.mean(noise_rel_errors)),
        ),
    )


def _handle(payload: dict[str, object], session: IsolatedGPUTestSession) -> dict[str, object]:
    case = str(payload["case"])
    results_dir = Path(str(payload["results_dir"]))
    if case == "persistence":
        result = _persistence_case(payload, session)
        benchmark_name = "lmc_persistence_harness"
    elif case == "sampling":
        result = _sampling_case(payload, session)
        benchmark_name = "lmc_sampling_harness"
    elif case == "ard_relevance":
        result = _ard_relevance_case(payload, session)
        benchmark_name = "lmc_ard_relevance_harness"
    elif case == "per_task_noise":
        result = _per_task_noise_case(payload, session)
        benchmark_name = "lmc_per_task_noise"
    else:
        raise ValueError(f"Unknown LMC workflow case '{case}'")
    result_path = save_result_artifact(result, results_dir, benchmark_name)
    return {"result_path": result_path}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
