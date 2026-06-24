"""Run one single-output workflow benchmark case in isolation."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from mojogp import SingleOutputGP, Kernel
from tests.benchmarks.prediction_workload import (
    BENCHMARK_PATHWISE_SAMPLING_N_TEST,
    BENCHMARK_PREDICTION_N_TEST,
)
from tests.shared.subprocess_harness import IsolatedGPUTestSession, run_child_main

from tests.shared.benchmarking.data_generators import generate_gp_prior_data
from tests.shared.benchmarking.report import save_result_artifact
from tests.shared.benchmarking.result_types import AccuracyResult, BenchmarkResult, HyperparameterResult, MemoryResult, SpeedResult
from tests.shared.benchmarking.startup_timing import measure_startup_profile


def _make_gp(method: str) -> SingleOutputGP:
    return SingleOutputGP(Kernel.rbf())


def _probe_single_output_compile(method: str, d: int) -> None:
    gp = _make_gp(method)
    gp.dim = d
    gp._training_method = method
    gp._ensure_compiled()


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


def _dense_single_output_posterior_reference(
    gp: SingleOutputGP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tr = gp._training_result
    params = np.asarray(tr.params, dtype=np.float32)
    noise = float(tr.noise)
    mean = float(tr.mean)

    K_train = gp.kernel.evaluate(X_train, X_train, params=params).astype(np.float64)
    K_cross = gp.kernel.evaluate(X_eval, X_train, params=params).astype(np.float64)
    K_eval = gp.kernel.evaluate(X_eval, X_eval, params=params).astype(np.float64)
    K_reg = K_train + noise * np.eye(X_train.shape[0], dtype=np.float64)

    y_centered = y_train.astype(np.float64) - mean
    alpha = np.linalg.solve(K_reg, y_centered)
    solves = np.linalg.solve(K_reg, K_cross.T)
    ref_mean = K_cross @ alpha + mean
    ref_cov = K_eval - K_cross @ solves
    ref_cov = 0.5 * (ref_cov + ref_cov.T)
    return ref_mean.astype(np.float32), ref_cov.astype(np.float32)


def _correlation_matrix(cov: np.ndarray) -> np.ndarray:
    diag = np.maximum(np.diag(cov), 1e-12)
    return cov / np.sqrt(np.outer(diag, diag))


def _pathwise_covariance_metrics(
    samples: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov: np.ndarray,
) -> dict[str, float]:
    empirical_mean = np.mean(samples, axis=0).astype(np.float32)
    empirical_cov = np.cov(samples, rowvar=False).astype(np.float32)
    ref_var = np.maximum(np.diag(ref_cov), 1e-12)
    empirical_var = np.maximum(np.diag(empirical_cov), 1e-12)
    ref_corr = _correlation_matrix(ref_cov)
    empirical_corr = _correlation_matrix(empirical_cov)
    upper = np.triu_indices_from(ref_corr, k=1)
    close_pairs = [(0, 1), (1, 2), (3, 4), (4, 5)]
    far_pairs = [(0, 3), (1, 4), (2, 5)]
    return {
        "mean_rmse": float(np.sqrt(np.mean((empirical_mean - ref_mean) ** 2))),
        "mean_mae": float(np.mean(np.abs(empirical_mean - ref_mean))),
        "var_rel_rmse": float(
            np.sqrt(np.mean((empirical_var - ref_var) ** 2))
            / (np.mean(ref_var) + 1e-12)
        ),
        "cov_rel_frobenius": float(
            np.linalg.norm(empirical_cov - ref_cov) / (np.linalg.norm(ref_cov) + 1e-12)
        ),
        "corr_rmse": float(
            np.sqrt(np.mean((empirical_corr[upper] - ref_corr[upper]) ** 2))
        ),
        "ref_close_corr": float(np.mean([ref_corr[i, j] for i, j in close_pairs])),
        "empirical_close_corr": float(
            np.mean([empirical_corr[i, j] for i, j in close_pairs])
        ),
        "ref_far_corr": float(np.mean([ref_corr[i, j] for i, j in far_pairs])),
        "empirical_far_corr": float(
            np.mean([empirical_corr[i, j] for i, j in far_pairs])
        ),
        "empirical_sharpness": float(np.mean(np.sqrt(empirical_var))),
    }


def _single_output_persistence_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    method = str(payload["training_method"])
    dataset = generate_gp_prior_data(
        n_train=int(payload.get("n_train", 2000)),
        n_test=int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST)),
        d=int(payload.get("d", 5)),
        kernel_type="rbf",
        seed=int(payload.get("seed", 42)),
    )
    gp, startup_profile = measure_startup_profile(
        prepare_factory=lambda: _make_gp(method),
        cold_start_probe=lambda: _probe_single_output_compile(
            method,
            int(dataset.X_train.shape[1]),
        ),
        cache_prefix="mojogp_single_output_workflow_",
    )

    fit_start = time.perf_counter()
    gp.fit(
        dataset.X_train,
        dataset.y_train,
        method=method,
        max_iterations=40,
        learning_rate=0.03,
        use_preconditioner=False,
        verbose=False,
    )
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred_before = gp.predict(dataset.X_test, variance_method="exact")
    persist_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mojogp-single-persist-") as tmpdir:
        model_path = Path(tmpdir) / "exactgp_model"
        gp.save(str(model_path))
        loaded = SingleOutputGP.load(str(model_path))
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
            "Single-output save/load prediction round trip changed outputs: "
            f"max_abs_mean_diff={max_abs_mean_diff:.3e}, "
            f"max_abs_var_diff={max_abs_var_diff:.3e}"
        )
    learned = gp.get_learned_params()
    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, persist_snapshot])

    return BenchmarkResult(
        config={
            "benchmark": "single_output_persistence_harness",
            "route_group": "single_output",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": "exact",
            "workflow": "persistence",
            "n": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "d": int(dataset.X_train.shape[1]),
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
            iterations_run=40,
            max_iterations=40,
            early_stopped=False,
            ms_per_iteration=training_time_s / 40.0 * 1000.0,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(learned.get("lengthscale", 1.0)),
            learned_noise=float(learned.get("noise", 0.1)),
            learned_outputscale=float(learned.get("outputscale", 1.0)),
            learned_mean=float(learned.get("mean", 0.0)),
            final_nll=-float(gp.log_marginal_likelihood()),
        ),
    )


def _single_output_sampling_case(payload: dict[str, object], session: IsolatedGPUTestSession) -> BenchmarkResult:
    method = str(payload["training_method"])
    sampling_method = str(payload["sampling_method"])
    default_n_test = BENCHMARK_PATHWISE_SAMPLING_N_TEST if sampling_method == "pathwise" else 96
    dataset = generate_gp_prior_data(
        n_train=int(payload.get("n_train", 2000)),
        n_test=int(payload.get("n_test", default_n_test)),
        d=int(payload.get("d", 5)),
        kernel_type="rbf",
        seed=int(payload.get("seed", 123)),
    )
    gp, startup_profile = measure_startup_profile(
        prepare_factory=lambda: _make_gp(method),
        cold_start_probe=lambda: _probe_single_output_compile(
            method,
            int(dataset.X_train.shape[1]),
        ),
        cache_prefix="mojogp_single_output_workflow_",
    )

    fit_start = time.perf_counter()
    gp.fit(
        dataset.X_train,
        dataset.y_train,
        method=method,
        max_iterations=40,
        learning_rate=0.03,
        use_preconditioner=False,
        verbose=False,
    )
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred = gp.predict(dataset.X_test)
    sample_start = time.perf_counter()
    samples = gp.sample_posterior(
        dataset.X_test,
        n_samples=int(payload.get("n_samples", 32)),
        method=sampling_method,
        n_rff_features=int(payload.get("n_rff_features", 512)),
        rng=np.random.default_rng(int(payload.get("sample_seed", 7))),
    )
    sample_time_s = time.perf_counter() - sample_start
    sample_snapshot = session.snapshot_gpu()

    sample_mean = np.mean(samples, axis=0).astype(np.float32)
    sample_std = np.std(samples, axis=0).astype(np.float32)
    mean_rmse = float(np.sqrt(np.mean((sample_mean - pred.mean) ** 2)))
    std_rmse = float(np.sqrt(np.mean((sample_std - pred.std) ** 2)))
    mean_threshold = float(payload.get("sample_mean_rmse_threshold", 1.0))
    std_threshold = float(payload.get("sample_std_rmse_threshold", 1.5))
    sample_info = dict(gp.backend_sample_info or {})
    expected_route = (
        "provider_pathwise"
        if sampling_method == "pathwise"
        else "diagonal_from_predictive_std"
    )
    actual_route = str(sample_info.get("actual_sampling_route", "missing"))
    if actual_route != expected_route:
        raise AssertionError(
            f"Single-output sampling used route {actual_route!r}, expected {expected_route!r}"
        )
    if not np.all(np.isfinite(samples)):
        raise AssertionError("Single-output posterior samples contain NaN or Inf values")
    if mean_rmse > mean_threshold or std_rmse > std_threshold:
        raise AssertionError(
            "Single-output sampling moments diverged from predictive moments: "
            f"sample_mean_rmse={mean_rmse:.3e} threshold={mean_threshold:.3e}, "
            f"sample_std_rmse={std_rmse:.3e} threshold={std_threshold:.3e}"
        )
    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, sample_snapshot])
    learned = gp.get_learned_params()

    return BenchmarkResult(
        config={
            "benchmark": "single_output_sampling_harness",
            "route_group": "single_output",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": sampling_method,
            "workflow": "sampling",
            "sampling_method": sampling_method,
            "n": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "d": int(dataset.X_train.shape[1]),
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
            iterations_run=40,
            max_iterations=40,
            early_stopped=False,
            ms_per_iteration=training_time_s / 40.0 * 1000.0,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(learned.get("lengthscale", 1.0)),
            learned_noise=float(learned.get("noise", 0.1)),
            learned_outputscale=float(learned.get("outputscale", 1.0)),
            learned_mean=float(learned.get("mean", 0.0)),
            final_nll=-float(gp.log_marginal_likelihood()),
        ),
    )


def _single_output_pathwise_covariance_sanity_case(
    payload: dict[str, object], session: IsolatedGPUTestSession
) -> BenchmarkResult:
    method = str(payload.get("training_method", "materialized"))
    seed = int(payload.get("seed", 8128))
    n_train = int(payload.get("n_train", 2000))
    max_iterations = int(payload.get("max_iterations", 20))
    rng = np.random.default_rng(seed)
    X_train = np.linspace(-3.0, 3.0, n_train, dtype=np.float32).reshape(-1, 1)
    y_train = (
        np.sin(1.25 * X_train[:, 0])
        + 0.2 * np.cos(2.4 * X_train[:, 0])
        + 0.08 * rng.standard_normal(n_train)
    ).astype(np.float32)
    X_test = np.array(
        [[-4.0], [-3.9], [-3.8], [3.8], [3.9], [4.0]], dtype=np.float32
    )

    gp, startup_profile = measure_startup_profile(
        prepare_factory=lambda: _make_gp(method),
        cold_start_probe=lambda: _probe_single_output_compile(method, 1),
        cache_prefix="mojogp_single_output_pathwise_covariance_",
    )

    fit_start = time.perf_counter()
    gp.fit(
        X_train,
        y_train,
        method=method,
        max_iterations=max_iterations,
        learning_rate=0.03,
        use_preconditioner=False,
        verbose=False,
    )
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    ref_mean, ref_cov = _dense_single_output_posterior_reference(
        gp, X_train, y_train, X_test
    )
    sample_start = time.perf_counter()
    samples = gp.sample_posterior(
        X_test,
        n_samples=int(payload.get("n_samples", 64)),
        method="pathwise",
        n_rff_features=int(payload.get("n_rff_features", 2048)),
        rng=np.random.default_rng(int(payload.get("sample_seed", 91))),
    )
    sample_time_s = time.perf_counter() - sample_start
    sample_snapshot = session.snapshot_gpu()
    sample_info = dict(gp.backend_sample_info or {})
    actual_route = str(sample_info.get("actual_sampling_route", "missing"))
    if actual_route != "provider_pathwise":
        raise AssertionError(
            f"Pathwise covariance sanity used route {actual_route!r}, expected 'provider_pathwise'"
        )
    if not np.all(np.isfinite(samples)):
        raise AssertionError("Pathwise covariance sanity samples contain NaN or Inf values")

    metrics = _pathwise_covariance_metrics(samples, ref_mean, ref_cov)
    mean_threshold = float(payload.get("mean_rmse_threshold", 0.35))
    var_threshold = float(payload.get("var_rel_rmse_threshold", 0.90))
    corr_threshold = float(payload.get("corr_rmse_threshold", 0.70))
    close_gap_threshold = float(payload.get("close_far_corr_gap_threshold", 0.15))
    min_close_corr = float(payload.get("min_empirical_close_corr", 0.30))
    close_far_gap = metrics["empirical_close_corr"] - metrics["empirical_far_corr"]
    if (
        metrics["mean_rmse"] > mean_threshold
        or metrics["var_rel_rmse"] > var_threshold
        or metrics["corr_rmse"] > corr_threshold
        or metrics["empirical_close_corr"] < min_close_corr
        or close_far_gap < close_gap_threshold
    ):
        raise AssertionError(
            "Pathwise empirical covariance diverged from dense posterior reference: "
            f"mean_rmse={metrics['mean_rmse']:.3e}/{mean_threshold:.3e}, "
            f"var_rel_rmse={metrics['var_rel_rmse']:.3e}/{var_threshold:.3e}, "
            f"corr_rmse={metrics['corr_rmse']:.3e}/{corr_threshold:.3e}, "
            f"empirical_close_corr={metrics['empirical_close_corr']:.3e}, "
            f"empirical_far_corr={metrics['empirical_far_corr']:.3e}, "
            f"close_far_gap={close_far_gap:.3e}/{close_gap_threshold:.3e}"
        )

    memory_stats = session.collect_memory_stats(snapshots=[fit_snapshot, sample_snapshot])
    learned = gp.get_learned_params()

    return BenchmarkResult(
        config={
            "benchmark": "single_output_pathwise_covariance_sanity",
            "route_group": "single_output",
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": method,
            "method": method,
            "prediction_mode": "pathwise",
            "workflow": "pathwise_covariance_sanity",
            "sampling_method": "pathwise",
            "n": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "d": int(X_train.shape[1]),
            "n_samples": int(samples.shape[0]),
            "n_rff_features": int(payload.get("n_rff_features", 2048)),
            "actual_sampling_route": actual_route,
            "sample_mean_rmse": metrics["mean_rmse"],
            "sample_mean_rmse_threshold": mean_threshold,
            "sample_var_rel_rmse": metrics["var_rel_rmse"],
            "sample_var_rel_rmse_threshold": var_threshold,
            "sample_corr_rmse": metrics["corr_rmse"],
            "sample_corr_rmse_threshold": corr_threshold,
            "sample_cov_rel_frobenius": metrics["cov_rel_frobenius"],
            "ref_close_corr": metrics["ref_close_corr"],
            "empirical_close_corr": metrics["empirical_close_corr"],
            "ref_far_corr": metrics["ref_far_corr"],
            "empirical_far_corr": metrics["empirical_far_corr"],
            "close_far_corr_gap": close_far_gap,
            "close_far_corr_gap_threshold": close_gap_threshold,
            "min_empirical_close_corr": min_close_corr,
            "sampling_consistency_policy": "dense_covariance_hard_assertion",
            "comparison_class": "dense_exact_reference",
            "baseline_backend": "numpy_dense_exact",
        },
        accuracy=AccuracyResult(
            rmse=metrics["mean_rmse"],
            mae=metrics["mean_mae"],
            r_squared=1.0,
            crps=metrics["var_rel_rmse"],
            msll=metrics["cov_rel_frobenius"],
            calibration_coverage={},
            calibration_error=metrics["corr_rmse"],
            sharpness=metrics["empirical_sharpness"],
            interval_width_95=3.92 * metrics["empirical_sharpness"],
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=sample_time_s,
            prediction_variance_time_s=sample_time_s,
            end_to_end_time_s=training_time_s + sample_time_s,
            iterations_run=max_iterations,
            max_iterations=max_iterations,
            early_stopped=False,
            ms_per_iteration=training_time_s / max(max_iterations, 1) * 1000.0,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
        ),
        memory=_memory_result(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(learned.get("lengthscale", 1.0)),
            learned_noise=float(learned.get("noise", 0.1)),
            learned_outputscale=float(learned.get("outputscale", 1.0)),
            learned_mean=float(learned.get("mean", 0.0)),
            final_nll=-float(gp.log_marginal_likelihood()),
        ),
    )


def _handle(payload: dict[str, object], session: IsolatedGPUTestSession) -> dict[str, object]:
    case = str(payload["case"])
    results_dir = Path(str(payload["results_dir"]))
    if case == "persistence":
        result = _single_output_persistence_case(payload, session)
        benchmark_name = "single_output_persistence_harness"
    elif case == "sampling":
        result = _single_output_sampling_case(payload, session)
        benchmark_name = "single_output_sampling_harness"
    elif case == "pathwise_covariance_sanity":
        result = _single_output_pathwise_covariance_sanity_case(payload, session)
        benchmark_name = "single_output_pathwise_covariance_sanity"
    else:
        raise ValueError(f"Unknown single-output workflow case '{case}'")
    result_path = save_result_artifact(result, results_dir, benchmark_name)
    return {"result_path": result_path}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
