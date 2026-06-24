"""Run one matched-data single-output route-parity case."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mojogp import SingleOutputGP, RBF
from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.shared.benchmarking.data_generators import generate_structured_function_data
from tests.shared.benchmarking.gpu_memory import measure_gpu_phase
from tests.shared.benchmarking.gpytorch_models import (
    is_keops_available,
    merge_gpytorch_benchmark_memory,
    predict_gpytorch_single_output,
    train_gpytorch_single_output,
)
from tests.shared.benchmarking.metrics import (
    calibration_coverage,
    calibration_error,
    crps_gaussian,
    interval_width,
    mae,
    mean_standardized_log_loss,
    rmse,
    r_squared,
    sharpness,
)
from tests.shared.benchmarking.mojogp_runners import normalize_single_output_benchmark_hparams
from tests.shared.benchmarking.report import save_result_artifact
from tests.shared.benchmarking.result_types import AccuracyResult, BenchmarkResult, HyperparameterResult, MemoryResult, SpeedResult
from tests.shared.subprocess_harness import run_child_main
from tests.system_benchmarks.test_mojogp_route_parity_harness import PARITY_SOLVER


def _accuracy(dataset, mean: np.ndarray, variance: np.ndarray) -> AccuracyResult:
    std = np.sqrt(np.maximum(variance, 1e-10))
    return AccuracyResult(
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
    )


def _config(*, framework: str, method: str, prediction_mode: str, n: int, d: int, seed: int, effective_backend: str | None = None, fallback_reason: str | None = None) -> dict[str, object]:
    near_fair = framework == "gpytorch_keops"
    return {
        "benchmark": "matched_data_route_parity",
        "suite_name": "mojogp_route_parity",
        "framework": framework,
        "model_type": "SingleOutputGP",
        "kernel": "rbf",
        "training_method": method,
        "method": method,
        "prediction_mode": prediction_mode,
        "comparison_class": "fair_match",
        "baseline_backend": "keops" if framework == "gpytorch_keops" else "standard" if framework == "gpytorch" else "none",
        "keops_supported": is_keops_available(),
        "keops_used": effective_backend == "keops",
        "backend_fallback_used": bool(fallback_reason),
        "backend_fallback_reason": fallback_reason,
        "fairness_note": (
            "N.B. KeOps route uses GPyTorch's KeOps lazy backend; compare as near-fair if backend support changes solver internals."
            if near_fair
            else "Strict matched-data route parity row: same generated data, optimizer budget, and CG/SLQ solver budget."
        ),
        "fairness_axes": {
            "sample_count_n": {"status": "aligned", "note": "All routes reuse the same deterministic dataset key."},
            "optimizer": {"status": "aligned", "note": "Learning rate, schedule, and iteration budget are matched."},
            "solver_budget": {"status": "aligned", "note": "CG tolerance, max iterations, trace samples, and Lanczos budget are matched."},
            "preconditioner": {"status": "aligned", "note": "Preconditioning is disabled for fair route comparison."},
            "prediction_mode": {"status": "aligned", "note": "Rows are compared within the same exact/LOVE prediction mode."},
            "backend": {
                "status": "near_fair" if near_fair else "aligned",
                "note": "KeOps changes the GPyTorch kernel evaluation backend." if near_fair else "Backend route matches the row label.",
            },
        },
        "n": n,
        "d": d,
        "n_test": BENCHMARK_PREDICTION_N_TEST,
        "seed": seed,
        "optimizer_config": {"max_iterations": 30, "learning_rate": 0.03, "lr_schedule": "cosine"},
        "training_solver_config": {
            "cg_tolerance": PARITY_SOLVER["cg_tolerance"],
            "max_cg_iterations": PARITY_SOLVER["max_cg_iterations"],
            "num_trace_samples": PARITY_SOLVER["num_trace_samples"],
            "max_tridiag_iter": PARITY_SOLVER["max_lanczos_quadrature_iterations"],
            "precond_rank": PARITY_SOLVER["precond_rank"],
        },
        "prediction_solver_config": {
            "cg_tolerance": PARITY_SOLVER["cg_tolerance"],
            "max_cg_iterations": PARITY_SOLVER["max_cg_iterations"],
            "max_root_decomposition_size": 20,
        },
        "preconditioner_config": {"family": "disabled", "method": 0, "rank": 0},
    }


def _memory_from_stats(stats: dict[str, object]) -> MemoryResult:
    return MemoryResult(
        gpu_mean_mb=float(stats.get("mean_mb", stats.get("max_mb", 0.0))),
        gpu_min_mb=float(stats.get("min_mb", 0.0)),
        gpu_max_mb=float(stats.get("max_mb", 0.0)),
        gpu_var_mb=float(stats.get("var_mb", 0.0)),
        torch_peak_mb=float(stats.get("torch_peak_mb", 0.0)),
        torch_current_mb=float(stats.get("torch_current_mb", 0.0)),
        cpu_peak_mb=float(stats.get("cpu_peak_mb", 0.0)),
        measurement_method=str(stats.get("method", "none")),
        num_samples=int(stats.get("samples", 0)),
        training_peak_gpu_mb=stats.get("training_peak_gpu_mb"),
        training_delta_gpu_mb=stats.get("training_delta_gpu_mb"),
        prediction_peak_gpu_mb=stats.get("prediction_peak_gpu_mb"),
        prediction_delta_gpu_mb=stats.get("prediction_delta_gpu_mb"),
        exact_prediction_peak_gpu_mb=stats.get("exact_prediction_peak_gpu_mb"),
        exact_prediction_delta_gpu_mb=stats.get("exact_prediction_delta_gpu_mb"),
        love_prediction_peak_gpu_mb=stats.get("love_prediction_peak_gpu_mb"),
        love_prediction_delta_gpu_mb=stats.get("love_prediction_delta_gpu_mb"),
    )


def _run_mojogp(dataset, *, method: str, prediction_mode: str, n: int, d: int, seed: int) -> BenchmarkResult:
    model = SingleOutputGP(RBF(), verbose=False)
    timing: dict[str, float] = {}

    def _fit():
        start = time.perf_counter()
        result = model.fit(
            dataset.X_train,
            dataset.y_train,
            method=method,
            max_iterations=30,
            learning_rate=0.03,
            lr_schedule="cosine",
            enable_early_stopping=False,
            max_cg_iterations=PARITY_SOLVER["max_cg_iterations"],
            cg_tolerance=PARITY_SOLVER["cg_tolerance"],
            num_probes=PARITY_SOLVER["num_trace_samples"],
            max_tridiag_iterations=PARITY_SOLVER["max_lanczos_quadrature_iterations"],
            preconditioner_rank=PARITY_SOLVER["precond_rank"],
            preconditioner=PARITY_SOLVER["precond"],
            use_preconditioner=PARITY_SOLVER["use_preconditioner"],
            verbose=False,
        )
        timing["training_time_s"] = time.perf_counter() - start
        return result

    train_result, train_mem = measure_gpu_phase(_fit, interval=0.02)

    def _predict():
        start = time.perf_counter()
        pred = model.predict(
            dataset.X_test,
                        variance_method=prediction_mode,
            max_cg_iterations=PARITY_SOLVER["max_cg_iterations"],
            cg_tolerance=PARITY_SOLVER["cg_tolerance"],
            max_root_decomposition_size=20,
        )
        timing["prediction_time_s"] = time.perf_counter() - start
        return pred

    pred, pred_mem = measure_gpu_phase(_predict, interval=0.02)
    pred_mean = np.asarray(pred.mean, dtype=np.float32)
    pred_variance = np.asarray(pred.variance, dtype=np.float32)
    memory_stats = {
        "max_mb": max(float(train_mem.get("phase_peak_gpu_mb", 0.0)), float(pred_mem.get("phase_peak_gpu_mb", 0.0))),
        "mean_mb": max(float(train_mem.get("phase_peak_gpu_mb", 0.0)), float(pred_mem.get("phase_peak_gpu_mb", 0.0))),
        "min_mb": min(float(train_mem.get("phase_baseline_gpu_mb", 0.0)), float(pred_mem.get("phase_baseline_gpu_mb", 0.0))),
        "var_mb": 0.0,
        "torch_peak_mb": max(float(train_mem.get("torch_peak_mb", 0.0)), float(pred_mem.get("torch_peak_mb", 0.0))),
        "torch_current_mb": float(pred_mem.get("torch_current_mb", 0.0)),
        "method": pred_mem.get("method", train_mem.get("method", "none")),
        "samples": int(train_mem.get("samples", 0)) + int(pred_mem.get("samples", 0)),
        "training_peak_gpu_mb": float(train_mem.get("phase_peak_gpu_mb", 0.0)),
        "training_delta_gpu_mb": float(train_mem.get("phase_delta_gpu_mb", 0.0)),
        "prediction_peak_gpu_mb": float(pred_mem.get("phase_peak_gpu_mb", 0.0)),
        "prediction_delta_gpu_mb": float(pred_mem.get("phase_delta_gpu_mb", 0.0)),
    }
    if prediction_mode == "love":
        memory_stats["love_prediction_peak_gpu_mb"] = memory_stats["prediction_peak_gpu_mb"]
        memory_stats["love_prediction_delta_gpu_mb"] = memory_stats["prediction_delta_gpu_mb"]
    else:
        memory_stats["exact_prediction_peak_gpu_mb"] = memory_stats["prediction_peak_gpu_mb"]
        memory_stats["exact_prediction_delta_gpu_mb"] = memory_stats["prediction_delta_gpu_mb"]
    params = normalize_single_output_benchmark_hparams(model.get_learned_params())
    config = _config(framework="mojogp", method=method, prediction_mode=prediction_mode, n=n, d=d, seed=seed)
    config.update(
        {
            "prediction_timing_quality": "total_only_combined_call",
            "iter_timing_quality": "derived_total_div_iterations",
            "backend_predict_info": dict(getattr(model, "backend_predict_info", {}) or {}),
        }
    )
    return BenchmarkResult(
        config=config,
        accuracy=_accuracy(dataset, pred_mean, pred_variance),
        speed=SpeedResult(
            training_time_s=timing["training_time_s"],
            prediction_mean_time_s=0.0,
            prediction_variance_time_s=timing["prediction_time_s"],
            end_to_end_time_s=timing["training_time_s"] + timing["prediction_time_s"],
            iterations_run=int(train_result.iterations),
            max_iterations=30,
            early_stopped=False,
            ms_per_iteration=(timing["training_time_s"] / max(int(train_result.iterations), 1)) * 1000.0,
            iter_timing_quality="derived_total_div_iterations",
        ),
        memory=_memory_from_stats(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(params["lengthscale"]),
            learned_noise=float(model.get_learned_params().get("noise", 0.1)),
            learned_outputscale=float(params["outputscale"]),
            final_nll=float(train_result.nll),
        ),
    )


def _run_gpytorch(dataset, *, framework: str, method: str, prediction_mode: str, n: int, d: int, seed: int) -> BenchmarkResult:
    mode = "keops" if framework == "gpytorch_keops" else "cg"
    train_result = train_gpytorch_single_output(
        dataset.X_train,
        dataset.y_train,
        kernel_type="rbf",
        mode=mode,
        n_iterations=30,
        lr=0.03,
        lr_schedule="cosine",
        early_stop_patience=10_000,
        cg_tolerance=PARITY_SOLVER["cg_tolerance"],
        max_cg_iterations=PARITY_SOLVER["max_cg_iterations"],
        num_trace_samples=PARITY_SOLVER["num_trace_samples"],
        max_preconditioner_size=0,
        max_lanczos_quadrature_iterations=PARITY_SOLVER["max_lanczos_quadrature_iterations"],
        min_preconditioning_size=0,
        memory_poll_interval=0.02,
        device="cuda",
    )
    pred_result = predict_gpytorch_single_output(
        train_result,
        dataset.X_test,
        mode=mode,
        cg_tolerance=PARITY_SOLVER["cg_tolerance"],
        max_cg_iterations=PARITY_SOLVER["max_cg_iterations"],
        max_preconditioner_size=0,
        max_lanczos_quadrature_iterations=PARITY_SOLVER["max_lanczos_quadrature_iterations"],
        min_preconditioning_size=0,
        max_root_decomposition_size=20,
        use_love=prediction_mode == "love",
    )
    memory_stats = merge_gpytorch_benchmark_memory(dict(train_result.get("memory_stats", {})), dict(pred_result.get("memory_stats", {})))
    phase_memory_quality = "phase_specific"
    if memory_stats.get("prediction_peak_gpu_mb") is None:
        phase_memory_quality = "prediction_phase_missing_fallback_to_process_peak"
        fallback_peak = float(memory_stats.get("max_mb", 0.0))
        memory_stats["prediction_peak_gpu_mb"] = fallback_peak
        memory_stats["prediction_delta_gpu_mb"] = fallback_peak
        if prediction_mode == "love":
            memory_stats["love_prediction_peak_gpu_mb"] = fallback_peak
            memory_stats["love_prediction_delta_gpu_mb"] = fallback_peak
        else:
            memory_stats["exact_prediction_peak_gpu_mb"] = fallback_peak
            memory_stats["exact_prediction_delta_gpu_mb"] = fallback_peak
    learned = train_result["learned_params"]
    config = _config(
        framework=framework,
        method=method,
        prediction_mode=prediction_mode,
        n=n,
        d=d,
        seed=seed,
        effective_backend=str(train_result.get("effective_mode", mode)),
        fallback_reason=train_result.get("backend_fallback_reason") or pred_result.get("backend_fallback_reason"),
    )
    config.update(
        {
            "prediction_timing_quality": "observed_mean_variance_split",
            "iter_timing_quality": "direct_per_iteration" if train_result.get("iter_times_ms") else "derived_total_div_iterations",
            "phase_memory_quality": phase_memory_quality,
            "cg_telemetry": {"training": train_result.get("cg_telemetry", {}), "prediction": pred_result.get("cg_telemetry", {})},
        }
    )
    return BenchmarkResult(
        config=config,
        accuracy=_accuracy(dataset, np.asarray(pred_result["mean"], dtype=np.float32), np.asarray(pred_result["variance"], dtype=np.float32)),
        speed=SpeedResult(
            training_time_s=float(train_result["training_time_s"]),
            prediction_mean_time_s=float(pred_result.get("mean_time_s", 0.0)),
            prediction_variance_time_s=float(pred_result.get("variance_time_s", 0.0)),
            end_to_end_time_s=float(train_result["training_time_s"]) + float(pred_result.get("total_time_s", 0.0)),
            iterations_run=int(train_result["iterations_run"]),
            max_iterations=int(train_result["max_iterations"]),
            early_stopped=bool(train_result["early_stopped"]),
            ms_per_iteration=float(train_result.get("iter_time_median_ms") or (float(train_result["training_time_s"]) / max(int(train_result["iterations_run"]), 1)) * 1000.0),
            iter_time_p5_ms=train_result.get("iter_time_p5_ms"),
            iter_time_p95_ms=train_result.get("iter_time_p95_ms"),
            iter_times_ms=train_result.get("iter_times_ms"),
            iter_timing_quality="direct_per_iteration" if train_result.get("iter_times_ms") else "derived_total_div_iterations",
        ),
        memory=_memory_from_stats(memory_stats),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(learned.get("lengthscale", 1.0)),
            learned_noise=float(learned.get("noise", 0.1)),
            learned_outputscale=float(learned.get("outputscale", 1.0)),
            final_nll=float(train_result["final_nll"]),
        ),
    )


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    framework = str(payload["framework"])
    method = str(payload["method"])
    prediction_mode = str(payload["prediction_mode"])
    n = int(payload.get("n", 5000))
    d = int(payload.get("d", 17))
    seed = int(payload.get("seed", 9000 + n + d * 1000))
    dataset = generate_structured_function_data(
        n_train=n,
        n_test=int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST)),
        d=d,
        function_type="smooth",
        noise_level="medium",
        seed=seed,
    )
    if framework == "mojogp":
        result = _run_mojogp(dataset, method=method, prediction_mode=prediction_mode, n=n, d=d, seed=seed)
    elif framework in {"gpytorch", "gpytorch_keops"}:
        result = _run_gpytorch(dataset, framework=framework, method=method, prediction_mode=prediction_mode, n=n, d=d, seed=seed)
    else:
        raise ValueError(f"Unknown framework: {framework}")
    result_path = save_result_artifact(result, Path(str(payload["results_dir"])), "matched_data_route_parity")
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
