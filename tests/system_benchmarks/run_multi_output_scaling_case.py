"""Run a single large-scale multi-output scaling case in isolation."""

from __future__ import annotations

import time
import os
from pathlib import Path

import numpy as np
import torch

from mojogp import MultiOutputGP
from tests.benchmarks.dataset_manifest import load_dataset_artifact
from tests.benchmarks.prediction_workload import (
    BENCHMARK_PREDICTION_N_TEST,
    clear_prediction_x_test_failure_memory,
    prediction_x_test_repeat_count,
    prediction_x_test_scaling_entry,
    prediction_x_test_scaling_failure_entry,
    prediction_x_test_should_record_failure,
    prediction_x_test_target_specs,
)
from tests.shared.subprocess_harness import IsolatedGPUTestSession, run_child_main
from tests.shared.benchmarking.gpu_memory import get_torch_memory_stats, measure_gpu_phase

from .conftest import get_vram_info
from tests.shared.benchmarking.ard_metrics import compute_ard_relevance_metrics
from tests.shared.benchmarking.data_generators import (
    MultiOutputDataset,
    generate_multi_output_data,
    generate_multi_output_structured_ard_data,
)
from tests.shared.benchmarking.gpytorch_models import (
    merge_gpytorch_benchmark_memory,
    predict_gpytorch_multi_output,
    train_gpytorch_multi_output,
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
from tests.shared.benchmarking.report import save_result_artifact
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)


FAIR_MULTI_OUTPUT_SOLVER = {
    "cg_tolerance": 1e-2,
    "max_cg_iterations": 100,
    "num_trace_samples": 10,
    "max_lanczos_quadrature_iterations": 20,
    "max_root_decomposition_size": 20,
    "max_preconditioner_size": 0,
    "min_preconditioning_size": 0,
    "precond_rank": 0,
    "precond_method": 0,
    "precond": "auto",
    "use_preconditioner": False,
}


GPYTORCH_SIZE_CAPS = {
    "materialized": {
        "xsmall": [1500],
        "small": [3000, 5000],
        "medium": [5000, 8000],
        "large": [2000],
        "xlarge": [20000],
    },
    "matrix_free": {
        "xsmall": [10000, 25000],
        "small": [25000, 50000],
        "medium": [50000, 75000],
        "large": [50000, 100000],
        "xlarge": [100000, 150000],
    },
}


MULTI_OUTPUT_ARD_MATERIALIZED_SHARED_SIZES = {
    "xsmall": [2000, 3000],
    "small": [3000, 5000],
    "medium": [5000, 8000],
    "large": [3000],
    "xlarge": [12000, 20000],
}


MATRIX_FREE_EXACT_SIZE_CAPS = {
    "mojogp": {
        "xsmall": [10000],
        "small": [25000],
        "medium": [50000],
        "large": [50000],
        "xlarge": [100000],
    },
    "gpytorch": {
        "xsmall": [1500],
        "small": [25000],
        "medium": [50000],
        "large": [50000],
        "xlarge": [100000],
    },
}


def _quick_enabled() -> bool:
    return os.environ.get("MOJOGP_BENCHMARK_QUICK", "0") == "1"


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _framework_sizes(
    *, framework: str, method: str, prediction_mode: str, tier: str
) -> list[int]:
    if framework == "gpytorch":
        return list(GPYTORCH_SIZE_CAPS[method][tier])

    benchmark_targets = {
        "xsmall": {
            "materialized": [1500, 3000],
            "matrix_free": [5000, 10000],
        },
        "small": {
            "materialized": [3000, 5000],
            "matrix_free": [25000, 50000],
        },
        "medium": {
            "materialized": [5000, 8000],
            "matrix_free": [50000, 75000],
        },
        "large": {
            "materialized": [2000, 12000],
            "matrix_free": [50000, 100000],
        },
        "xlarge": {
            "materialized": [20000, 30000],
            "matrix_free": [100000, 150000],
        },
    }
    if method == "matrix_free" and prediction_mode == "exact":
        return list(MATRIX_FREE_EXACT_SIZE_CAPS[framework][tier])
    return list(benchmark_targets[tier][method])


def _safe_average(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _prediction_repeat_count() -> int:
    return prediction_x_test_repeat_count()


def _time_prediction_call(callable_obj):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_obj()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, float(time.perf_counter() - start)


def _multi_output_prediction_inputs(
    *,
    n_test: int,
    d: int,
    dataset,
    seed: int,
) -> np.ndarray:
    if int(dataset.X_test.shape[0]) == int(n_test):
        return np.asarray(dataset.X_test, dtype=np.float32)
    rng = np.random.default_rng(seed + 1_000_003 + int(n_test))
    return rng.uniform(-3.0, 3.0, size=(int(n_test), d)).astype(np.float32)


def _measure_mojogp_multi_prediction_x_test_scaling(
    *,
    gp: MultiOutputGP,
    dataset,
    prediction_mode: str,
    framework: str,
    tier: str,
    canonical_first_time_s: float,
    canonical_memory_stats: dict[str, float],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = prediction_x_test_target_specs(
        variety="minimal" if _quick_enabled() else os.environ.get("MOJOGP_BENCHMARK_VARIETY", "standard"),
        tier=tier,
        framework=framework,
    )
    for spec in specs:
        n_test = int(spec["n_test"])
        try:
            if n_test == int(dataset.X_test.shape[0]):
                X_test = np.asarray(dataset.X_test, dtype=np.float32)
                first_time_s = canonical_first_time_s
                first_memory_stats = canonical_memory_stats
            else:
                X_test = _multi_output_prediction_inputs(
                    n_test=n_test,
                    d=int(dataset.X_train.shape[1]),
                    dataset=dataset,
                    seed=seed,
                )

                def _predict_size():
                    return gp.predict(X_test, return_var=True, variance_method=prediction_mode)

                first_start = time.perf_counter()
                _, first_memory_stats = measure_gpu_phase(_predict_size, interval=0.02)
                first_time_s = float(time.perf_counter() - first_start)

            def _predict_repeat():
                return gp.predict(X_test, return_var=True, variance_method=prediction_mode)

            repeat_times: list[float] = []
            for _ in range(_prediction_repeat_count()):
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


def _measure_gpytorch_multi_prediction_x_test_scaling(
    *,
    train_result: dict[str, object],
    pred_result: dict[str, object],
    dataset,
    prediction_mode: str,
    tier: str,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = prediction_x_test_target_specs(
        variety="minimal" if _quick_enabled() else os.environ.get("MOJOGP_BENCHMARK_VARIETY", "standard"),
        tier=tier,
        framework="gpytorch",
    )
    for spec in specs:
        n_test = int(spec["n_test"])
        if n_test == int(dataset.X_test.shape[0]):
            X_test = np.asarray(dataset.X_test, dtype=np.float32)
            first_pred = pred_result
        else:
            X_test = _multi_output_prediction_inputs(
                n_test=n_test,
                d=int(dataset.X_train.shape[1]),
                dataset=dataset,
                seed=seed,
            )
            first_pred = predict_gpytorch_multi_output(
                train_result,
                X_test,
                mode="cg",
                cg_tolerance=FAIR_MULTI_OUTPUT_SOLVER["cg_tolerance"],
                max_cg_iterations=FAIR_MULTI_OUTPUT_SOLVER["max_cg_iterations"],
                max_preconditioner_size=FAIR_MULTI_OUTPUT_SOLVER["max_preconditioner_size"],
                max_lanczos_quadrature_iterations=FAIR_MULTI_OUTPUT_SOLVER[
                    "max_lanczos_quadrature_iterations"
                ],
                min_preconditioning_size=FAIR_MULTI_OUTPUT_SOLVER[
                    "min_preconditioning_size"
                ],
                max_root_decomposition_size=FAIR_MULTI_OUTPUT_SOLVER[
                    "max_root_decomposition_size"
                ],
                use_love=prediction_mode == "love",
            )
        repeat_times: list[float] = []
        for _ in range(_prediction_repeat_count()):
            repeat_pred = predict_gpytorch_multi_output(
                train_result,
                X_test,
                mode="cg",
                cg_tolerance=FAIR_MULTI_OUTPUT_SOLVER["cg_tolerance"],
                max_cg_iterations=FAIR_MULTI_OUTPUT_SOLVER["max_cg_iterations"],
                max_preconditioner_size=FAIR_MULTI_OUTPUT_SOLVER["max_preconditioner_size"],
                max_lanczos_quadrature_iterations=FAIR_MULTI_OUTPUT_SOLVER[
                    "max_lanczos_quadrature_iterations"
                ],
                min_preconditioning_size=FAIR_MULTI_OUTPUT_SOLVER[
                    "min_preconditioning_size"
                ],
                max_root_decomposition_size=FAIR_MULTI_OUTPUT_SOLVER[
                    "max_root_decomposition_size"
                ],
                use_love=prediction_mode == "love",
            )
            repeat_times.append(float(repeat_pred.get("total_time_s", 0.0)))
        memory_stats = dict(first_pred.get("memory_stats", {}) or {})
        rows.append(
            prediction_x_test_scaling_entry(
                spec=spec,
                timing_quality="warm_repeated_prediction",
                cache_used=False,
                first_apply_time_s=float(first_pred.get("total_time_s", 0.0)),
                repeat_times_s=repeat_times,
                prediction_peak_gpu_mb=memory_stats.get("prediction_peak_gpu_mb"),
                prediction_delta_gpu_mb=memory_stats.get("prediction_delta_gpu_mb"),
                mean_time_s=first_pred.get("mean_time_s"),
                variance_time_s=first_pred.get("variance_time_s"),
            )
        )
    return rows


def _compute_accuracy(
    dataset, mean: np.ndarray, variance: np.ndarray
) -> AccuracyResult:
    std = np.sqrt(np.maximum(variance, 1e-10))
    task_rmses = [rmse(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
    task_maes = [mae(dataset.F_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
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


def _cg_telemetry_from_history(history: np.ndarray | None) -> dict[str, object]:
    if history is None or len(history) == 0:
        return {
            "measured": False,
            "cg_iterations_history": [],
            "cg_iterations_total": 0,
            "cg_iterations_mean": 0.0,
            "cg_iterations_max": 0,
            "cg_iterations_final_step": 0,
        }
    history = np.asarray(history, dtype=np.int32)
    return {
        "measured": True,
        "cg_iterations_history": history.tolist(),
        "cg_iterations_total": int(np.sum(history)),
        "cg_iterations_mean": float(np.mean(history)),
        "cg_iterations_max": int(np.max(history)),
        "cg_iterations_final_step": int(history[-1]),
    }


def _multi_output_benchmark_metadata(
    *, framework: str, method: str, prediction_mode: str, n_train: int, tier: str
) -> dict[str, object]:
    if framework == "mojogp" and method == "matrix_free":
        return {
            "comparison_class": "mojogp_only_scale",
            "baseline_backend": "none",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": (
                "N.B. MojoGP-only matrix-free multi-output row: no standard GPyTorch "
                "multi-output matrix-free comparator is published in this benchmark lane."
            ),
            "fairness_axes": {
                "comparator_scope": _fairness_axis(
                    "mojogp_only",
                    "No cross-framework comparator is published for this row.",
                ),
                "sample_count_n": _fairness_axis(
                    "mojogp_only",
                    "The row is intentionally outside the shared cross-framework n set.",
                ),
                "optimizer": _fairness_axis(
                    "aligned",
                    "The row keeps the same optimizer policy as the fair materialized lane.",
                ),
                "solver_budget": _fairness_axis(
                    "aligned",
                    "The row keeps the same CG and Lanczos budgets as the fair materialized lane.",
                ),
                "preconditioner": _fairness_axis(
                    "aligned",
                    "The row keeps MojoGP's benchmark preconditioner settings, but there is no direct comparator row.",
                ),
                "prediction_mode": _fairness_axis(
                    "mojogp_only",
                    "This route is not part of the published cross-framework comparison set.",
                ),
                "telemetry": _fairness_axis(
                    "observed", "MojoGP CG telemetry is observed for this row."
                ),
            },
        }

    gpytorch_sizes = set(
        _framework_sizes(
            framework="gpytorch",
            method=method,
            prediction_mode=prediction_mode,
            tier=tier,
        )
    )
    fair_match = n_train in gpytorch_sizes
    if fair_match:
        fairness_note = (
            "N.B. Shared-n fair materialized row: n, optimizer, CG budgets, prediction "
            "mode, and pivoted-Cholesky budget are aligned, but GPyTorch multi-output CG "
            "telemetry is still marked unverified because the current hook does not observe "
            "the lower-level solve path."
        )
        fairness_axes = {
            "comparator_scope": _fairness_axis(
                "aligned",
                "This row is part of the published fair cross-framework comparison set.",
            ),
            "sample_count_n": _fairness_axis(
                "aligned", "Both frameworks run at the same n."
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "Optimizer family, iteration budget, learning rate, and schedule are aligned.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "CG tolerance, iteration cap, trace-sample count, and Lanczos budget are aligned.",
            ),
            "preconditioner": _fairness_axis(
                "aligned", "Both sides use the pivoted-Cholesky benchmark budget."
            ),
            "prediction_mode": _fairness_axis(
                "aligned", "Both sides run the same published prediction mode."
            ),
            "telemetry": _fairness_axis(
                "unverified",
                "The GPyTorch multi-output row forces CG in config, but the current telemetry hook still does not observe the lower-level solve path.",
            ),
        }
    else:
        fairness_note = (
            "N.B. MojoGP-only materialized envelope row: no direct GPyTorch comparator is "
            "published at this n on the local GPU."
        )
        fairness_axes = {
            "comparator_scope": _fairness_axis(
                "mojogp_only",
                "No cross-framework comparator is published for this row.",
            ),
            "sample_count_n": _fairness_axis(
                "mojogp_only",
                "The row is intentionally outside the shared cross-framework n set.",
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "The row keeps the same optimizer policy as the fair materialized lane.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "The row keeps the same CG and Lanczos budgets as the fair materialized lane.",
            ),
            "preconditioner": _fairness_axis(
                "aligned",
                "The row keeps the same pivoted-Cholesky budget as the fair materialized lane.",
            ),
            "prediction_mode": _fairness_axis(
                "mojogp_only",
                "No same-n GPyTorch comparator is published for this row.",
            ),
            "telemetry": _fairness_axis(
                "observed", "MojoGP CG telemetry is observed for this row."
            ),
        }
    return {
        "comparison_class": "fair_match" if fair_match else "mojogp_only_scale",
        "baseline_backend": "none" if framework == "mojogp" else "standard",
        "keops_supported": False,
        "keops_used": False,
        "fairness_note": fairness_note,
        "fairness_axes": fairness_axes,
    }


def _multi_output_ard_benchmark_metadata(
    *, framework: str, method: str, prediction_mode: str, n_train: int, tier: str
) -> dict[str, object]:
    if method == "matrix_free":
        return {
            "comparison_class": "mojogp_only_scale",
            "baseline_backend": "none",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": (
                "N.B. MojoGP-only matrix-free multi-output ARD row: no strict "
                "GPyTorch multi-output matrix-free ARD comparator is published."
            ),
            "fairness_axes": {
                "comparator_scope": _fairness_axis(
                    "mojogp_only", "No cross-framework comparator is published for this row."
                ),
                "sample_count_n": _fairness_axis(
                    "mojogp_only",
                    "The row is intentionally outside the shared cross-framework n set.",
                ),
                "optimizer": _fairness_axis(
                    "aligned", "The row keeps the same optimizer policy as materialized ARD."
                ),
                "solver_budget": _fairness_axis(
                    "aligned", "The row keeps the same CG and Lanczos budgets as materialized ARD."
                ),
                "preconditioner": _fairness_axis(
                    "aligned", "MojoGP uses the ARD benchmark preconditioner policy."
                ),
                "prediction_mode": _fairness_axis(
                    "mojogp_only", "This route is not part of a cross-framework ARD lane."
                ),
                "telemetry": _fairness_axis(
                    "observed", "MojoGP CG telemetry is observed for this row."
                ),
            },
        }

    gpytorch_sizes = set(MULTI_OUTPUT_ARD_MATERIALIZED_SHARED_SIZES[tier])
    fair_n = n_train in gpytorch_sizes
    if not fair_n:
        return {
            "comparison_class": "mojogp_only_scale",
            "baseline_backend": "none" if framework == "mojogp" else "standard",
            "keops_supported": False,
            "keops_used": False,
            "fairness_note": (
                "N.B. MojoGP-only multi-output ARD envelope row: no direct GPyTorch "
                "materialized comparator is published at this n on the local GPU."
            ),
            "fairness_axes": {
                "comparator_scope": _fairness_axis(
                    "mojogp_only", "No cross-framework comparator is published for this row."
                ),
                "sample_count_n": _fairness_axis(
                    "mojogp_only",
                    "The row is intentionally outside the shared cross-framework n set.",
                ),
                "optimizer": _fairness_axis("aligned", "Optimizer policy is unchanged."),
                "solver_budget": _fairness_axis("aligned", "CG/Lanczos budgets are unchanged."),
                "preconditioner": _fairness_axis(
                    "aligned", "Preconditioner budget follows the ARD benchmark policy."
                ),
                "prediction_mode": _fairness_axis(
                    "mojogp_only", "No same-n GPyTorch comparator is published."
                ),
                "telemetry": _fairness_axis(
                    "observed", "MojoGP CG telemetry is observed for this row."
                ),
            },
        }

    return {
        "comparison_class": "near_fair_solver_unverified",
        "baseline_backend": "none" if framework == "mojogp" else "standard",
        "keops_supported": False,
        "keops_used": False,
        "fairness_note": (
            "N.B. GPyTorch multi-output ARD uses the same mathematical ICM + ARD "
            "data kernel, but lower-level solver telemetry is unverified for strict "
            "CG-vs-CG claims."
        ),
        "fairness_axes": {
            "comparator_scope": _fairness_axis(
                "near_fair", "Mathematical model matches; solver telemetry is not strict enough for headline fair CG-vs-CG."
            ),
            "sample_count_n": _fairness_axis("aligned", "Both frameworks run at the same n."),
            "optimizer": _fairness_axis(
                "aligned", "Optimizer family, iteration budget, learning rate, and schedule are aligned."
            ),
            "solver_budget": _fairness_axis(
                "aligned", "CG tolerance, iteration cap, trace samples, and Lanczos budget are aligned."
            ),
            "preconditioner": _fairness_axis(
                "aligned", "Both sides use the same benchmark preconditioner budget."
            ),
            "prediction_mode": _fairness_axis(
                "aligned", "Both sides run the same published prediction mode."
            ),
            "telemetry": _fairness_axis(
                "unverified", "GPyTorch multitask lower-level solve telemetry is not observed robustly enough for strict fairness."
            ),
        },
    }


def _preconditioner_config(
    solver_config: dict[str, object], *, framework: str
) -> dict[str, object]:
    return {
        "family": "pivoted_cholesky",
        "framework": framework,
        "rank": solver_config.get("precond_rank"),
        "method": solver_config.get("precond_method", "default"),
        "rebuild_threshold": solver_config.get("precond_rebuild_threshold"),
    }


def _cg_telemetry_quality(cg_telemetry: dict[str, object]) -> dict[str, object]:
    training = dict(cg_telemetry.get("training", {}))
    prediction = dict(cg_telemetry.get("prediction", {}))
    return {
        "training": training.get("telemetry_quality", "missing"),
        "prediction": prediction.get("telemetry_quality", "missing"),
        "configured_for_cg": bool(
            training.get("configured_for_cg") or prediction.get("configured_for_cg")
        ),
        "observed_cg_calls": bool(
            training.get("observed_cg_calls") or prediction.get("observed_cg_calls")
        ),
    }


def _timing_payload(iter_times_ms: list[float] | np.ndarray | None) -> dict[str, object]:
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


def _merge_mojogp_phase_memory_stats(
    memory_stats: dict[str, float],
    fit_memory_stats: dict[str, float],
    pred_memory_stats: dict[str, float],
    *,
    prediction_mode: str,
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
    if prediction_mode == "exact":
        merged["exact_prediction_peak_gpu_mb"] = pred_peak
        merged["exact_prediction_delta_gpu_mb"] = float(
            pred_memory_stats.get("phase_delta_gpu_mb", 0.0)
        )
    elif prediction_mode == "love":
        merged["love_prediction_peak_gpu_mb"] = pred_peak
        merged["love_prediction_delta_gpu_mb"] = float(
            pred_memory_stats.get("phase_delta_gpu_mb", 0.0)
        )
    return merged


def _multi_output_ard_config(
    *,
    dataset: MultiOutputDataset,
    lengthscales: object,
    ard: bool,
) -> dict[str, object]:
    if not ard:
        return {"ard": False}
    true_params = dict(dataset.true_params or {})
    relevant_indices = list(true_params.get("relevant_indices", []))
    irrelevant_indices = list(true_params.get("irrelevant_indices", []))
    metrics = compute_ard_relevance_metrics(
        lengthscales,
        relevant_indices=relevant_indices,
        irrelevant_indices=irrelevant_indices,
    )
    return {
        "ard": True,
        "relevant_dims": int(true_params.get("relevant_dims", len(relevant_indices))),
        "relevant_indices": relevant_indices,
        "irrelevant_indices": irrelevant_indices,
        "ard_metrics": metrics,
        **metrics,
    }


def _build_result(
    *,
    dataset,
    framework: str,
    model_type: str,
    training_method: str,
    method: str,
    prediction_mode: str,
    max_iterations: int,
    training_time_s: float,
    prediction_total_time_s: float,
    prediction_mean_time_s: float | None,
    prediction_variance_time_s: float | None,
    prediction_timing_quality: str | None,
    iterations_run: int,
    early_stopped: bool,
    memory_stats: dict[str, float],
    mean: np.ndarray,
    variance: np.ndarray,
    learned_lengthscale: float,
    learned_noise: float,
    learned_outputscale: float,
    final_nll: float,
    optimizer_config: dict[str, object],
    training_solver_config: dict[str, object],
    prediction_solver_config: dict[str, object],
    cg_telemetry: dict[str, object],
    tier: str,
    extra_config: dict[str, object] | None = None,
    training_iter_times_ms: list[float] | np.ndarray | None = None,
    prediction_x_test_scaling: list[dict[str, object]] | None = None,
    benchmark_name: str = "multi_output_scaling",
    kernel_label: str = "rbf",
) -> BenchmarkResult:
    vram_info = get_vram_info()
    extra_config = extra_config or {}
    timing_payload = _timing_payload(training_iter_times_ms)
    iter_timing_quality = (
        "direct_per_iteration"
        if timing_payload["iter_times_ms"] is not None
        else "derived_total_div_iterations"
    )
    if prediction_mean_time_s is None or prediction_variance_time_s is None:
        prediction_mean_time_s = 0.0
        prediction_variance_time_s = prediction_total_time_s
        prediction_timing_quality = prediction_timing_quality or "total_only_combined_call"
    else:
        prediction_timing_quality = prediction_timing_quality or "observed_mean_variance_split"
    metadata_fn = (
        _multi_output_ard_benchmark_metadata
        if benchmark_name == "multi_output_ard_scaling"
        else _multi_output_benchmark_metadata
    )
    return BenchmarkResult(
        config={
            "benchmark": benchmark_name,
            "benchmark_tier": tier,
            "framework": framework,
            "model_type": model_type,
            "kernel": kernel_label,
            "training_method": training_method,
            "method": method,
            "prediction_mode": prediction_mode,
            "n": int(dataset.X_train.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "n_test": int(dataset.X_test.shape[0]),
            "num_tasks": int(dataset.Y_train.shape[1]),
            "task_correlation": dataset.true_params["task_correlation"],
            "vram_tier": vram_info["tier"],
            "vram_gb": vram_info["vram_gb"],
            "optimizer_config": optimizer_config,
            "training_solver_config": training_solver_config,
            "prediction_solver_config": prediction_solver_config,
            "preconditioner_config": _preconditioner_config(
                training_solver_config,
                framework=framework,
            ),
            "cg_telemetry": cg_telemetry,
            "cg_telemetry_quality": _cg_telemetry_quality(cg_telemetry),
            "prediction_timing_quality": prediction_timing_quality,
            "iter_timing_quality": iter_timing_quality,
            "phase_memory_quality": (
                "phase_specific" if memory_stats.get("prediction_delta_gpu_mb") is not None else "overall_only"
            ),
            **extra_config,
            **metadata_fn(
                framework=framework,
                method=training_method,
                prediction_mode=prediction_mode,
                n_train=int(dataset.X_train.shape[0]),
                tier=tier,
            ),
        },
        accuracy=_compute_accuracy(dataset, mean, variance),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=prediction_mean_time_s,
            prediction_variance_time_s=prediction_variance_time_s,
            end_to_end_time_s=training_time_s + prediction_total_time_s,
            iterations_run=iterations_run,
            max_iterations=max_iterations,
            early_stopped=early_stopped,
            ms_per_iteration=float(
                timing_payload["iter_time_median_ms"]
                if timing_payload["iter_time_median_ms"] is not None
                else (training_time_s / max(iterations_run, 1)) * 1000.0
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
            exact_prediction_peak_gpu_mb=memory_stats.get("exact_prediction_peak_gpu_mb"),
            exact_prediction_delta_gpu_mb=memory_stats.get("exact_prediction_delta_gpu_mb"),
            love_prediction_peak_gpu_mb=memory_stats.get("love_prediction_peak_gpu_mb"),
            love_prediction_delta_gpu_mb=memory_stats.get("love_prediction_delta_gpu_mb"),
        ),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=learned_lengthscale,
            learned_noise=learned_noise,
            learned_outputscale=learned_outputscale,
            final_nll=final_nll,
        ),
    )


def _run_case(
    *,
    framework: str,
    prediction_mode: str,
    method: str,
    n_train: int,
    d: int,
    num_tasks: int,
    tier: str,
    results_dir: str,
    session: IsolatedGPUTestSession,
    dataset_path: str | None = None,
    specialization: dict[str, object] | None = None,
    ard: bool = False,
    relevant_dims: int | None = None,
) -> Path:
    dataset_seed = 1000 + n_train + d * 100 + (
        int(relevant_dims or min(3, d)) * 17 if ard else 0
    )
    if dataset_path is not None:
        dataset_payload = load_dataset_artifact(dataset_path)
        dataset = MultiOutputDataset(
            X_train=np.asarray(dataset_payload["X_train"], dtype=np.float32),
            Y_train=np.asarray(dataset_payload["Y_train"], dtype=np.float32),
            X_test=np.asarray(dataset_payload["X_test"], dtype=np.float32),
            Y_test=np.asarray(dataset_payload["Y_test"], dtype=np.float32),
            F_test=np.asarray(dataset_payload["F_test"], dtype=np.float32),
            true_params=dict(dataset_payload["true_params"]),
            name=str(dataset_payload["name"]),
            description=str(dataset_payload["description"]),
        )
    else:
        if ard:
            dataset = generate_multi_output_structured_ard_data(
                n_train=n_train,
                n_test=BENCHMARK_PREDICTION_N_TEST,
                d=d,
                num_tasks=num_tasks,
                relevant_dims=int(relevant_dims or min(3, d)),
                seed=dataset_seed,
            )
        else:
            dataset = generate_multi_output_data(
                n_train=n_train,
                n_test=BENCHMARK_PREDICTION_N_TEST,
                d=d,
                num_tasks=num_tasks,
                kernel_type="rbf",
                task_correlation="medium",
                seed=dataset_seed,
            )
    max_iterations = 60 if method == "materialized" else 40
    learning_rate = 0.03 if method == "materialized" else 0.02
    lr_schedule = "cosine"

    if framework == "mojogp":
        gp = MultiOutputGP(
            kernel="rbf",
            ard=ard,
            num_probes=FAIR_MULTI_OUTPUT_SOLVER["num_trace_samples"],
            max_cg_iterations=FAIR_MULTI_OUTPUT_SOLVER["max_cg_iterations"],
            cg_tolerance=FAIR_MULTI_OUTPUT_SOLVER["cg_tolerance"],
            max_tridiag_iterations=FAIR_MULTI_OUTPUT_SOLVER[
                "max_lanczos_quadrature_iterations"
            ],
            preconditioner_rank=FAIR_MULTI_OUTPUT_SOLVER["precond_rank"],
            preconditioner=FAIR_MULTI_OUTPUT_SOLVER["precond"],
            use_preconditioner=FAIR_MULTI_OUTPUT_SOLVER["use_preconditioner"],
        )
        if specialization is not None:
            from tests.benchmarks.specialization_adapter import apply_specialization_to_model

            apply_specialization_to_model(gp, specialization)
        fit_start = time.perf_counter()
        training_result, fit_memory_stats = measure_gpu_phase(
            lambda: gp.fit(
                dataset.X_train,
                dataset.Y_train,
                method=method,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                lr_schedule=lr_schedule,
                verbose=False,
            ),
            interval=0.02,
        )
        training_time_s = time.perf_counter() - fit_start

        pred_start = time.perf_counter()
        (mean, variance), pred_memory_stats = measure_gpu_phase(
            lambda: gp.predict(
                dataset.X_test,
                return_var=True,
                variance_method=prediction_mode,
            ),
            interval=0.02,
        )
        prediction_total_time_s = time.perf_counter() - pred_start

        memory_stats = _merge_mojogp_phase_memory_stats(
            session.collect_memory_stats(),
            fit_memory_stats,
            pred_memory_stats,
            prediction_mode=prediction_mode,
        )
        prediction_x_test_scaling = _measure_mojogp_multi_prediction_x_test_scaling(
            gp=gp,
            dataset=dataset,
            prediction_mode=prediction_mode,
            framework="mojogp",
            tier=tier,
            canonical_first_time_s=prediction_total_time_s,
            canonical_memory_stats=pred_memory_stats,
            seed=dataset_seed,
        )
        backend_train = getattr(gp, "_backend_train_info", {}) or {}
        if ard:
            learned_lengthscales = np.asarray(training_result.lengthscales, dtype=np.float32)
            learned_outputscale = float(training_result.outputscale)
        else:
            learned_lengthscales = np.asarray(training_result.params, dtype=np.float32)
            learned_outputscale = float(np.mean(training_result.effective_scales))
        benchmark = _build_result(
            dataset=dataset,
            framework="mojogp",
            model_type="MultiOutputGP",
            training_method=method,
            method=method,
            prediction_mode=prediction_mode,
            max_iterations=max_iterations,
            training_time_s=training_time_s,
            prediction_total_time_s=prediction_total_time_s,
            prediction_mean_time_s=None,
            prediction_variance_time_s=None,
            prediction_timing_quality="total_only_combined_call",
            iterations_run=int(training_result.iterations),
            early_stopped=bool(training_result.converged)
            or int(training_result.iterations) < max_iterations,
            memory_stats=memory_stats,
            mean=np.asarray(mean, dtype=np.float32),
            variance=np.asarray(variance, dtype=np.float32),
            learned_lengthscale=float(np.mean(learned_lengthscales)),
            learned_noise=float(np.mean(training_result.noise_per_task)),
            learned_outputscale=learned_outputscale,
            final_nll=float(training_result.final_nll),
            optimizer_config={
                "max_iterations": max_iterations,
                "learning_rate": learning_rate,
                "lr_schedule": lr_schedule,
            },
            training_solver_config={
                "framework": "mojogp",
                "model_family": "MultiOutputGP",
                "mode": method,
                "cg_tolerance": float(FAIR_MULTI_OUTPUT_SOLVER["cg_tolerance"]),
                "max_cg_iterations": int(FAIR_MULTI_OUTPUT_SOLVER["max_cg_iterations"]),
                "num_trace_samples": int(FAIR_MULTI_OUTPUT_SOLVER["num_trace_samples"]),
                "max_tridiag_iter": int(
                    FAIR_MULTI_OUTPUT_SOLVER["max_lanczos_quadrature_iterations"]
                ),
                "precond_rank": int(FAIR_MULTI_OUTPUT_SOLVER["precond_rank"]),
                "precond_method": backend_train.get("precond_method"),
                "precond_rebuild_threshold": backend_train.get(
                    "precond_rebuild_threshold"
                ),
                "min_preconditioning_size": int(
                    FAIR_MULTI_OUTPUT_SOLVER["min_preconditioning_size"]
                ),
            },
            prediction_solver_config={
                "framework": "mojogp",
                "mode": method,
                "prediction_mode": prediction_mode,
                "cg_tolerance": float(FAIR_MULTI_OUTPUT_SOLVER["cg_tolerance"]),
                "max_cg_iterations": int(FAIR_MULTI_OUTPUT_SOLVER["max_cg_iterations"]),
                "precond_rank": int(FAIR_MULTI_OUTPUT_SOLVER["precond_rank"]),
                "precond_method": backend_train.get("precond_method"),
                "max_root_decomposition_size": int(
                    FAIR_MULTI_OUTPUT_SOLVER["max_root_decomposition_size"]
                ),
            },
            cg_telemetry={
                "training": _cg_telemetry_from_history(
                    np.asarray(
                        gp._raw_result.get("cg_iterations_history", []), dtype=np.int32
                    )
                    if getattr(gp, "_raw_result", None) is not None
                    else None
                )
                | {
                    "configured_for_cg": True,
                    "observed_cg_calls": True,
                    "telemetry_quality": "observed",
                },
                "prediction": {
                    "configured_for_cg": prediction_mode == "exact",
                    "observed_cg_calls": prediction_mode == "exact",
                    "telemetry_quality": (
                        "observed" if prediction_mode == "exact" else "not_applicable"
                    ),
                },
            },
            tier=tier,
            extra_config={
                "specialization": specialization or {},
                **_multi_output_ard_config(
                    dataset=dataset,
                    lengthscales=learned_lengthscales,
                    ard=ard,
                ),
            },
            training_iter_times_ms=backend_train.get("iter_times_ms"),
            prediction_x_test_scaling=prediction_x_test_scaling,
            benchmark_name="multi_output_ard_scaling" if ard else "multi_output_scaling",
            kernel_label="rbf_ard" if ard else "rbf",
        )
    elif framework == "gpytorch":
        train_result = train_gpytorch_multi_output(
            dataset.X_train,
            dataset.Y_train,
            kernel_type="rbf",
            num_tasks=num_tasks,
            mode="cg",
            n_iterations=max_iterations,
            lr=learning_rate,
            lr_schedule=lr_schedule,
            cg_tolerance=FAIR_MULTI_OUTPUT_SOLVER["cg_tolerance"],
            max_cg_iterations=FAIR_MULTI_OUTPUT_SOLVER["max_cg_iterations"],
            num_trace_samples=FAIR_MULTI_OUTPUT_SOLVER["num_trace_samples"],
            max_preconditioner_size=FAIR_MULTI_OUTPUT_SOLVER["max_preconditioner_size"],
            max_lanczos_quadrature_iterations=FAIR_MULTI_OUTPUT_SOLVER[
                "max_lanczos_quadrature_iterations"
            ],
            min_preconditioning_size=FAIR_MULTI_OUTPUT_SOLVER[
                "min_preconditioning_size"
            ],
            monitor_memory=True,
            device="cuda",
            ard=ard,
        )
        pred_result = predict_gpytorch_multi_output(
            train_result,
            dataset.X_test,
            mode="cg",
            cg_tolerance=FAIR_MULTI_OUTPUT_SOLVER["cg_tolerance"],
            max_cg_iterations=FAIR_MULTI_OUTPUT_SOLVER["max_cg_iterations"],
            max_preconditioner_size=FAIR_MULTI_OUTPUT_SOLVER["max_preconditioner_size"],
            max_lanczos_quadrature_iterations=FAIR_MULTI_OUTPUT_SOLVER[
                "max_lanczos_quadrature_iterations"
            ],
            min_preconditioning_size=FAIR_MULTI_OUTPUT_SOLVER[
                "min_preconditioning_size"
            ],
            max_root_decomposition_size=FAIR_MULTI_OUTPUT_SOLVER[
                "max_root_decomposition_size"
            ],
            use_love=prediction_mode == "love",
        )
        gpytorch_memory_stats = merge_gpytorch_benchmark_memory(
            dict(train_result.get("memory_stats", {})),
            dict(pred_result.get("memory_stats", {})),
        )
        prediction_x_test_scaling = _measure_gpytorch_multi_prediction_x_test_scaling(
            train_result=train_result,
            pred_result=pred_result,
            dataset=dataset,
            prediction_mode=prediction_mode,
            tier=tier,
            seed=dataset_seed,
        )
        benchmark = _build_result(
            dataset=dataset,
            framework="gpytorch",
            model_type="MultiOutputGP",
            training_method=method,
            method="cg",
            prediction_mode=prediction_mode,
            max_iterations=max_iterations,
            training_time_s=float(train_result["training_time_s"]),
            prediction_total_time_s=float(pred_result["total_time_s"]),
            prediction_mean_time_s=float(pred_result.get("mean_time_s", 0.0)),
            prediction_variance_time_s=float(pred_result.get("variance_time_s", 0.0)),
            prediction_timing_quality="observed_mean_variance_split",
            iterations_run=int(train_result["iterations_run"]),
            early_stopped=bool(train_result["early_stopped"]),
            memory_stats=gpytorch_memory_stats,
            mean=np.asarray(pred_result["mean"], dtype=np.float32),
            variance=np.asarray(pred_result["variance"], dtype=np.float32),
            learned_lengthscale=float(
                train_result["learned_params"].get("lengthscale", 1.0)
            ),
            learned_noise=float(train_result["learned_params"].get("noise", 0.1)),
            learned_outputscale=float(
                train_result["learned_params"].get("outputscale", 1.0)
            ),
            final_nll=float(train_result["final_nll"]),
            optimizer_config=dict(train_result.get("optimizer_config", {})),
            training_solver_config=dict(train_result.get("solver_config", {})),
            prediction_solver_config=dict(pred_result.get("solver_config", {})),
            cg_telemetry={
                "training": train_result.get("cg_telemetry", {}),
                "prediction": pred_result.get("cg_telemetry", {}),
            },
            tier=tier,
            extra_config={
                "specialization": specialization or {},
                **_multi_output_ard_config(
                    dataset=dataset,
                    lengthscales=train_result["learned_params"].get(
                        "lengthscales",
                        [train_result["learned_params"].get("lengthscale", 1.0)],
                    ),
                    ard=ard,
                ),
            },
            training_iter_times_ms=train_result.get("iter_times_ms"),
            prediction_x_test_scaling=prediction_x_test_scaling,
            benchmark_name="multi_output_ard_scaling" if ard else "multi_output_scaling",
            kernel_label="rbf_ard" if ard else "rbf",
        )
    else:
        raise ValueError(f"Unknown framework '{framework}'")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    return save_result_artifact(
        benchmark,
        results_path,
        "multi_output_ard_scaling" if ard else "multi_output_scaling",
    )


def _handle_child(payload: dict[str, object], session: IsolatedGPUTestSession):
    result_path = _run_case(
        framework=str(payload["framework"]),
        prediction_mode=str(payload["prediction_mode"]),
        method=str(payload["method"]),
        n_train=int(payload["n_train"]),
        d=int(payload["d"]),
        num_tasks=int(payload["num_tasks"]),
        tier=str(payload["tier"]),
        results_dir=str(payload["results_dir"]),
        session=session,
        dataset_path=(
            None
            if payload.get("dataset_path") is None
            else str(payload.get("dataset_path"))
        ),
        specialization=(
            None
            if payload.get("specialization") is None
            else dict(payload.get("specialization", {}))
        ),
        ard=bool(payload.get("ard", False)),
        relevant_dims=(
            None
            if payload.get("relevant_dims") is None
            else int(payload.get("relevant_dims"))
        ),
    )
    return {"result_path": result_path}


def main() -> int:
    return run_child_main(_handle_child)


if __name__ == "__main__":
    raise SystemExit(main())
