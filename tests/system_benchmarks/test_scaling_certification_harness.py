"""VRAM-aware scaling certification benchmarks.

These tests certify that the active single-output wrapper routes can train and
predict at route-appropriate large-n settings on the current GPU without
accuracy collapse. They intentionally reuse the existing benchmark result and
report infrastructure rather than creating a parallel benchmarking path.

Important: route-to-route VRAM claims are validated by the isolated subprocess
checks in `tests/integration/test_gpu_memory_measurement.py`. This benchmark
records end-to-end memory telemetry for practitioner reporting, but it does not
replace the isolated VRAM proof surface.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
import os
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import numpy as np
import pytest
import torch

from mojogp import SingleOutputGP, RBF
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
from tests.benchmarks.single_output_scaling import run_single_output_scaling_subprocess
from tests.shared.subprocess_harness import run_isolated_case
from mojogp.gp import (
    _DEFAULT_PREDICT_NCOLS_HINT,
)

from .conftest import (
    assert_gpu_available,
    assert_gpu_was_used,
    get_matrix_free_capability_tier,
    get_vram_info,
    get_vram_tier,
    requires_cuda,
)
from tests.shared.benchmarking.data_generators import (
    generate_gp_prior_data,
    generate_single_output_structured_ard_data,
    generate_structured_function_data,
)
from tests.shared.benchmarking.data_generators import SyntheticDataset
from tests.shared.benchmarking.ard_metrics import compute_ard_relevance_metrics
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
    param_relative_error,
)
from tests.shared.benchmarking.mojogp_runners import normalize_single_output_benchmark_hparams
from tests.shared.benchmarking.gpytorch_models import (
    is_keops_available,
    keops_supported_kernels,
    merge_gpytorch_benchmark_memory,
    predict_gpytorch_single_output,
    train_gpytorch_single_output,
)
from tests.benchmarks.comparison_policy import policy_for
from tests.shared.benchmarking.report import (
    load_benchmark_result,
    print_result,
    save_summary_report,
)
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)


SCALING_DIMS = {
    "minimal": [5],
    "standard": [5, 17],
    "extensive": [5, 17, 31],
    "matrix_free_n_scaling": [5, 17],
}


MATERIALIZED_SCALING_DIMS = {
    "minimal": [5],
    "standard": [5, 17],
    "extensive": [5],
}


MATERIALIZED_TARGETS = {
    "minimal": {
        "xsmall": [5000],
        "small": [8000],
        "medium": [12000],
        "large": [20000],
        "xlarge": [30000],
    },
    "standard": {
        "xsmall": [5000, 6500, 8000],
        "small": [8000, 12000, 16000],
        "medium": [12000, 20000, 25000],
        "large": [20000, 35000, 45000],
        "xlarge": [30000, 60000, 75000],
    },
    "extensive": {
        "xsmall": [2000, 4000, 6000, 8000],
        "small": [4000, 8000, 12000],
        "medium": [5000, 10000, 15000, 20000],
        "large": [5000, 10000, 20000, 30000],
        "xlarge": [10000, 20000, 30000, 40000, 50000],
    },
}


GPYTORCH_MATERIALIZED_TARGETS = {
    variety: {tier: list(sizes) for tier, sizes in tiers.items()}
    for variety, tiers in MATERIALIZED_TARGETS.items()
}
GPYTORCH_MATERIALIZED_TARGETS["standard"]["large"] = [20000]
GPYTORCH_MATERIALIZED_TARGETS["extensive"]["large"] = [5000, 10000, 20000]


MATRIX_FREE_TARGETS = {
    "minimal": {
        "xsmall": [5000],
        "small": [10000],
        "medium": [25000],
        "large": [50000],
        "xlarge": [75000],
    },
    "standard": {
        "xsmall": [5000, 10000, 25000],
        "small": [10000, 25000, 50000],
        "medium": [25000, 50000, 75000],
        "large": [25000, 50000, 75000],
        "xlarge": [50000, 75000, 100000],
    },
    "extensive": {
        "xsmall": [5000, 10000, 25000],
        "small": [10000, 25000, 50000],
        "medium": [25000, 50000, 75000],
        "large": [25000, 50000, 75000],
        "xlarge": [50000, 75000, 100000],
    },
    "matrix_free_n_scaling": {
        "xsmall": [2_000, 5_000],
        "small": [5_000, 10_000],
        "medium": [10_000, 25_000, 50_000],
        "large": [10_000, 50_000, 100_000, 150_000],
        "xlarge": [10_000, 50_000, 100_000, 150_000],
    },
}


MATRIX_FREE_N_SCALING_TARGETS = MATRIX_FREE_TARGETS["matrix_free_n_scaling"]


MATRIX_FREE_EXACT_TARGETS = {
    "minimal": {
        "xsmall": [5000],
        "small": [10000],
        "medium": [25000],
        "large": [25000],
        "xlarge": [50000],
    },
    "standard": {
        "xsmall": [5000, 10000],
        "small": [10000, 25000],
        "medium": [25000, 50000],
        "large": [25000, 50000],
        "xlarge": [50000, 75000],
    },
    "extensive": {
        "xsmall": [5000, 10000],
        "small": [10000, 25000],
        "medium": [25000, 50000],
        "large": [25000, 50000],
        "xlarge": [50000, 75000],
    },
    "matrix_free_n_scaling": {
        "xsmall": [2_000],
        "small": [5_000],
        "medium": [10_000, 25_000],
        "large": [10_000, 50_000],
        "xlarge": [10_000, 50_000],
    },
}


STRICT_FAIR_INITIAL_NOISE = 0.1
SCALING_CONFIG_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "scaling_configs.json"


@lru_cache(maxsize=1)
def _scaling_config_registry() -> dict[str, dict[str, object]]:
    if not SCALING_CONFIG_PATH.exists():
        return {}
    payload = json.loads(SCALING_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Scaling config file must contain an object: {SCALING_CONFIG_PATH}")
    registry: dict[str, dict[str, object]] = {}
    for name, config in payload.items():
        if not isinstance(config, dict):
            raise ValueError(f"Scaling config '{name}' must be an object")
        registry[str(name)] = dict(config)
    return registry


def _active_scaling_config_name() -> str | None:
    raw = os.environ.get("MOJOGP_SCALING_CONFIG")
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _active_scaling_config() -> dict[str, object]:
    name = _active_scaling_config_name()
    if name is None:
        return {}
    registry = _scaling_config_registry()
    if name not in registry:
        raise ValueError(
            f"Unknown MOJOGP_SCALING_CONFIG='{name}'. Expected one of {sorted(registry)}"
        )
    config = dict(registry[name])
    config["config_name"] = name
    return config


def _scaling_config_methods() -> set[str] | None:
    config = _active_scaling_config()
    methods = config.get("methods")
    if methods is None:
        return None
    if not isinstance(methods, list):
        raise ValueError("scaling config 'methods' must be a list")
    return {str(method) for method in methods}


def _scaling_config_enables_method(method: str) -> bool:
    methods = _scaling_config_methods()
    return methods is None or method in methods


def _single_output_initial_params(*, ard: bool, d: int) -> np.ndarray | None:
    if not ard:
        return None
    return np.ones(d + 1, dtype=np.float32)


def _single_output_initialization_config(*, ard: bool, d: int) -> dict[str, object]:
    if not ard:
        return {
            "initial_noise": STRICT_FAIR_INITIAL_NOISE,
            "initialization_policy": "kernel_default",
            "strict_fair_aligned_init": True,
        }
    return {
        "initial_noise": STRICT_FAIR_INITIAL_NOISE,
        "initialization_policy": "strict_fair_ard_all_ones",
        "strict_fair_aligned_init": True,
        "initial_lengthscales": [1.0] * d,
        "initial_outputscale": 1.0,
    }


FAIR_SINGLE_OUTPUT_SOLVER = {
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


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _prediction_repeat_count() -> int:
    return prediction_x_test_repeat_count()


def _prediction_repeat_count_for_case(
    *, framework: str, method: str, prediction_mode: str
) -> int:
    if method == "matrix_free" and prediction_mode == "exact":
        # Exact matrix-free variance already performs a large multi-RHS CG solve.
        # Repeating that path can dominate training certification and mask the
        # fair KeOps/LOVE comparison lane.
        return 0
    return _prediction_repeat_count()


def _prediction_x_test_scaling_policy(
    *, framework: str, method: str, prediction_mode: str
) -> str:
    if method == "matrix_free" and prediction_mode == "exact":
        return "canonical_only_matrix_free_exact"
    return "full_core_plus_allowed_envelope"


def _emit_scaling_phase(
    phase: str,
    *,
    framework: str,
    method: str,
    prediction_mode: str,
    n_train: int,
    d: int,
    elapsed_s: float | None = None,
) -> None:
    message = (
        "[scaling_case] "
        f"phase={phase} framework={framework} method={method} "
        f"prediction_mode={prediction_mode} n={n_train} d={d}"
    )
    if elapsed_s is not None:
        message += f" elapsed_s={elapsed_s:.6f}"
    print(message, flush=True)


def _time_prediction_call(callable_obj):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_obj()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, float(time.perf_counter() - start)


def _prediction_time_quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "prediction_repeated_median_time_s": None,
            "prediction_repeated_p5_time_s": None,
            "prediction_repeated_p95_time_s": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "prediction_repeated_median_time_s": float(np.median(arr)),
        "prediction_repeated_p5_time_s": float(np.percentile(arr, 5)),
        "prediction_repeated_p95_time_s": float(np.percentile(arr, 95)),
    }


def _clear_mojogp_prediction_caches_for_benchmark(gp: SingleOutputGP) -> None:
    destroy_cache = getattr(gp, "_destroy_prediction_cache", None)
    if callable(destroy_cache):
        destroy_cache()
    gp._cached_alpha = None
    gp._cached_alpha_info = None
    gp._cached_love_method = None
    training_result = getattr(gp, "_training_result", None)
    if training_result is not None and hasattr(training_result, "lanczos_root"):
        training_result.lanczos_root = None


def _single_output_prediction_inputs(
    *,
    n_test: int,
    d: int,
    data_config: dict[str, object],
    dataset: SyntheticDataset,
) -> np.ndarray:
    if int(dataset.X_test.shape[0]) == int(n_test):
        return np.asarray(dataset.X_test, dtype=np.float32)

    seed = int(data_config.get("seed", _single_output_route_seed(n_train=0, d=d)))
    rng = np.random.default_rng(seed + 1_000_003 + int(n_test))
    dataset_family = str(data_config.get("dataset_family", "structured_function"))
    if dataset_family in {"structured_function", "structured_ard", "gp_prior"}:
        if dataset_family == "structured_function" and data_config.get("function_type") == "periodic_signal":
            low, high = 0.0, 4.0
        else:
            x_range = data_config.get("x_range", (-3.0, 3.0))
            low, high = float(x_range[0]), float(x_range[1])
        return rng.uniform(low, high, size=(int(n_test), d)).astype(np.float32)

    train = np.asarray(dataset.X_train, dtype=np.float32)
    low = np.min(train, axis=0)
    high = np.max(train, axis=0)
    span = np.maximum(high - low, 1e-3)
    return rng.uniform(low - 0.05 * span, high + 0.05 * span, size=(int(n_test), d)).astype(np.float32)


def _measure_mojogp_single_prediction_x_test_scaling(
    *,
    gp: SingleOutputGP,
    dataset: SyntheticDataset,
    data_config: dict[str, object],
    method: str,
    prediction_mode: str,
    benchmark_variety: str,
    tier: str,
    canonical_first_time_s: float,
    canonical_repeat_times_s: list[float],
    canonical_memory_stats: dict[str, float],
    canonical_backend_predict: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specs = prediction_x_test_target_specs(
        variety=benchmark_variety,
        tier=tier,
        framework="mojogp",
    )
    if _prediction_x_test_scaling_policy(
        framework="mojogp",
        method=method,
        prediction_mode=prediction_mode,
    ) == "canonical_only_matrix_free_exact":
        canonical_n_test = int(dataset.X_test.shape[0])
        specs = [spec for spec in specs if int(spec["n_test"]) == canonical_n_test]
    repeat_count = _prediction_repeat_count_for_case(
        framework="mojogp",
        method=method,
        prediction_mode=prediction_mode,
    )
    for spec in specs:
        n_test = int(spec["n_test"])
        if n_test == int(dataset.X_test.shape[0]):
            rows.append(
                prediction_x_test_scaling_entry(
                    spec=spec,
                    timing_quality="prepared_cache_split",
                    cache_used=True,
                    first_apply_time_s=canonical_first_time_s,
                    repeat_times_s=canonical_repeat_times_s,
                    prediction_peak_gpu_mb=canonical_memory_stats.get("phase_peak_gpu_mb"),
                    prediction_delta_gpu_mb=canonical_memory_stats.get("phase_delta_gpu_mb"),
                    mean_time_s=canonical_backend_predict.get("prediction_mean_time_s"),
                    variance_time_s=canonical_backend_predict.get("prediction_variance_time_s"),
                )
            )
            continue

        try:
            X_test = _single_output_prediction_inputs(
                n_test=n_test,
                d=int(dataset.X_train.shape[1]),
                data_config=data_config,
                dataset=dataset,
            )

            def _predict_size():
                return gp.predict(X_test, variance_method=prediction_mode, return_std=True)

            first_start = time.perf_counter()
            _, first_memory_stats = measure_gpu_phase(_predict_size, interval=0.02)
            first_time_s = float(time.perf_counter() - first_start)
            first_backend_predict = dict(getattr(gp, "_backend_predict_info", {}) or {})
            repeat_times: list[float] = []
            for _ in range(repeat_count):
                _, repeat_time_s = _time_prediction_call(_predict_size)
                repeat_times.append(repeat_time_s)
            rows.append(
                prediction_x_test_scaling_entry(
                    spec=spec,
                    timing_quality="prepared_cache_split",
                    cache_used=True,
                    first_apply_time_s=first_time_s,
                    repeat_times_s=repeat_times,
                    prediction_peak_gpu_mb=first_memory_stats.get("phase_peak_gpu_mb"),
                    prediction_delta_gpu_mb=first_memory_stats.get("phase_delta_gpu_mb"),
                    mean_time_s=first_backend_predict.get("prediction_mean_time_s"),
                    variance_time_s=first_backend_predict.get("prediction_variance_time_s"),
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
                    cache_used=True,
                )
            )
    return rows


def _measure_gpytorch_single_prediction_x_test_scaling(
    *,
    train_result: dict[str, object],
    pred_result: dict[str, object],
    dataset: SyntheticDataset,
    data_config: dict[str, object],
    prediction_mode: str,
    benchmark_variety: str,
    tier: str,
    mode: str,
    prediction_solver_profile: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    repeat_count = _prediction_repeat_count_for_case(
        framework="gpytorch",
        method="matrix_free" if mode == "keops" else "materialized",
        prediction_mode=prediction_mode,
    )
    specs = prediction_x_test_target_specs(
        variety=benchmark_variety,
        tier=tier,
        framework="gpytorch",
    )
    if _prediction_x_test_scaling_policy(
        framework="gpytorch",
        method="matrix_free" if mode == "keops" else "materialized",
        prediction_mode=prediction_mode,
    ) == "canonical_only_matrix_free_exact":
        canonical_n_test = int(dataset.X_test.shape[0])
        specs = [spec for spec in specs if int(spec["n_test"]) == canonical_n_test]
    for spec in specs:
        n_test = int(spec["n_test"])
        if n_test == int(dataset.X_test.shape[0]):
            first_pred = pred_result
            X_test = np.asarray(dataset.X_test, dtype=np.float32)
        else:
            X_test = _single_output_prediction_inputs(
                n_test=n_test,
                d=int(dataset.X_train.shape[1]),
                data_config=data_config,
                dataset=dataset,
            )
            first_pred = predict_gpytorch_single_output(
                train_result,
                X_test,
                mode=mode,
                cg_tolerance=prediction_solver_profile["cg_tolerance"],
                max_cg_iterations=prediction_solver_profile["max_cg_iterations"],
                max_preconditioner_size=prediction_solver_profile[
                    "max_preconditioner_size"
                ],
                max_lanczos_quadrature_iterations=prediction_solver_profile[
                    "max_lanczos_quadrature_iterations"
                ],
                min_preconditioning_size=prediction_solver_profile[
                    "min_preconditioning_size"
                ],
                max_root_decomposition_size=prediction_solver_profile[
                    "max_root_decomposition_size"
                ],
                use_love=prediction_mode == "love",
            )
        repeat_times: list[float] = []
        for _ in range(repeat_count):
            repeat_pred = predict_gpytorch_single_output(
                train_result,
                X_test,
                mode=mode,
                cg_tolerance=prediction_solver_profile["cg_tolerance"],
                max_cg_iterations=prediction_solver_profile["max_cg_iterations"],
                max_preconditioner_size=prediction_solver_profile[
                    "max_preconditioner_size"
                ],
                max_lanczos_quadrature_iterations=prediction_solver_profile[
                    "max_lanczos_quadrature_iterations"
                ],
                min_preconditioning_size=prediction_solver_profile[
                    "min_preconditioning_size"
                ],
                max_root_decomposition_size=prediction_solver_profile[
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


@lru_cache(maxsize=None)
def _measure_mojogp_jit_startup_profile(
    d: int, training_method: str, ard: bool = False
) -> dict[str, float]:
    from mojogp.codegen_engine.compiler import compile_kernel

    kernel_node = SingleOutputGP(RBF(ard=ard))._to_kernel_node()
    original_cache = os.environ.get("MOJOGP_JIT_CACHE_DIR")
    # This profiles the training module that fit() actually uses. Prediction-only
    # NCOLS hints change ARD schedule selection and must not be mixed into the
    # training startup metric.
    ncols_hint = None

    try:
        with tempfile.TemporaryDirectory(prefix="mojogp_jit_scaling_probe_") as cache_root:
            os.environ["MOJOGP_JIT_CACHE_DIR"] = cache_root

            cold_start = time.perf_counter()
            compile_kernel(
                kernel_node,
                d,
                mode="fn_ptr",
                ncols_hint=ncols_hint,
                force_recompile=False,
                verbose=False,
            )
            cold_compile_s = time.perf_counter() - cold_start

            warm_start = time.perf_counter()
            compile_kernel(
                kernel_node,
                d,
                mode="fn_ptr",
                ncols_hint=ncols_hint,
                force_recompile=False,
                verbose=False,
            )
            warm_cache_hit_s = time.perf_counter() - warm_start
    finally:
        if original_cache is None:
            os.environ.pop("MOJOGP_JIT_CACHE_DIR", None)
        else:
            os.environ["MOJOGP_JIT_CACHE_DIR"] = original_cache

    return {
        "startup_compile_time_s": float(cold_compile_s),
        "startup_warm_cache_hit_s": float(warm_cache_hit_s),
    }


def _measure_mojogp_jit_startup_profile_with_specialization(
    d: int,
    training_method: str,
    specialization_request: dict[str, object] | None,
    ard: bool = False,
) -> dict[str, float]:
    if specialization_request in (None, {}, {"mode": "disabled", "profile": None}):
        return _measure_mojogp_jit_startup_profile(d, training_method, ard)

    from mojogp import SingleOutputGP, RBF
    from mojogp.codegen_engine.compiler import compile_kernel
    from mojogp.specialization import (
        SpecializationRequest,
        build_single_output_descriptor,
        default_specialization_registry,
        translate_compile_inputs,
    )

    kernel_node = SingleOutputGP(RBF(ard=ard))._to_kernel_node()
    original_cache = os.environ.get("MOJOGP_JIT_CACHE_DIR")
    request = SpecializationRequest.from_dict(specialization_request)
    descriptor = build_single_output_descriptor(
        kernel=kernel_node,
        dim=d,
        training_method=training_method,
    )
    decision = default_specialization_registry(
        materialized_predict_ncols_hint=tuple(_DEFAULT_PREDICT_NCOLS_HINT)
    ).resolve(descriptor, request)
    translation = translate_compile_inputs(decision)
    resolved_ncols = translation.ncols_hint

    try:
        with tempfile.TemporaryDirectory(prefix="mojogp_jit_scaling_probe_") as cache_root:
            os.environ["MOJOGP_JIT_CACHE_DIR"] = cache_root

            cold_start = time.perf_counter()
            compile_kernel(
                kernel_node,
                d,
                mode="fn_ptr",
                schedule_overrides=translation.schedule_overrides,
                ncols_hint=resolved_ncols,
                module_suffix=translation.module_suffix,
                force_recompile=False,
                verbose=False,
            )
            cold_compile_s = time.perf_counter() - cold_start

            warm_start = time.perf_counter()
            compile_kernel(
                kernel_node,
                d,
                mode="fn_ptr",
                schedule_overrides=translation.schedule_overrides,
                ncols_hint=resolved_ncols,
                module_suffix=translation.module_suffix,
                force_recompile=False,
                verbose=False,
            )
            warm_cache_hit_s = time.perf_counter() - warm_start
    finally:
        if original_cache is None:
            os.environ.pop("MOJOGP_JIT_CACHE_DIR", None)
        else:
            os.environ["MOJOGP_JIT_CACHE_DIR"] = original_cache

    return {
        "startup_compile_time_s": float(cold_compile_s),
        "startup_warm_cache_hit_s": float(warm_cache_hit_s),
    }


def _single_output_solver_profile(
    method: str, baseline_backend: str, framework: str
) -> dict[str, object]:
    profile = dict(FAIR_SINGLE_OUTPUT_SOLVER)
    profile["max_preconditioner_size"] = 0
    profile["precond_rank"] = 0
    profile["precond_method"] = 0
    profile["precond_family"] = "disabled"
    return profile


def _prediction_solver_profile(
    solver_profile: dict[str, object], prediction_mode: str
) -> dict[str, object]:
    _ = prediction_mode
    return dict(solver_profile)


def _benchmark_variety() -> str:
    variety = os.environ.get("MOJOGP_BENCHMARK_VARIETY")
    if variety is not None:
        normalized = variety.strip().lower()
        if normalized not in SCALING_DIMS:
            raise ValueError(
                "MOJOGP_BENCHMARK_VARIETY must be one of "
                f"{sorted(SCALING_DIMS)}, got '{variety}'"
            )
        return normalized
    config = _active_scaling_config()
    if "benchmark_variety" in config:
        normalized = str(config["benchmark_variety"]).strip().lower()
        if normalized not in SCALING_DIMS:
            raise ValueError(
                "scaling config benchmark_variety must be one of "
                f"{sorted(SCALING_DIMS)}, got '{normalized}'"
            )
        return normalized
    if _quick_enabled():
        return "minimal"
    return "standard"


def _route_tier_and_policy(method: str) -> tuple[str, str]:
    if method == "materialized":
        return get_vram_tier(), "vram"
    return get_matrix_free_capability_tier(), "bandwidth"


def _benchmark_targets(
    method: str,
    *,
    benchmark_variety: str | None = None,
) -> tuple[list[int], list[int], str, str]:
    variety = benchmark_variety or _benchmark_variety()
    tier, n_selection_policy = _route_tier_and_policy(method)
    if method == "materialized":
        dims = list(MATERIALIZED_SCALING_DIMS[variety])
        sizes = list(MATERIALIZED_TARGETS[variety][tier])
    else:
        dims = list(SCALING_DIMS[variety])
        sizes = list(MATRIX_FREE_TARGETS[variety][tier])
    sizes = _configured_scaling_sizes(method, sizes, tier=tier)
    dims = _configured_scaling_dims(method, dims)
    return sizes, dims, tier, n_selection_policy


def _framework_sizes(
    *,
    framework: str,
    method: str,
    prediction_mode: str,
    tier: str,
    benchmark_variety: str,
) -> list[int]:
    configured_default: list[int] | None = None
    if framework == "gpytorch" and method == "matrix_free" and prediction_mode == "exact":
        configured_default = list(MATRIX_FREE_EXACT_TARGETS[benchmark_variety][tier])
        return _configured_scaling_sizes(method, configured_default, tier=tier)
    if framework == "mojogp" and method == "matrix_free" and prediction_mode == "exact":
        configured_default = list(MATRIX_FREE_EXACT_TARGETS[benchmark_variety][tier])
        return _configured_scaling_sizes(method, configured_default, tier=tier)
    if method == "materialized":
        if framework == "gpytorch":
            configured_default = list(GPYTORCH_MATERIALIZED_TARGETS[benchmark_variety][tier])
            return _configured_scaling_sizes(method, configured_default, tier=tier)
        configured_default = list(MATERIALIZED_TARGETS[benchmark_variety][tier])
        return _configured_scaling_sizes(method, configured_default, tier=tier)
    configured_default = list(MATRIX_FREE_TARGETS[benchmark_variety][tier])
    return _configured_scaling_sizes(method, configured_default, tier=tier)


def _prediction_modes_for_framework(*, framework: str, method: str) -> list[str]:
    return ["exact", "love"]


def _framework_prediction_mode_pairs(method: str) -> list[tuple[str, str]]:
    if method == "matrix_free":
        # Run the published fair KeOps/LOVE lane before the exact lane. Exact
        # matrix-free prediction can be intentionally much slower and should not
        # prevent the fair LOVE comparison from being persisted if it later hits
        # a timeout.
        return [
            ("mojogp", "love"),
            ("gpytorch", "love"),
            ("gpytorch", "exact"),
            ("mojogp", "exact"),
        ]
    return [
        (framework, prediction_mode)
        for framework in ("mojogp", "gpytorch")
        for prediction_mode in _prediction_modes_for_framework(
            framework=framework,
            method=method,
        )
    ]


@dataclass(frozen=True)
class ScalingCaseSpec:
    framework: str
    prediction_mode: str
    mojogp_solver_policy: str = "strict_fair"
    case_variant: str | None = None
    comparison_mojogp_case_variant: str | None = None


def _optional_case_token(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized in {"", "none", "default", "null"}:
        return None
    return normalized


def _parse_scaling_case_spec(entry: str) -> ScalingCaseSpec:
    parts = [part.strip() for part in entry.split(":")]
    if len(parts) < 2 or len(parts) > 5:
        raise ValueError(
            "MOJOGP_SCALING_CASES entries must use "
            "framework:prediction_mode[:mojogp_solver_policy[:case_variant[:comparison_mojogp_case_variant]]]"
        )
    framework = parts[0]
    prediction_mode = parts[1]
    mojogp_solver_policy = parts[2] if len(parts) >= 3 and parts[2] else "strict_fair"
    if framework not in {"mojogp", "gpytorch"}:
        raise ValueError(f"Unsupported scaling framework '{framework}'")
    if prediction_mode not in {"exact", "love"}:
        raise ValueError(f"Unsupported scaling prediction mode '{prediction_mode}'")
    if mojogp_solver_policy not in {"strict_fair", "route_default"}:
        raise ValueError(
            "mojogp_solver_policy must be 'strict_fair' or 'route_default', "
            f"got '{mojogp_solver_policy}'"
        )
    return ScalingCaseSpec(
        framework=framework,
        prediction_mode=prediction_mode,
        mojogp_solver_policy=mojogp_solver_policy,
        case_variant=_optional_case_token(parts[3] if len(parts) >= 4 else None),
        comparison_mojogp_case_variant=_optional_case_token(
            parts[4] if len(parts) >= 5 else None
        ),
    )


def _scaling_case_spec_from_config(item: object) -> ScalingCaseSpec:
    if not isinstance(item, dict):
        raise ValueError("scaling config 'cases' entries must be objects")
    framework = str(item.get("framework", ""))
    prediction_mode = str(item.get("prediction_mode", ""))
    mojogp_solver_policy = str(item.get("mojogp_solver_policy", "strict_fair"))
    if framework not in {"mojogp", "gpytorch"}:
        raise ValueError(f"Unsupported scaling framework '{framework}'")
    if prediction_mode not in {"exact", "love"}:
        raise ValueError(f"Unsupported scaling prediction mode '{prediction_mode}'")
    if mojogp_solver_policy not in {"strict_fair", "route_default"}:
        raise ValueError(
            "mojogp_solver_policy must be 'strict_fair' or 'route_default', "
            f"got '{mojogp_solver_policy}'"
        )
    return ScalingCaseSpec(
        framework=framework,
        prediction_mode=prediction_mode,
        mojogp_solver_policy=mojogp_solver_policy,
        case_variant=_optional_case_token(
            None if item.get("case_variant") is None else str(item.get("case_variant"))
        ),
        comparison_mojogp_case_variant=_optional_case_token(
            None
            if item.get("comparison_mojogp_case_variant") is None
            else str(item.get("comparison_mojogp_case_variant"))
        ),
    )


def _scaling_case_specs(method: str) -> list[ScalingCaseSpec]:
    raw = os.environ.get("MOJOGP_SCALING_CASES")
    if raw is None or raw.strip() == "":
        config = _active_scaling_config()
        cases = config.get("cases")
        if cases is not None:
            if not isinstance(cases, list) or not cases:
                raise ValueError("scaling config 'cases' must be a non-empty list")
            return [_scaling_case_spec_from_config(item) for item in cases]
        return [
            ScalingCaseSpec(framework=framework, prediction_mode=prediction_mode)
            for framework, prediction_mode in _framework_prediction_mode_pairs(method)
        ]
    specs = [
        _parse_scaling_case_spec(entry)
        for entry in raw.split(",")
        if entry.strip()
    ]
    if not specs:
        raise ValueError("MOJOGP_SCALING_CASES cannot be empty")
    return specs


def _int_csv_env(name: str) -> list[int] | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive, got {values}")
    return values


def _configured_scaling_sizes(method: str, default: list[int], *, tier: str | None = None) -> list[int]:
    method_key = method.upper()
    override = _int_csv_env(f"MOJOGP_{method_key}_SCALING_N_VALUES")
    if override is None:
        override = _int_csv_env("MOJOGP_SCALING_N_VALUES")
    if override is None:
        config = _active_scaling_config()
        method_values = config.get("n_values_by_method")
        if isinstance(method_values, dict) and method in method_values:
            value = method_values[method]
            if not isinstance(value, list):
                raise ValueError(f"scaling config n_values_by_method.{method} must be a list")
            override = [int(item) for item in value]
        tier_values = config.get("n_values_by_tier")
        if override is None and isinstance(tier_values, dict) and tier is not None:
            if tier in tier_values:
                value = tier_values[tier]
                if not isinstance(value, list):
                    raise ValueError(f"scaling config n_values_by_tier.{tier} must be a list")
                override = [int(item) for item in value]
        values = config.get("n_values")
        if override is None and values is not None:
            if not isinstance(values, list):
                raise ValueError("scaling config 'n_values' must be a list")
            override = [int(item) for item in values]
    return list(default if override is None else override)


def _configured_scaling_dims(method: str, default: list[int]) -> list[int]:
    method_key = method.upper()
    override = _int_csv_env(f"MOJOGP_{method_key}_SCALING_DIMS")
    if override is None:
        override = _int_csv_env("MOJOGP_SCALING_DIMS")
    if override is None:
        config = _active_scaling_config()
        method_values = config.get("dims_by_method")
        if isinstance(method_values, dict) and method in method_values:
            value = method_values[method]
            if not isinstance(value, list):
                raise ValueError(f"scaling config dims_by_method.{method} must be a list")
            override = [int(item) for item in value]
        values = config.get("dims")
        if override is None and values is not None:
            if not isinstance(values, list):
                raise ValueError("scaling config 'dims' must be a list")
            override = [int(item) for item in values]
    return list(default if override is None else override)


def _configured_scaling_max_iterations(default: int = 100) -> int:
    raw = os.environ.get("MOJOGP_SCALING_MAX_ITERATIONS")
    if raw is not None and raw.strip() != "":
        return int(raw)
    config = _active_scaling_config()
    if "max_iterations" in config:
        return int(config["max_iterations"])
    return default


def _configured_scaling_enable_early_stopping(default: bool = False) -> bool:
    raw = os.environ.get("MOJOGP_SCALING_ENABLE_EARLY_STOPPING")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    config = _active_scaling_config()
    if "enable_early_stopping" in config:
        return bool(config["enable_early_stopping"])
    return default


def _configured_scaling_track(default: str = "scaling") -> str:
    config = _active_scaling_config()
    return str(config.get("benchmark_track", default))


def _configured_n_selection_policy(default: str | None) -> str | None:
    config = _active_scaling_config()
    if "n_selection_policy" in config:
        return str(config["n_selection_policy"])
    name = _active_scaling_config_name()
    if name is not None:
        return f"config:{name}"
    return default


def _quick_enabled() -> bool:
    return os.environ.get("MOJOGP_BENCHMARK_QUICK", "0") == "1"


def _size_role(n_train: int, sizes: list[int]) -> str:
    if not sizes:
        return "unknown"
    if n_train not in sizes:
        return "custom"
    index = sizes.index(n_train)
    if len(sizes) == 1 or index < len(sizes) - 1:
        return f"anchor_{index + 1}"
    return "envelope"


def _allow_recorded_failure(
    *,
    framework: str,
    method: str,
    prediction_mode: str,
    benchmark_variety: str,
    size_role: str,
) -> bool:
    if framework != "gpytorch" or benchmark_variety not in {"standard", "extensive", "matrix_free_n_scaling"}:
        return False
    if method == "matrix_free" and prediction_mode in {"exact", "love"}:
        return True
    return False


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _cg_telemetry_from_history(history: np.ndarray | None) -> dict[str, object]:
    if history is None or len(history) == 0:
        return {
            "measured": False,
            "stage": "training",
            "timing_basis": "per_optimizer_iteration",
            "cg_iterations_history": [],
            "cg_iterations_total": 0,
            "cg_iterations_mean": 0.0,
            "cg_iterations_max": 0,
            "cg_iterations_final_step": 0,
        }
    history = np.asarray(history, dtype=np.int32)
    return {
        "measured": True,
        "stage": "training",
        "timing_basis": "per_optimizer_iteration",
        "cg_iterations_history": history.tolist(),
        "cg_iterations_total": int(np.sum(history)),
        "cg_iterations_mean": float(np.mean(history)),
        "cg_iterations_max": int(np.max(history)),
        "cg_iterations_final_step": int(history[-1]),
    }


def _cg_telemetry_from_predict_info(
    predict_info: dict[str, object] | None,
    *,
    prediction_mode: str,
) -> dict[str, object]:
    info = dict(predict_info or {})
    prediction_uses_cg = bool(
        info.get(
            "configured_for_cg",
            prediction_mode == "exact"
            and info.get("backend_variance_used", False)
            and info.get("actual_variance_route") is not None,
        )
    )
    return {
        "telemetry_quality": str(
            info.get(
                "telemetry_quality",
                "observed" if prediction_uses_cg else "not_applicable",
            )
        ),
        "stage": "prediction",
        "timing_basis": "diagnostic_not_aligned_to_warm_repeated_timing",
        "configured_for_cg": prediction_uses_cg,
        "observed_cg_calls": bool(
            info.get("observed_cg_calls", prediction_uses_cg)
        ),
    }


def _single_output_benchmark_metadata(
    *,
    framework: str,
    method: str,
    prediction_mode: str,
    n_train: int,
    tier: str,
    benchmark_variety: str,
    kernel: str,
    baseline_backend: str,
    keops_supported: bool,
    keops_used: bool,
    requested_backend: str | None = None,
    effective_training_backend: str | None = None,
    effective_prediction_backend: str | None = None,
    backend_fallback_used: bool = False,
    backend_fallback_reason: str | None = None,
) -> dict[str, object]:
    policy = policy_for("scaling_certification")
    published_gpytorch_matrix_free_modes = set(
        _prediction_modes_for_framework(framework="gpytorch", method=method)
    )
    gpytorch_sizes = set(
        _framework_sizes(
            framework="gpytorch",
            method=method,
            prediction_mode=prediction_mode,
            tier=tier,
            benchmark_variety=benchmark_variety,
        )
    )
    fair_match = n_train in gpytorch_sizes and (
        framework != "mojogp"
        or method != "matrix_free"
        or prediction_mode in published_gpytorch_matrix_free_modes
    )
    requested_backend = requested_backend or baseline_backend
    effective_training_backend = effective_training_backend or baseline_backend
    effective_prediction_backend = effective_prediction_backend or baseline_backend
    matrix_free_strict_keops_required = method == "matrix_free" and policy.strict_keops_required
    strict_keops_verified = (
        requested_backend == "keops"
        and effective_training_backend == "keops"
        and effective_prediction_backend == "keops"
        and not backend_fallback_used
    )
    if framework == "mojogp":
        if not fair_match:
            comparison_class = "mojogp_only_scale"
        elif matrix_free_strict_keops_required and not strict_keops_verified:
            comparison_class = "mojogp_only_scale"
        else:
            comparison_class = "fair_match"
    elif matrix_free_strict_keops_required and not strict_keops_verified:
        comparison_class = "unsupported_comparator"
    else:
        comparison_class = "fair_match"

    if comparison_class == "mojogp_only_scale":
        if method == "matrix_free" and prediction_mode == "exact":
            fairness_note = (
                "N.B. MojoGP-only matrix-free exact row: the published GPyTorch "
                "matrix-free comparator is LOVE-only because raw KeOps exact "
                "predictive variances are not robust at the benchmark-trained state."
            )
        else:
            fairness_note = (
                "N.B. MojoGP-only envelope row: no direct GPyTorch comparator is "
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
                "The row keeps the same optimizer policy as the fair benchmark lane.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "The row keeps the same CG and Lanczos budgets as the fair benchmark lane.",
            ),
            "preconditioner": _fairness_axis(
                "aligned",
                "Benchmark policy disables preconditioning on the published comparison lane.",
            ),
            "prediction_mode": _fairness_axis(
                "mojogp_only",
                "This prediction route is not part of the published cross-framework comparison lane.",
            ),
            "telemetry": _fairness_axis(
                "observed", "MojoGP CG telemetry is observed for this row."
            ),
        }
    elif comparison_class == "unsupported_comparator":
        fallback_clause = ""
        if backend_fallback_reason:
            fallback_clause = f" Fallback reason: {backend_fallback_reason}."
        fairness_note = (
            "N.B. Unsupported comparator row: strict matrix-free publication requires "
            "requested KeOps plus effective KeOps for both training and prediction, so this row is recorded for "
            "debugging only and is not part of the fair published comparison set."
            f"{fallback_clause}"
        )
        fairness_axes = {
            "comparator_scope": _fairness_axis(
                "unsupported",
                "This row is not publishable as a fair matrix-free comparison because the comparator is not strict KeOps.",
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
                "unsupported", "Matrix-free publication is blocked before preconditioner alignment matters."
            ),
            "prediction_mode": _fairness_axis(
                "unsupported", "This row is not part of the published matrix-free comparison lane."
            ),
            "telemetry": _fairness_axis(
                "observed", "The active CG path is observed in the exported telemetry."
            ),
        }
    elif method == "matrix_free" and strict_keops_verified:
        fairness_note = (
            "Fair strict-KeOps matrix-free row: n, initialization, optimizer, CG budgets, "
            f"and post-fit {prediction_mode} prediction mode are aligned, and the comparator backend remains on KeOps throughout the published lane."
        )
        fairness_axes = {
            "comparator_scope": _fairness_axis(
                "aligned",
                "This is the published cross-framework matrix-free comparison row.",
            ),
            "sample_count_n": _fairness_axis(
                "aligned", "Both frameworks run at the same n."
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "Optimizer family, iteration budget, learning rate, and schedule are aligned.",
            ),
            "initialization": _fairness_axis(
                "aligned",
                "Both frameworks use initial_noise=0.1 and all-ones ARD lengthscales when ARD is enabled.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "CG tolerance, iteration cap, trace-sample count, and Lanczos budget are aligned.",
            ),
            "preconditioner": _fairness_axis(
                "aligned",
                "Benchmark policy disables preconditioning on the published comparison lane.",
            ),
            "prediction_mode": _fairness_axis(
                "aligned",
                f"Both sides use post-fit {prediction_mode} prediction on the published matrix-free comparison lane.",
            ),
            "telemetry": _fairness_axis(
                "observed", "The active CG path is observed in the exported telemetry."
            ),
        }
    else:
        fairness_note = (
            "Fair cross-framework row: n, initialization, optimizer, CG budgets, prediction mode, "
            "and disabled-preconditioner benchmark policy are aligned."
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
            "initialization": _fairness_axis(
                "aligned",
                "Both frameworks use initial_noise=0.1 and all-ones ARD lengthscales when ARD is enabled.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "CG tolerance, iteration cap, trace-sample count, and Lanczos budget are aligned.",
            ),
            "preconditioner": _fairness_axis(
                "aligned", "Benchmark policy disables preconditioning on the published comparison lane."
            ),
            "prediction_mode": _fairness_axis(
                "aligned", "Both sides run the same published prediction mode."
            ),
            "telemetry": _fairness_axis(
                "observed", "The active CG path is observed in the exported telemetry."
            ),
        }

    return {
        "comparison_class": comparison_class,
        "baseline_backend": "none" if framework == "mojogp" else baseline_backend,
        "keops_supported": keops_supported,
        "keops_used": keops_used,
        "requested_backend": requested_backend,
        "effective_training_backend": effective_training_backend,
        "effective_prediction_backend": effective_prediction_backend,
        "backend_fallback_used": backend_fallback_used,
        "backend_fallback_reason": backend_fallback_reason,
        "kernel_backend_label": f"{kernel}:{baseline_backend}",
        "fairness_note": fairness_note,
        "fairness_axes": fairness_axes,
    }


def _solver_policy_metadata(
    metadata: dict[str, object],
    *,
    framework: str,
    mojogp_solver_policy: str,
) -> dict[str, object]:
    if framework != "mojogp" or mojogp_solver_policy == "strict_fair":
        return metadata

    adjusted = dict(metadata)
    axes = dict(adjusted.get("fairness_axes", {}) or {})
    axes["preconditioner"] = _fairness_axis(
        "near_fair",
        "MojoGP leaves preconditioner controls to the production route defaults; the GPyTorch comparator uses the strict no-preconditioner KeOps lane.",
    )
    axes["mojogp_solver_policy"] = _fairness_axis(
        "mojogp_default",
        f"MojoGP solver policy is '{mojogp_solver_policy}', so route-specific defaults remain enabled.",
    )
    if adjusted.get("comparison_class") == "fair_match":
        adjusted["comparison_class"] = "near_fair_mojogp_default"
        adjusted["fairness_note"] = (
            "N.B. Near-fair MojoGP production-default row: n, optimizer, CG tolerance, "
            "probe count, and prediction mode match the strict lane, but MojoGP leaves "
            "preconditioner controls unset so route-specific production defaults remain "
            "enabled. Compare against the strict MojoGP row for an exactly no-preconditioner "
            "MojoGP/GPyTorch KeOps lane."
        )
    else:
        adjusted["fairness_note"] = (
            str(adjusted.get("fairness_note", ""))
            + " MojoGP route-specific solver defaults remain enabled for this row."
        ).strip()
    adjusted["fairness_axes"] = axes
    return adjusted


def _single_output_route_seed(*, n_train: int, d: int) -> int:
    return 100 + n_train + d * 1000


def _single_output_learning_rate() -> float:
    return 0.03


def _single_output_gpytorch_backend(method: str, kernel: str) -> dict[str, object]:
    keops_supported = kernel in keops_supported_kernels()
    keops_used = method == "matrix_free" and keops_supported and is_keops_available()
    requested_backend = "keops" if keops_used else "standard"
    effective_prediction_backend = requested_backend
    backend_fallback_used = False
    backend_fallback_reason = None
    return {
        "baseline_backend": effective_prediction_backend,
        "keops_supported": keops_supported,
        "keops_used": keops_used,
        "mode": "keops" if keops_used else "cg",
        "requested_backend": requested_backend,
        "effective_training_backend": requested_backend,
        "effective_prediction_backend": effective_prediction_backend,
        "backend_fallback_used": backend_fallback_used,
        "backend_fallback_reason": backend_fallback_reason,
    }


def _prediction_materialization_label(mode: int | None) -> str:
    if mode == 0:
        return "none"
    if mode == 1:
        return "continuous_train_train"
    if mode == 2:
        return "mixed_train_train"
    return "unknown"


def _single_output_prediction_route_metadata(
    *,
    method: str,
    prediction_mode: str,
    n_train: int,
    n_test: int,
    backend_train: dict[str, object],
    backend_predict: dict[str, object],
    cache_info: dict[str, object],
    pred_memory_stats: dict[str, float],
) -> dict[str, object]:
    """Record direct evidence for the post-fit prediction route.

    The cross-framework metadata names the comparator backend. These fields name
    the MojoGP provider route itself so matrix-free rows can be audited without
    inferring behavior from timings alone.
    """

    prediction_method = str(backend_predict.get("prediction_method", method))
    cache_method = str(cache_info.get("prediction_cache_method", prediction_method))
    training_route = str(backend_train.get("training_route", method))
    materialization_mode = int(backend_train.get("materialization_mode", 0) or 0)
    materialization_label = _prediction_materialization_label(materialization_mode)
    train_train_materialized = (
        materialization_mode != 0 or prediction_method.startswith("materialized")
    )
    matrix_free_prediction_verified = (
        method == "matrix_free"
        and training_route == "matrix_free"
        and prediction_method == "matrix_free"
        and cache_method == "matrix_free"
        and not train_train_materialized
    )
    if method == "matrix_free" and prediction_mode == "love":
        memory_contract = "O(n_train * rank + n_test * rank)"
    elif method == "matrix_free" and prediction_mode == "exact":
        memory_contract = "O(n_train * n_test_active_block)"
    elif method == "matrix_free":
        memory_contract = "O(n_train * rhs_cols)"
    else:
        memory_contract = "O(n_train^2) materialized train kernel allowed"

    cross_strategy = None
    if prediction_mode == "love":
        cross_strategy = backend_predict.get("love_cross_strategy")
    elif prediction_mode == "exact":
        cross_strategy = backend_predict.get("exact_cross_mode")

    evidence = {
        "training_route": training_route,
        "prediction_method": prediction_method,
        "prediction_cache_method": cache_method,
        "actual_prediction_route": backend_predict.get("actual_prediction_route"),
        "actual_variance_route": backend_predict.get("actual_variance_route"),
        "prediction_cache_variance_method": cache_info.get(
            "prediction_cache_variance_method", prediction_mode
        ),
        "prediction_cache_used": bool(backend_predict.get("prediction_cache_used")),
        "prediction_cache_rank": int(
            cache_info.get("prediction_cache_rank", cache_info.get("rank", 0)) or 0
        ),
        "prediction_train_train_materialization_mode": materialization_mode,
        "prediction_train_train_materialization_label": materialization_label,
        "prediction_train_train_materialized": train_train_materialized,
        "prediction_cross_covariance_strategy": cross_strategy,
        "prediction_phase_peak_gpu_mb": pred_memory_stats.get("phase_peak_gpu_mb"),
        "prediction_phase_delta_gpu_mb": pred_memory_stats.get("phase_delta_gpu_mb"),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "memory_contract": memory_contract,
        "matrix_free_prediction_verified": matrix_free_prediction_verified,
    }
    return {
        **evidence,
        "prediction_route_evidence": evidence,
        "matrix_free_prediction_evidence": evidence
        if method == "matrix_free"
        else None,
    }


def _cg_telemetry_quality(cg_telemetry: dict[str, object]) -> dict[str, object]:
    if "training" in cg_telemetry or "prediction" in cg_telemetry:
        training = dict(cg_telemetry.get("training", {}))
        prediction = dict(cg_telemetry.get("prediction", {}))
        return {
            "training": training.get(
                "telemetry_quality",
                "observed" if training.get("measured") else "missing",
            ),
            "prediction": prediction.get(
                "telemetry_quality",
                "observed" if prediction.get("measured") else "not_applicable",
            ),
            "configured_for_cg": bool(
                training.get(
                    "configured_for_cg", training.get("measured", False)
                )
                or prediction.get(
                    "configured_for_cg", prediction.get("measured", False)
                )
            ),
            "observed_cg_calls": bool(
                training.get(
                    "observed_cg_calls", training.get("cg_iterations_total", 0) > 0
                )
                or prediction.get(
                    "observed_cg_calls",
                    prediction.get("cg_iterations_total", 0) > 0,
                )
            ),
        }
    return {
        "training": cg_telemetry.get(
            "telemetry_quality",
            "observed" if cg_telemetry.get("measured") else "missing",
        ),
        "prediction": cg_telemetry.get("telemetry_quality", "not_applicable"),
        "configured_for_cg": bool(
            cg_telemetry.get("configured_for_cg", cg_telemetry.get("measured", False))
        ),
        "observed_cg_calls": bool(
            cg_telemetry.get(
                "observed_cg_calls",
                cg_telemetry.get("cg_iterations_total", 0) > 0,
            )
        ),
    }


def _preconditioner_config(
    solver_config: dict[str, object], *, framework: str
) -> dict[str, object]:
    return {
        "family": solver_config.get("precond_family", "pivoted_cholesky"),
        "framework": framework,
        "rank": solver_config.get("precond_rank"),
        "method": solver_config.get("precond_method", "default"),
        "rebuild_threshold": solver_config.get("precond_rebuild_threshold"),
    }


def _single_output_ard_config(
    *,
    dataset: SyntheticDataset,
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
    training_time_s: float,
    prediction_mean_time_s: float,
    prediction_variance_time_s: float,
    training_iter_times_ms: list[float] | None,
    startup_profile: dict[str, float] | None,
    iterations_run: int,
    max_iterations: int,
    early_stopped: bool,
    memory_stats: dict[str, float],
    mean: np.ndarray,
    variance: np.ndarray,
    learned_lengthscale: float,
    learned_noise: float,
    learned_outputscale: float,
    learned_mean: float | None,
    final_nll: float,
    optimizer_config: dict[str, object],
    training_solver_config: dict[str, object],
    prediction_solver_config: dict[str, object],
    cg_telemetry: dict[str, object],
    extra_config: dict[str, object],
    prediction_phase_times: dict[str, object] | None = None,
    prediction_x_test_scaling: list[dict[str, object]] | None = None,
    benchmark_name: str = "scaling_certification",
    kernel_label: str = "rbf",
) -> BenchmarkResult:
    std = np.sqrt(np.maximum(variance, 1e-10))
    vram_info = get_vram_info()
    timing_payload = _timing_payload(training_iter_times_ms)
    iter_timing_quality = (
        "direct_per_iteration"
        if timing_payload["iter_times_ms"] is not None
        else "derived_total_div_iterations"
    )
    startup_profile = startup_profile or {}
    prediction_phase_times = prediction_phase_times or {}
    hyperparameters = HyperparameterResult(
        learned_lengthscale=learned_lengthscale,
        learned_noise=learned_noise,
        learned_outputscale=learned_outputscale,
        learned_mean=learned_mean,
        final_nll=final_nll,
    )
    true_params = dict(getattr(dataset, "true_params", {}) or {})
    if "lengthscale" in true_params:
        hyperparameters.lengthscale_rel_error = param_relative_error(
            learned_lengthscale, float(true_params["lengthscale"])
        )
    if "noise" in true_params:
        hyperparameters.noise_rel_error = param_relative_error(
            learned_noise, float(true_params["noise"])
        )
    if "outputscale" in true_params:
        hyperparameters.outputscale_rel_error = param_relative_error(
            learned_outputscale, float(true_params["outputscale"])
        )
    if learned_mean is not None and "mean" in true_params:
        hyperparameters.mean_rel_error = param_relative_error(
            learned_mean, float(true_params["mean"])
        )

    return BenchmarkResult(
        config={
            "benchmark": benchmark_name,
            "framework": framework,
            "model_type": model_type,
            "kernel": kernel_label,
            "training_method": training_method,
            "method": method,
            "prediction_mode": prediction_mode,
            "training_objective": "exact_marginal_log_likelihood",
            "prediction_variance_method": prediction_mode,
            "prediction_mode_semantics": "post_fit_variance_route",
            "n": int(dataset.X_train.shape[0]),
            "d": int(dataset.X_train.shape[1]),
            "n_test": int(dataset.X_test.shape[0]),
            "vram_tier": vram_info["tier"],
            "vram_gb": vram_info["vram_gb"],
            "requested_max_materialized": vram_info["max_n_materialized"],
            "requested_max_matrix_free": vram_info["max_n_matrix_free"],
            "bandwidth_gbps": vram_info["bandwidth_gbps"],
            "bandwidth_tier": vram_info["bandwidth_tier"],
            "matrix_free_capability_tier": vram_info[
                "matrix_free_capability_tier"
            ],
            "tier_overrides": vram_info.get("tier_overrides", {}),
            "optimizer_config": optimizer_config,
            "training_solver_config": training_solver_config,
            "prediction_solver_config": prediction_solver_config,
            "preconditioner_config": _preconditioner_config(
                training_solver_config,
                framework=framework,
            ),
            "cg_telemetry": cg_telemetry,
            "cg_telemetry_quality": _cg_telemetry_quality(cg_telemetry),
            "iter_timing_quality": iter_timing_quality,
            **extra_config,
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
            prediction_mean_time_s=prediction_mean_time_s,
            prediction_variance_time_s=prediction_variance_time_s,
            end_to_end_time_s=(
                training_time_s + prediction_mean_time_s + prediction_variance_time_s
            ),
            iterations_run=iterations_run,
            max_iterations=max_iterations,
            early_stopped=early_stopped,
            ms_per_iteration=float(
                timing_payload["iter_time_median_ms"]
                if timing_payload["iter_time_median_ms"] is not None
                else (training_time_s / max(iterations_run, 1)) * 1000.0
            ),
            iter_time_min_ms=(
                None
                if timing_payload["iter_time_min_ms"] is None
                else float(timing_payload["iter_time_min_ms"])
            ),
            iter_time_q25_ms=(
                None
                if timing_payload["iter_time_q25_ms"] is None
                else float(timing_payload["iter_time_q25_ms"])
            ),
            iter_time_mean_ms=(
                None
                if timing_payload["iter_time_mean_ms"] is None
                else float(timing_payload["iter_time_mean_ms"])
            ),
            iter_time_median_ms=(
                None
                if timing_payload["iter_time_median_ms"] is None
                else float(timing_payload["iter_time_median_ms"])
            ),
            iter_time_q75_ms=(
                None
                if timing_payload["iter_time_q75_ms"] is None
                else float(timing_payload["iter_time_q75_ms"])
            ),
            iter_time_max_ms=(
                None
                if timing_payload["iter_time_max_ms"] is None
                else float(timing_payload["iter_time_max_ms"])
            ),
            iter_time_p5_ms=(
                None
                if timing_payload["iter_time_p5_ms"] is None
                else float(timing_payload["iter_time_p5_ms"])
            ),
            iter_time_p95_ms=(
                None
                if timing_payload["iter_time_p95_ms"] is None
                else float(timing_payload["iter_time_p95_ms"])
            ),
            iter_times_ms=(
                None
                if timing_payload["iter_times_ms"] is None
                else list(timing_payload["iter_times_ms"])
            ),
            iter_timing_quality=iter_timing_quality,
            startup_compile_time_s=startup_profile.get("startup_compile_time_s"),
            startup_warm_cache_hit_s=startup_profile.get(
                "startup_warm_cache_hit_s"
            ),
            startup_prepare_time_s=startup_profile.get("startup_prepare_time_s"),
            prediction_cold_first_time_s=_optional_float(
                prediction_phase_times.get("prediction_cold_first_time_s")
            ),
            prediction_cache_prepare_time_s=_optional_float(
                prediction_phase_times.get("prediction_cache_prepare_time_s")
            ),
            prediction_prepared_apply_time_s=_optional_float(
                prediction_phase_times.get("prediction_prepared_apply_time_s")
            ),
            prediction_repeated_median_time_s=_optional_float(
                prediction_phase_times.get("prediction_repeated_median_time_s")
            ),
            prediction_repeated_p5_time_s=_optional_float(
                prediction_phase_times.get("prediction_repeated_p5_time_s")
            ),
            prediction_repeated_p95_time_s=_optional_float(
                prediction_phase_times.get("prediction_repeated_p95_time_s")
            ),
            prediction_alpha_time_s=_optional_float(
                prediction_phase_times.get("prediction_alpha_time_s")
            ),
            prediction_love_root_time_s=_optional_float(
                prediction_phase_times.get("prediction_love_root_time_s")
            ),
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
            exact_prediction_peak_gpu_mb=memory_stats.get(
                "exact_prediction_peak_gpu_mb"
            ),
            exact_prediction_delta_gpu_mb=memory_stats.get(
                "exact_prediction_delta_gpu_mb"
            ),
            love_prediction_peak_gpu_mb=memory_stats.get(
                "love_prediction_peak_gpu_mb"
            ),
            love_prediction_delta_gpu_mb=memory_stats.get(
                "love_prediction_delta_gpu_mb"
            ),
        ),
        hyperparameters=hyperparameters,
    )


def _run_scaling_case(
    method: str,
    n_train: int,
    d: int,
    *,
    framework: str,
    prediction_mode: str,
    tier: str,
    benchmark_variety: str = "standard",
    benchmark_track: str = "scaling",
    n_selection_policy: str | None = None,
    size_role: str | None = None,
    max_iterations: int = 100,
    enable_early_stopping: bool = False,
    benchmark_name: str = "scaling_certification",
    mojogp_preset: str | None = None,
    data_options: dict[str, object] | None = None,
    specialization: dict[str, object] | None = None,
    dataset_path: str | None = None,
    results_dir: Path | None = None,
    ard: bool = False,
    relevant_dims: int | None = None,
    mojogp_solver_policy: str = "strict_fair",
    case_variant: str | None = None,
    comparison_mojogp_case_variant: str | None = None,
    timeout_s: int | None = None,
) -> BenchmarkResult:
    if os.environ.get("MOJOGP_SINGLE_OUTPUT_CHILD", "0") != "1":
        if results_dir is None:
            raise ValueError(
                "results_dir is required for single-output subprocess benchmark runs"
            )
        return run_single_output_scaling_subprocess(
            method=method,
            n_train=n_train,
            d=d,
            framework=framework,
            prediction_mode=prediction_mode,
            tier=tier,
            benchmark_variety=benchmark_variety,
            benchmark_track=benchmark_track,
            n_selection_policy=n_selection_policy,
            size_role=size_role,
            max_iterations=max_iterations,
            enable_early_stopping=enable_early_stopping,
            benchmark_name=benchmark_name,
            mojogp_preset=mojogp_preset,
            data_options=data_options,
            specialization=specialization,
            results_dir=results_dir,
            ard=ard,
            relevant_dims=relevant_dims,
            mojogp_solver_policy=mojogp_solver_policy,
            case_variant=case_variant,
            comparison_mojogp_case_variant=comparison_mojogp_case_variant,
            timeout_s=timeout_s,
        )

    case_start = time.perf_counter()
    _emit_scaling_phase(
        "case_start",
        framework=framework,
        method=method,
        prediction_mode=prediction_mode,
        n_train=n_train,
        d=d,
    )
    data_config_for_metadata: dict[str, object] = {}
    if dataset_path is not None:
        dataset_payload = load_dataset_artifact(dataset_path)
        dataset = SyntheticDataset(
            X_train=np.asarray(dataset_payload["X_train"], dtype=np.float32),
            y_train=np.asarray(dataset_payload["y_train"], dtype=np.float32),
            X_test=np.asarray(dataset_payload["X_test"], dtype=np.float32),
            y_test=np.asarray(dataset_payload["y_test"], dtype=np.float32),
            f_test=np.asarray(dataset_payload["f_test"], dtype=np.float32),
            true_params=dict(dataset_payload["true_params"]),
            name=str(dataset_payload["name"]),
            description=str(dataset_payload["description"]),
        )
    else:
        seed = _single_output_route_seed(n_train=n_train, d=d)
        np.random.seed(seed)
        data_config = {
            "dataset_family": "structured_function",
            "n_train": n_train,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "d": d,
            "function_type": "smooth",
            "noise_level": "medium",
            "seed": seed,
        }
        data_config.update(dict(data_options or {}))
        data_config_for_metadata = dict(data_config)
        if ard:
            data_config["dataset_family"] = "structured_ard"
            data_config["relevant_dims"] = int(
                relevant_dims or data_config.get("relevant_dims", min(3, d))
            )
        dataset_family = str(data_config.get("dataset_family", "structured_function"))
        if dataset_family == "structured_function":
            dataset = generate_structured_function_data(
                n_train=int(data_config["n_train"]),
                n_test=int(data_config["n_test"]),
                d=int(data_config["d"]),
                function_type=str(data_config.get("function_type", "smooth")),
                noise_level=str(data_config.get("noise_level", "medium")),
                seed=int(data_config["seed"]),
                mean_offset=float(data_config.get("mean_offset", 0.0)),
            )
        elif dataset_family == "structured_ard":
            dataset = generate_single_output_structured_ard_data(
                n_train=int(data_config["n_train"]),
                n_test=int(data_config["n_test"]),
                d=int(data_config["d"]),
                relevant_dims=int(data_config.get("relevant_dims", relevant_dims or min(3, d))),
                noise_level=str(data_config.get("noise_level", "medium")),
                seed=int(data_config["seed"]),
                mean_offset=float(data_config.get("mean_offset", 0.0)),
            )
        elif dataset_family == "gp_prior":
            x_range = data_config.get("x_range", (-3.0, 3.0))
            dataset = generate_gp_prior_data(
                n_train=int(data_config["n_train"]),
                n_test=int(data_config["n_test"]),
                d=int(data_config["d"]),
                kernel_type=str(data_config.get("kernel_type", "rbf")),
                true_lengthscale=float(data_config.get("true_lengthscale", 1.0)),
                true_noise=float(data_config.get("true_noise", 0.1)),
                true_outputscale=float(data_config.get("true_outputscale", 1.0)),
                seed=int(data_config["seed"]),
                x_range=(float(x_range[0]), float(x_range[1])),
                true_period=float(data_config.get("true_period", 1.0)),
                true_alpha=float(data_config.get("true_alpha", 1.0)),
                mean_offset=float(data_config.get("mean_offset", 0.0)),
            )
        else:
            raise ValueError(f"Unknown dataset_family '{dataset_family}'")

    _emit_scaling_phase(
        "dataset_ready",
        framework=framework,
        method=method,
        prediction_mode=prediction_mode,
        n_train=n_train,
        d=d,
        elapsed_s=time.perf_counter() - case_start,
    )

    true_params_for_metadata = dict(getattr(dataset, "true_params", {}) or {})
    feature_surface = str(
        data_config_for_metadata.get(
            "feature_surface",
            "single_output_ard"
            if ard
            else (
                "single_output_nonzero_mean"
                if float(true_params_for_metadata.get("mean", 0.0)) != 0.0
                else "single_output_isotropic_rbf"
            ),
        )
    )
    feature_variant = str(
        data_config_for_metadata.get(
            "feature_variant",
            "ard"
            if ard
            else (
                "nonzero_mean"
                if float(true_params_for_metadata.get("mean", 0.0)) != 0.0
                else "isotropic_rbf"
            ),
        )
    )
    data_metadata = {
        "model_family": str(data_config_for_metadata.get("model_family", "ExactGP")),
        "feature_surface": feature_surface,
        "feature_variant": feature_variant,
        "dataset_family": str(
            true_params_for_metadata.get(
                "dataset_family", data_config_for_metadata.get("dataset_family", "unknown")
            )
        ),
        "data_mean_offset": float(true_params_for_metadata.get("mean", 0.0)),
        "data_noise_level": data_config_for_metadata.get(
            "noise_level", true_params_for_metadata.get("noise_level")
        ),
    }

    learning_rate = _single_output_learning_rate()
    lr_schedule = "cosine"
    initialization_config = _single_output_initialization_config(ard=ard, d=d)
    initial_params = _single_output_initial_params(ard=ard, d=d)

    if framework == "mojogp":
        if mojogp_solver_policy not in {"strict_fair", "route_default"}:
            raise ValueError(
                "mojogp_solver_policy must be 'strict_fair' or 'route_default', "
                f"got '{mojogp_solver_policy}'"
            )
        backend_meta = _single_output_gpytorch_backend(method, "rbf")
        if mojogp_preset is None:
            solver_profile = _single_output_solver_profile(
                method,
                str(backend_meta["baseline_backend"]),
                framework="mojogp",
            )
        else:
            solver_profile = SingleOutputGP._resolve_cg_params(preset=mojogp_preset)
            solver_profile = {
                "cg_tolerance": float(solver_profile["cg_tol"]),
                "max_cg_iterations": int(solver_profile["max_cg_iter"]),
                "num_trace_samples": int(solver_profile["num_probes"]),
                "max_lanczos_quadrature_iterations": int(
                    solver_profile["max_tridiag_iter"]
                ),
                "max_root_decomposition_size": 20,
                "max_preconditioner_size": int(solver_profile["precond_rank"]),
                "min_preconditioning_size": 0,
                "precond_rank": int(solver_profile["precond_rank"]),
                "precond_method": int(solver_profile["precond_method"]),
                "precond_family": "pivoted_cholesky",
                "precond_rebuild_threshold": float(
                    solver_profile["precond_rebuild_threshold"]
                ),
                "precond": str(solver_profile["precond"]),
                "preset": mojogp_preset,
            }
        prediction_solver_profile = _prediction_solver_profile(
            solver_profile,
            prediction_mode,
        )
        prediction_kwargs: dict[str, object] = {}
        if mojogp_solver_policy == "strict_fair":
            prediction_kwargs = {
                "max_cg_iterations": int(prediction_solver_profile["max_cg_iterations"]),
                "cg_tolerance": float(prediction_solver_profile["cg_tolerance"]),
                "preconditioner_rank": int(prediction_solver_profile["precond_rank"]),
                "max_root_decomposition_size": int(
                    prediction_solver_profile["max_root_decomposition_size"]
                ),
            }
        reset_torch_memory_stats()
        gpu_monitor = GPUMemoryMonitor(interval=0.1)
        gpu_monitor.start()
        tracemalloc.start()

        gp = SingleOutputGP(RBF(ard=ard))
        if specialization is not None:
            from tests.benchmarks.specialization_adapter import apply_specialization_to_model

            apply_specialization_to_model(gp, specialization)
        result = None
        try:
            from mojogp.specialization import specialization_request_dict

            startup_start = time.perf_counter()
            _emit_scaling_phase(
                "startup_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            startup_profile = _measure_mojogp_jit_startup_profile_with_specialization(
                d,
                method,
                (
                    None
                    if specialization is None
                    else specialization_request_dict(specialization)
                ),
                ard=ard,
            )
            prepare_start = time.perf_counter()
            gp.dim = dataset.X_train.shape[1]
            # Match ExactGP.fit()'s training compile path. Setting
            # _training_method here injects materialized prediction NCOLS hints,
            # which changes ARD schedule selection and makes the benchmark train
            # a different/slower module than the public fit() route.
            gp._ensure_compiled()
            startup_profile = {
                **startup_profile,
                "startup_prepare_time_s": float(time.perf_counter() - prepare_start),
            }
            _emit_scaling_phase(
                "startup_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=time.perf_counter() - startup_start,
            )
            _emit_scaling_phase(
                "fit_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            fit_start = time.perf_counter()
            fit_kwargs = {
                "method": method,
                "max_iterations": max_iterations,
                "learning_rate": learning_rate,
                "initial_noise": float(initialization_config["initial_noise"]),
                "initial_params": initial_params,
                "lr_schedule": lr_schedule,
                "enable_early_stopping": enable_early_stopping,
                "preset": mojogp_preset,
                "max_cg_iterations": solver_profile["max_cg_iterations"],
                "cg_tolerance": solver_profile["cg_tolerance"],
                "num_probes": solver_profile["num_trace_samples"],
                "max_tridiag_iterations": solver_profile[
                    "max_lanczos_quadrature_iterations"
                ],
                "verbose": False,
            }
            if mojogp_solver_policy == "strict_fair":
                fit_kwargs.update(
                    {
                        "preconditioner_rank": solver_profile["precond_rank"],
                        "preconditioner": str(
                            solver_profile.get(
                                "precond", FAIR_SINGLE_OUTPUT_SOLVER["precond"]
                            )
                        ),
                        "use_preconditioner": bool(
                            solver_profile.get(
                                "use_preconditioner",
                                FAIR_SINGLE_OUTPUT_SOLVER["use_preconditioner"],
                            )
                        ),
                    }
                )
            _, fit_memory_stats = measure_gpu_phase(
                lambda: gp.fit(dataset.X_train, dataset.y_train, **fit_kwargs),
                interval=0.02,
            )
            training_time_s = time.perf_counter() - fit_start
            _emit_scaling_phase(
                "fit_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=training_time_s,
            )

            def _predict_once():
                return gp.predict(
                    dataset.X_test,
                    variance_method=prediction_mode,
                    **prediction_kwargs,
                )

            _clear_mojogp_prediction_caches_for_benchmark(gp)
            _emit_scaling_phase(
                "prediction_cache_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            cache_start = time.perf_counter()
            cache_info = gp.prepare_prediction_cache(
                variance_method=prediction_mode,
                **prediction_kwargs,
            )
            cache_prepare_wall_s = float(time.perf_counter() - cache_start)
            _emit_scaling_phase(
                "prediction_cache_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=cache_prepare_wall_s,
            )

            _emit_scaling_phase(
                "prediction_apply_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            pred_start = time.perf_counter()
            pred, pred_memory_stats = measure_gpu_phase(_predict_once, interval=0.02)
            prepared_apply_time_s = float(time.perf_counter() - pred_start)
            prepared_backend_predict = dict(getattr(gp, "_backend_predict_info", {}) or {})
            _emit_scaling_phase(
                "prediction_apply_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=prepared_apply_time_s,
            )

            repeated_prediction_times: list[float] = []
            repeat_count = _prediction_repeat_count_for_case(
                framework="mojogp",
                method=method,
                prediction_mode=prediction_mode,
            )
            if repeat_count > 0:
                _emit_scaling_phase(
                    "prediction_repeats_start",
                    framework=framework,
                    method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    d=d,
                )
            repeats_start = time.perf_counter()
            for _ in range(repeat_count):
                _, repeat_time_s = _time_prediction_call(_predict_once)
                repeated_prediction_times.append(repeat_time_s)
            if repeat_count > 0:
                _emit_scaling_phase(
                    "prediction_repeats_done",
                    framework=framework,
                    method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    d=d,
                    elapsed_s=time.perf_counter() - repeats_start,
                )
            prediction_phase_times = {
                "prediction_cache_prepare_time_s": cache_prepare_wall_s,
                "prediction_prepared_apply_time_s": prepared_apply_time_s,
                "prediction_alpha_time_s": float(
                    cache_info.get(
                        "prediction_cache_alpha_time_s",
                        cache_info.get("alpha_time_s", 0.0),
                    )
                ),
                "prediction_love_root_time_s": float(
                    cache_info.get(
                        "prediction_cache_love_root_time_s",
                        cache_info.get("love_root_time_s", 0.0),
                    )
                ),
                **_prediction_time_quantiles(repeated_prediction_times),
            }

            gpu_monitor.stop()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del current

            memory_stats = gpu_monitor.get_stats()
            memory_stats.update(get_torch_memory_stats())
            memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)
            memory_stats["max_mb"] = max(
                float(memory_stats.get("max_mb", 0.0)),
                float(fit_memory_stats.get("phase_peak_gpu_mb", 0.0)),
                float(pred_memory_stats.get("phase_peak_gpu_mb", 0.0)),
            )
            memory_stats["mean_mb"] = max(
                float(memory_stats.get("mean_mb", 0.0)),
                float(memory_stats["max_mb"]),
            )
            memory_stats["torch_peak_mb"] = max(
                float(memory_stats.get("torch_peak_mb", 0.0)),
                float(fit_memory_stats.get("torch_peak_mb", 0.0)),
                float(pred_memory_stats.get("torch_peak_mb", 0.0)),
            )
            memory_stats["torch_current_mb"] = float(
                pred_memory_stats.get(
                    "torch_current_mb", memory_stats.get("torch_current_mb", 0.0)
                )
            )
            memory_stats["training_peak_gpu_mb"] = float(
                fit_memory_stats.get("phase_peak_gpu_mb", 0.0)
            )
            memory_stats["training_delta_gpu_mb"] = float(
                fit_memory_stats.get("phase_delta_gpu_mb", 0.0)
            )
            memory_stats["prediction_peak_gpu_mb"] = float(
                pred_memory_stats.get("phase_peak_gpu_mb", 0.0)
            )
            memory_stats["prediction_delta_gpu_mb"] = float(
                pred_memory_stats.get("phase_delta_gpu_mb", 0.0)
            )
            if prediction_mode == "exact":
                memory_stats["exact_prediction_peak_gpu_mb"] = float(
                    pred_memory_stats.get("phase_peak_gpu_mb", 0.0)
                )
                memory_stats["exact_prediction_delta_gpu_mb"] = float(
                    pred_memory_stats.get("phase_delta_gpu_mb", 0.0)
                )
            elif prediction_mode == "love":
                memory_stats["love_prediction_peak_gpu_mb"] = float(
                    pred_memory_stats.get("phase_peak_gpu_mb", 0.0)
                )
                memory_stats["love_prediction_delta_gpu_mb"] = float(
                    pred_memory_stats.get("phase_delta_gpu_mb", 0.0)
                )

            _emit_scaling_phase(
                "prediction_x_test_scaling_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            x_test_scaling_start = time.perf_counter()
            prediction_x_test_scaling = _measure_mojogp_single_prediction_x_test_scaling(
                gp=gp,
                dataset=dataset,
                data_config=data_config_for_metadata,
                method=method,
                prediction_mode=prediction_mode,
                benchmark_variety=benchmark_variety,
                tier=tier,
                canonical_first_time_s=prepared_apply_time_s,
                canonical_repeat_times_s=repeated_prediction_times,
                canonical_memory_stats=pred_memory_stats,
                canonical_backend_predict=prepared_backend_predict,
            )
            _emit_scaling_phase(
                "prediction_x_test_scaling_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=time.perf_counter() - x_test_scaling_start,
            )

            mean = np.asarray(pred.mean, dtype=np.float32)
            variance = np.asarray(pred.variance, dtype=np.float32)
            training_result = gp.training_result
            backend_train = getattr(gp, "_backend_train_info", {}) or {}
            backend_predict = prepared_backend_predict
            prediction_time_s = prepared_apply_time_s
            actual_train_precond_rank = int(
                backend_train.get("precond_rank", solver_profile["precond_rank"])
            )
            actual_train_precond_method = int(
                backend_train.get("precond_method", solver_profile["precond_method"])
            )
            actual_train_uses_precond = bool(
                backend_train.get(
                    "use_preconditioner",
                    actual_train_precond_rank > 0,
                )
            )
            actual_train_precond_family = (
                "disabled"
                if not actual_train_uses_precond or actual_train_precond_rank <= 0
                else (
                    "route_default"
                    if mojogp_solver_policy == "route_default"
                    else solver_profile["precond_family"]
                )
            )
            metadata = _solver_policy_metadata(
                _single_output_benchmark_metadata(
                    framework="mojogp",
                    method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    tier=tier,
                    benchmark_variety=benchmark_variety,
                    kernel="rbf",
                    baseline_backend=str(backend_meta["baseline_backend"]),
                    keops_supported=bool(backend_meta["keops_supported"]),
                    keops_used=bool(backend_meta["keops_used"]),
                    requested_backend=str(backend_meta["requested_backend"]),
                    effective_training_backend=str(
                        backend_meta["effective_training_backend"]
                    ),
                    effective_prediction_backend=str(
                        backend_meta["effective_prediction_backend"]
                    ),
                    backend_fallback_used=bool(
                        backend_meta["backend_fallback_used"]
                    ),
                    backend_fallback_reason=(
                        None
                        if backend_meta["backend_fallback_reason"] is None
                        else str(backend_meta["backend_fallback_reason"])
                    ),
                ),
                framework="mojogp",
                mojogp_solver_policy=mojogp_solver_policy,
            )
            cg_params = gp._resolve_cg_params(
                preset=None,
                max_cg_iter=None,
                cg_tol=None,
                num_probes=None,
                max_tridiag_iter=None,
                precond_rank=None,
                precond_rebuild_threshold=None,
                precond=None,
            )
            learned_params = normalize_single_output_benchmark_hparams(
                gp.get_learned_params()
            )
            result = _build_result(
                dataset=dataset,
                framework="mojogp",
                model_type="SingleOutputGP",
                training_method=method,
                method=method,
                prediction_mode=prediction_mode,
                training_time_s=training_time_s,
                prediction_mean_time_s=float(
                    backend_predict.get("prediction_mean_time_s", prediction_time_s)
                ),
                prediction_variance_time_s=float(
                    backend_predict.get(
                        "prediction_variance_time_s",
                        max(
                            backend_predict.get("prediction_total_time_s", prediction_time_s)
                            - backend_predict.get("prediction_mean_time_s", 0.0)
                            - backend_predict.get("prediction_alpha_time_s", 0.0),
                            0.0,
                        ),
                    )
                ),
                training_iter_times_ms=(
                    None
                    if training_result is None or training_result.iter_times_ms is None
                    else training_result.iter_times_ms.tolist()
                ),
                startup_profile=startup_profile,
                iterations_run=int(training_result.iterations),
                max_iterations=max_iterations,
                early_stopped=bool(training_result.converged)
                or int(training_result.iterations) < max_iterations,
                memory_stats=memory_stats,
                mean=mean,
                variance=variance,
                learned_lengthscale=float(learned_params["lengthscale"]),
                learned_noise=float(learned_params.get("noise", 0.1)),
                learned_outputscale=float(learned_params["outputscale"]),
                learned_mean=float(learned_params.get("mean", training_result.mean)),
                final_nll=float(training_result.nll),
                optimizer_config={
                    "max_iterations": max_iterations,
                    "learning_rate": learning_rate,
                    "lr_schedule": lr_schedule,
                },
                training_solver_config={
                    "framework": "mojogp",
                    "model_family": "SingleOutputGP",
                    "mode": method,
                    "cg_tolerance": float(solver_profile["cg_tolerance"]),
                    "max_cg_iterations": int(solver_profile["max_cg_iterations"]),
                    "num_trace_samples": int(solver_profile["num_trace_samples"]),
                    "max_tridiag_iter": int(
                        solver_profile["max_lanczos_quadrature_iterations"]
                    ),
                    "precond_rank": actual_train_precond_rank,
                    "precond_method": actual_train_precond_method,
                    "precond_rebuild_threshold": backend_train.get(
                        "precond_rebuild_threshold"
                    ),
                    "min_preconditioning_size": int(
                        solver_profile["min_preconditioning_size"]
                    ),
                    "precond_family": actual_train_precond_family,
                    "use_preconditioner": actual_train_uses_precond,
                    "mojogp_solver_policy": mojogp_solver_policy,
                },
                prediction_solver_config={
                    "framework": "mojogp",
                    "mode": method,
                    "prediction_mode": prediction_mode,
                    "cg_tolerance": float(
                        backend_predict.get(
                            "cg_tolerance", solver_profile["cg_tolerance"]
                        )
                    ),
                    "max_cg_iterations": int(
                        backend_predict.get(
                            "max_cg_iterations", solver_profile["max_cg_iterations"]
                        )
                    ),
                    "precond_rank": int(
                        backend_predict.get("precond_rank", actual_train_precond_rank)
                    ),
                    "precond_method": actual_train_precond_method,
                    "max_root_decomposition_size": int(
                        backend_predict.get(
                            "max_root_decomposition_size",
                            solver_profile["max_root_decomposition_size"],
                        )
                    ),
                    "precond_family": actual_train_precond_family,
                },
                cg_telemetry={
                    "training": _cg_telemetry_from_history(
                        getattr(training_result, "cg_iterations_history", None)
                    ),
                    "prediction": _cg_telemetry_from_predict_info(
                        backend_predict,
                        prediction_mode=prediction_mode,
                    ),
                },
                extra_config={
                    "mojogp_preset": mojogp_preset,
                    "training_route": backend_train.get("training_route", method),
                    "materialization_mode": backend_train.get("materialization_mode"),
                    **_single_output_prediction_route_metadata(
                        method=method,
                        prediction_mode=prediction_mode,
                        n_train=n_train,
                        n_test=int(dataset.X_test.shape[0]),
                        backend_train=dict(backend_train),
                        backend_predict=dict(backend_predict),
                        cache_info=dict(cache_info),
                        pred_memory_stats=dict(pred_memory_stats),
                    ),
                    "prediction_alpha_time_s": float(
                        prediction_phase_times.get("prediction_alpha_time_s", 0.0)
                    ),
                    "prediction_love_root_time_s": float(
                        prediction_phase_times.get("prediction_love_root_time_s", 0.0)
                    ),
                    "prediction_cold_first_time_s": _optional_float(
                        prediction_phase_times.get("prediction_cold_first_time_s")
                    ),
                    "prediction_cache_prepare_time_s": float(
                        prediction_phase_times.get("prediction_cache_prepare_time_s", 0.0)
                    ),
                    "prediction_prepared_apply_time_s": float(
                        prediction_phase_times.get("prediction_prepared_apply_time_s", 0.0)
                    ),
                    "prediction_repeated_median_time_s": prediction_phase_times.get(
                        "prediction_repeated_median_time_s"
                    ),
                    "prediction_repeated_p5_time_s": prediction_phase_times.get(
                        "prediction_repeated_p5_time_s"
                    ),
                    "prediction_repeated_p95_time_s": prediction_phase_times.get(
                        "prediction_repeated_p95_time_s"
                    ),
                    "prediction_repeated_times_s": repeated_prediction_times,
                    "prediction_repeat_count": repeat_count,
                    "prediction_x_test_scaling_policy": _prediction_x_test_scaling_policy(
                        framework="mojogp",
                        method=method,
                        prediction_mode=prediction_mode,
                    ),
                    "prediction_cache_alpha_time_s": float(
                        cache_info.get("prediction_cache_alpha_time_s", cache_info.get("alpha_time_s", 0.0))
                    ),
                    "prediction_cache_love_root_time_s": float(
                        cache_info.get(
                            "prediction_cache_love_root_time_s",
                            cache_info.get("love_root_time_s", 0.0),
                        )
                    ),
                    "prediction_cache_rank": int(
                        cache_info.get("prediction_cache_rank", cache_info.get("rank", 0)) or 0
                    ),
                    "prediction_cache_has_love_root": bool(
                        cache_info.get(
                            "prediction_cache_has_love_root",
                            cache_info.get("has_love_root", False),
                        )
                    ),
                    "prediction_timing_quality": (
                        "prepared_cache_split"
                        if backend_predict.get("prediction_cache_used")
                        else "cold_first_only"
                    ),
                    "exact_block_cols": backend_predict.get("exact_block_cols"),
                    "exact_cross_mode": backend_predict.get("exact_cross_mode"),
                    "exact_cg_block_count": backend_predict.get("exact_cg_block_count"),
                    "exact_cg_total_iterations": backend_predict.get(
                        "exact_cg_total_iterations"
                    ),
                    "exact_cg_max_iterations": backend_predict.get(
                        "exact_cg_max_iterations"
                    ),
                    "exact_alloc_time_s": backend_predict.get("exact_alloc_time_s"),
                    "exact_cross_time_s": backend_predict.get("exact_cross_time_s"),
                    "exact_diag_time_s": backend_predict.get("exact_diag_time_s"),
                    "exact_solve_time_s": backend_predict.get("exact_solve_time_s"),
                    "exact_post_time_s": backend_predict.get("exact_post_time_s"),
                    "startup_profile": startup_profile,
                    **metadata,
                    "benchmark_track": benchmark_track,
                    "benchmark_variety": benchmark_variety,
                    "benchmark_route_tier": tier,
                    "n_selection_policy": n_selection_policy,
                    "size_role": size_role,
                    "case_variant": case_variant,
                    "mojogp_solver_policy": mojogp_solver_policy,
                    "enable_early_stopping": enable_early_stopping,
                    "specialization": specialization or {},
                    "initialization_config": initialization_config,
                    "training_nll_history": (
                        None
                        if training_result is None or training_result.nll_history is None
                        else training_result.nll_history.tolist()
                    ),
                    **_single_output_ard_config(
                        dataset=dataset,
                        lengthscales=learned_params.get(
                            "lengthscales", [learned_params["lengthscale"]]
                        ),
                        ard=ard,
                    ),
                    **data_metadata,
                },
                prediction_phase_times=prediction_phase_times,
                prediction_x_test_scaling=prediction_x_test_scaling,
                benchmark_name=benchmark_name,
                kernel_label="rbf_ard" if ard else "rbf",
            )
        finally:
            destroy_provider_info = getattr(gp, "_destroy_provider_info", None)
            if callable(destroy_provider_info):
                destroy_provider_info()
        return result

    if framework == "gpytorch":
        backend_meta = _single_output_gpytorch_backend(method, "rbf")
        solver_profile = _single_output_solver_profile(
            method,
            str(backend_meta["baseline_backend"]),
            framework="gpytorch",
        )
        prediction_solver_profile = _prediction_solver_profile(
            solver_profile,
            prediction_mode,
        )
        train_result = None
        pred_result = None
        try:
            _emit_scaling_phase(
                "fit_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            train_result = train_gpytorch_single_output(
                dataset.X_train,
                dataset.y_train,
                kernel_type="rbf",
                mode=str(backend_meta["mode"]),
                n_iterations=max_iterations,
                lr=learning_rate,
                init_ls=1.0,
                init_noise=float(initialization_config["initial_noise"]),
                lr_schedule=lr_schedule,
                cg_tolerance=solver_profile["cg_tolerance"],
                max_cg_iterations=solver_profile["max_cg_iterations"],
                num_trace_samples=solver_profile["num_trace_samples"],
                max_preconditioner_size=solver_profile["max_preconditioner_size"],
                max_lanczos_quadrature_iterations=solver_profile[
                    "max_lanczos_quadrature_iterations"
                ],
                min_preconditioning_size=solver_profile["min_preconditioning_size"],
                early_stop_patience=(15 if enable_early_stopping else max_iterations + 1),
                early_stop_tol=1e-4,
                monitor_memory=True,
                device="cuda",
                ard=ard,
            )
            _emit_scaling_phase(
                "fit_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=float(train_result["training_time_s"]),
            )
            _emit_scaling_phase(
                "prediction_apply_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            pred_result = predict_gpytorch_single_output(
                train_result,
                dataset.X_test,
                mode=str(backend_meta["mode"]),
                cg_tolerance=prediction_solver_profile["cg_tolerance"],
                max_cg_iterations=prediction_solver_profile["max_cg_iterations"],
                max_preconditioner_size=prediction_solver_profile[
                    "max_preconditioner_size"
                ],
                max_lanczos_quadrature_iterations=prediction_solver_profile[
                    "max_lanczos_quadrature_iterations"
                ],
                min_preconditioning_size=prediction_solver_profile[
                    "min_preconditioning_size"
                ],
                max_root_decomposition_size=prediction_solver_profile[
                    "max_root_decomposition_size"
                ],
                use_love=prediction_mode == "love",
            )
            _emit_scaling_phase(
                "prediction_apply_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=float(pred_result.get("total_time_s", 0.0)),
            )
            training_solver_config = {
                **dict(train_result.get("solver_config", {})),
                "precond_method": solver_profile["precond_method"],
                "precond_family": solver_profile["precond_family"],
                "precond_rank": solver_profile["precond_rank"],
            }
            prediction_solver_config = {
                **dict(pred_result.get("solver_config", {})),
                "precond_method": solver_profile["precond_method"],
                "precond_family": solver_profile["precond_family"],
                "precond_rank": solver_profile["precond_rank"],
            }
            merged_memory_stats = merge_gpytorch_benchmark_memory(
                dict(train_result.get("memory_stats", {})),
                dict(pred_result.get("memory_stats", {})),
            )
            _emit_scaling_phase(
                "prediction_x_test_scaling_start",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
            x_test_scaling_start = time.perf_counter()
            prediction_x_test_scaling = _measure_gpytorch_single_prediction_x_test_scaling(
                train_result=train_result,
                pred_result=pred_result,
                dataset=dataset,
                data_config=data_config_for_metadata,
                prediction_mode=prediction_mode,
                benchmark_variety=benchmark_variety,
                tier=tier,
                mode=str(backend_meta["mode"]),
                prediction_solver_profile=prediction_solver_profile,
            )
            _emit_scaling_phase(
                "prediction_x_test_scaling_done",
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                elapsed_s=time.perf_counter() - x_test_scaling_start,
            )
            return _build_result(
                dataset=dataset,
                framework="gpytorch",
                model_type="SingleOutputGP",
                training_method=method,
                method=str(backend_meta["mode"]),
                prediction_mode=prediction_mode,
                training_time_s=float(train_result["training_time_s"]),
                prediction_mean_time_s=float(pred_result["mean_time_s"]),
                prediction_variance_time_s=float(pred_result["variance_time_s"]),
                training_iter_times_ms=list(train_result.get("iter_times_ms", [])),
                startup_profile=None,
                iterations_run=int(train_result["iterations_run"]),
                max_iterations=max_iterations,
                early_stopped=bool(train_result["early_stopped"]),
                memory_stats=merged_memory_stats,
                mean=np.asarray(pred_result["mean"], dtype=np.float32),
                variance=np.asarray(pred_result["variance"], dtype=np.float32),
                learned_lengthscale=float(
                    train_result["learned_params"].get("lengthscale", 1.0)
                ),
                learned_noise=float(train_result["learned_params"].get("noise", 0.1)),
                learned_outputscale=float(
                    train_result["learned_params"].get("outputscale", 1.0)
                ),
                learned_mean=float(train_result["learned_params"].get("mean", 0.0)),
                final_nll=float(train_result["final_nll"]),
                optimizer_config=dict(train_result.get("optimizer_config", {})),
                training_solver_config=training_solver_config,
                prediction_solver_config=prediction_solver_config,
                cg_telemetry={
                    "training": train_result.get("cg_telemetry", {}),
                    "prediction": pred_result.get("cg_telemetry", {}),
                },
                extra_config=_single_output_benchmark_metadata(
                    framework="gpytorch",
                    method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    tier=tier,
                    benchmark_variety=benchmark_variety,
                    kernel="rbf",
                    baseline_backend=str(backend_meta["baseline_backend"]),
                    keops_supported=bool(backend_meta["keops_supported"]),
                    keops_used=bool(train_result.get("effective_mode") == "keops"),
                    requested_backend=str(
                        train_result.get(
                            "requested_mode",
                            backend_meta["requested_backend"],
                        )
                    ),
                    effective_training_backend=str(
                        train_result.get(
                            "effective_mode",
                            backend_meta["effective_training_backend"],
                        )
                    ),
                    effective_prediction_backend=str(
                        pred_result.get(
                            "effective_prediction_mode",
                            backend_meta["effective_prediction_backend"],
                        )
                    ),
                    backend_fallback_used=bool(
                        train_result.get("backend_fallback_used")
                        or pred_result.get("backend_fallback_used")
                    ),
                    backend_fallback_reason=(
                        pred_result.get("backend_fallback_reason")
                        or train_result.get("backend_fallback_reason")
                    ),
                )
                | {
                    "prediction_alpha_time_s": float(pred_result.get("alpha_time_s", 0.0)),
                    "prediction_repeat_count": _prediction_repeat_count_for_case(
                        framework="gpytorch",
                        method=method,
                        prediction_mode=prediction_mode,
                    ),
                    "prediction_x_test_scaling_policy": _prediction_x_test_scaling_policy(
                        framework="gpytorch",
                        method=method,
                        prediction_mode=prediction_mode,
                    ),
                    "exact_block_cols": pred_result.get("exact_block_cols"),
                    "exact_cross_mode": pred_result.get("exact_cross_mode"),
                    "exact_cg_block_count": pred_result.get("exact_cg_block_count"),
                    "exact_cg_total_iterations": pred_result.get(
                        "exact_cg_total_iterations"
                    ),
                    "exact_cg_max_iterations": pred_result.get(
                        "exact_cg_max_iterations"
                    ),
                    "exact_alloc_time_s": pred_result.get("exact_alloc_time_s"),
                    "exact_cross_time_s": pred_result.get("exact_cross_time_s"),
                    "exact_diag_time_s": pred_result.get("exact_diag_time_s"),
                    "exact_solve_time_s": pred_result.get("exact_solve_time_s"),
                    "exact_post_time_s": pred_result.get("exact_post_time_s"),
                    "prediction_supported": bool(
                        pred_result.get("prediction_supported", True)
                    ),
                    "prediction_error": pred_result.get("prediction_error"),
                    "benchmark_track": benchmark_track,
                    "benchmark_variety": benchmark_variety,
                    "benchmark_route_tier": tier,
                    "n_selection_policy": n_selection_policy,
                    "size_role": size_role,
                    "case_variant": case_variant,
                    "enable_early_stopping": enable_early_stopping,
                    "initialization_config": initialization_config,
                    "training_nll_history": list(train_result.get("nll_history", [])),
                    **_single_output_ard_config(
                        dataset=dataset,
                        lengthscales=train_result["learned_params"].get(
                            "lengthscales",
                            [train_result["learned_params"].get("lengthscale", 1.0)],
                        ),
                        ard=ard,
                    ),
                    **data_metadata,
                },
                benchmark_name=benchmark_name,
                kernel_label="rbf_ard" if ard else "rbf",
                prediction_x_test_scaling=prediction_x_test_scaling,
                prediction_phase_times={
                    "prediction_alpha_time_s": float(pred_result.get("alpha_time_s", 0.0)),
                    "prediction_cache_prepare_time_s": 0.0,
                    "prediction_prepared_apply_time_s": float(
                        pred_result.get("total_time_s", 0.0)
                    ),
                },
            )
        finally:
            if pred_result is not None:
                del pred_result
            if train_result is not None:
                del train_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise ValueError(f"Unknown framework '{framework}'")


def _run_scaling_case_capture(
    method: str,
    n_train: int,
    d: int,
    *,
    framework: str,
    prediction_mode: str,
    tier: str,
    benchmark_variety: str,
    benchmark_track: str,
    n_selection_policy: str,
    size_role: str,
    allow_recorded_failure: bool,
    max_iterations: int = 100,
    enable_early_stopping: bool = False,
    benchmark_name: str = "scaling_certification",
    results_dir: Path | None = None,
    mojogp_solver_policy: str = "strict_fair",
    case_variant: str | None = None,
    comparison_mojogp_case_variant: str | None = None,
    timeout_s: int | None = None,
):
    try:
        result = _run_scaling_case(
            method,
            n_train,
            d,
            framework=framework,
            prediction_mode=prediction_mode,
            tier=tier,
            benchmark_variety=benchmark_variety,
            benchmark_track=benchmark_track,
            n_selection_policy=n_selection_policy,
            size_role=size_role,
            max_iterations=max_iterations,
            enable_early_stopping=enable_early_stopping,
            benchmark_name=benchmark_name,
            results_dir=results_dir,
            mojogp_solver_policy=mojogp_solver_policy,
            case_variant=case_variant,
            comparison_mojogp_case_variant=comparison_mojogp_case_variant,
            timeout_s=timeout_s,
        )
        return result, None
    except Exception as exc:
        failure = {
            "framework": framework,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "benchmark_track": benchmark_track,
            "benchmark_variety": benchmark_variety,
            "benchmark_route_tier": tier,
            "n_selection_policy": n_selection_policy,
            "size_role": size_role,
            "mojogp_solver_policy": mojogp_solver_policy,
            "case_variant": case_variant,
            "n": n_train,
            "d": d,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if allow_recorded_failure:
            return None, failure
        raise


def _matrix_free_prediction_memory_metric(
    result: BenchmarkResult, *, prediction_mode: str
) -> float | None:
    if prediction_mode == "exact":
        delta = result.memory.exact_prediction_delta_gpu_mb
        peak = result.memory.exact_prediction_peak_gpu_mb
    elif prediction_mode == "love":
        delta = result.memory.love_prediction_delta_gpu_mb
        peak = result.memory.love_prediction_peak_gpu_mb
    else:
        delta = result.memory.prediction_delta_gpu_mb
        peak = result.memory.prediction_peak_gpu_mb

    # Measurement deltas are allocator-reuse sensitive: a smaller case can reuse buffers
    # allocated during cache prep or earlier predictions and report a tiny delta.
    # The peak footprint is stable for detecting whether matrix-free prediction
    # scales closer to O(n) or O(n^2) while keeping the active route-specific peak.
    if peak is not None and float(peak) > 0.0:
        return float(peak)
    if result.memory.prediction_peak_gpu_mb is not None and float(
        result.memory.prediction_peak_gpu_mb
    ) > 0.0:
        return float(result.memory.prediction_peak_gpu_mb)
    if result.memory.gpu_max_mb > 0.0:
        return float(result.memory.gpu_max_mb)
    if delta is not None and float(delta) > 1.0:
        return float(delta)
    if result.memory.prediction_delta_gpu_mb is not None and float(
        result.memory.prediction_delta_gpu_mb
    ) > 1.0:
        return float(result.memory.prediction_delta_gpu_mb)
    return None


def _assert_matrix_free_prediction_memory_scaling(
    results: list[BenchmarkResult],
) -> None:
    grouped: dict[tuple[int, str, str | None], list[BenchmarkResult]] = {}
    for result in results:
        if result.config.get("framework") != "mojogp":
            continue
        if result.config.get("training_method") != "matrix_free":
            continue
        key = (
            int(result.config["d"]),
            str(result.config["prediction_mode"]),
            result.config.get("case_variant"),
        )
        grouped.setdefault(key, []).append(result)

    for (d, prediction_mode, case_variant), group in grouped.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: int(item.config["n"]))
        small = ordered[0]
        large = ordered[-1]
        small_metric = _matrix_free_prediction_memory_metric(
            small, prediction_mode=prediction_mode
        )
        large_metric = _matrix_free_prediction_memory_metric(
            large, prediction_mode=prediction_mode
        )
        if small_metric is None or large_metric is None or small_metric <= 0.0:
            continue

        n_ratio = float(large.config["n"]) / float(small.config["n"])
        observed_ratio = large_metric / small_metric
        linear_ratio = n_ratio
        quadratic_ratio = n_ratio * n_ratio
        verdict = (
            "linear_like"
            if abs(observed_ratio - linear_ratio)
            < abs(observed_ratio - quadratic_ratio)
            else "quadratic_like"
        )
        for result in ordered:
            result.config.setdefault("matrix_free_memory_verdicts", {})[
                prediction_mode
            ] = verdict

        assert abs(observed_ratio - linear_ratio) < abs(
            observed_ratio - quadratic_ratio
        ), (
            "Matrix-free prediction memory scaled too much like O(n^2): "
            f"prediction={prediction_mode}, variant={case_variant}, d={d}, n_small={small.config['n']}, "
            f"n_large={large.config['n']}, observed_ratio={observed_ratio:.2f}, "
            f"linear_ratio={linear_ratio:.2f}, quadratic_ratio={quadratic_ratio:.2f}, "
            f"small_metric_mb={small_metric:.2f}, large_metric_mb={large_metric:.2f}"
        )


def _assert_scaling_result_quality(results: list[BenchmarkResult], method: str) -> None:
    assert results, f"No successful results collected for {method} scaling benchmark"
    for result in results:
        if result.config.get("comparison_class") == "unsupported_comparator":
            continue
        context = (
            f"{method} framework={result.config.get('framework')} "
            f"prediction_mode={result.config.get('prediction_mode')} "
            f"n={result.config.get('n')} d={result.config.get('d')}"
        )
        for field_name in (
            "r_squared",
            "rmse",
            "mae",
            "crps",
            "msll",
            "calibration_error",
            "sharpness",
            "interval_width_95",
        ):
            value = getattr(result.accuracy, field_name)
            assert np.isfinite(value), (
                f"Scaling benchmark produced non-finite {field_name} for {context}"
            )
        for level, coverage in result.accuracy.calibration_coverage.items():
            assert np.isfinite(coverage), (
                f"Scaling benchmark produced non-finite calibration coverage at {level} for {context}"
            )
            assert 0.0 <= coverage <= 1.0, (
                f"Scaling benchmark produced invalid calibration coverage at {level} for {context}: {coverage}"
            )
        assert result.accuracy.sharpness > 0.0, (
            f"Scaling benchmark produced non-positive predictive sharpness for {context}"
        )
        assert np.isfinite(result.hyperparameters.final_nll), (
            f"Final NLL became non-finite for {context}"
        )


def _mojogp_exact_love_pair_key(result: BenchmarkResult) -> tuple[object, ...]:
    return (
        result.config.get("method"),
        result.config.get("n"),
        result.config.get("d"),
        result.config.get("kernel"),
        result.config.get("case_variant"),
        result.config.get("mojogp_solver_policy"),
    )


def _assert_mojogp_exact_love_prediction_consistency(
    results: list[BenchmarkResult],
) -> None:
    paired: dict[tuple[object, ...], dict[str, BenchmarkResult]] = {}
    for result in results:
        if result.config.get("framework") != "mojogp":
            continue
        prediction_mode = result.config.get("prediction_mode")
        if prediction_mode not in {"exact", "love"}:
            continue
        paired.setdefault(_mojogp_exact_love_pair_key(result), {})[
            str(prediction_mode)
        ] = result

    for key, by_mode in paired.items():
        if "exact" not in by_mode or "love" not in by_mode:
            continue
        exact = by_mode["exact"]
        love = by_mode["love"]
        exact_solver = dict(exact.config.get("prediction_solver_config", {}) or {})
        love_solver = dict(love.config.get("prediction_solver_config", {}) or {})
        context = f"MojoGP exact/LOVE pair {key}"
        assert int(exact_solver.get("max_cg_iterations", -1)) == int(
            love_solver.get("max_cg_iterations", -2)
        ), f"{context} used different alpha CG iteration budgets"
        assert float(exact_solver.get("cg_tolerance", float("nan"))) == pytest.approx(
            float(love_solver.get("cg_tolerance", float("nan")))
        ), f"{context} used different alpha CG tolerances"
        assert int(exact_solver.get("precond_rank", -1)) == int(
            love_solver.get("precond_rank", -2)
        ), f"{context} used different prediction preconditioner ranks"
        assert int(exact_solver.get("max_root_decomposition_size", -1)) == int(
            love_solver.get("max_root_decomposition_size", -2)
        ), f"{context} used different root decomposition ranks"

        rmse_delta = abs(float(love.accuracy.rmse) - float(exact.accuracy.rmse))
        rmse_limit = max(1e-3, 0.05 * max(float(exact.accuracy.rmse), 1e-8))
        assert rmse_delta <= rmse_limit, (
            f"{context} changed mean-driven RMSE too much: exact={exact.accuracy.rmse}, "
            f"LOVE={love.accuracy.rmse}, allowed_delta={rmse_limit}"
        )

        calibration_delta = abs(
            float(love.accuracy.calibration_error)
            - float(exact.accuracy.calibration_error)
        )
        assert calibration_delta <= 0.25, (
            f"{context} changed calibration error too much: exact={exact.accuracy.calibration_error}, "
            f"LOVE={love.accuracy.calibration_error}"
        )


def _assert_no_unexpected_prediction_ooms(
    results: list[BenchmarkResult], benchmark_variety: str
) -> None:
    if benchmark_variety not in {"standard", "extensive"}:
        return
    for result in results:
        rows = result.speed.prediction_x_test_scaling or []
        for row in rows:
            assert row.get("status") != "failed_oom", (
                "Standard scaling prediction X_test profile recorded an OOM envelope row: "
                f"framework={result.config.get('framework')} method={result.config.get('method')} "
                f"prediction_mode={result.config.get('prediction_mode')} n={result.config.get('n')} "
                f"d={result.config.get('d')} n_test={row.get('n_test')}"
            )


def _warm_up_scaling_route(
    method: str,
    d: int,
    *,
    prediction_modes: tuple[str, ...] | None = None,
) -> None:
    seed = (900 if method == "materialized" else 1200) + d * 100
    dataset = generate_structured_function_data(
        n_train=512 if method == "materialized" else 1024,
        n_test=32,
        d=d,
        function_type="smooth",
        noise_level="medium",
        seed=seed,
    )
    gp = SingleOutputGP(RBF())
    modes = prediction_modes
    if modes is None:
        modes = ("exact", "love") if method == "materialized" else ("love",)
    gp.fit(
        dataset.X_train,
        dataset.y_train,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        lr_schedule="cosine",
        enable_early_stopping=False,
        verbose=False,
    )
    for prediction_mode in modes:
        gp.predict(dataset.X_test, variance_method=prediction_mode, return_std=True)
    destroy_provider_info = getattr(gp, "_destroy_provider_info", None)
    if callable(destroy_provider_info):
        destroy_provider_info()


@pytest.mark.minimal
@pytest.mark.single_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_route_scaling_certification(results_dir, method: str):
    assert_gpu_available()
    if not _scaling_config_enables_method(method):
        pytest.skip(
            f"Scaling config {_active_scaling_config_name()} does not include method={method}"
        )
    benchmark_variety = _benchmark_variety()
    _, dims, tier, n_selection_policy = _benchmark_targets(
        method,
        benchmark_variety=benchmark_variety,
    )
    max_iterations = _configured_scaling_max_iterations()
    enable_early_stopping = _configured_scaling_enable_early_stopping()
    benchmark_track = _configured_scaling_track()
    n_selection_policy = _configured_n_selection_policy(n_selection_policy)

    results = []
    failures = []
    case_specs = _scaling_case_specs(method)
    for d in dims:
        _warm_up_scaling_route(method, d)
        for case_spec in case_specs:
            framework_sizes = _framework_sizes(
                framework=case_spec.framework,
                method=method,
                prediction_mode=case_spec.prediction_mode,
                tier=tier,
                benchmark_variety=benchmark_variety,
            )
            for n_train in framework_sizes:
                size_role = _size_role(n_train, framework_sizes)
                allow_recorded_failure = _allow_recorded_failure(
                    framework=case_spec.framework,
                    method=method,
                    prediction_mode=case_spec.prediction_mode,
                    benchmark_variety=benchmark_variety,
                    size_role=size_role,
                )
                result, failure = _run_scaling_case_capture(
                    method,
                    n_train,
                    d,
                    framework=case_spec.framework,
                    prediction_mode=case_spec.prediction_mode,
                    tier=tier,
                    benchmark_variety=benchmark_variety,
                    benchmark_track=benchmark_track,
                    n_selection_policy=n_selection_policy,
                    size_role=size_role,
                    allow_recorded_failure=allow_recorded_failure,
                    max_iterations=max_iterations,
                    enable_early_stopping=enable_early_stopping,
                    results_dir=results_dir,
                    mojogp_solver_policy=case_spec.mojogp_solver_policy,
                    case_variant=case_spec.case_variant,
                    comparison_mojogp_case_variant=case_spec.comparison_mojogp_case_variant,
                )
                if failure is not None:
                    failures.append(failure)
                    continue
                assert result is not None
                result.config.update({"benchmark_tier": tier})
                print_result(
                    result,
                    title=(
                        f"Scaling certification: {case_spec.framework} {method} "
                        f"pred={case_spec.prediction_mode} n={n_train} d={d} "
                        f"variety={benchmark_variety} role={size_role}"
                    ),
                )
                assert_gpu_was_used(result)
                results.append(result)

    if method == "matrix_free":
        _assert_matrix_free_prediction_memory_scaling(results)

    save_summary_report(
        results,
        results_dir,
        f"scaling_certification_{benchmark_variety}_{method}",
        failures=failures,
    )

    _assert_scaling_result_quality(results, method)
    _assert_mojogp_exact_love_prediction_consistency(results)
    _assert_no_unexpected_prediction_ooms(results, benchmark_variety)

    for d in dims:
        for case_spec in case_specs:
            framework = case_spec.framework
            prediction_mode = case_spec.prediction_mode
            case_variant = case_spec.case_variant
            per_dim = [
                result
                for result in results
                if result.config["d"] == d
                and result.config["framework"] == framework
                and result.config["prediction_mode"] == prediction_mode
                and result.config.get("case_variant") == case_variant
            ]
            if len(per_dim) >= 2:
                small, large = per_dim[0], per_dim[-1]
                assert large.config["n"] > small.config["n"]

    if method == "matrix_free" and benchmark_variety in {"standard", "extensive"}:
        for d in dims:
            gpytorch_envelope_results = [
                result
                for result in results
                if result.config["framework"] == "gpytorch"
                and result.config["prediction_mode"] == "love"
                and result.config["d"] == d
                and result.config.get("size_role") == "envelope"
            ]
            gpytorch_envelope_failures = [
                failure
                for failure in failures
                if failure["framework"] == "gpytorch"
                and failure["prediction_mode"] == "love"
                and failure["d"] == d
                and failure.get("size_role") == "envelope"
            ]
            assert gpytorch_envelope_results or gpytorch_envelope_failures, (
                "Expected a GPyTorch/KeOps matrix-free envelope attempt to be recorded "
                f"for d={d} in the {benchmark_variety} scaling benchmark"
            )


def test_single_output_benchmark_metadata_marks_keops_prediction_failure_unsupported():
    metadata = _single_output_benchmark_metadata(
        framework="gpytorch",
        method="matrix_free",
        prediction_mode="love",
        n_train=5000,
        tier="xsmall",
        benchmark_variety="minimal",
        kernel="rbf",
        baseline_backend="keops",
        keops_supported=True,
        keops_used=True,
        requested_backend="keops",
        effective_training_backend="keops",
        effective_prediction_backend="unsupported",
        backend_fallback_used=False,
        backend_fallback_reason="keops_prediction_failed:RuntimeError",
    )

    assert metadata["comparison_class"] == "unsupported_comparator"
    assert metadata["effective_training_backend"] == "keops"
    assert metadata["effective_prediction_backend"] == "unsupported"
    assert metadata["backend_fallback_reason"] == "keops_prediction_failed:RuntimeError"
    assert "strict matrix-free publication requires requested KeOps" in metadata[
        "fairness_note"
    ]


def test_allow_recorded_failure_keeps_mojogp_failures_hard():
    assert not _allow_recorded_failure(
        framework="mojogp",
        method="materialized",
        prediction_mode="exact",
        benchmark_variety="extensive",
        size_role="envelope",
    )
    assert not _allow_recorded_failure(
        framework="mojogp",
        method="matrix_free",
        prediction_mode="love",
        benchmark_variety="extensive",
        size_role="envelope",
    )


def test_allow_recorded_failure_for_gpytorch_matrix_free_comparator_lanes_only():
    assert not _allow_recorded_failure(
        framework="gpytorch",
        method="materialized",
        prediction_mode="exact",
        benchmark_variety="extensive",
        size_role="anchor_1",
    )
    assert not _allow_recorded_failure(
        framework="gpytorch",
        method="materialized",
        prediction_mode="exact",
        benchmark_variety="extensive",
        size_role="anchor_2",
    )
    assert not _allow_recorded_failure(
        framework="gpytorch",
        method="materialized",
        prediction_mode="love",
        benchmark_variety="standard",
        size_role="envelope",
    )
    assert _allow_recorded_failure(
        framework="gpytorch",
        method="matrix_free",
        prediction_mode="love",
        benchmark_variety="extensive",
        size_role="envelope",
    )
    assert _allow_recorded_failure(
        framework="gpytorch",
        method="matrix_free",
        prediction_mode="exact",
        benchmark_variety="extensive",
        size_role="envelope",
    )
    assert not _allow_recorded_failure(
        framework="gpytorch",
        method="materialized",
        prediction_mode="exact",
        benchmark_variety="minimal",
        size_role="envelope",
    )


def test_matrix_free_case_order_places_love_before_exact():
    assert _framework_prediction_mode_pairs("matrix_free") == [
        ("mojogp", "love"),
        ("gpytorch", "love"),
        ("gpytorch", "exact"),
        ("mojogp", "exact"),
    ]


def test_matrix_free_exact_prediction_uses_canonical_only_policy():
    assert (
        _prediction_x_test_scaling_policy(
            framework="mojogp",
            method="matrix_free",
            prediction_mode="exact",
        )
        == "canonical_only_matrix_free_exact"
    )
    assert (
        _prediction_repeat_count_for_case(
            framework="mojogp",
            method="matrix_free",
            prediction_mode="exact",
        )
        == 0
    )
    assert _prediction_x_test_scaling_policy(
        framework="mojogp",
        method="matrix_free",
        prediction_mode="love",
    ) == "full_core_plus_allowed_envelope"
    assert (
        _prediction_x_test_scaling_policy(
            framework="gpytorch",
            method="matrix_free",
            prediction_mode="exact",
        )
        == "canonical_only_matrix_free_exact"
    )


def test_matrix_free_exact_x_test_scaling_filters_to_canonical_only():
    dataset = SyntheticDataset(
        X_train=np.zeros((8, 2), dtype=np.float32),
        y_train=np.zeros(8, dtype=np.float32),
        X_test=np.zeros((BENCHMARK_PREDICTION_N_TEST, 2), dtype=np.float32),
        y_test=np.zeros(BENCHMARK_PREDICTION_N_TEST, dtype=np.float32),
        f_test=np.zeros(BENCHMARK_PREDICTION_N_TEST, dtype=np.float32),
        true_params={},
        name="policy_test",
        description="policy_test",
    )

    rows = _measure_mojogp_single_prediction_x_test_scaling(
        gp=object(),
        dataset=dataset,
        data_config={"seed": 1},
        method="matrix_free",
        prediction_mode="exact",
        benchmark_variety="extensive",
        tier="large",
        canonical_first_time_s=1.0,
        canonical_repeat_times_s=[],
        canonical_memory_stats={"phase_peak_gpu_mb": 10.0, "phase_delta_gpu_mb": 2.0},
        canonical_backend_predict={
            "prediction_mean_time_s": 0.1,
            "prediction_variance_time_s": 0.9,
        },
    )

    assert len(rows) == 1
    assert rows[0]["n_test"] == BENCHMARK_PREDICTION_N_TEST
    assert rows[0]["repeat_count"] == 0
    assert rows[0]["status"] == "ok"


def test_predict_gpytorch_single_output_marks_keops_prediction_unsupported(monkeypatch):
    def fail_predict(*args, **kwargs):
        raise RuntimeError("LazyTensor cannot be made into a Tensor")

    monkeypatch.setattr(
        "tests.shared.benchmarking.gpytorch_models.predict_gpytorch_model",
        fail_predict,
    )

    result = predict_gpytorch_single_output(
        {
            "model": object(),
            "likelihood": object(),
            "device": "cpu",
            "effective_mode": "keops",
            "requested_mode": "keops",
        },
        np.zeros((4, 2), dtype=np.float32),
        mode="keops",
        use_love=True,
    )

    assert result["prediction_supported"] is False
    assert result["effective_prediction_mode"] == "unsupported"
    assert result["backend_fallback_used"] is True
    assert result["backend_fallback_reason"] == "keops_prediction_failed:RuntimeError"
    assert result["prediction_error"] == "LazyTensor cannot be made into a Tensor"
    assert np.isnan(result["mean"]).all()
    assert np.isnan(result["variance"]).all()


@pytest.mark.minimal
@requires_cuda
def test_gpytorch_keops_exact_prediction_uses_chunked_matrix_free_path():
    assert_gpu_available()
    if not is_keops_available():
        pytest.skip("KeOps is not available in this environment")

    dataset = generate_structured_function_data(
        n_train=2000,
        n_test=192,
        d=5,
        function_type="smooth",
        noise_level="medium",
        seed=8105,
    )
    train_result = train_gpytorch_single_output(
        dataset.X_train,
        dataset.y_train,
        kernel_type="rbf",
        mode="keops",
        n_iterations=1,
        lr=0.03,
        init_ls=1.0,
        init_noise=0.1,
        lr_schedule="cosine",
        cg_tolerance=1e-2,
        max_cg_iterations=80,
        num_trace_samples=2,
        max_preconditioner_size=0,
        max_lanczos_quadrature_iterations=10,
        min_preconditioning_size=0,
        monitor_memory=False,
        device="cuda",
    )
    pred_result = predict_gpytorch_single_output(
        train_result,
        dataset.X_test,
        mode="keops",
        cg_tolerance=1e-3,
        max_cg_iterations=80,
        max_preconditioner_size=0,
        max_lanczos_quadrature_iterations=10,
        min_preconditioning_size=0,
        max_root_decomposition_size=20,
        use_love=False,
        exact_prediction_block_size=128,
    )

    assert pred_result["prediction_supported"] is True
    assert pred_result["effective_prediction_mode"] == "keops"
    assert pred_result["solver_config"]["prediction_mode"] == "exact"
    assert pred_result["exact_cross_mode"] == "chunked_keops_cross_covariance"
    assert pred_result["exact_block_cols"] == 128
    assert pred_result["exact_cg_block_count"] == 2
    assert pred_result["cg_telemetry"]["observed_cg_calls"] is True
    assert np.isfinite(pred_result["mean"]).all()
    assert np.isfinite(pred_result["variance"]).all()


@pytest.mark.minimal
@requires_cuda
def test_single_output_hparam_reporting_matches_exactgp_params():
    assert_gpu_available()
    np.random.seed(7105)
    dataset = generate_structured_function_data(
        n_train=2000,
        n_test=120,
        d=5,
        function_type="smooth",
        noise_level="medium",
    )

    gp = SingleOutputGP(RBF(lengthscale=0.2, outputscale=0.5))
    gp.fit(
        dataset.X_train,
        dataset.y_train,
        max_iterations=25,
        learning_rate=0.03,
        initial_noise=0.05,
        method="materialized",
        lr_schedule="cosine",
        max_cg_iterations=100,
        cg_tolerance=1e-2,
        num_probes=10,
        max_tridiag_iterations=20,
        preconditioner_rank=0,
        preconditioner="greedy",
        use_preconditioner=False,
        verbose=False,
    )

    learned = gp.get_learned_params()
    normalized = normalize_single_output_benchmark_hparams(learned)

    assert normalized["lengthscale"] == pytest.approx(learned["rbf_lengthscale"])
    assert normalized["outputscale"] == pytest.approx(learned["rbf_outputscale"])
    assert normalized["noise"] == pytest.approx(learned["noise"])
