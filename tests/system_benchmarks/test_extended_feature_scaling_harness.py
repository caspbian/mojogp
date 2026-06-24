"""Extended MojoGP feature scaling benchmark surface.

This suite complements the main ExactGP/MultiOutputGP/ARD scaling lanes with
feature-specific MojoGP rows that are not strict cross-framework claims: non-zero
mean, per-task noise, mixed continuous+categorical kernels, and LMC. Each row is
run through the shared subprocess benchmark runner so host and container runs are
persisted to SQLite and exported with the same session metadata as the core
scaling suites.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.benchmarks.workflow_runner import run_result_benchmark_subprocess
from tests.shared.benchmarking.report import print_result, save_summary_report
from tests.shared.benchmarking.result_types import BenchmarkResult

from .conftest import (
    assert_gpu_available,
    assert_gpu_was_used,
    get_matrix_free_capability_tier,
    get_vram_tier,
    requires_cuda,
)
from .test_scaling_certification_harness import _run_scaling_case, _size_role


EXTENDED_FEATURE_TARGETS = {
    "single_output_nonzero_mean": {
        "materialized": {
            "minimal": {
                "xsmall": [3000],
                "small": [4500],
                "medium": [8000],
                "large": [12000],
                "xlarge": [18000],
            },
            "standard": {
                "xsmall": [3000, 4500],
                "small": [4500, 7000],
                "medium": [8000, 13000],
                "large": [12000, 18000],
                "xlarge": [18000, 26000],
            },
            "extensive": {
                "xsmall": [2500, 3500, 4500],
                "small": [4500, 7000, 9000],
                "medium": [8000, 13000, 17000],
                "large": [12000, 18000, 24000],
                "xlarge": [18000, 26000, 34000],
            },
        },
        "matrix_free": {
            "minimal": {
                "xsmall": [7000],
                "small": [12000],
                "medium": [27000],
                "large": [37000],
                "xlarge": [55000],
            },
            "standard": {
                "xsmall": [7000, 12000],
                "small": [12000, 24000],
                "medium": [27000, 43000],
                "large": [35000, 60000],
                "xlarge": [55000, 85000],
            },
            "extensive": {
                "xsmall": [7000, 12000],
                "small": [12000, 24000, 36000],
                "medium": [27000, 43000, 60000],
                "large": [35000, 60000, 75000],
                "xlarge": [55000, 85000, 100000],
            },
        },
    },
    "multi_output_per_task_noise": {
        "materialized": {
            "minimal": {
                "xsmall": [2000],
                "small": [3000],
                "medium": [5000],
                "large": [7000],
                "xlarge": [9000],
            },
            "standard": {
                "xsmall": [2000, 3500],
                "small": [3000, 5000],
                "medium": [5000, 7000],
                "large": [7000, 10000],
                "xlarge": [9000, 12000],
            },
            "extensive": {
                "xsmall": [2000, 3500],
                "small": [3000, 5000, 7000],
                "medium": [5000, 7000, 9000],
                "large": [7000, 10000, 14000],
                "xlarge": [9000, 12000, 16000],
            },
        },
        "matrix_free": {
            "minimal": {
                "xsmall": [6000],
                "small": [12000],
                "medium": [25000],
                "large": [35000],
                "xlarge": [50000],
            },
            "standard": {
                "xsmall": [6000, 10000],
                "small": [12000, 24000],
                "medium": [25000, 40000],
                "large": [35000, 50000],
                "xlarge": [50000, 75000],
            },
            "extensive": {
                "xsmall": [6000, 10000],
                "small": [12000, 24000, 36000],
                "medium": [25000, 40000, 60000],
                "large": [35000, 50000, 70000],
                "xlarge": [50000, 75000, 100000],
            },
        },
    },
    "mixed_continuous_categorical": {
        "materialized": {
            "minimal": {
                "xsmall": [2000],
                "small": [3000],
                "medium": [5000],
                "large": [7000],
                "xlarge": [9000],
            },
            "standard": {
                "xsmall": [2000, 3500],
                "small": [3000, 5000],
                "medium": [5000, 7000],
                "large": [7000, 10000],
                "xlarge": [9000, 12000],
            },
            "extensive": {
                "xsmall": [2000, 3500],
                "small": [3000, 5000, 7000],
                "medium": [5000, 7000, 9000],
                "large": [7000, 10000, 14000],
                "xlarge": [9000, 12000, 16000],
            },
        },
        "matrix_free": {
            "minimal": {
                "xsmall": [5000],
                "small": [10000],
                "medium": [20000],
                "large": [30000],
                "xlarge": [40000],
            },
            "standard": {
                "xsmall": [5000, 9000],
                "small": [10000, 18000],
                "medium": [20000, 35000],
                "large": [30000, 50000],
                "xlarge": [40000, 70000],
            },
            "extensive": {
                "xsmall": [5000, 9000],
                "small": [10000, 18000, 30000],
                "medium": [20000, 35000, 50000],
                "large": [30000, 50000, 70000],
                "xlarge": [40000, 70000, 100000],
            },
        },
    },
    "multi_output_lmc": {
        "materialized": {
            "minimal": {
                "xsmall": [1500],
                "small": [2000],
                "medium": [3000],
                "large": [5000],
                "xlarge": [7000],
            },
            "standard": {
                "xsmall": [1500, 2500],
                "small": [2000, 3500],
                "medium": [3000, 5000],
                "large": [5000, 8000],
                "xlarge": [7000, 10000],
            },
            "extensive": {
                "xsmall": [1500, 2500],
                "small": [2000, 3500, 5000],
                "medium": [3000, 5000, 7000],
                "large": [5000, 8000, 12000],
                "xlarge": [7000, 10000, 14000],
            },
        },
        "matrix_free": {
            "minimal": {
                "xsmall": [3000],
                "small": [6000],
                "medium": [10000],
                "large": [15000],
                "xlarge": [20000],
            },
            "standard": {
                "xsmall": [3000, 6000],
                "small": [6000, 10000],
                "medium": [10000, 16000],
                "large": [15000, 22000],
                "xlarge": [20000, 30000],
            },
            "extensive": {
                "xsmall": [3000, 6000],
                "small": [6000, 10000, 14000],
                "medium": [10000, 16000, 22000],
                "large": [15000, 22000, 30000],
                "xlarge": [20000, 30000, 40000],
            },
        },
    },
}


def _benchmark_variety() -> str:
    variety = os.environ.get("MOJOGP_BENCHMARK_VARIETY")
    if variety is not None:
        normalized = variety.strip().lower()
        if normalized not in {"minimal", "standard", "extensive"}:
            raise ValueError(
                "MOJOGP_BENCHMARK_VARIETY must be minimal, standard, or extensive; "
                f"got '{variety}'"
            )
        return normalized
    if os.environ.get("MOJOGP_BENCHMARK_QUICK", "0") == "1":
        return "minimal"
    return "standard"


def _tier_for_method(method: str) -> tuple[str, str]:
    if method == "materialized":
        return get_vram_tier(), "vram"
    return get_matrix_free_capability_tier(), "bandwidth"


def _target_sizes(feature_surface: str, method: str, variety: str, tier: str) -> list[int]:
    return list(EXTENDED_FEATURE_TARGETS[feature_surface][method][variety][tier])


def _feature_config(
    *, feature_surface: str, method: str, n_train: int, tier: str, variety: str
) -> dict[str, object]:
    return {
        "suite_name": "extended_feature_scaling",
        "benchmark": f"{feature_surface}_scaling",
        "benchmark_track": "scaling",
        "benchmark_variety": variety,
        "benchmark_route_tier": tier,
        "feature_surface": feature_surface,
        "training_method": method,
        "n": int(n_train),
        "comparison_class": "mojogp_only_scale",
        "baseline_backend": "none",
        "fairness_note": (
            "N.B. MojoGP-only extended-feature scaling row: this suite measures "
            "feature-route speed, memory, and accuracy scaling and is not a "
            "cross-framework speedup claim."
        ),
        "fairness_axes": {
            "comparator_scope": {
                "status": "mojogp_only",
                "note": "No cross-framework comparator is published for this extended-feature row.",
            },
            "sample_count_n": {
                "status": "mojogp_only",
                "note": "The n ladder is feature-specific and selected for supported MojoGP routes.",
            },
            "optimizer": {
                "status": "aligned",
                "note": "Rows within a feature surface use a fixed optimizer policy per route.",
            },
            "solver_budget": {
                "status": "aligned",
                "note": "Rows within a feature surface use a fixed CG/Lanczos budget per route.",
            },
            "preconditioner": {
                "status": "aligned",
                "note": "Rows use the active feature harness preconditioner policy.",
            },
            "prediction_mode": {
                "status": "aligned",
                "note": "Rows use the supported prediction mode for the active route.",
            },
            "telemetry": {
                "status": "observed",
                "note": "MojoGP telemetry is observed through the benchmark subprocess wrapper.",
            },
        },
    }


def _case_id(feature_surface: str, method: str, n_train: int) -> str:
    return f"mojogp.extended_feature_scaling.{feature_surface}.{method}.n{n_train}"


def _group_id(feature_surface: str, method: str) -> str:
    return f"mojogp.extended_feature_scaling.{feature_surface}.{method}"


def _run_single_output_nonzero_mean(
    *, method: str, n_train: int, tier: str, variety: str, size_role: str, results_dir
) -> BenchmarkResult:
    prediction_mode = "exact" if method == "materialized" else "love"
    return _run_scaling_case(
        method,
        n_train,
        5,
        framework="mojogp",
        prediction_mode=prediction_mode,
        tier=tier,
        benchmark_variety=variety,
        benchmark_track="scaling",
        n_selection_policy="extended_feature_custom",
        size_role=size_role,
        max_iterations=60,
        enable_early_stopping=False,
        benchmark_name="single_output_mean_noise_scaling",
        data_options={
            "dataset_family": "structured_function",
            "function_type": "smooth",
            "noise_level": "high",
            "mean_offset": 1.75,
            "feature_surface": "single_output_nonzero_mean",
            "feature_variant": "structured_high_noise_mean_offset",
            "model_family": "SingleOutputGP",
        },
        results_dir=results_dir,
    )


def _run_per_task_noise(
    *, method: str, n_train: int, tier: str, variety: str, results_dir
) -> BenchmarkResult:
    feature_surface = "multi_output_per_task_noise"
    config = _feature_config(
        feature_surface=feature_surface,
        method=method,
        n_train=n_train,
        tier=tier,
        variety=variety,
    ) | {
        "framework": "mojogp",
        "model_family": "MultiOutputGP",
        "model_type": "MultiOutputGP",
        "kernel": "rbf",
        "prediction_mode": "love",
        "d": 5,
        "num_tasks": 3,
        "dataset_family": "structured_per_task_noise",
    }
    return run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_multi_output_per_task_noise_case",
        payload={
            "kernel": "rbf",
            "n_train": n_train,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "d": 5,
            "num_tasks": 3,
            "task_correlation": "medium",
            "noise_profile": "strong",
            "mean_profile": "offset",
            "method": method,
            "dataset_family": "structured",
            "extra_config": config,
        },
        suite_name="extended_feature_scaling",
        benchmark_name="multi_output_per_task_noise_scaling",
        framework="mojogp",
        case_id=_case_id(feature_surface, method, n_train),
        benchmark_group_id=_group_id(feature_surface, method),
        config=config,
        results_dir=results_dir,
        timeout=1800,
    )


def _run_mixed(
    *, method: str, n_train: int, tier: str, variety: str, results_dir
) -> BenchmarkResult:
    feature_surface = "mixed_continuous_categorical"
    config = _feature_config(
        feature_surface=feature_surface,
        method=method,
        n_train=n_train,
        tier=tier,
        variety=variety,
    ) | {
        "framework": "mojogp",
        "model_family": "SingleOutputGP",
        "model_type": "SingleOutputGP",
        "kernel": "rbf_x_ehh3",
        "prediction_mode": "love",
        "d": 3,
        "continuous_dim": 2,
        "num_categorical": 1,
        "categorical_levels": [3],
    }
    return run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_mixed_kernel_case",
        payload={
            "method": method,
            "n_train": n_train,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "cont_dim": 2,
            "cat_levels": [3],
            "max_iterations": 50 if method == "materialized" else 40,
            "learning_rate": 0.03,
            "extra_config": config,
        },
        suite_name="extended_feature_scaling",
        benchmark_name="mixed_continuous_categorical_scaling",
        framework="mojogp",
        case_id=_case_id(feature_surface, method, n_train),
        benchmark_group_id=_group_id(feature_surface, method),
        config=config,
        results_dir=results_dir,
        timeout=1800,
    )


def _run_lmc(
    *, method: str, n_train: int, tier: str, variety: str, results_dir
) -> BenchmarkResult:
    feature_surface = "multi_output_lmc"
    config = _feature_config(
        feature_surface=feature_surface,
        method=method,
        n_train=n_train,
        tier=tier,
        variety=variety,
    ) | {
        "framework": "mojogp",
        "model_family": "MultiOutputLMCGP",
        "model_type": "MultiOutputLMCGP",
        "kernel": "rbf_plus_matern52",
        "prediction_mode": "love",
        "d": 1,
        "num_tasks": 3,
    }
    return run_result_benchmark_subprocess(
        module="tests.system_benchmarks.run_lmc_accuracy_case",
        payload={
            "method": method,
            "n_train": n_train,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "max_iterations": 30 if method == "materialized" else 25,
            "learning_rate": 0.03 if method == "materialized" else 0.02,
            "extra_config": config,
        },
        suite_name="extended_feature_scaling",
        benchmark_name="multi_output_lmc_scaling",
        framework="mojogp",
        case_id=_case_id(feature_surface, method, n_train),
        benchmark_group_id=_group_id(feature_surface, method),
        config=config,
        results_dir=results_dir,
        timeout=2400,
    )


@pytest.mark.minimal
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_extended_feature_scaling(results_dir, method: str):
    assert_gpu_available()
    variety = _benchmark_variety()
    tier, _ = _tier_for_method(method)
    results: list[BenchmarkResult] = []

    runners = {
        "single_output_nonzero_mean": _run_single_output_nonzero_mean,
        "multi_output_per_task_noise": _run_per_task_noise,
        "mixed_continuous_categorical": _run_mixed,
        "multi_output_lmc": _run_lmc,
    }
    for feature_surface, runner in runners.items():
        sizes = _target_sizes(feature_surface, method, variety, tier)
        for n_train in sizes:
            kwargs = {
                "method": method,
                "n_train": n_train,
                "tier": tier,
                "variety": variety,
                "results_dir": results_dir,
            }
            if feature_surface == "single_output_nonzero_mean":
                kwargs["size_role"] = _size_role(n_train, sizes)
            result = runner(**kwargs)
            print_result(
                result,
                title=(
                    f"Extended feature scaling: {feature_surface} {method} "
                    f"n={n_train} actual_n={result.config.get('n')}"
                ),
            )
            assert_gpu_was_used(result)
            assert np.isfinite(result.accuracy.rmse)
            assert np.isfinite(result.hyperparameters.final_nll)
            results.append(result)

    save_summary_report(
        results,
        results_dir,
        f"extended_feature_scaling_{variety}_{method}",
    )
    assert results, f"No extended feature scaling results collected for {method}"
