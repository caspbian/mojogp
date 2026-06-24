"""Single-output mean-function and noise-learning ablations."""

from __future__ import annotations

import pytest

from tests.benchmarks.comparison_policy import policy_for
from .conftest import assert_gpu_available, assert_gpu_was_used, get_vram_tier, requires_cuda
from tests.shared.benchmarking.data_generators import NOISE_LEVELS
from .extensive_scaling_report import save_extensive_scaling_summary
from tests.shared.benchmarking.report import print_result
from .test_scaling_certification_harness import _quick_enabled, _run_scaling_case, _warm_up_scaling_route


ABLATION_CASES = [
    {"mean_offset": 0.0, "noise_level": "medium"},
    {"mean_offset": 1.75, "noise_level": "medium"},
    {"mean_offset": 1.75, "noise_level": "high"},
]

ABLATION_SIZES = {
    "materialized": {
        "xsmall": 5000,
        "small": 5000,
        "medium": 5000,
        "large": 10000,
        "xlarge": 10000,
    },
    "matrix_free": {
        "xsmall": 10000,
        "small": 25000,
        "medium": 25000,
        "large": 100000,
        "xlarge": 250000,
    },
}

ABLATION_DIMS = {
    "materialized": [5, 17],
    "matrix_free": [5, 17],
}


def _prediction_mode(method: str) -> str:
    return "exact" if method == "materialized" else "love"


def _ablation_data_options(case: dict[str, object]) -> dict[str, object]:
    noise_level = str(case["noise_level"])
    return {
        "dataset_family": "gp_prior",
        "kernel_type": "rbf",
        "true_lengthscale": 1.0,
        "true_outputscale": 1.0,
        "true_noise": float(NOISE_LEVELS[noise_level]),
        "mean_offset": float(case["mean_offset"]),
        "noise_level": noise_level,
    }


def _ablation_max_iterations(method: str) -> int:
    return 150 if method == "materialized" else 200


@pytest.mark.full
@pytest.mark.single_output
@pytest.mark.accuracy
@pytest.mark.speed
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_single_output_mean_noise_ablation(results_dir, method: str):
    assert_gpu_available()
    tier = get_vram_tier()
    n_train = ABLATION_SIZES[method][tier]
    dims = list(ABLATION_DIMS[method])
    cases = list(ABLATION_CASES)
    if _quick_enabled():
        dims = dims[:1]
        cases = cases[:2]

    benchmark_name = f"single_output_mean_noise_ablation_{method}"
    policy = policy_for(benchmark_name)
    max_iterations = _ablation_max_iterations(method)
    results = []
    failures = []
    frameworks = ["mojogp", "gpytorch"] if policy.published_cross_framework else ["mojogp"]
    for d in dims:
        _warm_up_scaling_route(method, d)
        for case in cases:
            for framework in frameworks:
                try:
                    result = _run_scaling_case(
                        method,
                        n_train,
                        d,
                        framework=framework,
                        prediction_mode=_prediction_mode(method),
                        tier=tier,
                        benchmark_name=benchmark_name,
                        max_iterations=max_iterations,
                        enable_early_stopping=False,
                        data_options=_ablation_data_options(case),
                        results_dir=results_dir,
                    )
                except Exception as exc:
                    if framework == "gpytorch":
                        failures.append(
                            {
                                "framework": framework,
                                "training_method": method,
                                "prediction_mode": _prediction_mode(method),
                                "mojogp_preset": None,
                                "n": n_train,
                                "d": d,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "data_mean_offset": case["mean_offset"],
                                "data_noise_level": case["noise_level"],
                            }
                        )
                        continue
                    raise

                result.config.update(
                    {
                        "benchmark_tier": tier,
                        "benchmark_track": "extensive",
                        "benchmark_suite": benchmark_name,
                        "data_mean_offset": float(case["mean_offset"]),
                        "data_noise_level": str(case["noise_level"]),
                        "ablation_max_iterations": max_iterations,
                    }
                )
                print_result(
                    result,
                    title=(
                        f"Mean/noise ablation: {framework} {method} "
                        f"pred={result.config['prediction_mode']} n={n_train} d={d} "
                        f"mean_offset={case['mean_offset']} noise={case['noise_level']}"
                    ),
                )
                assert_gpu_was_used(result)
                results.append(result)

    save_extensive_scaling_summary(results, failures, results_dir, benchmark_name)
    assert results, f"No successful results collected for {benchmark_name}"

    assert any(
        r.hyperparameters.mean_rel_error is not None
        for r in results
        if r.config.get("data_mean_offset") not in (None, 0.0)
    ), "Expected mean recovery errors for non-zero mean-offset ablation cases"
    assert all(
        r.hyperparameters.noise_rel_error is not None for r in results
    ), "Expected noise recovery errors for all ablation results"
    assert all(
        not r.speed.early_stopped for r in results
    ), "Mean/noise ablation should run to the configured iteration budget"
