"""Extensive single-output scaling and preset benchmarks.

This suite complements `test_scaling_certification_harness.py`:

1. the standard certification suite stays small and frequently rerunnable
2. this extensive suite explores broader `n` and `d` scaling behavior
3. MojoGP presets are benchmarked explicitly here rather than folded into the
   standard benchmark gate
"""

from __future__ import annotations

import traceback

import pytest

from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda
from .extensive_scaling_report import save_extensive_scaling_summary
from tests.shared.benchmarking.report import print_result
from .test_scaling_certification_harness import (
    _benchmark_targets,
    _run_scaling_case,
    _warm_up_scaling_route,
)


MOJOGP_PRESETS = ["fast", "balanced", "accurate"]


def _route_prediction_mode(method: str) -> str:
    return "exact" if method == "materialized" else "love"


def _run_extensive_case_capture(
    *,
    method: str,
    n_train: int,
    d: int,
    framework: str,
    tier: str,
    benchmark_name: str,
    max_size: int,
    results_dir,
    mojogp_preset: str | None = None,
):
    prediction_mode = _route_prediction_mode(method)
    try:
        result = _run_scaling_case(
            method,
            n_train,
            d,
            framework=framework,
            prediction_mode=prediction_mode,
            tier=tier,
            benchmark_variety="extensive",
            benchmark_track="scaling",
            n_selection_policy=(
                "vram" if method == "materialized" else "bandwidth"
            ),
            size_role="envelope" if n_train == max_size else "anchor",
            benchmark_name=benchmark_name,
            mojogp_preset=mojogp_preset,
            results_dir=results_dir,
        )
        result.config.update(
            {
                "benchmark_tier": tier,
                "benchmark_track": "scaling",
                "benchmark_variety": "extensive",
                "benchmark_suite": benchmark_name,
                "mojogp_preset": mojogp_preset,
            }
        )
        return result, None
    except Exception as exc:
        failure = {
            "framework": framework,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "mojogp_preset": mojogp_preset,
            "benchmark_track": "scaling",
            "benchmark_variety": "extensive",
            "n": n_train,
            "d": d,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if framework == "gpytorch":
            return None, failure
        raise


@pytest.mark.full
@pytest.mark.single_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_single_output_extensive_scaling(results_dir, method: str):
    assert_gpu_available()
    sizes, dims, tier, _ = _benchmark_targets(method, benchmark_variety="extensive")

    results = []
    failures = []
    benchmark_name = f"single_output_extensive_scaling_{method}"

    for d in dims:
        _warm_up_scaling_route(method, d)
        for framework in ["mojogp", "gpytorch"]:
            for n_train in sizes:
                result, failure = _run_extensive_case_capture(
                    method=method,
                    n_train=n_train,
                    d=d,
                    framework=framework,
                    tier=tier,
                    benchmark_name=benchmark_name,
                    max_size=sizes[-1],
                    results_dir=results_dir,
                )
                if failure is not None:
                    failures.append(failure)
                    continue
                assert result is not None
                print_result(
                    result,
                    title=(
                        f"Extensive scaling: {framework} {method} "
                        f"pred={result.config['prediction_mode']} n={n_train} d={d}"
                    ),
                )
                assert_gpu_was_used(result)
                results.append(result)

    save_extensive_scaling_summary(results, failures, results_dir, benchmark_name)

    assert results, f"No successful results collected for {benchmark_name}"
    matrix_free_gpytorch_envelope = [
        r
        for r in results
        if r.config["framework"] == "gpytorch"
        and r.config["training_method"] == "matrix_free"
        and r.config["n"] == sizes[-1]
    ]
    if method == "matrix_free":
        assert matrix_free_gpytorch_envelope or any(
            f["framework"] == "gpytorch"
            and f["training_method"] == "matrix_free"
            and f["n"] == sizes[-1]
            for f in failures
        ), "Expected a GPyTorch matrix-free envelope attempt to be recorded"


@pytest.mark.full
@pytest.mark.single_output
@pytest.mark.speed
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_single_output_mojogp_preset_sweep(results_dir, method: str):
    assert_gpu_available()
    sizes, dims, tier, _ = _benchmark_targets(method, benchmark_variety="extensive")
    benchmark_name = f"single_output_preset_sweep_{method}"
    presets = list(MOJOGP_PRESETS)

    # Keep the preset sweep tractable: representative small+large route points.
    if len(sizes) > 2:
        sizes = [sizes[0], sizes[-1]]

    results = []
    failures = []
    for d in dims:
        _warm_up_scaling_route(method, d)
        for preset in presets:
            for n_train in sizes:
                result, failure = _run_extensive_case_capture(
                    method=method,
                    n_train=n_train,
                    d=d,
                    framework="mojogp",
                    tier=tier,
                    benchmark_name=benchmark_name,
                    results_dir=results_dir,
                    mojogp_preset=preset,
                    max_size=sizes[-1],
                )
                if failure is not None:
                    failures.append(failure)
                    continue
                assert result is not None
                print_result(
                    result,
                    title=(
                        f"Preset sweep: mojogp {method} preset={preset} "
                        f"pred={result.config['prediction_mode']} n={n_train} d={d}"
                    ),
                )
                assert_gpu_was_used(result)
                results.append(result)

    save_extensive_scaling_summary(results, failures, results_dir, benchmark_name)
    assert results, f"No successful results collected for {benchmark_name}"
