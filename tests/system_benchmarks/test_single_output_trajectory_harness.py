"""Single-output trajectory benchmarks with fixed iteration checkpoints."""

from __future__ import annotations

import numpy as np
import pytest

from tests.benchmarks.comparison_policy import policy_for
from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda
from tests.shared.benchmarking.report import print_result, save_summary_report
from .test_scaling_certification_harness import (
    _allow_recorded_failure,
    _benchmark_targets,
    _benchmark_variety,
    _framework_sizes,
    _run_scaling_case_capture,
    _size_role,
    _warm_up_scaling_route,
)


TRAJECTORY_CHECKPOINTS = [10, 25, 50, 100]


def _trajectory_prediction_mode(method: str) -> str:
    return "exact" if method == "materialized" else "love"


@pytest.mark.minimal
@pytest.mark.single_output
@pytest.mark.speed
@pytest.mark.memory
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_single_output_trajectory(results_dir, method: str):
    assert_gpu_available()
    policy = policy_for("single_output_trajectory")
    benchmark_variety = _benchmark_variety()
    _, dims, tier, n_selection_policy = _benchmark_targets(
        method,
        benchmark_variety=benchmark_variety,
    )
    prediction_mode = _trajectory_prediction_mode(method)

    results = []
    failures = []
    benchmark_name = "single_output_trajectory"

    for d in dims:
        _warm_up_scaling_route(method, d)
        frameworks = ["mojogp", "gpytorch"] if policy.published_cross_framework else ["mojogp"]
        for framework in frameworks:
            framework_sizes = _framework_sizes(
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                tier=tier,
                benchmark_variety=benchmark_variety,
            )
            for n_train in framework_sizes:
                size_role = _size_role(n_train, framework_sizes)
                allow_recorded_failure = _allow_recorded_failure(
                    framework=framework,
                    method=method,
                    prediction_mode=prediction_mode,
                    benchmark_variety=benchmark_variety,
                    size_role=size_role,
                )
                for checkpoint in TRAJECTORY_CHECKPOINTS:
                    result, failure = _run_scaling_case_capture(
                        method,
                        n_train,
                        d,
                        framework=framework,
                        prediction_mode=prediction_mode,
                        tier=tier,
                        benchmark_variety=benchmark_variety,
                        benchmark_track="trajectory",
                        n_selection_policy=n_selection_policy,
                        size_role=size_role,
                        allow_recorded_failure=allow_recorded_failure,
                        max_iterations=checkpoint,
                        enable_early_stopping=False,
                        benchmark_name=benchmark_name,
                        results_dir=results_dir,
                    )
                    if failure is not None:
                        failure["checkpoint_budget"] = checkpoint
                        failures.append(failure)
                        break

                    assert result is not None
                    result.config.update(
                        {
                            "benchmark_tier": tier,
                            "benchmark_track": "trajectory",
                            "benchmark_variety": benchmark_variety,
                            "trajectory_mode": "fixed_iteration",
                            "checkpoint_budget": checkpoint,
                        }
                    )
                    print_result(
                        result,
                        title=(
                            f"Trajectory: {framework} {method} pred={prediction_mode} "
                            f"n={n_train} d={d} checkpoint={checkpoint} "
                            f"variety={benchmark_variety} role={size_role}"
                        ),
                    )
                    assert_gpu_was_used(result)
                    results.append(result)

    save_summary_report(
        results,
        results_dir,
        f"single_output_trajectory_{benchmark_variety}_{method}",
        failures=failures,
    )

    assert results, f"No successful results collected for {benchmark_name}:{method}"

    for result in results:
        checkpoint = int(result.config["checkpoint_budget"])
        assert result.speed.max_iterations == checkpoint
        assert result.speed.iterations_run == checkpoint
        assert not result.speed.early_stopped
        assert np.isfinite(result.hyperparameters.final_nll)
        assert np.isfinite(result.accuracy.rmse)
        assert np.isfinite(result.accuracy.crps)
        nll_history = result.config.get("training_nll_history") or []
        assert len(nll_history) == checkpoint, (
            "Trajectory row did not persist the full NLL history for "
            f"framework={result.config['framework']} method={method} n={result.config['n']} "
            f"d={result.config['d']} checkpoint={checkpoint}"
        )

    for d in dims:
        frameworks = ["mojogp", "gpytorch"] if policy.published_cross_framework else ["mojogp"]
        for framework in frameworks:
            framework_sizes = _framework_sizes(
                framework=framework,
                method=method,
                prediction_mode=prediction_mode,
                tier=tier,
                benchmark_variety=benchmark_variety,
            )
            for n_train in framework_sizes:
                size_role = _size_role(n_train, framework_sizes)
                allow_recorded_failure = _allow_recorded_failure(
                    framework=framework,
                    method=method,
                    prediction_mode=prediction_mode,
                    benchmark_variety=benchmark_variety,
                    size_role=size_role,
                )
                per_lane = [
                    result
                    for result in results
                    if result.config["framework"] == framework
                    and result.config["d"] == d
                    and result.config["n"] == n_train
                ]
                if per_lane:
                    observed = sorted(
                        int(result.config["checkpoint_budget"]) for result in per_lane
                    )
                    assert observed == TRAJECTORY_CHECKPOINTS, (
                        "Trajectory lane is missing checkpoints for "
                        f"framework={framework} method={method} n={n_train} d={d}"
                    )
                elif allow_recorded_failure and size_role == "envelope":
                    assert any(
                        failure["framework"] == framework
                        and failure["d"] == d
                        and failure["n"] == n_train
                        for failure in failures
                    ), (
                        "Expected a recorded trajectory failure for the matrix-free "
                        f"envelope lane framework={framework} n={n_train} d={d}"
                    )
