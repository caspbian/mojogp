from __future__ import annotations

import os

import pytest

from tests.benchmarks.gpytorch.multi_output_scaling import (
    run_gpytorch_multi_output_scaling_module,
)
from tests.benchmarks.mojogp.multi_output_scaling import (
    run_mojogp_multi_output_scaling_module,
)
from tests.benchmarks.multi_output_timeout_policy import multi_output_scaling_timeout_s


def _clear_multi_output_timeout_env(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("MOJOGP_MULTI_OUTPUT_SCALING_TIMEOUT_S"):
            monkeypatch.delenv(name, raising=False)


def test_multi_output_matrix_free_exact_timeout_default_exceeds_previous_cap(monkeypatch):
    _clear_multi_output_timeout_env(monkeypatch)

    timeout_s = multi_output_scaling_timeout_s(
        framework="mojogp",
        method="matrix_free",
        prediction_mode="exact",
        tier="large",
    )

    assert timeout_s == 7200
    assert timeout_s > 1800


def test_multi_output_timeout_global_env_override(monkeypatch):
    monkeypatch.setenv("MOJOGP_MULTI_OUTPUT_SCALING_TIMEOUT_S", "5400")

    timeout_s = multi_output_scaling_timeout_s(
        framework="mojogp",
        method="materialized",
        prediction_mode="love",
        tier="large",
    )

    assert timeout_s == 5400


def test_multi_output_timeout_lane_env_override_precedes_global(monkeypatch):
    monkeypatch.setenv("MOJOGP_MULTI_OUTPUT_SCALING_TIMEOUT_S", "5400")
    monkeypatch.setenv("MOJOGP_MULTI_OUTPUT_SCALING_TIMEOUT_S_MATRIX_FREE_EXACT", "9000")

    timeout_s = multi_output_scaling_timeout_s(
        framework="mojogp",
        method="matrix_free",
        prediction_mode="exact",
        tier="large",
    )

    assert timeout_s == 9000


@pytest.mark.parametrize(
    ("module_path", "runner", "framework"),
    [
        (
            "tests.benchmarks.mojogp.multi_output_scaling",
            run_mojogp_multi_output_scaling_module,
            "mojogp",
        ),
        (
            "tests.benchmarks.gpytorch.multi_output_scaling",
            run_gpytorch_multi_output_scaling_module,
            "gpytorch",
        ),
    ],
)
def test_multi_output_wrappers_pass_effective_timeout(monkeypatch, module_path, runner, framework):
    captured = {}

    def fake_run_benchmark_module(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(f"{module_path}.run_benchmark_module", fake_run_benchmark_module)
    monkeypatch.setenv("MOJOGP_MULTI_OUTPUT_SCALING_TIMEOUT_S_MATRIX_FREE_EXACT", "4321")

    result = runner(
        payload={
            "framework": framework,
            "prediction_mode": "exact",
            "method": "matrix_free",
            "n_train": 50000,
            "d": 12,
            "num_tasks": 6,
            "tier": "large",
        },
        session_store=object(),
        session_id="session",
        case_id="case",
        benchmark_group_id="group",
        benchmark_name="multi_output_scaling",
        git=object(),
        profiling=object(),
        config={},
        dataset_id="dataset",
        comparison_id=None,
    )

    assert result is not None
    assert captured["timeout"] == 4321
    assert captured["config"]["timeout_s"] == 4321
