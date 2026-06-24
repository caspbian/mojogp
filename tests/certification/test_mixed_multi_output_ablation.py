"""Certification-style ablations for mixed multi-output categorical routes."""

from __future__ import annotations

import pytest

from tests.certification.categorical_ablation_utils import CATEGORICAL_KERNELS
from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.certification.run_mixed_multi_output_ablation_case"


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
@pytest.mark.parametrize("surface", ["icm", "lmc"])
@pytest.mark.parametrize("kernel", CATEGORICAL_KERNELS)
def test_mixed_multi_output_beats_continuous_and_shuffled_controls(
    kernel: str, surface: str, method: str
):
    seed = 20_000 + (0 if surface == "icm" else 1_000) + (
        0 if method == "materialized" else 10_000
    ) + 37 * CATEGORICAL_KERNELS.index(kernel)
    # Keep each LMC wrapper in its own child process; sequential continuous and
    # mixed LMC fits can trip native provider lifecycle state before assertions.
    baseline = run_isolated_case(
        module=MODULE,
        payload={
            "surface": surface,
            "variant": "continuous",
            "method": method,
            "seed": seed,
        },
        timeout=900,
        description=(
            f"Runs mixed multi-output continuous control {surface}/{method}"
        ),
    )
    evidence = run_isolated_case(
        module=MODULE,
        payload={
            "surface": surface,
            "variant": "mixed",
            "kernel": kernel,
            "method": method,
            "seed": seed,
        },
        timeout=900,
        description=f"Runs mixed multi-output ablation certification {surface}/{method}",
    )
    metrics = evidence["metrics"]
    baseline_metrics = baseline["metrics"]
    routes = evidence["routes"]
    expected_route = "predict_multi_output_mixed" if surface == "icm" else "predict_lmc_mixed"

    assert baseline["passed"], baseline
    assert evidence["passed"], evidence
    assert evidence["kernel"] == kernel
    assert evidence["n_train"] >= 2000
    assert evidence["num_tasks"] == 2
    assert metrics["mixed_rmse"] <= 0.95 * baseline_metrics["baseline_rmse"]
    assert metrics["mixed_rmse"] <= 0.95 * metrics["shuffled_rmse"]
    assert metrics["min_category_effect_corr"] >= 0.5
    assert metrics["save_load_max_abs_diff"] <= 1e-3
    assert routes["training_route"] == method
    assert routes["prediction_route"] == expected_route
    assert routes["backend_prediction_used"] is True
    assert routes["fallback_used"] is False
