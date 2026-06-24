"""Certification-style ablations for SingleOutputGP categorical kernels."""

from __future__ import annotations

import pytest

from tests.certification.categorical_ablation_utils import CATEGORICAL_KERNELS
from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.certification.run_categorical_ablation_case"


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
@pytest.mark.parametrize("kernel", CATEGORICAL_KERNELS)
def test_categorical_kernel_beats_continuous_and_shuffled_controls(kernel: str, method: str):
    evidence = run_isolated_case(
        module=MODULE,
        payload={
            "kernel": kernel,
            "method": method,
            "seed": 1_000 + 37 * CATEGORICAL_KERNELS.index(kernel) + (0 if method == "materialized" else 10_000),
        },
        timeout=600,
        description=f"Runs categorical ablation certification {kernel}/{method}",
    )
    metrics = evidence["metrics"]
    routes = evidence["routes"]

    assert evidence["passed"], evidence
    assert evidence["n_train"] >= 2000
    assert metrics["mixed_rmse"] <= 0.95 * metrics["baseline_rmse"]
    assert metrics["mixed_rmse"] <= 0.95 * metrics["shuffled_rmse"]
    assert metrics["category_effect_corr"] >= 0.5
    assert metrics["save_load_max_abs_diff"] <= 1e-3
    assert routes["training_route"] == method
    assert routes["prediction_route"] == "predict_mixed"
    assert routes["backend_prediction_used"] is True
    assert routes["fallback_used"] is False
