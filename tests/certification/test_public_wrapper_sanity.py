"""Broad public wrapper sanity checks for user-facing workflows."""

from __future__ import annotations

import pytest

from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.certification.run_public_wrapper_sanity_case"

SURFACES = [
    "single_continuous",
    "single_mixed",
    "icm_continuous",
    "icm_mixed",
    "lmc_continuous",
    "lmc_mixed",
]


@pytest.mark.parametrize(
    "surface",
    SURFACES,
)
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_public_wrapper_fit_predict_sample_save_load_sanity(surface: str, method: str):
    evidence = run_isolated_case(
        module=MODULE,
        payload={
            "surface": surface,
            "method": method,
            "seed": 30_000
            + 101 * SURFACES.index(surface)
            + (0 if method == "materialized" else 10_000),
        },
        timeout=900,
        description=f"Runs public wrapper sanity workflow {surface}/{method}",
    )

    assert evidence["training_route"] == method
    assert evidence["save_load_max_abs_diff"] <= 1e-3
    assert evidence["love_fallback"] is False
    assert evidence["observed_variance_delta_min"] >= -1e-5
    assert tuple(evidence["pathwise_sample_shape"])[0] == 2
    assert evidence["pathwise_route"] == "provider_pathwise"

    if surface == "single_continuous":
        assert evidence["exact_route"] == "predict"
    elif surface == "single_mixed":
        assert evidence["exact_route"] == "predict_mixed"
    elif surface == "icm_continuous":
        assert evidence["exact_route"] == "predict_multi_output"
    elif surface == "icm_mixed":
        assert evidence["exact_route"] == "predict_multi_output_mixed"
    elif surface == "lmc_continuous":
        assert evidence["exact_route"] == "predict_lmc"
    elif surface == "lmc_mixed":
        assert evidence["exact_route"] == "predict_lmc_mixed"
