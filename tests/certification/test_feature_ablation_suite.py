"""Focused certification ablations for high-value public features."""

from __future__ import annotations

import numpy as np
import pytest

from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.certification.run_feature_ablation_case"


def _run(payload: dict[str, object], *, timeout: int = 900):
    return run_isolated_case(
        module=MODULE,
        payload=payload,
        timeout=timeout,
        description=f"Runs feature ablation case {payload}",
    )


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_ard_relevance_beats_isotropic_control(method: str):
    seed = 40_000 + (0 if method == "materialized" else 10_000)
    isotropic = _run({"case": "ard_single", "variant": "isotropic", "method": method, "seed": seed})
    ard = _run({"case": "ard_single", "variant": "ard", "method": method, "seed": seed})

    assert isotropic["training_route"] == method
    assert ard["training_route"] == method
    assert np.isfinite(isotropic["nll"])
    assert np.isfinite(ard["nll"])
    assert isotropic["nll_delta"] > 0.0
    assert ard["nll_delta"] > 0.0
    assert ard["nll"] <= isotropic["nll"]
    assert ard["rmse"] <= 0.95 * isotropic["rmse"]
    assert ard["relevant_lengthscale_max"] < ard["irrelevant_lengthscale_min"]
    assert ard["lengthscale_relevance_ratio"] >= 2.0


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_additive_composite_beats_single_component_controls(method: str):
    seed = 41_000 + (0 if method == "materialized" else 10_000)
    x0_only = _run({"case": "composite_single", "variant": "x0_only", "method": method, "seed": seed})
    x1_only = _run({"case": "composite_single", "variant": "x1_only", "method": method, "seed": seed})
    additive = _run({"case": "composite_single", "variant": "additive", "method": method, "seed": seed})

    assert additive["training_route"] == method
    assert additive["fallback_used"] is False
    assert additive["rmse"] <= 0.95 * min(x0_only["rmse"], x1_only["rmse"])


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_product_composite_beats_additive_and_shuffled_controls(method: str):
    seed = 48_000 + (0 if method == "materialized" else 10_000)
    evidence = _run({"case": "product_composite_single", "variant": "all", "method": method, "seed": seed})
    x0_only = evidence["results"]["x0_only"]
    x1_only = evidence["results"]["x1_only"]
    additive = evidence["results"]["additive"]
    shuffled = evidence["results"]["shuffled_product"]
    product = evidence["results"]["product"]

    assert product["training_route"] == method
    assert product["fallback_used"] is False
    assert product["rmse"] <= 0.6 * min(x0_only["rmse"], x1_only["rmse"], additive["rmse"])
    assert product["rmse"] <= 0.6 * shuffled["rmse"]
    assert product["nll"] <= min(x0_only["nll"], x1_only["nll"], additive["nll"], shuffled["nll"])


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
@pytest.mark.parametrize(
    "surface",
    [
        "single_continuous",
        "single_mixed",
        "icm_continuous",
        "icm_mixed",
        "lmc_continuous",
        "lmc_mixed",
    ],
)
def test_love_variance_tracks_exact_variance(surface: str, method: str):
    seed = 42_000 + 97 * [
        "single_continuous",
        "single_mixed",
        "icm_continuous",
        "icm_mixed",
        "lmc_continuous",
        "lmc_mixed",
    ].index(surface) + (0 if method == "materialized" else 10_000)
    evidence = _run({"case": "love_parity", "surface": surface, "method": method, "seed": seed})

    assert evidence["mean_max_abs_diff"] <= 1e-4
    assert evidence["min_exact_variance"] >= -1e-5
    assert evidence["min_love_variance"] >= -1e-5
    assert evidence["love_fallback_used"] is False
    assert evidence["variance_corr"] >= -0.25
    assert evidence["variance_rel_mae"] <= 1.0
    assert 0.2 <= evidence["variance_mean_ratio"] <= 1.5
    assert 0.0 <= evidence["exact_coverage_95"] <= 1.0
    assert 0.0 <= evidence["love_coverage_95"] <= 1.0
    assert evidence["love_coverage_95"] >= evidence["exact_coverage_95"] - 0.15
    assert np.isfinite(evidence["love_nlpd"])


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
@pytest.mark.parametrize(
    "variant",
    [
        "fixed_vector",
        "fixed_vector_calibration",
        "grouped",
        "input_dependent",
        "learned_input_dependent",
    ],
)
def test_single_output_noise_variants_have_correct_observed_noise_semantics(
    variant: str, method: str
):
    seed = 43_000 + 211 * [
        "fixed_vector",
        "fixed_vector_calibration",
        "grouped",
        "input_dependent",
        "learned_input_dependent",
    ].index(variant) + (0 if method == "materialized" else 10_000)
    evidence = _run({"case": "noise_single", "variant": variant, "method": method, "seed": seed})

    assert evidence["training_route"] == method
    if variant in {"fixed_vector", "grouped", "input_dependent"}:
        assert evidence["observed_delta_max_abs"] <= 2e-4
    if variant == "fixed_vector":
        assert evidence["noise_mode"] == "fixed_vector"
    elif variant == "fixed_vector_calibration":
        assert evidence["noise_mode"] == "fixed_vector"
        assert evidence["fixed_nlpd"] <= evidence["scalar_nlpd"] - 0.05
        assert 0.85 <= evidence["fixed_coverage_95"] <= 1.0
    elif variant == "grouped":
        assert evidence["noise_mode"] == "fixed_grouped"
        assert evidence["grouped_vector_mean_max_abs_diff"] <= 5e-2
    elif variant == "input_dependent":
        assert evidence["noise_mode"] == "fixed_input_dependent"
    else:
        assert evidence["noise_mode"] == "learned_input_dependent"
        assert evidence["has_noise_function_params"] is True
        assert np.isfinite(evidence["learned_noise_low_mean"])
        assert np.isfinite(evidence["learned_noise_high_mean"])
        assert evidence["learned_noise_high_mean"] > evidence["learned_noise_low_mean"]


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_icm_shared_task_structure_learns_correlated_predictions(method: str):
    seed = 44_000 + (0 if method == "materialized" else 10_000)
    independent = _run({"case": "multi_output_structure", "variant": "independent_shared", "method": method, "seed": seed})
    icm = _run({"case": "multi_output_structure", "variant": "icm_shared", "method": method, "seed": seed})

    assert icm["training_route"] == method
    assert icm["task_corr"] > 0.0
    assert icm["prediction_task_corr"] > 0.9
    assert icm["rmse"] <= 0.3
    assert icm["rmse"] <= 6.0 * independent["rmse"]


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_heterogeneous_latents_match_or_beat_icm_control(method: str):
    seed = 45_000 + (0 if method == "materialized" else 10_000)
    icm = _run({"case": "multi_output_structure", "variant": "icm_heterogeneous", "method": method, "seed": seed})
    lmc = _run({"case": "multi_output_structure", "variant": "lmc_heterogeneous", "method": method, "seed": seed})

    assert lmc["training_route"] == method
    assert lmc["rmse"] <= 1.25 * icm["rmse"]


def test_materialized_and_matrix_free_single_output_routes_have_prediction_parity():
    seed = 46_000
    materialized = _run({"case": "route_parity", "variant": "materialized", "seed": seed})
    matrix_free = _run({"case": "route_parity", "variant": "matrix_free", "seed": seed})
    mat_mean = np.asarray(materialized["mean"], dtype=np.float32)
    mf_mean = np.asarray(matrix_free["mean"], dtype=np.float32)

    assert materialized["training_route"] == "materialized"
    assert matrix_free["training_route"] == "matrix_free"
    assert np.max(np.abs(mat_mean - mf_mean)) <= 0.25
    assert matrix_free["rmse"] <= 1.25 * materialized["rmse"]


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_active_dims_and_non_rbf_kernels_beat_wrong_dimension_control(method: str):
    seed = 47_000 + (0 if method == "materialized" else 10_000)
    wrong = _run({"case": "active_dims_nonrbf", "variant": "wrong_dim", "method": method, "seed": seed})
    correct = _run({"case": "active_dims_nonrbf", "variant": "linear_poly_active_dim", "method": method, "seed": seed})

    assert correct["training_route"] == method
    assert correct["rmse"] <= 0.8 * wrong["rmse"]
