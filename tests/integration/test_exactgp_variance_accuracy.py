"""Integration tests for ExactGP variance surfaces.

These tests keep the evidence focused on the live wrapper API: exact and LOVE
variance should both be finite, positive, and reasonably aligned.
"""

import numpy as np
import pytest

from mojogp import SingleOutputGP, RBF
from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.integration.run_exactgp_variance_accuracy_case"


def _make_data(n_train: int = 2000, n_test: int = 256, seed: int = 0):
    rng = np.random.RandomState(seed)
    X_train = rng.uniform(-2.0, 2.0, size=(n_train, 1)).astype(np.float32)
    y_train = (np.sin(1.5 * X_train[:, 0]) + 0.05 * rng.randn(n_train)).astype(
        np.float32
    )
    X_test = np.linspace(-2.2, 2.2, n_test, dtype=np.float32).reshape(-1, 1)
    return X_train, y_train, X_test


def _closed_form_observed_posterior(gp: SingleOutputGP, X_train, y_train, X_test):
    params = np.asarray(gp.training_result.params, dtype=np.float32)
    mean = float(gp.training_result.mean)
    noise = float(gp.training_result.noise)

    K_train = gp.kernel.evaluate(X_train, params=params)
    K_cross = gp.kernel.evaluate(X_train, X_test, params=params)
    K_test = gp.kernel.evaluate(X_test, params=params)
    K_reg = K_train + noise * np.eye(X_train.shape[0], dtype=np.float32)
    centered_y = y_train.astype(np.float32) - np.float32(mean)
    alpha = np.linalg.solve(K_reg.astype(np.float64), centered_y.astype(np.float64))
    mean_pred = mean + K_cross.T.astype(np.float64) @ alpha
    solve_cross = np.linalg.solve(K_reg.astype(np.float64), K_cross.astype(np.float64))
    latent_var = np.diag(K_test.astype(np.float64)) - np.sum(
        K_cross.astype(np.float64) * solve_cross,
        axis=0,
    )
    observed_var = latent_var + noise
    return mean_pred.astype(np.float32), observed_var.astype(np.float32)


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_exact_and_love_variance_are_finite_and_reasonably_aligned(method):
    X_train, y_train, X_test = _make_data(seed=11 if method == "materialized" else 17)

    gp = SingleOutputGP(RBF(), verbose=False)
    gp.fit(
        X_train,
        y_train,
        max_iterations=6,
        learning_rate=0.03,
        method=method,
        num_probes=5,
        max_cg_iterations=25,
        preconditioner_rank=8,
    )

    prediction_kwargs = {
        "max_cg_iterations": 25,
        "cg_tolerance": 1e-2,
        "preconditioner_rank": 8,
    }
    exact = gp.predict(X_test, variance_method="exact", **prediction_kwargs)
    love = gp.predict(X_test, variance_method="love", **prediction_kwargs)

    assert exact.mean.shape == love.mean.shape == (X_test.shape[0],)
    assert exact.variance.shape == love.variance.shape == (X_test.shape[0],)
    assert np.all(np.isfinite(exact.variance))
    assert np.all(np.isfinite(love.variance))
    assert np.all(exact.variance >= 0.0)
    assert np.all(love.variance >= 0.0)

    mean_abs_diff = float(np.mean(np.abs(exact.mean - love.mean)))
    mean_var_ratio = float(
        np.mean(love.variance) / max(float(np.mean(exact.variance)), 1e-6)
    )
    rel_err = np.abs(exact.variance - love.variance) / np.maximum(
        exact.variance, 1e-6
    )
    pct_within_50 = float(np.mean(rel_err < 0.5) * 100.0)

    assert mean_abs_diff < 1e-4
    assert pct_within_50 > 90.0
    assert 0.15 < mean_var_ratio < 5.0


def test_default_exact_and_love_use_same_mean_solve_budget_materialized():
    """Default exact/LOVE prediction must not change the posterior mean."""
    rng = np.random.default_rng(123)
    n_train = 2000
    d = 5
    X_train = rng.normal(size=(n_train, d)).astype(np.float32)
    weights = np.linspace(0.3, 1.1, d, dtype=np.float32)
    y_train = (
        np.sin(X_train @ weights)
        + 0.05 * rng.normal(size=n_train).astype(np.float32)
    ).astype(np.float32)
    X_test = rng.normal(size=(32, d)).astype(np.float32)

    gp = SingleOutputGP(RBF(), verbose=False)
    gp.fit(X_train, y_train, method="materialized", max_iterations=5, verbose=False)

    gp._invalidate_prediction_caches()
    mu_love, _ = gp.predict(X_test, variance_method="love", return_std=True)
    love_info = dict(gp._backend_predict_info)

    gp._invalidate_prediction_caches()
    mu_exact, _ = gp.predict(X_test, variance_method="exact", return_std=True)
    exact_info = dict(gp._backend_predict_info)

    assert love_info["max_cg_iterations"] == exact_info["max_cg_iterations"]
    assert love_info["cg_tolerance"] == pytest.approx(exact_info["cg_tolerance"])
    np.testing.assert_allclose(
        mu_love,
        mu_exact,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Default LOVE and exact prediction should use the same alpha solve",
    )


def test_exact_prediction_matches_closed_form_observed_posterior_materialized():
    metrics = run_isolated_case(
        module=MODULE,
        payload={"case": "closed_form_materialized"},
        timeout=600,
        description="Runs exact-vs-closed-form variance comparison",
    )

    assert metrics["mean_rmse"] < 2e-2
    assert metrics["var_rmse"] < 1e-2


def test_exactgp_sampling_metadata_tracks_diagonal_and_pathwise_routes():
    X_train, y_train, X_test = _make_data(seed=23)

    gp = SingleOutputGP(RBF(), verbose=False)
    gp.fit(
        X_train,
        y_train,
        max_iterations=6,
        learning_rate=0.03,
        method="matrix_free",
        num_probes=3,
        max_cg_iterations=25,
        preconditioner_rank=8,
    )

    diag_samples = gp.sample_posterior(X_test[:16], n_samples=4, method="diagonal")
    diag_info = dict(gp.backend_sample_info)
    pathwise_samples = gp.sample_posterior(
        X_test[:16], n_samples=4, method="pathwise", n_rff_features=256
    )
    pathwise_info = dict(gp.backend_sample_info)

    assert diag_samples.shape == (4, 16)
    assert pathwise_samples.shape == (4, 16)
    assert diag_info["actual_sampling_method"] == "diagonal"
    assert diag_info["backend_sampling_used"] is True
    assert pathwise_info["actual_sampling_method"] == "pathwise"
    assert pathwise_info["backend_correction_used"] is True
    assert pathwise_info["actual_sampling_route"] == "provider_pathwise"
