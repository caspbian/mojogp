"""Single-output continuous variance, sampling, and polynomial contracts."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mojogp import SingleOutputGP, Kernel


def _skip_if_no_engine():
    if importlib.util.find_spec("mojogp_jit_engine") is None:
        pytest.skip("mojogp_jit_engine not built; run `task build` to enable")


def _make_polynomial_data(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.5, size=(n, 2)).astype(np.float32)
    signal = (0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.5) ** 2
    y = (signal + 0.02 * rng.normal(size=n)).astype(np.float32)
    X_test = rng.normal(scale=0.5, size=(8, 2)).astype(np.float32)
    return X, y, X_test


def _dense_observed_posterior(gp: SingleOutputGP, X_train, y_train, X_test):
    params = np.asarray(gp.training_result.params, dtype=np.float32)
    mean = float(gp.training_result.mean)
    noise = float(gp.training_result.noise)

    K_train = gp.kernel.evaluate(X_train, X_train, params=params)
    K_cross = gp.kernel.evaluate(X_train, X_test, params=params)
    K_test = gp.kernel.evaluate(X_test, X_test, params=params)
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


def test_polynomial_exact_prediction_matches_dense_reference():
    _skip_if_no_engine()
    X, y, X_test = _make_polynomial_data(seed=111)
    gp = SingleOutputGP(
        Kernel.polynomial(degree=2.0, offset=2.0, outputscale=1.0),
        verbose=False,
    )
    gp.fit(
        X,
        y,
        max_iterations=4,
        learning_rate=0.02,
        method="materialized",
        num_probes=3,
        max_cg_iterations=40,
        preconditioner_rank=8,
        verbose=False,
    )

    pred = gp.predict(
        X_test[:6],
        variance_method="exact",
        max_cg_iterations=60,
        cg_tolerance=1e-3,
        preconditioner_rank=8,
    )
    ref_mean, ref_var = _dense_observed_posterior(gp, X, y, X_test[:6])

    assert gp.backend_predict_info["variance_method"] == "exact"
    assert gp.backend_predict_info["actual_variance_route"] == "predict"
    assert float(np.sqrt(np.mean((pred.mean - ref_mean) ** 2))) < 5e-4
    assert float(np.sqrt(np.mean((pred.variance - ref_var) ** 2))) < 1e-2


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_polynomial_love_variance_metadata_and_bounds_for_both_training_routes(method):
    _skip_if_no_engine()
    X, y, X_test = _make_polynomial_data(seed=123)
    gp = SingleOutputGP(
        Kernel.polynomial(degree=2.0, offset=2.0, outputscale=1.0),
        verbose=False,
    )
    gp.fit(
        X,
        y,
        max_iterations=4,
        learning_rate=0.02,
        method=method,
        num_probes=3,
        max_cg_iterations=25,
        preconditioner_rank=8,
        verbose=False,
    )

    exact = gp.predict(
        X_test,
        variance_method="exact",
        max_cg_iterations=25,
        preconditioner_rank=8,
    )
    exact_info = dict(gp.backend_predict_info)
    love = gp.predict(
        X_test,
        variance_method="love",
        max_cg_iterations=25,
        preconditioner_rank=8,
    )
    love_info = dict(gp.backend_predict_info)

    np.testing.assert_allclose(love.mean, exact.mean, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(exact.variance))
    assert np.all(np.isfinite(love.variance))
    assert np.all(exact.variance >= 0.0)
    assert np.all(love.variance >= 0.0)
    rel = np.abs(love.variance - exact.variance) / np.maximum(exact.variance, 1e-6)
    assert float(np.mean(rel)) < 0.45
    assert exact_info["training_route"] == method
    assert exact_info["variance_method"] == "exact"
    assert love_info["training_route"] == method
    assert love_info["variance_method"] == "love"
    assert love_info["actual_prediction_route"] == "predict"
    assert love_info["actual_variance_route"] == "predict"
    assert love_info["backend_variance_used"] is True
    assert love_info["fallback_used"] is False


def test_rbf_pathwise_samples_match_mean_and_spatial_correlation_at_realistic_n():
    _skip_if_no_engine()
    rng = np.random.default_rng(77)
    X = rng.normal(size=(2000, 1)).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.03 * rng.normal(size=2000)).astype(np.float32)
    X_test = np.array([[0.0], [0.02], [4.0]], dtype=np.float32)

    gp = SingleOutputGP(Kernel.rbf(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=6,
        learning_rate=0.03,
        method="matrix_free",
        num_probes=3,
        max_cg_iterations=25,
        preconditioner_rank=8,
        verbose=False,
    )
    pred = gp.predict(
        X_test,
        variance_method="exact",
        max_cg_iterations=25,
        preconditioner_rank=8,
    )
    samples = gp.sample_posterior(
        X_test,
        n_samples=256,
        method="pathwise",
        n_rff_features=1024,
        rng=np.random.default_rng(5),
    )

    assert samples.shape == (256, 3)
    assert np.all(np.isfinite(samples))
    assert float(np.max(np.abs(samples.mean(axis=0) - pred.mean))) < 0.12
    close_corr = float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])
    far_corr = float(np.corrcoef(samples[:, 0], samples[:, 2])[0, 1])
    assert close_corr > 0.8
    assert close_corr > far_corr + 0.2
    info = gp.backend_sample_info
    assert info["actual_sampling_route"] == "provider_pathwise"
    assert info["backend_correction_route"] == "predict"
    assert info["training_route"] == "matrix_free"


def test_polynomial_pathwise_sampling_survives_save_load_and_matches_mean(tmp_path):
    _skip_if_no_engine()
    X, y, X_test = _make_polynomial_data(seed=321)
    gp = SingleOutputGP(
        Kernel.polynomial(degree=2.0, offset=2.0, outputscale=1.0),
        verbose=False,
    )
    gp.fit(
        X,
        y,
        max_iterations=4,
        learning_rate=0.02,
        method="materialized",
        num_probes=3,
        max_cg_iterations=25,
        preconditioner_rank=8,
        verbose=False,
    )

    pred = gp.predict(
        X_test[:5],
        variance_method="exact",
        max_cg_iterations=25,
        preconditioner_rank=8,
    )
    samples = gp.sample_posterior(
        X_test[:5],
        n_samples=64,
        method="pathwise",
        rng=np.random.default_rng(2),
    )
    sample_info = dict(gp.backend_sample_info)
    assert float(np.max(np.abs(samples.mean(axis=0) - pred.mean))) < 0.05
    assert sample_info["actual_sampling_route"] == "provider_pathwise"
    assert sample_info["prior_sampler_family"] == "shared_feature_map"

    model_path = tmp_path / "single_output_polynomial_pathwise"
    gp.save(str(model_path))
    loaded = SingleOutputGP.load(str(model_path))
    loaded_samples = loaded.sample_posterior(
        X_test[:5],
        n_samples=64,
        method="pathwise",
        rng=np.random.default_rng(2),
    )
    assert np.all(np.isfinite(loaded_samples))
    assert float(
        np.max(np.abs(loaded_samples.mean(axis=0) - samples.mean(axis=0)))
    ) < 0.02
    assert loaded.backend_sample_info["actual_sampling_route"] == "provider_pathwise"


def test_polynomial_training_keeps_degree_fixed_and_updates_trainable_scale():
    _skip_if_no_engine()
    rng = np.random.default_rng(987)
    X = rng.normal(scale=0.5, size=(2000, 2)).astype(np.float32)
    signal = (0.2 * X[:, 0] + 0.1 * X[:, 1] + 0.8) ** 3
    y = (signal + 0.02 * rng.normal(size=2000)).astype(np.float32)

    gp = SingleOutputGP(
        Kernel.polynomial(degree=3.0, offset=2.0, outputscale=0.7),
        verbose=False,
    )
    result = gp.fit(
        X,
        y,
        max_iterations=10,
        learning_rate=0.03,
        method="materialized",
        num_probes=3,
        max_cg_iterations=25,
        preconditioner_rank=8,
        verbose=False,
    )

    params = np.asarray(gp.training_result.params, dtype=np.float32)
    learned = gp.get_learned_params()
    assert params[0] == pytest.approx(3.0, abs=0.0)
    assert abs(float(params[2]) - 0.7) > 0.03
    assert np.all(np.isfinite(result.nll_history))
    assert learned["polynomial_degree"] == pytest.approx(3.0, abs=0.0)
    assert learned["polynomial_outputscale"] == pytest.approx(float(params[2]))
