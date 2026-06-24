"""Integration tests for fixed per-sample-task MultiOutputGP noise."""

import importlib.util

import numpy as np
import pytest

from mojogp import Kernel, MultiOutputGP


def _skip_if_no_engine():
    if importlib.util.find_spec("mojogp_jit_engine") is None:
        pytest.skip("mojogp_jit_engine not built; run `task build` to enable")


def _make_data(n: int = 500, seed: int = 123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2)).astype(np.float32)
    latent = np.sin(X[:, 0]) + 0.25 * X[:, 1]
    clean = np.stack([latent, 0.6 * latent + 0.2 * X[:, 0]], axis=1).astype(
        np.float32
    )
    noise = np.empty_like(clean)
    noise[:, 0] = 0.015 + 0.02 * (X[:, 0] > 0.0)
    noise[:, 1] = 0.03 + 0.025 * (X[:, 1] > 0.0)
    Y = clean + rng.normal(scale=np.sqrt(noise)).astype(np.float32)
    return X, Y.astype(np.float32), noise.astype(np.float32)


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_multi_output_fixed_per_sample_task_noise_fit_records_route(method):
    _skip_if_no_engine()
    X, Y, noise = _make_data()
    gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=10,
    )

    result = gp.fit(
        X,
        Y,
        observation_noise=noise,
        method=method,
        max_iterations=1,
        learning_rate=0.01,
        verbose=False,
    )

    assert gp.is_trained
    assert result.noise_per_task.shape == (2,)
    np.testing.assert_allclose(result.noise_per_task, noise.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(gp._observation_noise_train, noise)
    assert gp.backend_train_info["noise_mode"] == "fixed_per_sample_task"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    assert gp.backend_train_info["precond_rank"] == 0
    assert gp.backend_train_info["use_preconditioner"] is False
    assert np.isfinite(result.final_nll)


def test_multi_output_fixed_per_sample_task_noise_predicts_latent_shape():
    _skip_if_no_engine()
    X, Y, noise = _make_data()
    gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=0,
    )
    gp.fit(
        X,
        Y,
        observation_noise=noise,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.01,
        verbose=False,
    )

    pred = gp.predict(X[:8])

    assert pred.mean.shape == (8, 2)
    assert np.all(np.isfinite(pred.mean))


def test_multi_output_fixed_noise_observed_prediction_adds_test_noise():
    _skip_if_no_engine()
    X, Y, noise = _make_data(seed=777)
    gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=0,
    )
    gp.fit(
        X,
        Y,
        observation_noise=noise,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.01,
        verbose=False,
    )

    latent = gp.predict_latent(X[:8], variance_method="exact")
    observed = gp.predict_observed(
        X[:8], observation_noise=noise[:8], variance_method="exact"
    )

    np.testing.assert_allclose(observed.mean, latent.mean)
    np.testing.assert_allclose(observed.variance, latent.variance + noise[:8], rtol=1e-5)

    with pytest.raises(ValueError, match="requires explicit observation_noise"):
        gp.predict_observed(X[:8], variance_method="mean_only")


def test_multi_output_grouped_noise_expands_to_fixed_per_sample_task_route():
    _skip_if_no_engine()
    X, Y, _ = _make_data(seed=321)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([[0.015, 0.03], [0.04, 0.065]], dtype=np.float32)
    expected_noise = group_noise[groups]
    gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=10,
    )

    result = gp.fit(
        X,
        Y,
        noise_model="grouped",
        noise_group_train=groups,
        group_noise=group_noise,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.01,
        verbose=False,
    )

    assert gp.is_trained
    np.testing.assert_allclose(gp._observation_noise_train, expected_noise)
    np.testing.assert_array_equal(gp._noise_group_train, groups)
    np.testing.assert_allclose(gp._noise_group_values, group_noise)
    np.testing.assert_allclose(result.noise_per_task, expected_noise.mean(axis=0), rtol=1e-4)
    assert gp.backend_train_info["noise_mode"] == "fixed_per_sample_task"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    assert gp.backend_train_info["precond_rank"] == 0

    latent = gp.predict_latent(X[:8], variance_method="exact")
    observed = gp.predict_observed(
        X[:8], noise_group_test=groups[:8], variance_method="exact"
    )
    np.testing.assert_allclose(observed.mean, latent.mean)
    np.testing.assert_allclose(
        observed.variance, latent.variance + expected_noise[:8], rtol=1e-5
    )


def test_multi_output_fixed_and_grouped_noise_save_load_roundtrip(tmp_path):
    _skip_if_no_engine()
    X, Y, noise = _make_data(seed=456)
    fixed_gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=0,
    )
    fixed_gp.fit(
        X,
        Y,
        observation_noise=noise,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.01,
        verbose=False,
    )
    fixed_path = tmp_path / "multi_fixed_noise"
    fixed_gp.save(str(fixed_path))
    fixed_loaded = MultiOutputGP.load(str(fixed_path))

    np.testing.assert_allclose(fixed_loaded._observation_noise_train, noise)
    assert fixed_loaded._noise_group_train is None
    assert fixed_loaded._noise_group_values is None

    groups = (X[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([[0.015, 0.03], [0.04, 0.065]], dtype=np.float32)
    grouped_gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=0,
    )
    grouped_gp.fit(
        X,
        Y,
        noise_model="grouped",
        noise_group_train=groups,
        group_noise=group_noise,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.01,
        verbose=False,
    )
    grouped_path = tmp_path / "multi_grouped_noise"
    grouped_gp.save(str(grouped_path))
    grouped_loaded = MultiOutputGP.load(str(grouped_path))

    np.testing.assert_allclose(grouped_loaded._observation_noise_train, group_noise[groups])
    np.testing.assert_array_equal(grouped_loaded._noise_group_train, groups)
    np.testing.assert_allclose(grouped_loaded._noise_group_values, group_noise)
