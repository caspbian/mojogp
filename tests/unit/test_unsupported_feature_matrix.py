"""Centralized assertions for explicitly unsupported wrapper surfaces."""

import importlib.util

import numpy as np
import pytest

from mojogp import SingleOutputGP, MultiOutputGP, MultiOutputLMCGP
from mojogp.kernel import Kernel


def _skip_if_no_engine():
    if importlib.util.find_spec("mojogp_jit_engine") is None:
        pytest.skip("mojogp_jit_engine not built; run `task build` to enable")


def _make_exact_data(n: int = 2000, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.05 * rng.randn(n)).astype(np.float32)
    return X, y


def _make_mixed_data(n: int = 2000, seed: int = 0):
    rng = np.random.RandomState(seed)
    x_cont = rng.randn(n, 2).astype(np.float32)
    cat = rng.randint(0, 3, size=(n, 1)).astype(np.float32)
    X = np.concatenate([x_cont, cat], axis=1)
    y = (
        np.sin(x_cont[:, 0])
        + 0.25 * x_cont[:, 1]
        + 0.5 * (cat[:, 0] == 1)
        + 0.03 * rng.randn(n)
    ).astype(np.float32)
    return X, y


def _make_multi_output_data(n: int = 500, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3).astype(np.float32)
    Y = np.stack(
        [
            np.sin(X[:, 0]) + 0.2 * X[:, 1],
            np.cos(X[:, 0]) - 0.1 * X[:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    Y += 0.03 * rng.randn(n, 2).astype(np.float32)
    return X, Y


def test_exactgp_rejects_pure_categorical_kernels():
    X = np.random.randint(0, 3, size=(2000, 1)).astype(np.float32)
    y = np.random.randn(2000).astype(np.float32)

    gp = SingleOutputGP(Kernel.ehh(levels=3, active_dims=[0]))
    with pytest.raises(ValueError, match="Pure categorical kernels"):
        gp.fit(X, y, max_iterations=1)


def test_single_output_mixed_pathwise_sampling_returns_finite_samples():
    _skip_if_no_engine()
    X, y = _make_mixed_data()
    gp = SingleOutputGP(Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]))
    gp.fit(X, y, max_iterations=2, learning_rate=0.03, method="materialized")

    with pytest.raises(NotImplementedError, match="Public Cholesky Sampling"):
        gp.sample_posterior(X[:16], n_samples=2, method="cholesky")
    samples = gp.sample_posterior(X[:16], n_samples=2, method="pathwise")
    assert samples.shape == (2, 16)
    assert np.all(np.isfinite(samples))


def test_single_output_composite_pathwise_sampling_returns_finite_samples():
    _skip_if_no_engine()
    X, y = _make_exact_data()
    gp = SingleOutputGP(Kernel.rbf() + Kernel.matern52())
    gp.fit(X, y, max_iterations=2, learning_rate=0.03, method="matrix_free")

    samples = gp.sample_posterior(X[:16], n_samples=2, method="pathwise")
    assert samples.shape == (2, 16)
    assert np.all(np.isfinite(samples))


def test_exactgp_rejects_invalid_sampling_method():
    gp = SingleOutputGP(Kernel.rbf())
    gp._is_trained = True

    with pytest.raises(ValueError, match="method must be"):
        gp.sample_posterior(np.random.randn(8, 2).astype(np.float32), method="bogus")


@pytest.mark.parametrize(
    "gp",
    [
        SingleOutputGP(Kernel.rbf()),
        MultiOutputGP(kernel=Kernel.rbf()),
        MultiOutputLMCGP(kernels=["rbf"]),
    ],
)
def test_public_sampling_rejects_legacy_matheron_alias(gp):
    gp._is_trained = True

    with pytest.raises(ValueError, match="diagonal.*pathwise"):
        gp.sample_posterior(
            np.random.randn(8, 2).astype(np.float32), method="matheron"
        )


@pytest.mark.parametrize(
    "gp",
    [
        SingleOutputGP(Kernel.rbf()),
        MultiOutputGP(kernel=Kernel.rbf()),
        MultiOutputLMCGP(kernels=["rbf"]),
    ],
)
def test_public_prediction_rejects_legacy_none_variance_alias(gp):
    gp._is_trained = True

    with pytest.raises(ValueError, match="variance_method must be"):
        gp.predict(np.random.randn(8, 2).astype(np.float32), variance_method="none")


def test_exactgp_rejects_latent_noise_gp_model():
    X, y = _make_exact_data()
    gp = SingleOutputGP(Kernel.rbf())

    with pytest.raises(ValueError, match="noise_model must be one of"):
        gp.fit(
            X,
            y,
            noise_model="latent_gp",
            max_iterations=1,
            verbose=False,
        )


def test_multi_output_gp_rejects_pure_categorical_kernels():
    X = np.random.randint(0, 3, size=(500, 1)).astype(np.float32)
    Y = np.random.randn(500, 2).astype(np.float32)

    gp = MultiOutputGP(kernel=Kernel.ehh(levels=3, active_dims=[0]), task_rank=1)
    with pytest.raises(ValueError, match="Pure categorical"):
        gp.fit(X, Y, max_iterations=1, verbose=False)


def test_multi_output_gp_rejects_unsupported_heteroskedastic_noise_models():
    X, Y = _make_multi_output_data()
    gp = MultiOutputGP(kernel=Kernel.rbf(), task_rank=1)

    with pytest.raises(
        NotImplementedError,
        match="input-dependent heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            noise_model="input_dependent",
            observation_noise_fn=lambda X_eval: np.full(
                X_eval.shape[0], 0.05, dtype=np.float32
            ),
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(
        NotImplementedError,
        match="learned per-sample-task heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            noise_model="learned_vector",
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(
        NotImplementedError,
        match="learned per-sample-task heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            noise_model="latent_gp",
            max_iterations=1,
            verbose=False,
        )


def test_multi_output_gp_validates_fixed_per_sample_task_noise():
    X, Y = _make_multi_output_data()
    gp = MultiOutputGP(kernel=Kernel.rbf(), task_rank=1)

    with pytest.raises(ValueError, match="requires observation_noise"):
        gp.fit(
            X,
            Y,
            noise_model="fixed_vector",
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(ValueError, match="must have shape"):
        gp.fit(
            X,
            Y,
            observation_noise=np.full((X.shape[0],), 0.05, dtype=np.float32),
            max_iterations=1,
            verbose=False,
        )

    bad_noise = np.full(Y.shape, 0.05, dtype=np.float32)
    bad_noise[0, 0] = 0.0
    with pytest.raises(ValueError, match="values must be > 0"):
        gp.fit(
            X,
            Y,
            observation_noise=bad_noise,
            max_iterations=1,
            verbose=False,
        )


def test_multi_output_gp_rejects_fixed_noise_for_mixed_kernels():
    X, y_scalar = _make_mixed_data(seed=17)
    Y = np.stack([y_scalar, 0.5 * y_scalar], axis=1).astype(np.float32)
    gp = MultiOutputGP(
        kernel=Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
        task_rank=1,
    )

    with pytest.raises(NotImplementedError, match="continuous kernels only"):
        gp.fit(
            X,
            Y,
            observation_noise=np.full(Y.shape, 0.05, dtype=np.float32),
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(NotImplementedError, match="continuous kernels only"):
        gp.fit(
            X,
            Y,
            noise_model="grouped",
            noise_group_train=np.zeros(X.shape[0], dtype=np.int32),
            group_noise=np.full((1, Y.shape[1]), 0.05, dtype=np.float32),
            max_iterations=1,
            verbose=False,
        )


def test_multi_output_gp_validates_grouped_noise():
    X, Y = _make_multi_output_data()
    gp = MultiOutputGP(kernel=Kernel.rbf(), task_rank=1)

    with pytest.raises(ValueError, match="requires noise_group_train"):
        gp.fit(
            X,
            Y,
            noise_model="grouped",
            group_noise=np.full((1, Y.shape[1]), 0.05, dtype=np.float32),
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(ValueError, match="requires group_noise"):
        gp.fit(
            X,
            Y,
            noise_model="grouped",
            noise_group_train=np.zeros(X.shape[0], dtype=np.int32),
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(ValueError, match="must have shape"):
        gp.fit(
            X,
            Y,
            noise_model="grouped",
            noise_group_train=np.zeros(X.shape[0], dtype=np.int32),
            group_noise=np.full((1,), 0.05, dtype=np.float32),
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(ValueError, match="outside group_noise"):
        gp.fit(
            X,
            Y,
            noise_model="grouped",
            noise_group_train=np.full(X.shape[0], 2, dtype=np.int32),
            group_noise=np.full((2, Y.shape[1]), 0.05, dtype=np.float32),
            max_iterations=1,
            verbose=False,
        )


def test_multi_output_gp_mixed_pathwise_sampling_runs():
    _skip_if_no_engine()
    X, y_scalar = _make_mixed_data(seed=13)
    Y = np.stack([y_scalar, 0.6 * y_scalar + 0.1 * X[:, 0]], axis=1).astype(np.float32)
    gp = MultiOutputGP(
        kernel=Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
        task_rank=1,
        num_probes=2,
    )
    gp.fit(X, Y, max_iterations=2, learning_rate=0.03, verbose=False, method="matrix_free")

    samples = gp.sample_posterior(X[:12], n_samples=2, method="pathwise")
    assert samples.shape == (2, 12, 2)
    assert np.all(np.isfinite(samples))


def test_lmc_rejects_invalid_sampling_method():
    gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
    gp._is_trained = True

    with pytest.raises(
        ValueError,
        match="method must be 'diagonal' or 'pathwise'",
    ):
        gp.sample_posterior(np.random.randn(8, 3).astype(np.float32), method="bogus")


def test_lmc_fixed_observation_noise_validates_shape_before_backend():
    X, Y = _make_multi_output_data(n=8)
    gp = MultiOutputLMCGP(kernels=["rbf"])

    with pytest.raises(ValueError, match="fixed_observation_noise must have shape"):
        gp.fit(
            X,
            Y,
            fixed_observation_noise=np.ones((8,), dtype=np.float32),
            max_iterations=1,
        )


def test_lmc_mixed_fixed_observation_noise_is_explicitly_not_supported():
    X, y_scalar = _make_mixed_data(n=500, seed=21)
    Y = np.stack([y_scalar, 0.6 * y_scalar + 0.1 * X[:, 0]], axis=1).astype(np.float32)
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])
    gp = MultiOutputLMCGP(kernels=[kernel])

    with pytest.raises(NotImplementedError, match="Fixed per-sample-per-task noise"):
        gp.fit(
            X,
            Y,
            fixed_observation_noise=np.full(Y.shape, 0.01, dtype=np.float32),
            max_iterations=1,
        )


def test_lmc_fixed_observation_noise_reaches_backend_handoff(monkeypatch):
    X, Y = _make_multi_output_data(n=8)
    gp = MultiOutputLMCGP(kernels=["rbf"])
    fixed_noise = np.full(Y.shape, 0.01, dtype=np.float32)
    seen = {}

    def fake_fit_composite(*args, **kwargs):
        seen["fixed"] = kwargs["fixed_observation_noise"]
        raise RuntimeError("backend handoff reached")

    monkeypatch.setattr(gp, "_fit_composite", fake_fit_composite)
    with pytest.raises(RuntimeError, match="backend handoff reached"):
        gp.fit(X, Y, fixed_observation_noise=fixed_noise, max_iterations=1)

    np.testing.assert_allclose(seen["fixed"], fixed_noise)


def test_lmc_rejects_heteroskedastic_noise_models():
    X, Y = _make_multi_output_data()
    gp = MultiOutputLMCGP(kernels=["rbf"])

    with pytest.raises(
        NotImplementedError,
        match="per-sample-task and grouped heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            observation_noise=np.full(Y.shape, 0.05, dtype=np.float32),
            method="matrix_free",
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(
        NotImplementedError,
        match="per-sample-task and grouped heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            noise_model="input_dependent",
            observation_noise_fn=lambda X_eval: np.full(
                X_eval.shape[0], 0.05, dtype=np.float32
            ),
            method="matrix_free",
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(
        NotImplementedError,
        match="learned per-sample-task heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            noise_model="learned_vector",
            method="matrix_free",
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(
        NotImplementedError,
        match="learned per-sample-task heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            noise_model="latent_gp",
            method="matrix_free",
            max_iterations=1,
            verbose=False,
        )

    with pytest.raises(
        NotImplementedError,
        match="per-sample-task and grouped heteroskedastic noise",
    ):
        gp.fit(
            X,
            Y,
            noise_model="grouped",
            noise_group_train=np.zeros(X.shape[0], dtype=np.int32),
            group_noise=np.full((1, Y.shape[1]), 0.05, dtype=np.float32),
            max_iterations=1,
            verbose=False,
        )
