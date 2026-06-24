"""Wrapper-level guard tests for feature-support placeholders."""

import numpy as np
import pytest

from mojogp import MultiOutputGP, MultiOutputLMCGP, SingleOutputGP
from mojogp.kernel import Kernel


def test_single_output_mixed_fixed_observation_noise_rejects_before_backend(monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.randn(8, 2).astype(np.float32)
    y = rng.randn(8).astype(np.float32)
    X[:, 1] = rng.randint(0, 3, size=8).astype(np.float32)
    gp = SingleOutputGP(Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=3, active_dims=[1]))
    monkeypatch.setattr(gp, "_ensure_compiled", lambda: None)

    with pytest.raises(NotImplementedError, match="continuous ExactGP only"):
        gp.fit(
            X,
            y,
            observation_noise=np.full(8, 0.01, dtype=np.float32),
            learn_noise=False,
            max_iterations=1,
        )


def test_icm_mixed_fixed_observation_noise_rejects_before_backend():
    rng = np.random.RandomState(1)
    X = rng.randn(8, 2).astype(np.float32)
    X[:, 1] = rng.randint(0, 3, size=8).astype(np.float32)
    Y = rng.randn(8, 2).astype(np.float32)
    gp = MultiOutputGP(
        kernel=Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=3, active_dims=[1])
    )

    with pytest.raises(NotImplementedError, match="continuous kernels only"):
        gp.fit(
            X,
            Y,
            observation_noise=np.full(Y.shape, 0.01, dtype=np.float32),
            max_iterations=1,
        )


def test_materialized_grads_is_rejected_as_non_public_api():
    rng = np.random.RandomState(2)
    X = rng.randn(8, 2).astype(np.float32)
    y = rng.randn(8).astype(np.float32)
    gp = SingleOutputGP(Kernel.rbf())

    with pytest.raises(NotImplementedError, match="materialized_grads"):
        gp.fit(X, y, method="materialized_grads", max_iterations=1)


def test_input_dependent_noise_rejects_unknown_learned_function():
    rng = np.random.RandomState(3)
    X = rng.randn(8, 2).astype(np.float32)
    y = rng.randn(8).astype(np.float32)
    gp = SingleOutputGP(Kernel.rbf())

    with pytest.raises(ValueError, match="noise_function must be 'linear'"):
        gp.fit(
            X,
            y,
            noise_model="learned_input_dependent",
            noise_function="neural",
            max_iterations=1,
        )


def test_lmc_grouped_noise_placeholder_raises_before_backend():
    rng = np.random.RandomState(4)
    X = rng.randn(8, 2).astype(np.float32)
    Y = rng.randn(8, 2).astype(np.float32)
    gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])

    with pytest.raises(NotImplementedError, match="Grouped noise"):
        gp.fit(X, Y, grouped_noise=np.zeros((2, 2), dtype=np.float32), max_iterations=1)
