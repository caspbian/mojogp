"""System benchmark harness for single-output fixed heteroskedastic noise."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from mojogp import RBF, SingleOutputGP as ExactGP
from .conftest import assert_gpu_available, requires_cuda


def _known_noise_data(n: int = 2000, seed: int = 31):
    rng = np.random.default_rng(seed)
    X = np.linspace(-3.0, 3.0, n, dtype=np.float32).reshape(-1, 1)
    true_noise = (0.01 + 0.05 * (X[:, 0] > 0.0)).astype(np.float32)
    latent = np.sin(2.0 * X[:, 0]).astype(np.float32)
    y = (latent + rng.normal(0.0, np.sqrt(true_noise))).astype(np.float32)
    return X, y, latent, true_noise


def _known_noise_fn(X_eval: np.ndarray) -> np.ndarray:
    return (0.01 + 0.05 * (X_eval[:, 0] > 0.0)).astype(np.float32)


def _coverage(mean: np.ndarray, std: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((truth >= mean - 2.0 * std) & (truth <= mean + 2.0 * std)))


@pytest.mark.system
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_single_output_fixed_noise_known_variance_workflow(method: str):
    assert_gpu_available()
    X, y, latent, noise = _known_noise_data()
    X_test, _, latent_test, test_noise = _known_noise_data(n=256, seed=37)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        observation_noise=noise,
        learn_noise=False,
        method=method,
        max_iterations=5,
        learning_rate=0.03,
        num_probes=4,
        max_cg_iterations=80,
        max_tridiag_iterations=8,
        preconditioner_rank=10,
        verbose=False,
    )
    torch.cuda.synchronize()
    train_time_s = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    latent_pred = gp.predict_latent(
        X_test,
        variance_method="exact",
        method=method,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        preconditioner_rank=10,
    )
    observed_pred = gp.predict_observed(
        X_test,
        observation_noise=test_noise,
        variance_method="exact",
        method=method,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        preconditioner_rank=10,
    )

    rmse = float(np.sqrt(np.mean((latent_pred.mean - latent_test) ** 2)))
    observed_coverage = _coverage(observed_pred.mean, observed_pred.std, latent_test)
    assert gp.backend_train_info["noise_mode"] == "fixed_vector"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    assert np.all(observed_pred.variance >= latent_pred.variance)
    assert rmse < 1.0
    assert observed_coverage >= 0.5
    assert train_time_s > 0.0
    assert peak_mb >= 0.0


@pytest.mark.system
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
def test_single_output_grouped_noise_known_group_workflow():
    assert_gpu_available()
    X, y, _, _ = _known_noise_data(seed=41)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([0.01, 0.06], dtype=np.float32)
    X_test, _, latent_test, _ = _known_noise_data(n=256, seed=43)
    test_groups = (X_test[:, 0] > 0.0).astype(np.int32)

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="grouped",
        noise_group_train=groups,
        group_noise=group_noise,
        learn_noise=False,
        method="matrix_free",
        max_iterations=5,
        learning_rate=0.03,
        num_probes=4,
        max_cg_iterations=80,
        max_tridiag_iterations=8,
        preconditioner_rank=10,
        verbose=False,
    )
    observed = gp.predict_observed(
        X_test,
        noise_group_test=test_groups,
        variance_method="exact",
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        preconditioner_rank=10,
    )

    assert gp.backend_train_info["noise_mode"] == "fixed_grouped"
    assert _coverage(observed.mean, observed.std, latent_test) >= 0.5
    assert np.all(np.isfinite(observed.mean))
    assert np.all(np.isfinite(observed.variance))


@pytest.mark.system
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
def test_single_output_input_dependent_noise_function_workflow():
    assert_gpu_available()
    X, y, _, _ = _known_noise_data(seed=47)
    X_test, _, latent_test, _ = _known_noise_data(n=256, seed=53)

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="input_dependent",
        observation_noise_fn=_known_noise_fn,
        learn_noise=False,
        method="matrix_free",
        max_iterations=5,
        learning_rate=0.03,
        num_probes=4,
        max_cg_iterations=80,
        max_tridiag_iterations=8,
        preconditioner_rank=10,
        verbose=False,
    )
    latent = gp.predict_latent(
        X_test,
        variance_method="exact",
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        preconditioner_rank=10,
    )
    observed = gp.predict_observed(
        X_test,
        variance_method="exact",
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        preconditioner_rank=10,
    )

    assert gp.backend_train_info["noise_mode"] == "fixed_input_dependent"
    np.testing.assert_allclose(gp._observation_noise_train, _known_noise_fn(X))
    np.testing.assert_allclose(
        observed.variance,
        latent.variance + _known_noise_fn(X_test),
        rtol=1e-5,
        atol=1e-6,
    )
    assert _coverage(observed.mean, observed.std, latent_test) >= 0.5
    assert np.all(np.isfinite(observed.mean))
