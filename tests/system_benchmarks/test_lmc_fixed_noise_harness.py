"""System workflow tests for fixed/grouped MultiOutputLMCGP noise."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from mojogp import MultiOutputLMCGP
from .conftest import assert_gpu_available, requires_cuda


def _lmc_known_noise_data(n: int = 1000, seed: int = 801):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 2)).astype(np.float32)
    latent_a = np.sin(X[:, 0]) + 0.2 * X[:, 1]
    latent_b = np.cos(0.8 * X[:, 0]) - 0.1 * X[:, 1]
    latent = np.stack(
        [latent_a + 0.35 * latent_b, 0.55 * latent_a - 0.25 * latent_b],
        axis=1,
    ).astype(np.float32)
    noise = np.empty_like(latent)
    noise[:, 0] = 0.014 + 0.032 * (X[:, 0] > 0.0)
    noise[:, 1] = 0.028 + 0.04 * (X[:, 1] > 0.0)
    Y = latent + rng.normal(scale=np.sqrt(noise)).astype(np.float32)
    return X, Y.astype(np.float32), latent, noise.astype(np.float32)


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _coverage(mean: np.ndarray, std: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((truth >= mean - 2.0 * std) & (truth <= mean + 2.0 * std)))


@pytest.mark.system
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_fixed_per_sample_task_noise_workflow(method: str):
    assert_gpu_available()
    X, Y, _, noise = _lmc_known_noise_data()
    X_test, _, latent_test, _ = _lmc_known_noise_data(n=256, seed=809)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gp = MultiOutputLMCGP(
        kernels=["rbf"],
        method=method,
        num_probes=4,
        max_cg_iterations=50,
        preconditioner_rank=10,
    )
    result = gp.fit(
        X,
        Y,
        observation_noise=noise,
        max_iterations=3,
        learning_rate=0.02,
        verbose=False,
    )
    torch.cuda.synchronize()
    train_time_s = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    pred = gp.predict_latent(X_test)
    observed = gp.predict_observed(
        X_test,
        observation_noise=_lmc_known_noise_data(n=256, seed=809)[3],
        variance_method="exact",
    )

    assert gp.backend_train_info["noise_mode"] == "fixed_per_sample_task"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    assert gp.backend_train_info["precond_rank"] == 0
    np.testing.assert_allclose(result.noise_per_task, noise.mean(axis=0), rtol=1e-4)
    assert _rmse(pred.mean, latent_test) < 1.5
    assert _coverage(observed.mean, observed.std, latent_test) >= 0.5
    assert np.all(np.isfinite(pred.mean))
    assert np.isfinite(result.final_nll)
    assert train_time_s > 0.0
    assert peak_mb >= 0.0


@pytest.mark.system
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_lmc_grouped_noise_workflow():
    assert_gpu_available()
    X, Y, _, _ = _lmc_known_noise_data(seed=819)
    X_test, _, latent_test, _ = _lmc_known_noise_data(n=256, seed=827)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([[0.014, 0.028], [0.046, 0.068]], dtype=np.float32)

    gp = MultiOutputLMCGP(
        kernels=["rbf"],
        method="matrix_free",
        num_probes=4,
        max_cg_iterations=50,
        preconditioner_rank=10,
    )
    result = gp.fit(
        X,
        Y,
        noise_model="grouped",
        noise_group_train=groups,
        group_noise=group_noise,
        max_iterations=3,
        learning_rate=0.02,
        verbose=False,
    )
    pred = gp.predict_latent(X_test)
    observed = gp.predict_observed(
        X_test,
        noise_group_test=(X_test[:, 0] > 0.0).astype(np.int32),
        variance_method="exact",
    )

    assert gp.backend_train_info["noise_mode"] == "fixed_per_sample_task"
    np.testing.assert_allclose(gp._observation_noise_train, group_noise[groups])
    np.testing.assert_allclose(
        result.noise_per_task, group_noise[groups].mean(axis=0), rtol=1e-4
    )
    assert _rmse(pred.mean, latent_test) < 1.5
    assert _coverage(observed.mean, observed.std, latent_test) >= 0.5
    assert np.all(np.isfinite(pred.mean))
    assert np.isfinite(result.final_nll)
