"""System benchmark harness for in-development single-output learned noise."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mojogp import RBF, SingleOutputGP as ExactGP
from .conftest import assert_gpu_available, requires_cuda


def _heteroskedastic_group_data(n: int = 2000, seed: int = 55):
    rng = np.random.default_rng(seed)
    X = np.linspace(-3.0, 3.0, n, dtype=np.float32).reshape(-1, 1)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([0.005, 0.12], dtype=np.float32)
    latent = np.sin(2.0 * X[:, 0]).astype(np.float32)
    y = (latent + rng.normal(0.0, np.sqrt(group_noise[groups]))).astype(np.float32)
    return X, y, latent, groups, group_noise


def _linear_noise_data(n: int = 2000, seed: int = 63):
    rng = np.random.default_rng(seed)
    X = np.linspace(-2.0, 2.0, n, dtype=np.float32).reshape(-1, 1)
    true_noise = (0.012 + 0.035 / (1.0 + np.exp(-2.5 * X[:, 0]))).astype(np.float32)
    latent = (np.sin(2.0 * X[:, 0]) + 0.2 * X[:, 0]).astype(np.float32)
    y = (latent + rng.normal(0.0, np.sqrt(true_noise))).astype(np.float32)
    return X, y, latent, true_noise


@pytest.mark.system
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
def test_learned_vector_noise_homoscedastic_smoke():
    assert_gpu_available()
    rng = np.random.default_rng(57)
    X = np.linspace(-3.0, 3.0, 2000, dtype=np.float32).reshape(-1, 1)
    latent = np.sin(2.0 * X[:, 0]).astype(np.float32)
    y = (latent + rng.normal(0.0, np.sqrt(0.03), size=X.shape[0])).astype(np.float32)

    torch.cuda.reset_peak_memory_stats()
    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="learned_vector",
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.05,
        method="matrix_free",
        max_iterations=4,
        learning_rate=0.01,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=4,
        max_cg_iterations=40,
        cg_tolerance=1e-3,
        max_tridiag_iterations=8,
        preconditioner_rank=8,
        verbose=False,
    )
    torch.cuda.synchronize()
    learned_noise = gp.get_learned_params()["observation_noise_train"]
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    assert gp.backend_train_info["noise_mode"] == "learned_vector"
    assert learned_noise.shape == (X.shape[0],)
    assert np.all(np.isfinite(learned_noise))
    assert float(np.min(learned_noise)) >= 1e-5
    assert float(np.max(learned_noise)) < 0.2
    assert abs(float(np.mean(learned_noise)) - 0.03) < 0.01
    assert float(np.percentile(learned_noise, 95) / np.percentile(learned_noise, 5)) < 1.2
    assert float(np.mean(learned_noise <= 1.1e-5)) < 0.01
    assert peak_mb >= 0.0


@pytest.mark.system
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
def test_learned_grouped_noise_orders_known_groups():
    assert_gpu_available()
    X, y, _, groups, true_group_noise = _heteroskedastic_group_data()
    X_test, _, latent_test, test_groups, _ = _heteroskedastic_group_data(n=256, seed=59)

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="grouped",
        noise_group_train=groups,
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.01,
        method="matrix_free",
        max_iterations=80,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=4,
        max_cg_iterations=80,
        cg_tolerance=1e-3,
        max_tridiag_iterations=8,
        preconditioner_rank=8,
        verbose=False,
    )
    learned_group_noise = gp.get_learned_params()["group_noise"]
    observed = gp.predict_observed(
        X_test,
        noise_group_test=test_groups,
        variance_method="mean_only",
    )

    assert gp.backend_train_info["noise_mode"] == "learned_grouped"
    assert learned_group_noise.shape == true_group_noise.shape
    assert np.all(np.isfinite(learned_group_noise))
    assert learned_group_noise[1] > learned_group_noise[0]
    assert learned_group_noise[0] < 0.02
    assert 0.07 < learned_group_noise[1] < 0.16
    assert learned_group_noise[1] / learned_group_noise[0] > 6.0
    assert np.all(np.isfinite(observed.mean))
    assert np.all(np.isfinite(observed.variance))
    assert float(np.sqrt(np.mean((observed.mean - latent_test) ** 2))) < 1.0


@pytest.mark.system
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
def test_learned_vector_regularization_prevents_pathological_noise_values():
    assert_gpu_available()
    rng = np.random.default_rng(61)
    X = np.linspace(-3.0, 3.0, 2000, dtype=np.float32).reshape(-1, 1)
    latent = np.sin(2.0 * X[:, 0]).astype(np.float32)
    y = (latent + rng.normal(0.0, np.sqrt(0.03), size=X.shape[0])).astype(np.float32)

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="learned_vector",
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.1,
        method="matrix_free",
        max_iterations=30,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=4,
        max_cg_iterations=60,
        cg_tolerance=1e-3,
        max_tridiag_iterations=8,
        preconditioner_rank=8,
        verbose=False,
    )
    regularized_noise = gp.get_learned_params()["observation_noise_train"]

    assert np.all(np.isfinite(regularized_noise))
    assert abs(float(np.mean(regularized_noise)) - 0.03) < 0.01
    assert float(np.mean(regularized_noise <= 1.1e-5)) < 0.01
    assert float(np.min(regularized_noise)) > 0.005
    assert float(np.max(regularized_noise)) < 0.1


@pytest.mark.system
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
def test_learned_linear_input_dependent_noise_recovers_monotone_variance():
    assert_gpu_available()
    X, y, _, true_noise = _linear_noise_data()
    X_test = np.array([[-1.5], [0.0], [1.5]], dtype=np.float32)

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="learned_input_dependent",
        noise_function="linear",
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.01,
        method="matrix_free",
        max_iterations=30,
        learning_rate=0.01,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=8,
        max_cg_iterations=80,
        cg_tolerance=1e-3,
        max_tridiag_iterations=8,
        preconditioner_rank=8,
        verbose=False,
    )
    learned_params = gp.get_learned_params()
    learned_noise = learned_params["observation_noise_train"]
    raw_params = learned_params["noise_function_params"]
    latent = gp.predict_latent(X_test, variance_method="mean_only")
    observed = gp.predict_observed(X_test, variance_method="mean_only")
    inferred_test_noise = observed.variance - latent.variance

    assert gp.backend_train_info["noise_mode"] == "learned_input_dependent"
    assert gp.backend_train_info["learned_noise_function"] == "linear"
    assert learned_params["noise_function"] == "linear"
    assert raw_params.shape == (2,)
    assert raw_params[1] > 0.15
    assert np.all(np.isfinite(learned_noise))
    assert float(np.min(learned_noise)) >= 1e-5
    assert learned_noise[-1] > learned_noise[0] * 2.0
    assert float(np.corrcoef(X[:, 0], learned_noise)[0, 1]) > 0.9
    assert abs(float(np.mean(learned_noise)) - float(np.mean(true_noise))) < 0.02
    assert np.all(np.isfinite(observed.mean))
    assert np.all(np.isfinite(observed.variance))
    assert inferred_test_noise[2] > inferred_test_noise[0]
    np.testing.assert_allclose(observed.mean, latent.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(observed.variance, latent.variance + inferred_test_noise, rtol=1e-5, atol=1e-6)
