from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch required for GPyTorch ARD helper tests")
pytest.importorskip("gpytorch", reason="GPyTorch required for ARD helper shape tests")

from tests.shared.benchmarking.gpytorch_models import (  # noqa: E402
    GPyTorchMultiOutputGP,
    GPyTorchSingleOutputGP,
    train_gpytorch_multi_output,
)


def test_gpytorch_single_output_ard_lengthscale_shape():
    d = 9
    train_x = torch.randn(12, d)
    train_y = torch.randn(12)
    likelihood = __import__("gpytorch").likelihoods.GaussianLikelihood()

    model = GPyTorchSingleOutputGP(train_x, train_y, likelihood, kernel_type="rbf", ard=True)

    assert model.covar_module.base_kernel.lengthscale.shape[-1] == d


def test_gpytorch_multi_output_ard_lengthscale_shape():
    d = 17
    num_tasks = 3
    train_x = torch.randn(12, d)
    train_y = torch.randn(12, num_tasks)
    likelihood = __import__("gpytorch").likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks
    )

    model = GPyTorchMultiOutputGP(
        train_x,
        train_y,
        likelihood,
        kernel_type="rbf",
        num_tasks=num_tasks,
        ard=True,
    )

    assert model.covar_module.data_covar_module.lengthscale.shape[-1] == d


def test_train_gpytorch_multi_output_ard_returns_lengthscale_vector():
    rng = np.random.default_rng(42)
    d = 9
    num_tasks = 3
    X = rng.normal(size=(24, d)).astype(np.float32)
    Y = rng.normal(size=(24, num_tasks)).astype(np.float32)

    result = train_gpytorch_multi_output(
        X,
        Y,
        kernel_type="rbf",
        num_tasks=num_tasks,
        mode="cg",
        n_iterations=1,
        lr=0.01,
        ard=True,
        monitor_memory=False,
        device="cpu",
    )

    assert result["ard"] is True
    assert len(result["learned_params"]["lengthscales"]) == d


def test_train_gpytorch_multi_output_initializes_task_noise_only():
    rng = np.random.default_rng(123)
    num_tasks = 3
    X = rng.normal(size=(16, 4)).astype(np.float32)
    Y = rng.normal(size=(16, num_tasks)).astype(np.float32)

    result = train_gpytorch_multi_output(
        X,
        Y,
        kernel_type="rbf",
        num_tasks=num_tasks,
        mode="cg",
        n_iterations=0,
        init_noise=0.123,
        monitor_memory=False,
        device="cpu",
    )

    assert "task_noises" not in result["learned_params"]
    assert np.allclose(result["learned_params"]["noise_per_task"], [0.123] * num_tasks)
