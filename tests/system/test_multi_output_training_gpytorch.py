"""Explicit GPyTorch multi-output comparisons moved out of unit correctness."""

import numpy as np
import pytest
import torch


pytestmark = [pytest.mark.gpytorch, pytest.mark.reference]


def generate_multi_output_data(n=100, d=3, T=2, seed=42):
    """Generate synthetic multi-output data with shared latent structure."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = np.random.randn(n, d).astype(np.float32)
    f_latent = np.sin(X[:, 0]) + 0.5 * X[:, 1]

    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        scale = 1.0 + 0.5 * t
        noise = 0.1 * (1 + 0.2 * t)
        Y[:, t] = scale * f_latent + noise * np.random.randn(n)

    return X, Y


class TestMultiOutputTrainingGPyTorch:
    """Test multi-output baselines against GPyTorch."""

    def test_gpytorch_multitask_baseline(self):
        """Test GPyTorch MultitaskGP as a benchmark baseline."""
        try:
            import gpytorch
        except ImportError:
            pytest.skip("GPyTorch not installed")

        X, Y = generate_multi_output_data(n=50, d=3, T=2)
        train_x = torch.tensor(X)
        train_y = torch.tensor(Y)

        class MultitaskGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, num_tasks):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ConstantMean(), num_tasks=num_tasks
                )
                self.covar_module = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=num_tasks
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(
                    mean_x, covar_x
                )

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=2)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        nll_history = []
        for _ in range(50):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            nll_history.append(loss.item())

        assert nll_history[-1] < nll_history[0], "GPyTorch NLL should decrease"

    def test_nll_matches_gpytorch_multitask_kernel(self):
        """Compare fixed-parameter NLL with GPyTorch's MultitaskKernel."""
        try:
            import gpytorch
        except ImportError:
            pytest.skip("GPyTorch not installed")

        n = 40
        T = 2
        d = 3
        np.random.seed(42)
        torch.manual_seed(42)

        X = np.random.randn(n, d).astype(np.float32)
        Y = np.random.randn(n, T).astype(np.float32)
        lengthscale = 1.0
        outputscale = 1.0
        noise = 0.1

        train_x = torch.tensor(X, dtype=torch.float32)
        train_y = torch.tensor(Y, dtype=torch.float32)

        class MultitaskGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, num_tasks):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ZeroMean(), num_tasks=num_tasks
                )
                self.covar_module = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=num_tasks
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(
                    mean_x, covar_x
                )

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=T)
        model = MultitaskGPModel(train_x, train_y, likelihood, T)
        model.covar_module.data_covar_module.lengthscale = lengthscale
        model.covar_module.data_covar_module.outputscale = outputscale
        likelihood.noise = noise

        model.train()
        likelihood.train()

        with torch.no_grad():
            task_covar = model.covar_module.task_covar_module
            covar_factor = task_covar.covar_factor.detach().numpy().astype(np.float64)
            var = task_covar.var.detach().numpy().astype(np.float64)
            B_gpytorch = covar_factor @ covar_factor.T + np.diag(var)

            data_covar = model.covar_module.data_covar_module(train_x)
            K_X_gpytorch = data_covar.to_dense().detach().numpy().astype(np.float64)

            K_full = np.kron(B_gpytorch, K_X_gpytorch) + noise * np.eye(n * T)
            y_flat = Y.T.flatten().astype(np.float64)
            L = np.linalg.cholesky(K_full)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_flat))
            inv_quad_gpytorch = float(y_flat @ alpha)
            log_det_gpytorch = float(2 * np.sum(np.log(np.diag(L))))
            nll_gpytorch = 0.5 * (
                inv_quad_gpytorch + log_det_gpytorch + n * T * np.log(2 * np.pi)
            )

        eigenvalues, Q = np.linalg.eigh(B_gpytorch)
        Y_rotated = Y.astype(np.float64) @ Q

        nll_kronecker = 0.0
        for t in range(T):
            lambda_t = eigenvalues[t]
            K_t = lambda_t * K_X_gpytorch + noise * np.eye(n)
            y_t = Y_rotated[:, t]

            L_t = np.linalg.cholesky(K_t)
            alpha_t = np.linalg.solve(L_t.T, np.linalg.solve(L_t, y_t))
            inv_quad_t = y_t @ alpha_t
            log_det_t = 2 * np.sum(np.log(np.diag(L_t)))
            nll_t = 0.5 * (inv_quad_t + log_det_t + n * np.log(2 * np.pi))
            nll_kronecker += nll_t

        rel_error = abs(nll_gpytorch - nll_kronecker) / abs(nll_gpytorch)
        assert rel_error < 1e-6, (
            f"NLL mismatch: GPyTorch={nll_gpytorch}, Kronecker={nll_kronecker}"
        )
