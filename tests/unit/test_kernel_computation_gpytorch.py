"""Explicit GPyTorch kernel-matrix comparisons moved out of unit correctness."""

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist


pytestmark = [pytest.mark.gpytorch, pytest.mark.reference]


class TestKernelMatrixVsGPyTorch:
    """Test kernel matrices against GPyTorch."""

    @pytest.fixture
    def gpytorch_kernels(self):
        """Import GPyTorch kernels."""
        try:
            import gpytorch

            return gpytorch.kernels
        except ImportError:
            pytest.skip("GPyTorch not installed")

    @pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("outputscale", [0.5, 1.0, 2.0])
    def test_rbf_kernel_matches_gpytorch(
        self, lengthscale, outputscale, gpytorch_kernels, random_seed
    ):
        """Test RBF kernel matrix matches GPyTorch."""
        n = 50
        d = 5
        X = np.random.randn(n, d).astype(np.float32)

        X_torch = torch.tensor(X)
        kernel = gpytorch_kernels.ScaleKernel(gpytorch_kernels.RBFKernel())
        kernel.base_kernel.lengthscale = lengthscale
        kernel.outputscale = outputscale
        K_gpytorch = kernel(X_torch).evaluate().detach().numpy()

        dist_sq = cdist(X, X, metric="sqeuclidean")
        K_manual = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        np.testing.assert_allclose(K_manual, K_gpytorch, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
    def test_matern_kernel_matches_gpytorch(self, nu, gpytorch_kernels, random_seed):
        """Test Matern kernel matrix matches GPyTorch."""
        n = 50
        d = 5
        X = np.random.randn(n, d).astype(np.float32)
        lengthscale = 1.0
        outputscale = 1.0

        X_torch = torch.tensor(X)
        kernel = gpytorch_kernels.ScaleKernel(gpytorch_kernels.MaternKernel(nu=nu))
        kernel.base_kernel.lengthscale = lengthscale
        kernel.outputscale = outputscale
        K_gpytorch = kernel(X_torch).evaluate().detach().numpy()

        dist = cdist(X, X, metric="euclidean")
        if nu == 0.5:
            K_manual = outputscale * np.exp(-dist / lengthscale)
        elif nu == 1.5:
            sqrt3 = np.sqrt(3.0)
            sqrt3_r_l = sqrt3 * dist / lengthscale
            K_manual = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        else:
            sqrt5 = np.sqrt(5.0)
            sqrt5_r_l = sqrt5 * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist**2 / (lengthscale**2)
            K_manual = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(K_manual, K_gpytorch, rtol=1e-4, atol=1e-5)
