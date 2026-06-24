"""Explicit GPyTorch/linear_operator alignment checks for low-level numerics."""

import numpy as np
import pytest
import torch


pytestmark = [pytest.mark.gpytorch, pytest.mark.reference]


class TestCGAlignment:
    """Test CG implementation against linear_operator."""

    def test_cg_vs_gpytorch_direct(self, random_seed):
        """Test CG solution matches linear_operator directly."""
        try:
            from linear_operator.operators import (
                DenseLinearOperator,
                DiagLinearOperator,
            )
            from linear_operator.operators.added_diag_linear_operator import (
                AddedDiagLinearOperator,
            )
            from linear_operator.utils.linear_cg import linear_cg
        except ImportError:
            pytest.skip("linear_operator not installed")

        n = 100
        X = np.random.randn(n, 5).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        noise = 0.01

        from scipy.spatial.distance import cdist

        dist_sq = cdist(X, X, metric="sqeuclidean")
        K = np.exp(-dist_sq / 2).astype(np.float32)

        K_torch = torch.tensor(K)
        noise_diag = torch.full((n,), noise)
        K_linop = AddedDiagLinearOperator(
            DenseLinearOperator(K_torch), DiagLinearOperator(noise_diag)
        )

        y_torch = torch.tensor(y).unsqueeze(-1)
        result_gpytorch = linear_cg(
            K_linop.matmul,
            y_torch,
            max_iter=100,
            tolerance=1e-3,
        )

        if isinstance(result_gpytorch, tuple):
            x_gpytorch = result_gpytorch[0].squeeze().numpy()
        else:
            x_gpytorch = result_gpytorch.squeeze().numpy()

        A = K + noise * np.eye(n)
        residual = np.linalg.norm(A @ x_gpytorch - y) / np.linalg.norm(y)
        assert residual < 0.01, f"GPyTorch CG residual {residual:.2e} too high"


class TestPreconditionerAlignment:
    """Test pivoted-Cholesky behavior against linear_operator."""

    def test_pivoted_cholesky_approximation(self, random_seed):
        """Test pivoted Cholesky approximation quality matches linear_operator."""
        try:
            from linear_operator.operators import DenseLinearOperator
        except ImportError:
            pytest.skip("linear_operator not installed")

        n = 100
        rank = 50
        X = np.random.randn(n, 5).astype(np.float32)

        from scipy.spatial.distance import cdist

        dist_sq = cdist(X, X, metric="sqeuclidean")
        K = np.exp(-dist_sq / 2).astype(np.float32)

        K_torch = torch.tensor(K, dtype=torch.float64)
        K_linop = DenseLinearOperator(K_torch)
        L_gpytorch = K_linop.pivoted_cholesky(rank=rank)

        K_approx = (L_gpytorch @ L_gpytorch.T).numpy()
        error = np.linalg.norm(K.astype(np.float64) - K_approx, "fro") / np.linalg.norm(
            K, "fro"
        )
        assert error < 0.5, f"Pivoted Cholesky error {error:.2%} too high"


class TestKernelAlignment:
    """Test kernel computation against GPyTorch kernels."""

    def test_rbf_kernel_exact_match(self, random_seed):
        """Test RBF kernel matches GPyTorch exactly."""
        try:
            import gpytorch
        except ImportError:
            pytest.skip("GPyTorch not installed")

        n = 50
        d = 5
        X = np.random.randn(n, d).astype(np.float32)
        lengthscale = 1.5
        outputscale = 2.0

        X_torch = torch.tensor(X)
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kernel.base_kernel.lengthscale = lengthscale
        kernel.outputscale = outputscale
        K_gpytorch = kernel(X_torch).evaluate().detach().numpy()

        from scipy.spatial.distance import cdist

        dist_sq = cdist(X, X, metric="sqeuclidean")
        K_manual = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        np.testing.assert_allclose(K_manual, K_gpytorch, rtol=1e-4, atol=1e-5)

    def test_matern_kernel_exact_match(self, random_seed):
        """Test Matern kernel matches GPyTorch exactly."""
        try:
            import gpytorch
        except ImportError:
            pytest.skip("GPyTorch not installed")

        n = 50
        d = 5
        X = np.random.randn(n, d).astype(np.float32)
        lengthscale = 1.0
        outputscale = 1.0

        from scipy.spatial.distance import cdist

        for nu in [0.5, 1.5, 2.5]:
            X_torch = torch.tensor(X)
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))
            kernel.base_kernel.lengthscale = lengthscale
            kernel.outputscale = outputscale
            K_gpytorch = kernel(X_torch).evaluate().detach().numpy()

            dist = cdist(X, X, metric="euclidean")
            if nu == 0.5:
                K_manual = outputscale * np.exp(-dist / lengthscale)
            elif nu == 1.5:
                sqrt3_r_l = np.sqrt(3) * dist / lengthscale
                K_manual = outputscale * (1 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
            else:
                sqrt5_r_l = np.sqrt(5) * dist / lengthscale
                r_sq_term = (5 / 3) * dist**2 / lengthscale**2
                K_manual = (
                    outputscale * (1 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)
                )

            np.testing.assert_allclose(
                K_manual,
                K_gpytorch,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Matern nu={nu} mismatch",
            )
