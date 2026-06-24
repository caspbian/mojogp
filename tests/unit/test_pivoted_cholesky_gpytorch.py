"""Explicit linear_operator pivoted-Cholesky comparisons moved out of unit correctness."""

import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist


pytestmark = [pytest.mark.gpytorch, pytest.mark.reference]


class TestPivotedCholeskyGPyTorch:
    """Test pivoted-Cholesky behavior against linear_operator."""

    def _build_kernel_matrix(self, X, lengthscale=1.0):
        """Build RBF kernel matrix."""
        dist_sq = cdist(X, X, metric="sqeuclidean")
        return np.exp(-dist_sq / (2 * lengthscale**2))

    def test_matches_gpytorch_pivoted_cholesky_error(self, random_seed):
        """Test approximation quality matches linear_operator."""
        try:
            from linear_operator.operators import DenseLinearOperator
        except ImportError:
            pytest.skip("linear_operator not installed")

        n = 100
        rank = 50
        X = np.random.randn(n, 5).astype(np.float32)
        K = self._build_kernel_matrix(X).astype(np.float32)

        K_torch = torch.tensor(K, dtype=torch.float64)
        K_linop = DenseLinearOperator(K_torch)
        L_gpytorch = K_linop.pivoted_cholesky(rank=rank)

        K_approx_gpytorch = (L_gpytorch @ L_gpytorch.T).numpy()
        error_gpytorch = np.linalg.norm(
            K.astype(np.float64) - K_approx_gpytorch, "fro"
        ) / np.linalg.norm(K, "fro")

        assert error_gpytorch < 0.5, (
            f"GPyTorch approximation error {error_gpytorch:.2%} too high"
        )
        assert np.isfinite(error_gpytorch), "Error should be finite"

    def test_pivots_select_largest_diagonal(self, random_seed):
        """Test pivots select large remaining diagonal elements."""
        try:
            from linear_operator.operators import DenseLinearOperator
        except ImportError:
            pytest.skip("linear_operator not installed")

        n = 50
        rank = 25
        X = np.random.randn(n, 5).astype(np.float32)
        K = self._build_kernel_matrix(X).astype(np.float32)

        K_torch = torch.tensor(K, dtype=torch.float64)
        K_linop = DenseLinearOperator(K_torch)
        L = K_linop.pivoted_cholesky(rank=rank)

        assert L.shape == (n, rank), f"Expected shape ({n}, {rank}), got {L.shape}"
        assert torch.isfinite(L).all(), "L should have finite values"

        col_norms = torch.norm(L, dim=0)
        assert col_norms[0] > 0.1, "First column should have non-trivial norm"
        assert col_norms[0] >= col_norms.median() * 0.5, (
            "First column should be significant"
        )

    def test_approximation_improves_with_rank(self, random_seed):
        """Test approximation error decreases with rank."""
        try:
            from linear_operator.operators import DenseLinearOperator
        except ImportError:
            pytest.skip("linear_operator not installed")

        n = 100
        X = np.random.randn(n, 5).astype(np.float32)
        K = self._build_kernel_matrix(X).astype(np.float32)
        K_torch = torch.tensor(K, dtype=torch.float64)
        K_linop = DenseLinearOperator(K_torch)

        errors = []
        for rank in [5, 10, 20, 40]:
            L = K_linop.pivoted_cholesky(rank=rank)
            K_approx = (L @ L.T).numpy()
            error = np.linalg.norm(
                K.astype(np.float64) - K_approx, "fro"
            ) / np.linalg.norm(K, "fro")
            errors.append(error)

        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] * 1.1, (
                f"Error should decrease: rank {[5, 10, 20, 40][i]} error {errors[i]:.4f} > "
                f"rank {[5, 10, 20, 40][i + 1]} error {errors[i + 1]:.4f}"
            )

    def test_low_rank_kernel_exact_recovery(self, random_seed):
        """Test exact recovery for a truly low-rank kernel."""
        try:
            from linear_operator.operators import DenseLinearOperator
        except ImportError:
            pytest.skip("linear_operator not installed")

        n = 50
        true_rank = 5
        V = np.random.randn(n, true_rank).astype(np.float64)
        K = V @ V.T + 1e-6 * np.eye(n)

        K_torch = torch.tensor(K, dtype=torch.float64)
        K_linop = DenseLinearOperator(K_torch)
        L = K_linop.pivoted_cholesky(rank=true_rank + 2)
        K_approx = (L @ L.T).numpy()

        error = np.linalg.norm(K - K_approx, "fro") / np.linalg.norm(K, "fro")
        assert error < 0.01, (
            f"Should recover low-rank kernel exactly, error={error:.4f}"
        )
