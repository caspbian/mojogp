"""
Tests for kernel matrix computation.

Compares kernel formulas against sklearn and direct algebraic references.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF, Matern


class TestKernelMatrixVsSklearn:
    """Test kernel matrices match sklearn reference implementations."""

    @pytest.mark.parametrize("lengthscale", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("outputscale", [0.5, 1.0, 2.0])
    def test_rbf_kernel_matches_sklearn(self, lengthscale, outputscale, random_seed):
        """Test RBF kernel matrix matches sklearn."""
        n = 50
        d = 5
        X = np.random.randn(n, d).astype(np.float64)

        # Sklearn reference
        sklearn_kernel = RBF(length_scale=lengthscale)
        K_sklearn = outputscale * sklearn_kernel(X)

        # Manual computation (same as MojoGP)
        dist_sq = cdist(X, X, metric="sqeuclidean")
        K_manual = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))

        np.testing.assert_allclose(K_manual, K_sklearn, rtol=1e-10)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_matern_kernel_matches_sklearn(self, nu, random_seed):
        """Test Matern kernel matrix matches sklearn."""
        n = 50
        d = 5
        X = np.random.randn(n, d).astype(np.float64)
        lengthscale = 1.0
        outputscale = 1.0

        # Sklearn reference
        sklearn_kernel = Matern(length_scale=lengthscale, nu=nu)
        K_sklearn = outputscale * sklearn_kernel(X)

        # Manual computation
        dist = cdist(X, X, metric="euclidean")

        if nu == 1.5:  # Matern 3/2
            sqrt3 = np.sqrt(3.0)
            sqrt3_r_l = sqrt3 * dist / lengthscale
            K_manual = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif nu == 2.5:  # Matern 5/2
            sqrt5 = np.sqrt(5.0)
            sqrt5_r_l = sqrt5 * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist**2 / (lengthscale**2)
            K_manual = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(K_manual, K_sklearn, rtol=1e-10)


class TestKernelProperties:
    """Test general kernel properties."""

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "matern52"])
    def test_kernel_symmetry(self, kernel_name, random_seed):
        """Test kernel matrix is symmetric."""
        n = 50
        X = np.random.randn(n, 5).astype(np.float64)
        lengthscale = 1.0
        outputscale = 1.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        dist = np.sqrt(dist_sq)

        if kernel_name == "rbf":
            K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        elif kernel_name == "matern32":
            sqrt3_r_l = np.sqrt(3.0) * dist / lengthscale
            K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif kernel_name == "matern52":
            sqrt5_r_l = np.sqrt(5.0) * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
            K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "matern52"])
    def test_kernel_positive_definite(self, kernel_name, random_seed):
        """Test kernel matrix is positive definite."""
        n = 50
        X = np.random.randn(n, 5).astype(np.float64)
        lengthscale = 1.0
        outputscale = 1.0
        noise = 0.01

        dist_sq = cdist(X, X, metric="sqeuclidean")
        dist = np.sqrt(dist_sq)

        if kernel_name == "rbf":
            K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        elif kernel_name == "matern32":
            sqrt3_r_l = np.sqrt(3.0) * dist / lengthscale
            K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif kernel_name == "matern52":
            sqrt5_r_l = np.sqrt(5.0) * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
            K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        # Add noise for numerical stability
        K_noisy = K + noise * np.eye(n)

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(K_noisy)
        assert np.all(eigenvalues > 0), (
            f"Kernel {kernel_name} has non-positive eigenvalues"
        )

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "matern52"])
    def test_kernel_diagonal_is_outputscale(self, kernel_name, random_seed):
        """Test kernel diagonal equals outputscale."""
        n = 50
        X = np.random.randn(n, 5).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        dist = np.sqrt(dist_sq)

        if kernel_name == "rbf":
            K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        elif kernel_name == "matern32":
            sqrt3_r_l = np.sqrt(3.0) * dist / lengthscale
            K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif kernel_name == "matern52":
            sqrt5_r_l = np.sqrt(5.0) * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
            K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(np.diag(K), outputscale * np.ones(n), rtol=1e-10)

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "matern52"])
    def test_kernel_bounded(self, kernel_name, random_seed):
        """Test kernel values are bounded by outputscale."""
        n = 50
        X = np.random.randn(n, 5).astype(np.float64)
        lengthscale = 1.0
        outputscale = 2.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        dist = np.sqrt(dist_sq)

        if kernel_name == "rbf":
            K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        elif kernel_name == "matern32":
            sqrt3_r_l = np.sqrt(3.0) * dist / lengthscale
            K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif kernel_name == "matern52":
            sqrt5_r_l = np.sqrt(5.0) * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
            K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        assert np.all(K >= 0), f"Kernel {kernel_name} has negative values"
        assert np.all(K <= outputscale + 1e-10), (
            f"Kernel {kernel_name} exceeds outputscale"
        )


class TestKernelNumericalStability:
    """Test kernel numerical stability."""

    def test_rbf_large_distance(self, random_seed):
        """Test RBF kernel with large distances (should be ~0)."""
        n = 10
        X = np.random.randn(n, 5).astype(np.float64) * 100  # Large spread
        lengthscale = 0.1  # Small lengthscale
        outputscale = 1.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))

        # Off-diagonal should be very small
        off_diag = K[~np.eye(n, dtype=bool)]
        assert np.all(off_diag < 1e-10), "Large distance RBF should be ~0"

    def test_rbf_small_distance(self, random_seed):
        """Test RBF kernel with small distances (should be ~outputscale)."""
        n = 10
        X = np.random.randn(n, 5).astype(np.float64) * 0.001  # Very close points
        lengthscale = 1.0
        outputscale = 1.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))

        # All values should be close to outputscale
        assert np.all(K > 0.99 * outputscale), (
            "Small distance RBF should be ~outputscale"
        )

    def test_kernel_with_duplicate_points(self, random_seed):
        """Test kernel handles duplicate points correctly."""
        n = 10
        X = np.random.randn(n, 5).astype(np.float64)
        X = np.vstack([X, X[0:3]])  # Add duplicates

        lengthscale = 1.0
        outputscale = 1.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))

        # Duplicate rows should have K=outputscale
        np.testing.assert_allclose(K[0, n], outputscale, rtol=1e-10)
        np.testing.assert_allclose(K[1, n + 1], outputscale, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
