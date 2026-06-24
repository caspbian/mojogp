"""
Tests for kernel math functions.

Tests each kernel function against scipy/sklearn reference implementations.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import RBF, Matern


class TestRBFKernel:
    """Tests for RBF kernel."""

    def test_rbf_value_matches_sklearn(self):
        """Test RBF kernel values match sklearn."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        # Sklearn reference
        sklearn_kernel = RBF(length_scale=lengthscale)
        K_sklearn = outputscale * sklearn_kernel(X)

        # Manual computation (same as MojoGP)
        dist_sq = cdist(X, X, metric="sqeuclidean")
        inv_2ls2 = -1.0 / (2.0 * lengthscale**2)
        K_manual = outputscale * np.exp(dist_sq * inv_2ls2)

        np.testing.assert_allclose(K_manual, K_sklearn, rtol=1e-10)

    def test_rbf_diagonal_is_outputscale(self):
        """Test that RBF diagonal equals outputscale."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        inv_2ls2 = -1.0 / (2.0 * lengthscale**2)
        K = outputscale * np.exp(dist_sq * inv_2ls2)

        np.testing.assert_allclose(np.diag(K), outputscale * np.ones(10), rtol=1e-10)

    def test_rbf_symmetry(self):
        """Test that RBF kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist_sq = cdist(X, X, metric="sqeuclidean")
        inv_2ls2 = -1.0 / (2.0 * lengthscale**2)
        K = outputscale * np.exp(dist_sq * inv_2ls2)

        np.testing.assert_allclose(K, K.T, rtol=1e-10)


class TestMatern32Kernel:
    """Tests for Matern 3/2 kernel."""

    def test_matern32_value_matches_sklearn(self):
        """Test Matern 3/2 kernel values match sklearn."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        # Sklearn reference (nu=1.5 is Matern 3/2)
        sklearn_kernel = Matern(length_scale=lengthscale, nu=1.5)
        K_sklearn = outputscale * sklearn_kernel(X)

        # Manual computation (same as MojoGP)
        dist = cdist(X, X, metric="euclidean")
        sqrt3 = np.sqrt(3.0)
        sqrt3_r_l = sqrt3 * dist / lengthscale
        K_manual = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)

        np.testing.assert_allclose(K_manual, K_sklearn, rtol=1e-10)

    def test_matern32_diagonal_is_outputscale(self):
        """Test that Matern 3/2 diagonal equals outputscale."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist = cdist(X, X, metric="euclidean")
        sqrt3 = np.sqrt(3.0)
        sqrt3_r_l = sqrt3 * dist / lengthscale
        K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)

        np.testing.assert_allclose(np.diag(K), outputscale * np.ones(10), rtol=1e-10)

    def test_matern32_symmetry(self):
        """Test that Matern 3/2 kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist = cdist(X, X, metric="euclidean")
        sqrt3 = np.sqrt(3.0)
        sqrt3_r_l = sqrt3 * dist / lengthscale
        K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)

        np.testing.assert_allclose(K, K.T, rtol=1e-10)


class TestMatern52Kernel:
    """Tests for Matern 5/2 kernel."""

    def test_matern52_value_matches_sklearn(self):
        """Test Matern 5/2 kernel values match sklearn."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        # Sklearn reference (nu=2.5 is Matern 5/2)
        sklearn_kernel = Matern(length_scale=lengthscale, nu=2.5)
        K_sklearn = outputscale * sklearn_kernel(X)

        # Manual computation (same as MojoGP)
        dist = cdist(X, X, metric="euclidean")
        dist_sq = dist**2
        sqrt5 = np.sqrt(5.0)
        sqrt5_r_l = sqrt5 * dist / lengthscale
        r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
        K_manual = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(K_manual, K_sklearn, rtol=1e-10)

    def test_matern52_diagonal_is_outputscale(self):
        """Test that Matern 5/2 diagonal equals outputscale."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist = cdist(X, X, metric="euclidean")
        dist_sq = dist**2
        sqrt5 = np.sqrt(5.0)
        sqrt5_r_l = sqrt5 * dist / lengthscale
        r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
        K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(np.diag(K), outputscale * np.ones(10), rtol=1e-10)

    def test_matern52_symmetry(self):
        """Test that Matern 5/2 kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        dist = cdist(X, X, metric="euclidean")
        dist_sq = dist**2
        sqrt5 = np.sqrt(5.0)
        sqrt5_r_l = sqrt5 * dist / lengthscale
        r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
        K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(K, K.T, rtol=1e-10)


class TestKernelGradients:
    """Tests for kernel gradient functions."""

    def test_rbf_gradient_lengthscale_finite_diff(self):
        """Test RBF gradient w.r.t. lengthscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def rbf_kernel(X, ls, os):
            dist_sq = cdist(X, X, metric="sqeuclidean")
            inv_2ls2 = -1.0 / (2.0 * ls**2)
            return os * np.exp(dist_sq * inv_2ls2)

        # Finite difference gradient
        K_plus = rbf_kernel(X, lengthscale + eps, outputscale)
        K_minus = rbf_kernel(X, lengthscale - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical gradient: dK/dl = K * r^2 / l^3
        K = rbf_kernel(X, lengthscale, outputscale)
        dist_sq = cdist(X, X, metric="sqeuclidean")
        grad_analytical = K * dist_sq / (lengthscale**3)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_matern32_gradient_lengthscale_finite_diff(self):
        """Test Matern 3/2 gradient w.r.t. lengthscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def matern32_kernel(X, ls, os):
            dist = cdist(X, X, metric="euclidean")
            sqrt3 = np.sqrt(3.0)
            sqrt3_r_l = sqrt3 * dist / ls
            return os * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)

        # Finite difference gradient
        K_plus = matern32_kernel(X, lengthscale + eps, outputscale)
        K_minus = matern32_kernel(X, lengthscale - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical gradient: dK/dl = outputscale * 3 * r^2 / l^3 * exp(-sqrt(3)*r/l)
        dist = cdist(X, X, metric="euclidean")
        dist_sq = dist**2
        sqrt3 = np.sqrt(3.0)
        sqrt3_r_l = sqrt3 * dist / lengthscale
        grad_analytical = (
            outputscale * 3.0 * dist_sq / (lengthscale**3) * np.exp(-sqrt3_r_l)
        )

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_matern52_gradient_lengthscale_finite_diff(self):
        """Test Matern 5/2 gradient w.r.t. lengthscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def matern52_kernel(X, ls, os):
            dist = cdist(X, X, metric="euclidean")
            dist_sq = dist**2
            sqrt5 = np.sqrt(5.0)
            sqrt5_r_l = sqrt5 * dist / ls
            r_sq_term = (5.0 / 3.0) * dist_sq / (ls**2)
            return os * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        # Finite difference gradient
        K_plus = matern52_kernel(X, lengthscale + eps, outputscale)
        K_minus = matern52_kernel(X, lengthscale - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical gradient: dK/dl = outputscale * (5*r^2/3l^3) * (1 + sqrt(5)*r/l) * exp(-sqrt(5)*r/l)
        dist = cdist(X, X, metric="euclidean")
        dist_sq = dist**2
        sqrt5 = np.sqrt(5.0)
        sqrt5_r_l = sqrt5 * dist / lengthscale
        coeff = (5.0 / 3.0) * dist_sq / (lengthscale**3)
        grad_analytical = outputscale * coeff * (1.0 + sqrt5_r_l) * np.exp(-sqrt5_r_l)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)


class TestKernelProperties:
    """Tests for general kernel properties."""

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "matern52"])
    def test_kernel_positive_definite(self, kernel_name):
        """Test that kernels are positive definite."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        noise = 0.01

        if kernel_name == "rbf":
            dist_sq = cdist(X, X, metric="sqeuclidean")
            inv_2ls2 = -1.0 / (2.0 * lengthscale**2)
            K = outputscale * np.exp(dist_sq * inv_2ls2)
        elif kernel_name == "matern32":
            dist = cdist(X, X, metric="euclidean")
            sqrt3 = np.sqrt(3.0)
            sqrt3_r_l = sqrt3 * dist / lengthscale
            K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif kernel_name == "matern52":
            dist = cdist(X, X, metric="euclidean")
            dist_sq = dist**2
            sqrt5 = np.sqrt(5.0)
            sqrt5_r_l = sqrt5 * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
            K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        # Add noise for numerical stability
        K_noisy = K + noise * np.eye(len(X))

        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(K_noisy)
        assert np.all(eigenvalues > 0), (
            f"Kernel {kernel_name} has non-positive eigenvalues"
        )

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "matern52"])
    def test_kernel_bounded(self, kernel_name):
        """Test that kernel values are bounded by outputscale."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        if kernel_name == "rbf":
            dist_sq = cdist(X, X, metric="sqeuclidean")
            inv_2ls2 = -1.0 / (2.0 * lengthscale**2)
            K = outputscale * np.exp(dist_sq * inv_2ls2)
        elif kernel_name == "matern32":
            dist = cdist(X, X, metric="euclidean")
            sqrt3 = np.sqrt(3.0)
            sqrt3_r_l = sqrt3 * dist / lengthscale
            K = outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif kernel_name == "matern52":
            dist = cdist(X, X, metric="euclidean")
            dist_sq = dist**2
            sqrt5 = np.sqrt(5.0)
            sqrt5_r_l = sqrt5 * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
            K = outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)

        assert np.all(K >= 0), f"Kernel {kernel_name} has negative values"
        assert np.all(K <= outputscale + 1e-10), (
            f"Kernel {kernel_name} exceeds outputscale"
        )


class TestMatern12Kernel:
    """Tests for Matern 1/2 (exponential) kernel."""

    @staticmethod
    def _matern12(X, lengthscale, outputscale):
        dist = cdist(X, X, metric="euclidean")
        r = np.maximum(dist / lengthscale, 1e-10)
        return outputscale * np.exp(-r)

    def test_matern12_value_matches_sklearn(self):
        """Test Matern 1/2 kernel values match sklearn (nu=0.5)."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        sklearn_kernel = Matern(length_scale=lengthscale, nu=0.5)
        K_sklearn = outputscale * sklearn_kernel(X)
        K_manual = self._matern12(X, lengthscale, outputscale)

        # Slight tolerance due to r = max(r, 1e-10) clamp on diagonal
        np.testing.assert_allclose(K_manual, K_sklearn, rtol=1e-8, atol=1e-9)

    def test_matern12_diagonal_is_outputscale(self):
        """Test that Matern 1/2 diagonal equals outputscale (within clamp tolerance)."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._matern12(X, 1.5, 2.0)
        # r = max(0, 1e-10) on diagonal, so K_ii = os * exp(-1e-10) ≈ os
        np.testing.assert_allclose(np.diag(K), 2.0 * np.ones(10), rtol=1e-8)

    def test_matern12_symmetry(self):
        """Test that Matern 1/2 kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._matern12(X, 1.5, 2.0)
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_matern12_positive_definite(self):
        """Test that Matern 1/2 kernel is positive definite."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        K = self._matern12(X, 1.5, 2.0) + 0.01 * np.eye(20)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > 0)

    def test_matern12_bounded(self):
        """Test that Matern 1/2 values are bounded by outputscale."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        K = self._matern12(X, 1.5, 2.0)
        assert np.all(K >= 0)
        assert np.all(K <= 2.0 + 1e-10)


class TestRQKernel:
    """Tests for Rational Quadratic kernel."""

    @staticmethod
    def _rq(X, lengthscale, alpha, outputscale):
        dist_sq = cdist(X, X, metric="sqeuclidean")
        base = 1.0 + dist_sq / (2.0 * alpha * lengthscale**2)
        return outputscale * base ** (-alpha)

    def test_rq_diagonal_is_outputscale(self):
        """Test that RQ diagonal equals outputscale."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._rq(X, 1.5, 2.0, 3.0)
        np.testing.assert_allclose(np.diag(K), 3.0 * np.ones(10), rtol=1e-10)

    def test_rq_symmetry(self):
        """Test that RQ kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._rq(X, 1.5, 2.0, 3.0)
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_rq_positive_definite(self):
        """Test that RQ kernel is positive definite."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        K = self._rq(X, 1.5, 2.0, 3.0) + 0.01 * np.eye(20)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > 0)

    def test_rq_bounded(self):
        """Test that RQ values are bounded by outputscale."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        K = self._rq(X, 1.5, 2.0, 3.0)
        assert np.all(K >= 0)
        assert np.all(K <= 3.0 + 1e-10)

    def test_rq_converges_to_rbf_large_alpha(self):
        """Test that RQ -> RBF as alpha -> infinity."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        # RBF reference
        dist_sq = cdist(X, X, metric="sqeuclidean")
        K_rbf = outputscale * np.exp(-dist_sq / (2.0 * lengthscale**2))

        # RQ with very large alpha
        K_rq = self._rq(X, lengthscale, 1e6, outputscale)

        np.testing.assert_allclose(K_rq, K_rbf, rtol=1e-4)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0])
    def test_rq_gradient_lengthscale_finite_diff(self, alpha):
        """Test RQ gradient w.r.t. lengthscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._rq(X, lengthscale + eps, alpha, outputscale)
        K_minus = self._rq(X, lengthscale - eps, alpha, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/dl = K * alpha * dist_sq / (l^3 * base)
        dist_sq = cdist(X, X, metric="sqeuclidean")
        base = 1.0 + dist_sq / (2.0 * alpha * lengthscale**2)
        K = self._rq(X, lengthscale, alpha, outputscale)
        grad_analytical = K * dist_sq / (lengthscale**3 * base)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0])
    def test_rq_gradient_alpha_finite_diff(self, alpha):
        """Test RQ gradient w.r.t. alpha via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._rq(X, lengthscale, alpha + eps, outputscale)
        K_minus = self._rq(X, lengthscale, alpha - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/d(alpha) = K * (ratio/base - log(base))
        dist_sq = cdist(X, X, metric="sqeuclidean")
        ratio = dist_sq / (2.0 * alpha * lengthscale**2)
        base = 1.0 + ratio
        K = self._rq(X, lengthscale, alpha, outputscale)
        grad_analytical = K * (ratio / base - np.log(base))

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)


class TestLinearKernel:
    """Tests for Linear kernel."""

    @staticmethod
    def _linear_unified(X, param1, outputscale):
        """Linear kernel as in kernel_functions.mojo: k = os * param1 * (x^T x')."""
        return outputscale * param1 * (X @ X.T)

    @staticmethod
    def _linear_composable(X, variance, outputscale):
        """Linear kernel as in composable_kernel.mojo: k = os * (x^T x' + variance)."""
        return outputscale * (X @ X.T + variance)

    def test_linear_unified_symmetry(self):
        """Test that linear kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._linear_unified(X, 1.0, 2.0)
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_linear_composable_symmetry(self):
        """Test that composable linear kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._linear_composable(X, 0.5, 2.0)
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_linear_unified_positive_semidefinite(self):
        """Test that linear kernel is positive semi-definite."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        K = self._linear_unified(X, 1.0, 2.0) + 0.01 * np.eye(20)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > -1e-10)

    def test_linear_composable_with_variance(self):
        """Test that variance adds constant to all entries."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K_no_var = self._linear_composable(X, 0.0, 1.0)
        K_with_var = self._linear_composable(X, 1.0, 1.0)
        np.testing.assert_allclose(K_with_var - K_no_var, np.ones((10, 10)), rtol=1e-10)

    def test_linear_gradient_param1_finite_diff(self):
        """Test linear gradient w.r.t. param1 (variance) via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        param1 = 1.5
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._linear_unified(X, param1 + eps, outputscale)
        K_minus = self._linear_unified(X, param1 - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/d(param1) = outputscale * (x^T x')
        grad_analytical = outputscale * (X @ X.T)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_linear_diagonal_is_norm_squared(self):
        """Test that linear diagonal equals outputscale * param1 * ||x||^2."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        param1 = 1.5
        outputscale = 2.0
        K = self._linear_unified(X, param1, outputscale)
        expected_diag = outputscale * param1 * np.sum(X**2, axis=1)
        np.testing.assert_allclose(np.diag(K), expected_diag, rtol=1e-10)


class TestPeriodicKernel:
    """Tests for Periodic kernel."""

    @staticmethod
    def _periodic(X, lengthscale, period, outputscale):
        """Periodic kernel: k = os * exp(-2 * sum(sin^2(pi*diff/period)) / ls)."""
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                sin_sq_sum = np.sum(np.sin(np.pi * diff / period) ** 2)
                K[i, j] = outputscale * np.exp(-2.0 * sin_sq_sum / lengthscale)
        return K

    def test_periodic_diagonal_is_outputscale(self):
        """Test that periodic diagonal equals outputscale."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._periodic(X, 1.5, 2.0, 3.0)
        np.testing.assert_allclose(np.diag(K), 3.0 * np.ones(10), rtol=1e-10)

    def test_periodic_symmetry(self):
        """Test that periodic kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._periodic(X, 1.5, 2.0, 3.0)
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_periodic_positive_definite(self):
        """Test that periodic kernel is positive definite."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        K = self._periodic(X, 1.5, 2.0, 3.0) + 0.01 * np.eye(20)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > 0)

    def test_periodic_bounded(self):
        """Test that periodic values are bounded by outputscale."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        K = self._periodic(X, 1.5, 2.0, 3.0)
        assert np.all(K >= 0)
        assert np.all(K <= 3.0 + 1e-10)

    def test_periodic_is_periodic(self):
        """Test that kernel is periodic: k(x, x') = k(x + period, x')."""
        np.random.seed(42)
        period = 2.0
        X = np.random.randn(10, 1).astype(np.float64)
        X_shifted = X + period

        K_orig = self._periodic(X, 1.5, period, 2.0)
        # k(x+p, x'+p) should equal k(x, x') since both shift by same amount
        K_shifted = self._periodic(X_shifted, 1.5, period, 2.0)
        np.testing.assert_allclose(K_orig, K_shifted, rtol=1e-10)

    def test_periodic_gradient_lengthscale_finite_diff(self):
        """Test periodic gradient w.r.t. lengthscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        period = 2.0
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._periodic(X, lengthscale + eps, period, outputscale)
        K_minus = self._periodic(X, lengthscale - eps, period, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/dl = K * 2 * sin_sq_sum / l^2
        n = X.shape[0]
        grad_analytical = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                sin_sq_sum = np.sum(np.sin(np.pi * diff / period) ** 2)
                K_ij = outputscale * np.exp(-2.0 * sin_sq_sum / lengthscale)
                grad_analytical[i, j] = K_ij * 2.0 * sin_sq_sum / (lengthscale**2)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_periodic_gradient_period_finite_diff(self):
        """Test periodic gradient w.r.t. period via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        period = 2.0
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._periodic(X, lengthscale, period + eps, outputscale)
        K_minus = self._periodic(X, lengthscale, period - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/dp = K * (4*pi / (l * p^2)) * sum_d [diff_d * sin(u_d) * cos(u_d)]
        n = X.shape[0]
        grad_analytical = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                u = np.pi * diff / period
                sin_sq_sum = np.sum(np.sin(u) ** 2)
                K_ij = outputscale * np.exp(-2.0 * sin_sq_sum / lengthscale)
                sin_cos_sum = np.sum(diff * np.sin(u) * np.cos(u))
                grad_analytical[i, j] = (
                    K_ij * 4.0 * np.pi / (lengthscale * period**2) * sin_cos_sum
                )

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-3)


class TestPolynomialKernel:
    """Tests for Polynomial kernel."""

    @staticmethod
    def _polynomial(X, degree, offset, outputscale):
        """Polynomial kernel: k = os * (x^T x' + offset)^degree."""
        base = X @ X.T + offset
        base = np.maximum(base, 1e-10)
        return outputscale * base**degree

    def test_polynomial_symmetry(self):
        """Test that polynomial kernel is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        K = self._polynomial(X, 2.0, 1.0, 2.0)
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    def test_polynomial_positive_definite_degree2_positive_data(self):
        """Test that polynomial kernel (degree 2) is PD when base > 0."""
        np.random.seed(42)
        # Use positive data + large offset to ensure base > 0 everywhere
        X = np.abs(np.random.randn(20, 3).astype(np.float64))
        offset = 5.0  # Large offset ensures x^T x' + offset > 0
        K = self._polynomial(X, 2.0, offset, 2.0) + 0.01 * np.eye(20)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > 0)

    def test_polynomial_degree1_is_linear_plus_offset(self):
        """Test that polynomial with degree=1 is linear kernel + offset (when base > 0)."""
        np.random.seed(42)
        # Use positive data + large offset to avoid clamping
        X = np.abs(np.random.randn(10, 3).astype(np.float64))
        offset = 5.0
        outputscale = 2.0

        K_poly = self._polynomial(X, 1.0, offset, outputscale)
        K_linear = outputscale * (X @ X.T + offset)

        np.testing.assert_allclose(K_poly, K_linear, rtol=1e-10)

    def test_polynomial_diagonal(self):
        """Test polynomial diagonal: k(x,x) = os * (||x||^2 + offset)^degree."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        degree = 2.0
        offset = 1.0
        outputscale = 2.0

        K = self._polynomial(X, degree, offset, outputscale)
        expected_diag = outputscale * (np.sum(X**2, axis=1) + offset) ** degree
        np.testing.assert_allclose(np.diag(K), expected_diag, rtol=1e-10)

    def test_polynomial_gradient_offset_finite_diff(self):
        """Test polynomial gradient w.r.t. offset via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        degree = 2.0
        offset = 1.0
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._polynomial(X, degree, offset + eps, outputscale)
        K_minus = self._polynomial(X, degree, offset - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/d(offset) = os * degree * base^(degree-1)
        base = np.maximum(X @ X.T + offset, 1e-10)
        grad_analytical = outputscale * degree * base ** (degree - 1)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_polynomial_gradient_degree_finite_diff(self):
        """Test polynomial gradient w.r.t. degree via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        degree = 2.0
        offset = 1.0
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._polynomial(X, degree + eps, offset, outputscale)
        K_minus = self._polynomial(X, degree - eps, offset, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/d(degree) = K * log(base)
        base = np.maximum(X @ X.T + offset, 1e-10)
        K = self._polynomial(X, degree, offset, outputscale)
        grad_analytical = K * np.log(base)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_polynomial_gradient_outputscale_finite_diff(self):
        """Test polynomial gradient w.r.t. outputscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        degree = 2.0
        offset = 1.0
        outputscale = 2.0
        eps = 1e-5

        K_plus = self._polynomial(X, degree, offset, outputscale + eps)
        K_minus = self._polynomial(X, degree, offset, outputscale - eps)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/d(os) = K / os
        K = self._polynomial(X, degree, offset, outputscale)
        grad_analytical = K / outputscale

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)


class TestKernelPropertiesExtended:
    """Tests for general kernel properties across all kernel types."""

    @pytest.mark.parametrize(
        "kernel_name", ["rbf", "matern32", "matern52", "matern12", "rq", "periodic"]
    )
    def test_kernel_positive_definite(self, kernel_name):
        """Test that stationary kernels are positive definite."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        noise = 0.01

        K = self._compute_kernel(X, kernel_name, lengthscale, outputscale)
        K_noisy = K + noise * np.eye(len(X))
        eigenvalues = np.linalg.eigvalsh(K_noisy)
        assert np.all(eigenvalues > 0), (
            f"Kernel {kernel_name} has non-positive eigenvalues"
        )

    @pytest.mark.parametrize(
        "kernel_name", ["rbf", "matern32", "matern52", "matern12", "rq", "periodic"]
    )
    def test_kernel_bounded(self, kernel_name):
        """Test that stationary kernel values are bounded by outputscale."""
        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0

        K = self._compute_kernel(X, kernel_name, lengthscale, outputscale)
        assert np.all(K >= 0), f"Kernel {kernel_name} has negative values"
        assert np.all(K <= outputscale + 1e-10), (
            f"Kernel {kernel_name} exceeds outputscale"
        )

    @staticmethod
    def _compute_kernel(X, kernel_name, lengthscale, outputscale):
        dist = cdist(X, X, metric="euclidean")
        dist_sq = dist**2

        if kernel_name == "rbf":
            return outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
        elif kernel_name == "matern12":
            r = np.maximum(dist / lengthscale, 1e-10)
            return outputscale * np.exp(-r)
        elif kernel_name == "matern32":
            sqrt3_r_l = np.sqrt(3.0) * dist / lengthscale
            return outputscale * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        elif kernel_name == "matern52":
            sqrt5_r_l = np.sqrt(5.0) * dist / lengthscale
            r_sq_term = (5.0 / 3.0) * dist_sq / (lengthscale**2)
            return outputscale * (1.0 + sqrt5_r_l + r_sq_term) * np.exp(-sqrt5_r_l)
        elif kernel_name == "rq":
            alpha = 2.0
            base = 1.0 + dist_sq / (2.0 * alpha * lengthscale**2)
            return outputscale * base ** (-alpha)
        elif kernel_name == "periodic":
            n = X.shape[0]
            K = np.zeros((n, n))
            period = 2.0
            for i in range(n):
                for j in range(n):
                    diff = X[i] - X[j]
                    sin_sq_sum = np.sum(np.sin(np.pi * diff / period) ** 2)
                    K[i, j] = outputscale * np.exp(-2.0 * sin_sq_sum / lengthscale)
            return K
        else:
            raise ValueError(f"Unknown kernel: {kernel_name}")


class TestMatern12Gradients:
    """Tests for Matern 1/2 kernel gradients."""

    def test_matern12_gradient_lengthscale_finite_diff(self):
        """Test Matern 1/2 gradient w.r.t. lengthscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def matern12(X, ls, os):
            dist = cdist(X, X, metric="euclidean")
            r = np.maximum(dist / ls, 1e-10)
            return os * np.exp(-r)

        K_plus = matern12(X, lengthscale + eps, outputscale)
        K_minus = matern12(X, lengthscale - eps, outputscale)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        # Analytical: dK/dl = os * r * exp(-r) / l^2 = K * r / l
        # where r = dist / l, so dK/dl = K * dist / l^2
        dist = cdist(X, X, metric="euclidean")
        r = np.maximum(dist / lengthscale, 1e-10)
        K = outputscale * np.exp(-r)
        grad_analytical = K * dist / (lengthscale**2)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)

    def test_matern12_gradient_outputscale_finite_diff(self):
        """Test Matern 1/2 gradient w.r.t. outputscale via finite differences."""
        np.random.seed(42)
        X = np.random.randn(5, 2).astype(np.float64)
        lengthscale = 1.5
        outputscale = 2.0
        eps = 1e-5

        def matern12(X, ls, os):
            dist = cdist(X, X, metric="euclidean")
            r = np.maximum(dist / ls, 1e-10)
            return os * np.exp(-r)

        K_plus = matern12(X, lengthscale, outputscale + eps)
        K_minus = matern12(X, lengthscale, outputscale - eps)
        grad_fd = (K_plus - K_minus) / (2 * eps)

        K = matern12(X, lengthscale, outputscale)
        grad_analytical = K / outputscale

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
