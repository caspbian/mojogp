"""Unit tests for composite kernel operations.

Tests the fundamental operations of composite kernels:
- Add: K = K1 + K2
- Multiply: K = K1 * K2
- Scale: K = c * K1
- Nested: K = (K1 + K2) * K3

These tests verify that composite kernel matvec operations
produce correct results compared to explicit matrix computation.
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist


# =============================================================================
# Kernel Functions
# =============================================================================


def rbf_kernel(
    X1: np.ndarray, X2: np.ndarray, lengthscale: float = 1.0, outputscale: float = 1.0
) -> np.ndarray:
    """Compute RBF kernel matrix."""
    dists = cdist(X1 / lengthscale, X2 / lengthscale, metric="euclidean")
    return outputscale * np.exp(-0.5 * dists**2)


def matern52_kernel(
    X1: np.ndarray, X2: np.ndarray, lengthscale: float = 1.0, outputscale: float = 1.0
) -> np.ndarray:
    """Compute Matern 5/2 kernel matrix."""
    dists = cdist(X1 / lengthscale, X2 / lengthscale, metric="euclidean")
    sqrt5 = np.sqrt(5)
    return outputscale * (1 + sqrt5 * dists + 5 / 3 * dists**2) * np.exp(-sqrt5 * dists)


def matern32_kernel(
    X1: np.ndarray, X2: np.ndarray, lengthscale: float = 1.0, outputscale: float = 1.0
) -> np.ndarray:
    """Compute Matern 3/2 kernel matrix."""
    dists = cdist(X1 / lengthscale, X2 / lengthscale, metric="euclidean")
    sqrt3 = np.sqrt(3)
    return outputscale * (1 + sqrt3 * dists) * np.exp(-sqrt3 * dists)


def linear_kernel(
    X1: np.ndarray, X2: np.ndarray, outputscale: float = 1.0
) -> np.ndarray:
    """Compute linear kernel matrix."""
    return outputscale * (X1 @ X2.T)


def periodic_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    lengthscale: float = 1.0,
    period: float = 1.0,
    outputscale: float = 1.0,
) -> np.ndarray:
    """Compute periodic kernel matrix."""
    dists = cdist(X1, X2, metric="euclidean")
    return outputscale * np.exp(
        -2 * np.sin(np.pi * dists / period) ** 2 / lengthscale**2
    )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_data():
    """Small dataset for unit tests."""
    np.random.seed(42)
    n = 50
    d = 5
    X = np.random.randn(n, d).astype(np.float32)
    return X


@pytest.fixture
def medium_data():
    """Medium dataset for unit tests."""
    np.random.seed(42)
    n = 200
    d = 5
    X = np.random.randn(n, d).astype(np.float32)
    return X


# =============================================================================
# Test Classes
# =============================================================================


class TestAddKernel:
    """Test K = K1 + K2 operations."""

    def test_add_rbf_matern52(self, small_data):
        """Test RBF + Matern52 kernel."""
        X = small_data
        n = X.shape[0]

        # Compute individual kernels
        K_rbf = rbf_kernel(X, X, lengthscale=1.0, outputscale=1.0)
        K_matern = matern52_kernel(X, X, lengthscale=1.0, outputscale=1.0)

        # Composite kernel
        K_add = K_rbf + K_matern

        # Test matvec
        v = np.random.randn(n).astype(np.float32)

        # Direct matvec
        result_direct = K_add @ v

        # Component matvecs
        result_components = (K_rbf @ v) + (K_matern @ v)

        # Should be identical
        np.testing.assert_allclose(result_direct, result_components, rtol=1e-5)

    def test_add_three_kernels(self, small_data):
        """Test K1 + K2 + K3."""
        X = small_data
        n = X.shape[0]

        K1 = rbf_kernel(X, X, lengthscale=0.5)
        K2 = matern52_kernel(X, X, lengthscale=1.0)
        K3 = matern32_kernel(X, X, lengthscale=2.0)

        K_add = K1 + K2 + K3

        v = np.random.randn(n).astype(np.float32)

        result_direct = K_add @ v
        result_components = (K1 @ v) + (K2 @ v) + (K3 @ v)

        np.testing.assert_allclose(result_direct, result_components, rtol=1e-5)

    def test_add_preserves_symmetry(self, small_data):
        """Test that K1 + K2 is symmetric."""
        X = small_data

        K_rbf = rbf_kernel(X, X)
        K_matern = matern52_kernel(X, X)
        K_add = K_rbf + K_matern

        np.testing.assert_allclose(K_add, K_add.T, rtol=1e-6)

    def test_add_preserves_positive_definiteness(self, small_data):
        """Test that K1 + K2 is positive definite."""
        X = small_data

        K_rbf = rbf_kernel(X, X)
        K_matern = matern52_kernel(X, X)
        K_add = K_rbf + K_matern

        # Add small jitter for numerical stability
        K_add += 1e-6 * np.eye(K_add.shape[0])

        # Check eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(K_add)
        assert np.all(eigenvalues > 0), "K1 + K2 should be positive definite"


class TestMultiplyKernel:
    """Test K = K1 * K2 operations (element-wise product)."""

    def test_multiply_rbf_matern52(self, small_data):
        """Test RBF * Matern52 kernel."""
        X = small_data
        n = X.shape[0]

        K_rbf = rbf_kernel(X, X, lengthscale=1.0)
        K_matern = matern52_kernel(X, X, lengthscale=1.0)

        # Element-wise product
        K_mult = K_rbf * K_matern

        # Test matvec
        v = np.random.randn(n).astype(np.float32)

        result = K_mult @ v

        # Should be finite
        assert np.all(np.isfinite(result))

    def test_multiply_preserves_symmetry(self, small_data):
        """Test that K1 * K2 is symmetric."""
        X = small_data

        K_rbf = rbf_kernel(X, X)
        K_matern = matern52_kernel(X, X)
        K_mult = K_rbf * K_matern

        np.testing.assert_allclose(K_mult, K_mult.T, rtol=1e-6)

    def test_multiply_preserves_positive_definiteness(self, small_data):
        """Test that K1 * K2 is positive semi-definite (Schur product theorem)."""
        X = small_data

        K_rbf = rbf_kernel(X, X)
        K_matern = matern52_kernel(X, X)
        K_mult = K_rbf * K_matern

        # Add small jitter
        K_mult += 1e-6 * np.eye(K_mult.shape[0])

        # Check eigenvalues are non-negative
        eigenvalues = np.linalg.eigvalsh(K_mult)
        assert np.all(eigenvalues >= -1e-10), "K1 * K2 should be positive semi-definite"


class TestScaleKernel:
    """Test K = c * K1 operations."""

    def test_scale_rbf(self, small_data):
        """Test scaling RBF kernel."""
        X = small_data
        n = X.shape[0]

        K_rbf = rbf_kernel(X, X, lengthscale=1.0, outputscale=1.0)
        scale = 2.5
        K_scaled = scale * K_rbf

        v = np.random.randn(n).astype(np.float32)

        result_scaled = K_scaled @ v
        result_manual = scale * (K_rbf @ v)

        np.testing.assert_allclose(result_scaled, result_manual, rtol=1e-6)

    def test_scale_preserves_symmetry(self, small_data):
        """Test that c * K is symmetric."""
        X = small_data

        K_rbf = rbf_kernel(X, X)
        K_scaled = 3.0 * K_rbf

        np.testing.assert_allclose(K_scaled, K_scaled.T, rtol=1e-6)

    def test_scale_with_different_values(self, small_data):
        """Test scaling with various scale values."""
        X = small_data
        n = X.shape[0]

        K_rbf = rbf_kernel(X, X)
        v = np.random.randn(n).astype(np.float32)

        for scale in [0.1, 1.0, 10.0, 100.0]:
            K_scaled = scale * K_rbf
            result = K_scaled @ v
            expected = scale * (K_rbf @ v)
            np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestNestedCompositeKernel:
    """Test nested composite kernels like (K1 + K2) * K3."""

    def test_add_then_multiply(self, small_data):
        """Test (K1 + K2) * K3."""
        X = small_data
        n = X.shape[0]

        K1 = rbf_kernel(X, X, lengthscale=0.5)
        K2 = matern52_kernel(X, X, lengthscale=1.0)
        K3 = matern32_kernel(X, X, lengthscale=2.0)

        K_nested = (K1 + K2) * K3

        v = np.random.randn(n).astype(np.float32)
        result = K_nested @ v

        assert np.all(np.isfinite(result))

    def test_multiply_then_add(self, small_data):
        """Test K1 * K2 + K3."""
        X = small_data
        n = X.shape[0]

        K1 = rbf_kernel(X, X, lengthscale=0.5)
        K2 = matern52_kernel(X, X, lengthscale=1.0)
        K3 = matern32_kernel(X, X, lengthscale=2.0)

        K_nested = (K1 * K2) + K3

        v = np.random.randn(n).astype(np.float32)
        result = K_nested @ v

        assert np.all(np.isfinite(result))

    def test_scale_then_add(self, small_data):
        """Test c1 * K1 + c2 * K2."""
        X = small_data
        n = X.shape[0]

        K1 = rbf_kernel(X, X)
        K2 = matern52_kernel(X, X)

        c1, c2 = 2.0, 0.5
        K_nested = c1 * K1 + c2 * K2

        v = np.random.randn(n).astype(np.float32)

        result = K_nested @ v
        expected = c1 * (K1 @ v) + c2 * (K2 @ v)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_deeply_nested(self, small_data):
        """Test deeply nested: ((K1 + K2) * K3) + K4."""
        X = small_data
        n = X.shape[0]

        K1 = rbf_kernel(X, X, lengthscale=0.5)
        K2 = matern52_kernel(X, X, lengthscale=1.0)
        K3 = matern32_kernel(X, X, lengthscale=1.5)
        K4 = rbf_kernel(X, X, lengthscale=2.0)

        K_nested = ((K1 + K2) * K3) + K4

        v = np.random.randn(n).astype(np.float32)
        result = K_nested @ v

        assert np.all(np.isfinite(result))

        # Verify symmetry
        np.testing.assert_allclose(K_nested, K_nested.T, rtol=1e-6)


class TestCompositeKernelGradients:
    """Test that gradients flow correctly through composite kernels."""

    def test_add_kernel_gradient_lengthscale(self, small_data):
        """Test gradient of K1 + K2 w.r.t. lengthscale."""
        X = small_data
        n = X.shape[0]

        # Numerical gradient
        eps = 1e-5
        ls = 1.0

        K_plus = rbf_kernel(X, X, lengthscale=ls + eps) + matern52_kernel(
            X, X, lengthscale=ls + eps
        )
        K_minus = rbf_kernel(X, X, lengthscale=ls - eps) + matern52_kernel(
            X, X, lengthscale=ls - eps
        )

        dK_numerical = (K_plus - K_minus) / (2 * eps)

        # Should be finite
        assert np.all(np.isfinite(dK_numerical))

        # Should be symmetric
        np.testing.assert_allclose(dK_numerical, dK_numerical.T, rtol=1e-4)

    def test_multiply_kernel_gradient_lengthscale(self, small_data):
        """Test gradient of K1 * K2 w.r.t. lengthscale."""
        X = small_data

        eps = 1e-5
        ls = 1.0

        K_plus = rbf_kernel(X, X, lengthscale=ls + eps) * matern52_kernel(
            X, X, lengthscale=ls + eps
        )
        K_minus = rbf_kernel(X, X, lengthscale=ls - eps) * matern52_kernel(
            X, X, lengthscale=ls - eps
        )

        dK_numerical = (K_plus - K_minus) / (2 * eps)

        assert np.all(np.isfinite(dK_numerical))
        np.testing.assert_allclose(dK_numerical, dK_numerical.T, rtol=1e-4)


class TestCompositeKernelWithNoise:
    """Test composite kernels with added noise."""

    def test_add_kernel_with_noise(self, small_data):
        """Test (K1 + K2) + noise * I."""
        X = small_data
        n = X.shape[0]
        noise = 0.1

        K_rbf = rbf_kernel(X, X)
        K_matern = matern52_kernel(X, X)
        K_composite = K_rbf + K_matern + noise * np.eye(n)

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(K_composite)
        assert np.all(eigenvalues > 0)

        # Should be invertible
        K_inv = np.linalg.inv(K_composite)
        assert np.all(np.isfinite(K_inv))

    def test_multiply_kernel_with_noise(self, small_data):
        """Test (K1 * K2) + noise * I."""
        X = small_data
        n = X.shape[0]
        noise = 0.1

        K_rbf = rbf_kernel(X, X)
        K_matern = matern52_kernel(X, X)
        K_composite = (K_rbf * K_matern) + noise * np.eye(n)

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(K_composite)
        assert np.all(eigenvalues > 0)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
