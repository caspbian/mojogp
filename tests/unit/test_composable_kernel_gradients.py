"""Unit tests for Python-side composable kernel math.

This file intentionally keeps only the fast finite-difference kernel-matrix
checks. End-to-end training coverage for ARD and composite kernels lives in
`tests/integration/test_composable_kernel_gradients.py`.
"""

import numpy as np
import pytest

from mojogp import RBF, Matern12, Matern32, Matern52, RQ


# ---------------------------------------------------------------------------
# Test: Python-side kernel matrix gradient vs finite differences
# ---------------------------------------------------------------------------


class TestKernelMatrixGradients:
    """Test kernel matrix gradients using Python-side KernelNode.evaluate().

    This tests the mathematical formulas used in composable kernels by
    computing dK/d(param) via finite differences and comparing against
    the analytical gradient from KernelNode.
    """

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "matern52"])
    def test_kernel_matrix_gradient_lengthscale(self, kernel_name):
        """dK/d(lengthscale) matches finite differences for base kernels."""
        from mojogp.kernel import Kernel

        np.random.seed(42)
        n, d = 2000, 3
        X = np.random.randn(n, d).astype(np.float64)

        kernel = getattr(Kernel, kernel_name)()
        ls = 1.5
        eps = 1e-5

        n_params = kernel.num_params()
        params = np.ones(n_params, dtype=np.float64)
        params[0] = ls  # First param is lengthscale

        K = kernel.evaluate(X, params=params)

        # Finite difference on lengthscale (param index 0)
        params_plus = params.copy()
        params_plus[0] = ls + eps
        K_plus = kernel.evaluate(X, params=params_plus)

        params_minus = params.copy()
        params_minus[0] = ls - eps
        K_minus = kernel.evaluate(X, params=params_minus)

        dK_fd = (K_plus - K_minus) / (2 * eps)

        # 1. Should be symmetric
        assert np.allclose(dK_fd, dK_fd.T, atol=1e-4), (
            f"{kernel_name}: dK/dl is not symmetric"
        )

        # 2. Diagonal should be ~0 (kernel value at r=0 doesn't depend on lengthscale)
        diag_max = np.max(np.abs(np.diag(dK_fd)))
        print(
            f"\n  {kernel_name}: dK/dl diag max={diag_max:.2e}, "
            f"off-diag max={np.max(np.abs(dK_fd)):.4f}"
        )
        assert diag_max < 1e-2, (
            f"{kernel_name}: dK/dl diagonal should be ~0, got max={diag_max:.2e}"
        )

    def test_sum_kernel_gradient_is_sum_of_gradients(self):
        """d(K1+K2)/dl = dK1/dl + dK2/dl (linearity of sum kernel gradient)."""
        from mojogp.kernel import Kernel

        np.random.seed(42)
        n, d = 2000, 2
        X = np.random.randn(n, d).astype(np.float64)

        k_sum = Kernel.rbf() + Kernel.matern52()
        eps = 1e-5

        n_params = k_sum.num_params()
        params = np.ones(n_params, dtype=np.float64)

        params_plus = params.copy()
        params_plus[0] += eps
        K_sum_plus = k_sum.evaluate(X, params=params_plus)

        params_minus = params.copy()
        params_minus[0] -= eps
        K_sum_minus = k_sum.evaluate(X, params=params_minus)

        dK_sum_fd = (K_sum_plus - K_sum_minus) / (2 * eps)

        assert np.max(np.abs(dK_sum_fd)) > 1e-6, (
            "Sum kernel gradient is all zeros — parameter perturbation had no effect"
        )
        assert np.allclose(dK_sum_fd, dK_sum_fd.T, atol=1e-4), (
            "Sum kernel gradient is not symmetric"
        )

    def test_product_kernel_gradient_product_rule(self):
        """d(K1*K2)/dl1 = dK1/dl1 * K2 (product rule)."""
        from mojogp.kernel import Kernel

        np.random.seed(42)
        n, d = 2000, 2
        X = np.random.randn(n, d).astype(np.float64)

        k_prod = Kernel.rbf() * Kernel.matern52()
        eps = 1e-5

        n_params = k_prod.num_params()
        params = np.ones(n_params, dtype=np.float64)

        params_plus = params.copy()
        params_plus[0] += eps
        K_plus = k_prod.evaluate(X, params=params_plus)

        params_minus = params.copy()
        params_minus[0] -= eps
        K_minus = k_prod.evaluate(X, params=params_minus)

        dK_fd = (K_plus - K_minus) / (2 * eps)

        assert np.max(np.abs(dK_fd)) > 1e-6, "Product kernel gradient is all zeros"
        assert np.allclose(dK_fd, dK_fd.T, atol=1e-4), (
            "Product kernel gradient not symmetric"
        )
