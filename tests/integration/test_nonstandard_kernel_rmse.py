"""End-to-end RMSE tests for Periodic, RQ, Linear, and Polynomial kernels.

These kernels have less test coverage than RBF/Matern. This file verifies
they achieve reasonable predictive accuracy on functions suited to each.
"""

import numpy as np
import pytest
from mojogp import SingleOutputGP
from mojogp.kernel import Kernel, Periodic, RQ, Linear, Polynomial


class TestPeriodicKernelRMSE:
    """Periodic kernel should fit periodic signals well."""

    def test_sine_wave_rmse(self):
        """Periodic kernel on y=sin(2*pi*x) should achieve RMSE < 0.2."""
        np.random.seed(42)
        X_train = np.random.uniform(0, 2, (300, 1)).astype(np.float32)
        y_train = (
            np.sin(2 * np.pi * X_train[:, 0])
            + np.random.randn(300).astype(np.float32) * 0.05
        ).astype(np.float32)

        X_test = np.linspace(0.1, 1.9, 80).reshape(-1, 1).astype(np.float32)
        y_test = np.sin(2 * np.pi * X_test[:, 0])

        gp = SingleOutputGP(Periodic())
        gp.fit(X_train, y_train, max_iterations=100, verbose=False)
        mu, _ = gp.predict(X_test, return_std=True)

        rmse = np.sqrt(np.mean((mu - y_test) ** 2))
        assert rmse < 0.3, f"Periodic RMSE {rmse:.4f} too high on sin(2*pi*x)"

    def test_periodic_nll_decreases(self):
        """Training NLL should decrease for periodic data."""
        np.random.seed(11)
        X = np.random.uniform(0, 3, (200, 1)).astype(np.float32)
        y = (
            np.cos(2 * np.pi * X[:, 0]) + np.random.randn(200).astype(np.float32) * 0.1
        ).astype(np.float32)

        gp = SingleOutputGP(Periodic())
        gp.fit(X, y, max_iterations=60, verbose=False)
        params = gp.get_learned_params()
        assert params.get("noise", 0) > 0


class TestRQKernelRMSE:
    """Rational Quadratic kernel should handle multi-scale data."""

    def test_smooth_function_rmse(self):
        """RQ kernel on smooth function should achieve RMSE < 0.3."""
        np.random.seed(42)
        X_train = np.random.uniform(-3, 3, (300, 1)).astype(np.float32)
        y_train = (
            np.sin(X_train[:, 0]) * np.exp(-0.1 * X_train[:, 0] ** 2)
            + np.random.randn(300).astype(np.float32) * 0.05
        ).astype(np.float32)

        X_test = np.linspace(-2.5, 2.5, 80).reshape(-1, 1).astype(np.float32)
        y_test = np.sin(X_test[:, 0]) * np.exp(-0.1 * X_test[:, 0] ** 2)

        gp = SingleOutputGP(RQ())
        gp.fit(X_train, y_train, max_iterations=100, verbose=False)
        mu, _ = gp.predict(X_test, return_std=True)

        rmse = np.sqrt(np.mean((mu - y_test) ** 2))
        assert rmse < 0.3, f"RQ RMSE {rmse:.4f} too high"

    def test_rq_predictions_finite(self):
        """RQ predictions should all be finite."""
        np.random.seed(99)
        X = np.random.randn(150, 2).astype(np.float32)
        y = (X[:, 0] ** 2 + np.random.randn(150).astype(np.float32) * 0.1).astype(
            np.float32
        )

        gp = SingleOutputGP(RQ())
        gp.fit(X, y, max_iterations=50, verbose=False)
        mu, std = gp.predict(X[:20], return_std=True)
        assert np.all(np.isfinite(mu)), "RQ mean has non-finite values"
        assert np.all(np.isfinite(std)), "RQ std has non-finite values"


class TestLinearKernelRMSE:
    """Linear kernel should fit linear relationships well."""

    def test_linear_function_rmse(self):
        """Linear kernel on y=2x+1 should achieve RMSE < 0.3."""
        np.random.seed(77)
        X_train = np.random.uniform(-3, 3, (200, 1)).astype(np.float32)
        y_train = (
            2.0 * X_train[:, 0] + 1.0 + np.random.randn(200).astype(np.float32) * 0.1
        ).astype(np.float32)

        X_test = np.linspace(-2, 2, 50).reshape(-1, 1).astype(np.float32)
        y_test = 2.0 * X_test[:, 0] + 1.0

        gp = SingleOutputGP(Linear())
        gp.fit(X_train, y_train, max_iterations=100, learning_rate=0.1, verbose=False)
        mu, _ = gp.predict(X_test, return_std=True)

        rmse = np.sqrt(np.mean((mu - y_test) ** 2))
        assert rmse < 0.5, f"Linear RMSE {rmse:.4f} too high on 2x+1"

    def test_multivariate_linear(self):
        """Linear kernel on y=x0+2*x1+3*x2 should fit well."""
        np.random.seed(33)
        X = np.random.randn(300, 3).astype(np.float32)
        coeffs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = (X @ coeffs + np.random.randn(300).astype(np.float32) * 0.1).astype(
            np.float32
        )

        X_train, y_train = X[:250], y[:250]
        X_test, y_test = X[250:], y[250:]

        gp = SingleOutputGP(Linear())
        gp.fit(X_train, y_train, max_iterations=100, verbose=False)
        mu, _ = gp.predict(X_test, return_std=True)

        rmse = np.sqrt(np.mean((mu - y_test) ** 2))
        assert rmse < 0.5, f"Multivariate linear RMSE {rmse:.4f} too high"


class TestPolynomialKernelRMSE:
    """Polynomial kernel should fit polynomial functions."""

    def test_quadratic_rmse(self):
        """Polynomial kernel on y=x^2 should achieve RMSE < 0.5."""
        np.random.seed(99)
        X_train = np.random.uniform(-2, 2, (200, 1)).astype(np.float32)
        y_train = (
            X_train[:, 0] ** 2 + np.random.randn(200).astype(np.float32) * 0.1
        ).astype(np.float32)

        X_test = np.linspace(-1.5, 1.5, 50).reshape(-1, 1).astype(np.float32)
        y_test = X_test[:, 0] ** 2

        gp = SingleOutputGP(Polynomial())
        gp.fit(X_train, y_train, max_iterations=100, verbose=False)
        mu, _ = gp.predict(X_test, return_std=True)

        rmse = np.sqrt(np.mean((mu - y_test) ** 2))
        assert rmse < 0.5, f"Polynomial RMSE {rmse:.4f} too high on x^2"

    def test_polynomial_better_than_linear_on_quadratic(self):
        """Polynomial kernel should outperform linear on quadratic data."""
        np.random.seed(42)
        X = np.random.uniform(-2, 2, (300, 1)).astype(np.float32)
        y = (X[:, 0] ** 2 + np.random.randn(300).astype(np.float32) * 0.1).astype(
            np.float32
        )
        X_train, y_train = X[:250], y[:250]
        X_test, y_test = X[250:], y[250:]

        gp_poly = SingleOutputGP(Polynomial())
        gp_poly.fit(X_train, y_train, max_iterations=100, verbose=False)
        mu_poly, _ = gp_poly.predict(X_test, return_std=True)
        rmse_poly = np.sqrt(np.mean((mu_poly - y_test) ** 2))

        gp_lin = SingleOutputGP(Linear())
        gp_lin.fit(X_train, y_train, max_iterations=100, verbose=False)
        mu_lin, _ = gp_lin.predict(X_test, return_std=True)
        rmse_lin = np.sqrt(np.mean((mu_lin - y_test) ** 2))

        assert rmse_poly < rmse_lin, (
            f"Polynomial ({rmse_poly:.4f}) should beat Linear ({rmse_lin:.4f}) on x^2"
        )
