"""Integration tests that Linear and Polynomial kernels train correctly.

Tests verify:
1. Training converges to finite NLL
2. Learned parameters are reasonable
3. Predictions are accurate on training data
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from mojogp import SingleOutputGP, Linear, Polynomial


class TestLinearKernelGradientSkip:
    """Test Linear kernel training."""

    @pytest.fixture(scope="class")
    def linear_data(self):
        """Generate data with a linear relationship."""
        np.random.seed(123)
        n = 2000
        d = 3
        X = np.random.randn(n, d).astype(np.float32)
        # y = 2*x0 + 0.5*x1 - x2 + noise
        y = (2.0 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + 0.1 * np.random.randn(n)).astype(
            np.float32
        )
        return X, y

    def test_linear_training_converges(self, linear_data):
        """Linear kernel training should converge to a finite NLL."""
        X, y = linear_data

        gp = SingleOutputGP(Linear())
        result = gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
            initial_noise=0.1,
            method="materialized",
        )

        nll = float(result.nll)
        assert np.isfinite(nll), f"NLL is not finite: {nll}"
        # NLL should be reasonable for well-fitting data
        assert nll < 10.0, f"NLL too high: {nll}"

    def test_linear_noise_learned(self, linear_data):
        """Linear kernel should learn a reasonable noise level."""
        X, y = linear_data

        gp = SingleOutputGP(Linear())
        result = gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
            initial_noise=0.1,
            method="materialized",
        )

        noise = float(result.noise)
        assert noise > 0, f"Noise should be positive: {noise}"
        assert np.isfinite(noise), f"Noise not finite: {noise}"
        # For well-fitting linear data, noise should be learned to be small
        assert noise < 1.0, f"Noise too high for near-linear data: {noise}"

    def test_linear_prediction_accuracy(self, linear_data):
        """Linear kernel predictions should be accurate for linear data."""
        X, y = linear_data

        gp = SingleOutputGP(Linear())
        gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
            initial_noise=0.1,
            method="materialized",
        )

        mean, std = gp.predict(X, return_std=True)
        rmse = np.sqrt(np.mean((mean - y) ** 2))
        # Linear kernel on linear data should fit well
        assert rmse < 1.0, f"RMSE too high for linear data: {rmse:.3f}"

    def test_linear_nll_decreases(self):
        """Linear kernel NLL should decrease with more iterations."""
        np.random.seed(789)
        n = 2000
        X = np.random.randn(n, 2).astype(np.float32)
        y = (X[:, 0] + 0.1 * np.random.randn(n)).astype(np.float32)

        gp_early = SingleOutputGP(Linear())
        result_early = gp_early.fit(
            X, y, max_iterations=5, learning_rate=0.05, initial_noise=0.1
        )

        gp_late = SingleOutputGP(Linear())
        result_late = gp_late.fit(
            X, y, max_iterations=80, learning_rate=0.05, initial_noise=0.1
        )

        assert result_late.nll < result_early.nll, (
            f"Linear NLL did not decrease: early={result_early.nll:.4f}, "
            f"late={result_late.nll:.4f}"
        )


class TestPolynomialKernelGradientSkip:
    """Test Polynomial kernel training."""

    @pytest.fixture(scope="class")
    def poly_data(self):
        """Generate data with a quadratic relationship."""
        np.random.seed(456)
        n = 2000
        d = 2
        X = np.random.randn(n, d).astype(np.float32)
        # y = x0^2 + x0*x1 + noise (quadratic)
        y = (X[:, 0] ** 2 + X[:, 0] * X[:, 1] + 0.1 * np.random.randn(n)).astype(
            np.float32
        )
        return X, y

    def test_polynomial_training_converges(self, poly_data):
        """Polynomial kernel training should converge to a finite NLL."""
        X, y = poly_data

        gp = SingleOutputGP(Polynomial(degree=2.0, offset=1.0))
        result = gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
            initial_noise=0.1,
            method="materialized",
        )

        nll = float(result.nll)
        assert np.isfinite(nll), f"NLL is not finite: {nll}"
        assert nll < 10.0, f"NLL too high: {nll}"

    def test_polynomial_noise_learned(self, poly_data):
        """Polynomial kernel should learn a reasonable noise level."""
        X, y = poly_data

        gp = SingleOutputGP(Polynomial(degree=2.0, offset=1.0))
        result = gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
            initial_noise=0.1,
            method="materialized",
        )

        noise = float(result.noise)
        assert noise > 0, f"Noise should be positive: {noise}"
        assert np.isfinite(noise), f"Noise not finite: {noise}"

    def test_polynomial_prediction_accuracy(self, poly_data):
        """Polynomial kernel predictions should be accurate for quadratic data."""
        X, y = poly_data

        gp = SingleOutputGP(Polynomial(degree=2.0, offset=1.0))
        gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
            initial_noise=0.1,
            method="materialized",
        )

        mean, std = gp.predict(X, return_std=True)
        rmse = np.sqrt(np.mean((mean - y) ** 2))
        # Polynomial kernel on quadratic data should fit reasonably
        assert rmse < 2.0, f"RMSE too high for quadratic data: {rmse:.3f}"

    def test_polynomial_params_finite(self, poly_data):
        """Polynomial kernel parameters should be finite after training."""
        X, y = poly_data

        gp = SingleOutputGP(Polynomial(degree=2.0, offset=1.0))
        result = gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
            initial_noise=0.1,
            method="materialized",
        )

        assert np.all(np.isfinite(result.params)), (
            f"Polynomial params not finite: {result.params}"
        )
        # outputscale (last param) should be positive
        assert result.params[-1] > 0, f"outputscale not positive: {result.params[-1]}"


class TestLinearPolynomialNLLComparison:
    """Compare NLL values to ensure gradient skip doesn't affect training quality."""

    def test_linear_nll_reasonable(self):
        """Linear kernel NLL should be reasonable for simple linear data."""
        np.random.seed(789)
        n = 2000
        X = np.random.randn(n, 2).astype(np.float32)
        y = (X[:, 0] + 0.1 * np.random.randn(n)).astype(np.float32)

        gp = SingleOutputGP(Linear())
        result = gp.fit(X, y, max_iterations=80, learning_rate=0.05, initial_noise=0.1)

        nll = float(result.nll)
        assert np.isfinite(nll), f"NLL is not finite: {nll}"
        noise = float(result.noise)
        assert noise < 1.0, f"Noise too high: {noise}"
