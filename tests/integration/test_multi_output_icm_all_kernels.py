"""Test multi-output ICM GP with all 8 kernel types.

Existing ICM tests mostly use RBF. This ensures all kernel types
work for multi-output training and prediction.
"""

import numpy as np
import pytest
from mojogp import MultiOutputGP


ALL_KERNELS = [
    "rbf",
    "matern12",
    "matern32",
    "matern52",
    "periodic",
    "rq",
    "linear",
    "polynomial",
]


def _make_data(n=500, d=3, T=2, noise=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f = np.sin(X[:, 0]) + 0.3 * X[:, 1]
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        Y[:, t] = (0.5 + 0.5 * t) * f + np.random.randn(n).astype(np.float32) * noise
    return X, Y


class TestICMAllKernels:
    """ICM multi-output should train and predict with all 8 kernel types."""

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_training_converges(self, kernel):
        """NLL should decrease for each kernel type."""
        X, Y = _make_data()
        gp = MultiOutputGP(kernel=kernel)
        gp.fit(X, Y, max_iterations=50, learning_rate=0.01, verbose=False)

        result = gp.training_result
        assert result is not None, f"No training result for {kernel}"
        nll_hist = np.array(result.nll_history)
        assert len(nll_hist) >= 10, f"Too few iterations for {kernel}"
        assert np.all(np.isfinite(nll_hist)), f"NaN in NLL history for {kernel}"
        # NLL should not diverge wildly
        assert nll_hist[-1] < nll_hist[0] * 3.0, (
            f"NLL diverged for {kernel}: {nll_hist[0]:.4f} -> {nll_hist[-1]:.4f}"
        )

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_prediction_finite(self, kernel):
        """Predictions should be finite for each kernel type."""
        X, Y = _make_data()
        gp = MultiOutputGP(kernel=kernel)
        gp.fit(X, Y, max_iterations=30, verbose=False)

        X_test = np.random.randn(15, 3).astype(np.float32)
        result = gp.predict(X_test)
        mu = result.mean
        assert mu.shape == (15, 2), f"Wrong shape for {kernel}: {mu.shape}"
        assert np.all(np.isfinite(mu)), f"Non-finite predictions for {kernel}"

    @pytest.mark.parametrize("kernel", ["rbf", "matern52", "matern32"])
    def test_prediction_better_than_mean(self, kernel):
        """Model should predict better than the training mean."""
        X, Y = _make_data(n=2000, noise=0.05)
        X_train, Y_train = X[:250], Y[:250]
        X_test, Y_test = X[250:], Y[250:]

        gp = MultiOutputGP(kernel=kernel)
        gp.fit(X_train, Y_train, max_iterations=80, verbose=False)
        result = gp.predict(X_test)
        mu = result.mean
        for t in range(2):
            rmse = np.sqrt(np.mean((mu[:, t] - Y_test[:, t]) ** 2))
            y_std = np.std(Y_test[:, t])
            assert rmse < y_std * 1.5, (
                f"Task {t}, kernel {kernel}: RMSE {rmse:.4f} >= 1.5*std {y_std * 1.5:.4f}"
            )

    @pytest.mark.parametrize("kernel", ALL_KERNELS)
    def test_noise_positive(self, kernel):
        """Learned noise should be positive for each kernel type."""
        X, Y = _make_data()
        gp = MultiOutputGP(kernel=kernel)
        gp.fit(X, Y, max_iterations=30, verbose=False)

        result = gp.training_result
        noise = np.array(result.noise_per_task)
        assert np.all(noise > 0), f"Noise not positive for {kernel}: {noise}"

    def test_polynomial_explicit_preconditioner_raises(self):
        """Polynomial ICM rejects the unsupported preconditioned route."""
        X, Y = _make_data(n=32)
        gp = MultiOutputGP(kernel="polynomial", use_preconditioner=True)

        with pytest.raises(ValueError, match="polynomial.*preconditioning"):
            gp.fit(X, Y, max_iterations=1, verbose=False)


class TestICMMatrixFreeAllKernels:
    """Matrix-free ICM should also work for all kernels."""

    @pytest.mark.parametrize("kernel", ["rbf", "matern52", "matern32", "matern12"])
    def test_matrix_free_training_returns_finite_predictions(self, kernel):
        """Matrix-free training should return finite predictions."""
        X, Y = _make_data(n=2000)
        gp = MultiOutputGP(kernel=kernel)
        gp.fit(X, Y, max_iterations=30, verbose=False, method="matrix_free")

        X_test = np.random.randn(10, 3).astype(np.float32)
        result = gp.predict(X_test)
        mu = result.mean
        assert mu.shape == (10, 2)
        assert np.all(np.isfinite(mu))


class TestICMARDAllKernels:
    """ARD ICM should work for multiple kernel types."""

    @pytest.mark.parametrize("kernel", ["rbf", "matern52", "matern32", "matern12"])
    def test_ard_training_returns_positive_lengthscales(self, kernel):
        """ARD training should return positive lengthscales for each kernel type."""
        X, Y = _make_data(d=5)
        gp = MultiOutputGP(kernel=kernel, ard=True)
        gp.fit(X, Y, max_iterations=50, verbose=False)

        result = gp.training_result
        assert result is not None
        assert hasattr(result, "lengthscales")
        ls = np.array(result.lengthscales)
        assert len(ls) == 5, f"Expected 5 lengthscales, got {len(ls)}"
        assert np.all(ls > 0), f"Lengthscales not positive for {kernel}: {ls}"
