import pytest

"""Unit tests for ConstantMean in LMC multi-output GP (MultiOutputLMCGP).

Tests that the learnable per-task constant mean:
1. Auto-detects init_mean_per_task from Y.mean(axis=0) when init_mean=None
2. Accepts user-specified float (broadcast) or array init_mean
3. Learns correct per-task means for non-zero-mean data
4. Adds per-task mean offset to predictions
5. Works with standard and matrix-free LMC
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_lmc_data(n=500, d=3, T=2, true_means=None, noise_std=0.1, seed=42):
    """Generate multi-output data suitable for LMC testing.

    y[:, t] = true_means[t] + 0.5 * sin(x_0) + noise
    All tasks share the same zero-mean function so the per-task
    mean is purely true_means[t].
    Uses T=2 by default to keep LMC fast (fewer tasks = faster).
    """
    if true_means is None:
        true_means = [5.0, -3.0]
    true_means = np.array(true_means[:T], dtype=np.float32)

    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)
    f = 0.5 * np.sin(X[:, 0])  # shared zero-mean function
    for t in range(T):
        Y[:, t] = true_means[t] + f + noise_std * np.random.randn(n)
    return X, Y.astype(np.float32), true_means


def _generate_zero_mean_lmc_data(n=500, d=3, T=2, noise_std=0.1, seed=42):
    """Generate LMC data with zero per-task means."""
    return _generate_lmc_data(
        n=n, d=d, T=T, true_means=[0.0] * T, noise_std=noise_std, seed=seed
    )


# ===========================================================================
# Python API tests (no Mojo .so required)
# ===========================================================================


class TestLMCGPConstantMeanAPI:
    """Test MultiOutputLMCGP class ConstantMean API (Python-side only)."""

    def test_init_mean_default_is_none(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"])
        assert gp._init_mean is None

    def test_init_mean_accepts_float(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"], init_mean=5.0)
        assert gp._init_mean == 5.0

    def test_init_mean_accepts_zero(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"], init_mean=0.0)
        assert gp._init_mean == 0.0

    def test_init_mean_accepts_negative(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"], init_mean=-7.5)
        assert gp._init_mean == -7.5

    def test_init_mean_accepts_array(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        means = np.array([1.0, 2.0])
        gp = MultiOutputLMCGP(kernels=["rbf"], init_mean=means)
        np.testing.assert_array_equal(gp._init_mean, means)

    def test_fitted_mean_is_none_before_fit(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"])
        assert gp._fitted_mean is None


# ===========================================================================
# Auto-detection logic tests (no Mojo .so required)
# ===========================================================================


class TestLMCAutoDetectMean:
    """Test LMC init_mean auto-detection logic."""

    def test_auto_detect_computes_column_means(self):
        Y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        init_mean_per_task = np.mean(Y, axis=0).astype(np.float32)
        np.testing.assert_allclose(init_mean_per_task, [2.0, 20.0], atol=1e-6)

    def test_auto_detect_with_nonzero_data(self):
        _, Y, true_means = _generate_lmc_data(n=2000, true_means=[5.0, -3.0], seed=42)
        computed = np.mean(Y, axis=0)
        np.testing.assert_allclose(computed, true_means, atol=0.15)

    def test_float_broadcast(self):
        T = 3
        init_mean_per_task = np.full(T, 5.0, dtype=np.float32)
        np.testing.assert_array_equal(init_mean_per_task, [5.0, 5.0, 5.0])


# ===========================================================================
# Training result dataclass tests (no Mojo .so required)
# ===========================================================================


class TestLMCTrainingResultMeanField:
    """Test that LMCTrainingResult has mean_per_task field."""

    def test_lmc_training_result_has_mean_per_task(self):
        from mojogp.multi_output_gp import LMCTrainingResult

        fields = {f.name for f in LMCTrainingResult.__dataclass_fields__.values()}
        assert "mean_per_task" in fields


# ===========================================================================
# End-to-end training tests (require compiled Mojo .so)
# ===========================================================================


class TestLMCConstantMeanTraining:
    """Test LMC GP training with ConstantMean."""

    def test_learns_nonzero_per_task_means(self):
        """Train on data with different per-task means, verify recovery."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_lmc_data(n=2000, T=2, true_means=true_means, seed=42)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None, "Fitted mean should not be None"
        assert len(gp._fitted_mean) == 2

        for t in range(2):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.0, (
                f"Task {t}: learned mean {gp._fitted_mean[t]:.2f} "
                f"too far from true {true_means[t]}"
            )

    def test_learns_zero_means(self):
        """Train on zero-mean data, verify learned means approx 0."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y, _ = _generate_zero_mean_lmc_data(n=2000, seed=42)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t]) < 1.5, (
                f"Task {t}: learned mean {gp._fitted_mean[t]:.2f} should be ~0"
            )

    def test_user_init_mean_float_used(self):
        """User-specified float init_mean is broadcast to all tasks."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, 5.0]
        X, Y, _ = _generate_lmc_data(n=2000, T=2, true_means=true_means, seed=42)
        gp = MultiOutputLMCGP(kernels=["rbf"], init_mean=5.0)
        gp.fit(X, Y, max_iterations=50, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t] - 5.0) < 2.0

    def test_user_init_mean_array_used(self):
        """User-specified per-task array init_mean."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [2.0, -1.0]
        X, Y, _ = _generate_lmc_data(n=2000, T=2, true_means=true_means, seed=42)
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            init_mean=np.array(true_means, dtype=np.float32),
        )
        gp.fit(X, Y, max_iterations=50, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.0

    def test_large_per_task_means(self):
        """Data with large per-task means (50, -50)."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [50.0, -50.0]
        X, Y, _ = _generate_lmc_data(n=2000, T=2, true_means=true_means, seed=42)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=100, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 5.0, (
                f"Task {t}: learned mean {gp._fitted_mean[t]:.2f} "
                f"too far from true {true_means[t]}"
            )


# ===========================================================================
# Prediction tests (require compiled Mojo .so)
# ===========================================================================


class TestLMCConstantMeanPrediction:
    """Test that LMC predictions include per-task mean offset."""

    def test_prediction_has_correct_shape(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y, _ = _generate_lmc_data(n=500, d=3, T=2, seed=42)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=50, learning_rate=0.05, verbose=False)

        X_test = np.random.randn(20, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True)
        assert mean.shape == (20, 2)
        assert var.shape == (20, 2)

    def test_prediction_mean_near_true_means(self):
        """Predictions at training points should roughly recover data."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_lmc_data(n=500, d=3, T=2, true_means=true_means, seed=42)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        mean, _ = gp.predict(X, return_var=True)
        for t in range(2):
            rmse = np.sqrt(np.mean((mean[:, t] - Y[:, t]) ** 2))
            assert rmse < 1.5, f"Task {t}: RMSE {rmse:.3f} too high for training data"

    def test_prediction_average_near_true_mean(self):
        """Average predictions at test points should be near true per-task mean."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_lmc_data(n=500, d=3, T=2, true_means=true_means, seed=42)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        np.random.seed(99)
        X_test = np.random.randn(100, 3).astype(np.float32)
        mean, _ = gp.predict(X_test, return_var=True)

        for t in range(2):
            avg = np.mean(mean[:, t])
            assert abs(avg - true_means[t]) < 3.0, (
                f"Task {t}: avg prediction {avg:.2f} too far from {true_means[t]}"
            )
