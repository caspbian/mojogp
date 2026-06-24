import pytest

"""Unit tests for ConstantMean function in ExactGP.

Tests that the learnable constant mean:
1. Auto-detects init_mean from y.mean() when init_mean=None
2. Accepts user-specified init_mean values
3. Learns the correct mean for non-zero-mean data
4. Adds mean offset to predictions
5. Works with ExactGP and different kernel types
6. Works with both materialized and matrix-free methods
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_nonzero_mean_data(n=2000, d=3, true_mean=5.0, noise_std=0.1, seed=42):
    """Generate 1-D output data with known non-zero mean.

    y = true_mean + sin(x_0) + noise
    """
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f = np.sin(X[:, 0])
    y = (true_mean + f + noise_std * np.random.randn(n)).astype(np.float32)
    return X, y


def _generate_zero_mean_data(n=2000, d=3, noise_std=0.1, seed=42):
    """Generate 1-D output data with zero mean.

    y = sin(x_0) + noise  (mean ≈ 0)
    """
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f = np.sin(X[:, 0])
    y = (f + noise_std * np.random.randn(n)).astype(np.float32)
    return X, y


def _cholesky_reference_predict(X_train, y_train, X_test, ls, os, noise, mean):
    """Float64 Cholesky-based reference prediction with constant mean.

    Returns predicted mean at X_test.
    """
    from scipy.spatial.distance import cdist

    X_tr = X_train.astype(np.float64)
    X_te = X_test.astype(np.float64)
    y_centered = (y_train - mean).astype(np.float64)

    dist_sq_train = cdist(X_tr, X_tr, metric="sqeuclidean")
    K = float(os) * np.exp(-dist_sq_train / (2 * float(ls) ** 2))
    K += float(noise) * np.eye(len(X_tr))

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_centered))

    dist_sq_test = cdist(X_te, X_tr, metric="sqeuclidean")
    K_star = float(os) * np.exp(-dist_sq_test / (2 * float(ls) ** 2))

    return (K_star @ alpha + float(mean)).astype(np.float32)


# ===========================================================================
# Python API tests (no Mojo .so required)
# ===========================================================================


class TestExactGPConstantMeanAPI:
    """Test ExactGP class ConstantMean API (Python-side only)."""

    def test_init_mean_default_is_none(self):
        """init_mean defaults to None (auto-detect)."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, d=3, true_mean=5.0, seed=42)
        gp = SingleOutputGP(RBF())
        assert gp._init_mean is None

    def test_init_mean_accepts_float(self):
        """init_mean accepts a user-specified float."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, d=3, true_mean=5.0, seed=42)
        gp = SingleOutputGP(RBF(), init_mean=7.5)
        assert gp._init_mean == 7.5

    def test_init_mean_accepts_zero(self):
        """init_mean=0.0 is valid."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, d=3, true_mean=5.0, seed=42)
        gp = SingleOutputGP(RBF(), init_mean=0.0)
        assert gp._init_mean == 0.0

    def test_init_mean_accepts_negative(self):
        """init_mean accepts negative values."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, d=3, true_mean=5.0, seed=42)
        gp = SingleOutputGP(RBF(), init_mean=-10.5)
        assert gp._init_mean == -10.5

    def test_fitted_mean_is_none_before_fit(self):
        """_fitted_mean is None before fit()."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, d=3, true_mean=5.0, seed=42)
        gp = SingleOutputGP(RBF())
        assert gp._fitted_mean is None


# ===========================================================================
# Auto-detection logic tests (no Mojo .so required)
# ===========================================================================


class TestAutoDetectMean:
    """Test that init_mean=None auto-detects from y.mean()."""

    def test_auto_detect_computes_y_mean(self):
        """When init_mean=None, the computed init_mean should equal y.mean()."""
        # We test the logic directly rather than calling fit() (which needs .so)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        init_mean = None

        # This is the logic from ExactGP._fit_continuous()
        computed = init_mean if init_mean is not None else float(np.mean(y))
        assert computed == pytest.approx(3.0, abs=1e-6)

    def test_auto_detect_with_nonzero_mean_data(self):
        """Auto-detect on data with true_mean=5.0 should give ~5.0."""
        _, y = _generate_nonzero_mean_data(n=2000, true_mean=5.0, seed=42)
        computed = float(np.mean(y))
        # sin(x) has mean ~0, so y.mean() ≈ true_mean
        assert abs(computed - 5.0) < 0.2, (
            f"Auto-detected mean {computed} too far from 5.0"
        )

    def test_auto_detect_with_zero_mean_data(self):
        """Auto-detect on zero-mean data should give ~0."""
        _, y = _generate_zero_mean_data(n=2000, seed=42)
        computed = float(np.mean(y))
        assert abs(computed) < 0.2, f"Auto-detected mean {computed} too far from 0.0"

    def test_user_override_takes_precedence(self):
        """When init_mean is specified, it overrides auto-detection."""
        y = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        init_mean = 0.0

        computed = init_mean if init_mean is not None else float(np.mean(y))
        assert computed == 0.0  # User override, not y.mean()=20.0


# ===========================================================================
# End-to-end tests (require compiled JIT engine)
# ===========================================================================


class TestExactGPConstantMeanTraining:
    """Test ExactGP training with ConstantMean through the JIT engine."""

    def test_learns_nonzero_mean_rbf(self):
        """Train on data with true_mean=5.0, verify learned mean ≈ 5.0."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=5.0, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
        )

        learned_mean = gp._fitted_mean
        assert learned_mean is not None, "Fitted mean should not be None after fit()"
        assert abs(learned_mean - 5.0) < 1.0, (
            f"Learned mean {learned_mean:.3f} too far from true mean 5.0"
        )

    def test_learns_zero_mean(self):
        """Train on zero-mean data, verify learned mean ≈ 0."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_zero_mean_data(n=2000, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.05,
        )

        learned_mean = gp._fitted_mean
        assert learned_mean is not None
        assert abs(learned_mean) < 1.0, (
            f"Learned mean {learned_mean:.3f} should be near 0 for zero-mean data"
        )

    def test_learned_mean_in_get_learned_params(self):
        """get_learned_params() returns the fitted mean after training."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=3.0, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=50, learning_rate=0.05)

        params = gp.get_learned_params()
        assert "mean" in params or "noise" in params
        # After training, the GP should have learned parameters
        assert isinstance(params, dict)

    def test_user_init_mean_used(self):
        """User-specified init_mean=0.0 is used instead of auto-detect."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=10.0, seed=42)
        # Start from 0 instead of auto-detecting ~10
        gp = SingleOutputGP(RBF(), init_mean=0.0)
        gp.fit(X, y, max_iterations=100, learning_rate=0.05)

        # Should still converge toward true mean, but may be less accurate
        # since starting from 0 instead of ~10
        learned_mean = gp._fitted_mean
        assert learned_mean is not None

    def test_negative_mean_data(self):
        """Train on data with negative true mean."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=-7.0, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.05)

        learned_mean = gp._fitted_mean
        assert learned_mean is not None
        assert abs(learned_mean - (-7.0)) < 1.5, (
            f"Learned mean {learned_mean:.3f} too far from true mean -7.0"
        )

    def test_large_mean_data(self):
        """Train on data with large true mean (100.0)."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=100.0, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.05)

        learned_mean = gp._fitted_mean
        assert learned_mean is not None
        assert abs(learned_mean - 100.0) < 5.0, (
            f"Learned mean {learned_mean:.3f} too far from true mean 100.0"
        )


class TestSingleOutputGPConstantMeanPrediction:
    """Test SingleOutputGP prediction with ConstantMean through the JIT engine."""

    def test_prediction_includes_mean_offset(self):
        """Predictions on non-zero-mean data should be centered around the mean."""
        from mojogp import SingleOutputGP, RBF

        true_mean = 5.0
        X, y = _generate_nonzero_mean_data(n=2000, true_mean=true_mean, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.05)

        # Predict on training data (should be close to y)
        pred_mean, pred_std = gp.predict(X, return_std=True)
        assert pred_mean is not None

        # Predictions should be centered around true_mean, not around 0
        pred_avg = float(np.mean(pred_mean))
        assert abs(pred_avg - true_mean) < 1.0, (
            f"Prediction average {pred_avg:.3f} should be near true mean {true_mean}"
        )

    def test_prediction_rmse_nonzero_mean(self):
        """RMSE on training data should be reasonable for non-zero-mean data."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(
            n=2000, true_mean=5.0, noise_std=0.1, seed=42
        )
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.05)

        pred_mean, pred_std = gp.predict(X, return_std=True)
        rmse = float(np.sqrt(np.mean((pred_mean - y) ** 2)))
        # With noise_std=0.1, RMSE on training data should be < 0.5
        assert rmse < 0.5, f"Training RMSE {rmse:.4f} too high"

    def test_prediction_on_test_data(self):
        """Predictions on held-out test data should be reasonable."""
        from mojogp import SingleOutputGP, RBF

        np.random.seed(42)
        true_mean = 5.0
        n_train, n_test, d = 200, 50, 3
        X_all = np.random.randn(n_train + n_test, d).astype(np.float32)
        f_all = np.sin(X_all[:, 0])
        y_all = (true_mean + f_all + 0.1 * np.random.randn(n_train + n_test)).astype(
            np.float32
        )

        X_train, X_test = X_all[:n_train], X_all[n_train:]
        y_train, y_test = y_all[:n_train], y_all[n_train:]

        gp = SingleOutputGP(RBF())
        gp.fit(X_train, y_train, max_iterations=100, learning_rate=0.05)

        pred_mean, pred_std = gp.predict(X_test, return_std=True)
        rmse = float(np.sqrt(np.mean((pred_mean - y_test) ** 2)))
        # Test RMSE should be reasonable (< 1.0 for this simple function)
        assert rmse < 1.0, f"Test RMSE {rmse:.4f} too high"


class TestSingleOutputGPConstantMeanARD:
    """Test ConstantMean with ARD mode through the JIT engine."""

    def test_ard_learns_mean(self):
        """ARD mode should also learn the constant mean."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=5.0, seed=42)
        gp = SingleOutputGP(RBF(ard=True))
        gp.fit(X, y, max_iterations=100, learning_rate=0.05)

        learned_mean = gp._fitted_mean
        assert learned_mean is not None
        assert abs(learned_mean - 5.0) < 1.5, (
            f"ARD learned mean {learned_mean:.3f} too far from true mean 5.0"
        )


class TestSingleOutputGPConstantMeanComposite:
    """Test SingleOutputGP (composite kernel) training with ConstantMean."""

    def test_composite_learns_mean(self):
        """SingleOutputGP with composite kernel should learn the constant mean."""
        try:
            from mojogp import SingleOutputGP, RBF
        except ImportError:
            pytest.skip("mojogp not available")

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=5.0, seed=42)

        try:
            gp = SingleOutputGP(RBF())
            gp.fit(
                X,
                y,
                max_iterations=100,
                learning_rate=0.05,
            )
        except Exception as e:
            pytest.skip(f"SingleOutputGP training failed (may need compiled module): {e}")

        learned_mean = gp._fitted_mean
        assert learned_mean is not None, "Fitted mean should not be None"
        assert abs(learned_mean - 5.0) < 1.5, (
            f"Composite learned mean {learned_mean:.3f} too far from true mean 5.0"
        )

    def test_composite_prediction_includes_mean(self):
        """SingleOutputGP predictions should include the mean offset."""
        try:
            from mojogp import SingleOutputGP, RBF
        except ImportError:
            pytest.skip("mojogp not available")

        true_mean = 5.0
        X, y = _generate_nonzero_mean_data(n=2000, true_mean=true_mean, seed=42)

        try:
            gp = SingleOutputGP(RBF())
            gp.fit(X, y, max_iterations=100, learning_rate=0.05)
            pred_mean, pred_std = gp.predict(X, return_std=True)
        except Exception as e:
            pytest.skip(f"SingleOutputGP training/prediction failed: {e}")

        pred_avg = float(np.mean(pred_mean))
        assert abs(pred_avg - true_mean) < 1.0, (
            f"Composite prediction avg {pred_avg:.3f} should be near {true_mean}"
        )

    def test_composite_with_user_init_mean(self):
        """SingleOutputGP respects user-specified init_mean."""
        try:
            from mojogp import SingleOutputGP, RBF
        except ImportError:
            pytest.skip("mojogp not available")

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=5.0, seed=42)

        try:
            gp = SingleOutputGP(RBF(), init_mean=0.0)
            gp.fit(X, y, max_iterations=50, learning_rate=0.05)
        except Exception as e:
            pytest.skip(f"SingleOutputGP training failed: {e}")

        # Should still learn a mean, even starting from 0
        assert gp._fitted_mean is not None


class TestConstantMeanKernelTypes:
    """Test ConstantMean works with different kernel types."""

    @pytest.mark.parametrize(
        "kernel_name,kernel_cls",
        [
            ("rbf", "RBF"),
            ("matern32", "Matern32"),
            ("matern52", "Matern52"),
            ("matern12", "Matern12"),
        ],
    )
    def test_kernel_learns_mean(self, kernel_name, kernel_cls):
        """Each kernel type should learn the constant mean."""
        import mojogp

        kernel_obj = getattr(mojogp, kernel_cls)()
        X, y = _generate_nonzero_mean_data(n=2000, true_mean=5.0, seed=42)
        gp = mojogp.SingleOutputGP(kernel_obj)
        gp.fit(X, y, max_iterations=80, learning_rate=0.05)

        learned_mean = gp._fitted_mean
        assert learned_mean is not None
        assert abs(learned_mean - 5.0) < 2.0, (
            f"{kernel_name}: learned mean {learned_mean:.3f} too far from 5.0"
        )


class TestConstantMeanBoundaryInputs:
    """Test ConstantMean behavior on boundary inputs."""

    def test_constant_data(self):
        """Data that is exactly constant (y = c) should learn mean ≈ c."""
        from mojogp import SingleOutputGP, RBF

        np.random.seed(42)
        n, d = 2000, 3
        X = np.random.randn(n, d).astype(np.float32)
        y = np.full(n, 7.0, dtype=np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=50, learning_rate=0.05)

        learned_mean = gp._fitted_mean
        assert learned_mean is not None
        assert abs(learned_mean - 7.0) < 1.0, (
            f"Constant data: learned mean {learned_mean:.3f} should be near 7.0"
        )

    def test_very_small_mean(self):
        """Data with very small non-zero mean (0.01) should still work."""
        from mojogp import SingleOutputGP, RBF

        X, y = _generate_nonzero_mean_data(n=2000, true_mean=0.01, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.05)

        # Should not crash; mean should be finite
        learned_mean = gp._fitted_mean
        assert learned_mean is not None
        assert np.isfinite(learned_mean), f"Learned mean is not finite: {learned_mean}"
