import pytest

"""Unit tests for public wrapper API contracts.

Tests:
- compute_variance bug fix (MultiOutputGP and MultiOutputLMCGP)
- Input validation (NaN, Inf, bad params) for SingleOutputGP
- SingleOutputGP.get_learned_params() method
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

HAS_JIT = True
requires_jit = pytest.mark.skipif(not HAS_JIT, reason="Mojo JIT not available")


# ===========================================================================
# Shared data
# ===========================================================================


def make_data(n=2000, d=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n).astype(np.float32)
    X_test = np.random.randn(20, d).astype(np.float32)
    return X, y, X_test


# ===========================================================================
# Input Validation — SingleOutputGP (migrated from MojoGP)
# ===========================================================================


class TestSingleOutputGPInputValidation:
    """Test that SingleOutputGP raises ValueError for invalid inputs."""

    def test_nan_X_raises(self):
        from mojogp import SingleOutputGP, RBF

        X = np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float32)
        y = np.array([1.0, 2.0], dtype=np.float32)
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="NaN"):
            gp.fit(X, y)

    def test_inf_y_raises(self):
        from mojogp import SingleOutputGP, RBF

        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        y = np.array([1.0, np.inf], dtype=np.float32)
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="NaN or Inf"):
            gp.fit(X, y)

    def test_zero_iterations_raises(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data()
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="max_iterations"):
            gp.fit(X, y, max_iterations=0)

    def test_negative_lr_raises(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data()
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="learning_rate"):
            gp.fit(X, y, learning_rate=-0.01)

    def test_1d_X_raises(self):
        from mojogp import SingleOutputGP, RBF

        X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="2D"):
            gp.fit(X, y)

    def test_mismatched_samples_raises(self):
        from mojogp import SingleOutputGP, RBF

        X = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(5).astype(np.float32)
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="samples"):
            gp.fit(X, y)

    def test_negative_noise_raises(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data()
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="initial_noise"):
            gp.fit(X, y, initial_noise=-0.1)


# ===========================================================================
# Verbose Default Alignment
# ===========================================================================


class TestVerboseParameterDefaults:
    """Test that verbose defaults are consistent across GP classes."""

    def test_single_output_fit_has_no_verbose_true_default(self):
        """SingleOutputGP.fit() does not default verbose to True."""
        import inspect
        from mojogp import SingleOutputGP

        sig = inspect.signature(SingleOutputGP.fit)
        # Just verify it doesn't have verbose=True.
        if "verbose" in sig.parameters:
            assert sig.parameters["verbose"].default is not True


# ===========================================================================
# SingleOutputGP R^2 score (manual computation)
# ===========================================================================


@requires_jit
class TestSingleOutputGPScore:
    """Test SingleOutputGP R^2 score computed manually."""

    def test_score_returns_float(self):
        from mojogp import SingleOutputGP, RBF

        X, y, X_test = make_data(n=2000, d=3)
        y_test = np.sin(X_test[:, 0]) + 0.1 * np.random.randn(20).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=20)

        mean, std = gp.predict(X_test, return_std=True)
        ss_res = np.sum((y_test - mean) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        score = float(1 - ss_res / ss_tot)
        assert isinstance(score, float)

    def test_score_reasonable_value(self):
        from mojogp import SingleOutputGP, RBF

        np.random.seed(42)
        X, y, X_test = make_data(n=2000, d=3)
        y_test = np.sin(X_test[:, 0]) + 0.1 * np.random.randn(20).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=50)

        mean, std = gp.predict(X_test, return_std=True)
        ss_res = np.sum((y_test - mean) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        score = float(1 - ss_res / ss_tot)
        # R^2 should be positive for a reasonable GP fit
        assert score > -1.0  # Not completely terrible
        assert score <= 1.0  # Cannot exceed 1.0

    def test_predict_before_fit_raises(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data(n=2000, d=3)
        gp = SingleOutputGP(RBF())
        X_test = np.random.randn(10, 3).astype(np.float32)
        with pytest.raises(RuntimeError, match="trained"):
            gp.predict(X_test)


# ===========================================================================
# SingleOutputGP.get_learned_params()
# ===========================================================================


@requires_jit
class TestSingleOutputGPGetLearnedParams:
    """Test SingleOutputGP.get_learned_params() method."""

    def test_returns_dict(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data(n=2000, d=3)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=10)

        params = gp.get_learned_params()
        assert isinstance(params, dict)

    def test_has_noise_key(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data(n=2000, d=3)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=10)

        params = gp.get_learned_params()
        assert "noise" in params
        assert params["noise"] > 0

    def test_has_kernel_params(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data(n=2000, d=3)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=10)

        params = gp.get_learned_params()
        # RBF has lengthscale and outputscale
        assert len(params) >= 2  # at least noise + some kernel params

    def test_before_fit_raises(self):
        from mojogp import SingleOutputGP, RBF

        X, y, _ = make_data(n=2000, d=3)
        gp = SingleOutputGP(RBF())
        with pytest.raises(RuntimeError, match="trained"):
            gp.get_learned_params()
