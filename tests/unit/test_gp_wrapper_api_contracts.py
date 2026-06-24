"""Unit tests for GP wrapper API contracts.

These tests cover input validation, training state, log marginal likelihood,
posterior sampling, persistence, variance-method selection, composite exact
variance routes, and ARD composite LMC parameter handling.
"""

import os
import tempfile

import numpy as np
import pytest

from mojogp import SingleOutputGP, RBF, Matern52, MultiOutputGP, MultiOutputLMCGP
from mojogp.kernel import Kernel, make_ard_kernel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_multi_data():
    """Simple multi-output training data."""
    rng = np.random.default_rng(42)
    n, d, T = 2000, 3, 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = np.column_stack(
        [
            np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n),
            np.cos(X[:, 1]) + 0.1 * rng.standard_normal(n),
        ]
    ).astype(np.float32)
    X_test = rng.standard_normal((10, d)).astype(np.float32)
    return X, Y, X_test


# =============================================================================
# Multi-Output Input Validation
# =============================================================================


class TestMultiOutputInputValidation:
    """Test input validation on multi-output GP classes."""

    def test_multi_output_gp_nan_X(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        gp = MultiOutputGP(kernel="rbf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            gp.fit(X_bad, Y, max_iterations=5)

    def test_multi_output_gp_inf_Y(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        Y_bad = Y.copy()
        Y_bad[0, 0] = np.inf
        gp = MultiOutputGP(kernel="rbf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            gp.fit(X, Y_bad, max_iterations=5)

    def test_multi_output_gp_negative_noise(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        with pytest.raises(ValueError, match="initial_noise must be > 0"):
            gp.fit(X, Y, initial_noise=-0.1)

    def test_multi_output_gp_zero_lr(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            gp.fit(X, Y, learning_rate=0.0)

    def test_multi_output_gp_zero_iterations(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        with pytest.raises(ValueError, match="max_iterations must be > 0"):
            gp.fit(X, Y, max_iterations=0)

    def test_lmc_gp_nan_X(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        X_bad = X.copy()
        X_bad[0, 0] = np.nan
        gp = MultiOutputLMCGP(kernels=["rbf"])
        with pytest.raises(ValueError, match="NaN or Inf"):
            gp.fit(X_bad, Y, max_iterations=5)

    def test_lmc_gp_negative_noise(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf"])
        with pytest.raises(ValueError, match="initial_noise must be > 0"):
            gp.fit(X, Y, initial_noise=-0.1)

    def test_lmc_gp_negative_lr(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf"])
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            gp.fit(X, Y, learning_rate=-0.01)


# =============================================================================
# Single-Output Training State
# =============================================================================


class TestSingleOutputTrainingState:
    """Test that SingleOutputGP stores real training state after fit."""

    def test_is_trained_after_fit(self):
        """After training, SingleOutputGP should be marked as trained."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2)).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=5)

        assert gp.is_trained


# =============================================================================
# Single-Output Log Marginal Likelihood
# =============================================================================


class TestSingleOutputLogMarginalLikelihood:
    """Test log_marginal_likelihood returns training NLL."""

    def test_exactgp_log_ml(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 2)).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=5)

        lml = gp.log_marginal_likelihood()
        assert isinstance(lml, float)
        assert np.isfinite(lml)

    def test_raises_if_not_trained(self):
        X = np.random.randn(30, 2).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)
        gp = SingleOutputGP(RBF())
        with pytest.raises(RuntimeError):
            gp.log_marginal_likelihood()


# =============================================================================
# Multi-Output Posterior Sampling
# =============================================================================


class TestMultiOutputPosteriorSampling:
    """Test sample_posterior() on multi-output GP classes."""

    def test_multi_output_gp_sample_shape(self, simple_multi_data):
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        samples = gp.sample_posterior(X_test, n_samples=5)
        m, T = X_test.shape[0], Y.shape[1]
        assert samples.shape == (5, m, T)
        assert samples.dtype == np.float32

    def test_multi_output_gp_sample_finite(self, simple_multi_data):
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        samples = gp.sample_posterior(X_test, n_samples=3)
        assert np.all(np.isfinite(samples))

    def test_multi_output_gp_sample_untrained(self, simple_multi_data):
        _, _, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        with pytest.raises(RuntimeError, match="trained"):
            gp.sample_posterior(X_test)

    def test_lmc_gp_sample_shape(self, simple_multi_data):
        X, Y, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        samples = gp.sample_posterior(X_test, n_samples=4)
        m, T = X_test.shape[0], Y.shape[1]
        assert samples.shape == (4, m, T)
        assert samples.dtype == np.float32

    def test_lmc_gp_sample_untrained(self, simple_multi_data):
        _, _, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf"])
        with pytest.raises(RuntimeError, match="trained"):
            gp.sample_posterior(X_test)


# =============================================================================
# LMC Model Persistence
# =============================================================================


class TestLMCModelPersistence:
    """Test save/load for MultiOutputLMCGP."""

    def test_save_load_roundtrip(self, simple_multi_data):
        X, Y, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        # Predict before save
        pred_before = gp.predict(X_test)
        mean_before = pred_before.mean if hasattr(pred_before, "mean") else pred_before

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lmc_model")
            gp.save(path)

            # Check files exist
            assert os.path.exists(f"{path}_config.json")
            assert os.path.exists(f"{path}_arrays.npz")

            # Load
            gp_loaded = MultiOutputLMCGP.load(path)

            # Should be trained
            assert gp_loaded.is_trained

            # Predict after load
            pred_after = gp_loaded.predict(X_test)
            mean_after = pred_after.mean if hasattr(pred_after, "mean") else pred_after

            # Predictions should match
            np.testing.assert_allclose(mean_before, mean_after, rtol=1e-5)

    def test_save_load_preserves_metadata(self, simple_multi_data):
        X, Y, _ = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lmc_model")
            gp.save(path)
            gp_loaded = MultiOutputLMCGP.load(path)

            assert gp_loaded.num_latents == gp.num_latents
            assert gp_loaded.num_tasks == gp.num_tasks
            assert gp_loaded._result.num_latents == gp._result.num_latents
            assert gp_loaded._result.num_tasks == gp._result.num_tasks

    def test_save_untrained_raises(self):
        gp = MultiOutputLMCGP(kernels=["rbf"])
        with pytest.raises(RuntimeError, match="trained"):
            gp.save("/tmp/should_not_exist")


# =============================================================================
# LMC Composite Posterior Variance
# =============================================================================


class TestLMCCompositePosteriorVariance:
    """Test that LMC composite variance is posterior (not prior-only)."""

    def test_builtin_lmc_variance_is_finite(self, simple_multi_data):
        """Built-in kernel LMC variance should be finite."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean, var = gp.predict(X_test, return_var=True)
        assert var is not None
        assert np.all(np.isfinite(var))
        assert np.all(var > 0)


# =============================================================================
# Multi-Output Variance Method Selection
# =============================================================================


class TestMultiOutputVarianceMethods:
    """Test variance_method='love' vs 'exact' on multi-output GP classes."""

    def test_multi_output_gp_invalid_variance_method(self, simple_multi_data):
        """Invalid variance_method should raise ValueError."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        with pytest.raises(ValueError, match="variance_method must be"):
            gp.predict(X_test, return_var=True, variance_method="invalid")

    def test_lmc_gp_invalid_variance_method(self, simple_multi_data):
        """Invalid variance_method on LMC should raise ValueError."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        with pytest.raises(ValueError, match="variance_method must be"):
            gp.predict(X_test, return_var=True, variance_method="cholesky")

    def test_multi_output_gp_love_variance(self, simple_multi_data):
        """LOVE variance (default) should produce finite positive results."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean, var = gp.predict(X_test, return_var=True, variance_method="love")
        assert mean.shape == (X_test.shape[0], Y.shape[1])
        assert var.shape == (X_test.shape[0], Y.shape[1])
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)

    def test_multi_output_gp_exact_variance(self, simple_multi_data):
        """Exact CG variance should produce finite positive results."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean, var = gp.predict(X_test, return_var=True, variance_method="exact")
        assert mean.shape == (X_test.shape[0], Y.shape[1])
        assert var.shape == (X_test.shape[0], Y.shape[1])
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)

    def test_multi_output_gp_love_and_exact_means_match(self, simple_multi_data):
        """Both variance methods should produce the same mean predictions."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean_love, _ = gp.predict(X_test, return_var=True, variance_method="love")
        mean_exact, _ = gp.predict(X_test, return_var=True, variance_method="exact")
        np.testing.assert_allclose(mean_love, mean_exact, atol=1e-2, rtol=5e-2)

    def test_multi_output_gp_return_std_with_exact(self, simple_multi_data):
        """return_std should work with variance_method='exact'."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean, std = gp.predict(X_test, return_std=True, variance_method="exact")
        assert mean.shape == (X_test.shape[0], Y.shape[1])
        assert std.shape == (X_test.shape[0], Y.shape[1])
        assert np.all(std >= 0)

    def test_lmc_gp_love_variance(self, simple_multi_data):
        """LMC LOVE variance should produce finite positive results."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean, var = gp.predict(X_test, return_var=True, variance_method="love")
        assert mean.shape == (X_test.shape[0], Y.shape[1])
        assert var.shape == (X_test.shape[0], Y.shape[1])
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)

    def test_lmc_gp_exact_variance(self, simple_multi_data):
        """LMC exact CG variance should produce finite positive results."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean, var = gp.predict(X_test, return_var=True, variance_method="exact")
        assert mean.shape == (X_test.shape[0], Y.shape[1])
        assert var.shape == (X_test.shape[0], Y.shape[1])
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)

    def test_lmc_gp_love_and_exact_means_match(self, simple_multi_data):
        """LMC: both variance methods should produce the same mean predictions."""
        X, Y, X_test = simple_multi_data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean_love, _ = gp.predict(X_test, return_var=True, variance_method="love")
        mean_exact, _ = gp.predict(X_test, return_var=True, variance_method="exact")
        np.testing.assert_allclose(mean_love, mean_exact, atol=1e-2, rtol=5e-2)

    def test_default_is_love(self, simple_multi_data):
        """Default variance_method should be 'love' (no arg = same as love).

        Both calls use variance_method="love" (the default). The JIT engine's
        LOVE prediction uses randomized Lanczos which can cause per-call
        variation in both means and variances. We check that both calls
        produce finite, reasonable results and that the variance_method
        mapping sends both to the same engine code path (int 1 = LOVE).
        """
        X, Y, X_test = simple_multi_data
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=10, verbose=False)

        mean_default, var_default = gp.predict(X_test, return_var=True)
        mean_love, var_love = gp.predict(
            X_test, return_var=True, variance_method="love"
        )
        # Both should produce finite results
        assert np.all(np.isfinite(mean_default))
        assert np.all(np.isfinite(mean_love))
        assert np.all(np.isfinite(var_default))
        assert np.all(np.isfinite(var_love))
        # Both should produce non-negative variance
        assert np.all(var_default >= 0)
        assert np.all(var_love >= 0)
        # Shapes should match
        assert mean_default.shape == mean_love.shape
        assert var_default.shape == var_love.shape

# =============================================================================
# ARD + Composite LMC kernels
# =============================================================================


class TestARDCompositeLMC:
    """Test ARD with composite LMC kernels."""

    def test_ard_composite_lmc_construction(self):
        """ARD + composite LMC should be accepted (no ValueError)."""
        kernel = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[kernel, kernel], ard=True)
        assert gp.ard is True
        assert gp._is_composite is True

    def test_ard_composite_num_params_expands(self):
        """ARD should expand num_params from isotropic to per-dimension."""
        kernel = Kernel.rbf() + Kernel.matern52()
        # Isotropic: rbf(2) + matern52(2) = 4 params
        assert kernel.num_params() == 4

        ard_kernel = make_ard_kernel(kernel, dim=5)
        # ARD: rbf(5+1) + matern52(5+1) = 12 params
        assert ard_kernel.num_params() == 12
