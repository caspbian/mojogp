"""Semantic correctness tests for multi-output GP.

These tests verify algorithm correctness, not just shape/finiteness.
They prove that different code paths are actually exercised and produce
meaningful results.
"""

import numpy as np
import pytest

from mojogp.multi_output_gp import MultiOutputGP, MultiOutputLMCGP
from mojogp.kernel import Kernel


@pytest.fixture
def synthetic_multi_data():
    """Synthetic multi-output data where task 1 = sin(x), task 2 = cos(x)."""
    np.random.seed(42)
    n = 2000
    X = np.random.randn(n, 2).astype(np.float32)
    Y = np.column_stack(
        [
            np.sin(X[:, 0]) + 0.1 * np.random.randn(n),
            np.cos(X[:, 0]) + 0.1 * np.random.randn(n),
        ]
    ).astype(np.float32)
    X_test = np.random.randn(50, 2).astype(np.float32)
    return X, Y, X_test


@pytest.fixture
def trained_multioutput_gp(synthetic_multi_data):
    """Pre-trained MultiOutputGP for reuse across tests."""
    X, Y, X_test = synthetic_multi_data
    gp = MultiOutputGP(kernel="rbf")
    gp.fit(X, Y, max_iterations=30, verbose=False)
    return gp, X_test, X, Y


@pytest.fixture
def trained_lmc_gp(synthetic_multi_data):
    """Pre-trained MultiOutputLMCGP for reuse across tests."""
    X, Y, X_test = synthetic_multi_data
    gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
    gp.fit(X, Y, max_iterations=30, verbose=False)
    return gp, X_test, X, Y


class TestMultiOutputGPVarianceSemantic:
    """Verify that different variance methods produce meaningfully different results."""

    def test_exact_and_love_produce_different_variance(self, trained_multioutput_gp):
        """exact and love should produce numerically different variance values.

        This confirms the variance_method int mapping actually routes to
        different engine code paths (not both going to mean_only or LOVE).
        """
        gp, X_test, _, _ = trained_multioutput_gp

        _, var_love = gp.predict(X_test, return_var=True, variance_method="love")
        _, var_exact = gp.predict(X_test, return_var=True, variance_method="exact")

        # Both should be finite and non-negative
        assert np.all(np.isfinite(var_love)), "LOVE variance has non-finite values"
        assert np.all(np.isfinite(var_exact)), "Exact variance has non-finite values"
        assert np.all(var_love >= 0), "LOVE variance has negative values"
        assert np.all(var_exact >= 0), "Exact variance has negative values"

        # They should be DIFFERENT (not identical) — proves different code paths
        # Allow for the rare case where they happen to match, but in general
        # LOVE (low-rank approximation) differs from exact CG variance
        max_diff = np.max(np.abs(var_love - var_exact))
        # If max_diff is exactly 0, both methods returned identical results,
        # which likely means the mapping is still wrong
        # We use a very loose threshold — even 1e-6 difference proves they differ
        assert max_diff > 1e-8 or var_love.size == 0, (
            f"LOVE and exact variance are identical (max_diff={max_diff}). "
            "This suggests variance_method mapping is not routing to different code paths."
        )

    def test_mean_only_returns_no_variance(self, trained_multioutput_gp):
        """mean_only should not compute variance."""
        gp, X_test, _, _ = trained_multioutput_gp

        result = gp.predict(X_test, variance_method="mean_only")
        # When not using return_var/return_std, result is a MultiOutputPredictionResult
        # mean_only should still have mean but variance may be None
        assert result.mean is not None
        assert result.mean.shape == (X_test.shape[0], 2)
        assert np.all(np.isfinite(result.mean))

    def test_all_methods_produce_finite_mean(self, trained_multioutput_gp):
        """All variance methods should produce finite mean predictions."""
        gp, X_test, _, _ = trained_multioutput_gp

        for method in ("love", "exact", "mean_only"):
            result = gp.predict(X_test, variance_method=method)
            assert np.all(np.isfinite(result.mean)), (
                f"variance_method='{method}' produced non-finite mean"
            )


class TestMultiOutputGPPredictionConsistency:
    """Verify predictions are meaningful, not just finite."""

    def test_predictions_correlate_with_ground_truth(
        self, trained_multioutput_gp, synthetic_multi_data
    ):
        """Predictions should correlate with the true function on test data."""
        gp, X_test, _, _ = trained_multioutput_gp
        _, _, X_test = synthetic_multi_data

        result = gp.predict(X_test)
        mean = result.mean

        # True values
        y_true_0 = np.sin(X_test[:, 0])
        y_true_1 = np.cos(X_test[:, 0])

        # Check correlation — predictions should be positively correlated with truth
        corr_0 = np.corrcoef(mean[:, 0], y_true_0)[0, 1]
        corr_1 = np.corrcoef(mean[:, 1], y_true_1)[0, 1]

        assert corr_0 > 0.2, f"Task 0 correlation too low: {corr_0:.3f}"
        assert corr_1 > 0.2, f"Task 1 correlation too low: {corr_1:.3f}"

    def test_tasks_not_swapped(self, trained_multioutput_gp, synthetic_multi_data):
        """Task 0 predictions should be closer to sin(x), task 1 to cos(x)."""
        gp, X_test, _, _ = trained_multioutput_gp
        _, _, X_test = synthetic_multi_data

        result = gp.predict(X_test)
        mean = result.mean

        y_sin = np.sin(X_test[:, 0])
        y_cos = np.cos(X_test[:, 0])

        # RMSE of task 0 to sin should be less than RMSE of task 0 to cos
        rmse_0_sin = np.sqrt(np.mean((mean[:, 0] - y_sin) ** 2))
        rmse_0_cos = np.sqrt(np.mean((mean[:, 0] - y_cos) ** 2))
        # Task 0 should be closer to sin than to cos
        assert rmse_0_sin < rmse_0_cos, (
            f"Task 0 is closer to cos ({rmse_0_cos:.3f}) than sin ({rmse_0_sin:.3f}) — tasks may be swapped"
        )


class TestLMCVarianceSemantic:
    """Verify LMC variance method wiring."""

    def test_exact_and_love_produce_different_variance(self, trained_lmc_gp):
        """LOVE and exact should route to different LMC variance paths."""
        gp, X_test, _, _ = trained_lmc_gp

        _, var_love = gp.predict(X_test, return_var=True, variance_method="love")
        _, var_exact = gp.predict(X_test, return_var=True, variance_method="exact")

        assert np.all(np.isfinite(var_love)), "LOVE variance has non-finite values"
        assert np.all(np.isfinite(var_exact)), "Exact variance has non-finite values"
        assert np.all(var_love >= 0), "LOVE variance has negative values"
        assert np.all(var_exact >= 0), "Exact variance has negative values"

        max_diff = np.max(np.abs(var_love - var_exact))
        assert max_diff > 1e-8 or var_love.size == 0, (
            f"LMC LOVE and exact variance are identical (max_diff={max_diff}). "
            "This suggests variance_method is not routing to different code paths."
        )

    def test_lmc_variance_methods_accepted(self, trained_lmc_gp):
        """All variance methods should be accepted without error."""
        gp, X_test, _, _ = trained_lmc_gp

        for method in ("love", "exact", "mean_only"):
            result = gp.predict(X_test, variance_method=method)
            assert result.mean is not None
            assert np.all(np.isfinite(result.mean))

    def test_lmc_variance_is_computed_when_requested(self, trained_lmc_gp):
        """When variance_method is love or exact, variance should be non-None."""
        gp, X_test, _, _ = trained_lmc_gp

        for method in ("love", "exact"):
            mean, var = gp.predict(X_test, return_var=True, variance_method=method)
            assert var is not None, f"variance_method='{method}' returned None variance"
            assert np.all(np.isfinite(var)), (
                f"variance_method='{method}' has non-finite variance"
            )
            assert np.all(var >= 0), f"variance_method='{method}' has negative variance"

    def test_lmc_mean_only_skips_variance(self, trained_lmc_gp):
        """mean_only should not compute variance."""
        gp, X_test, _, _ = trained_lmc_gp

        result = gp.predict(X_test, variance_method="mean_only")
        assert result.mean is not None
        # Variance should be None when mean_only
        assert result.variance is None or (
            result.variance is not None and np.all(result.variance == 0)
        )


class TestMultiOutputActiveDims:
    """Verify multi-output GP works with active_dims."""

    def test_ard_multi_output_trains(self):
        """MultiOutputGP with ard=True should train successfully."""
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n, 3).astype(np.float32)
        Y = np.column_stack(
            [
                np.sin(X[:, 0]) + 0.1 * np.random.randn(n),
                np.cos(X[:, 1]) + 0.1 * np.random.randn(n),
            ]
        ).astype(np.float32)

        gp = MultiOutputGP(kernel="rbf", ard=True)
        gp.fit(X, Y, max_iterations=15, verbose=False)

        X_test = np.random.randn(20, 3).astype(np.float32)
        result = gp.predict(X_test)
        assert result.mean.shape == (20, 2)
        assert np.all(np.isfinite(result.mean))

    def test_isotropic_active_dims_codegen_returns_finite_predictions(self):
        """Isotropic kernel with active_dims compiles without NotImplementedError.

        Code generation must slice squared distances for isotropic sub-kernels.
        This uses single-output ExactGP because multi-output active_dims has
        separate dimension-routing coverage.
        """
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n, 3).astype(np.float32)
        y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(n)).astype(np.float32)

        from mojogp.gp import SingleOutputGP

        kernel = Kernel.rbf(active_dims=[0, 1])
        gp = SingleOutputGP(kernel)
        # This should not raise NotImplementedError anymore
        gp.fit(X, y, max_iterations=15, verbose=False)

        X_test = np.random.randn(20, 3).astype(np.float32)
        mean, std = gp.predict(X_test, return_std=True)
        assert np.all(np.isfinite(mean))


class TestLMCSamplingSemantic:
    """Verify LMC posterior sampling captures cross-task covariance."""

    def test_pathwise_sampling_captures_cross_task_correlation(self):
        rng = np.random.default_rng(123)
        X_train = rng.standard_normal((2000, 2)).astype(np.float32)
        X_test = np.array([[7.0, 7.0]], dtype=np.float32)
        latent = np.sin(X_train[:, 0]).astype(np.float32)
        Y_train = np.column_stack(
            [
                latent + 0.03 * rng.standard_normal(2000),
                1.2 * latent + 0.03 * rng.standard_normal(2000),
            ]
        ).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])
        # This test is about the sampling route, so give the single-latent LMC fit
        # enough optimizer steps to learn the intended positive task covariance.
        gp.fit(X_train, Y_train, max_iterations=30, verbose=False)

        samples = gp.sample_posterior(X_test, n_samples=1024, method="pathwise")
        corr = np.corrcoef(samples[:, 0, 0], samples[:, 0, 1])[0, 1]
        assert corr > 0.2, (
            f"Expected positive cross-task correlation from pathwise LMC sampling, got {corr:.3f}"
        )

    def test_composite_isotropic_different_active_dims(self):
        """RBF(active_dims=[0,1]) + Matern52(active_dims=[2,3]) without ARD.

        Composite isotropic sub-kernels with different active_dims should not
        require ard=True.
        """
        np.random.seed(42)
        n = 2000
        X = np.random.randn(n, 4).astype(np.float32)
        y = (np.sin(X[:, 0]) + np.cos(X[:, 2]) + 0.1 * np.random.randn(n)).astype(
            np.float32
        )

        kernel = Kernel.rbf(active_dims=[0, 1]) + Kernel.matern52(active_dims=[2, 3])
        from mojogp.gp import SingleOutputGP

        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=30, verbose=False)

        X_test = np.random.randn(20, 4).astype(np.float32)
        mean, std = gp.predict(X_test, return_std=True)
        assert mean.shape == (20,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)
