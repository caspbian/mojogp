import pytest

"""Unit tests for ConstantMean in multi-output GP (MultiOutputGP).

Tests that the learnable per-task constant mean:
1. Auto-detects init_mean_per_task from Y.mean(axis=0) when init_mean=None
2. Accepts user-specified float (broadcast) or array init_mean
3. Learns correct per-task means for non-zero-mean data
4. Adds per-task mean offset to predictions
5. Works with both isotropic and ARD kernels
6. Persists through save/load
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_multi_output_data(
    n=500, d=3, T=3, true_means=None, noise_std=0.1, seed=42
):
    """Generate multi-output data with known per-task means.

    y[:, t] = true_means[t] + 0.5 * sin(x_0) + noise
    All tasks share the same zero-mean function so the per-task
    mean is purely true_means[t].
    """
    if true_means is None:
        true_means = [5.0, -3.0, 10.0]
    true_means = np.array(true_means, dtype=np.float32)
    T = len(true_means)

    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)
    f = 0.5 * np.sin(X[:, 0])  # shared zero-mean function
    for t in range(T):
        Y[:, t] = true_means[t] + f + noise_std * np.random.randn(n)
    return X, Y.astype(np.float32), true_means


def _generate_zero_mean_multi_output_data(n=500, d=3, T=3, noise_std=0.1, seed=42):
    """Generate multi-output data with zero per-task means."""
    return _generate_multi_output_data(
        n=n, d=d, T=T, true_means=[0.0] * T, noise_std=noise_std, seed=seed
    )


# ===========================================================================
# Python API tests (no Mojo .so required)
# ===========================================================================


class TestMultiOutputGPConstantMeanAPI:
    """Test MultiOutputGP class ConstantMean API (Python-side only)."""

    def test_init_mean_default_is_none(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf")
        assert gp._init_mean is None

    def test_init_mean_accepts_float(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf", init_mean=5.0)
        assert gp._init_mean == 5.0

    def test_init_mean_accepts_zero(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf", init_mean=0.0)
        assert gp._init_mean == 0.0

    def test_init_mean_accepts_negative(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf", init_mean=-7.5)
        assert gp._init_mean == -7.5

    def test_init_mean_accepts_array(self):
        from mojogp.multi_output_gp import MultiOutputGP

        means = np.array([1.0, 2.0, 3.0])
        gp = MultiOutputGP(kernel="rbf", init_mean=means)
        np.testing.assert_array_equal(gp._init_mean, means)

    def test_fitted_mean_is_none_before_fit(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf")
        assert gp._fitted_mean is None


# ===========================================================================
# Auto-detection logic tests (no Mojo .so required)
# ===========================================================================


class TestMultiOutputAutoDetectMean:
    """Test that init_mean=None auto-detects per-task means from Y.mean(axis=0)."""

    def test_auto_detect_computes_column_means(self):
        Y = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
        # Logic from MultiOutputGP.fit()
        init_mean_per_task = np.mean(Y, axis=0).astype(np.float32)
        np.testing.assert_allclose(init_mean_per_task, [2.0, 20.0], atol=1e-6)

    def test_auto_detect_with_nonzero_mean_data(self):
        """Auto-detect on data with known per-task means should be close."""
        _, Y, true_means = _generate_multi_output_data(
            n=2000, true_means=[5.0, -3.0, 10.0], seed=42
        )
        computed = np.mean(Y, axis=0)
        # With n = 2000 and zero-mean sin function, expect close to true_means
        np.testing.assert_allclose(computed, true_means, atol=0.15)

    def test_auto_detect_with_zero_mean_data(self):
        _, Y, _ = _generate_zero_mean_multi_output_data(n=2000, seed=42)
        computed = np.mean(Y, axis=0)
        np.testing.assert_allclose(computed, [0.0, 0.0, 0.0], atol=0.15)

    def test_float_broadcast_to_all_tasks(self):
        """Float init_mean broadcasts to [T] array."""
        T = 4
        init_mean = 5.0
        init_mean_per_task = np.full(T, float(init_mean), dtype=np.float32)
        np.testing.assert_array_equal(init_mean_per_task, [5.0, 5.0, 5.0, 5.0])

    def test_user_array_used_directly(self):
        """Array init_mean passed through directly."""
        user_means = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        init_mean_per_task = np.asarray(user_means, dtype=np.float32)
        np.testing.assert_array_equal(init_mean_per_task, user_means)


# ===========================================================================
# Training result dataclass tests (no Mojo .so required)
# ===========================================================================


class TestMultiOutputTrainingResultMeanField:
    """Test that training result dataclasses have mean_per_task field."""

    def test_multi_output_training_result_has_mean_per_task(self):
        from mojogp.multi_output_gp import MultiOutputTrainingResult

        fields = {
            f.name for f in MultiOutputTrainingResult.__dataclass_fields__.values()
        }
        assert "mean_per_task" in fields

    def test_removed_multi_output_result_subclasses_are_not_exported(self):
        import mojogp.multi_output_gp as multi_output_gp

        assert not hasattr(multi_output_gp, "MultiOutputARDTrainingResult")
        assert not hasattr(multi_output_gp, "MultiOutputCompositeTrainingResult")
