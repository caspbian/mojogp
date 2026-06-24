"""Test LOVE variance for multi-output GP.

Single-output LOVE variance is tested in test_love_variance_comparison.py.
This file tests multi-output-specific variance computation.
"""

import numpy as np
import pytest
from mojogp import MultiOutputGP


def _make_multi_data(n=500, d=3, T=3, noise=0.05, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        Y[:, t] = (t + 1) * 0.3 * f + np.random.randn(n).astype(np.float32) * noise
    return X, Y


class TestMultiOutputLOVEVariance:
    """Multi-output GP LOVE variance should be positive and reasonable."""

    def test_love_variance_positive(self):
        """LOVE variance should be positive for all tasks."""
        X, Y = _make_multi_data()
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=50, verbose=False)

        X_test = np.random.randn(20, 3).astype(np.float32)
        mu, std = gp.predict(X_test, return_std=True, variance_method="love")
        assert mu.shape == (20, 3)
        assert std.shape == (20, 3)
        assert np.all(std >= 0), "LOVE std should be non-negative"
        assert np.all(np.isfinite(std)), "LOVE std should be finite"

    def test_love_variance_consistent_across_tasks(self):
        """Tasks with similar signal should have similar variance."""
        X, Y = _make_multi_data(noise=0.05)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=80, verbose=False)

        X_test = np.random.randn(30, 3).astype(np.float32)
        _, std = gp.predict(X_test, return_std=True, variance_method="love")

        mean_std_per_task = std.mean(axis=0)
        ratio = mean_std_per_task.max() / (mean_std_per_task.min() + 1e-10)
        assert ratio < 100, f"Variance ratio {ratio:.1f} across tasks is too large"

    def test_exact_variance_positive(self):
        """Exact CG variance should also be positive for multi-output."""
        X, Y = _make_multi_data(n=500, T=2)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=50, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mu, std = gp.predict(X_test, return_std=True, variance_method="exact")
        assert np.all(std >= 0), "Exact std should be non-negative"
        assert np.all(np.isfinite(std)), "Exact std should be finite"

    @pytest.mark.parametrize("kernel", ["rbf", "matern52", "matern32"])
    def test_love_mean_matches_no_variance(self, kernel):
        """LOVE prediction mean should match mean-only prediction."""
        X, Y = _make_multi_data(n=500, T=2)
        gp = MultiOutputGP(kernel=kernel)
        gp.fit(X, Y, max_iterations=50, verbose=False)

        X_test = np.random.randn(15, 3).astype(np.float32)
        mu_only = gp.predict(X_test).mean
        mu_love, _ = gp.predict(X_test, return_std=True, variance_method="love")

        np.testing.assert_allclose(
            mu_love,
            mu_only,
            atol=1e-3,
            err_msg=f"LOVE mean should match mean-only for {kernel}",
        )
