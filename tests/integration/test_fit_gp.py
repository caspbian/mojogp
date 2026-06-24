"""Test the fit_gp() convenience function.

fit_gp() is a one-liner that creates a SingleOutputGP and calls fit().
Currently has zero test coverage.
"""

import numpy as np
import pytest
from mojogp.gp import fit_gp
from mojogp.kernel import RBF, Matern52, Kernel


def _make_data(n=2000, d=2, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = (np.sin(X[:, 0]) + np.random.randn(n).astype(np.float32) * 0.1).astype(
        np.float32
    )
    return X, y


class TestFitGP:
    """Test fit_gp() convenience function."""

    def test_default_kernel(self):
        """fit_gp with no kernel should use RBF."""
        X, y = _make_data()
        gp = fit_gp(X, y, max_iterations=20, verbose=False)
        assert gp.is_trained
        mu, std = gp.predict(X[:5], return_std=True)
        assert len(mu) == 5
        assert np.all(np.isfinite(mu))

    def test_explicit_kernel(self):
        """fit_gp with explicit kernel should use it."""
        X, y = _make_data()
        gp = fit_gp(X, y, kernel=Matern52(), max_iterations=20, verbose=False)
        assert gp.is_trained
        mu, _ = gp.predict(X[:5], return_std=True)
        assert np.all(np.isfinite(mu))

    def test_returns_single_output_gp(self):
        """fit_gp should return a SingleOutputGP instance."""
        from mojogp import SingleOutputGP

        X, y = _make_data()
        gp = fit_gp(X, y, max_iterations=10, verbose=False)
        assert isinstance(gp, SingleOutputGP)

    def test_kwargs_forwarded(self):
        """fit_gp should forward kwargs to fit()."""
        X, y = _make_data()
        gp = fit_gp(X, y, max_iterations=5, learning_rate=0.1, verbose=False)
        assert gp.is_trained

    def test_composite_kernel(self):
        """fit_gp should work with composite kernels."""
        X, y = _make_data()
        gp = fit_gp(X, y, kernel=RBF() + Matern52(), max_iterations=20, verbose=False)
        assert gp.is_trained
        mu, _ = gp.predict(X[:5], return_std=True)
        assert np.all(np.isfinite(mu))
