"""
Tests for PivotedCholeskyPrecond across all three pivot methods (greedy, rpcholesky, nystrom).

Tests the preconditioner through the public SingleOutputGP API for all three
methods via gp.fit(preconditioner=...).

Uses n=200 for unit tests (preconditioner math, no GP training) and n=2000 for
training comparisons per AGENTS.md minimum sample size requirements.
"""

import numpy as np
import pytest

from mojogp import SingleOutputGP
from mojogp.kernel import RBF


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------


def _make_unit_data(n=200, d=5, seed=42):
    """Small dataset for preconditioner math tests (no GP training)."""
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    return X


def _make_training_data(n=2000, d=5, noise=0.1, seed=42):
    """Dataset for GP training comparisons (n >= 2000 per AGENTS.md)."""
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = (
        np.sin(X[:, 0])
        + 0.3 * X[:, 1]
        + np.random.randn(n).astype(np.float32) * np.sqrt(noise)
    ).astype(np.float32)
    return X, y


# ===========================================================================
# 1. TestPivotMethodConstruction
# ===========================================================================


class TestPivotMethodConstruction:
    """Verify preconditioner construction produces valid outputs for each method."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("precond", ["greedy", "rpcholesky", "nystrom"])
    def test_construction_finite_via_fit(self, precond):
        """All methods via ExactGP: training completes with finite NLL."""
        X, y = _make_training_data(n=2000)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=20, preconditioner=precond, verbose=False)

        assert gp.training_result is not None, "Training result is None"
        nll = gp.training_result.nll
        assert np.isfinite(nll), f"NLL not finite for precond={precond}: {nll}"
        assert gp.training_result.iterations > 0, "No iterations were run"


# ===========================================================================
# 2. TestPivotMethodsComparable  (all 3 methods via ExactGP)
# ===========================================================================


class TestPivotMethodsComparable:
    """Compare training across all three pivot methods via ExactGP."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("precond", ["greedy", "rpcholesky", "nystrom"])
    def test_all_methods_return_finite_training_result(self, precond):
        """Each method should train with finite NLL over 50 iterations."""
        X, y = _make_training_data(n=2000)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=50, preconditioner=precond, verbose=False)

        tr = gp.training_result
        assert tr is not None, f"No training result for precond={precond}"
        assert np.isfinite(tr.nll), f"Non-finite NLL for precond={precond}: {tr.nll}"
        assert tr.iterations > 0, f"Zero iterations for precond={precond}"

    @pytest.mark.gpu
    def test_methods_comparable_nll(self):
        """All three methods should reach comparable final NLL (within 50%)."""
        X, y = _make_training_data(n=2000)

        nlls = {}
        for precond in ["greedy", "rpcholesky", "nystrom"]:
            # Warm the method once before comparing the measured run. The
            # nystrom path is noticeably noisier on its first solve.
            SingleOutputGP(RBF()).fit(
                X, y, max_iterations=5, preconditioner=precond, verbose=False
            )
            gp = SingleOutputGP(RBF())
            gp.fit(X, y, max_iterations=50, preconditioner=precond, verbose=False)
            nlls[precond] = gp.training_result.nll

        # All NLLs should be finite
        for name, nll in nlls.items():
            assert np.isfinite(nll), f"Non-finite NLL for {name}: {nll}"

        # Compare pairwise. When NLLs land near zero, relative ratios become
        # unstable across sign changes, so compare absolute spread instead.
        nll_values = list(nlls.values())
        min_nll = min(nll_values)
        max_nll = max(nll_values)
        spread = max_nll - min_nll

        if min(abs(v) for v in nll_values) < 1.0:
            assert spread < 0.6, f"NLL spread too wide (spread={spread:.2f}): {nlls}"
        else:
            ratio = max_nll / min_nll
            assert ratio < 1.5, f"NLL spread too wide (ratio={ratio:.2f}): {nlls}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
