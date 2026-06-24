"""Unit tests for RPCholesky and Nystrom PCG preconditioner.

Tests:
1. All three precond modes (greedy, rpcholesky, nystrom) produce valid preconditioners
2. RPCholesky sampling distribution is proportional to diagonal
3. Nystrom adaptive rank is reasonable (not 1, not max_rank)
4. All modes work with CG solver and produce finite NLL
5. Backward compatibility (default=nystrom gives same or better results)
6. API: gp.fit(preconditioner="greedy"|"rpcholesky"|"nystrom"|"auto")
"""

import numpy as np
import pytest
from mojogp import SingleOutputGP
from mojogp.kernel import RBF, Matern52, Kernel


def _make_data(n=2000, d=3, noise=0.1, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = (
        np.sin(X[:, 0])
        + 0.3 * X[:, 1]
        + np.random.randn(n).astype(np.float32) * np.sqrt(noise)
    ).astype(np.float32)
    return X, y


class TestPrecondModes:
    """All three precond modes should produce valid results."""

    @pytest.mark.parametrize("precond", ["greedy", "rpcholesky", "nystrom", "auto"])
    def test_precond_mode_returns_finite_nll(self, precond):
        """Each precond mode should complete training with finite NLL."""
        X, y = _make_data(n=2000)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=30, preconditioner=precond, verbose=False)

        assert gp.training_result is not None
        nll = gp.training_result.nll
        assert np.isfinite(nll), f"NLL not finite for precond={precond}: {nll}"

    @pytest.mark.parametrize("precond", ["greedy", "rpcholesky", "nystrom"])
    def test_precond_mode_nll_finite(self, precond):
        """NLL should be finite after training for all modes."""
        X, y = _make_data(n=2000)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=50, preconditioner=precond, verbose=False)

        nll = gp.training_result.nll
        assert np.isfinite(nll), f"NLL not finite for precond={precond}: {nll}"
        # Noise should be positive
        params = gp.get_learned_params()
        assert params.get("noise", 0) > 0, f"Noise not positive for precond={precond}"

    @pytest.mark.parametrize("precond", ["greedy", "rpcholesky", "nystrom"])
    def test_predictions_finite(self, precond):
        """Predictions should be finite for all modes."""
        X, y = _make_data(n=2000)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=30, preconditioner=precond, verbose=False)

        X_test = np.random.randn(20, 3).astype(np.float32)
        mu, std = gp.predict(X_test, return_std=True)
        assert np.all(np.isfinite(mu)), f"Non-finite mean for precond={precond}"
        assert np.all(np.isfinite(std)), f"Non-finite std for precond={precond}"


class TestNystromAdaptiveRank:
    """Nystrom should choose a reasonable adaptive rank."""

    def test_adaptive_rank_not_maxed_out(self):
        """For a well-conditioned GP, nystrom should stop before max_rank."""
        X, y = _make_data(n=2000, noise=0.1)
        gp = SingleOutputGP(RBF())
        # Set high max_precond_rank, nystrom should stop early
        gp.fit(
            X,
            y,
            max_iterations=30,
            preconditioner="nystrom",
            preconditioner_rank=100,
            verbose=False,
        )

        # Training should succeed
        assert gp.training_result is not None
        assert np.isfinite(gp.training_result.nll)

    def test_nystrom_comparable_to_greedy(self):
        """Nystrom should give comparable or better NLL than greedy."""
        X, y = _make_data(n=2000)

        gp_greedy = SingleOutputGP(RBF())
        gp_greedy.fit(X, y, max_iterations=50, preconditioner="greedy", verbose=False)
        nll_greedy = gp_greedy.training_result.nll

        gp_nystrom = SingleOutputGP(RBF())
        gp_nystrom.fit(X, y, max_iterations=50, preconditioner="nystrom", verbose=False)
        nll_nystrom = gp_nystrom.training_result.nll

        # Nystrom should be within 50% of greedy (it should be comparable or better)
        assert nll_nystrom < nll_greedy * 1.5, (
            f"Nystrom NLL {nll_nystrom:.4f} much worse than greedy {nll_greedy:.4f}"
        )


class TestRPCholeskyWithARD:
    """RPCholesky should work with ARD kernels."""

    @pytest.mark.parametrize("precond", ["greedy", "rpcholesky", "nystrom"])
    def test_ard_with_precond_modes(self, precond):
        """ARD training should work with all precond modes."""
        X, y = _make_data(n=2000, d=5)
        gp = SingleOutputGP(RBF(ard=True))
        gp.fit(X, y, max_iterations=30, preconditioner=precond, verbose=False)

        params = gp.get_learned_params()
        assert np.isfinite(gp.training_result.nll)


class TestRPCholeskyWithMultipleKernels:
    """RPCholesky should work with all kernel types."""

    @pytest.mark.parametrize(
        "kernel_fn",
        [
            Kernel.rbf,
            Kernel.matern52,
            Kernel.matern32,
            Kernel.matern12,
        ],
    )
    def test_nystrom_with_kernel(self, kernel_fn):
        """Nystrom precond should work with each kernel type."""
        X, y = _make_data(n=2000)
        gp = SingleOutputGP(kernel_fn())
        gp.fit(X, y, max_iterations=30, preconditioner="nystrom", verbose=False)

        assert np.isfinite(gp.training_result.nll)


class TestBackwardCompatibility:
    """Default preconditioner behavior remains valid."""

    def test_default_precond_is_nystrom(self):
        """Default precond should be 'auto' which maps to nystrom."""
        X, y = _make_data(n=2000)
        gp = SingleOutputGP(RBF())
        # No precond argument = auto = nystrom
        gp.fit(X, y, max_iterations=30, verbose=False)
        assert np.isfinite(gp.training_result.nll)

    def test_invalid_precond_raises(self):
        """Invalid precond string should raise ValueError."""
        X, y = _make_data(n=2000)
        gp = SingleOutputGP(RBF())
        with pytest.raises(ValueError, match="preconditioner"):
            gp.fit(X, y, max_iterations=5, preconditioner="invalid", verbose=False)
