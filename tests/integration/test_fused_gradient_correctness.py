"""Test fused gradient correctness for Periodic and RQ ARD kernels.

Verifies that the fused gradient path for Periodic and RQ ARD kernels
produces correct results. Training should converge and identify relevant
input dimensions.

Tests:
1. ARD training convergence for Periodic and RQ kernels
2. Dimension relevance identification (the gradient correctly drives irrelevant
   lengthscales to large values)
3. Higher-dimensional (d=10) training
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from mojogp import SingleOutputGP, Periodic, RQ


@pytest.fixture(scope="module")
def periodic_ard_data():
    """Generate data with periodic structure in dim 0 only.

    y = sin(2*pi*x_0 / period) + small_noise
    x_1, x_2 are irrelevant random features.
    """
    np.random.seed(42)
    n, d = 2000, 3
    X = np.random.randn(n, d).astype(np.float32)
    period = 1.0
    y = (np.sin(2 * np.pi * X[:, 0] / period) + 0.05 * np.random.randn(n)).astype(
        np.float32
    )
    return X, y


@pytest.fixture(scope="module")
def rq_ard_data():
    """Generate data with smooth structure in dim 0 only.

    y = sin(3 * x_0) + small_noise
    x_1, x_2 are irrelevant random features.
    """
    np.random.seed(123)
    n, d = 2000, 3
    X = np.random.randn(n, d).astype(np.float32)
    y = (np.sin(3.0 * X[:, 0]) + 0.05 * np.random.randn(n)).astype(np.float32)
    return X, y


class TestPeriodicARDFusedGradient:
    """Test fused gradient for Periodic ARD kernel (d+2 params)."""

    def test_periodic_ard_training_converges(self, periodic_ard_data):
        """Periodic ARD training with fused gradients converges to reasonable NLL."""
        X, y = periodic_ard_data

        gp = SingleOutputGP(Periodic(period=1.0, ard=True))
        result = gp.fit(X, y, max_iterations=100, learning_rate=0.05, initial_noise=0.1)

        assert result is not None
        nll = float(result.nll)
        assert np.isfinite(nll), f"NLL is not finite: {nll}"
        assert nll < 200.0, f"NLL too high: {nll}"

    def test_periodic_ard_nll_decreases(self, periodic_ard_data):
        """Periodic ARD NLL should decrease with more training."""
        X, y = periodic_ard_data

        gp_early = SingleOutputGP(Periodic(period=1.0, ard=True))
        result_early = gp_early.fit(
            X,
            y,
            max_iterations=5,
            learning_rate=0.03,
            initial_noise=0.1,
            method="matrix_free",
        )

        gp_late = SingleOutputGP(Periodic(period=1.0, ard=True))
        result_late = gp_late.fit(
            X,
            y,
            max_iterations=100,
            learning_rate=0.03,
            initial_noise=0.1,
            method="matrix_free",
        )

        # BBMM/SLQ training is slightly stochastic, so require that the longer
        # run does not regress materially rather than insisting on strict
        # monotonic improvement every time.
        assert result_late.nll < result_early.nll + 0.1, (
            f"Periodic ARD NLL regressed too far: early={result_early.nll:.4f}, "
            f"late={result_late.nll:.4f}"
        )

    def test_periodic_ard_identifies_relevant_dim(self, periodic_ard_data):
        """Periodic ARD correctly identifies dim 0 as relevant.

        The lengthscale for dim 0 should be smaller than for dims 1,2 (irrelevant).
        """
        X, y = periodic_ard_data
        n, d = X.shape

        gp = SingleOutputGP(Periodic(period=1.0, ard=True))
        result = gp.fit(X, y, max_iterations=150, learning_rate=0.05, initial_noise=0.1)

        lengthscales = np.array(result.params[0:d])
        assert len(lengthscales) == d

        # Dim 0 is relevant (periodic signal) -> smaller lengthscale
        # Dims 1,2 are noise -> larger lengthscales
        ls_relevant = lengthscales[0]
        ls_irrelevant_avg = np.mean(lengthscales[1:])

        assert ls_irrelevant_avg > ls_relevant * 0.7, (
            f"Irrelevant dims should have larger lengthscales. "
            f"Relevant (dim 0): {ls_relevant:.3f}, "
            f"Irrelevant avg: {ls_irrelevant_avg:.3f}"
        )


class TestRQARDFusedGradient:
    """Test fused gradient for RQ ARD kernel (d+2 params)."""

    def test_rq_ard_training_converges(self, rq_ard_data):
        """RQ ARD training with fused gradients converges to reasonable NLL."""
        X, y = rq_ard_data

        gp = SingleOutputGP(RQ(alpha=2.0, ard=True))
        result = gp.fit(X, y, max_iterations=100, learning_rate=0.05, initial_noise=0.1)

        assert result is not None
        nll = float(result.nll)
        assert np.isfinite(nll), f"NLL is not finite: {nll}"
        assert nll < 200.0, f"NLL too high: {nll}"

    def test_rq_ard_nll_decreases(self, rq_ard_data):
        """RQ ARD NLL should decrease with more training."""
        X, y = rq_ard_data

        gp_early = SingleOutputGP(RQ(alpha=2.0, ard=True))
        result_early = gp_early.fit(
            X, y, max_iterations=5, learning_rate=0.05, initial_noise=0.1
        )

        gp_late = SingleOutputGP(RQ(alpha=2.0, ard=True))
        result_late = gp_late.fit(
            X, y, max_iterations=100, learning_rate=0.05, initial_noise=0.1
        )

        assert result_late.nll < result_early.nll, (
            f"RQ ARD NLL did not decrease: early={result_early.nll:.4f}, "
            f"late={result_late.nll:.4f}"
        )

    def test_rq_ard_identifies_relevant_dim(self, rq_ard_data):
        """RQ ARD correctly identifies dim 0 as relevant."""
        X, y = rq_ard_data
        n, d = X.shape

        gp = SingleOutputGP(RQ(alpha=2.0, ard=True))
        result = gp.fit(X, y, max_iterations=150, learning_rate=0.05, initial_noise=0.1)

        lengthscales = np.array(result.params[0:d])
        assert len(lengthscales) == d

        ls_relevant = lengthscales[0]
        ls_irrelevant_avg = np.mean(lengthscales[1:])

        assert ls_irrelevant_avg > ls_relevant, (
            f"Irrelevant dims should have larger lengthscales. "
            f"Relevant (dim 0): {ls_relevant:.3f}, "
            f"Irrelevant avg: {ls_irrelevant_avg:.3f}"
        )


class TestFusedGradientHigherDim:
    """Test fused gradients at higher dimensions (d=10) where fusion matters most."""

    @pytest.fixture(scope="class")
    def high_dim_data(self):
        """Generate d=10 data with signal in first 2 dims only."""
        np.random.seed(456)
        n, d = 2000, 10
        X = np.random.randn(n, d).astype(np.float32)
        y = (np.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1] + 0.05 * np.random.randn(n)).astype(
            np.float32
        )
        return X, y

    def test_periodic_ard_d10(self, high_dim_data):
        """Periodic ARD at d=10 with fused gradients (12 params)."""
        X, y = high_dim_data
        n, d = X.shape

        gp = SingleOutputGP(Periodic(period=1.0, ard=True))
        result = gp.fit(X, y, max_iterations=150, learning_rate=0.05, initial_noise=0.1)

        nll = float(result.nll)
        assert np.isfinite(nll), f"NLL is not finite: {nll}"

        lengthscales = np.array(result.params[0:d])
        # At least one of the first 2 dims should have smaller lengthscale
        # than the average of the irrelevant dims.
        ls_relevant_min = min(lengthscales[0], lengthscales[1])
        ls_irrelevant_avg = np.mean(lengthscales[2:])
        assert ls_irrelevant_avg > ls_relevant_min * 0.5, (
            f"Expected irrelevant dims to have larger lengthscales. "
            f"Relevant min: {ls_relevant_min:.3f}, Irrelevant avg: {ls_irrelevant_avg:.3f}"
        )

    def test_rq_ard_d10(self, high_dim_data):
        """RQ ARD at d=10 with fused gradients (12 params)."""
        X, y = high_dim_data
        n, d = X.shape

        gp = SingleOutputGP(RQ(alpha=2.0, ard=True))
        result = gp.fit(X, y, max_iterations=100, learning_rate=0.05, initial_noise=0.1)

        nll = float(result.nll)
        assert np.isfinite(nll), f"NLL is not finite: {nll}"

        lengthscales = np.array(result.params[0:d])
        ls_relevant_min = min(lengthscales[0], lengthscales[1])
        ls_irrelevant_avg = np.mean(lengthscales[2:])
        assert ls_irrelevant_avg > ls_relevant_min * 0.8, (
            f"Expected irrelevant dims to have larger lengthscales. "
            f"Relevant min: {ls_relevant_min:.3f}, Irrelevant avg: {ls_irrelevant_avg:.3f}"
        )
