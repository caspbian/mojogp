"""Integration tests for Kronecker preconditioner through multi-output GP training.

Tests preconditioner quality, rank sensitivity, and task count scaling
by training MultiOutputGP and verifying NLL convergence and stability.

All GP training tests use n >= 2000 as required by project policy.
"""

import numpy as np
import pytest

from mojogp.multi_output_gp import MultiOutputGP


# =============================================================================
# Data Generation
# =============================================================================


def generate_correlated_tasks(n: int, d: int, T: int, seed: int = 42):
    """Generate multi-output data with correlated tasks.

    Y[:, t] = sin(X[:, 0] + t * 0.5) + 0.1 * noise_t

    Args:
        n: Number of training points.
        d: Input dimensionality.
        T: Number of tasks.
        seed: Random seed for reproducibility.

    Returns:
        X: [n, d] float32 input data.
        Y: [n, T] float32 output data.
    """
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        noise_t = np.random.randn(n).astype(np.float32)
        Y[:, t] = np.sin(X[:, 0] + t * 0.5) + 0.1 * noise_t
    return X, Y


# =============================================================================
# TestKroneckerPrecondQuality
# =============================================================================


class TestKroneckerPreconditionerAccuracy:
    """Test that the Kronecker preconditioner enables NLL convergence."""

    def test_nll_decreases_T2(self):
        """T=2 RBF: NLL should decrease over 50 iterations with no NaN."""
        n, d, T = 2000, 3, 2
        X, Y = generate_correlated_tasks(n, d, T, seed=42)

        gp = MultiOutputGP(kernel="rbf")
        result = gp.fit(
            X,
            Y,
            max_iterations=50,
            learning_rate=0.1,
            initial_lengthscale=1.0,
            initial_noise=0.1,
            initial_outputscale=1.0,
            early_stop_tol=0.0,  # Disable early stopping
        )

        nll_history = np.array(result.nll_history)

        # No NaN anywhere in the history
        assert not np.any(np.isnan(nll_history)), (
            f"NaN found in NLL history: {nll_history}"
        )
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite values in NLL history: {nll_history}"
        )

        # NLL should decrease: mean of first 5 > mean of last 5
        first_5 = np.mean(nll_history[:5])
        last_5 = np.mean(nll_history[-5:])
        assert last_5 < first_5, (
            f"NLL did not decrease: first 5 mean={first_5:.4f}, "
            f"last 5 mean={last_5:.4f}"
        )

    def test_nll_decreases_T3(self):
        """T=3 RBF: NLL should decrease over 50 iterations with no NaN."""
        n, d, T = 2000, 3, 3
        X, Y = generate_correlated_tasks(n, d, T, seed=42)

        gp = MultiOutputGP(kernel="rbf")
        result = gp.fit(
            X,
            Y,
            max_iterations=50,
            learning_rate=0.1,
            initial_lengthscale=1.0,
            initial_noise=0.1,
            initial_outputscale=1.0,
            early_stop_tol=0.0,
        )

        nll_history = np.array(result.nll_history)

        assert not np.any(np.isnan(nll_history)), (
            f"NaN found in NLL history: {nll_history}"
        )
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite values in NLL history: {nll_history}"
        )

        first_5 = np.mean(nll_history[:5])
        last_5 = np.mean(nll_history[-5:])
        assert last_5 < first_5, (
            f"NLL did not decrease: first 5 mean={first_5:.4f}, "
            f"last 5 mean={last_5:.4f}"
        )


# =============================================================================
# TestKroneckerPrecondRank
# =============================================================================


class TestKroneckerPrecondRank:
    """Test preconditioner behavior across different ranks."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Generate shared data for rank tests (T=2)."""
        n, d, T = 2000, 3, 2
        self.X, self.Y = generate_correlated_tasks(n, d, T, seed=42)
        self.n_iterations = 50

    def _train_with_rank(self, rank: int):
        """Helper: train with a specific preconditioner rank."""
        gp = MultiOutputGP(
            kernel="rbf",
            preconditioner_rank=rank,
            precond_rebuild_threshold=0.5,
        )
        result = gp.fit(
            self.X,
            self.Y,
            max_iterations=self.n_iterations,
            learning_rate=0.1,
            initial_lengthscale=1.0,
            initial_noise=0.1,
            initial_outputscale=1.0,
            early_stop_tol=0.0,
        )
        return result

    def test_rank_5(self):
        """Rank=5 preconditioner trains without NaN."""
        result = self._train_with_rank(5)
        nll_history = np.array(result.nll_history)
        assert not np.any(np.isnan(nll_history)), (
            f"NaN in NLL history with rank=5: {nll_history}"
        )
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite NLL with rank=5: {nll_history}"
        )

    def test_rank_10(self):
        """Rank=10 preconditioner trains without NaN."""
        result = self._train_with_rank(10)
        nll_history = np.array(result.nll_history)
        assert not np.any(np.isnan(nll_history)), (
            f"NaN in NLL history with rank=10: {nll_history}"
        )
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite NLL with rank=10: {nll_history}"
        )

    def test_rank_20(self):
        """Rank=20 preconditioner trains without NaN."""
        result = self._train_with_rank(20)
        nll_history = np.array(result.nll_history)
        assert not np.any(np.isnan(nll_history)), (
            f"NaN in NLL history with rank=20: {nll_history}"
        )
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite NLL with rank=20: {nll_history}"
        )

    def test_higher_rank_not_worse(self):
        """Rank=20 final NLL should be comparable to rank=5 (within 20% absolute)."""
        result_5 = self._train_with_rank(5)
        result_20 = self._train_with_rank(20)

        nll_5 = float(result_5.final_nll)
        nll_20 = float(result_20.final_nll)

        # Skip if either produced NaN (covered by other tests)
        if np.isnan(nll_5) or np.isnan(nll_20):
            pytest.skip("NaN in final NLL; rank NaN tests cover this")

        # For negative NLL (lower is better), check that rank=20 is within
        # 20% absolute range of rank=5. Both should converge to similar values.
        abs_diff = abs(nll_20 - nll_5)
        scale = max(abs(nll_5), abs(nll_20), 1.0)
        rel_diff = abs_diff / scale
        assert rel_diff < 0.20, (
            f"Rank=20 NLL ({nll_20:.4f}) differs from rank=5 NLL ({nll_5:.4f}) "
            f"by {rel_diff:.2%}, exceeds 20%"
        )


# =============================================================================
# TestKroneckerPrecondTaskCount
# =============================================================================


class TestKroneckerPrecondTaskCount:
    """Test Kronecker preconditioner across different task counts."""

    def _train_with_tasks(self, T: int):
        """Helper: train with a specific number of tasks."""
        n, d = 2000, 3
        X, Y = generate_correlated_tasks(n, d, T, seed=42)

        gp = MultiOutputGP(
            kernel="rbf",
            preconditioner_rank=10,
            precond_rebuild_threshold=0.5,
        )
        result = gp.fit(
            X,
            Y,
            max_iterations=50,
            learning_rate=0.1,
            initial_lengthscale=1.0,
            initial_noise=0.1,
            initial_outputscale=1.0,
            early_stop_tol=0.0,
        )
        return result

    def test_T2_trains(self):
        """T=2: training completes with finite NLL."""
        result = self._train_with_tasks(2)
        nll_history = np.array(result.nll_history)
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite NLL with T=2: {nll_history}"
        )

    def test_T3_trains(self):
        """T=3: training completes with finite NLL."""
        result = self._train_with_tasks(3)
        nll_history = np.array(result.nll_history)
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite NLL with T=3: {nll_history}"
        )

    def test_T4_trains(self):
        """T=4: training completes with finite NLL."""
        result = self._train_with_tasks(4)
        nll_history = np.array(result.nll_history)
        assert np.all(np.isfinite(nll_history)), (
            f"Non-finite NLL with T=4: {nll_history}"
        )
