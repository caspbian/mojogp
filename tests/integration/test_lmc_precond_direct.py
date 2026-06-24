"""Integration tests for LMC preconditioner through LMC GP training.

Tests that the LMC preconditioner works correctly across different
preconditioner ranks, latent counts, and rebuild thresholds by verifying
that training converges (NLL decreases, no NaN) under each configuration.

All training tests use n=2000 to ensure numerical stability.
"""

import numpy as np
import pytest


def generate_lmc_data(n=2000, d=3, T=2, seed=42):
    """Generate synthetic multi-output data for LMC preconditioner tests.

    Y[:, t] = sin(X[:, 0] + t * 0.5) + 0.1 * noise_t

    Args:
        n: Number of training points.
        d: Input dimensionality.
        T: Number of tasks.
        seed: Random seed.

    Returns:
        X: [n, d] float32 input array.
        Y: [n, T] float32 target array.
    """
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        Y[:, t] = np.sin(X[:, 0] + t * 0.5) + 0.1 * np.random.randn(n).astype(
            np.float32
        )
    return X, Y


# ============================================================================
# TestLMCPrecondTraining - Basic training convergence
# ============================================================================


class TestLMCPrecondTraining:
    """Test that LMC training with preconditioner converges."""

    def test_nll_decreases(self):
        """R=2, T=2, RBF, 50 iters: NLL should decrease with no NaN."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_lmc_data(n=2000, d=3, T=2, seed=42)

        gp = MultiOutputLMCGP(
            kernels=["rbf", "rbf"],
            num_probes=5,
            max_cg_iterations=100,
            preconditioner_rank=10,
            precond_rebuild_threshold=0.5,
        )
        result = gp.fit(X, Y, max_iterations=50, learning_rate=0.1, verbose=False)

        nll = np.array(result.nll_history)

        # No NaN in NLL history
        assert np.all(np.isfinite(nll)), (
            f"NLL history contains non-finite values: {nll[~np.isfinite(nll)]}"
        )

        # NLL should decrease overall (final < initial, with tolerance for
        # stochastic CG noise)
        assert nll[-1] < nll[0], (
            f"NLL did not decrease: initial={nll[0]:.4f}, final={nll[-1]:.4f}"
        )


# ============================================================================
# TestLMCPrecondRank - Different preconditioner ranks
# ============================================================================


class TestLMCPrecondRank:
    """Test LMC training across different preconditioner ranks."""

    @pytest.fixture
    def data(self):
        return generate_lmc_data(n=2000, d=3, T=2, seed=42)

    def _train_with_rank(self, data, rank):
        """Helper: train LMC with given precond rank, return NLL history."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data
        gp = MultiOutputLMCGP(
            kernels=["rbf", "rbf"],
            num_probes=5,
            max_cg_iterations=100,
            preconditioner_rank=rank,
            precond_rebuild_threshold=0.5,
        )
        result = gp.fit(X, Y, max_iterations=50, learning_rate=0.1, verbose=False)
        return np.array(result.nll_history)

    def test_rank_5(self, data):
        """Preconditioner rank=5 trains without NaN, NLL is finite."""
        nll = self._train_with_rank(data, rank=5)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values at rank=5: {nll[~np.isfinite(nll)]}"
        )

    def test_rank_10(self, data):
        """Preconditioner rank=10 trains without NaN, NLL is finite."""
        nll = self._train_with_rank(data, rank=10)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values at rank=10: {nll[~np.isfinite(nll)]}"
        )

    def test_rank_15(self, data):
        """Preconditioner rank=15 trains without NaN, NLL is finite."""
        nll = self._train_with_rank(data, rank=15)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values at rank=15: {nll[~np.isfinite(nll)]}"
        )


# ============================================================================
# TestLMCPrecondLatentCount - Different numbers of latents (R)
# ============================================================================


class TestLMCPrecondLatentCount:
    """Test LMC training with different latent counts."""

    @pytest.fixture
    def data(self):
        return generate_lmc_data(n=2000, d=3, T=2, seed=42)

    def _train_with_latents(self, data, R):
        """Helper: train LMC with R latents (all RBF), return NLL history."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data
        gp = MultiOutputLMCGP(
            kernels=["rbf"] * R,
            num_probes=5,
            max_cg_iterations=100,
            preconditioner_rank=10,
            precond_rebuild_threshold=0.5,
        )
        result = gp.fit(X, Y, max_iterations=50, learning_rate=0.1, verbose=False)
        return np.array(result.nll_history)

    def test_R1_trains(self, data):
        """R=1, T=2, 50 iters: NLL should be finite throughout."""
        nll = self._train_with_latents(data, R=1)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values at R=1: {nll[~np.isfinite(nll)]}"
        )

    def test_R2_trains(self, data):
        """R=2, T=2, 50 iters: NLL should be finite throughout."""
        nll = self._train_with_latents(data, R=2)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values at R=2: {nll[~np.isfinite(nll)]}"
        )

    def test_R3_trains(self, data):
        """R=3, T=2, 50 iters: NLL should be finite throughout."""
        nll = self._train_with_latents(data, R=3)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values at R=3: {nll[~np.isfinite(nll)]}"
        )


# ============================================================================
# TestLMCPrecondRebuild - Different preconditioner rebuild intervals
# ============================================================================


class TestLMCPrecondRebuildThreshold:
    """Test LMC training with different preconditioner rebuild thresholds."""

    @pytest.fixture
    def data(self):
        return generate_lmc_data(n=2000, d=3, T=2, seed=42)

    def _train_with_rebuild_threshold(self, data, threshold):
        """Helper: train LMC with given rebuild threshold, return NLL history."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data
        gp = MultiOutputLMCGP(
            kernels=["rbf", "rbf"],
            num_probes=5,
            max_cg_iterations=100,
            preconditioner_rank=10,
            precond_rebuild_threshold=threshold,
        )
        result = gp.fit(X, Y, max_iterations=50, learning_rate=0.1, verbose=False)
        return np.array(result.nll_history)

    def test_rebuild_threshold_025(self, data):
        """Low rebuild threshold keeps NLL finite."""
        nll = self._train_with_rebuild_threshold(data, threshold=0.25)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values with rebuild_threshold=0.25: "
            f"{nll[~np.isfinite(nll)]}"
        )

    def test_rebuild_threshold_050(self, data):
        """Default rebuild threshold keeps NLL finite."""
        nll = self._train_with_rebuild_threshold(data, threshold=0.5)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values with rebuild_threshold=0.5: "
            f"{nll[~np.isfinite(nll)]}"
        )

    def test_rebuild_threshold_075(self, data):
        """High rebuild threshold keeps NLL finite."""
        nll = self._train_with_rebuild_threshold(data, threshold=0.75)
        assert np.all(np.isfinite(nll)), (
            f"NLL has non-finite values with rebuild_threshold=0.75: "
            f"{nll[~np.isfinite(nll)]}"
        )
