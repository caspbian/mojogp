"""Integration tests for the LMC max_tridiag_iterations wrapper setting.

The setting controls how many CG iterations are used for tridiagonal
construction in SLQ log-determinant estimation.
"""

import numpy as np


def generate_lmc_data(n=2000, d=3, T=2, seed=42, noise_std=0.1):
    """Generate synthetic multi-output data for LMC tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        w = rng.standard_normal(d).astype(np.float32)
        Y[:, t] = np.sin(X @ w) + noise_std * rng.standard_normal(n).astype(np.float32)
    return X, Y


def _fit_lmc_with_max_tridiag(max_tridiag_iterations=None):
    from mojogp.multi_output_gp import MultiOutputLMCGP

    X, Y = generate_lmc_data()
    kwargs = {}
    if max_tridiag_iterations is not None:
        kwargs["max_tridiag_iterations"] = max_tridiag_iterations
    gp = MultiOutputLMCGP(
        kernels=["rbf"],
        num_probes=5,
        max_cg_iterations=50,
        cg_tolerance=1.0,
        preconditioner_rank=5,
        **kwargs,
    )
    result = gp.fit(X, Y, max_iterations=10, learning_rate=0.05, verbose=False)
    return gp, result


class TestLMCMaxTridiagIterations:
    """Current wrapper passes max_tridiag_iterations through LMC training."""

    def test_default_max_tridiag_iterations_is_finite(self):
        """Default LMC max_tridiag_iterations trains with finite parameters."""
        gp, result = _fit_lmc_with_max_tridiag()

        assert gp.max_tridiag_iter == 30
        assert result is not None
        assert not np.any(np.isnan(result.lengthscales))

    def test_explicit_max_tridiag_iterations_20_is_finite(self):
        """Explicit max_tridiag_iterations=20 trains with finite parameters."""
        gp, result = _fit_lmc_with_max_tridiag(20)

        assert gp.max_tridiag_iter == 20
        assert result is not None
        assert not np.any(np.isnan(result.lengthscales))

    def test_small_max_tridiag_iterations_is_finite(self):
        """Small max_tridiag_iterations=5 trains with finite parameters."""
        gp, result = _fit_lmc_with_max_tridiag(5)

        assert gp.max_tridiag_iter == 5
        assert result is not None
        assert not np.any(np.isnan(result.lengthscales))

    def test_large_max_tridiag_iterations_is_finite(self):
        """Large max_tridiag_iterations=40 trains with finite parameters."""
        gp, result = _fit_lmc_with_max_tridiag(40)

        assert gp.max_tridiag_iter == 40
        assert result is not None
        assert not np.any(np.isnan(result.lengthscales))
