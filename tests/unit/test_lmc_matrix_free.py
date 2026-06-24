"""Unit tests for matrix-free LMC multi-output GP training.

Tests that MultiOutputLMCGP with method='matrix_free' produces results
comparable to the materialized variant.
"""

import numpy as np
import pytest

from mojogp.multi_output_gp import MultiOutputLMCGP


def _generate_lmc_data(n=500, d=3, T=2, seed=42):
    """Generate synthetic multi-output data for LMC testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    # Simple correlated outputs
    f1 = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    f2 = 0.8 * f1 + 0.3 * np.cos(X[:, 2])
    Y = np.column_stack(
        [
            f1 + 0.1 * rng.randn(n),
            f2 + 0.1 * rng.randn(n),
        ]
    ).astype(np.float32)
    return X, Y


class TestLMCMatrixFreeInit:
    """Test MultiOutputLMCGP fit-level method parameter."""

    def test_default_method_is_materialized(self):
        gp = MultiOutputLMCGP(kernels=["rbf"])
        assert gp.method == "materialized"

    def test_fit_exposes_method(self):
        import inspect

        sig = inspect.signature(MultiOutputLMCGP.fit)
        assert "method" in sig.parameters

    def test_invalid_method_raises(self):
        X, Y = _generate_lmc_data(n=8)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        with pytest.raises(ValueError, match="method must be"):
            gp.fit(X, Y, method="invalid")

    def test_fit_method_aliases_resolve_to_canonical_routes(self, monkeypatch):
        X, Y = _generate_lmc_data(n=8)
        gp = MultiOutputLMCGP(kernels=["rbf"])

        monkeypatch.setattr(gp, "_fit_composite", lambda *args, **kwargs: "trained")

        assert gp.fit(X, Y, method="mf") == "trained"
        assert gp.method == "matrix_free"

        assert gp.fit(X, Y, method="mat") == "trained"
        assert gp.method == "materialized"

    def test_matrix_free_with_ard(self):
        gp = MultiOutputLMCGP(kernels=["rbf"], ard=True)
        assert gp.method == "materialized"
        assert gp.ard is True

    def test_matrix_free_with_multiple_latents(self):
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        assert gp.num_latents == 2
        assert gp.method == "materialized"


class TestLMCMatrixFreeTraining:
    """Test that matrix-free LMC training runs and produces valid results."""

    @pytest.fixture
    def data(self):
        return _generate_lmc_data(n = 500, d=3, T=2)

    def test_matrix_free_training_returns_finite_nll(self, data):
        """Matrix-free LMC training should complete without errors."""
        X, Y = data
        gp = MultiOutputLMCGP(kernels=["rbf"])
        result = gp.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=20,
            learning_rate=0.05,
            verbose=False,
        )
        assert result is not None
        assert result.iterations > 0
        assert np.isfinite(result.final_nll)

    def test_training_with_two_latents(self, data):
        """Matrix-free LMC with R=2 latents should work."""
        X, Y = data
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        result = gp.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=20,
            learning_rate=0.05,
            verbose=False,
        )
        assert result is not None
        assert result.num_latents == 2
        assert result.lengthscales.shape[0] >= 2
        assert result.A_matrices.shape == (2, 2, 2)

    def test_training_with_ard(self, data):
        """Matrix-free LMC with ARD should work."""
        X, Y = data
        d = X.shape[1]
        gp = MultiOutputLMCGP(kernels=["rbf"], ard=True)
        result = gp.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=20,
            learning_rate=0.05,
            verbose=False,
        )
        assert result is not None
        assert result.use_ard is True
        # ARD: lengthscales should be [R*d]
        assert result.lengthscales.shape[0] == d

    def test_prediction_after_training(self, data):
        """Prediction should work after matrix-free LMC training."""
        X, Y = data
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=20,
            learning_rate=0.05,
            verbose=False,
        )
        X_test = X[:10]
        pred = gp.predict(X_test)
        assert pred.mean.shape == (10, 2)
        assert np.all(np.isfinite(pred.mean))

    def test_nll_decreases(self, data):
        """NLL should generally decrease during training."""
        X, Y = data
        gp = MultiOutputLMCGP(kernels=["rbf"])
        result = gp.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=50,
            learning_rate=0.05,
            verbose=False,
            early_stop_tol=0.0,
        )
        nll_history = result.nll_history
        # First NLL should be higher than last (on average)
        assert nll_history[-1] < nll_history[0], (
            f"NLL did not decrease: first={nll_history[0]:.4f}, last={nll_history[-1]:.4f}"
        )


class TestLMCMatrixFreeVsMaterialized:
    """Compare matrix-free and materialized LMC results."""

    @pytest.fixture
    def data(self):
        return _generate_lmc_data(n = 500, d=3, T=2, seed=123)

    def test_comparable_nll(self, data):
        """Matrix-free and materialized should achieve similar NLL."""
        X, Y = data
        n_iter = 50

        gp_mat = MultiOutputLMCGP(kernels=["rbf"])
        result_mat = gp_mat.fit(
            X,
            Y,
            method="materialized",
            max_iterations=n_iter,
            learning_rate=0.05,
            verbose=False,
            early_stop_tol=0.0,
        )

        gp_mf = MultiOutputLMCGP(kernels=["rbf"])
        result_mf = gp_mf.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=n_iter,
            learning_rate=0.05,
            verbose=False,
            early_stop_tol=0.0,
        )

        # Both should converge to similar NLL (within 50% relative)
        nll_mat = result_mat.final_nll
        nll_mf = result_mf.final_nll
        rel_diff = abs(nll_mat - nll_mf) / (abs(nll_mat) + 1e-8)
        assert rel_diff < 0.5, (
            f"NLL too different: materialized={nll_mat:.4f}, matrix_free={nll_mf:.4f}, "
            f"rel_diff={rel_diff:.4f}"
        )

    def test_comparable_predictions(self, data):
        """Matrix-free and materialized should produce similar predictions."""
        X, Y = data
        n_iter = 50

        gp_mat = MultiOutputLMCGP(kernels=["rbf"])
        gp_mat.fit(
            X,
            Y,
            method="materialized",
            max_iterations=n_iter,
            learning_rate=0.05,
            verbose=False,
            early_stop_tol=0.0,
        )

        gp_mf = MultiOutputLMCGP(kernels=["rbf"])
        gp_mf.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=n_iter,
            learning_rate=0.05,
            verbose=False,
            early_stop_tol=0.0,
        )

        X_test = X[:10]
        pred_mat = gp_mat.predict(X_test)
        pred_mf = gp_mf.predict(X_test)

        # Predictions should be in the same ballpark
        # (not identical because CG is stochastic with random probes)
        mean_diff = np.mean(np.abs(pred_mat.mean - pred_mf.mean))
        mean_scale = np.mean(np.abs(pred_mat.mean)) + 1e-8
        rel_mean_diff = mean_diff / mean_scale
        assert rel_mean_diff < 1.0, (
            f"Predictions too different: rel_mean_diff={rel_mean_diff:.4f}"
        )
