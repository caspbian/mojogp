"""Unit tests for LMC multi-output GP training.

Tests that the LMC model trains correctly for various kernel types and
configurations, with both ARD and isotropic lengthscales.

Tests:
1. LMC ARD training runs and NLL decreases
2. Multiple latent types (RBF + Matern52) with ARD
3. Matrix-free LMC training
"""

import pytest
import numpy as np

pytestmark = pytest.mark.integration

from mojogp import Kernel
from mojogp.multi_output_gp import MultiOutputLMCGP


def generate_lmc_data(n=2000, d=3, T=2, seed=42):
    """Generate synthetic multi-output data for LMC tests."""
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)

    # Shared latent: f(x) = sin(2*x_0) + 0.5*x_1
    f = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1]

    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        scale = 1.0 + 0.3 * t
        Y[:, t] = scale * f + 0.1 * np.random.randn(n).astype(np.float32)

    return X, Y


class TestLMCFusedGradientARD:
    """Test that LMC with ARD uses the fused gradient path and trains correctly."""

    def test_lmc_ard_rbf_training_converges(self):
        """LMC with single RBF latent + ARD should reduce NLL during training."""
        X, Y = generate_lmc_data(n=2000, d=3, T=2)

        gp_early = MultiOutputLMCGP(kernels=["rbf"], ard=True)
        result_early = gp_early.fit(
            X,
            Y,
            max_iterations=5,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        gp_late = MultiOutputLMCGP(kernels=["rbf"], ard=True)
        result_late = gp_late.fit(
            X,
            Y,
            max_iterations=40,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        nll_early = result_early.final_nll
        nll_late = result_late.final_nll

        assert np.isfinite(nll_early), f"Early NLL not finite: {nll_early}"
        assert np.isfinite(nll_late), f"Late NLL not finite: {nll_late}"
        assert nll_late < nll_early, (
            f"NLL did not decrease: early={nll_early:.4f}, late={nll_late:.4f}"
        )

    def test_lmc_ard_rbf_params_learned(self):
        """LMC ARD should produce per-dimension lengthscales."""
        X, Y = generate_lmc_data(n=2000, d=3, T=2)

        gp = MultiOutputLMCGP(kernels=["rbf"], ard=True)
        result = gp.fit(
            X,
            Y,
            max_iterations=40,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        # Check lengthscales exist and are reasonable
        ls = result.lengthscales
        assert len(ls) > 0, "No lengthscales returned"
        assert not np.any(np.isnan(ls)), f"NaN in lengthscales: {ls}"
        assert not np.any(np.isinf(ls)), f"Inf in lengthscales: {ls}"

    def test_lmc_ard_matern52_training_converges(self):
        """LMC with Matern52 + ARD should train successfully."""
        X, Y = generate_lmc_data(n=2000, d=3, T=2)

        gp_early = MultiOutputLMCGP(kernels=["matern52"], ard=True)
        result_early = gp_early.fit(
            X,
            Y,
            max_iterations=5,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        gp_late = MultiOutputLMCGP(kernels=["matern52"], ard=True)
        result_late = gp_late.fit(
            X,
            Y,
            max_iterations=40,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        assert result_late.final_nll < result_early.final_nll, (
            f"Matern52 NLL did not decrease: "
            f"early={result_early.final_nll:.4f}, late={result_late.final_nll:.4f}"
        )

    def test_lmc_ard_two_latents_training(self):
        """LMC with 2 latents (RBF + Matern52) + ARD should train."""
        X, Y = generate_lmc_data(n=2000, d=3, T=2)

        gp_early = MultiOutputLMCGP(kernels=["rbf", "matern52"], ard=True)
        result_early = gp_early.fit(
            X,
            Y,
            max_iterations=5,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        gp_late = MultiOutputLMCGP(kernels=["rbf", "matern52"], ard=True)
        result_late = gp_late.fit(
            X,
            Y,
            max_iterations=40,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        assert result_late.final_nll < result_early.final_nll, (
            f"2-latent NLL did not decrease: "
            f"early={result_early.final_nll:.4f}, late={result_late.final_nll:.4f}"
        )

    def test_lmc_ard_matrix_free_save_load_exact_and_love_variance(self, tmp_path):
        """ARD LMC save/load preserves params and backend prediction routes."""
        X, Y = generate_lmc_data(n=2000, d=5, T=2, seed=123)
        X_test = X[:8].copy()

        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"], ard=True)
        result = gp.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=5,
            learning_rate=0.02,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )
        assert result.lengthscales_per_dim is not None
        assert result.lengthscales_per_dim.shape == (2, 5)

        model_path = str(tmp_path / "lmc_ard_round_trip")
        gp.save(model_path)
        loaded = MultiOutputLMCGP.load(model_path)

        assert loaded._result.lengthscales_per_dim is not None
        np.testing.assert_allclose(
            loaded._result.lengthscales_per_dim,
            result.lengthscales_per_dim,
            rtol=0.0,
            atol=0.0,
        )

        mean_before, exact_before = gp.predict(
            X_test, return_var=True, variance_method="exact"
        )
        mean_after, exact_after = loaded.predict(
            X_test, return_var=True, variance_method="exact"
        )
        np.testing.assert_allclose(mean_after, mean_before, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(exact_after, exact_before, rtol=0.0, atol=1e-6)
        assert loaded._backend_predict_info["actual_variance_route"] == "predict_lmc_full_exact"
        assert loaded._backend_predict_info["backend_variance_used"] is True

        mean_before, love_before = gp.predict(
            X_test, return_var=True, variance_method="love"
        )
        mean_after, love_after = loaded.predict(
            X_test, return_var=True, variance_method="love"
        )
        np.testing.assert_allclose(mean_after, mean_before, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(love_after, love_before, rtol=0.0, atol=3e-2)
        assert loaded._backend_predict_info["actual_variance_route"] == "predict_lmc"
        assert loaded._backend_predict_info["backend_variance_used"] is True

    def test_lmc_isotropic_training(self):
        """LMC with isotropic (non-ARD) training should also converge."""
        X, Y = generate_lmc_data(n=2000, d=3, T=2)

        gp_early = MultiOutputLMCGP(kernels=["rbf"], ard=False)
        result_early = gp_early.fit(
            X,
            Y,
            max_iterations=5,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        gp_late = MultiOutputLMCGP(kernels=["rbf"], ard=False)
        result_late = gp_late.fit(
            X,
            Y,
            max_iterations=40,
            learning_rate=0.05,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        assert result_late.final_nll < result_early.final_nll, (
            f"Isotropic NLL did not decrease: "
            f"early={result_early.final_nll:.4f}, late={result_late.final_nll:.4f}"
        )

    def test_lmc_polynomial_keeps_fixed_degree_and_finite_variance(self):
        """Polynomial LMC keeps fixed power and returns finite backend variances."""
        X, Y = generate_lmc_data(n=2000, d=3, T=2, seed=321)
        X_test = X[:8].copy()

        gp = MultiOutputLMCGP(kernels=[Kernel.polynomial(degree=3.0)])
        result = gp.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=5,
            learning_rate=0.01,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
            early_stop_tol=0.0,
        )

        params = np.array(result.params_per_latent[0], dtype=np.float32)
        assert np.isfinite(result.final_nll)
        assert params[0] == pytest.approx(3.0, abs=1e-6)

        for variance_method in ("exact", "love"):
            mean, var = gp.predict(
                X_test, return_var=True, variance_method=variance_method
            )
            assert np.all(np.isfinite(mean))
            assert np.all(np.isfinite(var))
            expected_route = (
                "predict_lmc_full_exact"
                if variance_method == "exact"
                else "predict_lmc"
            )
            assert gp._backend_predict_info["actual_variance_route"] == expected_route
            assert gp._backend_predict_info["backend_variance_used"] is True


class TestLMCFusedGradientMatrixFree:
    """Test matrix-free LMC training."""

    def test_lmc_matrix_free_training(self):
        """Matrix-free LMC should also train correctly."""
        X, Y = generate_lmc_data(n=2000, d=3, T=2)

        gp_early = MultiOutputLMCGP(kernels=["rbf"], ard=True)
        result_early = gp_early.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=5,
            learning_rate=0.02,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        gp_late = MultiOutputLMCGP(kernels=["rbf"], ard=True)
        result_late = gp_late.fit(
            X,
            Y,
            method="matrix_free",
            max_iterations=60,
            learning_rate=0.02,
            initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
        )

        assert np.isfinite(result_early.final_nll), "Early NLL not finite"
        assert np.isfinite(result_late.final_nll), "Late NLL not finite"
        assert result_late.final_nll < result_early.final_nll, (
            f"Matrix-free NLL did not decrease: "
            f"early={result_early.final_nll:.4f}, late={result_late.final_nll:.4f}"
        )
