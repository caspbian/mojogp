"""Integration tests for LMC multi-output GP.

Tests the MultiOutputLMCGP class including:
- R=1 LMC should match ICM behavior
- R>1 with heterogeneous kernels
- Prediction mean and variance shapes
- Training convergence
- Score method
"""

import numpy as np
import pytest
import sys
import os
import gc
import time

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from tests.shared.subprocess_harness import run_isolated_case

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


MODULE = "tests.integration.run_lmc_integration_case"


def _cleanup_gpu_state():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _run_mixed_lmc_case(case, method):
    _cleanup_gpu_state()
    time.sleep(0.05)
    return run_isolated_case(
        module=MODULE,
        payload={"case": case, "method": method},
        timeout=600,
        description=f"Runs LMC integration case {case}/{method}",
    )


def generate_multi_output_data(n=500, d=3, T=2, seed=42, noise_std=0.1):
    """Generate synthetic multi-output data with known structure.

    Creates data where:
    - Task 0: sin(x_0) + 0.5 * x_1
    - Task 1: cos(x_0) - 0.3 * x_1
    - Task t (t >= 2): 0.5 * sin(x_0 + t) + noise
    """
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)

    for t in range(T):
        if t == 0:
            Y[:, t] = np.sin(X[:, 0]) + 0.5 * X[:, 1]
        elif t == 1:
            Y[:, t] = np.cos(X[:, 0]) - 0.3 * X[:, 1]
        else:
            Y[:, t] = 0.5 * np.sin(X[:, 0] + t)
        Y[:, t] += noise_std * np.random.randn(n).astype(np.float32)

    return X, Y


def generate_mixed_multi_output_data(
    n=2000, d_cont=2, T=2, levels=3, seed=123, noise_std=0.05
):
    """Generate mixed continuous+categorical multi-output data."""
    rng = np.random.default_rng(seed)
    X_cont = rng.standard_normal((n, d_cont)).astype(np.float32)
    cat = rng.integers(0, levels, size=(n, 1), dtype=np.int32)
    X = np.concatenate([X_cont, cat.astype(np.float32)], axis=1)

    cat_effect_table = np.linspace(-0.6, 0.6, levels, dtype=np.float32)
    cat_effect = cat_effect_table[cat[:, 0]]

    Y = np.zeros((n, T), dtype=np.float32)
    Y[:, 0] = (
        np.sin(X_cont[:, 0])
        + 0.35 * X_cont[:, 1]
        + 0.8 * cat_effect
        + noise_std * rng.standard_normal(n).astype(np.float32)
    )
    if T > 1:
        Y[:, 1] = (
            0.7 * np.cos(X_cont[:, 0])
            - 0.2 * X_cont[:, 1]
            + 0.5 * cat_effect
            + noise_std * rng.standard_normal(n).astype(np.float32)
        )
    for t in range(2, T):
        Y[:, t] = (
            0.4 * np.sin(X_cont[:, 0] + t)
            + (0.3 + 0.1 * t) * cat_effect
            + noise_std * rng.standard_normal(n).astype(np.float32)
        )

    return X.astype(np.float32), Y.astype(np.float32)


# ============================================================================
# LMC Initialization Tests
# ============================================================================


class TestLMCInit:
    """Test MultiOutputLMCGP initialization and validation."""

    def test_single_kernel_init(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP
        from mojogp.kernel import KernelNode

        gp = MultiOutputLMCGP(kernels=["rbf"])
        assert gp.num_latents == 1
        assert len(gp.kernels) == 1
        assert isinstance(gp.kernels[0], KernelNode)
        assert gp.is_trained is False

    def test_multi_kernel_init(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP
        from mojogp.kernel import KernelNode

        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        assert gp.num_latents == 2
        assert len(gp.kernels) == 2
        assert all(isinstance(k, KernelNode) for k in gp.kernels)

    def test_three_kernels_init(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf", "matern32", "matern52"])
        assert gp.num_latents == 3

    def test_invalid_kernel_raises(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        with pytest.raises(ValueError, match="Unknown kernel"):
            MultiOutputLMCGP(kernels=["rbf", "invalid_kernel"])

    def test_matrix_free_composite_init(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        assert gp.method == "materialized"

    def test_custom_hyperparams(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=100,
            cg_tolerance=0.5,
            preconditioner_rank=10,
        )
        assert gp.num_probes == 5
        assert gp.max_cg_iter == 100

    def test_repr_untrained(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        repr_str = repr(gp)
        assert "R=2" in repr_str
        assert "untrained" in repr_str
        # kernels are converted to KernelNode objects; repr shows their str form
        assert "RBF" in repr_str or "rbf" in repr_str.lower()

    def test_predict_before_train_raises(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"])
        X_test = np.random.randn(5, 3).astype(np.float32)
        with pytest.raises(RuntimeError, match="trained"):
            gp.predict(X_test)


# ============================================================================
# LMC R=1 Training Tests
# ============================================================================


class TestLMCR1Training:
    """Test LMC with R=1 (should behave like ICM)."""

    @pytest.fixture
    def data_2task(self):
        return generate_multi_output_data(n=500, d=3, T=2, seed=42)

    def test_single_latent_training_sets_model_state(self, data_2task):
        """R=1 LMC training should set trained model state."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False, method="matrix_free")

        assert gp.is_trained
        assert result.num_tasks == 2
        assert result.num_latents == 1

    def test_r1_result_fields(self, data_2task):
        """Check all fields of LMCTrainingResult."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        # Per-latent fields
        assert result.lengthscales.shape == (1,)
        assert result.outputscales.shape == (1,)
        # kernel_types is not set for composite kernel path (KernelNode)
        # assert result.kernel_types.shape == (1,)

        # Per-task fields
        assert result.noise_per_task.shape == (2,)

        # Coregionalization matrices
        assert result.A_matrices.shape == (1, 2, 2)
        # L_factors may be None for composite kernel path
        if result.L_factors is not None:
            assert result.L_factors.shape == (1, 2, 2)

        # Task covariance (computed from A_matrices if not returned by engine)
        assert result.B is not None
        assert result.B.shape == (2, 2)
        assert result.Q is not None
        assert result.Q.shape == (2, 2)

        # Alpha is returned flat [n*T] by engine
        assert result.alpha is not None
        assert result.alpha.size == 500 * 2  # n*T elements

        # Diagnostics
        assert result.final_nll != 0
        assert len(result.nll_history) > 0

    def test_r1_nll_decreases(self, data_2task):
        """NLL should generally decrease during training."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=30, verbose=False)

        nll = result.nll_history
        # Final NLL should be lower than initial (allow some tolerance for stochastic CG)
        assert nll[-1] < nll[0] * 1.1, (
            f"NLL did not decrease: initial={nll[0]:.4f}, final={nll[-1]:.4f}"
        )

    def test_matrix_free_trains_and_predicts(self):
        """matrix_free LMC should fit and predict on the public wrapper path."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_multi_output_data(n=2000, d=3, T=2, seed=7)
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=8,
        )
        result = gp.fit(X, Y, max_iterations=5, verbose=False, method="matrix_free")

        assert gp.is_trained
        assert result.num_tasks == 2

        pred = gp.predict(X[:8])
        assert pred.mean.shape == (8, 2)
        assert np.all(np.isfinite(pred.mean))

    def test_r1_prediction_shapes(self, data_2task):
        """Check prediction output shapes."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        gp.fit(X, Y, max_iterations=20, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)

        # Test return_var
        mean, var = gp.predict(X_test, return_var=True, variance_method="exact")
        assert mean.shape == (10, 2)
        assert var.shape == (10, 2)
        assert np.all(var >= 0), "Variance must be non-negative"

        # Test return_std
        mean, std = gp.predict(X_test, return_std=True)
        assert std.shape == (10, 2)
        assert np.all(std >= 0), "Std must be non-negative"

    def test_r1_prediction_reasonable(self, data_2task):
        """Predictions should not be NaN/Inf and should be in a reasonable range."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        gp.fit(X, Y, max_iterations=30, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True, variance_method="exact")

        assert np.all(np.isfinite(mean)), "Mean contains NaN/Inf"
        assert np.all(np.isfinite(var)), "Variance contains NaN/Inf"
        # Mean should be in a reasonable range given training data
        assert np.max(np.abs(mean)) < 100, f"Mean too large: max={np.max(np.abs(mean))}"

    def test_r1_score(self, data_2task):
        """Test the score() method."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        gp.fit(X, Y, max_iterations=30, verbose=False)

        X_test, Y_test = generate_multi_output_data(n=500, d=3, T=2, seed=99)
        scores = gp.score(X_test, Y_test)

        assert "rmse" in scores
        assert "mae" in scores
        assert "r2" in scores
        assert "rmse_per_task" in scores
        assert scores["rmse"] > 0
        assert scores["rmse_per_task"].shape == (2,)


# ============================================================================
# LMC R>1 Training Tests
# ============================================================================


class TestLMCR2Training:
    """Test LMC with R=2 (two latent kernels)."""

    @pytest.fixture
    def data_2task(self):
        return generate_multi_output_data(n=500, d=3, T=2, seed=42)

    @pytest.fixture
    def data_3task(self):
        return generate_multi_output_data(n=500, d=3, T=3, seed=42)

    def test_r2_same_kernel_trains(self, data_2task):
        """R=2 with same kernel type should train."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert gp.is_trained
        assert result.num_latents == 2
        assert result.lengthscales.shape == (2,)
        assert result.A_matrices.shape == (2, 2, 2)

    def test_r2_heterogeneous_kernels_trains(self, data_2task):
        """R=2 with RBF + Matern52 should train."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert gp.is_trained
        assert result.num_latents == 2

    def test_r2_prediction_shapes(self, data_2task):
        """R=2 prediction should produce correct shapes."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        gp.fit(X, Y, max_iterations=20, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True, variance_method="exact")

        assert mean.shape == (10, 2)
        assert var.shape == (10, 2)
        assert np.all(np.isfinite(mean)), "Mean contains NaN/Inf"
        assert np.all(np.isfinite(var)), "Variance contains NaN/Inf"
        assert np.all(var >= 0), "Variance must be non-negative"

    def test_r2_prediction_reasonable(self, data_2task):
        """R=2 predictions should be in a reasonable range."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        gp.fit(X, Y, max_iterations=30, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True, variance_method="exact")

        assert np.all(np.isfinite(mean)), "Mean contains NaN/Inf"
        assert np.all(np.isfinite(var)), "Variance contains NaN/Inf"
        assert np.max(np.abs(mean)) < 100, f"Mean too large: max={np.max(np.abs(mean))}"

    def test_r2_3tasks(self, data_3task):
        """R=2 with 3 tasks should work."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_3task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert result.num_tasks == 3
        assert result.A_matrices.shape == (2, 3, 3)

        X_test = np.random.randn(5, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True)
        assert mean.shape == (5, 3)
        assert var.shape == (5, 3)

    def test_r2_nll_decreases(self, data_2task):
        """NLL should decrease during R=2 training."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=30, verbose=False)

        nll = result.nll_history
        assert nll[-1] < nll[0] * 1.1, (
            f"NLL did not decrease: initial={nll[0]:.4f}, final={nll[-1]:.4f}"
        )

    def test_r2_score(self, data_2task):
        """Test score() with R=2."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        gp.fit(X, Y, max_iterations=30, verbose=False)

        X_test, Y_test = generate_multi_output_data(n=500, d=3, T=2, seed=99)
        scores = gp.score(X_test, Y_test)

        assert scores["rmse"] > 0
        assert np.all(np.isfinite(list(scores.values())[:3]))  # rmse, mae, r2

    def test_r3_trains(self, data_2task):
        """R=3 with three different kernels should train."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = data_2task
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern32", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=15, verbose=False)

        assert gp.is_trained
        assert result.num_latents == 3
        assert result.lengthscales.shape == (3,)
        assert result.A_matrices.shape == (3, 2, 2)


class TestLMCMixedTraining:
    """End-to-end mixed continuous+categorical LMC coverage."""

    @pytest.fixture(autouse=True)
    def _cleanup_between_tests(self):
        yield
        _cleanup_gpu_state()

    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    def test_mixed_lmc_fit_predict_save_load(self, method, tmp_path):
        from mojogp import Kernel, MultiOutputLMCGP

        n_train = 5000 if method == "materialized" else 8000
        X, Y = generate_mixed_multi_output_data(
            n=n_train, d_cont=2, T=2, levels=3, seed=17
        )
        kernels = [
            Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
            Kernel.matern52(active_dims=[0, 1]),
        ]
        gp = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=8,
        )

        result = gp.fit(X, Y, max_iterations=4, learning_rate=0.03, verbose=False, method=method)

        assert gp.is_trained
        assert gp._has_mixed_latents is True
        assert result.cat_params_per_latent is not None
        assert len(result.cat_params_per_latent) == 2
        assert np.all(np.isfinite(result.noise_per_task))

        X_test = X[:6].copy()
        mean, var = gp.predict(X_test, return_var=True, variance_method="exact")
        assert mean.shape == (6, 2)
        assert var.shape == (6, 2)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0.0)
        assert gp.backend_predict_info["actual_prediction_route"] == "predict_lmc_mixed"
        assert gp.backend_predict_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"
        assert gp.backend_predict_info["backend_prediction_used"] is True
        assert gp.backend_predict_info["backend_variance_used"] is True
        assert gp.backend_predict_info["fallback_used"] is False
        assert gp.backend_predict_info["precond_rank"] == 8
        assert gp.backend_predict_info["precond_method"] == gp.precond_method
        expected_exact_role = "inference" if method == "materialized" else "training"
        assert gp.backend_predict_info["provider_bundle_role"] == expected_exact_role
        first_inference_bundle_id = (
            None if gp._inference_bundle is None else id(gp._inference_bundle)
        )

        love_mean, love_var = gp.predict(
            X_test, return_var=True, variance_method="love"
        )
        love_info = dict(gp.backend_predict_info)
        assert love_mean.shape == (6, 2)
        assert love_var.shape == (6, 2)
        assert np.all(np.isfinite(love_mean))
        assert np.all(np.isfinite(love_var))
        assert np.all(love_var >= 0.0)
        assert love_info["variance_method"] == "love"
        assert love_info["actual_prediction_route"] == "predict_lmc_mixed"
        assert love_info["actual_variance_route"] == "predict_lmc_mixed"
        assert love_info["backend_prediction_used"] is True
        assert love_info["backend_variance_used"] is True
        assert love_info["fallback_used"] is False
        np.testing.assert_allclose(love_mean, mean, atol=1e-5, rtol=1e-5)

        samples = gp.sample_posterior(X_test, n_samples=2, method="pathwise")
        assert samples.shape == (2, 6, 2)
        assert np.all(np.isfinite(samples))
        expected_sample_role = "inference" if method == "materialized" else "training"
        assert gp.backend_sample_info["provider_bundle_role"] == expected_sample_role
        if method == "materialized":
            assert gp._inference_bundle is not None
        else:
            assert gp._inference_bundle is None

        save_path = tmp_path / f"mixed_lmc_{method}"
        gp.save(save_path)
        loaded = MultiOutputLMCGP.load(save_path)

        loaded_mean, loaded_var = loaded.predict(
            X_test, return_var=True, variance_method="exact"
        )
        assert (
            loaded.backend_predict_info["actual_prediction_route"]
            == "predict_lmc_mixed"
        )
        assert (
            loaded.backend_predict_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"
        )
        assert loaded.backend_predict_info["backend_prediction_used"] is True
        assert loaded.backend_predict_info["backend_variance_used"] is True
        assert loaded.backend_predict_info["fallback_used"] is False
        assert loaded.backend_predict_info["precond_rank"] == 8
        assert loaded.backend_predict_info["precond_method"] == loaded.precond_method
        assert loaded.backend_predict_info["provider_bundle_role"] == "inference"
        np.testing.assert_allclose(loaded_mean, mean, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(loaded_var, var, atol=1e-4, rtol=1e-4)
        loaded_inference_bundle_id = id(loaded._inference_bundle)

        loaded_love_mean, loaded_love_var = loaded.predict(
            X_test, return_var=True, variance_method="love"
        )
        assert (
            loaded.backend_predict_info["actual_prediction_route"]
            == "predict_lmc_mixed"
        )
        assert (
            loaded.backend_predict_info["actual_variance_route"]
            == "predict_lmc_mixed"
        )
        assert loaded.backend_predict_info["backend_variance_used"] is True
        assert loaded.backend_predict_info["fallback_used"] is False
        np.testing.assert_allclose(loaded_love_mean, love_mean, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(
            loaded_love_var, love_var, atol=1.5e-2, rtol=1e-1
        )

        loaded_samples = loaded.sample_posterior(X_test, n_samples=2, method="pathwise")
        assert loaded_samples.shape == (2, 6, 2)
        assert loaded.backend_sample_info["provider_bundle_role"] == "inference"
        assert id(loaded._inference_bundle) == loaded_inference_bundle_id

        repeat_mean, repeat_var = gp.predict(
            X_test, return_var=True, variance_method="exact"
        )
        np.testing.assert_allclose(repeat_mean, mean, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(repeat_var, var, atol=1e-4, rtol=1e-4)
        if method == "materialized":
            assert gp.backend_predict_info["provider_bundle_role"] == "inference"

    def test_mixed_lmc_alternating_fit_methods_share_process_stably(self):
        from mojogp import Kernel, MultiOutputLMCGP

        X, Y = generate_mixed_multi_output_data(
            n=2000, d_cont=2, T=2, levels=3, seed=57
        )
        kernels = [
            Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
            Kernel.matern52(active_dims=[0, 1]),
        ]

        gp_materialized = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=4,
        )
        gp_materialized.fit(X, Y, max_iterations=3, learning_rate=0.03, verbose=False, method="materialized")
        mean_mat, var_mat = gp_materialized.predict(
            X[:6], return_var=True, variance_method="exact"
        )

        gp_matrix_free = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=4,
        )
        gp_matrix_free.fit(X, Y, max_iterations=3, learning_rate=0.03, verbose=False, method="matrix_free")
        mean_mf, var_mf = gp_matrix_free.predict(
            X[:6], return_var=True, variance_method="exact"
        )

        assert np.all(np.isfinite(mean_mat))
        assert np.all(np.isfinite(var_mat))
        assert np.all(np.isfinite(mean_mf))
        assert np.all(np.isfinite(var_mf))
        assert gp_materialized.backend_predict_info["provider_bundle_role"] == "inference"
        assert gp_matrix_free.backend_predict_info["provider_bundle_role"] == "training"

    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    def test_mixed_lmc_predictions_change_with_category(self, method):
        summary = _run_mixed_lmc_case("category_change", method)
        assert summary["ok"] is True

    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    def test_mixed_lmc_supports_nested_mixed_tree(self, method):
        summary = _run_mixed_lmc_case("nested_tree", method)
        assert summary["actual_prediction_route"] == "predict_lmc_mixed"
        assert summary["actual_variance_route"] == "predict_lmc_mixed_full_exact"
        assert summary["precond_rank"] == 8

    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    @pytest.mark.parametrize("kernel_name", ["gd", "cr", "ehh", "hh", "fe"])
    def test_mixed_lmc_supports_each_categorical_kernel_route(self, kernel_name, method):
        summary = run_isolated_case(
            module=MODULE,
            payload={
                "case": "categorical_kernel",
                "method": method,
                "kernel_name": kernel_name,
            },
            timeout=600,
            description=f"Runs mixed LMC categorical kernel case {kernel_name}/{method}",
        )
        assert summary["kernel_name"] == kernel_name
        assert summary["actual_prediction_route"] == "predict_lmc_mixed"
        assert summary["actual_variance_route"] == "predict_lmc_mixed_full_exact"
        assert summary["love_variance_route"] == "predict_lmc_mixed"
        assert summary["loaded_variance_route"] == "predict_lmc_mixed_full_exact"
        assert summary["loaded_love_variance_route"] == "predict_lmc_mixed"
        assert summary["categorical_sensitivity"] > 0.05
        assert summary["cat_param_counts"][0] > 0

    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    def test_lmc_fixed_observation_noise_trains_and_round_trips(self, method):
        summary = _run_mixed_lmc_case("fixed_observation_noise", method)
        assert summary["ok"] is True
        assert summary["training_route"] == method
        assert summary["fixed_noise_mean"] > 0.0


# ============================================================================
# LMC Input Validation Tests
# ============================================================================


class TestLMCInputValidation:
    """Test input validation for LMC."""

    def test_wrong_x_dims(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"])
        X = np.random.randn(10).astype(np.float32)  # 1D, should be 2D
        Y = np.random.randn(10, 2).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            gp.fit(X, Y)

    def test_wrong_y_dims(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"])
        X = np.random.randn(10, 3).astype(np.float32)
        Y = np.random.randn(10).astype(np.float32)  # 1D, should be 2D
        with pytest.raises(ValueError, match="2D"):
            gp.fit(X, Y)

    def test_mismatched_samples(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=["rbf"])
        X = np.random.randn(10, 3).astype(np.float32)
        Y = np.random.randn(20, 2).astype(np.float32)  # Different n
        with pytest.raises(ValueError, match="must match"):
            gp.fit(X, Y)

    def test_predict_wrong_dims(self):
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputLMCGP(
            kernels=["rbf"], num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        gp.fit(X, Y, max_iterations=10, verbose=False)

        X_test = np.random.randn(5, 4).astype(np.float32)  # Wrong d
        with pytest.raises(ValueError, match="features"):
            gp.predict(X_test)


# ============================================================================
# LMC Task Covariance Tests
# ============================================================================


class TestLMCTaskCovariance:
    """Test task covariance properties."""

    def test_a_matrices_psd(self):
        """A_s = L_s L_s^T should be positive semi-definite."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        for s in range(result.num_latents):
            A_s = result.A_matrices[s]
            # A_s should be symmetric
            np.testing.assert_allclose(A_s, A_s.T, atol=1e-5)
            # Eigenvalues should be non-negative (PSD)
            eigvals = np.linalg.eigvalsh(A_s)
            assert np.all(eigvals >= -1e-5), f"A_{s} not PSD: eigenvalues={eigvals}"

    def test_b_equals_sum_a(self):
        """B = sum_s A_s should hold."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        B_expected = np.sum(result.A_matrices, axis=0)
        np.testing.assert_allclose(result.B, B_expected, atol=1e-5)

    def test_task_covariance_property(self):
        """Test the task_covariance property."""
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            num_probes=5,
            max_cg_iterations=50,
            preconditioner_rank=10,
        )
        gp.fit(X, Y, max_iterations=20, verbose=False)

        B = gp.task_covariance
        assert B is not None
        assert B.shape == (2, 2)
        np.testing.assert_allclose(B, B.T, atol=1e-5)
