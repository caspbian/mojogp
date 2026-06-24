"""Unit tests for multi-output categorical kernel detection.

MultiOutputGP and MultiOutputLMCGP support mixed continuous+categorical
kernels, but both still reject pure categorical multi-output models.
"""

import numpy as np
import pytest

from mojogp import MultiOutputGP, MultiOutputLMCGP, RBF, EHH
from mojogp import multi_output_gp as multi_output_gp_module
from mojogp.multi_output_gp import LMCTrainingResult


class _FakeKernelModule:
    def init_provider(self, X, params, noise):
        return {
            "provider_ptr": 1,
            "n": int(X.shape[0]),
            "num_gradient_params": int(len(params)),
            "supports_fused_gradient": True,
            "supports_fused_ls_os": False,
            "supports_fused_3param": False,
            "x_ptr": 1,
        }


class TestMultiOutputCategoricalDetection:
    """Test categorical-kernel support boundaries for multi-output wrappers."""

    def test_icm_accepts_pure_continuous_kernel(self):
        """Pure continuous kernel should be accepted without error."""
        gp = MultiOutputGP(kernel=RBF())
        # Should not raise

    def test_icm_pure_categorical_raises(self):
        """ICM with only categorical inputs should raise a clear ValueError."""
        kernel = EHH(levels=5, active_dims=[0])
        gp = MultiOutputGP(kernel=kernel)

        X = np.random.randint(0, 5, size=(50, 1)).astype(np.float32)
        Y = np.random.randn(50, 2).astype(np.float32)

        with pytest.raises(ValueError, match="Pure categorical multi-output"):
            gp.fit(X, Y, max_iterations=1)

    def test_lmc_pure_categorical_raises(self):
        """LMC with only categorical inputs should still raise a clear ValueError."""
        kernel = EHH(levels=5, active_dims=[0])
        gp = MultiOutputLMCGP(kernels=[kernel, kernel])

        X = np.random.randint(0, 5, size=(50, 1)).astype(np.float32)
        Y = np.random.randn(50, 2).astype(np.float32)

        with pytest.raises(ValueError, match="Pure categorical latent kernels"):
            gp.fit(X, Y, max_iterations=1)

    def test_lmc_mixed_kernel_routes_to_mixed_fit(self, monkeypatch):
        """Mixed LMC kernels should route to the mixed fit path instead of raising."""
        kernel = RBF(active_dims=[0, 1]) * EHH(levels=5, active_dims=[2])
        gp = MultiOutputLMCGP(kernels=[kernel, kernel])

        X = np.random.randn(50, 3).astype(np.float32)
        X[:, 2] = np.random.randint(0, 5, size=50).astype(np.float32)
        Y = np.random.randn(50, 2).astype(np.float32)

        called = {}

        def fake_fit(self, *args, **kwargs):
            called["mixed"] = True
            return "mixed-result"

        monkeypatch.setattr(MultiOutputLMCGP, "_fit_mixed_composite", fake_fit)

        result = gp.fit(X, Y, max_iterations=1)

        assert result == "mixed-result"
        assert called["mixed"] is True
        assert gp._has_mixed_latents is True

    def test_lmc_one_mixed_latent_routes_to_mixed_fit(self, monkeypatch):
        """A single mixed latent should still route the whole LMC model to mixed fit."""
        k_cont = RBF()
        k_cat = RBF(active_dims=[0, 1]) * EHH(levels=5, active_dims=[2])
        gp = MultiOutputLMCGP(kernels=[k_cont, k_cat])

        X = np.random.randn(50, 3).astype(np.float32)
        X[:, 2] = np.random.randint(0, 5, size=50).astype(np.float32)
        Y = np.random.randn(50, 2).astype(np.float32)

        called = {}

        def fake_fit(self, *args, **kwargs):
            called["mixed"] = True
            return "mixed-result"

        monkeypatch.setattr(MultiOutputLMCGP, "_fit_mixed_composite", fake_fit)

        result = gp.fit(X, Y, max_iterations=1)

        assert result == "mixed-result"
        assert called["mixed"] is True
        assert gp._has_mixed_latents is True

    def test_icm_mixed_kernel_is_supported_at_configuration_time(self):
        """Mixed continuous+categorical ICM kernels should configure successfully."""
        kernel = RBF(active_dims=[0]) * EHH(levels=3, active_dims=[1])
        gp = MultiOutputGP(kernel=kernel)

        X = np.random.randn(50, 2).astype(np.float32)
        X[:, 1] = np.random.randint(0, 3, size=50).astype(np.float32)
        Y = np.random.randn(50, 2).astype(np.float32)

        gp._configure_kernel_for_fit(X.shape[1])

        assert gp._is_mixed is True
        assert gp._cat_specs is not None
        assert gp._cont_dim == 1

    def test_lmc_mixed_backend_failure_does_not_silently_fallback(self):
        """Mixed LMC backend prediction failures should stay visible."""

        class BrokenEngine:
            def predict_lmc_mixed(self, *args, **kwargs):
                raise RuntimeError("simulated backend failure")

        kernel = RBF(active_dims=[0, 1]) * EHH(levels=3, active_dims=[2])
        gp = MultiOutputLMCGP(kernels=[kernel])

        n = 4
        T = 2
        X_train = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.2, -0.1, 1.0],
                [0.4, 0.3, 2.0],
                [0.1, 0.5, 1.0],
            ],
            dtype=np.float32,
        )

        gp._is_trained = True
        gp.method = "matrix_free"
        gp._X_train = X_train
        gp._engine = BrokenEngine()
        gp._provider_infos = [{}]
        gp._kernel_modules = [_FakeKernelModule()]
        gp._has_mixed_latents = True
        gp._fitted_mean = np.zeros(T, dtype=np.float32)
        gp._latent_is_mixed = [True]
        gp._latent_X_train_conts = [
            np.ascontiguousarray(X_train[:, :2], dtype=np.float32)
        ]
        gp._latent_C_trains = [np.ascontiguousarray(X_train[:, [2]], dtype=np.int32)]
        gp._latent_cat_specs = [[{"levels": 3, "kernel_type": "ehh"}]]
        gp._latent_cat_col_indices = [[2]]
        gp._latent_dim_permutations = [None]

        gp._result = LMCTrainingResult(
            final_nll=1.0,
            nll_history=np.array([1.2, 1.0], dtype=np.float32),
            iterations=2,
            converged=False,
            num_latents=1,
            num_tasks=T,
            noise_per_task=np.array([0.1, 0.12], dtype=np.float32),
            params_per_latent=[np.array([1.0, 1.0], dtype=np.float32)],
            A_matrices=np.array([[[1.0, 0.2], [0.2, 0.8]]], dtype=np.float32),
            alpha=np.zeros(n * T, dtype=np.float32),
            cat_params_per_latent=[np.array([0.1, 0.2, 0.3], dtype=np.float32)],
        )

        with pytest.raises(RuntimeError, match="predict_lmc_mixed"):
            gp.predict(X_train[:2], return_var=True, variance_method="exact")

        assert gp.backend_predict_info is not None
        assert gp.backend_predict_info["actual_prediction_route"] == "predict_lmc_mixed"
        assert gp.backend_predict_info["fallback_used"] is False
        assert "simulated backend failure" in gp.backend_predict_info["backend_error"]

    def test_lmc_mixed_pathwise_reuses_live_provider_infos(self, monkeypatch):
        """Mixed LMC pathwise sampling should reuse live matrix-free providers."""

        class FakeEngine:
            def __init__(self):
                self.provider_infos_arg = None

            def sample_lmc_mixed_pathwise(self, provider_infos, residual, *args, **kwargs):
                self.provider_infos_arg = provider_infos
                return {
                    "correction": np.zeros_like(residual[:2], dtype=np.float32),
                }

        kernel = RBF(active_dims=[0, 1]) * EHH(levels=3, active_dims=[2])
        gp = MultiOutputLMCGP(kernels=[kernel])

        X_train = np.array(
            [
                [0.0, 0.1, 0.0],
                [0.2, -0.1, 1.0],
                [0.4, 0.3, 2.0],
                [0.1, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        Y_train = np.array(
            [
                [0.5, -0.1],
                [0.2, 0.3],
                [-0.4, 0.7],
                [0.1, -0.2],
            ],
            dtype=np.float32,
        )
        live_provider_infos = [{"provider_ptr": 7}]
        engine = FakeEngine()

        gp._configure_latent_kernels_for_fit(X_train.shape[1])
        gp._is_trained = True
        gp.method = "matrix_free"
        gp._X_train = X_train
        gp._Y_train = Y_train
        gp._engine = engine
        gp._provider_infos = live_provider_infos
        gp._kernel_modules = [_FakeKernelModule()]
        gp._fitted_mean = np.zeros(2, dtype=np.float32)
        gp._latent_X_train_conts = [np.ascontiguousarray(X_train[:, :2], dtype=np.float32)]
        gp._latent_C_trains = [np.ascontiguousarray(X_train[:, [2]], dtype=np.int32)]
        gp._training_bundle = None
        gp._inference_bundle = None

        gp._result = LMCTrainingResult(
            final_nll=1.0,
            nll_history=np.array([1.2, 1.0], dtype=np.float32),
            iterations=2,
            converged=False,
            num_latents=1,
            num_tasks=2,
            noise_per_task=np.array([0.1, 0.12], dtype=np.float32),
            params_per_latent=[np.array([1.0, 1.0], dtype=np.float32)],
            A_matrices=np.array([[[1.0, 0.2], [0.2, 0.8]]], dtype=np.float32),
            alpha=np.zeros(X_train.shape[0] * 2, dtype=np.float32),
            cat_params_per_latent=[np.array([0.1, 0.2, 0.3], dtype=np.float32)],
        )

        def fail_get_bundle(*args, **kwargs):
            raise AssertionError("should reuse live provider infos")

        monkeypatch.setattr(gp, "_get_mixed_runtime_bundle", fail_get_bundle)
        monkeypatch.setattr(
            multi_output_gp_module,
            "build_pathwise_feature_map",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            multi_output_gp_module,
            "build_feature_weights",
            lambda *args, **kwargs: np.ones((1, 2), dtype=np.float64),
        )

        def fake_sample_prior_values(feature_map, X_cont, C, weights):
            del feature_map, C, weights
            return np.zeros((1, X_cont.shape[0]), dtype=np.float64)

        monkeypatch.setattr(
            multi_output_gp_module,
            "sample_prior_values",
            fake_sample_prior_values,
        )

        samples = gp.sample_posterior(X_train[:2], n_samples=1, method="pathwise")

        assert samples.shape == (1, 2, 2)
        assert engine.provider_infos_arg is live_provider_infos
        assert gp.backend_sample_info["provider_bundle_role"] == "training"
