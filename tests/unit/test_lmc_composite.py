"""Unit tests for LMC composite kernel support.

Tests the Python-side API for MultiOutputLMCGP with KernelNode composite kernels,
including validation, codegen, hashing, and prediction logic.
"""

import types

import numpy as np
import pytest

from mojogp.kernel import Kernel, KernelNode
from mojogp.multi_output_gp import MultiOutputLMCGP, LMCTrainingResult


# =============================================================================
# Construction and Validation Tests
# =============================================================================


class TestLMCCompositeConstruction:
    """Test MultiOutputLMCGP construction with composite kernels."""

    def test_accepts_kernel_node_list(self):
        """MultiOutputLMCGP should accept a list of KernelNode objects."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])
        assert gp._is_composite is True
        assert gp._composite_kernel is k
        assert gp.num_latents == 2

    def test_single_latent_composite(self):
        """Single latent with composite kernel."""
        k = Kernel.rbf() * Kernel.periodic()
        gp = MultiOutputLMCGP(kernels=[k])
        assert gp._is_composite is True
        assert gp.num_latents == 1

    def test_three_latents_composite(self):
        """Three latents with composite kernel."""
        k = Kernel.matern32() + Kernel.linear()
        gp = MultiOutputLMCGP(kernels=[k, k, k])
        assert gp._is_composite is True
        assert gp.num_latents == 3

    def test_rejects_mixed_string_and_kernel_node(self):
        """Cannot mix string and KernelNode kernels."""
        k = Kernel.rbf() + Kernel.matern52()
        with pytest.raises(ValueError, match="All kernels must be KernelNode"):
            MultiOutputLMCGP(kernels=[k, "rbf"])

    def test_accepts_ard_with_composite(self):
        """ARD is now supported with composite LMC kernels."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k], ard=True)
        assert gp.ard is True
        assert gp._is_composite is True

    def test_ard_compilation_uses_single_fit_time_transform(self):
        """Fit-time compiled kernels are the only ARD transform source."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, Kernel.matern32()], ard=True)

        gp._configure_latent_kernels_for_fit(total_dim=4)
        compiled_before = [kernel.to_mojo_type() for kernel in gp._latent_compiled_kernels]
        original_before = [kernel.to_mojo_type() for kernel in gp._original_kernels]

        gp._configure_latent_kernels_for_fit(total_dim=4)
        compiled_after = [kernel.to_mojo_type() for kernel in gp._latent_compiled_kernels]
        original_after = [kernel.to_mojo_type() for kernel in gp._original_kernels]

        assert compiled_after == compiled_before
        assert original_after == original_before
        assert all("ARD" in mojo_type for mojo_type in compiled_after)
        assert all("ARD" not in mojo_type for mojo_type in original_after)

    def test_ard_compilation_respects_active_dim_parameter_counts(self):
        """ARD parameter counts use active-dim widths per latent subtree."""
        gp = MultiOutputLMCGP(
            kernels=[
                Kernel.rbf(active_dims=[0, 2]),
                Kernel.matern52(active_dims=[1, 3, 4]),
            ],
            ard=True,
        )

        gp._configure_latent_kernels_for_fit(total_dim=5)

        assert [kernel.num_params() for kernel in gp._latent_compiled_kernels] == [3, 4]
        assert [kernel.ard_dim for kernel in gp._latent_compiled_kernels] == [2, 3]

    def test_initial_lengthscales_map_to_compiled_ard_params(self):
        """initial_lengthscales follows heterogeneous compiled latent layouts."""
        gp = MultiOutputLMCGP(
            kernels=[
                Kernel.rbf(active_dims=[0, 2]),
                Kernel.matern52(active_dims=[1, 3, 4]),
            ],
            ard=True,
        )
        captured = {}

        def fake_fit_composite(self, *args, **kwargs):
            captured["init_params"] = kwargs["init_params"].copy()
            return LMCTrainingResult(
                final_nll=0.0,
                nll_history=np.array([0.0], dtype=np.float32),
                iterations=1,
                converged=True,
                num_latents=2,
                num_tasks=2,
                noise_per_task=np.ones(2, dtype=np.float32),
            )

        gp._fit_composite = types.MethodType(fake_fit_composite, gp)
        X = np.zeros((4, 5), dtype=np.float32)
        Y = np.zeros((4, 2), dtype=np.float32)

        gp.fit(
            X,
            Y,
            initial_lengthscales=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        )

        np.testing.assert_allclose(
            captured["init_params"],
            np.array([0.1, 0.2, 1.0, 0.3, 0.4, 0.5, 1.0], dtype=np.float32),
        )

    def test_initial_lengthscales_shape_validates_before_native_fit(self):
        """Bad LMC initial_lengthscales fail before JIT warmup or native load."""
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()], ard=True)
        X = np.zeros((4, 3), dtype=np.float32)
        Y = np.zeros((4, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="initial_lengthscales must have shape"):
            gp.fit(X, Y, initial_lengthscales=np.ones(5, dtype=np.float32))

    def test_initial_lengthscales_and_initial_params_are_mutually_exclusive(self):
        """Ambiguous LMC initialization inputs are rejected explicitly."""
        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()], ard=True)
        X = np.zeros((4, 3), dtype=np.float32)
        Y = np.zeros((4, 2), dtype=np.float32)

        with pytest.raises(ValueError, match="either initial_params or initial_lengthscales"):
            gp.fit(
                X,
                Y,
                initial_lengthscales=np.ones(3, dtype=np.float32),
                initial_params=np.ones(4, dtype=np.float32),
            )

    def test_polynomial_engine_mask_freezes_fixed_power_and_hidden_variance(self):
        """Polynomial engine layout only trains public covariance scale slots."""
        k = Kernel.polynomial(degree=3.0, offset=1.5, outputscale=2.0)

        np.testing.assert_allclose(
            k.to_engine_params(k.get_initial_params()),
            np.array([1.0, 3.0, 1.5, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            k.engine_trainable_mask(),
            np.array([False, False, True, True], dtype=np.bool_),
        )

    def test_polynomial_pathwise_feature_map_matches_kernel_gram(self):
        """Polynomial pathwise features exactly reproduce the polynomial Gram."""
        from mojogp.pathwise_prior import build_pathwise_feature_map

        X = np.array([[0.2, -0.4], [1.1, 0.3], [-0.7, 0.5]], dtype=np.float32)
        k = Kernel.polynomial(degree=3.0, offset=1.25, outputscale=0.75)
        params = k.get_initial_params()

        fmap = build_pathwise_feature_map(
            k,
            params,
            input_dim=2,
            n_features=16,
            rng=np.random.default_rng(1),
        )
        features = fmap.evaluate(X, None)
        gram = features @ features.T
        expected = k.evaluate(X, X, params=params)

        assert fmap.is_exact is True
        np.testing.assert_allclose(gram, expected, rtol=1e-6, atol=1e-6)

    def test_composite_route_defaults_to_materialized_until_fit(self):
        """Composite LMC route selection is resolved by fit()."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])
        assert gp.method == "materialized"

    def test_string_kernels_still_work(self):
        """Standard string kernels are converted to KernelNode (JIT path)."""
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        # String kernels are now always converted to KernelNode at init
        assert gp._is_composite is True
        assert gp.num_latents == 2

    def test_repr_composite(self):
        """Repr should indicate composite mode."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])
        r = repr(gp)
        assert "composite=True" in r
        assert "R=2" in r

    def test_repr_string(self):
        """Repr shows composite=True since string kernels are converted to KernelNode."""
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        r = repr(gp)
        # All kernels go through composite path now (string → KernelNode at init)
        assert "composite=True" in r
        assert "R=2" in r


# =============================================================================
# Prediction Tests (with mock training result)
# =============================================================================


class TestLMCCompositePrediction:
    """Test composite LMC prediction using Python-side kernel evaluation."""

    def _make_trained_gp(self, R=2, T=3, n=2000, d=5):
        """Create a mock-trained composite LMC GP for prediction testing."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k] * R)

        # Create mock training data
        np.random.seed(42)
        X_train = np.random.randn(n, d).astype(np.float32)
        Y_train = np.random.randn(n, T).astype(np.float32)

        # Create mock training result
        A_matrices = np.zeros((R, T, T), dtype=np.float32)
        for s in range(R):
            L = np.eye(T, dtype=np.float32) * 0.5
            A_matrices[s] = L @ L.T

        B = np.sum(A_matrices, axis=0)
        eigvals, eigvecs = np.linalg.eigh(B)

        alpha = np.random.randn(n, T).astype(np.float32) * 0.1
        alpha_rotated = (alpha @ eigvecs).astype(np.float32)

        num_params = k.num_params()
        learned_params = np.ones((R, num_params), dtype=np.float32)

        gp._result = LMCTrainingResult(
            lengthscales=np.ones(R, dtype=np.float32),
            outputscales=np.ones(R, dtype=np.float32),
            params_per_latent=[learned_params[s].copy() for s in range(R)],
            kernel_types=np.zeros(R, dtype=np.int32),
            noise_per_task=np.full(T, 0.1, dtype=np.float32),
            A_matrices=A_matrices,
            L_factors=np.zeros((R, T, T), dtype=np.float32),
            B=B,
            Q=eigvecs.astype(np.float32),
            Lambda=eigvals.astype(np.float32),
            alpha=alpha,
            alpha_rotated=alpha_rotated,
            effective_scales=eigvals.astype(np.float32),
            final_nll=1.0,
            nll_history=np.array([2.0, 1.5, 1.0], dtype=np.float32),
            iterations=3,
            converged=True,
            num_latents=R,
            num_tasks=T,
        )
        gp._X_train = X_train
        gp._Y_train = Y_train
        gp._is_trained = True
        gp._learned_composite_params = learned_params

        return gp, X_train

    def test_predict_mean_shape(self):
        """Prediction mean should have correct shape [m, T]."""
        gp, X_train = self._make_trained_gp(R=2, T=3, n=2000, d=5)
        X_test = np.random.randn(10, 5).astype(np.float32)

        result = gp.predict(X_test)
        assert result.mean.shape == (10, 3)

    def test_predict_with_variance(self):
        """Prediction with variance should return (mean, var) tuple."""
        gp, X_train = self._make_trained_gp(R=2, T=3, n=2000, d=5)
        X_test = np.random.randn(10, 5).astype(np.float32)

        mean, var = gp.predict(X_test, return_var=True)
        assert mean.shape == (10, 3)
        assert var.shape == (10, 3)
        assert np.all(var >= 0), "Variance should be non-negative"

    def test_predict_with_std(self):
        """Prediction with std should return (mean, std) tuple."""
        gp, X_train = self._make_trained_gp(R=2, T=3, n=2000, d=5)
        X_test = np.random.randn(10, 5).astype(np.float32)

        mean, std = gp.predict(X_test, return_std=True)
        assert mean.shape == (10, 3)
        assert std.shape == (10, 3)
        assert np.all(std >= 0), "Std should be non-negative"

    def test_predict_mean_not_all_zeros(self):
        """Prediction mean should not be all zeros (sanity check)."""
        gp, X_train = self._make_trained_gp(R=2, T=3, n=2000, d=5)
        X_test = np.random.randn(10, 5).astype(np.float32)

        result = gp.predict(X_test)
        assert not np.allclose(result.mean, 0), "Mean should not be all zeros"

    def test_predict_single_latent(self):
        """Prediction should work with R=1."""
        gp, X_train = self._make_trained_gp(R=1, T=2, n=2000, d=3)
        X_test = np.random.randn(5, 3).astype(np.float32)

        result = gp.predict(X_test)
        assert result.mean.shape == (5, 2)

    def test_predict_not_trained_raises(self):
        """Prediction before training should raise."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])

        with pytest.raises(RuntimeError, match="trained"):
            gp.predict(np.random.randn(5, 3).astype(np.float32))

    def test_predict_dimension_mismatch_raises(self):
        """Prediction with wrong dimension should raise."""
        gp, X_train = self._make_trained_gp(R=2, T=3, n=2000, d=5)
        X_test = np.random.randn(10, 3).astype(np.float32)  # Wrong dim

        with pytest.raises(ValueError, match="features"):
            gp.predict(X_test)

    def test_matrix_free_exact_variance_refuses_dense_fallback(self):
        """Matrix-free exact variance should require a backend route."""
        gp, _ = self._make_trained_gp(R=1, T=2, n=64, d=3)
        gp.method = "matrix_free"
        X_test = np.random.randn(6, 3).astype(np.float32)

        with pytest.raises(RuntimeError, match="Matrix-free MultiOutputLMCGP exact variance"):
            gp.predict(X_test, return_var=True, variance_method="exact")

        assert gp.backend_predict_info["actual_prediction_route"] == "python_fallback"
        assert gp.backend_predict_info["actual_variance_route"] == "forbidden_python_fallback"
        assert gp.backend_predict_info["backend_variance_used"] is False
        assert gp.backend_predict_info["fallback_used"] is False

    def test_dense_exact_variance_includes_observation_noise(self):
        """Dense exact LMC variance returns observation variance, not latent-only variance."""
        gp, X_train = self._make_trained_gp(R=1, T=2, n=8, d=3)
        gp._configure_latent_kernels_for_fit(3)
        gp._result.A_matrices[:] = 0.0
        noise = np.asarray([0.05, 0.2], dtype=np.float32)

        var = gp._predict_lmc_dense_exact_variance(
            X_train[:3],
            [np.ones(gp.kernels[0].num_params(), dtype=np.float32)],
            None,
            gp._result.A_matrices,
            noise,
        )

        assert var.shape == (3, 2)
        expected = np.broadcast_to(noise[np.newaxis, :], var.shape)
        np.testing.assert_allclose(var, expected, rtol=1e-5, atol=1e-6)


# =============================================================================
# Fit Validation Tests (no JIT needed)
# =============================================================================


class TestLMCCompositeFitValidation:
    """Test fit() input validation for composite kernels."""

    def test_fit_validates_X_shape(self):
        """fit() should reject 1D X."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])

        with pytest.raises(ValueError, match="2D"):
            gp.fit(
                np.random.randn(50).astype(np.float32),
                np.random.randn(50, 3).astype(np.float32),
                method="matrix_free",
            )

    def test_fit_validates_Y_shape(self):
        """fit() should reject 1D Y."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])

        with pytest.raises(ValueError, match="2D"):
            gp.fit(
                np.random.randn(50, 5).astype(np.float32),
                np.random.randn(50).astype(np.float32),
            )

    def test_fit_validates_sample_count_match(self):
        """fit() should reject mismatched X and Y sample counts."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])

        with pytest.raises(ValueError, match="must match"):
            gp.fit(
                np.random.randn(50, 5).astype(np.float32),
                np.random.randn(40, 3).astype(np.float32),
            )

    def test_fit_validates_init_params_shape(self):
        """fit() should reject wrong init_params shape."""
        k = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k, k])
        num_params = k.num_params()

        with pytest.raises(ValueError, match="init_params"):
            gp.fit(
                np.random.randn(50, 5).astype(np.float32),
                np.random.randn(50, 3).astype(np.float32),
                initial_params=np.ones(3, dtype=np.float32),  # Wrong size
            )
