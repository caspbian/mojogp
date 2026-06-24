"""Unit tests for kernel evaluation, scaling utilities, and wrapper APIs."""

import os
import tempfile
import numpy as np
import pytest

from mojogp.kernel import Kernel, KernelNode, KernelType, make_ard_kernel
from mojogp.utils import StandardScaler


# =============================================================================
# Kernel Evaluation
# =============================================================================


class TestKernelEvaluate:
    """Test kernel.evaluate() for all kernel types."""

    def test_rbf_kernel_diagonal(self):
        """RBF kernel of X with itself should have outputscale on diagonal."""
        k = Kernel.rbf()
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        params = np.array([1.0, 1.0])  # lengthscale=1, outputscale=1

        K = k.evaluate(X, params=params)

        assert K.shape == (3, 3)
        # Diagonal should be outputscale (=1.0) since dist(x,x) = 0
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)

    def test_rbf_kernel_symmetry(self):
        """Kernel matrix should be symmetric."""
        k = Kernel.rbf()
        X = np.random.randn(10, 3).astype(np.float32)
        K = k.evaluate(X)

        np.testing.assert_allclose(K, K.T, atol=1e-5)

    def test_rbf_kernel_cross(self):
        """K(X, X2) should have correct shape."""
        k = Kernel.rbf()
        X = np.random.randn(5, 3).astype(np.float32)
        X2 = np.random.randn(7, 3).astype(np.float32)

        K = k.evaluate(X, X2)
        assert K.shape == (5, 7)

    def test_rbf_known_value(self):
        """RBF with known inputs should match manual computation."""
        k = Kernel.rbf()
        X = np.array([[0.0]], dtype=np.float32)
        X2 = np.array([[1.0]], dtype=np.float32)
        params = np.array([1.0, 1.0])  # ls=1, os=1

        K = k.evaluate(X, X2, params=params)
        expected = np.exp(-0.5 * 1.0)  # exp(-0.5 * (1/1)^2)
        np.testing.assert_allclose(K[0, 0], expected, atol=1e-5)

    def test_matern12_kernel(self):
        """Matern-1/2 kernel evaluation."""
        k = Kernel.matern12()
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        params = np.array([1.0, 1.0])  # ls=1, os=1

        K = k.evaluate(X, params=params)
        assert K.shape == (2, 2)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)
        # Off-diagonal: exp(-|1-0|/1) = exp(-1)
        np.testing.assert_allclose(K[0, 1], np.exp(-1.0), atol=1e-3)

    def test_matern32_kernel(self):
        """Matern-3/2 kernel evaluation."""
        k = Kernel.matern32()
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        params = np.array([1.0, 1.0])

        K = k.evaluate(X, params=params)
        assert K.shape == (2, 2)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)

    def test_matern52_kernel(self):
        """Matern-5/2 kernel evaluation."""
        k = Kernel.matern52()
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        params = np.array([1.0, 1.0])

        K = k.evaluate(X, params=params)
        assert K.shape == (2, 2)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)

    def test_linear_kernel(self):
        """Linear kernel: K = outputscale * variance * X @ X2.T"""
        k = Kernel.linear()
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        params = np.array([1.0, 1.0])  # variance=1, outputscale=1

        K = k.evaluate(X, params=params)
        expected = X.astype(np.float64) @ X.astype(np.float64).T
        np.testing.assert_allclose(K, expected.astype(np.float32), atol=1e-4)

    def test_polynomial_kernel(self):
        """Polynomial kernel: K = outputscale * (X @ X2.T + offset)^degree"""
        k = Kernel.polynomial()
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        params = np.array([2.0, 1.0, 1.0])  # degree=2, offset=1, outputscale=1

        K = k.evaluate(X, params=params)
        # K[0,0] = (1*1 + 0*0 + 1)^2 = 4
        np.testing.assert_allclose(K[0, 0], 4.0, atol=1e-4)

    def test_rq_kernel(self):
        """RQ kernel evaluation."""
        k = Kernel.rq()
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        params = np.array([1.0, 1.0, 1.0])  # ls=1, alpha=1, os=1

        K = k.evaluate(X, params=params)
        assert K.shape == (2, 2)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-5)

    def test_sum_kernel(self):
        """Sum kernel K1 + K2."""
        k = Kernel.rbf() + Kernel.matern52()
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        params = np.ones(4)  # 2 for RBF + 2 for Matern52

        K_sum = k.evaluate(X, params=params)

        # Compare with individual evaluations
        K_rbf = Kernel.rbf().evaluate(X, params=np.ones(2))
        K_m52 = Kernel.matern52().evaluate(X, params=np.ones(2))

        np.testing.assert_allclose(K_sum, K_rbf + K_m52, atol=1e-5)

    def test_product_kernel(self):
        """Product kernel K1 * K2."""
        k = Kernel.rbf() * Kernel.matern52()
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        params = np.ones(4)

        K_prod = k.evaluate(X, params=params)

        K_rbf = Kernel.rbf().evaluate(X, params=np.ones(2))
        K_m52 = Kernel.matern52().evaluate(X, params=np.ones(2))

        np.testing.assert_allclose(K_prod, K_rbf * K_m52, atol=1e-5)

    def test_scale_kernel(self):
        """Scale kernel c * K."""
        k = 3.0 * Kernel.rbf()
        X = np.array([[0.0], [1.0]], dtype=np.float32)
        # ScaleKernel has inner params (2) + 1 scale param
        params = np.array([1.0, 1.0, 3.0])

        K_scaled = k.evaluate(X, params=params)
        K_rbf = Kernel.rbf().evaluate(X, params=np.ones(2))

        np.testing.assert_allclose(K_scaled, 3.0 * K_rbf, atol=1e-5)

    def test_default_params(self):
        """evaluate() with no params should use all 1.0."""
        k = Kernel.rbf()
        X = np.random.randn(5, 2).astype(np.float32)

        K1 = k.evaluate(X)
        K2 = k.evaluate(X, params=np.ones(2))

        np.testing.assert_allclose(K1, K2, atol=1e-6)

    def test_ard_kernel(self):
        """ARD kernel with per-dimension lengthscales."""
        k = KernelNode(kernel_type=KernelType.RBF, ard_dim=3)
        X = np.random.randn(5, 3).astype(np.float32)
        # ARD RBF: 3 lengthscales + 1 outputscale = 4 params
        params = np.array([1.0, 2.0, 0.5, 1.0])

        K = k.evaluate(X, params=params)
        assert K.shape == (5, 5)
        np.testing.assert_allclose(K, K.T, atol=1e-5)

    def test_positive_definite(self):
        """Kernel matrix should be positive semi-definite."""
        k = Kernel.rbf()
        X = np.random.randn(20, 3).astype(np.float32)
        K = k.evaluate(X, params=np.array([1.0, 1.0]))

        eigenvalues = np.linalg.eigvalsh(K.astype(np.float64))
        assert np.all(eigenvalues >= -1e-6), f"Negative eigenvalue: {eigenvalues.min()}"


# =============================================================================
# StandardScaler
# =============================================================================


class TestStandardScaler:
    """Test StandardScaler normalization utilities."""

    def test_fit_transform_X(self):
        """X should have zero mean and unit variance after transform."""
        rng = np.random.default_rng(42)
        X = rng.normal(loc=5.0, scale=3.0, size=(100, 3)).astype(np.float32)

        scaler = StandardScaler()
        (X_scaled,) = scaler.fit_transform(X)

        np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(X_scaled, axis=0), 1.0, atol=0.1)

    def test_fit_transform_X_and_y(self):
        """Both X and y should be normalized."""
        rng = np.random.default_rng(42)
        X = rng.normal(loc=5.0, scale=3.0, size=(100, 3)).astype(np.float32)
        y = rng.normal(loc=10.0, scale=2.0, size=100).astype(np.float32)

        scaler = StandardScaler()
        X_scaled, y_scaled = scaler.fit_transform(X, y)

        np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.mean(y_scaled), 0.0, atol=0.1)

    def test_inverse_transform_is_identity(self):
        """inverse_transform(transform(y)) should return original y."""
        rng = np.random.default_rng(42)
        X = rng.normal(loc=5.0, scale=3.0, size=(100, 3)).astype(np.float32)
        y = rng.normal(loc=10.0, scale=2.0, size=100).astype(np.float32)

        scaler = StandardScaler()
        X_scaled, y_scaled = scaler.fit_transform(X, y)

        (y_recovered,) = scaler.inverse_transform_y(y_scaled)
        np.testing.assert_allclose(y_recovered, y, atol=1e-3)

    def test_inverse_transform_X_is_identity(self):
        """inverse_transform_X(transform_X(X)) should return original X."""
        rng = np.random.default_rng(42)
        X = rng.normal(loc=5.0, scale=3.0, size=(50, 4)).astype(np.float32)

        scaler = StandardScaler()
        (X_scaled,) = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform_X(X_scaled)

        np.testing.assert_allclose(X_recovered, X, atol=1e-3)

    def test_inverse_transform_std(self):
        """Std should scale by y_std."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2)).astype(np.float32)
        y = rng.normal(loc=0, scale=5.0, size=100).astype(np.float32)

        scaler = StandardScaler()
        _, y_scaled = scaler.fit_transform(X, y)

        std_scaled = np.ones(10, dtype=np.float32)
        _, std_orig = scaler.inverse_transform_y(np.zeros(10), std=std_scaled)

        np.testing.assert_allclose(std_orig, scaler.y_std_, atol=1e-3)

    def test_inverse_transform_variance(self):
        """Variance should scale by y_std^2."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2)).astype(np.float32)
        y = rng.normal(loc=0, scale=5.0, size=100).astype(np.float32)

        scaler = StandardScaler()
        _, y_scaled = scaler.fit_transform(X, y)

        var_scaled = np.ones(10, dtype=np.float32)
        _, var_orig = scaler.inverse_transform_y(np.zeros(10), variance=var_scaled)

        np.testing.assert_allclose(var_orig, scaler.y_std_**2, atol=0.1)

    def test_constant_feature_handled(self):
        """Constant features should not cause division by zero."""
        X = np.array([[1.0, 5.0], [1.0, 3.0], [1.0, 7.0]], dtype=np.float32)

        scaler = StandardScaler()
        (X_scaled,) = scaler.fit_transform(X)

        # First column is constant -> std replaced with 1.0
        assert not np.any(np.isnan(X_scaled))
        assert not np.any(np.isinf(X_scaled))

    def test_not_fitted_raises(self):
        """Using scaler before fit should raise RuntimeError."""
        scaler = StandardScaler()
        with pytest.raises(RuntimeError):
            scaler.transform_X(np.array([[1.0]]))
        with pytest.raises(RuntimeError):
            scaler.transform_y(np.array([1.0]))
        with pytest.raises(RuntimeError):
            scaler.inverse_transform_y(np.array([1.0]))

    def test_no_y_fitted_raises(self):
        """transform_y without fitting y should raise."""
        scaler = StandardScaler()
        scaler.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(RuntimeError):
            scaler.transform_y(np.array([1.0]))


# =============================================================================
# SingleOutputGP Serialization
# =============================================================================


class TestSingleOutputGPSerialization:
    """Test SingleOutputGP save/load without requiring GPU."""

    def test_save_requires_training(self):
        """save() should fail on untrained model."""
        from mojogp.gp import SingleOutputGP

        X = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        gp = SingleOutputGP(Kernel.rbf())
        with pytest.raises(RuntimeError, match="trained"):
            gp.save("/tmp/test_gp")

    def test_load_rejects_legacy_schema(self):
        """load() should reject old pre-RC ExactGP artifacts."""
        from mojogp.gp import SingleOutputGP

        # Create a minimal config file to test
        import json

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model")
            config = {"class": "ExactGP", "dim": 3}
            with open(f"{path}_config.json", "w") as f:
                json.dump(config, f)
            np.savez(
                f"{path}_arrays.npz", X_train=np.zeros((1, 3)), y_train=np.zeros(1)
            )

            with pytest.raises(ValueError, match="SingleOutputGP artifact"):
                SingleOutputGP.load(path)

    def test_save_load_roundtrip(self):
        """save then load should restore training state (mock)."""
        from mojogp.gp import SingleOutputGP, TrainingResult

        X_train = np.random.randn(20, 3).astype(np.float32)
        y_train = np.random.randn(20).astype(np.float32)
        gp = SingleOutputGP(Kernel.rbf() + Kernel.matern52())

        # Manually set data + training state (no GPU needed)
        gp._X_train = X_train
        gp._y_train = y_train
        gp.dim = X_train.shape[1]
        gp._cat_col_indices = []
        gp._cont_dim = gp.dim
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.array([1.0, 0.5, 0.8, 1.2], dtype=np.float32),
            noise=0.1,
            mean=0.0,
            nll=1.5,
            iterations=50,
            converged=True,
            lanczos_root=np.random.randn(20, 5).astype(np.float32),
            lanczos_rank=5,
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model")
            gp.save(path)

            # Check files exist
            assert os.path.exists(f"{path}_config.json")
            assert os.path.exists(f"{path}_arrays.npz")

            # Load
            kernel = Kernel.rbf() + Kernel.matern52()
            gp2 = SingleOutputGP.load(path, kernel=kernel)

            assert gp2.is_trained
            assert gp2.dim == 3
            np.testing.assert_array_equal(gp2._X_train, gp._X_train)
            np.testing.assert_array_equal(gp2._y_train, gp._y_train)
            np.testing.assert_allclose(
                gp2.training_result.params, gp.training_result.params
            )
            assert gp2.training_result.noise == pytest.approx(0.1)
            assert gp2.training_result.nll == pytest.approx(1.5)


# =============================================================================
# SingleOutputGP Posterior Sampling
# =============================================================================


class TestPosteriorSampling:
    """Test sample_posterior for SingleOutputGP (mock, no GPU)."""

    def test_single_output_sample_requires_training(self):
        """sample_posterior should fail on untrained model."""
        from mojogp.gp import SingleOutputGP

        X = np.random.randn(10, 2).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        gp = SingleOutputGP(Kernel.rbf())
        with pytest.raises(RuntimeError, match="trained"):
            gp.sample_posterior(np.random.randn(5, 2).astype(np.float32))

    def test_single_output_sample_rejects_unknown_method(self):
        """sample_posterior should reject unknown methods before backend use."""
        from mojogp.gp import SingleOutputGP

        gp = SingleOutputGP(Kernel.rbf())
        gp._is_trained = True
        with pytest.raises(ValueError, match="method must be"):
            gp.sample_posterior(
                np.random.randn(5, 2).astype(np.float32), method="bogus"
            )


# =============================================================================
# SingleOutputGP Log Marginal Likelihood
# =============================================================================


class TestLogMarginalLikelihood:
    """Test log_marginal_likelihood for SingleOutputGP."""

    def test_single_output_lml_requires_training(self):
        """log_marginal_likelihood should fail on untrained model."""
        from mojogp.gp import SingleOutputGP

        X = np.random.randn(10, 2).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        gp = SingleOutputGP(Kernel.rbf())
        with pytest.raises(RuntimeError, match="trained"):
            gp.log_marginal_likelihood()

    def test_single_output_lml_returns_negative_nll(self):
        """LML should be -NLL."""
        from mojogp.gp import SingleOutputGP, TrainingResult

        X = np.random.randn(10, 2).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        gp = SingleOutputGP(Kernel.rbf())
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.ones(2, dtype=np.float32),
            noise=0.1,
            mean=0.0,
            nll=5.3,
            iterations=50,
            converged=True,
            lanczos_root=np.zeros((1, 1), dtype=np.float32),
            lanczos_rank=0,
        )

        lml = gp.log_marginal_likelihood()
        assert lml == pytest.approx(-5.3)


# =============================================================================
# Composite Kernel Variance Methods
# =============================================================================


class TestCompositeVarianceMethod:
    """Test that SingleOutputGP.predict() supports variance_method for composite kernels."""

    @pytest.fixture
    def trained_composite_gp(self):
        """Train a composite kernel GP for testing."""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("GPU required for composite kernel tests")
        except ImportError:
            pass  # torch not needed for MojoGP, just checking GPU

        from mojogp.gp import SingleOutputGP
        from mojogp.kernel import Kernel

        np.random.seed(42)
        n = 2000
        d = 2
        X = np.random.randn(n, d).astype(np.float32)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n).astype(np.float32)

        kernel = Kernel.rbf() + Kernel.matern52()
        gp = SingleOutputGP(kernel=kernel, verbose=False)
        gp.fit(X, y, max_iterations=50)

        X_test = np.random.randn(10, d).astype(np.float32)
        return gp, X_test

    def test_predict_love_variance(self, trained_composite_gp):
        """Default LOVE variance should produce finite results."""
        gp, X_test = trained_composite_gp
        result = gp.predict(X_test, variance_method="love")
        assert result.mean.shape == (10,)
        assert result.variance.shape == (10,)
        assert np.all(np.isfinite(result.mean))
        assert np.all(np.isfinite(result.variance))
        # Note: LOVE variance quality depends on Lanczos convergence and
        # training quality. With short training, negative values are possible.
        # The key test is that it runs without crashing.

    def test_predict_exact_variance(self, trained_composite_gp):
        """Exact CG variance should work and produce non-negative values."""
        gp, X_test = trained_composite_gp
        result = gp.predict(X_test, variance_method="exact")
        assert result.mean.shape == (10,)
        assert result.variance.shape == (10,)
        assert np.all(np.isfinite(result.mean))
        assert np.all(np.isfinite(result.variance))
        assert np.all(result.variance >= 0), (
            f"Exact CG variance should be non-negative, got min={result.variance.min()}"
        )

    def test_love_and_exact_means_match(self, trained_composite_gp):
        """Both methods should produce the same mean predictions."""
        gp, X_test = trained_composite_gp
        # Hold the alpha-solve budget fixed so this checks variance-method
        # dispatch, not the intentionally different LOVE/exact default CG
        # tolerances.
        solve_kwargs = {"max_cg_iterations": 1000, "cg_tolerance": 1e-3}
        result_love = gp.predict(X_test, variance_method="love", **solve_kwargs)
        result_exact = gp.predict(X_test, variance_method="exact", **solve_kwargs)
        # Means should be close (both use CG alpha solve, but separate runs).
        np.testing.assert_allclose(
            result_love.mean, result_exact.mean, atol=1e-2, rtol=5e-2
        )

    def test_exact_variance_positive(self, trained_composite_gp):
        """Exact CG variance should be strictly positive for non-training points."""
        gp, X_test = trained_composite_gp
        result = gp.predict(X_test, variance_method="exact")
        assert np.all(result.variance > 0), (
            f"Expected all positive variances, got min={result.variance.min()}"
        )

    def test_return_std_with_variance_method(self, trained_composite_gp):
        """predict(return_std=True) returns (mean, std) with variance_method."""
        gp, X_test = trained_composite_gp
        mean, std = gp.predict(X_test, variance_method="exact", return_std=True)
        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert np.all(std >= 0)
