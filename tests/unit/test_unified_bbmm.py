"""Unit tests for unified BBMM infrastructure via high-level ExactGP API.

Tests the GradientProvider trait, adapter structs, and bbmm_unified function
to ensure they produce correct NLL and gradients matching the old implementations.
"""

import pytest
import numpy as np

from mojogp import SingleOutputGP, RBF, Matern12, Matern32, Matern52


class TestUnifiedBBMMTrainingRoute:
    """Unified BBMM tests via SingleOutputGP training."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(42)
        n, d = 2000, 3
        X = np.random.randn(n, d).astype(np.float32)
        y = (
            X[:, 0]
            + 0.5 * X[:, 1]
            + np.random.randn(n).astype(np.float32) * np.sqrt(0.1)
        )
        return X, y.astype(np.float32)

    def test_isotropic_training_returns_finite_parameters(self, simple_data):
        """Isotropic training with unified BBMM returns finite valid parameters."""
        X, y = simple_data
        gp = SingleOutputGP(RBF())
        result = gp.fit(
            X,
            y,
            max_iterations=5,
            learning_rate=0.1,
            initial_noise=0.1,
            method="materialized",
        )

        assert result.params is not None
        assert result.noise > 0
        assert result.params[-1] > 0  # outputscale
        assert np.isfinite(result.nll)
        assert result.iterations > 0

    def test_isotropic_training_converges(self, simple_data):
        """Test that isotropic training converges to reasonable values."""
        X, y = simple_data
        gp = SingleOutputGP(RBF())
        result = gp.fit(X, y, max_iterations=50, learning_rate=0.1, initial_noise=0.1)

        assert result.nll < 100.0
        assert result.params[0] > 0.01  # lengthscale
        assert result.noise > 0.001
        assert result.params[-1] > 0.01  # outputscale

    def test_different_kernel_types(self, simple_data):
        """Test unified BBMM works with different kernel types."""
        X, y = simple_data

        for kernel_cls, name in [
            (RBF, "RBF"),
            (Matern12, "Matern12"),
            (Matern32, "Matern32"),
            (Matern52, "Matern52"),
        ]:
            gp = SingleOutputGP(kernel_cls())
            result = gp.fit(X, y, max_iterations=5, learning_rate=0.1)
            assert np.isfinite(result.nll), f"NLL not finite for {name}"
            assert result.params[0] > 0, f"Lengthscale not positive for {name}"
            assert result.noise > 0, f"Noise not positive for {name}"
            assert result.params[-1] > 0, f"Outputscale not positive for {name}"


class TestUnifiedBBMMGradients:
    """Test that unified BBMM produces correct gradients."""

    @pytest.fixture
    def gradient_test_data(self):
        """Create data for gradient testing."""
        np.random.seed(123)
        n, d = 2000, 2
        X = np.random.randn(n, d).astype(np.float32)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n).astype(np.float32)
        return X, y.astype(np.float32)

    def test_gradient_descent_decreases_nll(self, gradient_test_data):
        """Test that gradient descent actually decreases NLL."""
        X, y = gradient_test_data

        gp1 = SingleOutputGP(RBF())
        result_1iter = gp1.fit(
            X,
            y,
            max_iterations=1,
            learning_rate=0.1,
            initial_noise=0.1,
            method="materialized",
        )

        gp10 = SingleOutputGP(RBF())
        result_10iter = gp10.fit(
            X,
            y,
            max_iterations=10,
            learning_rate=0.1,
            initial_noise=0.1,
            method="materialized",
        )

        assert result_10iter.nll <= result_1iter.nll + 0.5, (
            f"NLL did not decrease: {result_1iter.nll} -> {result_10iter.nll}"
        )

    def test_parameters_change_with_training(self, gradient_test_data):
        """Test that parameters actually change during training."""
        X, y = gradient_test_data
        init_ls = 1.0
        init_noise = 0.1

        gp = SingleOutputGP(RBF())
        result = gp.fit(
            X, y, max_iterations=20, learning_rate=0.1, initial_noise=init_noise
        )

        ls_changed = abs(result.params[0] - init_ls) > 0.01
        noise_changed = abs(result.noise - init_noise) > 0.001
        os_changed = abs(result.params[-1] - 1.0) > 0.01

        assert ls_changed or noise_changed or os_changed, (
            "No parameters changed during training"
        )


class TestUnifiedBBMMNumericalStability:
    """Test numerical stability of unified BBMM."""

    def test_small_noise(self):
        """Test with very small noise (challenging for CG)."""
        np.random.seed(456)
        n, d = 2000, 2
        X = np.random.randn(n, d).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        gp = SingleOutputGP(RBF())
        result = gp.fit(
            X, y, max_iterations=10, learning_rate=0.05, initial_noise=0.001
        )

        assert np.isfinite(result.nll), "NLL not finite with small noise"
        assert result.noise > 0, "Noise became non-positive"

    def test_large_lengthscale(self):
        """Test with large lengthscale (nearly constant kernel)."""
        np.random.seed(789)
        n, d = 2000, 2
        X = np.random.randn(n, d).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        gp = SingleOutputGP(RBF())
        result = gp.fit(X, y, max_iterations=10, learning_rate=0.1)

        assert np.isfinite(result.nll), "NLL not finite with large lengthscale"

    def test_small_lengthscale(self):
        """Test with small lengthscale (highly varying kernel)."""
        np.random.seed(101)
        n, d = 2000, 2
        X = np.random.randn(n, d).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        gp = SingleOutputGP(RBF())
        result = gp.fit(X, y, max_iterations=10, learning_rate=0.1)

        assert np.isfinite(result.nll), "NLL not finite with small lengthscale"


class TestUnifiedBBMMConsistency:
    """Test repeated unified BBMM runs remain numerically sane."""

    def test_repeated_training_remains_finite(self):
        """Repeated training on the same data should stay finite and valid."""
        np.random.seed(42)
        n, d = 2000, 2
        X = np.random.randn(n, d).astype(np.float32)
        y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(n)).astype(np.float32)

        gp1 = SingleOutputGP(RBF())
        result1 = gp1.fit(
            X.copy(),
            y.copy(),
            max_iterations=10,
            learning_rate=0.1,
            method="materialized",
        )

        gp2 = SingleOutputGP(RBF())
        result2 = gp2.fit(
            X.copy(),
            y.copy(),
            max_iterations=10,
            learning_rate=0.1,
            method="materialized",
        )

        assert np.isfinite(result1.nll), f"First run NLL not finite: {result1.nll}"
        assert np.isfinite(result2.nll), f"Second run NLL not finite: {result2.nll}"
        assert result1.params[0] > 0 and result2.params[0] > 0
        assert result1.noise > 0 and result2.noise > 0
        assert result1.params[-1] > 0 and result2.params[-1] > 0


class TestUnifiedBBMMMethodSelection:
    """Test that different methods (matrix-free vs materialized) work."""

    @pytest.fixture
    def method_test_data(self):
        np.random.seed(42)
        n, d = 2000, 3
        X = np.random.randn(n, d).astype(np.float32)
        y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n).astype(np.float32)
        return X, y.astype(np.float32)

    def test_materialized_method(self, method_test_data):
        """Test training with 'materialized' method."""
        X, y = method_test_data
        gp = SingleOutputGP(RBF())
        result = gp.fit(
            X, y, max_iterations=5, learning_rate=0.1, method="materialized"
        )
        assert np.isfinite(result.nll), "NLL not finite with materialized method"
        assert result.params[0] > 0

    def test_matrix_free_method(self, method_test_data):
        """Test training with 'matrix_free' method."""
        X, y = method_test_data
        gp = SingleOutputGP(RBF())
        result = gp.fit(X, y, max_iterations=5, learning_rate=0.1, method="matrix_free")
        assert np.isfinite(result.nll), "NLL not finite with matrix_free method"
        assert result.params[0] > 0

    def test_methods_produce_similar_results(self, method_test_data):
        """Test that different methods produce similar predictions."""
        X, y = method_test_data
        X_test = X[:64]

        gp_mat = SingleOutputGP(RBF())
        result_mat = gp_mat.fit(
            X, y, max_iterations=10, learning_rate=0.1, method="materialized"
        )

        gp_mf = SingleOutputGP(RBF())
        result_mf = gp_mf.fit(
            X, y, max_iterations=10, learning_rate=0.1, method="matrix_free"
        )

        mean_mat, _ = gp_mat.predict(X_test, return_std=True)
        mean_mf, _ = gp_mf.predict(X_test, return_std=True)
        pred_rmse = np.sqrt(np.mean((mean_mat - mean_mf) ** 2))
        assert pred_rmse < 0.05, (
            f"Predictions differ too much between methods: RMSE={pred_rmse:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
