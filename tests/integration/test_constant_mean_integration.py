"""Integration tests for ConstantMean across all GP types.

Tests the full train -> predict pipeline with non-zero-mean data,
verifying that ConstantMean works end-to-end for:
1. ExactGP single-output (materialized + matrix-free)
2. ExactGP composite kernels
3. MultiOutputGP (Kronecker)
4. MultiOutputLMCGP (LMC)
5. Save/load roundtrip preserves mean
6. ARD + ConstantMean interaction
"""

import numpy as np
import pytest
import tempfile
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_no_lib(module_name: str, build_cmd: str):
    """Skip test if the public-wrapper runtime is not available."""
    module_name = "mojogp_jit_engine"
    try:
        __import__(module_name)
    except ImportError:
        pytest.skip(f"{module_name} not built (run: {build_cmd})")


def _generate_shifted_sinusoid(n=2000, d=3, mean_offset=7.0, noise_std=0.1, seed=42):
    """Single-output data: y = mean_offset + sin(x_0) + noise."""
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f_true = np.sin(X[:, 0])
    y = (mean_offset + f_true + noise_std * np.random.randn(n)).astype(np.float32)
    return X, y, mean_offset


def _generate_multi_output_shifted(
    n=500, d=3, T=3, true_means=None, noise_std=0.1, seed=42
):
    """Multi-output data: y[:,t] = true_means[t] + 0.5*sin(x_0) + noise."""
    if true_means is None:
        true_means = [5.0, -3.0, 10.0]
    true_means = np.array(true_means[:T], dtype=np.float32)
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f = 0.5 * np.sin(X[:, 0])
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        Y[:, t] = true_means[t] + f + noise_std * np.random.randn(n)
    return X, Y.astype(np.float32), true_means


# ===========================================================================
# 1. ExactGP single-output integration
# ===========================================================================


class TestExactGPSingleOutputConstantMeanIntegration:
    """Full train->predict pipeline for ExactGP single-output with ConstantMean."""

    def test_materialized_nonzero_mean(self):
        """Materialized GP trains and predicts on shifted data."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, true_mean = _generate_shifted_sinusoid(n=2000, mean_offset=7.0)
        gp = SingleOutputGP(RBF())
        result = gp.fit(
            X, y, max_iterations=80, learning_rate=0.1, method="materialized"
        )

        # Mean should be learned
        assert result.mean is not None
        assert abs(result.mean - true_mean) < 2.0, (
            f"Learned mean {result.mean:.2f} too far from {true_mean}"
        )

        # Predictions should be near data
        y_pred, y_std = gp.predict(X, return_std=True)
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        assert rmse < 1.0, f"Training RMSE {rmse:.3f} too high"

    def test_matrix_free_nonzero_mean(self):
        """Matrix-free GP trains and predicts on shifted data."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, true_mean = _generate_shifted_sinusoid(n=2000, mean_offset=-5.0)
        gp = SingleOutputGP(RBF())
        result = gp.fit(
            X, y, max_iterations=80, learning_rate=0.1, method="matrix_free"
        )

        assert result.mean is not None
        assert abs(result.mean - true_mean) < 2.0

        y_pred, y_std = gp.predict(X, return_std=True)
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        assert rmse < 1.0, f"Matrix-free RMSE {rmse:.3f} too high"

    def test_ard_nonzero_mean(self):
        """ARD kernel + ConstantMean on shifted data."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, true_mean = _generate_shifted_sinusoid(
            n=2000, d=5, mean_offset=12.0, seed=99
        )
        gp = SingleOutputGP(RBF(ard=True))
        result = gp.fit(X, y, max_iterations=80, learning_rate=0.1)

        assert result.mean is not None
        assert abs(result.mean - true_mean) < 3.0

    def test_multiple_kernels_learn_mean(self):
        """All stationary kernels learn the mean offset."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF, Matern12, Matern32, Matern52

        X, y, true_mean = _generate_shifted_sinusoid(n=2000, mean_offset=8.0)

        kernels = [
            ("rbf", RBF()),
            ("matern32", Matern32()),
            ("matern52", Matern52()),
            ("matern12", Matern12()),
        ]
        for kernel_name, kernel_obj in kernels:
            gp = SingleOutputGP(kernel_obj)
            result = gp.fit(X, y, max_iterations=80, learning_rate=0.1)
            assert result.mean is not None, f"{kernel_name}: no fitted mean"
            assert abs(result.mean - true_mean) < 3.0, (
                f"{kernel_name}: learned {result.mean:.2f}, expected ~{true_mean}"
            )

    def test_prediction_on_unseen_data(self):
        """Predictions on test data have correct offset."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, true_mean = _generate_shifted_sinusoid(n=2000, mean_offset=15.0, seed=42)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        np.random.seed(99)
        X_test = np.random.randn(50, 3).astype(np.float32)
        y_pred, y_std = gp.predict(X_test, return_std=True)

        # Average prediction should be near true_mean (since sin averages ~0)
        avg_pred = np.mean(y_pred)
        assert abs(avg_pred - true_mean) < 3.0, (
            f"Avg prediction {avg_pred:.2f} too far from mean {true_mean}"
        )


# ===========================================================================
# 2. ExactGP composite kernel integration
# ===========================================================================


class TestExactGPConstantMeanIntegration:
    """Full train->predict for ExactGP (composite kernels) with ConstantMean."""

    def test_sum_kernel_learns_mean(self):
        """RBF + Matern52 composite learns the mean offset."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.gp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y, true_mean = _generate_shifted_sinusoid(n=2000, mean_offset=6.0)
        gp = SingleOutputGP(Kernel.rbf() + Kernel.matern52())
        result = gp.fit(X, y, max_iterations=80, learning_rate=0.1)

        assert result.mean is not None
        assert abs(result.mean - true_mean) < 2.5

        mean, std = gp.predict(X, return_std=True)
        rmse = np.sqrt(np.mean((mean - y) ** 2))
        assert rmse < 1.0

    def test_product_kernel_learns_mean(self):
        """RBF * Matern52 composite learns the mean offset."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.gp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y, true_mean = _generate_shifted_sinusoid(n=2000, mean_offset=-4.0)
        gp = SingleOutputGP(Kernel.rbf() * Kernel.matern52())
        result = gp.fit(X, y, max_iterations=80, learning_rate=0.1)

        assert result.mean is not None
        assert abs(result.mean - true_mean) < 3.0

    def test_user_init_mean_composite(self):
        """User-specified init_mean works with composite kernels."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.gp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y, true_mean = _generate_shifted_sinusoid(n=2000, mean_offset=5.0)
        gp = SingleOutputGP(Kernel.rbf() + Kernel.matern52(), init_mean=5.0)
        result = gp.fit(X, y, max_iterations=60, learning_rate=0.1)

        assert abs(result.mean - true_mean) < 2.0


# ===========================================================================
# 3. MultiOutputGP integration
# ===========================================================================


class TestMultiOutputGPConstantMeanIntegration:
    """Full train->predict for MultiOutputGP with per-task ConstantMean."""

    def test_kronecker_learns_per_task_means(self):
        """Kronecker GP learns per-task mean offsets."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_multi_output_shifted(n=2000, T=3, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(3):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.0, (
                f"Task {t}: learned {gp._fitted_mean[t]:.2f}, expected ~{true_means[t]}"
            )

    def test_kronecker_prediction_accuracy(self):
        """Kronecker predictions on training data are accurate."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_multi_output_shifted(n=2000, T=3, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        mean, var = gp.predict(X, return_var=True)
        assert mean.shape == (X.shape[0], 3)
        for t in range(3):
            rmse = np.sqrt(np.mean((mean[:, t] - Y[:, t]) ** 2))
            assert rmse < 1.5, f"Task {t}: RMSE {rmse:.3f}"

    def test_kronecker_ard_with_mean(self):
        """ARD + ConstantMean works together for multi-output."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_multi_output_shifted(n=500, d=3, T=2, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf", ard=True)
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.5


# ===========================================================================
# 4. MultiOutputLMCGP integration
# ===========================================================================


class TestLMCConstantMeanIntegration:
    """Full train->predict for LMC GP with per-task ConstantMean."""

    def test_lmc_learns_per_task_means(self):
        """LMC learns per-task mean offsets."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_multi_output_shifted(n=2000, T=2, true_means=true_means)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.5

    def test_lmc_prediction_with_mean(self):
        """LMC predictions include mean offset."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_multi_output_shifted(n=2000, T=2, true_means=true_means)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        mean, var = gp.predict(X, return_var=True)
        assert mean.shape == (X.shape[0], 2)
        for t in range(2):
            avg = np.mean(mean[:, t])
            assert abs(avg - true_means[t]) < 3.0

    def test_lmc_two_latents_with_mean(self):
        """LMC with 2 latent GPs learns per-task means."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [8.0, -2.0]
        X, Y, _ = _generate_multi_output_shifted(n=2000, T=2, true_means=true_means)
        gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 3.0


# ===========================================================================
# 5. Save/load roundtrip integration
# ===========================================================================


class TestConstantMeanSaveLoadIntegration:
    """Verify mean is preserved through save/load for all GP types."""

    def test_exactgp_single_output_save_load_preserves_mean(self):
        """ExactGP single-output save/load roundtrip preserves fitted mean."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, _ = _generate_shifted_sinusoid(n=2000, mean_offset=7.0)
        kernel = RBF()
        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=80, learning_rate=0.1)

        y_pred_before, _ = gp.predict(X, return_std=True, variance_method="exact")

        with tempfile.TemporaryDirectory() as d:
            gp.save(os.path.join(d, "model"))
            gp2 = SingleOutputGP.load(os.path.join(d, "model"), kernel=kernel)

            assert gp2._training_result.mean is not None
            assert abs(gp2._training_result.mean - gp._training_result.mean) < 1e-6

            y_pred_after, _ = gp2.predict(X, return_std=True, variance_method="exact")
            np.testing.assert_allclose(
                y_pred_before, y_pred_after, rtol=5e-3, atol=3e-2
            )

    def test_exactgp_composite_save_load_preserves_mean(self):
        """ExactGP composite save/load roundtrip preserves fitted mean."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.gp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y, _ = _generate_shifted_sinusoid(n=2000, mean_offset=6.0)
        kernel = Kernel.rbf() + Kernel.matern52()
        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=80, learning_rate=0.1)

        mean_before, _ = gp.predict(X, return_std=True, variance_method="exact")

        with tempfile.TemporaryDirectory() as d:
            gp.save(os.path.join(d, "model"))
            gp2 = SingleOutputGP.load(os.path.join(d, "model"), kernel=kernel)

            assert gp2._training_result.mean is not None
            np.testing.assert_allclose(
                gp2._training_result.mean,
                gp._training_result.mean,
                rtol=1e-5,
            )

            mean_after, _ = gp2.predict(X, return_std=True, variance_method="exact")
            np.testing.assert_allclose(
                mean_before, mean_after, rtol=5e-3, atol=3e-2
            )

    def test_multi_output_save_load_preserves_mean(self):
        """MultiOutputGP save/load preserves per-task fitted mean."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_multi_output_shifted(n=2000, T=2, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=60, learning_rate=0.05, verbose=False)

        mean_before, _ = gp.predict(X, return_var=True)

        with tempfile.TemporaryDirectory() as d:
            gp.save(os.path.join(d, "model"))
            gp2 = MultiOutputGP.load(os.path.join(d, "model"))

            assert gp2._fitted_mean is not None
            np.testing.assert_allclose(gp2._fitted_mean, gp._fitted_mean, rtol=1e-5)

            mean_after, _ = gp2.predict(X, return_var=True)
            np.testing.assert_allclose(mean_before, mean_after, rtol=1e-4)

    def test_lmc_save_load_preserves_mean(self):
        """LMC GP save/load preserves per-task fitted mean."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_multi_output_shifted(n=2000, T=2, true_means=true_means)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=60, learning_rate=0.05, verbose=False)

        pred_before = gp.predict(X)
        mean_before = pred_before.mean

        with tempfile.TemporaryDirectory() as d:
            gp.save(os.path.join(d, "model"))
            gp2 = MultiOutputLMCGP.load(os.path.join(d, "model"))

            assert gp2._fitted_mean is not None
            np.testing.assert_allclose(gp2._fitted_mean, gp._fitted_mean, rtol=1e-5)

            pred_after = gp2.predict(X)
            mean_after = pred_after.mean
            np.testing.assert_allclose(mean_before, mean_after, rtol=1e-4)


# ===========================================================================
# 6. Zero-mean backward compatibility
# ===========================================================================


class TestZeroMeanBackwardCompatibility:
    """Verify that zero-mean data still works correctly (no regression)."""

    def test_exactgp_zero_mean_unchanged(self):
        """Zero-mean data produces near-zero learned mean."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, _ = _generate_shifted_sinusoid(n=2000, mean_offset=0.0)
        gp = SingleOutputGP(RBF())
        result = gp.fit(X, y, max_iterations=80, learning_rate=0.1)

        assert abs(result.mean) < 1.0

    def test_multi_output_zero_mean_unchanged(self):
        """Zero-mean multi-output data produces near-zero per-task means."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y, _ = _generate_multi_output_shifted(n=2000, T=2, true_means=[0.0, 0.0])
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=60, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t]) < 1.0
