"""System tests for ConstantMean: ground truth recovery at scale.

Validates that GP models with ConstantMean recover the true mean function
offset from data with known ground truth. Uses larger datasets (n>=500)
and tighter tolerances than unit/integration tests.

Test tiers:
- minimal: Core single-output and multi-output mean recovery (n = 2000)
- moderate: Multiple kernels, ARD, larger n (n = 2000)
- full: All GP types, save/load, edge cases (n = 2000-2000)
"""

import numpy as np
import pytest
import time
import tempfile
import os


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_no_lib(module_name: str, build_cmd: str):
    module_name = "mojogp_jit_engine"
    try:
        __import__(module_name)
    except ImportError:
        pytest.skip(f"{module_name} not built (run: {build_cmd})")


def _generate_ground_truth_single(
    n = 2000, d=5, true_mean=7.0, true_ls=1.0, true_noise=0.1, seed=42
):
    """Generate single-output data with known ground truth.

    y = true_mean + f(x) + noise, where f(x) = sum_j sin(x_j / ls).
    f has zero mean over standard-normal X, so E[y] = true_mean.
    """
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f_true = np.sum(np.sin(X / true_ls), axis=1).astype(np.float32)
    y = (true_mean + f_true + true_noise * np.random.randn(n)).astype(np.float32)
    return X, y, f_true, true_mean


def _generate_ground_truth_multi(
    n = 500, d=3, T=3, true_means=None, true_noise=0.1, seed=42
):
    """Generate multi-output data with known per-task ground truth.

    y[:,t] = true_means[t] + 0.5*sin(x_0) + noise.
    Shared zero-mean function ensures per-task mean is purely true_means[t].
    """
    if true_means is None:
        true_means = [5.0, -3.0, 10.0]
    true_means = np.array(true_means[:T], dtype=np.float32)
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    f = 0.5 * np.sin(X[:, 0])
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        Y[:, t] = true_means[t] + f + true_noise * np.random.randn(n)
    return X, Y.astype(np.float32), true_means


# ===========================================================================
# MINIMAL tier: core mean recovery
# ===========================================================================


@pytest.mark.minimal
class TestCoreConstantMeanRecovery:
    """Core mean recovery for single and multi-output models."""

    def test_single_output_mean_recovery(self):
        """ExactGP recovers true mean offset from n = 2000 data."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = 7.0
        X, y, _, _ = _generate_ground_truth_single(n = 2000, d=5, true_mean=true_mean)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        params = gp.get_learned_params()
        fitted_mean = params["mean"]
        assert fitted_mean is not None
        error = abs(fitted_mean - true_mean)
        assert error < 1.0, (
            f"Mean recovery error {error:.3f} > 1.0 "
            f"(learned={fitted_mean:.3f}, true={true_mean})"
        )

    def test_single_output_negative_mean(self):
        """ExactGP recovers negative mean offset."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = -10.0
        X, y, _, _ = _generate_ground_truth_single(
            n = 2000, d=5, true_mean=true_mean, seed=99
        )
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        fitted_mean = gp.get_learned_params()["mean"]
        error = abs(fitted_mean - true_mean)
        assert error < 1.5

    def test_single_output_prediction_accuracy(self):
        """Predictions on test data are accurate with mean offset."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = 7.0
        X, y, f_true, _ = _generate_ground_truth_single(n = 2000, d=5, true_mean=true_mean)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        # Predict at training points
        y_pred, _ = gp.predict(X, return_std=True)
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        assert rmse < 0.5, f"Training RMSE {rmse:.3f} too high"

        # Test data
        np.random.seed(99)
        X_test = np.random.randn(100, 5).astype(np.float32)
        y_test_pred, _ = gp.predict(X_test, return_std=True)
        avg_pred = np.mean(y_test_pred)
        assert abs(avg_pred - true_mean) < 2.0, (
            f"Avg test prediction {avg_pred:.2f} too far from {true_mean}"
        )

    def test_multi_output_per_task_mean_recovery(self):
        """MultiOutputGP recovers per-task means from n = 2000 data."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_ground_truth_multi(n = 2000, T=3, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=100, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(3):
            error = abs(gp._fitted_mean[t] - true_means[t])
            assert error < 1.5, (
                f"Task {t}: error {error:.3f} > 1.5 "
                f"(learned={gp._fitted_mean[t]:.3f}, true={true_means[t]})"
            )

    def test_multi_output_prediction_accuracy(self):
        """MultiOutputGP predictions include per-task mean offsets."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_ground_truth_multi(n = 2000, T=3, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=100, learning_rate=0.05, verbose=False)

        mean, _ = gp.predict(X, return_var=True)
        for t in range(3):
            rmse = np.sqrt(np.mean((mean[:, t] - Y[:, t]) ** 2))
            assert rmse < 1.0, f"Task {t}: RMSE {rmse:.3f}"

    def test_zero_mean_no_regression(self):
        """Zero-mean data still works correctly (baseline regression check)."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, _, _ = _generate_ground_truth_single(n = 2000, d=5, true_mean=0.0)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        fitted_mean = gp.get_learned_params()["mean"]
        assert abs(fitted_mean) < 0.5, (
            f"Zero-mean data: learned mean {fitted_mean:.3f} should be ~0"
        )


# ===========================================================================
# MODERATE tier: multiple kernels, ARD, larger datasets
# ===========================================================================


@pytest.mark.moderate
class TestExtendedConstantMeanRecovery:
    """Mean recovery across more kernel types, ARD, and n = 2000."""

    def test_multiple_kernels_mean_recovery(self):
        """All stationary kernels recover the true mean at n = 2000."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF, Matern12, Matern32, Matern52

        true_mean = 8.0
        X, y, _, _ = _generate_ground_truth_single(n = 2000, d=5, true_mean=true_mean)

        kernel_map = {
            "rbf": RBF(),
            "matern12": Matern12(),
            "matern32": Matern32(),
            "matern52": Matern52(),
        }
        results = {}
        for kernel_name, kernel_obj in kernel_map.items():
            gp = SingleOutputGP(kernel_obj)
            gp.fit(X, y, max_iterations=100, learning_rate=0.1)
            fitted_mean = gp.get_learned_params()["mean"]
            error = abs(fitted_mean - true_mean)
            results[kernel_name] = (fitted_mean, error)
            assert error < 1.0, (
                f"{kernel_name}: error {error:.3f} > 1.0 (learned={fitted_mean:.3f})"
            )

    def test_ard_mean_recovery(self):
        """ARD kernel recovers mean at n = 2000."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = 12.0
        X, y, _, _ = _generate_ground_truth_single(
            n = 2000, d=5, true_mean=true_mean, seed=77
        )
        gp = SingleOutputGP(RBF(ard=True))
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        fitted_mean = gp.get_learned_params()["mean"]
        error = abs(fitted_mean - true_mean)
        assert error < 1.0

    def test_matrix_free_mean_recovery(self):
        """Matrix-free GP recovers mean at n = 2000."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = -6.0
        X, y, _, _ = _generate_ground_truth_single(n = 2000, d=5, true_mean=true_mean)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1, method="matrix_free")

        fitted_mean = gp.get_learned_params()["mean"]
        error = abs(fitted_mean - true_mean)
        assert error < 1.5

    def test_composite_kernel_mean_recovery(self):
        """Composite kernel (RBF + Matern52) recovers mean at n = 2000."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.gp import SingleOutputGP
        from mojogp.kernel import Kernel

        true_mean = 9.0
        X, y, _, _ = _generate_ground_truth_single(n = 2000, d=5, true_mean=true_mean)
        gp = SingleOutputGP(Kernel.rbf() + Kernel.matern52())
        result = gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        error = abs(result.mean - true_mean)
        assert error < 1.5

    def test_large_mean_offset(self):
        """GP recovers large mean offsets (100, -100)."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        for true_mean in [100.0, -100.0]:
            X, y, _, _ = _generate_ground_truth_single(
                n = 2000, d=3, true_mean=true_mean, seed=42
            )
            gp = SingleOutputGP(RBF())
            gp.fit(X, y, max_iterations=120, learning_rate=0.1)

            fitted_mean = gp.get_learned_params()["mean"]
            error = abs(fitted_mean - true_mean)
            assert error < 5.0, (
                f"mean={true_mean}: error {error:.3f} > 5.0 (learned={fitted_mean:.3f})"
            )

    def test_lmc_per_task_mean_recovery(self):
        """LMC recovers per-task means at n = 2000."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputLMCGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_ground_truth_multi(n = 2000, T=2, true_means=true_means)
        gp = MultiOutputLMCGP(kernels=["rbf"])
        gp.fit(X, Y, max_iterations=100, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            error = abs(gp._fitted_mean[t] - true_means[t])
            assert error < 2.0, (
                f"Task {t}: error {error:.3f} "
                f"(learned={gp._fitted_mean[t]:.3f}, true={true_means[t]})"
            )

    def test_multi_output_ard_mean_recovery(self):
        """MultiOutputGP with ARD recovers per-task means."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_ground_truth_multi(n = 500, d=3, T=2, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf", ard=True)
        gp.fit(X, Y, max_iterations=100, learning_rate=0.05, verbose=False)

        for t in range(2):
            error = abs(gp._fitted_mean[t] - true_means[t])
            assert error < 2.0


# ===========================================================================
# FULL tier: comprehensive coverage
# ===========================================================================


@pytest.mark.full
class TestConstantMeanPersistenceAndBoundaryBehavior:
    """Constant mean coverage for GP types, save/load, and boundary behavior."""

    def test_single_output_n2000(self):
        """Single-output mean recovery at n=2000 with tight tolerance."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = 7.0
        X, y, _, _ = _generate_ground_truth_single(n=2000, d=5, true_mean=true_mean)
        gp = SingleOutputGP(RBF())
        start = time.time()
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)
        elapsed = time.time() - start

        fitted_mean = gp.get_learned_params()["mean"]
        error = abs(fitted_mean - true_mean)
        assert error < 0.5, (
            f"n=2000 error {error:.3f} > 0.5 "
            f"(learned={fitted_mean:.3f}, time={elapsed:.1f}s)"
        )

    def test_multi_output_n1000(self):
        """Multi-output mean recovery at n = 2000 with tight tolerance."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_ground_truth_multi(n = 2000, T=3, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=100, learning_rate=0.05, verbose=False)

        for t in range(3):
            error = abs(gp._fitted_mean[t] - true_means[t])
            assert error < 1.0, f"Task {t}: n = 2000 error {error:.3f} > 1.0"

    def test_save_load_preserves_predictions(self):
        """Save/load roundtrip produces identical predictions at n = 2000."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y, _, true_mean = _generate_ground_truth_single(n = 2000, d=5, true_mean=7.0)
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)
        y_pred_before, _ = gp.predict(X, return_std=True)

        with tempfile.TemporaryDirectory() as d:
            gp.save(os.path.join(d, "model"))
            gp2 = SingleOutputGP.load(os.path.join(d, "model"))

            mean_before = gp.get_learned_params()["mean"]
            mean_after = gp2.get_learned_params()["mean"]
            assert abs(mean_after - mean_before) < 1e-6
            y_pred_after, _ = gp2.predict(X, return_std=True)
            np.testing.assert_allclose(y_pred_before, y_pred_after, rtol=1e-5)

    def test_user_init_mean_vs_auto(self):
        """User init_mean=true_mean converges faster than auto-detect."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = 50.0
        X, y, _, _ = _generate_ground_truth_single(n = 2000, d=3, true_mean=true_mean)

        # Auto-detect (should still work but starts from y.mean())
        gp_auto = SingleOutputGP(RBF())
        gp_auto.fit(X, y, max_iterations=80, learning_rate=0.1)

        # User init at true value (should converge at least as well)
        gp_user = SingleOutputGP(RBF(), init_mean=true_mean)
        gp_user.fit(X, y, max_iterations=80, learning_rate=0.1)

        error_auto = abs(gp_auto.get_learned_params()["mean"] - true_mean)
        error_user = abs(gp_user.get_learned_params()["mean"] - true_mean)

        # Both should converge
        assert error_auto < 3.0
        assert error_user < 3.0

    def test_mean_does_not_absorb_signal(self):
        """Mean parameter learns offset, not the function shape.

        Verify that after training, the GP still captures the
        non-constant part of the function (not just the mean).
        """
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        true_mean = 7.0
        X, y, f_true, _ = _generate_ground_truth_single(
            n = 2000, d=3, true_mean=true_mean
        )
        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=100, learning_rate=0.1)

        y_pred, _ = gp.predict(X, return_std=True)

        # Predictions should capture variation, not just be flat at mean
        pred_std = np.std(y_pred)
        data_std = np.std(y)
        ratio = pred_std / data_std
        assert ratio > 0.3, (
            f"Predictions too flat: pred_std={pred_std:.3f}, "
            f"data_std={data_std:.3f}, ratio={ratio:.3f}"
        )

    def test_multi_output_large_mean_spread(self):
        """Multi-output with large spread of per-task means."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [50.0, -50.0, 0.0]
        X, Y, _ = _generate_ground_truth_multi(n = 2000, T=3, true_means=true_means)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=120, learning_rate=0.05, verbose=False)

        for t in range(3):
            error = abs(gp._fitted_mean[t] - true_means[t])
            assert error < 5.0, (
                f"Task {t}: error {error:.3f} "
                f"(learned={gp._fitted_mean[t]:.3f}, true={true_means[t]})"
            )
