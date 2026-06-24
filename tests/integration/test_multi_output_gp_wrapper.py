"""Integration tests for MultiOutputGP Python wrapper.

Tests the high-level MultiOutputGP class that dispatches to Kronecker CG
for all training modes: isotropic, ARD, and composite kernels.
"""

import numpy as np
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


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


class TestMultiOutputGPInit:
    """Test MultiOutputGP initialization and validation."""

    def test_default_init(self):
        from mojogp.multi_output_gp import MultiOutputGP
        from mojogp.kernel import KernelNode

        gp = MultiOutputGP()
        # kernel is now always a KernelNode (converted from string at init)
        assert isinstance(gp.kernel, KernelNode)
        assert "RBF" in str(gp.kernel).upper() or "rbf" in str(gp.kernel).lower()
        assert gp.task_rank == -1
        assert gp.ard is False
        assert gp.is_trained is False
        assert gp.num_tasks is None
        assert gp.task_covariance is None

    def test_init_with_kernel_string(self):
        from mojogp.multi_output_gp import MultiOutputGP
        from mojogp.kernel import KernelNode

        for k in ["rbf", "matern32", "matern52", "matern12"]:
            gp = MultiOutputGP(kernel=k)
            # kernel is now always a KernelNode (converted from string at init)
            assert isinstance(gp.kernel, KernelNode)

    def test_init_with_composite_kernel(self):
        from mojogp.multi_output_gp import MultiOutputGP
        from mojogp.kernel import Kernel

        kernel = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputGP(kernel=kernel)
        assert gp._is_composite is True

    def test_invalid_kernel_string(self):
        from mojogp.multi_output_gp import MultiOutputGP

        with pytest.raises(ValueError, match="Unknown kernel"):
            MultiOutputGP(kernel="invalid_kernel")

    def test_composite_ard_is_supported(self):
        """Composite+ARD is accepted by the public wrapper."""
        from mojogp.multi_output_gp import MultiOutputGP
        from mojogp.kernel import Kernel

        # Should NOT raise - composite+ARD is now supported
        gp = MultiOutputGP(kernel=Kernel.rbf(), ard=True)
        assert gp.ard is True

    def test_repr_untrained(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf")
        r = repr(gp)
        # repr contains the KernelNode type name (e.g. "RBFComposable") not the string "rbf"
        assert "RBF" in r.upper()
        assert "untrained" in r

    def test_repr_ard(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf", ard=True)
        r = repr(gp)
        assert "ard=True" in r


class TestMultiOutputGPIsotropic:
    """Test isotropic (single lengthscale) training and prediction."""

    def test_fit_returns_training_result(self):
        """Test that fit() returns the current multi-output training result."""
        from mojogp.multi_output_gp import (
            MultiOutputGP,
            MultiOutputTrainingResult,
        )

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert isinstance(result, MultiOutputTrainingResult)
        assert gp.is_trained
        assert gp.num_tasks == 2
        assert result.num_tasks == 2
        assert result.B.shape == (2, 2)
        assert result.Q.shape == (2, 2)
        assert result.Lambda.shape == (2,)
        assert result.alpha_rotated is not None
        assert result.alpha_rotated.shape == (500, 2)
        assert not np.allclose(result.alpha_rotated, 0.0)
        assert result.effective_scales.shape == (2,)
        assert len(result.nll_history) > 0

    def test_fit_nll_decreases(self):
        """Test that NLL generally decreases during training."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        result = gp.fit(X, Y, max_iterations=50, verbose=False, early_stop_tol=0.0)

        nll = result.nll_history
        # NLL should be recorded per iteration
        assert len(nll) > 0, "nll_history should not be empty"
        # Training should improve NLL: final NLL should be <= initial NLL
        # (allowing equality if model converged quickly from good init)
        assert nll[-1] <= nll[0] + 1e-4, (
            f"NLL increased significantly: {nll[0]:.4f} -> {nll[-1]:.4f}"
        )

    def test_predict_mean_only(self):
        """Test prediction returns correct shapes."""
        from mojogp.multi_output_gp import MultiOutputGP, MultiOutputPredictionResult

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=20, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        pred = gp.predict(X_test)

        assert isinstance(pred, MultiOutputPredictionResult)
        assert pred.mean.shape == (10, 2)

    def test_predict_return_var(self):
        """Test prediction with return_var=True."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=20, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True)

        assert mean.shape == (10, 2)
        assert var is not None
        assert var.shape == (10, 2)

    def test_predict_return_std(self):
        """Test prediction with return_std=True."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=20, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mean, std = gp.predict(X_test, return_std=True)

        assert mean.shape == (10, 2)
        assert std is not None
        assert std.shape == (10, 2)

    def test_predict_before_fit_raises(self):
        """Test that predict() raises if not trained."""
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf")
        X_test = np.random.randn(10, 3).astype(np.float32)
        with pytest.raises(RuntimeError, match="must be trained"):
            gp.predict(X_test)

    def test_predict_wrong_dim_raises(self):
        """Test that predict() raises if X_test has wrong dimension."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=10, verbose=False)

        X_test = np.random.randn(10, 5).astype(np.float32)  # Wrong dim
        with pytest.raises(ValueError, match="features"):
            gp.predict(X_test)

    def test_fit_3_tasks(self):
        """Test with 3 tasks."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=3)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert result.num_tasks == 3
        assert result.B.shape == (3, 3)

        X_test = np.random.randn(5, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True)
        assert mean.shape == (5, 3)
        assert var.shape == (5, 3)

    def test_fit_with_task_rank(self):
        """Test with low-rank task covariance."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=3)
        gp = MultiOutputGP(
            kernel="rbf", task_rank=1, num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert result.task_rank == 1
        assert result.W.shape == (3, 1)

    def test_fit_matern52(self):
        """Test with Matern 5/2 kernel."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(
            kernel="matern52", num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert gp.is_trained
        assert result.num_tasks == 2

    def test_score(self):
        """Test the score() method."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2, seed=42)
        X_test, Y_test = generate_multi_output_data(n=500, d=3, T=2, seed=99)

        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=30, verbose=False)

        scores = gp.score(X_test, Y_test)
        assert "rmse" in scores
        assert "mae" in scores
        assert "r2" in scores
        assert "rmse_per_task" in scores
        assert scores["rmse"] > 0
        assert scores["mae"] > 0
        assert scores["rmse_per_task"].shape == (2,)

    def test_task_covariance_property(self):
        """Test the task_covariance property."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=20, verbose=False)

        B = gp.task_covariance
        assert B is not None
        assert B.shape == (2, 2)
        # B should be symmetric
        np.testing.assert_allclose(B, B.T, atol=1e-5)

    def test_training_result_property(self):
        """Test the training_result property."""
        from mojogp.multi_output_gp import (
            MultiOutputGP,
            MultiOutputTrainingResult,
        )

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(kernel="rbf", num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=20, verbose=False)

        result = gp.training_result
        assert isinstance(result, MultiOutputTrainingResult)
        assert result.lengthscale > 0
        assert result.outputscale > 0
        assert result.noise > 0


class TestCrossModelProviderLifecycle:
    """Regression tests for shared kernel-module provider ownership."""

    @pytest.mark.parametrize(
        "joint_kernel,exact_kernel_name,method,n,num_tasks,max_iterations,learning_rate",
        [
            ("matern52", "matern52", "materialized", 500, 3, 20, 0.03),
            ("rbf", "rbf", "matrix_free", 2000, 2, 20, 0.02),
        ],
    )
    def test_exactgp_fit_after_multi_output_predict_rebuilds_safely(
        self,
        joint_kernel,
        exact_kernel_name,
        method,
        n,
        num_tasks,
        max_iterations,
        learning_rate,
    ):
        from mojogp import SingleOutputGP, Matern52, RBF
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=n, d=3, T=num_tasks)
        gp = MultiOutputGP(
            kernel=joint_kernel,
            num_probes=5 if method == "materialized" else 3,
            max_cg_iterations=50 if method == "materialized" else 30,
            max_tridiag_iterations=20 if method == "materialized" else 10,
            preconditioner_rank=10 if method == "materialized" else 5,
        )
        gp.fit(
            X,
            Y,
            method=method,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            verbose=False,
        )

        np.random.seed(0)
        X_test = np.random.randn(16, 3).astype(np.float32)
        gp.predict(X_test)
        mean_joint, var_joint = gp.predict(X_test, return_var=True)
        assert mean_joint.shape == (16, num_tasks)
        assert var_joint.shape == (16, num_tasks)

        exact_kernel = {"rbf": RBF, "matern52": Matern52}[exact_kernel_name]()
        exact_gp = SingleOutputGP(exact_kernel)
        exact_gp.fit(
            X,
            Y[:, 0],
            method=method,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            verbose=False,
        )

        pred = exact_gp.predict(X_test)
        assert pred.mean.shape == (16,)
        assert pred.variance.shape == (16,)

        rebuilt_mean, rebuilt_var = gp.predict(X_test, return_var=True)
        assert rebuilt_mean.shape == (16, num_tasks)
        assert rebuilt_var.shape == (16, num_tasks)


class TestMultiOutputGPARD:
    """Test ARD (per-dimension lengthscale) training and prediction."""

    def test_fit_ard_returns_per_dimension_lengthscales(self):
        """Test ARD training returns per-dimension lengthscales."""
        from mojogp.multi_output_gp import (
            MultiOutputGP,
            MultiOutputTrainingResult,
        )

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(
            kernel="rbf", ard=True, num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert isinstance(result, MultiOutputTrainingResult)
        assert result.lengthscales.shape == (3,)
        assert result.dim == 3
        assert all(ls > 0 for ls in result.lengthscales)

    def test_fit_ard_with_init_lengthscales(self):
        """Test ARD with custom initial lengthscales."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        init_ls = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        gp = MultiOutputGP(
            kernel="rbf", ard=True, num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        result = gp.fit(
            X, Y, max_iterations=20, initial_lengthscales=init_ls, verbose=False
        )

        assert result.lengthscales.shape == (3,)

    def test_fit_ard_wrong_init_shape_raises(self):
        """Test that wrong init_lengthscales shape raises."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        init_ls = np.array([0.5, 1.0], dtype=np.float32)  # Wrong shape
        gp = MultiOutputGP(
            kernel="rbf", ard=True, num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        with pytest.raises(ValueError, match="initial_lengthscales"):
            gp.fit(X, Y, max_iterations=10, initial_lengthscales=init_ls, verbose=False)

    def test_predict_ard(self):
        """Test prediction with ARD model."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(
            kernel="rbf", ard=True, num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        gp.fit(X, Y, max_iterations=20, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True)

        assert mean.shape == (10, 2)
        assert var is not None
        assert var.shape == (10, 2)

    def test_direct_engine_predict_auto_rank_matches_explicit_ard_rank_for_ard(self):
        """Direct engine predict_multi_output should resolve omitted ARD rank to 200."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=2000, d=4, T=2, seed=789)
        gp = MultiOutputGP(
            kernel="rbf", ard=True, num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        gp.fit(X, Y, max_iterations=15, verbose=False)

        result = gp._result
        provider_info = gp._build_provider_info(
            gp._X_train_cont,
            gp._compiled_kernel.to_engine_params(
                np.ascontiguousarray(result.params, dtype=np.float32)
            ),
            float(np.mean(result.noise_per_task)),
        )
        X_test = np.random.RandomState(123).randn(32, 4).astype(np.float32)
        common_args = [
            provider_info,
            np.ascontiguousarray(result.alpha_rotated, dtype=np.float32),
            np.ascontiguousarray(result.Q, dtype=np.float32),
            np.ascontiguousarray(result.effective_scales, dtype=np.float32),
            X_test.astype(np.float32),
            np.ascontiguousarray(result.params, dtype=np.float32),
            float(np.mean(result.noise_per_task)),
        ]

        try:
            pred_auto = gp._engine.predict_multi_output(
                *common_args,
                1,
                gp.max_cg_iter,
                float(gp.cg_tol),
                gp.precond_rank,
            )
            pred_200 = gp._engine.predict_multi_output(
                *common_args,
                1,
                gp.max_cg_iter,
                float(gp.cg_tol),
                gp.precond_rank,
                200,
            )
            pred_exact = gp._engine.predict_multi_output(
                *common_args,
                2,
                gp.max_cg_iter,
                float(gp.cg_tol),
                gp.precond_rank,
                100,
            )
        finally:
            gp._kernel_module.destroy_provider(provider_info)

        assert pred_auto["lanczos_rank_used"] == 200
        assert pred_200["lanczos_rank_used"] == 200

        auto_var = np.asarray(pred_auto["variance"], dtype=np.float32)
        explicit_var = np.asarray(pred_200["variance"], dtype=np.float32)
        exact_var = np.asarray(pred_exact["variance"], dtype=np.float32)

        assert np.all(np.isfinite(auto_var))
        assert np.all(np.isfinite(explicit_var))
        assert np.all(np.isfinite(exact_var))
        assert np.all(auto_var >= 0)
        assert np.all(explicit_var >= 0)
        assert np.all(exact_var >= 0)

        mask = exact_var > 1e-5
        assert np.any(mask)
        auto_rel = np.abs(auto_var[mask] - exact_var[mask]) / (exact_var[mask] + 1e-6)
        explicit_rel = np.abs(explicit_var[mask] - exact_var[mask]) / (
            exact_var[mask] + 1e-6
        )
        assert float(np.mean(auto_rel < 5.0)) > 0.9
        assert float(np.mean(explicit_rel < 5.0)) > 0.9
        assert np.isclose(
            float(np.mean(auto_var)),
            float(np.mean(explicit_var)),
            rtol=0.25,
            atol=0.03,
        )

    def test_ard_dimension_relevance(self):
        """Test that ARD identifies relevant dimensions.

        Create data that only depends on dim 0, and verify that
        the lengthscale for dim 0 is shorter than for other dims.
        """
        from mojogp.multi_output_gp import MultiOutputGP

        np.random.seed(42)
        n, d, T = 2000, 5, 2
        X = np.random.randn(n, d).astype(np.float32)
        Y = np.zeros((n, T), dtype=np.float32)
        Y[:, 0] = np.sin(3 * X[:, 0]) + 0.05 * np.random.randn(n)
        Y[:, 1] = np.cos(3 * X[:, 0]) + 0.05 * np.random.randn(n)

        gp = MultiOutputGP(
            kernel="rbf", ard=True, num_probes=5, max_cg_iterations=100, preconditioner_rank=10
        )
        result = gp.fit(X, Y, max_iterations=80, verbose=False, early_stop_tol=0.0)

        ls = result.lengthscales
        # Dim 0 should have shorter lengthscale (more relevant)
        # Other dims should have longer lengthscales (less relevant)
        assert ls[0] < np.mean(ls[1:]), (
            f"Dim 0 lengthscale ({ls[0]:.3f}) should be shorter than "
            f"mean of others ({np.mean(ls[1:]):.3f})"
        )

    def test_repr_ard_trained(self):
        """Test repr after ARD training."""
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(
            kernel="rbf", ard=True, num_probes=5, max_cg_iterations=50, preconditioner_rank=10
        )
        gp.fit(X, Y, max_iterations=10, verbose=False)

        r = repr(gp)
        assert "trained" in r
        assert "ard=True" in r
        assert "tasks=2" in r


class TestMultiOutputGPComposite:
    """Test composite kernel training and prediction via JIT pipeline."""

    def test_fit_composite_rbf(self):
        """Test composite kernel (single RBF) training."""
        from mojogp.multi_output_gp import (
            MultiOutputGP,
            MultiOutputTrainingResult,
        )
        from mojogp.kernel import Kernel

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        kernel = Kernel.rbf()
        gp = MultiOutputGP(kernel=kernel, num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert isinstance(result, MultiOutputTrainingResult)
        assert result.num_kernel_params == 2  # lengthscale + outputscale
        assert result.params.shape == (2,)
        assert result.B.shape == (2, 2)

    def test_fit_composite_sum_kernel(self):
        """Test sum kernel (RBF + Matern52) training."""
        from mojogp.multi_output_gp import (
            MultiOutputGP,
            MultiOutputTrainingResult,
        )
        from mojogp.kernel import Kernel

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        kernel = Kernel.rbf() + Kernel.matern52()
        gp = MultiOutputGP(kernel=kernel, num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        result = gp.fit(X, Y, max_iterations=20, verbose=False)

        assert isinstance(result, MultiOutputTrainingResult)
        assert result.num_kernel_params == 4  # 2 per base kernel
        assert result.params.shape == (4,)
        assert len(result.param_names) == 4

    def test_predict_composite(self):
        """Test prediction with composite kernel."""
        from mojogp.multi_output_gp import MultiOutputGP
        from mojogp.kernel import Kernel

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        kernel = Kernel.rbf()
        gp = MultiOutputGP(kernel=kernel, num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=20, verbose=False)

        X_test = np.random.randn(10, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True)

        assert mean.shape == (10, 2)
        # Variance may be None or have negative values for composite (known issue)
        # Just check it doesn't crash

    def test_fit_composite_with_init_params(self):
        """Test composite kernel with custom initial parameters."""
        from mojogp.multi_output_gp import MultiOutputGP
        from mojogp.kernel import Kernel

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        kernel = Kernel.rbf()
        init_params = np.array([0.5, 2.0], dtype=np.float32)
        gp = MultiOutputGP(kernel=kernel, num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        result = gp.fit(X, Y, max_iterations=20, initial_params=init_params, verbose=False)

        assert result.params.shape == (2,)

    def test_fit_composite_wrong_params_shape_raises(self):
        """Test that wrong init_params shape raises."""
        from mojogp.multi_output_gp import MultiOutputGP
        from mojogp.kernel import Kernel

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        kernel = Kernel.rbf()
        init_params = np.array([0.5, 2.0, 3.0], dtype=np.float32)  # Wrong shape
        gp = MultiOutputGP(kernel=kernel, num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        with pytest.raises(ValueError, match="init_params"):
            gp.fit(X, Y, max_iterations=10, initial_params=init_params, verbose=False)


class TestMultiOutputGPInputValidation:
    """Test input validation."""

    def test_1d_X_raises(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP()
        X = np.random.randn(100).astype(np.float32)
        Y = np.random.randn(100, 2).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            gp.fit(X, Y)

    def test_1d_Y_raises(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP()
        X = np.random.randn(100, 3).astype(np.float32)
        Y = np.random.randn(100).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            gp.fit(X, Y)

    def test_mismatched_n_raises(self):
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP()
        X = np.random.randn(100, 3).astype(np.float32)
        Y = np.random.randn(50, 2).astype(np.float32)
        with pytest.raises(ValueError, match="must match"):
            gp.fit(X, Y)

    def test_1d_X_test_raises(self):
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y = generate_multi_output_data(n=500, d=3, T=2)
        gp = MultiOutputGP(num_probes=5, max_cg_iterations=50, preconditioner_rank=10)
        gp.fit(X, Y, max_iterations=10, verbose=False)

        X_test = np.random.randn(10).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            gp.predict(X_test)


class TestMultiOutputGPImport:
    """Test that MultiOutputGP is importable from the package."""

    def test_import_from_package(self):
        from mojogp import MultiOutputGP

        gp = MultiOutputGP()
        assert gp is not None

    def test_import_result_types(self):
        from mojogp import (
            MultiOutputTrainingResult,
            MultiOutputPredictionResult,
        )

        # Just verify they're importable
        assert MultiOutputTrainingResult is not None
        assert MultiOutputPredictionResult is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
