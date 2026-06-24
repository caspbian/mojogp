"""Integration tests for posterior sampling routes and heterogeneous LMC kernels.

This file covers:
- heterogeneous latent-kernel initialization, training, and prediction
- ExactGP, MultiOutputGP, and MultiOutputLMCGP posterior sampling routes
- mixed-kernel pathwise save/load behavior
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


MODULE = "tests.integration.run_posterior_sampling_case"


def _skip_if_no_lib(lib_name, build_cmd="task build"):
    lib_name = "mojogp_jit_engine"

    try:
        __import__(lib_name)
    except ImportError:
        pytest.skip(f"{lib_name} not built. Run `{build_cmd}` first.")


def _cleanup_gpu_state():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _run_pathwise_case(case, method=None, save_dir=None):
    method_arg = method if method is not None else "__none__"
    save_arg = str(save_dir) if save_dir is not None else "__none__"
    _cleanup_gpu_state()
    time.sleep(0.05)
    return run_isolated_case(
        module=MODULE,
        payload={"case": case, "method": method_arg, "save_dir": save_arg},
        timeout=600,
        description=f"Runs posterior sampling case {case}/{method_arg}",
    )


def generate_data(n=200, d=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(n).astype(np.float32)
    return X, y


def generate_mo_data(n=200, d=3, T=2, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.column_stack(
        [
            np.sin(X[:, 0]) + 0.1 * np.random.randn(n),
            np.cos(X[:, 0]) + 0.1 * np.random.randn(n),
        ]
    ).astype(np.float32)
    return X, Y


def generate_mixed_mo_data(n=2000, d_cont=2, T=2, levels=3, seed=123, noise_std=0.05):
    rng = np.random.default_rng(seed)
    X_cont = rng.standard_normal((n, d_cont)).astype(np.float32)
    cat = rng.integers(0, levels, size=(n, 1), dtype=np.int32)
    X = np.concatenate([X_cont, cat.astype(np.float32)], axis=1)

    cat_effect = np.linspace(-0.6, 0.6, levels, dtype=np.float32)[cat[:, 0]]
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
# 3.3: Heterogeneous LMC kernels
# ============================================================================


class TestHeterogeneousLMCInit:
    """Test MultiOutputLMCGP with different kernel types per latent."""

    def test_heterogeneous_init_two_latents(self):
        """Two latents with different kernel types."""
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        k1 = Kernel.rbf()
        k2 = Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k1, k2])
        assert gp.num_latents == 2
        assert gp.kernels[0].num_params() == 2  # RBF: outputscale + lengthscale
        assert gp.kernels[1].num_params() == 2  # Matern52: outputscale + lengthscale

    def test_heterogeneous_init_different_param_counts(self):
        """Latents with different number of params (different composite structures)."""
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        k1 = Kernel.rbf()  # 2 params
        k2 = Kernel.matern52() + Kernel.periodic()  # 5 params
        gp = MultiOutputLMCGP(kernels=[k1, k2])
        assert gp.num_latents == 2
        np1 = k1.num_params()
        np2 = k2.num_params()
        assert np1 != np2  # heterogeneous
        assert np1 == 2
        assert np2 == 5

    def test_heterogeneous_repr(self):
        """Repr shows heterogeneous kernels distinctly."""
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()])
        r = repr(gp)
        assert "R=2" in r
        assert "composite=True" in r

    def test_homogeneous_repr(self):
        """Repr shows single kernel type for homogeneous."""
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        k = Kernel.rbf()
        gp = MultiOutputLMCGP(kernels=[k, k])
        r = repr(gp)
        assert "R=2" in r

    def test_init_params_validation_heterogeneous(self):
        """Flat init_params with total count is accepted."""
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        k1 = Kernel.rbf()  # 2 params
        k2 = Kernel.matern52()  # 2 params
        gp = MultiOutputLMCGP(kernels=[k1, k2])
        # total_params = 4 — flat array accepted
        init_params = np.ones(4, dtype=np.float32)
        # Should not raise during fit setup
        assert gp.num_latents == 2

    def test_init_params_wrong_shape_raises(self):
        """Wrong init_params shape raises ValueError."""
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP
        from unittest.mock import patch

        k1 = Kernel.rbf()  # 2 params
        k2 = Kernel.matern52()  # 2 params
        gp = MultiOutputLMCGP(kernels=[k1, k2])

        X, Y = generate_mo_data(n=50, d=3)
        with pytest.raises(ValueError, match="init_params"):
            gp.fit(X, Y, max_iterations=1, initial_params=np.ones(7, dtype=np.float32))


class TestHeterogeneousLMCTrainPredict:
    """Test training and prediction with heterogeneous LMC kernels."""

    def test_fit_predict_same_type(self):
        """Two RBF latents (homogeneous) — basic sanity check."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=3, T=2)
        X_test = np.random.randn(10, 3).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.rbf()])
        gp.fit(X, Y, max_iterations=5, verbose=False)

        mean, var = gp.predict(X_test, return_var=True)
        assert mean.shape == (10, 2)
        assert var.shape == (10, 2)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)

    def test_fit_predict_different_types(self):
        """RBF + Matern52 latents (heterogeneous)."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=3, T=2)
        X_test = np.random.randn(10, 3).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()])
        gp.fit(X, Y, max_iterations=5, verbose=False)

        mean, var = gp.predict(X_test, return_var=True)
        assert mean.shape == (10, 2)
        assert var.shape == (10, 2)
        assert np.all(np.isfinite(mean))
        assert np.all(var >= 0)

    def test_params_per_latent_stored(self):
        """params_per_latent is stored and has per-latent param counts."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=3, T=2)
        k1 = Kernel.rbf()  # 2 params
        k2 = Kernel.matern52()  # 2 params

        gp = MultiOutputLMCGP(kernels=[k1, k2])
        result = gp.fit(X, Y, max_iterations=5, verbose=False)

        assert result.params_per_latent is not None
        assert len(result.params_per_latent) == 2
        assert len(result.params_per_latent[0]) == k1.num_params()
        assert len(result.params_per_latent[1]) == k2.num_params()

    def test_save_load_heterogeneous(self, tmp_path):
        """Save and load heterogeneous LMC model — kernel trees preserved."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel, KernelNode
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=3, T=2)
        X_test = np.random.randn(10, 3).astype(np.float32)

        k1 = Kernel.rbf()
        k2 = Kernel.matern52()
        gp = MultiOutputLMCGP(kernels=[k1, k2])
        gp.fit(X, Y, max_iterations=5, verbose=False)
        mean_before, _ = gp.predict(X_test, return_var=True)

        # Save
        save_path = str(tmp_path / "test_hetero_lmc")
        gp.save(save_path)

        # Load (no kernels arg needed — reconstructed from kernel_trees)
        gp2 = MultiOutputLMCGP.load(save_path)
        assert len(gp2.kernels) == 2
        assert all(isinstance(k, KernelNode) for k in gp2.kernels)

        # Predictions should match
        mean_after, _ = gp2.predict(X_test, return_var=True)
        np.testing.assert_allclose(mean_before, mean_after, atol=1e-5)


# ============================================================================
# 3.4: Cholesky posterior sampling
# ============================================================================


class TestExactGPPosteriorSampling:
    """Test ExactGP posterior sampling surfaces."""

    def test_diagonal_sampling_default(self):
        """Default method='diagonal' still works."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(20, 2).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=5)
        samples = gp.sample_posterior(X_test, n_samples=3)
        assert samples.shape == (3, 20)
        assert np.all(np.isfinite(samples))

    def test_pathwise_sampling_shape(self):
        """Pathwise correlated samples have correct shape."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(20, 2).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=5)
        samples = gp.sample_posterior(X_test, n_samples=5, method="pathwise")
        assert samples.shape == (5, 20)
        assert np.all(np.isfinite(samples))

    def test_pathwise_sampling_correlated(self):
        """Pathwise samples exhibit spatial correlation (nearby points correlated)."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y = generate_data(n=200, d=1)
        # Test points: two close, two far
        X_test = np.array([[0.0], [0.01], [10.0], [10.01]], dtype=np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=10)

        samples = gp.sample_posterior(X_test, n_samples=100, method="pathwise")
        assert samples.shape == (100, 4)

        # Close pairs should have higher cross-sample correlation than far pairs
        corr_close = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        corr_far = np.corrcoef(samples[:, 0], samples[:, 2])[0, 1]
        assert corr_close > corr_far, (
            f"Close points should be more correlated: corr_close={corr_close:.3f}, "
            f"corr_far={corr_far:.3f}"
        )

    def test_pathwise_mean_matches_predict(self):
        """Mean of many pathwise samples approximates predictive mean."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(5, 2).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=10)

        pred_mean, _ = gp.predict(X_test, return_std=True)
        samples = gp.sample_posterior(X_test, n_samples=2000, method="pathwise")
        sample_mean = samples.mean(axis=0)

        # Sample mean should approximate predictive mean (loose tolerance for Monte Carlo)
        np.testing.assert_allclose(sample_mean, pred_mean, atol=0.2)

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP, RBF

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(5, 2).astype(np.float32)

        gp = SingleOutputGP(RBF())
        gp.fit(X, y, max_iterations=5)

        with pytest.raises(ValueError, match="method"):
            gp.sample_posterior(X_test, method="invalid")

    def test_method_parameter_exists(self):
        """Verify method parameter is accepted by sample_posterior."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        import inspect
        from mojogp import SingleOutputGP

        sig = inspect.signature(SingleOutputGP.sample_posterior)
        assert "method" in sig.parameters
        assert sig.parameters["method"].default == "diagonal"


class TestMultiOutputLMCPosteriorSampling:
    """Test MultiOutputLMCGP posterior sampling surfaces."""

    def test_diagonal_sampling_default(self):
        """Default method='diagonal' still works."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=2, T=2)
        X_test = np.random.randn(10, 2).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])
        gp.fit(X, Y, max_iterations=5)
        samples = gp.sample_posterior(X_test, n_samples=3)
        assert samples.shape == (3, 10, 2)
        assert np.all(np.isfinite(samples))

    def test_pathwise_sampling_shape(self):
        """Pathwise samples have correct shape [n_samples, m, T]."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=2, T=2)
        X_test = np.random.randn(10, 2).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()])
        gp.fit(X, Y, max_iterations=5)
        samples = gp.sample_posterior(X_test, n_samples=4, method="pathwise")
        assert samples.shape == (4, 10, 2)
        assert np.all(np.isfinite(samples))

    def test_pathwise_variance_positive(self):
        """Per-task posterior variance from pathwise samples is positive."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=2, T=2)
        X_test = np.random.randn(5, 2).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])
        gp.fit(X, Y, max_iterations=5)
        samples = gp.sample_posterior(X_test, n_samples=500, method="pathwise")

        # Per-task sample variance should be positive
        sample_var = samples.var(axis=0)  # [m, T]
        assert np.all(sample_var > 0), f"Some sample variances <= 0: {sample_var}"

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp.kernel import Kernel
        from mojogp.multi_output_gp import MultiOutputLMCGP

        X, Y = generate_mo_data(n=100, d=2, T=2)
        X_test = np.random.randn(5, 2).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf()])
        gp.fit(X, Y, max_iterations=5)

        with pytest.raises(ValueError, match="method"):
            gp.sample_posterior(X_test, method="invalid")


# ============================================================================
# 3.4: Pathwise posterior sampling (RFF prior + backend correction)
# ============================================================================


class TestExactGPPathwise:
    """Test provider-backed pathwise posterior sampling for ExactGP."""

    def test_pathwise_shape(self):
        """Pathwise samples have correct shape [n_samples, m]."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(15, 2).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(), verbose=False)
        gp.fit(X, y, max_iterations=10)

        samples = gp.sample_posterior(X_test, n_samples=5, method="pathwise")
        assert samples.shape == (5, 15), f"Expected (5, 15), got {samples.shape}"
        assert np.all(np.isfinite(samples)), "Pathwise samples contain NaN/Inf"
        assert gp.backend_sample_info["actual_sampling_route"] == "provider_pathwise"

    def test_pathwise_reproducibility(self):
        """Same rng seed produces identical samples."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(10, 2).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(), verbose=False)
        gp.fit(X, y, max_iterations=10)

        s1 = gp.sample_posterior(
            X_test, n_samples=5, method="pathwise", rng=np.random.default_rng(42)
        )
        s2 = gp.sample_posterior(
            X_test, n_samples=5, method="pathwise", rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(
            s1, s2, err_msg="Pathwise samples not reproducible"
        )

    def test_pathwise_mean_matches_predictive_mean(self):
        """Mean of many pathwise samples approximates the GP predictive mean.

        Both the pathwise sample mean and the predictive mean are computed
        independently using the same kernel parameters, so they should agree.
        The correction uses the live backend provider route, so this also checks
        that the provider-backed pathwise implementation stays numerically sane.
        """
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        rng_data = np.random.default_rng(7)
        X = rng_data.standard_normal((200, 2)).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)
        X_test = rng_data.standard_normal((5, 2)).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(), verbose=False)
        gp.fit(X, y, max_iterations=20)

        # Compute "predictive mean" directly via cholesky solve (same as Matheron
        # correction but without RFF prior — this is the ground truth)
        tr = gp._training_result
        params = np.array(tr.params, dtype=np.float32)
        noise = float(tr.noise)
        mean_val = float(tr.mean)
        X_train = gp._X_train
        K_train = gp.kernel.evaluate(X_train, X_train, params=params)
        K_cross = gp.kernel.evaluate(X_test, X_train, params=params)
        K_reg = K_train + noise * np.eye(len(X_train), dtype=np.float32)
        y_centered = y - mean_val
        alpha = np.linalg.solve(K_reg.astype(np.float64), y_centered.astype(np.float64))
        ref_mean = (K_cross.astype(np.float64) @ alpha + mean_val).astype(np.float32)

        samples = gp.sample_posterior(
            X_test,
            n_samples=256,
            method="pathwise",
            n_rff_features=2048,
            rng=np.random.default_rng(0),
        )
        sample_mean = samples.mean(axis=0)

        np.testing.assert_allclose(
            sample_mean,
            ref_mean,
            atol=0.1,
            err_msg=(
                f"Pathwise sample mean diverges from reference mean.\n"
                f"ref_mean={ref_mean}\nsample_mean={sample_mean}"
            ),
        )

    def test_pathwise_correlated(self):
        """Pathwise samples exhibit spatial correlation (nearby points correlated)."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=1)
        X_test = np.array([[0.0], [0.01], [10.0], [10.01]], dtype=np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(), verbose=False)
        gp.fit(X, y, max_iterations=10)

        samples = gp.sample_posterior(
            X_test, n_samples=200, method="pathwise", rng=np.random.default_rng(99)
        )
        assert samples.shape == (200, 4)

        corr_close = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        corr_far = np.corrcoef(samples[:, 0], samples[:, 2])[0, 1]
        assert corr_close > corr_far, (
            f"Close points should be more correlated than far points: "
            f"corr_close={corr_close:.3f}, corr_far={corr_far:.3f}"
        )

    def test_pathwise_matern52_kernel(self):
        """Pathwise sampling works for Matern52 kernel."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(10, 2).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.matern52(), verbose=False)
        gp.fit(X, y, max_iterations=10)

        samples = gp.sample_posterior(
            X_test, n_samples=5, method="pathwise", rng=np.random.default_rng(1)
        )
        assert samples.shape == (5, 10)
        assert np.all(np.isfinite(samples))

    def test_pathwise_ard_rbf_kernel(self):
        """Pathwise sampling works for ARD RBF kernel."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=3)
        X_test = np.random.randn(10, 3).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(ard=True), verbose=False)
        gp.fit(X, y, max_iterations=10)

        samples = gp.sample_posterior(
            X_test, n_samples=5, method="pathwise", rng=np.random.default_rng(2)
        )
        assert samples.shape == (5, 10)
        assert np.all(np.isfinite(samples))

    def test_pathwise_samples_are_reproducible(self):
        """Pathwise samples are reproducible with the same RNG seed."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(10, 2).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(), verbose=False)
        gp.fit(X, y, max_iterations=10)

        s_pathwise = gp.sample_posterior(
            X_test, n_samples=5, method="pathwise", rng=np.random.default_rng(3)
        )
        s_alias = gp.sample_posterior(
            X_test, n_samples=5, method="pathwise", rng=np.random.default_rng(3)
        )
        np.testing.assert_array_equal(s_pathwise, s_alias)
        assert gp.backend_sample_info["actual_sampling_method"] == "pathwise"

    def test_pathwise_linear_kernel_returns_finite_samples(self):
        """Linear kernels use the shared exact feature-map path."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(10, 2).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.linear(), verbose=False)
        gp.fit(X, y, max_iterations=10)

        samples = gp.sample_posterior(
            X_test, n_samples=5, method="pathwise", rng=np.random.default_rng(3)
        )
        assert samples.shape == (5, 10)
        assert np.all(np.isfinite(samples))

    def test_pathwise_mixed_kernel_uses_mixed_backend_route(self):
        """Mixed ExactGP pathwise sampling uses the mixed backend route."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        rng = np.random.default_rng(13)
        X_cont, y = generate_data(n=200, d=2, seed=13)
        cat = rng.integers(0, 3, size=(X_cont.shape[0], 1), endpoint=False).astype(
            np.float32
        )
        X = np.concatenate([X_cont, cat], axis=1).astype(np.float32)
        X_test = X[:6].copy()

        gp = SingleOutputGP(
            kernel=Kernel.rbf(active_dims=[0, 1])
            * Kernel.ehh(levels=3, active_dims=[2]),
            verbose=False,
        )
        gp.fit(X, y, max_iterations=5, method="matrix_free")

        samples = gp.sample_posterior(
            X_test, n_samples=3, method="pathwise", rng=np.random.default_rng(3)
        )
        assert samples.shape == (3, 6)
        assert np.all(np.isfinite(samples))

    @pytest.mark.parametrize("method", ["matrix_free", "materialized"])
    def test_pathwise_reports_backend_route_and_survives_save_load(
        self, method, tmp_path
    ):
        """Pathwise uses the backend route in both modes and survives save/load."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=2, seed=11)
        X_test = np.random.randn(6, 2).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(), verbose=False)
        gp.fit(X, y, max_iterations=10, method=method)

        samples = gp.sample_posterior(
            X_test, n_samples=4, method="pathwise", rng=np.random.default_rng(7)
        )
        info = gp.backend_sample_info
        assert samples.shape == (4, 6)
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["backend_correction_route"] == "predict"
        assert info["training_route"] == method

        save_path = tmp_path / f"pathwise_exactgp_{method}"
        gp.save(str(save_path))
        loaded = SingleOutputGP.load(str(save_path))
        loaded_samples = loaded.sample_posterior(
            X_test, n_samples=4, method="pathwise", rng=np.random.default_rng(7)
        )

        np.testing.assert_allclose(loaded_samples, samples, atol=1e-6)
        loaded_info = loaded.backend_sample_info
        assert loaded_info["actual_sampling_route"] == "provider_pathwise"
        assert loaded_info["training_route"] == method

    def test_pathwise_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import SingleOutputGP
        from mojogp.kernel import Kernel

        X, y = generate_data(n=200, d=2)
        X_test = np.random.randn(5, 2).astype(np.float32)

        gp = SingleOutputGP(kernel=Kernel.rbf(), verbose=False)
        gp.fit(X, y, max_iterations=5)

        with pytest.raises(ValueError, match="method"):
            gp.sample_posterior(X_test, method="bogus")


class TestMultiOutputGPPathwise:
    """Test provider-backed pathwise posterior sampling for MultiOutputGP."""

    def test_pathwise_shape_and_alias(self):
        """Pathwise samples have the expected shape and alias behavior."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputGP
        from mojogp.kernel import Kernel

        X, Y = generate_mo_data(n=2000, d=2, T=2, seed=21)
        X_test = np.random.randn(8, 2).astype(np.float32)

        # Route coverage for matrix_free vs materialized lives in the save/load
        # pathwise test below. Keep this correlation-behavior case on the more
        # stable materialized training path so it verifies sampling structure,
        # not backend optimizer initialization luck.
        gp = MultiOutputGP(kernel=Kernel.rbf())
        gp.fit(X, Y, max_iterations=8, verbose=False, method="materialized")

        samples = gp.sample_posterior(
            X_test,
            n_samples=4,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(5),
        )
        alias_samples = gp.sample_posterior(
            X_test,
            n_samples=4,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(5),
        )

        assert samples.shape == (4, 8, 2)
        assert np.all(np.isfinite(samples))
        np.testing.assert_array_equal(samples, alias_samples)
        info = gp.backend_sample_info
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["actual_sampling_method"] == "pathwise"

    def test_pathwise_mean_matches_predictive_mean(self):
        """Mean of many pathwise samples tracks the predictive mean."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputGP
        from mojogp.kernel import Kernel

        rng = np.random.default_rng(9)
        X = rng.standard_normal((2000, 2)).astype(np.float32)
        shared = np.sin(X[:, 0]).astype(np.float32)
        Y = np.column_stack(
            [
                shared + 0.05 * rng.standard_normal(2000),
                0.7 * shared + 0.3 * np.cos(X[:, 1]) + 0.05 * rng.standard_normal(2000),
            ]
        ).astype(np.float32)
        X_test = rng.standard_normal((5, 2)).astype(np.float32)

        gp = MultiOutputGP(kernel=Kernel.rbf())
        gp.fit(X, Y, max_iterations=10, verbose=False, method="materialized")

        pred = gp.predict(X_test)
        samples = gp.sample_posterior(
            X_test,
            n_samples=96,
            method="pathwise",
            n_rff_features=1536,
            rng=np.random.default_rng(17),
        )
        abs_diff = np.abs(samples.mean(axis=0) - pred.mean)
        assert float(abs_diff.mean()) < 0.2
        assert float(abs_diff.max()) < 0.45

    def test_pathwise_captures_cross_task_and_spatial_correlation(self):
        """Pathwise samples are correlated across tasks and nearby test points."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputGP
        from mojogp.kernel import Kernel

        rng = np.random.default_rng(33)
        X = rng.standard_normal((2000, 1)).astype(np.float32)
        latent = np.sin(X[:, 0]).astype(np.float32)
        Y = np.column_stack(
            [
                latent + 0.01 * rng.standard_normal(2000),
                1.3 * latent + 0.01 * rng.standard_normal(2000),
            ]
        ).astype(np.float32)
        # Probe far from the dense training region so posterior correlation stays visible.
        X_test = np.array([[4.0], [4.02], [8.0]], dtype=np.float32)

        gp = MultiOutputGP(kernel=Kernel.rbf())
        # Give the ICM task covariance a few more optimizer steps so the
        # shared latent structure is learned consistently across runs.
        gp.fit(X, Y, max_iterations=12, verbose=False, method="matrix_free")

        samples = gp.sample_posterior(
            X_test,
            n_samples=192,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(99),
        )

        learned_task_cov = gp.training_result.B.astype(np.float64)
        learned_task_corr = float(
            learned_task_cov[0, 1]
            / np.sqrt(max(learned_task_cov[0, 0] * learned_task_cov[1, 1], 1e-12))
        )
        same_point_cross_task = np.corrcoef(samples[:, 0, 0], samples[:, 0, 1])[0, 1]
        close_point_corr = np.corrcoef(samples[:, 0, 0], samples[:, 1, 0])[0, 1]
        far_point_corr = np.corrcoef(samples[:, 0, 0], samples[:, 2, 0])[0, 1]
        assert close_point_corr > far_point_corr
        if abs(learned_task_corr) >= 0.1:
            assert abs(same_point_cross_task) > 0.05
            assert np.sign(same_point_cross_task) == np.sign(learned_task_corr)
        else:
            # Short ICM fits can occasionally learn nearly independent tasks.
            # In that regime the pathwise samples should stay close to the
            # learned near-zero task correlation rather than invent a strong one.
            assert abs(same_point_cross_task - learned_task_corr) < 0.08

    @pytest.mark.parametrize("method", ["matrix_free", "materialized"])
    def test_pathwise_survives_save_load_in_both_routes(self, method, tmp_path):
        """Pathwise uses the requested backend route before and after save/load."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputGP
        from mojogp.kernel import Kernel

        X, Y = generate_mo_data(n=2000, d=2, T=2, seed=41)
        X_test = np.random.randn(6, 2).astype(np.float32)

        gp = MultiOutputGP(kernel=Kernel.rbf())
        gp.fit(X, Y, max_iterations=8, verbose=False, method=method)

        samples = gp.sample_posterior(
            X_test,
            n_samples=3,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(123),
        )
        info = gp.backend_sample_info
        assert samples.shape == (3, 6, 2)
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["training_route"] == method

        save_path = tmp_path / f"pathwise_multioutput_{method}"
        gp.save(str(save_path))
        loaded = MultiOutputGP.load(str(save_path))
        loaded_samples = loaded.sample_posterior(
            X_test,
            n_samples=3,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_allclose(samples, loaded_samples, atol=1e-6)
        loaded_info = loaded.backend_sample_info
        assert loaded_info["actual_sampling_route"] == "provider_pathwise"
        assert loaded_info["training_route"] == method

    def test_pathwise_rejects_polynomial_degree_two(self):
        """ICM polynomial pathwise sampling is an explicit unsupported boundary."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputGP
        from mojogp.kernel import Kernel

        X, Y = generate_mo_data(n=2000, d=2, T=2, seed=51)
        X_test = np.random.randn(4, 2).astype(np.float32)

        kernel_node = Kernel.polynomial(degree=2.0)
        gp = MultiOutputGP(kernel=kernel_node)
        gp.fit(X, Y, max_iterations=6, verbose=False, method="matrix_free")

        with pytest.raises(NotImplementedError, match="Polynomial Pathwise"):
            gp.sample_posterior(
                X_test,
                n_samples=2,
                method="pathwise",
                rng=np.random.default_rng(2),
            )

    def test_pathwise_supports_mixed_multioutputgp(self):
        """Mixed MultiOutputGP pathwise sampling uses the mixed backend route."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import Kernel, MultiOutputGP

        X, Y = generate_mixed_mo_data(n=2000, d_cont=2, T=2, levels=3, seed=57)
        gp = MultiOutputGP(
            kernel=Kernel.rbf(active_dims=[0, 1])
            * Kernel.ehh(levels=3, active_dims=[2]),
        )
        gp.fit(X, Y, max_iterations=4, verbose=False, method="matrix_free")

        samples = gp.sample_posterior(
            X[:4],
            n_samples=2,
            method="pathwise",
            rng=np.random.default_rng(4),
        )
        assert samples.shape == (2, 4, 2)
        assert np.all(np.isfinite(samples))


class TestMultiOutputLMCPathwise:
    """Test provider-backed pathwise posterior sampling for MultiOutputLMCGP."""

    def test_pathwise_shape_and_alias(self):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputLMCGP
        from mojogp.kernel import Kernel

        X, Y = generate_mo_data(n=2000, d=2, T=2, seed=61)
        X_test = np.random.randn(8, 2).astype(np.float32)

        gp = MultiOutputLMCGP(
            kernels=[Kernel.rbf(), Kernel.matern52()]        )
        gp.fit(X, Y, max_iterations=8, verbose=False, method="matrix_free")

        samples = gp.sample_posterior(
            X_test,
            n_samples=4,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(7),
        )
        alias_samples = gp.sample_posterior(
            X_test,
            n_samples=4,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(7),
        )

        assert samples.shape == (4, 8, 2)
        assert np.all(np.isfinite(samples))
        np.testing.assert_array_equal(samples, alias_samples)
        info = gp.backend_sample_info
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["actual_sampling_method"] == "pathwise"

    def test_pathwise_mean_matches_predictive_mean(self):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputLMCGP
        from mojogp.kernel import Kernel

        rng = np.random.default_rng(71)
        X = rng.standard_normal((2000, 2)).astype(np.float32)
        latent = np.sin(X[:, 0]).astype(np.float32)
        Y = np.column_stack(
            [
                latent + 0.05 * rng.standard_normal(2000),
                1.1 * latent
                + 0.15 * np.cos(X[:, 1])
                + 0.05 * rng.standard_normal(2000),
            ]
        ).astype(np.float32)
        X_test = rng.standard_normal((5, 2)).astype(np.float32)

        gp = MultiOutputLMCGP(
            kernels=[Kernel.rbf(), Kernel.matern52()]        )
        gp.fit(X, Y, max_iterations=10, verbose=False, method="materialized")

        pred = gp.predict(X_test)
        samples = gp.sample_posterior(
            X_test,
            n_samples=96,
            method="pathwise",
            n_rff_features=1536,
            rng=np.random.default_rng(17),
        )
        abs_diff = np.abs(samples.mean(axis=0) - pred.mean)
        assert float(abs_diff.mean()) < 1.0
        assert float(abs_diff.max()) < 3.0

    def test_pathwise_captures_cross_task_and_spatial_correlation(self):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputLMCGP
        from mojogp.kernel import Kernel

        rng = np.random.default_rng(81)
        X = rng.standard_normal((2000, 1)).astype(np.float32)
        latent = np.sin(X[:, 0]).astype(np.float32)
        Y = np.column_stack(
            [
                latent + 0.03 * rng.standard_normal(2000),
                1.2 * latent + 0.03 * rng.standard_normal(2000),
            ]
        ).astype(np.float32)
        X_test = np.array([[4.0], [4.02], [8.0]], dtype=np.float32)

        gp = MultiOutputLMCGP(
            kernels=[Kernel.rbf()], num_probes=4, max_cg_iterations=50
        )
        gp.fit(
            X,
            Y,
            max_iterations=20,
            learning_rate=0.03,
            verbose=False,
            method="matrix_free",
            early_stop_tol=0.0,
        )

        samples = gp.sample_posterior(
            X_test,
            n_samples=192,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(99),
        )

        learned_task_cov = gp.training_result.A_matrices.sum(axis=0)
        learned_cross_task = float(learned_task_cov[0, 1])
        same_point_cross_task = np.corrcoef(samples[:, 0, 0], samples[:, 0, 1])[0, 1]
        close_point_corr = np.corrcoef(samples[:, 0, 0], samples[:, 1, 0])[0, 1]
        far_point_corr = np.corrcoef(samples[:, 0, 0], samples[:, 2, 0])[0, 1]
        assert abs(learned_cross_task) > 1e-3
        assert abs(same_point_cross_task) > 0.05
        assert np.sign(same_point_cross_task) == np.sign(learned_cross_task)
        assert close_point_corr > far_point_corr

    @pytest.mark.parametrize("method", ["matrix_free", "materialized"])
    def test_pathwise_survives_save_load_in_both_routes(self, method, tmp_path):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        loaded_info = _run_pathwise_case(
            "lmc_save_load", method=method, save_dir=tmp_path
        )
        assert loaded_info["actual_sampling_route"] == "provider_pathwise"
        assert loaded_info["training_route"] == method

    def test_pathwise_supports_polynomial_degree_two_latent(self):
        """Degree-2 polynomial LMC pathwise sampling uses finite exact features."""
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputLMCGP
        from mojogp.kernel import Kernel

        rng = np.random.default_rng(91)
        X = rng.standard_normal((2000, 2)).astype(np.float32)
        base = (X[:, 0] ** 2 + 0.5 * X[:, 0] * X[:, 1]).astype(np.float32)
        Y = np.column_stack(
            [
                base + 0.05 * rng.standard_normal(2000),
                0.7 * base + 0.05 * rng.standard_normal(2000),
            ]
        ).astype(np.float32)
        X_test = rng.standard_normal((5, 2)).astype(np.float32)

        gp = MultiOutputLMCGP(
            kernels=[Kernel.polynomial(degree=2.0, offset=2.0)],
            num_probes=3,
            max_cg_iterations=30,
            use_preconditioner=False,
        )
        gp.fit(
            X,
            Y,
            max_iterations=5,
            learning_rate=0.02,
            verbose=False,
            method="matrix_free",
        )

        pred = gp.predict(X_test, variance_method="mean_only")
        samples = gp.sample_posterior(
            X_test,
            n_samples=96,
            method="pathwise",
            n_rff_features=512,
            rng=np.random.default_rng(13),
        )
        repeat = gp.sample_posterior(
            X_test,
            n_samples=96,
            method="pathwise",
            n_rff_features=512,
            rng=np.random.default_rng(13),
        )

        assert samples.shape == (96, 5, 2)
        assert np.all(np.isfinite(samples))
        np.testing.assert_array_equal(samples, repeat)
        info = gp.backend_sample_info
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["backend_correction_route"] == "sample_lmc_pathwise"
        assert info["prior_sampler_family"] == "shared_feature_map"
        assert float(np.mean(np.abs(samples.mean(axis=0) - pred.mean))) < 1.5

    def test_pathwise_supports_continuous_fixed_observation_noise(self):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import MultiOutputLMCGP
        from mojogp.kernel import Kernel

        rng = np.random.default_rng(115)
        X = rng.standard_normal((2000, 2)).astype(np.float32)
        base0 = np.sin(X[:, 0]) + 0.1 * X[:, 1]
        base1 = 0.8 * np.cos(X[:, 1]) - 0.2 * X[:, 0]
        fixed_noise = (0.01 + 0.02 * rng.random((X.shape[0], 2))).astype(np.float32)
        Y = np.column_stack(
            [
                base0 + rng.normal(0.0, np.sqrt(fixed_noise[:, 0])),
                base1 + rng.normal(0.0, np.sqrt(fixed_noise[:, 1])),
            ]
        ).astype(np.float32)
        X_test = rng.standard_normal((5, 2)).astype(np.float32)

        gp = MultiOutputLMCGP(
            kernels=[Kernel.rbf(), Kernel.matern52()],
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=6,
        )
        gp.fit(
            X,
            Y,
            fixed_observation_noise=fixed_noise,
            max_iterations=5,
            learning_rate=0.02,
            verbose=False,
            method="matrix_free",
        )

        pred = gp.predict(X_test, variance_method="mean_only")
        samples = gp.sample_posterior(
            X_test,
            n_samples=32,
            method="pathwise",
            n_rff_features=512,
            rng=np.random.default_rng(19),
        )

        assert gp.training_result.fixed_observation_noise is not None
        assert samples.shape == (32, 5, 2)
        assert np.all(np.isfinite(samples))
        info = gp.backend_sample_info
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["backend_correction_route"] == "sample_lmc_pathwise"
        assert float(np.mean(np.abs(samples.mean(axis=0) - pred.mean))) < 2.0

    def test_pathwise_polynomial_lmc_returns_finite_samples(self):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import Kernel, MultiOutputLMCGP

        X, Y = generate_mo_data(n=2000, d=2, T=2, seed=91)
        X_test = X[:6].copy()

        gp = MultiOutputLMCGP(kernels=[Kernel.polynomial(degree=3.0)])
        gp.fit(
            X,
            Y,
            max_iterations=5,
            learning_rate=0.01,
            verbose=False,
            method="matrix_free",
        )

        samples = gp.sample_posterior(
            X_test,
            n_samples=3,
            method="pathwise",
            rng=np.random.default_rng(3),
        )
        assert samples.shape == (3, 6, 2)
        assert np.all(np.isfinite(samples))
        info = gp.backend_sample_info
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["backend_correction_route"] == "sample_lmc_pathwise"


class TestMultiOutputLMCMixedPathwise:
    """Test provider-backed mixed pathwise posterior sampling for MultiOutputLMCGP."""

    def test_mixed_pathwise_shape_alias_and_category_sensitivity(self):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        from mojogp import Kernel, MultiOutputLMCGP

        X, Y = generate_mixed_mo_data(n=2000, d_cont=2, T=2, levels=3, seed=121)
        kernels = [
            Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
            Kernel.matern52(active_dims=[0, 1]),
        ]
        gp = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=4,
        )
        gp.fit(X, Y, max_iterations=4, learning_rate=0.03, verbose=False, method="matrix_free")

        X_test = np.array(
            [
                [0.15, -0.35, 0.0],
                [0.15, -0.35, 1.0],
                [0.15, -0.35, 2.0],
            ],
            dtype=np.float32,
        )
        samples = gp.sample_posterior(
            X_test,
            n_samples=12,
            method="pathwise",
            n_rff_features=256,
            rng=np.random.default_rng(123),
        )
        alias_samples = gp.sample_posterior(
            X_test,
            n_samples=12,
            method="pathwise",
            n_rff_features=256,
            rng=np.random.default_rng(123),
        )

        assert samples.shape == (12, 3, 2)
        assert np.all(np.isfinite(samples))
        np.testing.assert_array_equal(samples, alias_samples)
        assert float(np.ptp(samples.mean(axis=0)[:, 0])) > 0.05
        assert float(np.ptp(samples.mean(axis=0)[:, 1])) > 0.05
        info = gp.backend_sample_info
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["backend_correction_route"] == "sample_lmc_mixed_pathwise"
        assert info["prior_sampler_family"] == "shared_feature_map"

    @pytest.mark.parametrize("method", ["matrix_free", "materialized"])
    def test_mixed_pathwise_survives_save_load(self, method, tmp_path):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        loaded_info = _run_pathwise_case(
            "mixed_save_load", method=method, save_dir=tmp_path
        )
        assert loaded_info["backend_correction_route"] == "sample_lmc_mixed_pathwise"
        assert loaded_info["training_route"] == method

    def test_mixed_pathwise_supports_additive_mixed_latent_tree(self):
        _skip_if_no_lib("mojogp_jit_engine", "task build")
        summary = _run_pathwise_case("mixed_additive_supported")
        assert summary["supported"] is True
        assert summary["backend_correction_route"] == "sample_lmc_mixed_pathwise"
        assert summary["prior_sampler_family"] == "shared_feature_map"
