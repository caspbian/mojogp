"""Integration coverage for JIT composite and LMC ARD wrapper workflows.

These cases train real JIT-backed models and assert public wrapper behavior,
backend telemetry, prediction, and save/load contracts.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from mojogp import SingleOutputGP, MultiOutputGP, MultiOutputLMCGP
from mojogp.kernel import Kernel


pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_gpu():
    import torch

    assert torch.cuda.is_available(), "CUDA GPU required for JIT integration tests"


def _multi_output_data(seed: int = 42, n: int = 2000, d: int = 3, t: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = np.column_stack(
        [
            np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n),
            np.cos(X[:, 1]) + 0.1 * rng.standard_normal(n),
        ][:t]
    ).astype(np.float32)
    X_test = rng.standard_normal((10, d)).astype(np.float32)
    return X, Y, X_test


def test_lmc_composite_save_load_restores_kernel_tree_without_explicit_kernels():
    X, Y, X_test = _multi_output_data(seed=11)
    kernel = Kernel.rbf() + Kernel.matern52()
    gp = MultiOutputLMCGP(kernels=[kernel], num_probes=2, max_cg_iterations=20)
    gp.fit(X, Y, max_iterations=4, learning_rate=0.03, verbose=False)

    pred_before = gp.predict(X_test)
    mean_before = pred_before.mean

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "lmc_composite")
        gp.save(path)
        gp_loaded = MultiOutputLMCGP.load(path)
        pred_after = gp_loaded.predict(X_test)

    np.testing.assert_allclose(mean_before, pred_after.mean, rtol=1e-5, atol=1e-5)


def test_lmc_composite_variance_is_posterior():
    X, Y, X_test = _multi_output_data(seed=42, d=2)
    kernel = Kernel.rbf() + Kernel.matern52()
    gp = MultiOutputLMCGP(kernels=[kernel], num_probes=2, max_cg_iterations=20)
    gp.fit(X, Y, max_iterations=6, learning_rate=0.03, verbose=False)

    _, var = gp.predict(X_test[:5], return_var=True)

    assert var is not None
    assert np.all(var > 0), "Variance should be positive"
    assert np.all(var < 10), f"Variance seems too large: max={var.max()}"


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_composite_exact_and_love_use_backend_routes(method):
    X, Y, X_test = _multi_output_data(seed=54, d=2)
    kernel = 1.2 * (Kernel.rbf() + Kernel.matern52())
    gp = MultiOutputLMCGP(kernels=[kernel], num_probes=3, max_cg_iterations=30)
    gp.fit(X, Y, max_iterations=6, learning_rate=0.03, method=method, verbose=False)

    exact = gp.predict(X_test[:6], variance_method="exact")
    exact_info = dict(gp.backend_predict_info)
    love = gp.predict(X_test[:6], variance_method="love")
    love_info = dict(gp.backend_predict_info)

    expected_exact_route = (
        "dense_exact_lmc"
        if method == "materialized"
        else "predict_lmc_full_exact"
    )
    assert gp.backend_train_info["training_route"] == method
    assert exact_info["actual_variance_route"] == expected_exact_route
    assert love_info["variance_method"] == "love"
    assert love_info["backend_variance_used"] is True
    assert love_info["actual_variance_route"] == "predict_lmc"
    np.testing.assert_allclose(exact.mean, love.mean, rtol=1e-5, atol=1e-5)
    assert np.all(np.isfinite(exact.variance))
    assert np.all(np.isfinite(love.variance))
    assert np.all(exact.variance >= 0)
    assert np.all(love.variance >= 0)


def test_lmc_composite_save_load_preserves_exact_and_love_predictions(tmp_path):
    X, Y, X_test = _multi_output_data(seed=55, d=2)
    kernels = [Kernel.rbf() + Kernel.matern52(), Kernel.rbf() * Kernel.rq()]
    gp = MultiOutputLMCGP(kernels=kernels, num_probes=3, max_cg_iterations=30)
    gp.fit(X, Y, max_iterations=6, learning_rate=0.03, method="materialized", verbose=False)

    exact_before = gp.predict(X_test[:5], variance_method="exact")
    love_before = gp.predict(X_test, variance_method="love")
    path = tmp_path / "lmc_composite_routes"
    gp.save(str(path))
    loaded = MultiOutputLMCGP.load(str(path))

    exact_after = loaded.predict(X_test[:5], variance_method="exact")
    exact_info = dict(loaded.backend_predict_info)
    love_after = loaded.predict(X_test, variance_method="love")
    love_info = dict(loaded.backend_predict_info)

    np.testing.assert_allclose(exact_before.mean, exact_after.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(exact_before.variance, exact_after.variance, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(love_before.mean, love_after.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(love_before.variance, love_after.variance, rtol=5e-2, atol=5e-3)
    assert exact_info["actual_variance_route"] == "dense_exact_lmc"
    assert love_info["actual_variance_route"] == "predict_lmc"


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_initial_params_seed_backend_latent_kernel_params(method):
    X, Y, _ = _multi_output_data(seed=57, d=2)
    init_params = np.array([0.35, 1.40, 1.80, 0.70], dtype=np.float32)
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=2,
        max_cg_iterations=20,
        use_preconditioner=False,
    )

    result = gp.fit(
        X,
        Y,
        max_iterations=1,
        learning_rate=1e-12,
        initial_params=init_params,
        method=method,
        verbose=False,
    )

    assert result.params_per_latent is not None
    np.testing.assert_allclose(
        np.concatenate([np.asarray(p, dtype=np.float32) for p in result.params_per_latent]),
        init_params,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_active_dims_exact_love_and_save_load_routes(method, tmp_path):
    X, Y, X_test = _multi_output_data(seed=56, d=3)
    kernels = [
        Kernel.rbf(active_dims=[0, 2]),
        Kernel.matern52(active_dims=[1]),
    ]
    gp = MultiOutputLMCGP(kernels=kernels, num_probes=3, max_cg_iterations=30)
    result = gp.fit(X, Y, max_iterations=6, learning_rate=0.03, method=method, verbose=False)

    exact_before = gp.predict(X_test[:5], variance_method="exact")
    exact_info = dict(gp.backend_predict_info)
    love_before = gp.predict(X_test, variance_method="love")
    love_info = dict(gp.backend_predict_info)
    path = tmp_path / f"lmc_active_dims_{method}"
    gp.save(str(path))
    loaded = MultiOutputLMCGP.load(str(path))
    exact_after = loaded.predict(X_test[:5], variance_method="exact")
    loaded_exact_info = dict(loaded.backend_predict_info)
    love_after = loaded.predict(X_test, variance_method="love")
    loaded_love_info = dict(loaded.backend_predict_info)

    expected_exact_route = (
        "dense_exact_lmc"
        if method == "materialized"
        else "predict_lmc_mixed_full_exact"
    )
    expected_per_latent_route = "predict_lmc_mixed"
    assert gp.backend_train_info["training_route"] == method
    assert result.params_per_latent is not None
    assert [len(p) for p in result.params_per_latent] == [2, 2]
    assert exact_info["actual_variance_route"] == expected_exact_route
    assert loaded_exact_info["actual_variance_route"] == expected_exact_route
    assert love_info["actual_variance_route"] == expected_per_latent_route
    assert loaded_love_info["actual_variance_route"] == expected_per_latent_route
    np.testing.assert_allclose(exact_before.mean, exact_after.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(exact_before.variance, exact_after.variance, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(love_before.mean, love_after.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(love_before.variance, love_after.variance, rtol=5e-2, atol=5e-3)
    assert np.all(np.isfinite(exact_before.variance))
    assert np.all(np.isfinite(love_before.variance))


def test_multi_output_composite_exact_variance_returns_finite_values():
    X, Y, X_test = _multi_output_data(seed=43, d=2)
    kernel = Kernel.rbf() + Kernel.matern52()
    gp = MultiOutputGP(kernel=kernel, num_probes=2, max_cg_iterations=20)
    gp.fit(X, Y, max_iterations=6, learning_rate=0.03, verbose=False)

    mean, var = gp.predict(X_test[:5], return_var=True, variance_method="exact")

    assert mean.shape == (5, Y.shape[1])
    assert var.shape == (5, Y.shape[1])
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert np.all(var >= 0)


def test_single_output_composite_exact_variance_returns_finite_values():
    rng = np.random.default_rng(44)
    n, d = 2000, 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)).astype(np.float32)
    X_test = rng.standard_normal((10, d)).astype(np.float32)

    gp = SingleOutputGP(Kernel.rbf() + Kernel.matern52(), verbose=False)
    gp.fit(X, y, max_iterations=10, learning_rate=0.03, verbose=False)

    result = gp.predict(X_test, variance_method="exact")

    assert result.mean.shape == (10,)
    assert result.variance.shape == (10,)
    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.variance))
    assert np.all(result.variance >= 0)


def test_single_output_composite_exact_variance_is_posterior_at_training_points():
    rng = np.random.default_rng(45)
    n, d = 2000, 2
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)).astype(np.float32)

    gp = SingleOutputGP(Kernel.rbf() + Kernel.matern52(), verbose=False)
    gp.fit(X, y, max_iterations=10, learning_rate=0.03, verbose=False)

    result = gp.predict(X[:5].copy(), variance_method="exact")

    assert np.all(result.variance < 5.0), (
        f"Variance at training points too large: max={result.variance.max():.4f}"
    )


def test_lmc_composite_jit_compile_train_and_predicts():
    X, Y, _ = _multi_output_data(seed=46)
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf() + Kernel.matern52(), Kernel.rbf() + Kernel.matern52()],
        num_probes=2,
        max_cg_iterations=20,
    )
    result = gp.fit(X, Y, max_iterations=6, learning_rate=0.03, verbose=False)

    assert result.iterations > 0
    assert result.final_nll < 1e10
    assert result.A_matrices.shape == (2, Y.shape[1], Y.shape[1])

    X_test = X[:20].copy()
    mean, var = gp.predict(X_test, return_var=True)
    assert mean.shape == (20, Y.shape[1])
    assert var.shape == (20, Y.shape[1])


class TestLMCARDIntegration:
    """Integration checks for LMC with per-dimension lengthscales."""

    def test_lmc_ard_returns_finite_nll(self):
        X, Y, _ = _multi_output_data(seed=47)
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"], ard=True, num_probes=2, max_cg_iterations=20
        )
        result = gp.fit(X, Y, max_iterations=6, learning_rate=0.03, verbose=False)

        assert result is not None
        assert result.use_ard is True
        assert np.isfinite(result.final_nll)

    def test_lmc_ard_per_dim_lengthscales(self):
        X, Y, _ = _multi_output_data(seed=48)
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"], ard=True, num_probes=2, max_cg_iterations=20
        )
        result = gp.fit(X, Y, max_iterations=6, learning_rate=0.03, verbose=False)

        assert result.lengthscales.shape == (6,)
        assert np.all(result.lengthscales > 0)
        assert result.lengthscales_per_dim is not None
        assert result.lengthscales_per_dim.shape == (2, 3)

    def test_lmc_ard_prediction_shapes(self):
        X, Y, _ = _multi_output_data(seed=49)
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"], ard=True, num_probes=2, max_cg_iterations=20
        )
        gp.fit(X, Y, max_iterations=6, learning_rate=0.03, verbose=False)

        X_test = X[:15].copy()
        mean, std = gp.predict(X_test, return_std=True)

        assert mean.shape == (15, Y.shape[1])
        assert std.shape == (15, Y.shape[1])
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_lmc_ard_save_load_preserves_lengthscales_and_predictions(self, tmp_path):
        X, Y, X_test = _multi_output_data(seed=52)
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"], ard=True, num_probes=2, max_cg_iterations=20
        )
        result = gp.fit(X, Y, max_iterations=6, learning_rate=0.03, verbose=False)

        assert result.lengthscales_per_dim is not None
        love_before = gp.predict(X_test, variance_method="love")
        exact_before = gp.predict(X_test[:5], variance_method="exact")
        path = tmp_path / "lmc_ard"
        gp.save(str(path))
        loaded = MultiOutputLMCGP.load(str(path))

        assert loaded.training_result.lengthscales_per_dim is not None
        np.testing.assert_allclose(
            loaded.training_result.lengthscales_per_dim,
            result.lengthscales_per_dim,
            rtol=0,
            atol=0,
        )
        love_after = loaded.predict(X_test, variance_method="love")
        exact_after = loaded.predict(X_test[:5], variance_method="exact")

        np.testing.assert_allclose(love_before.mean, love_after.mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(love_before.variance, love_after.variance, rtol=5e-2, atol=5e-3)
        np.testing.assert_allclose(exact_before.mean, exact_after.mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(exact_before.variance, exact_after.variance, rtol=1e-5, atol=1e-5)

    def test_lmc_ard_nll_decreases_to_target(self):
        X, Y, _ = _multi_output_data(seed=50)
        gp = MultiOutputLMCGP(
            kernels=["rbf", "matern52"], ard=True, num_probes=2, max_cg_iterations=20
        )
        result = gp.fit(X, Y, max_iterations=10, learning_rate=0.03, verbose=False)

        assert result.final_nll < 2.0, f"NLL did not decrease enough: {result.final_nll}"

    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    def test_lmc_ard_recovers_single_relevant_dimension(self, method):
        rng = np.random.default_rng(70)
        n, d, t = 2000, 3, 2
        X = rng.standard_normal((n, d)).astype(np.float32)
        latent = np.sin(2.0 * X[:, 0])
        Y = np.zeros((n, t), dtype=np.float32)
        Y[:, 0] = latent + 0.05 * rng.standard_normal(n)
        Y[:, 1] = 1.5 * latent + 0.05 * rng.standard_normal(n)

        gp = MultiOutputLMCGP(
            kernels=["rbf"],
            ard=True,
            num_probes=16,
            max_cg_iterations=100,
            cg_tolerance=1e-4,
            preconditioner_rank=20,
        )
        result = gp.fit(
            X,
            Y,
            method=method,
            max_iterations=120,
            learning_rate=0.003,
            initial_noise_per_task=np.full(t, 0.05, dtype=np.float32),
            early_stop_tol=0.0,
            verbose=False,
        )

        lengthscales = result.lengthscales_per_dim
        assert lengthscales is not None
        assert gp.backend_train_info["training_route"] == method
        avg_lengthscales = lengthscales.mean(axis=0)
        assert np.all(np.isfinite(avg_lengthscales))
        assert np.all(avg_lengthscales > 0.0)
        relevance_margin = float(np.min(avg_lengthscales[1:]) - avg_lengthscales[0])
        assert relevance_margin >= 0.15, (
            "LMC ARD did not separate the relevant dimension on the documented "
            f"synthetic workflow: avg_lengthscales={avg_lengthscales.tolist()}, "
            f"margin={relevance_margin}"
        )

        pred = gp.predict(X[:6].copy(), variance_method="love")
        info = dict(gp.backend_predict_info)
        assert pred.mean.shape == (6, t)
        assert pred.variance.shape == (6, t)
        assert info["actual_variance_route"] == "predict_lmc"
        assert info["fallback_used"] is False


class TestLMCAdditionalContinuousKernels:
    """Route checks for non-RBF/Matern continuous LMC kernels."""

    @staticmethod
    def _assert_love_round_trip_within_approximation_bound(before, after):
        rel_error = np.abs(before - after) / np.maximum(np.abs(after), 1e-6)
        assert float(np.median(rel_error)) < 0.15
        assert float(np.max(np.abs(before - after))) < 0.05

    @staticmethod
    def _assert_love_matches_exact_prediction(gp, X_test, method):
        exact = gp.predict(X_test[:6], variance_method="exact")
        exact_info = dict(gp.backend_predict_info)
        love = gp.predict(X_test[:6], variance_method="love")
        love_info = dict(gp.backend_predict_info)

        expected_exact_route = (
            "dense_exact_lmc"
            if method == "materialized"
            else "predict_lmc_full_exact"
        )
        assert exact_info["actual_variance_route"] == expected_exact_route
        assert love_info["variance_method"] == "love"
        assert love_info["backend_variance_used"] is True
        assert love_info["actual_variance_route"] == "predict_lmc"
        np.testing.assert_allclose(exact.mean, love.mean, rtol=1e-5, atol=1e-5)
        assert np.all(np.isfinite(exact.variance))
        assert np.all(np.isfinite(love.variance))
        assert np.all(exact.variance >= 0)
        assert np.all(love.variance >= 0)
        rel_error = np.abs(love.variance - exact.variance) / np.maximum(
            exact.variance, 1e-6
        )
        assert float(np.median(rel_error)) < 1.0
        return exact, love

    @pytest.mark.parametrize("kernel_name", ["rq", "periodic", "linear"])
    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    def test_lmc_continuous_kernel_exact_love_and_save_load_routes(
        self, kernel_name, method, tmp_path
    ):
        X, Y, X_test = _multi_output_data(seed=70, n=2000, d=3, t=2)
        gp = MultiOutputLMCGP(
            kernels=[kernel_name],
            num_probes=2,
            max_cg_iterations=20,
            use_preconditioner=False,
        )

        result = gp.fit(
            X,
            Y,
            max_iterations=3,
            learning_rate=0.02,
            method=method,
            verbose=False,
        )

        assert np.isfinite(result.final_nll)
        assert gp.backend_train_info["training_route"] == method
        exact_before, love_before = self._assert_love_matches_exact_prediction(
            gp, X_test, method
        )

        path = tmp_path / f"lmc_{kernel_name}_{method}"
        gp.save(path)
        loaded = MultiOutputLMCGP.load(path)
        exact_after, love_after = self._assert_love_matches_exact_prediction(
            loaded, X_test, method
        )
        np.testing.assert_allclose(exact_before.mean, exact_after.mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(exact_before.variance, exact_after.variance, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(love_before.mean, love_after.mean, rtol=1e-5, atol=1e-5)
        self._assert_love_round_trip_within_approximation_bound(
            love_before.variance, love_after.variance
        )

    @pytest.mark.parametrize("method", ["materialized", "matrix_free"])
    def test_lmc_polynomial_degree_two_exact_love_and_save_load_routes(
        self, method, tmp_path
    ):
        X, Y, X_test = _multi_output_data(seed=71, n=2000, d=3, t=2)
        gp = MultiOutputLMCGP(
            kernels=[Kernel.polynomial(degree=2.0, offset=2.0)],
            num_probes=2,
            max_cg_iterations=20,
            use_preconditioner=False,
        )

        result = gp.fit(
            X,
            Y,
            max_iterations=3,
            learning_rate=0.02,
            method=method,
            verbose=False,
        )

        assert np.isfinite(result.final_nll)
        assert gp.backend_train_info["training_route"] == method
        exact_before, love_before = self._assert_love_matches_exact_prediction(
            gp, X_test, method
        )

        path = tmp_path / f"lmc_polynomial_{method}"
        gp.save(path)
        loaded = MultiOutputLMCGP.load(path)
        exact_after, love_after = self._assert_love_matches_exact_prediction(
            loaded, X_test, method
        )
        np.testing.assert_allclose(exact_before.mean, exact_after.mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(exact_before.variance, exact_after.variance, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(love_before.mean, love_after.mean, rtol=1e-5, atol=1e-5)
        self._assert_love_round_trip_within_approximation_bound(
            love_before.variance, love_after.variance
        )
