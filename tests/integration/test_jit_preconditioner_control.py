"""JIT-native checks for preconditioner and routing behavior.

These tests exercise the public wrapper surface and backend telemetry for
preconditioner selection, route metadata, and mixed-kernel routing.
"""

from __future__ import annotations

import numpy as np
import pytest


def _single_output_data(seed: int = 11, n: int = 2000, d: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (
        np.sin(X[:, 0]) + 0.25 * np.cos(X[:, 1]) + 0.05 * rng.standard_normal(n)
    ).astype(np.float32)
    return X, y


def _multi_output_data(seed: int = 21, n: int = 2000, d: int = 3, t: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    base = np.sin(X[:, 0]).astype(np.float32)
    Y = np.column_stack(
        [
            base + 0.05 * rng.standard_normal(n),
            0.8 * base + 0.2 * np.cos(X[:, 1]) + 0.05 * rng.standard_normal(n),
        ][:t]
    ).astype(np.float32)
    return X, Y


def _mixed_multi_output_data(
    seed: int = 31, n: int = 2000, d_cont: int = 2, t: int = 2, levels: int = 3
):
    rng = np.random.default_rng(seed)
    X_cont = rng.standard_normal((n, d_cont)).astype(np.float32)
    cats = rng.integers(0, levels, size=(n, 1), endpoint=False).astype(np.float32)
    X = np.concatenate([X_cont, cats], axis=1).astype(np.float32)
    shared = np.sin(X_cont[:, 0]) + 0.3 * cats[:, 0]
    Y = np.column_stack(
        [
            shared + 0.05 * rng.standard_normal(n),
            0.9 * shared + 0.2 * np.cos(X_cont[:, 1]) + 0.05 * rng.standard_normal(n),
        ][:t]
    ).astype(np.float32)
    return X, Y


@pytest.mark.parametrize(
    ("precond", "expected_method"),
    [("greedy", 0), ("rpcholesky", 1), ("nystrom", 2)],
)
def test_exactgp_preconditioner_method_routes_into_live_jit_backend(
    precond, expected_method
):
    from mojogp import SingleOutputGP
    from mojogp.kernel import Kernel

    X, y = _single_output_data()
    gp = SingleOutputGP(Kernel.rbf())
    result = gp.fit(
        X,
        y,
        method="matrix_free",
        max_iterations=4,
        preconditioner=precond,
        preconditioner_rank=12,
        verbose=False,
    )

    assert result.iterations == 4
    assert result.nll_history is not None
    assert len(result.nll_history) == 4
    assert np.all(np.isfinite(result.nll_history))
    assert np.isfinite(result.nll)
    assert gp.backend_train_info is not None
    assert gp.backend_train_info["training_route"] == "matrix_free"
    assert gp.backend_train_info["precond_method"] == expected_method
    assert gp.backend_train_info["precond_rank"] == 12


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_exactgp_nested_mixed_tree_predicts_with_backend_metadata(method):
    from mojogp import SingleOutputGP
    from mojogp.kernel import Kernel

    X, y_base = _single_output_data(seed=12, n=2000, d=2)
    rng = np.random.default_rng(12)
    cat = rng.integers(0, 3, size=(X.shape[0], 1), endpoint=False).astype(np.float32)
    X_full = np.concatenate([X, cat], axis=1).astype(np.float32)
    y = (y_base + 0.2 * cat[:, 0]).astype(np.float32)
    X_test = X_full[:8].copy()

    kernel = (Kernel.rbf(active_dims=[0, 1]) + Kernel.matern32(active_dims=[0, 1])) * (
        Kernel.gd(levels=3, active_dims=[2]) + Kernel.cr(levels=3, active_dims=[2])
    )
    gp = SingleOutputGP(kernel)
    gp.fit(
        X_full,
        y,
        method=method,
        max_iterations=3,
        preconditioner="nystrom",
        preconditioner_rank=10,
        verbose=False,
    )
    pred = gp.predict(X_test, variance_method="exact")
    mean = pred.mean
    var = pred.variance

    assert mean.shape == (8,)
    assert var.shape == (8,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert gp.backend_train_info is not None
    assert gp.backend_train_info["training_route"] == method
    assert gp.backend_train_info["precond_method"] == 2
    assert gp.backend_train_info["precond_rank"] == 10
    assert gp.backend_predict_info is not None
    assert gp.backend_predict_info["requested_method"] == method
    assert gp.backend_predict_info["training_route"] == method
    assert gp.backend_predict_info["actual_prediction_route"] == "predict_mixed"
    assert gp.backend_predict_info["actual_variance_route"] == "predict_mixed"
    assert gp.backend_predict_info["backend_prediction_used"] is True
    assert gp.backend_predict_info["backend_variance_used"] is True
    assert gp.backend_predict_info["fallback_used"] is False
    assert gp.backend_predict_info["telemetry_quality"] == "observed"
    assert gp.backend_predict_info["configured_for_cg"] is True
    assert gp.backend_predict_info["observed_cg_calls"] is True


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_multioutputgp_reports_live_route_and_preconditioner_metadata(method):
    from mojogp import MultiOutputGP
    from mojogp.kernel import Kernel

    X, Y = _multi_output_data()
    X_test = X[:10].copy()
    gp = MultiOutputGP(
        kernel=Kernel.rbf() + Kernel.matern52(),
        num_probes=3,
        max_cg_iterations=30,
        preconditioner="greedy",
        preconditioner_rank=8,
    )

    result = gp.fit(X, Y, max_iterations=4, verbose=False, method=method)
    mean, var = gp.predict(X_test, return_var=True, variance_method="exact")

    assert result.nll_history is not None
    assert 1 <= len(result.nll_history) <= 4
    assert mean.shape == (10, 2)
    assert var.shape == (10, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert gp.backend_train_info is not None
    assert gp.backend_train_info["training_route"] == method
    assert gp.backend_train_info["precond_method"] == 0
    assert gp.backend_train_info["precond_rank"] == 8
    assert gp.backend_predict_info is not None
    assert gp.backend_predict_info["requested_method"] == method
    assert gp.backend_predict_info["training_route"] == method
    assert gp.backend_predict_info["backend_prediction_used"] is True
    assert gp.backend_predict_info["backend_variance_used"] is True
    assert gp.backend_predict_info["fallback_used"] is False
    assert gp.backend_predict_info["precond_method"] == 0
    assert gp.backend_predict_info["precond_rank"] == 8
    assert gp.backend_predict_info["precond_rebuild_count"] >= 0
    assert gp.backend_predict_info["actual_prediction_route"] != "python_fallback"
    assert gp.backend_predict_info["actual_variance_route"] != "python_fallback"


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_multioutputgp_default_preconditioner_is_greedy(method):
    from mojogp import MultiOutputGP
    from mojogp.kernel import Kernel

    X, Y = _multi_output_data(seed=22)
    gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        num_probes=3,
        max_cg_iterations=30,
    )

    result = gp.fit(X, Y, max_iterations=3, verbose=False, method=method)

    assert result.iterations >= 1
    assert gp.precond == "greedy"
    assert gp.precond_method == 0
    assert gp.backend_train_info is not None
    assert gp.backend_train_info["precond_method"] == 0


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_multioutputgp_use_preconditioner_false_disables_live_backend_preconditioning(
    method,
):
    from mojogp import MultiOutputGP
    from mojogp.kernel import Kernel

    X, Y = _multi_output_data(seed=23)
    gp = MultiOutputGP(
        kernel=Kernel.rbf(),
        num_probes=3,
        max_cg_iterations=30,
        preconditioner_rank=8,
        use_preconditioner=False,
    )

    result = gp.fit(X, Y, max_iterations=3, verbose=False, method=method)
    mean, var = gp.predict(X[:8], return_var=True, variance_method="exact")

    assert result.iterations >= 1
    assert mean.shape == (8, 2)
    assert var.shape == (8, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert gp.use_preconditioner is False
    assert gp.precond_rank == 0
    assert gp.backend_train_info is not None
    assert gp.backend_train_info["training_route"] == method
    assert gp.backend_train_info["use_preconditioner"] is False
    assert gp.backend_train_info["precond_rank"] == 0
    assert gp.backend_predict_info is not None
    assert gp.backend_predict_info["requested_method"] == method
    assert gp.backend_predict_info["training_route"] == method
    assert gp.backend_predict_info["precond_rank"] == 0


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_mixed_exactgp_default_preconditioner_propagates_to_backend(method):
    from mojogp import SingleOutputGP
    from mojogp.kernel import Kernel

    X, y_base = _single_output_data(seed=13, n=2000, d=2)
    rng = np.random.default_rng(13)
    cat = rng.integers(0, 3, size=(X.shape[0], 1), endpoint=False).astype(np.float32)
    X_full = np.concatenate([X, cat], axis=1).astype(np.float32)
    y = (y_base + 0.15 * cat[:, 0]).astype(np.float32)

    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.gd(levels=3, active_dims=[2])
    gp = SingleOutputGP(kernel)
    result = gp.fit(
        X_full,
        y,
        method=method,
        max_iterations=3,
        preconditioner_rank=9,
        verbose=False,
    )

    assert result.iterations >= 1
    assert gp.backend_train_info is not None
    assert gp.backend_train_info["training_route"] == method
    assert gp.backend_train_info["precond_method"] == 0
    assert gp.backend_train_info["precond_rank"] == 9


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_mixed_lmc_save_load_keeps_backend_prediction_route(method, tmp_path):
    from mojogp import Kernel, MultiOutputLMCGP

    X, Y = _mixed_multi_output_data()
    X_test = X[:6].copy()
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
    gp.fit(X, Y, max_iterations=3, learning_rate=0.03, verbose=False, method=method)

    mean, var = gp.predict(X_test, return_var=True, variance_method="exact")
    assert mean.shape == (6, 2)
    assert var.shape == (6, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert gp.backend_predict_info is not None
    assert gp.backend_predict_info["requested_method"] == method
    assert gp.backend_predict_info["training_route"] == method
    assert gp.backend_predict_info["actual_prediction_route"] == "predict_lmc_mixed"
    assert gp.backend_predict_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"
    assert gp.backend_predict_info["backend_prediction_used"] is True
    assert gp.backend_predict_info["backend_variance_used"] is True
    assert gp.backend_predict_info["fallback_used"] is False

    save_path = tmp_path / f"mixed_lmc_route_{method}"
    gp.save(save_path)
    loaded = MultiOutputLMCGP.load(save_path)
    loaded_mean, loaded_var = loaded.predict(
        X_test, return_var=True, variance_method="exact"
    )

    assert loaded.backend_predict_info is not None
    assert loaded.backend_predict_info["requested_method"] == method
    assert loaded.backend_predict_info["training_route"] == method
    assert loaded.backend_predict_info["actual_prediction_route"] == "predict_lmc_mixed"
    assert loaded.backend_predict_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"
    assert loaded.backend_predict_info["backend_prediction_used"] is True
    assert loaded.backend_predict_info["backend_variance_used"] is True
    assert loaded.backend_predict_info["fallback_used"] is False
    np.testing.assert_allclose(loaded_mean, mean, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(loaded_var, var, atol=5e-2, rtol=2e-1)


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_mixed_lmc_default_preconditioner_is_greedy(method):
    from mojogp import Kernel, MultiOutputLMCGP

    X, Y = _mixed_multi_output_data(seed=32)
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

    result = gp.fit(X, Y, max_iterations=3, learning_rate=0.03, verbose=False, method=method)

    assert result.iterations >= 1
    assert gp.precond == "greedy"
    assert gp.precond_method == 0
    assert gp._backend_train_info is not None
    assert gp._backend_train_info["precond_method"] == 0


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_lmc_use_preconditioner_false_disables_live_backend_preconditioning(method):
    from mojogp import MultiOutputLMCGP

    X, Y = _multi_output_data(seed=33)
    gp = MultiOutputLMCGP(
        kernels=["rbf", "matern52"],
        num_probes=3,
        max_cg_iterations=30,
        preconditioner_rank=8,
        use_preconditioner=False,
    )

    result = gp.fit(X, Y, max_iterations=3, learning_rate=0.03, verbose=False, method=method)
    mean, var = gp.predict(X[:6], return_var=True, variance_method="exact")

    assert result.iterations >= 1
    assert mean.shape == (6, 2)
    assert var.shape == (6, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert gp.use_preconditioner is False
    assert gp.precond_rank == 0
    assert gp.backend_train_info is not None
    assert gp.backend_train_info["training_route"] == method
    assert gp.backend_train_info["use_preconditioner"] is False
    assert gp.backend_train_info["precond_rank"] == 0
    assert gp.backend_predict_info is not None
    assert gp.backend_predict_info["requested_method"] == method
    assert gp.backend_predict_info["training_route"] == method
    assert gp.backend_predict_info["precond_rank"] == 0
