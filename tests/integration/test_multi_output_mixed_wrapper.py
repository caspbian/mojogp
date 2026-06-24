"""Integration tests for mixed continuous+categorical MultiOutputGP."""

import warnings

import numpy as np
import pytest

from mojogp import MultiOutputGP
from mojogp.kernel import Kernel


def _generate_mixed_multi_output_data(n: int = 2000, seed: int = 0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, 3).astype(np.float32)
    x[:, 2] = rng.randint(0, 3, size=n).astype(np.float32)

    cat = x[:, 2].astype(np.int32)
    y1 = np.sin(x[:, 0]) + 0.2 * x[:, 1] + 0.3 * (cat == 1) - 0.2 * (cat == 2)
    y2 = np.cos(x[:, 1]) - 0.1 * x[:, 0] + 0.4 * (cat == 2)
    y = np.stack([y1, y2], axis=1).astype(np.float32)
    y += 0.03 * rng.randn(n, 2).astype(np.float32)
    return x, y


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_mixed_multioutput_fit_predict_and_save_load(tmp_path, method):
    """Mixed MultiOutputGP should train, predict, save, and load on both methods."""
    n_train = 5000 if method == "materialized" else 10000
    x, y = _generate_mixed_multi_output_data(
        n=n_train, seed=17 if method == "materialized" else 23
    )
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])

    gp = MultiOutputGP(
        kernel=kernel,
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    result = gp.fit(x, y, max_iterations=1, learning_rate=0.03, verbose=False, method=method)

    assert gp.is_trained
    assert result.num_tasks == 2
    assert result.cat_params is not None
    assert result.cat_params.size > 0

    mean, var = gp.predict(x[:16], return_var=True)
    assert mean.shape == (16, 2)
    assert var.shape == (16, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert np.all(var >= 0)

    path = tmp_path / f"mixed_multi_output_{method}"
    gp.save(path)
    loaded = MultiOutputGP.load(path, kernel=kernel)

    mean_loaded, var_loaded = loaded.predict(x[:16], return_var=True)
    assert loaded._training_method == method
    assert loaded.training_result.cat_params is not None
    assert mean_loaded.shape == (16, 2)
    assert var_loaded.shape == (16, 2)
    assert np.all(np.isfinite(mean_loaded))
    assert np.all(np.isfinite(var_loaded))
    assert np.all(var_loaded >= 0)


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_mixed_multioutput_predictions_change_with_category(method):
    """Changing only categorical levels should change mixed MultiOutputGP predictions."""
    n_train = 5000 if method == "materialized" else 10000
    x, y = _generate_mixed_multi_output_data(
        n=n_train, seed=101 if method == "materialized" else 131
    )
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])

    gp = MultiOutputGP(
        kernel=kernel,
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp.fit(x, y, max_iterations=3, learning_rate=0.03, verbose=False, method=method)

    x_probe = np.array(
        [
            [0.25, -0.5, 0.0],
            [0.25, -0.5, 1.0],
            [0.25, -0.5, 2.0],
        ],
        dtype=np.float32,
    )
    mean = gp.predict(x_probe).mean

    assert mean.shape == (3, 2)
    assert np.all(np.isfinite(mean))
    assert float(np.ptp(mean[:, 0])) > 0.05
    assert float(np.ptp(mean[:, 1])) > 0.05


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_mixed_multioutput_supports_nested_mixed_tree(method):
    """Mixed MultiOutputGP should handle nested sum/product trees."""
    # Exact mixed variance is one of the most memory-hungry prediction paths in
    # the suite. Keep enough data to exercise the nested mixed backend route
    # without making this support test order-dependent on late-suite GPU state.
    n_train = 3000 if method == "materialized" else 6000
    x, y = _generate_mixed_multi_output_data(
        n=n_train, seed=151 if method == "materialized" else 173
    )
    kernel = (Kernel.rbf(active_dims=[0, 1]) + Kernel.matern32(active_dims=[0, 1])) * (
        Kernel.gd(levels=3, active_dims=[2]) + Kernel.cr(levels=3, active_dims=[2])
    )

    gp = MultiOutputGP(
        kernel=kernel,
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp.fit(x, y, max_iterations=2, learning_rate=0.03, verbose=False, method=method)

    mean, var = gp.predict(x[:12], return_var=True, variance_method="exact")
    assert mean.shape == (12, 2)
    assert var.shape == (12, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert gp.backend_predict_info["requested_method"] == method
    assert gp.backend_predict_info["training_route"] == method
    assert (
        gp.backend_predict_info["actual_prediction_route"]
        == "predict_multi_output_mixed"
    )
    assert gp.backend_predict_info["actual_variance_route"] in {
        "predict_multi_output_mixed",
        "predict_multi_output_mixed_exact_retry",
    }
    assert gp.backend_predict_info["backend_prediction_used"] is True
    assert gp.backend_predict_info["backend_variance_used"] is True
    assert gp.backend_predict_info["fallback_used"] is False
    assert gp.backend_predict_info["precond_rank"] == 5
    assert gp.backend_predict_info["precond_method"] == gp.precond_method


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_mixed_multioutput_love_variance_does_not_use_python_fallback(method):
    """Mixed multi-output LOVE variance should stay finite without wrapper fallback."""
    n_train = 5000 if method == "materialized" else 10000
    x, y = _generate_mixed_multi_output_data(
        n=n_train, seed=211 if method == "materialized" else 223
    )
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])

    gp = MultiOutputGP(
        kernel=kernel,
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp.fit(x, y, max_iterations=3, learning_rate=0.03, verbose=False, method=method)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean, var = gp.predict(x[:16], return_var=True, variance_method="love")

    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert not any(
        "Mixed multi-output LOVE variance produced non-finite values" in str(w.message)
        for w in caught
    )


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_mixed_multioutput_reports_backend_rebuild_metadata(method):
    """Mixed MultiOutputGP should surface adaptive rebuild metadata."""
    x, y = _generate_mixed_multi_output_data(
        n=2000, seed=307 if method == "materialized" else 331
    )
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])

    gp = MultiOutputGP(
        kernel=kernel,
        task_rank=1,
        num_probes=1,
        max_cg_iterations=10,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
        precond_rebuild_threshold=0.0,
    )
    gp.fit(x, y, max_iterations=2, learning_rate=0.03, verbose=False, method=method)

    assert gp._raw_result["precond_rebuild_threshold"] == 0.0
    assert gp._raw_result["precond_rebuild_count"] == 1
    assert gp._raw_result["training_route"] == method
