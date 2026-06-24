"""Integration tests for claim-level supported feature combinations."""

import numpy as np
import pytest

from mojogp import SingleOutputGP, MultiOutputGP, MultiOutputLMCGP
from mojogp.kernel import Kernel


def _make_exact_mixed_data(n: int = 2000, seed: int = 0):
    rng = np.random.RandomState(seed)
    x_cont = rng.randn(n, 3).astype(np.float32)
    cat = rng.randint(0, 3, size=(n, 1)).astype(np.float32)
    X = np.concatenate([x_cont, cat], axis=1)
    y = (
        np.sin(x_cont[:, 0])
        + 0.2 * x_cont[:, 1]
        - 0.1 * x_cont[:, 2]
        + 0.4 * (cat[:, 0] == 1)
        - 0.3 * (cat[:, 0] == 2)
        + 0.03 * rng.randn(n)
    ).astype(np.float32)
    return X, y


def _make_multi_output_data(n: int = 500, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3).astype(np.float32)
    Y = np.stack(
        [
            np.sin(X[:, 0]) + 0.2 * X[:, 1],
            np.cos(X[:, 0]) - 0.1 * X[:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    Y += 0.03 * rng.randn(n, 2).astype(np.float32)
    return X, Y


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_exactgp_supports_nested_mixed_additive_tree_with_active_dims(method):
    X, y = _make_exact_mixed_data(seed=31 if method == "materialized" else 37)
    kernel = (
        Kernel.rbf(ard=True, active_dims=[0, 1]) + Kernel.matern32(active_dims=[1, 2])
    ) * (Kernel.gd(levels=3, active_dims=[3]) + Kernel.cr(levels=3, active_dims=[3]))

    gp = SingleOutputGP(kernel, verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=3,
        learning_rate=0.03,
        method=method,
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=6,
    )

    pred = gp.predict(X[:16], variance_method="exact")

    assert pred.mean.shape == (16,)
    assert pred.variance.shape == (16,)
    assert np.all(np.isfinite(pred.mean))
    assert np.all(np.isfinite(pred.variance))
    assert gp.backend_predict_info["requested_method"] == method
    assert gp.backend_predict_info["backend_prediction_used"] is True


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_multi_output_gp_supports_composite_kernel_with_save_load(method, tmp_path):
    X, Y = _make_multi_output_data(seed=41 if method == "materialized" else 43)
    kernel = Kernel.rbf() + Kernel.matern52()
    gp = MultiOutputGP(
        kernel=kernel,
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp.fit(X, Y, max_iterations=3, learning_rate=0.03, verbose=False, method=method)

    mean_before, var_before = gp.predict(
        X[:12], return_var=True, variance_method="exact"
    )
    path = tmp_path / f"multi_output_combo_{method}"
    gp.save(path)
    loaded = MultiOutputGP.load(path)
    mean_after, var_after = loaded.predict(
        X[:12], return_var=True, variance_method="exact"
    )

    assert mean_before.shape == mean_after.shape == (12, 2)
    assert var_before.shape == var_after.shape == (12, 2)
    assert np.all(np.isfinite(mean_after))
    assert np.all(np.isfinite(var_after))
    assert loaded._training_method == method


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_supports_heterogeneous_kernels_and_pathwise_sampling(method):
    X, Y = _make_multi_output_data(seed=53 if method == "materialized" else 59)
    gp = MultiOutputLMCGP(
        kernels=["rbf", "matern52"],
        num_probes=2,
        max_cg_iterations=20,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
    )
    gp.fit(X, Y, max_iterations=3, learning_rate=0.03, verbose=False, method=method)

    samples = gp.sample_posterior(
        X[:10], n_samples=3, method="pathwise", n_rff_features=256
    )

    assert samples.shape == (3, 10, 2)
    assert np.all(np.isfinite(samples))
    assert gp.backend_sample_info["actual_sampling_method"] == "pathwise"
    assert gp.backend_sample_info["backend_correction_used"] is True
