"""Dense exact reference checks for MultiOutputLMCGP.

These tests provide a small-n numerical proof surface for the novel LMC path by
building the full dense posterior on the CPU from the learned latent kernels and
coregionalization matrices, then comparing it against the wrapper predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from mojogp import Kernel, MultiOutputGP, MultiOutputLMCGP

from tests.shared.dense_references import (
    build_lmc_cross_covariance,
    build_lmc_test_covariance,
    build_lmc_train_covariance,
    diagonal_task_variances,
    exact_gaussian_posterior,
    flatten_multi_output_targets,
    unflatten_multi_output_predictions,
)
from tests.shared.gpu_test_utils import assert_gpu_available, requires_cuda


pytestmark = [pytest.mark.reference]


def _generate_lmc_data(n_train: int = 24, n_test: int = 6, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, 2)).astype(np.float32)
    X_test = rng.standard_normal((n_test, 2)).astype(np.float32)
    f1 = np.sin(X_train[:, 0]) + 0.2 * X_train[:, 1]
    f2 = 0.7 * np.cos(X_train[:, 0]) - 0.3 * X_train[:, 1]
    Y_train = np.stack(
        [
            f1 + 0.05 * rng.standard_normal(n_train),
            f2 + 0.05 * rng.standard_normal(n_train),
        ],
        axis=1,
    ).astype(np.float32)
    return X_train, Y_train, X_test


def _dense_reference_for_lmc(gp: MultiOutputLMCGP, X_test: np.ndarray):
    result = gp.training_result
    assert result is not None
    assert result.A_matrices is not None
    assert result.params_per_latent is not None

    train_kernel_matrices = []
    cross_kernel_matrices = []
    test_kernel_matrices = []
    cat_params_per_latent = result.cat_params_per_latent or [None] * gp.num_latents

    for latent_idx in range(gp.num_latents):
        params = np.asarray(result.params_per_latent[latent_idx], dtype=np.float32)
        cat_params = cat_params_per_latent[latent_idx]
        train_kernel_matrices.append(
            gp._evaluate_latent_kernel_matrix(
                latent_idx, gp._X_train, gp._X_train, params, cat_params
            )
        )
        cross_kernel_matrices.append(
            gp._evaluate_latent_kernel_matrix(
                latent_idx, gp._X_train, X_test, params, cat_params
            )
        )
        test_kernel_matrices.append(
            gp._evaluate_latent_kernel_matrix(
                latent_idx, X_test, X_test, params, cat_params
            )
        )

    train_cov = build_lmc_train_covariance(
        gp._X_train,
        train_kernel_matrices,
        np.asarray(result.A_matrices, dtype=np.float64),
        np.asarray(result.noise_per_task, dtype=np.float64),
        result.fixed_observation_noise,
    )
    cross_cov = build_lmc_cross_covariance(
        cross_kernel_matrices,
        np.asarray(result.A_matrices, dtype=np.float64),
    )
    test_cov = build_lmc_test_covariance(
        test_kernel_matrices,
        np.asarray(result.A_matrices, dtype=np.float64),
    )
    y_centered = gp._Y_train.astype(np.float64)
    if result.mean_per_task is not None:
        y_centered = (
            y_centered
            - np.asarray(result.mean_per_task, dtype=np.float64)[np.newaxis, :]
        )

    mean_vec, cov = exact_gaussian_posterior(
        train_cov=train_cov,
        cross_cov=cross_cov,
        test_cov=test_cov,
        y_train=flatten_multi_output_targets(y_centered),
    )
    mean = unflatten_multi_output_predictions(mean_vec, gp.num_tasks)
    if result.mean_per_task is not None:
        mean = mean + np.asarray(result.mean_per_task, dtype=np.float64)[np.newaxis, :]
    variance = diagonal_task_variances(cov, gp.num_tasks)
    variance = variance + np.asarray(result.noise_per_task, dtype=np.float64)[
        np.newaxis, :
    ]
    return mean.astype(np.float32), np.maximum(variance.astype(np.float32), 0.0)


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_lmc_dense_reference_matches_exact_prediction():
    assert_gpu_available()
    X_train, Y_train, X_test = _generate_lmc_data()
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=120,
        cg_tolerance=1e-5,
        preconditioner_rank=8,
    )
    gp.fit(X_train, Y_train, max_iterations=25, learning_rate=0.03, verbose=False, method="materialized")

    pred_mean, pred_var = gp.predict(X_test, return_var=True, variance_method="exact")
    dense_mean, dense_var = _dense_reference_for_lmc(gp, X_test)

    assert gp.backend_predict_info["actual_variance_route"] == "dense_exact_lmc"
    assert gp.backend_predict_info["predictive_variance_kind"] == "observation"
    assert gp.backend_predict_info["variance_includes_observation_noise"] is True
    assert gp.backend_predict_info["lmc_exact_variance_source"] == "dense_exact_lmc"
    np.testing.assert_allclose(pred_mean, dense_mean, atol=5e-2, rtol=8e-2)
    np.testing.assert_allclose(pred_var, dense_var, atol=8e-2, rtol=1.5e-1)


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_lmc_matrix_free_exact_variance_matches_dense_reference():
    assert_gpu_available()
    X_train, Y_train, X_test = _generate_lmc_data(seed=43)
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=160,
        cg_tolerance=1e-5,
        preconditioner_rank=8,
    )
    gp.fit(
        X_train,
        Y_train,
        max_iterations=25,
        learning_rate=0.03,
        verbose=False,
        method="matrix_free",
    )

    pred_mean, pred_var = gp.predict(X_test, return_var=True, variance_method="exact")
    dense_mean, dense_var = _dense_reference_for_lmc(gp, X_test)

    assert gp.backend_predict_info["actual_variance_route"] == "predict_lmc_full_exact"
    assert gp.backend_predict_info["predictive_variance_kind"] == "observation"
    assert gp.backend_predict_info["variance_includes_observation_noise"] is True
    assert gp.backend_predict_info["lmc_exact_variance_source"] == "predict_lmc_full_exact"
    np.testing.assert_allclose(pred_mean, dense_mean, atol=5e-2, rtol=8e-2)
    np.testing.assert_allclose(pred_var, dense_var, atol=8e-2, rtol=1.5e-1)


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_lmc_save_load_preserves_dense_exact_observation_variance(tmp_path):
    assert_gpu_available()
    X_train, Y_train, X_test = _generate_lmc_data(seed=17)
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=120,
        cg_tolerance=1e-5,
        preconditioner_rank=8,
    )
    gp.fit(
        X_train,
        Y_train,
        max_iterations=25,
        learning_rate=0.03,
        verbose=False,
        method="materialized",
    )

    mean_before, var_before = gp.predict(
        X_test, return_var=True, variance_method="exact"
    )
    save_path = tmp_path / "lmc_dense_exact_roundtrip"
    gp.save(str(save_path))

    loaded = MultiOutputLMCGP.load(str(save_path))
    mean_after, var_after = loaded.predict(
        X_test, return_var=True, variance_method="exact"
    )

    np.testing.assert_allclose(mean_after, mean_before, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(var_after, var_before, atol=1e-5, rtol=1e-5)
    assert loaded.backend_predict_info["actual_variance_route"] == "dense_exact_lmc"
    assert loaded.backend_predict_info["predictive_variance_kind"] == "observation"


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_lmc_fixed_observation_noise_matches_dense_reference_and_persists(tmp_path):
    assert_gpu_available()
    X_train, Y_train, X_test = _generate_lmc_data(seed=29)
    rng = np.random.default_rng(29)
    fixed_noise = (0.01 + 0.02 * rng.random(Y_train.shape)).astype(np.float32)
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=120,
        cg_tolerance=1e-5,
        preconditioner_rank=8,
    )
    gp.fit(
        X_train,
        Y_train,
        max_iterations=25,
        learning_rate=0.03,
        verbose=False,
        method="materialized",
        fixed_observation_noise=fixed_noise,
    )

    pred_mean, pred_var = gp.predict(X_test, return_var=True, variance_method="exact")
    dense_mean, dense_var = _dense_reference_for_lmc(gp, X_test)

    assert gp.training_result.fixed_observation_noise is not None
    assert gp.backend_train_info["uses_fixed_observation_noise"] is True
    assert gp.backend_predict_info["uses_fixed_observation_noise"] is True
    np.testing.assert_allclose(pred_mean, dense_mean, atol=5e-2, rtol=8e-2)
    np.testing.assert_allclose(pred_var, dense_var, atol=8e-2, rtol=1.5e-1)

    save_path = tmp_path / "lmc_fixed_noise_roundtrip"
    gp.save(str(save_path))
    loaded = MultiOutputLMCGP.load(str(save_path))
    np.testing.assert_allclose(loaded.training_result.fixed_observation_noise, fixed_noise)
    loaded_mean, loaded_var = loaded.predict(
        X_test, return_var=True, variance_method="exact"
    )
    np.testing.assert_allclose(loaded_mean, pred_mean, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(loaded_var, pred_var, atol=1e-5, rtol=1e-5)
    assert loaded.backend_predict_info["actual_variance_route"] == "dense_exact_lmc"
    assert loaded.backend_predict_info["uses_fixed_observation_noise"] is True


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_lmc_love_variance_is_calibrated_against_exact_route():
    assert_gpu_available()
    X_train, Y_train, X_test = _generate_lmc_data(seed=37)
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=6,
        max_cg_iterations=120,
        cg_tolerance=1e-5,
        preconditioner_rank=8,
    )
    gp.fit(
        X_train,
        Y_train,
        max_iterations=25,
        learning_rate=0.03,
        verbose=False,
        method="materialized",
    )

    _, exact_var = gp.predict(X_test, return_var=True, variance_method="exact")
    exact_info = dict(gp.backend_predict_info)
    _, love_var = gp.predict(X_test, return_var=True, variance_method="love")
    love_info = dict(gp.backend_predict_info)

    assert exact_info["actual_variance_route"] == "dense_exact_lmc"
    assert love_info["variance_method"] == "love"
    assert love_info["backend_variance_used"] is True
    assert love_info["actual_variance_route"] == "predict_lmc"
    assert np.all(np.isfinite(love_var))
    assert np.all(love_var >= 0.0)
    rel_error = np.abs(love_var - exact_var) / np.maximum(exact_var, 1e-6)
    assert float(np.median(rel_error)) < 0.75


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_lmc_r1_matches_icm_predictions_on_same_kernel():
    assert_gpu_available()
    X_train, Y_train, X_test = _generate_lmc_data(seed=7)

    icm = MultiOutputGP(
        kernel="rbf",
        task_rank=1,
        num_probes=4,
        max_cg_iterations=40,
        preconditioner_rank=8,
    )
    lmc = MultiOutputLMCGP(
        kernels=[Kernel.rbf()],
        num_probes=4,
        max_cg_iterations=40,
        preconditioner_rank=8,
    )

    icm.fit(X_train, Y_train, max_iterations=25, learning_rate=0.03, verbose=False, method="materialized")
    lmc.fit(X_train, Y_train, max_iterations=25, learning_rate=0.03, verbose=False, method="materialized")

    icm_mean, icm_var = icm.predict(X_test, return_var=True, variance_method="exact")
    lmc_mean, lmc_var = lmc.predict(X_test, return_var=True, variance_method="exact")

    corr = np.corrcoef(icm_mean.reshape(-1), lmc_mean.reshape(-1))[0, 1]
    assert corr > 0.75, f"LMC R=1 and ICM predictions diverged too far: corr={corr:.3f}"
    np.testing.assert_allclose(icm_var, lmc_var, atol=0.35, rtol=0.50)
