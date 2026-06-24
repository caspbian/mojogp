"""Unit tests for persisted save/load schema completeness."""

import json

import numpy as np
import pytest

from mojogp import SingleOutputGP, MultiOutputGP, MultiOutputLMCGP, LMCTrainingResult
from mojogp.kernel import Kernel


def _make_exact_mixed_data(n: int = 2000, seed: int = 0):
    rng = np.random.RandomState(seed)
    x_cont = rng.randn(n, 2).astype(np.float32)
    cat = rng.randint(0, 3, size=(n, 1)).astype(np.float32)
    X = np.concatenate([x_cont, cat], axis=1)
    y = (
        np.sin(x_cont[:, 0])
        + 0.3 * x_cont[:, 1]
        + 0.5 * (cat[:, 0] == 1)
        - 0.4 * (cat[:, 0] == 2)
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


def _make_trained_lmc_for_schema():
    """Create a minimal current LMC state for save/load schema checks."""
    X, Y = _make_multi_output_data()
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        use_preconditioner=False,
    )
    R = 2
    T = Y.shape[1]
    n = X.shape[0]
    params_per_latent = [
        np.asarray(kernel.get_initial_params(), dtype=np.float32)
        for kernel in gp.kernels
    ]
    A_matrices = np.stack(
        [np.eye(T, dtype=np.float32) * 0.5, np.eye(T, dtype=np.float32) * 0.25]
    )
    B = np.sum(A_matrices, axis=0).astype(np.float32)
    eigvals, eigvecs = np.linalg.eigh(B.astype(np.float64))
    alpha = np.zeros((n, T), dtype=np.float32)
    mean_per_task = np.mean(Y, axis=0).astype(np.float32)
    gp._result = LMCTrainingResult(
        final_nll=1.0,
        nll_history=np.array([1.0], dtype=np.float32),
        iterations=1,
        converged=True,
        num_latents=R,
        num_tasks=T,
        noise_per_task=np.full(T, 0.1, dtype=np.float32),
        lengthscales=np.ones(R, dtype=np.float32),
        outputscales=np.ones(R, dtype=np.float32),
        params_per_latent=params_per_latent,
        kernel_types=np.zeros(R, dtype=np.int32),
        A_matrices=A_matrices,
        L_factors=np.zeros((R, T, T), dtype=np.float32),
        B=B,
        Q=eigvecs.astype(np.float32),
        Lambda=eigvals.astype(np.float32),
        alpha=alpha,
        alpha_rotated=(alpha @ eigvecs).astype(np.float32),
        effective_scales=eigvals.astype(np.float32),
        mean_per_task=mean_per_task,
    )
    gp._X_train = X
    gp._Y_train = Y
    gp._is_trained = True
    gp._fixed_observation_noise = None
    gp._fitted_mean = mean_per_task
    return gp


def test_single_output_mixed_save_schema_includes_kernel_and_categorical_state(tmp_path):
    X, y = _make_exact_mixed_data()
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])
    gp = SingleOutputGP(kernel)
    gp.fit(X, y, max_iterations=2, learning_rate=0.03, method="materialized")

    path = tmp_path / "exact_mixed"
    gp.save(path)

    with open(f"{path}_config.json", "r") as f:
        config = json.load(f)
    arrays = np.load(f"{path}_arrays.npz", allow_pickle=False)

    assert config["wrapper"] == "SingleOutputGP"
    assert config["schema_version"] == 1
    assert "mojogp_version" in config
    assert config["result_type"] == "MixedTrainingResult"
    assert config["is_mixed"] is True
    assert config["kernel_tree"] is not None
    assert config["training_method"] == "materialized"
    assert "X_train" in arrays
    assert "y_train" in arrays
    assert "params" in arrays
    assert "cat_params" in arrays
    assert "alpha" in arrays
    assert "C_train" in arrays


def test_multi_output_gp_save_schema_includes_mixed_arrays_and_metadata(tmp_path):
    X, Y = _make_exact_mixed_data(n=500, seed=11)
    Y = np.stack([Y, 0.7 * Y + 0.1 * X[:, 0]], axis=1).astype(np.float32)
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])
    gp = MultiOutputGP(
        kernel=kernel,
        task_rank=1,
        num_probes=2,
        use_preconditioner=False,
    )
    gp.fit(X, Y, max_iterations=2, learning_rate=0.03, method="matrix_free", verbose=False)

    path = tmp_path / "multi_output_mixed"
    gp.save(path)

    with open(f"{path}_config.json", "r") as f:
        config = json.load(f)
    arrays = np.load(f"{path}_arrays.npz", allow_pickle=False)

    assert config["wrapper"] == "MultiOutputGP"
    assert config["schema_version"] == 1
    assert "mojogp_version" in config
    assert config["result_type"] == "MultiOutputTrainingResult"
    assert config["training_method"] == "matrix_free"
    assert config["use_preconditioner"] is False
    assert config["precond_rank"] == 0
    assert config["kernel_tree"] is not None
    assert "X_train" in arrays
    assert "Y_train" in arrays
    assert "params" in arrays
    assert "cat_params" in arrays
    assert "X_train_cont" in arrays
    assert "C_train" in arrays
    assert "mean_per_task" in arrays


def test_multi_output_lmc_save_schema_includes_per_latent_arrays(tmp_path):
    gp = _make_trained_lmc_for_schema()

    path = tmp_path / "lmc"
    gp.save(path)

    with open(f"{path}_config.json", "r") as f:
        config = json.load(f)
    arrays = np.load(f"{path}_arrays.npz", allow_pickle=False)

    assert config["wrapper"] == "MultiOutputLMCGP"
    assert config["schema_version"] == 1
    assert "mojogp_version" in config
    assert config["num_latents"] == 2
    assert config["use_preconditioner"] is False
    assert config["precond_rank"] == 0
    assert "X_train" in arrays
    assert "Y_train" in arrays
    assert "A_matrices" in arrays
    assert "alpha" in arrays
    assert "mean_per_task" in arrays
    assert "params_latent_0" in arrays
    assert "params_latent_1" in arrays

    loaded = MultiOutputLMCGP.load(path)
    assert loaded.use_preconditioner is False
    assert loaded.precond_rank == 0
    assert loaded._specialization_request.mode == "disabled"
    assert loaded._specialization_decision is None


def test_multi_output_lmc_load_rejects_unsupported_schema_version(tmp_path):
    gp = _make_trained_lmc_for_schema()

    path = tmp_path / "lmc_bad_schema"
    gp.save(path)
    config_path = f"{path}_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config["schema_version"] = 999
    with open(config_path, "w") as f:
        json.dump(config, f)

    with pytest.raises(ValueError, match="Unsupported MultiOutputLMCGP schema_version=999"):
        MultiOutputLMCGP.load(path)


def _write_minimal_lmc_artifact(path, *, schema_version=1, omit_arrays=()):
    config = {
        "schema_version": schema_version,
        "mojogp_version": "0.1.0",
        "wrapper": "MultiOutputLMCGP",
        "is_composite": True,
        "has_mixed_latents": False,
        "ard": False,
        "method": "materialized",
        "num_probes": 2,
        "max_cg_iter": 10,
        "cg_tol": 1.0,
        "use_preconditioner": False,
        "precond_rank": 0,
        "precond": "greedy",
        "precond_rebuild_threshold": 0.5,
        "num_latents": 1,
        "num_tasks": 2,
        "final_nll": 1.0,
        "iterations": 1,
        "converged": False,
        "use_ard": False,
        "kernel_trees": [Kernel.rbf().to_dict()],
        "kernel_tree": Kernel.rbf().to_dict(),
    }
    arrays = {
        "X_train": np.zeros((4, 2), dtype=np.float32),
        "Y_train": np.zeros((4, 2), dtype=np.float32),
        "lengthscales": np.ones(1, dtype=np.float32),
        "outputscales": np.ones(1, dtype=np.float32),
        "kernel_types": np.asarray([], dtype=np.int32),
        "noise_per_task": np.ones(2, dtype=np.float32) * 0.1,
        "A_matrices": np.eye(2, dtype=np.float32)[None, :, :],
        "B": np.zeros((1, 2, 2), dtype=np.float32),
        "Q": np.zeros((2, 2), dtype=np.float32),
        "Lambda": np.ones(2, dtype=np.float32),
        "alpha": np.zeros(8, dtype=np.float32),
        "nll_history": np.ones(1, dtype=np.float32),
    }
    for key in omit_arrays:
        arrays.pop(key)
    with open(f"{path}_config.json", "w") as f:
        json.dump(config, f)
    np.savez(f"{path}_arrays.npz", **arrays)


def test_multi_output_lmc_load_rejects_unknown_schema_before_native_load(tmp_path):
    path = tmp_path / "bad_lmc_schema"
    _write_minimal_lmc_artifact(path, schema_version=999)

    with pytest.raises(ValueError, match="Unsupported MultiOutputLMCGP schema_version=999"):
        MultiOutputLMCGP.load(path)


def test_multi_output_lmc_load_rejects_missing_required_arrays(tmp_path):
    gp = _make_trained_lmc_for_schema()

    path = tmp_path / "lmc_missing_state"
    gp.save(path)
    arrays_path = f"{path}_arrays.npz"
    arrays = np.load(arrays_path, allow_pickle=False)
    rewritten = {key: arrays[key] for key in arrays.files if key != "A_matrices"}
    arrays.close()
    np.savez(arrays_path, **rewritten)

    with pytest.raises(ValueError, match=r"missing required array\(s\) A_matrices"):
        MultiOutputLMCGP.load(path)


def test_multi_output_lmc_load_rejects_missing_late_required_arrays(tmp_path):
    path = tmp_path / "bad_lmc_arrays"
    _write_minimal_lmc_artifact(path, omit_arrays=("alpha",))

    with pytest.raises(ValueError, match="missing required arrays: alpha"):
        MultiOutputLMCGP.load(path)
