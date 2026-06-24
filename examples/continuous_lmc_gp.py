"""Continuous MultiOutputLMCGP workflow with route metadata checks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mojogp import Kernel, MultiOutputLMCGP


def make_lmc_data(n_train: int = 2000, n_test: int = 64, seed: int = 7):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, 3)).astype(np.float32)
    X_test = rng.standard_normal((n_test, 3)).astype(np.float32)

    latent_smooth = np.sin(X_train[:, 0]) + 0.25 * X_train[:, 1]
    latent_rough = np.cos(1.5 * X_train[:, 0]) - 0.2 * X_train[:, 2]
    Y_train = np.column_stack(
        [
            latent_smooth + 0.4 * latent_rough,
            0.6 * latent_smooth - 0.7 * latent_rough,
        ]
    ).astype(np.float32)
    fixed_noise = (0.01 + 0.01 * rng.random(Y_train.shape)).astype(np.float32)
    Y_train = Y_train + rng.standard_normal(Y_train.shape).astype(np.float32) * np.sqrt(
        fixed_noise
    )
    return X_train, Y_train, X_test, fixed_noise


def run_example() -> None:
    X_train, Y_train, X_test, fixed_noise = make_lmc_data()
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=6,
        max_cg_iterations=80,
        preconditioner_rank=8,
    )
    gp.fit(
        X_train,
        Y_train,
        method="matrix_free",
        max_iterations=20,
        learning_rate=0.03,
        fixed_observation_noise=fixed_noise,
        verbose=False,
    )

    mean_only_pred = gp.predict(X_test[:16], variance_method="mean_only")
    mean_only_info = dict(gp.backend_predict_info)
    exact_mean, exact_var = gp.predict(X_test[:16], return_var=True, variance_method="exact")
    exact_info = dict(gp.backend_predict_info)

    assert gp.backend_train_info["training_route"] == "matrix_free"
    assert mean_only_info.get("actual_variance_route") is None
    assert exact_info["actual_variance_route"] == "predict_lmc_full_exact"
    assert mean_only_info["fallback_used"] is False
    assert exact_info["fallback_used"] is False

    with tempfile.TemporaryDirectory(prefix="mojogp_lmc_example_") as tmp_dir:
        model_path = Path(tmp_dir) / "continuous_lmc"
        gp.save(str(model_path))
        loaded = MultiOutputLMCGP.load(str(model_path))
        loaded_mean, _ = loaded.predict(
            X_test[:16], return_var=True, variance_method="exact"
        )
        assert loaded.backend_predict_info["actual_variance_route"] == "predict_lmc_full_exact"

    print("Continuous LMC example")
    print(f"Mean-only shape:       {mean_only_pred.mean.shape}")
    print(f"Exact variance mean:   {float(np.mean(exact_var)):.6f}")
    print(f"Loaded mean delta:     {float(np.max(np.abs(loaded_mean - exact_mean))):.6f}")
    print(f"Train route metadata:  {gp.backend_train_info}")
    print(f"Mean route metadata:   {mean_only_info}")
    print(f"Exact route metadata:  {exact_info}")


if __name__ == "__main__":
    run_example()
