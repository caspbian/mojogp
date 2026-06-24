"""Multi-output GPs with fixed and grouped observation noise."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mojogp import Kernel, MultiOutputGP


def make_data(n_train: int = 5000, n_test: int = 256, seed: int = 31):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(-3.0, 3.0, size=(n_train, 2)).astype(np.float32)
    X_test = rng.uniform(-2.8, 2.8, size=(n_test, 2)).astype(np.float32)

    latent_train = np.stack(
        [
            np.sin(X_train[:, 0]) + 0.2 * X_train[:, 1],
            0.6 * np.sin(X_train[:, 0]) - 0.15 * X_train[:, 1],
        ],
        axis=1,
    ).astype(np.float32)
    latent_test = np.stack(
        [
            np.sin(X_test[:, 0]) + 0.2 * X_test[:, 1],
            0.6 * np.sin(X_test[:, 0]) - 0.15 * X_test[:, 1],
        ],
        axis=1,
    ).astype(np.float32)

    train_noise = np.empty_like(latent_train)
    train_noise[:, 0] = 0.015 + 0.035 * (X_train[:, 0] > 0.0)
    train_noise[:, 1] = 0.03 + 0.045 * (X_train[:, 1] > 0.0)
    Y_train = latent_train + rng.normal(scale=np.sqrt(train_noise)).astype(np.float32)

    test_noise = np.empty_like(latent_test)
    test_noise[:, 0] = 0.015 + 0.035 * (X_test[:, 0] > 0.0)
    test_noise[:, 1] = 0.03 + 0.045 * (X_test[:, 1] > 0.0)

    train_groups = (X_train[:, 0] > 0.0).astype(np.int32)
    test_groups = (X_test[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([[0.015, 0.03], [0.05, 0.075]], dtype=np.float32)

    return (
        X_train,
        Y_train.astype(np.float32),
        train_noise.astype(np.float32),
        train_groups,
        test_noise.astype(np.float32),
        test_groups,
        group_noise,
        X_test,
        latent_test,
    )


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def run_example(method: str = "matrix_free", n_train: int = 5000, n_test: int = 256):
    (
        X_train,
        Y_train,
        train_noise,
        train_groups,
        test_noise,
        test_groups,
        group_noise,
        X_test,
        latent_test,
    ) = make_data(n_train=n_train, n_test=n_test)

    icm = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=4,
        max_cg_iterations=50,
        preconditioner_rank=10,
    )
    icm.fit(
        X_train,
        Y_train,
        observation_noise=train_noise,
        method=method,
        max_iterations=8,
        learning_rate=0.02,
        verbose=False,
    )
    icm_pred = icm.predict(X_test[:64])
    icm_observed = icm.predict_observed(
        X_test[:64], observation_noise=test_noise[:64], variance_method="exact"
    )

    grouped = MultiOutputGP(
        kernel=Kernel.rbf(),
        task_rank=1,
        num_probes=4,
        max_cg_iterations=50,
        preconditioner_rank=10,
    )
    grouped.fit(
        X_train,
        Y_train,
        noise_model="grouped",
        noise_group_train=train_groups,
        group_noise=group_noise,
        method=method,
        max_iterations=8,
        learning_rate=0.02,
        verbose=False,
    )
    grouped_pred = grouped.predict(X_test[:64])
    grouped_observed = grouped.predict_observed(
        X_test[:64], noise_group_test=test_groups[:64], variance_method="exact"
    )

    with tempfile.TemporaryDirectory(prefix="mojogp_multi_noise_example_") as tmp_dir:
        icm_path = Path(tmp_dir) / "icm_fixed_noise"
        grouped_path = Path(tmp_dir) / "icm_grouped_noise"
        icm.save(icm_path)
        grouped.save(grouped_path)
        icm_loaded = MultiOutputGP.load(icm_path)
        grouped_loaded = MultiOutputGP.load(grouped_path)
        assert icm_loaded._observation_noise_train is not None
        assert grouped_loaded._noise_group_values is not None

    print("Multi-output observation noise example")
    print(f"Method: {method}")
    print(f"ICM latent RMSE: { _rmse(icm_pred.mean, latent_test[:64]):.4f}")
    print(f"Grouped ICM latent RMSE: { _rmse(grouped_pred.mean, latent_test[:64]):.4f}")
    print(f"ICM observed std mean: {float(np.mean(icm_observed.std)):.4f}")
    print(f"Grouped observed std mean: {float(np.mean(grouped_observed.std)):.4f}")
    print(f"ICM train metadata: {icm.backend_train_info}")
    print(f"Grouped train metadata: {grouped.backend_train_info}")


if __name__ == "__main__":
    run_example("materialized")
    run_example("matrix_free")
