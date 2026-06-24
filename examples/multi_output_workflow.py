"""Public multi-output workflow example covering ICM and LMC wrappers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mojogp import MultiOutputGP, MultiOutputLMCGP, Kernel


def make_data(n_train: int = 2000, n_test: int = 64, seed: int = 42):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, 2).astype(np.float32)
    X_test = rng.randn(n_test, 2).astype(np.float32)

    latent_1 = np.sin(X_train[:, 0]) + 0.2 * X_train[:, 1]
    latent_2 = 0.7 * np.cos(X_train[:, 0]) - 0.1 * X_train[:, 1]
    Y_train = np.stack(
        [
            latent_1 + 0.3 * latent_2 + 0.05 * rng.randn(n_train),
            0.5 * latent_1 - 0.8 * latent_2 + 0.05 * rng.randn(n_train),
        ],
        axis=1,
    ).astype(np.float32)
    return X_train, Y_train, X_test


def run_example(n_train: int = 2000, n_test: int = 64) -> None:
    X_train, Y_train, X_test = make_data(n_train=n_train, n_test=n_test)

    icm = MultiOutputGP(
        kernel="rbf",
        task_rank=1,
        num_probes=4,
        max_cg_iterations=30,
        preconditioner_rank=8,
    )
    icm.fit(X_train, Y_train, max_iterations=8, learning_rate=0.03, verbose=False, method="matrix_free")
    icm_mean, icm_var = icm.predict(
        X_test[:16], return_var=True, variance_method="exact"
    )

    lmc = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=30,
        preconditioner_rank=8,
    )
    lmc.fit(X_train, Y_train, max_iterations=8, learning_rate=0.03, verbose=False, method="matrix_free")
    lmc_mean, lmc_std = lmc.predict(
        X_test[:16], return_std=True, variance_method="love"
    )

    with tempfile.TemporaryDirectory(prefix="mojogp_multi_output_example_") as tmp_dir:
        icm_path = Path(tmp_dir) / "icm_example_model"
        save_path = Path(tmp_dir) / "lmc_example_model"
        icm.save(icm_path)
        lmc.save(save_path)
        icm_loaded = MultiOutputGP.load(icm_path)
        loaded = MultiOutputLMCGP.load(save_path)
        icm_loaded_mean, _ = icm_loaded.predict(
            X_test[:16], return_var=True, variance_method="exact"
        )
        loaded_mean, _ = loaded.predict(
            X_test[:16], return_var=True, variance_method="exact"
        )
        samples = loaded.sample_posterior(
            X_test[:8],
            n_samples=2,
            method="pathwise",
            n_rff_features=512,
            rng=np.random.default_rng(123),
        )

    print("Multi-output workflow example")
    print(f"ICM mean shape:     {icm_mean.shape}")
    print(f"ICM var mean:       {float(np.mean(icm_var)):.6f}")
    print(
        f"ICM loaded delta:   {float(np.max(np.abs(icm_loaded_mean - icm_mean))):.6f}"
    )
    print(f"LMC mean shape:     {lmc_mean.shape}")
    print(f"LMC std mean:       {float(np.mean(lmc_std)):.6f}")
    print(f"Loaded mean delta:  {float(np.max(np.abs(loaded_mean - lmc_mean))):.6f}")
    print(f"Sample shape:       {samples.shape}")
    print(f"ICM train metadata: {icm.backend_train_info}")
    print(f"LMC train metadata: {lmc.backend_train_info}")


if __name__ == "__main__":
    run_example()
