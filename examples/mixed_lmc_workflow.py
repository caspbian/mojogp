"""Mixed continuous-categorical MultiOutputLMCGP workflow.

This example stays on the currently evidenced mixed LMC surface: each mixed
latent contains at least one continuous kernel component, pure categorical LMC
latents are not used, and route metadata is checked after prediction/sampling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mojogp import Kernel, MultiOutputLMCGP


def make_mixed_lmc_data(n_train: int = 2000, n_test: int = 32, seed: int = 23):
    rng = np.random.default_rng(seed)
    X_cont = rng.standard_normal((n_train, 2)).astype(np.float32)
    C = rng.integers(0, 3, size=(n_train, 1), dtype=np.int32)
    X_train = np.concatenate([X_cont, C.astype(np.float32)], axis=1)

    cat_effect = np.array([-0.4, 0.2, 0.5], dtype=np.float32)[C[:, 0]]
    Y_train = np.column_stack(
        [
            np.sin(X_cont[:, 0]) + 0.25 * X_cont[:, 1] + 0.7 * cat_effect,
            0.6 * np.cos(X_cont[:, 0]) - 0.2 * X_cont[:, 1] + 0.4 * cat_effect,
        ]
    ).astype(np.float32)
    Y_train += 0.04 * rng.standard_normal(Y_train.shape).astype(np.float32)

    X_test_cont = rng.standard_normal((n_test, 2)).astype(np.float32)
    C_test = rng.integers(0, 3, size=(n_test, 1), dtype=np.int32)
    X_test = np.concatenate([X_test_cont, C_test.astype(np.float32)], axis=1)
    return X_train.astype(np.float32), Y_train, X_test.astype(np.float32)


def run_example(n_train: int = 2000, n_test: int = 32) -> None:
    X_train, Y_train, X_test = make_mixed_lmc_data(
        n_train=n_train, n_test=n_test
    )
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
    gp.fit(
        X_train,
        Y_train,
        max_iterations=4,
        learning_rate=0.03,
        method="matrix_free",
        verbose=False,
    )

    exact = gp.predict(X_test[:8], variance_method="exact")
    exact_info = dict(gp.backend_predict_info)
    love = gp.predict(X_test[:8], variance_method="love")
    love_info = dict(gp.backend_predict_info)
    samples = gp.sample_posterior(
        X_test[:4],
        n_samples=2,
        method="pathwise",
        n_rff_features=256,
        rng=np.random.default_rng(123),
    )
    sample_info = dict(gp.backend_sample_info)

    assert gp.backend_train_info["training_route"] == "matrix_free"
    assert exact_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"
    assert love_info["actual_variance_route"] == "predict_lmc_mixed"
    assert love_info["fallback_used"] is False
    assert sample_info["backend_correction_route"] == "sample_lmc_mixed_pathwise"

    with tempfile.TemporaryDirectory(prefix="mojogp_mixed_lmc_example_") as tmp_dir:
        model_path = Path(tmp_dir) / "mixed_lmc"
        gp.save(str(model_path))
        loaded = MultiOutputLMCGP.load(str(model_path))
        loaded_exact = loaded.predict(X_test[:8], variance_method="exact")
        assert loaded.backend_predict_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"

    print("Mixed LMC example")
    print(f"Exact mean shape:       {exact.mean.shape}")
    print(f"LOVE variance mean:    {float(np.mean(love.variance)):.6f}")
    print(f"Loaded mean delta:     {float(np.max(np.abs(loaded_exact.mean - exact.mean))):.6f}")
    print(f"Pathwise sample shape: {samples.shape}")
    print(f"Train route metadata:  {gp.backend_train_info}")
    print(f"LOVE route metadata:   {love_info}")
    print(f"Sample metadata:       {sample_info}")


if __name__ == "__main__":
    run_example()
