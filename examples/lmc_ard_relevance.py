"""MultiOutputLMCGP ARD metadata workflow with route checks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from mojogp import Kernel, MultiOutputLMCGP


def make_ard_data(n_train: int = 2000, n_test: int = 32, seed: int = 70):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, 3)).astype(np.float32)
    X_test = rng.standard_normal((n_test, 3)).astype(np.float32)
    latent = np.sin(2.0 * X_train[:, 0])
    Y_train = np.zeros((n_train, 2), dtype=np.float32)
    Y_train[:, 0] = latent + 0.05 * rng.standard_normal(n_train)
    Y_train[:, 1] = 1.5 * latent + 0.05 * rng.standard_normal(n_train)
    return X_train, Y_train, X_test


def run_example(method: str = "matrix_free", max_iterations: int = 12) -> None:
    X_train, Y_train, X_test = make_ard_data()
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf()],
        ard=True,
        num_probes=8,
        max_cg_iterations=50,
        cg_tolerance=1e-4,
        preconditioner_rank=10,
    )
    result = gp.fit(
        X_train,
        Y_train,
        method=method,
        max_iterations=max_iterations,
        learning_rate=0.003,
        initial_noise_per_task=np.full(2, 0.05, dtype=np.float32),
        early_stop_tol=0.0,
        verbose=False,
    )

    assert gp.backend_train_info["training_route"] == method
    assert result.lengthscales_per_dim is not None
    avg_lengthscales = result.lengthscales_per_dim.mean(axis=0)
    assert np.all(np.isfinite(avg_lengthscales))
    assert np.all(avg_lengthscales > 0.0)
    relevance_margin = float(np.min(avg_lengthscales[1:]) - avg_lengthscales[0])

    pred = gp.predict(X_test, variance_method="love")
    love_info = dict(gp.backend_predict_info)
    assert love_info["actual_variance_route"] == "predict_lmc"
    assert love_info["fallback_used"] is False

    with tempfile.TemporaryDirectory(prefix="mojogp_lmc_ard_example_") as tmp_dir:
        model_path = Path(tmp_dir) / "lmc_ard"
        gp.save(str(model_path))
        loaded = MultiOutputLMCGP.load(str(model_path))
        loaded_mean, _ = loaded.predict(X_test[:8], return_var=True, variance_method="exact")
        direct_mean, _ = gp.predict(X_test[:8], return_var=True, variance_method="exact")
        np.testing.assert_allclose(loaded_mean, direct_mean, rtol=1e-5, atol=1e-5)

    print("LMC ARD metadata example")
    print(f"Average ARD lengthscales: {avg_lengthscales.tolist()}")
    print(f"Relevant-dimension margin: {relevance_margin:.4f}")
    print(f"LOVE prediction shape:    {pred.mean.shape}")
    print(f"Train route metadata:     {gp.backend_train_info}")
    print(f"LOVE route metadata:      {love_info}")


if __name__ == "__main__":
    run_example()
