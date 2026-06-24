"""Child entrypoints for posterior sampling and heterogeneous LMC tests."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from mojogp import Kernel, MultiOutputLMCGP
from tests.shared.subprocess_harness import run_child_main


def _cleanup_gpu_state() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_mo_data(n: int = 200, d: int = 3, T: int = 2, seed: int = 42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.column_stack(
        [
            np.sin(X[:, 0]) + 0.1 * np.random.randn(n),
            np.cos(X[:, 0]) + 0.1 * np.random.randn(n),
        ]
    ).astype(np.float32)
    return X, Y


def generate_mixed_mo_data(
    n: int = 2000,
    d_cont: int = 2,
    T: int = 2,
    levels: int = 3,
    seed: int = 123,
    noise_std: float = 0.05,
):
    rng = np.random.default_rng(seed)
    X_cont = rng.standard_normal((n, d_cont)).astype(np.float32)
    cat = rng.integers(0, levels, size=(n, 1), dtype=np.int32)
    X = np.concatenate([X_cont, cat.astype(np.float32)], axis=1)

    cat_effect = np.linspace(-0.6, 0.6, levels, dtype=np.float32)[cat[:, 0]]
    Y = np.zeros((n, T), dtype=np.float32)
    Y[:, 0] = (
        np.sin(X_cont[:, 0])
        + 0.35 * X_cont[:, 1]
        + 0.8 * cat_effect
        + noise_std * rng.standard_normal(n).astype(np.float32)
    )
    if T > 1:
        Y[:, 1] = (
            0.7 * np.cos(X_cont[:, 0])
            - 0.2 * X_cont[:, 1]
            + 0.5 * cat_effect
            + noise_std * rng.standard_normal(n).astype(np.float32)
        )
    for t in range(2, T):
        Y[:, t] = (
            0.4 * np.sin(X_cont[:, 0] + t)
            + (0.3 + 0.1 * t) * cat_effect
            + noise_std * rng.standard_normal(n).astype(np.float32)
        )
    return X.astype(np.float32), Y.astype(np.float32)


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    case = str(payload["case"])
    method_raw = payload.get("method")
    save_dir_raw = payload.get("save_dir")
    method = None if method_raw in (None, "__none__") else str(method_raw)
    save_dir = None if save_dir_raw in (None, "__none__") else Path(str(save_dir_raw))

    _cleanup_gpu_state()

    if case == "lmc_save_load":
        X, Y = generate_mo_data(n=2000, d=2, T=2, seed=91)
        X_test = np.random.randn(6, 2).astype(np.float32)

        gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()])
        gp.fit(X, Y, max_iterations=8, verbose=False, method=method)

        samples = gp.sample_posterior(
            X_test,
            n_samples=3,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(123),
        )
        info = dict(gp.backend_sample_info)
        assert samples.shape == (3, 6, 2)
        assert info["actual_sampling_route"] == "provider_pathwise"
        assert info["training_route"] == method

        save_path = save_dir / f"pathwise_lmc_{method}"
        gp.save(str(save_path))
        loaded = MultiOutputLMCGP.load(str(save_path))
        loaded_samples = loaded.sample_posterior(
            X_test,
            n_samples=3,
            method="pathwise",
            n_rff_features=1024,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_allclose(samples, loaded_samples, atol=1e-6)
        loaded_info = dict(loaded.backend_sample_info)
        assert loaded_info["actual_sampling_route"] == "provider_pathwise"
        assert loaded_info["training_route"] == method
        result = loaded_info
    elif case == "mixed_save_load":
        X, Y = generate_mixed_mo_data(n=2000, d_cont=2, T=2, levels=3, seed=131)
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
        gp.fit(X, Y, max_iterations=4, learning_rate=0.03, verbose=False, method=method)

        X_test = X[:6].copy()
        samples = gp.sample_posterior(
            X_test,
            n_samples=4,
            method="pathwise",
            n_rff_features=256,
            rng=np.random.default_rng(123),
        )
        info = dict(gp.backend_sample_info)
        assert samples.shape == (4, 6, 2)
        assert info["backend_correction_route"] == "sample_lmc_mixed_pathwise"
        assert info["training_route"] == method

        save_path = save_dir / f"mixed_pathwise_lmc_{method}"
        gp.save(str(save_path))
        loaded = MultiOutputLMCGP.load(str(save_path))
        loaded_samples = loaded.sample_posterior(
            X_test,
            n_samples=4,
            method="pathwise",
            n_rff_features=256,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_allclose(
            loaded_samples.mean(axis=0), samples.mean(axis=0), atol=0.2, rtol=0.3
        )
        np.testing.assert_allclose(
            loaded_samples.std(axis=0), samples.std(axis=0), atol=0.2, rtol=0.3
        )
        loaded_info = dict(loaded.backend_sample_info)
        assert loaded_info["backend_correction_route"] == "sample_lmc_mixed_pathwise"
        assert loaded_info["training_route"] == method
        result = loaded_info
    elif case == "mixed_additive_supported":
        X, Y = generate_mixed_mo_data(n=2000, d_cont=2, T=2, levels=3, seed=141)
        kernels = [
            Kernel.rbf(active_dims=[0, 1]) + Kernel.ehh(levels=3, active_dims=[2]),
            Kernel.matern52(active_dims=[0, 1]),
        ]
        gp = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=4,
        )
        gp.fit(X, Y, max_iterations=2, learning_rate=0.03, verbose=False, method="matrix_free")

        samples = gp.sample_posterior(
            X[:4],
            n_samples=2,
            method="pathwise",
            n_rff_features=256,
            rng=np.random.default_rng(7),
        )
        info = dict(gp.backend_sample_info)
        assert samples.shape == (2, 4, 2)
        assert np.all(np.isfinite(samples))
        assert info["backend_correction_route"] == "sample_lmc_mixed_pathwise"
        assert info["prior_sampler_family"] == "shared_feature_map"
        result = {
            "supported": True,
            "backend_correction_route": info["backend_correction_route"],
            "prior_sampler_family": info["prior_sampler_family"],
        }
    else:
        raise ValueError(f"unknown case: {case}")

    _cleanup_gpu_state()
    return {"payload": result}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
