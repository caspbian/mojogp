"""Child entrypoints for LMC integration subprocess tests."""

from __future__ import annotations

import gc
import tempfile

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


def generate_mixed_multi_output_data(
    n: int = 5000,
    d_cont: int = 2,
    T: int = 2,
    levels: int = 3,
    seed: int = 17,
):
    rng = np.random.default_rng(seed)
    X_cont = rng.standard_normal((n, d_cont)).astype(np.float32)
    cat = rng.integers(0, levels, size=(n, 1), dtype=np.int32)
    X = np.concatenate([X_cont, cat.astype(np.float32)], axis=1)
    cat_effect = np.linspace(-0.6, 0.6, levels, dtype=np.float32)[cat[:, 0]]
    Y = np.zeros((n, T), dtype=np.float32)
    Y[:, 0] = np.sin(X_cont[:, 0]) + 0.35 * X_cont[:, 1] + 0.8 * cat_effect
    Y[:, 1] = 0.7 * np.cos(X_cont[:, 0]) - 0.2 * X_cont[:, 1] + 0.5 * cat_effect
    Y += 0.05 * rng.standard_normal((n, T)).astype(np.float32)
    return X.astype(np.float32), Y.astype(np.float32)


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    case = str(payload["case"])
    method = str(payload["method"])

    _cleanup_gpu_state()

    if case == "category_change":
        n_train = 5000 if method == "materialized" else 8000
        X, Y = generate_mixed_multi_output_data(
            n=n_train,
            d_cont=2,
            T=2,
            levels=3,
            seed=31,
        )
        kernels = [
            Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
            Kernel.matern52(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
        ]
        gp = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=8,
        )
        gp.fit(X, Y, max_iterations=4, learning_rate=0.03, verbose=False, method=method)
        X_probe = np.array(
            [[0.15, -0.35, 0.0], [0.15, -0.35, 1.0], [0.15, -0.35, 2.0]],
            dtype=np.float32,
        )
        mean = gp.predict(X_probe).mean
        assert mean.shape == (3, 2)
        assert np.all(np.isfinite(mean))
        assert float(np.ptp(mean[:, 0])) > 0.05
        assert float(np.ptp(mean[:, 1])) > 0.05
        result = {"ok": True}
    elif case == "categorical_kernel":
        kernel_name = str(payload["kernel_name"])
        kernel_builders = {
            "gd": Kernel.gd,
            "cr": Kernel.cr,
            "ehh": Kernel.ehh,
            "hh": Kernel.hh,
            "fe": Kernel.fe,
        }
        if kernel_name not in kernel_builders:
            raise ValueError(f"unknown categorical kernel: {kernel_name}")
        n_train = 5000 if method == "materialized" else 8000
        X, Y = generate_mixed_multi_output_data(
            n=n_train,
            d_cont=2,
            T=2,
            levels=3,
            seed=61,
        )
        cat_kernel = kernel_builders[kernel_name](levels=3, active_dims=[2])
        kernels = [
            Kernel.rbf(active_dims=[0, 1]) * cat_kernel,
            Kernel.matern52(active_dims=[0, 1]),
        ]
        gp = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=8,
        )
        trained = gp.fit(
            X, Y, max_iterations=4, learning_rate=0.03, verbose=False, method=method
        )
        X_probe = np.array(
            [[0.15, -0.35, 0.0], [0.15, -0.35, 1.0], [0.15, -0.35, 2.0]],
            dtype=np.float32,
        )
        mean, var = gp.predict(X_probe, return_var=True, variance_method="exact")
        exact_info = dict(gp.backend_predict_info)
        love_mean, love_var = gp.predict(
            X_probe, return_var=True, variance_method="love"
        )
        love_info = dict(gp.backend_predict_info)
        assert mean.shape == (3, 2)
        assert var.shape == (3, 2)
        assert love_mean.shape == (3, 2)
        assert love_var.shape == (3, 2)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(np.isfinite(love_mean))
        assert np.all(np.isfinite(love_var))
        assert exact_info["actual_prediction_route"] == "predict_lmc_mixed"
        assert exact_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"
        assert exact_info["fallback_used"] is False
        assert love_info["variance_method"] == "love"
        assert love_info["actual_prediction_route"] == "predict_lmc_mixed"
        assert love_info["actual_variance_route"] == "predict_lmc_mixed"
        assert love_info["fallback_used"] is False
        np.testing.assert_allclose(love_mean, mean, atol=1e-5, rtol=1e-5)
        sensitivity = float(np.ptp(mean[:, 0]) + np.ptp(mean[:, 1]))
        assert sensitivity > 0.05

        with tempfile.TemporaryDirectory(prefix=f"mojogp_lmc_{kernel_name}_{method}_") as tmp_dir:
            gp.save(tmp_dir)
            loaded = MultiOutputLMCGP.load(tmp_dir)
            loaded_mean, loaded_var = loaded.predict(
                X_probe, return_var=True, variance_method="exact"
            )
            loaded_exact_info = dict(loaded.backend_predict_info)
            loaded_love_mean, loaded_love_var = loaded.predict(
                X_probe, return_var=True, variance_method="love"
            )
            loaded_love_info = dict(loaded.backend_predict_info)
        np.testing.assert_allclose(loaded_mean, mean, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(loaded_var, var, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(loaded_love_mean, love_mean, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(
            loaded_love_var, love_var, atol=1.5e-2, rtol=1e-1
        )
        result = {
            "kernel_name": kernel_name,
            "actual_prediction_route": exact_info["actual_prediction_route"],
            "actual_variance_route": exact_info["actual_variance_route"],
            "love_variance_route": love_info["actual_variance_route"],
            "loaded_variance_route": loaded_exact_info["actual_variance_route"],
            "loaded_love_variance_route": loaded_love_info["actual_variance_route"],
            "categorical_sensitivity": sensitivity,
            "cat_param_counts": [len(p) for p in trained.cat_params_per_latent],
        }
    elif case == "nested_tree":
        n_train = 5000 if method == "materialized" else 8000
        X, Y = generate_mixed_multi_output_data(
            n=n_train,
            d_cont=2,
            T=2,
            levels=3,
            seed=41,
        )
        kernels = [
            (Kernel.rbf(active_dims=[0, 1]) + Kernel.matern32(active_dims=[0, 1]))
            * (Kernel.gd(levels=3, active_dims=[2]) + Kernel.cr(levels=3, active_dims=[2])),
            Kernel.matern52(active_dims=[0, 1]),
        ]
        gp = MultiOutputLMCGP(
            kernels=kernels,
            num_probes=3,
            max_cg_iterations=30,
            preconditioner_rank=8,
        )
        gp.fit(X, Y, max_iterations=2, learning_rate=0.03, verbose=False, method=method)
        mean, var = gp.predict(X[:8], return_var=True, variance_method="exact")
        assert mean.shape == (8, 2)
        assert var.shape == (8, 2)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert gp.backend_predict_info["actual_prediction_route"] == "predict_lmc_mixed"
        assert gp.backend_predict_info["actual_variance_route"] == "predict_lmc_mixed_full_exact"
        assert gp.backend_predict_info["backend_prediction_used"] is True
        assert gp.backend_predict_info["backend_variance_used"] is True
        assert gp.backend_predict_info["fallback_used"] is False
        assert gp.backend_predict_info["precond_rank"] == 8
        assert gp.backend_predict_info["precond_method"] == gp.precond_method
        result = {
            "actual_prediction_route": gp.backend_predict_info["actual_prediction_route"],
            "actual_variance_route": gp.backend_predict_info["actual_variance_route"],
            "precond_rank": gp.backend_predict_info["precond_rank"],
            "precond_method": gp.backend_predict_info["precond_method"],
        }
    elif case == "fixed_observation_noise":
        rng = np.random.default_rng(53)
        n_train = 2000
        X = rng.standard_normal((n_train, 3)).astype(np.float32)
        fixed_noise = (
            0.01
            + 0.02 * (X[:, [0]] > 0).astype(np.float32)
            + np.array([[0.0, 0.015]], dtype=np.float32)
        ).astype(np.float32)
        Y = np.empty((n_train, 2), dtype=np.float32)
        Y[:, 0] = np.sin(X[:, 0]) + 0.2 * X[:, 1]
        Y[:, 1] = np.cos(X[:, 0]) - 0.1 * X[:, 2]
        Y += rng.standard_normal((n_train, 2)).astype(np.float32) * np.sqrt(fixed_noise)

        gp = MultiOutputLMCGP(
            kernels=[Kernel.rbf()],
            num_probes=2,
            max_cg_iterations=20,
            preconditioner_rank=6,
        )
        train_result = gp.fit(
            X,
            Y,
            fixed_observation_noise=fixed_noise,
            initial_noise_per_task=np.full(2, 0.02, dtype=np.float32),
            max_iterations=2,
            learning_rate=0.02,
            verbose=False,
            method=method,
        )
        mean, var = gp.predict(X[:5], return_var=True, variance_method="exact")
        assert mean.shape == (5, 2)
        assert var.shape == (5, 2)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(np.isfinite(train_result.nll_history))
        assert gp.backend_train_info["training_route"] == method
        assert gp.backend_train_info["uses_fixed_observation_noise"] is True
        assert gp.backend_predict_info["uses_fixed_observation_noise"] is True
        assert gp._fixed_observation_noise is not None
        np.testing.assert_allclose(gp._fixed_observation_noise[:4], fixed_noise[:4])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/fixed_lmc"
            gp.save(path)
            loaded = MultiOutputLMCGP.load(path)
            assert loaded._fixed_observation_noise is not None
            np.testing.assert_allclose(loaded._fixed_observation_noise[:4], fixed_noise[:4])
            loaded_mean, loaded_var = loaded.predict(
                X[:5], return_var=True, variance_method="exact"
            )
            assert np.all(np.isfinite(loaded_mean))
            assert np.all(np.isfinite(loaded_var))

        result = {
            "ok": True,
            "training_route": gp.backend_train_info["training_route"],
            "fixed_noise_mean": float(np.mean(fixed_noise)),
        }
    else:
        raise ValueError(f"unknown case: {case}")

    _cleanup_gpu_state()
    return {"payload": result}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
