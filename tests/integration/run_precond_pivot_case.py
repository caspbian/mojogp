"""Child entrypoints for preconditioner pivot-method integration tests."""

from __future__ import annotations

import gc

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from mojogp import SingleOutputGP
from mojogp.kernel import RBF, Matern52
from tests.shared.subprocess_harness import run_child_main


N_TRAIN = 2000
N_TEST = 200
DIM = 5
N_ITER = 50
LEARNING_RATE = 0.1
PRECOND_RANK = 8
SEED = 42


def _cleanup_gpu_state() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _make_train_data(seed: int = SEED):
    np.random.seed(seed)
    X = np.random.randn(N_TRAIN, DIM).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(N_TRAIN)).astype(np.float32)
    return X, y


def _make_test_data():
    np.random.seed(SEED + 1)
    return np.random.randn(N_TEST, DIM).astype(np.float32)


def _fit_model(kernel, X, y, method: str | None, max_iterations: int):
    gp = SingleOutputGP(kernel)
    result = gp.fit(
        X,
        y,
        max_iterations=max_iterations,
        learning_rate=LEARNING_RATE,
        method="matrix_free",
        preconditioner=method,
        preconditioner_rank=PRECOND_RANK,
        verbose=False,
    )
    return gp, result


def _summarize(result) -> dict[str, object]:
    return {
        "nll": float(result.nll),
        "noise": float(result.noise),
        "params": np.asarray(result.params, dtype=np.float32).tolist(),
        "nll_history": np.asarray(result.nll_history, dtype=np.float32).tolist(),
    }


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    case = str(payload["case"])
    method_raw = payload.get("method")
    method = None if method_raw in (None, "__none__") else str(method_raw)

    _cleanup_gpu_state()

    if case == "initial_nll":
        X, y = _make_train_data()
        _, result = _fit_model(RBF(), X, y, method, max_iterations=1)
        response = {"nll": float(result.nll)}
    elif case == "trained_summary":
        X, y = _make_train_data()
        _, result = _fit_model(RBF(), X, y, method, max_iterations=N_ITER)
        response = _summarize(result)
    elif case == "prediction_summary":
        X, y = _make_train_data()
        X_test = _make_test_data()
        gp, _ = _fit_model(RBF(), X, y, method, max_iterations=N_ITER)
        mean, std = gp.predict(X_test, return_std=True)
        response = {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
            "mean_all_finite": bool(np.all(np.isfinite(mean))),
            "std_all_finite": bool(np.all(np.isfinite(std))),
            "std_min": float(np.min(std)),
            "num_nonpositive_std": int(np.sum(std <= 0)),
        }
    elif case == "ard_summary":
        X, y = _make_train_data()
        gp_init, result_init = _fit_model(RBF(ard=True), X, y, method, max_iterations=1)
        initial_nll = float(result_init.nll)
        del gp_init, result_init
        _cleanup_gpu_state()
        _, result = _fit_model(RBF(ard=True), X, y, method, max_iterations=N_ITER)
        response = {
            "initial_nll": initial_nll,
            "final_nll": float(result.nll),
            "params": np.asarray(result.params, dtype=np.float32).tolist(),
        }
    elif case == "composite_summary":
        X, y = _make_train_data()
        gp_init, result_init = _fit_model(
            RBF() + Matern52(), X, y, "nystrom", max_iterations=1
        )
        initial_nll = float(result_init.nll)
        del gp_init, result_init
        _cleanup_gpu_state()
        _, result = _fit_model(RBF() + Matern52(), X, y, "nystrom", max_iterations=N_ITER)
        response = {
            "initial_nll": initial_nll,
            "final_nll": float(result.nll),
            "best_nll": float(np.min(np.asarray(result.nll_history, dtype=np.float32))),
            "params": np.asarray(result.params, dtype=np.float32).tolist(),
        }
    else:
        raise ValueError(f"unknown case: {case}")

    _cleanup_gpu_state()
    return {"payload": response}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
