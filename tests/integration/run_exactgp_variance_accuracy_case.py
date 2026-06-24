"""Child entrypoints for ExactGP variance accuracy subprocess tests."""

from __future__ import annotations

import numpy as np

from mojogp import SingleOutputGP, RBF
from tests.shared.subprocess_harness import run_child_main


def _make_data(n_train: int = 2000, n_test: int = 256, seed: int = 29):
    rng = np.random.RandomState(seed)
    X_train = rng.uniform(-2.0, 2.0, size=(n_train, 1)).astype(np.float32)
    y_train = (np.sin(1.5 * X_train[:, 0]) + 0.05 * rng.randn(n_train)).astype(np.float32)
    X_test = np.linspace(-2.2, 2.2, n_test, dtype=np.float32).reshape(-1, 1)
    return X_train, y_train, X_test


def _run_closed_form_materialized() -> dict[str, float]:
    X_train, y_train, X_test = _make_data()
    gp = SingleOutputGP(RBF(), verbose=False)
    gp.fit(
        X_train,
        y_train,
        max_iterations=12,
        learning_rate=0.03,
        method="materialized",
        num_probes=5,
        max_cg_iterations=50,
        preconditioner_rank=10,
        preconditioner="greedy",
    )
    exact = gp.predict(X_test, variance_method="exact")

    params = np.asarray(gp.training_result.params, dtype=np.float32)
    mean = float(gp.training_result.mean)
    noise = float(gp.training_result.noise)
    K_train = gp.kernel.evaluate(X_train, params=params)
    K_cross = gp.kernel.evaluate(X_train, X_test, params=params)
    K_test = gp.kernel.evaluate(X_test, params=params)
    K_reg = K_train + noise * np.eye(X_train.shape[0], dtype=np.float32)
    centered_y = y_train.astype(np.float32) - np.float32(mean)
    alpha = np.linalg.solve(K_reg.astype(np.float64), centered_y.astype(np.float64))
    ref_mean = mean + K_cross.T.astype(np.float64) @ alpha
    solve_cross = np.linalg.solve(K_reg.astype(np.float64), K_cross.astype(np.float64))
    latent_var = np.diag(K_test.astype(np.float64)) - np.sum(
        K_cross.astype(np.float64) * solve_cross,
        axis=0,
    )
    ref_observed_var = latent_var + noise

    return {
        "mean_rmse": float(
            np.sqrt(np.mean((exact.mean - ref_mean.astype(np.float32)) ** 2))
        ),
        "var_rmse": float(
            np.sqrt(np.mean((exact.variance - ref_observed_var.astype(np.float32)) ** 2))
        ),
    }


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    case = str(payload["case"])
    if case == "closed_form_materialized":
        return {"payload": _run_closed_form_materialized()}
    raise ValueError(f"unknown case: {case}")


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
