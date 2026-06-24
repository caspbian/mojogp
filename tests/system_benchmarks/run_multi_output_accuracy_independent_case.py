"""Run one independent ExactGP baseline case for multi-output accuracy."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mojogp import SingleOutputGP, RBF, Matern12, Matern32, Matern52, Periodic, RQ, Linear, Polynomial
from tests.shared.subprocess_harness import run_child_main


def _make_exactgp_kernel(kernel: str):
    mapping = {
        "rbf": RBF,
        "matern12": Matern12,
        "matern32": Matern32,
        "matern52": Matern52,
        "periodic": Periodic,
        "rq": RQ,
        "linear": Linear,
        "polynomial": Polynomial,
    }
    return mapping[kernel]()


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    input_path = Path(str(payload["input_path"]))
    output_path = Path(str(payload["output_path"]))
    data = np.load(input_path)
    gp = SingleOutputGP(_make_exactgp_kernel(str(payload["kernel"])))

    start = time.perf_counter()
    gp.fit(
        data["X_train"],
        data["y_train"],
        method=str(payload["method"]),
        max_iterations=int(payload["max_iterations"]),
        learning_rate=float(payload["learning_rate"]),
        verbose=False,
    )
    fit_time = time.perf_counter() - start
    pred = gp.predict(data["X_test"])
    np.savez(
        output_path,
        mean=np.asarray(pred.mean, dtype=np.float32),
        variance=np.asarray(pred.variance, dtype=np.float32),
        fit_time=np.asarray([fit_time], dtype=np.float64),
    )
    return {"result_path": output_path}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
