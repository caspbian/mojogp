"""Small inference benchmark for the live SingleOutputGP wrapper.

This script prints local timings only. It is not the canonical source of any
repo-level performance claim.
"""

import time

import numpy as np

from mojogp import SingleOutputGP, RBF


def benchmark(method: str, n_train: int = 5000, n_test: int = 2048):
    rng = np.random.RandomState(42)
    X_train = rng.randn(n_train, 2).astype(np.float32)
    y_train = (
        np.sin(X_train[:, 0]) + 0.1 * X_train[:, 1] + 0.05 * rng.randn(X_train.shape[0])
    ).astype(np.float32)
    X_test = rng.randn(n_test, 2).astype(np.float32)

    gp = SingleOutputGP(RBF(), verbose=False)

    t0 = time.perf_counter()
    gp.fit(X_train, y_train, max_iterations=15, learning_rate=0.03, method=method)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    mean, std = gp.predict(X_test, return_std=True)
    pred_s = time.perf_counter() - t1

    print(f"Method: {method}")
    print(f"  fit time (s):  {fit_s:.3f}")
    print(f"  pred time (s): {pred_s:.3f}")
    print(f"  pred/s:        {X_test.shape[0] / pred_s:.1f}")
    print(f"  mean/std avg:  {float(np.mean(mean)):.4f} / {float(np.mean(std)):.4f}")


if __name__ == "__main__":
    benchmark("materialized")
    benchmark("matrix_free")
