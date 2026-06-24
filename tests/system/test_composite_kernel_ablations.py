"""Composite-kernel ablation benchmarks on the live wrapper path."""

from __future__ import annotations

import numpy as np
import pytest

from mojogp import SingleOutputGP, Kernel, Periodic, RBF

from tests.shared.gpu_test_utils import assert_gpu_available, requires_cuda


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def generate_composite_signal_data(
    n_train: int,
    n_test: int,
    noise_level: str = "medium",
    seed: int = 42,
):
    noise_levels = {
        "low": 0.05,
        "medium": 0.1,
        "high": 0.2,
    }
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    X = rng.uniform(0.0, 6.0, size=(n_total, 1)).astype(np.float32)
    smooth = 0.6 * np.sin(0.6 * X[:, 0])
    periodic = 1.0 * np.sin(2.5 * X[:, 0])
    trend = 0.15 * X[:, 0]
    f = (smooth + periodic + trend).astype(np.float32)
    noise_std = noise_levels.get(noise_level, 0.1) * np.std(f)
    y = f + noise_std * rng.randn(n_total).astype(np.float32)

    class _Dataset:
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]
        f_test = f[n_train:]

    return _Dataset()


@pytest.mark.minimal
@pytest.mark.accuracy
@requires_cuda
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_composite_kernel_handles_mixed_signal_better_than_mismatched_baseline(
    method: str,
):
    assert_gpu_available()
    n_train = 2000 if method == "materialized" else 3500
    dataset = generate_composite_signal_data(
        n_train=n_train,
        n_test=200,
        noise_level="medium",
        seed=42 if method == "materialized" else 99,
    )

    composite = SingleOutputGP(Kernel.rbf() + Kernel.periodic())
    smooth_only = SingleOutputGP(RBF())
    periodic_only = SingleOutputGP(Periodic())

    fit_kwargs = {
        "method": method,
        "max_iterations": 80 if method == "materialized" else 50,
        "learning_rate": 0.03,
        "verbose": False,
    }
    composite.fit(dataset.X_train, dataset.y_train, **fit_kwargs)
    smooth_only.fit(dataset.X_train, dataset.y_train, **fit_kwargs)
    periodic_only.fit(dataset.X_train, dataset.y_train, **fit_kwargs)

    composite_pred = composite.predict(dataset.X_test).mean
    smooth_pred = smooth_only.predict(dataset.X_test).mean
    periodic_pred = periodic_only.predict(dataset.X_test).mean

    composite_rmse = rmse(dataset.f_test, composite_pred)
    smooth_rmse = rmse(dataset.f_test, smooth_pred)
    periodic_rmse = rmse(dataset.f_test, periodic_pred)

    composite_r2 = r_squared(dataset.f_test, composite_pred)
    smooth_r2 = r_squared(dataset.f_test, smooth_pred)

    # This benchmark's stable signal is mixed smooth + periodic structure.
    # In current wrapper training, an RBF-only model can still fit it well,
    # but the mismatched periodic-only model underfits badly. Require the
    # composite model to stay highly accurate, decisively beat the mismatched
    # single-kernel baseline, and avoid a large regression versus the best
    # simpler single-kernel fit.
    assert composite_rmse <= periodic_rmse * 0.2, (
        f"Composite kernel did not clearly beat periodic-only on {method}: "
        f"composite={composite_rmse:.4f}, periodic={periodic_rmse:.4f}"
    )
    assert composite_rmse <= smooth_rmse * 2.0, (
        f"Composite kernel regressed too far versus RBF on {method}: "
        f"composite={composite_rmse:.4f}, smooth={smooth_rmse:.4f}"
    )
    assert composite_r2 >= 0.999, (
        f"Composite kernel accuracy is too low on {method}: R^2={composite_r2:.4f}"
    )
    assert composite_r2 >= smooth_r2 - 0.01, (
        f"Composite kernel lost too much R^2 versus RBF on {method}: "
        f"composite={composite_r2:.4f}, smooth={smooth_r2:.4f}"
    )
