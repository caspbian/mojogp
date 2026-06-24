"""Shared helpers for categorical/mixed ablation certification tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mojogp.kernel import Kernel


CATEGORICAL_KERNELS: tuple[str, ...] = ("gd", "cr", "ehh", "hh", "fe")


@dataclass(frozen=True)
class CategoricalAblationDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    f_test: np.ndarray
    X_train_shuffled: np.ndarray
    X_test_shuffled: np.ndarray
    category_probe_X: np.ndarray
    true_category_effects: np.ndarray
    levels: int
    continuous_dim: int


@dataclass(frozen=True)
class MixedMultiOutputAblationDataset:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    F_test: np.ndarray
    X_test_shuffled: np.ndarray
    category_probe_X: np.ndarray
    true_category_effects: np.ndarray
    levels: int
    continuous_dim: int
    num_tasks: int


def generate_categorical_ablation_dataset(
    *,
    n_train: int = 2000,
    n_test: int = 128,
    continuous_dim: int = 2,
    levels: int = 3,
    seed: int = 0,
    noise_std: float = 0.08,
) -> CategoricalAblationDataset:
    """Generate a strong categorical-signal dataset with negative controls.

    The data-generating function is additive in a smooth continuous term and a
    known categorical offset. The shuffled control preserves categorical level
    marginals while breaking the semantic link between category and response.
    """

    rng = np.random.default_rng(seed)
    n_total = int(n_train) + int(n_test)
    X_cont = rng.normal(size=(n_total, continuous_dim)).astype(np.float32)
    categories = rng.integers(0, levels, size=n_total, endpoint=False)
    true_effects = np.linspace(-2.0, 2.0, levels, dtype=np.float32)
    continuous_signal = (
        np.sin(X_cont[:, 0]) + np.float32(0.25) * X_cont[:, 1]
    ).astype(np.float32)
    f = (continuous_signal + true_effects[categories]).astype(np.float32)
    y = (f + noise_std * rng.standard_normal(n_total)).astype(np.float32)

    X_full = np.column_stack([X_cont, categories.astype(np.float32)]).astype(np.float32)
    X_shuffled = X_full.copy()
    X_shuffled[:n_train, continuous_dim] = rng.permutation(
        X_shuffled[:n_train, continuous_dim]
    )
    X_shuffled[n_train:, continuous_dim] = rng.permutation(
        X_shuffled[n_train:, continuous_dim]
    )

    probe_cont = np.zeros((levels, continuous_dim), dtype=np.float32)
    probe_cat = np.arange(levels, dtype=np.float32).reshape(levels, 1)
    category_probe_X = np.column_stack([probe_cont, probe_cat]).astype(np.float32)

    return CategoricalAblationDataset(
        X_train=X_full[:n_train],
        y_train=y[:n_train],
        X_test=X_full[n_train:],
        f_test=f[n_train:],
        X_train_shuffled=X_shuffled[:n_train],
        X_test_shuffled=X_shuffled[n_train:],
        category_probe_X=category_probe_X,
        true_category_effects=true_effects,
        levels=levels,
        continuous_dim=continuous_dim,
    )


def generate_mixed_multi_output_ablation_dataset(
    *,
    n_train: int = 2000,
    n_test: int = 128,
    continuous_dim: int = 2,
    levels: int = 3,
    num_tasks: int = 2,
    seed: int = 0,
    noise_std: float = 0.08,
) -> MixedMultiOutputAblationDataset:
    """Generate a multi-output mixed dataset with a strong categorical signal."""

    if num_tasks != 2:
        raise ValueError("mixed multi-output certification currently expects two tasks")

    rng = np.random.default_rng(seed)
    n_total = int(n_train) + int(n_test)
    X_cont = rng.normal(size=(n_total, continuous_dim)).astype(np.float32)
    categories = rng.integers(0, levels, size=n_total, endpoint=False)
    true_effects = np.linspace(-2.0, 2.0, levels, dtype=np.float32)
    category_effect = true_effects[categories]

    F = np.empty((n_total, num_tasks), dtype=np.float32)
    F[:, 0] = np.sin(X_cont[:, 0]) + np.float32(0.25) * X_cont[:, 1] + category_effect
    F[:, 1] = (
        np.float32(0.65) * np.cos(X_cont[:, 0])
        - np.float32(0.2) * X_cont[:, 1]
        + np.float32(0.8) * category_effect
    )
    Y = (F + noise_std * rng.standard_normal((n_total, num_tasks))).astype(np.float32)

    X_full = np.column_stack([X_cont, categories.astype(np.float32)]).astype(np.float32)
    X_shuffled = X_full.copy()
    X_shuffled[n_train:, continuous_dim] = rng.permutation(
        X_shuffled[n_train:, continuous_dim]
    )

    probe_cont = np.zeros((levels, continuous_dim), dtype=np.float32)
    probe_cat = np.arange(levels, dtype=np.float32).reshape(levels, 1)
    category_probe_X = np.column_stack([probe_cont, probe_cat]).astype(np.float32)

    return MixedMultiOutputAblationDataset(
        X_train=X_full[:n_train],
        Y_train=Y[:n_train],
        X_test=X_full[n_train:],
        F_test=F[n_train:],
        X_test_shuffled=X_shuffled[n_train:],
        category_probe_X=category_probe_X,
        true_category_effects=true_effects,
        levels=levels,
        continuous_dim=continuous_dim,
        num_tasks=num_tasks,
    )


def continuous_kernel(continuous_dim: int):
    return Kernel.rbf(active_dims=list(range(int(continuous_dim))))


def mixed_kernel(kernel_name: str, *, continuous_dim: int, levels: int):
    try:
        categorical_factory = getattr(Kernel, kernel_name)
    except AttributeError as exc:  # pragma: no cover - guarded by test params
        raise ValueError(f"Unknown categorical kernel {kernel_name!r}") from exc
    return continuous_kernel(continuous_dim) * categorical_factory(
        levels=int(levels), active_dims=[int(continuous_dim)]
    )


def lmc_continuous_kernels(continuous_dim: int):
    return [
        Kernel.rbf(active_dims=list(range(int(continuous_dim)))),
        Kernel.matern52(active_dims=list(range(int(continuous_dim)))),
    ]


def lmc_mixed_kernels(kernel_name: str, *, continuous_dim: int, levels: int):
    categorical_factory = getattr(Kernel, kernel_name)
    active_dims = list(range(int(continuous_dim)))
    return [
        Kernel.rbf(active_dims=active_dims)
        * categorical_factory(levels=int(levels), active_dims=[int(continuous_dim)]),
        Kernel.matern52(active_dims=active_dims)
        * categorical_factory(levels=int(levels), active_dims=[int(continuous_dim)]),
    ]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def centered_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def route_summary(gp: Any) -> dict[str, Any]:
    train_info = dict(getattr(gp, "backend_train_info", {}) or {})
    predict_info = dict(getattr(gp, "backend_predict_info", {}) or {})
    return {
        "train": train_info,
        "predict": predict_info,
        "training_route": train_info.get("training_route"),
        "prediction_route": predict_info.get("actual_prediction_route"),
        "variance_route": predict_info.get("actual_variance_route"),
        "fallback_used": bool(predict_info.get("fallback_used", False)),
        "backend_prediction_used": bool(
            predict_info.get("backend_prediction_used", False)
        ),
    }
