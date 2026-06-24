"""Offline real datasets for benchmark-style validation.

These loaders use deterministic built-in scikit-learn datasets so the test
surface remains offline and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes, load_linnerud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "real"


def has_energy_efficiency_multi_output_data() -> bool:
    return (REAL_DATA_ROOT / "energy_efficiency.csv").exists()


@dataclass
class RealSingleOutputDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str
    description: str


@dataclass
class RealMultiOutputDataset:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    name: str
    description: str


def _scale_single_output(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> RealSingleOutputDataset:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)
    y_train_scaled = (
        y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    )
    y_test_scaled = (
        y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1).astype(np.float32)
    )
    return RealSingleOutputDataset(
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_test=X_test_scaled,
        y_test=y_test_scaled,
        name="",
        description="",
    )


def _scale_multi_output(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
) -> RealMultiOutputDataset:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = x_scaler.transform(X_test).astype(np.float32)
    Y_train_scaled = y_scaler.fit_transform(Y_train).astype(np.float32)
    Y_test_scaled = y_scaler.transform(Y_test).astype(np.float32)
    return RealMultiOutputDataset(
        X_train=X_train_scaled,
        Y_train=Y_train_scaled,
        X_test=X_test_scaled,
        Y_test=Y_test_scaled,
        name="",
        description="",
    )


def load_diabetes_regression(
    test_size: float = 0.2, seed: int = 42
) -> RealSingleOutputDataset:
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=seed,
    )
    dataset = _scale_single_output(X_train, X_test, y_train, y_test)
    dataset.name = "diabetes"
    dataset.description = "Built-in sklearn diabetes regression dataset"
    return dataset


def load_linnerud_multi_output(
    test_size: float = 0.25,
    seed: int = 42,
) -> RealMultiOutputDataset:
    data = load_linnerud()
    X_train, X_test, Y_train, Y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=seed,
    )
    dataset = _scale_multi_output(X_train, X_test, Y_train, Y_test)
    dataset.name = "linnerud"
    dataset.description = "Built-in sklearn Linnerud multi-output regression dataset"
    return dataset


def load_energy_efficiency_multi_output(
    test_size: float = 0.2,
    seed: int = 42,
) -> RealMultiOutputDataset:
    if not has_energy_efficiency_multi_output_data():
        raise FileNotFoundError(f"{REAL_DATA_ROOT / 'energy_efficiency.csv'} not found.")
    data = np.loadtxt(
        REAL_DATA_ROOT / "energy_efficiency.csv",
        delimiter=",",
        skiprows=1,
        dtype=np.float32,
    )
    X = data[:, :8]
    Y = data[:, 8:10]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=seed,
    )
    dataset = _scale_multi_output(X_train, X_test, Y_train, Y_test)
    dataset.name = "energy_efficiency"
    dataset.description = (
        "Vendored UCI Energy Efficiency multi-output regression dataset"
    )
    return dataset
