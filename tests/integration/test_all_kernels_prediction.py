#!/usr/bin/env python3
"""Test prediction with all 8 kernel types."""

import numpy as np
import pytest
from mojogp import (
    SingleOutputGP,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    RQ,
    Linear,
    Polynomial,
)


@pytest.fixture
def training_data():
    """Generate test data."""
    np.random.seed(42)
    n_train=2000
    n_test = 20
    dim = 5

    X_train = np.random.randn(n_train, dim).astype(np.float32)
    y_train = np.sin(X_train[:, 0]).astype(np.float32)
    X_test = np.random.randn(n_test, dim).astype(np.float32)

    return X_train, y_train, X_test, n_test


@pytest.mark.parametrize(
    "kernel",
    [
        RBF(),
        Matern32(),
        Matern52(),
        Matern12(),
        Periodic(),
        RQ(),
        Linear(),
        Polynomial(),
    ],
    ids=[
        "rbf",
        "matern32",
        "matern52",
        "matern12",
        "periodic",
        "rq",
        "linear",
        "polynomial",
    ],
)
def test_kernel_prediction(kernel, training_data):
    """Test that each kernel can train and predict without errors."""
    X_train, y_train, X_test, n_test = training_data

    # Create GP with kernel and data
    gp = SingleOutputGP(kernel)

    # Train
    gp.fit(X_train, y_train, max_iterations=20)

    # Predict (returns mean, std tuple)
    mean, std = gp.predict(X_test, return_std=True)

    # Check results
    assert mean is not None, "Prediction returned None"
    assert len(mean) == n_test, (
        f"Prediction returned wrong length: {len(mean)} != {n_test}"
    )
    assert np.all(np.isfinite(mean)), f"Prediction contains NaN/Inf: {mean[:5]}"


if __name__ == "__main__":
    # Allow running as a script for manual testing
    import sys

    kernels = [
        ("rbf", RBF()),
        ("matern32", Matern32()),
        ("matern52", Matern52()),
        ("matern12", Matern12()),
        ("periodic", Periodic()),
        ("rq", RQ()),
        ("linear", Linear()),
        ("polynomial", Polynomial()),
    ]

    print("=" * 60)
    print("Testing Prediction with All 8 Kernels")
    print("=" * 60)

    np.random.seed(42)
    n_train, n_test, dim = 100, 20, 5
    X_train = np.random.randn(n_train, dim).astype(np.float32)
    y_train = np.sin(X_train[:, 0]).astype(np.float32)
    X_test = np.random.randn(n_test, dim).astype(np.float32)

    print(f"\nDataset: n_train={n_train}, n_test={n_test}, dim={dim}\n")

    results = []
    for kernel_name, kernel_obj in kernels:
        try:
            print(f"Testing {kernel_name:12} ... ", end="", flush=True)
            gp = SingleOutputGP(kernel_obj)
            gp.fit(X_train, y_train, max_iterations=20)
            mean, std = gp.predict(X_test, return_std=True)

            assert (
                mean is not None and len(mean) == n_test and np.all(np.isfinite(mean))
            )

            nll = gp._training_result.nll if gp._training_result else "N/A"
            print(
                f"OK NLL: {nll:.4f}, Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]"
            )
            results.append(True)
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Summary: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    sys.exit(0 if all(results) else 1)
