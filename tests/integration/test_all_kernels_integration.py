#!/usr/bin/env python3
"""
Comprehensive integration test for all 8 kernel types with ExactGP.
Tests that all kernels work with the updated infrastructure including
training and prediction via the ExactGP wrapper.
"""

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

# Define all kernel configurations to test
KERNEL_CONFIGS = [
    {"name": "RBF", "kernel": RBF()},
    {"name": "Matern32", "kernel": Matern32()},
    {"name": "Matern52", "kernel": Matern52()},
    {"name": "Matern12", "kernel": Matern12()},
    {"name": "Periodic", "kernel": Periodic()},
    {"name": "RQ", "kernel": RQ()},
    {"name": "Linear", "kernel": Linear()},
    {"name": "Polynomial", "kernel": Polynomial()},
]


@pytest.fixture
def training_data():
    """Create small training/test dataset."""
    np.random.seed(42)
    n_train=2000
    n_test = 20
    dim = 3
    X_train = np.random.randn(n_train, dim).astype(np.float32)
    y_train = np.random.randn(n_train).astype(np.float32)
    X_test = np.random.randn(n_test, dim).astype(np.float32)
    return X_train, y_train, X_test


@pytest.mark.parametrize(
    "kernel_config", KERNEL_CONFIGS, ids=[c["name"] for c in KERNEL_CONFIGS]
)
def test_kernel_train_and_predict(kernel_config, training_data):
    """Test training and mean prediction for each kernel type."""
    X_train, y_train, X_test = training_data

    model = SingleOutputGP(kernel_config["kernel"])

    model.fit(X_train, y_train)

    y_pred, y_std = model.predict(X_test, return_std=True)

    assert not np.isnan(y_pred).any(), "Predictions contain NaN"
    assert not np.isinf(y_pred).any(), "Predictions contain Inf"
    assert len(y_pred) == len(X_test), "Wrong prediction length"


@pytest.mark.parametrize(
    "kernel_config", KERNEL_CONFIGS, ids=[c["name"] for c in KERNEL_CONFIGS]
)
def test_kernel_predict_with_variance(kernel_config, training_data):
    """Test prediction with variance for each kernel type."""
    X_train, y_train, X_test = training_data

    model = SingleOutputGP(kernel_config["kernel"])

    model.fit(X_train, y_train)

    result = model.predict(X_test)

    assert not np.isnan(result.mean).any(), "Mean predictions contain NaN"
    assert not np.isinf(result.mean).any(), "Mean predictions contain Inf"
    assert len(result.mean) == len(X_test), "Wrong mean prediction length"
    assert len(result.variance) == len(X_test), "Wrong variance prediction length"
