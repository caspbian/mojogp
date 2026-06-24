"""Pathwise feature-map support-boundary checks."""

import numpy as np
import pytest

from mojogp import Kernel
from mojogp.pathwise_prior import build_pathwise_feature_map


def test_continuous_product_feature_map_rejects_excessive_feature_count():
    rng = np.random.default_rng(11)
    kernel = Kernel.rbf() * Kernel.matern52()

    with pytest.raises(NotImplementedError, match="feature product exceeds"):
        build_pathwise_feature_map(
            kernel,
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            input_dim=2,
            n_features=9,
            rng=rng,
            feature_cap=64,
        )


def test_mixed_product_feature_map_rejects_excessive_feature_count():
    rng = np.random.default_rng(13)
    kernel = Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=4, active_dims=[1])

    with pytest.raises(NotImplementedError, match="feature product exceeds"):
        build_pathwise_feature_map(
            kernel,
            np.array([1.0, 1.0], dtype=np.float32),
            input_dim=1,
            n_features=32,
            rng=rng,
            cat_params=np.zeros(6, dtype=np.float32),
            cat_col_map={1: 0},
            feature_cap=100,
        )
