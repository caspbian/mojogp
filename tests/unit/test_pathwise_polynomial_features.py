"""Polynomial feature-map checks for pathwise posterior sampling."""

import numpy as np
import pytest

from mojogp import Kernel
from mojogp.pathwise_prior import build_pathwise_feature_map


@pytest.mark.parametrize("degree", [1.0, 2.0, 3.0, 4.0])
def test_polynomial_feature_map_matches_kernel_matrix_for_fixed_integer_degrees(degree):
    rng = np.random.default_rng(42 + int(degree))
    X = rng.standard_normal((6, 3)).astype(np.float32)
    params = np.array([degree, 1.25, 0.7], dtype=np.float32)
    kernel = Kernel.polynomial(degree=degree)

    fmap = build_pathwise_feature_map(
        kernel,
        params,
        input_dim=X.shape[1],
        n_features=16,
        rng=rng,
    )
    features = fmap.evaluate(X, None)

    expected = kernel.evaluate(X, X, params=params)
    np.testing.assert_allclose(features @ features.T, expected, atol=1e-5, rtol=1e-5)
    assert fmap.is_exact is True


def test_polynomial_pathwise_rejects_non_integer_degree():
    rng = np.random.default_rng(7)
    with pytest.raises(NotImplementedError, match="positive integer degree"):
        build_pathwise_feature_map(
            Kernel.polynomial(),
            np.array([2.4, 1.0, 1.0], dtype=np.float32),
            input_dim=2,
            n_features=8,
            rng=rng,
        )


def test_polynomial_pathwise_rejects_degree_zero_to_match_public_kernel_route():
    rng = np.random.default_rng(17)
    with pytest.raises(NotImplementedError, match="positive integer degree"):
        build_pathwise_feature_map(
            Kernel.polynomial(degree=0.0),
            np.array([0.0, 1.0, 1.0], dtype=np.float32),
            input_dim=2,
            n_features=8,
            rng=rng,
        )


def test_polynomial_pathwise_rejects_negative_offset_and_outputscale():
    rng = np.random.default_rng(23)
    kernel = Kernel.polynomial(degree=2.0)
    with pytest.raises(NotImplementedError, match="non-negative offset"):
        build_pathwise_feature_map(
            kernel,
            np.array([2.0, -0.1, 1.0], dtype=np.float32),
            input_dim=2,
            n_features=8,
            rng=rng,
        )
    with pytest.raises(NotImplementedError, match="non-negative outputscale"):
        build_pathwise_feature_map(
            kernel,
            np.array([2.0, 1.0, -0.1], dtype=np.float32),
            input_dim=2,
            n_features=8,
            rng=rng,
        )


def test_polynomial_pathwise_rejects_excessive_exact_feature_expansion():
    rng = np.random.default_rng(29)
    with pytest.raises(NotImplementedError, match="feature map exceeds"):
        build_pathwise_feature_map(
            Kernel.polynomial(degree=5.0),
            np.array([5.0, 1.0, 1.0], dtype=np.float32),
            input_dim=5,
            n_features=8,
            rng=rng,
            feature_cap=512,
        )
