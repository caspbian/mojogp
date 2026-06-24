"""Categorical correlation matrix formula checks for mixed GP routes."""

import numpy as np
import pytest

from mojogp.multi_output_gp import _compute_categorical_corr_matrix_py


def _params_for(kernel_type: str, levels: int) -> np.ndarray:
    if kernel_type == "gd":
        return np.array([0.7], dtype=np.float64)
    if kernel_type == "cr":
        return np.linspace(0.2, 0.6, levels, dtype=np.float64)
    if kernel_type in ("ehh", "hh"):
        return np.full(levels * (levels - 1) // 2, 0.4, dtype=np.float64)
    if kernel_type == "fe":
        num_angles = levels * (levels - 1) // 2
        return np.concatenate(
            [
                np.full(num_angles, 0.4, dtype=np.float64),
                np.full(levels, 0.3, dtype=np.float64),
            ]
        )
    raise AssertionError(f"unknown kernel_type={kernel_type}")


@pytest.mark.parametrize("kernel_type", ["gd", "cr", "ehh", "hh", "fe"])
def test_categorical_correlation_matrix_is_symmetric_psd(kernel_type: str):
    levels = 4
    corr = _compute_categorical_corr_matrix_py(
        kernel_type, levels, _params_for(kernel_type, levels)
    )

    np.testing.assert_allclose(corr, corr.T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.diag(corr), np.ones(levels), atol=1e-10, rtol=1e-10)
    eigvals = np.linalg.eigvalsh(corr)
    assert float(np.min(eigvals)) >= -1e-8


@pytest.mark.parametrize("kernel_type", ["gd", "cr", "ehh", "hh", "fe"])
def test_categorical_correlation_finite_difference_is_finite(kernel_type: str):
    levels = 4
    params = _params_for(kernel_type, levels)
    eps = 1e-5

    finite_diff_norms = []
    for idx in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[idx] += eps
        minus[idx] -= eps
        d_corr = (
            _compute_categorical_corr_matrix_py(kernel_type, levels, plus)
            - _compute_categorical_corr_matrix_py(kernel_type, levels, minus)
        ) / (2.0 * eps)
        assert np.all(np.isfinite(d_corr))
        np.testing.assert_allclose(d_corr, d_corr.T, atol=1e-8, rtol=1e-8)
        finite_diff_norms.append(float(np.linalg.norm(d_corr)))

    assert max(finite_diff_norms) > 1e-8
