"""Integration checks for MultiOutputGP specialization metadata."""

from __future__ import annotations

import numpy as np
import pytest

from mojogp import MultiOutputGP


def _multi_output_data(seed: int = 31, n: int = 2000, d: int = 3, t: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    base = np.sin(X[:, 0]) + 0.25 * X[:, 1]
    Y = np.column_stack(
        [
            base + 0.05 * rng.standard_normal(n),
            0.8 * base + 0.2 * np.cos(X[:, 1]) + 0.05 * rng.standard_normal(n),
        ][:t]
    ).astype(np.float32)
    return X, Y


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_multi_output_shadow_specialization_records_train_and_predict_metadata(method):
    X, Y = _multi_output_data(seed=71 if method == "materialized" else 73)
    gp = MultiOutputGP(
        kernel="rbf",
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp._set_specialization_request(
        {
            "mode": "shadow",
            "profile": {
                "specialization_key": f"shadow_{method}",
                "family": "jit_codegen",
                "source": "integration",
                "default_equivalent": True,
            },
        }
    )

    gp.fit(X, Y, max_iterations=2, learning_rate=0.03, verbose=False, method=method)

    train_info = gp.backend_train_info
    assert train_info is not None
    assert train_info["training_route"] == method
    assert train_info["specialization_mode"] == "shadow"
    assert train_info["specialization_key"] == f"shadow_{method}"
    assert train_info["specialization_default_equivalent"] is True

    mean, var = gp.predict(X[:12], return_var=True, variance_method="exact")
    assert mean.shape == (12, 2)
    assert var.shape == (12, 2)

    predict_info = gp.backend_predict_info
    assert predict_info is not None
    assert predict_info["training_route"] == method
    assert predict_info["specialization_mode"] == "shadow"
    assert predict_info["specialization_key"] == f"shadow_{method}"
    assert predict_info["backend_prediction_used"] is True
    assert predict_info["fallback_used"] is False
