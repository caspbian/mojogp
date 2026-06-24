"""Integration checks for specialization metadata on the live wrapper path."""

from __future__ import annotations

import numpy as np

from mojogp import SingleOutputGP, RBF


def _single_output_data(seed: int = 17, n: int = 2000, d: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (
        np.sin(X[:, 0]) + 0.2 * np.cos(X[:, 1]) + 0.05 * rng.standard_normal(n)
    ).astype(np.float32)
    return X, y


def test_shadow_specialization_records_backend_metadata_without_changing_route():
    X, y = _single_output_data()
    gp = SingleOutputGP(RBF(), verbose=False)
    gp._set_specialization_request(
        {
            "mode": "shadow",
            "profile": {
                "specialization_key": "shadow_probe",
                "family": "jit_codegen",
                "source": "integration",
                "default_equivalent": True,
            },
        }
    )

    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="matrix_free",
        num_probes=2,
        max_cg_iterations=12,
        preconditioner_rank=6,
        max_tridiag_iterations=6,
        verbose=False,
    )

    info = gp.backend_train_info
    assert info is not None
    assert info["training_route"] == "matrix_free"
    assert info["specialization_mode"] == "shadow"
    assert info["specialization_key"] == "shadow_probe"
    assert info["specialization_default_equivalent"] is True
