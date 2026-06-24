"""Save/load provenance checks for multi-output specialization metadata."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mojogp import MultiOutputGP


def _multi_output_data(seed: int = 41, n: int = 2000, d: int = 3, t: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = np.column_stack(
        [
            np.sin(X[:, 0]) + 0.2 * X[:, 1] + 0.05 * rng.standard_normal(n),
            np.cos(X[:, 0]) - 0.1 * X[:, 2] + 0.05 * rng.standard_normal(n),
        ][:t]
    ).astype(np.float32)
    return X, Y


@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_multi_output_save_load_preserves_specialization_as_provenance_only(tmp_path, method):
    X, Y = _multi_output_data(seed=91 if method == "materialized" else 93)
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

    path = tmp_path / f"multi_output_{method}"
    gp.save(path)

    with open(f"{path}_config.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    assert config["specialization"]["mode"] == "shadow"
    assert config["specialization"]["profile"]["specialization_key"] == f"shadow_{method}"

    loaded = MultiOutputGP.load(path, kernel="rbf")

    assert loaded._specialization_request.mode == "disabled"
    if loaded._specialization_decision is not None:
        assert loaded._specialization_decision.mode == "disabled"
        assert loaded._specialization_decision.profile.specialization_key == "default"

    mean, var = loaded.predict(X[:8], return_var=True)
    assert mean.shape == (8, 2)
    assert var.shape == (8, 2)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
