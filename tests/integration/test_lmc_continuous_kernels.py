"""Continuous-kernel route checks for MultiOutputLMCGP."""

import numpy as np
import pytest

from mojogp import Kernel
from mojogp.multi_output_gp import MultiOutputLMCGP


pytestmark = pytest.mark.integration


def _continuous_lmc_data(n: int = 2000, d: int = 3, seed: int = 456):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    f = np.sin(1.5 * X[:, 0]) + 0.25 * X[:, 1] ** 2 - 0.1 * X[:, 2]
    Y = np.stack(
        [
            f + 0.05 * rng.normal(size=n),
            -0.7 * f + 0.05 * rng.normal(size=n),
        ],
        axis=1,
    ).astype(np.float32)
    return X, Y


@pytest.mark.parametrize(
    "kernel_name,kernel",
    [
        ("rq", Kernel.rq()),
        ("periodic", Kernel.periodic()),
        ("linear", Kernel.linear()),
        ("polynomial", Kernel.polynomial()),
    ],
)
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_lmc_continuous_kernel_prediction_and_persistence_routes(
    kernel_name, kernel, method, tmp_path
):
    """Continuous LMC kernels expose finite exact/LOVE route metadata after load."""
    X, Y = _continuous_lmc_data(seed=456 + len(kernel_name))
    X_test = X[:8].copy()

    gp = MultiOutputLMCGP(kernels=[kernel])
    result = gp.fit(
        X,
        Y,
        method=method,
        max_iterations=5,
        learning_rate=0.01,
        verbose=False,
        early_stop_tol=0.0,
        initial_noise_per_task=np.full(2, 0.1, dtype=np.float32),
    )

    assert np.isfinite(result.final_nll)

    expected_exact_route = (
        "dense_exact_lmc" if method == "materialized" else "predict_lmc_full_exact"
    )
    for variance_method, expected_route in (
        ("exact", expected_exact_route),
        ("love", "predict_lmc"),
    ):
        mean, var = gp.predict(X_test, return_var=True, variance_method=variance_method)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert gp._backend_predict_info["actual_variance_route"] == expected_route
        assert gp._backend_predict_info["lmc_variance_output"] == "observation"
        assert gp._backend_predict_info[
            "observation_variance_includes_learned_task_noise"
        ] is True
        if expected_route in ("dense_exact_lmc", "predict_lmc_full_exact"):
            assert (
                gp._backend_predict_info["lmc_variance_exactness"]
                == "exact_full_lmc_covariance"
            )
        else:
            assert (
                gp._backend_predict_info["lmc_variance_exactness"]
                == "scalar_latent_approximation"
            )
        if expected_route == "predict_lmc":
            assert gp._backend_predict_info["backend_variance_used"] is True

    model_path = str(tmp_path / f"lmc_{kernel_name}_{method}")
    gp.save(model_path)
    loaded = MultiOutputLMCGP.load(model_path)

    for variance_method in ("exact", "love"):
        mean, var = loaded.predict(X_test, return_var=True, variance_method=variance_method)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert loaded._backend_predict_info["lmc_variance_output"] == "observation"
        assert "lmc_variance_exactness" in loaded._backend_predict_info
