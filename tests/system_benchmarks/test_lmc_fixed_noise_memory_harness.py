"""Matrix-free LMC fixed-noise memory and route checks."""

from __future__ import annotations

import numpy as np
import pytest

from mojogp import Kernel, MultiOutputLMCGP
from tests.shared.benchmarking.gpu_memory import measure_gpu_phase
from tests.system_benchmarks.conftest import assert_gpu_available, requires_cuda


def _fixed_noise_dataset(n_train: int, n_test: int = 8, seed: int = 61):
    rng = np.random.default_rng(seed + n_train)
    X_train = rng.uniform(-2.0, 2.0, size=(n_train, 3)).astype(np.float32)
    X_test = rng.uniform(-2.0, 2.0, size=(n_test, 3)).astype(np.float32)
    f0 = np.sin(X_train[:, 0]) + 0.15 * X_train[:, 1]
    f1 = 0.6 * np.cos(X_train[:, 0]) - 0.2 * X_train[:, 2]
    fixed_noise = np.stack(
        [
            0.015 + 0.006 * (1.0 + X_train[:, 0] ** 2),
            0.035 + 0.004 * (1.0 + X_train[:, 1] ** 2),
        ],
        axis=1,
    ).astype(np.float32)
    Y_train = np.stack(
        [
            f0 + rng.normal(0.0, np.sqrt(fixed_noise[:, 0])),
            f1 + rng.normal(0.0, np.sqrt(fixed_noise[:, 1])),
        ],
        axis=1,
    ).astype(np.float32)
    return X_train, Y_train, X_test, fixed_noise


def _run_matrix_free_fixed_noise_case(n_train: int) -> dict[str, object]:
    X_train, Y_train, X_test, fixed_noise = _fixed_noise_dataset(n_train)
    gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf()],
        num_probes=3,
        max_cg_iterations=30,
        max_tridiag_iterations=10,
        use_preconditioner=False,
    )
    _, fit_memory = measure_gpu_phase(
        lambda: gp.fit(
            X_train,
            Y_train,
            method="matrix_free",
            max_iterations=3,
            learning_rate=0.02,
            fixed_observation_noise=fixed_noise,
            verbose=False,
        ),
        interval=0.02,
    )
    (mean, var), pred_memory = measure_gpu_phase(
        lambda: gp.predict(X_test, return_var=True, variance_method="exact"),
        interval=0.02,
    )
    return {
        "mean": mean,
        "variance": var,
        "train_info": dict(gp.backend_train_info),
        "predict_info": dict(gp.backend_predict_info),
        "fit_delta_mb": float(fit_memory.get("phase_delta_gpu_mb", 0.0) or 0.0),
        "predict_delta_mb": float(pred_memory.get("phase_delta_gpu_mb", 0.0) or 0.0),
    }


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.memory
@requires_cuda
def test_lmc_matrix_free_fixed_noise_exact_variance_memory_contract():
    assert_gpu_available()
    # Exclude one-time JIT/module and GPU allocator startup growth from the
    # memory-scaling assertion. The contract under test is the steady-state
    # matrix-free route, not first-use library loading.
    _run_matrix_free_fixed_noise_case(2000)
    small = _run_matrix_free_fixed_noise_case(2000)
    large = _run_matrix_free_fixed_noise_case(2600)

    for result in (small, large):
        assert result["train_info"]["training_route"] == "matrix_free"
        assert result["train_info"]["uses_fixed_observation_noise"] is True
        assert result["predict_info"]["variance_method"] == "exact"
        assert result["predict_info"]["actual_variance_route"] == "predict_lmc_full_exact"
        assert result["predict_info"]["backend_variance_used"] is True
        assert result["predict_info"].get("fallback_used") is False
        assert np.all(np.isfinite(result["mean"]))
        assert np.all(np.isfinite(result["variance"]))
        assert np.all(result["variance"] >= 0.0)

    # nvidia-smi deltas can be near-zero when the CUDA allocator reuses warmed
    # blocks. Use a floor large enough to test the memory-scaling contract
    # rather than allocator quantization.
    measurement_floor_mb = 128.0
    fit_growth = max(large["fit_delta_mb"], measurement_floor_mb) / max(
        small["fit_delta_mb"], measurement_floor_mb
    )
    pred_growth = max(large["predict_delta_mb"], measurement_floor_mb) / max(
        small["predict_delta_mb"], measurement_floor_mb
    )
    assert fit_growth < 2.0
    assert pred_growth < 2.0
