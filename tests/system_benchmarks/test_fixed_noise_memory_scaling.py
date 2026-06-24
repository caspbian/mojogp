"""Memory-scaling checks for matrix-free fixed observation noise."""

from __future__ import annotations

import numpy as np
import pytest

from mojogp import RBF, SingleOutputGP as ExactGP
from tests.shared.benchmarking.gpu_memory import measure_gpu_phase
from .conftest import assert_gpu_available, requires_cuda


def _known_noise_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 1)).astype(np.float32)
    noise = (0.01 + 0.04 * (X[:, 0] > 0.0)).astype(np.float32)
    latent = np.sin(2.0 * X[:, 0]).astype(np.float32)
    y = (latent + rng.normal(scale=np.sqrt(noise))).astype(np.float32)
    return X, y, noise


def _fit_matrix_free_fixed_noise(n: int, seed: int):
    X, y, noise = _known_noise_data(n=n, seed=seed)
    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        observation_noise=noise,
        learn_noise=False,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.02,
        num_probes=2,
        max_cg_iterations=30,
        max_tridiag_iterations=6,
        preconditioner_rank=10,
        verbose=False,
    )
    assert gp.backend_train_info["noise_mode"] == "fixed_vector"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    return gp


def _fit_matrix_free_learned_vector_noise(n: int, seed: int):
    X, y, _ = _known_noise_data(n=n, seed=seed)
    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="learned_vector",
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.05,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.02,
        num_probes=2,
        max_cg_iterations=30,
        max_tridiag_iterations=6,
        preconditioner_rank=10,
        verbose=False,
    )
    assert gp.backend_train_info["noise_mode"] == "learned_vector"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    return gp


def _fit_matrix_free_learned_input_dependent_noise(n: int, seed: int):
    X, y, _ = _known_noise_data(n=n, seed=seed)
    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="learned_input_dependent",
        noise_function="linear",
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.05,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.02,
        num_probes=2,
        max_cg_iterations=30,
        max_tridiag_iterations=6,
        preconditioner_rank=10,
        verbose=False,
    )
    assert gp.backend_train_info["noise_mode"] == "learned_input_dependent"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    return gp


@pytest.mark.system
@pytest.mark.single_output
@requires_cuda
def test_matrix_free_fixed_noise_memory_scales_subquadratically():
    assert_gpu_available()

    _, small_mem = measure_gpu_phase(
        lambda: _fit_matrix_free_fixed_noise(2000, 901), interval=0.02
    )
    _, large_mem = measure_gpu_phase(
        lambda: _fit_matrix_free_fixed_noise(3000, 907), interval=0.02
    )

    small_peak = float(small_mem["phase_peak_gpu_mb"])
    large_peak = float(large_mem["phase_peak_gpu_mb"])

    assert small_mem["method"] != "none"
    assert large_mem["method"] != "none"
    assert small_peak > 0.0
    assert large_peak > 0.0
    # 3000/2000 = 1.5. A hidden n*n train matrix would tend toward 2.25x
    # memory growth; absolute monitored peaks are robust to CUDA allocator reuse.
    assert large_peak <= small_peak * 2.1 + 256.0


@pytest.mark.system
@pytest.mark.single_output
@requires_cuda
def test_matrix_free_learned_vector_noise_memory_scales_subquadratically():
    assert_gpu_available()

    _, small_mem = measure_gpu_phase(
        lambda: _fit_matrix_free_learned_vector_noise(2000, 911), interval=0.02
    )
    _, large_mem = measure_gpu_phase(
        lambda: _fit_matrix_free_learned_vector_noise(3000, 917), interval=0.02
    )

    small_peak = float(small_mem["phase_peak_gpu_mb"])
    large_peak = float(large_mem["phase_peak_gpu_mb"])

    assert small_mem["method"] != "none"
    assert large_mem["method"] != "none"
    assert small_peak > 0.0
    assert large_peak > 0.0
    assert large_peak <= small_peak * 2.1 + 256.0


@pytest.mark.system
@pytest.mark.single_output
@requires_cuda
def test_matrix_free_learned_input_dependent_noise_memory_scales_subquadratically():
    assert_gpu_available()

    _, small_mem = measure_gpu_phase(
        lambda: _fit_matrix_free_learned_input_dependent_noise(2000, 921), interval=0.02
    )
    _, large_mem = measure_gpu_phase(
        lambda: _fit_matrix_free_learned_input_dependent_noise(3000, 927), interval=0.02
    )

    small_peak = float(small_mem["phase_peak_gpu_mb"])
    large_peak = float(large_mem["phase_peak_gpu_mb"])

    assert small_mem["method"] != "none"
    assert large_mem["method"] != "none"
    assert small_peak > 0.0
    assert large_peak > 0.0
    assert large_peak <= small_peak * 2.1 + 256.0
