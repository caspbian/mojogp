from __future__ import annotations

import numpy as np

from tests.shared.benchmarking import data_generators
from tests.shared.benchmarking.data_generators import (
    _multi_output_structured_ard_latents,
    _single_output_structured_ard_signal,
    generate_multi_output_structured_ard_data,
    generate_multi_output_structured_per_task_noise_data,
    generate_single_output_structured_ard_data,
)


def test_single_output_structured_ard_data_is_deterministic_and_shaped():
    first = generate_single_output_structured_ard_data(
        n_train=25,
        n_test=7,
        d=9,
        relevant_dims=3,
        seed=123,
    )
    second = generate_single_output_structured_ard_data(
        n_train=25,
        n_test=7,
        d=9,
        relevant_dims=3,
        seed=123,
    )

    assert first.X_train.shape == (25, 9)
    assert first.y_train.shape == (25,)
    assert first.X_test.shape == (7, 9)
    assert first.f_test.shape == (7,)
    np.testing.assert_allclose(first.X_train, second.X_train)
    np.testing.assert_allclose(first.y_train, second.y_train)
    assert first.true_params["dataset_family"] == "structured_ard"
    assert first.true_params["relevant_indices"] == [0, 1, 2]
    assert first.true_params["irrelevant_indices"] == [3, 4, 5, 6, 7, 8]


def test_single_output_structured_ard_signal_ignores_irrelevant_columns():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 9)).astype(np.float32)
    changed = X.copy()
    changed[:, 3:] = rng.normal(size=changed[:, 3:].shape).astype(np.float32)

    f = _single_output_structured_ard_signal(X, relevant_dims=3)
    changed_f = _single_output_structured_ard_signal(changed, relevant_dims=3)

    assert np.std(X[:, 3:]) > 0.0
    assert np.std(changed[:, 3:]) > 0.0
    np.testing.assert_allclose(f, changed_f)


def test_multi_output_structured_ard_data_is_deterministic_and_shaped():
    first = generate_multi_output_structured_ard_data(
        n_train=30,
        n_test=8,
        d=17,
        num_tasks=4,
        relevant_dims=5,
        seed=321,
    )
    second = generate_multi_output_structured_ard_data(
        n_train=30,
        n_test=8,
        d=17,
        num_tasks=4,
        relevant_dims=5,
        seed=321,
    )

    assert first.X_train.shape == (30, 17)
    assert first.Y_train.shape == (30, 4)
    assert first.X_test.shape == (8, 17)
    assert first.F_test.shape == (8, 4)
    np.testing.assert_allclose(first.X_train, second.X_train)
    np.testing.assert_allclose(first.Y_train, second.Y_train)
    assert first.true_params["dataset_family"] == "structured_ard"
    assert first.true_params["relevant_indices"] == [0, 1, 2, 3, 4]
    assert first.true_params["irrelevant_indices"] == list(range(5, 17))


def test_multi_output_structured_ard_latents_ignore_irrelevant_columns():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(40, 17)).astype(np.float32)
    changed = X.copy()
    changed[:, 5:] = rng.normal(size=changed[:, 5:].shape).astype(np.float32)

    latent_a, latent_b = _multi_output_structured_ard_latents(X, relevant_dims=5)
    changed_a, changed_b = _multi_output_structured_ard_latents(
        changed, relevant_dims=5
    )

    np.testing.assert_allclose(latent_a, changed_a)
    np.testing.assert_allclose(latent_b, changed_b)


def test_structured_ard_generators_do_not_use_dense_kernel_cdist(monkeypatch):
    def fail_cdist(*_args, **_kwargs):
        raise AssertionError("structured ARD generators must not use dense cdist")

    monkeypatch.setattr(data_generators, "cdist", fail_cdist)

    generate_single_output_structured_ard_data(
        n_train=20,
        n_test=5,
        d=9,
        relevant_dims=3,
        seed=1,
    )
    generate_multi_output_structured_ard_data(
        n_train=20,
        n_test=5,
        d=9,
        num_tasks=3,
        relevant_dims=3,
        seed=1,
    )


def test_multi_output_structured_per_task_noise_is_deterministic_and_dense_free(monkeypatch):
    def fail_cdist(*_args, **_kwargs):
        raise AssertionError("structured per-task-noise generator must not use dense cdist")

    monkeypatch.setattr(data_generators, "cdist", fail_cdist)

    kwargs = {
        "n_train": 24,
        "n_test": 6,
        "d": 5,
        "num_tasks": 3,
        "noise_per_task": np.array([0.01, 0.05, 0.12], dtype=np.float32),
        "mean_per_task": np.array([-0.5, 0.0, 0.5], dtype=np.float32),
        "task_correlation": "medium",
        "seed": 11,
    }
    first = generate_multi_output_structured_per_task_noise_data(**kwargs)
    second = generate_multi_output_structured_per_task_noise_data(**kwargs)

    assert first.X_train.shape == (24, 5)
    assert first.Y_train.shape == (24, 3)
    assert first.X_test.shape == (6, 5)
    assert first.F_test.shape == (6, 3)
    np.testing.assert_allclose(first.X_train, second.X_train)
    np.testing.assert_allclose(first.Y_train, second.Y_train)
    assert first.true_params["dataset_family"] == "structured_per_task_noise"
    np.testing.assert_allclose(first.true_params["noise_per_task"], kwargs["noise_per_task"])
