"""Dense exact references for small-n multi-output correctness checks."""

from __future__ import annotations

import numpy as np


def flatten_multi_output_targets(Y: np.ndarray) -> np.ndarray:
    return np.asarray(Y, dtype=np.float64).reshape(-1)


def build_lmc_train_covariance(
    X_train: np.ndarray,
    kernel_matrices: list[np.ndarray],
    A_matrices: np.ndarray,
    noise_per_task: np.ndarray,
    fixed_observation_noise: np.ndarray | None = None,
) -> np.ndarray:
    n = X_train.shape[0]
    num_tasks = A_matrices.shape[1]
    full = np.zeros((n * num_tasks, n * num_tasks), dtype=np.float64)
    for latent_idx, K_latent in enumerate(kernel_matrices):
        full += np.kron(
            K_latent.astype(np.float64), A_matrices[latent_idx].astype(np.float64)
        )
    full += np.kron(
        np.eye(n, dtype=np.float64), np.diag(noise_per_task.astype(np.float64))
    )
    if fixed_observation_noise is not None:
        fixed = np.asarray(fixed_observation_noise, dtype=np.float64)
        if fixed.shape != (n, num_tasks):
            raise ValueError(
                f"fixed_observation_noise must have shape ({n}, {num_tasks}), got {fixed.shape}"
            )
        full += np.diag(fixed.reshape(-1))
    return full


def build_lmc_cross_covariance(
    train_test_kernel_matrices: list[np.ndarray],
    A_matrices: np.ndarray,
) -> np.ndarray:
    blocks = []
    for latent_idx, K_cross in enumerate(train_test_kernel_matrices):
        blocks.append(
            np.kron(
                K_cross.astype(np.float64), A_matrices[latent_idx].astype(np.float64)
            )
        )
    return np.sum(blocks, axis=0)


def build_lmc_test_covariance(
    test_kernel_matrices: list[np.ndarray],
    A_matrices: np.ndarray,
) -> np.ndarray:
    blocks = []
    for latent_idx, K_test in enumerate(test_kernel_matrices):
        blocks.append(
            np.kron(
                K_test.astype(np.float64), A_matrices[latent_idx].astype(np.float64)
            )
        )
    return np.sum(blocks, axis=0)


def exact_gaussian_posterior(
    train_cov: np.ndarray,
    cross_cov: np.ndarray,
    test_cov: np.ndarray,
    y_train: np.ndarray,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = train_cov.shape[0]
    eye = np.eye(n_train, dtype=np.float64)
    chol = np.linalg.cholesky(train_cov + jitter * eye)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train.astype(np.float64)))
    mean = cross_cov.T @ alpha
    v = np.linalg.solve(chol, cross_cov)
    cov = test_cov - v.T @ v
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def unflatten_multi_output_predictions(vec: np.ndarray, num_tasks: int) -> np.ndarray:
    return np.asarray(vec, dtype=np.float64).reshape(-1, num_tasks)


def diagonal_task_variances(cov: np.ndarray, num_tasks: int) -> np.ndarray:
    diag = np.diag(cov).astype(np.float64)
    return diag.reshape(-1, num_tasks)
