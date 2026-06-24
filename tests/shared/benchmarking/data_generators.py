"""Synthetic data generators for system benchmarks.

All generators are deterministic (seeded), return numpy float32 arrays,
and include ground truth for measuring prediction accuracy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from scipy.spatial.distance import cdist


@dataclass
class SyntheticDataset:
    """Container for synthetic GP data."""

    X_train: np.ndarray  # [n_train, d]
    y_train: np.ndarray  # [n_train]
    X_test: np.ndarray  # [n_test, d]
    y_test: np.ndarray  # [n_test] (noisy)
    f_test: np.ndarray  # [n_test] (noiseless ground truth)
    true_params: Dict[str, Any]  # {lengthscale, noise, outputscale, ...}
    name: str  # Human-readable name
    description: str  # What makes this dataset interesting


@dataclass
class MultiOutputDataset:
    """Container for multi-output synthetic data."""

    X_train: np.ndarray  # [n_train, d]
    Y_train: np.ndarray  # [n_train, T]
    X_test: np.ndarray  # [n_test, d]
    Y_test: np.ndarray  # [n_test, T] (noisy)
    F_test: np.ndarray  # [n_test, T] (noiseless)
    true_params: Dict[str, Any]  # {lengthscale, noise, outputscale, B, ...}
    name: str
    description: str


@dataclass
class MixedSyntheticDataset:
    """Container for mixed continuous + categorical synthetic data."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    f_test: np.ndarray
    true_params: Dict[str, Any]
    name: str
    description: str


@dataclass
class MultiOutputHeterogeneousDataset:
    """Container for heterogeneous-latent multi-output synthetic data."""

    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    F_test: np.ndarray
    true_params: Dict[str, Any]
    name: str
    description: str


# =============================================================================
# Kernel Matrix Computation (for GP prior sampling)
# =============================================================================


def compute_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel_type: str,
    lengthscale: float = 1.0,
    outputscale: float = 1.0,
    period: float = 1.0,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute kernel matrix K(X1, X2).

    Supports: rbf, matern12, matern32, matern52, periodic, rq, linear, polynomial
    """
    # Compute pairwise distances
    if kernel_type in ["rbf", "matern12", "matern32", "matern52", "rq"]:
        dists = cdist(X1 / lengthscale, X2 / lengthscale, metric="euclidean")

    if kernel_type == "rbf":
        K = outputscale * np.exp(-0.5 * dists**2)

    elif kernel_type == "matern12":
        K = outputscale * np.exp(-dists)

    elif kernel_type == "matern32":
        sqrt3 = np.sqrt(3)
        K = outputscale * (1 + sqrt3 * dists) * np.exp(-sqrt3 * dists)

    elif kernel_type == "matern52":
        sqrt5 = np.sqrt(5)
        K = (
            outputscale
            * (1 + sqrt5 * dists + 5 / 3 * dists**2)
            * np.exp(-sqrt5 * dists)
        )

    elif kernel_type == "periodic":
        # Periodic kernel: product of 1D periodic kernels across dimensions
        # k(x,x') = os * prod_i exp(-2 * sin^2(pi * |x_i - x'_i| / period) / ls^2)
        # This ensures the kernel is positive definite (Euclidean distance version is NOT PSD for d>1)
        d = X1.shape[1]
        K = np.ones((X1.shape[0], X2.shape[0]), dtype=np.float64)
        for dim_idx in range(d):
            dists_1d = cdist(
                X1[:, dim_idx : dim_idx + 1],
                X2[:, dim_idx : dim_idx + 1],
                metric="euclidean",
            )
            sin_term = np.sin(np.pi * dists_1d / period)
            K_dim = np.exp(-2 * sin_term**2 / (lengthscale**2))
            K = K * K_dim
        K = outputscale * K

    elif kernel_type == "rq":
        # Rational Quadratic: k(x,x') = os * (1 + |x-x'|^2 / (2*alpha*ls^2))^(-alpha)
        K = outputscale * (1 + dists**2 / (2 * alpha)) ** (-alpha)

    elif kernel_type == "linear":
        # Linear kernel: k(x,x') = os * x^T x'
        K = outputscale * (X1 @ X2.T)

    elif kernel_type == "polynomial":
        # Polynomial kernel: k(x,x') = os * (x_norm^T x_norm' + offset)^degree
        # Normalize X to avoid numerical issues with large polynomial values
        degree = 2
        offset = 1.0
        X1_norm = X1 / (np.linalg.norm(X1, axis=1, keepdims=True) + 1e-8)
        X2_norm = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-8)
        K = outputscale * (X1_norm @ X2_norm.T + offset) ** degree

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return K.astype(np.float32)


# =============================================================================
# GP Prior Data Generation
# =============================================================================


def generate_gp_prior_data(
    n_train: int,
    n_test: int,
    d: int,
    kernel_type: str,
    true_lengthscale: float = 1.0,
    true_noise: float = 0.1,
    true_outputscale: float = 1.0,
    seed: int = 42,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    true_period: float = 1.0,
    true_alpha: float = 1.0,
    mean_offset: float = 0.0,
) -> SyntheticDataset:
    """Generate data from a GP prior with known hyperparameters.

    This is the most principled data generator because:
    - The data IS from a GP, so the GP model is correctly specified
    - We know the true hyperparameters, so we can measure recovery
    - We know the true function values, so we can measure prediction accuracy
    """
    np.random.seed(seed)

    n_total = n_train + n_test

    # Generate X uniformly
    X = np.random.uniform(x_range[0], x_range[1], size=(n_total, d)).astype(np.float32)

    # Compute kernel matrix
    K = compute_kernel_matrix(
        X,
        X,
        kernel_type,
        lengthscale=true_lengthscale,
        outputscale=true_outputscale,
        period=true_period,
        alpha=true_alpha,
    )

    # Add jitter for numerical stability with adaptive retry
    # Some kernels (especially periodic) can produce ill-conditioned matrices
    # that need more jitter for Cholesky to succeed
    jitter = 1e-6
    L = None
    for attempt in range(5):
        try:
            K_jittered = K + jitter * np.eye(n_total)
            L = np.linalg.cholesky(K_jittered)
            break
        except np.linalg.LinAlgError:
            jitter *= 10  # Increase jitter: 1e-6 -> 1e-5 -> 1e-4 -> 1e-3 -> 1e-2

    if L is None:
        raise ValueError(
            f"Cholesky decomposition failed for {kernel_type} kernel even with "
            f"jitter={jitter}. The kernel matrix may be too ill-conditioned."
        )

    # Sample f ~ N(mean_offset, K) via Cholesky.
    f = L @ np.random.randn(n_total).astype(np.float32)
    f = (f + np.float32(mean_offset)).astype(np.float32)

    # Add observation noise with variance true_noise.
    y = f + np.sqrt(true_noise) * np.random.randn(n_total).astype(np.float32)

    # Split into train/test
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    f_test = f[n_train:]

    return SyntheticDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        f_test=f_test,
        true_params={
            "lengthscale": true_lengthscale,
            "noise": true_noise,
            "outputscale": true_outputscale,
            "mean": float(mean_offset),
            "period": true_period,
            "alpha": true_alpha,
        },
        name=f"gp_prior_{kernel_type}_n{n_train}_d{d}",
        description=(
            f"GP prior sample with {kernel_type} kernel and mean {mean_offset}"
        ),
    )


# =============================================================================
# Structured Function Data Generation
# =============================================================================

NOISE_LEVELS = {
    "low": 0.01,
    "medium": 0.1,
    "high": 0.5,
    "very_high": 1.0,
}


def generate_structured_function_data(
    n_train: int,
    n_test: int,
    d: int,
    function_type: str,
    noise_level: str = "medium",
    seed: int = 42,
    mean_offset: float = 0.0,
) -> SyntheticDataset:
    """Generate data from known analytic functions.

    Function types:
    - 'smooth': sin(x) + 0.5*cos(2x) -- good for RBF, Matern52
    - 'oscillatory': sin(5x) + sin(10x) -- tests high-frequency recovery
    - 'step': tanh(10x) -- discontinuous, hard for smooth kernels
    - 'polynomial': x^3 - 2x^2 + x -- good for polynomial kernels
    - 'linear': weighted sum of features -- good for linear kernels
    - 'friedman1': 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4
    - 'periodic_signal': sin(2*pi*x/period) -- good for periodic kernel
    """
    np.random.seed(seed)

    n_total = n_train + n_test
    noise_std = NOISE_LEVELS.get(noise_level, 0.1)

    # Generate X
    if function_type == "periodic_signal":
        X = np.random.uniform(0, 4, size=(n_total, d)).astype(np.float32)
    elif function_type == "friedman1":
        X = np.random.uniform(0, 1, size=(n_total, max(d, 5))).astype(np.float32)
        d = max(d, 5)
    else:
        X = np.random.uniform(-3, 3, size=(n_total, d)).astype(np.float32)

    # Compute function values
    if function_type == "smooth":
        if d == 1:
            f = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 0])
        else:
            f = np.sum(np.sin(X) + 0.5 * np.cos(2 * X), axis=1) / d

    elif function_type == "oscillatory":
        if d == 1:
            f = np.sin(5 * X[:, 0]) + np.sin(10 * X[:, 0])
        else:
            f = np.sum(np.sin(5 * X) + np.sin(10 * X), axis=1) / d

    elif function_type == "step":
        if d == 1:
            f = np.tanh(10 * X[:, 0])
        else:
            f = np.tanh(10 * np.linalg.norm(X, axis=1))

    elif function_type == "polynomial":
        if d == 1:
            f = X[:, 0] ** 3 - 2 * X[:, 0] ** 2 + X[:, 0]
        else:
            f = np.sum(X**3 - 2 * X**2 + X, axis=1) / d

    elif function_type == "linear":
        # Linear function: weighted sum of features
        # Use fixed weights for reproducibility
        weights = np.array(
            [2.0, -1.5, 1.0, -0.5, 0.3] + [0.0] * (d - 5), dtype=np.float32
        )[:d]
        f = X @ weights

    elif function_type == "friedman1":
        # Friedman #1: only first 5 dimensions matter
        f = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )

    elif function_type == "periodic_signal":
        period = 1.0
        if d == 1:
            f = np.sin(2 * np.pi * X[:, 0] / period)
        else:
            f = np.sum(np.sin(2 * np.pi * X / period), axis=1) / d

    else:
        raise ValueError(f"Unknown function type: {function_type}")

    f = (f + np.float32(mean_offset)).astype(np.float32)

    # Scale noise relative to function std
    actual_noise_std = noise_std * np.std(f)
    actual_noise_var = float(actual_noise_std**2)
    y = f + actual_noise_std * np.random.randn(n_total).astype(np.float32)

    # Split
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    f_test = f[n_train:]

    return SyntheticDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        f_test=f_test,
        true_params={
            "noise": actual_noise_var,
            "mean": float(mean_offset),
            "function_type": function_type,
            "noise_level": noise_level,
        },
        name=f"{function_type}_n{n_train}_d{d}_{noise_level}",
        description=f"{function_type} function with {noise_level} noise",
    )


# =============================================================================
# Multi-Output Data Generation
# =============================================================================


def generate_multi_output_data(
    n_train: int,
    n_test: int,
    d: int,
    num_tasks: int,
    kernel_type: str,
    true_lengthscale: float = 1.0,
    true_noise: float = 0.1,
    true_outputscale: float = 1.0,
    task_correlation: str = "medium",
    seed: int = 42,
) -> MultiOutputDataset:
    """Generate multi-output data with known task covariance.

    Creates B = WW^T + diag(v) with controlled correlation level.
    Generates Y from Kronecker GP: vec(Y) ~ N(0, K_X kron B + noise*I).
    """
    np.random.seed(seed)

    # Task correlation levels
    correlation_levels = {
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
        "independent": 0.0,
    }
    corr = correlation_levels.get(task_correlation, 0.5)

    n_total = n_train + n_test

    # Generate X
    X = np.random.uniform(-3, 3, size=(n_total, d)).astype(np.float32)

    # Create task covariance B = WW^T + diag(v)
    # W is T x R where R = T for full rank
    R = num_tasks
    W = np.random.randn(num_tasks, R).astype(np.float32) * np.sqrt(corr / R)
    v = np.ones(num_tasks).astype(np.float32) * (1 - corr)
    B = W @ W.T + np.diag(v)

    # Compute kernel matrix K_X
    K_X = compute_kernel_matrix(
        X,
        X,
        kernel_type,
        lengthscale=true_lengthscale,
        outputscale=true_outputscale,
    )
    K_X += 1e-6 * np.eye(n_total)

    # Full covariance: K_full = K_X kron B
    # For efficiency, sample each task from its marginal
    L_X = np.linalg.cholesky(K_X)
    L_B = np.linalg.cholesky(B)

    # Sample F: each column is a task
    Z = np.random.randn(n_total, num_tasks).astype(np.float32)
    F = (L_X @ Z) @ L_B.T

    # Add observation noise with variance true_noise.
    Y = F + np.sqrt(true_noise) * np.random.randn(n_total, num_tasks).astype(
        np.float32
    )

    # Split
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    F_test = F[n_train:]

    return MultiOutputDataset(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        F_test=F_test,
        true_params={
            "lengthscale": true_lengthscale,
            "noise": true_noise,
            "noise_per_task": np.full(num_tasks, true_noise, dtype=np.float32),
            "outputscale": true_outputscale,
            "B": B,
            "W": W,
            "v": v,
            "task_correlation": task_correlation,
        },
        name=f"multi_output_{kernel_type}_n{n_train}_T{num_tasks}_{task_correlation}",
        description=f"Multi-output GP with {num_tasks} tasks, {task_correlation} correlation",
    )


def generate_multi_output_per_task_noise_data(
    n_train: int,
    n_test: int,
    d: int,
    num_tasks: int,
    kernel_type: str,
    noise_per_task: np.ndarray,
    true_lengthscale: float = 1.0,
    true_outputscale: float = 1.0,
    task_correlation: str = "medium",
    mean_per_task: Optional[np.ndarray] = None,
    seed: int = 42,
) -> MultiOutputDataset:
    """Generate multi-output data with explicit per-task observation noise.

    This is the active JIT-era replacement for the legacy per-task noise system
    generator. It keeps the same Kronecker-style latent signal structure as
    ``generate_multi_output_data`` but adds distinct observation variances per
    task and optional per-task mean offsets.
    """
    np.random.seed(seed)

    correlation_levels = {
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
        "independent": 0.0,
    }
    corr = correlation_levels.get(task_correlation, 0.5)

    noise_per_task = np.ascontiguousarray(noise_per_task, dtype=np.float32)
    if noise_per_task.shape != (num_tasks,):
        raise ValueError(
            f"noise_per_task must have shape ({num_tasks},), got {noise_per_task.shape}"
        )
    if np.any(noise_per_task <= 0):
        raise ValueError("noise_per_task must be strictly positive")

    if mean_per_task is None:
        mean_per_task = np.zeros(num_tasks, dtype=np.float32)
    else:
        mean_per_task = np.ascontiguousarray(mean_per_task, dtype=np.float32)
        if mean_per_task.shape != (num_tasks,):
            raise ValueError(
                f"mean_per_task must have shape ({num_tasks},), got {mean_per_task.shape}"
            )

    n_total = n_train + n_test
    X = np.random.randn(n_total, d).astype(np.float32)

    R = num_tasks
    W = np.random.randn(num_tasks, R).astype(np.float32) * np.sqrt(corr / max(R, 1))
    v = np.ones(num_tasks, dtype=np.float32) * (1 - corr)
    B = W @ W.T + np.diag(v)

    K_X = compute_kernel_matrix(
        X,
        X,
        kernel_type,
        lengthscale=true_lengthscale,
        outputscale=true_outputscale,
    )
    K_X += 1e-6 * np.eye(n_total, dtype=np.float32)

    L_X = np.linalg.cholesky(K_X)
    L_B = np.linalg.cholesky(B)
    Z = np.random.randn(n_total, num_tasks).astype(np.float32)
    F = (L_X @ Z) @ L_B.T
    F = F + mean_per_task[np.newaxis, :]

    Y = F.copy()
    for task_idx in range(num_tasks):
        Y[:, task_idx] += np.sqrt(noise_per_task[task_idx]) * np.random.randn(
            n_total
        ).astype(np.float32)

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    F_test = F[n_train:]

    return MultiOutputDataset(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        F_test=F_test,
        true_params={
            "lengthscale": true_lengthscale,
            "outputscale": true_outputscale,
            "noise_per_task": noise_per_task,
            "mean_per_task": mean_per_task,
            "B": B,
            "W": W,
            "v": v,
            "task_correlation": task_correlation,
        },
        name=(
            f"multi_output_per_task_noise_{kernel_type}_n{n_train}_T{num_tasks}_{task_correlation}"
        ),
        description=(
            f"Multi-output GP with {num_tasks} tasks, {task_correlation} correlation, "
            "and explicit per-task observation noise"
        ),
    )


# =============================================================================
# ARD Data Generation
# =============================================================================


def generate_ard_data(
    n_train: int,
    n_test: int,
    d: int,
    relevant_dims: int,
    true_noise: float = 0.1,
    true_outputscale: float = 1.0,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate data where only some dimensions are relevant.

    Irrelevant dimensions have large true lengthscales (low relevance).
    Tests whether ARD can identify the relevant dimensions.
    """
    np.random.seed(seed)

    n_total = n_train + n_test

    # Generate X
    X = np.random.uniform(-3, 3, size=(n_total, d)).astype(np.float32)

    # True lengthscales: small for relevant dims, large for irrelevant
    true_lengthscales = np.ones(d).astype(np.float32)
    true_lengthscales[:relevant_dims] = 0.5  # Relevant dims: short lengthscale
    true_lengthscales[relevant_dims:] = 10.0  # Irrelevant dims: long lengthscale

    # Compute ARD RBF kernel
    X_scaled = X / true_lengthscales
    dists = cdist(X_scaled, X_scaled, metric="euclidean")
    K = true_outputscale * np.exp(-0.5 * dists**2)
    K += 1e-6 * np.eye(n_total)

    # Sample f
    L = np.linalg.cholesky(K)
    f = L @ np.random.randn(n_total).astype(np.float32)

    # Add observation noise with variance true_noise.
    y = f + np.sqrt(true_noise) * np.random.randn(n_total).astype(np.float32)

    # Split
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    f_test = f[n_train:]

    return SyntheticDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        f_test=f_test,
        true_params={
            "lengthscales": true_lengthscales,
            "noise": true_noise,
            "outputscale": true_outputscale,
            "relevant_dims": relevant_dims,
        },
        name=f"ard_n{n_train}_d{d}_rel{relevant_dims}",
        description=f"ARD data with {relevant_dims}/{d} relevant dimensions",
    )


def _validate_structured_ard_dims(d: int, relevant_dims: int) -> None:
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if relevant_dims <= 0:
        raise ValueError(f"relevant_dims must be positive, got {relevant_dims}")
    if relevant_dims > d:
        raise ValueError(
            f"relevant_dims must be <= d, got relevant_dims={relevant_dims}, d={d}"
        )


def _single_output_structured_ard_signal(
    X: np.ndarray,
    relevant_dims: int,
    *,
    mean_offset: float = 0.0,
) -> np.ndarray:
    """O(n*d) structured ARD signal that only depends on relevant columns."""

    _validate_structured_ard_dims(int(X.shape[1]), relevant_dims)
    f = np.full(X.shape[0], float(mean_offset), dtype=np.float32)
    weights = np.linspace(1.15, 0.65, relevant_dims, dtype=np.float32)
    freqs = np.linspace(0.75, 2.35, relevant_dims, dtype=np.float32)
    for dim_idx in range(relevant_dims):
        x = X[:, dim_idx]
        f += weights[dim_idx] * (
            np.sin(freqs[dim_idx] * x)
            + np.float32(0.35) * np.cos(np.float32(0.5) * freqs[dim_idx] * x)
        ).astype(np.float32)
    f = (f / np.sqrt(np.float32(relevant_dims))).astype(np.float32)
    return f


def generate_single_output_structured_ard_data(
    n_train: int,
    n_test: int,
    d: int,
    relevant_dims: int,
    noise_level: str = "medium",
    seed: int = 42,
    mean_offset: float = 0.0,
    x_range: Tuple[float, float] = (-3.0, 3.0),
) -> SyntheticDataset:
    """Generate scalable single-output ARD data from a structured signal.

    The generator is O(n*d): all dimensions vary, but only the first
    ``relevant_dims`` columns affect the noiseless target. This is intended for
    large-n benchmark scaling rows where dense GP-prior sampling would make data
    generation itself the bottleneck.
    """

    _validate_structured_ard_dims(d, relevant_dims)
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.uniform(x_range[0], x_range[1], size=(n_total, d)).astype(np.float32)
    f = _single_output_structured_ard_signal(
        X,
        relevant_dims,
        mean_offset=mean_offset,
    )
    noise_std = np.float32(NOISE_LEVELS.get(noise_level, 0.1)) * np.float32(
        max(float(np.std(f)), 1e-6)
    )
    noise_var = float(noise_std**2)
    y = (f + noise_std * rng.standard_normal(n_total).astype(np.float32)).astype(
        np.float32
    )

    relevant_indices = list(range(relevant_dims))
    irrelevant_indices = list(range(relevant_dims, d))
    return SyntheticDataset(
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
        f_test=f[n_train:],
        true_params={
            "dataset_family": "structured_ard",
            "signal_family": "single_output_structured_ard",
            "relevant_dims": int(relevant_dims),
            "relevant_indices": relevant_indices,
            "irrelevant_indices": irrelevant_indices,
            "noise": noise_var,
            "noise_std": float(noise_std),
            "noise_level": noise_level,
            "mean": float(mean_offset),
        },
        name=f"structured_ard_single_n{n_train}_d{d}_rel{relevant_dims}_{noise_level}",
        description=(
            f"Structured ARD signal with {relevant_dims}/{d} relevant dimensions "
            f"and {noise_level} noise"
        ),
    )


def _multi_output_structured_ard_latents(
    X: np.ndarray,
    relevant_dims: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two structured latent functions sharing the same relevant dimensions."""

    _validate_structured_ard_dims(int(X.shape[1]), relevant_dims)
    latent_a = np.zeros(X.shape[0], dtype=np.float32)
    latent_b = np.zeros(X.shape[0], dtype=np.float32)
    freqs_a = np.linspace(0.65, 1.95, relevant_dims, dtype=np.float32)
    freqs_b = np.linspace(1.10, 2.70, relevant_dims, dtype=np.float32)
    weights_a = np.linspace(1.10, 0.70, relevant_dims, dtype=np.float32)
    weights_b = np.linspace(0.75, 1.25, relevant_dims, dtype=np.float32)
    for dim_idx in range(relevant_dims):
        x = X[:, dim_idx]
        latent_a += weights_a[dim_idx] * (
            np.sin(freqs_a[dim_idx] * x)
            + np.float32(0.25) * np.cos(np.float32(0.5) * freqs_a[dim_idx] * x)
        ).astype(np.float32)
        latent_b += weights_b[dim_idx] * (
            np.cos(freqs_b[dim_idx] * x)
            + np.float32(0.30) * np.sin(np.float32(0.75) * freqs_b[dim_idx] * x)
        ).astype(np.float32)
    scale = np.sqrt(np.float32(relevant_dims))
    return (latent_a / scale).astype(np.float32), (latent_b / scale).astype(np.float32)


def generate_multi_output_structured_ard_data(
    n_train: int,
    n_test: int,
    d: int,
    num_tasks: int,
    relevant_dims: int,
    noise_per_task: Optional[np.ndarray] = None,
    mean_per_task: Optional[np.ndarray] = None,
    seed: int = 42,
    x_range: Tuple[float, float] = (-3.0, 3.0),
) -> MultiOutputDataset:
    """Generate scalable multi-output ARD data from shared structured latents."""

    _validate_structured_ard_dims(d, relevant_dims)
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.uniform(x_range[0], x_range[1], size=(n_total, d)).astype(np.float32)

    if noise_per_task is None:
        noise_per_task = np.linspace(0.035, 0.075, num_tasks, dtype=np.float32)
    else:
        noise_per_task = np.ascontiguousarray(noise_per_task, dtype=np.float32)
    if noise_per_task.shape != (num_tasks,):
        raise ValueError(
            f"noise_per_task must have shape ({num_tasks},), got {noise_per_task.shape}"
        )
    if np.any(noise_per_task <= 0):
        raise ValueError("noise_per_task must be strictly positive")

    if mean_per_task is None:
        mean_per_task = np.linspace(-0.15, 0.15, num_tasks, dtype=np.float32)
    else:
        mean_per_task = np.ascontiguousarray(mean_per_task, dtype=np.float32)
    if mean_per_task.shape != (num_tasks,):
        raise ValueError(
            f"mean_per_task must have shape ({num_tasks},), got {mean_per_task.shape}"
        )

    latent_a, latent_b = _multi_output_structured_ard_latents(X, relevant_dims)
    phase = np.linspace(0.0, np.pi, num_tasks, dtype=np.float32)
    W = np.stack(
        [
            np.float32(0.85) + np.float32(0.25) * np.cos(phase),
            np.float32(0.65) * np.sin(phase + np.float32(0.35))
            + np.float32(0.20),
        ],
        axis=1,
    ).astype(np.float32)
    F = (
        W[np.newaxis, :, 0] * latent_a[:, np.newaxis]
        + W[np.newaxis, :, 1] * latent_b[:, np.newaxis]
        + mean_per_task[np.newaxis, :]
    ).astype(np.float32)
    Y = (
        F
        + rng.standard_normal((n_total, num_tasks)).astype(np.float32)
        * np.sqrt(noise_per_task)[np.newaxis, :]
    ).astype(np.float32)

    relevant_indices = list(range(relevant_dims))
    irrelevant_indices = list(range(relevant_dims, d))
    return MultiOutputDataset(
        X_train=X[:n_train],
        Y_train=Y[:n_train],
        X_test=X[n_train:],
        Y_test=Y[n_train:],
        F_test=F[n_train:],
        true_params={
            "dataset_family": "structured_ard",
            "signal_family": "multi_output_structured_ard",
            "relevant_dims": int(relevant_dims),
            "relevant_indices": relevant_indices,
            "irrelevant_indices": irrelevant_indices,
            "noise_per_task": noise_per_task,
            "mean_per_task": mean_per_task,
            "W": W,
            "task_correlation": "structured_latent",
        },
        name=(
            f"structured_ard_multi_n{n_train}_d{d}_t{num_tasks}_rel{relevant_dims}"
        ),
        description=(
            f"Structured multi-output ARD signal with {relevant_dims}/{d} relevant "
            f"dimensions and {num_tasks} tasks"
        ),
    )


def generate_multi_output_structured_per_task_noise_data(
    n_train: int,
    n_test: int,
    d: int,
    num_tasks: int,
    noise_per_task: np.ndarray,
    task_correlation: str = "medium",
    mean_per_task: Optional[np.ndarray] = None,
    seed: int = 42,
    x_range: Tuple[float, float] = (-3.0, 3.0),
) -> MultiOutputDataset:
    """Generate scalable multi-output data with exact per-task noise.

    This is the O(n*d) benchmark-scaling counterpart to
    ``generate_multi_output_per_task_noise_data``. It keeps explicit per-task
    observation variances and means, but uses structured latent functions rather
    than dense GP-prior sampling so data generation does not dominate large-n
    scaling rows.
    """

    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")

    noise_per_task = np.ascontiguousarray(noise_per_task, dtype=np.float32)
    if noise_per_task.shape != (num_tasks,):
        raise ValueError(
            f"noise_per_task must have shape ({num_tasks},), got {noise_per_task.shape}"
        )
    if np.any(noise_per_task <= 0):
        raise ValueError("noise_per_task must be strictly positive")

    if mean_per_task is None:
        mean_per_task = np.zeros(num_tasks, dtype=np.float32)
    else:
        mean_per_task = np.ascontiguousarray(mean_per_task, dtype=np.float32)
    if mean_per_task.shape != (num_tasks,):
        raise ValueError(
            f"mean_per_task must have shape ({num_tasks},), got {mean_per_task.shape}"
        )

    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.uniform(x_range[0], x_range[1], size=(n_total, d)).astype(np.float32)
    relevant_dims = min(3, d)
    latent_a, latent_b = _multi_output_structured_ard_latents(X, relevant_dims)

    correlation_levels = {
        "high": 0.85,
        "medium": 0.55,
        "low": 0.30,
        "independent": 0.10,
    }
    corr = np.float32(correlation_levels.get(task_correlation, 0.55))
    phase = np.linspace(0.0, np.pi, num_tasks, dtype=np.float32)
    W = np.stack(
        [
            corr * (np.float32(0.9) + np.float32(0.2) * np.cos(phase)),
            (np.float32(1.0) - np.float32(0.35) * corr)
            * (np.float32(0.35) + np.float32(0.65) * np.sin(phase + np.float32(0.2))),
        ],
        axis=1,
    ).astype(np.float32)
    F = (
        W[np.newaxis, :, 0] * latent_a[:, np.newaxis]
        + W[np.newaxis, :, 1] * latent_b[:, np.newaxis]
        + mean_per_task[np.newaxis, :]
    ).astype(np.float32)
    Y = (
        F
        + rng.standard_normal((n_total, num_tasks)).astype(np.float32)
        * np.sqrt(noise_per_task)[np.newaxis, :]
    ).astype(np.float32)
    B = W @ W.T

    return MultiOutputDataset(
        X_train=X[:n_train],
        Y_train=Y[:n_train],
        X_test=X[n_train:],
        Y_test=Y[n_train:],
        F_test=F[n_train:],
        true_params={
            "dataset_family": "structured_per_task_noise",
            "signal_family": "multi_output_structured_per_task_noise",
            "relevant_dims": int(relevant_dims),
            "relevant_indices": list(range(relevant_dims)),
            "irrelevant_indices": list(range(relevant_dims, d)),
            "noise_per_task": noise_per_task,
            "mean_per_task": mean_per_task,
            "B": B,
            "W": W,
            "task_correlation": task_correlation,
        },
        name=f"structured_per_task_noise_n{n_train}_d{d}_t{num_tasks}_{task_correlation}",
        description=(
            f"Structured multi-output signal with exact per-task observation noise, "
            f"{num_tasks} tasks, and {task_correlation} task correlation"
        ),
    )


def generate_mixed_categorical_data(
    n_train: int,
    n_test: int,
    cont_dim: int,
    cat_levels: list[int],
    true_noise: float = 0.1,
    seed: int = 42,
) -> MixedSyntheticDataset:
    """Generate mixed continuous + categorical regression data.

    The signal is a sum of smooth continuous structure and per-level categorical
    offsets so that a mixed kernel should outperform a continuous-only baseline.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    num_cat_vars = len(cat_levels)

    X_cont = rng.randn(n_total, cont_dim).astype(np.float32)
    C = np.column_stack(
        [rng.randint(0, levels, size=n_total) for levels in cat_levels]
    ).astype(np.float32)

    f_cont = np.zeros(n_total, dtype=np.float32)
    for dim_idx in range(min(cont_dim, 3)):
        f_cont += (1.0 / (dim_idx + 1)) * np.sin(2.0 * X_cont[:, dim_idx]).astype(
            np.float32
        )

    f_cat = np.zeros(n_total, dtype=np.float32)
    cat_effects = []
    for var_idx, levels in enumerate(cat_levels):
        level_effects = rng.randn(levels).astype(np.float32) * 0.8
        cat_effects.append(level_effects)
        f_cat += level_effects[C[:, var_idx].astype(np.int32)]

    f_true = f_cont + f_cat
    y = f_true + np.sqrt(true_noise) * rng.randn(n_total).astype(np.float32)
    X_full = np.column_stack([X_cont, C]).astype(np.float32)

    return MixedSyntheticDataset(
        X_train=X_full[:n_train],
        y_train=y[:n_train],
        X_test=X_full[n_train:],
        y_test=y[n_train:],
        f_test=f_true[n_train:],
        true_params={
            "cont_dim": cont_dim,
            "cat_levels": cat_levels,
            "noise": true_noise,
            "cat_effects": cat_effects,
        },
        name=f"mixed_cont{cont_dim}_cat{'x'.join(map(str, cat_levels))}_n{n_train}",
        description="Mixed continuous + categorical synthetic regression",
    )


def generate_mixed_multi_output_categorical_data(
    n_train: int,
    n_test: int,
    cont_dim: int,
    cat_levels: list[int],
    num_tasks: int = 2,
    noise_per_task: Optional[np.ndarray] = None,
    seed: int = 42,
) -> MultiOutputDataset:
    """Generate mixed continuous-categorical multi-output regression data."""
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be positive, got {num_tasks}")
    if noise_per_task is None:
        noise_per_task = np.linspace(0.04, 0.10, num_tasks, dtype=np.float32)
    else:
        noise_per_task = np.ascontiguousarray(noise_per_task, dtype=np.float32)
    if noise_per_task.shape != (num_tasks,):
        raise ValueError(
            f"noise_per_task must have shape ({num_tasks},), got {noise_per_task.shape}"
        )

    X_cont = rng.randn(n_total, cont_dim).astype(np.float32)
    C = np.column_stack(
        [rng.randint(0, levels, size=n_total) for levels in cat_levels]
    ).astype(np.float32)
    X_full = np.column_stack([X_cont, C]).astype(np.float32)

    cont_latent_a = np.zeros(n_total, dtype=np.float32)
    cont_latent_b = np.zeros(n_total, dtype=np.float32)
    for dim_idx in range(min(cont_dim, 3)):
        x = X_cont[:, dim_idx]
        cont_latent_a += (1.0 / (dim_idx + 1)) * np.sin(1.4 * x).astype(np.float32)
        cont_latent_b += (0.8 / (dim_idx + 1)) * np.cos(1.1 * x).astype(np.float32)

    cat_latent = np.zeros(n_total, dtype=np.float32)
    cat_effects = []
    for var_idx, levels in enumerate(cat_levels):
        level_effects = rng.randn(levels).astype(np.float32) * 0.55
        cat_effects.append(level_effects)
        cat_latent += level_effects[C[:, var_idx].astype(np.int32)]

    phase = np.linspace(0.0, np.pi, num_tasks, dtype=np.float32)
    F = np.stack(
        [
            (0.85 + 0.25 * np.cos(phase[t])) * cont_latent_a
            + (0.55 + 0.35 * np.sin(phase[t])) * cont_latent_b
            + (0.45 + 0.20 * t) * cat_latent
            for t in range(num_tasks)
        ],
        axis=1,
    ).astype(np.float32)
    Y = (
        F
        + rng.randn(n_total, num_tasks).astype(np.float32)
        * np.sqrt(noise_per_task)[np.newaxis, :]
    ).astype(np.float32)

    return MultiOutputDataset(
        X_train=X_full[:n_train],
        Y_train=Y[:n_train],
        X_test=X_full[n_train:],
        Y_test=Y[n_train:],
        F_test=F[n_train:],
        true_params={
            "cont_dim": int(cont_dim),
            "cat_levels": cat_levels,
            "num_tasks": int(num_tasks),
            "noise_per_task": noise_per_task,
            "cat_effects": cat_effects,
        },
        name=f"mixed_multi_output_cont{cont_dim}_cat{'x'.join(map(str, cat_levels))}_n{n_train}_t{num_tasks}",
        description="Mixed continuous + categorical multi-output synthetic regression",
    )


def generate_composite_signal_data(
    n_train: int,
    n_test: int,
    noise_level: str = "medium",
    seed: int = 42,
) -> SyntheticDataset:
    """Generate a 1D signal requiring both smooth and periodic structure."""
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    X = rng.uniform(0.0, 6.0, size=(n_total, 1)).astype(np.float32)
    smooth = 0.6 * np.sin(0.6 * X[:, 0])
    periodic = 1.0 * np.sin(2.5 * X[:, 0])
    trend = 0.15 * X[:, 0]
    f = (smooth + periodic + trend).astype(np.float32)
    noise_std = NOISE_LEVELS.get(noise_level, 0.1) * np.std(f)
    noise_var = float(noise_std**2)
    y = f + noise_std * rng.randn(n_total).astype(np.float32)
    return SyntheticDataset(
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train:],
        y_test=y[n_train:],
        f_test=f[n_train:],
        true_params={"noise": noise_var},
        name=f"composite_signal_n{n_train}",
        description="Smooth + periodic 1D signal for composite-kernel ablations",
    )


def generate_multi_output_heterogeneous_latent_data(
    n_train: int,
    n_test: int,
    seed: int = 42,
) -> MultiOutputHeterogeneousDataset:
    """Generate multi-output data with heterogeneous latent structure.

    Task 0 is dominated by a broad smooth latent, task 1 by a shorter-scale
    oscillatory latent, and task 2 mixes both. This is designed so a two-latent
    LMC model can outperform a single-kernel ICM baseline while staying on a
    numerically stable training path for the active wrapper implementation.
    """
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test
    X = rng.uniform(-2.5, 2.5, size=(n_total, 1)).astype(np.float32)

    latent_smooth = (0.9 * np.sin(0.8 * X[:, 0]) + 0.2 * X[:, 0]).astype(np.float32)
    latent_short_scale = (
        0.8 * np.cos(2.2 * X[:, 0]) + 0.2 * np.sin(1.3 * X[:, 0])
    ).astype(np.float32)

    W = np.array(
        [
            [1.0, 0.15],
            [0.15, 1.0],
            [0.75, 0.75],
        ],
        dtype=np.float32,
    )
    F = np.stack(
        [
            W[t, 0] * latent_smooth + W[t, 1] * latent_short_scale
            for t in range(W.shape[0])
        ],
        axis=1,
    ).astype(np.float32)
    noise_std = np.float32(0.08)
    noise = noise_std * rng.standard_normal(F.shape).astype(np.float32)
    Y = F + noise

    return MultiOutputHeterogeneousDataset(
        X_train=X[:n_train],
        Y_train=Y[:n_train],
        X_test=X[n_train:],
        Y_test=Y[n_train:],
        F_test=F[n_train:],
        true_params={
            "W": W,
            "noise_per_task": np.full(W.shape[0], float(noise_std**2), dtype=np.float32),
        },
        name=f"heterogeneous_latent_n{n_train}",
        description="Multi-output signal with broad and short-scale latent structure",
    )


# =============================================================================
# Convenience Function
# =============================================================================


def generate_data(
    n_train: int,
    n_test: int,
    d: int,
    kernel_type: str,
    data_type: str,
    noise_level: str = "medium",
    seed: int = 42,
    **kwargs,
) -> SyntheticDataset:
    """Convenience function to generate data based on type.

    Args:
        data_type: 'gp_prior', 'smooth', 'oscillatory', 'step', 'polynomial',
                   'friedman1', 'periodic_signal'
    """
    if data_type == "gp_prior":
        return generate_gp_prior_data(
            n_train,
            n_test,
            d,
            kernel_type,
            true_noise=NOISE_LEVELS.get(noise_level, 0.1),
            seed=seed,
            **kwargs,
        )
    else:
        return generate_structured_function_data(
            n_train,
            n_test,
            d,
            data_type,
            noise_level=noise_level,
            seed=seed,
        )
