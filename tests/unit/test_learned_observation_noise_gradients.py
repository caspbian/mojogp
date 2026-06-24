"""Dense reference tests for learned per-sample observation-noise gradients."""

import numpy as np


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nll_per_sample(K, y, raw_noise, noise_floor, regularization, target_noise):
    noise = _softplus(raw_noise) + noise_floor
    A = K + np.diag(noise)
    sign, logdet = np.linalg.slogdet(A)
    assert sign > 0
    alpha = np.linalg.solve(A, y)
    n = len(y)
    nll = 0.5 * (y @ alpha + logdet + n * np.log(2.0 * np.pi)) / n
    if regularization:
        log_ratio = np.log(noise / target_noise)
        nll += 0.5 * regularization * np.mean(log_ratio * log_ratio)
    return float(nll)


def _analytic_raw_noise_grad(K, y, raw_noise, noise_floor, regularization, target_noise):
    noise = _softplus(raw_noise) + noise_floor
    A_inv = np.linalg.inv(K + np.diag(noise))
    alpha = A_inv @ y
    n = len(y)
    grad_noise = 0.5 * (np.diag(A_inv) - alpha * alpha) / n
    if regularization:
        grad_noise += regularization * np.log(noise / target_noise) / (noise * n)
    return grad_noise * _sigmoid(raw_noise)


def _nll_grouped(K, y, group_ids, raw_group_noise, noise_floor, regularization, target_noise):
    group_noise = _softplus(raw_group_noise) + noise_floor
    noise = group_noise[group_ids]
    A = K + np.diag(noise)
    sign, logdet = np.linalg.slogdet(A)
    assert sign > 0
    alpha = np.linalg.solve(A, y)
    n = len(y)
    nll = 0.5 * (y @ alpha + logdet + n * np.log(2.0 * np.pi)) / n
    if regularization:
        log_ratio = np.log(group_noise / target_noise)
        nll += 0.5 * regularization * np.mean(log_ratio * log_ratio)
    return float(nll)


def _analytic_raw_group_noise_grad(
    K, y, group_ids, raw_group_noise, noise_floor, regularization, target_noise
):
    group_noise = _softplus(raw_group_noise) + noise_floor
    noise = group_noise[group_ids]
    A_inv = np.linalg.inv(K + np.diag(noise))
    alpha = A_inv @ y
    per_sample_grad = 0.5 * (np.diag(A_inv) - alpha * alpha) / len(y)
    grad_group = np.zeros_like(raw_group_noise)
    for group_id in range(len(raw_group_noise)):
        grad_group[group_id] = np.sum(per_sample_grad[group_ids == group_id])
    if regularization:
        grad_group += (
            regularization
            * np.log(group_noise / target_noise)
            / (group_noise * len(raw_group_noise))
        )
    return grad_group * _sigmoid(raw_group_noise)


def _nll_linear_noise_function(
    K, X, y, raw_noise_params, noise_floor, regularization, target_noise
):
    raw_noise = raw_noise_params[0] + X @ raw_noise_params[1:]
    noise = _softplus(raw_noise) + noise_floor
    A = K + np.diag(noise)
    sign, logdet = np.linalg.slogdet(A)
    assert sign > 0
    alpha = np.linalg.solve(A, y)
    n = len(y)
    nll = 0.5 * (y @ alpha + logdet + n * np.log(2.0 * np.pi)) / n
    if regularization:
        log_ratio = np.log(noise / target_noise)
        nll += 0.5 * regularization * np.mean(log_ratio * log_ratio)
        nll += 0.5 * regularization * np.mean(raw_noise_params[1:] ** 2)
    return float(nll)


def _analytic_linear_noise_function_grad(
    K, X, y, raw_noise_params, noise_floor, regularization, target_noise
):
    raw_noise = raw_noise_params[0] + X @ raw_noise_params[1:]
    noise = _softplus(raw_noise) + noise_floor
    A_inv = np.linalg.inv(K + np.diag(noise))
    alpha = A_inv @ y
    n = len(y)
    grad_noise = 0.5 * (np.diag(A_inv) - alpha * alpha) / n
    if regularization:
        grad_noise += regularization * np.log(noise / target_noise) / (noise * n)
    grad_raw = grad_noise * _sigmoid(raw_noise)
    grad_params = np.empty_like(raw_noise_params)
    grad_params[0] = np.sum(grad_raw)
    grad_params[1:] = X.T @ grad_raw
    if regularization and X.shape[1] > 0:
        grad_params[1:] += regularization * raw_noise_params[1:] / X.shape[1]
    return grad_params


def test_learned_vector_noise_gradient_matches_dense_finite_difference():
    rng = np.random.default_rng(42)
    X = np.linspace(-1.0, 1.0, 6, dtype=np.float64).reshape(-1, 1)
    d2 = (X - X.T) ** 2
    K = 1.3 * np.exp(-0.5 * d2 / (0.7**2))
    y = rng.normal(size=6)
    raw_noise = np.linspace(-4.0, -2.0, 6)
    noise_floor = 1e-5
    regularization = 0.03
    target_noise = 0.05

    analytic = _analytic_raw_noise_grad(
        K, y, raw_noise, noise_floor, regularization, target_noise
    )
    finite_diff = np.zeros_like(raw_noise)
    eps = 1e-5
    for i in range(len(raw_noise)):
        plus = raw_noise.copy()
        minus = raw_noise.copy()
        plus[i] += eps
        minus[i] -= eps
        finite_diff[i] = (
            _nll_per_sample(K, y, plus, noise_floor, regularization, target_noise)
            - _nll_per_sample(K, y, minus, noise_floor, regularization, target_noise)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, finite_diff, rtol=2e-5, atol=2e-6)


def test_log_noise_regularization_gradient_is_zero_at_target():
    K = np.eye(4, dtype=np.float64)
    y = np.zeros(4, dtype=np.float64)
    target_noise = 0.04
    noise_floor = 1e-5
    raw_noise = np.full(4, np.log(np.exp(target_noise - noise_floor) - 1.0))

    grad = _analytic_raw_noise_grad(
        K,
        y,
        raw_noise,
        noise_floor=noise_floor,
        regularization=0.1,
        target_noise=target_noise,
    )
    unregularized = _analytic_raw_noise_grad(
        K,
        y,
        raw_noise,
        noise_floor=noise_floor,
        regularization=0.0,
        target_noise=target_noise,
    )
    np.testing.assert_allclose(grad, unregularized, rtol=1e-6, atol=1e-8)


def test_learned_grouped_noise_gradient_matches_dense_finite_difference():
    rng = np.random.default_rng(7)
    X = np.linspace(-1.0, 1.0, 7, dtype=np.float64).reshape(-1, 1)
    d2 = (X - X.T) ** 2
    K = 1.1 * np.exp(-0.5 * d2 / (0.8**2))
    y = rng.normal(size=7)
    group_ids = np.array([0, 1, 0, 2, 1, 2, 0], dtype=np.int64)
    raw_group_noise = np.array([-4.0, -3.0, -2.5], dtype=np.float64)
    noise_floor = 1e-5
    regularization = 0.04
    target_noise = 0.05

    analytic = _analytic_raw_group_noise_grad(
        K,
        y,
        group_ids,
        raw_group_noise,
        noise_floor,
        regularization,
        target_noise,
    )
    finite_diff = np.zeros_like(raw_group_noise)
    eps = 1e-5
    for group_id in range(len(raw_group_noise)):
        plus = raw_group_noise.copy()
        minus = raw_group_noise.copy()
        plus[group_id] += eps
        minus[group_id] -= eps
        finite_diff[group_id] = (
            _nll_grouped(
                K, y, group_ids, plus, noise_floor, regularization, target_noise
            )
            - _nll_grouped(
                K, y, group_ids, minus, noise_floor, regularization, target_noise
            )
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, finite_diff, rtol=2e-5, atol=2e-6)


def test_learned_linear_input_dependent_noise_gradient_matches_dense_finite_difference():
    rng = np.random.default_rng(19)
    X = np.column_stack(
        [
            np.linspace(-1.0, 1.0, 6, dtype=np.float64),
            rng.normal(size=6),
        ]
    )
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    K = 1.2 * np.exp(-0.5 * d2 / (0.9**2))
    y = rng.normal(size=6)
    raw_noise_params = np.array([-3.5, 0.35, -0.2], dtype=np.float64)
    noise_floor = 1e-5
    regularization = 0.03
    target_noise = 0.04

    analytic = _analytic_linear_noise_function_grad(
        K, X, y, raw_noise_params, noise_floor, regularization, target_noise
    )
    finite_diff = np.zeros_like(raw_noise_params)
    eps = 1e-5
    for p in range(len(raw_noise_params)):
        plus = raw_noise_params.copy()
        minus = raw_noise_params.copy()
        plus[p] += eps
        minus[p] -= eps
        finite_diff[p] = (
            _nll_linear_noise_function(K, X, y, plus, noise_floor, regularization, target_noise)
            - _nll_linear_noise_function(K, X, y, minus, noise_floor, regularization, target_noise)
        ) / (2.0 * eps)

    np.testing.assert_allclose(analytic, finite_diff, rtol=2e-5, atol=2e-6)
