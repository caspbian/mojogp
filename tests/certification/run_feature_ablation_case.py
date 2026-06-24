"""Run focused feature ablation/certification cases in isolated processes."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mojogp import Kernel, MultiOutputGP, MultiOutputLMCGP, SingleOutputGP
from tests.shared.subprocess_harness import run_child_main


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _nlpd(y_true, mean, variance) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-9)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * variance) + 0.5 * ((y_true - mean) ** 2) / variance))


def _coverage_95(y_true, mean, variance) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 0.0))
    return float(np.mean(np.abs(y_true - mean) <= 1.96 * std))


def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _single_fit_kwargs(method: str, *, iters: int = 5) -> dict[str, Any]:
    return dict(
        method=method,
        max_iterations=iters,
        learning_rate=0.035,
        verbose=False,
        num_probes=2,
        max_cg_iterations=24,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )


def _multi_kwargs() -> dict[str, Any]:
    return dict(
        num_probes=2,
        max_cg_iterations=24,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )


def _ard_data(seed: int, n_train: int = 2000, n_test: int = 128):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, 6)).astype(np.float32)
    f = (np.sin(np.float32(2.0) * X[:, 0]) + np.float32(0.7) * X[:, 1]).astype(
        np.float32
    )
    y = (f + np.float32(0.05) * rng.standard_normal(n_train + n_test)).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], f[n_train:]


def _composite_data(seed: int, n_train: int = 2000, n_test: int = 128):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, 2)).astype(np.float32)
    f = (np.sin(np.float32(2.0) * X[:, 0]) + np.cos(np.float32(2.5) * X[:, 1])).astype(
        np.float32
    )
    y = (f + np.float32(0.04) * rng.standard_normal(n_train + n_test)).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], f[n_train:]


def _product_composite_data(seed: int, n_train: int = 2000, n_test: int = 128):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, 2)).astype(np.float32)
    f = (
        np.sin(np.float32(2.0) * X[:, 0])
        * np.cos(np.float32(2.5) * X[:, 1])
    ).astype(np.float32)
    y = (f + np.float32(0.04) * rng.standard_normal(n_train + n_test)).astype(np.float32)
    X_shuffled = X.copy()
    X_shuffled[:n_train, 1] = rng.permutation(X_shuffled[:n_train, 1])
    return X[:n_train], y[:n_train], X[n_train:], f[n_train:], X_shuffled[:n_train]


def _mixed_multi_data(seed: int, n_train: int = 2000, n_test: int = 96):
    rng = np.random.default_rng(seed)
    X_cont = rng.normal(size=(n_train + n_test, 2)).astype(np.float32)
    cat = rng.integers(0, 3, size=n_train + n_test, endpoint=False)
    effect = np.array([-1.6, 0.0, 1.6], dtype=np.float32)[cat]
    F = np.stack(
        [
            np.sin(X_cont[:, 0]) + np.float32(0.25) * X_cont[:, 1] + effect,
            np.float32(0.7) * np.cos(X_cont[:, 0]) - np.float32(0.2) * X_cont[:, 1] + np.float32(0.8) * effect,
        ],
        axis=1,
    ).astype(np.float32)
    Y = (F + np.float32(0.06) * rng.standard_normal(F.shape)).astype(np.float32)
    X = np.column_stack([X_cont, cat.astype(np.float32)]).astype(np.float32)
    X_shuffled = X.copy()
    X_shuffled[n_train:, 2] = rng.permutation(X_shuffled[n_train:, 2])
    return X[:n_train], Y[:n_train], X[n_train:], F[n_train:], X_shuffled[n_train:]


def _noise_data(seed: int, n_train: int = 2000, n_test: int = 64):
    rng = np.random.default_rng(seed)
    X = np.linspace(-2.0, 2.0, n_train + n_test, dtype=np.float32).reshape(-1, 1)
    f = np.sin(np.float32(3.0) * X[:, 0]).astype(np.float32)
    noise = (0.012 + 0.055 * (X[:, 0] > 0.0)).astype(np.float32)
    y = (f + rng.normal(scale=np.sqrt(noise)).astype(np.float32)).astype(np.float32)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    return X[:n_train], y[:n_train], noise[:n_train], groups[:n_train], X[n_train:], f[n_train:], noise[n_train:], groups[n_train:]


def _balanced_noise_data(seed: int, n_train: int = 2000, n_test: int = 256):
    rng = np.random.default_rng(seed)
    n_total = int(n_train) + int(n_test)
    half_train = int(n_train) // 2
    half_test = int(n_test) // 2
    x_train_low = rng.uniform(-2.0, -0.05, size=half_train)
    x_train_high = rng.uniform(0.05, 2.0, size=int(n_train) - half_train)
    x_test_low = rng.uniform(-2.0, -0.05, size=half_test)
    x_test_high = rng.uniform(0.05, 2.0, size=int(n_test) - half_test)
    X = np.concatenate([x_train_low, x_train_high, x_test_low, x_test_high]).astype(np.float32).reshape(n_total, 1)
    f = np.sin(np.float32(3.0) * X[:, 0]).astype(np.float32)
    noise = (0.012 + 0.055 * (X[:, 0] > 0.0)).astype(np.float32)
    y = (f + rng.normal(scale=np.sqrt(noise)).astype(np.float32)).astype(np.float32)
    train_order = rng.permutation(int(n_train))
    test_order = int(n_train) + rng.permutation(int(n_test))
    X_train = X[:n_train][train_order]
    y_train = y[:n_train][train_order]
    noise_train = noise[:n_train][train_order]
    X_test = X[n_train:][test_order - int(n_train)]
    y_test = y[n_train:][test_order - int(n_train)]
    f_test = f[n_train:][test_order - int(n_train)]
    noise_test = noise[n_train:][test_order - int(n_train)]
    return X_train, y_train, noise_train, X_test, y_test, f_test, noise_test


def _multi_continuous_data(seed: int, n_train: int = 2000, n_test: int = 96):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_train + n_test, 2)).astype(np.float32)
    shared = np.sin(X[:, 0]) + np.float32(0.25) * X[:, 1]
    second = np.cos(np.float32(1.7) * X[:, 1])
    F_shared = np.stack([shared, shared], axis=1).astype(np.float32)
    F_hetero = np.stack([shared + second, shared - second], axis=1).astype(np.float32)
    Y_shared = F_shared.copy()
    Y_shared[:, 0] += np.float32(0.03) * rng.standard_normal(F_shared.shape[0])
    Y_shared[:, 1] += np.float32(0.45) * rng.standard_normal(F_shared.shape[0])
    Y_shared = Y_shared.astype(np.float32)
    Y_hetero = (F_hetero + np.float32(0.08) * rng.standard_normal(F_hetero.shape)).astype(np.float32)
    return X[:n_train], X[n_train:], F_shared[n_train:], Y_shared[:n_train], F_hetero[n_train:], Y_hetero[:n_train]


def _fit_single_predict(kernel, X_train, y_train, X_test, method: str, *, iters: int = 5, **fit_overrides):
    gp = SingleOutputGP(kernel)
    kwargs = _single_fit_kwargs(method, iters=iters)
    kwargs.update(fit_overrides)
    result = gp.fit(X_train, y_train, **kwargs)
    pred = gp.predict(X_test, variance_method="love")
    return gp, result, pred


def _case_ard_single(payload: dict[str, object]) -> dict[str, Any]:
    method = str(payload["method"])
    variant = str(payload["variant"])
    seed = int(payload.get("seed", 0))
    X_train, y_train, X_test, f_test = _ard_data(seed)
    if variant == "ard":
        init = np.array([0.45, 0.8, 4.0, 4.0, 4.0, 4.0, 1.0], dtype=np.float32)
        gp, result, pred = _fit_single_predict(
            Kernel.rbf(ard=True),
            X_train,
            y_train,
            X_test,
            method,
            iters=4,
            initial_params=init,
        )
        params = np.asarray(result.params, dtype=np.float32)
        relevant = params[:2]
        irrelevant = params[2:6]
        return {
            "variant": variant,
            "method": method,
            "nll": float(result.nll),
            "nll_delta": float(result.nll_history[0] - result.nll_history[-1]) if result.nll_history is not None and len(result.nll_history) > 1 else 0.0,
            "rmse": _rmse(f_test, pred.mean),
            "lengthscales": params[:6],
            "relevant_lengthscale_mean": float(np.mean(relevant)),
            "relevant_lengthscale_max": float(np.max(relevant)),
            "irrelevant_lengthscale_mean": float(np.mean(irrelevant)),
            "irrelevant_lengthscale_min": float(np.min(irrelevant)),
            "lengthscale_relevance_ratio": float(np.mean(irrelevant) / np.mean(relevant)),
            "training_route": gp.backend_train_info.get("training_route"),
            "prediction_route": gp.backend_predict_info.get("actual_prediction_route"),
        }
    if variant == "isotropic":
        gp, _result, pred = _fit_single_predict(
            Kernel.rbf(),
            X_train,
            y_train,
            X_test,
            method,
            iters=4,
            initial_params=np.array([2.5, 1.0], dtype=np.float32),
        )
        return {
            "variant": variant,
            "method": method,
            "nll": float(_result.nll),
            "nll_delta": float(_result.nll_history[0] - _result.nll_history[-1]) if _result.nll_history is not None and len(_result.nll_history) > 1 else 0.0,
            "rmse": _rmse(f_test, pred.mean),
            "training_route": gp.backend_train_info.get("training_route"),
            "prediction_route": gp.backend_predict_info.get("actual_prediction_route"),
        }
    raise ValueError(f"unknown ARD variant: {variant}")


def _case_composite_single(payload: dict[str, object]) -> dict[str, Any]:
    method = str(payload["method"])
    variant = str(payload["variant"])
    seed = int(payload.get("seed", 0))
    X_train, y_train, X_test, f_test = _composite_data(seed)
    kernels = {
        "x0_only": Kernel.rbf(active_dims=[0]),
        "x1_only": Kernel.rbf(active_dims=[1]),
        "additive": Kernel.rbf(active_dims=[0]) + Kernel.rbf(active_dims=[1]),
    }
    if variant not in kernels:
        raise ValueError(f"unknown composite variant: {variant}")
    gp, _result, pred = _fit_single_predict(
        kernels[variant], X_train, y_train, X_test, method, iters=5
    )
    return {
        "variant": variant,
        "method": method,
        "nll": float(_result.nll),
        "rmse": _rmse(f_test, pred.mean),
        "training_route": gp.backend_train_info.get("training_route"),
        "prediction_route": gp.backend_predict_info.get("actual_prediction_route"),
        "fallback_used": bool(gp.backend_predict_info.get("fallback_used", False)),
    }


def _case_product_composite_single(payload: dict[str, object]) -> dict[str, Any]:
    method = str(payload["method"])
    variant = str(payload["variant"])
    seed = int(payload.get("seed", 0))
    X_train, y_train, X_test, f_test, X_train_shuffled = _product_composite_data(seed)
    kernels = {
        "x0_only": Kernel.rbf(active_dims=[0]),
        "x1_only": Kernel.rbf(active_dims=[1]),
        "additive": Kernel.rbf(active_dims=[0]) + Kernel.rbf(active_dims=[1]),
        "product": Kernel.rbf(active_dims=[0]) * Kernel.rbf(active_dims=[1]),
        "shuffled_product": Kernel.rbf(active_dims=[0]) * Kernel.rbf(active_dims=[1]),
    }
    if variant != "all" and variant not in kernels:
        raise ValueError(f"unknown product composite variant: {variant}")

    def run_variant(name: str) -> dict[str, Any]:
        train_X = X_train_shuffled if name == "shuffled_product" else X_train
        gp, result, pred = _fit_single_predict(
            kernels[name], train_X, y_train, X_test, method, iters=5
        )
        out = {
            "variant": name,
            "method": method,
            "nll": float(result.nll),
            "rmse": _rmse(f_test, pred.mean),
            "training_route": gp.backend_train_info.get("training_route"),
            "prediction_route": gp.backend_predict_info.get("actual_prediction_route"),
            "fallback_used": bool(gp.backend_predict_info.get("fallback_used", False)),
        }
        gp._destroy_provider_info()
        return out

    if variant == "all":
        return {
            "variant": variant,
            "method": method,
            "results": {name: run_variant(name) for name in kernels},
        }
    return run_variant(variant)


def _case_love_parity(payload: dict[str, object]) -> dict[str, Any]:
    surface = str(payload["surface"])
    method = str(payload["method"])
    seed = int(payload.get("seed", 0))
    mixed = surface.endswith("mixed")
    f_ref = None
    if surface.startswith("single"):
        if mixed:
            X_train, Y_train, X_test, F, _X_shuf = _mixed_multi_data(seed)
            y_train = Y_train[:, 0]
            f_ref = F[:, 0]
            kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])
        else:
            X_train, y_train, X_test, f_ref = _composite_data(seed)
            kernel = Kernel.rbf()
        gp, _result, _pred = _fit_single_predict(kernel, X_train, y_train, X_test, method, iters=3)
        exact = gp.predict(X_test[:10], variance_method="exact")
        exact_info = dict(gp.backend_predict_info)
        love = gp.predict(X_test[:10], variance_method="love")
        love_info = dict(gp.backend_predict_info)
    elif surface.startswith("icm"):
        if mixed:
            X_train, Y_train, X_test, f_ref, _X_shuf = _mixed_multi_data(seed)
            kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])
        else:
            X_train, X_test, f_ref, Y_train, _Fh, _Yh = _multi_continuous_data(seed)
            kernel = Kernel.rbf()
        gp = MultiOutputGP(kernel=kernel, task_rank=1, **_multi_kwargs())
        gp.fit(X_train, Y_train, method=method, max_iterations=3, learning_rate=0.03, verbose=False)
        exact = gp.predict(X_test[:10], variance_method="exact")
        exact_info = dict(gp.backend_predict_info)
        love = gp.predict(X_test[:10], variance_method="love")
        love_info = dict(gp.backend_predict_info)
    elif surface.startswith("lmc"):
        if mixed:
            X_train, Y_train, X_test, f_ref, _X_shuf = _mixed_multi_data(seed)
            kernels = [
                Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
                Kernel.matern52(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
            ]
        else:
            X_train, X_test, _F, Y_train, f_ref, Y_train = _multi_continuous_data(seed)
            kernels = [Kernel.rbf(), Kernel.matern52()]
        gp = MultiOutputLMCGP(kernels=kernels, **_multi_kwargs())
        gp.fit(X_train, Y_train, method=method, max_iterations=3, learning_rate=0.03, verbose=False)
        exact = gp.predict(X_test[:10], variance_method="exact")
        exact_info = dict(gp.backend_predict_info)
        love = gp.predict(X_test[:10], variance_method="love")
        love_info = dict(gp.backend_predict_info)
    else:
        raise ValueError(f"unknown LOVE parity surface: {surface}")

    exact_var = np.asarray(exact.variance, dtype=np.float32)
    love_var = np.asarray(love.variance, dtype=np.float32)
    f_eval = np.asarray(f_ref[:10], dtype=np.float32)
    return {
        "surface": surface,
        "method": method,
        "mean_max_abs_diff": float(np.max(np.abs(exact.mean - love.mean))),
        "variance_corr": _corr(exact_var, love_var),
        "variance_rel_mae": float(
            np.mean(np.abs(exact_var - love_var)) / max(float(np.mean(np.abs(exact_var))), 1e-6)
        ),
        "variance_mean_ratio": float(
            np.mean(love_var) / max(float(np.mean(exact_var)), 1e-6)
        ),
        "exact_coverage_95": _coverage_95(f_eval, exact.mean, exact_var),
        "love_coverage_95": _coverage_95(f_eval, love.mean, love_var),
        "love_nlpd": _nlpd(f_eval, love.mean, love_var),
        "exact_prediction_route": exact_info.get("actual_prediction_route"),
        "exact_variance_route": exact_info.get("actual_variance_route"),
        "love_variance_route": love_info.get("actual_variance_route"),
        "love_fallback_used": bool(love_info.get("fallback_used", False)),
        "min_exact_variance": float(np.min(exact_var)),
        "min_love_variance": float(np.min(love_var)),
    }


def _case_noise_single(payload: dict[str, object]) -> dict[str, Any]:
    method = str(payload["method"])
    variant = str(payload["variant"])
    seed = int(payload.get("seed", 0))
    X, y, noise, groups, X_test, _f_test, noise_test, groups_test = _noise_data(seed)
    if variant == "fixed_vector":
        gp = SingleOutputGP(Kernel.rbf())
        gp.fit(
            X,
            y,
            observation_noise=noise,
            learn_noise=False,
            method=method,
            max_iterations=2,
            learning_rate=0.03,
            num_probes=2,
            max_cg_iterations=30,
            max_tridiag_iterations=8,
            preconditioner_rank=0,
            verbose=False,
        )
        latent = gp.predict_latent(X_test[:8], variance_method="exact", preconditioner_rank=0)
        observed = gp.predict_observed(
            X_test[:8], observation_noise=noise_test[:8], variance_method="exact", preconditioner_rank=0
        )
        return {
            "variant": variant,
            "method": method,
            "noise_mode": gp.backend_train_info.get("noise_mode"),
            "observed_delta_max_abs": float(
                np.max(np.abs(observed.variance - latent.variance - noise_test[:8]))
            ),
            "training_route": gp.backend_train_info.get("training_route"),
        }
    if variant == "fixed_vector_calibration":
        X, y, noise, X_test, y_test, _f_test, noise_test = _balanced_noise_data(seed)
        fixed = SingleOutputGP(Kernel.rbf())
        fixed.fit(
            X,
            y,
            observation_noise=noise,
            learn_noise=False,
            method=method,
            max_iterations=3,
            learning_rate=0.03,
            num_probes=2,
            max_cg_iterations=30,
            max_tridiag_iterations=8,
            preconditioner_rank=0,
            verbose=False,
        )
        fixed_pred = fixed.predict_observed(
            X_test, observation_noise=noise_test, variance_method="exact", preconditioner_rank=0
        )
        scalar_noise = np.full_like(noise, float(np.mean(noise)))
        scalar = SingleOutputGP(Kernel.rbf())
        scalar.fit(
            X,
            y,
            observation_noise=scalar_noise,
            learn_noise=False,
            method=method,
            max_iterations=3,
            learning_rate=0.03,
            num_probes=2,
            max_cg_iterations=30,
            max_tridiag_iterations=8,
            preconditioner_rank=0,
            verbose=False,
        )
        scalar_pred = scalar.predict_observed(
            X_test,
            observation_noise=np.full_like(noise_test, float(np.mean(noise))),
            variance_method="exact",
            preconditioner_rank=0,
        )
        return {
            "variant": variant,
            "method": method,
            "noise_mode": fixed.backend_train_info.get("noise_mode"),
            "fixed_nlpd": _nlpd(y_test, fixed_pred.mean, fixed_pred.variance),
            "scalar_nlpd": _nlpd(y_test, scalar_pred.mean, scalar_pred.variance),
            "fixed_coverage_95": _coverage_95(y_test, fixed_pred.mean, fixed_pred.variance),
            "scalar_coverage_95": _coverage_95(y_test, scalar_pred.mean, scalar_pred.variance),
            "training_route": fixed.backend_train_info.get("training_route"),
        }
    if variant == "grouped":
        group_noise = np.array([0.012, 0.067], dtype=np.float32)
        grouped = SingleOutputGP(Kernel.rbf())
        grouped.fit(
            X,
            y,
            noise_model="grouped",
            noise_group_train=groups,
            group_noise=group_noise,
            learn_noise=False,
            method=method,
            max_iterations=2,
            learning_rate=0.03,
            num_probes=2,
            max_cg_iterations=30,
            max_tridiag_iterations=8,
            preconditioner_rank=0,
            verbose=False,
        )
        vector = SingleOutputGP(Kernel.rbf())
        vector.fit(
            X,
            y,
            observation_noise=group_noise[groups],
            learn_noise=False,
            method=method,
            max_iterations=2,
            learning_rate=0.03,
            num_probes=2,
            max_cg_iterations=30,
            max_tridiag_iterations=8,
            preconditioner_rank=0,
            verbose=False,
        )
        grouped_pred = grouped.predict_latent(X_test[:8], variance_method="mean_only", preconditioner_rank=0)
        vector_pred = vector.predict_latent(X_test[:8], variance_method="mean_only", preconditioner_rank=0)
        observed = grouped.predict_observed(
            X_test[:8], noise_group_test=groups_test[:8], variance_method="exact", preconditioner_rank=0
        )
        latent = grouped.predict_latent(X_test[:8], variance_method="exact", preconditioner_rank=0)
        return {
            "variant": variant,
            "method": method,
            "noise_mode": grouped.backend_train_info.get("noise_mode"),
            "grouped_vector_mean_max_abs_diff": float(
                np.max(np.abs(grouped_pred.mean - vector_pred.mean))
            ),
            "observed_delta_max_abs": float(
                np.max(np.abs(observed.variance - latent.variance - group_noise[groups_test[:8]]))
            ),
            "training_route": grouped.backend_train_info.get("training_route"),
        }
    if variant == "input_dependent":
        noise_fn = lambda x: (0.012 + 0.055 * (x[:, 0] > 0.0)).astype(np.float32)
        gp = SingleOutputGP(Kernel.rbf())
        gp.fit(
            X,
            y,
            observation_noise_fn=noise_fn,
            learn_noise=False,
            method=method,
            max_iterations=2,
            learning_rate=0.03,
            num_probes=2,
            max_cg_iterations=30,
            max_tridiag_iterations=8,
            preconditioner_rank=0,
            verbose=False,
        )
        latent = gp.predict_latent(X_test[:8], variance_method="exact", preconditioner_rank=0)
        observed = gp.predict_observed(X_test[:8], variance_method="exact", preconditioner_rank=0)
        return {
            "variant": variant,
            "method": method,
            "noise_mode": gp.backend_train_info.get("noise_mode"),
            "observed_delta_max_abs": float(
                np.max(np.abs(observed.variance - latent.variance - noise_fn(X_test[:8])))
            ),
            "training_route": gp.backend_train_info.get("training_route"),
        }
    if variant == "learned_input_dependent":
        gp = SingleOutputGP(Kernel.rbf())
        gp.fit(
            X,
            y,
            noise_model="learned_input_dependent",
            noise_function="linear",
            initial_noise=0.03,
            method=method,
            max_iterations=4,
            learning_rate=0.02,
            num_probes=2,
            max_cg_iterations=30,
            max_tridiag_iterations=8,
            preconditioner_rank=0,
            verbose=False,
        )
        low = np.array([[-1.0], [-0.5]], dtype=np.float32)
        high = np.array([[0.5], [1.0]], dtype=np.float32)
        low_noise = gp._evaluate_learned_noise_function(low, expected_n=2)
        high_noise = gp._evaluate_learned_noise_function(high, expected_n=2)
        return {
            "variant": variant,
            "method": method,
            "noise_mode": gp.backend_train_info.get("noise_mode"),
            "learned_noise_low_mean": float(np.mean(low_noise)),
            "learned_noise_high_mean": float(np.mean(high_noise)),
            "has_noise_function_params": gp._noise_function_params is not None,
            "training_route": gp.backend_train_info.get("training_route"),
        }
    raise ValueError(f"unknown noise variant: {variant}")


def _case_multi_output_structure(payload: dict[str, object]) -> dict[str, Any]:
    method = str(payload["method"])
    variant = str(payload["variant"])
    seed = int(payload.get("seed", 0))
    X_train, X_test, F_shared, Y_shared, F_hetero, Y_hetero = _multi_continuous_data(seed)
    if variant == "independent_shared":
        means = []
        for task in range(2):
            gp, _result, pred = _fit_single_predict(
                Kernel.rbf(), X_train, Y_shared[:, task], X_test, method, iters=4
            )
            means.append(pred.mean)
        mean = np.stack(means, axis=1)
        return {"variant": variant, "method": method, "rmse": _rmse(F_shared, mean)}
    if variant == "icm_shared":
        gp = MultiOutputGP(kernel=Kernel.rbf(), task_rank=1, **_multi_kwargs())
        result = gp.fit(X_train, Y_shared, method=method, max_iterations=30, learning_rate=0.03, verbose=False)
        pred = gp.predict(X_test, variance_method="love")
        return {
            "variant": variant,
            "method": method,
            "rmse": _rmse(F_shared, pred.mean),
            "task_corr": float(result.B[0, 1] / np.sqrt(max(result.B[0, 0] * result.B[1, 1], 1e-12))),
            "prediction_task_corr": _corr(pred.mean[:, 0], pred.mean[:, 1]),
            "training_route": gp.backend_train_info.get("training_route"),
        }
    if variant == "icm_heterogeneous":
        gp = MultiOutputGP(kernel=Kernel.rbf(), task_rank=1, **_multi_kwargs())
        gp.fit(X_train, Y_hetero, method=method, max_iterations=4, learning_rate=0.03, verbose=False)
        pred = gp.predict(X_test, variance_method="love")
        return {"variant": variant, "method": method, "rmse": _rmse(F_hetero, pred.mean)}
    if variant == "lmc_heterogeneous":
        gp = MultiOutputLMCGP(
            kernels=[Kernel.rbf(active_dims=[0]), Kernel.rbf(active_dims=[1])],
            **_multi_kwargs(),
        )
        gp.fit(X_train, Y_hetero, method=method, max_iterations=4, learning_rate=0.03, verbose=False)
        pred = gp.predict(X_test, variance_method="love")
        return {
            "variant": variant,
            "method": method,
            "rmse": _rmse(F_hetero, pred.mean),
            "training_route": gp.backend_train_info.get("training_route"),
        }
    raise ValueError(f"unknown multi-output structure variant: {variant}")


def _case_route_parity(payload: dict[str, object]) -> dict[str, Any]:
    variant = str(payload["variant"])
    seed = int(payload.get("seed", 0))
    if variant == "materialized":
        method = "materialized"
    elif variant == "matrix_free":
        method = "matrix_free"
    else:
        raise ValueError(f"unknown route parity variant: {variant}")
    X_train, y_train, X_test, f_test = _composite_data(seed)
    gp, result, pred = _fit_single_predict(
        Kernel.rbf(),
        X_train,
        y_train,
        X_test,
        method,
        iters=2,
        learning_rate=1e-4,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
    )
    return {
        "variant": variant,
        "method": method,
        "rmse": _rmse(f_test, pred.mean),
        "mean": np.asarray(pred.mean, dtype=np.float32),
        "params": np.asarray(result.params, dtype=np.float32),
        "noise": float(result.noise),
        "training_route": gp.backend_train_info.get("training_route"),
        "prediction_route": gp.backend_predict_info.get("actual_prediction_route"),
    }


def _case_active_dims_nonrbf(payload: dict[str, object]) -> dict[str, Any]:
    method = str(payload["method"])
    variant = str(payload["variant"])
    seed = int(payload.get("seed", 0))
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2128, 3)).astype(np.float32)
    f = (np.float32(1.4) * X[:, 2]).astype(np.float32)
    y = (f + np.float32(0.04) * rng.standard_normal(2128)).astype(np.float32)
    kernels = {
        "wrong_dim": Kernel.linear(active_dims=[0]),
        "linear_poly_active_dim": Kernel.linear(active_dims=[2]),
    }
    if variant not in kernels:
        raise ValueError(f"unknown active-dim variant: {variant}")
    gp, _result, pred = _fit_single_predict(
        kernels[variant], X[:2000], y[:2000], X[2000:], method, iters=4
    )
    return {
        "variant": variant,
        "method": method,
        "rmse": _rmse(f[2000:], pred.mean),
        "training_route": gp.backend_train_info.get("training_route"),
        "prediction_route": gp.backend_predict_info.get("actual_prediction_route"),
    }


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    warnings.simplefilter("ignore")
    case = str(payload["case"])
    if case == "ard_single":
        result = _case_ard_single(payload)
    elif case == "composite_single":
        result = _case_composite_single(payload)
    elif case == "product_composite_single":
        result = _case_product_composite_single(payload)
    elif case == "love_parity":
        result = _case_love_parity(payload)
    elif case == "noise_single":
        result = _case_noise_single(payload)
    elif case == "multi_output_structure":
        result = _case_multi_output_structure(payload)
    elif case == "route_parity":
        result = _case_route_parity(payload)
    elif case == "active_dims_nonrbf":
        result = _case_active_dims_nonrbf(payload)
    else:
        raise ValueError(f"unknown feature ablation case: {case}")
    return {"payload": result}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
