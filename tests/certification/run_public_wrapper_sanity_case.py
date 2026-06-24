"""Run one public wrapper sanity workflow in an isolated process."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mojogp import Kernel, MultiOutputGP, MultiOutputLMCGP, SingleOutputGP
from tests.shared.subprocess_harness import run_child_main


def _single_data(*, mixed: bool, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_cont = rng.normal(size=(n + 12, 2)).astype(np.float32)
    y = (
        np.sin(X_cont[:, 0])
        + np.float32(0.25) * X_cont[:, 1]
        + np.float32(0.04) * rng.standard_normal(n + 12)
    ).astype(np.float32)
    if not mixed:
        return X_cont[:n], y[:n], X_cont[n:]
    cat = rng.integers(0, 3, size=n + 12, endpoint=False)
    y = (y + np.array([-0.6, 0.1, 0.7], dtype=np.float32)[cat]).astype(np.float32)
    X = np.column_stack([X_cont, cat.astype(np.float32)]).astype(np.float32)
    return X[:n], y[:n], X[n:]


def _multi_data(*, mixed: bool, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_cont = rng.normal(size=(n + 12, 2)).astype(np.float32)
    latent = np.sin(X_cont[:, 0]) + np.float32(0.25) * X_cont[:, 1]
    Y = np.stack(
        [latent, np.float32(0.7) * latent + np.float32(0.2) * X_cont[:, 0]],
        axis=1,
    ).astype(np.float32)
    if mixed:
        cat = rng.integers(0, 3, size=n + 12, endpoint=False)
        effects = np.array([-0.7, 0.1, 0.8], dtype=np.float32)[cat]
        Y[:, 0] += effects
        Y[:, 1] += np.float32(0.6) * effects
        X = np.column_stack([X_cont, cat.astype(np.float32)]).astype(np.float32)
    else:
        X = X_cont
    Y += np.float32(0.04) * rng.standard_normal(Y.shape).astype(np.float32)
    return X[:n], Y[:n], X[n:]


def _kernel_for(surface: str):
    if surface.endswith("mixed"):
        return Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])
    return Kernel.rbf()


def _lmc_kernels_for(surface: str):
    if surface == "lmc_mixed":
        return [
            Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
            Kernel.matern52(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2]),
        ]
    return [Kernel.rbf(), Kernel.matern52()]


def _finite_prediction(payload: dict[str, Any], mean, variance) -> None:
    payload["mean_shape"] = tuple(np.asarray(mean).shape)
    if variance is not None:
        payload["variance_shape"] = tuple(np.asarray(variance).shape)
        payload["min_variance"] = float(np.min(variance))
        payload["max_variance"] = float(np.max(variance))
    assert np.all(np.isfinite(mean))
    if variance is not None:
        assert np.all(np.isfinite(variance))
        assert np.all(np.asarray(variance) >= 0.0)


def _release_provider_for_load(gp: Any, surface: str) -> None:
    if surface == "single_mixed":
        gp._destroy_provider_info()
    elif surface.startswith("icm"):
        gp._destroy_persistent_provider()


def _single_workflow(surface: str, method: str, seed: int) -> dict[str, Any]:
    mixed = surface == "single_mixed"
    X, y, X_test = _single_data(mixed=mixed, seed=seed)
    gp = SingleOutputGP(_kernel_for(surface))
    fit_kwargs = dict(
        method=method,
        max_iterations=3,
        learning_rate=0.035,
        verbose=False,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp.fit(X, y, **fit_kwargs)

    exact = gp.predict(X_test[:6], variance_method="exact")
    exact_info = dict(gp.backend_predict_info)
    love = gp.predict(X_test[:6], variance_method="love")
    love_info = dict(gp.backend_predict_info)
    observed = gp.predict_observed(
        X_test[:6], observation_noise=np.full(6, 0.02, dtype=np.float32)
    )
    diagonal_samples = gp.sample_posterior(
        X_test[:5], n_samples=2, method="diagonal", rng=np.random.default_rng(seed + 1)
    )
    pathwise_samples = gp.sample_posterior(
        X_test[:5],
        n_samples=2,
        method="pathwise",
        n_rff_features=32,
        rng=np.random.default_rng(seed + 2),
    )
    sample_info = dict(gp.backend_sample_info)
    reference = np.asarray(gp.predict(X_test[:6]).mean, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix=f"mojogp_sanity_{surface}_") as tmp_dir:
        path = Path(tmp_dir) / surface
        gp.save(path)
        _release_provider_for_load(gp, surface)
        loaded = SingleOutputGP.load(path, kernel=_kernel_for(surface))
        loaded_mean = np.asarray(loaded.predict(X_test[:6]).mean, dtype=np.float32)

    payload: dict[str, Any] = {
        "surface": surface,
        "method": method,
        "training_route": gp.backend_train_info.get("training_route"),
        "exact_route": exact_info.get("actual_prediction_route"),
        "love_route": love_info.get("actual_variance_route"),
        "love_fallback": bool(love_info.get("fallback_used", False)),
        "observed_variance_delta_min": float(
            np.min(observed.variance - love.variance)
        ),
        "diagonal_sample_shape": tuple(diagonal_samples.shape),
        "pathwise_sample_shape": tuple(pathwise_samples.shape),
        "pathwise_route": sample_info.get("actual_sampling_route"),
        "save_load_max_abs_diff": float(np.max(np.abs(loaded_mean - reference))),
    }
    _finite_prediction(payload, exact.mean, exact.variance)
    _finite_prediction(payload, love.mean, love.variance)
    assert np.all(np.isfinite(diagonal_samples))
    assert np.all(np.isfinite(pathwise_samples))
    return payload


def _icm_workflow(surface: str, method: str, seed: int) -> dict[str, Any]:
    mixed = surface == "icm_mixed"
    X, Y, X_test = _multi_data(mixed=mixed, seed=seed)
    gp = MultiOutputGP(
        kernel=_kernel_for(surface),
        task_rank=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp.fit(X, Y, method=method, max_iterations=3, learning_rate=0.03, verbose=False)
    exact = gp.predict(X_test[:6], variance_method="exact")
    exact_info = dict(gp.backend_predict_info)
    love = gp.predict(X_test[:6], variance_method="love")
    love_info = dict(gp.backend_predict_info)
    observed = gp.predict_observed(
        X_test[:6], observation_noise=np.full((6, 2), 0.02, dtype=np.float32)
    )
    samples = gp.sample_posterior(
        X_test[:5],
        n_samples=2,
        method="pathwise",
        n_rff_features=32,
        rng=np.random.default_rng(seed + 3),
    )
    sample_info = dict(gp.backend_sample_info)
    reference = np.asarray(gp.predict(X_test[:6]).mean, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix=f"mojogp_sanity_{surface}_") as tmp_dir:
        path = Path(tmp_dir) / surface
        gp.save(path)
        _release_provider_for_load(gp, surface)
        loaded = MultiOutputGP.load(path, kernel=_kernel_for(surface))
        loaded_mean = np.asarray(loaded.predict(X_test[:6]).mean, dtype=np.float32)

    payload: dict[str, Any] = {
        "surface": surface,
        "method": method,
        "training_route": gp.backend_train_info.get("training_route"),
        "exact_route": exact_info.get("actual_prediction_route"),
        "love_route": love_info.get("actual_variance_route"),
        "love_fallback": bool(love_info.get("fallback_used", False)),
        "observed_variance_delta_min": float(np.min(observed.variance - love.variance)),
        "pathwise_sample_shape": tuple(samples.shape),
        "pathwise_route": sample_info.get("actual_sampling_route"),
        "save_load_max_abs_diff": float(np.max(np.abs(loaded_mean - reference))),
    }
    _finite_prediction(payload, exact.mean, exact.variance)
    _finite_prediction(payload, love.mean, love.variance)
    assert np.all(np.isfinite(samples))
    return payload


def _lmc_workflow(surface: str, method: str, seed: int) -> dict[str, Any]:
    mixed = surface == "lmc_mixed"
    X, Y, X_test = _multi_data(mixed=mixed, seed=seed)
    gp = MultiOutputLMCGP(
        kernels=_lmc_kernels_for(surface),
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    gp.fit(X, Y, method=method, max_iterations=3, learning_rate=0.03, verbose=False)
    exact = gp.predict(X_test[:6], variance_method="exact")
    exact_info = dict(gp.backend_predict_info)
    love = gp.predict(X_test[:6], variance_method="love")
    love_info = dict(gp.backend_predict_info)
    observed = gp.predict_observed(
        X_test[:6], observation_noise=np.full((6, 2), 0.02, dtype=np.float32)
    )
    samples = gp.sample_posterior(
        X_test[:5],
        n_samples=2,
        method="pathwise",
        n_rff_features=32,
        rng=np.random.default_rng(seed + 4),
    )
    sample_info = dict(gp.backend_sample_info)
    reference = np.asarray(gp.predict(X_test[:6]).mean, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix=f"mojogp_sanity_{surface}_") as tmp_dir:
        path = Path(tmp_dir) / surface
        gp.save(path)
        loaded = MultiOutputLMCGP.load(path)
        loaded_mean = np.asarray(loaded.predict(X_test[:6]).mean, dtype=np.float32)

    payload: dict[str, Any] = {
        "surface": surface,
        "method": method,
        "training_route": gp.backend_train_info.get("training_route"),
        "exact_route": exact_info.get("actual_prediction_route"),
        "love_route": love_info.get("actual_variance_route"),
        "love_fallback": bool(love_info.get("fallback_used", False)),
        "observed_variance_delta_min": float(np.min(observed.variance - love.variance)),
        "pathwise_sample_shape": tuple(samples.shape),
        "pathwise_route": sample_info.get("actual_sampling_route"),
        "save_load_max_abs_diff": float(np.max(np.abs(loaded_mean - reference))),
    }
    _finite_prediction(payload, exact.mean, exact.variance)
    _finite_prediction(payload, love.mean, love.variance)
    assert np.all(np.isfinite(samples))
    return payload


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    warnings.simplefilter("ignore")
    surface = str(payload["surface"])
    method = str(payload["method"])
    seed = int(payload.get("seed", 0))
    if surface.startswith("single"):
        result = _single_workflow(surface, method, seed)
    elif surface.startswith("icm"):
        result = _icm_workflow(surface, method, seed)
    elif surface.startswith("lmc"):
        result = _lmc_workflow(surface, method, seed)
    else:
        raise ValueError(f"unknown sanity surface: {surface}")
    return {"payload": result}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
