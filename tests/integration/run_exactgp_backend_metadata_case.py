"""Child entrypoints for ExactGP backend metadata subprocess tests."""

from __future__ import annotations

import os
import tempfile

import numpy as np

from mojogp import SingleOutputGP, RBF
from tests.shared.subprocess_harness import run_child_main


def _predict_reports_route() -> dict[str, object]:
    rng = np.random.RandomState(7)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:8]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="materialized",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    pred = gp.predict(X_test, variance_method="exact")
    return {"shape": list(pred.mean.shape), "info": dict(gp.backend_predict_info)}


def _predict_override_route() -> dict[str, object]:
    rng = np.random.RandomState(17)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:12]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="materialized",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    pred = gp.predict(
        X_test,
                variance_method="exact",
        method="matrix_free",
    )
    return {"shape": list(pred.mean.shape), "info": dict(gp.backend_predict_info)}


def _exact_prediction_parity(method: str) -> dict[str, object]:
    rng = np.random.RandomState(9)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:16]
    init_params = np.array([1.0, 1.0], dtype=np.float32)

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        method=method,
        max_iterations=1,
        learning_rate=1e-8,
        initial_params=init_params,
        initial_noise=0.1,
        num_probes=4,
        max_cg_iterations=25,
        preconditioner_rank=8,
        max_tridiag_iterations=10,
        preconditioner="greedy",
        verbose=False,
    )
    pred = gp.predict(X_test, variance_method="exact")
    return {
        "mean": pred.mean,
        "var": pred.variance,
        "info": dict(gp.backend_predict_info),
    }


def _blocked_blas_exact_prediction_parity() -> dict[str, object]:
    rng = np.random.RandomState(11)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = rng.randn(128, 3).astype(np.float32)
    init_params = np.array([1.0, 1.0], dtype=np.float32)

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        method="matrix_free",
        max_iterations=1,
        learning_rate=1e-8,
        initial_params=init_params,
        initial_noise=0.1,
        num_probes=4,
        max_cg_iterations=25,
        preconditioner_rank=0,
        max_tridiag_iterations=10,
        preconditioner="greedy",
        use_preconditioner=False,
        verbose=False,
    )

    env_keys = [
        "MOJOGP_EXACT_BLOCKED_BLAS_MATVEC",
        "MOJOGP_EXACT_BLOCKED_BLAS_TILE_COLS",
        "MOJOGP_EXACT_BLOCKED_BLAS_MIN_COLS",
    ]
    previous_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["MOJOGP_EXACT_BLOCKED_BLAS_MATVEC"] = "0"
        baseline = gp.predict(
            X_test,
                        variance_method="exact",
            method="matrix_free",
            max_cg_iterations=80,
            cg_tolerance=1e-6,
            preconditioner_rank=0,
            exact_prediction_block_cols=128,
        )
        baseline_info = dict(gp.backend_predict_info)

        os.environ["MOJOGP_EXACT_BLOCKED_BLAS_MATVEC"] = "1"
        os.environ["MOJOGP_EXACT_BLOCKED_BLAS_TILE_COLS"] = "1024"
        os.environ["MOJOGP_EXACT_BLOCKED_BLAS_MIN_COLS"] = "1"
        blocked = gp.predict(
            X_test,
                        variance_method="exact",
            method="matrix_free",
            max_cg_iterations=80,
            cg_tolerance=1e-6,
            preconditioner_rank=0,
            exact_prediction_block_cols=128,
        )
        blocked_info = dict(gp.backend_predict_info)

        for key in env_keys:
            os.environ.pop(key, None)
        default = gp.predict(
            X_test,
                        variance_method="exact",
            method="matrix_free",
            max_cg_iterations=80,
            cg_tolerance=1e-6,
            preconditioner_rank=0,
            exact_prediction_block_cols=128,
        )
        default_info = dict(gp.backend_predict_info)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    mean_diff = np.max(np.abs(baseline.mean - blocked.mean))
    var_diff = np.max(np.abs(baseline.variance - blocked.variance))
    default_blocked_mean_diff = np.max(np.abs(default.mean - blocked.mean))
    default_blocked_var_diff = np.max(np.abs(default.variance - blocked.variance))
    return {
        "shape": list(blocked.mean.shape),
        "baseline_info": baseline_info,
        "blocked_info": blocked_info,
        "default_info": default_info,
        "mean_max_abs_diff": float(mean_diff),
        "variance_max_abs_diff": float(var_diff),
        "default_blocked_mean_max_abs_diff": float(default_blocked_mean_diff),
        "default_blocked_variance_max_abs_diff": float(default_blocked_var_diff),
        "baseline_mean": baseline.mean,
        "blocked_mean": blocked.mean,
        "default_mean": default.mean,
        "baseline_var": baseline.variance,
        "blocked_var": blocked.variance,
        "default_var": default.variance,
    }


def _blocked_blas_repeated_and_save_load() -> dict[str, object]:
    rng = np.random.RandomState(13)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = rng.randn(128, 3).astype(np.float32)
    init_params = np.array([1.0, 1.0], dtype=np.float32)

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        method="matrix_free",
        max_iterations=1,
        learning_rate=1e-8,
        initial_params=init_params,
        initial_noise=0.1,
        num_probes=4,
        max_cg_iterations=25,
        preconditioner_rank=0,
        max_tridiag_iterations=10,
        preconditioner="greedy",
        use_preconditioner=False,
        verbose=False,
    )

    predict_kwargs = {
        "variance_method": "exact",
        "method": "matrix_free",
        "max_cg_iterations": 80,
        "cg_tolerance": 1e-6,
        "preconditioner_rank": 0,
        "exact_prediction_block_cols": 128,
    }
    env_keys = [
        "MOJOGP_EXACT_BLOCKED_BLAS_MATVEC",
        "MOJOGP_EXACT_BLOCKED_BLAS_TILE_COLS",
        "MOJOGP_EXACT_BLOCKED_BLAS_MIN_COLS",
    ]
    previous_env = {key: os.environ.get(key) for key in env_keys}
    try:
        os.environ["MOJOGP_EXACT_BLOCKED_BLAS_MATVEC"] = "1"
        os.environ["MOJOGP_EXACT_BLOCKED_BLAS_TILE_COLS"] = "1024"
        os.environ["MOJOGP_EXACT_BLOCKED_BLAS_MIN_COLS"] = "1"

        first = gp.predict(X_test, **predict_kwargs)
        first_info = dict(gp.backend_predict_info)
        second = gp.predict(X_test, **predict_kwargs)
        second_info = dict(gp.backend_predict_info)

        with tempfile.TemporaryDirectory() as tmp:
            save_path = os.path.join(tmp, "exactgp_blocked_blas")
            gp.save(save_path)
            loaded = SingleOutputGP.load(save_path)
            loaded_pred = loaded.predict(X_test, **predict_kwargs)
            loaded_info = dict(loaded.backend_predict_info)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return {
        "shape": list(second.mean.shape),
        "first_info": first_info,
        "second_info": second_info,
        "loaded_info": loaded_info,
        "repeat_mean_max_abs_diff": float(np.max(np.abs(first.mean - second.mean))),
        "repeat_variance_max_abs_diff": float(
            np.max(np.abs(first.variance - second.variance))
        ),
        "loaded_mean_max_abs_diff": float(
            np.max(np.abs(second.mean - loaded_pred.mean))
        ),
        "loaded_variance_max_abs_diff": float(
            np.max(np.abs(second.variance - loaded_pred.variance))
        ),
    }


def _matrix_free_exact_prediction_no_fallback() -> dict[str, object]:
    rng = np.random.RandomState(27)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:10]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="matrix_free",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    pred = gp.predict(X_test, variance_method="exact")
    return {"shape": list(pred.mean.shape), "info": dict(gp.backend_predict_info)}


def _materialized_repeated_predict_reuses_provider_state() -> dict[str, object]:
    rng = np.random.RandomState(37)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:10]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="materialized",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    first = gp.predict(X_test, variance_method="exact")
    first_info = dict(gp.backend_predict_info)
    second = gp.predict(X_test, variance_method="exact")
    return {
        "shape": list(second.mean.shape),
        "info": dict(gp.backend_predict_info),
        "first_info": first_info,
        "mean_match": bool(np.allclose(first.mean, second.mean, atol=1e-5, rtol=1e-5)),
        "var_match": bool(
            np.allclose(first.variance, second.variance, atol=1e-5, rtol=1e-5)
        ),
    }


def _predict_default_block_cap(method: str) -> dict[str, object]:
    rng = np.random.RandomState(47)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = rng.randn(600, 3).astype(np.float32)

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method=method,
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    pred = gp.predict(X_test, variance_method="exact")
    return {"shape": list(pred.mean.shape), "info": dict(gp.backend_predict_info)}


def _materialized_repeated_love_predict_reuses_root_state() -> dict[str, object]:
    rng = np.random.RandomState(53)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:10]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="materialized",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    first = gp.predict(X_test, variance_method="love")
    first_info = dict(gp.backend_predict_info)
    second = gp.predict(X_test, variance_method="love")
    return {
        "shape": list(second.mean.shape),
        "info": dict(gp.backend_predict_info),
        "first_info": first_info,
        "mean_match": bool(np.allclose(first.mean, second.mean, atol=1e-5, rtol=1e-5)),
        "var_match": bool(
            np.allclose(first.variance, second.variance, atol=1e-5, rtol=1e-5)
        ),
    }


def _love_first_predict_uses_exact_alpha_defaults() -> dict[str, object]:
    rng = np.random.RandomState(59)
    X = rng.randn(2000, 5).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:16]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="materialized",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    pred = gp.predict(X_test, variance_method="love")
    return {"shape": list(pred.mean.shape), "info": dict(gp.backend_predict_info)}


def _prepared_prediction_cache_reuses_device_state() -> dict[str, object]:
    rng = np.random.RandomState(61)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:12]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="materialized",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
    )
    cache_info = gp.prepare_prediction_cache(
        variance_method="love",
        max_root_decomposition_size=9,
    )
    first = gp.predict(
        X_test,
                variance_method="love",
        max_root_decomposition_size=9,
    )
    first_info = dict(gp.backend_predict_info)
    second = gp.predict(
        X_test,
                variance_method="love",
        max_root_decomposition_size=9,
    )
    return {
        "shape": list(second.mean.shape),
        "cache_info": cache_info,
        "first_info": first_info,
        "info": dict(gp.backend_predict_info),
        "mean_match": bool(np.allclose(first.mean, second.mean, atol=1e-5, rtol=1e-5)),
        "var_match": bool(
            np.allclose(first.variance, second.variance, atol=1e-5, rtol=1e-5)
        ),
    }


def _fit_prepares_prediction_cache() -> dict[str, object]:
    rng = np.random.RandomState(67)
    X = rng.randn(2000, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    X_test = X[:12]

    gp = SingleOutputGP(kernel=RBF(), verbose=False)
    gp.fit(
        X,
        y,
        max_iterations=1,
        learning_rate=0.01,
        method="materialized",
        num_probes=2,
        max_cg_iterations=15,
        preconditioner_rank=6,
        max_tridiag_iterations=7,
        preconditioner="greedy",
        verbose=False,
        prepare_prediction_cache=True,
        prediction_cache_rank=9,
    )
    pred = gp.predict(
        X_test,
                variance_method="love",
        max_root_decomposition_size=9,
    )
    return {"shape": list(pred.mean.shape), "info": dict(gp.backend_predict_info)}


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    case = str(payload["case"])
    if case == "predict_reports_route":
        return {"payload": _predict_reports_route()}
    if case == "predict_override_route":
        return {"payload": _predict_override_route()}
    if case == "exact_prediction_parity":
        return {"payload": _exact_prediction_parity(str(payload["method"]))}
    if case == "blocked_blas_exact_prediction_parity":
        return {"payload": _blocked_blas_exact_prediction_parity()}
    if case == "blocked_blas_repeated_and_save_load":
        return {"payload": _blocked_blas_repeated_and_save_load()}
    if case == "matrix_free_exact_prediction_no_fallback":
        return {"payload": _matrix_free_exact_prediction_no_fallback()}
    if case == "materialized_repeated_predict_reuses_provider_state":
        return {"payload": _materialized_repeated_predict_reuses_provider_state()}
    if case == "predict_default_block_cap":
        return {"payload": _predict_default_block_cap(str(payload["method"]))}
    if case == "materialized_repeated_love_predict_reuses_root_state":
        return {"payload": _materialized_repeated_love_predict_reuses_root_state()}
    if case == "love_first_predict_uses_exact_alpha_defaults":
        return {"payload": _love_first_predict_uses_exact_alpha_defaults()}
    if case == "prepared_prediction_cache_reuses_device_state":
        return {"payload": _prepared_prediction_cache_reuses_device_state()}
    if case == "fit_prepares_prediction_cache":
        return {"payload": _fit_prepares_prediction_cache()}
    raise ValueError(f"unknown case: {case}")


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
