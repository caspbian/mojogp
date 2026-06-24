"""Run one categorical SingleOutputGP ablation case in an isolated process."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np

from mojogp import SingleOutputGP
from tests.certification.categorical_ablation_utils import (
    centered_correlation,
    continuous_kernel,
    generate_categorical_ablation_dataset,
    mixed_kernel,
    rmse,
    route_summary,
)
from tests.shared.subprocess_harness import run_child_main


def _fit_predict(gp: SingleOutputGP, X_train, y_train, X_test, *, method: str):
    gp.fit(
        X_train,
        y_train,
        method=method,
        max_iterations=8,
        learning_rate=0.04,
        verbose=False,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
    )
    mean, std = gp.predict(X_test, return_std=True)
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    warnings.simplefilter("ignore")
    kernel_name = str(payload["kernel"])
    method = str(payload["method"])
    seed = int(payload.get("seed", 0))
    dataset = generate_categorical_ablation_dataset(seed=seed)

    baseline_gp = SingleOutputGP(continuous_kernel(dataset.continuous_dim))
    baseline_mean, _baseline_std = _fit_predict(
        baseline_gp,
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        method=method,
    )
    baseline_rmse = rmse(dataset.f_test, baseline_mean)

    positive_kernel = mixed_kernel(
        kernel_name, continuous_dim=dataset.continuous_dim, levels=dataset.levels
    )
    mixed_gp = SingleOutputGP(positive_kernel)
    mixed_mean, mixed_std = _fit_predict(
        mixed_gp,
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        method=method,
    )
    mixed_rmse = rmse(dataset.f_test, mixed_mean)
    category_probe_mean = np.asarray(
        mixed_gp.predict(dataset.category_probe_X).mean, dtype=np.float32
    )
    category_effect_corr = centered_correlation(
        category_probe_mean, dataset.true_category_effects
    )

    # Negative control: keep the fitted model fixed but break the semantic link
    # between categorical level and response at prediction time. This avoids a
    # separate pathological shuffled-label training solve while still proving the
    # learned categorical mapping is being used.
    shuffled_mean = np.asarray(
        mixed_gp.predict(dataset.X_test_shuffled).mean, dtype=np.float32
    )
    shuffled_rmse = rmse(dataset.f_test, shuffled_mean)

    routes = route_summary(mixed_gp)
    reference_for_load = np.asarray(mixed_gp.predict(dataset.X_test[:16]).mean, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="mojogp_cat_cert_") as temp_dir:
        save_path = Path(temp_dir) / f"{kernel_name}_{method}"
        mixed_gp.save(save_path)
        # Loaded mixed SingleOutputGP prediction can require provider takeover;
        # release the original live provider before the loaded model predicts.
        mixed_gp._destroy_provider_info()
        loaded = SingleOutputGP.load(save_path, kernel=positive_kernel)
        loaded_mean = np.asarray(loaded.predict(dataset.X_test[:16]).mean, dtype=np.float32)
    save_load_max_abs_diff = float(
        np.max(np.abs(loaded_mean - reference_for_load))
    )
    metrics = {
        "baseline_rmse": float(baseline_rmse),
        "mixed_rmse": float(mixed_rmse),
        "shuffled_rmse": float(shuffled_rmse),
        "mixed_to_baseline_rmse_ratio": float(mixed_rmse / baseline_rmse),
        "mixed_to_shuffled_rmse_ratio": float(mixed_rmse / shuffled_rmse),
        "category_effect_corr": float(category_effect_corr),
        "save_load_max_abs_diff": save_load_max_abs_diff,
        "min_predictive_std": float(np.min(mixed_std)),
        "max_predictive_std": float(np.max(mixed_std)),
    }
    passed = (
        metrics["mixed_to_baseline_rmse_ratio"] <= 0.95
        and metrics["mixed_to_shuffled_rmse_ratio"] <= 0.95
        and metrics["category_effect_corr"] >= 0.5
        and metrics["save_load_max_abs_diff"] <= 1e-3
        and bool(routes["backend_prediction_used"])
        and not bool(routes["fallback_used"])
        and routes["training_route"] == method
        and routes["prediction_route"] == "predict_mixed"
        and bool(np.all(np.isfinite(mixed_mean)))
        and bool(np.all(np.isfinite(mixed_std)))
        and bool(np.all(mixed_std >= 0.0))
    )
    return {
        "payload": {
            "suite": "categorical_single_output_ablation",
            "surface": "single_output_mixed",
            "kernel": kernel_name,
            "method": method,
            "n_train": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "seed": seed,
            "metrics": metrics,
            "routes": routes,
            "passed": bool(passed),
        }
    }


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
