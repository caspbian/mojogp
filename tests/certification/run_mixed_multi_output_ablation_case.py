"""Run one mixed multi-output ablation case in an isolated process."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mojogp import MultiOutputGP, MultiOutputLMCGP
from tests.certification.categorical_ablation_utils import (
    centered_correlation,
    continuous_kernel,
    generate_mixed_multi_output_ablation_dataset,
    lmc_continuous_kernels,
    lmc_mixed_kernels,
    mixed_kernel,
    rmse,
    route_summary,
)
from tests.shared.subprocess_harness import run_child_main


def _fit_predict(gp: Any, X_train, Y_train, X_test, *, method: str):
    gp.fit(
        X_train,
        Y_train,
        method=method,
        max_iterations=5,
        learning_rate=0.03,
        verbose=False,
    )
    mean, std = gp.predict(X_test, return_std=True)
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _build_gp(surface: str, kernel_name: str, *, continuous_dim: int, levels: int):
    if surface == "icm":
        return MultiOutputGP(
            kernel=mixed_kernel(
                kernel_name, continuous_dim=continuous_dim, levels=levels
            ),
            task_rank=1,
            num_probes=2,
            max_cg_iterations=20,
            max_tridiag_iterations=8,
            preconditioner_rank=5,
        )
    if surface == "lmc":
        return MultiOutputLMCGP(
            kernels=lmc_mixed_kernels(
                kernel_name, continuous_dim=continuous_dim, levels=levels
            ),
            num_probes=2,
            max_cg_iterations=20,
            max_tridiag_iterations=8,
            preconditioner_rank=5,
        )
    raise ValueError(f"unknown mixed multi-output surface: {surface}")


def _build_baseline_gp(surface: str, *, continuous_dim: int):
    if surface == "icm":
        return MultiOutputGP(
            kernel=continuous_kernel(continuous_dim),
            task_rank=1,
            num_probes=2,
            max_cg_iterations=20,
            max_tridiag_iterations=8,
            preconditioner_rank=5,
        )
    if surface == "lmc":
        return MultiOutputLMCGP(
            kernels=lmc_continuous_kernels(continuous_dim),
            num_probes=2,
            max_cg_iterations=20,
            max_tridiag_iterations=8,
            preconditioner_rank=5,
        )
    raise ValueError(f"unknown mixed multi-output surface: {surface}")


def _prediction_route_for_surface(surface: str) -> str:
    if surface == "icm":
        return "predict_multi_output_mixed"
    if surface == "lmc":
        return "predict_lmc_mixed"
    raise ValueError(f"unknown mixed multi-output surface: {surface}")


def _release_for_loaded_prediction(gp: Any, surface: str) -> None:
    if surface == "icm":
        gp._destroy_persistent_provider()


def _load_saved_gp(surface: str, save_path: Path, kernel_name: str, dataset):
    if surface == "icm":
        return MultiOutputGP.load(
            save_path,
            kernel=mixed_kernel(
                kernel_name,
                continuous_dim=dataset.continuous_dim,
                levels=dataset.levels,
            ),
        )
    if surface == "lmc":
        return MultiOutputLMCGP.load(save_path)
    raise ValueError(f"unknown mixed multi-output surface: {surface}")


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    warnings.simplefilter("ignore")
    surface = str(payload["surface"])
    variant = str(payload.get("variant", "mixed"))
    kernel_name = str(payload.get("kernel", "ehh"))
    method = str(payload["method"])
    seed = int(payload.get("seed", 0))
    dataset = generate_mixed_multi_output_ablation_dataset(seed=seed)

    if variant == "continuous":
        baseline_gp = _build_baseline_gp(surface, continuous_dim=dataset.continuous_dim)
        baseline_mean, baseline_std = _fit_predict(
            baseline_gp,
            dataset.X_train,
            dataset.Y_train,
            dataset.X_test,
            method=method,
        )
        routes = route_summary(baseline_gp)
        metrics = {
            "baseline_rmse": rmse(dataset.F_test, baseline_mean),
            "min_predictive_std": float(np.min(baseline_std)),
            "max_predictive_std": float(np.max(baseline_std)),
        }
        passed = (
            routes["training_route"] == method
            and bool(np.all(np.isfinite(baseline_mean)))
            and bool(np.all(np.isfinite(baseline_std)))
            and bool(np.all(baseline_std >= 0.0))
        )
        return {
            "payload": {
                "suite": "mixed_multi_output_ablation",
                "surface": f"multi_output_{surface}_continuous",
                "variant": variant,
                "kernel": "continuous",
                "method": method,
                "n_train": int(dataset.X_train.shape[0]),
                "n_test": int(dataset.X_test.shape[0]),
                "num_tasks": int(dataset.num_tasks),
                "seed": seed,
                "metrics": metrics,
                "routes": routes,
                "passed": bool(passed),
            }
        }

    if variant != "mixed":
        raise ValueError(f"unknown mixed multi-output ablation variant: {variant}")

    mixed_gp = _build_gp(
        surface,
        kernel_name,
        continuous_dim=dataset.continuous_dim,
        levels=dataset.levels,
    )
    mixed_mean, mixed_std = _fit_predict(
        mixed_gp,
        dataset.X_train,
        dataset.Y_train,
        dataset.X_test,
        method=method,
    )
    mixed_rmse = rmse(dataset.F_test, mixed_mean)

    category_probe_mean = np.asarray(
        mixed_gp.predict(dataset.category_probe_X).mean, dtype=np.float32
    )
    category_effect_corrs = [
        centered_correlation(category_probe_mean[:, task], dataset.true_category_effects)
        for task in range(dataset.num_tasks)
    ]

    shuffled_mean = np.asarray(
        mixed_gp.predict(dataset.X_test_shuffled).mean, dtype=np.float32
    )
    shuffled_rmse = rmse(dataset.F_test, shuffled_mean)
    routes = route_summary(mixed_gp)

    reference_for_load = np.asarray(
        mixed_gp.predict(dataset.X_test[:16]).mean, dtype=np.float32
    )
    with tempfile.TemporaryDirectory(prefix=f"mojogp_mixed_{surface}_cert_") as temp_dir:
        save_path = Path(temp_dir) / f"{surface}_{kernel_name}_{method}"
        mixed_gp.save(save_path)
        _release_for_loaded_prediction(mixed_gp, surface)
        loaded = _load_saved_gp(surface, save_path, kernel_name, dataset)
        loaded_mean = np.asarray(
            loaded.predict(dataset.X_test[:16]).mean, dtype=np.float32
        )
    save_load_max_abs_diff = float(np.max(np.abs(loaded_mean - reference_for_load)))

    metrics = {
        "mixed_rmse": float(mixed_rmse),
        "shuffled_rmse": float(shuffled_rmse),
        "mixed_to_shuffled_rmse_ratio": float(mixed_rmse / shuffled_rmse),
        "min_category_effect_corr": float(np.min(category_effect_corrs)),
        "mean_category_effect_corr": float(np.mean(category_effect_corrs)),
        "save_load_max_abs_diff": save_load_max_abs_diff,
        "min_predictive_std": float(np.min(mixed_std)),
        "max_predictive_std": float(np.max(mixed_std)),
    }
    expected_prediction_route = _prediction_route_for_surface(surface)
    passed = (
        metrics["mixed_to_shuffled_rmse_ratio"] <= 0.95
        and metrics["min_category_effect_corr"] >= 0.5
        and metrics["save_load_max_abs_diff"] <= 1e-3
        and bool(routes["backend_prediction_used"])
        and not bool(routes["fallback_used"])
        and routes["training_route"] == method
        and routes["prediction_route"] == expected_prediction_route
        and bool(np.all(np.isfinite(mixed_mean)))
        and bool(np.all(np.isfinite(mixed_std)))
        and bool(np.all(mixed_std >= 0.0))
    )
    return {
        "payload": {
            "suite": "mixed_multi_output_ablation",
            "surface": f"multi_output_{surface}_mixed",
            "variant": variant,
            "kernel": kernel_name,
            "method": method,
            "n_train": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "num_tasks": int(dataset.num_tasks),
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
