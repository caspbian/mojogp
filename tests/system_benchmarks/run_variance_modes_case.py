"""Run one single-output variance-mode benchmark case in isolation."""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np

from mojogp import SingleOutputGP, RBF
from tests.shared.subprocess_harness import run_child_main

from tests.shared.benchmarking.gpu_memory import GPUMemoryMonitor, get_torch_memory_stats, reset_torch_memory_stats
from tests.shared.benchmarking.report import save_result_artifact
from tests.system_benchmarks.test_single_output_variance_modes_harness import _benchmark_from_prediction, _build_love_dataset


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    case = str(payload["case"])
    results_dir = Path(str(payload["results_dir"]))
    if case == "love_vs_exact":
        method = str(payload["method"])
        dataset = _build_love_dataset(seed=42 if method == "materialized" else 123)
        reset_torch_memory_stats()
        monitor = GPUMemoryMonitor(interval=0.1)
        monitor.start()
        tracemalloc.start()
        gp = SingleOutputGP(RBF())
        fit_start = time.perf_counter()
        train_result = gp.fit(
            dataset.X_train,
            dataset.y_train,
            method=method,
            max_iterations=40 if method == "materialized" else 30,
            learning_rate=0.03,
            verbose=False,
        )
        training_time_s = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        pred_exact = gp.predict(dataset.X_test, variance_method="exact")
        pred_love = gp.predict(dataset.X_test, variance_method="love")
        prediction_total_time_s = time.perf_counter() - pred_start
        monitor.stop()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del current
        memory_stats = monitor.get_stats()
        memory_stats.update(get_torch_memory_stats())
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)
        learned_params = gp.get_learned_params()
        mean_diff_rmse = float(np.sqrt(np.mean((pred_love.mean - pred_exact.mean) ** 2)))
        var_ratio = float(
            np.mean(np.asarray(pred_love.variance, dtype=np.float32))
            / (np.mean(np.asarray(pred_exact.variance, dtype=np.float32)) + 1e-6)
        )
        benchmark = _benchmark_from_prediction(
            dataset=dataset,
            method=method,
            prediction_mode="love",
            training_time_s=training_time_s,
            prediction_total_time_s=prediction_total_time_s,
            iterations_run=int(train_result.iterations),
            max_iterations=40 if method == "materialized" else 30,
            memory_stats=memory_stats,
            mean=np.asarray(pred_love.mean, dtype=np.float32),
            variance=np.asarray(pred_love.variance, dtype=np.float32),
            learned_params=learned_params,
            final_nll=float(train_result.nll),
            baseline_config={
                "baseline_type": "mojogp_exact_prediction",
                "mean_rmse_vs_exact": mean_diff_rmse,
                "variance_mean_ratio_vs_exact": var_ratio,
            },
        )
    else:
        raise ValueError(f"Unknown case '{case}'")

    result_path = save_result_artifact(benchmark, results_dir, "love_variance_comparison")
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
