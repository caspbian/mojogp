"""Single-output real-data benchmarks.

These are the first practitioner-oriented real-data validation surfaces in the
benchmark harness. They intentionally start with offline deterministic datasets.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from mojogp import SingleOutputGP, RBF

from tests.shared.benchmarking.environment import assert_gpu_available, assert_gpu_was_used, requires_cuda
from tests.shared.benchmarking.gpu_memory import (
    GPUMemoryMonitor,
    get_torch_memory_stats,
    reset_torch_memory_stats,
)
from tests.shared.benchmarking.metrics import (
    calibration_coverage,
    calibration_error,
    crps_gaussian,
    mae,
    mean_standardized_log_loss,
    rmse,
    r_squared,
    sharpness,
    interval_width,
)
from tests.shared.benchmarking.real_datasets import load_diabetes_regression
from tests.shared.benchmarking.report import print_result, save_system_result
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)


@pytest.mark.minimal
@pytest.mark.single_output
@pytest.mark.accuracy
@requires_cuda
def test_diabetes_rbf_ard_real_data(results_dir):
    assert_gpu_available()
    data = load_diabetes_regression()

    reset_torch_memory_stats()
    gpu_monitor = GPUMemoryMonitor(interval=0.1)
    gpu_monitor.start()
    tracemalloc.start()

    gp = SingleOutputGP(RBF(ard=True))
    fit_start = time.perf_counter()
    gp.fit(
        data.X_train,
        data.y_train,
        method="materialized",
        max_iterations=100,
        learning_rate=0.03,
        verbose=False,
    )
    training_time_s = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    pred = gp.predict(data.X_test)
    prediction_time_s = time.perf_counter() - pred_start

    gpu_monitor.stop()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del current

    memory_stats = gpu_monitor.get_stats()
    memory_stats.update(get_torch_memory_stats())
    memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)

    mean = np.asarray(pred.mean, dtype=np.float32)
    variance = np.asarray(pred.variance, dtype=np.float32)
    std = np.sqrt(np.maximum(variance, 1e-10))
    learned_params = gp.get_learned_params()
    learned_lengthscales = [
        value
        for name, value in learned_params.items()
        if "_ls_" in name or name.endswith("lengthscale")
    ]
    learned_outputscale = next(
        (
            value
            for name, value in learned_params.items()
            if name.endswith("outputscale")
        ),
        1.0,
    )

    ridge = Ridge(alpha=1.0)
    ridge.fit(data.X_train, data.y_train)
    ridge_pred = ridge.predict(data.X_test).astype(np.float32)

    accuracy = AccuracyResult(
        rmse=rmse(data.y_test, mean),
        mae=mae(data.y_test, mean),
        r_squared=r_squared(data.y_test, mean),
        crps=crps_gaussian(data.y_test, mean, std),
        msll=mean_standardized_log_loss(
            data.y_test,
            mean,
            std,
            y_train_mean=float(np.mean(data.y_train)),
            y_train_std=float(np.std(data.y_train)),
        ),
        calibration_coverage=calibration_coverage(data.y_test, mean, std),
        calibration_error=calibration_error(data.y_test, mean, std),
        sharpness=sharpness(std),
        interval_width_95=interval_width(mean, std),
    )
    result = BenchmarkResult(
        config={
            "dataset": data.name,
            "kernel": "rbf_ard",
            "method": "materialized",
            "n": int(data.X_train.shape[0]),
            "d": int(data.X_train.shape[1]),
            "ridge_rmse": float(rmse(data.y_test, ridge_pred)),
            "gain_vs_ridge": float(rmse(data.y_test, ridge_pred) - accuracy.rmse),
        },
        accuracy=accuracy,
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=prediction_time_s,
            prediction_variance_time_s=prediction_time_s,
            end_to_end_time_s=training_time_s + prediction_time_s,
            iterations_run=100,
            max_iterations=100,
            early_stopped=False,
            ms_per_iteration=training_time_s / 100.0 * 1000.0,
        ),
        memory=MemoryResult(
            gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
            gpu_min_mb=memory_stats.get("min_mb", 0.0),
            gpu_max_mb=memory_stats.get("max_mb", 0.0),
            gpu_var_mb=memory_stats.get("var_mb", 0.0),
            torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
            torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
            cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
            measurement_method=memory_stats.get("method", "none"),
            num_samples=int(memory_stats.get("samples", 0)),
        ),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(learned_lengthscales))),
            learned_noise=float(learned_params["noise"]),
            learned_outputscale=float(learned_outputscale),
            final_nll=float(np.asarray(gp.training_result.nll_history)[-1]),
        ),
    )

    print_result(result)
    save_system_result(result, results_dir, "single_output_real_data")
    assert_gpu_was_used(result)
    assert accuracy.r_squared > 0.2, f"Real-data R^2 too low: {accuracy.r_squared:.3f}"
    assert accuracy.rmse <= result.config["ridge_rmse"] * 1.15, (
        f"MojoGP regressed too far versus ridge on diabetes: gp={accuracy.rmse:.4f}, ridge={result.config['ridge_rmse']:.4f}"
    )
