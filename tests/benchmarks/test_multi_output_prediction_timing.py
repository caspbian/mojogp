from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tests.system_benchmarks.run_multi_output_scaling_case import (
    _build_result,
    _merge_mojogp_phase_memory_stats,
)
from tests.system_benchmarks.test_mojogp_route_parity_harness import _build_multi_output_result


def _multi_output_dataset() -> SimpleNamespace:
    x_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    x_test = np.array([[0.5], [1.5]], dtype=np.float32)
    y_train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    y_test = np.array([[0.4, 1.4], [1.4, 2.4]], dtype=np.float32)
    return SimpleNamespace(
        X_train=x_train,
        X_test=x_test,
        Y_train=y_train,
        Y_test=y_test,
        F_test=y_test,
        true_params={"task_correlation": "medium"},
    )


def test_multi_output_scaling_builder_records_combined_prediction_once():
    dataset = _multi_output_dataset()
    result = _build_result(
        dataset=dataset,
        framework="mojogp",
        model_type="MultiOutputGP",
        training_method="materialized",
        method="materialized",
        prediction_mode="exact",
        max_iterations=10,
        training_time_s=1.2,
        prediction_total_time_s=0.4,
        prediction_mean_time_s=None,
        prediction_variance_time_s=None,
        prediction_timing_quality="total_only_combined_call",
        iterations_run=8,
        early_stopped=False,
        memory_stats={"method": "none", "samples": 0},
        mean=np.asarray(dataset.F_test, dtype=np.float32),
        variance=np.full_like(dataset.F_test, 0.25, dtype=np.float32),
        learned_lengthscale=1.0,
        learned_noise=0.1,
        learned_outputscale=1.5,
        final_nll=0.3,
        optimizer_config={},
        training_solver_config={},
        prediction_solver_config={},
        cg_telemetry={},
        tier="xsmall",
    )

    assert result.speed.prediction_mean_time_s == 0.0
    assert result.speed.prediction_variance_time_s == 0.4
    assert result.speed.end_to_end_time_s == 1.6
    assert result.config["prediction_timing_quality"] == "total_only_combined_call"


def test_multi_output_scaling_builder_records_direct_timing_and_route_memory():
    dataset = _multi_output_dataset()
    result = _build_result(
        dataset=dataset,
        framework="mojogp",
        model_type="MultiOutputGP",
        training_method="materialized",
        method="materialized",
        prediction_mode="love",
        max_iterations=10,
        training_time_s=1.2,
        prediction_total_time_s=0.4,
        prediction_mean_time_s=None,
        prediction_variance_time_s=None,
        prediction_timing_quality="total_only_combined_call",
        iterations_run=3,
        early_stopped=False,
        memory_stats={
            "method": "pynvml",
            "samples": 4,
            "max_mb": 300.0,
            "training_peak_gpu_mb": 220.0,
            "training_delta_gpu_mb": 120.0,
            "prediction_peak_gpu_mb": 260.0,
            "prediction_delta_gpu_mb": 60.0,
            "love_prediction_peak_gpu_mb": 260.0,
            "love_prediction_delta_gpu_mb": 60.0,
        },
        mean=np.asarray(dataset.F_test, dtype=np.float32),
        variance=np.full_like(dataset.F_test, 0.25, dtype=np.float32),
        learned_lengthscale=1.0,
        learned_noise=0.1,
        learned_outputscale=1.5,
        final_nll=0.3,
        optimizer_config={},
        training_solver_config={},
        prediction_solver_config={},
        cg_telemetry={},
        tier="xsmall",
        training_iter_times_ms=[10.0, 20.0, 40.0],
    )

    assert result.config["iter_timing_quality"] == "direct_per_iteration"
    assert result.speed.iter_timing_quality == "direct_per_iteration"
    assert result.speed.ms_per_iteration == 20.0
    assert result.speed.iter_time_p5_ms is not None
    assert result.speed.iter_time_p95_ms is not None
    assert result.speed.iter_times_ms == [10.0, 20.0, 40.0]
    assert result.config["phase_memory_quality"] == "phase_specific"
    assert result.memory.training_peak_gpu_mb == 220.0
    assert result.memory.prediction_peak_gpu_mb == 260.0
    assert result.memory.love_prediction_peak_gpu_mb == 260.0


def test_multi_output_mojogp_memory_merge_preserves_route_fields(monkeypatch):
    monkeypatch.setattr(
        "tests.system_benchmarks.run_multi_output_scaling_case.get_torch_memory_stats",
        lambda: {"torch_peak_mb": 7.0, "torch_current_mb": 3.0},
    )
    merged = _merge_mojogp_phase_memory_stats(
        {"max_mb": 100.0, "mean_mb": 80.0, "torch_peak_mb": 4.0},
        {"phase_peak_gpu_mb": 220.0, "phase_delta_gpu_mb": 120.0, "torch_peak_mb": 9.0},
        {"phase_peak_gpu_mb": 260.0, "phase_delta_gpu_mb": 60.0, "torch_peak_mb": 11.0},
        prediction_mode="exact",
    )

    assert merged["max_mb"] == 260.0
    assert merged["mean_mb"] == 260.0
    assert merged["torch_peak_mb"] == 11.0
    assert merged["training_peak_gpu_mb"] == 220.0
    assert merged["training_delta_gpu_mb"] == 120.0
    assert merged["prediction_peak_gpu_mb"] == 260.0
    assert merged["prediction_delta_gpu_mb"] == 60.0
    assert merged["exact_prediction_peak_gpu_mb"] == 260.0
    assert merged["exact_prediction_delta_gpu_mb"] == 60.0


def test_route_parity_builder_records_combined_prediction_once():
    dataset = _multi_output_dataset()
    train_result = SimpleNamespace(
        params=np.array([1.0, 1.2], dtype=np.float32),
        noise_per_task=np.array([0.1, 0.2], dtype=np.float32),
        effective_scales=np.array([1.5, 1.6], dtype=np.float32),
        final_nll=0.25,
    )
    result = _build_multi_output_result(
        dataset=dataset,
        method="matrix_free",
        prediction_mode="love",
        training_time_s=2.0,
        prediction_time_s=0.5,
        iterations_run=6,
        memory_stats={"method": "none", "samples": 0},
        mean=np.asarray(dataset.F_test, dtype=np.float32),
        variance=np.full_like(dataset.F_test, 0.5, dtype=np.float32),
        train_result=train_result,
    )

    assert result.speed.prediction_mean_time_s == 0.0
    assert result.speed.prediction_variance_time_s == 0.5
    assert result.speed.end_to_end_time_s == 2.5
    assert result.config["prediction_timing_quality"] == "total_only_combined_call"
