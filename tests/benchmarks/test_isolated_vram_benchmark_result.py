from __future__ import annotations

from tests.system_benchmarks.run_isolated_vram_measurement_case import _build_result


def test_isolated_vram_builder_preserves_route_memory_fields():
    result = _build_result(
        {
            "framework": "gpytorch",
            "n": 2000,
            "d": 5,
            "method": "materialized",
            "prediction_mode": "exact",
            "n_test": 16,
        },
        {
            "framework": "gpytorch",
            "peak_mb": 256.0,
            "delta_mb": 64.0,
            "baseline_mb": 192.0,
            "torch_peak_mb": 128.0,
            "training_peak_mb": 220.0,
            "prediction_peak_mb": 240.0,
            "prediction_delta_mb": 48.0,
            "exact_prediction_peak_mb": 240.0,
            "exact_prediction_delta_mb": 48.0,
            "mean_time_s": 0.12,
        },
    )

    assert result.config["suite_name"] == "isolated_vram_measurement"
    assert result.memory.gpu_max_mb == 256.0
    assert result.memory.gpu_delta_mb == 64.0
    assert result.memory.training_peak_gpu_mb == 220.0
    assert result.memory.prediction_peak_gpu_mb == 240.0
    assert result.memory.exact_prediction_peak_gpu_mb == 240.0
    assert result.memory.exact_prediction_delta_gpu_mb == 48.0
    assert result.speed.prediction_mean_time_s == 0.12
