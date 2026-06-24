from __future__ import annotations

from tests.shared.benchmarking.gpytorch_models import merge_gpytorch_benchmark_memory


def test_merge_gpytorch_benchmark_memory_preserves_prediction_route_fields():
    merged = merge_gpytorch_benchmark_memory(
        {
            "max_mb": 120.0,
            "mean_mb": 90.0,
            "torch_peak_mb": 80.0,
            "samples": 4,
            "method": "pynvml",
        },
        {
            "prediction_peak_gpu_mb": 96.0,
            "prediction_delta_gpu_mb": 24.0,
            "exact_prediction_peak_gpu_mb": 96.0,
            "exact_prediction_delta_gpu_mb": 24.0,
            "torch_peak_mb": 88.0,
            "torch_current_mb": 40.0,
            "samples": 3,
            "method": "torch.cuda",
        },
    )

    assert merged["training_peak_gpu_mb"] == 120.0
    assert merged["prediction_peak_gpu_mb"] == 96.0
    assert merged["prediction_delta_gpu_mb"] == 24.0
    assert merged["exact_prediction_peak_gpu_mb"] == 96.0
    assert merged["exact_prediction_delta_gpu_mb"] == 24.0
    assert merged["torch_peak_mb"] == 88.0
    assert merged["torch_current_mb"] == 40.0
    assert merged["samples"] == 7
    assert merged["method"] == "torch.cuda"


def test_merge_gpytorch_benchmark_memory_preserves_love_route_fields():
    merged = merge_gpytorch_benchmark_memory(
        {"max_mb": 64.0, "samples": 2, "method": "pynvml"},
        {
            "prediction_peak_gpu_mb": 48.0,
            "prediction_delta_gpu_mb": 12.0,
            "love_prediction_peak_gpu_mb": 48.0,
            "love_prediction_delta_gpu_mb": 12.0,
            "samples": 5,
            "method": "pynvml",
        },
    )

    assert merged["love_prediction_peak_gpu_mb"] == 48.0
    assert merged["love_prediction_delta_gpu_mb"] == 12.0
    assert merged["prediction_peak_gpu_mb"] == 48.0
    assert merged["prediction_delta_gpu_mb"] == 12.0
