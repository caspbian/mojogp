from __future__ import annotations

from pathlib import Path

from tests.shared.benchmarking.report import (
    load_benchmark_result,
    load_comparison_result,
    save_comparison_artifact,
    save_result_artifact,
    save_system_result,
)
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    ComparisonResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)


def _sample_result() -> BenchmarkResult:
    return BenchmarkResult(
        config={
            "framework": "mojogp",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "training_method": "materialized",
            "method": "materialized",
            "prediction_mode": "exact",
            "comparison_class": "cross_framework",
            "fairness_note": "fixture fairness note",
            "fairness_axes": {"solver": {"status": "aligned", "note": "same budget"}},
            "n": 5000,
            "d": 5,
        },
        accuracy=AccuracyResult(
            rmse=0.1,
            mae=0.1,
            r_squared=0.9,
            crps=0.1,
            msll=0.1,
            calibration_coverage={0.95: 0.94},
            calibration_error=0.01,
            sharpness=0.2,
            interval_width_95=0.5,
        ),
        speed=SpeedResult(
            training_time_s=1.0,
            prediction_mean_time_s=0.1,
            prediction_variance_time_s=0.2,
            end_to_end_time_s=1.3,
            iterations_run=10,
            max_iterations=10,
            early_stopped=False,
            ms_per_iteration=100.0,
            iter_time_min_ms=95.0,
            iter_time_q25_ms=98.0,
            iter_time_mean_ms=100.5,
            iter_time_median_ms=100.0,
            iter_time_q75_ms=103.0,
            iter_time_max_ms=108.0,
            iter_time_p5_ms=96.0,
            iter_time_p95_ms=107.0,
            startup_compile_time_s=0.4,
            startup_warm_cache_hit_s=0.06,
            startup_prepare_time_s=0.02,
        ),
        memory=MemoryResult(
            gpu_mean_mb=120.0,
            gpu_min_mb=100.0,
            gpu_max_mb=150.0,
            gpu_var_mb=10.0,
            torch_peak_mb=112.0,
            torch_current_mb=88.0,
            cpu_peak_mb=64.0,
            measurement_method="torch.cuda",
            num_samples=5,
            gpu_baseline_mb=96.0,
            gpu_delta_mb=54.0,
            gpu_isolated_peak_mb=32.0,
            torch_peak_delta_mb=32.0,
            training_peak_gpu_mb=150.0,
            training_delta_gpu_mb=50.0,
            prediction_peak_gpu_mb=120.0,
            prediction_delta_gpu_mb=20.0,
            exact_prediction_delta_gpu_mb=80.0,
        ),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=1.0,
            learned_noise=0.1,
            learned_outputscale=1.0,
            final_nll=1.2,
        ),
    )


def test_save_result_artifact_round_trips(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    result = _sample_result()
    path = save_result_artifact(result, results_dir, "single_output_scaling")
    loaded = load_benchmark_result(path)

    assert path.exists()
    assert loaded.config["framework"] == "mojogp"
    assert loaded.memory.gpu_isolated_peak_mb == 32.0
    assert loaded.accuracy.calibration_coverage[0.95] == 0.94
    assert loaded.speed.iter_time_q75_ms == 103.0
    assert loaded.speed.startup_compile_time_s == 0.4


def test_save_system_result_does_not_require_benchmark_runtime(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    result = _sample_result()
    path = save_system_result(result, results_dir, "single_output_accuracy")

    assert path.exists()
    assert path.read_text()


def test_save_comparison_artifact_round_trips(tmp_path: Path):
    mojogp = _sample_result()
    gpytorch = _sample_result()
    gpytorch.run_id = "gpytorch01"
    gpytorch.config.update({"framework": "gpytorch", "training_method": "cg"})
    comparison = ComparisonResult(
        config={
            "framework": "cross_framework",
            "model_type": "SingleOutputGP",
            "kernel": "rbf",
            "n": 5000,
            "d": 5,
            "comparison_class": "cross_framework",
            "baseline_backend": "gpytorch_cg",
            "fairness_note": "fixture fairness note",
            "fairness_axes": {"solver": {"status": "aligned", "note": "same budget"}},
            "comparison_id": "cmp-1",
        },
        mojogp_materialized=mojogp,
        gpytorch_cg=gpytorch,
    )
    comparison.compute_comparisons()

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    path = save_comparison_artifact(comparison, results_dir, "single_output_ground_truth_active")
    loaded = load_comparison_result(path)

    assert path.exists()
    assert loaded.gpytorch_cg is not None
    assert loaded.mojogp_materialized is not None
    assert loaded.config["comparison_id"] == "cmp-1"
    assert loaded.gpytorch_cg.config["framework"] == "gpytorch"
