"""Persist isolated VRAM measurements as benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

from tests.integration.run_gpu_memory_measurement_case import (
    _run_gpytorch_case,
    _run_mojogp_case,
    _run_pytorch_case,
)
from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.shared.benchmarking.report import save_result_artifact
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)
from tests.shared.subprocess_harness import run_child_main


def _build_result(payload: dict[str, object], measured: dict[str, object]) -> BenchmarkResult:
    framework = str(measured["framework"])
    prediction_mode = str(payload.get("prediction_mode", measured.get("prediction_mode", "exact")))
    method = str(payload.get("method", "materialized"))
    config = {
        "benchmark": "isolated_vram_measurement",
        "framework": framework,
        "model_type": "SingleOutputGP" if framework != "pytorch" else "tensor_allocation",
        "kernel": str(payload.get("kernel", "rbf")),
        "training_method": method,
        "method": method,
        "prediction_mode": prediction_mode,
        "comparison_class": "memory_measurement",
        "baseline_backend": "none" if framework in {"mojogp", "pytorch"} else "standard",
        "n": int(payload["n"]),
        "d": int(payload.get("d", 5)),
        "n_test": int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST)),
        "suite_name": "isolated_vram_measurement",
        "memory_telemetry_quality": "isolated_phase_numeric",
    }
    if isinstance(measured.get("backend_predict_info"), dict):
        config["backend_predict_info"] = dict(measured["backend_predict_info"])
    speed = SpeedResult(
        training_time_s=0.0,
        prediction_mean_time_s=float(measured.get("mean_time_s", 0.0)),
        prediction_variance_time_s=float(measured.get("variance_time_s", 0.0)),
        end_to_end_time_s=float(measured.get("mean_time_s", 0.0))
        + float(measured.get("variance_time_s", 0.0)),
        iterations_run=int(payload.get("max_iterations", 1)),
        max_iterations=int(payload.get("max_iterations", 1)),
        early_stopped=False,
        ms_per_iteration=0.0,
        iter_timing_quality="not_applicable_memory_probe",
    )
    memory = MemoryResult(
        gpu_mean_mb=float(measured.get("peak_mb", 0.0)),
        gpu_min_mb=float(measured.get("baseline_mb", 0.0)),
        gpu_max_mb=float(measured.get("peak_mb", 0.0)),
        gpu_var_mb=0.0,
        torch_peak_mb=float(measured.get("torch_peak_mb", 0.0)),
        torch_current_mb=float(measured.get("torch_peak_mb", 0.0)),
        cpu_peak_mb=0.0,
        measurement_method="isolated_gpu_phase",
        num_samples=1,
        gpu_baseline_mb=float(measured.get("baseline_mb", 0.0)),
        gpu_delta_mb=float(measured.get("delta_mb", 0.0)),
        training_peak_gpu_mb=float(measured.get("training_peak_mb", 0.0)),
        training_delta_gpu_mb=float(measured.get("training_delta_mb", 0.0)),
        prediction_peak_gpu_mb=float(measured.get("prediction_peak_mb", 0.0)),
        prediction_delta_gpu_mb=float(measured.get("prediction_delta_mb", 0.0)),
        exact_prediction_peak_gpu_mb=float(measured.get("exact_prediction_peak_mb", 0.0)),
        exact_prediction_delta_gpu_mb=float(measured.get("exact_prediction_delta_mb", 0.0)),
        love_prediction_peak_gpu_mb=float(measured.get("love_prediction_peak_mb", 0.0)),
        love_prediction_delta_gpu_mb=float(measured.get("love_prediction_delta_mb", 0.0)),
    )
    return BenchmarkResult(
        config=config,
        accuracy=AccuracyResult(
            rmse=0.0,
            mae=0.0,
            r_squared=0.0,
            crps=0.0,
            msll=0.0,
            calibration_coverage={},
            calibration_error=0.0,
            sharpness=0.0,
            interval_width_95=0.0,
        ),
        speed=speed,
        memory=memory,
        hyperparameters=HyperparameterResult(
            learned_lengthscale=0.0,
            learned_noise=0.0,
            learned_outputscale=0.0,
            final_nll=0.0,
        ),
    )


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    framework = str(payload["framework"])
    if framework == "mojogp":
        measured = _run_mojogp_case(payload)
    elif framework == "pytorch":
        measured = _run_pytorch_case(payload)
    elif framework == "gpytorch":
        measured = _run_gpytorch_case(payload, mode="cg")
    elif framework == "gpytorch_keops":
        measured = _run_gpytorch_case(payload, mode="keops")
    else:
        raise ValueError(f"Unknown framework '{framework}'")

    result = _build_result(payload, measured)
    results_dir = Path(str(payload["results_dir"]))
    result_path = save_result_artifact(result, results_dir, "isolated_vram_measurement")
    return {"payload": result.to_dict(), "result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
