"""Benchmark wrapper around the shared subprocess harness."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tests.shared.subprocess_harness import (
    IsolatedSubprocessError,
    load_child_result,
    run_isolated_module,
)

from .artifact_store import save_json_artifact
from .contracts import ContractEvaluationSummary
from .preflight import GitMetadata, ProfilingMetadata, get_container_metadata, utc_now_iso
from .session_store import BenchmarkSessionStore
from .specialization_columns import extract_specialization_columns
from .specialization_studies import finalize_trial_from_run


@dataclass
class BenchmarkHarnessResult:
    envelope: dict[str, Any]
    loaded_result: Any
    artifact_id: str
    artifact_path: Path


def _derive_scaling_memory_metric(loaded_memory: Any, gpu_isolated_peak_mb: Any) -> tuple[float | None, str]:
    if loaded_memory is not None:
        candidates: list[tuple[str, float]] = []
        for key in (
            "training_delta_gpu_mb",
            "prediction_delta_gpu_mb",
            "exact_prediction_delta_gpu_mb",
            "love_prediction_delta_gpu_mb",
        ):
            value = getattr(loaded_memory, key, None)
            if value is None:
                continue
            value_f = float(value)
            if value_f > 0.0:
                candidates.append((key, value_f))
        if candidates:
            metric_name, metric_value = max(candidates, key=lambda item: item[1])
            return metric_value, metric_name
    if gpu_isolated_peak_mb is None:
        return None, "overall_process_peak"
    return float(gpu_isolated_peak_mb), "overall_process_peak"


def _export_session_snapshot(
    session_store: BenchmarkSessionStore,
    session_id: str,
) -> None:
    """Write an importable session bundle after each benchmark case."""
    session_store.export_session_json(session_id)


def _register_failed_child_run(
    *,
    session_store: BenchmarkSessionStore,
    session_id: str,
    case_id: str,
    benchmark_group_id: str,
    benchmark_name: str,
    framework: str,
    git: GitMetadata,
    profiling: ProfilingMetadata,
    config: dict[str, Any],
    dataset_id: str | None,
    comparison_id: str | None,
    artifact_id: str,
    envelope: Mapping[str, Any],
    started_at: str,
    finished_at: str,
) -> str:
    container = get_container_metadata()
    telemetry = dict(envelope.get("telemetry", {}) or {})
    memory = dict(telemetry.get("memory", {}) or {})
    gpu = dict(memory.get("gpu", {}) or {})
    torch = dict(memory.get("torch", {}) or {})
    cpu = dict(memory.get("cpu", {}) or {})
    error = envelope.get("error") if isinstance(envelope.get("error"), dict) else {}
    failed_config = dict(config)
    failed_config["child_error"] = error
    run_id = uuid.uuid4().hex[:8]
    session_store.register_run(
        {
            "run_id": run_id,
            "session_id": session_id,
            "case_id": case_id,
            "benchmark_group_id": benchmark_group_id,
            "n": config.get("n"),
            "d": config.get("d"),
            "num_tasks": config.get("num_tasks"),
            "kernel": config.get("kernel"),
            "model_type": config.get("model_type"),
            "training_method": config.get("training_method"),
            "prediction_mode": config.get("prediction_mode"),
            "suite_name": config.get("suite_name"),
            "comparison_class": config.get("comparison_class"),
            "baseline_backend": config.get("baseline_backend"),
            "fairness_note": config.get("fairness_note"),
            "dataset_id": dataset_id,
            "comparison_id": comparison_id,
            "artifact_id": artifact_id,
            "status": str(envelope.get("status", "error")),
            "started_at": started_at,
            "finished_at": finished_at,
            "framework": framework,
            "config_json": json.dumps(failed_config, sort_keys=True, default=str),
            "benchmark_name": benchmark_name,
            "result_json_path": envelope.get("result_path"),
            "branch_name": git.branch_name,
            "commit_hash": git.commit_hash,
            "git_clean": int(git.git_clean),
            "profiling_probe_passed": int(profiling.profiling_probe_passed),
            "container_runtime": container.runtime,
            "container_image_tag": container.image_tag,
            "container_image_digest": container.image_digest,
            "container_image_id": container.image_id,
            "gpu_name": container.gpu_name,
            "gpu_total_vram_mb": container.gpu_total_vram_mb,
            "gpu_driver_version": container.gpu_driver_version,
            "cuda_version": container.cuda_version,
            "gpu_target": container.gpu_target,
            "gpu_baseline_mb": gpu.get("baseline_mb"),
            "gpu_current_mb": gpu.get("current_mb"),
            "gpu_delta_mb": gpu.get("delta_mb"),
            "gpu_max_mb": gpu.get("max_mb"),
            "gpu_isolated_peak_mb": gpu.get("isolated_peak_mb"),
            "gpu_isolated_current_mb": gpu.get("isolated_current_mb"),
            "gpu_samples": gpu.get("samples"),
            "measurement_method_primary": gpu.get("method"),
            "torch_baseline_mb": torch.get("baseline_mb"),
            "torch_peak_mb": torch.get("peak_mb"),
            "torch_peak_delta_mb": torch.get("peak_delta_mb"),
            "torch_current_delta_mb": torch.get("current_delta_mb"),
            "torch_reserved_delta_mb": torch.get("reserved_delta_mb"),
            "cpu_peak_mb": cpu.get("peak_mb"),
        }
    )
    return run_id


def run_benchmark_module(
    *,
    module: str,
    payload: Mapping[str, Any],
    timeout: int,
    description: str,
    result_loader: Callable[[Path], Any] | None,
    session_store: BenchmarkSessionStore,
    session_id: str,
    case_id: str,
    benchmark_group_id: str,
    benchmark_name: str,
    framework: str,
    git: GitMetadata,
    profiling: ProfilingMetadata,
    config: dict[str, Any],
    dataset_id: str | None = None,
    comparison_id: str | None = None,
    cwd: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> BenchmarkHarnessResult:
    started_at = utc_now_iso()
    env = dict(extra_env or {})
    env.setdefault("MOJOGP_JIT_CACHE_DIR", profiling.jit_cache_dir)
    artifact_id = uuid.uuid4().hex[:16]
    try:
        envelope = run_isolated_module(
            module=module,
            payload=payload,
            timeout=timeout,
            cwd=cwd,
            extra_env=env,
            description=description,
        )
    except IsolatedSubprocessError as exc:
        finished_at = utc_now_iso()
        envelope = exc.envelope or {
            "status": "error",
            "payload": None,
            "result_path": None,
            "telemetry": {},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        artifact_payload = {
            "case_id": case_id,
            "benchmark_group_id": benchmark_group_id,
            "benchmark_name": benchmark_name,
            "framework": framework,
            "payload": dict(payload),
            "envelope": envelope,
        }
        artifact_path, sha256, _ = save_json_artifact(artifact_id, artifact_payload)
        session_store.register_artifact(
            artifact_id=artifact_id,
            run_id=None,
            artifact_type="benchmark_run",
            path=str(artifact_path),
            sha256=sha256,
            created_at=finished_at,
        )
        run_id = _register_failed_child_run(
            session_store=session_store,
            session_id=session_id,
            case_id=case_id,
            benchmark_group_id=benchmark_group_id,
            benchmark_name=benchmark_name,
            framework=framework,
            git=git,
            profiling=profiling,
            config=dict(config),
            dataset_id=dataset_id,
            comparison_id=comparison_id,
            artifact_id=artifact_id,
            envelope=envelope,
            started_at=started_at,
            finished_at=finished_at,
        )
        session_store.register_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            artifact_type="benchmark_run",
            path=str(artifact_path),
            sha256=sha256,
            created_at=finished_at,
        )
        _export_session_snapshot(session_store, session_id)
        raise

    finished_at = utc_now_iso()
    container = get_container_metadata()
    artifact_payload = {
        "case_id": case_id,
        "benchmark_group_id": benchmark_group_id,
        "benchmark_name": benchmark_name,
        "framework": framework,
        "payload": dict(payload),
        "envelope": envelope,
    }
    artifact_path, sha256, _ = save_json_artifact(artifact_id, artifact_payload)
    session_store.register_artifact(
        artifact_id=artifact_id,
        run_id=None,
        artifact_type="benchmark_run",
        path=str(artifact_path),
        sha256=sha256,
        created_at=finished_at,
    )
    loaded = load_child_result(envelope, result_loader=result_loader)

    telemetry = dict(envelope.get("telemetry", {}))
    memory = dict(telemetry.get("memory", {}))
    gpu = dict(memory.get("gpu", {}))
    torch = dict(memory.get("torch", {}))
    cpu = dict(memory.get("cpu", {}))
    result_path = envelope.get("result_path")
    if hasattr(loaded, "config"):
        loaded.config.setdefault("case_id", case_id)
        loaded.config.setdefault("benchmark_group_id", benchmark_group_id)
        if dataset_id is not None:
            loaded.config.setdefault("dataset_id", dataset_id)
        loaded.config.setdefault("artifact_id", artifact_id)
        loaded.config.setdefault("session_id", session_id)
        if hasattr(loaded, "memory"):
            loaded.memory.gpu_baseline_mb = gpu.get("baseline_mb")
            loaded.memory.gpu_current_mb = gpu.get("current_mb")
            loaded.memory.gpu_delta_mb = gpu.get("delta_mb")
            loaded.memory.gpu_isolated_peak_mb = gpu.get("isolated_peak_mb")
            loaded.memory.gpu_isolated_current_mb = gpu.get("isolated_current_mb")
            loaded.memory.torch_baseline_mb = torch.get("baseline_mb")
            loaded.memory.torch_peak_delta_mb = torch.get("peak_delta_mb")
            loaded.memory.torch_current_delta_mb = torch.get("current_delta_mb")
            loaded.memory.torch_reserved_mb = torch.get("reserved_mb")
            loaded.memory.torch_reserved_baseline_mb = torch.get("reserved_baseline_mb")
            loaded.memory.torch_reserved_delta_mb = torch.get("reserved_delta_mb")
            if loaded.memory.measurement_method in (None, "none"):
                loaded.memory.measurement_method = str(gpu.get("method", "none"))
    if hasattr(loaded, "to_dict"):
        loaded_dict = loaded.to_dict()
        loaded_config = dict(getattr(loaded, "config", {}))
        loaded_memory = getattr(loaded, "memory", None)
    else:
        loaded_dict = dict(loaded)
        if isinstance(loaded_dict.get("benchmark"), dict):
            benchmark_payload = dict(loaded_dict["benchmark"])
            loaded_config = dict(benchmark_payload.get("config", {}))
            loaded_dict = {
                **benchmark_payload,
                **{key: value for key, value in loaded_dict.items() if key != "benchmark"},
            }
        else:
            loaded_config = {}
        loaded_memory = None
    merged_config = dict(config)
    merged_config.update(loaded_config)
    case_row = session_store.fetch_case(case_id)
    suite_name = loaded_config.get("suite_name") or config.get("suite_name")
    if suite_name is None and case_row is not None:
        suite_name = case_row.get("suite_name")
    if suite_name is not None:
        loaded_config["suite_name"] = suite_name
        if hasattr(loaded, "config") and isinstance(loaded.config, dict):
            loaded.config["suite_name"] = suite_name
        if isinstance(loaded_dict.get("config"), dict):
            loaded_dict["config"]["suite_name"] = suite_name
        else:
            loaded_dict.setdefault("config", dict(loaded_config))
    for key, value in {
        "gpu_name": container.gpu_name,
        "gpu_total_vram_mb": container.gpu_total_vram_mb,
        "gpu_driver_version": container.gpu_driver_version,
        "cuda_version": container.cuda_version,
        "gpu_target": container.gpu_target,
        "container_runtime": container.runtime,
        "container_image_tag": container.image_tag,
        "container_image_digest": container.image_digest,
        "container_image_id": container.image_id,
    }.items():
        if value is not None:
            loaded_config.setdefault(key, value)
    if hasattr(loaded, "config") and isinstance(loaded.config, dict):
        for key, value in loaded_config.items():
            loaded.config.setdefault(key, value)
    if isinstance(loaded_dict.get("config"), dict):
        for key, value in loaded_config.items():
            loaded_dict["config"].setdefault(key, value)
    merged_config = dict(config)
    merged_config.update(loaded_config)
    specialization = extract_specialization_columns(merged_config)
    if result_path:
        result_target = Path(str(result_path))
        if hasattr(loaded, "to_json"):
            result_target.write_text(loaded.to_json(), encoding="utf-8")
        else:
            result_target.write_text(json.dumps(loaded_dict, indent=2, sort_keys=True, default=str), encoding="utf-8")
    run_id = loaded_dict.get("run_id", uuid.uuid4().hex[:8])
    speed = loaded_dict.get("speed", {})
    early_stopped = speed.get("early_stopped")
    contract_summary = loaded_config.get("benchmark_contracts")
    contract_passed = None
    contract_summary_json = None
    if isinstance(contract_summary, ContractEvaluationSummary):
        contract_passed = int(contract_summary.passed)
        contract_summary_json = json.dumps(contract_summary.to_dict(), sort_keys=True)
    elif isinstance(contract_summary, dict):
        passed_value = contract_summary.get("passed")
        if passed_value is not None:
            contract_passed = int(bool(passed_value))
        contract_summary_json = json.dumps(contract_summary, sort_keys=True, default=str)
    scaling_peak_gpu_mb, scaling_memory_metric = _derive_scaling_memory_metric(
        loaded_memory,
        gpu.get("isolated_peak_mb"),
    )
    accuracy = loaded_dict.get("accuracy", {})
    hyperparameters = loaded_dict.get("hyperparameters", {})
    session_store.register_run(
        {
            "run_id": run_id,
            "session_id": session_id,
            "case_id": case_id,
            "benchmark_group_id": benchmark_group_id,
            "n": loaded_config.get("n", config.get("n")),
            "d": loaded_config.get("d", config.get("d")),
            "num_tasks": loaded_config.get("num_tasks", config.get("num_tasks")),
            "kernel": loaded_config.get("kernel", config.get("kernel")),
            "model_type": loaded_config.get("model_type", config.get("model_type")),
            "training_method": loaded_config.get("training_method", config.get("training_method")),
            "prediction_mode": loaded_config.get("prediction_mode", config.get("prediction_mode")),
            "suite_name": suite_name,
            "comparison_class": loaded_config.get("comparison_class"),
            "baseline_backend": loaded_config.get("baseline_backend"),
            "fairness_note": loaded_config.get("fairness_note"),
            "base_case_id": specialization["base_case_id"],
            "specialization_key": specialization["specialization_key"],
            "specialization_family": specialization["specialization_family"],
            "specialization_mode": specialization["specialization_mode"],
            "specialization_source": specialization["specialization_source"],
            "specialization_descriptor_json": json.dumps(
                specialization["specialization_descriptor_json"],
                sort_keys=True,
                default=str,
            ),
            "specialization_config_json": json.dumps(
                specialization["specialization_config_json"],
                sort_keys=True,
                default=str,
            ),
            "study_id": merged_config.get("study_id"),
            "trial_id": merged_config.get("trial_id"),
            "objective_name": merged_config.get("objective_name"),
            "objective_metric": merged_config.get("objective_metric"),
            "constraint_json": json.dumps(
                dict(merged_config.get("constraint_json", {})),
                sort_keys=True,
                default=str,
            ),
            "dataset_id": dataset_id,
            "comparison_id": comparison_id,
            "artifact_id": artifact_id,
            "status": str(envelope.get("status", "unknown")),
            "started_at": started_at,
            "finished_at": finished_at,
            "framework": framework,
            "config_json": json.dumps(merged_config, sort_keys=True, default=str),
            "benchmark_name": benchmark_name,
            "result_json_path": None if result_path is None else str(result_path),
            "training_time_s": speed.get("training_time_s"),
            "prediction_mean_time_s": speed.get("prediction_mean_time_s"),
            "prediction_variance_time_s": speed.get("prediction_variance_time_s"),
            "prediction_cold_first_time_s": speed.get("prediction_cold_first_time_s"),
            "prediction_cache_prepare_time_s": speed.get("prediction_cache_prepare_time_s"),
            "prediction_prepared_apply_time_s": speed.get("prediction_prepared_apply_time_s"),
            "prediction_repeated_median_time_s": speed.get("prediction_repeated_median_time_s"),
            "prediction_repeated_p5_time_s": speed.get("prediction_repeated_p5_time_s"),
            "prediction_repeated_p95_time_s": speed.get("prediction_repeated_p95_time_s"),
            "prediction_alpha_time_s": speed.get("prediction_alpha_time_s"),
            "prediction_love_root_time_s": speed.get("prediction_love_root_time_s"),
            "prediction_x_test_scaling": speed.get("prediction_x_test_scaling"),
            "end_to_end_time_s": speed.get("end_to_end_time_s"),
            "iterations_run": speed.get("iterations_run"),
            "max_iterations": speed.get("max_iterations"),
            "early_stopped": None if early_stopped is None else int(bool(early_stopped)),
            "ms_per_iteration": speed.get("ms_per_iteration"),
            "iter_time_min_ms": speed.get("iter_time_min_ms"),
            "iter_time_q25_ms": speed.get("iter_time_q25_ms"),
            "iter_time_mean_ms": speed.get("iter_time_mean_ms"),
            "iter_time_median_ms": speed.get("iter_time_median_ms"),
            "iter_time_q75_ms": speed.get("iter_time_q75_ms"),
            "iter_time_max_ms": speed.get("iter_time_max_ms"),
            "iter_time_p5_ms": speed.get("iter_time_p5_ms"),
            "iter_time_p95_ms": speed.get("iter_time_p95_ms"),
            "startup_compile_time_s": speed.get("startup_compile_time_s"),
            "startup_warm_cache_hit_s": speed.get("startup_warm_cache_hit_s"),
            "startup_prepare_time_s": speed.get("startup_prepare_time_s"),
            "contract_passed": contract_passed,
            "contract_summary_json": contract_summary_json,
            "training_peak_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "training_peak_gpu_mb", None),
            "training_delta_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "training_delta_gpu_mb", None),
            "prediction_peak_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "prediction_peak_gpu_mb", None),
            "prediction_delta_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "prediction_delta_gpu_mb", None),
            "exact_prediction_peak_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "exact_prediction_peak_gpu_mb", None),
            "exact_prediction_delta_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "exact_prediction_delta_gpu_mb", None),
            "love_prediction_peak_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "love_prediction_peak_gpu_mb", None),
            "love_prediction_delta_gpu_mb": None if loaded_memory is None else getattr(loaded_memory, "love_prediction_delta_gpu_mb", None),
            "scaling_peak_gpu_mb": scaling_peak_gpu_mb,
            "scaling_memory_metric": scaling_memory_metric,
            "gpu_baseline_mb": gpu.get("baseline_mb"),
            "gpu_current_mb": gpu.get("current_mb"),
            "gpu_delta_mb": gpu.get("delta_mb"),
            "gpu_max_mb": gpu.get("max_mb"),
            "gpu_isolated_peak_mb": gpu.get("isolated_peak_mb"),
            "gpu_isolated_current_mb": gpu.get("isolated_current_mb"),
            "gpu_samples": gpu.get("samples"),
            "measurement_method_primary": gpu.get("method"),
            "torch_baseline_mb": torch.get("baseline_mb"),
            "torch_peak_mb": torch.get("peak_mb"),
            "torch_peak_delta_mb": torch.get("peak_delta_mb"),
            "torch_current_delta_mb": torch.get("current_delta_mb"),
            "torch_reserved_delta_mb": torch.get("reserved_delta_mb"),
            "cpu_peak_mb": cpu.get("peak_mb"),
            "rmse": accuracy.get("rmse"),
            "mae": accuracy.get("mae"),
            "r_squared": accuracy.get("r_squared"),
            "crps": accuracy.get("crps"),
            "msll": accuracy.get("msll"),
            "calibration_error": accuracy.get("calibration_error"),
            "sharpness": accuracy.get("sharpness"),
            "interval_width_95": accuracy.get("interval_width_95"),
            "final_nll": hyperparameters.get("final_nll"),
            "branch_name": git.branch_name,
            "commit_hash": git.commit_hash,
            "git_clean": int(git.git_clean),
            "profiling_probe_passed": int(profiling.profiling_probe_passed),
            "container_runtime": container.runtime,
            "container_image_tag": container.image_tag,
            "container_image_digest": container.image_digest,
            "container_image_id": container.image_id,
            "gpu_name": container.gpu_name,
            "gpu_total_vram_mb": container.gpu_total_vram_mb,
            "gpu_driver_version": container.gpu_driver_version,
            "cuda_version": container.cuda_version,
            "gpu_target": container.gpu_target,
        }
    )
    parameter_recovery = merged_config.get("parameter_recovery")
    if parameter_recovery is None:
        parameter_recovery = loaded_dict.get("parameter_recovery")
    if parameter_recovery is not None:
        session_store.register_parameter_recovery(
            run_id=str(run_id),
            rows=parameter_recovery,
        )
    if merged_config.get("trial_id") and merged_config.get("objective_metric"):
        finalize_trial_from_run(
            session_store,
            trial_id=str(merged_config["trial_id"]),
            run_row={
                "run_id": run_id,
                "case_id": case_id,
                "base_case_id": specialization["base_case_id"],
                "specialization_key": specialization["specialization_key"],
                "training_time_s": speed.get("training_time_s"),
                "prediction_mean_time_s": speed.get("prediction_mean_time_s"),
                "prediction_variance_time_s": speed.get("prediction_variance_time_s"),
                "end_to_end_time_s": speed.get("end_to_end_time_s"),
                "scaling_peak_gpu_mb": scaling_peak_gpu_mb,
                "contract_passed": contract_passed,
            },
            objective_metric=str(merged_config["objective_metric"]),
            constraints=dict(merged_config.get("constraint_json", {})),
        )
    session_store.register_artifact(
        artifact_id=artifact_id,
        run_id=run_id,
        artifact_type="benchmark_run",
        path=str(artifact_path),
        sha256=sha256,
        created_at=finished_at,
    )
    _export_session_snapshot(session_store, session_id)
    return BenchmarkHarnessResult(
        envelope=envelope,
        loaded_result=loaded,
        artifact_id=artifact_id,
        artifact_path=artifact_path,
    )
