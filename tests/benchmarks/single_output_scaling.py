"""Benchmark runner integration for single-output scaling cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mojogp.specialization import merge_specialization_config

from tests.benchmarks.gpytorch.single_output_scaling import (
    run_gpytorch_single_output_scaling_module,
)
from tests.benchmarks.mojogp.single_output_scaling import (
    run_mojogp_single_output_scaling_module,
)
from tests.shared.benchmarking.data_generators import (
    generate_gp_prior_data,
    generate_structured_function_data,
)
from tests.shared.benchmarking.report import load_benchmark_result

from .comparison_registry import (
    comparison_id_from_cases,
    single_output_ard_scaling_case_id,
    single_output_ard_scaling_group_id,
    single_output_scaling_case_id,
    single_output_scaling_group_id,
)
from .dataset_manifest import DatasetSpec, ensure_dataset, save_dataset_artifact_bundle
from .harness_runner import run_benchmark_module
from .specialization_adapter import prepare_specialization_payload
from .specialization_studies import merge_study_trial_config
from .runtime import BenchmarkRuntimeContext, get_or_create_default_context
from .session_store import BenchmarkSessionStore
from .preflight import utc_now_iso
from .prediction_workload import BENCHMARK_PREDICTION_N_TEST


def _single_output_data_config(
    *,
    method: str,
    n_train: int,
    d: int,
    data_options: dict[str, object] | None,
) -> dict[str, object]:
    seed = (100 if method == "materialized" else 500) + n_train + d * 1000
    data_config: dict[str, object] = {
        "dataset_family": "structured_function",
        "n_train": n_train,
        "n_test": BENCHMARK_PREDICTION_N_TEST,
        "d": d,
        "function_type": "smooth",
        "noise_level": "medium",
        "seed": seed,
    }
    data_config.update(dict(data_options or {}))
    return data_config


def _ensure_single_output_dataset(
    *,
    method: str,
    n_train: int,
    d: int,
    data_options: dict[str, object] | None,
    context: BenchmarkRuntimeContext,
) -> tuple[str, Path, Path]:
    data_config = _single_output_data_config(
        method=method,
        n_train=n_train,
        d=d,
        data_options=data_options,
    )
    dataset_family = str(data_config.get("dataset_family", "structured_function"))
    if dataset_family == "structured_function":
        spec = DatasetSpec(
            generator_name="single_output_structured_function",
            config={
                "n_train": int(data_config["n_train"]),
                "n_test": int(data_config["n_test"]),
                "d": int(data_config["d"]),
                "function_type": str(data_config.get("function_type", "smooth")),
                "noise_level": str(data_config.get("noise_level", "medium")),
                "seed": int(data_config["seed"]),
                "mean_offset": float(data_config.get("mean_offset", 0.0)),
            },
        )
        dataset_id, npz_path, meta_path = ensure_dataset(spec, generate_structured_function_data)
    elif dataset_family == "structured_ard":
        from tests.shared.benchmarking.data_generators import (
            generate_single_output_structured_ard_data,
        )

        spec = DatasetSpec(
            generator_name="single_output_structured_ard",
            config={
                "n_train": int(data_config["n_train"]),
                "n_test": int(data_config["n_test"]),
                "d": int(data_config["d"]),
                "relevant_dims": int(data_config.get("relevant_dims", min(3, d))),
                "noise_level": str(data_config.get("noise_level", "medium")),
                "seed": int(data_config["seed"]),
                "mean_offset": float(data_config.get("mean_offset", 0.0)),
            },
        )
        dataset_id, npz_path, meta_path = ensure_dataset(
            spec, generate_single_output_structured_ard_data
        )
    elif dataset_family == "gp_prior":
        x_range = data_config.get("x_range", (-3.0, 3.0))
        spec = DatasetSpec(
            generator_name="single_output_gp_prior",
            config={
                "n_train": int(data_config["n_train"]),
                "n_test": int(data_config["n_test"]),
                "d": int(data_config["d"]),
                "kernel_type": str(data_config.get("kernel_type", "rbf")),
                "true_lengthscale": float(data_config.get("true_lengthscale", 1.0)),
                "true_noise": float(data_config.get("true_noise", 0.1)),
                "true_outputscale": float(data_config.get("true_outputscale", 1.0)),
                "seed": int(data_config["seed"]),
                "x_range": (float(x_range[0]), float(x_range[1])),
                "true_period": float(data_config.get("true_period", 1.0)),
                "true_alpha": float(data_config.get("true_alpha", 1.0)),
                "mean_offset": float(data_config.get("mean_offset", 0.0)),
            },
        )
        dataset_id, npz_path, meta_path = ensure_dataset(spec, generate_gp_prior_data)
    else:
        raise ValueError(f"Unknown dataset_family '{dataset_family}'")

    artifact_path, artifact_sha256, _ = save_dataset_artifact_bundle(
        dataset_id,
        dataset_npz_path=npz_path,
        dataset_meta_path=meta_path,
    )
    context.session_store.register_dataset(
        dataset_id=dataset_id,
        generator_name=spec.generator_name,
        config=spec.config,
        seed=int(spec.config.get("seed", 0)),
        artifact_path=str(artifact_path),
        artifact_sha256=artifact_sha256,
        created_at=utc_now_iso(),
    )
    return dataset_id, npz_path, meta_path


def _single_output_scaling_suite_name(benchmark_name: str) -> str:
    if "matrix_free_n_scaling" in benchmark_name:
        return "matrix_free_n_scaling"
    if "single_output_mean_noise_scaling" in benchmark_name:
        return "extended_feature_scaling"
    if "single_output_ard_scaling" in benchmark_name:
        return "single_output_ard_scaling"
    if "single_output_preset_sweep" in benchmark_name:
        return "single_output_preset_sweep"
    if "single_output_extensive_scaling" in benchmark_name:
        return "single_output_extensive_scaling"
    if "scaling_certification" in benchmark_name:
        return "scaling_certification"
    return "single_output_scaling"


def _apply_case_variant(case_id: str, case_variant: str | None) -> str:
    if case_variant in (None, "", "default"):
        return case_id
    return f"{case_id}.variant.{case_variant}"


def run_single_output_scaling_subprocess(
    *,
    method: str,
    n_train: int,
    d: int,
    framework: str,
    prediction_mode: str,
    tier: str,
    benchmark_variety: str,
    benchmark_track: str,
    n_selection_policy: str | None,
    size_role: str | None,
    max_iterations: int,
    enable_early_stopping: bool,
    benchmark_name: str,
    mojogp_preset: str | None,
    data_options: dict[str, object] | None,
    specialization: dict[str, object] | None,
    study_id: str | None = None,
    trial_id: str | None = None,
    objective_name: str | None = None,
    objective_metric: str | None = None,
    constraint_json: dict[str, object] | None = None,
    results_dir: Path,
    context: BenchmarkRuntimeContext | None = None,
    ard: bool = False,
    relevant_dims: int | None = None,
    mojogp_solver_policy: str = "strict_fair",
    case_variant: str | None = None,
    comparison_mojogp_case_variant: str | None = None,
    timeout_s: int | None = None,
):
    context = context or get_or_create_default_context()
    specialization = prepare_specialization_payload(
        context.session_store,
        specialization,
        created_at=utc_now_iso(),
    )
    effective_data_options = dict(data_options or {})
    if ard:
        effective_data_options["dataset_family"] = "structured_ard"
        effective_data_options["relevant_dims"] = int(
            relevant_dims or effective_data_options.get("relevant_dims", min(3, d))
        )
    effective_relevant_dims = int(
        effective_data_options.get("relevant_dims", relevant_dims or min(3, d))
    )
    dataset_id, dataset_path, dataset_meta_path = _ensure_single_output_dataset(
        method=method,
        n_train=n_train,
        d=d,
        data_options=effective_data_options,
        context=context,
    )
    suite_name = _single_output_scaling_suite_name(benchmark_name)
    if suite_name == "single_output_ard_scaling":
        base_case_id = single_output_ard_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
            relevant_dims=effective_relevant_dims,
        )
        case_id = single_output_ard_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
            relevant_dims=effective_relevant_dims,
            specialization=specialization,
        )
        benchmark_group_id = single_output_ard_scaling_group_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
        )
    else:
        base_case_id = single_output_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
        )
        case_id = single_output_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
            specialization=specialization,
        )
        benchmark_group_id = single_output_scaling_group_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
        )
    base_case_id = _apply_case_variant(base_case_id, case_variant)
    case_id = _apply_case_variant(case_id, case_variant)
    if case_variant not in (None, "", "default"):
        benchmark_group_id = f"{benchmark_group_id}.variant.{case_variant}"
    config = merge_specialization_config(
        {
            "framework": framework,
            "suite_name": suite_name,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "n": n_train,
            "d": d,
            "tier": tier,
            "benchmark_variety": benchmark_variety,
            "benchmark_track": benchmark_track,
            "dataset_id": dataset_id,
            "ard": ard,
            "relevant_dims": effective_relevant_dims if ard else None,
            "mojogp_solver_policy": mojogp_solver_policy,
            "case_variant": case_variant,
        },
        specialization,
        base_case_id=base_case_id,
    )
    config = merge_study_trial_config(
        config,
        study_id=study_id,
        trial_id=trial_id,
        objective_name=objective_name,
        objective_metric=objective_metric,
        constraint_json=None if constraint_json is None else dict(constraint_json),
    )
    context.session_store.register_case(
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        framework=framework,
        suite_name=suite_name,
        benchmark_name=benchmark_name,
        config=config,
    )

    comparison_id = None
    if framework in {"mojogp", "gpytorch"}:
        comparison_case_variant = (
            case_variant if framework == "mojogp" else comparison_mojogp_case_variant
        )
        if suite_name == "single_output_ard_scaling":
            mojogp_case_id = single_output_ard_scaling_case_id(
                framework="mojogp",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                relevant_dims=effective_relevant_dims,
                specialization=specialization if framework == "mojogp" else None,
            )
            mojogp_case_id = _apply_case_variant(mojogp_case_id, comparison_case_variant)
            gpytorch_case_id = single_output_ard_scaling_case_id(
                framework="gpytorch",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                relevant_dims=effective_relevant_dims,
            )
        else:
            mojogp_case_id = single_output_scaling_case_id(
                framework="mojogp",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                specialization=specialization if framework == "mojogp" else None,
            )
            mojogp_case_id = _apply_case_variant(mojogp_case_id, comparison_case_variant)
            gpytorch_case_id = single_output_scaling_case_id(
                framework="gpytorch",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
        comparison_id = comparison_id_from_cases(mojogp_case_id, gpytorch_case_id)

    payload = {
        "method": method,
        "n_train": n_train,
        "d": d,
        "framework": framework,
        "prediction_mode": prediction_mode,
        "tier": tier,
        "benchmark_variety": benchmark_variety,
        "benchmark_track": benchmark_track,
        "n_selection_policy": n_selection_policy,
        "size_role": size_role,
        "max_iterations": max_iterations,
        "enable_early_stopping": enable_early_stopping,
        "benchmark_name": benchmark_name,
        "mojogp_preset": mojogp_preset,
        "data_options": effective_data_options,
        "ard": ard,
        "relevant_dims": effective_relevant_dims if ard else relevant_dims,
        "results_dir": str(results_dir),
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_meta_path": str(dataset_meta_path),
        "specialization": specialization,
        "mojogp_solver_policy": mojogp_solver_policy,
        "case_variant": case_variant,
        "comparison_mojogp_case_variant": comparison_mojogp_case_variant,
    }
    if framework == "mojogp":
        benchmark = run_mojogp_single_output_scaling_module(
            payload=payload,
            session_store=context.session_store,
            session_id=context.session_id,
            case_id=case_id,
            benchmark_group_id=benchmark_group_id,
            benchmark_name=benchmark_name,
            git=context.git,
            profiling=context.profiling,
            config=config,
            dataset_id=dataset_id,
            comparison_id=comparison_id,
            timeout_s=timeout_s,
        )
    elif framework == "gpytorch":
        benchmark = run_gpytorch_single_output_scaling_module(
            payload=payload,
            session_store=context.session_store,
            session_id=context.session_id,
            case_id=case_id,
            benchmark_group_id=benchmark_group_id,
            benchmark_name=benchmark_name,
            git=context.git,
            profiling=context.profiling,
            config=config,
            dataset_id=dataset_id,
            comparison_id=comparison_id,
            timeout_s=timeout_s,
        )
    else:
        raise ValueError(f"Unsupported framework '{framework}'")
    result = benchmark.loaded_result
    result.config.setdefault("dataset_id", dataset_id)
    result.config.setdefault("benchmark_group_id", benchmark_group_id)
    result.config.setdefault("case_id", case_id)
    result.config.setdefault("artifact_id", benchmark.artifact_id)

    fairness_note = str(result.config.get("fairness_note", ""))
    fairness_axes = dict(result.config.get("fairness_axes", {}))
    comparison_class = str(result.config.get("comparison_class", "unknown"))
    if comparison_id is not None:
        comparison_case_variant = (
            case_variant if framework == "mojogp" else comparison_mojogp_case_variant
        )
        if suite_name == "single_output_ard_scaling":
            registered_mojogp_case_id = single_output_ard_scaling_case_id(
                framework="mojogp",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                relevant_dims=effective_relevant_dims,
                specialization=specialization if framework == "mojogp" else None,
            )
            registered_mojogp_case_id = _apply_case_variant(
                registered_mojogp_case_id,
                comparison_case_variant,
            )
            registered_gpytorch_case_id = single_output_ard_scaling_case_id(
                framework="gpytorch",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                relevant_dims=effective_relevant_dims,
            )
        else:
            registered_mojogp_case_id = single_output_scaling_case_id(
                framework="mojogp",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                specialization=specialization if framework == "mojogp" else None,
            )
            registered_mojogp_case_id = _apply_case_variant(
                registered_mojogp_case_id,
                comparison_case_variant,
            )
            registered_gpytorch_case_id = single_output_scaling_case_id(
                framework="gpytorch",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
            )
        context.session_store.register_comparison(
            comparison_id=comparison_id,
            mojogp_case_id=registered_mojogp_case_id,
            gpytorch_case_id=registered_gpytorch_case_id,
            comparison_class=comparison_class,
            fairness_note=fairness_note,
            fairness_axes=fairness_axes,
            mojogp_specialization_key=(
                None
                if specialization["specialization_key"] == "default"
                else str(specialization["specialization_key"])
            ),
            specialization_family=(
                None
                if specialization["specialization_key"] == "default"
                else str(specialization["specialization_family"])
            ),
            specialization_config=(
                {}
                if specialization["specialization_key"] == "default"
                else dict(specialization["specialization_config"])
            ),
        )
        context.session_store.export_session_json(context.session_id)
    return result
