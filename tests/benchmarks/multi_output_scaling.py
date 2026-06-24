"""Benchmark runner integration for multi-output scaling cases."""

from __future__ import annotations

from pathlib import Path

from mojogp.specialization import merge_specialization_config

from tests.benchmarks.gpytorch.multi_output_scaling import (
    run_gpytorch_multi_output_scaling_module,
)
from tests.benchmarks.mojogp.multi_output_scaling import (
    run_mojogp_multi_output_scaling_module,
)
from tests.shared.benchmarking.data_generators import (
    generate_multi_output_data,
    generate_multi_output_structured_ard_data,
)
from tests.shared.benchmarking.report import load_benchmark_result

from .comparison_registry import (
    comparison_id_from_cases,
    multi_output_ard_scaling_case_id,
    multi_output_ard_scaling_group_id,
    multi_output_scaling_case_id,
    multi_output_scaling_group_id,
)
from .dataset_manifest import DatasetSpec, ensure_dataset, save_dataset_artifact_bundle
from .harness_runner import run_benchmark_module
from .multi_output_timeout_policy import multi_output_scaling_timeout_s
from .specialization_adapter import prepare_specialization_payload
from .specialization_studies import merge_study_trial_config
from .runtime import BenchmarkRuntimeContext, get_or_create_default_context
from .preflight import utc_now_iso
from .prediction_workload import BENCHMARK_PREDICTION_N_TEST


def _ensure_multi_output_dataset(
    *,
    n_train: int,
    d: int,
    num_tasks: int,
    context: BenchmarkRuntimeContext,
    ard: bool = False,
    relevant_dims: int | None = None,
) -> tuple[str, Path, Path]:
    if ard:
        effective_relevant_dims = int(relevant_dims or min(3, d))
        spec = DatasetSpec(
            generator_name="multi_output_structured_ard",
            config={
                "n_train": n_train,
                "n_test": BENCHMARK_PREDICTION_N_TEST,
                "d": d,
                "num_tasks": num_tasks,
                "relevant_dims": effective_relevant_dims,
                "seed": 1000 + n_train + d * 100 + effective_relevant_dims * 17,
            },
        )
        dataset_id, npz_path, meta_path = ensure_dataset(
            spec,
            generate_multi_output_structured_ard_data,
        )
    else:
        spec = DatasetSpec(
            generator_name="multi_output_scaling_gp_prior",
            config={
                "n_train": n_train,
                "n_test": BENCHMARK_PREDICTION_N_TEST,
                "d": d,
                "num_tasks": num_tasks,
                "kernel_type": "rbf",
                "task_correlation": "medium",
                "seed": 1000 + n_train + d * 100,
            },
        )
        dataset_id, npz_path, meta_path = ensure_dataset(spec, generate_multi_output_data)
    artifact_path, artifact_sha256, _ = save_dataset_artifact_bundle(
        dataset_id,
        dataset_npz_path=npz_path,
        dataset_meta_path=meta_path,
    )
    context.session_store.register_dataset(
        dataset_id=dataset_id,
        generator_name=spec.generator_name,
        config=spec.config,
        seed=int(spec.config["seed"]),
        artifact_path=str(artifact_path),
        artifact_sha256=artifact_sha256,
        created_at=utc_now_iso(),
    )
    return dataset_id, npz_path, meta_path


def run_multi_output_scaling_subprocess(
    *,
    framework: str,
    prediction_mode: str,
    method: str,
    n_train: int,
    d: int,
    num_tasks: int,
    tier: str,
    specialization: dict[str, object] | None,
    ard: bool = False,
    relevant_dims: int | None = None,
    benchmark_name: str = "multi_output_scaling",
    study_id: str | None = None,
    trial_id: str | None = None,
    objective_name: str | None = None,
    objective_metric: str | None = None,
    constraint_json: dict[str, object] | None = None,
    results_dir: Path,
    context: BenchmarkRuntimeContext | None = None,
    timeout_s: int | None = None,
):
    context = context or get_or_create_default_context()
    specialization = prepare_specialization_payload(
        context.session_store,
        specialization,
        created_at=utc_now_iso(),
    )
    dataset_id, dataset_path, dataset_meta_path = _ensure_multi_output_dataset(
        n_train=n_train,
        d=d,
        num_tasks=num_tasks,
        context=context,
        ard=ard,
        relevant_dims=relevant_dims,
    )
    effective_relevant_dims = int(relevant_dims or min(3, d))
    if ard or benchmark_name == "multi_output_ard_scaling":
        benchmark_name = "multi_output_ard_scaling"
        base_case_id = multi_output_ard_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
            num_tasks=num_tasks,
            relevant_dims=effective_relevant_dims,
        )
        case_id = multi_output_ard_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
            num_tasks=num_tasks,
            relevant_dims=effective_relevant_dims,
            specialization=specialization,
        )
        benchmark_group_id = multi_output_ard_scaling_group_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            num_tasks=num_tasks,
        )
    else:
        base_case_id = multi_output_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
            num_tasks=num_tasks,
        )
        case_id = multi_output_scaling_case_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            n_train=n_train,
            d=d,
            num_tasks=num_tasks,
            specialization=specialization,
        )
        benchmark_group_id = multi_output_scaling_group_id(
            framework=framework,
            training_method=method,
            prediction_mode=prediction_mode,
            num_tasks=num_tasks,
        )
    effective_timeout_s = multi_output_scaling_timeout_s(
        framework=framework,
        method=method,
        prediction_mode=prediction_mode,
        tier=tier,
        timeout_s=timeout_s,
    )
    config = merge_specialization_config(
        {
            "framework": framework,
            "training_method": method,
            "prediction_mode": prediction_mode,
            "n": n_train,
            "d": d,
            "num_tasks": num_tasks,
            "ard": ard,
            "relevant_dims": effective_relevant_dims if ard else None,
            "tier": tier,
            "dataset_id": dataset_id,
            "timeout_s": effective_timeout_s,
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
        suite_name=benchmark_name,
        benchmark_name=benchmark_name,
        config=config,
    )

    comparison_id = None
    if framework in {"mojogp", "gpytorch"}:
        if benchmark_name == "multi_output_ard_scaling":
            comparison_id = comparison_id_from_cases(
                multi_output_ard_scaling_case_id(
                    framework="mojogp",
                    training_method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    d=d,
                    num_tasks=num_tasks,
                    relevant_dims=effective_relevant_dims,
                    specialization=specialization if framework == "mojogp" else None,
                ),
                multi_output_ard_scaling_case_id(
                    framework="gpytorch",
                    training_method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    d=d,
                    num_tasks=num_tasks,
                    relevant_dims=effective_relevant_dims,
                ),
            )
        else:
            comparison_id = comparison_id_from_cases(
                multi_output_scaling_case_id(
                    framework="mojogp",
                    training_method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    d=d,
                    num_tasks=num_tasks,
                    specialization=specialization if framework == "mojogp" else None,
                ),
                multi_output_scaling_case_id(
                    framework="gpytorch",
                    training_method=method,
                    prediction_mode=prediction_mode,
                    n_train=n_train,
                    d=d,
                    num_tasks=num_tasks,
                ),
            )

    payload = {
        "framework": framework,
        "prediction_mode": prediction_mode,
        "method": method,
        "n_train": n_train,
        "d": d,
        "num_tasks": num_tasks,
        "tier": tier,
        "results_dir": str(results_dir),
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_path),
        "dataset_meta_path": str(dataset_meta_path),
        "specialization": specialization,
        "ard": ard,
        "relevant_dims": effective_relevant_dims if ard else relevant_dims,
    }
    if framework == "mojogp":
        benchmark = run_mojogp_multi_output_scaling_module(
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
            timeout_s=effective_timeout_s,
        )
    elif framework == "gpytorch":
        benchmark = run_gpytorch_multi_output_scaling_module(
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
            timeout_s=effective_timeout_s,
        )
    else:
        raise ValueError(f"Unsupported framework '{framework}'")
    result = benchmark.loaded_result
    result.config.setdefault("dataset_id", dataset_id)
    result.config.setdefault("benchmark_group_id", benchmark_group_id)
    result.config.setdefault("case_id", case_id)
    result.config.setdefault("artifact_id", benchmark.artifact_id)
    result.config.setdefault("timeout_s", effective_timeout_s)
    fairness_note = str(result.config.get("fairness_note", ""))
    fairness_axes = dict(result.config.get("fairness_axes", {}))
    comparison_class = str(result.config.get("comparison_class", "unknown"))
    if comparison_id is not None:
        if benchmark_name == "multi_output_ard_scaling":
            registered_mojogp_case_id = multi_output_ard_scaling_case_id(
                framework="mojogp",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                num_tasks=num_tasks,
                relevant_dims=effective_relevant_dims,
                specialization=specialization if framework == "mojogp" else None,
            )
            registered_gpytorch_case_id = multi_output_ard_scaling_case_id(
                framework="gpytorch",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                num_tasks=num_tasks,
                relevant_dims=effective_relevant_dims,
            )
        else:
            registered_mojogp_case_id = multi_output_scaling_case_id(
                framework="mojogp",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                num_tasks=num_tasks,
                specialization=specialization if framework == "mojogp" else None,
            )
            registered_gpytorch_case_id = multi_output_scaling_case_id(
                framework="gpytorch",
                training_method=method,
                prediction_mode=prediction_mode,
                n_train=n_train,
                d=d,
                num_tasks=num_tasks,
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
    return result
