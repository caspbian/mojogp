"""Pre-generate benchmark datasets for active benchmark lanes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.shared.benchmarking.data_generators import (
    generate_gp_prior_data,
    generate_multi_output_data,
    generate_structured_function_data,
)

from .dataset_manifest import DatasetSpec, ensure_dataset, save_dataset_artifact_bundle
from .preflight import utc_now_iso
from .runtime import BenchmarkRuntimeContext, get_or_create_default_context


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    npz_path: Path
    meta_path: Path
    generator_name: str
    config: dict[str, object]


def _register_dataset(spec: DatasetSpec, *, context: BenchmarkRuntimeContext, npz_path: Path, meta_path: Path) -> PreparedDataset:
    dataset_id = spec.dataset_id
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
    return PreparedDataset(
        dataset_id=dataset_id,
        npz_path=npz_path,
        meta_path=meta_path,
        generator_name=spec.generator_name,
        config=spec.config,
    )


def prepare_single_output_scaling_dataset(
    *,
    dataset_family: str,
    n_train: int,
    n_test: int,
    d: int,
    seed: int,
    context: BenchmarkRuntimeContext | None = None,
    **kwargs,
) -> PreparedDataset:
    context = context or get_or_create_default_context()
    if dataset_family == "structured_function":
        spec = DatasetSpec(
            generator_name="single_output_structured_function",
            config={
                "n_train": n_train,
                "n_test": n_test,
                "d": d,
                "function_type": str(kwargs.get("function_type", "smooth")),
                "noise_level": str(kwargs.get("noise_level", "medium")),
                "seed": seed,
                "mean_offset": float(kwargs.get("mean_offset", 0.0)),
            },
        )
        dataset_id, npz_path, meta_path = ensure_dataset(spec, generate_structured_function_data)
    elif dataset_family == "gp_prior":
        x_range = kwargs.get("x_range", (-3.0, 3.0))
        spec = DatasetSpec(
            generator_name="single_output_gp_prior",
            config={
                "n_train": n_train,
                "n_test": n_test,
                "d": d,
                "kernel_type": str(kwargs.get("kernel_type", "rbf")),
                "true_lengthscale": float(kwargs.get("true_lengthscale", 1.0)),
                "true_noise": float(kwargs.get("true_noise", 0.1)),
                "true_outputscale": float(kwargs.get("true_outputscale", 1.0)),
                "seed": seed,
                "x_range": (float(x_range[0]), float(x_range[1])),
                "true_period": float(kwargs.get("true_period", 1.0)),
                "true_alpha": float(kwargs.get("true_alpha", 1.0)),
                "mean_offset": float(kwargs.get("mean_offset", 0.0)),
            },
        )
        dataset_id, npz_path, meta_path = ensure_dataset(spec, generate_gp_prior_data)
    else:
        raise ValueError(f"Unknown single-output dataset family '{dataset_family}'")
    return _register_dataset(spec, context=context, npz_path=npz_path, meta_path=meta_path)


def prepare_multi_output_scaling_dataset(
    *,
    n_train: int,
    n_test: int,
    d: int,
    num_tasks: int,
    seed: int,
    kernel_type: str = "rbf",
    task_correlation: str = "medium",
    context: BenchmarkRuntimeContext | None = None,
) -> PreparedDataset:
    context = context or get_or_create_default_context()
    spec = DatasetSpec(
        generator_name="multi_output_scaling_gp_prior",
        config={
            "n_train": n_train,
            "n_test": n_test,
            "d": d,
            "num_tasks": num_tasks,
            "kernel_type": kernel_type,
            "task_correlation": task_correlation,
            "seed": seed,
        },
    )
    dataset_id, npz_path, meta_path = ensure_dataset(spec, generate_multi_output_data)
    return _register_dataset(spec, context=context, npz_path=npz_path, meta_path=meta_path)
