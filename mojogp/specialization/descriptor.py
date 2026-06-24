"""Workload descriptors for specialization selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def bucket_dimension(dim: int) -> str:
    if dim <= 4:
        return "tiny"
    if dim <= 10:
        return "low"
    if dim <= 32:
        return "medium"
    return "high"


def bucket_training_size(n_train: int | None) -> str | None:
    if n_train is None or n_train <= 0:
        return None
    if n_train <= 2_000:
        return "small"
    if n_train <= 10_000:
        return "medium"
    if n_train <= 50_000:
        return "large"
    return "xlarge"


@dataclass(frozen=True)
class WorkloadDescriptor:
    """Canonical description of a workload lane.

    This is used for specialization matching and benchmark identity metadata.
    """

    wrapper_type: str
    model_family: str
    workload_kind: str
    training_method: str | None
    prediction_mode: str | None
    kernel_family: str
    leaf_family: str | None
    is_ard: bool
    is_composite: bool
    is_mixed: bool
    dim: int
    dim_bucket: str
    n_train: int | None
    n_bucket: str | None
    num_tasks: int | None
    gpu_target: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrapper_type": self.wrapper_type,
            "model_family": self.model_family,
            "workload_kind": self.workload_kind,
            "training_method": self.training_method,
            "prediction_mode": self.prediction_mode,
            "kernel_family": self.kernel_family,
            "leaf_family": self.leaf_family,
            "is_ard": self.is_ard,
            "is_composite": self.is_composite,
            "is_mixed": self.is_mixed,
            "dim": self.dim,
            "dim_bucket": self.dim_bucket,
            "n_train": self.n_train,
            "n_bucket": self.n_bucket,
            "num_tasks": self.num_tasks,
            "gpu_target": self.gpu_target,
        }
