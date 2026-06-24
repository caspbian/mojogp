"""Built-in specialization helpers and rules."""

from __future__ import annotations

from dataclasses import dataclass

from mojogp.kernel import KernelNode, KernelType

from .descriptor import WorkloadDescriptor, bucket_dimension, bucket_training_size
from .profile import SpecializationProfile
from .rules import SpecializationRule


_LOW_D_STATIONARY_TYPES = {
    KernelType.RBF,
    KernelType.MATERN12,
    KernelType.MATERN32,
    KernelType.MATERN52,
    KernelType.RQ,
}


def manual_benchmark_override_profile(
    specialization_key: str,
    *,
    schedule_overrides: dict[str, object] | None = None,
    ncols_hint: tuple[int, ...] = (),
    module_suffix: str | None = None,
    notes: str | None = None,
    policy_version: str = "v1",
) -> SpecializationProfile:
    return SpecializationProfile(
        specialization_key=specialization_key,
        source="benchmark",
        policy_version=policy_version,
        schedule_overrides=None if schedule_overrides is None else dict(schedule_overrides),
        ncols_hint=ncols_hint,
        module_suffix=module_suffix,
        notes=notes or "Explicit benchmark override profile.",
        default_equivalent=False,
    )


EXISTING_LOW_D_STATIONARY_LEAF_PROFILE = SpecializationProfile(
    specialization_key="existing_low_d_stationary_leaf",
    source="builtin_rule",
    policy_version="v1",
    notes="Mirrors existing narrow low-d stationary-leaf schedule policy.",
    default_equivalent=True,
)


def existing_materialized_predict_ncols_profile(
    ncols_hint: tuple[int, ...],
) -> SpecializationProfile:
    return SpecializationProfile(
        specialization_key="existing_materialized_predict_ncols",
        source="builtin_rule",
        policy_version="v1",
        ncols_hint=ncols_hint,
        notes="Mirrors existing materialized SingleOutputGP NCOLS hint.",
        default_equivalent=True,
    )


def _unwrap_scale_nodes(kernel: KernelNode) -> KernelNode:
    node = kernel
    while getattr(node, "operator", None) == "scale" and getattr(node, "left", None) is not None:
        node = node.left
    return node


def infer_leaf_family(kernel: KernelNode) -> str | None:
    node = _unwrap_scale_nodes(kernel)
    kernel_type = getattr(node, "kernel_type", None)
    if kernel_type is None:
        return None
    return str(kernel_type.name).lower()


def infer_kernel_family(kernel: KernelNode) -> str:
    node = _unwrap_scale_nodes(kernel)
    if getattr(node, "operator", None) is None and getattr(node, "kernel_type", None) is not None:
        return infer_leaf_family(node) or "unknown"
    if getattr(kernel, "has_categorical", lambda: False)():
        return "mixed"
    if getattr(kernel, "operator", None) is not None:
        return str(kernel.operator)
    return "composite"


def is_low_d_stationary_leaf(kernel: KernelNode, dim: int) -> bool:
    node = _unwrap_scale_nodes(kernel)
    return (
        getattr(node, "operator", None) is None
        and getattr(node, "kernel_type", None) in _LOW_D_STATIONARY_TYPES
        and dim <= 10
    )


def build_single_output_descriptor(
    *,
    kernel: KernelNode,
    dim: int,
    training_method: str | None,
    prediction_mode: str | None = None,
    n_train: int | None = None,
    gpu_target: str | None = None,
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        wrapper_type="SingleOutputGP",
        model_family="single_output",
        workload_kind="train_predict",
        training_method=training_method,
        prediction_mode=prediction_mode,
        kernel_family=infer_kernel_family(kernel),
        leaf_family=infer_leaf_family(kernel),
        is_ard=bool(kernel.has_ard()),
        is_composite=getattr(kernel, "operator", None) is not None,
        is_mixed=bool(kernel.has_categorical()),
        dim=int(dim),
        dim_bucket=bucket_dimension(int(dim)),
        n_train=None if n_train is None else int(n_train),
        n_bucket=bucket_training_size(n_train),
        num_tasks=None,
        gpu_target=gpu_target,
    )


def build_multi_output_descriptor(
    *,
    kernel: KernelNode,
    dim: int,
    training_method: str | None,
    num_tasks: int,
    prediction_mode: str | None = None,
    n_train: int | None = None,
    gpu_target: str | None = None,
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        wrapper_type="MultiOutputGP",
        model_family="multi_output",
        workload_kind="train_predict",
        training_method=training_method,
        prediction_mode=prediction_mode,
        kernel_family=infer_kernel_family(kernel),
        leaf_family=infer_leaf_family(kernel),
        is_ard=bool(kernel.has_ard()),
        is_composite=getattr(kernel, "operator", None) is not None,
        is_mixed=bool(kernel.has_categorical()),
        dim=int(dim),
        dim_bucket=bucket_dimension(int(dim)),
        n_train=None if n_train is None else int(n_train),
        n_bucket=bucket_training_size(n_train),
        num_tasks=int(num_tasks),
        gpu_target=gpu_target,
    )


@dataclass(frozen=True)
class LowDStationaryLeafRule(SpecializationRule):
    def matches(self, descriptor: WorkloadDescriptor) -> bool:
        return (
            descriptor.leaf_family in {"rbf", "matern12", "matern32", "matern52", "rq"}
            and descriptor.dim <= 10
            and not descriptor.is_composite
            and not descriptor.is_mixed
        )

    def build_profile(self, descriptor: WorkloadDescriptor) -> SpecializationProfile:
        return EXISTING_LOW_D_STATIONARY_LEAF_PROFILE

    def priority(self) -> int:
        return 20


@dataclass(frozen=True)
class MaterializedPredictNcolsRule(SpecializationRule):
    ncols_hint: tuple[int, ...]

    def matches(self, descriptor: WorkloadDescriptor) -> bool:
        return (
            descriptor.wrapper_type == "SingleOutputGP"
            and descriptor.training_method == "materialized"
            and not descriptor.is_mixed
        )

    def build_profile(self, descriptor: WorkloadDescriptor) -> SpecializationProfile:
        return existing_materialized_predict_ncols_profile(self.ncols_hint)

    def priority(self) -> int:
        return 10
