"""Explicit benchmark case and comparison mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mojogp.specialization import apply_specialization_to_case_id


@dataclass(frozen=True)
class BenchmarkCaseDefinition:
    case_id: str
    benchmark_group_id: str
    framework: str
    suite_name: str
    benchmark_name: str
    config: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkComparisonDefinition:
    comparison_id: str
    mojogp_case_id: str
    gpytorch_case_id: str
    comparison_class: str
    fairness_note: str
    fairness_axes: dict[str, Any]


def single_output_scaling_case_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
    n_train: int,
    d: int,
    specialization: dict[str, Any] | None = None,
) -> str:
    base_case_id = (
        f"{framework}.single_output.scaling.{training_method}."
        f"{prediction_mode}.n{n_train}.d{d}"
    )
    return apply_specialization_to_case_id(base_case_id, specialization)


def single_output_scaling_group_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
) -> str:
    return f"{framework}.single_output.scaling.{training_method}.{prediction_mode}"


def single_output_ard_scaling_case_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
    n_train: int,
    d: int,
    relevant_dims: int,
    specialization: dict[str, Any] | None = None,
) -> str:
    base_case_id = (
        f"{framework}.single_output.ard_scaling.{training_method}."
        f"{prediction_mode}.n{n_train}.d{d}.rel{relevant_dims}"
    )
    return apply_specialization_to_case_id(base_case_id, specialization)


def single_output_ard_scaling_group_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
) -> str:
    return f"{framework}.single_output.ard_scaling.{training_method}.{prediction_mode}"


def multi_output_scaling_case_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
    n_train: int,
    d: int,
    num_tasks: int,
    specialization: dict[str, Any] | None = None,
) -> str:
    base_case_id = (
        f"{framework}.multi_output.scaling.{training_method}."
        f"{prediction_mode}.n{n_train}.d{d}.t{num_tasks}"
    )
    return apply_specialization_to_case_id(base_case_id, specialization)


def multi_output_scaling_group_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
    num_tasks: int,
) -> str:
    return (
        f"{framework}.multi_output.scaling.{training_method}."
        f"{prediction_mode}.t{num_tasks}"
    )


def multi_output_ard_scaling_case_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
    n_train: int,
    d: int,
    num_tasks: int,
    relevant_dims: int,
    specialization: dict[str, Any] | None = None,
) -> str:
    base_case_id = (
        f"{framework}.multi_output.ard_scaling.{training_method}."
        f"{prediction_mode}.n{n_train}.d{d}.t{num_tasks}.rel{relevant_dims}"
    )
    return apply_specialization_to_case_id(base_case_id, specialization)


def multi_output_ard_scaling_group_id(
    *,
    framework: str,
    training_method: str,
    prediction_mode: str,
    num_tasks: int,
) -> str:
    return (
        f"{framework}.multi_output.ard_scaling.{training_method}."
        f"{prediction_mode}.t{num_tasks}"
    )


def comparison_id_from_cases(mojogp_case_id: str, gpytorch_case_id: str) -> str:
    return f"compare::{mojogp_case_id}::{gpytorch_case_id}"
