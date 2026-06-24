"""Study/trial helpers for specialization benchmarking.

This module prepares the benchmark infrastructure for future search workflows
without implementing Bayesian optimization itself.
"""

from __future__ import annotations

import uuid
from typing import Any

from .preflight import utc_now_iso
from .session_store import BenchmarkSessionStore


def new_study_id() -> str:
    return f"study-{uuid.uuid4().hex[:12]}"


def new_trial_id() -> str:
    return f"trial-{uuid.uuid4().hex[:12]}"


def merge_study_trial_config(
    config: dict[str, Any],
    *,
    study_id: str | None = None,
    trial_id: str | None = None,
    objective_name: str | None = None,
    objective_metric: str | None = None,
    constraint_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(config)
    if study_id is not None:
        merged["study_id"] = study_id
    if trial_id is not None:
        merged["trial_id"] = trial_id
    if objective_name is not None:
        merged["objective_name"] = objective_name
    if objective_metric is not None:
        merged["objective_metric"] = objective_metric
    if constraint_json is not None:
        merged["constraint_json"] = dict(constraint_json)
    return merged


def register_study(
    session_store: BenchmarkSessionStore,
    *,
    benchmark_group_id: str,
    objective_name: str,
    objective_metric: str,
    constraints: dict[str, Any] | None = None,
    search_space: dict[str, Any] | None = None,
    notes: str | None = None,
    study_id: str | None = None,
    created_at: str | None = None,
) -> str:
    study_id = study_id or new_study_id()
    session_store.register_specialization_study(
        study_id=study_id,
        created_at=created_at or utc_now_iso(),
        benchmark_group_id=benchmark_group_id,
        objective_name=objective_name,
        objective_metric=objective_metric,
        constraints=dict(constraints or {}),
        search_space=dict(search_space or {}),
        notes=notes,
    )
    return study_id


def register_trial(
    session_store: BenchmarkSessionStore,
    *,
    study_id: str,
    base_case_id: str,
    specialization_key: str,
    trial_config: dict[str, Any],
    result_summary: dict[str, Any] | None = None,
    status: str = "registered",
    trial_id: str | None = None,
    created_at: str | None = None,
) -> str:
    trial_id = trial_id or new_trial_id()
    session_store.register_specialization_trial(
        trial_id=trial_id,
        study_id=study_id,
        base_case_id=base_case_id,
        specialization_key=specialization_key,
        status=status,
        trial_config=dict(trial_config),
        result_summary=dict(result_summary or {}),
        created_at=created_at or utc_now_iso(),
    )
    return trial_id


def evaluate_constraints(
    run_row: dict[str, Any],
    constraints: dict[str, Any] | None,
) -> dict[str, Any]:
    constraints = dict(constraints or {})
    checks: list[dict[str, Any]] = []
    passed = True

    for metric, threshold in dict(constraints.get("max_metrics", {})).items():
        value = run_row.get(metric)
        ok = value is not None and float(value) <= float(threshold)
        checks.append({"type": "max", "metric": metric, "threshold": threshold, "value": value, "passed": ok})
        passed = passed and ok

    for metric, threshold in dict(constraints.get("min_metrics", {})).items():
        value = run_row.get(metric)
        ok = value is not None and float(value) >= float(threshold)
        checks.append({"type": "min", "metric": metric, "threshold": threshold, "value": value, "passed": ok})
        passed = passed and ok

    for metric, expected in dict(constraints.get("equals", {})).items():
        value = run_row.get(metric)
        ok = value == expected
        checks.append({"type": "equals", "metric": metric, "expected": expected, "value": value, "passed": ok})
        passed = passed and ok

    return {
        "passed": passed,
        "checks": checks,
        "raw_constraints": constraints,
    }


def summarize_trial_from_run(
    run_row: dict[str, Any],
    *,
    objective_metric: str,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    objective_value = run_row.get(objective_metric)
    constraint_eval = evaluate_constraints(run_row, constraints)
    return {
        "run_id": run_row.get("run_id"),
        "case_id": run_row.get("case_id"),
        "base_case_id": run_row.get("base_case_id"),
        "specialization_key": run_row.get("specialization_key"),
        "objective_metric": objective_metric,
        "objective_value": objective_value,
        "constraint_evaluation": constraint_eval,
    }


def finalize_trial_from_run(
    session_store: BenchmarkSessionStore,
    *,
    trial_id: str,
    run_row: dict[str, Any],
    objective_metric: str,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_trial_from_run(
        run_row,
        objective_metric=objective_metric,
        constraints=constraints,
    )
    constraint_eval = dict(summary["constraint_evaluation"])
    session_store.update_specialization_trial_result(
        trial_id=trial_id,
        run_id=str(run_row["run_id"]),
        status="completed",
        objective_value=(
            None
            if summary["objective_value"] is None
            else float(summary["objective_value"])
        ),
        constraint_status=("passed" if constraint_eval["passed"] else "failed"),
        result_summary=summary,
    )
    return summary
