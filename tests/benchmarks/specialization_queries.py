"""Specialization-aware convenience queries for the benchmark SQLite store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import DEFAULT_DB_PATH
from .session_store import BenchmarkSessionStore

_JSON_FIELDS = {
    "config_json",
    "specialization_descriptor_json",
    "specialization_config_json",
    "contract_summary_json",
}


def load_store(db_path: Path = DEFAULT_DB_PATH) -> BenchmarkSessionStore:
    return BenchmarkSessionStore(db_path)


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in '{["':
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def normalize_run_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in _JSON_FIELDS:
        if key in normalized:
            normalized[key] = _decode_json(normalized[key])
    return normalized


def runs_for_base_case(
    base_case_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    return [
        normalize_run_row(row)
        for row in load_store(db_path).fetch_runs_by_base_case(base_case_id)
    ]


def runs_for_specialization(
    specialization_key: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    return [
        normalize_run_row(row)
        for row in load_store(db_path).fetch_runs_by_specialization_key(specialization_key)
    ]


def group_variants(
    benchmark_group_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, list[dict[str, Any]]]:
    rows = [
        normalize_run_row(row)
        for row in load_store(db_path).fetch_runs_by_group(benchmark_group_id)
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        base_case_id = row.get("base_case_id") or row.get("case_id")
        grouped.setdefault(str(base_case_id), []).append(row)
    return grouped


def _is_default_row(row: dict[str, Any]) -> bool:
    key = row.get("specialization_key")
    mode = row.get("specialization_mode")
    return key in (None, "default") or mode in (None, "disabled")


def _latest_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            str(row.get("finished_at") or ""),
            str(row.get("started_at") or ""),
            str(row.get("run_id") or ""),
        ),
    )


def default_and_variants(
    base_case_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    rows = runs_for_base_case(base_case_id, db_path=db_path)
    default_rows = [row for row in rows if _is_default_row(row)]
    variant_rows = [row for row in rows if not _is_default_row(row)]
    return {
        "base_case_id": base_case_id,
        "default_runs": default_rows,
        "variant_runs": variant_rows,
        "default_run": _latest_run(default_rows),
    }


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def variant_summary(
    base_case_id: str,
    db_path: Path = DEFAULT_DB_PATH,
    *,
    objective: str = "training_time_s",
) -> dict[str, Any]:
    grouped = default_and_variants(base_case_id, db_path=db_path)
    default_run = grouped["default_run"]
    variants: list[dict[str, Any]] = []
    for row in grouped["variant_runs"]:
        summary = {
            "run": row,
            "specialization_key": row.get("specialization_key"),
            "specialization_mode": row.get("specialization_mode"),
            "training_time_delta_s": (
                None if default_run is None else _delta(row.get("training_time_s"), default_run.get("training_time_s"))
            ),
            "prediction_mean_time_delta_s": (
                None
                if default_run is None
                else _delta(row.get("prediction_mean_time_s"), default_run.get("prediction_mean_time_s"))
            ),
            "end_to_end_time_delta_s": (
                None if default_run is None else _delta(row.get("end_to_end_time_s"), default_run.get("end_to_end_time_s"))
            ),
            "scaling_peak_gpu_delta_mb": (
                None if default_run is None else _delta(row.get("scaling_peak_gpu_mb"), default_run.get("scaling_peak_gpu_mb"))
            ),
            "objective_metric": objective,
            "objective_value": row.get(objective),
            "objective_delta": (
                None if default_run is None else _delta(row.get(objective), default_run.get(objective))
            ),
        }
        objective_delta = summary["objective_delta"]
        summary["better_than_default"] = None if objective_delta is None else objective_delta < 0.0
        variants.append(summary)

    best_variant = None
    comparable_variants = [variant for variant in variants if variant["objective_value"] is not None]
    if comparable_variants:
        best_variant = min(comparable_variants, key=lambda variant: float(variant["objective_value"]))

    return {
        "base_case_id": base_case_id,
        "default_run": default_run,
        "variants": variants,
        "best_variant": best_variant,
        "objective_metric": objective,
    }


def guard_lane_regressions(
    specialization_key: str,
    guard_case_ids: list[str],
    db_path: Path = DEFAULT_DB_PATH,
    *,
    metric: str = "training_time_s",
    tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    specialization_rows = runs_for_specialization(specialization_key, db_path=db_path)
    for base_case_id in guard_case_ids:
        grouped = default_and_variants(base_case_id, db_path=db_path)
        default_run = grouped["default_run"]
        matching_variant = _latest_run(
            [
                row
                for row in specialization_rows
                if (row.get("base_case_id") or row.get("case_id")) == base_case_id
            ]
        )
        metric_delta = (
            None
            if default_run is None or matching_variant is None
            else _delta(matching_variant.get(metric), default_run.get(metric))
        )
        regressions.append(
            {
                "base_case_id": base_case_id,
                "metric": metric,
                "default_run": default_run,
                "variant_run": matching_variant,
                "metric_delta": metric_delta,
                "regressed": None if metric_delta is None else metric_delta > tolerance,
            }
        )
    return regressions
