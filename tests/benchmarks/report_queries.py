"""Convenience queries for the benchmark SQLite store."""

from __future__ import annotations

from pathlib import Path

from .paths import DEFAULT_DB_PATH
from .specialization_queries import (
    default_and_variants,
    group_variants,
    guard_lane_regressions,
    runs_for_base_case,
    runs_for_specialization,
    variant_summary,
)
from .session_store import BenchmarkSessionStore


def load_store(db_path: Path = DEFAULT_DB_PATH) -> BenchmarkSessionStore:
    return BenchmarkSessionStore(db_path)


def scaling_rows(benchmark_group_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, object]]:
    return load_store(db_path).fetch_runs_by_group(benchmark_group_id)


def branch_rows(branch_name: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, object]]:
    return load_store(db_path).fetch_runs_by_branch(branch_name)


def commit_rows(commit_hash: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, object]]:
    return load_store(db_path).fetch_runs_by_commit(commit_hash)


def export_session(session_id: str, db_path: Path = DEFAULT_DB_PATH) -> Path:
    return load_store(db_path).export_session_json(session_id)


def historical_metric_rows(
    filters: dict[str, object],
    db_path: Path = DEFAULT_DB_PATH,
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    return load_store(db_path).fetch_runs_matching(filters, limit=limit)


def study_row(study_id: str, db_path: Path = DEFAULT_DB_PATH) -> dict[str, object] | None:
    return load_store(db_path).fetch_study(study_id)


def study_trials(study_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, object]]:
    return load_store(db_path).fetch_trials_for_study(study_id)


def session_rows(session_id: str, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, object]]:
    return load_store(db_path).fetch_runs_for_session(session_id)


def view_rows(
    view_name: str,
    db_path: Path = DEFAULT_DB_PATH,
    *,
    filters: dict[str, object] | None = None,
    order_by: str | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    return load_store(db_path).fetch_view_rows(
        view_name,
        filters={} if filters is None else dict(filters),
        order_by=order_by,
        limit=limit,
    )


__all__ = [
    "branch_rows",
    "commit_rows",
    "default_and_variants",
    "export_session",
    "group_variants",
    "guard_lane_regressions",
    "historical_metric_rows",
    "load_store",
    "runs_for_base_case",
    "runs_for_specialization",
    "scaling_rows",
    "session_rows",
    "study_row",
    "study_trials",
    "variant_summary",
    "view_rows",
]
