"""Historical baseline helpers for benchmark contracts."""

from __future__ import annotations

from typing import Any

from .contracts import HistoricalBaselineKey, rolling_median
from .session_store import BenchmarkSessionStore


def fetch_historical_metric_values(
    store: BenchmarkSessionStore,
    *,
    key: HistoricalBaselineKey,
    metric: str,
    limit: int = 20,
) -> list[float]:
    filters: dict[str, Any] = {
        "benchmark_name": key.benchmark_name,
        "framework": key.framework,
    }
    optional_filters = {
        "benchmark_group_id": key.benchmark_group_id,
        "case_id": key.case_id,
        "training_method": key.training_method,
        "prediction_mode": key.prediction_mode,
    }
    filters.update({name: value for name, value in optional_filters.items() if value is not None})

    rows = store.fetch_runs_matching(filters, limit=limit)
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        values.append(float(value))
    return values


def historical_metric_baseline(
    store: BenchmarkSessionStore,
    *,
    key: HistoricalBaselineKey,
    metric: str,
    limit: int = 20,
) -> float | None:
    return rolling_median(fetch_historical_metric_values(store, key=key, metric=metric, limit=limit))
