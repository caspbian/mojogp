"""ARD relevance-recovery metrics for benchmark rows."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_lengthscale_vector(lengthscales: Any) -> np.ndarray:
    values = np.asarray(lengthscales, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)

    # Average tied ranks so constant vectors produce zero variance cleanly.
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            avg_rank = float(np.mean(np.arange(start, end, dtype=np.float64)))
            ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size != b.size or a.size < 2:
        return None
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - np.mean(ra)
    rb = rb - np.mean(rb)
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom <= 0.0:
        return None
    return float(np.dot(ra, rb) / denom)


def compute_ard_relevance_metrics(
    lengthscales: Any,
    *,
    relevant_indices: list[int] | tuple[int, ...],
    irrelevant_indices: list[int] | tuple[int, ...] | None = None,
    gpytorch_lengthscales: Any | None = None,
) -> dict[str, Any]:
    """Compute benchmark telemetry for ARD relevance recovery.

    Smaller lengthscales indicate higher inferred relevance. The metrics are
    intentionally conservative telemetry for structured-data benchmarks rather
    than claims of absolute GP-prior lengthscale recovery.
    """

    ls = _as_lengthscale_vector(lengthscales)
    relevant = [int(idx) for idx in relevant_indices]
    if irrelevant_indices is None:
        irrelevant = [idx for idx in range(ls.size) if idx not in set(relevant)]
    else:
        irrelevant = [int(idx) for idx in irrelevant_indices]

    valid = ls.size > 0 and all(0 <= idx < ls.size for idx in relevant + irrelevant)
    if not valid or not relevant or not irrelevant:
        return {
            "learned_lengthscales": ls.tolist(),
            "mean_relevant_lengthscale": None,
            "mean_irrelevant_lengthscale": None,
            "relevance_separation_ratio": None,
            "pairwise_relevance_accuracy": None,
            "top_k_relevance_hit_rate": None,
            "lengthscale_spearman_vs_gpytorch": None,
            "lengthscale_rel_rmse_vs_gpytorch": None,
            "ard_quality_status": "unavailable",
        }

    relevant_ls = ls[relevant]
    irrelevant_ls = ls[irrelevant]
    mean_relevant = float(np.mean(relevant_ls))
    mean_irrelevant = float(np.mean(irrelevant_ls))
    pairwise = float(np.mean(relevant_ls[:, None] < irrelevant_ls[None, :]))
    k = len(relevant)
    top_k = set(np.argsort(ls)[:k].tolist())
    hit_rate = float(len(top_k.intersection(relevant)) / max(k, 1))
    if mean_relevant > 0.0:
        separation = mean_irrelevant / mean_relevant
    else:
        separation = None

    spearman = None
    rel_rmse = None
    if gpytorch_lengthscales is not None:
        gp_ls = _as_lengthscale_vector(gpytorch_lengthscales)
        if gp_ls.size == ls.size:
            spearman = _spearman(ls, gp_ls)
            denom = np.maximum(np.abs(gp_ls), 1e-8)
            rel_rmse = float(np.sqrt(np.mean(((ls - gp_ls) / denom) ** 2)))

    if pairwise >= 0.5 and np.isfinite(mean_relevant) and np.isfinite(mean_irrelevant):
        status = "pass"
    else:
        status = "weak"

    return {
        "learned_lengthscales": ls.tolist(),
        "mean_relevant_lengthscale": mean_relevant,
        "mean_irrelevant_lengthscale": mean_irrelevant,
        "relevance_separation_ratio": separation,
        "pairwise_relevance_accuracy": pairwise,
        "top_k_relevance_hit_rate": hit_rate,
        "lengthscale_spearman_vs_gpytorch": spearman,
        "lengthscale_rel_rmse_vs_gpytorch": rel_rmse,
        "ard_quality_status": status,
    }
