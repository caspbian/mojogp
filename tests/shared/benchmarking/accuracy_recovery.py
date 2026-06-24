"""Observational parameter-recovery helpers for accuracy benchmarks.

These helpers intentionally produce descriptive recovery records rather than
pass/fail assertions. Accuracy benchmarks use them to answer "what did the
trained model recover?" across scalar, vector, and matrix parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class ParameterRecoveryRecord:
    parameter_name: str
    truth: Any
    learned: Any
    status: str
    parameter_group: str | None = None
    parameter_index: str = ""
    abs_error: float | None = None
    rel_error: float | None = None
    signed_error: float | None = None
    log_abs_error: float | None = None
    truth_norm: float | None = None
    learned_norm: float | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.dtype == np.dtype("O"):
        return None
    return array


def _safe_norm(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(np.linalg.norm(value.reshape(-1)))


def _log_abs_error(learned: np.ndarray, truth: np.ndarray) -> float | None:
    if learned.shape != truth.shape:
        return None
    if np.any(learned <= 0.0) or np.any(truth <= 0.0):
        return None
    return float(np.linalg.norm((np.log(learned) - np.log(truth)).reshape(-1)))


def make_recovery_record(
    *,
    parameter_name: str,
    truth: Any,
    learned: Any,
    parameter_group: str | None = None,
    parameter_index: str = "",
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ParameterRecoveryRecord:
    truth_array = _numeric_array(truth)
    learned_array = _numeric_array(learned)
    metadata_dict = dict(metadata or {})

    if truth_array is None:
        return ParameterRecoveryRecord(
            parameter_name=parameter_name,
            parameter_group=parameter_group,
            parameter_index=parameter_index,
            truth=_jsonable(truth),
            learned=_jsonable(learned),
            status="missing_truth",
            notes=notes,
            metadata=metadata_dict,
        )
    if learned_array is None:
        return ParameterRecoveryRecord(
            parameter_name=parameter_name,
            parameter_group=parameter_group,
            parameter_index=parameter_index,
            truth=_jsonable(truth),
            learned=_jsonable(learned),
            status="missing_learned",
            truth_norm=_safe_norm(truth_array),
            notes=notes,
            metadata=metadata_dict,
        )
    if truth_array.shape != learned_array.shape:
        return ParameterRecoveryRecord(
            parameter_name=parameter_name,
            parameter_group=parameter_group,
            parameter_index=parameter_index,
            truth=_jsonable(truth),
            learned=_jsonable(learned),
            status="shape_mismatch",
            truth_norm=_safe_norm(truth_array),
            learned_norm=_safe_norm(learned_array),
            notes=notes,
            metadata={
                **metadata_dict,
                "truth_shape": list(truth_array.shape),
                "learned_shape": list(learned_array.shape),
            },
        )
    if not (np.all(np.isfinite(truth_array)) and np.all(np.isfinite(learned_array))):
        return ParameterRecoveryRecord(
            parameter_name=parameter_name,
            parameter_group=parameter_group,
            parameter_index=parameter_index,
            truth=_jsonable(truth),
            learned=_jsonable(learned),
            status="nonfinite",
            truth_norm=_safe_norm(truth_array),
            learned_norm=_safe_norm(learned_array),
            notes=notes,
            metadata=metadata_dict,
        )

    diff = learned_array - truth_array
    abs_error = _safe_norm(diff)
    truth_norm = _safe_norm(truth_array)
    learned_norm = _safe_norm(learned_array)
    rel_error = abs_error if truth_norm < 1e-12 else abs_error / truth_norm
    signed_error = None
    if diff.shape == () or diff.size == 1:
        signed_error = float(diff.reshape(-1)[0])

    return ParameterRecoveryRecord(
        parameter_name=parameter_name,
        parameter_group=parameter_group,
        parameter_index=parameter_index,
        truth=_jsonable(truth),
        learned=_jsonable(learned),
        status="ok",
        abs_error=float(abs_error),
        rel_error=float(rel_error),
        signed_error=signed_error,
        log_abs_error=_log_abs_error(learned_array, truth_array),
        truth_norm=truth_norm,
        learned_norm=learned_norm,
        notes=notes,
        metadata=metadata_dict,
    )


def make_indexed_recovery_records(
    *,
    parameter_name: str,
    truth: Sequence[Any],
    learned: Sequence[Any],
    parameter_group: str | None = None,
    index_prefix: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> list[ParameterRecoveryRecord]:
    truth_array = np.asarray(truth, dtype=np.float64).reshape(-1)
    learned_array = np.asarray(learned, dtype=np.float64).reshape(-1)
    count = min(truth_array.size, learned_array.size)
    records = [
        make_recovery_record(
            parameter_name=parameter_name,
            parameter_group=parameter_group,
            parameter_index=f"{index_prefix}{idx}",
            truth=float(truth_array[idx]),
            learned=float(learned_array[idx]),
            metadata=metadata,
        )
        for idx in range(count)
    ]
    if truth_array.size != learned_array.size:
        records.append(
            make_recovery_record(
                parameter_name=parameter_name,
                parameter_group=parameter_group,
                parameter_index="shape",
                truth=truth_array.tolist(),
                learned=learned_array.tolist(),
                metadata=metadata,
            )
        )
    return records


def build_recovery_records(
    *,
    true_params: Mapping[str, Any],
    learned_params: Mapping[str, Any],
    parameter_names: Iterable[str] | None = None,
    parameter_group: str = "hyperparameter",
    metadata: Mapping[str, Any] | None = None,
) -> list[ParameterRecoveryRecord]:
    if parameter_names is None:
        parameter_names = sorted(set(true_params) | set(learned_params))
    records: list[ParameterRecoveryRecord] = []
    for name in parameter_names:
        if str(name).startswith("_"):
            continue
        records.append(
            make_recovery_record(
                parameter_name=str(name),
                parameter_group=parameter_group,
                truth=true_params.get(name),
                learned=learned_params.get(name),
                metadata=metadata,
            )
        )
    return records


def recovery_summary(records: Sequence[ParameterRecoveryRecord | Mapping[str, Any]]) -> dict[str, Any]:
    rows = [record.to_dict() if isinstance(record, ParameterRecoveryRecord) else dict(record) for record in records]
    status_counts: dict[str, int] = {}
    rel_errors = []
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "ok" and row.get("rel_error") is not None:
            rel_errors.append(float(row["rel_error"]))

    quality_flags: list[str] = []
    if not rows:
        quality_flags.append("no_recovery_records")
    for status, count in sorted(status_counts.items()):
        if status != "ok" and count:
            quality_flags.append(f"{status}:{count}")
    if rel_errors and max(rel_errors) > 1.0:
        quality_flags.append("large_relative_error")

    if not rows:
        quality_status = "missing"
    elif any(flag.startswith("nonfinite") for flag in quality_flags):
        quality_status = "nonfinite"
    elif any(flag.startswith("missing") or flag.startswith("shape_mismatch") for flag in quality_flags):
        quality_status = "incomplete"
    elif rel_errors and max(rel_errors) > 1.0:
        quality_status = "needs_review"
    else:
        quality_status = "observed"

    return {
        "record_count": len(rows),
        "status_counts": status_counts,
        "quality_status": quality_status,
        "quality_flags": quality_flags,
        "mean_rel_error": float(np.mean(rel_errors)) if rel_errors else None,
        "max_rel_error": float(np.max(rel_errors)) if rel_errors else None,
        "median_rel_error": float(np.median(rel_errors)) if rel_errors else None,
    }


def recovery_payload(records: Sequence[ParameterRecoveryRecord | Mapping[str, Any]]) -> dict[str, Any]:
    rows = [record.to_dict() if isinstance(record, ParameterRecoveryRecord) else dict(record) for record in records]
    return {"records": rows, "summary": recovery_summary(rows)}
