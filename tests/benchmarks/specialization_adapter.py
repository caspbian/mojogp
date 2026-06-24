"""Benchmark-only specialization adapter helpers."""

from __future__ import annotations

from typing import Any

from mojogp.specialization import normalize_benchmark_specialization, specialization_request_dict

from .preflight import utc_now_iso
from .session_store import BenchmarkSessionStore


def prepare_specialization_payload(
    session_store: BenchmarkSessionStore | None,
    payload: dict[str, Any] | None,
    *,
    created_at: str | None = None,
    policy_version: str = "v1",
) -> dict[str, Any]:
    specialization = normalize_benchmark_specialization(payload)
    if (
        session_store is not None
        and specialization["specialization_key"] != "default"
    ):
        session_store.register_specialization(
            specialization_key=str(specialization["specialization_key"]),
            specialization_family=str(specialization["specialization_family"]),
            specialization_source=str(specialization["specialization_source"]),
            policy_version=str(specialization.get("policy_version", policy_version)),
            config=dict(specialization["specialization_config"]),
            notes=None,
            created_at=created_at or utc_now_iso(),
            active=True,
        )
    return specialization


def apply_specialization_to_model(model: Any, payload: dict[str, Any] | None) -> None:
    request = specialization_request_dict(payload)
    if request is None:
        return
    setter = getattr(model, "_set_specialization_request", None)
    if setter is None:
        raise AttributeError(
            f"Model type {type(model).__name__} does not support benchmark specialization injection"
        )
    setter(request)
