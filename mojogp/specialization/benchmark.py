"""Benchmark-facing specialization helpers."""

from __future__ import annotations

from typing import Any

from .builtins import manual_benchmark_override_profile
from .profile import SpecializationProfile


def default_specialization_payload() -> dict[str, Any]:
    return {
        "specialization_mode": "disabled",
        "specialization_key": "default",
        "specialization_family": "jit_codegen",
        "specialization_source": "default",
        "policy_version": "v1",
        "specialization_descriptor": {},
        "specialization_config": {},
    }


def normalize_benchmark_specialization(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = default_specialization_payload()
    if payload is None:
        return normalized
    normalized.update(
        {
            "specialization_mode": str(payload.get("specialization_mode", normalized["specialization_mode"])),
            "specialization_key": str(payload.get("specialization_key", normalized["specialization_key"])),
            "specialization_family": str(payload.get("specialization_family", normalized["specialization_family"])),
            "specialization_source": str(payload.get("specialization_source", normalized["specialization_source"])),
            "policy_version": str(payload.get("policy_version", normalized["policy_version"])),
            "specialization_descriptor": dict(payload.get("specialization_descriptor", normalized["specialization_descriptor"])),
            "specialization_config": dict(payload.get("specialization_config", normalized["specialization_config"])),
        }
    )
    return normalized


def specialization_request_dict(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    payload = normalize_benchmark_specialization(payload)
    mode = str(payload.get("specialization_mode", "disabled"))
    if mode == "disabled":
        return {"mode": "disabled", "profile": None}

    config = dict(payload.get("specialization_config", {}))
    profile = manual_benchmark_override_profile(
        str(payload["specialization_key"]),
        schedule_overrides=(
            None if config.get("schedule_overrides") is None else dict(config["schedule_overrides"])
        ),
        ncols_hint=tuple(int(v) for v in config.get("ncols_hint", [])),
        module_suffix=(
            None
            if config.get("module_suffix") in (None, "")
            else str(config.get("module_suffix"))
        ),
        notes=None if payload.get("notes") is None else str(payload.get("notes")),
        policy_version=str(payload.get("policy_version", "v1")),
    )
    if payload.get("specialization_family") not in (None, "jit_codegen"):
        profile = SpecializationProfile(
            specialization_key=profile.specialization_key,
            family=str(payload.get("specialization_family", "jit_codegen")),
            source=profile.source,
            policy_version=profile.policy_version,
            schedule_overrides=profile.schedule_overrides,
            ncols_hint=profile.ncols_hint,
            module_suffix=profile.module_suffix,
            notes=profile.notes,
            default_equivalent=bool(payload.get("default_equivalent", False)),
        )
    return {"mode": mode, "profile": profile.to_dict()}


def apply_specialization_to_case_id(base_case_id: str, payload: dict[str, Any] | None) -> str:
    payload = normalize_benchmark_specialization(payload)
    mode = str(payload.get("specialization_mode", "disabled"))
    key = str(payload.get("specialization_key", "default"))
    if mode != "applied" or key == "default":
        return base_case_id
    return f"{base_case_id}.spec.{key}"


def merge_specialization_config(
    config: dict[str, Any],
    payload: dict[str, Any] | None,
    *,
    base_case_id: str,
) -> dict[str, Any]:
    merged = dict(config)
    specialization = normalize_benchmark_specialization(payload)
    merged.setdefault("base_case_id", base_case_id)
    merged.update(specialization)
    return merged


def extract_specialization_columns(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_case_id": config.get("base_case_id"),
        "specialization_key": config.get("specialization_key"),
        "specialization_family": config.get("specialization_family"),
        "specialization_mode": config.get("specialization_mode"),
        "specialization_source": config.get("specialization_source"),
        "specialization_descriptor_json": config.get("specialization_descriptor", {}),
        "specialization_config_json": config.get("specialization_config", {}),
    }
