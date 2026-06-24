"""Specialization profile definitions.

These profiles describe additive compile/load adjustments for a narrow workload.
They do not change kernel math or engine behavior directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SpecializationProfile:
    """Named compile/load policy for a narrow workload lane."""

    specialization_key: str
    family: str = "jit_codegen"
    source: str = "default"
    policy_version: str = "v1"
    schedule_overrides: dict[str, Any] | None = None
    ncols_hint: tuple[int, ...] = field(default_factory=tuple)
    module_suffix: str | None = None
    notes: str | None = None
    default_equivalent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialization_key": self.specialization_key,
            "family": self.family,
            "source": self.source,
            "policy_version": self.policy_version,
            "schedule_overrides": None
            if self.schedule_overrides is None
            else dict(self.schedule_overrides),
            "ncols_hint": list(self.ncols_hint),
            "module_suffix": self.module_suffix,
            "notes": self.notes,
            "default_equivalent": self.default_equivalent,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SpecializationProfile | None":
        if payload is None:
            return None
        return cls(
            specialization_key=str(payload["specialization_key"]),
            family=str(payload.get("family", "jit_codegen")),
            source=str(payload.get("source", "default")),
            policy_version=str(payload.get("policy_version", "v1")),
            schedule_overrides=(
                None
                if payload.get("schedule_overrides") is None
                else dict(payload["schedule_overrides"])
            ),
            ncols_hint=tuple(int(v) for v in payload.get("ncols_hint", ())),
            module_suffix=(
                None
                if payload.get("module_suffix") in (None, "")
                else str(payload.get("module_suffix"))
            ),
            notes=(None if payload.get("notes") is None else str(payload.get("notes"))),
            default_equivalent=bool(payload.get("default_equivalent", False)),
        )


DEFAULT_PROFILE = SpecializationProfile(
    specialization_key="default",
    source="default",
    policy_version="v1",
    notes="Default compile/load path.",
    default_equivalent=True,
)
