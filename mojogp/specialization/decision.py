"""Specialization request and resolution results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .descriptor import WorkloadDescriptor
from .profile import DEFAULT_PROFILE, SpecializationProfile


SPECIALIZATION_MODES = {"disabled", "shadow", "applied"}


@dataclass(frozen=True)
class SpecializationRequest:
    """User- or benchmark-supplied specialization request."""

    mode: str = "disabled"
    profile: SpecializationProfile | None = None

    def __post_init__(self):
        if self.mode not in SPECIALIZATION_MODES:
            raise ValueError(
                f"Unknown specialization mode '{self.mode}'. "
                f"Must be one of {sorted(SPECIALIZATION_MODES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": None if self.profile is None else self.profile.to_dict(),
        }

    @classmethod
    def disabled(cls) -> "SpecializationRequest":
        return cls(mode="disabled", profile=None)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SpecializationRequest":
        if payload is None:
            return cls.disabled()
        return cls(
            mode=str(payload.get("mode", "disabled")),
            profile=SpecializationProfile.from_dict(payload.get("profile")),
        )


@dataclass(frozen=True)
class SpecializationDecision:
    """Resolved specialization decision for one workload."""

    mode: str
    descriptor: WorkloadDescriptor
    profile: SpecializationProfile
    applied: bool
    reason: str
    default_equivalent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "descriptor": self.descriptor.to_dict(),
            "profile": self.profile.to_dict(),
            "applied": self.applied,
            "reason": self.reason,
            "default_equivalent": self.default_equivalent,
        }

    @classmethod
    def disabled(cls, descriptor: WorkloadDescriptor, *, reason: str) -> "SpecializationDecision":
        return cls(
            mode="disabled",
            descriptor=descriptor,
            profile=DEFAULT_PROFILE,
            applied=False,
            reason=reason,
            default_equivalent=True,
        )
