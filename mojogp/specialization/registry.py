"""Specialization registry and resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

from .builtins import LowDStationaryLeafRule, MaterializedPredictNcolsRule
from .decision import SpecializationDecision, SpecializationRequest
from .descriptor import WorkloadDescriptor
from .profile import DEFAULT_PROFILE, SpecializationProfile
from .rules import SpecializationRule


@dataclass
class SpecializationRegistry:
    rules: list[SpecializationRule] = field(default_factory=list)

    def resolve(
        self,
        descriptor: WorkloadDescriptor,
        request: SpecializationRequest | None = None,
    ) -> SpecializationDecision:
        request = request or SpecializationRequest.disabled()
        if request.mode == "disabled":
            return SpecializationDecision.disabled(descriptor, reason="specialization disabled")

        if request.profile is not None:
            return SpecializationDecision(
                mode=request.mode,
                descriptor=descriptor,
                profile=request.profile,
                applied=request.mode == "applied",
                reason="explicit profile request",
                default_equivalent=bool(request.profile.default_equivalent),
            )

        for rule in sorted(self.rules, key=lambda rule: rule.priority(), reverse=True):
            if not rule.matches(descriptor):
                continue
            profile = rule.build_profile(descriptor)
            return SpecializationDecision(
                mode=request.mode,
                descriptor=descriptor,
                profile=profile,
                applied=request.mode == "applied" and not profile.default_equivalent,
                reason=f"matched builtin rule {rule.__class__.__name__}",
                default_equivalent=bool(profile.default_equivalent),
            )

        return SpecializationDecision(
            mode=request.mode,
            descriptor=descriptor,
            profile=DEFAULT_PROFILE,
            applied=False,
            reason="fell back to default profile",
            default_equivalent=True,
        )


def default_specialization_registry(
    *,
    materialized_predict_ncols_hint: tuple[int, ...] = (11, 6, 1),
) -> SpecializationRegistry:
    return SpecializationRegistry(
        rules=[
            LowDStationaryLeafRule(),
            MaterializedPredictNcolsRule(ncols_hint=materialized_predict_ncols_hint),
        ]
    )
