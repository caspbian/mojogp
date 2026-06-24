"""Translate specialization decisions into compile/load inputs."""

from __future__ import annotations

from dataclasses import dataclass

from mojogp.codegen_engine.schedule import ScheduleConfig

from .decision import SpecializationDecision


@dataclass(frozen=True)
class CompileTranslation:
    schedule_overrides: ScheduleConfig | None
    ncols_hint: list[int] | None
    module_suffix: str | None


def _sanitize_module_suffix(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def translate_compile_inputs(decision: SpecializationDecision | None) -> CompileTranslation:
    if decision is None or not decision.applied:
        return CompileTranslation(None, None, None)

    profile = decision.profile
    schedule_overrides = (
        None
        if profile.schedule_overrides is None
        else ScheduleConfig.from_dict(profile.schedule_overrides)
    )
    ncols_hint = list(profile.ncols_hint) if profile.ncols_hint else None
    module_suffix = _sanitize_module_suffix(
        profile.module_suffix or profile.specialization_key
    )
    return CompileTranslation(schedule_overrides, ncols_hint, module_suffix)
