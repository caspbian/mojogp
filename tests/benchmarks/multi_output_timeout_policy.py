"""Timeout policy for multi-output benchmark subprocesses."""

from __future__ import annotations

import os


DEFAULT_MULTI_OUTPUT_SCALING_TIMEOUT_S = 3600
DEFAULT_MULTI_OUTPUT_MATRIX_FREE_EXACT_TIMEOUT_S = 7200
MULTI_OUTPUT_SCALING_TIMEOUT_ENV = "MOJOGP_MULTI_OUTPUT_SCALING_TIMEOUT_S"


def _env_suffix(*parts: object) -> str:
    return "_".join(str(part).replace("-", "_").upper() for part in parts if part is not None)


def _positive_int(value: object, *, source: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive integer number of seconds") from exc
    if parsed <= 0:
        raise ValueError(f"{source} must be a positive integer number of seconds")
    return parsed


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return _positive_int(raw, source=name)


def multi_output_scaling_timeout_s(
    *,
    framework: str,
    method: str,
    prediction_mode: str,
    tier: str | None,
    timeout_s: int | None = None,
) -> int:
    """Return the subprocess timeout for a multi-output scaling benchmark lane.

    Override priority is explicit argument, framework/method/prediction/tier env,
    framework/method/prediction env, method/prediction/tier env,
    method/prediction env, global env, then lane default.
    """

    if timeout_s is not None:
        return _positive_int(timeout_s, source="timeout_s")

    env_names = [
        f"{MULTI_OUTPUT_SCALING_TIMEOUT_ENV}_{_env_suffix(framework, method, prediction_mode, tier)}",
        f"{MULTI_OUTPUT_SCALING_TIMEOUT_ENV}_{_env_suffix(framework, method, prediction_mode)}",
        f"{MULTI_OUTPUT_SCALING_TIMEOUT_ENV}_{_env_suffix(method, prediction_mode, tier)}",
        f"{MULTI_OUTPUT_SCALING_TIMEOUT_ENV}_{_env_suffix(method, prediction_mode)}",
        MULTI_OUTPUT_SCALING_TIMEOUT_ENV,
    ]
    for env_name in env_names:
        value = _positive_int_env(env_name)
        if value is not None:
            return value

    if method == "matrix_free" and prediction_mode == "exact":
        return DEFAULT_MULTI_OUTPUT_MATRIX_FREE_EXACT_TIMEOUT_S
    return DEFAULT_MULTI_OUTPUT_SCALING_TIMEOUT_S
