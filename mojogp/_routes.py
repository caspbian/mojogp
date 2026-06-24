"""Internal route-name normalization helpers."""

from __future__ import annotations


_FIT_METHOD_ALIASES = {
    "mat": "materialized",
    "mf": "matrix_free",
}


def normalize_fit_method(method: str, *, allow_auto: bool = False) -> str:
    """Return the canonical training method name for public fit() calls."""
    if not isinstance(method, str):
        raise ValueError("method must be a string")

    if method in _FIT_METHOD_ALIASES:
        return _FIT_METHOD_ALIASES[method]

    valid_methods = ["materialized", "matrix_free"]
    if allow_auto:
        valid_methods.append("auto")
    if method in valid_methods:
        return method

    aliases = ", ".join(f"'{alias}'" for alias in sorted(_FIT_METHOD_ALIASES))
    valid = ", ".join(f"'{value}'" for value in valid_methods)
    raise ValueError(f"method must be one of {valid}; accepted aliases: {aliases}")
