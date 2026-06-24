"""Runtime settings for MojoGP's Python API."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from mojogp.feature_support import (
    feature_warnings_suppressed as _feature_warnings_suppressed,
    get_feature_warnings_enabled as _get_feature_warnings_enabled,
    set_feature_warnings_enabled as _set_feature_warnings_enabled,
)

_PROGRESS_ENABLED: bool | str = True


def get_feature_warnings_enabled() -> bool:
    """Return whether MojoGP feature-maturity warnings are enabled."""

    return _get_feature_warnings_enabled()


def set_feature_warnings_enabled(enabled: bool) -> None:
    """Enable or disable MojoGP feature-maturity warnings globally."""

    _set_feature_warnings_enabled(enabled)


def get_progress_enabled() -> bool | str:
    """Return the global default for MojoGP progress reporting."""

    return _PROGRESS_ENABLED


def set_progress_enabled(enabled: bool | str) -> None:
    """Set the global default for progress reporting.

    Valid values are ``False``, ``True``, or ``"auto"``.
    """

    if enabled not in (False, True, "auto"):
        raise ValueError("progress setting must be False, True, or 'auto'")
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = enabled


@contextmanager
def feature_warnings_suppressed() -> Iterator[None]:
    """Temporarily suppress MojoGP feature-maturity warnings."""

    with _feature_warnings_suppressed():
        yield


@contextmanager
def progress_enabled(enabled: bool | str) -> Iterator[None]:
    """Temporarily set the global progress-reporting default."""

    old = get_progress_enabled()
    set_progress_enabled(enabled)
    try:
        yield
    finally:
        set_progress_enabled(old)


__all__ = [
    "feature_warnings_suppressed",
    "get_feature_warnings_enabled",
    "get_progress_enabled",
    "progress_enabled",
    "set_feature_warnings_enabled",
    "set_progress_enabled",
]
