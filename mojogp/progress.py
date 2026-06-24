"""Progress reporting helpers for MojoGP public wrappers."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol


DEFAULT_PROGRESS_STATS = (
    "nll",
    "cg_iter",
    "iter_ms",
    "noise",
    "precond_rank",
    "n_test",
    "variance_method",
    "prediction_total_time_s",
    "exact_cg_block_count",
)

_PROGRESS_STAT_LABELS = {
    "cg_iter": "cg",
    "iter_ms": "iter",
    "precond_rank": "rank",
    "n_test": "test",
    "variance_method": "var",
    "prediction_total_time_s": "time",
    "exact_cg_block_count": "blocks",
}

_PROGRESS_STAT_SUFFIXES = {
    "iter_ms": "ms",
    "prediction_total_time_s": "s",
    "prediction_alpha_time_s": "s",
    "prediction_mean_time_s": "s",
    "prediction_variance_time_s": "s",
}


def prediction_progress_stats(
    *,
    n_test: int,
    variance_method: str,
    backend_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build progress stats from already-available prediction metadata."""

    stats: dict[str, Any] = {
        "n_test": int(n_test),
        "variance_method": variance_method,
    }
    if backend_info is None:
        return stats
    for key in (
        "prediction_total_time_s",
        "prediction_alpha_time_s",
        "prediction_mean_time_s",
        "prediction_variance_time_s",
        "prediction_cache_used",
        "exact_block_cols",
        "exact_cg_block_count",
        "exact_cg_total_iterations",
    ):
        if key in backend_info:
            stats[key] = backend_info[key]
    return stats


@dataclass(frozen=True)
class ProgressEvent:
    """Structured progress event emitted by training and prediction routes."""

    operation: str
    model: str
    route: Optional[str]
    phase: str
    current: int
    total: Optional[int] = None
    elapsed_s: Optional[float] = None
    message: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        operation: str,
        model: str,
        route: Optional[str],
        started_at: float,
    ) -> "ProgressEvent":
        if isinstance(payload, ProgressEvent):
            return payload
        if not isinstance(payload, dict):
            payload = {"phase": "update", "message": str(payload)}
        elapsed = payload.get("elapsed_s")
        if elapsed is None:
            elapsed = time.perf_counter() - started_at
        total = payload.get("total")
        if total is not None:
            total = int(total)
        stats = payload.get("stats") or {}
        if not isinstance(stats, dict):
            stats = dict(stats)
        if "converged" in payload and "converged" not in stats:
            stats["converged"] = bool(payload["converged"])
        payload_route = payload.get("route", route)
        if payload_route in (None, "", "unknown"):
            payload_route = route
        return cls(
            operation=str(payload.get("operation", operation)),
            model=str(payload.get("model", model)),
            route=payload_route,
            phase=str(payload.get("phase", "update")),
            current=int(payload.get("current", 0) or 0),
            total=total,
            elapsed_s=float(elapsed),
            message=payload.get("message"),
            stats=stats,
        )


class ProgressReporter(Protocol):
    """Reporter protocol used by wrapper-level progress adapters."""

    def start(self, event: ProgressEvent) -> None: ...

    def update(self, event: ProgressEvent) -> None: ...

    def close(self, event: ProgressEvent) -> None: ...


class NoOpProgressReporter:
    """Reporter that deliberately does nothing."""

    def start(self, event: ProgressEvent) -> None:
        return None

    def update(self, event: ProgressEvent) -> None:
        return None

    def close(self, event: ProgressEvent) -> None:
        return None


class CallbackProgressReporter:
    """Forward every structured event to a user callback."""

    def __init__(self, callback: Callable[[ProgressEvent], Any]):
        self._callback = callback

    def start(self, event: ProgressEvent) -> None:
        self._callback(event)

    def update(self, event: ProgressEvent) -> None:
        self._callback(event)

    def close(self, event: ProgressEvent) -> None:
        self._callback(event)


class TqdmProgressReporter:
    """Default tqdm-backed reporter with an import-time fallback."""

    def __init__(
        self,
        *,
        desc: str,
        stats: Optional[Iterable[str]] | Callable[[ProgressEvent], dict[str, Any]] = None,
        leave: Optional[bool] = None,
    ):
        self._desc = desc
        self._stats = stats
        self._leave = True if leave is None else bool(leave)
        self._bar = None
        self._last_current = 0
        self._fallback_started = False

    def _selected_stats(self, event: ProgressEvent) -> dict[str, Any]:
        if callable(self._stats):
            selected = dict(self._stats(event))
            return {key: self._format_stat_value(value) for key, value in selected.items()}
        keys = DEFAULT_PROGRESS_STATS if self._stats is None else tuple(self._stats)
        return {
            key: self._format_stat_value(event.stats[key])
            for key in keys
            if key in event.stats
        }

    @staticmethod
    def _format_stat_value(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return f"{value:.2f}"
        return value

    def _postfix_text(self, event: ProgressEvent) -> str:
        selected = self._selected_stats(event)
        if not selected:
            return ""
        parts = []
        for key, value in selected.items():
            label = _PROGRESS_STAT_LABELS.get(key, key)
            suffix = _PROGRESS_STAT_SUFFIXES.get(key, "")
            parts.append(f"{label}={value}{suffix}")
        return " | ".join(parts)

    def _ensure_bar(self, event: ProgressEvent) -> None:
        if self._bar is not None or self._fallback_started:
            return
        tqdm = None
        module_names = ("tqdm.auto", "tqdm.notebook", "tqdm")
        try:
            import marimo as mo

            if mo.running_in_notebook():
                module_names = ("tqdm.notebook", "tqdm.auto", "tqdm")
        except Exception:
            pass
        for module_name in module_names:
            try:
                module = __import__(module_name, fromlist=["tqdm"])
                tqdm = getattr(module, "tqdm")
                break
            except Exception:
                continue
        if tqdm is None:
            self._fallback_started = True
            total = "?" if event.total is None else str(event.total)
            print(f"{self._desc}: {event.current}/{total}", file=sys.stderr)
            return
        unit = "iter" if self._desc.lower() == "train" else "phase"
        self._bar = tqdm(
            total=event.total,
            desc=self._desc,
            unit=unit,
            dynamic_ncols=False,
            mininterval=0.0,
            miniters=1,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar:20}| "
                "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
            leave=self._leave,
        )

    def start(self, event: ProgressEvent) -> None:
        self._ensure_bar(event)
        self._last_current = max(0, int(event.current))

    def update(self, event: ProgressEvent) -> None:
        self._ensure_bar(event)
        if self._bar is None:
            total = "?" if event.total is None else str(event.total)
            msg = f"{self._desc}: {event.current}/{total}"
            postfix = self._postfix_text(event)
            if postfix:
                msg += " " + postfix
            print(msg, file=sys.stderr)
            self._last_current = max(self._last_current, int(event.current))
            return
        current = int(event.current)
        delta = current - self._last_current
        if delta > 0:
            self._bar.update(delta)
            self._last_current = current
        postfix = self._postfix_text(event)
        if postfix:
            if hasattr(self._bar, "set_postfix_str"):
                self._bar.set_postfix_str(postfix, refresh=True)
            elif self._try_update_subtitle(postfix):
                return
            elif hasattr(self._bar, "set_description_str"):
                self._bar.set_description_str(f"{self._desc} {postfix}", refresh=True)

    def _try_update_subtitle(self, subtitle: str) -> bool:
        if self._bar is None:
            return False

        progress = getattr(self._bar, "progress", None)
        if progress is not None and hasattr(progress, "update"):
            try:
                progress.update(increment=0, title=self._desc, subtitle=subtitle)
                return True
            except TypeError:
                try:
                    progress.update(increment=0, subtitle=subtitle)
                    return True
                except TypeError:
                    pass

        if not hasattr(self._bar, "update"):
            return False
        for kwargs in (
            {"increment": 0, "title": self._desc, "subtitle": subtitle},
            {"increment": 0, "subtitle": subtitle},
        ):
            try:
                self._bar.update(**kwargs)
                return True
            except TypeError:
                continue
        return False

    def close(self, event: ProgressEvent) -> None:
        if event.current > self._last_current:
            self.update(event)
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class EngineProgressAdapter:
    """Callable object passed into Mojo engine bindings as a Python callback."""

    def __init__(
        self,
        reporter: ProgressReporter,
        *,
        operation: str,
        model: str,
        route: Optional[str],
    ):
        self._reporter = reporter
        self._operation = operation
        self._model = model
        self._route = route
        self._started_at = time.perf_counter()
        self._started = False
        self._closed = False

    @property
    def callback(self) -> "EngineProgressAdapter":
        return self

    def __call__(self, payload: Any) -> None:
        event = ProgressEvent.from_payload(
            payload,
            operation=self._operation,
            model=self._model,
            route=self._route,
            started_at=self._started_at,
        )
        if event.phase == "start":
            self._started = True
            self._reporter.start(event)
        elif event.phase in ("complete", "failed", "finalize"):
            self._closed = True
            self._reporter.close(event)
        else:
            if not self._started:
                self._started = True
                start_event = ProgressEvent(
                    operation=event.operation,
                    model=event.model,
                    route=event.route,
                    phase="start",
                    current=0,
                    total=event.total,
                    elapsed_s=event.elapsed_s,
                    stats={},
                )
                self._reporter.start(start_event)
            self._reporter.update(event)

    def emit(
        self,
        *,
        phase: str,
        current: int = 0,
        total: Optional[int] = None,
        message: Optional[str] = None,
        stats: Optional[dict[str, Any]] = None,
    ) -> None:
        self(
            {
                "operation": self._operation,
                "model": self._model,
                "route": self._route,
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "stats": stats or {},
            }
        )

    def close_if_needed(self, *, failed: bool = False, message: Optional[str] = None) -> None:
        if self._closed:
            return
        self.emit(phase="failed" if failed else "complete", message=message)


def progress_requested(progress: Any) -> bool:
    """Cheap disabled-path check that avoids reporter construction."""

    if progress is False:
        return False
    if progress is not None:
        return True
    try:
        from mojogp.settings import get_progress_enabled

        return bool(get_progress_enabled())
    except Exception:
        return False


def resolve_progress_adapter(
    progress: Any,
    *,
    operation: str,
    model: str,
    route: Optional[str],
    progress_stats: Optional[Iterable[str]] | Callable[[ProgressEvent], dict[str, Any]] = None,
) -> Optional[EngineProgressAdapter]:
    """Return an engine callback adapter, or None for the strict no-op path."""

    if not progress_requested(progress):
        return None
    if progress is None:
        from mojogp.settings import get_progress_enabled

        progress = get_progress_enabled()
    if progress == "auto" and not sys.stderr.isatty():
        return None
    if progress is True or progress == "auto":
        reporter: ProgressReporter = TqdmProgressReporter(
            desc=operation.title(), stats=progress_stats
        )
    elif callable(progress):
        reporter = CallbackProgressReporter(progress)
    elif all(hasattr(progress, name) for name in ("start", "update", "close")):
        reporter = progress
    else:
        raise ValueError(
            "progress must be None, bool, 'auto', a callback, or a ProgressReporter"
        )
    return EngineProgressAdapter(
        reporter,
        operation=operation,
        model=model,
        route=route,
    )
