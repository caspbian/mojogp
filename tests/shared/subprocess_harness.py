"""Shared subprocess harness for isolated test and benchmark workloads.

Contract:
- parent side uses `run_isolated_module()` or `run_isolated_case()`
- payload is written as JSON to a harness-managed temp file
- child loads the payload with `load_child_payload()`
- child exits via `run_child_main()` which emits one final JSON envelope
- the envelope always contains `status`, `payload`, `result_path`, `telemetry`, and `error`

The harness keeps subprocess orchestration, cleanup, telemetry capture, and
failure formatting in one place so benchmark and integration call sites do not
need to reimplement them.
"""

from __future__ import annotations

import gc
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from tests.shared.benchmarking.gpu_memory import (
    GPUMemoryMonitor,
    GPUMemoryTracker,
    get_torch_memory_stats,
    reset_torch_memory_stats,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_WORKSPACE_ROOT = PROJECT_ROOT.parent.parent if PROJECT_ROOT.parent.name == ".worktrees" else PROJECT_ROOT
ENV_HARNESS_ACTIVE = "MOJOGP_SUBPROCESS_HARNESS"
ENV_HARNESS_ROLE = "MOJOGP_SUBPROCESS_ROLE"
ENV_PAYLOAD_PATH = "MOJOGP_SUBPROCESS_PAYLOAD_PATH"
ENV_RESULT_PATH = "MOJOGP_SUBPROCESS_RESULT_PATH"
ENV_STREAM_LOGS = "MOJOGP_SUBPROCESS_STREAM_LOGS"


class HarnessJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common test payload types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if np is not None:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        return super().default(obj)


@dataclass
class IsolatedSubprocessError(RuntimeError):
    """Uniform error raised when an isolated child fails."""

    message: str
    command: list[str]
    cwd: str
    timeout: int
    returncode: int | None
    stdout: str
    stderr: str
    envelope: dict[str, Any] | None = None

    def __str__(self) -> str:
        lines = [self.message]
        lines.append(f"command: {' '.join(self.command)}")
        lines.append(f"cwd: {self.cwd}")
        lines.append(f"timeout_s: {self.timeout}")
        if self.returncode is not None:
            lines.append(f"returncode: {self.returncode}")
        if self.envelope is not None:
            status = self.envelope.get("status")
            if status is not None:
                lines.append(f"child_status: {status}")
            error = self.envelope.get("error")
            if isinstance(error, dict):
                err_type = error.get("type")
                err_message = error.get("message")
                if err_type or err_message:
                    lines.append(f"child_error: {err_type}: {err_message}")
        lines.append("stdout_tail:")
        lines.append(_tail_text(self.stdout))
        lines.append("stderr_tail:")
        lines.append(_tail_text(self.stderr))
        return "\n".join(lines)


def _tail_text(text: str, *, max_lines: int = 80, max_chars: int = 8000) -> str:
    if not text:
        return "<empty>"
    lines = text.splitlines()[-max_lines:]
    tail = "\n".join(lines)
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _prepend_pythonpath(existing: str | None, entry: str) -> str:
    if not existing:
        return entry
    parts = [part for part in existing.split(os.pathsep) if part]
    if entry in parts:
        return existing
    return os.pathsep.join([entry, *parts])


def _fallback_engine_path() -> str | None:
    worktree_engine = PROJECT_ROOT / "mojogp_jit_engine.so"
    if worktree_engine.exists():
        return None
    candidate = MAIN_WORKSPACE_ROOT / "mojogp_jit_engine.so"
    if candidate.exists():
        return str(candidate)
    return None


def _cleanup_gpu_state() -> None:
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:  # pragma: no cover
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover
        pass


class IsolatedGPUTestSession:
    """Centralized child-side cleanup and telemetry session."""

    def __init__(self, *, monitor_interval: float = 0.1):
        self._monitor_interval = monitor_interval
        self._monitor: GPUMemoryMonitor | None = None
        self._tracker: GPUMemoryTracker | None = None
        self._process = psutil.Process() if psutil is not None else None
        self._started = False
        self._finalized = False
        self._start_time = 0.0
        self._memory_stats: dict[str, float] | None = None
        self._cpu_peak_mb = 0.0
        self._baseline_gpu_mb = 0.0
        self._torch_baseline_mb = 0.0
        self._torch_reserved_baseline_mb = 0.0

    def __enter__(self) -> "IsolatedGPUTestSession":
        _cleanup_gpu_state()
        reset_torch_memory_stats()
        self._tracker = GPUMemoryTracker()
        self._tracker.reset()
        baseline_snapshot = self._tracker.snapshot()
        self._baseline_gpu_mb = float(baseline_snapshot.get("baseline_gpu_mb", 0.0))
        self._torch_baseline_mb = float(baseline_snapshot.get("torch_current_mb", 0.0))
        self._torch_reserved_baseline_mb = float(
            get_torch_memory_stats().get("torch_reserved_mb", 0.0)
        )
        self._monitor = GPUMemoryMonitor(interval=self._monitor_interval)
        self._monitor.start()
        tracemalloc.start()
        self._start_time = time.perf_counter()
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._finalized:
            self.collect_memory_stats()

    def snapshot_gpu(self) -> dict[str, float]:
        if self._tracker is None:
            return {
                "peak_gpu_mb": 0.0,
                "current_gpu_mb": 0.0,
                "torch_peak_mb": 0.0,
                "torch_current_mb": 0.0,
                "method": "none",
            }
        return self._tracker.snapshot()

    def collect_memory_stats(
        self, *, snapshots: list[dict[str, float]] | None = None
    ) -> dict[str, float]:
        if self._memory_stats is not None:
            return dict(self._memory_stats)

        monitor_stats = (
            self._monitor.get_stats() if self._monitor is not None else {
                "mean_mb": 0.0,
                "min_mb": 0.0,
                "max_mb": 0.0,
                "var_mb": 0.0,
                "samples": 0,
                "method": "none",
            }
        )
        if self._monitor is not None:
            self._monitor.stop()

        current = 0
        peak = 0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        del current

        torch_stats = get_torch_memory_stats()
        final_snapshot = self.snapshot_gpu()
        memory_stats = {
            **dict(monitor_stats),
            **torch_stats,
            "cpu_peak_mb": peak / (1024 * 1024),
        }
        if self._process is not None:
            memory_stats["cpu_peak_mb"] = max(
                float(memory_stats["cpu_peak_mb"]),
                float(self._process.memory_info().rss / (1024 * 1024)),
            )

        snapshot_list = list(snapshots or [])
        if snapshot_list:
            memory_stats["max_mb"] = max(
                float(memory_stats.get("max_mb", 0.0)),
                max(float(s.get("peak_gpu_mb", 0.0)) for s in snapshot_list),
            )
            memory_stats["min_mb"] = min(
                float(memory_stats.get("min_mb", 0.0)),
                min(float(s.get("current_gpu_mb", 0.0)) for s in snapshot_list),
            )
            memory_stats["mean_mb"] = max(
                float(memory_stats.get("mean_mb", 0.0)),
                float(memory_stats.get("max_mb", 0.0)),
            )

        global_peak_mb = max(
            float(memory_stats.get("max_mb", 0.0)),
            float(final_snapshot.get("peak_gpu_mb", 0.0)),
        )
        global_current_mb = float(final_snapshot.get("current_gpu_mb", 0.0))
        global_peak_mb = max(global_peak_mb, global_current_mb)
        memory_stats["max_mb"] = global_peak_mb
        global_delta_mb = max(
            global_peak_mb - self._baseline_gpu_mb,
            float(final_snapshot.get("delta_gpu_mb", 0.0)),
        )

        torch_peak_mb = float(memory_stats.get("torch_peak_mb", 0.0))
        torch_current_mb = float(memory_stats.get("torch_current_mb", 0.0))
        torch_reserved_mb = float(memory_stats.get("torch_reserved_mb", 0.0))
        torch_peak_delta_mb = max(torch_peak_mb - self._torch_baseline_mb, 0.0)
        torch_current_delta_mb = max(torch_current_mb - self._torch_baseline_mb, 0.0)
        torch_reserved_delta_mb = max(
            torch_reserved_mb - self._torch_reserved_baseline_mb,
            0.0,
        )

        isolated_peak_mb = max(
            global_peak_mb - self._baseline_gpu_mb,
            torch_peak_delta_mb,
            0.0,
        )
        isolated_current_mb = max(
            global_current_mb - self._baseline_gpu_mb,
            torch_current_delta_mb,
            0.0,
        )

        memory_stats.update(
            {
                "baseline_gpu_mb": self._baseline_gpu_mb,
                "current_gpu_mb": global_current_mb,
                "delta_gpu_mb": global_delta_mb,
                "isolated_peak_gpu_mb": isolated_peak_mb,
                "isolated_current_gpu_mb": isolated_current_mb,
                "torch_baseline_mb": self._torch_baseline_mb,
                "torch_peak_delta_mb": torch_peak_delta_mb,
                "torch_current_delta_mb": torch_current_delta_mb,
                "torch_reserved_baseline_mb": self._torch_reserved_baseline_mb,
                "torch_reserved_delta_mb": torch_reserved_delta_mb,
            }
        )

        self._cpu_peak_mb = float(memory_stats.get("cpu_peak_mb", 0.0))
        self._memory_stats = memory_stats
        self._finalized = True
        _cleanup_gpu_state()
        return dict(memory_stats)

    def collect_telemetry(
        self,
        *,
        snapshots: list[dict[str, float]] | None = None,
        timing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        memory_stats = self.collect_memory_stats(snapshots=snapshots)
        gpu_memory = {
            "mean_mb": float(memory_stats.get("mean_mb", 0.0)),
            "min_mb": float(memory_stats.get("min_mb", 0.0)),
            "max_mb": float(memory_stats.get("max_mb", 0.0)),
            "baseline_mb": float(memory_stats.get("baseline_gpu_mb", 0.0)),
            "current_mb": float(memory_stats.get("current_gpu_mb", 0.0)),
            "delta_mb": float(memory_stats.get("delta_gpu_mb", 0.0)),
            "isolated_peak_mb": float(
                memory_stats.get("isolated_peak_gpu_mb", 0.0)
            ),
            "isolated_current_mb": float(
                memory_stats.get("isolated_current_gpu_mb", 0.0)
            ),
            "var_mb": float(memory_stats.get("var_mb", 0.0)),
            "samples": int(memory_stats.get("samples", 0)),
            "method": str(memory_stats.get("method", "none")),
        }
        torch_memory = {
            "peak_mb": float(memory_stats.get("torch_peak_mb", 0.0)),
            "current_mb": float(memory_stats.get("torch_current_mb", 0.0)),
            "reserved_mb": float(memory_stats.get("torch_reserved_mb", 0.0)),
            "baseline_mb": float(memory_stats.get("torch_baseline_mb", 0.0)),
            "peak_delta_mb": float(memory_stats.get("torch_peak_delta_mb", 0.0)),
            "current_delta_mb": float(
                memory_stats.get("torch_current_delta_mb", 0.0)
            ),
            "reserved_baseline_mb": float(
                memory_stats.get("torch_reserved_baseline_mb", 0.0)
            ),
            "reserved_delta_mb": float(
                memory_stats.get("torch_reserved_delta_mb", 0.0)
            ),
        }
        cpu_memory = {"peak_mb": float(self._cpu_peak_mb)}
        return {
            "timing": {
                "wall_time_s": float(time.perf_counter() - self._start_time)
                if self._started
                else 0.0,
                **dict(timing or {}),
            },
            "memory": {
                "gpu": gpu_memory,
                "torch": torch_memory,
                "cpu": cpu_memory,
                "flat": memory_stats,
            },
        }


def isolated_gpu_test_session(
    *, monitor_interval: float = 0.1
) -> IsolatedGPUTestSession:
    return IsolatedGPUTestSession(monitor_interval=monitor_interval)


def is_isolated_subprocess_child() -> bool:
    return os.environ.get(ENV_HARNESS_ACTIVE) == "1" and os.environ.get(
        ENV_HARNESS_ROLE
    ) == "child"


def load_child_payload() -> dict[str, Any]:
    payload_path = os.environ.get(ENV_PAYLOAD_PATH)
    if not payload_path:
        raise RuntimeError(f"Missing required env var {ENV_PAYLOAD_PATH}")
    with Path(payload_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Child payload must deserialize to a dict")
    return payload


def emit_child_result(envelope: Mapping[str, Any]) -> None:
    result_path = os.environ.get(ENV_RESULT_PATH)
    serialized = json.dumps(dict(envelope), cls=HarnessJSONEncoder)
    if result_path:
        Path(result_path).write_text(serialized, encoding="utf-8")
    print(serialized)


def _parse_child_output(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "status" in parsed:
            return parsed
    return None


def _read_result_envelope(result_path: Path, stdout: str) -> dict[str, Any] | None:
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    return _parse_child_output(stdout)


def _stream_child_output_enabled() -> bool:
    return os.environ.get(ENV_STREAM_LOGS, "0") not in {"", "0", "false", "False"}


def _run_child_command(
    *, command: list[str], cwd: Path, env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    if not _stream_child_output_enabled():
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            env=env,
        )

    print(
        f"=== MOJOGP ISOLATED CHILD START {' '.join(command)} ===",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    output_chunks: list[str] = []
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    deadline = time.monotonic() + float(timeout)
    timed_out = False

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0 and process.poll() is None:
            timed_out = True
            process.kill()
            break
        try:
            item = output_queue.get(timeout=max(0.0, min(0.25, remaining)))
        except queue.Empty:
            if process.poll() is not None and not reader.is_alive():
                break
            continue
        if item is None:
            break
        output_chunks.append(item)
        sys.stdout.write(item)
        sys.stdout.flush()

    if timed_out:
        try:
            remaining_output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            remaining_output, _ = process.communicate()
        if remaining_output:
            output_chunks.append(remaining_output)
            sys.stdout.write(remaining_output)
            sys.stdout.flush()
        stdout = "".join(output_chunks)
        raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr="")

    returncode = process.wait()
    reader.join(timeout=1)
    print(
        f"=== MOJOGP ISOLATED CHILD END returncode={returncode} {' '.join(command)} ===",
        flush=True,
    )
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout="".join(output_chunks),
        stderr="",
    )


def run_isolated_module(
    *,
    module: str,
    payload: Mapping[str, Any],
    timeout: int,
    cwd: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Run a child module under the standard harness and return its envelope."""

    cwd = cwd or PROJECT_ROOT
    with tempfile.TemporaryDirectory(prefix="mojogp_subprocess_harness_") as temp_dir:
        temp_path = Path(temp_dir)
        payload_path = temp_path / "payload.json"
        result_path = temp_path / "result.json"
        payload_path.write_text(
            json.dumps(dict(payload), cls=HarnessJSONEncoder), encoding="utf-8"
        )

        env = os.environ.copy()
        env[ENV_HARNESS_ACTIVE] = "1"
        env[ENV_HARNESS_ROLE] = "child"
        env[ENV_PAYLOAD_PATH] = str(payload_path)
        env[ENV_RESULT_PATH] = str(result_path)
        env["PYTHONPATH"] = _prepend_pythonpath(
            env.get("PYTHONPATH"), str(PROJECT_ROOT)
        )
        fallback_engine_path = _fallback_engine_path()
        if fallback_engine_path and "MOJOGP_JIT_ENGINE_PATH" not in env:
            env["MOJOGP_JIT_ENGINE_PATH"] = fallback_engine_path
        if extra_env is not None:
            env.update({key: str(value) for key, value in extra_env.items()})

        command = [sys.executable, "-m", module]
        try:
            completed = _run_child_command(
                command=command,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise IsolatedSubprocessError(
                message=(description or f"Isolated subprocess timed out for {module}"),
                command=command,
                cwd=str(cwd),
                timeout=timeout,
                returncode=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            ) from exc

        envelope = _read_result_envelope(result_path, completed.stdout)
        if completed.returncode != 0 or envelope is None or envelope.get("status") != "ok":
            raise IsolatedSubprocessError(
                message=(description or f"Isolated subprocess failed for {module}"),
                command=command,
                cwd=str(cwd),
                timeout=timeout,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                envelope=envelope,
            )
        return envelope


def load_child_result(
    envelope: Mapping[str, Any],
    *,
    result_loader: Callable[[Path], Any] | None = None,
) -> Any:
    result_path = envelope.get("result_path")
    if result_loader is not None:
        if not result_path:
            raise ValueError("Expected child envelope to include result_path")
        return result_loader(Path(str(result_path)))
    return envelope.get("payload")


def run_isolated_case(
    *,
    module: str,
    payload: Mapping[str, Any],
    timeout: int,
    cwd: Path | None = None,
    extra_env: Mapping[str, str] | None = None,
    description: str | None = None,
    result_loader: Callable[[Path], Any] | None = None,
) -> Any:
    envelope = run_isolated_module(
        module=module,
        payload=payload,
        timeout=timeout,
        cwd=cwd,
        extra_env=extra_env,
        description=description,
    )
    return load_child_result(envelope, result_loader=result_loader)


def _normalize_child_output(output: Any) -> dict[str, Any]:
    if output is None:
        return {}
    if isinstance(output, Mapping):
        return dict(output)
    return {"payload": output}


def _child_error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def run_child_main(
    handler: Callable[[dict[str, Any], IsolatedGPUTestSession], Any],
    *,
    monitor_interval: float = 0.1,
) -> int:
    """Run a child entrypoint under the standard harness envelope."""

    session: IsolatedGPUTestSession | None = None
    try:
        payload = load_child_payload()
        with isolated_gpu_test_session(monitor_interval=monitor_interval) as session:
            output = _normalize_child_output(handler(payload, session))
            snapshots = output.pop("memory_snapshots", None)
            timing = output.pop("timing", None)
            envelope = {
                "status": "ok",
                "payload": output.get("payload"),
                "result_path": (
                    None
                    if output.get("result_path") is None
                    else str(Path(str(output["result_path"])).resolve())
                ),
                "telemetry": session.collect_telemetry(
                    snapshots=list(snapshots or []),
                    timing=dict(timing or {}),
                ),
                "error": None,
            }
            emit_child_result(envelope)
        return 0
    except Exception as exc:
        telemetry = None
        if session is not None:
            telemetry = session.collect_telemetry()
        emit_child_result(
            {
                "status": "error",
                "payload": None,
                "result_path": None,
                "telemetry": telemetry,
                "error": _child_error_payload(exc),
            }
        )
        return 1
