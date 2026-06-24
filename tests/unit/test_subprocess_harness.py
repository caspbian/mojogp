"""Contract coverage for the shared subprocess harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from tests.shared.subprocess_harness import (
    IsolatedGPUTestSession,
    IsolatedSubprocessError,
    run_isolated_case,
    run_isolated_module,
)


MODULE = "tests.shared.subprocess_harness_fixture"


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _assert_mb_close(measured_mb: float, expected_mb: float, *, tolerance_mb: float) -> None:
    assert abs(measured_mb - expected_mb) <= tolerance_mb, (
        f"measured {measured_mb:.1f} MB vs expected {expected_mb:.1f} MB "
        f"(tolerance {tolerance_mb:.1f} MB)"
    )


def test_run_isolated_module_returns_success_envelope():
    envelope = run_isolated_module(
        module=MODULE,
        payload={"case": "echo", "message": "hello"},
        timeout=30,
        description="Runs echo harness fixture",
    )

    assert envelope["status"] == "ok"
    assert envelope["payload"] == {"message": "hello", "child_flag": True}
    assert envelope["error"] is None
    telemetry = envelope["telemetry"]
    assert telemetry["timing"]["case"] == "echo"
    assert "memory" in telemetry
    assert "gpu" in telemetry["memory"]
    assert "torch" in telemetry["memory"]
    assert "cpu" in telemetry["memory"]


def test_run_isolated_case_can_load_result_path_payload():
    def _load_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    result = run_isolated_case(
        module=MODULE,
        payload={"case": "result_path", "message": "from-file"},
        timeout=30,
        description="Runs result-path harness fixture",
        result_loader=_load_json,
    )

    assert result == {"message": "from-file"}


def test_run_isolated_module_surfaces_uniform_failure_details():
    with pytest.raises(IsolatedSubprocessError) as exc_info:
        run_isolated_module(
            module=MODULE,
            payload={"case": "failure", "message": "boom"},
            timeout=30,
            description="Runs failing harness fixture",
        )

    message = str(exc_info.value)
    assert "Runs failing harness fixture" in message
    assert "child_status: error" in message
    assert "child_error: RuntimeError: boom" in message
    assert "stdout_tail:" in message
    assert "stderr_tail:" in message


def test_collect_memory_stats_prefers_larger_global_delta_over_smaller_torch_delta(monkeypatch):
    import tests.shared.subprocess_harness as harness_module

    class _FakeMonitor:
        def get_stats(self):
            return {
                "mean_mb": 850.0,
                "min_mb": 520.0,
                "max_mb": 900.0,
                "var_mb": 1000.0,
                "samples": 5,
                "method": "pynvml",
            }

        def stop(self):
            return None

    session = IsolatedGPUTestSession()
    session._monitor = _FakeMonitor()
    session._process = None
    session._baseline_gpu_mb = 500.0
    session._torch_baseline_mb = 0.0
    session._torch_reserved_baseline_mb = 0.0
    monkeypatch.setattr(
        harness_module,
        "get_torch_memory_stats",
        lambda: {
            "torch_peak_mb": 128.0,
            "torch_current_mb": 64.0,
            "torch_reserved_mb": 128.0,
        },
    )
    monkeypatch.setattr(
        session,
        "snapshot_gpu",
        lambda: {
            "peak_gpu_mb": 700.0,
            "current_gpu_mb": 800.0,
            "delta_gpu_mb": 300.0,
            "torch_peak_mb": 128.0,
            "torch_current_mb": 64.0,
            "method": "pynvml",
        },
    )

    memory = session.collect_memory_stats()

    assert float(memory["isolated_peak_gpu_mb"]) == 400.0
    assert float(memory["isolated_current_gpu_mb"]) == 300.0


def test_collect_memory_stats_keeps_max_consistent_with_final_current(monkeypatch):
    import tests.shared.subprocess_harness as harness_module

    class _FakeMonitor:
        def get_stats(self):
            return {
                "mean_mb": 550.0,
                "min_mb": 500.0,
                "max_mb": 600.0,
                "var_mb": 50.0,
                "samples": 5,
                "method": "pynvml",
            }

        def stop(self):
            return None

    session = IsolatedGPUTestSession()
    session._monitor = _FakeMonitor()
    session._process = None
    session._baseline_gpu_mb = 500.0
    session._torch_baseline_mb = 0.0
    session._torch_reserved_baseline_mb = 0.0
    monkeypatch.setattr(
        harness_module,
        "get_torch_memory_stats",
        lambda: {
            "torch_peak_mb": 0.0,
            "torch_current_mb": 0.0,
            "torch_reserved_mb": 0.0,
        },
    )
    monkeypatch.setattr(
        session,
        "snapshot_gpu",
        lambda: {
            "peak_gpu_mb": 550.0,
            "current_gpu_mb": 700.0,
            "delta_gpu_mb": 200.0,
            "torch_peak_mb": 0.0,
            "torch_current_mb": 0.0,
            "method": "pynvml",
        },
    )

    memory = session.collect_memory_stats()

    assert float(memory["max_mb"]) == 700.0
    assert float(memory["delta_gpu_mb"]) == 200.0
    assert float(memory["isolated_peak_gpu_mb"]) == 200.0


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_run_isolated_module_reports_known_gpu_injection_delta():
    import_baseline_mb = 96.0
    inject_mb = 64.0

    envelope = run_isolated_module(
        module=MODULE,
        payload={"case": "gpu_injection", "inject_mb": inject_mb, "hold_s": 0.35},
        timeout=60,
        description="Runs harness GPU injection fixture",
        extra_env={"MOJOGP_HARNESS_IMPORT_BASELINE_MB": str(import_baseline_mb)},
    )

    telemetry = envelope["telemetry"]
    gpu_memory = telemetry["memory"]["gpu"]
    torch_memory = telemetry["memory"]["torch"]

    assert torch_memory["baseline_mb"] >= import_baseline_mb * 0.75
    _assert_mb_close(
        float(torch_memory["peak_delta_mb"]),
        inject_mb,
        tolerance_mb=32.0,
    )
    _assert_mb_close(
        float(gpu_memory["isolated_peak_mb"]),
        inject_mb,
        tolerance_mb=32.0,
    )
    assert float(torch_memory["peak_delta_mb"]) < import_baseline_mb + inject_mb - 16.0
