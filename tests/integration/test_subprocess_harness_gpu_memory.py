"""Integration tests for subprocess-harness GPU memory isolation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.shared.subprocess_harness import run_isolated_module

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS_FIXTURE_MODULE = "tests.shared.subprocess_harness_fixture"
GPU_HOLDER_MODULE = "tests.shared.gpu_memory_holder"


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _assert_mb_close(measured_mb: float, expected_mb: float, *, tolerance_mb: float) -> None:
    assert abs(measured_mb - expected_mb) <= tolerance_mb, (
        f"measured {measured_mb:.1f} MB vs expected {expected_mb:.1f} MB "
        f"(tolerance {tolerance_mb:.1f} MB)"
    )


@contextmanager
def _external_gpu_holder(size_mb: float, *, timeout_s: float = 30.0):
    with tempfile.TemporaryDirectory(prefix="mojogp_gpu_holder_") as temp_dir:
        temp_path = Path(temp_dir)
        ready_path = temp_path / "ready"
        stop_path = temp_path / "stop"

        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH")
        if pythonpath:
            env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), pythonpath])
        else:
            env["PYTHONPATH"] = str(PROJECT_ROOT)
        env.update(
            {
                "MOJOGP_TEST_GPU_HOLDER_MB": str(size_mb),
                "MOJOGP_TEST_GPU_HOLDER_READY": str(ready_path),
                "MOJOGP_TEST_GPU_HOLDER_STOP": str(stop_path),
                "MOJOGP_TEST_GPU_HOLDER_TIMEOUT_S": str(timeout_s),
            }
        )

        proc = subprocess.Popen(
            [sys.executable, "-m", GPU_HOLDER_MODULE],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            deadline = time.perf_counter() + timeout_s
            while not ready_path.exists():
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate(timeout=1)
                    raise AssertionError(
                        "GPU holder exited before becoming ready\n"
                        f"stdout:\n{stdout}\n"
                        f"stderr:\n{stderr}"
                    )
                if time.perf_counter() >= deadline:
                    raise AssertionError("Timed out waiting for GPU holder readiness")
                time.sleep(0.05)
            yield
        finally:
            stop_path.write_text("stop", encoding="utf-8")
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate(timeout=5)
                raise AssertionError(
                    "GPU holder did not exit cleanly\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )
            assert proc.returncode == 0, (
                "GPU holder exited with failure\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_harness_ignores_preexisting_external_gpu_usage_in_isolated_delta():
    external_mb = 256.0
    inject_mb = 128.0

    with _external_gpu_holder(external_mb):
        envelope = run_isolated_module(
            module=HARNESS_FIXTURE_MODULE,
            payload={"case": "gpu_injection", "inject_mb": inject_mb, "hold_s": 0.35},
            timeout=90,
            description="Runs harness GPU injection with external baseline",
        )

    telemetry = envelope["telemetry"]
    gpu_memory = telemetry["memory"]["gpu"]
    torch_memory = telemetry["memory"]["torch"]

    assert float(gpu_memory["baseline_mb"]) >= external_mb * 0.75
    assert float(torch_memory["peak_delta_mb"]) >= inject_mb
    assert float(gpu_memory["isolated_peak_mb"]) >= inject_mb
    expected_peak = max(
        float(gpu_memory["delta_mb"]),
        float(torch_memory["peak_delta_mb"]),
    )
    assert abs(float(gpu_memory["isolated_peak_mb"]) - expected_peak) <= 1e-6
    assert float(gpu_memory["baseline_mb"]) > float(gpu_memory["isolated_peak_mb"])


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_harness_reports_similar_isolated_delta_for_pytorch_and_mojogp_style_children():
    inject_mb = 96.0

    envelope_a = run_isolated_module(
        module=HARNESS_FIXTURE_MODULE,
        payload={"case": "gpu_injection", "inject_mb": inject_mb, "hold_s": 0.35},
        timeout=60,
        description="Runs first harness GPU injection child",
    )
    envelope_b = run_isolated_module(
        module=HARNESS_FIXTURE_MODULE,
        payload={"case": "gpu_injection", "inject_mb": inject_mb, "hold_s": 0.35},
        timeout=60,
        description="Runs second harness GPU injection child",
    )

    delta_a = float(envelope_a["telemetry"]["memory"]["gpu"]["isolated_peak_mb"])
    delta_b = float(envelope_b["telemetry"]["memory"]["gpu"]["isolated_peak_mb"])

    assert delta_a >= inject_mb, f"isolated delta should not undercount injected allocation: {delta_a:.1f} MB"
    assert delta_b >= inject_mb, f"isolated delta should not undercount injected allocation: {delta_b:.1f} MB"
    assert abs(delta_a - delta_b) <= 16.0, (
        f"isolated deltas diverged too much across subprocesses: {delta_a:.1f} vs {delta_b:.1f} MB"
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_harness_isolated_peak_uses_larger_of_global_and_torch_signals():
    inject_mb = 96.0

    envelope = run_isolated_module(
        module=HARNESS_FIXTURE_MODULE,
        payload={"case": "gpu_injection", "inject_mb": inject_mb, "hold_s": 0.35},
        timeout=60,
        description="Runs harness GPU injection for larger-signal contract",
    )

    gpu_memory = envelope["telemetry"]["memory"]["gpu"]
    torch_memory = envelope["telemetry"]["memory"]["torch"]

    expected_peak = max(
        float(gpu_memory["delta_mb"]),
        float(torch_memory["peak_delta_mb"]),
    )
    assert abs(float(gpu_memory["isolated_peak_mb"]) - expected_peak) <= 1e-6, (
        "isolated peak should follow the larger of the global GPU peak delta and "
        "the torch allocator peak delta"
    )
    assert float(gpu_memory["isolated_peak_mb"]) >= inject_mb
