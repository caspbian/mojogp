"""Integration tests for native Mojo GPU allocation accounting."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOJO_SCRIPT = PROJECT_ROOT / "tests" / "fixtures" / "mojo" / "mojo_gpu_allocation_probe.mojo"
GPU_HOLDER_MODULE = "tests.shared.gpu_memory_holder"


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _mojo_available() -> bool:
    return shutil.which("mojo") is not None


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
            stdout, stderr = proc.communicate(timeout=10)
            assert proc.returncode == 0, (
                "GPU holder exited with failure\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )


def _parse_probe_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.startswith("mojo_gpu_allocation_probe_"):
            metrics[key] = float(value.strip())
    return metrics


def _run_mojo_injection(*, inject_mb: float, hold_s: float) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="mojogp_mojo_injection_") as temp_dir:
        binary_path = Path(temp_dir) / "mojo_gpu_allocation_probe"
        build = subprocess.run(
            ["mojo", "build", str(MOJO_SCRIPT), "-o", str(binary_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if build.returncode != 0:
            raise AssertionError(
                f"mojo build failed\nstdout:\n{build.stdout}\nstderr:\n{build.stderr}"
            )

        env = os.environ.copy()
        env["MOJOGP_TEST_MOJO_GPU_INJECT_MB"] = str(int(inject_mb))
        env["MOJOGP_TEST_MOJO_GPU_HOLD_S"] = f"{hold_s:.2f}"
        proc = subprocess.run(
            [str(binary_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            raise AssertionError(
                f"mojo allocation probe failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        metrics = _parse_probe_metrics(proc.stdout)
        required = [
            "mojo_gpu_allocation_probe_mb",
            "mojo_gpu_allocation_probe_free_before_bytes",
            "mojo_gpu_allocation_probe_free_after_bytes",
            "mojo_gpu_allocation_probe_delta_bytes",
        ]
        missing = [key for key in required if key not in metrics]
        if missing:
            raise AssertionError(
                "mojo allocation probe did not report required metrics\n"
                f"missing={missing}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        return {
            "requested_mb": metrics["mojo_gpu_allocation_probe_mb"],
            "free_before_mb": metrics["mojo_gpu_allocation_probe_free_before_bytes"]
            / (1024 * 1024),
            "free_after_mb": metrics["mojo_gpu_allocation_probe_free_after_bytes"]
            / (1024 * 1024),
            "delta_mb": metrics["mojo_gpu_allocation_probe_delta_bytes"]
            / (1024 * 1024),
        }


@pytest.mark.skipif(not (_cuda_available() and _mojo_available()), reason="CUDA GPU and mojo required")
def test_native_mojo_gpu_allocation_probe_reports_memory_delta():
    inject_mb = 128.0
    mem = _run_mojo_injection(inject_mb=inject_mb, hold_s=0.15)
    assert mem["requested_mb"] == inject_mb
    assert mem["free_after_mb"] <= mem["free_before_mb"], mem
    assert mem["delta_mb"] >= inject_mb * 0.9, mem


@pytest.mark.skipif(not (_cuda_available() and _mojo_available()), reason="CUDA GPU and mojo required")
def test_native_mojo_gpu_allocation_probe_scales_with_requested_size():
    small = _run_mojo_injection(inject_mb=64.0, hold_s=0.15)
    large = _run_mojo_injection(inject_mb=256.0, hold_s=0.15)
    assert small["delta_mb"] > 0, small
    assert large["delta_mb"] > small["delta_mb"], (
        f"larger native Mojo GPU allocation should produce a larger native delta: "
        f"small={small['delta_mb']:.1f} MB, large={large['delta_mb']:.1f} MB"
    )
