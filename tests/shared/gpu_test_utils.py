"""Shared GPU test helpers for correctness and benchmark suites."""

from __future__ import annotations

import pytest


def has_gpytorch() -> bool:
    try:
        import gpytorch  # noqa: F401

        return True
    except ImportError:
        return False


def has_mojogp() -> bool:
    try:
        import mojogp  # noqa: F401

        return True
    except ImportError:
        return False


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, RuntimeError):
        return False


requires_gpytorch = pytest.mark.skipif(not has_gpytorch(), reason="GPyTorch not installed")
requires_mojogp = pytest.mark.skipif(not has_mojogp(), reason="MojoGP not installed")
requires_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")


def assert_gpu_available() -> None:
    import torch

    assert torch.cuda.is_available(), "GPU required for this test - CUDA not available"
    try:
        device_count = torch.cuda.device_count()
        assert device_count > 0, "GPU required for this test - no CUDA devices found"
    except RuntimeError as exc:
        pytest.fail(f"GPU required for this test - CUDA error: {exc}")


def assert_gpu_was_used(result) -> None:
    if hasattr(result, "memory"):
        gpu_mem = result.memory.gpu_max_mb
    elif isinstance(result, dict):
        gpu_mem = result.get("memory_stats", {}).get("max_mb", 0)
        if gpu_mem == 0:
            gpu_mem = result.get("peak_memory_mb", 0)
    else:
        gpu_mem = 0

    assert gpu_mem > 0, (
        f"GPU was not used during benchmark (gpu_max_mb={gpu_mem}). "
        "This test requires GPU execution."
    )
