"""Fixtures for benchmark-only infrastructure tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .session_store import BenchmarkSessionStore


@pytest.fixture
def benchmark_db_path(tmp_path: Path) -> Path:
    return tmp_path / "benchmark_results.sqlite"


@pytest.fixture
def benchmark_store(benchmark_db_path: Path) -> BenchmarkSessionStore:
    return BenchmarkSessionStore(benchmark_db_path)
