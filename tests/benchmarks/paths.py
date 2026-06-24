"""Shared storage paths for benchmark infrastructure."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYSTEM_BENCHMARK_ROOT = PROJECT_ROOT / "tests" / "system_benchmarks"
ACCURACY_BENCHMARK_ROOT = PROJECT_ROOT / "tests" / "accuracy_benchmarks"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


DEFAULT_DB_PATH = _env_path(
    "MOJOGP_BENCHMARK_DB_PATH", SYSTEM_BENCHMARK_ROOT / "benchmark_results.sqlite"
)
DEFAULT_ARTIFACT_ROOT = _env_path(
    "MOJOGP_BENCHMARK_ARTIFACT_ROOT", SYSTEM_BENCHMARK_ROOT / "artifacts"
)
DEFAULT_DATASET_ROOT = _env_path(
    "MOJOGP_BENCHMARK_DATASET_ROOT", SYSTEM_BENCHMARK_ROOT / "datasets"
)
DEFAULT_SESSION_EXPORT_ROOT = _env_path(
    "MOJOGP_BENCHMARK_SESSION_EXPORT_ROOT", SYSTEM_BENCHMARK_ROOT / "session_exports"
)

DEFAULT_ACCURACY_DB_PATH = _env_path(
    "MOJOGP_ACCURACY_BENCHMARK_DB_PATH",
    ACCURACY_BENCHMARK_ROOT / "accuracy_results.sqlite",
)
DEFAULT_ACCURACY_ARTIFACT_ROOT = _env_path(
    "MOJOGP_ACCURACY_BENCHMARK_ARTIFACT_ROOT",
    ACCURACY_BENCHMARK_ROOT / "artifacts",
)
DEFAULT_ACCURACY_DATASET_ROOT = _env_path(
    "MOJOGP_ACCURACY_BENCHMARK_DATASET_ROOT",
    ACCURACY_BENCHMARK_ROOT / "datasets",
)
DEFAULT_ACCURACY_SESSION_EXPORT_ROOT = _env_path(
    "MOJOGP_ACCURACY_BENCHMARK_SESSION_EXPORT_ROOT",
    ACCURACY_BENCHMARK_ROOT / "session_exports",
)
DEFAULT_ACCURACY_RESULTS_DIR = _env_path(
    "MOJOGP_ACCURACY_BENCHMARK_RESULTS_DIR",
    ACCURACY_BENCHMARK_ROOT / "results",
)


def ensure_storage_dirs() -> None:
    DEFAULT_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_SESSION_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def accuracy_benchmark_env_defaults() -> dict[str, str]:
    """Env overrides that route generic benchmark storage to the accuracy track."""

    return {
        "MOJOGP_BENCHMARK_DB_PATH": str(DEFAULT_ACCURACY_DB_PATH),
        "MOJOGP_BENCHMARK_ARTIFACT_ROOT": str(DEFAULT_ACCURACY_ARTIFACT_ROOT),
        "MOJOGP_BENCHMARK_DATASET_ROOT": str(DEFAULT_ACCURACY_DATASET_ROOT),
        "MOJOGP_BENCHMARK_SESSION_EXPORT_ROOT": str(DEFAULT_ACCURACY_SESSION_EXPORT_ROOT),
    }


def ensure_accuracy_storage_dirs() -> None:
    DEFAULT_ACCURACY_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_ACCURACY_DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_ACCURACY_SESSION_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_ACCURACY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
