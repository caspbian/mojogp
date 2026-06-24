"""Raw artifact storage for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .paths import DEFAULT_ARTIFACT_ROOT, ensure_storage_dirs


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json_artifact(
    artifact_id: str,
    payload: dict[str, Any],
    *,
    artifact_type: str = "benchmark_run",
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> tuple[Path, str, str]:
    ensure_storage_dirs()
    target_dir = artifact_root / artifact_type
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / f"{artifact_id}.json"
    serialized = json.dumps(payload, indent=2, default=_json_default, sort_keys=True)
    if artifact_path.exists():
        existing = artifact_path.read_text(encoding="utf-8")
        if existing == serialized:
            sha256 = hashlib.sha256(existing.encode("utf-8")).hexdigest()
            return artifact_path, sha256, existing
    temp_path = target_dir / f".{artifact_path.name}.{os.getpid()}.tmp"
    try:
        temp_path.write_text(serialized, encoding="utf-8")
        temp_path.replace(artifact_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return artifact_path, sha256, serialized
