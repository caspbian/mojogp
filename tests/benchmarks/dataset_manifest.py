"""Persisted benchmark dataset generation and loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifact_store import save_json_artifact
from .paths import DEFAULT_DATASET_ROOT, ensure_storage_dirs


@dataclass(frozen=True)
class DatasetSpec:
    generator_name: str
    config: dict[str, Any]

    @property
    def dataset_id(self) -> str:
        normalized = json.dumps(
            {
                "generator_name": self.generator_name,
                "config": self.config,
            },
            sort_keys=True,
            default=_json_default,
        )
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"{self.generator_name}_{digest}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _dataset_npz_path(dataset_id: str, dataset_root: Path) -> Path:
    return dataset_root / f"{dataset_id}.npz"


def _dataset_meta_path(dataset_id: str, dataset_root: Path) -> Path:
    return dataset_root / f"{dataset_id}.json"


def save_dataset(
    spec: DatasetSpec,
    dataset: Any,
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> tuple[str, Path, Path]:
    ensure_storage_dirs()
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_id = spec.dataset_id
    npz_path = _dataset_npz_path(dataset_id, dataset_root)
    meta_path = _dataset_meta_path(dataset_id, dataset_root)

    payload: dict[str, Any] = {}
    for name, value in vars(dataset).items():
        payload[name] = value
    np.savez(npz_path, **payload)
    meta = {
        "dataset_id": dataset_id,
        "generator_name": spec.generator_name,
        "config": spec.config,
        "npz_path": str(npz_path),
        "fields": sorted(payload.keys()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
    return dataset_id, npz_path, meta_path


def ensure_dataset(
    spec: DatasetSpec,
    generator: Callable[..., Any],
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
) -> tuple[str, Path, Path]:
    ensure_storage_dirs()
    dataset_id = spec.dataset_id
    npz_path = _dataset_npz_path(dataset_id, dataset_root)
    meta_path = _dataset_meta_path(dataset_id, dataset_root)
    if npz_path.exists() and meta_path.exists():
        return dataset_id, npz_path, meta_path
    dataset = generator(**spec.config)
    return save_dataset(spec, dataset, dataset_root=dataset_root)


def load_dataset_artifact(dataset_path: str | Path) -> dict[str, Any]:
    with np.load(Path(dataset_path), allow_pickle=True) as data:
        loaded: dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype == object:
                loaded[key] = value.item()
            else:
                loaded[key] = value
        return loaded


def save_dataset_artifact_bundle(
    dataset_id: str,
    *,
    dataset_npz_path: Path,
    dataset_meta_path: Path,
) -> tuple[Path, str, str]:
    payload = {
        "dataset_id": dataset_id,
        "dataset_npz_path": str(dataset_npz_path),
        "dataset_meta_path": str(dataset_meta_path),
        "dataset_meta": json.loads(dataset_meta_path.read_text(encoding="utf-8")),
    }
    return save_json_artifact(dataset_id, payload, artifact_type="dataset_manifest")
