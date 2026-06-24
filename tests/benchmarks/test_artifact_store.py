from __future__ import annotations

import json

from tests.benchmarks.artifact_store import save_json_artifact


def test_save_json_artifact_reuses_identical_read_only_artifact(tmp_path):
    artifact_root = tmp_path / "artifacts"
    target_dir = artifact_root / "dataset_manifest"
    target_dir.mkdir(parents=True)
    artifact_path = target_dir / "deterministic.json"
    payload = {"b": 2, "a": 1}
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    artifact_path.write_text(serialized, encoding="utf-8")
    artifact_path.chmod(0o444)

    path, sha256, written = save_json_artifact(
        "deterministic",
        payload,
        artifact_type="dataset_manifest",
        artifact_root=artifact_root,
    )

    assert path == artifact_path
    assert written == serialized
    assert len(sha256) == 64


def test_save_json_artifact_replaces_stale_read_only_artifact(tmp_path):
    artifact_root = tmp_path / "artifacts"
    target_dir = artifact_root / "dataset_manifest"
    target_dir.mkdir(parents=True)
    artifact_path = target_dir / "deterministic.json"
    artifact_path.write_text('{"old": true}', encoding="utf-8")
    artifact_path.chmod(0o444)
    payload = {"new": True}

    try:
        path, _, written = save_json_artifact(
            "deterministic",
            payload,
            artifact_type="dataset_manifest",
            artifact_root=artifact_root,
        )
    finally:
        if artifact_path.exists():
            artifact_path.chmod(0o644)

    assert path == artifact_path
    assert written == json.dumps(payload, indent=2, sort_keys=True)
    assert artifact_path.read_text(encoding="utf-8") == written
