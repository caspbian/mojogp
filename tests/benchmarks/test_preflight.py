from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.preflight import (
    BenchmarkPreflightError,
    DIRTY_WORKTREE_COMMIT_HASH,
    DIRTY_WORKTREE_COMMIT_HASH_SHORT,
    get_benchmark_git_metadata,
    get_git_metadata,
    profiling_config_is_disabled,
    require_clean_git_worktree,
)


def test_profiling_config_is_disabled_reads_false_flag(tmp_path: Path):
    kernels_dir = tmp_path / "mojogp" / "kernels"
    kernels_dir.mkdir(parents=True)
    (kernels_dir / "profiling_config.mojo").write_text(
        '"""Compile-time profiling configuration for Mojo kernels."""\n\nalias PROFILING = False\n',
        encoding="utf-8",
    )
    assert profiling_config_is_disabled(project_root=tmp_path)


def test_require_clean_git_worktree_fails_when_dirty(monkeypatch):
    from tests.benchmarks import preflight as module

    def _fake_run_git(args, *, cwd):
        if args[:2] == ["status", "--porcelain"]:
            return " M dirty_file.py"
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return "feature/test"
        if args[:2] == ["rev-parse", "--short=8"]:
            return "abc123de"
        if args[:1] == ["rev-parse"]:
            return "abc123def456"
        raise AssertionError(args)

    monkeypatch.setattr(module, "_run_git", _fake_run_git)
    with pytest.raises(BenchmarkPreflightError):
        require_clean_git_worktree()


def test_get_benchmark_git_metadata_uses_sentinel_hash_when_dirty_in_development(monkeypatch):
    from tests.benchmarks import preflight as module

    def _fake_run_git(args, *, cwd):
        if args[:2] == ["status", "--porcelain"]:
            return " M dirty_file.py"
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return "feature/test"
        if args[:2] == ["rev-parse", "--short=8"]:
            return "abc123de"
        if args[:1] == ["rev-parse"]:
            return "abc123def456"
        raise AssertionError(args)

    monkeypatch.setattr(module, "_run_git", _fake_run_git)
    monkeypatch.delenv("MOJOGP_BENCHMARK_ENFORCE_CLEAN_GIT", raising=False)
    monkeypatch.delenv("CI", raising=False)

    metadata = get_benchmark_git_metadata()

    assert metadata.branch_name == "feature/test"
    assert metadata.commit_hash == DIRTY_WORKTREE_COMMIT_HASH
    assert metadata.commit_hash_short == DIRTY_WORKTREE_COMMIT_HASH_SHORT
    assert metadata.git_clean is False


def test_get_git_metadata_uses_env_override(monkeypatch):
    monkeypatch.setenv("MOJOGP_SOURCE_BRANCH_NAME", "feature/container-provenance")
    monkeypatch.setenv("MOJOGP_SOURCE_COMMIT_HASH", "abc123def4567890")
    monkeypatch.setenv("MOJOGP_SOURCE_COMMIT_HASH_SHORT", "abc123de")
    monkeypatch.setenv("MOJOGP_SOURCE_GIT_CLEAN", "true")
    monkeypatch.setenv("MOJOGP_SOURCE_WORKTREE_PATH", "/workspace-host")

    metadata = get_git_metadata()

    assert metadata.branch_name == "feature/container-provenance"
    assert metadata.commit_hash == "abc123def4567890"
    assert metadata.commit_hash_short == "abc123de"
    assert metadata.git_clean is True
    assert metadata.worktree_path == "/workspace-host"
