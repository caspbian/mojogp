from __future__ import annotations

from pathlib import Path

from tests.benchmarks.preflight import ContainerMetadata, GitMetadata, ProfilingMetadata, utc_now_iso
from tests.benchmarks.runtime import (
    BenchmarkRuntimeContext,
    current_default_context,
    finalize_default_context,
)
from tests.benchmarks.session_store import BenchmarkSessionRecord


def test_finalize_default_context_exports_session_and_cleans_cache(benchmark_store, tmp_path: Path, monkeypatch):
    from tests.benchmarks import runtime as runtime_module

    export_root = tmp_path / "exports"
    monkeypatch.setattr(
        "tests.benchmarks.session_store.DEFAULT_SESSION_EXPORT_ROOT",
        export_root,
    )

    jit_cache_dir = tmp_path / "jit-cache"
    jit_cache_dir.mkdir()
    git = GitMetadata(
        branch_name="feature/test",
        commit_hash="abc123def456",
        commit_hash_short="abc123de",
        git_clean=True,
        worktree_path="/tmp/worktree",
    )
    profiling = ProfilingMetadata(
        profiling_config_false=True,
        profiling_probe_passed=True,
        jit_cache_dir=str(jit_cache_dir),
    )
    benchmark_store.upsert_session(
        BenchmarkSessionRecord(
            session_id="runtime-session",
            started_at=utc_now_iso(),
            finished_at=None,
            branch_name=git.branch_name,
            commit_hash=git.commit_hash,
            commit_hash_short=git.commit_hash_short,
            git_clean=git.git_clean,
            worktree_path=git.worktree_path,
            profiling_config_false=profiling.profiling_config_false,
            profiling_probe_passed=profiling.profiling_probe_passed,
            jit_cache_dir=profiling.jit_cache_dir,
        )
    )
    runtime_module._DEFAULT_CONTEXT = BenchmarkRuntimeContext(
        session_id="runtime-session",
        session_store=benchmark_store,
        git=git,
        container=ContainerMetadata(
            runtime="docker",
            image_tag="mojogp-benchmark:test",
            image_id="sha256:test",
            gpu_name="NVIDIA GeForce RTX 4050",
            gpu_total_vram_mb=6141.0,
            gpu_driver_version="580.126.09",
            cuda_version="13.0",
            gpu_target="sm_89",
        ),
        profiling=profiling,
        started_at=utc_now_iso(),
    )

    finalize_default_context()

    assert current_default_context() is None
    assert not jit_cache_dir.exists()
    export_path = export_root / "runtime-session.json"
    assert export_path.exists()
    with benchmark_store._connect() as conn:
        row = conn.execute(
            "SELECT finished_at, gpu_name, gpu_target FROM benchmark_sessions WHERE session_id='runtime-session'"
        ).fetchone()
        assert row is not None
        assert row["finished_at"] is not None
        assert row["gpu_name"] == "NVIDIA GeForce RTX 4050"
        assert row["gpu_target"] == "sm_89"
