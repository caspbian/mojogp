from __future__ import annotations

from tests.benchmarks.dataset_preparation import (
    prepare_multi_output_scaling_dataset,
    prepare_single_output_scaling_dataset,
)
from tests.benchmarks.preflight import ContainerMetadata, GitMetadata, ProfilingMetadata, utc_now_iso
from tests.benchmarks.runtime import BenchmarkRuntimeContext
from tests.benchmarks.session_store import BenchmarkSessionRecord


def _context(benchmark_store):
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
        jit_cache_dir="/tmp/jit-cache",
    )
    benchmark_store.upsert_session(
        BenchmarkSessionRecord(
            session_id="dataset-session",
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
    return BenchmarkRuntimeContext(
        session_id="dataset-session",
        session_store=benchmark_store,
        git=git,
        container=ContainerMetadata(),
        profiling=profiling,
        started_at=utc_now_iso(),
    )


def test_prepare_single_output_scaling_dataset_registers_dataset(benchmark_store):
    context = _context(benchmark_store)
    prepared = prepare_single_output_scaling_dataset(
        dataset_family="structured_function",
        n_train=2000,
        n_test=200,
        d=5,
        seed=42,
        context=context,
    )
    assert prepared.npz_path.exists()
    with benchmark_store._connect() as conn:
        row = conn.execute(
            "SELECT generator_name FROM benchmark_datasets WHERE dataset_id=?",
            (prepared.dataset_id,),
        ).fetchone()
        assert row is not None
        assert row["generator_name"] == "single_output_structured_function"


def test_prepare_multi_output_scaling_dataset_registers_dataset(benchmark_store):
    context = _context(benchmark_store)
    prepared = prepare_multi_output_scaling_dataset(
        n_train=3000,
        n_test=120,
        d=5,
        num_tasks=3,
        seed=123,
        context=context,
    )
    assert prepared.meta_path.exists()
    with benchmark_store._connect() as conn:
        row = conn.execute(
            "SELECT generator_name FROM benchmark_datasets WHERE dataset_id=?",
            (prepared.dataset_id,),
        ).fetchone()
        assert row is not None
        assert row["generator_name"] == "multi_output_scaling_gp_prior"
