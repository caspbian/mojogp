"""Runtime benchmark session context."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from .preflight import (
    ContainerMetadata,
    GitMetadata,
    ProfilingMetadata,
    BenchmarkPreflightError,
    get_container_metadata,
    new_session_id,
    require_benchmark_preflight,
    utc_now_iso,
)
from .session_store import BenchmarkSessionRecord, BenchmarkSessionStore


@dataclass
class BenchmarkRuntimeContext:
    session_id: str
    session_store: BenchmarkSessionStore
    git: GitMetadata
    container: ContainerMetadata
    profiling: ProfilingMetadata
    started_at: str


_DEFAULT_CONTEXT: BenchmarkRuntimeContext | None = None


def get_or_create_default_context() -> BenchmarkRuntimeContext:
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is not None:
        return _DEFAULT_CONTEXT

    git, profiling = require_benchmark_preflight()
    container = get_container_metadata()
    store = BenchmarkSessionStore()
    started_at = utc_now_iso()
    session_id = new_session_id()
    store.upsert_session(
        BenchmarkSessionRecord(
            session_id=session_id,
            started_at=started_at,
            finished_at=None,
            branch_name=git.branch_name,
            commit_hash=git.commit_hash,
            commit_hash_short=git.commit_hash_short,
            git_clean=git.git_clean,
            worktree_path=git.worktree_path,
            profiling_config_false=profiling.profiling_config_false,
            profiling_probe_passed=profiling.profiling_probe_passed,
            jit_cache_dir=profiling.jit_cache_dir,
            container_runtime=container.runtime,
            container_image_tag=container.image_tag,
            container_image_digest=container.image_digest,
            container_image_id=container.image_id,
            gpu_name=container.gpu_name,
            gpu_total_vram_mb=container.gpu_total_vram_mb,
            gpu_driver_version=container.gpu_driver_version,
            cuda_version=container.cuda_version,
            gpu_target=container.gpu_target,
        )
    )
    _DEFAULT_CONTEXT = BenchmarkRuntimeContext(
        session_id=session_id,
        session_store=store,
        git=git,
        container=container,
        profiling=profiling,
        started_at=started_at,
    )
    return _DEFAULT_CONTEXT


def current_default_context() -> BenchmarkRuntimeContext | None:
    return _DEFAULT_CONTEXT


def finalize_default_context() -> None:
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        return
    finished_at = utc_now_iso()
    _DEFAULT_CONTEXT.session_store.upsert_session(
        BenchmarkSessionRecord(
            session_id=_DEFAULT_CONTEXT.session_id,
            started_at=_DEFAULT_CONTEXT.started_at,
            finished_at=finished_at,
            branch_name=_DEFAULT_CONTEXT.git.branch_name,
            commit_hash=_DEFAULT_CONTEXT.git.commit_hash,
            commit_hash_short=_DEFAULT_CONTEXT.git.commit_hash_short,
            git_clean=_DEFAULT_CONTEXT.git.git_clean,
            worktree_path=_DEFAULT_CONTEXT.git.worktree_path,
            profiling_config_false=_DEFAULT_CONTEXT.profiling.profiling_config_false,
            profiling_probe_passed=_DEFAULT_CONTEXT.profiling.profiling_probe_passed,
            jit_cache_dir=_DEFAULT_CONTEXT.profiling.jit_cache_dir,
            container_runtime=_DEFAULT_CONTEXT.container.runtime,
            container_image_tag=_DEFAULT_CONTEXT.container.image_tag,
            container_image_digest=_DEFAULT_CONTEXT.container.image_digest,
            container_image_id=_DEFAULT_CONTEXT.container.image_id,
            gpu_name=_DEFAULT_CONTEXT.container.gpu_name,
            gpu_total_vram_mb=_DEFAULT_CONTEXT.container.gpu_total_vram_mb,
            gpu_driver_version=_DEFAULT_CONTEXT.container.gpu_driver_version,
            cuda_version=_DEFAULT_CONTEXT.container.cuda_version,
            gpu_target=_DEFAULT_CONTEXT.container.gpu_target,
        )
    )
    _DEFAULT_CONTEXT.session_store.export_session_json(_DEFAULT_CONTEXT.session_id)
    shutil.rmtree(_DEFAULT_CONTEXT.profiling.jit_cache_dir, ignore_errors=True)
    _DEFAULT_CONTEXT = None


def reset_default_context() -> None:
    global _DEFAULT_CONTEXT
    _DEFAULT_CONTEXT = None
