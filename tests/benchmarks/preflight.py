"""Benchmark preflight checks and session metadata."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

from .paths import PROJECT_ROOT


class BenchmarkPreflightError(RuntimeError):
    pass


DIRTY_WORKTREE_COMMIT_HASH = "dirty-worktree"
DIRTY_WORKTREE_COMMIT_HASH_SHORT = "dirtywrk"


@dataclass(frozen=True)
class GitMetadata:
    branch_name: str
    commit_hash: str
    commit_hash_short: str
    git_clean: bool
    worktree_path: str


@dataclass(frozen=True)
class ProfilingMetadata:
    profiling_config_false: bool
    profiling_probe_passed: bool
    jit_cache_dir: str


@dataclass(frozen=True)
class ContainerMetadata:
    runtime: str | None = None
    image_tag: str | None = None
    image_digest: str | None = None
    image_id: str | None = None
    gpu_name: str | None = None
    gpu_total_vram_mb: float | None = None
    gpu_driver_version: str | None = None
    cuda_version: str | None = None
    gpu_target: str | None = None

    @property
    def active(self) -> bool:
        return any(
            value
            for value in (
                self.runtime,
                self.image_tag,
                self.image_digest,
                self.image_id,
            )
        )


def _env_git_metadata() -> GitMetadata | None:
    branch_name = os.environ.get("MOJOGP_SOURCE_BRANCH_NAME")
    commit_hash = os.environ.get("MOJOGP_SOURCE_COMMIT_HASH")
    commit_hash_short = os.environ.get("MOJOGP_SOURCE_COMMIT_HASH_SHORT")
    git_clean_raw = os.environ.get("MOJOGP_SOURCE_GIT_CLEAN")
    worktree_path = os.environ.get("MOJOGP_SOURCE_WORKTREE_PATH") or str(PROJECT_ROOT)
    if not all((branch_name, commit_hash, commit_hash_short, git_clean_raw)):
        return None
    return GitMetadata(
        branch_name=branch_name,
        commit_hash=commit_hash,
        commit_hash_short=commit_hash_short,
        git_clean=git_clean_raw.strip().lower() in {"1", "true", "yes", "on"},
        worktree_path=worktree_path,
    )


def _run_git(args: list[str], *, cwd: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=str(cwd), timeout=10)
    if result.returncode != 0:
        raise BenchmarkPreflightError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def get_git_metadata(*, cwd: Path = PROJECT_ROOT) -> GitMetadata:
    env_metadata = _env_git_metadata()
    if env_metadata is not None:
        return env_metadata
    branch_name = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    commit_hash = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    commit_hash_short = _run_git(["rev-parse", "--short=8", "HEAD"], cwd=cwd)
    dirty = bool(_run_git(["status", "--porcelain"], cwd=cwd))
    return GitMetadata(
        branch_name=branch_name,
        commit_hash=commit_hash,
        commit_hash_short=commit_hash_short,
        git_clean=not dirty,
        worktree_path=str(cwd),
    )


def clean_git_enforcement_enabled() -> bool:
    raw = os.environ.get("MOJOGP_BENCHMARK_ENFORCE_CLEAN_GIT")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return os.environ.get("CI", "0").strip().lower() in {"1", "true", "yes", "on"}


def require_clean_git_worktree(*, cwd: Path = PROJECT_ROOT) -> GitMetadata:
    metadata = get_git_metadata(cwd=cwd)
    if not metadata.git_clean:
        raise BenchmarkPreflightError(
            "Benchmark runs require a clean git worktree; `git status --porcelain` was non-empty."
        )
    return metadata


def get_benchmark_git_metadata(*, cwd: Path = PROJECT_ROOT) -> GitMetadata:
    metadata = get_git_metadata(cwd=cwd)
    if metadata.git_clean or clean_git_enforcement_enabled():
        return require_clean_git_worktree(cwd=cwd) if not metadata.git_clean else metadata
    return GitMetadata(
        branch_name=metadata.branch_name,
        commit_hash=DIRTY_WORKTREE_COMMIT_HASH,
        commit_hash_short=DIRTY_WORKTREE_COMMIT_HASH_SHORT,
        git_clean=False,
        worktree_path=metadata.worktree_path,
    )


def get_container_metadata() -> ContainerMetadata:
    gpu_total_vram_mb_raw = os.environ.get("MOJOGP_GPU_TOTAL_VRAM_MB")
    return ContainerMetadata(
        runtime=os.environ.get("MOJOGP_CONTAINER_RUNTIME") or None,
        image_tag=os.environ.get("MOJOGP_CONTAINER_IMAGE_TAG") or None,
        image_digest=os.environ.get("MOJOGP_CONTAINER_IMAGE_DIGEST") or None,
        image_id=os.environ.get("MOJOGP_CONTAINER_IMAGE_ID") or None,
        gpu_name=os.environ.get("MOJOGP_GPU_NAME") or None,
        gpu_total_vram_mb=None if gpu_total_vram_mb_raw in (None, "") else float(gpu_total_vram_mb_raw),
        gpu_driver_version=os.environ.get("MOJOGP_GPU_DRIVER_VERSION") or None,
        cuda_version=os.environ.get("MOJOGP_CUDA_VERSION") or None,
        gpu_target=os.environ.get("MOJOGP_GPU_TARGET") or None,
    )


def profiling_config_is_disabled(*, project_root: Path = PROJECT_ROOT) -> bool:
    target = project_root / "mojogp" / "kernels" / "profiling_config.mojo"
    text = target.read_text(encoding="utf-8")
    return "alias PROFILING = False" in text


def _run_profiling_probe(*, cwd: Path, jit_cache_dir: Path) -> bool:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["MOJOGP_JIT_CACHE_DIR"] = str(jit_cache_dir)
    completed = subprocess.run(
        [sys.executable, "-m", "tests.benchmarks.profiling_probe"],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=600,
    )
    if completed.returncode != 0:
        raise BenchmarkPreflightError(
            "Benchmark profiling probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    combined = f"{completed.stdout}\n{completed.stderr}"
    if "@ " in combined or "ProfileBlock" in combined:
        return False
    return True


def run_profiling_preflight(*, cwd: Path = PROJECT_ROOT) -> ProfilingMetadata:
    config_false = profiling_config_is_disabled(project_root=cwd)
    jit_cache_dir = Path(tempfile.mkdtemp(prefix="mojogp_benchmark_jit_cache_"))
    probe_passed = False
    if config_false:
        probe_passed = _run_profiling_probe(cwd=cwd, jit_cache_dir=jit_cache_dir)
    return ProfilingMetadata(
        profiling_config_false=config_false,
        profiling_probe_passed=probe_passed,
        jit_cache_dir=str(jit_cache_dir),
    )


def require_benchmark_preflight(*, cwd: Path = PROJECT_ROOT) -> tuple[GitMetadata, ProfilingMetadata]:
    git = get_benchmark_git_metadata(cwd=cwd)
    profiling = run_profiling_preflight(cwd=cwd)
    if not profiling.profiling_config_false:
        raise BenchmarkPreflightError("Benchmark runs require mojogp/kernels/profiling_config.mojo to set alias PROFILING = False.")
    if not profiling.profiling_probe_passed:
        raise BenchmarkPreflightError("Benchmark profiling preflight detected active Mojo profiling output.")
    return git, profiling


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
