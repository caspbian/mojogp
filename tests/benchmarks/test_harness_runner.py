from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from tests.benchmarks.harness_runner import _derive_scaling_memory_metric, run_benchmark_module
from tests.benchmarks.preflight import GitMetadata, ProfilingMetadata, utc_now_iso
from tests.benchmarks.session_store import BenchmarkSessionRecord
from tests.shared.subprocess_harness import IsolatedSubprocessError


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _assert_mb_close(measured_mb: float, expected_mb: float, *, tolerance_mb: float) -> None:
    assert abs(measured_mb - expected_mb) <= tolerance_mb, (
        f"measured {measured_mb:.1f} MB vs expected {expected_mb:.1f} MB "
        f"(tolerance {tolerance_mb:.1f} MB)"
    )


def test_derive_scaling_memory_metric_prefers_largest_positive_route_delta():
    memory = SimpleNamespace(
        training_delta_gpu_mb=48.0,
        prediction_delta_gpu_mb=20.0,
        exact_prediction_delta_gpu_mb=96.0,
        love_prediction_delta_gpu_mb=None,
    )

    peak_mb, metric_name = _derive_scaling_memory_metric(memory, 256.0)

    assert peak_mb == 96.0
    assert metric_name == "exact_prediction_delta_gpu_mb"


def test_derive_scaling_memory_metric_falls_back_to_overall_process_peak():
    memory = SimpleNamespace(
        training_delta_gpu_mb=0.0,
        prediction_delta_gpu_mb=None,
        exact_prediction_delta_gpu_mb=-5.0,
        love_prediction_delta_gpu_mb=None,
    )

    peak_mb, metric_name = _derive_scaling_memory_metric(memory, 64.0)

    assert peak_mb == 64.0
    assert metric_name == "overall_process_peak"


def test_benchmark_wrapper_exports_failed_child_envelope(
    benchmark_store,
    tmp_path,
    monkeypatch,
):
    export_root = tmp_path / "exports"
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        "tests.benchmarks.session_store.DEFAULT_SESSION_EXPORT_ROOT",
        export_root,
    )

    def fake_save_json_artifact(artifact_id, payload, *, artifact_type="benchmark_run"):
        target = artifact_root / artifact_type / f"{artifact_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)
        target.write_text(serialized, encoding="utf-8")
        return target, "sha256-fixture", serialized

    def fake_run_isolated_module(**kwargs):
        envelope = {
            "status": "error",
            "payload": None,
            "result_path": None,
            "telemetry": {
                "memory": {
                    "gpu": {
                        "baseline_mb": 100.0,
                        "max_mb": 40960.0,
                        "isolated_peak_mb": 40800.0,
                        "method": "pynvml",
                    }
                }
            },
            "error": {
                "type": "Exception",
                "message": "CUDA_ERROR_OUT_OF_MEMORY",
            },
        }
        raise IsolatedSubprocessError(
            message="child failed",
            command=["python", "-m", "fixture"],
            cwd="/tmp",
            timeout=10,
            returncode=1,
            stdout="",
            stderr="",
            envelope=envelope,
        )

    monkeypatch.setattr(
        "tests.benchmarks.harness_runner.save_json_artifact",
        fake_save_json_artifact,
    )
    monkeypatch.setattr(
        "tests.benchmarks.harness_runner.run_isolated_module",
        fake_run_isolated_module,
    )
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
            session_id="session-failed",
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
    benchmark_store.register_case(
        case_id="fixture.failed.case.n35000.d5",
        benchmark_group_id="fixture.failed.case",
        framework="mojogp",
        suite_name="scaling_certification",
        benchmark_name="scaling_certification",
        config={"n": 35_000, "d": 5},
    )

    with pytest.raises(IsolatedSubprocessError):
        run_benchmark_module(
            module="tests.benchmarks.benchmark_fixture",
            payload={"n_train": 35_000, "d": 5},
            timeout=10,
            description="Runs failing fixture child",
            result_loader=None,
            session_store=benchmark_store,
            session_id="session-failed",
            case_id="fixture.failed.case.n35000.d5",
            benchmark_group_id="fixture.failed.case",
            benchmark_name="scaling_certification",
            framework="mojogp",
            git=git,
            profiling=profiling,
            config={
                "n": 35_000,
                "d": 5,
                "suite_name": "scaling_certification",
                "training_method": "materialized",
                "prediction_mode": "exact",
            },
        )

    export_path = export_root / "session-failed.json"
    assert export_path.exists()
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(exported["runs"]) == 1
    assert exported["runs"][0]["status"] == "error"
    assert exported["runs"][0]["n"] == 35_000
    assert exported["artifacts"]
    assert exported["artifacts"][0]["artifact_type"] == "benchmark_run"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_benchmark_wrapper_persists_isolated_peak_gpu_metric(benchmark_store, monkeypatch):
    inject_mb = 128.0
    monkeypatch.setenv("MOJOGP_GPU_NAME", "NVIDIA GeForce RTX 4050")
    monkeypatch.setenv("MOJOGP_GPU_TOTAL_VRAM_MB", "6141")
    monkeypatch.setenv("MOJOGP_GPU_DRIVER_VERSION", "580.126.09")
    monkeypatch.setenv("MOJOGP_CUDA_VERSION", "13.0")
    monkeypatch.setenv("MOJOGP_GPU_TARGET", "sm_89")

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
            session_id="session-1",
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
            container_runtime="docker",
            container_image_tag="mojogp-benchmark:test",
            container_image_id="sha256:test",
        )
    )
    benchmark_store.register_case(
        case_id="fixture.case.n1.d1",
        benchmark_group_id="fixture.case",
        framework="fixture",
        suite_name="fixture_suite",
        benchmark_name="fixture_benchmark",
        config={"n": 1, "d": 1},
    )

    result = run_benchmark_module(
        module="tests.benchmarks.benchmark_fixture",
        payload={"inject_mb": inject_mb, "hold_s": 0.35, "run_id": "fixture-run"},
        timeout=90,
        description="Runs benchmark fixture child",
        result_loader=None,
        session_store=benchmark_store,
        session_id="session-1",
        case_id="fixture.case.n1.d1",
        benchmark_group_id="fixture.case",
        benchmark_name="fixture_benchmark",
        framework="fixture",
        git=git,
        profiling=profiling,
        config={"n": 1, "d": 1},
    )

    gpu_memory = result.envelope["telemetry"]["memory"]["gpu"]
    assert float(gpu_memory["isolated_peak_mb"]) >= inject_mb

    with benchmark_store._connect() as conn:
        row = conn.execute(
            "SELECT gpu_isolated_peak_mb, benchmark_group_id, suite_name, measurement_method_primary, "
            "iterations_run, iter_time_median_ms, startup_compile_time_s, gpu_name, gpu_target "
            "FROM benchmark_runs WHERE run_id='fixture-run'"
        ).fetchone()
        assert row is not None
        assert abs(float(row["gpu_isolated_peak_mb"]) - float(gpu_memory["isolated_peak_mb"])) <= 1e-6
        assert row["benchmark_group_id"] == "fixture.case"
        assert row["suite_name"] == "fixture_suite"
        assert row["measurement_method_primary"] is not None
        assert row["iterations_run"] == 4
        assert float(row["iter_time_median_ms"]) == 62.0
        assert float(row["startup_compile_time_s"]) == 0.3
        assert row["gpu_name"] == "NVIDIA GeForce RTX 4050"
        assert row["gpu_target"] == "sm_89"


@pytest.mark.skipif(not _cuda_available(), reason="CUDA GPU required")
def test_benchmark_wrapper_flattens_nested_serialized_benchmark_payload(benchmark_store):
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
            session_id="session-nested",
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
            container_runtime="docker",
            container_image_tag="mojogp-benchmark:test",
            container_image_id="sha256:test",
        )
    )
    benchmark_store.register_case(
        case_id="fixture.nested.case.n150.d3",
        benchmark_group_id="fixture.nested.case",
        framework="mojogp",
        suite_name="fixture_suite",
        benchmark_name="fixture_benchmark",
        config={"n": 150, "d": 3},
    )

    run_benchmark_module(
        module="tests.benchmarks.benchmark_fixture",
        payload={"case": "nested_benchmark", "inject_mb": 32.0, "hold_s": 0.25, "run_id": "nested-run"},
        timeout=90,
        description="Runs nested benchmark fixture child",
        result_loader=None,
        session_store=benchmark_store,
        session_id="session-nested",
        case_id="fixture.nested.case.n150.d3",
        benchmark_group_id="fixture.nested.case",
        benchmark_name="fixture_benchmark",
        framework="mojogp",
        git=git,
        profiling=profiling,
        config={"n": 150, "d": 3},
    )

    with benchmark_store._connect() as conn:
        row = conn.execute(
            "SELECT n, d, model_type, training_method, prediction_mode, fairness_note, iter_time_q75_ms, startup_prepare_time_s "
            "FROM benchmark_runs WHERE run_id='nested-run'"
        ).fetchone()
        assert row is not None
        assert row["n"] == 150
        assert row["d"] == 3
        assert row["model_type"] == "MultiOutputGP"
        assert row["training_method"] == "materialized"
        assert row["prediction_mode"] == "love"
        assert row["fairness_note"] == "fixture nested benchmark"
        assert float(row["iter_time_q75_ms"]) == 123.0
        assert float(row["startup_prepare_time_s"]) == 0.03
