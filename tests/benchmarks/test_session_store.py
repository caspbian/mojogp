from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from tests.benchmarks.report_queries import scaling_rows
from tests.benchmarks.session_store import BenchmarkSessionRecord
from tests.benchmarks.preflight import utc_now_iso


def test_session_store_initializes_schema(benchmark_store, tmp_path: Path):
    record = BenchmarkSessionRecord(
        session_id="session-1",
        started_at=utc_now_iso(),
        finished_at=None,
        branch_name="feature/test",
        commit_hash="abc123def456",
        commit_hash_short="abc123de",
        git_clean=True,
        worktree_path="/tmp/worktree",
        profiling_config_false=True,
        profiling_probe_passed=True,
        jit_cache_dir="/tmp/jit-cache",
        gpu_name="NVIDIA GeForce RTX 4050",
        gpu_total_vram_mb=6141.0,
        gpu_driver_version="580.126.09",
        cuda_version="13.0",
        gpu_target="sm_89",
    )
    benchmark_store.upsert_session(record)
    benchmark_store.register_case(
        case_id="mojogp.single_output.scaling.materialized.exact.n5000.d5",
        benchmark_group_id="mojogp.single_output.scaling.materialized.exact",
        framework="mojogp",
        suite_name="single_output_scaling",
        benchmark_name="scaling_certification",
        config={"n": 5000, "d": 5},
    )
    benchmark_store.register_run(
        {
            "run_id": "run-1",
            "session_id": "session-1",
            "case_id": "mojogp.single_output.scaling.materialized.exact.n5000.d5",
            "benchmark_group_id": "mojogp.single_output.scaling.materialized.exact",
            "n": 5000,
            "d": 5,
            "num_tasks": None,
            "kernel": "rbf",
            "model_type": "SingleOutputGP",
            "training_method": "materialized",
            "prediction_mode": "exact",
            "suite_name": "single_output_scaling",
            "comparison_class": None,
            "baseline_backend": None,
            "fairness_note": None,
            "dataset_id": "dataset-1",
            "comparison_id": None,
            "artifact_id": "artifact-1",
            "status": "ok",
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "framework": "mojogp",
            "config_json": json.dumps(
                {
                    "prediction_method": "matrix_free",
                    "prediction_cache_method": "matrix_free",
                    "actual_prediction_route": "predict",
                    "actual_variance_route": "predict",
                    "prediction_train_train_materialization_label": "none",
                    "prediction_train_train_materialized": False,
                    "memory_contract": "O(n_train * rank + n_test * rank)",
                    "matrix_free_prediction_verified": True,
                }
            ),
            "benchmark_name": "scaling_certification",
            "result_json_path": None,
            "training_time_s": 1.0,
            "prediction_mean_time_s": 0.1,
            "prediction_variance_time_s": 0.2,
            "prediction_cold_first_time_s": 0.35,
            "prediction_cache_prepare_time_s": 0.04,
            "prediction_prepared_apply_time_s": 0.12,
            "prediction_repeated_median_time_s": 0.11,
            "prediction_repeated_p5_time_s": 0.1,
            "prediction_repeated_p95_time_s": 0.13,
            "prediction_alpha_time_s": 0.05,
            "prediction_love_root_time_s": 0.07,
            "prediction_x_test_scaling": [
                {
                    "n_test": 1000,
                    "size_role": "core",
                    "comparison_class": "fair_match",
                    "timing_quality": "prepared_cache_split",
                    "cache_used": True,
                    "first_apply_time_s": 0.04,
                    "repeated_median_time_s": 0.035,
                    "repeated_p5_time_s": 0.033,
                    "repeated_p95_time_s": 0.038,
                    "repeat_count": 3,
                    "prediction_peak_gpu_mb": 118.0,
                    "prediction_delta_gpu_mb": 18.0,
                    "mean_time_s": 0.01,
                    "variance_time_s": 0.025,
                },
                {
                    "n_test": 100000,
                    "size_role": "envelope",
                    "comparison_class": "mojogp_only_prediction_envelope",
                    "status": "failed_oom",
                    "failure_reason": "oom",
                    "failure_stage": "prediction_apply",
                    "error_type": "RuntimeError",
                    "error_message": "CUDA out of memory",
                    "timing_quality": "failed_prediction_envelope",
                    "cache_used": True,
                    "first_apply_time_s": None,
                    "repeated_median_time_s": None,
                    "repeated_p5_time_s": None,
                    "repeated_p95_time_s": None,
                    "repeat_count": 0,
                    "prediction_peak_gpu_mb": None,
                    "prediction_delta_gpu_mb": None,
                    "mean_time_s": None,
                    "variance_time_s": None,
                }
            ],
            "end_to_end_time_s": 1.3,
            "iterations_run": 10,
            "max_iterations": 12,
            "early_stopped": 1,
            "ms_per_iteration": 100.0,
            "iter_time_min_ms": 95.0,
            "iter_time_q25_ms": 98.0,
            "iter_time_mean_ms": 100.5,
            "iter_time_median_ms": 100.0,
            "iter_time_q75_ms": 103.0,
            "iter_time_max_ms": 108.0,
            "iter_time_p5_ms": 96.0,
            "iter_time_p95_ms": 107.0,
            "startup_compile_time_s": 0.4,
            "startup_warm_cache_hit_s": 0.06,
            "startup_prepare_time_s": 0.02,
            "training_peak_gpu_mb": 150.0,
            "training_delta_gpu_mb": 50.0,
            "prediction_peak_gpu_mb": 120.0,
            "prediction_delta_gpu_mb": 20.0,
            "exact_prediction_peak_gpu_mb": 125.0,
            "exact_prediction_delta_gpu_mb": 25.0,
            "love_prediction_peak_gpu_mb": 118.0,
            "love_prediction_delta_gpu_mb": 18.0,
            "scaling_peak_gpu_mb": 50.0,
            "scaling_memory_metric": "training_delta_gpu_mb",
            "gpu_baseline_mb": 100.0,
            "gpu_current_mb": 120.0,
            "gpu_delta_mb": 20.0,
            "gpu_max_mb": 150.0,
            "gpu_isolated_peak_mb": 32.0,
            "gpu_isolated_current_mb": 10.0,
            "gpu_samples": 5,
            "measurement_method_primary": "torch.cuda",
            "torch_baseline_mb": 80.0,
            "torch_peak_mb": 112.0,
            "torch_peak_delta_mb": 32.0,
            "torch_current_delta_mb": 8.0,
            "torch_reserved_delta_mb": 16.0,
            "cpu_peak_mb": 64.0,
            "branch_name": "feature/test",
            "commit_hash": "abc123def456",
            "git_clean": 1,
            "profiling_probe_passed": 1,
            "gpu_name": "NVIDIA GeForce RTX 4050",
            "gpu_total_vram_mb": 6141.0,
            "gpu_driver_version": "580.126.09",
            "cuda_version": "13.0",
            "gpu_target": "sm_89",
        }
    )
    benchmark_store.register_artifact(
        artifact_id="artifact-1",
        run_id="run-1",
        artifact_type="benchmark_run",
        path="/tmp/artifact-1.json",
        sha256="deadbeef",
        created_at=utc_now_iso(),
    )

    with benchmark_store._connect() as conn:
        row = conn.execute(
            "SELECT benchmark_group_id, suite_name, gpu_isolated_peak_mb, exact_prediction_delta_gpu_mb, love_prediction_delta_gpu_mb, "
            "iter_time_median_ms, startup_compile_time_s, prediction_prepared_apply_time_s, prediction_alpha_time_s, "
            "prediction_x_test_scaling_json, rmse, container_runtime, gpu_name, cuda_version "
            "FROM benchmark_runs WHERE run_id='run-1'"
        ).fetchone()
        assert row["benchmark_group_id"] == "mojogp.single_output.scaling.materialized.exact"
        assert row["suite_name"] == "single_output_scaling"
        assert float(row["gpu_isolated_peak_mb"]) == 32.0
        assert float(row["exact_prediction_delta_gpu_mb"]) == 25.0
        assert float(row["love_prediction_delta_gpu_mb"]) == 18.0
        assert float(row["iter_time_median_ms"]) == 100.0
        assert float(row["startup_compile_time_s"]) == 0.4
        assert float(row["prediction_prepared_apply_time_s"]) == 0.12
        assert float(row["prediction_alpha_time_s"]) == 0.05
        assert '"n_test": 1000' in row["prediction_x_test_scaling_json"]
        assert '"status": "failed_oom"' in row["prediction_x_test_scaling_json"]
        assert row["rmse"] is None
        assert row["container_runtime"] is None
        assert row["gpu_name"] == "NVIDIA GeForce RTX 4050"
        assert row["cuda_version"] == "13.0"

        enriched = conn.execute(
            "SELECT prediction_total_time_s, prediction_apply_time_s, prediction_cache_plus_apply_time_s, "
            "source_commit_hash, provenance_kind, canonical_suite_name, "
            "training_peak_gpu_mb, exact_prediction_delta_gpu_mb, love_prediction_delta_gpu_mb, "
            "prediction_method, prediction_cache_method, actual_prediction_route, "
            "prediction_train_train_materialized, prediction_memory_contract, "
            "matrix_free_prediction_verified "
            "FROM v_benchmark_run_enriched WHERE run_id='run-1'"
        ).fetchone()
        assert enriched is not None
        assert abs(float(enriched["prediction_total_time_s"]) - 0.3) < 1e-9
        assert abs(float(enriched["prediction_apply_time_s"]) - 0.12) < 1e-9
        assert abs(float(enriched["prediction_cache_plus_apply_time_s"]) - 0.16) < 1e-9
        assert enriched["source_commit_hash"] == "abc123def456"
        assert enriched["provenance_kind"] == "host"
        assert enriched["canonical_suite_name"] == "single_output_scaling"
        assert float(enriched["training_peak_gpu_mb"]) == 150.0
        assert float(enriched["exact_prediction_delta_gpu_mb"]) == 25.0
        assert float(enriched["love_prediction_delta_gpu_mb"]) == 18.0
        assert enriched["prediction_method"] == "matrix_free"
        assert enriched["prediction_cache_method"] == "matrix_free"
        assert enriched["actual_prediction_route"] == "predict"
        assert enriched["prediction_train_train_materialized"] == 0
        assert enriched["prediction_memory_contract"] == "O(n_train * rank + n_test * rank)"
        assert enriched["matrix_free_prediction_verified"] == 1

        prediction_scaling = conn.execute(
            "SELECT * FROM benchmark_prediction_x_test_scaling "
            "WHERE run_id='run-1' AND n_test=1000"
        ).fetchone()
        assert prediction_scaling is not None
        assert prediction_scaling["n_test"] == 1000
        assert prediction_scaling["cache_used"] == 1
        assert float(prediction_scaling["repeated_median_time_s"]) == 0.035
        failed_scaling = conn.execute(
            "SELECT status, failure_reason, error_type, repeated_median_time_s "
            "FROM benchmark_prediction_x_test_scaling "
            "WHERE run_id='run-1' AND n_test=100000"
        ).fetchone()
        assert failed_scaling is not None
        assert failed_scaling["status"] == "failed_oom"
        assert failed_scaling["failure_reason"] == "oom"
        assert failed_scaling["error_type"] == "RuntimeError"
        assert failed_scaling["repeated_median_time_s"] is None

    runs = benchmark_store.fetch_runs_by_group("mojogp.single_output.scaling.materialized.exact")
    assert len(runs) == 1
    assert runs[0]["n"] == 5000
    assert runs[0]["d"] == 5
    queried_runs = scaling_rows(
        "mojogp.single_output.scaling.materialized.exact",
        db_path=benchmark_store.db_path,
    )
    assert len(queried_runs) == 1
    assert queried_runs[0]["run_id"] == "run-1"

    export_path = benchmark_store.export_session_json("session-1", export_root=tmp_path / "exports")
    assert export_path.exists()

    imported_store_path = tmp_path / "imported.sqlite"
    from tests.benchmarks.session_store import BenchmarkSessionStore

    imported_store = BenchmarkSessionStore(imported_store_path)
    counts = imported_store.import_session_json(export_path)
    assert counts["sessions"] == 1
    assert counts["cases"] == 1
    assert counts["runs"] == 1
    assert counts["artifacts"] == 1
    assert counts["runs_inserted"] == 1
    assert counts["duplicates_skipped"] == 0
    assert counts["datasets"] == 0
    assert counts["comparisons"] == 0
    second_counts = imported_store.import_session_bundle(export_path)
    assert second_counts["runs"] == 1
    assert second_counts["runs_updated"] == 1
    assert second_counts["artifacts_updated"] == 1
    assert second_counts["duplicates_skipped"] >= 3
    imported_session = imported_store.fetch_session("session-1")
    assert imported_session is not None
    assert imported_session["gpu_target"] == "sm_89"
    imported_run = imported_store.fetch_runs_for_session("session-1")[0]
    assert imported_run["suite_name"] == "single_output_scaling"
    with imported_store._connect() as conn:
        imported_scaling = conn.execute(
            "SELECT repeated_median_time_s FROM benchmark_prediction_x_test_scaling "
            "WHERE run_id='run-1' AND n_test=1000"
        ).fetchone()
        imported_failed_scaling = conn.execute(
            "SELECT status FROM benchmark_prediction_x_test_scaling "
            "WHERE run_id='run-1' AND n_test=100000"
        ).fetchone()
    assert imported_scaling is not None
    assert float(imported_scaling["repeated_median_time_s"]) == 0.035
    assert imported_failed_scaling is not None
    assert imported_failed_scaling["status"] == "failed_oom"


def test_session_store_refreshes_stale_existing_views(tmp_path: Path):
    db_path = tmp_path / "stale_views.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE benchmark_case_registry (
                case_id TEXT PRIMARY KEY,
                benchmark_group_id TEXT NOT NULL,
                framework TEXT NOT NULL,
                suite_name TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                config_json TEXT NOT NULL
            );

            CREATE TABLE benchmark_runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                benchmark_group_id TEXT NOT NULL,
                n INTEGER,
                d INTEGER,
                num_tasks INTEGER,
                kernel TEXT,
                model_type TEXT,
                training_method TEXT,
                prediction_mode TEXT,
                comparison_class TEXT,
                baseline_backend TEXT,
                fairness_note TEXT,
                dataset_id TEXT,
                comparison_id TEXT,
                artifact_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                framework TEXT NOT NULL,
                config_json TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                result_json_path TEXT,
                training_time_s REAL,
                prediction_mean_time_s REAL,
                prediction_variance_time_s REAL,
                end_to_end_time_s REAL,
                gpu_baseline_mb REAL,
                gpu_current_mb REAL,
                gpu_delta_mb REAL,
                gpu_max_mb REAL,
                gpu_isolated_peak_mb REAL,
                gpu_isolated_current_mb REAL,
                gpu_samples INTEGER,
                measurement_method_primary TEXT,
                torch_baseline_mb REAL,
                torch_peak_mb REAL,
                torch_peak_delta_mb REAL,
                torch_current_delta_mb REAL,
                torch_reserved_delta_mb REAL,
                cpu_peak_mb REAL,
                branch_name TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                git_clean INTEGER NOT NULL,
                profiling_probe_passed INTEGER NOT NULL,
                suite_name TEXT
            );

            CREATE TABLE benchmark_sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                branch_name TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                commit_hash_short TEXT NOT NULL,
                git_clean INTEGER NOT NULL,
                worktree_path TEXT NOT NULL,
                profiling_config_false INTEGER NOT NULL,
                profiling_probe_passed INTEGER NOT NULL,
                jit_cache_dir TEXT NOT NULL
            );

            CREATE TABLE benchmark_comparison_registry (
                comparison_id TEXT PRIMARY KEY,
                mojogp_case_id TEXT NOT NULL,
                gpytorch_case_id TEXT NOT NULL,
                comparison_class TEXT NOT NULL,
                fairness_note TEXT NOT NULL,
                fairness_axes_json TEXT NOT NULL
            );

            CREATE VIEW v_benchmark_run_enriched AS
            SELECT
                br.*,
                br.commit_hash AS source_commit_hash,
                CASE
                    WHEN br.commit_hash IS NULL THEN NULL
                    ELSE substr(br.commit_hash, 1, 8)
                END AS source_commit_hash_short,
                COALESCE(br.prediction_mean_time_s, 0.0) + COALESCE(br.prediction_variance_time_s, 0.0)
                    AS prediction_total_time_s,
                CAST(COALESCE(json_extract(br.config_json, '$.backend_fallback_used'), 0) AS INTEGER)
                    AS backend_fallback_used,
                json_extract(br.config_json, '$.backend_fallback_reason') AS backend_fallback_reason,
                json_extract(br.config_json, '$.effective_training_backend') AS effective_training_backend,
                json_extract(br.config_json, '$.effective_prediction_backend') AS effective_prediction_backend,
                json_extract(br.config_json, '$.requested_backend') AS requested_backend,
                json_extract(br.config_json, '$.exact_block_cols') AS exact_block_cols,
                json_extract(br.config_json, '$.exact_cross_mode') AS exact_cross_mode,
                CASE
                    WHEN br.container_image_digest IS NOT NULL OR br.container_image_id IS NOT NULL THEN 'container'
                    ELSE 'host'
                END AS provenance_kind
            FROM benchmark_runs br;
            """
        )

    from tests.benchmarks.session_store import BenchmarkSessionStore

    refreshed_store = BenchmarkSessionStore(db_path)
    with refreshed_store._connect() as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='v_benchmark_run_enriched'"
        ).fetchone()
    assert row is not None
    assert "canonical_suite_name" in row[0]
    assert "prediction_apply_time_s" in row[0]
    assert "prediction_cache_alpha_time_s" in row[0]
    assert "matrix_free_prediction_verified" in row[0]
