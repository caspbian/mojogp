"""SQLite-backed benchmark session and run persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import DEFAULT_DB_PATH, DEFAULT_SESSION_EXPORT_ROOT, ensure_storage_dirs
from .specialization_columns import extract_specialization_columns


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _prediction_scaling_rows(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _parameter_recovery_rows(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = value.get("records", [])
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


@dataclass(frozen=True)
class BenchmarkSessionRecord:
    session_id: str
    started_at: str
    finished_at: str | None
    branch_name: str
    commit_hash: str
    commit_hash_short: str
    git_clean: bool
    worktree_path: str
    profiling_config_false: bool
    profiling_probe_passed: bool
    jit_cache_dir: str
    container_runtime: str | None = None
    container_image_tag: str | None = None
    container_image_digest: str | None = None
    container_image_id: str | None = None
    gpu_name: str | None = None
    gpu_total_vram_mb: float | None = None
    gpu_driver_version: str | None = None
    cuda_version: str | None = None
    gpu_target: str | None = None


class BenchmarkSessionStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        ensure_storage_dirs()
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_sessions (
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
                    jit_cache_dir TEXT NOT NULL,
                    container_runtime TEXT,
                    container_image_tag TEXT,
                    container_image_digest TEXT,
                    container_image_id TEXT,
                    gpu_name TEXT,
                    gpu_total_vram_mb REAL,
                    gpu_driver_version TEXT,
                    cuda_version TEXT,
                    gpu_target TEXT
                );

                CREATE TABLE IF NOT EXISTS benchmark_datasets (
                    dataset_id TEXT PRIMARY KEY,
                    generator_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    seed INTEGER,
                    artifact_path TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_case_registry (
                    case_id TEXT PRIMARY KEY,
                    benchmark_group_id TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    config_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_specialization_registry (
                    specialization_key TEXT PRIMARY KEY,
                    specialization_family TEXT NOT NULL,
                    specialization_source TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    active INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_specialization_studies (
                    study_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    benchmark_group_id TEXT NOT NULL,
                    objective_name TEXT NOT NULL,
                    objective_metric TEXT NOT NULL,
                    constraint_json TEXT NOT NULL,
                    search_space_json TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS benchmark_specialization_trials (
                    trial_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    base_case_id TEXT NOT NULL,
                    specialization_key TEXT NOT NULL,
                    run_id TEXT,
                    status TEXT NOT NULL,
                    objective_value REAL,
                    constraint_status TEXT,
                    trial_config_json TEXT NOT NULL,
                    result_summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_comparison_registry (
                    comparison_id TEXT PRIMARY KEY,
                    mojogp_case_id TEXT NOT NULL,
                    gpytorch_case_id TEXT NOT NULL,
                    comparison_class TEXT NOT NULL,
                    fairness_note TEXT NOT NULL,
                    fairness_axes_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_runs (
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
                    suite_name TEXT,
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
                    prediction_cold_first_time_s REAL,
                    prediction_cache_prepare_time_s REAL,
                    prediction_prepared_apply_time_s REAL,
                    prediction_repeated_median_time_s REAL,
                    prediction_repeated_p5_time_s REAL,
                    prediction_repeated_p95_time_s REAL,
                    prediction_alpha_time_s REAL,
                    prediction_love_root_time_s REAL,
                    prediction_x_test_scaling_json TEXT,
                    end_to_end_time_s REAL,
                    iterations_run INTEGER,
                    max_iterations INTEGER,
                    early_stopped INTEGER,
                    ms_per_iteration REAL,
                    iter_time_min_ms REAL,
                    iter_time_q25_ms REAL,
                    iter_time_mean_ms REAL,
                    iter_time_median_ms REAL,
                    iter_time_q75_ms REAL,
                    iter_time_max_ms REAL,
                    iter_time_p5_ms REAL,
                    iter_time_p95_ms REAL,
                    startup_compile_time_s REAL,
                    startup_warm_cache_hit_s REAL,
                    startup_prepare_time_s REAL,
                    contract_passed INTEGER,
                    contract_summary_json TEXT,
                    training_peak_gpu_mb REAL,
                    training_delta_gpu_mb REAL,
                    prediction_peak_gpu_mb REAL,
                    prediction_delta_gpu_mb REAL,
                    exact_prediction_peak_gpu_mb REAL,
                    exact_prediction_delta_gpu_mb REAL,
                    love_prediction_peak_gpu_mb REAL,
                    love_prediction_delta_gpu_mb REAL,
                    scaling_peak_gpu_mb REAL,
                    scaling_memory_metric TEXT,
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
                    rmse REAL,
                    mae REAL,
                    r_squared REAL,
                    crps REAL,
                    msll REAL,
                    calibration_error REAL,
                    sharpness REAL,
                    interval_width_95 REAL,
                    final_nll REAL,
                    branch_name TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    git_clean INTEGER NOT NULL,
                    profiling_probe_passed INTEGER NOT NULL,
                    container_runtime TEXT,
                    container_image_tag TEXT,
                    container_image_digest TEXT,
                    container_image_id TEXT,
                    gpu_name TEXT,
                    gpu_total_vram_mb REAL,
                    gpu_driver_version TEXT,
                    cuda_version TEXT,
                    gpu_target TEXT
                );

                CREATE TABLE IF NOT EXISTS benchmark_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS benchmark_prediction_x_test_scaling (
                    run_id TEXT NOT NULL,
                    n_test INTEGER NOT NULL,
                    size_role TEXT,
                    comparison_class TEXT,
                    status TEXT,
                    failure_reason TEXT,
                    failure_stage TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    timing_quality TEXT,
                    cache_used INTEGER,
                    first_apply_time_s REAL,
                    repeated_median_time_s REAL,
                    repeated_p5_time_s REAL,
                    repeated_p95_time_s REAL,
                    repeat_count INTEGER,
                    prediction_peak_gpu_mb REAL,
                    prediction_delta_gpu_mb REAL,
                    mean_time_s REAL,
                    variance_time_s REAL,
                    PRIMARY KEY (run_id, n_test, size_role)
                );

                CREATE TABLE IF NOT EXISTS benchmark_parameter_recovery (
                    run_id TEXT NOT NULL,
                    parameter_name TEXT NOT NULL,
                    parameter_group TEXT,
                    parameter_index TEXT NOT NULL DEFAULT '',
                    truth_json TEXT,
                    learned_json TEXT,
                    abs_error REAL,
                    rel_error REAL,
                    signed_error REAL,
                    log_abs_error REAL,
                    truth_norm REAL,
                    learned_norm REAL,
                    status TEXT NOT NULL,
                    notes TEXT,
                    metadata_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, parameter_name, parameter_index)
                );
                """
            )
            self._ensure_columns(
                conn,
                "benchmark_sessions",
                {
                    "container_runtime": "TEXT",
                    "container_image_tag": "TEXT",
                    "container_image_digest": "TEXT",
                    "container_image_id": "TEXT",
                    "gpu_name": "TEXT",
                    "gpu_total_vram_mb": "REAL",
                    "gpu_driver_version": "TEXT",
                    "cuda_version": "TEXT",
                    "gpu_target": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "benchmark_case_registry",
                {
                    "benchmark_group_id": "TEXT NOT NULL DEFAULT ''",
                    "base_case_id": "TEXT",
                    "specialization_key": "TEXT",
                    "specialization_family": "TEXT",
                    "specialization_mode": "TEXT",
                    "specialization_source": "TEXT",
                    "specialization_descriptor_json": "TEXT",
                    "specialization_config_json": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "benchmark_runs",
                {
                    "benchmark_group_id": "TEXT NOT NULL DEFAULT ''",
                    "n": "INTEGER",
                    "d": "INTEGER",
                    "num_tasks": "INTEGER",
                    "kernel": "TEXT",
                    "model_type": "TEXT",
                    "training_method": "TEXT",
                    "prediction_mode": "TEXT",
                    "suite_name": "TEXT",
                    "comparison_class": "TEXT",
                    "baseline_backend": "TEXT",
                    "fairness_note": "TEXT",
                    "base_case_id": "TEXT",
                    "specialization_key": "TEXT",
                    "specialization_family": "TEXT",
                    "specialization_mode": "TEXT",
                    "specialization_source": "TEXT",
                    "specialization_descriptor_json": "TEXT",
                    "specialization_config_json": "TEXT",
                    "study_id": "TEXT",
                    "trial_id": "TEXT",
                    "objective_name": "TEXT",
                    "objective_metric": "TEXT",
                    "constraint_json": "TEXT",
                    "iterations_run": "INTEGER",
                    "max_iterations": "INTEGER",
                    "early_stopped": "INTEGER",
                    "ms_per_iteration": "REAL",
                    "iter_time_min_ms": "REAL",
                    "iter_time_q25_ms": "REAL",
                    "iter_time_mean_ms": "REAL",
                    "iter_time_median_ms": "REAL",
                    "iter_time_q75_ms": "REAL",
                    "iter_time_max_ms": "REAL",
                    "iter_time_p5_ms": "REAL",
                    "iter_time_p95_ms": "REAL",
                    "startup_compile_time_s": "REAL",
                    "startup_warm_cache_hit_s": "REAL",
                    "startup_prepare_time_s": "REAL",
                    "prediction_cold_first_time_s": "REAL",
                    "prediction_cache_prepare_time_s": "REAL",
                    "prediction_prepared_apply_time_s": "REAL",
                    "prediction_repeated_median_time_s": "REAL",
                    "prediction_repeated_p5_time_s": "REAL",
                    "prediction_repeated_p95_time_s": "REAL",
                    "prediction_alpha_time_s": "REAL",
                    "prediction_love_root_time_s": "REAL",
                    "prediction_x_test_scaling_json": "TEXT",
                    "contract_passed": "INTEGER",
                    "contract_summary_json": "TEXT",
                    "training_peak_gpu_mb": "REAL",
                    "training_delta_gpu_mb": "REAL",
                    "prediction_peak_gpu_mb": "REAL",
                    "prediction_delta_gpu_mb": "REAL",
                    "exact_prediction_peak_gpu_mb": "REAL",
                    "exact_prediction_delta_gpu_mb": "REAL",
                    "love_prediction_peak_gpu_mb": "REAL",
                    "love_prediction_delta_gpu_mb": "REAL",
                    "scaling_peak_gpu_mb": "REAL",
                    "scaling_memory_metric": "TEXT",
                    "rmse": "REAL",
                    "mae": "REAL",
                    "r_squared": "REAL",
                    "crps": "REAL",
                    "msll": "REAL",
                    "calibration_error": "REAL",
                    "sharpness": "REAL",
                    "interval_width_95": "REAL",
                    "final_nll": "REAL",
                    "container_runtime": "TEXT",
                    "container_image_tag": "TEXT",
                    "container_image_digest": "TEXT",
                    "container_image_id": "TEXT",
                    "gpu_name": "TEXT",
                    "gpu_total_vram_mb": "REAL",
                    "gpu_driver_version": "TEXT",
                    "cuda_version": "TEXT",
                    "gpu_target": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "benchmark_prediction_x_test_scaling",
                {
                    "status": "TEXT",
                    "failure_reason": "TEXT",
                    "failure_stage": "TEXT",
                    "error_type": "TEXT",
                    "error_message": "TEXT",
                },
            )
            self._ensure_columns(
                conn,
                "benchmark_parameter_recovery",
                {
                    "parameter_group": "TEXT",
                    "truth_json": "TEXT",
                    "learned_json": "TEXT",
                    "abs_error": "REAL",
                    "rel_error": "REAL",
                    "signed_error": "REAL",
                    "log_abs_error": "REAL",
                    "truth_norm": "REAL",
                    "learned_norm": "REAL",
                    "notes": "TEXT",
                    "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                },
            )
            self._ensure_columns(
                conn,
                "benchmark_comparison_registry",
                {
                    "mojogp_specialization_key": "TEXT",
                    "specialization_family": "TEXT",
                    "specialization_config_json": "TEXT",
                },
            )
            self._create_views(conn)

    @staticmethod
    def _create_views(conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP VIEW IF EXISTS v_benchmark_scaling_steps;
            DROP VIEW IF EXISTS v_benchmark_pairwise_compare;
            DROP VIEW IF EXISTS v_benchmark_fallback_audit;
            DROP VIEW IF EXISTS v_benchmark_session_summary;
            DROP VIEW IF EXISTS v_benchmark_run_enriched;
            DROP VIEW IF EXISTS v_benchmark_parameter_recovery;

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
                COALESCE(
                    br.prediction_prepared_apply_time_s,
                    COALESCE(br.prediction_mean_time_s, 0.0) + COALESCE(br.prediction_variance_time_s, 0.0)
                ) AS prediction_apply_time_s,
                COALESCE(br.prediction_cache_prepare_time_s, 0.0) + COALESCE(
                    br.prediction_prepared_apply_time_s,
                    COALESCE(br.prediction_mean_time_s, 0.0) + COALESCE(br.prediction_variance_time_s, 0.0)
                ) AS prediction_cache_plus_apply_time_s,
                CAST(COALESCE(json_extract(br.config_json, '$.backend_fallback_used'), 0) AS INTEGER)
                    AS backend_fallback_used,
                json_extract(br.config_json, '$.backend_fallback_reason') AS backend_fallback_reason,
                json_extract(br.config_json, '$.effective_training_backend') AS effective_training_backend,
                json_extract(br.config_json, '$.effective_prediction_backend') AS effective_prediction_backend,
                json_extract(br.config_json, '$.requested_backend') AS requested_backend,
                json_extract(br.config_json, '$.prediction_cache_alpha_time_s') AS prediction_cache_alpha_time_s,
                json_extract(br.config_json, '$.prediction_cache_love_root_time_s') AS prediction_cache_love_root_time_s,
                json_extract(br.config_json, '$.prediction_method') AS prediction_method,
                json_extract(br.config_json, '$.prediction_cache_method') AS prediction_cache_method,
                json_extract(br.config_json, '$.actual_prediction_route') AS actual_prediction_route,
                json_extract(br.config_json, '$.actual_variance_route') AS actual_variance_route,
                json_extract(br.config_json, '$.prediction_train_train_materialization_label')
                    AS prediction_train_train_materialization_label,
                CAST(COALESCE(json_extract(br.config_json, '$.prediction_train_train_materialized'), 0) AS INTEGER)
                    AS prediction_train_train_materialized,
                json_extract(br.config_json, '$.memory_contract') AS prediction_memory_contract,
                CAST(COALESCE(json_extract(br.config_json, '$.matrix_free_prediction_verified'), 0) AS INTEGER)
                    AS matrix_free_prediction_verified,
                json_extract(br.config_json, '$.exact_block_cols') AS exact_block_cols,
                json_extract(br.config_json, '$.exact_cross_mode') AS exact_cross_mode,
                json_extract(br.config_json, '$.prediction_timing_quality') AS prediction_timing_quality,
                CAST(COALESCE(json_extract(br.config_json, '$.ard'), 0) AS INTEGER) AS ard,
                json_extract(br.config_json, '$.relevant_dims') AS relevant_dims,
                json_extract(br.config_json, '$.mean_relevant_lengthscale') AS mean_relevant_lengthscale,
                json_extract(br.config_json, '$.mean_irrelevant_lengthscale') AS mean_irrelevant_lengthscale,
                json_extract(br.config_json, '$.relevance_separation_ratio') AS relevance_separation_ratio,
                json_extract(br.config_json, '$.pairwise_relevance_accuracy') AS pairwise_relevance_accuracy,
                json_extract(br.config_json, '$.top_k_relevance_hit_rate') AS top_k_relevance_hit_rate,
                json_extract(br.config_json, '$.ard_quality_status') AS ard_quality_status,
                COALESCE(
                    json_extract(br.config_json, '$.iter_timing_quality'),
                    json_extract(br.config_json, '$.speed.iter_timing_quality')
                ) AS iter_timing_quality,
                json_extract(br.config_json, '$.cg_telemetry.training.cg_iterations_mean') AS training_cg_mean,
                json_extract(br.config_json, '$.cg_telemetry.training.cg_iterations_final_step') AS training_cg_final,
                json_extract(br.config_json, '$.cg_telemetry.training.cg_iterations_max') AS training_cg_max,
                json_extract(br.config_json, '$.cg_telemetry.prediction.cg_iterations_mean') AS prediction_cg_mean,
                json_extract(br.config_json, '$.cg_telemetry.prediction.cg_iterations_final_step') AS prediction_cg_final,
                json_extract(br.config_json, '$.cg_telemetry.prediction.cg_iterations_max') AS prediction_cg_max,
                json_extract(br.config_json, '$.cg_telemetry.prediction.solve_count') AS prediction_cg_solve_count,
                json_extract(br.config_json, '$.cg_telemetry.prediction.telemetry_quality') AS prediction_cg_quality,
                COALESCE(br.suite_name, json_extract(bcr.config_json, '$.suite_name'), bcr.suite_name) AS canonical_suite_name,
                CASE
                    WHEN br.container_image_digest IS NOT NULL OR br.container_image_id IS NOT NULL THEN 'container'
                    ELSE 'host'
                END AS provenance_kind
            FROM benchmark_runs br
            LEFT JOIN benchmark_case_registry bcr ON bcr.case_id = br.case_id;

            CREATE VIEW v_benchmark_session_summary AS
            SELECT
                br.session_id,
                bs.started_at AS session_started_at,
                bs.finished_at AS session_finished_at,
                bs.branch_name,
                bs.commit_hash AS source_commit_hash,
                bs.container_runtime,
                bs.container_image_tag,
                bs.container_image_digest,
                bs.container_image_id,
                bs.gpu_name,
                bs.gpu_total_vram_mb,
                bs.gpu_driver_version,
                bs.cuda_version,
                bs.gpu_target,
                br.benchmark_name,
                COALESCE(br.suite_name, bcr.suite_name) AS suite_name,
                br.framework,
                br.comparison_class,
                COUNT(*) AS run_count,
                SUM(CASE WHEN br.status = 'ok' THEN 1 ELSE 0 END) AS ok_count,
                SUM(CASE WHEN br.status != 'ok' THEN 1 ELSE 0 END) AS non_ok_count
            FROM benchmark_runs br
            JOIN benchmark_sessions bs ON bs.session_id = br.session_id
            LEFT JOIN benchmark_case_registry bcr ON bcr.case_id = br.case_id
            GROUP BY
                br.session_id,
                bs.started_at,
                bs.finished_at,
                bs.branch_name,
                bs.commit_hash,
                bs.container_runtime,
                bs.container_image_tag,
                bs.container_image_digest,
                bs.container_image_id,
                bs.gpu_name,
                bs.gpu_total_vram_mb,
                bs.gpu_driver_version,
                bs.cuda_version,
                bs.gpu_target,
                br.benchmark_name,
                COALESCE(br.suite_name, bcr.suite_name),
                br.framework,
                br.comparison_class;

            CREATE VIEW v_benchmark_fallback_audit AS
            SELECT *
            FROM v_benchmark_run_enriched
            WHERE backend_fallback_used = 1
               OR comparison_class IN ('unsupported_comparator', 'mojogp_only_scale', 'mojogp_only');

            CREATE VIEW v_benchmark_pairwise_compare AS
            SELECT
                moj.session_id,
                moj.comparison_id,
                moj.benchmark_name,
                moj.kernel,
                moj.n,
                moj.d,
                moj.max_iterations,
                moj.training_method,
                moj.prediction_mode,
                moj.comparison_class,
                moj.fairness_note,
                moj.source_commit_hash,
                moj.container_runtime,
                moj.container_image_tag,
                moj.container_image_digest,
                moj.container_image_id,
                moj.gpu_name,
                moj.gpu_total_vram_mb,
                moj.gpu_driver_version,
                moj.cuda_version,
                moj.gpu_target,
                moj.run_id AS mojogp_run_id,
                gp.run_id AS gpytorch_run_id,
                moj.training_time_s AS mojogp_training_time_s,
                gp.training_time_s AS gpytorch_training_time_s,
                moj.prediction_total_time_s AS mojogp_prediction_time_s,
                gp.prediction_total_time_s AS gpytorch_prediction_time_s,
                moj.prediction_apply_time_s AS mojogp_prediction_apply_time_s,
                gp.prediction_apply_time_s AS gpytorch_prediction_apply_time_s,
                moj.prediction_cache_prepare_time_s AS mojogp_prediction_cache_prepare_time_s,
                gp.prediction_cache_prepare_time_s AS gpytorch_prediction_cache_prepare_time_s,
                moj.prediction_cache_plus_apply_time_s AS mojogp_prediction_cache_plus_apply_time_s,
                gp.prediction_cache_plus_apply_time_s AS gpytorch_prediction_cache_plus_apply_time_s,
                moj.end_to_end_time_s AS mojogp_end_to_end_time_s,
                gp.end_to_end_time_s AS gpytorch_end_to_end_time_s,
                moj.training_peak_gpu_mb AS mojogp_training_peak_gpu_mb,
                gp.training_peak_gpu_mb AS gpytorch_training_peak_gpu_mb,
                moj.training_delta_gpu_mb AS mojogp_training_delta_gpu_mb,
                gp.training_delta_gpu_mb AS gpytorch_training_delta_gpu_mb,
                moj.prediction_peak_gpu_mb AS mojogp_prediction_peak_gpu_mb,
                gp.prediction_peak_gpu_mb AS gpytorch_prediction_peak_gpu_mb,
                moj.prediction_delta_gpu_mb AS mojogp_prediction_delta_gpu_mb,
                gp.prediction_delta_gpu_mb AS gpytorch_prediction_delta_gpu_mb,
                moj.exact_prediction_peak_gpu_mb AS mojogp_exact_prediction_peak_gpu_mb,
                gp.exact_prediction_peak_gpu_mb AS gpytorch_exact_prediction_peak_gpu_mb,
                moj.exact_prediction_delta_gpu_mb AS mojogp_exact_prediction_delta_gpu_mb,
                gp.exact_prediction_delta_gpu_mb AS gpytorch_exact_prediction_delta_gpu_mb,
                moj.love_prediction_peak_gpu_mb AS mojogp_love_prediction_peak_gpu_mb,
                gp.love_prediction_peak_gpu_mb AS gpytorch_love_prediction_peak_gpu_mb,
                moj.love_prediction_delta_gpu_mb AS mojogp_love_prediction_delta_gpu_mb,
                gp.love_prediction_delta_gpu_mb AS gpytorch_love_prediction_delta_gpu_mb,
                moj.scaling_peak_gpu_mb AS mojogp_scaling_peak_gpu_mb,
                gp.scaling_peak_gpu_mb AS gpytorch_scaling_peak_gpu_mb,
                moj.scaling_memory_metric AS mojogp_scaling_memory_metric,
                gp.scaling_memory_metric AS gpytorch_scaling_memory_metric,
                moj.gpu_max_mb AS mojogp_gpu_max_mb,
                gp.gpu_max_mb AS gpytorch_gpu_max_mb,
                moj.iter_time_p5_ms AS mojogp_iter_time_p5_ms,
                gp.iter_time_p5_ms AS gpytorch_iter_time_p5_ms,
                moj.iter_time_p95_ms AS mojogp_iter_time_p95_ms,
                gp.iter_time_p95_ms AS gpytorch_iter_time_p95_ms,
                moj.training_cg_mean AS mojogp_training_cg_mean,
                gp.training_cg_mean AS gpytorch_training_cg_mean,
                moj.prediction_cg_mean AS mojogp_prediction_cg_mean,
                gp.prediction_cg_mean AS gpytorch_prediction_cg_mean,
                moj.prediction_timing_quality AS mojogp_prediction_timing_quality,
                gp.prediction_timing_quality AS gpytorch_prediction_timing_quality,
                moj.rmse AS mojogp_rmse,
                gp.rmse AS gpytorch_rmse,
                moj.r_squared AS mojogp_r_squared,
                gp.r_squared AS gpytorch_r_squared,
                CASE WHEN moj.training_time_s > 0 THEN gp.training_time_s / moj.training_time_s END AS train_speedup_vs_mojogp,
                CASE WHEN moj.prediction_total_time_s > 0 THEN gp.prediction_total_time_s / moj.prediction_total_time_s END AS prediction_speedup_vs_mojogp,
                CASE WHEN moj.prediction_apply_time_s > 0 THEN gp.prediction_apply_time_s / moj.prediction_apply_time_s END AS prediction_apply_speedup_vs_mojogp,
                CASE WHEN moj.end_to_end_time_s > 0 THEN gp.end_to_end_time_s / moj.end_to_end_time_s END AS end_to_end_speedup_vs_mojogp,
                CASE WHEN gp.gpu_max_mb > 0 THEN moj.gpu_max_mb / gp.gpu_max_mb END AS gpu_memory_ratio_vs_gpytorch,
                CASE WHEN gp.scaling_peak_gpu_mb > 0 THEN moj.scaling_peak_gpu_mb / gp.scaling_peak_gpu_mb END AS scaling_memory_ratio_vs_gpytorch,
                CASE WHEN gp.rmse > 0 THEN moj.rmse / gp.rmse END AS rmse_ratio_vs_gpytorch,
                (moj.r_squared - gp.r_squared) AS r_squared_delta_vs_gpytorch
            FROM v_benchmark_run_enriched moj
            JOIN v_benchmark_run_enriched gp
              ON moj.session_id = gp.session_id
             AND moj.comparison_id = gp.comparison_id
             AND moj.framework = 'mojogp'
             AND gp.framework = 'gpytorch';

            CREATE VIEW v_benchmark_scaling_steps AS
            WITH ordered AS (
                SELECT
                    bre.*,
                    LAG(run_id) OVER lane AS prev_run_id,
                    LAG(n) OVER lane AS n_lo,
                    LAG(training_time_s) OVER lane AS training_time_lo_s,
                    LAG(prediction_total_time_s) OVER lane AS prediction_time_lo_s,
                    LAG(prediction_apply_time_s) OVER lane AS prediction_apply_time_lo_s,
                    LAG(end_to_end_time_s) OVER lane AS end_to_end_time_lo_s,
                    LAG(gpu_max_mb) OVER lane AS gpu_max_lo_mb,
                    LAG(scaling_peak_gpu_mb) OVER lane AS scaling_peak_lo_mb,
                    LAG(rmse) OVER lane AS rmse_lo,
                    LAG(r_squared) OVER lane AS r_squared_lo
                FROM v_benchmark_run_enriched bre
                WINDOW lane AS (
                    PARTITION BY
                        session_id,
                        benchmark_name,
                        framework,
                        training_method,
                        prediction_mode,
                        kernel,
                        d,
                        max_iterations,
                        comparison_class
                    ORDER BY n ASC, started_at ASC
                )
            )
            SELECT
                session_id,
                benchmark_name,
                framework,
                training_method,
                prediction_mode,
                kernel,
                d,
                max_iterations,
                comparison_class,
                prev_run_id,
                run_id,
                n_lo,
                n AS n_hi,
                training_time_lo_s,
                training_time_s AS training_time_hi_s,
                prediction_time_lo_s,
                prediction_total_time_s AS prediction_time_hi_s,
                prediction_apply_time_lo_s,
                prediction_apply_time_s AS prediction_apply_time_hi_s,
                end_to_end_time_lo_s,
                end_to_end_time_s AS end_to_end_time_hi_s,
                gpu_max_lo_mb,
                gpu_max_mb AS gpu_max_hi_mb,
                scaling_memory_metric,
                scaling_peak_lo_mb,
                scaling_peak_gpu_mb AS scaling_peak_hi_mb,
                rmse_lo,
                rmse AS rmse_hi,
                r_squared_lo,
                r_squared AS r_squared_hi,
                CASE WHEN training_time_lo_s > 0 THEN training_time_s / training_time_lo_s END AS training_time_ratio,
                CASE WHEN prediction_time_lo_s > 0 THEN prediction_total_time_s / prediction_time_lo_s END AS prediction_time_ratio,
                CASE WHEN prediction_apply_time_lo_s > 0 THEN prediction_apply_time_s / prediction_apply_time_lo_s END AS prediction_apply_time_ratio,
                CASE WHEN end_to_end_time_lo_s > 0 THEN end_to_end_time_s / end_to_end_time_lo_s END AS end_to_end_time_ratio,
                CASE WHEN gpu_max_lo_mb > 0 THEN gpu_max_mb / gpu_max_lo_mb END AS gpu_max_ratio,
                CASE WHEN scaling_peak_lo_mb > 0 THEN scaling_peak_gpu_mb / scaling_peak_lo_mb END AS scaling_peak_ratio,
                CASE WHEN n_lo > 0 AND training_time_lo_s > 0 AND training_time_s > 0
                    THEN (training_time_s / training_time_lo_s) / (CAST(n AS REAL) / CAST(n_lo AS REAL))
                END AS training_ratio_over_n_ratio,
                CASE WHEN n_lo > 0 AND end_to_end_time_lo_s > 0 AND end_to_end_time_s > 0
                    THEN (end_to_end_time_s / end_to_end_time_lo_s) / (CAST(n AS REAL) / CAST(n_lo AS REAL))
                END AS end_to_end_ratio_over_n_ratio,
                (rmse - rmse_lo) AS rmse_delta,
                (r_squared - r_squared_lo) AS r_squared_delta
            FROM ordered
            WHERE n_lo IS NOT NULL;

            CREATE VIEW v_benchmark_parameter_recovery AS
            SELECT
                bpr.*,
                br.session_id,
                br.case_id,
                br.benchmark_group_id,
                br.benchmark_name,
                br.suite_name,
                br.framework,
                br.comparison_class,
                br.baseline_backend,
                br.fairness_note,
                br.n,
                br.d,
                br.num_tasks,
                br.kernel,
                br.model_type,
                br.training_method,
                br.prediction_mode,
                br.status AS run_status,
                br.started_at,
                br.finished_at,
                br.commit_hash,
                br.gpu_name,
                br.gpu_total_vram_mb,
                json_extract(br.config_json, '$.accuracy_case_id') AS accuracy_case_id,
                json_extract(br.config_json, '$.difficulty') AS difficulty,
                json_extract(br.config_json, '$.dataset_family') AS dataset_family,
                json_extract(br.config_json, '$.quality_status') AS quality_status,
                json_extract(br.config_json, '$.quality_flags') AS quality_flags
            FROM benchmark_parameter_recovery bpr
            JOIN benchmark_runs br ON br.run_id = bpr.run_id;
            """
        )

    @staticmethod
    def _ensure_columns(
        conn: sqlite3.Connection,
        table_name: str,
        columns: dict[str, str],
    ) -> None:
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        for name, column_type in columns.items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {name} {column_type}")

    def upsert_session(self, record: BenchmarkSessionRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_sessions (
                    session_id, started_at, finished_at, branch_name, commit_hash,
                    commit_hash_short, git_clean, worktree_path, profiling_config_false,
                    profiling_probe_passed, jit_cache_dir, container_runtime,
                    container_image_tag, container_image_digest, container_image_id,
                    gpu_name, gpu_total_vram_mb, gpu_driver_version, cuda_version, gpu_target
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    finished_at=excluded.finished_at,
                    branch_name=excluded.branch_name,
                    commit_hash=excluded.commit_hash,
                    commit_hash_short=excluded.commit_hash_short,
                    git_clean=excluded.git_clean,
                    worktree_path=excluded.worktree_path,
                    profiling_config_false=excluded.profiling_config_false,
                    profiling_probe_passed=excluded.profiling_probe_passed,
                    jit_cache_dir=excluded.jit_cache_dir,
                    container_runtime=excluded.container_runtime,
                    container_image_tag=excluded.container_image_tag,
                    container_image_digest=excluded.container_image_digest,
                    container_image_id=excluded.container_image_id,
                    gpu_name=excluded.gpu_name,
                    gpu_total_vram_mb=excluded.gpu_total_vram_mb,
                    gpu_driver_version=excluded.gpu_driver_version,
                    cuda_version=excluded.cuda_version,
                    gpu_target=excluded.gpu_target
                """,
                (
                    record.session_id,
                    record.started_at,
                    record.finished_at,
                    record.branch_name,
                    record.commit_hash,
                    record.commit_hash_short,
                    int(record.git_clean),
                    record.worktree_path,
                    int(record.profiling_config_false),
                    int(record.profiling_probe_passed),
                    record.jit_cache_dir,
                    record.container_runtime,
                    record.container_image_tag,
                    record.container_image_digest,
                    record.container_image_id,
                    record.gpu_name,
                    record.gpu_total_vram_mb,
                    record.gpu_driver_version,
                    record.cuda_version,
                    record.gpu_target,
                ),
            )

    def register_case(
        self,
        *,
        case_id: str,
        benchmark_group_id: str,
        framework: str,
        suite_name: str,
        benchmark_name: str,
        config: dict[str, Any],
    ) -> None:
        specialization = extract_specialization_columns(config)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_case_registry(
                    case_id, benchmark_group_id, framework, suite_name, benchmark_name,
                    config_json, base_case_id, specialization_key,
                    specialization_family, specialization_mode,
                    specialization_source, specialization_descriptor_json,
                    specialization_config_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    benchmark_group_id=excluded.benchmark_group_id,
                    framework=excluded.framework,
                    suite_name=excluded.suite_name,
                    benchmark_name=excluded.benchmark_name,
                    config_json=excluded.config_json,
                    base_case_id=excluded.base_case_id,
                    specialization_key=excluded.specialization_key,
                    specialization_family=excluded.specialization_family,
                    specialization_mode=excluded.specialization_mode,
                    specialization_source=excluded.specialization_source,
                    specialization_descriptor_json=excluded.specialization_descriptor_json,
                    specialization_config_json=excluded.specialization_config_json
                """,
                (
                    case_id,
                    benchmark_group_id,
                    framework,
                    suite_name,
                    benchmark_name,
                    _json(config),
                    specialization["base_case_id"],
                    specialization["specialization_key"],
                    specialization["specialization_family"],
                    specialization["specialization_mode"],
                    specialization["specialization_source"],
                    _json(specialization["specialization_descriptor_json"]),
                    _json(specialization["specialization_config_json"]),
                ),
            )

    def register_specialization(
        self,
        *,
        specialization_key: str,
        specialization_family: str,
        specialization_source: str,
        policy_version: str,
        config: dict[str, Any],
        notes: str | None,
        created_at: str,
        active: bool = True,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_specialization_registry(
                    specialization_key, specialization_family, specialization_source,
                    policy_version, config_json, notes, created_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(specialization_key) DO UPDATE SET
                    specialization_family=excluded.specialization_family,
                    specialization_source=excluded.specialization_source,
                    policy_version=excluded.policy_version,
                    config_json=excluded.config_json,
                    notes=excluded.notes,
                    created_at=excluded.created_at,
                    active=excluded.active
                """,
                (
                    specialization_key,
                    specialization_family,
                    specialization_source,
                    policy_version,
                    _json(config),
                    notes,
                    created_at,
                    int(active),
                ),
            )

    def register_comparison(
        self,
        *,
        comparison_id: str,
        mojogp_case_id: str,
        gpytorch_case_id: str,
        comparison_class: str,
        fairness_note: str,
        fairness_axes: dict[str, Any],
        mojogp_specialization_key: str | None = None,
        specialization_family: str | None = None,
        specialization_config: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_comparison_registry(
                    comparison_id, mojogp_case_id, gpytorch_case_id, comparison_class,
                    fairness_note, fairness_axes_json, mojogp_specialization_key,
                    specialization_family, specialization_config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(comparison_id) DO UPDATE SET
                    mojogp_case_id=excluded.mojogp_case_id,
                    gpytorch_case_id=excluded.gpytorch_case_id,
                    comparison_class=excluded.comparison_class,
                    fairness_note=excluded.fairness_note,
                    fairness_axes_json=excluded.fairness_axes_json,
                    mojogp_specialization_key=excluded.mojogp_specialization_key,
                    specialization_family=excluded.specialization_family,
                    specialization_config_json=excluded.specialization_config_json
                """,
                (
                    comparison_id,
                    mojogp_case_id,
                    gpytorch_case_id,
                    comparison_class,
                    fairness_note,
                    _json(fairness_axes),
                    mojogp_specialization_key,
                    specialization_family,
                    _json(specialization_config or {}),
                ),
            )

    def register_dataset(
        self,
        *,
        dataset_id: str,
        generator_name: str,
        config: dict[str, Any],
        seed: int | None,
        artifact_path: str,
        artifact_sha256: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_datasets(
                    dataset_id, generator_name, config_json, seed, artifact_path,
                    artifact_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    generator_name=excluded.generator_name,
                    config_json=excluded.config_json,
                    seed=excluded.seed,
                    artifact_path=excluded.artifact_path,
                    artifact_sha256=excluded.artifact_sha256,
                    created_at=excluded.created_at
                """,
                (dataset_id, generator_name, _json(config), seed, artifact_path, artifact_sha256, created_at),
            )

    def register_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str | None,
        artifact_type: str,
        path: str,
        sha256: str,
        created_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_artifacts(artifact_id, run_id, artifact_type, path, sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    artifact_type=excluded.artifact_type,
                    path=excluded.path,
                    sha256=excluded.sha256,
                    created_at=excluded.created_at
                """,
                (artifact_id, run_id, artifact_type, path, sha256, created_at),
            )

    def register_run(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("base_case_id", None)
        row.setdefault("specialization_key", None)
        row.setdefault("specialization_family", None)
        row.setdefault("specialization_mode", None)
        row.setdefault("specialization_source", None)
        row.setdefault("specialization_descriptor_json", _json({}))
        row.setdefault("specialization_config_json", _json({}))
        row.setdefault("study_id", None)
        row.setdefault("trial_id", None)
        row.setdefault("objective_name", None)
        row.setdefault("objective_metric", None)
        row.setdefault("constraint_json", _json({}))
        row.setdefault("training_time_s", None)
        row.setdefault("prediction_mean_time_s", None)
        row.setdefault("prediction_variance_time_s", None)
        row.setdefault("end_to_end_time_s", None)
        row.setdefault("iterations_run", None)
        row.setdefault("max_iterations", None)
        row.setdefault("early_stopped", None)
        row.setdefault("ms_per_iteration", None)
        row.setdefault("iter_time_min_ms", None)
        row.setdefault("iter_time_q25_ms", None)
        row.setdefault("iter_time_mean_ms", None)
        row.setdefault("iter_time_median_ms", None)
        row.setdefault("iter_time_q75_ms", None)
        row.setdefault("iter_time_max_ms", None)
        row.setdefault("iter_time_p5_ms", None)
        row.setdefault("iter_time_p95_ms", None)
        row.setdefault("startup_compile_time_s", None)
        row.setdefault("startup_warm_cache_hit_s", None)
        row.setdefault("startup_prepare_time_s", None)
        row.setdefault("prediction_cold_first_time_s", None)
        row.setdefault("prediction_cache_prepare_time_s", None)
        row.setdefault("prediction_prepared_apply_time_s", None)
        row.setdefault("prediction_repeated_median_time_s", None)
        row.setdefault("prediction_repeated_p5_time_s", None)
        row.setdefault("prediction_repeated_p95_time_s", None)
        row.setdefault("prediction_alpha_time_s", None)
        row.setdefault("prediction_love_root_time_s", None)
        prediction_scaling = _prediction_scaling_rows(row.get("prediction_x_test_scaling"))
        if not prediction_scaling:
            prediction_scaling = _prediction_scaling_rows(
                row.get("prediction_x_test_scaling_json")
            )
        row["prediction_x_test_scaling_json"] = (
            _json(prediction_scaling) if prediction_scaling else None
        )
        row.setdefault("contract_passed", None)
        row.setdefault("contract_summary_json", None)
        row.setdefault("rmse", None)
        row.setdefault("mae", None)
        row.setdefault("r_squared", None)
        row.setdefault("crps", None)
        row.setdefault("msll", None)
        row.setdefault("calibration_error", None)
        row.setdefault("sharpness", None)
        row.setdefault("interval_width_95", None)
        row.setdefault("final_nll", None)
        row.setdefault("container_runtime", None)
        row.setdefault("container_image_tag", None)
        row.setdefault("container_image_digest", None)
        row.setdefault("container_image_id", None)
        row.setdefault("suite_name", None)
        row.setdefault("training_peak_gpu_mb", None)
        row.setdefault("training_delta_gpu_mb", None)
        row.setdefault("prediction_peak_gpu_mb", None)
        row.setdefault("prediction_delta_gpu_mb", None)
        row.setdefault("exact_prediction_peak_gpu_mb", None)
        row.setdefault("exact_prediction_delta_gpu_mb", None)
        row.setdefault("love_prediction_peak_gpu_mb", None)
        row.setdefault("love_prediction_delta_gpu_mb", None)
        row.setdefault("scaling_peak_gpu_mb", None)
        row.setdefault("scaling_memory_metric", None)
        row.setdefault("gpu_name", None)
        row.setdefault("gpu_total_vram_mb", None)
        row.setdefault("gpu_driver_version", None)
        row.setdefault("cuda_version", None)
        row.setdefault("gpu_target", None)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_runs(
                    run_id, session_id, case_id, benchmark_group_id, n, d, num_tasks,
                    kernel, model_type, training_method, prediction_mode, suite_name,
                    comparison_class, baseline_backend, fairness_note,
                    base_case_id, specialization_key, specialization_family,
                    specialization_mode, specialization_source,
                    specialization_descriptor_json, specialization_config_json,
                    study_id, trial_id, objective_name, objective_metric,
                    constraint_json,
                    dataset_id, comparison_id, artifact_id,
                    status, started_at, finished_at, framework, config_json, benchmark_name,
                    result_json_path, training_time_s, prediction_mean_time_s,
                    prediction_variance_time_s,
                    prediction_cold_first_time_s, prediction_cache_prepare_time_s,
                    prediction_prepared_apply_time_s,
                    prediction_repeated_median_time_s,
                    prediction_repeated_p5_time_s, prediction_repeated_p95_time_s,
                    prediction_alpha_time_s, prediction_love_root_time_s,
                    prediction_x_test_scaling_json,
                    end_to_end_time_s,
                    iterations_run, max_iterations, early_stopped, ms_per_iteration,
                    iter_time_min_ms, iter_time_q25_ms, iter_time_mean_ms,
                    iter_time_median_ms, iter_time_q75_ms, iter_time_max_ms,
                    iter_time_p5_ms, iter_time_p95_ms,
                    startup_compile_time_s, startup_warm_cache_hit_s,
                    startup_prepare_time_s,
                    contract_passed, contract_summary_json,
                    training_peak_gpu_mb, training_delta_gpu_mb,
                    prediction_peak_gpu_mb, prediction_delta_gpu_mb,
                    exact_prediction_peak_gpu_mb, exact_prediction_delta_gpu_mb,
                    love_prediction_peak_gpu_mb, love_prediction_delta_gpu_mb,
                    scaling_peak_gpu_mb, scaling_memory_metric,
                    gpu_baseline_mb,
                    gpu_current_mb, gpu_delta_mb, gpu_max_mb, gpu_isolated_peak_mb,
                    gpu_isolated_current_mb, gpu_samples, measurement_method_primary,
                    torch_baseline_mb, torch_peak_mb, torch_peak_delta_mb,
                    torch_current_delta_mb, torch_reserved_delta_mb, cpu_peak_mb,
                    rmse, mae, r_squared, crps, msll, calibration_error,
                    sharpness, interval_width_95, final_nll,
                    branch_name, commit_hash, git_clean, profiling_probe_passed,
                    container_runtime, container_image_tag, container_image_digest,
                    container_image_id, gpu_name, gpu_total_vram_mb,
                    gpu_driver_version, cuda_version, gpu_target
                ) VALUES (
                    :run_id, :session_id, :case_id, :benchmark_group_id, :n, :d, :num_tasks,
                    :kernel, :model_type, :training_method, :prediction_mode, :suite_name,
                    :comparison_class, :baseline_backend, :fairness_note,
                    :base_case_id, :specialization_key, :specialization_family,
                    :specialization_mode, :specialization_source,
                    :specialization_descriptor_json, :specialization_config_json,
                    :study_id, :trial_id, :objective_name, :objective_metric,
                    :constraint_json,
                    :dataset_id, :comparison_id, :artifact_id,
                    :status, :started_at, :finished_at, :framework, :config_json, :benchmark_name,
                    :result_json_path, :training_time_s, :prediction_mean_time_s,
                    :prediction_variance_time_s,
                    :prediction_cold_first_time_s, :prediction_cache_prepare_time_s,
                    :prediction_prepared_apply_time_s,
                    :prediction_repeated_median_time_s,
                    :prediction_repeated_p5_time_s, :prediction_repeated_p95_time_s,
                    :prediction_alpha_time_s, :prediction_love_root_time_s,
                    :prediction_x_test_scaling_json,
                    :end_to_end_time_s,
                    :iterations_run, :max_iterations, :early_stopped, :ms_per_iteration,
                    :iter_time_min_ms, :iter_time_q25_ms, :iter_time_mean_ms,
                    :iter_time_median_ms, :iter_time_q75_ms, :iter_time_max_ms,
                    :iter_time_p5_ms, :iter_time_p95_ms,
                    :startup_compile_time_s, :startup_warm_cache_hit_s,
                    :startup_prepare_time_s,
                    :contract_passed, :contract_summary_json,
                    :training_peak_gpu_mb, :training_delta_gpu_mb,
                    :prediction_peak_gpu_mb, :prediction_delta_gpu_mb,
                    :exact_prediction_peak_gpu_mb, :exact_prediction_delta_gpu_mb,
                    :love_prediction_peak_gpu_mb, :love_prediction_delta_gpu_mb,
                    :scaling_peak_gpu_mb, :scaling_memory_metric,
                    :gpu_baseline_mb,
                    :gpu_current_mb, :gpu_delta_mb, :gpu_max_mb, :gpu_isolated_peak_mb,
                    :gpu_isolated_current_mb, :gpu_samples, :measurement_method_primary,
                    :torch_baseline_mb, :torch_peak_mb, :torch_peak_delta_mb,
                    :torch_current_delta_mb, :torch_reserved_delta_mb, :cpu_peak_mb,
                    :rmse, :mae, :r_squared, :crps, :msll, :calibration_error,
                    :sharpness, :interval_width_95, :final_nll,
                    :branch_name, :commit_hash, :git_clean, :profiling_probe_passed,
                    :container_runtime, :container_image_tag, :container_image_digest,
                    :container_image_id, :gpu_name, :gpu_total_vram_mb,
                    :gpu_driver_version, :cuda_version, :gpu_target
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    session_id=excluded.session_id,
                    case_id=excluded.case_id,
                    benchmark_group_id=excluded.benchmark_group_id,
                    n=excluded.n,
                    d=excluded.d,
                    num_tasks=excluded.num_tasks,
                    kernel=excluded.kernel,
                    model_type=excluded.model_type,
                    training_method=excluded.training_method,
                    prediction_mode=excluded.prediction_mode,
                    suite_name=excluded.suite_name,
                    comparison_class=excluded.comparison_class,
                    baseline_backend=excluded.baseline_backend,
                    fairness_note=excluded.fairness_note,
                    base_case_id=excluded.base_case_id,
                    specialization_key=excluded.specialization_key,
                    specialization_family=excluded.specialization_family,
                    specialization_mode=excluded.specialization_mode,
                    specialization_source=excluded.specialization_source,
                    specialization_descriptor_json=excluded.specialization_descriptor_json,
                    specialization_config_json=excluded.specialization_config_json,
                    study_id=excluded.study_id,
                    trial_id=excluded.trial_id,
                    objective_name=excluded.objective_name,
                    objective_metric=excluded.objective_metric,
                    constraint_json=excluded.constraint_json,
                    dataset_id=excluded.dataset_id,
                    comparison_id=excluded.comparison_id,
                    artifact_id=excluded.artifact_id,
                    status=excluded.status,
                    started_at=excluded.started_at,
                    finished_at=excluded.finished_at,
                    framework=excluded.framework,
                    config_json=excluded.config_json,
                    benchmark_name=excluded.benchmark_name,
                    result_json_path=excluded.result_json_path,
                    training_time_s=excluded.training_time_s,
                    prediction_mean_time_s=excluded.prediction_mean_time_s,
                    prediction_variance_time_s=excluded.prediction_variance_time_s,
                    prediction_cold_first_time_s=excluded.prediction_cold_first_time_s,
                    prediction_cache_prepare_time_s=excluded.prediction_cache_prepare_time_s,
                    prediction_prepared_apply_time_s=excluded.prediction_prepared_apply_time_s,
                    prediction_repeated_median_time_s=excluded.prediction_repeated_median_time_s,
                    prediction_repeated_p5_time_s=excluded.prediction_repeated_p5_time_s,
                    prediction_repeated_p95_time_s=excluded.prediction_repeated_p95_time_s,
                    prediction_alpha_time_s=excluded.prediction_alpha_time_s,
                    prediction_love_root_time_s=excluded.prediction_love_root_time_s,
                    prediction_x_test_scaling_json=excluded.prediction_x_test_scaling_json,
                    end_to_end_time_s=excluded.end_to_end_time_s,
                    iterations_run=excluded.iterations_run,
                    max_iterations=excluded.max_iterations,
                    early_stopped=excluded.early_stopped,
                    ms_per_iteration=excluded.ms_per_iteration,
                    iter_time_min_ms=excluded.iter_time_min_ms,
                    iter_time_q25_ms=excluded.iter_time_q25_ms,
                    iter_time_mean_ms=excluded.iter_time_mean_ms,
                    iter_time_median_ms=excluded.iter_time_median_ms,
                    iter_time_q75_ms=excluded.iter_time_q75_ms,
                    iter_time_max_ms=excluded.iter_time_max_ms,
                    iter_time_p5_ms=excluded.iter_time_p5_ms,
                    iter_time_p95_ms=excluded.iter_time_p95_ms,
                    startup_compile_time_s=excluded.startup_compile_time_s,
                    startup_warm_cache_hit_s=excluded.startup_warm_cache_hit_s,
                    startup_prepare_time_s=excluded.startup_prepare_time_s,
                    contract_passed=excluded.contract_passed,
                    contract_summary_json=excluded.contract_summary_json,
                    training_peak_gpu_mb=excluded.training_peak_gpu_mb,
                    training_delta_gpu_mb=excluded.training_delta_gpu_mb,
                    prediction_peak_gpu_mb=excluded.prediction_peak_gpu_mb,
                    prediction_delta_gpu_mb=excluded.prediction_delta_gpu_mb,
                    exact_prediction_peak_gpu_mb=excluded.exact_prediction_peak_gpu_mb,
                    exact_prediction_delta_gpu_mb=excluded.exact_prediction_delta_gpu_mb,
                    love_prediction_peak_gpu_mb=excluded.love_prediction_peak_gpu_mb,
                    love_prediction_delta_gpu_mb=excluded.love_prediction_delta_gpu_mb,
                    scaling_peak_gpu_mb=excluded.scaling_peak_gpu_mb,
                    scaling_memory_metric=excluded.scaling_memory_metric,
                    gpu_baseline_mb=excluded.gpu_baseline_mb,
                    gpu_current_mb=excluded.gpu_current_mb,
                    gpu_delta_mb=excluded.gpu_delta_mb,
                    gpu_max_mb=excluded.gpu_max_mb,
                    gpu_isolated_peak_mb=excluded.gpu_isolated_peak_mb,
                    gpu_isolated_current_mb=excluded.gpu_isolated_current_mb,
                    gpu_samples=excluded.gpu_samples,
                    measurement_method_primary=excluded.measurement_method_primary,
                    torch_baseline_mb=excluded.torch_baseline_mb,
                    torch_peak_mb=excluded.torch_peak_mb,
                    torch_peak_delta_mb=excluded.torch_peak_delta_mb,
                    torch_current_delta_mb=excluded.torch_current_delta_mb,
                    torch_reserved_delta_mb=excluded.torch_reserved_delta_mb,
                    cpu_peak_mb=excluded.cpu_peak_mb,
                    rmse=excluded.rmse,
                    mae=excluded.mae,
                    r_squared=excluded.r_squared,
                    crps=excluded.crps,
                    msll=excluded.msll,
                    calibration_error=excluded.calibration_error,
                    sharpness=excluded.sharpness,
                    interval_width_95=excluded.interval_width_95,
                    final_nll=excluded.final_nll,
                    branch_name=excluded.branch_name,
                    commit_hash=excluded.commit_hash,
                    git_clean=excluded.git_clean,
                    profiling_probe_passed=excluded.profiling_probe_passed,
                    container_runtime=excluded.container_runtime,
                    container_image_tag=excluded.container_image_tag,
                    container_image_digest=excluded.container_image_digest,
                    container_image_id=excluded.container_image_id,
                    gpu_name=excluded.gpu_name,
                    gpu_total_vram_mb=excluded.gpu_total_vram_mb,
                    gpu_driver_version=excluded.gpu_driver_version,
                    cuda_version=excluded.cuda_version,
                    gpu_target=excluded.gpu_target
                """,
                row,
            )
            conn.execute(
                "DELETE FROM benchmark_prediction_x_test_scaling WHERE run_id = ?",
                (row["run_id"],),
            )
            for item in prediction_scaling:
                conn.execute(
                    """
                    INSERT INTO benchmark_prediction_x_test_scaling(
                        run_id, n_test, size_role, comparison_class, timing_quality,
                        status, failure_reason, failure_stage, error_type, error_message,
                        cache_used, first_apply_time_s, repeated_median_time_s,
                        repeated_p5_time_s, repeated_p95_time_s, repeat_count,
                        prediction_peak_gpu_mb, prediction_delta_gpu_mb,
                        mean_time_s, variance_time_s
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, n_test, size_role) DO UPDATE SET
                        comparison_class=excluded.comparison_class,
                        timing_quality=excluded.timing_quality,
                        status=excluded.status,
                        failure_reason=excluded.failure_reason,
                        failure_stage=excluded.failure_stage,
                        error_type=excluded.error_type,
                        error_message=excluded.error_message,
                        cache_used=excluded.cache_used,
                        first_apply_time_s=excluded.first_apply_time_s,
                        repeated_median_time_s=excluded.repeated_median_time_s,
                        repeated_p5_time_s=excluded.repeated_p5_time_s,
                        repeated_p95_time_s=excluded.repeated_p95_time_s,
                        repeat_count=excluded.repeat_count,
                        prediction_peak_gpu_mb=excluded.prediction_peak_gpu_mb,
                        prediction_delta_gpu_mb=excluded.prediction_delta_gpu_mb,
                        mean_time_s=excluded.mean_time_s,
                        variance_time_s=excluded.variance_time_s
                    """,
                    (
                        row["run_id"],
                        int(item["n_test"]),
                        item.get("size_role"),
                        item.get("comparison_class"),
                        item.get("timing_quality"),
                        item.get("status", "ok"),
                        item.get("failure_reason"),
                        item.get("failure_stage"),
                        item.get("error_type"),
                        item.get("error_message"),
                        None
                        if item.get("cache_used") is None
                        else int(bool(item.get("cache_used"))),
                        item.get("first_apply_time_s"),
                        item.get("repeated_median_time_s"),
                        item.get("repeated_p5_time_s"),
                        item.get("repeated_p95_time_s"),
                        item.get("repeat_count"),
                        item.get("prediction_peak_gpu_mb"),
                        item.get("prediction_delta_gpu_mb"),
                        item.get("mean_time_s"),
                        item.get("variance_time_s"),
                    ),
                )

    def register_parameter_recovery(
        self,
        *,
        run_id: str,
        rows: Any,
    ) -> None:
        recovery_rows = _parameter_recovery_rows(rows)
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM benchmark_parameter_recovery WHERE run_id = ?",
                (run_id,),
            )
            for item in recovery_rows:
                parameter_name = str(item.get("parameter_name") or item.get("name") or "")
                if not parameter_name:
                    continue
                parameter_index = item.get("parameter_index", item.get("index", ""))
                if parameter_index is None:
                    parameter_index = ""
                metadata = item.get("metadata", item.get("metadata_json", {}))
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"raw": metadata}
                conn.execute(
                    """
                    INSERT INTO benchmark_parameter_recovery(
                        run_id, parameter_name, parameter_group, parameter_index,
                        truth_json, learned_json, abs_error, rel_error,
                        signed_error, log_abs_error, truth_norm, learned_norm,
                        status, notes, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, parameter_name, parameter_index) DO UPDATE SET
                        parameter_group=excluded.parameter_group,
                        truth_json=excluded.truth_json,
                        learned_json=excluded.learned_json,
                        abs_error=excluded.abs_error,
                        rel_error=excluded.rel_error,
                        signed_error=excluded.signed_error,
                        log_abs_error=excluded.log_abs_error,
                        truth_norm=excluded.truth_norm,
                        learned_norm=excluded.learned_norm,
                        status=excluded.status,
                        notes=excluded.notes,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        run_id,
                        parameter_name,
                        item.get("parameter_group"),
                        str(parameter_index),
                        _json(item.get("truth")) if "truth" in item else item.get("truth_json"),
                        _json(item.get("learned")) if "learned" in item else item.get("learned_json"),
                        item.get("abs_error"),
                        item.get("rel_error"),
                        item.get("signed_error"),
                        item.get("log_abs_error"),
                        item.get("truth_norm"),
                        item.get("learned_norm"),
                        str(item.get("status", "unknown")),
                        item.get("notes"),
                        _json(metadata),
                    ),
                )

    def fetch_cases_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT bcr.*
                FROM benchmark_case_registry bcr
                JOIN benchmark_runs br ON br.case_id = bcr.case_id
                WHERE br.session_id = ?
                ORDER BY bcr.case_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_case(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_case_registry WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def fetch_datasets_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT bd.*
                FROM benchmark_datasets bd
                WHERE bd.dataset_id IN (
                    SELECT dataset_id
                    FROM benchmark_runs
                    WHERE session_id = ? AND dataset_id IS NOT NULL
                )
                ORDER BY bd.dataset_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_comparisons_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT bcmp.*
                FROM benchmark_comparison_registry bcmp
                WHERE bcmp.comparison_id IN (
                    SELECT comparison_id
                    FROM benchmark_runs
                    WHERE session_id = ? AND comparison_id IS NOT NULL
                )
                ORDER BY bcmp.comparison_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_specializations_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT bsr.*
                FROM benchmark_specialization_registry bsr
                WHERE bsr.specialization_key IN (
                    SELECT specialization_key
                    FROM benchmark_runs
                    WHERE session_id = ? AND specialization_key IS NOT NULL
                )
                ORDER BY bsr.specialization_key ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_specialization_studies_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT bss.*
                FROM benchmark_specialization_studies bss
                WHERE bss.study_id IN (
                    SELECT study_id
                    FROM benchmark_runs
                    WHERE session_id = ? AND study_id IS NOT NULL
                )
                ORDER BY bss.study_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_specialization_trials_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT bst.*
                FROM benchmark_specialization_trials bst
                WHERE bst.trial_id IN (
                    SELECT trial_id
                    FROM benchmark_runs
                    WHERE session_id = ? AND trial_id IS NOT NULL
                )
                ORDER BY bst.trial_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_parameter_recovery_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT bpr.*
                FROM benchmark_parameter_recovery bpr
                JOIN benchmark_runs br ON br.run_id = bpr.run_id
                WHERE br.session_id = ?
                ORDER BY bpr.run_id ASC, bpr.parameter_name ASC, bpr.parameter_index ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_artifacts_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT ba.*
                FROM benchmark_artifacts ba
                WHERE ba.run_id IN (
                    SELECT run_id FROM benchmark_runs WHERE session_id = ?
                ) OR ba.artifact_id IN (
                    SELECT artifact_id FROM benchmark_runs WHERE session_id = ?
                )
                ORDER BY ba.created_at ASC, ba.artifact_id ASC
                """,
                (session_id, session_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def import_session_json(self, export_path: Path) -> dict[str, int]:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        session = payload.get("session")
        if not isinstance(session, dict):
            raise ValueError("session export is missing 'session' object")
        session_existed = self.fetch_session(str(session["session_id"])) is not None
        existing_cases = self._existing_keys(
            "benchmark_case_registry",
            "case_id",
            [str(case["case_id"]) for case in payload.get("cases", [])],
        )
        existing_datasets = self._existing_keys(
            "benchmark_datasets",
            "dataset_id",
            [str(dataset["dataset_id"]) for dataset in payload.get("datasets", [])],
        )
        existing_comparisons = self._existing_keys(
            "benchmark_comparison_registry",
            "comparison_id",
            [str(comparison["comparison_id"]) for comparison in payload.get("comparisons", [])],
        )
        existing_specializations = self._existing_keys(
            "benchmark_specialization_registry",
            "specialization_key",
            [str(specialization["specialization_key"]) for specialization in payload.get("specializations", [])],
        )
        existing_studies = self._existing_keys(
            "benchmark_specialization_studies",
            "study_id",
            [str(study["study_id"]) for study in payload.get("specialization_studies", [])],
        )
        existing_trials = self._existing_keys(
            "benchmark_specialization_trials",
            "trial_id",
            [str(trial["trial_id"]) for trial in payload.get("specialization_trials", [])],
        )
        existing_runs = self._existing_keys(
            "benchmark_runs",
            "run_id",
            [str(run["run_id"]) for run in payload.get("runs", [])],
        )
        existing_parameter_recovery = self._existing_parameter_recovery_keys(
            [dict(row) for row in payload.get("parameter_recovery", []) if isinstance(row, dict)]
        )
        existing_artifacts = self._existing_keys(
            "benchmark_artifacts",
            "artifact_id",
            [str(artifact["artifact_id"]) for artifact in payload.get("artifacts", [])],
        )
        self.upsert_session(BenchmarkSessionRecord(**session))

        imported_cases = 0
        for case in payload.get("cases", []):
            self.register_case(
                case_id=str(case["case_id"]),
                benchmark_group_id=str(case["benchmark_group_id"]),
                framework=str(case["framework"]),
                suite_name=str(case["suite_name"]),
                benchmark_name=str(case["benchmark_name"]),
                config=json.loads(case.get("config_json") or "{}"),
            )
            imported_cases += 1

        imported_datasets = 0
        for dataset in payload.get("datasets", []):
            self.register_dataset(
                dataset_id=str(dataset["dataset_id"]),
                generator_name=str(dataset["generator_name"]),
                config=json.loads(dataset.get("config_json") or "{}"),
                seed=dataset.get("seed"),
                artifact_path=str(dataset["artifact_path"]),
                artifact_sha256=str(dataset["artifact_sha256"]),
                created_at=str(dataset["created_at"]),
            )
            imported_datasets += 1

        imported_comparisons = 0
        for comparison in payload.get("comparisons", []):
            self.register_comparison(
                comparison_id=str(comparison["comparison_id"]),
                mojogp_case_id=str(comparison["mojogp_case_id"]),
                gpytorch_case_id=str(comparison["gpytorch_case_id"]),
                comparison_class=str(comparison["comparison_class"]),
                fairness_note=str(comparison["fairness_note"]),
                fairness_axes=json.loads(comparison.get("fairness_axes_json") or "{}"),
                mojogp_specialization_key=comparison.get("mojogp_specialization_key"),
                specialization_family=comparison.get("specialization_family"),
                specialization_config=json.loads(comparison.get("specialization_config_json") or "{}"),
            )
            imported_comparisons += 1

        imported_specializations = 0
        for specialization in payload.get("specializations", []):
            self.register_specialization(
                specialization_key=str(specialization["specialization_key"]),
                specialization_family=str(specialization["specialization_family"]),
                specialization_source=str(specialization["specialization_source"]),
                policy_version=str(specialization["policy_version"]),
                config=json.loads(specialization.get("config_json") or "{}"),
                notes=specialization.get("notes"),
                created_at=str(specialization["created_at"]),
                active=bool(specialization.get("active", 0)),
            )
            imported_specializations += 1

        imported_studies = 0
        for study in payload.get("specialization_studies", []):
            self.register_specialization_study(
                study_id=str(study["study_id"]),
                created_at=str(study["created_at"]),
                benchmark_group_id=str(study["benchmark_group_id"]),
                objective_name=str(study["objective_name"]),
                objective_metric=str(study["objective_metric"]),
                constraints=json.loads(study.get("constraint_json") or "{}"),
                search_space=json.loads(study.get("search_space_json") or "{}"),
                notes=study.get("notes"),
            )
            imported_studies += 1

        imported_trials = 0
        for trial in payload.get("specialization_trials", []):
            self.register_specialization_trial(
                trial_id=str(trial["trial_id"]),
                study_id=str(trial["study_id"]),
                base_case_id=str(trial["base_case_id"]),
                specialization_key=str(trial["specialization_key"]),
                status=str(trial["status"]),
                trial_config=json.loads(trial.get("trial_config_json") or "{}"),
                result_summary=json.loads(trial.get("result_summary_json") or "{}"),
                created_at=str(trial["created_at"]),
                run_id=trial.get("run_id"),
                objective_value=trial.get("objective_value"),
                constraint_status=trial.get("constraint_status"),
            )
            imported_trials += 1

        imported_runs = 0
        for run in payload.get("runs", []):
            self.register_run(dict(run))
            imported_runs += 1

        imported_parameter_recovery = 0
        parameter_rows_by_run: dict[str, list[dict[str, Any]]] = {}
        for row in payload.get("parameter_recovery", []):
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id", ""))
            if not run_id:
                continue
            parameter_rows_by_run.setdefault(run_id, []).append(dict(row))
            imported_parameter_recovery += 1
        for run_id, rows in parameter_rows_by_run.items():
            self.register_parameter_recovery(run_id=run_id, rows=rows)

        imported_artifacts = 0
        for artifact in payload.get("artifacts", []):
            self.register_artifact(
                artifact_id=str(artifact["artifact_id"]),
                run_id=artifact.get("run_id"),
                artifact_type=str(artifact["artifact_type"]),
                path=str(artifact["path"]),
                sha256=str(artifact["sha256"]),
                created_at=str(artifact["created_at"]),
            )
            imported_artifacts += 1

        existing_total = (
            int(session_existed)
            + len(existing_cases)
            + len(existing_datasets)
            + len(existing_comparisons)
            + len(existing_specializations)
            + len(existing_studies)
            + len(existing_trials)
            + len(existing_runs)
            + len(existing_parameter_recovery)
            + len(existing_artifacts)
        )
        return {
            "sessions": 1,
            "cases": imported_cases,
            "datasets": imported_datasets,
            "comparisons": imported_comparisons,
            "specializations": imported_specializations,
            "specialization_studies": imported_studies,
            "specialization_trials": imported_trials,
            "runs": imported_runs,
            "parameter_recovery": imported_parameter_recovery,
            "artifacts": imported_artifacts,
            "sessions_inserted": 0 if session_existed else 1,
            "sessions_updated": 1 if session_existed else 0,
            "cases_inserted": imported_cases - len(existing_cases),
            "cases_updated": len(existing_cases),
            "datasets_inserted": imported_datasets - len(existing_datasets),
            "datasets_updated": len(existing_datasets),
            "comparisons_inserted": imported_comparisons - len(existing_comparisons),
            "comparisons_updated": len(existing_comparisons),
            "specializations_inserted": imported_specializations - len(existing_specializations),
            "specializations_updated": len(existing_specializations),
            "specialization_studies_inserted": imported_studies - len(existing_studies),
            "specialization_studies_updated": len(existing_studies),
            "specialization_trials_inserted": imported_trials - len(existing_trials),
            "specialization_trials_updated": len(existing_trials),
            "runs_inserted": imported_runs - len(existing_runs),
            "runs_updated": len(existing_runs),
            "parameter_recovery_inserted": imported_parameter_recovery - len(existing_parameter_recovery),
            "parameter_recovery_updated": len(existing_parameter_recovery),
            "artifacts_inserted": imported_artifacts - len(existing_artifacts),
            "artifacts_updated": len(existing_artifacts),
            "duplicates_skipped": existing_total,
        }

    def import_session_bundle(self, export_path: Path) -> dict[str, int]:
        """Import an exported benchmark session bundle JSON.

        The bundle format is intentionally the same deterministic JSON emitted by
        `export_session_json()`. This alias makes the remote/import workflow name
        explicit without breaking existing callers.
        """

        return self.import_session_json(export_path)

    def _existing_keys(
        self,
        table_name: str,
        key_column: str,
        values: list[str],
    ) -> set[str]:
        if not values:
            return set()
        placeholders = ",".join("?" for _ in values)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {key_column} FROM {table_name} WHERE {key_column} IN ({placeholders})",
                tuple(values),
            ).fetchall()
        return {str(row[key_column]) for row in rows}

    def _existing_parameter_recovery_keys(self, rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
        if not rows:
            return set()
        requested = {
            (
                str(row.get("run_id", "")),
                str(row.get("parameter_name", "")),
                str(row.get("parameter_index", "") or ""),
            )
            for row in rows
            if row.get("run_id") and row.get("parameter_name")
        }
        if not requested:
            return set()
        run_ids = sorted({str(row.get("run_id", "")) for row in rows if row.get("run_id")})
        if not run_ids:
            return set()
        placeholders = ",".join("?" for _ in run_ids)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT run_id, parameter_name, parameter_index
                FROM benchmark_parameter_recovery
                WHERE run_id IN ("""
                + placeholders
                + ")",
                tuple(run_ids),
            ).fetchall()
        existing_keys = {
            (str(row["run_id"]), str(row["parameter_name"]), str(row["parameter_index"] or ""))
            for row in existing
        }
        return existing_keys & requested

    def register_specialization_study(
        self,
        *,
        study_id: str,
        created_at: str,
        benchmark_group_id: str,
        objective_name: str,
        objective_metric: str,
        constraints: dict[str, Any],
        search_space: dict[str, Any],
        notes: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_specialization_studies(
                    study_id, created_at, benchmark_group_id, objective_name,
                    objective_metric, constraint_json, search_space_json, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(study_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    benchmark_group_id=excluded.benchmark_group_id,
                    objective_name=excluded.objective_name,
                    objective_metric=excluded.objective_metric,
                    constraint_json=excluded.constraint_json,
                    search_space_json=excluded.search_space_json,
                    notes=excluded.notes
                """,
                (
                    study_id,
                    created_at,
                    benchmark_group_id,
                    objective_name,
                    objective_metric,
                    _json(constraints),
                    _json(search_space),
                    notes,
                ),
            )

    def register_specialization_trial(
        self,
        *,
        trial_id: str,
        study_id: str,
        base_case_id: str,
        specialization_key: str,
        status: str,
        trial_config: dict[str, Any],
        result_summary: dict[str, Any],
        created_at: str,
        run_id: str | None = None,
        objective_value: float | None = None,
        constraint_status: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_specialization_trials(
                    trial_id, study_id, base_case_id, specialization_key, run_id,
                    status, objective_value, constraint_status, trial_config_json,
                    result_summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trial_id) DO UPDATE SET
                    study_id=excluded.study_id,
                    base_case_id=excluded.base_case_id,
                    specialization_key=excluded.specialization_key,
                    run_id=excluded.run_id,
                    status=excluded.status,
                    objective_value=excluded.objective_value,
                    constraint_status=excluded.constraint_status,
                    trial_config_json=excluded.trial_config_json,
                    result_summary_json=excluded.result_summary_json,
                    created_at=excluded.created_at
                """,
                (
                    trial_id,
                    study_id,
                    base_case_id,
                    specialization_key,
                    run_id,
                    status,
                    objective_value,
                    constraint_status,
                    _json(trial_config),
                    _json(result_summary),
                    created_at,
                ),
            )

    def update_specialization_trial_result(
        self,
        *,
        trial_id: str,
        run_id: str,
        status: str,
        objective_value: float | None,
        constraint_status: str | None,
        result_summary: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE benchmark_specialization_trials
                SET run_id = ?,
                    status = ?,
                    objective_value = ?,
                    constraint_status = ?,
                    result_summary_json = ?
                WHERE trial_id = ?
                """,
                (
                    run_id,
                    status,
                    objective_value,
                    constraint_status,
                    _json(result_summary),
                    trial_id,
                ),
            )

    def fetch_runs_matching(
        self,
        filters: dict[str, Any],
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            if value is None:
                clauses.append(f"{key} IS NULL")
            else:
                clauses.append(f"{key} = ?")
                params.append(value)
        where = " AND ".join(clauses) if clauses else "1 = 1"
        sql = (
            f"SELECT * FROM benchmark_runs WHERE {where} "
            "ORDER BY finished_at DESC LIMIT ?"
        )
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def fetch_runs_by_group(self, benchmark_group_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs WHERE benchmark_group_id = ? ORDER BY n ASC, d ASC, started_at ASC",
                (benchmark_group_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_runs_by_commit(self, commit_hash: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs WHERE commit_hash = ? ORDER BY started_at ASC",
                (commit_hash,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_runs_by_specialization_key(self, specialization_key: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs WHERE specialization_key = ? ORDER BY started_at ASC",
                (specialization_key,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_runs_by_base_case(self, base_case_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs WHERE base_case_id = ? ORDER BY started_at ASC",
                (base_case_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_runs_by_branch(self, branch_name: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs WHERE branch_name = ? ORDER BY started_at ASC",
                (branch_name,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_study(self, study_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_specialization_studies WHERE study_id = ?",
                (study_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def fetch_trials_for_study(self, study_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_specialization_trials WHERE study_id = ? ORDER BY created_at ASC",
                (study_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM benchmark_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def fetch_runs_for_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM benchmark_runs WHERE session_id = ? ORDER BY started_at ASC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_view_rows(
        self,
        view_name: str,
        *,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in (filters or {}).items():
            if value is None:
                clauses.append(f"{key} IS NULL")
            else:
                clauses.append(f"{key} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = f" ORDER BY {order_by}" if order_by else ""
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT ?"
            params.append(limit)
        sql = f"SELECT * FROM {view_name}{where}{order}{limit_sql}"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def export_session_json(
        self,
        session_id: str,
        *,
        export_root: Path | None = None,
    ) -> Path:
        ensure_storage_dirs()
        if export_root is None:
            export_root = DEFAULT_SESSION_EXPORT_ROOT
        export_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "session": self.fetch_session(session_id),
            "cases": self.fetch_cases_for_session(session_id),
            "datasets": self.fetch_datasets_for_session(session_id),
            "comparisons": self.fetch_comparisons_for_session(session_id),
            "specializations": self.fetch_specializations_for_session(session_id),
            "specialization_studies": self.fetch_specialization_studies_for_session(session_id),
            "specialization_trials": self.fetch_specialization_trials_for_session(session_id),
            "runs": self.fetch_runs_for_session(session_id),
            "parameter_recovery": self.fetch_parameter_recovery_for_session(session_id),
            "artifacts": self.fetch_artifacts_for_session(session_id),
        }
        target = export_root / f"{session_id}.json"
        target.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8")
        return target

    def export_session_bundle(
        self,
        session_id: str,
        *,
        export_root: Path | None = None,
    ) -> Path:
        """Export a deterministic importable benchmark session bundle."""

        return self.export_session_json(session_id, export_root=export_root)
