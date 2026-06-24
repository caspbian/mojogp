from __future__ import annotations

import pytest

from tests.benchmarks.preflight import utc_now_iso
from tests.benchmarks.specialization_queries import (
    default_and_variants,
    group_variants,
    guard_lane_regressions,
    runs_for_base_case,
    runs_for_specialization,
    variant_summary,
)


def _register_run(
    benchmark_store,
    *,
    run_id: str,
    case_id: str,
    base_case_id: str,
    specialization_key: str | None,
    specialization_mode: str | None,
    training_time_s: float,
    benchmark_group_id: str = "mojogp.single_output.scaling.matrix_free.exact",
):
    benchmark_store.register_run(
        {
            "run_id": run_id,
            "session_id": "session-1",
            "case_id": case_id,
            "benchmark_group_id": benchmark_group_id,
            "n": 5000,
            "d": 5,
            "num_tasks": None,
            "kernel": "rbf",
            "model_type": "SingleOutputGP",
            "training_method": "matrix_free",
            "prediction_mode": "exact",
            "comparison_class": None,
            "baseline_backend": None,
            "fairness_note": None,
            "base_case_id": base_case_id,
            "specialization_key": specialization_key,
            "specialization_family": None if specialization_key is None else "jit_codegen",
            "specialization_mode": specialization_mode,
            "specialization_source": None if specialization_key is None else "benchmark",
            "specialization_descriptor_json": "{}",
            "specialization_config_json": "{}",
            "dataset_id": "dataset-1",
            "comparison_id": None,
            "artifact_id": f"artifact-{run_id}",
            "status": "ok",
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "framework": "mojogp",
            "config_json": "{}",
            "benchmark_name": "scaling_certification",
            "result_json_path": None,
            "training_time_s": training_time_s,
            "prediction_mean_time_s": 0.1,
            "prediction_variance_time_s": 0.2,
            "end_to_end_time_s": training_time_s + 0.3,
            "contract_passed": 1,
            "contract_summary_json": "{}",
            "training_peak_gpu_mb": 10.0,
            "training_delta_gpu_mb": 5.0,
            "prediction_peak_gpu_mb": 9.0,
            "prediction_delta_gpu_mb": 2.0,
            "scaling_peak_gpu_mb": 5.0,
            "scaling_memory_metric": "training_delta_gpu_mb",
            "gpu_baseline_mb": 0.0,
            "gpu_current_mb": 0.0,
            "gpu_delta_mb": 0.0,
            "gpu_max_mb": 0.0,
            "gpu_isolated_peak_mb": 0.0,
            "gpu_isolated_current_mb": 0.0,
            "gpu_samples": 0,
            "measurement_method_primary": "none",
            "torch_baseline_mb": 0.0,
            "torch_peak_mb": 0.0,
            "torch_peak_delta_mb": 0.0,
            "torch_current_delta_mb": 0.0,
            "torch_reserved_delta_mb": 0.0,
            "cpu_peak_mb": 0.0,
            "branch_name": "feature/test",
            "commit_hash": "abc123",
            "git_clean": 1,
            "profiling_probe_passed": 1,
        }
    )


def test_default_and_variants_groups_rows_by_base_case(benchmark_store):
    base_case_id = "mojogp.single_output.scaling.matrix_free.exact.n5000.d5"
    _register_run(
        benchmark_store,
        run_id="default-1",
        case_id=base_case_id,
        base_case_id=base_case_id,
        specialization_key="default",
        specialization_mode="disabled",
        training_time_s=2.0,
    )
    _register_run(
        benchmark_store,
        run_id="variant-1",
        case_id=f"{base_case_id}.spec.rbf_tm1",
        base_case_id=base_case_id,
        specialization_key="rbf_tm1",
        specialization_mode="applied",
        training_time_s=1.5,
    )

    grouped = default_and_variants(base_case_id, db_path=benchmark_store.db_path)

    assert grouped["default_run"] is not None
    assert grouped["default_run"]["run_id"] == "default-1"
    assert len(grouped["variant_runs"]) == 1
    assert grouped["variant_runs"][0]["specialization_key"] == "rbf_tm1"


def test_variant_summary_reports_best_variant_and_deltas(benchmark_store):
    base_case_id = "mojogp.single_output.scaling.matrix_free.exact.n5000.d5"
    _register_run(
        benchmark_store,
        run_id="default-2",
        case_id=base_case_id,
        base_case_id=base_case_id,
        specialization_key="default",
        specialization_mode="disabled",
        training_time_s=2.0,
    )
    _register_run(
        benchmark_store,
        run_id="variant-2a",
        case_id=f"{base_case_id}.spec.rbf_tm1",
        base_case_id=base_case_id,
        specialization_key="rbf_tm1",
        specialization_mode="applied",
        training_time_s=1.4,
    )
    _register_run(
        benchmark_store,
        run_id="variant-2b",
        case_id=f"{base_case_id}.spec.rbf_tm2",
        base_case_id=base_case_id,
        specialization_key="rbf_tm2",
        specialization_mode="applied",
        training_time_s=1.7,
    )

    summary = variant_summary(base_case_id, db_path=benchmark_store.db_path)

    assert summary["best_variant"] is not None
    assert summary["best_variant"]["specialization_key"] == "rbf_tm1"
    assert summary["best_variant"]["objective_delta"] == pytest.approx(-0.6)


def test_group_variants_groups_rows_by_base_case(benchmark_store):
    base_case_a = "mojogp.single_output.scaling.matrix_free.exact.n5000.d5"
    base_case_b = "mojogp.single_output.scaling.matrix_free.exact.n10000.d5"
    _register_run(
        benchmark_store,
        run_id="group-a",
        case_id=base_case_a,
        base_case_id=base_case_a,
        specialization_key="default",
        specialization_mode="disabled",
        training_time_s=2.0,
    )
    _register_run(
        benchmark_store,
        run_id="group-b",
        case_id=base_case_b,
        base_case_id=base_case_b,
        specialization_key="rbf_tm1",
        specialization_mode="applied",
        training_time_s=3.0,
    )

    grouped = group_variants(
        "mojogp.single_output.scaling.matrix_free.exact",
        db_path=benchmark_store.db_path,
    )

    assert set(grouped) == {base_case_a, base_case_b}
    assert grouped[base_case_b][0]["run_id"] == "group-b"


def test_specialization_and_base_case_queries_decode_rows(benchmark_store):
    base_case_id = "mojogp.single_output.scaling.matrix_free.exact.n5000.d5"
    _register_run(
        benchmark_store,
        run_id="decode-1",
        case_id=f"{base_case_id}.spec.rbf_tm1",
        base_case_id=base_case_id,
        specialization_key="rbf_tm1",
        specialization_mode="applied",
        training_time_s=1.5,
    )

    by_spec = runs_for_specialization("rbf_tm1", db_path=benchmark_store.db_path)
    by_case = runs_for_base_case(base_case_id, db_path=benchmark_store.db_path)

    assert by_spec[0]["specialization_config_json"] == {}
    assert by_case[0]["contract_summary_json"] == {}


def test_guard_lane_regressions_flags_slower_guard_variants(benchmark_store):
    guard_case_id = "mojogp.single_output.scaling.matrix_free.exact.n10000.d5"
    _register_run(
        benchmark_store,
        run_id="guard-default",
        case_id=guard_case_id,
        base_case_id=guard_case_id,
        specialization_key="default",
        specialization_mode="disabled",
        training_time_s=2.0,
    )
    _register_run(
        benchmark_store,
        run_id="guard-variant",
        case_id=f"{guard_case_id}.spec.rbf_tm1",
        base_case_id=guard_case_id,
        specialization_key="rbf_tm1",
        specialization_mode="applied",
        training_time_s=2.4,
    )

    regressions = guard_lane_regressions(
        "rbf_tm1",
        [guard_case_id],
        db_path=benchmark_store.db_path,
    )

    assert len(regressions) == 1
    assert regressions[0]["metric_delta"] == pytest.approx(0.4)
    assert regressions[0]["regressed"] is True
