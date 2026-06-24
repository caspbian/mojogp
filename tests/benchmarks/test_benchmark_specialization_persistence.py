from __future__ import annotations

from tests.benchmarks.preflight import utc_now_iso


def test_session_store_persists_specialization_registry_and_run_queries(benchmark_store):
    benchmark_store.register_specialization(
        specialization_key="rbf_tm1_probe",
        specialization_family="jit_codegen",
        specialization_source="benchmark",
        policy_version="v1",
        config={"schedule_overrides": {"tm": 1}, "ncols_hint": [6, 1]},
        notes="probe",
        created_at=utc_now_iso(),
        active=True,
    )
    benchmark_store.register_case(
        case_id="mojogp.single_output.scaling.matrix_free.exact.n5000.d5.spec.rbf_tm1_probe",
        benchmark_group_id="mojogp.single_output.scaling.matrix_free.exact",
        framework="mojogp",
        suite_name="single_output_scaling",
        benchmark_name="scaling_certification",
        config={
            "base_case_id": "mojogp.single_output.scaling.matrix_free.exact.n5000.d5",
            "specialization_key": "rbf_tm1_probe",
            "specialization_family": "jit_codegen",
            "specialization_mode": "applied",
            "specialization_source": "benchmark",
            "specialization_descriptor": {"kernel_family": "rbf", "is_ard": True},
            "specialization_config": {"schedule_overrides": {"tm": 1}},
        },
    )
    benchmark_store.register_run(
        {
            "run_id": "run-spec-1",
            "session_id": "session-spec-1",
            "case_id": "mojogp.single_output.scaling.matrix_free.exact.n5000.d5.spec.rbf_tm1_probe",
            "benchmark_group_id": "mojogp.single_output.scaling.matrix_free.exact",
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
            "base_case_id": "mojogp.single_output.scaling.matrix_free.exact.n5000.d5",
            "specialization_key": "rbf_tm1_probe",
            "specialization_family": "jit_codegen",
            "specialization_mode": "applied",
            "specialization_source": "benchmark",
            "specialization_descriptor_json": "{\"kernel_family\": \"rbf\"}",
            "specialization_config_json": "{\"schedule_overrides\": {\"tm\": 1}}",
            "dataset_id": "dataset-1",
            "comparison_id": None,
            "artifact_id": "artifact-1",
            "status": "ok",
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "framework": "mojogp",
            "config_json": "{}",
            "benchmark_name": "scaling_certification",
            "result_json_path": None,
            "training_time_s": 1.0,
            "prediction_mean_time_s": 0.1,
            "prediction_variance_time_s": 0.2,
            "end_to_end_time_s": 1.3,
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

    specialization_runs = benchmark_store.fetch_runs_by_specialization_key("rbf_tm1_probe")
    assert len(specialization_runs) == 1
    assert specialization_runs[0]["base_case_id"] == (
        "mojogp.single_output.scaling.matrix_free.exact.n5000.d5"
    )

    base_case_runs = benchmark_store.fetch_runs_by_base_case(
        "mojogp.single_output.scaling.matrix_free.exact.n5000.d5"
    )
    assert len(base_case_runs) == 1
    assert base_case_runs[0]["specialization_mode"] == "applied"
