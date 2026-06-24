from __future__ import annotations

from tests.benchmarks.preflight import utc_now_iso
from tests.benchmarks.specialization_adapter import prepare_specialization_payload
from tests.benchmarks.specialization_studies import (
    evaluate_constraints,
    finalize_trial_from_run,
    merge_study_trial_config,
    register_study,
    register_trial,
)


def test_register_study_and_trial_roundtrip(benchmark_store):
    study_id = register_study(
        benchmark_store,
        benchmark_group_id="mojogp.single_output.scaling.matrix_free.exact",
        objective_name="minimize training time",
        objective_metric="training_time_s",
        constraints={"max_metrics": {"training_time_s": 5.0}},
        search_space={"tm": [1, 2, 4]},
        notes="study notes",
    )
    trial_id = register_trial(
        benchmark_store,
        study_id=study_id,
        base_case_id="mojogp.single_output.scaling.matrix_free.exact.n5000.d5",
        specialization_key="rbf_tm1_probe",
        trial_config={"tm": 1},
    )

    study = benchmark_store.fetch_study(study_id)
    trials = benchmark_store.fetch_trials_for_study(study_id)

    assert study is not None
    assert study["objective_metric"] == "training_time_s"
    assert len(trials) == 1
    assert trials[0]["trial_id"] == trial_id
    assert trials[0]["status"] == "registered"


def test_finalize_trial_from_run_updates_result_summary(benchmark_store):
    study_id = register_study(
        benchmark_store,
        benchmark_group_id="mojogp.single_output.scaling.matrix_free.exact",
        objective_name="minimize training time",
        objective_metric="training_time_s",
        constraints={"max_metrics": {"training_time_s": 2.0}},
        search_space={"tm": [1]},
    )
    trial_id = register_trial(
        benchmark_store,
        study_id=study_id,
        base_case_id="mojogp.single_output.scaling.matrix_free.exact.n5000.d5",
        specialization_key="rbf_tm1_probe",
        trial_config={"tm": 1},
    )
    summary = finalize_trial_from_run(
        benchmark_store,
        trial_id=trial_id,
        run_row={
            "run_id": "run-1",
            "case_id": "case-1",
            "base_case_id": "mojogp.single_output.scaling.matrix_free.exact.n5000.d5",
            "specialization_key": "rbf_tm1_probe",
            "training_time_s": 1.8,
        },
        objective_metric="training_time_s",
        constraints={"max_metrics": {"training_time_s": 2.0}},
    )

    trials = benchmark_store.fetch_trials_for_study(study_id)
    assert summary["constraint_evaluation"]["passed"] is True
    assert trials[0]["run_id"] == "run-1"
    assert trials[0]["status"] == "completed"
    assert trials[0]["constraint_status"] == "passed"


def test_merge_study_trial_config_and_constraints_helpers():
    merged = merge_study_trial_config(
        {"n": 5000},
        study_id="study-1",
        trial_id="trial-1",
        objective_name="minimize training time",
        objective_metric="training_time_s",
        constraint_json={"max_metrics": {"training_time_s": 2.0}},
    )
    evaluation = evaluate_constraints(
        {"training_time_s": 1.5, "contract_passed": 1},
        {"max_metrics": {"training_time_s": 2.0}, "equals": {"contract_passed": 1}},
    )

    assert merged["study_id"] == "study-1"
    assert merged["trial_id"] == "trial-1"
    assert evaluation["passed"] is True


def test_prepare_specialization_payload_registers_non_default_profile(benchmark_store):
    payload = prepare_specialization_payload(
        benchmark_store,
        {
            "specialization_mode": "applied",
            "specialization_key": "rbf_tm1_probe",
            "specialization_family": "jit_codegen",
            "specialization_source": "benchmark",
            "specialization_config": {"schedule_overrides": {"tm": 1}},
        },
        created_at=utc_now_iso(),
    )

    rows = benchmark_store.fetch_runs_by_specialization_key("rbf_tm1_probe")
    study = payload["specialization_key"]
    assert study == "rbf_tm1_probe"
    assert rows == []
