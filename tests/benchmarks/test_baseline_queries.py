from __future__ import annotations

from tests.benchmarks.baseline_queries import fetch_historical_metric_values, historical_metric_baseline
from tests.benchmarks.contracts import HistoricalBaselineKey


def _run_row(*, run_id: str, case_id: str, benchmark_group_id: str, commit_hash: str, training_time_s: float) -> dict[str, object]:
    return {
        "run_id": run_id,
        "session_id": "s1",
        "case_id": case_id,
        "benchmark_group_id": benchmark_group_id,
        "n": 5000,
        "d": 5,
        "num_tasks": None,
        "kernel": "rbf",
        "model_type": "SingleOutputGP",
        "training_method": "materialized",
        "prediction_mode": "exact",
        "comparison_class": None,
        "baseline_backend": None,
        "fairness_note": None,
        "artifact_id": f"artifact-{run_id}",
        "dataset_id": None,
        "comparison_id": None,
        "status": "ok",
        "started_at": "2026-04-21T00:00:00",
        "finished_at": "2026-04-21T00:00:01",
        "framework": "mojogp",
        "config_json": "{}",
        "benchmark_name": "single_output_truth_harness",
        "result_json_path": None,
        "training_time_s": training_time_s,
        "prediction_mean_time_s": None,
        "prediction_variance_time_s": None,
        "end_to_end_time_s": None,
        "training_peak_gpu_mb": None,
        "training_delta_gpu_mb": None,
        "prediction_peak_gpu_mb": None,
        "prediction_delta_gpu_mb": None,
        "scaling_peak_gpu_mb": None,
        "scaling_memory_metric": None,
        "gpu_baseline_mb": None,
        "gpu_current_mb": None,
        "gpu_delta_mb": None,
        "gpu_max_mb": None,
        "gpu_isolated_peak_mb": None,
        "gpu_isolated_current_mb": None,
        "gpu_samples": None,
        "measurement_method_primary": None,
        "torch_baseline_mb": None,
        "torch_peak_mb": None,
        "torch_peak_delta_mb": None,
        "torch_current_delta_mb": None,
        "torch_reserved_delta_mb": None,
        "cpu_peak_mb": None,
        "branch_name": "main",
        "commit_hash": commit_hash,
        "git_clean": 1,
        "profiling_probe_passed": 1,
    }


def test_fetch_historical_metric_values_filters_by_case_identity(benchmark_store):
    benchmark_store.register_run(_run_row(run_id="r1", case_id="case.a", benchmark_group_id="group.a", commit_hash="abc", training_time_s=1.0))
    benchmark_store.register_run(_run_row(run_id="r2", case_id="case.b", benchmark_group_id="group.b", commit_hash="def", training_time_s=3.0))

    values = fetch_historical_metric_values(
        benchmark_store,
        key=HistoricalBaselineKey(
            benchmark_name="single_output_truth_harness",
            framework="mojogp",
            case_id="case.a",
            training_method="materialized",
            prediction_mode="exact",
        ),
        metric="training_time_s",
    )

    assert values == [1.0]


def test_historical_metric_baseline_returns_rolling_median(benchmark_store):
    for idx, value in enumerate([5.0, 1.0, 9.0], start=1):
        benchmark_store.register_run(
            _run_row(
                run_id=f"r{idx}",
                case_id="case.a",
                benchmark_group_id="group.a",
                commit_hash=f"hash{idx}",
                training_time_s=value,
            )
        )

    baseline = historical_metric_baseline(
        benchmark_store,
        key=HistoricalBaselineKey(
            benchmark_name="single_output_truth_harness",
            framework="mojogp",
            benchmark_group_id="group.a",
            training_method="materialized",
            prediction_mode="exact",
        ),
        metric="training_time_s",
    )

    assert baseline == 5.0
