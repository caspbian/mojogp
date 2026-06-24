"""GPyTorch single-output scaling benchmark wrapper."""

from __future__ import annotations

from tests.shared.benchmarking.report import load_benchmark_result

from tests.benchmarks.harness_runner import run_benchmark_module


def run_gpytorch_single_output_scaling_module(
    *,
    payload: dict[str, object],
    session_store,
    session_id: str,
    case_id: str,
    benchmark_group_id: str,
    benchmark_name: str,
    git,
    profiling,
    config: dict[str, object],
    dataset_id: str,
    comparison_id: str | None,
    timeout_s: int | None = None,
):
    return run_benchmark_module(
        module="tests.system_benchmarks.run_single_output_scaling_case",
        payload=payload,
        timeout=3600 if timeout_s is None else int(timeout_s),
        description=(
            f"Runs GPyTorch single-output scaling case prediction={payload['prediction_mode']} "
            f"method={payload['method']} n={payload['n_train']} d={payload['d']}"
        ),
        result_loader=load_benchmark_result,
        session_store=session_store,
        session_id=session_id,
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        benchmark_name=benchmark_name,
        framework="gpytorch",
        git=git,
        profiling=profiling,
        config=dict(config),
        dataset_id=dataset_id,
        comparison_id=comparison_id,
    )
