"""GPyTorch multi-output scaling benchmark wrapper."""

from __future__ import annotations

from tests.shared.benchmarking.report import load_benchmark_result

from tests.benchmarks.harness_runner import run_benchmark_module
from tests.benchmarks.multi_output_timeout_policy import multi_output_scaling_timeout_s


def run_gpytorch_multi_output_scaling_module(
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
    effective_timeout_s = multi_output_scaling_timeout_s(
        framework="gpytorch",
        method=str(payload["method"]),
        prediction_mode=str(payload["prediction_mode"]),
        tier=str(payload.get("tier", "")),
        timeout_s=timeout_s,
    )
    effective_config = dict(config)
    effective_config.setdefault("timeout_s", effective_timeout_s)
    return run_benchmark_module(
        module="tests.system_benchmarks.run_multi_output_scaling_case",
        payload=payload,
        timeout=effective_timeout_s,
        description=(
            f"Runs GPyTorch multi-output scaling case prediction={payload['prediction_mode']} "
            f"method={payload['method']} n={payload['n_train']} d={payload['d']} tasks={payload['num_tasks']}"
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
        config=effective_config,
        dataset_id=dataset_id,
        comparison_id=comparison_id,
    )
