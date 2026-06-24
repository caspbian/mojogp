"""Benchmark runner integration for multi-output real-data cases."""

from __future__ import annotations

from pathlib import Path

from .harness_runner import run_benchmark_module
from .runtime import BenchmarkRuntimeContext, get_or_create_default_context


def run_multi_output_real_data_case(
    *,
    case: str,
    results_dir: Path,
    context: BenchmarkRuntimeContext | None = None,
) -> dict[str, object]:
    context = context or get_or_create_default_context()
    case_id = f"mojogp.multi_output.real_data.{case}"
    benchmark_group_id = f"mojogp.multi_output.real_data.{case}"
    config = {
        "framework": "mojogp",
        "benchmark": "multi_output_real_data",
        "case": case,
    }
    context.session_store.register_case(
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        framework="mojogp",
        suite_name="multi_output_real_data",
        benchmark_name="multi_output_real_data",
        config=config,
    )
    result = run_benchmark_module(
        module="tests.system_benchmarks.run_multi_output_real_case",
        payload={"case": case, "results_dir": str(results_dir)},
        timeout=1200,
        description=f"Runs multi-output real-data case {case}",
        result_loader=None,
        session_store=context.session_store,
        session_id=context.session_id,
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        benchmark_name="multi_output_real_data",
        framework="mojogp",
        git=context.git,
        profiling=context.profiling,
        config=config,
        dataset_id=f"real_data::{case}",
    )
    return dict(result.loaded_result)
