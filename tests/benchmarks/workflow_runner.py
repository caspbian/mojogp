"""Shared harness runner for workflow-oriented benchmark families."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tests.shared.benchmarking.report import load_benchmark_result, load_comparison_result

from .harness_runner import run_benchmark_module
from .runtime import BenchmarkRuntimeContext, get_or_create_default_context


def run_workflow_benchmark_subprocess(
    *,
    module: str,
    payload: Mapping[str, Any],
    suite_name: str,
    benchmark_name: str,
    framework: str,
    case_id: str,
    benchmark_group_id: str,
    config: dict[str, Any],
    results_dir: Path,
    timeout: int = 1200,
    context: BenchmarkRuntimeContext | None = None,
):
    context = context or get_or_create_default_context()
    context.session_store.register_case(
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        framework=framework,
        suite_name=suite_name,
        benchmark_name=benchmark_name,
        config=config,
    )
    return run_benchmark_module(
        module=module,
        payload={**dict(payload), "results_dir": str(results_dir)},
        timeout=timeout,
        description=f"Runs {benchmark_name} benchmark case",
        result_loader=load_benchmark_result,
        session_store=context.session_store,
        session_id=context.session_id,
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        benchmark_name=benchmark_name,
        framework=framework,
        git=context.git,
        profiling=context.profiling,
        config=config,
        dataset_id=None,
        comparison_id=None,
    ).loaded_result


def run_result_benchmark_subprocess(
    *,
    module: str,
    payload: Mapping[str, Any],
    suite_name: str,
    benchmark_name: str,
    framework: str,
    case_id: str,
    benchmark_group_id: str,
    config: dict[str, Any],
    results_dir: Path,
    timeout: int = 1200,
    context: BenchmarkRuntimeContext | None = None,
):
    context = context or get_or_create_default_context()
    context.session_store.register_case(
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        framework=framework,
        suite_name=suite_name,
        benchmark_name=benchmark_name,
        config=config,
    )
    return run_benchmark_module(
        module=module,
        payload={**dict(payload), "results_dir": str(results_dir)},
        timeout=timeout,
        description=f"Runs {benchmark_name} benchmark case",
        result_loader=load_benchmark_result,
        session_store=context.session_store,
        session_id=context.session_id,
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        benchmark_name=benchmark_name,
        framework=framework,
        git=context.git,
        profiling=context.profiling,
        config=config,
        dataset_id=None,
        comparison_id=None,
    ).loaded_result


def run_comparison_benchmark_subprocess(
    *,
    module: str,
    payload: Mapping[str, Any],
    suite_name: str,
    benchmark_name: str,
    framework: str,
    case_id: str,
    benchmark_group_id: str,
    config: dict[str, Any],
    results_dir: Path,
    timeout: int = 1200,
    context: BenchmarkRuntimeContext | None = None,
):
    context = context or get_or_create_default_context()
    context.session_store.register_case(
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        framework=framework,
        suite_name=suite_name,
        benchmark_name=benchmark_name,
        config=config,
    )
    return run_benchmark_module(
        module=module,
        payload={**dict(payload), "results_dir": str(results_dir)},
        timeout=timeout,
        description=f"Runs {benchmark_name} comparison benchmark case",
        result_loader=load_comparison_result,
        session_store=context.session_store,
        session_id=context.session_id,
        case_id=case_id,
        benchmark_group_id=benchmark_group_id,
        benchmark_name=benchmark_name,
        framework=framework,
        git=context.git,
        profiling=context.profiling,
        config=config,
        dataset_id=None,
        comparison_id=None,
    ).loaded_result
