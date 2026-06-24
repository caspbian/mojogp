"""Benchmark runner integration for multi-output ARD scaling cases."""

from __future__ import annotations

from pathlib import Path

from .multi_output_scaling import run_multi_output_scaling_subprocess


def run_multi_output_ard_scaling_subprocess(
    *,
    framework: str,
    prediction_mode: str,
    method: str,
    n_train: int,
    d: int,
    num_tasks: int,
    relevant_dims: int,
    tier: str,
    specialization: dict[str, object] | None = None,
    results_dir: Path,
    context=None,
):
    return run_multi_output_scaling_subprocess(
        framework=framework,
        prediction_mode=prediction_mode,
        method=method,
        n_train=n_train,
        d=d,
        num_tasks=num_tasks,
        tier=tier,
        specialization=specialization,
        ard=True,
        relevant_dims=int(relevant_dims),
        benchmark_name="multi_output_ard_scaling",
        results_dir=results_dir,
        context=context,
    )
