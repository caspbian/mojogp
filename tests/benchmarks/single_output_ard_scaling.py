"""Benchmark runner integration for single-output ARD scaling cases."""

from __future__ import annotations

from pathlib import Path

from .single_output_scaling import run_single_output_scaling_subprocess


def run_single_output_ard_scaling_subprocess(
    *,
    method: str,
    n_train: int,
    d: int,
    relevant_dims: int,
    framework: str,
    prediction_mode: str,
    tier: str,
    benchmark_variety: str,
    benchmark_track: str,
    n_selection_policy: str | None,
    size_role: str | None,
    max_iterations: int,
    enable_early_stopping: bool,
    mojogp_preset: str | None = None,
    data_options: dict[str, object] | None = None,
    specialization: dict[str, object] | None = None,
    results_dir: Path,
    context=None,
):
    effective_data_options = dict(data_options or {})
    effective_data_options.update(
        {
            "dataset_family": "structured_ard",
            "relevant_dims": int(relevant_dims),
        }
    )
    return run_single_output_scaling_subprocess(
        method=method,
        n_train=n_train,
        d=d,
        framework=framework,
        prediction_mode=prediction_mode,
        tier=tier,
        benchmark_variety=benchmark_variety,
        benchmark_track=benchmark_track,
        n_selection_policy=n_selection_policy,
        size_role=size_role,
        max_iterations=max_iterations,
        enable_early_stopping=enable_early_stopping,
        benchmark_name="single_output_ard_scaling",
        mojogp_preset=mojogp_preset,
        data_options=effective_data_options,
        specialization=specialization,
        results_dir=results_dir,
        context=context,
        ard=True,
        relevant_dims=int(relevant_dims),
    )
