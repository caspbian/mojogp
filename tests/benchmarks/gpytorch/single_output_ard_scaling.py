"""GPyTorch single-output ARD scaling benchmark wrapper."""

from __future__ import annotations

from tests.benchmarks.gpytorch.single_output_scaling import (
    run_gpytorch_single_output_scaling_module,
)


def run_gpytorch_single_output_ard_scaling_module(**kwargs):
    return run_gpytorch_single_output_scaling_module(**kwargs)
