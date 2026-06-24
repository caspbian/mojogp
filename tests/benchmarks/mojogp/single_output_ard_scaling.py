"""MojoGP single-output ARD scaling benchmark wrapper."""

from __future__ import annotations

from tests.benchmarks.mojogp.single_output_scaling import (
    run_mojogp_single_output_scaling_module,
)


def run_mojogp_single_output_ard_scaling_module(**kwargs):
    return run_mojogp_single_output_scaling_module(**kwargs)
