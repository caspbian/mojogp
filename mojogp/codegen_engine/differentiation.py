"""Symbolic gradient computation for kernel expressions.

Uses SymPy's diff() for automatic differentiation, with an override
system for numerically sensitive cases (e.g., Matern at r=0).
"""

import sympy as sp
from dataclasses import dataclass
from typing import Optional

from .expressions import KernelExpr


@dataclass
class GradientExpr:
    """Forward expression + all parameter gradients."""

    forward: sp.Expr
    gradients: dict  # sp.Symbol -> sp.Expr (param -> dk/dparam)
    params: list  # ordered parameter symbols
    shared: dict  # shared intermediates from KernelExpr
    param_layout: list  # human-readable param names


def compute_gradients(
    kernel_expr: KernelExpr,
    overrides: Optional[dict] = None,
) -> GradientExpr:
    """Compute dk/d(param) for every parameter via SymPy diff().

    Args:
        kernel_expr: The kernel expression to differentiate.
        overrides: Optional dict mapping sp.Symbol -> sp.Expr for manual gradient overrides.
            Used for numerically sensitive cases where SymPy's auto-diff produces
            expressions that are 0/0 at certain points.

    Returns:
        GradientExpr with forward value and all parameter gradients.
    """
    grads = {}
    for param in kernel_expr.params:
        if overrides and param in overrides:
            grads[param] = overrides[param]
        else:
            grads[param] = sp.diff(kernel_expr.forward, param)

    return GradientExpr(
        forward=kernel_expr.forward,
        gradients=grads,
        params=kernel_expr.params,
        shared=kernel_expr.shared,
        param_layout=kernel_expr.param_layout,
    )
