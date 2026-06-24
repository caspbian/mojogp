"""Override registry for kernel-specific gradient and schedule customizations.

Some kernel gradients have numerical issues at specific points (e.g., Matern
at r=0 produces 0/0). The override system lets us replace SymPy's auto-diff
with hand-crafted expressions that handle these edge cases.
"""

import sympy as sp
from typing import Optional

from .expressions import KernelExpr


class GradientOverride:
    """Base class for kernel-specific gradient overrides."""

    def applies_to(self, kernel_type: str, param_name: str) -> bool:
        raise NotImplementedError

    def compute(self, kernel_expr: KernelExpr, param: sp.Symbol) -> sp.Expr:
        raise NotImplementedError


class MaternDistClampOverride(GradientOverride):
    """For Matern kernels: clamp r to avoid 1/0 at dist=0 in ARD lengthscale gradients.

    SymPy's diff of sqrt(x) at x=0 gives 1/(2*sqrt(0)) = inf.
    We replace with Max(r, eps) in the gradient expression.
    """

    def applies_to(self, kernel_type: str, param_name: str) -> bool:
        return kernel_type in (
            "matern12",
            "matern32",
            "matern52",
        ) and param_name.startswith("l_")

    def compute(self, kernel_expr: KernelExpr, param: sp.Symbol) -> sp.Expr:
        # Clamping is handled during IR emission.
        return sp.diff(kernel_expr.forward, param)


class PolynomialDegreeFixedOverride(GradientOverride):
    """Polynomial degree is kernel structure, not a learnable hyperparameter."""

    def applies_to(self, kernel_type: str, param_name: str) -> bool:
        return kernel_type == "polynomial" and param_name == "degree"

    def compute(self, kernel_expr: KernelExpr, param: sp.Symbol) -> sp.Expr:
        return sp.Integer(0)


# Global registry of gradient overrides
OVERRIDE_REGISTRY: list[GradientOverride] = [
    MaternDistClampOverride(),
    PolynomialDegreeFixedOverride(),
]


def get_overrides_for_kernel(kernel_expr: KernelExpr) -> Optional[dict]:
    """Get gradient overrides applicable to this kernel expression.

    Returns:
        Dict mapping param symbol -> override expression, or None if no overrides apply.
    """
    overrides = {}
    kt = kernel_expr.kernel_type
    if kt is None:
        return None

    for i, param in enumerate(kernel_expr.params):
        param_name = (
            kernel_expr.param_layout[i] if i < len(kernel_expr.param_layout) else ""
        )
        for override in OVERRIDE_REGISTRY:
            if override.applies_to(kt, param_name):
                overrides[param] = override.compute(kernel_expr, param)
                break

    return overrides if overrides else None
