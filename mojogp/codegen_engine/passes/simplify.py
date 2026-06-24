"""Algebraic simplification pass.

Wraps sympy.simplify() for final cleanup. Applied conservatively
since over-simplification can produce numerically worse expressions.
"""

from ..ir import IRKernel


def simplify_pass(kernel: IRKernel) -> IRKernel:
    """Apply algebraic simplification.

    Currently a no-op — SymPy's simplify() can be very slow on large
    expressions and may produce forms that are numerically worse.
    CSE + strength reduction handle the important cases.
    """
    return kernel
