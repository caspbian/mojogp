"""Strength reduction pass.

Replaces expensive operations with cheaper equivalents:
- pow(x, 2) -> x * x
- pow(x, 3) -> x * x * x
- pow(x, 0.5) -> sqrt(x)
- pow(x, -1) -> 1 / x
- x * 1 -> x
- x + 0 -> x
"""

from ..ir import IRKernel, IRExpr, Const, BinOp, UnaryFn, Pow, Var, Param, MaxExpr, LetBinding
import math


def _reduce_expr(expr: IRExpr) -> IRExpr:
    """Recursively apply strength reductions to an expression."""
    if isinstance(expr, (Var, Param, Const)):
        return expr

    if isinstance(expr, BinOp):
        left = _reduce_expr(expr.left)
        right = _reduce_expr(expr.right)
        # x * 1 -> x, 1 * x -> x
        if expr.op == "*":
            if isinstance(right, Const) and right.value == 1.0:
                return left
            if isinstance(left, Const) and left.value == 1.0:
                return right
            if isinstance(right, Const) and right.value == 0.0:
                return Const(0.0)
            if isinstance(left, Const) and left.value == 0.0:
                return Const(0.0)
        # x + 0 -> x, 0 + x -> x
        if expr.op == "+":
            if isinstance(right, Const) and right.value == 0.0:
                return left
            if isinstance(left, Const) and left.value == 0.0:
                return right
        return BinOp(expr.op, left, right)

    if isinstance(expr, UnaryFn):
        arg = _reduce_expr(expr.arg)
        return UnaryFn(expr.fn, arg)

    if isinstance(expr, Pow):
        base = _reduce_expr(expr.base)
        exp_val = _reduce_expr(expr.exp)
        if isinstance(exp_val, Const):
            v = exp_val.value
            if v == 2.0:
                return BinOp("*", base, base)
            if v == 3.0:
                return BinOp("*", BinOp("*", base, base), base)
            if v == 0.5:
                return UnaryFn("sqrt", base)
            if v == -1.0:
                return BinOp("/", Const(1.0), base)
            if v == 1.0:
                return base
            if v == 0.0:
                return Const(1.0)
        return Pow(base, exp_val)

    if isinstance(expr, MaxExpr):
        return MaxExpr(_reduce_expr(expr.left), _reduce_expr(expr.right))

    return expr


def strength_reduce_pass(kernel: IRKernel) -> IRKernel:
    """Apply strength reductions to forward and all gradient expressions."""
    return IRKernel(
        forward=_reduce_expr(kernel.forward),
        gradients={i: _reduce_expr(g) for i, g in kernel.gradients.items()},
        num_params=kernel.num_params,
        param_names=kernel.param_names,
        needs_diffs=kernel.needs_diffs,
        needs_dist_sq=kernel.needs_dist_sq,
        needs_dot=kernel.needs_dot,
        dim=kernel.dim,
        lets=[LetBinding(let.name, _reduce_expr(let.value)) for let in kernel.lets],
    )
