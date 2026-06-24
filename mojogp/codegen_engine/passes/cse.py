"""Common Subexpression Elimination pass.

Wraps sympy.cse() to find shared subexpressions between forward and
all gradient expressions, then converts back to IR with Let bindings.
"""

import sympy as sp
from ..ir import IRKernel, LetBinding, from_sympy


def _ir_to_sympy(ir_expr, param_symbols):
    """Convert IR back to SymPy for CSE. Simplified round-trip."""
    from ..ir import Var, Param, Const, BinOp, UnaryFn, Pow, MaxExpr

    if isinstance(ir_expr, Var):
        return sp.Symbol(ir_expr.name)
    if isinstance(ir_expr, Param):
        return param_symbols[ir_expr.index]
    if isinstance(ir_expr, Const):
        return sp.Float(ir_expr.value)
    if isinstance(ir_expr, BinOp):
        l = _ir_to_sympy(ir_expr.left, param_symbols)
        r = _ir_to_sympy(ir_expr.right, param_symbols)
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
        }
        return ops[ir_expr.op](l, r)
    if isinstance(ir_expr, UnaryFn):
        arg = _ir_to_sympy(ir_expr.arg, param_symbols)
        fn_map = {
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "log": sp.log,
            "abs": sp.Abs,
        }
        return fn_map[ir_expr.fn](arg)
    if isinstance(ir_expr, Pow):
        return _ir_to_sympy(ir_expr.base, param_symbols) ** _ir_to_sympy(
            ir_expr.exp, param_symbols
        )
    if isinstance(ir_expr, MaxExpr):
        return sp.Max(
            _ir_to_sympy(ir_expr.left, param_symbols),
            _ir_to_sympy(ir_expr.right, param_symbols),
        )
    raise ValueError(f"Unknown IR node: {type(ir_expr)}")


def cse_pass(kernel: IRKernel) -> IRKernel:
    """Extract common subexpressions shared between forward and all gradients."""
    # Create param symbols for round-trip
    param_symbols = [
        sp.Symbol(f"p_{i}", positive=True) for i in range(kernel.num_params)
    ]
    param_map = {
        s: (i, kernel.param_names[i] if i < len(kernel.param_names) else "")
        for i, s in enumerate(param_symbols)
    }

    # Collect all expressions
    all_ir = [kernel.forward] + [kernel.gradients[i] for i in sorted(kernel.gradients)]

    # Convert to SymPy
    all_sympy = [_ir_to_sympy(e, param_symbols) for e in all_ir]

    # Run CSE
    replacements, reduced = sp.cse(all_sympy)

    # Convert back to IR
    lets = [
        LetBinding(str(sym), from_sympy(val, param_map)) for sym, val in replacements
    ]

    forward_ir = from_sympy(reduced[0], param_map)
    gradients_ir = {
        i: from_sympy(reduced[1 + i], param_map) for i in range(len(kernel.gradients))
    }

    return IRKernel(
        forward=forward_ir,
        gradients=gradients_ir,
        num_params=kernel.num_params,
        param_names=kernel.param_names,
        needs_diffs=kernel.needs_diffs,
        needs_dist_sq=kernel.needs_dist_sq,
        needs_dot=kernel.needs_dot,
        dim=kernel.dim,
        lets=lets,
    )
