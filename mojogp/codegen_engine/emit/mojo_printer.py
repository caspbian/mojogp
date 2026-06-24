"""IR -> Mojo source string emission.

Converts IR expression nodes to Mojo source code strings.
Also provides math import collection from IR expressions.
"""

from ..ir import (
    IRExpr,
    IRKernel,
    Var,
    Param,
    Const,
    BinOp,
    UnaryFn,
    Pow,
    MaxExpr,
    LetBinding,
    collect_functions_all,
)


def emit_ir(expr: IRExpr) -> str:
    """Convert an IR expression to a Mojo source string."""
    if isinstance(expr, Var):
        return expr.name

    if isinstance(expr, Param):
        return f"p[{expr.index}]"

    if isinstance(expr, Const):
        v = expr.value
        if v == int(v) and abs(v) < 1e15:
            return f"Float32({int(v)})"
        return f"Float32({v})"

    if isinstance(expr, BinOp):
        left = emit_ir(expr.left)
        right = emit_ir(expr.right)
        return f"({left} {expr.op} {right})"

    if isinstance(expr, UnaryFn):
        arg = emit_ir(expr.arg)
        fn_map = {
            "exp": "math_exp",
            "sqrt": "sqrt",
            "sin": "sin",
            "cos": "cos",
            "log": "log",
            "abs": "abs",
        }
        mojo_fn = fn_map.get(expr.fn, expr.fn)
        return f"{mojo_fn}({arg})"

    if isinstance(expr, Pow):
        base = emit_ir(expr.base)
        exp_val = emit_ir(expr.exp)
        return f"pow({base}, {exp_val})"

    if isinstance(expr, MaxExpr):
        left = emit_ir(expr.left)
        right = emit_ir(expr.right)
        return f"max({left}, {right})"

    raise ValueError(f"Unknown IR node type: {type(expr)}")


def emit_let_bindings(lets: list, indent: str = "    ") -> str:
    """Emit Let bindings as Mojo variable declarations."""
    lines = []
    for let in lets:
        lines.append(f"{indent}var {let.name} = {emit_ir(let.value)}")
    return "\n".join(lines)


def collect_math_imports(kernel: IRKernel) -> str:
    """Generate the Mojo math import line based on functions used in the IR.

    Scans all IR nodes including LetBindings for function references.
    Always includes exp and sqrt (needed by virtually every kernel).
    """
    # Collect from all expressions including let bindings
    fns = collect_functions_all(kernel)

    # Also scan let bindings which may contain the actual function calls
    if hasattr(kernel, "lets") and kernel.lets:
        for let in kernel.lets:
            _collect_fns_recursive(let.value, fns)
    # Scan forward/gradient exprs recursively too
    _collect_fns_recursive(kernel.forward, fns)
    for g in kernel.gradients:
        if isinstance(g, IRExpr):
            _collect_fns_recursive(g, fns)

    # Map IR function names to Mojo import names
    # Note: pow and abs are builtins in Mojo, not in math module
    import_map = {
        "exp": "exp as math_exp",
        "sqrt": "sqrt",
        "sin": "sin",
        "cos": "cos",
        "log": "log",
    }

    imports = sorted(import_map[f] for f in fns if f in import_map)

    # Always include exp and sqrt — needed by all GP kernels
    if "exp as math_exp" not in imports:
        imports.append("exp as math_exp")
    if "sqrt" not in imports:
        imports.append("sqrt")
    imports = sorted(set(imports))

    return f"from math import {', '.join(imports)}"


def _collect_fns_recursive(expr, fns: set):
    """Recursively collect function names from an IR expression."""
    if not isinstance(expr, IRExpr):
        return
    if isinstance(expr, UnaryFn):
        fns.add(expr.fn)
        _collect_fns_recursive(expr.arg, fns)
    elif isinstance(expr, Pow):
        fns.add("pow")
        _collect_fns_recursive(expr.base, fns)
        _collect_fns_recursive(expr.exp, fns)
    elif isinstance(expr, BinOp):
        _collect_fns_recursive(expr.left, fns)
        _collect_fns_recursive(expr.right, fns)
    elif isinstance(expr, MaxExpr):
        _collect_fns_recursive(expr.left, fns)
        _collect_fns_recursive(expr.right, fns)
    elif isinstance(expr, LetBinding):
        _collect_fns_recursive(expr.value, fns)
