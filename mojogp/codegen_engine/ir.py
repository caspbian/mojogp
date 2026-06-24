"""Lightweight dataclass IR for kernel expressions.

Decouples the math layer (SymPy) from the optimization passes and emitter.
All downstream code operates on this IR, not SymPy objects.

IR nodes are frozen dataclasses — no surprises, easy to pattern-match.
"""

from dataclasses import dataclass, field
from typing import Union, Optional
import sympy as sp


# =========================================================================
# IR Node Types
# =========================================================================


@dataclass(frozen=True)
class Var:
    """Named variable (diff_0, dist_sq, x_row_3, etc.)."""

    name: str


@dataclass(frozen=True)
class Param:
    """Parameter from the flat param array: p[index]."""

    index: int
    name: str = ""


@dataclass(frozen=True)
class Const:
    """Literal floating-point constant."""

    value: float


@dataclass(frozen=True)
class BinOp:
    """Binary operation: +, -, *, /."""

    op: str
    left: "IRExpr"
    right: "IRExpr"


@dataclass(frozen=True)
class UnaryFn:
    """Unary function: exp, sqrt, sin, cos, log, abs."""

    fn: str
    arg: "IRExpr"


@dataclass(frozen=True)
class Pow:
    """Power: base ** exp."""

    base: "IRExpr"
    exp: "IRExpr"


@dataclass(frozen=True)
class MaxExpr:
    """Max of two expressions."""

    left: "IRExpr"
    right: "IRExpr"


@dataclass(frozen=True)
class LetBinding:
    """Named temporary: var name = value (used for CSE results)."""

    name: str
    value: "IRExpr"


# Union type for all IR expression nodes
IRExpr = Union[Var, Param, Const, BinOp, UnaryFn, Pow, MaxExpr]


@dataclass
class IRKernel:
    """Complete kernel IR: forward + gradients + metadata."""

    forward: IRExpr
    gradients: dict  # int (param_index) -> IRExpr
    num_params: int
    param_names: list  # human-readable names
    needs_diffs: bool
    needs_dist_sq: bool
    needs_dot: bool
    dim: int
    lets: list = field(default_factory=list)  # list[LetBinding] from CSE


# =========================================================================
# SymPy -> IR Conversion
# =========================================================================


def from_sympy(expr: sp.Expr, param_map: Optional[dict] = None) -> IRExpr:
    """Convert a SymPy expression to IR.

    Args:
        expr: SymPy expression
        param_map: Dict mapping sp.Symbol -> (index, name) for parameter symbols.
            Symbols starting with 'p_' are auto-detected if param_map is None.
    """
    if param_map is None:
        param_map = {}

    # Handle atoms
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        # Check if it's a parameter symbol (p_0, p_1, etc.)
        if name.startswith("p_"):
            try:
                idx = int(name[2:])
                return Param(
                    idx,
                    param_map.get(expr, (idx, ""))[1]
                    if isinstance(param_map.get(expr), tuple)
                    else "",
                )
            except ValueError:
                pass
        # Check param_map
        if expr in param_map:
            info = param_map[expr]
            if isinstance(info, tuple):
                return Param(info[0], info[1])
            return Param(info, "")
        return Var(name)

    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
        return Const(float(expr))

    if isinstance(expr, (int, float)):
        return Const(float(expr))

    if isinstance(expr, sp.NumberSymbol):
        return Const(float(expr))

    # Handle functions — use expr.func for reliable type checking
    # (sp.sqrt is a function, not a class, so isinstance doesn't work on it)
    if isinstance(expr, sp.Function) or hasattr(expr, "func"):
        func = getattr(expr, "func", None)
        if func == sp.exp:
            return UnaryFn("exp", from_sympy(expr.args[0], param_map))
        if func == sp.sin:
            return UnaryFn("sin", from_sympy(expr.args[0], param_map))
        if func == sp.cos:
            return UnaryFn("cos", from_sympy(expr.args[0], param_map))
        if func == sp.log:
            return UnaryFn("log", from_sympy(expr.args[0], param_map))
        if func == sp.Abs:
            return UnaryFn("abs", from_sympy(expr.args[0], param_map))
        if func == sp.Max:
            return MaxExpr(
                from_sympy(expr.args[0], param_map),
                from_sympy(expr.args[1], param_map),
            )

    # Handle Pow
    if isinstance(expr, sp.Pow):
        base = from_sympy(expr.args[0], param_map)
        exp_val = from_sympy(expr.args[1], param_map)
        return Pow(base, exp_val)

    # Handle Add
    if isinstance(expr, sp.Add):
        terms = list(expr.args)
        result = from_sympy(terms[0], param_map)
        for t in terms[1:]:
            result = BinOp("+", result, from_sympy(t, param_map))
        return result

    # Handle Mul
    if isinstance(expr, sp.Mul):
        factors = list(expr.args)
        # Handle negation: -1 * x -> neg
        result = from_sympy(factors[0], param_map)
        for f in factors[1:]:
            result = BinOp("*", result, from_sympy(f, param_map))
        return result

    # Fallback: try to evaluate numerically
    try:
        val = float(expr)
        return Const(val)
    except (TypeError, ValueError):
        raise ValueError(
            f"Cannot convert SymPy expression to IR: {expr} (type: {type(expr)})"
        )


def to_ir(
    grad_expr,
    dim: int,
    needs_diffs: bool = True,
    needs_dist_sq: bool = False,
    needs_dot: bool = False,
) -> IRKernel:
    """Convert a GradientExpr to IRKernel.

    Args:
        grad_expr: GradientExpr from differentiation.py
        dim: Input dimension
        needs_diffs, needs_dist_sq, needs_dot: Feature flags
    """
    # Build param_map: symbol -> (index, name)
    param_map = {}
    for i, param in enumerate(grad_expr.params):
        param_map[param] = (
            i,
            grad_expr.param_layout[i] if i < len(grad_expr.param_layout) else "",
        )

    forward_ir = from_sympy(grad_expr.forward, param_map)

    gradients_ir = {}
    for i, param in enumerate(grad_expr.params):
        grad = grad_expr.gradients.get(param, sp.Integer(0))
        gradients_ir[i] = from_sympy(grad, param_map)

    return IRKernel(
        forward=forward_ir,
        gradients=gradients_ir,
        num_params=len(grad_expr.params),
        param_names=list(grad_expr.param_layout),
        needs_diffs=needs_diffs,
        needs_dist_sq=needs_dist_sq,
        needs_dot=needs_dot,
        dim=dim,
    )


# =========================================================================
# IR Utilities
# =========================================================================


def collect_functions(expr: IRExpr) -> set:
    """Collect all function names used in the expression.

    Returns set of function names like {"exp", "sqrt", "sin", "cos", "log"}.
    Used to generate correct math import lines.
    """
    fns = set()
    _walk(expr, lambda node: fns.add(node.fn) if isinstance(node, UnaryFn) else None)
    if _has_pow(expr):
        fns.add("pow")
    return fns


def collect_functions_all(kernel: IRKernel) -> set:
    """Collect all functions from forward + all gradients."""
    fns = collect_functions(kernel.forward)
    for g in kernel.gradients.values():
        fns |= collect_functions(g)
    return fns


def _walk(expr: IRExpr, visitor):
    """Walk all nodes in the IR tree, calling visitor on each."""
    visitor(expr)
    if isinstance(expr, BinOp):
        _walk(expr.left, visitor)
        _walk(expr.right, visitor)
    elif isinstance(expr, UnaryFn):
        _walk(expr.arg, visitor)
    elif isinstance(expr, Pow):
        _walk(expr.base, visitor)
        _walk(expr.exp, visitor)
    elif isinstance(expr, MaxExpr):
        _walk(expr.left, visitor)
        _walk(expr.right, visitor)


def _has_pow(expr: IRExpr) -> bool:
    """Check if expression contains any Pow nodes."""
    found = [False]

    def check(node):
        if isinstance(node, Pow):
            found[0] = True

    _walk(expr, check)
    return found[0]
