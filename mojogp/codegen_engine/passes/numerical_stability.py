"""Numerical stability pass.

Guards against common numerical issues in GPU kernel expressions:
- exp(-x) where x can be large → clamp input to avoid underflow
- sqrt(x) where x can be zero → guard with max(x, epsilon)

These are critical for Matern kernels (sqrt(dist_sq) at dist_sq=0)
and all kernels using exp(-scaled_dist) at large distances.
"""

from ..ir import IRKernel, IRExpr, Var, Param, Const, BinOp, UnaryFn, Pow, MaxExpr, LetBinding


# Maximum exponent before exp() underflows to 0 in float32
_EXP_CLAMP = 20.0
# Epsilon for sqrt guard
_SQRT_EPS = 1e-20


def _stabilize_expr(expr: IRExpr) -> IRExpr:
    """Recursively apply numerical stability transformations."""
    if isinstance(expr, UnaryFn):
        stabilized_arg = _stabilize_expr(expr.arg)

        if expr.fn == "sqrt":
            # sqrt(x) → sqrt(max(x, 1e-20)) to avoid NaN at x=0
            return UnaryFn("sqrt", MaxExpr(stabilized_arg, Const(_SQRT_EPS)))

        if expr.fn == "exp":
            # exp(x) where x might be very negative → clamp
            # Check if argument is negative (common pattern: exp(-0.5 * dist_sq * ...))
            if _is_likely_negative(stabilized_arg):
                # exp(-x) → exp(max(-x, -20)) = exp(-min(x, 20))
                clamped = MaxExpr(stabilized_arg, Const(-_EXP_CLAMP))
                return UnaryFn("exp", clamped)
            return UnaryFn("exp", stabilized_arg)

        return UnaryFn(expr.fn, stabilized_arg)

    if isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _stabilize_expr(expr.left),
            _stabilize_expr(expr.right),
        )

    if isinstance(expr, Pow):
        return Pow(
            _stabilize_expr(expr.base),
            _stabilize_expr(expr.exp),
        )

    if isinstance(expr, MaxExpr):
        return MaxExpr(
            _stabilize_expr(expr.left),
            _stabilize_expr(expr.right),
        )

    return expr


def _is_likely_negative(expr: IRExpr) -> bool:
    """Heuristic: check if an expression is likely to produce negative values.

    Looks for patterns like:
    - Const(negative)
    - Mul(Const(negative), anything)
    - Neg(anything) represented as Mul(Const(-1), x) or BinOp('-', 0, x)
    """
    if isinstance(expr, Const):
        return expr.value < 0

    if isinstance(expr, BinOp):
        if expr.op == "*":
            # Check if either operand is a negative constant
            if isinstance(expr.left, Const) and expr.left.value < 0:
                return True
            if isinstance(expr.right, Const) and expr.right.value < 0:
                return True
        if expr.op == "-":
            # a - b is likely negative if a is zero or small
            if isinstance(expr.left, Const) and expr.left.value <= 0:
                return True

    return False


def numerical_stability_pass(kernel: IRKernel) -> IRKernel:
    """Apply numerical stability guards to kernel expressions.

    Wraps sqrt() with max(x, eps) and exp() with input clamping
    to prevent NaN/Inf in GPU kernel computations.
    """
    new_forward = _stabilize_expr(kernel.forward)
    new_gradients = {
        i: _stabilize_expr(kernel.gradients[i]) for i in kernel.gradients
    }
    new_lets = [
        LetBinding(let.name, _stabilize_expr(let.value)) for let in kernel.lets
    ]

    return IRKernel(
        forward=new_forward,
        gradients=new_gradients,
        num_params=kernel.num_params,
        param_names=kernel.param_names,
        needs_diffs=kernel.needs_diffs,
        needs_dist_sq=kernel.needs_dist_sq,
        needs_dot=kernel.needs_dot,
        dim=kernel.dim,
        lets=new_lets,
    )
