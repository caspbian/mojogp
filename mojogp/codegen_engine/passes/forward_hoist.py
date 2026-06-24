"""Forward value hoisting pass.

Detects when gradient expressions contain the same subexpression as the
forward kernel value, and replaces it with a reference to a pre-computed
'kval' variable. This avoids recomputing the forward expression in each
gradient computation.

Important for Product kernels where gradients reference k1_val and k2_val,
and for any kernel where the gradient formula includes the kernel value.
"""

from ..ir import IRKernel, IRExpr, Var, Param, Const, BinOp, UnaryFn, Pow, MaxExpr, LetBinding


def _expr_equal(a: IRExpr, b: IRExpr) -> bool:
    """Check structural equality of two IR expressions."""
    if type(a) != type(b):
        return False
    if isinstance(a, Var):
        return a.name == b.name
    if isinstance(a, Param):
        return a.index == b.index
    if isinstance(a, Const):
        return a.value == b.value
    if isinstance(a, BinOp):
        return a.op == b.op and _expr_equal(a.left, b.left) and _expr_equal(a.right, b.right)
    if isinstance(a, UnaryFn):
        return a.fn == b.fn and _expr_equal(a.arg, b.arg)
    if isinstance(a, Pow):
        return _expr_equal(a.base, b.base) and _expr_equal(a.exp, b.exp)
    if isinstance(a, MaxExpr):
        return _expr_equal(a.left, b.left) and _expr_equal(a.right, b.right)
    return False


def _count_nodes(expr: IRExpr) -> int:
    """Count the number of nodes in an IR expression tree."""
    if isinstance(expr, (Var, Param, Const)):
        return 1
    if isinstance(expr, BinOp):
        return 1 + _count_nodes(expr.left) + _count_nodes(expr.right)
    if isinstance(expr, UnaryFn):
        return 1 + _count_nodes(expr.arg)
    if isinstance(expr, Pow):
        return 1 + _count_nodes(expr.base) + _count_nodes(expr.exp)
    if isinstance(expr, MaxExpr):
        return 1 + _count_nodes(expr.left) + _count_nodes(expr.right)
    return 1


def _replace_expr(expr: IRExpr, target: IRExpr, replacement: IRExpr) -> IRExpr:
    """Replace all occurrences of target with replacement in expr."""
    if _expr_equal(expr, target):
        return replacement
    if isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _replace_expr(expr.left, target, replacement),
            _replace_expr(expr.right, target, replacement),
        )
    if isinstance(expr, UnaryFn):
        return UnaryFn(expr.fn, _replace_expr(expr.arg, target, replacement))
    if isinstance(expr, Pow):
        return Pow(
            _replace_expr(expr.base, target, replacement),
            _replace_expr(expr.exp, target, replacement),
        )
    if isinstance(expr, MaxExpr):
        return MaxExpr(
            _replace_expr(expr.left, target, replacement),
            _replace_expr(expr.right, target, replacement),
        )
    return expr


def _contains_expr(expr: IRExpr, target: IRExpr) -> bool:
    """Check if target appears anywhere in expr."""
    if _expr_equal(expr, target):
        return True
    if isinstance(expr, BinOp):
        return _contains_expr(expr.left, target) or _contains_expr(expr.right, target)
    if isinstance(expr, UnaryFn):
        return _contains_expr(expr.arg, target)
    if isinstance(expr, Pow):
        return _contains_expr(expr.base, target) or _contains_expr(expr.exp, target)
    if isinstance(expr, MaxExpr):
        return _contains_expr(expr.left, target) or _contains_expr(expr.right, target)
    return False


def forward_hoist_pass(kernel: IRKernel) -> IRKernel:
    """Hoist forward expression into a 'kval' variable if referenced by gradients.

    If the forward expression appears as a subexpression in any gradient,
    add a Let binding 'kval = <forward_expr>' and replace all occurrences
    in gradients with Var('kval').

    Only applies when forward expression is non-trivial (>3 nodes) to avoid
    unnecessary variable creation for simple kernels.
    """
    forward = kernel.forward
    forward_size = _count_nodes(forward)

    # Don't hoist trivial expressions
    if forward_size <= 3:
        return kernel

    # Check if any gradient contains the forward expression
    any_contains = False
    for idx in kernel.gradients:
        if _contains_expr(kernel.gradients[idx], forward):
            any_contains = True
            break

    if not any_contains:
        return kernel

    # Add kval Let binding and replace in gradients
    kval_var = Var("kval")
    new_lets = list(kernel.lets) + [LetBinding("kval", forward)]

    new_gradients = {}
    for idx in kernel.gradients:
        new_gradients[idx] = _replace_expr(kernel.gradients[idx], forward, kval_var)

    # Forward expression itself references kval now
    new_forward = kval_var

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
