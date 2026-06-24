"""Dead code elimination pass.

Removes Let bindings that are not referenced by any live expression.
"""

from ..ir import (
    IRKernel,
    IRExpr,
    LetBinding,
    Var,
    Param,
    Const,
    BinOp,
    UnaryFn,
    Pow,
    MaxExpr,
)


def _collect_var_refs(expr: IRExpr) -> set:
    """Collect all variable names referenced in an expression."""
    refs = set()
    if isinstance(expr, Var):
        refs.add(expr.name)
    elif isinstance(expr, BinOp):
        refs |= _collect_var_refs(expr.left)
        refs |= _collect_var_refs(expr.right)
    elif isinstance(expr, UnaryFn):
        refs |= _collect_var_refs(expr.arg)
    elif isinstance(expr, Pow):
        refs |= _collect_var_refs(expr.base)
        refs |= _collect_var_refs(expr.exp)
    elif isinstance(expr, MaxExpr):
        refs |= _collect_var_refs(expr.left)
        refs |= _collect_var_refs(expr.right)
    return refs


def dead_code_pass(kernel: IRKernel) -> IRKernel:
    """Remove Let bindings not referenced by any live expression."""
    if not kernel.lets:
        return kernel

    # Collect all variable references from live expressions
    all_refs = _collect_var_refs(kernel.forward)
    for g in kernel.gradients.values():
        all_refs |= _collect_var_refs(g)

    # Also collect refs from other Let bindings (transitive deps)
    changed = True
    while changed:
        changed = False
        for let in kernel.lets:
            if let.name in all_refs:
                new_refs = _collect_var_refs(let.value)
                if not new_refs.issubset(all_refs):
                    all_refs |= new_refs
                    changed = True

    # Keep only referenced lets
    live_lets = [let for let in kernel.lets if let.name in all_refs]

    return IRKernel(
        forward=kernel.forward,
        gradients=kernel.gradients,
        num_params=kernel.num_params,
        param_names=kernel.param_names,
        needs_diffs=kernel.needs_diffs,
        needs_dist_sq=kernel.needs_dist_sq,
        needs_dot=kernel.needs_dot,
        dim=kernel.dim,
        lets=live_lets,
    )
