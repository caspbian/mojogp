"""Inverse lengthscale precomputation pass.

Detects patterns like diff_d / p[offset+d] and converts to
diff_d * inv_ls[d] where inv_ls[d] = 1/p[offset+d] is precomputed.

This reduces per-pair division to a single precomputed reciprocal,
which is significant for ARD kernels with high-dimensional inputs.
"""

from ..ir import IRKernel, IRExpr, Var, Param, Const, BinOp, UnaryFn, Pow, MaxExpr, LetBinding


def _find_div_by_param(expr: IRExpr, found: dict):
    """Find all Div(expr, Param(i)) patterns recursively."""
    if isinstance(expr, BinOp):
        if expr.op == "/" and isinstance(expr.right, Param):
            idx = expr.right.index
            if idx not in found:
                found[idx] = f"inv_p_{idx}"
        _find_div_by_param(expr.left, found)
        _find_div_by_param(expr.right, found)
    elif isinstance(expr, UnaryFn):
        _find_div_by_param(expr.arg, found)
    elif isinstance(expr, Pow):
        _find_div_by_param(expr.base, found)
        _find_div_by_param(expr.exp, found)
    elif isinstance(expr, MaxExpr):
        _find_div_by_param(expr.left, found)
        _find_div_by_param(expr.right, found)


def _pow_tag(exp_val: float) -> str | None:
    if exp_val == -1.0:
        return "inv"
    if exp_val == -2.0:
        return "sq"
    if exp_val == -3.0:
        return "cu"
    return None


def _find_inverse_param_powers(expr: IRExpr, found: dict):
    """Find Param(i)^-k patterns worth hoisting to precomputed inverses."""
    if isinstance(expr, Pow):
        if isinstance(expr.base, Param) and isinstance(expr.exp, Const):
            tag = _pow_tag(expr.exp.value)
            if tag is not None:
                idx = expr.base.index
                if idx not in found:
                    found[idx] = {"inv": f"inv_p_{idx}"}
                if tag == "inv":
                    found[idx]["inv"] = f"inv_p_{idx}"
                elif tag == "sq":
                    found[idx]["sq"] = f"inv_p_{idx}_sq"
                elif tag == "cu":
                    found[idx]["cu"] = f"inv_p_{idx}_cu"
        _find_inverse_param_powers(expr.base, found)
        _find_inverse_param_powers(expr.exp, found)
    elif isinstance(expr, BinOp):
        _find_inverse_param_powers(expr.left, found)
        _find_inverse_param_powers(expr.right, found)
    elif isinstance(expr, UnaryFn):
        _find_inverse_param_powers(expr.arg, found)
    elif isinstance(expr, MaxExpr):
        _find_inverse_param_powers(expr.left, found)
        _find_inverse_param_powers(expr.right, found)


def _replace_div_by_param(expr: IRExpr, inv_map: dict) -> IRExpr:
    """Replace Div(x, Param(i)) with Mul(x, Var(inv_p_i))."""
    if isinstance(expr, BinOp):
        if expr.op == "/" and isinstance(expr.right, Param):
            idx = expr.right.index
            if idx in inv_map:
                new_left = _replace_div_by_param(expr.left, inv_map)
                return BinOp("*", new_left, Var(inv_map[idx]))
        return BinOp(
            expr.op,
            _replace_div_by_param(expr.left, inv_map),
            _replace_div_by_param(expr.right, inv_map),
        )
    elif isinstance(expr, UnaryFn):
        return UnaryFn(expr.fn, _replace_div_by_param(expr.arg, inv_map))
    elif isinstance(expr, Pow):
        return Pow(
            _replace_div_by_param(expr.base, inv_map),
            _replace_div_by_param(expr.exp, inv_map),
        )
    elif isinstance(expr, MaxExpr):
        return MaxExpr(
            _replace_div_by_param(expr.left, inv_map),
            _replace_div_by_param(expr.right, inv_map),
        )
    return expr


def _replace_inverse_param_powers(expr: IRExpr, inverse_map: dict) -> IRExpr:
    """Replace Param(i)^-k with precomputed inverse variables."""
    if isinstance(expr, Pow):
        if isinstance(expr.base, Param) and isinstance(expr.exp, Const):
            tag = _pow_tag(expr.exp.value)
            if tag is not None:
                idx = expr.base.index
                name = inverse_map.get(idx, {}).get(tag)
                if name is not None:
                    return Var(name)
        return Pow(
            _replace_inverse_param_powers(expr.base, inverse_map),
            _replace_inverse_param_powers(expr.exp, inverse_map),
        )
    if isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _replace_inverse_param_powers(expr.left, inverse_map),
            _replace_inverse_param_powers(expr.right, inverse_map),
        )
    if isinstance(expr, UnaryFn):
        return UnaryFn(expr.fn, _replace_inverse_param_powers(expr.arg, inverse_map))
    if isinstance(expr, MaxExpr):
        return MaxExpr(
            _replace_inverse_param_powers(expr.left, inverse_map),
            _replace_inverse_param_powers(expr.right, inverse_map),
        )
    return expr


def inv_ls_pass(kernel: IRKernel) -> IRKernel:
    """Detect division by params and precompute inverses as Let bindings.

    For each pattern Div(x, Param(i)) found in forward or gradient expressions,
    adds a Let binding: var inv_p_i = Float32(1.0) / p[i]
    and replaces the division with multiplication by inv_p_i.
    """
    # Find all Div-by-Param patterns
    found = {}
    power_found = {}
    _find_div_by_param(kernel.forward, found)
    _find_inverse_param_powers(kernel.forward, power_found)
    for idx in kernel.gradients:
        _find_div_by_param(kernel.gradients[idx], found)
        _find_inverse_param_powers(kernel.gradients[idx], power_found)
    for let in kernel.lets:
        _find_div_by_param(let.value, found)
        _find_inverse_param_powers(let.value, power_found)

    if not found and not power_found:
        return kernel

    # Create Let bindings for precomputed inverses. These must come before
    # existing loop lets because later CSE temporaries may depend on them.
    prefix_lets = []
    for param_idx, var_name in sorted(found.items()):
        prefix_lets.append(
            LetBinding(
                var_name,
                BinOp("/", Const(1.0), Param(param_idx, kernel.param_names[param_idx] if param_idx < len(kernel.param_names) else "")),
            )
        )
    for param_idx, spec in sorted(power_found.items()):
        inv_name = spec.get("inv", f"inv_p_{param_idx}")
        if not any(let.name == inv_name for let in prefix_lets):
            prefix_lets.append(
                LetBinding(
                    inv_name,
                    BinOp("/", Const(1.0), Param(param_idx, kernel.param_names[param_idx] if param_idx < len(kernel.param_names) else "")),
                )
            )
        if "sq" in spec and not any(let.name == spec["sq"] for let in prefix_lets):
            prefix_lets.append(
                LetBinding(spec["sq"], BinOp("*", Var(inv_name), Var(inv_name)))
            )
        if "cu" in spec and not any(let.name == spec["cu"] for let in prefix_lets):
            sq_name = spec.get("sq")
            if sq_name is None:
                sq_name = f"inv_p_{param_idx}_sq"
                if not any(let.name == sq_name for let in prefix_lets):
                    prefix_lets.append(
                        LetBinding(sq_name, BinOp("*", Var(inv_name), Var(inv_name)))
                    )
            prefix_lets.append(LetBinding(spec["cu"], BinOp("*", Var(sq_name), Var(inv_name))))
    new_lets = prefix_lets + list(kernel.lets)

    # Replace divisions in all expressions
    new_forward = _replace_inverse_param_powers(
        _replace_div_by_param(kernel.forward, found), power_found
    )
    new_gradients = {
        i: _replace_inverse_param_powers(
            _replace_div_by_param(kernel.gradients[i], found), power_found
        )
        for i in kernel.gradients
    }
    new_let_values = []
    for let in new_lets:
        # Don't replace in the inv_p_i definitions themselves
        if let.name in found.values() or any(let.name in spec.values() for spec in power_found.values()):
            new_let_values.append(let)
        else:
            new_let_values.append(
                LetBinding(
                    let.name,
                    _replace_inverse_param_powers(
                        _replace_div_by_param(let.value, found), power_found
                    ),
                )
            )

    return IRKernel(
        forward=new_forward,
        gradients=new_gradients,
        num_params=kernel.num_params,
        param_names=kernel.param_names,
        needs_diffs=kernel.needs_diffs,
        needs_dist_sq=kernel.needs_dist_sq,
        needs_dot=kernel.needs_dot,
        dim=kernel.dim,
        lets=new_let_values,
    )
