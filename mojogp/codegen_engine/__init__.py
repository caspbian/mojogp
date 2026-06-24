"""Internal MojoGP codegen DSL engine.

Structured pipeline for generating GPU kernel code:
    KernelNode -> SymPy expressions -> IR -> optimization passes -> Mojo source

Internal API:
    generate_module(kernel, dim) -> str         # Trait-based module (self-contained)
    generate_fn_ptr_module(kernel, dim) -> str  # Fn-ptr module (lightweight, ~5s)
    generate_hash(kernel, dim) -> str           # Cache key generation

Usage:
    from mojogp.codegen_engine import generate_fn_ptr_module
    code = generate_fn_ptr_module(kernel_node, dim=5)
"""

import hashlib
from typing import Optional

from .expressions import KernelExprBuilder
from .differentiation import compute_gradients
from .overrides import get_overrides_for_kernel
from .ir import to_ir
from .passes import optimize
from .schedule import plan_schedule, ScheduleConfig
from .emit import emit_module
from .emit.fn_ptr_module import emit_fn_ptr_module as _emit_fn_ptr_module
from mojogp.kernel import KernelType


def _resolve_ard(kernel, dim: int):
    """Ensure ARD kernels have ard_dim set.

    When kernel.has_ard() is True but base kernel nodes lack ard_dim,
    apply make_ard_kernel to set it.
    """
    if _has_unresolved_ard(kernel):
        from mojogp.kernel import make_ard_kernel

        return make_ard_kernel(kernel, dim)
    return kernel


def _has_unresolved_ard(kernel) -> bool:
    """Whether any continuous leaf still requests ARD without a resolved ard_dim."""
    if kernel.kernel_type is not None:
        return (
            getattr(kernel, "ard", False) and getattr(kernel, "ard_dim", None) is None
        )
    return (kernel.left is not None and _has_unresolved_ard(kernel.left)) or (
        kernel.right is not None and _has_unresolved_ard(kernel.right)
    )


def _collect_ard_types(kernel, ard_types: set):
    """Recursively collect ARD Mojo type names from a kernel tree."""
    if kernel.kernel_type is not None:
        if getattr(kernel, "ard_dim", None) is not None and hasattr(
            kernel, "_ARD_TYPE_MAP"
        ):
            if kernel.kernel_type in kernel._ARD_TYPE_MAP:
                ard_types.add(kernel._ARD_TYPE_MAP[kernel.kernel_type])
        return
    if kernel.left is not None:
        _collect_ard_types(kernel.left, ard_types)
    if kernel.right is not None:
        _collect_ard_types(kernel.right, ard_types)


def _ard_imports_block(kernel) -> str:
    """Generate Mojo import statements for any ARD kernel types used."""
    ard_types = set()
    _collect_ard_types(kernel, ard_types)
    if not ard_types:
        return ""
    imports = ", ".join(sorted(ard_types))
    return f"""
# ARD kernel types
from kernels.composable_kernel import (
    {imports},
)
"""


def _schedule_policy_tag(kernel) -> Optional[str]:
    """Return a narrow schedule policy tag for measured hot lanes.

    The generic codegen path is shared across multiple wrappers and kernel
    families, so keep aggressive schedule overrides tied to the narrowest kernel
    family we can identify from the kernel tree.
    """

    node = kernel
    while getattr(node, "operator", None) == "scale" and getattr(node, "left", None) is not None:
        node = node.left

    if getattr(node, "operator", None) is None and getattr(node, "kernel_type", None) == KernelType.RBF:
        return "rbf_leaf"

    if getattr(node, "operator", None) is None and getattr(node, "kernel_type", None) in {
        KernelType.MATERN12,
        KernelType.MATERN32,
        KernelType.MATERN52,
        KernelType.RQ,
    }:
        return "low_d_stationary_leaf"
    return None


def generate_module(
    kernel,
    dim: int,
    module_name: str = "fused_kernel",
    schedule_overrides: Optional[ScheduleConfig] = None,
    ncols_hint: Optional[list] = None,
) -> str:
    """Generate complete Mojo module source for a single-output kernel.

    Returns a Mojo source string ready for `mojo build --emit shared-lib`.

    Args:
        kernel: KernelNode from mojogp.kernel
        dim: Input dimension (compile-time constant)
        module_name: Name for the generated Python module
        schedule_overrides: Optional manual schedule configuration
        ncols_hint: Optional list of NCOLS values to specialize for

    Returns:
        Complete Mojo source code as a string
    """
    # Resolve ARD
    kernel = _resolve_ard(kernel, dim)
    kernel_type_str = kernel.to_mojo_type()

    # 1. Build SymPy expression from kernel tree
    builder = KernelExprBuilder(dim)
    kernel_expr = builder.from_kernel_node(kernel)

    # 2. Compute gradients (with overrides for numerical stability)
    overrides = get_overrides_for_kernel(kernel_expr)
    grad_expr = compute_gradients(kernel_expr, overrides)

    # 3. Convert to IR
    ir_kernel = to_ir(
        grad_expr,
        dim,
        needs_diffs=kernel_expr.needs_diffs,
        needs_dist_sq=kernel_expr.needs_dist_sq,
        needs_dot=kernel_expr.needs_dot,
    )

    # 4. Plan schedule and optimize
    if schedule_overrides:
        schedule = schedule_overrides
    else:
        schedule = plan_schedule(
            ir_kernel,
            ncols_hint,
            schedule_policy_tag=_schedule_policy_tag(kernel),
        )
    ir_kernel = optimize(ir_kernel)

    # 5. Get ARD imports block
    ard_imports = _ard_imports_block(kernel)

    # 6. Emit Mojo source
    return emit_module(
        ir_kernel,
        schedule,
        module_name,
        kernel_type_str=kernel_type_str,
        dim=dim,
        ard_imports=ard_imports,
    )


def generate_fn_ptr_module(
    kernel,
    dim: int,
    module_name: str = "jit_kernel",
    schedule_overrides: Optional[ScheduleConfig] = None,
    ncols_hint: Optional[list] = None,
) -> str:
    """Generate lightweight fn-ptr Mojo module for use with JIT engine .so.

    This is the production codegen path. The generated module contains only
    kernel math and fn-ptr exports (~5s compile). Training infrastructure
    lives in the pre-compiled engine .so.

    Args:
        kernel: KernelNode from mojogp.kernel
        dim: Input dimension (compile-time constant)
        module_name: Name for the generated Python module
        schedule_overrides: Optional manual schedule configuration
        ncols_hint: Optional list of NCOLS values to specialize for

    Returns:
        Complete Mojo source code as a string
    """
    # Resolve ARD
    kernel = _resolve_ard(kernel, dim)

    # 1. Build SymPy expression from kernel tree
    builder = KernelExprBuilder(dim)
    kernel_expr = builder.from_kernel_node(kernel)

    # 2. Compute gradients (with overrides for numerical stability)
    overrides = get_overrides_for_kernel(kernel_expr)
    grad_expr = compute_gradients(kernel_expr, overrides)

    # 3. Convert to IR
    ir_kernel = to_ir(
        grad_expr,
        dim,
        needs_diffs=kernel_expr.needs_diffs,
        needs_dist_sq=kernel_expr.needs_dist_sq,
        needs_dot=kernel_expr.needs_dot,
    )

    # 4. Plan schedule and optimize
    if schedule_overrides:
        schedule = schedule_overrides
    else:
        schedule = plan_schedule(
            ir_kernel,
            ncols_hint,
            schedule_policy_tag=_schedule_policy_tag(kernel),
        )
    ir_kernel = optimize(ir_kernel)

    # 5. Emit fn-ptr module source
    return _emit_fn_ptr_module(
        ir_kernel,
        schedule,
        module_name=module_name,
        dim=dim,
    )


def generate_hash(kernel, dim: int) -> str:
    """Generate a deterministic hash for caching compiled kernels.

    Args:
        kernel: KernelNode from mojogp.kernel
        dim: Input dimension

    Returns:
        16-character hex hash string
    """
    kernel = _resolve_ard(kernel, dim)

    # Include engine version in hash to invalidate cache on engine changes
    engine_version = "engine_v11"
    key = f"{engine_version}_{kernel.to_mojo_type()}_{dim}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
