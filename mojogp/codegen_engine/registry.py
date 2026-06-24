"""Kernel type registry mapping names to expression builders."""

from .expressions import KernelExprBuilder, KernelExpr

KERNEL_BUILDERS = {
    "rbf": lambda b, ard, off: b.rbf(ard, off),
    "matern12": lambda b, ard, off: b.matern12(ard, off),
    "matern32": lambda b, ard, off: b.matern32(ard, off),
    "matern52": lambda b, ard, off: b.matern52(ard, off),
    "periodic": lambda b, ard, off: b.periodic(ard, off),
    "rq": lambda b, ard, off: b.rq(ard, off),
    "linear": lambda b, ard, off: b.linear(ard, off),
    "polynomial": lambda b, ard, off: b.polynomial(ard, off),
}


def build_kernel_expr(
    kernel_type: str, dim: int, ard: bool = False, param_offset: int = 0
) -> KernelExpr:
    """Build a KernelExpr from a kernel type name."""
    builder = KernelExprBuilder(dim)
    fn = KERNEL_BUILDERS.get(kernel_type)
    if fn is None:
        raise ValueError(
            f"Unknown kernel type: {kernel_type}. Available: {list(KERNEL_BUILDERS.keys())}"
        )
    return fn(builder, ard, param_offset)
