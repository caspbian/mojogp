"""Gradient sharing pass.

For Product(k1, k2): the gradient of k1's params needs k2's forward value.
This pass detects when gradient expressions contain the forward value of
another sub-kernel and replaces with a Let binding to avoid recomputation.

Currently a no-op — CSE handles most of this automatically.
"""

from ..ir import IRKernel


def gradient_sharing_pass(kernel: IRKernel) -> IRKernel:
    """Reuse forward values in gradient expressions.

    For Sum(k1, k2): k1's gradient doesn't need k2's forward — no sharing needed.
    For Product(k1, k2): gradient uses k2_val from forward pass, not recomputed.

    Currently deferred to CSE which handles this automatically.
    """
    # CSE already handles the common case of shared subexpressions
    # between forward and gradient. This pass would add explicit
    # Let bindings for the forward values of sub-kernels in product
    # compositions. Deferred until profiling shows it's needed.
    return kernel
