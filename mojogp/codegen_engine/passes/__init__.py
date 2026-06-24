"""Optimization pass pipeline for kernel IR."""

from ..ir import IRKernel
from .cse import cse_pass
from .inv_ls import inv_ls_pass
from .strength import strength_reduce_pass
from .gradient_sharing import gradient_sharing_pass
from .dead_code import dead_code_pass
from .simplify import simplify_pass
from .forward_hoist import forward_hoist_pass
from .numerical_stability import numerical_stability_pass


def optimize(kernel: IRKernel) -> IRKernel:
    """Run all optimization passes in order.

    Pass ordering rationale:
    1. CSE first — find shared subexpressions across forward + all gradients
    2. inv_ls — precompute 1/lengthscale (before strength reduction changes patterns)
    3. strength reduction — pow(x,2)→x*x, pow(x,0.5)→sqrt, etc.
    4. forward hoist — share kval between forward and gradient expressions
    5. gradient sharing — additional gradient-specific sharing (deferred to CSE)
    6. dead code — remove unreferenced Let bindings
    7. numerical stability — add safety guards (LAST so it doesn't interfere with patterns)
    8. simplify — final simplification (currently no-op)
    """
    kernel = cse_pass(kernel)
    kernel = inv_ls_pass(kernel)
    kernel = strength_reduce_pass(kernel)
    kernel = forward_hoist_pass(kernel)
    kernel = gradient_sharing_pass(kernel)
    kernel = dead_code_pass(kernel)
    kernel = numerical_stability_pass(kernel)
    kernel = simplify_pass(kernel)
    return kernel
