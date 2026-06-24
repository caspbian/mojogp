"""Mojo code emission from optimized IR."""

from .gpu_kernel import (
    emit_forward_matvec,
    emit_gradient_matvec,
    emit_cross_matvec,
    emit_extract_diagonal,
    emit_fill_kernel_matrix,
    emit_noise_matvec,
    emit_single_gradient_matvec,
    emit_mixed_forward_matvec,
    emit_mixed_gradient_matvec,
    emit_mixed_materialize,
    emit_kronecker_forward_matvec,
    emit_kronecker_gradient_matvec,
)
from .mojo_printer import emit_ir, collect_math_imports
from .module import emit_module
from .fn_ptr_module import emit_fn_ptr_module
from .builder import MojoBuilder
