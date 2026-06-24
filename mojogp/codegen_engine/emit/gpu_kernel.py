"""GPU kernel scaffolding using the programmatic MojoBuilder.

Generates complete GPU kernel functions (forward matvec, gradient matvec,
cross matvec, extract diagonal) by building Mojo source programmatically.
"""

import re

from .builder import MojoBuilder
from .mojo_printer import emit_ir
from ..ir import IRKernel
from ..schedule import ScheduleConfig


_CAT_VAR_RE = re.compile(r"^cat_(\d+)$")


def _emit_xy_aliases(b: MojoBuilder, kernel: IRKernel, tm_suffix: str = ""):
    """Bind per-dimension x_row_d / x_col_d aliases when IR references them."""
    if kernel.dim <= 0:
        return
    for d in range(kernel.dim):
        b.line(f"var x_row_{d} = x_row{tm_suffix}[{d}]")
        b.line(f"var x_col_{d} = yj[sb + {d}]")


def _emit_intermediates(b: MojoBuilder, kernel: IRKernel, tm_suffix: str = ""):
    """Emit shared intermediate computation (diffs, dist_sq, dot_prod)."""
    if kernel.needs_diffs:
        b.line(f"var diffs = InlineArray[Float32, DIM](uninitialized=True)")
    if kernel.needs_dist_sq:
        b.line("var dist_sq = Float32(0)")
    if kernel.needs_dot:
        b.line("var dot_prod = Float32(0)")

    if kernel.needs_diffs:
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line(f"diffs[d] = x_row{tm_suffix}[d] - yj[sb + d]")
            if kernel.needs_dist_sq:
                b.line("dist_sq += diffs[d] * diffs[d]")
            if kernel.needs_dot:
                b.line(f"dot_prod += x_row{tm_suffix}[d] * yj[sb + d]")
        # Unroll per-dimension diff aliases for kernels that reference diff_N
        # (e.g., Periodic kernel uses sin(pi * diff_d / period) per dimension)
        if kernel.dim > 0:
            for d in range(kernel.dim):
                b.line(f"var diff_{d} = diffs[{d}]")
    elif kernel.needs_dot:
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line(f"dot_prod += x_row{tm_suffix}[d] * yj[sb + d]")


def _emit_let_bindings(b: MojoBuilder, kernel: IRKernel):
    """Emit CSE Let bindings."""
    for let in kernel.lets:
        b.line(f"var {let.name} = {emit_ir(let.value)}")


def _expr_is_loop_invariant(expr, invariant_let_names: set[str]) -> bool:
    """Return True when an IR expression depends only on params/constants."""
    from ..ir import Var, Param, Const, BinOp, UnaryFn, Pow, MaxExpr

    if isinstance(expr, (Param, Const)):
        return True
    if isinstance(expr, Var):
        return expr.name in invariant_let_names
    if isinstance(expr, BinOp):
        return _expr_is_loop_invariant(expr.left, invariant_let_names) and _expr_is_loop_invariant(expr.right, invariant_let_names)
    if isinstance(expr, UnaryFn):
        return _expr_is_loop_invariant(expr.arg, invariant_let_names)
    if isinstance(expr, Pow):
        return _expr_is_loop_invariant(expr.base, invariant_let_names) and _expr_is_loop_invariant(expr.exp, invariant_let_names)
    if isinstance(expr, MaxExpr):
        return _expr_is_loop_invariant(expr.left, invariant_let_names) and _expr_is_loop_invariant(expr.right, invariant_let_names)
    return False


def _split_loop_invariant_lets(kernel: IRKernel):
    """Split let bindings into invariant and per-pair groups preserving order."""
    invariant_names: set[str] = set()
    invariant_lets = []
    loop_lets = []
    for let in kernel.lets:
        if _expr_is_loop_invariant(let.value, invariant_names):
            invariant_lets.append(let)
            invariant_names.add(let.name)
        else:
            loop_lets.append(let)
    return invariant_lets, loop_lets


def _collect_mixed_cat_var_indices(kernel: IRKernel) -> list[int]:
    """Collect categorical IR variable indices (cat_0, cat_1, ...) from the kernel."""

    indices = set()

    def walk(expr):
        if expr is None:
            return
        name = getattr(expr, "name", None)
        if isinstance(name, str):
            match = _CAT_VAR_RE.match(name)
            if match:
                indices.add(int(match.group(1)))
        for attr in ("left", "right", "arg", "base", "exp", "value"):
            child = getattr(expr, attr, None)
            if child is not None:
                walk(child)

    walk(kernel.forward)
    for grad_expr in kernel.gradients.values():
        walk(grad_expr)
    for let in kernel.lets:
        walk(let.value)

    return sorted(indices)


def _emit_mixed_cat_var_bindings(
    b: MojoBuilder,
    kernel: IRKernel,
    cat_i_template: str,
    cat_j_template: str,
):
    """Emit per-pair categorical lookup vars used by arbitrary mixed-tree IR."""

    for cat_idx in _collect_mixed_cat_var_indices(kernel):
        b.line(f"var cat_{cat_idx} = Float32(1.0)")
        with b.block(f"if {cat_idx} < num_cat_vars:"):
            b.line(f"var off = Int(offsets_ptr[{cat_idx}])")
            b.line(f"var lev = Int(levels_ptr[{cat_idx}])")
            b.line(f"var ci = {cat_i_template.format(cat_idx=cat_idx)}")
            b.line(f"var cj = {cat_j_template.format(cat_idx=cat_idx)}")
            b.line(f"cat_{cat_idx} = corr_flat_ptr[off + ci * lev + cj]")


def _emit_default_cat_var_bindings(b: MojoBuilder, kernel: IRKernel):
    """Bind mixed categorical leaves for non-mixed helper kernels.

    Mixed fn-ptr modules still emit the standard helper kernels. Those helpers do
    not receive categorical buffers, so symbolic categorical leaves must default
    to 1.0 there while the real mixed kernels use `_emit_mixed_cat_var_bindings()`.
    """

    for cat_idx in _collect_mixed_cat_var_indices(kernel):
        b.line(f"var cat_{cat_idx} = Float32(1.0)")


def emit_forward_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate fused forward matvec GPU kernel with shmem tiling."""
    b = MojoBuilder()
    tm = schedule.tm if schedule.use_shmem else 1
    invariant_lets, loop_lets = _split_loop_invariant_lets(kernel)
    b.blank()
    b.comment("=" * 77)
    b.comment("FUSED Forward Matvec (auto-generated by codegen_engine)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_forward_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int, noise: Float32,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Fused shmem-tiled forward matvec."""')
        b.line("alias DIMY = DIM + NCOLS")
        b.line(f"alias TM = {tm}")
        b.line("var tid = Int(thread_idx.x)")
        b.line("var bs = Int(block_dim.x)")
        b.line("var i_base = Int(block_idx.x) * (bs * TM) + tid")
        b.line(
            "var yj = external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16]()"
        )
        for t in range(tm):
            suffix = "" if tm == 1 else str(t)
            row_expr = "i_base" if t == 0 else f"i_base + bs * {t}"
            b.line(f"var i{suffix} = {row_expr}")
            b.line(f"var valid{suffix} = i{suffix} < n")
        b.blank()

        # Load params
        b.comment(f"Pre-load ALL {kernel.num_params} params into registers")
        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        if invariant_lets:
            b.comment("Loop-invariant lets hoisted out of the O(n^2) inner loop")
            for let in invariant_lets:
                b.line(f"var {let.name} = {emit_ir(let.value)}")
            b.blank()

        # Load TM x_row tiles into registers
        for t in range(tm):
            suffix = "" if tm == 1 else str(t)
            x_name = "x_row" if tm == 1 else f"x_row{t}"
            b.line(f"var {x_name} = InlineArray[Float32, DIM](uninitialized=True)")
            with b.block(f"if valid{suffix}:"):
                b.line(f"var ro{suffix} = UInt(i{suffix}) * UInt(DIM)")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line(f"{x_name}[d] = x_ptr[ro{suffix} + UInt(d)]")

        # Per-row accumulators
        for t in range(tm):
            acc_name = "acc" if tm == 1 else f"acc{t}"
            b.line(f"var {acc_name} = InlineArray[Float32, NCOLS](fill=Float32(0.0))")

        # Shmem tile loop
        b.line("var jstart = 0")
        with b.block("while jstart < n:"):
            b.line("var j = jstart + tid")
            with b.block("if j < n:"):
                b.line("var sb = tid * DIMY")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("yj[sb + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]")
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line("yj[sb + DIM + c] = v_ptr[UInt(c) * UInt(n) + UInt(j)]")
            b.line("barrier()")
            b.line("var te = bs if jstart + bs <= n else n - jstart")
            with b.block("for jrel in range(te):"):
                b.line("var sb = jrel * DIMY")
                for t in range(tm):
                    suffix = "" if tm == 1 else str(t)
                    acc_name = "acc" if tm == 1 else f"acc{t}"
                    tm_suffix = "" if tm == 1 else str(t)
                    with b.block(f"if valid{suffix}:"):
                        _emit_intermediates(b, kernel, tm_suffix)
                        _emit_xy_aliases(b, kernel, tm_suffix)
                        _emit_default_cat_var_bindings(b, kernel)
                        for let in loop_lets:
                            b.line(f"var {let.name} = {emit_ir(let.value)}")
                        b.line(f"var kval = {emit_ir(kernel.forward)}")
                        b.line("@parameter")
                        with b.block("for c in range(NCOLS):"):
                            b.line(f"{acc_name}[c] += kval * yj[sb + DIM + c]")
            b.line("barrier()")
            b.line("jstart += bs")

        # Write TM output rows with noise outside the hot inner loop.
        for t in range(tm):
            suffix = "" if tm == 1 else str(t)
            acc_name = "acc" if tm == 1 else f"acc{t}"
            with b.block(f"if valid{suffix}:"):
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line(f"var off{suffix} = UInt(c) * UInt(n) + UInt(i{suffix})")
                    b.line(f"out_ptr[off{suffix}] = {acc_name}[c] + noise * v_ptr[off{suffix}]")

    return b.build()


def emit_forward_matvec_splitj(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate experimental split-j forward matvec kernels.

    The default forward kernel maps one block dimension to output rows and each
    block loops over all j-tiles. This experimental route adds a second block
    dimension over j chunks, writes partial sums, then reduces those partials.
    It is env-gated by the fn-ptr wrapper and intended for A/B measurement only.
    """
    b = MojoBuilder()
    invariant_lets, loop_lets = _split_loop_invariant_lets(kernel)
    b.blank()
    b.comment("=" * 77)
    b.comment("EXPERIMENTAL Split-J Forward Matvec (auto-generated by codegen_engine)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_forward_matvec_splitj_partials[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("partial_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int, chunk_size: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Compute one j-chunk of K @ V into partial_ptr."""')
        b.line("alias DIMY = DIM + NCOLS")
        b.line("var tid = Int(thread_idx.x)")
        b.line("var bs = Int(block_dim.x)")
        b.line("var i = Int(block_idx.x) * bs + tid")
        b.line("var chunk_id = Int(block_idx.y)")
        b.line("var chunk_start = chunk_id * chunk_size")
        b.line("var chunk_end = chunk_start + chunk_size")
        with b.block("if chunk_end > n:"):
            b.line("chunk_end = n")
        b.line(
            "var yj = external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16]()"
        )
        b.line("var valid = i < n")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        if invariant_lets:
            b.comment("Loop-invariant lets hoisted out of the O(n^2) inner loop")
            for let in invariant_lets:
                b.line(f"var {let.name} = {emit_ir(let.value)}")
            b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        with b.block("if valid:"):
            b.line("var ro = UInt(i) * UInt(DIM)")
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.line("var acc = InlineArray[Float32, NCOLS](fill=Float32(0.0))")
        b.blank()

        b.line("var jstart = chunk_start")
        with b.block("while jstart < chunk_end:"):
            b.line("var j = jstart + tid")
            with b.block("if j < chunk_end:"):
                b.line("var sb = tid * DIMY")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("yj[sb + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]")
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line("yj[sb + DIM + c] = v_ptr[UInt(c) * UInt(n) + UInt(j)]")
            b.line("barrier()")
            b.line("var te = bs if jstart + bs <= chunk_end else chunk_end - jstart")
            with b.block("if valid:"):
                with b.block("for jrel in range(te):"):
                    b.line("var sb = jrel * DIMY")
                    _emit_intermediates(b, kernel)
                    _emit_xy_aliases(b, kernel)
                    _emit_default_cat_var_bindings(b, kernel)
                    for let in loop_lets:
                        b.line(f"var {let.name} = {emit_ir(let.value)}")
                    b.line(f"var kval = {emit_ir(kernel.forward)}")
                    b.line("@parameter")
                    with b.block("for c in range(NCOLS):"):
                        b.line("acc[c] += kval * yj[sb + DIM + c]")
            b.line("barrier()")
            b.line("jstart += bs")
        b.blank()

        with b.block("if valid:"):
            b.line("@parameter")
            with b.block("for c in range(NCOLS):"):
                b.line("partial_ptr[(chunk_id * NCOLS + c) * n + i] = acc[c]")

    b.blank()
    b.line("fn reduce_splitj_forward_ncols[NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("partial_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int, num_chunks: Int, noise: Float32,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Reduce split-j partials and add diagonal noise."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        b.line("var c = Int(block_idx.y)")
        with b.block("if i >= n or c >= NCOLS:"):
            b.line("return")
        b.line("var sum_val = Float32(0.0)")
        with b.block("for chunk_id in range(num_chunks):"):
            b.line("sum_val += partial_ptr[(chunk_id * NCOLS + c) * n + i]")
        b.line("var off = UInt(c) * UInt(n) + UInt(i)")
        b.line("out_ptr[off] = sum_val + noise * v_ptr[off]")

    return b.build()


def emit_gradient_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate fused all-gradients matvec GPU kernel."""
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("FUSED All-Gradients Matvec (auto-generated by codegen_engine)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_all_gradients_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Fused shmem-tiled all-gradients matvec."""')
        b.line("alias DIMY = DIM + NCOLS")
        b.line(f"alias NPARAMS = {kernel.num_params}")
        b.line("var tid = Int(thread_idx.x)")
        b.line("var bs = Int(block_dim.x)")
        b.line("var i = Int(block_idx.x) * bs + tid")
        b.line(
            "var yj = external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16]()"
        )
        b.line("var valid = i < n")
        b.blank()

        # Load params
        b.line(f"var p = InlineArray[Float32, NPARAMS](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range(NPARAMS):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        # Load x_row
        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        with b.block("if valid:"):
            b.line("var ro = UInt(i) * UInt(DIM)")
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()

        # Gradient accumulators
        b.line(
            "var grad_acc = InlineArray[Float32, NPARAMS * NCOLS](fill=Float32(0.0))"
        )
        b.blank()

        # Shmem tile loop
        b.line("var jstart = 0")
        with b.block("while jstart < n:"):
            b.line("var j = jstart + tid")
            with b.block("if j < n:"):
                b.line("var sb = tid * DIMY")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("yj[sb + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]")
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line("yj[sb + DIM + c] = v_ptr[UInt(c) * UInt(n) + UInt(j)]")
            b.line("barrier()")
            with b.block("if valid:"):
                b.line("var te = bs if jstart + bs <= n else n - jstart")
                with b.block("for jrel in range(te):"):
                    b.line("var sb = jrel * DIMY")
                    _emit_intermediates(b, kernel)
                    _emit_xy_aliases(b, kernel)
                    _emit_default_cat_var_bindings(b, kernel)
                    _emit_let_bindings(b, kernel)
                    # Compute gradient values
                    b.line(
                        "var grad = InlineArray[Float32, NPARAMS](fill=Float32(0.0))"
                    )
                    for p_idx in sorted(kernel.gradients.keys()):
                        b.line(f"grad[{p_idx}] = {emit_ir(kernel.gradients[p_idx])}")
                    # Accumulate
                    b.line("@parameter")
                    with b.block("for p_idx in range(NPARAMS):"):
                        b.line("var gv = grad[p_idx]")
                        b.line("@parameter")
                        with b.block("for c in range(NCOLS):"):
                            b.line(
                                "grad_acc[p_idx * NCOLS + c] += gv * yj[sb + DIM + c]"
                            )
            b.line("barrier()")
            b.line("jstart += bs")

        # Write output
        with b.block("if valid:"):
            b.line("@parameter")
            with b.block("for p_idx in range(NPARAMS):"):
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line(
                        "var off = UInt(p_idx) * UInt(n) * UInt(NCOLS) + UInt(c) * UInt(n) + UInt(i)"
                    )
                    b.line("out_ptr[off] = grad_acc[p_idx * NCOLS + c]")

    return b.build()


def emit_cross_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate fused cross-covariance matvec for prediction."""
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("FUSED Cross-Covariance Matvec (auto-generated for prediction)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_cross_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n_test: Int, n_train: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Fused cross-covariance K(X_test, X_train) @ v."""')
        b.line("alias DIMY = DIM + NCOLS")
        b.line("var tid = Int(thread_idx.x)")
        b.line("var bs = Int(block_dim.x)")
        b.line("var i = Int(block_idx.x) * bs + tid")
        b.line(
            "var yj = external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16]()"
        )
        b.line("var valid = i < n_test")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        with b.block("if valid:"):
            b.line("var ro = UInt(i) * UInt(DIM)")
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("x_row[d] = x_test_ptr[ro + UInt(d)]")

        b.line("var acc = InlineArray[Float32, NCOLS](fill=Float32(0.0))")
        b.line("var jstart = 0")
        with b.block("while jstart < n_train:"):
            b.line("var j = jstart + tid")
            with b.block("if j < n_train:"):
                b.line("var sb = tid * DIMY")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("yj[sb + d] = x_train_ptr[UInt(j) * UInt(DIM) + UInt(d)]")
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line(
                        "yj[sb + DIM + c] = v_ptr[UInt(c) * UInt(n_train) + UInt(j)]"
                    )
            b.line("barrier()")
            with b.block("if valid:"):
                b.line("var te = bs if jstart + bs <= n_train else n_train - jstart")
                with b.block("for jrel in range(te):"):
                    b.line("var sb = jrel * DIMY")
                    _emit_intermediates(b, kernel)
                    _emit_xy_aliases(b, kernel)
                    _emit_default_cat_var_bindings(b, kernel)
                    _emit_let_bindings(b, kernel)
                    b.line(f"var kval = {emit_ir(kernel.forward)}")
                    b.line("@parameter")
                    with b.block("for c in range(NCOLS):"):
                        b.line("acc[c] += kval * yj[sb + DIM + c]")
            b.line("barrier()")
            b.line("jstart += bs")

        with b.block("if valid:"):
            b.line("@parameter")
            with b.block("for c in range(NCOLS):"):
                b.line("out_ptr[UInt(c) * UInt(n_test) + UInt(i)] = acc[c]")

    return b.build()


def emit_extract_diagonal(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate fused diagonal extraction for prediction."""
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("FUSED Extract Diagonal (auto-generated for prediction)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_extract_diagonal[DIM: Int](")
    with b.block():
        b.line("diag_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Fused diagonal extraction: diag[i] = k(x_i, x_i)."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n:"):
            b.line("return")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()

        b.comment("Self-evaluation: diffs=0, dist_sq=0, dot_prod=x^T@x")
        if kernel.needs_diffs:
            b.line("var diffs = InlineArray[Float32, DIM](fill=Float32(0.0))")
            # Unroll per-dimension diff aliases (all zero for self-evaluation)
            if kernel.dim > 0:
                for d in range(kernel.dim):
                    b.line(f"var diff_{d} = Float32(0.0)")
        b.line("var dist_sq = Float32(0.0)")
        b.line("var dot_prod = Float32(0.0)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("dot_prod += x_row[d] * x_row[d]")
        b.blank()

        b.comment("yj/sb aliases for eval code that references yj[sb+d]")
        b.line("var sb = Int(ro)")
        b.line("var yj = x_ptr")
        b.blank()

        _emit_xy_aliases(b, kernel)
        _emit_default_cat_var_bindings(b, kernel)
        _emit_let_bindings(b, kernel)
        b.line(f"diag_ptr[i] = {emit_ir(kernel.forward)}")

    return b.build()


def emit_fill_kernel_matrix(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate GPU kernel to fill K[i,j] = k(x_i, x_j) for materialized mode."""
    b = MojoBuilder()
    invariant_lets, loop_lets = _split_loop_invariant_lets(kernel)
    b.blank()
    b.comment("=" * 77)
    b.comment("Fill Kernel Matrix (materialized mode, auto-generated)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_fill_kernel_matrix[DIM: Int](")
    with b.block():
        b.line("k_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Fill K[i,j] = k(x_i, x_j) for all i. One thread per row."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n:"):
            b.line("return")
        b.blank()

        # Load params
        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        if invariant_lets:
            b.comment("Loop-invariant lets hoisted out of the O(n^2) fill loop")
            for let in invariant_lets:
                b.line(f"var {let.name} = {emit_ir(let.value)}")
            b.blank()

        # Load x_row[i]
        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()

        # Loop over j columns
        with b.block("for j in range(n):"):
            b.line("var joff = UInt(j) * UInt(DIM)")

            # Compute intermediates from global memory
            if kernel.needs_diffs:
                b.line("var diffs = InlineArray[Float32, DIM](uninitialized=True)")
            if kernel.needs_dist_sq:
                b.line("var dist_sq = Float32(0)")
            if kernel.needs_dot:
                b.line("var dot_prod = Float32(0)")

            if kernel.needs_diffs:
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("diffs[d] = x_row[d] - x_ptr[joff + UInt(d)]")
                    if kernel.needs_dist_sq:
                        b.line("dist_sq += diffs[d] * diffs[d]")
                    if kernel.needs_dot:
                        b.line("dot_prod += x_row[d] * x_ptr[joff + UInt(d)]")
                if kernel.dim > 0:
                    for d in range(kernel.dim):
                        b.line(f"var diff_{d} = diffs[{d}]")
            elif kernel.needs_dot:
                # dot_prod already declared above, just compute it
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("dot_prod += x_row[d] * x_ptr[joff + UInt(d)]")

            # Alias for IR that references yj[sb+d]
            b.line("var sb = Int(joff)")
            b.line("var yj = x_ptr")

            _emit_xy_aliases(b, kernel)
            _emit_default_cat_var_bindings(b, kernel)
            for let in loop_lets:
                b.line(f"var {let.name} = {emit_ir(let.value)}")
            b.line(f"k_ptr[UInt(i) * UInt(n) + UInt(j)] = {emit_ir(kernel.forward)}")

    b.blank()
    b.line("fn fused_fill_kernel_matrix_2d[DIM: Int](")
    with b.block():
        b.line("k_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Fill K[i,j] with a 2D thread grid: one thread per matrix entry."""')
        b.line("var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        b.line("var i = Int(block_idx.y) * Int(block_dim.y) + Int(thread_idx.y)")
        with b.block("if i >= n or j >= n:"):
            b.line("return")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        if invariant_lets:
            b.comment("Loop-invariant lets hoisted out of the pair evaluation")
            for let in invariant_lets:
                b.line(f"var {let.name} = {emit_ir(let.value)}")
            b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("var joff = UInt(j) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()
        b.line("var sb = Int(joff)")
        b.line("var yj = x_ptr")
        _emit_intermediates(b, kernel)
        _emit_xy_aliases(b, kernel)
        _emit_default_cat_var_bindings(b, kernel)
        for let in loop_lets:
            b.line(f"var {let.name} = {emit_ir(let.value)}")
        b.line(f"k_ptr[UInt(i) * UInt(n) + UInt(j)] = {emit_ir(kernel.forward)}")

    return b.build()


def emit_fill_cross_covariance(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate GPU kernel to fill K_train_test[:, j] = k(x_train_i, x_test_j)."""
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("Fill Cross Covariance (prediction, auto-generated)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_fill_cross_covariance[DIM: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n_test: Int, n_train: Int,")
    b.line(") -> None:")
    with b.block():
        b.line(
            '"""Fill out_ptr[test_col * n_train + train_row] = k(x_train_row, x_test_col)."""'
        )
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n_train:"):
            b.line("return")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_train_ptr[ro + UInt(d)]")
        b.blank()

        with b.block("for j in range(n_test):"):
            b.line("var joff = UInt(j) * UInt(DIM)")

            if kernel.needs_diffs:
                b.line("var diffs = InlineArray[Float32, DIM](uninitialized=True)")
            if kernel.needs_dist_sq:
                b.line("var dist_sq = Float32(0)")
            if kernel.needs_dot:
                b.line("var dot_prod = Float32(0)")

            if kernel.needs_diffs:
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("diffs[d] = x_row[d] - x_test_ptr[joff + UInt(d)]")
                    if kernel.needs_dist_sq:
                        b.line("dist_sq += diffs[d] * diffs[d]")
                    if kernel.needs_dot:
                        b.line("dot_prod += x_row[d] * x_test_ptr[joff + UInt(d)]")
                if kernel.dim > 0:
                    for d in range(kernel.dim):
                        b.line(f"var diff_{d} = diffs[{d}]")
            elif kernel.needs_dot:
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("dot_prod += x_row[d] * x_test_ptr[joff + UInt(d)]")

            b.line("var sb = Int(joff)")
            b.line("var yj = x_test_ptr")
            _emit_xy_aliases(b, kernel)
            _emit_default_cat_var_bindings(b, kernel)
            _emit_let_bindings(b, kernel)
            b.line(f"out_ptr[UInt(j) * UInt(n_train) + UInt(i)] = {emit_ir(kernel.forward)}")

    return b.build()


def emit_noise_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate diagonal-noise injection kernels."""
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("Noise Matvec: out[i] += diagonal_noise[i] * v[i]")
    b.comment("=" * 77)
    b.blank()

    b.line("fn kernel_add_noise_vec(")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("total: Int, noise: Float32,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Add noise*v to output: out[i] += noise * v[i]."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= total:"):
            b.line("return")
        b.line("out_ptr[i] = out_ptr[i] + noise * v_ptr[i]")

    b.blank()
    b.line("fn kernel_add_noise_vector_vec(")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("noise_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("total: Int, n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Add per-row diagonal noise: out[row, col] += noise[row] * v[row, col]."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= total:"):
            b.line("return")
        b.line("var row = i % n")
        b.line("out_ptr[i] = out_ptr[i] + noise_ptr[row] * v_ptr[i]")

    return b.build()


def emit_single_gradient_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate single-param gradient matvec (fallback for non-fused path)."""
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("Single-Param Gradient Matvec (fallback, auto-generated)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn single_gradient_matvec[DIM: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int, num_cols: Int, param_index: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Single param gradient: out = dK/dtheta[param_index] @ v."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n:"):
            b.line("return")
        b.blank()

        # Load params
        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        # Load x_row
        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()

        # This fallback is used by wrapper adapters such as LMC, where the same
        # workspace buffer is reused across parameters. Start each output entry
        # from zero before accumulating over j to avoid stale-gradient carryover.
        with b.block("for c in range(num_cols):"):
            b.line("out_ptr[UInt(c) * UInt(n) + UInt(i)] = Float32(0.0)")
        b.blank()

        # Accumulate per column
        b.line("# Accumulate per column")
        with b.block("for j in range(n):"):
            b.line("var joff = UInt(j) * UInt(DIM)")

            # Compute intermediates
            if kernel.needs_diffs:
                b.line("var diffs = InlineArray[Float32, DIM](uninitialized=True)")
            if kernel.needs_dist_sq:
                b.line("var dist_sq = Float32(0)")
            if kernel.needs_dot:
                b.line("var dot_prod = Float32(0)")

            if kernel.needs_diffs:
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("diffs[d] = x_row[d] - x_ptr[joff + UInt(d)]")
                    if kernel.needs_dist_sq:
                        b.line("dist_sq += diffs[d] * diffs[d]")
                    if kernel.needs_dot:
                        b.line("dot_prod += x_row[d] * x_ptr[joff + UInt(d)]")
                if kernel.dim > 0:
                    for d in range(kernel.dim):
                        b.line(f"var diff_{d} = diffs[{d}]")
            elif kernel.needs_dot:
                # dot_prod already declared above, just compute it
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("dot_prod += x_row[d] * x_ptr[joff + UInt(d)]")

            b.line("var sb = Int(joff)")
            b.line("var yj = x_ptr")
            _emit_xy_aliases(b, kernel)
            _emit_default_cat_var_bindings(b, kernel)
            _emit_let_bindings(b, kernel)

            # Compute all gradients, pick the requested one
            b.line(
                f"var grad = InlineArray[Float32, {kernel.num_params}](fill=Float32(0.0))"
            )
            for p_idx in sorted(kernel.gradients.keys()):
                b.line(f"grad[{p_idx}] = {emit_ir(kernel.gradients[p_idx])}")

            b.line("var gv = grad[param_index]")
            with b.block("for c in range(num_cols):"):
                b.line(
                    "out_ptr[UInt(c) * UInt(n) + UInt(i)] += gv * v_ptr[UInt(c) * UInt(n) + UInt(j)]"
                )

    return b.build()


def emit_mixed_forward_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate NCOLS-specialized shmem-tiled mixed forward matvec GPU kernel.

    Evaluates the full mixed IR tree, where categorical leaves are looked up as
    ``cat_k`` variables and composed with continuous terms via arbitrary sums/products.

    Shared memory layout per thread slot: [x_j[DIM] | v_j[NCOLS] | cat_j[MAX_CAT_VARS]]
    where MAX_CAT_VARS=8 is a compile-time upper bound; actual num_cat_vars is runtime.
    cat indices stored as Float32 in shmem (Int32 values fit exactly for small counts).
    cat_indices layout in global memory: cat_indices[cv * n + i] (variable-major).
    """
    b = MojoBuilder()
    MAX_CAT_VARS = 8
    b.blank()
    b.comment("=" * 77)
    b.comment("Mixed Forward Matvec — NCOLS specialization (auto-generated)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_mixed_forward_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("cat_indices: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("levels_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("num_cat_vars: Int,")
        b.line("n: Int, noise: Float32,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Mixed fwd shmem-tiled: evaluate arbitrary mixed kernel IR."""')
        b.line(f"alias MAX_CAT_VARS = {MAX_CAT_VARS}")
        b.line("alias DIMY = DIM + NCOLS + MAX_CAT_VARS")
        b.line("var tid = Int(thread_idx.x)")
        b.line("var bs = Int(block_dim.x)")
        b.line("var i = Int(block_idx.x) * bs + tid")
        b.line(
            "var yj = external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16]()"
        )
        b.line("var valid = i < n")
        b.blank()

        b.comment(f"Pre-load {kernel.num_params} params into registers")
        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.comment("Pre-load cat indices for row i into registers")
        b.line(f"var cat_i = InlineArray[Int32, MAX_CAT_VARS](fill=Int32(0))")
        with b.block("if valid:"):
            b.line("var ro = UInt(i) * UInt(DIM)")
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("x_row[d] = x_ptr[ro + UInt(d)]")
            with b.block("for cv in range(num_cat_vars):"):
                b.line("cat_i[cv] = cat_indices[cv * n + i]")
        b.blank()

        b.line("var acc = InlineArray[Float32, NCOLS](fill=Float32(0.0))")
        b.line("var jstart = 0")
        with b.block("while jstart < n:"):
            b.line("var j = jstart + tid")
            with b.block("if j < n:"):
                b.line("var sb = tid * DIMY")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("yj[sb + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]")
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line("yj[sb + DIM + c] = v_ptr[UInt(c) * UInt(n) + UInt(j)]")
                b.comment("Store cat_j indices as Float32 (small integers fit exactly)")
                with b.block("for cv in range(num_cat_vars):"):
                    b.line(
                        f"yj[sb + DIM + NCOLS + cv] = Float32(cat_indices[cv * n + j])"
                    )
            b.line("barrier()")
            with b.block("if valid:"):
                b.line("var te = bs if jstart + bs <= n else n - jstart")
                with b.block("for jrel in range(te):"):
                    b.line("var sb = jrel * DIMY")
                    _emit_intermediates(b, kernel)
                    _emit_xy_aliases(b, kernel)
                    _emit_mixed_cat_var_bindings(
                        b,
                        kernel,
                        "Int(cat_i[{cat_idx}])",
                        "Int(yj[sb + DIM + NCOLS + {cat_idx}])",
                    )
                    _emit_let_bindings(b, kernel)
                    b.line(f"var k_val = {emit_ir(kernel.forward)}")
                    b.line("@parameter")
                    with b.block("for c in range(NCOLS):"):
                        b.line("acc[c] += k_val * yj[sb + DIM + c]")
            b.line("barrier()")
            b.line("jstart += bs")

        with b.block("if valid:"):
            b.line("@parameter")
            with b.block("for c in range(NCOLS):"):
                b.line("var off = UInt(c) * UInt(n) + UInt(i)")
                b.line("out_ptr[off] = acc[c] + noise * v_ptr[off]")

    return b.build()


def emit_mixed_gradient_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate NCOLS-specialized shmem-tiled mixed all-gradients matvec.

    Computes ``(dK/dtheta_p) @ v`` for the full mixed IR tree. The JIT parameter
    vector still covers only codegen-managed params; wrapper-managed categorical
    params remain outside this symbolic gradient path.
    Output layout: out[p_idx * n * NCOLS + c * n + i].
    Same shmem layout as forward: [x_j[DIM] | v_j[NCOLS] | cat_j[MAX_CAT_VARS]].
    """
    b = MojoBuilder()
    MAX_CAT_VARS = 8
    b.blank()
    b.comment("=" * 77)
    b.comment("Mixed All-Gradients Matvec — NCOLS specialization (auto-generated)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_mixed_all_gradients_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("cat_indices: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("levels_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("num_cat_vars: Int,")
        b.line("n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Mixed fused all-gradients matvec for arbitrary mixed IR."""')
        b.line(f"alias MAX_CAT_VARS = {MAX_CAT_VARS}")
        b.line("alias DIMY = DIM + NCOLS + MAX_CAT_VARS")
        b.line(f"alias NPARAMS = {kernel.num_params}")
        b.line("var tid = Int(thread_idx.x)")
        b.line("var bs = Int(block_dim.x)")
        b.line("var i = Int(block_idx.x) * bs + tid")
        b.line(
            "var yj = external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16]()"
        )
        b.line("var valid = i < n")
        b.blank()

        b.line("var p = InlineArray[Float32, NPARAMS](uninitialized=True)")
        b.line("@parameter")
        with b.block("for pi in range(NPARAMS):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line(f"var cat_i = InlineArray[Int32, MAX_CAT_VARS](fill=Int32(0))")
        with b.block("if valid:"):
            b.line("var ro = UInt(i) * UInt(DIM)")
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("x_row[d] = x_ptr[ro + UInt(d)]")
            with b.block("for cv in range(num_cat_vars):"):
                b.line("cat_i[cv] = cat_indices[cv * n + i]")
        b.blank()

        b.line(
            "var grad_acc = InlineArray[Float32, NPARAMS * NCOLS](fill=Float32(0.0))"
        )
        b.blank()

        b.line("var jstart = 0")
        with b.block("while jstart < n:"):
            b.line("var j = jstart + tid")
            with b.block("if j < n:"):
                b.line("var sb = tid * DIMY")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("yj[sb + d] = x_ptr[UInt(j) * UInt(DIM) + UInt(d)]")
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line("yj[sb + DIM + c] = v_ptr[UInt(c) * UInt(n) + UInt(j)]")
                with b.block("for cv in range(num_cat_vars):"):
                    b.line(
                        f"yj[sb + DIM + NCOLS + cv] = Float32(cat_indices[cv * n + j])"
                    )
            b.line("barrier()")
            with b.block("if valid:"):
                b.line("var te = bs if jstart + bs <= n else n - jstart")
                with b.block("for jrel in range(te):"):
                    b.line("var sb = jrel * DIMY")
                    _emit_intermediates(b, kernel)
                    _emit_xy_aliases(b, kernel)
                    _emit_mixed_cat_var_bindings(
                        b,
                        kernel,
                        "Int(cat_i[{cat_idx}])",
                        "Int(yj[sb + DIM + NCOLS + {cat_idx}])",
                    )
                    _emit_let_bindings(b, kernel)
                    b.line(
                        "var grad = InlineArray[Float32, NPARAMS](fill=Float32(0.0))"
                    )
                    for p_idx in sorted(kernel.gradients.keys()):
                        b.line(f"grad[{p_idx}] = {emit_ir(kernel.gradients[p_idx])}")
                    b.line("@parameter")
                    with b.block("for p_idx in range(NPARAMS):"):
                        b.line("var gv = grad[p_idx]")
                        b.line("@parameter")
                        with b.block("for c in range(NCOLS):"):
                            b.line(
                                "grad_acc[p_idx * NCOLS + c] += gv * yj[sb + DIM + c]"
                            )
            b.line("barrier()")
            b.line("jstart += bs")

        with b.block("if valid:"):
            b.line("@parameter")
            with b.block("for p_idx in range(NPARAMS):"):
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line(
                        "var off = UInt(p_idx) * UInt(n) * UInt(NCOLS) + UInt(c) * UInt(n) + UInt(i)"
                    )
                    b.line("out_ptr[off] = grad_acc[p_idx * NCOLS + c]")

    return b.build()


def emit_mixed_cross_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate NCOLS-specialized shmem-tiled mixed cross-covariance matvec.

    Evaluates the full mixed IR tree for ``(x_test_i, x_train_j)`` pairs.
    Shmem layout: [x_train_j[DIM] | v_j[NCOLS] | cat_train_j[MAX_CAT_VARS]]
    cat_test loaded into registers (indexed variable-major: cat[cv * n_test + i]).
    """
    b = MojoBuilder()
    MAX_CAT_VARS = 8
    b.blank()
    b.comment("=" * 77)
    b.comment("Mixed Cross-Covariance Matvec — NCOLS specialization (auto-generated)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_mixed_cross_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_train_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("cat_test: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("cat_train: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("levels_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("num_cat_vars: Int,")
        b.line("n_test: Int, n_train: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Mixed cross-covariance for arbitrary mixed kernel IR."""')
        b.line(f"alias MAX_CAT_VARS = {MAX_CAT_VARS}")
        b.line("alias DIMY = DIM + NCOLS + MAX_CAT_VARS")
        b.line("var tid = Int(thread_idx.x)")
        b.line("var bs = Int(block_dim.x)")
        b.line("var i = Int(block_idx.x) * bs + tid")
        b.line(
            "var yj = external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16]()"
        )
        b.line("var valid = i < n_test")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line(f"var cat_i = InlineArray[Int32, MAX_CAT_VARS](fill=Int32(0))")
        with b.block("if valid:"):
            b.line("var ro = UInt(i) * UInt(DIM)")
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("x_row[d] = x_test_ptr[ro + UInt(d)]")
            with b.block("for cv in range(num_cat_vars):"):
                b.line("cat_i[cv] = cat_test[cv * n_test + i]")
        b.blank()

        b.line("var acc = InlineArray[Float32, NCOLS](fill=Float32(0.0))")
        b.line("var jstart = 0")
        with b.block("while jstart < n_train:"):
            b.line("var j = jstart + tid")
            with b.block("if j < n_train:"):
                b.line("var sb = tid * DIMY")
                b.line("@parameter")
                with b.block("for d in range(DIM):"):
                    b.line("yj[sb + d] = x_train_ptr[UInt(j) * UInt(DIM) + UInt(d)]")
                b.line("@parameter")
                with b.block("for c in range(NCOLS):"):
                    b.line(
                        "yj[sb + DIM + c] = v_ptr[UInt(c) * UInt(n_train) + UInt(j)]"
                    )
                with b.block("for cv in range(num_cat_vars):"):
                    b.line(
                        f"yj[sb + DIM + NCOLS + cv] = Float32(cat_train[cv * n_train + j])"
                    )
            b.line("barrier()")
            with b.block("if valid:"):
                b.line("var te = bs if jstart + bs <= n_train else n_train - jstart")
                with b.block("for jrel in range(te):"):
                    b.line("var sb = jrel * DIMY")
                    _emit_intermediates(b, kernel)
                    _emit_xy_aliases(b, kernel)
                    _emit_mixed_cat_var_bindings(
                        b,
                        kernel,
                        "Int(cat_i[{cat_idx}])",
                        "Int(yj[sb + DIM + NCOLS + {cat_idx}])",
                    )
                    _emit_let_bindings(b, kernel)
                    b.line(f"var k_val = {emit_ir(kernel.forward)}")
                    b.line("@parameter")
                    with b.block("for c in range(NCOLS):"):
                        b.line("acc[c] += k_val * yj[sb + DIM + c]")
            b.line("barrier()")
            b.line("jstart += bs")

        with b.block("if valid:"):
            b.line("@parameter")
            with b.block("for c in range(NCOLS):"):
                b.line("out_ptr[UInt(c) * UInt(n_test) + UInt(i)] = acc[c]")

    return b.build()


def emit_mixed_materialize(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate GPU kernel to fill the full mixed kernel matrix.

    Evaluates the full mixed IR tree at each ``(i, j)`` pair.
    """
    b = MojoBuilder()
    MAX_CAT_VARS = 8
    b.blank()
    b.comment("=" * 77)
    b.comment("Fill Mixed Kernel Matrix (materialized mixed mode, auto-generated)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_mixed_fill_kernel_matrix[DIM: Int](")
    with b.block():
        b.line("k_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("cat_indices: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("levels_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("num_cat_vars: Int,")
        b.line("n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Fill K_mixed[i,j] for all i,j. One thread per matrix entry."""')
        b.line(f"alias MAX_CAT_VARS = {MAX_CAT_VARS}")
        b.line("var i = Int(block_idx.y) * Int(block_dim.y) + Int(thread_idx.y)")
        b.line("var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n or j >= n:"):
            b.line("return")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("var joff = UInt(j) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()

        if kernel.needs_diffs:
            b.line("var diffs = InlineArray[Float32, DIM](uninitialized=True)")
        if kernel.needs_dist_sq:
            b.line("var dist_sq = Float32(0)")
        if kernel.needs_dot:
            b.line("var dot_prod = Float32(0)")

        if kernel.needs_diffs:
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("diffs[d] = x_row[d] - x_ptr[joff + UInt(d)]")
                if kernel.needs_dist_sq:
                    b.line("dist_sq += diffs[d] * diffs[d]")
                if kernel.needs_dot:
                    b.line("dot_prod += x_row[d] * x_ptr[joff + UInt(d)]")
            if kernel.dim > 0:
                for d in range(kernel.dim):
                    b.line(f"var diff_{d} = diffs[{d}]")
        elif kernel.needs_dot:
            b.line("@parameter")
            with b.block("for d in range(DIM):"):
                b.line("dot_prod += x_row[d] * x_ptr[joff + UInt(d)]")

        b.line("var sb = Int(joff)")
        b.line("var yj = x_ptr")
        _emit_xy_aliases(b, kernel)
        _emit_mixed_cat_var_bindings(
            b,
            kernel,
            "Int(cat_indices[{cat_idx} * n + i])",
            "Int(cat_indices[{cat_idx} * n + j])",
        )
        _emit_let_bindings(b, kernel)
        b.line(f"var k_val = {emit_ir(kernel.forward)}")
        b.line("k_ptr[UInt(i) * UInt(n) + UInt(j)] = k_val")

    return b.build()


def emit_mixed_extract_diagonal(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate mixed diagonal extraction kernel.

    Evaluates the full mixed IR tree at ``(x_i, x_i)``.
    """
    b = MojoBuilder()
    MAX_CAT_VARS = 8
    b.blank()
    b.comment("=" * 77)
    b.comment("Mixed Extract Diagonal (auto-generated for mixed prediction)")
    b.comment("=" * 77)
    b.blank()

    b.line("fn fused_mixed_extract_diagonal[DIM: Int](")
    with b.block():
        b.line("diag_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("cat_indices: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("levels_ptr: UnsafePointer[Int32, MutAnyOrigin],")
        b.line("num_cat_vars: Int,")
        b.line("n: Int,")
    b.line(") -> None:")
    with b.block():
        b.line(
            '"""Mixed diagonal: evaluate arbitrary mixed kernel IR at self-pairs."""'
        )
        b.line(f"alias MAX_CAT_VARS = {MAX_CAT_VARS}")
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n:"):
            b.line("return")
        b.blank()

        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()

        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()

        b.comment("Self-evaluation: diffs=0, dist_sq=0, dot_prod=x^T@x")
        if kernel.needs_diffs:
            b.line("var diffs = InlineArray[Float32, DIM](fill=Float32(0.0))")
            if kernel.dim > 0:
                for d in range(kernel.dim):
                    b.line(f"var diff_{d} = Float32(0.0)")
        b.line("var dist_sq = Float32(0.0)")
        b.line("var dot_prod = Float32(0.0)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("dot_prod += x_row[d] * x_row[d]")
        b.blank()

        b.comment("yj/sb aliases for eval code that references yj[sb+d]")
        b.line("var sb = Int(ro)")
        b.line("var yj = x_ptr")
        b.blank()

        _emit_xy_aliases(b, kernel)
        _emit_mixed_cat_var_bindings(
            b,
            kernel,
            "Int(cat_indices[{cat_idx} * n + i])",
            "Int(cat_indices[{cat_idx} * n + i])",
        )
        _emit_let_bindings(b, kernel)
        b.line(f"var k_self = {emit_ir(kernel.forward)}")
        b.line("diag_ptr[i] = k_self")

    return b.build()


def emit_kronecker_forward_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate fused Kronecker forward matvec GPU kernel — NCOLS specialization.

    Restructures from col→s→t→j (kernel recomputed T²×num_cols times per row)
    to j→NCOLS×T accumulators (kernel computed ONCE per j).
    """
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("Fused Kronecker Forward Matvec — NCOLS specialization (auto-generated)")
    b.comment("=" * 77)
    b.blank()

    # MAX_T: compile-time bound on number of output tasks.
    MAX_T = 8

    b.line("fn fused_kronecker_forward_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("B_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("noise_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int, T: Int, num_cols: Int, outputscale: Float32,")
        b.line("col_offset: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Kronecker fwd: (os*K_X⊗B+D)@v. Kernel computed once per j."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n:"):
            b.line("return")
        b.line("var nT = n * T")
        b.blank()
        b.line(f"var p = InlineArray[Float32, {kernel.num_params}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({kernel.num_params}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()
        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()
        # kv[c * MAX_T + t] = sum_j( K(xi,xj) * v[col_offset+c, t, j] )
        b.line(f"var kv = InlineArray[Float32, NCOLS * {MAX_T}](fill=Float32(0.0))")
        b.blank()
        # j outermost — kernel computed ONCE per j
        with b.block("for j in range(n):"):
            b.line("var sb = Int(UInt(j) * UInt(DIM))")
            b.line("var yj = x_ptr")
            _emit_intermediates(b, kernel)
            _emit_xy_aliases(b, kernel)
            _emit_default_cat_var_bindings(b, kernel)
            _emit_let_bindings(b, kernel)
            b.line(f"var k_val = {emit_ir(kernel.forward)}")
            b.line("@parameter")
            with b.block("for c in range(NCOLS):"):
                b.line("var col = col_offset + c")
                with b.block("for t in range(T):"):
                    b.line(
                        f"kv[c * {MAX_T} + t] += k_val * v_ptr[UInt(col) * UInt(nT) + UInt(t) * UInt(n) + UInt(j)]"
                    )
        b.blank()
        # Combine with B matrix and write output
        b.line("@parameter")
        with b.block("for c in range(NCOLS):"):
            b.line("var col = col_offset + c")
            with b.block("if col < num_cols:"):
                with b.block("for s in range(T):"):
                    b.line("var acc = Float32(0.0)")
                    with b.block("for t in range(T):"):
                        b.line(f"acc += B_ptr[s * T + t] * kv[c * {MAX_T} + t]")
                    b.line(
                        "var out_idx = UInt(col) * UInt(nT) + UInt(s) * UInt(n) + UInt(i)"
                    )
                    b.line(
                        "out_ptr[out_idx] = outputscale * acc + noise_ptr[s] * v_ptr[out_idx]"
                    )
    return b.build()


def emit_kronecker_gradient_matvec(kernel: IRKernel, schedule: ScheduleConfig) -> str:
    """Generate fused Kronecker gradient matvec GPU kernel — NCOLS specialization.

    Restructures from col→s→t→j (kernel recomputed T²×num_cols times per row)
    to j→NCOLS×T×NPARAMS accumulators (kernel computed ONCE per j).
    Uses MAX_T=4 for gradient to keep register count manageable.
    """
    b = MojoBuilder()
    b.blank()
    b.comment("=" * 77)
    b.comment("Fused Kronecker Gradient Matvec — NCOLS specialization (auto-generated)")
    b.comment("=" * 77)
    b.blank()

    # MAX_T_GRAD: compile-time bound. Smaller than forward to save registers
    # since we also carry NPARAMS accumulators per (col, t) slot.
    MAX_T_GRAD = 4
    num_grads = kernel.num_params

    b.line("fn fused_kronecker_gradient_matvec_ncols[DIM: Int, NCOLS: Int](")
    with b.block():
        b.line("out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("v_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("x_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("params_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("B_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("n: Int, T: Int, num_cols: Int, outputscale: Float32,")
        b.line("col_offset: Int,")
    b.line(") -> None:")
    with b.block():
        b.line('"""Kronecker grad: all params x B. Kernel computed once per j."""')
        b.line("var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)")
        with b.block("if i >= n:"):
            b.line("return")
        b.line("var nT = n * T")
        b.blank()
        b.line(f"var p = InlineArray[Float32, {num_grads}](uninitialized=True)")
        b.line("@parameter")
        with b.block(f"for pi in range({num_grads}):"):
            b.line("p[pi] = params_ptr[pi]")
        b.blank()
        b.line("var x_row = InlineArray[Float32, DIM](uninitialized=True)")
        b.line("var ro = UInt(i) * UInt(DIM)")
        b.line("@parameter")
        with b.block("for d in range(DIM):"):
            b.line("x_row[d] = x_ptr[ro + UInt(d)]")
        b.blank()
        # kv_g[c * MAX_T_GRAD * NPARAMS + t * NPARAMS + ki]
        # = sum_j( dK/dtheta_ki(xi,xj) * v[col_offset+c, t, j] )
        b.line(
            f"var kv_g = InlineArray[Float32, NCOLS * {MAX_T_GRAD} * {num_grads}](fill=Float32(0.0))"
        )
        # Temp array for per-j gradient values
        b.line(f"var grad = InlineArray[Float32, {num_grads}](uninitialized=True)")
        b.blank()
        # j outermost — kernel gradients computed ONCE per j
        with b.block("for j in range(n):"):
            b.line("var sb = Int(UInt(j) * UInt(DIM))")
            b.line("var yj = x_ptr")
            _emit_intermediates(b, kernel)
            _emit_xy_aliases(b, kernel)
            _emit_default_cat_var_bindings(b, kernel)
            _emit_let_bindings(b, kernel)
            for p_idx in sorted(kernel.gradients.keys()):
                b.line(f"grad[{p_idx}] = {emit_ir(kernel.gradients[p_idx])}")
            b.line("@parameter")
            with b.block("for c in range(NCOLS):"):
                b.line("var col = col_offset + c")
                with b.block("for t in range(T):"):
                    b.line(
                        "var v_val = v_ptr[UInt(col) * UInt(nT) + UInt(t) * UInt(n) + UInt(j)]"
                    )
                    b.line("@parameter")
                    with b.block(f"for ki in range({num_grads}):"):
                        b.line(
                            f"kv_g[c * {MAX_T_GRAD} * {num_grads} + t * {num_grads} + ki] += grad[ki] * v_val"
                        )
        b.blank()
        # Combine with B matrix and write output
        # Output layout: out[ki, col, s, i] = out_ptr[ki*nT*num_cols + col*nT + s*n + i]
        b.line("@parameter")
        with b.block("for c in range(NCOLS):"):
            b.line("var col = col_offset + c")
            with b.block("if col < num_cols:"):
                with b.block("for s in range(T):"):
                    b.line(
                        "var out_base = UInt(col) * UInt(nT) + UInt(s) * UInt(n) + UInt(i)"
                    )
                    b.line("@parameter")
                    with b.block(f"for ki in range({num_grads}):"):
                        b.line("var param_acc = Float32(0.0)")
                        with b.block("for t in range(T):"):
                            b.line(
                                f"param_acc += B_ptr[s * T + t] * kv_g[c * {MAX_T_GRAD} * {num_grads} + t * {num_grads} + ki]"
                            )
                        b.line(
                            "out_ptr[UInt(ki) * UInt(nT) * UInt(num_cols) + out_base] = outputscale * param_acc"
                        )
    return b.build()
