"""Tests for the codegen engine expression layer."""

import pytest
import sympy as sp


class TestKernelExprBuilder:
    """Test SymPy expression building for all 8 kernel types."""

    def test_rbf_iso(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        expr = b.rbf(ard=False)
        assert len(expr.params) == 2
        assert expr.needs_dist_sq is True
        assert expr.needs_dot is False
        assert expr.kernel_type == "rbf"
        assert expr.is_ard is False

    def test_rbf_ard(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=3)
        expr = b.rbf(ard=True)
        assert len(expr.params) == 4  # 3 ls + 1 os
        assert expr.needs_dist_sq is False  # ARD computes own
        assert expr.is_ard is True

    def test_matern52_iso(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        expr = b.matern52(ard=False)
        assert len(expr.params) == 2
        assert expr.needs_dist_sq is True

    def test_periodic_iso(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        expr = b.periodic(ard=False)
        assert len(expr.params) == 3  # ls, period, os
        assert expr.needs_dist_sq is False  # periodic uses sin
        assert expr.needs_dot is False

    def test_linear_iso(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        expr = b.linear(ard=False)
        assert len(expr.params) == 2
        assert expr.needs_dot is True
        assert expr.needs_dist_sq is False

    def test_polynomial_ard(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=3)
        expr = b.polynomial(ard=True)
        assert len(expr.params) == 6  # 3 var + degree + offset + os

    def test_rq_iso(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        expr = b.rq(ard=False)
        assert len(expr.params) == 3  # ls, alpha, os

    def test_sum_composition(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        k1 = b.rbf(ard=False, param_offset=0)
        k2 = b.matern52(ard=False, param_offset=2)
        s = b.sum(k1, k2)
        assert len(s.params) == 4
        assert s.needs_dist_sq is True

    def test_product_composition(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        k1 = b.rbf(ard=False, param_offset=0)
        k2 = b.linear(ard=False, param_offset=2)
        p = b.product(k1, k2)
        assert len(p.params) == 4
        assert p.needs_dist_sq is True
        assert p.needs_dot is True

    def test_all_8_kernels_build(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        b = KernelExprBuilder(dim=5)
        kernels = [
            b.rbf,
            b.matern12,
            b.matern32,
            b.matern52,
            b.periodic,
            b.rq,
            b.linear,
            b.polynomial,
        ]
        for fn in kernels:
            for ard in [False, True]:
                expr = fn(ard=ard)
                assert len(expr.params) > 0
                assert expr.forward is not None


class TestDifferentiation:
    """Test gradient computation."""

    def test_rbf_gradients(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder
        from mojogp.codegen_engine.differentiation import compute_gradients

        b = KernelExprBuilder(dim=3)
        expr = b.rbf(ard=False)
        grad = compute_gradients(expr)
        assert len(grad.gradients) == 2
        # Both gradients should be non-zero expressions
        for param, g in grad.gradients.items():
            assert g != 0

    def test_product_gradient_count(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder
        from mojogp.codegen_engine.differentiation import compute_gradients

        b = KernelExprBuilder(dim=3)
        k1 = b.rbf(ard=False, param_offset=0)
        k2 = b.matern52(ard=False, param_offset=2)
        prod = b.product(k1, k2)
        grad = compute_gradients(prod)
        assert len(grad.gradients) == 4  # 2 + 2 params


class TestIR:
    """Test IR conversion."""

    def test_sympy_to_ir_roundtrip(self):
        from mojogp.codegen_engine.expressions import KernelExprBuilder
        from mojogp.codegen_engine.differentiation import compute_gradients
        from mojogp.codegen_engine.ir import to_ir

        b = KernelExprBuilder(dim=3)
        expr = b.rbf(ard=False)
        grad = compute_gradients(expr)
        ir = to_ir(grad, dim=3, needs_dist_sq=True)
        assert ir.num_params == 2
        assert ir.dim == 3
        assert ir.forward is not None
        assert len(ir.gradients) == 2

    def test_collect_functions(self):
        from mojogp.codegen_engine.ir import (
            collect_functions,
            UnaryFn,
            BinOp,
            Const,
            Var,
        )

        expr = UnaryFn("cos", BinOp("*", Const(3.14), Var("x")))
        fns = collect_functions(expr)
        assert "cos" in fns


class TestPasses:
    """Test optimization passes."""

    def test_strength_reduce_pow2(self):
        from mojogp.codegen_engine.ir import Pow, Var, Const, BinOp
        from mojogp.codegen_engine.passes.strength import _reduce_expr

        expr = Pow(Var("x"), Const(2.0))
        result = _reduce_expr(expr)
        assert isinstance(result, BinOp)
        assert result.op == "*"

    def test_strength_reduce_pow_half(self):
        from mojogp.codegen_engine.ir import Pow, Var, Const, UnaryFn
        from mojogp.codegen_engine.passes.strength import _reduce_expr

        expr = Pow(Var("x"), Const(0.5))
        result = _reduce_expr(expr)
        assert isinstance(result, UnaryFn)
        assert result.fn == "sqrt"


class TestSchedule:
    """Test schedule planner."""

    def test_low_dim_simple_rbf_defaults_to_tm1(self):
        from mojogp.codegen_engine.ir import IRKernel, Var
        from mojogp.codegen_engine.schedule import plan_schedule

        kernel = IRKernel(
            forward=Var("k"),
            gradients={0: Var("g0"), 1: Var("g1")},
            num_params=2,
            param_names=["ls", "os"],
            needs_diffs=True,
            needs_dist_sq=True,
            needs_dot=False,
            dim=5,
        )
        schedule = plan_schedule(kernel)
        assert schedule.tm == 1

    def test_low_dim_stationary_leaf_tag_adds_ncols10(self):
        from mojogp.codegen_engine.ir import IRKernel, Var
        from mojogp.codegen_engine.schedule import plan_schedule

        kernel = IRKernel(
            forward=Var("k"),
            gradients={i: Var(f"g{i}") for i in range(10)},
            num_params=10,
            param_names=[f"p{i}" for i in range(10)],
            needs_diffs=True,
            needs_dist_sq=True,
            needs_dot=False,
            dim=9,
        )
        unscoped = plan_schedule(kernel)
        scoped = plan_schedule(kernel, schedule_policy_tag="low_d_stationary_leaf")

        assert 10 not in unscoped.ncols
        assert scoped.tm == 1
        assert 10 in scoped.ncols

    def test_rbf_leaf_tag_adds_ncols10_through_d17_only(self):
        from mojogp.codegen_engine.ir import IRKernel, Var
        from mojogp.codegen_engine.schedule import plan_schedule

        def kernel_for_dim(d):
            return IRKernel(
                forward=Var("k"),
                gradients={i: Var(f"g{i}") for i in range(10)},
                num_params=10,
                param_names=[f"p{i}" for i in range(10)],
                needs_diffs=True,
                needs_dist_sq=True,
                needs_dot=False,
                dim=d,
            )

        scoped_d17 = plan_schedule(kernel_for_dim(17), schedule_policy_tag="rbf_leaf")
        scoped_d31 = plan_schedule(kernel_for_dim(31), schedule_policy_tag="rbf_leaf")

        assert scoped_d17.tm == 1
        assert 10 in scoped_d17.ncols
        assert 10 not in scoped_d31.ncols

    def test_rbf_leaf_splitj_default_is_scoped_to_non_ard_measured_dims(self):
        from mojogp.codegen_engine.ir import IRKernel, Var
        from mojogp.codegen_engine.schedule import plan_schedule

        non_ard = IRKernel(
            forward=Var("k"),
            gradients={0: Var("g0"), 1: Var("g1")},
            num_params=2,
            param_names=["ls", "os"],
            needs_diffs=True,
            needs_dist_sq=True,
            needs_dot=False,
            dim=17,
        )
        ard_like = IRKernel(
            forward=Var("k"),
            gradients={i: Var(f"g{i}") for i in range(18)},
            num_params=18,
            param_names=[f"l_{i}" for i in range(17)] + ["os"],
            needs_diffs=True,
            needs_dist_sq=False,
            needs_dot=False,
            dim=17,
        )
        high_dim = IRKernel(
            forward=Var("k"),
            gradients={0: Var("g0"), 1: Var("g1")},
            num_params=2,
            param_names=["ls", "os"],
            needs_diffs=True,
            needs_dist_sq=True,
            needs_dot=False,
            dim=31,
        )

        assert plan_schedule(non_ard, schedule_policy_tag="rbf_leaf").splitj_forward is True
        assert plan_schedule(ard_like, schedule_policy_tag="rbf_leaf").splitj_forward is False
        assert plan_schedule(high_dim, schedule_policy_tag="rbf_leaf").splitj_forward is False


class TestEmitter:
    """Test Mojo code emission."""

    def test_builder_indentation(self):
        from mojogp.codegen_engine.emit.builder import MojoBuilder

        b = MojoBuilder()
        b.line("fn foo():")
        with b.block():
            b.line("var x = 1")
            with b.block("if x > 0:"):
                b.line("return x")
        result = b.build()
        lines = result.split("\n")
        assert lines[0] == "fn foo():"
        assert lines[1] == "    var x = 1"
        assert lines[2] == "    if x > 0:"
        assert lines[3] == "        return x"

    def test_emit_ir_const(self):
        from mojogp.codegen_engine.emit.mojo_printer import emit_ir
        from mojogp.codegen_engine.ir import Const

        assert emit_ir(Const(2.0)) == "Float32(2)"
        assert emit_ir(Const(3.14)) == "Float32(3.14)"

    def test_emit_ir_param(self):
        from mojogp.codegen_engine.emit.mojo_printer import emit_ir
        from mojogp.codegen_engine.ir import Param

        assert emit_ir(Param(0)) == "p[0]"
        assert emit_ir(Param(5, "os")) == "p[5]"

    def test_emit_ir_unary(self):
        from mojogp.codegen_engine.emit.mojo_printer import emit_ir
        from mojogp.codegen_engine.ir import UnaryFn, Var

        result = emit_ir(UnaryFn("exp", Var("x")))
        assert result == "math_exp(x)"

    def test_collect_math_imports(self):
        from mojogp.codegen_engine.emit.mojo_printer import collect_math_imports
        from mojogp.codegen_engine.ir import IRKernel, UnaryFn, Var, BinOp

        kernel = IRKernel(
            forward=UnaryFn("exp", Var("x")),
            gradients={0: UnaryFn("cos", Var("y"))},
            num_params=1,
            param_names=["p"],
            needs_diffs=True,
            needs_dist_sq=False,
            needs_dot=False,
            dim=3,
        )
        imports = collect_math_imports(kernel)
        assert "cos" in imports
        assert "exp as math_exp" in imports


class TestEndToEnd:
    """Test the full pipeline from KernelNode to Mojo source."""

    def test_rbf_generates_code(self):
        """RBF kernel generates valid-looking Mojo source."""
        from mojogp.codegen_engine.expressions import KernelExprBuilder
        from mojogp.codegen_engine.differentiation import compute_gradients
        from mojogp.codegen_engine.ir import to_ir
        from mojogp.codegen_engine.passes import optimize
        from mojogp.codegen_engine.schedule import plan_schedule
        from mojogp.codegen_engine.emit import emit_module

        b = KernelExprBuilder(dim=5)
        expr = b.rbf(ard=False)
        grad = compute_gradients(expr)
        ir = to_ir(grad, dim=5, needs_diffs=True, needs_dist_sq=True)
        ir = optimize(ir)
        schedule = plan_schedule(ir)
        code = emit_module(
            ir, schedule, "test_rbf", kernel_type_str="RBFComposable", dim=5
        )

        assert "fn fused_forward_matvec_ncols" in code
        assert "fn fused_all_gradients_matvec_ncols" in code
        assert "fn fused_cross_matvec_ncols" in code
        assert "fn fused_extract_diagonal" in code
        assert "JITAdapter" in code
        assert "PyInit_test_rbf" in code
        assert "math_exp" in code

    def test_rbf_fn_ptr_generates_scoped_splitj_default(self):
        from mojogp import RBF
        from mojogp.codegen_engine import generate_fn_ptr_module

        code = generate_fn_ptr_module(RBF(), dim=5, module_name="test_rbf_fn_ptr")

        assert "fn _use_splitj_forward(n: Int)" in code
        assert "return True and n >= 2000" in code
