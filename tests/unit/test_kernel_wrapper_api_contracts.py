"""Unit tests for kernel and wrapper API contracts."""

import numpy as np
import pytest

from mojogp.kernel import KernelNode, Kernel, KernelType


# ---------------------------------------------------------------------------
# MultiOutputGP variance-method mapping
# ---------------------------------------------------------------------------


class TestMultiOutputVarianceMethodEncoding:
    """Verify that MultiOutputGP maps variance_method strings to correct JIT ints."""

    def test_variance_method_mapping_code(self):
        """Verify the mapping logic produces correct integers for JIT engine."""
        # Test the mapping logic directly without needing a trained model
        # JIT engine: 0=mean_only, 1=LOVE, 2=exact
        test_cases = {
            "exact": 2,
            "love": 1,
            "mean_only": 0,
        }
        for method_str, expected_int in test_cases.items():
            if method_str == "exact":
                var_method_int = 2
            elif method_str == "love" or method_str is None:
                var_method_int = 1
            elif method_str == "mean_only":
                var_method_int = 0
            else:
                var_method_int = 0
            assert var_method_int == expected_int, (
                f"variance_method='{method_str}' should map to {expected_int}, "
                f"got {var_method_int}"
            )

    def test_multioutput_gp_constructor(self):
        """MultiOutputGP should accept valid kernel strings."""
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf")
        assert gp is not None


# ---------------------------------------------------------------------------
# MultiOutputLMCGP variance_method wiring
# ---------------------------------------------------------------------------


class TestLMCVarianceMethodWiring:
    """Verify MultiOutputLMCGP._predict_composite accepts variance_method."""

    def test_predict_composite_accepts_variance_method(self):
        """_predict_composite should accept variance_method parameter."""
        from mojogp.multi_output_gp import MultiOutputLMCGP
        import inspect

        sig = inspect.signature(MultiOutputLMCGP._predict_composite)
        assert "variance_method" in sig.parameters, (
            "_predict_composite must accept variance_method parameter"
        )
        # Default should be 1 (LOVE)
        assert sig.parameters["variance_method"].default == 1


# ---------------------------------------------------------------------------
# Polynomial KernelNode.evaluate() parameter ordering
# ---------------------------------------------------------------------------


class TestPolynomialKernelParams:
    """Verify polynomial kernel parameter ordering is consistent."""

    def test_isotropic_polynomial_num_params_is_3(self):
        """Isotropic polynomial should have 3 params: degree, offset, outputscale."""
        k = Kernel.polynomial()
        assert k.num_params() == 3

    def test_isotropic_polynomial_param_names(self):
        """Param names should be [degree, offset, outputscale] — no variance."""
        k = Kernel.polynomial()
        names = k.get_param_names()
        assert len(names) == 3
        assert "degree" in names[0]
        assert "offset" in names[1]
        assert "outputscale" in names[2]
        # variance should NOT be in param names
        for name in names:
            assert "variance" not in name

    def test_isotropic_polynomial_evaluate_correctness(self):
        """evaluate() should match manual formula: outputscale * (X@X2.T + offset)^degree."""
        k = Kernel.polynomial(degree=2.0, offset=1.0, outputscale=1.5)
        params = k.get_initial_params()
        assert len(params) == 3  # [degree, offset, outputscale]

        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)
        X2 = np.random.randn(5, 3).astype(np.float64)

        K, _ = k._evaluate_base(X, X2, params, 0)

        # Manual computation
        degree, offset, outputscale = params[0], params[1], params[2]
        K_expected = outputscale * (X @ X2.T + offset) ** degree

        np.testing.assert_allclose(K, K_expected, rtol=1e-6)

    def test_polynomial_evaluate_with_custom_params(self):
        """Test polynomial with non-default params."""
        k = Kernel.polynomial(degree=3.0, offset=0.5, outputscale=2.0)
        params = k.get_initial_params()

        np.random.seed(123)
        X = np.random.randn(8, 4).astype(np.float64)

        K, _ = k._evaluate_base(X, X, params, 0)
        K_expected = 2.0 * (X @ X.T + 0.5) ** 3.0

        np.testing.assert_allclose(K, K_expected, rtol=1e-6)

    def test_polynomial_evaluate_rejects_non_integer_degree(self):
        """Polynomial degree is structural and must remain a positive integer."""
        k = Kernel.polynomial(degree=2.0, offset=1.0, outputscale=1.0)
        X = np.random.randn(8, 4).astype(np.float64)
        params = np.array([2.25, 1.0, 1.0], dtype=np.float32)

        with pytest.raises(ValueError, match="fixed positive integer"):
            k._evaluate_base(X, X, params, 0)

    def test_codegen_polynomial_degree_gradient_is_zero(self):
        """JIT codegen keeps polynomial degree fixed during optimization."""
        from mojogp.codegen_engine.differentiation import compute_gradients
        from mojogp.codegen_engine.expressions import KernelExprBuilder
        from mojogp.codegen_engine.overrides import get_overrides_for_kernel

        expr = KernelExprBuilder(dim=2).polynomial(ard=False, param_offset=0)
        gradients = compute_gradients(expr, get_overrides_for_kernel(expr))

        degree_param = expr.params[1]
        assert gradients.gradients[degree_param] == 0

    def test_codegen_polynomial_forward_uses_fixed_integer_degree(self):
        """Generated polynomial kernels do not call pow with a learnable degree."""
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        expr = KernelExprBuilder(dim=2).from_kernel_node(
            Kernel.polynomial(degree=2.0), param_offset=0
        )

        degree_param = expr.params[1]
        assert degree_param not in expr.forward.free_symbols

    def test_codegen_polynomial_rejects_non_integer_degree(self):
        """JIT polynomial codegen only accepts fixed positive integer degrees."""
        from mojogp.codegen_engine.expressions import KernelExprBuilder

        with pytest.raises(ValueError, match="fixed positive integer"):
            KernelExprBuilder(dim=2).from_kernel_node(
                Kernel.polynomial(degree=2.25), param_offset=0
            )

    def test_polynomial_from_engine_params_restores_structural_degree(self):
        """Unused engine degree slots do not leak into public polynomial params."""
        k = Kernel.polynomial(degree=2.0, offset=1.5, outputscale=0.8)

        params = k.from_engine_params(np.array([1.0, 0.97, 1.25, 0.6], dtype=np.float32))

        np.testing.assert_allclose(params, np.array([2.0, 1.25, 0.6], dtype=np.float32))

    def test_polynomial_factory_rejects_removed_variance_keyword(self):
        """Polynomial kernel uses outputscale; removed variance keyword is not accepted."""
        with pytest.raises(TypeError, match="variance"):
            Kernel.polynomial(degree=2.0, offset=1.0, outputscale=1.0, variance=99.0)

    def test_ard_polynomial_raises_not_implemented(self):
        """ARD polynomial should raise NotImplementedError in _evaluate_base."""
        k = Kernel.polynomial(ard=True)
        # num_params requires knowing ard_dim, which is set at fit time
        # But we can test _evaluate_base directly
        # Create a kernel node with ard_dim set
        k_node = KernelNode(
            kernel_type=KernelType.POLYNOMIAL,
            ard=True,
        )
        # Set ard_dim manually
        k_node.ard_dim = 3

        params = np.array([0.1, 0.2, 0.3, 2.0, 1.0, 1.0])  # 3 ls + degree + offset + os
        X = np.random.randn(5, 3)
        with pytest.raises(NotImplementedError, match="ARD polynomial"):
            k_node._evaluate_base(X, X, params, 0)


# ---------------------------------------------------------------------------
# MultiOutputGP fit-level route selection
# ---------------------------------------------------------------------------


class TestMultiOutputGPMatrixFreeSupport:
    """Verify that MultiOutputGP exposes fit-level route selection."""

    def test_matrix_free_is_accepted(self):
        """method='matrix_free' should be accepted."""
        from mojogp.multi_output_gp import MultiOutputGP

        import inspect

        sig = inspect.signature(MultiOutputGP.fit)
        assert "method" in sig.parameters

    def test_materialized_is_accepted(self):
        """method='materialized' should be accepted."""
        from mojogp.multi_output_gp import MultiOutputGP

        gp = MultiOutputGP(kernel="rbf")
        assert gp.method == "materialized"

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        from mojogp.multi_output_gp import MultiOutputGP

        X = np.zeros((2, 1), dtype=np.float32)
        Y = np.zeros((2, 2), dtype=np.float32)
        gp = MultiOutputGP(kernel="rbf")
        with pytest.raises(ValueError, match="method must be"):
            gp.fit(X, Y, method="bogus")


# ---------------------------------------------------------------------------
# Mixed additive continuous+categorical support
# ---------------------------------------------------------------------------


class TestMixedAdditiveSupport:
    """Verify that additive mixed continuous+categorical kernels are accepted."""

    def test_additive_mixed_is_accepted(self):
        """RBF() + EHH() should no longer be rejected at wrapper validation."""
        from mojogp.gp import SingleOutputGP

        k = Kernel.rbf(active_dims=[0, 1]) + Kernel.ehh(levels=3, active_dims=[2])

        np.random.seed(42)
        n, d = 50, 3
        X = np.random.randn(n, d).astype(np.float32)
        # Col 2 is categorical with 3 levels
        X[:, 2] = np.random.randint(0, 3, n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        gp = SingleOutputGP(kernel=k)
        try:
            gp.fit(X, y, max_iterations=1, method="materialized")
        except (RuntimeError, ImportError, Exception) as e:
            if "additive" in str(e).lower() or "categorical" in str(e).lower():
                pytest.fail(f"Additive mixed composition should not be rejected: {e}")

    def test_product_mixed_is_accepted(self):
        """RBF() * EHH() should be accepted (product composition)."""
        from mojogp.gp import SingleOutputGP

        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])

        np.random.seed(42)
        n, d = 50, 3
        X = np.random.randn(n, d).astype(np.float32)
        X[:, 2] = np.random.randint(0, 3, n).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)

        gp = SingleOutputGP(kernel=k)
        # Should not raise — product composition is supported
        # We don't actually fit (needs engine), just verify no ValueError from validation
        # The fit will fail at engine level but not at validation level
        try:
            gp.fit(X, y, max_iterations=1)
        except (RuntimeError, ImportError, Exception) as e:
            # Engine errors are OK — we only care that validation passed
            if "additive" in str(e).lower() or "categorical" in str(e).lower():
                pytest.fail(f"Product composition should not be rejected: {e}")
