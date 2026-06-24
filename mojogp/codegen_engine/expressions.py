"""SymPy expression builders for all GP kernel types.

Builds symbolic kernel expressions that can be differentiated, optimized,
and emitted as Mojo GPU kernel code. Each kernel type (iso + ARD) has a
builder method that returns a KernelExpr dataclass.

Composition operators (Sum, Product, Scale, DimSlice) combine KernelExprs
using standard algebraic rules, preserving shared intermediate tracking.
"""

import sympy as sp
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KernelExpr:
    """Result of building a kernel expression."""

    forward: sp.Expr  # k(x_i, x_j)
    params: list  # ordered parameter symbols (sp.Symbol)
    shared: dict  # named intermediates (dist_sq, dot_prod, etc.)
    needs_diffs: bool  # requires per-dim differences
    needs_dist_sq: bool  # requires precomputed squared distance
    needs_dot: bool  # requires dot product
    dim_symbols: list  # diff_0, diff_1, ... for DIM dimensions
    param_layout: list  # human-readable param names ["ls", "os"]
    kernel_type: Optional[str] = None  # e.g. "rbf", "matern52"
    is_ard: bool = False
    categorical_vars: list = field(default_factory=list)  # ["cat_0", "cat_1", ...]


class KernelExprBuilder:
    """Builds SymPy expressions for GP kernel types.

    Usage:
        builder = KernelExprBuilder(dim=5)
        expr = builder.rbf(ard=False)
        # expr.forward is a SymPy expression for k(x_i, x_j)
        # expr.params is [ls, os]
        # expr.shared may contain {"dist_sq": ...}
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.diffs = [sp.Symbol(f"diff_{d}") for d in range(dim)]
        self.x_row = [sp.Symbol(f"x_row_{d}") for d in range(dim)]
        self.x_col = [sp.Symbol(f"x_col_{d}") for d in range(dim)]
        # Precomputed shared intermediates (for isotropic kernels)
        self._dist_sq = sp.Symbol("dist_sq", nonnegative=True)
        self._dot_prod = sp.Symbol("dot_prod")

    def _make_params(self, names: list[str], offset: int) -> list[sp.Symbol]:
        """Create parameter symbols with positional indices."""
        return [sp.Symbol(f"p_{offset + i}", positive=True) for i in range(len(names))]

    def _freeze_polynomial_degree(self, node, result: KernelExpr) -> KernelExpr:
        """Replace polynomial degree parameter usage with a fixed integer literal."""
        if node.kernel_type is None or node.kernel_type.name != "POLYNOMIAL":
            return result
        degree_value = float(getattr(node, "initial_values", {}).get("degree", 2.0))
        degree_int = int(round(degree_value))
        if abs(degree_value - float(degree_int)) > 1e-5 or degree_int < 1:
            raise ValueError(
                "Polynomial kernel degree must be a fixed positive integer for JIT codegen, "
                f"got {degree_value:.6g}."
            )
        degree_param_index = result.param_layout.index("degree")
        degree_symbol = result.params[degree_param_index]
        result.forward = result.forward.subs(degree_symbol, sp.Integer(degree_int))
        return result

    # =========================================================================
    # Base kernels
    # =========================================================================

    def rbf(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """RBF (Squared Exponential) kernel.

        Iso: k = os * exp(-0.5 * dist_sq / ls^2), params = [ls, os]
        ARD: k = os * exp(-0.5 * sum_d (diff_d / l_d)^2), params = [l_0..l_{d-1}, os]
        """
        if ard:
            ls = self._make_params([f"l_{d}" for d in range(self.dim)], param_offset)
            os_sym = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            ard_dist_sq = sum((self.diffs[d] / ls[d]) ** 2 for d in range(self.dim))
            forward = os_sym * sp.exp(sp.Rational(-1, 2) * ard_dist_sq)
            return KernelExpr(
                forward=forward,
                params=ls + [os_sym],
                shared={},
                needs_diffs=True,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=[f"l_{d}" for d in range(self.dim)] + ["os"],
                kernel_type="rbf",
                is_ard=True,
            )
        else:
            ls, os_sym = self._make_params(["ls", "os"], param_offset)
            forward = os_sym * sp.exp(sp.Rational(-1, 2) * self._dist_sq / ls**2)
            return KernelExpr(
                forward=forward,
                params=[ls, os_sym],
                shared={"dist_sq": self._dist_sq},
                needs_diffs=True,
                needs_dist_sq=True,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=["ls", "os"],
                kernel_type="rbf",
                is_ard=False,
            )

    def matern12(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """Matern 1/2 (exponential) kernel.

        k = os * exp(-r), where r = ||x-x'|| / ls (iso) or sqrt(sum (diff_d/l_d)^2) (ARD)
        """
        if ard:
            ls = self._make_params([f"l_{d}" for d in range(self.dim)], param_offset)
            os_sym = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            ard_dist_sq = sum((self.diffs[d] / ls[d]) ** 2 for d in range(self.dim))
            r = sp.sqrt(ard_dist_sq + sp.Float(1e-20))  # numerical safety
            forward = os_sym * sp.exp(-r)
            return KernelExpr(
                forward=forward,
                params=ls + [os_sym],
                shared={},
                needs_diffs=True,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=[f"l_{d}" for d in range(self.dim)] + ["os"],
                kernel_type="matern12",
                is_ard=True,
            )
        else:
            ls, os_sym = self._make_params(["ls", "os"], param_offset)
            r = sp.sqrt(self._dist_sq + sp.Float(1e-20)) / ls
            forward = os_sym * sp.exp(-r)
            return KernelExpr(
                forward=forward,
                params=[ls, os_sym],
                shared={"dist_sq": self._dist_sq},
                needs_diffs=True,
                needs_dist_sq=True,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=["ls", "os"],
                kernel_type="matern12",
                is_ard=False,
            )

    def matern32(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """Matern 3/2 kernel.

        k = os * (1 + sqrt(3)*r) * exp(-sqrt(3)*r)
        """
        sqrt3 = sp.sqrt(3)
        if ard:
            ls = self._make_params([f"l_{d}" for d in range(self.dim)], param_offset)
            os_sym = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            ard_dist_sq = sum((self.diffs[d] / ls[d]) ** 2 for d in range(self.dim))
            r = sp.sqrt(ard_dist_sq + sp.Float(1e-20))
            forward = os_sym * (1 + sqrt3 * r) * sp.exp(-sqrt3 * r)
            return KernelExpr(
                forward=forward,
                params=ls + [os_sym],
                shared={},
                needs_diffs=True,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=[f"l_{d}" for d in range(self.dim)] + ["os"],
                kernel_type="matern32",
                is_ard=True,
            )
        else:
            ls, os_sym = self._make_params(["ls", "os"], param_offset)
            r = sp.sqrt(self._dist_sq + sp.Float(1e-20)) / ls
            forward = os_sym * (1 + sqrt3 * r) * sp.exp(-sqrt3 * r)
            return KernelExpr(
                forward=forward,
                params=[ls, os_sym],
                shared={"dist_sq": self._dist_sq},
                needs_diffs=True,
                needs_dist_sq=True,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=["ls", "os"],
                kernel_type="matern32",
                is_ard=False,
            )

    def matern52(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """Matern 5/2 kernel.

        k = os * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
        """
        sqrt5 = sp.sqrt(5)
        if ard:
            ls = self._make_params([f"l_{d}" for d in range(self.dim)], param_offset)
            os_sym = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            ard_dist_sq = sum((self.diffs[d] / ls[d]) ** 2 for d in range(self.dim))
            r = sp.sqrt(ard_dist_sq + sp.Float(1e-20))
            forward = (
                os_sym * (1 + sqrt5 * r + sp.Rational(5, 3) * r**2) * sp.exp(-sqrt5 * r)
            )
            return KernelExpr(
                forward=forward,
                params=ls + [os_sym],
                shared={},
                needs_diffs=True,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=[f"l_{d}" for d in range(self.dim)] + ["os"],
                kernel_type="matern52",
                is_ard=True,
            )
        else:
            ls, os_sym = self._make_params(["ls", "os"], param_offset)
            r = sp.sqrt(self._dist_sq + sp.Float(1e-20)) / ls
            forward = (
                os_sym * (1 + sqrt5 * r + sp.Rational(5, 3) * r**2) * sp.exp(-sqrt5 * r)
            )
            return KernelExpr(
                forward=forward,
                params=[ls, os_sym],
                shared={"dist_sq": self._dist_sq},
                needs_diffs=True,
                needs_dist_sq=True,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=["ls", "os"],
                kernel_type="matern52",
                is_ard=False,
            )

    def periodic(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """Periodic kernel.

        Iso: k = os * exp(-2 * sum sin^2(pi*diff/period) / ls), params = [ls, period, os]
        ARD: k = os * exp(-2 * sum sin^2(pi*diff_d/period) / l_d), params = [l_0..l_{d-1}, period, os]
        """
        pi = sp.pi
        if ard:
            ls = self._make_params([f"l_{d}" for d in range(self.dim)], param_offset)
            period = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            os_sym = sp.Symbol(f"p_{param_offset + self.dim + 1}", positive=True)
            ssq = sum(
                sp.sin(pi * self.diffs[d] / period) ** 2 / ls[d]
                for d in range(self.dim)
            )
            forward = os_sym * sp.exp(-2 * ssq)
            return KernelExpr(
                forward=forward,
                params=ls + [period, os_sym],
                shared={},
                needs_diffs=True,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=[f"l_{d}" for d in range(self.dim)] + ["period", "os"],
                kernel_type="periodic",
                is_ard=True,
            )
        else:
            ls, period, os_sym = self._make_params(["ls", "period", "os"], param_offset)
            ssq = sum(sp.sin(pi * self.diffs[d] / period) ** 2 for d in range(self.dim))
            forward = os_sym * sp.exp(-2 * ssq / ls)
            return KernelExpr(
                forward=forward,
                params=[ls, period, os_sym],
                shared={},
                needs_diffs=True,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=["ls", "period", "os"],
                kernel_type="periodic",
                is_ard=False,
            )

    def rq(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """Rational Quadratic kernel.

        Iso: k = os * (1 + dist_sq/(2*alpha*ls^2))^(-alpha), params = [ls, alpha, os]
        ARD: k = os * (1 + ard_dist_sq/(2*alpha))^(-alpha), params = [l_0..l_{d-1}, alpha, os]
        """
        if ard:
            ls = self._make_params([f"l_{d}" for d in range(self.dim)], param_offset)
            alpha = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            os_sym = sp.Symbol(f"p_{param_offset + self.dim + 1}", positive=True)
            ard_dist_sq = sum((self.diffs[d] / ls[d]) ** 2 for d in range(self.dim))
            forward = os_sym * (1 + ard_dist_sq / (2 * alpha)) ** (-alpha)
            return KernelExpr(
                forward=forward,
                params=ls + [alpha, os_sym],
                shared={},
                needs_diffs=True,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=[f"l_{d}" for d in range(self.dim)] + ["alpha", "os"],
                kernel_type="rq",
                is_ard=True,
            )
        else:
            ls, alpha, os_sym = self._make_params(["ls", "alpha", "os"], param_offset)
            forward = os_sym * (1 + self._dist_sq / (2 * alpha * ls**2)) ** (-alpha)
            return KernelExpr(
                forward=forward,
                params=[ls, alpha, os_sym],
                shared={"dist_sq": self._dist_sq},
                needs_diffs=True,
                needs_dist_sq=True,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=["ls", "alpha", "os"],
                kernel_type="rq",
                is_ard=False,
            )

    def linear(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """Linear kernel.

        Iso: k = os * variance * dot_prod, params = [variance, os]
        ARD: k = os * sum(var_d * x_d * x'_d), params = [var_0..var_{d-1}, os]
        """
        if ard:
            variances = self._make_params(
                [f"var_{d}" for d in range(self.dim)], param_offset
            )
            os_sym = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            weighted_dot = sum(
                variances[d] * self.x_row[d] * self.x_col[d] for d in range(self.dim)
            )
            forward = os_sym * weighted_dot
            return KernelExpr(
                forward=forward,
                params=variances + [os_sym],
                shared={},
                needs_diffs=False,
                needs_dist_sq=False,
                needs_dot=False,  # ARD computes its own weighted dot
                dim_symbols=self.diffs,
                param_layout=[f"var_{d}" for d in range(self.dim)] + ["os"],
                kernel_type="linear",
                is_ard=True,
            )
        else:
            variance, os_sym = self._make_params(["variance", "os"], param_offset)
            forward = os_sym * variance * self._dot_prod
            return KernelExpr(
                forward=forward,
                params=[variance, os_sym],
                shared={"dot_prod": self._dot_prod},
                needs_diffs=False,
                needs_dist_sq=False,
                needs_dot=True,
                dim_symbols=self.diffs,
                param_layout=["variance", "os"],
                kernel_type="linear",
                is_ard=False,
            )

    def polynomial(self, ard: bool = False, param_offset: int = 0) -> KernelExpr:
        """Polynomial kernel.

        Iso: k = os * (variance * dot_prod + offset)^degree, params = [variance, degree, offset, os]
        ARD: k = os * (sum(var_d * x_d * x'_d) + offset)^degree, params = [var_0..var_{d-1}, degree, offset, os]
        """
        if ard:
            variances = self._make_params(
                [f"var_{d}" for d in range(self.dim)], param_offset
            )
            degree = sp.Symbol(f"p_{param_offset + self.dim}", positive=True)
            offset = sp.Symbol(f"p_{param_offset + self.dim + 1}", positive=True)
            os_sym = sp.Symbol(f"p_{param_offset + self.dim + 2}", positive=True)
            weighted_dot = sum(
                variances[d] * self.x_row[d] * self.x_col[d] for d in range(self.dim)
            )
            base = weighted_dot + offset
            forward = os_sym * base**degree
            return KernelExpr(
                forward=forward,
                params=variances + [degree, offset, os_sym],
                shared={},
                needs_diffs=False,
                needs_dist_sq=False,
                needs_dot=False,
                dim_symbols=self.diffs,
                param_layout=[f"var_{d}" for d in range(self.dim)]
                + ["degree", "offset", "os"],
                kernel_type="polynomial",
                is_ard=True,
            )
        else:
            variance, degree, offset, os_sym = self._make_params(
                ["variance", "degree", "offset", "os"], param_offset
            )
            base = variance * self._dot_prod + offset
            forward = os_sym * base**degree
            return KernelExpr(
                forward=forward,
                params=[variance, degree, offset, os_sym],
                shared={"dot_prod": self._dot_prod},
                needs_diffs=False,
                needs_dist_sq=False,
                needs_dot=True,
                dim_symbols=self.diffs,
                param_layout=["variance", "degree", "offset", "os"],
                kernel_type="polynomial",
                is_ard=False,
            )

    # =========================================================================
    # Composition operators
    # =========================================================================

    def sum(self, k1: KernelExpr, k2: KernelExpr) -> KernelExpr:
        """Sum of two kernels: k = k1 + k2."""
        return KernelExpr(
            forward=k1.forward + k2.forward,
            params=k1.params + k2.params,
            shared={**k1.shared, **k2.shared},
            needs_diffs=k1.needs_diffs or k2.needs_diffs,
            needs_dist_sq=k1.needs_dist_sq or k2.needs_dist_sq,
            needs_dot=k1.needs_dot or k2.needs_dot,
            dim_symbols=self.diffs,
            param_layout=k1.param_layout + k2.param_layout,
            kernel_type="sum",
            categorical_vars=k1.categorical_vars + k2.categorical_vars,
        )

    def product(self, k1: KernelExpr, k2: KernelExpr) -> KernelExpr:
        """Product of two kernels: k = k1 * k2."""
        return KernelExpr(
            forward=k1.forward * k2.forward,
            params=k1.params + k2.params,
            shared={**k1.shared, **k2.shared},
            needs_diffs=k1.needs_diffs or k2.needs_diffs,
            needs_dist_sq=k1.needs_dist_sq or k2.needs_dist_sq,
            needs_dot=k1.needs_dot or k2.needs_dot,
            dim_symbols=self.diffs,
            param_layout=k1.param_layout + k2.param_layout,
            kernel_type="product",
            categorical_vars=k1.categorical_vars + k2.categorical_vars,
        )

    def scale(self, k: KernelExpr, scale_param_offset: int) -> KernelExpr:
        """Scaled kernel: k_new = scale * k_base. params = [scale | k.params]."""
        scale_sym = sp.Symbol(f"p_{scale_param_offset}", positive=True)
        return KernelExpr(
            forward=scale_sym * k.forward,
            params=[scale_sym] + k.params,
            shared=k.shared,
            needs_diffs=k.needs_diffs,
            needs_dist_sq=k.needs_dist_sq,
            needs_dot=k.needs_dot,
            dim_symbols=self.diffs,
            param_layout=["scale"] + k.param_layout,
            kernel_type="scale",
            categorical_vars=k.categorical_vars,
        )

    # =========================================================================
    # KernelNode conversion
    # =========================================================================

    def from_kernel_node(self, node, param_offset: int = 0) -> KernelExpr:
        expr, _ = self._from_kernel_node(
            node, param_offset=param_offset, categorical_offset=0
        )
        return expr

    def _from_kernel_node(
        self, node, param_offset: int = 0, categorical_offset: int = 0
    ) -> tuple[KernelExpr, int]:
        """Build KernelExpr from a KernelNode tree.

        Args:
            node: A KernelNode from mojogp.kernel
            param_offset: Starting parameter index
            categorical_offset: Starting categorical leaf index
        """
        # Lazy import to avoid circular dependency
        from mojogp.kernel import KernelType

        if node.kernel_type is not None:
            if node.kernel_type in {
                KernelType.GD,
                KernelType.CR,
                KernelType.EHH,
                KernelType.HH,
                KernelType.FE,
            }:
                cat_name = f"cat_{categorical_offset}"
                return (
                    KernelExpr(
                        forward=sp.Symbol(cat_name),
                        params=[],
                        shared={},
                        needs_diffs=False,
                        needs_dist_sq=False,
                        needs_dot=False,
                        dim_symbols=self.diffs,
                        param_layout=[],
                        kernel_type=node.kernel_type.name.lower(),
                        categorical_vars=[cat_name],
                    ),
                    categorical_offset + 1,
                )

            # is_ard: true if user set ard=True OR ard_dim is already resolved
            is_ard = getattr(node, "ard", False) or node.ard_dim is not None
            dim_start = getattr(node, "_dim_start", None) or 0
            dim_end = getattr(node, "_dim_end", None) or self.dim
            has_dim_slice = dim_start != 0 or dim_end != self.dim

            builder_map = {
                KernelType.RBF: self.rbf,
                KernelType.MATERN12: self.matern12,
                KernelType.MATERN32: self.matern32,
                KernelType.MATERN52: self.matern52,
                KernelType.PERIODIC: self.periodic,
                KernelType.RQ: self.rq,
                KernelType.LINEAR: self.linear,
                KernelType.POLYNOMIAL: self.polynomial,
            }
            builder_fn = builder_map.get(node.kernel_type)
            if builder_fn is None:
                raise ValueError(f"Unknown kernel type: {node.kernel_type}")

            if not has_dim_slice:
                result = builder_fn(
                    ard=is_ard, param_offset=param_offset
                )
                return self._freeze_polynomial_degree(node, result), categorical_offset

            # Per-sub-kernel active_dims: kernel only operates on dims [dim_start, dim_end)
            if not is_ard:
                # Isotropic kernel with dim slice: compute dist_sq over only the
                # active dims instead of using the global precomputed dist_sq.
                # We temporarily override self.dim, self.diffs, AND self._dist_sq
                # so the isotropic builder produces an expression using a local
                # dist_sq computed from only the sliced dimensions.
                orig_dim = self.dim
                orig_diffs = self.diffs
                orig_x_row = self.x_row
                orig_x_col = self.x_col
                orig_dist_sq = self._dist_sq
                orig_dot_prod = self._dot_prod
                slice_dim = dim_end - dim_start
                slice_diffs = orig_diffs[dim_start:dim_end]
                slice_x_row = orig_x_row[dim_start:dim_end]
                slice_x_col = orig_x_col[dim_start:dim_end]
                self.dim = slice_dim
                self.diffs = slice_diffs
                self.x_row = slice_x_row
                self.x_col = slice_x_col
                # Replace the global dist_sq symbol with an explicit sum over sliced dims
                self._dist_sq = sum(d**2 for d in slice_diffs)
                self._dot_prod = sum(
                    slice_x_row[d] * slice_x_col[d] for d in range(slice_dim)
                )
                try:
                    result = builder_fn(ard=False, param_offset=param_offset)
                    result = self._freeze_polynomial_degree(node, result)
                    # The expression now uses the sliced dist_sq inline.
                    # Mark it as NOT needing the precomputed dist_sq (it's inline),
                    # but still needing diffs for the per-dim differences.
                    result.needs_dist_sq = False
                    result.needs_diffs = True
                    # Remove dist_sq from shared since it's now inline
                    result.shared.pop("dist_sq", None)
                    # Linear/polynomial slices inline their local dot product.
                    result.needs_dot = False
                    result.shared.pop("dot_prod", None)
                    # Restore dim_symbols to full set so the GPU kernel
                    # generates diff computation for all dims
                    result.dim_symbols = orig_diffs
                finally:
                    self.dim = orig_dim
                    self.diffs = orig_diffs
                    self.x_row = orig_x_row
                    self.x_col = orig_x_col
                    self._dist_sq = orig_dist_sq
                    self._dot_prod = orig_dot_prod
                return result, categorical_offset

            # ARD with dim slice: temporarily override self.dim and self.diffs so that
            # the builder creates the right number of lengthscale params and references
            # the correct diff symbols (diff_{dim_start} .. diff_{dim_end-1}).
            orig_dim = self.dim
            orig_diffs = self.diffs
            orig_x_row = self.x_row
            orig_x_col = self.x_col
            self.dim = dim_end - dim_start
            self.diffs = orig_diffs[dim_start:dim_end]
            self.x_row = orig_x_row[dim_start:dim_end]
            self.x_col = orig_x_col[dim_start:dim_end]
            try:
                result = builder_fn(ard=True, param_offset=param_offset)
                result = self._freeze_polynomial_degree(node, result)
            finally:
                self.dim = orig_dim
                self.diffs = orig_diffs
                self.x_row = orig_x_row
                self.x_col = orig_x_col
            return result, categorical_offset

        if node.operator == "sum":
            k1, next_categorical = self._from_kernel_node(
                node.left,
                param_offset,
                categorical_offset,
            )
            k2, final_categorical = self._from_kernel_node(
                node.right,
                param_offset + len(k1.params),
                next_categorical,
            )
            return self.sum(k1, k2), final_categorical

        if node.operator == "product":
            k1, next_categorical = self._from_kernel_node(
                node.left,
                param_offset,
                categorical_offset,
            )
            k2, final_categorical = self._from_kernel_node(
                node.right,
                param_offset + len(k1.params),
                next_categorical,
            )
            return self.product(k1, k2), final_categorical

        if node.operator == "scale":
            k_inner, next_categorical = self._from_kernel_node(
                node.left,
                param_offset + 1,
                categorical_offset,
            )
            return self.scale(k_inner, param_offset), next_categorical

        raise ValueError(f"Unknown operator: {node.operator}")
