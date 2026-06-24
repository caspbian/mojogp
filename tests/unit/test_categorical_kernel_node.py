"""Unit tests for categorical kernel KernelNode support.

Tests the Python-level categorical kernel types: KernelType enum entries,
factory methods, num_params, get_param_names, serialization, and composition.
"""

import pytest
import numpy as np
from mojogp.kernel import (
    Kernel,
    KernelNode,
    KernelType,
    make_ard_kernel,
    _CATEGORICAL_TYPES,
    _CONTINUOUS_TYPES,
    _CAT_PARAM_COUNTS,
    _cat_default_raw_param,
)


class TestKernelTypeEnum:
    """Test that categorical types exist in the enum."""

    def test_categorical_types_exist(self):
        assert KernelType.GD.value == "GDKernel"
        assert KernelType.CR.value == "CRKernel"
        assert KernelType.EHH.value == "EHHKernel"
        assert KernelType.HH.value == "HHKernel"
        assert KernelType.FE.value == "FEKernel"

    def test_type_sets_disjoint(self):
        """Continuous and categorical sets must not overlap."""
        assert len(_CONTINUOUS_TYPES & _CATEGORICAL_TYPES) == 0

    def test_all_types_classified(self):
        """Every KernelType must be in exactly one set."""
        all_types = set(KernelType)
        classified = _CONTINUOUS_TYPES | _CATEGORICAL_TYPES
        assert all_types == classified


class TestFactoryMethods:
    """Test the Kernel.gd(), .cr(), .ehh(), .hh(), .fe() factory methods."""

    def test_gd_factory_sets_type_levels_and_active_dim(self):
        k = Kernel.gd(levels=5, active_dims=[2])
        assert k.kernel_type == KernelType.GD
        assert k.levels == 5
        assert k.active_dims == (2,)

    def test_cr_factory_sets_type_and_levels(self):
        k = Kernel.cr(levels=3, active_dims=[0])
        assert k.kernel_type == KernelType.CR
        assert k.levels == 3

    def test_ehh_factory_sets_type_and_levels(self):
        k = Kernel.ehh(levels=4, active_dims=[1])
        assert k.kernel_type == KernelType.EHH
        assert k.levels == 4

    def test_hh_factory_sets_type_and_levels(self):
        k = Kernel.hh(levels=6, active_dims=[3])
        assert k.kernel_type == KernelType.HH
        assert k.levels == 6

    def test_fe_factory_sets_type_and_levels(self):
        k = Kernel.fe(levels=3, active_dims=[0])
        assert k.kernel_type == KernelType.FE
        assert k.levels == 3

    def test_levels_too_small(self):
        with pytest.raises(ValueError, match="levels >= 2"):
            Kernel.ehh(levels=1)

    def test_levels_one(self):
        with pytest.raises(ValueError, match="levels >= 2"):
            Kernel.gd(levels=1, active_dims=[0])

    def test_multiple_active_dims_error(self):
        with pytest.raises(ValueError, match="exactly 1 active_dim"):
            Kernel.ehh(levels=5, active_dims=[0, 1])

    def test_missing_active_dims_are_allowed_for_later_analysis(self):
        """Categorical without active_dims is allowed (set later during analysis)."""
        k = Kernel.gd(levels=3)
        assert k.active_dims is None

    def test_active_dims_sorted(self):
        k = Kernel.cr(levels=3, active_dims=[5])
        assert k.active_dims == (5,)


class TestConvenienceAliases:
    """Test that top-level aliases work."""

    def test_aliases_importable(self):
        from mojogp import GD, CR, EHH, HH, FE

        k = EHH(levels=5, active_dims=[0])
        assert k.kernel_type == KernelType.EHH


class TestIsCategorical:
    """Test is_categorical(), is_continuous(), has_categorical()."""

    def test_leaf_categorical(self):
        k = Kernel.ehh(levels=5, active_dims=[0])
        assert k.is_categorical()
        assert not k.is_continuous()

    def test_leaf_continuous(self):
        k = Kernel.rbf()
        assert k.is_continuous()
        assert not k.is_categorical()

    def test_composite_has_categorical(self):
        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=5, active_dims=[2])
        assert k.has_categorical()

    def test_pure_continuous_no_categorical(self):
        k = Kernel.rbf() + Kernel.matern52()
        assert not k.has_categorical()

    def test_operator_node_not_categorical(self):
        k = Kernel.rbf() + Kernel.matern52()
        assert not k.is_categorical()
        assert not k.is_continuous()  # operator node, not a leaf


class TestNumParams:
    """Test num_params() for categorical kernels."""

    def test_gd_1_param(self):
        k = Kernel.gd(levels=5, active_dims=[0])
        assert k.num_params() == 1

    def test_gd_1_param_any_levels(self):
        for L in [2, 5, 10, 20]:
            k = Kernel.gd(levels=L, active_dims=[0])
            assert k.num_params() == 1

    def test_cr_L_params(self):
        for L in [2, 3, 5, 10]:
            k = Kernel.cr(levels=L, active_dims=[0])
            assert k.num_params() == L

    def test_ehh_params(self):
        for L in [2, 3, 4, 5]:
            k = Kernel.ehh(levels=L, active_dims=[0])
            expected = L * (L - 1) // 2
            assert k.num_params() == expected, (
                f"L={L}: expected {expected}, got {k.num_params()}"
            )

    def test_hh_params(self):
        k = Kernel.hh(levels=4, active_dims=[0])
        assert k.num_params() == 4 * 3 // 2  # 6

    def test_fe_params(self):
        for L in [2, 3, 4, 5]:
            k = Kernel.fe(levels=L, active_dims=[0])
            expected = L * (L + 1) // 2
            assert k.num_params() == expected

    def test_composite_params(self):
        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=5, active_dims=[2])
        # RBF: 2 params, EHH(L=5): 5*4/2 = 10 params
        assert k.num_params() == 2 + 10


class TestGetParamNames:
    """Test get_param_names() for categorical kernels."""

    def test_gd_single_param(self):
        k = Kernel.gd(levels=5, active_dims=[3])
        names = k.get_param_names()
        assert names == ["gd_col3_theta"]

    def test_cr_multi_params(self):
        k = Kernel.cr(levels=3, active_dims=[2])
        names = k.get_param_names()
        assert names == ["cr_col2_theta_0", "cr_col2_theta_1", "cr_col2_theta_2"]

    def test_ehh_params(self):
        k = Kernel.ehh(levels=3, active_dims=[0])
        names = k.get_param_names()
        assert len(names) == 3  # 3*(3-1)/2 = 3
        assert names[0] == "ehh_col0_theta_0"

    def test_no_active_dims_no_col_suffix(self):
        k = Kernel.gd(levels=3)
        names = k.get_param_names()
        assert names == ["gd_theta"]

    def test_composite_names(self):
        k = Kernel.rbf(active_dims=[0]) * Kernel.gd(levels=3, active_dims=[1])
        names = k.get_param_names()
        # Left: rbf params, Right: gd params
        assert "left_rbf_lengthscale" in names
        assert "right_gd_col1_theta" in names


class TestInitialParams:
    """Test get_initial_params() for categorical kernels."""

    def test_gd_default_init(self):
        k = Kernel.gd(levels=5, active_dims=[0])
        params = k.get_initial_params()
        assert len(params) == 1
        # Should be inv_softplus(0.5)
        expected = float(np.log(np.exp(0.5) - 1.0))
        assert abs(params[0] - expected) < 1e-5

    def test_ehh_default_init(self):
        k = Kernel.ehh(levels=3, active_dims=[0])
        params = k.get_initial_params()
        assert len(params) == 3
        # All should be inv_sigmoid(0.25)
        expected = float(np.log(0.25 / 0.75))
        for p in params:
            assert abs(p - expected) < 1e-5

    def test_fe_mixed_init(self):
        L = 3
        k = Kernel.fe(levels=L, active_dims=[0])
        params = k.get_initial_params()
        n_angles = L * (L - 1) // 2  # 3
        n_diag = L  # 3
        assert len(params) == n_angles + n_diag
        # First n_angles should be inv_sigmoid(0.25)
        inv_sig = float(np.log(0.25 / 0.75))
        for i in range(n_angles):
            assert abs(params[i] - inv_sig) < 1e-5
        # Last n_diag should be inv_softplus(0.3)
        inv_sp = float(np.log(np.exp(0.3) - 1.0))
        for i in range(n_angles, n_angles + n_diag):
            assert abs(params[i] - inv_sp) < 1e-5


class TestSerialization:
    """Test to_dict() / from_dict() round-trip for categorical kernels."""

    def test_leaf_categorical_roundtrip(self):
        k = Kernel.ehh(levels=5, active_dims=[2])
        d = k.to_dict()
        assert d["kernel_type"] == "EHH"
        assert d["levels"] == 5
        assert d["active_dims"] == [2]

        k2 = KernelNode.from_dict(d)
        assert k2.kernel_type == KernelType.EHH
        assert k2.levels == 5
        assert k2.active_dims == (2,)

    def test_composite_roundtrip(self):
        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.gd(levels=3, active_dims=[2])
        d = k.to_dict()
        k2 = KernelNode.from_dict(d)
        assert k2.operator == "product"
        assert k2.right.kernel_type == KernelType.GD
        assert k2.right.levels == 3

    def test_no_levels_roundtrip(self):
        """Continuous kernels don't have levels in serialized form."""
        k = Kernel.rbf()
        d = k.to_dict()
        assert "levels" not in d
        k2 = KernelNode.from_dict(d)
        assert k2.levels is None


class TestRepr:
    """Test __repr__() for categorical kernels."""

    def test_ehh_repr(self):
        k = Kernel.ehh(levels=5, active_dims=[2])
        r = repr(k)
        assert "EHH" in r
        assert "levels=5" in r
        assert "active_dims=[2]" in r

    def test_gd_repr(self):
        k = Kernel.gd(levels=3)
        r = repr(k)
        assert "GD" in r
        assert "levels=3" in r

    def test_composite_repr(self):
        k = Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=5, active_dims=[1])
        r = repr(k)
        assert "RBF" in r
        assert "EHH" in r


class TestMakeArdKernel:
    """Test that make_ard_kernel skips categorical kernels."""

    def test_categorical_passthrough(self):
        k = Kernel.ehh(levels=5, active_dims=[2])
        k_ard = make_ard_kernel(k, dim=3)
        # Should NOT have ard_dim set (categorical kernels don't use ARD)
        assert k_ard.ard_dim is None
        assert k_ard.levels == 5
        assert k_ard.active_dims == (2,)

    def test_mixed_composite_ard(self):
        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=5, active_dims=[2])
        k_ard = make_ard_kernel(k, dim=3)
        # Left (RBF) should get ard_dim=2 (len of active_dims)
        assert k_ard.left.ard_dim == 2
        # Right (EHH) should NOT get ard_dim
        assert k_ard.right.ard_dim is None
        assert k_ard.right.levels == 5


class TestComposition:
    """Test composing categorical with continuous kernels."""

    def test_product_composition(self):
        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=5, active_dims=[2])
        assert k.operator == "product"
        assert k.left.is_continuous()
        assert k.right.is_categorical()

    def test_sum_composition(self):
        k = Kernel.rbf(active_dims=[0, 1]) + Kernel.ehh(levels=5, active_dims=[2])
        assert k.operator == "sum"

    def test_multi_categorical(self):
        k = (
            Kernel.rbf(active_dims=[0, 1])
            * Kernel.ehh(levels=5, active_dims=[2])
            * Kernel.gd(levels=3, active_dims=[3])
        )
        assert k.has_categorical()
        # Total params: RBF(2) + EHH(10) + GD(1) = 13
        assert k.num_params() == 2 + 10 + 1

    def test_pure_categorical(self):
        k = Kernel.ehh(levels=5, active_dims=[0]) * Kernel.gd(levels=3, active_dims=[1])
        assert k.has_categorical()
        assert k.num_params() == 10 + 1
