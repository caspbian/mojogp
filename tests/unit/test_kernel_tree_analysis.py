"""Unit tests for kernel tree analysis.

Tests the analyze_kernel_tree() function that decomposes a kernel tree
into continuous + categorical components.
"""

import pytest
from mojogp.kernel import (
    Kernel,
    KernelNode,
    KernelType,
    analyze_kernel_tree,
    KernelTreeAnalysis,
    CategoricalSpec,
)


class TestPureContinuous:
    """Tests where no categorical kernels are present."""

    def test_single_rbf(self):
        k = Kernel.rbf()
        a = analyze_kernel_tree(k, total_dim=3)
        assert a.is_pure_continuous
        assert not a.is_pure_categorical
        assert a.continuous_kernel is not None
        assert len(a.categorical_specs) == 0
        assert a.continuous_dims == [0, 1, 2]
        assert a.categorical_dims == []

    def test_sum_of_continuous(self):
        k = Kernel.rbf() + Kernel.matern52()
        a = analyze_kernel_tree(k, total_dim=5)
        assert a.is_pure_continuous
        assert a.continuous_kernel is k  # Same tree returned
        assert len(a.categorical_specs) == 0

    def test_product_of_continuous(self):
        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.periodic(active_dims=[2])
        a = analyze_kernel_tree(k, total_dim=3)
        assert a.is_pure_continuous
        assert a.continuous_kernel is k

    def test_scaled_continuous(self):
        k = 2.0 * Kernel.rbf()
        a = analyze_kernel_tree(k, total_dim=3)
        assert a.is_pure_continuous


class TestProductComposition:
    """Tests for K_cont * K_cat product composition."""

    def test_rbf_times_ehh(self):
        k = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=5, active_dims=[2])
        a = analyze_kernel_tree(k, total_dim=3)
        assert not a.is_pure_continuous
        assert not a.is_pure_categorical
        assert a.compose_op == "product"

        # Categorical specs
        assert len(a.categorical_specs) == 1
        spec = a.categorical_specs[0]
        assert spec.kernel_type == KernelType.EHH
        assert spec.levels == 5
        assert spec.col_index == 2

        # Continuous kernel
        assert a.continuous_kernel is not None
        assert a.continuous_kernel.kernel_type == KernelType.RBF

        # Dimension lists
        assert a.categorical_dims == [2]
        assert a.continuous_dims == [0, 1]

    def test_multi_categorical(self):
        k = (
            Kernel.rbf(active_dims=[0, 1])
            * Kernel.ehh(levels=5, active_dims=[2])
            * Kernel.gd(levels=3, active_dims=[3])
        )
        a = analyze_kernel_tree(k, total_dim=4)
        assert not a.is_pure_continuous
        assert not a.is_pure_categorical
        assert a.compose_op == "product"

        assert len(a.categorical_specs) == 2
        # Specs should be in tree order (EHH before GD)
        assert a.categorical_specs[0].kernel_type == KernelType.EHH
        assert a.categorical_specs[0].col_index == 2
        assert a.categorical_specs[1].kernel_type == KernelType.GD
        assert a.categorical_specs[1].col_index == 3

        assert a.continuous_kernel.kernel_type == KernelType.RBF
        assert a.categorical_dims == [2, 3]
        assert a.continuous_dims == [0, 1]

    def test_different_cat_kernels_per_column(self):
        """Feature not possible with old cat_dims API."""
        k = (
            Kernel.rbf(active_dims=[0, 1])
            * Kernel.ehh(levels=5, active_dims=[2])
            * Kernel.cr(levels=3, active_dims=[3])
        )
        a = analyze_kernel_tree(k, total_dim=4)
        assert a.categorical_specs[0].kernel_type == KernelType.EHH
        assert a.categorical_specs[1].kernel_type == KernelType.CR


class TestSumComposition:
    """Tests for K_cont + K_cat sum composition."""

    def test_rbf_plus_ehh(self):
        k = Kernel.rbf(active_dims=[0, 1]) + Kernel.ehh(levels=5, active_dims=[2])
        a = analyze_kernel_tree(k, total_dim=3)
        assert not a.is_pure_continuous
        assert a.compose_op == "sum"
        assert len(a.categorical_specs) == 1
        assert a.continuous_kernel is not None

    def test_nested_mixed_tree_preserves_duplicate_categorical_leaves(self):
        k = (Kernel.rbf(active_dims=[0, 1]) + Kernel.matern32(active_dims=[0, 1])) * (
            Kernel.gd(levels=3, active_dims=[2]) + Kernel.cr(levels=3, active_dims=[2])
        )
        a = analyze_kernel_tree(k, total_dim=3)

        assert not a.is_pure_continuous
        assert not a.is_pure_categorical
        assert len(a.categorical_specs) == 2
        assert [spec.kernel_type for spec in a.categorical_specs] == [
            KernelType.GD,
            KernelType.CR,
        ]
        assert [spec.col_index for spec in a.categorical_specs] == [2, 2]
        assert a.categorical_dims == [2]
        assert a.continuous_dims == [0, 1]


class TestPureCategorical:
    """Tests for kernels with only categorical nodes."""

    def test_single_categorical(self):
        k = Kernel.ehh(levels=5, active_dims=[0])
        a = analyze_kernel_tree(k, total_dim=3)
        assert not a.is_pure_continuous
        assert a.is_pure_categorical
        assert a.continuous_kernel is None
        assert len(a.categorical_specs) == 1
        assert a.categorical_dims == [0]
        assert a.continuous_dims == [1, 2]

    def test_multi_categorical_product(self):
        k = Kernel.ehh(levels=5, active_dims=[0]) * Kernel.gd(levels=3, active_dims=[1])
        a = analyze_kernel_tree(k, total_dim=3)
        assert a.is_pure_categorical
        assert len(a.categorical_specs) == 2
        assert a.categorical_dims == [0, 1]
        assert a.continuous_dims == [2]


class TestComplexPatterns:
    """Tests for composite continuous inside mixed composition."""

    def test_composite_continuous_times_categorical(self):
        """(RBF + Matern52) * EHH — composite continuous treated as single unit."""
        cont = Kernel.rbf(active_dims=[0, 1]) + Kernel.matern52(active_dims=[0, 1])
        k = cont * Kernel.ehh(levels=5, active_dims=[2])
        a = analyze_kernel_tree(k, total_dim=3)
        assert not a.is_pure_continuous
        assert a.compose_op == "product"
        assert len(a.categorical_specs) == 1
        # Continuous kernel should be the composite (RBF + Matern52)
        assert a.continuous_kernel.operator == "sum"


class TestUnsupportedPatterns:
    """Tests that unsupported patterns give clear errors."""

    def test_categorical_in_scale(self):
        with pytest.raises(ValueError, match="Cannot apply scale"):
            k = 2.0 * Kernel.ehh(levels=5, active_dims=[0])
            analyze_kernel_tree(k, total_dim=3)

    def test_missing_active_dims(self):
        """Categorical without active_dims should fail during analysis."""
        k = Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=5)
        with pytest.raises(ValueError, match="exactly 1 active_dim"):
            analyze_kernel_tree(k, total_dim=3)

    def test_complex_mixed_composition(self):
        """(RBF * EHH) + (Matern * GD) is now supported as a mixed additive tree."""
        left = Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=5, active_dims=[1])
        right = Kernel.matern52(active_dims=[0]) * Kernel.gd(levels=3, active_dims=[2])
        k = left + right
        analysis = analyze_kernel_tree(k, total_dim=3)
        assert analysis.continuous_kernel.operator == "sum"
        assert len(analysis.categorical_specs) == 2


class TestCategoricalSpec:
    """Tests for the CategoricalSpec data class."""

    def test_param_names_in_spec(self):
        k = Kernel.ehh(levels=3, active_dims=[2])
        a = analyze_kernel_tree(k, total_dim=5)
        spec = a.categorical_specs[0]
        assert spec.param_names == [
            "ehh_col2_theta_0",
            "ehh_col2_theta_1",
            "ehh_col2_theta_2",
        ]

    def test_gd_spec(self):
        k = Kernel.rbf(active_dims=[0]) * Kernel.gd(levels=4, active_dims=[1])
        a = analyze_kernel_tree(k, total_dim=2)
        spec = a.categorical_specs[0]
        assert spec.kernel_type == KernelType.GD
        assert spec.levels == 4
        assert spec.col_index == 1
        assert spec.param_names == ["gd_col1_theta"]
