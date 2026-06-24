"""Unit tests for active_dims / DimSliceKernel feature.

Tests the kernel dimension routing infrastructure:
- KernelNode with active_dims
- Column permutation logic
- to_mojo_type() generation with DimSlice
- ARD interaction
- NumPy evaluate() reference with dimension slicing
- Serialization round-trip
"""

import numpy as np
import pytest

from mojogp.kernel import (
    Kernel,
    KernelNode,
    KernelType,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    Linear,
    RQ,
    Polynomial,
    make_ard_kernel,
    compute_dim_permutation,
)


class TestKernelNodeActiveDims:
    """Test KernelNode with active_dims field."""

    def test_rbf_with_active_dims(self):
        k = RBF(active_dims=[0, 1])
        assert k.active_dims == (0, 1)
        assert k.kernel_type == KernelType.RBF

    def test_active_dims_sorted(self):
        k = RBF(active_dims=[3, 1, 0])
        assert k.active_dims == (0, 1, 3)

    def test_no_active_dims_default(self):
        k = RBF()
        assert k.active_dims is None

    def test_with_active_dims_method(self):
        k = (RBF() * Matern52()).with_active_dims([0, 1])
        assert k.active_dims == (0, 1)
        assert k.operator == "product"

    def test_has_active_dims_leaf(self):
        k = RBF(active_dims=[0])
        assert k.has_active_dims() is True

    def test_has_active_dims_false(self):
        k = RBF() + Matern52()
        assert k.has_active_dims() is False

    def test_has_active_dims_nested(self):
        k = RBF(active_dims=[0, 1]) + Matern52()
        assert k.has_active_dims() is True

    def test_all_factory_methods_accept_active_dims(self):
        """Every kernel factory should accept active_dims."""
        factories = [
            RBF,
            Matern12,
            Matern32,
            Matern52,
            Periodic,
            Linear,
            RQ,
            Polynomial,
        ]
        for factory in factories:
            k = factory(active_dims=[0, 2])
            assert k.active_dims == (0, 2), f"{factory.__name__} failed"

    def test_repr_with_active_dims(self):
        k = RBF(active_dims=[0, 1])
        r = repr(k)
        assert "active_dims" in r
        assert "[0, 1]" in r

    def test_repr_composite_with_active_dims(self):
        k = RBF(active_dims=[0, 1]) * Periodic(active_dims=[2])
        r = repr(k)
        assert "active_dims" in r


class TestToMojoType:
    """Test Mojo type string generation with DimSlice."""

    def test_no_active_dims_unchanged(self):
        k = RBF()
        assert k.to_mojo_type() == "RBFComposable"

    def test_active_dims_generates_dimslice(self):
        k = RBF(active_dims=[0, 1])
        # Set the dim range (normally done by compute_dim_permutation)
        k._dim_start = 0
        k._dim_end = 2
        mojo_type = k.to_mojo_type()
        assert mojo_type == "DimSliceKernel[RBFComposable, 0, 2]"

    def test_product_with_dimslice(self):
        k1 = RBF(active_dims=[0, 1])
        k2 = Periodic(active_dims=[2])
        k1._dim_start, k1._dim_end = 0, 2
        k2._dim_start, k2._dim_end = 2, 3
        k = k1 * k2
        mojo_type = k.to_mojo_type()
        assert "DimSliceKernel[RBFComposable, 0, 2]" in mojo_type
        assert "DimSliceKernel[PeriodicComposable, 2, 3]" in mojo_type
        assert mojo_type.startswith("ProductKernel[")

    def test_composite_with_active_dims(self):
        """(RBF * Matern).with_active_dims([0,1]) should wrap the whole product."""
        inner = RBF() * Matern52()
        k = inner.with_active_dims([0, 1])
        k._dim_start = 0
        k._dim_end = 2
        mojo_type = k.to_mojo_type()
        assert mojo_type == (
            "DimSliceKernel[ProductKernel[RBFComposable, Matern52Composable], 0, 2]"
        )

    def test_ard_with_dimslice(self):
        k = RBF(active_dims=[0, 1, 2], ard=True)
        k.ard_dim = 3  # set as if make_ard_kernel was called
        k._dim_start, k._dim_end = 0, 3
        mojo_type = k.to_mojo_type()
        assert mojo_type == "DimSliceKernel[RBFComposableARD[3], 0, 3]"


class TestColumnPermutation:
    """Test the compute_dim_permutation function."""

    def test_contiguous_dims_no_reorder(self):
        """Contiguous active_dims [0,1] and [2] on 3D input → no permutation."""
        k = RBF(active_dims=[0, 1]) * Periodic(active_dims=[2])
        perm, eff_dim = compute_dim_permutation(k, total_dim=3)
        assert perm is None  # identity, no reorder needed
        assert eff_dim == 3

    def test_non_contiguous_dims_reorder(self):
        """Non-contiguous [0, 2] should produce a permutation."""
        k = RBF(active_dims=[0, 2]) * Periodic(active_dims=[1])
        perm, eff_dim = compute_dim_permutation(k, total_dim=3)
        assert perm is not None
        assert eff_dim == 3
        # RBF dims [0,2] should be contiguous in reordered layout
        rbf_start = k.left._dim_start
        rbf_end = k.left._dim_end
        assert rbf_end - rbf_start == 2

    def test_dim_ranges_set_correctly(self):
        """Verify _dim_start/_dim_end are set on kernel nodes."""
        k = RBF(active_dims=[0, 1]) * Matern52(active_dims=[2, 3])
        compute_dim_permutation(k, total_dim=4)
        assert k.left._dim_start == 0
        assert k.left._dim_end == 2
        assert k.right._dim_start == 2
        assert k.right._dim_end == 4

    def test_overlapping_dims(self):
        """Overlapping dims should duplicate in the permutation."""
        k = RBF(active_dims=[0, 1]) * Matern52(active_dims=[1, 2])
        perm, eff_dim = compute_dim_permutation(k, total_dim=3)
        # Dim 1 appears in both groups → duplicated
        assert eff_dim == 4  # 2 + 2 dims
        assert perm is not None
        assert perm.count(1) == 2  # dim 1 duplicated

    def test_validation_out_of_range(self):
        k = RBF(active_dims=[0, 5])
        with pytest.raises(ValueError, match="out of range"):
            compute_dim_permutation(k, total_dim=3)

    def test_validation_cat_dims_overlap(self):
        k = RBF(active_dims=[0, 1])
        with pytest.raises(ValueError, match="overlap with cat_dims"):
            compute_dim_permutation(k, total_dim=3, cat_col_indices=[1])

    def test_single_dim(self):
        """Single dim active_dims should work."""
        k = Periodic(active_dims=[2])
        perm, eff_dim = compute_dim_permutation(k, total_dim=3)
        assert k._dim_start is not None
        assert k._dim_end - k._dim_start == 1


class TestARDInteraction:
    """Test that ARD uses sliced dim count, not full input dim."""

    def test_ard_with_active_dims(self):
        k = RBF(active_dims=[0, 1, 2], ard=True)
        k_ard = make_ard_kernel(k, dim=5)  # full input is 5D
        assert k_ard.ard_dim == 3  # only 3 active dims

    def test_ard_without_active_dims(self):
        k = RBF(ard=True)
        k_ard = make_ard_kernel(k, dim=5)
        assert k_ard.ard_dim == 5  # full input dim

    def test_ard_preserves_active_dims(self):
        k = RBF(active_dims=[0, 1], ard=True)
        k_ard = make_ard_kernel(k, dim=5)
        assert k_ard.active_dims == (0, 1)
        assert k_ard.ard_dim == 2

    def test_ard_in_composition(self):
        k = RBF(active_dims=[0, 1], ard=True) + Matern52(active_dims=[2, 3], ard=True)
        k_ard = make_ard_kernel(k, dim=5)
        assert k_ard.left.ard_dim == 2
        assert k_ard.right.ard_dim == 2

    def test_ard_num_params(self):
        """ARD with active_dims should have len(active_dims)+1 params, not full_dim+1."""
        k = RBF(active_dims=[0, 1, 2], ard=True)
        k_ard = make_ard_kernel(k, dim=5)
        assert k_ard.num_params() == 4  # 3 lengthscales + 1 outputscale


class TestNumPyEvaluate:
    """Test that the NumPy evaluate() reference handles active_dims."""

    def test_active_dims_slices_input(self):
        """RBF(active_dims=[0,1]) should only use columns 0,1."""
        np.random.seed(42)
        X = np.random.randn(10, 4).astype(np.float64)
        params = np.array([1.0, 1.0], dtype=np.float64)  # ls=1, os=1

        # With active_dims
        k_sliced = RBF(active_dims=[0, 1])
        K_sliced = k_sliced.evaluate(X, params=params)

        # Without active_dims on 2D subset
        k_full = RBF()
        K_ref = k_full.evaluate(X[:, [0, 1]], params=params)

        np.testing.assert_allclose(K_sliced, K_ref, atol=1e-6)

    def test_product_with_active_dims(self):
        """RBF(active_dims=[0]) * Linear(active_dims=[1]) should split dims."""
        np.random.seed(42)
        X = np.random.randn(10, 3).astype(np.float64)

        k = RBF(active_dims=[0]) * Linear(active_dims=[1])
        K = k.evaluate(X)

        # Reference: compute each part separately
        k_rbf = RBF()
        K_rbf = k_rbf.evaluate(X[:, [0]])

        k_lin = Linear()
        K_lin = k_lin.evaluate(X[:, [1]])

        np.testing.assert_allclose(K, K_rbf * K_lin, atol=1e-6)


class TestSerialization:
    """Test to_dict/from_dict round-trip with active_dims."""

    def test_round_trip_leaf(self):
        k = RBF(active_dims=[0, 2], ard=True)
        d = k.to_dict()
        k2 = KernelNode.from_dict(d)
        assert k2.active_dims == (0, 2)
        assert k2.ard is True
        assert k2.kernel_type == KernelType.RBF

    def test_round_trip_composite(self):
        k = RBF(active_dims=[0, 1]) + Periodic(active_dims=[2])
        d = k.to_dict()
        k2 = KernelNode.from_dict(d)
        assert k2.left.active_dims == (0, 1)
        assert k2.right.active_dims == (2,)

    def test_round_trip_no_active_dims(self):
        k = RBF() + Matern52()
        d = k.to_dict()
        k2 = KernelNode.from_dict(d)
        assert k2.left.active_dims is None
        assert k2.right.active_dims is None
