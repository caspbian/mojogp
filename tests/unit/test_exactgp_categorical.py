"""Unit tests for single-output categorical kernel detection from kernel trees.

Tests that ExactGP correctly detects categorical kernels in the kernel tree
and routes through the mixed training path.
"""

import pytest
import numpy as np
from mojogp import SingleOutputGP, RBF, Matern52, EHH, GD, CR


class TestCategoricalDetection:
    """Test that ExactGP detects categorical kernels from the tree."""

    def test_pure_continuous_no_detection(self):
        """Pure continuous kernel should not trigger mixed path."""
        kernel = RBF()
        gp = SingleOutputGP(kernel)
        assert not gp._is_mixed
        assert gp.cat_dims == {}

    def test_detect_single_ehh(self):
        """RBF * EHH should be detected as mixed."""
        kernel = RBF(active_dims=[0, 1]) * EHH(levels=5, active_dims=[2])
        gp = SingleOutputGP(kernel)
        # Before fit, cat_dims is empty (detection happens in fit)
        assert gp.cat_dims == {}

    def test_fit_detects_categorical(self):
        """During fit(), categorical nodes should be detected and routed correctly."""
        np.random.seed(42)
        n = 100  # small n for unit test (no actual GP training)
        X = np.random.randn(n, 3).astype(np.float32)
        X[:, 2] = np.random.randint(0, 5, size=n).astype(
            np.float32
        )  # categorical column
        y = np.sin(X[:, 0]).astype(np.float32)

        kernel = RBF(active_dims=[0, 1]) * EHH(levels=5, active_dims=[2])
        gp = SingleOutputGP(kernel)

        # After fit() starts, the kernel tree analysis should detect the categorical
        # We can't do full training without GPU, but we can check detection in fit()
        # by using try/except to catch the compilation/GPU error
        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass  # Expected to fail without GPU/compiled modules

        # Check that detection happened
        assert gp._is_mixed, "Should be detected as mixed"
        assert 2 in gp.cat_dims, "Column 2 should be detected as categorical"
        assert gp.cat_dims[2] == 5, "Column 2 should have 5 levels"
        assert gp._cont_dim == 2, "Should have 2 continuous dims"

    def test_detect_multi_categorical(self):
        """Multiple categorical columns should all be detected."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 4).astype(np.float32)
        X[:, 2] = np.random.randint(0, 5, size=n).astype(np.float32)
        X[:, 3] = np.random.randint(0, 3, size=n).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        kernel = (
            RBF(active_dims=[0, 1])
            * EHH(levels=5, active_dims=[2])
            * GD(levels=3, active_dims=[3])
        )
        gp = SingleOutputGP(kernel)

        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        assert gp._is_mixed
        assert 2 in gp.cat_dims
        assert 3 in gp.cat_dims
        assert gp.cat_dims[2] == 5
        assert gp.cat_dims[3] == 3
        assert gp._cont_dim == 2

    def test_different_cat_kernels_per_column(self):
        """Different categorical kernel types should be preserved."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 4).astype(np.float32)
        X[:, 2] = np.random.randint(0, 5, size=n).astype(np.float32)
        X[:, 3] = np.random.randint(0, 3, size=n).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        kernel = (
            RBF(active_dims=[0, 1])
            * EHH(levels=5, active_dims=[2])
            * CR(levels=3, active_dims=[3])
        )
        gp = SingleOutputGP(kernel)

        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        assert gp._is_mixed
        # With different kernel types, cat_kernel should be a dict
        assert isinstance(gp.cat_kernel, dict)
        assert gp.cat_kernel[2] == "ehh"
        assert gp.cat_kernel[3] == "cr"

    def test_same_cat_kernels_simplified(self):
        """Same categorical kernel type should be simplified to string."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 4).astype(np.float32)
        X[:, 2] = np.random.randint(0, 5, size=n).astype(np.float32)
        X[:, 3] = np.random.randint(0, 3, size=n).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        kernel = (
            RBF(active_dims=[0, 1])
            * EHH(levels=5, active_dims=[2])
            * EHH(levels=3, active_dims=[3])
        )
        gp = SingleOutputGP(kernel)

        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        assert gp._is_mixed
        # Same kernel type: should be simplified to string
        assert gp.cat_kernel == "ehh"

    def test_analysis_stored(self):
        """The kernel tree analysis should be stored on the GP."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3).astype(np.float32)
        X[:, 2] = np.random.randint(0, 5, size=n).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        kernel = RBF(active_dims=[0, 1]) * EHH(levels=5, active_dims=[2])
        gp = SingleOutputGP(kernel)

        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        assert gp._analysis is not None
        assert not gp._analysis.is_pure_continuous
        assert len(gp._analysis.categorical_specs) == 1
