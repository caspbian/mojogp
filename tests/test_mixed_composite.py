"""Tests for mixed composite and categorical kernel integration.

Tests the ExactGP API with categorical kernel nodes in the kernel tree, verifying:
1. ExactGP construction with categorical kernels (auto-detected from tree)
2. Data splitting (continuous vs categorical columns)
3. JIT compilation of mixed composite modules
4. End-to-end training with composite + categorical kernels
5. Prediction with trained mixed GP
6. Multiple categorical kernel variants
7. Both matrix-free and materialized methods

NOTE: These tests require JIT compilation of Mojo modules, which can take
15-20 minutes per unique kernel configuration. Tests are designed to share
compiled modules where possible.
"""

import numpy as np
import pytest

from mojogp import SingleOutputGP, Kernel, EHH, GD, CR
from mojogp.gp import MixedTrainingResult


# =============================================================================
# Helper functions
# =============================================================================


def generate_mixed_data(
    n: int = 200,
    cont_dim: int = 3,
    cat_levels: list = None,
    noise_std: float = 0.1,
    seed: int = 42,
):
    """Generate synthetic mixed continuous + categorical data.

    The function is: y = sin(x_0) + cos(x_1) + cat_effect + noise
    where cat_effect depends on the categorical variable levels.

    Returns:
        X: [n, cont_dim + num_cat_vars] with continuous and categorical columns
        y: [n] targets
        cat_col_indices: list of column indices that are categorical
    """
    if cat_levels is None:
        cat_levels = [3]

    rng = np.random.RandomState(seed)
    num_cat_vars = len(cat_levels)

    # Continuous features
    X_cont = rng.randn(n, cont_dim).astype(np.float32)

    # Categorical features (integer indices)
    C = np.column_stack([rng.randint(0, L, size=n) for L in cat_levels]).astype(
        np.float32
    )  # float32 because we'll stack with X_cont

    # Target: continuous signal + categorical effect
    y = np.sin(X_cont[:, 0]) + 0.5 * np.cos(X_cont[:, 1])

    # Add categorical effects
    for v in range(num_cat_vars):
        # Each level adds a different offset
        level_effects = rng.randn(cat_levels[v]) * 0.5
        for i in range(n):
            y[i] += level_effects[int(C[i, v])]

    y += rng.randn(n).astype(np.float32) * noise_std
    y = y.astype(np.float32)

    # Stack into single array: [X_cont | C]
    # Categorical columns are at indices [cont_dim, cont_dim+1, ...]
    X = np.column_stack([X_cont, C]).astype(np.float32)

    cat_col_indices = list(range(cont_dim, cont_dim + num_cat_vars))

    return X, y, cat_col_indices


def _dummy_Xy(dim, n=10):
    """Create dummy X and y data for construction-only tests."""
    X = np.zeros((n, dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    return X, y


# =============================================================================
# Test 1: ExactGP construction with categorical config
# =============================================================================


class TestExactGPConstruction:
    """Test SingleOutputGP initialization with categorical dimensions via kernel tree."""

    def test_categorical_kernel_nodes_mark_model_as_mixed(self):
        """SingleOutputGP with categorical kernel nodes should detect mixed after fit."""
        kernel = (
            Kernel.rbf(active_dims=[0, 1, 2])
            * EHH(levels=4, active_dims=[3])
            * EHH(levels=3, active_dims=[4])
        )
        X, y = _dummy_Xy(5)
        gp = SingleOutputGP(kernel)

        # Before fit, cat_dims is empty (detection happens in fit)
        assert gp.cat_dims == {}

        # Trigger detection by calling fit (will fail without GPU, but detection happens first)
        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        assert gp._is_mixed
        assert gp._cont_dim == 3  # 5 total - 2 categorical = 3 continuous
        assert gp.dim == 5
        assert gp.cat_dims == {3: 4, 4: 3}
        assert gp.cat_kernel == "ehh"

    def test_no_categorical(self):
        """ExactGP without cat_dims should behave as pure continuous."""
        kernel = Kernel.rbf()
        X, y = _dummy_Xy(5)
        gp = SingleOutputGP(kernel)

        assert not gp._is_mixed
        assert gp.cat_dims == {}

    def test_invalid_dims(self):
        """Should raise if all dims are categorical."""
        kernel = EHH(levels=3, active_dims=[0]) * EHH(levels=4, active_dims=[1])
        X, y = _dummy_Xy(2)
        with pytest.raises((ValueError,), match="(continuous|Pure categorical)"):
            gp = SingleOutputGP(kernel)
            gp.fit(X, y, max_iterations=1)

    def test_repr_pure(self):
        """Repr without categorical should not show cat info."""
        kernel = Kernel.rbf()
        X, y = _dummy_Xy(4)
        gp = SingleOutputGP(kernel)
        r = repr(gp)
        assert "cat_dims" not in r

    def test_per_variable_kernel(self):
        """Per-variable categorical kernel selection via kernel tree."""
        kernel = (
            Kernel.rbf(active_dims=[0, 1, 2])
            * EHH(levels=4, active_dims=[3])
            * GD(levels=3, active_dims=[4])
        )
        X, y = _dummy_Xy(5)
        gp = SingleOutputGP(kernel)

        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        assert gp._is_mixed
        assert isinstance(gp.cat_kernel, dict)
        assert gp.cat_kernel[3] == "ehh"
        assert gp.cat_kernel[4] == "gd"

    def test_sum_kernel_with_categorical(self):
        """Sum kernel + categorical should work."""
        kernel = (Kernel.rbf() + Kernel.matern52()) * GD(levels=3, active_dims=[4])
        X, y = _dummy_Xy(5)
        gp = SingleOutputGP(kernel)

        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        assert gp._is_mixed


# =============================================================================
# Test 2: Data splitting
# =============================================================================


class TestDataSplitting:
    """Test splitting of input data into continuous and categorical parts."""

    def test_continuous_and_categorical_columns_are_split(self):
        """Continuous and categorical columns are split into separate arrays."""
        kernel = (
            Kernel.rbf(active_dims=[0, 1, 2])
            * EHH(levels=4, active_dims=[3])
            * EHH(levels=3, active_dims=[4])
        )
        X = np.random.randn(10, 5).astype(np.float32)
        X[:, 3] = np.random.randint(0, 4, size=10).astype(np.float32)
        X[:, 4] = np.random.randint(0, 3, size=10).astype(np.float32)
        y = np.zeros(10, dtype=np.float32)
        gp = SingleOutputGP(kernel)

        # Trigger cat_dims detection
        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        X_cont, C = gp._split_data(X)

        assert X_cont.shape == (10, 3)  # Columns 0, 1, 2
        assert C.shape == (10, 2)  # Columns 3, 4
        np.testing.assert_array_equal(X_cont, X[:, [0, 1, 2]])
        np.testing.assert_array_equal(C, X[:, [3, 4]].astype(np.int32))

    def test_split_no_categorical(self):
        """No categorical dims should return None for C."""
        kernel = Kernel.rbf()
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.zeros(10, dtype=np.float32)
        gp = SingleOutputGP(kernel)

        X_cont, C = gp._split_data(X)

        assert X_cont.shape == (10, 5)
        assert C is None

    def test_split_preserves_values(self):
        """Splitting should preserve exact values."""
        kernel = (
            Kernel.rbf(active_dims=[0, 2])
            * GD(levels=3, active_dims=[1])
            * GD(levels=5, active_dims=[3])
        )
        X = np.array(
            [
                [1.0, 0, 2.0, 1],
                [3.0, 2, 4.0, 0],
            ],
            dtype=np.float32,
        )
        y = np.zeros(2, dtype=np.float32)
        gp = SingleOutputGP(kernel)

        # Trigger detection
        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass

        X_cont, C = gp._split_data(X)

        # Continuous columns: 0, 2
        np.testing.assert_array_equal(X_cont, [[1.0, 2.0], [3.0, 4.0]])
        # Categorical columns: 1, 3
        np.testing.assert_array_equal(C, [[0, 1], [2, 0]])


# =============================================================================
# Test 3: JIT compilation (smoke test)
# =============================================================================


class TestJITCompilation:
    """Test that mixed composite modules can be JIT-compiled."""

    @pytest.mark.slow
    def test_compile_rbf_with_categorical(self):
        """Compile RBF + categorical module."""
        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=[3])
        X, y = _dummy_Xy(4)
        X[:, 3] = np.random.randint(0, 3, size=X.shape[0]).astype(np.float32)
        gp = SingleOutputGP(kernel, verbose=True)
        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass
        gp._ensure_compiled()
        assert gp._module is not None

    @pytest.mark.slow
    def test_compile_sum_kernel_with_categorical(self):
        """Compile sum kernel + categorical module."""
        kernel = (Kernel.rbf() + Kernel.matern52()) * EHH(levels=3, active_dims=[4])
        X, y = _dummy_Xy(5)
        X[:, 4] = np.random.randint(0, 3, size=X.shape[0]).astype(np.float32)
        gp = SingleOutputGP(kernel, verbose=True)
        try:
            gp.fit(X, y, max_iterations=1)
        except Exception:
            pass
        gp._ensure_compiled()
        assert gp._module is not None


# =============================================================================
# Test 4: End-to-end training
# =============================================================================


class TestMixedTraining:
    """End-to-end training tests for mixed composite + categorical GP."""

    @pytest.mark.slow
    def test_train_rbf_gd(self):
        """Train RBF + GD categorical kernel."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3])

        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)

        result = gp.fit(X, y, max_iterations=50, learning_rate=0.01)

        assert isinstance(result, MixedTrainingResult)
        assert result.params.shape[0] == 2  # RBF: lengthscale + outputscale
        assert result.cat_params.shape[0] > 0  # GD has 1 param per variable
        assert result.alpha.shape[0] == 2000
        assert np.isfinite(result.nll)
        assert result.noise > 0

    @pytest.mark.slow
    def test_train_rbf_ehh(self):
        """Train RBF + EHH categorical kernel."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[4])

        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * EHH(levels=4, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)

        result = gp.fit(X, y, max_iterations=50, learning_rate=0.01)

        assert isinstance(result, MixedTrainingResult)
        # EHH has L*(L-1)/2 = 4*3/2 = 6 params for L=4
        assert result.cat_params.shape[0] == 6
        assert np.isfinite(result.nll)

    @pytest.mark.slow
    def test_train_sum_kernel_with_categorical(self):
        """Train (RBF + Matern52) + categorical."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3])

        kernel = (Kernel.rbf() + Kernel.matern52()) * GD(levels=3, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)

        result = gp.fit(X, y, max_iterations=50, learning_rate=0.01)

        assert isinstance(result, MixedTrainingResult)
        assert result.params.shape[0] == 4  # 2 (RBF) + 2 (Matern52)
        assert np.isfinite(result.nll)

    @pytest.mark.slow
    def test_train_multiple_cat_vars(self):
        """Train with multiple categorical variables."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3, 4])

        kernel = (
            Kernel.rbf(active_dims=[0, 1, 2])
            * GD(levels=3, active_dims=[cat_cols[0]])
            * GD(levels=4, active_dims=[cat_cols[1]])
        )
        gp = SingleOutputGP(kernel)

        result = gp.fit(X, y, max_iterations=50, learning_rate=0.01)

        assert isinstance(result, MixedTrainingResult)
        # GD: 1 param per variable = 2 total
        assert result.cat_params.shape[0] == 2
        assert np.isfinite(result.nll)

    @pytest.mark.slow
    def test_train_materialized_method(self):
        """Train with materialized method."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3])

        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)

        result = gp.fit(
            X, y, max_iterations=50, learning_rate=0.01, method="materialized"
        )

        assert isinstance(result, MixedTrainingResult)
        assert np.isfinite(result.nll)


# =============================================================================
# Test 5: Prediction
# =============================================================================


class TestMixedPrediction:
    """Test prediction with trained mixed GP."""

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["matrix_free", "materialized"])
    def test_predict_love_variance_tracks_exact_on_new_data(self, method):
        """Mixed LOVE variance should stay informative on new test points."""
        X, y, cat_cols = generate_mixed_data(
            n=2000, cont_dim=3, cat_levels=[3], seed=123
        )

        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * EHH(levels=3, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=20, learning_rate=0.01, method=method)

        X_test, _, _ = generate_mixed_data(n=64, cont_dim=3, cat_levels=[3], seed=999)
        pred_love = gp.predict(X_test, variance_method="love")
        pred_exact = gp.predict(X_test, variance_method="exact")

        var_love = pred_love.variance
        var_exact = pred_exact.variance
        assert np.all(np.isfinite(var_love))
        assert np.all(np.isfinite(var_exact))
        assert np.all(var_love >= 0)
        assert np.all(var_exact >= 0)

        # Regression guard for the broken probe-solve path, which collapsed to 1e-10.
        assert float(np.max(var_love)) > 1e-4
        assert float(np.std(var_love)) > 1e-6

        mask = var_exact > 1e-5
        assert np.any(mask)
        rel_err = np.abs(var_love[mask] - var_exact[mask]) / (var_exact[mask] + 1e-6)
        assert float(np.mean(rel_err < 5.0)) > 0.5

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["matrix_free", "materialized"])
    def test_predict_love_variance_tracks_exact_on_new_data_with_ard(self, method):
        """Mixed ARD LOVE variance should stay aligned with exact prediction."""
        rng = np.random.RandomState(456)
        X_cont = rng.randn(2000, 3).astype(np.float32)
        C = rng.randint(0, 3, size=(2000, 1)).astype(np.float32)
        level_effects = np.array([0.0, 0.35, -0.2], dtype=np.float32)
        y = (
            np.sin(1.4 * X_cont[:, 0])
            + 0.3 * np.cos(X_cont[:, 1])
            + 0.1 * X_cont[:, 2]
            + level_effects[C[:, 0].astype(np.int32)]
            + 0.08 * rng.randn(2000)
        ).astype(np.float32)
        X = np.column_stack([X_cont, C]).astype(np.float32)

        Xc_test = rng.randn(64, 3).astype(np.float32)
        Ct_test = rng.randint(0, 3, size=(64, 1)).astype(np.float32)
        X_test = np.column_stack([Xc_test, Ct_test]).astype(np.float32)

        kernel = Kernel.rbf(active_dims=[0, 1, 2], ard=True) * EHH(
            levels=3, active_dims=[3]
        )
        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=20, learning_rate=0.01, method=method)

        pred_love = gp.predict(X_test, variance_method="love")
        pred_exact = gp.predict(X_test, variance_method="exact")

        var_love = pred_love.variance
        var_exact = pred_exact.variance
        assert np.all(np.isfinite(var_love))
        assert np.all(np.isfinite(var_exact))
        assert np.all(var_love >= 0)
        assert np.all(var_exact >= 0)

        mask = var_exact > 1e-5
        assert np.any(mask)
        rel_err = np.abs(var_love[mask] - var_exact[mask]) / (var_exact[mask] + 1e-6)
        assert float(np.mean(rel_err < 5.0)) > 0.9

    @pytest.mark.slow
    def test_predict_mean(self):
        """Predict mean with trained mixed GP."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3])

        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=50, learning_rate=0.01)

        # Predict on training data (should be close to y)
        result = gp.predict(X)
        assert result.mean.shape == (2000,)
        assert np.all(np.isfinite(result.mean))

    @pytest.mark.slow
    def test_predict_new_data(self):
        """Predict on new test data."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3])

        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=50, learning_rate=0.01)

        # Generate new test data
        X_test, _, _ = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3], seed=99)
        result = gp.predict(X_test)

        assert result.mean.shape == (2000,)
        assert np.all(np.isfinite(result.mean))

    @pytest.mark.slow
    def test_predict_return_std(self):
        """Predict with return_std=True."""
        X, y, cat_cols = generate_mixed_data(n=2000, cont_dim=3, cat_levels=[3])

        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=cat_cols)
        gp = SingleOutputGP(kernel)
        gp.fit(X, y, max_iterations=50, learning_rate=0.01)

        pred = gp.predict(X)
        assert pred.mean.shape == (2000,)
        assert pred.std.shape == (2000,)
        assert np.all(pred.std > 0)

    @pytest.mark.slow
    def test_predict_before_train_raises(self):
        """Predict before training should raise."""
        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=[3])
        gp = SingleOutputGP(kernel)

        with pytest.raises(RuntimeError, match="trained"):
            gp.predict(np.random.randn(10, 4).astype(np.float32))


# =============================================================================
# Test 6: Input validation
# =============================================================================


class TestInputValidation:
    """Test input validation for mixed GP."""

    def test_wrong_dim(self):
        """Wrong input dimension should raise during fit."""
        # active_dims=[4] references col 4, but X only has 3 columns
        kernel = Kernel.rbf(active_dims=[0, 1, 2]) * GD(levels=3, active_dims=[4])

        X = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)

        with pytest.raises((ValueError, IndexError)):
            gp = SingleOutputGP(kernel)
            gp.fit(X, y, max_iterations=1)

    def test_mismatched_samples(self):
        """Mismatched X and y sizes should raise."""
        kernel = Kernel.rbf(active_dims=[0, 1, 2, 3]) * GD(levels=3, active_dims=[4])

        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(8).astype(np.float32)

        with pytest.raises(ValueError, match="samples"):
            gp = SingleOutputGP(kernel)
            gp.fit(X, y, max_iterations=1)

    def test_1d_input_raises(self):
        """1D input should raise."""
        kernel = Kernel.rbf(active_dims=[0, 1, 2, 3]) * GD(levels=3, active_dims=[4])

        X = np.random.randn(10).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)

        with pytest.raises(ValueError, match="2D"):
            gp = SingleOutputGP(kernel)
            gp.fit(X, y, max_iterations=1)
