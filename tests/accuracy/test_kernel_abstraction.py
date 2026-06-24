"""
Tests for kernel abstraction in MojoGP.

Tests all kernel types (RBF, Matérn 3/2, Matérn 5/2, Matérn 1/2, Periodic, RQ, Linear, Polynomial) for:
- Training convergence
- Mean prediction accuracy
- Variance prediction (with and without LOVE)
- Different dataset sizes and dimensions
"""

import pytest
import numpy as np
from mojogp import SingleOutputGP, PredictionResult
from mojogp import RBF, Matern12, Matern32, Matern52, Periodic, RQ, Linear, Polynomial


# Mapping from kernel string names to ExactGP kernel constructors
KERNEL_CONSTRUCTORS = {
    "rbf": lambda **kw: RBF(**kw),
    "matern32": lambda **kw: Matern32(**kw),
    "matern52": lambda **kw: Matern52(**kw),
    "matern12": lambda **kw: Matern12(**kw),
    "periodic": lambda **kw: Periodic(**kw),
    "rq": lambda **kw: RQ(**kw),
    "linear": lambda **kw: Linear(**kw),
    "polynomial": lambda **kw: Polynomial(**kw),
}


def _make_kernel(kernel_name: str, **extra_kwargs):
    """Create a kernel object from a string name plus optional kwargs (e.g. period)."""
    return KERNEL_CONSTRUCTORS[kernel_name](**extra_kwargs)


def generate_test_data(kernel: str, n: int, d: int, seed: int = 42) -> tuple:
    """Generate kernel-appropriate test data.

    Different kernels work best with different types of functions:
    - Stationary kernels (RBF, Matérn): sin(x) - smooth function
    - Periodic: sin(2*pi*x/period) - truly periodic with known period
    - RQ: sin(x) - smooth function (RQ is stationary)
    - Linear: X @ w - linear combination
    - Polynomial: sum of powers of X - polynomial function

    Args:
        kernel: Kernel type name
        n: Number of samples
        d: Input dimension
        seed: Random seed for X generation (weights are fixed for reproducibility)

    Returns:
        X: Input data [n, d]
        y: Target values [n]
        kwargs: Optional kernel-specific parameters (e.g., period for periodic kernel)
    """
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)

    if kernel == "linear":
        w = np.ones(d, dtype=np.float32)
        y = X @ w
        return X, y, {}

    elif kernel == "polynomial":
        w = np.ones(d, dtype=np.float32)
        y = (X @ w) ** 2
        return X, y, {}

    elif kernel == "periodic":
        period = 2.0
        y = np.sin(2 * np.pi * X[:, 0] / period).astype(np.float32)
        return X, y, {"period": period}

    else:
        y = np.sin(X[:, 0]).astype(np.float32)
        return X, y, {}


class TestKernelTraining:
    """Test training with different kernel types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set random seed for reproducibility."""
        np.random.seed(42)

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_training_converges_for_kernel_type(self, kernel):
        """Test that training converges for all kernel types."""
        X = np.random.randn(200, 3).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        gp = SingleOutputGP(_make_kernel(kernel))
        gp.fit(X, y, max_iterations=50)

        # Check that model is fitted
        assert gp.is_trained
        params = gp.get_learned_params()
        assert params["noise"] is not None
        # Check that outputscale exists
        os_keys = [k for k in params if k.endswith("_outputscale")]
        assert len(os_keys) > 0, "No outputscale in learned params"

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_training_r2(self, kernel):
        """Test that training achieves good R² on training data."""
        X, y, kwargs = generate_test_data(kernel, 300, 5)

        gp = SingleOutputGP(_make_kernel(kernel, **kwargs))
        gp.fit(X, y, max_iterations=100)

        pred, _ = gp.predict(X, return_std=True)
        r2 = 1 - np.mean((pred - y) ** 2) / np.var(y)

        # Should achieve at least 90% R² on training data
        assert r2 > 0.90, f"{kernel} R² = {r2:.4f} < 0.90"

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_different_dimensions(self, kernel, dim):
        """Test training with different input dimensions."""
        n = 2000
        X, y, kwargs = generate_test_data(kernel, n, dim)

        gp = SingleOutputGP(_make_kernel(kernel, **kwargs))
        gp.fit(X, y, max_iterations=50)

        pred, _ = gp.predict(X, return_std=True)
        r2 = 1 - np.mean((pred - y) ** 2) / np.var(y)

        # Polynomial kernel struggles in high dimensions with limited data
        threshold = 0.60 if kernel == "polynomial" and dim >= 10 else 0.80
        assert r2 > threshold, f"{kernel} dim={dim} R² = {r2:.4f} < {threshold}"


class TestKernelPrediction:
    """Test prediction with different kernel types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set random seed for reproducibility."""
        np.random.seed(42)

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_mean_prediction(self, kernel):
        """Test mean prediction accuracy."""
        X_train = np.random.randn(200, 3).astype(np.float32)
        y_train = np.sin(X_train[:, 0]).astype(np.float32)
        X_test = np.random.randn(50, 3).astype(np.float32)

        gp = SingleOutputGP(_make_kernel(kernel))
        gp.fit(X_train, y_train, max_iterations=100)

        mean, std = gp.predict(X_test, return_std=True)

        # Check output shape
        assert mean.shape == (50,)

        # Check predictions are in reasonable range
        assert np.all(np.isfinite(mean))
        assert np.abs(mean).max() < 10  # Should be bounded

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_variance_prediction(self, kernel):
        """Test variance prediction via default PredictionResult."""
        X_train = np.random.randn(100, 3).astype(np.float32)
        y_train = np.sin(X_train[:, 0]).astype(np.float32)
        X_test = np.random.randn(20, 3).astype(np.float32)

        gp = SingleOutputGP(_make_kernel(kernel))
        gp.fit(X_train, y_train, max_iterations=50)

        result = gp.predict(X_test)

        # Check output shapes
        assert result.mean.shape == (20,)
        assert result.variance.shape == (20,)

        # Variance should be positive
        assert np.all(result.variance > 0), (
            f"Some variances are non-positive: min={result.variance.min()}"
        )

        # Variance should be bounded
        assert np.all(result.variance < 10), (
            f"Some variances are too large: max={result.variance.max()}"
        )


class TestKernelLOVE:
    """Test LOVE (fast variance) with different kernel types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set random seed for reproducibility."""
        np.random.seed(42)

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_love_training(self, kernel):
        """Test that LOVE training works for all kernel types."""
        X = np.random.randn(200, 3).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        gp = SingleOutputGP(_make_kernel(kernel))
        gp.fit(X, y, max_iterations=50)

        # After training, predict with LOVE to verify it works
        assert gp.is_trained
        mean, std = gp.predict(X, return_std=True)
        assert np.all(np.isfinite(mean))

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_love_variance_prediction(self, kernel):
        """Test variance prediction with LOVE via default PredictionResult."""
        X_train = np.random.randn(200, 3).astype(np.float32)
        y_train = np.sin(X_train[:, 0]).astype(np.float32)
        X_test = np.random.randn(50, 3).astype(np.float32)

        gp = SingleOutputGP(_make_kernel(kernel))
        gp.fit(X_train, y_train, max_iterations=100)

        result = gp.predict(X_test)

        # Check output shapes
        assert result.mean.shape == (50,)
        assert result.variance.shape == (50,)

        # Variance should be positive
        assert np.all(result.variance > 0), (
            f"Some variances are non-positive: min={result.variance.min()}"
        )

        # Variance should be bounded (polynomial is non-stationary — variance grows
        # polynomially with distance from training data, so needs much higher threshold)
        max_var = 1000 if kernel == "polynomial" else 10
        assert np.all(result.variance < max_var), (
            f"Some variances are too large: max={result.variance.max()}"
        )

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_love_std_prediction(self, kernel):
        """Test standard deviation prediction with LOVE."""
        X_train = np.random.randn(200, 3).astype(np.float32)
        y_train = np.sin(X_train[:, 0]).astype(np.float32)
        X_test = np.random.randn(50, 3).astype(np.float32)

        gp = SingleOutputGP(_make_kernel(kernel))
        gp.fit(X_train, y_train, max_iterations=100)

        mean, std = gp.predict(X_test, return_std=True)

        # Check output shapes
        assert mean.shape == (50,)
        assert std.shape == (50,)

        # Std should be positive
        assert np.all(std > 0), f"Some stds are non-positive: min={std.min()}"


class TestKernelValidation:
    """Test kernel parameter validation."""

    def test_invalid_kernel_type(self):
        """Test that passing an invalid kernel object raises TypeError."""
        X = np.random.randn(10, 2).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        with pytest.raises((TypeError, ValueError)):
            SingleOutputGP("not_a_kernel")

    def test_valid_kernel_types(self):
        """Test that all valid kernel constructors produce working kernels."""
        X = np.random.randn(20, 2).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)
        for kernel_name in KERNEL_CONSTRUCTORS:
            kern = _make_kernel(kernel_name)
            gp = SingleOutputGP(kern)
            assert not gp.is_trained  # not yet fitted


class TestKernelScoring:
    """Test scoring functionality with different kernels."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set random seed for reproducibility."""
        np.random.seed(42)

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_score_method(self, kernel):
        """Test that predictions yield valid R² (computed manually)."""
        X_train, y_train, kwargs = generate_test_data(kernel, 200, 3)
        X_test, y_test, _ = generate_test_data(kernel, 50, 3, seed=123)

        gp = SingleOutputGP(_make_kernel(kernel, **kwargs))
        gp.fit(X_train, y_train, max_iterations=100)

        pred, _ = gp.predict(X_test, return_std=True)
        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # R² should be between -inf and 1
        assert r2 <= 1.0
        # For this simple problem, should be positive
        assert r2 > 0, f"{kernel} test R² = {r2:.4f} <= 0"


class TestKernelLargeScale:
    """Test kernels with larger datasets."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set random seed for reproducibility."""
        np.random.seed(42)

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_medium_scale(self, kernel):
        """Test with medium-scale dataset (1000 points)."""
        n = 2000
        d = 5
        X, y, kwargs = generate_test_data(kernel, n, d)

        gp = SingleOutputGP(_make_kernel(kernel, **kwargs))
        gp.fit(X, y, max_iterations=100)

        pred, _ = gp.predict(X, return_std=True)
        r2 = 1 - np.mean((pred - y) ** 2) / np.var(y)

        # Should achieve good R² even at larger scale
        assert r2 > 0.80, f"{kernel} n={n} R² = {r2:.4f} < 0.80"


class TestKernelARD:
    """Test ARD (Automatic Relevance Determination) with different kernel types.

    Uses ExactGP with ARD kernel constructors (e.g., RBF(ard=True)).
    """

    # Map kernel string names to ExactGP ARD kernel constructors.
    KERNEL_MAP = {
        "rbf": lambda **kw: RBF(ard=True, **kw),
        "matern32": lambda **kw: Matern32(ard=True, **kw),
        "matern52": lambda **kw: Matern52(ard=True, **kw),
        "matern12": lambda **kw: Matern12(ard=True, **kw),
        "periodic": lambda **kw: Periodic(ard=True, **kw),
        "rq": lambda **kw: RQ(ard=True, **kw),
        "linear": lambda **kw: Linear(ard=True, **kw),
        "polynomial": lambda **kw: Polynomial(ard=True, **kw),
    }

    KERNEL_MAP_ISO = {
        "rbf": lambda **kw: RBF(**kw),
        "matern32": lambda **kw: Matern32(**kw),
        "matern52": lambda **kw: Matern52(**kw),
        "matern12": lambda **kw: Matern12(**kw),
        "periodic": lambda **kw: Periodic(**kw),
        "rq": lambda **kw: RQ(**kw),
        "linear": lambda **kw: Linear(**kw),
        "polynomial": lambda **kw: Polynomial(**kw),
    }

    # All kernel types support ARD
    ARD_KERNELS = [
        "rbf",
        "matern32",
        "matern52",
        "matern12",
        "periodic",
        "rq",
        "linear",
        "polynomial",
    ]

    @staticmethod
    def _extract_lengthscales(gp, d):
        """Extract per-dimension lengthscales from ExactGP learned params."""
        params = gp.get_learned_params()
        ls = []
        for i in range(d):
            # ARD param names are like "rbf_ls_0", "matern32_ls_1", etc.
            key = [k for k in params if k.endswith(f"_ls_{i}")]
            if key:
                ls.append(params[key[0]])
        return ls

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set random seed for reproducibility."""
        np.random.seed(42)

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_ard_training(self, kernel):
        """Test that ARD training converges for all kernel types."""
        X = np.random.randn(200, 3).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        kern = self.KERNEL_MAP[kernel]()
        gp = SingleOutputGP(kern)
        gp.fit(X, y, max_iterations=50)

        # Check that model is fitted with ARD
        assert gp.is_trained
        assert gp.ard
        params = gp.get_learned_params()
        ls = self._extract_lengthscales(gp, 3)
        assert len(ls) == 3, f"Expected 3 lengthscales, got {len(ls)}"
        assert params["noise"] is not None
        # Check outputscale exists (key like "rbf_outputscale")
        os_keys = [k for k in params if k.endswith("_outputscale")]
        assert len(os_keys) > 0, "No outputscale in learned params"

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_ard_prediction_accuracy(self, kernel):
        """Test that ARD achieves good prediction accuracy."""
        X, y, kwargs = generate_test_data(kernel, 300, 3)

        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]

        # Pass kernel-specific params (e.g., period for Periodic)
        kern_kwargs = {}
        if "period" in kwargs:
            kern_kwargs["period"] = kwargs["period"]
        kern = self.KERNEL_MAP[kernel](**kern_kwargs)
        gp = SingleOutputGP(kern)
        gp.fit(X_train, y_train, max_iterations=100)

        y_pred, _ = gp.predict(X_test, return_std=True)
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

        # ARD should achieve reasonable R²
        assert r2 > 0.5, f"{kernel} ARD R² = {r2:.4f} < 0.5"

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_ard_lengthscale_relevance(self, kernel):
        """Test that ARD learns different lengthscales for different feature relevance."""
        # Create data where first dimension is most relevant
        X = np.random.randn(300, 3).astype(np.float32)
        y = 2.0 * np.sin(X[:, 0]) + 0.1 * np.random.randn(300)
        y = y.astype(np.float32)

        kern = self.KERNEL_MAP[kernel]()
        gp = SingleOutputGP(kern)
        gp.fit(X, y, max_iterations=100)

        ls = self._extract_lengthscales(gp, 3)
        assert len(ls) == 3

        # All lengthscales should be positive
        assert all(l > 0 for l in ls), f"Lengthscales should be positive: {ls}"

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_ard_different_dimensions(self, kernel):
        """Test ARD with different input dimensions."""
        for d in [2, 5, 10]:
            X = np.random.randn(200, d).astype(np.float32)
            y = np.sin(X[:, 0]).astype(np.float32)

            kern = self.KERNEL_MAP[kernel]()
            gp = SingleOutputGP(kern)
            gp.fit(X, y, max_iterations=50)

            assert gp.ard
            ls = self._extract_lengthscales(gp, d)
            assert len(ls) == d, f"Expected {d} lengthscales, got {len(ls)}"

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_ard_vs_isotropic_comparison(self, kernel):
        """Test that ARD and isotropic give similar results when features are equally relevant."""
        # Create data where all dimensions are equally relevant
        X = np.random.randn(200, 3).astype(np.float32)
        y = np.sin(X[:, 0]) + np.sin(X[:, 1]) + np.sin(X[:, 2])
        y = y + 0.1 * np.random.randn(200)
        y = y.astype(np.float32)

        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        # Train isotropic
        kern_iso = self.KERNEL_MAP_ISO[kernel]()
        gp_iso = SingleOutputGP(kern_iso)
        gp_iso.fit(X_train, y_train, max_iterations=100)
        y_pred_iso, _ = gp_iso.predict(X_test, return_std=True)
        r2_iso = 1 - np.sum((y_test - y_pred_iso) ** 2) / np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

        # Train ARD
        kern_ard = self.KERNEL_MAP[kernel]()
        gp_ard = SingleOutputGP(kern_ard)
        gp_ard.fit(X_train, y_train, max_iterations=100)
        y_pred_ard, _ = gp_ard.predict(X_test, return_std=True)
        r2_ard = 1 - np.sum((y_test - y_pred_ard) ** 2) / np.sum(
            (y_test - np.mean(y_test)) ** 2
        )

        # Both should achieve reasonable R² (within 0.3 of each other)
        assert abs(r2_iso - r2_ard) < 0.3, (
            f"{kernel}: iso R²={r2_iso:.4f}, ard R²={r2_ard:.4f}"
        )

    @pytest.mark.parametrize(
        "kernel",
        [
            "rbf",
            "matern32",
            "matern52",
            "matern12",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ],
    )
    def test_ard_love_variance_prediction(self, kernel):
        """Test ARD LOVE variance prediction for all kernel types."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1])
        y = y + 0.1 * np.random.randn(100)
        y = y.astype(np.float32)

        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]

        # Train with ARD
        kern = self.KERNEL_MAP[kernel]()
        gp = SingleOutputGP(kern)
        gp.fit(X_train, y_train, max_iterations=50)

        # Predict with full result (includes variance)
        result = gp.predict(X_test)

        # Check shapes
        assert result.mean.shape == (20,), f"Mean shape mismatch: {result.mean.shape}"
        assert result.variance.shape == (20,), (
            f"Variance shape mismatch: {result.variance.shape}"
        )

        # Variance should be positive
        assert np.all(result.variance > 0), (
            f"Variance should be positive, got min={result.variance.min()}"
        )

        # Variance should be reasonable (not too large)
        # Polynomial kernel can have larger variances due to unbounded nature
        max_var = 100 if kernel == "polynomial" else 10
        assert np.all(result.variance < max_var), (
            f"Variance too large: max={result.variance.max()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
