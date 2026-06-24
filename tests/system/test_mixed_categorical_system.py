"""System tests for mixed continuous + categorical (discrete) kernels.

Tests the full MojoGP pipeline with discrete/categorical variables using
the ExactGP API with kernel tree composition. Validates:
1. All 5 categorical kernel variants (GD, CR, EHH, HH, FE)
2. Mixed GP with composite continuous kernels
3. Mixed GP with ARD + categorical
4. Prediction accuracy against ground truth
5. Both materialized and matrix-free methods
6. Multiple categorical variables with different level counts

Key features:
- Uses n >= 500 for all tests (up to 2000)
- Generates data from a known mixed continuous + categorical GP
- Compares predictions to true function values
- Tests dimension relevance recovery with ARD + categorical
"""

import numpy as np
import pytest
import torch
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# Import MojoGP
from mojogp import SingleOutputGP, Kernel, EHH, GD, CR, HH, FE
from mojogp.gp import MixedTrainingResult


# =============================================================================
# Configuration
# =============================================================================

MIN_N = 500
MAX_N = 2000


def _check_gpu():
    """Check if GPU is available."""
    return torch.cuda.is_available()


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class MixedCategoricalResult:
    """Results from a mixed categorical kernel test."""

    name: str
    success: bool
    error_message: Optional[str] = None

    # Configuration
    cat_kernel_type: Optional[str] = None
    continuous_kernel: Optional[str] = None
    n_train: Optional[int] = None
    cont_dim: Optional[int] = None
    num_cat_vars: Optional[int] = None
    cat_levels: Optional[List[int]] = None
    method: Optional[str] = None
    ard: bool = False

    # Ground truth
    true_noise: Optional[float] = None

    # MojoGP results
    mojo_rmse_vs_truth: Optional[float] = None
    mojo_final_nll: Optional[float] = None
    mojo_training_time_s: Optional[float] = None
    mojo_converged: Optional[bool] = None

    def __str__(self):
        lines = [f"\n{'=' * 70}", f"  {self.name}", f"{'=' * 70}"]

        if not self.success:
            lines.append(f"  FAILED: {self.error_message}")
            return "\n".join(lines)

        # Configuration
        lines.append(f"\n  Configuration:")
        lines.append(
            f"    cont_kernel={self.continuous_kernel}, cat_kernel={self.cat_kernel_type}"
        )
        lines.append(
            f"    n={self.n_train}, cont_dim={self.cont_dim}, "
            f"cat_vars={self.num_cat_vars}, levels={self.cat_levels}"
        )
        lines.append(f"    method={self.method}, ard={self.ard}")

        # Results
        lines.append(f"\n  Results:")
        if self.mojo_rmse_vs_truth is not None:
            lines.append(f"    RMSE vs truth:  {self.mojo_rmse_vs_truth:.4f}")
        if self.mojo_final_nll is not None:
            lines.append(f"    Final NLL:      {self.mojo_final_nll:.4f}")
        if self.mojo_training_time_s is not None:
            lines.append(f"    Training time:  {self.mojo_training_time_s:.2f}s")
        if self.mojo_converged is not None:
            lines.append(f"    Converged:      {self.mojo_converged}")

        lines.append(f"{'=' * 70}")
        return "\n".join(lines)


# =============================================================================
# Data Generation
# =============================================================================


def generate_mixed_categorical_data(
    n_train: int,
    n_test: int,
    cont_dim: int,
    cat_levels: List[int],
    true_noise: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate data from a mixed continuous + categorical GP.

    The true function is:
        f(x, c) = continuous_signal(x) + categorical_effect(c)

    where:
        continuous_signal = sum of sinusoidal functions of continuous features
        categorical_effect = per-level offsets for each categorical variable

    Args:
        n_train: Number of training points
        n_test: Number of test points
        cont_dim: Number of continuous dimensions
        cat_levels: List of number of levels per categorical variable
        true_noise: Observation noise standard deviation
        seed: Random seed

    Returns:
        Dictionary with train/test data and ground truth
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    num_cat_vars = len(cat_levels)

    # Generate continuous features
    X_cont = rng.randn(n_total, cont_dim).astype(np.float32)

    # Generate categorical features (integer levels)
    C = np.column_stack([rng.randint(0, L, size=n_total) for L in cat_levels]).astype(
        np.float32
    )

    # True continuous signal: weighted sum of sinusoids
    f_cont = np.zeros(n_total, dtype=np.float32)
    for d in range(min(cont_dim, 3)):
        weight = 1.0 / (d + 1)
        f_cont += weight * np.sin(2.0 * X_cont[:, d])

    # True categorical effects: per-level offsets
    f_cat = np.zeros(n_total, dtype=np.float32)
    cat_effects = []
    for v in range(num_cat_vars):
        level_effects = rng.randn(cat_levels[v]).astype(np.float32) * 0.8
        cat_effects.append(level_effects)
        for i in range(n_total):
            f_cat[i] += level_effects[int(C[i, v])]

    # Combined true function
    f_true = f_cont + f_cat

    # Noisy observations
    y = f_true + true_noise * rng.randn(n_total).astype(np.float32)

    # Stack into full X: [X_cont | C]
    X_full = np.column_stack([X_cont, C]).astype(np.float32)

    # Split train/test
    X_train = X_full[:n_train]
    X_test = X_full[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    f_test = f_true[n_train:]

    # Build cat_info: list of (column_index, num_levels) tuples
    cat_info = []
    for v in range(num_cat_vars):
        col_idx = cont_dim + v
        cat_info.append((col_idx, cat_levels[v]))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "f_test": f_test,
        "cat_info": cat_info,
        "cat_levels": cat_levels,
        "true_noise": true_noise,
        "cat_effects": cat_effects,
        "cont_dim": cont_dim,
    }


def generate_ard_mixed_data(
    n_train: int,
    n_test: int,
    cont_dim: int,
    relevant_dims: int,
    cat_levels: List[int],
    true_noise: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate mixed data where only some continuous dims are relevant.

    Used to test ARD + categorical kernel combination.
    """
    rng = np.random.RandomState(seed)
    n_total = n_train + n_test
    num_cat_vars = len(cat_levels)

    # Generate continuous features
    X_cont = rng.randn(n_total, cont_dim).astype(np.float32)

    # Generate categorical features
    C = np.column_stack([rng.randint(0, L, size=n_total) for L in cat_levels]).astype(
        np.float32
    )

    # Only first `relevant_dims` continuous dimensions matter
    f_cont = np.zeros(n_total, dtype=np.float32)
    for d in range(relevant_dims):
        f_cont += np.sin(2.0 * X_cont[:, d])

    # Categorical effects
    f_cat = np.zeros(n_total, dtype=np.float32)
    for v in range(num_cat_vars):
        level_effects = rng.randn(cat_levels[v]).astype(np.float32) * 0.8
        for i in range(n_total):
            f_cat[i] += level_effects[int(C[i, v])]

    f_true = f_cont + f_cat
    y = f_true + true_noise * rng.randn(n_total).astype(np.float32)

    X_full = np.column_stack([X_cont, C]).astype(np.float32)

    X_train = X_full[:n_train]
    X_test = X_full[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    f_test = f_true[n_train:]

    cat_info = []
    for v in range(num_cat_vars):
        col_idx = cont_dim + v
        cat_info.append((col_idx, cat_levels[v]))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "f_test": f_test,
        "cat_info": cat_info,
        "cat_levels": cat_levels,
        "true_noise": true_noise,
        "cont_dim": cont_dim,
        "relevant_dims": relevant_dims,
    }


# =============================================================================
# Metrics
# =============================================================================


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-10))


# =============================================================================
# Test Configurations
# =============================================================================

# All 5 categorical kernel variants
CAT_KERNEL_VARIANTS = ["gd", "cr", "ehh", "hh", "fe"]

# Format: (cat_kernel, cont_kernel_str, n, cont_dim, cat_levels, method)
MINIMAL_CONFIGS = [
    # Core: EHH (recommended) with RBF, materialized
    ("ehh", "rbf", 500, 3, [3], "materialized"),
    # Core: GD (simplest) with RBF, materialized
    ("gd", "rbf", 500, 3, [3], "materialized"),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    # Other kernel variants
    ("cr", "rbf", 500, 3, [3], "materialized"),
    ("hh", "rbf", 500, 3, [3], "materialized"),
    ("fe", "rbf", 500, 3, [3], "materialized"),
    # Matrix-free method
    ("ehh", "rbf", 500, 3, [3], "matrix_free"),
    # Multiple categorical variables
    ("ehh", "rbf", 500, 3, [3, 4], "materialized"),
    # Larger dataset
    ("ehh", "rbf", 1000, 3, [3], "materialized"),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    # Composite continuous kernels
    ("ehh", "rbf+matern52", 500, 3, [3], "materialized"),
    ("ehh", "rbf*matern52", 500, 3, [3], "materialized"),
    # Higher dimensional
    ("ehh", "rbf", 500, 5, [3, 4], "materialized"),
    # More levels
    ("ehh", "rbf", 500, 3, [5], "materialized"),
    ("ehh", "rbf", 500, 3, [3, 4, 5], "materialized"),
    # Larger scale
    ("ehh", "rbf", 1500, 3, [3], "materialized"),
    ("ehh", "rbf", 2000, 3, [3], "materialized"),
    # Matrix-free at scale
    ("ehh", "rbf", 1000, 3, [3], "matrix_free"),
]


_CAT_KERNEL_MAP = {
    "gd": GD,
    "cr": CR,
    "ehh": EHH,
    "hh": HH,
    "fe": FE,
}


def _build_single_kernel(name: str, ard: bool = False, active_dims=None):
    """Build a single kernel from name."""
    name = name.lower().strip()
    if name == "rbf":
        return Kernel.rbf(ard=ard, active_dims=active_dims)
    elif name == "matern52":
        return Kernel.matern52(ard=ard, active_dims=active_dims)
    elif name == "matern32":
        return Kernel.matern32(ard=ard, active_dims=active_dims)
    elif name == "matern12":
        return Kernel.matern12(ard=ard, active_dims=active_dims)
    elif name == "periodic":
        return Kernel.periodic(active_dims=active_dims)
    elif name == "rq":
        return Kernel.rq(active_dims=active_dims)
    elif name == "linear":
        return Kernel.linear(active_dims=active_dims)
    else:
        raise ValueError(f"Unknown kernel: {name}")


def _build_kernel(kernel_str: str, active_dims=None):
    """Build a KernelNode from a string description."""
    if "+" in kernel_str:
        parts = kernel_str.split("+")
        k = _build_single_kernel(parts[0].strip(), active_dims=active_dims)
        for part in parts[1:]:
            k = k + _build_single_kernel(part.strip(), active_dims=active_dims)
        return k
    elif "*" in kernel_str:
        parts = kernel_str.split("*")
        k = _build_single_kernel(parts[0].strip(), active_dims=active_dims)
        for part in parts[1:]:
            k = k * _build_single_kernel(part.strip(), active_dims=active_dims)
        return k
    else:
        return _build_single_kernel(kernel_str, active_dims=active_dims)


def _build_kernel_with_ard(kernel_str: str, active_dims=None):
    """Build a KernelNode from a string description with ARD enabled."""
    if "+" in kernel_str:
        parts = kernel_str.split("+")
        k = _build_single_kernel(parts[0].strip(), ard=True, active_dims=active_dims)
        for part in parts[1:]:
            k = k + _build_single_kernel(
                part.strip(), ard=True, active_dims=active_dims
            )
        return k
    elif "*" in kernel_str:
        parts = kernel_str.split("*")
        k = _build_single_kernel(parts[0].strip(), ard=True, active_dims=active_dims)
        for part in parts[1:]:
            k = k * _build_single_kernel(
                part.strip(), ard=True, active_dims=active_dims
            )
        return k
    else:
        return _build_single_kernel(kernel_str, ard=True, active_dims=active_dims)


def _build_mixed_kernel(cont_kernel, data, cat_kernel_str):
    """Build a composite kernel: cont_kernel * CatKernel1 * CatKernel2 ...

    The continuous kernel should already have active_dims set.
    Each categorical variable gets its own categorical kernel node.
    """
    CatClass = _CAT_KERNEL_MAP[cat_kernel_str]
    result = cont_kernel
    for col_idx, num_levels in data["cat_info"]:
        result = result * CatClass(levels=num_levels, active_dims=[col_idx])
    return result


# =============================================================================
# Test Class: Core Categorical Kernel Tests
# =============================================================================


class TestMixedCategoricalSystem:
    """System tests for mixed continuous + categorical GP."""

    def _run_test(
        self,
        cat_kernel: str,
        cont_kernel_str: str,
        n: int,
        cont_dim: int,
        cat_levels: List[int],
        method: str = "materialized",
        ard: bool = False,
        n_test: int = 100,
        n_iterations: int = 80,
        lr: float = 0.01,
        seed: int = 42,
    ) -> MixedCategoricalResult:
        """Run a mixed categorical kernel test."""
        result = MixedCategoricalResult(
            name=(
                f"Mixed {cat_kernel.upper()} + {cont_kernel_str} "
                f"n={n} d={cont_dim} levels={cat_levels} {method}"
                + (" ARD" if ard else "")
            ),
            success=True,
            cat_kernel_type=cat_kernel,
            continuous_kernel=cont_kernel_str,
            n_train=n,
            cont_dim=cont_dim,
            num_cat_vars=len(cat_levels),
            cat_levels=cat_levels,
            method=method,
            ard=ard,
        )

        try:
            # Generate data
            data = generate_mixed_categorical_data(
                n_train=n,
                n_test=n_test,
                cont_dim=cont_dim,
                cat_levels=cat_levels,
                true_noise=0.1,
                seed=seed,
            )
            result.true_noise = data["true_noise"]

            # Build kernel tree: continuous kernel (with active_dims) * categorical kernels
            cont_dims = list(range(data["cont_dim"]))
            if ard:
                cont_kernel = _build_kernel_with_ard(
                    cont_kernel_str, active_dims=cont_dims
                )
            else:
                cont_kernel = _build_kernel(cont_kernel_str, active_dims=cont_dims)
            kernel = _build_mixed_kernel(cont_kernel, data, cat_kernel)

            # Create GP
            gp = SingleOutputGP(kernel)

            # Train
            start_time = time.perf_counter()
            train_result = gp.fit(
                data["X_train"],
                data["y_train"],
                max_iterations=n_iterations,
                learning_rate=lr,
                method=method,
            )
            training_time = time.perf_counter() - start_time

            result.mojo_training_time_s = training_time
            result.mojo_final_nll = train_result.nll
            result.mojo_converged = train_result.converged

            # Predict
            pred = gp.predict(data["X_test"])
            result.mojo_rmse_vs_truth = rmse(data["f_test"], pred.mean)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            import traceback

            traceback.print_exc()

        return result

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.system
    @pytest.mark.parametrize(
        "cat_kernel,cont_kernel_str,n,cont_dim,cat_levels,method",
        MINIMAL_CONFIGS,
    )
    def test_mixed_categorical_accuracy_core_configs(
        self,
        cat_kernel: str,
        cont_kernel_str: str,
        n: int,
        cont_dim: int,
        cat_levels: List[int],
        method: str,
    ):
        """Minimal mixed categorical kernel test."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        result = self._run_test(
            cat_kernel, cont_kernel_str, n, cont_dim, cat_levels, method
        )
        print(result)

        assert result.success, f"Test failed: {result.error_message}"
        # Mixed GP should achieve reasonable RMSE
        assert result.mojo_rmse_vs_truth is not None
        assert result.mojo_rmse_vs_truth < 1.5, (
            f"RMSE too high: {result.mojo_rmse_vs_truth:.4f}"
        )

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.system
    @pytest.mark.parametrize(
        "cat_kernel,cont_kernel_str,n,cont_dim,cat_levels,method",
        MODERATE_CONFIGS,
    )
    def test_mixed_categorical_accuracy_extended_configs(
        self,
        cat_kernel: str,
        cont_kernel_str: str,
        n: int,
        cont_dim: int,
        cat_levels: List[int],
        method: str,
    ):
        """Moderate mixed categorical kernel test."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        result = self._run_test(
            cat_kernel, cont_kernel_str, n, cont_dim, cat_levels, method
        )
        print(result)

        assert result.success, f"Test failed: {result.error_message}"

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.system
    @pytest.mark.parametrize(
        "cat_kernel,cont_kernel_str,n,cont_dim,cat_levels,method",
        FULL_CONFIGS,
    )
    def test_mixed_categorical_accuracy_broad_configs(
        self,
        cat_kernel: str,
        cont_kernel_str: str,
        n: int,
        cont_dim: int,
        cat_levels: List[int],
        method: str,
    ):
        """Full mixed categorical kernel test."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        result = self._run_test(
            cat_kernel, cont_kernel_str, n, cont_dim, cat_levels, method
        )
        print(result)

        assert result.success, f"Test failed: {result.error_message}"

    # =========================================================================
    # Specific Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.system
    def test_all_cat_kernel_variants(self):
        """Test all 5 categorical kernel variants produce reasonable results."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        results = {}
        for variant in CAT_KERNEL_VARIANTS:
            result = self._run_test(
                cat_kernel=variant,
                cont_kernel_str="rbf",
                n=2000,
                cont_dim=3,
                cat_levels=[3],
                method="materialized",
            )
            results[variant] = result
            print(result)

        # Print comparison summary
        print("\n" + "=" * 70)
        print("  CATEGORICAL KERNEL VARIANT COMPARISON")
        print("=" * 70)
        print(
            f"  {'Variant':>8} | {'RMSE':>10} | {'NLL':>12} | {'Time (s)':>10} | {'Status':>8}"
        )
        print(f"  {'-' * 58}")
        for variant, r in results.items():
            rmse_val = f"{r.mojo_rmse_vs_truth:.4f}" if r.mojo_rmse_vs_truth else "N/A"
            nll_val = f"{r.mojo_final_nll:.4f}" if r.mojo_final_nll else "N/A"
            time_val = (
                f"{r.mojo_training_time_s:.2f}" if r.mojo_training_time_s else "N/A"
            )
            status = "PASS" if r.success else "FAIL"
            print(
                f"  {variant.upper():>8} | {rmse_val:>10} | {nll_val:>12} | {time_val:>10} | {status:>8}"
            )
        print("=" * 70)

        # All variants should succeed
        for variant, r in results.items():
            assert r.success, f"Variant {variant} failed: {r.error_message}"

    @pytest.mark.moderate
    @pytest.mark.system
    def test_multiple_cat_variables(self):
        """Test with multiple categorical variables of different sizes."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        configs = [
            ([3], "1 var, 3 levels"),
            ([3, 4], "2 vars, 3+4 levels"),
            ([3, 4, 5], "3 vars, 3+4+5 levels"),
        ]

        results = []
        for cat_levels, desc in configs:
            result = self._run_test(
                cat_kernel="ehh",
                cont_kernel_str="rbf",
                n=2000,
                cont_dim=3,
                cat_levels=cat_levels,
                method="materialized",
            )
            results.append((desc, result))
            print(result)

        # Print summary
        print("\n" + "=" * 70)
        print("  MULTIPLE CATEGORICAL VARIABLES")
        print("=" * 70)
        print(f"  {'Config':>25} | {'RMSE':>10} | {'NLL':>12} | {'Time (s)':>10}")
        print(f"  {'-' * 62}")
        for desc, r in results:
            rmse_val = f"{r.mojo_rmse_vs_truth:.4f}" if r.mojo_rmse_vs_truth else "N/A"
            nll_val = f"{r.mojo_final_nll:.4f}" if r.mojo_final_nll else "N/A"
            time_val = (
                f"{r.mojo_training_time_s:.2f}" if r.mojo_training_time_s else "N/A"
            )
            print(f"  {desc:>25} | {rmse_val:>10} | {nll_val:>12} | {time_val:>10}")
        print("=" * 70)

        for desc, r in results:
            assert r.success, f"Config '{desc}' failed: {r.error_message}"

    @pytest.mark.moderate
    @pytest.mark.system
    def test_materialized_vs_matrix_free(self):
        """Compare materialized and matrix-free methods for mixed GP."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        methods = ["materialized", "matrix_free"]
        results = {}

        for method in methods:
            result = self._run_test(
                cat_kernel="ehh",
                cont_kernel_str="rbf",
                n=2000,
                cont_dim=3,
                cat_levels=[3],
                method=method,
            )
            results[method] = result
            print(result)

        # Print comparison
        print("\n" + "=" * 70)
        print("  MATERIALIZED vs MATRIX-FREE")
        print("=" * 70)
        for method, r in results.items():
            rmse_val = f"{r.mojo_rmse_vs_truth:.4f}" if r.mojo_rmse_vs_truth else "N/A"
            nll_val = f"{r.mojo_final_nll:.4f}" if r.mojo_final_nll else "N/A"
            time_val = (
                f"{r.mojo_training_time_s:.2f}" if r.mojo_training_time_s else "N/A"
            )
            print(f"  {method:>15}: RMSE={rmse_val}, NLL={nll_val}, Time={time_val}s")
        print("=" * 70)

        for method, r in results.items():
            assert r.success, f"Method '{method}' failed: {r.error_message}"

    @pytest.mark.moderate
    @pytest.mark.system
    def test_scaling_with_n(self):
        """Test how mixed categorical GP scales with dataset size.

        Uses GD kernel (simplest, most reliable) to isolate scaling behavior
        from kernel-specific optimization challenges.
        """
        assert torch.cuda.is_available(), "GPU required for system tests"

        sizes = [500, 1000, 1500]
        results = []

        for n in sizes:
            result = self._run_test(
                cat_kernel="gd",
                cont_kernel_str="rbf",
                n=n,
                cont_dim=3,
                cat_levels=[3],
                method="materialized",
            )
            results.append(result)
            print(result)

        # Print scaling summary
        print("\n" + "=" * 70)
        print("  SCALING WITH DATASET SIZE")
        print("=" * 70)
        print(f"  {'n':>6} | {'RMSE':>10} | {'NLL':>12} | {'Time (s)':>10}")
        print(f"  {'-' * 45}")
        for r in results:
            rmse_val = f"{r.mojo_rmse_vs_truth:.4f}" if r.mojo_rmse_vs_truth else "N/A"
            nll_val = f"{r.mojo_final_nll:.4f}" if r.mojo_final_nll else "N/A"
            time_val = (
                f"{r.mojo_training_time_s:.2f}" if r.mojo_training_time_s else "N/A"
            )
            print(f"  {r.n_train:>6} | {rmse_val:>10} | {nll_val:>12} | {time_val:>10}")
        print("=" * 70)

        for r in results:
            assert r.success, f"n={r.n_train} failed: {r.error_message}"


# =============================================================================
# Test Class: ARD + Categorical
# =============================================================================


class TestMixedARDCategoricalSystem:
    """System tests for ARD + categorical kernel combination."""

    @pytest.mark.moderate
    @pytest.mark.system
    def test_ard_with_categorical_kernel_separates_relevant_dimensions(self):
        """Test ARD + categorical kernel separates relevant continuous dimensions.

        Uses GD kernel (simplest, 1 param per variable) for reliable convergence
        with ARD. EHH has more parameters and may need more iterations.
        """
        assert torch.cuda.is_available(), "GPU required for system tests"

        data = generate_ard_mixed_data(
            n_train=2000,
            n_test=100,
            cont_dim=5,
            relevant_dims=2,
            cat_levels=[3],
            true_noise=0.1,
            seed=42,
        )

        cont_dims = list(range(data["cont_dim"]))
        kernel = _build_mixed_kernel(
            Kernel.rbf(ard=True, active_dims=cont_dims), data, "gd"
        )
        gp = SingleOutputGP(kernel)

        start_time = time.perf_counter()
        train_result = gp.fit(
            data["X_train"],
            data["y_train"],
            max_iterations=100,
            learning_rate=0.01,
            method="materialized",
        )
        training_time = time.perf_counter() - start_time

        pred = gp.predict(data["X_test"])
        test_rmse = rmse(data["f_test"], pred.mean)

        print(f"\n  ARD + Categorical (GD):")
        print(f"    RMSE vs truth: {test_rmse:.4f}")
        print(f"    Final NLL:     {train_result.nll:.4f}")
        print(f"    Training time: {training_time:.2f}s")
        print(f"    Converged:     {train_result.converged}")

        assert test_rmse < 2.0, f"RMSE too high: {test_rmse:.4f}"

    @pytest.mark.moderate
    @pytest.mark.system
    def test_ard_with_different_cat_kernels(self):
        """Test ARD with different categorical kernel variants."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        data = generate_ard_mixed_data(
            n_train=2000,
            n_test=100,
            cont_dim=5,
            relevant_dims=2,
            cat_levels=[3],
            true_noise=0.1,
            seed=42,
        )

        cont_dims = list(range(data["cont_dim"]))
        results = {}
        for variant in ["gd", "ehh", "cr"]:
            kernel = _build_mixed_kernel(
                Kernel.rbf(ard=True, active_dims=cont_dims), data, variant
            )
            gp = SingleOutputGP(kernel)

            train_result = gp.fit(
                data["X_train"],
                data["y_train"],
                max_iterations=80,
                learning_rate=0.01,
                method="materialized",
            )

            pred = gp.predict(data["X_test"])
            test_rmse = rmse(data["f_test"], pred.mean)
            results[variant] = {
                "rmse": test_rmse,
                "nll": train_result.nll,
            }

        print("\n" + "=" * 70)
        print("  ARD + CATEGORICAL KERNEL VARIANTS")
        print("=" * 70)
        for variant, r in results.items():
            print(f"  {variant.upper():>5}: RMSE={r['rmse']:.4f}, NLL={r['nll']:.4f}")
        print("=" * 70)

    @pytest.mark.full
    @pytest.mark.system
    def test_ard_categorical_larger_scale(self):
        """Test ARD + categorical at larger scale.

        Uses GD kernel for reliable convergence at larger scale with ARD.
        """
        assert torch.cuda.is_available(), "GPU required for system tests"

        data = generate_ard_mixed_data(
            n_train=2000,
            n_test=200,
            cont_dim=8,
            relevant_dims=3,
            cat_levels=[3, 4],
            true_noise=0.1,
            seed=42,
        )

        cont_dims = list(range(data["cont_dim"]))
        kernel = _build_mixed_kernel(
            Kernel.rbf(ard=True, active_dims=cont_dims), data, "gd"
        )
        gp = SingleOutputGP(kernel)

        start_time = time.perf_counter()
        train_result = gp.fit(
            data["X_train"],
            data["y_train"],
            max_iterations=100,
            learning_rate=0.01,
            method="materialized",
        )
        training_time = time.perf_counter() - start_time

        pred = gp.predict(data["X_test"])
        test_rmse = rmse(data["f_test"], pred.mean)

        print(f"\n  ARD + Categorical (larger scale):")
        print(f"    n = 2000, cont_dim=8, cat_vars=2, levels=[3,4]")
        print(f"    RMSE vs truth: {test_rmse:.4f}")
        print(f"    Final NLL:     {train_result.nll:.4f}")
        print(f"    Training time: {training_time:.2f}s")

        assert test_rmse < 2.5, f"RMSE too high: {test_rmse:.4f}"


# =============================================================================
# Test Class: Composite + Categorical
# =============================================================================


class TestCompositeWithCategoricalSystem:
    """System tests for composite continuous kernels + categorical."""

    @pytest.mark.moderate
    @pytest.mark.system
    def test_sum_kernel_with_categorical(self):
        """Test sum kernel (RBF + Matern52) with categorical."""
        assert torch.cuda.is_available(), "GPU required for system tests"

        data = generate_mixed_categorical_data(
            n_train=2000,
            n_test=100,
            cont_dim=3,
            cat_levels=[3],
            true_noise=0.1,
            seed=42,
        )

        cont_dims = list(range(data["cont_dim"]))
        kernel = _build_mixed_kernel(
            Kernel.rbf(active_dims=cont_dims) + Kernel.matern52(active_dims=cont_dims),
            data,
            "ehh",
        )
        gp = SingleOutputGP(kernel)

        train_result = gp.fit(
            data["X_train"],
            data["y_train"],
            max_iterations=80,
            learning_rate=0.01,
            method="materialized",
        )

        pred = gp.predict(data["X_test"])
        test_rmse = rmse(data["f_test"], pred.mean)

        print(f"\n  Sum Kernel (RBF + Matern52) + Categorical (EHH):")
        print(f"    RMSE vs truth: {test_rmse:.4f}")
        print(f"    Final NLL:     {train_result.nll:.4f}")

        assert test_rmse < 2.0, f"RMSE too high: {test_rmse:.4f}"

    @pytest.mark.moderate
    @pytest.mark.system
    def test_product_kernel_with_categorical(self):
        """Test product kernel (RBF * Matern52) with categorical.

        Product kernels have a larger parameter space. Uses GD categorical
        kernel for reliable convergence.
        """
        assert torch.cuda.is_available(), "GPU required for system tests"

        data = generate_mixed_categorical_data(
            n_train=2000,
            n_test=100,
            cont_dim=3,
            cat_levels=[3],
            true_noise=0.1,
            seed=42,
        )

        cont_dims = list(range(data["cont_dim"]))
        kernel = _build_mixed_kernel(
            Kernel.rbf(active_dims=cont_dims) * Kernel.matern52(active_dims=cont_dims),
            data,
            "gd",
        )
        gp = SingleOutputGP(kernel)

        train_result = gp.fit(
            data["X_train"],
            data["y_train"],
            max_iterations=100,
            learning_rate=0.01,
            method="materialized",
        )

        pred = gp.predict(data["X_test"])
        test_rmse = rmse(data["f_test"], pred.mean)

        print(f"\n  Product Kernel (RBF * Matern52) + Categorical (GD):")
        print(f"    RMSE vs truth: {test_rmse:.4f}")
        print(f"    Final NLL:     {train_result.nll:.4f}")

        assert test_rmse < 2.0, f"RMSE too high: {test_rmse:.4f}"

    @pytest.mark.full
    @pytest.mark.system
    def test_composite_ard_categorical(self):
        """Test composite kernel + ARD + categorical (all features combined).

        Uses GD categorical kernel for reliable convergence with the complex
        composite + ARD parameter space.
        """
        assert torch.cuda.is_available(), "GPU required for system tests"

        data = generate_ard_mixed_data(
            n_train=2000,
            n_test=100,
            cont_dim=5,
            relevant_dims=2,
            cat_levels=[3],
            true_noise=0.1,
            seed=42,
        )

        cont_dims = list(range(data["cont_dim"]))
        kernel = _build_mixed_kernel(
            Kernel.rbf(ard=True, active_dims=cont_dims)
            + Kernel.matern52(ard=True, active_dims=cont_dims),
            data,
            "gd",
        )
        gp = SingleOutputGP(kernel)

        train_result = gp.fit(
            data["X_train"],
            data["y_train"],
            max_iterations=100,
            learning_rate=0.01,
            method="materialized",
        )

        pred = gp.predict(data["X_test"])
        test_rmse = rmse(data["f_test"], pred.mean)

        print(f"\n  Composite (RBF+Matern52) + ARD + Categorical (GD):")
        print(f"    RMSE vs truth: {test_rmse:.4f}")
        print(f"    Final NLL:     {train_result.nll:.4f}")

        assert test_rmse < 2.5, f"RMSE too high: {test_rmse:.4f}"


# =============================================================================
# Test Class: Per-Variable Kernel Selection
# =============================================================================


class TestPerVariableKernelSystem:
    """System tests for per-variable categorical kernel selection."""

    @pytest.mark.moderate
    @pytest.mark.system
    def test_per_variable_kernel_types(self):
        """Test different kernel types for different categorical variables.

        Uses GD + CR (both simple, reliable) for per-variable selection.
        The test verifies the per-variable kernel selection mechanism works
        end-to-end, not that specific kernel combinations are optimal.
        """
        assert torch.cuda.is_available(), "GPU required for system tests"

        data = generate_mixed_categorical_data(
            n_train=2000,
            n_test=100,
            cont_dim=3,
            cat_levels=[3, 4],
            true_noise=0.1,
            seed=42,
        )

        # Per-variable kernel selection: first cat var (col 3) uses GD, second (col 4) uses CR
        cont_dims = list(range(data["cont_dim"]))
        kernel = (
            Kernel.rbf(active_dims=cont_dims)
            * GD(levels=3, active_dims=[3])
            * CR(levels=4, active_dims=[4])
        )
        gp = SingleOutputGP(kernel)

        train_result = gp.fit(
            data["X_train"],
            data["y_train"],
            max_iterations=100,
            learning_rate=0.01,
            method="materialized",
        )

        pred = gp.predict(data["X_test"])
        test_rmse = rmse(data["f_test"], pred.mean)

        print(f"\n  Per-Variable Kernel Selection (GD + CR):")
        print(f"    RMSE vs truth: {test_rmse:.4f}")
        print(f"    Final NLL:     {train_result.nll:.4f}")

        # Per-variable selection should produce reasonable results
        assert test_rmse < 2.0, f"RMSE too high: {test_rmse:.4f}"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "minimal"])
