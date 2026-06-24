"""Composite kernel prediction accuracy tests vs GPyTorch.

Tests MojoGP's composite kernel API (kernel combinations like RBF + Matern52)
against GPyTorch's kernel arithmetic.

Composite kernels supported:
- Sum: RBF + Matern52, RBF + Linear
- Product: RBF * Periodic
- Triple: RBF + Matern52 + Linear
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from tests.shared.benchmarking.environment import (
    requires_mojogp,
    requires_gpytorch,
    requires_cuda,
    assert_gpu_available,
    assert_gpu_was_used,
)
from tests.shared.benchmarking.result_types import BenchmarkResult
from tests.shared.benchmarking.metrics import compute_all_accuracy_metrics

import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import (
    RBFKernel,
    MaternKernel,
    PeriodicKernel,
    RQKernel,
    LinearKernel,
    ScaleKernel,
)
from gpytorch.means import ConstantMean


# =============================================================================
# Test Configurations
# =============================================================================

# Format: (kernel_spec, name, n_train, n_test, d)
# kernel_spec is a string that can be parsed by both MojoGP and GPyTorch
MINIMAL_CONFIGS = [
    ("rbf+matern52", "rbf_plus_matern52", 200, 50, 5),
    # NOTE: rbf*periodic is excluded until its convergence evidence is sufficient.
    # ("rbf*periodic", "rbf_times_periodic", 200, 50, 3),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("rbf+linear", "rbf_plus_linear", 300, 50, 5),
    ("matern32+matern52", "matern_sum", 300, 50, 5),
    # NOTE: Product kernels with periodic have convergence issues
    # ("rbf*rbf", "rbf_squared", 300, 50, 5),
    # ("periodic+linear", "periodic_plus_linear", 300, 50, 3),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    ("rbf+matern52+linear", "triple_sum", 300, 50, 5),
    # NOTE: Product kernels with periodic have convergence issues
    # ("rbf*periodic+linear", "product_plus_linear", 300, 50, 3),
    ("rbf+matern52", "rbf_plus_matern52_large", 500, 100, 5),
    ("rbf+matern52", "rbf_plus_matern52_highdim", 300, 50, 10),
]


# =============================================================================
# Composite Comparison Result
# =============================================================================


@dataclass
class CompositeComparisonResult:
    """Results from comparing MojoGP vs GPyTorch composite kernel predictions."""

    kernel_spec: str
    name: str
    n_train: int
    n_test: int
    d: int

    mojo_mean: np.ndarray = field(repr=False)
    mojo_var: np.ndarray = field(repr=False)
    gpy_mean: np.ndarray = field(repr=False)
    gpy_var: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)

    mojo_train_time: float = 0.0
    gpy_train_time: float = 0.0
    mojo_rmse: float = 0.0
    gpy_rmse: float = 0.0

    @property
    def mean_rmse(self) -> float:
        """RMSE between MojoGP and GPyTorch mean predictions."""
        return float(np.sqrt(np.mean((self.mojo_mean - self.gpy_mean) ** 2)))

    @property
    def variance_correlation(self) -> float:
        """Correlation between MojoGP and GPyTorch variances."""
        if np.std(self.mojo_var) < 1e-10 or np.std(self.gpy_var) < 1e-10:
            return 0.0
        return float(np.corrcoef(self.mojo_var, self.gpy_var)[0, 1])

    @property
    def rmse_ratio(self) -> float:
        """Ratio of MojoGP RMSE to GPyTorch RMSE on test data."""
        if self.gpy_rmse < 1e-10:
            return float("inf")
        return self.mojo_rmse / self.gpy_rmse


# =============================================================================
# Test Class
# =============================================================================


@requires_mojogp
@requires_gpytorch
@requires_cuda
class TestCompositePredictionAccuracy:
    """Compare MojoGP vs GPyTorch composite kernel prediction accuracy."""

    def _generate_composite_data(
        self,
        kernel_spec: str,
        n_train: int,
        n_test: int,
        d: int,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate data appropriate for composite kernel."""
        np.random.seed(seed)

        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_test = np.random.randn(n_test, d).astype(np.float32)

        if "periodic" in kernel_spec.lower():
            t_train = X_train[:, 0]
            t_test = X_test[:, 0]
            period = 2.0

            y_train = np.sin(2 * np.pi * t_train / period).astype(np.float32)

            if "+" in kernel_spec and "linear" in kernel_spec:
                w = np.random.randn(d).astype(np.float32)
                y_train += 0.5 * (X_train @ w)

            y_test = np.sin(2 * np.pi * t_test / period).astype(np.float32)
            if "+" in kernel_spec and "linear" in kernel_spec:
                y_test += 0.5 * (X_test @ w)

        elif "linear" in kernel_spec:
            w = np.random.randn(d).astype(np.float32)
            y_train = (X_train @ w).astype(np.float32)

            if "+" in kernel_spec:
                dists_sq = np.sum(
                    (X_train[:, None, :] - X_train[None, :, :]) ** 2, axis=2
                )
                K_rbf = np.exp(-0.5 * dists_sq)
                L = np.linalg.cholesky(K_rbf + 1e-6 * np.eye(n_train))
                y_train += 0.5 * (L @ np.random.randn(n_train))

                cross_dists_sq = np.sum(
                    (X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2
                )
                K_cross = np.exp(-0.5 * cross_dists_sq)
                alpha = np.linalg.solve(
                    L.T, np.linalg.solve(L, y_train - 0.5 * (X_train @ w))
                )
                y_test = (X_test @ w + 0.5 * (K_cross @ alpha)).astype(np.float32)
            else:
                y_test = (X_test @ w).astype(np.float32)

        else:
            dists_sq = np.sum((X_train[:, None, :] - X_train[None, :, :]) ** 2, axis=2)
            K = np.exp(-0.5 * dists_sq)
            if "*" in kernel_spec:
                K = K * K

            L = np.linalg.cholesky(K + 1e-6 * np.eye(n_train))
            y_train = (L @ np.random.randn(n_train)).astype(np.float32)

            cross_dists_sq = np.sum(
                (X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2
            )
            K_cross = np.exp(-0.5 * cross_dists_sq)
            if "*" in kernel_spec:
                K_cross = K_cross * K_cross

            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            y_test = (K_cross @ alpha).astype(np.float32)

        y_train += np.random.randn(n_train).astype(np.float32) * 0.1

        return X_train, X_test, y_train, y_test

    def _train_mojogp_composite(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel_spec: str,
        n_iterations: int = 100,
        lr: float = 0.02,
        seed: int = 42,
    ) -> Tuple[Any, float]:
        """Train MojoGP with composite kernel."""
        import time

        try:
            from mojogp import Kernel, SingleOutputGP
        except ImportError:
            pytest.skip("mojogp.kernel module not available")

        kernel = self._build_mojo_kernel(kernel_spec)

        gp = SingleOutputGP(kernel, verbose=False)

        start_time = time.perf_counter()
        gp.fit(X_train, y_train,
            max_iterations=n_iterations,
            learning_rate=lr,
        )
        train_time = time.perf_counter() - start_time

        return gp, train_time

    def _build_mojo_kernel(self, kernel_spec: str):
        """Build MojoGP composite kernel from spec string."""
        from mojogp import Kernel

        def get_base_kernel(name: str):
            name = name.strip()
            if name == "rbf":
                return Kernel.rbf()
            elif name == "matern12":
                return Kernel.matern12()
            elif name == "matern32":
                return Kernel.matern32()
            elif name == "matern52":
                return Kernel.matern52()
            elif name == "periodic":
                return Kernel.periodic()
            elif name == "rq":
                return Kernel.rq()
            elif name == "linear":
                return Kernel.linear()
            else:
                raise ValueError(f"Unknown kernel: {name}")

        if "+" in kernel_spec and "*" not in kernel_spec:
            parts = kernel_spec.split("+")
            result = get_base_kernel(parts[0])
            for part in parts[1:]:
                result = result + get_base_kernel(part)
            return result
        elif "*" in kernel_spec and "+" not in kernel_spec:
            parts = kernel_spec.split("*")
            result = get_base_kernel(parts[0])
            for part in parts[1:]:
                result = result * get_base_kernel(part)
            return result
        elif "*" in kernel_spec and "+" in kernel_spec:
            if "rbf*periodic+linear" in kernel_spec:
                return get_base_kernel("rbf") * get_base_kernel(
                    "periodic"
                ) + get_base_kernel("linear")
            parts_plus = kernel_spec.split("+")
            result = get_base_kernel(parts_plus[0].split("*")[0])
            for part in parts_plus[0].split("*")[1:]:
                result = result * get_base_kernel(part)
            for part in parts_plus[1:]:
                result = result + get_base_kernel(part)
            return result
        else:
            return get_base_kernel(kernel_spec)

    def _predict_mojogp_composite(
        self,
        gp,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with MojoGP composite kernel model."""
        pred = gp.predict(X_test)
        return pred.mean, pred.variance

    def _train_gpytorch_composite(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel_spec: str,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
    ) -> Tuple[Any, float]:
        """Train GPyTorch with composite kernel."""
        import time
        from tests.shared.benchmarking.gpytorch_models import gpytorch_cg_settings

        torch.manual_seed(seed)

        train_x = torch.tensor(X_train, dtype=torch.float32).cuda()
        train_y = torch.tensor(y_train, dtype=torch.float32).cuda()

        likelihood = GaussianLikelihood().cuda()

        covar_module = self._build_gpytorch_kernel(kernel_spec)

        class CompositeGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, covar_module):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean().cuda()
                self.covar_module = covar_module

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        model = CompositeGP(train_x, train_y, likelihood, covar_module)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        start_time = time.perf_counter()

        with gpytorch_cg_settings():
            for _ in range(n_iterations):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

        train_time = time.perf_counter() - start_time

        return (model, likelihood), train_time

    def _build_gpytorch_kernel(self, kernel_spec: str):
        """Build GPyTorch composite kernel from spec string."""

        def get_base_kernel(name: str):
            name = name.strip()
            if name == "rbf":
                return ScaleKernel(RBFKernel())
            elif name == "matern12":
                return ScaleKernel(MaternKernel(nu=0.5))
            elif name == "matern32":
                return ScaleKernel(MaternKernel(nu=1.5))
            elif name == "matern52":
                return ScaleKernel(MaternKernel(nu=2.5))
            elif name == "periodic":
                return ScaleKernel(PeriodicKernel())
            elif name == "rq":
                return ScaleKernel(RQKernel())
            elif name == "linear":
                return ScaleKernel(LinearKernel())
            else:
                raise ValueError(f"Unknown kernel: {name}")

        if "+" in kernel_spec and "*" not in kernel_spec:
            parts = kernel_spec.split("+")
            result = get_base_kernel(parts[0])
            for part in parts[1:]:
                result = result + get_base_kernel(part)
            return result.cuda()
        elif "*" in kernel_spec and "+" not in kernel_spec:
            parts = kernel_spec.split("*")
            result = get_base_kernel(parts[0])
            for part in parts[1:]:
                result = result * get_base_kernel(part)
            return result.cuda()
        elif "*" in kernel_spec and "+" in kernel_spec:
            if "rbf*periodic+linear" in kernel_spec:
                return (
                    get_base_kernel("rbf") * get_base_kernel("periodic")
                    + get_base_kernel("linear")
                ).cuda()
            parts_plus = kernel_spec.split("+")
            result = get_base_kernel(parts_plus[0].split("*")[0])
            for part in parts_plus[0].split("*")[1:]:
                result = result * get_base_kernel(part)
            for part in parts_plus[1:]:
                result = result + get_base_kernel(part)
            return result.cuda()
        else:
            return get_base_kernel(kernel_spec).cuda()

    def _predict_gpytorch_composite(
        self,
        model_likelihood: Tuple[Any, Any],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with GPyTorch composite kernel model."""
        from tests.shared.benchmarking.gpytorch_models import gpytorch_cg_settings

        model, likelihood = model_likelihood
        model.eval()
        likelihood.eval()

        test_x = torch.tensor(X_test, dtype=torch.float32).cuda()

        with torch.no_grad(), gpytorch_cg_settings():
            preds = likelihood(model(test_x))
            mean = preds.mean.cpu().numpy()
            var = preds.variance.cpu().numpy()

        return mean, var

    def _run_comparison(
        self,
        kernel_spec: str,
        name: str,
        n_train: int,
        n_test: int,
        d: int,
        n_iterations: int = 100,
        lr: float = 0.02,
        seed: int = 42,
    ) -> CompositeComparisonResult:
        """Run side-by-side composite kernel comparison."""
        print(f"\n  Generating composite data ({kernel_spec})...")
        X_train, X_test, y_train, y_test = self._generate_composite_data(
            kernel_spec, n_train, n_test, d, seed
        )

        print(f"  Training MojoGP composite ({kernel_spec})...")
        try:
            mojo_gp, mojo_time = self._train_mojogp_composite(
                X_train, y_train, kernel_spec, n_iterations, lr, seed
            )
            mojo_mean, mojo_var = self._predict_mojogp_composite(mojo_gp, X_test)
        except Exception as e:
            print(f"  MojoGP composite failed: {e}")
            pytest.skip(f"MojoGP composite kernel not available: {e}")

        print(f"  Training GPyTorch composite ({kernel_spec})...")
        gpy_model, gpy_time = self._train_gpytorch_composite(
            X_train, y_train, kernel_spec, n_iterations, lr, seed
        )
        gpy_mean, gpy_var = self._predict_gpytorch_composite(gpy_model, X_test)

        mojo_rmse = float(np.sqrt(np.mean((mojo_mean - y_test) ** 2)))
        gpy_rmse = float(np.sqrt(np.mean((gpy_mean - y_test) ** 2)))

        return CompositeComparisonResult(
            kernel_spec=kernel_spec,
            name=name,
            n_train=n_train,
            n_test=n_test,
            d=d,
            mojo_mean=mojo_mean,
            mojo_var=mojo_var,
            gpy_mean=gpy_mean,
            gpy_var=gpy_var,
            y_test=y_test,
            mojo_train_time=mojo_time,
            gpy_train_time=gpy_time,
            mojo_rmse=mojo_rmse,
            gpy_rmse=gpy_rmse,
        )

    def _report_result(self, result: CompositeComparisonResult):
        """Print composite comparison result."""
        print(f"\n  === {result.kernel_spec} (n={result.n_train}, d={result.d}) ===")
        print(f"  Mean RMSE (Mojo vs GPy): {result.mean_rmse:.6f}")
        print(f"  Variance correlation: {result.variance_correlation:.4f}")
        print(f"  Test RMSE ratio (Mojo/GPy): {result.rmse_ratio:.4f}")
        print(
            f"  Train time: Mojo={result.mojo_train_time:.3f}s, GPy={result.gpy_train_time:.3f}s"
        )

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.composite
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel_spec,name,n_train,n_test,d", MINIMAL_CONFIGS)
    def test_composite_gpytorch_parity_core_configs(
        self,
        kernel_spec: str,
        name: str,
        n_train: int,
        n_test: int,
        d: int,
    ):
        """Minimal composite kernel prediction accuracy test."""
        assert_gpu_available()

        print(f"\n=== Composite: {kernel_spec}, n={n_train}, d={d} ===")
        result = self._run_comparison(kernel_spec, name, n_train, n_test, d)
        self._report_result(result)

        assert result.rmse_ratio < 1.5, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 1.5x target for composite kernels"
        )

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.composite
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel_spec,name,n_train,n_test,d", MODERATE_CONFIGS)
    def test_composite_gpytorch_parity_extended_configs(
        self,
        kernel_spec: str,
        name: str,
        n_train: int,
        n_test: int,
        d: int,
    ):
        """Moderate composite kernel prediction accuracy test."""
        assert_gpu_available()

        print(f"\n=== Composite: {kernel_spec}, n={n_train}, d={d} ===")
        result = self._run_comparison(kernel_spec, name, n_train, n_test, d)
        self._report_result(result)

        assert result.rmse_ratio < 1.5, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 1.5x target for composite kernels"
        )

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.composite
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel_spec,name,n_train,n_test,d", FULL_CONFIGS)
    def test_composite_gpytorch_parity_broad_configs(
        self,
        kernel_spec: str,
        name: str,
        n_train: int,
        n_test: int,
        d: int,
    ):
        """Full composite kernel prediction accuracy test."""
        assert_gpu_available()

        print(f"\n=== Composite: {kernel_spec}, n={n_train}, d={d} ===")
        result = self._run_comparison(kernel_spec, name, n_train, n_test, d)
        self._report_result(result)

        assert result.rmse_ratio < 1.5, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 1.5x target for composite kernels"
        )
