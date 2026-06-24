"""Multi-output GP prediction accuracy tests vs GPyTorch.

Tests MojoGP's multi-output GP capability against GPyTorch's MultitaskKernel
with known task covariance structure.

Also includes multi-output + ARD combinations.
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
from tests.shared.benchmarking.mojogp_runners import train_mojogp_multi_output, predict_mojogp_multi_output

import torch
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, MultitaskKernel
from gpytorch.means import ConstantMean, MultitaskMean


# =============================================================================
# Test Configurations
# =============================================================================

# Format: (kernel, n_train, n_test, d, num_tasks, task_correlation)
MINIMAL_CONFIGS = [
    ("rbf", 200, 50, 5, 2, "medium"),
    ("matern52", 200, 50, 5, 3, "medium"),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("rbf", 300, 50, 5, 2, "high"),
    ("rbf", 300, 50, 5, 2, "low"),
    ("rbf", 300, 50, 5, 4, "medium"),
    ("matern52", 300, 50, 5, 3, "high"),
    ("matern32", 300, 50, 5, 3, "medium"),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    ("rbf", 300, 50, 5, 5, "medium"),
    ("rbf", 300, 50, 5, 8, "medium"),
    ("rbf", 300, 50, 5, 3, "independent"),
    ("rbf", 500, 100, 5, 3, "medium"),
    ("rbf", 300, 50, 10, 3, "medium"),
    ("matern12", 300, 50, 5, 3, "medium"),
    ("matern52", 500, 100, 5, 3, "medium"),
]

# Larger predictive-parity rows that are feasible locally, but should still be
# treated as fallback multi-output comparisons rather than CG-vs-CG benchmarks.
PUBLISHABLE_CONFIGS = [
    ("rbf", 2000, 100, 5, 2, "medium"),
]

# Multi-output + ARD configurations
# Format: (kernel, n_train, n_test, d, num_tasks, relevant_dims)
ARD_CONFIGS = [
    ("rbf", 200, 50, 5, 2, 2),
    ("matern52", 200, 50, 5, 3, 2),
]


# =============================================================================
# Multi-Output Comparison Result
# =============================================================================


@dataclass
class MultiOutputComparisonResult:
    """Results from comparing MojoGP vs GPyTorch multi-output predictions."""

    kernel: str
    n_train: int
    n_test: int
    d: int
    num_tasks: int

    mojo_mean: np.ndarray = field(repr=False)  # [n_test, num_tasks]
    mojo_var: np.ndarray = field(repr=False)
    gpy_mean: np.ndarray = field(repr=False)
    gpy_var: np.ndarray = field(repr=False)
    Y_test: np.ndarray = field(repr=False)

    mojo_B: np.ndarray = field(repr=False)  # Task covariance [T, T]
    gpy_B: np.ndarray = field(repr=False)

    mojo_train_time: float = 0.0
    gpy_train_time: float = 0.0
    comparison_class: str = "diagnostic_small_n"
    fairness_note: str = (
        "Diagnostic multi-output predictive-parity row: current GPyTorch multitask exact "
        "comparators are useful for prediction checks, but are not treated as fair CG-vs-CG "
        "solver parity baselines."
    )

    @property
    def mean_rmse(self) -> float:
        """RMSE between MojoGP and GPyTorch mean predictions."""
        return float(np.sqrt(np.mean((self.mojo_mean - self.gpy_mean) ** 2)))

    @property
    def per_task_rmse(self) -> np.ndarray:
        """RMSE per task between MojoGP and GPyTorch."""
        return np.sqrt(np.mean((self.mojo_mean - self.gpy_mean) ** 2, axis=0))

    @property
    def task_covariance_correlation(self) -> float:
        """Correlation between MojoGP and GPyTorch task covariances."""
        mojo_B_flat = self.mojo_B.flatten()
        gpy_B_flat = self.gpy_B.flatten()
        if np.std(mojo_B_flat) < 1e-10 or np.std(gpy_B_flat) < 1e-10:
            return 0.0
        return float(np.corrcoef(mojo_B_flat, gpy_B_flat)[0, 1])

    @property
    def mojo_rmse(self) -> float:
        """RMSE of MojoGP predictions on test data."""
        return float(np.sqrt(np.mean((self.mojo_mean - self.Y_test) ** 2)))

    @property
    def gpy_rmse(self) -> float:
        """RMSE of GPyTorch predictions on test data."""
        return float(np.sqrt(np.mean((self.gpy_mean - self.Y_test) ** 2)))

    @property
    def rmse_ratio(self) -> float:
        """Ratio of MojoGP RMSE to GPyTorch RMSE."""
        if self.gpy_rmse < 1e-10:
            return float("inf")
        return self.mojo_rmse / self.gpy_rmse


def _comparison_metadata(n_train: int) -> tuple[str, str]:
    if n_train < 2000:
        return (
            "diagnostic_small_n",
            "Diagnostic small-n predictive-parity row: current multitask GPyTorch checks "
            "prediction agreement on modest problems only and should not be used as a headline "
            "performance or CG-vs-CG fairness claim.",
        )
    return (
        "fallback_match",
        "Predictive-parity multi-output row: the comparator is GPyTorch multitask exact, "
        "but the current reference path is treated as a different solver class rather than "
        "a fair CG-vs-CG baseline.",
    )


# =============================================================================
# Test Class
# =============================================================================


@requires_mojogp
@requires_gpytorch
@requires_cuda
class TestMultiOutputPredictionAccuracy:
    """Compare MojoGP vs GPyTorch multi-output prediction accuracy."""

    def _generate_multi_output_data(
        self,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        kernel: str,
        task_correlation: str,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate multi-output GP data with task correlation."""
        np.random.seed(seed)

        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_test = np.random.randn(n_test, d).astype(np.float32)

        if task_correlation == "high":
            corr = 0.9
        elif task_correlation == "medium":
            corr = 0.5
        elif task_correlation == "low":
            corr = 0.1
        else:
            corr = 0.0

        B = np.eye(num_tasks) * (1 - corr) + np.ones((num_tasks, num_tasks)) * corr
        B = B.astype(np.float32)

        L_B = np.linalg.cholesky(B)

        X_train_scaled = X_train / 1.0
        dists_sq = np.sum(
            (X_train_scaled[:, None, :] - X_train_scaled[None, :, :]) ** 2, axis=2
        )

        if kernel == "rbf":
            K = np.exp(-0.5 * dists_sq)
        elif kernel == "matern12":
            dists = np.sqrt(np.maximum(dists_sq, 1e-10))
            K = np.exp(-dists)
        elif kernel == "matern32":
            dists = np.sqrt(np.maximum(dists_sq, 1e-10))
            K = (1 + np.sqrt(3) * dists) * np.exp(-np.sqrt(3) * dists)
        elif kernel == "matern52":
            dists = np.sqrt(np.maximum(dists_sq, 1e-10))
            K = (1 + np.sqrt(5) * dists + 5.0 / 3.0 * dists_sq) * np.exp(
                -np.sqrt(5) * dists
            )
        else:
            K = np.exp(-0.5 * dists_sq)

        K += 1e-6 * np.eye(n_train)
        L_K = np.linalg.cholesky(K)

        f_train = L_K @ np.random.randn(n_train, num_tasks)
        Y_train = (f_train @ L_B.T + np.random.randn(n_train, num_tasks) * 0.1).astype(
            np.float32
        )

        X_test_scaled = X_test / 1.0
        cross_dists_sq = np.sum(
            (X_test_scaled[:, None, :] - X_train_scaled[None, :, :]) ** 2, axis=2
        )

        if kernel == "rbf":
            K_cross = np.exp(-0.5 * cross_dists_sq)
        elif kernel == "matern12":
            cross_dists = np.sqrt(np.maximum(cross_dists_sq, 1e-10))
            K_cross = np.exp(-cross_dists)
        elif kernel == "matern32":
            cross_dists = np.sqrt(np.maximum(cross_dists_sq, 1e-10))
            K_cross = (1 + np.sqrt(3) * cross_dists) * np.exp(-np.sqrt(3) * cross_dists)
        elif kernel == "matern52":
            cross_dists = np.sqrt(np.maximum(cross_dists_sq, 1e-10))
            K_cross = (
                1 + np.sqrt(5) * cross_dists + 5.0 / 3.0 * cross_dists_sq
            ) * np.exp(-np.sqrt(5) * cross_dists)
        else:
            K_cross = np.exp(-0.5 * cross_dists_sq)

        alpha = np.linalg.solve(L_K.T, np.linalg.solve(L_K, f_train))
        f_test = K_cross @ alpha
        Y_test = f_test.astype(np.float32)

        return X_train, X_test, Y_train, Y_test, B

    def _train_mojogp_multi_output(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        kernel: str,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
    ) -> Tuple[Dict[str, Any], np.ndarray, float]:
        """Train MojoGP multi-output model."""
        import time

        start_time = time.perf_counter()

        result = train_mojogp_multi_output(
            X_train,
            Y_train,
            kernel,
            n_iterations=n_iterations,
            lr=lr,
        )

        train_time = time.perf_counter() - start_time

        B = np.array(
            result.get("learned_params", {}).get("B", np.eye(Y_train.shape[1])),
            dtype=np.float32,
        )

        return result, B, train_time

    def _predict_mojogp_multi_output(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        result: Dict[str, Any],
        kernel: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with MojoGP multi-output model."""
        pred = predict_mojogp_multi_output(
            X_train,
            Y_train,
            X_test,
            result,
            kernel,
        )

        mean = pred["mean"]
        variance = pred["variance"]

        return mean, variance

    def _train_gpytorch_multi_output(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        kernel: str,
        num_tasks: int,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
    ) -> Tuple[Any, np.ndarray, float]:
        """Train GPyTorch multi-output model."""
        import time
        from tests.shared.benchmarking.gpytorch_models import gpytorch_cg_settings

        torch.manual_seed(seed)

        train_x = torch.tensor(X_train, dtype=torch.float32).cuda()
        train_y = torch.tensor(Y_train, dtype=torch.float32).cuda()

        likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks).cuda()

        class MultiOutputGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel, num_tasks):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = MultitaskMean(
                    ConstantMean(), num_tasks=num_tasks
                ).cuda()

                if kernel == "rbf":
                    base_kernel = RBFKernel()
                elif kernel == "matern12":
                    base_kernel = MaternKernel(nu=0.5)
                elif kernel == "matern32":
                    base_kernel = MaternKernel(nu=1.5)
                elif kernel == "matern52":
                    base_kernel = MaternKernel(nu=2.5)
                else:
                    base_kernel = RBFKernel()

                self.covar_module = MultitaskKernel(
                    base_kernel, num_tasks=num_tasks
                ).cuda()

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(
                    mean_x, covar_x
                )

        model = MultiOutputGP(train_x, train_y, likelihood, kernel, num_tasks)

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

        task_covar = model.covar_module.task_covar_module.covar_matrix.to_dense()
        B = task_covar.detach().cpu().numpy()

        return (model, likelihood), B, train_time

    def _predict_gpytorch_multi_output(
        self,
        model_likelihood: Tuple[Any, Any],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with GPyTorch multi-output model."""
        from tests.shared.benchmarking.gpytorch_models import gpytorch_cg_settings

        model, likelihood = model_likelihood
        model.eval()
        likelihood.eval()

        test_x = torch.tensor(X_test, dtype=torch.float32).cuda()

        with torch.no_grad(), gpytorch_cg_settings():
            preds = likelihood(model(test_x))
            mean = preds.mean.cpu().numpy()
            variance = preds.variance.cpu().numpy()

        return mean, variance

    def _run_comparison(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_correlation: str,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
    ) -> MultiOutputComparisonResult:
        """Run side-by-side multi-output comparison."""
        print(
            f"\n  Generating multi-output data (T={num_tasks}, corr={task_correlation})..."
        )
        X_train, X_test, Y_train, Y_test, true_B = self._generate_multi_output_data(
            n_train, n_test, d, num_tasks, kernel, task_correlation, seed
        )

        print(f"  Training MojoGP multi-output ({kernel})...")
        mojo_result, mojo_B, mojo_time = self._train_mojogp_multi_output(
            X_train, Y_train, kernel, n_iterations, lr, seed
        )
        mojo_mean, mojo_var = self._predict_mojogp_multi_output(
            X_train, Y_train, X_test, mojo_result, kernel
        )

        print(f"  Training GPyTorch multi-output ({kernel})...")
        gpy_model, gpy_B, gpy_time = self._train_gpytorch_multi_output(
            X_train, Y_train, kernel, num_tasks, n_iterations, lr, seed
        )
        gpy_mean, gpy_var = self._predict_gpytorch_multi_output(gpy_model, X_test)

        return MultiOutputComparisonResult(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            num_tasks=num_tasks,
            mojo_mean=mojo_mean,
            mojo_var=mojo_var,
            gpy_mean=gpy_mean,
            gpy_var=gpy_var,
            Y_test=Y_test,
            mojo_B=mojo_B,
            gpy_B=gpy_B,
            mojo_train_time=mojo_time,
            gpy_train_time=gpy_time,
            comparison_class=_comparison_metadata(n_train)[0],
            fairness_note=_comparison_metadata(n_train)[1],
        )

    def _report_result(self, result: MultiOutputComparisonResult):
        """Print multi-output comparison result."""
        print(f"\n  === {result.kernel} Multi-Output (T={result.num_tasks}) ===")
        print(f"  Mean RMSE (Mojo vs GPy): {result.mean_rmse:.6f}")
        print(f"  Per-task RMSE: {result.per_task_rmse}")
        print(
            f"  Task covariance correlation: {result.task_covariance_correlation:.4f}"
        )
        print(f"  Test RMSE ratio (Mojo/GPy): {result.rmse_ratio:.4f}")
        print(
            f"  Train time: Mojo={result.mojo_train_time:.3f}s, GPy={result.gpy_train_time:.3f}s"
        )
        print(f"  Comparison class: {result.comparison_class}")
        print(f"  Fairness note: {result.fairness_note}")

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.multi_output
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,task_corr", MINIMAL_CONFIGS
    )
    def test_multi_output_gpytorch_parity_core_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_corr: str,
    ):
        """Minimal multi-output prediction accuracy test."""
        assert_gpu_available()

        print(
            f"\n=== Multi-Output: {kernel}, n={n_train}, T={num_tasks}, corr={task_corr} ==="
        )
        result = self._run_comparison(kernel, n_train, n_test, d, num_tasks, task_corr)
        self._report_result(result)

        assert result.rmse_ratio < 2.5, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 2.5x target for multi-output GP"
        )

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.multi_output
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,task_corr", MODERATE_CONFIGS
    )
    def test_multi_output_gpytorch_parity_extended_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_corr: str,
    ):
        """Moderate multi-output prediction accuracy test."""
        assert_gpu_available()

        print(
            f"\n=== Multi-Output: {kernel}, n={n_train}, T={num_tasks}, corr={task_corr} ==="
        )
        result = self._run_comparison(kernel, n_train, n_test, d, num_tasks, task_corr)
        self._report_result(result)

        assert result.rmse_ratio < 2.5, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 2.5x target for multi-output GP"
        )

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.multi_output
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,task_corr", FULL_CONFIGS
    )
    def test_multi_output_gpytorch_parity_broad_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_corr: str,
    ):
        """Full multi-output prediction accuracy test."""
        assert_gpu_available()

        print(
            f"\n=== Multi-Output: {kernel}, n={n_train}, T={num_tasks}, corr={task_corr} ==="
        )
        result = self._run_comparison(kernel, n_train, n_test, d, num_tasks, task_corr)
        self._report_result(result)

        assert result.rmse_ratio < 2.5, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 2.5x target for multi-output GP"
        )

    @pytest.mark.moderate
    @pytest.mark.multi_output
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize(
        "kernel,n_train,n_test,d,num_tasks,task_corr", PUBLISHABLE_CONFIGS
    )
    def test_publishable_predictive_parity_row(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        num_tasks: int,
        task_corr: str,
    ):
        """Larger predictive-parity row kept separate from the diagnostic small-n pool."""
        assert_gpu_available()

        print(
            f"\n=== Multi-Output publishable row: {kernel}, n={n_train}, T={num_tasks}, corr={task_corr} ==="
        )
        result = self._run_comparison(kernel, n_train, n_test, d, num_tasks, task_corr)
        self._report_result(result)

        assert result.comparison_class == "fallback_match"
        assert result.rmse_ratio < 2.5, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 2.5x target for larger multi-output parity row"
        )
