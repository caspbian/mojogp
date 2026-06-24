"""ARD (Automatic Relevance Determination) prediction accuracy tests vs GPyTorch.

Tests MojoGP's ARD capability - learning per-dimension lengthscales
to identify relevant vs irrelevant input dimensions.

Compared against GPyTorch with ARD kernels.
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
from tests.shared.benchmarking.mojogp_runners import train_mojogp_ard, predict_mojogp_ard

import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, RQKernel, ScaleKernel


# =============================================================================
# Test Configurations
# =============================================================================

# Format: (kernel, n_train, n_test, d, relevant_dims)
MINIMAL_CONFIGS = [
    ("rbf", 2000, 80, 5, 2),
    ("matern52", 2000, 80, 5, 2),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("rbf", 3000, 80, 10, 3),
    ("matern32", 3000, 80, 5, 2),
    ("matern12", 3000, 80, 5, 2),
    ("rq", 3000, 80, 5, 2),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    ("rbf", 5000, 100, 10, 3),
    ("rbf", 5000, 100, 20, 5),
    ("rbf", 5000, 100, 30, 5),
    ("matern52", 5000, 100, 10, 3),
    ("matern32", 5000, 100, 10, 3),
    ("matern12", 5000, 100, 10, 3),
    ("rq", 5000, 100, 10, 3),
]


# =============================================================================
# ARD Comparison Result
# =============================================================================


@dataclass
class ARDComparisonResult:
    """Results from comparing MojoGP vs GPyTorch ARD predictions."""

    kernel: str
    n_train: int
    n_test: int
    d: int
    relevant_dims: int

    mojo_lengthscales: np.ndarray = field(repr=False)
    gpy_lengthscales: np.ndarray = field(repr=False)
    true_lengthscales: np.ndarray = field(repr=False)

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
    def separation_ratio_mojo(self) -> float:
        """Ratio of mean irrelevant to mean relevant lengthscales for MojoGP."""
        relevant_mask = np.arange(self.d) < self.relevant_dims
        mojo_relevant = self.mojo_lengthscales[relevant_mask]
        mojo_irrelevant = self.mojo_lengthscales[~relevant_mask]
        if len(mojo_irrelevant) == 0:
            return float("inf")
        return float(np.mean(mojo_irrelevant) / max(np.mean(mojo_relevant), 1e-6))

    @property
    def separation_ratio_gpy(self) -> float:
        """Ratio of mean irrelevant to mean relevant lengthscales for GPyTorch."""
        relevant_mask = np.arange(self.d) < self.relevant_dims
        gpy_relevant = self.gpy_lengthscales[relevant_mask]
        gpy_irrelevant = self.gpy_lengthscales[~relevant_mask]
        if len(gpy_irrelevant) == 0:
            return float("inf")
        return float(np.mean(gpy_irrelevant) / max(np.mean(gpy_relevant), 1e-6))

    @property
    def correct_ranking_mojo(self) -> float:
        """Fraction of dimensions correctly ranked by MojoGP."""
        true_order = np.argsort(self.true_lengthscales)
        mojo_order = np.argsort(self.mojo_lengthscales)
        correct = np.sum(true_order == mojo_order)
        return float(correct / self.d)

    @property
    def correct_ranking_gpy(self) -> float:
        """Fraction of dimensions correctly ranked by GPyTorch."""
        true_order = np.argsort(self.true_lengthscales)
        gpy_order = np.argsort(self.gpy_lengthscales)
        correct = np.sum(true_order == gpy_order)
        return float(correct / self.d)

    @property
    def lengthscale_correlation(self) -> float:
        """Correlation between MojoGP and GPyTorch lengthscales."""
        if (
            np.std(self.mojo_lengthscales) < 1e-10
            or np.std(self.gpy_lengthscales) < 1e-10
        ):
            return 0.0
        return float(np.corrcoef(self.mojo_lengthscales, self.gpy_lengthscales)[0, 1])

    @property
    def mean_rmse(self) -> float:
        """RMSE between MojoGP and GPyTorch mean predictions."""
        return float(np.sqrt(np.mean((self.mojo_mean - self.gpy_mean) ** 2)))

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
class TestARDPredictionAccuracy:
    """Compare MojoGP vs GPyTorch ARD prediction accuracy."""

    @staticmethod
    def _stable_cholesky(K: np.ndarray) -> np.ndarray:
        """Factor a symmetric PSD matrix with adaptive jitter.

        The ARD synthetic generators can land extremely close to singular for
        large `n`, especially in the RBF cases. Retrying with larger jitter
        keeps the benchmark focused on ARD training quality rather than
        incidental NumPy factorization failures.
        """
        K = 0.5 * (K + K.T)
        eye = np.eye(K.shape[0], dtype=K.dtype)
        jitter = 1e-6
        last_error = None
        for _ in range(8):
            try:
                return np.linalg.cholesky(K + jitter * eye)
            except np.linalg.LinAlgError as exc:
                last_error = exc
                jitter *= 10.0
        raise np.linalg.LinAlgError(
            f"Matrix is not positive definite even with adaptive jitter up to {jitter / 10.0:.1e}"
        ) from last_error

    def _generate_ard_data(
        self,
        n_train: int,
        n_test: int,
        d: int,
        relevant_dims: int,
        kernel: str,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate ARD data with relevant and irrelevant dimensions."""
        np.random.seed(seed)

        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_test = np.random.randn(n_test, d).astype(np.float32)

        true_lengthscales = np.ones(d, dtype=np.float32) * 5.0
        true_lengthscales[:relevant_dims] = 0.5

        X_train_scaled = X_train / true_lengthscales
        X_test_scaled = X_test / true_lengthscales

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
        elif kernel == "rq":
            alpha = 2.0
            K = (1 + dists_sq / (2 * alpha)) ** (-alpha)
        else:
            K = np.exp(-0.5 * dists_sq)

        L = self._stable_cholesky(K)
        y_train = (L @ np.random.randn(n_train)).astype(np.float32)

        X_test_scaled_test = X_test / true_lengthscales
        cross_dists_sq = np.sum(
            (X_test_scaled_test[:, None, :] - X_train_scaled[None, :, :]) ** 2, axis=2
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
        elif kernel == "rq":
            alpha = 2.0
            K_cross = (1 + cross_dists_sq / (2 * alpha)) ** (-alpha)
        else:
            K_cross = np.exp(-0.5 * cross_dists_sq)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        y_test = (K_cross @ alpha).astype(np.float32)

        return X_train, X_test, y_train, y_test, true_lengthscales

    def _train_mojogp_ard(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel: str,
        n_iterations: int = 100,
        lr: float = 0.02,
        seed: int = 42,
    ) -> Tuple[Dict[str, Any], np.ndarray, float]:
        """Train MojoGP with ARD."""
        result = train_mojogp_ard(
            X_train,
            y_train,
            kernel_type=kernel,
            n_iterations=n_iterations,
            lr=lr,
            init_noise=0.1,
            init_os=1.0,
            monitor_memory=False,
        )
        lengthscales = np.asarray(
            result.get("learned_params", {}).get(
                "lengthscales", np.ones(X_train.shape[1])
            ),
            dtype=np.float32,
        )
        result["gp"] = result["learned_params"]["_gp"]
        result["train_time"] = result["training_time_s"]
        return result, lengthscales, float(result["training_time_s"])

    def _predict_mojogp_ard(
        self,
        result: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with MojoGP ARD model."""
        pred = predict_mojogp_ard(X_train, y_train, X_test, result)
        return pred["mean"], pred["variance"]

    def _train_gpytorch_ard(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel: str,
        n_iterations: int = 150,
        lr: float = 0.02,
        seed: int = 42,
    ) -> Tuple[Any, np.ndarray, float]:
        """Train GPyTorch with ARD."""
        import time
        from tests.shared.benchmarking.gpytorch_models import gpytorch_cg_settings

        torch.manual_seed(seed)

        train_x = torch.tensor(X_train, dtype=torch.float32).cuda()
        train_y = torch.tensor(y_train, dtype=torch.float32).cuda()
        d = X_train.shape[1]

        likelihood = GaussianLikelihood().cuda()

        if kernel == "rbf":
            base_kernel = RBFKernel(ard_num_dims=d).cuda()
        elif kernel == "matern12":
            base_kernel = MaternKernel(nu=0.5, ard_num_dims=d).cuda()
        elif kernel == "matern32":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=d).cuda()
        elif kernel == "matern52":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=d).cuda()
        elif kernel == "rq":
            base_kernel = RQKernel(ard_num_dims=d).cuda()
        else:
            base_kernel = RBFKernel(ard_num_dims=d).cuda()

        covar_module = ScaleKernel(base_kernel).cuda()

        class ARDGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, covar_module):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean().cuda()
                self.covar_module = covar_module

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        model = ARDGP(train_x, train_y, likelihood, covar_module)

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

        lengthscales = (
            model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        )

        return (model, likelihood), lengthscales, train_time

    def _predict_gpytorch_ard(
        self,
        model_likelihood: Tuple[Any, Any],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with GPyTorch ARD model."""
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
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        relevant_dims: int,
        n_iterations: int = 100,
        lr: float = 0.02,
        seed: int = 42,
    ) -> ARDComparisonResult:
        """Run side-by-side ARD comparison of MojoGP vs GPyTorch."""
        print(f"\n  Generating ARD data (d={d}, relevant={relevant_dims})...")
        X_train, X_test, y_train, y_test, true_lengthscales = self._generate_ard_data(
            n_train, n_test, d, relevant_dims, kernel, seed
        )

        print(f"  Training MojoGP ARD ({kernel})...")
        mojo_result, mojo_ls, mojo_time = self._train_mojogp_ard(
            X_train, y_train, kernel, n_iterations, lr, seed
        )
        mojo_mean, mojo_var = self._predict_mojogp_ard(
            mojo_result, X_train, y_train, X_test
        )

        print(f"  Training GPyTorch ARD ({kernel})...")
        gpy_model, gpy_ls, gpy_time = self._train_gpytorch_ard(
            X_train, y_train, kernel, n_iterations, lr, seed
        )
        gpy_mean, gpy_var = self._predict_gpytorch_ard(gpy_model, X_test)

        mojo_rmse = float(np.sqrt(np.mean((mojo_mean - y_test) ** 2)))
        gpy_rmse = float(np.sqrt(np.mean((gpy_mean - y_test) ** 2)))

        return ARDComparisonResult(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            relevant_dims=relevant_dims,
            mojo_lengthscales=mojo_ls,
            gpy_lengthscales=gpy_ls,
            true_lengthscales=true_lengthscales,
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

    def _report_result(self, result: ARDComparisonResult):
        """Print ARD comparison result."""
        print(
            f"\n  === {result.kernel} ARD (d={result.d}, rel={result.relevant_dims}) ==="
        )
        print(f"  MojoGP lengthscales: {result.mojo_lengthscales}")
        print(f"  GPyTorch lengthscales: {result.gpy_lengthscales}")
        print(f"  True lengthscales: {result.true_lengthscales}")
        print(
            f"  Separation ratio - Mojo: {result.separation_ratio_mojo:.2f}, GPy: {result.separation_ratio_gpy:.2f}"
        )
        print(
            f"  Correct ranking - Mojo: {result.correct_ranking_mojo:.2f}, GPy: {result.correct_ranking_gpy:.2f}"
        )
        print(f"  Lengthscale correlation: {result.lengthscale_correlation:.4f}")
        print(f"  Test RMSE ratio (Mojo/GPy): {result.rmse_ratio:.4f}")
        print(
            f"  Train time: Mojo={result.mojo_train_time:.3f}s, GPy={result.gpy_train_time:.3f}s"
        )

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.ard
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n_train,n_test,d,relevant_dims", MINIMAL_CONFIGS)
    def test_ard_gpytorch_parity_core_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        relevant_dims: int,
    ):
        """Minimal ARD prediction accuracy test."""
        assert_gpu_available()

        print(
            f"\n=== ARD Accuracy: {kernel}, n={n_train}, d={d}, rel={relevant_dims} ==="
        )
        result = self._run_comparison(kernel, n_train, n_test, d, relevant_dims)
        self._report_result(result)

        assert result.separation_ratio_mojo > 1.2, (
            f"ARD separation ratio {result.separation_ratio_mojo:.2f} < 1.2"
        )
        assert result.rmse_ratio < 1.35, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 1.35x target"
        )

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.ard
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n_train,n_test,d,relevant_dims", MODERATE_CONFIGS)
    def test_ard_gpytorch_parity_extended_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        relevant_dims: int,
    ):
        """Moderate ARD prediction accuracy test."""
        assert_gpu_available()

        print(
            f"\n=== ARD Accuracy: {kernel}, n={n_train}, d={d}, rel={relevant_dims} ==="
        )
        result = self._run_comparison(kernel, n_train, n_test, d, relevant_dims)
        self._report_result(result)

        assert result.separation_ratio_mojo > 1.2, (
            f"ARD separation ratio {result.separation_ratio_mojo:.2f} < 1.2"
        )
        assert result.rmse_ratio < 1.35, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 1.35x target"
        )

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.ard
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n_train,n_test,d,relevant_dims", FULL_CONFIGS)
    def test_ard_gpytorch_parity_broad_configs(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        relevant_dims: int,
    ):
        """Full ARD prediction accuracy test."""
        assert_gpu_available()

        print(
            f"\n=== ARD Accuracy: {kernel}, n={n_train}, d={d}, rel={relevant_dims} ==="
        )
        result = self._run_comparison(kernel, n_train, n_test, d, relevant_dims)
        self._report_result(result)

        assert result.separation_ratio_mojo > 1.2, (
            f"ARD separation ratio {result.separation_ratio_mojo:.2f} < 1.2"
        )
        assert result.rmse_ratio < 1.35, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds 1.35x target"
        )
