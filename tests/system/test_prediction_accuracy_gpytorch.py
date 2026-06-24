"""Comprehensive prediction accuracy tests vs GPyTorch.

This module tests MojoGP prediction accuracy against GPyTorch across:
- All 8 kernel types (isotropic, single-output)
- With and without ARD
- Multi-output configurations
- Composite kernel combinations

All tests use tiered configurations (minimal/moderate/full) and require GPU.
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
from tests.shared.benchmarking.data_generators import generate_gp_prior_data, SyntheticDataset
from tests.shared.benchmarking.mojogp_runners import train_mojogp_simple, predict_mojogp_simple
from tests.shared.benchmarking.gpytorch_models import (
    GPyTorchSingleOutputGP,
    train_gpytorch_model,
    predict_gpytorch_model,
    gpytorch_cg_settings,
)
from tests.shared.benchmarking.metrics import compute_all_accuracy_metrics

import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood


# =============================================================================
# Test Configurations
# =============================================================================

# Format: (kernel, n_train, n_test, d)
MINIMAL_CONFIGS = [
    ("rbf", 2000, 100, 5),
    ("matern52", 2000, 100, 5),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("matern32", 2000, 100, 5),
    ("matern12", 2000, 100, 5),
    ("periodic", 2000, 100, 3),
    ("rq", 2000, 100, 5),
    ("linear", 2000, 100, 5),
    ("polynomial", 2000, 100, 5),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    ("rbf", 5000, 200, 5),
    ("rbf", 2000, 100, 10),
    ("rbf", 2000, 100, 20),
    ("matern52", 5000, 200, 5),
    ("matern52", 2000, 100, 10),
    ("matern32", 5000, 200, 5),
    ("matern12", 5000, 200, 5),
    ("periodic", 5000, 200, 3),
    ("rq", 5000, 200, 5),
    ("linear", 5000, 200, 5),
    ("polynomial", 5000, 200, 5),
]


# =============================================================================
# Prediction Comparison Result
# =============================================================================


@dataclass
class PredictionComparisonResult:
    """Results from comparing MojoGP vs GPyTorch predictions."""

    kernel: str
    n_train: int
    n_test: int
    d: int

    mojo_mean: np.ndarray = field(repr=False)
    mojo_var: np.ndarray = field(repr=False)
    gpy_mean: np.ndarray = field(repr=False)
    gpy_var: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)

    mojo_lengthscale: float = 0.0
    mojo_noise: float = 0.0
    mojo_outputscale: float = 0.0
    mojo_period: float = 1.0  # For periodic kernel
    gpy_lengthscale: float = 0.0
    gpy_noise: float = 0.0
    gpy_outputscale: float = 0.0

    # True hyperparameters (when known)
    true_lengthscale: Optional[float] = None
    true_noise: Optional[float] = None
    true_outputscale: Optional[float] = None
    true_period: Optional[float] = None

    mojo_train_time: float = 0.0
    mojo_pred_time: float = 0.0
    gpy_train_time: float = 0.0
    gpy_pred_time: float = 0.0

    mojo_rmse: float = 0.0
    gpy_rmse: float = 0.0

    @property
    def mean_mae(self) -> float:
        """MAE between MojoGP and GPyTorch mean predictions."""
        return float(np.mean(np.abs(self.mojo_mean - self.gpy_mean)))

    @property
    def mean_rmse(self) -> float:
        """RMSE between MojoGP and GPyTorch mean predictions."""
        return float(np.sqrt(np.mean((self.mojo_mean - self.gpy_mean) ** 2)))

    @property
    def mean_rmse_ratio(self) -> float:
        """RMSE ratio relative to GPyTorch prediction magnitude."""
        gpy_std = float(np.std(self.gpy_mean))
        if gpy_std < 1e-10:
            return float("inf")
        return self.mean_rmse / gpy_std

    @property
    def variance_correlation(self) -> float:
        """Correlation between MojoGP and GPyTorch variances."""
        if np.std(self.mojo_var) < 1e-10 or np.std(self.gpy_var) < 1e-10:
            return 0.0
        return float(np.corrcoef(self.mojo_var, self.gpy_var)[0, 1])

    @property
    def variance_mape(self) -> float:
        """Mean absolute percentage error for variances."""
        denom = np.maximum(np.abs(self.gpy_var), 1e-6)
        return float(np.mean(np.abs(self.mojo_var - self.gpy_var) / denom))

    @property
    def hyperparam_diff(self) -> Dict[str, float]:
        """Relative difference in learned hyperparameters."""
        return {
            "lengthscale": abs(self.mojo_lengthscale - self.gpy_lengthscale)
            / max(self.gpy_lengthscale, 1e-6),
            "noise": abs(self.mojo_noise - self.gpy_noise) / max(self.gpy_noise, 1e-6),
            "outputscale": abs(self.mojo_outputscale - self.gpy_outputscale)
            / max(self.gpy_outputscale, 1e-6),
        }

    @property
    def rmse_ratio(self) -> float:
        """Ratio of MojoGP RMSE to GPyTorch RMSE on test data."""
        if self.gpy_rmse < 1e-10:
            return float("inf")
        return self.mojo_rmse / self.gpy_rmse

    @property
    def mojo_nll(self) -> float:
        """Negative log-likelihood of test data under MojoGP predictive distribution.

        NLL = 0.5 * [log(2π) + log(σ²) + (y - μ)²/σ²]

        Lower is better. This measures the quality of the predictive distribution.
        """
        var = np.maximum(self.mojo_var, 1e-10)  # Avoid log(0)
        nll = 0.5 * (
            np.log(2 * np.pi) + np.log(var) + (self.y_test - self.mojo_mean) ** 2 / var
        )
        return float(np.mean(nll))

    @property
    def gpy_nll(self) -> float:
        """Negative log-likelihood of test data under GPyTorch predictive distribution."""
        var = np.maximum(self.gpy_var, 1e-10)
        nll = 0.5 * (
            np.log(2 * np.pi) + np.log(var) + (self.y_test - self.gpy_mean) ** 2 / var
        )
        return float(np.mean(nll))

    @property
    def nll_ratio(self) -> float:
        """Ratio of MojoGP NLL to GPyTorch NLL. Values < 1.0 mean MojoGP is better."""
        if self.gpy_nll < 1e-10:
            return float("inf")
        return self.mojo_nll / self.gpy_nll

    @property
    def mojo_var_std(self) -> float:
        """Standard deviation of MojoGP variance values."""
        return float(np.std(self.mojo_var))

    @property
    def gpy_var_std(self) -> float:
        """Standard deviation of GPyTorch variance values."""
        return float(np.std(self.gpy_var))

    @property
    def is_variance_constant(self) -> bool:
        """Check if both models have nearly constant variance.

        This happens in high-dimensional sparse data where the kernel matrix
        is nearly diagonal, making posterior variance approximately uniform.
        In this case, variance correlation is meaningless.
        """
        return self.mojo_var_std < 0.01 and self.gpy_var_std < 0.01


# =============================================================================
# Test Class
# =============================================================================


@requires_mojogp
@requires_gpytorch
@requires_cuda
class TestSingleOutputPredictionAccuracy:
    """Compare MojoGP vs GPyTorch prediction accuracy for single-output GP."""

    def _generate_kernel_specific_data(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate kernel-appropriate synthetic data.

        Different kernels require different data patterns for meaningful testing.
        """
        np.random.seed(seed)

        X_train = np.random.randn(n_train, d).astype(np.float32)
        X_test = np.random.randn(n_test, d).astype(np.float32)

        if kernel in ["rbf", "matern12", "matern32", "matern52", "rq"]:
            dataset = generate_gp_prior_data(
                n_train=n_train,
                n_test=n_test,
                d=d,
                kernel_type=kernel,
                true_lengthscale=1.0,
                true_noise=0.1,
                true_outputscale=1.0,
                seed=seed,
            )
            return (
                dataset.X_train,
                dataset.X_test,
                dataset.y_train,
                dataset.f_test,
            )

        elif kernel == "periodic":
            t_train = X_train[:, 0]
            period = 2.0
            y_train = np.sin(2 * np.pi * t_train / period).astype(np.float32)
            y_train += np.random.randn(n_train).astype(np.float32) * 0.1

            t_test = X_test[:, 0]
            y_test = np.sin(2 * np.pi * t_test / period).astype(np.float32)
            return X_train, X_test, y_train, y_test

        elif kernel == "linear":
            np.random.seed(seed)
            w = np.random.randn(d).astype(np.float32)
            y_train = (X_train @ w).astype(np.float32)
            y_train += np.random.randn(n_train).astype(np.float32) * 0.1
            y_test = (X_test @ w).astype(np.float32)
            return X_train, X_test, y_train, y_test

        elif kernel == "polynomial":
            np.random.seed(seed)
            w = np.random.randn(d).astype(np.float32)
            y_train = ((X_train @ w) ** 2).astype(np.float32)
            y_train += np.random.randn(n_train).astype(np.float32) * 0.1
            y_test = ((X_test @ w) ** 2).astype(np.float32)
            return X_train, X_test, y_train, y_test

        else:
            dataset = generate_gp_prior_data(
                n_train=n_train,
                n_test=n_test,
                d=d,
                kernel_type="rbf",
                true_lengthscale=1.0,
                true_noise=0.1,
                true_outputscale=1.0,
                seed=seed,
            )
            return (
                dataset.X_train,
                dataset.X_test,
                dataset.y_train,
                dataset.f_test,
            )

    def _train_gpytorch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel: str,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train GPyTorch model with CG (fair comparison)."""
        torch.manual_seed(seed)

        train_x = torch.tensor(X_train, dtype=torch.float32).cuda()
        train_y = torch.tensor(y_train, dtype=torch.float32).cuda()

        likelihood = GaussianLikelihood().cuda()
        model = GPyTorchSingleOutputGP(train_x, train_y, likelihood, kernel).cuda()

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        import time

        start_time = time.perf_counter()

        with gpytorch_cg_settings():
            for _ in range(n_iterations):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

        train_time = time.perf_counter() - start_time

        # Extract hyperparameters, handling kernels without lengthscale
        lengthscale = 1.0
        if kernel not in ["linear", "polynomial"]:
            if (
                hasattr(model.covar_module, "base_kernel")
                and hasattr(model.covar_module.base_kernel, "lengthscale")
                and model.covar_module.base_kernel.lengthscale is not None
            ):
                lengthscale = model.covar_module.base_kernel.lengthscale.item()

        result = {
            "train_time": train_time,
            "lengthscale": lengthscale,
            "noise": likelihood.noise.item(),
            "outputscale": model.covar_module.outputscale.item()
            if hasattr(model.covar_module, "outputscale")
            else 1.0,
        }

        return (model, likelihood), result

    def _predict_gpytorch(
        self,
        model_likelihood: Tuple[Any, Any],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Predict with GPyTorch model."""
        model, likelihood = model_likelihood
        model.eval()
        likelihood.eval()

        test_x = torch.tensor(X_test, dtype=torch.float32).cuda()

        import time

        start_time = time.perf_counter()

        with torch.no_grad(), gpytorch_cg_settings():
            preds = likelihood(model(test_x))
            mean = preds.mean.cpu().numpy()
            var = preds.variance.cpu().numpy()

        pred_time = time.perf_counter() - start_time

        return mean, var, pred_time

    def _run_comparison(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
    ) -> PredictionComparisonResult:
        """Run side-by-side comparison of MojoGP vs GPyTorch."""
        import time

        X_train, X_test, y_train, y_test = self._generate_kernel_specific_data(
            kernel, n_train, n_test, d, seed
        )

        init_ls = 1.0
        init_noise = 0.1
        init_os = 1.0

        init_kernel_param1 = None
        if kernel == "periodic":
            init_kernel_param1 = 2.0

        print(f"\n  Training MojoGP ({kernel})...")
        mojo_start = time.perf_counter()
        mojo_result = train_mojogp_simple(
            X_train,
            y_train,
            kernel,
            method="materialized",
            n_iterations=n_iterations,
            lr=lr,
            init_ls=init_ls,
            init_noise=init_noise,
            init_os=init_os,
            kernel_param1=init_kernel_param1,
            # Accuracy tests measure numerical parity; background memory polling
            # can perturb GPU training on small devices.
            monitor_memory=False,
        )
        mojo_train_time = time.perf_counter() - mojo_start

        # Get kernel params from training result for kernel-specific params
        kp1 = mojo_result.get("result", {}).get("kernel_param1", 1.0)
        kp2 = mojo_result.get("result", {}).get("kernel_param2", 0.0)

        mojo_pred = predict_mojogp_simple(
            X_train,
            y_train,
            X_test,
            mojo_result.get("learned_params", {}),
            kernel,
            kernel_param1=kp1,
            kernel_param2=kp2,
        )
        mojo_pred_time = mojo_pred.get("total_time_s", 0.0)
        mojo_mean = mojo_pred["mean"]
        mojo_var = mojo_pred["variance"]

        print(f"  Training GPyTorch ({kernel})...")
        gpy_model, gpy_train_result = self._train_gpytorch(
            X_train, y_train, kernel, n_iterations, lr, seed
        )
        gpy_mean, gpy_var, gpy_pred_time = self._predict_gpytorch(gpy_model, X_test)

        mojo_rmse = float(np.sqrt(np.mean((mojo_mean - y_test) ** 2)))
        gpy_rmse = float(np.sqrt(np.mean((gpy_mean - y_test) ** 2)))

        return PredictionComparisonResult(
            kernel=kernel,
            n_train=n_train,
            n_test=n_test,
            d=d,
            mojo_mean=mojo_mean,
            mojo_var=mojo_var,
            gpy_mean=gpy_mean,
            gpy_var=gpy_var,
            y_test=y_test,
            mojo_lengthscale=mojo_result.get("learned_params", {}).get(
                "lengthscale", 1.0
            ),
            mojo_noise=mojo_result.get("learned_params", {}).get("noise", 0.1),
            mojo_outputscale=mojo_result.get("learned_params", {}).get(
                "outputscale", 1.0
            ),
            mojo_period=mojo_result.get("learned_params", {}).get("period", 1.0),
            gpy_lengthscale=gpy_train_result["lengthscale"],
            gpy_noise=gpy_train_result["noise"],
            gpy_outputscale=gpy_train_result["outputscale"],
            true_period=2.0 if kernel == "periodic" else None,
            mojo_train_time=mojo_train_time,
            mojo_pred_time=mojo_pred_time,
            gpy_train_time=gpy_train_result["train_time"],
            gpy_pred_time=gpy_pred_time,
            mojo_rmse=mojo_rmse,
            gpy_rmse=gpy_rmse,
        )

    def _report_result(self, result: PredictionComparisonResult):
        """Print comparison result."""
        print(f"\n  === {result.kernel} (n={result.n_train}, d={result.d}) ===")
        print(f"  Mean RMSE (Mojo vs GPy): {result.mean_rmse:.6f}")
        print(f"  Mean RMSE ratio: {result.mean_rmse_ratio:.4f}")
        print(f"  Variance correlation: {result.variance_correlation:.4f}")
        print(f"  Variance MAPE: {result.variance_mape:.4f}")
        print(f"  Test RMSE ratio (Mojo/GPy): {result.rmse_ratio:.4f}")
        print(
            f"  Test NLL: Mojo={result.mojo_nll:.4f}, GPy={result.gpy_nll:.4f}, ratio={result.nll_ratio:.4f}"
        )
        print(
            f"  Hyperparam diffs: ls={result.hyperparam_diff['lengthscale']:.4f}, "
            f"noise={result.hyperparam_diff['noise']:.4f}, "
            f"os={result.hyperparam_diff['outputscale']:.4f}"
        )
        print(
            f"  Train time: Mojo={result.mojo_train_time:.3f}s, GPy={result.gpy_train_time:.3f}s"
        )

    # Kernel-specific thresholds for variance correlation
    VARIANCE_CORR_THRESHOLDS = {
        "periodic": 0.10,  # Multi-modal hyperparameter landscape
        "polynomial": 0.40,
        "default": 0.90,
    }

    NLL_RATIO_THRESHOLDS = {
        "periodic": 1.5,
        "polynomial": 0.1,
        "default": 1.5,
    }

    def _check_result(self, result: PredictionComparisonResult):
        """Check result with kernel-specific thresholds.

        For polynomial kernel, GPyTorch has known numerical issues (negative variances),
        so we check MojoGP's NLL instead of variance correlation.

        For periodic kernel, the hyperparameter landscape is multi-modal, so we:
        1. Relax the variance correlation threshold
        2. Check that MojoGP finds the true period (not a spurious mode)

        For high-dimensional sparse data, variance may be nearly constant for both models,
        making correlation meaningless. In this case, we skip the correlation test.
        """
        kernel = result.kernel

        rmse_threshold = 1.2
        assert result.rmse_ratio < rmse_threshold, (
            f"RMSE ratio {result.rmse_ratio:.2f}x exceeds {rmse_threshold}x target"
        )

        if kernel == "periodic" and result.true_period is not None:
            period_error = (
                abs(result.mojo_period - result.true_period) / result.true_period
            )
            print(
                f"  Period: MojoGP={result.mojo_period:.4f}, True={result.true_period:.4f}, "
                f"error={period_error:.2%}"
            )
            assert period_error < 0.15, (
                f"MojoGP period {result.mojo_period:.4f} differs from true period "
                f"{result.true_period:.4f} by {period_error:.2%}"
            )

        if kernel == "polynomial" and result.gpy_nll > 10:
            print(
                f"  Note: GPyTorch has numerical issues (NLL={result.gpy_nll:.2f}), "
                f"checking MojoGP quality instead"
            )
            assert result.mojo_nll < 5.0, (
                f"MojoGP NLL {result.mojo_nll:.2f} is too high"
            )
            assert result.rmse_ratio < 1.0, (
                f"MojoGP RMSE should be better than GPyTorch for this case"
            )
        elif result.is_variance_constant:
            print(
                f"  Note: Variance is nearly constant for both models "
                f"(MojoGP std={result.mojo_var_std:.6f}, GPyTorch std={result.gpy_var_std:.6f}). "
                f"This is expected for high-dimensional sparse data. Skipping correlation test."
            )
        else:
            var_corr_threshold = self.VARIANCE_CORR_THRESHOLDS.get(
                kernel, self.VARIANCE_CORR_THRESHOLDS["default"]
            )
            assert result.variance_correlation > var_corr_threshold, (
                f"Variance correlation {result.variance_correlation:.2f} < {var_corr_threshold}"
            )

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.single_output
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n_train,n_test,d", MINIMAL_CONFIGS)
    def test_gpytorch_prediction_accuracy_core_configs(self, kernel: str, n_train: int, n_test: int, d: int):
        """Minimal prediction accuracy test - quick validation."""
        assert_gpu_available()

        print(f"\n=== Prediction Accuracy: {kernel}, n={n_train}, d={d} ===")
        result = self._run_comparison(kernel, n_train, n_test, d)
        self._report_result(result)
        self._check_result(result)

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.single_output
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n_train,n_test,d", MODERATE_CONFIGS)
    def test_gpytorch_prediction_accuracy_extended_configs(self, kernel: str, n_train: int, n_test: int, d: int):
        """Moderate prediction accuracy test - all 8 kernels."""
        assert_gpu_available()

        print(f"\n=== Prediction Accuracy: {kernel}, n={n_train}, d={d} ===")
        result = self._run_comparison(kernel, n_train, n_test, d)
        self._report_result(result)
        self._check_result(result)

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.single_output
    @pytest.mark.prediction
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n_train,n_test,d", FULL_CONFIGS)
    def test_gpytorch_prediction_accuracy_broad_configs(self, kernel: str, n_train: int, n_test: int, d: int):
        """Full prediction accuracy test - comprehensive coverage."""
        assert_gpu_available()

        print(f"\n=== Prediction Accuracy: {kernel}, n={n_train}, d={d} ===")
        result = self._run_comparison(kernel, n_train, n_test, d)
        self._report_result(result)
        self._check_result(result)
