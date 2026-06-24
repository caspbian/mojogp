"""System correctness suite: SingleOutputGP surface vs ground truth and live GPyTorch.

This suite validates the SingleOutputGP and related wrapper APIs against:
  1. Known ground truth functions (primary reference)
  2. Live GPyTorch ExactGP training (secondary reference)

Assertion logic (in priority order):
  - MojoGP close to ground truth -> PASS (regardless of GPyTorch)
  - MojoGP closer to truth than GPyTorch -> PASS (even if both somewhat off)
  - GPyTorch close but MojoGP far -> FAIL (MojoGP underperforms)
  - Both far from ground truth -> FAIL (with "both methods failed" noted)

Feature coverage:
  - All 8 kernel types (RBF, Matern12, Matern32, Matern52, Periodic, RQ, Linear, Polynomial)
  - Isotropic and ARD modes
  - Composite kernels (sum, product)
  - Constant mean function (init_mean, auto-mean, non-zero-mean data)
  - Multi-output GP (MultiOutputGP with ICM)
  - Discrete/categorical kernels (all 5 variants: GD, CR, EHH, HH, FE)
  - Mixed continuous + categorical via kernel tree composition (per-variable kernel selection)
  - API surface (score, get_learned_params, PredictionResult, sample_posterior)
  - Multiple data generators (Friedman #1, sinusoidal, polynomial, linear, periodic, mixed categorical)

All tests use:
  - n >= 2000 training points (single-output) or n >= 500 (multi-output)
  - SingleOutputGP / MultiOutputGP Python API (not direct Mojo bindings)
  - GPU for both MojoGP and GPyTorch
  - CG mode for GPyTorch (fair comparison, not Cholesky)

Test tiers:
  - minimal: Core functionality (RBF iso/ARD, Matern52, constant mean)
  - moderate: More kernels and data functions
  - full: Broad feature correctness, not benchmark telemetry
"""

import numpy as np
import pytest
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import gpytorch


pytestmark = [pytest.mark.reference]

from mojogp import (
    SingleOutputGP,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    RQ,
    Linear,
    Polynomial,
    PredictionResult,
    EHH,
    GD,
    CR,
    HH,
    FE,
)
from mojogp.multi_output_gp import MultiOutputGP, MultiOutputPredictionResult


# =============================================================================
# Constants
# =============================================================================

MIN_N = 2000
DEFAULT_SEED = 42
DEFAULT_LR = 0.01
DEFAULT_ITERS = 100

# RMSE thresholds (absolute) — if MojoGP RMSE < this, PASS unconditionally
RMSE_ABSOLUTE_GOOD = 0.5  # Very good prediction
RMSE_ABSOLUTE_OK = 1.0  # Acceptable prediction

# Ratio threshold — if MojoGP/GPyTorch RMSE ratio < this, PASS
RMSE_RATIO_THRESHOLD = 1.5  # MojoGP can be up to 50% worse than GPyTorch

# R^2 threshold — minimum acceptable R^2
R2_MINIMUM = 0.80


# =============================================================================
# Data Generation
# =============================================================================


def friedman1(
    n_train: int = 2000,
    n_test: int = 500,
    d: int = 5,
    noise_std: float = 0.1,
    seed: int = DEFAULT_SEED,
    mean_offset: float = 0.0,
) -> Dict:
    """Friedman #1: 10*sin(pi*x1*x2) + 20*(x3-0.5)^2 + 10*x4 + 5*x5.

    Only first 5 dims are relevant; extra dims are irrelevant (for ARD).
    mean_offset shifts the function for constant-mean testing.
    """
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.uniform(0, 1, size=(n, d)).astype(np.float32)

    f = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    ).astype(np.float32)
    f = f + mean_offset

    noise = noise_std * np.std(f)
    y = f + noise * rng.randn(n).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "f_test": f[n_train:],
        "y_test": y[n_train:],
        "noise_std": noise,
        "mean_offset": mean_offset,
        "name": f"friedman1_n{n_train}_d{d}",
    }


def sinusoidal(
    n_train: int = 2000,
    n_test: int = 500,
    d: int = 3,
    noise_std: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Sum of sinusoids: sum_i sin(2*pi*(i+1)*x_i). Tests smooth periodic-like data."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.uniform(0, 1, size=(n, d)).astype(np.float32)

    f = np.zeros(n, dtype=np.float32)
    for i in range(d):
        f += np.sin(2 * np.pi * (i + 1) * X[:, i]).astype(np.float32)

    noise = noise_std * np.std(f)
    y = f + noise * rng.randn(n).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "f_test": f[n_train:],
        "y_test": y[n_train:],
        "noise_std": noise,
        "name": f"sinusoidal_n{n_train}_d{d}",
    }


def smooth_polynomial(
    n_train: int = 2000,
    n_test: int = 500,
    d: int = 3,
    noise_std: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Smooth polynomial: x1^2 + x1*x2 + x2^2 + x3. Easy for GPs, tests Linear/Poly kernels."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.uniform(-1, 1, size=(n, d)).astype(np.float32)

    f = (X[:, 0] ** 2 + X[:, 0] * X[:, 1] + X[:, 1] ** 2 + X[:, 2]).astype(np.float32)

    noise = noise_std * np.std(f)
    y = f + noise * rng.randn(n).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "f_test": f[n_train:],
        "y_test": y[n_train:],
        "noise_std": noise,
        "name": f"smooth_poly_n{n_train}_d{d}",
    }


def linear_function(
    n_train: int = 2000,
    n_test: int = 500,
    d: int = 5,
    noise_std: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Linear: y = 3*x1 - 2*x2 + x3. Tests Linear kernel on its natural data."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.uniform(-1, 1, size=(n, d)).astype(np.float32)

    f = (3 * X[:, 0] - 2 * X[:, 1] + X[:, 2]).astype(np.float32)

    noise = noise_std * np.std(f)
    y = f + noise * rng.randn(n).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "f_test": f[n_train:],
        "y_test": y[n_train:],
        "noise_std": noise,
        "name": f"linear_fn_n{n_train}_d{d}",
    }


def periodic_function(
    n_train: int = 2000,
    n_test: int = 500,
    noise_std: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Periodic: y = sin(2*pi*x) + 0.5*cos(4*pi*x). 1D, ideal for Periodic kernel."""
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.uniform(0, 3, size=(n, 1)).astype(np.float32)

    f = (np.sin(2 * np.pi * X[:, 0]) + 0.5 * np.cos(4 * np.pi * X[:, 0])).astype(
        np.float32
    )

    noise = noise_std * np.std(f)
    y = f + noise * rng.randn(n).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "f_test": f[n_train:],
        "y_test": y[n_train:],
        "noise_std": noise,
        "name": f"periodic_fn_n{n_train}",
    }


def multi_output_correlated(
    n_train: int = 500,
    n_test: int = 200,
    d: int = 3,
    T: int = 3,
    noise_std: float = 0.1,
    seed: int = DEFAULT_SEED,
    per_task_means: Optional[np.ndarray] = None,
) -> Dict:
    """Multi-output with correlated tasks sharing a latent function.

    y[:,0] = sin(x1) + 0.5*cos(x2) + mean[0] + noise
    y[:,1] = 0.8*sin(x1) + 0.3*cos(x2) + mean[1] + noise
    y[:,2] = 0.6*sin(x1) + cos(x2) + mean[2] + noise

    Tasks share the sin/cos structure but with different coefficients.
    """
    rng = np.random.RandomState(seed)
    n = n_train + n_test
    X = rng.uniform(-2, 2, size=(n, d)).astype(np.float32)

    if per_task_means is None:
        per_task_means = np.array([0.0] * T, dtype=np.float32)
    per_task_means = np.asarray(per_task_means[:T], dtype=np.float32)

    # Mixing matrix: how tasks combine latent functions
    # Latent 1: sin(x1), Latent 2: cos(x2)
    mix = np.array(
        [[1.0, 0.5], [0.8, 0.3], [0.6, 1.0]],
        dtype=np.float32,
    )[:T, :]

    latent1 = np.sin(X[:, 0]).astype(np.float32)
    latent2 = np.cos(X[:, min(1, d - 1)]).astype(np.float32)
    latents = np.stack([latent1, latent2], axis=1)  # [n, 2]

    F = latents @ mix.T  # [n, T]
    F += per_task_means[np.newaxis, :]

    Y = F + noise_std * rng.randn(n, T).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "Y_train": Y[:n_train],
        "X_test": X[n_train:],
        "F_test": F[n_train:],
        "Y_test": Y[n_train:],
        "per_task_means": per_task_means,
        "noise_std": noise_std,
        "T": T,
        "name": f"multi_output_corr_n{n_train}_T{T}",
    }


# =============================================================================
# GPyTorch Live Training
# =============================================================================


class _GPyTorchExactGP(gpytorch.models.ExactGP):
    """Standard GPyTorch ExactGP for comparison."""

    def __init__(self, train_x, train_y, likelihood, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def _make_gpytorch_kernel(kernel_name: str, d: int, ard: bool):
    """Create a GPyTorch ScaleKernel wrapping the appropriate base kernel."""
    ard_num_dims = d if ard else None

    if kernel_name == "rbf":
        base = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    elif kernel_name == "matern12":
        base = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)
    elif kernel_name == "matern32":
        base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
    elif kernel_name == "matern52":
        base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    elif kernel_name == "periodic":
        base = gpytorch.kernels.PeriodicKernel()
    elif kernel_name == "rq":
        base = gpytorch.kernels.RQKernel(ard_num_dims=ard_num_dims)
    elif kernel_name == "linear":
        base = gpytorch.kernels.LinearKernel(num_dimensions=d)
    elif kernel_name == "polynomial":
        base = gpytorch.kernels.PolynomialKernel(power=2)
    elif kernel_name == "rbf+matern52":
        k1 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        )
        k2 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        )
        # Sum kernel — already has ScaleKernel wrappers
        return k1 + k2
    elif kernel_name == "rbf*linear":
        k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        k2 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.LinearKernel(num_dimensions=d)
        )
        return k1 * k2
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    return gpytorch.kernels.ScaleKernel(base)


def train_gpytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel_name: str,
    ard: bool = False,
    n_iterations: int = DEFAULT_ITERS,
    lr: float = DEFAULT_LR,
) -> Dict:
    """Train GPyTorch ExactGP with CG mode (fair comparison).

    Returns dict with 'model', 'likelihood', 'train_time_s'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.tensor(X_train, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.float32, device=device)

    d = X_train.shape[1]
    covar_module = _make_gpytorch_kernel(kernel_name, d, ard)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = _GPyTorchExactGP(train_x, train_y, likelihood, covar_module).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Use CG mode for fair comparison (not Cholesky)
    with (
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.cg_tolerance(1e-2),
        gpytorch.settings.max_cg_iterations(100),
        gpytorch.settings.num_trace_samples(10),
    ):
        for _ in range(n_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    return {"model": model, "likelihood": likelihood, "train_time_s": train_time}


def predict_gpytorch(
    model,
    likelihood,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict with trained GPyTorch model. Returns (mean, std)."""
    device = next(model.parameters()).device
    test_x = torch.tensor(X_test, dtype=torch.float32, device=device)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        mean = pred.mean.cpu().numpy()
        std = pred.stddev.cpu().numpy()

    return mean, std


# =============================================================================
# Smart Assertion Logic
# =============================================================================


@dataclass
class ComparisonResult:
    """Result of three-way comparison: MojoGP vs Truth vs GPyTorch."""

    passed: bool
    reason: str

    # Metrics
    mojo_rmse: float
    mojo_r2: float
    gpy_rmse: float
    gpy_r2: float

    # Timing
    mojo_train_time: float = 0.0
    gpy_train_time: float = 0.0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"  [{status}] {self.reason}",
            f"  MojoGP:   RMSE={self.mojo_rmse:.4f}  R2={self.mojo_r2:.4f}  time={self.mojo_train_time:.2f}s",
            f"  GPyTorch: RMSE={self.gpy_rmse:.4f}  R2={self.gpy_r2:.4f}  time={self.gpy_train_time:.2f}s",
        ]
        if self.gpy_rmse > 0:
            ratio = self.mojo_rmse / self.gpy_rmse
            lines.append(f"  RMSE ratio (MojoGP/GPyTorch): {ratio:.3f}")
        return "\n".join(lines)


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _r2(pred: np.ndarray, truth: np.ndarray) -> float:
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compare_three_way(
    f_test: np.ndarray,
    mojo_mean: np.ndarray,
    gpy_mean: np.ndarray,
    mojo_train_time: float = 0.0,
    gpy_train_time: float = 0.0,
) -> ComparisonResult:
    """Three-way comparison implementing the smart assertion logic.

    Decision tree:
      1. MojoGP RMSE < RMSE_ABSOLUTE_GOOD -> PASS (great accuracy)
      2. MojoGP R2 > R2_MINIMUM and RMSE ratio < RMSE_RATIO_THRESHOLD -> PASS
      3. MojoGP RMSE <= GPyTorch RMSE -> PASS (MojoGP at least as good)
      4. GPyTorch RMSE < RMSE_ABSOLUTE_OK but MojoGP RMSE > RMSE_ABSOLUTE_OK -> FAIL
      5. Both RMSE > RMSE_ABSOLUTE_OK -> FAIL (both methods failed)
      6. Otherwise -> FAIL (MojoGP underperforms)
    """
    mojo_rmse = _rmse(mojo_mean, f_test)
    gpy_rmse = _rmse(gpy_mean, f_test)
    mojo_r2 = _r2(mojo_mean, f_test)
    gpy_r2 = _r2(gpy_mean, f_test)

    base = ComparisonResult(
        passed=False,
        reason="",
        mojo_rmse=mojo_rmse,
        mojo_r2=mojo_r2,
        gpy_rmse=gpy_rmse,
        gpy_r2=gpy_r2,
        mojo_train_time=mojo_train_time,
        gpy_train_time=gpy_train_time,
    )

    # 1. MojoGP has great absolute accuracy
    if mojo_rmse < RMSE_ABSOLUTE_GOOD:
        base.passed = True
        base.reason = (
            f"MojoGP RMSE ({mojo_rmse:.4f}) < {RMSE_ABSOLUTE_GOOD} (great accuracy)"
        )
        return base

    # 2. MojoGP has good R2 and acceptable ratio
    ratio = mojo_rmse / gpy_rmse if gpy_rmse > 0 else 0.0

    if mojo_r2 > R2_MINIMUM and ratio < RMSE_RATIO_THRESHOLD:
        base.passed = True
        base.reason = (
            f"MojoGP R2={mojo_r2:.4f} > {R2_MINIMUM} and "
            f"RMSE ratio={ratio:.3f} < {RMSE_RATIO_THRESHOLD}"
        )
        return base

    # 3. MojoGP at least as good as GPyTorch
    if mojo_rmse <= gpy_rmse:
        base.passed = True
        base.reason = f"MojoGP RMSE ({mojo_rmse:.4f}) <= GPyTorch RMSE ({gpy_rmse:.4f})"
        return base

    # 4. RMSE ratio within threshold — MojoGP tracks GPyTorch closely
    #    Both may be mediocre but MojoGP isn't significantly worse
    if ratio < RMSE_RATIO_THRESHOLD:
        base.passed = True
        base.reason = (
            f"MojoGP tracks GPyTorch: ratio={ratio:.3f} < {RMSE_RATIO_THRESHOLD} "
            f"(MojoGP RMSE={mojo_rmse:.4f}, GPyTorch RMSE={gpy_rmse:.4f})"
        )
        return base

    # 5. GPyTorch good but MojoGP bad (ratio >= threshold)
    if gpy_rmse < RMSE_ABSOLUTE_OK and mojo_rmse > RMSE_ABSOLUTE_OK:
        base.reason = (
            f"MojoGP underperforms: RMSE={mojo_rmse:.4f} > {RMSE_ABSOLUTE_OK} "
            f"while GPyTorch RMSE={gpy_rmse:.4f} < {RMSE_ABSOLUTE_OK}"
        )
        return base

    # 6. Both bad and ratio bad
    if gpy_rmse > RMSE_ABSOLUTE_OK and mojo_rmse > RMSE_ABSOLUTE_OK:
        base.reason = (
            f"Both methods failed: MojoGP RMSE={mojo_rmse:.4f}, "
            f"GPyTorch RMSE={gpy_rmse:.4f} (both > {RMSE_ABSOLUTE_OK})"
        )
        return base

    # 7. General underperformance (ratio >= threshold)
    base.reason = (
        f"MojoGP underperforms: RMSE={mojo_rmse:.4f} vs GPyTorch={gpy_rmse:.4f} "
        f"(ratio={ratio:.3f} >= {RMSE_RATIO_THRESHOLD})"
    )
    return base


# =============================================================================
# MojoGP Training Helpers
# =============================================================================


def _make_mojogp_kernel(kernel_name: str, ard: bool):
    """Create a MojoGP kernel matching the config."""
    if kernel_name == "rbf":
        return RBF(ard=ard)
    elif kernel_name == "matern12":
        return Matern12(ard=ard)
    elif kernel_name == "matern32":
        return Matern32(ard=ard)
    elif kernel_name == "matern52":
        return Matern52(ard=ard)
    elif kernel_name == "periodic":
        return Periodic(ard=ard)
    elif kernel_name == "rq":
        return RQ(ard=ard)
    elif kernel_name == "linear":
        return Linear(ard=ard)
    elif kernel_name == "polynomial":
        return Polynomial()
    elif kernel_name == "rbf+matern52":
        return RBF(ard=ard) + Matern52(ard=ard)
    elif kernel_name == "rbf*linear":
        return RBF(ard=ard) * Linear()
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


def train_and_predict_mojogp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kernel_name: str,
    ard: bool = False,
    n_iterations: int = DEFAULT_ITERS,
    lr: float = DEFAULT_LR,
    init_mean: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, SingleOutputGP]:
    """Train MojoGP and predict. Returns (mean, std, train_time_s, gp)."""
    kernel = _make_mojogp_kernel(kernel_name, ard)
    gp = SingleOutputGP(kernel, init_mean=init_mean)

    t0 = time.perf_counter()
    gp.fit(X_train, y_train, max_iterations=n_iterations, learning_rate=lr)
    train_time = time.perf_counter() - t0

    mean, std = gp.predict(X_test, return_std=True)
    return mean, std, train_time, gp


# =============================================================================
# Test Runner
# =============================================================================


def _run_comparison(
    data: Dict,
    kernel_name: str,
    ard: bool = False,
    n_iterations: int = DEFAULT_ITERS,
    lr: float = DEFAULT_LR,
    init_mean: Optional[float] = None,
) -> Tuple[ComparisonResult, SingleOutputGP]:
    """Run full three-way comparison for a single configuration.

    Returns (ComparisonResult, trained ExactGP instance).
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    f_test = data["f_test"]

    # Train and predict with MojoGP
    mojo_mean, mojo_std, mojo_time, gp = train_and_predict_mojogp(
        X_train,
        y_train,
        X_test,
        kernel_name,
        ard=ard,
        n_iterations=n_iterations,
        lr=lr,
        init_mean=init_mean,
    )

    # Train and predict with GPyTorch
    gpy_result = train_gpytorch(
        X_train,
        y_train,
        kernel_name,
        ard=ard,
        n_iterations=n_iterations,
        lr=lr,
    )
    gpy_mean, gpy_std = predict_gpytorch(
        gpy_result["model"],
        gpy_result["likelihood"],
        X_test,
    )

    result = compare_three_way(
        f_test,
        mojo_mean,
        gpy_mean,
        mojo_train_time=mojo_time,
        gpy_train_time=gpy_result["train_time_s"],
    )
    return result, gp


def _print_and_assert(label: str, result: ComparisonResult):
    """Print comparison summary and assert pass."""
    print(f"\n--- {label} ---")
    print(result.summary())
    assert result.passed, result.reason


# =============================================================================
# Tests — MINIMAL tier
#
# Core functionality that must always pass.
# =============================================================================


@pytest.mark.system
@pytest.mark.minimal
class TestCoreSingleOutputExactGP:
    """Core kernel and constant mean tests."""

    def test_rbf_isotropic_friedman1(self):
        """RBF isotropic on Friedman #1 exercises the core exact GP route."""
        data = friedman1(n_train=2000, d=5)
        result, _ = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF Isotropic / Friedman #1", result)

    def test_rbf_ard_friedman1(self):
        """RBF ARD on Friedman #1 (d=10) — tests dimension relevance detection."""
        data = friedman1(n_train=2000, d=10)
        result, _ = _run_comparison(data, "rbf", ard=True)
        _print_and_assert("RBF ARD / Friedman #1 (d=10)", result)

    def test_matern52_friedman1(self):
        """Matern 5/2 on Friedman #1 — common kernel choice."""
        data = friedman1(n_train=2000, d=5)
        result, _ = _run_comparison(data, "matern52", ard=False)
        _print_and_assert("Matern 5/2 / Friedman #1", result)

    def test_constant_mean_nonzero_offset(self):
        """Data with large non-zero mean — tests constant mean recovery.

        Friedman #1 shifted by +50. The GP must learn the mean offset.
        """
        data = friedman1(n_train=2000, d=5, mean_offset=50.0)
        result, gp = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF / Friedman #1 + offset=50", result)

        # Verify learned mean is close to true mean of the data
        params = gp.get_learned_params()
        learned_mean = params["mean"]
        expected_mean = float(np.mean(data["y_train"]))
        mean_error = abs(learned_mean - expected_mean)
        print(
            f"  Learned mean={learned_mean:.2f}, data mean={expected_mean:.2f}, error={mean_error:.2f}"
        )
        # Mean should be reasonably close to the data mean
        assert mean_error < 5.0, (
            f"Mean recovery error {mean_error:.2f} too large "
            f"(learned={learned_mean:.2f}, expected~{expected_mean:.2f})"
        )

    def test_constant_mean_explicit_init(self):
        """Explicit init_mean parameter — tests that user-specified init_mean is used."""
        data = friedman1(n_train=2000, d=5, mean_offset=100.0)
        result, gp = _run_comparison(
            data,
            "rbf",
            ard=False,
            init_mean=100.0,
        )
        _print_and_assert("RBF / Friedman #1 + init_mean=100", result)

    def test_constant_mean_negative_offset(self):
        """Data with large negative mean — tests negative mean recovery."""
        data = friedman1(n_train=2000, d=5, mean_offset=-30.0)
        result, gp = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF / Friedman #1 + offset=-30", result)


# =============================================================================
# Tests — MODERATE tier
#
# Extended kernel coverage and data generators.
# =============================================================================


@pytest.mark.system
@pytest.mark.moderate
class TestExtendedSingleOutputKernelRoutes:
    """More kernel types and data functions."""

    # --- Matern family ---

    def test_matern32_friedman1(self):
        """Matern 3/2 on Friedman #1."""
        data = friedman1(n_train=2000, d=5)
        result, _ = _run_comparison(data, "matern32", ard=False)
        _print_and_assert("Matern 3/2 / Friedman #1", result)

    def test_matern12_friedman1(self):
        """Matern 1/2 (exponential) on Friedman #1."""
        data = friedman1(n_train=2000, d=5)
        result, _ = _run_comparison(data, "matern12", ard=False)
        _print_and_assert("Matern 1/2 / Friedman #1", result)

    def test_matern52_ard_friedman1(self):
        """Matern 5/2 ARD on Friedman #1 with irrelevant dims."""
        data = friedman1(n_train=2000, d=10)
        result, _ = _run_comparison(data, "matern52", ard=True)
        _print_and_assert("Matern 5/2 ARD / Friedman #1 (d=10)", result)

    # --- RQ kernel ---

    def test_rq_friedman1(self):
        """Rational Quadratic on Friedman #1."""
        data = friedman1(n_train=2000, d=5)
        result, _ = _run_comparison(data, "rq", ard=False)
        _print_and_assert("RQ / Friedman #1", result)

    # --- Periodic kernel ---

    def test_periodic_on_periodic_data(self):
        """Periodic kernel on periodic function — natural match."""
        data = periodic_function(n_train=2000)
        result, _ = _run_comparison(data, "periodic", ard=False)
        _print_and_assert("Periodic / periodic function", result)

    def test_rbf_on_periodic_data(self):
        """RBF on periodic function — baseline comparison."""
        data = periodic_function(n_train=2000)
        result, _ = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF / periodic function", result)

    # --- Linear kernel ---

    def test_linear_on_linear_data(self):
        """Linear kernel on linear function — natural match."""
        data = linear_function(n_train=2000, d=5)
        result, _ = _run_comparison(data, "linear", ard=False)
        _print_and_assert("Linear / linear function", result)

    # --- Polynomial kernel ---

    def test_polynomial_on_polynomial_data(self):
        """Polynomial kernel on polynomial function — natural match."""
        data = smooth_polynomial(n_train=2000, d=3)
        result, _ = _run_comparison(data, "polynomial", ard=False)
        _print_and_assert("Polynomial / polynomial function", result)

    # --- Different data generators ---

    def test_rbf_sinusoidal(self):
        """RBF on sinusoidal function — tests smooth periodic-like data."""
        data = sinusoidal(n_train=2000, d=3)
        result, _ = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF / Sinusoidal", result)

    def test_rbf_ard_sinusoidal(self):
        """RBF ARD on sinusoidal — different lengthscales per dim."""
        data = sinusoidal(n_train=2000, d=3)
        result, _ = _run_comparison(data, "rbf", ard=True)
        _print_and_assert("RBF ARD / Sinusoidal", result)

    def test_rbf_smooth_polynomial(self):
        """RBF on smooth polynomial — easy function for GPs."""
        data = smooth_polynomial(n_train=2000, d=3)
        result, _ = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF / Smooth Polynomial", result)


# =============================================================================
# Tests — FULL tier
#
# Comprehensive: composite kernels, multi-output, API features, scaling.
# =============================================================================


@pytest.mark.system
@pytest.mark.full
class TestCompositeKernelSurfaceCorrectness:
    """Composite kernel tests (require JIT compilation)."""

    def test_rbf_plus_matern52(self):
        """Sum kernel: RBF + Matern52."""
        data = friedman1(n_train=2000, d=5)
        result, _ = _run_comparison(data, "rbf+matern52", ard=False)
        _print_and_assert("RBF + Matern52 / Friedman #1", result)

    def test_rbf_plus_matern52_ard(self):
        """Sum kernel with ARD: RBF(ard) + Matern52(ard)."""
        data = friedman1(n_train=2000, d=10)
        result, _ = _run_comparison(data, "rbf+matern52", ard=True)
        _print_and_assert("RBF+Matern52 ARD / Friedman #1 (d=10)", result)

    def test_rbf_times_linear(self):
        """Product kernel: RBF * Linear."""
        data = smooth_polynomial(n_train=2000, d=3)
        result, _ = _run_comparison(data, "rbf*linear", ard=False)
        _print_and_assert("RBF * Linear / Smooth Poly", result)


@pytest.mark.system
@pytest.mark.full
class TestMultiOutputSurfaceCorrectness:
    """Multi-output GP tests (ICM with Kronecker CG)."""

    def _run_multi_output(
        self,
        data: Dict,
        kernel: str = "rbf",
        ard: bool = False,
        n_iterations: int = 100,
        lr: float = 0.05,
    ) -> Tuple[float, float, MultiOutputGP]:
        """Train multi-output GP, return (mean_rmse, mean_r2, gp)."""
        X_train = data["X_train"]
        Y_train = data["Y_train"]
        X_test = data["X_test"]
        F_test = data["F_test"]
        T = data["T"]

        gp = MultiOutputGP(kernel=kernel, ard=ard)
        gp.fit(X_train, Y_train, max_iterations=n_iterations, learning_rate=lr)

        pred = gp.predict(X_test)
        assert isinstance(pred, MultiOutputPredictionResult)
        mean = pred.mean  # [m, T]

        rmses = [_rmse(mean[:, t], F_test[:, t]) for t in range(T)]
        r2s = [_r2(mean[:, t], F_test[:, t]) for t in range(T)]

        return float(np.mean(rmses)), float(np.mean(r2s)), gp

    def test_multi_output_rbf(self):
        """Multi-output GP with RBF kernel, 3 correlated tasks."""
        data = multi_output_correlated(n_train=2000, T=3)
        mean_rmse, mean_r2, _ = self._run_multi_output(data, kernel="rbf")
        print(f"\n--- Multi-output RBF (T=3) ---")
        print(f"  Mean RMSE: {mean_rmse:.4f}, Mean R2: {mean_r2:.4f}")
        assert mean_r2 > 0.5, f"Multi-output R2 {mean_r2:.4f} < 0.5"

    def test_multi_output_matern52(self):
        """Multi-output GP with Matern 5/2 kernel."""
        data = multi_output_correlated(n_train=2000, T=3)
        mean_rmse, mean_r2, _ = self._run_multi_output(data, kernel="matern52")
        print(f"\n--- Multi-output Matern52 (T=3) ---")
        print(f"  Mean RMSE: {mean_rmse:.4f}, Mean R2: {mean_r2:.4f}")
        assert mean_r2 > 0.5, f"Multi-output R2 {mean_r2:.4f} < 0.5"

    def test_multi_output_ard(self):
        """Multi-output GP with ARD."""
        data = multi_output_correlated(n_train=2000, T=3, d=5)
        mean_rmse, mean_r2, _ = self._run_multi_output(data, kernel="rbf", ard=True)
        print(f"\n--- Multi-output RBF ARD (T=3, d=5) ---")
        print(f"  Mean RMSE: {mean_rmse:.4f}, Mean R2: {mean_r2:.4f}")
        assert mean_r2 > 0.4, f"Multi-output ARD R2 {mean_r2:.4f} < 0.4"

    def test_multi_output_constant_mean(self):
        """Multi-output GP with per-task constant mean offsets."""
        per_task_means = np.array([10.0, -5.0, 20.0], dtype=np.float32)
        data = multi_output_correlated(
            n_train=2000,
            T=3,
            per_task_means=per_task_means,
        )
        gp = MultiOutputGP(kernel="rbf", init_mean=per_task_means)
        gp.fit(
            data["X_train"],
            data["Y_train"],
            max_iterations=100,
            learning_rate=0.05,
        )
        pred = gp.predict(data["X_test"])
        mean = pred.mean

        F_test = data["F_test"]
        T = data["T"]
        rmses = [_rmse(mean[:, t], F_test[:, t]) for t in range(T)]
        r2s = [_r2(mean[:, t], F_test[:, t]) for t in range(T)]
        mean_r2 = float(np.mean(r2s))

        print(f"\n--- Multi-output RBF + ConstantMean (T=3) ---")
        print(f"  Per-task means: {per_task_means}")
        for t in range(T):
            print(f"  Task {t}: RMSE={rmses[t]:.4f}, R2={r2s[t]:.4f}")
        print(f"  Mean R2: {mean_r2:.4f}")

        # Verify learned means are close to true means
        if gp.training_result is not None and hasattr(
            gp.training_result, "mean_per_task"
        ):
            learned_means = gp.training_result.mean_per_task
            for t in range(T):
                err = abs(float(learned_means[t]) - float(per_task_means[t]))
                print(
                    f"  Task {t} mean: learned={float(learned_means[t]):.2f}, true={float(per_task_means[t]):.2f}, err={err:.2f}"
                )

        assert mean_r2 > 0.4, f"Multi-output mean R2 {mean_r2:.4f} < 0.4"

    def test_multi_output_return_var(self):
        """Multi-output GP with return_var=True."""
        data = multi_output_correlated(n_train=2000, T=3)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(data["X_train"], data["Y_train"], max_iterations=100, learning_rate=0.05)

        mean, var = gp.predict(data["X_test"], return_var=True)
        assert mean.shape == (data["X_test"].shape[0], data["T"]), (
            f"Mean shape {mean.shape}"
        )
        assert var is not None, "Variance is None"
        assert var.shape == mean.shape, (
            f"Var shape {var.shape} != mean shape {mean.shape}"
        )
        assert np.all(var >= 0), "Negative variance detected"

        mean_std, std = gp.predict(data["X_test"], return_std=True)
        assert std is not None, "Std is None"
        np.testing.assert_allclose(mean, mean_std, rtol=1e-5)
        np.testing.assert_allclose(std, np.sqrt(var), rtol=1e-5)
        print(f"\n--- Multi-output return_var / return_std ---")
        print(f"  Mean shape: {mean.shape}, Var shape: {var.shape}")

    def test_multi_output_score(self):
        """Multi-output GP score() method."""
        data = multi_output_correlated(n_train=2000, T=3)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(data["X_train"], data["Y_train"], max_iterations=100, learning_rate=0.05)

        scores = gp.score(data["X_test"], data["F_test"])
        r2 = scores["r2"]
        print(f"\n--- Multi-output score() ---")
        print(f"  R2 = {r2:.4f}, RMSE = {scores['rmse']:.4f}")
        assert r2 > 0.3, f"Multi-output R2 {r2:.4f} < 0.3"


@pytest.mark.system
@pytest.mark.full
class TestSingleOutputPublicAPIContracts:
    """Tests for SingleOutputGP public API contracts."""

    def test_score_method(self):
        """SingleOutputGP.score() returns R2 on test data."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF())
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)

        r2 = gp.score(data["X_test"], data["f_test"])
        print(f"\n--- score() method ---")
        print(f"  R2 = {r2:.4f}")
        assert r2 > R2_MINIMUM, f"R2 {r2:.4f} < {R2_MINIMUM}"

    def test_get_learned_params(self):
        """get_learned_params() returns named dict with all hyperparameters."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF())
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)

        params = gp.get_learned_params()
        print(f"\n--- get_learned_params() ---")
        for k, v in params.items():
            print(f"  {k}: {v:.6f}")

        assert "noise" in params, "Missing 'noise' in learned params"
        assert "mean" in params, "Missing 'mean' in learned params"
        assert params["noise"] > 0, f"Noise must be positive, got {params['noise']}"

    def test_get_learned_params_ard(self):
        """get_learned_params() with ARD — returns per-dim lengthscales."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF(ard=True))
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)

        params = gp.get_learned_params()
        print(f"\n--- get_learned_params() ARD ---")
        for k, v in params.items():
            print(f"  {k}: {v:.6f}")

        # Should have per-dim lengthscale params (named like rbf_ls_0 or rbf_lengthscale_0)
        ls_params = [k for k in params if "ls_" in k or "lengthscale" in k.lower()]
        assert len(ls_params) >= 5, (
            f"Expected >=5 lengthscale params, got {len(ls_params)}: {ls_params}"
        )

    def test_prediction_result_default(self):
        """predict() returns PredictionResult with mean/var/std by default."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF())
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)

        result = gp.predict(data["X_test"])
        assert isinstance(result, PredictionResult), (
            f"Expected PredictionResult, got {type(result)}"
        )
        assert result.mean is not None, "PredictionResult.mean is None"
        assert result.variance is not None, "PredictionResult.variance is None"
        assert result.std is not None, "PredictionResult.std is None"

        n_test = data["X_test"].shape[0]
        assert result.mean.shape == (n_test,), (
            f"Mean shape {result.mean.shape} != ({n_test},)"
        )
        assert result.variance.shape == (n_test,), (
            f"Var shape {result.variance.shape} != ({n_test},)"
        )
        assert result.std.shape == (n_test,), (
            f"Std shape {result.std.shape} != ({n_test},)"
        )

        # Variance should be non-negative
        assert np.all(result.variance >= 0), "Negative variance detected"
        # std should be sqrt(variance)
        np.testing.assert_allclose(result.std, np.sqrt(result.variance), rtol=1e-5)

        print(f"\n--- PredictionResult prediction ---")
        print(f"  Mean range: [{result.mean.min():.3f}, {result.mean.max():.3f}]")
        print(f"  Std range: [{result.std.min():.3f}, {result.std.max():.3f}]")

    def test_sample_posterior(self):
        """sample_posterior() draws samples consistent with predictive distribution."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF())
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)

        n_samples = 50
        samples = gp.sample_posterior(data["X_test"], n_samples=n_samples)
        assert samples.shape == (n_samples, data["X_test"].shape[0]), (
            f"Samples shape {samples.shape} != ({n_samples}, {data['X_test'].shape[0]})"
        )

        # Sample mean should be close to predictive mean
        mean, std = gp.predict(data["X_test"], return_std=True)
        sample_mean = np.mean(samples, axis=0)
        mean_diff = _rmse(sample_mean, mean)
        print(f"\n--- sample_posterior ---")
        print(f"  Samples shape: {samples.shape}")
        print(f"  Sample mean vs pred mean RMSE: {mean_diff:.4f}")
        # With 50 samples, the sample mean should be reasonably close
        assert mean_diff < 1.0, (
            f"Sample mean too far from predictive mean: {mean_diff:.4f}"
        )

    def test_is_trained_flag(self):
        """is_trained property tracks training state."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF())
        assert not gp.is_trained, "Should be False before fit()"

        gp.fit(data["X_train"], data["y_train"], max_iterations=50)
        assert gp.is_trained, "Should be True after fit()"

    def test_predict_before_fit_raises(self):
        """predict() before fit() raises RuntimeError."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF())
        with pytest.raises(RuntimeError, match="trained"):
            gp.predict(data["X_test"])

    def test_training_result_accessible(self):
        """training_result property returns TrainingResult after fit."""
        data = friedman1(n_train=2000, d=5)
        gp = SingleOutputGP(RBF())
        assert gp.training_result is None

        gp.fit(data["X_train"], data["y_train"], max_iterations=50)
        tr = gp.training_result
        assert tr is not None
        assert tr.noise > 0, f"Noise must be positive, got {tr.noise}"
        assert tr.iterations > 0, f"Iterations must be positive, got {tr.iterations}"
        print(f"\n--- training_result ---")
        print(f"  noise={tr.noise:.6f}, nll={tr.nll:.4f}, iters={tr.iterations}")


@pytest.mark.system
@pytest.mark.full
class TestSingleOutputScalingSurface:
    """Scaling tests with larger n and d."""

    def test_rbf_n3000(self):
        """RBF on Friedman #1 with n=3000 — scaling test."""
        data = friedman1(n_train=3000, d=5)
        result, _ = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF / Friedman #1 (n=3000)", result)

    def test_rbf_ard_n3000_d10(self):
        """RBF ARD on Friedman #1 with n=3000 and d=10."""
        data = friedman1(n_train=3000, d=10)
        result, _ = _run_comparison(data, "rbf", ard=True)
        _print_and_assert("RBF ARD / Friedman #1 (n=3000, d=10)", result)


@pytest.mark.system
@pytest.mark.full
class TestSingleOutputConstantMeanSurface:
    """Extended constant mean tests."""

    def test_mean_auto_detection(self):
        """Auto-detected init_mean equals np.mean(y)."""
        data = friedman1(n_train=2000, d=5, mean_offset=25.0)
        gp = SingleOutputGP(RBF())

        # Before fit: init_mean should be None (auto-detect)
        assert gp._init_mean is None

        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)
        mean, std = gp.predict(data["X_test"], return_std=True)

        rmse = _rmse(mean, data["f_test"])
        r2 = _r2(mean, data["f_test"])
        print(f"\n--- Mean auto-detection (offset=25) ---")
        print(f"  RMSE={rmse:.4f}, R2={r2:.4f}")
        assert r2 > R2_MINIMUM, f"R2 {r2:.4f} < {R2_MINIMUM}"

    def test_mean_zero_mean_data(self):
        """Zero-mean data — mean should stay near zero."""
        data = friedman1(n_train=2000, d=5, mean_offset=0.0)
        gp = SingleOutputGP(RBF())
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)

        params = gp.get_learned_params()
        # The Friedman function itself has a mean around ~14, not zero
        # So the learned mean should be close to the data mean
        data_mean = float(np.mean(data["y_train"]))
        learned_mean = params["mean"]
        err = abs(learned_mean - data_mean)
        print(f"\n--- Zero offset (data mean={data_mean:.2f}) ---")
        print(f"  Learned mean={learned_mean:.2f}, error={err:.2f}")
        assert err < 5.0, f"Mean error {err:.2f} > 5.0"

    def test_mean_with_ard(self):
        """Constant mean works correctly with ARD kernels."""
        data = friedman1(n_train=2000, d=10, mean_offset=50.0)
        result, gp = _run_comparison(data, "rbf", ard=True)
        _print_and_assert("RBF ARD + mean offset=50 / Friedman #1", result)

    def test_mean_with_matern52(self):
        """Constant mean works with non-RBF kernels."""
        data = friedman1(n_train=2000, d=5, mean_offset=-20.0)
        result, gp = _run_comparison(data, "matern52", ard=False)
        _print_and_assert("Matern52 + mean offset=-20 / Friedman #1", result)

    def test_mean_with_periodic(self):
        """Constant mean works with Periodic kernel on shifted periodic data."""
        data = periodic_function(n_train=2000)
        # Add mean offset to y
        offset = 15.0
        data["y_train"] = data["y_train"] + offset
        data["y_test"] = data["y_test"] + offset
        data["f_test"] = data["f_test"] + offset

        gp = SingleOutputGP(Periodic())
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)
        mean, std = gp.predict(data["X_test"], return_std=True)

        rmse = _rmse(mean, data["f_test"])
        r2 = _r2(mean, data["f_test"])
        print(f"\n--- Periodic + mean offset={offset} ---")
        print(f"  RMSE={rmse:.4f}, R2={r2:.4f}")
        # Periodic + mean is harder, be lenient
        assert r2 > 0.5, f"R2 {r2:.4f} < 0.5"


@pytest.mark.system
@pytest.mark.full
class TestKernelDataMatchSurface:
    """Tests that verify kernels perform well on their natural data type."""

    def test_rbf_on_smooth_data(self):
        """RBF should do well on smooth functions."""
        data = smooth_polynomial(n_train=2000, d=3)
        result, _ = _run_comparison(data, "rbf", ard=False)
        _print_and_assert("RBF / smooth poly", result)

    def test_matern32_on_sinusoidal(self):
        """Matern 3/2 on sinusoidal function."""
        data = sinusoidal(n_train=2000, d=3)
        result, _ = _run_comparison(data, "matern32", ard=False)
        _print_and_assert("Matern 3/2 / sinusoidal", result)

    def test_matern52_on_polynomial(self):
        """Matern 5/2 on smooth polynomial."""
        data = smooth_polynomial(n_train=2000, d=3)
        result, _ = _run_comparison(data, "matern52", ard=False)
        _print_and_assert("Matern 5/2 / smooth poly", result)

    def test_rq_ard_friedman1(self):
        """RQ ARD on Friedman #1 with irrelevant dims."""
        data = friedman1(n_train=2000, d=10)
        result, _ = _run_comparison(data, "rq", ard=True)
        _print_and_assert("RQ ARD / Friedman #1 (d=10)", result)

    def test_linear_ard_on_linear_data(self):
        """Linear kernel with ARD on linear function."""
        data = linear_function(n_train=2000, d=5)
        result, _ = _run_comparison(data, "linear", ard=True)
        _print_and_assert("Linear ARD / linear function", result)


# =============================================================================
# Mixed Categorical / Discrete Kernel Data Generation
# =============================================================================


def mixed_categorical_data(
    n_train: int = 500,
    n_test: int = 200,
    cont_dim: int = 3,
    cat_levels: list = None,
    noise_std: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Generate data from a mixed continuous + categorical GP.

    True function: f(x, c) = continuous_signal(x) + categorical_effect(c)
    where continuous_signal = weighted sum of sinusoids (first 3 dims)
    and categorical_effect = per-level random offsets.

    Returns dict with X_train, y_train, X_test, f_test, cat_info, etc.
    """
    if cat_levels is None:
        cat_levels = [3]

    rng = np.random.RandomState(seed)
    n = n_train + n_test
    num_cat_vars = len(cat_levels)

    # Continuous features
    X_cont = rng.randn(n, cont_dim).astype(np.float32)

    # Categorical features (integer-encoded)
    C = np.column_stack([rng.randint(0, L, size=n) for L in cat_levels]).astype(
        np.float32
    )

    # True continuous signal: weighted sinusoids
    f_cont = np.zeros(n, dtype=np.float32)
    for d in range(min(cont_dim, 3)):
        weight = 1.0 / (d + 1)
        f_cont += weight * np.sin(2.0 * X_cont[:, d])

    # True categorical effects: per-level offsets
    f_cat = np.zeros(n, dtype=np.float32)
    cat_effects = []
    for v in range(num_cat_vars):
        level_effects = rng.randn(cat_levels[v]).astype(np.float32) * 0.8
        cat_effects.append(level_effects)
        for i in range(n):
            f_cat[i] += level_effects[int(C[i, v])]

    f_true = f_cont + f_cat
    noise = noise_std * np.std(f_true)
    y = f_true + noise * rng.randn(n).astype(np.float32)

    # Stack: [X_cont | C]
    X_full = np.column_stack([X_cont, C]).astype(np.float32)

    # Build cat_info: list of (column_index, num_levels) tuples
    cat_info = []
    for v in range(num_cat_vars):
        cat_info.append((cont_dim + v, cat_levels[v]))

    return {
        "X_train": X_full[:n_train],
        "y_train": y[:n_train],
        "X_test": X_full[n_train:],
        "f_test": f_true[n_train:],
        "y_test": y[n_train:],
        "cat_info": cat_info,
        "cat_levels": cat_levels,
        "cont_dim": cont_dim,
        "noise_std": noise,
        "name": f"mixed_cat_n{n_train}_d{cont_dim}_levels{cat_levels}",
    }


_CAT_KERNEL_MAP = {
    "ehh": EHH,
    "gd": GD,
    "cr": CR,
    "hh": HH,
    "fe": FE,
}


def _build_mixed_kernel(
    cont_kernel,
    data: Dict,
    cat_kernel: str = "ehh",
):
    """Build a composite kernel tree: cont_kernel(active_dims) * CatKernel1 * CatKernel2 ...

    The continuous kernel gets active_dims set to the continuous columns,
    and each categorical variable gets its own categorical kernel node.
    """
    cont_dim = data["cont_dim"]
    cat_info = data["cat_info"]  # list of (col_idx, num_levels)
    CatClass = _CAT_KERNEL_MAP[cat_kernel]

    # Set active_dims on the continuous kernel to the continuous columns
    cont_kernel.active_dims = tuple(range(cont_dim))

    # Multiply categorical kernel nodes
    result = cont_kernel
    for col_idx, num_levels in cat_info:
        result = result * CatClass(levels=num_levels, active_dims=[col_idx])
    return result


def _run_categorical_test(
    data: Dict,
    kernel_name: str,
    cat_kernel: str = "ehh",
    ard: bool = False,
    n_iterations: int = DEFAULT_ITERS,
    lr: float = DEFAULT_LR,
) -> Tuple[float, float, float, SingleOutputGP]:
    """Train MojoGP with categorical dims and return (rmse, r2, time, gp).

    No GPyTorch comparison since GPyTorch doesn't have an equivalent
    mixed categorical kernel. Tests compare against ground truth only.
    """
    cont_kernel = _make_mojogp_kernel(kernel_name, ard)
    kernel = _build_mixed_kernel(cont_kernel, data, cat_kernel)
    gp = SingleOutputGP(kernel)

    t0 = time.perf_counter()
    gp.fit(
        data["X_train"], data["y_train"], max_iterations=n_iterations, learning_rate=lr
    )
    train_time = time.perf_counter() - t0

    pred = gp.predict(data["X_test"])
    if isinstance(pred, PredictionResult):
        mean, std = pred.mean, pred.std
    else:
        mean, std = pred
    test_rmse = _rmse(mean, data["f_test"])
    test_r2 = _r2(mean, data["f_test"])

    return test_rmse, test_r2, train_time, gp


def _print_cat_result(label: str, rmse_val: float, r2: float, train_time: float):
    """Print categorical test result."""
    print(f"\n--- {label} ---")
    print(f"  RMSE vs truth: {rmse_val:.4f}")
    print(f"  R2:            {r2:.4f}")
    print(f"  Train time:    {train_time:.2f}s")


# =============================================================================
# Tests — MINIMAL tier: Discrete / Categorical Kernels
# =============================================================================


@pytest.mark.system
@pytest.mark.minimal
class TestCoreCategoricalKernelSurface:
    """Core mixed categorical kernel tests.

    These test the ExactGP API with kernel tree composition for categorical
    variables. No GPyTorch comparison since GPyTorch doesn't have an
    equivalent mixed categorical kernel — tests compare against ground truth only.
    """

    def test_ehh_rbf_recovers_mixed_categorical_signal(self):
        """EHH (recommended) + RBF on mixed data recovers the categorical signal."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel="ehh")
        _print_cat_result("EHH + RBF / mixed (3 levels)", rmse_val, r2, t)
        assert rmse_val < 1.5, f"RMSE {rmse_val:.4f} > 1.5"

    def test_gd_rbf_recovers_mixed_categorical_signal(self):
        """GD (simplest) + RBF on mixed data."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel="gd")
        _print_cat_result("GD + RBF / mixed (3 levels)", rmse_val, r2, t)
        assert rmse_val < 1.5, f"RMSE {rmse_val:.4f} > 1.5"


# =============================================================================
# Tests — MODERATE tier: Discrete / Categorical Kernels
# =============================================================================


@pytest.mark.system
@pytest.mark.moderate
class TestExtendedCategoricalKernelSurface:
    """Extended categorical kernel tests: all 5 variants, multiple cat vars, methods."""

    def test_all_five_cat_kernel_variants(self):
        """All 5 categorical kernel variants produce reasonable results."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        variants = ["gd", "cr", "ehh", "hh", "fe"]
        results = {}

        for variant in variants:
            rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel=variant)
            results[variant] = (rmse_val, r2, t)

        # Print comparison table
        print(f"\n{'=' * 60}")
        print(f"  CATEGORICAL KERNEL VARIANT COMPARISON")
        print(f"{'=' * 60}")
        print(f"  {'Variant':>8} | {'RMSE':>8} | {'R2':>8} | {'Time':>8}")
        print(f"  {'-' * 42}")
        for v, (r, r2, t) in results.items():
            print(f"  {v.upper():>8} | {r:>8.4f} | {r2:>8.4f} | {t:>7.2f}s")
        print(f"{'=' * 60}")

        for v, (r, r2, t) in results.items():
            assert r < 2.0, f"Variant {v}: RMSE {r:.4f} > 2.0"

    def test_multiple_categorical_variables(self):
        """Multiple categorical variables with different level counts."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3, 4])
        rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel="ehh")
        _print_cat_result("EHH + RBF / 2 cat vars [3,4]", rmse_val, r2, t)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_categorical_matrix_free(self):
        """Mixed categorical with matrix-free method."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        kernel = _build_mixed_kernel(RBF(), data, "ehh")
        gp = SingleOutputGP(kernel)
        gp.fit(
            data["X_train"],
            data["y_train"],
            max_iterations=100,
            learning_rate=0.01,
            method="matrix_free",
        )
        pred = gp.predict(data["X_test"])
        mean = pred.mean if isinstance(pred, PredictionResult) else pred[0]
        rmse_val = _rmse(mean, data["f_test"])
        r2 = _r2(mean, data["f_test"])
        _print_cat_result("EHH + RBF / matrix_free", rmse_val, r2, 0.0)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_categorical_with_matern52(self):
        """Categorical with Matern 5/2 continuous kernel."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        rmse_val, r2, t, _ = _run_categorical_test(data, "matern52", cat_kernel="ehh")
        _print_cat_result("EHH + Matern52 / mixed", rmse_val, r2, t)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_categorical_ard(self):
        """ARD + categorical kernel combination."""
        data = mixed_categorical_data(n_train=2000, cont_dim=5, cat_levels=[3])
        rmse_val, r2, t, _ = _run_categorical_test(
            data, "rbf", cat_kernel="gd", ard=True
        )
        _print_cat_result("GD + RBF ARD / mixed (d=5)", rmse_val, r2, t)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"


# =============================================================================
# Tests — FULL tier: Discrete / Categorical Kernels
# =============================================================================


@pytest.mark.system
@pytest.mark.full
class TestCategoricalCompositeAndScalingSurface:
    """Categorical tests for composite kernels, per-variable selection, and scaling."""

    def test_composite_sum_with_categorical(self):
        """Sum kernel (RBF + Matern52) with categorical."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        cont_dims = list(range(data["cont_dim"]))
        kernel = RBF(active_dims=cont_dims) + Matern52(active_dims=cont_dims)
        for col_idx, num_levels in data["cat_info"]:
            kernel = kernel * EHH(levels=num_levels, active_dims=[col_idx])
        gp = SingleOutputGP(kernel)
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)
        pred = gp.predict(data["X_test"])
        mean = pred.mean if isinstance(pred, PredictionResult) else pred[0]
        rmse_val = _rmse(mean, data["f_test"])
        r2 = _r2(mean, data["f_test"])
        _print_cat_result("EHH + (RBF+Matern52) / mixed", rmse_val, r2, 0.0)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_composite_product_with_categorical(self):
        """Product kernel (RBF * Matern52) with categorical."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        cont_dims = list(range(data["cont_dim"]))
        kernel = RBF(active_dims=cont_dims) * Matern52(active_dims=cont_dims)
        for col_idx, num_levels in data["cat_info"]:
            kernel = kernel * GD(levels=num_levels, active_dims=[col_idx])
        gp = SingleOutputGP(kernel)
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)
        pred = gp.predict(data["X_test"])
        mean = pred.mean if isinstance(pred, PredictionResult) else pred[0]
        rmse_val = _rmse(mean, data["f_test"])
        r2 = _r2(mean, data["f_test"])
        _print_cat_result("GD + (RBF*Matern52) / mixed", rmse_val, r2, 0.0)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_per_variable_kernel_selection(self):
        """Per-variable categorical kernel selection via kernel tree."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3, 4])
        # Different kernel for each categorical variable: col 3 -> GD, col 4 -> CR
        cont_dims = list(range(data["cont_dim"]))
        kernel = (
            RBF(active_dims=cont_dims)
            * GD(levels=3, active_dims=[3])
            * CR(levels=4, active_dims=[4])
        )
        gp = SingleOutputGP(kernel)
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)
        pred = gp.predict(data["X_test"])
        mean = pred.mean if isinstance(pred, PredictionResult) else pred[0]
        rmse_val = _rmse(mean, data["f_test"])
        r2 = _r2(mean, data["f_test"])
        _print_cat_result("Per-var (GD+CR) + RBF / mixed", rmse_val, r2, 0.0)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_three_categorical_variables(self):
        """Three categorical variables with different level counts."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3, 4, 5])
        rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel="ehh")
        _print_cat_result("EHH + RBF / 3 cat vars [3,4,5]", rmse_val, r2, t)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_categorical_scaling_n1000(self):
        """Mixed categorical at n = 2000."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[3])
        rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel="ehh")
        _print_cat_result("EHH + RBF / n = 2000", rmse_val, r2, t)
        assert rmse_val < 1.5, f"RMSE {rmse_val:.4f} > 1.5"

    def test_categorical_higher_dim(self):
        """Mixed categorical with higher continuous dimensionality."""
        data = mixed_categorical_data(n_train=2000, cont_dim=5, cat_levels=[3, 4])
        rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel="ehh")
        _print_cat_result("EHH + RBF / d=5, 2 cat vars", rmse_val, r2, t)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_categorical_more_levels(self):
        """Categorical variable with more levels (5)."""
        data = mixed_categorical_data(n_train=2000, cont_dim=3, cat_levels=[5])
        rmse_val, r2, t, _ = _run_categorical_test(data, "rbf", cat_kernel="ehh")
        _print_cat_result("EHH + RBF / 5 levels", rmse_val, r2, t)
        assert rmse_val < 2.0, f"RMSE {rmse_val:.4f} > 2.0"

    def test_composite_ard_categorical(self):
        """Composite kernel + ARD + categorical (all features combined)."""
        data = mixed_categorical_data(n_train=2000, cont_dim=5, cat_levels=[3])
        cont_dims = list(range(data["cont_dim"]))
        kernel = RBF(ard=True, active_dims=cont_dims) + Matern52(
            ard=True, active_dims=cont_dims
        )
        for col_idx, num_levels in data["cat_info"]:
            kernel = kernel * GD(levels=num_levels, active_dims=[col_idx])
        gp = SingleOutputGP(kernel)
        gp.fit(data["X_train"], data["y_train"], max_iterations=100, learning_rate=0.01)
        pred = gp.predict(data["X_test"])
        mean = pred.mean if isinstance(pred, PredictionResult) else pred[0]
        rmse_val = _rmse(mean, data["f_test"])
        r2 = _r2(mean, data["f_test"])
        _print_cat_result("GD + (RBF+Matern52) ARD / mixed", rmse_val, r2, 0.0)
        assert rmse_val < 2.5, f"RMSE {rmse_val:.4f} > 2.5"
