"""GPyTorch model definitions and training helpers for system benchmarks.

Provides unified interfaces for training and prediction with GPyTorch,
supporting both standard CG and KeOps modes.
"""

import time
import tracemalloc
import importlib
import numpy as np
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from contextlib import nullcontext

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import (
    ScaleKernel,
    RBFKernel,
    MaternKernel,
    PeriodicKernel,
    RQKernel,
    LinearKernel,
    PolynomialKernel,
)
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from .gpu_memory import (
    GPUMemoryMonitor,
    get_torch_memory_stats,
    measure_gpu_phase,
    reset_torch_memory_stats,
)
from .result_types import (
    AccuracyResult,
    SpeedResult,
    MemoryResult,
    HyperparameterResult,
    BenchmarkResult,
)
from .metrics import compute_all_accuracy_metrics, param_relative_error


def _direct_iteration_timing_payload(iter_times_s: List[float]) -> Dict[str, Any]:
    iter_times_ms = [float(t * 1000.0) for t in iter_times_s]
    if not iter_times_ms:
        return {
            "iter_times_ms": [],
            "iter_time_min_ms": 0.0,
            "iter_time_q25_ms": 0.0,
            "iter_time_mean_ms": 0.0,
            "iter_time_median_ms": 0.0,
            "iter_time_q75_ms": 0.0,
            "iter_time_max_ms": 0.0,
            "iter_time_p5_ms": 0.0,
            "iter_time_p95_ms": 0.0,
        }
    return {
        "iter_times_ms": iter_times_ms,
        "iter_time_min_ms": float(np.min(iter_times_ms)),
        "iter_time_q25_ms": float(np.percentile(iter_times_ms, 25)),
        "iter_time_mean_ms": float(np.mean(iter_times_ms)),
        "iter_time_median_ms": float(np.median(iter_times_ms)),
        "iter_time_q75_ms": float(np.percentile(iter_times_ms, 75)),
        "iter_time_max_ms": float(np.max(iter_times_ms)),
        "iter_time_p5_ms": float(np.percentile(iter_times_ms, 5)),
        "iter_time_p95_ms": float(np.percentile(iter_times_ms, 95)),
    }


def _requested_effective_backend_payload(
    *,
    requested_mode: str,
    effective_mode: str,
    prediction_mode: str | None = None,
    backend_fallback_reason: str | None = None,
) -> Dict[str, Any]:
    backend_fallback_used = requested_mode != effective_mode or backend_fallback_reason is not None
    payload: Dict[str, Any] = {
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "backend_fallback_used": backend_fallback_used,
        "backend_fallback_reason": backend_fallback_reason,
    }
    if prediction_mode is not None:
        payload["effective_prediction_mode"] = prediction_mode
    return payload


def merge_gpytorch_benchmark_memory(
    training_memory_stats: Dict[str, Any],
    prediction_memory_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge GPyTorch training/prediction telemetry into benchmark memory fields."""

    train = dict(training_memory_stats or {})
    pred = dict(prediction_memory_stats or {})
    merged = dict(train)

    train_peak = train.get("max_mb")
    if train_peak is not None:
        merged["training_peak_gpu_mb"] = float(train_peak)
    train_delta = train.get("phase_delta_gpu_mb") or train.get("delta_mb")
    if train_delta is None and train.get("min_mb") is not None and train_peak is not None:
        train_delta = float(train_peak) - float(train.get("min_mb", 0.0))
    if train_delta is not None:
        merged["training_delta_gpu_mb"] = max(float(train_delta), 0.0)

    prediction_peak = pred.get("prediction_peak_gpu_mb")
    prediction_delta = pred.get("prediction_delta_gpu_mb")
    if prediction_peak is None:
        prediction_peak = pred.get("phase_peak_gpu_mb", pred.get("max_mb"))
    if prediction_delta is None:
        prediction_delta = pred.get("phase_delta_gpu_mb", pred.get("delta_mb"))
    if prediction_peak is not None:
        merged["prediction_peak_gpu_mb"] = float(prediction_peak)
    if prediction_delta is not None:
        merged["prediction_delta_gpu_mb"] = float(prediction_delta)

    exact_peak = pred.get("exact_prediction_peak_gpu_mb")
    exact_delta = pred.get("exact_prediction_delta_gpu_mb")
    love_peak = pred.get("love_prediction_peak_gpu_mb")
    love_delta = pred.get("love_prediction_delta_gpu_mb")
    if exact_peak is not None:
        merged["exact_prediction_peak_gpu_mb"] = float(exact_peak)
    if exact_delta is not None:
        merged["exact_prediction_delta_gpu_mb"] = float(exact_delta)
    if love_peak is not None:
        merged["love_prediction_peak_gpu_mb"] = float(love_peak)
    if love_delta is not None:
        merged["love_prediction_delta_gpu_mb"] = float(love_delta)

    merged["max_mb"] = max(
        float(train.get("max_mb", 0.0)),
        float(pred.get("prediction_peak_gpu_mb", 0.0)),
        float(pred.get("max_mb", 0.0)),
    )
    merged["mean_mb"] = max(
        float(merged.get("mean_mb", 0.0)),
        float(merged.get("max_mb", 0.0)),
    )
    merged["torch_peak_mb"] = max(
        float(train.get("torch_peak_mb", 0.0)),
        float(pred.get("torch_peak_mb", 0.0)),
    )
    if pred.get("torch_current_mb") is not None:
        merged["torch_current_mb"] = float(pred["torch_current_mb"])
    merged["samples"] = int(train.get("samples", 0)) + int(pred.get("samples", 0))
    merged["method"] = str(pred.get("method") or train.get("method") or "none")
    return merged


# Try to import KeOps kernels
try:
    from gpytorch.kernels.keops import RBFKernel as KeOpsRBFKernel
    from gpytorch.kernels.keops import MaternKernel as KeOpsMaternKernel
    from gpytorch.kernels.keops import PeriodicKernel as KeOpsPeriodicKernel

    HAS_KEOPS = True
except ImportError:
    HAS_KEOPS = False


# =============================================================================
# CG Settings Context Manager
# =============================================================================


@contextmanager
def gpytorch_cg_settings(
    cg_tolerance: float = 1e-2,
    max_cg_iterations: int = 100,
    num_trace_samples: int = 10,
    max_preconditioner_size: int = 15,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    max_root_decomposition_size: int = 20,
):
    """Context manager for GPyTorch CG settings.

    Forces CG mode by setting max_cholesky_size to 0.
    """
    with (
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.cg_tolerance(cg_tolerance),
        gpytorch.settings.eval_cg_tolerance(cg_tolerance),
        gpytorch.settings.max_cg_iterations(max_cg_iterations),
        gpytorch.settings.num_trace_samples(num_trace_samples),
        gpytorch.settings.max_preconditioner_size(max_preconditioner_size),
        gpytorch.settings.min_preconditioning_size(min_preconditioning_size),
        gpytorch.settings.max_lanczos_quadrature_iterations(
            max_lanczos_quadrature_iterations
        ),
        gpytorch.settings.max_root_decomposition_size(max_root_decomposition_size),
    ):
        yield


@contextmanager
def gpytorch_cholesky_settings():
    """Context manager for GPyTorch Cholesky settings (default behavior)."""
    # Use default settings (Cholesky for small matrices)
    with gpytorch.settings.max_cholesky_size(800):
        yield


def _aggregate_cg_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    actual_iters = [int(record.get("actual_iterations", 0)) for record in records]
    return {
        "solve_records": records,
        "solve_count": len(records),
        "cg_iterations_history": actual_iters,
        "cg_iterations_total": int(sum(actual_iters)),
        "cg_iterations_mean": float(np.mean(actual_iters)) if actual_iters else 0.0,
        "cg_iterations_max": int(max(actual_iters)) if actual_iters else 0,
        "cg_iterations_final_step": int(actual_iters[-1]) if actual_iters else 0,
    }


def _cg_telemetry_payload(
    *,
    records: List[Dict[str, Any]],
    configured_for_cg: bool,
    stage: str,
) -> Dict[str, Any]:
    payload = _aggregate_cg_records(records)
    observed = payload["solve_count"] > 0
    payload.update(
        {
            "measured": configured_for_cg,
            "configured_for_cg": configured_for_cg,
            "observed_cg_calls": observed,
            "telemetry_quality": (
                "observed"
                if observed
                else ("unverified" if configured_for_cg else "not_applicable")
            ),
            "stage": stage,
            "timing_basis": (
                "per_optimizer_iteration"
                if stage == "training"
                else "diagnostic_not_aligned_to_warm_repeated_timing"
            ),
        }
    )
    return payload


def _cuda_synchronize_for_tensor(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _time_cuda_phase(fn, *, sync_tensor: torch.Tensor):
    _cuda_synchronize_for_tensor(sync_tensor)
    start = time.perf_counter()
    result = fn()
    _cuda_synchronize_for_tensor(sync_tensor)
    return result, float(time.perf_counter() - start)


def _phase_memory_stat(
    stats: Dict[str, Any], key: str, default: float = 0.0
) -> float:
    value = stats.get(key)
    if value is None:
        return default
    return float(value)


def _prediction_memory_from_phases(*phase_stats: Dict[str, Any], use_love: bool) -> Dict[str, Any]:
    prediction_peak = max(
        _phase_memory_stat(stats, "phase_peak_gpu_mb") for stats in phase_stats
    )
    prediction_delta = max(
        _phase_memory_stat(stats, "phase_delta_gpu_mb") for stats in phase_stats
    )
    torch_peak = max(_phase_memory_stat(stats, "torch_peak_mb") for stats in phase_stats)
    torch_current = 0.0
    for stats in reversed(phase_stats):
        if stats.get("torch_current_mb") is not None:
            torch_current = float(stats["torch_current_mb"])
            break
    memory_stats: Dict[str, Any] = {
        "method": next(
            (str(stats.get("method")) for stats in phase_stats if stats.get("method")),
            "none",
        ),
        "samples": int(sum(int(stats.get("samples", 0) or 0) for stats in phase_stats)),
        "max_mb": prediction_peak,
        "prediction_peak_gpu_mb": prediction_peak,
        "prediction_delta_gpu_mb": prediction_delta,
        "torch_peak_mb": torch_peak,
        "torch_current_mb": torch_current,
    }
    if use_love:
        memory_stats["love_prediction_peak_gpu_mb"] = prediction_peak
        memory_stats["love_prediction_delta_gpu_mb"] = prediction_delta
    else:
        memory_stats["exact_prediction_peak_gpu_mb"] = prediction_peak
        memory_stats["exact_prediction_delta_gpu_mb"] = prediction_delta
    return memory_stats


def _keops_stationary_kernel_diag(model: ExactGP, x: torch.Tensor) -> torch.Tensor:
    """Return k(x, x) diagonal without GPyTorch's brittle KeOps diag path."""

    covar_module = model.covar_module
    if isinstance(covar_module, ScaleKernel):
        outputscale = covar_module.outputscale.to(device=x.device, dtype=x.dtype).reshape(())
        return outputscale.expand(x.shape[-2])

    dense = covar_module(x, x).to_dense()
    return dense.diagonal(dim1=-1, dim2=-2)


@contextmanager
def capture_linear_cg_stats(phase_provider=None):
    """Capture realized linear_cg iteration counts during GPyTorch execution."""
    import linear_operator.utils as linear_utils

    linear_cg_module = importlib.import_module("linear_operator.utils.linear_cg")

    original_module_fn = linear_cg_module.linear_cg
    original_utils_fn = getattr(linear_utils, "linear_cg", None)
    records: List[Dict[str, Any]] = []

    def wrapped_linear_cg(
        matmul_closure,
        rhs,
        n_tridiag=0,
        tolerance=None,
        eps=1e-10,
        stop_updating_after=1e-10,
        max_iter=None,
        max_tridiag_iter=None,
        initial_guess=None,
        preconditioner=None,
    ):
        call_count = 0
        base_closure = (
            matmul_closure.matmul if torch.is_tensor(matmul_closure) else matmul_closure
        )

        def counted_closure(x):
            nonlocal call_count
            call_count += 1
            return base_closure(x)

        result = original_module_fn(
            counted_closure,
            rhs,
            n_tridiag=n_tridiag,
            tolerance=tolerance,
            eps=eps,
            stop_updating_after=stop_updating_after,
            max_iter=max_iter,
            max_tridiag_iter=max_tridiag_iter,
            initial_guess=initial_guess,
            preconditioner=preconditioner,
        )
        phase = phase_provider() if phase_provider is not None else None
        rhs_cols = int(rhs.shape[-1]) if getattr(rhs, "ndim", 0) > 1 else 1
        records.append(
            {
                "phase": phase,
                "rhs_cols": rhs_cols,
                "n_tridiag": int(n_tridiag),
                "max_iter": None if max_iter is None else int(max_iter),
                "max_tridiag_iter": None
                if max_tridiag_iter is None
                else int(max_tridiag_iter),
                "tolerance": None if tolerance is None else float(tolerance),
                "used_preconditioner": preconditioner is not None,
                "matmul_calls": int(call_count),
                "actual_iterations": int(max(call_count - 1, 0)),
            }
        )
        return result

    linear_cg_module.linear_cg = wrapped_linear_cg
    linear_utils.linear_cg = wrapped_linear_cg
    try:
        yield records
    finally:
        linear_cg_module.linear_cg = original_module_fn
        if original_utils_fn is not None:
            linear_utils.linear_cg = original_utils_fn


# =============================================================================
# Single-Output GP Models
# =============================================================================


class GPyTorchSingleOutputGP(ExactGP):
    """Configurable single-output GP for all kernel types."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        kernel_type: str = "rbf",
        ard: bool = False,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        # Get input dimension for ARD
        input_dim = train_x.shape[-1] if ard else None

        # Create kernel based on type
        base_kernel = self._create_kernel(kernel_type, input_dim)
        self.covar_module = ScaleKernel(base_kernel)

    def _create_kernel(self, kernel_type: str, input_dim: Optional[int] = None):
        """Create kernel based on type string."""
        ard_num_dims = input_dim if input_dim else None

        if kernel_type == "rbf":
            return RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == "matern12":
            return MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)
        elif kernel_type == "matern32":
            return MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        elif kernel_type == "matern52":
            return MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        elif kernel_type == "periodic":
            return PeriodicKernel()
        elif kernel_type == "rq":
            return RQKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == "linear":
            return LinearKernel()
        elif kernel_type == "polynomial":
            return PolynomialKernel(power=3)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPyTorchKeOpsSingleOutputGP(ExactGP):
    """KeOps-accelerated single-output GP (RBF, Matern, Periodic only)."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        kernel_type: str = "rbf",
        ard: bool = False,
    ):
        if not HAS_KEOPS:
            raise ImportError("KeOps not available")

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        # Create KeOps kernel
        input_dim = train_x.shape[-1] if ard else None
        base_kernel = self._create_keops_kernel(kernel_type, input_dim)
        self.covar_module = ScaleKernel(base_kernel)

    def _create_keops_kernel(self, kernel_type: str, input_dim: Optional[int] = None):
        """Create KeOps kernel based on type string."""
        ard_num_dims = input_dim if input_dim else None
        if kernel_type == "rbf":
            return KeOpsRBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type in ["matern12", "matern32", "matern52"]:
            nu_map = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}
            return KeOpsMaternKernel(nu=nu_map[kernel_type], ard_num_dims=ard_num_dims)
        elif kernel_type == "periodic":
            return KeOpsPeriodicKernel()
        else:
            raise ValueError(f"KeOps not supported for kernel type: {kernel_type}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# =============================================================================
# Composite Kernel GP
# =============================================================================


class GPyTorchCompositeGP(ExactGP):
    """Composite kernel GP for GPyTorch."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        kernel_spec: str = "rbf+matern52",
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = self._build_composite_kernel(kernel_spec)

    def _build_composite_kernel(self, kernel_spec: str):
        """Build composite kernel from spec string."""
        # Parse the kernel spec
        # Supports: +, *, and parentheses for nesting

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
            elif name == "polynomial":
                return ScaleKernel(PolynomialKernel(power=3))
            else:
                raise ValueError(f"Unknown kernel: {name}")

        # Simple parsing for common nested composite patterns.
        if "(" in kernel_spec:
            if kernel_spec.startswith("(") and ")" in kernel_spec:
                inner_end = kernel_spec.index(")")
                inner = kernel_spec[1:inner_end]
                outer = kernel_spec[inner_end + 1 :]
                inner_kernel = self._build_composite_kernel(inner)
                if outer.startswith("*"):
                    outer_name = outer[1:]
                    return inner_kernel * get_base_kernel(outer_name)
                elif outer.startswith("+"):
                    outer_name = outer[1:]
                    return inner_kernel + get_base_kernel(outer_name)
            # Handle pattern like rbf*(matern52+linear)
            elif "*(" in kernel_spec:
                parts = kernel_spec.split("*(")
                left = parts[0]
                right = parts[1].rstrip(")")
                return get_base_kernel(left) * self._build_composite_kernel(right)

        # Handle simple sum/product
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
        else:
            # Single kernel
            return get_base_kernel(kernel_spec)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# =============================================================================
# Multi-Output GP
# =============================================================================


class GPyTorchMultiOutputGP(ExactGP):
    """Multi-task GP using MultitaskKernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: MultitaskGaussianLikelihood,
        kernel_type: str = "rbf",
        num_tasks: int = 2,
        rank: Optional[int] = None,
        ard: bool = False,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            ConstantMean(), num_tasks=num_tasks
        )

        # Create base kernel
        ard_num_dims = train_x.shape[-1] if ard else None
        if kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == "matern12":
            base_kernel = MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)
        elif kernel_type == "matern32":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        elif kernel_type == "matern52":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
        else:
            raise ValueError(f"Multi-output not supported for kernel: {kernel_type}")

        # Wrap in MultitaskKernel
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            base_kernel, num_tasks=num_tasks, rank=rank or num_tasks
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# =============================================================================
# Training Functions
# =============================================================================


def train_gpytorch_model(
    model: ExactGP,
    likelihood: GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iterations: int = 100,
    lr: float = 0.05,
    mode: str = "cg",
    cg_tolerance: float = 1e-2,
    max_cg_iterations: int = 100,
    num_trace_samples: int = 10,
    max_preconditioner_size: int = 15,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    lr_schedule: str = "constant",
    early_stop_patience: int = 15,
    early_stop_tol: float = 1e-4,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.1,
) -> Dict[str, Any]:
    """Unified GPyTorch training function.

    Args:
        model: GPyTorch model
        likelihood: Gaussian likelihood
        train_x: Training inputs
        train_y: Training targets
        n_iterations: Maximum iterations
        lr: Learning rate
        mode: 'cg', 'cholesky', or 'keops'
        cg_tolerance: CG convergence tolerance
        max_cg_iterations: Max CG iterations
        num_trace_samples: Number of probe vectors for SLQ
        lr_schedule: Learning rate schedule ("constant" or "cosine")
        early_stop_patience: Patience for early stopping
        early_stop_tol: Tolerance for early stopping
        monitor_memory: Whether to monitor GPU memory
        memory_poll_interval: Memory polling interval

    Returns:
        dict with training results
    """
    device = train_x.device

    # Setup memory monitoring
    memory_stats = {}
    if monitor_memory:
        reset_torch_memory_stats()
        monitor = GPUMemoryMonitor(interval=memory_poll_interval)
        monitor.start()

    tracemalloc.start()

    # Setup training
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(
            f"lr_schedule must be 'constant' or 'cosine', got {lr_schedule}"
        )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_iterations, 1)
        )
        if lr_schedule == "cosine"
        else None
    )
    mll = ExactMarginalLogLikelihood(likelihood, model)

    nll_history = []
    best_nll = float("inf")
    patience_counter = 0
    iterations_run = 0
    current_phase = "idle"
    cg_records_per_step = []
    iter_times_s = []

    # Choose settings context
    if mode == "cg" or mode == "keops":
        settings_context = gpytorch_cg_settings(
            cg_tolerance=cg_tolerance,
            max_cg_iterations=max_cg_iterations,
            num_trace_samples=num_trace_samples,
            max_preconditioner_size=max_preconditioner_size,
            max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
            min_preconditioning_size=min_preconditioning_size,
        )
    else:
        settings_context = gpytorch_cholesky_settings()

    start_time = time.perf_counter()
    try:
        with (
            settings_context,
            capture_linear_cg_stats(lambda: current_phase) as cg_records,
        ):
            for i in range(n_iterations):
                iter_start = time.perf_counter()
                step_start = len(cg_records)
                optimizer.zero_grad()
                current_phase = f"train_step_{i + 1}_forward"
                output = model(train_x)
                loss = -mll(output, train_y)
                current_phase = f"train_step_{i + 1}_backward"
                loss.backward()
                current_phase = f"train_step_{i + 1}_optimizer"
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                nll = loss.item()
                iter_times_s.append(time.perf_counter() - iter_start)
                nll_history.append(nll)
                iterations_run = i + 1
                step_records = cg_records[step_start:]
                cg_records_per_step.append(
                    {
                        "step": i + 1,
                        "solve_count": len(step_records),
                        "cg_iterations_total": int(
                            sum(record["actual_iterations"] for record in step_records)
                        ),
                        "solve_records": step_records,
                    }
                )

                # Early stopping check
                if nll < best_nll - early_stop_tol:
                    best_nll = nll
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stop_patience:
                    break
            current_phase = "complete"
    finally:
        training_time = time.perf_counter() - start_time

        if monitor_memory:
            monitor.stop()
            memory_stats = monitor.get_stats()
            torch_stats = get_torch_memory_stats()
            memory_stats.update(torch_stats)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)

    # Extract learned parameters
    learned_params = {}
    try:
        # Try to get lengthscale
        if hasattr(model.covar_module, "data_covar_module"):
            data_kernel = model.covar_module.data_covar_module
            lengthscale = getattr(data_kernel, "lengthscale", None)
            if lengthscale is not None:
                ls = lengthscale.detach().cpu().numpy()
                learned_params["lengthscale"] = float(ls.mean())
                if ls.ndim > 1 and ls.shape[-1] > 1:
                    learned_params["lengthscales"] = ls.flatten().tolist()
        elif hasattr(model.covar_module, "base_kernel"):
            base = model.covar_module.base_kernel
            lengthscale = getattr(base, "lengthscale", None)
            if lengthscale is not None:
                ls = lengthscale.detach().cpu().numpy()
                learned_params["lengthscale"] = float(ls.mean())
                if ls.ndim > 1 and ls.shape[-1] > 1:
                    learned_params["lengthscales"] = ls.flatten().tolist()
        elif hasattr(model.covar_module, "lengthscale"):
            lengthscale = getattr(model.covar_module, "lengthscale", None)
            if lengthscale is not None:
                ls = lengthscale.detach().cpu().numpy()
                learned_params["lengthscale"] = float(ls.mean())

        # Outputscale
        if hasattr(model.covar_module, "outputscale"):
            learned_params["outputscale"] = float(
                model.covar_module.outputscale.detach().cpu().item()
            )
        else:
            learned_params["outputscale"] = 1.0

        # Noise. MultitaskGaussianLikelihood can expose both global noise and
        # task-specific noise; some configurations return None for one of them.
        noise_value = getattr(likelihood, "noise", None)
        task_noises_value = getattr(likelihood, "task_noises", None)
        if noise_value is not None:
            learned_params["noise"] = float(noise_value.detach().cpu().mean().item())
        if task_noises_value is not None:
            task_noises = task_noises_value.detach().cpu().numpy().astype(np.float32)
            learned_params["noise_per_task"] = task_noises.reshape(-1).tolist()
            learned_params.setdefault("noise", float(np.mean(task_noises)))
        if "noise" not in learned_params:
            learned_params["noise"] = 0.1
        if hasattr(model.covar_module, "base_kernel"):
            base_kernel = model.covar_module.base_kernel
            if hasattr(base_kernel, "period_length"):
                learned_params["period"] = float(base_kernel.period_length.detach().cpu().mean().item())
            if hasattr(base_kernel, "alpha"):
                learned_params["alpha"] = float(base_kernel.alpha.detach().cpu().mean().item())
        if hasattr(model, "mean_module") and hasattr(model.mean_module, "constant"):
            learned_params["mean"] = float(model.mean_module.constant.detach().cpu().item())
    except Exception as e:
        print(f"Warning: Could not extract hyperparameters: {e}")
        learned_params = {"lengthscale": 1.0, "outputscale": 1.0, "noise": 0.1, "mean": 0.0}

    early_stopped = iterations_run < n_iterations
    all_cg_records = [
        record for step in cg_records_per_step for record in step["solve_records"]
    ]

    return {
        "model": model,
        "likelihood": likelihood,
        "final_nll": nll_history[-1] if nll_history else 0.0,
        "nll_history": nll_history,
        "training_time_s": training_time,
        "iterations_run": iterations_run,
        "max_iterations": n_iterations,
        "early_stopped": early_stopped,
        "learned_params": learned_params,
        "peak_memory_mb": memory_stats.get("max_mb", 0.0),
        "memory_stats": memory_stats,
        "optimizer_config": {
            "max_iterations": n_iterations,
            "learning_rate": lr,
            "lr_schedule": lr_schedule,
            "early_stop_patience": early_stop_patience,
            "early_stop_tol": early_stop_tol,
        },
        "solver_config": {
            "framework": "gpytorch",
            "model_family": type(model).__name__,
            "mode": mode,
            "max_cholesky_size": 0 if mode in ("cg", "keops") else 800,
            "cg_tolerance": cg_tolerance,
            "max_cg_iterations": max_cg_iterations,
            "num_trace_samples": num_trace_samples,
            "max_preconditioner_size": max_preconditioner_size,
            "max_lanczos_quadrature_iterations": max_lanczos_quadrature_iterations,
            "min_preconditioning_size": min_preconditioning_size,
        },
        "cg_telemetry": {
            **_cg_telemetry_payload(
                records=all_cg_records,
                configured_for_cg=mode in ("cg", "keops"),
                stage="training",
            ),
            "per_step": cg_records_per_step,
        },
        **_direct_iteration_timing_payload(iter_times_s),
    }


def predict_gpytorch_model(
    model: ExactGP,
    likelihood: GaussianLikelihood,
    test_x: torch.Tensor,
    mode: str = "cg",
    cg_tolerance: float = 1e-2,
    max_cg_iterations: int = 100,
    max_preconditioner_size: int = 15,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    max_root_decomposition_size: int = 20,
    use_love: bool = False,
    predictive_target: str = "observed",
    train_x: Optional[torch.Tensor] = None,
    train_y: Optional[torch.Tensor] = None,
    exact_prediction_block_size: int = 512,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.02,
) -> Dict[str, Any]:
    """Unified GPyTorch prediction function.

    Args:
        model: Trained GPyTorch model
        likelihood: Gaussian likelihood
        test_x: Test inputs
        mode: 'cg', 'cholesky', or 'keops'
        cg_tolerance: CG convergence tolerance
        max_cg_iterations: Maximum CG iterations
        use_love: If True, use LOVE (fast_pred_var) for variance approximation.
            Default False uses exact variance (full CG solve).
        predictive_target: "observed" includes likelihood noise in the returned
            variance. "latent" returns the latent function posterior variance.

    Returns:
        dict with prediction results
    """
    model.eval()
    likelihood.eval()
    if predictive_target not in ("observed", "latent"):
        raise ValueError(
            "predictive_target must be 'observed' or 'latent', "
            f"got {predictive_target!r}"
        )
    if mode == "keops" and not use_love:
        if train_x is None or train_y is None:
            raise ValueError("KeOps exact prediction requires train_x and train_y")
        return predict_gpytorch_keops_exact_model(
            model,
            likelihood,
            train_x,
            train_y,
            test_x,
            cg_tolerance=cg_tolerance,
            max_cg_iterations=max_cg_iterations,
            max_preconditioner_size=max_preconditioner_size,
            max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
            min_preconditioning_size=min_preconditioning_size,
            max_root_decomposition_size=max_root_decomposition_size,
            predictive_target=predictive_target,
            exact_prediction_block_size=exact_prediction_block_size,
            monitor_memory=monitor_memory,
            memory_poll_interval=memory_poll_interval,
        )

    # Choose settings context
    if mode == "cg" or mode == "keops":
        settings_context = gpytorch_cg_settings(
            cg_tolerance=cg_tolerance,
            max_cg_iterations=max_cg_iterations,
            max_preconditioner_size=max_preconditioner_size,
            max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
            min_preconditioning_size=min_preconditioning_size,
            max_root_decomposition_size=max_root_decomposition_size,
        )
    else:
        settings_context = gpytorch_cholesky_settings()

    eager_kernel_context = (
        gpytorch.settings.lazily_evaluate_kernels(False)
        if mode == "keops" and not use_love
        else nullcontext()
    )

    start_time = time.perf_counter()
    current_phase = "predict_setup"
    with (
        torch.no_grad(),
        settings_context,
        eager_kernel_context,
        gpytorch.settings.fast_pred_var(use_love),
        capture_linear_cg_stats(lambda: current_phase) as cg_records,
    ):
        def _run_mean_phase():
            nonlocal current_phase
            current_phase = "predict_mean"
            pred_local = model(test_x)
            mean_local = pred_local.mean.cpu().numpy()
            return pred_local, mean_local

        start_mean = time.perf_counter()
        if monitor_memory:
            (pred, mean), mean_memory_stats = measure_gpu_phase(
                _run_mean_phase,
                interval=memory_poll_interval,
            )
        else:
            pred, mean = _run_mean_phase()
            mean_memory_stats = {}
        mean_time = time.perf_counter() - start_mean

        def _run_variance_phase():
            nonlocal current_phase
            current_phase = (
                "predict_variance_love" if use_love else "predict_variance_exact"
            )
            pred_for_variance = likelihood(pred) if predictive_target == "observed" else pred
            return pred_for_variance.variance.cpu().numpy()

        start_var = time.perf_counter()
        if monitor_memory:
            variance, variance_memory_stats = measure_gpu_phase(
                _run_variance_phase,
                interval=memory_poll_interval,
            )
        else:
            variance = _run_variance_phase()
            variance_memory_stats = {}
        current_phase = "predict_complete"
        variance_time = time.perf_counter() - start_var

    total_time = time.perf_counter() - start_time

    std = np.sqrt(np.maximum(variance, 1e-10))
    prediction_peak = max(
        float(mean_memory_stats.get("phase_peak_gpu_mb", 0.0)),
        float(variance_memory_stats.get("phase_peak_gpu_mb", 0.0)),
    )
    prediction_delta = max(
        float(mean_memory_stats.get("phase_delta_gpu_mb", 0.0)),
        float(variance_memory_stats.get("phase_delta_gpu_mb", 0.0)),
    )
    prediction_memory_stats: Dict[str, Any] = {
        "method": str(
            variance_memory_stats.get("method", mean_memory_stats.get("method", "none"))
        ),
        "samples": int(mean_memory_stats.get("samples", 0))
        + int(variance_memory_stats.get("samples", 0)),
        "max_mb": prediction_peak,
        "prediction_peak_gpu_mb": prediction_peak,
        "prediction_delta_gpu_mb": prediction_delta,
        "torch_peak_mb": max(
            float(mean_memory_stats.get("torch_peak_mb", 0.0)),
            float(variance_memory_stats.get("torch_peak_mb", 0.0)),
        ),
        "torch_current_mb": float(
            variance_memory_stats.get(
                "torch_current_mb",
                mean_memory_stats.get("torch_current_mb", 0.0),
            )
        ),
    }
    if use_love:
        prediction_memory_stats["love_prediction_peak_gpu_mb"] = prediction_peak
        prediction_memory_stats["love_prediction_delta_gpu_mb"] = prediction_delta
    else:
        prediction_memory_stats["exact_prediction_peak_gpu_mb"] = prediction_peak
        prediction_memory_stats["exact_prediction_delta_gpu_mb"] = prediction_delta

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "mean_time_s": mean_time,
        "variance_time_s": variance_time,
        "total_time_s": total_time,
        "solver_config": {
            "framework": "gpytorch",
            "mode": mode,
            "prediction_mode": "love" if use_love else "exact",
            "predictive_target": predictive_target,
            "max_cholesky_size": 0 if mode in ("cg", "keops") else 800,
            "max_cg_iterations": max_cg_iterations,
            "cg_tolerance": cg_tolerance,
            "max_preconditioner_size": max_preconditioner_size,
            "max_lanczos_quadrature_iterations": max_lanczos_quadrature_iterations,
            "min_preconditioning_size": min_preconditioning_size,
            "max_root_decomposition_size": max_root_decomposition_size,
        },
        "cg_telemetry": _cg_telemetry_payload(
            records=cg_records,
            configured_for_cg=mode in ("cg", "keops"),
            stage="prediction",
        ),
        "memory_stats": prediction_memory_stats,
    }


# =============================================================================
# High-Level Training Wrappers
# =============================================================================


def predict_gpytorch_keops_exact_model(
    model: ExactGP,
    likelihood: GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    *,
    cg_tolerance: float = 1e-3,
    max_cg_iterations: int = 300,
    max_preconditioner_size: int = 0,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    max_root_decomposition_size: int = 20,
    predictive_target: str = "observed",
    exact_prediction_block_size: int = 512,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.02,
) -> Dict[str, Any]:
    """Exact matrix-free prediction for a trained KeOps ExactGP.

    GPyTorch's default exact predictive covariance path densifies the full
    test-train covariance and has been brittle for KeOps LazyTensor objects in
    benchmark runs. This comparator keeps the train-train operator lazy, uses
    KeOps-backed CG solves, and materializes only bounded `[m_block x n_train]`
    cross-covariance blocks, matching MojoGP's matrix-free exact prediction
    memory contract.
    """

    if predictive_target not in ("observed", "latent"):
        raise ValueError(
            "predictive_target must be 'observed' or 'latent', "
            f"got {predictive_target!r}"
        )
    if exact_prediction_block_size <= 0:
        raise ValueError("exact_prediction_block_size must be positive")

    model.eval()
    likelihood.eval()
    n_test = int(test_x.shape[0])
    block_size = min(int(exact_prediction_block_size), max(n_test, 1))
    current_phase = "predict_setup"
    train_covar = None
    alpha = None

    settings_context = gpytorch_cg_settings(
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        max_preconditioner_size=max_preconditioner_size,
        max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
        min_preconditioning_size=min_preconditioning_size,
        max_root_decomposition_size=max_root_decomposition_size,
    )

    start_time = time.perf_counter()
    with (
        torch.no_grad(),
        settings_context,
        gpytorch.settings.fast_pred_var(False),
        capture_linear_cg_stats(lambda: current_phase) as cg_records,
    ):
        def _prepare_alpha():
            nonlocal train_covar, alpha, current_phase
            current_phase = "predict_alpha"
            train_mean = model.mean_module(train_x)
            train_prior = MultivariateNormal(train_mean, model.covar_module(train_x))
            observed_train_prior = likelihood(train_prior)
            train_covar = observed_train_prior.lazy_covariance_matrix
            rhs = (train_y - train_mean).unsqueeze(-1)
            alpha = train_covar.solve(rhs)
            return alpha

        alpha_start = time.perf_counter()
        if monitor_memory:
            _, alpha_memory_stats = measure_gpu_phase(
                _prepare_alpha,
                interval=memory_poll_interval,
            )
        else:
            _prepare_alpha()
            alpha_memory_stats = {}
        _cuda_synchronize_for_tensor(test_x)
        alpha_time = time.perf_counter() - alpha_start

        assert train_covar is not None
        assert alpha is not None

        mean = torch.empty(n_test, dtype=torch.float32, device=test_x.device)

        def _compute_mean():
            nonlocal current_phase
            current_phase = "predict_mean"
            for start in range(0, n_test, block_size):
                end = min(start + block_size, n_test)
                test_block = test_x[start:end]
                cross = model.covar_module(test_block, train_x)
                mean[start:end] = (
                    model.mean_module(test_block) + cross.matmul(alpha).squeeze(-1)
                )
            return mean

        mean_start = time.perf_counter()
        if monitor_memory:
            _, mean_memory_stats = measure_gpu_phase(
                _compute_mean,
                interval=memory_poll_interval,
            )
        else:
            _compute_mean()
            mean_memory_stats = {}
        _cuda_synchronize_for_tensor(test_x)
        mean_time = time.perf_counter() - mean_start

        variance = torch.empty(n_test, dtype=torch.float32, device=test_x.device)
        exact_cross_time = 0.0
        exact_diag_time = 0.0
        exact_solve_time = 0.0
        exact_post_time = 0.0
        exact_alloc_time = 0.0

        def _compute_variance():
            nonlocal current_phase
            nonlocal exact_cross_time, exact_diag_time, exact_solve_time, exact_post_time, exact_alloc_time
            current_phase = "predict_variance_exact"
            for start in range(0, n_test, block_size):
                end = min(start + block_size, n_test)
                test_block = test_x[start:end]

                alloc_start = time.perf_counter()
                _cuda_synchronize_for_tensor(test_x)
                exact_alloc_time += time.perf_counter() - alloc_start

                def _cross_dense():
                    cross = model.covar_module(test_block, train_x)
                    if hasattr(cross, "to_dense"):
                        return cross.to_dense()
                    return torch.as_tensor(cross, device=test_x.device)

                cross_dense, cross_time = _time_cuda_phase(
                    _cross_dense,
                    sync_tensor=test_x,
                )
                exact_cross_time += cross_time

                def _diag():
                    return _keops_stationary_kernel_diag(model, test_block)

                diag, diag_time = _time_cuda_phase(_diag, sync_tensor=test_x)
                exact_diag_time += diag_time

                def _solve():
                    return train_covar.solve(cross_dense.transpose(-1, -2).contiguous())

                solve, solve_time = _time_cuda_phase(_solve, sync_tensor=test_x)
                exact_solve_time += solve_time

                def _post():
                    correction = (cross_dense * solve.transpose(-1, -2)).sum(dim=-1)
                    block_var = diag - correction
                    if predictive_target == "observed":
                        block_var = block_var + likelihood.noise.reshape(()).to(block_var.device)
                    return torch.clamp(block_var, min=1e-10)

                block_var, post_time = _time_cuda_phase(_post, sync_tensor=test_x)
                exact_post_time += post_time
                variance[start:end] = block_var
            return variance

        variance_start = time.perf_counter()
        if monitor_memory:
            _, variance_memory_stats = measure_gpu_phase(
                _compute_variance,
                interval=memory_poll_interval,
            )
        else:
            _compute_variance()
            variance_memory_stats = {}
        _cuda_synchronize_for_tensor(test_x)
        variance_time = time.perf_counter() - variance_start
        current_phase = "predict_complete"

    total_time = time.perf_counter() - start_time
    mean_np = mean.detach().cpu().numpy()
    variance_np = variance.detach().cpu().numpy()
    std = np.sqrt(np.maximum(variance_np, 1e-10))

    variance_records = [
        record
        for record in cg_records
        if str(record.get("phase", "")).startswith("predict_variance_exact")
    ]
    exact_cg_iterations = [
        int(record.get("actual_iterations", 0)) for record in variance_records
    ]
    prediction_memory_stats = _prediction_memory_from_phases(
        alpha_memory_stats,
        mean_memory_stats,
        variance_memory_stats,
        use_love=False,
    )
    return {
        "mean": mean_np,
        "variance": variance_np,
        "std": std,
        "mean_time_s": mean_time,
        "variance_time_s": variance_time,
        "total_time_s": total_time,
        "alpha_time_s": alpha_time,
        "solver_config": {
            "framework": "gpytorch",
            "mode": "keops",
            "prediction_mode": "exact",
            "predictive_target": predictive_target,
            "max_cholesky_size": 0,
            "max_cg_iterations": max_cg_iterations,
            "cg_tolerance": cg_tolerance,
            "max_preconditioner_size": max_preconditioner_size,
            "max_lanczos_quadrature_iterations": max_lanczos_quadrature_iterations,
            "min_preconditioning_size": min_preconditioning_size,
            "max_root_decomposition_size": max_root_decomposition_size,
            "exact_prediction_block_size": block_size,
        },
        "cg_telemetry": _cg_telemetry_payload(
            records=cg_records,
            configured_for_cg=True,
            stage="prediction",
        ),
        "memory_stats": prediction_memory_stats,
        "exact_block_cols": block_size,
        "exact_cross_mode": "chunked_keops_cross_covariance",
        "exact_cg_block_count": len(variance_records),
        "exact_cg_total_iterations": int(sum(exact_cg_iterations)),
        "exact_cg_max_iterations": int(max(exact_cg_iterations)) if exact_cg_iterations else 0,
        "exact_alloc_time_s": exact_alloc_time,
        "exact_cross_time_s": exact_cross_time,
        "exact_diag_time_s": exact_diag_time,
        "exact_solve_time_s": exact_solve_time,
        "exact_post_time_s": exact_post_time,
    }


def train_gpytorch_single_output(
    X: np.ndarray,
    y: np.ndarray,
    kernel_type: str,
    mode: str = "cg",
    n_iterations: int = 100,
    lr: float = 0.05,
    init_ls: float = 1.0,
    init_noise: float = 0.1,
    init_os: float = 1.0,
    ard: bool = False,
    lr_schedule: str = "constant",
    early_stop_patience: int = 15,
    early_stop_tol: float = 1e-4,
    cg_tolerance: float = 1e-2,
    max_cg_iterations: int = 100,
    num_trace_samples: int = 10,
    max_preconditioner_size: int = 15,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.1,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train GPyTorch single-output GP.

    Args:
        X: Training data [n, d]
        y: Training targets [n]
        kernel_type: Kernel type string
        mode: 'cg', 'cholesky', or 'keops'
        n_iterations: Maximum iterations
        lr: Learning rate
        init_ls: Initial lengthscale
        init_noise: Initial noise
        init_os: Initial outputscale
        ard: Whether to use ARD
        monitor_memory: Whether to monitor memory
        memory_poll_interval: Memory polling interval
        device: Device to use

    Returns:
        dict with training results
    """
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Convert to tensors
    train_x = torch.tensor(X, dtype=torch.float32, device=device)
    train_y = torch.tensor(y, dtype=torch.float32, device=device)

    # Create likelihood and model
    likelihood = GaussianLikelihood()
    likelihood.noise = init_noise

    requested_mode = mode
    if (
        mode == "keops"
        and HAS_KEOPS
        and kernel_type in ["rbf", "matern12", "matern32", "matern52", "periodic"]
    ):
        model = GPyTorchKeOpsSingleOutputGP(
            train_x,
            train_y,
            likelihood,
            kernel_type,
            ard=ard,
        )
    else:
        model = GPyTorchSingleOutputGP(
            train_x, train_y, likelihood, kernel_type, ard=ard
        )

    effective_mode = (
        "keops"
        if isinstance(model, GPyTorchKeOpsSingleOutputGP)
        else ("cg" if requested_mode in ("cg", "keops") else requested_mode)
    )
    backend_fallback_reason = None
    if requested_mode == "keops" and effective_mode != "keops":
        if not HAS_KEOPS:
            backend_fallback_reason = "keops_unavailable"
        elif kernel_type not in keops_supported_kernels():
            backend_fallback_reason = f"kernel_not_supported:{kernel_type}"
        else:
            backend_fallback_reason = "keops_training_fallback"

    # Set initial hyperparameters
    if hasattr(model.covar_module, "base_kernel"):
        base_kernel = model.covar_module.base_kernel
        if getattr(base_kernel, "has_lengthscale", False):
            base_kernel.lengthscale = init_ls
        if hasattr(base_kernel, "period_length"):
            base_kernel.period_length = 1.0
    if hasattr(model.covar_module, "outputscale"):
        model.covar_module.outputscale = init_os

    # Move to device
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Train
    result = train_gpytorch_model(
        model,
        likelihood,
        train_x,
        train_y,
        n_iterations=n_iterations,
        lr=lr,
        mode=mode,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        num_trace_samples=num_trace_samples,
        max_preconditioner_size=max_preconditioner_size,
        max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
        min_preconditioning_size=min_preconditioning_size,
        lr_schedule=lr_schedule,
        early_stop_patience=early_stop_patience,
        early_stop_tol=early_stop_tol,
        monitor_memory=monitor_memory,
        memory_poll_interval=memory_poll_interval,
    )

    result["train_x"] = train_x
    result["train_y"] = train_y
    result["device"] = device
    result["kernel_type"] = kernel_type
    result["ard"] = ard
    result.update(
        _requested_effective_backend_payload(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            backend_fallback_reason=backend_fallback_reason,
        )
    )

    return result


def predict_gpytorch_single_output(
    training_result: Dict[str, Any],
    X_test: np.ndarray,
    mode: str = "cg",
    cg_tolerance: float = 1e-2,
    max_cg_iterations: int = 100,
    max_preconditioner_size: int = 15,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    max_root_decomposition_size: int = 20,
    use_love: bool = False,
    exact_prediction_block_size: int = 512,
) -> Dict[str, Any]:
    """Predict using trained GPyTorch single-output GP.

    Args:
        training_result: Result from train_gpytorch_single_output
        X_test: Test data [m, d]
        mode: 'cg', 'cholesky', or 'keops'

    Returns:
        dict with prediction results
    """
    model = training_result["model"]
    likelihood = training_result["likelihood"]
    device = training_result.get("device", "cuda")
    requested_mode = mode
    effective_prediction_mode = mode
    backend_fallback_reason = None

    test_x = torch.tensor(X_test, dtype=torch.float32, device=device)

    try:
        result = predict_gpytorch_model(
            model,
            likelihood,
            test_x,
            mode=mode,
            cg_tolerance=cg_tolerance,
            max_cg_iterations=max_cg_iterations,
            max_preconditioner_size=max_preconditioner_size,
            max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
            min_preconditioning_size=min_preconditioning_size,
            max_root_decomposition_size=max_root_decomposition_size,
            use_love=use_love,
            train_x=training_result.get("train_x"),
            train_y=training_result.get("train_y"),
            exact_prediction_block_size=exact_prediction_block_size,
        )
    except Exception as exc:
        if requested_mode != "keops":
            raise
        effective_prediction_mode = "unsupported"
        backend_fallback_reason = f"keops_prediction_failed:{type(exc).__name__}"
        nan_vec = np.full((X_test.shape[0],), np.nan, dtype=np.float32)
        result = {
            "mean": nan_vec.copy(),
            "variance": nan_vec.copy(),
            "std": nan_vec.copy(),
            "mean_time_s": float("nan"),
            "variance_time_s": float("nan"),
            "total_time_s": float("nan"),
            "solver_config": {
                "framework": "gpytorch",
                "mode": mode,
                "prediction_mode": "love" if use_love else "exact",
                "max_cholesky_size": 0,
                "max_cg_iterations": max_cg_iterations,
                "cg_tolerance": cg_tolerance,
                "max_preconditioner_size": max_preconditioner_size,
                "max_lanczos_quadrature_iterations": max_lanczos_quadrature_iterations,
                "min_preconditioning_size": min_preconditioning_size,
                "max_root_decomposition_size": max_root_decomposition_size,
            },
            "cg_telemetry": {
                "measured": True,
                "configured_for_cg": True,
                "observed_cg_calls": False,
                "telemetry_quality": "unsupported",
                "stage": "prediction",
                "timing_basis": "diagnostic_not_aligned_to_warm_repeated_timing",
                "solve_records": [],
                "solve_count": 0,
                "cg_iterations_history": [],
                "cg_iterations_total": 0,
                "cg_iterations_mean": 0.0,
                "cg_iterations_max": 0,
                "cg_iterations_final_step": 0,
            },
            "prediction_supported": False,
            "prediction_error": str(exc),
        }
    else:
        result["prediction_supported"] = True
        result["prediction_error"] = None
    result.update(
        _requested_effective_backend_payload(
            requested_mode=requested_mode,
            effective_mode=str(training_result.get("effective_mode", requested_mode)),
            prediction_mode=effective_prediction_mode,
            backend_fallback_reason=backend_fallback_reason,
        )
    )
    return result


def train_gpytorch_multi_output(
    X: np.ndarray,
    Y: np.ndarray,
    kernel_type: str,
    num_tasks: int,
    mode: str = "cg",
    n_iterations: int = 100,
    lr: float = 0.05,
    init_ls: float = 1.0,
    init_noise: float = 0.1,
    lr_schedule: str = "constant",
    cg_tolerance: float = 1e-2,
    max_cg_iterations: int = 100,
    num_trace_samples: int = 10,
    max_preconditioner_size: int = 15,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.1,
    device: str = "cuda",
    ard: bool = False,
    init_task_noises: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_x = torch.tensor(X, dtype=torch.float32, device=device)
    train_y = torch.tensor(Y, dtype=torch.float32, device=device)
    likelihood = MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        has_global_noise=False,
        has_task_noise=True,
    )
    if init_task_noises is None:
        task_noise_init = np.full(num_tasks, init_noise, dtype=np.float32)
    else:
        task_noise_init = np.asarray(init_task_noises, dtype=np.float32)
        if task_noise_init.shape != (num_tasks,):
            raise ValueError(
                f"init_task_noises must have shape ({num_tasks},), got {task_noise_init.shape}"
            )
    if np.any(task_noise_init <= 0):
        raise ValueError("init_task_noises entries must be strictly positive")
    likelihood.task_noises = torch.tensor(task_noise_init, dtype=torch.float32)
    model = GPyTorchMultiOutputGP(
        train_x,
        train_y,
        likelihood,
        kernel_type=kernel_type,
        num_tasks=num_tasks,
        ard=ard,
    )
    data_kernel = model.covar_module.data_covar_module
    if getattr(data_kernel, "has_lengthscale", False):
        data_kernel.lengthscale = init_ls

    model = model.to(device)
    likelihood = likelihood.to(device)
    result = train_gpytorch_model(
        model,
        likelihood,
        train_x,
        train_y,
        n_iterations=n_iterations,
        lr=lr,
        mode=mode,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        num_trace_samples=num_trace_samples,
        max_preconditioner_size=max_preconditioner_size,
        max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
        min_preconditioning_size=min_preconditioning_size,
        lr_schedule=lr_schedule,
        monitor_memory=monitor_memory,
        memory_poll_interval=memory_poll_interval,
    )
    result["train_x"] = train_x
    result["train_y"] = train_y
    result["device"] = device
    result["num_tasks"] = num_tasks
    result["kernel_type"] = kernel_type
    result["ard"] = ard
    return result


def predict_gpytorch_multi_output(
    training_result: Dict[str, Any],
    X_test: np.ndarray,
    mode: str = "cg",
    cg_tolerance: float = 1e-2,
    max_cg_iterations: int = 100,
    max_preconditioner_size: int = 15,
    max_lanczos_quadrature_iterations: int = 20,
    min_preconditioning_size: int = 0,
    max_root_decomposition_size: int = 20,
    use_love: bool = False,
) -> Dict[str, Any]:
    model = training_result["model"]
    likelihood = training_result["likelihood"]
    device = training_result.get("device", "cuda")
    test_x = torch.tensor(X_test, dtype=torch.float32, device=device)
    return predict_gpytorch_model(
        model,
        likelihood,
        test_x,
        mode=mode,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        max_preconditioner_size=max_preconditioner_size,
        max_lanczos_quadrature_iterations=max_lanczos_quadrature_iterations,
        min_preconditioning_size=min_preconditioning_size,
        max_root_decomposition_size=max_root_decomposition_size,
        use_love=use_love,
    )


# =============================================================================
# Full Benchmark Runner
# =============================================================================


def run_gpytorch_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    f_test: np.ndarray,
    kernel_type: str,
    mode: str = "cg",
    n_iterations: int = 100,
    lr: float = 0.05,
    init_ls: float = 1.0,
    init_noise: float = 0.1,
    init_os: float = 1.0,
    true_params: Optional[Dict[str, float]] = None,
    monitor_memory: bool = True,
    device: str = "cuda",
    variance_method: str = "exact",
) -> BenchmarkResult:
    """Run a complete GPyTorch benchmark and return structured results.

    Args:
        X_train, y_train: Training data
        X_test: Test inputs
        f_test: Ground truth test outputs (noiseless)
        kernel_type: Kernel type string
        mode: 'cg', 'cholesky', or 'keops'
        n_iterations: Max iterations
        lr: Learning rate
        init_ls, init_noise, init_os: Initial hyperparameters
        true_params: True hyperparameters (for recovery error calculation)
        monitor_memory: Whether to monitor GPU memory
        device: Device to use

    Returns:
        BenchmarkResult with all metrics
    """
    # Train
    train_result = train_gpytorch_single_output(
        X_train,
        y_train,
        kernel_type,
        mode,
        n_iterations,
        lr,
        init_ls,
        init_noise,
        init_os,
        monitor_memory=monitor_memory,
        device=device,
    )

    # Predict
    if variance_method not in ("exact", "love"):
        raise ValueError(
            "GPyTorch accuracy comparator supports variance_method='exact' or 'love', "
            f"got {variance_method!r}"
        )
    pred_result = predict_gpytorch_single_output(
        train_result,
        X_test,
        mode=mode,
        use_love=(variance_method == "love"),
    )

    # Compute accuracy metrics
    accuracy_metrics = compute_all_accuracy_metrics(
        f_test,
        pred_result["mean"],
        pred_result["std"],
        y_train_mean=float(np.mean(y_train)),
        y_train_std=float(np.std(y_train)),
    )

    # Build result objects
    accuracy = AccuracyResult(
        rmse=accuracy_metrics["rmse"],
        mae=accuracy_metrics["mae"],
        r_squared=accuracy_metrics["r_squared"],
        crps=accuracy_metrics["crps"],
        msll=accuracy_metrics["msll"],
        calibration_coverage={
            0.5: accuracy_metrics["calibration_50"],
            0.9: accuracy_metrics["calibration_90"],
            0.95: accuracy_metrics["calibration_95"],
            0.99: accuracy_metrics["calibration_99"],
        },
        calibration_error=accuracy_metrics["calibration_error"],
        sharpness=accuracy_metrics["sharpness"],
        interval_width_95=accuracy_metrics["interval_width_95"],
    )

    speed = SpeedResult(
        training_time_s=train_result["training_time_s"],
        prediction_mean_time_s=pred_result["mean_time_s"],
        prediction_variance_time_s=pred_result["variance_time_s"],
        end_to_end_time_s=train_result["training_time_s"] + pred_result["total_time_s"],
        iterations_run=train_result["iterations_run"],
        max_iterations=train_result["max_iterations"],
        early_stopped=train_result["early_stopped"],
        ms_per_iteration=float(
            train_result.get(
                "iter_time_median_ms",
                (train_result["training_time_s"] / max(train_result["iterations_run"], 1))
                * 1000,
            )
        ),
        iter_time_min_ms=train_result.get("iter_time_min_ms"),
        iter_time_q25_ms=train_result.get("iter_time_q25_ms"),
        iter_time_mean_ms=train_result.get("iter_time_mean_ms"),
        iter_time_median_ms=train_result.get("iter_time_median_ms"),
        iter_time_q75_ms=train_result.get("iter_time_q75_ms"),
        iter_time_max_ms=train_result.get("iter_time_max_ms"),
        iter_time_p5_ms=train_result.get("iter_time_p5_ms"),
        iter_time_p95_ms=train_result.get("iter_time_p95_ms"),
        iter_times_ms=train_result.get("iter_times_ms"),
    )

    memory_stats = merge_gpytorch_benchmark_memory(
        dict(train_result.get("memory_stats", {})),
        dict(pred_result.get("memory_stats", {})),
    )
    memory = MemoryResult(
        gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
        gpu_min_mb=memory_stats.get("min_mb", 0.0),
        gpu_max_mb=memory_stats.get("max_mb", 0.0),
        gpu_var_mb=memory_stats.get("var_mb", 0.0),
        torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
        torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
        cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
        measurement_method=memory_stats.get("method", "none"),
        num_samples=memory_stats.get("samples", 0),
        training_peak_gpu_mb=memory_stats.get("training_peak_gpu_mb"),
        prediction_peak_gpu_mb=memory_stats.get("prediction_peak_gpu_mb"),
        prediction_delta_gpu_mb=memory_stats.get("prediction_delta_gpu_mb"),
        exact_prediction_peak_gpu_mb=memory_stats.get("exact_prediction_peak_gpu_mb"),
        exact_prediction_delta_gpu_mb=memory_stats.get("exact_prediction_delta_gpu_mb"),
        love_prediction_peak_gpu_mb=memory_stats.get("love_prediction_peak_gpu_mb"),
        love_prediction_delta_gpu_mb=memory_stats.get("love_prediction_delta_gpu_mb"),
    )

    learned = train_result["learned_params"]
    hyperparameters = HyperparameterResult(
        learned_lengthscale=learned.get("lengthscale", 1.0),
        learned_noise=learned.get("noise", 0.1),
        learned_outputscale=learned.get("outputscale", 1.0),
        final_nll=train_result["final_nll"],
    )

    # Compute recovery errors if true params provided
    if true_params:
        if "lengthscale" in true_params:
            hyperparameters.lengthscale_rel_error = param_relative_error(
                learned.get("lengthscale", 1.0), true_params["lengthscale"]
            )
        if "noise" in true_params:
            hyperparameters.noise_rel_error = param_relative_error(
                learned.get("noise", 0.1), true_params["noise"]
            )
        if "outputscale" in true_params:
            hyperparameters.outputscale_rel_error = param_relative_error(
                learned.get("outputscale", 1.0), true_params["outputscale"]
            )

    config = {
        "kernel": kernel_type,
        "n": len(X_train),
        "d": X_train.shape[1],
        "mode": mode,
        "variance_method": variance_method,
        "n_iterations": n_iterations,
        "lr": lr,
        "framework": "gpytorch",
        "requested_mode": train_result.get("requested_mode", mode),
        "effective_mode": train_result.get("effective_mode", mode),
        "effective_prediction_mode": pred_result.get(
            "effective_prediction_mode",
            pred_result.get("solver_config", {}).get("mode", mode),
        ),
        "backend_fallback_used": bool(
            train_result.get("backend_fallback_used")
            or pred_result.get("backend_fallback_used")
        ),
        "backend_fallback_reason": pred_result.get(
            "backend_fallback_reason",
            train_result.get("backend_fallback_reason"),
        ),
    }

    return BenchmarkResult(
        config=config,
        accuracy=accuracy,
        speed=speed,
        memory=memory,
        hyperparameters=hyperparameters,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def keops_supported_kernels() -> List[str]:
    """Return list of kernel types supported by KeOps."""
    return ["rbf", "matern12", "matern32", "matern52", "periodic"]


def is_keops_available() -> bool:
    """Check if KeOps is available."""
    return HAS_KEOPS
