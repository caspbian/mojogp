"""Result dataclasses for system benchmarks.

These dataclasses capture all metrics from benchmark runs and provide
serialization to JSON for persistent storage.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import json
import subprocess
import numpy as np


# =============================================================================
# Module-level constants for result file naming
# =============================================================================

# Session ID: Generated once per pytest session (or module import)
_SESSION_ID = uuid.uuid4().hex[:8]


def _get_git_hash() -> str:
    """Get the current git commit hash (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# Git hash: Captured once at module import
_GIT_HASH = _get_git_hash()


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


@dataclass
class AccuracyResult:
    """Predictive accuracy metrics for a single run.

    These are the PRIMARY metrics -- what we care about most.
    """

    rmse: float
    mae: float
    r_squared: float
    crps: float  # CRPS (Gaussian) -- key probabilistic metric
    msll: float  # Mean Standardized Log Loss
    calibration_coverage: Dict[float, float]  # {0.5: x, 0.9: x, 0.95: x, 0.99: x}
    calibration_error: float  # Mean absolute calibration error
    sharpness: float  # Mean predictive std
    interval_width_95: float  # Mean 95% interval width

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeedResult:
    """Speed metrics for a single run.

    Records both raw time AND iteration counts so the user can understand
    the trade-off between runtime and accuracy.
    """

    training_time_s: float
    prediction_mean_time_s: float
    prediction_variance_time_s: float
    end_to_end_time_s: float  # training + prediction (mean + var)
    iterations_run: int  # ACTUAL iterations completed
    max_iterations: int  # Maximum iterations that were requested
    early_stopped: bool  # Whether early stopping triggered
    ms_per_iteration: float  # Primary per-iteration metric in ms (median when available)
    iter_time_min_ms: Optional[float] = None
    iter_time_q25_ms: Optional[float] = None
    iter_time_mean_ms: Optional[float] = None
    iter_time_median_ms: Optional[float] = None
    iter_time_q75_ms: Optional[float] = None
    iter_time_max_ms: Optional[float] = None
    iter_time_p5_ms: Optional[float] = None
    iter_time_p95_ms: Optional[float] = None
    iter_times_ms: Optional[List[float]] = None
    iter_timing_quality: Optional[str] = None
    startup_compile_time_s: Optional[float] = None
    startup_warm_cache_hit_s: Optional[float] = None
    startup_prepare_time_s: Optional[float] = None
    prediction_cold_first_time_s: Optional[float] = None
    prediction_cache_prepare_time_s: Optional[float] = None
    prediction_prepared_apply_time_s: Optional[float] = None
    prediction_repeated_median_time_s: Optional[float] = None
    prediction_repeated_p5_time_s: Optional[float] = None
    prediction_repeated_p95_time_s: Optional[float] = None
    prediction_alpha_time_s: Optional[float] = None
    prediction_love_root_time_s: Optional[float] = None
    prediction_x_test_scaling: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryResult:
    """Memory metrics for a single run.

    Records mean, min, max, and variance for GPU memory to capture
    the full memory profile during training/prediction.
    """

    # GPU memory statistics (from background polling during execution)
    gpu_mean_mb: float  # Mean GPU memory during execution
    gpu_min_mb: float  # Minimum GPU memory during execution
    gpu_max_mb: float  # Maximum (peak) GPU memory during execution
    gpu_var_mb: float  # Variance in GPU memory during execution
    # PyTorch-specific memory (from torch.cuda APIs)
    torch_peak_mb: float  # Peak PyTorch tensor allocations
    torch_current_mb: float  # Current PyTorch tensor allocations at end
    # CPU memory (from tracemalloc)
    cpu_peak_mb: float  # Peak CPU memory
    # Metadata
    measurement_method: str  # 'pynvml', 'nvidia-smi', 'torch.cuda', 'none'
    num_samples: int  # Number of memory samples taken
    gpu_baseline_mb: Optional[float] = None
    gpu_current_mb: Optional[float] = None
    gpu_delta_mb: Optional[float] = None
    gpu_isolated_peak_mb: Optional[float] = None
    gpu_isolated_current_mb: Optional[float] = None
    torch_baseline_mb: Optional[float] = None
    torch_peak_delta_mb: Optional[float] = None
    torch_current_delta_mb: Optional[float] = None
    torch_reserved_mb: Optional[float] = None
    torch_reserved_baseline_mb: Optional[float] = None
    torch_reserved_delta_mb: Optional[float] = None
    # Optional route-specific peaks/deltas used for matrix-free memory-law checks
    training_peak_gpu_mb: Optional[float] = None
    training_delta_gpu_mb: Optional[float] = None
    prediction_peak_gpu_mb: Optional[float] = None
    prediction_delta_gpu_mb: Optional[float] = None
    exact_prediction_peak_gpu_mb: Optional[float] = None
    exact_prediction_delta_gpu_mb: Optional[float] = None
    love_prediction_peak_gpu_mb: Optional[float] = None
    love_prediction_delta_gpu_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HyperparameterResult:
    """Learned hyperparameters and recovery errors.

    NOTE: Hyperparameter comparison is INVESTIGATORY ONLY.
    Different frameworks may converge to different local optima
    and still produce equally good predictions.
    """

    learned_lengthscale: float
    learned_noise: float
    learned_outputscale: float
    final_nll: float  # Recorded for information, NOT a primary metric
    learned_mean: Optional[float] = None
    # Recovery errors (only when true params known -- Benchmark B)
    lengthscale_rel_error: Optional[float] = None
    noise_rel_error: Optional[float] = None
    outputscale_rel_error: Optional[float] = None
    mean_rel_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Complete result for a single benchmark run."""

    config: Dict[str, Any]  # {kernel, n, d, method, ...}
    accuracy: AccuracyResult
    speed: SpeedResult
    memory: MemoryResult
    hyperparameters: HyperparameterResult
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "accuracy": self.accuracy.to_dict(),
            "speed": self.speed.to_dict(),
            "memory": self.memory.to_dict(),
            "hyperparameters": self.hyperparameters.to_dict(),
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, cls=NumpyEncoder)


@dataclass
class ComparisonResult:
    """Comparison between MojoGP and GPyTorch.

    Primary comparison metrics are predictive accuracy (RMSE, CRPS, calibration).
    Hyperparameter alignment is recorded for investigation only.
    """

    config: Dict[str, Any]
    mojogp_materialized: Optional[BenchmarkResult] = None
    mojogp_matrix_free: Optional[BenchmarkResult] = None
    gpytorch_cg: Optional[BenchmarkResult] = None
    gpytorch_keops: Optional[BenchmarkResult] = None
    # PRIMARY derived comparisons (predictive accuracy)
    rmse_ratio_vs_cg: Optional[float] = None  # mojogp_rmse / gpytorch_rmse
    crps_ratio_vs_cg: Optional[float] = None  # mojogp_crps / gpytorch_crps
    calibration_diff_vs_cg: Optional[float] = (
        None  # abs difference in calibration error
    )
    # Speed and memory comparisons
    speedup_vs_cg: Optional[float] = None  # gpytorch_time / mojogp_time
    memory_ratio_vs_cg: Optional[float] = None  # mojogp_mem / gpytorch_mem
    # INVESTIGATORY (not pass/fail)
    nll_ratio_vs_cg: Optional[float] = None  # Informational only
    hyperparam_alignment: Optional[Dict[str, float]] = None  # {param: relative_diff}
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def compute_comparisons(self):
        """Compute derived comparison metrics from individual results."""
        # Use materialized as the primary MojoGP result for comparison
        mojogp = self.mojogp_materialized or self.mojogp_matrix_free
        gpytorch = self.gpytorch_cg

        if mojogp and gpytorch:
            # Accuracy ratios (< 1 means MojoGP is better)
            if gpytorch.accuracy.rmse > 0:
                self.rmse_ratio_vs_cg = mojogp.accuracy.rmse / gpytorch.accuracy.rmse
            if gpytorch.accuracy.crps > 0:
                self.crps_ratio_vs_cg = mojogp.accuracy.crps / gpytorch.accuracy.crps

            # Calibration difference
            self.calibration_diff_vs_cg = abs(
                mojogp.accuracy.calibration_error - gpytorch.accuracy.calibration_error
            )

            # Speed (> 1 means MojoGP is faster)
            if mojogp.speed.training_time_s > 0:
                self.speedup_vs_cg = (
                    gpytorch.speed.training_time_s / mojogp.speed.training_time_s
                )

            # Memory (< 1 means MojoGP uses less)
            if gpytorch.memory.gpu_max_mb > 0:
                self.memory_ratio_vs_cg = (
                    mojogp.memory.gpu_max_mb / gpytorch.memory.gpu_max_mb
                )

            # NLL ratio (informational)
            if gpytorch.hyperparameters.final_nll != 0:
                self.nll_ratio_vs_cg = (
                    mojogp.hyperparameters.final_nll
                    / gpytorch.hyperparameters.final_nll
                )

            # Hyperparameter alignment
            self.hyperparam_alignment = {
                "lengthscale": abs(
                    mojogp.hyperparameters.learned_lengthscale
                    - gpytorch.hyperparameters.learned_lengthscale
                )
                / max(gpytorch.hyperparameters.learned_lengthscale, 1e-6),
                "noise": abs(
                    mojogp.hyperparameters.learned_noise
                    - gpytorch.hyperparameters.learned_noise
                )
                / max(gpytorch.hyperparameters.learned_noise, 1e-6),
                "outputscale": abs(
                    mojogp.hyperparameters.learned_outputscale
                    - gpytorch.hyperparameters.learned_outputscale
                )
                / max(gpytorch.hyperparameters.learned_outputscale, 1e-6),
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "mojogp_materialized": self.mojogp_materialized.to_dict()
            if self.mojogp_materialized
            else None,
            "mojogp_matrix_free": self.mojogp_matrix_free.to_dict()
            if self.mojogp_matrix_free
            else None,
            "gpytorch_cg": self.gpytorch_cg.to_dict() if self.gpytorch_cg else None,
            "gpytorch_keops": self.gpytorch_keops.to_dict()
            if self.gpytorch_keops
            else None,
            "rmse_ratio_vs_cg": self.rmse_ratio_vs_cg,
            "crps_ratio_vs_cg": self.crps_ratio_vs_cg,
            "calibration_diff_vs_cg": self.calibration_diff_vs_cg,
            "speedup_vs_cg": self.speedup_vs_cg,
            "memory_ratio_vs_cg": self.memory_ratio_vs_cg,
            "nll_ratio_vs_cg": self.nll_ratio_vs_cg,
            "hyperparam_alignment": self.hyperparam_alignment,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, cls=NumpyEncoder)


def generate_result_filename(test_type: str, config: Dict[str, Any]) -> str:
    """Generate a unique filename for a test result.

    Format: {YYYYMMDD_HHMMSS}_{commit_hash}_{session_id}_{test_type}_{config}_{noise}_{data_type}.json

    The filename includes:
    - Timestamp for chronological ordering
    - Git commit hash for reproducibility
    - Session ID for grouping results from the same pytest run
    - Test type (e.g., 'accuracy', 'comparison')
    - Config details (kernel, n, d, method, etc.)
    - Noise level and data type for filtering
    """
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Build config string from key values
    config_parts = []
    for key in [
        "framework",
        "model_type",
        "kernel",
        "n",
        "d",
        "comparison_class",
        "baseline_backend",
        "training_method",
        "method",
        "prediction_mode",
        "mojogp_preset",
        "specialization_key",
        "num_tasks",
        "keops_supported",
        "keops_used",
        "noise_level",
        "data_type",
    ]:
        if key in config:
            val = config[key]
            if isinstance(val, (str, int)):
                config_parts.append(str(val))
    config_str = "_".join(config_parts) if config_parts else "default"
    return f"{dt}_{_GIT_HASH}_{_SESSION_ID}_{test_type}_{config_str}.json"


def get_session_id() -> str:
    """Get the current session ID for this pytest run."""
    return _SESSION_ID


def get_git_hash() -> str:
    """Get the git commit hash captured at module import."""
    return _GIT_HASH
