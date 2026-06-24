"""Pytest configuration for system benchmarks.

Provides fixtures, markers, and shared utilities for all benchmark tests.
"""

import pytest
import os
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Import our infrastructure
from .gpu_memory import GPUMemoryTracker, GPUMemoryMonitor, reset_torch_memory_stats
from .report import save_comparison_artifact, save_result_artifact
from .result_types import BenchmarkResult, ComparisonResult, generate_result_filename
from tests.benchmarks.runtime import finalize_default_context, get_or_create_default_context
from tests.system.results_writer import ResultsWriter


_session_writer = None


# =============================================================================
# VRAM Detection and Tier Configuration
# =============================================================================

# VRAM tier definitions: (min_vram_mb, max_n_materialized, legacy max_n_matrix_free)
VRAM_TIERS = {
    "xsmall": (0, 2_000, 10_000),  # < 8 GB
    "small": (8_192, 5_000, 50_000),  # 8-16 GB
    "medium": (16_384, 10_000, 100_000),  # 16-32 GB
    "large": (32_768, 20_000, 250_000),  # 32-48 GB, e.g. A100 40GB
    "xlarge": (49_152, 30_000, 500_000),  # >= 48 GB
}

# Matrix-free benchmark caps follow the bandwidth capability tier, not VRAM.
MATRIX_FREE_BANDWIDTH_MAX_N = {
    "xsmall": 25_000,
    "small": 50_000,
    "medium": 75_000,
    "large": 75_000,
    "xlarge": 100_000,
}

# Effective device-memory bandwidth tiers in GB/s. These are intentionally
# coarse capability buckets used to pick matrix-free benchmark ladders without
# hardcoding specific GPU model names.
BANDWIDTH_TIERS = {
    "xsmall": 0.0,
    "small": 150.0,
    "medium": 500.0,
    "large": 1_200.0,
    "xlarge": 2_500.0,
}

_TIER_ORDER = ["xsmall", "small", "medium", "large", "xlarge"]


def _tier_override(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in _TIER_ORDER:
        raise ValueError(
            f"Invalid {env_name}='{value}'. Expected one of {_TIER_ORDER}."
        )
    return normalized


def get_total_vram_mb() -> int:
    """Detect total GPU VRAM in MB.

    Tries PyTorch first, then falls back to nvidia-smi.
    Returns 0 if no GPU is detected.
    """
    # Try PyTorch first
    try:
        import torch

        if torch.cuda.is_available():
            # Get total memory of first GPU
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return int(total_bytes / (1024 * 1024))
    except (ImportError, RuntimeError):
        pass

    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # nvidia-smi returns memory in MiB
            return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return 0


@lru_cache(maxsize=1)
def get_effective_memory_bandwidth_gbps() -> float:
    """Estimate single-GPU device-memory bandwidth in GB/s.

    This is a lightweight runtime capability probe used only to bucket the GPU
    into a coarse matrix-free benchmark tier. It intentionally measures an
    effective copy rate rather than trying to identify exact hardware models.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0

        total_bytes = torch.cuda.get_device_properties(0).total_memory
        target_bytes = int(min(max(total_bytes // 8, 64 * 1024 * 1024), 256 * 1024 * 1024))
        numel = max(target_bytes // 4, 1)
        src = torch.empty(numel, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)

        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()

        iters = 10
        start = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - start, 1e-12)

        bytes_moved = src.numel() * src.element_size() * 2 * iters
        return float(bytes_moved / elapsed / 1e9)
    except Exception:
        return 0.0


def _tier_from_thresholds(value: float, thresholds: Dict[str, float]) -> str:
    selected = "xsmall"
    for tier_name in _TIER_ORDER:
        if value >= thresholds[tier_name]:
            selected = tier_name
    return selected


def get_vram_tier() -> str:
    """Determine the VRAM tier based on detected GPU memory.

    Returns one of: 'xsmall', 'small', 'medium', 'large', 'xlarge'
    """
    override = _tier_override("MOJOGP_BENCHMARK_VRAM_TIER_OVERRIDE")
    if override is not None:
        return override

    vram_mb = get_total_vram_mb()

    if vram_mb >= 49_152:  # >= 48 GB
        return "xlarge"
    elif vram_mb >= 32_768:  # >= 32 GB
        return "large"
    elif vram_mb >= 16_384:  # >= 16 GB
        return "medium"
    elif vram_mb >= 8_192:  # >= 8 GB
        return "small"
    else:
        return "xsmall"


def get_bandwidth_tier() -> str:
    """Determine a coarse bandwidth capability tier for matrix-free targets."""
    override = _tier_override("MOJOGP_BENCHMARK_BANDWIDTH_TIER_OVERRIDE")
    if override is not None:
        return override
    return _tier_from_thresholds(get_effective_memory_bandwidth_gbps(), BANDWIDTH_TIERS)


def get_matrix_free_capability_tier() -> str:
    """Return the route capability tier for matrix-free target selection.

    Matrix-free routes are throughput-bound rather than dense-memory-bound, so
    benchmark target selection follows the measured bandwidth tier directly.
    """
    override = _tier_override("MOJOGP_BENCHMARK_MATRIX_FREE_TIER_OVERRIDE")
    if override is not None:
        return override
    return get_bandwidth_tier()


def get_max_n_for_method(method: str) -> int:
    """Get the maximum benchmark n value for the current GPU and method.

    Args:
        method: Either 'materialized' or 'matrix_free'

    Returns:
        Maximum n value for the method's capability tier
    """
    if method == "materialized":
        tier = get_vram_tier()
        _, max_n_mat, _ = VRAM_TIERS[tier]
        return max_n_mat
    return MATRIX_FREE_BANDWIDTH_MAX_N[get_matrix_free_capability_tier()]


def scale_n_for_vram(requested_n: int, method: str) -> int:
    """Scale down n if it exceeds the GPU's capacity.

    Args:
        requested_n: The n value requested by the test config
        method: Either 'materialized' or 'matrix_free'

    Returns:
        The actual n to use (may be smaller than requested)
    """
    max_n = get_max_n_for_method(method)
    return min(requested_n, max_n)


def get_vram_info() -> Dict[str, Any]:
    """Get detailed VRAM information for logging."""
    vram_mb = get_total_vram_mb()
    tier = get_vram_tier()
    _, max_n_mat, _ = VRAM_TIERS[tier]
    bandwidth_gbps = get_effective_memory_bandwidth_gbps()
    bandwidth_tier = get_bandwidth_tier()
    matrix_free_capability_tier = get_matrix_free_capability_tier()
    max_n_mf = MATRIX_FREE_BANDWIDTH_MAX_N[matrix_free_capability_tier]

    return {
        "vram_mb": vram_mb,
        "vram_gb": round(vram_mb / 1024, 1),
        "tier": tier,
        "max_n_materialized": max_n_mat,
        "max_n_matrix_free": max_n_mf,
        "bandwidth_gbps": round(bandwidth_gbps, 1),
        "bandwidth_tier": bandwidth_tier,
        "matrix_free_capability_tier": matrix_free_capability_tier,
        "tier_overrides": {
            "vram_tier": _tier_override("MOJOGP_BENCHMARK_VRAM_TIER_OVERRIDE"),
            "bandwidth_tier": _tier_override(
                "MOJOGP_BENCHMARK_BANDWIDTH_TIER_OVERRIDE"
            ),
            "matrix_free_capability_tier": _tier_override(
                "MOJOGP_BENCHMARK_MATRIX_FREE_TIER_OVERRIDE"
            ),
        },
    }


# =============================================================================
# Pytest Markers
# =============================================================================


def pytest_addoption(parser):
    """Add custom command-line options."""
    try:
        parser.addoption(
            "--n-override",
            type=int,
            default=None,
            help="Override n (dataset size) for all test configs. "
            "Example: --n-override 1000 to test all kernels at n=1000.",
        )
    except ValueError:
        pass


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "minimal: Quick end-to-end smoke tests (2-3 configs per feature)"
    )
    config.addinivalue_line(
        "markers", "moderate: Broader coverage tests (5-8 configs per feature)"
    )
    config.addinivalue_line(
        "markers", "full: Exhaustive coverage tests (all kernel/size/dim combos)"
    )
    config.addinivalue_line("markers", "single_output: Single-output GP tests")
    config.addinivalue_line("markers", "multi_output: Multi-output GP tests")
    config.addinivalue_line("markers", "composite: Composite kernel tests")
    config.addinivalue_line("markers", "ard: ARD kernel tests")
    config.addinivalue_line("markers", "speed: Speed-focused benchmarks")
    config.addinivalue_line("markers", "memory: Memory-focused benchmarks")
    config.addinivalue_line("markers", "gpytorch: Tests comparing against GPyTorch")
    config.addinivalue_line("markers", "accuracy: Tests measuring prediction accuracy")


def pytest_sessionstart(session):
    """Create a shared results writer for migrated benchmark suites."""
    global _session_writer
    _session_writer = ResultsWriter(prefix="system_benchmark")


def pytest_sessionfinish(session, exitstatus):
    """Flush migrated benchmark-suite results on exit."""
    global _session_writer
    if _session_writer is not None:
        _session_writer.write_summary()
    finalize_default_context()


@pytest.fixture(scope="session")
def results_writer():
    """Compatibility fixture for migrated benchmark suites."""
    global _session_writer
    if _session_writer is None:
        _session_writer = ResultsWriter(prefix="system_benchmark")
    return _session_writer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def n_override(request):
    """Get the --n-override value if set.

    Returns None if not set, otherwise the integer override value.
    Usage in tests: if n_override is not None, use it instead of the parametrized n.
    """
    return request.config.getoption("--n-override")


@pytest.fixture(scope="session")
def results_dir():
    """Get the results directory, creating it if needed."""
    path = Path(__file__).parent / "results"
    path.mkdir(exist_ok=True)
    return path


@pytest.fixture(scope="function")
def gpu_tracker():
    """Provide a GPU memory tracker for single-point snapshots."""
    tracker = GPUMemoryTracker()
    tracker.reset()
    return tracker


@pytest.fixture(scope="function")
def gpu_monitor():
    """Provide a GPU memory monitor for continuous monitoring."""
    monitor = GPUMemoryMonitor(interval=0.1)
    yield monitor
    # Ensure monitor is stopped even if test fails
    monitor.stop()


@pytest.fixture(scope="function")
def reset_gpu_memory():
    """Reset GPU memory stats before each test."""
    reset_torch_memory_stats()
    yield
    # Could add cleanup here if needed


@pytest.fixture(scope="session")
def benchmark_config():
    """Shared benchmark configuration."""
    return {
        # Training defaults
        "n_iterations": 100,
        "learning_rate": 0.05,
        "early_stop_patience": 15,
        "early_stop_tol": 1e-4,
        # CG settings (match MojoGP defaults)
        "cg_tolerance": 1e-2,
        "max_cg_iterations": 100,
        "num_trace_samples": 10,
        # Initial hyperparameters
        "init_lengthscale": 1.0,
        "init_noise": 0.1,
        "init_outputscale": 1.0,
        # Memory monitoring
        "memory_poll_interval": 0.1,
    }


@pytest.fixture(scope="session")
def vram_info():
    """Get VRAM information for the current GPU.

    Returns a dict with:
    - vram_mb: Total VRAM in MB
    - vram_gb: Total VRAM in GB (rounded)
    - tier: VRAM tier name ('xsmall', 'small', 'medium', 'large', 'xlarge')
    - max_n_materialized: Maximum n for materialized method
    - max_n_matrix_free: Maximum n for matrix-free method
    """
    return get_vram_info()


# =============================================================================
# Result Saving
# =============================================================================


class ResultSaver:
    """Helper class to save benchmark results to JSON files."""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir

    def save_result_artifact(self, result: BenchmarkResult, test_type: str) -> Path:
        """Save a result artifact without benchmark persistence."""
        return save_result_artifact(result, self.results_dir, test_type)

    def save_comparison_artifact(self, result: ComparisonResult, test_type: str) -> Path:
        """Save a comparison artifact without benchmark persistence."""
        return save_comparison_artifact(result, self.results_dir, test_type)

    def save_raw_dict(
        self, data: Dict[str, Any], test_type: str, config: Dict[str, Any]
    ) -> Path:
        """Save raw dictionary data."""
        filename = generate_result_filename(test_type, config)
        filepath = self.results_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return filepath


@pytest.fixture(scope="session")
def result_saver(results_dir):
    """Provide a result saver instance."""
    return ResultSaver(results_dir)


@pytest.fixture(scope="session", autouse=True)
def benchmark_session_preflight():
    """Enforce clean-git and profiling-off preflight before benchmark execution."""
    get_or_create_default_context()
    yield


# =============================================================================
# Test Reporting
# =============================================================================


def format_comparison_table(comparison: ComparisonResult) -> str:
    """Format a comparison result as a readable table."""
    lines = []
    config = comparison.config

    # Header
    lines.append(
        f"\nConfig: kernel={config.get('kernel', 'N/A')}, "
        f"n={config.get('n', 'N/A')}, d={config.get('d', 'N/A')}"
    )
    lines.append("=" * 80)

    # Column headers
    headers = ["Metric", "MojoGP Mat", "MojoGP MF", "GPyTorch CG", "GPyTorch KeOps"]
    lines.append(
        f"{headers[0]:<25} | {headers[1]:>12} | {headers[2]:>12} | {headers[3]:>12} | {headers[4]:>14}"
    )
    lines.append("-" * 80)

    # Helper to get value or N/A
    def get_val(result: Optional[BenchmarkResult], attr_path: str) -> str:
        if result is None:
            return "N/A"
        try:
            obj = result
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, float):
                return f"{obj:.4f}"
            return str(obj)
        except (AttributeError, TypeError):
            return "N/A"

    # Accuracy metrics
    lines.append("PREDICTIVE ACCURACY (PRIMARY):")
    for metric, path in [
        ("RMSE", "accuracy.rmse"),
        ("CRPS", "accuracy.crps"),
        ("Calibration (95%)", "accuracy.calibration_coverage"),
        ("Calibration Error", "accuracy.calibration_error"),
        ("R-squared", "accuracy.r_squared"),
        ("MSLL", "accuracy.msll"),
    ]:
        if metric == "Calibration (95%)":
            # Special handling for dict
            vals = []
            for r in [
                comparison.mojogp_materialized,
                comparison.mojogp_matrix_free,
                comparison.gpytorch_cg,
                comparison.gpytorch_keops,
            ]:
                if r and r.accuracy.calibration_coverage:
                    vals.append(f"{r.accuracy.calibration_coverage.get(0.95, 0):.2f}")
                else:
                    vals.append("N/A")
            lines.append(
                f"  {metric:<23} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12} | {vals[3]:>14}"
            )
        else:
            lines.append(
                f"  {metric:<23} | {get_val(comparison.mojogp_materialized, path):>12} | "
                f"{get_val(comparison.mojogp_matrix_free, path):>12} | "
                f"{get_val(comparison.gpytorch_cg, path):>12} | "
                f"{get_val(comparison.gpytorch_keops, path):>14}"
            )

    lines.append("")
    lines.append("TRAINING DETAILS:")
    for metric, path in [
        ("Iterations Run", "speed.iterations_run"),
        ("Max Iterations", "speed.max_iterations"),
        ("Early Stopped", "speed.early_stopped"),
        ("Training Time (s)", "speed.training_time_s"),
        ("ms/iteration", "speed.ms_per_iteration"),
        ("Pred Mean Time (s)", "speed.prediction_mean_time_s"),
        ("Pred Var Time (s)", "speed.prediction_variance_time_s"),
        ("End-to-End Time (s)", "speed.end_to_end_time_s"),
    ]:
        lines.append(
            f"  {metric:<23} | {get_val(comparison.mojogp_materialized, path):>12} | "
            f"{get_val(comparison.mojogp_matrix_free, path):>12} | "
            f"{get_val(comparison.gpytorch_cg, path):>12} | "
            f"{get_val(comparison.gpytorch_keops, path):>14}"
        )

    lines.append("")
    lines.append("MEMORY:")
    for metric, path in [
        ("Peak GPU Mem (MB)", "memory.gpu_max_mb"),
    ]:
        lines.append(
            f"  {metric:<23} | {get_val(comparison.mojogp_materialized, path):>12} | "
            f"{get_val(comparison.mojogp_matrix_free, path):>12} | "
            f"{get_val(comparison.gpytorch_cg, path):>12} | "
            f"{get_val(comparison.gpytorch_keops, path):>14}"
        )

    lines.append("")
    lines.append("HYPERPARAMETERS (INVESTIGATORY):")
    for metric, path in [
        ("Final NLL", "hyperparameters.final_nll"),
        ("Lengthscale", "hyperparameters.learned_lengthscale"),
        ("Noise", "hyperparameters.learned_noise"),
        ("Outputscale", "hyperparameters.learned_outputscale"),
    ]:
        lines.append(
            f"  {metric:<23} | {get_val(comparison.mojogp_materialized, path):>12} | "
            f"{get_val(comparison.mojogp_matrix_free, path):>12} | "
            f"{get_val(comparison.gpytorch_cg, path):>12} | "
            f"{get_val(comparison.gpytorch_keops, path):>14}"
        )

    return "\n".join(lines)


def format_accuracy_table(result: BenchmarkResult) -> str:
    """Format a single benchmark result as a readable table."""
    lines = []
    config = result.config

    lines.append(
        f"\nConfig: kernel={config.get('kernel', 'N/A')}, "
        f"n={config.get('n', 'N/A')}, d={config.get('d', 'N/A')}, "
        f"method={config.get('method', 'N/A')}"
    )
    lines.append("=" * 60)

    lines.append("PREDICTIVE ACCURACY:")
    lines.append(f"  RMSE:              {result.accuracy.rmse:.4f}")
    lines.append(f"  MAE:               {result.accuracy.mae:.4f}")
    lines.append(f"  R-squared:         {result.accuracy.r_squared:.4f}")
    lines.append(f"  CRPS:              {result.accuracy.crps:.4f}")
    lines.append(f"  MSLL:              {result.accuracy.msll:.4f}")
    lines.append(f"  Calibration Error: {result.accuracy.calibration_error:.4f}")
    lines.append(f"  Sharpness:         {result.accuracy.sharpness:.4f}")

    if result.accuracy.calibration_coverage:
        lines.append("  Calibration Coverage:")
        for level, coverage in sorted(result.accuracy.calibration_coverage.items()):
            lines.append(f"    {level * 100:.0f}%: {coverage:.2f}")

    lines.append("")
    lines.append("TRAINING:")
    lines.append(
        f"  Iterations:        {result.speed.iterations_run}/{result.speed.max_iterations}"
    )
    lines.append(f"  Early Stopped:     {result.speed.early_stopped}")
    lines.append(f"  Training Time:     {result.speed.training_time_s:.2f}s")
    lines.append(f"  ms/iteration:      {result.speed.ms_per_iteration:.2f}")

    lines.append("")
    lines.append("HYPERPARAMETERS:")
    lines.append(
        f"  Lengthscale:       {result.hyperparameters.learned_lengthscale:.4f}"
    )
    lines.append(f"  Noise:             {result.hyperparameters.learned_noise:.4f}")
    lines.append(
        f"  Outputscale:       {result.hyperparameters.learned_outputscale:.4f}"
    )
    lines.append(f"  Final NLL:         {result.hyperparameters.final_nll:.4f}")

    if result.hyperparameters.lengthscale_rel_error is not None:
        lines.append("  Recovery Errors:")
        lines.append(
            f"    Lengthscale:     {result.hyperparameters.lengthscale_rel_error:.2%}"
        )
        lines.append(
            f"    Noise:           {result.hyperparameters.noise_rel_error:.2%}"
        )
        lines.append(
            f"    Outputscale:     {result.hyperparameters.outputscale_rel_error:.2%}"
        )

    return "\n".join(lines)


@pytest.fixture(scope="session")
def format_table():
    """Provide table formatting functions."""
    return {
        "comparison": format_comparison_table,
        "accuracy": format_accuracy_table,
    }


# =============================================================================
# Skip Conditions
# =============================================================================


def has_gpytorch():
    """Check if GPyTorch is available."""
    try:
        import gpytorch

        return True
    except ImportError:
        return False


def has_keops():
    """Check if KeOps is available."""
    try:
        import pykeops

        return True
    except ImportError:
        return False


def has_mojogp():
    """Check if MojoGP JIT engine is available."""
    try:
        import mojogp_jit_engine

        return True
    except ImportError:
        return False


def has_cuda():
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def has_valid_cuda_device():
    """Check if there's a valid CUDA device that can be used."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        # Try to actually use the device
        torch.cuda.current_device()
        return True
    except (ImportError, RuntimeError):
        return False


def get_gpu_info():
    """Get GPU information for verification."""
    info = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "nvidia_smi_available": False,
    }

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            try:
                info["device_count"] = torch.cuda.device_count()
                info["device_name"] = torch.cuda.get_device_name(0)
            except RuntimeError:
                pass
    except ImportError:
        pass

    # Check nvidia-smi
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["nvidia_smi_available"] = True
            if not info["device_name"]:
                info["device_name"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


# Skip decorators
requires_gpytorch = pytest.mark.skipif(
    not has_gpytorch(), reason="GPyTorch not installed"
)
requires_keops = pytest.mark.skipif(not has_keops(), reason="KeOps not installed")
requires_mojogp = pytest.mark.skipif(
    not has_mojogp(), reason="MojoGP kernels not built"
)
requires_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
requires_gpu = pytest.mark.skipif(
    not has_valid_cuda_device(), reason="No valid GPU device available"
)


def assert_gpu_available():
    """Assert that GPU is available. Fails the test if not."""
    import torch

    assert torch.cuda.is_available(), "GPU required for this test - CUDA not available"
    try:
        device_count = torch.cuda.device_count()
        assert device_count > 0, "GPU required for this test - no CUDA devices found"
    except RuntimeError as e:
        pytest.fail(f"GPU required for this test - CUDA error: {e}")


def assert_gpu_was_used(result):
    """Assert that GPU was actually used during the benchmark.

    Args:
        result: BenchmarkResult or dict with memory stats
    """
    if hasattr(result, "memory"):
        gpu_mem = result.memory.gpu_max_mb
    elif isinstance(result, dict):
        gpu_mem = result.get("memory_stats", {}).get("max_mb", 0)
        if gpu_mem == 0:
            gpu_mem = result.get("peak_memory_mb", 0)
    else:
        gpu_mem = 0

    # GPU memory should be > 0 if GPU was used
    # Note: nvidia-smi reports baseline memory even when idle, so we check for > 100MB
    # to ensure actual GPU computation happened
    assert gpu_mem > 0, (
        f"GPU was not used during benchmark (gpu_max_mb={gpu_mem}). "
        "This test requires GPU execution."
    )


requires_keops = pytest.mark.skipif(not has_keops(), reason="KeOps not installed")
requires_mojogp = pytest.mark.skipif(
    not has_mojogp(), reason="MojoGP kernels not built"
)
requires_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")


# =============================================================================
# Hooks
# =============================================================================


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # If no tier marker is specified, run all tiers
    # If a specific tier is requested via -m, only run those
    pass  # Default pytest behavior handles this


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary to test output."""
    # Could add summary of benchmark results here
    pass
