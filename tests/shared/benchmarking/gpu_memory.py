"""GPU memory measurement utilities.

Provides lightweight memory tracking with fallback chain:
pynvml -> nvidia-smi subprocess -> torch.cuda

IMPORTANT: Memory tracking must NOT impact benchmark performance.
Accurate timing is more important than accurate memory measurement.
"""

import subprocess
import threading
import time
from typing import Dict, List, Optional
import numpy as np

# Try to import torch for CUDA memory tracking
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import pynvml for direct GPU memory access
try:
    import pynvml

    pynvml.nvmlInit()
    HAS_PYNVML = True
except (ImportError, Exception):
    HAS_PYNVML = False


class GPUMemoryTracker:
    """Lightweight GPU memory tracker with fallback chain.

    Provides single-point snapshots (no polling overhead).
    Use GPUMemoryMonitor for continuous monitoring with statistics.
    """

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._method = self._detect_method()
        self._baseline_mb = 0.0

        if HAS_PYNVML:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        else:
            self._handle = None

    def _detect_method(self) -> str:
        """Detect the best available memory tracking method."""
        if HAS_PYNVML:
            return "pynvml"

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "nvidia-smi"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if HAS_TORCH and torch.cuda.is_available():
            return "torch.cuda"

        return "none"

    def reset(self):
        """Reset peak memory counters and record baseline."""
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self._baseline_mb = self._get_current_gpu_mb()

    def _get_current_gpu_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if self._method == "pynvml" and self._handle:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024 * 1024)

        elif self._method == "nvidia-smi":
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        f"--id={self._device_index}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            except (subprocess.TimeoutExpired, ValueError):
                pass
            return 0.0

        elif self._method == "torch.cuda" and HAS_TORCH:
            return torch.cuda.memory_allocated(self._device_index) / (1024 * 1024)

        return 0.0

    def snapshot(self) -> Dict[str, float]:
        """Take a memory snapshot.

        Returns:
            dict with keys: peak_gpu_mb, current_gpu_mb, baseline_gpu_mb,
                           delta_gpu_mb, torch_peak_mb, torch_current_mb, method
        """
        current_gpu_mb = self._get_current_gpu_mb()

        torch_peak_mb = 0.0
        torch_current_mb = 0.0
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch_peak_mb = torch.cuda.max_memory_allocated(self._device_index) / (
                1024 * 1024
            )
            torch_current_mb = torch.cuda.memory_allocated(self._device_index) / (
                1024 * 1024
            )

        return {
            "peak_gpu_mb": current_gpu_mb,  # Best estimate of peak
            "current_gpu_mb": current_gpu_mb,
            "baseline_gpu_mb": self._baseline_mb,
            "delta_gpu_mb": max(current_gpu_mb - self._baseline_mb, 0.0),
            "torch_peak_mb": torch_peak_mb,
            "torch_current_mb": torch_current_mb,
            "method": self._method,
        }


class GPUMemoryMonitor:
    """Background thread that polls GPU memory at regular intervals.

    IMPORTANT: Uses a background thread with configurable polling interval
    to minimize performance impact. Default interval is 100ms which adds
    negligible overhead.

    For very short benchmarks (<1s), consider disabling monitoring or using
    GPUMemoryTracker for single snapshots instead.
    """

    def __init__(self, interval: float = 0.1, device_index: int = 0):
        """Initialize the memory monitor.

        Args:
            interval: Polling interval in seconds. Default 0.1 (100ms).
            device_index: GPU device index to monitor.
        """
        self._device_index = device_index
        self._method = self._detect_method()

        # Adjust interval for slow methods
        if self._method == "nvidia-smi" and interval < 0.5:
            self._interval = 0.5  # Minimum 500ms for subprocess calls
        else:
            self._interval = interval

        self._samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if HAS_PYNVML:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        else:
            self._handle = None

    def _detect_method(self) -> str:
        """Detect the best available memory tracking method."""
        if HAS_PYNVML:
            return "pynvml"

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "nvidia-smi"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if HAS_TORCH and torch.cuda.is_available():
            return "torch.cuda"

        return "none"

    def _get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if self._method == "pynvml" and self._handle:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024 * 1024)

        elif self._method == "nvidia-smi":
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        f"--id={self._device_index}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    return float(result.stdout.strip())
            except (subprocess.TimeoutExpired, ValueError):
                pass
            return 0.0

        elif self._method == "torch.cuda" and HAS_TORCH:
            return torch.cuda.memory_allocated(self._device_index) / (1024 * 1024)

        return 0.0

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            mem = self._get_memory_mb()
            self._samples.append(mem)
            time.sleep(self._interval)

    def start(self):
        """Start background memory monitoring."""
        if self._method == "none":
            return

        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics (mean, min, max, var, std, samples).

        Returns:
            dict with keys: mean_mb, min_mb, max_mb, var_mb, std_mb, samples, method
        """
        if not self._samples:
            return {
                "mean_mb": 0.0,
                "min_mb": 0.0,
                "max_mb": 0.0,
                "var_mb": 0.0,
                "std_mb": 0.0,
                "samples": 0,
                "method": self._method,
            }

        arr = np.array(self._samples)
        return {
            "mean_mb": float(np.mean(arr)),
            "min_mb": float(np.min(arr)),
            "max_mb": float(np.max(arr)),
            "var_mb": float(np.var(arr)),
            "std_mb": float(np.std(arr)),
            "samples": len(arr),
            "method": self._method,
        }


def measure_gpu_phase(run_fn, *, interval: float = 0.01) -> tuple[object, Dict[str, float]]:
    """Execute one workload segment with isolated GPU memory telemetry.

    This is intentionally lightweight: reset the baseline, run the workload,
    and return both the result and a small telemetry bundle with baseline, peak,
    and delta GPU memory.
    """

    reset_torch_memory_stats()
    tracker = GPUMemoryTracker()
    tracker.reset()
    monitor = GPUMemoryMonitor(interval=interval)
    monitor.start()
    try:
        result = run_fn()
    finally:
        monitor.stop()

    snapshot = tracker.snapshot()
    stats = monitor.get_stats()
    stats.update(get_torch_memory_stats())
    phase_peak_gpu_mb = max(
        float(stats.get("max_mb", 0.0)),
        float(snapshot.get("peak_gpu_mb", 0.0)),
    )
    phase_baseline_gpu_mb = float(snapshot.get("baseline_gpu_mb", 0.0))
    stats["phase_peak_gpu_mb"] = phase_peak_gpu_mb
    stats["phase_baseline_gpu_mb"] = phase_baseline_gpu_mb
    stats["phase_delta_gpu_mb"] = max(
        phase_peak_gpu_mb - phase_baseline_gpu_mb,
        float(snapshot.get("delta_gpu_mb", 0.0)),
    )
    return result, stats


def get_torch_memory_stats(device_index: int = 0) -> Dict[str, float]:
    """Get PyTorch CUDA memory statistics.

    This is a lightweight call that doesn't require polling.
    """
    if not HAS_TORCH or not torch.cuda.is_available():
        return {
            "torch_peak_mb": 0.0,
            "torch_current_mb": 0.0,
            "torch_reserved_mb": 0.0,
        }

    try:
        torch.cuda.synchronize()
        return {
            "torch_peak_mb": torch.cuda.max_memory_allocated(device_index)
            / (1024 * 1024),
            "torch_current_mb": torch.cuda.memory_allocated(device_index)
            / (1024 * 1024),
            "torch_reserved_mb": torch.cuda.memory_reserved(device_index)
            / (1024 * 1024),
        }
    except RuntimeError:
        # No valid CUDA device
        return {
            "torch_peak_mb": 0.0,
            "torch_current_mb": 0.0,
            "torch_reserved_mb": 0.0,
        }


def reset_torch_memory_stats(device_index: int = 0):
    """Reset PyTorch CUDA peak memory statistics."""
    if HAS_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device_index)
            torch.cuda.synchronize()
        except RuntimeError:
            # No valid CUDA device
            pass
