"""Results writer for system tests.

Provides utilities for writing test results to JSON files in datetime-stamped folders.
"""

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def get_results_dir() -> Path:
    """Get the base results directory."""
    return Path(__file__).parent.parent.parent / "results"


def create_session_dir(prefix: str = "system_test") -> Path:
    """Create a unique session directory with datetime stamp.

    Args:
        prefix: Prefix for the directory name

    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = get_results_dir() / f"{prefix}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        # Try MAX API first
        from max.diagnostics.gpu import GPUDiagContext

        with GPUDiagContext() as ctx:
            stats = ctx.get_stats()
            for gpu_id, gpu_stats in stats.items():
                return gpu_stats.memory.used_bytes / 1024**2
    except Exception:
        pass

    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass

    return 0.0


def get_system_info() -> Dict[str, Any]:
    """Get system information for the test run."""
    import torch
    import platform

    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

        # Get GPU memory info
        try:
            from max.diagnostics.gpu import GPUDiagContext

            with GPUDiagContext() as ctx:
                stats = ctx.get_stats()
                for gpu_id, gpu_stats in stats.items():
                    info["gpu_total_memory_mb"] = gpu_stats.memory.total_bytes / 1024**2
                    break
        except Exception:
            pass

    return info


def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass to dict, handling nested dataclasses and numpy arrays."""
    import numpy as np

    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    else:
        return obj


class ResultsWriter:
    """Writer for test results to JSON files."""

    def __init__(self, session_dir: Optional[Path] = None, prefix: str = "system_test"):
        """Initialize the results writer.

        Args:
            session_dir: Directory to write results to. If None, creates a new one.
            prefix: Prefix for auto-created session directory
        """
        if session_dir is None:
            self.session_dir = create_session_dir(prefix)
        else:
            self.session_dir = Path(session_dir)
            self.session_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict[str, Any]] = []
        self.system_info = get_system_info()

        # Write system info
        self._write_json("system_info.json", self.system_info)

        print(f"\nResults will be written to: {self.session_dir}")

    def _write_json(self, filename: str, data: Any) -> Path:
        """Write data to a JSON file."""
        filepath = self.session_dir / filename
        with open(filepath, "w") as f:
            json.dump(dataclass_to_dict(data), f, indent=2, default=str)
        return filepath

    def add_result(self, result: Any, test_name: Optional[str] = None) -> None:
        """Add a test result.

        Args:
            result: Test result (dataclass or dict)
            test_name: Optional test name for the individual result file
        """
        result_dict = dataclass_to_dict(result)
        result_dict["_recorded_at"] = datetime.now().isoformat()
        self.results.append(result_dict)

        # Write individual result file if test_name provided
        if test_name:
            safe_name = (
                test_name.replace(" ", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
            )
            self._write_json(f"result_{safe_name}.json", result_dict)

    def write_summary(self) -> Path:
        """Write summary of all results."""
        summary = {
            "system_info": self.system_info,
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.get("success", True)),
            "failed_tests": sum(1 for r in self.results if not r.get("success", True)),
            "results": self.results,
        }

        # Compute aggregate statistics
        mojo_times = [
            r.get("mojo_training_time_s")
            for r in self.results
            if r.get("mojo_training_time_s")
        ]
        gpy_times = [
            r.get("gpy_training_time_s")
            for r in self.results
            if r.get("gpy_training_time_s")
        ]
        mojo_rmses = [
            r.get("mojo_rmse_vs_truth")
            for r in self.results
            if r.get("mojo_rmse_vs_truth")
        ]
        gpy_rmses = [
            r.get("gpy_rmse_vs_truth")
            for r in self.results
            if r.get("gpy_rmse_vs_truth")
        ]

        if mojo_times:
            import numpy as np

            summary["aggregate"] = {
                "mojo_avg_training_time_s": float(np.mean(mojo_times)),
                "gpy_avg_training_time_s": float(np.mean(gpy_times))
                if gpy_times
                else None,
                "mojo_avg_rmse": float(np.mean(mojo_rmses)) if mojo_rmses else None,
                "gpy_avg_rmse": float(np.mean(gpy_rmses)) if gpy_rmses else None,
                "avg_speedup": float(np.mean(gpy_times) / np.mean(mojo_times))
                if gpy_times and mojo_times
                else None,
            }

        filepath = self._write_json("summary.json", summary)
        print(f"\nSummary written to: {filepath}")
        return filepath

    def get_session_dir(self) -> Path:
        """Get the session directory path."""
        return self.session_dir


# Global writer instance for pytest fixtures
_global_writer: Optional[ResultsWriter] = None


def get_global_writer(prefix: str = "system_test") -> ResultsWriter:
    """Get or create the global results writer."""
    global _global_writer
    if _global_writer is None:
        _global_writer = ResultsWriter(prefix=prefix)
    return _global_writer


def reset_global_writer() -> None:
    """Reset the global writer (for new test sessions)."""
    global _global_writer
    if _global_writer is not None:
        _global_writer.write_summary()
    _global_writer = None
