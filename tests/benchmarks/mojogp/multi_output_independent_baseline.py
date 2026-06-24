"""Helpers for multi-output benchmark baselines built from independent ExactGP tasks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.system_benchmarks.run_multi_output_accuracy_independent_case"


def run_independent_exactgp_baseline(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kernel: str,
    method: str,
    max_iterations: int,
    learning_rate: float,
    task_idx: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    def _load_independent_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        baseline = np.load(path)
        return (
            np.asarray(baseline["mean"], dtype=np.float32),
            np.asarray(baseline["variance"], dtype=np.float32),
            {"training_time_s": float(baseline["fit_time"][0])},
        )

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as input_f:
        input_path = Path(input_f.name)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as output_f:
        output_path = Path(output_f.name)
    try:
        np.savez(
            input_path,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
        )
        return run_isolated_case(
            module=MODULE,
            payload={
                "kernel": kernel,
                "method": method,
                "max_iterations": max_iterations,
                "learning_rate": learning_rate,
                "input_path": str(input_path),
                "output_path": str(output_path),
            },
            timeout=1200,
            description=(
                "Runs independent ExactGP baseline "
                f"kernel={kernel} method={method} task={task_idx}"
            ),
            result_loader=_load_independent_result,
        )
    finally:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()
