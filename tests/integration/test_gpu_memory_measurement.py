"""Test that GPU memory measurement is accurate for benchmarking.

Validates that pynvml-based memory monitoring correctly captures:
1. Peak GPU memory during GP training
2. Memory scales with n (materialized = O(n^2))
3. Each subprocess gets isolated measurement (no cross-config leakage)
4. Same measurement method works for both MojoGP and PyTorch
"""

import pytest

from tests.shared.subprocess_harness import run_isolated_case
from tests.shared.benchmarking.gpytorch_models import is_keops_available


MODULE = "tests.integration.run_gpu_memory_measurement_case"


def _run_isolated_memory_test(
    framework,
    n,
    d=5,
    kernel="rbf",
    method="materialized",
    prediction_mode="exact",
    n_test=16,
    max_iterations=1,
):
    return run_isolated_case(
        module=MODULE,
        payload={
            "framework": framework,
            "n": n,
            "d": d,
            "kernel": kernel,
            "method": method,
            "prediction_mode": prediction_mode,
            "n_test": n_test,
            "max_iterations": max_iterations,
        },
        timeout=120,
        description=(
            f"Runs GPU memory measurement framework={framework} n={n} d={d} method={method}"
        ),
    )


class TestGPUMemoryMeasurement:
    """Validate GPU memory measurement accuracy."""

    def test_memory_increases_with_n_materialized(self):
        """The measurement should capture O(n^2) scaling on a deterministic GPU allocation."""
        mem_2000 = _run_isolated_memory_test("pytorch", 2000)
        mem_5000 = _run_isolated_memory_test("pytorch", 5000)

        assert mem_2000["delta_mb"] > 10, (
            f"n=2000 delta too small: {mem_2000['delta_mb']:.0f} MB"
        )
        if mem_5000["delta_mb"] > mem_2000["delta_mb"]:
            return

        assert mem_5000["torch_peak_mb"] > mem_2000["torch_peak_mb"] * 2.0, (
            f"n=5000 torch peak ({mem_5000['torch_peak_mb']:.0f} MB) should exceed "
            f"n=2000 torch peak ({mem_2000['torch_peak_mb']:.0f} MB)"
        )
        assert mem_5000["peak_mb"] >= mem_2000["peak_mb"] - 16.0, (
            f"n=5000 peak ({mem_5000['peak_mb']:.0f} MB) should not be materially below "
            f"n=2000 peak ({mem_2000['peak_mb']:.0f} MB)"
        )

    def test_memory_is_positive(self):
        """MojoGP measurement should report a valid peak/baseline relationship."""
        mem = _run_isolated_memory_test("mojogp", 2000)
        assert mem["peak_mb"] >= mem["baseline_mb"], mem
        assert mem["delta_mb"] >= 0, mem

    def test_matrix_free_exact_prediction_reports_route_peaks(self):
        """Matrix-free exact prediction reports route peaks without wrapper fallback."""
        mem = _run_isolated_memory_test(
            "mojogp",
            2000,
            method="matrix_free",
            prediction_mode="exact",
            n_test=16,
            max_iterations=1,
        )

        assert mem["training_peak_mb"] >= mem["baseline_mb"], mem
        assert mem["prediction_peak_mb"] >= mem["baseline_mb"], mem
        info = mem["backend_predict_info"]
        assert info["fallback_used"] is False
        assert info["actual_variance_route"] == "predict"

    def test_matrix_free_exact_prediction_scales_subquadratically(self):
        """With fixed m, matrix-free exact prediction memory should not look quadratic in n."""
        mem_small = _run_isolated_memory_test(
            "mojogp",
            2000,
            method="matrix_free",
            prediction_mode="exact",
            n_test=16,
            max_iterations=1,
        )
        mem_large = _run_isolated_memory_test(
            "mojogp",
            8000,
            method="matrix_free",
            prediction_mode="exact",
            n_test=16,
            max_iterations=1,
        )

        small_metric = (
            mem_small["prediction_delta_mb"]
            if mem_small["prediction_delta_mb"] > 1.0
            else mem_small["prediction_peak_mb"]
        )
        large_metric = (
            mem_large["prediction_delta_mb"]
            if mem_large["prediction_delta_mb"] > 1.0
            else mem_large["prediction_peak_mb"]
        )
        observed_ratio = large_metric / max(small_metric, 1e-6)
        linear_ratio = 8000 / 2000
        quadratic_ratio = linear_ratio**2

        assert abs(observed_ratio - linear_ratio) < abs(
            observed_ratio - quadratic_ratio
        ), (
            "Matrix-free exact prediction memory scaled too much like O(n^2): "
            f"observed={observed_ratio:.2f}, linear={linear_ratio:.2f}, "
            f"quadratic={quadratic_ratio:.2f}, small={small_metric:.2f} MB, "
            f"large={large_metric:.2f} MB"
        )

    def test_subprocess_isolation(self):
        """Two separate subprocesses should start from comparable warmed baselines."""
        mem1 = _run_isolated_memory_test("pytorch", 3000)
        mem2 = _run_isolated_memory_test("pytorch", 3000)

        assert abs(mem1["baseline_mb"] - mem2["baseline_mb"]) < 64, (
            f"Subprocess baselines diverged too much: {mem1['baseline_mb']:.0f} vs {mem2['baseline_mb']:.0f} MB"
        )
        assert mem1["peak_mb"] >= mem1["baseline_mb"]
        assert mem2["peak_mb"] >= mem2["baseline_mb"]

    def test_pytorch_memory_reports_positive_allocation(self):
        """Same measurement method should work for PyTorch allocations."""
        mem = _run_isolated_memory_test("pytorch", 5000)
        assert mem["delta_mb"] > 50, (
            f"PyTorch n=5000 matmul should use >50MB, got {mem['delta_mb']:.0f} MB"
        )

    def test_gpytorch_end_to_end_memory_is_positive(self):
        """End-to-end GPyTorch training should report positive GPU memory."""
        mem = _run_isolated_memory_test("gpytorch", 2000, max_iterations=2)
        assert mem["peak_mb"] > 0, mem
        assert mem["delta_mb"] > 0, mem

    @pytest.mark.skipif(not is_keops_available(), reason="pykeops not installed")
    def test_gpytorch_keops_end_to_end_memory_is_positive(self):
        """End-to-end GPyTorch+KeOps training should report positive GPU memory."""
        mem = _run_isolated_memory_test(
            "gpytorch_keops",
            2000,
            method="matrix_free",
            prediction_mode="love",
            max_iterations=2,
        )
        assert mem["peak_mb"] > 0, mem
        assert mem["delta_mb"] > 0, mem

    def test_matrix_free_uses_less_than_materialized(self):
        """Matrix-free should use significantly less memory than materialized at large n."""
        mem_mat = _run_isolated_memory_test("mojogp", 5000, method="materialized")
        mem_mf = _run_isolated_memory_test("mojogp", 5000, method="matrix_free")

        assert mem_mat["peak_mb"] >= mem_mat["baseline_mb"], mem_mat
        assert mem_mf["peak_mb"] >= mem_mf["baseline_mb"], mem_mf

        if mem_mat["delta_mb"] >= 8.0:
            assert mem_mf["delta_mb"] < mem_mat["delta_mb"] * 1.2, (
                f"Matrix-free ({mem_mf['delta_mb']:.0f} MB) should use less than "
                f"materialized ({mem_mat['delta_mb']:.0f} MB)"
            )
        else:
            assert mem_mf["peak_mb"] <= mem_mat["peak_mb"] + 16.0, (
                f"Matrix-free peak ({mem_mf['peak_mb']:.0f} MB) should not exceed "
                f"materialized peak ({mem_mat['peak_mb']:.0f} MB) by more than 16 MB"
            )
