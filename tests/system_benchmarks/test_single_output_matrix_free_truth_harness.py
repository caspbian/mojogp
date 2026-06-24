"""Harness benchmark for matrix-free ground-truth comparisons.

The benchmark compares matrix-free and materialized MojoGP routes against known
synthetic truth, with GPyTorch ExactGP CG retained as the fair external baseline.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.workflow_runner import run_comparison_benchmark_subprocess
from .conftest import (
    assert_gpu_available,
    assert_gpu_was_used,
    requires_cuda,
    requires_mojogp,
)
from tests.shared.benchmarking.report import format_comparison_table


MINIMAL_CONFIGS = [("rbf", 2000, 5), ("matern52", 2000, 5)]


@requires_mojogp
@requires_cuda
class TestMatrixFreeGroundTruthActive:
    @pytest.mark.minimal
    @pytest.mark.single_output
    @pytest.mark.accuracy
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n,d", MINIMAL_CONFIGS)
    def test_matrix_free_ground_truth_comparison(
        self, kernel: str, n: int, d: int, results_dir
    ):
        assert_gpu_available()
        comparison = run_comparison_benchmark_subprocess(
            module="tests.system_benchmarks.run_single_output_truth_case",
            payload={
                "benchmark": "matrix_free_ground_truth_active",
                "kernel": kernel,
                "n": n,
                "d": d,
            },
            suite_name="single_output_truth_harness",
            benchmark_name="matrix_free_ground_truth_active",
            framework="cross_framework",
            case_id=f"single_output.matrix_free_truth.{kernel}.n{n}.d{d}",
            benchmark_group_id=f"single_output.matrix_free_truth.{kernel}",
            config={
                "benchmark": "matrix_free_ground_truth_active",
                "framework": "mojogp",
                "model_type": "SingleOutputGP",
                "kernel": kernel,
                "n": n,
                "d": d,
                "baseline_backend": "none",
                "comparison_class": "intra_mojogp",
            },
            results_dir=results_dir,
        )
        print(format_comparison_table(comparison))
        mojogp_mat = comparison.mojogp_materialized
        mojogp_mf = comparison.mojogp_matrix_free
        assert mojogp_mat is not None
        assert mojogp_mf is not None
        assert comparison.gpytorch_cg is None
        rmse_gap = abs(mojogp_mf.accuracy.rmse - mojogp_mat.accuracy.rmse)
        mat_rmse_scale = max(mojogp_mat.accuracy.rmse, 1e-6)
        memory_ratio_mat_over_mf = comparison.config.get("materialized_over_matrix_free_gpu_peak_ratio")
        assert_gpu_was_used(mojogp_mat)
        assert_gpu_was_used(mojogp_mf)

        assert mojogp_mat.accuracy.rmse < 2.5
        assert mojogp_mf.accuracy.rmse < 2.5
        assert (rmse_gap / mat_rmse_scale) < 0.35

        assert mojogp_mf.hyperparameters.lengthscale_rel_error is not None
        assert mojogp_mat.hyperparameters.lengthscale_rel_error is not None

        if memory_ratio_mat_over_mf is not None:
            assert memory_ratio_mat_over_mf >= 0.95
