"""Harness benchmark for single-output ground-truth comparisons.

The benchmark compares MojoGP and GPyTorch against the known noiseless test
function from GP-prior data, not just against each other.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.workflow_runner import run_comparison_benchmark_subprocess
from .conftest import assert_gpu_available, assert_gpu_was_used, requires_cuda, requires_gpytorch, requires_mojogp
from tests.shared.benchmarking.report import format_comparison_table


MINIMAL_CONFIGS = [("rbf", 2000, 5), ("matern52", 2000, 5)]


@requires_mojogp
@requires_gpytorch
@requires_cuda
class TestSingleOutputGroundTruthActive:
    @pytest.mark.minimal
    @pytest.mark.single_output
    @pytest.mark.accuracy
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n,d", MINIMAL_CONFIGS)
    def test_ground_truth_exactgp_comparison(self, kernel: str, n: int, d: int, results_dir):
        assert_gpu_available()
        comparison = run_comparison_benchmark_subprocess(
            module="tests.system_benchmarks.run_single_output_truth_case",
            payload={
                "benchmark": "single_output_ground_truth_active",
                "kernel": kernel,
                "n": n,
                "d": d,
            },
            suite_name="single_output_truth_harness",
            benchmark_name="single_output_ground_truth_active",
            framework="cross_framework",
            case_id=f"single_output.truth.{kernel}.n{n}.d{d}",
            benchmark_group_id=f"single_output.truth.{kernel}",
            config={
                "benchmark": "single_output_ground_truth_active",
                "framework": "cross_framework",
                "model_type": "SingleOutputGP",
                "kernel": kernel,
                "n": n,
                "d": d,
                "baseline_backend": "gpytorch_cg",
                "comparison_class": "fair_match",
            },
            results_dir=results_dir,
        )
        print(format_comparison_table(comparison))
        mojogp_mat = comparison.mojogp_materialized
        mojogp_mf = comparison.mojogp_matrix_free
        gpytorch_cg = comparison.gpytorch_cg
        assert mojogp_mat is not None
        assert mojogp_mf is not None
        assert gpytorch_cg is not None
        assert_gpu_was_used(mojogp_mat)
        assert_gpu_was_used(mojogp_mf)
        assert_gpu_was_used(gpytorch_cg)
        assert mojogp_mat.accuracy.rmse < 2.5
        assert mojogp_mf.accuracy.rmse < 2.5
        assert gpytorch_cg.accuracy.rmse < 2.5
        assert comparison.rmse_ratio_vs_cg is None or comparison.rmse_ratio_vs_cg < 2.0
        assert mojogp_mat.hyperparameters.lengthscale_rel_error is not None
        assert mojogp_mf.hyperparameters.lengthscale_rel_error is not None
        assert gpytorch_cg.hyperparameters.lengthscale_rel_error is not None
