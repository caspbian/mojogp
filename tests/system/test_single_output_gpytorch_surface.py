"""Benchmark A: Single-output GP comparison vs GPyTorch.

Compares MojoGP against GPyTorch on identical data with identical initial conditions.
Both frameworks get the same training/test data, same initial hyperparameters,
same optimizer settings, and same CG settings.
"""

import pytest
import numpy as np
from typing import Dict, Any, Optional

from tests.shared.benchmarking.environment import (
    requires_mojogp,
    requires_gpytorch,
    requires_cuda,
    assert_gpu_available,
    assert_gpu_was_used,
)
from tests.shared.benchmarking.data_generators import generate_gp_prior_data, SyntheticDataset
from tests.shared.benchmarking.mojogp_runners import (
    run_mojogp_benchmark,
    train_mojogp_simple,
    predict_mojogp_simple,
)
from tests.shared.benchmarking.gpytorch_models import (
    run_gpytorch_benchmark,
    is_keops_available,
    keops_supported_kernels,
    train_gpytorch_single_output,
    predict_gpytorch_model,
)
from tests.shared.benchmarking.result_types import BenchmarkResult, ComparisonResult
from tests.shared.benchmarking.report import print_comparison, format_comparison_table, save_system_comparison
from tests.shared.benchmarking.metrics import compute_all_accuracy_metrics


# =============================================================================
# Test Configurations
# =============================================================================

# Format: (kernel, n, d)

MINIMAL_CONFIGS = [
    ("rbf", 300, 5),
    ("matern52", 300, 5),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("matern32", 300, 5),
    ("matern12", 300, 5),
    ("periodic", 300, 3),
    ("rq", 300, 5),
    ("rbf", 500, 5),
    ("rbf", 300, 10),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    # ============================================================
    # ALL 8 KERNELS at standard size
    # ============================================================
    ("rbf", 500, 5),
    ("matern12", 500, 5),
    ("matern32", 500, 5),
    ("matern52", 500, 5),
    ("periodic", 500, 3),
    ("rq", 500, 5),
    ("linear", 500, 5),
    ("polynomial", 500, 5),
    # ============================================================
    # SIZE SCALING for key kernels
    # ============================================================
    ("rbf", 1000, 5),
    ("rbf", 2000, 5),
    ("matern52", 1000, 5),
    ("matern52", 2000, 5),
    ("matern32", 1000, 5),
    ("matern12", 1000, 5),
    # ============================================================
    # DIMENSION SCALING
    # ============================================================
    ("rbf", 500, 10),
    ("rbf", 500, 20),
    ("matern52", 500, 10),
    ("matern52", 500, 20),
    # ============================================================
    # ALL 8 KERNELS at larger size (n = 2000)
    # ============================================================
    ("periodic", 1000, 3),
    ("rq", 1000, 5),
    ("linear", 1000, 5),
    ("polynomial", 1000, 5),
]


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _single_output_comparison_metadata(
    *,
    kernel: str,
    n: int,
    d: int,
    n_test: int,
    n_iterations: int,
    lr: float,
    seed: int,
) -> dict[str, object]:
    return {
        "kernel": kernel,
        "n": n,
        "d": d,
        "n_test": n_test,
        "n_iterations": n_iterations,
        "lr": lr,
        "seed": seed,
        "comparison_class": "fair_match",
        "fairness_note": (
            "ExactGP parity row: MojoGP materialized/matrix_free are compared against GPyTorch ExactGP CG "
            "on the same GP-prior dataset with matched optimizer budget and CG settings."
        ),
        "fairness_axes": {
            "comparator_scope": _fairness_axis(
                "fair_match",
                "The primary comparator is GPyTorch ExactGP with CG enabled.",
            ),
            "sample_count_n": _fairness_axis(
                "aligned",
                "All frameworks use the same train/test split and synthetic truth.",
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "All frameworks use Adam with the same learning rate and iteration budget.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "The primary comparison uses CG with matched tolerance, iteration cap, and trace-sample budget.",
            ),
            "preconditioner": _fairness_axis(
                "narrow_mismatch",
                "Framework-specific preconditioner implementations may differ, but both routes run as ExactGP CG baselines.",
            ),
            "prediction_mode": _fairness_axis(
                "aligned",
                "Saved rows compare exact predictive variance by default.",
            ),
            "telemetry": _fairness_axis(
                "observed",
                "MojoGP and GPyTorch telemetry are both recorded for the primary comparison rows.",
            ),
        },
    }


# =============================================================================
# Test Class
# =============================================================================


@requires_mojogp
@requires_gpytorch
class TestSingleOutputGPyTorch:
    """Benchmark A: MojoGP single-output vs GPyTorch comparison."""

    def _run_comparison(
        self,
        kernel: str,
        n: int,
        d: int,
        n_test: int = 100,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
        run_keops: bool = True,
    ) -> ComparisonResult:
        """Run comparison between MojoGP and GPyTorch.

        Both frameworks get:
        - Same training/test data
        - Same initial hyperparameters
        - Same optimizer settings (Adam, same lr)
        - Same CG settings (tolerance, max iterations, probe count)
        """
        # Generate data
        dataset = generate_gp_prior_data(
            n_train=n,
            n_test=n_test,
            d=d,
            kernel_type=kernel,
            true_lengthscale=1.0,
            true_noise=0.1,
            true_outputscale=1.0,
            seed=seed,
        )

        # Common settings
        init_ls = 1.0
        init_noise = 0.1
        init_os = 1.0

        # Run MojoGP materialized
        print(f"\n  Running MojoGP (materialized)...")
        mojogp_mat = run_mojogp_benchmark(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            f_test=dataset.f_test,
            kernel_type=kernel,
            method="materialized",
            n_iterations=n_iterations,
            lr=lr,
            init_ls=init_ls,
            init_noise=init_noise,
            init_os=init_os,
            true_params=dataset.true_params,
        )

        # Run MojoGP matrix-free
        print(f"  Running MojoGP (matrix-free)...")
        mojogp_mf = run_mojogp_benchmark(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            f_test=dataset.f_test,
            kernel_type=kernel,
            method="matrix_free",
            n_iterations=n_iterations,
            lr=lr,
            init_ls=init_ls,
            init_noise=init_noise,
            init_os=init_os,
            true_params=dataset.true_params,
        )

        # Run GPyTorch CG
        print(f"  Running GPyTorch (CG)...")
        gpytorch_cg = run_gpytorch_benchmark(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            f_test=dataset.f_test,
            kernel_type=kernel,
            mode="cg",
            n_iterations=n_iterations,
            lr=lr,
            init_ls=init_ls,
            init_noise=init_noise,
            init_os=init_os,
            true_params=dataset.true_params,
        )

        # Run GPyTorch KeOps (if available and supported)
        gpytorch_keops = None
        if run_keops and is_keops_available() and kernel in keops_supported_kernels():
            print(f"  Running GPyTorch (KeOps)...")
            try:
                gpytorch_keops = run_gpytorch_benchmark(
                    X_train=dataset.X_train,
                    y_train=dataset.y_train,
                    X_test=dataset.X_test,
                    f_test=dataset.f_test,
                    kernel_type=kernel,
                    mode="keops",
                    n_iterations=n_iterations,
                    lr=lr,
                    init_ls=init_ls,
                    init_noise=init_noise,
                    init_os=init_os,
                    true_params=dataset.true_params,
                )
            except Exception as e:
                print(f"  KeOps failed: {e}")

        # Build comparison result
        comparison = ComparisonResult(
            config=_single_output_comparison_metadata(
                kernel=kernel,
                n=n,
                d=d,
                n_test=n_test,
                n_iterations=n_iterations,
                lr=lr,
                seed=seed,
            ),
            mojogp_materialized=mojogp_mat,
            mojogp_matrix_free=mojogp_mf,
            gpytorch_cg=gpytorch_cg,
            gpytorch_keops=gpytorch_keops,
        )

        # Compute derived comparisons
        comparison.compute_comparisons()

        return comparison

    def _report_comparison(self, comparison: ComparisonResult, results_dir=None):
        """Print and optionally save the comparison."""
        print(format_comparison_table(comparison))
        if results_dir:
            save_system_comparison(comparison, results_dir, "single_output_gpytorch")

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.single_output
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n,d", MINIMAL_CONFIGS)
    def test_single_output_gpytorch_parity_core_configs(
        self,
        kernel: str,
        n: int,
        d: int,
        results_dir,
        n_override,
    ):
        """Minimal comparison test - quick validation."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        if n_override is not None:
            n = n_override

        print(f"\n=== Comparison: {kernel}, n={n}, d={d} ===")
        comparison = self._run_comparison(kernel, n, d, run_keops=False)
        self._report_comparison(comparison, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(comparison.mojogp_materialized)
        assert_gpu_was_used(comparison.gpytorch_cg)

        # Soft assertions: verify both frameworks produce valid results
        assert comparison.mojogp_materialized is not None
        assert comparison.gpytorch_cg is not None
        assert comparison.mojogp_materialized.accuracy.rmse < np.inf
        assert comparison.gpytorch_cg.accuracy.rmse < np.inf

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.single_output
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n,d", MODERATE_CONFIGS)
    def test_single_output_gpytorch_parity_extended_configs(
        self,
        kernel: str,
        n: int,
        d: int,
        results_dir,
        n_override,
    ):
        """Moderate comparison test - broader coverage."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        if n_override is not None:
            n = n_override

        print(f"\n=== Comparison: {kernel}, n={n}, d={d} ===")
        comparison = self._run_comparison(kernel, n, d, run_keops=True)
        self._report_comparison(comparison, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(comparison.mojogp_materialized)
        assert_gpu_was_used(comparison.gpytorch_cg)

        # Soft assertions
        assert comparison.mojogp_materialized is not None
        assert comparison.gpytorch_cg is not None

        # MojoGP should produce accuracy comparable to GPyTorch
        # Target: RMSE within 1.2x of GPyTorch CG
        if comparison.rmse_ratio_vs_cg is not None:
            assert comparison.rmse_ratio_vs_cg < 1.2, (
                f"MojoGP RMSE should be within 1.2x of GPyTorch, got {comparison.rmse_ratio_vs_cg:.2f}x"
            )

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.single_output
    @pytest.mark.gpytorch
    @pytest.mark.parametrize("kernel,n,d", FULL_CONFIGS)
    def test_single_output_gpytorch_parity_broad_configs(
        self,
        kernel: str,
        n: int,
        d: int,
        results_dir,
    ):
        """Full comparison test - exhaustive coverage."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        print(f"\n=== Comparison: {kernel}, n={n}, d={d} ===")
        comparison = self._run_comparison(kernel, n, d, run_keops=True)
        self._report_comparison(comparison, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(comparison.mojogp_materialized)
        assert_gpu_was_used(comparison.gpytorch_cg)

        # Soft assertions
        assert comparison.mojogp_materialized is not None
        assert comparison.gpytorch_cg is not None

    # =========================================================================
    # Specific Comparison Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.single_output
    @pytest.mark.gpytorch
    def test_rbf_detailed_comparison(self, results_dir):
        """Detailed RBF comparison with analysis."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        print("\n=== Detailed RBF Comparison ===")
        comparison = self._run_comparison("rbf", 300, 5, run_keops=True)
        self._report_comparison(comparison, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(comparison.mojogp_materialized)
        assert_gpu_was_used(comparison.gpytorch_cg)

        # Print detailed analysis
        print("\n--- Analysis ---")
        if comparison.rmse_ratio_vs_cg:
            if comparison.rmse_ratio_vs_cg < 1.0:
                print(
                    f"MojoGP has {(1 - comparison.rmse_ratio_vs_cg) * 100:.1f}% better RMSE"
                )
            else:
                print(
                    f"GPyTorch has {(comparison.rmse_ratio_vs_cg - 1) * 100:.1f}% better RMSE"
                )

        if comparison.speedup_vs_cg:
            if comparison.speedup_vs_cg > 1.0:
                print(f"MojoGP is {comparison.speedup_vs_cg:.2f}x faster")
            else:
                print(f"GPyTorch is {1 / comparison.speedup_vs_cg:.2f}x faster")

        if comparison.memory_ratio_vs_cg:
            if comparison.memory_ratio_vs_cg < 1.0:
                print(
                    f"MojoGP uses {(1 - comparison.memory_ratio_vs_cg) * 100:.1f}% less memory"
                )
            else:
                print(
                    f"GPyTorch uses {(1 - 1 / comparison.memory_ratio_vs_cg) * 100:.1f}% less memory"
                )

    @pytest.mark.moderate
    @pytest.mark.single_output
    @pytest.mark.gpytorch
    def test_scaling_comparison(self, results_dir):
        """Test how MojoGP vs GPyTorch scales with n."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        print("\n=== Scaling Comparison ===")
        sizes = [200, 500, 1000]
        comparisons = []

        for n in sizes:
            print(f"\n--- n={n} ---")
            comparison = self._run_comparison("rbf", n, 5, run_keops=False)
            comparisons.append(comparison)

        # CRITICAL: Verify GPU was actually used
        for comp in comparisons:
            assert_gpu_was_used(comp.mojogp_materialized)
            assert_gpu_was_used(comp.gpytorch_cg)

        # Print scaling summary
        print("\n=== Scaling Summary ===")
        print(
            f"{'n':>6} | {'MojoGP Time':>12} | {'GPyTorch Time':>13} | {'Speedup':>8} | {'RMSE Ratio':>10}"
        )
        print("-" * 60)
        for comp in comparisons:
            n = comp.config["n"]
            mojo_time = comp.mojogp_materialized.speed.training_time_s
            gpy_time = comp.gpytorch_cg.speed.training_time_s
            speedup = comp.speedup_vs_cg or 0
            rmse_ratio = comp.rmse_ratio_vs_cg or 0
            print(
                f"{n:>6} | {mojo_time:>12.2f}s | {gpy_time:>13.2f}s | {speedup:>8.2f}x | {rmse_ratio:>10.2f}"
            )

    @pytest.mark.full
    @pytest.mark.single_output
    @pytest.mark.gpytorch
    def test_all_kernels_comparison(self, results_dir):
        """Compare all kernel types against GPyTorch."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        print("\n=== All Kernels Comparison ===")
        kernels = [
            "rbf",
            "matern12",
            "matern32",
            "matern52",
            "periodic",
            "rq",
            "linear",
            "polynomial",
        ]
        comparisons = {}

        for kernel in kernels:
            d = 3 if kernel == "periodic" else 5
            print(f"\n--- {kernel} ---")
            try:
                comparison = self._run_comparison(kernel, 300, d, run_keops=False)
                comparisons[kernel] = comparison
            except Exception as e:
                print(f"  Failed: {e}")

        # CRITICAL: Verify GPU was actually used
        for comp in comparisons.values():
            assert_gpu_was_used(comp.mojogp_materialized)
            assert_gpu_was_used(comp.gpytorch_cg)

        # Print summary
        print("\n=== All Kernels Summary ===")
        print(
            f"{'Kernel':>12} | {'MojoGP RMSE':>11} | {'GPyTorch RMSE':>13} | {'Ratio':>6} | {'Speedup':>8}"
        )
        print("-" * 65)
        for kernel, comp in comparisons.items():
            mojo_rmse = comp.mojogp_materialized.accuracy.rmse
            gpy_rmse = comp.gpytorch_cg.accuracy.rmse
            ratio = comp.rmse_ratio_vs_cg or 0
            speedup = comp.speedup_vs_cg or 0
            print(
                f"{kernel:>12} | {mojo_rmse:>11.4f} | {gpy_rmse:>13.4f} | {ratio:>6.2f} | {speedup:>8.2f}x"
            )


# =============================================================================
# Variance Method Comparison Tests
# =============================================================================


@requires_mojogp
@requires_gpytorch
class TestVarianceMethods:
    """Compare exact vs LOVE variance for both MojoGP and GPyTorch."""

    @pytest.mark.minimal
    @pytest.mark.accuracy
    def test_variance_comparison_rbf(self, n_override):
        """Compare variance methods: MojoGP exact/LOVE vs GPyTorch exact/LOVE."""
        import torch
        import gpytorch

        assert_gpu_available()

        n = 2000 if n_override is None else n_override
        d = 5
        kernel = "rbf"

        print(f"\n=== Variance Method Comparison: {kernel}, n={n}, d={d} ===")

        # Generate data
        dataset = generate_gp_prior_data(
            n_train=n,
            n_test=100,
            d=d,
            kernel_type=kernel,
            true_lengthscale=1.0,
            true_noise=0.1,
            true_outputscale=1.0,
            seed=42,
        )

        # Train MojoGP once
        train_result = train_mojogp_simple(
            dataset.X_train,
            dataset.y_train,
            kernel,
            "materialized",
            n_iterations=100,
            lr=0.05,
        )
        learned = train_result["learned_params"]

        # Train GPyTorch once
        gpy_result = train_gpytorch_single_output(
            dataset.X_train,
            dataset.y_train,
            kernel,
            mode="cg",
            n_iterations=100,
            lr=0.05,
        )
        gpy_model = gpy_result["model"]
        gpy_likelihood = gpy_result["likelihood"]
        test_x = torch.tensor(dataset.X_test, dtype=torch.float32).cuda()

        # Predict with 4 configurations
        configs = {}

        # MojoGP Exact
        pred_exact = predict_mojogp_simple(
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            learned,
            kernel,
            "materialized",
            variance_method="exact",
        )
        configs["MojoGP Exact"] = pred_exact

        # MojoGP LOVE
        pred_love = predict_mojogp_simple(
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            learned,
            kernel,
            "materialized",
            variance_method="love",
        )
        configs["MojoGP LOVE"] = pred_love

        # GPyTorch Exact (default, no fast_pred_var)
        pred_gpy_exact = predict_gpytorch_model(
            gpy_model,
            gpy_likelihood,
            test_x,
            mode="cg",
            use_love=False,
        )
        configs["GPyTorch Exact"] = pred_gpy_exact

        # GPyTorch LOVE (fast_pred_var=True)
        pred_gpy_love = predict_gpytorch_model(
            gpy_model,
            gpy_likelihood,
            test_x,
            mode="cg",
            use_love=True,
        )
        configs["GPyTorch LOVE"] = pred_gpy_love

        # Compute metrics for each
        print(
            f"\n{'Metric':<20} | {'MojoGP Exact':>12} | {'MojoGP LOVE':>12} | {'GPyTorch Exact':>14} | {'GPyTorch LOVE':>13}"
        )
        print("-" * 85)

        results = {}
        for name, pred in configs.items():
            metrics = compute_all_accuracy_metrics(
                dataset.f_test,
                pred["mean"],
                pred["std"],
                y_train_mean=float(np.mean(dataset.y_train)),
                y_train_std=float(np.std(dataset.y_train)),
            )
            results[name] = metrics

        # Print comparison table
        for metric_name in [
            "rmse",
            "crps",
            "calibration_error",
            "sharpness",
            "calibration_95",
        ]:
            vals = []
            for name in [
                "MojoGP Exact",
                "MojoGP LOVE",
                "GPyTorch Exact",
                "GPyTorch LOVE",
            ]:
                v = results[name].get(metric_name, 0.0)
                vals.append(f"{v:.4f}" if abs(v) < 1e6 else f"{v:.2e}")
            print(
                f"  {metric_name:<18} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>14} | {vals[3]:>13}"
            )

        # Print variance statistics
        print(
            f"\n{'Variance Stats':<20} | {'MojoGP Exact':>12} | {'MojoGP LOVE':>12} | {'GPyTorch Exact':>14} | {'GPyTorch LOVE':>13}"
        )
        print("-" * 85)
        for stat_name, stat_fn in [
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max),
        ]:
            vals = []
            for name in [
                "MojoGP Exact",
                "MojoGP LOVE",
                "GPyTorch Exact",
                "GPyTorch LOVE",
            ]:
                v = stat_fn(configs[name]["variance"])
                vals.append(f"{v:.6f}" if abs(v) < 1e6 else f"{v:.2e}")
            print(
                f"  {stat_name:<18} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>14} | {vals[3]:>13}"
            )

        # Assertions: MojoGP exact should be close to GPyTorch exact
        mojo_exact_rmse = results["MojoGP Exact"]["rmse"]
        gpy_exact_rmse = results["GPyTorch Exact"]["rmse"]
        rmse_ratio = mojo_exact_rmse / max(gpy_exact_rmse, 1e-10)
        print(f"\nRMSE ratio (MojoGP Exact / GPyTorch Exact): {rmse_ratio:.4f}")

        # Exact variance should produce similar calibration
        mojo_exact_cal = results["MojoGP Exact"]["calibration_95"]
        gpy_exact_cal = results["GPyTorch Exact"]["calibration_95"]
        print(
            f"Calibration 95% - MojoGP Exact: {mojo_exact_cal:.2f}, GPyTorch Exact: {gpy_exact_cal:.2f}"
        )

        assert rmse_ratio < 1.5, (
            f"MojoGP exact RMSE should be within 1.5x of GPyTorch, got {rmse_ratio:.2f}x"
        )
        assert mojo_exact_cal > 0.80, (
            f"MojoGP exact calibration should be > 0.80, got {mojo_exact_cal:.2f}"
        )


# =============================================================================
# Standalone Execution
# =============================================================================


if __name__ == "__main__":
    # Run minimal tests when executed directly
    pytest.main([__file__, "-v", "-m", "minimal"])
