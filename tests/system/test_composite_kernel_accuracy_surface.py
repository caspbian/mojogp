"""Benchmark: Composite kernel accuracy vs synthetic ground truth.

Tests MojoGP's composite kernel API (kernel combinations like RBF + Matern52)
against synthetic data with known ground truth.
"""

import pytest
import numpy as np
from typing import Dict, Any

from tests.shared.benchmarking.environment import requires_mojogp, assert_gpu_available, assert_gpu_was_used
from tests.shared.benchmarking.data_generators import generate_gp_prior_data, generate_structured_function_data
from tests.shared.benchmarking.mojogp_runners import train_mojogp_composite, predict_mojogp_composite
from tests.shared.benchmarking.result_types import (
    BenchmarkResult,
    AccuracyResult,
    SpeedResult,
    MemoryResult,
    HyperparameterResult,
)
from tests.shared.benchmarking.metrics import compute_all_accuracy_metrics
from tests.shared.benchmarking.report import print_result


def _has_mojogp_kernel():
    """Check if mojogp.kernel module is available."""
    try:
        from mojogp.kernel import Kernel

        return True
    except ImportError:
        return False


requires_composite_api = pytest.mark.skipif(
    not _has_mojogp_kernel(), reason="mojogp.kernel module not available"
)


# =============================================================================
# Test Configurations
# =============================================================================

# Format: (kernel_expr, name, data_type, n_train, d)
# kernel_expr is a string that will be evaluated to create the kernel

MINIMAL_CONFIGS = [
    ("Kernel.rbf() + Kernel.matern52()", "rbf_plus_matern52", "gp_prior", 200, 5),
    (
        "Kernel.rbf() * Kernel.periodic()",
        "rbf_times_periodic",
        "periodic_signal",
        200,
        3,
    ),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("Kernel.rbf() + Kernel.linear()", "rbf_plus_linear", "smooth", 300, 5),
    (
        "Kernel.matern32() + Kernel.matern52()",
        "matern32_plus_matern52",
        "gp_prior",
        300,
        5,
    ),
    ("Kernel.rbf() * Kernel.rbf()", "rbf_times_rbf", "gp_prior", 300, 5),
    (
        "Kernel.periodic() + Kernel.linear()",
        "periodic_plus_linear",
        "periodic_signal",
        300,
        3,
    ),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    # More complex compositions
    (
        "Kernel.rbf() + Kernel.matern52() + Kernel.linear()",
        "triple_sum",
        "smooth",
        300,
        5,
    ),
    (
        "Kernel.rbf() * Kernel.periodic() + Kernel.linear()",
        "product_plus_linear",
        "periodic_signal",
        300,
        3,
    ),
    # Scaling tests
    ("Kernel.rbf() + Kernel.matern52()", "rbf_plus_matern52_large", "gp_prior", 500, 5),
    (
        "Kernel.rbf() + Kernel.matern52()",
        "rbf_plus_matern52_highdim",
        "gp_prior",
        300,
        10,
    ),
]


# =============================================================================
# Test Class
# =============================================================================


@requires_mojogp
@requires_composite_api
class TestCompositeKernelAccuracy:
    """Benchmark: Composite kernel accuracy vs synthetic ground truth."""

    def _create_kernel(self, kernel_expr: str):
        """Create a kernel from a string expression."""
        from mojogp.kernel import Kernel

        return eval(kernel_expr)

    def _run_composite_test(
        self,
        kernel_expr: str,
        data_type: str,
        n_train: int,
        d: int,
        n_test: int = 50,
        n_iterations: int = 100,
        lr: float = 0.05,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run a composite kernel benchmark."""
        # Generate data
        if data_type == "gp_prior":
            dataset = generate_gp_prior_data(
                n_train,
                n_test,
                d,
                "rbf",  # Use RBF for GP prior
                true_lengthscale=1.0,
                true_noise=0.1,
                true_outputscale=1.0,
                seed=seed,
            )
        elif data_type == "periodic_signal":
            dataset = generate_structured_function_data(
                n_train,
                n_test,
                d,
                "periodic_signal",
                noise_level="medium",
                seed=seed,
            )
        else:
            dataset = generate_structured_function_data(
                n_train,
                n_test,
                d,
                data_type,
                noise_level="medium",
                seed=seed,
            )

        # Create kernel
        kernel = self._create_kernel(kernel_expr)

        # Train
        train_result = train_mojogp_composite(
            dataset.X_train,
            dataset.y_train,
            kernel,
            method="auto",
            n_iterations=n_iterations,
            lr=lr,
            init_noise=0.1,
            monitor_memory=True,
        )

        # Predict
        pred_result = predict_mojogp_composite(
            train_result["gp"],
            dataset.X_test,
        )

        # Compute accuracy metrics
        accuracy_metrics = compute_all_accuracy_metrics(
            dataset.f_test,
            pred_result["mean"],
            pred_result["std"],
            y_train_mean=float(np.mean(dataset.y_train)),
            y_train_std=float(np.std(dataset.y_train)),
        )

        # Build result
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
            end_to_end_time_s=train_result["training_time_s"]
            + pred_result["total_time_s"],
            iterations_run=train_result["iterations_run"],
            max_iterations=train_result["max_iterations"],
            early_stopped=train_result["early_stopped"],
            ms_per_iteration=(
                train_result["training_time_s"] / max(train_result["iterations_run"], 1)
            )
            * 1000,
        )

        memory_stats = train_result.get("memory_stats", {})
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
        )

        learned = train_result["learned_params"]
        hyperparameters = HyperparameterResult(
            learned_lengthscale=learned.get("lengthscale", 1.0),
            learned_noise=learned.get("noise", 0.1),
            learned_outputscale=learned.get("outputscale", 1.0),
            final_nll=train_result["final_nll"],
        )

        config = {
            "kernel_expr": kernel_expr,
            "data_type": data_type,
            "n": n_train,
            "d": d,
            "n_iterations": n_iterations,
            "lr": lr,
            "seed": seed,
        }

        return BenchmarkResult(
            config=config,
            accuracy=accuracy,
            speed=speed,
            memory=memory,
            hyperparameters=hyperparameters,
        )

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.composite
    @pytest.mark.accuracy
    @pytest.mark.parametrize("kernel_expr,name,data_type,n,d", MINIMAL_CONFIGS)
    def test_composite_kernel_accuracy_core_configs(
        self,
        kernel_expr: str,
        name: str,
        data_type: str,
        n: int,
        d: int,
        results_dir,
    ):
        """Minimal composite kernel test."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_composite_test(kernel_expr, data_type, n, d)
        print_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions
        assert result.accuracy.rmse < np.inf, "RMSE should be finite"
        assert result.accuracy.crps < np.inf, "CRPS should be finite"
        assert result.speed.training_time_s > 0, "Training time should be positive"

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.composite
    @pytest.mark.accuracy
    @pytest.mark.parametrize("kernel_expr,name,data_type,n,d", MODERATE_CONFIGS)
    def test_composite_kernel_accuracy_extended_configs(
        self,
        kernel_expr: str,
        name: str,
        data_type: str,
        n: int,
        d: int,
        results_dir,
    ):
        """Moderate composite kernel test."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_composite_test(kernel_expr, data_type, n, d)
        print_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions
        assert result.accuracy.rmse < np.inf, "RMSE should be finite"
        assert result.accuracy.crps < np.inf, "CRPS should be finite"

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.composite
    @pytest.mark.accuracy
    @pytest.mark.parametrize("kernel_expr,name,data_type,n,d", FULL_CONFIGS)
    def test_composite_kernel_accuracy_broad_configs(
        self,
        kernel_expr: str,
        name: str,
        data_type: str,
        n: int,
        d: int,
        results_dir,
    ):
        """Full composite kernel test."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_composite_test(kernel_expr, data_type, n, d)
        print_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions
        assert result.accuracy.rmse < np.inf, "RMSE should be finite"
        assert result.accuracy.crps < np.inf, "CRPS should be finite"

    # =========================================================================
    # Specific Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.composite
    def test_sum_kernel_matches_gp_prior_signal(self, results_dir):
        """Test sum kernel (RBF + Matern52) on GP-prior data."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_composite_test(
            "Kernel.rbf() + Kernel.matern52()",
            "gp_prior",
            n_train=2000,
            d=5,
        )
        print_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        assert result.accuracy.rmse < 2.0, "RMSE should be reasonable"

    @pytest.mark.minimal
    @pytest.mark.composite
    def test_product_kernel_matches_periodic_signal(self, results_dir):
        """Test product kernel (RBF * Periodic) on periodic-signal data."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_composite_test(
            "Kernel.rbf() * Kernel.periodic()",
            "periodic_signal",
            n_train=2000,
            d=3,
        )
        print_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        assert result.accuracy.rmse < 2.0, "RMSE should be reasonable"


# =============================================================================
# Standalone Execution
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "minimal"])
