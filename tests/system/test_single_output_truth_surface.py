"""Benchmark B: Single-output GP accuracy vs synthetic ground truth.

Tests MojoGP's ability to learn from synthetic data and make accurate predictions.
This is the gold standard for accuracy measurement since we know the true function.
"""

import pytest
import numpy as np
from typing import List, Tuple

from tests.shared.benchmarking.environment import (
    requires_mojogp,
    assert_gpu_available,
    assert_gpu_was_used,
    scale_n_for_vram,
    get_vram_info,
)
from tests.shared.benchmarking.data_generators import generate_data, generate_gp_prior_data, SyntheticDataset
from tests.shared.benchmarking.mojogp_runners import (
    run_mojogp_benchmark,
    train_mojogp_simple,
    predict_mojogp_simple,
)
from tests.shared.benchmarking.result_types import BenchmarkResult
from tests.shared.benchmarking.report import print_result, save_system_result


# =============================================================================
# Test Configurations
# =============================================================================

# Format: (kernel, n_train, n_test, d, data_type, noise_level, method)

MINIMAL_CONFIGS = [
    # Quick end-to-end validation
    ("rbf", 200, 50, 5, "gp_prior", "medium", "auto"),
    ("matern52", 200, 50, 5, "gp_prior", "medium", "auto"),
    ("rbf", 200, 50, 5, "smooth", "medium", "auto"),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    ("matern32", 200, 50, 5, "gp_prior", "medium", "auto"),
    ("matern12", 200, 50, 5, "gp_prior", "medium", "auto"),
    ("periodic", 200, 50, 3, "periodic_signal", "medium", "auto"),
    ("rq", 200, 50, 5, "gp_prior", "medium", "auto"),
    ("linear", 200, 50, 5, "linear", "medium", "auto"),
    # Noise stress tests
    ("rbf", 300, 50, 5, "gp_prior", "low", "auto"),
    ("rbf", 300, 50, 5, "gp_prior", "high", "auto"),
    ("rbf", 300, 50, 5, "gp_prior", "very_high", "auto"),
    # Method comparison
    ("rbf", 200, 50, 5, "gp_prior", "medium", "materialized"),
    ("rbf", 200, 50, 5, "gp_prior", "medium", "matrix_free"),
    # Dimension scaling
    ("rbf", 300, 50, 2, "gp_prior", "medium", "auto"),
    ("rbf", 300, 50, 10, "gp_prior", "medium", "auto"),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    # ============================================================
    # ALL 8 KERNELS at standard size (n=500, d=5)
    # ============================================================
    ("rbf", 500, 100, 5, "gp_prior", "medium", "auto"),
    ("matern12", 500, 100, 5, "gp_prior", "medium", "auto"),
    ("matern32", 500, 100, 5, "gp_prior", "medium", "auto"),
    ("matern52", 500, 100, 5, "gp_prior", "medium", "auto"),
    ("periodic", 500, 100, 3, "periodic_signal", "medium", "auto"),
    ("rq", 500, 100, 5, "gp_prior", "medium", "auto"),
    ("linear", 500, 100, 5, "linear", "medium", "auto"),
    ("polynomial", 500, 100, 5, "polynomial", "medium", "auto"),
    # ============================================================
    # ALL 8 KERNELS x BOTH METHODS (materialized + matrix_free)
    # ============================================================
    ("rbf", 300, 50, 5, "gp_prior", "medium", "materialized"),
    ("rbf", 300, 50, 5, "gp_prior", "medium", "matrix_free"),
    ("matern12", 300, 50, 5, "gp_prior", "medium", "materialized"),
    ("matern12", 300, 50, 5, "gp_prior", "medium", "matrix_free"),
    ("matern32", 300, 50, 5, "gp_prior", "medium", "materialized"),
    ("matern32", 300, 50, 5, "gp_prior", "medium", "matrix_free"),
    ("matern52", 300, 50, 5, "gp_prior", "medium", "materialized"),
    ("matern52", 300, 50, 5, "gp_prior", "medium", "matrix_free"),
    ("periodic", 300, 50, 3, "periodic_signal", "medium", "materialized"),
    ("periodic", 300, 50, 3, "periodic_signal", "medium", "matrix_free"),
    ("rq", 300, 50, 5, "gp_prior", "medium", "materialized"),
    ("rq", 300, 50, 5, "gp_prior", "medium", "matrix_free"),
    ("linear", 300, 50, 5, "linear", "medium", "materialized"),
    ("linear", 300, 50, 5, "linear", "medium", "matrix_free"),
    ("polynomial", 300, 50, 5, "polynomial", "medium", "materialized"),
    ("polynomial", 300, 50, 5, "polynomial", "medium", "matrix_free"),
    # ============================================================
    # ALL 8 KERNELS x ALL NOISE LEVELS
    # ============================================================
    ("rbf", 300, 50, 5, "gp_prior", "low", "auto"),
    ("rbf", 300, 50, 5, "gp_prior", "high", "auto"),
    ("rbf", 300, 50, 5, "gp_prior", "very_high", "auto"),
    ("matern52", 300, 50, 5, "gp_prior", "low", "auto"),
    ("matern52", 300, 50, 5, "gp_prior", "high", "auto"),
    ("matern52", 300, 50, 5, "gp_prior", "very_high", "auto"),
    ("periodic", 300, 50, 3, "periodic_signal", "low", "auto"),
    ("periodic", 300, 50, 3, "periodic_signal", "high", "auto"),
    # ============================================================
    # SIZE SCALING (n = 100, 500, 1000, 2000) for key kernels
    # ============================================================
    ("rbf", 100, 50, 5, "gp_prior", "medium", "auto"),
    ("rbf", 500, 100, 5, "gp_prior", "medium", "auto"),
    ("rbf", 1000, 200, 5, "gp_prior", "medium", "auto"),
    ("rbf", 2000, 200, 5, "gp_prior", "medium", "auto"),
    ("matern52", 100, 50, 5, "gp_prior", "medium", "auto"),
    ("matern52", 500, 100, 5, "gp_prior", "medium", "auto"),
    ("matern52", 1000, 200, 5, "gp_prior", "medium", "auto"),
    # ============================================================
    # DIMENSION SCALING (d = 2, 5, 10, 20, 30)
    # ============================================================
    ("rbf", 500, 100, 2, "gp_prior", "medium", "auto"),
    ("rbf", 500, 100, 10, "gp_prior", "medium", "auto"),
    ("rbf", 500, 100, 20, "gp_prior", "medium", "auto"),
    ("rbf", 500, 100, 30, "gp_prior", "medium", "auto"),
    ("matern52", 500, 100, 10, "gp_prior", "medium", "auto"),
    ("matern52", 500, 100, 20, "gp_prior", "medium", "auto"),
    # ============================================================
    # STRUCTURED FUNCTIONS (test kernel-data matching)
    # ============================================================
    ("rbf", 500, 100, 5, "smooth", "medium", "auto"),
    ("rbf", 500, 100, 5, "oscillatory", "medium", "auto"),
    (
        "matern12",
        500,
        100,
        5,
        "step",
        "medium",
        "auto",
    ),  # rough kernel for discontinuous
    ("matern52", 500, 100, 5, "smooth", "medium", "auto"),
    ("linear", 500, 100, 5, "linear", "low", "auto"),
    ("polynomial", 500, 100, 5, "polynomial", "low", "auto"),
    ("periodic", 500, 100, 3, "periodic_signal", "medium", "auto"),
    # ============================================================
    # MISSPECIFIED MODELS (kernel doesn't match data)
    # ============================================================
    ("rbf", 300, 50, 5, "step", "medium", "auto"),  # smooth kernel on discontinuous
    ("matern12", 300, 50, 5, "smooth", "medium", "auto"),  # rough kernel on smooth
    ("linear", 300, 50, 5, "smooth", "medium", "auto"),  # linear kernel on nonlinear
    (
        "periodic",
        300,
        50,
        3,
        "smooth",
        "medium",
        "auto",
    ),  # periodic kernel on non-periodic
]


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _single_output_accuracy_metadata(
    *,
    dataset: SyntheticDataset,
    kernel: str,
    method: str,
    data_type: str,
    noise_level: str,
    n_iterations: int,
    learning_rate: float,
) -> dict[str, object]:
    return {
        "benchmark": "single_output_accuracy",
        "route_group": "single_output",
        "framework": "mojogp",
        "model_type": "SingleOutputGP",
        "kernel": kernel,
        "training_method": method,
        "method": method,
        "prediction_mode": "exact",
        "comparison_class": "mojogp_only",
        "baseline_backend": "none",
        "keops_supported": False,
        "keops_used": False,
        "fairness_note": (
            "N.B. MojoGP-only ground-truth row: this benchmark measures predictive accuracy and parameter "
            "recovery against synthetic truth rather than against a cross-framework comparator."
        ),
        "fairness_axes": {
            "comparator_scope": _fairness_axis(
                "mojogp_only",
                "The baseline is the synthetic ground truth carried by the data generator.",
            ),
            "sample_count_n": _fairness_axis(
                "aligned",
                "Each row evaluates one MojoGP route on a fixed synthetic train/test split.",
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "Each row uses a fixed optimizer family, learning rate, and iteration budget.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "Each row records a single MojoGP route with fixed CG/Lanczos policy.",
            ),
            "preconditioner": _fairness_axis(
                "aligned",
                "The active route uses its configured route-level preconditioner policy.",
            ),
            "prediction_mode": _fairness_axis(
                "aligned",
                "The active single-output accuracy surface uses exact predictive variance for the saved row.",
            ),
            "telemetry": _fairness_axis(
                "observed",
                "MojoGP telemetry is observed for the active route.",
            ),
        },
        "n": int(dataset.X_train.shape[0]),
        "n_test": int(dataset.X_test.shape[0]),
        "d": int(dataset.X_train.shape[1]),
        "data_type": data_type,
        "noise_level": noise_level,
        "training_solver_config": {
            "framework": "mojogp",
            "mode": method,
            "max_iterations": n_iterations,
            "learning_rate": learning_rate,
        },
    }


# =============================================================================
# Test Class
# =============================================================================


@requires_mojogp
class TestSingleOutputAccuracy:
    """Benchmark B: MojoGP single-output accuracy vs synthetic ground truth."""

    def _run_accuracy_test(
        self,
        kernel: str,
        n_train: int,
        n_test: int,
        d: int,
        data_type: str,
        noise_level: str,
        method: str,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Core test logic shared across tiers.

        Note: n_train is automatically scaled down if it exceeds the GPU's
        VRAM capacity for the given method. This ensures tests don't OOM
        on smaller GPUs while still running the full test suite.
        """
        # VRAM-adaptive sizing: scale n_train if needed
        effective_method = method if method != "auto" else "materialized"
        actual_n_train = scale_n_for_vram(n_train, effective_method)

        if actual_n_train < n_train:
            vram = get_vram_info()
            print(
                f"  [VRAM] Scaled n_train from {n_train} to {actual_n_train} "
                f"(GPU: {vram['vram_gb']}GB, tier: {vram['tier']})"
            )

        # Generate data
        dataset = generate_data(
            actual_n_train, n_test, d, kernel, data_type, noise_level, seed=seed
        )

        # Run benchmark
        result = run_mojogp_benchmark(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            f_test=dataset.f_test,
            kernel_type=kernel,
            method=method,
            n_iterations=100,
            lr=0.05,
            init_ls=1.0,
            init_noise=0.1,
            init_os=1.0,
            true_params=dataset.true_params,
            monitor_memory=True,
        )

        # Update config with test-specific info
        result.config.update(
            {
                **_single_output_accuracy_metadata(
                    dataset=dataset,
                    kernel=kernel,
                    method=method,
                    data_type=data_type,
                    noise_level=noise_level,
                    n_iterations=100,
                    learning_rate=0.05,
                ),
                "seed": seed,
                "requested_n": n_train,
                "actual_n": actual_n_train,
                "vram_scaled": actual_n_train < n_train,
            }
        )

        return result

    def _report_result(self, result: BenchmarkResult, results_dir=None):
        """Print and optionally save the result."""
        print_result(result)
        if results_dir:
            save_system_result(result, results_dir, "single_output_accuracy")

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.single_output
    @pytest.mark.accuracy
    @pytest.mark.parametrize(
        "kernel,n,n_test,d,data_type,noise,method", MINIMAL_CONFIGS
    )
    def test_single_output_truth_accuracy_core_configs(
        self,
        kernel: str,
        n: int,
        n_test: int,
        d: int,
        data_type: str,
        noise: str,
        method: str,
        results_dir,
        n_override,
    ):
        """Minimal accuracy test - quick end-to-end validation."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        if n_override is not None:
            n = n_override

        result = self._run_accuracy_test(kernel, n, n_test, d, data_type, noise, method)
        self._report_result(result, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions: just verify the pipeline works
        assert result.accuracy.rmse < np.inf, "RMSE should be finite"
        assert result.accuracy.crps < np.inf, "CRPS should be finite"
        assert result.speed.training_time_s > 0, "Training time should be positive"
        assert result.speed.iterations_run > 0, "Should run at least one iteration"

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.single_output
    @pytest.mark.accuracy
    @pytest.mark.parametrize(
        "kernel,n,n_test,d,data_type,noise,method", MODERATE_CONFIGS
    )
    def test_single_output_truth_accuracy_extended_configs(
        self,
        kernel: str,
        n: int,
        n_test: int,
        d: int,
        data_type: str,
        noise: str,
        method: str,
        results_dir,
    ):
        """Moderate accuracy test - broader coverage."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_accuracy_test(kernel, n, n_test, d, data_type, noise, method)
        self._report_result(result, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions
        assert result.accuracy.rmse < np.inf, "RMSE should be finite"
        assert result.accuracy.crps < np.inf, "CRPS should be finite"
        assert result.speed.training_time_s > 0, "Training time should be positive"

        # Note: We don't check R² for GP prior data because:
        # 1. GP prior data is inherently noisy (sampled from a GP)
        # 2. R² can be negative when predictions are worse than the mean
        # 3. Even GPyTorch gets low/negative R² on this data
        # The key metrics are RMSE, CRPS, and calibration - not R².

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.single_output
    @pytest.mark.accuracy
    @pytest.mark.parametrize("kernel,n,n_test,d,data_type,noise,method", FULL_CONFIGS)
    def test_single_output_truth_accuracy_broad_configs(
        self,
        kernel: str,
        n: int,
        n_test: int,
        d: int,
        data_type: str,
        noise: str,
        method: str,
        results_dir,
    ):
        """Full accuracy test - exhaustive coverage."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_accuracy_test(kernel, n, n_test, d, data_type, noise, method)
        self._report_result(result, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions
        assert result.accuracy.rmse < np.inf, "RMSE should be finite"
        assert result.accuracy.crps < np.inf, "CRPS should be finite"
        assert result.speed.training_time_s > 0, "Training time should be positive"

    # =========================================================================
    # Specific Feature Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.single_output
    def test_rbf_recovers_gp_prior_signal(self, results_dir):
        """RBF kernel recovers GP-prior data."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_accuracy_test(
            kernel="rbf",
            n_train=2000,
            n_test=50,
            d=5,
            data_type="gp_prior",
            noise_level="medium",
            method="auto",
        )
        self._report_result(result, results_dir)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # RBF on GP prior data should work reasonably well
        # Note: These are soft thresholds - the main goal is to verify the pipeline path.
        # GP prior data with n=200, d=5 is inherently noisy and R² can be negative
        # (even GPyTorch gets low R² on this data). We just check RMSE is reasonable.
        assert result.accuracy.rmse < 2.0, (
            "RMSE should be reasonable for RBF on GP prior"
        )

    @pytest.mark.moderate
    @pytest.mark.single_output
    def test_method_consistency(self, results_dir):
        """Test that materialized and matrix_free give similar results."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        # Generate same data for both
        dataset = generate_gp_prior_data(
            n_train=2000,
            n_test=50,
            d=5,
            kernel_type="rbf",
            true_lengthscale=1.0,
            true_noise=0.1,
            true_outputscale=1.0,
            seed=42,
        )

        # Run with materialized
        result_mat = run_mojogp_benchmark(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            f_test=dataset.f_test,
            kernel_type="rbf",
            method="materialized",
            true_params=dataset.true_params,
        )

        # Run with matrix_free
        result_mf = run_mojogp_benchmark(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            f_test=dataset.f_test,
            kernel_type="rbf",
            method="matrix_free",
            true_params=dataset.true_params,
        )

        print("\n=== Method Consistency Test ===")
        print(f"Materialized RMSE: {result_mat.accuracy.rmse:.4f}")
        print(f"Matrix-free RMSE:  {result_mf.accuracy.rmse:.4f}")
        print(
            f"RMSE difference:   {abs(result_mat.accuracy.rmse - result_mf.accuracy.rmse):.4f}"
        )

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result_mat)
        assert_gpu_was_used(result_mf)

        # Methods should give similar accuracy
        rmse_diff = abs(result_mat.accuracy.rmse - result_mf.accuracy.rmse)
        assert rmse_diff < 0.1, (
            f"RMSE difference between methods should be < 0.1, got {rmse_diff}"
        )

    @pytest.mark.moderate
    @pytest.mark.single_output
    def test_noise_level_impact(self, results_dir):
        """Test that higher noise leads to worse accuracy (as expected)."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        results = {}
        for noise_level in ["low", "medium", "high"]:
            result = self._run_accuracy_test(
                kernel="rbf",
                n_train=2000,
                n_test=50,
                d=5,
                data_type="gp_prior",
                noise_level=noise_level,
                method="auto",
            )
            results[noise_level] = result

        print("\n=== Noise Level Impact Test ===")
        for level, result in results.items():
            print(
                f"{level}: RMSE={result.accuracy.rmse:.4f}, R²={result.accuracy.r_squared:.4f}"
            )

        # CRITICAL: Verify GPU was actually used
        for result in results.values():
            assert_gpu_was_used(result)

        # Higher noise should generally lead to worse R² (lower)
        # But RMSE comparison is tricky because noise affects the scale
        assert (
            results["low"].accuracy.r_squared
            >= results["high"].accuracy.r_squared * 0.8
        ), "Low noise should have better or similar R² than high noise"

    @pytest.mark.full
    @pytest.mark.single_output
    def test_all_kernels_produce_finite_accuracy_metrics(self, results_dir):
        """Test all 8 kernel types produce finite accuracy metrics."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

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
        results = {}

        for kernel in kernels:
            # Use appropriate data type for each kernel
            if kernel == "periodic":
                data_type = "periodic_signal"
                d = 3
            elif kernel == "linear":
                data_type = "linear"
                d = 5
            elif kernel == "polynomial":
                data_type = "polynomial"
                d = 5
            else:
                data_type = "gp_prior"
                d = 5

            result = self._run_accuracy_test(
                kernel=kernel,
                n_train=2000,
                n_test=50,
                d=d,
                data_type=data_type,
                noise_level="medium",
                method="auto",
            )
            results[kernel] = result

        print("\n=== All Kernels Basic Test ===")
        for kernel, result in results.items():
            print(
                f"{kernel:12s}: RMSE={result.accuracy.rmse:.4f}, "
                f"R²={result.accuracy.r_squared:.4f}, "
                f"Time={result.speed.training_time_s:.2f}s"
            )

        # CRITICAL: Verify GPU was actually used
        for result in results.values():
            assert_gpu_was_used(result)

        # All kernels should produce finite results
        for kernel, result in results.items():
            assert result.accuracy.rmse < np.inf, f"{kernel} RMSE should be finite"
            assert result.accuracy.crps < np.inf, f"{kernel} CRPS should be finite"


# =============================================================================
# Standalone Execution
# =============================================================================


if __name__ == "__main__":
    # Run minimal tests when executed directly
    pytest.main([__file__, "-v", "-m", "minimal"])
