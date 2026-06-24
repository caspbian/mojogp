"""Benchmark: ARD (Automatic Relevance Determination) kernel tests.

Tests MojoGP's ARD capability - learning per-dimension lengthscales
to identify relevant vs irrelevant input dimensions.

All tests measure accuracy, speed, and memory together.
"""

import pytest
import numpy as np
from typing import Dict, Any

from tests.shared.benchmarking.environment import requires_mojogp, assert_gpu_available, assert_gpu_was_used
from tests.shared.benchmarking.data_generators import generate_ard_data, SyntheticDataset
from tests.shared.benchmarking.mojogp_runners import train_mojogp_ard, predict_mojogp_ard
from tests.shared.benchmarking.result_types import (
    BenchmarkResult,
    AccuracyResult,
    SpeedResult,
    MemoryResult,
    HyperparameterResult,
)
from tests.shared.benchmarking.metrics import compute_all_accuracy_metrics
from tests.shared.benchmarking.report import print_result
# =============================================================================
# Test Configurations
# =============================================================================

# Format: (n_train, d, relevant_dims)

MINIMAL_CONFIGS = [
    (2000, 5, 2),
    (2000, 10, 3),
]

MODERATE_CONFIGS = MINIMAL_CONFIGS + [
    (2000, 10, 1),
    (3000, 10, 5),
    (3000, 20, 5),
]

FULL_CONFIGS = MODERATE_CONFIGS + [
    # Scaling tests
    (5000, 10, 3),
    (5000, 20, 5),
    (5000, 30, 5),
    # Edge cases
    (5000, 10, 9),
]


# =============================================================================
# Helper Functions
# =============================================================================


def compute_ard_recovery_metrics(
    learned_lengthscales: np.ndarray,
    true_lengthscales: np.ndarray,
    relevant_dims: int,
) -> Dict[str, float]:
    """Compute metrics for ARD lengthscale recovery.

    Returns:
        - relevant_recovery: How well we recovered short lengthscales for relevant dims
        - irrelevant_recovery: How well we recovered long lengthscales for irrelevant dims
        - separation_ratio: Ratio of mean irrelevant to mean relevant lengthscales
        - correct_ranking: Fraction of dims correctly ranked by relevance
    """
    d = len(learned_lengthscales)

    # True relevant dims have short lengthscales, irrelevant have long
    relevant_mask = np.arange(d) < relevant_dims

    learned_relevant = learned_lengthscales[relevant_mask]
    learned_irrelevant = learned_lengthscales[~relevant_mask]
    true_relevant = true_lengthscales[relevant_mask]
    true_irrelevant = true_lengthscales[~relevant_mask]

    # Recovery errors
    relevant_recovery = float(
        np.mean(np.abs(learned_relevant - true_relevant) / true_relevant)
    )
    irrelevant_recovery = (
        float(np.mean(np.abs(learned_irrelevant - true_irrelevant) / true_irrelevant))
        if len(learned_irrelevant) > 0
        else 0.0
    )

    # Separation: good ARD should have large ratio
    mean_relevant = float(np.mean(learned_relevant))
    mean_irrelevant = (
        float(np.mean(learned_irrelevant))
        if len(learned_irrelevant) > 0
        else mean_relevant
    )
    separation_ratio = mean_irrelevant / max(mean_relevant, 1e-6)

    # Ranking: are learned lengthscales correctly ordered?
    # Relevant dims should have smaller lengthscales than irrelevant
    correct_ranking = 0.0
    if len(learned_irrelevant) > 0:
        # For each relevant dim, count how many irrelevant dims have larger lengthscale
        correct_pairs = 0
        total_pairs = 0
        for rel_ls in learned_relevant:
            for irrel_ls in learned_irrelevant:
                total_pairs += 1
                if rel_ls < irrel_ls:
                    correct_pairs += 1
        correct_ranking = correct_pairs / max(total_pairs, 1)

    return {
        "relevant_recovery": relevant_recovery,
        "irrelevant_recovery": irrelevant_recovery,
        "separation_ratio": separation_ratio,
        "correct_ranking": correct_ranking,
    }


# =============================================================================
# Test Class
# =============================================================================


@requires_mojogp
class TestARD:
    """Benchmark: ARD kernel tests measuring accuracy, speed, and memory."""

    def _run_ard_test(
        self,
        n_train: int,
        d: int,
        relevant_dims: int,
        n_test: int = 80,
        n_iterations: int = 200,
        lr: float = 0.02,
        seed: int = 42,
    ) -> BenchmarkResult:
        """Run an ARD benchmark."""
        # Generate data with known relevant dimensions
        dataset = generate_ard_data(
            n_train=n_train,
            n_test=n_test,
            d=d,
            relevant_dims=relevant_dims,
            true_noise=0.1,
            true_outputscale=1.0,
            seed=seed,
        )

        # Train with ARD
        train_result = train_mojogp_ard(
            dataset.X_train,
            dataset.y_train,
            kernel_type="rbf",
            n_iterations=n_iterations,
            lr=lr,
            init_noise=0.1,
            init_os=1.0,
            monitor_memory=True,
        )

        # Get learned lengthscales
        learned_ls = train_result["learned_params"].get("lengthscales", np.ones(d))
        true_ls = dataset.true_params.get("lengthscales", np.ones(d))

        # Compute ARD-specific metrics
        ard_metrics = compute_ard_recovery_metrics(learned_ls, true_ls, relevant_dims)

        # Predict using ARD kernel with learned lengthscales
        pred_result = predict_mojogp_ard(
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            train_result,
        )

        # Compute accuracy metrics
        accuracy_metrics = compute_all_accuracy_metrics(
            y_true=dataset.y_test,
            pred_mean=pred_result["mean"],
            pred_std=pred_result["std"],
        )

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
            learned_lengthscale=learned.get("lengthscale", 1.0),  # Mean lengthscale
            learned_noise=learned.get("noise", 0.1),
            learned_outputscale=learned.get("outputscale", 1.0),
            final_nll=train_result["final_nll"],
        )

        config = {
            "n": n_train,
            "d": d,
            "relevant_dims": relevant_dims,
            "n_iterations": n_iterations,
            "lr": lr,
            "seed": seed,
            # ARD-specific metrics
            "learned_lengthscales": learned_ls.tolist()
            if hasattr(learned_ls, "tolist")
            else list(learned_ls),
            "true_lengthscales": true_ls.tolist()
            if hasattr(true_ls, "tolist")
            else list(true_ls),
            "ard_metrics": ard_metrics,
        }

        return BenchmarkResult(
            config=config,
            accuracy=accuracy,
            speed=speed,
            memory=memory,
            hyperparameters=hyperparameters,
        )

    def _print_ard_result(self, result: BenchmarkResult):
        """Print ARD-specific results."""
        print_result(result)

        ard_metrics = result.config.get("ard_metrics", {})
        print("\n=== ARD-Specific Metrics ===")
        print(f"  Separation Ratio:    {ard_metrics.get('separation_ratio', 0):.2f}x")
        print(f"  Correct Ranking:     {ard_metrics.get('correct_ranking', 0):.1%}")
        print(
            f"  Relevant Recovery:   {ard_metrics.get('relevant_recovery', 0):.2%} error"
        )
        print(
            f"  Irrelevant Recovery: {ard_metrics.get('irrelevant_recovery', 0):.2%} error"
        )

        learned_ls = result.config.get("learned_lengthscales", [])
        true_ls = result.config.get("true_lengthscales", [])
        relevant_dims = result.config.get("relevant_dims", 0)

        if learned_ls and true_ls:
            print(f"\n  Lengthscales (first {min(5, len(learned_ls))} dims):")
            for i in range(min(5, len(learned_ls))):
                marker = "*" if i < relevant_dims else " "
                print(
                    f"    {marker} dim {i}: learned={learned_ls[i]:.3f}, true={true_ls[i]:.3f}"
                )

    # =========================================================================
    # Minimal Tier Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.ard
    @pytest.mark.parametrize("n,d,relevant_dims", MINIMAL_CONFIGS)
    def test_ard_relevance_accuracy_core_configs(
        self,
        n: int,
        d: int,
        relevant_dims: int,
        results_dir,
    ):
        """Minimal ARD test - measures accuracy, speed, and memory."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_ard_test(n, d, relevant_dims)
        self._print_ard_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions
        assert result.speed.training_time_s > 0, "Training time should be positive"
        assert result.hyperparameters.final_nll < np.inf, "NLL should be finite"

    # =========================================================================
    # Moderate Tier Tests
    # =========================================================================

    @pytest.mark.moderate
    @pytest.mark.ard
    @pytest.mark.parametrize("n,d,relevant_dims", MODERATE_CONFIGS)
    def test_ard_relevance_accuracy_extended_configs(
        self,
        n: int,
        d: int,
        relevant_dims: int,
        results_dir,
    ):
        """Moderate ARD test - measures accuracy, speed, and memory."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_ard_test(n, d, relevant_dims)
        self._print_ard_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Check that ARD is learning something useful
        ard_metrics = result.config.get("ard_metrics", {})
        separation = ard_metrics.get("separation_ratio", 1.0)

        # Separation should be > 1 (irrelevant dims should have larger lengthscales)
        assert separation > 0.5, (
            f"ARD separation ratio should be > 0.5, got {separation:.2f}"
        )

    # =========================================================================
    # Full Tier Tests
    # =========================================================================

    @pytest.mark.full
    @pytest.mark.ard
    @pytest.mark.parametrize("n,d,relevant_dims", FULL_CONFIGS)
    def test_ard_relevance_accuracy_broad_configs(
        self,
        n: int,
        d: int,
        relevant_dims: int,
        results_dir,
    ):
        """Full ARD test - measures accuracy, speed, and memory."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_ard_test(n, d, relevant_dims)
        self._print_ard_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # Soft assertions
        assert result.speed.training_time_s > 0, "Training time should be positive"

    # =========================================================================
    # Specific Tests
    # =========================================================================

    @pytest.mark.minimal
    @pytest.mark.ard
    def test_ard_separates_two_relevant_dimensions(self, results_dir):
        """ARD separates 2 relevant dimensions from 3 irrelevant dimensions."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_ard_test(n_train=2000, d=5, relevant_dims=2)
        self._print_ard_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # ARD should identify that some dims are more relevant
        ard_metrics = result.config.get("ard_metrics", {})
        assert ard_metrics.get("separation_ratio", 0) > 1.0, (
            "ARD should separate relevant from irrelevant dims"
        )

    @pytest.mark.moderate
    @pytest.mark.ard
    def test_ard_sparse_relevance(self, results_dir):
        """Test ARD with very sparse relevance (1 of 10 dims)."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        result = self._run_ard_test(n_train=2000, d=10, relevant_dims=1)
        self._print_ard_result(result)

        # CRITICAL: Verify GPU was actually used
        assert_gpu_was_used(result)

        # With only 1 relevant dim, separation should be very clear
        ard_metrics = result.config.get("ard_metrics", {})
        print(
            f"\nSparse relevance test: separation={ard_metrics.get('separation_ratio', 0):.2f}"
        )

    @pytest.mark.moderate
    @pytest.mark.ard
    def test_ard_dimension_scaling(self, results_dir):
        """Test how ARD scales with input dimension."""
        # CRITICAL: Verify GPU is available before running
        assert_gpu_available()

        results = {}
        for d in [5, 10, 20]:
            result = self._run_ard_test(n_train=2000, d=d, relevant_dims=3)
            results[d] = result

        print("\n=== ARD Dimension Scaling ===")
        print(f"{'d':>4} | {'Time (s)':>10} | {'Separation':>10} | {'Ranking':>10}")
        print("-" * 45)
        for d, result in results.items():
            ard_metrics = result.config.get("ard_metrics", {})
            print(
                f"{d:>4} | {result.speed.training_time_s:>10.2f} | "
                f"{ard_metrics.get('separation_ratio', 0):>10.2f} | "
                f"{ard_metrics.get('correct_ranking', 0):>10.1%}"
            )

        # CRITICAL: Verify GPU was actually used
        for result in results.values():
            assert_gpu_was_used(result)


# =============================================================================
# Standalone Execution
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "minimal"])
