"""Report generation for system benchmarks.

Provides functions to generate formatted tables and JSON reports
from benchmark results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from .result_types import BenchmarkResult, ComparisonResult, generate_result_filename


# =============================================================================
# Table Formatting
# =============================================================================


def format_value(value: Any, precision: int = 4) -> str:
    """Format a value for display in a table."""
    if value is None:
        return "N/A"
    elif isinstance(value, bool):
        return "Yes" if value else "No"
    elif isinstance(value, float):
        if abs(value) < 0.0001 or abs(value) > 10000:
            return f"{value:.2e}"
        return f"{value:.{precision}f}"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, dict):
        return str(value)
    else:
        return str(value)


def _config_metadata(result: BenchmarkResult) -> list[str]:
    config = result.config
    metadata = []
    for key in (
        "comparison_class",
        "baseline_backend",
        "ard",
        "relevant_dims",
        "keops_supported",
        "keops_used",
        "cg_telemetry_quality",
        "fairness_note",
    ):
        if key in config:
            metadata.append(f"{key}={config[key]}")

    precond = config.get("preconditioner_config")
    if isinstance(precond, dict):
        family = precond.get("family", "n/a")
        rank = precond.get("rank", "n/a")
        method = precond.get("method", "n/a")
        metadata.append(f"precond={family}:{method}:rank={rank}")

    return metadata


def format_accuracy_table(result: BenchmarkResult, title: Optional[str] = None) -> str:
    """Format a single benchmark result as a readable table."""
    lines = []
    config = result.config

    if title:
        lines.append(f"\n{title}")
    else:
        lines.append(f"\nBenchmark Result")

    lines.append(
        f"Config: framework={config.get('framework', 'N/A')}, "
        f"model={config.get('model_type', 'N/A')}, "
        f"kernel={config.get('kernel', 'N/A')}, "
        f"n={config.get('n', 'N/A')}, d={config.get('d', 'N/A')}, "
        f"training={config.get('training_method', config.get('method', 'N/A'))}, "
        f"method={config.get('method', 'N/A')}, "
        f"prediction={config.get('prediction_mode', 'N/A')}"
    )
    metadata = _config_metadata(result)
    if metadata:
        lines.append("Metadata: " + ", ".join(metadata))
    lines.append("=" * 60)

    lines.append("PREDICTIVE ACCURACY:")
    lines.append(f"  RMSE:              {format_value(result.accuracy.rmse)}")
    lines.append(f"  MAE:               {format_value(result.accuracy.mae)}")
    lines.append(f"  R-squared:         {format_value(result.accuracy.r_squared)}")
    lines.append(f"  CRPS:              {format_value(result.accuracy.crps)}")
    lines.append(f"  MSLL:              {format_value(result.accuracy.msll)}")
    lines.append(
        f"  Calibration Error: {format_value(result.accuracy.calibration_error)}"
    )
    lines.append(f"  Sharpness:         {format_value(result.accuracy.sharpness)}")

    if result.accuracy.calibration_coverage:
        lines.append("  Calibration Coverage:")
        for level, coverage in sorted(result.accuracy.calibration_coverage.items()):
            lines.append(f"    {level * 100:.0f}%: {format_value(coverage, 2)}")

    lines.append("")
    lines.append("TRAINING:")
    lines.append(
        f"  Iterations:        {result.speed.iterations_run}/{result.speed.max_iterations}"
    )
    lines.append(f"  Early Stopped:     {format_value(result.speed.early_stopped)}")
    lines.append(f"  Training Time:     {format_value(result.speed.training_time_s)}s")
    lines.append(f"  ms/iteration:      {format_value(result.speed.ms_per_iteration)}")
    if result.speed.iter_time_p5_ms is not None:
        lines.append(
            f"  Iter p5/p95 ms:    {format_value(result.speed.iter_time_p5_ms)}/{format_value(result.speed.iter_time_p95_ms)}"
        )
    lines.append(
        f"  Pred Mean Time:    {format_value(result.speed.prediction_mean_time_s)}s"
    )
    lines.append(
        f"  Pred Var Time:     {format_value(result.speed.prediction_variance_time_s)}s"
    )
    if result.speed.prediction_cold_first_time_s is not None:
        lines.append(
            f"  Pred Cold First:   {format_value(result.speed.prediction_cold_first_time_s)}s"
        )
    if result.speed.prediction_cache_prepare_time_s is not None:
        lines.append(
            f"  Pred Cache Prep:   {format_value(result.speed.prediction_cache_prepare_time_s)}s"
        )
    if result.speed.prediction_prepared_apply_time_s is not None:
        lines.append(
            f"  Pred Prepared:     {format_value(result.speed.prediction_prepared_apply_time_s)}s"
        )
    if result.speed.prediction_repeated_median_time_s is not None:
        lines.append(
            "  Pred Repeat p5/med/p95: "
            f"{format_value(result.speed.prediction_repeated_p5_time_s)}/"
            f"{format_value(result.speed.prediction_repeated_median_time_s)}/"
            f"{format_value(result.speed.prediction_repeated_p95_time_s)}s"
        )
    if result.speed.prediction_alpha_time_s is not None:
        lines.append(
            f"  Pred Alpha Setup:  {format_value(result.speed.prediction_alpha_time_s)}s"
        )
    if result.speed.prediction_love_root_time_s is not None:
        lines.append(
            f"  Pred LOVE Root:    {format_value(result.speed.prediction_love_root_time_s)}s"
        )
    if result.speed.prediction_x_test_scaling:
        lines.append(
            f"  Pred X_test Rows:  {len(result.speed.prediction_x_test_scaling)}"
        )
    lines.append(
        f"  End-to-End Time:   {format_value(result.speed.end_to_end_time_s)}s"
    )
    if result.speed.startup_compile_time_s is not None:
        lines.append(
            f"  JIT Cold Compile:  {format_value(result.speed.startup_compile_time_s)}s"
        )
    if result.speed.startup_warm_cache_hit_s is not None:
        lines.append(
            f"  JIT Warm Cache:    {format_value(result.speed.startup_warm_cache_hit_s)}s"
        )
    if result.speed.startup_prepare_time_s is not None:
        lines.append(
            f"  JIT Prepare:       {format_value(result.speed.startup_prepare_time_s)}s"
        )

    lines.append("")
    lines.append("MEMORY:")
    lines.append(f"  GPU Peak:          {format_value(result.memory.gpu_max_mb)} MB")
    lines.append(f"  GPU Mean:          {format_value(result.memory.gpu_mean_mb)} MB")
    lines.append(f"  Torch Peak:        {format_value(result.memory.torch_peak_mb)} MB")
    lines.append(f"  CPU Peak:          {format_value(result.memory.cpu_peak_mb)} MB")

    lines.append("")
    lines.append("HYPERPARAMETERS:")
    lines.append(
        f"  Lengthscale:       {format_value(result.hyperparameters.learned_lengthscale)}"
    )
    lines.append(
        f"  Noise:             {format_value(result.hyperparameters.learned_noise)}"
    )
    lines.append(
        f"  Outputscale:       {format_value(result.hyperparameters.learned_outputscale)}"
    )
    if result.hyperparameters.learned_mean is not None:
        lines.append(
            f"  Mean:              {format_value(result.hyperparameters.learned_mean)}"
        )
    lines.append(
        f"  Final NLL:         {format_value(result.hyperparameters.final_nll)}"
    )

    if result.hyperparameters.lengthscale_rel_error is not None:
        lines.append("  Recovery Errors:")
        lines.append(
            f"    Lengthscale:     {result.hyperparameters.lengthscale_rel_error:.2%}"
        )
        if result.hyperparameters.noise_rel_error is not None:
            lines.append(
                f"    Noise:           {result.hyperparameters.noise_rel_error:.2%}"
            )
        if result.hyperparameters.outputscale_rel_error is not None:
            lines.append(
                f"    Outputscale:     {result.hyperparameters.outputscale_rel_error:.2%}"
            )
        if result.hyperparameters.mean_rel_error is not None:
            lines.append(
                f"    Mean:            {result.hyperparameters.mean_rel_error:.2%}"
            )

    if config.get("ard"):
        lines.append("")
        lines.append("ARD RELEVANCE:")
        lines.append(
            "  Pairwise Acc:      "
            f"{format_value(config.get('pairwise_relevance_accuracy'))}"
        )
        lines.append(
            "  Top-k Hit Rate:    "
            f"{format_value(config.get('top_k_relevance_hit_rate'))}"
        )
        lines.append(
            "  Separation Ratio:  "
            f"{format_value(config.get('relevance_separation_ratio'))}"
        )
        lines.append(
            "  Mean Rel/Irrel LS: "
            f"{format_value(config.get('mean_relevant_lengthscale'))}/"
            f"{format_value(config.get('mean_irrelevant_lengthscale'))}"
        )
        lines.append(
            "  Status:            "
            f"{format_value(config.get('ard_quality_status'))}"
        )

    return "\n".join(lines)


def format_comparison_table(comparison: ComparisonResult) -> str:
    """Format a comparison result as a readable table."""
    lines = []
    config = comparison.config

    lines.append(
        f"\nComparison: kernel={config.get('kernel', 'N/A')}, "
        f"n={config.get('n', 'N/A')}, d={config.get('d', 'N/A')}"
    )
    lines.append("=" * 90)

    # Column headers
    headers = ["Metric", "MojoGP Mat", "MojoGP MF", "GPyTorch CG", "GPyTorch KeOps"]
    col_widths = [25, 12, 12, 12, 14]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * 90)

    def get_val(result: Optional[BenchmarkResult], attr_path: str) -> str:
        if result is None:
            return "N/A"
        try:
            obj = result
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return format_value(obj)
        except (AttributeError, TypeError):
            return "N/A"

    def row(metric: str, path: str) -> str:
        vals = [
            get_val(comparison.mojogp_materialized, path),
            get_val(comparison.mojogp_matrix_free, path),
            get_val(comparison.gpytorch_cg, path),
            get_val(comparison.gpytorch_keops, path),
        ]
        return f"{metric:<25} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12} | {vals[3]:>14}"

    # Accuracy metrics
    lines.append("PREDICTIVE ACCURACY (PRIMARY):")
    lines.append(row("  RMSE", "accuracy.rmse"))
    lines.append(row("  CRPS", "accuracy.crps"))
    lines.append(row("  Calibration Error", "accuracy.calibration_error"))
    lines.append(row("  R-squared", "accuracy.r_squared"))
    lines.append(row("  MSLL", "accuracy.msll"))
    lines.append(row("  Sharpness", "accuracy.sharpness"))

    # Calibration coverage (special handling)
    lines.append("  Calibration (95%):")
    vals = []
    for r in [
        comparison.mojogp_materialized,
        comparison.mojogp_matrix_free,
        comparison.gpytorch_cg,
        comparison.gpytorch_keops,
    ]:
        if r and r.accuracy.calibration_coverage:
            vals.append(format_value(r.accuracy.calibration_coverage.get(0.95, 0), 2))
        else:
            vals.append("N/A")
    lines.append(
        f"{'':25} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12} | {vals[3]:>14}"
    )

    lines.append("")
    lines.append("TRAINING DETAILS:")
    lines.append(row("  Iterations Run", "speed.iterations_run"))
    lines.append(row("  Max Iterations", "speed.max_iterations"))
    lines.append(row("  Early Stopped", "speed.early_stopped"))
    lines.append(row("  Training Time (s)", "speed.training_time_s"))
    lines.append(row("  ms/iteration", "speed.ms_per_iteration"))
    lines.append(row("  Pred Mean Time (s)", "speed.prediction_mean_time_s"))
    lines.append(row("  Pred Var Time (s)", "speed.prediction_variance_time_s"))
    lines.append(row("  Pred Cold First (s)", "speed.prediction_cold_first_time_s"))
    lines.append(row("  Pred Cache Prep (s)", "speed.prediction_cache_prepare_time_s"))
    lines.append(row("  Pred Prepared (s)", "speed.prediction_prepared_apply_time_s"))
    lines.append(row("  Pred Repeat Median", "speed.prediction_repeated_median_time_s"))
    lines.append(row("  Pred Alpha Setup", "speed.prediction_alpha_time_s"))
    lines.append(row("  Pred LOVE Root", "speed.prediction_love_root_time_s"))
    lines.append(row("  Pred X_test Rows", "speed.prediction_x_test_scaling"))
    lines.append(row("  End-to-End Time (s)", "speed.end_to_end_time_s"))

    lines.append("")
    lines.append("MEMORY:")
    lines.append(row("  Peak GPU Mem (MB)", "memory.gpu_max_mb"))
    lines.append(row("  Torch Peak (MB)", "memory.torch_peak_mb"))

    lines.append("")
    lines.append("HYPERPARAMETERS (INVESTIGATORY):")
    lines.append(row("  Final NLL", "hyperparameters.final_nll"))
    lines.append(row("  Lengthscale", "hyperparameters.learned_lengthscale"))
    lines.append(row("  Noise", "hyperparameters.learned_noise"))
    lines.append(row("  Outputscale", "hyperparameters.learned_outputscale"))

    # Derived comparisons
    if comparison.rmse_ratio_vs_cg is not None:
        lines.append("")
        lines.append("DERIVED COMPARISONS (vs GPyTorch CG):")
        lines.append(
            f"  RMSE Ratio:        {format_value(comparison.rmse_ratio_vs_cg)} (< 1 = MojoGP better)"
        )
        lines.append(
            f"  CRPS Ratio:        {format_value(comparison.crps_ratio_vs_cg)} (< 1 = MojoGP better)"
        )
        lines.append(
            f"  Speedup:           {format_value(comparison.speedup_vs_cg)} (> 1 = MojoGP faster)"
        )
        lines.append(
            f"  Memory Ratio:      {format_value(comparison.memory_ratio_vs_cg)} (< 1 = MojoGP leaner)"
        )

    return "\n".join(lines)


def format_summary_table(results: List[BenchmarkResult], title: str = "Summary") -> str:
    """Format a summary table of multiple benchmark results."""
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 120)

    # Header
    headers = [
        "Kernel",
        "Framework",
        "Class",
        "Base",
        "Model",
        "n",
        "d",
        "Train",
        "Solver",
        "Pred",
        "KeOps",
        "Precond",
        "RMSE",
        "CRPS",
        "Cal.Err",
        "Time(s)",
        "Mem(MB)",
        "Iters",
    ]
    col_widths = [10, 10, 12, 10, 14, 6, 4, 12, 10, 8, 7, 14, 8, 8, 8, 8, 8, 6]
    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * 120)

    for result in results:
        config = result.config
        row = [
            str(config.get("kernel", "N/A"))[:10],
            str(config.get("framework", "N/A"))[:10],
            str(config.get("comparison_class", "N/A"))[:12],
            str(config.get("baseline_backend", "N/A"))[:10],
            str(config.get("model_type", "N/A"))[:14],
            str(config.get("n", "N/A")),
            str(config.get("d", "N/A")),
            str(config.get("training_method", config.get("method", "N/A")))[:12],
            str(config.get("method", "N/A"))[:12],
            str(config.get("prediction_mode", "N/A"))[:8],
            str(config.get("keops_used", "N/A"))[:7],
            str((config.get("preconditioner_config") or {}).get("method", "N/A"))[:14],
            format_value(result.accuracy.rmse, 4),
            format_value(result.accuracy.crps, 4),
            format_value(result.accuracy.calibration_error, 4),
            format_value(result.speed.training_time_s, 2),
            format_value(result.memory.gpu_max_mb, 1),
            str(result.speed.iterations_run),
        ]
        row_line = " | ".join(v.center(w) for v, w in zip(row, col_widths))
        lines.append(row_line)

    return "\n".join(lines)


# =============================================================================
# Report Persistence
# =============================================================================


def _write_json_payload(payload: str, results_dir: Path, test_type: str, config: Dict[str, Any]) -> Path:
    """Write a JSON payload to the results directory."""
    filename = generate_result_filename(test_type, config)
    filepath = results_dir / filename
    with open(filepath, "w") as f:
        f.write(payload)
    return filepath


def save_result_artifact(
    result: BenchmarkResult,
    results_dir: Path,
    test_type: str,
) -> Path:
    """Save a result artifact without benchmark-session persistence."""
    return _write_json_payload(result.to_json(), results_dir, test_type, result.config)


def save_comparison_artifact(
    result: ComparisonResult,
    results_dir: Path,
    test_type: str,
) -> Path:
    """Save a comparison artifact without benchmark-session persistence."""
    return _write_json_payload(result.to_json(), results_dir, test_type, result.config)


def save_system_result(
    result: BenchmarkResult,
    results_dir: Path,
    test_type: str,
) -> Path:
    """Save a non-benchmark system-test result artifact."""
    return save_result_artifact(result, results_dir, test_type)


def save_system_comparison(
    result: ComparisonResult,
    results_dir: Path,
    test_type: str,
) -> Path:
    """Save a non-benchmark system-test comparison artifact."""
    return save_comparison_artifact(result, results_dir, test_type)


def save_summary_report(
    results: List[BenchmarkResult],
    results_dir: Path,
    report_name: str,
    failures: list[dict[str, Any]] | None = None,
) -> Path:
    """Save a summary report with all results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{report_name}_summary.json"
    filepath = results_dir / filename

    summary = {
        "report_name": report_name,
        "timestamp": datetime.now().isoformat(),
        "num_results": len(results),
        "results": [r.to_dict() for r in results],
        "statistics": compute_summary_statistics(results),
        "failures": list(failures or []),
    }

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return filepath


def compute_summary_statistics(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute summary statistics across all results."""
    if not results:
        return {}

    rmses = [r.accuracy.rmse for r in results]
    crps_vals = [r.accuracy.crps for r in results]
    times = [r.speed.training_time_s for r in results]
    memories = [r.memory.gpu_max_mb for r in results if r.memory.gpu_max_mb > 0]

    stats = {
        "rmse": {
            "mean": float(np.mean(rmses)),
            "std": float(np.std(rmses)),
            "min": float(np.min(rmses)),
            "max": float(np.max(rmses)),
        },
        "crps": {
            "mean": float(np.mean(crps_vals)),
            "std": float(np.std(crps_vals)),
            "min": float(np.min(crps_vals)),
            "max": float(np.max(crps_vals)),
        },
        "training_time_s": {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "total": float(np.sum(times)),
        },
    }

    if memories:
        stats["gpu_memory_mb"] = {
            "mean": float(np.mean(memories)),
            "std": float(np.std(memories)),
            "min": float(np.min(memories)),
            "max": float(np.max(memories)),
        }

    return stats


# =============================================================================
# Report Loading
# =============================================================================


def load_benchmark_result(filepath: Path) -> BenchmarkResult:
    """Load a benchmark result from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Reconstruct the dataclasses
    from .result_types import (
        AccuracyResult,
        SpeedResult,
        MemoryResult,
        HyperparameterResult,
    )

    accuracy_data = dict(data["accuracy"])
    coverage = accuracy_data.get("calibration_coverage")
    if isinstance(coverage, dict):
        accuracy_data["calibration_coverage"] = {
            float(level): value for level, value in coverage.items()
        }

    accuracy = AccuracyResult(**accuracy_data)
    speed = SpeedResult(**data["speed"])
    memory = MemoryResult(**data["memory"])
    hyperparameters = HyperparameterResult(**data["hyperparameters"])

    return BenchmarkResult(
        config=data["config"],
        accuracy=accuracy,
        speed=speed,
        memory=memory,
        hyperparameters=hyperparameters,
        timestamp=data.get("timestamp", ""),
        run_id=data.get("run_id", ""),
    )


def load_comparison_result(filepath: Path) -> ComparisonResult:
    """Load a comparison result from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    def _maybe_result(payload: dict[str, Any] | None) -> BenchmarkResult | None:
        if payload is None:
            return None
        accuracy_data = dict(payload["accuracy"])
        coverage = accuracy_data.get("calibration_coverage")
        if isinstance(coverage, dict):
            accuracy_data["calibration_coverage"] = {
                float(level): value for level, value in coverage.items()
            }

        from .result_types import AccuracyResult, HyperparameterResult, MemoryResult, SpeedResult

        return BenchmarkResult(
            config=payload["config"],
            accuracy=AccuracyResult(**accuracy_data),
            speed=SpeedResult(**payload["speed"]),
            memory=MemoryResult(**payload["memory"]),
            hyperparameters=HyperparameterResult(**payload["hyperparameters"]),
            timestamp=payload.get("timestamp", ""),
            run_id=payload.get("run_id", ""),
        )

    return ComparisonResult(
        config=data["config"],
        mojogp_materialized=_maybe_result(data.get("mojogp_materialized")),
        mojogp_matrix_free=_maybe_result(data.get("mojogp_matrix_free")),
        gpytorch_cg=_maybe_result(data.get("gpytorch_cg")),
        gpytorch_keops=_maybe_result(data.get("gpytorch_keops")),
        rmse_ratio_vs_cg=data.get("rmse_ratio_vs_cg"),
        crps_ratio_vs_cg=data.get("crps_ratio_vs_cg"),
        calibration_diff_vs_cg=data.get("calibration_diff_vs_cg"),
        speedup_vs_cg=data.get("speedup_vs_cg"),
        memory_ratio_vs_cg=data.get("memory_ratio_vs_cg"),
        nll_ratio_vs_cg=data.get("nll_ratio_vs_cg"),
        hyperparam_alignment=data.get("hyperparam_alignment"),
        timestamp=data.get("timestamp", ""),
        run_id=data.get("run_id", ""),
    )


def load_all_results(
    results_dir: Path, pattern: str = "*.json"
) -> List[BenchmarkResult]:
    """Load all benchmark results from a directory."""
    results = []
    for filepath in results_dir.glob(pattern):
        try:
            result = load_benchmark_result(filepath)
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    return results


# =============================================================================
# Console Output
# =============================================================================


def print_result(result: BenchmarkResult, title: Optional[str] = None):
    """Print a benchmark result to console."""
    print(format_accuracy_table(result, title))


def print_comparison(comparison: ComparisonResult):
    """Print a comparison result to console."""
    print(format_comparison_table(comparison))


def print_summary(results: List[BenchmarkResult], title: str = "Summary"):
    """Print a summary table to console."""
    print(format_summary_table(results, title))
