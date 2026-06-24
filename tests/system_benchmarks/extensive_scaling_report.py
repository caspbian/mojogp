"""Reporting helpers for extensive single-output scaling benchmarks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.shared.benchmarking.result_types import BenchmarkResult


def _prediction_apply_time(result: BenchmarkResult) -> float:
    prepared = result.speed.prediction_prepared_apply_time_s
    if prepared is not None:
        return float(prepared)
    return float(result.speed.prediction_mean_time_s + result.speed.prediction_variance_time_s)


def _result_row(result: BenchmarkResult) -> dict[str, Any]:
    config = result.config
    return {
        "framework": config.get("framework"),
        "training_method": config.get("training_method"),
        "prediction_mode": config.get("prediction_mode"),
        "mojogp_preset": config.get("mojogp_preset"),
        "baseline_backend": config.get("baseline_backend"),
        "data_mean_offset": config.get("data_mean_offset"),
        "data_noise_level": config.get("data_noise_level"),
        "n": int(config.get("n", 0)),
        "d": int(config.get("d", 0)),
        "comparison_class": config.get("comparison_class"),
        "fairness_note": config.get("fairness_note"),
        "training_time_s": float(result.speed.training_time_s),
        "ms_per_iteration": float(result.speed.ms_per_iteration),
        "iter_time_p5_ms": result.speed.iter_time_p5_ms,
        "iter_time_p95_ms": result.speed.iter_time_p95_ms,
        "prediction_mean_time_s": float(result.speed.prediction_mean_time_s),
        "prediction_variance_time_s": float(result.speed.prediction_variance_time_s),
        "prediction_total_time_s": float(
            result.speed.prediction_mean_time_s + result.speed.prediction_variance_time_s
        ),
        "prediction_cold_first_time_s": result.speed.prediction_cold_first_time_s,
        "prediction_cache_prepare_time_s": result.speed.prediction_cache_prepare_time_s,
        "prediction_prepared_apply_time_s": result.speed.prediction_prepared_apply_time_s,
        "prediction_repeated_median_time_s": result.speed.prediction_repeated_median_time_s,
        "prediction_repeated_p5_time_s": result.speed.prediction_repeated_p5_time_s,
        "prediction_repeated_p95_time_s": result.speed.prediction_repeated_p95_time_s,
        "prediction_apply_time_s": _prediction_apply_time(result),
        "prediction_alpha_time_s": result.speed.prediction_alpha_time_s,
        "prediction_love_root_time_s": result.speed.prediction_love_root_time_s,
        "iter_timing_quality": result.speed.iter_timing_quality,
        "gpu_max_mb": result.memory.gpu_max_mb,
        "training_peak_gpu_mb": result.memory.training_peak_gpu_mb,
        "training_delta_gpu_mb": result.memory.training_delta_gpu_mb,
        "prediction_peak_gpu_mb": result.memory.prediction_peak_gpu_mb,
        "prediction_delta_gpu_mb": result.memory.prediction_delta_gpu_mb,
        "exact_prediction_peak_gpu_mb": result.memory.exact_prediction_peak_gpu_mb,
        "exact_prediction_delta_gpu_mb": result.memory.exact_prediction_delta_gpu_mb,
        "love_prediction_peak_gpu_mb": result.memory.love_prediction_peak_gpu_mb,
        "love_prediction_delta_gpu_mb": result.memory.love_prediction_delta_gpu_mb,
        "prediction_timing_quality": config.get("prediction_timing_quality"),
        "rmse": float(result.accuracy.rmse),
        "r_squared": float(result.accuracy.r_squared),
        "crps": float(result.accuracy.crps),
        "msll": float(result.accuracy.msll),
        "final_nll": float(result.hyperparameters.final_nll),
        "learned_noise": float(result.hyperparameters.learned_noise),
        "learned_mean": result.hyperparameters.learned_mean,
        "noise_rel_error": result.hyperparameters.noise_rel_error,
        "mean_rel_error": result.hyperparameters.mean_rel_error,
        "cg_training": config.get("cg_telemetry", {}).get("training", {}),
        "cg_prediction": config.get("cg_telemetry", {}).get("prediction", {}),
    }


def _group_key(result: BenchmarkResult) -> tuple[Any, ...]:
    config = result.config
    return (
        config.get("framework"),
        config.get("training_method"),
        config.get("prediction_mode"),
        config.get("mojogp_preset"),
        config.get("baseline_backend"),
        config.get("d"),
    )


def _group_key_by_n(result: BenchmarkResult) -> tuple[Any, ...]:
    config = result.config
    return (
        config.get("framework"),
        config.get("training_method"),
        config.get("prediction_mode"),
        config.get("mojogp_preset"),
        config.get("baseline_backend"),
        config.get("n"),
    )


def _compute_n_scaling(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(_group_key(result), []).append(result)

    rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda r: int(r.config.get("n", 0)))
        if len(items) < 2:
            continue
        for smaller, larger in zip(items[:-1], items[1:]):
            n1 = int(smaller.config["n"])
            n2 = int(larger.config["n"])
            rows.append(
                {
                    "framework": key[0],
                    "training_method": key[1],
                    "prediction_mode": key[2],
                    "mojogp_preset": key[3],
                    "baseline_backend": key[4],
                    "d": key[5],
                    "n1": n1,
                    "n2": n2,
                    "n_growth_ratio": float(n2 / n1),
                    "train_time_ratio": float(
                        larger.speed.training_time_s / smaller.speed.training_time_s
                    ),
                    "iter_time_ratio": float(
                        larger.speed.ms_per_iteration / smaller.speed.ms_per_iteration
                    ),
                    "prediction_total_ratio": float(
                        (larger.speed.prediction_mean_time_s + larger.speed.prediction_variance_time_s)
                        / (smaller.speed.prediction_mean_time_s + smaller.speed.prediction_variance_time_s)
                    ),
                    "prediction_apply_ratio": float(
                        _prediction_apply_time(larger) / _prediction_apply_time(smaller)
                    ),
                    "rmse_ratio": float(larger.accuracy.rmse / smaller.accuracy.rmse),
                }
            )
    return rows


def _compute_d_scaling(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(_group_key_by_n(result), []).append(result)

    rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda r: int(r.config.get("d", 0)))
        if len(items) < 2:
            continue
        for smaller, larger in zip(items[:-1], items[1:]):
            d1 = int(smaller.config["d"])
            d2 = int(larger.config["d"])
            rows.append(
                {
                    "framework": key[0],
                    "training_method": key[1],
                    "prediction_mode": key[2],
                    "mojogp_preset": key[3],
                    "baseline_backend": key[4],
                    "n": key[5],
                    "d1": d1,
                    "d2": d2,
                    "train_time_ratio": float(
                        larger.speed.training_time_s / smaller.speed.training_time_s
                    ),
                    "iter_time_ratio": float(
                        larger.speed.ms_per_iteration / smaller.speed.ms_per_iteration
                    ),
                    "prediction_total_ratio": float(
                        (larger.speed.prediction_mean_time_s + larger.speed.prediction_variance_time_s)
                        / (smaller.speed.prediction_mean_time_s + smaller.speed.prediction_variance_time_s)
                    ),
                    "prediction_apply_ratio": float(
                        _prediction_apply_time(larger) / _prediction_apply_time(smaller)
                    ),
                    "rmse_ratio": float(larger.accuracy.rmse / smaller.accuracy.rmse),
                }
            )
    return rows


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["_None_"]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---" for _ in columns]) + "|"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return [header, divider, *body]


def save_extensive_scaling_summary(
    results: list[BenchmarkResult],
    failures: list[dict[str, Any]],
    results_dir: Path,
    report_name: str,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"{timestamp}_{report_name}_summary.json"
    md_path = results_dir / f"{timestamp}_{report_name}_summary.md"

    sorted_results = sorted(
        results,
        key=lambda r: (
            str(r.config.get("framework")),
            str(r.config.get("training_method")),
            str(r.config.get("prediction_mode")),
            str(r.config.get("mojogp_preset")),
            int(r.config.get("n", 0)),
            int(r.config.get("d", 0)),
        ),
    )
    payload = {
        "report_name": report_name,
        "timestamp": datetime.now().isoformat(),
        "num_results": len(sorted_results),
        "num_failures": len(failures),
        "rows": [_result_row(r) for r in sorted_results],
        "n_scaling": _compute_n_scaling(sorted_results),
        "d_scaling": _compute_d_scaling(sorted_results),
        "failures": failures,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        f"# {report_name}",
        "",
        f"Generated: {payload['timestamp']}",
        "",
        f"Results: {payload['num_results']}",
        f"Failures: {payload['num_failures']}",
        "",
        "## Rows",
        "",
        *_markdown_table(
            payload["rows"],
            [
                "framework",
                "training_method",
                "prediction_mode",
                "mojogp_preset",
                "baseline_backend",
                "data_mean_offset",
                "data_noise_level",
                "n",
                "d",
                "training_time_s",
                "ms_per_iteration",
                "prediction_total_time_s",
                "prediction_apply_time_s",
                "prediction_cold_first_time_s",
                "prediction_cache_prepare_time_s",
                "prediction_prepared_apply_time_s",
                "prediction_repeated_median_time_s",
                "prediction_alpha_time_s",
                "prediction_love_root_time_s",
                "prediction_timing_quality",
                "gpu_max_mb",
                "training_delta_gpu_mb",
                "prediction_delta_gpu_mb",
                "exact_prediction_delta_gpu_mb",
                "love_prediction_delta_gpu_mb",
                "rmse",
                "r_squared",
                "learned_noise",
                "noise_rel_error",
                "learned_mean",
                "mean_rel_error",
            ],
        ),
        "",
        "## n Scaling",
        "",
        *_markdown_table(
            payload["n_scaling"],
            [
                "framework",
                "training_method",
                "prediction_mode",
                "mojogp_preset",
                "baseline_backend",
                "data_mean_offset",
                "data_noise_level",
                "d",
                "n1",
                "n2",
                "n_growth_ratio",
                "train_time_ratio",
                "iter_time_ratio",
                "prediction_total_ratio",
                "prediction_apply_ratio",
                "rmse_ratio",
            ],
        ),
        "",
        "## d Scaling",
        "",
        *_markdown_table(
            payload["d_scaling"],
            [
                "framework",
                "training_method",
                "prediction_mode",
                "mojogp_preset",
                "baseline_backend",
                "data_mean_offset",
                "data_noise_level",
                "n",
                "d1",
                "d2",
                "train_time_ratio",
                "iter_time_ratio",
                "prediction_total_ratio",
                "prediction_apply_ratio",
                "rmse_ratio",
            ],
        ),
        "",
        "## Failures",
        "",
        *_markdown_table(
            payload["failures"],
            [
                "framework",
                "training_method",
                "prediction_mode",
                "mojogp_preset",
                "n",
                "d",
                "error_type",
                "error",
            ],
        ),
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path
