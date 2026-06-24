"""Run one single-output ground-truth comparison benchmark case in isolation."""

from __future__ import annotations

from pathlib import Path

from tests.benchmarks.comparison_policy import policy_for
from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.shared.subprocess_harness import run_child_main

from tests.shared.benchmarking.data_generators import generate_gp_prior_data
from tests.shared.benchmarking.gpytorch_models import run_gpytorch_benchmark
from tests.shared.benchmarking.mojogp_runners import run_mojogp_benchmark
from tests.shared.benchmarking.report import save_comparison_artifact
from tests.shared.benchmarking.result_types import ComparisonResult


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _annotate_truth_result(result, *, framework: str, training_method: str, benchmark: str) -> None:
    policy = policy_for(benchmark)
    fairness_note = (
        "N.B. Fair cross-framework ground-truth row: both frameworks train ExactGP models on the "
        "same GP-prior sample with matched CG settings, and accuracy is evaluated against the known "
        "noiseless test function."
        if benchmark == "single_output_ground_truth_active"
        else "N.B. MojoGP-only matrix-free ground-truth row: matrix-free and materialized MojoGP routes are evaluated "
        "against the same GP-prior ground truth without a published external cross-framework comparator."
    )
    result.config.update(
        {
            "benchmark": benchmark,
            "route_group": "single_output",
            "framework": framework,
            "model_type": "SingleOutputGP",
            "training_method": training_method,
            "method": training_method,
            "prediction_mode": "exact",
            "comparison_class": (
                "fair_match"
                if policy.published_cross_framework
                else policy.comparator_type
            ),
            "baseline_backend": "gpytorch_cg" if policy.published_cross_framework else "none",
            "fairness_note": fairness_note,
            "fairness_axes": {
                "comparator_scope": _fairness_axis(
                    "fair_match" if policy.published_cross_framework else "mojogp_only",
                    "MojoGP and GPyTorch both train ExactGP models on the same synthetic GP-prior dataset."
                    if policy.published_cross_framework
                    else "This benchmark publishes only MojoGP rows and compares matrix-free against materialized on the same synthetic GP-prior dataset.",
                ),
                "sample_count_n": _fairness_axis(
                    "aligned",
                    "All compared rows use the same train/test split and the same known noiseless test function.",
                ),
                "optimizer": _fairness_axis(
                    "aligned",
                    "All compared rows use Adam with the same learning rate and iteration budget.",
                ),
                "solver_budget": _fairness_axis(
                    "aligned",
                    "The compared rows use CG-based ExactGP training with the same tolerance and iteration cap.",
                ),
                "preconditioner": _fairness_axis(
                    "aligned",
                    "Benchmark policy disables preconditioning on this lane.",
                ),
                "prediction_mode": _fairness_axis(
                    "aligned",
                    "All compared rows report exact predictive variance against the same noiseless ground truth.",
                ),
                "telemetry": _fairness_axis(
                    "observed",
                    "MojoGP and GPyTorch benchmark telemetry are both captured for the comparison rows.",
                ),
            },
        }
    )


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    benchmark = str(payload["benchmark"])
    kernel = str(payload["kernel"])
    n = int(payload["n"])
    d = int(payload["d"])
    results_dir = Path(str(payload["results_dir"]))

    dataset = generate_gp_prior_data(
        n_train=n,
        n_test=BENCHMARK_PREDICTION_N_TEST,
        d=d,
        kernel_type=kernel,
        true_lengthscale=1.0,
        true_noise=0.1,
        true_outputscale=1.0,
        seed=42,
    )

    init_ls = 1.0
    init_noise = 0.1
    init_os = 1.0
    n_iterations = 40
    lr = 0.03

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
    gpytorch_cg = None
    if policy_for(benchmark).published_cross_framework:
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

    _annotate_truth_result(mojogp_mat, framework="mojogp", training_method="materialized", benchmark=benchmark)
    _annotate_truth_result(mojogp_mf, framework="mojogp", training_method="matrix_free", benchmark=benchmark)
    if gpytorch_cg is not None:
        _annotate_truth_result(gpytorch_cg, framework="gpytorch", training_method="cg", benchmark=benchmark)

    config = {
        "benchmark": benchmark,
        "route_group": "single_output",
        "kernel": kernel,
        "n": n,
        "d": d,
        "n_test": BENCHMARK_PREDICTION_N_TEST,
        "framework": "cross_framework" if policy_for(benchmark).published_cross_framework else "mojogp",
        "model_type": "SingleOutputGP",
        "baseline_backend": "gpytorch_cg" if policy_for(benchmark).published_cross_framework else "none",
        "comparison_class": "fair_match" if policy_for(benchmark).published_cross_framework else "intra_mojogp",
        "fairness_note": (
            "N.B. Fair ground-truth row: all models are judged against the same GP-prior ground truth, not just against each other."
            if benchmark == "single_output_ground_truth_active"
            else "N.B. Matrix-free ground-truth row: this surface is a MojoGP-only route check that matrix-free stays close to materialized truth recovery against the same synthetic truth."
        ),
        "fairness_axes": {
            "comparator_scope": _fairness_axis(
                "mojogp_only" if benchmark == "matrix_free_ground_truth_active" else "fair_match",
                "This benchmark only publishes MojoGP matrix-free versus materialized rows on the same dataset."
                if benchmark == "matrix_free_ground_truth_active"
                else "MojoGP and GPyTorch both train ExactGP models on the same synthetic GP-prior dataset.",
            ),
            "sample_count_n": _fairness_axis("aligned", "All rows use the same GP-prior train/test split and the same noiseless targets."),
            "optimizer": _fairness_axis("aligned", "All rows use Adam with the same learning rate and iteration budget."),
            "solver_budget": _fairness_axis("aligned", "All rows use matched CG-based ExactGP training settings."),
            "preconditioner": _fairness_axis("aligned", "Benchmark policy disables preconditioning on this lane."),
            "prediction_mode": _fairness_axis("aligned", "All rows report exact predictive variance against the same noiseless ground truth."),
            "telemetry": _fairness_axis(
                "observed",
                "MojoGP telemetry is captured for the route rows and GPyTorch telemetry is captured only on published cross-framework lanes."
            ),
        },
    }
    if benchmark == "matrix_free_ground_truth_active":
        rmse_gap = abs(mojogp_mf.accuracy.rmse - mojogp_mat.accuracy.rmse)
        mat_rmse_scale = max(mojogp_mat.accuracy.rmse, 1e-6)
        mat_mem = mojogp_mat.memory.gpu_max_mb
        mf_mem = mojogp_mf.memory.gpu_max_mb
        memory_ratio_mat_over_mf = None if not (mat_mem > 0 and mf_mem > 0) else mat_mem / mf_mem
        config.update(
            {
                "matrix_free_vs_materialized_rmse_ratio": rmse_gap / mat_rmse_scale,
                "materialized_over_matrix_free_gpu_peak_ratio": memory_ratio_mat_over_mf,
            }
        )

    comparison = ComparisonResult(
        config=config,
        mojogp_materialized=mojogp_mat,
        mojogp_matrix_free=mojogp_mf,
        gpytorch_cg=gpytorch_cg,
    )
    comparison.compute_comparisons()
    result_path = save_comparison_artifact(comparison, results_dir, benchmark)
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
