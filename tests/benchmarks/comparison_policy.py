"""Authoritative comparison policy for active benchmark families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkComparisonPolicy:
    benchmark_name: str
    comparator_type: str
    published_cross_framework: bool
    strict_keops_required: bool
    fallback_allowed: bool
    notes: str


COMPARISON_POLICIES: dict[str, BenchmarkComparisonPolicy] = {
    "single_output_ground_truth_active": BenchmarkComparisonPolicy(
        benchmark_name="single_output_ground_truth_active",
        comparator_type="gpytorch_cg_materialized_fair",
        published_cross_framework=True,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Published fair materialized ExactGP comparison against GPyTorch CG.",
    ),
    "matrix_free_ground_truth_active": BenchmarkComparisonPolicy(
        benchmark_name="matrix_free_ground_truth_active",
        comparator_type="intra_mojogp",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="MojoGP-only matrix-free versus materialized truth recovery surface.",
    ),
    "scaling_certification": BenchmarkComparisonPolicy(
        benchmark_name="scaling_certification",
        comparator_type="mixed_by_lane",
        published_cross_framework=True,
        strict_keops_required=True,
        fallback_allowed=False,
        notes="Materialized rows may compare to GPyTorch CG; matrix-free rows require strict KeOps or become MojoGP-only.",
    ),
    "single_output_ard_scaling": BenchmarkComparisonPolicy(
        benchmark_name="single_output_ard_scaling",
        comparator_type="mixed_by_lane",
        published_cross_framework=True,
        strict_keops_required=True,
        fallback_allowed=False,
        notes="Single-output RBF ARD scaling: materialized rows compare to GPyTorch CG with strict aligned all-ones ARD initialization; matrix-free rows require strict KeOps or become MojoGP-only.",
    ),
    "single_output_extensive_scaling_materialized": BenchmarkComparisonPolicy(
        benchmark_name="single_output_extensive_scaling_materialized",
        comparator_type="gpytorch_cg_materialized_fair",
        published_cross_framework=True,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Published fair materialized scaling comparison against GPyTorch CG.",
    ),
    "single_output_extensive_scaling_matrix_free": BenchmarkComparisonPolicy(
        benchmark_name="single_output_extensive_scaling_matrix_free",
        comparator_type="gpytorch_keops_strict",
        published_cross_framework=True,
        strict_keops_required=True,
        fallback_allowed=False,
        notes="Published matrix-free scaling rows require strict end-to-end GPyTorch+KeOps.",
    ),
    "single_output_trajectory": BenchmarkComparisonPolicy(
        benchmark_name="single_output_trajectory",
        comparator_type="mixed_by_lane",
        published_cross_framework=True,
        strict_keops_required=True,
        fallback_allowed=False,
        notes="Materialized checkpoints may compare cross-framework; matrix-free checkpoints require strict KeOps or become MojoGP-only.",
    ),
    "single_output_mean_noise_ablation_materialized": BenchmarkComparisonPolicy(
        benchmark_name="single_output_mean_noise_ablation_materialized",
        comparator_type="gpytorch_cg_materialized_fair",
        published_cross_framework=True,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Materialized ablation rows may compare to GPyTorch CG but should record metrics rather than benchmark correctness verdicts.",
    ),
    "single_output_mean_noise_ablation_matrix_free": BenchmarkComparisonPolicy(
        benchmark_name="single_output_mean_noise_ablation_matrix_free",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=True,
        fallback_allowed=False,
        notes="Matrix-free mean/noise ablation rows remain MojoGP-only unless strict KeOps is proven.",
    ),
    "love_variance_comparison": BenchmarkComparisonPolicy(
        benchmark_name="love_variance_comparison",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="LOVE comparison is MojoGP-only exact-vs-LOVE telemetry; no published GPyTorch fallback row.",
    ),
    "multi_output_scaling": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_scaling",
        comparator_type="mixed_by_lane",
        published_cross_framework=True,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Materialized rows compare to GPyTorch CG; matrix-free rows remain MojoGP-only.",
    ),
    "multi_output_ard_scaling": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_ard_scaling",
        comparator_type="mixed_by_lane",
        published_cross_framework=True,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Multi-output RBF ARD scaling: materialized rows compare against GPyTorch multitask ARD with solver telemetry labeled conservatively; matrix-free rows remain MojoGP-only.",
    ),
    "extended_feature_scaling": BenchmarkComparisonPolicy(
        benchmark_name="extended_feature_scaling",
        comparator_type="mojogp_only_feature_scaling",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Extended feature scaling covers supported MojoGP-only non-zero mean, per-task noise, mixed categorical, and LMC routes without publishing cross-framework speedup claims.",
    ),
    "multi_output_accuracy": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_accuracy",
        comparator_type="mojogp_independent_baseline",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Synthetic multi-output accuracy uses the in-repo independent ExactGP baseline, not GPyTorch.",
    ),
    "multi_output_per_task_noise": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_per_task_noise",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Per-task-noise benchmark is MojoGP-only synthetic truth recovery.",
    ),
    "lmc_ablation_accuracy": BenchmarkComparisonPolicy(
        benchmark_name="lmc_ablation_accuracy",
        comparator_type="mojogp_icm_baseline",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="LMC benchmark compares against the in-repo ICM baseline.",
    ),
    "mixed_accuracy": BenchmarkComparisonPolicy(
        benchmark_name="mixed_accuracy",
        comparator_type="simple_baseline_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Mixed-kernel benchmark compares against the in-repo continuous-only baseline.",
    ),
    "multi_output_real_data": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_real_data",
        comparator_type="simple_baseline_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Real-data multi-output benchmark uses simple or in-repo baselines, not GPyTorch.",
    ),
    "single_output_persistence_harness": BenchmarkComparisonPolicy(
        benchmark_name="single_output_persistence_harness",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Single-output workflow persistence telemetry is MojoGP-only.",
    ),
    "single_output_sampling_harness": BenchmarkComparisonPolicy(
        benchmark_name="single_output_sampling_harness",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Single-output workflow sampling telemetry is MojoGP-only.",
    ),
    "multi_output_persistence_harness": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_persistence_harness",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Multi-output workflow persistence telemetry is MojoGP-only.",
    ),
    "multi_output_sampling_harness": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_sampling_harness",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Multi-output workflow sampling telemetry is MojoGP-only.",
    ),
    "lmc_persistence_harness": BenchmarkComparisonPolicy(
        benchmark_name="lmc_persistence_harness",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="LMC workflow persistence telemetry is MojoGP-only.",
    ),
    "lmc_sampling_harness": BenchmarkComparisonPolicy(
        benchmark_name="lmc_sampling_harness",
        comparator_type="mojogp_only",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="LMC workflow sampling telemetry is MojoGP-only.",
    ),
    "multi_output_route_parity": BenchmarkComparisonPolicy(
        benchmark_name="multi_output_route_parity",
        comparator_type="intra_mojogp",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Multi-output route parity is intra-MojoGP only.",
    ),
    "single_output_route_parity": BenchmarkComparisonPolicy(
        benchmark_name="single_output_route_parity",
        comparator_type="intra_mojogp",
        published_cross_framework=False,
        strict_keops_required=False,
        fallback_allowed=False,
        notes="Single-output route parity is intra-MojoGP only.",
    ),
}


def policy_for(benchmark_name: str) -> BenchmarkComparisonPolicy:
    return COMPARISON_POLICIES[benchmark_name]


def all_policy_names() -> set[str]:
    return set(COMPARISON_POLICIES)
