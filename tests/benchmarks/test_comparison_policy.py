from __future__ import annotations

from tests.benchmarks.comparison_policy import all_policy_names, policy_for


def test_policy_registry_covers_active_benchmark_families():
    expected = {
        "single_output_ground_truth_active",
        "matrix_free_ground_truth_active",
        "scaling_certification",
        "single_output_ard_scaling",
        "single_output_extensive_scaling_materialized",
        "single_output_extensive_scaling_matrix_free",
        "single_output_trajectory",
        "single_output_mean_noise_ablation_materialized",
        "single_output_mean_noise_ablation_matrix_free",
        "love_variance_comparison",
        "multi_output_scaling",
        "multi_output_ard_scaling",
        "extended_feature_scaling",
        "multi_output_accuracy",
        "multi_output_per_task_noise",
        "lmc_ablation_accuracy",
        "mixed_accuracy",
        "multi_output_real_data",
        "single_output_persistence_harness",
        "single_output_sampling_harness",
        "multi_output_persistence_harness",
        "multi_output_sampling_harness",
        "lmc_persistence_harness",
        "lmc_sampling_harness",
        "single_output_route_parity",
        "multi_output_route_parity",
    }

    assert expected <= all_policy_names()


def test_matrix_free_truth_policy_is_not_cross_framework():
    policy = policy_for("matrix_free_ground_truth_active")

    assert policy.published_cross_framework is False
    assert policy.comparator_type == "intra_mojogp"


def test_multi_output_scaling_policy_keeps_only_materialized_cross_framework_rows():
    policy = policy_for("multi_output_scaling")

    assert policy.published_cross_framework is True
    assert policy.comparator_type == "mixed_by_lane"
    assert policy.strict_keops_required is False


def test_single_output_matrix_free_scaling_policy_requires_strict_keops():
    policy = policy_for("single_output_extensive_scaling_matrix_free")

    assert policy.published_cross_framework is True
    assert policy.strict_keops_required is True
    assert policy.fallback_allowed is False
