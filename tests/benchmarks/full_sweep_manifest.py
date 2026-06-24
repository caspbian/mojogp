"""Authoritative benchmark manifest for readiness and full-sweep runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSuiteEntry:
    path: str
    category: str
    tier_policy: str
    requires_gpytorch: bool
    readiness_probe: bool
    notes: str


FULL_SWEEP_MANIFEST: tuple[BenchmarkSuiteEntry, ...] = (
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_truth_harness.py",
        category="single_output_truth",
        tier_policy="minimal_only",
        requires_gpytorch=True,
        readiness_probe=False,
        notes="Truth-backed ExactGP CG vs GPyTorch CG comparison.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_matrix_free_truth_harness.py",
        category="single_output_truth",
        tier_policy="minimal_only",
        requires_gpytorch=True,
        readiness_probe=False,
        notes="Matrix-free truth-backed comparison surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_variance_modes_harness.py",
        category="single_output_variance",
        tier_policy="minimal_only",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="MojoGP exact vs LOVE plus GPyTorch-backed reasonableness check.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_mojogp_route_parity_harness.py",
        category="route_parity",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=False,
        notes="Internal MojoGP route parity benchmark surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_mixed_kernel_harness.py",
        category="mixed_kernel",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=False,
        notes="Mixed categorical benchmark surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_lmc_accuracy_harness.py",
        category="lmc_accuracy",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=False,
        notes="LMC accuracy ablation benchmark surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_multi_output_accuracy_harness.py",
        category="multi_output_accuracy",
        tier_policy="minimal_moderate_full",
        requires_gpytorch=False,
        readiness_probe=False,
        notes="Primary multi-output accuracy benchmark surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_multi_output_per_task_noise_harness.py",
        category="multi_output_noise",
        tier_policy="minimal_moderate",
        requires_gpytorch=False,
        readiness_probe=False,
        notes="Per-task noise benchmark surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_scaling_certification_harness.py",
        category="single_output_scaling",
        tier_policy="minimal_only",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="Certification benchmark; imported by several extended single-output suites.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_extensive_scaling_harness.py",
        category="single_output_scaling",
        tier_policy="full_only",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="Broad scaling and preset sweep surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_ard_scaling_harness.py",
        category="single_output_scaling",
        tier_policy="full_only",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="Single-output RBF(ARD) scaling and relevance-recovery surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_trajectory_harness.py",
        category="single_output_scaling",
        tier_policy="minimal_only",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="Trajectory checkpoint benchmark surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_mean_noise_ablation_harness.py",
        category="single_output_scaling",
        tier_policy="full_only",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="Mean/noise recovery ablation surface.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_multi_output_scaling_harness.py",
        category="multi_output_scaling",
        tier_policy="unmarked_explicit_manifest",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="Unmarked scaling surface; must be included by explicit manifest, not marker-only commands.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_multi_output_ard_scaling_harness.py",
        category="multi_output_scaling",
        tier_policy="unmarked_explicit_manifest",
        requires_gpytorch=True,
        readiness_probe=True,
        notes="Multi-output RBF(ARD) scaling surface with conservative GPyTorch fairness labeling.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_extended_feature_scaling_harness.py",
        category="extended_feature_scaling",
        tier_policy="unmarked_explicit_manifest",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="MojoGP-only scaling surface for non-zero mean, per-task noise, mixed categorical, and LMC routes.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_persistence_harness.py",
        category="workflow_single_output",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="Single-output save/load workflow benchmark.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_single_output_sampling_harness.py",
        category="workflow_single_output",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="Single-output posterior sampling workflow benchmark.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_multi_output_persistence_harness.py",
        category="workflow_multi_output",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="Multi-output save/load workflow benchmark.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_multi_output_sampling_harness.py",
        category="workflow_multi_output",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="Multi-output posterior sampling workflow benchmark.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_lmc_persistence_harness.py",
        category="workflow_lmc",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="LMC save/load workflow benchmark.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_lmc_sampling_harness.py",
        category="workflow_lmc",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="LMC posterior sampling workflow benchmark.",
    ),
    BenchmarkSuiteEntry(
        path="tests/system_benchmarks/test_multi_output_real_data_harness.py",
        category="workflow_multi_output",
        tier_policy="minimal_only",
        requires_gpytorch=False,
        readiness_probe=True,
        notes="Real-data multi-output benchmark surface.",
    ),
)


def full_sweep_entries() -> tuple[BenchmarkSuiteEntry, ...]:
    return FULL_SWEEP_MANIFEST


def full_sweep_paths() -> list[str]:
    return [entry.path for entry in FULL_SWEEP_MANIFEST]


def readiness_probe_entries() -> tuple[BenchmarkSuiteEntry, ...]:
    return tuple(entry for entry in FULL_SWEEP_MANIFEST if entry.readiness_probe)


def readiness_probe_paths() -> list[str]:
    return [entry.path for entry in readiness_probe_entries()]


def manifest_requires_gpytorch(paths: list[str]) -> bool:
    selected = {entry.path for entry in FULL_SWEEP_MANIFEST if entry.requires_gpytorch}
    return any(path in selected for path in paths)
