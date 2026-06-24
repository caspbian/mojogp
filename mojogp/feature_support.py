"""Central feature-support registry for MojoGP.

This module is the single source of truth for public feature maturity labels,
runtime warnings/errors, and generated feature-matrix documentation. Wrapper
code should call ``check_feature_support`` or ``warn_surface_status`` instead of
hard-coding user-facing maturity messages.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Iterable, Mapping
import warnings


class FeatureStatus(str, Enum):
    """Public maturity/status markers used in generated feature tables."""

    NOT_STARTED = "--"
    IN_DEV = "in-dev"
    EXP = "exp"
    ALPHA = "alpha"
    BETA = "beta"
    UNSUPPORTED = "unsupported"
    NA = "n/a"


class MojoGPFeatureWarning(UserWarning):
    """Base class for feature-maturity warnings emitted by MojoGP."""


class ExperimentalFeatureWarning(MojoGPFeatureWarning):
    """Warning emitted when an experimental feature surface is used."""


class InDevelopmentFeatureWarning(MojoGPFeatureWarning):
    """Warning emitted when an in-development feature surface is used."""


_FEATURE_WARNINGS_ENABLED = True


def get_feature_warnings_enabled() -> bool:
    """Return whether MojoGP feature-maturity warnings are enabled."""

    return _FEATURE_WARNINGS_ENABLED


def set_feature_warnings_enabled(enabled: bool) -> None:
    """Enable or disable MojoGP feature-maturity warnings globally."""

    global _FEATURE_WARNINGS_ENABLED
    _FEATURE_WARNINGS_ENABLED = bool(enabled)


@contextmanager
def feature_warnings_suppressed() -> Iterator[None]:
    """Temporarily suppress MojoGP feature-maturity warnings."""

    previous = get_feature_warnings_enabled()
    set_feature_warnings_enabled(False)
    try:
        yield
    finally:
        set_feature_warnings_enabled(previous)


def _emit_feature_warning(
    message: str,
    category: type[MojoGPFeatureWarning],
    stacklevel: int,
) -> None:
    if _FEATURE_WARNINGS_ENABLED:
        warnings.warn(message, category, stacklevel=stacklevel + 1)


@dataclass(frozen=True)
class SurfaceDef:
    key: str
    label: str
    definition: str


@dataclass(frozen=True)
class MatrixRow:
    family: str
    key: str
    label: str


@dataclass(frozen=True)
class FeatureEntry:
    status: FeatureStatus
    message: str = ""
    scope: str = ""


@dataclass(frozen=True)
class KernelPrimitive:
    family: str
    primitive: str
    single_output: FeatureStatus
    icm: FeatureStatus
    lmc: FeatureStatus
    notes: str


@dataclass(frozen=True)
class BoundaryEntry:
    combination: str
    category: str
    runtime_behavior: str


TABLE_MAIN = "main"
TABLE_EXECUTION = "execution"
TABLE_PREDICTION = "prediction"
TABLE_SAMPLING = "sampling"

SURFACE_SINGLE_CONTINUOUS = "single_output_continuous"
SURFACE_SINGLE_MIXED = "single_output_mixed"
SURFACE_ICM_CONTINUOUS = "icm_continuous"
SURFACE_ICM_MIXED = "icm_mixed"
SURFACE_LMC_CONTINUOUS = "lmc_continuous"
SURFACE_LMC_MIXED = "lmc_mixed"

SURFACES: tuple[SurfaceDef, ...] = (
    SurfaceDef(
        SURFACE_SINGLE_CONTINUOUS,
        "SingleOutput: Continuous",
        "`SingleOutputGP` with continuous-only kernel trees",
    ),
    SurfaceDef(
        SURFACE_SINGLE_MIXED,
        "SingleOutput: Mixed",
        "`SingleOutputGP` with continuous plus categorical kernel leaves",
    ),
    SurfaceDef(
        SURFACE_ICM_CONTINUOUS,
        "ICM: Continuous",
        "`MultiOutputGP` with continuous-only kernel trees",
    ),
    SurfaceDef(
        SURFACE_ICM_MIXED,
        "ICM: Mixed",
        "`MultiOutputGP` with continuous plus categorical kernel leaves",
    ),
    SurfaceDef(
        SURFACE_LMC_CONTINUOUS,
        "LMC: Continuous",
        "`MultiOutputLMCGP` where all latent kernels are continuous-only",
    ),
    SurfaceDef(
        SURFACE_LMC_MIXED,
        "LMC: Mixed",
        "`MultiOutputLMCGP` where at least one latent is mixed continuous-categorical",
    ),
)

SURFACE_LABELS: dict[str, str] = {surface.key: surface.label for surface in SURFACES}
SURFACE_KEYS: tuple[str, ...] = tuple(surface.key for surface in SURFACES)

SURFACE_STATUS: dict[str, FeatureEntry] = {
    SURFACE_SINGLE_CONTINUOUS: FeatureEntry(
        FeatureStatus.ALPHA,
        scope="Core continuous single-output exact GP surface.",
    ),
    SURFACE_SINGLE_MIXED: FeatureEntry(
        FeatureStatus.EXP,
        scope="Mixed continuous-categorical single-output support is public on a narrow tested scope.",
    ),
    SURFACE_ICM_CONTINUOUS: FeatureEntry(
        FeatureStatus.EXP,
        scope="ICM support is implemented but has thinner scaling/evidence depth than the single-output core.",
    ),
    SURFACE_ICM_MIXED: FeatureEntry(
        FeatureStatus.EXP,
        scope="Mixed ICM support is implemented on targeted mixed continuous-categorical routes.",
    ),
    SURFACE_LMC_CONTINUOUS: FeatureEntry(
        FeatureStatus.ALPHA,
        scope="Continuous LMC routes have the strongest multi-output LMC evidence.",
    ),
    SURFACE_LMC_MIXED: FeatureEntry(
        FeatureStatus.EXP,
        scope="Mixed LMC support is implemented on targeted mixed latent routes.",
    ),
}


def _entry(status: FeatureStatus | str, scope: str = "", message: str = "") -> FeatureEntry:
    return FeatureEntry(FeatureStatus(status), message=message, scope=scope)


def _row(statuses: Mapping[str, FeatureStatus | str]) -> dict[str, FeatureEntry]:
    missing = set(SURFACE_KEYS) - set(statuses)
    if missing:
        raise ValueError(f"feature row missing surfaces: {sorted(missing)}")
    return {surface: _entry(status) for surface, status in statuses.items()}


MAIN_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow("Model Boundaries", "pure_categorical", "Pure categorical model / latent"),
    MatrixRow("Model Structure", "heterogeneous_latents", "Heterogeneous latent kernels"),
    MatrixRow("Dimensionality Controls", "active_dims", "Active dimensions"),
    MatrixRow("Parameterization", "isotropic", "Isotropic lengthscales"),
    MatrixRow("Parameterization", "ard", "ARD lengthscales"),
    MatrixRow("Kernel Algebra", "additive_composites", "Additive composites"),
    MatrixRow("Kernel Algebra", "product_composites", "Product composites"),
    MatrixRow("Kernel Algebra", "nested_composites", "Nested composites"),
    MatrixRow("Kernel Algebra", "multiple_categorical_leaves", "Multiple categorical leaves"),
    MatrixRow("Kernel Algebra", "scaled_categorical_tree", "Scaled categorical-containing trees"),
    MatrixRow("Mean / Likelihood", "mean_offset", "Constant / learned mean offset"),
    MatrixRow("Noise / Likelihood", "learned_scalar_noise", "Learned scalar homoskedastic noise"),
    MatrixRow("Noise / Likelihood", "learned_per_task_noise", "Learned per-task homoskedastic noise `[T]`"),
    MatrixRow("Noise / Likelihood", "fixed_per_sample_noise", "Fixed per-sample noise `[n]`"),
    MatrixRow("Noise / Likelihood", "fixed_per_sample_per_task_noise", "Fixed per-sample-per-task noise `[n, T]`"),
    MatrixRow("Noise / Likelihood", "learned_input_dependent_noise", "Learned input-dependent heteroskedasticity"),
    MatrixRow("Noise / Likelihood", "grouped_noise", "Grouped noise"),
    MatrixRow("Lifecycle", "save_load", "Save / load"),
    MatrixRow("Lifecycle", "route_metadata", "Route metadata"),
)

MAIN_MATRIX: dict[str, dict[str, FeatureEntry]] = {
    "pure_categorical": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_SINGLE_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_ICM_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_LMC_MIXED: FeatureStatus.UNSUPPORTED,
    }),
    "heterogeneous_latents": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.NA,
        SURFACE_SINGLE_MIXED: FeatureStatus.NA,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NA,
        SURFACE_ICM_MIXED: FeatureStatus.NA,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "active_dims": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "isotropic": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "ard": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.IN_DEV,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.IN_DEV,
    }),
    "additive_composites": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.IN_DEV,
    }),
    "product_composites": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "nested_composites": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_LMC_MIXED: FeatureStatus.IN_DEV,
    }),
    "multiple_categorical_leaves": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.NA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NA,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.NA,
        SURFACE_LMC_MIXED: FeatureStatus.IN_DEV,
    }),
    "scaled_categorical_tree": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.NA,
        SURFACE_SINGLE_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NA,
        SURFACE_ICM_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.NA,
        SURFACE_LMC_MIXED: FeatureStatus.UNSUPPORTED,
    }),
    "mean_offset": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "learned_scalar_noise": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NA,
        SURFACE_ICM_MIXED: FeatureStatus.NA,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.NA,
        SURFACE_LMC_MIXED: FeatureStatus.NA,
    }),
    "learned_per_task_noise": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.NA,
        SURFACE_SINGLE_MIXED: FeatureStatus.NA,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "fixed_per_sample_noise": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.IN_DEV,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NA,
        SURFACE_ICM_MIXED: FeatureStatus.NA,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.NA,
        SURFACE_LMC_MIXED: FeatureStatus.NA,
    }),
    "fixed_per_sample_per_task_noise": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.NA,
        SURFACE_SINGLE_MIXED: FeatureStatus.NA,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_ICM_MIXED: FeatureStatus.IN_DEV,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.IN_DEV,
    }),
    "learned_input_dependent_noise": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.IN_DEV,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NOT_STARTED,
        SURFACE_ICM_MIXED: FeatureStatus.NOT_STARTED,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.NOT_STARTED,
        SURFACE_LMC_MIXED: FeatureStatus.NOT_STARTED,
    }),
    "grouped_noise": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.IN_DEV,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_ICM_MIXED: FeatureStatus.IN_DEV,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_LMC_MIXED: FeatureStatus.UNSUPPORTED,
    }),
    "save_load": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "route_metadata": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
}

EXECUTION_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow("Execution", "materialized_training", "Materialized Training"),
    MatrixRow("Execution", "matrix_free_training", "Matrix-Free Training"),
    MatrixRow("Execution", "auto_selection", "Auto Selection"),
)

EXECUTION_MATRIX: dict[str, dict[str, FeatureEntry]] = {
    "materialized_training": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "matrix_free_training": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "auto_selection": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NA,
        SURFACE_ICM_MIXED: FeatureStatus.NA,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.NA,
        SURFACE_LMC_MIXED: FeatureStatus.NA,
    }),
}

PREDICTION_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow("Prediction", "mean_only", "Mean-Only Prediction"),
    MatrixRow("Prediction", "exact_variance", "Exact Variance"),
    MatrixRow("Prediction", "love_variance", "LOVE Variance"),
    MatrixRow("Prediction", "prediction_cache", "Prediction Cache"),
)

PREDICTION_MATRIX: dict[str, dict[str, FeatureEntry]] = {
    "mean_only": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "exact_variance": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "love_variance": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "prediction_cache": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.NA,
        SURFACE_ICM_MIXED: FeatureStatus.NA,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.NA,
        SURFACE_LMC_MIXED: FeatureStatus.NA,
    }),
}

SAMPLING_ROWS: tuple[MatrixRow, ...] = (
    MatrixRow("Sampling", "diagonal_sampling", "Diagonal Sampling"),
    MatrixRow("Sampling", "pathwise_sampling", "Pathwise Sampling"),
    MatrixRow("Sampling", "polynomial_pathwise", "Polynomial Pathwise"),
    MatrixRow("Sampling", "cholesky_sampling", "Public Cholesky Sampling"),
)

SAMPLING_MATRIX: dict[str, dict[str, FeatureEntry]] = {
    "diagonal_sampling": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "pathwise_sampling": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.EXP,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.EXP,
        SURFACE_ICM_MIXED: FeatureStatus.EXP,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.EXP,
    }),
    "polynomial_pathwise": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_SINGLE_MIXED: FeatureStatus.IN_DEV,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_ICM_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.ALPHA,
        SURFACE_LMC_MIXED: FeatureStatus.IN_DEV,
    }),
    "cholesky_sampling": _row({
        SURFACE_SINGLE_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_SINGLE_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_ICM_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_ICM_MIXED: FeatureStatus.UNSUPPORTED,
        SURFACE_LMC_CONTINUOUS: FeatureStatus.UNSUPPORTED,
        SURFACE_LMC_MIXED: FeatureStatus.UNSUPPORTED,
    }),
}

MATRICES: dict[str, tuple[tuple[MatrixRow, ...], dict[str, dict[str, FeatureEntry]]]] = {
    TABLE_MAIN: (MAIN_ROWS, MAIN_MATRIX),
    TABLE_EXECUTION: (EXECUTION_ROWS, EXECUTION_MATRIX),
    TABLE_PREDICTION: (PREDICTION_ROWS, PREDICTION_MATRIX),
    TABLE_SAMPLING: (SAMPLING_ROWS, SAMPLING_MATRIX),
}

KERNEL_PRIMITIVES: tuple[KernelPrimitive, ...] = (
    KernelPrimitive("Continuous base kernel", "RBF", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.ALPHA, "Broadest support"),
    KernelPrimitive("Continuous base kernel", "Matern12", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.ALPHA, "Continuous and mixed continuous component"),
    KernelPrimitive("Continuous base kernel", "Matern32", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.ALPHA, "Continuous and mixed continuous component"),
    KernelPrimitive("Continuous base kernel", "Matern52", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.ALPHA, "Continuous and mixed continuous component"),
    KernelPrimitive("Continuous base kernel", "RQ", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.ALPHA, "Less evidence than RBF/Matern"),
    KernelPrimitive("Continuous base kernel", "Periodic", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.ALPHA, "Less evidence than RBF/Matern"),
    KernelPrimitive("Continuous base kernel", "Linear", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.ALPHA, "Dot-product kernel"),
    KernelPrimitive("Continuous base kernel", "Polynomial", FeatureStatus.ALPHA, FeatureStatus.EXP, FeatureStatus.EXP, "Fixed positive integer degree; exact feature-map pathwise within feature cap"),
    KernelPrimitive("Categorical correlation kernel", "GD", FeatureStatus.EXP, FeatureStatus.EXP, FeatureStatus.EXP, "Mixed surfaces only"),
    KernelPrimitive("Categorical correlation kernel", "CR", FeatureStatus.EXP, FeatureStatus.EXP, FeatureStatus.EXP, "Mixed surfaces only"),
    KernelPrimitive("Categorical correlation kernel", "EHH", FeatureStatus.EXP, FeatureStatus.EXP, FeatureStatus.EXP, "Mixed surfaces only"),
    KernelPrimitive("Categorical correlation kernel", "HH", FeatureStatus.EXP, FeatureStatus.EXP, FeatureStatus.EXP, "Mixed surfaces only"),
    KernelPrimitive("Categorical correlation kernel", "FE", FeatureStatus.EXP, FeatureStatus.EXP, FeatureStatus.EXP, "Mixed surfaces only"),
)

BOUNDARY_ENTRIES: tuple[BoundaryEntry, ...] = (
    BoundaryEntry("Pure categorical single-output model", "unsupported", "raise clear error"),
    BoundaryEntry("Pure categorical ICM model", "unsupported", "raise clear error"),
    BoundaryEntry("Pure categorical LMC latent", "unsupported", "raise clear error"),
    BoundaryEntry("Scaled tree containing categorical leaves", "unsupported", "raise clear error"),
    BoundaryEntry("SingleOutput fixed per-sample noise `[n]`", "alpha", "continuous route accepts `observation_noise`"),
    BoundaryEntry("ICM fixed per-sample-per-task noise `[n, T]`", "alpha", "continuous route accepts `observation_noise`"),
    BoundaryEntry("Mixed LMC fixed observation noise `[n, T]`", "in-dev", "raise `NotImplementedError` until evidenced"),
    BoundaryEntry("LMC fixed observation noise plus LOVE variance", "in-dev", "raise `NotImplementedError` until evidenced"),
    BoundaryEntry("SingleOutput learned input-dependent heteroskedasticity", "alpha", "continuous linear noise-function route is documented and tested"),
    BoundaryEntry("Mixed and multi-output learned input-dependent heteroskedasticity", "in-dev", "raise or warn until separately evidenced"),
    BoundaryEntry("Grouped noise", "alpha / in-dev / unsupported depending model", "run documented continuous routes and raise clear error where explicitly rejected"),
    BoundaryEntry("ICM polynomial pathwise sampling", "unsupported", "raise `NotImplementedError`"),
    BoundaryEntry("Public Cholesky posterior sampling", "unsupported", "reject as non-public API"),
    BoundaryEntry("Excessive product/pathwise feature expansion", "unsupported-current-scope", "raise `NotImplementedError` with mitigation"),
)

STATUS_LEGEND: tuple[tuple[str, str], ...] = (
    (FeatureStatus.NOT_STARTED.value, "not started / no implementation yet"),
    (FeatureStatus.IN_DEV.value, "implementation exists but is not public-ready"),
    (FeatureStatus.EXP.value, "experimental; implemented on a narrow tested scope"),
    (FeatureStatus.ALPHA.value, "built and fully tested on the documented scope"),
    (FeatureStatus.BETA.value, "scaling/accuracy benchmark validated for broader dev exposure"),
    ("released:<version>", "shipped in a numbered release"),
    (FeatureStatus.UNSUPPORTED.value, "intentionally rejected"),
    (FeatureStatus.NA.value, "not meaningful for that surface"),
)


def surface_for_single_output(is_mixed: bool) -> str:
    return SURFACE_SINGLE_MIXED if is_mixed else SURFACE_SINGLE_CONTINUOUS


def surface_for_icm(is_mixed: bool) -> str:
    return SURFACE_ICM_MIXED if is_mixed else SURFACE_ICM_CONTINUOUS


def surface_for_lmc(has_mixed_latents: bool) -> str:
    return SURFACE_LMC_MIXED if has_mixed_latents else SURFACE_LMC_CONTINUOUS


def get_feature_entry(table: str, surface: str, feature: str) -> FeatureEntry:
    try:
        _rows, matrix = MATRICES[table]
    except KeyError as exc:
        raise KeyError(f"unknown feature support table '{table}'") from exc
    try:
        return matrix[feature][surface]
    except KeyError as exc:
        raise KeyError(f"unknown feature support cell table={table} surface={surface} feature={feature}") from exc


def _feature_label(table: str, feature: str) -> str:
    rows, _matrix = MATRICES[table]
    for row_def in rows:
        if row_def.key == feature:
            return row_def.label
    return feature


def _warning_message(surface: str, feature_label: str, entry: FeatureEntry) -> str:
    surface_label = SURFACE_LABELS.get(surface, surface)
    base = (
        f"MojoGP feature '{feature_label}' on '{surface_label}' is {entry.status.value}."
    )
    if entry.scope:
        base += f" Scope: {entry.scope}"
    if entry.message:
        base += f" {entry.message}"
    return base


def _error_message(surface: str, feature_label: str, entry: FeatureEntry) -> str:
    surface_label = SURFACE_LABELS.get(surface, surface)
    if entry.message:
        return entry.message
    if entry.status == FeatureStatus.NA:
        return f"Feature '{feature_label}' is not meaningful for {surface_label}."
    if entry.status == FeatureStatus.UNSUPPORTED:
        return f"Feature '{feature_label}' is unsupported for {surface_label}."
    return f"Feature '{feature_label}' is not implemented for {surface_label}."


def check_feature_support(
    table: str,
    surface: str,
    feature: str,
    *,
    fail_on_in_dev: bool = False,
    stacklevel: int = 3,
) -> FeatureEntry:
    """Validate or warn for a feature usage on a public surface.

    ``exp`` and ``in-dev`` statuses emit maturity warnings. ``--``,
    ``unsupported``, and ``n/a`` raise because those should not silently run.
    Callers may set ``fail_on_in_dev`` for correctness-sensitive in-development
    routes that have code present but are not safe to expose.
    """

    entry = get_feature_entry(table, surface, feature)
    feature_label = _feature_label(table, feature)
    if entry.status in (
        FeatureStatus.NOT_STARTED,
        FeatureStatus.UNSUPPORTED,
        FeatureStatus.NA,
    ):
        raise NotImplementedError(_error_message(surface, feature_label, entry))
    if entry.status == FeatureStatus.IN_DEV:
        if fail_on_in_dev:
            raise NotImplementedError(_error_message(surface, feature_label, entry))
        _emit_feature_warning(
            _warning_message(surface, feature_label, entry),
            InDevelopmentFeatureWarning,
            stacklevel,
        )
    elif entry.status == FeatureStatus.EXP:
        _emit_feature_warning(
            _warning_message(surface, feature_label, entry),
            ExperimentalFeatureWarning,
            stacklevel,
        )
    return entry


def warn_surface_status(surface: str, *, stacklevel: int = 3) -> FeatureEntry:
    """Emit a maturity warning for a public model/input surface when needed."""

    entry = SURFACE_STATUS[surface]
    label = SURFACE_LABELS[surface]
    if entry.status == FeatureStatus.IN_DEV:
        _emit_feature_warning(
            _warning_message(surface, label, entry),
            InDevelopmentFeatureWarning,
            stacklevel,
        )
    elif entry.status == FeatureStatus.EXP:
        _emit_feature_warning(
            _warning_message(surface, label, entry),
            ExperimentalFeatureWarning,
            stacklevel,
        )
    return entry


def _kernel_operator(node: object) -> str | None:
    return getattr(node, "operator", None)


def kernel_tree_has_operator(node: object, operator: str) -> bool:
    if _kernel_operator(node) == operator:
        return True
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    return bool(
        (left is not None and kernel_tree_has_operator(left, operator))
        or (right is not None and kernel_tree_has_operator(right, operator))
    )


def kernel_tree_has_nested_composition(node: object) -> bool:
    operator = _kernel_operator(node)
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    if operator in ("sum", "product"):
        if _kernel_operator(left) in ("sum", "product"):
            return True
        if _kernel_operator(right) in ("sum", "product"):
            return True
    return bool(
        (left is not None and kernel_tree_has_nested_composition(left))
        or (right is not None and kernel_tree_has_nested_composition(right))
    )


def kernel_tree_categorical_leaf_count(node: object) -> int:
    is_categorical = getattr(node, "is_categorical", None)
    if callable(is_categorical) and is_categorical():
        return 1
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    return (
        (kernel_tree_categorical_leaf_count(left) if left is not None else 0)
        + (kernel_tree_categorical_leaf_count(right) if right is not None else 0)
    )


def kernel_tree_contains_kernel_name(node: object, kernel_name: str) -> bool:
    kernel_type = getattr(node, "kernel_type", None)
    if getattr(kernel_type, "name", None) == kernel_name:
        return True
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    return bool(
        (left is not None and kernel_tree_contains_kernel_name(left, kernel_name))
        or (right is not None and kernel_tree_contains_kernel_name(right, kernel_name))
    )


def guard_kernel_tree_features(surface: str, kernel: object, *, stacklevel: int = 3) -> None:
    """Warn/raise for kernel-tree capabilities used on a public surface."""

    has_active_dims = getattr(kernel, "has_active_dims", None)
    if callable(has_active_dims) and has_active_dims():
        check_feature_support(TABLE_MAIN, surface, "active_dims", stacklevel=stacklevel)
    if kernel_tree_has_operator(kernel, "sum"):
        check_feature_support(TABLE_MAIN, surface, "additive_composites", stacklevel=stacklevel)
    if kernel_tree_has_operator(kernel, "product"):
        check_feature_support(TABLE_MAIN, surface, "product_composites", stacklevel=stacklevel)
    if kernel_tree_has_nested_composition(kernel):
        check_feature_support(TABLE_MAIN, surface, "nested_composites", stacklevel=stacklevel)
    if kernel_tree_categorical_leaf_count(kernel) > 1:
        check_feature_support(TABLE_MAIN, surface, "multiple_categorical_leaves", stacklevel=stacklevel)


def _status(entry: FeatureEntry | FeatureStatus) -> str:
    if isinstance(entry, FeatureEntry):
        return entry.status.value
    return entry.value


def _markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _render_surface_definitions() -> str:
    return _markdown_table(
        ["Surface", "Definition"],
        ((surface.label, surface.definition) for surface in SURFACES),
    )


def _render_status_legend() -> str:
    return _markdown_table(["Marker", "Meaning"], STATUS_LEGEND)


def _render_matrix(rows: tuple[MatrixRow, ...], matrix: dict[str, dict[str, FeatureEntry]]) -> str:
    headers = ["Feature Family", "Feature"] + [surface.label for surface in SURFACES]
    rendered_rows = []
    for row_def in rows:
        entries = matrix[row_def.key]
        rendered_rows.append(
            [row_def.family, row_def.label]
            + [_status(entries[surface.key]) for surface in SURFACES]
        )
    return _markdown_table(headers, rendered_rows)


def _render_execution_matrix() -> str:
    headers = ["Surface", "Materialized Training", "Matrix-Free Training", "Auto Selection"]
    rendered_rows = []
    for surface in SURFACES:
        rendered_rows.append(
            [
                surface.label,
                _status(EXECUTION_MATRIX["materialized_training"][surface.key]),
                _status(EXECUTION_MATRIX["matrix_free_training"][surface.key]),
                _status(EXECUTION_MATRIX["auto_selection"][surface.key]),
            ]
        )
    return _markdown_table(headers, rendered_rows)


def _render_prediction_matrix() -> str:
    headers = ["Surface", "Mean-Only Prediction", "Exact Variance", "LOVE Variance", "Prediction Cache"]
    rendered_rows = []
    for surface in SURFACES:
        rendered_rows.append(
            [
                surface.label,
                _status(PREDICTION_MATRIX["mean_only"][surface.key]),
                _status(PREDICTION_MATRIX["exact_variance"][surface.key]),
                _status(PREDICTION_MATRIX["love_variance"][surface.key]),
                _status(PREDICTION_MATRIX["prediction_cache"][surface.key]),
            ]
        )
    return _markdown_table(headers, rendered_rows)


def _render_sampling_matrix() -> str:
    headers = ["Surface", "Diagonal Sampling", "Pathwise Sampling", "Polynomial Pathwise", "Public Cholesky Sampling"]
    rendered_rows = []
    for surface in SURFACES:
        rendered_rows.append(
            [
                surface.label,
                _status(SAMPLING_MATRIX["diagonal_sampling"][surface.key]),
                _status(SAMPLING_MATRIX["pathwise_sampling"][surface.key]),
                _status(SAMPLING_MATRIX["polynomial_pathwise"][surface.key]),
                _status(SAMPLING_MATRIX["cholesky_sampling"][surface.key]),
            ]
        )
    return _markdown_table(headers, rendered_rows)


def _render_kernel_primitives() -> str:
    return _markdown_table(
        ["Primitive Family", "Primitive", "SingleOutput", "ICM", "LMC", "Scope Notes"],
        (
            (
                primitive.family,
                primitive.primitive,
                primitive.single_output.value,
                primitive.icm.value,
                primitive.lmc.value,
                primitive.notes,
            )
            for primitive in KERNEL_PRIMITIVES
        ),
    )


def _render_boundaries() -> str:
    return _markdown_table(
        ["Combination", "Current Category", "Recommended Runtime Behavior"],
        ((entry.combination, entry.category, entry.runtime_behavior) for entry in BOUNDARY_ENTRIES),
    )


def render_feature_matrix_markdown() -> str:
    """Render the public feature matrix documentation from the registry."""

    sections = [
        "# MojoGP Feature Matrix",
        "",
        "This file is generated from `mojogp/feature_support.py`. Do not edit it by hand.",
        "",
        "## Status Legend",
        "",
        _render_status_legend(),
        "",
        "## Surface Definitions",
        "",
        _render_surface_definitions(),
        "",
        "## Main Capability Matrix",
        "",
        _render_matrix(MAIN_ROWS, MAIN_MATRIX),
        "",
        "## Execution Route Matrix",
        "",
        _render_execution_matrix(),
        "",
        "## Prediction / Variance Matrix",
        "",
        _render_prediction_matrix(),
        "",
        "## Posterior Sampling Matrix",
        "",
        _render_sampling_matrix(),
        "",
        "## Kernel Primitive Matrix",
        "",
        _render_kernel_primitives(),
        "",
        "## Boundary / Placeholder Matrix",
        "",
        _render_boundaries(),
        "",
    ]
    return "\n".join(sections)


def assert_registry_complete() -> None:
    """Raise if any matrix row lacks one of the public surfaces."""

    for table_name, (rows, matrix) in MATRICES.items():
        row_keys = {row.key for row in rows}
        if set(matrix) != row_keys:
            raise AssertionError(
                f"{table_name} rows and matrix keys differ: rows={sorted(row_keys)} matrix={sorted(matrix)}"
            )
        for row in rows:
            surfaces = set(matrix[row.key])
            if surfaces != set(SURFACE_KEYS):
                raise AssertionError(
                    f"{table_name}:{row.key} surfaces differ: {sorted(surfaces)}"
                )
