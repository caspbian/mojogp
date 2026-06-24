"""MojoGP public Python API.

The supported wrapper surface is centered on:

1. `SingleOutputGP`
2. `MultiOutputGP`
3. `MultiOutputLMCGP`

Quick example:
    >>> import numpy as np
    >>> from mojogp import SingleOutputGP as SoGP, RBF
    >>> X = np.random.randn(5000, 2).astype(np.float32)
    >>> y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(5000)).astype(np.float32)
    >>> gp = SoGP(RBF(ard=True))
    >>> gp.fit(X, y, max_iterations=50, method="matrix_free")
    >>> X_test = np.random.randn(100, 2).astype(np.float32)
    >>> mean, std = gp.predict(X_test, return_std=True)

Composite kernel example:
    >>> from mojogp import SingleOutputGP as SoGP, RBF, Matern52
    >>> kernel = RBF(active_dims=[0, 1]) + Matern52(active_dims=[1, 2])
    >>> gp = SoGP(kernel)
    >>> gp.fit(X, y, max_iterations=50, method="materialized")
    >>> pred = gp.predict(X_test, variance_method="exact")

Multi-output example:
    >>> from mojogp import MultiOutputGP, MultiOutputLMCGP
    >>> Y = np.random.randn(5000, 2).astype(np.float32)
    >>> icm = MultiOutputGP(kernel="rbf")
    >>> icm.fit(X, Y, max_iterations=30, method="matrix_free")
    >>> mean, var = icm.predict(X_test, return_var=True)
    >>> lmc = MultiOutputLMCGP(kernels=["rbf", "matern52"])
    >>> lmc.fit(X, Y, max_iterations=30, method="materialized")

Support boundaries worth knowing:

1. pure categorical models are unsupported
2. public `cholesky` posterior sampling is not part of the live API
3. pathwise sampling is narrower than the full kernel surface
4. `MultiOutputLMCGP(ard=True)` is available on the documented continuous LMC scope; mixed ARD applies only to continuous dimensions after categorical splitting
5. fixed per-sample-per-task LMC observation noise `[n, T]` is available on the documented continuous LMC scope; mixed fixed noise is not public yet

The generated feature/status matrix lives in `docs/FEATURE_MATRIX.md` and is
backed by `mojogp.feature_support`.
"""

# Primary API - single-output GP with JIT-compiled composite kernels
from mojogp.gp import (
    SingleOutputGP,
    TrainingResult,
    PredictionResult,
    fit_gp,
)
from mojogp._version import __version__
from mojogp.feature_support import (
    ExperimentalFeatureWarning,
    FeatureStatus,
    InDevelopmentFeatureWarning,
    MojoGPFeatureWarning,
)
from mojogp import settings as settings
from mojogp.settings import (
    feature_warnings_suppressed,
    get_feature_warnings_enabled,
    get_progress_enabled,
    progress_enabled,
    set_feature_warnings_enabled,
    set_progress_enabled,
)
from mojogp.progress import ProgressEvent, ProgressReporter

# Kernel specification
from mojogp.kernel import (
    Kernel,
    KernelNode,
    KernelType,
    make_ard_kernel,
    compute_dim_permutation,
    analyze_kernel_tree,
    KernelTreeAnalysis,
    CategoricalSpec,
)
from mojogp.kernel import (
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    Linear,
    RQ,
    Polynomial,
    # Categorical kernel aliases
    GD,
    CR,
    EHH,
    HH,
    FE,
)

# Utilities
from mojogp.utils import StandardScaler

# Multi-output GP API
from mojogp.multi_output_gp import (
    MultiOutputGP,
    MultiOutputLMCGP,
    MultiOutputTrainingResult,
    MultiOutputPredictionResult,
    LMCTrainingResult,
)

__all__ = [
    # Primary API
    "SingleOutputGP",
    "TrainingResult",
    "PredictionResult",
    "fit_gp",
    # Multi-output GP
    "MultiOutputGP",
    "MultiOutputLMCGP",
    "MultiOutputTrainingResult",
    "MultiOutputPredictionResult",
    "LMCTrainingResult",
    # Kernel specification
    "Kernel",
    "KernelNode",
    "KernelType",
    "make_ard_kernel",
    # Continuous kernel shortcuts
    "RBF",
    "Matern12",
    "Matern32",
    "Matern52",
    "Periodic",
    "Linear",
    "RQ",
    "Polynomial",
    # Categorical kernel shortcuts
    "GD",
    "CR",
    "EHH",
    "HH",
    "FE",
    # Utilities
    "StandardScaler",
    # Version
    "__version__",
    # Feature support metadata/warnings
    "FeatureStatus",
    "MojoGPFeatureWarning",
    "ExperimentalFeatureWarning",
    "InDevelopmentFeatureWarning",
    "settings",
    "feature_warnings_suppressed",
    "get_feature_warnings_enabled",
    "get_progress_enabled",
    "ProgressEvent",
    "ProgressReporter",
    "progress_enabled",
    "set_feature_warnings_enabled",
    "set_progress_enabled",
]
