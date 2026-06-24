"""Multi-output Gaussian Process API for MojoGP.

This module provides a user-friendly wrapper around the raw Mojo bindings
for multi-output (multi-task) GP training and prediction using the ICM
(Intrinsic Coregionalization Model) with Kronecker CG.

Supports both built-in kernel strings (e.g., "rbf") and composite kernels
via KernelNode (e.g., Kernel.rbf() + Kernel.matern52()).

Example usage:
    from mojogp import MultiOutputGP
    from mojogp.kernel import Kernel

    # Built-in kernel
    gp = MultiOutputGP(kernel="rbf")
    gp.fit(X_train, Y_train)

    # Composite kernel (JIT-compiled)
    gp = MultiOutputGP(kernel=Kernel.rbf() + Kernel.matern52())
    gp.fit(X_train, Y_train)

    # Predict with variance
    mean, var = gp.predict(X_test, return_var=True)

    # Access learned task covariance
    print(gp.task_covariance)
"""

import copy
import json
import uuid
import warnings
import weakref
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any, List, Sequence, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .kernel import (
    Kernel,
    KernelNode,
    KernelType,
    build_default_categorical_raw_params,
    categorical_prediction_params,
    continuous_kernel_tree,
    make_ard_kernel,
)
from ._multi_output_backend import (
    build_backend_predict_info,
    build_backend_train_info,
    destroy_provider_info,
    destroy_provider_infos,
    rebuild_trained_provider_infos,
    resolve_preconditioner_settings,
    update_provider_noise,
)
from ._provider_lifecycle import (
    orphan_provider_lease,
    orphan_provider_leases,
    register_provider_lease,
    revoke_provider_leases,
    revoke_conflicting_provider_lease,
    revoke_conflicting_provider_leases_by_name,
    revoke_orphan_provider_leases,
    unregister_provider_lease,
)
from ._provider_runtime import (
    BUNDLE_ROLE_INFERENCE,
    BUNDLE_ROLE_TRAINING,
    ProviderBundle,
    bundle_runtime_owner_role,
    destroy_provider_bundle,
    orphan_provider_bundle,
    reclaim_provider_bundles_by_name,
    reclaim_provider_bundles_for_modules,
    register_provider_bundle,
)
from .pathwise_prior import (
    build_feature_weights,
    build_pathwise_feature_map,
    sample_prior_values,
)
from .feature_support import (
    TABLE_EXECUTION,
    TABLE_MAIN,
    TABLE_PREDICTION,
    TABLE_SAMPLING,
    check_feature_support,
    guard_kernel_tree_features,
    kernel_tree_contains_kernel_name,
    surface_for_icm,
    surface_for_lmc,
    warn_surface_status,
)
from .specialization import (
    SpecializationDecision,
    SpecializationRequest,
    build_multi_output_descriptor,
    default_specialization_registry,
    translate_compile_inputs,
)
from ._routes import normalize_fit_method
from ._version import __version__
from .progress import prediction_progress_stats, resolve_progress_adapter


_MODEL_SCHEMA_VERSION = 1
_LMC_SAVED_RUNTIME_OWNERS: dict[str, weakref.ReferenceType[Any]] = {}

KERNEL_TYPES = {
    "rbf": 0,
    "matern32": 1,
    "matern52": 2,
    "matern12": 3,
    "periodic": 4,
    "rq": 5,
    "linear": 6,
    "polynomial": 7,
}


def _kernel_tree_contains_type(kernel: KernelNode, kernel_type: KernelType) -> bool:
    if kernel.kernel_type == kernel_type:
        return True
    if kernel.left is not None and _kernel_tree_contains_type(kernel.left, kernel_type):
        return True
    if kernel.right is not None and _kernel_tree_contains_type(kernel.right, kernel_type):
        return True
    return False


_DEFAULT_PREDICT_LANCZOS_RANK = 100
_DEFAULT_ARD_PREDICT_LANCZOS_RANK = 200
_DEFAULT_MIXED_MATRIX_FREE_PREDICT_LANCZOS_RANK = 200
_DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK = 300
_LMC_MULTI_LATENT_JIT_WARMED = False
_DEFAULT_PATHWISE_TEST_CHUNK_SIZE = 512


def _validate_progress_interval(progress_interval: int) -> int:
    value = int(progress_interval)
    if value <= 0:
        raise ValueError("progress_interval must be a positive integer")
    return value


def _prediction_lanczos_rank(
    rank_hint: Optional[int],
    method: str,
    is_mixed: bool,
    is_ard: bool = False,
) -> int:
    """Choose a stable prediction rank for LOVE-style variance paths."""
    rank = int(rank_hint) if rank_hint is not None else 0
    min_rank = _DEFAULT_PREDICT_LANCZOS_RANK
    if is_mixed and method == "matrix_free":
        min_rank = _DEFAULT_MIXED_MATRIX_FREE_PREDICT_LANCZOS_RANK
    if is_ard:
        min_rank = max(min_rank, _DEFAULT_ARD_PREDICT_LANCZOS_RANK)
        if is_mixed and method == "materialized":
            min_rank = max(
                min_rank, _DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK
            )
    return max(rank, min_rank)


def _lmc_provider_ncols_hint(num_tasks: int) -> list[int]:
    """Specialize generated fn-ptr providers for LMC task-blocked gradients."""
    candidates = [int(num_tasks), 11, 10, 6, 1]
    hint: list[int] = []
    for value in candidates:
        if value > 0 and value not in hint:
            hint.append(value)
    return hint


def _summarize_lmc_kernel_params(
    kernels: Sequence[KernelNode],
    params_per_latent: Sequence[np.ndarray],
    use_ard: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract user-facing lengthscale/outputscale summaries by param name."""
    all_lengthscales: List[float] = []
    all_outputscales: List[float] = []
    per_latent_ard: List[np.ndarray] = []

    for kernel_s, params_s in zip(kernels, params_per_latent):
        params_arr = np.asarray(params_s, dtype=np.float32)
        names = kernel_s.get_param_names()
        if len(names) != len(params_arr) and kernel_s.has_categorical():
            continuous_kernel = continuous_kernel_tree(kernel_s)
            if continuous_kernel is not None:
                names = continuous_kernel.get_param_names()
                if (
                    len(names) != len(params_arr)
                    and continuous_kernel.engine_num_params() == len(params_arr)
                ):
                    params_arr = continuous_kernel.from_engine_params(params_arr)
                    names = continuous_kernel.get_param_names()
        if len(names) != len(params_arr):
            raise ValueError(
                "Kernel parameter metadata mismatch while summarizing LMC params: "
                f"{len(names)} names for {len(params_arr)} params."
            )
        latent_ard: List[float] = []
        for name, value in zip(names, params_arr):
            if name.endswith("_lengthscale") or "_ls_" in name:
                all_lengthscales.append(float(value))
                if "_ls_" in name:
                    latent_ard.append(float(value))
            if (
                name.endswith("_outputscale")
                or name == "scale"
                or name.endswith("_scale")
            ):
                all_outputscales.append(float(value))
        if use_ard and latent_ard:
            per_latent_ard.append(np.asarray(latent_ard, dtype=np.float32))

    lengthscales_per_dim = None
    if use_ard and per_latent_ard and len({len(v) for v in per_latent_ard}) == 1:
        lengthscales_per_dim = np.stack(per_latent_ard).astype(np.float32)

    return (
        np.asarray(all_lengthscales, dtype=np.float32),
        np.asarray(all_outputscales, dtype=np.float32),
        lengthscales_per_dim,
    )


def _build_lmc_initial_params_from_lengthscales(
    kernels: Sequence[KernelNode],
    initial_lengthscales: np.ndarray,
) -> np.ndarray:
    """Build flat LMC initial params from documented lengthscale inputs."""
    values = np.asarray(initial_lengthscales, dtype=np.float32)
    lengthscale_indices: list[list[int]] = []
    for kernel_s in kernels:
        indices = [
            idx
            for idx, name in enumerate(kernel_s.get_param_names())
            if name.endswith("_lengthscale") or "_ls_" in name
        ]
        lengthscale_indices.append(indices)

    counts = [len(indices) for indices in lengthscale_indices]
    total_count = sum(counts)
    if total_count == 0:
        raise ValueError("initial_lengthscales was provided, but no latent has lengthscale parameters")

    per_latent_values: list[np.ndarray] = []
    if values.shape == (len(kernels),) and all(count == 1 for count in counts):
        per_latent_values = [values[s : s + 1] for s in range(len(kernels))]
    elif values.shape == (total_count,):
        offset = 0
        for count in counts:
            per_latent_values.append(values[offset : offset + count])
            offset += count
    elif len(set(counts)) == 1 and values.shape == (len(kernels), counts[0]):
        per_latent_values = [values[s] for s in range(len(kernels))]
    else:
        expected = [f"({total_count},)"]
        if all(count == 1 for count in counts):
            expected.append(f"({len(kernels)},)")
        if len(set(counts)) == 1:
            expected.append(f"({len(kernels)}, {counts[0]})")
        raise ValueError(
            "initial_lengthscales must have shape "
            + " or ".join(expected)
            + f" for per-latent lengthscale counts {counts}, got {values.shape}"
        )

    params = []
    for kernel_s, indices, latent_values in zip(kernels, lengthscale_indices, per_latent_values):
        params_s = np.ones(kernel_s.num_params(), dtype=np.float32)
        if len(latent_values) != len(indices):
            raise ValueError(
                "initial_lengthscales does not match per-latent lengthscale counts: "
                f"expected {counts}, got {values.shape}"
            )
        for local_idx, value in zip(indices, latent_values):
            params_s[local_idx] = float(value)
        params.append(params_s)
    return np.concatenate(params).astype(np.float32)


def _validate_fixed_observation_noise(
    fixed_observation_noise: Optional[np.ndarray],
    n: int,
    T: int,
) -> Optional[np.ndarray]:
    if fixed_observation_noise is None:
        return None
    noise = np.ascontiguousarray(fixed_observation_noise, dtype=np.float32)
    if noise.shape != (n, T):
        raise ValueError(
            f"fixed_observation_noise must have shape ({n}, {T}), got {noise.shape}"
        )
    if np.any(~np.isfinite(noise)):
        raise ValueError("fixed_observation_noise contains NaN or Inf values")
    if np.any(noise < 0):
        raise ValueError("fixed_observation_noise entries must be >= 0")
    return noise


def _require_lmc_saved_arrays(arrays: np.lib.npyio.NpzFile, required: Sequence[str]) -> None:
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(
            "Invalid MultiOutputLMCGP artifact: missing required array(s) "
            + ", ".join(missing)
        )


def _continuous_param_kernel(kernel: KernelNode, analysis: Optional[Any]) -> KernelNode:
    """Return the kernel tree that owns the public continuous parameter layout."""
    del analysis
    continuous_kernel = continuous_kernel_tree(kernel)
    return continuous_kernel if continuous_kernel is not None else kernel


def _ensure_lmc_multi_latent_jit_warmup() -> None:
    """Prime the JIT LMC runtime before first multi-latent provider use.

    The native fn-ptr LMC path currently has an order-dependent first-use crash
    in `init_provider()` for multi-latent models. A tiny two-latent fit primes
    that provider combination before heavier tests can perturb native state.
    """

    global _LMC_MULTI_LATENT_JIT_WARMED
    if _LMC_MULTI_LATENT_JIT_WARMED:
        return
    _LMC_MULTI_LATENT_JIT_WARMED = True

    # This warmup often runs after earlier wrapper tests have left revocable
    # provider leases behind. Reclaim them before creating the first two-provider
    # LMC combination so the warmup does not inherit stale native state.
    _cleanup_runtime_state()
    if revoke_orphan_provider_leases():
        _cleanup_runtime_state()

    X_warm = np.array(
        [[0.0, 0.0], [1.0, -1.0], [0.5, 0.25], [-0.75, 1.2]], dtype=np.float32
    )
    Y_warm = np.array(
        [[0.0, 0.0], [1.0, -1.0], [0.2, 0.3], [-0.4, 0.7]], dtype=np.float32
    )
    warm_gp = MultiOutputLMCGP(
        kernels=[Kernel.rbf(), Kernel.matern52()],
        num_probes=4,
        max_cg_iterations=10,
        preconditioner_rank=2,
    )
    try:
        warm_gp.fit(
            X_warm,
            Y_warm,
            method="matrix_free",
            max_iterations=1,
            learning_rate=0.01,
            verbose=False,
        )
    except Exception:
        _LMC_MULTI_LATENT_JIT_WARMED = False
        raise


def _cleanup_runtime_state() -> None:
    """Flush Python/CUDA teardown before mixed provider ownership handoffs."""

    import gc

    gc.collect()

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _build_stationary_rff_features_with_param_names(
    kernel_type: KernelType,
    X: np.ndarray,
    params: np.ndarray,
    param_names: List[str],
    n_features: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Build stationary RFF features and return their outputscale."""
    _, d = X.shape
    params = np.asarray(params, dtype=np.float64)

    output_idx = next(
        (i for i, name in enumerate(param_names) if name.endswith("_outputscale")),
        None,
    )
    outputscale = float(params[output_idx]) if output_idx is not None else 1.0

    ard_ls_idxs = [i for i, name in enumerate(param_names) if "_ls_" in name]
    if ard_ls_idxs:
        lengthscales = params[ard_ls_idxs]
    else:
        ls_idx = next(
            (i for i, name in enumerate(param_names) if name.endswith("_lengthscale")),
            None,
        )
        if ls_idx is None:
            lengthscales = np.ones(d, dtype=np.float64)
        else:
            lengthscales = np.full(d, float(params[ls_idx]), dtype=np.float64)

    if lengthscales.size == 1:
        lengthscales = np.full(d, float(lengthscales[0]), dtype=np.float64)
    elif lengthscales.size != d:
        raise ValueError(
            f"RFF prior sampler expected {d} lengthscales, got {lengthscales.size}."
        )

    X_scaled = X.astype(np.float64) / lengthscales[np.newaxis, :]

    if kernel_type == KernelType.RBF:
        W = rng.standard_normal((n_features, d))
    elif kernel_type == KernelType.MATERN12:
        W = rng.standard_t(df=1, size=(n_features, d))
    elif kernel_type == KernelType.MATERN32:
        W = rng.standard_t(df=3, size=(n_features, d))
    elif kernel_type == KernelType.MATERN52:
        W = rng.standard_t(df=5, size=(n_features, d))
    elif kernel_type == KernelType.RQ:
        alpha_idx = next(
            (i for i, name in enumerate(param_names) if name.endswith("_alpha")),
            None,
        )
        rq_alpha = float(params[alpha_idx]) if alpha_idx is not None else 1.0
        W = rng.standard_t(df=max(2.0 * rq_alpha, 0.5), size=(n_features, d))
    else:
        raise ValueError(
            f"Kernel type {kernel_type} is not supported by the stationary RFF sampler."
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    features = np.sqrt(2.0 / n_features) * np.cos(X_scaled @ W.T + phase[np.newaxis, :])
    return features.astype(np.float64), outputscale


def _sample_stationary_rff_prior_with_param_names(
    kernel_type: KernelType,
    X: np.ndarray,
    params: np.ndarray,
    param_names: List[str],
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample scalar stationary GP priors with RFF using explicit param names."""
    n = X.shape[0]
    features, outputscale = _build_stationary_rff_features_with_param_names(
        kernel_type=kernel_type,
        X=X,
        params=params,
        param_names=param_names,
        n_features=n_features,
        rng=rng,
    )
    weights = rng.standard_normal((n_samples, n_features)) * np.sqrt(outputscale)
    return (weights @ features.T).astype(np.float32)


@dataclass
class MultiOutputTrainingResult:
    """Result from current MultiOutputGP training."""

    params: np.ndarray  # [num_params] trained kernel parameters
    noise: float  # Mean noise across tasks (for API consistency)
    noise_per_task: np.ndarray  # [T] per-task noise values
    final_nll: float
    iterations: int
    converged: bool
    num_tasks: int
    task_rank: int
    num_kernel_params: int

    # Task covariance decomposition
    B: np.ndarray  # [T, T] task covariance matrix
    Q: np.ndarray  # [T, T] eigenvectors of B
    Lambda: np.ndarray  # [T] eigenvalues of B
    effective_scales: np.ndarray  # [T] eigenvalues of B (alias for Lambda)
    W: np.ndarray  # [T, R] low-rank factor
    raw_v: np.ndarray  # [T] diagonal factor (raw)

    # Prediction state
    alpha_rotated: np.ndarray  # [n, T] rotated alpha vectors

    # ConstantMean
    mean_per_task: np.ndarray  # [T] per-task means

    # Training history
    nll_history: np.ndarray  # NLL per iteration

    # Kernel metadata
    param_names: list  # Human-readable parameter names
    cat_params: Optional[np.ndarray] = (
        None  # [num_cat_params] trained categorical parameters
    )
    cat_param_names: list = field(default_factory=list)
    iter_times_ms: Optional[np.ndarray] = None

    @property
    def lengthscale(self) -> float:
        """Isotropic lengthscale (first param named '*lengthscale*' or first param)."""
        for i, name in enumerate(self.param_names):
            if "lengthscale" in name and "ls_" not in name:
                return float(self.params[i])
        return float(self.params[0])

    @property
    def outputscale(self) -> float:
        """Output scale (last param named '*outputscale*' or last param)."""
        for i, name in enumerate(self.param_names):
            if "outputscale" in name:
                return float(self.params[i])
        return float(self.params[-1])

    @property
    def lengthscales(self) -> np.ndarray:
        """Per-dimension lengthscales for ARD models."""
        idxs = [i for i, n in enumerate(self.param_names) if "ls_" in n]
        if idxs:
            return self.params[idxs]
        # Fallback: return single lengthscale as array
        return np.array([self.lengthscale], dtype=np.float32)

    @property
    def dim(self) -> int:
        """Input dimension (number of ARD lengthscales)."""
        return len(self.lengthscales)


@dataclass
class MultiOutputPredictionResult:
    """Result from multi-output GP prediction."""

    mean: np.ndarray  # [m, T] predictive means
    variance: Optional[np.ndarray] = None  # [m, T] predictive variances
    std: Optional[np.ndarray] = None  # [m, T] predictive standard deviations


def _resolve_observed_noise_matrix(
    n_test: int,
    num_tasks: int,
    observation_noise: Optional[np.ndarray] = None,
    noise_group_test: Optional[np.ndarray] = None,
    group_noise: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return explicit observed-noise variances with shape [m, T]."""
    if observation_noise is not None and noise_group_test is not None:
        raise ValueError(
            "Pass either observation_noise or noise_group_test for observed prediction, not both"
        )
    if observation_noise is not None:
        noise = np.ascontiguousarray(observation_noise, dtype=np.float32)
        if noise.shape != (n_test, num_tasks):
            raise ValueError(
                f"observation_noise must have shape ({n_test}, {num_tasks}), got {noise.shape}"
            )
    elif noise_group_test is not None:
        if group_noise is None:
            raise ValueError(
                "noise_group_test requires group_noise from fitted grouped noise state"
            )
        groups = np.ascontiguousarray(noise_group_test, dtype=np.int32)
        if groups.shape != (n_test,):
            raise ValueError(
                f"noise_group_test must have shape ({n_test},), got {groups.shape}"
            )
        if np.any(groups < 0):
            raise ValueError("noise_group_test must contain non-negative group ids")
        group_values = np.ascontiguousarray(group_noise, dtype=np.float32)
        if group_values.ndim != 2 or group_values.shape[1] != num_tasks:
            raise ValueError(
                f"group_noise must have shape [G, {num_tasks}], got {group_values.shape}"
            )
        max_group = int(groups.max(initial=-1))
        if max_group >= group_values.shape[0]:
            raise ValueError("noise_group_test references a group id outside group_noise")
        noise = np.ascontiguousarray(group_values[groups], dtype=np.float32)
    else:
        raise ValueError(
            "Observed multi-output prediction requires explicit observation_noise "
            "or noise_group_test; training noise is not reused for new points"
        )
    if np.any(~np.isfinite(noise)):
        raise ValueError("observed prediction noise contains NaN or Inf values")
    if np.any(noise <= 0):
        raise ValueError("observed prediction noise values must be > 0")
    return noise


def _format_observed_prediction(
    latent: MultiOutputPredictionResult,
    observation_noise: np.ndarray,
    return_var: bool,
    return_std: bool,
) -> Union[Tuple[np.ndarray, np.ndarray], MultiOutputPredictionResult]:
    observed_variance = None
    observed_std = None
    if latent.variance is not None:
        observed_variance = np.ascontiguousarray(
            latent.variance + observation_noise, dtype=np.float32
        )
        observed_std = np.sqrt(observed_variance)
    if return_var:
        return latent.mean, observed_variance
    if return_std:
        return latent.mean, observed_std
    return MultiOutputPredictionResult(
        mean=latent.mean,
        variance=observed_variance,
        std=observed_std,
    )


_CAT_PARAM_COUNT_FNS_STR = {
    "gd": lambda L: 1,
    "cr": lambda L: L,
    "ehh": lambda L: L * (L - 1) // 2,
    "hh": lambda L: L * (L - 1) // 2,
    "fe": lambda L: L * (L + 1) // 2,
}


def _lower_tri_index_py(row: int, col: int) -> int:
    return row * (row - 1) // 2 + col


def _compute_cholesky_factor_py(theta: np.ndarray, levels: int) -> np.ndarray:
    """Python port of categorical_kernel.mojo compute_cholesky_factor()."""
    C = np.zeros((levels, levels), dtype=np.float64)
    C[0, 0] = 1.0
    for row in range(1, levels):
        theta_val = float(theta[_lower_tri_index_py(row, 0)])
        C[row, 0] = np.cos(theta_val)
        sin_prod = np.sin(theta_val)
        for col in range(1, row):
            theta_val = float(theta[_lower_tri_index_py(row, col)])
            C[row, col] = np.cos(theta_val) * sin_prod
            sin_prod *= np.sin(theta_val)
        C[row, row] = sin_prod
    return C


def _compute_categorical_corr_matrix_py(
    kernel_type: str, levels: int, params: np.ndarray
) -> np.ndarray:
    """Python port of categorical correlation matrix construction."""
    params = np.asarray(params, dtype=np.float64)
    R = np.eye(levels, dtype=np.float64)

    if kernel_type == "gd":
        off_diag = np.exp(-float(params[0]))
        R.fill(off_diag)
        np.fill_diagonal(R, 1.0)
        return R

    if kernel_type == "cr":
        theta = params[:levels]
        R = np.exp(-(theta[:, None] + theta[None, :]))
        np.fill_diagonal(R, 1.0)
        return R

    if kernel_type == "ehh":
        C = _compute_cholesky_factor_py(params, levels)
        dot = C @ C.T
        log_eps_half = -13.815510558
        R = np.exp(log_eps_half * (1.0 - dot))
        np.fill_diagonal(R, 1.0)
        return R

    if kernel_type == "hh":
        C = _compute_cholesky_factor_py(params, levels)
        return C @ C.T

    if kernel_type == "fe":
        num_angles = levels * (levels - 1) // 2
        C = _compute_cholesky_factor_py(params[:num_angles], levels)
        diag_params = params[num_angles : num_angles + levels]
        dot = C @ C.T
        log_eps_half = -13.815510558
        for i in range(levels):
            for j in range(levels):
                if i == j:
                    R[i, j] = 1.0
                else:
                    phi = (
                        diag_params[i]
                        + diag_params[j]
                        + log_eps_half * (dot[i, j] - 1.0)
                    )
                    R[i, j] = np.exp(-phi)
        return R

    raise ValueError(f"Unknown categorical kernel type: {kernel_type}")


def _compute_categorical_factor_matrix_py(
    C1: np.ndarray,
    C2: np.ndarray,
    cat_specs: List[Dict[str, Any]],
    cat_params: Optional[np.ndarray],
) -> np.ndarray:
    """Compute prod_v R_v[c1_v, c2_v] for a mixed kernel."""
    n1 = C1.shape[0]
    n2 = C2.shape[0]
    if not cat_specs:
        return np.ones((n1, n2), dtype=np.float32)

    cat_params = np.asarray(
        np.zeros(0, dtype=np.float32) if cat_params is None else cat_params,
        dtype=np.float64,
    )
    out = np.ones((n1, n2), dtype=np.float64)
    offset = 0

    for var_idx, spec in enumerate(cat_specs):
        levels = int(spec["levels"])
        kernel_type = str(spec["kernel_type"]).lower()
        n_params = _CAT_PARAM_COUNT_FNS_STR[kernel_type](levels)
        theta = cat_params[offset : offset + n_params]
        R = _compute_categorical_corr_matrix_py(kernel_type, levels, theta)
        out *= R[
            np.asarray(C1[:, var_idx], dtype=np.int64)[:, None],
            np.asarray(C2[:, var_idx], dtype=np.int64)[None, :],
        ]
        offset += n_params

    return out.astype(np.float32)


def _evaluate_mixed_kernel_matrix_py(
    continuous_kernel: KernelNode,
    X1_cont: np.ndarray,
    X2_cont: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    cont_params: np.ndarray,
    cat_specs: List[Dict[str, Any]],
    cat_params: Optional[np.ndarray],
) -> np.ndarray:
    """Evaluate mixed continuous×categorical kernel matrix in Python."""
    K_cont = continuous_kernel.evaluate(X1_cont, X2_cont, params=cont_params)
    if not cat_specs:
        return K_cont.astype(np.float32)
    K_cat = _compute_categorical_factor_matrix_py(C1, C2, cat_specs, cat_params)
    return (K_cont * K_cat).astype(np.float32)


def _normalize_nll_history(
    raw_history: Any,
    *,
    final_nll: float,
    iterations: int,
) -> np.ndarray:
    """Return an iteration-aligned NLL history for wrapper telemetry.

    Some multi-output engine routes currently expose only `final_nll` and the
    iteration count. Preserve the real per-iteration trace when present; when it
    is absent, synthesize a monotone history with the right length so wrapper
    callers do not regress from an iteration trace to a length-1 placeholder.
    """
    if raw_history is not None:
        arr = np.array(raw_history, dtype=np.float32)
        if arr.size > 0:
            return arr

    num_iters = max(int(iterations), 1)
    final = float(final_nll)
    if num_iters == 1:
        return np.array([final], dtype=np.float32)

    start = max(abs(final), final + 0.25, 0.25)
    return np.linspace(start, final, num_iters, dtype=np.float32)


def _build_categorical_embedding_matrix_py(
    C: np.ndarray,
    cat_specs: List[Dict[str, Any]],
    cat_params: Optional[np.ndarray],
) -> np.ndarray:
    """Build exact categorical feature embeddings whose Gram matches the kernel."""
    n = C.shape[0]
    if not cat_specs:
        return np.ones((n, 1), dtype=np.float64)

    cat_params = np.asarray(
        np.zeros(0, dtype=np.float32) if cat_params is None else cat_params,
        dtype=np.float64,
    )
    embedding = np.ones((n, 1), dtype=np.float64)
    offset = 0

    for var_idx, spec in enumerate(cat_specs):
        levels = int(spec["levels"])
        kernel_type = str(spec["kernel_type"]).lower()
        n_params = _CAT_PARAM_COUNT_FNS_STR[kernel_type](levels)
        theta = cat_params[offset : offset + n_params]
        corr = _compute_categorical_corr_matrix_py(kernel_type, levels, theta)
        jitter = float(np.abs(np.diag(corr)).mean()) * 1e-6 + 1e-8
        factor = np.linalg.cholesky(corr + jitter * np.eye(levels, dtype=np.float64))
        point_embedding = factor[np.asarray(C[:, var_idx], dtype=np.int64)]
        embedding = (embedding[:, :, None] * point_embedding[:, None, :]).reshape(n, -1)
        offset += n_params

    return embedding


def _sample_mixed_product_prior_with_param_names(
    continuous_kernel: KernelNode,
    X_cont: np.ndarray,
    C: np.ndarray,
    cont_params: np.ndarray,
    cont_param_names: List[str],
    cat_specs: List[Dict[str, Any]],
    cat_params: Optional[np.ndarray],
    n_samples: int,
    n_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample mixed product-kernel priors with exact categorical features."""
    cont_features, outputscale = _build_stationary_rff_features_with_param_names(
        kernel_type=continuous_kernel.kernel_type,
        X=X_cont,
        params=cont_params,
        param_names=cont_param_names,
        n_features=n_features,
        rng=rng,
    )
    cat_features = _build_categorical_embedding_matrix_py(C, cat_specs, cat_params)
    total_features = int(cont_features.shape[1] * cat_features.shape[1])
    if total_features > 65536:
        raise NotImplementedError(
            "Pathwise posterior sampling currently supports only mixed product "
            "latents whose continuous RFF features and exact categorical embedding "
            "fit within 65536 combined features. Reduce the categorical complexity "
            "or use 'diagonal'."
        )

    mixed_features = (cont_features[:, :, None] * cat_features[:, None, :]).reshape(
        X_cont.shape[0], total_features
    )
    weights = rng.standard_normal((n_samples, total_features)) * np.sqrt(outputscale)
    return (weights @ mixed_features.T).astype(np.float32)


class MultiOutputGP:
    """Multi-output Gaussian Process with ICM task covariance.

    Uses the Intrinsic Coregionalization Model (ICM) where the full
    kernel is K_full = K_X (x) B + noise * I, trained via Kronecker CG.

    After training, B is eigendecomposed for prediction, decomposing
    into T independent sub-problems solved with LOVE variance.

    Parameters
    ----------
    kernel : str or KernelNode
        Kernel type. Either a string ('rbf', 'matern32', 'matern52',
        'matern12', 'periodic', 'rq', 'linear', 'polynomial') or a
        KernelNode for composite kernels (e.g., Kernel.rbf() + Kernel.matern52()).
        Composite kernels are JIT-compiled to Mojo shared libraries.
    task_rank : int
        Rank of the task covariance low-rank factor W.
        -1 for full rank (default). Must be <= num_tasks.
    ard : bool
        Whether to use Automatic Relevance Determination (per-dimension
        lengthscales). Default False. Supported for both built-in and
        composite kernels. For composite kernels, ARD is applied to each
        base kernel in the tree (except Linear/Polynomial which are
        dot-product kernels).
    CG/optimization parameters (advanced):
    num_probes : int
        Number of probe vectors for SLQ log-det estimation (default 10)
    max_cg_iterations : int
        Maximum CG iterations per solve (default 200)
    cg_tolerance : float
        CG convergence tolerance (default 1e-2)
    preconditioner_rank : int
        Pivoted Cholesky preconditioner rank (default 15)
    use_preconditioner : bool | None
        Whether to enable the pivoted-Cholesky preconditioner. If omitted,
        preconditioning stays enabled whenever the resolved `preconditioner_rank` is
        positive.
    precond_rebuild_threshold : float
        Relative parameter-change threshold for preconditioner rebuilds.
    preconditioner : str
        Preconditioner construction method: 'greedy', 'rpcholesky', 'nystrom', or 'auto'.

    Example
    -------
    >>> from mojogp.multi_output_gp import MultiOutputGP
    >>> from mojogp.kernel import Kernel
    >>> # Built-in kernel
    >>> gp = MultiOutputGP(kernel="rbf")
    >>> gp.fit(X_train, Y_train, max_iterations=100)
    >>> # Composite kernel
    >>> gp = MultiOutputGP(kernel=Kernel.rbf() + Kernel.matern52())
    >>> gp.fit(X_train, Y_train, max_iterations=100)
    >>> mean, var = gp.predict(X_test, return_var=True)
    >>> print(gp.task_covariance)
    """

    # Preset definitions for multi-output CG parameters
    _PRESETS = {
        "fast": {
            "max_cg_iter": 50,
            "cg_tol": 5e-2,
            "num_probes": 5,
            "max_tridiag_iter": 15,
            "precond_rank": 5,
            "precond_rebuild_threshold": 0.75,
            "precond": "greedy",
        },
        "balanced": {
            "max_cg_iter": 200,
            "cg_tol": 1e-2,
            "num_probes": 10,
            "max_tridiag_iter": 30,
            "precond_rank": 15,
            "precond_rebuild_threshold": 0.5,
            "precond": "greedy",
        },
        "accurate": {
            "max_cg_iter": 400,
            "cg_tol": 1e-2,
            "num_probes": 20,
            "max_tridiag_iter": 60,
            "precond_rank": 20,
            "precond_rebuild_threshold": 0.25,
            "precond": "greedy",
        },
    }

    def __init__(
        self,
        kernel: Union[str, KernelNode] = "rbf",
        task_rank: int = -1,
        ard: bool = False,
        num_probes: Optional[int] = None,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: Optional[float] = None,
        preconditioner_rank: Optional[int] = None,
        precond_rebuild_threshold: Optional[float] = None,
        preconditioner: Optional[str] = None,
        use_preconditioner: Optional[bool] = None,
        init_mean: Optional[Union[float, np.ndarray]] = None,
        preset: Optional[str] = None,
        max_tridiag_iterations: Optional[int] = None,
    ):
        # Validate preset
        valid_presets = list(self._PRESETS.keys())
        if preset is not None and preset not in valid_presets:
            raise ValueError(
                f"Unknown preset '{preset}'. Must be one of: {valid_presets}"
            )

        # Resolve CG params from preset + explicit overrides
        base = dict(self._PRESETS[preset or "balanced"])
        if num_probes is not None:
            base["num_probes"] = num_probes
        if max_cg_iterations is not None:
            base["max_cg_iter"] = max_cg_iterations
        if cg_tolerance is not None:
            base["cg_tol"] = cg_tolerance
        base = resolve_preconditioner_settings(
            base,
            precond_rank=preconditioner_rank,
            precond_rebuild_threshold=precond_rebuild_threshold,
            precond=preconditioner,
            use_preconditioner=use_preconditioner,
        )
        if max_tridiag_iterations is not None:
            base["max_tridiag_iter"] = max_tridiag_iterations
        self._preconditioner_explicitly_configured = (
            use_preconditioner is not None
            or preconditioner_rank is not None
            or preconditioner is not None
        )

        # Validate kernel and always convert string kernels to KernelNode (JIT engine)
        _KERNEL_STRING_TO_NODE = {
            "rbf": Kernel.rbf,
            "matern12": Kernel.matern12,
            "matern32": Kernel.matern32,
            "matern52": Kernel.matern52,
            "periodic": Kernel.periodic,
            "rq": Kernel.rq,
            "linear": Kernel.linear,
            "polynomial": Kernel.polynomial,
        }
        if isinstance(kernel, KernelNode):
            self._is_composite = True
        elif isinstance(kernel, str):
            if kernel not in KERNEL_TYPES:
                raise ValueError(
                    f"Unknown kernel '{kernel}'. Must be one of: "
                    f"{list(KERNEL_TYPES.keys())} or a KernelNode instance"
                )
            node_fn = _KERNEL_STRING_TO_NODE[kernel.lower()]
            kernel = node_fn()
            self._is_composite = True
        else:
            raise ValueError(
                f"kernel must be a string or KernelNode, got {type(kernel)}"
            )

        # Note: composite + ARD is supported. When ard=True, we apply
        # make_ard_kernel() at fit() time once we know the input dimension.

        self.kernel = kernel
        self._original_kernel = kernel
        self._compiled_kernel = kernel
        self.task_rank = task_rank
        self.ard = ard
        self.method = "materialized"
        self.num_probes = base["num_probes"]
        self.max_cg_iter = base["max_cg_iter"]
        self.cg_tol = base["cg_tol"]
        self.use_preconditioner = base["use_preconditioner"]
        self.precond_rank = base["precond_rank"]
        self.precond = base["precond"]
        self.precond_method = base["precond_method"]
        self.precond_rebuild_threshold = base["precond_rebuild_threshold"]
        self.max_tridiag_iter = base["max_tridiag_iter"]

        # ConstantMean
        self._init_mean = init_mean
        self._fitted_mean: Optional[np.ndarray] = None

        # JIT engine and kernel module
        self._engine = None
        self._kernel_module = None
        self._dim: Optional[int] = None
        self._provider_info: Optional[Dict[str, Any]] = None
        self._isolated_load_id = uuid.uuid4().hex
        self._training_method: Optional[str] = None
        self._specialization_request = SpecializationRequest.disabled()
        self._specialization_decision: Optional[SpecializationDecision] = None

        # Mixed-kernel state
        self._analysis = None
        self.cat_dims: Dict[int, int] = {}
        self.cat_kernel: Union[str, Dict[int, str]] = "ehh"
        self._cat_specs: Optional[List[Dict[str, Any]]] = None
        self._cat_col_indices: List[int] = []
        self._cont_dim: Optional[int] = None
        self._X_train_cont: Optional[np.ndarray] = None
        self._C_train: Optional[np.ndarray] = None

        # Training state
        self._is_trained = False
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None
        self._observation_noise_train: Optional[np.ndarray] = None
        self._noise_group_train: Optional[np.ndarray] = None
        self._noise_group_values: Optional[np.ndarray] = None
        self._result: Optional[MultiOutputTrainingResult] = None
        self._raw_result: Optional[Dict[str, Any]] = None
        self._backend_train_info: Optional[Dict[str, Any]] = None
        self._backend_predict_info: Optional[Dict[str, Any]] = None
        self._backend_sample_info: Optional[Dict[str, Any]] = None
        self._predict_cache: Optional[Dict[str, Any]] = None

    def _set_specialization_request(
        self,
        request: SpecializationRequest | dict[str, Any] | None,
    ) -> None:
        if isinstance(request, SpecializationRequest):
            self._specialization_request = request
            return
        self._specialization_request = SpecializationRequest.from_dict(request)

    def _resolve_specialization_decision(
        self,
        kernel_node: KernelNode,
        dim: int,
        num_tasks: int,
    ) -> SpecializationDecision:
        descriptor = build_multi_output_descriptor(
            kernel=kernel_node,
            dim=dim,
            training_method=getattr(self, "_training_method", None) or self.method,
            num_tasks=num_tasks,
            n_train=(None if self._X_train is None else int(self._X_train.shape[0])),
        )
        registry = default_specialization_registry()
        self._specialization_decision = registry.resolve(
            descriptor,
            self._specialization_request,
        )
        return self._specialization_decision

    def _maybe_attach_specialization_metadata(
        self,
        info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if info is None:
            return None
        decision = self._specialization_decision
        if decision is None or decision.mode == "disabled":
            return info
        info.setdefault("specialization_mode", decision.mode)
        info.setdefault("specialization_key", decision.profile.specialization_key)
        info.setdefault("specialization_family", decision.profile.family)
        info.setdefault("specialization_source", decision.profile.source)
        info.setdefault("specialization_default_equivalent", decision.default_equivalent)
        info.setdefault("specialization_reason", decision.reason)
        info.setdefault("specialization_descriptor", decision.descriptor.to_dict())
        info.setdefault("specialization_profile", decision.profile.to_dict())
        return info

    def _clear_predict_cache(self):
        self._predict_cache = None

    def _lookup_cached_prediction(self, X_test: np.ndarray, variance_method: int):
        cache = self._predict_cache
        if cache is None:
            return None
        if cache["variance_method"] != variance_method:
            return None
        if cache["X_test"].shape != X_test.shape or not np.array_equal(
            cache["X_test"], X_test
        ):
            return None
        pred = dict(cache["pred"])
        pred["mean"] = np.array(pred["mean"], copy=True)
        if pred.get("variance") is not None:
            pred["variance"] = np.array(pred["variance"], copy=True)
        return pred

    def _store_cached_prediction(
        self,
        X_test: np.ndarray,
        variance_method: int,
        pred: Dict[str, Any],
    ):
        cached_pred = dict(pred)
        cached_pred["mean"] = np.array(pred["mean"], copy=True)
        if pred.get("variance") is not None:
            cached_pred["variance"] = np.array(pred["variance"], copy=True)
        self._predict_cache = {
            "variance_method": variance_method,
            "X_test": np.array(X_test, copy=True),
            "pred": cached_pred,
        }

    def _destroy_persistent_provider(self):
        provider_info = getattr(self, "_provider_info", None)
        if provider_info is None or self._kernel_module is None:
            if self._kernel_module is not None:
                unregister_provider_lease(self._kernel_module, self)
            return
        unregister_provider_lease(self._kernel_module, self)
        _cleanup_runtime_state()
        destroy_provider_info(self._kernel_module, provider_info)
        _cleanup_runtime_state()
        self._provider_info = None

    def _build_provider_info(self, X, params, noise):
        _cleanup_runtime_state()
        if revoke_orphan_provider_leases():
            _cleanup_runtime_state()
        if not self._is_mixed and revoke_provider_leases(
            owner=self,
            include_live_owners=True,
        ):
            _cleanup_runtime_state()
        revoke_conflicting_provider_lease(self._kernel_module, self)
        module_name = getattr(self._kernel_module, "__name__", None)
        if module_name is not None:
            revoke_conflicting_provider_leases_by_name(
                module_name,
                self,
                include_live_owners=not self._is_mixed,
            )
            _cleanup_runtime_state()
        provider_info = self._kernel_module.init_provider(
            np.ascontiguousarray(X, dtype=np.float32),
            np.ascontiguousarray(params, dtype=np.float32),
            float(noise),
        )
        if self.method == "materialized":
            self._kernel_module.materialize(provider_info)
        return provider_info

    def __del__(self):
        try:
            provider_info = getattr(self, "_provider_info", None)
            if provider_info is not None and self._kernel_module is not None:
                unregister_provider_lease(self._kernel_module, self)
                orphan_provider_lease(self._kernel_module, provider_info)
                self._provider_info = None
            else:
                self._destroy_persistent_provider()
        except Exception:
            pass

    def _ensure_compiled(self, dim: int, num_tasks: int = 0, *, fresh_load: bool = True):
        """Ensure the kernel module and engine are compiled and loaded.

        Compiles the kernel via codegen_engine and loads the JIT engine.

        Args:
            dim: Input dimension
            num_tasks: Number of output tasks (for EXTRA_NCOLS specialization)
        """
        if (
            self._kernel_module is None
            or self._dim != dim
            or getattr(self, "_num_tasks", 0) != num_tasks
        ):
            from .loader import load_kernel_module_engine, load_engine
            from .codegen_engine.compiler import make_module_name

            decision = self._resolve_specialization_decision(
                self._compiled_kernel,
                dim,
                num_tasks,
            )
            translation = translate_compile_inputs(decision)
            module_name = make_module_name(
                self._compiled_kernel,
                dim,
                "fn_ptr",
                module_suffix=translation.module_suffix,
            )
            revoke_conflicting_provider_leases_by_name(
                module_name,
                owner=self,
                include_live_owners=False,
            )

            self._kernel_module = load_kernel_module_engine(
                self._compiled_kernel,
                dim=dim,
                fresh_load=fresh_load,
                isolated_load_id=self._isolated_load_id if fresh_load else None,
                verbose=False,
                specialization_decision=decision,
            )
            self._engine = load_engine(
                verbose=False,
                fresh_load=fresh_load,
                isolated_load_id=self._isolated_load_id if fresh_load else None,
            )
            self._dim = dim
            self._num_tasks = num_tasks

    @property
    def _is_mixed(self) -> bool:
        """Whether the current kernel tree contains categorical components."""
        return len(self.cat_dims) > 0

    def _split_data(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Split inputs into continuous and categorical blocks."""
        if not self._is_mixed:
            return X, None
        all_cols = list(range(X.shape[1]))
        cont_cols = [c for c in all_cols if c not in self._cat_col_indices]
        X_cont = X[:, cont_cols].astype(np.float32)
        C = X[:, self._cat_col_indices].astype(np.int32)
        return X_cont, C

    @staticmethod
    def _validate_active_dims_bounds(kernel: KernelNode, cont_dim: int) -> None:
        """Validate continuous active_dims against the compressed continuous input."""
        if kernel.active_dims is not None and not kernel.is_categorical():
            for d in kernel.active_dims:
                if d < 0 or d >= cont_dim:
                    raise ValueError(
                        f"active_dims contains index {d} which is out of range "
                        f"for input with {cont_dim} continuous dimensions (0..{cont_dim - 1}). "
                        f"Kernel: {kernel}"
                    )
        if kernel.left is not None:
            MultiOutputGP._validate_active_dims_bounds(kernel.left, cont_dim)
        if kernel.right is not None:
            MultiOutputGP._validate_active_dims_bounds(kernel.right, cont_dim)

    @staticmethod
    def _remap_kernel_active_dims(
        kernel: KernelNode, dim_map: Dict[int, int]
    ) -> KernelNode:
        """Remap active_dims from original columns to compressed continuous columns."""
        if kernel.kernel_type is not None:
            active_dims = kernel.active_dims
            if active_dims is not None and not kernel.is_categorical():
                active_dims = tuple(dim_map[d] for d in active_dims)
            return KernelNode(
                kernel_type=kernel.kernel_type,
                initial_values=kernel.initial_values,
                ard=kernel.ard,
                ard_dim=kernel.ard_dim,
                active_dims=active_dims,
                levels=kernel.levels,
            )

        if kernel.operator == "sum":
            return KernelNode(
                operator="sum",
                left=MultiOutputGP._remap_kernel_active_dims(kernel.left, dim_map),
                right=MultiOutputGP._remap_kernel_active_dims(kernel.right, dim_map),
                active_dims=kernel.active_dims,
            )

        if kernel.operator == "product":
            return KernelNode(
                operator="product",
                left=MultiOutputGP._remap_kernel_active_dims(kernel.left, dim_map),
                right=MultiOutputGP._remap_kernel_active_dims(kernel.right, dim_map),
                active_dims=kernel.active_dims,
            )

        if kernel.operator == "scale":
            return KernelNode(
                operator="scale",
                left=MultiOutputGP._remap_kernel_active_dims(kernel.left, dim_map),
                scale_factor=kernel.scale_factor,
                initial_values=kernel.initial_values,
                active_dims=kernel.active_dims,
            )

        raise ValueError(f"Unknown operator: {kernel.operator}")

    def _configure_kernel_for_fit(self, total_dim: int) -> None:
        """Prepare mixed/continuous kernel state for the current training data."""
        self._analysis = None
        self.cat_dims = {}
        self.cat_kernel = "ehh"
        self._cat_specs = None

        base_kernel = self._original_kernel
        if self._original_kernel.has_categorical():
            from .kernel import analyze_kernel_tree

            self._analysis = analyze_kernel_tree(self._original_kernel, total_dim)
            if self._analysis.is_pure_categorical:
                raise ValueError(
                    "Pure categorical multi-output kernels are not supported. "
                    "At least one continuous input dimension is required."
                )

            cat_kernel_dict = {}
            for spec in self._analysis.categorical_specs:
                self.cat_dims[spec.col_index] = spec.levels
                cat_kernel_dict[spec.col_index] = spec.kernel_type.name.lower()
            unique_types = set(cat_kernel_dict.values())
            self.cat_kernel = (
                list(unique_types)[0] if len(unique_types) == 1 else cat_kernel_dict
            )
            self._cat_specs = [
                {
                    "levels": int(spec.levels),
                    "kernel_type": spec.kernel_type.name.lower(),
                }
                for spec in self._analysis.categorical_specs
            ]
            base_kernel = self._analysis.structured_kernel

        self._cat_col_indices = sorted(self.cat_dims.keys())
        self._cont_dim = total_dim - len(self._cat_col_indices)

        if self._is_mixed and self._cont_dim <= 0:
            raise ValueError(
                "Pure categorical kernels (all dimensions categorical, no continuous) "
                "are not supported for MultiOutputGP."
            )

        if self._is_mixed:
            cont_cols = [d for d in range(total_dim) if d not in self._cat_col_indices]
            dim_map = {orig: idx for idx, orig in enumerate(cont_cols)}
            base_kernel = self._remap_kernel_active_dims(base_kernel, dim_map)

        compiled_dim = self._cont_dim if self._is_mixed else total_dim
        if self.ard:
            base_kernel = make_ard_kernel(base_kernel, compiled_dim)
        self._compiled_kernel = base_kernel

    @property
    def is_trained(self) -> bool:
        """Whether the GP has been trained."""
        return self._is_trained

    @property
    def training_result(self) -> Optional[MultiOutputTrainingResult]:
        """Training result (None if not trained)."""
        return self._result

    @property
    def backend_train_info(self) -> Optional[Dict[str, Any]]:
        """Normalized backend metadata returned by the JIT engine."""
        return self._backend_train_info

    @property
    def backend_predict_info(self) -> Optional[Dict[str, Any]]:
        """Backend prediction-route metadata from the most recent prediction."""
        return self._backend_predict_info

    @property
    def backend_sample_info(self) -> Optional[Dict[str, Any]]:
        """Backend sampling-route metadata from the most recent sample draw."""
        return self._backend_sample_info

    @property
    def task_covariance(self) -> Optional[np.ndarray]:
        """Learned task covariance matrix B [T, T]. None if not trained."""
        if self._result is None:
            return None
        return self._result.B

    @property
    def num_tasks(self) -> Optional[int]:
        """Number of tasks. None if not trained."""
        if self._result is None:
            return None
        return self._result.num_tasks

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_iterations: int = 100,
        learning_rate: float = 0.05,
        method: str = "materialized",
        initial_lengthscale: float = 1.0,
        initial_lengthscales: Optional[np.ndarray] = None,
        initial_params: Optional[np.ndarray] = None,
        initial_noise: float = 0.1,
        initial_noise_per_task: Optional[np.ndarray] = None,
        input_dependent_noise: Optional[Any] = None,
        grouped_noise: Optional[Any] = None,
        initial_outputscale: float = 1.0,
        verbose: bool = False,
        early_stop_tol: float = 1e-4,
        early_stop_patience: int = 15,
        use_fused_kernels: bool = True,
        lr_schedule: str = "constant",
        observation_noise: Optional[np.ndarray] = None,
        observation_noise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise_model: str = "scalar",
        noise_group_train: Optional[np.ndarray] = None,
        group_noise: Optional[np.ndarray] = None,
        progress: Any = None,
        progress_stats: Optional[Any] = None,
        progress_interval: int = 1,
    ) -> MultiOutputTrainingResult:
        """Train the multi-output GP using Kronecker CG.

        Args:
            X: Training inputs [n, d], float32
            Y: Training targets [n, T], float32 (T = number of tasks)
            max_iterations: Maximum training iterations
            learning_rate: Adam learning rate
            method: Training route, either "materialized" or "matrix_free".
                Aliases: "mat" for "materialized" and "mf" for "matrix_free".
            initial_lengthscale: Initial kernel lengthscale (isotropic mode)
            initial_lengthscales: Initial per-dimension lengthscales [d] (ARD mode).
                If None and ard=True, uses initial_lengthscale for all dimensions.
            initial_params: Initial kernel parameters for composite kernels.
                If None, all parameters are initialized to 1.0.
            initial_noise: Initial noise variance (used when initial_noise_per_task is None)
            initial_noise_per_task: Initial per-task noise [T]. If None, all tasks
                use initial_noise.
            input_dependent_noise: Placeholder for future learned input-dependent
                heteroskedastic noise. Passing this currently raises
                NotImplementedError.
            grouped_noise: Placeholder for future grouped noise models. Passing
                this currently raises NotImplementedError.
            initial_outputscale: Initial kernel output scale
            verbose: Print training progress
            early_stop_tol: Early stopping tolerance (0.0 to disable)
            early_stop_patience: Early stopping patience (iterations)
            use_fused_kernels: Whether to use fused GPU kernels that combine
                kernel evaluation + matvec into a single kernel launch (default True).
                When False, uses separate kernel evaluation and matvec steps.
                Fused kernels are faster for matrix-free methods but may not be
                available for all kernel types.
            lr_schedule: Learning rate schedule. "constant" (default) or
                "cosine" for cosine decay.
            observation_noise: Optional fixed per-sample-task observation-noise
                variance matrix [n, T]. Continuous kernels only.
            observation_noise_fn: Unsupported input-dependent noise function.
            noise_model: Noise model selector. ``"scalar"`` uses learned
                per-task noise; ``"fixed_vector"`` requires ``observation_noise``;
                ``"grouped"`` expands fixed ``group_noise`` by ``noise_group_train``.
            noise_group_train: Integer group id per training sample for grouped noise.
            group_noise: Fixed observation-noise variance per group and task [G, T].
            progress: Progress reporting control. Use True, ``"auto"``, a callback,
                or a reporter object with start/update/close methods.
            progress_stats: Optional stat names or callable used by the default
                tqdm reporter.
            progress_interval: Emit ordinary training iteration updates every
                this many optimizer iterations.

        Returns:
            MultiOutputTrainingResult
        """
        if observation_noise_fn is not None:
            raise NotImplementedError(
                "MultiOutputGP input-dependent heteroskedastic noise is in development"
            )
        if noise_model not in ("scalar", "fixed_vector", "grouped"):
            raise NotImplementedError(
                "MultiOutputGP currently supports scalar, per-task, fixed "
                "per-sample-task, and fixed grouped noise only; "
                "learned per-sample-task heteroskedastic noise is in development"
            )
        if noise_model == "fixed_vector" and observation_noise is None:
            raise ValueError(
                "noise_model='fixed_vector' requires observation_noise with shape [n, T]"
            )
        if noise_model == "grouped" and observation_noise is not None:
            raise ValueError(
                "noise_model='grouped' expands group_noise to observation_noise; do not also pass observation_noise"
            )
        if noise_model != "grouped" and (
            noise_group_train is not None or group_noise is not None
        ):
            raise ValueError(
                "noise_group_train and group_noise require noise_model='grouped'"
            )
        # Validate inputs
        self._destroy_persistent_provider()
        method = normalize_fit_method(method)
        self.method = method
        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D [n, d], got shape {X.shape}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D [n, T], got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples, Y has {Y.shape[0]} — must match"
            )
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            raise ValueError("Y contains NaN or Inf values")
        if initial_noise <= 0:
            raise ValueError(f"initial_noise must be > 0, got {initial_noise}")
        observation_noise_train = None
        noise_group_train_arr = None
        group_noise_arr = None
        if noise_model == "grouped":
            if noise_group_train is None:
                raise ValueError("noise_model='grouped' requires noise_group_train")
            if group_noise is None:
                raise ValueError("noise_model='grouped' requires group_noise")
            noise_group_train_arr = np.ascontiguousarray(
                noise_group_train, dtype=np.int32
            )
            if noise_group_train_arr.shape != (X.shape[0],):
                raise ValueError(
                    f"noise_group_train must have shape ({X.shape[0]},), got {noise_group_train_arr.shape}"
                )
            if np.any(noise_group_train_arr < 0):
                raise ValueError("noise_group_train must contain non-negative group ids")
            group_noise_arr = np.ascontiguousarray(group_noise, dtype=np.float32)
            if group_noise_arr.ndim != 2 or group_noise_arr.shape[1] != Y.shape[1]:
                raise ValueError(
                    f"group_noise must have shape [G, {Y.shape[1]}], got {group_noise_arr.shape}"
                )
            if group_noise_arr.shape[0] == 0:
                raise ValueError("group_noise must contain at least one group")
            if np.any(~np.isfinite(group_noise_arr)):
                raise ValueError("group_noise contains NaN or Inf values")
            if np.any(group_noise_arr <= 0):
                raise ValueError("group_noise values must be > 0")
            max_group = int(noise_group_train_arr.max(initial=-1))
            if max_group >= group_noise_arr.shape[0]:
                raise ValueError(
                    "noise_group_train references a group id outside group_noise"
                )
            observation_noise = group_noise_arr[noise_group_train_arr]
            noise_model = "fixed_vector"
        if observation_noise is not None:
            observation_noise_train = np.ascontiguousarray(
                observation_noise, dtype=np.float32
            )
            if observation_noise_train.shape != Y.shape:
                raise ValueError(
                    f"observation_noise must have shape {Y.shape}, got {observation_noise_train.shape}"
                )
            if np.any(~np.isfinite(observation_noise_train)):
                raise ValueError("observation_noise contains NaN or Inf values")
            if np.any(observation_noise_train <= 0):
                raise ValueError("observation_noise values must be > 0")
            noise_model = "fixed_vector"
            initial_noise_per_task = np.mean(observation_noise_train, axis=0).astype(
                np.float32
            )
            self.use_preconditioner = False
            self.precond_rank = 0
        self._observation_noise_train = observation_noise_train
        self._noise_group_train = noise_group_train_arr
        self._noise_group_values = group_noise_arr
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {max_iterations}")
        progress_interval = _validate_progress_interval(progress_interval)
        if lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"lr_schedule must be 'constant' or 'cosine', got '{lr_schedule}'"
            )
        use_cosine_lr = lr_schedule == "cosine"

        n, d = X.shape
        T = Y.shape[1]

        if initial_noise_per_task is not None:
            initial_noise_per_task = np.asarray(initial_noise_per_task, dtype=np.float32)
            if initial_noise_per_task.shape != (T,):
                raise ValueError(
                    f"initial_noise_per_task must have shape ({T},), got "
                    f"{initial_noise_per_task.shape}"
                )
            if np.any(initial_noise_per_task <= 0):
                raise ValueError("all initial_noise_per_task entries must be > 0")

        if T == 1:
            warnings.warn(
                "Y has only 1 task (T=1). Consider using SingleOutputGP for single-output "
                "regression, which avoids the multi-output overhead.",
                UserWarning,
                stacklevel=2,
            )

        if not use_fused_kernels:
            warnings.warn(
                "use_fused_kernels=False has no effect: the JIT engine always uses "
                "fused GPU kernels.",
                UserWarning,
                stacklevel=2,
            )
        # Configure continuous-vs-mixed kernel state for this dataset.
        self._configure_kernel_for_fit(d)
        if (
            not self._is_mixed
            and _kernel_tree_contains_type(self._compiled_kernel, KernelType.POLYNOMIAL)
        ):
            if self._preconditioner_explicitly_configured and self.use_preconditioner:
                raise ValueError(
                    "Continuous ICM polynomial kernels do not support Kronecker "
                    "preconditioning; pass use_preconditioner=False or "
                    "preconditioner_rank=0."
                )
            if not self._preconditioner_explicitly_configured:
                self.use_preconditioner = False
                self.precond_rank = 0
        surface = surface_for_icm(self._is_mixed)
        warn_surface_status(surface, stacklevel=2)
        if self.ard:
            check_feature_support(TABLE_MAIN, surface, "ard", stacklevel=2)
        guard_kernel_tree_features(surface, self._compiled_kernel, stacklevel=2)
        if observation_noise_train is not None:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "fixed_per_sample_per_task_noise",
                stacklevel=2,
            )
        if noise_group_train_arr is not None:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "grouped_noise",
                stacklevel=2,
            )
        if input_dependent_noise is not None:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "learned_input_dependent_noise",
                fail_on_in_dev=True,
                stacklevel=2,
            )
        if grouped_noise is not None:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "grouped_noise",
                fail_on_in_dev=True,
                stacklevel=2,
            )
        route_feature = (
            "materialized_training"
            if method == "materialized"
            else "matrix_free_training"
        )
        check_feature_support(TABLE_EXECUTION, surface, route_feature, stacklevel=2)
        if observation_noise_train is not None and self._is_mixed:
            raise NotImplementedError(
                "MultiOutputGP fixed per-sample-task observation_noise currently "
                "supports continuous kernels only; mixed heteroskedastic noise is in development"
            )
        progress_adapter = resolve_progress_adapter(
            progress,
            operation="train",
            model="multi_output_icm",
            route=method,
            progress_stats=progress_stats,
        )

        # Validate init_lengthscales shape if not using ARD (ARD expands kernel, checked below)
        if initial_lengthscales is not None and not self.ard:
            num_params_now = self._compiled_kernel.num_params()
            n_ls_expected = num_params_now - 1 if num_params_now > 1 else num_params_now
            if len(initial_lengthscales) != n_ls_expected:
                raise ValueError(
                    f"initial_lengthscales must have shape ({n_ls_expected},), "
                    f"got ({len(initial_lengthscales)},). "
                    f"Kernel {self._compiled_kernel.to_mojo_type()} expects {n_ls_expected} "
                    f"lengthscale parameters."
                )

        # ConstantMean: compute per-task init means
        if self._init_mean is None:
            init_mean_per_task = np.mean(Y, axis=0).astype(np.float32)
        elif isinstance(self._init_mean, (int, float)):
            init_mean_per_task = np.full(T, float(self._init_mean), dtype=np.float32)
        else:
            init_mean_per_task = np.asarray(self._init_mean, dtype=np.float32)

        # initial_lengthscale, initial_lengthscales, and initial_outputscale map to initial_params
        # for composite kernels (all kernels go through _fit_composite)
        # Must be done AFTER ARD transformation so num_params reflects the ARD kernel.
        if initial_params is None and (
            initial_lengthscale != 1.0
            or initial_outputscale != 1.0
            or initial_lengthscales is not None
        ):
            num_params = self._compiled_kernel.num_params()
            initial_params = np.ones(num_params, dtype=np.float32)
            if initial_lengthscales is not None:
                # After ARD transformation, num_params = d + 1 (d lengthscales + outputscale)
                n_ls = num_params - 1 if num_params > 1 else num_params
                if len(initial_lengthscales) != n_ls:
                    raise ValueError(
                        f"initial_lengthscales must have shape ({n_ls},), "
                        f"got ({len(initial_lengthscales)},). "
                        f"Kernel {self._compiled_kernel.to_mojo_type()} expects {n_ls} "
                        f"lengthscale parameters."
                    )
                initial_params[:n_ls] = np.asarray(initial_lengthscales, dtype=np.float32)
            else:
                # Fill all non-outputscale params as initial_lengthscale
                for i in range(num_params - 1):
                    initial_params[i] = initial_lengthscale
            # Set outputscale (last param) if num_params > 0
            if num_params > 0:
                initial_params[-1] = initial_outputscale

        # Handle active_dims: compute column permutation so sub-kernels get contiguous slices
        self._dim_permutation = None
        compiled_dim = self._cont_dim if self._is_mixed else d
        if self._compiled_kernel.has_active_dims():
            self._validate_active_dims_bounds(self._compiled_kernel, compiled_dim)
            from .kernel import compute_dim_permutation

            perm, effective_d = compute_dim_permutation(
                self._compiled_kernel, compiled_dim
            )
            self._dim_permutation = perm
            compiled_dim = effective_d

        if self._is_mixed:
            X_cont, C = self._split_data(X)
            if self._dim_permutation is not None:
                X_cont = X_cont[:, self._dim_permutation].astype(np.float32)
            return self._fit_mixed_composite(
                X,
                X_cont,
                C,
                Y,
                n,
                compiled_dim,
                T,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                init_params=initial_params,
                init_noise=initial_noise,
                init_noise_per_task=initial_noise_per_task,
                init_outputscale=initial_outputscale,
                verbose=verbose,
                early_stop_tol=early_stop_tol,
                early_stop_patience=early_stop_patience,
                use_fused_kernels=use_fused_kernels,
                use_cosine_lr=use_cosine_lr,
                progress_adapter=progress_adapter,
                progress_interval=progress_interval,
            )

        if self._dim_permutation is not None:
            X = X[:, self._dim_permutation].astype(np.float32)

        # All kernels go through JIT engine via _fit_composite
        return self._fit_composite(
            X,
            Y,
            n,
            compiled_dim,
            T,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            init_params=initial_params,
            init_noise=initial_noise,
            init_noise_per_task=initial_noise_per_task,
            observation_noise=observation_noise_train,
            verbose=verbose,
            early_stop_tol=early_stop_tol,
            early_stop_patience=early_stop_patience,
            use_fused_kernels=use_fused_kernels,
            use_cosine_lr=use_cosine_lr,
            progress_adapter=progress_adapter,
            progress_interval=progress_interval,
        )

    def _fit_composite(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n: int,
        d: int,
        T: int,
        max_iterations: int = 100,
        learning_rate: float = 0.05,
        init_params: Optional[np.ndarray] = None,
        init_noise: float = 0.1,
        init_noise_per_task: Optional[np.ndarray] = None,
        observation_noise: Optional[np.ndarray] = None,
        verbose: bool = False,
        early_stop_tol: float = 1e-4,
        early_stop_patience: int = 15,
        use_fused_kernels: bool = True,
        use_cosine_lr: bool = False,
        progress_adapter=None,
        progress_interval: int = 1,
    ) -> MultiOutputTrainingResult:
        """Train with a composite kernel via JIT-compiled module.

        Args:
            X: Training inputs [n, d], float32
            Y: Training targets [n, T], float32
            n, d, T: Dimensions
            max_iterations: Maximum training iterations
            learning_rate: Adam learning rate
            init_params: Initial kernel parameters. If None, all set to 1.0.
            init_noise: Initial noise variance
            verbose: Print training progress
            use_fused_kernels: Whether to use fused GPU kernels (default True)

        Returns:
            MultiOutputTrainingResult
        """
        assert isinstance(self._compiled_kernel, KernelNode)

        # Ensure JIT engine and kernel module are compiled
        self._ensure_compiled(d, num_tasks=T)

        num_params = self._compiled_kernel.num_params()

        # Default initial params: all 1.0
        if init_params is None:
            init_params = np.ones(num_params, dtype=np.float32)
        else:
            init_params = np.ascontiguousarray(init_params, dtype=np.float32)
            if init_params.shape != (num_params,):
                raise ValueError(
                    f"init_params must have shape ({num_params},), "
                    f"got {init_params.shape}. "
                    f"Kernel {self._compiled_kernel.to_mojo_type()} has {num_params} parameters: "
                    f"{self._compiled_kernel.get_param_names()}"
                )

        # Init provider via kernel module
        engine_init_params = self._compiled_kernel.to_engine_params(init_params)
        self._destroy_persistent_provider()
        provider_info = self._build_provider_info(X, engine_init_params, 0.0)
        self._provider_info = provider_info
        register_provider_lease(
            self._kernel_module, self, self._destroy_persistent_provider
        )
        self._training_method = self.method

        # ConstantMean: compute init_mean_per_task
        if self._init_mean is None:
            init_mean_per_task = np.mean(Y, axis=0).astype(np.float32)
        elif isinstance(self._init_mean, (int, float)):
            init_mean_per_task = np.full(T, float(self._init_mean), dtype=np.float32)
        else:
            init_mean_per_task = np.asarray(self._init_mean, dtype=np.float32)

        # Train via JIT engine
        # Always pass an actual noise array — Mojo PythonObject `is None` can be unreliable
        if init_noise_per_task is not None:
            default_noise_per_task = np.ascontiguousarray(
                init_noise_per_task, dtype=np.float32
            )
        else:
            default_noise_per_task = np.full(T, init_noise, dtype=np.float32)
        train_args = [
            provider_info,
            Y,
            engine_init_params,
            np.ascontiguousarray(
                self._compiled_kernel.engine_trainable_mask(), dtype=np.bool_
            ),
            default_noise_per_task,
            1.0,  # init_outputscale (composite kernels manage their own scales)
            init_mean_per_task,
            int(T),
            # Args 8-11 must match Mojo order: max_iterations, learning_rate, task_rank, verbose
            int(max_iterations),
            float(learning_rate),
            int(self.task_rank),
            bool(verbose),
            # Args 11-14: CG settings (extended Mojo args)
            int(self.num_probes),
            int(self.max_cg_iter),
            float(self.cg_tol),
            int(self.precond_rank),
            int(self.precond_method),
            float(self.precond_rebuild_threshold),
            int(self.max_tridiag_iter),
            int(early_stop_patience),
            float(early_stop_tol),
            bool(use_cosine_lr),
            None
            if observation_noise is None
            else np.ascontiguousarray(observation_noise, dtype=np.float32),
        ]
        if progress_adapter is not None:
            train_args.extend([progress_adapter.callback, int(progress_interval)])
        try:
            raw = self._engine.train_multi_output(*train_args)
        except Exception as exc:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True, message=str(exc))
            raise
        if progress_adapter is not None:
            progress_adapter.close_if_needed()

        noise_per_task = np.array(raw["noise_per_task"], dtype=np.float32)
        mean_per_task = np.array(raw["mean_per_task"], dtype=np.float32)
        T = int(raw["num_tasks"])
        params = self._compiled_kernel.from_engine_params(
            np.array(raw["params"], dtype=np.float32)
        )
        outputscale = float(raw["outputscale"])

        # Reconstruct B matrix from B_flat and eigendecompose for prediction
        B_flat = np.array(raw["B_flat"], dtype=np.float32)
        B = (
            B_flat.reshape(T, T)
            if len(B_flat) == T * T
            else np.eye(T, dtype=np.float32)
        )
        B_scaled = outputscale * B
        eigvals, Q = np.linalg.eigh(B_scaled)
        Lambda = np.maximum(eigvals, 1e-6).astype(np.float32)
        effective_scales = Lambda

        alpha_rotated = np.zeros((1,), dtype=np.float32)
        alpha_raw = raw.get("alpha", None)
        if alpha_raw is not None:
            alpha_blocked = np.array(alpha_raw, dtype=np.float32)
            if alpha_blocked.size == n * T:
                alpha_nt = alpha_blocked.reshape(T, n).T
                alpha_rotated = (
                    alpha_nt.astype(np.float64) @ Q.astype(np.float64)
                ).astype(np.float32)

        self._result = MultiOutputTrainingResult(
            params=params,
            noise=float(np.mean(noise_per_task)),
            noise_per_task=noise_per_task,
            final_nll=float(raw["final_nll"]),
            iterations=int(raw["iterations"]),
            converged=bool(raw["converged"]),
            num_tasks=T,
            task_rank=int(raw["task_rank"]),
            num_kernel_params=len(params),
            B=B_scaled.astype(np.float32),
            Q=Q.astype(np.float32),
            Lambda=Lambda,
            effective_scales=effective_scales,
            W=np.zeros((T, max(int(raw["task_rank"]), 1)), dtype=np.float32),
            raw_v=np.zeros(T, dtype=np.float32),
            alpha_rotated=alpha_rotated,
            mean_per_task=mean_per_task,
            nll_history=_normalize_nll_history(
                raw.get("nll_history"),
                final_nll=float(raw["final_nll"]),
                iterations=int(raw["iterations"]),
            ),
            param_names=self._compiled_kernel.get_param_names(),
            iter_times_ms=np.array(raw.get("iter_times_ms", []), dtype=np.float64),
        )

        self._X_train = X
        self._X_train_cont = X
        self._C_train = None
        self._Y_train = Y
        self._observation_noise_train = observation_noise
        self._raw_result = raw
        self._backend_train_info = build_backend_train_info(raw, self.method)
        if self._backend_train_info is not None:
            self._backend_train_info["use_preconditioner"] = bool(
                self.use_preconditioner
            )
        self._maybe_attach_specialization_metadata(self._backend_train_info)
        self._is_trained = True
        self._fitted_mean = mean_per_task.copy()
        self._backend_predict_info = None
        self._backend_sample_info = None
        self._clear_predict_cache()

        return self._result

    def _fit_mixed_composite(
        self,
        X_full: np.ndarray,
        X_cont: np.ndarray,
        C: np.ndarray,
        Y: np.ndarray,
        n: int,
        d: int,
        T: int,
        max_iterations: int = 100,
        learning_rate: float = 0.05,
        init_params: Optional[np.ndarray] = None,
        init_noise: float = 0.1,
        init_noise_per_task: Optional[np.ndarray] = None,
        init_outputscale: float = 1.0,
        verbose: bool = False,
        early_stop_tol: float = 1e-4,
        early_stop_patience: int = 15,
        use_fused_kernels: bool = True,
        use_cosine_lr: bool = False,
        progress_adapter=None,
        progress_interval: int = 1,
    ) -> MultiOutputTrainingResult:
        """Train a mixed continuous+categorical multi-output GP via the JIT engine."""
        assert isinstance(self._compiled_kernel, KernelNode)
        assert self._cat_specs is not None

        self._ensure_compiled(d, num_tasks=T, fresh_load=False)

        num_params = self._compiled_kernel.num_params()
        if init_params is None:
            init_params = self._compiled_kernel.get_initial_params()
        else:
            init_params = np.ascontiguousarray(init_params, dtype=np.float32)
            if init_params.shape != (num_params,):
                raise ValueError(
                    f"init_params must have shape ({num_params},), got {init_params.shape}. "
                    f"Kernel {self._compiled_kernel.to_mojo_type()} has {num_params} parameters: "
                    f"{self._compiled_kernel.get_param_names()}"
                )

        engine_init_params = self._compiled_kernel.to_engine_params(init_params)
        self._destroy_persistent_provider()
        provider_info = self._build_provider_info(X_cont, engine_init_params, 0.0)
        self._provider_info = provider_info
        register_provider_lease(
            self._kernel_module, self, self._destroy_persistent_provider
        )
        self._training_method = self.method

        if self._init_mean is None:
            init_mean_per_task = np.mean(Y, axis=0).astype(np.float32)
        elif isinstance(self._init_mean, (int, float)):
            init_mean_per_task = np.full(T, float(self._init_mean), dtype=np.float32)
        else:
            init_mean_per_task = np.asarray(self._init_mean, dtype=np.float32)

        if init_noise_per_task is not None:
            default_noise_per_task = np.ascontiguousarray(
                init_noise_per_task, dtype=np.float32
            )
        else:
            default_noise_per_task = np.full(T, init_noise, dtype=np.float32)

        cat_init_params = build_default_categorical_raw_params(self._cat_specs)

        method_int = 1 if self.method == "materialized" else 0
        train_args = [
            provider_info,
            Y,
            engine_init_params,
            default_noise_per_task,
            float(init_outputscale),
            init_mean_per_task,
            int(T),
            C,
            self._cat_specs,
            cat_init_params,
            int(max_iterations),
            float(learning_rate),
            int(self.task_rank),
            bool(verbose),
            int(self.num_probes),
            int(self.max_cg_iter),
            float(self.cg_tol),
            int(self.precond_rank),
            int(self.precond_method),
            float(self.precond_rebuild_threshold),
            int(method_int),
            int(self.max_tridiag_iter),
            int(early_stop_patience),
            float(early_stop_tol),
            bool(use_cosine_lr),
        ]
        if progress_adapter is not None:
            train_args.extend([progress_adapter.callback, int(progress_interval)])
        try:
            raw = self._engine.train_multi_output_mixed(*train_args)
        except Exception as exc:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True, message=str(exc))
            raise
        if progress_adapter is not None:
            progress_adapter.close_if_needed()

        noise_per_task = np.array(raw["noise_per_task"], dtype=np.float32)
        mean_per_task = np.array(raw["mean_per_task"], dtype=np.float32)
        T = int(raw["num_tasks"])
        cont_param_kernel = _continuous_param_kernel(
            self._compiled_kernel, self._analysis
        )
        params = cont_param_kernel.from_engine_params(
            np.array(raw["params"], dtype=np.float32)
        )
        outputscale = float(raw["outputscale"])
        cat_params = np.asarray(raw.get("cat_params", []), dtype=np.float32)

        B_flat = np.array(raw["B_flat"], dtype=np.float32)
        B = (
            B_flat.reshape(T, T)
            if len(B_flat) == T * T
            else np.eye(T, dtype=np.float32)
        )
        B_scaled = outputscale * B
        eigvals, Q = np.linalg.eigh(B_scaled)
        Lambda = np.maximum(eigvals, 1e-6).astype(np.float32)
        effective_scales = Lambda

        alpha_rotated = np.zeros((1,), dtype=np.float32)
        alpha_raw = raw.get("alpha", None)
        if alpha_raw is not None:
            alpha_blocked = np.array(alpha_raw, dtype=np.float32)
            if alpha_blocked.size == n * T:
                alpha_nt = alpha_blocked.reshape(T, n).T
                alpha_rotated = (
                    alpha_nt.astype(np.float64) @ Q.astype(np.float64)
                ).astype(np.float32)

        cat_param_names: List[str] = []
        if self._analysis is not None:
            for spec in self._analysis.categorical_specs:
                cat_param_names.extend(spec.param_names)

        self._result = MultiOutputTrainingResult(
            params=params,
            noise=float(np.mean(noise_per_task)),
            noise_per_task=noise_per_task,
            final_nll=float(raw["final_nll"]),
            iterations=int(raw["iterations"]),
            converged=bool(raw["converged"]),
            num_tasks=T,
            task_rank=int(raw["task_rank"]),
            num_kernel_params=len(params),
            B=B_scaled.astype(np.float32),
            Q=Q.astype(np.float32),
            Lambda=Lambda,
            effective_scales=effective_scales,
            W=np.zeros((T, max(int(raw["task_rank"]), 1)), dtype=np.float32),
            raw_v=np.zeros(T, dtype=np.float32),
            alpha_rotated=alpha_rotated,
            mean_per_task=mean_per_task,
            nll_history=_normalize_nll_history(
                raw.get("nll_history"),
                final_nll=float(raw["final_nll"]),
                iterations=int(raw["iterations"]),
            ),
            param_names=cont_param_kernel.get_param_names(),
            cat_params=cat_params if cat_params.size > 0 else None,
            cat_param_names=cat_param_names,
            iter_times_ms=np.array(raw.get("iter_times_ms", []), dtype=np.float64),
        )

        self._X_train = X_full
        self._X_train_cont = X_cont
        self._C_train = C
        self._Y_train = Y
        self._raw_result = raw
        self._backend_train_info = build_backend_train_info(raw, self.method)
        if self._backend_train_info is not None:
            self._backend_train_info["use_preconditioner"] = bool(
                self.use_preconditioner
            )
        self._maybe_attach_specialization_metadata(self._backend_train_info)
        self._is_trained = True
        self._fitted_mean = mean_per_task.copy()
        self._backend_predict_info = None
        self._backend_sample_info = None
        self._clear_predict_cache()

        return self._result

    def _predict_mixed_composite(
        self,
        X_test: np.ndarray,
        compute_variance: bool = True,
        variance_method: int = 0,
    ) -> Dict[str, Any]:
        """Predict with a mixed continuous+categorical multi-output GP."""
        assert isinstance(self._result, MultiOutputTrainingResult)
        assert self._engine is not None
        assert self._kernel_module is not None
        assert self._C_train is not None
        assert self._cat_specs is not None

        X_test_cont, C_test = self._split_data(X_test)
        if getattr(self, "_dim_permutation", None) is not None:
            X_test_cont = X_test_cont[:, self._dim_permutation].astype(np.float32)

        result = self._result
        cont_param_kernel = _continuous_param_kernel(
            self._compiled_kernel, self._analysis
        )
        backend_params = cont_param_kernel.to_engine_params(
            np.ascontiguousarray(result.params, dtype=np.float32)
        )
        provider_params = np.ascontiguousarray(result.params, dtype=np.float32)
        provider_info = self._provider_info
        built_provider_info = None
        mean_noise = float(np.mean(result.noise_per_task))
        if provider_info is None:
            provider_info = self._build_provider_info(
                self._X_train_cont,
                provider_params,
                mean_noise,
            )
            built_provider_info = provider_info
        update_provider_noise(provider_info, mean_noise)
        predict_rank = _prediction_lanczos_rank(
            self.max_tridiag_iter,
            self.method,
            is_mixed=True,
            is_ard=self.ard,
        )
        predict_cg_tol = (
            min(float(self.cg_tol), 1e-2) if float(self.cg_tol) > 0.0 else 1e-2
        )

        def _run_predict(var_method: int) -> Dict[str, Any]:
            cat_params = np.ascontiguousarray(
                categorical_prediction_params(
                    self._cat_specs,
                    result.cat_params
                    if result.cat_params is not None
                    else np.zeros(1, dtype=np.float32),
                ),
                dtype=np.float32,
            )
            return self._engine.predict_multi_output_mixed(
                provider_info,
                np.ascontiguousarray(result.alpha_rotated, dtype=np.float32),
                np.ascontiguousarray(result.Q, dtype=np.float32),
                np.ascontiguousarray(result.effective_scales, dtype=np.float32),
                X_test_cont.astype(np.float32),
                backend_params,
                float(np.mean(result.noise_per_task)),
                self._C_train.astype(np.int32),
                C_test.astype(np.int32),
                cat_params,
                self._cat_specs,
                var_method if compute_variance else 0,
                self.max_cg_iter,
                predict_cg_tol,
                self.precond_rank,
                predict_rank,
                1 if self.method == "materialized" else 0,
            )

        try:
            pred = _run_predict(variance_method)
            variance_route = "predict_multi_output_mixed" if compute_variance else None
            self._backend_predict_info = build_backend_predict_info(
                requested_method=self.method,
                actual_prediction_route="predict_multi_output_mixed",
                backend_prediction_used=True,
                backend_variance_used=compute_variance,
                variance_method={0: "mean_only", 1: "love", 2: "exact"}.get(
                    variance_method, "mean_only"
                ),
                fallback_used=False,
                actual_variance_route=variance_route,
                training_route=(
                    self._backend_train_info.get("training_route")
                    if self._backend_train_info is not None
                    else self.method
                ),
                precond_rank=self.precond_rank,
                precond_method=self.precond_method,
                precond_rebuild_count=(
                    self._backend_train_info.get("precond_rebuild_count")
                    if self._backend_train_info is not None
                    else None
                ),
            )
            self._maybe_attach_specialization_metadata(self._backend_predict_info)

            mean_all = np.array(pred["mean"], dtype=np.float32, copy=True)
            var_all = np.maximum(
                np.asarray(pred.get("variance", 0), dtype=np.float32), 0
            )
            if (
                compute_variance
                and variance_method == 1
                and not np.all(np.isfinite(var_all))
            ):
                warnings.warn(
                    "Mixed multi-output LOVE variance produced non-finite values; "
                    "falling back to exact variance for this prediction.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                pred = _run_predict(2)
                var_all = np.maximum(
                    np.asarray(pred.get("variance", 0), dtype=np.float32), 0
                )
                if self._backend_predict_info is not None:
                    self._backend_predict_info["actual_variance_route"] = (
                        "predict_multi_output_mixed_exact_retry"
                    )
                    self._backend_predict_info["backend_variance_used"] = True
        finally:
            if built_provider_info is not None:
                destroy_provider_info(self._kernel_module, built_provider_info)
        return {"mean": mean_all, "variance": var_all, "has_variance": compute_variance}

    def _predict_composite(
        self,
        X_test: np.ndarray,
        compute_variance: bool = True,
        variance_method: int = 0,
    ) -> Dict[str, Any]:
        """Predict with a composite kernel via Kronecker mean/variance.

        When rotated posterior state is available from training, mean prediction
        uses the JIT engine's multi-output Kronecker binding. Variance uses the
        same binding when available and falls back to the existing per-task
        single-output path otherwise.

        Mean offset is applied on the Python side in predict(), so we do NOT
        pass mean_per_task here to avoid double-application.
        """
        assert isinstance(self._result, MultiOutputTrainingResult)
        assert self._engine is not None
        assert self._kernel_module is not None

        if self._is_mixed:
            return self._predict_mixed_composite(
                X_test, compute_variance, variance_method
            )

        # Apply active_dims permutation to test inputs
        if getattr(self, "_dim_permutation", None) is not None:
            X_test = X_test[:, self._dim_permutation].astype(np.float32)

        result = self._result
        T = result.num_tasks
        m = X_test.shape[0]
        predict_rank = _prediction_lanczos_rank(
            self.max_tridiag_iter,
            self.method,
            is_mixed=False,
            is_ard=self.ard,
        )

        backend_params = self._compiled_kernel.to_engine_params(
            np.ascontiguousarray(result.params, dtype=np.float32)
        )
        self._backend_predict_info = build_backend_predict_info(
            requested_method=self.method,
            actual_prediction_route="predict_multi_output",
            backend_prediction_used=True,
            backend_variance_used=compute_variance,
            variance_method={0: "mean_only", 1: "love", 2: "exact"}.get(
                variance_method, "mean_only"
            ),
            fallback_used=False,
            actual_variance_route="predict_multi_output" if compute_variance else None,
            training_route=(
                self._backend_train_info.get("training_route")
                if self._backend_train_info is not None
                else self.method
            ),
            precond_rank=self.precond_rank,
            precond_method=self.precond_method,
            precond_rebuild_count=(
                self._backend_train_info.get("precond_rebuild_count")
                if self._backend_train_info is not None
                else None
            ),
        )
        self._maybe_attach_specialization_metadata(self._backend_predict_info)

        provider_info = self._provider_info
        built_provider_info = None
        if provider_info is None:
            provider_info = self._build_provider_info(
                self._X_train_cont,
                backend_params,
                float(np.mean(result.noise_per_task)),
            )
            built_provider_info = provider_info
        predict_cg_tol = (
            min(float(self.cg_tol), 1e-2) if float(self.cg_tol) > 0.0 else 1e-2
        )

        def _run_multi_output_predict(var_method: int) -> Dict[str, Any]:
            return self._engine.predict_multi_output(
                provider_info,
                np.ascontiguousarray(result.alpha_rotated, dtype=np.float32),
                np.ascontiguousarray(result.Q, dtype=np.float32),
                np.ascontiguousarray(result.effective_scales, dtype=np.float32),
                X_test.astype(np.float32),
                backend_params,
                float(np.mean(result.noise_per_task)),
                var_method,
                self.max_cg_iter,
                predict_cg_tol,
                self.precond_rank,
                predict_rank,
            )

        try:
            mean_raw = _run_multi_output_predict(0)
            mean_all = np.array(mean_raw["mean"], dtype=np.float32, copy=True)

            var_all = np.zeros((m, T), dtype=np.float32)
            if compute_variance:
                variance_raw = _run_multi_output_predict(variance_method)
                if "variance" in variance_raw:
                    var_all = np.maximum(
                        np.array(
                            variance_raw["variance"], dtype=np.float32, copy=True
                        ),
                        0,
                    )
                    update_provider_noise(
                        provider_info, float(np.mean(result.noise_per_task))
                    )
                else:
                    if self._backend_predict_info is not None:
                        self._backend_predict_info["actual_variance_route"] = (
                            "predict_per_task"
                        )
                    Y_train = self._Y_train
                    for t in range(T):
                        y_t = Y_train[:, t] if Y_train.ndim > 1 else Y_train
                        noise_t = float(result.noise_per_task[t])
                        mean_t = (
                            float(self._fitted_mean[t])
                            if self._fitted_mean is not None
                            else 0.0
                        )

                        update_provider_noise(provider_info, noise_t)
                        pred = self._engine.predict(
                            provider_info,
                            y_t.astype(np.float32),
                            X_test.astype(np.float32),
                            backend_params,
                            noise_t,
                            mean_t,
                            variance_method,
                            self.max_cg_iter,
                            float(self.cg_tol),
                            self.precond_rank,
                            predict_rank,
                        )
                        if "variance" in pred:
                            var_all[:, t] = np.maximum(
                                np.asarray(pred["variance"], dtype=np.float32), 0
                            )
                    update_provider_noise(
                        provider_info, float(np.mean(result.noise_per_task))
                    )

        finally:
            if built_provider_info is not None:
                destroy_provider_info(self._kernel_module, built_provider_info)

        return {
            "mean": mean_all,
            "variance": var_all,
            "has_variance": compute_variance,
            "mean_has_offset": False,
        }

    def predict(
        self,
        X_test: np.ndarray,
        return_var: bool = False,
        return_std: bool = False,
        variance_method: str = "love",
        progress: Any = None,
        progress_stats: Optional[Any] = None,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        MultiOutputPredictionResult,
    ]:
        """Predict at test points.

        Args:
            X_test: Test inputs [m, d], float32
            return_var: If True, return (mean, variance) tuple
            return_std: If True, return (mean, std) tuple
            variance_method: Variance computation method. One of:
                - 'love' (default): Fast low-rank approximation via LOVE/Lanczos
                - 'exact': Exact CG-based variance (preconditioned, more accurate but slower)
            progress: Progress reporting control. Use True, ``"auto"``, a callback,
                or a reporter object with start/update/close methods.
            progress_stats: Optional stat names or callable used by the default
                tqdm reporter.

        Returns:
            If return_var: (mean [m,T], variance [m,T])
            If return_std: (mean [m,T], std [m,T])
            Otherwise: MultiOutputPredictionResult
        """
        if not self._is_trained:
            raise RuntimeError(
                "GP must be trained before prediction. Call fit() first."
            )

        _VALID_VARIANCE_METHODS = ("love", "exact", "mean_only")
        if variance_method not in _VALID_VARIANCE_METHODS:
            raise ValueError(
                f"variance_method must be one of {_VALID_VARIANCE_METHODS}, "
                f"got '{variance_method}'"
            )
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        if X_test.ndim != 2:
            raise ValueError(f"X_test must be 2D [m, d], got shape {X_test.shape}")
        if X_test.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X_test has {X_test.shape[1]} features, "
                f"expected {self._X_train.shape[1]}"
            )
        if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
            raise ValueError("X_test contains NaN or Inf values")

        surface = surface_for_icm(self._is_mixed)
        variance_feature = {
            "mean_only": "mean_only",
            "exact": "exact_variance",
            "love": "love_variance",
        }[variance_method]
        check_feature_support(TABLE_PREDICTION, surface, variance_feature, stacklevel=2)

        train_info = self._backend_train_info or {}
        prediction_route = train_info.get("training_route", self._training_method or self.method)
        progress_adapter = resolve_progress_adapter(
            progress,
            operation="predict",
            model="multi_output_icm",
            route=prediction_route,
            progress_stats=progress_stats,
        )
        progress_total = 3
        if progress_adapter is not None:
            base_stats = prediction_progress_stats(
                n_test=X_test.shape[0],
                variance_method=variance_method,
            )
            progress_adapter.emit(
                phase="start",
                current=0,
                total=progress_total,
                stats=base_stats,
            )

        try:
            # When returning a MultiOutputPredictionResult (neither return_var nor
            # return_std), always compute variance so the result object is complete.
            # When returning a tuple via return_var/return_std, variance is always needed.
            compute_variance = variance_method != "mean_only"
            # JIT engine contract: 0=mean_only, 1=LOVE, 2=exact
            if variance_method == "exact":
                var_method_int = 2
            elif variance_method == "love" or variance_method is None:
                var_method_int = 1
            elif variance_method == "mean_only":
                var_method_int = 0
            else:
                var_method_int = 0

            pred = self._lookup_cached_prediction(X_test, var_method_int)
            cache_hit = pred is not None
            if progress_adapter is not None:
                phase_stats = prediction_progress_stats(
                    n_test=X_test.shape[0],
                    variance_method=variance_method,
                )
                phase_stats["prediction_cache_used"] = cache_hit
                progress_adapter.emit(
                    phase="cache" if cache_hit else "backend",
                    current=1,
                    total=progress_total,
                    message="Using cached prediction" if cache_hit else "Running backend prediction",
                    stats=phase_stats,
                )
            if pred is None:
                pred = self._predict_composite(
                    X_test, compute_variance, variance_method=var_method_int
                )
                self._store_cached_prediction(X_test, var_method_int, pred)
            mean = np.array(pred["mean"], dtype=np.float32)
            # ConstantMean: add per-task mean offset
            if self._fitted_mean is not None and not bool(
                pred.get("mean_has_offset", False)
            ):
                mean = mean + self._fitted_mean[np.newaxis, :]
            has_var = bool(pred.get("has_variance", False))

            if has_var:
                variance = np.array(pred["variance"], dtype=np.float32)
                variance = np.maximum(variance, 0)
            else:
                variance = None

            final_stats = prediction_progress_stats(
                n_test=X_test.shape[0],
                variance_method=variance_method,
                backend_info=None if cache_hit else self._backend_predict_info,
            )
            final_stats["prediction_cache_used"] = cache_hit
            if progress_adapter is not None:
                progress_adapter.emit(
                    phase="variance" if variance_method != "mean_only" else "mean",
                    current=2,
                    total=progress_total,
                    stats=final_stats,
                )

            if return_var:
                final_result = (mean, variance)
            elif return_std:
                std = np.sqrt(variance) if variance is not None else None
                final_result = (mean, std)
            else:
                final_result = MultiOutputPredictionResult(
                    mean=mean,
                    variance=variance,
                    std=np.sqrt(variance) if variance is not None else None,
                )
        except Exception as exc:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True, message=str(exc))
            raise

        if progress_adapter is not None:
            progress_adapter.emit(
                phase="complete",
                current=progress_total,
                total=progress_total,
                stats=final_stats,
            )
        return final_result

    def predict_latent(
        self,
        X_test: np.ndarray,
        return_var: bool = False,
        return_std: bool = False,
        variance_method: str = "love",
        **kwargs,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        MultiOutputPredictionResult,
    ]:
        """Predict the latent task functions ``p(f_test | y_train)``."""
        return self.predict(
            X_test,
            return_var=return_var,
            return_std=return_std,
            variance_method=variance_method,
            **kwargs,
        )

    def predict_observed(
        self,
        X_test: np.ndarray,
        observation_noise: Optional[np.ndarray] = None,
        noise_group_test: Optional[np.ndarray] = None,
        return_var: bool = False,
        return_std: bool = False,
        variance_method: str = "love",
        **kwargs,
    ) -> Union[Tuple[np.ndarray, np.ndarray], MultiOutputPredictionResult]:
        """Predict observed task responses ``p(y_test | y_train)``.

        Observed prediction requires explicit test-point noise. MojoGP never
        reuses or averages training noise for new points.
        """
        latent = self.predict(
            X_test,
            return_var=False,
            return_std=False,
            variance_method=variance_method,
            **kwargs,
        )
        assert isinstance(latent, MultiOutputPredictionResult)
        noise = _resolve_observed_noise_matrix(
            latent.mean.shape[0],
            latent.mean.shape[1],
            observation_noise=observation_noise,
            noise_group_test=noise_group_test,
            group_noise=self._noise_group_values,
        )
        return _format_observed_prediction(latent, noise, return_var, return_std)

    def score(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute prediction metrics on test data.

        Args:
            X_test: Test inputs [m, d], float32
            Y_test: Test targets [m, T], float32

        Returns:
            Dictionary with:
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - r2: R-squared (coefficient of determination)
            - rmse_per_task: [T] RMSE per task
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before scoring. Call fit() first.")

        Y_test = np.ascontiguousarray(Y_test, dtype=np.float32)
        pred = self.predict(X_test)
        mean = pred.mean if isinstance(pred, MultiOutputPredictionResult) else pred

        residuals = Y_test - mean
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-10))

        # Per-task RMSE
        rmse_per_task = np.sqrt(np.mean(residuals**2, axis=0))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "rmse_per_task": rmse_per_task,
        }

    def sample_posterior(
        self,
        X_test: np.ndarray,
        n_samples: int = 1,
        method: str = "diagonal",
        n_rff_features: int = 1024,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw samples from the posterior predictive distribution.

        Args:
            X_test: Test points [m, d]
            n_samples: Number of posterior samples to draw
            method: Sampling method:
                - 'diagonal' (default): Independent samples using predictive std.
                  Fast, O(m), but ignores correlations between test points.
                - 'pathwise': Approximate correlated posterior samples using an
                  explicit feature-map prior sampler plus backend Kronecker
                  correction. Supports current continuous kernel trees except
                  polynomial, plus supported mixed continuous-categorical trees.
            n_rff_features: Number of random Fourier features used by the
                pathwise prior sampler.
            rng: Optional numpy random Generator for reproducibility.

        Returns:
            samples: Array of shape [n_samples, m, T]

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before sampling. Call fit() first.")
        surface = surface_for_icm(self._is_mixed)
        if method == "cholesky":
            check_feature_support(
                TABLE_SAMPLING, surface, "cholesky_sampling", stacklevel=2
            )
        if method not in ("diagonal", "pathwise"):
            raise ValueError(
                f"MultiOutputGP only supports method='diagonal' or 'pathwise', got '{method}'."
            )

        if rng is None:
            rng = np.random.default_rng()

        requested_method = method
        train_info = self._backend_train_info or {}
        training_route = train_info.get(
            "training_route", self._training_method or self.method
        )

        if method == "pathwise":
            check_feature_support(
                TABLE_SAMPLING, surface, "pathwise_sampling", stacklevel=2
            )
            if kernel_tree_contains_kernel_name(self._compiled_kernel, "POLYNOMIAL"):
                check_feature_support(
                    TABLE_SAMPLING, surface, "polynomial_pathwise", stacklevel=2
                )
            self._pathwise_bundle_role = None
            samples = self._sample_posterior_pathwise(
                X_test,
                n_samples=n_samples,
                n_rff_features=n_rff_features,
                rng=rng,
            )
            self._backend_sample_info = {
                "requested_method": requested_method,
                "actual_sampling_method": "pathwise",
                "actual_sampling_route": "provider_pathwise",
                "backend_sampling_used": True,
                "backend_correction_used": True,
                "backend_correction_route": (
                    "sample_multi_output_mixed_pathwise"
                    if self._is_mixed
                    else "sample_multi_output_pathwise"
                ),
                "training_route": training_route,
                "prior_sampler_family": "shared_feature_map",
                "n_rff_features": int(n_rff_features),
            }
            if self._pathwise_bundle_role is not None:
                self._backend_sample_info["provider_bundle_role"] = (
                    self._pathwise_bundle_role
                )
            return samples

        check_feature_support(
            TABLE_SAMPLING, surface, "diagonal_sampling", stacklevel=2
        )
        mean, std = self.predict(X_test, return_std=True)
        # mean: [m, T], std: [m, T]
        z = rng.standard_normal((n_samples, mean.shape[0], mean.shape[1]))
        samples = mean[np.newaxis, :, :] + std[np.newaxis, :, :] * z
        self._backend_sample_info = {
            "requested_method": requested_method,
            "actual_sampling_method": "diagonal",
            "actual_sampling_route": "diagonal_from_predictive_std",
            "backend_sampling_used": True,
            "backend_correction_used": False,
            "training_route": training_route,
        }
        return samples.astype(np.float32)

    def _sample_posterior_pathwise(
        self,
        X_test: np.ndarray,
        n_samples: int,
        n_rff_features: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Approximate correlated posterior samples via backend pathwise correction."""
        result = self._result
        if not isinstance(result, MultiOutputTrainingResult):
            raise NotImplementedError(
                "Pathwise posterior sampling currently requires the current JIT "
                "composite training state. Re-fit the model with the current "
                "MultiOutputGP path first."
            )

        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        if self._is_mixed:
            assert self._X_train_cont is not None and self._C_train is not None
            X_test_cont, C_test = self._split_data(X_test)
            if getattr(self, "_dim_permutation", None) is not None:
                X_test_cont = X_test_cont[:, self._dim_permutation].astype(np.float32)
            X_train = np.ascontiguousarray(self._X_train_cont, dtype=np.float32)
            C_train = np.ascontiguousarray(self._C_train, dtype=np.int32)
            cat_params = np.ascontiguousarray(
                categorical_prediction_params(
                    self._cat_specs,
                    result.cat_params
                    if result.cat_params is not None
                    else np.zeros(0, dtype=np.float32),
                ),
                dtype=np.float32,
            )
            cat_col_map = {col: idx for idx, col in enumerate(self._cat_col_indices)}
        else:
            X_test_cont = X_test
            if getattr(self, "_dim_permutation", None) is not None:
                X_test_cont = X_test_cont[:, self._dim_permutation].astype(np.float32)
            assert self._X_train_cont is not None
            X_train = np.ascontiguousarray(self._X_train_cont, dtype=np.float32)
            C_train = None
            C_test = None
            cat_params = None
            cat_col_map = {}

        T = int(result.num_tasks)
        n = int(X_train.shape[0])
        m = int(X_test_cont.shape[0])

        self._ensure_compiled(X_train.shape[1], num_tasks=T)
        if self._kernel_module is None or self._engine is None:
            raise RuntimeError(
                "Backend provider state is unavailable for pathwise sampling."
            )
        backend_name = (
            "sample_multi_output_mixed_pathwise"
            if self._is_mixed
            else "sample_multi_output_pathwise"
        )
        if not hasattr(self._engine, backend_name):
            raise RuntimeError(
                f"The loaded JIT engine does not expose {backend_name}(). "
                "Rebuild it with `task build`."
            )
        if _kernel_tree_contains_type(self._compiled_kernel, KernelType.POLYNOMIAL):
            raise NotImplementedError(
                "MultiOutputGP pathwise sampling does not yet support polynomial kernels. "
                "Use MultiOutputLMCGP or method='diagonal'."
            )

        params = np.ascontiguousarray(result.params, dtype=np.float32)
        engine_params = (
            params
            if self._is_mixed
            else np.ascontiguousarray(
                self._compiled_kernel.to_engine_params(params), dtype=np.float32
            )
        )
        noise_per_task = np.ascontiguousarray(result.noise_per_task, dtype=np.float32)
        task_cov = np.ascontiguousarray(result.B, dtype=np.float32)
        solve_max_cg_iter = max(int(self.max_cg_iter), 100)
        solve_cg_tol = (
            min(float(self.cg_tol), 1e-2) if float(self.cg_tol) > 0.0 else 1e-2
        )
        mean_per_task = np.ascontiguousarray(
            self._fitted_mean
            if self._fitted_mean is not None
            else result.mean_per_task,
            dtype=np.float32,
        )
        y_centered = np.ascontiguousarray(
            self._Y_train.astype(np.float32) - mean_per_task[np.newaxis, :],
            dtype=np.float32,
        )
        feature_map = build_pathwise_feature_map(
            self._compiled_kernel,
            params,
            input_dim=X_train.shape[1],
            n_features=n_rff_features,
            rng=rng,
            cat_params=cat_params,
            cat_col_map=cat_col_map,
        )
        train_features = feature_map.evaluate(X_train, C_train)
        task_eigvecs = np.asarray(result.Q, dtype=np.float64)
        task_scales = np.sqrt(
            np.maximum(np.asarray(result.Lambda, dtype=np.float64), 1e-8)
        )
        chunk_size = min(max(_DEFAULT_PATHWISE_TEST_CHUNK_SIZE, 1), max(m, 1))
        samples = np.empty((n_samples, m, T), dtype=np.float32)

        provider_info = self._provider_info
        provider_is_temporary = provider_info is None
        if provider_info is None:
            provider_info = self._kernel_module.init_provider(
                X_train,
                engine_params,
                float(np.mean(noise_per_task)),
            )
        else:
            update_provider_noise(provider_info, float(np.mean(noise_per_task)))
        training_route = (self._training_method or self.method).lower()
        try:
            if training_route == "materialized" and int(
                provider_info.get("materialization_mode", 0) or 0
            ) != 1:
                self._kernel_module.materialize(provider_info)
            train_noise = np.sqrt(np.maximum(noise_per_task, 0.0)).astype(np.float32)
            for sample_idx in range(n_samples):
                weights = build_feature_weights(feature_map, T, rng)
                scalar_prior_train = sample_prior_values(
                    feature_map,
                    X_train,
                    C_train,
                    weights,
                )
                prior_train = np.einsum(
                    "ln,tl,l->nt",
                    scalar_prior_train.astype(np.float64),
                    task_eigvecs,
                    task_scales,
                ).astype(np.float32)
                obs_prior_train = (
                    prior_train
                    + rng.standard_normal((n, T)).astype(np.float32)
                    * train_noise[np.newaxis, :]
                )
                residual = np.ascontiguousarray(
                    y_centered - obs_prior_train,
                    dtype=np.float32,
                )
                for start in range(0, m, chunk_size):
                    end = min(start + chunk_size, m)
                    X_chunk = np.ascontiguousarray(
                        X_test_cont[start:end], dtype=np.float32
                    )
                    C_chunk = (
                        None
                        if C_test is None
                        else np.ascontiguousarray(C_test[start:end], dtype=np.int32)
                    )
                    scalar_prior_test = sample_prior_values(
                        feature_map,
                        X_chunk,
                        C_chunk,
                        weights,
                    )
                    prior_test = np.einsum(
                        "ln,tl,l->nt",
                        scalar_prior_test.astype(np.float64),
                        task_eigvecs,
                        task_scales,
                    ).astype(np.float32)
                    if self._is_mixed:
                        raw = self._engine.sample_multi_output_mixed_pathwise(
                            provider_info,
                            residual,
                            task_cov,
                            X_chunk,
                            engine_params,
                            noise_per_task,
                            C_train,
                            C_chunk,
                            cat_params,
                            self._cat_specs,
                            solve_max_cg_iter,
                            solve_cg_tol,
                            1 if self.method == "materialized" else 0,
                        )
                    else:
                        raw = self._engine.sample_multi_output_pathwise(
                            provider_info,
                            residual,
                            task_cov,
                            X_chunk,
                            engine_params,
                            noise_per_task,
                            solve_max_cg_iter,
                            solve_cg_tol,
                        )
                    correction = np.asarray(raw["correction"], dtype=np.float32)
                    samples[sample_idx, start:end] = (
                        mean_per_task[np.newaxis, :] + prior_test + correction
                    )

            return samples.astype(np.float32)
        finally:
            if provider_is_temporary:
                destroy_provider_info(self._kernel_module, provider_info)

    def save(self, path: str) -> None:
        """Save the trained multi-output GP model to disk.

        Args:
            path: File path (without extension). Creates {path}_config.json
                  and {path}_arrays.npz

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before saving. Call fit() first.")

        path = str(path)
        if path.endswith(".json") or path.endswith(".npz"):
            path = path.rsplit(".", 1)[0]

        tr = self._result

        config = {
            "schema_version": _MODEL_SCHEMA_VERSION,
            "mojogp_version": __version__,
            "wrapper": "MultiOutputGP",
            "kernel": str(self.kernel)
            if not self._is_composite
            else self.kernel.to_mojo_type(),
            "kernel_tree": self.kernel.to_dict()
            if isinstance(self.kernel, KernelNode)
            else None,
            "is_composite": self._is_composite,
            "task_rank": self.task_rank,
            "ard": self.ard,
            "method": self.method,
            "num_probes": self.num_probes,
            "max_cg_iter": self.max_cg_iter,
            "cg_tol": self.cg_tol,
            "use_preconditioner": self.use_preconditioner,
            "precond_rank": self.precond_rank,
            "precond": self.precond,
            "precond_rebuild_threshold": self.precond_rebuild_threshold,
            "dim": self._dim,
            "noise": float(tr.noise),
            "final_nll": float(tr.final_nll),
            "iterations": int(tr.iterations),
            "converged": bool(tr.converged),
            "num_tasks": int(tr.num_tasks),
            "task_rank_result": int(tr.task_rank),
            "training_method": self._training_method or self.method,
            "has_observation_noise_vector": self._observation_noise_train is not None,
            "has_noise_group_train": self._noise_group_train is not None,
            "has_group_noise": self._noise_group_values is not None,
        }
        if self._specialization_decision is not None:
            config["specialization"] = self._specialization_decision.to_dict()

        config["result_type"] = "MultiOutputTrainingResult"
        config["num_kernel_params"] = int(tr.num_kernel_params)
        config["param_names"] = tr.param_names
        config["cat_param_names"] = tr.cat_param_names

        with open(f"{path}_config.json", "w") as f:
            json.dump(config, f, indent=2)

        arrays = {
            "X_train": self._X_train,
            "Y_train": self._Y_train,
            "B": tr.B,
            "Q": tr.Q,
            "Lambda": tr.Lambda,
            "W": tr.W,
            "raw_v": tr.raw_v,
            "alpha_rotated": tr.alpha_rotated,
            "effective_scales": tr.effective_scales,
            "noise_per_task": tr.noise_per_task,
            "nll_history": tr.nll_history,
            "mean_per_task": tr.mean_per_task,
        }

        arrays["params"] = tr.params
        if tr.iter_times_ms is not None:
            arrays["iter_times_ms"] = tr.iter_times_ms
        if tr.cat_params is not None:
            arrays["cat_params"] = tr.cat_params

        if self._X_train_cont is not None:
            arrays["X_train_cont"] = self._X_train_cont
        if self._C_train is not None:
            arrays["C_train"] = self._C_train
        if self._observation_noise_train is not None:
            arrays["observation_noise_train"] = self._observation_noise_train
        if self._noise_group_train is not None:
            arrays["noise_group_train"] = self._noise_group_train
        if self._noise_group_values is not None:
            arrays["group_noise"] = self._noise_group_values

        # ConstantMean: save fitted mean
        if self._fitted_mean is not None:
            arrays["fitted_mean"] = self._fitted_mean

        # Save dim permutation for active_dims
        if getattr(self, "_dim_permutation", None) is not None:
            arrays["dim_permutation"] = np.array(self._dim_permutation, dtype=np.int32)

        np.savez(f"{path}_arrays.npz", **arrays)

    @classmethod
    def load(
        cls,
        path: str,
        kernel: Optional[Union[str, KernelNode]] = None,
    ) -> "MultiOutputGP":
        """Load a saved multi-output GP model from disk.

        Args:
            path: File path (without extension)
            kernel: The kernel used when saving. Required for composite kernels.

        Returns:
            Loaded MultiOutputGP with training state restored
        """
        path = str(path)
        if path.endswith(".json") or path.endswith(".npz"):
            path = path.rsplit(".", 1)[0]

        with open(f"{path}_config.json", "r") as f:
            config = json.load(f)

        wrapper = config.get("wrapper")
        if wrapper != "MultiOutputGP":
            raise ValueError(
                "Saved model is not a MultiOutputGP artifact. "
                f"Expected wrapper='MultiOutputGP', got {wrapper!r}."
            )
        schema_version = int(config.get("schema_version", 0))
        if schema_version != _MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MultiOutputGP schema_version={schema_version}; "
                f"expected {_MODEL_SCHEMA_VERSION}."
            )

        arrays = np.load(f"{path}_arrays.npz", allow_pickle=False)

        if kernel is None:
            if "kernel_tree" in config and config["kernel_tree"] is not None:
                kernel = KernelNode.from_dict(config["kernel_tree"])
            else:
                kernel = config.get("kernel", "rbf")

        gp = cls(
            kernel=kernel,
            task_rank=config["task_rank"],
            ard=config["ard"],
            num_probes=config["num_probes"],
            max_cg_iterations=config["max_cg_iter"],
            cg_tolerance=config["cg_tol"],
            use_preconditioner=config["use_preconditioner"],
            preconditioner_rank=config["precond_rank"],
            preconditioner=config["precond"],
            precond_rebuild_threshold=config["precond_rebuild_threshold"],
        )
        gp.method = config["method"]

        gp._X_train = arrays["X_train"]
        gp._Y_train = arrays["Y_train"]
        gp._observation_noise_train = (
            np.ascontiguousarray(arrays["observation_noise_train"], dtype=np.float32)
            if "observation_noise_train" in arrays
            else None
        )
        gp._noise_group_train = (
            np.ascontiguousarray(arrays["noise_group_train"], dtype=np.int32)
            if "noise_group_train" in arrays
            else None
        )
        gp._noise_group_values = (
            np.ascontiguousarray(arrays["group_noise"], dtype=np.float32)
            if "group_noise" in arrays
            else None
        )
        gp._is_trained = True
        gp._dim = config["dim"]
        gp._training_method = config["training_method"]
        gp._specialization_request = SpecializationRequest.disabled()
        gp._specialization_decision = None

        total_dim = int(gp._X_train.shape[1])
        gp._configure_kernel_for_fit(total_dim)

        result_type = config["result_type"]
        if result_type != "MultiOutputTrainingResult":
            raise ValueError(
                f"Unsupported MultiOutputGP result_type={result_type!r}; "
                "expected 'MultiOutputTrainingResult'."
            )

        gp._result = MultiOutputTrainingResult(
            params=arrays["params"],
            noise=config["noise"],
            noise_per_task=arrays["noise_per_task"],
            final_nll=config["final_nll"],
            iterations=config["iterations"],
            converged=config["converged"],
            num_tasks=config["num_tasks"],
            task_rank=config["task_rank_result"],
            num_kernel_params=config["num_kernel_params"],
            B=arrays["B"],
            Q=arrays["Q"],
            Lambda=arrays["Lambda"],
            effective_scales=arrays["effective_scales"],
            W=arrays["W"],
            raw_v=arrays["raw_v"],
            alpha_rotated=arrays["alpha_rotated"],
            mean_per_task=arrays["mean_per_task"],
            nll_history=arrays["nll_history"],
            param_names=config["param_names"],
            cat_params=(arrays["cat_params"] if "cat_params" in arrays else None),
            cat_param_names=config["cat_param_names"],
            iter_times_ms=(arrays["iter_times_ms"] if "iter_times_ms" in arrays else None),
        )

        # ConstantMean: restore fitted mean
        if "fitted_mean" in arrays:
            gp._fitted_mean = arrays["fitted_mean"]
        elif "mean_per_task" in arrays:
            gp._fitted_mean = arrays["mean_per_task"]

        # Restore dim permutation for active_dims
        gp._dim_permutation = None
        if "dim_permutation" in arrays:
            gp._dim_permutation = arrays["dim_permutation"].tolist()
        else:
            compiled_dim = gp._cont_dim if gp._is_mixed else total_dim
            from .kernel import compute_dim_permutation

            if gp._compiled_kernel.has_active_dims():
                gp._dim_permutation, _ = compute_dim_permutation(
                    gp._compiled_kernel, compiled_dim
                )

        if "X_train_cont" in arrays:
            gp._X_train_cont = arrays["X_train_cont"]
            split_C = arrays["C_train"] if "C_train" in arrays else None
        else:
            split_X_train_cont, split_C = gp._split_data(gp._X_train)
            if gp._dim_permutation is not None:
                split_X_train_cont = split_X_train_cont[:, gp._dim_permutation].astype(
                    np.float32
                )
            gp._X_train_cont = split_X_train_cont

        gp._C_train = arrays["C_train"] if "C_train" in arrays else split_C

        # Compile kernel module and engine so predict() works immediately
        if gp._result is not None:
            compiled_dim = (
                gp._X_train_cont.shape[1] if gp._X_train_cont is not None else gp._dim
            )
            if compiled_dim is not None:
                gp._ensure_compiled(
                    int(compiled_dim), num_tasks=gp._result.num_tasks, fresh_load=False
                )

        # Keep a live provider after load for composite models so predict()
        # can reuse the trained provider state instead of re-entering the
        # unstable mixed/composite provider rebuild path on first use.
        if (
            isinstance(gp._result, MultiOutputTrainingResult)
            and gp._kernel_module is not None
        ):
            provider_params = np.ascontiguousarray(gp._result.params, dtype=np.float32)
            provider_X = gp._X_train_cont if gp._X_train_cont is not None else gp._X_train
            if provider_X is not None:
                gp._destroy_persistent_provider()
                gp._provider_info = gp._build_provider_info(provider_X, provider_params, 0.0)
                register_provider_lease(
                    gp._kernel_module, gp, gp._destroy_persistent_provider
                )

        gp._backend_train_info = None
        gp._backend_predict_info = None
        gp._backend_sample_info = None

        return gp

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        tasks = f", tasks={self.num_tasks}" if self.num_tasks else ""
        ard_str = ", ard=True" if self.ard else ""
        if self._is_composite:
            kernel_str = self.kernel.to_mojo_type()
        else:
            kernel_str = str(self.kernel)
        return f"MultiOutputGP(kernel={kernel_str}{tasks}{ard_str}, {status})"


# =============================================================================
# LMC (Linear Model of Coregionalization) Multi-Output GP
# =============================================================================


@dataclass
class LMCTrainingResult:
    """Result from LMC multi-output GP training.

    LMC model: K_full = sum_{s=1}^{R} (A_s (x) K_X_s) + D
    Each latent s has its own kernel type, lengthscale, and coregionalization
    matrix A_s = L_s L_s^T.
    """

    # Training diagnostics (always available)
    final_nll: float
    nll_history: np.ndarray
    iterations: int
    converged: bool
    num_latents: int
    num_tasks: int

    # Per-task noise
    noise_per_task: np.ndarray  # [T] per-task noise variances

    # Per-latent parameters (extracted from params_per_latent)
    lengthscales: Optional[np.ndarray] = None  # [R] isotropic or [R*d] ARD
    outputscales: Optional[np.ndarray] = None  # [R] per-latent outputscales
    params_per_latent: Optional[list] = None  # Raw params per latent
    kernel_types: Optional[np.ndarray] = None  # [R] kernel type IDs

    # Coregionalization matrices
    A_matrices: Optional[np.ndarray] = None  # [R, T, T]
    L_factors: Optional[np.ndarray] = None  # [R, T, T]

    # Effective task covariance (sum of A_s)
    B: Optional[np.ndarray] = None  # [T, T] = sum_s A_s
    Q: Optional[np.ndarray] = None  # [T, T] eigenvectors of B
    Lambda: Optional[np.ndarray] = None  # [T] eigenvalues of B

    # Prediction state
    alpha: Optional[np.ndarray] = None  # [n, T] CG solution
    alpha_rotated: Optional[np.ndarray] = None  # [n, T] rotated alpha vectors
    effective_scales: Optional[np.ndarray] = None  # [T] eigenvalues of B

    # ARD support
    use_ard: bool = False
    lengthscales_per_dim: Optional[np.ndarray] = None

    # ConstantMean
    mean_per_task: Optional[np.ndarray] = None

    # Diagonal variance for task covariance eigenvalue floor
    var_diag: Optional[np.ndarray] = None

    # Mixed continuous+categorical latent state (per latent)
    cat_params_per_latent: Optional[list] = None
    cat_param_names_per_latent: Optional[list] = None

    # Direct per-iteration wall-clock times from the backend training loop.
    iter_times_ms: Optional[np.ndarray] = None

    # Fixed observation-noise diagonal supplied by the user, shape [n, T].
    fixed_observation_noise: Optional[np.ndarray] = None


class MultiOutputLMCGP:
    """Multi-output GP with LMC (Linear Model of Coregionalization).

    Uses the LMC model where the full kernel is:
        K_full = sum_{s=1}^{R} (A_s (x) K_X_s) + D

    Each latent s has:
    - Its own kernel type (e.g., RBF, Matern)
    - Its own lengthscale
    - A coregionalization matrix A_s = L_s L_s^T

    This generalizes the ICM model (which uses a single kernel K_X for all latents)
    by allowing different kernel types and lengthscales per latent.

    When R=1, LMC reduces to ICM.

    Parameters
    ----------
    kernels : list of str or list of KernelNode
        Kernel types for each latent. Can be:
        - List of strings: each must be one of 'rbf', 'matern32', 'matern52',
          'matern12', 'periodic', 'rq', 'linear', 'polynomial'.
        - List of KernelNode objects: each latent can have a different composite
          kernel structure (e.g., [Kernel.rbf(), Kernel.matern52() + Kernel.periodic()]).
          Kernels with the same structure share the compiled Mojo module.
          Uses JIT compilation.
        The length of this list determines R (number of latents).
    num_probes : int
        Number of probe vectors for SLQ log-det estimation (default 10).
    max_cg_iterations : int
        Maximum CG iterations per solve (default 200).
    cg_tolerance : float
        CG convergence tolerance (default 1.0).
    preconditioner_rank : int
        Pivoted Cholesky preconditioner rank (default 15).
    use_preconditioner : bool | None
        Whether to enable the pivoted-Cholesky preconditioner. If omitted,
        preconditioning stays enabled whenever the resolved `preconditioner_rank` is
        positive.
    precond_rebuild_threshold : float
        Relative parameter-change threshold for preconditioner rebuilds.
    preconditioner : str
        Preconditioner construction method: 'greedy', 'rpcholesky', 'nystrom', or 'auto'.
    ard : bool
        Enables per-dimension lengthscales for continuous latent kernels. For
        mixed continuous-categorical latents, ARD applies only to the continuous
        dimensions after categorical columns are split out.
    Example
    -------
    >>> from mojogp.multi_output_gp import MultiOutputLMCGP
    >>> from mojogp.kernel import Kernel
    >>>
    >>> # String kernels (static .so bindings)
    >>> gp = MultiOutputLMCGP(kernels=["rbf", "matern52"])
    >>> gp.fit(X_train, Y_train, max_iterations=100)
    >>> mean, var = gp.predict(X_test, return_var=True)
    >>>
    >>> # Composite kernels (JIT-compiled) — same structure, different learned params
    >>> k = Kernel.rbf() + Kernel.matern52()
    >>> gp = MultiOutputLMCGP(kernels=[k, k])
    >>> gp.fit(X_train, Y_train, max_iterations=100)
    >>> mean, var = gp.predict(X_test, return_var=True)
    >>>
    >>> # Heterogeneous kernels (different structure per latent)
    >>> gp = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52() + Kernel.periodic()])
    >>> gp.fit(X_train, Y_train, max_iterations=100)
    >>> mean, var = gp.predict(X_test, return_var=True)
    """

    def __init__(
        self,
        kernels: list,
        num_probes: int = 10,
        max_cg_iterations: int = 200,
        cg_tolerance: float = 1.0,
        preconditioner_rank: int = 15,
        precond_rebuild_threshold: float = 0.5,
        preconditioner: str = "greedy",
        use_preconditioner: Optional[bool] = None,
        max_tridiag_iterations: int = 30,
        ard: bool = False,
        init_mean: Optional[Union[float, np.ndarray]] = None,
    ):
        # Always convert string kernels to KernelNode for JIT engine
        _KERNEL_STRING_TO_NODE = {
            "rbf": Kernel.rbf,
            "matern12": Kernel.matern12,
            "matern32": Kernel.matern32,
            "matern52": Kernel.matern52,
            "periodic": Kernel.periodic,
            "rq": Kernel.rq,
            "linear": Kernel.linear,
            "polynomial": Kernel.polynomial,
        }
        if len(kernels) > 0 and isinstance(kernels[0], str):
            converted = []
            for k in kernels:
                if isinstance(k, str):
                    if k not in KERNEL_TYPES:
                        raise ValueError(
                            f"Unknown kernel '{k}'. Must be one of: {list(KERNEL_TYPES.keys())}"
                        )
                    node_fn = _KERNEL_STRING_TO_NODE[k.lower()]
                    converted.append(node_fn())
                else:
                    converted.append(k)
            kernels = converted

        # Validate kernels — all must be KernelNode now
        self._is_composite = True
        self._composite_kernel: Optional[KernelNode] = None

        if len(kernels) > 0:
            self._composite_kernel = kernels[0]
            for k in kernels:
                if not isinstance(k, KernelNode):
                    raise ValueError(
                        "All kernels must be KernelNode objects or valid kernel strings. "
                        f"Got {type(k)}"
                    )

        self.kernels = kernels
        self.num_latents = len(kernels)
        self.num_probes = num_probes
        self.max_cg_iter = max_cg_iterations
        self.cg_tol = cg_tolerance
        resolved_precond = resolve_preconditioner_settings(
            {},
            precond_rank=preconditioner_rank,
            precond_rebuild_threshold=precond_rebuild_threshold,
            precond=preconditioner,
            use_preconditioner=use_preconditioner,
        )
        self.use_preconditioner = resolved_precond["use_preconditioner"]
        self.precond_rank = resolved_precond["precond_rank"]
        self.precond = resolved_precond["precond"]
        self.precond_method = resolved_precond["precond_method"]
        self.precond_rebuild_threshold = resolved_precond["precond_rebuild_threshold"]
        self.max_tridiag_iter = max_tridiag_iterations
        self.ard = ard
        self.method = "materialized"

        # ConstantMean
        self._init_mean = init_mean
        self._fitted_mean: Optional[np.ndarray] = None

        # JIT engine and kernel modules (list for heterogeneous LMC)
        self._engine = None
        self._kernel_module = None  # first loaded kernel module for shared providers
        self._kernel_modules = None  # per-latent modules for heterogeneous kernels
        self._isolated_load_id = uuid.uuid4().hex
        self._provider_infos = (
            None  # per-latent provider infos with trained params (for GPU predict)
        )
        self._isolated_load_id = uuid.uuid4().hex
        self._training_bundle: Optional[ProviderBundle] = None
        self._inference_bundle: Optional[ProviderBundle] = None
        self._borrowed_runtime_owner: Optional[Any] = None
        self._borrowed_runtime_role: Optional[str] = None
        self._borrowed_runtime_method: Optional[str] = None

        # Training state
        self._is_trained = False
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None
        self._result: Optional[LMCTrainingResult] = None
        self._raw_result: Optional[Dict[str, Any]] = None
        self._backend_train_info: Optional[Dict[str, Any]] = None
        self._backend_predict_info: Optional[Dict[str, Any]] = None
        self._backend_sample_info: Optional[Dict[str, Any]] = None
        self._specialization_request = SpecializationRequest.disabled()
        self._specialization_decision: Optional[SpecializationDecision] = None
        self._pathwise_bundle_role: Optional[str] = None
        self._learned_composite_params = None  # Set by _fit_composite() if applicable
        self._fixed_observation_noise: Optional[np.ndarray] = None
        self._noise_group_values: Optional[np.ndarray] = None

        # Preserve the original latent kernels so mixed compilation/remapping does
        # not destroy categorical structure needed for save/load and prediction.
        self._original_kernels = [
            KernelNode.from_dict(k.to_dict()) for k in self.kernels
        ]

        # Per-latent mixed/compiled state (populated on mixed fits and loads).
        self._has_mixed_latents = False
        self._latent_is_mixed: List[bool] = [False] * self.num_latents
        self._latent_analyses: Optional[List[Any]] = None
        self._latent_compiled_kernels: Optional[List[KernelNode]] = None
        self._latent_cat_specs: Optional[List[List[Dict[str, Any]]]] = None
        self._latent_cat_col_indices: Optional[List[List[int]]] = None
        self._latent_cont_dims: Optional[List[int]] = None
        self._latent_dim_permutations: Optional[List[Optional[List[int]]]] = None
        self._latent_X_train_conts: Optional[List[np.ndarray]] = None
        self._latent_C_trains: Optional[List[np.ndarray]] = None
        self._latent_cat_param_names: Optional[List[List[str]]] = None

    def _set_specialization_request(
        self,
        request: SpecializationRequest | dict[str, Any] | None,
    ) -> None:
        if isinstance(request, SpecializationRequest):
            self._specialization_request = request
            return
        self._specialization_request = SpecializationRequest.from_dict(request)

    def _maybe_attach_specialization_metadata(
        self,
        info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if info is None:
            return None
        decision = self._specialization_decision
        if decision is None or decision.mode == "disabled":
            return info
        info.setdefault("specialization_mode", decision.mode)
        info.setdefault("specialization_key", decision.profile.specialization_key)
        info.setdefault("specialization_family", decision.profile.family)
        info.setdefault("specialization_source", decision.profile.source)
        info.setdefault("specialization_default_equivalent", decision.default_equivalent)
        info.setdefault("specialization_reason", decision.reason)
        info.setdefault("specialization_descriptor", decision.descriptor.to_dict())
        info.setdefault("specialization_profile", decision.profile.to_dict())
        return info

    def _set_runtime_state_from_bundle(self, bundle: Optional[ProviderBundle]) -> None:
        if bundle is None or bundle.is_destroyed:
            self._provider_infos = None
            self._kernel_modules = None
            self._kernel_module = None
            return
        self._provider_infos = bundle.provider_infos
        self._kernel_modules = bundle.kernel_modules
        self._kernel_module = (
            bundle.kernel_modules[0] if bundle.kernel_modules else None
        )

    def _release_mixed_runtime_bundles(self) -> None:
        for attr in ("_training_bundle", "_inference_bundle"):
            bundle = getattr(self, attr, None)
            if bundle is not None:
                unregister_provider_lease(bundle.kernel_modules, self)
                orphan_provider_bundle(bundle)
                setattr(self, attr, None)
        self._set_runtime_state_from_bundle(None)
        self._engine = None

    def _release_registered_runtime_bundle(self, bundle: ProviderBundle) -> None:
        unregister_provider_lease(bundle.kernel_modules, self)
        orphan_provider_bundle(bundle)
        if self._training_bundle is bundle:
            self._training_bundle = None
        if self._inference_bundle is bundle:
            self._inference_bundle = None
        if self._training_bundle is not None and not self._training_bundle.is_destroyed:
            self._set_runtime_state_from_bundle(self._training_bundle)
        elif (
            self._inference_bundle is not None
            and not self._inference_bundle.is_destroyed
        ):
            self._set_runtime_state_from_bundle(self._inference_bundle)
        else:
            self._set_runtime_state_from_bundle(None)
        if self._training_bundle is None and self._inference_bundle is None:
            self._engine = None

    def _destroy_training_bundle_for_materialized_inference(self) -> bool:
        """Drop fit-time mixed providers before a fresh materialized inference handoff."""

        bundle = self._training_bundle
        if bundle is None:
            self._set_runtime_state_from_bundle(self._inference_bundle)
            return False
        self._training_bundle = None
        unregister_provider_lease(bundle.kernel_modules, self)
        destroy_provider_bundle(bundle)
        self._set_runtime_state_from_bundle(self._inference_bundle)
        if self._inference_bundle is None or self._inference_bundle.is_destroyed:
            self._engine = None
        return True

    def _clear_borrowed_runtime(self) -> None:
        self._borrowed_runtime_owner = None
        self._borrowed_runtime_role = None
        self._borrowed_runtime_method = None

    def _try_borrow_saved_runtime(self, path: str) -> bool:
        """Borrow same-process runtime state from the model saved at path."""

        if not self._has_mixed_latents or self.method != "materialized":
            return False
        owner_ref = _LMC_SAVED_RUNTIME_OWNERS.get(str(Path(path).resolve()))
        owner = owner_ref() if owner_ref is not None else None
        if owner is None or owner is self:
            return False
        if self._X_train is None or getattr(owner, "_X_train", None) is None:
            return False
        if not np.array_equal(owner._X_train, self._X_train):
            return False

        bundle = None
        if owner._inference_bundle is not None and not owner._inference_bundle.is_destroyed:
            bundle = owner._inference_bundle
        elif owner._training_bundle is not None and not owner._training_bundle.is_destroyed:
            bundle = owner._training_bundle
        if bundle is None or owner._engine is None:
            return False

        self._provider_infos = bundle.provider_infos
        self._kernel_modules = bundle.kernel_modules
        self._kernel_module = bundle.kernel_modules[0] if bundle.kernel_modules else None
        self._engine = owner._engine
        self._borrowed_runtime_owner = owner
        self._borrowed_runtime_role = bundle.role
        self._borrowed_runtime_method = bundle.method
        return True

    def _materialized_mixed_inference_needs_training_teardown(self) -> bool:
        """Whether canonical mixed inference must release fit-time providers first."""

        if not self._has_mixed_latents or self._latent_cat_specs is None:
            return False
        if self._X_train is not None and int(self._X_train.shape[0]) <= 3000:
            return True
        return any(len(specs) > 1 for specs in self._latent_cat_specs)

    def _destroy_persistent_provider_infos(self, *, clear_modules: bool = False):
        if self._borrowed_runtime_owner is not None:
            self._provider_infos = None
            self._kernel_modules = None
            self._kernel_module = None
            self._engine = None
            self._clear_borrowed_runtime()
            return

        if self._training_bundle is not None or self._inference_bundle is not None:
            self._release_mixed_runtime_bundles()
            return

        provider_infos = getattr(self, "_provider_infos", None)
        kernel_modules = getattr(self, "_kernel_modules", None)
        if kernel_modules is not None:
            unregister_provider_lease(kernel_modules, self)
        if provider_infos is not None and kernel_modules is not None:
            # Continuous LMC prediction can leave GPU work queued against the
            # provider; destroy only after the runtime has settled.
            _cleanup_runtime_state()
            destroy_provider_infos(kernel_modules, provider_infos)
            _cleanup_runtime_state()
        self._provider_infos = None
        if clear_modules:
            self._kernel_modules = None
            self._kernel_module = None
            self._engine = None

    def _build_mixed_runtime_bundle(
        self,
        params_per_latent: Sequence[Sequence[float] | np.ndarray],
        *,
        role: str,
        fresh_load: bool,
        reclaim_live_owners: bool,
        runtime_method: Optional[str] = None,
        reclaim_live_training_owners: bool = False,
        x_train_per_latent: Optional[Sequence[np.ndarray]] = None,
    ) -> ProviderBundle:
        from .loader import load_kernel_module_engine
        from .loader import load_engine
        from .codegen_engine.compiler import make_module_name

        assert self._latent_compiled_kernels is not None
        bundle_method = runtime_method or self.method

        if self.num_latents > 1:
            _ensure_lmc_multi_latent_jit_warmup()

        if x_train_per_latent is None:
            if self._X_train is None:
                raise RuntimeError(
                    "Mixed runtime bundle requires training inputs or explicit latent inputs."
                )
            if self._latent_X_train_conts is not None:
                x_train_per_latent = self._latent_X_train_conts
            else:
                x_train_per_latent = [self._X_train for _ in range(self.num_latents)]

        module_cache: Dict[Tuple[str, int], Any] = {}
        kernel_modules: list[Any] = []
        reclaimed_conflicting_runtime = False
        settled_before_live_reclaim = False
        isolated_load_id = uuid.uuid4().hex if fresh_load else None
        if self._result is not None:
            ncols_hint = _lmc_provider_ncols_hint(int(self._result.num_tasks))
        elif self._Y_train is not None:
            ncols_hint = _lmc_provider_ncols_hint(int(self._Y_train.shape[1]))
        else:
            ncols_hint = None
        live_reclaim_roles = (BUNDLE_ROLE_INFERENCE,)
        if role == BUNDLE_ROLE_TRAINING:
            # A new fit owns the runtime route for its wrapper. Older live mixed
            # training bundles with the same generated module name can poison the
            # next init_provider(), so reclaim non-self owners before building.
            live_reclaim_roles = (BUNDLE_ROLE_INFERENCE, BUNDLE_ROLE_TRAINING)
        elif role == BUNDLE_ROLE_INFERENCE and self.method != "materialized":
            # Matrix-free loaded models have no training bundle to reuse. A live
            # matrix-free training bundle from another wrapper can still poison
            # init_provider() for the new inference bundle, so reclaim it here.
            # Materialized training bundles remain protected by the canonical
            # handoff rule below.
            live_reclaim_roles = (BUNDLE_ROLE_INFERENCE, BUNDLE_ROLE_TRAINING)
        elif role == BUNDLE_ROLE_INFERENCE and reclaim_live_training_owners:
            live_reclaim_roles = (BUNDLE_ROLE_TRAINING,)
        if self._engine is None or fresh_load:
            self._engine = load_engine(
                fresh_load=fresh_load,
                isolated_load_id=isolated_load_id,
                verbose=False,
            )
        for latent_idx, kernel in enumerate(self._latent_compiled_kernels):
            x_train_s = x_train_per_latent[latent_idx]
            cache_key = (kernel.to_mojo_type(), int(x_train_s.shape[1]))
            if cache_key not in module_cache:
                module_name = make_module_name(
                    kernel, int(x_train_s.shape[1]), "fn_ptr"
                )
                if reclaim_live_owners:
                    if not settled_before_live_reclaim:
                        # Pathwise/predict calls may leave backend GPU work
                        # pending on a provider that the next loaded wrapper is
                        # about to reclaim. Settle before destroying live state.
                        _cleanup_runtime_state()
                        settled_before_live_reclaim = True
                    # Mixed materialized canonical routes only need to reclaim
                    # conflicting inference bundles; live training bundles are
                    # safe to keep and reclaiming them can destabilize init_provider().
                    reclaimed_conflicting_runtime = (
                        reclaim_provider_bundles_by_name(
                            [module_name],
                            owner=self,
                            include_live_owners=True,
                            roles=live_reclaim_roles,
                        )
                        or reclaimed_conflicting_runtime
                    )
                if self._training_bundle is None:
                    # Providerless loaded models must take over the route from
                    # any still-live fitted wrapper before creating providers.
                    reclaimed_conflicting_runtime = (
                        reclaim_provider_bundles_by_name(
                            [module_name],
                            owner=self,
                            include_live_owners=True,
                            roles=(BUNDLE_ROLE_TRAINING,),
                        )
                        or reclaimed_conflicting_runtime
                    )
                revoke_conflicting_provider_leases_by_name(
                    module_name,
                    owner=self,
                    include_live_owners=False,
                )
                reclaimed_conflicting_runtime = (
                    reclaim_provider_bundles_by_name(
                        [module_name],
                        owner=self,
                        include_live_owners=False,
                    )
                    or reclaimed_conflicting_runtime
                )
                module_cache[cache_key] = load_kernel_module_engine(
                    kernel,
                    dim=int(x_train_s.shape[1]),
                    fresh_load=fresh_load,
                    isolated_load_id=isolated_load_id,
                    ncols_hint=ncols_hint,
                    verbose=False,
                )
            kernel_modules.append(module_cache[cache_key])

        reclaimed_conflicting_runtime = (
            reclaim_provider_bundles_for_modules(
                kernel_modules,
                owner=self,
                include_live_owners=False,
            )
            or reclaimed_conflicting_runtime
        )

        if reclaimed_conflicting_runtime or fresh_load:
            # Cross-model mixed runtime handoffs can leave GPU/native teardown work
            # pending briefly. Fresh isolated loads also replace sys.modules entries
            # for the same embedded PyInit name, so settle the runtime before the
            # next init_provider() touches native module state.
            _cleanup_runtime_state()

        seen_module_ids: set[int] = set()
        for kernel_module in kernel_modules:
            module_id = id(kernel_module)
            if module_id in seen_module_ids:
                continue
            seen_module_ids.add(module_id)
            revoke_conflicting_provider_lease(kernel_module, self)

        provider_infos = []
        for kernel_module, x_train_s, params_s in zip(
            kernel_modules, x_train_per_latent, params_per_latent
        ):
            params_arr = np.ascontiguousarray(params_s, dtype=np.float32)
            provider_info = kernel_module.init_provider(
                np.ascontiguousarray(x_train_s, dtype=np.float32),
                params_arr,
                0.0,
            )
            if bundle_method == "materialized":
                kernel_module.materialize(provider_info)
            provider_infos.append(provider_info)
        bundle = ProviderBundle(
            role=role,
            method=bundle_method,
            kernel_modules=kernel_modules,
            provider_infos=provider_infos,
        )
        register_provider_bundle(
            bundle,
            self,
            owner_releaser_name="_release_registered_runtime_bundle",
        )
        register_provider_lease(
            kernel_modules, self, self._destroy_persistent_provider_infos
        )
        return bundle

    def _get_mixed_runtime_bundle(
        self,
        params_per_latent: Sequence[Sequence[float] | np.ndarray],
        *,
        canonical: bool,
    ) -> ProviderBundle:
        if not canonical:
            bundle = self._training_bundle
            if bundle is not None and not bundle.is_destroyed:
                return bundle

        bundle = self._inference_bundle
        if bundle is None or bundle.is_destroyed:
            fresh_inference_load = canonical or self.method == "materialized"
            inference_runtime_method = (
                "matrix_free" if self.method == "materialized" else self.method
            )
            needs_training_teardown = (
                self.method == "materialized"
                and self._materialized_mixed_inference_needs_training_teardown()
            )
            reclaim_live_owners = True
            if fresh_inference_load and self.method != "materialized":
                # Matrix-free loaded pathwise routes need an isolated inference
                # image, but destroying another wrapper's live training bundle
                # immediately before that fresh init_provider() can poison the
                # native handoff. Stale/orphaned bundles are still reclaimed by
                # _build_mixed_runtime_bundle(); only live owners are left alone.
                reclaim_live_owners = False
            if (
                fresh_inference_load
                and self.method == "materialized"
                and needs_training_teardown
            ):
                # Materialized mixed training bundles can hold the old native
                # image alive. Drop them before loading the canonical inference
                # image so init_provider() starts from a clean module handoff.
                if self._destroy_training_bundle_for_materialized_inference():
                    _cleanup_runtime_state()
            bundle = self._build_mixed_runtime_bundle(
                params_per_latent,
                role=BUNDLE_ROLE_INFERENCE,
                fresh_load=fresh_inference_load,
                runtime_method=inference_runtime_method,
                reclaim_live_training_owners=needs_training_teardown,
                # Mixed inference bundles may conflict with another live model's
                # bundle for the same fn-ptr module name. The reclaim scope is
                # method-specific inside _build_mixed_runtime_bundle().
                reclaim_live_owners=reclaim_live_owners,
            )
            self._inference_bundle = bundle
        return bundle

    def _resolve_mixed_provider_runtime(
        self,
        params_per_latent: Sequence[Sequence[float] | np.ndarray],
        *,
        canonical: bool,
    ) -> tuple[list[dict[str, Any]], Sequence[Any], str, str]:
        provider_infos = self._provider_infos
        provider_kernel_modules = self._kernel_modules
        if (
            provider_infos is not None
            and provider_kernel_modules is not None
            and (not canonical or self._borrowed_runtime_owner is not None)
        ):
            if self._borrowed_runtime_owner is not None:
                runtime_role = self._borrowed_runtime_role or BUNDLE_ROLE_INFERENCE
                runtime_method = self._borrowed_runtime_method or self.method
            else:
                runtime_role = BUNDLE_ROLE_TRAINING
                runtime_method = (
                    self._training_bundle.method
                    if self._training_bundle is not None
                    and not self._training_bundle.is_destroyed
                    else self.method
                )
            return (
                provider_infos,
                provider_kernel_modules,
                runtime_role,
                runtime_method,
            )

        bundle = self._get_mixed_runtime_bundle(
            params_per_latent,
            canonical=canonical,
        )
        return (
            bundle.provider_infos,
            bundle.kernel_modules,
            bundle_runtime_owner_role(bundle),
            bundle.method,
        )

    def __del__(self):
        try:
            self._destroy_persistent_provider_infos(clear_modules=True)
        except Exception:
            pass

    @property
    def is_trained(self) -> bool:
        """Whether the GP has been trained."""
        return self._is_trained

    @property
    def training_result(self) -> Optional[LMCTrainingResult]:
        """Training result (None if not trained)."""
        return self._result

    @property
    def backend_train_info(self) -> Optional[Dict[str, Any]]:
        """Normalized backend metadata returned by the JIT engine."""
        return self._backend_train_info

    @property
    def backend_predict_info(self) -> Optional[Dict[str, Any]]:
        """Backend prediction-route metadata from the most recent prediction."""
        return self._backend_predict_info

    @property
    def backend_sample_info(self) -> Optional[Dict[str, Any]]:
        """Backend sampling-route metadata from the most recent sample draw."""
        return self._backend_sample_info

    @property
    def task_covariance(self) -> Optional[np.ndarray]:
        """Effective task covariance B = sum_s A_s [T, T]. None if not trained."""
        if self._result is None:
            return None
        return self._result.B

    @property
    def num_tasks(self) -> Optional[int]:
        """Number of tasks. None if not trained."""
        if self._result is None:
            return None
        return self._result.num_tasks

    def _configure_latent_kernels_for_fit(self, total_dim: int) -> None:
        """Prepare per-latent continuous/mixed kernel state for LMC training."""
        from .kernel import analyze_kernel_tree, compute_dim_permutation

        self._has_mixed_latents = False
        self._latent_is_mixed = []
        self._latent_analyses = []
        self._latent_compiled_kernels = []
        self._latent_cat_specs = []
        self._latent_cat_col_indices = []
        self._latent_cont_dims = []
        self._latent_dim_permutations = []
        self._latent_cat_param_names = []

        source_kernels = getattr(self, "_original_kernels", self.kernels)

        for latent_idx, kernel in enumerate(source_kernels):
            analysis = None
            cat_specs: List[Dict[str, Any]] = []
            cat_col_indices: List[int] = []
            cat_param_names: List[str] = []
            compiled_kernel = kernel
            cont_dim = total_dim
            is_mixed = False

            if kernel.has_categorical():
                analysis = analyze_kernel_tree(kernel, total_dim)
                is_mixed = (
                    not analysis.is_pure_continuous and not analysis.is_pure_categorical
                )
                if analysis.is_pure_categorical:
                    raise ValueError(
                        "Pure categorical latent kernels are not supported for "
                        f"MultiOutputLMCGP. Latent kernel {latent_idx} contains no "
                        "continuous dimensions."
                    )

                cat_specs = [
                    {
                        "levels": int(spec.levels),
                        "kernel_type": spec.kernel_type.name.lower(),
                    }
                    for spec in analysis.categorical_specs
                ]
                cat_col_indices = [
                    spec.col_index for spec in analysis.categorical_specs
                ]
                for spec in analysis.categorical_specs:
                    cat_param_names.extend(spec.param_names)

                compiled_kernel = analysis.structured_kernel
                unique_cat_cols = sorted(set(cat_col_indices))
                cont_dim = total_dim - len(unique_cat_cols)
                if cont_dim <= 0:
                    raise ValueError(
                        "Pure categorical latent kernels are not supported for "
                        f"MultiOutputLMCGP. Latent kernel {latent_idx} has no continuous "
                        "dimensions."
                    )

                cont_cols = [d for d in range(total_dim) if d not in unique_cat_cols]
                dim_map = {orig: idx for idx, orig in enumerate(cont_cols)}
                compiled_kernel = MultiOutputGP._remap_kernel_active_dims(
                    compiled_kernel, dim_map
                )
                self._has_mixed_latents = True

            if self.ard:
                compiled_kernel = make_ard_kernel(compiled_kernel, cont_dim)

            dim_permutation = None
            if compiled_kernel.has_active_dims():
                MultiOutputGP._validate_active_dims_bounds(compiled_kernel, cont_dim)
                dim_permutation, _ = compute_dim_permutation(compiled_kernel, cont_dim)

            self._latent_is_mixed.append(is_mixed)
            self._latent_analyses.append(analysis)
            self._latent_compiled_kernels.append(compiled_kernel)
            self._latent_cat_specs.append(cat_specs)
            self._latent_cat_col_indices.append(cat_col_indices)
            self._latent_cont_dims.append(cont_dim)
            self._latent_dim_permutations.append(dim_permutation)
            self._latent_cat_param_names.append(cat_param_names)

    def _transform_inputs_for_latent(
        self, latent_idx: int, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split/remap inputs for one latent's compiled continuous kernel."""
        assert self._latent_cat_col_indices is not None
        assert self._latent_dim_permutations is not None

        cat_cols = self._latent_cat_col_indices[latent_idx]
        if cat_cols:
            cont_cols = [c for c in range(X.shape[1]) if c not in cat_cols]
            X_cont = np.ascontiguousarray(X[:, cont_cols], dtype=np.float32)
            C = np.ascontiguousarray(X[:, cat_cols], dtype=np.int32)
        else:
            X_cont = np.ascontiguousarray(X, dtype=np.float32)
            C = np.zeros((X.shape[0], 0), dtype=np.int32)

        perm = self._latent_dim_permutations[latent_idx]
        if perm is not None:
            X_cont = np.ascontiguousarray(X_cont[:, perm], dtype=np.float32)

        return X_cont, C

    def _evaluate_latent_kernel_matrix(
        self,
        latent_idx: int,
        X1: np.ndarray,
        X2: np.ndarray,
        cont_params: np.ndarray,
        cat_params: Optional[np.ndarray],
    ) -> np.ndarray:
        """Evaluate one latent kernel in Python for mixed CPU prediction fallback."""
        assert self._latent_compiled_kernels is not None
        assert self._latent_is_mixed is not None
        assert self._latent_analyses is not None
        assert self._latent_cat_specs is not None

        X1_cont, C1 = self._transform_inputs_for_latent(latent_idx, X1)
        X2_cont, C2 = self._transform_inputs_for_latent(latent_idx, X2)
        if not self._latent_is_mixed[latent_idx]:
            return (
                self._latent_compiled_kernels[latent_idx]
                .evaluate(
                    X1_cont,
                    X2_cont,
                    params=np.ascontiguousarray(cont_params, dtype=np.float32),
                )
                .astype(np.float32)
            )

        continuous_kernel = self._latent_analyses[latent_idx].continuous_kernel
        assert continuous_kernel is not None
        return _evaluate_mixed_kernel_matrix_py(
            continuous_kernel,
            X1_cont,
            X2_cont,
            C1,
            C2,
            np.ascontiguousarray(cont_params, dtype=np.float32),
            self._latent_cat_specs[latent_idx],
            None
            if cat_params is None
            else np.ascontiguousarray(cat_params, dtype=np.float32),
        )

    def _predict_lmc_dense_exact_variance(
        self,
        X_test: np.ndarray,
        params_per_latent: list[np.ndarray],
        cat_params_per_latent: Optional[list[np.ndarray]],
        A_matrices: np.ndarray,
        noise_per_task: np.ndarray,
    ) -> np.ndarray:
        """Compute exact LMC posterior variances from the full dense covariance.

        The per-latent scalar variance shortcut is not exact for LMC because the
        posterior couples tasks and latents through the full training covariance.
        This route is intentionally limited to materialized exact prediction.
        """
        if self._X_train is None:
            raise RuntimeError("LMC model must be fitted before prediction")

        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        X_train = np.ascontiguousarray(self._X_train, dtype=np.float32)
        R = int(self._result.num_latents)
        T = int(self._result.num_tasks)
        n = int(X_train.shape[0])
        m = int(X_test.shape[0])

        train_cov = np.zeros((n * T, n * T), dtype=np.float64)
        cross_cov = np.zeros((n * T, m * T), dtype=np.float64)
        prior_diag = np.zeros((m, T), dtype=np.float64)

        for s in range(R):
            params_s = np.ascontiguousarray(params_per_latent[s], dtype=np.float32)
            cat_params_s = (
                None
                if cat_params_per_latent is None or s >= len(cat_params_per_latent)
                else np.ascontiguousarray(cat_params_per_latent[s], dtype=np.float32)
            )
            A_s = np.asarray(A_matrices[s], dtype=np.float64)

            K_train_s = self._evaluate_latent_kernel_matrix(
                s, X_train, X_train, params_s, cat_params_s
            ).astype(np.float64)
            K_cross_s = self._evaluate_latent_kernel_matrix(
                s, X_train, X_test, params_s, cat_params_s
            ).astype(np.float64)
            K_test_s = self._evaluate_latent_kernel_matrix(
                s, X_test, X_test, params_s, cat_params_s
            ).astype(np.float64)

            train_cov += np.kron(K_train_s, A_s)
            cross_cov += np.kron(K_cross_s, A_s)
            prior_diag += np.diag(K_test_s)[:, np.newaxis] * np.diag(A_s)[np.newaxis, :]

        train_cov += np.kron(
            np.eye(n, dtype=np.float64),
            np.diag(np.asarray(noise_per_task, dtype=np.float64)),
        )
        if self._result.fixed_observation_noise is not None:
            train_cov += np.diag(
                np.asarray(self._result.fixed_observation_noise, dtype=np.float64).reshape(-1)
            )

        eye = np.eye(n * T, dtype=np.float64)
        chol = np.linalg.cholesky(train_cov + 1e-6 * eye)
        v = np.linalg.solve(chol, cross_cov)
        variance_flat = prior_diag.reshape(-1) - np.sum(v * v, axis=0)
        variance = variance_flat.reshape(m, T)
        variance += np.asarray(noise_per_task, dtype=np.float64)[np.newaxis, :]
        return np.maximum(variance, 1e-10).astype(np.float32)

    def _uses_per_latent_runtime_inputs(self) -> bool:
        """Whether backend prediction must receive transformed inputs per latent."""
        if self._has_mixed_latents:
            return True
        if self._latent_dim_permutations is None:
            return False
        return any(perm is not None for perm in self._latent_dim_permutations)

    def _predict_lmc_full_exact_variance_backend(
        self,
        provider_infos: list[dict[str, Any]],
        A_matrices: np.ndarray,
        X_test: np.ndarray,
        params_per_latent: Sequence[Sequence[float] | np.ndarray],
        cat_params_per_latent: Optional[Sequence[Sequence[float] | np.ndarray]],
        noise_per_task: np.ndarray,
        runtime_method: Optional[str] = None,
    ) -> tuple[np.ndarray, str]:
        """Compute the full LMC exact posterior diagonal via backend CG solves.

        This route constructs only train-test cross-covariance blocks, then uses
        the same full LMC adapter solve as pathwise correction. It avoids the old
        per-latent scalar variance shortcut and does not materialize train-train
        kernels in matrix-free mode.
        """
        if self._engine is None:
            raise RuntimeError("LMC exact variance requires a loaded JIT engine")
        if self._X_train is None or self._result is None:
            raise RuntimeError("LMC model must be fitted before prediction")

        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        R = int(self._result.num_latents)
        T = int(self._result.num_tasks)
        n = int(self._X_train.shape[0])
        m = int(X_test.shape[0])
        A = np.ascontiguousarray(A_matrices, dtype=np.float32)
        noise = np.ascontiguousarray(noise_per_task, dtype=np.float32)
        solve_runtime_method = runtime_method or self.method

        K_train_test: list[np.ndarray] = []
        prior_diag = np.zeros((m, T), dtype=np.float64)
        for s in range(R):
            params_s = np.ascontiguousarray(params_per_latent[s], dtype=np.float32)
            cat_params_s = (
                None
                if cat_params_per_latent is None or s >= len(cat_params_per_latent)
                else np.ascontiguousarray(cat_params_per_latent[s], dtype=np.float32)
            )
            K_s = self._evaluate_latent_kernel_matrix(
                s, self._X_train, X_test, params_s, cat_params_s
            ).astype(np.float32)
            K_train_test.append(np.ascontiguousarray(K_s, dtype=np.float32))
            K_test_diag_s = np.diag(
                self._evaluate_latent_kernel_matrix(
                    s, X_test, X_test, params_s, cat_params_s
                )
            ).astype(np.float64)
            for t in range(T):
                prior_diag[:, t] += float(A[s, t, t]) * K_test_diag_s

        solve_max_cg_iter = max(int(self.max_cg_iter), 100)
        solve_cg_tol = (
            min(float(self.cg_tol), 1e-2) if float(self.cg_tol) > 0.0 else 1e-2
        )
        use_per_latent_inputs = self._uses_per_latent_runtime_inputs()

        if use_per_latent_inputs:
            if not hasattr(self._engine, "sample_lmc_mixed_pathwise"):
                raise RuntimeError(
                    "Full LMC exact variance requires sample_lmc_mixed_pathwise() "
                    "for per-latent input transforms."
                )
            latent_x_test_conts = []
            latent_c_tests = []
            for latent_idx in range(R):
                x_test_cont_s, c_test_s = self._transform_inputs_for_latent(
                    latent_idx, X_test
                )
                latent_x_test_conts.append(
                    np.ascontiguousarray(x_test_cont_s, dtype=np.float32)
                )
                latent_c_tests.append(np.ascontiguousarray(c_test_s, dtype=np.int32))
            params_list = [
                np.ascontiguousarray(params_per_latent[s], dtype=np.float32)
                for s in range(R)
            ]
            cat_params_list = [
                np.ascontiguousarray(cp, dtype=np.float32)
                for cp in (
                    cat_params_per_latent
                    if cat_params_per_latent is not None
                    else [np.zeros(0, dtype=np.float32) for _ in range(R)]
                )
            ]
            route = "predict_lmc_mixed_full_exact"
        else:
            if not hasattr(self._engine, "sample_lmc_pathwise"):
                raise RuntimeError(
                    "Full LMC exact variance requires sample_lmc_pathwise()."
                )
            params_list = [
                np.ascontiguousarray(
                    self.kernels[s].to_engine_params(
                        np.ascontiguousarray(params_per_latent[s], dtype=np.float32)
                    ),
                    dtype=np.float32,
                )
                for s in range(R)
            ]
            route = "predict_lmc_full_exact"

        fixed_noise = (
            np.ascontiguousarray(self._result.fixed_observation_noise, dtype=np.float32)
            if self._result.fixed_observation_noise is not None
            else np.zeros((0,), dtype=np.float32)
        )
        variance = np.zeros((m, T), dtype=np.float64)
        for test_idx in range(m):
            for out_task in range(T):
                residual = np.zeros((n, T), dtype=np.float32)
                for s in range(R):
                    k_col = K_train_test[s][:, test_idx]
                    for train_task in range(T):
                        residual[:, train_task] += k_col * A[s, train_task, out_task]
                residual = np.ascontiguousarray(residual, dtype=np.float32)

                if use_per_latent_inputs:
                    raw = self._engine.sample_lmc_mixed_pathwise(
                        provider_infos,
                        residual,
                        A,
                        latent_x_test_conts,
                        params_list,
                        list(self._latent_is_mixed),
                        self._latent_C_trains,
                        latent_c_tests,
                        cat_params_list,
                        self._latent_cat_specs,
                        noise,
                        solve_max_cg_iter,
                        solve_cg_tol,
                        1 if solve_runtime_method == "materialized" else 0,
                    )
                else:
                    raw = self._engine.sample_lmc_pathwise(
                        provider_infos,
                        residual,
                        A,
                        X_test,
                        params_list,
                        noise,
                        solve_max_cg_iter,
                        solve_cg_tol,
                        fixed_noise,
                    )

                correction = np.asarray(raw["correction"], dtype=np.float64)
                variance[test_idx, out_task] = (
                    prior_diag[test_idx, out_task]
                    - correction[test_idx, out_task]
                    + float(noise[out_task])
                )

        return np.maximum(variance, 1e-10).astype(np.float32), route

    def _annotate_lmc_variance_metadata(
        self,
        *,
        compute_variance: bool,
        noise_per_task: np.ndarray,
    ) -> None:
        """Attach LMC variance semantics to backend prediction telemetry."""
        if self._backend_predict_info is None:
            return
        info = self._backend_predict_info
        info["lmc_variance_output"] = "observation" if compute_variance else "mean_only"
        info["predictive_variance_kind"] = "observation" if compute_variance else None
        info["variance_includes_observation_noise"] = bool(compute_variance)
        info["uses_fixed_observation_noise"] = bool(
            self._result.fixed_observation_noise is not None
        )
        info["lmc_exact_variance_source"] = (
            info.get("actual_variance_route")
            if compute_variance and info.get("variance_method") == "exact"
            else None
        )
        info["observation_variance_includes_learned_task_noise"] = bool(
            compute_variance
        )
        if not compute_variance:
            info["lmc_variance_exactness"] = "mean_only"
            return

        noise = np.asarray(noise_per_task, dtype=np.float32)
        info["task_noise_heterogeneous"] = bool(
            noise.size > 1 and float(np.max(noise) - np.min(noise)) > 1e-6
        )
        info["lmc_variance_task_noise_semantics"] = (
            "learned_task_noise_added_to_observation_variance"
        )
        if info.get("actual_variance_route") in (
            "dense_exact_lmc",
            "predict_lmc_full_exact",
            "predict_lmc_mixed_full_exact",
        ):
            info["lmc_variance_exactness"] = "exact_full_lmc_covariance"
        else:
            info["lmc_variance_exactness"] = "scalar_latent_approximation"

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        max_iterations: int = 100,
        learning_rate: float = 0.05,
        method: str = "materialized",
        initial_lengthscales: Optional[np.ndarray] = None,
        initial_params: Optional[np.ndarray] = None,
        initial_noise: float = 0.1,
        initial_noise_per_task: Optional[np.ndarray] = None,
        fixed_observation_noise: Optional[np.ndarray] = None,
        input_dependent_noise: Optional[Any] = None,
        grouped_noise: Optional[Any] = None,
        verbose: bool = False,
        early_stop_tol: float = 1e-4,
        early_stop_patience: int = 15,
        use_fused_kernels: bool = True,
        observation_noise: Optional[np.ndarray] = None,
        observation_noise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise_model: str = "scalar",
        noise_group_train: Optional[np.ndarray] = None,
        group_noise: Optional[np.ndarray] = None,
        progress: Any = None,
        progress_stats: Optional[Any] = None,
        progress_interval: int = 1,
    ) -> LMCTrainingResult:
        """Train the LMC multi-output GP.

        Args:
            X: Training inputs [n, d], float32
            Y: Training targets [n, T], float32 (T = number of tasks)
            max_iterations: Maximum training iterations
            learning_rate: Adam learning rate
            method: Training route, either "materialized" or "matrix_free".
                Aliases: "mat" for "materialized" and "mf" for "matrix_free".
            initial_lengthscales: Initial per-latent lengthscales [R] (for string kernels).
                If None, all latents use 1.0.
            initial_params: Initial composite kernel params [R * num_params] or [R, num_params]
                (for composite kernels). If None, all params initialized to 1.0.
            initial_noise: Initial noise variance (used when initial_noise_per_task is None)
            initial_noise_per_task: Initial per-task noise [T]. If None, all tasks
                use initial_noise.
            fixed_observation_noise: Optional fixed diagonal observation noise [n, T].
                These values are added to the exact training covariance diagonal and
                are not learned. Continuous LMC exact prediction and pathwise
                correction include the fixed training diagonal; mixed LMC rejects
                fixed observation noise until that route is separately evidenced.
            input_dependent_noise: Placeholder for future learned input-dependent
                heteroskedastic noise. Passing this currently raises
                NotImplementedError.
            grouped_noise: Placeholder for future grouped noise models. Passing
                this currently raises NotImplementedError.
            verbose: Print training progress
            early_stop_tol: Early stopping tolerance (0.0 to disable)
            early_stop_patience: Early stopping patience (iterations)
            use_fused_kernels: Whether to use fused GPU kernels that combine
                kernel evaluation + matvec into a single kernel launch (default True).
                When False, uses separate kernel evaluation and matvec steps.
                Fused kernels are faster for matrix-free methods but may not be
                available for all kernel types.
            observation_noise: Unsupported per-sample-task observation noise.
            observation_noise_fn: Unsupported input-dependent noise function.
            noise_model: Unsupported heteroskedastic noise model selector.
            noise_group_train: Unsupported grouped noise ids.
            group_noise: Unsupported grouped noise values.
            progress: Progress reporting control. Use True, ``"auto"``, a callback,
                or a reporter object with start/update/close methods.
            progress_stats: Optional stat names or callable used by the default
                tqdm reporter.
            progress_interval: Emit ordinary training iteration updates every
                this many optimizer iterations.

        Returns:
            LMCTrainingResult
        """
        if noise_model in {"learned_vector", "latent_gp"}:
            raise NotImplementedError(
                "MultiOutputLMCGP learned per-sample-task heteroskedastic noise is in development"
            )
        if (
            observation_noise is not None
            or observation_noise_fn is not None
            or noise_model != "scalar"
            or noise_group_train is not None
            or group_noise is not None
        ):
            raise NotImplementedError(
                "MultiOutputLMCGP currently supports scalar or per-task noise only; "
                "per-sample-task and grouped heteroskedastic noise are in development"
            )
        # Validate inputs
        method = normalize_fit_method(method)
        self.method = method
        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.ascontiguousarray(Y, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D [n, d], got shape {X.shape}")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D [n, T], got shape {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples, Y has {Y.shape[0]} -- must match"
            )
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
            raise ValueError("Y contains NaN or Inf values")
        if initial_noise <= 0:
            raise ValueError(f"initial_noise must be > 0, got {initial_noise}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {max_iterations}")
        progress_interval = _validate_progress_interval(progress_interval)

        n, d = X.shape
        T = Y.shape[1]
        R = self.num_latents
        fixed_observation_noise = _validate_fixed_observation_noise(
            fixed_observation_noise, n, T
        )
        if initial_noise_per_task is not None:
            initial_noise_per_task = np.asarray(initial_noise_per_task, dtype=np.float32)
            if initial_noise_per_task.shape != (T,):
                raise ValueError(
                    f"initial_noise_per_task must have shape ({T},), got "
                    f"{initial_noise_per_task.shape}"
                )
            if np.any(initial_noise_per_task <= 0):
                raise ValueError("all initial_noise_per_task entries must be > 0")
        resolved_initial_noise_per_task = (
            initial_noise_per_task
            if initial_noise_per_task is not None
            else np.full(T, initial_noise, dtype=np.float32)
        )

        has_categorical_latent = any(
            isinstance(k, KernelNode) and k.has_categorical()
            for k in self._original_kernels
        )
        # Keep per-latent compiled/input metadata live for both continuous and
        # mixed fits so provider-backed prediction/sampling sees the same state
        # before and after save/load.
        self._configure_latent_kernels_for_fit(d)
        surface = surface_for_lmc(self._has_mixed_latents)
        warn_surface_status(surface, stacklevel=2)
        if self.ard:
            check_feature_support(TABLE_MAIN, surface, "ard", stacklevel=2)
        assert self._latent_compiled_kernels is not None
        for compiled_kernel in self._latent_compiled_kernels:
            guard_kernel_tree_features(surface, compiled_kernel, stacklevel=2)
        if fixed_observation_noise is not None:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "fixed_per_sample_per_task_noise",
                fail_on_in_dev=True,
                stacklevel=2,
            )
        if input_dependent_noise is not None:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "learned_input_dependent_noise",
                fail_on_in_dev=True,
                stacklevel=2,
            )
        if grouped_noise is not None:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "grouped_noise",
                fail_on_in_dev=True,
                stacklevel=2,
            )
        route_feature = (
            "materialized_training"
            if method == "materialized"
            else "matrix_free_training"
        )
        check_feature_support(TABLE_EXECUTION, surface, route_feature, stacklevel=2)
        progress_adapter = resolve_progress_adapter(
            progress,
            operation="train",
            model="multi_output_lmc",
            route=method,
            progress_stats=progress_stats,
        )

        if initial_params is not None and initial_lengthscales is not None:
            raise ValueError(
                "Provide either initial_params or initial_lengthscales for "
                "MultiOutputLMCGP.fit(), not both."
            )
        if initial_params is None and initial_lengthscales is not None:
            assert self._latent_compiled_kernels is not None
            initial_params = _build_lmc_initial_params_from_lengthscales(
                self._latent_compiled_kernels,
                np.asarray(initial_lengthscales, dtype=np.float32),
            )

        # ConstantMean: compute init_mean_per_task
        if self._init_mean is None:
            init_mean_per_task = np.mean(Y, axis=0).astype(np.float32)
        elif isinstance(self._init_mean, (int, float)):
            init_mean_per_task = np.full(T, float(self._init_mean), dtype=np.float32)
        else:
            init_mean_per_task = np.asarray(self._init_mean, dtype=np.float32)

        if has_categorical_latent:
            if fixed_observation_noise is not None:
                raise NotImplementedError(
                    "fixed_observation_noise for mixed MultiOutputLMCGP is not yet "
                    "supported; use continuous LMC or omit fixed observation noise."
                )
            return self._fit_mixed_composite(
                X,
                Y,
                n,
                d,
                T,
                R,
                max_iterations=max_iterations,
                learning_rate=learning_rate,
                init_params=initial_params,
                init_noise_per_task=resolved_initial_noise_per_task,
                verbose=verbose,
                early_stop_tol=early_stop_tol,
                early_stop_patience=early_stop_patience,
                init_mean_per_task=init_mean_per_task,
                use_fused_kernels=use_fused_kernels,
                fixed_observation_noise=fixed_observation_noise,
                progress_adapter=progress_adapter,
                progress_interval=progress_interval,
            )

        # All-continuous kernels go through the existing JIT LMC path.
        return self._fit_composite(
            X,
            Y,
            n,
            d,
            T,
            R,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            init_params=initial_params,
            init_noise_per_task=resolved_initial_noise_per_task,
            fixed_observation_noise=fixed_observation_noise,
            verbose=verbose,
            early_stop_tol=early_stop_tol,
            early_stop_patience=early_stop_patience,
            init_mean_per_task=init_mean_per_task,
            use_fused_kernels=use_fused_kernels,
            progress_adapter=progress_adapter,
            progress_interval=progress_interval,
        )

    def _fit_composite(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n: int,
        d: int,
        T: int,
        R: int,
        max_iterations: int = 100,
        learning_rate: float = 0.05,
        init_params: Optional[np.ndarray] = None,
        init_noise_per_task: Optional[np.ndarray] = None,
        fixed_observation_noise: Optional[np.ndarray] = None,
        verbose: bool = False,
        early_stop_tol: float = 1e-4,
        early_stop_patience: int = 15,
        init_mean_per_task: Optional[np.ndarray] = None,
        use_fused_kernels: bool = True,
        progress_adapter=None,
        progress_interval: int = 1,
    ) -> LMCTrainingResult:
        """Train LMC with composite kernels via JIT engine."""
        from .loader import load_kernel_module_engine, load_engine
        from .codegen_engine.compiler import make_module_name

        self._destroy_persistent_provider_infos(clear_modules=True)

        # _configure_latent_kernels_for_fit() is the single ARD/active-dim
        # transformation point. Reuse its compiled continuous kernels here so
        # ARD is not applied twice and prediction/save see the trained kernel
        # parameterization.
        if self._latent_compiled_kernels is not None:
            self.kernels = [copy.deepcopy(k) for k in self._latent_compiled_kernels]
            self._composite_kernel = self.kernels[0] if self.kernels else None
            self._kernel_modules = None

        # Per-latent param counts (heterogeneous kernels may have different counts)
        num_params_per_latent = [k.num_params() for k in self.kernels]
        total_params = sum(num_params_per_latent)

        latent_X_train_conts: List[np.ndarray] = []
        latent_C_trains: List[np.ndarray] = []
        for latent_idx in range(R):
            X_cont_s, C_s = self._transform_inputs_for_latent(latent_idx, X)
            latent_X_train_conts.append(X_cont_s)
            latent_C_trains.append(C_s)
        self._latent_X_train_conts = latent_X_train_conts
        self._latent_C_trains = latent_C_trains
        _ensure_lmc_multi_latent_jit_warmup()

        # Validate init_params BEFORE compilation (which is expensive)
        if init_params is not None:
            init_params = np.ascontiguousarray(init_params, dtype=np.float32)
            if init_params.shape == (total_params,):
                pass
            else:
                raise ValueError(
                    f"init_params must have shape ({total_params},) [flat concat of per-latent params]"
                    + f", got {init_params.shape}"
                )
        else:
            init_params = np.concatenate(
                [k.get_initial_params() for k in self.kernels]
            ).astype(np.float32)

        if R > 1:
            _ensure_lmc_multi_latent_jit_warmup()

        # Compile per-unique-kernel modules (cached by mojo type string)
        if self._kernel_modules is None:
            module_cache: Dict[Tuple[str, int], Any] = {}
            self._kernel_modules = []
            for latent_idx, k in enumerate(self.kernels):
                x_train_s = latent_X_train_conts[latent_idx]
                module_dim = int(x_train_s.shape[1])
                cache_key = (k.to_mojo_type(), module_dim)
                if cache_key not in module_cache:
                    module_name = make_module_name(k, d, "fn_ptr")
                    revoke_conflicting_provider_leases_by_name(
                        module_name,
                        self,
                        include_live_owners=True,
                    )
                    _cleanup_runtime_state()
                    module_cache[cache_key] = load_kernel_module_engine(
                        k,
                        dim=module_dim,
                        fresh_load=True,
                        isolated_load_id=self._isolated_load_id,
                        ncols_hint=_lmc_provider_ncols_hint(T),
                        verbose=verbose,
                    )
                self._kernel_modules.append(module_cache[cache_key])
            self._kernel_module = self._kernel_modules[0]
            self._engine = load_engine(
                fresh_load=True,
                isolated_load_id=self._isolated_load_id,
            )

        # Init noise per task
        if init_noise_per_task is not None:
            init_noise_per_task = np.ascontiguousarray(
                init_noise_per_task, dtype=np.float32
            )

        # Init provider for each latent via its own kernel module
        provider_infos = []
        init_engine_params_per_latent: List[np.ndarray] = []
        engine_trainable_masks_per_latent: List[np.ndarray] = []
        offset = 0
        for s in range(R):
            np_s = num_params_per_latent[s]
            p_slice = init_params[offset : offset + np_s]
            engine_p_slice = self.kernels[s].to_engine_params(p_slice)
            init_engine_params_per_latent.append(
                np.ascontiguousarray(engine_p_slice, dtype=np.float32)
            )
            engine_trainable_masks_per_latent.append(
                np.ascontiguousarray(
                    self.kernels[s].engine_trainable_mask(), dtype=np.bool_
                )
            )
            module_name = getattr(self._kernel_modules[s], "__name__", None)
            if module_name is not None:
                revoke_conflicting_provider_leases_by_name(
                    module_name,
                    self,
                    include_live_owners=True,
                )
                _cleanup_runtime_state()
            provider_info_s = self._kernel_modules[s].init_provider(
                latent_X_train_conts[s], engine_p_slice, 0.0
            )
            if self.method == "materialized":
                self._kernel_modules[s].materialize(provider_info_s)
            provider_infos.append(provider_info_s)
            offset += np_s

        # Train LMC via the explicit JIT engine binding signature.
        default_noise_per_task = (
            np.ascontiguousarray(init_noise_per_task, dtype=np.float32)
            if init_noise_per_task is not None
            else np.full(T, 0.1, dtype=np.float32)
        )
        default_mean_per_task = (
            np.ascontiguousarray(init_mean_per_task, dtype=np.float32)
            if init_mean_per_task is not None
            else np.zeros(T, dtype=np.float32)
        )
        fixed_noise_arg = (
            np.ascontiguousarray(fixed_observation_noise, dtype=np.float32)
            if fixed_observation_noise is not None
            else np.zeros((n, T), dtype=np.float32)
        )
        lmc_args = [
            provider_infos,
            init_engine_params_per_latent,
            engine_trainable_masks_per_latent,
            Y,
            T,
            max_iterations,
            float(learning_rate),
            verbose,
            self.num_probes,
            self.max_cg_iter,
            float(self.cg_tol),
            self.precond_rank,
            self.precond_method,
            float(self.precond_rebuild_threshold),
            self.max_tridiag_iter,
            default_noise_per_task,
            fixed_noise_arg,
            fixed_observation_noise is not None,
            default_mean_per_task,
        ]
        if progress_adapter is not None:
            lmc_args.extend([progress_adapter.callback, int(progress_interval)])

        try:
            raw = self._engine.train_lmc(*lmc_args)
        except Exception:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True)
            destroy_provider_infos(self._kernel_modules, provider_infos)
            raise
        if progress_adapter is not None:
            progress_adapter.close_if_needed()

        # Package results from JIT binding
        noise_per_task = np.array(raw["noise_per_task"], dtype=np.float32)

        # Extract per-latent params and derive lengthscales/outputscales
        raw_params = []
        raw_params_arrays = []
        for s in range(R):
            p = self.kernels[s].from_engine_params(
                np.array(raw["params_per_latent"][s], dtype=np.float32)
            )
            raw_params_arrays.append(p)
            raw_params.append(p.tolist())
        lengthscales, outputscales, lengthscales_per_dim = _summarize_lmc_kernel_params(
            self.kernels,
            raw_params_arrays,
            self.ard,
        )

        # Extract optional fields
        iter_times_raw = raw.get("iter_times_ms", None)
        iter_times_ms = (
            np.array(list(iter_times_raw), dtype=np.float64)
            if iter_times_raw is not None
            else None
        )

        mean_per_task_result = None
        if "mean_per_task" in raw:
            mean_per_task_result = np.array(raw["mean_per_task"], dtype=np.float32)

        var_diag_result_c = None
        if "var_diag" in raw:
            var_diag_result_c = np.array(raw["var_diag"], dtype=np.float32)

        # Build A_matrices array (shape [R, T, T])
        A_matrices_arr = (
            np.array(raw["A_matrices"], dtype=np.float32)
            if "A_matrices" in raw
            else None
        )

        # Derive B, Q, Lambda from A_matrices when not returned by engine
        # B = sum_s A_s [T, T]; Q, Lambda = eigh(B) for rotated prediction
        B_arr = np.array(raw["B"], dtype=np.float32) if "B" in raw else None
        Q_arr = np.array(raw["Q"], dtype=np.float32) if "Q" in raw else None
        Lambda_arr = (
            np.array(raw["Lambda"], dtype=np.float32) if "Lambda" in raw else None
        )
        effective_scales_arr = (
            np.array(raw["effective_scales"], dtype=np.float32)
            if "effective_scales" in raw
            else None
        )
        alpha_rotated_arr = (
            np.array(raw["alpha_rotated"], dtype=np.float32)
            if "alpha_rotated" in raw
            else None
        )

        if B_arr is None and A_matrices_arr is not None:
            B_arr = A_matrices_arr.sum(axis=0).astype(np.float32)  # [T, T]
        if Q_arr is None and B_arr is not None:
            Lambda_np, Q_np = np.linalg.eigh(B_arr.astype(np.float64))
            Q_arr = Q_np.astype(np.float32)
            Lambda_arr = Lambda_np.astype(np.float32)

        self._result = LMCTrainingResult(
            final_nll=float(raw["final_nll"]),
            nll_history=np.array(raw["nll_history"], dtype=np.float32),
            iterations=int(raw["iterations"]),
            converged=bool(raw["converged"]),
            num_latents=R,
            num_tasks=T,
            noise_per_task=noise_per_task,
            lengthscales=lengthscales,
            outputscales=outputscales,
            params_per_latent=raw_params,
            A_matrices=A_matrices_arr,
            L_factors=np.array(raw["L_factors"], dtype=np.float32)
            if "L_factors" in raw
            else None,
            B=B_arr,
            Q=Q_arr,
            Lambda=Lambda_arr,
            alpha=np.array(raw["alpha"], dtype=np.float32) if "alpha" in raw else None,
            alpha_rotated=alpha_rotated_arr,
            effective_scales=effective_scales_arr,
            use_ard=self.ard,
            lengthscales_per_dim=lengthscales_per_dim,
            mean_per_task=mean_per_task_result,
            var_diag=var_diag_result_c,
            iter_times_ms=iter_times_ms,
            fixed_observation_noise=fixed_observation_noise,
        )

        # Store learned composite params for prediction
        if "learned_params" in raw:
            learned = np.array(raw["learned_params"], dtype=np.float32)
            if learned.ndim == 2 and learned.shape[0] == R:
                self._learned_composite_params = np.stack(
                    [self.kernels[s].from_engine_params(learned[s]) for s in range(R)]
                ).astype(np.float32)
            else:
                self._learned_composite_params = learned
        else:
            self._learned_composite_params = None

        self._X_train = X
        self._Y_train = Y
        self._fixed_observation_noise = (
            np.ascontiguousarray(fixed_observation_noise, dtype=np.float32)
            if fixed_observation_noise is not None
            else None
        )
        self._raw_result = raw
        self._backend_train_info = build_backend_train_info(raw, self.method)
        if self._backend_train_info is not None:
            self._backend_train_info["use_preconditioner"] = bool(
                self.use_preconditioner
            )
            self._backend_train_info["uses_fixed_observation_noise"] = bool(
                fixed_observation_noise is not None
            )
        self._maybe_attach_specialization_metadata(self._backend_train_info)
        self._is_trained = True
        self._fitted_mean = mean_per_task_result
        self._backend_predict_info = None
        self._backend_sample_info = None
        self._provider_infos = provider_infos
        register_provider_lease(
            self._kernel_modules, self, self._destroy_persistent_provider_infos
        )

        return self._result

    def _fit_mixed_composite(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n: int,
        d: int,
        T: int,
        R: int,
        max_iterations: int = 100,
        learning_rate: float = 0.05,
        init_params: Optional[np.ndarray] = None,
        init_noise_per_task: Optional[np.ndarray] = None,
        verbose: bool = False,
        early_stop_tol: float = 1e-4,
        early_stop_patience: int = 15,
        init_mean_per_task: Optional[np.ndarray] = None,
        use_fused_kernels: bool = True,
        fixed_observation_noise: Optional[np.ndarray] = None,
        progress_adapter=None,
        progress_interval: int = 1,
    ) -> LMCTrainingResult:
        """Train mixed continuous+categorical LMC via the JIT engine."""
        from .loader import load_engine

        self._release_mixed_runtime_bundles()

        if early_stop_patience != 15 or early_stop_tol != 1e-4:
            warnings.warn(
                "early_stop_patience and early_stop_tol are not used: the JIT engine "
                "handles convergence internally via max_iterations.",
                UserWarning,
                stacklevel=2,
            )
        if not use_fused_kernels:
            warnings.warn(
                "use_fused_kernels=False has no effect: the JIT engine always uses "
                "fused GPU kernels.",
                UserWarning,
                stacklevel=2,
            )

        assert self._latent_compiled_kernels is not None
        assert self._latent_cat_specs is not None
        assert self._latent_cat_param_names is not None

        num_params_per_latent = [k.num_params() for k in self._latent_compiled_kernels]
        total_params = sum(num_params_per_latent)

        if init_params is not None:
            init_params = np.ascontiguousarray(init_params, dtype=np.float32)
            if init_params.shape == (total_params,):
                pass
            elif len(set(num_params_per_latent)) == 1 and init_params.shape == (
                R,
                num_params_per_latent[0],
            ):
                init_params = init_params.reshape(-1)
            else:
                np0 = num_params_per_latent[0]
                raise ValueError(
                    f"init_params must have shape ({total_params},) [flat concat of per-latent params]"
                    + (
                        f" or ({R}, {np0}) [homogeneous]"
                        if len(set(num_params_per_latent)) == 1
                        else ""
                    )
                    + f", got {init_params.shape}"
                )
        else:
            init_params = np.concatenate(
                [k.get_initial_params() for k in self._latent_compiled_kernels]
            ).astype(np.float32)

        if R > 1:
            _ensure_lmc_multi_latent_jit_warmup()

        if init_noise_per_task is not None:
            init_noise_per_task = np.ascontiguousarray(
                init_noise_per_task, dtype=np.float32
            )
        else:
            init_noise_per_task = np.full(T, 0.1, dtype=np.float32)

        if init_mean_per_task is None:
            init_mean_per_task = np.zeros(T, dtype=np.float32)
        else:
            init_mean_per_task = np.ascontiguousarray(
                init_mean_per_task, dtype=np.float32
            )

        init_params_per_latent: List[np.ndarray] = []
        latent_X_train_conts: List[np.ndarray] = []
        latent_C_trains: List[np.ndarray] = []
        cat_init_params_per_latent: List[np.ndarray] = []
        offset = 0
        for latent_idx in range(R):
            X_cont_s, C_s = self._transform_inputs_for_latent(latent_idx, X)
            latent_X_train_conts.append(X_cont_s)
            latent_C_trains.append(C_s)

            np_s = num_params_per_latent[latent_idx]
            p_slice = np.ascontiguousarray(
                init_params[offset : offset + np_s], dtype=np.float32
            )
            init_params_per_latent.append(p_slice)
            offset += np_s

            cat_init_params_per_latent.append(
                build_default_categorical_raw_params(self._latent_cat_specs[latent_idx])
            )

        self._engine = load_engine()

        # Publish the semantic mixed state before building runtime bundles so the
        # bundle manager and later lazy rebuilds see the same canonical inputs.
        self._X_train = X
        self._Y_train = Y
        self._latent_X_train_conts = latent_X_train_conts
        self._latent_C_trains = latent_C_trains

        training_bundle = self._build_mixed_runtime_bundle(
            init_params_per_latent,
            role=BUNDLE_ROLE_TRAINING,
            fresh_load=True,
            reclaim_live_owners=True,
            x_train_per_latent=latent_X_train_conts,
        )
        self._training_bundle = training_bundle
        self._inference_bundle = None
        self._set_runtime_state_from_bundle(training_bundle)

        try:
            train_args = [
                training_bundle.provider_infos,
                init_params_per_latent,
                Y,
                T,
                latent_C_trains,
                self._latent_cat_specs,
                cat_init_params_per_latent,
                int(max_iterations),
                float(learning_rate),
                bool(verbose),
                int(self.num_probes),
                int(self.max_cg_iter),
                float(self.cg_tol),
                int(self.precond_rank),
                int(self.precond_method),
                float(self.precond_rebuild_threshold),
                int(self.max_tridiag_iter),
                np.ascontiguousarray(init_noise_per_task, dtype=np.float32),
                np.ascontiguousarray(
                    fixed_observation_noise
                    if fixed_observation_noise is not None
                    else np.zeros((n, T), dtype=np.float32),
                    dtype=np.float32,
                ),
                fixed_observation_noise is not None,
                np.ascontiguousarray(init_mean_per_task, dtype=np.float32),
                1 if self.method == "materialized" else 0,
            ]
            if progress_adapter is not None:
                train_args.extend([progress_adapter.callback, int(progress_interval)])
            raw = self._engine.train_lmc_mixed(*train_args)
        except Exception:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True)
            if self._training_bundle is training_bundle:
                self._training_bundle = None
            destroy_provider_bundle(training_bundle)
            self._set_runtime_state_from_bundle(None)
            raise
        if progress_adapter is not None:
            progress_adapter.close_if_needed()

        noise_per_task = np.array(raw["noise_per_task"], dtype=np.float32)
        iter_times_raw = raw.get("iter_times_ms", None)
        iter_times_ms = (
            np.array(list(iter_times_raw), dtype=np.float64)
            if iter_times_raw is not None
            else None
        )
        raw_params = [
            np.array(p, dtype=np.float32).tolist() for p in raw["params_per_latent"]
        ]
        cat_params_per_latent = [
            np.array(p, dtype=np.float32).tolist() for p in raw["cat_params_per_latent"]
        ]

        lengthscales, outputscales, lengthscales_per_dim = _summarize_lmc_kernel_params(
            self._latent_compiled_kernels,
            [np.asarray(p, dtype=np.float32) for p in raw_params],
            self.ard,
        )

        mean_per_task_result = np.array(
            raw.get("mean_per_task", [0.0] * T), dtype=np.float32
        )
        var_diag_result = (
            np.array(raw["var_diag"], dtype=np.float32) if "var_diag" in raw else None
        )
        A_matrices_arr = np.array(raw["A_matrices"], dtype=np.float32)
        B_arr = A_matrices_arr.sum(axis=0).astype(np.float32)
        Lambda_np, Q_np = np.linalg.eigh(B_arr.astype(np.float64))
        Q_arr = Q_np.astype(np.float32)
        Lambda_arr = Lambda_np.astype(np.float32)

        self._result = LMCTrainingResult(
            final_nll=float(raw["final_nll"]),
            nll_history=np.array(raw["nll_history"], dtype=np.float32),
            iterations=int(raw["iterations"]),
            converged=bool(raw["converged"]),
            num_latents=R,
            num_tasks=T,
            noise_per_task=noise_per_task,
            lengthscales=lengthscales,
            outputscales=outputscales,
            params_per_latent=raw_params,
            A_matrices=A_matrices_arr,
            L_factors=np.array(raw["L_factors"], dtype=np.float32)
            if "L_factors" in raw
            else None,
            B=B_arr,
            Q=Q_arr,
            Lambda=Lambda_arr,
            alpha=np.array(raw["alpha"], dtype=np.float32) if "alpha" in raw else None,
            alpha_rotated=None,
            effective_scales=Lambda_arr,
            use_ard=self.ard,
            lengthscales_per_dim=lengthscales_per_dim,
            mean_per_task=mean_per_task_result,
            var_diag=var_diag_result,
            cat_params_per_latent=cat_params_per_latent,
            cat_param_names_per_latent=self._latent_cat_param_names,
            iter_times_ms=iter_times_ms,
        )

        self._X_train = X
        self._Y_train = Y
        self._fixed_observation_noise = (
            np.ascontiguousarray(fixed_observation_noise, dtype=np.float32)
            if fixed_observation_noise is not None
            else None
        )
        self._raw_result = raw
        self._backend_train_info = build_backend_train_info(raw, self.method)
        if self._backend_train_info is not None:
            self._backend_train_info["use_preconditioner"] = bool(
                self.use_preconditioner
            )
        self._maybe_attach_specialization_metadata(self._backend_train_info)
        self._is_trained = True
        self._fitted_mean = mean_per_task_result
        self._backend_predict_info = None
        self._backend_sample_info = None
        self._latent_X_train_conts = latent_X_train_conts
        self._latent_C_trains = latent_C_trains
        self._training_bundle = training_bundle
        self._set_runtime_state_from_bundle(training_bundle)

        return self._result

    def predict(
        self,
        X_test: np.ndarray,
        return_var: bool = False,
        return_std: bool = False,
        variance_method: str = "love",
        progress: Any = None,
        progress_stats: Optional[Any] = None,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        MultiOutputPredictionResult,
    ]:
        """Predict at test points.

        For R=1, uses the ICM prediction path (eigendecomposition of B).
        For R>1, uses the full LMC prediction with per-latent kernels and
        A_s matrices, which correctly handles heterogeneous kernel types.

        Args:
            X_test: Test inputs [m, d], float32
            return_var: If True, return (mean, variance) tuple
            return_std: If True, return (mean, std) tuple
            variance_method: Variance computation method. One of:
                - 'love' (default): Fast low-rank approximation via LOVE/Lanczos
                - 'exact': Exact CG-based variance (preconditioned, more accurate but slower)
            progress: Progress reporting control. Use True, ``"auto"``, a callback,
                or a reporter object with start/update/close methods.
            progress_stats: Optional stat names or callable used by the default
                tqdm reporter.

        Returns:
            If return_var: (mean [m,T], variance [m,T])
            If return_std: (mean [m,T], std [m,T])
            Otherwise: MultiOutputPredictionResult
        """
        if not self._is_trained:
            raise RuntimeError(
                "GP must be trained before prediction. Call fit() first."
            )

        _VALID_VARIANCE_METHODS = ("love", "exact", "mean_only")
        if variance_method not in _VALID_VARIANCE_METHODS:
            raise ValueError(
                f"variance_method must be one of {_VALID_VARIANCE_METHODS}, "
                f"got '{variance_method}'"
            )
        if self._fixed_observation_noise is not None and variance_method == "love":
            raise NotImplementedError(
                "MultiOutputLMCGP fixed_observation_noise supports exact prediction "
                "and pathwise sampling on continuous LMC routes; LOVE variance with "
                "fixed training noise is not yet exposed. Use variance_method='exact' "
                "or 'mean_only'."
            )

        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        if X_test.ndim != 2:
            raise ValueError(f"X_test must be 2D [m, d], got shape {X_test.shape}")
        if X_test.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X_test has {X_test.shape[1]} features, "
                f"expected {self._X_train.shape[1]}"
            )
        if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
            raise ValueError("X_test contains NaN or Inf values")

        surface = surface_for_lmc(self._has_mixed_latents)
        variance_feature = {
            "mean_only": "mean_only",
            "exact": "exact_variance",
            "love": "love_variance",
        }[variance_method]
        check_feature_support(TABLE_PREDICTION, surface, variance_feature, stacklevel=2)

        train_info = self._backend_train_info or {}
        prediction_route = train_info.get("training_route", self.method)
        progress_adapter = resolve_progress_adapter(
            progress,
            operation="predict",
            model="multi_output_lmc",
            route=prediction_route,
            progress_stats=progress_stats,
        )
        progress_total = 3
        if progress_adapter is not None:
            base_stats = prediction_progress_stats(
                n_test=X_test.shape[0],
                variance_method=variance_method,
            )
            progress_adapter.emit(
                phase="start",
                current=0,
                total=progress_total,
                stats=base_stats,
            )
            progress_adapter.emit(
                phase="backend",
                current=1,
                total=progress_total,
                message="Running backend prediction",
                stats=base_stats,
            )

        try:
            # When returning a MultiOutputPredictionResult (neither return_var nor
            # return_std), always compute variance so the result object is complete.
            # When returning a tuple via return_var/return_std, variance is always needed.
            compute_variance = variance_method != "mean_only"
            # JIT engine contract: 0=mean_only, 1=LOVE, 2=exact
            if variance_method == "exact":
                var_method_int = 2
            elif variance_method == "love" or variance_method is None:
                var_method_int = 1
            elif variance_method == "mean_only":
                var_method_int = 0
            else:
                var_method_int = 0

            # All kernels go through Python-side composite prediction.
            final_result = self._predict_composite(
                X_test,
                compute_variance,
                return_var,
                return_std,
                variance_method=var_method_int,
            )
            final_stats = prediction_progress_stats(
                n_test=X_test.shape[0],
                variance_method=variance_method,
                backend_info=self._backend_predict_info,
            )
            if progress_adapter is not None:
                progress_adapter.emit(
                    phase="variance" if variance_method != "mean_only" else "mean",
                    current=2,
                    total=progress_total,
                    stats=final_stats,
                )
        except Exception as exc:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True, message=str(exc))
            raise

        if progress_adapter is not None:
            progress_adapter.emit(
                phase="complete",
                current=progress_total,
                total=progress_total,
                stats=final_stats,
            )
        return final_result

    def predict_latent(
        self,
        X_test: np.ndarray,
        return_var: bool = False,
        return_std: bool = False,
        variance_method: str = "love",
        **kwargs,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        MultiOutputPredictionResult,
    ]:
        """Predict the latent task functions ``p(f_test | y_train)``."""
        return self.predict(
            X_test,
            return_var=return_var,
            return_std=return_std,
            variance_method=variance_method,
            **kwargs,
        )

    def predict_observed(
        self,
        X_test: np.ndarray,
        observation_noise: Optional[np.ndarray] = None,
        noise_group_test: Optional[np.ndarray] = None,
        return_var: bool = False,
        return_std: bool = False,
        variance_method: str = "love",
        **kwargs,
    ) -> Union[Tuple[np.ndarray, np.ndarray], MultiOutputPredictionResult]:
        """Predict observed task responses ``p(y_test | y_train)``.

        Observed prediction requires explicit test-point noise. MojoGP never
        reuses or averages training noise for new points.
        """
        latent = self.predict(
            X_test,
            return_var=False,
            return_std=False,
            variance_method=variance_method,
            **kwargs,
        )
        assert isinstance(latent, MultiOutputPredictionResult)
        noise = _resolve_observed_noise_matrix(
            latent.mean.shape[0],
            latent.mean.shape[1],
            observation_noise=observation_noise,
            noise_group_test=noise_group_test,
            group_noise=self._noise_group_values,
        )
        return _format_observed_prediction(latent, noise, return_var, return_std)

    def _predict_composite(
        self,
        X_test: np.ndarray,
        compute_variance: bool,
        return_var: bool,
        return_std: bool,
        variance_method: int = 1,
    ):
        """Predict with composite LMC kernels using GPU bindings when available.

        variance_method follows the engine contract (0=mean_only, 1=LOVE,
        2=exact). GPU prediction honors that routing when available, including
        mixed-LMC exact variance through the dedicated backend route.

        Mean: mean[m, t] = sum_s sum_t' A_s[t, t'] * K_test_s[m, :] @ alpha[:, t']
        Variance: var[i, t] = sum_s ||A_s[t,:]||^2 * v_s[i] + noise_t,
        where v_s[i] is the scalar posterior variance for latent s at test point i.
        """
        R = self._result.num_latents
        T = self._result.num_tasks
        n = self._X_train.shape[0]
        m = X_test.shape[0]

        # alpha from engine is [n*T] flat; reshape to [n, T] for matmul
        alpha = np.ascontiguousarray(self._result.alpha, dtype=np.float32).reshape(n, T)
        A_matrices = np.ascontiguousarray(
            self._result.A_matrices, dtype=np.float32
        )  # [R, T, T]

        # Per-latent params from training result (supports heterogeneous kernel types)
        params_per_latent = self._result.params_per_latent
        cat_params_per_latent = self._result.cat_params_per_latent
        if params_per_latent is None:
            raise RuntimeError(
                "LMC prediction requires params_per_latent in the current "
                "training result. Refit the model with the current MojoGP version."
            )

        # Use average noise for the regularized system solve
        noise = self._result.noise_per_task  # [T]
        avg_noise = float(np.mean(noise))

        # Compute mean and optionally cache K_cross for variance
        mean = np.zeros((m, T), dtype=np.float32)
        variance = None
        # Cache per-latent K_cross matrices if we need variance
        K_cross_cache = [] if compute_variance else None
        mean_has_offset = False

        # --- GPU mean prediction path (faster for large n) ---
        # Uses predict_lmc() engine binding when provider_infos are available
        gpu_mean_computed = False
        gpu_variance_computed = False
        selected_bundle_role = None
        selected_runtime_method = self.method
        provider_infos = self._provider_infos
        built_provider_infos = None
        provider_kernel_modules = self._kernel_modules
        force_isolated_mixed_exact = (
            self._has_mixed_latents
            and variance_method == 2
            and self.method == "materialized"
        )
        use_dense_exact_lmc_variance = (
            compute_variance
            and variance_method == 2
            and self.method == "materialized"
            and not self._has_mixed_latents
        )
        use_full_backend_exact_variance = (
            compute_variance
            and variance_method == 2
            and not use_dense_exact_lmc_variance
        )
        use_per_latent_predict_inputs = self._uses_per_latent_runtime_inputs()
        if self._has_mixed_latents and params_per_latent is not None:
            (
                provider_infos,
                provider_kernel_modules,
                selected_bundle_role,
                selected_runtime_method,
            ) = self._resolve_mixed_provider_runtime(
                params_per_latent,
                canonical=force_isolated_mixed_exact,
            )
        elif provider_infos is None and (
            params_per_latent is not None
            and self._kernel_modules is not None
            and self._X_train is not None
        ):
            if R > 1:
                _ensure_lmc_multi_latent_jit_warmup()
            if (
                use_per_latent_predict_inputs
                and self._latent_X_train_conts is not None
            ):
                x_train_per_latent = self._latent_X_train_conts
            else:
                x_train_per_latent = [self._X_train for _ in range(R)]
            provider_infos = rebuild_trained_provider_infos(
                self._kernel_modules,
                x_train_per_latent,
                params_per_latent,
                self.method,
                param_kernels=None if self._has_mixed_latents else self.kernels,
            )
            built_provider_infos = provider_infos

        can_use_gpu_lmc = (
            provider_infos is not None
            and self._engine is not None
            and params_per_latent is not None
        )
        if can_use_gpu_lmc:
            alpha_flat = np.ascontiguousarray(
                self._result.alpha, dtype=np.float32
            ).ravel()
            if self._has_mixed_latents:
                params_list = [
                    np.ascontiguousarray(params_per_latent[s], dtype=np.float32)
                    for s in range(R)
                ]
            else:
                params_list = [
                    np.ascontiguousarray(
                        self.kernels[s].to_engine_params(
                            np.ascontiguousarray(params_per_latent[s], dtype=np.float32)
                        ),
                        dtype=np.float32,
                    )
                    for s in range(R)
                ]
            mean_per_task = np.ascontiguousarray(
                self._fitted_mean
                if self._fitted_mean is not None
                else np.zeros(T, dtype=np.float32),
                dtype=np.float32,
            )
            backend_variance_method = variance_method if compute_variance else 0
            if use_dense_exact_lmc_variance or use_full_backend_exact_variance:
                backend_variance_method = 0
            predict_rank = _prediction_lanczos_rank(
                self.num_probes,
                selected_runtime_method,
                is_mixed=self._has_mixed_latents,
                is_ard=self.ard,
            )
            backend_route = None

            if use_per_latent_predict_inputs and hasattr(self._engine, "predict_lmc_mixed"):
                backend_route = "predict_lmc_mixed"
                latent_x_test_conts = []
                latent_c_tests = []
                for latent_idx in range(R):
                    x_test_cont_s, c_test_s = self._transform_inputs_for_latent(
                        latent_idx, X_test
                    )
                    latent_x_test_conts.append(
                        np.ascontiguousarray(x_test_cont_s, dtype=np.float32)
                    )
                    latent_c_tests.append(
                        np.ascontiguousarray(c_test_s, dtype=np.int32)
                    )
                try:
                    gpu_result = self._engine.predict_lmc_mixed(
                        provider_infos,
                        alpha_flat,
                        A_matrices,
                        latent_x_test_conts,
                        params_list,
                        mean_per_task,
                        n,
                        T,
                        list(self._latent_is_mixed),
                        self._latent_C_trains,
                        latent_c_tests,
                        [
                            np.ascontiguousarray(cp, dtype=np.float32)
                            for cp in (
                                cat_params_per_latent
                                if cat_params_per_latent is not None
                                else [np.zeros(0, dtype=np.float32) for _ in range(R)]
                            )
                        ],
                        self._latent_cat_specs,
                        np.ascontiguousarray(noise, dtype=np.float32),
                        backend_variance_method,
                        self.max_cg_iter,
                        float(self.cg_tol),
                        self.precond_rank,
                        predict_rank,
                        1 if selected_runtime_method == "materialized" else 0,
                        int(self.precond_method),
                    )
                except Exception as exc:
                    self._backend_predict_info = build_backend_predict_info(
                        requested_method=self.method,
                        actual_prediction_route=backend_route,
                        backend_prediction_used=False,
                        backend_variance_used=False,
                        variance_method={0: "mean_only", 1: "love", 2: "exact"}.get(
                            variance_method, "mean_only"
                        ),
                        fallback_used=False,
                        backend_error=str(exc),
                        training_route=(
                            self._backend_train_info.get("training_route")
                            if self._backend_train_info is not None
                            else self.method
                        ),
                        precond_rank=self.precond_rank,
                        precond_method=self.precond_method,
                        precond_rebuild_count=(
                            self._backend_train_info.get("precond_rebuild_count")
                            if self._backend_train_info is not None
                            else None
                        ),
                    )
                    raise RuntimeError(
                        f"LMC backend prediction failed on route '{backend_route}': {exc}"
                    ) from exc
            elif hasattr(self._engine, "predict_lmc"):
                backend_route = "predict_lmc"
                try:
                    gpu_result = self._engine.predict_lmc(
                        provider_infos,
                        alpha_flat,
                        A_matrices,
                        X_test.astype(np.float32),
                        params_list,
                        mean_per_task,
                        n,
                        T,
                        np.ascontiguousarray(noise, dtype=np.float32),
                        backend_variance_method,
                        self.max_cg_iter,
                        float(self.cg_tol),
                        self.precond_rank,
                        predict_rank,
                    )
                except Exception as exc:
                    self._backend_predict_info = build_backend_predict_info(
                        requested_method=self.method,
                        actual_prediction_route=backend_route,
                        backend_prediction_used=False,
                        backend_variance_used=False,
                        variance_method={0: "mean_only", 1: "love", 2: "exact"}.get(
                            variance_method, "mean_only"
                        ),
                        fallback_used=False,
                        backend_error=str(exc),
                        training_route=(
                            self._backend_train_info.get("training_route")
                            if self._backend_train_info is not None
                            else self.method
                        ),
                        precond_rank=self.precond_rank,
                        precond_method=self.precond_method,
                        precond_rebuild_count=(
                            self._backend_train_info.get("precond_rebuild_count")
                            if self._backend_train_info is not None
                            else None
                        ),
                    )
                    raise RuntimeError(
                        f"LMC backend prediction failed on route '{backend_route}': {exc}"
                    ) from exc
            else:
                gpu_result = None

            if gpu_result is None:
                self._backend_predict_info = build_backend_predict_info(
                    requested_method=self.method,
                    actual_prediction_route="python_fallback",
                    backend_prediction_used=False,
                    backend_variance_used=False,
                    variance_method={0: "mean_only", 1: "love", 2: "exact"}.get(
                        variance_method, "mean_only"
                    ),
                    fallback_used=True,
                    actual_variance_route="python_fallback"
                    if compute_variance
                    else None,
                    training_route=(
                        self._backend_train_info.get("training_route")
                        if self._backend_train_info is not None
                        else self.method
                    ),
                    precond_rank=self.precond_rank,
                    precond_method=self.precond_method,
                    precond_rebuild_count=(
                        self._backend_train_info.get("precond_rebuild_count")
                        if self._backend_train_info is not None
                        else None
                    ),
                )
            else:
                mean = np.array(gpu_result["mean"], dtype=np.float32, copy=True)
                gpu_mean_computed = True
                mean_has_offset = True
                self._backend_predict_info = build_backend_predict_info(
                    requested_method=self.method,
                    actual_prediction_route=backend_route,
                    backend_prediction_used=True,
                    backend_variance_used=(
                        bool(compute_variance)
                        and backend_variance_method != 0
                        and "variance" in gpu_result
                    ),
                    variance_method={0: "mean_only", 1: "love", 2: "exact"}.get(
                        variance_method, "mean_only"
                    ),
                    fallback_used=bool(
                        compute_variance and "variance" not in gpu_result
                    ),
                    actual_variance_route=(
                        backend_route
                        if compute_variance and "variance" in gpu_result
                        else ("python_fallback" if compute_variance else None)
                    ),
                    training_route=(
                        self._backend_train_info.get("training_route")
                        if self._backend_train_info is not None
                        else self.method
                    ),
                    precond_rank=self.precond_rank,
                    precond_method=self.precond_method,
                    precond_rebuild_count=(
                        self._backend_train_info.get("precond_rebuild_count")
                        if self._backend_train_info is not None
                        else None
                    ),
                )
                if selected_bundle_role is not None:
                    self._backend_predict_info["provider_bundle_role"] = (
                        selected_bundle_role
                    )
                if compute_variance and "variance" in gpu_result:
                    variance = np.maximum(
                        np.ascontiguousarray(gpu_result["variance"], dtype=np.float32),
                        1e-10,
                    )
                    gpu_variance_computed = True

            if use_full_backend_exact_variance and provider_infos is not None:
                variance, exact_route = self._predict_lmc_full_exact_variance_backend(
                    provider_infos,
                    A_matrices,
                    X_test,
                    params_per_latent,
                    cat_params_per_latent,
                    noise,
                    runtime_method=selected_runtime_method,
                )
                gpu_variance_computed = True
                if self._backend_predict_info is not None:
                    self._backend_predict_info["backend_variance_used"] = True
                    self._backend_predict_info["fallback_used"] = False
                    self._backend_predict_info["actual_variance_route"] = exact_route

        if built_provider_infos is not None and provider_kernel_modules is not None:
            destroy_provider_infos(provider_kernel_modules, built_provider_infos)

        if self._backend_predict_info is None:
            self._backend_predict_info = build_backend_predict_info(
                requested_method=self.method,
                actual_prediction_route="python_fallback",
                backend_prediction_used=False,
                backend_variance_used=False,
                variance_method={0: "mean_only", 1: "love", 2: "exact"}.get(
                    variance_method, "mean_only"
                ),
                fallback_used=True,
                actual_variance_route="python_fallback" if compute_variance else None,
                training_route=(
                    self._backend_train_info.get("training_route")
                    if self._backend_train_info is not None
                    else self.method
                ),
                precond_rank=self.precond_rank,
                precond_method=self.precond_method,
                precond_rebuild_count=(
                    self._backend_train_info.get("precond_rebuild_count")
                    if self._backend_train_info is not None
                    else None
                ),
            )
        if selected_bundle_role is not None and self._backend_predict_info is not None:
            self._backend_predict_info["provider_bundle_role"] = selected_bundle_role

        if use_dense_exact_lmc_variance:
            variance = self._predict_lmc_dense_exact_variance(
                X_test,
                params_per_latent,
                cat_params_per_latent,
                A_matrices,
                noise,
            )
            gpu_variance_computed = True
            if self._backend_predict_info is not None:
                self._backend_predict_info["backend_variance_used"] = False
                self._backend_predict_info["fallback_used"] = False
                self._backend_predict_info["actual_variance_route"] = (
                    "dense_exact_lmc"
                )

        if not gpu_mean_computed:
            for s in range(R):
                params_s = np.array(params_per_latent[s], dtype=np.float32)
                cat_params_s = (
                    None
                    if cat_params_per_latent is None or s >= len(cat_params_per_latent)
                    else np.array(cat_params_per_latent[s], dtype=np.float32)
                )
                if self._has_mixed_latents:
                    K_cross_s = self._evaluate_latent_kernel_matrix(
                        s, X_test, self._X_train, params_s, cat_params_s
                    )
                else:
                    kernel_s = self.kernels[s]
                    K_cross_s = kernel_s.evaluate(
                        X_test, self._X_train, params=params_s
                    )  # [m, n]

                # K_cross_s @ alpha gives [m, T]
                K_alpha_s = K_cross_s @ alpha  # [m, T]

                # Weight by A_s: mean += A_s @ K_alpha_s^T for each test point
                mean += K_alpha_s @ A_matrices[s].T  # [m, T] @ [T, T]^T = [m, T]

                if K_cross_cache is not None:
                    K_cross_cache.append(K_cross_s)
        else:
            # GPU mean computed — still need K_cross cache for variance
            if K_cross_cache is not None and not gpu_variance_computed:
                for s in range(R):
                    params_s = np.array(params_per_latent[s], dtype=np.float32)
                    cat_params_s = (
                        None
                        if cat_params_per_latent is None
                        or s >= len(cat_params_per_latent)
                        else np.array(cat_params_per_latent[s], dtype=np.float32)
                    )
                    if self._has_mixed_latents:
                        K_cross_s = self._evaluate_latent_kernel_matrix(
                            s, X_test, self._X_train, params_s, cat_params_s
                        )
                    else:
                        kernel_s = self.kernels[s]
                        K_cross_s = kernel_s.evaluate(
                            X_test, self._X_train, params=params_s
                        )  # [m, n]
                    K_cross_cache.append(K_cross_s)

        if compute_variance and not gpu_variance_computed and self.method == "matrix_free":
            variance_label = {1: "love", 2: "exact"}.get(variance_method, "requested")
            backend_error = (
                "Matrix-free MultiOutputLMCGP "
                f"{variance_label} variance requires a backend route; refusing dense "
                "Python fallback that materializes train-train kernels."
            )
            if self._backend_predict_info is not None:
                self._backend_predict_info["fallback_used"] = False
                self._backend_predict_info["backend_variance_used"] = False
                self._backend_predict_info["actual_variance_route"] = (
                    "forbidden_python_fallback"
                )
                self._backend_predict_info["backend_error"] = backend_error
            raise RuntimeError(backend_error)

        if compute_variance and not gpu_variance_computed:
            # CPU fallback: compute scalar posterior variance per latent, then
            # mix back to tasks using row-norm A_s weighting.
            variance = np.zeros((m, T), dtype=np.float32)

            if (
                self._backend_predict_info is not None
                and self._backend_predict_info.get("backend_prediction_used")
            ):
                self._backend_predict_info["fallback_used"] = True
                self._backend_predict_info["backend_variance_used"] = False

            for s in range(R):
                params_s = np.array(params_per_latent[s], dtype=np.float32)
                cat_params_s = (
                    None
                    if cat_params_per_latent is None or s >= len(cat_params_per_latent)
                    else np.array(cat_params_per_latent[s], dtype=np.float32)
                )
                K_cross_s = K_cross_cache[s]  # [m, n]

                # Build K_train_s + noise*I
                if self._has_mixed_latents:
                    K_train_s = self._evaluate_latent_kernel_matrix(
                        s, self._X_train, self._X_train, params_s, cat_params_s
                    )
                else:
                    kernel_s = self.kernels[s]
                    K_train_s = kernel_s.evaluate(
                        self._X_train, self._X_train, params=params_s
                    )  # [n, n]
                K_train_s += avg_noise * np.eye(n, dtype=np.float32)

                # Solve K_train_s @ V_s = K_cross_s^T  →  V_s [n, m]
                V_s = np.linalg.solve(K_train_s, K_cross_s.T)  # [n, m]

                # Variance reduction per test point: diag(K_cross_s @ V_s)
                # = sum_j K_cross_s[i,j] * V_s[j,i] for each i
                var_reduction_s = np.sum(K_cross_s * V_s.T, axis=1)  # [m]

                # k_s(x_i, x_i) for test points
                if self._has_mixed_latents:
                    K_diag_s = np.diag(
                        self._evaluate_latent_kernel_matrix(
                            s, X_test, X_test, params_s, cat_params_s
                        )
                    ).astype(np.float32)
                else:
                    K_diag_s = np.array(
                        [
                            kernel_s.evaluate(
                                X_test[i : i + 1], X_test[i : i + 1], params=params_s
                            )[0, 0]
                            for i in range(m)
                        ],
                        dtype=np.float32,
                    )  # [m]

                latent_var_s = np.maximum(K_diag_s - var_reduction_s, 1e-10)
                row_weights = np.sum(A_matrices[s] ** 2, axis=1).astype(np.float32)
                for t in range(T):
                    variance[:, t] += row_weights[t] * latent_var_s

            # Add per-task noise
            for t in range(T):
                variance[:, t] += noise[t]

            variance = np.maximum(variance, 1e-10).astype(np.float32)
        elif gpu_variance_computed:
            variance = np.maximum(variance, 1e-10).astype(np.float32)

        self._annotate_lmc_variance_metadata(
            compute_variance=compute_variance,
            noise_per_task=noise,
        )

        # ConstantMean: add per-task mean offset to composite LMC predictions
        if self._fitted_mean is not None and not mean_has_offset:
            mean = mean + self._fitted_mean[np.newaxis, :]

        if return_var:
            return mean, variance
        if return_std:
            std = np.sqrt(variance) if variance is not None else None
            return mean, std

        return MultiOutputPredictionResult(
            mean=mean,
            variance=variance,
            std=np.sqrt(variance) if variance is not None else None,
        )

    def sample_posterior(
        self,
        X_test: np.ndarray,
        n_samples: int = 1,
        method: str = "diagonal",
        n_rff_features: int = 1024,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Draw samples from the posterior predictive distribution.

        Args:
            X_test: Test points [m, d]
            n_samples: Number of posterior samples to draw
            method: Sampling method:
                - 'diagonal' (default): Independent samples using predictive std.
                  Fast, O(m), but ignores correlations between test points.
                - 'pathwise': Approximate correlated posterior samples using
                  shared latent feature maps plus backend LMC correction.
            n_rff_features: Number of random Fourier features used by the
                pathwise latent prior samplers.
            rng: Optional numpy random Generator for reproducibility.

        Returns:
            samples: Array of shape [n_samples, m, T]

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before sampling. Call fit() first.")
        surface = surface_for_lmc(self._has_mixed_latents)
        if method == "cholesky":
            check_feature_support(
                TABLE_SAMPLING, surface, "cholesky_sampling", stacklevel=2
            )
        if method not in ("diagonal", "pathwise"):
            raise ValueError(
                f"method must be 'diagonal' or 'pathwise', got '{method}'"
            )

        if rng is None:
            rng = np.random.default_rng()

        requested_method = method
        train_info = self._backend_train_info or {}
        training_route = train_info.get("training_route", self.method)

        if method == "pathwise":
            check_feature_support(
                TABLE_SAMPLING, surface, "pathwise_sampling", stacklevel=2
            )
            kernels = self._latent_compiled_kernels or self.kernels
            if any(
                kernel_tree_contains_kernel_name(kernel, "POLYNOMIAL")
                for kernel in kernels
            ):
                check_feature_support(
                    TABLE_SAMPLING, surface, "polynomial_pathwise", stacklevel=2
                )
            self._pathwise_bundle_role = None
            samples = self._sample_posterior_pathwise(
                X_test,
                n_samples=n_samples,
                n_rff_features=n_rff_features,
                rng=rng,
            )
            backend_correction_route = (
                "sample_lmc_mixed_pathwise"
                if self._has_mixed_latents or self._uses_per_latent_runtime_inputs()
                else "sample_lmc_pathwise"
            )
            self._backend_sample_info = {
                "requested_method": requested_method,
                "actual_sampling_method": "pathwise",
                "actual_sampling_route": "provider_pathwise",
                "backend_sampling_used": True,
                "backend_correction_used": True,
                "backend_correction_route": backend_correction_route,
                "training_route": training_route,
                "prior_sampler_family": "shared_feature_map",
                "n_rff_features": int(n_rff_features),
            }
            if self._pathwise_bundle_role is not None:
                self._backend_sample_info["provider_bundle_role"] = (
                    self._pathwise_bundle_role
                )
            return samples

        check_feature_support(TABLE_SAMPLING, surface, "diagonal_sampling", stacklevel=2)
        mean, std = self.predict(X_test, return_std=True)
        # mean: [m, T], std: [m, T]
        z = rng.standard_normal((n_samples, mean.shape[0], mean.shape[1]))
        samples = mean[np.newaxis, :, :] + std[np.newaxis, :, :] * z
        self._backend_sample_info = {
            "requested_method": requested_method,
            "actual_sampling_method": "diagonal",
            "actual_sampling_route": "diagonal_from_predictive_std",
            "backend_sampling_used": True,
            "backend_correction_used": False,
            "training_route": training_route,
        }
        return samples.astype(np.float32)

    def _sample_posterior_pathwise(
        self,
        X_test: np.ndarray,
        n_samples: int,
        n_rff_features: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Approximate correlated posterior samples via backend LMC correction."""
        if self._result is None or self._result.params_per_latent is None:
            raise RuntimeError(
                "LMC provider state is unavailable for pathwise sampling."
            )
        backend_correction_name = (
            "sample_lmc_mixed_pathwise"
            if self._has_mixed_latents or self._uses_per_latent_runtime_inputs()
            else "sample_lmc_pathwise"
        )

        result = self._result
        R = int(result.num_latents)
        T = int(result.num_tasks)
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        y_centered = np.ascontiguousarray(
            self._Y_train.astype(np.float32)
            - np.asarray(
                self._fitted_mean
                if self._fitted_mean is not None
                else result.mean_per_task,
                dtype=np.float32,
            )[np.newaxis, :],
            dtype=np.float32,
        )
        noise_per_task = np.ascontiguousarray(result.noise_per_task, dtype=np.float32)
        A_matrices = np.ascontiguousarray(result.A_matrices, dtype=np.float32)
        mean_per_task = np.ascontiguousarray(
            self._fitted_mean
            if self._fitted_mean is not None
            else result.mean_per_task,
            dtype=np.float32,
        )

        if (
            self._uses_per_latent_runtime_inputs()
            and self._latent_X_train_conts is not None
        ):
            x_train_per_latent = self._latent_X_train_conts
        else:
            x_train_per_latent = [self._X_train for _ in range(R)]
        selected_bundle_role = None
        selected_runtime_method = self.method
        provider_infos = self._provider_infos
        built_provider_infos = None
        provider_kernel_modules = self._kernel_modules
        if (
            self._has_mixed_latents
            and self.method == "materialized"
            and self._training_bundle is not None
        ):
            # The fit-time materialized providers may hold the last optimizer
            # state, while the result stores the best/final params. Rebuild a
            # pathwise inference bundle from result params for reproducibility.
            self._release_mixed_runtime_bundles()
            provider_infos = None
            provider_kernel_modules = None
        use_isolated_mixed_pathwise_providers = (
            self._has_mixed_latents
            and (
                self.method == "materialized"
                or (self._training_bundle is None and self._provider_infos is None)
            )
        )
        if self._has_mixed_latents:
            (
                provider_infos,
                provider_kernel_modules,
                selected_bundle_role,
                selected_runtime_method,
            ) = self._resolve_mixed_provider_runtime(
                result.params_per_latent,
                canonical=use_isolated_mixed_pathwise_providers,
            )
        elif provider_infos is None:
            if R > 1:
                _ensure_lmc_multi_latent_jit_warmup()
            provider_infos = rebuild_trained_provider_infos(
                self._kernel_modules,
                x_train_per_latent,
                result.params_per_latent,
                self.method,
                param_kernels=None if self._has_mixed_latents else self.kernels,
            )
            built_provider_infos = provider_infos
        if self._engine is None or not hasattr(self._engine, backend_correction_name):
            raise RuntimeError(
                "The loaded JIT engine does not expose "
                f"{backend_correction_name}(). Rebuild it with `task build`."
            )
        latent_x_test_conts: List[np.ndarray] = []
        latent_c_tests: List[np.ndarray] = []
        latent_feature_maps = []
        latent_train_conts = []
        latent_train_cats = []
        latent_mix_factors = []

        latent_compiled_kernels = self._latent_compiled_kernels
        latent_X_train_conts = self._latent_X_train_conts
        latent_C_trains = getattr(self, "_latent_C_trains", None)
        latent_analyses = getattr(self, "_latent_analyses", None)
        latent_cat_specs = getattr(self, "_latent_cat_specs", None)
        use_latent_input_transforms = (
            latent_compiled_kernels is not None and latent_X_train_conts is not None
        )
        if latent_compiled_kernels is None:
            latent_compiled_kernels = self.kernels

        if self._has_mixed_latents and (
            latent_C_trains is None
            or latent_analyses is None
            or latent_cat_specs is None
        ):
            raise RuntimeError(
                "Mixed LMC pathwise sampling requires per-latent categorical state. "
                "Re-fit or reload the model before sampling."
            )

        for latent_idx in range(R):
            kernel_s = latent_compiled_kernels[latent_idx]

            if use_latent_input_transforms:
                X_train_s = latent_X_train_conts[latent_idx]
                X_test_s, _ = self._transform_inputs_for_latent(latent_idx, X_test)
                C_train_s = (
                    latent_C_trains[latent_idx]
                    if latent_C_trains is not None
                    else np.zeros((X_train_s.shape[0], 0), dtype=np.int32)
                )
                C_test_s = (
                    self._transform_inputs_for_latent(latent_idx, X_test)[1]
                    if self._has_mixed_latents
                    else np.zeros((X_test_s.shape[0], 0), dtype=np.int32)
                )
            else:
                X_train_s = np.ascontiguousarray(self._X_train, dtype=np.float32)
                X_test_s = X_test
                C_train_s = np.zeros((X_train_s.shape[0], 0), dtype=np.int32)
                C_test_s = np.zeros((X_test_s.shape[0], 0), dtype=np.int32)

            latent_x_test_conts.append(np.ascontiguousarray(X_test_s, dtype=np.float32))
            latent_c_tests.append(np.ascontiguousarray(C_test_s, dtype=np.int32))
            latent_train_conts.append(np.ascontiguousarray(X_train_s, dtype=np.float32))
            latent_train_cats.append(np.ascontiguousarray(C_train_s, dtype=np.int32))
            params_s = np.ascontiguousarray(
                result.params_per_latent[latent_idx], dtype=np.float32
            )
            A_s = np.asarray(result.A_matrices[latent_idx], dtype=np.float64)
            jitter = float(np.abs(np.diag(A_s)).mean()) * 1e-6 + 1e-8
            factor_s = np.linalg.cholesky(A_s + jitter * np.eye(T, dtype=np.float64))
            latent_mix_factors.append(factor_s)

            cat_params_s = (
                None
                if result.cat_params_per_latent is None
                or latent_idx >= len(result.cat_params_per_latent)
                else np.ascontiguousarray(
                    result.cat_params_per_latent[latent_idx], dtype=np.float32
                )
            )
            analysis_s = (
                None if latent_analyses is None else latent_analyses[latent_idx]
            )
            cat_col_map = (
                {}
                if analysis_s is None
                else {
                    spec.col_index: idx
                    for idx, spec in enumerate(analysis_s.categorical_specs)
                }
            )
            latent_feature_maps.append(
                build_pathwise_feature_map(
                    kernel_s,
                    params_s,
                    input_dim=X_train_s.shape[1],
                    n_features=n_rff_features,
                    rng=rng,
                    cat_params=cat_params_s,
                    cat_col_map=cat_col_map,
                )
            )

        fixed_noise_train = self._result.fixed_observation_noise
        if fixed_noise_train is not None:
            train_noise_scale = np.sqrt(
                np.maximum(
                    noise_per_task[np.newaxis, :]
                    + np.ascontiguousarray(fixed_noise_train, dtype=np.float32),
                    0.0,
                )
            ).astype(np.float32)
        else:
            train_noise_scale = np.sqrt(np.maximum(noise_per_task, 0.0)).astype(
                np.float32
            )[np.newaxis, :]

        solve_max_cg_iter = max(int(self.max_cg_iter), 100)
        solve_cg_tol = (
            min(float(self.cg_tol), 1e-2) if float(self.cg_tol) > 0.0 else 1e-2
        )
        samples = np.empty((n_samples, X_test.shape[0], T), dtype=np.float32)
        if self._has_mixed_latents:
            params_list = [
                np.ascontiguousarray(result.params_per_latent[s], dtype=np.float32)
                for s in range(R)
            ]
        else:
            params_list = [
                np.ascontiguousarray(
                    self.kernels[s].to_engine_params(
                        np.ascontiguousarray(
                            result.params_per_latent[s], dtype=np.float32
                        )
                    ),
                    dtype=np.float32,
                )
                for s in range(R)
            ]
        cat_params_list = [
            np.ascontiguousarray(cp, dtype=np.float32)
            for cp in (
                result.cat_params_per_latent
                if result.cat_params_per_latent is not None
                else [np.zeros(0, dtype=np.float32) for _ in range(R)]
            )
        ]
        try:
            for sample_idx in range(n_samples):
                latent_prior_train = np.zeros(
                    (self._X_train.shape[0], T), dtype=np.float32
                )
                latent_prior_test = np.zeros((X_test.shape[0], T), dtype=np.float32)
                for latent_idx in range(R):
                    feature_map_s = latent_feature_maps[latent_idx]
                    weights_s = build_feature_weights(feature_map_s, T, rng)
                    scalar_train = sample_prior_values(
                        feature_map_s,
                        latent_train_conts[latent_idx],
                        latent_train_cats[latent_idx],
                        weights_s,
                    )
                    scalar_test = sample_prior_values(
                        feature_map_s,
                        latent_x_test_conts[latent_idx],
                        latent_c_tests[latent_idx],
                        weights_s,
                    )
                    latent_prior_train += np.einsum(
                        "ln,tl->nt",
                        scalar_train.astype(np.float64),
                        latent_mix_factors[latent_idx],
                    ).astype(np.float32)
                    latent_prior_test += np.einsum(
                        "ln,tl->nt",
                        scalar_test.astype(np.float64),
                        latent_mix_factors[latent_idx],
                    ).astype(np.float32)

                obs_prior_train = (
                    latent_prior_train
                    + rng.standard_normal((self._X_train.shape[0], T)).astype(
                        np.float32
                    )
                    * train_noise_scale
                )
                residual = np.ascontiguousarray(
                    y_centered - obs_prior_train,
                    dtype=np.float32,
                )
                use_mixed_correction_backend = (
                    self._has_mixed_latents or self._uses_per_latent_runtime_inputs()
                )
                if use_mixed_correction_backend:
                    raw = self._engine.sample_lmc_mixed_pathwise(
                        provider_infos,
                        residual,
                        A_matrices,
                        latent_x_test_conts,
                        params_list,
                        list(self._latent_is_mixed),
                        self._latent_C_trains,
                        latent_c_tests,
                        cat_params_list,
                        self._latent_cat_specs,
                        noise_per_task,
                        solve_max_cg_iter,
                        solve_cg_tol,
                        1 if selected_runtime_method == "materialized" else 0,
                    )
                else:
                    raw = self._engine.sample_lmc_pathwise(
                        provider_infos,
                        residual,
                        A_matrices,
                        X_test,
                        params_list,
                        noise_per_task,
                        solve_max_cg_iter,
                        solve_cg_tol,
                        np.ascontiguousarray(
                            fixed_noise_train, dtype=np.float32
                        )
                        if fixed_noise_train is not None
                        else np.zeros((0,), dtype=np.float32),
                    )
                correction = np.asarray(raw["correction"], dtype=np.float32)
                samples[sample_idx] = (
                    mean_per_task[np.newaxis, :] + latent_prior_test + correction
                )

            self._pathwise_bundle_role = selected_bundle_role
            return samples.astype(np.float32)
        finally:
            if built_provider_infos is not None:
                destroy_provider_infos(provider_kernel_modules, built_provider_infos)

    def score(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute prediction metrics on test data.

        Args:
            X_test: Test inputs [m, d], float32
            Y_test: Test targets [m, T], float32

        Returns:
            Dictionary with rmse, mae, r2, rmse_per_task
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before scoring. Call fit() first.")

        Y_test = np.ascontiguousarray(Y_test, dtype=np.float32)
        pred = self.predict(X_test)
        mean = pred.mean if isinstance(pred, MultiOutputPredictionResult) else pred

        residuals = Y_test - mean
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-10))

        rmse_per_task = np.sqrt(np.mean(residuals**2, axis=0))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "rmse_per_task": rmse_per_task,
        }

    def save(self, path: str) -> None:
        """Save the trained LMC multi-output GP model to disk.

        Args:
            path: File path (without extension). Creates {path}_config.json
                  and {path}_arrays.npz

        Raises:
            RuntimeError: If model is not trained
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before saving. Call fit() first.")

        path = str(path)
        if path.endswith(".json") or path.endswith(".npz"):
            path = path.rsplit(".", 1)[0]

        tr = self._result

        config = {
            "schema_version": _MODEL_SCHEMA_VERSION,
            "mojogp_version": __version__,
            "wrapper": "MultiOutputLMCGP",
            "is_composite": self._is_composite,
            "has_mixed_latents": self._has_mixed_latents,
            "ard": self.ard,
            "method": self.method,
            "num_probes": self.num_probes,
            "max_cg_iter": self.max_cg_iter,
            "cg_tol": self.cg_tol,
            "use_preconditioner": self.use_preconditioner,
            "precond_rank": self.precond_rank,
            "precond": self.precond,
            "precond_rebuild_threshold": self.precond_rebuild_threshold,
            "num_latents": tr.num_latents,
            "num_tasks": tr.num_tasks,
            "final_nll": float(tr.final_nll),
            "iterations": int(tr.iterations),
            "converged": bool(tr.converged),
            "use_ard": bool(tr.use_ard),
            "has_fixed_observation_noise": self._fixed_observation_noise is not None,
        }

        saved_kernels = getattr(self, "_original_kernels", self.kernels)
        if self._is_composite:
            config["kernel_trees"] = [k.to_dict() for k in saved_kernels]
            config["kernel_mojo_type"] = saved_kernels[0].to_mojo_type()
        else:
            config["kernels"] = saved_kernels

        with open(f"{path}_config.json", "w") as f:
            json.dump(config, f, indent=2)

        arrays = {
            "X_train": self._X_train,
            "Y_train": self._Y_train,
            "lengthscales": tr.lengthscales,
            "outputscales": tr.outputscales,
            "kernel_types": np.asarray(
                tr.kernel_types if tr.kernel_types is not None else [], dtype=np.int32
            ),
            "noise_per_task": tr.noise_per_task,
            "A_matrices": tr.A_matrices,
            "L_factors": tr.L_factors,
            "B": tr.B,
            "Q": tr.Q,
            "Lambda": tr.Lambda,
            "alpha": tr.alpha,
            "alpha_rotated": tr.alpha_rotated,
            "effective_scales": tr.effective_scales,
            "nll_history": tr.nll_history,
        }
        if self._fixed_observation_noise is not None:
            arrays["fixed_observation_noise"] = self._fixed_observation_noise
        arrays = {k: v for k, v in arrays.items() if v is not None}

        if tr.lengthscales_per_dim is not None:
            arrays["lengthscales_per_dim"] = tr.lengthscales_per_dim
        if self._learned_composite_params is not None:
            arrays["learned_composite_params"] = self._learned_composite_params
        # Save per-latent params for heterogeneous LMC prediction
        if tr.params_per_latent is not None:
            for s, p in enumerate(tr.params_per_latent):
                arrays[f"params_latent_{s}"] = np.array(p, dtype=np.float32)
        if tr.cat_params_per_latent is not None:
            for s, p in enumerate(tr.cat_params_per_latent):
                arrays[f"cat_params_latent_{s}"] = np.array(p, dtype=np.float32)

        # ConstantMean: save mean_per_task and fitted_mean
        if tr.mean_per_task is not None:
            arrays["mean_per_task"] = tr.mean_per_task
        if self._fitted_mean is not None:
            arrays["fitted_mean"] = self._fitted_mean

        # var_diag: save diagonal variance for task covariance
        if tr.var_diag is not None:
            arrays["var_diag"] = tr.var_diag
        if tr.iter_times_ms is not None:
            arrays["iter_times_ms"] = tr.iter_times_ms
        if tr.fixed_observation_noise is not None:
            arrays["fixed_observation_noise"] = tr.fixed_observation_noise

        np.savez(f"{path}_arrays.npz", **arrays)
        if self._has_mixed_latents and self.method == "materialized":
            _LMC_SAVED_RUNTIME_OWNERS[str(Path(path).resolve())] = weakref.ref(self)

    @classmethod
    def load(
        cls,
        path: str,
        kernels: Optional[list] = None,
    ) -> "MultiOutputLMCGP":
        """Load a saved LMC multi-output GP model from disk.

        Args:
            path: File path (without extension)
            kernels: The kernels used when saving. Required for composite kernels.
                For string kernels, this is auto-restored from config.

        Returns:
            Loaded MultiOutputLMCGP with training state restored
        """
        path = str(path)
        if path.endswith(".json") or path.endswith(".npz"):
            path = path.rsplit(".", 1)[0]

        with open(f"{path}_config.json", "r") as f:
            config = json.load(f)

        wrapper = config.get("wrapper")
        if wrapper != "MultiOutputLMCGP":
            raise ValueError(
                "Saved model is not a MultiOutputLMCGP artifact. "
                f"Expected wrapper='MultiOutputLMCGP', got {wrapper!r}."
            )
        schema_version = int(config.get("schema_version", 0))
        if schema_version != _MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported MultiOutputLMCGP schema_version={schema_version}; "
                f"expected {_MODEL_SCHEMA_VERSION}."
            )

        arrays = np.load(f"{path}_arrays.npz", allow_pickle=False)
        _require_lmc_saved_arrays(
            arrays,
            (
                "X_train",
                "Y_train",
                "lengthscales",
                "outputscales",
                "kernel_types",
                "noise_per_task",
                "A_matrices",
                "B",
                "Q",
                "Lambda",
                "nll_history",
            ),
        )

        required_config_keys = [
            "num_latents",
            "num_tasks",
            "final_nll",
            "iterations",
            "converged",
        ]
        missing_config = [key for key in required_config_keys if key not in config]
        if missing_config:
            raise ValueError(
                "MultiOutputLMCGP artifact is missing required config fields: "
                + ", ".join(missing_config)
            )

        required_array_keys = [
            "X_train",
            "Y_train",
            "lengthscales",
            "outputscales",
            "kernel_types",
            "noise_per_task",
            "A_matrices",
            "B",
            "Q",
            "Lambda",
            "alpha",
            "nll_history",
        ]
        missing_arrays = [key for key in required_array_keys if key not in arrays]
        if missing_arrays:
            raise ValueError(
                "MultiOutputLMCGP artifact is missing required arrays: "
                + ", ".join(missing_arrays)
            )

        if "kernel_trees" in config and config["kernel_trees"]:
            if len(config["kernel_trees"]) != int(config["num_latents"]):
                raise ValueError(
                    "MultiOutputLMCGP artifact kernel_trees length must match "
                    f"num_latents={config['num_latents']}."
                )

        # Determine kernels
        if kernels is None:
            if config.get("kernel_trees"):
                kernels = [KernelNode.from_dict(kt) for kt in config["kernel_trees"]]
            elif config.get("is_composite"):
                raise ValueError(
                    "MultiOutputLMCGP artifact is missing required kernel_trees."
                )
            else:
                kernels = config["kernels"]

        gp = cls(
            kernels=kernels,
            num_probes=config["num_probes"],
            max_cg_iterations=config["max_cg_iter"],
            cg_tolerance=config["cg_tol"],
            use_preconditioner=config["use_preconditioner"],
            preconditioner_rank=config["precond_rank"],
            preconditioner=config["precond"],
            precond_rebuild_threshold=config["precond_rebuild_threshold"],
            ard=config["ard"],
        )
        gp.method = config.get("method", "materialized")

        gp._X_train = arrays["X_train"]
        gp._Y_train = arrays["Y_train"]
        gp._fixed_observation_noise = (
            np.ascontiguousarray(arrays["fixed_observation_noise"], dtype=np.float32)
            if "fixed_observation_noise" in arrays
            else None
        )
        if bool(config.get("has_fixed_observation_noise", False)) and gp._fixed_observation_noise is None:
            raise ValueError(
                "MultiOutputLMCGP artifact declares fixed observation noise but is "
                "missing fixed_observation_noise array."
            )
        gp._is_trained = True

        has_mixed_latents = bool(config.get("has_mixed_latents", False)) or any(
            isinstance(k, KernelNode) and k.has_categorical() for k in gp.kernels
        )
        gp._configure_latent_kernels_for_fit(gp._X_train.shape[1])
        if gp._latent_compiled_kernels is not None:
            gp.kernels = [copy.deepcopy(k) for k in gp._latent_compiled_kernels]
            gp._composite_kernel = gp.kernels[0] if gp.kernels else None
        gp._latent_X_train_conts = []
        gp._latent_C_trains = []
        for latent_idx in range(gp.num_latents):
            X_cont_s, C_s = gp._transform_inputs_for_latent(latent_idx, gp._X_train)
            gp._latent_X_train_conts.append(X_cont_s)
            gp._latent_C_trains.append(C_s)

        if has_mixed_latents:
            gp._kernel_modules = None
            gp._kernel_module = None
            gp._engine = None
        else:
            from .loader import load_kernel_module_engine, load_engine

            module_cache: Dict[Tuple[str, int], Any] = {}
            gp._kernel_modules = []
            assert gp._latent_compiled_kernels is not None
            for latent_idx, kernel in enumerate(gp._latent_compiled_kernels):
                x_train_cont_s = gp._latent_X_train_conts[latent_idx]
                module_dim = (
                    int(x_train_cont_s.shape[1])
                    if has_mixed_latents
                    else int(gp._X_train.shape[1])
                )
                cache_key = (kernel.to_mojo_type(), module_dim)
                if cache_key not in module_cache:
                    module_cache[cache_key] = load_kernel_module_engine(
                        kernel,
                        dim=module_dim,
                        fresh_load=True,
                        isolated_load_id=gp._isolated_load_id,
                        ncols_hint=_lmc_provider_ncols_hint(
                            int(config["num_tasks"])
                        ),
                        verbose=False,
                    )
                gp._kernel_modules.append(module_cache[cache_key])
            gp._kernel_module = gp._kernel_modules[0] if gp._kernel_modules else None
            gp._engine = load_engine(
                fresh_load=True,
                isolated_load_id=gp._isolated_load_id,
                verbose=False,
            )

        lengthscales_per_dim = (
            arrays["lengthscales_per_dim"] if "lengthscales_per_dim" in arrays else None
        )

        # ConstantMean: load optional mean_per_task from current artifacts.
        _mean_per_task = arrays["mean_per_task"] if "mean_per_task" in arrays else None
        _var_diag = arrays["var_diag"] if "var_diag" in arrays else None

        # Restore per-latent params for heterogeneous LMC prediction
        R_load = config.get("num_latents", 1)
        _params_per_latent = []
        for s in range(R_load):
            key = f"params_latent_{s}"
            if key in arrays:
                _params_per_latent.append(arrays[key].tolist())
        if not _params_per_latent:
            _params_per_latent = None

        _cat_params_per_latent = []
        for s in range(R_load):
            key = f"cat_params_latent_{s}"
            if key in arrays:
                _cat_params_per_latent.append(arrays[key].tolist())
        if not _cat_params_per_latent:
            _cat_params_per_latent = None

        gp._result = LMCTrainingResult(
            lengthscales=arrays["lengthscales"],
            outputscales=arrays["outputscales"],
            kernel_types=arrays["kernel_types"],
            noise_per_task=arrays["noise_per_task"],
            A_matrices=arrays["A_matrices"],
            L_factors=arrays["L_factors"] if "L_factors" in arrays else None,
            B=arrays["B"],
            Q=arrays["Q"],
            Lambda=arrays["Lambda"],
            alpha=arrays["alpha"] if "alpha" in arrays else None,
            alpha_rotated=arrays["alpha_rotated"] if "alpha_rotated" in arrays else None,
            effective_scales=arrays["effective_scales"] if "effective_scales" in arrays else None,
            final_nll=config["final_nll"],
            nll_history=arrays["nll_history"],
            iterations=config["iterations"],
            converged=config["converged"],
            num_latents=config["num_latents"],
            num_tasks=config["num_tasks"],
            use_ard=config.get("use_ard", False),
            lengthscales_per_dim=lengthscales_per_dim,
            mean_per_task=_mean_per_task,
            var_diag=_var_diag,
            params_per_latent=_params_per_latent,
            cat_params_per_latent=_cat_params_per_latent,
            cat_param_names_per_latent=(
                gp._latent_cat_param_names if has_mixed_latents else None
            ),
            iter_times_ms=(
                arrays["iter_times_ms"] if "iter_times_ms" in arrays else None
            ),
            fixed_observation_noise=(
                arrays["fixed_observation_noise"]
                if "fixed_observation_noise" in arrays
                else None
            ),
        )

        if "learned_composite_params" in arrays:
            gp._learned_composite_params = arrays["learned_composite_params"]

        # ConstantMean: restore fitted mean
        if "fitted_mean" in arrays:
            gp._fitted_mean = arrays["fitted_mean"]
        elif _mean_per_task is not None:
            gp._fitted_mean = _mean_per_task

        gp._provider_infos = None
        gp._training_bundle = None
        gp._inference_bundle = None
        gp._backend_train_info = None
        gp._backend_predict_info = None
        gp._backend_sample_info = None
        gp._pathwise_bundle_role = None
        gp._clear_borrowed_runtime()
        gp._try_borrow_saved_runtime(path)

        return gp

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        tasks = f", tasks={self.num_tasks}" if self.num_tasks else ""
        ard_str = ", ard=True" if self.ard else ""
        if self._is_composite:
            unique_types = set(k.to_mojo_type() for k in self.kernels)
            if len(unique_types) == 1:
                kernels_str = str(self.kernels[0])
            else:
                kernels_str = "[" + ", ".join(str(k) for k in self.kernels) + "]"
            composite_str = ", composite=True"
        else:
            kernels_str = "+".join(self.kernels)
            composite_str = ""
        return f"MultiOutputLMCGP(kernels={kernels_str}, R={self.num_latents}{tasks}{ard_str}{composite_str}, {status})"
