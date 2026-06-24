"""High-level Gaussian Process API for MojoGP.

This module provides a user-friendly API for training and prediction with
arbitrary kernel compositions, similar to GPyTorch's ExactGP.

All kernels are JIT-compiled via the fn-ptr codegen engine, which produces
a lightweight kernel .so (~5s compilation) and dispatches training/prediction
through the pre-compiled JIT engine.

Example usage:
    from mojogp import SingleOutputGP as SoGP, RBF, Matern52

    # Create GP
    gp = SoGP(RBF(ard=True))

    # Train
    gp.fit(X_train, y_train, max_iterations=100)

    # Predict
    pred = gp.predict(X_test)
    mean, std = pred.mean, pred.std
"""

import json
import uuid
import warnings
import numpy as np
from typing import Optional, Tuple, Any, Union, Dict, Callable
from dataclasses import dataclass
from pathlib import Path

from .kernel import (
    KernelNode,
    KernelType,
    Kernel,
    build_default_categorical_raw_params,
    categorical_prediction_params,
    continuous_kernel_tree,
    make_ard_kernel,
)
from ._provider_lifecycle import (
    register_provider_lease,
    revoke_conflicting_provider_lease,
    revoke_conflicting_provider_leases_by_name,
    revoke_orphan_provider_leases,
    unregister_provider_lease,
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
    surface_for_single_output,
    warn_surface_status,
)
from .specialization import (
    SpecializationDecision,
    SpecializationRequest,
    build_single_output_descriptor,
    default_specialization_registry,
    translate_compile_inputs,
)
from ._routes import normalize_fit_method
from ._version import __version__
from .progress import prediction_progress_stats, resolve_progress_adapter


_MODEL_SCHEMA_VERSION = 1


_DEFAULT_PREDICT_LANCZOS_RANK = 100
_DEFAULT_ARD_PREDICT_LANCZOS_RANK = 200
_DEFAULT_MIXED_MATRIX_FREE_PREDICT_LANCZOS_RANK = 200
_DEFAULT_MIXED_MATERIALIZED_PREDICT_LANCZOS_RANK = 200
_DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK = 300
_DEFAULT_PATHWISE_TEST_CHUNK_SIZE = 512
_DEFAULT_EXACT_PREDICT_MAX_CG_ITER = 300
_DEFAULT_EXACT_PREDICT_CG_TOL = 1e-3
# Prediction means must be invariant to the requested variance route. Use a
# shared alpha solve budget for exact, LOVE, and mean-only predictions unless the
# caller explicitly overrides it.
_DEFAULT_PREDICT_MAX_CG_ITER = _DEFAULT_EXACT_PREDICT_MAX_CG_ITER
_DEFAULT_PREDICT_CG_TOL = _DEFAULT_EXACT_PREDICT_CG_TOL
_DEFAULT_PREDICT_NCOLS_HINT = [32, 16, 11, 6, 1]


def _validate_progress_interval(progress_interval: int) -> int:
    value = int(progress_interval)
    if value <= 0:
        raise ValueError("progress_interval must be a positive integer")
    return value


def _sync_cuda_runtime() -> None:
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prediction_lanczos_rank(
    rank_hint: Optional[int],
    training_method: Optional[str],
    is_mixed: bool,
    is_ard: bool,
) -> int:
    """Choose a stable prediction rank for LOVE.

    GPyTorch defaults root decompositions to rank 100. Mixed matrix-free models
    need a higher floor to keep the JIT LOVE variance informative, and single-
    output ARD models are materially more rank-hungry than isotropic models.
    """
    rank = int(rank_hint) if rank_hint is not None else 0
    min_rank = _DEFAULT_PREDICT_LANCZOS_RANK
    is_materialized = training_method == "materialized"
    if is_mixed:
        if is_materialized:
            min_rank = _DEFAULT_MIXED_MATERIALIZED_PREDICT_LANCZOS_RANK
        else:
            min_rank = _DEFAULT_MIXED_MATRIX_FREE_PREDICT_LANCZOS_RANK
    if is_ard:
        min_rank = max(min_rank, _DEFAULT_ARD_PREDICT_LANCZOS_RANK)
        if is_mixed and is_materialized:
            min_rank = max(
                min_rank, _DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK
            )
    return max(rank, min_rank)


def _prediction_solver_config(
    variance_method: str,
    train_info: Optional[dict[str, Any]],
    rank_hint: Optional[int],
    training_method: Optional[str],
    *,
    is_mixed: bool,
    is_ard: bool,
    max_cg_iterations: Optional[int] = None,
    cg_tolerance: Optional[float] = None,
    precond_rank: Optional[int] = None,
    max_root_decomposition_size: Optional[int] = None,
) -> dict[str, float | int]:
    train_info = train_info or {}
    resolved_precond_rank = int(
        precond_rank
        if precond_rank is not None
        else train_info.get("precond_rank", 10)
    )
    predict_rank = _prediction_lanczos_rank(
        rank_hint,
        training_method,
        is_mixed=is_mixed,
        is_ard=is_ard,
    )
    if max_root_decomposition_size is not None:
        predict_rank = int(max_root_decomposition_size)
    train_predict_max_cg_iter = int(
        train_info.get(
            "max_cg_iterations",
            train_info.get("max_cg_iter", _DEFAULT_PREDICT_MAX_CG_ITER),
        )
    )
    train_predict_cg_tol = float(
        train_info.get(
            "cg_tolerance",
            train_info.get("cg_tol", _DEFAULT_PREDICT_CG_TOL),
        )
    )
    default_predict_max_cg_iter = max(
        train_predict_max_cg_iter,
        _DEFAULT_EXACT_PREDICT_MAX_CG_ITER,
    )
    default_predict_cg_tol = min(
        train_predict_cg_tol,
        _DEFAULT_EXACT_PREDICT_CG_TOL,
    )
    return {
        "max_cg_iterations": int(
            max_cg_iterations
            if max_cg_iterations is not None
            else default_predict_max_cg_iter
        ),
        "cg_tolerance": float(
            cg_tolerance if cg_tolerance is not None else default_predict_cg_tol
        ),
        "precond_rank": resolved_precond_rank,
        "max_root_decomposition_size": predict_rank,
    }


@dataclass
class TrainingResult:
    """Result from GP training."""

    params: np.ndarray  # Optimized kernel parameters
    noise: float  # Optimized noise variance
    mean: float  # Learned constant mean function value
    nll: float  # Final negative log-likelihood
    iterations: int  # Number of training iterations
    converged: bool  # Whether training converged
    lanczos_root: np.ndarray  # Lanczos root for LOVE variance
    lanczos_rank: int  # Rank of Lanczos approximation
    nll_history: Optional[np.ndarray] = None  # NLL per training iteration
    cg_iterations_history: Optional[np.ndarray] = None  # Realized CG iterations
    iter_times_ms: Optional[np.ndarray] = None  # Direct per-iteration wall times


@dataclass
class MixedTrainingResult:
    """Result from mixed continuous + categorical GP training."""

    params: np.ndarray  # Optimized continuous kernel parameters
    cat_params: np.ndarray  # Optimized categorical kernel parameters
    noise: float  # Optimized noise variance
    mean: float  # Learned constant mean function value
    nll: float  # Final negative log-likelihood
    iterations: int  # Number of training iterations
    converged: bool  # Whether training converged
    alpha: np.ndarray  # K^{-1} @ y for prediction
    nll_history: Optional[np.ndarray] = None  # NLL per training iteration
    cg_iterations_history: Optional[np.ndarray] = None  # Realized CG iterations
    iter_times_ms: Optional[np.ndarray] = None  # Direct per-iteration wall times


@dataclass
class PredictionResult:
    """Result from GP prediction."""

    mean: np.ndarray  # Predictive mean
    variance: np.ndarray  # Predictive variance
    std: np.ndarray  # Predictive standard deviation


class SingleOutputGP:
    """Exact Gaussian Process with arbitrary kernel compositions.

    All kernels (single and composite) are JIT-compiled via the fn-ptr
    codegen engine for GPU-accelerated training and prediction.

    Example:
        >>> from mojogp import SingleOutputGP as SoGP, RBF, Matern52
        >>>
        >>> # Simple GP
        >>> gp = SoGP(RBF())
        >>> gp.fit(X_train, y_train)
        >>> mean, std = gp.predict(X_test, return_std=True)
        >>>
        >>> # ARD with named initial values
        >>> gp = SoGP(RBF(ard=True, lengthscale=0.5))
        >>>
        >>> # Composite kernel
        >>> gp = SoGP(RBF() + Matern52())
    """

    def __init__(
        self,
        kernel: KernelNode,
        *,
        init_mean: Optional[float] = None,
        verbose: bool = False,
    ):
        """Initialize the GP with kernel specification.

        Args:
            kernel: Kernel composition (e.g., RBF(ard=True) + Matern52())
            init_mean: Initial constant mean value. If None (default), auto-detected
                       from np.mean(y) during fit().
            verbose: Print progress during compilation and training
        """
        # Validate kernel type
        if not isinstance(kernel, KernelNode):
            raise TypeError(
                f"kernel must be a KernelNode instance (e.g. RBF(), Matern52()), "
                f"got {type(kernel).__name__}"
            )

        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._C_train: Optional[np.ndarray] = None
        self._cat_specs: Optional[list] = None
        self.dim: Optional[int] = None
        self.verbose = verbose
        self.cat_dims = {}  # Populated from kernel tree analysis during fit()
        self.cat_kernel = (
            "ehh"  # Default, overridden from kernel tree analysis during fit()
        )
        self._dim_permutation: Optional[list[int]] = None
        self._init_mean = init_mean
        self._fitted_mean: Optional[float] = None

        # Store original kernel for serialization (before ARD transform)
        self._original_kernel = kernel
        # ARD is resolved once we know dim (in fit())
        self._kernel_pre_ard = kernel
        self.ard = kernel.has_ard()
        self.kernel = kernel  # will be replaced in fit() if ARD

        # Training state
        self._is_trained = False
        self._training_result: Optional[Union[TrainingResult, MixedTrainingResult]] = (
            None
        )
        self._cached_alpha: Optional[np.ndarray] = (
            None  # K^{-1} @ y for fast prediction
        )
        self._cached_alpha_info: Optional[dict[str, Any]] = None
        self._cached_love_method: Optional[str] = None

        # Compiled module (lazy initialization — used by mixed kernel path)
        # Engine path state
        self._engine = None  # mojogp_jit_engine module
        self._kernel_module = None  # fn-ptr kernel .so module
        self._provider_info = None  # fn-ptr dict from init_provider()
        self._provider_state_current = False
        self._isolated_load_id = uuid.uuid4().hex
        self._training_method: Optional[str] = None
        self._specialization_request = SpecializationRequest.disabled()
        self._specialization_decision: Optional[SpecializationDecision] = None
        self._backend_train_info: Optional[dict[str, Any]] = None
        self._backend_predict_info: Optional[dict[str, Any]] = None
        self._backend_sample_info: Optional[dict[str, Any]] = None
        self._engine_predict_mean: Optional[float] = None
        self._prediction_cache_handle: Optional[int] = None
        self._prediction_cache_signature: Optional[tuple[Any, ...]] = None
        self._prediction_cache_info: Optional[dict[str, Any]] = None
        self._noise_mode: str = "scalar"
        self._observation_noise_train: Optional[np.ndarray] = None
        self._noise_group_train: Optional[np.ndarray] = None
        self._noise_group_values: Optional[np.ndarray] = None
        self._observation_noise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._noise_function: Optional[str] = None
        self._noise_function_params: Optional[np.ndarray] = None
        self._noise_floor: float = 1e-6
        self._noise_regularization: float = 0.0
        self._provider_noise_mode_int: Optional[int] = None

    def _set_specialization_request(
        self,
        request: SpecializationRequest | dict[str, Any] | None,
    ) -> None:
        """Set an internal specialization request for benchmark experiments."""
        if isinstance(request, SpecializationRequest):
            self._specialization_request = request
            return
        self._specialization_request = SpecializationRequest.from_dict(request)

    def _resolve_specialization_decision(
        self,
        kernel_node: KernelNode,
        dim: int,
    ) -> SpecializationDecision:
        descriptor = build_single_output_descriptor(
            kernel=kernel_node,
            dim=dim,
            training_method=getattr(self, "_training_method", None),
            n_train=(None if self._X_train is None else int(self._X_train.shape[0])),
        )
        registry = default_specialization_registry(
            materialized_predict_ncols_hint=tuple(_DEFAULT_PREDICT_NCOLS_HINT)
        )
        self._specialization_decision = registry.resolve(
            descriptor,
            self._specialization_request,
        )
        return self._specialization_decision

    def _maybe_attach_specialization_metadata(
        self,
        info: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
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

    def _destroy_prediction_cache(self):
        handle = getattr(self, "_prediction_cache_handle", None)
        if handle:
            engine = getattr(self, "_engine", None)
            destroy = getattr(engine, "destroy_prediction_cache", None)
            if destroy is not None:
                try:
                    destroy(int(handle))
                except Exception:
                    pass
        self._prediction_cache_handle = None
        self._prediction_cache_signature = None
        self._prediction_cache_info = None

    def _invalidate_prediction_caches(self):
        self._destroy_prediction_cache()
        self._cached_alpha = None
        self._cached_alpha_info = None
        self._cached_love_method = None
        tr = self._training_result
        if isinstance(tr, TrainingResult):
            tr.lanczos_root = None

    def _destroy_provider_info(self):
        """Release any live provider owned by the kernel module."""
        provider_info = getattr(self, "_provider_info", None)
        kernel_module = getattr(self, "_kernel_module", None)
        if kernel_module is not None:
            unregister_provider_lease(kernel_module, self)
        if provider_info is None or kernel_module is None:
            self._provider_state_current = False
            return
        destroy = getattr(kernel_module, "destroy_provider", None)
        if destroy is None:
            self._provider_state_current = False
            return
        try:
            _sync_cuda_runtime()
            destroy(provider_info)
            _sync_cuda_runtime()
        except Exception:
            pass
        self._provider_info = None
        self._provider_state_current = False

    def _revoke_provider_info(self):
        """Release a provider lease and force the next use to reload modules."""
        self._destroy_prediction_cache()
        self._destroy_provider_info()
        self._kernel_module = None
        self._engine = None

    def _build_provider_info(
        self,
        X,
        engine_params,
        noise,
        method: Optional[str] = None,
        observation_noise: Optional[np.ndarray] = None,
        noise_mode_int: Optional[int] = None,
        noise_group_train: Optional[np.ndarray] = None,
        num_noise_groups: Optional[int] = None,
    ):
        """Create a provider for one fit/predict/sample call."""
        if self._kernel_module is None or self._engine is None:
            self._ensure_compiled()
        import gc

        gc.collect()
        _sync_cuda_runtime()
        if revoke_orphan_provider_leases():
            _sync_cuda_runtime()
        revoke_conflicting_provider_lease(self._kernel_module, self)
        provider_args = [
            np.ascontiguousarray(X, dtype=np.float32),
            np.ascontiguousarray(engine_params, dtype=np.float32),
            float(noise),
        ]
        if observation_noise is not None:
            provider_args.append(np.ascontiguousarray(observation_noise, dtype=np.float32))
            if noise_mode_int is not None:
                provider_args.append(int(noise_mode_int))
                if noise_group_train is not None:
                    provider_args.append(np.ascontiguousarray(noise_group_train, dtype=np.int32))
                    provider_args.append(int(num_noise_groups or 0))
        provider_info = self._kernel_module.init_provider(*provider_args)
        resolved_method = method or getattr(self, "_training_method", None)
        if resolved_method == "materialized":
            self._kernel_module.materialize(provider_info)
        return provider_info

    def _destroy_temporary_provider_info(self, provider_info: dict[str, Any]) -> None:
        """Release a short-lived prediction provider after queued GPU work settles."""
        kernel_module = getattr(self, "_kernel_module", None)
        destroy = getattr(kernel_module, "destroy_provider", None) if kernel_module is not None else None
        if destroy is None:
            return
        try:
            _sync_cuda_runtime()
            destroy(provider_info)
            _sync_cuda_runtime()
        except Exception:
            pass

    def __del__(self):
        try:
            self._destroy_prediction_cache()
            self._destroy_provider_info()
        except Exception:
            pass

    @property
    def _is_mixed(self) -> bool:
        """Whether this GP uses mixed continuous + categorical kernels."""
        return len(self.cat_dims) > 0

    def _to_kernel_node(self) -> KernelNode:
        """Convert kernel to KernelNode for JIT path.

        If kernel is already a KernelNode, return as-is.
        If kernel is a string (standard kernel name), convert to KernelNode.
        ARD is applied if self.ard is True and dim is known.
        """
        if isinstance(self.kernel, KernelNode):
            return self.kernel
        # Should not reach here for non-string kernels
        raise TypeError(f"Cannot convert kernel to KernelNode: {type(self.kernel)}")

    def _continuous_param_kernel(self) -> KernelNode:
        """Kernel tree that owns the public continuous parameter layout."""
        continuous_kernel = continuous_kernel_tree(self.kernel)
        if continuous_kernel is not None:
            return continuous_kernel
        return self.kernel

    def _split_data(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Split input data into continuous and categorical parts."""
        if not self._is_mixed:
            return X, None
        all_cols = list(range(X.shape[1]))
        cont_cols = [c for c in all_cols if c not in self._cat_col_indices]
        X_cont = X[:, cont_cols].astype(np.float32)
        C = X[:, self._cat_col_indices].astype(np.int32)
        return X_cont, C

    @staticmethod
    def _validate_active_dims_bounds(kernel, cont_dim: int):
        """Recursively check that all active_dims are within [0, cont_dim)."""
        if kernel.active_dims is not None and not kernel.is_categorical():
            for d in kernel.active_dims:
                if d < 0 or d >= cont_dim:
                    raise ValueError(
                        f"active_dims contains index {d} which is out of range "
                        f"for input with {cont_dim} continuous dimensions (0..{cont_dim - 1}). "
                        f"Kernel: {kernel}"
                    )
        if kernel.left is not None:
            SingleOutputGP._validate_active_dims_bounds(kernel.left, cont_dim)
        if kernel.right is not None:
            SingleOutputGP._validate_active_dims_bounds(kernel.right, cont_dim)

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
                left=SingleOutputGP._remap_kernel_active_dims(kernel.left, dim_map),
                right=SingleOutputGP._remap_kernel_active_dims(kernel.right, dim_map),
                active_dims=kernel.active_dims,
            )

        if kernel.operator == "product":
            return KernelNode(
                operator="product",
                left=SingleOutputGP._remap_kernel_active_dims(kernel.left, dim_map),
                right=SingleOutputGP._remap_kernel_active_dims(kernel.right, dim_map),
                active_dims=kernel.active_dims,
            )

        if kernel.operator == "scale":
            return KernelNode(
                operator="scale",
                left=SingleOutputGP._remap_kernel_active_dims(kernel.left, dim_map),
                scale_factor=kernel.scale_factor,
                initial_values=kernel.initial_values,
                active_dims=kernel.active_dims,
            )

        raise ValueError(f"Unknown operator: {kernel.operator}")

    def _apply_dim_permutation(self, X_cont: np.ndarray) -> np.ndarray:
        """Apply dimension permutation for active_dims routing.

        If a permutation was computed during fit() (because kernels have
        active_dims), reorder columns accordingly. Otherwise return as-is.
        """
        if self._dim_permutation is not None:
            return X_cont[:, self._dim_permutation].astype(np.float32)
        return X_cont

    def _ensure_compiled(self):
        """Ensure the kernel module is compiled and loaded.

        Continuous kernels use the JIT engine path (fn-ptr codegen).
        """
        if self._engine is not None:
            return

        if self._is_mixed:
            cont_kernel = self.kernel

            from .loader import load_kernel_module_engine, load_engine

            decision = self._resolve_specialization_decision(cont_kernel, self._cont_dim)

            self._kernel_module = load_kernel_module_engine(
                cont_kernel,
                self._cont_dim,
                fresh_load=True,
                isolated_load_id=self._isolated_load_id,
                verbose=self.verbose,
                specialization_decision=decision,
            )
            self._engine = load_engine(
                verbose=self.verbose,
                fresh_load=True,
                isolated_load_id=self._isolated_load_id,
            )
        else:
            from .loader import load_kernel_module_engine, load_engine
            from .codegen_engine.compiler import make_module_name

            kernel_node = self._to_kernel_node()
            decision = self._resolve_specialization_decision(kernel_node, self.dim)
            translation = translate_compile_inputs(decision)
            training_method = getattr(self, "_training_method", None)
            ncols_hint = (
                _DEFAULT_PREDICT_NCOLS_HINT
                if training_method == "materialized"
                else None
            )
            resolved_ncols_hint = translation.ncols_hint or ncols_hint
            module_suffix = translation.module_suffix
            if resolved_ncols_hint:
                ncols_suffix = "ncols_" + "_".join(
                    str(int(n)) for n in resolved_ncols_hint
                )
                module_suffix = (
                    ncols_suffix
                    if module_suffix in (None, "")
                    else f"{module_suffix}_{ncols_suffix}"
                )
            module_name = make_module_name(
                kernel_node,
                self.dim,
                "fn_ptr",
                module_suffix=module_suffix,
            )
            revoke_conflicting_provider_leases_by_name(
                module_name,
                owner=self,
                include_live_owners=False,
            )
            self._kernel_module = load_kernel_module_engine(
                kernel_node,
                self.dim,
                fresh_load=True,
                isolated_load_id=self._isolated_load_id,
                verbose=self.verbose,
                ncols_hint=ncols_hint,
                specialization_decision=decision,
            )
            self._engine = load_engine(
                verbose=self.verbose,
                fresh_load=True,
                isolated_load_id=self._isolated_load_id,
            )

    def _has_periodic_kernel(self) -> bool:
        """Check if the kernel tree contains a periodic kernel."""

        def _check(node: KernelNode) -> bool:
            if node.kernel_type == KernelType.PERIODIC:
                return True
            if node.left and _check(node.left):
                return True
            if node.right and _check(node.right):
                return True
            return False

        return _check(self.kernel)

    def _has_kernel_type(self, *kernel_types: KernelType) -> bool:
        """Check if the kernel tree contains any of the requested base types."""

        target_types = set(kernel_types)

        def _check(node: KernelNode) -> bool:
            if node.kernel_type in target_types:
                return True
            if node.left and _check(node.left):
                return True
            if node.right and _check(node.right):
                return True
            return False

        return _check(self.kernel)

    def _is_single_kernel_type(self, kernel_type: KernelType) -> bool:
        """Whether the resolved kernel tree is exactly one base kernel type."""
        return self.kernel.operator is None and self.kernel.kernel_type == kernel_type

    def _supports_materialized_ard_precond_defaults(self) -> bool:
        """Whether materialized ARD should use ARD-specific preconditioner defaults."""

        return any(
            self._is_single_kernel_type(kernel_type)
            for kernel_type in (
                KernelType.RBF,
                KernelType.MATERN12,
                KernelType.MATERN32,
                KernelType.MATERN52,
                KernelType.RQ,
            )
        )

    @property
    def num_params(self) -> int:
        """Number of kernel parameters."""
        return self.kernel.num_params()

    @property
    def param_names(self) -> list:
        """Names of kernel parameters."""
        return self.kernel.get_param_names()

    @property
    def is_trained(self) -> bool:
        """Whether the GP has been trained."""
        return self._is_trained

    @property
    def training_result(self) -> Optional[Union[TrainingResult, MixedTrainingResult]]:
        """Training result (None if not trained)."""
        return self._training_result

    @property
    def nll_history(self) -> Optional[np.ndarray]:
        """NLL per training iteration (None if not trained or not available)."""
        if self._training_result is None:
            return None
        return self._training_result.nll_history

    @property
    def backend_train_info(self) -> Optional[dict[str, Any]]:
        """Backend-reported metadata from the most recent training run."""
        return self._backend_train_info

    @property
    def backend_predict_info(self) -> Optional[dict[str, Any]]:
        """Backend prediction-route metadata from the most recent prediction."""
        return self._backend_predict_info

    @property
    def backend_sample_info(self) -> Optional[dict[str, Any]]:
        """Backend sampling-route metadata from the most recent sample draw."""
        return self._backend_sample_info

    # Map precond string to integer for Mojo side
    _PRECOND_METHOD_MAP = {
        "greedy": 0,
        "rpcholesky": 1,
        "nystrom": 2,
        "auto": 2,  # auto = nystrom
    }

    # Preset definitions for CG parameters
    _PRESETS = {
        "fast": {
            "max_cg_iter": 50,
            "cg_tol": 5e-2,
            "num_probes": 5,
            "max_tridiag_iter": 15,
            "precond_rank": 50,
            "precond_rebuild_threshold": 0.75,
            "precond": "greedy",
        },
        "balanced": {
            "max_cg_iter": 100,
            "cg_tol": 1e-2,
            "num_probes": 10,
            "max_tridiag_iter": 30,
            "precond_rank": 100,
            "precond_rebuild_threshold": 0.5,
            "precond": "greedy",
        },
        "accurate": {
            "max_cg_iter": 200,
            "cg_tol": 1e-3,
            "num_probes": 20,
            "max_tridiag_iter": 60,
            "precond_rank": 200,
            "precond_rebuild_threshold": 0.25,
            "precond": "greedy",
        },
    }

    @staticmethod
    def _resolve_cg_params(
        preset: Optional[str] = None,
        max_cg_iter: Optional[int] = None,
        cg_tol: Optional[float] = None,
        num_probes: Optional[int] = None,
        max_tridiag_iter: Optional[int] = None,
        precond_rank: Optional[int] = None,
        precond_rebuild_threshold: Optional[float] = None,
        precond: Optional[str] = None,
    ) -> dict:
        """Resolve CG parameters from preset + explicit overrides.

        Preset sets defaults; explicit params override the preset values.
        If no preset is given, uses 'balanced' defaults.
        """
        base = dict(SingleOutputGP._PRESETS[preset or "balanced"])
        if max_cg_iter is not None:
            base["max_cg_iter"] = max_cg_iter
        if cg_tol is not None:
            base["cg_tol"] = cg_tol
        if num_probes is not None:
            base["num_probes"] = num_probes
        if max_tridiag_iter is not None:
            base["max_tridiag_iter"] = max_tridiag_iter
        if precond_rank is not None:
            base["precond_rank"] = precond_rank
        if precond_rebuild_threshold is not None:
            base["precond_rebuild_threshold"] = precond_rebuild_threshold
        if precond is not None:
            base["precond"] = precond
        if "precond" not in base:
            base["precond"] = "auto"
        # Resolve precond string to integer
        base["precond_method"] = SingleOutputGP._PRECOND_METHOD_MAP.get(
            base["precond"],
            2,  # default to nystrom
        )
        return base

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
        initial_noise: float = 0.1,
        initial_params: Optional[np.ndarray] = None,
        method: str = "auto",
        enable_early_stopping: bool = False,
        early_stop_patience: int = 10,
        early_stop_tol: float = 1e-4,
        verbose: Optional[bool] = None,
        preset: Optional[str] = None,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: Optional[float] = None,
        num_probes: Optional[int] = None,
        max_tridiag_iterations: Optional[int] = None,
        preconditioner_rank: Optional[int] = None,
        precond_rebuild_threshold: Optional[float] = None,
        use_fused_kernels: bool = True,
        preconditioner: Optional[str] = None,
        use_preconditioner: Optional[bool] = None,
        lr_schedule: str = "constant",
        prepare_prediction_cache: bool = False,
        prediction_cache_rank: Optional[int] = None,
        observation_noise: Optional[np.ndarray] = None,
        observation_noise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise_function: Optional[str] = None,
        learn_noise: bool = True,
        noise_floor: float = 1e-6,
        noise_regularization: float = 0.01,
        noise_model: str = "scalar",
        noise_group_train: Optional[np.ndarray] = None,
        group_noise: Optional[np.ndarray] = None,
        progress: Any = None,
        progress_stats: Optional[Any] = None,
        progress_interval: int = 1,
    ) -> Union[TrainingResult, MixedTrainingResult]:
        """Train the GP on data.

        Args:
            max_iterations: Maximum training iterations
            learning_rate: Adam learning rate
            initial_noise: Initial noise variance
            initial_params: Initial kernel parameters. If None, uses values
                from kernel factory (e.g., RBF(lengthscale=0.5) -> [0.5, 1.0])
            lr_schedule: Learning rate schedule. "constant" (default) for a
                fixed learning rate throughout training, or "cosine" for cosine
                decay.
            prepare_prediction_cache: If True, prepare process-local train-side
                prediction state immediately after continuous SingleOutputGP training.
                This caches alpha and the LOVE inverse root on device without
                caching any test-dependent cross-covariance.
            prediction_cache_rank: Optional LOVE inverse-root rank for fit-time
                cache preparation. If omitted, prediction defaults are used.
            observation_noise: Optional known per-sample observation noise
                variances with shape ``(n,)``. Supplying this selects exact fixed
                diagonal-noise training with covariance ``K + diag(noise_i)``.
            observation_noise_fn: Optional callable that maps ``X`` to known
                per-sample observation-noise variances with shape ``(n,)``. This
                selects exact fixed input-dependent diagonal-noise training by
                evaluating the function on the supplied training inputs.
            noise_function: Learned input-dependent noise-function type. The
                only implemented value is ``"linear"`` with
                ``noise_model="learned_input_dependent"``.
            learn_noise: Whether to learn scalar observation noise. Must be
                False when ``observation_noise`` or ``observation_noise_fn`` is supplied.
            noise_floor: Minimum allowed observation noise variance.
            noise_regularization: L2 penalty strength on learned log-noise
                values toward the initial scalar noise. Used for in-development
                learned vector and learned grouped training.
            noise_model: Noise mode. ``"scalar"``, ``"fixed_vector"``, and
                ``"input_dependent"`` are implemented for continuous ExactGP.
                ``"grouped"`` supports
                fixed known group variances when ``learn_noise=False`` and
                learned group variances when ``learn_noise=True``. Learned
                vector, learned grouped, and learned input-dependent modes are
                in development, not public experimental functionality.
            noise_group_train: Integer group id per training sample for
                ``noise_model="grouped"``.
            group_noise: Fixed observation-noise variance per group for
                ``noise_model="grouped"``.
            progress: Progress reporting control. Use True for the default tqdm
                reporter, ``"auto"`` for tty-only tqdm, a callback, or a
                reporter object with start/update/close methods.
            progress_stats: Optional stat names or callable used by the default
                tqdm reporter.
            progress_interval: Emit ordinary training iteration updates every
                this many optimizer iterations. Start, complete, early-stop,
                and NaN events are not interval-gated.
            use_preconditioner: Whether to apply the pivoted-Cholesky
                preconditioner during training. If omitted, preconditioning is
                enabled only when `preconditioner_rank > 0`.
            method: Training method - one of:
                - "matrix_free": On-the-fly kernel computation (memory efficient)
                - "materialized": Pre-compute full kernel matrix (faster for small n)
                - "auto": Automatically choose based on n (default)
                Aliases: "mf" for "matrix_free", "mat" for "materialized".
            enable_early_stopping: Whether to stop training when the NLL fails
                to improve by ``early_stop_tol`` for ``early_stop_patience``
                iterations. Disabled by default so fixed-iteration benchmark
                runs remain explicit and reproducible.
            early_stop_patience: Stop if NLL doesn't improve for this many iterations
            early_stop_tol: Minimum NLL improvement to count as progress
            verbose: Print progress (overrides constructor setting if not None)
            preset: CG parameter preset - "fast", "balanced" (default), or "accurate".
                Sets defaults for CG params; explicit params override preset values.
            max_cg_iterations: Maximum CG iterations per solve
            cg_tolerance: CG convergence tolerance
            num_probes: Number of probe vectors for SLQ log-det estimation
            max_tridiag_iterations: Maximum tridiagonal iterations for SLQ
            preconditioner_rank: Maximum rank for Pivoted Cholesky preconditioner.
                For nystrom, this is the upper bound (actual rank is adaptive).
            precond_rebuild_threshold: NLL change threshold for preconditioner rebuild
                (lower = rebuild more often = more accurate but slower)
            use_fused_kernels: Whether to use fused GPU kernels that combine
                kernel evaluation + matvec into a single kernel launch (default True).
                When False, uses separate kernel evaluation and matvec steps.
                Fused kernels are faster for matrix-free methods but may not be
                available for all kernel types.
            preconditioner: Preconditioner construction method - one of:
                - "greedy": Deterministic argmax pivot selection (GPyTorch-compatible)
                - "rpcholesky": Randomized proportional sampling, fixed rank
                - "nystrom": RPCholesky + adaptive rank based on noise floor (best)
                - "auto": Same as "nystrom" (default)
                If None, uses the preset's default (typically "greedy").

        Returns:
            TrainingResult or MixedTrainingResult with optimized parameters
        """
        # Validate and store data
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples, y has {y.shape[0]}")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values")

        if noise_floor <= 0 or not np.isfinite(noise_floor):
            raise ValueError("noise_floor must be finite and > 0")
        if noise_regularization < 0 or not np.isfinite(noise_regularization):
            raise ValueError("noise_regularization must be finite and >= 0")
        if noise_model not in (
            "scalar",
            "fixed_vector",
            "input_dependent",
            "grouped",
            "learned_vector",
            "learned_input_dependent",
        ):
            raise ValueError(
                "noise_model must be one of 'scalar', 'fixed_vector', "
                "'input_dependent', 'grouped', 'learned_vector', or "
                "'learned_input_dependent'"
            )
        if noise_function is not None and noise_function != "linear":
            raise ValueError("noise_function must be 'linear' for learned input-dependent noise")
        if noise_function is not None and noise_model != "learned_input_dependent":
            raise ValueError("noise_function is only supported with noise_model='learned_input_dependent'")
        if observation_noise_fn is not None and not callable(observation_noise_fn):
            raise ValueError("observation_noise_fn must be callable")
        if observation_noise_fn is not None and observation_noise is not None:
            raise ValueError(
                "Pass either observation_noise or observation_noise_fn, not both"
            )
        if observation_noise_fn is not None and noise_model == "grouped":
            raise ValueError(
                "noise_model='grouped' expands group_noise to observation_noise; do not also pass observation_noise_fn"
            )
        if noise_model == "input_dependent" and observation_noise_fn is None:
            raise ValueError("noise_model='input_dependent' requires observation_noise_fn")
        if noise_model == "learned_input_dependent":
            if noise_function != "linear":
                raise ValueError(
                    "noise_model='learned_input_dependent' requires noise_function='linear'"
                )
            if observation_noise is not None:
                raise ValueError(
                    "noise_model='learned_input_dependent' learns training noise; do not also pass observation_noise"
                )
            if observation_noise_fn is not None:
                raise ValueError(
                    "noise_model='learned_input_dependent' learns its own noise function; do not also pass observation_noise_fn"
                )
            if noise_group_train is not None or group_noise is not None:
                raise ValueError(
                    "noise_model='learned_input_dependent' does not use grouped noise inputs"
                )
            if not learn_noise:
                raise ValueError(
                    "learn_noise must be True for noise_model='learned_input_dependent'"
                )
            if initial_noise <= noise_floor:
                raise ValueError(
                    "initial_noise must be greater than noise_floor for learned input-dependent noise"
                )
        if observation_noise is not None and noise_model == "grouped":
            raise ValueError(
                "noise_model='grouped' expands group_noise to observation_noise; do not also pass observation_noise"
            )
        if observation_noise_fn is not None:
            if learn_noise:
                raise ValueError("learn_noise must be False when observation_noise_fn is supplied")
            observation_noise = self._evaluate_observation_noise_fn(
                observation_noise_fn,
                X,
                expected_n=X.shape[0],
                name="observation_noise_fn",
                noise_floor=float(noise_floor),
            )
            noise_model = "fixed_input_dependent"
        if observation_noise is not None and noise_model == "scalar":
            noise_model = "fixed_vector"
        noise_group_train_arr = None
        group_noise_arr = None
        if noise_model == "grouped":
            if noise_group_train is None:
                raise ValueError("noise_model='grouped' requires noise_group_train")
            noise_group_train_arr = np.ascontiguousarray(noise_group_train, dtype=np.int32)
            if noise_group_train_arr.shape != (X.shape[0],):
                raise ValueError(
                    f"noise_group_train must have shape ({X.shape[0]},), got {noise_group_train_arr.shape}"
                )
            if np.any(noise_group_train_arr < 0):
                raise ValueError("noise_group_train must contain non-negative group ids")
            max_group = int(noise_group_train_arr.max(initial=-1))
            if learn_noise:
                if initial_noise <= noise_floor:
                    raise ValueError(
                        "initial_noise must be greater than noise_floor for learned grouped noise"
                    )
                if group_noise is None:
                    group_noise_arr = np.full(max_group + 1, float(initial_noise), dtype=np.float32)
                else:
                    group_noise_arr = np.ascontiguousarray(group_noise, dtype=np.float32)
                noise_model = "learned_grouped"
            else:
                if group_noise is None:
                    raise ValueError("noise_model='grouped' requires group_noise when learn_noise=False")
                group_noise_arr = np.ascontiguousarray(group_noise, dtype=np.float32)
                noise_model = "fixed_grouped"
            if group_noise_arr.ndim != 1:
                raise ValueError("group_noise must be a 1D array")
            if group_noise_arr.size == 0:
                raise ValueError("group_noise must contain at least one group variance")
            if np.any(~np.isfinite(group_noise_arr)):
                raise ValueError("group_noise contains NaN or Inf values")
            if np.any(group_noise_arr < noise_floor):
                raise ValueError("group_noise values must be >= noise_floor")
            if max_group >= group_noise_arr.shape[0]:
                raise ValueError(
                    "noise_group_train references a group id outside group_noise"
                )
            observation_noise = group_noise_arr[noise_group_train_arr]
        observation_noise_train = None
        provider_noise_mode_int = None
        if noise_model == "learned_vector":
            if observation_noise is not None:
                raise ValueError(
                    "noise_model='learned_vector' learns training noise; do not also pass observation_noise"
                )
            if not learn_noise:
                raise ValueError("learn_noise must be True for noise_model='learned_vector'")
            if initial_noise <= noise_floor:
                raise ValueError("initial_noise must be greater than noise_floor for learned_vector noise")
            observation_noise_train = np.full(X.shape[0], float(initial_noise), dtype=np.float32)
            provider_noise_mode_int = 2
        if noise_model == "learned_grouped":
            observation_noise_train = np.ascontiguousarray(observation_noise, dtype=np.float32)
            provider_noise_mode_int = 3
        if noise_model == "learned_input_dependent":
            observation_noise_train = np.full(X.shape[0], float(initial_noise), dtype=np.float32)
            provider_noise_mode_int = 4
        if observation_noise is None and noise_model == "fixed_vector":
            raise ValueError("noise_model='fixed_vector' requires observation_noise")
        if observation_noise is None and not learn_noise:
            raise NotImplementedError(
                "Fixed scalar noise is not implemented in the JIT training API yet; use learn_noise=True or provide observation_noise"
            )
        if observation_noise is not None and noise_model != "learned_grouped":
            if learn_noise:
                raise ValueError("learn_noise must be False when observation_noise is supplied")
            observation_noise_train = np.ascontiguousarray(observation_noise, dtype=np.float32)
            provider_noise_mode_int = 1
            if observation_noise_train.shape != (X.shape[0],):
                raise ValueError(
                    f"observation_noise must have shape ({X.shape[0]},), got {observation_noise_train.shape}"
                )
            if np.any(~np.isfinite(observation_noise_train)):
                raise ValueError("observation_noise contains NaN or Inf values")
            if np.any(observation_noise_train < noise_floor):
                raise ValueError("observation_noise values must be >= noise_floor")

        self._X_train = X
        self._y_train = y
        self._observation_noise_train = observation_noise_train
        self._noise_mode = noise_model
        self._noise_group_train = noise_group_train_arr
        self._noise_group_values = group_noise_arr
        self._observation_noise_fn = observation_noise_fn
        self._noise_function = "linear" if noise_model == "learned_input_dependent" else None
        self._noise_function_params = None
        self._noise_floor = float(noise_floor)
        self._noise_regularization = float(noise_regularization)
        self._provider_noise_mode_int = provider_noise_mode_int
        self.dim = X.shape[1]

        # Auto-detect categorical kernels from the kernel tree
        # This enables the new API: RBF(active_dims=[0,1]) * EHH(active_dims=[2], levels=5)
        if self._kernel_pre_ard.has_categorical():
            from .kernel import analyze_kernel_tree

            self._analysis = analyze_kernel_tree(self._kernel_pre_ard, self.dim)
            # Derive cat_dims and cat_kernel from the analysis
            # This bridges the new kernel-tree API to the existing mixed code path
            cat_kernel_dict = {}
            for spec in self._analysis.categorical_specs:
                self.cat_dims[spec.col_index] = spec.levels
                cat_kernel_dict[spec.col_index] = spec.kernel_type.name.lower()
            # Simplify cat_kernel if all same type
            unique_types = set(cat_kernel_dict.values())
            if len(unique_types) == 1:
                self.cat_kernel = list(unique_types)[0]
            else:
                self.cat_kernel = cat_kernel_dict
        else:
            self._analysis = None

        # Compute categorical column indices and continuous dimension
        self._cat_col_indices = sorted(self.cat_dims.keys())
        self._cont_dim = self.dim - len(self._cat_col_indices)

        if self._is_mixed and self._cont_dim <= 0:
            raise ValueError(
                "Pure categorical kernels (all dimensions categorical, no continuous) "
                "are not supported. At least one continuous input dimension is required. "
                "Consider using a mixed kernel with both continuous and categorical "
                "components, e.g.: RBF(active_dims=[0]) * EHH(levels=5, active_dims=[1]). "
                f"Got dim={self.dim} with {len(self._cat_col_indices)} categorical columns "
                f"leaving {self._cont_dim} continuous dims."
            )

        base_kernel = self._kernel_pre_ard
        if self._analysis is not None:
            base_kernel = self._analysis.structured_kernel

        if self._is_mixed:
            cont_cols = [d for d in range(self.dim) if d not in self._cat_col_indices]
            dim_map = {orig: idx for idx, orig in enumerate(cont_cols)}
            base_kernel = self._remap_kernel_active_dims(base_kernel, dim_map)

        # Handle ARD: now that we know dim, resolve the compiled continuous kernel.
        if self.ard:
            self.kernel = make_ard_kernel(base_kernel, self._cont_dim)
        else:
            self.kernel = base_kernel

        # Validate active_dims bounds
        if self.kernel.has_active_dims():
            self._validate_active_dims_bounds(self.kernel, self._cont_dim)

        # Handle active_dims: compute column permutation for DimSlice routing
        self._dim_permutation = None
        if self.kernel.has_active_dims():
            from .kernel import compute_dim_permutation

            perm, effective_dim = compute_dim_permutation(self.kernel, self._cont_dim)
            self._dim_permutation = perm
            if effective_dim != self._cont_dim:
                self._cont_dim = effective_dim

        surface = surface_for_single_output(self._is_mixed)
        warn_surface_status(surface, stacklevel=2)
        if self.ard:
            check_feature_support(TABLE_MAIN, surface, "ard", stacklevel=2)
        guard_kernel_tree_features(surface, self.kernel, stacklevel=2)
        if self._noise_mode in {"fixed_vector", "fixed_input_dependent"}:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "fixed_per_sample_noise",
                stacklevel=2,
            )
        if self._noise_mode == "learned_input_dependent":
            check_feature_support(
                TABLE_MAIN,
                surface,
                "learned_input_dependent_noise",
                stacklevel=2,
            )
        if self._noise_mode in {"fixed_grouped", "learned_grouped"}:
            check_feature_support(
                TABLE_MAIN,
                surface,
                "grouped_noise",
                stacklevel=2,
            )

        if preset is not None and preset not in self._PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Must be one of: {list(self._PRESETS.keys())}"
            )
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {max_iterations}")
        progress_interval = _validate_progress_interval(progress_interval)
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if initial_noise <= 0:
            raise ValueError(f"initial_noise must be > 0, got {initial_noise}")
        if early_stop_patience < 0:
            raise ValueError(
                f"early_stop_patience must be >= 0, got {early_stop_patience}"
            )
        if early_stop_tol < 0:
            raise ValueError(f"early_stop_tol must be >= 0, got {early_stop_tol}")
        if prediction_cache_rank is not None and prediction_cache_rank <= 0:
            raise ValueError("prediction_cache_rank must be positive or None")

        if not enable_early_stopping and (
            early_stop_patience != 10 or early_stop_tol != 1e-4
        ):
            warnings.warn(
                "early_stop_patience and early_stop_tol are set but "
                "enable_early_stopping=False, so training will run to "
                "max_iterations.",
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
        if lr_schedule not in ("cosine", "constant"):
            raise ValueError(
                f"lr_schedule must be 'cosine' or 'constant', got '{lr_schedule}'"
            )

        use_cosine_lr = lr_schedule == "cosine"

        self._destroy_provider_info()

        # Validate preconditioner parameter
        if preconditioner is not None and preconditioner not in self._PRECOND_METHOD_MAP:
            raise ValueError(
                f"Unknown preconditioner '{preconditioner}'. Must be one of: "
                f"{list(self._PRECOND_METHOD_MAP.keys())}"
            )

        # Use kernel's initial values if no explicit params provided
        default_initial_params = initial_params is None
        if default_initial_params:
            initial_params = self.kernel.get_initial_params()
        else:
            initial_params = np.asarray(initial_params, dtype=np.float32)
            if initial_params.shape[0] != self.num_params:
                raise ValueError(
                    f"initial_params has {initial_params.shape[0]} values, "
                    f"expected {self.num_params}"
                )

        # Minimum noise floor for periodic kernels
        if self._has_periodic_kernel() and initial_noise < 0.05:
            initial_noise = max(initial_noise, 0.05)

        # Resolve verbose
        _verbose = verbose if verbose is not None else self.verbose

        if method == "materialized_grads":
            raise NotImplementedError(
                "method='materialized_grads' is no longer part of the public "
                "SingleOutputGP API; use method='materialized' or "
                "method='matrix_free'."
            )
        method = normalize_fit_method(method, allow_auto=True)
        method_map = {
            "matrix_free": 0,
            "materialized": 1,
            "auto": 3,
        }
        resolved_method = method
        if self._is_mixed:
            if method == "auto":
                resolved_method = (
                    "materialized" if X.shape[0] <= 2000 else "matrix_free"
                )
        elif method == "auto" and X.shape[0] <= 2000:
            # Small exact GP workloads benefit from the more accurate dense path.
            resolved_method = "materialized"

        if method == "auto":
            check_feature_support(TABLE_EXECUTION, surface, "auto_selection", stacklevel=2)
        route_feature = (
            "materialized_training"
            if resolved_method == "materialized"
            else "matrix_free_training"
        )
        check_feature_support(TABLE_EXECUTION, surface, route_feature, stacklevel=2)
        self._training_method = resolved_method

        # Ensure compiled after public route selection so specialization metadata
        # sees the actual training route.
        self._ensure_compiled()

        resolved_preset = preset

        if (
            default_initial_params
            and resolved_method == "materialized"
            and learning_rate <= 0.05
            and self._is_single_kernel_type(KernelType.RBF)
            and self._cont_dim == 1
            and initial_params.shape[0] >= 1
        ):
            # Pure 1D RBF fits on small exact workloads are materially more stable
            # when they start from a shorter lengthscale than the generic 1.0
            # default. This keeps periodic-like data from getting stuck in an
            # overly smooth basin while leaving explicit user initializations
            # untouched.
            initial_params = np.asarray(initial_params, dtype=np.float32).copy()
            initial_params[0] = min(float(initial_params[0]), 0.5)

        if (
            default_initial_params
            and resolved_method == "matrix_free"
            and learning_rate <= 0.05
            and self._is_single_kernel_type(KernelType.RBF)
            and not self.ard
            and 1 < self._cont_dim <= 8
            and X.shape[0] >= 10000
            and initial_params.shape[0] >= 1
        ):
            # Large low-dimensional matrix-free RBF fits can get stuck in an
            # overly smooth unpreconditioned basin when they start from the
            # generic lengthscale=1.0 default. A milder 0.6 start materially
            # improves the d=5 large-n fair lane without affecting explicit
            # user initializations or higher-dimensional routes.
            initial_params = np.asarray(initial_params, dtype=np.float32).copy()
            initial_params[0] = min(float(initial_params[0]), 0.6)

        resolved_precond = preconditioner
        resolved_precond_rank = preconditioner_rank
        preconditioner_controls_explicit = (
            preconditioner is not None
            or preconditioner_rank is not None
            or use_preconditioner is not None
        )
        if (
            resolved_precond is None
            and resolved_method == "materialized"
            and learning_rate <= 0.05
            and self._is_single_kernel_type(KernelType.MATERN32)
        ):
            # The low-LR materialized Matern-3/2 path was intermittently landing
            # on poor randomized pivots. Greedy pivots are slower but much more
            # stable for these exact small-data fits.
            resolved_precond = "greedy"

        if (
            not preconditioner_controls_explicit
            and resolved_method == "materialized"
            and self._is_single_kernel_type(KernelType.LINEAR)
        ):
            # Linear kernels are already low-rank plus diagonal noise; the dense
            # route solves them reliably without the generic kernel preconditioner.
            resolved_precond_rank = 0

        if (
            not preconditioner_controls_explicit
            and resolved_method == "matrix_free"
            and not self._is_mixed
        ):
            if (
                self._is_single_kernel_type(KernelType.RBF)
                and not self.ard
                and 1 < self._cont_dim <= 8
                and X.shape[0] >= 5000
            ):
                # Fixed-seed local sweeps show matrix-free non-ARD RBF in low
                # dimensions has enough CG pressure to repay the build cost.
                resolved_precond = "greedy"
                resolved_precond_rank = 256
            else:
                # Preconditioning is not a safe global default: smooth d=17 and
                # structured ARD lanes kept the same CG count and only paid
                # build/apply overhead. Keep unsupported matrix-free routes off
                # unless the caller opts in explicitly.
                resolved_precond_rank = 0

        resolved_precond_rebuild_threshold = precond_rebuild_threshold
        if (
            resolved_precond_rebuild_threshold is None
            and resolved_method == "materialized"
            and self.ard
            and not self._is_mixed
            and self._supports_materialized_ard_precond_defaults()
        ):
            resolved_precond_rebuild_threshold = 0.1

        # Resolve CG parameters from preset + explicit overrides after auto mode
        # has selected the concrete training route.
        cg_params = self._resolve_cg_params(
            preset=resolved_preset,
            max_cg_iter=max_cg_iterations,
            cg_tol=cg_tolerance,
            num_probes=num_probes,
            max_tridiag_iter=max_tridiag_iterations,
            precond_rank=resolved_precond_rank,
            precond_rebuild_threshold=resolved_precond_rebuild_threshold,
            precond=resolved_precond,
        )

        method_int = method_map[resolved_method]
        self._training_method = resolved_method
        progress_adapter = resolve_progress_adapter(
            progress,
            operation="train",
            model="single_output",
            route=resolved_method,
            progress_stats=progress_stats,
        )

        # Split data into continuous and categorical
        X_cont, C = self._split_data(self._X_train)
        X_cont = self._apply_dim_permutation(X_cont)

        if self._is_mixed:
            if self._noise_mode != "scalar":
                raise NotImplementedError(
                    "Heteroskedastic observation noise is currently implemented for continuous ExactGP only; mixed, multi-output, and LMC noise extensions are in development"
                )
            if prepare_prediction_cache:
                raise NotImplementedError(
                    "fit(prepare_prediction_cache=True) currently supports continuous SingleOutputGP only"
                )
            return self._fit_mixed(
                self._X_train,
                X_cont,
                C,
                self._y_train,
                initial_params,
                initial_noise,
                max_iterations,
                learning_rate,
                method_int,
                enable_early_stopping,
                early_stop_patience,
                early_stop_tol,
                _verbose,
                cg_params,
                use_preconditioner=use_preconditioner,
                use_fused_kernels=use_fused_kernels,
                use_cosine_lr=use_cosine_lr,
                progress_adapter=progress_adapter,
                progress_interval=progress_interval,
            )
        else:
            fit_result = self._fit_continuous(
                X_cont,
                self._y_train,
                initial_params,
                initial_noise,
                max_iterations,
                learning_rate,
                method_int,
                enable_early_stopping,
                early_stop_patience,
                early_stop_tol,
                _verbose,
                cg_params,
                use_preconditioner=use_preconditioner,
                use_fused_kernels=use_fused_kernels,
                use_cosine_lr=use_cosine_lr,
                learn_noise=learn_noise,
                progress_adapter=progress_adapter,
                progress_interval=progress_interval,
            )
            if prepare_prediction_cache:
                self.prepare_prediction_cache(
                    variance_method="love",
                    max_root_decomposition_size=prediction_cache_rank,
                )
            return fit_result

    def _fit_continuous(
        self,
        X,
        y,
        initial_params,
        initial_noise,
        max_iterations,
        learning_rate,
        method_int,
        enable_early_stopping,
        early_stop_patience,
        early_stop_tol,
        verbose,
        cg_params,
        use_fused_kernels=True,
        use_cosine_lr=True,
        use_preconditioner=None,
        learn_noise=True,
        progress_adapter=None,
        progress_interval=1,
    ) -> TrainingResult:
        """Train a continuous-only GP."""
        init_mean = (
            self._init_mean if self._init_mean is not None else float(np.mean(y))
        )

        if verbose:
            print(f"Training GP with {self.kernel.to_mojo_type()}")
            print(f"  n={X.shape[0]}, dim={self.dim}, params={self.num_params}")
            print(f"  init_mean={init_mean:.4f}")

        # Always use engine path
        self._ensure_compiled()
        return self._fit_engine(
            X,
            y,
            initial_params,
            initial_noise,
            max_iterations,
            learning_rate,
            method_int,
            enable_early_stopping,
            verbose,
            cg_params,
            use_preconditioner=use_preconditioner,
            use_cosine_lr=use_cosine_lr,
            init_mean=init_mean,
            early_stop_patience=early_stop_patience,
            early_stop_tol=early_stop_tol,
            learn_noise=learn_noise,
            progress_adapter=progress_adapter,
            progress_interval=progress_interval,
        )

    def _fit_engine(
        self,
        X,
        y,
        initial_params,
        initial_noise,
        max_iterations,
        learning_rate,
        method_int,
        enable_early_stopping,
        verbose,
        cg_params,
        use_cosine_lr,
        init_mean,
        use_preconditioner=None,
        early_stop_patience=10,
        early_stop_tol=1e-4,
        learn_noise=True,
        progress_adapter=None,
        progress_interval=1,
    ):
        """Train using the production fn-ptr engine path.

        Two-step process:
        1. kernel .so: init_provider(X, params, noise) -> fn-ptr dict
        2. engine .so: train(provider_info, y, ...) -> result dict
        """
        import numpy as np

        # Step 1: Initialize provider in kernel .so
        engine_params = self.kernel.to_engine_params(
            np.array(initial_params, dtype=np.float32)
        )
        self._destroy_provider_info()
        self._training_method = "materialized" if method_int == 1 else "matrix_free"
        self._provider_info = self._build_provider_info(
            X,
            engine_params,
            initial_noise,
            observation_noise=self._observation_noise_train,
            noise_mode_int=getattr(self, "_provider_noise_mode_int", None),
            noise_group_train=self._noise_group_train,
            num_noise_groups=None if self._noise_group_values is None else len(self._noise_group_values),
        )
        self._provider_state_current = True
        register_provider_lease(self._kernel_module, self, self._revoke_provider_info)

        resolved_use_preconditioner = (
            bool(use_preconditioner)
            if use_preconditioner is not None
            else int(cg_params["precond_rank"]) > 0
        )
        effective_precond_rank = (
            int(cg_params["precond_rank"]) if resolved_use_preconditioner else 0
        )

        # Step 2: Train via engine .so
        train_args = [
            self._provider_info,
            y.astype(np.float32),
            engine_params,
            float(initial_noise),
            max_iterations,
            float(learning_rate),
            cg_params["num_probes"],
            cg_params["max_cg_iter"],
            float(cg_params["cg_tol"]),
            effective_precond_rank,
            verbose,
            use_cosine_lr,
            resolved_use_preconditioner,
            cg_params["max_tridiag_iter"],
            float(cg_params["precond_rebuild_threshold"]),
            cg_params["precond_method"],
            float(init_mean),
            bool(enable_early_stopping),
            int(early_stop_patience),
            float(early_stop_tol),
            bool(learn_noise),
            float(self._noise_floor),
            float(self._noise_regularization),
        ]
        if progress_adapter is not None:
            train_args.extend([progress_adapter.callback, int(progress_interval)])
        try:
            result = self._engine.train(*train_args)
        except Exception as exc:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True, message=str(exc))
            raise
        if progress_adapter is not None:
            progress_adapter.close_if_needed()

        self._is_trained = True
        self._invalidate_prediction_caches()
        self._provider_state_current = False
        learned_mean = float(result.get("mean", init_mean))
        self._engine_predict_mean = learned_mean
        self._fitted_mean = learned_mean
        self._backend_train_info = {
            "training_route": str(
                result.get(
                    "training_route",
                    "materialized" if method_int == 1 else "matrix_free",
                )
            ),
            "materialization_mode": int(
                result.get(
                    "materialization_mode",
                    self._provider_info.get("materialization_mode", 0),
                )
            ),
            "is_ard": bool(
                result.get("is_ard", self._provider_info.get("is_ard", False))
            ),
            "precond_method": int(
                result.get("precond_method", cg_params["precond_method"])
            ),
            "precond_rank": int(result.get("precond_rank", cg_params["precond_rank"])),
            "max_cg_iter": int(cg_params["max_cg_iter"]),
            "max_cg_iterations": int(cg_params["max_cg_iter"]),
            "cg_tol": float(cg_params["cg_tol"]),
            "cg_tolerance": float(cg_params["cg_tol"]),
            "max_tridiag_iter": int(
                result.get("max_tridiag_iter", cg_params["max_tridiag_iter"])
            ),
            "precond_rebuild_threshold": float(
                result.get(
                    "precond_rebuild_threshold",
                    cg_params["precond_rebuild_threshold"],
                )
            ),
            "enable_early_stopping": bool(
                result.get("enable_early_stopping", enable_early_stopping)
            ),
            "early_stop_patience": int(
                result.get("early_stop_patience", early_stop_patience)
            ),
            "early_stop_tol": float(result.get("early_stop_tol", early_stop_tol)),
            "use_preconditioner": bool(
                result.get("use_preconditioner", resolved_use_preconditioner)
            ),
            "noise_mode": self._noise_mode,
            "learn_noise": bool(result.get("learn_noise", learn_noise)),
            "has_observation_noise_vector": self._observation_noise_train is not None,
            "noise_regularization": float(result.get("noise_regularization", self._noise_regularization)),
        }
        self._maybe_attach_specialization_metadata(self._backend_train_info)
        for key in (
            "actual_precond_rank",
            "precond_build_count",
            "precond_rebuild_count",
            "precond_build_total_ms",
            "precond_rank_history",
            "precond_rebuild_steps",
        ):
            if key in result:
                value = result[key]
                if key in {"precond_rank_history", "precond_rebuild_steps"}:
                    self._backend_train_info[key] = list(value)
                elif key == "precond_build_total_ms":
                    self._backend_train_info[key] = float(value)
                else:
                    self._backend_train_info[key] = int(value)

        # Convert engine result to TrainingResult
        engine_result_params = np.array(list(result["params"]), dtype=np.float32)
        params = self._continuous_param_kernel().from_engine_params(
            engine_result_params
        )
        nll_history_raw = result.get("nll_history", None)
        nll_history = (
            np.array(list(nll_history_raw), dtype=np.float32)
            if nll_history_raw is not None
            else None
        )
        iter_times_raw = result.get("iter_times_ms", None)
        iter_times_ms = (
            np.array(list(iter_times_raw), dtype=np.float32)
            if iter_times_raw is not None
            else None
        )
        cg_iterations_history_raw = result.get("cg_iterations_history", None)
        cg_iterations_history = (
            np.array(list(cg_iterations_history_raw), dtype=np.int32)
            if cg_iterations_history_raw is not None
            else None
        )
        self._training_result = TrainingResult(
            params=params,
            noise=float(result["noise"]),
            mean=learned_mean,
            nll=float(result["final_nll"]),
            iterations=int(result["iterations"]),
            converged=bool(result["converged"]),
            lanczos_root=None,  # Computed lazily at prediction time
            lanczos_rank=int(cg_params["max_tridiag_iter"]),
            nll_history=nll_history,
            cg_iterations_history=cg_iterations_history,
            iter_times_ms=iter_times_ms,
        )
        learned_observation_noise = result.get("learned_observation_noise")
        if learned_observation_noise is not None:
            self._observation_noise_train = np.ascontiguousarray(
                learned_observation_noise, dtype=np.float32
            )
            self._backend_train_info["has_observation_noise_vector"] = True
            self._backend_train_info["learned_observation_noise_min"] = float(
                np.min(self._observation_noise_train)
            )
            self._backend_train_info["learned_observation_noise_max"] = float(
                np.max(self._observation_noise_train)
            )
            self._backend_train_info["learned_observation_noise_mean"] = float(
                np.mean(self._observation_noise_train)
            )
            if self._noise_mode == "learned_grouped" and self._noise_group_train is not None:
                group_values = np.zeros(int(np.max(self._noise_group_train)) + 1, dtype=np.float32)
                for group_id in range(len(group_values)):
                    mask = self._noise_group_train == group_id
                    if np.any(mask):
                        group_values[group_id] = float(np.mean(self._observation_noise_train[mask]))
                self._noise_group_values = np.ascontiguousarray(group_values, dtype=np.float32)
                self._backend_train_info["learned_group_noise"] = self._noise_group_values.tolist()
        learned_noise_function_params = result.get("learned_noise_function_params")
        if learned_noise_function_params is not None:
            self._noise_function_params = np.ascontiguousarray(
                learned_noise_function_params, dtype=np.float32
            )
            self._backend_train_info["learned_noise_function"] = self._noise_function
            self._backend_train_info["learned_noise_function_params"] = self._noise_function_params.tolist()
        cached_alpha_result = result.get("cached_alpha")
        if cached_alpha_result is not None and self._noise_mode not in (
            "learned_vector",
            "learned_grouped",
            "learned_input_dependent",
        ):
            self._store_cached_alpha(
                cached_alpha_result,
                {
                    "max_cg_iterations": int(cg_params["max_cg_iter"]),
                    "cg_tolerance": float(cg_params["cg_tol"]),
                    "precond_rank": int(cg_params["precond_rank"]),
                },
                "training",
            )
        if iter_times_ms is not None and len(iter_times_ms) > 0:
            self._backend_train_info["iter_times_ms"] = iter_times_ms.tolist()
            self._backend_train_info["iter_time_median_ms"] = float(
                np.median(iter_times_ms)
            )
            self._backend_train_info["iter_time_p5_ms"] = float(
                np.percentile(iter_times_ms, 5)
            )
            self._backend_train_info["iter_time_p95_ms"] = float(
                np.percentile(iter_times_ms, 95)
            )
        if cg_iterations_history is not None and len(cg_iterations_history) > 0:
            self._backend_train_info["cg_iterations_history"] = (
                cg_iterations_history.tolist()
            )
            self._backend_train_info["cg_iterations_total"] = int(
                np.sum(cg_iterations_history)
            )
            self._backend_train_info["cg_iterations_mean"] = float(
                np.mean(cg_iterations_history)
            )
            self._backend_train_info["cg_iterations_max"] = int(
                np.max(cg_iterations_history)
            )
            self._backend_train_info["cg_iterations_final_step"] = int(
                cg_iterations_history[-1]
            )
            self._backend_train_info["cg_iterations_measured"] = True
        self._backend_sample_info = None
        return self._training_result

    def _predict_engine(
        self,
        X_test,
        variance_method,
        method: Optional[str] = None,
        *,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: Optional[float] = None,
        preconditioner_rank: Optional[int] = None,
        max_root_decomposition_size: Optional[int] = None,
        exact_prediction_block_cols: Optional[int] = None,
    ):
        """Predict using the production fn-ptr engine path."""
        import numpy as np

        self._ensure_compiled()

        tr = self._training_result
        params = self.kernel.to_engine_params(np.array(tr.params, dtype=np.float32))
        train_info = self._backend_train_info or {}
        prediction_method = method or getattr(self, "_training_method", None) or "matrix_free"
        solver_config = _prediction_solver_config(
            variance_method,
            train_info,
            getattr(tr, "lanczos_rank", 0),
            prediction_method,
            is_mixed=False,
            is_ard=self.ard,
            max_cg_iterations=max_cg_iterations,
            cg_tolerance=cg_tolerance,
            precond_rank=preconditioner_rank,
            max_root_decomposition_size=max_root_decomposition_size,
        )

        # Map variance method to int: 0=mean_only, 1=love, 2=exact
        if variance_method == "exact":
            vm_int = 2  # PREDICT_EXACT
        elif variance_method == "love" or variance_method is None:
            vm_int = 1  # PREDICT_LOVE
        elif variance_method == "mean_only":
            vm_int = 0  # PREDICT_MEAN_ONLY
        else:
            vm_int = 0  # PREDICT_MEAN_ONLY (fallback)

        provider_info = self._provider_info
        built_provider_info = None
        can_reuse_training_provider = (
            provider_info is not None
            and prediction_method == (getattr(self, "_training_method", None) or "matrix_free")
            and (
                self._provider_state_current
                or self._noise_mode
                in (
                    "learned_vector",
                    "learned_grouped",
                    "learned_input_dependent",
                )
            )
        )
        if not can_reuse_training_provider:
            provider_info = self._build_provider_info(
                self._apply_dim_permutation(self._X_train),
                params,
                tr.noise,
                method=prediction_method,
                observation_noise=self._observation_noise_train,
                noise_mode_int=getattr(self, "_provider_noise_mode_int", None),
                noise_group_train=self._noise_group_train,
                num_noise_groups=None if self._noise_group_values is None else len(self._noise_group_values),
            )
            built_provider_info = provider_info
        provider_state_current = (
            (can_reuse_training_provider and self._provider_state_current)
            or built_provider_info is not None
        )
        try:
            mean_value = float(
                self._engine_predict_mean
                if self._engine_predict_mean is not None
                else tr.mean
            )
            if self._prediction_cache_valid(
                prediction_method=prediction_method,
                variance_method=variance_method,
                solver_config=solver_config,
                engine_params=params,
                noise=float(tr.noise),
                mean=mean_value,
            ):
                engine_args = [
                    provider_info,
                    int(self._prediction_cache_handle),
                    X_test.astype(np.float32),
                    params,
                    float(tr.noise),
                    mean_value,
                    vm_int,
                    int(solver_config["max_cg_iterations"]),
                    float(solver_config["cg_tolerance"]),
                    int(solver_config["precond_rank"]),
                    int(solver_config["max_root_decomposition_size"]),
                    bool(provider_state_current),
                    int(exact_prediction_block_cols or 0),
                ]
                result = self._engine.predict_with_cache(*engine_args)
            else:
                cached_alpha = self._cached_alpha_for_solver(solver_config)
                cached_lanczos_root = None
                if variance_method == "love" and isinstance(tr, TrainingResult):
                    root = tr.lanczos_root
                    if root is not None and self._cached_love_method == prediction_method:
                        root_rank = int(solver_config["max_root_decomposition_size"])
                        if len(root) == len(self._y_train) * root_rank:
                            cached_lanczos_root = np.ascontiguousarray(
                                root, dtype=np.float32
                            )
                        else:
                            tr.lanczos_root = None
                            self._cached_love_method = None
                engine_args = [
                    provider_info,
                    self._y_train.astype(np.float32),
                    X_test.astype(np.float32),
                    params,
                    float(tr.noise),
                    mean_value,
                    vm_int,
                    int(solver_config["max_cg_iterations"]),
                    float(solver_config["cg_tolerance"]),
                    int(solver_config["precond_rank"]),
                    int(solver_config["max_root_decomposition_size"]),
                    bool(provider_state_current),
                    int(exact_prediction_block_cols or 0),
                ]
                if cached_alpha is not None:
                    engine_args.append(np.ascontiguousarray(cached_alpha, dtype=np.float32))
                if cached_alpha is not None and cached_lanczos_root is not None:
                    engine_args.append(cached_lanczos_root)
                result = self._engine.predict(*engine_args)
                cached_alpha_result = result.get("cached_alpha")
                if cached_alpha_result is not None:
                    self._store_cached_alpha(
                        cached_alpha_result,
                        solver_config,
                        "prediction",
                    )
                cached_lanczos_root_result = result.get("cached_lanczos_root")
                if cached_lanczos_root_result is not None and isinstance(tr, TrainingResult):
                    tr.lanczos_root = np.asarray(cached_lanczos_root_result, dtype=np.float32)
                    self._cached_love_method = prediction_method
        finally:
            if built_provider_info is not None:
                self._destroy_temporary_provider_info(built_provider_info)
        if can_reuse_training_provider and built_provider_info is None:
            self._provider_state_current = True

        return (
            result["mean"],
            result.get("variance", np.zeros(len(X_test))),
            result,
            solver_config,
            prediction_method,
        )

    def _prediction_cache_signature_for(
        self,
        *,
        prediction_method: str,
        variance_method: str,
        solver_config: dict[str, Any],
        engine_params: np.ndarray,
        noise: float,
        mean: float,
    ) -> tuple[Any, ...]:
        return (
            prediction_method,
            variance_method,
            int(len(self._y_train) if self._y_train is not None else 0),
            int(self.dim or 0),
            int(solver_config["max_cg_iterations"]),
            float(solver_config["cg_tolerance"]),
            int(solver_config["precond_rank"]),
            int(solver_config["max_root_decomposition_size"]),
            tuple(np.asarray(engine_params, dtype=np.float32).ravel().tolist()),
            float(noise),
            self._noise_mode,
            None
            if self._observation_noise_train is None
            else tuple(np.asarray(self._observation_noise_train, dtype=np.float32).ravel().tolist()),
            float(mean),
        )

    def _prediction_cache_valid(
        self,
        *,
        prediction_method: str,
        variance_method: str,
        solver_config: dict[str, Any],
        engine_params: np.ndarray,
        noise: float,
        mean: float,
    ) -> bool:
        handle = self._prediction_cache_handle
        if not handle:
            return False
        expected = self._prediction_cache_signature_for(
            prediction_method=prediction_method,
            variance_method=variance_method,
            solver_config=solver_config,
            engine_params=engine_params,
            noise=noise,
            mean=mean,
        )
        return self._prediction_cache_signature == expected

    def _cached_alpha_for_solver(
        self,
        solver_config: dict[str, Any],
    ) -> Optional[np.ndarray]:
        cached_alpha = self._cached_alpha
        if cached_alpha is None or self._y_train is None:
            return None
        if len(cached_alpha) != len(self._y_train):
            self._cached_alpha = None
            self._cached_alpha_info = None
            return None
        info = self._cached_alpha_info or {}
        cached_max_iter = int(info.get("max_cg_iterations", 0) or 0)
        cached_tol = float(info.get("cg_tolerance", float("inf")))
        requested_max_iter = int(solver_config["max_cg_iterations"])
        requested_tol = float(solver_config["cg_tolerance"])
        if cached_max_iter < requested_max_iter or cached_tol > requested_tol:
            return None
        return cached_alpha

    def _store_cached_alpha(
        self,
        alpha: Any,
        solver_config: dict[str, Any],
        source: str,
    ) -> None:
        self._cached_alpha = np.asarray(alpha, dtype=np.float32)
        self._cached_alpha_info = {
            "source": source,
            "max_cg_iterations": int(solver_config["max_cg_iterations"]),
            "cg_tolerance": float(solver_config["cg_tolerance"]),
            "precond_rank": int(solver_config["precond_rank"]),
        }

    def prepare_prediction_cache(
        self,
        variance_method: str = "love",
        method: Optional[str] = None,
        *,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: Optional[float] = None,
        preconditioner_rank: Optional[int] = None,
        max_root_decomposition_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """Prepare train-side prediction state for fast future predictions.

        The prepared cache is independent of `X_test`: it stores only alpha and,
        for LOVE, the train-side inverse root. It is process-local and is
        invalidated by refitting or changing prediction settings.
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before preparing prediction cache")
        surface = surface_for_single_output(self._is_mixed)
        check_feature_support(TABLE_PREDICTION, surface, "prediction_cache", stacklevel=2)
        if self._is_mixed:
            raise NotImplementedError(
                "prepare_prediction_cache() currently supports continuous SingleOutputGP only"
            )
        if variance_method not in ("love", "exact", "mean_only"):
            raise ValueError("variance_method must be one of 'love', 'exact', 'mean_only'")
        if method not in (None, "materialized", "matrix_free"):
            raise ValueError("method must be one of None, 'materialized', or 'matrix_free'")
        self._ensure_compiled()
        tr = self._training_result
        assert isinstance(tr, TrainingResult)
        engine_params = self.kernel.to_engine_params(np.array(tr.params, dtype=np.float32))
        train_info = self._backend_train_info or {}
        prediction_method = method or getattr(self, "_training_method", None) or "matrix_free"
        solver_config = _prediction_solver_config(
            variance_method,
            train_info,
            getattr(tr, "lanczos_rank", 0),
            prediction_method,
            is_mixed=False,
            is_ard=self.ard,
            max_cg_iterations=max_cg_iterations,
            cg_tolerance=cg_tolerance,
            precond_rank=preconditioner_rank,
            max_root_decomposition_size=max_root_decomposition_size,
        )
        mean_value = float(
            self._engine_predict_mean if self._engine_predict_mean is not None else tr.mean
        )
        signature = self._prediction_cache_signature_for(
            prediction_method=prediction_method,
            variance_method=variance_method,
            solver_config=solver_config,
            engine_params=engine_params,
            noise=float(tr.noise),
            mean=mean_value,
        )
        if self._prediction_cache_handle and self._prediction_cache_signature == signature:
            return dict(self._prediction_cache_info or {})

        self._destroy_prediction_cache()
        if variance_method == "exact":
            vm_int = 2
        elif variance_method == "love" or variance_method is None:
            vm_int = 1
        else:
            vm_int = 0

        provider_info = self._provider_info
        built_provider_info = None
        can_reuse_training_provider = (
            provider_info is not None
            and prediction_method == (getattr(self, "_training_method", None) or "matrix_free")
        )
        if not can_reuse_training_provider:
            provider_info = self._build_provider_info(
                self._apply_dim_permutation(self._X_train),
                engine_params,
                tr.noise,
                method=prediction_method,
                observation_noise=self._observation_noise_train,
                noise_mode_int=getattr(self, "_provider_noise_mode_int", None),
                noise_group_train=self._noise_group_train,
                num_noise_groups=None if self._noise_group_values is None else len(self._noise_group_values),
            )
            built_provider_info = provider_info
        provider_state_current = (
            (can_reuse_training_provider and self._provider_state_current)
            or built_provider_info is not None
        )
        try:
            cached_alpha = self._cached_alpha_for_solver(solver_config)
            cached_lanczos_root = None
            if variance_method == "love":
                root = tr.lanczos_root
                if root is not None and self._cached_love_method == prediction_method:
                    root_rank = int(solver_config["max_root_decomposition_size"])
                    if len(root) == len(self._y_train) * root_rank:
                        cached_lanczos_root = np.ascontiguousarray(root, dtype=np.float32)
                    else:
                        tr.lanczos_root = None
                        self._cached_love_method = None
            engine_args = [
                provider_info,
                self._y_train.astype(np.float32),
                np.ascontiguousarray(engine_params, dtype=np.float32),
                float(tr.noise),
                mean_value,
                vm_int,
                int(solver_config["max_cg_iterations"]),
                float(solver_config["cg_tolerance"]),
                int(solver_config["precond_rank"]),
                int(solver_config["max_root_decomposition_size"]),
                bool(provider_state_current),
            ]
            if cached_alpha is not None:
                engine_args.append(np.ascontiguousarray(cached_alpha, dtype=np.float32))
            if cached_alpha is not None and cached_lanczos_root is not None:
                engine_args.append(cached_lanczos_root)
            result = self._engine.prepare_prediction_cache(*engine_args)
        finally:
            if built_provider_info is not None:
                destroy = getattr(self._kernel_module, "destroy_provider", None)
                if destroy is not None:
                    destroy(built_provider_info)
        if can_reuse_training_provider and built_provider_info is None:
            self._provider_state_current = True

        handle = int(result.get("cache_handle", 0) or 0)
        if handle == 0:
            raise RuntimeError("Backend returned an invalid prediction cache handle")
        self._prediction_cache_handle = handle
        self._prediction_cache_signature = signature
        self._prediction_cache_info = {
            "prediction_cache_handle": handle,
            "prediction_cache_method": prediction_method,
            "prediction_cache_variance_method": variance_method,
            "prediction_cache_rank": int(result.get("rank", 0) or 0),
            "prediction_cache_has_love_root": bool(result.get("has_love_root", False)),
            "prediction_cache_prepare_time_s": float(result.get("prepare_time_s", 0.0)),
            "prediction_cache_alpha_time_s": float(result.get("alpha_time_s", 0.0)),
            "prediction_cache_love_root_time_s": float(result.get("love_root_time_s", 0.0)),
            "prediction_cache_alpha_reused_host": bool(result.get("alpha_cache_used", False)),
            "prediction_cache_love_root_reused_host": bool(result.get("love_root_cache_used", False)),
        }
        return dict(self._prediction_cache_info)

    def _fit_mixed(
        self,
        X_full,
        X_cont,
        C,
        y,
        initial_params,
        initial_noise,
        max_iterations,
        learning_rate,
        method_int,
        enable_early_stopping,
        early_stop_patience,
        early_stop_tol,
        verbose,
        cg_params,
        use_fused_kernels=True,
        use_cosine_lr=True,
        use_preconditioner=None,
        progress_adapter=None,
        progress_interval=1,
    ) -> MixedTrainingResult:
        """Train a mixed continuous + categorical GP."""
        _ = use_preconditioner
        init_mean = (
            self._init_mean if self._init_mean is not None else float(np.mean(y))
        )
        cat_kernel_map = {"gd": 0, "cr": 1, "ehh": 2, "hh": 3, "fe": 4}
        sorted_cat_cols = sorted(self.cat_dims.keys())
        cat_levels = [self.cat_dims[col] for col in sorted_cat_cols]

        if isinstance(self.cat_kernel, dict):
            cat_kernel_types = [
                cat_kernel_map[self.cat_kernel[col].lower()] for col in sorted_cat_cols
            ]
        else:
            cat_kernel_const = cat_kernel_map[self.cat_kernel.lower()]
            cat_kernel_types = [cat_kernel_const] * len(sorted_cat_cols)

        if verbose:
            print(f"Training mixed GP with {self.kernel.to_mojo_type()} + categorical")
            print(f"  n={X_cont.shape[0]}, cont_dim={self._cont_dim}")
            print(f"  cat_vars={len(sorted_cat_cols)}, levels={cat_levels}")
            print(f"  init_mean={init_mean:.4f}")

        # Initialize continuous kernel provider
        self._destroy_provider_info()
        engine_params = self.kernel.to_engine_params(
            np.array(initial_params, dtype=np.float32)
        )
        self._provider_info = self._build_provider_info(
            X_cont, engine_params, initial_noise
        )
        register_provider_lease(self._kernel_module, self, self._revoke_provider_info)

        # Prepare categorical specs for engine (use string names, not integer codes)
        _cat_kernel_int_to_str = {0: "gd", 1: "cr", 2: "ehh", 3: "hh", 4: "fe"}
        cat_specs = [
            {"levels": int(lev), "kernel_type": _cat_kernel_int_to_str[int(kt)]}
            for lev, kt in zip(cat_levels, cat_kernel_types)
        ]
        # Cache cat_specs for use in prediction
        self._cat_specs = cat_specs

        cat_init_params = build_default_categorical_raw_params(cat_specs)

        train_args = [
            self._provider_info,
            y.astype(np.float32),
            engine_params,
            float(initial_noise),
            C.astype(np.int32),
            cat_specs,
            cat_init_params,
            max_iterations,
            float(learning_rate),
            int(cg_params["num_probes"]),
            int(cg_params["max_cg_iter"]),
            float(cg_params["cg_tol"]),
            int(cg_params["precond_rank"]),
            bool(verbose),
            int(method_int),
            int(cg_params["precond_method"]),
            bool(enable_early_stopping),
            int(early_stop_patience),
            float(early_stop_tol),
        ]
        if progress_adapter is not None:
            train_args.extend([progress_adapter.callback, int(progress_interval)])
        try:
            result = self._engine.train_mixed(*train_args)
        except Exception as exc:
            if progress_adapter is not None:
                progress_adapter.close_if_needed(failed=True, message=str(exc))
            raise
        if progress_adapter is not None:
            progress_adapter.close_if_needed()

        self._is_trained = True
        self._C_train = C
        self._fitted_mean = float(result.get("mean", 0.0))

        nll_history_raw = result.get("nll_history", None)
        nll_history = (
            np.array(list(nll_history_raw), dtype=np.float32)
            if nll_history_raw is not None
            else None
        )
        cg_iterations_history_raw = result.get("cg_iterations_history", None)
        cg_iterations_history = (
            np.array(list(cg_iterations_history_raw), dtype=np.int32)
            if cg_iterations_history_raw is not None
            else None
        )
        iter_times_raw = result.get("iter_times_ms", None)
        if iter_times_raw is None:
            iter_times_ns_raw = result.get("iter_times_ns", None)
            iter_times_ms = (
                np.array(list(iter_times_ns_raw), dtype=np.float64) / 1e6
                if iter_times_ns_raw is not None
                else None
            )
        else:
            iter_times_ms = np.array(list(iter_times_raw), dtype=np.float64)
        self._training_result = MixedTrainingResult(
            params=self._continuous_param_kernel().from_engine_params(
                np.array(list(result["params"]), dtype=np.float32)
            ),
            cat_params=np.asarray(result["cat_params"], dtype=np.float32),
            noise=float(result["noise"]),
            mean=self._fitted_mean,
            nll=float(result["nll"]),
            iterations=int(result["iterations"]),
            converged=bool(result["converged"]),
            alpha=np.zeros(len(y), dtype=np.float32),  # recomputed in prediction
            nll_history=nll_history,
            cg_iterations_history=cg_iterations_history,
            iter_times_ms=iter_times_ms,
        )
        self._backend_train_info = {
            "training_route": str(
                result.get(
                    "training_route",
                    "materialized" if method_int == 1 else "matrix_free",
                )
            ),
            "materialization_mode": int(result.get("materialization_mode", method_int)),
            "precond_method": int(
                result.get("precond_method", cg_params["precond_method"])
            ),
            "precond_rank": int(result.get("precond_rank", cg_params["precond_rank"])),
            "enable_early_stopping": bool(
                result.get("enable_early_stopping", enable_early_stopping)
            ),
            "early_stop_patience": int(
                result.get("early_stop_patience", early_stop_patience)
            ),
            "early_stop_tol": float(result.get("early_stop_tol", early_stop_tol)),
        }
        if iter_times_ms is not None and len(iter_times_ms) > 0:
            self._backend_train_info["iter_times_ms"] = iter_times_ms.tolist()
        self._maybe_attach_specialization_metadata(self._backend_train_info)
        self._backend_sample_info = None
        return self._training_result

    def predict(
        self,
        X: np.ndarray,
        return_var: bool = False,
        return_std: bool = False,
        variance_method: str = "love",
        method: Optional[str] = None,
        target: str = "latent",
        observation_noise: Optional[np.ndarray] = None,
        observation_noise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise_group_test: Optional[np.ndarray] = None,
        *,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: Optional[float] = None,
        preconditioner_rank: Optional[int] = None,
        max_root_decomposition_size: Optional[int] = None,
        exact_prediction_block_cols: Optional[int] = None,
        progress: Any = None,
        progress_stats: Optional[Any] = None,
    ) -> Union[PredictionResult, Tuple[np.ndarray, np.ndarray]]:
        """Make latent or observed predictions on new data.

        By default returns a PredictionResult with mean, variance, and std.

        Args:
            X: Test data [m, dim]
            return_var: If True, return (mean, variance) tuple.
            return_std: If True, return (mean, std) tuple.
            variance_method: "love" (fast approximate) or "exact" (CG-based).
            method: Prediction route. One of None, "materialized", or "matrix_free".
                None reuses the training route.
            target: ``"latent"`` returns ``p(f_test | y_train)``. ``"observed"``
                returns ``p(y_test | y_train)`` and requires explicit
                ``observation_noise`` for fixed-vector/free learned-noise models.
            observation_noise: Test observation-noise variances for observed
                prediction. Must have shape ``(m,)`` when ``target="observed"``.
            observation_noise_fn: Optional callable mapping test inputs to
                known observation-noise variances. Mutually exclusive with
                ``observation_noise`` and ``noise_group_test``.
            noise_group_test: Test group ids for observed prediction from a
                grouped-noise model. Mutually exclusive with ``observation_noise``.
            exact_prediction_block_cols: Optional exact-variance RHS block size.
                Intended for targeted performance tuning; None uses route defaults.
            progress: Progress reporting control. Use True for the default tqdm
                reporter, ``"auto"`` for tty-only tqdm, a callback, or a
                reporter object with start/update/close methods.
            progress_stats: Optional stat names or callable used by the default
                tqdm reporter.

        Returns:
            Default: PredictionResult with mean, variance, and std.
            If return_var=True: (mean, variance) tuple.
            If return_std=True: (mean, std) tuple.
        """
        if not self._is_trained:
            raise RuntimeError(
                "GP must be trained before prediction. Call fit() first."
            )

        if return_var and return_std:
            raise ValueError("Only one of return_var or return_std may be True.")
        return_full = not return_var and not return_std

        _VALID_VARIANCE_METHODS = ("love", "exact", "mean_only")
        _VALID_METHODS = (None, "materialized", "matrix_free")
        if variance_method not in _VALID_VARIANCE_METHODS:
            raise ValueError(
                f"variance_method must be one of {_VALID_VARIANCE_METHODS}, "
                f"got '{variance_method}'"
            )
        if method not in _VALID_METHODS:
            raise ValueError(
                "method must be one of None, 'materialized', or 'matrix_free', "
                f"got '{method}'"
            )
        if target not in ("latent", "observed"):
            raise ValueError("target must be 'latent' or 'observed'")
        if (
            exact_prediction_block_cols is not None
            and exact_prediction_block_cols <= 0
        ):
            raise ValueError("exact_prediction_block_cols must be positive or None")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.dim:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.dim}")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X_test contains NaN or Inf values")

        surface = surface_for_single_output(self._is_mixed)
        variance_feature = {
            "mean_only": "mean_only",
            "exact": "exact_variance",
            "love": "love_variance",
        }[variance_method]
        check_feature_support(TABLE_PREDICTION, surface, variance_feature, stacklevel=2)

        train_info = self._backend_train_info or {}
        prediction_route = method or train_info.get(
            "training_route", self._training_method or "matrix_free"
        )
        progress_adapter = resolve_progress_adapter(
            progress,
            operation="predict",
            model="single_output",
            route=prediction_route,
            progress_stats=progress_stats,
        )
        progress_total = 4 if target == "observed" else 3
        if progress_adapter is not None:
            base_stats = prediction_progress_stats(
                n_test=X.shape[0],
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
            if self._is_mixed:
                result = self._predict_mixed(
                    X,
                    return_var=return_var,
                    return_std=return_std,
                    variance_method=variance_method,
                )
            else:
                result = self._predict_continuous(
                    X,
                    return_var,
                    return_std,
                    variance_method,
                    method=method,
                    max_cg_iterations=max_cg_iterations,
                    cg_tolerance=cg_tolerance,
                    preconditioner_rank=preconditioner_rank,
                    max_root_decomposition_size=max_root_decomposition_size,
                    exact_prediction_block_cols=exact_prediction_block_cols,
                )

            final_stats = prediction_progress_stats(
                n_test=X.shape[0],
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

            if target == "latent":
                final_result = result
            else:
                if observation_noise_fn is not None:
                    if observation_noise is not None or noise_group_test is not None:
                        raise ValueError(
                            "Pass only one of observation_noise, observation_noise_fn, or noise_group_test"
                        )
                    observation_noise = self._evaluate_observation_noise_fn(
                        observation_noise_fn,
                        X,
                        expected_n=X.shape[0],
                        name="observation_noise_fn",
                    )
                elif observation_noise is None and noise_group_test is None:
                    fn = getattr(self, "_observation_noise_fn", None)
                    if fn is not None:
                        observation_noise = self._evaluate_observation_noise_fn(
                            fn,
                            X,
                            expected_n=X.shape[0],
                            name="stored observation_noise_fn",
                        )
                    elif self._noise_mode == "learned_input_dependent":
                        observation_noise = self._evaluate_learned_noise_function(
                            X,
                            expected_n=X.shape[0],
                        )
                if noise_group_test is not None:
                    if observation_noise is not None:
                        raise ValueError("Pass either observation_noise or noise_group_test, not both")
                    observation_noise = self._observation_noise_from_test_groups(
                        noise_group_test,
                        expected_n=X.shape[0],
                    )
                if progress_adapter is not None:
                    progress_adapter.emit(
                        phase="observed_noise",
                        current=3,
                        total=progress_total,
                        stats=final_stats,
                    )
                final_result = self._add_observation_noise_to_prediction(
                    result,
                    observation_noise,
                    expected_n=X.shape[0],
                    return_full=return_full,
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
                stats=prediction_progress_stats(
                    n_test=X.shape[0],
                    variance_method=variance_method,
                    backend_info=self._backend_predict_info,
                ),
            )
        return final_result

    def predict_latent(self, X: np.ndarray, **kwargs):
        """Predict the latent function ``p(f_test | y_train)``."""
        kwargs.pop("target", None)
        return self.predict(X, target="latent", **kwargs)

    def predict_observed(
        self,
        X: np.ndarray,
        observation_noise: Optional[np.ndarray] = None,
        *,
        observation_noise_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        noise_group_test: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Predict observed responses ``p(y_test | y_train)``.

        The test-time observation noise is explicit by design; MojoGP does not
        silently reuse or average training noise for new points.
        """
        kwargs.pop("target", None)
        return self.predict(
            X,
            target="observed",
            observation_noise=observation_noise,
            observation_noise_fn=observation_noise_fn,
            noise_group_test=noise_group_test,
            **kwargs,
        )

    def _evaluate_observation_noise_fn(
        self,
        observation_noise_fn,
        X: np.ndarray,
        *,
        expected_n: int,
        name: str,
        noise_floor: Optional[float] = None,
    ) -> np.ndarray:
        if not callable(observation_noise_fn):
            raise ValueError(f"{name} must be callable")
        try:
            noise = observation_noise_fn(np.ascontiguousarray(X, dtype=np.float32))
        except Exception as exc:
            raise ValueError(f"{name} failed while evaluating observation noise") from exc
        obs_noise = np.ascontiguousarray(noise, dtype=np.float32)
        if obs_noise.shape != (expected_n,):
            raise ValueError(
                f"{name} must return shape ({expected_n},), got {obs_noise.shape}"
            )
        if np.any(~np.isfinite(obs_noise)):
            raise ValueError(f"{name} returned NaN or Inf values")
        floor = self._noise_floor if noise_floor is None else float(noise_floor)
        if np.any(obs_noise < floor):
            raise ValueError(f"{name} values must be >= noise_floor")
        return obs_noise

    def _evaluate_learned_noise_function(
        self,
        X: np.ndarray,
        *,
        expected_n: int,
    ) -> np.ndarray:
        if self._noise_mode != "learned_input_dependent":
            raise ValueError("learned noise function is only available for learned_input_dependent models")
        if self._noise_function != "linear" or self._noise_function_params is None:
            raise ValueError("learned input-dependent noise parameters are unavailable")
        X_eval = self._apply_dim_permutation(np.ascontiguousarray(X, dtype=np.float32))
        params = np.asarray(self._noise_function_params, dtype=np.float32)
        if params.shape != (X_eval.shape[1] + 1,):
            raise ValueError(
                "learned input-dependent noise parameters do not match the model input dimension"
            )
        raw_noise = params[0] + X_eval @ params[1:]
        obs_noise = (np.logaddexp(0.0, raw_noise.astype(np.float64)) + self._noise_floor).astype(np.float32)
        if obs_noise.shape != (expected_n,):
            raise ValueError(
                f"learned input-dependent noise must have shape ({expected_n},), got {obs_noise.shape}"
            )
        return np.ascontiguousarray(obs_noise, dtype=np.float32)

    def _observation_noise_from_test_groups(
        self,
        noise_group_test,
        *,
        expected_n: int,
    ) -> np.ndarray:
        if self._noise_group_values is None:
            raise ValueError(
                "noise_group_test requires a grouped-noise model with saved group_noise values"
            )
        group_ids = np.ascontiguousarray(noise_group_test, dtype=np.int32)
        if group_ids.shape != (expected_n,):
            raise ValueError(
                f"noise_group_test must have shape ({expected_n},), got {group_ids.shape}"
            )
        if np.any(group_ids < 0):
            raise ValueError("noise_group_test must contain non-negative group ids")
        max_group = int(group_ids.max(initial=-1))
        if max_group >= self._noise_group_values.shape[0]:
            raise ValueError("noise_group_test references an unknown group id")
        return np.ascontiguousarray(self._noise_group_values[group_ids], dtype=np.float32)

    def _add_observation_noise_to_prediction(
        self,
        result,
        observation_noise,
        *,
        expected_n: int,
        return_full: bool,
    ):
        if observation_noise is None:
            raise ValueError(
                "predict_observed()/predict(target='observed') requires observation_noise for test points"
            )
        obs_noise = np.ascontiguousarray(observation_noise, dtype=np.float32)
        if obs_noise.shape != (expected_n,):
            raise ValueError(
                f"observation_noise must have shape ({expected_n},), got {obs_noise.shape}"
            )
        if np.any(~np.isfinite(obs_noise)):
            raise ValueError("observation_noise contains NaN or Inf values")
        if np.any(obs_noise < self._noise_floor):
            raise ValueError("observation_noise values must be >= noise_floor")
        if return_full:
            assert isinstance(result, PredictionResult)
            variance = np.asarray(result.variance, dtype=np.float32) + obs_noise
            return PredictionResult(
                mean=result.mean,
                variance=variance,
                std=np.sqrt(np.maximum(variance, 0)),
            )
        mean, std = result
        variance = np.asarray(std, dtype=np.float32) ** 2 + obs_noise
        return mean, np.sqrt(np.maximum(variance, 0))

    def _predict_continuous(
        self,
        X,
        return_var,
        return_std,
        variance_method,
        method: Optional[str] = None,
        *,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: Optional[float] = None,
        preconditioner_rank: Optional[int] = None,
        max_root_decomposition_size: Optional[int] = None,
        exact_prediction_block_cols: Optional[int] = None,
    ):
        """Predict with continuous-only GP."""
        tr = self._training_result
        assert isinstance(tr, TrainingResult)

        # Always use engine path
        X_test = self._apply_dim_permutation(X)
        mean, variance, engine_result, solver_config, prediction_method = self._predict_engine(
            X_test,
            variance_method,
            method=method,
            max_cg_iterations=max_cg_iterations,
            cg_tolerance=cg_tolerance,
            preconditioner_rank=preconditioner_rank,
            max_root_decomposition_size=max_root_decomposition_size,
            exact_prediction_block_cols=exact_prediction_block_cols,
        )
        train_info = self._backend_train_info or {}
        prediction_uses_cg = variance_method == "exact"
        exact_cross_mode_raw = int(engine_result.get("exact_cross_mode", 0) or 0)
        exact_cross_mode = {
            0: None,
            1: "direct_fill_cross_covariance",
            2: "chunked_cross_matvec_fallback",
        }.get(exact_cross_mode_raw, f"unknown_{exact_cross_mode_raw}")
        self._backend_predict_info = {
            "requested_method": prediction_method,
            "prediction_method": prediction_method,
            "training_route": train_info.get(
                "training_route", self._training_method or "matrix_free"
            ),
            "actual_prediction_route": "predict",
            "actual_variance_route": "predict"
            if variance_method != "mean_only"
            else None,
            "backend_prediction_used": True,
            "backend_variance_used": variance_method != "mean_only",
            "fallback_used": False,
            "variance_method": variance_method,
            "precond_method": train_info.get("precond_method"),
            "precond_rank": int(solver_config["precond_rank"]),
            "max_cg_iterations": int(solver_config["max_cg_iterations"]),
            "cg_tolerance": float(solver_config["cg_tolerance"]),
            "max_root_decomposition_size": int(
                solver_config["max_root_decomposition_size"]
            ),
            "prediction_total_time_s": float(engine_result.get("total_time_s", 0.0)),
            "prediction_alpha_time_s": float(engine_result.get("alpha_time_s", 0.0)),
            "prediction_mean_time_s": float(engine_result.get("mean_time_s", 0.0)),
            "prediction_variance_time_s": float(
                engine_result.get("variance_time_s", 0.0)
            ),
            "prediction_love_root_time_s": float(
                engine_result.get("love_root_time_s", 0.0)
            ),
            "love_alloc_time_s": float(engine_result.get("love_alloc_time_s", 0.0)),
            "love_cross_time_s": float(engine_result.get("love_cross_time_s", 0.0)),
            "love_diag_time_s": float(engine_result.get("love_diag_time_s", 0.0)),
            "love_post_time_s": float(engine_result.get("love_post_time_s", 0.0)),
            "love_cross_strategy": {
                0: "fused",
                1: "materialize_blas",
            }.get(int(engine_result.get("love_cross_strategy", 0) or 0), "unknown"),
            "love_cross_chunk_width": int(
                engine_result.get("love_cross_chunk_width", 0) or 0
            ),
            "prediction_output_copy_time_s": float(
                engine_result.get("output_copy_time_s", 0.0)
            ),
            "prediction_cache_used": bool(
                engine_result.get("prediction_cache_used", False)
            ),
            "prediction_cache_handle": int(
                (self._prediction_cache_info or {}).get("prediction_cache_handle", 0)
                or 0
            ),
            "prediction_cache_prepare_time_s": float(
                (self._prediction_cache_info or {}).get(
                    "prediction_cache_prepare_time_s", 0.0
                )
            ),
            "prediction_cache_alpha_time_s": float(
                (self._prediction_cache_info or {}).get(
                    "prediction_cache_alpha_time_s", 0.0
                )
            ),
            "prediction_cache_love_root_time_s": float(
                (self._prediction_cache_info or {}).get(
                    "prediction_cache_love_root_time_s", 0.0
                )
            ),
            "prediction_cache_has_love_root": bool(
                (self._prediction_cache_info or {}).get(
                    "prediction_cache_has_love_root", False
                )
            ),
            "prediction_cache_rank": int(
                (self._prediction_cache_info or {}).get("prediction_cache_rank", 0)
                or 0
            ),
            "exact_block_cols": int(engine_result.get("exact_block_cols", 0) or 0),
            "exact_block_cols_requested": int(exact_prediction_block_cols or 0),
            "exact_cross_mode": exact_cross_mode,
            "exact_cg_block_count": int(
                engine_result.get("exact_cg_block_count", 0) or 0
            ),
            "exact_cg_total_iterations": int(
                engine_result.get("exact_cg_total_iterations", 0) or 0
            ),
            "exact_cg_max_iterations": int(
                engine_result.get("exact_cg_max_iterations", 0) or 0
            ),
            "exact_alloc_time_s": float(engine_result.get("exact_alloc_time_s", 0.0)),
            "exact_cross_time_s": float(engine_result.get("exact_cross_time_s", 0.0)),
            "exact_diag_time_s": float(engine_result.get("exact_diag_time_s", 0.0)),
            "exact_solve_time_s": float(engine_result.get("exact_solve_time_s", 0.0)),
            "exact_post_time_s": float(engine_result.get("exact_post_time_s", 0.0)),
            "alpha_cache_used": bool(engine_result.get("alpha_cache_used", False)),
            "love_root_cache_used": bool(
                engine_result.get("love_root_cache_used", False)
            ),
            "provider_state_update_skipped": bool(
                engine_result.get("provider_state_update_skipped", False)
            ),
            "telemetry_quality": (
                "observed" if prediction_uses_cg else "not_applicable"
            ),
            "configured_for_cg": prediction_uses_cg,
            "observed_cg_calls": prediction_uses_cg,
        }
        self._maybe_attach_specialization_metadata(self._backend_predict_info)
        std = np.sqrt(np.maximum(variance, 0))
        if return_var:
            return mean, variance
        if return_std:
            return mean, std
        return PredictionResult(mean=mean, variance=variance, std=std)

    def _predict_mixed(self, X, return_var=False, return_std=False, variance_method="love"):
        """Predict with mixed continuous + categorical GP."""
        tr = self._training_result
        assert isinstance(tr, MixedTrainingResult)

        X_test_cont, C_test = self._split_data(X)
        X_test_cont = self._apply_dim_permutation(X_test_cont)

        # Split training data the same way
        X_train_cont, C_train = self._split_data(self._X_train)
        X_train_cont = self._apply_dim_permutation(X_train_cont)
        y_train = self._y_train.astype(np.float32)

        _variance_method_map = {"mean_only": 0, "love": 1, "exact": 2}
        variance_method_int = _variance_method_map.get(variance_method, 1)

        cont_params = self._continuous_param_kernel().to_engine_params(
            np.array(list(tr.params), dtype=np.float32)
        )
        cat_params = categorical_prediction_params(
            self._cat_specs,
            np.array(list(tr.cat_params), dtype=np.float32),
        )
        predict_rank = _prediction_lanczos_rank(
            rank_hint=None,
            training_method=getattr(self, "_training_method", None),
            is_mixed=True,
            is_ard=self.ard,
        )

        provider_info = self._provider_info
        built_provider_info = None
        if provider_info is None:
            provider_info = self._build_provider_info(
                X_train_cont, cont_params, tr.noise
            )
            built_provider_info = provider_info
        try:
            pred = self._engine.predict_mixed(
                provider_info,
                y_train,
                X_test_cont.astype(np.float32),
                C_train.astype(np.int32),
                C_test.astype(np.int32),
                cont_params,
                float(tr.noise),
                float(tr.mean),
                cat_params,
                self._cat_specs,
                variance_method_int,
                100,
                1e-2,
                10,
                predict_rank,
                1
                if getattr(self, "_training_method", "matrix_free") == "materialized"
                else 0,
            )
        finally:
            if built_provider_info is not None:
                destroy = getattr(self._kernel_module, "destroy_provider", None)
                if destroy is not None:
                    destroy(built_provider_info)

        mean = np.asarray(pred["mean"], dtype=np.float32)
        variance = np.maximum(np.asarray(pred["variance"], dtype=np.float32), 0)
        train_info = self._backend_train_info or {}
        prediction_uses_cg = variance_method == "exact"
        self._backend_predict_info = {
            "requested_method": self._training_method or "matrix_free",
            "training_route": train_info.get(
                "training_route", self._training_method or "matrix_free"
            ),
            "actual_prediction_route": "predict_mixed",
            "actual_variance_route": (
                "predict_mixed" if variance_method != "mean_only" else None
            ),
            "backend_prediction_used": True,
            "backend_variance_used": variance_method != "mean_only",
            "fallback_used": False,
            "variance_method": variance_method,
            "precond_method": train_info.get("precond_method"),
            "precond_rank": train_info.get("precond_rank"),
            "telemetry_quality": (
                "observed" if prediction_uses_cg else "not_applicable"
            ),
            "configured_for_cg": prediction_uses_cg,
            "observed_cg_calls": prediction_uses_cg,
        }
        self._maybe_attach_specialization_metadata(self._backend_predict_info)
        std = np.sqrt(variance)
        if return_var:
            return mean, variance
        if return_std:
            return mean, std
        return PredictionResult(mean=mean, variance=variance, std=std)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute R^2 score on test data."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before scoring")
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)
        mean, _ = self.predict(X_test, return_std=True)
        ss_res = np.sum((y_test - mean) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def get_learned_params(self) -> dict:
        """Get learned parameters as a named dictionary."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before getting params")
        tr = self._training_result
        param_names = self.kernel.get_param_names()
        params = {}
        for i, name in enumerate(param_names):
            if i < len(tr.params):
                params[name] = float(tr.params[i])
        params["noise"] = float(tr.noise)
        if self._noise_mode == "learned_vector" and self._observation_noise_train is not None:
            params["observation_noise_train"] = np.asarray(
                self._observation_noise_train, dtype=np.float32
            ).copy()
        if self._noise_mode == "learned_grouped" and self._noise_group_values is not None:
            params["group_noise"] = np.asarray(self._noise_group_values, dtype=np.float32).copy()
        if self._noise_mode == "learned_input_dependent" and self._noise_function_params is not None:
            params["noise_function"] = self._noise_function
            params["noise_function_params"] = np.asarray(
                self._noise_function_params, dtype=np.float32
            ).copy()
            if self._observation_noise_train is not None:
                params["observation_noise_train"] = np.asarray(
                    self._observation_noise_train, dtype=np.float32
                ).copy()
        params["mean"] = float(tr.mean)
        return params

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
            X_test: Test points [m, dim]
            n_samples: Number of posterior samples to draw
            method: Sampling method:
                - 'diagonal' (default): Independent samples using predictive std.
                  Fast, O(m), but ignores correlations between test points.
                - 'pathwise': Approximate correlated posterior samples using
                  pathwise conditioning / Matheron's rule. The prior sampler is
                  approximate (RFF), but the correction uses the live JIT backend
                  provider route in both matrix-free and materialized modes.
                  Supports current continuous kernel trees except polynomial,
                  plus supported mixed continuous-categorical kernel trees.
            n_rff_features: Number of random Fourier features for the pathwise
                prior sampler. Default 1024. Only used when method is 'pathwise'.
            rng: Optional numpy random Generator for reproducibility. If None, a new
                generator is created.

        Returns:
            samples: Array of shape [n_samples, m]
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before sampling.")
        surface = surface_for_single_output(self._is_mixed)
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
        training_route = train_info.get(
            "training_route", self._training_method or "matrix_free"
        )

        if method == "pathwise":
            check_feature_support(
                TABLE_SAMPLING, surface, "pathwise_sampling", stacklevel=2
            )
            if kernel_tree_contains_kernel_name(self.kernel, "POLYNOMIAL"):
                check_feature_support(
                    TABLE_SAMPLING, surface, "polynomial_pathwise", stacklevel=2
                )
            samples = self._sample_posterior_pathwise(
                X_test, n_samples, n_rff_features=n_rff_features, rng=rng
            )
            self._backend_sample_info = {
                "requested_method": requested_method,
                "actual_sampling_method": "pathwise",
                "actual_sampling_route": "provider_pathwise",
                "backend_sampling_used": True,
                "backend_correction_used": True,
                "backend_correction_route": (
                    "predict_mixed" if self._is_mixed else "predict"
                ),
                "training_route": training_route,
                "prior_sampler_family": "shared_feature_map",
                "n_rff_features": int(n_rff_features),
            }
            self._maybe_attach_specialization_metadata(self._backend_sample_info)
            return samples

        check_feature_support(
            TABLE_SAMPLING, surface, "diagonal_sampling", stacklevel=2
        )
        pred = self.predict(X_test)
        assert isinstance(pred, PredictionResult)

        z = rng.standard_normal((n_samples, len(pred.mean)))
        samples = pred.mean[np.newaxis, :] + pred.std[np.newaxis, :] * z
        self._backend_sample_info = {
            "requested_method": requested_method,
            "actual_sampling_method": "diagonal",
            "actual_sampling_route": "diagonal_from_predictive_std",
            "backend_sampling_used": True,
            "backend_correction_used": False,
            "training_route": training_route,
        }
        self._maybe_attach_specialization_metadata(self._backend_sample_info)
        return samples.astype(np.float32)

    def _sample_posterior_cholesky(
        self,
        X_test: np.ndarray,
        n_samples: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Correlated posterior samples via Cholesky decomposition.

        Computes full posterior covariance K_post = K(X*,X*) - K(X*,X) @ (K+σ²I)^{-1} @ K(X,X*)
        then samples: f* = mean + L @ z where L = cholesky(K_post).
        """
        tr = self._training_result
        X_train = self._apply_dim_permutation(self._X_train)
        X_test_p = self._apply_dim_permutation(
            np.ascontiguousarray(X_test, dtype=np.float32)
        )

        params = np.array(tr.params, dtype=np.float32)
        noise = float(tr.noise)
        mean_val = float(tr.mean)
        n = X_train.shape[0]
        m = X_test_p.shape[0]

        # Build kernel matrices using Python-side KernelNode.evaluate()
        K_train = self.kernel.evaluate(X_train, X_train, params=params)  # [n, n]
        K_cross = self.kernel.evaluate(X_test_p, X_train, params=params)  # [m, n]
        K_test = self.kernel.evaluate(X_test_p, X_test_p, params=params)  # [m, m]

        # Regularize training covariance
        K_train_reg = K_train + noise * np.eye(n, dtype=np.float32)

        # Solve (K_train + σ²I) @ A = K_cross^T  →  A [n, m]
        A = np.linalg.solve(
            K_train_reg.astype(np.float64), K_cross.T.astype(np.float64)
        ).astype(np.float32)

        # Posterior mean: K_cross @ (K_train + σ²I)^{-1} @ (y - mean) + mean
        y_centered = self._y_train.astype(np.float32) - mean_val
        alpha = np.linalg.solve(
            K_train_reg.astype(np.float64), y_centered.astype(np.float64)
        ).astype(np.float32)
        post_mean = K_cross @ alpha + mean_val  # [m]

        # Posterior covariance: K** - K*X @ A
        K_post = K_test - K_cross @ A  # [m, m]

        # Symmetrize and add jitter for numerical stability
        K_post = 0.5 * (K_post + K_post.T)
        jitter = float(np.abs(np.diag(K_post)).mean()) * 1e-6 + 1e-8
        K_post += jitter * np.eye(m, dtype=np.float32)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K_post.astype(np.float64)).astype(
                np.float32
            )  # [m, m]
        except np.linalg.LinAlgError:
            # Increase jitter and retry
            K_post += 1e-4 * np.eye(m, dtype=np.float32)
            L = np.linalg.cholesky(K_post.astype(np.float64)).astype(np.float32)

        # Draw samples: [n_samples, m]
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal((m, n_samples)).astype(np.float32)  # [m, n_samples]
        samples = post_mean[np.newaxis, :] + (L @ z).T  # [n_samples, m]
        return samples.astype(np.float32)

    def _sample_posterior_pathwise(
        self,
        X_test: np.ndarray,
        n_samples: int,
        n_rff_features: int = 1024,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Approximate correlated posterior samples via backend pathwise correction."""
        if rng is None:
            rng = np.random.default_rng()

        tr = self._training_result
        assert isinstance(tr, (TrainingResult, MixedTrainingResult))

        self._ensure_compiled()
        if self._engine is None:
            raise RuntimeError("Backend engine is unavailable for pathwise sampling.")

        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        noise = float(tr.noise)
        mean_val = float(tr.mean)
        y_centered = self._y_train.astype(np.float32) - np.float32(mean_val)

        if self._is_mixed:
            assert isinstance(tr, MixedTrainingResult)
            X_train_cont, C_train = self._split_data(self._X_train)
            X_test_cont, C_test = self._split_data(X_test)
            X_train_cont = self._apply_dim_permutation(X_train_cont)
            X_test_cont = self._apply_dim_permutation(X_test_cont)
            cont_public_params = np.ascontiguousarray(tr.params, dtype=np.float32)
            backend_params = cont_public_params
            cat_params = np.ascontiguousarray(
                categorical_prediction_params(
                    self._cat_specs,
                    np.asarray(tr.cat_params, dtype=np.float32),
                ),
                dtype=np.float32,
            )
            cat_col_map = {col: idx for idx, col in enumerate(self._cat_col_indices)}
            feature_map = build_pathwise_feature_map(
                self.kernel,
                cont_public_params,
                input_dim=X_train_cont.shape[1],
                n_features=n_rff_features,
                rng=rng,
                cat_params=cat_params,
                cat_col_map=cat_col_map,
            )
        else:
            X_train_cont = self._apply_dim_permutation(self._X_train)
            X_test_cont = self._apply_dim_permutation(X_test)
            C_train = None
            C_test = None
            cont_public_params = np.ascontiguousarray(tr.params, dtype=np.float32)
            backend_params = self.kernel.to_engine_params(cont_public_params)
            feature_map = build_pathwise_feature_map(
                self.kernel,
                cont_public_params,
                input_dim=X_train_cont.shape[1],
                n_features=n_rff_features,
                rng=rng,
            )

        n = X_train_cont.shape[0]
        m = X_test_cont.shape[0]
        samples = np.empty((n_samples, m), dtype=np.float32)
        chunk_size = min(max(_DEFAULT_PATHWISE_TEST_CHUNK_SIZE, 1), max(m, 1))

        noise_scale = np.float32(np.sqrt(max(noise, 0.0)))

        predict_rank = _prediction_lanczos_rank(
            getattr(tr, "lanczos_rank", 0),
            getattr(self, "_training_method", None),
            is_mixed=self._is_mixed,
            is_ard=self.ard,
        )

        provider_info = self._provider_info
        built_provider_info = None
        if provider_info is None:
            provider_info = self._build_provider_info(
                X_train_cont, backend_params, noise
            )
            built_provider_info = provider_info
        provider_state_current = (
            built_provider_info is not None
            or bool(getattr(self, "_provider_state_current", False))
        )
        try:
            for sample_idx in range(n_samples):
                weights = build_feature_weights(feature_map, 1, rng)
                prior_train = sample_prior_values(
                    feature_map,
                    X_train_cont,
                    C_train,
                    weights,
                )[0]
                obs_prior = prior_train + noise_scale * rng.standard_normal(n).astype(
                    np.float32
                )
                residual = np.ascontiguousarray(
                    y_centered - obs_prior,
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
                    prior_chunk = sample_prior_values(
                        feature_map, X_chunk, C_chunk, weights
                    )[0]
                    if self._is_mixed:
                        result = self._engine.predict_mixed(
                            provider_info,
                            residual,
                            X_chunk,
                            C_train.astype(np.int32),
                            C_chunk.astype(np.int32),
                            backend_params,
                            noise,
                            0.0,
                            cat_params,
                            self._cat_specs,
                            0,
                            100,
                            1e-2,
                            10,
                            predict_rank,
                            1
                            if getattr(self, "_training_method", "matrix_free")
                            == "materialized"
                            else 0,
                        )
                    else:
                        result = self._engine.predict(
                            provider_info,
                            residual,
                            X_chunk,
                            backend_params,
                            noise,
                            0.0,
                            0,
                            100,
                            1e-2,
                            10,
                            predict_rank,
                            bool(provider_state_current),
                        )
                    correction = np.asarray(result["mean"], dtype=np.float32)
                    if built_provider_info is None:
                        provider_state_current = True
                        self._provider_state_current = True
                    samples[sample_idx, start:end] = (
                        np.float32(mean_val) + prior_chunk + correction
                    )
        finally:
            if built_provider_info is not None:
                destroy = getattr(self._kernel_module, "destroy_provider", None)
                if destroy is not None:
                    destroy(built_provider_info)

        return samples.astype(np.float32)

    def _rff_sample_prior(
        self,
        X: np.ndarray,
        params: np.ndarray,
        n_samples: int,
        n_features: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample from the GP prior using Random Fourier Features (RFF).

        Approximates f ~ GP(0, k) using Bochner's theorem:
            f(x) ≈ sqrt(2/D) * cos(W^T x + b) @ w
        where W ~ spectral distribution of k, b ~ Uniform(0, 2π), w ~ N(0, I).

        Args:
            X: Input points [n, d]
            params: Kernel hyperparameters
            n_samples: Number of prior samples
            n_features: Number of RFF features D
            rng: Random generator

        Returns:
            prior_samples: [n_samples, n]
        """
        from .kernel import KernelType

        n, d = X.shape

        # Detect stationary kernel type from the root node
        root = self._original_kernel
        kt = root.kernel_type

        # Extract lengthscale(s) and outputscale from params
        # Standard param layout: [outputscale, lengthscale] or [outputscale, ls_0, ..., ls_{d-1}]
        outputscale = float(params[0]) if len(params) > 0 else 1.0

        if self.ard and len(params) >= d + 1:
            lengthscales = params[1 : d + 1].astype(np.float64)  # [d]
        elif len(params) >= 2:
            lengthscales = np.full(d, float(params[1]), dtype=np.float64)
        else:
            lengthscales = np.ones(d, dtype=np.float64)

        # Scale X by inverse lengthscales: X_scaled [n, d]
        X_scaled = X.astype(np.float64) / lengthscales[np.newaxis, :]

        # Sample spectral frequencies W ~ p(w) according to kernel type
        STATIONARY_TYPES = {
            KernelType.RBF,
            KernelType.MATERN12,
            KernelType.MATERN32,
            KernelType.MATERN52,
            KernelType.RQ,
        }
        if kt not in STATIONARY_TYPES:
            raise ValueError(
                f"Kernel type {kt} is not stationary. RFF prior sampling requires "
                "a stationary kernel (RBF, Matern12/32/52, RQ). "
                "Use method='cholesky' instead."
            )

        if kt == KernelType.RBF:
            # Spectral density: N(0, I)
            W = rng.standard_normal((n_features, d))  # [D, d]
        elif kt == KernelType.MATERN12:
            # Spectral density: Student-t with nu=0.5 → df=1 (Cauchy)
            W = rng.standard_t(df=1, size=(n_features, d))
        elif kt == KernelType.MATERN32:
            # Spectral density: Student-t with nu=1.5 → df=3
            W = rng.standard_t(df=3, size=(n_features, d))
        elif kt == KernelType.MATERN52:
            # Spectral density: Student-t with nu=2.5 → df=5
            W = rng.standard_t(df=5, size=(n_features, d))
        elif kt == KernelType.RQ:
            # Spectral density: mixture of Gaussians (scale mixture)
            # RQ is k(r) = (1 + r²/(2α))^{-α}. Its spectral density is a
            # Student-t with 2α degrees of freedom. Extract alpha from params.
            # RQ param layout: [outputscale, lengthscale, alpha]
            rq_alpha = float(params[2]) if len(params) >= 3 else 1.0
            df = max(2.0 * rq_alpha, 0.5)
            W = rng.standard_t(df=df, size=(n_features, d))
        else:
            W = rng.standard_normal((n_features, d))

        # Random phase offsets b ~ Uniform(0, 2π) [D]
        b = rng.uniform(0, 2 * np.pi, size=n_features)

        # Feature map: Z = sqrt(2/D) * cos(X_scaled @ W^T + b)  [n, D]
        Z = np.sqrt(2.0 / n_features) * np.cos(
            X_scaled @ W.T + b[np.newaxis, :]
        )  # [n, D]

        # Prior weights w ~ N(0, outputscale * I)  [n_samples, D]
        w = rng.standard_normal((n_samples, n_features)) * np.sqrt(outputscale)

        # Prior samples: [n_samples, n]
        return (w @ Z.T).astype(np.float32)

    def _sample_posterior_matheron(
        self,
        X_test: np.ndarray,
        n_samples: int,
        n_rff_features: int = 1024,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Correlated posterior samples via Matheron's rule (decoupled sampling).

        Matheron's rule:
            f_post(x*) = f_prior(x*) + K(x*, X) @ (K + σ²I)^{-1} @ (y - f_prior(X))

        This decouples the prior sample from the posterior correction, allowing
        O(n) memory and scaling to large datasets via CG. The prior sample is
        drawn using Random Fourier Features (RFF) which approximates the GP prior
        with n_rff_features Fourier basis functions.

        Ref: Wilson et al. (2020) "Efficiently Sampling Functions from Gaussian
        Process Posteriors" https://arxiv.org/abs/2002.09309

        Args:
            X_test: Test points [m, dim]
            n_samples: Number of posterior samples
            n_rff_features: Number of RFF features for prior approximation
            rng: Random generator for reproducibility

        Returns:
            samples: [n_samples, m]
        """
        from .kernel import KernelType

        if rng is None:
            rng = np.random.default_rng()

        tr = self._training_result
        params = np.array(tr.params, dtype=np.float32)
        noise = float(tr.noise)
        mean_val = float(tr.mean)

        X_train = self._apply_dim_permutation(self._X_train)
        X_test_p = self._apply_dim_permutation(
            np.ascontiguousarray(X_test, dtype=np.float32)
        )

        n = X_train.shape[0]
        m = X_test_p.shape[0]

        # Check if kernel is stationary — fall back to Cholesky if not
        STATIONARY_TYPES = {
            KernelType.RBF,
            KernelType.MATERN12,
            KernelType.MATERN32,
            KernelType.MATERN52,
            KernelType.RQ,
        }
        root = self._original_kernel
        if root.kernel_type not in STATIONARY_TYPES:
            if self.verbose:
                print(
                    f"[MojoGP] Matheron sampling: kernel {root.kernel_type} is not "
                    "stationary. Falling back to Cholesky sampling."
                )
            return self._sample_posterior_cholesky(X_test_p, n_samples, rng=rng)

        # Stack train + test for joint prior evaluation
        X_all = np.vstack([X_train, X_test_p])  # [n+m, d]

        # Pre-compute kernel matrices (shared across all samples)
        K_train = self.kernel.evaluate(X_train, X_train, params=params)  # [n, n]
        K_cross = self.kernel.evaluate(X_test_p, X_train, params=params)  # [m, n]
        K_train_reg = K_train + noise * np.eye(n, dtype=np.float32)

        # Pre-solve the cholesky factor for (K + σ²I) once — reuse across samples
        K_train_reg_d = K_train_reg.astype(np.float64)
        # Factor using Cholesky for efficient repeated solves
        try:
            L_train = np.linalg.cholesky(K_train_reg_d)
        except np.linalg.LinAlgError:
            jitter = float(np.abs(np.diag(K_train_reg_d)).mean()) * 1e-6 + 1e-8
            K_train_reg_d += jitter * np.eye(n, dtype=np.float64)
            L_train = np.linalg.cholesky(K_train_reg_d)

        y_centered = self._y_train.astype(np.float64) - mean_val

        samples_list = []
        for _ in range(n_samples):
            # 1. Draw one prior sample on all points [1, n+m]
            f_prior_all = self._rff_sample_prior(
                X_all, params, n_samples=1, n_features=n_rff_features, rng=rng
            )  # [1, n+m]
            f_prior_train = f_prior_all[0, :n].astype(np.float64)  # [n]
            f_prior_test = f_prior_all[0, n:]  # [m]

            # 2. Residual: r = y - mean - f_prior(X_train)  [n]
            r = y_centered - f_prior_train  # [n]

            # 3. Solve (K + σ²I) @ alpha = r  →  alpha via pre-factored Cholesky
            #    alpha = L^{-T} @ (L^{-1} @ r)
            alpha = np.linalg.solve(L_train.T, np.linalg.solve(L_train, r))  # [n]

            # 4. Correction: K(x*, X) @ alpha  [m]
            correction = K_cross.astype(np.float64) @ alpha  # [m]

            # 5. Matheron formula: f_post(x*) = mean + f_prior(x*) + correction
            f_post = mean_val + f_prior_test.astype(np.float64) + correction  # [m]
            samples_list.append(f_post)

        return np.stack(samples_list, axis=0).astype(np.float32)  # [n_samples, m]

    def log_marginal_likelihood(self) -> float:
        """Return the log marginal likelihood at the learned hyperparameters."""
        if not self._is_trained:
            raise RuntimeError("GP must be trained first.")
        return -float(self._training_result.nll)

    def save(self, path: str) -> None:
        """Save the trained GP model to disk.

        Saves kernel config as JSON and arrays as .npz.
        """
        if not self._is_trained:
            raise RuntimeError("GP must be trained before saving.")

        path = str(path)
        if path.endswith(".json") or path.endswith(".npz"):
            path = path.rsplit(".", 1)[0]

        tr = self._training_result

        config = {
            "schema_version": _MODEL_SCHEMA_VERSION,
            "mojogp_version": __version__,
            "wrapper": "SingleOutputGP",
            "dim": self.dim,
            "ard": self.ard,
            "verbose": self.verbose,
            "cat_dims": {str(k): v for k, v in self.cat_dims.items()},
            "cat_kernel": self.cat_kernel,
            "kernel_mojo_type": self.kernel.to_mojo_type(),
            "kernel_tree": self._original_kernel.to_dict(),
            "is_mixed": self._is_mixed,
            "training_method": self._training_method,
            "engine_mean": self._engine_predict_mean,
            "backend_train_info": self._backend_train_info,
            "cached_love_method": self._cached_love_method,
            "noise_mode": self._noise_mode,
            "noise_floor": self._noise_floor,
            "noise_regularization": self._noise_regularization,
            "noise_function": self._noise_function,
            "provider_noise_mode_int": self._provider_noise_mode_int,
            "has_observation_noise_train": self._observation_noise_train is not None,
            "has_observation_noise_fn": self._observation_noise_fn is not None,
            "has_noise_group_train": self._noise_group_train is not None,
            "has_group_noise": self._noise_group_values is not None,
        }
        if self._specialization_decision is not None:
            config["specialization"] = self._specialization_decision.to_dict()
        if self._backend_train_info is not None:
            config["backend_train_info"] = self._backend_train_info

        if isinstance(tr, TrainingResult):
            config["result_type"] = "TrainingResult"
            config["noise"] = float(tr.noise)
            config["mean"] = float(tr.mean)
            config["nll"] = float(tr.nll)
            config["iterations"] = int(tr.iterations)
            config["converged"] = bool(tr.converged)
            config["lanczos_rank"] = int(tr.lanczos_rank)
        elif isinstance(tr, MixedTrainingResult):
            config["result_type"] = "MixedTrainingResult"
            config["noise"] = float(tr.noise)
            config["mean"] = float(tr.mean)
            config["nll"] = float(tr.nll)
            config["iterations"] = int(tr.iterations)
            config["converged"] = bool(tr.converged)

        with open(f"{path}_config.json", "w") as f:
            json.dump(config, f, indent=2)

        arrays = {"X_train": self._X_train, "y_train": self._y_train}
        if self._observation_noise_train is not None:
            arrays["observation_noise_train"] = self._observation_noise_train
        if self._noise_group_train is not None:
            arrays["noise_group_train"] = self._noise_group_train
        if self._noise_group_values is not None:
            arrays["group_noise"] = self._noise_group_values
        if self._noise_function_params is not None:
            arrays["noise_function_params"] = self._noise_function_params
        if isinstance(tr, TrainingResult):
            arrays["params"] = tr.params
        elif isinstance(tr, MixedTrainingResult):
            arrays["params"] = tr.params
            arrays["cat_params"] = tr.cat_params
            arrays["alpha"] = tr.alpha
            if self._C_train is not None:
                arrays["C_train"] = self._C_train
        if hasattr(self, "_dim_permutation") and self._dim_permutation is not None:
            arrays["dim_permutation"] = np.array(self._dim_permutation, dtype=np.int32)

        np.savez(f"{path}_arrays.npz", **arrays)

    @classmethod
    def load(cls, path: str, kernel: Optional[KernelNode] = None) -> "SingleOutputGP":
        """Load a saved GP model from disk.

        Args:
            path: File path (without extension)
            kernel: The kernel tree. If None, reconstructed from saved config.
        """
        path = str(path)
        if path.endswith(".json") or path.endswith(".npz"):
            path = path.rsplit(".", 1)[0]

        with open(f"{path}_config.json", "r") as f:
            config = json.load(f)

        wrapper = config.get("wrapper")
        if wrapper != "SingleOutputGP":
            raise ValueError(
                "Saved model is not a SingleOutputGP artifact. "
                f"Expected wrapper='SingleOutputGP', got {wrapper!r}."
            )
        schema_version = int(config.get("schema_version", 0))
        if schema_version != _MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported SingleOutputGP schema_version={schema_version}; "
                f"expected {_MODEL_SCHEMA_VERSION}."
            )

        arrays = np.load(f"{path}_arrays.npz", allow_pickle=False)

        # Reconstruct kernel from saved tree if not provided
        if kernel is None:
            if "kernel_tree" not in config:
                raise ValueError(
                    "SingleOutputGP artifact is missing required kernel_tree."
                )
            kernel = KernelNode.from_dict(config["kernel_tree"])

        X_train = arrays["X_train"]
        y_train = arrays["y_train"]

        cat_dims = {int(k): v for k, v in config.get("cat_dims", {}).items()}

        gp = cls(
            kernel=kernel,
            init_mean=config.get("init_mean"),
            verbose=config.get("verbose", False),
        )
        # Restore categorical state from saved config
        gp.cat_dims = cat_dims
        gp.cat_kernel = config.get("cat_kernel", "ehh")
        gp._training_method = config.get("training_method")
        gp._engine_predict_mean = config.get("engine_mean")
        gp._backend_train_info = config.get("backend_train_info")
        gp._cached_love_method = config.get("cached_love_method")
        gp._noise_mode = config.get("noise_mode", "scalar")
        gp._noise_floor = float(config.get("noise_floor", 1e-6))
        gp._noise_regularization = float(config.get("noise_regularization", 0.0))
        gp._noise_function = config.get("noise_function")
        gp._provider_noise_mode_int = config.get("provider_noise_mode_int")
        gp._backend_train_info = config.get("backend_train_info")
        gp._observation_noise_fn = None
        gp._specialization_request = SpecializationRequest.disabled()
        gp._specialization_decision = None
        # Restore data (bypass fit() since we're restoring trained state)
        gp._X_train = X_train
        gp._y_train = y_train
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
        gp._noise_function_params = (
            np.ascontiguousarray(arrays["noise_function_params"], dtype=np.float32)
            if "noise_function_params" in arrays
            else None
        )
        gp.dim = X_train.shape[1]
        gp._cat_col_indices = sorted(gp.cat_dims.keys())
        gp._cont_dim = gp.dim - len(gp._cat_col_indices)
        if gp._kernel_pre_ard.has_categorical():
            from .kernel import analyze_kernel_tree

            gp._analysis = analyze_kernel_tree(gp._kernel_pre_ard, gp.dim)
            base_kernel = gp._analysis.structured_kernel
            if gp._is_mixed:
                cont_cols = [d for d in range(gp.dim) if d not in gp._cat_col_indices]
                dim_map = {orig: idx for idx, orig in enumerate(cont_cols)}
                base_kernel = gp._remap_kernel_active_dims(base_kernel, dim_map)
        else:
            gp._analysis = None
            base_kernel = gp._kernel_pre_ard

        if gp.ard:
            gp.kernel = make_ard_kernel(base_kernel, gp._cont_dim)
        else:
            gp.kernel = base_kernel

        # Restore dimension permutation for active_dims
        gp._dim_permutation = None
        if "dim_permutation" in arrays:
            gp._dim_permutation = arrays["dim_permutation"].tolist()
        # Resolve active_dims dim ranges if the kernel has them
        if gp.kernel.has_active_dims():
            from .kernel import compute_dim_permutation

            compute_dim_permutation(gp.kernel, gp._cont_dim)

        # Restore training state
        gp._is_trained = True
        result_type = config.get("result_type", "TrainingResult")

        if result_type == "TrainingResult":
            gp._training_result = TrainingResult(
                params=arrays["params"],
                noise=config["noise"],
                mean=config.get("mean", 0.0),
                nll=config["nll"],
                iterations=config["iterations"],
                converged=config["converged"],
                lanczos_root=None,
                lanczos_rank=config["lanczos_rank"],
            )
        elif result_type == "MixedTrainingResult":
            gp._training_result = MixedTrainingResult(
                params=arrays["params"],
                cat_params=arrays["cat_params"],
                noise=config["noise"],
                mean=config.get("mean", 0.0),
                nll=config["nll"],
                iterations=config["iterations"],
                converged=config["converged"],
                alpha=arrays["alpha"],
            )
            if "C_train" in arrays:
                gp._C_train = arrays["C_train"]

        # Restore _fitted_mean from the training result (needed for prediction)
        gp._fitted_mean = gp._training_result.mean

        gp._ensure_compiled()

        # Rebuild lightweight categorical metadata only; provider handles are
        # recreated lazily for predict()/sample() to avoid persistent GPU usage.
        if isinstance(gp._training_result, MixedTrainingResult):
            cat_kernel_map = {"gd": 0, "cr": 1, "ehh": 2, "hh": 3, "fe": 4}
            _cat_kernel_int_to_str = {0: "gd", 1: "cr", 2: "ehh", 3: "hh", 4: "fe"}
            sorted_cat_cols = sorted(gp.cat_dims.keys())
            cat_levels = [gp.cat_dims[col] for col in sorted_cat_cols]
            if isinstance(gp.cat_kernel, dict):
                cat_kernel_types = [
                    cat_kernel_map[gp.cat_kernel[col].lower()]
                    for col in sorted_cat_cols
                ]
            else:
                cat_kernel_const = cat_kernel_map[gp.cat_kernel.lower()]
                cat_kernel_types = [cat_kernel_const] * len(sorted_cat_cols)
            gp._cat_specs = [
                {
                    "levels": int(lev),
                    "kernel_type": _cat_kernel_int_to_str[int(kt)],
                }
                for lev, kt in zip(cat_levels, cat_kernel_types)
            ]

        gp._backend_sample_info = None

        return gp

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "untrained"
        parts = [f"kernel={self.kernel!r}"]
        if self._X_train is not None:
            parts.append(f"n={self._X_train.shape[0]}")
        if self.dim is not None:
            parts.append(f"dim={self.dim}")
        if self._is_mixed:
            parts.append(f"cat_dims={self.cat_dims}")
        if self.ard:
            parts.append("ard=True")
        parts.append(status)
        return f"SingleOutputGP({', '.join(parts)})"


def fit_gp(
    X: np.ndarray,
    y: np.ndarray,
    kernel: Optional[KernelNode] = None,
    **kwargs: Any,
) -> SingleOutputGP:
    """Convenience function to create and fit a GP in one call.

    Args:
        X: Training data [n, dim]
        y: Training targets [n]
        kernel: Kernel composition (default: RBF)
        **kwargs: Additional arguments for fit()

    Returns:
        Trained SingleOutputGP
    """
    if kernel is None:
        kernel = Kernel.rbf()
    gp = SingleOutputGP(kernel)
    gp.fit(X, y, **kwargs)
    return gp
