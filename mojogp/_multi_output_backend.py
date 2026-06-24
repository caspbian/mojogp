"""Shared backend helpers for multi-output JIT wrappers."""

from __future__ import annotations

import ctypes
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ._provider_lifecycle import (
    revoke_conflicting_provider_lease,
    revoke_conflicting_provider_leases_by_name,
)


PRECOND_METHOD_MAP = {
    "greedy": 0,
    "rpcholesky": 1,
    "nystrom": 2,
    "auto": 2,
}


_UPDATE_NOISE_FNS: dict[int, Any] = {}


def resolve_preconditioner_settings(
    base: dict[str, Any],
    *,
    precond_rank: Optional[int] = None,
    precond_rebuild_threshold: Optional[float] = None,
    precond: Optional[str] = None,
    use_preconditioner: Optional[bool] = None,
) -> dict[str, Any]:
    """Resolve live preconditioner settings from preset + overrides."""
    resolved = dict(base)
    if precond_rank is not None:
        resolved["precond_rank"] = precond_rank
    if precond_rebuild_threshold is not None:
        resolved["precond_rebuild_threshold"] = precond_rebuild_threshold
    elif "precond_rebuild_threshold" not in resolved:
        resolved["precond_rebuild_threshold"] = 0.5
    if precond is not None:
        resolved["precond"] = precond
    if "precond" not in resolved:
        resolved["precond"] = "greedy"
    resolved_use_preconditioner = (
        bool(use_preconditioner)
        if use_preconditioner is not None
        else int(resolved.get("precond_rank", 0)) > 0
    )
    if int(resolved.get("precond_rank", 0)) <= 0:
        resolved_use_preconditioner = False
        resolved["precond_rank"] = 0
    elif not resolved_use_preconditioner:
        resolved["precond_rank"] = 0
    resolved["precond_method"] = PRECOND_METHOD_MAP.get(str(resolved["precond"]), 0)
    resolved["use_preconditioner"] = resolved_use_preconditioner
    return resolved


def build_backend_train_info(
    raw: Optional[dict[str, Any]],
    method: str,
) -> Optional[dict[str, Any]]:
    """Normalize backend training metadata returned by the JIT engine."""
    if raw is None:
        return None

    route = raw.get("training_route")
    if route is None:
        route = "materialized" if method == "materialized" else "matrix_free"

    materialization_mode = raw.get("materialization_mode")
    if materialization_mode is None:
        materialization_mode = 1 if route == "materialized" else 0

    info: dict[str, Any] = {
        "training_route": str(route),
        "materialization_mode": int(materialization_mode),
    }
    for key in (
        "max_tridiag_iter",
        "precond_rebuild_threshold",
        "precond_rebuild_count",
        "precond_rank",
        "precond_method",
        "use_preconditioner",
        "noise_mode",
        "has_observation_noise_vector",
        "iter_times_ms",
    ):
        if key in raw:
            info[key] = raw[key]
    return info


def rebuild_trained_provider_infos(
    kernel_modules: Sequence[Any],
    x_train_per_latent: Sequence[np.ndarray],
    params_per_latent: Iterable[Sequence[float] | np.ndarray],
    method: str,
    param_kernels: Optional[Sequence[Any]] = None,
) -> list[dict[str, Any]]:
    """Recreate trained provider-info dicts for backend prediction bindings."""
    provider_infos: list[dict[str, Any]] = []
    if param_kernels is None:
        param_kernels = [None] * len(kernel_modules)

    seen_module_ids: set[int] = set()
    for kernel_module in kernel_modules:
        module_id = id(kernel_module)
        if module_id in seen_module_ids:
            continue
        seen_module_ids.add(module_id)
        revoke_conflicting_provider_lease(kernel_module)
        module_name = getattr(kernel_module, "__name__", None)
        if module_name is not None:
            revoke_conflicting_provider_leases_by_name(
                module_name,
                include_live_owners=True,
            )

    for kernel_module, x_train_s, params_s, param_kernel in zip(
        kernel_modules, x_train_per_latent, params_per_latent, param_kernels
    ):
        params_arr = np.ascontiguousarray(params_s, dtype=np.float32)
        if param_kernel is not None:
            params_arr = np.ascontiguousarray(
                param_kernel.to_engine_params(params_arr), dtype=np.float32
            )
        provider_info = kernel_module.init_provider(
            np.ascontiguousarray(x_train_s, dtype=np.float32), params_arr, 0.0
        )
        if method == "materialized":
            kernel_module.materialize(provider_info)
        provider_infos.append(provider_info)
    return provider_infos


def destroy_provider_info(
    kernel_module: Any, provider_info: Optional[dict[str, Any]]
) -> None:
    """Destroy a temporary provider-info dict if the kernel module supports it."""
    if not provider_info:
        return
    provider_ptr = int(provider_info.get("provider_ptr", 0) or 0)
    if provider_ptr == 0:
        return
    destroy = getattr(kernel_module, "destroy_provider", None)
    if destroy is None:
        return
    destroy(provider_info)


def destroy_provider_infos(
    kernel_modules: Sequence[Any], provider_infos: Optional[Sequence[dict[str, Any]]]
) -> None:
    """Destroy a sequence of temporary provider-info dicts."""
    if provider_infos is None:
        return
    for kernel_module, provider_info in zip(kernel_modules, provider_infos):
        destroy_provider_info(kernel_module, provider_info)


def update_provider_noise(
    provider_info: Optional[dict[str, Any]], noise: float
) -> None:
    """Update a live provider's noise in place via its exported fn ptr."""
    if not provider_info:
        return
    provider_ptr = int(provider_info.get("provider_ptr", 0) or 0)
    update_noise_ptr = int(provider_info.get("update_noise", 0) or 0)
    if provider_ptr == 0 or update_noise_ptr == 0:
        return

    updater = _UPDATE_NOISE_FNS.get(update_noise_ptr)
    if updater is None:
        updater = ctypes.CFUNCTYPE(None, ctypes.c_ssize_t, ctypes.c_float)(
            update_noise_ptr
        )
        _UPDATE_NOISE_FNS[update_noise_ptr] = updater
    updater(provider_ptr, float(noise))


def build_backend_predict_info(
    *,
    requested_method: str,
    actual_prediction_route: str,
    backend_prediction_used: bool,
    backend_variance_used: Optional[bool] = None,
    variance_method: Optional[str] = None,
    fallback_used: bool = False,
    backend_error: Optional[str] = None,
    actual_variance_route: Optional[str] = None,
    training_route: Optional[str] = None,
    precond_rank: Optional[int] = None,
    precond_method: Optional[int | str] = None,
    precond_rebuild_count: Optional[int] = None,
) -> dict[str, Any]:
    """Normalize prediction-route telemetry for wrappers."""
    info = {
        "requested_method": requested_method,
        "actual_prediction_route": actual_prediction_route,
        "backend_prediction_used": bool(backend_prediction_used),
        "fallback_used": bool(fallback_used),
        "variance_method": variance_method,
    }
    if backend_variance_used is not None:
        info["backend_variance_used"] = bool(backend_variance_used)
    if actual_variance_route is not None:
        info["actual_variance_route"] = actual_variance_route
    if training_route is not None:
        info["training_route"] = training_route
    if precond_rank is not None:
        info["precond_rank"] = int(precond_rank)
    if precond_method is not None:
        info["precond_method"] = precond_method
    if precond_rebuild_count is not None:
        info["precond_rebuild_count"] = int(precond_rebuild_count)
    if backend_error is not None:
        info["backend_error"] = str(backend_error)
    return info
