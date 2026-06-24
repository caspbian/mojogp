"""Bundle-aware runtime ownership for fn-ptr provider state.

This module is the long-term lifecycle primitive for wrappers that need to keep
native provider state alive across repeated inference routes without relying on
per-call module reloads or ad hoc `sys.modules` cleanup.
"""

from __future__ import annotations

import atexit
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from ._multi_output_backend import destroy_provider_infos


BUNDLE_ROLE_TRAINING = "training"
BUNDLE_ROLE_INFERENCE = "inference"

BUNDLE_STATE_LIVE_OWNED = "live_owned"
BUNDLE_STATE_ORPHANED = "orphaned_reclaimable"
BUNDLE_STATE_DESTROYED = "destroyed"


def _iter_modules(kernel_modules: Sequence[Any]) -> Iterable[Any]:
    seen: set[int] = set()
    for kernel_module in kernel_modules:
        module_id = id(kernel_module)
        if module_id in seen:
            continue
        seen.add(module_id)
        yield kernel_module


@dataclass
class ProviderBundle:
    """Runtime cache bundle for provider-backed inference/training state."""

    role: str
    method: str
    kernel_modules: list[Any]
    provider_infos: list[dict[str, Any]]
    state: str = field(default=BUNDLE_STATE_LIVE_OWNED, init=False)
    owner_ref: Optional[weakref.ReferenceType[Any]] = field(default=None, init=False)
    owner_id: Optional[int] = field(default=None, init=False)
    owner_releaser_name: Optional[str] = field(default=None, init=False)
    module_ids: tuple[int, ...] = field(default_factory=tuple, init=False)
    module_names: tuple[str, ...] = field(default_factory=tuple, init=False)
    _destroyed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.module_ids = tuple(id(m) for m in _iter_modules(self.kernel_modules))
        self.module_names = tuple(
            name
            for name in (
                getattr(m, "__name__", None) for m in _iter_modules(self.kernel_modules)
            )
            if name
        )

    @property
    def is_destroyed(self) -> bool:
        return self._destroyed

    @property
    def is_orphaned(self) -> bool:
        return self.state == BUNDLE_STATE_ORPHANED and not self._destroyed

    def bind_owner(self, owner: Any | None) -> None:
        if owner is None:
            self.owner_ref = None
            self.owner_id = None
            self.state = BUNDLE_STATE_ORPHANED
            return
        self.owner_ref = weakref.ref(owner)
        self.owner_id = id(owner)
        self.state = BUNDLE_STATE_LIVE_OWNED

    def owner(self) -> Any | None:
        return self.owner_ref() if self.owner_ref is not None else None

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self.state = BUNDLE_STATE_DESTROYED
        self.owner_ref = None
        self.owner_id = None
        destroy_provider_infos(self.kernel_modules, self.provider_infos)


_BUNDLES: dict[int, ProviderBundle] = {}
_BUNDLE_IDS_BY_MODULE_ID: dict[int, int] = {}
_BUNDLE_IDS_BY_MODULE_NAME: dict[str, set[int]] = {}


def _register_bundle_indexes(bundle_id: int, bundle: ProviderBundle) -> None:
    _BUNDLES[bundle_id] = bundle
    for module_id in bundle.module_ids:
        _BUNDLE_IDS_BY_MODULE_ID[module_id] = bundle_id
    for module_name in bundle.module_names:
        _BUNDLE_IDS_BY_MODULE_NAME.setdefault(module_name, set()).add(bundle_id)


def _pop_bundle_indexes(bundle_id: int) -> ProviderBundle | None:
    bundle = _BUNDLES.pop(bundle_id, None)
    if bundle is None:
        return None
    for module_id in bundle.module_ids:
        current = _BUNDLE_IDS_BY_MODULE_ID.get(module_id)
        if current == bundle_id:
            _BUNDLE_IDS_BY_MODULE_ID.pop(module_id, None)
    for module_name in bundle.module_names:
        bundle_ids = _BUNDLE_IDS_BY_MODULE_NAME.get(module_name)
        if bundle_ids is None:
            continue
        bundle_ids.discard(bundle_id)
        if not bundle_ids:
            _BUNDLE_IDS_BY_MODULE_NAME.pop(module_name, None)
    return bundle


def register_provider_bundle(
    bundle: ProviderBundle,
    owner: Any,
    *,
    owner_releaser_name: str | None = None,
) -> ProviderBundle:
    """Register a live owned runtime bundle."""

    bundle.bind_owner(owner)
    bundle.owner_releaser_name = owner_releaser_name
    _register_bundle_indexes(id(bundle), bundle)
    return bundle


def orphan_provider_bundle(bundle: ProviderBundle | None) -> None:
    """Mark a bundle as reclaimable without destroying native state."""

    if bundle is None or bundle.is_destroyed:
        return
    bundle.bind_owner(None)
    _register_bundle_indexes(id(bundle), bundle)


def destroy_provider_bundle(bundle: ProviderBundle | None) -> None:
    """Destroy a bundle and remove it from the registry."""

    if bundle is None:
        return
    _pop_bundle_indexes(id(bundle))
    bundle.destroy()


def _should_reclaim_bundle(
    bundle: ProviderBundle,
    owner: Any | None,
    include_live_owners: bool,
    roles: Sequence[str] | None,
) -> bool:
    if roles is not None and bundle.role not in roles:
        return False
    if bundle.is_destroyed:
        return True
    live_owner = bundle.owner()
    if bundle.owner_ref is not None and live_owner is None:
        return True
    if owner is not None and live_owner is owner:
        return False
    if bundle.is_orphaned:
        return True
    return bool(include_live_owners)


def _invoke_owner_releaser(bundle: ProviderBundle) -> None:
    live_owner = bundle.owner()
    if live_owner is None:
        return
    releaser_name = bundle.owner_releaser_name
    if releaser_name is None:
        return
    releaser = getattr(live_owner, releaser_name, None)
    if releaser is None:
        return
    releaser(bundle)


def reclaim_provider_bundles_for_modules(
    kernel_modules: Sequence[Any],
    *,
    owner: Any | None = None,
    include_live_owners: bool = False,
    roles: Sequence[str] | None = None,
) -> bool:
    """Reclaim bundles tied to concrete module objects."""

    seen_bundle_ids: set[int] = set()
    reclaimed_any = False
    for kernel_module in _iter_modules(kernel_modules):
        bundle_id = _BUNDLE_IDS_BY_MODULE_ID.get(id(kernel_module))
        if bundle_id is None or bundle_id in seen_bundle_ids:
            continue
        seen_bundle_ids.add(bundle_id)
        bundle = _BUNDLES.get(bundle_id)
        if bundle is None:
            continue
        if _should_reclaim_bundle(bundle, owner, include_live_owners, roles):
            if bundle.owner() is not None and owner is not bundle.owner():
                _invoke_owner_releaser(bundle)
            destroy_provider_bundle(bundle)
            reclaimed_any = True
    return reclaimed_any


def reclaim_provider_bundles_by_name(
    module_names: Sequence[str],
    *,
    owner: Any | None = None,
    include_live_owners: bool = False,
    roles: Sequence[str] | None = None,
) -> bool:
    """Reclaim bundles tied to matching module names."""

    seen_bundle_ids: set[int] = set()
    reclaimed_any = False
    for module_name in module_names:
        for bundle_id in list(_BUNDLE_IDS_BY_MODULE_NAME.get(module_name, ())):
            if bundle_id in seen_bundle_ids:
                continue
            seen_bundle_ids.add(bundle_id)
            bundle = _BUNDLES.get(bundle_id)
            if bundle is None:
                continue
            if _should_reclaim_bundle(bundle, owner, include_live_owners, roles):
                if bundle.owner() is not None and owner is not bundle.owner():
                    _invoke_owner_releaser(bundle)
                destroy_provider_bundle(bundle)
                reclaimed_any = True
    return reclaimed_any


def bundle_runtime_owner_role(bundle: ProviderBundle | None) -> str | None:
    if bundle is None or bundle.is_destroyed:
        return None
    return bundle.role


@atexit.register
def _cleanup_provider_bundles() -> None:
    for bundle_id in list(_BUNDLES):
        bundle = _pop_bundle_indexes(bundle_id)
        if bundle is None:
            continue
        try:
            bundle.destroy()
        except Exception:
            pass
