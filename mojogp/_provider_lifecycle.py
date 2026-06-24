"""Shared lifecycle helpers for persistent JIT provider handles.

The codegen engine caches kernel Python modules process-wide via ``sys.modules``.
When a wrapper keeps a provider handle alive after ``fit()``, another wrapper can
later load the same kernel module and attempt ``init_provider()`` on top of that
live state. Track persistent providers as revocable leases so other wrappers can
reclaim the module and force lazy rebuilds when they need it again.
"""

from __future__ import annotations

import atexit
import weakref
from typing import Any, Callable, Iterable


_LIVE_PROVIDER_LEASES: dict[
    int, tuple[weakref.ReferenceType[Any] | None, Callable[[], None], str | None]
] = {}
_LIVE_PROVIDER_LEASE_KEYS_BY_NAME: dict[str, set[int]] = {}


def _iter_modules(kernel_modules: Any | Iterable[Any]) -> Iterable[Any]:
    if isinstance(kernel_modules, (list, tuple)):
        return kernel_modules
    return (kernel_modules,)


def _iter_unique_modules(kernel_modules: Any | Iterable[Any]) -> Iterable[Any]:
    seen: set[int] = set()
    for kernel_module in _iter_modules(kernel_modules):
        key = id(kernel_module)
        if key in seen:
            continue
        seen.add(key)
        yield kernel_module


def _once(releaser: Callable[[], None]) -> Callable[[], None]:
    called = False

    def release() -> None:
        nonlocal called
        if called:
            return
        called = True
        releaser()

    return release


def _register_lease_entry(
    kernel_module: Any,
    owner_ref: weakref.ReferenceType[Any] | None,
    releaser: Callable[[], None],
) -> None:
    key = id(kernel_module)
    module_name = getattr(kernel_module, "__name__", None)
    _LIVE_PROVIDER_LEASES[key] = (owner_ref, releaser, module_name)
    if module_name:
        _LIVE_PROVIDER_LEASE_KEYS_BY_NAME.setdefault(module_name, set()).add(key)


def _pop_lease_entry(
    key: int,
) -> tuple[weakref.ReferenceType[Any] | None, Callable[[], None], str | None] | None:
    entry = _LIVE_PROVIDER_LEASES.pop(key, None)
    if entry is None:
        return None
    module_name = entry[2]
    if module_name:
        keys = _LIVE_PROVIDER_LEASE_KEYS_BY_NAME.get(module_name)
        if keys is not None:
            keys.discard(key)
            if not keys:
                _LIVE_PROVIDER_LEASE_KEYS_BY_NAME.pop(module_name, None)
    return entry


def revoke_conflicting_provider_lease(
    kernel_module: Any, owner: Any | None = None
) -> None:
    """Release another wrapper's persistent provider lease for this module."""

    entry = _LIVE_PROVIDER_LEASES.get(id(kernel_module))
    if entry is None:
        return

    owner_ref, releaser, _ = entry
    live_owner = owner_ref() if owner_ref is not None else None
    if owner_ref is not None and live_owner is None:
        _pop_lease_entry(id(kernel_module))
        return
    if owner is not None and live_owner is owner:
        return

    _pop_lease_entry(id(kernel_module))
    releaser()


def revoke_conflicting_provider_leases_by_name(
    module_name: str,
    owner: Any | None = None,
    include_live_owners: bool = False,
) -> None:
    """Release stale orphaned provider leases for an existing module name."""

    while True:
        progress = False
        for key in list(_LIVE_PROVIDER_LEASE_KEYS_BY_NAME.get(module_name, ())):
            entry = _LIVE_PROVIDER_LEASES.get(key)
            if entry is None:
                continue

            owner_ref, releaser, _ = entry
            live_owner = owner_ref() if owner_ref is not None else None
            if owner_ref is not None and live_owner is None:
                _pop_lease_entry(key)
                try:
                    releaser()
                except Exception:
                    pass
                progress = True
                continue
            if owner is not None and live_owner is owner:
                continue
            if (
                owner_ref is not None
                and live_owner is not None
                and not include_live_owners
            ):
                continue

            _pop_lease_entry(key)
            releaser()
            progress = True

        if not progress:
            break


def revoke_provider_leases(
    owner: Any | None = None,
    include_live_owners: bool = False,
) -> bool:
    """Release provider leases across all loaded kernel modules."""

    reclaimed = False
    while True:
        progress = False
        for key, entry in list(_LIVE_PROVIDER_LEASES.items()):
            owner_ref, releaser, _ = entry
            live_owner = owner_ref() if owner_ref is not None else None
            if owner_ref is not None and live_owner is None:
                _pop_lease_entry(key)
                try:
                    releaser()
                except Exception:
                    pass
                progress = True
                reclaimed = True
                continue
            if owner is not None and live_owner is owner:
                continue
            if (
                owner_ref is not None
                and live_owner is not None
                and not include_live_owners
            ):
                continue

            _pop_lease_entry(key)
            releaser()
            progress = True
            reclaimed = True

        if not progress:
            break
    return reclaimed


def register_provider_lease(
    kernel_modules: Any | Iterable[Any], owner: Any, releaser: Any
) -> None:
    """Record that ``owner`` currently keeps persistent providers alive."""

    owner_ref = weakref.ref(owner)
    releaser_name = str(releaser.__name__)

    def release_inner() -> None:
        live_owner = owner_ref()
        if live_owner is None:
            return
        getattr(live_owner, releaser_name)()

    release = _once(release_inner)

    for kernel_module in _iter_unique_modules(kernel_modules):
        revoke_conflicting_provider_lease(kernel_module, owner)
        _register_lease_entry(kernel_module, owner_ref, release)


def orphan_provider_leases(
    kernel_modules: Any | Iterable[Any], releaser: Callable[[], None]
) -> None:
    """Retain destroyable orphan leases across one or more kernel modules."""

    release = _once(releaser)
    for kernel_module in _iter_unique_modules(kernel_modules):
        _register_lease_entry(kernel_module, None, release)


def orphan_provider_lease(kernel_module: Any, provider_info: dict[str, Any]) -> None:
    """Retain a destroyable lease after the owning wrapper is gone.

    Some wrapper/provider combinations can be safely reused while still live, but
    destabilize the next ``init_provider()`` if they are destroyed eagerly during
    object finalization. Keep the raw provider handle revocable so another wrapper
    can explicitly reclaim the module later, or the atexit cleanup can release it.
    """

    if not provider_info:
        return
    provider_ptr = int(provider_info.get("provider_ptr", 0) or 0)
    if provider_ptr == 0:
        return
    destroy = getattr(kernel_module, "destroy_provider", None)
    if destroy is None:
        return

    orphan_provider_leases((kernel_module,), lambda: destroy(provider_info))


def unregister_provider_lease(kernel_modules: Any | Iterable[Any], owner: Any) -> None:
    """Drop lease records for providers explicitly destroyed by ``owner``."""

    for kernel_module in _iter_modules(kernel_modules):
        key = id(kernel_module)
        entry = _LIVE_PROVIDER_LEASES.get(key)
        if entry is None:
            continue
        owner_ref, _, _ = entry
        if owner_ref is None:
            continue
        live_owner = owner_ref()
        if live_owner is None or live_owner is owner:
            _pop_lease_entry(key)


def revoke_orphan_provider_leases() -> bool:
    """Release provider leases whose owners are gone or intentionally orphaned."""

    reclaimed = False
    for key, entry in list(_LIVE_PROVIDER_LEASES.items()):
        owner_ref, releaser, _ = entry
        live_owner = owner_ref() if owner_ref is not None else None
        if owner_ref is not None and live_owner is not None:
            continue
        _pop_lease_entry(key)
        try:
            releaser()
        except Exception:
            pass
        reclaimed = True
    return reclaimed


@atexit.register
def _cleanup_provider_leases() -> None:
    for key, (_, releaser, _) in list(_LIVE_PROVIDER_LEASES.items()):
        _pop_lease_entry(key)
        try:
            releaser()
        except Exception:
            pass
