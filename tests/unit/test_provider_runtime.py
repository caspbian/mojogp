"""Unit tests for bundle-aware provider runtime ownership."""

from mojogp._provider_runtime import (
    BUNDLE_ROLE_INFERENCE,
    BUNDLE_ROLE_TRAINING,
    BUNDLE_STATE_DESTROYED,
    BUNDLE_STATE_LIVE_OWNED,
    BUNDLE_STATE_ORPHANED,
    ProviderBundle,
    destroy_provider_bundle,
    orphan_provider_bundle,
    reclaim_provider_bundles_by_name,
    reclaim_provider_bundles_for_modules,
    register_provider_bundle,
)


class _DummyKernelModule:
    def __init__(self, name: str):
        self.__name__ = name
        self.destroy_calls = []

    def destroy_provider(self, provider_info):
        self.destroy_calls.append(dict(provider_info))


class _Owner:
    pass


def _bundle(
    module_name: str, role: str = BUNDLE_ROLE_TRAINING
) -> tuple[ProviderBundle, _DummyKernelModule]:
    module = _DummyKernelModule(module_name)
    bundle = ProviderBundle(
        role=role,
        method="materialized",
        kernel_modules=[module],
        provider_infos=[{"provider_ptr": 7}],
    )
    return bundle, module


def test_register_orphan_and_destroy_bundle_transitions():
    bundle, module = _bundle("test_bundle_transitions")
    owner = _Owner()

    register_provider_bundle(bundle, owner)
    assert bundle.state == BUNDLE_STATE_LIVE_OWNED
    assert bundle.owner() is owner

    orphan_provider_bundle(bundle)
    assert bundle.state == BUNDLE_STATE_ORPHANED
    assert bundle.owner() is None

    destroy_provider_bundle(bundle)
    assert bundle.state == BUNDLE_STATE_DESTROYED
    assert module.destroy_calls == [{"provider_ptr": 7}]


def test_reclaim_by_module_name_skips_live_owner_by_default():
    bundle, module = _bundle("test_bundle_name_skip", role=BUNDLE_ROLE_INFERENCE)
    owner = _Owner()

    register_provider_bundle(bundle, owner)
    reclaim_provider_bundles_by_name([module.__name__], include_live_owners=False)

    assert bundle.state == BUNDLE_STATE_LIVE_OWNED
    assert module.destroy_calls == []

    reclaim_provider_bundles_by_name([module.__name__], include_live_owners=True)

    assert bundle.state == BUNDLE_STATE_DESTROYED
    assert module.destroy_calls == [{"provider_ptr": 7}]


def test_reclaim_by_module_object_destroys_orphaned_bundle():
    bundle, module = _bundle("test_bundle_object_orphan")
    owner = _Owner()

    register_provider_bundle(bundle, owner)
    orphan_provider_bundle(bundle)
    reclaim_provider_bundles_for_modules([module])

    assert bundle.state == BUNDLE_STATE_DESTROYED
    assert module.destroy_calls == [{"provider_ptr": 7}]
