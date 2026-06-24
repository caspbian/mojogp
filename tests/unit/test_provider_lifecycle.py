"""Unit tests for backend provider lifecycle helpers."""

import numpy as np

from mojogp._multi_output_backend import (
    destroy_provider_info,
    destroy_provider_infos,
    rebuild_trained_provider_infos,
)


class _DummyKernelModule:
    def __init__(self):
        self.init_calls = []
        self.materialize_calls = []
        self.destroy_calls = []

    def init_provider(self, X, params, noise):
        call = {
            "X_shape": tuple(X.shape),
            "params": np.array(params, copy=True),
            "noise": float(noise),
        }
        self.init_calls.append(call)
        return {"provider_ptr": len(self.init_calls), "shape": tuple(X.shape)}

    def materialize(self, provider_info):
        self.materialize_calls.append(dict(provider_info))

    def destroy_provider(self, provider_info):
        self.destroy_calls.append(dict(provider_info))


def test_rebuild_trained_provider_infos_initializes_each_provider():
    modules = [_DummyKernelModule(), _DummyKernelModule()]
    x_train = [np.ones((5, 2), dtype=np.float32), np.ones((7, 3), dtype=np.float32)]
    params = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0], dtype=np.float32)]

    infos = rebuild_trained_provider_infos(
        modules, x_train, params, method="matrix_free"
    )

    assert len(infos) == 2
    assert infos[0]["provider_ptr"] == 1
    assert infos[1]["provider_ptr"] == 1
    assert modules[0].init_calls[0]["X_shape"] == (5, 2)
    assert modules[1].init_calls[0]["X_shape"] == (7, 3)
    assert modules[0].materialize_calls == []
    assert modules[1].materialize_calls == []


def test_rebuild_trained_provider_infos_materializes_in_materialized_mode():
    module = _DummyKernelModule()

    infos = rebuild_trained_provider_infos(
        [module],
        [np.ones((4, 2), dtype=np.float32)],
        [np.array([1.0, 2.0], dtype=np.float32)],
        method="materialized",
    )

    assert len(infos) == 1
    assert len(module.materialize_calls) == 1
    assert module.materialize_calls[0]["provider_ptr"] == infos[0]["provider_ptr"]


def test_destroy_provider_info_ignores_empty_or_zero_ptr():
    module = _DummyKernelModule()

    destroy_provider_info(module, None)
    destroy_provider_info(module, {})
    destroy_provider_info(module, {"provider_ptr": 0})

    assert module.destroy_calls == []


def test_destroy_provider_infos_destroys_each_live_provider():
    modules = [_DummyKernelModule(), _DummyKernelModule()]
    infos = [{"provider_ptr": 11}, {"provider_ptr": 12}]

    destroy_provider_infos(modules, infos)

    assert modules[0].destroy_calls == [{"provider_ptr": 11}]
    assert modules[1].destroy_calls == [{"provider_ptr": 12}]
