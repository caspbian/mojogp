"""Unit tests for mixed-kernel method routing.

These tests mock the JIT engine so they only verify Python-side method
resolution/plumbing, not actual GPU training.
"""

import numpy as np
import pytest

from mojogp import ExperimentalFeatureWarning, SingleOutputGP, GD, Kernel

_TRAIN_MIXED_METHOD_ARG = 14


class _FakeKernelModule:
    def init_provider(self, X, params, noise):
        return {
            "provider_ptr": 1,
            "n": int(X.shape[0]),
            "num_gradient_params": int(len(params)),
            "supports_fused_gradient": True,
            "supports_fused_ls_os": False,
            "supports_fused_3param": False,
            "x_ptr": 1,
        }

    def materialize(self, provider_info):
        provider_info["materialized"] = True


class _FakeEngine:
    def __init__(self):
        self.train_mixed_calls = []
        self.predict_mixed_calls = []

    def train_mixed(self, *args):
        self.train_mixed_calls.append(args)
        y = np.asarray(args[1], dtype=np.float32)
        return {
            "params": np.array(args[2], dtype=np.float32),
            "cat_params": np.array(args[6], dtype=np.float32),
            "noise": float(args[3]),
            "mean": float(np.mean(y)),
            "nll": 1.23,
            "iterations": 1,
            "converged": False,
            "nll_history": [1.23],
        }

    def predict_mixed(self, *args):
        self.predict_mixed_calls.append(args)
        n_test = int(np.asarray(args[2]).shape[0])
        return {
            "mean": np.full(n_test, 0.5, dtype=np.float32),
            "variance": np.full(n_test, 0.25, dtype=np.float32),
        }


def _make_gp():
    kernel = Kernel.rbf(active_dims=[0, 1]) * GD(levels=3, active_dims=[2])
    gp = SingleOutputGP(kernel)
    gp._kernel_module = _FakeKernelModule()
    gp._engine = _FakeEngine()
    gp._ensure_compiled = lambda: None
    return gp


def _make_data(n):
    rng = np.random.RandomState(0)
    X_cont = rng.randn(n, 2).astype(np.float32)
    C = rng.randint(0, 3, size=(n, 1)).astype(np.float32)
    X = np.concatenate([X_cont, C], axis=1)
    y = rng.randn(n).astype(np.float32)
    return X, y


class TestMixedMethodPlumbing:
    def test_auto_resolves_to_materialized_for_small_mixed_gp(self):
        gp = _make_gp()
        X, y = _make_data(128)

        gp.fit(X, y, max_iterations=1, method="auto")

        assert gp._training_method == "materialized"
        assert gp._engine.train_mixed_calls[-1][_TRAIN_MIXED_METHOD_ARG] == 1

    def test_auto_resolves_to_matrix_free_for_large_mixed_gp(self):
        gp = _make_gp()
        X, y = _make_data(2001)

        gp.fit(X, y, max_iterations=1, method="auto")

        assert gp._training_method == "matrix_free"
        assert gp._engine.train_mixed_calls[-1][_TRAIN_MIXED_METHOD_ARG] == 0

    def test_materialized_predict_passes_materialized_flag_and_persists(
        self, tmp_path, monkeypatch
    ):
        gp = _make_gp()
        X, y = _make_data(64)

        with pytest.warns(ExperimentalFeatureWarning):
            gp.fit(X, y, max_iterations=1, method="materialized")

        pred = gp.predict(X[:5])

        assert gp._training_method == "materialized"
        assert gp._engine.train_mixed_calls[-1][_TRAIN_MIXED_METHOD_ARG] == 1
        assert gp._engine.predict_mixed_calls[-1][-1] == 1
        assert pred.mean.shape == (5,)
        assert pred.std.shape == (5,)

        path = tmp_path / "mixed_gp"
        gp.save(path)

        def _fake_ensure_compiled(self):
            self._kernel_module = _FakeKernelModule()
            self._engine = _FakeEngine()

        monkeypatch.setattr(SingleOutputGP, "_ensure_compiled", _fake_ensure_compiled)
        gp_loaded = SingleOutputGP.load(path, kernel=gp._original_kernel)
        assert gp_loaded._training_method == "materialized"

    def test_materialized_grads_is_not_public_api(self):
        gp = _make_gp()
        X, y = _make_data(64)

        with pytest.warns(ExperimentalFeatureWarning):
            with pytest.raises(NotImplementedError, match="materialized_grads"):
                gp.fit(X, y, max_iterations=1, method="materialized_grads")
