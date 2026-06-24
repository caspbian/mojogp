"""Unit tests for mixed MultiOutputGP Python-side plumbing.

These tests mock the JIT engine so they validate routing, persistence, and
state reconstruction without running full GPU training.
"""

import numpy as np

from mojogp import MultiOutputGP
from mojogp.kernel import Kernel


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
        self.train_calls = []
        self.predict_calls = []

    def train_multi_output_mixed(self, *args):
        self.train_calls.append(args)
        y = np.asarray(args[1], dtype=np.float32)
        params = np.asarray(args[2], dtype=np.float32)
        noise_per_task = np.asarray(args[3], dtype=np.float32)
        outputscale = float(args[4])
        mean_per_task = np.asarray(args[5], dtype=np.float32)
        num_tasks = int(args[6])
        cat_params = np.asarray(args[9], dtype=np.float32)
        task_rank = int(args[12])
        n = y.shape[0]

        return {
            "params": params.copy(),
            "cat_params": cat_params.copy(),
            "noise_per_task": noise_per_task.copy(),
            "final_nll": 1.23,
            "iterations": 1,
            "converged": False,
            "num_tasks": num_tasks,
            "task_rank": task_rank,
            "outputscale": outputscale,
            "B_flat": np.eye(num_tasks, dtype=np.float32).ravel(),
            "alpha": np.zeros(n * num_tasks, dtype=np.float32),
            "mean_per_task": mean_per_task.copy(),
            "nll_history": np.array([1.23], dtype=np.float32),
        }

    def predict_multi_output_mixed(self, *args):
        self.predict_calls.append(args)
        x_test = np.asarray(args[4], dtype=np.float32)
        num_tasks = int(np.asarray(args[1]).shape[1])
        n_test = x_test.shape[0]
        return {
            "mean": np.full((n_test, num_tasks), 0.5, dtype=np.float32),
            "variance": np.full((n_test, num_tasks), 0.25, dtype=np.float32),
        }


def _make_gp() -> MultiOutputGP:
    kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.ehh(levels=3, active_dims=[2])
    gp = MultiOutputGP(kernel=kernel, task_rank=1)
    gp._kernel_module = _FakeKernelModule()
    gp._engine = _FakeEngine()
    gp._ensure_compiled = lambda *args, **kwargs: None
    return gp


def _make_data(n: int = 32):
    rng = np.random.RandomState(0)
    x_cont = rng.randn(n, 2).astype(np.float32)
    c = rng.randint(0, 3, size=(n, 1)).astype(np.float32)
    x = np.concatenate([x_cont, c], axis=1)
    y = rng.randn(n, 2).astype(np.float32)
    return x, y


class TestMultiOutputMixedPlumbing:
    def test_materialized_mixed_fit_predict_routes_to_mixed_engine(self):
        gp = _make_gp()
        x, y = _make_data()

        result = gp.fit(x, y, max_iterations=1, method="materialized")
        mean, var = gp.predict(x[:5], return_var=True)

        assert gp._training_method == "materialized"
        assert gp._engine.train_calls[-1][18] == gp.precond_method
        assert gp._engine.train_calls[-1][19] == gp.precond_rebuild_threshold
        assert gp._engine.train_calls[-1][20] == 1
        assert gp._engine.train_calls[-1][21] == gp.max_tridiag_iter
        assert gp._engine.predict_calls[-1][-1] == 1
        assert result.cat_params is not None
        assert mean.shape == (5, 2)
        assert var.shape == (5, 2)

    def test_matrix_free_mixed_fit_predict_routes_to_mixed_engine(self):
        gp = _make_gp()
        x, y = _make_data()

        gp.fit(x, y, max_iterations=1, method="matrix_free")
        mean, var = gp.predict(x[:4], return_var=True)

        assert gp._training_method == "matrix_free"
        assert gp._engine.train_calls[-1][18] == gp.precond_method
        assert gp._engine.train_calls[-1][19] == gp.precond_rebuild_threshold
        assert gp._engine.train_calls[-1][20] == 0
        assert gp._engine.train_calls[-1][21] == gp.max_tridiag_iter
        assert gp._engine.predict_calls[-1][-1] == 0
        assert mean.shape == (4, 2)
        assert var.shape == (4, 2)

    def test_fit_method_aliases_resolve_to_canonical_routes(self):
        x, y = _make_data()

        gp_mat = _make_gp()
        gp_mat.fit(x, y, max_iterations=1, method="mat")
        assert gp_mat.method == "materialized"
        assert gp_mat._training_method == "materialized"
        assert gp_mat._engine.train_calls[-1][20] == 1

        gp_mf = _make_gp()
        gp_mf.fit(x, y, max_iterations=1, method="mf")
        assert gp_mf.method == "matrix_free"
        assert gp_mf._training_method == "matrix_free"
        assert gp_mf._engine.train_calls[-1][20] == 0

    def test_mixed_fit_passes_max_tridiag_iter_to_engine(self):
        gp = _make_gp()
        gp.max_tridiag_iter = 7
        x, y = _make_data()

        gp.fit(x, y, max_iterations=1)

        assert gp._engine.train_calls[-1][18] == gp.precond_method
        assert gp._engine.train_calls[-1][19] == gp.precond_rebuild_threshold
        assert gp._engine.train_calls[-1][20] == 1
        assert gp._engine.train_calls[-1][21] == 7

    def test_mixed_fit_passes_precond_rebuild_threshold_to_engine(self):
        gp = _make_gp()
        gp.precond_rebuild_threshold = 0.125
        x, y = _make_data()

        gp.fit(x, y, max_iterations=1, method="matrix_free")

        assert gp._engine.train_calls[-1][19] == 0.125

    def test_mixed_fit_passes_early_stopping_controls_to_engine(self):
        gp = _make_gp()
        x, y = _make_data()

        gp.fit(
            x,
            y,
            max_iterations=1,
            method="matrix_free",
            early_stop_patience=7,
            early_stop_tol=0.0,
        )

        assert gp._engine.train_calls[-1][22] == 7
        assert gp._engine.train_calls[-1][23] == 0.0

    def test_mixed_save_load_restores_categorical_state(self, tmp_path, monkeypatch):
        gp = _make_gp()
        x, y = _make_data()
        gp.fit(x, y, max_iterations=1)

        path = tmp_path / "mixed_multi_output_gp"
        gp.save(path)

        def _fake_ensure_compiled(self, dim, num_tasks=0, fresh_load=False):
            _ = fresh_load
            self._kernel_module = _FakeKernelModule()
            self._engine = _FakeEngine()
            self._dim = dim
            self._num_tasks = num_tasks

        monkeypatch.setattr(MultiOutputGP, "_ensure_compiled", _fake_ensure_compiled)

        loaded = MultiOutputGP.load(path, kernel=gp._original_kernel)
        mean, var = loaded.predict(x[:6], return_var=True)

        assert loaded._is_mixed is True
        assert loaded._C_train is not None
        assert loaded._X_train_cont is not None
        assert loaded._cat_specs is not None
        assert loaded._training_method == "materialized"
        assert loaded.training_result.cat_params is not None
        assert loaded._engine.predict_calls[-1][-1] == 1
        assert mean.shape == (6, 2)
        assert var.shape == (6, 2)
