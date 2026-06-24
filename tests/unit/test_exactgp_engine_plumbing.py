"""Unit tests for SingleOutputGP continuous engine plumbing.

These tests mock the JIT engine so they validate Python-side argument routing
and backend metadata capture without running the real GPU backend.
"""

import numpy as np

from mojogp import SingleOutputGP, RBF


class _FakeKernelModule:
    def __init__(self):
        self.materialize_calls = 0

    def init_provider(self, X, params, noise):
        return {
            "provider_ptr": 1,
            "n": int(X.shape[0]),
            "num_gradient_params": int(len(params)),
            "materialization_mode": 0,
            "is_ard": False,
        }

    def materialize(self, provider_info):
        self.materialize_calls += 1
        provider_info["materialization_mode"] = 1


class _FakeEngine:
    def __init__(self):
        self.train_calls = []

    def train(self, *args):
        self.train_calls.append(args)
        info = args[0]
        return {
            "params": np.array(args[2], dtype=np.float32),
            "noise": float(args[3]),
            "mean": 0.0,
            "final_nll": 1.23,
            "iterations": 1,
            "converged": False,
            "nll_history": [1.23],
            "training_route": (
                "materialized"
                if int(info.get("materialization_mode", 0)) == 1
                else "matrix_free"
            ),
            "materialization_mode": int(info.get("materialization_mode", 0)),
            "is_ard": bool(info.get("is_ard", False)),
            "precond_method": int(args[15]),
            "precond_rank": int(args[9]),
            "max_tridiag_iter": int(args[13]),
            "precond_rebuild_threshold": float(args[14]),
            "use_preconditioner": bool(args[12]),
        }


def _make_data(n=2000, d=5):
    rng = np.random.RandomState(0)
    X = rng.randn(n, d).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(n)).astype(np.float32)
    return X, y


class TestExactGPEnginePlumbing:
    def test_fit_passes_resolved_preconditioner_controls_to_engine(self):
        gp = SingleOutputGP(RBF())
        gp._kernel_module = _FakeKernelModule()
        gp._engine = _FakeEngine()
        gp._ensure_compiled = lambda: None

        X, y = _make_data()
        result = gp.fit(
            X,
            y,
            max_iterations=1,
            method="materialized",
            max_tridiag_iterations=17,
            precond_rebuild_threshold=0.125,
            preconditioner="greedy",
            verbose=False,
        )

        train_args = gp._engine.train_calls[-1]
        assert gp._kernel_module.materialize_calls == 1
        assert train_args[13] == 17
        assert np.isclose(float(train_args[14]), 0.125)
        assert train_args[15] == 0

        assert result.lanczos_rank == 17
        assert gp.backend_train_info == {
            "training_route": "materialized",
            "materialization_mode": 1,
            "is_ard": False,
            "precond_method": 0,
            "precond_rank": 100,
            "max_cg_iter": 100,
            "max_cg_iterations": 100,
            "cg_tol": 0.01,
            "cg_tolerance": 0.01,
            "max_tridiag_iter": 17,
            "precond_rebuild_threshold": 0.125,
            "enable_early_stopping": False,
            "early_stop_patience": 10,
            "early_stop_tol": 1e-4,
            "use_preconditioner": True,
            "noise_mode": "scalar",
            "learn_noise": True,
            "has_observation_noise_vector": False,
            "noise_regularization": 0.01,
        }

    def test_matrix_free_default_preconditioner_policy_enables_low_dim_rbf(self):
        gp = SingleOutputGP(RBF())
        gp._kernel_module = _FakeKernelModule()
        gp._engine = _FakeEngine()
        gp._ensure_compiled = lambda: None

        X, y = _make_data(n=5000, d=5)
        gp.fit(X, y, max_iterations=1, method="matrix_free", verbose=False)

        train_args = gp._engine.train_calls[-1]
        assert train_args[9] == 256
        assert train_args[12] is True
        assert train_args[15] == 0
        assert gp.backend_train_info["precond_rank"] == 256
        assert gp.backend_train_info["use_preconditioner"] is True

    def test_fit_method_aliases_resolve_to_canonical_routes(self):
        X, y = _make_data(n=5000, d=5)

        gp_mat = SingleOutputGP(RBF())
        gp_mat._kernel_module = _FakeKernelModule()
        gp_mat._engine = _FakeEngine()
        gp_mat._ensure_compiled = lambda: None
        gp_mat.fit(X, y, max_iterations=1, method="mat", verbose=False)

        assert gp_mat._training_method == "materialized"
        assert gp_mat._kernel_module.materialize_calls == 1

        gp_mf = SingleOutputGP(RBF())
        gp_mf._kernel_module = _FakeKernelModule()
        gp_mf._engine = _FakeEngine()
        gp_mf._ensure_compiled = lambda: None
        gp_mf.fit(X, y, max_iterations=1, method="mf", verbose=False)

        assert gp_mf._training_method == "matrix_free"
        assert gp_mf._kernel_module.materialize_calls == 0

    def test_matrix_free_default_preconditioner_policy_disables_unsupported_route(self):
        gp = SingleOutputGP(RBF())
        gp._kernel_module = _FakeKernelModule()
        gp._engine = _FakeEngine()
        gp._ensure_compiled = lambda: None

        X, y = _make_data(n=5000, d=17)
        gp.fit(X, y, max_iterations=1, method="matrix_free", verbose=False)

        train_args = gp._engine.train_calls[-1]
        assert train_args[9] == 0
        assert train_args[12] is False
        assert gp.backend_train_info["precond_rank"] == 0
        assert gp.backend_train_info["use_preconditioner"] is False

    def test_matrix_free_explicit_preconditioner_overrides_route_policy(self):
        gp = SingleOutputGP(RBF())
        gp._kernel_module = _FakeKernelModule()
        gp._engine = _FakeEngine()
        gp._ensure_compiled = lambda: None

        X, y = _make_data(n=5000, d=17)
        gp.fit(
            X,
            y,
            max_iterations=1,
            method="matrix_free",
            preconditioner="nystrom",
            preconditioner_rank=128,
            verbose=False,
        )

        train_args = gp._engine.train_calls[-1]
        assert train_args[9] == 128
        assert train_args[12] is True
        assert train_args[15] == 2
        assert gp.backend_train_info["precond_rank"] == 128
        assert gp.backend_train_info["use_preconditioner"] is True
