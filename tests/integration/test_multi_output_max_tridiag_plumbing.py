"""Integration tests for MultiOutputGP train-time max_tridiag_iter plumbing."""

import numpy as np

from mojogp import MultiOutputGP
from mojogp.kernel import Kernel
from mojogp._multi_output_backend import destroy_provider_info
from mojogp.loader import load_engine, load_kernel_module_engine
from mojogp.kernel import RBF


def _make_data(n=2000, d=4, T=2, seed=123):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        Y[:, t] = (np.sin(X[:, 0] * (t + 1)) + 0.1 * rng.randn(n)).astype(np.float32)
    return X, Y


def _make_provider():
    X, Y = _make_data()
    params = np.array([1.0, 1.0], dtype=np.float32)
    noise_per_task = np.full(Y.shape[1], 0.1, dtype=np.float32)
    mean_per_task = np.mean(Y, axis=0).astype(np.float32)
    kernel_module = load_kernel_module_engine(RBF(), dim=X.shape[1], verbose=False)
    engine = load_engine(verbose=False)
    info = kernel_module.init_provider(X, params, 0.0)
    trainable_mask = np.ones_like(params, dtype=np.bool_)
    return kernel_module, engine, info, Y, params, trainable_mask, noise_per_task, mean_per_task


class TestMultiOutputMaxTridiagPlumbing:
    def test_train_multi_output_accepts_explicit_default_max_tridiag_iter(self):
        kernel_module, engine, info, Y, params, trainable_mask, noise_per_task, mean_per_task = _make_provider()

        try:
            result = engine.train_multi_output(
                info,
                Y,
                params,
                trainable_mask,
                noise_per_task,
                1.0,
                mean_per_task,
                Y.shape[1],
                1,
                0.05,
                -1,
                False,
                5,
                25,
                1.0,
                8,
                2,
                0.5,
                30,
                15,
                1e-4,
            )
        finally:
            destroy_provider_info(kernel_module, info)

        assert result["max_tridiag_iter"] == 30

    def test_train_multi_output_accepts_explicit_max_tridiag_iter(self):
        kernel_module, engine, info, Y, params, trainable_mask, noise_per_task, mean_per_task = _make_provider()

        try:
            result = engine.train_multi_output(
                info,
                Y,
                params,
                trainable_mask,
                noise_per_task,
                1.0,
                mean_per_task,
                Y.shape[1],
                1,
                0.05,
                -1,
                False,
                5,
                25,
                1.0,
                8,
                2,
                0.5,
                7,
                15,
                1e-4,
            )
        finally:
            destroy_provider_info(kernel_module, info)

        assert result["max_tridiag_iter"] == 7

    def test_train_multi_output_reports_adaptive_rebuild_metadata(self):
        kernel_module, engine, info, Y, params, trainable_mask, noise_per_task, mean_per_task = _make_provider()

        try:
            result = engine.train_multi_output(
                info,
                Y,
                params,
                trainable_mask,
                noise_per_task,
                1.0,
                mean_per_task,
                Y.shape[1],
                2,
                0.05,
                -1,
                False,
                2,
                10,
                1.0,
                5,
                2,
                0.0,
                5,
                15,
                1e-4,
            )
        finally:
            destroy_provider_info(kernel_module, info)

        assert result["precond_rebuild_threshold"] == 0.0
        assert result["precond_rebuild_count"] == 1

    def test_wrapper_fit_preserves_backend_rebuild_metadata(self):
        X, Y = _make_data(seed=321)
        gp = MultiOutputGP(
            kernel=Kernel.rbf(),
            task_rank=1,
            num_probes=2,
            max_cg_iterations=10,
            max_tridiag_iterations=5,
            preconditioner_rank=5,
            precond_rebuild_threshold=0.0,
        )

        gp.fit(
            X,
            Y,
            method="materialized",
            max_iterations=2,
            learning_rate=0.03,
            verbose=False,
            early_stop_patience=9,
            early_stop_tol=0.0,
        )

        assert gp._raw_result["precond_rebuild_threshold"] == 0.0
        assert gp._raw_result["precond_rebuild_count"] == 1
        assert gp._raw_result["training_route"] == "materialized"
        assert gp._raw_result["early_stop_patience"] == 9
        assert gp._raw_result["early_stop_tol"] == 0.0
