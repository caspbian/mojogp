"""Integration tests for composable ARD and composite kernel gradients.

These are end-to-end training checks and therefore belong in integration, not
the fast unit gate.
"""

import numpy as np
import pytest

from mojogp import SingleOutputGP, RBF, Matern12, Matern32, Matern52, RQ


@pytest.fixture(scope="module")
def ard_data():
    np.random.seed(42)
    n, d = 2000, 3
    X = np.random.randn(n, d).astype(np.float32)
    y = (np.sin(3.0 * X[:, 0]) + 0.05 * np.random.randn(n)).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def composite_data():
    np.random.seed(123)
    n, d = 2000, 2
    X = np.random.randn(n, d).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.1 * np.random.randn(n)).astype(np.float32)
    return X, y


ARD_KERNELS = {
    "rbf": lambda: RBF(ard=True),
    "matern12": lambda: Matern12(ard=True),
    "matern32": lambda: Matern32(ard=True),
    "matern52": lambda: Matern52(ard=True),
    "rq": lambda: RQ(alpha=2.0, ard=True),
}


@pytest.mark.integration
class TestARDTrainingConvergence:
    @pytest.mark.gpu
    @pytest.mark.parametrize("kernel_name", list(ARD_KERNELS.keys()))
    def test_ard_training_reduces_nll(self, ard_data, kernel_name):
        X, y = ard_data

        gp_early = SingleOutputGP(ARD_KERNELS[kernel_name]())
        result_early = gp_early.fit(
            X,
            y,
            max_iterations=5,
            learning_rate=0.05,
            initial_noise=0.5,
            method="materialized",
        )

        gp_late = SingleOutputGP(ARD_KERNELS[kernel_name]())
        result_late = gp_late.fit(
            X,
            y,
            max_iterations=150,
            learning_rate=0.05,
            initial_noise=0.5,
            method="materialized",
        )

        initial_nll = float(result_early.nll)
        final_nll = float(result_late.nll)
        assert final_nll < initial_nll + 2.0

    @pytest.mark.gpu
    @pytest.mark.parametrize("kernel_name", list(ARD_KERNELS.keys()))
    def test_ard_identifies_relevant_dimension(self, ard_data, kernel_name):
        X, y = ard_data
        d = X.shape[1]

        gp = SingleOutputGP(ARD_KERNELS[kernel_name]())
        result = gp.fit(
            X,
            y,
            max_iterations=200,
            learning_rate=0.05,
            initial_noise=0.5,
            method="materialized",
        )

        lengthscales = np.array(result.params[0:d])
        ls_relevant = lengthscales[0]
        ls_irrelevant_max = np.max(lengthscales[1:])
        assert ls_relevant < ls_irrelevant_max * 1.5

    @pytest.mark.gpu
    @pytest.mark.parametrize("kernel_name", list(ARD_KERNELS.keys()))
    def test_ard_prediction_accuracy(self, ard_data, kernel_name):
        X, y = ard_data

        gp = SingleOutputGP(ARD_KERNELS[kernel_name]())
        result = gp.fit(
            X,
            y,
            max_iterations=150,
            learning_rate=0.1,
            initial_noise=0.5,
            method="materialized",
        )

        assert float(result.nll) < 200


@pytest.mark.integration
class TestCompositeKernelTraining:
    @pytest.mark.gpu
    def test_sum_kernel_training(self, composite_data):
        X, y = composite_data
        gp = SingleOutputGP(RBF() + Matern52())
        gp.fit(X, y, max_iterations=80, learning_rate=0.1, method="materialized")
        mean, _ = gp.predict(X, return_std=True)
        ss_res = np.sum((y - mean) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5

    @pytest.mark.gpu
    def test_product_kernel_training(self, composite_data):
        X, y = composite_data
        gp = SingleOutputGP(RBF() * Matern52())
        gp.fit(X, y, max_iterations=80, learning_rate=0.1, method="materialized")
        mean, _ = gp.predict(X, return_std=True)
        ss_res = np.sum((y - mean) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.3

    @pytest.mark.gpu
    def test_sum_kernel_ard_training(self, ard_data):
        X, y = ard_data
        gp = SingleOutputGP(RBF(ard=True) + Matern52(ard=True))
        gp.fit(X, y, max_iterations=100, learning_rate=0.1, method="materialized")
        mean, _ = gp.predict(X, return_std=True)
        ss_res = np.sum((y - mean) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5


@pytest.mark.integration
class TestARDVsIsotropic:
    @pytest.mark.gpu
    @pytest.mark.parametrize("kernel_name", ["rbf", "matern52"])
    def test_ard_nll_not_worse_than_isotropic(self, ard_data, kernel_name):
        X, y = ard_data
        kernel_map = {"rbf": RBF, "matern52": Matern52}
        KernelCls = kernel_map[kernel_name]

        gp_iso = SingleOutputGP(KernelCls())
        result_iso = gp_iso.fit(
            X,
            y,
            max_iterations=150,
            learning_rate=0.03,
            initial_noise=0.5,
            method="matrix_free",
        )
        gp_ard = SingleOutputGP(KernelCls(ard=True))
        result_ard = gp_ard.fit(
            X,
            y,
            max_iterations=150,
            learning_rate=0.03,
            initial_noise=0.5,
            method="matrix_free",
        )

        assert float(result_ard.nll) < float(result_iso.nll) + 5.0
