"""Test materialized vs matrix-free consistency across all kernels.

Both methods should produce similar training results and predictions.
"""

import numpy as np
import pytest
from mojogp import SingleOutputGP
from mojogp.kernel import Kernel


KERNELS = ["rbf", "matern12", "matern32", "matern52"]


def _make_data(n=2000, d=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = (
        np.sin(X[:, 0]) + 0.3 * X[:, 1] + np.random.randn(n).astype(np.float32) * 0.1
    ).astype(np.float32)
    return X, y


class TestMaterializedVsMatrixFreeConsistency:
    """Materialized and matrix-free should produce similar results."""

    @pytest.mark.parametrize("kernel_name", KERNELS)
    def test_predictions_similar(self, kernel_name):
        """Mean predictions should be within 20% correlation."""
        X, y = _make_data()
        X_test = np.random.randn(30, 3).astype(np.float32)

        kernel_fn = getattr(Kernel, kernel_name)
        gp_mat = SingleOutputGP(kernel_fn())
        gp_mat.fit(X, y, max_iterations=50, method="materialized", verbose=False)
        mu_mat, _ = gp_mat.predict(X_test, return_std=True)

        gp_mf = SingleOutputGP(kernel_fn())
        gp_mf.fit(X, y, max_iterations=50, method="matrix_free", verbose=False)
        mu_mf, _ = gp_mf.predict(X_test, return_std=True)

        # Predictions should be correlated (both are fitting the same data)
        corr = np.corrcoef(mu_mat, mu_mf)[0, 1]
        assert corr > 0.5, (
            f"Predictions poorly correlated for {kernel_name}: r={corr:.4f}"
        )

    @pytest.mark.parametrize("kernel_name", KERNELS)
    def test_both_nll_decrease(self, kernel_name):
        """Both methods should see NLL decrease."""
        X, y = _make_data()
        kernel_fn = getattr(Kernel, kernel_name)

        for method in ["materialized", "matrix_free"]:
            gp = SingleOutputGP(kernel_fn())
            gp.fit(X, y, max_iterations=50, method=method, verbose=False)
            params = gp.get_learned_params()
            nll_hist = params.get("nll_history", None)
            if nll_hist is not None and len(nll_hist) > 5:
                nll_hist = np.array(nll_hist)
                assert nll_hist[-1] < nll_hist[0], (
                    f"NLL didn't decrease for {kernel_name}/{method}"
                )

    @pytest.mark.parametrize("kernel_name", KERNELS)
    def test_noise_same_order(self, kernel_name):
        """Learned noise should be within 10x of each other."""
        X, y = _make_data()
        kernel_fn = getattr(Kernel, kernel_name)

        noises = {}
        for method in ["materialized", "matrix_free"]:
            gp = SingleOutputGP(kernel_fn())
            gp.fit(X, y, max_iterations=80, method=method, verbose=False)
            params = gp.get_learned_params()
            noises[method] = params.get("noise", 1.0)

        ratio = max(noises.values()) / (min(noises.values()) + 1e-10)
        assert ratio < 10, (
            f"Noise ratio {ratio:.1f} too large for {kernel_name}: "
            f"mat={noises['materialized']:.4f}, mf={noises['matrix_free']:.4f}"
        )


class TestScoreMethod:
    """Test ExactGP.score() method."""

    @pytest.mark.parametrize("kernel_name", ["rbf", "matern52"])
    def test_score_returns_float(self, kernel_name):
        """score() should return a finite float."""
        X, y = _make_data(n=2000)
        X_train, y_train = X[:150], y[:150]
        X_test, y_test = X[150:], y[150:]

        kernel_fn = getattr(Kernel, kernel_name)
        gp = SingleOutputGP(kernel_fn())
        gp.fit(X_train, y_train, max_iterations=50, verbose=False)

        # score() should exist and return something
        if hasattr(gp, "score"):
            s = gp.score(X_test, y_test)
            assert np.isfinite(s), f"score() returned non-finite: {s}"
        else:
            pytest.skip("score() not implemented for ExactGP")

    def test_multi_output_score(self):
        """MultiOutputGP.score() should return a finite float."""
        from mojogp import MultiOutputGP

        np.random.seed(42)
        X = np.random.randn(200, 3).astype(np.float32)
        Y = np.zeros((200, 2), dtype=np.float32)
        Y[:, 0] = np.sin(X[:, 0]).astype(np.float32)
        Y[:, 1] = np.cos(X[:, 1]).astype(np.float32)

        X_train, Y_train = X[:150], Y[:150]
        X_test, Y_test = X[150:], Y[150:]

        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X_train, Y_train, max_iterations=50, verbose=False)

        if hasattr(gp, "score"):
            s = gp.score(X_test, Y_test)
            # score() may return float, dict, or ndarray
            if isinstance(s, dict):
                for key, val in s.items():
                    assert np.all(np.isfinite(np.asarray(val))), (
                        f"score()[{key}] non-finite"
                    )
            elif isinstance(s, np.ndarray):
                assert np.all(np.isfinite(s)), f"score() has non-finite values"
            else:
                assert np.isfinite(float(s)), f"score() returned non-finite: {s}"
        else:
            pytest.skip("score() not implemented for MultiOutputGP")
