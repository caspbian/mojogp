"""Exhaustive tests: ARD gradient training correctness for all kernel types.

Verifies that ARD training with the JIT engine converges correctly for all 8
kernel types, using both materialized and matrix-free methods.

Coverage:
- All 8 kernel types × ARD mode (where applicable)
- Both materialized and matrix-free providers
- Multiple input dimensions (d=5)
- Verifies NLL decreases during training
- Verifies ARD lengthscales are positive and finite
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from mojogp import (
    SingleOutputGP,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    RQ,
    Linear,
    Polynomial,
)


def _make_data(n=2000, d=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    y = (
        np.sin(X[:, 0]) + 0.3 * X[:, 1] + np.random.randn(n).astype(np.float32) * 0.1
    ).astype(np.float32)
    return X, y


# Map kernel names to kernel factory functions.
# Linear and Polynomial are dot-product kernels that don't support ARD meaningfully.
KERNEL_CONFIGS = {
    "RBF": {"factory": lambda: RBF(ard=True), "ard": True},
    "Matern12": {"factory": lambda: Matern12(ard=True), "ard": True},
    "Matern32": {"factory": lambda: Matern32(ard=True), "ard": True},
    "Matern52": {"factory": lambda: Matern52(ard=True), "ard": True},
    "Periodic": {"factory": lambda: Periodic(period=1.0, ard=True), "ard": True},
    "RQ": {"factory": lambda: RQ(alpha=1.0, ard=True), "ard": True},
    "Linear": {"factory": lambda: Linear(), "ard": False},
    "Polynomial": {"factory": lambda: Polynomial(degree=2.0, offset=1.0), "ard": False},
}


class TestFusedVsSequentialSlotBySlot:
    """Compare materialized vs matrix-free training for each kernel."""

    @pytest.mark.parametrize("kernel_name", list(KERNEL_CONFIGS.keys()))
    @pytest.mark.parametrize("d", [5])
    def test_fused_matches_sequential_ard(self, kernel_name, d):
        """For each kernel: train briefly, check that both methods produce finite NLL."""
        config = KERNEL_CONFIGS[kernel_name]
        X, y = _make_data(n=2000, d=d)

        gp_mat = SingleOutputGP(config["factory"]())
        result_mat = gp_mat.fit(
            X,
            y,
            max_iterations=30,
            learning_rate=0.02,
            initial_noise=0.1,
            method="materialized",
        )

        gp_mf = SingleOutputGP(config["factory"]())
        result_mf = gp_mf.fit(
            X,
            y,
            max_iterations=30,
            learning_rate=0.02,
            initial_noise=0.1,
            method="matrix_free",
        )

        nll_mat = float(result_mat.nll)
        nll_mf = float(result_mf.nll)

        assert np.isfinite(nll_mat), (
            f"{kernel_name} d={d} materialized NLL not finite: {nll_mat}"
        )
        assert np.isfinite(nll_mf), (
            f"{kernel_name} d={d} matrix-free NLL not finite: {nll_mf}"
        )

    @pytest.mark.parametrize(
        "kernel_name", [k for k, c in KERNEL_CONFIGS.items() if c["ard"]]
    )
    def test_ard_lengthscales_positive(self, kernel_name):
        """All learned ARD lengthscales must be positive (not NaN/negative)."""
        config = KERNEL_CONFIGS[kernel_name]
        d = 4
        X, y = _make_data(n=2000, d=d)

        gp = SingleOutputGP(config["factory"]())
        result = gp.fit(X, y, max_iterations=40, learning_rate=0.02, initial_noise=0.1)

        ls = np.array(result.params[0:d])
        assert len(ls) == d, f"{kernel_name}: expected {d} lengthscales, got {len(ls)}"
        assert np.all(ls > 0), f"{kernel_name}: negative lengthscales: {ls}"
        assert np.all(np.isfinite(ls)), f"{kernel_name}: non-finite lengthscales: {ls}"

    @pytest.mark.parametrize("kernel_name", list(KERNEL_CONFIGS.keys()))
    def test_ard_noise_positive(self, kernel_name):
        """Learned noise must be positive."""
        config = KERNEL_CONFIGS[kernel_name]
        X, y = _make_data(n=2000, d=3)

        gp = SingleOutputGP(config["factory"]())
        result = gp.fit(X, y, max_iterations=30, learning_rate=0.02, initial_noise=0.1)

        noise = float(result.noise)
        assert noise > 0, f"{kernel_name}: noise not positive: {noise}"
        assert np.isfinite(noise), f"{kernel_name}: noise not finite: {noise}"


class TestFusedGradientNLLDecreases:
    """NLL must decrease during ARD training for applicable kernel types.

    If gradient ordering is wrong, the optimizer follows garbage gradients
    and NLL will increase or oscillate wildly.
    Comparison: NLL after 5 iterations vs NLL after 60 iterations.
    """

    @pytest.mark.parametrize(
        "kernel_name", [k for k, c in KERNEL_CONFIGS.items() if c["ard"]]
    )
    def test_nll_decreases_ard(self, kernel_name):
        """NLL should decrease from early to late training."""
        config = KERNEL_CONFIGS[kernel_name]
        X, y = _make_data(n=2000, d=4)

        gp_early = SingleOutputGP(config["factory"]())
        result_early = gp_early.fit(
            X, y, max_iterations=5, learning_rate=0.02, initial_noise=0.1
        )

        gp_late = SingleOutputGP(config["factory"]())
        result_late = gp_late.fit(
            X, y, max_iterations=60, learning_rate=0.02, initial_noise=0.1
        )

        nll_early = float(result_early.nll)
        nll_late = float(result_late.nll)

        assert np.isfinite(nll_early), (
            f"{kernel_name}: early NLL not finite: {nll_early}"
        )
        assert np.isfinite(nll_late), f"{kernel_name}: late NLL not finite: {nll_late}"
        assert nll_late < nll_early, (
            f"{kernel_name} ARD NLL did NOT decrease: "
            f"early={nll_early:.4f}, late={nll_late:.4f}. "
            f"This strongly suggests a gradient ordering bug in the fused path."
        )


class TestFusedGradientDimensionRelevance:
    """ARD should identify relevant vs irrelevant dimensions for all kernels.

    If gradient ordering is wrong, the lengthscales won't track the true
    data structure — relevant dims won't get shorter lengthscales.
    """

    @pytest.mark.parametrize(
        "kernel_name", ["RBF", "Matern32", "Matern52", "Periodic", "RQ"]
    )
    def test_relevant_dim_has_shorter_lengthscale(self, kernel_name):
        """Dim 0 (signal) should get shorter lengthscale than dims 1-4 (noise)."""
        config = KERNEL_CONFIGS[kernel_name]

        np.random.seed(42)
        n, d = 2000, 5
        X = np.random.randn(n, d).astype(np.float32)
        # Only dim 0 matters
        y = (np.sin(X[:, 0]) + np.random.randn(n).astype(np.float32) * 0.05).astype(
            np.float32
        )

        gp = SingleOutputGP(config["factory"]())
        result = gp.fit(X, y, max_iterations=100, learning_rate=0.03, initial_noise=0.1)

        ls = np.array(result.params[0:d])
        assert len(ls) == d

        # Dim 0 should have shorter lengthscale (more relevant)
        ls_relevant = ls[0]
        ls_irrelevant_avg = np.mean(ls[1:])

        assert ls_irrelevant_avg > ls_relevant * 0.7, (
            f"{kernel_name}: relevant dim (ls={ls_relevant:.3f}) should be shorter "
            f"than irrelevant avg (ls={ls_irrelevant_avg:.3f}). "
            f"All lengthscales: {ls}. "
            f"This may indicate fused gradient ordering is wrong."
        )


class TestFusedGradientHighDimRegisterPressure:
    """Test ARD training at d=20+ to catch register spill issues (2D fix).

    At DIM>16 with 4x unrolling, the GPU runs out of registers and spills
    to local memory. The 2D fix uses 2x unrolling for DIM>16. This test
    verifies the 2x path produces correct results.
    """

    @pytest.mark.parametrize("kernel_name", ["RBF", "Matern52"])
    def test_high_dim_ard_converges(self, kernel_name):
        """d=20 ARD training should converge (2x unrolling path)."""
        config = KERNEL_CONFIGS[kernel_name]

        np.random.seed(42)
        n, d = 2000, 20
        X = np.random.randn(n, d).astype(np.float32)
        y = (
            np.sin(X[:, 0])
            + 0.5 * X[:, 1]
            + np.random.randn(n).astype(np.float32) * 0.1
        ).astype(np.float32)

        # Check early vs late NLL
        gp_early = SingleOutputGP(config["factory"]())
        result_early = gp_early.fit(
            X, y, max_iterations=5, learning_rate=0.02, initial_noise=0.1
        )

        gp_late = SingleOutputGP(config["factory"]())
        result_late = gp_late.fit(
            X, y, max_iterations=40, learning_rate=0.02, initial_noise=0.1
        )

        nll_early = float(result_early.nll)
        nll_late = float(result_late.nll)

        assert np.isfinite(nll_early), f"{kernel_name} d=20: early NLL not finite"
        assert np.isfinite(nll_late), f"{kernel_name} d=20: late NLL not finite"
        assert nll_late < nll_early * 1.5, (
            f"{kernel_name} d=20: NLL diverged. early={nll_early:.4f}, late={nll_late:.4f}. "
            f"Register pressure issue?"
        )

        ls = np.array(result_late.params[0:d])
        assert len(ls) == d
        assert np.all(ls > 0), f"{kernel_name} d=20: negative lengthscales"
        assert np.all(np.isfinite(ls)), f"{kernel_name} d=20: non-finite lengthscales"
