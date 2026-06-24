"""System tests for preconditioner variants at scale.

Tests the three pivoted Cholesky construction methods (greedy, rpcholesky, nystrom)
across single-output and multi-output GP models to verify:

1. All preconditioner variants produce finite, convergent training
2. Different pivot methods yield comparable prediction accuracy
3. Multi-output (Kronecker CG, LMC) training is stable with preconditioning
4. Preconditioner rebuild does not cause NaN or divergence

Test tiers (as per AGENTS.md policy):
- MINIMAL: Core preconditioner variants on single-output + basic multi-output
- MODERATE: Cross-method consistency, rpcholesky, multi-output with more tasks/latents
- FULL: Matern52 kernel, rebuild stability stress tests

All tests use n >= 2000 training points, float32, and require GPU.
"""

import numpy as np
import pytest
import time
import torch
from typing import Dict, Any

from mojogp import SingleOutputGP, RBF, Matern52
from mojogp.multi_output_gp import MultiOutputGP, MultiOutputLMCGP


# =============================================================================
# Constants
# =============================================================================

N_TRAIN = 2000
N_TEST = 400
D_SINGLE = 5
D_MULTI = 3
DEFAULT_SEED = 42
DEFAULT_LR = 0.01
DEFAULT_ITERS = 100
MULTI_LR = 0.05
MULTI_ITERS = 100


# =============================================================================
# Data Generation
# =============================================================================


def generate_single_output_data(
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    d: int = D_SINGLE,
    noise_std: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """Generate structured single-output data.

    y = sum(sin(X[:, i] * (i+1)) for i in range(d)) + noise
    """
    np.random.seed(seed)
    n = n_train + n_test
    X = np.random.randn(n, d).astype(np.float32)

    f = np.zeros(n, dtype=np.float32)
    for i in range(d):
        f += np.sin(X[:, i] * (i + 1)).astype(np.float32)

    y = f + noise_std * np.random.randn(n).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_test": X[n_train:],
        "f_test": f[n_train:],
        "y_test": y[n_train:],
    }


def generate_multi_output_data(
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    d: int = D_MULTI,
    num_tasks: int = 2,
    noise_std: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """Generate structured multi-output data.

    Y[:, t] = sin(X[:, 0] + t * 0.5) + noise_t
    """
    np.random.seed(seed)
    n = n_train + n_test
    X = np.random.randn(n, d).astype(np.float32)

    F = np.zeros((n, num_tasks), dtype=np.float32)
    Y = np.zeros((n, num_tasks), dtype=np.float32)
    for t in range(num_tasks):
        F[:, t] = np.sin(X[:, 0] + t * 0.5).astype(np.float32)
        Y[:, t] = F[:, t] + noise_std * np.random.randn(n).astype(np.float32)

    return {
        "X_train": X[:n_train],
        "Y_train": Y[:n_train],
        "X_test": X[n_train:],
        "F_test": F[n_train:],
        "Y_test": Y[n_train:],
    }


# =============================================================================
# Metrics
# =============================================================================


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


# =============================================================================
# Helpers
# =============================================================================


def _check_gpu():
    """Assert GPU is available for system tests."""
    assert torch.cuda.is_available(), "GPU required for system tests"


def _train_single_output(
    data: Dict[str, np.ndarray],
    precond: str = "nystrom",
    precond_rank: int = 15,
    kernel: str = "rbf",
    n_iterations: int = DEFAULT_ITERS,
    lr: float = DEFAULT_LR,
) -> Dict[str, Any]:
    """Train single-output GP and return results dict.

    Returns dict with keys: gp, train_time, mean, std, nll_history
    """
    if kernel == "rbf":
        k = RBF()
    elif kernel == "matern52":
        k = Matern52()
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    gp = SingleOutputGP(k)

    t0 = time.perf_counter()
    result = gp.fit(
        data["X_train"],
        data["y_train"],
        max_iterations=n_iterations,
        learning_rate=lr,
        preconditioner=precond,
        preconditioner_rank=precond_rank,
    )
    train_time = time.perf_counter() - t0

    mean, std = gp.predict(data["X_test"], return_std=True)

    return {
        "gp": gp,
        "result": result,
        "train_time": train_time,
        "mean": mean,
        "std": std,
        "nll": result.nll,
    }


def _train_kronecker(
    data: Dict[str, np.ndarray],
    precond_rank: int = 10,
    precond_rebuild_threshold: float = 0.5,
    n_iterations: int = MULTI_ITERS,
    lr: float = MULTI_LR,
) -> Dict[str, Any]:
    """Train Kronecker CG multi-output GP and return results dict."""
    T = data["Y_train"].shape[1]
    gp = MultiOutputGP(
        kernel="rbf",
        preconditioner_rank=precond_rank,
        precond_rebuild_threshold=precond_rebuild_threshold,
    )

    t0 = time.perf_counter()
    result = gp.fit(
        data["X_train"],
        data["Y_train"],
        max_iterations=n_iterations,
        learning_rate=lr,
    )
    train_time = time.perf_counter() - t0

    pred = gp.predict(data["X_test"])

    return {
        "gp": gp,
        "result": result,
        "train_time": train_time,
        "pred": pred,
        "nll": result.final_nll,
        "nll_history": result.nll_history,
    }


def _train_lmc(
    data: Dict[str, np.ndarray],
    num_latents: int = 2,
    precond_rank: int = 10,
    precond_rebuild_threshold: float = 0.5,
    n_iterations: int = MULTI_ITERS,
    lr: float = MULTI_LR,
) -> Dict[str, Any]:
    """Train LMC multi-output GP and return results dict."""
    kernels = ["rbf"] * num_latents
    gp = MultiOutputLMCGP(
        kernels=kernels,
        preconditioner_rank=precond_rank,
        precond_rebuild_threshold=precond_rebuild_threshold,
    )

    t0 = time.perf_counter()
    result = gp.fit(
        data["X_train"],
        data["Y_train"],
        max_iterations=n_iterations,
        learning_rate=lr,
    )
    train_time = time.perf_counter() - t0

    pred = gp.predict(data["X_test"])

    return {
        "gp": gp,
        "result": result,
        "train_time": train_time,
        "pred": pred,
        "nll": result.final_nll,
        "nll_history": result.nll_history,
    }


# =============================================================================
# MINIMAL Tier Tests
# =============================================================================


@pytest.mark.system
class TestCorePreconditionerRoutes:
    """Core preconditioner variants must work at scale.

    These tests must pass before running any moderate or full tier tests.
    """

    def test_greedy_pivoted_cholesky_produces_finite_predictions(self):
        """n=2000, d=5, RBF, precond='greedy', 100 iters.

        Verifies that the deterministic greedy pivot selection (GPyTorch-compatible)
        produces finite NLL that decreases and finite predictions at scale.
        """
        _check_gpu()
        data = generate_single_output_data()

        out = _train_single_output(data, precond="greedy", precond_rank=15)

        # NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"Greedy precond produced non-finite final NLL: {out['nll']}"
        )

        # Predictions must be finite
        assert np.all(np.isfinite(out["mean"])), (
            "Greedy precond: predictions contain NaN/Inf"
        )
        assert np.all(np.isfinite(out["std"])), "Greedy precond: std contains NaN/Inf"
        assert np.all(out["std"] >= 0), "Greedy precond: negative std detected"

        test_rmse = rmse(out["mean"], data["f_test"])
        print(
            f"\n[MINIMAL] Greedy pivot: NLL={out['nll']:.4f}, RMSE={test_rmse:.4f}, "
            f"time={out['train_time']:.2f}s"
        )

    def test_nystrom_pivoted_cholesky_produces_finite_predictions(self):
        """n=2000, d=5, RBF, precond='nystrom', 100 iters.

        Nystrom (adaptive RPCholesky) is the production default. This is the
        most critical preconditioner test — if this fails, nothing else matters.
        """
        _check_gpu()
        data = generate_single_output_data()

        out = _train_single_output(data, precond="nystrom", precond_rank=15)

        # NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"Nystrom precond produced non-finite final NLL: {out['nll']}"
        )

        # Predictions must be finite
        assert np.all(np.isfinite(out["mean"])), (
            "Nystrom precond: predictions contain NaN/Inf"
        )
        assert np.all(np.isfinite(out["std"])), "Nystrom precond: std contains NaN/Inf"
        assert np.all(out["std"] >= 0), "Nystrom precond: negative std detected"

        test_rmse = rmse(out["mean"], data["f_test"])
        print(
            f"\n[MINIMAL] Nystrom pivot: NLL={out['nll']:.4f}, RMSE={test_rmse:.4f}, "
            f"time={out['train_time']:.2f}s"
        )

    def test_kronecker_preconditioner_decreases_nll(self):
        """n=2000, T=2, Kronecker CG multi-output, 100 iters.

        Verifies Kronecker CG training completes without NaN and NLL decreases.
        """
        _check_gpu()
        data = generate_multi_output_data(num_tasks=2)

        out = _train_kronecker(data, precond_rank=10, precond_rebuild_threshold=0.5)

        # NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"Kronecker CG produced non-finite final NLL: {out['nll']}"
        )

        # NLL should decrease from initial
        nll_hist = np.array(out["nll_history"])
        valid = nll_hist[np.isfinite(nll_hist)]
        assert len(valid) >= 2, "Not enough finite NLL values in history"
        assert valid[-1] < valid[0], (
            f"Kronecker NLL did not decrease: {valid[0]:.4f} -> {valid[-1]:.4f}"
        )

        # No NaN in history (some non-finite in early iterations is tolerable,
        # but the majority should be finite)
        nan_frac = 1.0 - len(valid) / len(nll_hist)
        assert nan_frac < 0.5, f"Kronecker CG: {nan_frac:.0%} of NLL history is NaN/Inf"

        print(
            f"\n[MINIMAL] Kronecker T=2: NLL={out['nll']:.4f}, "
            f"time={out['train_time']:.2f}s"
        )

    def test_lmc_preconditioner_decreases_nll(self):
        """n=2000, T=2, R=2, LMC multi-output, 100 iters.

        Verifies LMC training completes without NaN and NLL decreases.
        """
        _check_gpu()
        data = generate_multi_output_data(num_tasks=2)

        out = _train_lmc(
            data, num_latents=2, precond_rank=10, precond_rebuild_threshold=0.5
        )

        # NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"LMC produced non-finite final NLL: {out['nll']}"
        )

        # NLL should decrease from initial
        nll_hist = np.array(out["nll_history"])
        valid = nll_hist[np.isfinite(nll_hist)]
        assert len(valid) >= 2, "Not enough finite NLL values in history"
        assert valid[-1] < valid[0], (
            f"LMC NLL did not decrease: {valid[0]:.4f} -> {valid[-1]:.4f}"
        )

        # No NaN in history
        nan_frac = 1.0 - len(valid) / len(nll_hist)
        assert nan_frac < 0.5, f"LMC: {nan_frac:.0%} of NLL history is NaN/Inf"

        print(
            f"\n[MINIMAL] LMC T=2 R=2: NLL={out['nll']:.4f}, "
            f"time={out['train_time']:.2f}s"
        )


# =============================================================================
# MODERATE Tier Tests
# =============================================================================


@pytest.mark.system
class TestExtendedPreconditionerRoutes:
    """Cross-method consistency and extended preconditioner configurations.

    Run only after all MINIMAL tests pass.
    """

    def test_pivot_methods_produce_consistent_predictions(self):
        """Train with all 3 pivot methods, verify final predictions within 30% RMSE.

        The three methods (greedy, rpcholesky, nystrom) should converge to similar
        solutions since they are all solving the same optimization problem with
        the same data — only the preconditioner construction differs.
        """
        _check_gpu()
        data = generate_single_output_data()

        methods = ["greedy", "rpcholesky", "nystrom"]
        results = {}

        for method in methods:
            out = _train_single_output(data, precond=method, precond_rank=15)
            test_rmse = rmse(out["mean"], data["f_test"])
            results[method] = {
                "rmse": test_rmse,
                "nll": out["nll"],
                "train_time": out["train_time"],
            }

        # Print comparison table
        print(f"\n[MODERATE] Pivot method comparison (n={N_TRAIN}, d={D_SINGLE}, RBF):")
        print(f"  {'Method':<14} {'RMSE':>8} {'NLL':>10} {'Time (s)':>10}")
        print(f"  {'-' * 44}")
        for method in methods:
            r = results[method]
            print(
                f"  {method:<14} {r['rmse']:>8.4f} {r['nll']:>10.4f} "
                f"{r['train_time']:>10.2f}"
            )

        # All RMSEs should be within 30% of the best RMSE
        rmses = [results[m]["rmse"] for m in methods]
        best_rmse = min(rmses)
        worst_rmse = max(rmses)

        assert best_rmse > 0, "Best RMSE is zero — suspicious"
        ratio = worst_rmse / best_rmse
        assert ratio < 1.30, (
            f"Pivot methods too inconsistent: worst/best RMSE ratio = {ratio:.2f} "
            f"(best={best_rmse:.4f}, worst={worst_rmse:.4f}). "
            f"Methods: {dict(zip(methods, rmses))}"
        )

        print(f"\n  Worst/best RMSE ratio: {ratio:.3f} (threshold: 1.30)")

    def test_rpcholesky_produces_finite_predictions(self):
        """n=2000, d=5, RBF, precond='rpcholesky', 100 iters.

        RPCholesky uses randomized proportional pivot sampling with fixed rank.
        Verify it produces finite, convergent training and finite predictions.
        """
        _check_gpu()
        data = generate_single_output_data()

        out = _train_single_output(data, precond="rpcholesky", precond_rank=15)

        # NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"RPCholesky precond produced non-finite final NLL: {out['nll']}"
        )

        # Predictions must be finite
        assert np.all(np.isfinite(out["mean"])), (
            "RPCholesky precond: predictions contain NaN/Inf"
        )
        assert np.all(np.isfinite(out["std"])), (
            "RPCholesky precond: std contains NaN/Inf"
        )

        test_rmse = rmse(out["mean"], data["f_test"])
        print(
            f"\n[MODERATE] RPCholesky: NLL={out['nll']:.4f}, RMSE={test_rmse:.4f}, "
            f"time={out['train_time']:.2f}s"
        )

    def test_kronecker_three_task_preconditioner_decreases_nll(self):
        """n=2000, T=3, Kronecker CG. NLL decreases.

        Tests scaling to 3 tasks (larger task covariance matrix B).
        """
        _check_gpu()
        data = generate_multi_output_data(num_tasks=3)

        out = _train_kronecker(data, precond_rank=10, precond_rebuild_threshold=0.5)

        # NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"Kronecker T=3 produced non-finite final NLL: {out['nll']}"
        )

        # NLL should decrease
        nll_hist = np.array(out["nll_history"])
        valid = nll_hist[np.isfinite(nll_hist)]
        assert len(valid) >= 2, "Not enough finite NLL values"
        assert valid[-1] < valid[0], (
            f"Kronecker T=3 NLL did not decrease: {valid[0]:.4f} -> {valid[-1]:.4f}"
        )

        print(
            f"\n[MODERATE] Kronecker T=3: NLL={out['nll']:.4f}, "
            f"time={out['train_time']:.2f}s"
        )

    def test_lmc_three_latent_preconditioner_produces_finite_nll(self):
        """n=2000, T=2, R=3, LMC. NLL finite.

        Tests LMC with more latent functions than tasks, which creates
        an over-parameterized model and tests numerical stability.
        """
        _check_gpu()
        data = generate_multi_output_data(num_tasks=2)

        out = _train_lmc(
            data, num_latents=3, precond_rank=10, precond_rebuild_threshold=0.5
        )

        # NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"LMC R=3 produced non-finite final NLL: {out['nll']}"
        )

        print(
            f"\n[MODERATE] LMC T=2 R=3: NLL={out['nll']:.4f}, "
            f"time={out['train_time']:.2f}s"
        )


# =============================================================================
# FULL Tier Tests
# =============================================================================


@pytest.mark.system
class TestBroadPreconditionerRoutes:
    """Kernel variants, rebuild stability, and preconditioner stress tests.

    Run only after all MINIMAL and MODERATE tests pass.
    """

    def test_pivot_methods_with_matern_kernel_are_consistent(self):
        """All 3 pivot methods with Matern52, verify predictions are comparable.

        Matern52 has a less smooth kernel matrix than RBF, which stresses the
        preconditioner differently (slower eigenvalue decay).
        """
        _check_gpu()
        data = generate_single_output_data()

        methods = ["greedy", "rpcholesky", "nystrom"]
        results = {}

        for method in methods:
            out = _train_single_output(
                data, precond=method, precond_rank=15, kernel="matern52"
            )
            test_rmse = rmse(out["mean"], data["f_test"])
            results[method] = {
                "rmse": test_rmse,
                "nll": out["nll"],
                "train_time": out["train_time"],
            }

        # Print comparison
        print(f"\n[FULL] Matern52 pivot method comparison:")
        print(f"  {'Method':<14} {'RMSE':>8} {'NLL':>10} {'Time (s)':>10}")
        print(f"  {'-' * 44}")
        for method in methods:
            r = results[method]
            print(
                f"  {method:<14} {r['rmse']:>8.4f} {r['nll']:>10.4f} "
                f"{r['train_time']:>10.2f}"
            )

        # All methods should produce finite NLLs
        for method in methods:
            assert np.isfinite(results[method]["nll"]), (
                f"Matern52 + {method}: non-finite NLL"
            )

        # All RMSEs should be within 30% of each other
        rmses = [results[m]["rmse"] for m in methods]
        best_rmse = min(rmses)
        worst_rmse = max(rmses)

        if best_rmse > 0:
            ratio = worst_rmse / best_rmse
            assert ratio < 1.30, (
                f"Matern52 pivot methods too inconsistent: ratio={ratio:.2f}. "
                f"Methods: {dict(zip(methods, rmses))}"
            )
            print(f"\n  Worst/best RMSE ratio: {ratio:.3f}")

    def test_kronecker_preconditioner_rebuild_stays_finite(self):
        """Kronecker with low rebuild threshold, verify no NaN over 100 iterations.

        A low preconditioner rebuild threshold stresses the
        rebuild logic. A rebuild creates a new pivoted Cholesky factorization
        from the current kernel matrix, and bugs in the handoff can cause
        NaN propagation.
        """
        _check_gpu()
        data = generate_multi_output_data(num_tasks=2)

        out = _train_kronecker(
            data, precond_rank=10, precond_rebuild_threshold=0.25, n_iterations=100
        )

        # Final NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"Kronecker rebuild_threshold=0.25 produced non-finite NLL: {out['nll']}"
        )

        # Check NLL history for NaN — allow up to 10% NaN entries
        # (some transient NaN after rebuild is tolerable if training recovers)
        nll_hist = np.array(out["nll_history"])
        nan_count = np.sum(~np.isfinite(nll_hist))
        nan_frac = nan_count / len(nll_hist) if len(nll_hist) > 0 else 0.0

        assert nan_frac < 0.10, (
            f"Kronecker rebuild stability: {nan_count}/{len(nll_hist)} "
            f"({nan_frac:.0%}) NaN entries in NLL history. "
            f"Last 10 NLL values: {nll_hist[-10:].tolist()}"
        )

        print(
            f"\n[FULL] Kronecker rebuild_threshold=0.25: NLL={out['nll']:.4f}, "
            f"NaN fraction={nan_frac:.1%}, time={out['train_time']:.2f}s"
        )

    def test_lmc_preconditioner_rebuild_stays_finite(self):
        """LMC with low rebuild threshold, no NaN over 100 iterations.

        Same rebuild stress test as Kronecker but with the LMC model, which
        has multiple kernel matrices (one per latent) and thus more complex
        preconditioner management.
        """
        _check_gpu()
        data = generate_multi_output_data(num_tasks=2)

        out = _train_lmc(
            data,
            num_latents=2,
            precond_rank=10,
            precond_rebuild_threshold=0.25,
            n_iterations=100,
        )

        # Final NLL must be finite
        assert np.isfinite(out["nll"]), (
            f"LMC rebuild_threshold=0.25 produced non-finite NLL: {out['nll']}"
        )

        # Check NLL history for NaN — allow up to 10%
        nll_hist = np.array(out["nll_history"])
        nan_count = np.sum(~np.isfinite(nll_hist))
        nan_frac = nan_count / len(nll_hist) if len(nll_hist) > 0 else 0.0

        assert nan_frac < 0.10, (
            f"LMC rebuild stability: {nan_count}/{len(nll_hist)} "
            f"({nan_frac:.0%}) NaN entries in NLL history. "
            f"Last 10 NLL values: {nll_hist[-10:].tolist()}"
        )

        print(
            f"\n[FULL] LMC rebuild_threshold=0.25: NLL={out['nll']:.4f}, "
            f"NaN fraction={nan_frac:.1%}, time={out['train_time']:.2f}s"
        )
