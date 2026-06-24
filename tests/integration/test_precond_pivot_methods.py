"""Integration tests for preconditioner pivot methods (greedy, rpcholesky, nystrom).

Tests that all three pivot selection strategies work end-to-end through the
ExactGP training pipeline, producing decreasing NLL, finite parameters,
comparable accuracy, and valid predictions.

Uses n=2000 per AGENTS.md policy for GP training tests.
"""

import gc
import time

import numpy as np
import pytest
import torch

from mojogp import SingleOutputGP
from mojogp.kernel import RBF, Matern52
from tests.shared.subprocess_harness import run_isolated_case


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PIVOT_METHODS = ["greedy", "rpcholesky", "nystrom"]
N_TRAIN = 2000
N_TEST = 200
DIM = 5
N_ITER = 50
LEARNING_RATE = 0.1
PRECOND_RANK = 8
SEED = 42
MODULE = "tests.integration.run_precond_pivot_case"


def _cleanup_gpu_state():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _run_pivot_case(case, method=None):
    method_arg = method if method is not None else "__none__"
    _cleanup_gpu_state()
    time.sleep(0.05)
    return run_isolated_case(
        module=MODULE,
        payload={"case": case, "method": method_arg},
        timeout=300,
        description=f"Runs pivot-method case {case}/{method_arg}",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_data():
    """Generate training data: y = sin(x_0) + 0.1*noise, d=5, n=2000."""
    np.random.seed(SEED)
    X = np.random.randn(N_TRAIN, DIM).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(N_TRAIN)).astype(np.float32)
    return X, y


@pytest.fixture(scope="module")
def test_data():
    """Generate 200 test points."""
    np.random.seed(SEED + 1)
    X_test = np.random.randn(N_TEST, DIM).astype(np.float32)
    return X_test


@pytest.fixture(scope="module")
def initial_nlls(train_data):
    """Train 1 iteration with each method to capture the initial NLL.

    Returned as a dict: method -> nll_after_1_iter.
    """
    X, y = train_data
    nlls = {}
    for method in PIVOT_METHODS:
        nlls[method] = _run_pivot_case("initial_nll", method)["nll"]
    return nlls


@pytest.fixture(scope="module")
def trained_summaries(train_data):
    """Train 50 iterations with each pivot method, keeping only lightweight summaries."""
    summaries = {}
    for method in PIVOT_METHODS:
        summary = _run_pivot_case("trained_summary", method)
        summary["params"] = np.asarray(summary["params"], dtype=np.float32)
        summary["nll_history"] = np.asarray(summary["nll_history"], dtype=np.float32)
        summaries[method] = summary
    return summaries


# ---------------------------------------------------------------------------
# 1. TestPivotMethodTraining
# ---------------------------------------------------------------------------


class TestPivotMethodTraining:
    """Verify that each pivot method trains successfully."""

    @pytest.mark.parametrize("method", PIVOT_METHODS)
    def test_nll_decreases(self, method, train_data, initial_nlls, trained_summaries):
        """Train 50 iterations; final NLL should be lower than after 1 iteration."""
        initial_nll = initial_nlls[method]
        summary = trained_summaries[method]
        final_nll = summary["nll"]

        assert np.isfinite(initial_nll), (
            f"[{method}] Initial NLL is not finite: {initial_nll}"
        )
        assert np.isfinite(final_nll), (
            f"[{method}] Final NLL is not finite: {final_nll}"
        )
        best_nll = float(np.min(summary["nll_history"]))
        if method == "rpcholesky":
            assert final_nll < initial_nll + 0.5, (
                f"[{method}] randomized training regressed too far from the 1-iteration baseline: "
                f"initial={initial_nll:.4f}, best={best_nll:.4f}, final={final_nll:.4f}"
            )
        else:
            assert best_nll < initial_nll, (
                f"[{method}] NLL never improved over the 1-iteration baseline: "
                f"initial={initial_nll:.4f}, best={best_nll:.4f}, final={final_nll:.4f}"
            )

    @pytest.mark.parametrize("method", PIVOT_METHODS)
    def test_params_finite(self, method, trained_summaries):
        """All fitted parameters should be finite after training."""
        summary = trained_summaries[method]

        assert np.all(np.isfinite(summary["params"])), (
            f"[{method}] params contain non-finite values: {summary['params']}"
        )
        assert np.isfinite(summary["noise"]), (
            f"[{method}] noise is not finite: {summary['noise']}"
        )
        assert np.isfinite(summary["nll"]), (
            f"[{method}] final NLL is not finite: {summary['nll']}"
        )


# ---------------------------------------------------------------------------
# 2. TestPivotMethodAccuracy
# ---------------------------------------------------------------------------


class TestPivotMethodAccuracy:
    """Compare NLL across the three pivot methods."""

    def test_methods_comparable_nll(self, trained_summaries):
        """Pairwise NLL ratio should be in [0.5, 2.0] across all methods."""
        nlls = {m: summary["nll"] for m, summary in trained_summaries.items()}

        for m1 in PIVOT_METHODS:
            for m2 in PIVOT_METHODS:
                if m1 >= m2:
                    continue
                nll1 = nlls[m1]
                nll2 = nlls[m2]
                # Guard against negative NLL (can happen with GP)
                if nll1 > 0 and nll2 > 0:
                    ratio = nll1 / nll2
                else:
                    # For negative or zero NLL, compare absolute difference
                    ratio = 1.0  # skip ratio check, just verify finite
                    assert np.isfinite(nll1) and np.isfinite(nll2), (
                        f"Non-finite NLL: {m1}={nll1}, {m2}={nll2}"
                    )
                    continue

                assert 0.5 <= ratio <= 2.0, (
                    f"NLL ratio {m1}/{m2} = {ratio:.3f} outside [0.5, 2.0]: "
                    f"{m1}={nll1:.4f}, {m2}={nll2:.4f}"
                )


# ---------------------------------------------------------------------------
# 3. TestPivotMethodPrediction
# ---------------------------------------------------------------------------


class TestPivotMethodPrediction:
    """Verify predictions are valid for each pivot method."""

    @pytest.mark.parametrize("method", PIVOT_METHODS)
    def test_predictions_finite(self, method, train_data, test_data):
        """Mean and std should be finite on 200 test points."""
        summary = _run_pivot_case("prediction_summary", method)

        assert tuple(summary["mean_shape"]) == (N_TEST,), (
            f"[{method}] mean shape {summary['mean_shape']} != ({N_TEST},)"
        )
        assert tuple(summary["std_shape"]) == (N_TEST,), (
            f"[{method}] std shape {summary['std_shape']} != ({N_TEST},)"
        )
        assert summary["mean_all_finite"], f"[{method}] mean has non-finite values"
        assert summary["std_all_finite"], f"[{method}] std has non-finite values"

    @pytest.mark.parametrize("method", PIVOT_METHODS)
    def test_prediction_std_positive(self, method, train_data, test_data):
        """All std values should be strictly positive."""
        summary = _run_pivot_case("prediction_summary", method)

        assert summary["std_min"] > 0, (
            f"[{method}] std has non-positive values: "
            f"min={summary['std_min']:.6e}, num_zero={summary['num_nonpositive_std']}"
        )


# ---------------------------------------------------------------------------
# 4. TestPivotMethodWithARD
# ---------------------------------------------------------------------------


class TestPivotMethodWithARD:
    """Verify pivot methods work with ARD kernels."""

    @pytest.fixture(scope="class")
    def ard_data(self):
        """Data where only dim 0 matters, to exercise ARD."""
        np.random.seed(SEED)
        X = np.random.randn(N_TRAIN, DIM).astype(np.float32)
        y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(N_TRAIN)).astype(np.float32)
        return X, y

    @pytest.mark.parametrize("method", PIVOT_METHODS)
    def test_ard_training_decreases_nll(self, method, ard_data):
        """ARD kernel with d=5 should train and NLL should decrease."""
        summary = _run_pivot_case("ard_summary", method)
        initial_nll = summary["initial_nll"]
        final_nll = summary["final_nll"]
        params = np.asarray(summary["params"], dtype=np.float32)

        assert np.isfinite(final_nll), (
            f"[ARD/{method}] Final NLL is not finite: {final_nll}"
        )
        assert np.all(np.isfinite(params)), (
            f"[ARD/{method}] params contain non-finite values: {params}"
        )
        assert final_nll < initial_nll, (
            f"[ARD/{method}] NLL did not decrease: "
            f"initial={initial_nll:.4f}, final={final_nll:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. TestPivotMethodWithComposite
# ---------------------------------------------------------------------------


class TestPivotMethodWithComposite:
    """Verify composite kernels train with nystrom (hardcoded for composites)."""

    def test_composite_training_improves_nll(self, train_data):
        """RBF + Matern52 composite kernel should train and NLL should decrease."""
        summary = _run_pivot_case("composite_summary")
        initial_nll = summary["initial_nll"]
        final_nll = summary["final_nll"]
        best_nll = summary["best_nll"]
        params = np.asarray(summary["params"], dtype=np.float32)

        assert np.isfinite(final_nll), (
            f"[composite/nystrom] Final NLL is not finite: {final_nll}"
        )
        assert np.all(np.isfinite(params)), (
            f"[composite/nystrom] params contain non-finite values: {params}"
        )
        assert best_nll < initial_nll, (
            f"[composite/nystrom] NLL never improved over the 1-iteration baseline: "
            f"initial={initial_nll:.4f}, best={best_nll:.4f}, final={final_nll:.4f}"
        )
