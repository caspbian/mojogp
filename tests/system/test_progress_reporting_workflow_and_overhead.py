"""System tests for progress reporting workflow and overhead guardrails."""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest
import torch

from mojogp import ProgressEvent, RBF, SingleOutputGP
from tests.shared.benchmarking.environment import assert_gpu_available, requires_mojogp


TRAIN_OVERHEAD_MAX_RATIO = 1.35
TRAIN_OVERHEAD_MAX_ABS_S = 0.75
PREDICT_OVERHEAD_MAX_RATIO = 2.50
PREDICT_OVERHEAD_MAX_ABS_S = 0.25


def _make_data(n: int = 2000, d: int = 3, seed: int = 211):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.2 * x[:, 1] + 0.03 * rng.randn(n)).astype(np.float32)
    return x, y


def _sync_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(callable_):
    _sync_gpu()
    start = time.perf_counter()
    result = callable_()
    _sync_gpu()
    return time.perf_counter() - start, result


def _assert_low_overhead(*, disabled_s: float, enabled_s: float, ratio: float, abs_s: float):
    limit = disabled_s * ratio + abs_s
    assert enabled_s <= limit, (
        f"progress overhead too high: disabled={disabled_s:.4f}s, "
        f"enabled={enabled_s:.4f}s, limit={limit:.4f}s "
        f"(ratio={ratio}, abs_s={abs_s})"
    )


def _fit_once(x, y, *, method: str, progress):
    gp = SingleOutputGP(RBF())
    result = gp.fit(
        x,
        y,
        max_iterations=2,
        learning_rate=0.03,
        method=method,
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
        progress=progress,
        progress_interval=1,
    )
    assert np.isfinite(result.nll)
    return gp


@requires_mojogp
@pytest.mark.system
@pytest.mark.minimal
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_training_progress_callback_overhead_stays_below_guardrail(method):
    """Callback progress must not add significant training wall-time overhead."""
    assert_gpu_available()
    x, y = _make_data(seed=223)

    _fit_once(x, y, method=method, progress=False)

    disabled_times = []
    enabled_times = []
    enabled_events = []
    for seed in (227, 229):
        x_case, y_case = _make_data(seed=seed)
        disabled_s, _ = _timed(
            lambda: _fit_once(x_case, y_case, method=method, progress=False)
        )
        events = []
        enabled_s, _ = _timed(
            lambda: _fit_once(x_case, y_case, method=method, progress=events.append)
        )
        disabled_times.append(disabled_s)
        enabled_times.append(enabled_s)
        enabled_events.extend(events)

    disabled_median = statistics.median(disabled_times)
    enabled_median = statistics.median(enabled_times)
    _assert_low_overhead(
        disabled_s=disabled_median,
        enabled_s=enabled_median,
        ratio=TRAIN_OVERHEAD_MAX_RATIO,
        abs_s=TRAIN_OVERHEAD_MAX_ABS_S,
    )
    assert enabled_events
    assert all(isinstance(event, ProgressEvent) for event in enabled_events)
    assert {event.route for event in enabled_events} == {method}
    assert {event.operation for event in enabled_events} == {"train"}


@requires_mojogp
@pytest.mark.system
@pytest.mark.minimal
@pytest.mark.parametrize("method", ["materialized", "matrix_free"])
def test_prediction_progress_callback_overhead_stays_below_guardrail(method):
    """Callback progress must not add significant prediction wall-time overhead."""
    assert_gpu_available()
    x, y = _make_data(seed=239)
    gp = _fit_once(x, y, method=method, progress=False)
    x_test = np.ascontiguousarray(x[:32], dtype=np.float32)

    for variance_method in ("mean_only", "love", "exact"):
        kwargs = {
            "variance_method": variance_method,
            "exact_prediction_block_cols": 8 if variance_method == "exact" else None,
        }
        gp.predict(x_test, progress=False, **kwargs)

        disabled_times = []
        enabled_times = []
        enabled_events = []
        for _ in range(2):
            disabled_s, disabled_pred = _timed(
                lambda: gp.predict(x_test, progress=False, **kwargs)
            )
            events = []
            enabled_s, enabled_pred = _timed(
                lambda: gp.predict(x_test, progress=events.append, **kwargs)
            )
            np.testing.assert_allclose(enabled_pred.mean, disabled_pred.mean, rtol=1e-5, atol=1e-5)
            disabled_times.append(disabled_s)
            enabled_times.append(enabled_s)
            enabled_events.extend(events)

        disabled_median = statistics.median(disabled_times)
        enabled_median = statistics.median(enabled_times)
        _assert_low_overhead(
            disabled_s=disabled_median,
            enabled_s=enabled_median,
            ratio=PREDICT_OVERHEAD_MAX_RATIO,
            abs_s=PREDICT_OVERHEAD_MAX_ABS_S,
        )
        assert enabled_events
        assert all(isinstance(event, ProgressEvent) for event in enabled_events)
        assert {event.route for event in enabled_events} == {method}
        assert {event.operation for event in enabled_events} == {"predict"}
        assert {event.stats["variance_method"] for event in enabled_events} == {variance_method}


@requires_mojogp
@pytest.mark.system
@pytest.mark.minimal
def test_default_progress_bar_mode_runs_training_and_prediction():
    """The default progress bar path should run on a real GPU workflow."""
    assert_gpu_available()
    x, y = _make_data(seed=251)
    gp = SingleOutputGP(RBF())

    result = gp.fit(
        x,
        y,
        max_iterations=2,
        learning_rate=0.03,
        method="matrix_free",
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
        progress=True,
        progress_interval=1,
        progress_stats=("nll", "cg_iter", "iter_ms"),
    )
    pred = gp.predict(
        x[:16],
        variance_method="exact",
        exact_prediction_block_cols=8,
        progress=True,
        progress_stats=("n_test", "variance_method", "exact_cg_block_count"),
    )

    assert np.isfinite(result.nll)
    assert pred.mean.shape == (16,)
    assert pred.variance.shape == (16,)
