"""Integration tests for real training progress callbacks."""

import numpy as np
import pytest

from mojogp import MultiOutputGP, MultiOutputLMCGP, ProgressEvent, RBF, SingleOutputGP


def _single_output_data(n: int = 2000, d: int = 3, seed: int = 11):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.2 * x[:, 1] + 0.03 * rng.randn(n)).astype(np.float32)
    return x, y


def _multi_output_data(n: int = 2000, d: int = 3, tasks: int = 2, seed: int = 23):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    y = np.zeros((n, tasks), dtype=np.float32)
    y[:, 0] = np.sin(x[:, 0]) + 0.2 * x[:, 1]
    y[:, 1] = np.cos(x[:, 0]) - 0.1 * x[:, 1]
    y += 0.03 * rng.randn(n, tasks).astype(np.float32)
    return x, y


def _assert_training_progress_contract(events, *, model: str, route: str, total: int):
    assert events, "expected progress events"
    assert all(isinstance(event, ProgressEvent) for event in events)
    assert events[0].phase == "start"
    assert events[-1].phase == "complete"
    assert {event.operation for event in events} == {"train"}
    assert {event.model for event in events} == {model}
    assert {event.route for event in events} == {route}
    assert all(event.total == total for event in events)

    iteration_events = [event for event in events if event.phase == "iteration"]
    assert iteration_events, "expected at least one iteration event"
    assert iteration_events[0].current == 1
    assert iteration_events[-1].current == total
    for event in iteration_events:
        assert np.isfinite(event.stats["nll"])
        assert "cg_iter" in event.stats
        assert event.stats["cg_iter"] >= 0
        assert "iter_ms" in event.stats
        assert event.stats["iter_ms"] >= 0


def test_single_output_training_progress_reports_interval_and_route_metadata():
    """SingleOutputGP progress should come from the real engine at iteration boundaries."""
    x, y = _single_output_data()
    events = []
    gp = SingleOutputGP(RBF())

    gp.fit(
        x,
        y,
        max_iterations=4,
        learning_rate=0.03,
        method="matrix_free",
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
        progress=events.append,
        progress_interval=2,
    )

    _assert_training_progress_contract(
        events, model="single_output", route="matrix_free", total=4
    )
    assert [event.current for event in events if event.phase == "iteration"] == [1, 2, 4]


def test_multi_output_training_progress_reports_materialized_route_metadata():
    """MultiOutputGP progress should preserve model and materialized route metadata."""
    x, y = _multi_output_data()
    events = []
    gp = MultiOutputGP(
        kernel="rbf",
        task_rank=1,
        num_probes=1,
        max_cg_iterations=100,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )

    gp.fit(
        x,
        y,
        max_iterations=2,
        learning_rate=0.03,
        method="materialized",
        progress=events.append,
    )

    _assert_training_progress_contract(
        events, model="multi_output_icm", route="materialized", total=2
    )


def test_lmc_training_progress_reports_matrix_free_route_metadata():
    """MultiOutputLMCGP progress should preserve model and matrix-free route metadata."""
    x, y = _multi_output_data(seed=37)
    events = []
    gp = MultiOutputLMCGP(
        kernels=["rbf"],
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )

    gp.fit(
        x,
        y,
        max_iterations=2,
        learning_rate=0.03,
        method="matrix_free",
        progress=events.append,
    )

    _assert_training_progress_contract(
        events, model="multi_output_lmc", route="matrix_free", total=2
    )


@pytest.mark.parametrize(
    "model_family,route",
    [
        ("single_output", "materialized"),
        ("single_output", "matrix_free"),
        ("multi_output_icm", "materialized"),
        ("multi_output_icm", "matrix_free"),
        ("multi_output_lmc", "materialized"),
        ("multi_output_lmc", "matrix_free"),
    ],
)
def test_training_progress_reports_all_model_route_combinations(model_family, route):
    """Every public training wrapper should report progress on both backend routes."""
    events = []
    solver_kwargs = dict(
        num_probes=1,
        max_cg_iterations=100,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )

    if model_family == "single_output":
        x, y = _single_output_data(seed=101)
        gp = SingleOutputGP(RBF())
        gp.fit(
            x,
            y,
            max_iterations=1,
            learning_rate=0.03,
            method=route,
            progress=events.append,
            **solver_kwargs,
        )
    elif model_family == "multi_output_icm":
        x, y = _multi_output_data(seed=103)
        gp = MultiOutputGP(kernel="rbf", task_rank=1, **solver_kwargs)
        gp.fit(
            x,
            y,
            max_iterations=1,
            learning_rate=0.03,
            method=route,
            progress=events.append,
        )
    else:
        x, y = _multi_output_data(seed=107)
        gp = MultiOutputLMCGP(kernels=["rbf"], **solver_kwargs)
        gp.fit(
            x,
            y,
            max_iterations=1,
            learning_rate=0.03,
            method=route,
            progress=events.append,
        )

    _assert_training_progress_contract(
        events, model=model_family, route=route, total=1
    )
