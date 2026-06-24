"""Integration tests for public prediction progress callbacks."""

import numpy as np
import pytest

from mojogp import MultiOutputGP, MultiOutputLMCGP, ProgressEvent, RBF, SingleOutputGP


def _single_output_data(n: int = 2000, d: int = 3, seed: int = 41):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.2 * x[:, 1] + 0.03 * rng.randn(n)).astype(np.float32)
    return x, y


def _multi_output_data(n: int = 2000, d: int = 3, tasks: int = 2, seed: int = 53):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype(np.float32)
    y = np.zeros((n, tasks), dtype=np.float32)
    y[:, 0] = np.sin(x[:, 0]) + 0.2 * x[:, 1]
    y[:, 1] = np.cos(x[:, 0]) - 0.1 * x[:, 1]
    y += 0.03 * rng.randn(n, tasks).astype(np.float32)
    return x, y


def _assert_prediction_progress_contract(
    events, *, model: str, route: str, n_test: int, variance_method: str = "love"
):
    phase = "mean" if variance_method == "mean_only" else "variance"
    assert [event.phase for event in events] == [
        "start",
        "backend",
        phase,
        "complete",
    ]
    assert all(isinstance(event, ProgressEvent) for event in events)
    assert {event.operation for event in events} == {"predict"}
    assert {event.model for event in events} == {model}
    assert {event.route for event in events} == {route}
    assert events[-1].stats["n_test"] == n_test
    assert events[-1].stats["variance_method"] == variance_method


def test_single_output_exact_prediction_progress_reports_backend_metadata():
    """SingleOutputGP exact prediction progress should include existing CG metadata."""
    x, y = _single_output_data()
    gp = SingleOutputGP(RBF())
    gp.fit(
        x,
        y,
        max_iterations=1,
        learning_rate=0.03,
        method="matrix_free",
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )
    events = []

    pred = gp.predict(
        x[:8],
        variance_method="exact",
        exact_prediction_block_cols=4,
        progress=events.append,
    )

    assert pred.mean.shape == (8,)
    assert pred.variance.shape == (8,)
    _assert_prediction_progress_contract(
        events,
        model="single_output",
        route="matrix_free",
        n_test=8,
        variance_method="exact",
    )
    assert events[-1].stats["variance_method"] == "exact"
    assert events[-1].stats["exact_cg_block_count"] >= 1


def test_multi_output_prediction_progress_reports_materialized_route():
    """MultiOutputGP prediction progress should preserve route metadata."""
    x, y = _multi_output_data(seed=67)
    gp = MultiOutputGP(
        kernel="rbf",
        task_rank=1,
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )
    gp.fit(x, y, max_iterations=1, learning_rate=0.03, method="materialized")
    events = []

    pred = gp.predict(x[:6], variance_method="love", progress=events.append)

    assert pred.mean.shape == (6, 2)
    _assert_prediction_progress_contract(
        events,
        model="multi_output_icm",
        route="materialized",
        n_test=6,
        variance_method="love",
    )
    assert events[-1].stats["variance_method"] == "love"


def test_lmc_prediction_progress_reports_matrix_free_route():
    """MultiOutputLMCGP prediction progress should preserve route metadata."""
    x, y = _multi_output_data(seed=79)
    gp = MultiOutputLMCGP(
        kernels=["rbf"],
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )
    gp.fit(x, y, max_iterations=1, learning_rate=0.03, method="matrix_free")
    events = []

    pred = gp.predict(x[:6], variance_method="love", progress=events.append)

    assert pred.mean.shape == (6, 2)
    _assert_prediction_progress_contract(
        events,
        model="multi_output_lmc",
        route="matrix_free",
        n_test=6,
        variance_method="love",
    )
    assert events[-1].stats["variance_method"] == "love"


@pytest.mark.parametrize("route", ["materialized", "matrix_free"])
def test_single_output_prediction_progress_reports_all_variance_methods(route):
    """SingleOutputGP should report progress for mean-only, LOVE, and exact prediction."""
    x, y = _single_output_data(seed=113)
    gp = SingleOutputGP(RBF())
    gp.fit(
        x,
        y,
        max_iterations=1,
        learning_rate=0.03,
        method=route,
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )

    for variance_method in ("mean_only", "love", "exact"):
        events = []
        pred = gp.predict(
            x[:6],
            variance_method=variance_method,
            exact_prediction_block_cols=3 if variance_method == "exact" else None,
            progress=events.append,
        )
        assert pred.mean.shape == (6,)
        _assert_prediction_progress_contract(
            events,
            model="single_output",
            route=route,
            n_test=6,
            variance_method=variance_method,
        )


@pytest.mark.parametrize(
    "model_family,route",
    [
        ("multi_output_icm", "materialized"),
        ("multi_output_icm", "matrix_free"),
        ("multi_output_lmc", "materialized"),
        ("multi_output_lmc", "matrix_free"),
    ],
)
def test_multi_output_prediction_progress_reports_all_model_route_variance_combinations(
    model_family, route
):
    """Multi-output wrappers should report progress across routes and variance modes."""
    x, y = _multi_output_data(seed=127)
    solver_kwargs = dict(
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )
    if model_family == "multi_output_icm":
        gp = MultiOutputGP(kernel="rbf", task_rank=1, **solver_kwargs)
    else:
        gp = MultiOutputLMCGP(kernels=["rbf"], **solver_kwargs)
    gp.fit(x, y, max_iterations=1, learning_rate=0.03, method=route)

    for variance_method in ("mean_only", "love", "exact"):
        events = []
        pred = gp.predict(x[:5], variance_method=variance_method, progress=events.append)
        assert pred.mean.shape == (5, 2)
        _assert_prediction_progress_contract(
            events,
            model=model_family,
            route=route,
            n_test=5,
            variance_method=variance_method,
        )


def test_single_output_latent_and_observed_prediction_progress_forwarding():
    """Convenience prediction wrappers should forward progress kwargs."""
    x, y = _single_output_data(seed=139)
    gp = SingleOutputGP(RBF())
    gp.fit(
        x,
        y,
        max_iterations=1,
        learning_rate=0.03,
        method="materialized",
        num_probes=1,
        max_cg_iterations=8,
        max_tridiag_iterations=5,
        preconditioner_rank=3,
    )

    latent_events = []
    gp.predict_latent(x[:4], variance_method="mean_only", progress=latent_events.append)
    _assert_prediction_progress_contract(
        latent_events,
        model="single_output",
        route="materialized",
        n_test=4,
        variance_method="mean_only",
    )

    observed_events = []
    gp.predict_observed(
        x[:4],
        observation_noise=np.full(4, 0.05, dtype=np.float32),
        variance_method="exact",
        exact_prediction_block_cols=2,
        progress=observed_events.append,
    )
    assert [event.phase for event in observed_events] == [
        "start",
        "backend",
        "variance",
        "observed_noise",
        "complete",
    ]
    assert observed_events[-1].stats["variance_method"] == "exact"
