"""Unit tests for public training progress reporting."""

import sys

import numpy as np
import pytest

from mojogp import ProgressEvent, RBF, SingleOutputGP
from mojogp.progress import resolve_progress_adapter
from mojogp.settings import get_progress_enabled, progress_enabled


class _FakeKernelModule:
    def init_provider(self, X, params, noise, **kwargs):
        return {
            "provider_ptr": 1,
            "n": int(X.shape[0]),
            "num_gradient_params": int(len(params)),
            "materialization_mode": 0,
            "is_ard": False,
        }

    def materialize(self, provider_info):
        provider_info["materialization_mode"] = 1


class _FakeEngine:
    def __init__(self):
        self.train_calls = []
        self.predict_calls = []

    def train(self, *args):
        self.train_calls.append(args)
        if len(args) > 23:
            callback = args[23]
            callback(
                {
                    "operation": "train",
                    "model": "single_output",
                    "route": "matrix_free",
                    "phase": "start",
                    "current": 0,
                    "total": int(args[4]),
                }
            )
            callback(
                {
                    "operation": "train",
                    "model": "single_output",
                    "route": "matrix_free",
                    "phase": "iteration",
                    "current": 1,
                    "total": int(args[4]),
                    "stats": {"nll": 1.23, "cg_iter": 7},
                }
            )
            callback(
                {
                    "operation": "train",
                    "model": "single_output",
                    "route": "matrix_free",
                    "phase": "complete",
                    "current": 1,
                    "total": int(args[4]),
                    "stats": {"nll": 1.23},
                    "converged": False,
                }
            )
        return {
            "params": np.array(args[2], dtype=np.float32),
            "noise": float(args[3]),
            "mean": 0.0,
            "final_nll": 1.23,
            "iterations": 1,
            "converged": False,
            "nll_history": [1.23],
            "training_route": "matrix_free",
            "materialization_mode": 0,
            "is_ard": False,
            "precond_method": int(args[15]),
            "precond_rank": int(args[9]),
            "max_tridiag_iter": int(args[13]),
            "precond_rebuild_threshold": float(args[14]),
            "use_preconditioner": bool(args[12]),
        }

    def predict(self, *args):
        self.predict_calls.append(args)
        n_test = int(args[2].shape[0])
        return {
            "mean": np.zeros(n_test, dtype=np.float32),
            "variance": np.ones(n_test, dtype=np.float32),
            "total_time_s": 0.25,
            "alpha_time_s": 0.05,
            "mean_time_s": 0.02,
            "variance_time_s": 0.18,
            "exact_cg_block_count": 2,
        }


class _RecordingReporter:
    def __init__(self):
        self.events = []

    def start(self, event):
        self.events.append(event)

    def update(self, event):
        self.events.append(event)

    def close(self, event):
        self.events.append(event)


def _make_data(n=2000, d=5):
    rng = np.random.RandomState(42)
    X = rng.randn(n, d).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.1 * rng.randn(n)).astype(np.float32)
    return X, y


def _make_gp():
    gp = SingleOutputGP(RBF())
    gp._kernel_module = _FakeKernelModule()
    gp._engine = _FakeEngine()
    gp._ensure_compiled = lambda: None
    return gp


def test_disabled_progress_does_not_pass_callback_to_engine():
    X, y = _make_data()
    gp = _make_gp()

    gp.fit(X, y, max_iterations=1, method="matrix_free", progress=False)

    assert len(gp._engine.train_calls[-1]) == 23


def test_progress_false_overrides_global_default():
    X, y = _make_data()
    gp = _make_gp()

    with progress_enabled(True):
        gp.fit(X, y, max_iterations=1, method="matrix_free", progress=False)

    assert len(gp._engine.train_calls[-1]) == 23


def test_callback_progress_passes_adapter_and_interval_to_engine():
    X, y = _make_data()
    gp = _make_gp()
    events = []

    gp.fit(
        X,
        y,
        max_iterations=5,
        method="matrix_free",
        progress=events.append,
        progress_interval=3,
    )

    train_args = gp._engine.train_calls[-1]
    assert len(train_args) == 25
    assert train_args[-1] == 3
    assert [event.phase for event in events] == ["start", "iteration", "complete"]
    assert all(isinstance(event, ProgressEvent) for event in events)
    assert events[1].route == "matrix_free"
    assert events[1].stats["cg_iter"] == 7
    assert events[-1].stats["converged"] is False


def test_reporter_object_progress_receives_training_events():
    X, y = _make_data()
    gp = _make_gp()
    reporter = _RecordingReporter()

    gp.fit(X, y, max_iterations=2, method="matrix_free", progress=reporter)

    assert [event.phase for event in reporter.events] == [
        "start",
        "iteration",
        "complete",
    ]
    assert all(isinstance(event, ProgressEvent) for event in reporter.events)


def test_invalid_progress_mode_raises_before_engine_call():
    X, y = _make_data()
    gp = _make_gp()

    with pytest.raises(ValueError, match="progress must"):
        gp.fit(X, y, max_iterations=1, method="matrix_free", progress=object())

    assert gp._engine.train_calls == []


def test_progress_interval_must_be_positive_before_engine_call():
    X, y = _make_data()
    gp = _make_gp()

    with pytest.raises(ValueError, match="progress_interval"):
        gp.fit(
            X,
            y,
            max_iterations=1,
            method="matrix_free",
            progress=lambda event: None,
            progress_interval=0,
        )

    assert gp._engine.train_calls == []


def test_global_progress_context_restores_previous_setting():
    previous = get_progress_enabled()

    with progress_enabled(True):
        assert get_progress_enabled() is True

    assert get_progress_enabled() == previous


def test_progress_is_enabled_by_default():
    assert get_progress_enabled() is True


def test_auto_progress_is_disabled_when_stderr_is_not_tty(monkeypatch):
    class _FakeStderr:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stderr", _FakeStderr())

    adapter = resolve_progress_adapter(
        "auto",
        operation="train",
        model="single_output",
        route="matrix_free",
    )

    assert adapter is None


def test_default_reporter_receives_progress_stats(monkeypatch):
    captured = {}

    class _FakeTqdmReporter:
        def __init__(self, *, desc, stats=None):
            captured["desc"] = desc
            captured["stats"] = stats
            self.events = []

        def start(self, event):
            self.events.append(event)

        def update(self, event):
            self.events.append(event)

        def close(self, event):
            self.events.append(event)

    monkeypatch.setattr("mojogp.progress.TqdmProgressReporter", _FakeTqdmReporter)
    stats = ("nll", "cg_iter")

    adapter = resolve_progress_adapter(
        True,
        operation="train",
        model="single_output",
        route="materialized",
        progress_stats=stats,
    )

    assert adapter is not None
    assert captured == {"desc": "Train", "stats": stats}


def test_default_reporter_formats_selected_float_stats():
    from mojogp.progress import ProgressEvent, TqdmProgressReporter

    reporter = TqdmProgressReporter(desc="Train", stats=("nll", "cg_iter", "iter_ms"))
    event = ProgressEvent(
        operation="train",
        model="single_output",
        route="materialized",
        phase="iteration",
        current=1,
        total=2,
        stats={"nll": 1.23456, "cg_iter": 30, "iter_ms": 7.89123, "noise": 0.12},
    )

    assert reporter._selected_stats(event) == {
        "nll": "1.23",
        "cg_iter": 30,
        "iter_ms": "7.89",
    }
    assert reporter._postfix_text(event) == "nll=1.23 | cg=30 | iter=7.89ms"


def test_default_reporter_tolerates_tqdm_without_postfix(monkeypatch):
    from mojogp.progress import ProgressEvent, TqdmProgressReporter

    class _MarimoProgress:
        def __init__(self):
            self.updates = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

    class _PatchedTqdm:
        def __init__(self, **kwargs):
            self.updates = []
            self.progress = _MarimoProgress()
            self.closed = False

        def update(self, delta=None, **kwargs):
            if kwargs:
                raise AssertionError("underlying Marimo progress object should be used")
            self.updates.append(delta)

        def set_description_str(self, text, refresh=True):
            raise AssertionError("subtitle update should be preferred")

        def close(self):
            self.closed = True

    class _Module:
        tqdm = _PatchedTqdm

    def fake_import(name, fromlist=(), level=0):
        if name == "tqdm.auto":
            return _Module
        return original_import(name, fromlist, level)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    reporter = TqdmProgressReporter(desc="Train")
    event = ProgressEvent(
        operation="train",
        model="single_output",
        route="materialized",
        phase="iteration",
        current=1,
        total=2,
        stats={"nll": 1.23, "cg_iter": 7},
    )

    reporter.update(event)

    assert reporter._bar.updates == [1]
    assert reporter._bar.progress.updates == [
        {"increment": 0, "title": "Train", "subtitle": "nll=1.23 | cg=7"}
    ]


def test_default_reporter_falls_back_to_description_without_subtitle(monkeypatch):
    from mojogp.progress import ProgressEvent, TqdmProgressReporter

    class _BasicTqdm:
        def __init__(self, **kwargs):
            self.updates = []
            self.descriptions = []

        def update(self, delta=None, **kwargs):
            if kwargs:
                raise TypeError("subtitle kwargs unsupported")
            self.updates.append(delta)

        def set_description_str(self, text, refresh=True):
            self.descriptions.append((text, refresh))

        def close(self):
            return None

    class _Module:
        tqdm = _BasicTqdm

    def fake_import(name, fromlist=(), level=0):
        if name == "tqdm.auto":
            return _Module
        return original_import(name, fromlist, level)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    reporter = TqdmProgressReporter(desc="Train")
    event = ProgressEvent(
        operation="train",
        model="single_output",
        route="materialized",
        phase="iteration",
        current=1,
        total=2,
        stats={"nll": 1.23, "cg_iter": 7},
    )

    reporter.update(event)

    assert reporter._bar.updates == [1]
    assert reporter._bar.descriptions == [("Train nll=1.23 | cg=7", True)]


def test_default_reporter_accepts_custom_stat_callable():
    from mojogp.progress import ProgressEvent, TqdmProgressReporter

    reporter = TqdmProgressReporter(
        desc="Train",
        stats=lambda event: {"loss": event.stats["nll"], "rank": event.stats["precond_rank"]},
    )
    event = ProgressEvent(
        operation="train",
        model="single_output",
        route="materialized",
        phase="iteration",
        current=1,
        total=2,
        stats={"nll": 1.23456, "precond_rank": 59},
    )

    assert reporter._selected_stats(event) == {"loss": "1.23", "rank": 59}


def test_engine_adapter_uses_wrapper_route_for_unknown_payload_route():
    events = []
    adapter = resolve_progress_adapter(
        events.append,
        operation="train",
        model="single_output",
        route="materialized",
    )
    assert adapter is not None

    adapter.callback(
        {
            "phase": "iteration",
            "route": "unknown",
            "current": 1,
            "total": 2,
            "stats": {"nll": 2.0},
        }
    )

    assert [event.phase for event in events] == ["start", "iteration"]
    assert events[-1].route == "materialized"


def test_disabled_prediction_progress_does_not_pass_callback_to_engine():
    X, y = _make_data()
    gp = _make_gp()
    gp.fit(X, y, max_iterations=1, method="matrix_free", progress=False)

    gp.predict(X[:4], variance_method="exact", progress=False)

    assert len(gp._engine.predict_calls[-1]) == 13


def test_prediction_callback_reports_wrapper_phase_events():
    X, y = _make_data()
    gp = _make_gp()
    gp.fit(X, y, max_iterations=1, method="matrix_free", progress=False)
    events = []

    gp.predict(X[:4], variance_method="exact", progress=events.append)

    assert [event.phase for event in events] == [
        "start",
        "backend",
        "variance",
        "complete",
    ]
    assert all(event.operation == "predict" for event in events)
    assert all(event.model == "single_output" for event in events)
    assert all(event.route == "matrix_free" for event in events)
    assert events[-1].stats["n_test"] == 4
    assert events[-1].stats["variance_method"] == "exact"
    assert events[-1].stats["exact_cg_block_count"] == 2


def test_prediction_reporter_object_receives_phase_events():
    X, y = _make_data()
    gp = _make_gp()
    gp.fit(X, y, max_iterations=1, method="matrix_free", progress=False)
    reporter = _RecordingReporter()

    gp.predict(X[:4], variance_method="mean_only", progress=reporter)

    assert [event.phase for event in reporter.events] == [
        "start",
        "backend",
        "mean",
        "complete",
    ]
    assert reporter.events[-1].stats["variance_method"] == "mean_only"
