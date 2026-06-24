"""Pytest configuration for system tests.

Provides fixtures for results writing and GPU memory monitoring.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.system.results_writer import ResultsWriter, get_gpu_memory_mb


# Session-scoped results writer
_session_writer = None


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "minimal: minimal coverage tests")
    config.addinivalue_line("markers", "moderate: moderate coverage tests")
    config.addinivalue_line("markers", "full: full coverage tests")
    config.addinivalue_line("markers", "system: system tests")
    config.addinivalue_line("markers", "ground_truth: ground truth comparison tests")


def pytest_sessionstart(session):
    """Called after the Session object has been created."""
    global _session_writer
    _session_writer = ResultsWriter(prefix="system_test")


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished."""
    global _session_writer
    if _session_writer is not None:
        _session_writer.write_summary()
        print(f"\n{'=' * 60}")
        print(f"Results saved to: {_session_writer.get_session_dir()}")
        print(f"{'=' * 60}")


@pytest.fixture(scope="session")
def results_writer():
    """Fixture providing the session results writer."""
    global _session_writer
    if _session_writer is None:
        _session_writer = ResultsWriter(prefix="system_test")
    return _session_writer


@pytest.fixture(scope="session")
def results_dir(results_writer):
    """Session results directory for moved system benchmark-style tests."""
    return results_writer.get_session_dir()


@pytest.fixture
def gpu_memory_tracker():
    """Fixture for tracking GPU memory during a test."""

    class GPUMemoryTracker:
        def __init__(self):
            self.before_mb = 0
            self.after_mb = 0
            self.peak_mb = 0

        def start(self):
            self.before_mb = get_gpu_memory_mb()
            self.peak_mb = self.before_mb

        def update(self):
            current = get_gpu_memory_mb()
            if current > self.peak_mb:
                self.peak_mb = current

        def stop(self):
            self.after_mb = get_gpu_memory_mb()
            if self.after_mb > self.peak_mb:
                self.peak_mb = self.after_mb

        def get_stats(self):
            return {
                "before_mb": self.before_mb,
                "after_mb": self.after_mb,
                "peak_mb": self.peak_mb,
                "used_mb": self.peak_mb - self.before_mb,
            }

    return GPUMemoryTracker()
