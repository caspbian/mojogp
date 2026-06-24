"""Pytest configuration for profile tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "minimal: Quick smoke tests (1-2 configs)")
    config.addinivalue_line("markers", "moderate: Medium coverage tests (3-4 configs)")
    config.addinivalue_line(
        "markers", "full: Comprehensive coverage tests (all configs)"
    )
