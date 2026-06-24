"""GPyTorch reference result cache for benchmark comparison suites.

Instead of running GPyTorch live during every benchmark comparison run, results are
pre-computed and cached in gpytorch_cache.json. This file provides utilities
to load and query the cache.

Cache regeneration:
    task generate-gpytorch-cache          # regenerate all suites
    task generate-gpytorch-cache-suite -- ground_truth  # single suite

The cache is committed to git so system tests can run without GPyTorch installed.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any


# Path to the cache file (next to this module)
CACHE_FILE = Path(__file__).parent / "gpytorch_cache.json"

# Module-level cache (loaded once on first access)
_cache: Optional[Dict[str, Any]] = None


def _load_cache() -> Dict[str, Any]:
    """Load cache from disk (once, then memoized)."""
    global _cache
    if _cache is not None:
        return _cache

    if not CACHE_FILE.exists():
        _cache = {}
        return _cache

    with open(CACHE_FILE, "r") as f:
        _cache = json.load(f)
    return _cache


def cache_available() -> bool:
    """Check if the cache file exists and has content."""
    return CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 10


def get_metadata() -> Dict[str, Any]:
    """Return cache metadata (GPyTorch version, generation date, etc.)."""
    cache = _load_cache()
    return cache.get("metadata", {})


def get_result(suite: str, config_key: str) -> Optional[Dict[str, Any]]:
    """Look up a cached GPyTorch result.

    Args:
        suite: Test suite name (e.g., "ground_truth", "ard", "composite_kernels")
        config_key: Deterministic key built from test config parameters

    Returns:
        Result dict if found, None otherwise
    """
    cache = _load_cache()
    suite_data = cache.get(suite, {})
    return suite_data.get(config_key)


def get_suite(suite: str) -> Dict[str, Any]:
    """Return all cached results for a test suite."""
    cache = _load_cache()
    return cache.get(suite, {})


def list_suites() -> list:
    """List all suites present in the cache."""
    cache = _load_cache()
    return [k for k in cache.keys() if k != "metadata"]


def build_key(**kwargs) -> str:
    """Build a deterministic cache key from config parameters.

    Sorts kwargs alphabetically and joins as key=value pairs separated by "::".

    Example:
        build_key(kernel="rbf", n=2000, d=5, source="gp_prior", seed=42)
        => "d=5::kernel=rbf::n=2000::seed=42::source=gp_prior"
    """
    parts = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        # Convert lists to sorted tuples for consistent hashing
        if isinstance(v, list):
            v = str(v)
        parts.append(f"{k}={v}")
    return "::".join(parts)


def invalidate_cache():
    """Force cache reload on next access (for testing)."""
    global _cache
    _cache = None
