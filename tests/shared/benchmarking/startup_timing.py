"""Helpers for separating wrapper startup/JIT timing from benchmark timings."""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


@contextmanager
def temporary_jit_cache(prefix: str):
    """Point MojoGP JIT compilation at a temporary cache directory."""

    original_cache = os.environ.get("MOJOGP_JIT_CACHE_DIR")
    try:
        with tempfile.TemporaryDirectory(prefix=prefix) as cache_root:
            os.environ["MOJOGP_JIT_CACHE_DIR"] = cache_root
            yield cache_root
    finally:
        if original_cache is None:
            os.environ.pop("MOJOGP_JIT_CACHE_DIR", None)
        else:
            os.environ["MOJOGP_JIT_CACHE_DIR"] = original_cache


def measure_prepare_time(factory: Callable[[], T]) -> tuple[T, float]:
    """Construct an object and measure that preparation time."""

    start = time.perf_counter()
    obj = factory()
    return obj, float(time.perf_counter() - start)


def measure_startup_profile(
    *,
    prepare_factory: Callable[[], T],
    cold_start_probe: Callable[[], None] | None = None,
    warm_cache_probe: Callable[[], None] | None = None,
    cache_prefix: str = "mojogp_benchmark_startup_",
) -> tuple[T, dict[str, float]]:
    """Build an object and separately measure prepare/cold/warm startup costs."""

    obj, prepare_time_s = measure_prepare_time(prepare_factory)
    startup_profile: dict[str, float] = {
        "startup_prepare_time_s": prepare_time_s,
    }

    if cold_start_probe is None:
        return obj, startup_profile

    warm_probe = warm_cache_probe or cold_start_probe
    try:
        with temporary_jit_cache(cache_prefix):
            cold_start = time.perf_counter()
            cold_start_probe()
            startup_profile["startup_compile_time_s"] = float(
                time.perf_counter() - cold_start
            )

            warm_start = time.perf_counter()
            warm_probe()
            startup_profile["startup_warm_cache_hit_s"] = float(
                time.perf_counter() - warm_start
            )
    except OSError:
        # Low-disk environments may not be able to create a temporary JIT cache.
        # Keep the always-available wrapper-construction timing instead of failing
        # the benchmark route outright.
        pass

    return obj, startup_profile
