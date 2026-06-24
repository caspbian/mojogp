"""Compilation and caching for production JIT codegen.

Compiles generated .mojo source to .so using content-hash caching.
Cache location: ~/.cache/mojogp/jit_kernels/v3/<gpu-target>/

Usage:
    from mojogp.codegen_engine.compiler import compile_kernel
    so_path = compile_kernel(kernel_node, dim=5)
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from mojogp.compiler import _find_mojo_binary_and_env

from . import generate_module, generate_fn_ptr_module
from .schedule import ScheduleConfig


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "mojogp" / "jit_kernels" / "v3"


def _effective_gpu_target(gpu_target: Optional[str]) -> str:
    return gpu_target or os.environ.get("GPU_TARGET") or "auto"


def _target_segment(gpu_target: Optional[str]) -> str:
    return _effective_gpu_target(gpu_target).replace("/", "_").replace(" ", "_")


def _cache_dir_for_target(gpu_target: Optional[str]) -> Path:
    cache_dir = get_cache_dir() / _target_segment(gpu_target)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compile_kernel(
    kernel,
    dim: int,
    mode: str = "fn_ptr",
    schedule_overrides: Optional[ScheduleConfig] = None,
    ncols_hint: Optional[list] = None,
    module_suffix: Optional[str] = None,
    force_recompile: bool = False,
    gpu_target: Optional[str] = None,
    timeout: int = 120,
    verbose: bool = False,
) -> str:
    """Compile a JIT kernel .so and return the path.

    Uses content-hash caching: hash(source_code) -> cached .so.
    Returns immediately if cached.

    Args:
        kernel: KernelNode from mojogp.kernel
        dim: Input dimension
        mode: "fn_ptr" (lightweight, ~5s) or "trait" (self-contained, ~60s)
        schedule_overrides: Optional manual schedule configuration
        ncols_hint: Optional list of NCOLS values to specialize for
        force_recompile: Skip cache and recompile
        gpu_target: GPU target (e.g., "sm_89"), None for auto-detect
        timeout: Compilation timeout in seconds
        verbose: Print compilation info

    Returns:
        Path to compiled .so file
    """
    timeout = int(os.environ.get("MOJOGP_JIT_COMPILE_TIMEOUT", str(timeout)))

    # Generate module name
    module_name = make_module_name(kernel, dim, mode, module_suffix=module_suffix)

    # Generate source
    if mode == "fn_ptr":
        source = generate_fn_ptr_module(
            kernel,
            dim,
            module_name=module_name,
            schedule_overrides=schedule_overrides,
            ncols_hint=ncols_hint,
        )
    else:
        source = generate_module(
            kernel,
            dim,
            module_name=module_name,
            schedule_overrides=schedule_overrides,
            ncols_hint=ncols_hint,
        )

    # Reuse the shared MAX/Mojo environment setup so subprocess JIT compiles
    # can locate stdlib and runtime libraries outside the active Python env.
    mojo_bin, mojo_env = _find_mojo_binary_and_env()

    # Content hash for caching. Include the compiler/runtime identity so shared
    # libraries built against a different MAX runtime are not reused.
    cache_key = "\n".join(
        [
            source,
            mojo_bin,
            mojo_env.get("MODULAR_MOJO_MAX_IMPORT_PATH", ""),
            mojo_env.get("MODULAR_MOJO_MAX_COMPILERRT_PATH", ""),
        ]
    )
    content_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    cache_dir = _cache_dir_for_target(gpu_target)
    cache_path = cache_dir / f"{content_hash}.so"

    if cache_path.exists() and not force_recompile:
        if verbose:
            print(f"Cache hit: {cache_path}")
        return str(cache_path)

    if verbose:
        print(f"Compiling {module_name} (mode={mode})...")

    # Write source to temp file and compile
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".mojo", delete=False
    ) as f:
        f.write(source)
        source_path = f.name

    try:
        cmd = [
            mojo_bin,
            "build",
            source_path,
            "--emit",
            "shared-lib",
            "-o",
            str(cache_path),
        ]

        if gpu_target:
            cmd.extend(["--target-accelerator", gpu_target])
        elif os.environ.get("GPU_TARGET"):
            cmd.extend(["--target-accelerator", os.environ["GPU_TARGET"]])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=mojo_env,
            )
        except subprocess.TimeoutExpired:
            cache_path.unlink(missing_ok=True)
            raise

        if result.returncode != 0:
            cache_path.unlink(missing_ok=True)
            # Save source for debugging
            debug_path = cache_dir / f"{content_hash}_FAILED.mojo"
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(source)
            raise RuntimeError(
                f"JIT kernel compilation failed.\n"
                f"Source saved to: {debug_path}\n"
                f"Errors:\n{result.stderr}"
            )

        if verbose:
            print(f"Compiled: {cache_path}")
        return str(cache_path)

    finally:
        os.unlink(source_path)


def compile_kernel_from_source(
    source: str,
    module_name: str = "jit_kernel",
    force_recompile: bool = False,
    gpu_target: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """Compile raw .mojo source to .so with caching.

    Lower-level API for when you have pre-generated source code.
    """
    timeout = int(os.environ.get("MOJOGP_JIT_COMPILE_TIMEOUT", str(timeout)))
    mojo_bin, mojo_env = _find_mojo_binary_and_env()
    cache_key = "\n".join(
        [
            source,
            mojo_bin,
            mojo_env.get("MODULAR_MOJO_MAX_IMPORT_PATH", ""),
            mojo_env.get("MODULAR_MOJO_MAX_COMPILERRT_PATH", ""),
        ]
    )
    content_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    cache_dir = _cache_dir_for_target(gpu_target)
    cache_path = cache_dir / f"{content_hash}.so"

    if cache_path.exists() and not force_recompile:
        return str(cache_path)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".mojo", delete=False
    ) as f:
        f.write(source)
        source_path = f.name

    try:
        cmd = [
            mojo_bin,
            "build",
            source_path,
            "--emit",
            "shared-lib",
            "-o",
            str(cache_path),
        ]
        if gpu_target:
            cmd.extend(["--target-accelerator", gpu_target])
        elif os.environ.get("GPU_TARGET"):
            cmd.extend(["--target-accelerator", os.environ["GPU_TARGET"]])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=mojo_env,
            )
        except subprocess.TimeoutExpired:
            cache_path.unlink(missing_ok=True)
            raise

        if result.returncode != 0:
            cache_path.unlink(missing_ok=True)
            debug_path = cache_dir / f"{content_hash}_FAILED.mojo"
            with open(debug_path, "w", encoding="utf-8") as df:
                df.write(source)
            raise RuntimeError(
                f"JIT kernel compilation failed.\n"
                f"Source saved to: {debug_path}\n"
                f"Errors:\n{result.stderr}"
            )

        return str(cache_path)
    finally:
        os.unlink(source_path)


def get_cache_dir(gpu_target: Optional[str] = None) -> Path:
    """Return the JIT kernel cache directory.

    With no target, returns the cache root that contains per-target subdirs.
    With a target, returns the concrete cache dir for that target.
    """
    cache_root = Path(os.environ.get("MOJOGP_JIT_CACHE_DIR", str(DEFAULT_CACHE_ROOT)))
    cache_root.mkdir(parents=True, exist_ok=True)
    if gpu_target is None:
        return cache_root
    cache_dir = cache_root / _target_segment(gpu_target)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_cache():
    """Clear all cached JIT kernel .so files."""
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


def make_module_name(
    kernel,
    dim: int,
    mode: str,
    module_suffix: Optional[str] = None,
) -> str:
    """Generate a module name from kernel config."""
    # Use kernel's Mojo type string for unique naming
    try:
        ktype = (
            kernel.to_mojo_type()
            .replace("[", "_")
            .replace("]", "")
            .replace(",", "_")
            .replace(" ", "")
        )
        # Truncate if too long
        if len(ktype) > 40:
            ktype = ktype[:40]
        signature = f"{kernel.to_mojo_type()}|p{kernel.engine_num_params()}"
    except Exception:
        ktype = "kernel"
        signature = f"kernel|d{dim}|{mode}"
    sig_hash = hashlib.sha256(signature.encode()).hexdigest()[:10]
    suffix = ""
    if module_suffix not in (None, ""):
        safe_suffix = "".join(
            ch if ch.isalnum() or ch == "_" else "_" for ch in str(module_suffix)
        )
        suffix = f"_{safe_suffix}"
    return f"jit_{ktype}_{sig_hash}_d{dim}_{mode}{suffix}"
