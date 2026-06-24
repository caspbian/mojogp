"""Mojo Compiler for MojoGP.

This module handles compilation of generated Mojo code to shared libraries,
with hash-based caching for fast reuse.

Shared utilities (_find_mojo_binary_and_env, _build_mojo_cmd, get_cache_dir,
clear_jit_cache) are used by codegen_engine.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _find_modular_package() -> Optional[Path]:
    """Find the modular package directory containing the mojo binary and stdlib.

    Searches site-packages directories for the 'modular' package.

    Returns:
        Path to the modular package directory, or None if not found.
    """
    import glob
    import site
    import sys

    search_dirs = list(sys.path)
    try:
        search_dirs.extend(site.getsitepackages())
    except AttributeError:
        pass
    if hasattr(site, "getusersitepackages"):
        search_dirs.append(site.getusersitepackages())

    # Also check common venv locations
    mojogp_venv = Path.home() / ".venv" / "mojogp"
    if mojogp_venv.exists():
        for site_pkg in glob.glob(str(mojogp_venv / "lib/python*/site-packages")):
            search_dirs.insert(0, site_pkg)

    home_venvs = Path.home() / ".venv"
    if home_venvs.exists():
        for site_pkg in glob.glob(str(home_venvs / "*/lib/python*/site-packages")):
            search_dirs.append(site_pkg)

    for site_dir in search_dirs:
        modular_dir = Path(site_dir) / "modular"
        if (modular_dir / "bin" / "mojo").exists():
            return modular_dir

    return None


def _complete_mojo_env(binary: str, env: dict) -> dict:
    """Make the Mojo subprocess environment use one coherent SDK root."""

    modular_dir = Path(binary).resolve().parents[1]
    lib_dir = modular_dir / "lib"
    mojo_lib_dir = lib_dir / "mojo"

    completed = dict(env)
    completed["MODULAR_MOJO_MAX_IMPORT_PATH"] = str(mojo_lib_dir)
    completed["MODULAR_MOJO_MAX_DRIVER_PATH"] = str(Path(binary).resolve())
    completed["MODULAR_MOJO_MAX_PACKAGE_ROOT"] = str(modular_dir)
    completed["MODULAR_MAX_PACKAGE_ROOT"] = str(modular_dir)
    completed["MODULAR_MAX_PATH"] = str(modular_dir)
    completed["MODULAR_HOME"] = str(modular_dir)

    compilerrt = lib_dir / "libKGENCompilerRTShared.so"
    if compilerrt.exists():
        completed["MODULAR_MOJO_MAX_COMPILERRT_PATH"] = str(compilerrt)

    lld = modular_dir / "bin" / "lld"
    if lld.exists():
        completed["MODULAR_MOJO_MAX_LLD_PATH"] = str(lld)

    shared_libs = []
    for lib_name in [
        "libAsyncRTMojoBindings.so",
        "libAsyncRTRuntimeGlobals.so",
        "libMSupportGlobals.so",
    ]:
        lib_path = lib_dir / lib_name
        if lib_path.exists():
            shared_libs.append(str(lib_path))
    if shared_libs:
        shared_libs.extend(["-Xlinker", "-rpath", "-Xlinker", str(lib_dir)])
        completed["MODULAR_MOJO_MAX_SHARED_LIBS"] = ",".join(shared_libs)

    completed["MODULAR_MOJO_MAX_SYSTEM_LIBS"] = "-lrt,-ldl,-lpthread,-lm"
    existing_ld = completed.get("LD_LIBRARY_PATH")
    completed["LD_LIBRARY_PATH"] = (
        str(lib_dir) if not existing_ld else str(lib_dir) + os.pathsep + existing_ld
    )
    return completed


def _find_mojo_binary_and_env() -> tuple[str, dict]:
    """Find the mojo binary and compute the environment needed to run it.

    The venv wrapper script at .venv/bin/mojo is a Python shim that calls
    exec_mojo() to set up environment variables (stdlib path, etc.) before
    invoking the real binary. This shim fails when called via subprocess.run()
    from a different venv because the subprocess Python can't import the
    'mojo' package.

    Instead, we find the real binary at modular/bin/mojo and compute the
    required environment variables ourselves, matching what the Python shim
    would do. This works reliably regardless of which venv is active.

    Returns:
        Tuple of (mojo_binary_path, env_dict) where env_dict contains
        the environment variables needed for the mojo binary to find stdlib.
    """
    import sys

    # Allow explicit override
    env_mojo = os.environ.get("MOJO_BINARY")
    if env_mojo and Path(env_mojo).exists():
        return env_mojo, _complete_mojo_env(env_mojo, dict(os.environ))

    package_root = Path(__file__).resolve().parents[1]
    max_sdk = Path(os.environ.get("MOJOGP_MAX_SDK", package_root / "max_sdk_bundle" / "max"))
    bundled_mojo = max_sdk / "bin" / "mojo"
    if bundled_mojo.exists():
        return str(bundled_mojo), _complete_mojo_env(str(bundled_mojo), dict(os.environ))

    # Try to import mojo._entrypoints to get the canonical environment
    # This works when running in the same venv where mojo is installed
    try:
        from mojo._entrypoints import _mojo_env

        env = _mojo_env()
        binary = env.get("MODULAR_MOJO_MAX_DRIVER_PATH", "")
        if binary and Path(binary).exists():
            return binary, _complete_mojo_env(binary, env)
    except (ImportError, Exception):
        pass

    # Fallback: find the modular package and compute env ourselves
    modular_dir = _find_modular_package()
    if modular_dir is None:
        raise RuntimeError(
            "Could not find the Mojo binary required for MojoGP runtime JIT "
            "compilation. Reinstall mojogp with its runtime dependencies, ensure "
            "mojo is on your PATH, or set MOJO_BINARY."
        )

    binary = str(modular_dir / "bin" / "mojo")
    return binary, _complete_mojo_env(binary, dict(os.environ))


def _build_mojo_cmd(
    mojo_bin: str, source_path: Path, output_path: Path, mojogp_dir: Path
) -> list:
    """Build the mojo compile command with optional GPU target.

    Reads GPU_TARGET env var to add --target-accelerator flag.
    """
    cmd = [
        mojo_bin,
        "build",
        str(source_path),
        "--emit",
        "shared-lib",
        "-I",
        str(mojogp_dir),
        "-o",
        str(output_path),
    ]
    gpu_target = os.environ.get("GPU_TARGET")
    if gpu_target:
        cmd.extend(["--target-accelerator", gpu_target])
    return cmd


# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mojogp" / "kernels" / "v1"


def get_cache_dir() -> Path:
    """Get the cache directory, creating it if necessary."""
    cache_dir = Path(os.environ.get("MOJOGP_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def clear_jit_cache() -> int:
    """Clear all cached JIT-compiled kernel .so files.

    Returns:
        Number of .so files removed.
    """
    cache_dir = get_cache_dir()
    count = 0
    for so_file in cache_dir.glob("*.so"):
        so_file.unlink()
        count += 1
    return count
