"""Module Loader for MojoGP.

This module handles loading compiled Mojo shared libraries as Python modules.
"""

import importlib.util
import importlib
import importlib.resources
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from .kernel import KernelNode
from .specialization import SpecializationDecision, translate_compile_inputs

# =============================================================================
# Production JIT Engine Loader (codegen_engine fn-ptr path)
# =============================================================================

_engine_module = None  # Cached engine .so module
_SUPPORTED_ACCELERATOR_TARGETS = ("sm75", "sm80", "sm86", "sm89", "sm90", "sm100", "sm120")


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_gpu_target(target: str) -> str:
    normalized = target.strip().lower().replace("_", "")
    if normalized.startswith("cuda-"):
        normalized = normalized[len("cuda-") :]
    return normalized


def _detect_gpu_target() -> str:
    gpu_target = os.environ.get("GPU_TARGET")
    if gpu_target:
        return _normalize_gpu_target(gpu_target)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None

    if result is not None and result.returncode == 0:
        first_capability = result.stdout.strip().splitlines()[0].strip()
        digits = first_capability.replace(".", "")
        if digits.isdigit():
            return f"sm{digits}"

    supported = ", ".join(f"sm_{target[2:]}" for target in _SUPPORTED_ACCELERATOR_TARGETS)
    raise RuntimeError(
        "Could not detect a CUDA compute capability for MojoGP. Set GPU_TARGET "
        f"to one of: {supported}. Example: GPU_TARGET=sm_89 python train.py"
    )


def _accelerator_install_hint(target: str) -> str:
    return f'pip install --upgrade "mojogp[{target}]"'


def _source_build_hint(target: str) -> str:
    mojo_target = f"sm_{target[2:]}"
    return (
        "mojo build mojogp/kernels/jit/jit_engine_bindings.mojo "
        "--emit shared-lib -I mojogp/ -o mojogp_jit_engine.so "
        f"--target-accelerator {mojo_target}"
    )


def _load_accelerator_engine_path() -> str:
    target = _detect_gpu_target()
    if target not in _SUPPORTED_ACCELERATOR_TARGETS:
        supported = ", ".join(f"sm_{item[2:]}" for item in _SUPPORTED_ACCELERATOR_TARGETS)
        raise RuntimeError(
            "MojoGP does not have a prebuilt accelerator package for "
            f"GPU_TARGET={target}. Available prebuilt targets are: {supported}. "
            "Build from source with `mojo build ... --target-accelerator <sm_xx>` or install a "
            "matching supported accelerator extra."
        )

    package_name = f"mojogp_cuda_{target}"
    distribution_name = f"mojogp-cuda-{target}"
    try:
        accelerator_module = importlib.import_module(package_name)
    except ImportError as exc:
        raise RuntimeError(
            f"MojoGP needs the {distribution_name} accelerator package for "
            f"GPU_TARGET=sm_{target[2:]}. Install it with:\n"
            f"  {_accelerator_install_hint(target)}"
        ) from exc

    from ._version import __version__ as mojogp_version

    accelerator_version = getattr(accelerator_module, "__version__", None)
    if accelerator_version != mojogp_version:
        raise RuntimeError(
            f"Installed MojoGP version is {mojogp_version}, but "
            f"{distribution_name} is {accelerator_version or 'unknown'}.\n"
            "Install matching packages with:\n"
            f"  {_accelerator_install_hint(target)}"
        )

    engine_path_func = getattr(accelerator_module, "engine_path", None)
    if engine_path_func is not None:
        candidate = Path(str(engine_path_func()))
    else:
        candidate = Path(
            str(importlib.resources.files(package_name).joinpath("mojogp_jit_engine.so"))
        )
    if not candidate.exists():
        raise RuntimeError(
            f"Installed {distribution_name}=={accelerator_version} does not contain "
            f"the expected mojogp_jit_engine.so for GPU_TARGET=sm_{target[2:]}. "
            "Reinstall the matching accelerator package with:\n"
            f"  {_accelerator_install_hint(target)}"
        )
    return str(candidate)


def _source_checkout_engine_path(project_root: Optional[Path] = None) -> str:
    """Return the source-checkout accelerator engine for development runs."""
    target = _detect_gpu_target()
    if target not in _SUPPORTED_ACCELERATOR_TARGETS:
        supported = ", ".join(f"sm_{item[2:]}" for item in _SUPPORTED_ACCELERATOR_TARGETS)
        raise RuntimeError(
            "MojoGP source-checkout mode does not have a supported accelerator target for "
            f"GPU_TARGET={target}. Available targets are: {supported}."
        )

    if project_root is None:
        project_root = Path(__file__).parent.parent
    candidate = (
        project_root
        / "accelerator_packages"
        / target
        / "src"
        / f"mojogp_cuda_{target}"
        / "mojogp_jit_engine.so"
    )
    if not candidate.exists():
        raise RuntimeError(
            "MOJOGP_FROM_SOURCE=1 is set, but MojoGP could not find the local "
            f"accelerator engine for GPU_TARGET=sm_{target[2:]} at:\n"
            f"  {candidate}\n"
            "Build it with:\n"
            f"  {_source_build_hint(target)}"
        )
    return str(candidate)


def _engine_cache_path() -> Path:
    """Return the cache path for the compiled JIT engine."""
    cache_root = Path(
        os.environ.get(
            "MOJOGP_ENGINE_CACHE_DIR",
            str(Path.home() / ".cache" / "mojogp" / "jit_engine"),
        )
    )
    target = os.environ.get("GPU_TARGET", "auto")
    target_dir = target.replace("/", "_").replace(" ", "_")
    return cache_root / target_dir / "mojogp_jit_engine.so"


def _build_engine_from_package_source(verbose: bool = False) -> str:
    """Build the JIT engine into the user cache from packaged Mojo sources."""
    package_dir = Path(__file__).parent
    source_path = package_dir / "kernels" / "jit" / "jit_engine_bindings.mojo"
    if not source_path.exists():
        raise FileNotFoundError(
            "Could not find MojoGP JIT engine source in the installed package: "
            f"{source_path}"
        )

    cache_path = _engine_cache_path()
    if cache_path.exists():
        return str(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    from .compiler import _find_mojo_binary_and_env

    mojo_bin, mojo_env = _find_mojo_binary_and_env()
    cmd = [
        mojo_bin,
        "build",
        str(source_path),
        "--emit",
        "shared-lib",
        "-I",
        str(package_dir),
        "-o",
        str(cache_path),
    ]
    gpu_target = os.environ.get("GPU_TARGET")
    if gpu_target:
        cmd.extend(["--target-accelerator", gpu_target])

    if verbose:
        print(f"Building MojoGP JIT engine: {' '.join(cmd)}")

    timeout = int(os.environ.get("MOJOGP_ENGINE_BUILD_TIMEOUT", "900"))
    result = subprocess.run(
        cmd,
        env=mojo_env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build MojoGP JIT engine from installed Mojo sources.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not cache_path.exists():
        raise RuntimeError(
            "MojoGP JIT engine build completed without producing expected file: "
            f"{cache_path}"
        )
    return str(cache_path)


def load_engine(
    verbose: bool = False,
    *,
    fresh_load: bool = False,
    isolated_load_id: Optional[str] = None,
) -> Any:
    """Load the pre-compiled JIT engine .so (mojogp_jit_engine).

    The engine provides train() and predict() functions that work with
    fn-ptr kernel modules. It is compiled once and cached.

    Returns:
        The loaded mojogp_jit_engine Python module
    """
    module_name = "mojogp_jit_engine"

    global _engine_module
    if not fresh_load and _engine_module is not None:
        return _engine_module

    # Check if already in sys.modules
    if not fresh_load and module_name in sys.modules:
        _engine_module = sys.modules[module_name]
        return _engine_module

    # Find the engine .so. Editable checkouts can use a local build artifact,
    # while installed packages load a version-matched accelerator wheel.
    override_path = os.environ.get("MOJOGP_JIT_ENGINE_PATH")
    package_dir = Path(__file__).parent
    search_paths = [
        package_dir.parent,  # project root in editable checkouts
        package_dir,  # mojogp/
        Path.cwd(),  # current directory
    ]
    if override_path:
        search_paths.insert(0, Path(override_path).expanduser().resolve().parent)

    so_path = None
    if override_path:
        override_candidate = Path(override_path).expanduser().resolve()
        if override_candidate.exists():
            so_path = str(override_candidate)
    for search_dir in search_paths:
        if so_path is not None:
            break
        candidate = search_dir / f"{module_name}.so"
        if candidate.exists():
            so_path = str(candidate)
            break

    if so_path is None and _env_flag_enabled("MOJOGP_FROM_SOURCE"):
        so_path = _source_checkout_engine_path()

    if so_path is None:
        cached_engine = _engine_cache_path()
        if cached_engine.exists():
            so_path = str(cached_engine)
        else:
            so_path = _load_accelerator_engine_path()

    load_so_path = so_path
    if fresh_load and isolated_load_id:
        source_path = Path(so_path)
        isolated_dir = source_path.parent / "isolated_engine_loads"
        isolated_dir.mkdir(parents=True, exist_ok=True)
        isolated_path = isolated_dir / f"{source_path.stem}__{isolated_load_id}.so"
        if not isolated_path.exists():
            shutil.copy2(source_path, isolated_path)
        load_so_path = str(isolated_path)

    if fresh_load and module_name in sys.modules:
        del sys.modules[module_name]
    elif module_name in sys.modules:
        _engine_module = sys.modules[module_name]
        return _engine_module

    if verbose:
        print(f"Loading engine: {load_so_path}")

    spec = importlib.util.spec_from_file_location(module_name, load_so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {load_so_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not fresh_load:
        _engine_module = module
    return module


def load_kernel_module_engine(
    kernel: KernelNode,
    dim: int,
    force_recompile: bool = False,
    fresh_load: bool = False,
    isolated_load_id: Optional[str] = None,
    verbose: bool = False,
    gpu_target: Optional[str] = None,
    ncols_hint: Optional[list[int]] = None,
    specialization_decision: Optional[SpecializationDecision] = None,
) -> Any:
    """Load a fn-ptr kernel module compiled by the production codegen engine.

    Uses codegen_engine to generate and compile a lightweight kernel .so
    that exports function pointers for use with the JIT engine.

    Args:
        kernel: The kernel composition tree
        dim: Input dimension
        force_recompile: If True, recompile even if cached
        fresh_load: If True, reload the Python module from the cached .so even
            when it is already present in ``sys.modules``.
        isolated_load_id: Optional token used to load from a unique copied .so
            path while keeping the same embedded PyInit module name.
        verbose: Print progress
        gpu_target: GPU target (e.g. "sm_89"), None for auto
        ncols_hint: Optional list of NCOLS values to specialize for

    Returns:
        The loaded Python module with init_provider() and materialize()
    """
    from .codegen_engine.compiler import compile_kernel as engine_compile
    from .codegen_engine.compiler import make_module_name

    translation = translate_compile_inputs(specialization_decision)
    resolved_ncols_hint = translation.ncols_hint or ncols_hint
    module_suffix = translation.module_suffix
    if resolved_ncols_hint:
        ncols_suffix = "ncols_" + "_".join(str(int(n)) for n in resolved_ncols_hint)
        module_suffix = (
            ncols_suffix
            if module_suffix in (None, "")
            else f"{module_suffix}_{ncols_suffix}"
        )
    # Compile (or get from cache)
    so_path = engine_compile(
        kernel,
        dim,
        mode="fn_ptr",
        schedule_overrides=translation.schedule_overrides,
        ncols_hint=resolved_ncols_hint,
        module_suffix=module_suffix,
        force_recompile=force_recompile,
        gpu_target=gpu_target,
        verbose=verbose,
    )

    # Generate module name matching the PyInit_ function in the .so
    module_name = make_module_name(
        kernel,
        dim,
        "fn_ptr",
        module_suffix=module_suffix,
    )
    load_so_path = so_path

    if fresh_load and isolated_load_id:
        source_path = Path(so_path)
        isolated_dir = source_path.parent / "isolated_loads"
        isolated_dir.mkdir(parents=True, exist_ok=True)
        isolated_path = isolated_dir / f"{source_path.stem}__{isolated_load_id}.so"
        if not isolated_path.exists():
            shutil.copy2(source_path, isolated_path)
        load_so_path = str(isolated_path)

    if fresh_load and module_name in sys.modules:
        del sys.modules[module_name]

    # Check if already loaded
    if not force_recompile and module_name in sys.modules:
        if verbose:
            print(f"Engine kernel already loaded: {module_name}")
        return sys.modules[module_name]

    if force_recompile and module_name in sys.modules:
        del sys.modules[module_name]

    if verbose:
        print(f"Loading engine kernel: {load_so_path}")

    spec = importlib.util.spec_from_file_location(module_name, load_so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {load_so_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
