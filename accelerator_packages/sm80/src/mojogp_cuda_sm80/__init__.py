"""MojoGP CUDA sm80 accelerator package."""

from pathlib import Path

__version__ = "0.26.6.0"
SM_TARGET = "sm80"


def engine_path() -> str:
    """Return the packaged JIT engine path."""
    return str(Path(__file__).with_name("mojogp_jit_engine.so"))
