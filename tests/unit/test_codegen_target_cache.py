"""Unit tests for target-aware JIT compiler cache paths."""

from __future__ import annotations

from pathlib import Path

from mojogp.codegen_engine import compiler


class _CompletedProcess:
    def __init__(self):
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""


def test_compile_kernel_from_source_uses_target_specific_cache_dirs(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MOJOGP_JIT_CACHE_DIR", str(tmp_path / "jit_cache"))
    monkeypatch.setattr(compiler, "_find_mojo_binary_and_env", lambda: ("mojo", {}))

    def _fake_run(cmd, capture_output, text, timeout, env=None):
        _ = env
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text("fake-so", encoding="utf-8")
        return _CompletedProcess()

    monkeypatch.setattr(compiler.subprocess, "run", _fake_run)

    source = "fn main():\n    pass\n"
    sm80_path = Path(
        compiler.compile_kernel_from_source(
            source, module_name="jit_kernel", gpu_target="sm_80"
        )
    )
    sm90_path = Path(
        compiler.compile_kernel_from_source(
            source, module_name="jit_kernel", gpu_target="sm_90"
        )
    )

    assert sm80_path != sm90_path
    assert sm80_path.parent.name == "sm_80"
    assert sm90_path.parent.name == "sm_90"
    assert sm80_path.exists()
    assert sm90_path.exists()


def test_compile_kernel_from_source_uses_env_target_in_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJOGP_JIT_CACHE_DIR", str(tmp_path / "jit_cache"))
    monkeypatch.setattr(compiler, "_find_mojo_binary_and_env", lambda: ("mojo", {}))

    def _fake_run(cmd, capture_output, text, timeout, env=None):
        _ = env
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_text("fake-so", encoding="utf-8")
        return _CompletedProcess()

    monkeypatch.setattr(compiler.subprocess, "run", _fake_run)
    monkeypatch.setenv("GPU_TARGET", "sm_80")

    cache_path = Path(
        compiler.compile_kernel_from_source(
            "fn main():\n    pass\n", module_name="jit_kernel"
        )
    )

    assert cache_path.parent.name == "sm_80"
    assert cache_path.exists()
