import sys
import types
import tomllib
from pathlib import Path

import pytest

from mojogp import loader


def test_accelerator_loader_supported_targets_match_package_extras():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    accelerator_extras = {name for name in extras if name.startswith("sm")}
    canonical_extras = {name for name in accelerator_extras if "_" not in name}
    underscored_aliases = {name for name in accelerator_extras if "_" in name}

    expected = set(loader._SUPPORTED_ACCELERATOR_TARGETS)
    assert canonical_extras == expected
    assert underscored_aliases == {f"sm_{target[2:]}" for target in expected}


def test_accelerator_loader_reports_missing_package(monkeypatch):
    monkeypatch.setenv("GPU_TARGET", "sm_90")
    monkeypatch.delitem(sys.modules, "mojogp_cuda_sm90", raising=False)

    with pytest.raises(RuntimeError, match=r'pip install --upgrade "mojogp\[sm90\]"'):
        loader._load_accelerator_engine_path()


def test_accelerator_loader_rejects_version_mismatch(monkeypatch, tmp_path):
    module = types.ModuleType("mojogp_cuda_sm90")
    module.__version__ = "0.26.4.0"
    module.engine_path = lambda: str(tmp_path / "mojogp_jit_engine.so")
    monkeypatch.setitem(sys.modules, "mojogp_cuda_sm90", module)
    monkeypatch.setenv("GPU_TARGET", "sm_90")

    with pytest.raises(RuntimeError) as exc_info:
        loader._load_accelerator_engine_path()

    message = str(exc_info.value)
    assert "Installed MojoGP version is 0.26.6.0" in message
    assert "mojogp-cuda-sm90 is 0.26.4.0" in message
    assert 'pip install --upgrade "mojogp[sm90]"' in message


def test_accelerator_loader_returns_packaged_engine(monkeypatch, tmp_path):
    engine_path = tmp_path / "mojogp_jit_engine.so"
    engine_path.write_bytes(b"placeholder")
    module = types.ModuleType("mojogp_cuda_sm89")
    module.__version__ = "0.26.6.0"
    module.engine_path = lambda: str(engine_path)
    monkeypatch.setitem(sys.modules, "mojogp_cuda_sm89", module)
    monkeypatch.setenv("GPU_TARGET", "sm_89")

    assert Path(loader._load_accelerator_engine_path()) == engine_path


def test_source_checkout_loader_returns_local_accelerator_engine(monkeypatch, tmp_path):
    engine_path = (
        tmp_path
        / "accelerator_packages"
        / "sm89"
        / "src"
        / "mojogp_cuda_sm89"
        / "mojogp_jit_engine.so"
    )
    engine_path.parent.mkdir(parents=True)
    engine_path.write_bytes(b"placeholder")
    monkeypatch.setenv("GPU_TARGET", "sm_89")

    assert Path(loader._source_checkout_engine_path(tmp_path)) == engine_path


def test_source_checkout_loader_reports_missing_local_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("GPU_TARGET", "sm_89")

    with pytest.raises(RuntimeError) as exc_info:
        loader._source_checkout_engine_path(tmp_path)

    message = str(exc_info.value)
    assert "MOJOGP_FROM_SOURCE=1 is set" in message
    assert "mojo build mojogp/kernels/jit/jit_engine_bindings.mojo" in message
    assert "--target-accelerator sm_89" in message
