"""TOML config loading for JIT GPU optimization benchmarks."""

from __future__ import annotations
import tomllib
from pathlib import Path
from dataclasses import dataclass, field
from mojogp.codegen_engine.schedule import ScheduleConfig as Strategy


@dataclass
class KernelSpec:
    """Kernel specification from benchmark config."""

    type: str  # "rbf_iso", "matern52_ard", "rbf+matern52", etc.
    dim: int = 5
    n_values: list[int] = field(default_factory=lambda: [5000, 10000])


@dataclass
class ExperimentSpec:
    """Full experiment specification."""

    name: str
    strategies: list[str]
    kernels: list[KernelSpec]
    iters: int = 10
    seed: int = 42
    warmup_iters: int = 3
    cg_tol: float = 1e-2
    max_cg_iter: int = 100
    num_probes: int = 10
    precond_rank: int = 10
    lr: float = 0.1
    compare_against: list[str] = field(default_factory=list)
    output_dir: str = "results"


def load_strategy(path: Path) -> tuple[str, Strategy]:
    """Load strategy TOML -> (name, Strategy)."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    s = dict(data["strategy"])  # copy so we can pop
    name = s.pop("name")
    return name, Strategy.from_dict(s)


def load_experiment(
    path: Path, config_root: Path
) -> tuple[ExperimentSpec, dict[str, Strategy]]:
    """Load experiment TOML + resolve strategy references.

    Returns (experiment_spec, {strategy_name: Strategy}).
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    exp = data["experiment"]
    kernels = []
    for k in data.get("kernels", []):
        kernels.append(
            KernelSpec(
                type=k["type"],
                dim=k.get("dim", 5),
                n_values=k.get("n_values", [5000, 10000]),
            )
        )

    spec = ExperimentSpec(
        name=exp["name"],
        strategies=exp["strategies"],
        kernels=kernels,
        iters=exp.get("iters", 10),
        seed=exp.get("seed", 42),
        warmup_iters=exp.get("warmup_iters", 3),
        cg_tol=exp.get("cg_tol", 1e-2),
        max_cg_iter=exp.get("max_cg_iter", 100),
        num_probes=exp.get("num_probes", 10),
        precond_rank=exp.get("precond_rank", 10),
        lr=exp.get("lr", 0.1),
        compare_against=exp.get("compare_against", []),
        output_dir=exp.get("output_dir", "results"),
    )

    # Resolve strategy references
    strategies = {}
    for sname in spec.strategies:
        spath = config_root / "strategies" / f"{sname}.toml"
        if not spath.exists():
            raise FileNotFoundError(f"Strategy file not found: {spath}")
        _, strat = load_strategy(spath)
        strategies[sname] = strat

    return spec, strategies
