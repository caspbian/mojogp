"""Run one LMC accuracy benchmark case in isolation."""

from __future__ import annotations

from pathlib import Path

from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.shared.subprocess_harness import run_child_main

from tests.shared.benchmarking.report import save_result_artifact
from tests.system_benchmarks.test_lmc_accuracy_harness import _run_lmc_accuracy_case


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    benchmark = _run_lmc_accuracy_case(
        n_train=int(payload.get("n_train", 2000)),
        n_test=int(payload.get("n_test", BENCHMARK_PREDICTION_N_TEST)),
        method=str(payload.get("method", "materialized")),
        max_iterations=int(payload.get("max_iterations", 40)),
        learning_rate=float(payload.get("learning_rate", 0.03)),
        seed=int(payload.get("seed", 42)),
        extra_config=dict(payload.get("extra_config", {})),
    )
    result_path = save_result_artifact(benchmark, Path(str(payload["results_dir"])), "lmc_ablation_accuracy")
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
