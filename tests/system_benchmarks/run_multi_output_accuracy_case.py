"""Run one multi-output accuracy benchmark case in isolation."""

from __future__ import annotations

from pathlib import Path

from tests.shared.subprocess_harness import run_child_main

from tests.shared.benchmarking.report import save_result_artifact
from tests.system_benchmarks.test_multi_output_accuracy_harness import TestMultiOutputAccuracy


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    tester = TestMultiOutputAccuracy()
    result = tester._run_multi_output_test(
        kernel=str(payload["kernel"]),
        n_train=int(payload["n_train"]),
        n_test=int(payload["n_test"]),
        d=int(payload["d"]),
        num_tasks=int(payload["num_tasks"]),
        correlation=str(payload["correlation"]),
        method=str(payload["method"]),
        seed=int(payload.get("seed", 42)),
    )
    results_dir = Path(str(payload["results_dir"]))
    result_path = save_result_artifact(result, results_dir, "multi_output_accuracy")
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
