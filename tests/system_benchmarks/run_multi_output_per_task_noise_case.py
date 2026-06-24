"""Run one multi-output per-task-noise benchmark case in isolation."""

from __future__ import annotations

from pathlib import Path

from tests.shared.subprocess_harness import run_child_main

from tests.shared.benchmarking.report import save_result_artifact
from tests.system_benchmarks.test_multi_output_per_task_noise_harness import TestMultiOutputPerTaskNoise


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    tester = TestMultiOutputPerTaskNoise()
    result = tester._run_case(
        kernel=str(payload["kernel"]),
        n_train=int(payload["n_train"]),
        n_test=int(payload["n_test"]),
        d=int(payload["d"]),
        num_tasks=int(payload["num_tasks"]),
        task_correlation=str(payload["task_correlation"]),
        noise_profile=str(payload["noise_profile"]),
        mean_profile=str(payload["mean_profile"]),
        method=str(payload["method"]),
        dataset_family=str(payload.get("dataset_family", "gp_prior")),
        extra_config=dict(payload.get("extra_config", {})),
    )
    results_dir = Path(str(payload["results_dir"]))
    result_path = save_result_artifact(result, results_dir, "multi_output_per_task_noise")
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
