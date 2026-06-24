"""Run one single-output ARD scaling benchmark case in isolation."""

from __future__ import annotations

import os
from pathlib import Path

from tests.shared.subprocess_harness import run_child_main
from tests.shared.benchmarking.report import save_result_artifact

from .test_scaling_certification_harness import _run_scaling_case


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    os.environ["MOJOGP_SINGLE_OUTPUT_CHILD"] = "1"

    benchmark_name = str(payload.get("benchmark_name", "single_output_ard_scaling"))
    results_dir = Path(str(payload["results_dir"]))
    result = _run_scaling_case(
        str(payload["method"]),
        int(payload["n_train"]),
        int(payload["d"]),
        framework=str(payload["framework"]),
        prediction_mode=str(payload["prediction_mode"]),
        tier=str(payload["tier"]),
        benchmark_variety=str(payload.get("benchmark_variety", "minimal")),
        benchmark_track=str(payload.get("benchmark_track", "scaling")),
        n_selection_policy=(
            None
            if payload.get("n_selection_policy") is None
            else str(payload["n_selection_policy"])
        ),
        size_role=None if payload.get("size_role") is None else str(payload["size_role"]),
        max_iterations=int(payload.get("max_iterations", 100)),
        enable_early_stopping=bool(payload.get("enable_early_stopping", False)),
        benchmark_name=benchmark_name,
        mojogp_preset=(
            None
            if payload.get("mojogp_preset") is None
            else str(payload["mojogp_preset"])
        ),
        data_options=dict(payload.get("data_options", {})),
        specialization=(
            None
            if payload.get("specialization") is None
            else dict(payload.get("specialization", {}))
        ),
        dataset_path=(
            None
            if payload.get("dataset_path") is None
            else str(payload.get("dataset_path"))
        ),
        results_dir=results_dir,
        ard=True,
        relevant_dims=int(payload["relevant_dims"]),
    )

    result_path = save_result_artifact(result, results_dir, benchmark_name)
    return {"result_path": result_path}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
