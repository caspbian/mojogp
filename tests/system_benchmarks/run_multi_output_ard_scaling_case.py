"""Run one multi-output ARD scaling benchmark case in isolation."""

from __future__ import annotations

from tests.shared.subprocess_harness import IsolatedGPUTestSession, run_child_main

from .run_multi_output_scaling_case import _run_case


def _handle_child(payload: dict[str, object], session: IsolatedGPUTestSession):
    result_path = _run_case(
        framework=str(payload["framework"]),
        prediction_mode=str(payload["prediction_mode"]),
        method=str(payload["method"]),
        n_train=int(payload["n_train"]),
        d=int(payload["d"]),
        num_tasks=int(payload["num_tasks"]),
        tier=str(payload["tier"]),
        results_dir=str(payload["results_dir"]),
        session=session,
        dataset_path=(
            None
            if payload.get("dataset_path") is None
            else str(payload.get("dataset_path"))
        ),
        specialization=(
            None
            if payload.get("specialization") is None
            else dict(payload.get("specialization", {}))
        ),
        ard=True,
        relevant_dims=int(payload["relevant_dims"]),
    )
    return {"result_path": result_path}


def main() -> int:
    return run_child_main(_handle_child)


if __name__ == "__main__":
    raise SystemExit(main())
