"""Run one MojoGP route-parity benchmark case in isolation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mojogp import SingleOutputGP, MultiOutputGP, RBF
from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.shared.subprocess_harness import run_child_main

from tests.shared.benchmarking.data_generators import generate_multi_output_data, generate_structured_function_data
from tests.shared.benchmarking.report import save_result_artifact
from tests.system_benchmarks.test_mojogp_route_parity_harness import (
    PARITY_SOLVER,
    _build_multi_output_result,
    _build_single_output_result,
    _finish_memory_stats,
    _memory_stats,
)


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    case = str(payload["case"])
    method = str(payload["method"])
    prediction_mode = str(payload["prediction_mode"])
    if case == "single_output":
        dataset = generate_structured_function_data(n_train=2000, n_test=BENCHMARK_PREDICTION_N_TEST, d=5, function_type="smooth", noise_level="medium", seed=4100)
        monitor, _ = _memory_stats()
        gp = SingleOutputGP(RBF())
        fit_start = time.perf_counter()
        train_result = gp.fit(dataset.X_train, dataset.y_train, method=method, max_iterations=8, learning_rate=0.03, lr_schedule="cosine", max_cg_iterations=PARITY_SOLVER["max_cg_iterations"], cg_tolerance=PARITY_SOLVER["cg_tolerance"], num_probes=PARITY_SOLVER["num_trace_samples"], max_tridiag_iterations=PARITY_SOLVER["max_lanczos_quadrature_iterations"], preconditioner_rank=PARITY_SOLVER["precond_rank"], preconditioner=PARITY_SOLVER["precond"], use_preconditioner=PARITY_SOLVER["use_preconditioner"], verbose=False)
        training_time_s = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        pred = gp.predict(dataset.X_test, variance_method=prediction_mode)
        prediction_time_s = time.perf_counter() - pred_start
        memory_stats = _finish_memory_stats(monitor)
        result = _build_single_output_result(dataset=dataset, method=method, prediction_mode=prediction_mode, training_time_s=training_time_s, prediction_time_s=prediction_time_s, iterations_run=int(train_result.iterations), memory_stats=memory_stats, mean=np.asarray(pred.mean, dtype=np.float32), variance=np.asarray(pred.variance, dtype=np.float32), params=gp.get_learned_params(), final_nll=float(train_result.nll))
        benchmark_name = "single_output_route_parity"
    elif case == "multi_output":
        dataset = generate_multi_output_data(n_train=2000, n_test=BENCHMARK_PREDICTION_N_TEST, d=5, num_tasks=3, kernel_type="rbf", task_correlation="medium", seed=5100)
        monitor, _ = _memory_stats()
        gp = MultiOutputGP(kernel="rbf", num_probes=PARITY_SOLVER["num_trace_samples"], max_cg_iterations=PARITY_SOLVER["max_cg_iterations"], cg_tolerance=PARITY_SOLVER["cg_tolerance"], max_tridiag_iterations=PARITY_SOLVER["max_lanczos_quadrature_iterations"], preconditioner_rank=PARITY_SOLVER["precond_rank"], preconditioner=PARITY_SOLVER["precond"], use_preconditioner=PARITY_SOLVER["use_preconditioner"])
        fit_start = time.perf_counter()
        train_result = gp.fit(dataset.X_train, dataset.Y_train, max_iterations=6, learning_rate=0.03, lr_schedule="cosine", verbose=False, method=method)
        training_time_s = time.perf_counter() - fit_start
        pred_start = time.perf_counter()
        mean, variance = gp.predict(dataset.X_test, return_var=True, variance_method=prediction_mode)
        prediction_time_s = time.perf_counter() - pred_start
        memory_stats = _finish_memory_stats(monitor)
        result = _build_multi_output_result(dataset=dataset, method=method, prediction_mode=prediction_mode, training_time_s=training_time_s, prediction_time_s=prediction_time_s, iterations_run=int(train_result.iterations), memory_stats=memory_stats, mean=np.asarray(mean, dtype=np.float32), variance=np.asarray(variance, dtype=np.float32), train_result=train_result)
        benchmark_name = "multi_output_route_parity"
    else:
        raise ValueError(f"Unknown case '{case}'")

    result_path = save_result_artifact(result, Path(str(payload["results_dir"])), benchmark_name)
    return {"result_path": str(result_path)}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
