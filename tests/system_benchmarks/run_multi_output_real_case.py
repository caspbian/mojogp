"""Run a single multi-output real-data benchmark case in isolation."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from mojogp import SingleOutputGP, Kernel, MultiOutputGP, MultiOutputLMCGP, RBF
from tests.shared.subprocess_harness import (
    IsolatedGPUTestSession,
    run_child_main,
    run_isolated_case,
)

from tests.shared.benchmarking.metrics import rmse, r_squared
from tests.shared.benchmarking.real_datasets import (
    load_energy_efficiency_multi_output,
    load_linnerud_multi_output,
)
from tests.shared.benchmarking.report import save_result_artifact
from tests.shared.benchmarking.result_types import (
    AccuracyResult,
    BenchmarkResult,
    HyperparameterResult,
    MemoryResult,
    SpeedResult,
)


def _fairness_axis(status: str, note: str) -> dict[str, str]:
    return {"status": status, "note": note}


def _real_data_metadata(
    *,
    data,
    gp,
    kernel_name: str,
    baseline_type: str,
) -> dict[str, object]:
    return {
        "benchmark": "multi_output_real_data",
        "route_group": "multi_output",
        "framework": "mojogp",
        "model_type": type(gp).__name__,
        "kernel": kernel_name,
        "training_method": getattr(gp, "method", "materialized"),
        "method": getattr(gp, "method", "materialized"),
        "prediction_mode": "love",
        "comparison_class": "mojogp_only",
        "baseline_backend": "none",
        "keops_supported": False,
        "keops_used": False,
        "fairness_note": (
            "N.B. MojoGP-only real-data row: this benchmark compares the active MojoGP multi-output "
            "model against in-repo ridge and independent ExactGP baselines on the same dataset."
        ),
        "fairness_axes": {
            "comparator_scope": _fairness_axis(
                "mojogp_only",
                "The published baselines are in-repo ridge and independent ExactGP models, not a cross-framework comparator.",
            ),
            "sample_count_n": _fairness_axis(
                "aligned",
                "All baselines run on the same real-data split.",
            ),
            "optimizer": _fairness_axis(
                "aligned",
                "The MojoGP row uses a fixed optimizer family and training budget for the reported model.",
            ),
            "solver_budget": _fairness_axis(
                "aligned",
                "The reported MojoGP row records a single route with fixed CG/Lanczos settings.",
            ),
            "preconditioner": _fairness_axis(
                "aligned",
                "The reported MojoGP row uses its configured route-level preconditioner budget.",
            ),
            "prediction_mode": _fairness_axis(
                "aligned",
                "The row reports the default wrapper predictive path for the saved model.",
            ),
            "telemetry": _fairness_axis(
                "observed",
                "MojoGP telemetry is observed for the active real-data route.",
            ),
        },
        "dataset": data.name,
        "baseline_type": baseline_type,
    }


def _serialize_benchmark(benchmark: BenchmarkResult) -> dict[str, object]:
    return {
        "config": benchmark.config,
        "accuracy": {
            "rmse": benchmark.accuracy.rmse,
            "r_squared": benchmark.accuracy.r_squared,
        },
        "speed": {
            "training_time_s": benchmark.speed.training_time_s,
            "end_to_end_time_s": benchmark.speed.end_to_end_time_s,
        },
        "memory": {
            "gpu_max_mb": benchmark.memory.gpu_max_mb,
            "torch_peak_mb": benchmark.memory.torch_peak_mb,
            "cpu_peak_mb": benchmark.memory.cpu_peak_mb,
        },
        "hyperparameters": {"final_nll": benchmark.hyperparameters.final_nll},
    }


def _run_child(case: str, **payload: object) -> dict[str, object]:
    return run_isolated_case(
        module="tests.system_benchmarks.run_multi_output_real_case",
        payload={"case": case, **payload},
        timeout=1200,
        description=f"Runs nested multi-output real-data case {case}",
    )


def _run_multi_output_real_benchmark(
    data,
    gp,
    *,
    kernel_name: str,
    max_iterations: int,
    learning_rate: float,
    results_dir: str,
    session: IsolatedGPUTestSession,
):
    results_dir = Path(results_dir)

    fit_start = time.perf_counter()
    result = gp.fit(
        data.X_train,
        data.Y_train,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        verbose=False,
    )
    training_time_s = time.perf_counter() - fit_start
    fit_snapshot = session.snapshot_gpu()

    pred_start = time.perf_counter()
    prediction = gp.predict(data.X_test)
    prediction_time_s = time.perf_counter() - pred_start
    pred_snapshot = session.snapshot_gpu()

    memory_stats = session.collect_memory_stats(
        snapshots=[fit_snapshot, pred_snapshot]
    )

    mean = np.asarray(prediction.mean, dtype=np.float32)
    variance = np.asarray(prediction.variance, dtype=np.float32)

    ridge = Ridge(alpha=1.0)
    ridge.fit(data.X_train, data.Y_train)
    ridge_pred = ridge.predict(data.X_test).astype(np.float32)

    independent_preds = []
    for task_idx in range(data.Y_train.shape[1]):
        single_gp = SingleOutputGP(RBF())
        single_gp.fit(
            data.X_train,
            data.Y_train[:, task_idx],
            method="materialized",
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            verbose=False,
        )
        single_mean, _ = single_gp.predict(data.X_test, return_std=True)
        independent_preds.append(np.asarray(single_mean, dtype=np.float32))
    independent_mean = np.stack(independent_preds, axis=1)

    learned_lengthscales = getattr(result, "lengthscales", None)
    if learned_lengthscales is None:
        learned_lengthscales = np.array([getattr(result, "lengthscale", 1.0)])
    learned_outputscale = getattr(result, "outputscale", None)
    if learned_outputscale is None:
        effective_scales = getattr(result, "effective_scales", None)
        if effective_scales is not None:
            effective_arr = np.asarray(effective_scales, dtype=np.float32)
            learned_outputscale = float(np.mean(effective_arr))
        else:
            learned_outputscale = 1.0

    per_task_rmse = [rmse(data.Y_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
    per_task_r2 = [
        r_squared(data.Y_test[:, t], mean[:, t]) for t in range(mean.shape[1])
    ]
    avg_rmse = float(np.mean(per_task_rmse))
    ridge_rmse = float(
        np.mean(
            [rmse(data.Y_test[:, t], ridge_pred[:, t]) for t in range(mean.shape[1])]
        )
    )
    independent_rmse = float(
        np.mean(
            [
                rmse(data.Y_test[:, t], independent_mean[:, t])
                for t in range(mean.shape[1])
            ]
        )
    )

    benchmark = BenchmarkResult(
        config={
            **_real_data_metadata(
                data=data,
                gp=gp,
                kernel_name=kernel_name,
                baseline_type="ridge_plus_independent_exactgp",
            ),
            "n": int(data.X_train.shape[0]),
            "d": int(data.X_train.shape[1]),
            "num_tasks": int(data.Y_train.shape[1]),
            "ridge_rmse": ridge_rmse,
            "independent_rmse": independent_rmse,
            "joint_gain_vs_ridge": ridge_rmse - avg_rmse,
            "joint_gain_vs_independent": independent_rmse - avg_rmse,
            "task_covariance_fro_error": float(
                np.linalg.norm(gp.task_covariance - np.cov(data.Y_train.T))
            ),
        },
        accuracy=AccuracyResult(
            rmse=avg_rmse,
            mae=float(np.mean(np.abs(data.Y_test - mean))),
            r_squared=float(np.mean(per_task_r2)),
            crps=0.0,
            msll=0.0,
            calibration_coverage={},
            calibration_error=0.0,
            sharpness=float(np.mean(np.sqrt(np.maximum(variance, 1e-10)))),
            interval_width_95=0.0,
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=prediction_time_s,
            prediction_variance_time_s=prediction_time_s,
            end_to_end_time_s=training_time_s + prediction_time_s,
            iterations_run=int(result.iterations),
            max_iterations=max_iterations,
            early_stopped=int(result.iterations) < max_iterations,
            ms_per_iteration=training_time_s / max(int(result.iterations), 1) * 1000.0,
        ),
        memory=MemoryResult(
            gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
            gpu_min_mb=memory_stats.get("min_mb", 0.0),
            gpu_max_mb=memory_stats.get("max_mb", 0.0),
            gpu_var_mb=memory_stats.get("var_mb", 0.0),
            torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
            torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
            cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
            measurement_method=memory_stats.get("method", "none"),
            num_samples=int(memory_stats.get("samples", 0)),
        ),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(learned_lengthscales))),
            learned_noise=float(np.mean(np.asarray(result.noise_per_task))),
            learned_outputscale=float(learned_outputscale),
            final_nll=float(result.final_nll),
        ),
    )
    save_result_artifact(benchmark, results_dir, "multi_output_real_data")
    return benchmark


def _evaluate_saved_multi_output_model(
    *,
    model_path: str,
    data,
    results_dir: str,
    kernel_name: str,
    training_time_s: float,
    session: IsolatedGPUTestSession,
    loader_cls=MultiOutputGP,
) -> BenchmarkResult:
    results_dir = Path(results_dir)

    gp = loader_cls.load(model_path)

    pred_start = time.perf_counter()
    prediction = gp.predict(data.X_test)
    prediction_time_s = time.perf_counter() - pred_start
    pred_snapshot = session.snapshot_gpu()

    memory_stats = session.collect_memory_stats(snapshots=[pred_snapshot])

    mean = np.asarray(prediction.mean, dtype=np.float32)
    variance = np.asarray(prediction.variance, dtype=np.float32)

    ridge = Ridge(alpha=1.0)
    ridge.fit(data.X_train, data.Y_train)
    ridge_pred = ridge.predict(data.X_test).astype(np.float32)

    independent_preds = []
    for task_idx in range(data.Y_train.shape[1]):
        single_gp = SingleOutputGP(RBF())
        single_gp.fit(
            data.X_train,
            data.Y_train[:, task_idx],
            method="materialized",
            max_iterations=int(gp._result.iterations),
            learning_rate=0.02,
            verbose=False,
        )
        single_mean, _ = single_gp.predict(data.X_test, return_std=True)
        independent_preds.append(np.asarray(single_mean, dtype=np.float32))
    independent_mean = np.stack(independent_preds, axis=1)

    result = gp._result
    learned_lengthscales = getattr(result, "lengthscales", None)
    if learned_lengthscales is None:
        learned_lengthscales = np.array([getattr(result, "lengthscale", 1.0)])
    learned_outputscale = getattr(result, "outputscale", None)
    if learned_outputscale is None:
        effective_scales = getattr(result, "effective_scales", None)
        if effective_scales is not None:
            effective_arr = np.asarray(effective_scales, dtype=np.float32)
            learned_outputscale = float(np.mean(effective_arr))
        else:
            learned_outputscale = 1.0

    per_task_rmse = [rmse(data.Y_test[:, t], mean[:, t]) for t in range(mean.shape[1])]
    per_task_r2 = [
        r_squared(data.Y_test[:, t], mean[:, t]) for t in range(mean.shape[1])
    ]
    avg_rmse = float(np.mean(per_task_rmse))
    ridge_rmse = float(
        np.mean(
            [rmse(data.Y_test[:, t], ridge_pred[:, t]) for t in range(mean.shape[1])]
        )
    )
    independent_rmse = float(
        np.mean(
            [
                rmse(data.Y_test[:, t], independent_mean[:, t])
                for t in range(mean.shape[1])
            ]
        )
    )

    benchmark = BenchmarkResult(
        config={
            **_real_data_metadata(
                data=data,
                gp=gp,
                kernel_name=kernel_name,
                baseline_type="ridge_plus_independent_exactgp",
            ),
            "n": int(data.X_train.shape[0]),
            "d": int(data.X_train.shape[1]),
            "num_tasks": int(data.Y_train.shape[1]),
            "ridge_rmse": ridge_rmse,
            "independent_rmse": independent_rmse,
            "joint_gain_vs_ridge": ridge_rmse - avg_rmse,
            "joint_gain_vs_independent": independent_rmse - avg_rmse,
            "task_covariance_fro_error": float(
                np.linalg.norm(gp.task_covariance - np.cov(data.Y_train.T))
            ),
        },
        accuracy=AccuracyResult(
            rmse=avg_rmse,
            mae=float(np.mean(np.abs(data.Y_test - mean))),
            r_squared=float(np.mean(per_task_r2)),
            crps=0.0,
            msll=0.0,
            calibration_coverage={},
            calibration_error=0.0,
            sharpness=float(np.mean(np.sqrt(np.maximum(variance, 1e-10)))),
            interval_width_95=0.0,
        ),
        speed=SpeedResult(
            training_time_s=training_time_s,
            prediction_mean_time_s=prediction_time_s,
            prediction_variance_time_s=prediction_time_s,
            end_to_end_time_s=training_time_s + prediction_time_s,
            iterations_run=int(result.iterations),
            max_iterations=int(result.iterations),
            early_stopped=False,
            ms_per_iteration=training_time_s / max(int(result.iterations), 1) * 1000.0,
        ),
        memory=MemoryResult(
            gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
            gpu_min_mb=memory_stats.get("min_mb", 0.0),
            gpu_max_mb=memory_stats.get("max_mb", 0.0),
            gpu_var_mb=memory_stats.get("var_mb", 0.0),
            torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
            torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
            cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
            measurement_method=memory_stats.get("method", "none"),
            num_samples=int(memory_stats.get("samples", 0)),
        ),
        hyperparameters=HyperparameterResult(
            learned_lengthscale=float(np.mean(np.asarray(learned_lengthscales))),
            learned_noise=float(np.mean(np.asarray(result.noise_per_task))),
            learned_outputscale=float(learned_outputscale),
            final_nll=float(result.final_nll),
        ),
    )
    save_result_artifact(benchmark, results_dir, "multi_output_real_data")
    return benchmark


def _handle_child(payload: dict[str, object], session: IsolatedGPUTestSession):
    case = str(payload["case"])
    results_dir = str(payload.get("results_dir", ""))
    if case == "linnerud_icm":
        data = load_linnerud_multi_output()
        gp = MultiOutputGP(
            kernel="rbf",
            task_rank=1,
            num_probes=5,
            max_cg_iterations=50,
            use_preconditioner=False,
        )
        fit_start = time.perf_counter()
        gp.fit(
            data.X_train,
            data.Y_train,
            method="materialized",
            max_iterations=40,
            learning_rate=0.03,
            verbose=False,
        )
        training_time_s = time.perf_counter() - fit_start
        model_dir = tempfile.mkdtemp(prefix="mojogp_linnerud_icm_")
        model_path = f"{model_dir}/model"
        gp.save(model_path)
        payload = _run_child(
            "linnerud_icm_eval",
            model_path=model_path,
            results_dir=results_dir,
            training_time_s=training_time_s,
        )
        return {"payload": payload}

    if case == "linnerud_icm_eval":
        model_path = str(payload["model_path"])
        results_dir = str(payload["results_dir"])
        training_time_s = float(payload["training_time_s"])
        data = load_linnerud_multi_output()
        benchmark = _evaluate_saved_multi_output_model(
            model_path=model_path,
            data=data,
            results_dir=results_dir,
            kernel_name="rbf",
            training_time_s=training_time_s,
            session=session,
            loader_cls=MultiOutputGP,
        )
        return {"payload": {"benchmark": _serialize_benchmark(benchmark)}}

    if case == "energy_lmc":
        data = load_energy_efficiency_multi_output()
        lmc = MultiOutputLMCGP(
            kernels=[Kernel.rbf(), Kernel.rbf()],
            num_probes=5,
            max_cg_iterations=50,
            use_preconditioner=False,
        )
        fit_start = time.perf_counter()
        lmc.fit(
            data.X_train,
            data.Y_train,
            method="materialized",
            max_iterations=60,
            learning_rate=0.02,
            verbose=False,
        )
        training_time_s = time.perf_counter() - fit_start
        model_dir = tempfile.mkdtemp(prefix="mojogp_energy_lmc_")
        model_path = f"{model_dir}/model"
        lmc.save(model_path)
        payload = _run_child(
            "energy_lmc_eval",
            model_path=model_path,
            results_dir=results_dir,
            training_time_s=training_time_s,
        )
        return {"payload": payload}

    if case == "energy_lmc_eval":
        model_path = str(payload["model_path"])
        results_dir = str(payload["results_dir"])
        training_time_s = float(payload["training_time_s"])
        data = load_energy_efficiency_multi_output()
        benchmark = _evaluate_saved_multi_output_model(
            model_path=model_path,
            data=data,
            results_dir=results_dir,
            kernel_name="lmc_rbf_r2",
            training_time_s=training_time_s,
            session=session,
            loader_cls=MultiOutputLMCGP,
        )
        icm = MultiOutputGP(
            kernel="rbf",
            task_rank=1,
            num_probes=5,
            max_cg_iterations=50,
            use_preconditioner=False,
        )
        icm.fit(
            data.X_train,
            data.Y_train,
            method="materialized",
            max_iterations=60,
            learning_rate=0.02,
            verbose=False,
        )
        model_dir = tempfile.mkdtemp(prefix="mojogp_energy_icm_")
        icm_path = f"{model_dir}/model"
        icm.save(icm_path)
        icm_payload = _run_child("energy_icm_eval", model_path=icm_path)
        return {
            "payload": {
                "benchmark": _serialize_benchmark(benchmark),
                "icm_rmse": icm_payload["icm_rmse"],
            }
        }

    if case == "energy_icm_eval":
        model_path = str(payload["model_path"])
        data = load_energy_efficiency_multi_output()
        icm = MultiOutputGP.load(model_path)
        icm_mean = np.asarray(icm.predict(data.X_test).mean, dtype=np.float32)
        icm_rmse = float(
            np.mean(
                [
                    rmse(data.Y_test[:, t], icm_mean[:, t])
                    for t in range(icm_mean.shape[1])
                ]
            )
        )
        return {"payload": {"icm_rmse": icm_rmse}}

    raise ValueError(f"Unknown case: {case}")


def main() -> int:
    return run_child_main(_handle_child)


if __name__ == "__main__":
    raise SystemExit(main())
