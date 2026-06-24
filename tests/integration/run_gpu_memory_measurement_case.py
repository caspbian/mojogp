"""Child entrypoints for isolated GPU-memory measurement tests."""

from __future__ import annotations

import gc

import numpy as np

from tests.shared.subprocess_harness import run_child_main
from tests.shared.benchmarking.gpu_memory import measure_gpu_phase


def _stabilize_torch_cuda_baseline(torch_module) -> None:
    """Warm the CUDA context and release cached allocator state before measuring."""

    if not torch_module.cuda.is_available():
        return

    warm = torch_module.empty(1, device="cuda")
    del warm
    gc.collect()
    torch_module.cuda.synchronize()
    torch_module.cuda.empty_cache()
    torch_module.cuda.ipc_collect()
    torch_module.cuda.synchronize()


def _run_mojogp_case(payload: dict[str, object]) -> dict[str, object]:
    from mojogp import SingleOutputGP
    from mojogp.kernel import RBF

    n = int(payload["n"])
    d = int(payload.get("d", 5))
    method = str(payload.get("method", "materialized"))
    prediction_mode = str(payload.get("prediction_mode", "exact"))
    n_test = int(payload.get("n_test", 16))
    max_iterations = int(payload.get("max_iterations", 1))

    rng = np.random.RandomState(42 + n + d)
    X = rng.randn(n, d).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.05 * rng.randn(n)).astype(np.float32)
    X_test = np.ascontiguousarray(X[:n_test], dtype=np.float32)

    gp = SingleOutputGP(RBF(), verbose=False)
    _, fit_stats = measure_gpu_phase(
        lambda: gp.fit(
            X,
            y,
            method=method,
            max_iterations=max_iterations,
            learning_rate=0.03,
            num_probes=2,
            max_cg_iterations=15,
            max_tridiag_iterations=8,
            preconditioner_rank=5,
            verbose=False,
        ),
        interval=0.02,
    )
    _, pred_stats = measure_gpu_phase(
        lambda: gp.predict(
            X_test,
                        variance_method=prediction_mode,
        ),
        interval=0.02,
    )

    peak_mb = max(
        float(fit_stats.get("phase_peak_gpu_mb", 0.0)),
        float(pred_stats.get("phase_peak_gpu_mb", 0.0)),
    )
    delta_mb = max(
        float(fit_stats.get("phase_delta_gpu_mb", 0.0)),
        float(pred_stats.get("phase_delta_gpu_mb", 0.0)),
    )
    result = {
        "framework": "mojogp",
        "peak_mb": peak_mb,
        "delta_mb": delta_mb,
        "baseline_mb": float(fit_stats.get("phase_baseline_gpu_mb", 0.0)),
        "torch_peak_mb": max(
            float(fit_stats.get("torch_peak_mb", 0.0)),
            float(pred_stats.get("torch_peak_mb", 0.0)),
        ),
        "training_peak_mb": float(fit_stats.get("phase_peak_gpu_mb", 0.0)),
        "training_delta_mb": float(fit_stats.get("phase_delta_gpu_mb", 0.0)),
        "prediction_peak_mb": float(pred_stats.get("phase_peak_gpu_mb", 0.0)),
        "prediction_delta_mb": float(pred_stats.get("phase_delta_gpu_mb", 0.0)),
        "backend_predict_info": dict(getattr(gp, "backend_predict_info", {}) or {}),
    }
    if prediction_mode == "love":
        result["love_prediction_peak_mb"] = result["prediction_peak_mb"]
        result["love_prediction_delta_mb"] = result["prediction_delta_mb"]
    else:
        result["exact_prediction_peak_mb"] = result["prediction_peak_mb"]
        result["exact_prediction_delta_mb"] = result["prediction_delta_mb"]
    return result


def _run_pytorch_case(payload: dict[str, object]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the PyTorch memory case")

    n = int(payload["n"])
    d = int(payload.get("d", 5))
    rng = np.random.RandomState(42 + n + d)
    X = rng.randn(n, d).astype(np.float32)

    _stabilize_torch_cuda_baseline(torch)

    def _phase() -> float:
        t_x = torch.tensor(X, dtype=torch.float32, device="cuda")
        K = torch.mm(t_x, t_x.T)
        torch.cuda.synchronize()
        return float(K[0, 0].item())

    _, stats = measure_gpu_phase(_phase, interval=0.02)
    return {
        "framework": "pytorch",
        "peak_mb": float(stats.get("phase_peak_gpu_mb", 0.0)),
        "delta_mb": float(stats.get("phase_delta_gpu_mb", 0.0)),
        "baseline_mb": float(stats.get("phase_baseline_gpu_mb", 0.0)),
        "torch_peak_mb": float(stats.get("torch_peak_mb", 0.0)),
    }


def _run_gpytorch_case(payload: dict[str, object], *, mode: str) -> dict[str, object]:
    import numpy as np

    from tests.shared.benchmarking.gpytorch_models import (
        merge_gpytorch_benchmark_memory,
        predict_gpytorch_single_output,
        train_gpytorch_single_output,
    )

    n = int(payload["n"])
    d = int(payload.get("d", 5))
    kernel = str(payload.get("kernel", "rbf"))
    prediction_mode = str(payload.get("prediction_mode", "exact"))
    n_test = int(payload.get("n_test", 16))
    max_iterations = int(payload.get("max_iterations", 2))

    rng = np.random.RandomState(42 + n + d)
    X = rng.randn(n, d).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.05 * rng.randn(n)).astype(np.float32)
    X_test = np.ascontiguousarray(X[:n_test], dtype=np.float32)

    train_result = train_gpytorch_single_output(
        X,
        y,
        kernel_type=kernel,
        mode=mode,
        n_iterations=max_iterations,
        lr=0.03,
        cg_tolerance=1e-2,
        max_cg_iterations=15,
        num_trace_samples=2,
        max_preconditioner_size=0 if mode == "keops" else 5,
        max_lanczos_quadrature_iterations=8,
        min_preconditioning_size=n + 1 if mode == "keops" else 0,
        memory_poll_interval=0.02,
        device="cuda",
    )
    pred_result = predict_gpytorch_single_output(
        train_result,
        X_test,
        mode=mode,
        cg_tolerance=1e-2,
        max_cg_iterations=15,
        max_preconditioner_size=0 if mode == "keops" else 5,
        max_lanczos_quadrature_iterations=8,
        min_preconditioning_size=n + 1 if mode == "keops" else 0,
        max_root_decomposition_size=8,
        use_love=(prediction_mode == "love"),
    )
    train_mem = dict(train_result.get("memory_stats", {}))
    pred_mem = dict(pred_result.get("memory_stats", {}))
    merged_mem = merge_gpytorch_benchmark_memory(train_mem, pred_mem)
    peak_mb = max(
        float(merged_mem.get("max_mb", 0.0)),
        float(merged_mem.get("torch_peak_mb", 0.0)),
    )
    baseline_mb = 0.0
    delta_mb = peak_mb
    return {
        "framework": "gpytorch_keops" if mode == "keops" else "gpytorch",
        "peak_mb": peak_mb,
        "delta_mb": delta_mb,
        "baseline_mb": baseline_mb,
        "torch_peak_mb": float(merged_mem.get("torch_peak_mb", 0.0)),
        "training_peak_mb": float(merged_mem.get("training_peak_gpu_mb", 0.0)),
        "prediction_peak_mb": float(merged_mem.get("prediction_peak_gpu_mb", 0.0)),
        "prediction_delta_mb": float(merged_mem.get("prediction_delta_gpu_mb", 0.0)),
        "exact_prediction_peak_mb": float(
            merged_mem.get("exact_prediction_peak_gpu_mb", 0.0)
        ),
        "exact_prediction_delta_mb": float(
            merged_mem.get("exact_prediction_delta_gpu_mb", 0.0)
        ),
        "love_prediction_peak_mb": float(
            merged_mem.get("love_prediction_peak_gpu_mb", 0.0)
        ),
        "love_prediction_delta_mb": float(
            merged_mem.get("love_prediction_delta_gpu_mb", 0.0)
        ),
        "prediction_mode": prediction_mode,
        "mean_time_s": float(pred_result.get("mean_time_s", 0.0)),
        "variance_time_s": float(pred_result.get("variance_time_s", 0.0)),
    }


def _handle(payload: dict[str, object], _session) -> dict[str, object]:
    framework = str(payload["framework"])
    if framework == "mojogp":
        result = _run_mojogp_case(payload)
    elif framework == "pytorch":
        result = _run_pytorch_case(payload)
    elif framework == "gpytorch":
        result = _run_gpytorch_case(payload, mode="cg")
    elif framework == "gpytorch_keops":
        result = _run_gpytorch_case(payload, mode="keops")
    else:
        raise ValueError(f"Unknown framework '{framework}'")
    return {"payload": result}


def main() -> int:
    return run_child_main(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
