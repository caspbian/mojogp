"""MojoGP training and prediction wrappers for system benchmarks.

These helpers normalize the current public Python wrapper APIs (`ExactGP` and
`MultiOutputGP`) into the historical result dictionaries used by the older
benchmark modules.
"""

import time
import tracemalloc
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .gpu_memory import (
    GPUMemoryMonitor,
    get_torch_memory_stats,
    reset_torch_memory_stats,
)
from .result_types import (
    AccuracyResult,
    SpeedResult,
    MemoryResult,
    HyperparameterResult,
    BenchmarkResult,
)
from .metrics import compute_all_accuracy_metrics, param_relative_error


def _find_param_value(
    params: Dict[str, Any], suffixes: Tuple[str, ...], default: float
) -> float:
    for suffix in suffixes:
        for key, value in params.items():
            if key == suffix or key.endswith(suffix):
                return float(value)
    return float(default)


def normalize_single_output_benchmark_hparams(
    params: Dict[str, Any],
    *,
    default_lengthscale: float = 1.0,
    default_noise: float = 0.1,
    default_outputscale: float = 1.0,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "lengthscale": _find_param_value(
            params,
            ("_lengthscale", "_variance"),
            default_lengthscale,
        ),
        "noise": float(params.get("noise", default_noise)),
        "outputscale": _find_param_value(
            params,
            ("_outputscale",),
            default_outputscale,
        ),
    }
    ard_lengthscales = _extract_ard_lengthscales(params)
    if ard_lengthscales.size:
        normalized["lengthscales"] = ard_lengthscales.tolist()
        normalized["lengthscale"] = float(np.mean(ard_lengthscales))
    return normalized


def _extract_ard_lengthscales(params: Dict[str, Any]) -> np.ndarray:
    ard_items = []
    for key, value in params.items():
        if "_ls_" not in key:
            continue
        try:
            idx = int(key.rsplit("_", 1)[1])
        except ValueError:
            continue
        ard_items.append((idx, float(value)))
    ard_items.sort(key=lambda item: item[0])
    return np.asarray([value for _, value in ard_items], dtype=np.float32)


def _build_simple_kernel(
    kernel_type: str,
    *,
    init_ls: float,
    init_os: float,
    kernel_param1: Optional[float],
    kernel_param2: Optional[float],
    ard: bool = False,
):
    from mojogp import Kernel

    if kernel_type == "rbf":
        return Kernel.rbf(lengthscale=init_ls, outputscale=init_os, ard=ard)
    if kernel_type == "matern12":
        return Kernel.matern12(lengthscale=init_ls, outputscale=init_os, ard=ard)
    if kernel_type == "matern32":
        return Kernel.matern32(lengthscale=init_ls, outputscale=init_os, ard=ard)
    if kernel_type == "matern52":
        return Kernel.matern52(lengthscale=init_ls, outputscale=init_os, ard=ard)
    if kernel_type == "periodic":
        return Kernel.periodic(
            lengthscale=init_ls,
            period=1.0 if kernel_param1 is None else float(kernel_param1),
            outputscale=init_os,
            ard=ard,
        )
    if kernel_type == "rq":
        return Kernel.rq(
            lengthscale=init_ls,
            alpha=1.0 if kernel_param1 is None else float(kernel_param1),
            outputscale=init_os,
            ard=ard,
        )
    if kernel_type == "linear":
        return Kernel.linear(
            variance=init_ls,
            outputscale=init_os,
            ard=ard,
        )
    if kernel_type == "polynomial":
        return Kernel.polynomial(
            degree=3.0 if kernel_param1 is None else float(kernel_param1),
            offset=1.0 if kernel_param2 is None else float(kernel_param2),
            outputscale=init_os,
            ard=ard,
        )
    raise ValueError(f"Unknown kernel type: {kernel_type}")


def _normalize_simple_learned_params(
    gp,
    kernel_type: str,
    *,
    init_ls: float,
    init_noise: float,
    init_os: float,
    kernel_param1: Optional[float],
    kernel_param2: Optional[float],
) -> Dict[str, Any]:
    learned = gp.get_learned_params()
    normalized = {
        "lengthscale": _find_param_value(
            learned,
            ("_lengthscale", "_variance"),
            init_ls,
        ),
        "outputscale": _find_param_value(learned, ("_outputscale",), init_os),
        "noise": float(learned.get("noise", init_noise)),
        "_gp": gp,
    }

    if kernel_type == "periodic":
        normalized["period"] = _find_param_value(
            learned,
            ("_period",),
            1.0 if kernel_param1 is None else float(kernel_param1),
        )
    elif kernel_type == "rq":
        normalized["alpha"] = _find_param_value(
            learned,
            ("_alpha",),
            1.0 if kernel_param1 is None else float(kernel_param1),
        )
    elif kernel_type == "polynomial":
        normalized["degree"] = _find_param_value(
            learned,
            ("_degree",),
            3.0 if kernel_param1 is None else float(kernel_param1),
        )
        normalized["offset"] = _find_param_value(
            learned,
            ("_offset",),
            1.0 if kernel_param2 is None else float(kernel_param2),
        )

    return normalized


# =============================================================================
# Simple API (Pre-compiled kernels)
# =============================================================================


def train_mojogp_simple(
    X: np.ndarray,
    y: np.ndarray,
    kernel_type: str,
    method: str = "auto",
    n_iterations: int = 100,
    lr: float = 0.02,
    init_ls: float = 1.0,
    init_noise: float = 0.1,
    init_os: float = 1.0,
    kernel_param1: Optional[float] = None,
    kernel_param2: Optional[float] = None,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.1,
) -> Dict[str, Any]:
    """Train MojoGP using the current ExactGP wrapper.

    Args:
        X: Training data [n, d]
        y: Training targets [n]
        kernel_type: Kernel type string (e.g., 'rbf', 'matern52')
        method: Training method ('auto', 'materialized', 'matrix_free')
        n_iterations: Maximum training iterations
        lr: Learning rate
        init_ls: Initial lengthscale
        init_noise: Initial noise
        init_os: Initial outputscale
        kernel_param1: Kernel-specific param (auto-set based on kernel_type if None)
            - periodic: period length (default 1.0)
            - rq: alpha (default 1.0)
            - polynomial: degree (default 3.0)
        kernel_param2: Kernel-specific param (auto-set based on kernel_type if None)
            - polynomial: offset (default 1.0)
        monitor_memory: Whether to monitor GPU memory
        memory_poll_interval: Memory polling interval in seconds

    Returns:
        dict with keys:
            - result: Raw Mojo result dict
            - final_nll: Final NLL value
            - nll_history: List of NLL values (if available)
            - training_time_s: Training time in seconds
            - iterations_run: Actual iterations completed
            - max_iterations: Requested max iterations
            - early_stopped: Whether early stopping triggered
            - learned_params: {lengthscale, outputscale, noise}
            - peak_memory_mb: Peak GPU memory
            - memory_stats: Full memory statistics
            - lanczos_root: Lanczos root for variance prediction
            - lanczos_rank: Lanczos rank
    """
    from mojogp import SingleOutputGP

    # Ensure correct dtypes
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Set kernel-specific defaults if not provided
    if kernel_param1 is None:
        if kernel_type == "polynomial":
            kernel_param1 = 3.0  # degree
        elif kernel_type == "periodic":
            kernel_param1 = 1.0  # period
        elif kernel_type == "rq":
            kernel_param1 = 1.0  # alpha
        else:
            kernel_param1 = 1.0  # default

    if kernel_param2 is None:
        if kernel_type == "polynomial":
            kernel_param2 = 1.0  # offset
        else:
            kernel_param2 = 0.0  # default

    gp = SingleOutputGP(
        _build_simple_kernel(
            kernel_type,
            init_ls=init_ls,
            init_os=init_os,
            kernel_param1=kernel_param1,
            kernel_param2=kernel_param2,
        ),
        verbose=False,
    )

    # Setup memory monitoring
    memory_stats = {}
    if monitor_memory:
        reset_torch_memory_stats()
        monitor = GPUMemoryMonitor(interval=memory_poll_interval)
        monitor.start()

    # Start CPU memory tracking
    tracemalloc.start()

    # Train
    start_time = time.perf_counter()
    try:
        training_result = gp.fit(
            X,
            y,
            max_iterations=n_iterations,
            learning_rate=lr,
            initial_noise=init_noise,
            method=method,
            verbose=False,
        )
    finally:
        training_time = time.perf_counter() - start_time

        # Stop memory monitoring
        if monitor_memory:
            monitor.stop()
            memory_stats = monitor.get_stats()
            torch_stats = get_torch_memory_stats()
            memory_stats.update(torch_stats)

        # Get CPU memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)

    # Extract results
    final_nll = float(training_result.nll)
    nll_history = []
    if training_result.nll_history is not None:
        nll_history = np.asarray(training_result.nll_history, dtype=np.float32).tolist()
    iterations_run = int(training_result.iterations)

    # Determine if early stopped (if iterations < max and NLL stabilized)
    early_stopped = bool(training_result.converged or iterations_run < n_iterations)

    learned_params = _normalize_simple_learned_params(
        gp,
        kernel_type,
        init_ls=init_ls,
        init_noise=init_noise,
        init_os=init_os,
        kernel_param1=kernel_param1,
        kernel_param2=kernel_param2,
    )

    result = {
        "kernel_param1": 1.0,
        "kernel_param2": 0.0,
    }
    if kernel_type == "periodic":
        result["kernel_param1"] = learned_params["period"]
    elif kernel_type == "rq":
        result["kernel_param1"] = learned_params["alpha"]
    elif kernel_type == "polynomial":
        result["kernel_param1"] = learned_params["degree"]
        result["kernel_param2"] = learned_params["offset"]

    return {
        "gp": gp,
        "result": result,
        "final_nll": final_nll,
        "nll_history": nll_history,
        "training_time_s": training_time,
        "iterations_run": iterations_run,
        "max_iterations": n_iterations,
        "early_stopped": early_stopped,
        "learned_params": learned_params,
        "peak_memory_mb": memory_stats.get("max_mb", 0.0),
        "memory_stats": memory_stats,
        "lanczos_root": training_result.lanczos_root,
        "lanczos_rank": int(training_result.lanczos_rank),
    }


def predict_mojogp_simple(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    learned_params: Dict[str, float],
    kernel_type: str,
    method: str = "auto",
    lanczos_root: Optional[np.ndarray] = None,
    lanczos_rank: int = 0,
    kernel_param1: float = 1.0,
    kernel_param2: float = 0.0,
    variance_method: str = "exact",
) -> Dict[str, Any]:
    """Predict using a trained ExactGP instance cached in ``learned_params``.

    Args:
        X_train: Training data [n, d]
        y_train: Training targets [n]
        X_test: Test data [m, d]
        learned_params: Dict with lengthscale, outputscale, noise
        kernel_type: Kernel type string
        method: Prediction method ("auto", "materialized", "matrix_free")
        lanczos_root: Cached Lanczos root for variance (unused, kept for API compat)
        lanczos_rank: Lanczos rank (unused, kept for API compat)
        kernel_param1: Kernel-specific param (e.g., period for periodic)
        kernel_param2: Kernel-specific param (e.g., alpha for RQ)
        variance_method: "exact" (default, matches GPyTorch) or "love" (fast approximation)

    Returns:
        dict with keys:
            - mean: Predictive mean [m]
            - variance: Predictive variance [m]
            - std: Predictive std [m]
            - mean_time_s: Time for mean prediction
            - variance_time_s: Time for variance prediction
            - total_time_s: Total prediction time
    """
    # Ensure correct dtypes
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    y_train = np.ascontiguousarray(y_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    del X_train, y_train, method, lanczos_root, lanczos_rank, kernel_type
    del kernel_param1, kernel_param2

    gp = learned_params.get("_gp")
    if gp is None:
        raise ValueError(
            "predict_mojogp_simple() requires learned_params from train_mojogp_simple()"
        )

    start_time = time.perf_counter()
    pred = gp.predict(X_test, variance_method=variance_method)
    mean = np.asarray(pred.mean, dtype=np.float32)
    variance = np.asarray(pred.variance, dtype=np.float32)

    total_time = time.perf_counter() - start_time

    # Ensure positive variance
    variance = np.maximum(variance, 1e-10)
    std = np.sqrt(variance)

    # Split time evenly for compatibility (we don't have separate timings anymore)
    mean_time = total_time / 2
    variance_time = total_time / 2

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "mean_time_s": mean_time,
        "variance_time_s": variance_time,
        "total_time_s": total_time,
    }


# =============================================================================
# Composite API (JIT compilation)
# =============================================================================


def train_mojogp_composite(
    X: np.ndarray,
    y: np.ndarray,
    kernel_node,  # KernelNode from mojogp.kernel
    method: str = "auto",
    n_iterations: int = 100,
    lr: float = 0.02,
    init_noise: float = 0.1,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.1,
) -> Dict[str, Any]:
    """Train MojoGP using the composite (JIT) API via ExactGP.

    Args:
        X: Training data [n, d]
        y: Training targets [n]
        kernel_node: Kernel composition (e.g., Kernel.rbf() + Kernel.matern52())
        method: Training method
        n_iterations: Maximum iterations
        lr: Learning rate
        init_noise: Initial noise
        monitor_memory: Whether to monitor memory
        memory_poll_interval: Memory polling interval

    Returns:
        Same structure as train_mojogp_simple, plus:
            - gp: The trained ExactGP instance
    """
    from mojogp import SingleOutputGP

    # Ensure correct dtypes
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)

    # Create GP with training data
    gp = SingleOutputGP(kernel_node, verbose=False)

    # Setup memory monitoring
    memory_stats = {}
    if monitor_memory:
        reset_torch_memory_stats()
        monitor = GPUMemoryMonitor(interval=memory_poll_interval)
        monitor.start()

    tracemalloc.start()

    # Train
    start_time = time.perf_counter()
    try:
        training_result = gp.fit(
            X,
            y,
            initial_noise=init_noise,
            max_iterations=n_iterations,
            learning_rate=lr,
            method=method,
            verbose=False,
        )
    finally:
        training_time = time.perf_counter() - start_time

        if monitor_memory:
            monitor.stop()
            memory_stats = monitor.get_stats()
            torch_stats = get_torch_memory_stats()
            memory_stats.update(torch_stats)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)

    # Extract results
    final_nll = float(training_result.nll)
    iterations_run = int(training_result.iterations)
    early_stopped = training_result.converged or iterations_run < n_iterations

    params = np.asarray(training_result.params, dtype=np.float32)
    learned = gp.get_learned_params()
    learned_params = {
        "params": params,
        "noise": float(learned.get("noise", training_result.noise)),
        "lengthscale": _find_param_value(
            learned,
            ("_lengthscale", "_variance", "_ls_0"),
            float(params[0]) if len(params) > 0 else 1.0,
        ),
        "outputscale": _find_param_value(
            learned,
            ("_outputscale",),
            float(params[-1]) if len(params) > 0 else 1.0,
        ),
        "_gp": gp,
    }

    return {
        "gp": gp,
        "training_result": training_result,
        "final_nll": final_nll,
        "nll_history": [],  # Not available from ExactGP
        "training_time_s": training_time,
        "iterations_run": iterations_run,
        "max_iterations": n_iterations,
        "early_stopped": early_stopped,
        "learned_params": learned_params,
        "peak_memory_mb": memory_stats.get("max_mb", 0.0),
        "memory_stats": memory_stats,
        "lanczos_root": training_result.lanczos_root,
        "lanczos_rank": training_result.lanczos_rank,
    }


def predict_mojogp_composite(
    gp,  # ExactGP instance
    X_test: np.ndarray,
) -> Dict[str, Any]:
    """Predict using MojoGP composite API via ExactGP.

    Args:
        gp: Trained ExactGP instance
        X_test: Test data [m, d]

    Returns:
        Same structure as predict_mojogp_simple
    """
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)

    # Predict (ExactGP.predict returns PredictionResult with mean, variance, std)
    start_time = time.perf_counter()
    pred_result = gp.predict(X_test)
    total_time = time.perf_counter() - start_time

    return {
        "mean": np.asarray(pred_result.mean, dtype=np.float32),
        "variance": np.asarray(pred_result.variance, dtype=np.float32),
        "std": np.asarray(pred_result.std, dtype=np.float32),
        "mean_time_s": total_time / 2,  # Approximate split
        "variance_time_s": total_time / 2,
        "total_time_s": total_time,
    }


# =============================================================================
# Multi-Output API
# =============================================================================


def train_mojogp_multi_output(
    X: np.ndarray,
    Y: np.ndarray,
    kernel_type: Any,
    method: str = "materialized",
    n_iterations: int = 100,
    lr: float = 0.02,
    init_ls: float = 1.0,
    init_noise: float = 0.1,
    init_os: float = 1.0,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.1,
) -> Dict[str, Any]:
    """Train MojoGP multi-output GP via the public wrapper.

    Args:
        X: Training data [n, d]
        Y: Training targets [n, T] (T tasks)
        kernel_type: Kernel type string
        n_iterations: Maximum iterations
        lr: Learning rate
        init_ls: Initial lengthscale
        init_noise: Initial noise
        init_os: Initial outputscale
        monitor_memory: Whether to monitor memory
        memory_poll_interval: Memory polling interval

    Returns:
        Same structure as train_mojogp_simple
    """
    from mojogp import MultiOutputGP

    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)

    num_tasks = Y.shape[1]
    gp = MultiOutputGP(kernel=kernel_type)

    memory_stats = {}
    if monitor_memory:
        reset_torch_memory_stats()
        monitor = GPUMemoryMonitor(interval=memory_poll_interval)
        monitor.start()

    tracemalloc.start()

    start_time = time.perf_counter()
    try:
        result = gp.fit(
            X,
            Y,
            method=method,
            max_iterations=n_iterations,
            learning_rate=lr,
            initial_lengthscale=init_ls,
            initial_noise=init_noise,
            initial_outputscale=init_os,
            verbose=False,
        )
    finally:
        training_time = time.perf_counter() - start_time

        if monitor_memory:
            monitor.stop()
            memory_stats = monitor.get_stats()
            torch_stats = get_torch_memory_stats()
            memory_stats.update(torch_stats)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)

    final_nll = float(result.final_nll)
    iterations_run = int(result.iterations)
    early_stopped = bool(result.converged or iterations_run < n_iterations)

    learned_params = {
        "lengthscale": float(np.mean(result.params[:-1]))
        if len(result.params) > 1
        else init_ls,
        "outputscale": float(init_os),
        "noise": float(np.mean(result.noise_per_task)),
        "B": np.asarray(result.B, dtype=np.float32),
        "noise_per_task": np.asarray(result.noise_per_task, dtype=np.float32),
        "params": np.asarray(result.params, dtype=np.float32),
        "_gp": gp,
    }

    return {
        "result": result,
        "final_nll": final_nll,
        "nll_history": np.asarray(result.nll_history, dtype=np.float32).tolist()
        if result.nll_history is not None
        else [],
        "training_time_s": training_time,
        "iterations_run": iterations_run,
        "max_iterations": n_iterations,
        "early_stopped": early_stopped,
        "learned_params": learned_params,
        "peak_memory_mb": memory_stats.get("max_mb", 0.0),
        "memory_stats": memory_stats,
        "num_tasks": num_tasks,
    }


def predict_mojogp_multi_output(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    training_result: Dict[str, Any],
    kernel_type: str,
    variance_method: str = "exact",
) -> Dict[str, Any]:
    """Predict using a trained MultiOutputGP instance cached in the result.

    Args:
        X_train: Training data [n, d]
        Y_train: Training targets [n, T]
        X_test: Test data [m, d]
        training_result: Result from train_mojogp_multi_output
        kernel_type: Kernel type string

    Returns:
        dict with keys:
            - mean: Predictive mean [m, T]
            - variance: Predictive variance [m, T]
            - std: Predictive std [m, T]
            - mean_time_s, variance_time_s, total_time_s
    """
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    Y_train = np.ascontiguousarray(Y_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    del X_train, Y_train, kernel_type

    params = training_result["learned_params"]
    gp = params.get("_gp")
    if gp is None:
        raise ValueError(
            "predict_mojogp_multi_output() requires the result from train_mojogp_multi_output()"
        )

    start_time = time.perf_counter()
    mean, variance = gp.predict(X_test, return_var=True, variance_method=variance_method)
    total_time = time.perf_counter() - start_time

    variance = np.maximum(variance, 1e-10)
    std = np.sqrt(variance)

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "mean_time_s": total_time / 2,
        "variance_time_s": total_time / 2,
        "total_time_s": total_time,
    }


# =============================================================================
# ARD API
# =============================================================================


def train_mojogp_ard(
    X: np.ndarray,
    y: np.ndarray,
    kernel_type: str = "rbf",
    n_iterations: int = 100,
    lr: float = 0.01,
    init_noise: float = 0.1,
    init_os: float = 1.0,
    monitor_memory: bool = True,
    memory_poll_interval: float = 0.1,
) -> Dict[str, Any]:
    """Train MojoGP with ARD lengthscales via the public wrapper.

    Args:
        X: Training data [n, d]
        y: Training targets [n]
        n_iterations: Maximum iterations
        lr: Learning rate
        init_noise: Initial noise
        init_os: Initial outputscale
        monitor_memory: Whether to monitor memory
        memory_poll_interval: Memory polling interval

    Returns:
        Same structure as train_mojogp_simple, with:
            - learned_params['lengthscales']: Per-dimension lengthscales [d]
    """
    from mojogp import SingleOutputGP

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    gp = SingleOutputGP(
        _build_simple_kernel(
            kernel_type,
            init_ls=1.0,
            init_os=init_os,
            kernel_param1=None,
            kernel_param2=None,
            ard=True,
        ),
        verbose=False,
    )

    memory_stats = {}
    if monitor_memory:
        reset_torch_memory_stats()
        monitor = GPUMemoryMonitor(interval=memory_poll_interval)
        monitor.start()

    tracemalloc.start()

    start_time = time.perf_counter()
    try:
        result = gp.fit(
            X,
            y,
            max_iterations=n_iterations,
            learning_rate=lr,
            initial_noise=init_noise,
            method="materialized",
            verbose=False,
        )
    finally:
        training_time = time.perf_counter() - start_time

        if monitor_memory:
            monitor.stop()
            memory_stats = monitor.get_stats()
            torch_stats = get_torch_memory_stats()
            memory_stats.update(torch_stats)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_stats["cpu_peak_mb"] = peak / (1024 * 1024)

    final_nll = float(result.nll)
    iterations_run = int(result.iterations)
    early_stopped = bool(result.converged or iterations_run < n_iterations)

    learned = gp.get_learned_params()
    lengthscales = _extract_ard_lengthscales(learned)
    if lengthscales.size == 0:
        lengthscales = np.ones(X.shape[1], dtype=np.float32)

    learned_params = {
        "lengthscales": lengthscales,
        "lengthscale": float(np.mean(lengthscales)),  # Mean for compatibility
        "outputscale": _find_param_value(learned, ("_outputscale",), init_os),
        "noise": float(learned.get("noise", init_noise)),
        "_gp": gp,
    }

    return {
        "result": result,
        "final_nll": final_nll,
        "nll_history": np.asarray(result.nll_history, dtype=np.float32).tolist()
        if result.nll_history is not None
        else [],
        "training_time_s": training_time,
        "iterations_run": iterations_run,
        "max_iterations": n_iterations,
        "early_stopped": early_stopped,
        "learned_params": learned_params,
        "peak_memory_mb": memory_stats.get("max_mb", 0.0),
        "memory_stats": memory_stats,
    }


def predict_mojogp_ard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    training_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Predict using the trained ARD ExactGP cached in the result.

    Args:
        X_train: Training data [n, d]
        y_train: Training targets [n]
        X_test: Test data [m, d]
        training_result: Result from train_mojogp_ard

    Returns:
        dict with keys:
            - mean: Predictive mean [m]
            - variance: Predictive variance [m]
            - std: Predictive std [m]
            - mean_time_s, variance_time_s, total_time_s
    """
    del X_train, y_train
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)

    params = training_result["learned_params"]
    gp = params.get("_gp")
    if gp is None:
        raise ValueError(
            "predict_mojogp_ard() requires the result from train_mojogp_ard()"
        )

    start_time = time.perf_counter()
    pred = gp.predict(X_test, variance_method="exact")
    total_time = time.perf_counter() - start_time

    mean = np.asarray(pred.mean, dtype=np.float32)
    variance = np.maximum(np.asarray(pred.variance, dtype=np.float32), 1e-10)
    std = np.asarray(pred.std, dtype=np.float32)
    mean_time = total_time / 2
    variance_time = total_time / 2

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "mean_time_s": mean_time,
        "variance_time_s": variance_time,
        "total_time_s": total_time,
    }


# =============================================================================
# Full Benchmark Runner
# =============================================================================


def run_mojogp_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    f_test: np.ndarray,  # Ground truth (noiseless)
    kernel_type: str,
    method: str = "auto",
    n_iterations: int = 100,
    lr: float = 0.02,
    init_ls: float = 1.0,
    init_noise: float = 0.1,
    init_os: float = 1.0,
    true_params: Optional[Dict[str, float]] = None,
    monitor_memory: bool = True,
    variance_method: str = "exact",
) -> BenchmarkResult:
    """Run a complete MojoGP benchmark and return structured results.

    Args:
        X_train, y_train: Training data
        X_test: Test inputs
        f_test: Ground truth test outputs (noiseless)
        kernel_type: Kernel type string
        method: Training method
        n_iterations: Max iterations
        lr: Learning rate
        init_ls, init_noise, init_os: Initial hyperparameters
        true_params: True hyperparameters (for recovery error calculation)
        monitor_memory: Whether to monitor GPU memory

    Returns:
        BenchmarkResult with all metrics
    """
    # Train
    train_result = train_mojogp_simple(
        X_train,
        y_train,
        kernel_type,
        method,
        n_iterations,
        lr,
        init_ls,
        init_noise,
        init_os,
        monitor_memory=monitor_memory,
    )

    # Predict
    pred_result = predict_mojogp_simple(
        X_train,
        y_train,
        X_test,
        train_result["learned_params"],
        kernel_type,
        method,
        train_result.get("lanczos_root"),
        train_result.get("lanczos_rank", 0),
        variance_method=variance_method,
    )

    # Compute accuracy metrics
    accuracy_metrics = compute_all_accuracy_metrics(
        f_test,
        pred_result["mean"],
        pred_result["std"],
        y_train_mean=float(np.mean(y_train)),
        y_train_std=float(np.std(y_train)),
    )

    # Build result objects
    accuracy = AccuracyResult(
        rmse=accuracy_metrics["rmse"],
        mae=accuracy_metrics["mae"],
        r_squared=accuracy_metrics["r_squared"],
        crps=accuracy_metrics["crps"],
        msll=accuracy_metrics["msll"],
        calibration_coverage={
            0.5: accuracy_metrics["calibration_50"],
            0.9: accuracy_metrics["calibration_90"],
            0.95: accuracy_metrics["calibration_95"],
            0.99: accuracy_metrics["calibration_99"],
        },
        calibration_error=accuracy_metrics["calibration_error"],
        sharpness=accuracy_metrics["sharpness"],
        interval_width_95=accuracy_metrics["interval_width_95"],
    )

    speed = SpeedResult(
        training_time_s=train_result["training_time_s"],
        prediction_mean_time_s=pred_result["mean_time_s"],
        prediction_variance_time_s=pred_result["variance_time_s"],
        end_to_end_time_s=train_result["training_time_s"] + pred_result["total_time_s"],
        iterations_run=train_result["iterations_run"],
        max_iterations=train_result["max_iterations"],
        early_stopped=train_result["early_stopped"],
        ms_per_iteration=(
            train_result["training_time_s"] / max(train_result["iterations_run"], 1)
        )
        * 1000,
    )

    memory_stats = train_result.get("memory_stats", {})
    memory = MemoryResult(
        gpu_mean_mb=memory_stats.get("mean_mb", 0.0),
        gpu_min_mb=memory_stats.get("min_mb", 0.0),
        gpu_max_mb=memory_stats.get("max_mb", 0.0),
        gpu_var_mb=memory_stats.get("var_mb", 0.0),
        torch_peak_mb=memory_stats.get("torch_peak_mb", 0.0),
        torch_current_mb=memory_stats.get("torch_current_mb", 0.0),
        cpu_peak_mb=memory_stats.get("cpu_peak_mb", 0.0),
        measurement_method=memory_stats.get("method", "none"),
        num_samples=memory_stats.get("samples", 0),
    )

    learned = train_result["learned_params"]
    hyperparameters = HyperparameterResult(
        learned_lengthscale=learned["lengthscale"],
        learned_noise=learned["noise"],
        learned_outputscale=learned["outputscale"],
        final_nll=train_result["final_nll"],
    )

    # Compute recovery errors if true params provided
    if true_params:
        if "lengthscale" in true_params:
            hyperparameters.lengthscale_rel_error = param_relative_error(
                learned["lengthscale"], true_params["lengthscale"]
            )
        if "noise" in true_params:
            hyperparameters.noise_rel_error = param_relative_error(
                learned["noise"], true_params["noise"]
            )
        if "outputscale" in true_params:
            hyperparameters.outputscale_rel_error = param_relative_error(
                learned["outputscale"], true_params["outputscale"]
            )

    config = {
        "kernel": kernel_type,
        "n": len(X_train),
        "d": X_train.shape[1],
        "method": method,
        "variance_method": variance_method,
        "n_iterations": n_iterations,
        "lr": lr,
    }

    return BenchmarkResult(
        config=config,
        accuracy=accuracy,
        speed=speed,
        memory=memory,
        hyperparameters=hyperparameters,
    )
