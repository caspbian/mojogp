"""Progress callback helpers for JIT engine routes."""

from python import Python, PythonObject


fn progress_interval_should_emit(
    iteration: Int,
    max_iterations: Int,
    progress_interval: Int,
) -> Bool:
    var current = iteration + 1
    if current <= 1:
        return True
    if current >= max_iterations:
        return True
    var interval = progress_interval
    if interval <= 0:
        interval = 1
    return current % interval == 0


fn emit_progress_event(
    callback: PythonObject,
    operation: String,
    model: String,
    route: String,
    phase: String,
    current: Int,
    total: Int,
    nll: Float32 = Float32(0.0),
    best_nll: Float32 = Float32(0.0),
    cg_iter: Int = -1,
    iter_time_ns: Int = -1,
    noise: Float32 = Float32(-1.0),
    mean: Float32 = Float32(0.0),
    precond_rank: Int = -1,
    precond_rebuild_count: Int = -1,
    converged: Bool = False,
) raises:
    var event = Python.dict()
    event["operation"] = operation
    event["model"] = model
    event["route"] = route
    event["phase"] = phase
    event["current"] = current
    event["total"] = total
    event["converged"] = converged

    var stats = Python.dict()
    stats["nll"] = Float64(nll)
    stats["best_nll"] = Float64(best_nll)
    if cg_iter >= 0:
        stats["cg_iter"] = cg_iter
    if iter_time_ns >= 0:
        stats["iter_ms"] = Float64(iter_time_ns) / 1e6
    if noise >= Float32(0.0):
        stats["noise"] = Float64(noise)
    stats["mean"] = Float64(mean)
    if precond_rank >= 0:
        stats["precond_rank"] = precond_rank
    if precond_rebuild_count >= 0:
        stats["precond_rebuild_count"] = precond_rebuild_count
    event["stats"] = stats

    _ = callback.__call__(event)
