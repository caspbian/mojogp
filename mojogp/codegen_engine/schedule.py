"""GPU kernel schedule planner.

Decides TM (tile-rows per thread), shmem usage, unroll factors,
and NCOLS specialization based on register budget and kernel properties.
"""

from dataclasses import dataclass, field
from .ir import IRKernel


_DEFAULT_DYNAMIC_SHARED_MEM_BYTES = 48 * 1024


@dataclass
class ScheduleConfig:
    """GPU kernel execution configuration."""

    tm: int = 4  # tile-rows per thread
    use_shmem: bool = True  # shared memory tiling
    j_unroll: int = 1  # j-loop unroll factor (only for non-shmem)
    ncols: list = field(default_factory=lambda: [11, 6, 1])  # compile-time NCOLS
    block_size: int = 256  # threads per block
    max_registers: int = 200  # target register budget (headroom below 255)
    precompute_inv_ls: bool = False  # precompute 1/l_d
    splitj_forward: bool = False  # use split-j forward by default for measured routes

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduleConfig":
        """Create from a dict (e.g. loaded from TOML)."""
        return cls(
            tm=d.get("tm", 4),
            use_shmem=d.get("use_shmem", True),
            j_unroll=d.get("j_unroll", 1),
            ncols=d.get("ncols", [11, 6, 1]),
            block_size=d.get("block_size", 256),
            max_registers=d.get("max_registers", 200),
            precompute_inv_ls=d.get("precompute_inv_ls", False),
            splitj_forward=d.get("splitj_forward", False),
        )


def estimate_registers(kernel: IRKernel, tm: int, ncols: int) -> int:
    """Estimate GPU register usage for a kernel configuration.

    This is a heuristic — actual register usage depends on compiler decisions.
    """
    dim = kernel.dim
    num_params = kernel.num_params

    regs = 0
    regs += dim * tm  # x_row (TM copies)
    regs += ncols * tm  # acc (per-col, per-TM-row)
    regs += num_params  # params in registers
    regs += num_params * ncols  # gradient accumulators (fused path, single TM row)
    regs += len(kernel.lets)  # CSE temporaries
    regs += dim  # diffs
    regs += 10  # loop vars, misc (tid, bs, i, j, sb, etc.)

    return regs


def plan_schedule(
    kernel: IRKernel,
    ncols_hint: list = None,
    schedule_policy_tag: str | None = None,
) -> ScheduleConfig:
    """Auto-select optimal schedule based on kernel properties.

    Tries TM=4 first (best for simple kernels), falls back to TM=2, TM=1
    based on register pressure.
    """
    config = ScheduleConfig()
    if ncols_hint:
        config.ncols = ncols_hint
    elif kernel.num_params > 6:
        # Adaptive NCOLS: more params → fewer simultaneous CG columns
        # to stay within register budget (e.g., ARD with d>5)
        config.ncols = [6, 1]

    # Generated shmem kernels do not opt into larger dynamic shared memory, so
    # keep NCOLS within the conservative 48 KiB launch envelope. This avoids
    # emitting specializations that can compile but fail at runtime with
    # CUDA_ERROR_INVALID_VALUE once shared_mem_bytes exceeds the default limit.
    max_shared_floats = _DEFAULT_DYNAMIC_SHARED_MEM_BYTES // (config.block_size * 4)
    config.ncols = [nc for nc in config.ncols if kernel.dim + nc <= max_shared_floats]
    if not config.ncols:
        config.ncols = [1]

    max_ncols = max(config.ncols)

    # NCOLS=10 was measured as a win for RBF d=17 because training's 10 probe
    # columns otherwise fall through to slower per-parameter gradient kernels.
    # Keep this scoped to RBF leaves; other kernel/model families need their own
    # A/B evidence before changing specializations.
    if schedule_policy_tag == "rbf_leaf" and kernel.dim <= 17:
        if not ncols_hint:
            config.ncols = [nc for nc in [11, 10, 6, 1] if kernel.dim + nc <= max_shared_floats]
            if not config.ncols:
                config.ncols = [1]
        config.tm = 1
        config.use_shmem = True
        config.splitj_forward = kernel.num_params == 2
        return config

    if schedule_policy_tag == "low_d_stationary_leaf" and kernel.dim <= 10:
        if not ncols_hint:
            config.ncols = [nc for nc in [11, 10, 6, 1] if kernel.dim + nc <= max_shared_floats]
            if not config.ncols:
                config.ncols = [1]
        config.tm = 1
        config.use_shmem = True
        return config

    if kernel.num_params <= 2 and kernel.dim <= 17:
        config.tm = 1
        config.use_shmem = True
        return config

    # Try TM=4 first (best performance for simple kernels)
    regs_tm4 = estimate_registers(kernel, tm=4, ncols=max_ncols)
    if regs_tm4 <= config.max_registers:
        config.tm = 4
        config.use_shmem = True
        return config

    # Fall back to TM=2
    regs_tm2 = estimate_registers(kernel, tm=2, ncols=max_ncols)
    if regs_tm2 <= config.max_registers:
        config.tm = 2
        config.use_shmem = True
        return config

    # Fall back to TM=1 with shmem
    regs_tm1 = estimate_registers(kernel, tm=1, ncols=max_ncols)
    if regs_tm1 <= config.max_registers:
        config.tm = 1
        config.use_shmem = True
        return config

    # Last resort: TM=1 without shmem (split gradient into separate kernels)
    config.tm = 1
    config.use_shmem = False
    config.j_unroll = 4
    return config
