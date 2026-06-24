"""Tiny subprocess probe used to confirm benchmark profiling is off."""

from __future__ import annotations

import numpy as np

from mojogp import SingleOutputGP, RBF


def main() -> int:
    rng = np.random.RandomState(0)
    x = rng.randn(2000, 2).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.1 * rng.randn(2000)).astype(np.float32)
    gp = SingleOutputGP(RBF(), verbose=False)
    gp.fit(
        x,
        y,
        method="materialized",
        max_iterations=1,
        learning_rate=0.01,
        num_probes=2,
        max_cg_iterations=10,
        preconditioner_rank=4,
        max_tridiag_iterations=4,
        verbose=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
