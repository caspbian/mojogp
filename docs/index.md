# MojoGP

MojoGP is an exact Gaussian Process regression library with JIT-compiled Mojo GPU kernels.

The current public surface is centered on three Python wrappers:

1. `SingleOutputGP` for single-output exact GPs.
2. `MultiOutputGP` for ICM-style multi-output exact GPs.
3. `MultiOutputLMCGP` for LMC-style multi-output exact GPs.

MojoGP is exact-GP only. It uses CG, SLQ, pivoted-Cholesky preconditioning, and LOVE-style prediction accelerators where documented, but it does not use inducing points, variational inference, sparse GP approximations, SVI, or surrogate training objectives.

## Start Here

1. [Installation](getting-started/installation.md)
2. [Quickstart](getting-started/quickstart.md)
3. [Support Status](features/support-status.md)
4. [Feature Matrix](FEATURE_MATRIX.md)
5. [API Surface](API.md)
6. [API Reference](reference/index.md)

## Support Boundary

Feature status is evidence-based. Code existence alone is not a public support claim.

Use `docs/FEATURE_MATRIX.md` as the canonical status matrix for public maturity labels. Any feature marked `in-dev` may have implementation work in the repository, but it is not public functionality and may raise `NotImplementedError` or emit an in-development warning when reached.

## Minimal Example

```python
import numpy as np
from mojogp import RBF, SingleOutputGP

X = np.random.randn(5000, 2).astype(np.float32)
y = (np.sin(X[:, 0]) + 0.1 * np.random.randn(5000)).astype(np.float32)

gp = SingleOutputGP(RBF(ard=True))
gp.fit(X, y, max_iterations=50, method="matrix_free")

X_test = np.random.randn(128, 2).astype(np.float32)
mean, std = gp.predict(X_test, return_std=True)
```

See [Quickstart](getting-started/quickstart.md) for fixed-noise, grouped-noise, composite-kernel, and multi-output examples.
