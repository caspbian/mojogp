# Quickstart

These examples use the public wrapper APIs and avoid in-development routes.

## SingleOutputGP

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

## Fixed Observation Noise

Use `observation_noise` when each training point has a known variance. MojoGP trains the exact covariance `K + diag(observation_noise)`.

```python
import numpy as np
from mojogp import RBF, SingleOutputGP

X = np.linspace(-3, 3, 5000, dtype=np.float32).reshape(-1, 1)
noise = (0.01 + 0.04 * (X[:, 0] > 0)).astype(np.float32)
y = (np.sin(2.5 * X[:, 0]) + np.random.randn(5000) * np.sqrt(noise)).astype(np.float32)

gp = SingleOutputGP(RBF())
gp.fit(
    X,
    y,
    observation_noise=noise,
    learn_noise=False,
    method="matrix_free",
    preconditioner_rank=10,
)

X_test = X[:128]
observed = gp.predict_observed(X_test, observation_noise=noise[:128])
```

## Fixed Grouped Observation Noise

Use `noise_model="grouped"` when each training point belongs to a known group with a known variance.

```python
groups = (X[:, 0] > 0).astype(np.int32)
group_noise = np.array([0.01, 0.05], dtype=np.float32)

gp = SingleOutputGP(RBF())
gp.fit(
    X,
    y,
    noise_model="grouped",
    noise_group_train=groups,
    group_noise=group_noise,
    learn_noise=False,
    method="matrix_free",
    preconditioner_rank=10,
)

observed = gp.predict_observed(X_test, noise_group_test=groups[:128])
```

Learned free per-sample and learned grouped noise routes are in development. They are not public feature claims.

## Composite Kernels

```python
from mojogp import Matern52, RBF, SingleOutputGP

kernel = RBF(active_dims=[0, 1]) + Matern52(active_dims=[1, 2])
gp = SingleOutputGP(kernel)
gp.fit(X, y, max_iterations=50, method="materialized")
pred = gp.predict(X_test, variance_method="exact")
```

## Multi-Output

```python
import numpy as np
from mojogp import Kernel, MultiOutputGP, MultiOutputLMCGP

X = np.random.randn(5000, 3).astype(np.float32)
X_test = np.random.randn(128, 3).astype(np.float32)
Y = np.random.randn(5000, 2).astype(np.float32)

icm = MultiOutputGP(kernel=Kernel.rbf())
icm.fit(X, Y, max_iterations=30, method="matrix_free")
mean, var = icm.predict(X_test, return_var=True)

lmc = MultiOutputLMCGP(kernels=[Kernel.rbf(), Kernel.matern52()])
lmc.fit(X, Y, max_iterations=30, method="materialized")
samples = lmc.sample_posterior(X_test[:32], n_samples=4, method="diagonal")
```

Check [Support Status](../features/support-status.md) and [Feature Matrix](../FEATURE_MATRIX.md) before using mixed, LMC, learned-noise, or sampling combinations outside these examples.
