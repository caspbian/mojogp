# MojoGP API

This document describes the live public Python API.

The generated feature/status matrix is maintained in `docs/FEATURE_MATRIX.md`.

## Primary Classes

### `SingleOutputGP`

Single-output exact GP wrapper.

Constructor:

```python
SingleOutputGP(kernel, *, init_mean=None, verbose=False)
```

Core methods:

```python
fit(
    X,
    y,
    max_iterations=100,
    learning_rate=0.01,
    initial_noise=0.1,
    initial_params=None,
    method="auto",
    enable_early_stopping=False,
    early_stop_patience=10,
    early_stop_tol=1e-4,
    verbose=None,
    preset=None,
    max_cg_iterations=None,
    cg_tolerance=None,
    num_probes=None,
    max_tridiag_iterations=None,
    preconditioner_rank=None,
    precond_rebuild_threshold=None,
    use_fused_kernels=True,
    preconditioner=None,
    use_preconditioner=None,
    lr_schedule="constant",
    prepare_prediction_cache=False,
    prediction_cache_rank=None,
    observation_noise=None,
    observation_noise_fn=None,
    noise_function=None,
    learn_noise=True,
    noise_floor=1e-6,
    noise_regularization=0.01,
    noise_model="scalar",
    noise_group_train=None,
    group_noise=None,
    progress=None,
    progress_stats=None,
    progress_interval=1,
)

predict(
    X,
    return_var=False,
    return_std=False,
    variance_method="love",
    method=None,
    target="latent",
    observation_noise=None,
    observation_noise_fn=None,
    noise_group_test=None,
    *,
    max_cg_iterations=None,
    cg_tolerance=None,
    preconditioner_rank=None,
    max_root_decomposition_size=None,
    exact_prediction_block_cols=None,
    progress=None,
    progress_stats=None,
)

predict_latent(X, **kwargs)
predict_observed(X, observation_noise=None, *, observation_noise_fn=None, noise_group_test=None, **kwargs)

prepare_prediction_cache(
    variance_method="love",
    method=None,
    *,
    max_cg_iterations=None,
    cg_tolerance=None,
    preconditioner_rank=None,
    max_root_decomposition_size=None,
)

sample_posterior(X_test, n_samples=1, method="diagonal", n_rff_features=1024, rng=None)
save(path)
SingleOutputGP.load(path, kernel=None)
get_learned_params()
```

Prediction behavior:

1. default return is `PredictionResult(mean, variance, std)`
2. `return_var=True` returns `(mean, variance)`
3. `return_std=True` returns `(mean, std)`

Public option values:

1. training `method`: `"matrix_free"` (`"mf"`), `"materialized"` (`"mat"`); `SingleOutputGP.fit` also accepts `"auto"`
2. `variance_method`: `"love"`, `"exact"`, `"mean_only"`
3. sampling `method`: `"diagonal"`, `"pathwise"`

Progress reporting:

1. `progress=None` uses the global progress default, which is enabled by default; `progress=False` is a strict no-op path and does not allocate a reporter or pass a callback to the backend.
2. The default reporter uses `tqdm.auto` when available, with notebook and standard tqdm fallbacks before a simple stderr fallback if `tqdm` is not installed. `progress="auto"` enables the default reporter only for interactive stderr.
3. `progress=callback` calls `callback(event)` with a `ProgressEvent`. A reporter object with `start(event)`, `update(event)`, and `close(event)` is also accepted.
4. `progress_stats` filters the default reporter postfix by stat name, or may be a callable that returns a stats dict for each event.
5. `progress_interval` applies only to ordinary training-iteration updates. Start, completion, early-stop, and NaN events are always emitted when progress is enabled.
6. Prediction progress reports coarse wrapper phases and already-available backend metadata. It does not claim per-test-point progress for monolithic backend operations.

Global defaults:

```python
from mojogp import progress_enabled, set_progress_enabled

set_progress_enabled("auto")

with progress_enabled(True):
    gp.fit(X, y)
```

#### SingleOutput Continuous Support Notes

| Surface | State | Evidence notes |
|---|---|---|
| Exact and LOVE variance | Alpha on continuous SingleOutput scope | Materialized and matrix-free routes have exact/LOVE alignment checks, route-metadata assertions, and the active `single_output_variance_modes` benchmark harness comparing LOVE against exact prediction on the same trained state. |
| Pathwise sampling | Alpha on continuous SingleOutput scope | Provider-backed correction is tested for finite samples, reproducibility, save/load, route metadata, predictive-mean consistency, spatial correlation, and dense posterior covariance sanity at realistic `n=2000` sizes. The active `single_output_sampling` benchmark harness covers diagonal and pathwise workflow timing/memory rows with hard route/moment assertions. |
| Polynomial kernel primitive | Alpha on fixed-positive-integer SingleOutput scope | Polynomial degree is structural and fixed during training; offset/outputscale remain trainable. Unit tests cover public/evaluation/codegen parameter contracts and exact feature-map covariance, while integration tests cover dense-reference exact prediction, exact/LOVE prediction, pathwise sampling, save/load, and metadata. |
| Polynomial pathwise sampling | Alpha on fixed-positive-integer SingleOutput scope | Polynomial pathwise uses an exact monomial feature map for fixed positive integer degree and non-negative offset/outputscale. Excessive exact feature expansion raises `NotImplementedError` instead of silently changing route. |
| Learned input-dependent noise | Alpha on continuous SingleOutput linear scope | `noise_model="learned_input_dependent"` with `noise_function="linear"` has gradient reference checks, materialized and matrix-free fit/predict/save-load integration coverage, system recovery evidence for monotone variance, and matrix-free memory scaling checks. Learned free per-sample and grouped noise remain in development. |

### `MultiOutputGP`

ICM-style multi-output exact GP wrapper.

Constructor:

```python
MultiOutputGP(
    kernel="rbf",
    task_rank=-1,
    ard=False,
    num_probes=None,
    max_cg_iterations=None,
    cg_tolerance=None,
    preconditioner_rank=None,
    precond_rebuild_threshold=None,
    preconditioner=None,
    use_preconditioner=None,
    init_mean=None,
    preset=None,
    max_tridiag_iterations=None,
)
```

Core methods:

```python
fit(
    X,
    Y,
    max_iterations=100,
    learning_rate=0.05,
    method="materialized",
    initial_lengthscale=1.0,
    initial_lengthscales=None,
    initial_params=None,
    initial_noise=0.1,
    initial_noise_per_task=None,
    input_dependent_noise=None,
    grouped_noise=None,
    initial_outputscale=1.0,
    verbose=False,
    early_stop_tol=1e-4,
    early_stop_patience=15,
    use_fused_kernels=True,
    lr_schedule="constant",
    observation_noise=None,
    observation_noise_fn=None,
    noise_model="scalar",
    noise_group_train=None,
    group_noise=None,
    progress=None,
    progress_stats=None,
    progress_interval=1,
)

predict(
    X_test,
    return_var=False,
    return_std=False,
    variance_method="love",
    progress=None,
    progress_stats=None,
)
predict_latent(X_test, return_var=False, return_std=False, variance_method="love", **kwargs)
predict_observed(
    X_test,
    observation_noise=None,
    noise_group_test=None,
    return_var=False,
    return_std=False,
    variance_method="love",
    **kwargs,
)
sample_posterior(X_test, n_samples=1, method="diagonal", n_rff_features=1024, rng=None)
save(path)
MultiOutputGP.load(path, kernel=None)
```

Continuous `MultiOutputGP` supports fixed per-sample-task observation noise through
`observation_noise`, and fixed grouped noise through
`noise_model="grouped"`, `noise_group_train`, and `group_noise`. The
`input_dependent_noise` and `grouped_noise` keyword placeholders remain reserved
for unsupported future surfaces and raise when used directly.

### `MultiOutputLMCGP`

LMC-style multi-output exact GP wrapper. Training uses the exact LMC marginal
log-likelihood with either materialized dense kernels or matrix-free BBMM/CG.
LOVE is a fast approximate predictive variance route; it is not a training
approximation.

Support boundary: `MultiOutputLMCGP` is exact GP only. It uses the exact LMC marginal likelihood with dense materialized solves or matrix-free CG/SLQ solves. It does not use inducing points, variational inference, sparse GP approximations, SVI, or surrogate objectives.

#### LMC Support Matrix

| Surface | State | Notes |
|---|---|---|
| Continuous RBF and Matern latents | Alpha on tested scope | Compiled JIT routes, dense-reference integration checks, minimal LMC system gates, and workflow benchmark gates cover materialized and matrix-free training with exact and LOVE prediction. |
| RQ, Periodic, Linear latents | Alpha on tested scope | Materialized and matrix-free route tests cover exact and LOVE variance metadata plus save/load exact and LOVE round trips. LOVE is approximate and uses bounded variance drift after provider rebuild. |
| Polynomial latents | Alpha for fixed degree 1 or 2 pathwise scope | Degree is fixed positive-integer kernel structure, not a learned hyperparameter. Degree-2 route checks cover materialized and matrix-free exact/LOVE prediction and save/load; pathwise sampling is covered publicly for degree 1 or 2 feature maps. Broader fixed-integer feature maps are capped and remain evidence-limited. |
| Heterogeneous continuous latents | Alpha on tested scope | Different latent kernel trees are serialized per latent and covered by dense-reference, exact/LOVE, save/load, and sampling tests on representative RBF/Matern/RQ/Periodic/Linear/Polynomial combinations. |
| `ard=True` | Alpha on tested continuous scope | ARD applies once per continuous latent after active-dim/categorical remapping. RBF/Matern-style LMC ARD covers materialized and matrix-free per-dimension metadata, prediction, save/load, active dims, route metadata, and a narrow synthetic single-relevant-dimension recovery gate. Arbitrary relevance recovery remains workflow-dependent and is not a public guarantee. Mixed ARD is limited to unambiguous continuous dimensions after categorical splitting. |
| Active dims | Alpha on tested scope | Active dims are per latent, are serialized through kernel trees, and have exact/LOVE/save-load route coverage including overlapping per-latent active dimensions. |
| Composite latent kernels | Alpha on targeted continuous scope; mixed composites vary by matrix entry | Sum/product/nested/scale routes have materialized and matrix-free exact/LOVE metadata checks, exact/LOVE save/load checks, and pathwise coverage where feature-map expansion is supported. Mixed composite routes remain feature-specific: product routes are experimental, while several additive/nested mixed combinations remain in development. |
| Mixed continuous-categorical latents | Experimental on targeted mixed scope | Tested structures include mixed product routes with at least one continuous component; additive/nested mixed trees have narrower route-specific checks. Pure categorical LMC remains unsupported. `GD`, `CR`, `EHH`, `HH`, and `FE` have targeted materialized and matrix-free categorical-sensitivity, exact/LOVE, save/load, and route-metadata checks; broader cross-framework benchmark evidence is still in development. |
| Pure categorical LMC | Unsupported | `fit()` raises a clear `ValueError`; exact GP support would need a separate evidence plan. |
| Learned per-task noise `[T]` | Alpha on tested narrow scope | Noise is learned as exact diagonal observation noise per task and is covered by compiled LMC integration and system benchmark gates. |
| Fixed per-sample-per-task noise `[n, T]` | Alpha on targeted continuous LMC scope | Exact fixed diagonal observation-noise code is covered by dense-reference persistence checks, matrix-free exact-variance memory scaling, and pathwise sampling smoke coverage. Mixed LMC and separately certified preconditioned fixed-noise routes remain unsupported for this noise tier. |
| Grouped or input-dependent noise | Unsupported | Deferred follow-on surfaces. |
| Diagonal sampling | Alpha on tested narrow scope | Independent marginal predictive samples from the returned predictive variance; the LMC workflow benchmark hard-asserts finite samples and moment consistency. |
| Pathwise sampling | Alpha on tested narrow scope | Supported for current feature-map kernels, including exact polynomial degree 1 or 2 feature maps; broader polynomial degrees and excessive product expansions remain boundary-checked. The LMC workflow benchmark hard-asserts finite samples and moment consistency. |
| Save/load | Alpha on tested narrow scope | Saved kernel trees allow load-time prediction without user-supplied kernels; workflow benchmarks hard-assert compiled round trips. |

#### LMC Predictive Variance Semantics

`predict(..., variance_method="exact")` and
`variance_method="love"` return marginal predictive observation variance by
default, with learned per-task observation noise included. `variance_method="mean_only"`
returns no variance. Exact materialized continuous LMC uses the full dense LMC
posterior reference route for small/medium validation; matrix-free prediction
must use backend routes and refuses dense Python train-train fallback.

`backend_predict_info["lmc_variance_exactness"]` records the variance route's
posterior-diagonal semantics. `dense_exact_lmc`, `predict_lmc_full_exact`, and
`predict_lmc_mixed_full_exact` report `"exact_full_lmc_covariance"` because they
solve against the full LMC train covariance for the requested posterior diagonal.
LOVE routes such as `predict_lmc` and `predict_lmc_mixed` report
`"scalar_latent_approximation"`: they compute fast scalar latent variances and mix
them back to tasks before adding learned per-task observation noise. This LOVE
variance approximation is a prediction accelerator for the trained exact model,
not a training approximation.

The exact LMC train covariance is `sum_s kron(K_s, A_s) + I_n kron(D_task)`, where `A_s` is the learned coregionalization matrix for latent `s` and `D_task` is the learned per-task diagonal noise. The LMC cross covariance is `sum_s kron(K_cross_s, A_s)`.

Fixed observation noise: `fit(..., fixed_observation_noise=array)` accepts a
non-negative `float32` array with shape `[n, T]` on the targeted continuous LMC
routes. These values are added to the exact training covariance diagonal, are
serialized, and are included by continuous LMC exact prediction and pathwise
correction routes. The learned `noise_per_task` remains a separate per-task test
observation-noise term. Mixed continuous-categorical LMC rejects fixed `[n, T]`
noise until its separate route evidence is complete.

Route metadata: after `fit()`, `predict()`, or `sample_posterior()`, inspect
`backend_train_info`, `backend_predict_info`, and `backend_sample_info`. LMC
tests assert fields such as `training_route`, `backend_predict_info["actual_prediction_route"]`,
`backend_predict_info["actual_variance_route"]`, `variance_method`, and sampling correction routes.

#### Mixed LMC Supported Patterns

| Latent kernel pattern | Status | Notes |
|---|---|---|
| Continuous-only latent kernels | Alpha on documented continuous LMC scope | RBF/Matern have the strongest route evidence; RQ, Periodic, Linear, and Polynomial have integration coverage but still need broader benchmark evidence before any broader maturity claim. |
| Continuous `*` categorical subtrees with explicit active dims | Experimental on targeted mixed LMC scope | Current public certification tests exercise `RBF * {GD, CR, EHH, HH, FE}` and `Matern52 * {GD, CR, EHH, HH, FE}` on materialized and matrix-free mixed LMC routes. Additive/nested mixed LMC trees remain narrower until route-matrix and scaling evidence are added. |
| Heterogeneous continuous plus mixed latent lists | Experimental on targeted mixed LMC scope | Supported when every mixed latent includes a continuous component and categorical columns are represented by documented categorical subtrees. Unsupported or in-development sub-combinations still raise according to the feature matrix. |
| Pure categorical latent kernels | Unsupported | Raises a clear boundary error; use a continuous component or a different model family. |
| Categorical-only LMC models | Unsupported | LMC mixed routing is not a sparse or variational categorical-only approximation. |

#### LMC Sampling Boundaries

`method="diagonal"` draws independent samples from the marginal predictive distribution implied by the returned mean and diagonal variance. `method="pathwise"` uses feature-map prior samples plus backend correction on supported continuous and mixed feature-map structures. Polynomial LMC pathwise sampling is publicly evidenced for degree 1 or 2 feature maps; broader fixed non-negative integer monomial maps are capped and remain evidence-limited. Excessive product feature expansions raise clear errors instead of silently changing route.

Constructor:

```python
MultiOutputLMCGP(
    kernels,
    num_probes=10,
    max_cg_iterations=200,
    cg_tolerance=1.0,
    preconditioner_rank=15,
    precond_rebuild_threshold=0.5,
    preconditioner="greedy",
    use_preconditioner=None,
    max_tridiag_iterations=30,
    ard=False,
    init_mean=None,
)
```

Core methods:

```python
fit(
    X,
    Y,
    max_iterations=100,
    learning_rate=0.05,
    method="materialized",
    initial_lengthscales=None,
    initial_params=None,
    initial_noise=0.1,
    initial_noise_per_task=None,
    fixed_observation_noise=None,
    input_dependent_noise=None,
    grouped_noise=None,
    verbose=False,
    early_stop_tol=1e-4,
    early_stop_patience=15,
    use_fused_kernels=True,
    observation_noise=None,
    observation_noise_fn=None,
    noise_model="scalar",
    noise_group_train=None,
    group_noise=None,
    progress=None,
    progress_stats=None,
    progress_interval=1,
)

predict(
    X_test,
    return_var=False,
    return_std=False,
    variance_method="love",
    progress=None,
    progress_stats=None,
)
predict_latent(X_test, return_var=False, return_std=False, variance_method="love", **kwargs)
predict_observed(
    X_test,
    observation_noise=None,
    noise_group_test=None,
    return_var=False,
    return_std=False,
    variance_method="love",
    **kwargs,
)
sample_posterior(X_test, n_samples=1, method="diagonal", n_rff_features=1024, rng=None)
save(path)
MultiOutputLMCGP.load(path, kernels=None)
```

Continuous `MultiOutputLMCGP` supports fixed per-sample-task observation noise
through `fixed_observation_noise`. The `observation_noise`,
`observation_noise_fn`, `noise_model`, `noise_group_train`, and `group_noise`
parameters are present for API consistency but are not public LMC noise surfaces
today; non-scalar or grouped uses raise clear `NotImplementedError` messages.

## Kernel Builders

Continuous kernels:

1. `RBF`
2. `Matern12`
3. `Matern32`
4. `Matern52`
5. `RQ`
6. `Periodic`
7. `Linear`
8. `Polynomial`

Categorical kernels:

1. `GD`
2. `CR`
3. `EHH`
4. `HH`
5. `FE`

Composable builder namespace:

1. `Kernel.rbf()`
2. `Kernel.matern12()`
3. `Kernel.matern32()`
4. `Kernel.matern52()`
5. `Kernel.rq()`
6. `Kernel.periodic()`
7. `Kernel.linear()`
8. `Kernel.polynomial()`
9. `Kernel.gd()` / `Kernel.cr()` / `Kernel.ehh()` / `Kernel.hh()` / `Kernel.fe()`

Supported composition operators:

1. `+`
2. `*`
3. `.with_active_dims(...)`

## Metadata Surface

After a live wrapper call, route telemetry is available through:

1. `backend_train_info`
2. `backend_predict_info`
3. `backend_sample_info`

These may include:

1. `training_route`
2. `materialization_mode`
3. `requested_method`
4. `actual_prediction_route`
5. `actual_variance_route`
6. `actual_sampling_route`
7. `precond_rank`
8. `precond_method`
9. `precond_rebuild_count`
10. `max_tridiag_iter`

## Persistence

All live wrapper families persist two files:

1. `{path}_config.json`
2. `{path}_arrays.npz`

The saved state includes kernel-tree configuration, training arrays, learned parameters, and enough provider-rebuild information for immediate prediction after load.

## Unsupported Surfaces

Important explicit unsupported surfaces:

1. pure categorical models
2. public posterior `cholesky` sampling is not part of the live API
3. `MultiOutputGP` pathwise posterior sampling still rejects polynomial kernels; use `MultiOutputLMCGP` or `method="diagonal"`
4. LMC polynomial pathwise support is publicly evidenced only on the documented degree 1 or 2 feature-map scope
5. very large polynomial/product-feature expansions can exceed the current feature cap and must fall back to `method="diagonal"`

## Source Files

The public API is implemented primarily in:

1. `mojogp/gp.py`
2. `mojogp/multi_output_gp.py`
3. `mojogp/kernel.py`
4. `mojogp/_multi_output_backend.py`
5. `mojogp/feature_support.py`
