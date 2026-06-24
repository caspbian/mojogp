# MojoGP Feature Matrix

This file is generated from `mojogp/feature_support.py`. Do not edit it by hand.

## Status Legend

| Marker | Meaning |
|---|---|
| -- | not started / no implementation yet |
| in-dev | implementation exists but is not public-ready |
| exp | experimental; implemented on a narrow tested scope |
| alpha | built and fully tested on the documented scope |
| beta | scaling/accuracy benchmark validated for broader dev exposure |
| released:<version> | shipped in a numbered release |
| unsupported | intentionally rejected |
| n/a | not meaningful for that surface |

## Surface Definitions

| Surface | Definition |
|---|---|
| SingleOutput: Continuous | `SingleOutputGP` with continuous-only kernel trees |
| SingleOutput: Mixed | `SingleOutputGP` with continuous plus categorical kernel leaves |
| ICM: Continuous | `MultiOutputGP` with continuous-only kernel trees |
| ICM: Mixed | `MultiOutputGP` with continuous plus categorical kernel leaves |
| LMC: Continuous | `MultiOutputLMCGP` where all latent kernels are continuous-only |
| LMC: Mixed | `MultiOutputLMCGP` where at least one latent is mixed continuous-categorical |

## Main Capability Matrix

| Feature Family | Feature | SingleOutput: Continuous | SingleOutput: Mixed | ICM: Continuous | ICM: Mixed | LMC: Continuous | LMC: Mixed |
|---|---|---|---|---|---|---|---|
| Model Boundaries | Pure categorical model / latent | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported |
| Model Structure | Heterogeneous latent kernels | n/a | n/a | n/a | n/a | alpha | exp |
| Dimensionality Controls | Active dimensions | alpha | exp | exp | exp | alpha | exp |
| Parameterization | Isotropic lengthscales | alpha | exp | exp | exp | alpha | exp |
| Parameterization | ARD lengthscales | alpha | exp | exp | in-dev | alpha | in-dev |
| Kernel Algebra | Additive composites | alpha | exp | exp | exp | alpha | in-dev |
| Kernel Algebra | Product composites | alpha | exp | exp | exp | alpha | exp |
| Kernel Algebra | Nested composites | alpha | exp | exp | exp | exp | in-dev |
| Kernel Algebra | Multiple categorical leaves | n/a | exp | n/a | exp | n/a | in-dev |
| Kernel Algebra | Scaled categorical-containing trees | n/a | unsupported | n/a | unsupported | n/a | unsupported |
| Mean / Likelihood | Constant / learned mean offset | alpha | exp | exp | exp | alpha | exp |
| Noise / Likelihood | Learned scalar homoskedastic noise | alpha | exp | n/a | n/a | n/a | n/a |
| Noise / Likelihood | Learned per-task homoskedastic noise `[T]` | n/a | n/a | exp | exp | alpha | exp |
| Noise / Likelihood | Fixed per-sample noise `[n]` | alpha | in-dev | n/a | n/a | n/a | n/a |
| Noise / Likelihood | Fixed per-sample-per-task noise `[n, T]` | n/a | n/a | alpha | in-dev | alpha | in-dev |
| Noise / Likelihood | Learned input-dependent heteroskedasticity | alpha | in-dev | -- | -- | -- | -- |
| Noise / Likelihood | Grouped noise | alpha | in-dev | alpha | in-dev | unsupported | unsupported |
| Lifecycle | Save / load | alpha | exp | exp | exp | alpha | exp |
| Lifecycle | Route metadata | alpha | exp | exp | exp | alpha | exp |

## Execution Route Matrix

| Surface | Materialized Training | Matrix-Free Training | Auto Selection |
|---|---|---|---|
| SingleOutput: Continuous | alpha | alpha | alpha |
| SingleOutput: Mixed | exp | exp | exp |
| ICM: Continuous | exp | exp | n/a |
| ICM: Mixed | exp | exp | n/a |
| LMC: Continuous | alpha | alpha | n/a |
| LMC: Mixed | exp | exp | n/a |

## Prediction / Variance Matrix

| Surface | Mean-Only Prediction | Exact Variance | LOVE Variance | Prediction Cache |
|---|---|---|---|---|
| SingleOutput: Continuous | alpha | alpha | alpha | alpha |
| SingleOutput: Mixed | exp | exp | exp | unsupported |
| ICM: Continuous | exp | exp | exp | n/a |
| ICM: Mixed | exp | exp | exp | n/a |
| LMC: Continuous | alpha | alpha | exp | n/a |
| LMC: Mixed | exp | exp | exp | n/a |

## Posterior Sampling Matrix

| Surface | Diagonal Sampling | Pathwise Sampling | Polynomial Pathwise | Public Cholesky Sampling |
|---|---|---|---|---|
| SingleOutput: Continuous | alpha | alpha | alpha | unsupported |
| SingleOutput: Mixed | exp | exp | in-dev | unsupported |
| ICM: Continuous | exp | exp | unsupported | unsupported |
| ICM: Mixed | exp | exp | unsupported | unsupported |
| LMC: Continuous | alpha | alpha | alpha | unsupported |
| LMC: Mixed | exp | exp | in-dev | unsupported |

## Kernel Primitive Matrix

| Primitive Family | Primitive | SingleOutput | ICM | LMC | Scope Notes |
|---|---|---|---|---|---|
| Continuous base kernel | RBF | alpha | exp | alpha | Broadest support |
| Continuous base kernel | Matern12 | alpha | exp | alpha | Continuous and mixed continuous component |
| Continuous base kernel | Matern32 | alpha | exp | alpha | Continuous and mixed continuous component |
| Continuous base kernel | Matern52 | alpha | exp | alpha | Continuous and mixed continuous component |
| Continuous base kernel | RQ | alpha | exp | alpha | Less evidence than RBF/Matern |
| Continuous base kernel | Periodic | alpha | exp | alpha | Less evidence than RBF/Matern |
| Continuous base kernel | Linear | alpha | exp | alpha | Dot-product kernel |
| Continuous base kernel | Polynomial | alpha | exp | exp | Fixed positive integer degree; exact feature-map pathwise within feature cap |
| Categorical correlation kernel | GD | exp | exp | exp | Mixed surfaces only |
| Categorical correlation kernel | CR | exp | exp | exp | Mixed surfaces only |
| Categorical correlation kernel | EHH | exp | exp | exp | Mixed surfaces only |
| Categorical correlation kernel | HH | exp | exp | exp | Mixed surfaces only |
| Categorical correlation kernel | FE | exp | exp | exp | Mixed surfaces only |

## Boundary / Placeholder Matrix

| Combination | Current Category | Recommended Runtime Behavior |
|---|---|---|
| Pure categorical single-output model | unsupported | raise clear error |
| Pure categorical ICM model | unsupported | raise clear error |
| Pure categorical LMC latent | unsupported | raise clear error |
| Scaled tree containing categorical leaves | unsupported | raise clear error |
| SingleOutput fixed per-sample noise `[n]` | alpha | continuous route accepts `observation_noise` |
| ICM fixed per-sample-per-task noise `[n, T]` | alpha | continuous route accepts `observation_noise` |
| Mixed LMC fixed observation noise `[n, T]` | in-dev | raise `NotImplementedError` until evidenced |
| LMC fixed observation noise plus LOVE variance | in-dev | raise `NotImplementedError` until evidenced |
| SingleOutput learned input-dependent heteroskedasticity | alpha | continuous linear noise-function route is documented and tested |
| Mixed and multi-output learned input-dependent heteroskedasticity | in-dev | raise or warn until separately evidenced |
| Grouped noise | alpha / in-dev / unsupported depending model | run documented continuous routes and raise clear error where explicitly rejected |
| ICM polynomial pathwise sampling | unsupported | raise `NotImplementedError` |
| Public Cholesky posterior sampling | unsupported | reject as non-public API |
| Excessive product/pathwise feature expansion | unsupported-current-scope | raise `NotImplementedError` with mitigation |
