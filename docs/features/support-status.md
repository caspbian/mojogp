# Support Status

MojoGP uses evidence-based feature maturity labels. The generated [Feature Matrix](../FEATURE_MATRIX.md) is the canonical status table.

## Status Labels

| Marker | Meaning |
|---|---|
| `--` | Not started or no implementation yet. |
| `in-dev` | Code may exist, but the route is not public functionality. It may raise `NotImplementedError` or emit an in-development warning. |
| `exp` | Public experimental support on a narrow tested scope. The API or behavior may change. |
| `alpha` | Built and tested on the documented scope. Broader scaling or accuracy certification may still be ongoing. |
| `beta` | Scaling and accuracy benchmark validated for broader development exposure. |
| `released:<version>` | Shipped in a numbered release. |
| `unsupported` | Intentionally rejected. |
| `n/a` | Not meaningful for that model surface. |

## Public Evidence Rule

Do not treat implementation presence as a support claim. A public feature claim needs theory, unit tests, integration tests, realistic workflow/system checks, route metadata checks, documentation, and appropriate benchmark or reference evidence for the exact documented scope.

## Exact-GP Boundary

MojoGP is an exact GP library. Accepted accelerators include CG, SLQ, pivoted-Cholesky preconditioning, and LOVE-style variance prediction after exact training. Inducing points, variational inference, sparse GP approximations, SVI, and surrogate objectives are outside the project scope.

## Runtime Behavior

Runtime checks use `mojogp.feature_support` where possible. Unsupported or not-started surfaces should fail clearly. In-development surfaces either fail with `NotImplementedError` when correctness-sensitive or emit an `InDevelopmentFeatureWarning` when the route is intentionally reachable for internal validation.

## Current Public Anchors

The strongest public surface is continuous `SingleOutputGP`. Multi-output, LMC, mixed continuous-categorical, posterior sampling, LOVE variance, and heteroskedastic variants are documented only on their evidenced scopes.

Use these references together:

1. [Feature Matrix](../FEATURE_MATRIX.md)
2. [API Surface](../API.md)
3. [API Reference](../reference/index.md)
