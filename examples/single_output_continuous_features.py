"""Single-output continuous LOVE, pathwise, and polynomial workflow."""

from __future__ import annotations

import numpy as np

from mojogp import SingleOutputGP, Kernel


def run_example(n_train: int = 2000, n_test: int = 32, method: str = "materialized"):
    rng = np.random.default_rng(2026)
    X = rng.normal(scale=0.5, size=(n_train, 2)).astype(np.float32)
    signal = (0.4 * X[:, 0] - 0.2 * X[:, 1] + 0.5) ** 2
    y = (signal + 0.02 * rng.normal(size=n_train)).astype(np.float32)
    X_test = rng.normal(scale=0.5, size=(n_test, 2)).astype(np.float32)

    gp = SingleOutputGP(
        Kernel.polynomial(degree=2.0, offset=2.0, outputscale=1.0),
        verbose=False,
    )
    result = gp.fit(
        X,
        y,
        method=method,
        max_iterations=6,
        learning_rate=0.02,
        num_probes=3,
        max_cg_iterations=25,
        preconditioner_rank=8,
        verbose=False,
    )

    exact = gp.predict(X_test, variance_method="exact", max_cg_iterations=25)
    exact_info = dict(gp.backend_predict_info)
    love = gp.predict(X_test, variance_method="love", max_cg_iterations=25)
    love_info = dict(gp.backend_predict_info)
    samples = gp.sample_posterior(
        X_test[:8],
        n_samples=16,
        method="pathwise",
        rng=np.random.default_rng(7),
    )
    sample_info = dict(gp.backend_sample_info)

    assert np.all(np.isfinite(exact.mean))
    assert np.all(np.isfinite(love.variance))
    assert np.all(np.isfinite(samples))
    assert love_info["variance_method"] == "love"
    assert love_info["actual_variance_route"] == "predict"
    assert sample_info["actual_sampling_route"] == "provider_pathwise"

    print(f"Training route:        {method}")
    print(f"Final NLL:             {float(result.nll):.6f}")
    print(f"Exact variance mean:   {float(np.mean(exact.variance)):.6f}")
    print(f"LOVE variance mean:    {float(np.mean(love.variance)):.6f}")
    print(f"Pathwise sample shape: {samples.shape}")
    print(f"Exact route metadata:  {exact_info}")
    print(f"LOVE route metadata:   {love_info}")
    print(f"Sample route metadata: {sample_info}")

    return {
        "training_result": result,
        "exact": exact,
        "love": love,
        "samples": samples,
        "love_info": love_info,
        "sample_info": sample_info,
    }


if __name__ == "__main__":
    run_example()
