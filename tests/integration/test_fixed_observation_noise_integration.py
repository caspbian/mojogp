"""Integration tests for fixed per-sample observation noise."""

import numpy as np
import pytest

from mojogp import Kernel, RBF, SingleOutputGP as ExactGP


def _heteroskedastic_data(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    X = np.linspace(-2.0, 2.0, n, dtype=np.float32).reshape(-1, 1)
    noise = (0.015 + 0.025 * (X[:, 0] > 0.0)).astype(np.float32)
    f = np.sin(3.0 * X[:, 0]).astype(np.float32)
    y = (f + rng.normal(0.0, np.sqrt(noise)).astype(np.float32)).astype(np.float32)
    return X, y, noise


def _rbf_reference_mean(X_train, y_train, X_test, params, noise, mean):
    lengthscale = float(params[0])
    outputscale = float(params[1])
    centered = y_train.astype(np.float64) - float(mean)

    Xtr = X_train.astype(np.float64)
    Xte = X_test.astype(np.float64)
    train_sq = np.sum(Xtr**2, axis=1, keepdims=True)
    test_sq = np.sum(Xte**2, axis=1, keepdims=True)
    train_d2 = train_sq + train_sq.T - 2.0 * Xtr @ Xtr.T
    cross_d2 = test_sq + train_sq.T - 2.0 * Xte @ Xtr.T
    K = outputscale * np.exp(-0.5 * train_d2 / (lengthscale * lengthscale))
    K[np.diag_indices_from(K)] += noise.astype(np.float64)
    K_cross = outputscale * np.exp(-0.5 * cross_d2 / (lengthscale * lengthscale))
    alpha = np.linalg.solve(K, centered)
    return (K_cross @ alpha + float(mean)).astype(np.float32)


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_fixed_vector_noise_latent_mean_matches_dense_reference(method):
    X, y, noise = _heteroskedastic_data()
    X_test = np.array([[-1.5], [-0.5], [0.25], [1.25]], dtype=np.float32)

    gp = ExactGP(RBF())
    result = gp.fit(
        X,
        y,
        observation_noise=noise,
        learn_noise=False,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=0,
        verbose=False,
    )
    pred = gp.predict_latent(
        X_test,
        variance_method="mean_only",
        method=method,
        max_cg_iterations=160,
        cg_tolerance=1e-5,
        preconditioner_rank=0,
    )
    reference_mean = _rbf_reference_mean(
        X,
        y,
        X_test,
        np.asarray(result.params, dtype=np.float32),
        noise,
        result.mean,
    )

    assert gp.backend_train_info["noise_mode"] == "fixed_vector"
    assert gp.backend_train_info["learn_noise"] is False
    assert gp.backend_train_info["precond_rank"] == 0
    np.testing.assert_allclose(pred.mean, reference_mean, rtol=5e-2, atol=5e-2)
    gp._revoke_provider_info()


@pytest.mark.gpytorch
@pytest.mark.reference
@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_fixed_vector_noise_latent_mean_matches_gpytorch_fixed_noise(method):
    gpytorch = pytest.importorskip("gpytorch")
    torch = pytest.importorskip("torch")

    X, y, noise = _heteroskedastic_data(seed=11)
    X_test = np.array([[-1.5], [-0.5], [0.25], [1.25]], dtype=np.float32)

    gp = ExactGP(RBF())
    result = gp.fit(
        X,
        y,
        observation_noise=noise,
        learn_noise=False,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=0 if method == "materialized" else 8,
        verbose=False,
    )
    mojo_pred = gp.predict_latent(
        X_test,
        variance_method="mean_only",
        method=method,
        max_cg_iterations=160,
        cg_tolerance=1e-5,
        preconditioner_rank=10,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.as_tensor(X, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y, dtype=torch.float32, device=device)
    train_noise = torch.as_tensor(noise, dtype=torch.float32, device=device)
    test_x = torch.as_tensor(X_test, dtype=torch.float32, device=device)

    class FixedNoiseExactGP(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=train_noise,
        learn_additional_noise=False,
    ).to(device)
    model = FixedNoiseExactGP(train_x, train_y, likelihood).to(device)
    model.mean_module.constant = float(result.mean)
    model.covar_module.base_kernel.lengthscale = float(result.params[0])
    model.covar_module.outputscale = float(result.params[1])
    model.eval()
    likelihood.eval()

    with (
        torch.no_grad(),
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.max_cg_iterations(200),
        gpytorch.settings.cg_tolerance(1e-5),
    ):
        gpytorch_mean = model(test_x).mean.detach().cpu().numpy()

    np.testing.assert_allclose(mojo_pred.mean, gpytorch_mean, rtol=5e-2, atol=5e-2)
    gp._revoke_provider_info()


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_fixed_vector_noise_supports_pivoted_cholesky_preconditioner(method):
    X, y, noise = _heteroskedastic_data(seed=13)
    X_test = np.array([[-1.25], [0.0], [1.25]], dtype=np.float32)

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        observation_noise=noise,
        learn_noise=False,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=10,
        verbose=False,
    )
    pred = gp.predict_latent(
        X_test,
        variance_method="mean_only",
        method=method,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        preconditioner_rank=0,
    )

    assert gp.backend_train_info["noise_mode"] == "fixed_vector"
    assert gp.backend_train_info["precond_rank"] == 10
    assert gp.backend_train_info["actual_precond_rank"] == 10
    assert gp.backend_train_info["use_preconditioner"] is True
    assert np.all(np.isfinite(pred.mean))
    gp._revoke_provider_info()


def test_fixed_vector_observed_prediction_adds_explicit_test_noise():
    X, y, noise = _heteroskedastic_data()
    X_test = X[:6]
    test_noise = noise[:6] * 1.5

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        observation_noise=noise,
        learn_noise=False,
        method="matrix_free",
        max_iterations=1,
        num_probes=2,
        max_cg_iterations=20,
        max_tridiag_iterations=5,
        preconditioner_rank=0,
        verbose=False,
    )
    latent = gp.predict_latent(
        X_test,
        variance_method="exact",
        max_cg_iterations=40,
        cg_tolerance=1e-4,
        preconditioner_rank=0,
    )
    observed = gp.predict_observed(
        X_test,
        observation_noise=test_noise,
        variance_method="exact",
        max_cg_iterations=40,
        cg_tolerance=1e-4,
        preconditioner_rank=0,
    )

    np.testing.assert_allclose(observed.mean, latent.mean)
    np.testing.assert_allclose(observed.variance, latent.variance + test_noise, rtol=1e-5)


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_fixed_grouped_noise_matches_expanded_vector_route(method):
    X, y, noise = _heteroskedastic_data(seed=17)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([0.015, 0.04], dtype=np.float32)
    X_test = np.array([[-1.0], [0.5]], dtype=np.float32)

    grouped = ExactGP(RBF())
    grouped_result = grouped.fit(
        X,
        y,
        noise_model="grouped",
        noise_group_train=groups,
        group_noise=group_noise,
        learn_noise=False,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=10,
        verbose=False,
    )
    expanded = ExactGP(RBF())
    expanded_result = expanded.fit(
        X,
        y,
        observation_noise=group_noise[groups],
        learn_noise=False,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=10,
        verbose=False,
    )
    grouped_pred = grouped.predict_latent(
        X_test,
        variance_method="mean_only",
        method=method,
        preconditioner_rank=10,
    )
    expanded_pred = expanded.predict_latent(
        X_test,
        variance_method="mean_only",
        method=method,
        preconditioner_rank=10,
    )

    assert grouped.backend_train_info["noise_mode"] == "fixed_grouped"
    np.testing.assert_allclose(grouped._observation_noise_train, group_noise[groups])
    np.testing.assert_allclose(grouped_result.params, expanded_result.params, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grouped_pred.mean, expanded_pred.mean, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["matrix_free", "materialized"])
def test_input_dependent_noise_function_matches_expanded_vector_route(method):
    X, y, _ = _heteroskedastic_data(seed=31)
    X_test = np.array([[-1.0], [0.5]], dtype=np.float32)

    def noise_fn(X_eval):
        return (0.015 + 0.025 * (X_eval[:, 0] > 0.0)).astype(np.float32)

    function_gp = ExactGP(RBF())
    function_result = function_gp.fit(
        X,
        y,
        noise_model="input_dependent",
        observation_noise_fn=noise_fn,
        learn_noise=False,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=10,
        verbose=False,
    )
    expanded_gp = ExactGP(RBF())
    expanded_result = expanded_gp.fit(
        X,
        y,
        observation_noise=noise_fn(X),
        learn_noise=False,
        method=method,
        max_iterations=1,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=10,
        verbose=False,
    )
    function_pred = function_gp.predict_latent(
        X_test,
        variance_method="mean_only",
        method=method,
        preconditioner_rank=10,
    )
    expanded_pred = expanded_gp.predict_latent(
        X_test,
        variance_method="mean_only",
        method=method,
        preconditioner_rank=10,
    )
    observed = function_gp.predict_observed(
        X_test,
        variance_method="mean_only",
        method=method,
    )

    assert function_gp.backend_train_info["noise_mode"] == "fixed_input_dependent"
    np.testing.assert_allclose(function_gp._observation_noise_train, noise_fn(X))
    np.testing.assert_allclose(
        function_result.params, expanded_result.params, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        function_pred.mean, expanded_pred.mean, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        observed.variance, function_pred.variance + noise_fn(X_test), rtol=1e-5
    )
    function_gp._revoke_provider_info()
    expanded_gp._revoke_provider_info()


def test_fixed_grouped_observed_prediction_uses_test_group_ids():
    X, y, _ = _heteroskedastic_data(seed=19)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    group_noise = np.array([0.015, 0.04], dtype=np.float32)
    X_test = np.array([[-1.0], [0.5]], dtype=np.float32)
    test_groups = np.array([0, 1], dtype=np.int32)

    gp = ExactGP(RBF())
    gp.fit(
        X,
        y,
        noise_model="grouped",
        noise_group_train=groups,
        group_noise=group_noise,
        learn_noise=False,
        method="matrix_free",
        max_iterations=1,
        num_probes=2,
        max_cg_iterations=40,
        max_tridiag_iterations=5,
        preconditioner_rank=0,
        verbose=False,
    )
    latent = gp.predict_latent(
        X_test,
        variance_method="exact",
        max_cg_iterations=40,
        cg_tolerance=1e-4,
        preconditioner_rank=0,
    )
    observed = gp.predict_observed(
        X_test,
        noise_group_test=test_groups,
        variance_method="exact",
        max_cg_iterations=40,
        cg_tolerance=1e-4,
        preconditioner_rank=0,
    )

    np.testing.assert_allclose(observed.mean, latent.mean)
    np.testing.assert_allclose(observed.variance, latent.variance + group_noise[test_groups], rtol=1e-5)


@pytest.mark.parametrize(
    "kernel,initial_params",
    [
        (RBF(ard=True), np.array([0.7, 0.9, 1.0], dtype=np.float32)),
        (Kernel.rbf(active_dims=[0]) + Kernel.matern32(active_dims=[1]), np.array([0.8, 0.9, 0.7, 0.6], dtype=np.float32)),
    ],
)
def test_fixed_vector_noise_continuous_kernel_route_matrix(kernel, initial_params):
    rng = np.random.default_rng(23)
    n = 2000
    x0 = np.linspace(-2.0, 2.0, n, dtype=np.float32)
    x1 = rng.normal(0.0, 1.0, n).astype(np.float32)
    X = np.column_stack([x0, x1]).astype(np.float32)
    noise = (0.012 + 0.018 * (x0 > 0.0)).astype(np.float32)
    y = (np.sin(2.0 * x0) + 0.2 * x1 + rng.normal(0.0, np.sqrt(noise))).astype(np.float32)
    X_test = np.array([[-1.0, -0.5], [0.75, 0.25]], dtype=np.float32)

    gp = ExactGP(kernel)
    result = gp.fit(
        X,
        y,
        observation_noise=noise,
        learn_noise=False,
        method="matrix_free",
        max_iterations=1,
        learning_rate=0.03,
        initial_params=initial_params,
        num_probes=2,
        max_cg_iterations=80,
        cg_tolerance=1e-4,
        max_tridiag_iterations=5,
        preconditioner_rank=0,
        verbose=False,
    )
    pred = gp.predict_latent(
        X_test,
        variance_method="mean_only",
        method="matrix_free",
        preconditioner_rank=0,
    )

    assert gp.backend_train_info["noise_mode"] == "fixed_vector"
    assert gp.backend_train_info["has_observation_noise_vector"] is True
    assert np.all(np.isfinite(result.params))
    assert np.all(np.isfinite(pred.mean))
    gp._revoke_provider_info()


def test_fixed_vector_noise_mixed_route_is_explicitly_in_development():
    n = 2000
    X = np.column_stack(
        [
            np.linspace(-1.0, 1.0, n, dtype=np.float32),
            (np.arange(n) % 3).astype(np.float32),
        ]
    ).astype(np.float32)
    y = np.sin(X[:, 0]).astype(np.float32)
    noise = np.full(n, 0.02, dtype=np.float32)
    gp = ExactGP(Kernel.rbf(active_dims=[0]) * Kernel.ehh(levels=3, active_dims=[1]))

    with pytest.raises(NotImplementedError, match="mixed, multi-output, and LMC noise extensions"):
        gp.fit(
            X,
            y,
            observation_noise=noise,
            learn_noise=False,
            method="matrix_free",
            max_iterations=1,
            num_probes=2,
            max_cg_iterations=20,
            preconditioner_rank=0,
            verbose=False,
        )


def test_learned_vector_noise_trains_and_persists_state(tmp_path):
    X, y, _ = _heteroskedastic_data(seed=23)
    gp = ExactGP(RBF())
    result = gp.fit(
        X,
        y,
        noise_model="learned_vector",
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.02,
        method="matrix_free",
        max_iterations=2,
        learning_rate=0.01,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=4,
        max_cg_iterations=30,
        cg_tolerance=1e-3,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
        verbose=False,
    )
    learned_noise = gp.get_learned_params()["observation_noise_train"]

    assert gp.backend_train_info["noise_mode"] == "learned_vector"
    assert gp.backend_train_info["noise_regularization"] == pytest.approx(0.02)
    assert learned_noise.shape == (len(y),)
    assert np.all(np.isfinite(learned_noise))
    assert float(learned_noise.min()) >= 1e-5
    assert np.isfinite(result.nll)

    X_test = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
    test_noise = np.array([0.02, 0.03, 0.04], dtype=np.float32)
    latent = gp.predict_latent(X_test, variance_method="mean_only")
    observed = gp.predict_observed(
        X_test,
        observation_noise=test_noise,
        variance_method="mean_only",
    )

    path = tmp_path / "learned_vector_noise_gp"
    gp.save(str(path))
    loaded = ExactGP.load(str(path))
    loaded_latent = loaded.predict_latent(
        X_test, variance_method="mean_only"
    )
    loaded_observed = loaded.predict_observed(
        X_test,
        observation_noise=test_noise,
        variance_method="mean_only",
    )
    assert loaded._noise_mode == "learned_vector"
    assert loaded._provider_noise_mode_int == 2
    assert loaded._noise_floor == pytest.approx(1e-5)
    assert loaded._noise_regularization == pytest.approx(0.02)
    np.testing.assert_allclose(loaded._observation_noise_train, learned_noise)
    np.testing.assert_allclose(loaded_latent.mean, latent.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        loaded_latent.variance, latent.variance, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        loaded_observed.mean, observed.mean, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        observed.variance,
        latent.variance + test_noise,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        loaded_observed.variance, observed.variance, rtol=1e-5, atol=1e-6
    )
    with pytest.raises(ValueError, match="requires observation_noise"):
        loaded.predict_observed(X_test, variance_method="mean_only")


def test_learned_grouped_noise_trains_predicts_and_persists_group_state(tmp_path):
    X, y, _ = _heteroskedastic_data(seed=29)
    groups = (X[:, 0] > 0.0).astype(np.int32)
    gp = ExactGP(RBF())
    result = gp.fit(
        X,
        y,
        noise_model="grouped",
        noise_group_train=groups,
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.02,
        method="matrix_free",
        max_iterations=2,
        learning_rate=0.01,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=4,
        max_cg_iterations=30,
        cg_tolerance=1e-3,
        max_tridiag_iterations=8,
        preconditioner_rank=5,
        verbose=False,
    )
    learned_group_noise = gp.get_learned_params()["group_noise"]
    X_test = np.array([[-1.0], [1.0]], dtype=np.float32)
    latent = gp.predict_latent(X_test, variance_method="mean_only")
    observed = gp.predict_observed(
        X_test,
        noise_group_test=np.array([0, 1], dtype=np.int32),
        variance_method="mean_only",
    )

    assert gp.backend_train_info["noise_mode"] == "learned_grouped"
    assert learned_group_noise.shape == (2,)
    assert np.all(np.isfinite(learned_group_noise))
    assert np.isfinite(result.nll)
    np.testing.assert_allclose(
        observed.variance,
        latent.variance + learned_group_noise,
        rtol=1e-5,
        atol=1e-6,
    )
    path = tmp_path / "learned_grouped_noise_gp"
    gp.save(str(path))
    loaded = ExactGP.load(str(path))
    loaded_latent = loaded.predict_latent(
        X_test, variance_method="mean_only"
    )
    loaded_observed = loaded.predict_observed(
        X_test,
        noise_group_test=np.array([0, 1], dtype=np.int32),
        variance_method="mean_only",
    )
    assert loaded._noise_mode == "learned_grouped"
    assert loaded._provider_noise_mode_int == 3
    assert loaded._noise_floor == pytest.approx(1e-5)
    assert loaded._noise_regularization == pytest.approx(0.02)
    np.testing.assert_array_equal(loaded._noise_group_train, groups)
    np.testing.assert_allclose(loaded._noise_group_values, learned_group_noise)
    np.testing.assert_allclose(loaded._observation_noise_train, gp._observation_noise_train)
    np.testing.assert_allclose(loaded_latent.mean, latent.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        loaded_latent.variance, latent.variance, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(loaded_observed.mean, observed.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        loaded_observed.variance, observed.variance, rtol=1e-5, atol=1e-6
    )
def test_learned_input_dependent_noise_matrix_free_route_trains_predicts_and_persists_function_state(
    tmp_path,
):
    _assert_learned_input_dependent_noise_trains_predicts_and_persists_function_state(
        "matrix_free", tmp_path
    )


def test_learned_input_dependent_noise_materialized_route_trains_predicts_and_persists_function_state(
    tmp_path,
):
    _assert_learned_input_dependent_noise_trains_predicts_and_persists_function_state(
        "materialized", tmp_path
    )


def _assert_learned_input_dependent_noise_trains_predicts_and_persists_function_state(
    method, tmp_path
):
    rng = np.random.default_rng(37)
    n = 2000
    X = np.linspace(-2.0, 2.0, n, dtype=np.float32).reshape(-1, 1)
    true_noise = (0.012 + 0.035 / (1.0 + np.exp(-2.5 * X[:, 0]))).astype(
        np.float32
    )
    latent_fn = (np.sin(2.0 * X[:, 0]) + 0.2 * X[:, 0]).astype(np.float32)
    y = (latent_fn + rng.normal(0.0, np.sqrt(true_noise))).astype(np.float32)
    X_test = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)

    gp = ExactGP(RBF())
    result = gp.fit(
        X,
        y,
        noise_model="learned_input_dependent",
        noise_function="linear",
        initial_noise=0.03,
        noise_floor=1e-5,
        noise_regularization=0.01,
        method=method,
        max_iterations=20,
        learning_rate=0.03,
        initial_params=np.array([0.8, 1.1], dtype=np.float32),
        num_probes=4,
        max_cg_iterations=40,
        cg_tolerance=1e-3,
        max_tridiag_iterations=8,
        preconditioner_rank=0 if method == "materialized" else 8,
        verbose=False,
    )
    learned_params = gp.get_learned_params()
    learned_noise = np.asarray(learned_params["observation_noise_train"], dtype=np.float32)
    fn_params = np.asarray(learned_params["noise_function_params"], dtype=np.float32)
    latent = gp.predict_latent(X_test, variance_method="mean_only")
    observed = gp.predict_observed(X_test, variance_method="mean_only")
    inferred_test_noise = observed.variance - latent.variance

    assert gp.backend_train_info["noise_mode"] == "learned_input_dependent"
    assert gp.backend_train_info["learned_noise_function"] == "linear"
    assert learned_params["noise_function"] == "linear"
    assert fn_params.shape == (2,)
    assert np.all(np.isfinite(fn_params))
    assert fn_params[1] > 0.0, (
        f"expected learned linear noise slope to be positive for {method}; "
        f"fn_params={fn_params}, iterations={result.iterations}, "
        f"nll_history={result.nll_history}"
    )
    assert learned_noise.shape == (2000,)
    assert np.all(np.isfinite(learned_noise))
    assert float(learned_noise.min()) >= 1e-5
    assert learned_noise[-1] - learned_noise[0] > 1e-4
    assert np.isfinite(result.nll)
    np.testing.assert_allclose(observed.mean, latent.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        observed.variance,
        latent.variance + inferred_test_noise,
        rtol=1e-5,
        atol=1e-6,
    )
    assert inferred_test_noise[2] - inferred_test_noise[0] > 1e-4

    path = tmp_path / f"learned_input_dependent_noise_gp_{method}"
    gp.save(str(path))
    loaded = ExactGP.load(str(path))
    loaded_params = loaded.get_learned_params()
    loaded_latent = loaded.predict_latent(
        X_test, variance_method="mean_only"
    )
    loaded_observed = loaded.predict_observed(
        X_test, variance_method="mean_only"
    )
    assert loaded._noise_mode == "learned_input_dependent"
    assert loaded._provider_noise_mode_int == 4
    assert loaded._noise_floor == pytest.approx(1e-5)
    assert loaded._noise_regularization == pytest.approx(0.01)
    assert loaded_params["noise_function"] == "linear"
    np.testing.assert_allclose(loaded_params["noise_function_params"], fn_params)
    np.testing.assert_allclose(loaded_params["observation_noise_train"], learned_noise)
    np.testing.assert_allclose(loaded_latent.mean, latent.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        loaded_latent.variance, latent.variance, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(loaded_observed.mean, observed.mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        loaded_observed.variance, observed.variance, rtol=1e-5, atol=1e-6
    )
    gp._revoke_provider_info()
    loaded._revoke_provider_info()
