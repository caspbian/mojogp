"""API tests for fixed per-sample observation noise."""

import numpy as np
import pytest

from mojogp import PredictionResult, RBF, SingleOutputGP as ExactGP
from mojogp.gp import TrainingResult


def _tiny_data(n=5):
    X = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)
    y = np.sin(X[:, 0]).astype(np.float32)
    return X, y


def test_fixed_observation_noise_validation_requires_learn_noise_false():
    X, y = _tiny_data()
    gp = ExactGP(RBF())
    with pytest.raises(ValueError, match="learn_noise must be False"):
        gp.fit(X, y, observation_noise=np.full(len(y), 0.1, dtype=np.float32))


def test_fixed_observation_noise_validation_shape_and_floor(monkeypatch):
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    with pytest.raises(ValueError, match="shape"):
        gp.fit(X, y, observation_noise=np.full((len(y), 1), 0.1), learn_noise=False)

    with pytest.raises(ValueError, match=">= noise_floor"):
        gp.fit(X, y, observation_noise=np.zeros(len(y), dtype=np.float32), learn_noise=False)

    def fake_fit_continuous(*args, **kwargs):
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.array([1.0, 1.0], dtype=np.float32),
            noise=0.0,
            mean=0.0,
            nll=0.0,
            iterations=1,
            converged=True,
            lanczos_root=None,
            lanczos_rank=0,
        )
        return gp._training_result

    monkeypatch.setattr(gp, "_ensure_compiled", lambda: None)
    monkeypatch.setattr(gp, "_fit_continuous", fake_fit_continuous)
    noise = np.linspace(0.01, 0.05, len(y), dtype=np.float32)
    gp.fit(X, y, observation_noise=noise, learn_noise=False, method="matrix_free")
    assert gp._noise_mode == "fixed_vector"
    np.testing.assert_allclose(gp._observation_noise_train, noise)


def test_grouped_fixed_noise_expands_to_observation_noise(monkeypatch):
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    def fake_fit_continuous(*args, **kwargs):
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.array([1.0, 1.0], dtype=np.float32),
            noise=0.0,
            mean=0.0,
            nll=0.0,
            iterations=1,
            converged=True,
            lanczos_root=None,
            lanczos_rank=0,
        )
        return gp._training_result

    monkeypatch.setattr(gp, "_ensure_compiled", lambda: None)
    monkeypatch.setattr(gp, "_fit_continuous", fake_fit_continuous)
    groups = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    group_noise = np.array([0.01, 0.03, 0.05], dtype=np.float32)
    gp.fit(
        X,
        y,
        noise_model="grouped",
        noise_group_train=groups,
        group_noise=group_noise,
        learn_noise=False,
    )
    assert gp._noise_mode == "fixed_grouped"
    np.testing.assert_array_equal(gp._noise_group_train, groups)
    np.testing.assert_allclose(gp._noise_group_values, group_noise)
    np.testing.assert_allclose(gp._observation_noise_train, group_noise[groups])


def test_input_dependent_noise_function_expands_to_observation_noise(monkeypatch):
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    def fake_fit_continuous(*args, **kwargs):
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.array([1.0, 1.0], dtype=np.float32),
            noise=0.0,
            mean=0.0,
            nll=0.0,
            iterations=1,
            converged=True,
            lanczos_root=None,
            lanczos_rank=0,
        )
        return gp._training_result

    monkeypatch.setattr(gp, "_ensure_compiled", lambda: None)
    monkeypatch.setattr(gp, "_fit_continuous", fake_fit_continuous)

    def noise_fn(X_eval):
        return (0.01 + 0.02 * X_eval[:, 0]).astype(np.float32)

    gp.fit(
        X,
        y,
        noise_model="input_dependent",
        observation_noise_fn=noise_fn,
        learn_noise=False,
    )
    assert gp._noise_mode == "fixed_input_dependent"
    assert gp._provider_noise_mode_int == 1
    assert gp._observation_noise_fn is noise_fn
    np.testing.assert_allclose(gp._observation_noise_train, noise_fn(X))


def test_input_dependent_noise_function_validation():
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    with pytest.raises(ValueError, match="requires observation_noise_fn"):
        gp.fit(X, y, noise_model="input_dependent", learn_noise=False)
    with pytest.raises(ValueError, match="learn_noise must be False"):
        gp.fit(
            X,
            y,
            observation_noise_fn=lambda X_eval: np.full(
                len(X_eval), 0.01, dtype=np.float32
            ),
        )
    with pytest.raises(ValueError, match="Pass either observation_noise or observation_noise_fn"):
        gp.fit(
            X,
            y,
            observation_noise=np.full(len(y), 0.01, dtype=np.float32),
            observation_noise_fn=lambda X_eval: np.full(
                len(X_eval), 0.01, dtype=np.float32
            ),
            learn_noise=False,
        )
    with pytest.raises(ValueError, match="must return shape"):
        gp.fit(
            X,
            y,
            observation_noise_fn=lambda X_eval: np.full(
                (len(X_eval), 1), 0.01, dtype=np.float32
            ),
            learn_noise=False,
        )


def test_grouped_noise_validation_and_learned_noise_boundary():
    X, y = _tiny_data()
    gp = ExactGP(RBF())
    with pytest.raises(ValueError, match="requires noise_group_train"):
        gp.fit(X, y, noise_model="grouped", learn_noise=False, group_noise=np.array([0.1], dtype=np.float32))
    with pytest.raises(ValueError, match="outside group_noise"):
        gp.fit(
            X,
            y,
            noise_model="grouped",
            noise_group_train=np.arange(len(y), dtype=np.int32),
            group_noise=np.array([0.1], dtype=np.float32),
            learn_noise=False,
        )


def test_learned_grouped_noise_initializes_internal_state(monkeypatch):
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    def fake_fit_continuous(*args, **kwargs):
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.array([1.0, 1.0], dtype=np.float32),
            noise=0.0,
            mean=0.0,
            nll=0.0,
            iterations=1,
            converged=True,
            lanczos_root=None,
            lanczos_rank=0,
        )
        return gp._training_result

    monkeypatch.setattr(gp, "_ensure_compiled", lambda: None)
    monkeypatch.setattr(gp, "_fit_continuous", fake_fit_continuous)
    groups = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    gp.fit(X, y, noise_model="grouped", noise_group_train=groups, initial_noise=0.04)
    assert gp._noise_mode == "learned_grouped"
    assert gp._provider_noise_mode_int == 3
    np.testing.assert_array_equal(gp._noise_group_train, groups)
    np.testing.assert_allclose(gp._noise_group_values, np.full(3, 0.04, dtype=np.float32))
    np.testing.assert_allclose(gp._observation_noise_train, np.full(len(y), 0.04, dtype=np.float32))


def test_learned_vector_noise_initializes_internal_state(monkeypatch):
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    def fake_fit_continuous(*args, **kwargs):
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.array([1.0, 1.0], dtype=np.float32),
            noise=0.0,
            mean=0.0,
            nll=0.0,
            iterations=1,
            converged=True,
            lanczos_root=None,
            lanczos_rank=0,
        )
        return gp._training_result

    monkeypatch.setattr(gp, "_ensure_compiled", lambda: None)
    monkeypatch.setattr(gp, "_fit_continuous", fake_fit_continuous)
    gp.fit(
        X,
        y,
        noise_model="learned_vector",
        initial_noise=0.05,
        noise_floor=1e-5,
        noise_regularization=0.02,
    )
    assert gp._noise_mode == "learned_vector"
    assert gp._provider_noise_mode_int == 2
    assert gp._noise_regularization == pytest.approx(0.02)
    np.testing.assert_allclose(gp._observation_noise_train, np.full(len(y), 0.05, dtype=np.float32))


def test_learned_input_dependent_noise_initializes_internal_state(monkeypatch):
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    def fake_fit_continuous(*args, **kwargs):
        gp._is_trained = True
        gp._training_result = TrainingResult(
            params=np.array([1.0, 1.0], dtype=np.float32),
            noise=0.0,
            mean=0.0,
            nll=0.0,
            iterations=1,
            converged=True,
            lanczos_root=None,
            lanczos_rank=0,
        )
        return gp._training_result

    monkeypatch.setattr(gp, "_ensure_compiled", lambda: None)
    monkeypatch.setattr(gp, "_fit_continuous", fake_fit_continuous)
    gp.fit(
        X,
        y,
        noise_model="learned_input_dependent",
        noise_function="linear",
        initial_noise=0.04,
        noise_floor=1e-5,
        noise_regularization=0.02,
    )
    assert gp._noise_mode == "learned_input_dependent"
    assert gp._noise_function == "linear"
    assert gp._provider_noise_mode_int == 4
    assert gp._noise_regularization == pytest.approx(0.02)
    np.testing.assert_allclose(gp._observation_noise_train, np.full(len(y), 0.04, dtype=np.float32))


def test_learned_noise_params_are_returned_as_copies():
    X, y = _tiny_data()
    gp = ExactGP(RBF())
    gp._X_train = X
    gp._y_train = y
    gp.dim = 1
    gp._noise_mode = "learned_vector"
    gp._observation_noise_train = np.full(len(y), 0.03, dtype=np.float32)
    gp._is_trained = True
    gp._training_result = TrainingResult(
        params=np.array([1.0, 1.0], dtype=np.float32),
        noise=0.0,
        mean=0.0,
        nll=0.0,
        iterations=1,
        converged=True,
        lanczos_root=None,
        lanczos_rank=0,
    )

    params = gp.get_learned_params()
    params["observation_noise_train"][0] = 99.0
    assert gp._observation_noise_train[0] == pytest.approx(0.03)

    gp._noise_mode = "learned_grouped"
    gp._noise_group_values = np.array([0.01, 0.05], dtype=np.float32)
    params = gp.get_learned_params()
    params["group_noise"][0] = 99.0
    assert gp._noise_group_values[0] == pytest.approx(0.01)

    gp._noise_mode = "learned_input_dependent"
    gp._noise_function = "linear"
    gp._noise_function_params = np.array([-3.5, 0.2], dtype=np.float32)
    params = gp.get_learned_params()
    params["noise_function_params"][0] = 99.0
    params["observation_noise_train"][0] = 77.0
    assert gp._noise_function_params[0] == pytest.approx(-3.5)
    assert gp._observation_noise_train[0] == pytest.approx(0.03)


def test_learned_vector_noise_validation_rejects_invalid_options():
    X, y = _tiny_data()
    gp = ExactGP(RBF())
    with pytest.raises(ValueError, match="do not also pass observation_noise"):
        gp.fit(
            X,
            y,
            noise_model="learned_vector",
            observation_noise=np.full(len(y), 0.05, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="learn_noise must be True"):
        gp.fit(X, y, noise_model="learned_vector", learn_noise=False)
    with pytest.raises(ValueError, match="initial_noise must be greater than noise_floor"):
        gp.fit(X, y, noise_model="learned_vector", initial_noise=1e-6, noise_floor=1e-5)
    with pytest.raises(ValueError, match="noise_regularization"):
        gp.fit(X, y, noise_model="learned_vector", noise_regularization=-1.0)


def test_learned_grouped_noise_validation_rejects_invalid_options():
    X, y = _tiny_data()
    gp = ExactGP(RBF())
    groups = np.array([0, 1, -1, 1, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="non-negative"):
        gp.fit(X, y, noise_model="grouped", noise_group_train=groups)

    with pytest.raises(ValueError, match="initial_noise must be greater than noise_floor"):
        gp.fit(
            X,
            y,
            noise_model="grouped",
            noise_group_train=np.array([0, 1, 0, 1, 0], dtype=np.int32),
            initial_noise=1e-6,
            noise_floor=1e-5,
        )


def test_learned_input_dependent_noise_validation_rejects_invalid_options():
    X, y = _tiny_data()
    gp = ExactGP(RBF())

    with pytest.raises(ValueError, match="requires noise_function='linear'"):
        gp.fit(X, y, noise_model="learned_input_dependent")
    with pytest.raises(ValueError, match="noise_function must be 'linear'"):
        gp.fit(X, y, noise_model="learned_input_dependent", noise_function="quadratic")
    with pytest.raises(ValueError, match="only supported"):
        gp.fit(X, y, noise_model="scalar", noise_function="linear")
    with pytest.raises(ValueError, match="do not also pass observation_noise"):
        gp.fit(
            X,
            y,
            noise_model="learned_input_dependent",
            noise_function="linear",
            observation_noise=np.full(len(y), 0.05, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="do not also pass observation_noise_fn"):
        gp.fit(
            X,
            y,
            noise_model="learned_input_dependent",
            noise_function="linear",
            observation_noise_fn=lambda X_eval: np.full(len(X_eval), 0.05, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="does not use grouped noise inputs"):
        gp.fit(
            X,
            y,
            noise_model="learned_input_dependent",
            noise_function="linear",
            noise_group_train=np.zeros(len(y), dtype=np.int32),
        )
    with pytest.raises(ValueError, match="learn_noise must be True"):
        gp.fit(
            X,
            y,
            noise_model="learned_input_dependent",
            noise_function="linear",
            learn_noise=False,
        )
    with pytest.raises(ValueError, match="initial_noise must be greater than noise_floor"):
        gp.fit(
            X,
            y,
            noise_model="learned_input_dependent",
            noise_function="linear",
            initial_noise=1e-6,
            noise_floor=1e-5,
        )


def test_predict_observed_requires_explicit_test_noise():
    gp = ExactGP(RBF())
    latent = PredictionResult(
        mean=np.array([1.0, 2.0], dtype=np.float32),
        variance=np.array([0.25, 0.5], dtype=np.float32),
        std=np.sqrt(np.array([0.25, 0.5], dtype=np.float32)),
    )

    with pytest.raises(ValueError, match="requires observation_noise"):
        gp._add_observation_noise_to_prediction(
            latent, None, expected_n=2, return_full=True
        )

    observed = gp._add_observation_noise_to_prediction(
        latent,
        np.array([0.1, 0.2], dtype=np.float32),
        expected_n=2,
        return_full=True,
    )
    np.testing.assert_allclose(observed.mean, latent.mean)
    np.testing.assert_allclose(observed.variance, np.array([0.35, 0.7], dtype=np.float32))
    np.testing.assert_allclose(observed.std, np.sqrt(observed.variance))


def test_predict_observed_accepts_group_ids():
    gp = ExactGP(RBF())
    gp._noise_group_values = np.array([0.1, 0.2], dtype=np.float32)
    latent = PredictionResult(
        mean=np.array([1.0, 2.0], dtype=np.float32),
        variance=np.array([0.25, 0.5], dtype=np.float32),
        std=np.sqrt(np.array([0.25, 0.5], dtype=np.float32)),
    )

    obs_noise = gp._observation_noise_from_test_groups(
        np.array([1, 0], dtype=np.int32),
        expected_n=2,
    )
    np.testing.assert_allclose(obs_noise, np.array([0.2, 0.1], dtype=np.float32))
    observed = gp._add_observation_noise_to_prediction(
        latent,
        obs_noise,
        expected_n=2,
        return_full=True,
    )
    np.testing.assert_allclose(observed.variance, np.array([0.45, 0.6], dtype=np.float32))


def test_predict_observed_accepts_noise_function():
    gp = ExactGP(RBF())
    latent = PredictionResult(
        mean=np.array([1.0, 2.0], dtype=np.float32),
        variance=np.array([0.25, 0.5], dtype=np.float32),
        std=np.sqrt(np.array([0.25, 0.5], dtype=np.float32)),
    )
    X_test = np.array([[0.0], [1.0]], dtype=np.float32)
    gp._observation_noise_fn = lambda X_eval: (0.1 + 0.1 * X_eval[:, 0]).astype(
        np.float32
    )

    observed = gp._add_observation_noise_to_prediction(
        latent,
        gp._evaluate_observation_noise_fn(
            gp._observation_noise_fn, X_test, expected_n=2, name="test"
        ),
        expected_n=2,
        return_full=True,
    )
    np.testing.assert_allclose(observed.variance, np.array([0.35, 0.7], dtype=np.float32))


def test_predict_observed_infers_learned_linear_input_dependent_noise():
    gp = ExactGP(RBF())
    gp._noise_mode = "learned_input_dependent"
    gp._noise_function = "linear"
    gp._noise_function_params = np.array([-3.0, 0.5], dtype=np.float32)
    gp._noise_floor = 1e-5
    latent = PredictionResult(
        mean=np.array([1.0, 2.0], dtype=np.float32),
        variance=np.array([0.25, 0.5], dtype=np.float32),
        std=np.sqrt(np.array([0.25, 0.5], dtype=np.float32)),
    )
    X_test = np.array([[-1.0], [1.0]], dtype=np.float32)
    noise = gp._evaluate_learned_noise_function(X_test, expected_n=2)

    observed = gp._add_observation_noise_to_prediction(
        latent,
        noise,
        expected_n=2,
        return_full=True,
    )
    assert noise[1] > noise[0]
    np.testing.assert_allclose(observed.mean, latent.mean)
    np.testing.assert_allclose(observed.variance, latent.variance + noise, rtol=1e-6)


def test_predict_observed_rejects_unseen_group_ids():
    gp = ExactGP(RBF())
    gp._noise_group_values = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="unknown group id"):
        gp._observation_noise_from_test_groups(
            np.array([0, 2], dtype=np.int32),
            expected_n=2,
        )


def test_fixed_observation_noise_save_load_roundtrip(tmp_path, monkeypatch):
    X, y = _tiny_data()
    noise = np.linspace(0.01, 0.05, len(y), dtype=np.float32)
    gp = ExactGP(RBF())
    gp._X_train = X
    gp._y_train = y
    gp.dim = 1
    gp._noise_mode = "fixed_vector"
    gp._noise_floor = 1e-6
    gp._observation_noise_train = noise
    gp._is_trained = True
    gp._training_method = "matrix_free"
    gp._training_result = TrainingResult(
        params=np.array([1.0, 1.0], dtype=np.float32),
        noise=0.0,
        mean=0.0,
        nll=0.0,
        iterations=1,
        converged=True,
        lanczos_root=None,
        lanczos_rank=0,
    )

    path = tmp_path / "fixed_noise_gp"
    gp.save(str(path))
    monkeypatch.setattr(ExactGP, "_ensure_compiled", lambda self: None)
    loaded = ExactGP.load(str(path))
    assert loaded._noise_mode == "fixed_vector"
    np.testing.assert_allclose(loaded._observation_noise_train, noise)


def test_grouped_observation_noise_save_load_roundtrip(tmp_path, monkeypatch):
    X, y = _tiny_data()
    groups = np.array([0, 1, 0, 1, 2], dtype=np.int32)
    group_noise = np.array([0.01, 0.03, 0.05], dtype=np.float32)
    gp = ExactGP(RBF())
    gp._X_train = X
    gp._y_train = y
    gp.dim = 1
    gp._noise_mode = "fixed_grouped"
    gp._noise_floor = 1e-6
    gp._noise_group_train = groups
    gp._noise_group_values = group_noise
    gp._observation_noise_train = group_noise[groups]
    gp._is_trained = True
    gp._training_method = "matrix_free"
    gp._training_result = TrainingResult(
        params=np.array([1.0, 1.0], dtype=np.float32),
        noise=0.0,
        mean=0.0,
        nll=0.0,
        iterations=1,
        converged=True,
        lanczos_root=None,
        lanczos_rank=0,
    )

    path = tmp_path / "grouped_noise_gp"
    gp.save(str(path))
    monkeypatch.setattr(ExactGP, "_ensure_compiled", lambda self: None)
    loaded = ExactGP.load(str(path))
    assert loaded._noise_mode == "fixed_grouped"
    np.testing.assert_array_equal(loaded._noise_group_train, groups)
    np.testing.assert_allclose(loaded._noise_group_values, group_noise)
    np.testing.assert_allclose(loaded._observation_noise_train, group_noise[groups])


def test_input_dependent_observation_noise_save_load_keeps_train_diagonal_only(
    tmp_path, monkeypatch
):
    X, y = _tiny_data()
    noise = (0.01 + 0.02 * X[:, 0]).astype(np.float32)
    gp = ExactGP(RBF())
    gp._X_train = X
    gp._y_train = y
    gp.dim = 1
    gp._noise_mode = "fixed_input_dependent"
    gp._noise_floor = 1e-6
    gp._observation_noise_train = noise
    gp._observation_noise_fn = lambda X_eval: (0.01 + 0.02 * X_eval[:, 0]).astype(
        np.float32
    )
    gp._is_trained = True
    gp._training_method = "matrix_free"
    gp._training_result = TrainingResult(
        params=np.array([1.0, 1.0], dtype=np.float32),
        noise=0.0,
        mean=0.0,
        nll=0.0,
        iterations=1,
        converged=True,
        lanczos_root=None,
        lanczos_rank=0,
    )

    path = tmp_path / "input_dependent_noise_gp"
    gp.save(str(path))
    monkeypatch.setattr(ExactGP, "_ensure_compiled", lambda self: None)
    loaded = ExactGP.load(str(path))
    assert loaded._noise_mode == "fixed_input_dependent"
    assert loaded._observation_noise_fn is None
    np.testing.assert_allclose(loaded._observation_noise_train, noise)


def test_learned_input_dependent_observation_noise_save_load_roundtrip(
    tmp_path, monkeypatch
):
    X, y = _tiny_data()
    gp = ExactGP(RBF())
    gp._X_train = X
    gp._y_train = y
    gp.dim = 1
    gp._noise_mode = "learned_input_dependent"
    gp._noise_floor = 1e-5
    gp._noise_regularization = 0.02
    gp._provider_noise_mode_int = 4
    gp._noise_function = "linear"
    gp._noise_function_params = np.array([-3.0, 0.5], dtype=np.float32)
    gp._observation_noise_train = gp._evaluate_learned_noise_function(
        X,
        expected_n=len(y),
    )
    gp._is_trained = True
    gp._training_method = "matrix_free"
    gp._training_result = TrainingResult(
        params=np.array([1.0, 1.0], dtype=np.float32),
        noise=0.0,
        mean=0.0,
        nll=0.0,
        iterations=1,
        converged=True,
        lanczos_root=None,
        lanczos_rank=0,
    )

    path = tmp_path / "learned_input_dependent_noise_gp"
    gp.save(str(path))
    monkeypatch.setattr(ExactGP, "_ensure_compiled", lambda self: None)
    loaded = ExactGP.load(str(path))
    assert loaded._noise_mode == "learned_input_dependent"
    assert loaded._provider_noise_mode_int == 4
    assert loaded._noise_function == "linear"
    assert loaded._noise_regularization == pytest.approx(0.02)
    np.testing.assert_allclose(loaded._noise_function_params, gp._noise_function_params)
    np.testing.assert_allclose(loaded._observation_noise_train, gp._observation_noise_train)
