"""Integration tests for multi-output constant-mean training and prediction."""

import numpy as np


def _generate_multi_output_data(
    n=500, d=3, T=3, true_means=None, noise_std=0.1, seed=42
):
    if true_means is None:
        true_means = [5.0, -3.0, 10.0]
    true_means = np.array(true_means, dtype=np.float32)
    T = len(true_means)

    np.random.seed(seed)
    X = np.random.randn(n, d).astype(np.float32)
    Y = np.zeros((n, T), dtype=np.float32)
    f = 0.5 * np.sin(X[:, 0])
    for t in range(T):
        Y[:, t] = true_means[t] + f + noise_std * np.random.randn(n)
    return X, Y.astype(np.float32), true_means


def _generate_zero_mean_multi_output_data(n=500, d=3, T=3, noise_std=0.1, seed=42):
    return _generate_multi_output_data(
        n=n, d=d, T=T, true_means=[0.0] * T, noise_std=noise_std, seed=seed
    )


class TestMultiOutputConstantMeanTraining:
    def test_learns_nonzero_per_task_means(self):
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_multi_output_data(
            n=2000, T=3, true_means=true_means, seed=42
        )
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        assert len(gp._fitted_mean) == 3
        for t in range(3):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.0

    def test_learns_zero_means(self):
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y, _ = _generate_zero_mean_multi_output_data(n=2000, seed=42)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(3):
            assert abs(gp._fitted_mean[t]) < 1.5

    def test_user_init_mean_float_used(self):
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y, _ = _generate_multi_output_data(
            n=2000, true_means=[5.0, 5.0, 5.0], seed=42
        )
        gp = MultiOutputGP(kernel="rbf", init_mean=5.0)
        gp.fit(X, Y, max_iterations=50, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(3):
            assert abs(gp._fitted_mean[t] - 5.0) < 2.0

    def test_user_init_mean_array_used(self):
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [2.0, -1.0, 8.0]
        X, Y, _ = _generate_multi_output_data(n=2000, true_means=true_means, seed=42)
        gp = MultiOutputGP(
            kernel="rbf", init_mean=np.array(true_means, dtype=np.float32)
        )
        gp.fit(X, Y, max_iterations=50, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(3):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.0

    def test_large_per_task_means(self):
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [50.0, -50.0, 100.0]
        X, Y, _ = _generate_multi_output_data(n=2000, true_means=true_means, seed=42)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=100, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(3):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 5.0


class TestMultiOutputConstantMeanPrediction:
    def test_prediction_has_correct_shape(self):
        from mojogp.multi_output_gp import MultiOutputGP

        X, Y, _ = _generate_multi_output_data(n=500, d=3, T=3, seed=42)
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=50, learning_rate=0.05, verbose=False)
        X_test = np.random.randn(20, 3).astype(np.float32)
        mean, var = gp.predict(X_test, return_var=True)
        assert mean.shape == (20, 3)
        assert var.shape == (20, 3)

    def test_prediction_mean_near_true_means(self):
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_multi_output_data(
            n=500, d=3, T=3, true_means=true_means, seed=42
        )
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)
        mean, _ = gp.predict(X, return_var=True)
        for t in range(3):
            rmse = np.sqrt(np.mean((mean[:, t] - Y[:, t]) ** 2))
            assert rmse < 1.0

    def test_prediction_average_near_true_mean(self):
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0, 10.0]
        X, Y, _ = _generate_multi_output_data(
            n=500, d=3, T=3, true_means=true_means, seed=42
        )
        gp = MultiOutputGP(kernel="rbf")
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)
        np.random.seed(99)
        X_test = np.random.randn(100, 3).astype(np.float32)
        mean, _ = gp.predict(X_test, return_var=True)
        for t in range(3):
            avg = np.mean(mean[:, t])
            assert abs(avg - true_means[t]) < 3.0


class TestMultiOutputConstantMeanARD:
    def test_ard_learns_nonzero_means(self):
        from mojogp.multi_output_gp import MultiOutputGP

        true_means = [5.0, -3.0]
        X, Y, _ = _generate_multi_output_data(
            n=500, d=3, T=2, true_means=true_means, seed=42
        )
        gp = MultiOutputGP(kernel="rbf", ard=True)
        gp.fit(X, Y, max_iterations=80, learning_rate=0.05, verbose=False)

        assert gp._fitted_mean is not None
        for t in range(2):
            assert abs(gp._fitted_mean[t] - true_means[t]) < 2.0
