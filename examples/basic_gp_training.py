"""Basic SingleOutputGP training example using the live public API."""

import numpy as np

from mojogp import SingleOutputGP, RBF


def make_data(n_train: int = 5000, n_test: int = 256, seed: int = 42):
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, 2).astype(np.float32)
    y_train = (
        np.sin(X_train[:, 0]) + 0.2 * X_train[:, 1] + 0.05 * rng.randn(n_train)
    ).astype(np.float32)
    X_test = rng.randn(n_test, 2).astype(np.float32)
    y_test = (np.sin(X_test[:, 0]) + 0.2 * X_test[:, 1]).astype(np.float32)
    return X_train, y_train, X_test, y_test


def run_example(method: str = "matrix_free", n_train: int = 5000, n_test: int = 256):
    X_train, y_train, X_test, y_test = make_data(n_train=n_train, n_test=n_test)

    gp = SingleOutputGP(RBF(ard=True), verbose=False)
    gp.fit(
        X_train,
        y_train,
        max_iterations=20,
        learning_rate=0.03,
        method=method,
        num_probes=4,
        max_cg_iterations=30,
        preconditioner_rank=10,
    )

    mean, std = gp.predict(X_test, return_std=True)
    rmse = float(np.sqrt(np.mean((mean - y_test) ** 2)))

    print(f"Method: {method}")
    print(f"RMSE:   {rmse:.4f}")
    print(f"Train route metadata:   {gp.backend_train_info}")
    print(f"Predict route metadata: {gp.backend_predict_info}")
    print(f"Learned params:         {gp.get_learned_params()}")
    print(f"Std summary: mean={float(np.mean(std)):.4f}, max={float(np.max(std)):.4f}")


if __name__ == "__main__":
    run_example("materialized")
    run_example("matrix_free")
