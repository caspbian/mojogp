"""SingleOutputGP with fixed input-dependent observation noise."""

import numpy as np

from mojogp import RBF, SingleOutputGP


def observation_noise_fn(X_eval: np.ndarray) -> np.ndarray:
    """Known sensor variance as a deterministic function of x."""
    return (0.01 + 0.04 * (X_eval[:, 0] > 0.0)).astype(np.float32)


def make_data(n_train: int = 5000, n_test: int = 256, seed: int = 41):
    rng = np.random.default_rng(seed)
    X_train = np.linspace(-3.0, 3.0, n_train, dtype=np.float32).reshape(-1, 1)
    train_noise = observation_noise_fn(X_train)
    latent_train = np.sin(2.5 * X_train[:, 0]).astype(np.float32)
    y_train = (
        latent_train + rng.normal(0.0, np.sqrt(train_noise)).astype(np.float32)
    ).astype(np.float32)

    X_test = np.linspace(-2.8, 2.8, n_test, dtype=np.float32).reshape(-1, 1)
    latent_test = np.sin(2.5 * X_test[:, 0]).astype(np.float32)
    return X_train, y_train, X_test, latent_test


def run_example(method: str = "matrix_free", n_train: int = 5000, n_test: int = 256):
    X_train, y_train, X_test, latent_test = make_data(
        n_train=n_train,
        n_test=n_test,
    )

    gp = SingleOutputGP(RBF(), verbose=False)
    gp.fit(
        X_train,
        y_train,
        noise_model="input_dependent",
        observation_noise_fn=observation_noise_fn,
        learn_noise=False,
        method=method,
        max_iterations=20,
        learning_rate=0.03,
        num_probes=4,
        max_cg_iterations=50,
        preconditioner_rank=0,
    )

    latent = gp.predict_latent(X_test, variance_method="mean_only")
    observed = gp.predict_observed(
        X_test,
        observation_noise_fn=observation_noise_fn,
        variance_method="exact",
        max_cg_iterations=80,
        preconditioner_rank=0,
    )
    rmse = float(np.sqrt(np.mean((latent.mean - latent_test) ** 2)))

    print(f"Method: {method}")
    print(f"Latent RMSE: {rmse:.4f}")
    print(f"Observed std mean: {float(np.mean(observed.std)):.4f}")
    print(f"Train route metadata: {gp.backend_train_info}")


if __name__ == "__main__":
    run_example("materialized")
    run_example("matrix_free")
