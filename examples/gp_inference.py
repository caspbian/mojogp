"""SingleOutputGP inference example with exact and LOVE variance."""

import numpy as np

from mojogp import SingleOutputGP, RBF


def main(n_train: int = 4000, n_test: int = 256):
    rng = np.random.RandomState(42)
    X_train = rng.uniform(-4.0, 4.0, size=(n_train, 1)).astype(np.float32)
    y_train = (np.sin(1.5 * X_train[:, 0]) + 0.05 * rng.randn(X_train.shape[0])).astype(
        np.float32
    )
    X_test = np.linspace(-4.5, 4.5, n_test, dtype=np.float32).reshape(-1, 1)

    gp = SingleOutputGP(RBF(), verbose=False)
    gp.fit(
        X_train, y_train, max_iterations=20, learning_rate=0.03, method="matrix_free"
    )

    exact = gp.predict(X_test, variance_method="exact")
    love = gp.predict(X_test, variance_method="love")

    print("SingleOutputGP inference example")
    print(f"Mean shape:        {exact.mean.shape}")
    print(f"Exact var mean:    {float(np.mean(exact.variance)):.6f}")
    print(f"LOVE var mean:     {float(np.mean(love.variance)):.6f}")
    print(
        f"Variance corr:     {float(np.corrcoef(exact.variance, love.variance)[0, 1]):.4f}"
    )
    print(f"Predict metadata:  {gp.backend_predict_info}")


if __name__ == "__main__":
    main()
