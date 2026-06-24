"""Hyperparameter training example using the live SingleOutputGP wrapper."""

import numpy as np

from mojogp import SingleOutputGP, RBF


def train_with_kernel(kernel, label: str, n_train: int = 5000):
    rng = np.random.RandomState(42)
    X = rng.randn(n_train, 3).astype(np.float32)
    y = (np.sin(X[:, 0]) + 0.2 * X[:, 1] + 0.05 * rng.randn(X.shape[0])).astype(
        np.float32
    )

    gp = SingleOutputGP(kernel, verbose=False)
    result = gp.fit(
        X,
        y,
        max_iterations=25,
        learning_rate=0.03,
        method="matrix_free",
        num_probes=4,
        max_cg_iterations=30,
        preconditioner_rank=10,
    )

    print(label)
    print(f"  final nll: {result.nll:.4f}")
    print(f"  learned:   {gp.get_learned_params()}")


if __name__ == "__main__":
    train_with_kernel(RBF(), "Isotropic RBF")
    train_with_kernel(RBF(ard=True), "ARD RBF")
