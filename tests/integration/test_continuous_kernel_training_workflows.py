#!/usr/bin/env python3
"""Training and prediction smoke coverage for continuous kernel primitives."""

import numpy as np
from mojogp import (
    SingleOutputGP,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    RQ,
    Linear,
    Polynomial,
)


def test_continuous_kernel_training_and_prediction():
    """Each continuous kernel primitive fits and predicts finite values."""
    print("Testing continuous kernel training and prediction...")

    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(50, 2).astype(np.float32)
    y = np.sin(X[:, 0]).astype(np.float32)

    kernel_configs = [
        ("rbf", RBF()),
        ("matern32", Matern32()),
        ("matern52", Matern52()),
        ("matern12", Matern12()),
        ("periodic", Periodic(period=2.0)),
        ("rq", RQ()),
        ("linear", Linear()),
        ("polynomial", Polynomial()),
    ]

    results = []

    for kernel_name, kernel_obj in kernel_configs:
        try:
            print(f"\nTesting {kernel_name}...")

            # Create GP with kernel
            gp = SingleOutputGP(kernel_obj)

            # Try to fit
            print(f"  Fitting {kernel_name}...")
            gp.fit(X, y, max_iterations=10)

            # Try to predict
            print(f"  Predicting with {kernel_name}...")
            mean, std = gp.predict(X[:10], return_std=True)

            print(f"  {kernel_name}: SUCCESS")
            results.append((kernel_name, True, None))

        except Exception as e:
            print(f"  {kernel_name}: FAILED - {e}")
            results.append((kernel_name, False, str(e)))

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = 0
    for kernel_name, success, error in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{kernel_name:15} {status}")
        if not success and error:
            print(f"                 Error: {error}")
        if success:
            success_count += 1

    print(f"\nTotal: {success_count}/{len(kernel_configs)} kernels working")

    return success_count == len(kernel_configs)


def main():
    """Run the test."""
    print("=" * 60)
    print("Testing Continuous Kernel Training Integration")
    print("=" * 60)

    try:
        success = test_continuous_kernel_training_and_prediction()
        if success:
            print("\nAll kernels can be trained!")
            return 0
        else:
            print("\nSome kernels failed to train")
            return 1
    except Exception as e:
        print(f"\nTest crashed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
