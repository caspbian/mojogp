"""Isolated child cases for public example smoke tests."""

from __future__ import annotations

from tests.shared.subprocess_harness import run_child_main


def _run_case(payload, _session):
    case = payload["case"]

    if case == "gp_training":
        from examples.basic_gp_training import run_example

        run_example(method="matrix_free", n_train=2000, n_test=32)
    elif case == "gp_inference":
        from examples.gp_inference import main as inference_main

        inference_main(n_train=2000, n_test=64)
    elif case == "fixed_observation_noise":
        from examples.fixed_observation_noise import run_example

        run_example(method="matrix_free", n_train=2000, n_test=32)
    elif case == "grouped_observation_noise":
        from examples.grouped_observation_noise import run_example

        run_example(method="matrix_free", n_train=2000, n_test=32)
    elif case == "input_dependent_observation_noise":
        from examples.input_dependent_observation_noise import run_example

        run_example(method="matrix_free", n_train=2000, n_test=32)
    elif case == "hyperparameter_optimization":
        from examples.hyperparameter_optimization import train_with_kernel
        from mojogp import RBF

        train_with_kernel(RBF(), "smoke", n_train=2000)
    elif case == "inference_benchmark":
        from examples.inference_benchmark import benchmark

        benchmark("materialized", n_train=2000, n_test=128)
    elif case == "single_output_continuous_features":
        from examples.single_output_continuous_features import run_example

        run_example(n_train=2000, n_test=16, method="materialized")
    elif case == "multi_output_workflow":
        from examples.multi_output_workflow import run_example

        run_example(n_train=2000, n_test=32)
    elif case == "multi_output_observation_noise":
        from examples.multi_output_observation_noise import run_example

        run_example(method="matrix_free", n_train=2000, n_test=32)
    elif case == "continuous_lmc":
        from examples.continuous_lmc_gp import run_example

        run_example()
    elif case == "lmc_ard":
        from examples.lmc_ard_relevance import run_example

        run_example(method="matrix_free")
    elif case == "mixed_lmc":
        from examples.mixed_lmc_workflow import run_example

        run_example(n_train=2000, n_test=32)
    else:
        raise ValueError(f"Unknown example smoke case: {case}")

    return {"payload": {"case": case}}


if __name__ == "__main__":
    raise SystemExit(run_child_main(_run_case))
