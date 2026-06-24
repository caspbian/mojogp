"""Smoke tests for the rewritten public examples."""

from tests.shared.subprocess_harness import run_isolated_case


MODULE = "tests.integration.run_examples_smoke_case"


def _run_example_case(case: str):
    return run_isolated_case(
        module=MODULE,
        payload={"case": case},
        timeout=600,
        description=f"Runs example smoke case {case}",
    )


def test_gp_training_example_completes():
    _run_example_case("gp_training")


def test_gp_inference_example_completes():
    _run_example_case("gp_inference")


def test_fixed_observation_noise_example_completes():
    _run_example_case("fixed_observation_noise")


def test_grouped_observation_noise_example_completes():
    _run_example_case("grouped_observation_noise")


def test_input_dependent_observation_noise_example_completes():
    _run_example_case("input_dependent_observation_noise")


def test_hyperparameter_optimization_example_completes():
    _run_example_case("hyperparameter_optimization")


def test_inference_benchmark_example_completes():
    _run_example_case("inference_benchmark")


def test_single_output_continuous_features_example_completes():
    _run_example_case("single_output_continuous_features")


def test_multi_output_workflow_example_completes():
    _run_example_case("multi_output_workflow")


def test_continuous_lmc_example_completes():
    _run_example_case("continuous_lmc")


def test_lmc_ard_example_completes():
    _run_example_case("lmc_ard")


def test_mixed_lmc_example_completes():
    _run_example_case("mixed_lmc")


def test_multi_output_observation_noise_example_completes():
    _run_example_case("multi_output_observation_noise")
