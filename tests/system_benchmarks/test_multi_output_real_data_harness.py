"""Multi-output real-data benchmarks."""

from __future__ import annotations

import pytest

from tests.benchmarks.multi_output_real_data import run_multi_output_real_data_case
from tests.shared.benchmarking.real_datasets import has_energy_efficiency_multi_output_data

from .conftest import assert_gpu_available, requires_cuda


def _run_case_subprocess(case: str, results_dir) -> dict[str, object]:
    return run_multi_output_real_data_case(case=case, results_dir=results_dir)


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_linnerud_multioutput_real_data(results_dir):
    assert_gpu_available()
    payload = _run_case_subprocess("linnerud_icm", results_dir)
    benchmark = payload["benchmark"]

    assert benchmark["memory"]["gpu_max_mb"] > 0, "GPU was not used during benchmark"
    assert benchmark["accuracy"]["rmse"] < 3.0, (
        "Real multi-output RMSE is implausibly large"
    )


@pytest.mark.minimal
@pytest.mark.multi_output
@pytest.mark.accuracy
@requires_cuda
def test_energy_efficiency_multioutput_real_data(results_dir):
    assert_gpu_available()
    if not has_energy_efficiency_multi_output_data():
        pytest.skip("Energy Efficiency benchmark data is not vendored in this checkout.")
    payload = _run_case_subprocess("energy_lmc", results_dir)
    benchmark = payload["benchmark"]
    icm_rmse = float(payload["icm_rmse"])

    assert benchmark["memory"]["gpu_max_mb"] > 0, "GPU was not used during benchmark"
    assert benchmark["accuracy"]["r_squared"] >= 0.95, (
        f"Energy Efficiency LMC R^2 too low: {benchmark['accuracy']['r_squared']:.4f}"
    )
    assert benchmark["accuracy"]["rmse"] <= benchmark["config"]["ridge_rmse"] * 0.7, (
        f"Energy Efficiency LMC did not beat ridge strongly enough: "
        f"lmc={benchmark['accuracy']['rmse']:.4f}, ridge={benchmark['config']['ridge_rmse']:.4f}"
    )
    assert (
        benchmark["accuracy"]["rmse"] <= benchmark["config"]["independent_rmse"] * 1.6
    ), (
        f"Energy Efficiency LMC regressed too far versus independent ExactGP: "
        f"lmc={benchmark['accuracy']['rmse']:.4f}, independent={benchmark['config']['independent_rmse']:.4f}"
    )
    assert benchmark["accuracy"]["rmse"] <= icm_rmse * 0.5, (
        f"Energy Efficiency LMC did not improve enough over ICM: "
        f"lmc={benchmark['accuracy']['rmse']:.4f}, icm={icm_rmse:.4f}"
    )
