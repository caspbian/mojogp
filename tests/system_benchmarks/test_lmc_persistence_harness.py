"""Harness benchmark for MultiOutputLMCGP save/load workflows."""

from __future__ import annotations

import pytest

from tests.benchmarks.prediction_workload import BENCHMARK_PREDICTION_N_TEST
from tests.benchmarks.workflow_runner import run_workflow_benchmark_subprocess
from tests.shared.gpu_test_utils import assert_gpu_available, assert_gpu_was_used, requires_cuda


@pytest.mark.minimal
@pytest.mark.multi_output
@requires_cuda
def test_lmc_persistence_harness(results_dir):
    assert_gpu_available()
    benchmark = run_workflow_benchmark_subprocess(
        module="tests.system_benchmarks.run_lmc_workflow_case",
        payload={
            "case": "persistence",
            "n_train": 700,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "n_samples": 24,
        },
        suite_name="lmc_workflow",
        benchmark_name="lmc_persistence_harness",
        framework="mojogp",
        case_id="lmc.persistence.materialized",
        benchmark_group_id="lmc.persistence.materialized",
        config={
            "framework": "mojogp",
            "training_method": "materialized",
            "prediction_mode": "love",
            "workflow": "persistence",
            "n": 700,
            "n_test": BENCHMARK_PREDICTION_N_TEST,
            "d": 1,
            "num_tasks": 3,
        },
        results_dir=results_dir,
    )
    assert_gpu_was_used(benchmark)
    assert "benchmark_contracts" not in benchmark.config
    assert benchmark.config["variance_round_trip_policy"] == "hard_assertion"
    assert benchmark.config["iteration_timing_policy"] == "direct_backend_median"
    assert benchmark.speed.ms_per_iteration is not None
    assert benchmark.speed.startup_prepare_time_s is not None
    if benchmark.speed.startup_compile_time_s is not None:
        assert benchmark.speed.startup_warm_cache_hit_s is not None
