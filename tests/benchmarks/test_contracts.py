from __future__ import annotations

from tests.benchmarks.contracts import (
    BenchmarkContract,
    ContractEvaluation,
    evaluate_contract,
    rolling_median,
    summarize_evaluations,
)


def test_evaluate_absolute_max_contract_passes_when_observed_below_threshold():
    evaluation = evaluate_contract(
        BenchmarkContract(
            name="rmse_cap",
            metric="rmse",
            mode="absolute_max",
            threshold=0.5,
            source="absolute",
            description="RMSE must stay below the cap.",
        ),
        observed_value=0.3,
    )

    assert evaluation.passed is True
    assert "observed=0.3" in evaluation.detail


def test_evaluate_historical_regression_contract_uses_baseline_percentage():
    evaluation = evaluate_contract(
        BenchmarkContract(
            name="training_time_regression",
            metric="training_time_s",
            mode="max_regression_pct",
            threshold=10.0,
            source="historical",
            description="Training time may regress by at most 10%.",
        ),
        observed_value=10.9,
        baseline_value=10.0,
    )

    assert evaluation.passed is True
    assert "baseline*(1+10%" in evaluation.detail


def test_summarize_evaluations_requires_all_contracts_to_pass():
    summary = summarize_evaluations(
        [
            ContractEvaluation(
                contract_name="a",
                metric="rmse",
                mode="absolute_max",
                passed=True,
                observed_value=0.1,
                threshold=0.5,
                baseline_value=None,
                source="absolute",
                detail="ok",
            ),
            ContractEvaluation(
                contract_name="b",
                metric="training_time_s",
                mode="absolute_max",
                passed=False,
                observed_value=11.0,
                threshold=10.0,
                baseline_value=None,
                source="absolute",
                detail="too slow",
            ),
        ]
    )

    assert summary.passed is False
    assert len(summary.evaluations) == 2


def test_rolling_median_uses_sorted_middle_value():
    assert rolling_median([5.0, 1.0, 9.0]) == 5.0
