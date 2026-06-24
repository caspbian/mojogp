"""Benchmark contract definitions for historical and absolute benchmark gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Iterable, Literal


ComparisonMode = Literal["max_regression_pct", "min_improvement_pct", "absolute_max", "absolute_min"]


@dataclass(frozen=True)
class HistoricalBaselineKey:
    benchmark_name: str
    framework: str
    benchmark_group_id: str | None = None
    case_id: str | None = None
    training_method: str | None = None
    prediction_mode: str | None = None
    gpu_name: str | None = None
    gpu_class: str | None = None


@dataclass(frozen=True)
class BenchmarkContract:
    name: str
    metric: str
    mode: ComparisonMode
    threshold: float
    source: Literal["absolute", "historical", "cross_framework", "parity", "scaling"]
    description: str
    baseline_key: HistoricalBaselineKey | None = None


@dataclass(frozen=True)
class ContractEvaluation:
    contract_name: str
    metric: str
    mode: ComparisonMode
    passed: bool
    observed_value: float | None
    threshold: float
    baseline_value: float | None
    source: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContractEvaluationSummary:
    passed: bool
    evaluations: tuple[ContractEvaluation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }


def rolling_median(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return float(median(materialized))


def evaluate_contract(
    contract: BenchmarkContract,
    *,
    observed_value: float | None,
    baseline_value: float | None = None,
) -> ContractEvaluation:
    passed = False
    detail = "metric missing"

    if observed_value is not None:
        observed = float(observed_value)
        threshold = float(contract.threshold)
        if contract.mode == "absolute_max":
            passed = observed <= threshold
            detail = f"observed={observed:.6g} <= max={threshold:.6g}"
        elif contract.mode == "absolute_min":
            passed = observed >= threshold
            detail = f"observed={observed:.6g} >= min={threshold:.6g}"
        elif baseline_value is not None:
            baseline = float(baseline_value)
            if contract.mode == "max_regression_pct":
                max_allowed = baseline * (1.0 + threshold / 100.0)
                passed = observed <= max_allowed
                detail = (
                    f"observed={observed:.6g} <= baseline*(1+{threshold:.3g}%)={max_allowed:.6g}"
                )
            elif contract.mode == "min_improvement_pct":
                min_required = baseline * (1.0 - threshold / 100.0)
                passed = observed <= min_required
                detail = (
                    f"observed={observed:.6g} <= baseline*(1-{threshold:.3g}%)={min_required:.6g}"
                )
            else:
                detail = "baseline comparison requested with unsupported mode"
        else:
            detail = "baseline missing"

    return ContractEvaluation(
        contract_name=contract.name,
        metric=contract.metric,
        mode=contract.mode,
        passed=passed,
        observed_value=None if observed_value is None else float(observed_value),
        threshold=float(contract.threshold),
        baseline_value=None if baseline_value is None else float(baseline_value),
        source=contract.source,
        detail=detail,
    )


def summarize_evaluations(evaluations: Iterable[ContractEvaluation]) -> ContractEvaluationSummary:
    evaluation_tuple = tuple(evaluations)
    return ContractEvaluationSummary(
        passed=all(evaluation.passed for evaluation in evaluation_tuple),
        evaluations=evaluation_tuple,
    )
