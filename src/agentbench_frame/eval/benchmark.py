"""Frozen benchmark specifications and aggregate scoring."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


_OUTCOMES = {"win", "loss", "draw"}


@dataclass(frozen=True)
class BenchmarkCase:
    """One deterministic case in a versioned benchmark."""

    case_id: str
    opponent: str
    seed: int
    first_player: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.first_player not in (0, 1):
            raise ValueError("first_player must be 0 or 1")


@dataclass
class BenchmarkSpec:
    """A frozen, versioned collection of benchmark cases."""

    version: str
    cases: List[BenchmarkCase]

    def __post_init__(self) -> None:
        self.cases = list(self.cases)
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("benchmark case_id values must be unique")

    @property
    def case_ids(self) -> set[str]:
        return {case.case_id for case in self.cases}


@dataclass
class GameResult:
    """Raw result for one benchmark case, relative to the evaluated agent."""

    case_id: str
    outcome: str
    valid: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"outcome must be one of {_OUTCOMES}")


@dataclass
class BenchmarkEvaluation:
    """Derived aggregate while retaining every raw case result."""

    spec_version: str
    status: str
    score: Optional[float]
    wins: int
    losses: int
    draws: int
    results: List[GameResult]
    per_case: Dict[str, GameResult]


def evaluate_benchmark(
    spec: BenchmarkSpec, results: Iterable[GameResult]
) -> BenchmarkEvaluation:
    """Evaluate a result set against a frozen benchmark specification.

    An aggregate score exists only when every specified case has one valid
    result. External failures remain in ``results`` but do not become losses.
    """

    result_list = list(results)
    per_case: Dict[str, GameResult] = {}
    for result in result_list:
        if result.case_id not in spec.case_ids:
            raise ValueError(f"result references unknown benchmark case {result.case_id!r}")
        if result.case_id in per_case:
            raise ValueError(f"duplicate result for benchmark case {result.case_id!r}")
        per_case[result.case_id] = result

    complete = len(per_case) == len(spec.cases) and all(
        result.valid for result in per_case.values()
    )
    valid_results = [result for result in result_list if result.valid]
    wins = sum(result.outcome == "win" for result in valid_results)
    losses = sum(result.outcome == "loss" for result in valid_results)
    draws = sum(result.outcome == "draw" for result in valid_results)
    score = ((wins + 0.5 * draws) / (wins + losses + draws)) if complete else None

    return BenchmarkEvaluation(
        spec_version=spec.version,
        status="complete" if complete else "incomplete",
        score=score,
        wins=wins,
        losses=losses,
        draws=draws,
        results=result_list,
        per_case=per_case,
    )
