"""Game-neutral evaluation boundary consumed by the HL controller."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Protocol

from agentbench_frame.hl.codebase import Version


@dataclasses.dataclass(frozen=True)
class CandidateEvaluation:
    status: str
    score: Optional[float]
    error: Optional[str] = None
    matches: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "matches", tuple(self.matches))
        if self.status not in {"complete", "incomplete", "failed", "timeout"}:
            raise ValueError(f"unknown evaluation status: {self.status}")
        if self.status == "complete":
            if self.score is None or not 0.0 <= self.score <= 1.0:
                raise ValueError("complete evaluation requires score in [0, 1]")
        elif self.score is not None:
            raise ValueError("non-complete evaluation cannot have aggregate score")


class CandidateEvaluator(Protocol):
    def evaluate(self, version: Version) -> CandidateEvaluation:
        ...
