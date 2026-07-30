"""Shared HL lifecycle value objects."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Optional


class LifecycleStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    TIMEOUT = "timeout"


@dataclasses.dataclass(frozen=True)
class EvaluationPoint:
    act_id: str
    version_id: str
    status: LifecycleStatus
    score: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status is LifecycleStatus.COMPLETED and self.score is None:
            raise ValueError("completed evaluation requires a score")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("evaluation score must be in [0, 1]")
