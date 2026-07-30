"""Champion selection and explicit rollback decisions."""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class VersionEvaluation:
    version_id: str
    parent_version_id: Optional[str]
    status: str
    score: Optional[float]


@dataclasses.dataclass(frozen=True)
class ParentDecision:
    rollback: bool
    from_version_id: str
    to_version_id: str
    reason: str


class LineageManager:
    """Retain all attempts while selecting a safe parent for the next act."""

    def __init__(
        self,
        *,
        rollback_patience: int = 3,
        rollback_margin: float = 0.05,
        rollback_enabled: bool = True,
    ) -> None:
        if rollback_patience < 1:
            raise ValueError("rollback_patience must be >= 1")
        if not 0.0 <= rollback_margin <= 1.0:
            raise ValueError("rollback_margin must be in [0, 1]")
        self.rollback_patience = rollback_patience
        self.rollback_margin = rollback_margin
        self.rollback_enabled = rollback_enabled
        self.versions: dict[str, VersionEvaluation] = {}
        self.latest_attempt_version_id: Optional[str] = None
        self.champion_version_id: Optional[str] = None
        self._champion_score: Optional[float] = None
        self._degradation_streak = 0
        self._rollback_pending = False

    def record_evaluation(
        self,
        version_id: str,
        *,
        parent_version_id: Optional[str],
        status: str,
        score: Optional[float],
    ) -> bool:
        if version_id in self.versions:
            raise ValueError(f"version already recorded: {version_id}")
        if status not in {"complete", "incomplete", "failed", "timeout"}:
            raise ValueError(f"unknown evaluation status: {status}")
        if status == "complete":
            if score is None or not 0.0 <= score <= 1.0:
                raise ValueError("complete evaluation requires score in [0, 1]")
        elif score is not None:
            raise ValueError("non-complete evaluation must have missing score")

        evaluation = VersionEvaluation(version_id, parent_version_id, status, score)
        self.versions[version_id] = evaluation
        self.latest_attempt_version_id = version_id

        promoted = False
        if status == "complete" and score is not None:
            if self._champion_score is None or score > self._champion_score:
                self.champion_version_id = version_id
                self._champion_score = score
                self._degradation_streak = 0
                self._rollback_pending = False
                promoted = True
            elif score <= self._champion_score - self.rollback_margin:
                self._degradation_streak += 1
                self._rollback_pending = (
                    self.rollback_enabled
                    and self._degradation_streak >= self.rollback_patience
                )
            else:
                self._degradation_streak = 0
                self._rollback_pending = False
        return promoted

    def select_next_parent(self) -> ParentDecision:
        if self.latest_attempt_version_id is None:
            raise ValueError("cannot select parent before any version")
        if self._rollback_pending and self.champion_version_id is not None:
            decision = ParentDecision(
                rollback=True,
                from_version_id=self.latest_attempt_version_id,
                to_version_id=self.champion_version_id,
                reason="sustained_degradation",
            )
            self._rollback_pending = False
            self._degradation_streak = 0
            return decision
        return ParentDecision(
            rollback=False,
            from_version_id=self.latest_attempt_version_id,
            to_version_id=self.latest_attempt_version_id,
            reason="continue_latest",
        )
