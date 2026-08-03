"""Champion selection and explicit rollback decisions."""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Optional


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
        self.lineage_head_version_id: Optional[str] = None
        self.champion_version_id: Optional[str] = None
        self._champion_score: Optional[float] = None
        self._degradation_streak = 0
        self._rollback_pending = False

    @classmethod
    def from_events(
        cls,
        events: Iterable[Mapping[str, Any]],
        *,
        rollback_patience: int = 3,
        rollback_margin: float = 0.05,
        rollback_enabled: bool = True,
    ) -> "LineageManager":
        manager = cls(
            rollback_patience=rollback_patience,
            rollback_margin=rollback_margin,
            rollback_enabled=rollback_enabled,
        )
        for event in events:
            event_type = event.get("event_type")
            if event_type == "version_created":
                manager.register_candidate(
                    str(event["version_id"]),
                    parent_version_id=(
                        None
                        if event.get("parent_version_id") is None
                        else str(event["parent_version_id"])
                    ),
                    status=str(event["evaluation_status"]),
                    score=(
                        None
                        if event.get("benchmark_score") is None
                        else float(event["benchmark_score"])
                    ),
                )
            elif event_type == "candidate_selected":
                manager.select_version(str(event["version_id"]))
            elif event_type == "search_parent_selected":
                manager.select_search_parent(str(event["version_id"]))
            elif event_type == "evaluation_completed":
                version_id = str(event["version_id"])
                existing = manager.versions.get(version_id)
                if existing is not None and existing.status != "complete":
                    manager.update_incomplete_evaluation(
                        version_id,
                        status=str(event["status"]),
                        score=(
                            None
                            if event.get("benchmark_score") is None
                            else float(event["benchmark_score"])
                        ),
                        select=(
                            manager.lineage_head_version_id == version_id
                        ),
                    )
            elif event_type == "rollback_selected":
                decision = manager.force_parent(
                    str(event["to_version_id"]),
                    reason=str(event["reason"]),
                )
                if not decision.rollback:
                    raise ValueError("rollback event is inconsistent with lineage state")
                if decision.from_version_id != str(event["from_version_id"]):
                    raise ValueError("rollback source does not match rebuilt lineage")
                if decision.to_version_id != str(event["to_version_id"]):
                    raise ValueError("rollback target does not match rebuilt lineage")
        return manager

    def record_evaluation(
        self,
        version_id: str,
        *,
        parent_version_id: Optional[str],
        status: str,
        score: Optional[float],
    ) -> bool:
        self.register_candidate(
            version_id,
            parent_version_id=parent_version_id,
            status=status,
            score=score,
        )
        return self.select_version(version_id)

    def register_candidate(
        self,
        version_id: str,
        *,
        parent_version_id: Optional[str],
        status: str,
        score: Optional[float],
    ) -> None:
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

    def update_incomplete_evaluation(
        self,
        version_id: str,
        *,
        status: str,
        score: Optional[float],
        select: bool = True,
    ) -> bool:
        """Finalize a new evaluation attempt for an existing incomplete version."""

        if version_id not in self.versions:
            raise KeyError(version_id)
        existing = self.versions[version_id]
        if existing.status == "complete":
            raise ValueError("complete version evaluation cannot be retried")
        if status == "complete":
            if score is None or not 0.0 <= score <= 1.0:
                raise ValueError("complete evaluation requires score in [0, 1]")
        elif status not in {"incomplete", "failed", "timeout"} or score is not None:
            raise ValueError("non-complete evaluation must have missing score")
        self.versions[version_id] = VersionEvaluation(
            version_id,
            existing.parent_version_id,
            status,
            score,
        )
        return self.select_version(version_id) if select else False

    def select_version(self, version_id: str) -> bool:
        if version_id not in self.versions:
            raise KeyError(version_id)
        evaluation = self.versions[version_id]
        self.lineage_head_version_id = version_id

        promoted = False
        if evaluation.status == "complete" and evaluation.score is not None:
            if self._champion_score is None or evaluation.score > self._champion_score:
                self.champion_version_id = evaluation.version_id
                self._champion_score = evaluation.score
                self._degradation_streak = 0
                self._rollback_pending = False
                promoted = True
            elif evaluation.score <= self._champion_score - self.rollback_margin:
                self._degradation_streak += 1
                self._rollback_pending = (
                    self.rollback_enabled
                    and self._degradation_streak >= self.rollback_patience
                )
            else:
                self._degradation_streak = 0
                self._rollback_pending = False
        return promoted

    def select_search_parent(self, version_id: str) -> bool:
        """Advance the exploratory parent without changing the champion."""

        if version_id not in self.versions:
            raise KeyError(version_id)
        self.lineage_head_version_id = version_id
        self._degradation_streak = 0
        self._rollback_pending = False
        return False

    def promote_champion(self, version_id: str) -> bool:
        """Promote only after the caller has completed frozen certification."""

        if version_id not in self.versions:
            raise KeyError(version_id)
        evaluation = self.versions[version_id]
        if evaluation.status != "complete" or evaluation.score is None:
            raise ValueError("champion promotion requires complete evaluation")
        if self._champion_score is not None and evaluation.score < self._champion_score:
            return False
        changed = self.champion_version_id != version_id
        self.champion_version_id = version_id
        self._champion_score = evaluation.score
        return changed

    def begin_stage(self, version_id: str, *, score: float) -> None:
        """Reset score comparison when the learning opponent changes."""

        if version_id not in self.versions:
            raise KeyError(version_id)
        if not 0.0 <= score <= 1.0:
            raise ValueError("stage score must be in [0, 1]")
        existing = self.versions[version_id]
        if existing.status != "complete":
            raise ValueError("stage origin requires a complete evaluation")
        self.versions[version_id] = VersionEvaluation(
            version_id=existing.version_id,
            parent_version_id=existing.parent_version_id,
            status="complete",
            score=float(score),
        )
        self.lineage_head_version_id = version_id
        self.champion_version_id = version_id
        self._champion_score = float(score)
        self._degradation_streak = 0
        self._rollback_pending = False

    def force_parent(
        self,
        version_id: str,
        *,
        reason: str = "curriculum_regression",
    ) -> ParentDecision:
        """Move to an explicit safe parent after curriculum regression."""

        if version_id not in self.versions:
            raise KeyError(version_id)
        if self.lineage_head_version_id is None:
            raise ValueError("cannot select parent before any version")
        if not reason:
            raise ValueError("rollback reason must be non-empty")
        from_version_id = self.lineage_head_version_id
        decision = ParentDecision(
            rollback=from_version_id != version_id,
            from_version_id=from_version_id,
            to_version_id=version_id,
            reason=reason,
        )
        self.lineage_head_version_id = version_id
        self._degradation_streak = 0
        self._rollback_pending = False
        return decision

    def select_next_parent(self) -> ParentDecision:
        if self.lineage_head_version_id is None:
            raise ValueError("cannot select parent before any version")
        if self._rollback_pending and self.champion_version_id is not None:
            decision = ParentDecision(
                rollback=True,
                from_version_id=self.lineage_head_version_id,
                to_version_id=self.champion_version_id,
                reason="sustained_degradation",
            )
            self._rollback_pending = False
            self._degradation_streak = 0
            self.lineage_head_version_id = decision.to_version_id
            return decision
        return ParentDecision(
            rollback=False,
            from_version_id=self.lineage_head_version_id,
            to_version_id=self.lineage_head_version_id,
            reason="continue_latest",
        )
