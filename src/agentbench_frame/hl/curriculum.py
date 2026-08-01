"""Event-rebuildable weakest-failed opponent curriculum state."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class CertificationSummary:
    pass_rates: Mapping[str, float]
    ranks: Mapping[str, int]
    passing_opponents: int
    passed_opponents: tuple[str, ...]
    failed_opponents: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CurriculumState:
    active_target: Optional[str]
    locked_opponents: tuple[str, ...]
    stage_origin_version_id: str
    stage_best_version_id: str
    stage_best_score: Optional[float]
    stagnation_count: int
    completed: bool


@dataclasses.dataclass(frozen=True)
class CurriculumDecision:
    kind: str
    parent_version_id: str
    next_target: Optional[str] = None
    lost_locked_opponents: tuple[str, ...] = ()


def summarize_certification(
    matches: Iterable[Mapping[str, Any]],
    *,
    required_win_rate: float,
    expected_opponents: int,
) -> CertificationSummary:
    """Reduce a complete fixed-seed certification into per-opponent rates."""

    if not 0.0 <= required_win_rate <= 1.0:
        raise ValueError("required_win_rate must be in [0, 1]")
    if expected_opponents < 1:
        raise ValueError("expected_opponents must be positive")
    results: dict[str, list[str]] = {}
    ranks: dict[str, int] = {}
    for match in matches:
        if match.get("status") != "complete":
            raise ValueError("certification requires complete matches")
        result = match.get("result")
        if result not in {"win", "draw", "loss"}:
            raise ValueError("complete certification match has invalid result")
        opponent = str(match.get("opponent") or "")
        if not opponent:
            raise ValueError("certification match is missing opponent")
        rank_value = match.get("opponent_rank")
        if not isinstance(rank_value, int) or isinstance(rank_value, bool):
            raise ValueError("certification match is missing opponent rank")
        if opponent in ranks and ranks[opponent] != rank_value:
            raise ValueError(f"inconsistent rank for opponent {opponent}")
        ranks[opponent] = rank_value
        results.setdefault(opponent, []).append(str(result))
    if len(results) != expected_opponents:
        raise ValueError(
            f"certification requires {expected_opponents} opponents"
        )
    pass_rates = {
        opponent: (
            sum(result == "win" for result in opponent_results)
            + 0.5 * sum(result == "draw" for result in opponent_results)
        )
        / len(opponent_results)
        for opponent, opponent_results in results.items()
    }
    ordered = tuple(sorted(results, key=lambda opponent: ranks[opponent]))
    passed = tuple(
        opponent
        for opponent in ordered
        if pass_rates[opponent] >= required_win_rate
    )
    failed = tuple(
        opponent
        for opponent in ordered
        if pass_rates[opponent] < required_win_rate
    )
    return CertificationSummary(
        pass_rates=pass_rates,
        ranks=ranks,
        passing_opponents=len(passed),
        passed_opponents=passed,
        failed_opponents=failed,
    )


def select_weakest_failed(
    summary: CertificationSummary,
) -> Optional[str]:
    """Return the numerically largest rank among failed opponents."""

    if not summary.failed_opponents:
        return None
    return max(
        summary.failed_opponents,
        key=lambda opponent: summary.ranks[opponent],
    )


def rerank_certification(
    summary: CertificationSummary,
    *,
    hardest_to_easiest: Iterable[str],
) -> CertificationSummary:
    """Replace filename ranks with one frozen empirical Ghost order."""

    order = tuple(str(opponent) for opponent in hardest_to_easiest)
    expected = set(summary.pass_rates)
    if len(order) != len(expected) or set(order) != expected:
        raise ValueError("empirical order must contain every certified opponent once")
    ranks = {opponent: index + 1 for index, opponent in enumerate(order)}
    passed = tuple(
        opponent for opponent in order if opponent in summary.passed_opponents
    )
    failed = tuple(
        opponent for opponent in order if opponent in summary.failed_opponents
    )
    return CertificationSummary(
        pass_rates=summary.pass_rates,
        ranks=ranks,
        passing_opponents=summary.passing_opponents,
        passed_opponents=passed,
        failed_opponents=failed,
    )


class CurriculumManager:
    """Own target, locked pool, safe stage origin, and stagnation state."""

    def __init__(
        self,
        state: CurriculumState,
        *,
        required_human_opponents: int,
        stagnation_patience: int,
        rollback_patience: int | None,
    ) -> None:
        if required_human_opponents < 1:
            raise ValueError("required_human_opponents must be positive")
        if stagnation_patience < 1:
            raise ValueError("stagnation_patience must be positive")
        if rollback_patience is not None and rollback_patience < 1:
            raise ValueError("rollback_patience must be positive")
        self._state = state
        self.required_human_opponents = required_human_opponents
        self.stagnation_patience = stagnation_patience
        self.rollback_patience = rollback_patience

    @property
    def state(self) -> CurriculumState:
        return self._state

    @classmethod
    def start(
        cls,
        *,
        version_id: str,
        summary: CertificationSummary,
        required_human_opponents: int,
        stagnation_patience: int,
        rollback_patience: int | None,
    ) -> "CurriculumManager":
        completed = (
            summary.passing_opponents >= required_human_opponents
            and not summary.failed_opponents
        )
        active_target = None if completed else select_weakest_failed(summary)
        if not completed and active_target is None:
            raise ValueError("curriculum has no selectable failed opponent")
        state = CurriculumState(
            active_target=active_target,
            locked_opponents=summary.passed_opponents,
            stage_origin_version_id=version_id,
            stage_best_version_id=version_id,
            stage_best_score=None,
            stagnation_count=0,
            completed=completed,
        )
        return cls(
            state,
            required_human_opponents=required_human_opponents,
            stagnation_patience=stagnation_patience,
            rollback_patience=rollback_patience,
        )

    @classmethod
    def from_events(
        cls,
        events: Iterable[Mapping[str, Any]],
        *,
        required_human_opponents: int,
        stagnation_patience: int,
        rollback_patience: int | None,
    ) -> "CurriculumManager":
        manager: CurriculumManager | None = None
        for event in events:
            event_type = event.get("event_type")
            if event_type == "curriculum_started":
                version_id = str(event["stage_origin_version_id"])
                state = CurriculumState(
                    active_target=(
                        None
                        if event.get("active_target") is None
                        else str(event["active_target"])
                    ),
                    locked_opponents=tuple(
                        str(value)
                        for value in event.get("locked_opponents", ())
                    ),
                    stage_origin_version_id=version_id,
                    stage_best_version_id=version_id,
                    stage_best_score=None,
                    stagnation_count=0,
                    completed=event.get("active_target") is None,
                )
                manager = cls(
                    state,
                    required_human_opponents=required_human_opponents,
                    stagnation_patience=stagnation_patience,
                    rollback_patience=rollback_patience,
                )
            elif manager is None:
                continue
            elif event_type == "curriculum_gate_completed":
                manager._state = dataclasses.replace(
                    manager._state,
                    stage_best_version_id=str(
                        event["stage_best_version_id"]
                    ),
                    stage_best_score=float(event["stage_best_score"]),
                    stagnation_count=int(event["stagnation_count"]),
                )
            elif event_type == "curriculum_stage_promoted":
                next_target = event.get("next_target")
                manager._state = CurriculumState(
                    active_target=(
                        None if next_target is None else str(next_target)
                    ),
                    locked_opponents=tuple(
                        str(value)
                        for value in event["locked_opponents"]
                    ),
                    stage_origin_version_id=str(event["version_id"]),
                    stage_best_version_id=str(event["version_id"]),
                    stage_best_score=None,
                    stagnation_count=0,
                    completed=next_target is None,
                )
            elif event_type == "curriculum_resumed":
                manager._state = dataclasses.replace(
                    manager._state,
                    stage_best_version_id=str(
                        event["stage_best_version_id"]
                    ),
                    stage_best_score=float(event["stage_best_score"]),
                    stagnation_count=0,
                )
        if manager is None:
            raise ValueError("curriculum events do not contain a start event")
        return manager

    def begin_stage_gate(self, *, version_id: str, score: float) -> None:
        if self._state.completed or self._state.active_target is None:
            raise RuntimeError("completed curriculum has no stage gate")
        if not 0.0 <= score <= 1.0:
            raise ValueError("gate score must be in [0, 1]")
        self._state = dataclasses.replace(
            self._state,
            stage_best_version_id=version_id,
            stage_best_score=float(score),
            stagnation_count=0,
        )

    def observe_gate(
        self,
        *,
        version_id: str,
        score: float,
    ) -> CurriculumDecision:
        if self._state.completed or self._state.active_target is None:
            raise RuntimeError("completed curriculum cannot observe a gate")
        if not 0.0 <= score <= 1.0:
            raise ValueError("gate score must be in [0, 1]")
        best = self._state.stage_best_score
        if best is None or score > best:
            self._state = dataclasses.replace(
                self._state,
                stage_best_version_id=version_id,
                stage_best_score=float(score),
                stagnation_count=0,
            )
            return CurriculumDecision(
                kind="improved",
                parent_version_id=version_id,
            )
        count = self._state.stagnation_count + 1
        self._state = dataclasses.replace(
            self._state,
            stagnation_count=count,
        )
        if count >= self.stagnation_patience:
            return CurriculumDecision(
                kind="stagnated",
                parent_version_id=self._state.stage_best_version_id,
            )
        if (
            self.rollback_patience is not None
            and count >= self.rollback_patience
        ):
            return CurriculumDecision(
                kind="rollback",
                parent_version_id=self._state.stage_best_version_id,
            )
        return CurriculumDecision(
            kind="continue",
            parent_version_id=version_id,
        )

    def resume_after_stagnation(self) -> str:
        if self._state.completed or self._state.active_target is None:
            raise RuntimeError("completed curriculum cannot resume a stage")
        if self._state.stage_best_score is None:
            raise RuntimeError("curriculum stage has no established gate")
        if self._state.stagnation_count < self.stagnation_patience:
            raise RuntimeError("curriculum stage has not stagnated")
        self._state = dataclasses.replace(
            self._state,
            stagnation_count=0,
        )
        return self._state.stage_best_version_id

    def observe_certification(
        self,
        *,
        version_id: str,
        summary: CertificationSummary,
    ) -> CurriculumDecision:
        active = self._state.active_target
        if self._state.completed or active is None:
            raise RuntimeError(
                "completed curriculum cannot observe certification"
            )
        passed = set(summary.passed_opponents)
        lost = tuple(
            opponent
            for opponent in self._state.locked_opponents
            if opponent not in passed
        )
        if lost or active not in passed:
            return CurriculumDecision(
                kind="reject",
                parent_version_id=self._state.stage_origin_version_id,
                next_target=active,
                lost_locked_opponents=lost,
            )
        completed = (
            summary.passing_opponents >= self.required_human_opponents
            and not summary.failed_opponents
        )
        next_target = None if completed else select_weakest_failed(summary)
        if not completed and next_target is None:
            raise ValueError("promoted curriculum has no next target")
        self._state = CurriculumState(
            active_target=next_target,
            locked_opponents=summary.passed_opponents,
            stage_origin_version_id=version_id,
            stage_best_version_id=version_id,
            stage_best_score=None,
            stagnation_count=0,
            completed=completed,
        )
        return CurriculumDecision(
            kind="complete" if completed else "promote",
            parent_version_id=version_id,
            next_target=next_target,
        )
