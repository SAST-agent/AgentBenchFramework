"""Redacted learning replays passed to the coding agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping

from .models import MatchResult


@dataclass(frozen=True)
class DecisionRecord:
    state_id: str
    round_number: int
    seat: int
    state: Mapping[str, Any]
    action: tuple[tuple[int, ...], ...]
    outcome: str


@dataclass(frozen=True)
class LearningReplay:
    replay_id: str
    seed: int
    evaluated_seat: int
    opponent_tier: str
    termination_type: str
    decisions: tuple[DecisionRecord, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def build_learning_replay(
    match: MatchResult,
    evaluated_agent_id: str,
    opponent_tier: str = "unknown",
) -> LearningReplay:
    del evaluated_agent_id
    if match.winner == -1:
        outcome = "draw"
    elif match.winner == match.evaluated_seat:
        outcome = "win"
    else:
        outcome = "loss"
    decisions = tuple(
        DecisionRecord(
            state_id=turn.state_id_before,
            round_number=turn.round_number,
            seat=turn.player,
            state=turn.state_before,
            action=turn.commands,
            outcome=outcome,
        )
        for turn in match.turns
        if turn.player == match.evaluated_seat
    )
    return LearningReplay(
        replay_id=f"learning-{match.seed}-seat-{match.evaluated_seat}",
        seed=match.seed,
        evaluated_seat=match.evaluated_seat,
        opponent_tier=opponent_tier,
        termination_type=match.termination_type,
        decisions=decisions,
    )
