"""Redacted learning replays passed to the coding agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping

from .dense import DenseEpisodeSummary, DenseStateSample
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


@dataclass(frozen=True)
class CompactDecision:
    state_id: str
    round_number: int
    seat: int
    action: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class CompactLearningEvidence:
    replay_id: str
    seed: int
    evaluated_seat: int
    opponent_tier: str
    termination_type: str
    outcome: str
    dense: Mapping[str, Any]
    decisions: tuple[CompactDecision, ...]

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


def build_compact_evidence(
    replay: LearningReplay,
    dense_summary: DenseEpisodeSummary,
    dense_trace: tuple[DenseStateSample, ...],
    max_decisions: int = 4,
) -> CompactLearningEvidence:
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    count = len(replay.decisions)
    if count <= max_decisions:
        selected = replay.decisions
    else:
        tail_start = count - (max_decisions - 1)
        selected = (replay.decisions[0], *replay.decisions[tail_start:])
    decisions = tuple(
        CompactDecision(
            state_id=item.state_id,
            round_number=item.round_number,
            seat=item.seat,
            action=item.action,
        )
        for item in selected
    )
    dense = {
        "terminal_round": dense_summary.terminal_round,
        "completed_rounds_survived": dense_summary.completed_rounds_survived,
        "summary_sample_count": dense_summary.sample_count,
        "trace_sample_count": len(dense_trace),
        "territory_share": asdict(dense_summary.territory_share),
        "territory_margin": asdict(dense_summary.territory_margin),
        "army_share": asdict(dense_summary.army_share),
        "army_margin": asdict(dense_summary.army_margin),
        "coin_share": asdict(dense_summary.coin_share),
        "coin_margin": asdict(dense_summary.coin_margin),
        "net_main_pressure": asdict(dense_summary.net_main_pressure),
    }
    return CompactLearningEvidence(
        replay_id=replay.replay_id,
        seed=replay.seed,
        evaluated_seat=replay.evaluated_seat,
        opponent_tier=replay.opponent_tier,
        termination_type=replay.termination_type,
        outcome=dense_summary.outcome,
        dense=dense,
        decisions=decisions,
    )
