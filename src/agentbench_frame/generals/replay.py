"""Redacted learning replays passed to the coding agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from collections.abc import Sequence
from typing import Any, Mapping

from .dense import DenseEpisodeSummary, DenseStateSample, dense_sample
from .measurement import classify_macro_action
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


@dataclass(frozen=True)
class CriticalDecision:
    state_id: str
    round_number: int
    seat: int
    selection_reasons: tuple[str, ...]
    action: tuple[tuple[int, ...], ...]
    decision_class: str
    features: Mapping[str, Any]


@dataclass(frozen=True)
class CriticalLearningEvidence:
    replay_id: str
    seed: int
    evaluated_seat: int
    opponent_tier: str
    termination_type: str
    outcome: str
    dense: Mapping[str, Any]
    total_decision_count: int
    omitted_decision_count: int
    decisions: tuple[CriticalDecision, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class CriticalWindowSelection:
    replay_id: str
    total_decision_count: int
    selected_state_ids: tuple[str, ...]
    reasons: Mapping[str, tuple[str, ...]]
    omitted_decision_count: int


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


def _main_general(
    state: Mapping[str, Any],
    seat: int,
) -> Mapping[str, Any] | None:
    candidates = [
        item
        for item in state.get("generals", {}).values()
        if isinstance(item, Mapping)
        and item.get("type") == "main"
        and int(item.get("player", -1)) == seat
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: int(item.get("id", 0)))


def _cell_at(
    state: Mapping[str, Any],
    position: Sequence[int] | None,
) -> Mapping[str, Any] | None:
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 2
    ):
        return None
    return state.get("cells", {}).get(
        f"{int(position[0])},{int(position[1])}"
    )


def _adjacent_enemy_pressure(
    state: Mapping[str, Any],
    seat: int,
) -> int | None:
    main = _main_general(state, seat)
    if main is None:
        return None
    position = main.get("position")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 2
    ):
        return None
    row, column = int(position[0]), int(position[1])
    cells = state.get("cells", {})
    if not isinstance(cells, Mapping):
        return None
    return sum(
        max(int(cell.get("army", 0)) - 1, 0)
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1))
        for cell in (
            cells.get(f"{row + delta_row},{column + delta_column}"),
        )
        if isinstance(cell, Mapping)
        and int(cell.get("player", -1)) == 1 - seat
        and int(cell.get("type", 2)) != 2
    )


def _state_features(
    decision: DecisionRecord,
) -> dict[str, Any]:
    state = decision.state
    seat = decision.seat
    opponent = 1 - seat
    target_main = _main_general(state, seat)
    opponent_main = _main_general(state, opponent)
    target_main_cell = _cell_at(
        state,
        target_main.get("position") if target_main else None,
    )
    opponent_main_cell = _cell_at(
        state,
        opponent_main.get("position") if opponent_main else None,
    )
    cell_items = [
        (str(key), item)
        for key, item in state.get("cells", {}).items()
        if isinstance(item, Mapping)
        and int(item.get("type", 2)) != 2
    ]
    cells = [item for _, item in cell_items]
    target_cells = [
        item for item in cells
        if int(item.get("player", -1)) == seat
    ]
    opponent_cells = [
        item for item in cells
        if int(item.get("player", -1)) == opponent
    ]
    target_army = sum(int(item.get("army", 0)) for item in target_cells)
    opponent_army = sum(int(item.get("army", 0)) for item in opponent_cells)
    coins = state.get("coins")
    coin_share = None
    if (
        isinstance(coins, Sequence)
        and not isinstance(coins, (str, bytes))
        and len(coins) > max(seat, opponent)
    ):
        target_coins = int(coins[seat])
        opponent_coins = int(coins[opponent])
        total_coins = target_coins + opponent_coins
        coin_share = target_coins / total_coins if total_coins else None
    own_generals: dict[str, int] = {}
    for general in state.get("generals", {}).values():
        if (
            isinstance(general, Mapping)
            and int(general.get("player", -1)) == seat
        ):
            kind = str(general.get("type", "unknown"))
            own_generals[kind] = own_generals.get(kind, 0) + 1
    movable_records = sorted(
        (
            {
                "position": [
                    int(value)
                    for value in key.split(",", 1)
                ],
                "army": int(item.get("army", 0)),
                "movable_army": max(
                    int(item.get("army", 0)) - 1,
                    0,
                ),
                "terrain_type": int(item.get("type", 0)),
                "general_id": item.get("general_id"),
            }
            for key, item in cell_items
            if int(item.get("player", -1)) == seat
            and int(item.get("army", 0)) > 1
        ),
        key=lambda item: (
            -int(item["movable_army"]),
            item["position"],
        ),
    )
    movable = [
        int(item["movable_army"])
        for item in movable_records
    ]
    visible_generals = []
    for general in sorted(
        (
            item
            for item in state.get("generals", {}).values()
            if isinstance(item, Mapping)
        ),
        key=lambda item: int(item.get("id", 0)),
    ):
        position = general.get("position")
        cell = _cell_at(state, position)
        visible_generals.append({
            "id": int(general.get("id", 0)),
            "player": int(general.get("player", -1)),
            "type": str(general.get("type", "unknown")),
            "position": (
                [int(position[0]), int(position[1])]
                if isinstance(position, Sequence)
                and not isinstance(position, (str, bytes))
                and len(position) == 2
                else None
            ),
            "cell_army": (
                int(cell.get("army", 0))
                if cell is not None
                else None
            ),
            "produce_level": int(
                general.get("produce_level", 0)
            ),
            "defense_level": int(
                general.get("defense_level", 0)
            ),
            "mobility_level": int(
                general.get("mobility_level", 0)
            ),
        })
    adjacent_pressure = _adjacent_enemy_pressure(state, seat)
    return {
        "own_main_position": (
            list(target_main.get("position"))
            if target_main is not None
            and isinstance(target_main.get("position"), Sequence)
            else None
        ),
        "own_main_army": (
            int(target_main_cell.get("army", 0))
            if target_main_cell is not None
            else None
        ),
        "enemy_main_position": (
            list(opponent_main.get("position"))
            if opponent_main is not None
            and isinstance(opponent_main.get("position"), Sequence)
            else None
        ),
        "enemy_main_army": (
            int(opponent_main_cell.get("army", 0))
            if opponent_main_cell is not None
            else None
        ),
        "adjacent_enemy_pressure": adjacent_pressure,
        "one_step_reachable_enemy_pressure": adjacent_pressure,
        "owned_territory": len(target_cells),
        "enemy_territory": len(opponent_cells),
        "owned_army": target_army,
        "enemy_army": opponent_army,
        "army_margin": target_army - opponent_army,
        "coin_share": coin_share,
        "owned_general_counts": dict(sorted(own_generals.items())),
        "movable_stack_count": len(movable),
        "largest_movable_stack": max(movable, default=0),
        "largest_movable_stacks": movable_records[:8],
        "visible_generals": visible_generals,
    }


def _is_non_end(action: Sequence[Sequence[int]]) -> bool:
    try:
        canonical = tuple(
            tuple(int(value) for value in command)
            for command in action
        )
    except (TypeError, ValueError):
        return False
    return canonical != ((8,),)


def _has_strategic_opportunity(
    decision: DecisionRecord,
) -> bool:
    state = decision.state
    seat = decision.seat
    for general in state.get("generals", {}).values():
        if (
            isinstance(general, Mapping)
            and int(general.get("player", -1)) != seat
            and general.get("type") in {"resource", "sub"}
        ):
            return True
    coins = state.get("coins")
    own_coins = (
        int(coins[seat])
        if isinstance(coins, Sequence)
        and not isinstance(coins, (str, bytes))
        and len(coins) > seat
        else 0
    )
    return own_coins >= 20 and any(
        isinstance(general, Mapping)
        and int(general.get("player", -1)) == seat
        and int(general.get("produce_level", 0)) < 4
        for general in state.get("generals", {}).values()
    )


def _summary_payload(
    dense_summary: DenseEpisodeSummary,
    dense_trace: Sequence[DenseStateSample],
) -> dict[str, Any]:
    return {
        "terminal_round": dense_summary.terminal_round,
        "completed_rounds_survived": (
            dense_summary.completed_rounds_survived
        ),
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


def build_critical_learning_evidence(
    replay: LearningReplay,
    dense_summary: DenseEpisodeSummary,
    dense_trace: Sequence[DenseStateSample],
) -> tuple[CriticalLearningEvidence, CriticalWindowSelection]:
    """Select deterministic high-signal v4 decision windows."""
    decisions = replay.decisions
    reasons_by_index: dict[int, list[str]] = {}

    def add(index: int | None, reason: str) -> None:
        if index is None or not 0 <= index < len(decisions):
            return
        reasons_by_index.setdefault(index, []).append(reason)

    if decisions:
        add(0, "first_decision")
        add(
            next(
                (
                    index
                    for index, item in enumerate(decisions)
                    if _is_non_end(item.action)
                ),
                None,
            ),
            "first_non_end_action",
        )
        add(
            next(
                (
                    index
                    for index, item in enumerate(decisions)
                    if (_adjacent_enemy_pressure(item.state, item.seat) or 0)
                    > 0
                ),
                None,
            ),
            "first_main_pressure",
        )
        samples = [
            dense_sample(
                item.state,
                item.seat,
                index,
                "initial",
            )
            for index, item in enumerate(decisions)
        ]
        territory_drops = [
            (
                current.target_territory - previous.target_territory,
                index,
            )
            for index, (previous, current) in enumerate(
                zip(samples, samples[1:])
            )
        ]
        if territory_drops:
            drop, index = min(territory_drops)
            if drop < 0:
                add(index, "before_steepest_territory_drop")
        army_drops = [
            (
                current.target_army - previous.target_army,
                index,
            )
            for index, (previous, current) in enumerate(
                zip(samples, samples[1:])
            )
        ]
        if army_drops:
            drop, index = min(army_drops)
            if drop < 0:
                add(index, "before_steepest_army_drop")
        add(
            next(
                (
                    index
                    for index, item in enumerate(decisions)
                    if _has_strategic_opportunity(item)
                ),
                None,
            ),
            "first_strategic_opportunity",
        )
        add(len(decisions) - 2, "penultimate_decision")
        add(len(decisions) - 1, "final_decision")

    selected_indices = tuple(sorted(reasons_by_index))
    selected = tuple(
        CriticalDecision(
            state_id=decisions[index].state_id,
            round_number=decisions[index].round_number,
            seat=decisions[index].seat,
            selection_reasons=tuple(reasons_by_index[index]),
            action=decisions[index].action,
            decision_class=classify_macro_action(
                decisions[index].state,
                decisions[index].seat,
                decisions[index].action,
            ),
            features=_state_features(decisions[index]),
        )
        for index in selected_indices
    )
    reasons = {
        decisions[index].state_id: tuple(reasons_by_index[index])
        for index in selected_indices
    }
    omitted = len(decisions) - len(selected)
    selection = CriticalWindowSelection(
        replay_id=replay.replay_id,
        total_decision_count=len(decisions),
        selected_state_ids=tuple(item.state_id for item in selected),
        reasons=reasons,
        omitted_decision_count=omitted,
    )
    evidence = CriticalLearningEvidence(
        replay_id=replay.replay_id,
        seed=replay.seed,
        evaluated_seat=replay.evaluated_seat,
        opponent_tier=replay.opponent_tier,
        termination_type=replay.termination_type,
        outcome=dense_summary.outcome,
        dense=_summary_payload(dense_summary, dense_trace),
        total_decision_count=len(decisions),
        omitted_decision_count=omitted,
        decisions=selected,
    )
    return evidence, selection
