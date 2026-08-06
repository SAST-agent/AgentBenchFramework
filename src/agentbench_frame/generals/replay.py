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
    *,
    include_strategic_targets: bool = False,
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
    strategic_targets = []
    if include_strategic_targets:
        own_main_position = (
            target_main.get("position")
            if target_main is not None
            else None
        )
        candidates = []
        for general in state.get("generals", {}).values():
            if (
                not isinstance(general, Mapping)
                or int(general.get("player", -1)) == seat
                or general.get("type") not in {"resource", "sub"}
            ):
                continue
            position = general.get("position")
            if (
                not isinstance(position, Sequence)
                or isinstance(position, (str, bytes))
                or len(position) != 2
            ):
                continue
            normalized_position = [
                int(position[0]),
                int(position[1]),
            ]
            distance = (
                abs(normalized_position[0] - int(own_main_position[0]))
                + abs(
                    normalized_position[1]
                    - int(own_main_position[1])
                )
                if isinstance(own_main_position, Sequence)
                and not isinstance(
                    own_main_position,
                    (str, bytes),
                )
                and len(own_main_position) == 2
                else None
            )
            cell = _cell_at(state, position)
            candidates.append({
                "id": int(general.get("id", 0)),
                "player": int(general.get("player", -1)),
                "type": str(general.get("type", "unknown")),
                "position": normalized_position,
                "distance_from_own_main": distance,
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
        strategic_targets = sorted(
            candidates,
            key=lambda item: (
                item["distance_from_own_main"]
                if item["distance_from_own_main"] is not None
                else 10**9,
                item["id"],
            ),
        )[:6]
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
        "largest_movable_stacks": movable_records[:4],
        **(
            {"strategic_general_targets": strategic_targets}
            if include_strategic_targets
            else {}
        ),
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
    *,
    max_decisions: int | None = None,
    selection_reasons: Sequence[str] | None = None,
    include_strategic_targets: bool = False,
) -> tuple[CriticalLearningEvidence, CriticalWindowSelection]:
    """Select deterministic high-signal v4 decision windows."""
    decisions = replay.decisions
    reasons_by_index: dict[int, list[str]] = {}
    default_reasons = (
        "first_decision",
        "first_non_end_action",
        "first_main_pressure",
        "before_steepest_territory_drop",
        "before_steepest_army_drop",
        "first_strategic_opportunity",
        "penultimate_decision",
        "final_decision",
        "first_main_danger",
        "large_stack_inactive",
        "missed_counter_or_reinforcement",
        "economy_defense_conflict",
        "first_dense_divergence",
        "unsafe_or_low_value_macro",
    )
    requested = tuple(selection_reasons or default_reasons)
    unknown = set(requested) - set(default_reasons)
    if unknown:
        raise ValueError(
            f"unknown critical-window selection reasons: {sorted(unknown)}"
        )
    if max_decisions is not None and max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

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
        first_pressure = next(
            (
                index
                for index, item in enumerate(decisions)
                if (_adjacent_enemy_pressure(item.state, item.seat) or 0) > 0
            ),
            None,
        )
        add(first_pressure, "first_main_danger")
        add(
            next(
                (
                    index
                    for index, item in enumerate(decisions)
                    if max(
                        (
                            int(cell.get("army", 0)) - 1
                            for cell in item.state.get("cells", {}).values()
                            if isinstance(cell, Mapping)
                            and int(cell.get("player", -1)) == item.seat
                        ),
                        default=0,
                    )
                    >= 50
                    and not any(command and command[0] == 1 for command in item.action)
                ),
                None,
            ),
            "large_stack_inactive",
        )
        add(
            next(
                (
                    index
                    for index, item in enumerate(decisions)
                    if (_adjacent_enemy_pressure(item.state, item.seat) or 0) > 0
                    and not any(command and command[0] == 1 for command in item.action)
                ),
                None,
            ),
            "missed_counter_or_reinforcement",
        )
        add(
            next(
                (
                    index
                    for index, item in enumerate(decisions)
                    if (_adjacent_enemy_pressure(item.state, item.seat) or 0) > 0
                    and any(command and command[0] in {3, 5} for command in item.action)
                ),
                None,
            ),
            "economy_defense_conflict",
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
        divergence_candidates = [
            (delta, index)
            for delta, index in (*territory_drops, *army_drops)
            if delta < 0
        ]
        if divergence_candidates:
            _, divergence_index = min(
                divergence_candidates,
                key=lambda item: (item[1], item[0]),
            )
            add(divergence_index, "first_dense_divergence")
        add(
            next(
                (
                    index
                    for index, item in enumerate(decisions)
                    if len([command for command in item.action if command and command[0] != 8]) > 4
                    and (_adjacent_enemy_pressure(item.state, item.seat) or 0) > 0
                ),
                None,
            ),
            "unsafe_or_low_value_macro",
        )
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

    requested_set = set(requested)
    reasons_by_index = {
        index: [
            reason for reason in reasons
            if reason in requested_set
        ]
        for index, reasons in reasons_by_index.items()
        if any(reason in requested_set for reason in reasons)
    }
    if (
        max_decisions is not None
        and len(reasons_by_index) > max_decisions
    ):
        prioritized: list[int] = []
        for reason in requested:
            for index in sorted(reasons_by_index):
                if (
                    reason in reasons_by_index[index]
                    and index not in prioritized
                ):
                    prioritized.append(index)
                    if len(prioritized) == max_decisions:
                        break
            if len(prioritized) == max_decisions:
                break
        selected_set = set(prioritized)
        reasons_by_index = {
            index: reasons
            for index, reasons in reasons_by_index.items()
            if index in selected_set
        }

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
            features=_state_features(
                decisions[index],
                include_strategic_targets=(
                    include_strategic_targets
                    or "first_strategic_opportunity"
                    in reasons_by_index[index]
                ),
            ),
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
