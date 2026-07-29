"""Read-only diagnostics for evaluated Generals macro actions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .evaluator import GeneralsEvaluation


_MOVE_DESTINATIONS = frozenset(
    {"owned", "enemy", "neutral_general", "neutral_plain"}
)
_DIRECTION_DELTAS = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


@dataclass(frozen=True)
class ActionProfile:
    turn_count: int
    primitive_command_count: int
    mean_primitives_per_turn: float
    max_primitives_per_turn: int
    multi_command_turn_count: int
    command_counts: Mapping[int, int]
    end_only_turn_count: int
    general_upgrade_count: int
    technology_upgrade_count: int
    move_destination_counts: Mapping[str, int]
    neutral_plain_move_ratio: float | None


def _frozen_sorted_counts(counts: Mapping[Any, int]) -> Mapping[Any, int]:
    return MappingProxyType(dict(sorted(counts.items())))


def _is_end_command(command: Sequence[int]) -> bool:
    return tuple(command) == (8,)


def _movement_destination(
    state: Mapping[str, Any],
    seat: int,
    command: Sequence[int],
) -> str:
    if len(command) != 5:
        return "unknown"
    try:
        row, column, direction = (int(command[index]) for index in (1, 2, 3))
        row_delta, column_delta = _DIRECTION_DELTAS[direction]
    except (KeyError, TypeError, ValueError):
        return "unknown"
    cells = state.get("cells")
    if not isinstance(cells, Mapping):
        return "unknown"
    destination = cells.get(f"{row + row_delta},{column + column_delta}")
    if not isinstance(destination, Mapping):
        return "unknown"
    try:
        owner = int(destination["player"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    if owner == seat:
        return "owned"
    if owner != -1:
        return "enemy"
    general_id = destination.get("general_id")
    try:
        has_general = general_id is not None and int(general_id) >= 0
    except (TypeError, ValueError):
        return "unknown"
    return "neutral_general" if has_general else "neutral_plain"


def summarize_action_profile(evaluation: GeneralsEvaluation) -> ActionProfile:
    """Summarize only turns played by each match's evaluated seat."""
    command_counts: Counter[int] = Counter()
    destination_counts: Counter[str] = Counter()
    primitives_per_turn: list[int] = []
    end_only_turn_count = 0
    general_upgrade_count = 0
    technology_upgrade_count = 0

    for match in evaluation.matches:
        for turn in match.turns:
            if turn.player != match.evaluated_seat:
                continue
            commands = turn.commands
            if len(commands) == 1 and _is_end_command(commands[0]):
                end_only_turn_count += 1
            primitive_count = 0
            for command in commands:
                if _is_end_command(command):
                    continue
                primitive_count += 1
                if not command:
                    continue
                opcode = command[0]
                command_counts[opcode] += 1
                if opcode == 3:
                    general_upgrade_count += 1
                elif opcode == 5:
                    technology_upgrade_count += 1
                elif opcode == 1:
                    destination_counts[_movement_destination(
                        turn.state_before, match.evaluated_seat, command
                    )] += 1
            primitives_per_turn.append(primitive_count)

    turn_count = len(primitives_per_turn)
    classifiable_moves = sum(
        count
        for destination, count in destination_counts.items()
        if destination in _MOVE_DESTINATIONS
    )
    neutral_plain_move_ratio = (
        destination_counts["neutral_plain"] / classifiable_moves
        if classifiable_moves
        else None
    )
    primitive_command_count = sum(primitives_per_turn)
    return ActionProfile(
        turn_count=turn_count,
        primitive_command_count=primitive_command_count,
        mean_primitives_per_turn=(
            primitive_command_count / turn_count if turn_count else 0.0
        ),
        max_primitives_per_turn=max(primitives_per_turn, default=0),
        multi_command_turn_count=sum(count > 1 for count in primitives_per_turn),
        command_counts=_frozen_sorted_counts(command_counts),
        end_only_turn_count=end_only_turn_count,
        general_upgrade_count=general_upgrade_count,
        technology_upgrade_count=technology_upgrade_count,
        move_destination_counts=_frozen_sorted_counts(destination_counts),
        neutral_plain_move_ratio=neutral_plain_move_ratio,
    )


def action_profile_payload(profile: ActionProfile) -> dict[str, object]:
    """Return the stable JSON-ready representation used by downstream prompts."""
    return {
        "turn_count": profile.turn_count,
        "primitive_command_count": profile.primitive_command_count,
        "mean_primitives_per_turn": profile.mean_primitives_per_turn,
        "max_primitives_per_turn": profile.max_primitives_per_turn,
        "multi_command_turn_count": profile.multi_command_turn_count,
        "command_counts": dict(profile.command_counts),
        "end_only_turn_count": profile.end_only_turn_count,
        "general_upgrade_count": profile.general_upgrade_count,
        "technology_upgrade_count": profile.technology_upgrade_count,
        "move_destination_counts": dict(profile.move_destination_counts),
        "neutral_plain_move_ratio": profile.neutral_plain_move_ratio,
    }
