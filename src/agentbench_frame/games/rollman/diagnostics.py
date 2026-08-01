"""Primitive Rollman hindsight diagnostics and empirical Ghost ordering."""

from __future__ import annotations

import copy
import dataclasses
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class ActionDiagnostic:
    action: int
    resulting_position: tuple[int, int]
    rollman_score_delta: int
    stayed_in_place: bool
    minimum_ghost_distance: int


def primitive_counterfactuals(
    *,
    state: Mapping[str, Any],
    ghosts_action: tuple[int, int, int],
    tracker_factory: Callable[[], Any],
) -> tuple[ActionDiagnostic, ...]:
    """Replay-fixed one-step outcomes for the exact primitive support 0..4."""

    pacman = state.get("pacman_coord")
    score = state.get("score")
    if not isinstance(pacman, (list, tuple)) or len(pacman) != 2:
        raise ValueError("counterfactual state lacks pacman_coord")
    if not isinstance(score, (list, tuple)) or len(score) != 2:
        raise ValueError("counterfactual state lacks score")
    before_position = (int(pacman[0]), int(pacman[1]))
    before_score = int(score[0])
    values: list[ActionDiagnostic] = []
    for action in range(5):
        tracker = tracker_factory()
        tracker.reset(copy.deepcopy(dict(state)))
        tracker.step(action, tuple(int(value) for value in ghosts_action))
        after = tracker.state()
        position_value = after.get("pacman_coord")
        score_value = after.get("score")
        ghosts_value = after.get("ghosts_coord")
        if not isinstance(position_value, (list, tuple)) or len(position_value) != 2:
            raise ValueError("counterfactual result lacks pacman_coord")
        if not isinstance(score_value, (list, tuple)) or len(score_value) != 2:
            raise ValueError("counterfactual result lacks score")
        if not isinstance(ghosts_value, (list, tuple)) or len(ghosts_value) != 3:
            raise ValueError("counterfactual result lacks three ghosts")
        position = (int(position_value[0]), int(position_value[1]))
        distances = [
            abs(position[0] - int(ghost[0])) + abs(position[1] - int(ghost[1]))
            for ghost in ghosts_value
        ]
        values.append(
            ActionDiagnostic(
                action=action,
                resulting_position=position,
                rollman_score_delta=int(score_value[0]) - before_score,
                stayed_in_place=position == before_position,
                minimum_ghost_distance=min(distances),
            )
        )
    return tuple(values)


def calibrate_opponent_difficulty(
    matches: Iterable[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return hardest-to-easiest valid Ghost order against one frozen panel."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for match in matches:
        if (
            match.get("status") == "complete"
            and match.get("result") in {"win", "draw", "loss"}
            and isinstance(match.get("rollman_score"), (int, float))
            and isinstance(match.get("ghosts_score"), (int, float))
            and match.get("opponent")
        ):
            grouped[str(match["opponent"])].append(match)

    def key(opponent: str) -> tuple[float, float, float, str]:
        values = grouped[opponent]
        ghost_points = [
            1.0 if value["result"] == "loss" else 0.5 if value["result"] == "draw" else 0.0
            for value in values
        ]
        margins = [
            float(value["ghosts_score"]) - float(value["rollman_score"])
            for value in values
        ]
        return (
            -(sum(ghost_points) / len(ghost_points)),
            -(sum(margins) / len(margins)),
            -min(margins),
            opponent,
        )

    return tuple(sorted(grouped, key=key))
