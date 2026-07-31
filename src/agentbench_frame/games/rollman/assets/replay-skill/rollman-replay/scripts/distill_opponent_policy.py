#!/usr/bin/env python3
"""Distill bounded, coordinate-free Ghost action statistics from HL traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = {0: "STAY", 1: "UP", 2: "LEFT", 3: "DOWN", 4: "RIGHT"}
DELTAS = {0: (0, 0), 1: (1, 0), 2: (0, -1), 3: (-1, 0), 4: (0, 1)}


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _distance_bucket(distance: int) -> str:
    if distance <= 1:
        return "0-1"
    if distance <= 3:
        return "2-3"
    if distance <= 7:
        return "4-7"
    return "8+"


def _legal_actions(board: list[list[Any]], position: list[int]) -> tuple[int, ...]:
    legal = [0]
    height = len(board)
    width = len(board[0]) if height else 0
    for action in (1, 2, 3, 4):
        dx, dy = DELTAS[action]
        x, y = int(position[0]) + dx, int(position[1]) + dy
        if 0 <= x < height and 0 <= y < width and board[x][y] != 0:
            legal.append(action)
    return tuple(legal)


def _records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"trace line {line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"trace line {line_number}: record must be an object"
                )
            yield value


def distill(paths: list[Path], *, max_patterns: int = 80) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one trace is required")
    action_counts: Counter[int] = Counter()
    fine: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    coarse: dict[tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    persistence_hits = chase_hits = samples = frozen_omitted = 0

    for path in paths:
        pending_state: dict[str, Any] | None = None
        last_actions: dict[int, int] = {}
        for record in _records(path):
            if record.get("type") == "action" and record.get("player") == 0:
                decision = record.get("decision")
                state = decision.get("state") if isinstance(decision, dict) else None
                pending_state = state if isinstance(state, dict) else None
                continue
            if record.get("type") != "action" or record.get("player") != 1:
                continue
            actions = record.get("action")
            state = pending_state
            pending_state = None
            if not isinstance(state, dict) or not isinstance(actions, list):
                continue
            if len(actions) != 3 or any(action not in ACTIONS for action in actions):
                raise ValueError("Ghost action must contain three codes in [0, 4]")
            skills = state.get("pacman_skill_status")
            if isinstance(skills, list) and len(skills) > 4 and int(skills[4]) > 0:
                frozen_omitted += 3
                continue
            pacman = state.get("pacman_coord")
            ghosts = state.get("ghosts_coord")
            board = state.get("board")
            if (
                not isinstance(pacman, list)
                or len(pacman) != 2
                or not isinstance(ghosts, list)
                or len(ghosts) != 3
                or not isinstance(board, list)
                or not board
            ):
                raise ValueError("decision state lacks board or role coordinates")
            for slot, (ghost, action) in enumerate(zip(ghosts, actions)):
                if not isinstance(ghost, list) or len(ghost) != 2:
                    raise ValueError("Ghost coordinate must be [x, y]")
                dx = int(pacman[0]) - int(ghost[0])
                dy = int(pacman[1]) - int(ghost[1])
                distance = abs(dx) + abs(dy)
                dominant = "x" if abs(dx) > abs(dy) else "y" if abs(dy) > abs(dx) else "tie"
                legal = _legal_actions(board, ghost)
                previous = last_actions.get(slot)
                coarse_key = (
                    _sign(dx), _sign(dy), dominant, _distance_bucket(distance)
                )
                fine_key = (
                    slot,
                    *coarse_key,
                    "none" if previous is None else str(previous),
                    ",".join(str(value) for value in legal),
                )
                coarse[coarse_key][int(action)] += 1
                fine[fine_key][int(action)] += 1
                action_counts[int(action)] += 1
                samples += 1
                persistence_hits += int(previous == action)
                step_dx, step_dy = DELTAS[int(action)]
                end = [int(ghost[0]) + step_dx, int(ghost[1]) + step_dy]
                if action not in legal:
                    end = [int(ghost[0]), int(ghost[1])]
                new_distance = abs(int(pacman[0]) - end[0]) + abs(
                    int(pacman[1]) - end[1]
                )
                chase_hits += int(new_distance < distance)
                last_actions[slot] = int(action)

    if not samples:
        raise ValueError("traces contain no usable Ghost decisions")

    def patterns(table: dict[tuple[Any, ...], Counter[int]], fields: list[str]):
        ranked = sorted(
            table.items(), key=lambda item: (-sum(item[1].values()), item[0])
        )[:max_patterns]
        result = []
        for key, counts in ranked:
            total = sum(counts.values())
            modal_action, modal_count = counts.most_common(1)[0]
            result.append(
                {
                    "context": dict(zip(fields, key)),
                    "count": total,
                    "action_counts": {
                        ACTIONS[action]: counts.get(action, 0) for action in ACTIONS
                    },
                    "modal_action": ACTIONS[modal_action],
                    "modal_confidence": modal_count / total,
                }
            )
        return result

    return {
        "schema_version": "1.0",
        "trace_count": len(paths),
        "ghost_decision_samples": samples,
        "frozen_samples_omitted": frozen_omitted,
        "action_counts": {
            ACTIONS[action]: action_counts.get(action, 0) for action in ACTIONS
        },
        "chase_step_rate": chase_hits / samples,
        "same_action_as_previous_rate": persistence_hits / samples,
        "feature_contract": {
            "relative_dx_sign": "sign(pacman_x - ghost_x)",
            "relative_dy_sign": "sign(pacman_y - ghost_y)",
            "distance_bucket": ["0-1", "2-3", "4-7", "8+"],
            "legal_actions": "protocol action codes after wall masking",
        },
        "coarse_backoff_patterns": patterns(
            coarse,
            ["relative_dx_sign", "relative_dy_sign", "dominant_axis", "distance_bucket"],
        ),
        "fine_patterns": patterns(
            fine,
            [
                "ghost_slot",
                "relative_dx_sign",
                "relative_dy_sign",
                "dominant_axis",
                "distance_bucket",
                "previous_action",
                "legal_actions",
            ],
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--max-patterns", type=int, default=80)
    args = parser.parse_args(argv)
    if not 1 <= args.max_patterns <= 100:
        parser.error("--max-patterns must be in [1, 100]")
    try:
        result = distill(
            [Path(value) for value in args.traces], max_patterns=args.max_patterns
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if len(output.encode("utf-8")) > 64 * 1024:
        print("distillation output exceeds 64 KiB", file=sys.stderr)
        return 2
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
