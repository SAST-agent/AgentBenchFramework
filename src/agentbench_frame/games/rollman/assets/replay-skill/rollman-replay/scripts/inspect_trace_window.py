#!/usr/bin/env python3
"""Extract bounded Rollman decision/outcome windows from a match trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ACTIONS = {
    0: "STAY",
    1: "UP",
    2: "LEFT",
    3: "DOWN",
    4: "RIGHT",
}
EVENTS = {
    0: "EATEN_BY_GHOST",
    1: "SHIELD_DESTROYED",
    2: "FINISH_LEVEL",
    3: "TIMEOUT",
}
STATE_FIELDS = (
    "level",
    "round",
    "board_size",
    "pacman_coord",
    "ghosts_coord",
    "pacman_skill_status",
    "score",
    "beannumber",
    "portal_available",
    "portal_coord",
)
OUTCOME_FIELDS = (
    "level",
    "round",
    "pacman_step_block",
    "pacman_coord",
    "pacman_skills",
    "ghosts_step_block",
    "ghosts_coord",
    "score",
    "portal_available",
    "StopReason",
)
MAX_REQUESTED_ROUNDS = 20
MAX_EXPANDED_ROUNDS = 40
MAX_OUTPUT_BYTES = 64 * 1024


class TraceError(ValueError):
    pass


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise TraceError(
                    f"line {line_number}: trace record must be an object"
                )
            records.append(value)
    if not records:
        raise TraceError("trace is empty")
    return records


def _event_names(
    values: Any,
    *,
    level: int,
    round_number: int,
) -> list[str]:
    if not isinstance(values, list):
        raise TraceError(
            f"level {level} round {round_number}: events must be a list"
        )
    names = []
    for value in values:
        if value not in EVENTS:
            raise TraceError(
                f"level {level} round {round_number}: unknown event {value}"
            )
        names.append(EVENTS[value])
    return names


def inspect(
    path: str | Path,
    *,
    level: int,
    rounds: list[int],
    radius: int,
) -> dict[str, Any]:
    if len(rounds) > MAX_REQUESTED_ROUNDS:
        raise TraceError(
            f"request at most {MAX_REQUESTED_ROUNDS} center rounds"
        )
    if not rounds:
        raise TraceError("request at least one center round")
    if level < 1 or any(round_number < 0 for round_number in rounds):
        raise TraceError("level must be >= 1 and rounds must be >= 0")
    if not 0 <= radius <= 2:
        raise TraceError("radius must be in [0, 2]")
    selected_rounds = sorted(
        {
            value
            for center in rounds
            for value in range(
                max(0, center - radius),
                center + radius + 1,
            )
        }
    )
    if len(selected_rounds) > MAX_EXPANDED_ROUNDS:
        raise TraceError(
            f"expanded request exceeds {MAX_EXPANDED_ROUNDS} rounds"
        )

    records = _load_records(Path(path))
    outcomes: dict[tuple[int, int], dict[str, Any]] = {}
    requested_outcome_rounds = {
        round_number + 1 for round_number in selected_rounds
    }
    for record in records:
        if record.get("type") != "watch":
            continue
        content = record.get("content")
        if not isinstance(content, dict):
            continue
        if "level" not in content or "round" not in content:
            continue
        key = (int(content["level"]), int(content["round"]))
        if key[0] != level or key[1] not in requested_outcome_rounds:
            continue
        if key in outcomes:
            raise TraceError(
                f"level {key[0]} round {key[1]}: duplicate watch frame"
            )
        outcome = {
            field: content[field]
            for field in OUTCOME_FIELDS
            if field in content
        }
        outcome["events"] = _event_names(
            content.get("events", []),
            level=key[0],
            round_number=key[1],
        )
        outcomes[key] = outcome

    decisions = []
    seen: set[tuple[int, int]] = set()
    requested = set(selected_rounds)
    for record in records:
        if record.get("type") != "action" or record.get("player") != 0:
            continue
        decision = record.get("decision")
        if not isinstance(decision, dict):
            continue
        state = decision.get("state")
        if not isinstance(state, dict):
            continue
        decision_level = int(state.get("level", -1))
        decision_round = int(state.get("round", -1))
        if decision_level != level or decision_round not in requested:
            continue
        key = (decision_level, decision_round)
        if key in seen:
            raise TraceError(
                f"level {decision_level} round {decision_round}: "
                "duplicate Rollman decision"
            )
        seen.add(key)
        action = decision.get("action", record.get("action"))
        if action not in ACTIONS:
            raise TraceError(
                f"level {decision_level} round {decision_round}: "
                f"unknown action {action}"
            )
        pre_state = {
            field: state[field]
            for field in STATE_FIELDS
            if field in state
        }
        decisions.append(
            {
                "level": decision_level,
                "round": decision_round,
                "action": {
                    "code": int(action),
                    "name": ACTIONS[int(action)],
                },
                "memory_id": decision.get("memory_id"),
                "pre_state": pre_state,
                "outcome": outcomes.get(
                    (decision_level, decision_round + 1)
                ),
            }
        )
    decisions.sort(key=lambda item: (item["level"], item["round"]))
    if not decisions:
        raise TraceError("no Rollman decisions matched the requested window")
    return {
        "schema_version": "1.0",
        "level": level,
        "requested_rounds": rounds,
        "radius": radius,
        "decision_count": len(decisions),
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument(
        "--round",
        dest="rounds",
        type=int,
        action="append",
        required=True,
    )
    parser.add_argument("--radius", type=int, default=1)
    args = parser.parse_args()
    try:
        result = inspect(
            args.trace,
            level=args.level,
            rounds=args.rounds,
            radius=args.radius,
        )
        output = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n"
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise TraceError(
                f"trace window exceeds {MAX_OUTPUT_BYTES} bytes"
            )
    except (OSError, TraceError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
