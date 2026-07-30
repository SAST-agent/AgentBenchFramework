#!/usr/bin/env python3
"""Validate and summarize a frozen Rollman JSONL replay."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EVENTS = {
    0: "EATEN_BY_GHOST",
    1: "SHIELD_DESTROYED",
    2: "FINISH_LEVEL",
    3: "TIMEOUT",
}
INIT_FIELDS = {
    "level",
    "round",
    "board_size",
    "board",
    "pacman_skill_status",
    "pacman_coord",
    "ghosts_coord",
    "score",
    "beannumber",
    "portal_available",
    "portal_coord",
}
ROUND_FIELDS = {
    "round",
    "level",
    "pacman_step_block",
    "pacman_coord",
    "pacman_skills",
    "ghosts_step_block",
    "ghosts_coord",
    "score",
    "events",
    "portal_available",
    "StopReason",
}


class ReplayError(ValueError):
    pass


def _objects(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ReplayError(f"line {line_number}: replay record must be an object")
            records.append(value)
    if not records:
        raise ReplayError("replay is empty")
    return records


def _require(record: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(record))
    if missing:
        raise ReplayError(f"{label}: missing fields {missing}")


def _coord(value: Any, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, (int, float)) for item in value)
    ):
        raise ReplayError(f"{label}: expected [x, y] coordinate")
    return [int(value[0]), int(value[1])]


def _normalize_path(value: Any, label: str) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ReplayError(f"{label}: path must be a non-empty list")
    normalized = []
    last_valid = None
    for index, raw in enumerate(value):
        coordinate = _coord(raw, f"{label}[{index}]")
        if coordinate[0] < 0 or coordinate[1] < 0:
            if last_valid is None:
                raise ReplayError(f"{label}[{index}]: wall sentinel before valid coordinate")
            normalized.append(list(last_valid))
        else:
            last_valid = coordinate
            normalized.append(coordinate)
    return normalized


def summarize(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    records = _objects(source)
    initializations = []
    rounds = []
    terminals = []
    for index, record in enumerate(records, start=1):
        if "board" in record:
            _require(record, INIT_FIELDS, f"line {index} initialization")
            initializations.append(record)
        elif record.get("StopReason") is not None:
            _require(record, ROUND_FIELDS, f"line {index} terminal")
            terminals.append(record)
        else:
            _require(record, ROUND_FIELDS, f"line {index} round")
            rounds.append(record)
    if not initializations:
        raise ReplayError("replay has no level initialization")
    if len(terminals) != 1:
        raise ReplayError(f"replay must have exactly one terminal frame, found {len(terminals)}")

    levels = [int(record["level"]) for record in initializations]
    event_counts: Counter[str] = Counter()
    evidence = []
    score_deltas = []
    initial_score_by_level = {
        int(record["level"]): list(record["score"]) for record in initializations
    }
    previous_score_by_level = dict(initial_score_by_level)
    previous_round_by_level = {level: 0 for level in levels}
    round_gaps = []
    normalized_rounds = []
    for record in rounds:
        level = int(record["level"])
        round_number = int(record["round"])
        if level not in previous_round_by_level:
            raise ReplayError(
                f"level {level} round {round_number}: no level initialization"
            )
        previous_round = previous_round_by_level[level]
        if round_number <= previous_round:
            raise ReplayError(
                f"level {level} round {round_number}: rounds must be strictly increasing"
            )
        if round_number > previous_round + 1:
            round_gaps.append(
                {
                    "level": level,
                    "after_round": previous_round,
                    "before_round": round_number,
                    "missing_count": round_number - previous_round - 1,
                }
            )
        previous_round_by_level[level] = round_number
        event_names = []
        for raw_event in record["events"]:
            if raw_event not in EVENTS:
                raise ReplayError(
                    f"level {level} round {round_number}: unknown event {raw_event}"
                )
            name = EVENTS[raw_event]
            event_counts[name] += 1
            event_names.append(name)
        if event_names:
            evidence.append(
                {
                    "location": f"level {level} round {round_number}",
                    "events": event_names,
                    "score": list(record["score"]),
                    "pacman_coord": _coord(record["pacman_coord"], "pacman_coord"),
                    "portal_available": bool(record["portal_available"]),
                }
            )
        previous = previous_score_by_level.get(level, list(record["score"]))
        current = list(record["score"])
        if len(current) != 2 or len(previous) != 2:
            raise ReplayError(f"level {level} round {round_number}: score must have two values")
        score_deltas.append(
            {
                "location": f"level {level} round {round_number}",
                "rollman": current[0] - previous[0],
                "ghosts": current[1] - previous[1],
            }
        )
        previous_score_by_level[level] = current
        normalized_rounds.append(
            {
                "level": level,
                "round": round_number,
                "pacman_path": _normalize_path(
                    record["pacman_step_block"], "pacman_step_block"
                ),
                "ghost_paths": [
                    _normalize_path(path_value, f"ghosts_step_block[{ghost_index}]")
                    for ghost_index, path_value in enumerate(record["ghosts_step_block"])
                ],
                "events": event_names,
                "score": current,
                "portal_available": bool(record["portal_available"]),
            }
        )
    terminal = terminals[0]
    return {
        "schema_version": "1.0",
        "valid": True,
        "replay": str(source.resolve()),
        "levels": levels,
        "initialization_frames": len(initializations),
        "round_frames": len(rounds),
        "final_score": list(terminal["score"]),
        "stop_reason": str(terminal["StopReason"]),
        "event_counts": dict(sorted(event_counts.items())),
        "round_gaps": round_gaps,
        "evidence": evidence,
        "score_deltas": score_deltas,
        "rounds": normalized_rounds,
    }


def to_markdown(summary: dict[str, Any]) -> str:
    score = summary["final_score"]
    lines = [
        "# Rollman replay evidence",
        "",
        f"- Levels: {', '.join(str(value) for value in summary['levels'])}",
        f"- Round frames: {summary['round_frames']}",
        f"- Final score: Rollman {score[0]}, Ghosts {score[1]}",
        f"- Stop reason: {summary['stop_reason']}",
        "",
        "## Round coverage",
        "",
    ]
    if not summary["round_gaps"]:
        lines.append("- No missing ordinary rounds detected.")
    for gap in summary["round_gaps"]:
        first_missing = gap["after_round"] + 1
        last_missing = gap["before_round"] - 1
        lines.append(
            f"- WARNING: level {gap['level']} is missing rounds "
            f"{first_missing}-{last_missing}; behavior and causality in this interval are unknown."
        )
    lines.extend(
        [
        "",
        "## Event evidence",
        "",
        ]
    )
    if not summary["evidence"]:
        lines.append("- No event-coded failure or transition.")
    for item in summary["evidence"]:
        lines.append(
            f"- {item['location']}: {', '.join(item['events'])}; "
            f"score={item['score']}; pacman={item['pacman_coord']}; "
            f"portal_available={str(item['portal_available']).lower()}"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Facts above come directly from replay fields.",
            "- The replay does not expose hidden goals or high-level action intent.",
            "- Diagnose a strategy only after checking the corresponding path and visible state.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()
    try:
        result = summarize(args.replay)
    except (OSError, ReplayError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(to_markdown(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
