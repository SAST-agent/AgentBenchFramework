#!/usr/bin/env python3
"""Extract a size-capped AntWar2 public-state and operation round window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAX_OUTPUT_BYTES = 64 * 1024
STATE_FIELDS = (
    "round",
    "coins",
    "camps",
    "speed_levels",
    "ant_hp_levels",
    "weapon_cooldowns",
    "active_effects",
    "towers",
    "ants",
)


def _compact(record: dict[str, Any]) -> dict[str, Any] | None:
    round_index = record.get("round")
    if isinstance(round_index, bool) or not isinstance(round_index, int):
        return None
    kind = record.get("kind")
    if kind == "public_state":
        state = record.get("public_state")
        if not isinstance(state, dict):
            return None
        return {
            "round": round_index,
            "kind": kind,
            "public_state": {
                field: state[field]
                for field in STATE_FIELDS
                if field in state
            },
        }
    if kind == "ai_operations":
        operations = record.get("operations")
        if not isinstance(operations, list):
            raise ValueError("ai_operations event requires an operation list")
        return {
            "round": round_index,
            "kind": kind,
            "player": int(record["player"]),
            "operations": operations,
        }
    return None


def inspect(path: Path, *, round_index: int, radius: int) -> dict[str, Any]:
    if round_index < 0:
        raise ValueError("round must be non-negative")
    if not 0 <= radius <= 3:
        raise ValueError("radius must be in [0, 3]")
    selected = set(
        range(max(0, round_index - radius), round_index + radius + 1)
    )
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"trace line {line_number} is invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"trace line {line_number} must be an object")
            if record.get("round") not in selected:
                continue
            compact = _compact(record)
            if compact is not None:
                events.append(compact)
    if not events:
        raise ValueError("requested trace window contains no public events")
    return {
        "schema_version": "1.0",
        "requested_round": round_index,
        "radius": radius,
        "rounds": sorted({int(event["round"]) for event in events}),
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = inspect(args.trace, round_index=args.round, radius=args.radius)
    rendered = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    if len(rendered.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError("trace window exceeds 64 KiB; reduce radius")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "requested_round": value["requested_round"],
                "rounds": value["rounds"],
                "event_count": len(value["events"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
