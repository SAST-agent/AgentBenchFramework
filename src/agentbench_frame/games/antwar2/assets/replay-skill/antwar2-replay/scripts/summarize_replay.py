#!/usr/bin/env python3
"""Produce a bounded AntWar2 match summary and atomic event index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


OPERATIONS = {
    11: "BUILD_TOWER",
    12: "UPGRADE_TOWER",
    13: "DOWNGRADE_TOWER",
    21: "USE_LIGHTNING_STORM",
    22: "USE_EMP_BLASTER",
    23: "USE_DEFLECTOR",
    24: "USE_EMERGENCY_EVASION",
    31: "UPGRADE_GENERATION_SPEED",
    32: "UPGRADE_GENERATED_ANT",
}
TARGETED = frozenset({11, 21, 22, 23, 24})
TOWER_ID = frozenset({12, 13})


def _atomic(operation: dict[str, Any]) -> dict[str, Any]:
    code = operation.get("type")
    if isinstance(code, bool) or not isinstance(code, int) or code not in OPERATIONS:
        raise ValueError(f"unknown operation code: {code}")
    atom: dict[str, Any] = {"code": code, "name": OPERATIONS[code]}
    if code in TARGETED:
        pos = operation.get("pos")
        if not isinstance(pos, dict):
            raise ValueError(f"operation {code} requires pos")
        atom["arguments"] = [int(pos["x"]), int(pos["y"])]
    elif code == 12:
        atom["arguments"] = [int(operation["id"]), int(operation["args"])]
    elif code in TOWER_ID:
        atom["arguments"] = [int(operation["id"])]
    else:
        atom["arguments"] = []
    return atom


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("replay is empty")
    counters = [Counter(), Counter()]
    breaches = [0, 0]
    events: list[dict[str, Any]] = []
    first_state = records[0].get("round_state")
    if not isinstance(first_state, dict):
        raise ValueError("round 0 lacks round_state")
    previous_camps = first_state.get("camps")
    if not isinstance(previous_camps, list) or len(previous_camps) != 2:
        raise ValueError("round 0 camps must contain two values")
    active_towers: dict[int, dict[str, Any]] = {}

    for round_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"round {round_index} must be an object")
        state = record.get("round_state")
        if not isinstance(state, dict):
            raise ValueError(f"round {round_index} lacks round_state")
        for delta in state.get("towers", []):
            tower_id = int(delta["id"])
            if int(delta.get("type", -1)) == -1:
                active_towers.pop(tower_id, None)
            else:
                active_towers[tower_id] = dict(delta)
        for player in (0, 1):
            operations = record.get(f"op{player}") or []
            if not isinstance(operations, list):
                raise ValueError(f"round {round_index} op{player} must be a list")
            if not operations:
                counters[player]["HOLD"] += 1
            for position, operation in enumerate(operations):
                if not isinstance(operation, dict):
                    raise ValueError("operation must be an object")
                atom = _atomic(operation)
                counters[player][atom["name"]] += 1
                events.append(
                    {
                        "round": round_index,
                        "player": player,
                        "kind": "accepted_operation",
                        "sequence_index": position,
                        "operation": atom,
                    }
                )
        camps = state.get("camps")
        if not isinstance(camps, list) or len(camps) != 2:
            raise ValueError(f"round {round_index} camps must contain two values")
        for defender in (0, 1):
            damage = int(previous_camps[defender]) - int(camps[defender])
            if damage > 0:
                attacker = 1 - defender
                breaches[attacker] += damage
                events.append(
                    {
                        "round": round_index,
                        "player": attacker,
                        "kind": "breach",
                        "defender": defender,
                        "camp_damage": damage,
                        "camp_hp_after": int(camps[defender]),
                    }
                )
        previous_camps = camps

    terminal = records[-1]["round_state"]
    winner = terminal.get("winner", -1)
    if isinstance(winner, bool) or not isinstance(winner, int) or winner not in {-1, 0, 1}:
        raise ValueError("terminal winner must be -1, 0, or 1")
    return {
        "schema_version": "1.0",
        "seed": records[0].get("seed"),
        "rounds": len(records),
        "winner": winner,
        "terminal_camps": [int(value) for value in previous_camps],
        "terminal_coins": [int(value) for value in terminal.get("coins", [0, 0])],
        "active_tower_count": len(active_towers),
        "players": [
            {
                "role": f"P{player}",
                "accepted_operation_counts": dict(
                    sorted(counters[player].items())
                ),
                "breaches": breaches[player],
            }
            for player in (0, 1)
        ],
        "event_index": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.replay.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("replay must be a JSON array")
    value = summarize(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value[key] for key in ("rounds", "winner", "terminal_camps")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
