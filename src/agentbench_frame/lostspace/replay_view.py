"""Text viewer for native LostSpace replays.

A native replay is the JSON array written by the game logic's ``Replay`` object
(see ``backend_sources/corpus/25_lostspace/logic/gamecode_logic/src/replay.py``)::

    [
      birthplaces,          # lines[0]: [[x,y,z], ...] x4, coords shifted by -3
      round_1,              # list of 4 turns; each turn = list of action dicts
      round_2,
      ...
      round_N,
      score_dic             # lines[-1]: {"0": 4, "1": 3, ...} (4=1st ... 1=4th)
    ]

This viewer reconstructs a per-player running state (position, hp, keys,
status) from the action stream and prints a readable turn-by-turn log. Run as::

    python -m agentbench_frame.lostspace.replay_view <replay.json> [--round N] [--player P]

Coordinate convention: replay coords are centred on 0 (range -3..3); we display
grid coords (``+3``) to match the 7x7 board.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


def _unshift(pos: list[int]) -> list[int]:
    """Replay coords are [x-3, y-3, z]; return grid coords [x, y, z]."""
    if not pos or len(pos) < 2:
        return pos
    return [pos[0] + 3, pos[1] + 3, pos[2] if len(pos) > 2 else 1]


def _load(path: str | Path) -> list:
    return json.loads(Path(path).read_text())


def _new_state(birthplaces: list[list[int]]) -> dict[int, dict[str, Any]]:
    state = {}
    for pid, bp in enumerate(birthplaces):
        gx, gy = bp[0] + 3, bp[1] + 3
        gz = bp[2] if len(bp) > 2 else 1
        state[pid] = {"pos": [gx, gy, gz], "hp": 200, "keys": set(), "status": "alive"}
    return state


_STATUS_FROM_TYPE = {
    "died": "dead",
    "escaped": "escaped",
    "regenerate": "alive",
    "ai_error": "error",
}


def _update(state: dict[int, dict[str, Any]], d: dict[str, Any]) -> None:
    """Mutate the running state from one action dict."""
    pid = d.get("playerid")
    t = d.get("type")
    if pid is None or pid not in state:
        return
    s = state[pid]
    if t in ("move", "flink", "regenerate"):
        s["pos"] = _unshift(d.get("pos", s["pos"]))
    if t in ("hp_update", "cure", "use_kit"):
        s["hp"] = d.get("hp", s["hp"])
    if t == "getkey":
        for k in d.get("keyid", []):
            s["keys"].add(k)
    if t in _STATUS_FROM_TYPE:
        s["status"] = _STATUS_FROM_TYPE[t]


def _fmt_action(d: dict[str, Any]) -> str:
    t = d.get("type", "?")
    pid = d.get("playerid", "-")
    if t in ("move", "flink"):
        return f"P{pid} {t} -> {_unshift(d.get('pos', []))}"
    if t == "attack":
        a = d.get("attack", [[], []])
        return f"P{pid} attack {_unshift(a[0])} -> {_unshift(a[1])}"
    if t == "died":
        extra = f" (box {_unshift(d['box'])})" if "box" in d else ""
        return f"P{pid} DIED{extra}"
    if t == "regenerate":
        return f"P{pid} regenerate {_unshift(d.get('pos', []))}"
    if t == "escaped":
        return f"P{pid} *** ESCAPED ***"
    if t in ("hp_update",):
        return f"P{pid} hp={d.get('hp')}"
    if t in ("cure", "use_kit"):
        return f"P{pid} {t} hp={d.get('hp')}"
    if t == "getkey":
        return f"P{pid} getkey {d.get('keyid')}"
    if t == "keymachine":
        return f"P{pid} open keymachine {_unshift(d.get('pos', []))}"
    if t == "escape_capsule":
        return f"P{pid} escape_capsule start={d.get('to_escape')}"
    if t == "place_trap":
        return f"P{pid} place {d.get('trap_type')} {_unshift(d.get('pos', []))}"
    if t == "detect":
        return f"P{pid} detect {_unshift(d.get('tar_pos', []))}"
    if t == "ai_error":
        return f"P{pid} AI_ERROR {d.get('error_log', '')}"
    if t == "inspect":
        return f"P{pid} inspect {_unshift(d.get('pos', []))} {d.get('interprops')}"
    if t == "tool_update":
        return f"P{pid} tools={d.get('tools')}"
    if t == "map_update":
        return f"MAP {d.get('args')}"
    return f"{t} {d}"


def _state_line(state: dict[int, dict[str, Any]]) -> str:
    parts = []
    for pid in sorted(state):
        s = state[pid]
        parts.append(
            f"P{pid}[{s['status']} hp{s['hp']} keys{sorted(s['keys'])} @{s['pos']}]"
        )
    return "  | ".join(parts)


def render(
    replay: list,
    *,
    round_filter: int | None = None,
    player_filter: int | None = None,
) -> Iterator[str]:
    if not replay:
        yield "(empty replay)"
        return

    birthplaces = replay[0]
    score_dic = replay[-1] if isinstance(replay[-1], dict) else None
    rounds = replay[1 : len(replay) - 1] if score_dic is not None else replay[1:]

    yield "=== LostSpace replay ==="
    yield f"birthplaces (grid): {[_unshift(bp) for bp in birthplaces]}"
    if score_dic is not None:
        order = sorted(score_dic, key=lambda p: -score_dic[p])
        yield "final ranking: " + ", ".join(
            f"{i+1}.P{p}({score_dic[p]}pts)" for i, p in enumerate(order)
        )
    yield ""

    state = _new_state(birthplaces)
    for r_idx, round_turns in enumerate(rounds, start=1):
        if round_filter is not None and r_idx != round_filter:
            # still advance state so later rounds are accurate
            for turn in round_turns:
                for d in turn:
                    _update(state, d)
            continue

        yield f"--- Round {r_idx} ---"
        for t_idx, turn in enumerate(round_turns):
            for d in turn:
                _update(state, d)
                if d.get("type") == "map_update":
                    if player_filter is None:
                        yield f"  T{t_idx}: {_fmt_action(d)}"
                    continue
                pid = d.get("playerid")
                if player_filter is not None and pid != player_filter:
                    continue
                yield f"  T{t_idx}: {_fmt_action(d)}"
        yield f"  state: {_state_line(state)}"
        yield ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentbench_frame.lostspace.replay_view",
        description="Print a readable turn-by-turn log of a native LostSpace replay.",
    )
    parser.add_argument("replay", help="path to a native replay .json")
    parser.add_argument("--round", type=int, default=None, help="show only this round (1-indexed)")
    parser.add_argument("--player", type=int, default=None, help="filter actions to this player id")
    args = parser.parse_args(argv)

    replay = _load(args.replay)
    for line in render(
        replay, round_filter=args.round, player_filter=args.player
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
