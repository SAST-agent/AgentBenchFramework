"""Build the probe-compatible reference state sets nu-v2-t / nu-v2-t2.

Why these exist (the ν problem the 2026-08-02 run hit):
- ``nu-v2.json`` has real ``legal_actions`` (diverse A(s)) but NO transcript —
  the current ``ReferenceProbe`` rejects a transcript-less sample
  (``ReferenceSampleError: reference sample has no transcript``).
- ``nu-recorded.json`` has transcripts but EMPTY ``legal_actions`` on every
  sample (the recorder cannot see ``Player.get_legal_actions()``, which lives
  in the logic, not the wire frames) — so A(s) degenerates and KL is
  meaningless.

The fix: merge. ``nu-v2-t.json`` = nu-v2 samples + a synthetic 2-frame
transcript (``id`` + the sample's ``roundbegin``), the minimum the probe
replays (identical shape to what ``reference_recorder`` produces).
``nu-v2-t2.json`` additionally appends hand-authored decision points whose
A(s) makes the seed candidate's FIRST emitted action in-support (low-HP+Kit →
``tool Kit``; Box-at-feet → ``interact Box``), so an edit that flips that
first action actually registers as policy_kl > 0 instead of collapsing to
out-of-support.

Run: ``python -m agentbench_frame.hl.nu_build`` (writes
``agentbench_data/reference/nu-v2-t.json`` + ``nu-v2-t2.json``).
"""
from __future__ import annotations

import json
from pathlib import Path

# src/agentbench_frame/hl/nu_build.py -> AgentBenchFramework/ (repo root)
_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NU_V2 = _ROOT / "agentbench_data" / "reference" / "nu-v2.json"
OUT_T = _ROOT / "agentbench_data" / "reference" / "nu-v2-t.json"
OUT_T2 = _ROOT / "agentbench_data" / "reference" / "nu-v2-t2.json"
OUT_T3 = _ROOT / "agentbench_data" / "reference" / "nu-v2-t3.json"

MV8 = [True] * 8  # open area: all 8 move directions legal


def _id_frame() -> dict:
    return {"type": "id", "id": 0, "birth_pos": [0, 0]}


def _add_transcripts(data: dict) -> None:
    """Give every sample the minimum 2-frame transcript (id + roundbegin)."""
    for s in data["samples"]:
        s["transcript"] = [_id_frame(), s["observation"]]


def _obs(round_: int, hp: int, keys: list, pos: list, *,
         kit: int = 0, landmine: int = 0, sticky: int = 0,
         transport: int = 0) -> dict:
    return {
        "type": "roundbegin", "inturn": 0, "state": 3, "round": round_,
        "status": 0, "hp": hp, "keys": keys, "pos": pos,
        "tools": {"LandMine": [landmine, 0], "Sticky": [sticky, 0],
                  "Kit": kit, "Transport": transport},
        "others": [
            {"player_id": 1, "status": 0, "keys": [1], "hp": 120},
            {"player_id": 2, "status": 0, "keys": [0], "hp": 180},
            {"player_id": 3, "status": 1, "keys": [0], "hp": 0},
        ],
    }


def _inv(kit: int = 0, landmine: int = 0, sticky: int = 0,
         transport: int = 0) -> dict:
    return {"LandMine": landmine, "Sticky": sticky,
            "Transport": transport, "Kit": kit}


def _sample(o: dict, legal: dict, inventory: dict) -> dict:
    return {"observation": o, "legal_actions": legal, "inventory": inventory,
            "status": 0, "seat": 0, "opponent": "rank06",
            "transcript": [_id_frame(), o]}


#: Hand-authored decision points whose A(s) keeps the EVOLVING candidate's
#: first emitted action IN-SUPPORT, so policy_kl is actually measurable.
#:
#: The coding agent keeps rewriting ``play()``'s first branch (KeyMachine
#: interact, escape, Box/Materials loot, tool Kit, move...), so a ν pinned to
#: one strategy goes stale: the new first action lands out-of-support and the
#: KL point is dropped. These points use MULTI-PROP tiles at the seat-0 spawn
#: so several first-action strategies are simultaneously in A(s) — a strategy
#: flip (e.g. KeyMachine-first -> move-first) then registers as policy_kl on
#: the points where BOTH versions emit in-support.
#:
#: ``pos`` is [0,0,1] (seat-0 spawn) because the probe replays a 2-frame
#: transcript (id + roundbegin) and the candidate's ``start_turn`` ignores the
#: roundbegin ``pos`` — its tracked position stays at the spawn set by the id
#: frame. So the candidate computes moves FROM [0,0,1]; making the ν state
#: also [0,0,1] lets move-coordinate normalization (``normalize_emitted``)
#: map the emitted target back to the A(s) direction index.
SPAWN = [0, 0, 1]
_K4 = [0, 1, 2, 3]
_K2 = [0, 1]


def _dp(round_, hp, keys, *props, kit=0, sticky=0, landmine=0, transport=0) -> dict:
    return _sample(
        _obs(round_, hp, keys, SPAWN, kit=kit, sticky=sticky,
             landmine=landmine, transport=transport),
        {"attack": [], "move": MV8, "detect": True, "interprops": list(props)},
        _inv(kit=kit, sticky=sticky, landmine=landmine, transport=transport))


EXTRA_SAMPLES = [
    # KeyMachine + EscapeCapsule with all 4 keys: KeyMachine-first,
    # escape-first, and move-first all in-support.
    _dp(20, 200, _K4, "KeyMachine", "EscapeCapsule"),
    # KeyMachine + Materials: KeyMachine-first vs Materials-loot vs move.
    _dp(22, 200, _K2, "KeyMachine", "Materials"),
    # KeyMachine + Box: KeyMachine-first vs Box-first vs move.
    _dp(24, 200, _K2, "KeyMachine", "Box"),
    # KeyMachine + low HP + Kit: KeyMachine-first vs tool-Kit vs move.
    _dp(26, 100, _K2, "KeyMachine", kit=2),
    # EscapeCapsule + Box with all 4 keys (no KeyMachine): escape-first,
    # Box-first, move.
    _dp(28, 200, _K4, "EscapeCapsule", "Box"),
    # Materials + low HP + Kit: Materials-loot vs tool-Kit vs move.
    _dp(30, 100, _K2, "Materials", kit=2),
    # Sticky trap + Materials: trap-first vs Materials-loot vs move.
    _dp(32, 200, _K2, "Materials", sticky=1),
    # Low HP + single Kit, empty tile: tool-Kit vs move.
    _dp(34, 70, [], kit=1),
    # Critical HP + Sticky trap + LandMine: trap-first vs tool-Kit vs move.
    _dp(36, 60, _K2, sticky=1, landmine=1),
    # Plain open tile: move / detect vs any prop-interact (out-of-support —
    # honest control point that only move-first candidates hit).
    _dp(38, 200, _K2),
]


def build(nu_v2: Path = DEFAULT_NU_V2, *, write: bool = True):
    """Load nu-v2.json, add transcripts, append in-support decision points.

    Returns the ``nu-v2-t3`` dict (the superset). ``write`` writes the three
    files: ``nu-v2-t`` (transcripts only), ``nu-v2-t2`` and ``nu-v2-t3`` (both
    carry the multi-prop strategy-agnostic extras; ``-t3`` is canonical).
    """
    data = json.loads(nu_v2.read_text(encoding="utf-8"))
    _add_transcripts(data)
    t = json.loads(json.dumps(data))
    t["spec_id"] = "nu-v2-t"
    t["n"] = len(t["samples"])
    t2 = json.loads(json.dumps(t))
    t2["samples"] = list(t2["samples"]) + [dict(s) for s in EXTRA_SAMPLES]
    t2["spec_id"] = "nu-v2-t2"
    t2["n"] = len(t2["samples"])
    t3 = json.loads(json.dumps(t))
    t3["samples"] = list(t3["samples"]) + [dict(s) for s in EXTRA_SAMPLES]
    t3["spec_id"] = "nu-v2-t3"
    t3["n"] = len(t3["samples"])
    if write:
        OUT_T.write_text(json.dumps(t, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        OUT_T2.write_text(json.dumps(t2, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        OUT_T3.write_text(json.dumps(t3, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return t3


if __name__ == "__main__":
    build()
    t = json.loads(OUT_T.read_text(encoding="utf-8"))
    t2 = json.loads(OUT_T2.read_text(encoding="utf-8"))
    t3 = json.loads(OUT_T3.read_text(encoding="utf-8"))
    print(f"wrote {OUT_T} ({t['n']}s), {OUT_T2} ({t2['n']}s), "
          f"{OUT_T3} ({t3['n']}s)")
