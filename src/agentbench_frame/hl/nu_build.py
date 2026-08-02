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

DEFAULT_NU_V2 = (
    Path(__file__).resolve().parents[2]
    / "agentbench_data" / "reference" / "nu-v2.json"
)
OUT_T = (
    Path(__file__).resolve().parents[2]
    / "agentbench_data" / "reference" / "nu-v2-t.json"
)
OUT_T2 = (
    Path(__file__).resolve().parents[2]
    / "agentbench_data" / "reference" / "nu-v2-t2.json"
)

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


#: Hand-authored decision points whose A(s) makes the v1 candidate's first
#: emitted action IN-SUPPORT, so policy_kl is actually measurable there.
EXTRA_SAMPLES = [
    # injured with a Kit -> v1 emits `tool Kit` (legal when Kit > 0)
    _sample(_obs(16, 100, [0, 1], [2, 2, 1], kit=2),
            {"attack": [], "move": MV8, "detect": True, "interprops": []},
            _inv(kit=2)),
    # a dead player dropped a Box at the tile -> v1 emits `interact Box` (legal)
    _sample(_obs(18, 200, [0, 1], [1, 1, 1]),
            {"attack": [], "move": MV8, "detect": True, "interprops": ["Box"]},
            _inv()),
    # critical HP with a single Kit -> `tool Kit`
    _sample(_obs(25, 70, [0, 1, 2], [4, 4, 1], kit=1),
            {"attack": [], "move": MV8, "detect": True, "interprops": []},
            _inv(kit=1)),
    # Box + Materials both at the tile -> v1 still tries `interact Box` first
    _sample(_obs(27, 200, [0, 1, 2], [3, 2, 1], sticky=1),
            {"attack": [], "move": MV8, "detect": True,
             "interprops": ["Box", "Materials"]},
            _inv(sticky=1)),
]


def build(nu_v2: Path = DEFAULT_NU_V2, *, write: bool = True):
    """Load nu-v2.json, add transcripts, optionally append in-support points.

    Returns the ``nu-v2-t2`` dict (the superset). ``write`` writes both files.
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
    if write:
        OUT_T.write_text(json.dumps(t, ensure_ascii=False, indent=1),
                         encoding="utf-8")
        OUT_T2.write_text(json.dumps(t2, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return t2


if __name__ == "__main__":
    build()
    t = json.loads(OUT_T.read_text(encoding="utf-8"))
    t2 = json.loads(OUT_T2.read_text(encoding="utf-8"))
    print(f"wrote {OUT_T} ({t['n']} samples) and {OUT_T2} ({t2['n']} samples)")
