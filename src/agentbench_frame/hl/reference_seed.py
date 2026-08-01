"""Hand-authored ReferenceStateSet — **TEST FIXTURE ONLY, not a production ν.**

This module is preserved for the unit-test suite (``tests/hl/test_reference_seed.py``)
and for ``ReferenceSample`` / ``ReferenceStateSet`` schema + round-trip tests.
It is **no longer a production source of ν**.

The production ν is produced by ``agentbench_frame.hl.reference_recorder`` from
a real saved match trace (``trace.jsonl``) — see that module's docstring and
the "Record a ν, then iterate" section of ``hl/README.md``.

The samples here carry ``transcript=()`` (the empty default). That is fine for
schema / round-trip / count unit tests, but the ``ReferenceProbe`` explicitly
REJECTS empty-transcript samples by raising ``ReferenceSampleError`` ("re-record
this ν") — a legacy single-frame ν cannot build the candidate's world model at
the decision point (the probe would replay nothing and reach a state detached
from the real game). Re-record ν from a real roll via the recorder; do not add
transcripts to these seed samples — they stay single-frame by design (the
fixture's purpose is structural schema coverage, not behavioral fidelity).

The seed covers 8 diverse decision points (early move, attack, escape,
KeyMachine interact, trap placement, Kit heal, multi-key routing, last-key
transit) so any structural change to ``legal_actions`` shape registers on at
least one point. ``build_seed`` / ``main`` are kept working — the CLI still
writes the JSON fixture — but the resulting file is a fixture, not a ν to
iterate against.

Usage (test fixture only)::

    python -m agentbench_frame.hl.reference_seed --out nu-seed-fixture.json --spec-id hl-seed-v1

To produce a REAL ν for iteration::

    python -m agentbench_frame.hl.reference_recorder \
      --trace <trace.jsonl> --spec-id <id> --opponent <name> --out nu.json
    python -m agentbench_frame.hl --reference nu.json ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceSample, ReferenceStateSet,
)


def _seed_samples():
    """Representative decision points for the LostSpace candidate.

    Coordinates are *grid* coords (0..6, layer 0=top..2=bottom). The probe
    sends these as the ``roundbegin`` observation; only the fields the
    candidate actually reads matter, but we include the common ones.

    The set is deliberately DIVERSE across the situations a real strategy edit
    is likely to touch — early movement, looting Materials, attacking,
    trap-placement, Kit use, multi-key transit, and the endgame escape — so a
    behavioral change registers on at least one point (KL>0) instead of every
    edit washing out to KL=0 against a 3-point ν.
    """
    # 1. Early game, NW corner area, 2 cardinal moves open, on Materials.
    s1 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 2,
            "status": 0, "hp": 200, "keys": [0],
            "pos": [0, 0, 1],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [0], "hp": 200},
                {"player_id": 2, "status": 0, "keys": [0], "hp": 200},
                {"player_id": 3, "status": 0, "keys": [0], "hp": 200},
            ],
        },
        legal_actions={
            "attack": [],
            "move": [True, False, True, False, False, False, False, False],
            "detect": True,
            "interprops": ["Materials"],
        },
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank06",
    )
    # 2. Mid game, adjacent to an enemy (can attack), carrying a LandMine.
    s2 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 12,
            "status": 0, "hp": 150, "keys": [0, 1],
            "pos": [3, 2, 1],
            "tools": {"LandMine": [1, 0], "Sticky": [1, 0], "Kit": 1,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [1], "hp": 120},
                {"player_id": 2, "status": 2, "keys": [0, 1, 2, 3], "hp": 200},
                {"player_id": 3, "status": 1, "keys": [0], "hp": 0},
            ],
        },
        legal_actions={
            "attack": [1],
            "move": [True, True, True, True, True, True, True, True],
            "detect": True,
            "interprops": [],
        },
        inventory={"LandMine": 1, "Sticky": 1, "Transport": 0, "Kit": 1},
        status=0, seat=0, opponent="rank06",
    )
    # 3. End game: 4 keys, standing on the escape capsule at center top.
    s3 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 30,
            "status": 0, "hp": 80, "keys": [0, 1, 2, 3],
            "pos": [3, 3, 0],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [0, 1], "hp": 60},
                {"player_id": 2, "status": 2, "keys": [0, 1, 2, 3], "hp": 200},
                {"player_id": 3, "status": 1, "keys": [0], "hp": 0},
            ],
        },
        legal_actions={
            "attack": [],
            "move": [True, True, True, True, True, True, True, True],
            "detect": False,
            "interprops": ["EscapeCapsule"],
        },
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank06",
    )
    # 4. On a KeyMachine with 2 keys — the interact-to-get-key decision.
    s4 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 8,
            "status": 0, "hp": 180, "keys": [0, 1],
            "pos": [5, 2, 0],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [0], "hp": 180},
                {"player_id": 2, "status": 0, "keys": [0, 1], "hp": 160},
                {"player_id": 3, "status": 0, "keys": [0], "hp": 200},
            ],
        },
        legal_actions={
            "attack": [],
            "move": [True, True, False, False, False, False, False, False],
            "detect": True,
            "interprops": ["KeyMachine"],
        },
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank06",
    )
    # 5. Carrying a Sticky trap, open corridor — trap-placement vs move decision.
    s5 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 10,
            "status": 0, "hp": 160, "keys": [0, 1],
            "pos": [4, 4, 1],
            "tools": {"LandMine": [0, 0], "Sticky": [1, 0], "Kit": 0,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [0], "hp": 140},
                {"player_id": 2, "status": 0, "keys": [1], "hp": 170},
                {"player_id": 3, "status": 1, "keys": [0], "hp": 0},
            ],
        },
        legal_actions={
            "attack": [],
            "move": [True, True, True, True, True, True, True, True],
            "detect": True,
            "interprops": [],
        },
        inventory={"LandMine": 0, "Sticky": 1, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank06",
    )
    # 6. Low HP, carrying a Kit — heal vs keep moving decision.
    s6 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 15,
            "status": 0, "hp": 40, "keys": [0, 1],
            "pos": [2, 5, 1],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 1,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [0], "hp": 100},
                {"player_id": 2, "status": 2, "keys": [0, 1, 2, 3], "hp": 200},
                {"player_id": 3, "status": 1, "keys": [0], "hp": 0},
            ],
        },
        legal_actions={
            "attack": [],
            "move": [True, True, True, True, False, False, False, False],
            "detect": True,
            "interprops": [],
        },
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 1},
        status=0, seat=0, opponent="rank06",
    )
    # 7. Mid-board, 3 keys, multiple move dirs — routing toward the last key.
    s7 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 20,
            "status": 0, "hp": 130, "keys": [0, 1, 2],
            "pos": [3, 4, 1],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [0, 1], "hp": 110},
                {"player_id": 2, "status": 0, "keys": [0, 1, 2], "hp": 140},
                {"player_id": 3, "status": 1, "keys": [0], "hp": 0},
            ],
        },
        legal_actions={
            "attack": [],
            "move": [True, True, True, True, True, True, False, False],
            "detect": True,
            "interprops": [],
        },
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank06",
    )
    # 8. Near the capsule, 3 keys, last-key routing — another mid-board route
    #    point (status Alive so the candidate emits; the WAIT_FOR_ESCAPE status
    #    crashes the sample AI, which would waste the point on a None emission).
    s8 = ReferenceSample(
        observation={
            "type": "roundbegin", "inturn": 0, "state": 3, "round": 24,
            "status": 0, "hp": 110, "keys": [0, 1, 2],
            "pos": [2, 3, 1],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0},
            "others": [
                {"player_id": 1, "status": 0, "keys": [0, 1], "hp": 90},
                {"player_id": 2, "status": 0, "keys": [0, 1, 2], "hp": 120},
                {"player_id": 3, "status": 1, "keys": [0], "hp": 0},
            ],
        },
        legal_actions={
            "attack": [],
            "move": [True, True, True, True, True, True, True, True],
            "detect": True,
            "interprops": [],
        },
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank06",
    )
    return [s1, s2, s3, s4, s5, s6, s7, s8]


def build_seed(spec_id: str = "hl-seed-v1") -> ReferenceStateSet:
    return ReferenceStateSet(spec_id=spec_id, samples=tuple(_seed_samples()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m agentbench_frame.hl.reference_seed",
        description="Write a hand-authored seed ReferenceStateSet (ν) JSON.",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--spec-id", default="hl-seed-v1")
    args = p.parse_args(argv)
    nu = build_seed(args.spec_id)
    nu.save(args.out)
    print(f"wrote {len(nu)} reference samples to {args.out} "
          f"(spec_id={nu.spec_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
