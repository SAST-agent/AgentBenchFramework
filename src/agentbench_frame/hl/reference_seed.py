"""Emit a seed ReferenceStateSet (ν) for a first real HL run.

The *proper* way to populate ν is to instrument the logic's
``Player.get_legal_actions()`` during a frozen reference roll (see
``reference.py``'s docstring) — that is a follow-up. This module ships a
**hand-authored seed ν**: a few representative decision points whose
``legal_actions`` dicts match the exact ``get_legal_actions()`` shape
(``{'attack': [ids], 'move': [8 bools], 'detect': bool, 'interprops': [...]}}``)
so the iteration loop is runnable and policy-KL is measurable out of the box.
KL is coarse at this size (3 decision points) but real; expand ν later.

Usage::

    python -m agentbench_frame.hl.reference_seed --out nu-v1.json --spec-id hl-v1
    python -m agentbench_frame.hl --reference nu-v1.json ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceSample, ReferenceStateSet,
)


def _seed_samples():
    """Three representative decision points for the LostSpace candidate.

    Coordinates are *grid* coords (0..6, layer 0=top..2=bottom). The probe
    sends these as the ``roundbegin`` observation; only the fields the
    candidate actually reads matter, but we include the common ones.
    """
    # 1. Early game, NW corner area, all 4 cardinal moves open, on Materials.
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
    return [s1, s2, s3]


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
