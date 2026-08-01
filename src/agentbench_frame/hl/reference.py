"""
Frozen benchmark spec + reference state set for policy KL.

BenchmarkSpec (doc §7): a versioned, frozen evaluation protocol — opponent
set, pairs, seats, timeout, and protocol notes. Within one ``spec_id``, these
are fixed so any two agent versions evaluated against this spec are
comparable. The spec *records* the map/seed/seat protocol; it does not invent
seeding the LostSpace logic lacks (the logic owns randomness via
``mapconf2.map`` — a documented limitation).

ReferenceStateSet (doc §4, §15): the fixed reference distribution ``ν`` for
policy KL — a frozen list of decision-point samples ``(observation,
get_legal_actions() dict, inventory, status)``. The *same* set is used for
every version pair so ``local_policy_kl_trace`` is comparable across versions.
Populated by probing the actual game logic (which owns
``Player.get_legal_actions()``), never mirrored by hand — see the controller,
which records the reference set by instrumenting a frozen reference roll.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class BenchmarkSpec:
    """A frozen evaluation protocol. Identity = spec_id + contents."""
    spec_id: str
    opponents: Tuple[str, ...]
    pairs: int
    seats: str               # "all" | "0".."3"
    timeout: float
    notes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "opponents", tuple(self.opponents))


@dataclass(frozen=True)
class ReferenceSample:
    """One decision point captured for the reference set.

    Carries the ordered judger→AI frame transcript from the ``id`` frame
    through the decision-point ``roundbegin`` (inclusive) — recorded from a
    real game so the probe can replay the candidate's world model faithfully
    (spec: "Reference sample carries the full frame transcript"). The probe
    boundary enforces non-empty transcript (§3); the dataclass itself stays
    a dumb carrier and accepts ``transcript=()`` so legacy unit fixtures and
    ``reference_seed.py`` still construct.
    """
    observation: Dict[str, Any]
    legal_actions: Dict[str, Any]   # the get_legal_actions() dict
    inventory: Dict[str, int]
    status: int
    seat: int
    opponent: str
    transcript: Tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": dict(self.observation),
            "legal_actions": dict(self.legal_actions),
            "inventory": dict(self.inventory),
            "status": self.status,
            "seat": self.seat,
            "opponent": self.opponent,
            "transcript": [dict(f) for f in self.transcript],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReferenceSample":
        transcript = tuple(dict(f) for f in (d.get("transcript") or ()))
        return cls(
            observation=dict(d["observation"]),
            legal_actions=dict(d["legal_actions"]),
            inventory=dict(d["inventory"]),
            status=int(d["status"]),
            seat=int(d["seat"]),
            opponent=str(d["opponent"]),
            transcript=transcript,
        )


@dataclass(frozen=True)
class ReferenceStateSet:
    """The fixed ν. Pinned to a benchmark spec_id for comparability."""
    spec_id: str
    samples: Tuple[ReferenceSample, ...]

    def __post_init__(self):
        object.__setattr__(self, "samples", tuple(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "spec_id": self.spec_id,
            "n": len(self.samples),
            "samples": [s.to_dict() for s in self.samples],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ReferenceStateSet":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        samples = tuple(ReferenceSample.from_dict(s) for s in d["samples"])
        return cls(spec_id=d["spec_id"], samples=samples)
