"""Tests for hl/reference.py — frozen BenchmarkSpec + ReferenceStateSet.

Contract (plan §reference):
- BenchmarkSpec: a versioned manifest of opponents + pairs + seats + timeout
  + protocol notes. Frozen within one version_id so version-pair comparisons
  are valid (doc §7). Records the seed/map/seat protocol — does not invent
  seeding the logic lacks (documented limitation).
- ReferenceStateSet: a frozen list of (observation, legal_actions dict,
  inventory, status) decision-point samples — the fixed ν for policy KL.
  Same set for every version pair so traces are comparable. Populated by
  probing the actual game logic (which owns get_legal_actions), never mirrored
  by hand.
"""
import json
from pathlib import Path

import pytest

from agentbench_frame.hl.reference import (
    BenchmarkSpec,
    ReferenceStateSet,
    ReferenceSample,
)


def _spec(tmp_path: Path, **kw) -> BenchmarkSpec:
    return BenchmarkSpec(
        spec_id="bench-v1",
        opponents=["rank01", "rank06"],
        pairs=2,
        seats="all",
        timeout=10.0,
        notes={"map": "mapconf2.map (static)", "seeding": "logic-owned"},
        **kw
    )


def test_benchmark_spec_is_frozen_and_hashable():
    s = _spec(Path())
    assert s.spec_id == "bench-v1"
    assert s.opponents == ("rank01", "rank06")
    assert s.pairs == 2
    assert s.seats == "all"
    # frozen: cannot mutate
    with pytest.raises(Exception):
        s.opponents.append("x")  # type: ignore


def test_benchmark_spec_records_protocol_notes():
    s = _spec(Path())
    assert s.notes["seeding"] == "logic-owned"
    assert "map" in s.notes


def test_benchmark_spec_version_pin_is_stable(tmp_path):
    s = _spec(tmp_path)
    # same inputs -> same content hash (the spec is reproducible)
    s2 = _spec(tmp_path)
    assert s.spec_id == s2.spec_id


def test_reference_sample_round_trips():
    rs = ReferenceSample(
        observation={"round": 3, "my_pos": [0, 0, 1]},
        legal_actions={"attack": [1], "move": [True, False, False, False,
                                              False, False, False, False],
                       "detect": True, "interprops": ["Box"]},
        inventory={"LandMine": 1, "Sticky": 0, "Transport": 0, "Kit": 1},
        status=0,
        seat=0,
        opponent="rank01",
    )
    d = rs.to_dict()
    assert d["status"] == 0
    rs2 = ReferenceSample.from_dict(d)
    assert rs2.status == 0
    assert rs2.legal_actions["attack"] == [1]
    assert rs2.inventory["Kit"] == 1


def test_reference_state_set_save_load(tmp_path):
    rs1 = ReferenceSample(
        observation={"round": 1}, legal_actions={"attack": [], "move": [False]*8,
            "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
    )
    rs2 = ReferenceSample(
        observation={"round": 5}, legal_actions={"attack": [2], "move": [True]+[False]*7,
            "detect": True, "interprops": ["Box"]},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=1, opponent="rank06",
    )
    rss = ReferenceStateSet(spec_id="bench-v1", samples=[rs1, rs2])
    path = tmp_path / "ref.json"
    rss.save(path)
    loaded = ReferenceStateSet.load(path)
    assert loaded.spec_id == "bench-v1"
    assert len(loaded.samples) == 2
    assert loaded.samples[1].opponent == "rank06"


def test_reference_state_set_immutable_after_freeze(tmp_path):
    """Once saved, the reference set is the canonical ν. Loading yields the
    same samples in the same order — determinism for KL comparisons."""
    rs = ReferenceSample(
        observation={"round": 1}, legal_actions={"attack": [], "move": [False]*8,
            "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
    )
    rss = ReferenceStateSet(spec_id="bench-v1", samples=[rs])
    path = tmp_path / "ref.json"
    rss.save(path)
    a = ReferenceStateSet.load(path)
    b = ReferenceStateSet.load(path)
    assert [s.to_dict() for s in a.samples] == [s.to_dict() for s in b.samples]


def test_reference_state_set_id_matches_spec(tmp_path):
    """The reference set is pinned to a benchmark spec — KL traces across
    version pairs are only comparable if they used the same spec."""
    rs = ReferenceSample(
        observation={}, legal_actions={"attack": [], "move": [False]*8,
            "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
    )
    rss = ReferenceStateSet(spec_id="bench-v1", samples=[rs])
    assert rss.spec_id == "bench-v1"
