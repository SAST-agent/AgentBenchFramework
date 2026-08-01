"""Tests for hl/reference_recorder.py — pure parser of a `run_match` trace.

Contract (plan §2):
- ``record_reference_states(trace_path, *, spec_id, opponent, seat=0)`` reads
  a match-trace JSONL and emits a ``ReferenceStateSet`` with one
  ``ReferenceSample`` per Alive on-turn ``roundbegin`` decision point.
- The sample's ``transcript`` is the ordered list of ALL seat-0
  ``type:"observation"`` frames from the ``id`` frame through that
  ``roundbegin`` (inclusive) — on-turn AND off-turn.
- ``legal_actions`` / ``inventory`` / ``status`` are projected from the
  decision-point ``roundbegin`` frame's content.
- The recorder is a pure parser: it touches no match.py / logic code.
"""
import json
from pathlib import Path

from agentbench_frame.hl.distribution import STATUS_ALIVE, STATUS_SKIP
from agentbench_frame.hl.reference import ReferenceStateSet
from agentbench_frame.hl.reference_recorder import record_reference_states


def _write_trace(tmp_path: Path, entries: list) -> Path:
    p = tmp_path / "trace.jsonl"
    p.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )
    return p


def _obs(player: int, content) -> dict:
    return {"state": 0, "type": "observation", "player": player,
            "content": content}


def test_recorder_emits_one_sample_per_alive_decision_point(tmp_path):
    """A synthetic seat-0 stream: id + round1(Alive) + off-turn see +
    round2(Alive) + round3(Skip) + round4(Alive) + an `action` entry (ignored).

    Assert exactly 3 samples (rounds 1, 2, 4); round 3 (Skip) emits none.
    Each sample's transcript begins with the `id` frame and ends with its own
    `roundbegin`; round 2's prefix includes the off-turn `see` frame.
    """
    id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
    rb1 = {"type": "roundbegin", "round": 1, "inturn": 0, "status": STATUS_ALIVE,
           "state": [0, 0, 1], "hp": 3, "keys": 0,
           "attack": [], "move": [False] * 8, "detect": False, "interprops": [],
           "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                     "Transport": 0}}
    see = {"type": "see", "player": 0, "round": 1,
           "see": [{"pos": [1, 2], "type": "Box"}]}
    rb2 = {"type": "roundbegin", "round": 2, "inturn": 0, "status": STATUS_ALIVE,
           "state": [0, 0, 1], "hp": 3, "keys": 0,
           "attack": [], "move": [False] * 8, "detect": False, "interprops": [],
           "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                     "Transport": 0}}
    rb3 = {"type": "roundbegin", "round": 3, "inturn": 0, "status": STATUS_SKIP,
           "state": [0, 0, 1], "hp": 3, "keys": 0,
           "attack": [], "move": [False] * 8, "detect": False, "interprops": [],
           "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                     "Transport": 0}}
    rb4 = {"type": "roundbegin", "round": 4, "inturn": 0, "status": STATUS_ALIVE,
           "state": [0, 0, 1], "hp": 3, "keys": 0,
           "attack": [], "move": [False] * 8, "detect": False, "interprops": [],
           "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                     "Transport": 0}}

    entries = [
        _obs(0, id_frame),
        _obs(0, rb1),
        # An AI→judger action entry — MUST be ignored by the recorder.
        {"state": 0, "type": "action", "player": 0, "content": ["finish"]},
        _obs(0, see),
        _obs(0, rb2),
        _obs(0, rb3),
        _obs(0, rb4),
    ]
    trace_path = _write_trace(tmp_path, entries)

    rss = record_reference_states(
        trace_path, spec_id="bench-v1", opponent="rank01", seat=0,
    )
    assert isinstance(rss, ReferenceStateSet)
    assert rss.spec_id == "bench-v1"
    samples = rss.samples
    assert len(samples) == 3

    # round 1: transcript = [id, rb1]
    assert samples[0].transcript[0] == id_frame
    assert samples[0].transcript[-1] == rb1
    assert len(samples[0].transcript) == 2

    # round 2: transcript = [id, rb1, see, rb2] (includes the off-turn see)
    assert samples[1].transcript[0] == id_frame
    assert samples[1].transcript[-1] == rb2
    assert see in samples[1].transcript
    assert len(samples[1].transcript) == 4

    # round 4: transcript = [id, rb1, see, rb2, rb3, rb4]
    assert samples[2].transcript[-1] == rb4
    assert len(samples[2].transcript) == 6


def test_recorder_sample_fields_match_roundbegin(tmp_path):
    """One Alive roundbegin with known attack/move/detect/interprops/tools →
    sample.legal_actions equals the projection; status==0; inventory matches
    the derived counts (traps: made-used; Kit/Transport: int)."""
    id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
    rb = {
        "type": "roundbegin", "round": 1, "inturn": 0, "status": STATUS_ALIVE,
        "state": [0, 0, 1], "hp": 3, "keys": 1,
        "attack": [1, 2],
        "move": [True, False, True, False, False, False, False, False],
        "detect": True,
        "interprops": ["KeyMachine"],
        "tools": {"LandMine": [3, 1], "Sticky": [2, 2], "Kit": 1,
                  "Transport": 2},
    }
    entries = [_obs(0, id_frame), _obs(0, rb)]
    trace_path = _write_trace(tmp_path, entries)

    rss = record_reference_states(
        trace_path, spec_id="bench-v1", opponent="rank06", seat=0,
    )
    assert len(rss.samples) == 1
    s = rss.samples[0]

    expected_legal = {
        "attack": [1, 2],
        "move": [True, False, True, False, False, False, False, False],
        "detect": True,
        "interprops": ["KeyMachine"],
    }
    assert s.legal_actions == expected_legal
    assert s.status == 0
    # traps: made - used; Kit / Transport: int as-is
    assert s.inventory == {"LandMine": 2, "Sticky": 0, "Kit": 1, "Transport": 2}
    # observation is the full roundbegin content
    assert s.observation == rb
    assert s.seat == 0
    assert s.opponent == "rank06"


def test_recorder_is_deterministic(tmp_path):
    """Two calls on the same trace produce equal sample tuples
    (compare to_dict() lists)."""
    id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
    rb_a = {"type": "roundbegin", "round": 1, "inturn": 0, "status": STATUS_ALIVE,
            "state": [0, 0, 1], "hp": 3, "keys": 0,
            "attack": [1], "move": [True] + [False] * 7, "detect": True,
            "interprops": ["Box"],
            "tools": {"LandMine": [1, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0}}
    rb_b = {"type": "roundbegin", "round": 2, "inturn": 0, "status": STATUS_ALIVE,
            "state": [0, 0, 1], "hp": 3, "keys": 0,
            "attack": [], "move": [False] * 8, "detect": False,
            "interprops": [],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0}}
    entries = [_obs(0, id_frame), _obs(0, rb_a), _obs(0, rb_b)]
    trace_path = _write_trace(tmp_path, entries)

    a = record_reference_states(
        trace_path, spec_id="bench-v1", opponent="rank01", seat=0,
    )
    b = record_reference_states(
        trace_path, spec_id="bench-v1", opponent="rank01", seat=0,
    )
    assert [s.to_dict() for s in a.samples] == [s.to_dict() for s in b.samples]
