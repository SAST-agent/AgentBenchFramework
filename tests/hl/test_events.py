"""Tests for hl/events.py — public-schema event layer for HL iteration.

Contract (from the measurement contract / plan):
- Every event written carries public fields: schema_version, event_id,
  run_id, created_at, event_type.
- New HL event types: agent_act, version, policy_kl, occupancy_shift, budget.
- Append-only: events are only added, never rewritten.
- Forward-compat: the loader skips+warns on unknown event_type, never crashes.
- Unknown extra fields are preserved (not dropped) so newer events survive an
  older reader that only knows the core fields.
"""
import json
from pathlib import Path

import pytest

from agentbench_frame.hl.events import HLEventWriter, read_events, PUBLIC_FIELDS


def _writer(tmp_path: Path) -> HLEventWriter:
    return HLEventWriter(
        run_id="run_abc",
        path=tmp_path / "events.jsonl",
    )


def test_write_agent_act_carries_public_fields(tmp_path):
    w = _writer(tmp_path)
    w.write("agent_act", act_id="a1", version_before="v0", edit_policy_mode="free")
    w.close()

    events = read_events(tmp_path / "events.jsonl")
    assert len(events) == 1
    ev = events[0]
    for f in PUBLIC_FIELDS:
        assert f in ev, f"missing public field {f}"
    assert ev["event_type"] == "agent_act"
    assert ev["run_id"] == "run_abc"
    assert ev["schema_version"] >= 1
    assert ev["event_id"]  # non-empty
    assert ev["created_at"]  # non-empty
    assert ev["act_id"] == "a1"
    assert ev["version_before"] == "v0"


def test_event_ids_are_unique(tmp_path):
    w = _writer(tmp_path)
    for i in range(5):
        w.write("agent_act", act_id=f"a{i}")
    w.close()
    events = read_events(tmp_path / "events.jsonl")
    ids = [e["event_id"] for e in events]
    assert len(set(ids)) == len(ids)


def test_all_new_event_types_round_trip(tmp_path):
    w = _writer(tmp_path)
    w.write("agent_act", act_id="a1")
    w.write("version", version_id="v1", content_hash="h1",
            parent_version_id="v0", edit_type="add_rule")
    w.write("eval", evaluation_status="complete", win_rate=0.5)
    w.write("policy_kl", version_before="v0", version_after="v1",
            local_policy_kl_trace=[0.1, 0.2, 0.0])
    w.write("occupancy_shift", version_before="v0", version_after="v1",
            shift=0.07)
    w.write("budget", scope="learning",
            coding_agent_acts=1, prompt_tokens=1200,
            completion_tokens=None)  # None -> unknown, not 0
    w.close()

    events = read_events(tmp_path / "events.jsonl")
    types = [e["event_type"] for e in events]
    assert types == ["agent_act", "version", "eval",
                     "policy_kl", "occupancy_shift", "budget"]

    budget = events[-1]
    # unknown token/time must be recorded as missing, not coerced to 0
    assert "completion_tokens" in budget
    assert budget["completion_tokens"] is None


def test_loader_skips_unknown_event_type_without_crashing(tmp_path):
    path = tmp_path / "events.jsonl"
    # Hand-write a file with one known and one unknown event type, plus a
    # record missing the public fields entirely (legacy line).
    with path.open("w") as f:
        f.write(json.dumps({
            "schema_version": 1, "event_id": "e1", "run_id": "r",
            "created_at": "t", "event_type": "agent_act", "act_id": "a1",
        }) + "\n")
        f.write(json.dumps({
            "schema_version": 1, "event_id": "e2", "run_id": "r",
            "created_at": "t", "event_type": "from_the_future",
            "weird_payload": {"x": 1},
        }) + "\n")
        f.write(json.dumps({"event": "step", "reward": 1.0}) + "\n")  # legacy

    events, skipped = read_events(path, return_skipped=True)
    # The legacy {"event":"step"} line has a recognized type, so it is
    # recovered (best-effort) rather than skipped. Only the genuinely-unknown
    # "from_the_future" type is skipped.
    assert len(events) == 2
    assert events[0]["event_type"] == "agent_act"
    assert events[1]["event_type"] == "step"
    assert skipped == 1


def test_append_only_does_not_truncate(tmp_path):
    path = tmp_path / "events.jsonl"
    w = _writer(tmp_path)
    w.write("agent_act", act_id="a1")
    w.close()

    # Reopen for append and add more — earlier events must survive.
    w2 = HLEventWriter(run_id="run_abc", path=path, append=True)
    w2.write("agent_act", act_id="a2")
    w2.close()

    events = read_events(path)
    assert [e["act_id"] for e in events] == ["a1", "a2"]


def test_missing_score_stays_missing(tmp_path):
    """eval events with incomplete status carry no win_rate, not 0."""
    w = _writer(tmp_path)
    w.write("eval", evaluation_status="incomplete", win_rate=None)
    w.close()
    ev = read_events(tmp_path / "events.jsonl")[0]
    assert ev["evaluation_status"] == "incomplete"
    assert "win_rate" in ev
    assert ev["win_rate"] is None
