"""Tests for the REPLAY_SKILL validation act (doc Fix-D).

Contract:
- rules_validation is a known event type.
- The coding agent's final message carries FIELDS_CHECKED / SCORE_DIC /
  MISMATCHES; the harness INDEPENDENTLY cross-checks the claimed score_dic
  against the replay's actual r[-1] -> pass | fail | no_score_claim.
- No replay resolvable -> validation_status="no_replay", no crash.
- The agent's report is persisted outside the workspace (artifacts/RULES_VALIDATION.md)
  so it never enters a snapshot diff.
"""
import json
from pathlib import Path

import pytest

from agentbench_frame.hl.controller import (
    HLIterationController,
    _parse_validation_report,
    _read_score_dic,
)
from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.runner import FakeRunner
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceStateSet, ReferenceSample,
)
from agentbench_frame.hl.events import KNOWN_EVENT_TYPES, read_events

GAME = "25_lostspace"


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text("THRESHOLD = 10\n", encoding="utf-8")
    return ws


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        spec_id="bench-v1", opponents=("rank01",),
        pairs=1, seats="0", timeout=5.0, notes={},
    )


def _ref_set() -> ReferenceStateSet:
    s = ReferenceSample(
        observation={"round": 1, "inturn": 0},
        legal_actions={"attack": [], "move": [True] + [False]*7,
                       "detect": False, "interprops": []},
        inventory={}, status=0, seat=0, opponent="rank01",
    )
    return ReferenceStateSet(spec_id="bench-v1", samples=(s,))


def _make_replay(tmp_path: Path, *, agent: str, score_dic) -> Path:
    """A real replay file + its matches.jsonl record under data_root/runs."""
    run_dir = (tmp_path / "runs" / GAME / agent / "run-01")
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    replay = artifacts / "match_0.json"
    replay.write_text(json.dumps([[[0, 0, 1]] * 4, [{"type": "move", "playerid": 0}], score_dic]),
                      encoding="utf-8")
    matches = run_dir / "matches.jsonl"
    matches.write_text(json.dumps({"opponent": "rank01", "replay": "artifacts/match_0.json",
                                   "candidate_result": "loss"}) + "\n",
                       encoding="utf-8")
    return replay


def _controller(tmp_path: Path, *, runner, data_root: Path) -> HLIterationController:
    ws = _make_workspace(tmp_path)
    codebase = HLCodebase(root=ws, store=tmp_path / "store")
    return HLIterationController(
        codebase=codebase, runner=runner, spec=_spec(), reference=_ref_set(),
        run_id="hl-test", events_path=tmp_path / "events.jsonl",
        evaluator_factory=None, data_root=data_root, game=GAME,
    )


def test_rules_validation_is_a_known_event_type():
    assert "rules_validation" in KNOWN_EVENT_TYPES


def test_parse_validation_report():
    text = (
        "The first action dict means: type=move, playerid=0, pos shifted.\n"
        "FIELDS_CHECKED: 5\n"
        'SCORE_DIC: {"0": 4, "1": 3, "2": 2, "3": 1}\n'
        "MISMATCHES: none\n"
    )
    rep = _parse_validation_report(text)
    assert rep["fields_checked"] == 5
    assert rep["score_dic"] == {"0": 4, "1": 3, "2": 2, "3": 1}
    assert rep["mismatches"] == []


def test_parse_validation_report_missing_score_is_none():
    rep = _parse_validation_report("I could not parse the replay.")
    assert rep["score_dic"] is None
    assert rep["fields_checked"] is None


def test_parse_validation_report_mismatches_list():
    rep = _parse_validation_report(
        "MISMATCHES: coord offset, wrong index\n")
    assert rep["mismatches"] == ["coord offset", "wrong index"]


def test_read_score_dic(tmp_path):
    replay = _make_replay(tmp_path, agent="a", score_dic={"0": 4, "1": 3, "2": 2, "3": 1})
    assert _read_score_dic(replay) == {"0": 4, "1": 3, "2": 2, "3": 1}
    assert _read_score_dic(tmp_path / "missing.json") is None


def test_rules_validation_act_pass(tmp_path):
    data_root = tmp_path / "data"
    replay = _make_replay(data_root, agent="hl-test",
                          score_dic={"0": 4, "1": 3, "2": 2, "3": 1})
    report = (
        "FIELDS_CHECKED: 3\n"
        'SCORE_DIC: {"0": 4, "1": 3, "2": 2, "3": 1}\n'
        "MISMATCHES: none\n"
    )
    ctrl = _controller(tmp_path, runner=FakeRunner(
        transform=lambda ws: None, edit_type="noop", final_text=report),
        data_root=data_root)
    payload = ctrl.rules_validation_act()
    assert payload is not None
    assert payload["validation_status"] == "pass"
    assert payload["fields_checked"] == 3
    assert payload["replay"] == str(replay)
    events = read_events(tmp_path / "events.jsonl")
    assert [e["event_type"] for e in events] == ["rules_validation"]
    ev = events[0]
    assert ev["validation_status"] == "pass"
    assert ev["mismatches"] == []
    # the report was persisted outside the workspace
    doc = Path(ev["doc_path"])
    assert doc.exists()
    assert "RULES_VALIDATION" in doc.read_text(encoding="utf-8")


def test_rules_validation_act_fail_on_score_mismatch(tmp_path):
    data_root = tmp_path / "data"
    _make_replay(data_root, agent="hl-test",
                 score_dic={"0": 4, "1": 3, "2": 2, "3": 1})
    report = (
        "FIELDS_CHECKED: 2\n"
        'SCORE_DIC: {"0": 3, "1": 4, "2": 2, "3": 1}\n'
        "MISMATCHES: none\n"
    )
    ctrl = _controller(tmp_path, runner=FakeRunner(
        transform=lambda ws: None, edit_type="noop", final_text=report),
        data_root=data_root)
    payload = ctrl.rules_validation_act()
    assert payload["validation_status"] == "fail"
    assert any("score_dic" in m for m in payload["mismatches"])


def test_rules_validation_act_no_score_claim_is_honest(tmp_path):
    data_root = tmp_path / "data"
    _make_replay(data_root, agent="hl-test",
                 score_dic={"0": 4, "1": 3, "2": 2, "3": 1})
    ctrl = _controller(tmp_path, runner=FakeRunner(
        transform=lambda ws: None, edit_type="noop", final_text="no report"),
        data_root=data_root)
    payload = ctrl.rules_validation_act()
    assert payload["validation_status"] == "no_score_claim"
    # never a false pass when the agent was unresponsive
    assert payload["fields_checked"] is None


def test_rules_validation_no_replay_writes_no_replay_event(tmp_path):
    data_root = tmp_path / "empty"
    data_root.mkdir()
    ctrl = _controller(tmp_path, runner=FakeRunner(
        transform=lambda ws: None, edit_type="noop"),
        data_root=data_root)
    payload = ctrl.rules_validation_act()
    assert payload is None
    events = read_events(tmp_path / "events.jsonl")
    assert events and events[0]["event_type"] == "rules_validation"
    assert events[0]["validation_status"] == "no_replay"


def test_rules_validation_no_data_root_skips(tmp_path):
    ctrl = _controller(tmp_path, runner=FakeRunner(
        transform=lambda ws: None, edit_type="noop"),
        data_root=tmp_path / "nope")
    # data_root exists but has no runs -> no replay -> no_replay, no crash
    assert ctrl.rules_validation_act() is None
