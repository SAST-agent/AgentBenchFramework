"""Tests for the action-frequency KL channel (hl/action_freq.py + controller).

Contract (the user-facing channel):
- ``canonical_action`` maps a wire action frame to a stable token: coordinate
  actions collapse to their macro, prop/tool args are kept, ``finish`` is
  ``("finish",)``.
- ``count_actions`` aggregates a seat's canonical actions from one trace JSONL.
- ``action_freq_kl`` is 0 for identical frequency distributions, > 0 for
  differing ones, finite when the supports differ (epsilon-smoothed).
- Controller (``--action-freq``): after each eval the version's action mix is
  counted from its real traces; acts with a prior version emit an
  ``action_freq`` event with ``kl`` vs the previous version and surface it in
  the next act's feedback.
"""
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from agentbench_frame.hl.action_freq import (
    action_freq_kl, canonical_action, count_actions,
)
from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.controller import HLIterationController
from agentbench_frame.hl.events import read_events
from agentbench_frame.hl.probe import EmittedAction
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceSample, ReferenceStateSet,
)
from agentbench_frame.hl.runner import FakeRunner


# ---- canonical_action -------------------------------------------------------

def test_canonical_action_mapping():
    assert canonical_action({"type": "finish"}) == ("finish",)
    assert canonical_action({"type": "action", "action": ["move", [1, 2, 3]]}) \
        == ("move",)
    assert canonical_action({"type": "action", "action": ["detect", [1, 2, 3]]}) \
        == ("detect",)
    assert canonical_action({"type": "action", "action": ["attack", [0, 0, 1], 2]}) \
        == ("attack",)
    assert canonical_action(
        {"type": "action", "action": ["interact", "KeyMachine"]}) \
        == ("interact", "KeyMachine")
    assert canonical_action(
        {"type": "action", "action": ["interact", "Box"]}) == ("interact", "Box")
    # EscapeCapsule keeps its start/abort flag
    assert canonical_action(
        {"type": "action", "action": ["interact", "EscapeCapsule", 1]}) \
        == ("interact", "EscapeCapsule", "1")
    # tool and use_tool unify
    assert canonical_action({"type": "action", "action": ["tool", "Kit"]}) \
        == ("tool", "Kit")
    assert canonical_action({"type": "action", "action": ["use_tool", "Kit"]}) \
        == ("tool", "Kit")
    assert canonical_action(None) == ("<malformed>",)
    assert canonical_action({"type": "action", "action": []}) == ("<empty>",)


def test_count_actions_aggregates_seat(tmp_path):
    p = tmp_path / "t.trace.jsonl"
    entries = [
        {"state": 0, "type": "action", "player": 0,
         "content": {"type": "action", "action": ["move", [1, 2, 3]]}},
        {"state": 1, "type": "action", "player": 0,
         "content": {"type": "action", "action": ["move", [4, 5, 6]]}},
        {"state": 2, "type": "action", "player": 1,  # other seat ignored
         "content": {"type": "action", "action": ["tool", "Kit"]}},
        {"state": 3, "type": "action", "player": 0,
         "content": {"type": "action", "action": ["interact", "KeyMachine"]}},
        {"state": 4, "type": "action", "player": 0, "content": {"type": "finish"}},
        {"state": 5, "type": "observation", "player": 0, "content": {}},
    ]
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n",
                 encoding="utf-8")
    c = count_actions(p, seat=0)
    assert c == Counter({("move",): 2, ("interact", "KeyMachine"): 1,
                         ("finish",): 1})
    # missing file -> empty, never raises
    assert count_actions(tmp_path / "nope.jsonl", seat=0) == Counter()


# ---- action_freq_kl ---------------------------------------------------------

def test_action_freq_kl_zero_when_identical():
    c = Counter({("move",): 5, ("interact", "KeyMachine"): 2, ("finish",): 1})
    assert action_freq_kl(Counter(c), Counter(c)) == pytest.approx(0.0, abs=1e-9)


def test_action_freq_kl_positive_when_distributions_differ():
    old = Counter({("move",): 5, ("interact", "KeyMachine"): 1, ("finish",): 1})
    new = Counter({("move",): 5, ("interact", "KeyMachine"): 1,
                   ("tool", "Kit"): 2, ("finish",): 1})
    assert action_freq_kl(new, old) > 0


def test_action_freq_kl_finite_on_one_sided_flip():
    # the new version takes an action the old never took -> KL must be finite
    old = Counter({("move",): 8, ("finish",): 1})
    new = Counter({("move",): 8, ("tool", "Kit"): 1, ("finish",): 1})
    kl = action_freq_kl(new, old)
    assert kl > 0 and kl != float("inf")
    # empty -> 0
    assert action_freq_kl(Counter(), Counter()) == 0.0


# ---- controller wiring ------------------------------------------------------

def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text("THRESHOLD = 10\n", encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    return ws


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(spec_id="bench-v1", opponents=("rank01",),
                         pairs=1, seats="0", timeout=5.0,
                         notes={"map": "mapconf2.map"})


def _ref_set() -> ReferenceStateSet:
    s = ReferenceSample(
        observation={"round": 1, "inturn": 0, "pos": [0, 0, 1]},
        legal_actions={"attack": [], "move": [True] + [False] * 7,
                       "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=({"type": "id", "id": 0, "birth_pos": [0, 0]},),
    )
    return ReferenceStateSet(spec_id="seed", samples=(s,))


def _action_trace(path: Path, *, moves: int, interacts: int, kits: int) -> Path:
    entries = []
    for i in range(moves):
        entries.append({"state": i, "type": "action", "player": 0,
                        "content": {"type": "action",
                                    "action": ["move", [i, 0, 1]]}})
    for i in range(interacts):
        entries.append({"state": 100 + i, "type": "action", "player": 0,
                        "content": {"type": "action",
                                    "action": ["interact", "KeyMachine"]}})
    for i in range(kits):
        entries.append({"state": 200 + i, "type": "action", "player": 0,
                        "content": {"type": "action", "action": ["tool", "Kit"]}})
    entries.append({"state": 300, "type": "action", "player": 0,
                    "content": {"type": "finish"}})
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n",
                    encoding="utf-8")
    return path


@dataclass
class _StubEvalResult:
    run_dir: Path
    summary: dict
    matches: list
    error_count: int


class _QueueEvaluator:
    _queue: List = []

    def __init__(self, *a, **k):
        pass

    def evaluate(self):
        return _QueueEvaluator._queue.pop(0)


class _StubProbe:
    _queue: List = []

    def __init__(self, cmd, cwd, timeout):
        pass

    def probe_set(self, samples):
        q = _StubProbe._queue.pop(0) if _StubProbe._queue else []
        return [EmittedAction(primitive=q[i] if i < len(q) else ("finish",),
                              out_of_support=False, sample_index=i)
                for i in range(len(samples))]

    def close(self):
        pass


def test_action_freq_event_and_feedback(tmp_path):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    rd1, rd2 = tmp_path / "run1", tmp_path / "run2"
    rd1.mkdir()
    rd2.mkdir()
    _action_trace(rd1 / "a.trace.jsonl", moves=5, interacts=1, kits=0)
    _action_trace(rd2 / "a.trace.jsonl", moves=5, interacts=1, kits=2)
    results = [
        _StubEvalResult(
            rd1, {"win_rate": 0.5, "evaluation_status": "complete",
                  "lostspace": {"aggregate": {"win_rate": 0.5}}},
            [{"trace": "a.trace.jsonl", "candidate_seat": 0}], 0),
        _StubEvalResult(
            rd2, {"win_rate": 0.5, "evaluation_status": "complete",
                  "lostspace": {"aggregate": {"win_rate": 0.5}}},
            [{"trace": "a.trace.jsonl", "candidate_seat": 0}], 0),
    ]
    _QueueEvaluator._queue = list(results)
    _StubProbe._queue = [[("finish",)], [("finish",)]]  # 1-sample reference

    state = {"n": 0}

    def change(w):
        state["n"] += 1
        (w / "agent.py").write_text(f"THRESHOLD = {10 + state['n']}\n",
                                    encoding="utf-8")

    ctrl = HLIterationController(
        codebase=cb, runner=FakeRunner(transform=change, edit_type="parametrize"),
        spec=_spec(), reference=_ref_set(), run_id="r",
        events_path=tmp_path / "e.jsonl", epsilon=0.1,
        action_freq=True,
        evaluator_factory=_QueueEvaluator, probe_factory=_StubProbe,
        stage_root=tmp_path / "stage",
    )
    v = ctrl.act(version_before=None)   # act 1: count v1's mix, no prior to KL
    v = ctrl.act(version_before=v)      # act 2: KL(v2 || v1)

    events = read_events(tmp_path / "e.jsonl")
    af = [e for e in events if e["event_type"] == "action_freq"]
    assert len(af) == 1
    assert af[0]["kl"] > 0
    assert af[0]["version_before"] != af[0]["version_after"]
    assert af[0]["total_actions_new"] == 9   # 5 move + 1 interact + 2 tool + 1 finish
    assert af[0]["total_actions_old"] == 7
    assert af[0]["vocab_size"] == 4
    top = af[0]["top_actions"]
    assert any(r["action"] == ["tool", "Kit"] and r["count_new"] == 2
               and r["count_old"] == 0 for r in top)

    # the next act's prompt learns the action-mix shift
    assert ctrl._prev_feedback["action_kl"] > 0
    assert any(r["action"] == ["tool", "Kit"] for r in ctrl._prev_feedback["action_top"])
