"""Tests for the agent-callable `revert_to_best` tool + controller best-tracking.

Contract (openspec/changes/hl-revert-to-best-tool/design.md):
- The controller tracks the best-scoring version across acts: win_rate primary
  (higher better), avg_rank tiebreak (lower better). Updated only on strict
  improvement, after a complete eval.
- `revert_to_best` is a TERMINAL tool symmetric with `edit`: on success it ends
  the act (the agent cannot edit in the same act — revert invalidates its read
  context). On failure (no best yet) the loop continues so the agent can edit.
- A successful revert restores the workspace to the best version's code via
  `codebase.restore`; the act's version_after is the rollback handle
  (content_hash == best, parent_version_id == best.version_id); re-eval is
  skipped (result already known); a `revert` event is emitted.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from agentbench_frame.hl.controller import HLIterationController
from agentbench_frame.hl.codebase import HLCodebase, VersionHandle
from agentbench_frame.hl.runner import FakeRunner, AgentRunResult, ApiCodingRunner
from agentbench_frame.hl.llm import LLMResponse, ToolCall, Usage
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceStateSet, ReferenceSample,
)
from agentbench_frame.hl.events import read_events
from agentbench_frame.hl.probe import EmittedAction


# ---------- shared stubs (mirror test_controller.py) ----------

def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text("THRESHOLD = 10\n", encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    return ws


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        spec_id="bench-v1", opponents=("rank01",),
        pairs=1, seats="0", timeout=5.0, notes={"map": "mapconf2.map"},
    )


def _ref_set() -> ReferenceStateSet:
    s = ReferenceSample(
        observation={"round": 1, "inturn": 0},
        legal_actions={"attack": [], "move": [True] + [False]*7,
                       "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
    )
    return ReferenceStateSet(spec_id="bench-v1", samples=(s,))


@dataclass
class _StubEvalResult:
    run_dir: Path
    summary: dict
    matches: list
    error_count: int


def _result(win_rate, avg_rank=None, status="complete"):
    agg = {"win_rate": win_rate}
    if avg_rank is not None:
        agg["avg_rank"] = avg_rank
    return _StubEvalResult(
        run_dir=Path("/tmp"),
        summary={"win_rate": win_rate, "evaluation_status": status,
                 "lostspace": {"aggregate": agg}},
        matches=[], error_count=0,
    )


class _StubEvaluator:
    _queue: List = []
    calls = 0
    def __init__(self, *a, **k): pass
    def evaluate(self):
        _StubEvaluator.calls += 1
        return _StubEvaluator._queue.pop(0) if _StubEvaluator._queue else \
            _result(None, status="incomplete")


def _eval_factory(results):
    _StubEvaluator._queue = list(results)
    _StubEvaluator.calls = 0
    return _StubEvaluator


class _StubProbe:
    _queue: List = []
    def __init__(self, cmd, cwd, timeout): pass
    def probe_set(self, samples):
        emitted = _StubProbe._queue.pop(0) if _StubProbe._queue else []
        return [EmittedAction(primitive=e, out_of_support=False,
                              sample_index=i) for i, e in enumerate(emitted)]
    def close(self): pass


def _probe_factory(per_version_emissions):
    _StubProbe._queue = list(per_version_emissions)
    return _StubProbe


def _controller(tmp_path, *, runner, eval_results, probe_emissions=None):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    v0 = cb.snapshot(parent_version_id=None)
    ctrl = HLIterationController(
        codebase=cb, runner=runner, spec=_spec(), reference=_ref_set(),
        run_id="r", events_path=tmp_path / "e.jsonl",
        evaluator_factory=_eval_factory(eval_results),
        probe_factory=_probe_factory(probe_emissions or []),
        stage_root=tmp_path / "stage",
    )
    return ctrl, v0


class _RevertRunner:
    """FakeRunner analogue that simulates the agent calling `revert_to_best`.

    Optionally mutates agent.py to BAD first (to prove the revert overwrites it),
    then invokes context['revert_fn']() — exactly what ApiCodingRunner does when
    the model emits a revert_to_best tool call. Records the context it saw."""
    def __init__(self, *, mutate_first=True):
        self._mutate = mutate_first
        self.context_seen = None

    def run(self, *, workspace, context):
        self.context_seen = context
        if self._mutate:
            (workspace / "agent.py").write_text("BAD = 'broken'\n",
                                                encoding="utf-8")
        rf = context.get("revert_fn")
        if rf is None:
            return AgentRunResult(edit_type="noop", time_s=0.0)
        res = rf()
        ok = bool(res.get("ok"))
        return AgentRunResult(
            edit_type="rollback" if ok else "noop",
            reverted=ok,
            reverted_to=res.get("best_version_id"),
            files_touched=["agent.py"] if ok else [],
            time_s=0.0,
        )


class _CapturingRunner:
    """No-op runner that records the context dict handed to it."""
    def __init__(self):
        self.context_seen = None
    def run(self, *, workspace, context):
        self.context_seen = context
        return AgentRunResult(edit_type="noop", time_s=0.0)


# ---------- best tracking ----------

def test_best_tracking_updates_on_strict_improvement(tmp_path):
    """best follows the highest win_rate across acts; regression keeps prior best."""
    runner = FakeRunner(transform=lambda w: None, edit_type="noop")
    ctrl, v0 = _controller(tmp_path, runner=runner,
                           eval_results=[_result(0.7, 2.0),    # act1
                                         _result(0.0, 4.0),    # act2 regression
                                         _result(0.8, 1.5)])   # act3 new best
    v = v0
    v = ctrl.act(version_before=v)            # act1 -> wr 0.7
    assert ctrl._best is not None
    assert ctrl._best["win_rate"] == 0.7
    best_after_act1 = ctrl._best["version_id"]
    v = ctrl.act(version_before=v)            # act2 -> wr 0.0 (regression)
    assert ctrl._best["version_id"] == best_after_act1   # unchanged
    v = ctrl.act(version_before=v)            # act3 -> wr 0.8 (new best)
    assert ctrl._best["win_rate"] == 0.8
    assert ctrl._best["version_id"] != best_after_act1


def test_best_tracking_avg_rank_tiebreak(tmp_path):
    """equal win_rate: lower avg_rank wins."""
    runner = FakeRunner(transform=lambda w: None, edit_type="noop")
    ctrl, v0 = _controller(tmp_path, runner=runner,
                           eval_results=[_result(0.7, 2.5),   # act1
                                         _result(0.7, 1.5)])  # act2 same wr, better rank
    v = ctrl.act(version_before=v0)
    best1 = ctrl._best["version_id"]
    v = ctrl.act(version_before=v)
    assert ctrl._best["avg_rank"] == 1.5
    assert ctrl._best["version_id"] != best1


def test_best_tracking_keeps_best_on_equal_metric(tmp_path):
    """identical (win_rate, avg_rank) does NOT replace best (strict improvement only)."""
    runner = FakeRunner(transform=lambda w: None, edit_type="noop")
    ctrl, v0 = _controller(tmp_path, runner=runner,
                           eval_results=[_result(0.7, 2.0), _result(0.7, 2.0)])
    ctrl.act(version_before=v0)
    best1 = ctrl._best["version_id"]
    ctrl.act(version_before=v0)
    assert ctrl._best["version_id"] == best1


# ---------- controller revert act ----------

def test_revert_act_restores_workspace_lineage_and_skips_eval(tmp_path):
    """A revert act: workspace restored to best code, rollback handle lineage
    chains from best, eval is skipped, revert event emitted, best unchanged."""
    # act1: establish best (wr 0.7) with a noop edit so workspace == v0 code.
    runner1 = FakeRunner(transform=lambda w: None, edit_type="noop")
    ctrl, v0 = _controller(tmp_path, runner=runner1,
                           eval_results=[_result(0.7, 2.0),   # act1 -> best
                                         _result(0.9, 1.0)])  # act2 (unused: skipped)
    v_after_act1 = ctrl.act(version_before=v0)
    assert ctrl._best["win_rate"] == 0.7
    best_vid = ctrl._best["version_id"]
    best_hash = ctrl._best["content_hash"]
    eval_calls_after_act1 = _StubEvaluator.calls
    best_code = (ctrl.codebase.root / "agent.py").read_text()

    # act2: the agent reverts (RevertRunner mutates to BAD then calls revert_fn).
    ctrl.runner = _RevertRunner(mutate_first=True)
    rollback = ctrl.act(version_before=v_after_act1)

    # workspace restored to best code, not BAD
    assert (ctrl.codebase.root / "agent.py").read_text() == best_code
    # rollback handle lineage
    assert rollback.content_hash == best_hash
    assert rollback.parent_version_id == best_vid
    assert rollback.edit_type == "rollback"
    # eval skipped (no new matches run)
    assert _StubEvaluator.calls == eval_calls_after_act1
    # best unchanged by the revert
    assert ctrl._best["version_id"] == best_vid
    # revert event emitted
    evs = [e for e in read_events(ctrl._events.path)
           if e["event_type"] == "revert"]
    assert len(evs) == 1
    assert evs[0]["to_version_id"] == best_vid


def test_revert_with_no_best_returns_not_ok(tmp_path):
    """Before any best exists, revert_fn reports ok=False (agent can still edit)."""
    ctrl, v0 = _controller(tmp_path, runner=_CapturingRunner(),
                           eval_results=[_result(0.7, 2.0)])
    # build the revert closure the controller would inject; best is None pre-act1
    assert ctrl._best is None
    fn = ctrl._make_revert_fn(version_before=v0)
    res = fn()
    assert res["ok"] is False


# ---------- context injection ----------

def test_context_carries_best_version(tmp_path):
    """After best is established, the next act's context exposes best_version."""
    runner1 = FakeRunner(transform=lambda w: None, edit_type="noop")
    ctrl, v0 = _controller(tmp_path, runner=runner1,
                           eval_results=[_result(0.7, 2.0), _result(0.5, 3.0)])
    ctrl.act(version_before=v0)               # act1 establishes best
    cap = _CapturingRunner()
    ctrl.runner = cap
    ctrl.act(version_before=v0)               # act2 sees context
    assert cap.context_seen is not None
    assert cap.context_seen["best_version"] is not None
    assert cap.context_seen["best_version"]["win_rate"] == 0.7
    assert callable(cap.context_seen["revert_fn"])


def test_context_best_version_none_before_first_eval(tmp_path):
    ctrl, v0 = _controller(tmp_path, runner=_CapturingRunner(),
                           eval_results=[_result(0.7, 2.0)])
    cap = ctrl.runner
    ctrl.act(version_before=v0)
    assert cap.context_seen["best_version"] is None


# ---------- runner tool dispatch ----------

class _ScriptedClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
    def complete(self, *, system, messages, tools, max_tokens, timeout=None):
        self.calls += 1
        return self._responses.pop(0)


def test_runner_revert_terminal_on_success(tmp_path):
    """ApiCodingRunner: a revert_to_best tool call with ok=True ends the act,
    calls revert_fn, marks reverted, applies no edit."""
    (tmp_path / "agent.py").write_text("X = 1\n", encoding="utf-8")
    called = {"n": 0}
    def revert_fn():
        called["n"] += 1
        return {"ok": True, "best_version_id": "vBEST",
                "win_rate": 0.7, "avg_rank": 2.0}
    client = _ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("revert_to_best", {})],
                    usage=Usage(5, 1)),
    ])
    r = ApiCodingRunner(client=client, system_prompt="S", max_turns=4)
    res = r.run(workspace=tmp_path,
                context={"prompt": "p", "revert_fn": revert_fn,
                         "best_version": {"version_id": "vBEST"}})
    assert called["n"] == 1
    assert res.reverted is True
    assert res.reverted_to == "vBEST"
    assert res.edit_type == "rollback"
    assert client.calls == 1           # terminal: no second turn


def test_runner_revert_no_best_continues_to_edit(tmp_path):
    """revert_to_best with ok=False does NOT end the act; the agent can still edit."""
    (tmp_path / "agent.py").write_text("def step():\n    pass\n", encoding="utf-8")
    def revert_fn():
        return {"ok": False, "reason": "no best yet"}
    client = _ScriptedClient([
        LLMResponse(text="", tool_calls=[ToolCall("revert_to_best", {})],
                    usage=Usage(5, 1)),
        LLMResponse(text="done", tool_calls=[ToolCall("edit",
                    {"old_string": "def step():\n    pass\n",
                     "new_string": "def step():\n    return 1\n"})],
                    usage=Usage(7, 2)),
    ])
    r = ApiCodingRunner(client=client, system_prompt="S", max_turns=4)
    res = r.run(workspace=tmp_path,
                context={"prompt": "p", "revert_fn": revert_fn,
                         "best_version": None})
    assert res.reverted is False
    assert (tmp_path / "agent.py").read_text() == "def step():\n    return 1\n"
    assert client.calls == 2
