"""Tests for hl/controller.py — HLIterationController end-to-end act loop.

Contract (plan §controller):
- One act = agent_act → runner.run → snapshot(version_after) → classify edit
  → eval on frozen BenchmarkSpec → probe both versions over ν → policy_kl +
  occupancy_shift → emit version/eval/policy_kl/occupancy_shift/budget events.
- version_after=None on an unreadable/crashed workspace, but the act event
  still persists (doc §12.1).
- Missing eval scores stay missing (incomplete), never coerced to 0 or a loss.
- Budget: learning/eval/total split; unknown tokens stay None.
- Same ν for both versions (Q2 decision).
- The FIRST act (version_before=None) is the initial version: edit_type
  'initial', no policy_kl (nothing to compare against). KL appears from the
  second act onward.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from agentbench_frame.hl.controller import HLIterationController
from agentbench_frame.hl.codebase import HLCodebase, VersionHandle
from agentbench_frame.hl.runner import FakeRunner
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceStateSet, ReferenceSample,
)
from agentbench_frame.hl.events import read_events
from agentbench_frame.hl.probe import EmittedAction


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


def _stub_result(win_rate, eval_status):
    return _StubEvalResult(
        run_dir=Path("/tmp"),
        summary={"win_rate": win_rate, "evaluation_status": eval_status,
                 "lostspace": {"aggregate": {"win_rate": win_rate}}},
        matches=[],
        error_count=0,
    )


class _StubEvaluator:
    """Evaluates to a queued result; ignores all constructor args."""
    _queue: List = []
    def __init__(self, *a, **k): pass
    def evaluate(self):
        return _StubEvaluator._queue.pop(0) if _StubEvaluator._queue else \
            _stub_result(None, "incomplete")


def _stub_evaluator_factory(results):
    _StubEvaluator._queue = list(results)
    return _StubEvaluator


class _StubProbe:
    """Returns a queued list of emitted primitives per probe_set call."""
    _queue: List = []
    def __init__(self, cmd, cwd, timeout): pass
    def probe_set(self, samples):
        emitted = _StubProbe._queue.pop(0) if _StubProbe._queue else []
        return [EmittedAction(primitive=e, out_of_support=False,
                              sample_index=i)
                for i, e in enumerate(emitted)]
    def close(self): pass


def _stub_probe_factory(per_version_emissions):
    _StubProbe._queue = list(per_version_emissions)
    return _StubProbe


def _two_act_controller(tmp_path, *, runner, eval_results, probe_emissions):
    """Controller with v0 pre-snapshotted so the first .act() has a real
    version_before and can emit policy_kl."""
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    # pre-snapshot v0 (the 'before' version)
    v0 = cb.snapshot(parent_version_id=None)
    ctrl = HLIterationController(
        codebase=cb, runner=runner, spec=_spec(), reference=_ref_set(),
        run_id="r", events_path=tmp_path / "e.jsonl",
        evaluator_factory=_stub_evaluator_factory(eval_results),
        probe_factory=_stub_probe_factory(probe_emissions),
        stage_root=tmp_path / "stage",
    )
    return ctrl, v0


def test_first_act_is_initial_no_kl(tmp_path):
    """Act with version_before=None -> edit_type 'initial', no policy_kl."""
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    runner = FakeRunner(transform=lambda w: None, edit_type="noop")
    ctrl = HLIterationController(
        codebase=cb, runner=runner, spec=_spec(), reference=_ref_set(),
        run_id="r", events_path=tmp_path / "e.jsonl",
        evaluator_factory=_stub_evaluator_factory([_stub_result(0.5, "complete")]),
        probe_factory=_stub_probe_factory([[("finish",)], [("finish",)]]),
        stage_root=tmp_path / "stage",
    )
    h = ctrl.act(version_before=None)
    events = read_events(tmp_path / "e.jsonl")
    types = [e["event_type"] for e in events]
    assert "agent_act" in types and "version" in types
    assert "policy_kl" not in types  # nothing to compare against
    v_ev = next(e for e in events if e["event_type"] == "version")
    assert v_ev["edit_type"] == "initial"


def test_second_act_emits_full_chain_including_kl(tmp_path):
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],  # identical -> KL 0
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    types = [e["event_type"] for e in events]
    for t in ("agent_act", "version", "eval", "policy_kl",
              "occupancy_shift", "budget"):
        assert t in types, f"missing event type {t}"


def test_policy_kl_zero_when_identical_primitives(tmp_path):
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    kl_ev = next(e for e in events if e["event_type"] == "policy_kl")
    assert kl_ev["local_policy_kl_trace"] == [pytest.approx(0.0, abs=1e-9)]


def test_policy_kl_positive_when_versions_differ(tmp_path):
    def change(w):
        (w / "agent.py").write_text("THRESHOLD = 99\n", encoding="utf-8")
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=change, edit_type="parametrize"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("move", 0)]],  # differ -> KL > 0
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    kl_ev = next(e for e in events if e["event_type"] == "policy_kl")
    assert kl_ev["local_policy_kl_trace"][0] > 0


def test_incomplete_eval_keeps_score_missing(tmp_path):
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        eval_results=[_stub_result(None, "incomplete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    eval_ev = next(e for e in events if e["event_type"] == "eval")
    assert eval_ev["evaluation_status"] == "incomplete"
    assert eval_ev.get("win_rate") is None


def test_budget_learning_scope_unknown_tokens_none(tmp_path):
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    budget = next(e for e in events if e["event_type"] == "budget")
    assert budget["scope"] == "learning"
    assert budget["coding_agent_acts"] == 1
    assert budget.get("prompt_tokens") is None  # unknown, not 0


def test_version_event_records_content_hash_and_edit_type(tmp_path):
    def change(w):
        (w / "agent.py").write_text("THRESHOLD = 20\n", encoding="utf-8")
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=change, edit_type="parametrize"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    h = ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    v_ev = next(e for e in events if e["event_type"] == "version"
                 and e["version_id"] == h.version_id)
    assert v_ev["content_hash"] == h.content_hash
    assert v_ev["edit_type"] == "parametrize"
    assert v_ev["parent_version_id"] == v0.version_id


def test_version_event_records_session_id(tmp_path):
    """The claude session_id + transcript path flow from the runner through to
    the version event, so each act's claude history is locatable in the run
    output (no mtime guessing)."""
    sid = "cafe1234-0000-1111-2222-333333333333"
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop",
                          session_id=sid),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    h = ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    v_ev = next(e for e in events if e["event_type"] == "version"
                and e["version_id"] == h.version_id)
    assert v_ev["session_id"] == sid
    # transcript_path is None here because the sid isn't a real claude home
    # entry; what matters is the field is present and honest.
    assert v_ev["transcript_path"] is None


def test_act_emits_occupancy_shift(tmp_path):
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    os_ev = next(e for e in events if e["event_type"] == "occupancy_shift")
    assert "shift" in os_ev
    assert os_ev["version_before"] == v0.version_id


# ---- failure visibility (D2) ----

class _FailingRunner:
    """A runner that simulates a coding-agent failure (e.g. timeout).

    Returns edit_type='noop' with failure_reason set, like ClaudeCodeRunner
    does on TimeoutExpired / FileNotFoundError / non-zero exit. Does NOT
    mutate the workspace, so the snapshot is identical to v0.
    """
    def __init__(self, *, failure_reason):
        self._reason = failure_reason
    def run(self, *, workspace, context):
        from agentbench_frame.hl.runner import AgentRunResult
        return AgentRunResult(
            edit_type="noop", failure_reason=self._reason,
            error=self._reason, time_s=0.0,
        )


def test_failed_run_records_failure_reason_in_version_and_budget(tmp_path):
    """A timed-out/failed run must NOT be silently recorded as a clean noop.

    The version and budget events carry failure_reason so the silent-timeout
    failure mode is visible in the research stream.
    """
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=_FailingRunner(failure_reason="claude CLI timed out"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    v_ev = next(e for e in events if e["event_type"] == "version")
    b_ev = next(e for e in events if e["event_type"] == "budget")
    assert v_ev["edit_type"] == "noop"
    assert v_ev["failure_reason"] == "claude CLI timed out"
    assert b_ev["failure_reason"] == "claude CLI timed out"


def test_successful_run_has_no_failure_reason(tmp_path):
    """A clean run (FakeRunner, no failure_reason) omits/nulls the field."""
    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    v_ev = next(e for e in events if e["event_type"] == "version")
    assert v_ev.get("failure_reason") is None


# ---- A2/A3: thread previous measurements into the next prompt ----

class _CapturingRunner:
    """Records the prompt each act and mutates agent.py so versions differ."""
    def __init__(self):
        self.prompts: List[str] = []
    def run(self, *, workspace, context):
        from agentbench_frame.hl.runner import AgentRunResult
        self.prompts.append(context.get("prompt", ""))
        (workspace / "agent.py").write_text(
            f"# act {len(self.prompts)}\n", encoding="utf-8")
        return AgentRunResult(edit_type="parametrize", files_touched=["agent.py"],
                              time_s=0.0)


def _controller_with_context(tmp_path, *, eval_results, probe_emissions):
    """Controller wired with a real ContextBuilder so prompts get built."""
    from agentbench_frame.hl.context import ContextBuilder
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    v0 = cb.snapshot(parent_version_id=None)
    builder = ContextBuilder(codebase=cb, data_root=tmp_path, game="25_lostspace",
                             agent_name="r", spec=_spec())
    runner = _CapturingRunner()
    ctrl = HLIterationController(
        codebase=cb, runner=runner, spec=_spec(), reference=_ref_set(),
        run_id="r", events_path=tmp_path / "e.jsonl",
        evaluator_factory=_stub_evaluator_factory(eval_results),
        probe_factory=_stub_probe_factory(probe_emissions),
        stage_root=tmp_path / "stage", context_builder=builder.build,
    )
    return ctrl, v0, runner


def test_first_act_prompt_has_no_feedback_section(tmp_path):
    ctrl, v0, runner = _controller_with_context(
        tmp_path,
        eval_results=[_stub_result(0.0, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    ctrl.act(version_before=v0)
    assert len(runner.prompts) == 1
    assert "Feedback on your last edit" not in runner.prompts[0]


def test_second_act_prompt_carries_previous_eval_and_noop_callout(tmp_path):
    """A2/A3: act 2's prompt shows act 1's eval + policy_kl, and the explicit
    NO-MEASURABLE-EFFECT callout when the edit changed zero reference decisions
    (kl=0). This is the feedback that was previously measured then hidden."""
    # 2 acts × 2 probes each, all identical primitives -> kl=0 both acts.
    emissions = [[("finish",)], [("finish",)], [("finish",)], [("finish",)]]
    ctrl, v0, runner = _controller_with_context(
        tmp_path,
        eval_results=[_stub_result(0.0, "complete"),
                      _stub_result(0.0, "complete")],
        probe_emissions=emissions,
    )
    v1 = ctrl.act(version_before=v0)
    ctrl.act(version_before=v1)
    assert len(runner.prompts) == 2
    p2 = runner.prompts[1]
    assert "Feedback on your last edit" in p2
    assert "win_rate" in p2
    assert "policy_kl" in p2
    # zero behavior change -> the explicit no-op callout fires
    assert "NO MEASURABLE EFFECT" in p2
    assert "0/" in p2  # "changed the chosen action on 0/N"


def test_second_act_prompt_no_callout_when_edit_changed_behavior(tmp_path):
    """When policy_kl > 0 (the edit changed decisions), the no-op callout is
    suppressed — positive signal only."""
    # act 1: v0 emits finish, v1 emits move -> kl>0
    emissions = [[("finish",)], [("move", 0)], [("finish",)], [("move", 0)]]
    ctrl, v0, runner = _controller_with_context(
        tmp_path,
        eval_results=[_stub_result(0.5, "complete"),
                      _stub_result(0.5, "complete")],
        probe_emissions=emissions,
    )
    v1 = ctrl.act(version_before=v0)
    ctrl.act(version_before=v1)
    p2 = runner.prompts[1]
    assert "Feedback on your last edit" in p2
    assert "NO MEASURABLE EFFECT" not in p2


# ---- B1: curriculum (weak -> strong opponent ladder) ----

def test_curriculum_evaluates_current_tier_and_promotes(tmp_path):
    """B1: with curriculum on, each act evaluates ONLY the current-tier
    opponent; when avg_rank meets the promote threshold the controller advances
    to the next tier for subsequent acts. Stops the loop from throwing a weak
    agent straight at the strongest opponent (guaranteed 0.0)."""
    seen = []

    def factory(*, version, spec, run_id, opponent_names=None):
        seen.append(opponent_names)
        class _Rec:
            def evaluate(self):
                class R:
                    summary = _promo_results.pop(0)
                    matches = []
                    error_count = 0
                    run_dir = tmp_path
                return R()
        return _Rec()

    _promo_results = [
        {"win_rate": 0.0, "evaluation_status": "complete",
         "lostspace": {"aggregate": {"avg_rank": 3.0}}},   # act1: no promote
        {"win_rate": 0.5, "evaluation_status": "complete",
         "lostspace": {"aggregate": {"avg_rank": 1.0}}},   # act2: promote
    ]

    def change(w):
        (w / "agent.py").write_text(f"# {len(seen)}\n", encoding="utf-8")

    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    v0 = cb.snapshot(parent_version_id=None)
    ctrl = HLIterationController(
        codebase=cb, runner=FakeRunner(transform=change, edit_type="parametrize"),
        spec=BenchmarkSpec(spec_id="b", opponents=("rank12", "rank06", "rank01"),
                           pairs=1, seats="0", timeout=5.0),
        reference=_ref_set(), run_id="r", events_path=tmp_path / "e.jsonl",
        evaluator_factory=factory,
        probe_factory=_stub_probe_factory([[("finish",)]] * 4),
        stage_root=tmp_path / "stage", curriculum=True, promote_rank=2.0,
    )
    v1 = ctrl.act(version_before=v0)
    ctrl.act(version_before=v1)
    # Both acts evaluated tier 0 (rank12); promotion happens AFTER act2's eval.
    assert seen[0] == ("rank12",)
    assert seen[1] == ("rank12",)
    assert ctrl._tier == 1   # promoted to rank06 after act2


def test_no_curriculum_passes_none_so_factory_uses_all_opponents(tmp_path):
    """Without curriculum the controller does not filter opponents (None -> the
    factory evaluates the full pool), preserving existing behavior."""
    seen = []

    def factory(*, version, spec, run_id, opponent_names=None):
        seen.append(opponent_names)
        class _Rec:
            def evaluate(self):
                class R:
                    summary = {"win_rate": 0.5, "evaluation_status": "complete",
                               "lostspace": {"aggregate": {"avg_rank": 2.0}}}
                    matches = []
                    error_count = 0
                    run_dir = tmp_path
                return R()
        return _Rec()

    ctrl, v0 = _two_act_controller(
        tmp_path,
        runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        eval_results=[_stub_result(0.5, "complete")],
        probe_emissions=[[("finish",)], [("finish",)]],
    )
    # swap in the recording factory on a curriculum-OFF controller
    ctrl._evaluator_factory = factory
    ctrl.act(version_before=v0)
    assert seen == [None]
