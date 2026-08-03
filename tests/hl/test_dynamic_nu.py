"""Tests for rolling dynamic ν (--dynamic-nu) in hl/controller.py.

Contract (plan: dynamic-ν):
- ``subsample`` (reference_recorder) round-caps, dedupes near-identical
  states, and spread-selects up to max_count deterministically.
- The controller re-records ν each act from the evaluated version's real
  match traces, prepends the fixed spawn-pinned anchor, swaps ``self.reference``
  for the NEXT act's KL, and announces every swap via a ``reference_refresh``
  event + the next act's feedback (``nu_refreshed``).
- A recording that yields too few usable samples, or no traces at all, keeps
  the previous ν (fallback) — never silently empties the reference set.
- The saturation regression: with dynamic ν, policy KL stays nonzero on every
  edit act (not just the first), because each act compares v_{k-1} vs v_k
  over the states v_{k-1} actually reached.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.controller import HLIterationController
from agentbench_frame.hl.distribution import STATUS_ALIVE
from agentbench_frame.hl.events import read_events
from agentbench_frame.hl.probe import EmittedAction
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceSample, ReferenceStateSet,
)
from agentbench_frame.hl.reference_recorder import subsample
from agentbench_frame.hl.runner import FakeRunner

MV8 = [True] * 8
ZERO_INV = {"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0}


# ---- fixtures ---------------------------------------------------------------

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


def _sample(*, round_, hp, pos, keys=0, status=STATUS_ALIVE,
            transcript=()) -> ReferenceSample:
    return ReferenceSample(
        observation={"round": round_, "inturn": 0, "hp": hp, "keys": keys,
                     "pos": pos},
        legal_actions={"attack": [], "move": MV8, "detect": False,
                       "interprops": []},
        inventory=dict(ZERO_INV), status=status,
        seat=0, opponent="rank01", transcript=transcript,
    )


def _seed_ref(n_anchor: int) -> ReferenceStateSet:
    """A spawn-pinned seed reference of ``n_anchor`` samples (the extras-style
    points `_pick_anchor` prefers)."""
    return ReferenceStateSet(spec_id="seed", samples=tuple(
        _sample(round_=1, hp=200, pos=[0, 0, 1],
                transcript=({"type": "id", "id": 0, "birth_pos": [0, 0]},
                            {"type": "roundbegin", "round": 1, "inturn": 0,
                             "status": STATUS_ALIVE, "pos": [0, 0, 1]}))
        for _ in range(n_anchor)
    ))


def _write_trace(path: Path, n_points: int, *, offset_hp: int = 0) -> Path:
    """A real-trace-format JSONL: id frame + ``n_points`` Alive roundbegins
    (each carrying the logic-merged legal_actions projection)."""
    id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
    entries = [{"state": 0, "type": "observation", "player": 0,
                "content": id_frame}]
    for r in range(1, n_points + 1):
        content = {
            "type": "roundbegin", "inturn": 0, "status": STATUS_ALIVE,
            "round": r, "hp": 200 - offset_hp - r, "keys": [0] if r > 1 else [],
            "pos": [0, 0, 1], "attack": [], "move": MV8, "detect": False,
            "interprops": [],
            "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0,
                      "Transport": 0},
        }
        entries.append({"state": 0, "type": "observation", "player": 0,
                        "content": content})
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n",
                    encoding="utf-8")
    return path


@dataclass
class _StubEvalResult:
    run_dir: Path
    summary: dict
    matches: list
    error_count: int


def _eval_result(run_dir: Path, trace_rel: str) -> _StubEvalResult:
    return _StubEvalResult(
        run_dir=run_dir,
        summary={"win_rate": 0.5, "evaluation_status": "complete",
                 "lostspace": {"aggregate": {"win_rate": 0.5}}},
        matches=[{"trace": trace_rel}],
        error_count=0,
    )


class _QueueEvaluator:
    """Evaluates to a queued result; ignores all constructor args."""
    _queue: List = []

    def __init__(self, *a, **k):
        pass

    def evaluate(self):
        return _QueueEvaluator._queue.pop(0)


def _stub_evaluator_factory(results):
    _QueueEvaluator._queue = list(results)
    return _QueueEvaluator


class _RollingProbe:
    """Returns a queued per-sample emission list per probe_set call."""
    _queue: List = []

    def __init__(self, cmd, cwd, timeout):
        pass

    def probe_set(self, samples):
        q = _RollingProbe._queue.pop(0) if _RollingProbe._queue else []
        return [EmittedAction(primitive=q[i] if i < len(q) else ("finish",),
                              out_of_support=False, sample_index=i)
                for i in range(len(samples))]

    def close(self):
        pass


def _stub_probe_factory(per_call):
    _RollingProbe._queue = list(per_call)
    return _RollingProbe


def _controller(tmp_path, *, reference, eval_results, probe_emissions,
                dynamic_nu=True, nu_anchor=1, **kw):
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    return HLIterationController(
        codebase=cb, runner=FakeRunner(transform=kw.pop("transform", lambda w: None),
                                       edit_type=kw.pop("edit_type", "noop")),
        spec=_spec(), reference=reference, run_id="r",
        events_path=tmp_path / "e.jsonl", epsilon=0.1,
        dynamic_nu=dynamic_nu, nu_samples=kw.pop("nu_samples", 10),
        nu_anchor=nu_anchor, nu_max_round=kw.pop("nu_max_round", None),
        evaluator_factory=_stub_evaluator_factory(eval_results),
        probe_factory=_stub_probe_factory(probe_emissions),
        stage_root=tmp_path / "stage",
    ), cb


# ---- subsample (reference_recorder) -----------------------------------------

def test_subsample_round_cap_dedupes_and_spreads():
    samples = [
        _sample(round_=1, hp=100, pos=[0, 0, 1]),
        _sample(round_=2, hp=100, pos=[0, 0, 1]),   # near-identical to round 1
        _sample(round_=50, hp=50, pos=[1, 2, 1]),
        _sample(round_=80, hp=30, pos=[2, 1, 1]),
        _sample(round_=120, hp=20, pos=[3, 0, 1]),
    ]
    # round cap drops 80 and 120; dedupe collapses {1,2} (same hp/pos/props)
    capped = subsample(samples, max_round=60)
    assert [s.observation["round"] for s in capped] == [1, 50]

    # even spread: max_count=2 picks the two ends of the (ordered) sequence
    spread = subsample([_sample(round_=i, hp=i, pos=[i, 0, 1])
                        for i in range(1, 6)], max_count=2)
    assert [s.observation["round"] for s in spread] == [1, 5]

    assert subsample([]) == ()


def test_subsample_max_round_none_keeps_all():
    samples = [_sample(round_=i, hp=i, pos=[i, 0, 1]) for i in range(1, 6)]
    assert len(subsample(samples, max_round=None, max_count=None)) == 5


# ---- _pick_anchor -----------------------------------------------------------

def test_pick_anchor_prefers_spawn_pinned():
    ref = ReferenceStateSet(spec_id="s", samples=(
        _sample(round_=1, hp=200, pos=[3, 2, 1]),
        _sample(round_=1, hp=200, pos=[0, 0, 1]),
        _sample(round_=1, hp=200, pos=[0, 0, 1]),
    ))
    anchor = HLIterationController._pick_anchor(ref, 1)
    assert len(anchor) == 1 and anchor[0].observation["pos"] == [0, 0, 1]
    assert HLIterationController._pick_anchor(ref, 0) == ()
    assert HLIterationController._pick_anchor(
        ReferenceStateSet(spec_id="s", samples=()), 2) == ()


# ---- _capture_eval_traces ---------------------------------------------------

def test_capture_eval_traces_resolves_existing_files(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    _write_trace(rd / "a.trace.jsonl", n_points=2)
    result = _StubEvalResult(run_dir=rd, summary={}, matches=[
        {"trace": "a.trace.jsonl"}, {"trace": "missing.trace.jsonl"}],
        error_count=0)
    ctrl, _ = _controller(tmp_path, reference=_seed_ref(1),
                          eval_results=[result], probe_emissions=[])
    ctrl._capture_eval_traces(result)
    assert [p.name for p in ctrl._pending_traces] == ["a.trace.jsonl"]


# ---- rolling ν end-to-end ---------------------------------------------------

def test_rolling_nu_keeps_kl_live_across_acts(tmp_path):
    """The saturation regression: with dynamic ν, KL stays > 0 on every edit
    act (acts 2 and 3), and ν refreshes to fresh states each act."""
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    results = []
    for k in range(3):
        rd = tmp_path / f"run{k}"
        rd.mkdir()
        _write_trace(rd / "m.trace.jsonl", n_points=4, offset_hp=k)
        results.append(_eval_result(rd, "m.trace.jsonl"))

    state = {"n": 0}

    def change(w):
        state["n"] += 1
        (w / "agent.py").write_text(f"THRESHOLD = {10 + state['n']}\n",
                                    encoding="utf-8")

    # Each act probes 2 versions (old, new) over the current ν. ν size after
    # act 1 refresh = 1 anchor + 4 fresh = 5.
    FIVE = 5
    probe_emissions = [
        [("move", 0)] * FIVE, [("move", 1)] * FIVE,   # act 2: v1 vs v2
        [("move", 1)] * FIVE, [("move", 2)] * FIVE,   # act 3: v2 vs v3
    ]
    ctrl = HLIterationController(
        codebase=cb, runner=FakeRunner(transform=change, edit_type="parametrize"),
        spec=_spec(), reference=_seed_ref(1), run_id="r",
        events_path=tmp_path / "e.jsonl", epsilon=0.1,
        dynamic_nu=True, nu_samples=10, nu_anchor=1, nu_max_round=None,
        evaluator_factory=_stub_evaluator_factory(results),
        probe_factory=_stub_probe_factory(probe_emissions),
        stage_root=tmp_path / "stage",
    )
    v = None
    for _ in range(3):
        v = ctrl.act(version_before=v)

    events = read_events(tmp_path / "e.jsonl")
    kls = [e for e in events if e["event_type"] == "policy_kl"]
    assert len(kls) == 2  # acts 2 and 3 (act 1 has nothing to compare)
    for kl in kls:
        assert kl["n_ok"] == FIVE
        assert kl["kl_mean"] > 0
        # each measurement is tagged with the ν it was measured over
        assert kl["nu_spec_id"].startswith("seed-a")

    refreshes = [e for e in events if e["event_type"] == "reference_refresh"]
    assert len(refreshes) == 3
    assert refreshes[0]["fresh"] == 4
    assert refreshes[0]["anchor"] == 1
    assert refreshes[0]["n"] == FIVE
    assert len(refreshes[0]["sources"]) == 1
    # spec_id rolls forward act by act
    assert [r["spec_id"] for r in refreshes] == ["seed-a1", "seed-a2", "seed-a3"]

    # the agent's next prompt learns the reference was refreshed
    assert ctrl._prev_feedback["nu_refreshed"]["n"] == FIVE


def test_dynamic_nu_falls_back_when_no_traces(tmp_path):
    """Empty matches (no traces) -> previous ν kept, no reference_refresh, and
    KL still measured over the seed reference."""
    rd = tmp_path / "run"
    rd.mkdir()
    result = _StubEvalResult(
        run_dir=rd,
        summary={"win_rate": 0.5, "evaluation_status": "complete",
                 "lostspace": {"aggregate": {"win_rate": 0.5}}},
        matches=[], error_count=0,
    )
    ctrl, _ = _controller(
        tmp_path, reference=_seed_ref(1),
        eval_results=[result, result],
        probe_emissions=[[("move", 0)], [("move", 1)]],
    )
    v = ctrl.act(version_before=None)
    assert ctrl.reference.spec_id == "seed"  # fallback: unchanged
    assert ctrl._last_nu_refresh is None
    v = ctrl.act(version_before=v)
    events = read_events(tmp_path / "e.jsonl")
    assert not [e for e in events if e["event_type"] == "reference_refresh"]
    kls = [e for e in events if e["event_type"] == "policy_kl"]
    assert len(kls) == 1
    assert kls[0]["n_ok"] == 1 and kls[0]["kl_mean"] > 0
