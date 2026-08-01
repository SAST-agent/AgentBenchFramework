"""Phase 9: end-to-end HL iteration smoke test.

Integration gate: a real two-act iteration with the FakeRunner against a
frozen 2-opponent spec, verifying the complete
act→version→eval→policy_kl→budget chain in events.jsonl with no missing-score
coercion. The probe is exercised against a REAL subprocess candidate (not the
stub), so the wire protocol is genuinely tested in the loop.

This is the gate: if this passes, the HL interface is end-to-end functional.
Swapping FakeRunner -> ClaudeCodeRunner is then the only step to real HL
iteration.
"""
import json
import os
from pathlib import Path

import pytest

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.controller import HLIterationController
from agentbench_frame.hl.runner import FakeRunner
from agentbench_frame.hl.reference import (
    BenchmarkSpec, ReferenceStateSet, ReferenceSample,
)
from agentbench_frame.hl.events import read_events
from agentbench_frame.hl.probe import ReferenceProbe
from agentbench_frame.hl.adapter import candidate_command


# A real subprocess candidate that speaks the Saiblo wire protocol.
# On its turn it emits a deterministic primitive so the probe records it.
SMOKE_CANDIDATE = r'''
import json, sys

def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4: return None
    n = int.from_bytes(hdr, "big", signed=True)
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))

def send(frame):
    s = json.dumps(frame)
    sys.stdout.buffer.write(len(s).to_bytes(4, "big", signed=True) + s.encode("utf-8"))
    sys.stdout.buffer.flush()

while True:
    msg = read_frame()
    if msg is None: break
    t = msg.get("type")
    if t == "id":
        send({"type": "id", "player_num": 1, "player_list": [1,1,1,1]})
    elif t == "roundbegin":
        # deterministic: emit finish immediately
        send({"type": "action", "action": ["finish"]})
    elif t == "action":
        send({"type": "action", "action": ["finish"]})
'''


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(SMOKE_CANDIDATE, encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    return ws


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        spec_id="smoke-v1", opponents=("rank01", "rank06"),
        pairs=1, seats="0", timeout=5.0, notes={"map": "mapconf2.map"},
    )


def _ref_set() -> ReferenceStateSet:
    # two decision points with 2 legal actions (move + finish) so KL is
    # meaningful (non-degenerate A(s)). Each sample carries a minimal
    # transcript (id + roundbegin) so the probe's transcript-replay path is
    # exercised (the §3 fail-fast rejects samples without transcripts).
    samples = []
    for i in range(2):
        obs = {"round": i + 1, "inturn": 0}
        rb = dict(obs)
        rb.setdefault("type", "roundbegin")
        rb.setdefault("state", i + 1)
        rb.setdefault("status", 0)
        rb.setdefault("hp", 200)
        rb.setdefault("keys", [0])
        rb.setdefault("pos", [0, 0, 1])
        rb.setdefault("tools", {"LandMine": [0, 0], "Sticky": [0, 0],
                                 "Kit": 0, "Transport": 0})
        rb.setdefault("others", [
            {"player_id": 1, "status": 0, "keys": [0], "hp": 200},
            {"player_id": 2, "status": 0, "keys": [0], "hp": 200},
            {"player_id": 3, "status": 0, "keys": [0], "hp": 200},
        ])
        id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
        samples.append(ReferenceSample(
            observation=obs,
            legal_actions={"attack": [], "move": [True] + [False]*7,
                           "detect": False, "interprops": []},
            inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
            status=0, seat=0, opponent="rank01",
            transcript=(id_frame, rb),
        ))
    return ReferenceStateSet(spec_id="smoke-v1", samples=tuple(samples))


def _stub_evaluator_factory(win_rate=0.5):
    from dataclasses import dataclass
    @dataclass
    class R:
        run_dir: Path = Path("/tmp")
        summary: dict = None
        matches: list = None
        error_count: int = 0
    class _Ev:
        def __init__(self, *a, **k): pass
        def evaluate(self):
            return R(summary={"win_rate": win_rate,
                              "evaluation_status": "complete",
                              "lostspace": {"aggregate": {"win_rate": win_rate}}},
                      matches=[], error_count=0)
    return _Ev


def test_e2e_two_act_iteration_full_chain(tmp_path):
    """The gate: two real acts produce a complete, coherent event stream.

    Act 1: version_before=None (initial, no KL).
    Act 2: real version_before -> policy_kl + occupancy_shift emitted.
    Both acts use the REAL probe against a REAL subprocess candidate.
    """
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")

    # Act 1's transform: tweak the candidate (so v1 has different content,
    # but stays syntactically valid — append a comment line, not mid-expression)
    def act1_edit(workspace):
        (workspace / "agent.py").write_text(
            SMOKE_CANDIDATE + "\n# v1\n", encoding="utf-8")

    runner = FakeRunner(transform=act1_edit, edit_type="parametrize")
    ctrl = HLIterationController(
        codebase=cb, runner=runner, spec=_spec(), reference=_ref_set(),
        run_id="smoke", events_path=tmp_path / "events.jsonl",
        evaluator_factory=_stub_evaluator_factory(0.5),
        # real probe against the real subprocess candidate
        probe_factory=ReferenceProbe,
        stage_root=tmp_path / "stage",
    )

    # Act 1: initial version (no prior -> no KL yet)
    v1 = ctrl.act(version_before=None)
    assert v1 is not None
    assert v1.edit_type == "initial"

    # Act 2: real version_before -> full chain including KL
    def act2_edit(workspace):
        (workspace / "agent.py").write_text(
            SMOKE_CANDIDATE + "\n# v2\n", encoding="utf-8")
    ctrl.runner = FakeRunner(transform=act2_edit, edit_type="parametrize")
    v2 = ctrl.act(version_before=v1)
    assert v2 is not None
    assert v2.parent_version_id == v1.version_id

    # Verify the full event chain
    events = read_events(tmp_path / "events.jsonl")
    types = [e["event_type"] for e in events]
    for t in ("agent_act", "version", "eval", "policy_kl",
              "occupancy_shift", "budget"):
        assert t in types, f"missing event type {t}"

    # two acts -> two agent_act events, two budget events
    assert types.count("agent_act") == 2
    assert types.count("budget") == 2

    # the policy_kl event (from act 2) has a raw trace over ν (2 decision pts)
    kl_events = [e for e in events if e["event_type"] == "policy_kl"]
    assert len(kl_events) == 1  # only act 2 emits KL
    kl = kl_events[0]
    assert kl["version_before"] == v1.version_id
    assert kl["version_after"] == v2.version_id
    assert len(kl["local_policy_kl_trace"]) == 2  # 2 reference decision points
    # both versions emit finish on both points -> KL = 0 (no behavior change)
    for v in kl["local_policy_kl_trace"]:
        assert v == pytest.approx(0.0, abs=1e-9)

    # budget: learning scope, 2 acts, unknown tokens None (FakeRunner)
    budgets = [e for e in events if e["event_type"] == "budget"]
    assert budgets[-1]["coding_agent_acts"] == 2
    assert budgets[-1]["scope"] == "learning"
    assert budgets[-1].get("prompt_tokens") is None

    # version events carry content hashes; v2 differs from v1
    vers = [e for e in events if e["event_type"] == "version"]
    assert vers[0]["content_hash"] == v1.content_hash
    assert vers[1]["content_hash"] == v2.content_hash
    assert v1.content_hash != v2.content_hash


def test_e2e_no_missing_score_coercion_on_incomplete_eval(tmp_path):
    """If an eval is incomplete, win_rate stays None through the whole chain
    — never coerced to 0 or a loss (doc §12)."""
    ws = _make_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    v0 = cb.snapshot(parent_version_id=None)
    ctrl = HLIterationController(
        codebase=cb, runner=FakeRunner(transform=lambda w: None, edit_type="noop"),
        spec=_spec(), reference=_ref_set(),
        run_id="smoke2", events_path=tmp_path / "e.jsonl",
        evaluator_factory=_stub_evaluator_factory(None),  # incomplete
        probe_factory=ReferenceProbe, stage_root=tmp_path / "stage",
    )
    # force the stub to report incomplete
    class _Incomplete:
        def __init__(self, *a, **k): pass
        def evaluate(self):
            from dataclasses import dataclass
            @dataclass
            class R:
                run_dir: Path = Path("/tmp")
                summary: dict = None
                matches: list = None
                error_count: int = 0
            return R(summary={"win_rate": None,
                              "evaluation_status": "incomplete"})
    ctrl._evaluator_factory = _Incomplete
    ctrl.act(version_before=v0)
    events = read_events(tmp_path / "e.jsonl")
    eval_ev = next(e for e in events if e["event_type"] == "eval")
    assert eval_ev["evaluation_status"] == "incomplete"
    assert eval_ev.get("win_rate") is None
