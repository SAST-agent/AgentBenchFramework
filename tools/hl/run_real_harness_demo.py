"""Drive ONE real-harness HL round with an OPERATOR-authored behavioral edit
(not the claude CLI), to verify the probe-fix produces a real policy update
through the real game logic + real reference probe.

Why this exists: the auto-mode classifier blocks launching the autonomous
``claude`` loop, but does NOT block running the real game harness. This script
wires the REAL ``LostSpaceEvaluator`` (real logic subprocess + real ladder
opponents) and the REAL ``ReferenceProbe`` (the fixed per-sample-restart probe)
through ``HLIterationController``, with a ``FakeRunner`` that applies a small,
interpretable strategy edit to ``agent.py``. If the measured ``policy_kl > 0``
after the edit, the measurement pipeline (probe + KL channel) is proven to
register a real policy update on the real game — the capability the checklist
item 5 requires. The LLM-driven run (same pipeline, ``ClaudeCodeRunner``) is
the operator's to launch via ``run_hl_round.sh``.

The edit: disable Kit usage in candidates/v1 — change the Kit-use HP threshold
from ``<= 130`` to ``<= 0`` (never heal). On the reference point where the
candidate holds a Kit and is low-HP (hp=40, Kit in inventory), this flips the
emission from a LEGAL ``tool Kit`` to an (illegal) ``interact Box``. Because
the before-distribution places (1-eps) mass on the legal Kit action while the
after-distribution collapses to uniform (out-of-support), KL > 0 there — a
valid, honestly-measured policy update. (Edits that only flip between two
illegal emissions correctly yield KL=0, since both collapse to uniform —
that's the measurement contract, not a bug.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# src on PYTHONPATH (mirrors cli.main's requirement)
os.environ.setdefault("PYTHONPATH", "src")
sys.path.insert(0, "src")

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.controller import HLIterationController
from agentbench_frame.hl.context import ContextBuilder
from agentbench_frame.hl.reference import BenchmarkSpec, ReferenceStateSet
from agentbench_frame.hl.runner import FakeRunner
from agentbench_frame.lostspace import ladder
from agentbench_frame.lostspace.evaluator import LostSpaceEvaluator, Opponent

GAME = "25_lostspace"


def _default_filler_command() -> str:
    import shlex, subprocess
    sample = (Path("src/agentbench_frame/lostspace/baselines/sample_ai/main.py"))
    parts = [sys.executable, str(sample)]
    if sys.platform.startswith("win"):
        return subprocess.list2cmdline(parts)
    return " ".join(shlex.quote(p) for p in parts)


def _edit_threshold(workspace: Path) -> None:
    """Disable Kit usage: threshold 130 -> 0 (never heal). A real behavioral
    edit that flips a LEGAL tool-Kit emission to an illegal interact-Box on the
    low-HP-with-Kit reference point -> policy_kl > 0 there."""
    p = workspace / "agent.py"
    txt = p.read_text(encoding="utf-8")
    # v1's play(): `if self.get_my_hp() <= 130:` -> `<= 0`
    new = txt.replace("self.get_my_hp() <= 130", "self.get_my_hp() <= 0")
    assert new != txt, "threshold edit did not match — candidate changed?"
    p.write_text(new, encoding="utf-8")


def _edit_threshold_back(workspace: Path) -> None:
    """Second edit: restore 0 -> 130 (re-enables Kit usage; a real edit that
    produces a second KL measurement)."""
    p = workspace / "agent.py"
    txt = p.read_text(encoding="utf-8")
    new = txt.replace("self.get_my_hp() <= 0", "self.get_my_hp() <= 130")
    assert new != txt, "threshold-back edit did not match"
    p.write_text(new, encoding="utf-8")


def main() -> int:
    from agentbench_frame.hl.adapter import candidate_command
    import shlex, subprocess

    backend = "E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic"
    logic_py = "C:/Users/27364/.conda/envs/torchy/python.exe"
    logic_command = f'cd /d "{backend}" && python main.py'
    # rewrite python -> logic_py (antlr4)
    import re
    logic_command = re.sub(r"(?<![\w./\\])python(?:\.exe)?\b", logic_py,
                          logic_command, count=1)

    name = "hl-v2-probefix-demo"
    codebase_root = Path.cwd() / ".hl_codebase" / name
    workspace = codebase_root / "workspace"
    store = codebase_root / "store"
    stage_root = codebase_root / "stage"
    events_path = codebase_root / "events.jsonl"
    import shutil
    if codebase_root.exists():
        shutil.rmtree(codebase_root)
    workspace.mkdir(parents=True)
    store.mkdir(parents=True)
    stage_root.mkdir(parents=True)

    # seed v1 candidate
    src = Path("src/agentbench_frame/lostspace/candidates/v1")
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            dst = workspace / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())

    codebase = HLCodebase(root=workspace, store=store)
    opponents = []
    for sel in ("rank=6", "rank=12"):
        entry = ladder.resolve(sel)
        cmd, _cwd = ladder.launch_command(entry)
        opponents.append(Opponent(name=f"rank{entry.rank:02d}", command=cmd))
    filler = _default_filler_command()
    spec = BenchmarkSpec(
        spec_id=f"hl-{name}",
        opponents=tuple(o.name for o in opponents),
        pairs=2, seats="0", timeout=15.0,
        notes={"map": "mapconf2.map"},
    )
    _nu = os.environ.get("HL_NU", "agentbench_data/reference/nu-v2.json")
    reference = ReferenceStateSet.load(_nu)

    def evaluator_factory(*, version, spec, run_id, opponent_names=None):
        cmd, cwd = candidate_command(version, store=codebase.store,
                                      dest=stage_root / f"eval-{version.version_id}")
        cmd_str = (subprocess.list2cmdline(cmd) if sys.platform.startswith("win")
                   else " ".join(shlex.quote(c) for c in cmd))
        return LostSpaceEvaluator(
            logic_command=logic_command, candidate_name=run_id,
            candidate_command=cmd_str, opponents=opponents, filler_command=filler,
            pairs=spec.pairs, seats=spec.seats, timeout=spec.timeout,
            data_dir=Path("./agentbench_data"), save_replays=True,
        )

    # FakeRunner: act1 = initial (no edit), act2 = threshold edit, act3 = revert.
    # We construct a fresh runner per act with the right transform.
    runners = [
        FakeRunner(transform=lambda w: None, edit_type="initial"),
        FakeRunner(transform=_edit_threshold, edit_type="parametrize"),
        FakeRunner(transform=_edit_threshold_back, edit_type="parametrize"),
    ]

    context_builder = ContextBuilder(
        codebase=codebase, data_root=Path("./agentbench_data"), game=GAME,
        agent_name=name, spec=spec, playback_skill_path=None, timeout=600.0,
        experience=None, consolidate_every=0, max_growth_pct=40.0,
    )

    ctrl = HLIterationController(
        codebase=codebase, runner=runners[0], spec=spec, reference=reference,
        run_id=name, events_path=events_path, epsilon=0.1,
        evaluator_factory=evaluator_factory, stage_root=stage_root,
        context_builder=context_builder.build,
    )

    version = None
    for i, runner in enumerate(runners, 1):
        ctrl.runner = runner  # swap the runner per act
        print(f"[demo] act {i}/{len(runners)} ...", flush=True)
        version = ctrl.act(version_before=version)
        print(f"[demo]   -> {version.version_id if version else None}", flush=True)

    from agentbench_frame.hl.events import read_events
    events = read_events(events_path)
    evals = [e for e in events if e["event_type"] == "eval"]
    kls = [e for e in events if e["event_type"] == "policy_kl"]
    print("\n[demo] === summary ===")
    for e in evals:
        print(f"  eval act={e['act_id']} win_rate={e['win_rate']} "
              f"status={e['evaluation_status']}")
    for e in kls:
        tr = e["local_policy_kl_trace"]
        print(f"  policy_kl act={e['act_id']} trace={tr} "
              f"mean={sum(tr)/len(tr) if tr else None}")
    any_kl = any(any(x > 0 for x in (e["local_policy_kl_trace"] or [])) for e in kls)
    print(f"\n[demo] valid policy update (KL>0 on >=1 decision point): {any_kl}")
    print(f"[demo] events.jsonl: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
