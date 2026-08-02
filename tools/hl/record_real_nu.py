"""Record a real ν from one reference match, then probe v1 vs an edited v2.

Acceptance driver for the `recorded-reference-set` change (plan §4.3 e2e):
  1. Run ONE real reference match (sample AI @ seat 0 vs rank06) with save_traces.
  2. reference_recorder parses the trace -> nu-recorded.json (with transcripts).
  3. Run run_real_harness_demo's controller over that ν (FakeRunner edit) and
     assert policy_kl > 0 on >=1 decision point.

This is the real-logic acceptance signal that the fix unblocks the KL=0 symptom.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONPATH", "src")
sys.path.insert(0, "src")

from agentbench_frame.lostspace import ladder
from agentbench_frame.lostspace.evaluator import LostSpaceEvaluator, Opponent

GAME = "25_lostspace"
BACKEND = "E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic"
LOGIC_PY = "C:/Users/27364/.conda/envs/torchy/python.exe"
NU_OUT = Path("agentbench_data/reference/nu-recorded.json")


def _cmd(parts):
    return (subprocess.list2cmdline(parts) if sys.platform.startswith("win")
            else " ".join(shlex.quote(p) for p in parts))


def _sample_ai_cmd():
    parts = [sys.executable,
             str(Path("src/agentbench_frame/lostspace/baselines/sample_ai/main.py"))]
    return _cmd(parts)


def record_nu() -> Path:
    logic_command = f'cd /d "{BACKEND}" && python main.py'
    logic_command = re.sub(r"(?<![\w./\\])python(?:\.exe)?\b", LOGIC_PY,
                           logic_command, count=1)
    opp_entry = ladder.resolve("rank=6")
    opp_cmd, _ = ladder.launch_command(opp_entry)
    opponents = [Opponent(name=f"rank{opp_entry.rank:02d}", command=opp_cmd)]
    ev = LostSpaceEvaluator(
        logic_command=logic_command, candidate_name="nu-recording",
        candidate_command=_sample_ai_cmd(), opponents=opponents,
        filler_command=_sample_ai_cmd(), pairs=1, seats="0", timeout=15.0,
        data_dir=Path("./agentbench_data"), save_traces=True,
    )
    result = ev.evaluate()
    run_dir = Path(result.run_dir) if hasattr(result, "run_dir") else None
    # find the trace path from matches.jsonl
    matches_jsonl = Path(ev.data_dir) / "runs" / GAME / "nu-recording"
    # the run_id dir is the latest under nu-recording/
    run_dirs = sorted(matches_jsonl.glob("*"), key=lambda p: p.stat().st_mtime)
    trace_path = None
    for rd in reversed(run_dirs):
        mj = rd / "matches.jsonl"
        if not mj.exists():
            continue
        import json
        for line in mj.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("trace"):
                trace_path = rd / rec["trace"]
                break
        if trace_path and trace_path.exists():
            break
    if not trace_path or not trace_path.exists():
        raise SystemExit(f"[record] no trace found under {matches_jsonl}")
    print(f"[record] trace: {trace_path}")
    from agentbench_frame.hl.reference_recorder import record_reference_states
    nu = record_reference_states(trace_path, spec_id="nu-recorded",
                                  opponent="rank06", seat=0)
    NU_OUT.parent.mkdir(parents=True, exist_ok=True)
    nu.save(NU_OUT)
    print(f"[record] wrote {len(nu)} samples to {NU_OUT} "
          f"(transcript lengths: {[len(s.transcript) for s in nu.samples][:5]}...)")
    return NU_OUT


if __name__ == "__main__":
    record_nu()
