#!/usr/bin/env python3
"""Evaluate v4 vs human rank6 THROUGH the evaluator (writes CI schema).

Writes runs/25_lostspace/cand-v4/{run_id}/{run.toml,summary.json,...} so the
CONVENTIONS CI aggregator picks it up for visualization.
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentbench_frame.lostspace import ladder                  # noqa: E402
from agentbench_frame.lostspace.evaluator import (             # noqa: E402
    LostSpaceEvaluator, Opponent,
)

_IS_WIN = sys.platform.startswith("win")
_MODULE = Path(__file__).resolve().parent.parent
_GAMECODE = Path("E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic")
_DATA = Path(os.environ.get("AGENTBENCH_DATA",
            "E:/HL_Agent/AgentBenchFramework/agentbench_data"))


def _cmd(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts) if _IS_WIN else " ".join(_quote(p) for p in parts)


def main() -> int:
    logic_py = os.environ.get("LS_LOGIC_PYTHON", r"D:\pymol\python.exe")
    logic = f'cd /d "{_GAMECODE}" && "{logic_py}" main.py'
    v4 = _cmd([sys.executable, str(_MODULE / "candidates" / "v4" / "agent.py")])
    filler = _cmd([sys.executable, str(_MODULE / "baselines" / "sample_ai" / "main.py")])
    rank6_cmd, _ = ladder.launch_command(ladder.resolve("rank=6"))

    result = LostSpaceEvaluator(
        logic_command=logic,
        candidate_name="cand-v4",
        candidate_command=v4,
        opponents=[Opponent("rank6_leehow", rank6_cmd)],
        filler_command=filler,
        pairs=1, seats="0", timeout=20.0,
        data_dir=str(_DATA),
        save_replays=True,
    ).evaluate()

    agg = result.summary["lostspace"]["aggregate"]
    print(f"\nrun_dir : {result.run_dir}")
    print(f"win_rate: {result.summary['win_rate']}")
    print(f"avg_rank: {agg.get('avg_rank')}  avg_score: {agg.get('avg_score')}")
    print(f"errors  : {result.error_count}")
    print(f"\nCI schema written under: {result.run_dir}")
    print("  - run.toml + summary.json (CI required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
