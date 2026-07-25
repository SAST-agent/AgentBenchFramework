#!/usr/bin/env python3
"""Closed-loop demo: evaluate candidates/v1, then candidates/v2, then compare.

Runs real LostSpace matches, so it takes several minutes. Tune with env vars:
  LS_PAIRS   (default 1)     games per (opponent x seat)
  LS_SEATS   (default "0")   "all" or a single seat 0..3
  LS_DATA    (default ./agentbench_data)

Usage:
  cd AgentBenchFramework
  python src/agentbench_frame/lostspace/scripts/iterate_demo.py
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Make `agentbench_frame` importable when run as a bare script path.
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentbench_frame.lostspace.evaluator import (  # noqa: E402
    LostSpaceEvaluator,
    Opponent,
)

_IS_WIN = sys.platform.startswith("win")
_HERE = Path(__file__).resolve().parent
_MODULE = _HERE.parent                      # .../agentbench_frame/lostspace
_BACKEND = Path(
    os.environ.get(
        "LS_BACKEND",
        "E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic",
    )
)
_GAMECODE = _BACKEND / "gamecode_logic"


def _cmd(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts) if _IS_WIN else " ".join(
        shlex.quote(p) for p in parts
    )


def _logic_command() -> str:
    cd = f"cd /d {_GAMECODE}" if _IS_WIN else f"cd {_GAMECODE}"
    return f"{cd} && {_cmd([sys.executable, 'main.py'])}"


def _agent_cmd(path: Path) -> str:
    return _cmd([sys.executable, str(path)])


def _evaluate(version: str, agent_path: Path, logic: str, opponent: Opponent,
              pairs: int, seats: str, data_dir: Path) -> dict:
    print(f"\n=== evaluating {version} ({agent_path.name}) ===", flush=True)
    result = LostSpaceEvaluator(
        logic_command=logic,
        candidate_name=f"cand-{version}",
        candidate_command=_agent_cmd(agent_path),
        opponents=[opponent],
        filler_command=_agent_cmd(_MODULE / "baselines" / "sample_ai" / "main.py"),
        pairs=pairs,
        seats=seats,
        timeout=float(os.environ.get("LS_TIMEOUT", "15")),
        data_dir=str(data_dir),
        save_replays=True,
    ).evaluate()
    agg = result.summary["lostspace"]["aggregate"]
    print(
        f"  {version}: win_rate={agg['win_rate']} "
        f"avg_rank={agg['avg_rank']} avg_score={agg['avg_score']} "
        f"({agg['valid_games']}/{agg['attempted_games']} valid, "
        f"{agg['errors']} errors)",
        flush=True,
    )
    print(f"  run_dir: {result.run_dir}", flush=True)
    return {"version": version, "agg": agg, "run_dir": str(result.run_dir)}


def main() -> int:
    pairs = int(os.environ.get("LS_PAIRS", "1"))
    seats = os.environ.get("LS_SEATS", "0")
    data_dir = Path(os.environ.get("LS_DATA", "./agentbench_data"))

    logic = _logic_command()
    opponent = Opponent(
        "random", _agent_cmd(_MODULE / "baselines" / "random_agent.py")
    )

    v1 = _MODULE / "candidates" / "v1" / "agent.py"
    v2 = _MODULE / "candidates" / "v2" / "agent.py"
    if not v1.is_file():
        print(f"missing {v1}", file=sys.stderr)
        return 2
    if not v2.is_file():
        print(
            f"missing {v2} — copy v1 to v2 and edit it, then rerun",
            file=sys.stderr,
        )
        return 2

    results = [
        _evaluate("v1", v1, logic, opponent, pairs, seats, data_dir),
        _evaluate("v2", v2, logic, opponent, pairs, seats, data_dir),
    ]

    print("\n=== comparison ===")
    for r in results:
        a = r["agg"]
        print(
            f"  {r['version']}: win_rate={a['win_rate']} "
            f"avg_rank={a['avg_rank']} avg_score={a['avg_score']}"
        )
    print(
        "\nReplay a v1 game with:\n"
        f"  python -m agentbench_frame.lostspace.replay_view "
        f"{results[0]['run_dir']}/artifacts/*.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
