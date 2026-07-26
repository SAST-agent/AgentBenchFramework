#!/usr/bin/env python3
"""Run my v4 if-else agent vs human rank6 (leehow / Stra) and export the replay.

1 real 4-player game: seat0=v4, seat1=rank6, seats2&3=sample_ai fillers.
Saves the native replay JSON under agentbench_data and prints the result +
where to view the log.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentbench_frame.lostspace import ladder                  # noqa: E402
from agentbench_frame.lostspace.match import run_match         # noqa: E402

_IS_WIN = sys.platform.startswith("win")
_MODULE = Path(__file__).resolve().parent.parent
_BACKEND = Path("E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic")
_GAMECODE = _BACKEND / "gamecode_logic"
_DATA = Path("E:/HL_Agent/AgentBenchFramework/agentbench_data")


def _cmd(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts) if _IS_WIN else " ".join(parts)


def main() -> int:
    # Logic: must run with the python that has antlr4 (D:/pymol) at cwd gamecode_logic.
    logic_py = os.environ.get("LS_LOGIC_PYTHON", r"D:\pymol\python.exe")
    logic = f'cd /d "{_GAMECODE}" && "{logic_py}" main.py'

    v4 = _cmd([sys.executable, str(_MODULE / "candidates" / "v4" / "agent.py")])
    filler = _cmd([sys.executable, str(_MODULE / "baselines" / "sample_ai" / "main.py")])
    rank6_cmd, _ = ladder.launch_command(ladder.resolve("rank=6"))

    out_root = _DATA / "v4_vs_rank6"
    out_root.mkdir(parents=True, exist_ok=True)
    replay = out_root / "v4_vs_rank6.game.json"
    trace = out_root / "v4_vs_rank6.trace.jsonl"

    ai_commands = [v4, rank6_cmd, filler, filler]  # seat0=v4, seat1=rank6
    print("=== v4 (seat0) vs human rank6 leehow (seat1), 2 fillers ===", flush=True)
    result = run_match(logic, ai_commands, timeout=20.0, replay_path=replay,
                       trace_path=trace)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    ranking = result["ranking"]
    scores = result["end_info"]
    print(f"\nranking (1st..4th seats): {ranking}")
    print(f"scores: {scores}")
    v4_rank = ranking.index(0) + 1
    r6_rank = ranking.index(1) + 1
    print(f"\nv4   (seat 0) finished rank {v4_rank}  (score {scores.get('0')})")
    print(f"rank6(seat 1) finished rank {r6_rank}  (score {scores.get('1')})")
    print(f"\nreplay: {replay}")
    print(f"view:   python -m agentbench_frame.lostspace.replay_view \"{replay}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
