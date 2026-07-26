#!/usr/bin/env python3
"""Run v4 vs human rank6 once and export a clean human-readable game log.

Writes:
  agentbench_data/v4_vs_rank6/v4_vs_rank6.game.json   (native replay)
  agentbench_data/v4_vs_rank6/v4_vs_rank6.log.txt     (per-round text log)
  agentbench_data/v4_vs_rank6/v4_vs_rank6.events.txt   (compact key-event log)
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from io import StringIO

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agentbench_frame.lostspace import ladder                  # noqa: E402
from agentbench_frame.lostspace.match import run_match         # noqa: E402
import agentbench_frame.lostspace.replay_view as rv            # noqa: E402

_IS_WIN = sys.platform.startswith("win")
_MODULE = Path(__file__).resolve().parent.parent
_GAMECODE = Path("E:/HL_Agent/AgentBench/backend_sources/corpus/25_lostspace/logic/gamecode_logic")
_OUT = Path("E:/HL_Agent/AgentBenchFramework/agentbench_data/v4_vs_rank6")


def _cmd(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts) if _IS_WIN else " ".join(_quote(p) for p in parts)


def main() -> int:
    logic_py = os.environ.get("LS_LOGIC_PYTHON", r"D:\pymol\python.exe")
    logic = f'cd /d "{_GAMECODE}" && "{logic_py}" main.py'
    v4 = _cmd([sys.executable, str(_MODULE / "candidates" / "v4" / "agent.py")])
    filler = _cmd([sys.executable, str(_MODULE / "baselines" / "sample_ai" / "main.py")])
    rank6_cmd, _ = ladder.launch_command(ladder.resolve("rank=6"))

    _OUT.mkdir(parents=True, exist_ok=True)
    replay = _OUT / "v4_vs_rank6.game.json"
    trace = _OUT / "v4_vs_rank6.trace.jsonl"

    print("=== v4 (seat0) vs human rank6 leehow (seat1), 2 sample fillers ===", flush=True)
    result = run_match(logic, [v4, rank6_cmd, filler, filler], timeout=20.0,
                       replay_path=replay, trace_path=trace)
    ranking = result["ranking"]; scores = result["end_info"]
    v4_rank = ranking.index(0) + 1
    r6_rank = ranking.index(1) + 1
    print(f"ranking (1st..4th seats): {ranking}")
    print(f"v4   (seat 0) -> rank {v4_rank}  (score {scores.get('0')})")
    print(f"rank6(seat 1) -> rank {r6_rank}  (score {scores.get('1')})")

    # Export full per-round text log (render is a generator yielding lines)
    replay_data = json.loads(replay.read_text(encoding="utf-8"))
    text = rv.render(replay_data)
    if not isinstance(text, str):
        text = "\n".join(text)
    (_OUT / "v4_vs_rank6.log.txt").write_text(text, encoding="utf-8")

    # Export compact key-event log from the replay
    events = []
    data = replay_data
    rounds = data[1:-1]
    for ridx, rnd in enumerate(rounds, 1):
        for seat, turn in enumerate(rnd):
            for a in turn:
                t = a.get("type")
                if t in ("getkey", "escaped", "died", "regenerate", "attack",
                         "place_trap", "ai_error", "escape_capsule", "keymachine"):
                    events.append(f"R{ridx:03d} S{seat} {t} " +
                                  json.dumps({k: v for k, v in a.items()
                                              if k != "type"}, ensure_ascii=False))
    ev_text = "\n".join(events) + "\n"
    (_OUT / "v4_vs_rank6.events.txt").write_text(ev_text, encoding="utf-8")

    print(f"\nreplay JSON : {replay}")
    print(f"full log     : {_OUT / 'v4_vs_rank6.log.txt'}")
    print(f"event log    : {_OUT / 'v4_vs_rank6.events.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
