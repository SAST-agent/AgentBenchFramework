#!/usr/bin/env python3
"""One-match worker for pool_tournament, run in an isolated subprocess.

The parent (`pool_tournament._run_one`) spawns this per match so a hung match
(a deadlock the per-read TLE doesn't catch) is bounded by a wall-clock timeout:
the parent kills this process's whole tree (logic + 4 AIs) and records the match
as a wall_timeout error, letting the tournament advance instead of hanging
forever on the same seeded 4-subset.

Reads a job.json = {logic_cmd, ai_cmds, timeout, replay_path, result_path},
runs one match.run_match, writes result.json = {ok, ranking/end_info/turns/
winner} or {ok:false, error}.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from agentbench_frame.lostspace import match


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    t0 = time.time()
    try:
        res = match.run_match(
            job["logic_cmd"], job["ai_cmds"], job["timeout"],
            Path(job["replay_path"]), trace_path=None, media_player_seat=None,
        )
        out = {
            "ok": True,
            "ranking": res["ranking"],
            "end_info": res["end_info"],
            "turns": res["turns"],
            "winner": res["winner"],
            "wall_s": round(time.time() - t0, 2),
        }
    except Exception as exc:  # noqa: BLE001
        out = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-400:],
            "wall_s": round(time.time() - t0, 2),
        }
    Path(job["result_path"]).write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
