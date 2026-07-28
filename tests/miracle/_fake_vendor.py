"""Fake vendor runner for match_runner tests. Simulates vendor run_match.py
behaviours WITHOUT the real Judge, so the integration tests are fast, hermetic,
and need none of the Miracle assets.

Accepts the same standard args as the real vendor runner (--p0-dir, --p1-dir,
--p0-name, --p1-name, --timeout, --out, --tag, --result-json) plus a --fake-mode
that selects the simulated behaviour:
  normal        write clean result-json + trace (3 ai_operations) + valid replay, exit 0
  no_result     exit 0 but write NO result-json
  corrupt       write truncated/invalid result-json, exit 0
  nonzero       exit 2 immediately
  hang          spawn a long-lived child then hang the parent (wrapper-timeout test)
  bigio         write several MiB to stdout AND stderr, exit 0
"""
import argparse
import json
import pathlib
import struct
import subprocess
import sys
import time


def _write_replay(path: pathlib.Path, map_type=0, day_time=1):
    path.write_bytes(struct.pack(">7i", 0, 0, 0, map_type, day_time, 0, 0))


def _write_trace(path: pathlib.Path, events):
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _proc(role, cleanup=True, natural=False, term_req=False, rc=0):
    return {"role": role, "cleanup_succeeded": cleanup,
            "natural_exit": natural, "termination_requested": term_req,
            "final_returncode": rc, "forced_kill": False,
            "identity_confirmed": True, "started_at": 0.0, "pid": 0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--p0-dir"); p.add_argument("--p1-dir")
    p.add_argument("--p0-name", default="p0"); p.add_argument("--p1-name", default="p1")
    p.add_argument("--timeout", type=float, default=12.0)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--tag", default="fake")
    p.add_argument("--result-json", type=pathlib.Path, default=None)
    p.add_argument("--fake-mode", default="normal")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    trace = a.out / f"{a.tag}.jsonl"
    replay = a.out / f"{a.tag}.replay"

    def write_result(d):
        if a.result_json is not None:
            a.result_json.parent.mkdir(parents=True, exist_ok=True)
            a.result_json.write_text(json.dumps(d))

    if a.fake_mode == "normal":
        _write_trace(trace, [
            {"kind": "match_start", "players": [a.p0_name, a.p1_name]},
            {"kind": "ai_operation", "player": 0},
            {"kind": "ai_operation", "player": 1},
            {"kind": "ai_operation", "player": 0},
            {"kind": "match_end", "end_info": json.dumps({"0": 5, "1": 2})},
        ])
        _write_replay(replay, 0, 1)
        write_result({
            "schema_version": 1, "tag": a.tag,
            "end_info_received": True, "end_info": {"0": 5, "1": 2},
            "scores": {"0": 5, "1": 2}, "raw_winner": 0,
            "score_tie": False, "judge_tiebreak_applied": False,
            "timeout": {"ai0": False, "ai1": False},
            "ai_error": {"ai0": False, "ai1": False},
            "trace_path": str(trace), "replay_path": str(replay),
            "cleanup_all_succeeded": True, "exception": None,
            "run_match_returncode": 0,
            "judge": _proc("judge"),
            "ai0": _proc("ai0", cleanup=True, natural=True, term_req=False, rc=0),
            "ai1": _proc("ai1", cleanup=True, natural=True, term_req=False, rc=0),
        })
        sys.stdout.write("ok\n")
        return 0

    if a.fake_mode == "no_result":
        return 0  # exit clean, write nothing

    if a.fake_mode == "corrupt":
        if a.result_json is not None:
            a.result_json.parent.mkdir(parents=True, exist_ok=True)
            a.result_json.write_text('{"schema_version": 1, "end_info": {')  # truncated
        return 0

    if a.fake_mode == "nonzero":
        return 2

    if a.fake_mode == "hang":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
        sys.stdout.write(f"CHILD {child.pid}\n")
        sys.stdout.flush()
        time.sleep(3600)
        return 0

    if a.fake_mode == "bigio":
        chunk = b"x" * (1024 * 1024)
        for _ in range(3):
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
            sys.stderr.buffer.write(chunk); sys.stderr.buffer.flush()
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
