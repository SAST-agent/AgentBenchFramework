#!/usr/bin/env python3
"""LOCAL VERIFICATION COPY of 高翔's ``tools/miracle/run_match.py``.

VENDOR PROVENANCE
  Original source : C:\\Users\\gongh\\Desktop\\AgentBench-gaoxiang\\AgentBench-gaoxiang\\tools\\miracle\\run_match.py
  Original SHA256 : 91d5693651402a3579146181ee6ada5586c103f38ba5f321b92c23a17a7a8ae2
  Authoritative Judge : C:\\Users\\gongh\\Documents\\agentbench\\backend_sources\\corpus\\24_miracle\\logic\\judge_dev_logic
                       (高翔's in-tree Judge copy is byte-identical to it, diff verified.)
  Role : local Windows verification only; the original stays frozen, this is a
         minimally-patched vendored copy used by match_runner.py.

MODIFICATIONS (minimal, cross-platform portability; NO protocol/contract change)
  1. JUDGE_DIR overridable via env MIRACLE_JUDGE_DIR (default unchanged). Formal
     smoke sets it to the authoritative Judge path.
  2. ``terminate()`` (Unix-only ``os.killpg``) replaced by
     ``agentbench_frame.games.miracle.proctree.ProcessTreeManager`` — graceful
     terminate -> grace window -> force-kill tree, by EXACT pid with psutil
     create_time identity check. Works on Windows and POSIX, no name-based kill.
  3. ``spawn`` flags: POSIX ``start_new_session=True``; Windows
     ``CREATE_NEW_PROCESS_GROUP``.
  4. ``read_ai_operation`` no longer uses ``selectors`` (broken on Windows pipes);
     it uses a daemon reader thread + queue with timeout — cross-platform.
  5. NEW ``--result-json <path>``: writes a machine-readable structured result
     (process metadata, end_info, derived raw_winner, scores, tie flags,
     per-AI timeout/error, trace/replay paths, cleanup status, timings, any
     exception). match_runner.py reads THIS file, not the human-readable stdout.
  6. Per-process metadata (pid, started_at, natural_exit, natural_returncode,
     termination_requested, termination_reason, final_returncode, forced_kill,
     cleanup_succeeded, identity_confirmed) is captured so the adapter can tell a
     post-end_info cleanup-kill of an idling AI apart from a strategy crash.

Everything else is byte-faithful to the original protocol bridge.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

# --- make the framework's proctree importable when launched as a subprocess --- #
_FW_SRC = os.environ.get("MIRACLE_FRAMEWORK_SRC")
if _FW_SRC and _FW_SRC not in sys.path:
    sys.path.insert(0, _FW_SRC)
from agentbench_frame.games.miracle.proctree import ProcessTreeManager  # noqa: E402
from agentbench_frame.games.miracle.entry import resolve_ai_command  # noqa: E402
from agentbench_frame.games.miracle.atomicio import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JUDGE_DIR = ROOT / "backend_sources/corpus/24_miracle/logic/judge_dev_logic"
JUDGE_DIR = Path(os.environ.get("MIRACLE_JUDGE_DIR") or DEFAULT_JUDGE_DIR)


class ProtocolError(RuntimeError):
    pass


# ---- low-level framed I/O (unchanged protocol) --------------------------- #
def read_exact(stream, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("unexpected EOF")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_judge_input(proc: subprocess.Popen, obj: dict) -> None:
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    proc.stdin.write(struct.pack(">i", len(payload)) + payload)
    proc.stdin.flush()


def read_judge_frame(proc: subprocess.Popen):
    header = read_exact(proc.stdout, 8)
    length, goal = struct.unpack(">ii", header)
    if length < 0:
        raise ProtocolError(f"negative judge frame length: {length}")
    payload = read_exact(proc.stdout, length)
    return goal, json.loads(payload.decode("utf-8"))


def write_ai_state(proc: subprocess.Popen, payload: str) -> None:
    proc.stdin.write(payload.encode("utf-8"))
    proc.stdin.flush()


def read_ai_operation(proc: subprocess.Popen, timeout: float):
    """Read one AI operation with a timeout. Cross-platform: a daemon reader
    thread + queue (the original used ``selectors`` which does not work on
    Windows pipes). Raises ``TimeoutError`` on timeout."""
    q: "queue.Queue[tuple]" = queue.Queue()

    def _read():
        try:
            header = read_exact(proc.stdout, 4)
            length = struct.unpack(">i", header)[0]
            if length < 0:
                q.put(("error", ProtocolError(f"negative AI frame length: {length}")))
                return
            payload = read_exact(proc.stdout, length)
            q.put(("ok", json.loads(payload.decode("utf-8"))))
        except BaseException as exc:  # noqa: BLE001 - report any read failure upstream
            q.put(("error", exc))

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    try:
        kind, val = q.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("AI operation timed out")
    if kind == "error":
        raise val
    return val


# resolve_ai_command is imported from agentbench_frame.games.miracle.entry
# (cross-platform: Windows main.exe / POSIX ./main / main.py; explicit priority;
# paths with spaces kept whole). See tests/miracle/test_entry.py.


def _creation_flags():
    """Platform-appropriate process-group flag so the tree can be managed."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def spawn(cmd, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_creation_flags(),
    )


def drain_stderr(proc: subprocess.Popen) -> str:
    if proc.stderr is None:
        return ""
    try:
        return proc.stderr.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def player_error(player: int, state: int, error: int = 0) -> dict:
    return {
        "player": -1,
        "content": json.dumps({"error": error, "player": player, "state": state}),
    }


def derive_raw_winner(end_info):
    """Judge end_info = {"0": score0, "1": score1}; winner = 0 if s0>s1 else 1.
    Ties are broken toward player1 (Judge main.py:448-449). Returns (winner, s0, s1, tie)."""
    if not isinstance(end_info, dict) or "0" not in end_info or "1" not in end_info:
        return None, None, None, False
    try:
        s0, s1 = int(end_info["0"]), int(end_info["1"])
    except (TypeError, ValueError):
        return None, None, None, False
    tie = (s0 == s1)
    return (0 if s0 > s1 else 1), s0, s1, tie


# ---- match entry point --------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-dir", required=True, type=Path)
    parser.add_argument("--p1-dir", required=True, type=Path)
    parser.add_argument("--p0-cmd")
    parser.add_argument("--p1-cmd")
    parser.add_argument("--p0-name", default="player0")
    parser.add_argument("--p1-name", default="player1")
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/miracle_rollouts")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--result-json", type=Path, default=None,
                        help="machine-readable structured result (used by match_runner)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = args.tag or f"{args.p0_name}_vs_{args.p1_name}_{stamp}"
    replay = args.out / f"{tag}.replay"
    trace = args.out / f"{tag}.jsonl"

    mgr = ProcessTreeManager()
    match_started = time.time()
    end_info_received = False
    end_info = None
    timeout_flag = {"ai0": False, "ai1": False}
    ai_error_flag = {"ai0": False, "ai1": False}
    last_state = 0
    run_exc = None

    def log(obj: dict) -> None:
        with trace.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    try:
        judge_p = spawn([sys.executable, "main.py"], JUDGE_DIR)
        judge_mp = mgr.register_popen(judge_p, "judge")
        ai0_p = spawn(resolve_ai_command(args.p0_dir, args.p0_cmd), args.p0_dir)
        ai1_p = spawn(resolve_ai_command(args.p1_dir, args.p1_cmd), args.p1_dir)
        ai0_mp = mgr.register_popen(ai0_p, "ai0")
        ai1_mp = mgr.register_popen(ai1_p, "ai1")
        procs = [ai0_p, ai1_p]
        mps = {"ai0": ai0_mp, "ai1": ai1_mp}
        names = [args.p0_name, args.p1_name]

        write_judge_input(judge_p, {"replay": str(replay), "player_list": [1, 1]})
        log({"kind": "match_start", "players": names, "replay": str(replay),
             "judge_dir": str(JUDGE_DIR.resolve())})

        while True:
            goal, frame = read_judge_frame(judge_p)
            last_state = frame.get("state", last_state)
            log({"kind": "judge_frame", "goal": goal, "frame": frame})

            if frame.get("state") == -1:
                end_info_received = True
                try:
                    end_info = json.loads(frame.get("end_info", "{}"))
                except Exception:
                    end_info = None
                log({"kind": "match_end", "end_info": frame.get("end_info")})
                break

            players = frame.get("player") or []
            contents = frame.get("content") or []
            if not players or not contents:
                continue

            for idx, player in enumerate(players):
                payload = contents[idx]
                write_ai_state(procs[player], payload)
                try:
                    operation = read_ai_operation(procs[player], args.timeout)
                except TimeoutError:
                    timeout_flag[f"ai{player}"] = True
                    log({"kind": "ai_timeout", "player": player, "state": last_state})
                    write_judge_input(judge_p, player_error(player, last_state, error=1))
                    continue
                except Exception as exc:
                    ai_error_flag[f"ai{player}"] = True
                    log({"kind": "ai_error", "player": player, "state": last_state, "error": repr(exc)})
                    write_judge_input(judge_p, player_error(player, last_state, error=0))
                    continue

                log({"kind": "ai_operation", "player": player, "name": names[player], "operation": operation})
                write_judge_input(
                    judge_p,
                    {"player": operation["player"], "content": json.dumps(operation, separators=(",", ":"))},
                )

        returncode = 0
    except BaseException as exc:  # noqa: BLE001 - capture ANY failure (e.g. Judge crash)
        run_exc = repr(exc)
        returncode = 1
    finally:
        # clean up judge + AIs (graceful -> force tree, exact PID, identity-checked)
        mgr.cleanup_all("match-end")
        match_finished = time.time()

        # drain stderr after processes are dead (safe, non-blocking once EOF)
        for idx, p in enumerate([judge_p] + [mp.popen for mp in [ai0_mp, ai1_mp]]):
            err = drain_stderr(p)
            if err:
                who = "judge" if idx == 0 else f"ai{idx-1}"
                log({"kind": "stderr", "who": who, "stderr": err[-12000:]})

        status = {s["role"]: s for s in mgr.status()}
        raw_winner, s0, s1, tie = derive_raw_winner(end_info)
        result = {
            "schema_version": 1,
            "tag": tag,
            "started_at": match_started,
            "finished_at": match_finished,
            "duration_s": round(match_finished - match_started, 3),
            "judge_dir_resolved": str(JUDGE_DIR.resolve()),
            "p0": {"name": args.p0_name, "dir": str(args.p0_dir.resolve())},
            "p1": {"name": args.p1_name, "dir": str(args.p1_dir.resolve())},
            "judge": status.get("judge"),
            "ai0": status.get("ai0"),
            "ai1": status.get("ai1"),
            "end_info_received": end_info_received,
            "end_info": end_info,
            "scores": ({"0": s0, "1": s1} if s0 is not None else None),
            "raw_winner": raw_winner,
            "score_tie": bool(tie),
            "judge_tiebreak_applied": bool(tie),
            "timeout": timeout_flag,
            "ai_error": ai_error_flag,
            "trace_path": str(trace),
            "replay_path": str(replay),
            "cleanup_all_succeeded": bool(mgr.status() and all(s["cleanup_succeeded"] for s in mgr.status())),
            "exception": run_exc,
            "run_match_returncode": returncode,
        }
        if args.result_json is not None:
            atomic_write_json(args.result_json, result)  # temp + fsync + os.replace (atomic)
        # human-readable stdout kept for back-compat (match_runner uses result-json)
        print(json.dumps({"trace": str(trace), "replay": str(replay),
                          "end_info_received": end_info_received}, ensure_ascii=False))

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
