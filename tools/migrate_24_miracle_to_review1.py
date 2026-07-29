#!/usr/bin/env python3
"""24_miracle offline migration tool — re-generates CI-compatible run output
from the original 32-game session's first-hand evidence.

No hardcoded paths: all I/O via CLI args. No subprocess/Judge/AI calls.

Usage:
    python tools/migrate_24_miracle_to_review1.py \
        --session <path_to_matrix_session> \
        --original-run-dir <path_to_original_Results_run> \
        --output-root <path_to_new_preview_root>

Outputs: <output-root>/data/runs/24_miracle/<agent>/<run_id>/{events.jsonl,summary.json,run.toml}
         <output-root>/migration_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentbench_frame.games.miracle.atomicio import atomic_write_json
from agentbench_frame.games.miracle.match_runner import (
    REPLAY_HEADER_BYTES, _VALID_MAP_VALUES, load_result_json,
    read_replay_info, stream_trace, sha256_file,
)
from agentbench_frame.games.miracle.result import build_seed_provenance
from agentbench_frame.tracking.run import Run

EXPECTED = dict(attempts=32, valid=20, invalid=12, wins=2, losses=18,
                win_rate=0.1, total_steps=18144, total_episodes=20)

# Original execution provenance (NOT migration provenance)
ORIGINAL_FRAMEWORK_HEAD = "4bd67fa"
MIGRATION_SCHEMA_COMMIT = "1a61320"
ORIGINAL_STARTED_AT = 1784650770.0399437
ORIGINAL_FINISHED_AT = 1784651015.0399437


def _same_or_within(child: Path, parent: Path) -> bool:
    """Return whether ``child`` is ``parent`` or lives below it."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _deterministic_event_id(game_id: str, run_id: str) -> str:
    h = hashlib.sha256(f"{game_id}|{run_id}".encode("utf-8")).hexdigest()
    return f"evt_{h[:16]}"


def _build_game_record(work_dir: Path, tag: str, run_id: str,
                       orig_created_iso: str) -> Dict[str, Any]:
    rj_path = work_dir / f"{tag}.result.json"
    trace_path = work_dir / f"{tag}.jsonl"
    replay_path = work_dir / f"{tag}.replay"
    stdout_path = work_dir / f"{tag}.stdout"
    stderr_path = work_dir / f"{tag}.stderr"

    rj_status, rj = load_result_json(rj_path)
    rj = rj or {}
    ts = stream_trace(trace_path)
    ri = read_replay_info(replay_path)

    rr = {"map_type": ri.map_type, "day_time": ri.day_time} if ri.header_valid else None
    replay_sha = sha256_file(replay_path) if replay_path.exists() else None
    judge_block = rj.get("judge") or {}
    ai0_block = rj.get("ai0") or {}
    ai1_block = rj.get("ai1") or {}

    vendor = {"role": "vendor", "source": "result_json_derived",
              "final_returncode": rj.get("run_match_returncode"),
              "cleanup_succeeded": bool(rj.get("cleanup_all_succeeded"))}
    process_cleanup = [vendor]
    for role_name, block in (("judge", judge_block), ("ai0", ai0_block), ("ai1", ai1_block)):
        if isinstance(block, dict):
            row = dict(block)
            row.setdefault("role", role_name)
            row["source"] = "result_json"
            process_cleanup.append(row)

    rank, camp = int(tag.replace("m_rank", "").split("_camp")[0]), \
                int(tag.replace("m_rank", "").split("_camp")[1])

    end_info = rj.get("end_info")
    raw_winner = rj.get("raw_winner")
    if raw_winner is None and isinstance(end_info, dict):
        try:
            s0, s1 = int(end_info["0"]), int(end_info["1"])
            raw_winner = 0 if s0 > s1 else 1
        except (KeyError, ValueError, TypeError):
            pass
    end_received = bool(rj.get("end_info_received"))

    error_type = None
    reason = ""
    normalized_result = "error"
    valid = False
    ai_err = rj.get("ai_error", {})
    if isinstance(ai_err, dict):
        if ai_err.get("ai0"): error_type, reason = "ai_crash", "trace ai_error player 0"
        elif ai_err.get("ai1"): error_type, reason = "ai_crash", "trace ai_error player 1"
        elif rj.get("timeout", {}).get("ai0"): error_type, reason = "ai_timeout", "trace ai_timeout player 0"
        elif rj.get("timeout", {}).get("ai1"): error_type, reason = "ai_timeout", "trace ai_timeout player 1"
    if error_type is None and not end_received:
        if raw_winner is None: error_type, reason = "judge_crash", "no legal end_info produced"
    if error_type is None and raw_winner is None:
        error_type, reason = "no_decisive_winner", "raw_winner is None"
    if error_type is None:
        valid = True
        normalized_result = "win" if raw_winner == camp else "loss"

    scores = rj.get("scores")
    if scores is None and isinstance(end_info, dict):
        scores = {"0": end_info.get("0"), "1": end_info.get("1")}
    s0 = scores.get("0") if isinstance(scores, dict) else None
    s1 = scores.get("1") if isinstance(scores, dict) else None

    return {
        "event": "game", "schema_version": "1.0", "run_id": run_id,
        "event_id": _deterministic_event_id(tag, run_id),
        "created_at": orig_created_iso,
        "timestamp": rj.get("finished_at"),
        "game_id": tag, "rank": rank, "camp": camp, "ifelse_camp": camp,
        "valid": valid, "normalized_result": normalized_result,
        "raw_winner": raw_winner,
        "winner_agent": ("miracle_ifelse" if raw_winner == camp else f"rank{rank:02d}")
                         if raw_winner in (0, 1) else None,
        "scores": {"0": s0, "1": s1} if scores is not None else None,
        "steps": ts.n_ai_operation,
        "realized_randomization": rr,
        "seed": build_seed_provenance(rr),
        "error_type": error_type, "reason": reason,
        "wrapper_timeout": False,
        "ai_crash_player": int("ai0" in reason) if error_type == "ai_crash" else None,
        "judge_crash": bool(error_type == "judge_crash"),
        "normal_cleanup_nonzero": False,
        "result_json_status": rj_status,
        "judge_exit": judge_block.get("final_returncode"),
        "ai0_exit": ai0_block.get("final_returncode"),
        "ai1_exit": ai1_block.get("final_returncode"),
        "replay_sha256": replay_sha,
        "process_cleanup": process_cleanup,
        "timeout": rj.get("timeout"), "timeout_s": 8.0,
        "wrapper_exception": None,
        "vendor_exception": rj.get("exception"),
        "run_match_returncode": rj.get("run_match_returncode"),
        "evidence_paths": {
            "stdout": f"work/{tag}.stdout" if stdout_path.exists() else "",
            "stderr": f"work/{tag}.stderr" if stderr_path.exists() else "",
            "trace": f"work/{tag}.jsonl",
            "replay": f"work/{tag}.replay",
            "result_json": f"work/{tag}.result.json",
        },
        "is_resume": False, "is_rerun": False,
        "started_at": rj.get("started_at"),
        "finished_at": rj.get("finished_at"),
        "duration": rj.get("duration_s"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True, type=Path,
                    help="Path to the original matrix session directory")
    ap.add_argument("--original-run-dir", required=True, type=Path,
                    help="Path to the original Results run directory")
    ap.add_argument("--output-root", required=True, type=Path,
                    help="Path to the output preview root (must not exist or be empty)")
    args = ap.parse_args()

    session: Path = args.session
    orig_run_dir: Path = args.original_run_dir
    out_root: Path = args.output_root
    work = session / "work"

    # --- input validation ---
    if not session.exists():
        print(f"FATAL: session not found: {session}", file=sys.stderr); return 2
    if not orig_run_dir.exists():
        print(f"FATAL: original run dir not found: {orig_run_dir}", file=sys.stderr); return 2
    if out_root.name in {
        "24_miracle_results_migration_preview",
        "24_miracle_results_migration_preview_v2",
    }:
        print("FATAL: output root must be a new preview, not a legacy preview", file=sys.stderr); return 2
    if out_root.exists() and any(out_root.iterdir()):
        print(f"FATAL: output root not empty: {out_root}", file=sys.stderr); return 2
    # A child of an authority input would be just as destructive as an equal
    # path.  Do not guess local defaults and never write under either input.
    try:
        if (_same_or_within(out_root, session)
                or _same_or_within(out_root, orig_run_dir)
                or _same_or_within(out_root, orig_run_dir.parent)):
            print("FATAL: output root must not be inside input paths", file=sys.stderr); return 2
    except OSError:
        pass

    # --- load original run metadata ---
    import re
    def _seg(text, key):
        m = re.search(rf'^\s*{key}\s*=\s*"([^"]*)"', text, re.MULTILINE)
        return m.group(1) if m else None
    run_toml_text = (orig_run_dir / "run.toml").read_text(encoding="utf-8")
    orig_summary = json.loads((orig_run_dir / "summary.json").read_text(encoding="utf-8"))
    run_id = _seg(run_toml_text, "run_id") or orig_summary.get("run_id")
    agent = _seg(run_toml_text, "agent") or orig_summary.get("agent")
    game = _seg(run_toml_text, "game") or orig_summary.get("game") or "24_miracle"
    orig_created = _seg(run_toml_text, "created") or orig_summary.get("created") or ""
    if not run_id:
        print("FATAL: run_id missing", file=sys.stderr); return 2

    out_data_root = out_root / "data"
    out_run_dir = out_data_root / "runs" / game / agent / run_id
    print(f"[migrate] run_id={run_id} created={orig_created!r}")

    # --- load original manifest ---
    orig_manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))

    # --- compose 32 per-game records ---
    plan_ids = [f"m_rank{r:02d}_camp{c}" for r in range(1, 17) for c in (0, 1)]
    records = [_build_game_record(work, gid, run_id, orig_created) for gid in plan_ids]

    # --- hard-assert anchors ---
    n = len(records)
    valid_recs = [r for r in records if r["valid"]]
    invalid_recs = [r for r in records if not r["valid"]]
    wins = [r for r in valid_recs if r["normalized_result"] == "win"]
    losses = [r for r in valid_recs if r["normalized_result"] == "loss"]
    total_steps = sum(int(r["steps"]) for r in valid_recs)
    metrics = dict(attempts=n, valid=len(valid_recs), invalid=len(invalid_recs),
                   wins=len(wins), losses=len(losses),
                   win_rate=round(len(wins) / len(valid_recs), 4) if valid_recs else None,
                   total_steps=total_steps, total_episodes=len(valid_recs))
    print(f"[migrate] metrics={metrics}")
    failed = [f"{k}: expected {v!r}, got {metrics.get(k)!r}"
              for k, v in EXPECTED.items() if metrics.get(k) != v]
    if failed:
        print("FATAL: hard-assert violation", file=sys.stderr)
        for f in failed: print(f"  - {f}", file=sys.stderr)
        return 1

    # --- drive framework Run lifecycle with ORIGINAL execution timestamps ---
    migration_started_at = time.time()
    run = Run.start(
        game=game, agent=agent, run_type="eval",
        data_dir=str(out_data_root), run_id=run_id, append=False,
        created=orig_created, git_commit="",
        started_at=ORIGINAL_STARTED_AT,
        config={
            "matrix": "plan_a_32", "total_attempts": 32,
            "original_framework_head": ORIGINAL_FRAMEWORK_HEAD,
            "migration_schema_commit": MIGRATION_SCHEMA_COMMIT,
            "original_code_hashes": orig_manifest.get("code_hashes", {}),
            "protocol_sha256": orig_manifest.get("protocol_sha256"),
            "migration_started_at": migration_started_at,
        },
    )
    for rec in records:
        run.write("game", **rec)

    recomputed = run.recompute_totals_from_events(game_event_type="game")
    assert recomputed["total_steps"] == EXPECTED["total_steps"]
    assert recomputed["episodes"] == EXPECTED["total_episodes"]

    agg = {
        "total_attempts": n, "valid_games": len(valid_recs),
        "invalid_games": len(invalid_recs), "wins": len(wins), "losses": len(losses),
        "win_rate": EXPECTED["win_rate"],
        "steps_stats": {"n": len(valid_recs), "total": total_steps},
        "small_sample_note": "32 局为小样本；含 invalid；不把对手崩溃包装为 Agent 胜利；win_rate 分母仅 valid games。",
    }
    run.log_h2h({})

    run.finish(
        extra_summary={
            "win_rate": EXPECTED["win_rate"],
            "win_rate_available": (len(valid_recs) > 0),
            "total_episodes": recomputed["episodes"],
            "total_steps": recomputed["total_steps"],
            "wins": recomputed["wins"], "losses": recomputed["losses"],
            "matrix_aggregate": agg, "evaluation_status": "COMPLETE",
        },
        finished_at=ORIGINAL_FINISHED_AT,
    )

    # --- audit JSON (portable: NO absolute paths) ---
    audit = {
        "run_id": run_id,
        "orig_created": orig_created,
        "metrics": metrics,
        "expected_metrics": EXPECTED,
        "hard_asserts_passed": not failed,
        "events_sha256": sha256_file(out_run_dir / "events.jsonl"),
        "summary_sha256": sha256_file(out_run_dir / "summary.json"),
        "run_toml_sha256": sha256_file(out_run_dir / "run.toml"),
        "event_id_service": "deterministic sha256(game_id|run_id)[:16]",
        "timestamp_source": "result-json finished_at (real per-game epoch)",
        "created_at_source": "original run.toml 'created' field",
        "git_commit": "",
        "original_framework_head": ORIGINAL_FRAMEWORK_HEAD,
        "migration_schema_commit": MIGRATION_SCHEMA_COMMIT,
        "original_code_hashes": orig_manifest.get("code_hashes", {}),
        "protocol_sha256": orig_manifest.get("protocol_sha256"),
        "original_started_at": ORIGINAL_STARTED_AT,
        "original_finished_at": ORIGINAL_FINISHED_AT,
        "migration_started_at": migration_started_at,
        "totals_source": "Run.recompute_totals_from_events",
        "path_sanitization": "evidence_paths=session-relative; audit=logical refs only",
    }
    atomic_write_json(out_root / "migration_audit.json", audit, indent=2)
    print(f"[migrate] output: {out_run_dir}")
    print("[migrate] DONE - all hard-asserts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
