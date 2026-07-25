#!/usr/bin/env python3
"""24_miracle local smoke (阶段6, pre-authorized, AT MOST 4 games total).

Evidence-safety design (this round's fixes):
  * Every run gets a unique session dir under .smoke/sessions/<session_id>/.
    A pre-existing session id is REFUSED — this script NEVER deletes or
    overwrites prior sessions, and NEVER re-runs a successful game.
  * sampleA/sampleB are ASSET dirs, kept separate from session products.
  * A manifest (time, code+asset hashes, python version, auth cap=4) is written
    atomically before any match starts.
  * Group 1 (sampleA vs sampleB, 2 games, camps swapped) must pass a STRICT gate
    before Group 2 (if-else vs sampleB, 2 games, camps swapped) is even considered.
  * Residual processes are checked INDEPENDENTLY by exact PID + psutil
    create_time (never trusting cleanup_succeeded, never killing by name).
  * Full stdout/stderr are written to UTF-8 files in the session dir; the console
    shows a summary. Tool rejections / exceptions are recorded separately and are
    NEVER written as game results.

Run from the repo root with a 3.11+ interpreter, UTF-8 mode:
    PYTHONUTF8=1 py -3.13 tools/miracle_smoke.py
"""
from __future__ import annotations

import json
import platform
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentbench_frame.games.miracle.runner import MiracleEvalRunner  # noqa: E402
from agentbench_frame.games.miracle.smoke_audit import (  # noqa: E402
    build_manifest,
    check_residual_procs,
    ensure_fresh_session,
    group1_strict_clean,
    load_managed_procs_from_result_json,
    make_session_id,
    should_run_group2,
    write_manifest_atomic,
)

SESSIONS_ROOT = REPO / ".smoke" / "sessions"
from agentbench_frame.games.miracle.paths import judge_dir, ifelse_dir, sample_ai_dir
JUDGE = judge_dir()
SAMPLE_AI_SRC = sample_ai_dir()
VENDOR = REPO / "vendor" / "miracle_local" / "run_match.py"
FW_SRC = REPO / "src"
SAMPLE_A = REPO / ".smoke" / "sampleA"
SAMPLE_B = REPO / ".smoke" / "sampleB"
IFELSE = ifelse_dir()
AUTH_CAP = 4

_LOG = None


def log(msg=""):
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(str(msg) + "\n")
        _LOG.flush()


def _attempt_report(att) -> dict:
    rr = att.realized_randomization or {}
    rj_path = att.evidence_paths.get("result_json")
    procs = load_managed_procs_from_result_json(rj_path) if rj_path else []
    res = check_residual_procs(procs) if procs else {"clean": [], "residual": [], "reused": []}
    rec = {
        "game_id": att.game_id,
        "valid": att.valid,
        "normalized_result": att.normalized_result,
        "raw_winner": att.raw_winner,
        "winner_agent": att.winner_agent,
        "scores": att.scores,
        "evaluated_agent_camp": att.evaluated_agent_camp,
        "steps": att.steps,
        "seed": {"requested_seed": None, "effective_seed": None,
                 "reproducible_from_seed": False,
                 "realized_randomization": rr},
        "error_type": att.error_type,
        "reason": att.reason,
        "wrapper_timeout": att.wrapper_timeout,
        "normal_cleanup_nonzero": att.normal_cleanup_nonzero,
        "result_json_status": att.result_json_status,
        "discrepancies": att.discrepancies,
        "evidence_paths": att.evidence_paths,
        "residual_check": {
            "n_identity_procs": len(procs),
            "clean": [(p.pid, p.role) for p in res["clean"]],
            "residual": [(p.pid, p.role) for p in res["residual"]],
            "reused": [(p.pid, p.role) for p in res["reused"]],
        },
    }
    return rec


def _run_group(name, *, agent, opponent, evaluated_dir, opponent_dir, data_root, work_dir,
               timeout, wrapper_timeout_s, prefix):
    log(f"\n===== {name}: {agent} vs {opponent} (2 games, camps swapped) =====")
    runner = MiracleEvalRunner(
        agent=agent, opponent=opponent,
        evaluated_dir=evaluated_dir, opponent_dir=opponent_dir,
        n_games=2, data_dir=str(data_root), judge_dir=JUDGE,
        vendor_script=VENDOR, framework_src=FW_SRC, work_dir=work_dir,
        timeout=timeout, wrapper_timeout_s=wrapper_timeout_s, prefix=prefix,
        config={"group": name, "judge_dir_resolved": str(JUDGE.resolve())},
    )
    summary = runner.run()
    reports = [_attempt_report(att) for att in runner.attempts]
    for rec in reports:
        rr = rec["seed"]["realized_randomization"]
        log(f"  {rec['game_id']}: valid={rec['valid']} result={rec['normalized_result']} "
            f"raw_winner={rec['raw_winner']} camp={rec['evaluated_agent_camp']} "
            f"scores={rec['scores']} steps={rec['steps']} "
            f"map=(mt={rr.get('map_type')},dt={rr.get('day_time')}) "
            f"err={rec['error_type']} residual={rec['residual_check']['residual']}")
    s = {k: summary.get(k) for k in
         ("attempted_games", "valid_games", "invalid_games", "wins", "losses", "draws",
          "win_rate_denominator", "win_rate", "evaluation_status", "total_steps", "attempted_steps")}
    log(f"  summary[{agent}]: {json.dumps(s, ensure_ascii=False)}")
    ok, reasons = group1_strict_clean(runner.attempts, summary)
    log(f"  {name} strict gate: {'PASS' if ok else 'FAIL'}" + ("" if ok else f" -> {reasons}"))
    return {"ok": ok, "reasons": reasons, "summary": s, "reports": reports,
            "run_dir": f"{data_root}/runs/24_miracle/{agent}/<run_id>/"}


def main() -> int:
    global _LOG
    session_id = make_session_id()
    session_dir = ensure_fresh_session(SESSIONS_ROOT, session_id)
    _LOG = open(session_dir / "smoke.full.log", "w", encoding="utf-8")
    data_root = session_dir / "data"
    work_dir = session_dir / "work"
    data_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    log(f"session_id: {session_id}")
    log(f"session_dir: {session_dir}")
    log(f"Judge (authoritative): {JUDGE.resolve()}")
    log(f"vendor runner: {VENDOR}")
    log(f"sample A / B: {SAMPLE_A} / {SAMPLE_B}")
    log(f"if-else bot: {IFELSE}")
    log(f"auth_cap_games: {AUTH_CAP}")
    log("NOTE: the previous smoke attempt was rejected by a 429/tool error before "
        "any subprocess launched; that is NOT a game result.")

    manifest = build_manifest(
        session_id=session_id, auth_cap=AUTH_CAP,
        python_executable=sys.executable, python_version=platform.python_version(),
        code_files=[REPO / "tools" / "miracle_smoke.py", REPO / "src" / "agentbench_frame" / "games" / "miracle" / "runner.py",
                    REPO / "src" / "agentbench_frame" / "games" / "miracle" / "match_runner.py",
                    REPO / "src" / "agentbench_frame" / "games" / "miracle" / "proctree.py",
                    REPO / "src" / "agentbench_frame" / "games" / "miracle" / "result.py",
                    REPO / "src" / "agentbench_frame" / "games" / "miracle" / "driver.py",
                    REPO / "src" / "agentbench_frame" / "games" / "miracle" / "smoke_audit.py",
                    VENDOR],
        asset_files=[JUDGE / "main.py", JUDGE / "Data.json",
                     SAMPLE_AI_SRC / "main.py",
                     IFELSE / "main.py", SAMPLE_A / "main.py", SAMPLE_B / "main.py"],
        groups_planned=[{"group": "GROUP1", "agent": "sampleA", "opponent": "sampleB", "n_games": 2},
                        {"group": "GROUP2", "agent": "miracle_ifelse", "opponent": "sampleB", "n_games": 2,
                         "conditional_on": "GROUP1 strict pass"}],
        notes=["429/tool rejection on prior attempt was not a game result",
               "no session is ever deleted or overwritten; successful games are never re-run"],
    )
    mp = write_manifest_atomic(session_dir, manifest)
    log(f"manifest: {mp}")

    try:
        g1 = _run_group("GROUP1", agent="sampleA", opponent="sampleB",
                        evaluated_dir=SAMPLE_A, opponent_dir=SAMPLE_B,
                        data_root=data_root, work_dir=work_dir,
                        timeout=8.0, wrapper_timeout_s=120.0, prefix="g1")
        (session_dir / "group1.json").write_text(
            json.dumps(g1, ensure_ascii=False, indent=2), encoding="utf-8")

        g2 = None
        if should_run_group2(g1["ok"]):
            log("\n>> GROUP 1 strict pass — proceeding to GROUP 2.")
            g2 = _run_group("GROUP2", agent="miracle_ifelse", opponent="sampleB",
                            evaluated_dir=IFELSE, opponent_dir=SAMPLE_B,
                            data_root=data_root, work_dir=work_dir,
                            timeout=10.0, wrapper_timeout_s=180.0, prefix="g2")
            (session_dir / "group2.json").write_text(
                json.dumps(g2, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            log("\n>> GROUP 1 strict gate FAILED — GROUP 2 NOT started (per spec).")

        log("\n===== residual process check (independent, exact PID + create_time) =====")
        for grp_name, grp in (("GROUP1", g1), ("GROUP2", g2)):
            if grp is None:
                continue
            for rec in grp["reports"]:
                rc = rec["residual_check"]
                log(f"  {rec['game_id']}: identity_procs={rc['n_identity_procs']} "
                    f"clean={len(rc['clean'])} residual={rc['residual']} reused={len(rc['reused'])}")

        overall = g1["ok"] and (g2 is None or g2["ok"]) and (g2 is not None)
        # overall green only if BOTH groups ran and passed; G2-not-started is not green
        log(f"\nSMOKE RESULT: {'BOTH_GROUPS_PASS' if (g1['ok'] and g2 and g2['ok']) else 'NOT_FULLY_PASSING'}")
        return 0 if (g1["ok"] and g2 is not None and g2["ok"]) else 1
    except BaseException as exc:  # noqa: BLE001
        # record the exception SEPARATELY from game results; preserve the session
        tb = traceback.format_exc()
        (session_dir / "exception.log").write_text(tb, encoding="utf-8")
        log(f"\nEXCEPTION (not a game result): {exc!r}")
        log(tb)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
