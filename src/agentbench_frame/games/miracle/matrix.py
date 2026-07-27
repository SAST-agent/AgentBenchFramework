"""32-game matrix orchestrator logic for 24_miracle (Plan A).

Pure, unit-tested logic (no Judge / no real match in this module): the fixed
32-attempt plan, atomic+resumable progress, per-game classification wiring,
per-rank audit gate, infrastructure-stop decision, independent PID+create_time
residual check, and final aggregation. The runner that drives real games lives
in ``tools/miracle_matrix.py`` and reuses these functions.

Plan A (frozen): if-else Agent vs rank01–16, each opponent camp0 + camp1 once,
total attempts = 32. game_id = ``m_rank{NN}_camp{C}``; camp = the camp the
if-else Agent plays.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentbench_frame.games.miracle.result import build_seed_provenance
from agentbench_frame.games.miracle.smoke_audit import (
    check_residual_procs, load_managed_procs_from_result_json, make_session_id,
)

#: error_types that STOP the whole matrix (infrastructure failures).
#: ai_crash / ai_timeout are recorded invalid but do NOT stop (continue).
INFRA_STOP_ERRORS = {
    "wrapper_timeout", "result_json_missing", "result_json_corrupt",
    "evidence_mismatch", "replay_missing", "replay_corrupt", "judge_crash",
    "cleanup_failure", "vendor_exception",
}

_GAMEID_RE = re.compile(r"^m_rank(\d{2})_camp([01])$")


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def make_attempt_plan() -> List[Dict[str, Any]]:
    """32 attempts: rank01 camp0, rank01 camp1, ..., rank16 camp1."""
    plan = []
    for rank in range(1, 17):
        for camp in (0, 1):
            plan.append({"rank": rank, "camp": camp,
                         "game_id": f"m_rank{rank:02d}_camp{camp}"})
    return plan


def parse_game_id(game_id: str) -> Optional[Tuple[int, int]]:
    m = _GAMEID_RE.match(game_id or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------- #
# progress (atomic + resumable)
# --------------------------------------------------------------------------- #
def load_progress(path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"attempts": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"attempts": {}}


def write_progress_atomic(path, progress: Dict[str, Any]) -> Path:
    from agentbench_frame.games.miracle.atomicio import atomic_write_json
    return atomic_write_json(path, progress)


def state_of(progress: Dict[str, Any], game_id: str) -> str:
    return progress.get("attempts", {}).get(game_id, {}).get("state", "not_started")


def mark_running(progress: Dict[str, Any], game_id: str) -> None:
    progress.setdefault("attempts", {})[game_id] = {"state": "running", "game_id": game_id}


def mark_done(progress: Dict[str, Any], game_id: str, result: Dict[str, Any]) -> None:
    entry = {"state": "done", "game_id": game_id}
    entry.update(result)
    progress.setdefault("attempts", {})[game_id] = entry


def is_done(progress: Dict[str, Any], game_id: str) -> bool:
    return state_of(progress, game_id) == "done"


def next_incomplete(progress: Dict[str, Any], plan: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for a in plan:
        if not is_done(progress, a["game_id"]):
            return a
    return None


# --------------------------------------------------------------------------- #
# events append (no duplicate game_id)
# --------------------------------------------------------------------------- #
def append_event_atomic(path, event: Dict[str, Any]) -> bool:
    """Append a JSON line; skip if its game_id already present (no rerun/dup)."""
    p = Path(path)
    gid = event.get("game_id")
    existing = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    e = json.loads(line)
                    if "game_id" in e:
                        existing.add(e["game_id"])
                except json.JSONDecodeError:
                    pass
    if gid in existing:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return True


# --------------------------------------------------------------------------- #
# per-game classification
# --------------------------------------------------------------------------- #
def classify_game(att) -> Dict[str, Any]:
    """Map a MatchAttempt (with rank/camp or parseable game_id) to a per-game record."""
    rc = getattr(att, "rank", None), getattr(att, "camp", None)
    if rc[0] is None or rc[1] is None:
        parsed = parse_game_id(getattr(att, "game_id", ""))
        rank = rc[0] if rc[0] is not None else (parsed[0] if parsed else None)
        camp = rc[1] if rc[1] is not None else (parsed[1] if parsed else None)
    else:
        rank, camp = rc
    rr = getattr(att, "realized_randomization", None) or {}
    scores = getattr(att, "scores", None) or {}
    return {
        "game_id": att.game_id,
        "rank": rank,
        "camp": camp,
        "ifelse_camp": camp,
        "valid": bool(att.valid),
        "normalized_result": att.normalized_result,
        "raw_winner": getattr(att, "raw_winner", None),
        "winner_agent": getattr(att, "winner_agent", None),
        "scores": {"0": scores.get("0"), "1": scores.get("1")} if isinstance(scores, dict) else None,
        "steps": int(getattr(att, "steps", 0) or 0),
        "realized_randomization": rr,
        "seed": build_seed_provenance(rr if rr else None),
        "error_type": getattr(att, "error_type", None),
        "reason": getattr(att, "reason", None),
        "wrapper_timeout": bool(getattr(att, "wrapper_timeout", False)),
        "ai_crash_player": getattr(att, "ai_crash_player", None),
        "judge_crash": bool(getattr(att, "judge_crash", False)),
        "normal_cleanup_nonzero": bool(getattr(att, "normal_cleanup_nonzero", False)),
        "result_json_status": getattr(att, "result_json_status", "ok"),
        "evidence_paths": getattr(att, "evidence_paths", {}),
    }


# --------------------------------------------------------------------------- #
# stop decision
# --------------------------------------------------------------------------- #
def should_stop(att) -> Tuple[str, Optional[str]]:
    """Infrastructure failures stop the matrix; AI crash/timeout (invalid) do NOT."""
    if getattr(att, "wrapper_timeout", False):
        return ("stop", "wrapper_timeout")
    et = getattr(att, "error_type", None)
    if et in INFRA_STOP_ERRORS:
        return ("stop", et)
    rjs = getattr(att, "result_json_status", "ok")
    if rjs in ("missing", "corrupt"):
        return ("stop", f"result_json_{rjs}")
    if getattr(att, "discrepancies", None):
        return ("stop", "evidence_mismatch")
    return ("continue", None)


def should_stop_with_residual(att) -> Tuple[str, str]:
    """Like should_stop, plus an INDEPENDENT PID+create_time residual check on
    the result-json's managed processes."""
    decision, reason = should_stop(att)
    if decision == "stop":
        return ("stop", reason)
    rj_path = getattr(att, "evidence_paths", {}).get("result_json")
    procs = load_managed_procs_from_result_json(rj_path) if rj_path else []
    if not procs:
        return ("continue", "no_managed_procs_identity")
    res = check_residual_procs(procs)
    if res["residual"]:
        return ("stop", f"residual_pids_{[(p.pid, p.role) for p in res['residual']]}")
    return ("continue", "clean")


# --------------------------------------------------------------------------- #
# per-rank audit
# --------------------------------------------------------------------------- #
def rank_audit(rank: int, games) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    recs = [g if isinstance(g, dict) else classify_game(g) for g in games]
    if len(recs) != 2:
        reasons.append(f"rank{rank}: expected 2 games, got {len(recs)}")
    camps = sorted(r.get("camp") for r in recs if r.get("camp") is not None)
    if camps != [0, 1]:
        reasons.append(f"rank{rank}: camps={camps} != [0,1]")
    ids = [r.get("game_id") for r in recs]
    if len(set(ids)) != len(ids):
        reasons.append(f"rank{rank}: duplicate game_id {ids}")
    for r in recs:
        if r.get("valid") and r.get("normalized_result") not in ("win", "loss", "draw"):
            reasons.append(f"{r.get('game_id')}: valid but normalized_result={r.get('normalized_result')}")
        if not r.get("valid") and not r.get("error_type"):
            reasons.append(f"{r.get('game_id')}: invalid without error_type")
    return (len(reasons) == 0, reasons)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _safe_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Final matrix aggregate. win_rate = valid_wins / valid_games (null if 0)."""
    valid = [r for r in records if r.get("valid")]
    wins = [r for r in valid if r.get("normalized_result") == "win"]
    losses = [r for r in valid if r.get("normalized_result") == "loss"]
    invalid = [r for r in records if not r.get("valid")]
    valid_games = len(valid)

    # per-rank
    per_rank: Dict[int, Dict[str, Any]] = {}
    for rank in range(1, 17):
        rs = [r for r in records if r.get("rank") == rank]
        rv = [r for r in rs if r.get("valid")]
        rw = [r for r in rv if r.get("normalized_result") == "win"]
        rl = [r for r in rv if r.get("normalized_result") == "loss"]
        per_rank[rank] = {
            "attempts": len(rs), "valid": len(rv), "invalid": len(rs) - len(rv),
            "wins": len(rw), "losses": len(rl),
            "win_rate": (len(rw) / len(rv)) if rv else None,
        }

    # camp split
    camp_split = {}
    for camp in (0, 1):
        cv = [r for r in valid if r.get("camp") == camp]
        cw = [r for r in cv if r.get("normalized_result") == "win"]
        camp_split[camp] = {"valid": len(cv), "wins": len(cw),
                            "win_rate": (len(cw) / len(cv)) if cv else None}

    # score stats (valid games, ifelse score minus opponent score)
    diffs = []
    for r in valid:
        s = r.get("scores") or {}
        s0, s1 = _safe_num(s.get("0")), _safe_num(s.get("1"))
        if s0 is not None and s1 is not None:
            camp = r.get("camp")
            diffs.append(s0 - s1 if camp == 0 else s1 - s0)  # from ifelse perspective
    diffs_sorted = sorted(diffs)
    def median(xs):
        n = len(xs)
        if n == 0:
            return None
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    score_stats = {
        "n": len(diffs),
        "mean_diff": (sum(diffs) / len(diffs)) if diffs else None,
        "median_diff": median(diffs_sorted),
    }

    # steps stats (valid)
    vsteps = [int(r.get("steps", 0) or 0) for r in valid]
    steps_stats = {
        "n": len(vsteps),
        "total": sum(vsteps),
        "mean": (sum(vsteps) / len(vsteps)) if vsteps else None,
        "median": median(sorted(vsteps)),
    }

    # realized randomization distribution (valid games)
    map_dist = {"map_type": {}, "day_time": {}}
    for r in valid:
        rr = r.get("realized_randomization") or {}
        for k in ("map_type", "day_time"):
            v = rr.get(k)
            if v is not None:
                map_dist[k][v] = map_dist[k].get(v, 0) + 1

    # invalid reason distribution
    invalid_reasons: Dict[str, int] = {}
    for r in invalid:
        et = r.get("error_type") or "unknown"
        invalid_reasons[et] = invalid_reasons.get(et, 0) + 1

    return {
        "total_attempts": len(records),
        "valid_games": valid_games,
        "invalid_games": len(invalid),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / valid_games) if valid_games else None,
        "per_rank": per_rank,
        "camp_split": camp_split,
        "score_stats": score_stats,
        "steps_stats": steps_stats,
        "realized_randomization_distribution": map_dist,
        "invalid_reason_distribution": invalid_reasons,
        "small_sample_note": "32 局为小样本；含 invalid；不把对手崩溃包装为 Agent 胜利；win_rate 分母仅 valid games。",
    }
