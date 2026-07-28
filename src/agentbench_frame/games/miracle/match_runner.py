"""Miracle match wrapper: run one game through the (vendored) run_match runner,
then independently cross-verify result-json + trace + Replay and classify the
outcome on the event timeline (not just the final returncode).

Design points (阶段4b-4 spec):
  * Invokes the vendor runner with ``sys.executable -u``, ``shell=False``, an
    explicit arg array, and explicit ``MIRACLE_JUDGE_DIR`` / paths / identities.
  * stdout/stderr are redirected to FILES (never pipes) so huge output cannot
    deadlock or grow without bound in memory.
  * A wrapper-level overall timeout; on expiry the vendor subtree (and its AI
    descendants) is cleaned by exact-PID process-tree kill (proctree).
  * All evidence (stdout/stderr/trace/result-json) is preserved on disk.
  * The vendor runner writes result-json atomically (temp + replace). If it is
    missing (e.g. force-killed mid-write) the attempt is classified
    ``result_json_missing`` — never guessed.
  * Three-way cross-validation: result-json vs trace (streamed) vs Replay. Any
    authoritative conflict -> ``evidence_mismatch``; nothing is silently picked.
  * Returns a ``MatchAttempt``. It does NOT mutate any aggregate; the runner
    (MiracleEvalRunner) decides what to log.

This module is psutil-backed (via proctree); see the ``miracle`` extra.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agentbench_frame.games.miracle.proctree import ProcessTreeManager
from agentbench_frame.games.miracle.result import sha256_file

#: replay header = 7 big-endian signed int32: [0,0,0,map_type,day_time,0,0]
REPLAY_HEADER_BYTES = 28
#: the Judge draws map_type/day_time via random.randint(0,1); anything outside
#: {0,1} means the header is not a valid Miracle replay.
_VALID_MAP_VALUES = {0, 1}


# --------------------------------------------------------------------------- #
# streaming trace stats
# --------------------------------------------------------------------------- #
@dataclass
class TraceStats:
    n_ai_operation: int = 0
    ai_error_players: List[int] = field(default_factory=list)
    ai_timeout_players: List[int] = field(default_factory=list)
    end_info_seen: bool = False
    end_info: Optional[Dict[str, int]] = None
    error_before_end: bool = False


def stream_trace(path) -> TraceStats:
    """Stream a trace JSONL line by line (never load it whole)."""
    ts = TraceStats()
    p = Path(path)
    if not p.exists():
        return ts
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get("kind")
            if kind == "ai_operation":
                ts.n_ai_operation += 1
            elif kind == "ai_error":
                pl = e.get("player")
                if pl is not None:
                    ts.ai_error_players.append(pl)
                if not ts.end_info_seen:
                    ts.error_before_end = True
            elif kind == "ai_timeout":
                pl = e.get("player")
                if pl is not None:
                    ts.ai_timeout_players.append(pl)
                if not ts.end_info_seen:
                    ts.error_before_end = True
            elif kind == "match_end":
                ts.end_info_seen = True
                ei = e.get("end_info")
                if isinstance(ei, str):
                    try:
                        ts.end_info = json.loads(ei)
                    except json.JSONDecodeError:
                        ts.end_info = None
                elif isinstance(ei, dict):
                    ts.end_info = ei
    return ts


# --------------------------------------------------------------------------- #
# result-json loader
# --------------------------------------------------------------------------- #
def load_result_json(path) -> Tuple[str, Optional[dict]]:
    """Return (status, data) where status is 'ok' | 'missing' | 'corrupt'."""
    p = Path(path)
    if not p.exists():
        return ("missing", None)
    try:
        return ("ok", json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return ("corrupt", None)


# --------------------------------------------------------------------------- #
# replay info
# --------------------------------------------------------------------------- #
@dataclass
class ReplayInfo:
    exists: bool = False
    length_ok: bool = False
    header_valid: bool = False
    map_type: Optional[int] = None
    day_time: Optional[int] = None
    sha256: Optional[str] = None


def read_replay_info(path) -> ReplayInfo:
    p = Path(path)
    info = ReplayInfo(exists=p.exists())
    if not info.exists:
        return info
    info.sha256 = sha256_file(p)
    data = p.read_bytes()
    info.length_ok = len(data) >= REPLAY_HEADER_BYTES
    if info.length_ok:
        try:
            vals = struct.unpack(">7i", data[:REPLAY_HEADER_BYTES])
            info.map_type, info.day_time = int(vals[3]), int(vals[4])
            info.header_valid = (
                info.map_type in _VALID_MAP_VALUES and info.day_time in _VALID_MAP_VALUES
            )
        except struct.error:
            info.header_valid = False
    return info


# --------------------------------------------------------------------------- #
# cross-validation (result-json vs trace vs scores)
# --------------------------------------------------------------------------- #
def cross_validate(rj: Optional[dict], ts: TraceStats, ri: ReplayInfo) -> List[str]:
    """Return a list of authoritative-conflict descriptions. Only CONFLICTS are
    reported here (both sides present and disagreeing); missing end_info in the
    trace is handled by the classifier, not treated as a field conflict."""
    discs: List[str] = []
    if rj is None:
        return discs
    if rj.get("schema_version") != 1:
        discs.append("schema_version_unexpected")
    rj_end = rj.get("end_info") if rj.get("end_info_received") else None
    if rj_end is None:
        return discs
    # end_info conflict (only when trace also has one)
    if ts.end_info is not None and ts.end_info != rj_end:
        discs.append("end_info_scores_mismatch_between_result_json_and_trace")
    # scores field vs end_info
    rj_scores = rj.get("scores")
    if rj_scores is not None:
        try:
            if (int(rj_scores.get("0")) != int(rj_end.get("0"))
                    or int(rj_scores.get("1")) != int(rj_end.get("1"))):
                discs.append("scores_field_conflicts_end_info")
        except (TypeError, ValueError, AttributeError):
            discs.append("scores_or_end_info_unparseable")
    # raw_winner vs score rule (0 if s0>s1 else 1; ties -> 1)
    try:
        s0, s1 = int(rj_end.get("0")), int(rj_end.get("1"))
        expected = 0 if s0 > s1 else 1
        if rj.get("raw_winner") != expected:
            discs.append(f"raw_winner_mismatch_expected_{expected}_got_{rj.get('raw_winner')}")
    except (TypeError, ValueError, AttributeError):
        discs.append("end_info_unparseable")
    return discs


# --------------------------------------------------------------------------- #
# timeline classification
# --------------------------------------------------------------------------- #
@dataclass
class Classification:
    normalized_result: str = "error"
    error_type: Optional[str] = None
    valid: bool = False
    reason: str = ""
    ai_crash_player: Optional[int] = None
    ai_timeout_player: Optional[int] = None
    judge_crash: bool = False
    wrapper_timeout: bool = False
    normal_cleanup_nonzero: bool = False
    raw_winner: Optional[int] = None
    winner_agent: Optional[str] = None


def classify(*, rj_status: str, rj: Optional[dict], ts: TraceStats, ri: ReplayInfo,
             discrepancies: List[str], vendor_returncode: int, wrapper_timeout: bool,
             evaluated_agent_camp: int, evaluated_agent: str = "eval",
             opponent: str = "opp") -> Classification:
    """Classify a game by event timeline + end state. Precedence (most severe
    first): wrapper_timeout > result_json missing/corrupt > evidence_mismatch >
    ai_crash > ai_timeout > judge_crash > replay_missing > replay_corrupt > valid."""
    c = Classification()
    agent_at = lambda camp: evaluated_agent if camp == evaluated_agent_camp else opponent

    if wrapper_timeout:
        c.wrapper_timeout = True
        c.error_type = "wrapper_timeout"
        c.reason = f"vendor tree exceeded wrapper timeout (vendor rc={vendor_returncode})"
        return c
    if rj_status == "missing":
        c.error_type = "result_json_missing"
        c.reason = f"no result-json produced (vendor rc={vendor_returncode})"
        return c
    if rj_status == "corrupt":
        c.error_type = "result_json_corrupt"
        c.reason = "result-json unparseable"
        return c
    if discrepancies:
        c.error_type = "evidence_mismatch"
        c.reason = "; ".join(discrepancies)
        return c

    end_received = bool(rj and rj.get("end_info_received"))

    # AI crash: trace ai_error, OR an AI natural-exited before end_info
    if ts.ai_error_players:
        c.error_type = "ai_crash"
        c.ai_crash_player = ts.ai_error_players[0]
        c.reason = f"trace ai_error (player {c.ai_crash_player})"
        return c
    if not end_received:
        for role, idx in (("ai0", 0), ("ai1", 1)):
            p = (rj or {}).get(role) or {}
            if p.get("natural_exit") and not p.get("termination_requested"):
                c.error_type = "ai_crash"
                c.ai_crash_player = idx
                c.reason = f"{role} exited naturally before end_info"
                return c

    if ts.ai_timeout_players:
        c.error_type = "ai_timeout"
        c.ai_timeout_player = ts.ai_timeout_players[0]
        c.reason = f"trace ai_timeout (player {c.ai_timeout_player})"
        return c

    if not end_received:
        c.error_type = "judge_crash"
        c.judge_crash = True
        c.reason = "no legal end_info produced"
        return c

    if not ri.exists:
        c.error_type = "replay_missing"
        c.reason = "replay file absent"
        return c
    if not ri.header_valid:
        c.error_type = "replay_corrupt"
        c.reason = "replay header invalid (length/format/range)"
        return c

    # valid candidate
    raw = rj.get("raw_winner")
    c.raw_winner = raw
    c.winner_agent = agent_at(raw) if raw in (0, 1) else None
    c.valid = True
    c.error_type = None
    if raw == evaluated_agent_camp:
        c.normalized_result = "win"
    elif raw in (0, 1):
        c.normalized_result = "loss"
    else:
        c.normalized_result = "draw"
    # post-end_info cleanup of idling AIs may yield nonzero returncodes; that's normal
    for role in ("ai0", "ai1"):
        p = (rj or {}).get(role) or {}
        if p.get("termination_requested") and p.get("final_returncode") not in (0, None):
            c.normal_cleanup_nonzero = True
    return c


# --------------------------------------------------------------------------- #
# per-game attempt
# --------------------------------------------------------------------------- #
@dataclass
class MatchAttempt:
    game_id: str
    evaluated_agent: str
    opponent: str
    evaluated_agent_camp: int
    valid: bool
    normalized_result: str
    error_type: Optional[str]
    reason: str
    raw_winner: Optional[int]
    winner_agent: Optional[str]
    scores: Optional[Dict[str, int]]
    steps: int
    realized_randomization: Optional[Dict[str, int]]
    result_json_status: str
    discrepancies: List[str]
    ai_crash_player: Optional[int]
    ai_timeout_player: Optional[int]
    judge_crash: bool
    wrapper_timeout: bool
    normal_cleanup_nonzero: bool
    evidence_paths: Dict[str, str]
    collision_detected: bool
    process_cleanup: List[dict]
    vendor_returncode: int
    started_at: float
    finished_at: float
    duration_s: float
    exception: Optional[str]


def run_match_attempt(*, game_id: str, p0_dir, p1_dir, p0_name: str, p1_name: str,
                      judge_dir, work_dir, vendor_script, framework_src,
                      timeout: float = 12.0, wrapper_timeout_s: float = 60.0,
                      evaluated_agent_camp: int = 0,
                      evaluated_agent: str = "ifelse", opponent: str = "rank04",
                      extra_vendor_args: Optional[List[str]] = None,
                      extra_env: Optional[Dict[str, str]] = None,
                      python: Optional[str] = None) -> MatchAttempt:
    """Run ONE game via the vendor runner and return a fully classified attempt.

    Does not mutate any aggregate. All evidence is preserved under work_dir.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable

    # collision handling: never clobber a prior attempt's evidence for this game_id
    result_json = work_dir / f"{game_id}.result.json"
    collision = result_json.exists()
    tag = game_id
    if collision:
        i = 2
        while (work_dir / f"{game_id}__{i}.result.json").exists():
            i += 1
        tag = f"{game_id}__{i}"
        result_json = work_dir / f"{tag}.result.json"
    trace = work_dir / f"{tag}.jsonl"
    replay = work_dir / f"{tag}.replay"
    stdout_file = work_dir / f"{tag}.stdout"
    stderr_file = work_dir / f"{tag}.stderr"

    cmd: List[str] = [
        python, "-u", str(vendor_script),
        "--p0-dir", str(p0_dir), "--p1-dir", str(p1_dir),
        "--p0-name", str(p0_name), "--p1-name", str(p1_name),
        "--timeout", str(timeout),
        "--out", str(work_dir), "--tag", tag,
        "--result-json", str(result_json),
    ]
    if extra_vendor_args:
        cmd.extend(extra_vendor_args)

    env = dict(os.environ)
    env["MIRACLE_JUDGE_DIR"] = str(judge_dir)
    env["MIRACLE_FRAMEWORK_SRC"] = str(framework_src)
    if extra_env:
        env.update(extra_env)

    started = time.time()
    mgr = ProcessTreeManager()
    out_fh = open(stdout_file, "wb")
    err_fh = open(stderr_file, "wb")
    wrapper_timeout = False
    try:
        proc = subprocess.Popen(cmd, shell=False, stdout=out_fh, stderr=err_fh, env=env)
        mgr.register_popen(proc, "vendor")
        try:
            proc.communicate(timeout=wrapper_timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            wrapper_timeout = True
            mgr.cleanup_all("wrapper-timeout")
            try:
                rc = proc.wait(timeout=5)
            except Exception:
                rc = proc.returncode
    except Exception as exc:  # noqa: BLE001
        rc = -1
        _exc = repr(exc)
    else:
        _exc = None
    finally:
        out_fh.close()
        err_fh.close()
    finished = time.time()

    rj_status, rj = load_result_json(result_json)
    ts = stream_trace(trace)
    ri = read_replay_info(replay)
    discs = cross_validate(rj, ts, ri) if rj is not None else []
    c = classify(rj_status=rj_status, rj=rj, ts=ts, ri=ri, discrepancies=discs,
                 vendor_returncode=(rc if isinstance(rc, int) else -1),
                 wrapper_timeout=wrapper_timeout,
                 evaluated_agent_camp=evaluated_agent_camp,
                 evaluated_agent=evaluated_agent, opponent=opponent)

    return MatchAttempt(
        game_id=game_id, evaluated_agent=evaluated_agent, opponent=opponent,
        evaluated_agent_camp=evaluated_agent_camp,
        valid=c.valid, normalized_result=c.normalized_result,
        error_type=c.error_type, reason=c.reason,
        raw_winner=c.raw_winner, winner_agent=c.winner_agent,
        scores=(rj.get("scores") if rj else None),
        steps=ts.n_ai_operation,
        realized_randomization=(
            {"map_type": ri.map_type, "day_time": ri.day_time}
            if ri.header_valid else None
        ),
        result_json_status=rj_status, discrepancies=discs,
        ai_crash_player=c.ai_crash_player, ai_timeout_player=c.ai_timeout_player,
        judge_crash=c.judge_crash, wrapper_timeout=c.wrapper_timeout,
        normal_cleanup_nonzero=c.normal_cleanup_nonzero,
        evidence_paths={
            "stdout": str(stdout_file), "stderr": str(stderr_file),
            "trace": str(trace), "replay": str(replay),
            "result_json": str(result_json),
        },
        collision_detected=collision,
        process_cleanup=mgr.status(),
        vendor_returncode=(rc if isinstance(rc, int) else -1),
        started_at=started, finished_at=finished,
        duration_s=round(finished - started, 3),
        exception=_exc,
    )
