"""Matrix runner: drives the 32-game Plan A using the tested matrix.py logic.

Designed for ONE long-running process (single background runner). Safety:
  * ``--dry-run`` / ``dry_run()`` never calls attempt_fn (no Judge/AI).
  * Unique session; existing session refused (no overwrite/delete).
  * Atomic manifest written before game 1.
  * ``running`` entries are UNCERTAIN_IN_FLIGHT on restart → STOP, never auto-rerun.
  * ``done`` written only after the attempt returned + event landed.
  * valid / AI-invalid (continue, not in win-rate denom) / infra-failure (halt) distinct.
  * Per-rank audit after both camps; PID+create_time residual halts.
  * events.jsonl is the source of truth → summary independently re-computable.

The real attempt_fn wraps ``match_runner.run_match_attempt``; tests inject a fake.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentbench_frame.games.miracle import matrix
from agentbench_frame.games.miracle.matrix import (
    aggregate,
    append_event_atomic,
    classify_game,
    is_done,
    load_progress,
    make_attempt_plan,
    make_session_id,
    mark_done,
    mark_running,
    parse_game_id,
    rank_audit,
    should_stop_with_residual,
    write_progress_atomic,
)
from agentbench_frame.games.miracle.smoke_audit import ensure_fresh_session

# re-export for tests
__all__ = ["MatrixRunner", "make_attempt_plan", "parse_game_id", "mark_running",
           "write_progress_atomic", "has_uncertain"]


def has_uncertain(progress: Dict[str, Any]) -> List[str]:
    """game_ids left in 'running' state — uncertain whether they played."""
    return [gid for gid, e in progress.get("attempts", {}).items() if e.get("state") == "running"]


# --------------------------------------------------------------------------- #
# strict manifest / progress loaders (review#2 严格性收口)
# --------------------------------------------------------------------------- #
# The 32 standard Plan A game_ids. plan_in_manifest must overwhelmingly equal
# this set; used to reject unknown progress entries when no expected_plan is
# available and to derive the known-gid set.
_LEGAL_STATES = ("not_started", "running", "done")


def _known_gids_from_plan(*plans) -> set:
    """Derive the authoritative set of game_ids from one or more plan lists."""
    out: set = set()
    for p in plans:
        if p:
            for a in p:
                if isinstance(a, dict) and a.get("game_id") is not None:
                    out.add(a["game_id"])
    return out


def _load_progress_strict(path) -> Dict[str, Any]:
    """Strict progress.json parse — NEVER swallow errors. Raises if the file is
    present but corrupt/non-dict so resume() cannot accidentally proceed on a
    half-broken progress that the prior verify_session_for_resume already
    flagged (review#2 gap 9: no lenient re-parse after strict verify)."""
    p = Path(path)
    if not p.exists():
        return {"attempts": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        raise RuntimeError(f"progress.json corrupt/unparseable: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"progress.json not a JSON object: got {type(data).__name__}")
    if not isinstance(data.get("attempts", {}), dict):
        raise RuntimeError("progress.attempts is not a dict")
    return data


def _load_manifest_strict(path) -> Dict[str, Any]:
    """Strict manifest.json parse for resume(): raises on corrupt/non-dict so
    a broken manifest cannot silently let a wrong run_id through."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        raise RuntimeError(f"manifest.json corrupt/unparseable: {e}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"manifest.json not a JSON object: got {type(data).__name__}")
    return data


def verify_session_for_resume(session_dir, *,
                              protocol_sha: Optional[str] = None,
                              code_files: Optional[Dict[str, str]] = None,
                              expected_plan: Optional[List[Dict]] = None,
                              expected_ifelse_sha: Optional[str] = None,
                              expected_judge_sha: Optional[str] = None,
                              expected_opponent_shas: Optional[Dict[int, str]] = None,
                              expected_build_shas: Optional[Dict[int, str]] = None,
                              expected_python: Optional[str] = None,
                              expected_platform: Optional[str] = None,
                              expected_control_inputs: Optional[Dict[str, str]] = None) -> Tuple[bool, List[str]]:
    """Full READ-ONLY session verification before resume. Returns (ok, errors).
    Does NOT write or modify anything. If ANY check fails, the session must be
    left byte-for-byte unchanged."""
    import hashlib
    sd = Path(session_dir)
    errs: List[str] = []

    # --- manifest ---
    mp = sd / "manifest.json"
    if not mp.exists():
        return (False, ["manifest missing"])
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return (False, ["manifest corrupt/unparseable"])
    if expected_control_inputs is not None:
        stored_inputs = m.get("control_inputs")
        if not isinstance(stored_inputs, dict):
            errs.append("control input hash mismatch: control_inputs missing")
        else:
            expected_names = set(expected_control_inputs)
            stored_names = set(stored_inputs)
            for name in sorted(expected_names - stored_names):
                errs.append(f"control input hash mismatch: {name} missing")
            for name in sorted(stored_names - expected_names):
                errs.append(f"control input hash mismatch: {name} unexpected")
            for name in sorted(expected_names & stored_names):
                item = stored_inputs[name]
                stored_sha = item.get("sha256") if isinstance(item, dict) else None
                if stored_sha != expected_control_inputs[name]:
                    errs.append(f"control input hash mismatch: {name}")
    if protocol_sha and m.get("protocol_sha256") != protocol_sha:
        errs.append(f"protocol sha mismatch")
    if code_files:
        stored = m.get("code_hashes", {})
        for name, p in code_files.items():
            p = Path(p)
            if not p.exists():
                errs.append(f"code file missing: {name}")
            elif stored.get(name) != hashlib.sha256(p.read_bytes()).hexdigest():
                errs.append(f"code hash mismatch: {name}")
    if m.get("timeout") != 8.0:
        errs.append(f"timeout mismatch: {m.get('timeout')}")
    _wts = m.get("wrapper_timeout_s")
    if _wts is None:
        errs.append("wrapper_timeout_s missing or null")
    elif _wts != 180.0:
        errs.append(f"wrapper_timeout_s mismatch: {_wts}")
    # --- asset hashes (identity frozen at session creation) ---
    if expected_ifelse_sha and m.get("ifelse_sha256") != expected_ifelse_sha:
        errs.append("if-else sha mismatch")
    if expected_judge_sha and m.get("judge_sha256") != expected_judge_sha:
        errs.append("judge sha mismatch")
    if expected_opponent_shas:
        stored_opp = m.get("opponent_archive_sha256", {})
        for rk, v in expected_opponent_shas.items():
            key = f"rank{rk:02d}" if isinstance(rk, int) else str(rk)
            if stored_opp.get(key) != v:
                errs.append(f"opponent archive sha mismatch: {key}")
    if expected_build_shas:
        stored_build = m.get("cpp_build_sha256", {})
        for rk, v in expected_build_shas.items():
            key = f"rank{rk:02d}" if isinstance(rk, int) else str(rk)
            if stored_build.get(key) != v:
                errs.append(f"build sha mismatch: {key}")
    # --- plan_count must equal len(plan) AND (when len-plan fallback) 32 ---
    plan_in_manifest = m.get("plan", [])
    if not isinstance(plan_in_manifest, list):
        errs.append("plan field is not a list")
        plan_in_manifest = []
    plan_len = len(plan_in_manifest)
    declared_count = m.get("plan_count")
    if declared_count is None:
        errs.append("plan_count field missing")
    elif not isinstance(declared_count, int) or declared_count != plan_len:
        errs.append(f"plan_count={declared_count!r} != len(plan)={plan_len}")
    # --- plan: full 32 content + order ---
    if expected_plan:
        if plan_len != len(expected_plan):
            errs.append(f"plan length {plan_len} != {len(expected_plan)}")
        else:
            for i, (got, want) in enumerate(zip(plan_in_manifest, expected_plan)):
                if not isinstance(got, dict):
                    errs.append(f"plan[{i}] not a dict: {got!r}")
                    continue
                if got.get("game_id") != want.get("game_id"):
                    errs.append(f"plan[{i}] game_id {got.get('game_id')} != {want.get('game_id')}")
                if got.get("rank") != want.get("rank"):
                    errs.append(f"plan[{i}] rank {got.get('rank')} != {want.get('rank')}")
                if got.get("camp") != want.get("camp"):
                    errs.append(f"plan[{i}] camp {got.get('camp')} != {want.get('camp')}")
                # reject extra/unexpected keys in the per-attempt plan entry
                _want_keys = set(want.keys())
                _got_keys = set(got.keys())
                _extra = _got_keys - _want_keys
                if _extra:
                    errs.append(f"plan[{i}] unexpected keys: {sorted(_extra)}")
    elif plan_len != 32:
        errs.append(f"plan_count={plan_len} != 32 (len(plan) fallback)")
    # --- session_id MUST exist (review#2 gap 4: missing → reject) ---
    m_sid = m.get("session_id")
    if not m_sid:
        errs.append("session_id missing or empty")
    elif m_sid != sd.name:
        errs.append(f"session_id mismatch: manifest={m_sid} != dir={sd.name}")
    # --- run_id MUST exist ---
    if not m.get("run_id"):
        errs.append("run_id missing")
    # --- runtime identity ---
    if expected_python and m.get("python_version") and m.get("python_version") != expected_python:
        errs.append(f"python version mismatch: {m.get('python_version')} != {expected_python}")
    if expected_platform and m.get("platform") and m.get("platform") != expected_platform:
        errs.append(f"platform mismatch: {m.get('platform')} != {expected_platform}")
    # --- opponent / build / code_hashes: EXACT key-set match (review#2 gap 3) ---
    if expected_opponent_shas is not None:
        stored_opp = m.get("opponent_archive_sha256", {})
        if not isinstance(stored_opp, dict):
            errs.append("opponent_archive_sha256 not a dict")
        else:
            expected_keys = {f"rank{rk:02d}" if isinstance(rk, int) else str(rk) for rk in expected_opponent_shas}
            extra = set(stored_opp.keys()) - expected_keys
            missing = expected_keys - set(stored_opp.keys())
            if extra:
                errs.append(f"opponent_archive_sha256 unexpected/extra keys: {sorted(extra)}")
            if missing:
                errs.append(f"opponent_archive_sha256 missing keys: {sorted(missing)}")
    if expected_build_shas is not None:
        stored_build = m.get("cpp_build_sha256", {})
        if not isinstance(stored_build, dict):
            errs.append("cpp_build_sha256 not a dict")
        else:
            expected_keys = {f"rank{rk:02d}" if isinstance(rk, int) else str(rk) for rk in expected_build_shas}
            extra = set(stored_build.keys()) - expected_keys
            missing = expected_keys - set(stored_build.keys())
            if extra:
                errs.append(f"cpp_build_sha256 unexpected/extra keys: {sorted(extra)}")
            if missing:
                errs.append(f"cpp_build_sha256 missing keys: {sorted(missing)}")
    if code_files is not None:
        stored_code = m.get("code_hashes", {})
        if not isinstance(stored_code, dict):
            errs.append("code_hashes not a dict")
        else:
            expected_keys = set(code_files.keys())
            extra = set(stored_code.keys()) - expected_keys
            missing = expected_keys - set(stored_code.keys())
            if extra:
                errs.append(f"code_hashes unexpected/extra keys: {sorted(extra)}")
            if missing:
                errs.append(f"code_hashes missing keys: {sorted(missing)}")

    # --- progress / events / audit consistency ---
    pp = sd / "progress.json"
    ep = sd / "events.jsonl"
    ad = sd / "audit"
    # STRICT progress parsing (do NOT use lenient load_progress which swallows errors)
    if not pp.exists():
        errs.append("progress.json missing")
        attempts = {}
    else:
        try:
            progress = json.loads(pp.read_text(encoding="utf-8"))
        except Exception:
            errs.append("progress.json corrupt")
            progress = {}
        if not isinstance(progress, dict):
            errs.append("progress.json not a dict")
            progress = {}
        attempts = progress.get("attempts", {})
        if not isinstance(attempts, dict):
            errs.append("progress attempts not a dict")
            attempts = {}
    # events.jsonl — STRICT line-by-line: each non-empty line must be a JSON OBJECT
    event_gids: List[str] = []
    if ep.exists():
        for line_no, line in enumerate(ep.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                errs.append(f"events.jsonl line {line_no}: corrupt (not JSON)")
                continue
            if not isinstance(e, dict):
                errs.append(f"events.jsonl line {line_no}: top-level not a JSON object")
                continue
            if "game_id" in e:
                event_gids.append(e["game_id"])
    # --- progress attempts: known gids + legal state (review#2 gaps 1, 2) ---
    known_gids = _known_gids_from_plan(expected_plan, plan_in_manifest)
    if not known_gids:
        # expected_plan not provided AND plan was rejected; use the std 32.
        known_gids = {a["game_id"] for a in make_attempt_plan()}
    for gid, entry in attempts.items():
        if not isinstance(entry, dict):
            errs.append(f"progress attempt {gid!r}: entry not a dict")
            continue
        if gid not in known_gids:
            errs.append(f"progress entry unknown game_id: {gid}")
            continue
        state = entry.get("state", "not_started")
        if state not in _LEGAL_STATES:
            errs.append(f"progress {gid}: illegal state {state!r}")
    # running → UNCERTAIN
    running = [gid for gid, e in attempts.items() if isinstance(e, dict) and e.get("state") == "running"]
    if running:
        errs.append(f"UNCERTAIN_IN_FLIGHT: {running}")
    # done → exactly one event
    done_gids = {gid for gid, e in attempts.items() if isinstance(e, dict) and e.get("state") == "done"}
    from collections import Counter
    ev_counts = Counter(event_gids)
    dups = {gid: c for gid, c in ev_counts.items() if c > 1}
    if dups:
        errs.append(f"duplicate events: {dups}")
    for gid in done_gids:
        if ev_counts.get(gid, 0) != 1:
            errs.append(f"done {gid} has {ev_counts.get(gid, 0)} events (expected 1)")
    # not_started must not have events
    not_started_with_ev = [gid for gid in ev_counts if gid not in done_gids and gid not in running]
    if not_started_with_ev:
        errs.append(f"not_started with events: {not_started_with_ev}")
    # --- per-rank audit (review#2 gaps 6, 7) ---
    for rank in range(1, 17):
        c0 = f"m_rank{rank:02d}_camp0"
        c1 = f"m_rank{rank:02d}_camp1"
        both_done = c0 in done_gids and c1 in done_gids
        af = ad / f"rank{rank:02d}.json"
        if af.exists() and not both_done:
            # premature/partial audit: audit present while neither OR only one camp done
            only_one = (c0 in done_gids) ^ (c1 in done_gids)
            errs.append(f"rank{rank:02d} premature/partial complete audit present "
                        f"(both_done={both_done} only_one_done={only_one})")
            continue
        if not af.exists():
            if both_done:
                errs.append(f"complete rank{rank:02d} missing audit file")
            continue
        # audit present AND both done → deep validation (review#2 gap 6)
        try:
            audit = json.loads(af.read_text(encoding="utf-8"))
        except Exception:
            errs.append(f"rank{rank:02d} audit corrupt/unparseable")
            continue
        if not isinstance(audit, dict):
            errs.append(f"rank{rank:02d} audit not a JSON object")
            continue
        if audit.get("rank") != rank:
            errs.append(f"rank{rank:02d} audit rank mismatch: {audit.get('rank')!r}")
        if audit.get("ok") is not True:
            errs.append(f"rank{rank:02d} audit ok is not True (got {audit.get('ok')!r})")
        audit_games = audit.get("games")
        audit_pairs: List[Tuple] = []
        if isinstance(audit_games, list):
            for g in audit_games:
                if isinstance(g, dict):
                    audit_pairs.append((g.get("game_id"), g.get("camp")))
        expected_pairs = {(c0, 0), (c1, 1)}
        actual_set = set(audit_pairs)
        if actual_set != expected_pairs:
            errs.append(f"rank{rank:02d} audit game_id/camp mismatch: "
                        f"{sorted(map(str, actual_set))} != {sorted(map(str, expected_pairs))}")

    return (len(errs) == 0, errs)


class MatrixRunner:
    def __init__(self, *, session_root, judge_dir, ifelse_dir, opponent_dir_of,
                 vendor_script, framework_src, timeout: float = 8.0,
                 wrapper_timeout_s: float = 60.0, attempt_fn: Optional[Callable] = None,
                 protocol_sha: str = "", run_id: Optional[str] = None,
                 python: Optional[str] = None, evaluated_agent: str = "miracle_ifelse",
                 auth_text: str = ""):
        self.session_root = Path(session_root)
        self.judge_dir = Path(judge_dir)
        self.ifelse_dir = Path(ifelse_dir)
        self.opponent_dir_of = opponent_dir_of
        self.vendor_script = Path(vendor_script)
        self.framework_src = Path(framework_src)
        self.timeout = timeout
        self.wrapper_timeout_s = wrapper_timeout_s
        self.attempt_fn = attempt_fn or self._default_attempt_fn
        self.protocol_sha = protocol_sha
        self.run_id = run_id or make_session_id()
        self.python = python
        self.evaluated_agent = evaluated_agent
        self.auth_text = auth_text
        self.session_id: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.progress: Dict[str, Any] = {"attempts": {}}
        self.plan = make_attempt_plan()

    # ---- session / paths ---- #
    def _setup_paths(self):
        self.work_dir = self.session_dir / "work"
        self.events_path = self.session_dir / "events.jsonl"
        self.progress_path = self.session_dir / "progress.json"
        self.data_dir = self.session_dir / "data"
        self.run_dir = self.data_dir / "runs" / "24_miracle" / self.evaluated_agent / self.run_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def prepare_session(self) -> Path:
        self.session_id = make_session_id()
        self.session_dir = ensure_fresh_session(self.session_root, self.session_id)
        self._setup_paths()
        write_progress_atomic(self.progress_path, self.progress)
        return self.session_dir

    def prepare_session_for_existing(self, session_id: str) -> Path:
        sd = self.session_root / session_id
        if sd.exists():
            raise FileExistsError(f"session already exists; refusing to overwrite: {sd}")
        self.session_id = session_id
        self.session_dir = ensure_fresh_session(self.session_root, session_id)
        self._setup_paths()
        return self.session_dir

    def resume(self, session_id: str) -> Path:
        """Open an EXISTING session for resumption (no new session created).
        Restores the original run_id from manifest BEFORE _setup_paths so the
        run_dir matches the original session, not a new run_id.

        Uses STRICT manifest+progress parse (review#2 gap 9): a corrupt file
        must RAISE, never silently swallow on lenient re-parse after a strict
        verify_session_for_resume already approved the session.
        """
        sd = self.session_root / session_id
        if not sd.exists():
            raise FileNotFoundError(f"cannot resume: session not found: {sd}")
        # restore run_id from manifest BEFORE _setup_paths (which builds run_dir from run_id)
        _m = _load_manifest_strict(sd / "manifest.json")
        _rid = _m.get("run_id") if _m else None
        if _rid:
            self.run_id = _rid
        self.session_id = session_id
        self.session_dir = sd
        self._setup_paths()
        # STRICT progress parse — never swallow errors after a successful verify
        self.progress = _load_progress_strict(self.progress_path)
        return self.session_dir

    # ---- manifest ---- #
    def record_manifest(self, *, opponent_hashes: Dict[int, str], build_hashes: Dict[int, str],
                        ifelse_sha: str, judge_sha: str, code_hashes: Dict[str, str],
                        control_inputs: Optional[Dict[str, Dict[str, str]]] = None,
                    ) -> Path:
        import platform, time
        m = {
            "session_id": self.session_id, "run_id": self.run_id,
            "created_unix": time.time(),
            "protocol_sha256": self.protocol_sha, "auth_text": self.auth_text,
            "plan_count": len(self.plan), "plan": self.plan,
            "timeout": self.timeout, "wrapper_timeout_s": self.wrapper_timeout_s,
            "evaluated_agent": self.evaluated_agent,
            "ifelse_sha256": ifelse_sha, "judge_sha256": judge_sha,
            "opponent_archive_sha256": {f"rank{k:02d}": v for k, v in opponent_hashes.items()},
            "cpp_build_sha256": {f"rank{k:02d}": v for k, v in build_hashes.items()},
            "code_hashes": code_hashes,
            "platform": platform.platform(),
        }
        if control_inputs is not None:
            m["control_inputs"] = control_inputs
        mp = self.session_dir / "manifest.json"
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, mp)
        return mp

    # ---- dry run (no subprocess) ---- #
    def dry_run(self) -> Dict[str, Any]:
        return {"session_id": self.session_id, "run_id": self.run_id,
                "plan_count": len(self.plan), "plan": self.plan,
                "judge_dir": str(self.judge_dir), "ifelse_dir": str(self.ifelse_dir),
                "evaluated_agent": self.evaluated_agent, "timeout": self.timeout,
                "wrapper_timeout_s": self.wrapper_timeout_s}

    # ---- per-game ---- #
    def _attempt_kwargs(self, attempt: Dict[str, Any]) -> Dict[str, Any]:
        rank, camp = attempt["rank"], attempt["camp"]
        opp_dir = self.opponent_dir_of(rank)
        opp_name = f"rank{rank:02d}"
        if camp == 0:
            p0_dir, p1_dir = self.ifelse_dir, opp_dir
            p0_name, p1_name = self.evaluated_agent, opp_name
        else:
            p0_dir, p1_dir = opp_dir, self.ifelse_dir
            p0_name, p1_name = opp_name, self.evaluated_agent
        return dict(game_id=attempt["game_id"], p0_dir=p0_dir, p1_dir=p1_dir,
                    p0_name=p0_name, p1_name=p1_name, evaluated_agent_camp=camp,
                    evaluated_agent=self.evaluated_agent, opponent=opp_name,
                    work_dir=self.work_dir, judge_dir=self.judge_dir,
                    vendor_script=self.vendor_script, framework_src=self.framework_src,
                    timeout=self.timeout, wrapper_timeout_s=self.wrapper_timeout_s,
                    python=self.python)

    def _default_attempt_fn(self, **kw):
        from agentbench_frame.games.miracle.match_runner import run_match_attempt
        return run_match_attempt(**kw)

    def _run_one(self, attempt: Dict[str, Any]) -> Dict[str, Any]:
        gid = attempt["game_id"]
        self.progress = load_progress(self.progress_path)  # resumable read
        if is_done(self.progress, gid):
            return {"halted": False, "skipped": True, "game_id": gid}
        uncertain = has_uncertain(self.progress)
        if uncertain:
            return {"halted": True, "state": "HALTED_INFRA_FAILURE",
                    "reason": f"UNCERTAIN_IN_FLIGHT: {uncertain}"}
        mark_running(self.progress, gid)
        write_progress_atomic(self.progress_path, self.progress)
        att = self.attempt_fn(**self._attempt_kwargs(attempt))
        rec = classify_game(att)
        append_event_atomic(self.events_path, rec)
        decision, reason = should_stop_with_residual(att)
        if decision == "stop":
            rec["halt_reason"] = reason
            mark_done(self.progress, gid, rec)
            write_progress_atomic(self.progress_path, self.progress)
            return {"halted": True, "state": "HALTED_INFRA_FAILURE",
                    "reason": reason, "record": rec, "game_id": gid}
        mark_done(self.progress, gid, rec)
        write_progress_atomic(self.progress_path, self.progress)
        return {"halted": False, "record": rec, "game_id": gid}

    # ---- per-rank ---- #
    def execute_rank(self, rank: int) -> Dict[str, Any]:
        attempts = [a for a in self.plan if a["rank"] == rank]
        audit_dir = self.session_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        for att in attempts:
            res = self._run_one(att)
            if res.get("halted"):
                return res
        # rank audit
        self.progress = load_progress(self.progress_path)
        games = [self.progress["attempts"][a["game_id"]] for a in attempts]
        ok, reasons = rank_audit(rank, games)
        (audit_dir / f"rank{rank:02d}.json").write_text(
            json.dumps({"rank": rank, "ok": ok, "reasons": reasons,
                        "games": games}, ensure_ascii=False, indent=2), encoding="utf-8")
        if not ok:
            return {"halted": True, "state": "HALTED_INFRA_FAILURE",
                    "reason": f"rank{rank:02d} audit failed: {reasons}"}
        return {"halted": False, "rank": rank}

    # ---- whole matrix ---- #
    def execute(self) -> Dict[str, Any]:
        self.progress = load_progress(self.progress_path)
        uncertain = has_uncertain(self.progress)
        if uncertain:
            return {"halted": True, "state": "HALTED_INFRA_FAILURE",
                    "reason": f"UNCERTAIN_IN_FLIGHT: {uncertain}"}
        for rank in range(1, 17):
            rank_atts = [a for a in self.plan if a["rank"] == rank]
            if all(is_done(self.progress, a["game_id"]) for a in rank_atts):
                continue
            res = self.execute_rank(rank)
            if res.get("halted"):
                return res
        return {"halted": False, "completed": True, "state": "COMPLETE"}

    # ---- aggregate from events (independently re-computable) ---- #
    def aggregate_from_events(self) -> Dict[str, Any]:
        records = []
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return aggregate(records)

    # ---- Run-compatible output for Results pipeline (review #1) ---- #
    def write_run_compatible_output(self) -> Path:
        """Drive the framework ``Run`` lifecycle to produce CI-compatible
        run output.

        Review #1 §5 + §6 收口：使用 staging → validation → atomic promotion
        模式。**绝不破坏性覆写**既有 run 目录。

        步骤：
          1. 在 *同一文件系统* 的 staging 目录（``<data_dir>/.staging/<run_id>``)
             下生成完整候选 run（events.jsonl + summary.json + run.toml）。
          2. 对候选 run 完整校验：32 unique game_ids（计划数）、event quality
             合法、run_id 一致性、events↔summary 重算一致、totals 通过
             ``Run.recompute_totals_from_events()`` 来自真实事件计数。
          3. 校验全部通过后，对既有 live run 若存在则备份到
             ``<staging>/.backup_<ts>``（同文件系统），然后原子 os.replace
             交换 staging ↔ live。
          4. promotion 失败时从备份回滚，不留下半截 events。
          5. 最终 live 目录正好含一个 run_id 目录（不生成第二个）。

        framework envelope 由 ``Run.write("game", **rec)`` 自动填，Miracle
        一手字段逐字透传。
        """
        from agentbench_frame.tracking.run import Run
        from agentbench_frame.tracking.quality import inspect_event_file

        agg = self.aggregate_from_events()

        # 1) staging directory: <data_dir>/.staging/<run_id>
        #    same fs as data_dir so os.replace is atomic.
        staging_root = self.data_dir / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = staging_root / self.run_id
        live_run_dir = self.run_dir
        # A previous writer may have been terminated between the two directory
        # renames.  Recover before deleting/reusing any staging path.
        self._recover_promotion_transaction(
            live_run_dir,
            live_run_dir.with_name(live_run_dir.name + ".backup_promote"),
            live_run_dir.with_name(live_run_dir.name + ".promote_marker.json"),
        )
        # always start staging clean (this is a scratch path, not run storage)
        if staging.exists():
            import shutil as _sh
            _sh.rmtree(staging)
        # build a Run inside staging (NOT in the final runs/<game>/<agent>/<id>
        # path): we point data_dir at staging so Run writes its files there.
        run = Run.start(game="24_miracle", agent=self.evaluated_agent,
                        run_type="eval", data_dir=str(staging),
                        run_id=self.run_id, append=False,
                        config={"matrix": "plan_a_32",
                                "total_attempts": agg["total_attempts"]})

        # 2) re-emit every matrix session event through framework envelope
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                run.write("game", **rec)

        # 3) recompute totals from valid game events on disk
        recomputed = run.recompute_totals_from_events(game_event_type="game")
        # log h2h (matrix synth-extracted) for summary wpis
        run.log_h2h(agg.get("h2h", {}) if isinstance(agg, dict) else {})

        # 4) finish in staging
        run.finish(extra_summary={
            "win_rate": agg["win_rate"],
            "win_rate_available": (agg["valid_games"] > 0),
            "total_episodes": recomputed["episodes"],
            "total_steps": recomputed["total_steps"],
            "matrix_aggregate": agg,
            "wins": recomputed["wins"],
            "losses": recomputed["losses"],
            "evaluation_status":
                ("COMPLETE" if agg["valid_games"] > 0 else "NO_VALID_GAMES"),
        })

        # 5) validate candidate BEFORE promotion
        candidate_run_dir = staging / "runs" / "24_miracle" / self.evaluated_agent / self.run_id
        errs = self._validate_candidate_run(candidate_run_dir, expected_count=agg["total_attempts"])
        if errs:
            # leave staging in place for inspection (it's under .staging/, not
            # the real runs/ path), but do NOT promote. Caller-visible live run
            # (if any) is byte-identical to its prior state.
            raise RuntimeError(f"candidate run validation failed: {errs}")

        # 6) atomic promotion: staging → live, with backup-and-rollback.
        self._atomic_promote(candidate_run_dir, live_run_dir)
        self._cleanup_staging(staging_root, self.run_id)
        return live_run_dir

    # ---- promotion helpers ------------------------------------------------ #
    @staticmethod
    def _validate_candidate_run(run_dir: Path, *,
                                 expected_count: int) -> List[str]:
        """Validate the candidate run directory prior to promotion. Returns
        a list of failure descriptions (empty list = candidate valid)."""
        errs: List[str] = []
        if not run_dir.exists():
            return [f"candidate run dir absent: {run_dir}"]
        for fname in ("events.jsonl", "summary.json", "run.toml"):
            if not (run_dir / fname).exists():
                errs.append(f"candidate missing: {fname}")
        if errs:
            return errs
        # event quality
        from agentbench_frame.tracking.quality import inspect_event_file
        rep = inspect_event_file(run_dir / "events.jsonl")
        if rep.malformed_lines:
            errs.append(f"candidate event_quality: malformed_lines={rep.malformed_lines}")
        if rep.duplicate_event_ids:
            errs.append(f"candidate event_quality: duplicates={rep.duplicate_event_ids}")
        if rep.missing_event_ids:
            errs.append(f"candidate event_quality: missing event_ids={rep.missing_event_ids}")
        if rep.missing_run_ids:
            errs.append(f"candidate event_quality: missing run_ids={rep.missing_run_ids}")
        # 32 (or expected_count) unique game_ids
        ev = []
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    ev.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        game_events = [e for e in ev
                       if isinstance(e, dict)
                       and (e.get("event_type") == "game" or e.get("event") == "game")]
        gids = [e.get("game_id") for e in game_events if e.get("game_id") is not None]
        unique = set(gids)
        if len(unique) != expected_count:
            errs.append(f"candidate unique game_ids={len(unique)} != expected {expected_count}")
        if len(gids) != len(unique):
            errs.append(f"candidate duplicate game_ids in events.jsonl={len(gids)-len(unique)}")
        # summary.json wins/losses/total_episodes == recomputed from events
        s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        valid = [e for e in game_events if e.get("valid")]
        wins_ev = sum(1 for e in valid if e.get("normalized_result") == "win")
        loss_ev = sum(1 for e in valid if e.get("normalized_result") == "loss")
        steps_ev = sum(int(e.get("steps") or 0) for e in valid)
        if s.get("total_episodes") != len(valid):
            errs.append(f"summary.total_episodes={s.get('total_episodes')} != valid_events={len(valid)}")
        if s.get("total_steps") != steps_ev:
            errs.append(f"summary.total_steps={s.get('total_steps')} != events_steps_sum={steps_ev}")
        if s.get("wins") != wins_ev:
            errs.append(f"summary.wins={s.get('wins')} != events_wins={wins_ev}")
        if s.get("losses") != loss_ev:
            errs.append(f"summary.losses={s.get('losses')} != events_losses={loss_ev}")
        # run.toml totals
        try:
            import tomllib
            with open(run_dir / "run.toml", "rb") as f:
                t = tomllib.load(f)
            if t.get("run", {}).get("total_episodes") != len(valid):
                errs.append(f"run.toml.total_episodes != {len(valid)}")
            if t.get("run", {}).get("total_steps") != steps_ev:
                errs.append(f"run.toml.total_steps != {steps_ev}")
        except Exception as e:
            errs.append(f"run.toml parse fail: {e!r}")
        return errs

    @staticmethod
    def _atomic_promote(candidate_dir: Path, live_dir: Path) -> None:
        """Directory-level atomic promotion: candidate → live.

        Two-rename transaction on the SAME filesystem (guaranteed by staging
        under ``<live_parent>/.staging/``):

          1. (recovery) If backup exists but live doesn't → crash interrupted
             between step 3 and 4 → restore backup → live.
          2. (recovery) If both live and backup exist → crash interrupted
             after step 4 but before cleanup → promotion succeeded, delete backup.
          3. If live exists: ``os.rename(live, backup)`` → live now absent.
          4. ``os.rename(candidate, live)`` → candidate now at live path.
          5. If step 4 raised: ``os.rename(backup, live)`` → restore old.
          6. If step 4 succeeded: ``shutil.rmtree(backup)``.

        Uses ``os.rename`` (not ``os.replace``) because on Windows renaming
        to a NON-existent destination works for directories, whereas
        ``os.replace`` on a non-empty dir raises WinError 5. We guarantee
        the destination doesn't exist by renaming live→backup first.
        """
        # The implementation below is deliberately directory-granular: the
        # three publishable files move together, never one at a time.  Keep
        # the legacy implementation below unreachable for compatibility with
        # older patches; all callers return through this transaction path.
        import shutil as _sh
        from agentbench_frame.games.miracle.atomicio import atomic_write_json
        if not candidate_dir.exists():
            raise FileNotFoundError(f"candidate missing: {candidate_dir}")
        backup_dir = live_dir.with_name(live_dir.name + ".backup_promote")
        marker_path = live_dir.with_name(live_dir.name + ".promote_marker.json")
        MatrixRunner._recover_promotion_transaction(live_dir, backup_dir, marker_path)
        live_dir.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(marker_path, {
            "schema_version": 1,
            "state": "prepared",
            "live_name": live_dir.name,
            "backup_name": backup_dir.name,
            "candidate_name": candidate_dir.name,
        })
        prior_live = live_dir.exists()
        if prior_live:
            if backup_dir.exists():
                raise RuntimeError(f"refusing to overwrite promotion backup: {backup_dir}")
            os.rename(str(live_dir), str(backup_dir))
        # Test-only seam: a real child process exits after the dangerous
        # first rename.  It is never enabled by normal callers.
        if os.environ.get("MIRACLE_TEST_KILL_AFTER_LIVE_RENAME") == "1":
            os._exit(86)
        try:
            os.rename(str(candidate_dir), str(live_dir))
        except Exception:
            if prior_live and backup_dir.exists():
                if live_dir.exists():
                    preserved = live_dir.with_name(live_dir.name + ".failed_candidate")
                    if preserved.exists():
                        raise RuntimeError(f"refusing to overwrite {preserved}")
                    os.rename(str(live_dir), str(preserved))
                os.rename(str(backup_dir), str(live_dir))
            raise
        if prior_live and backup_dir.exists():
            _sh.rmtree(backup_dir)
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        return

        import shutil as _sh
        if not candidate_dir.exists():
            raise FileNotFoundError(f"candidate missing: {candidate_dir}")
        backup_dir = live_dir.with_name(live_dir.name + ".backup_promote")

        # --- crash recovery ---
        MatrixRunner._recover_interrupted_promote(live_dir, backup_dir)

        live_parent = live_dir.parent
        live_parent.mkdir(parents=True, exist_ok=True)

        had_existing = live_dir.exists() and any(live_dir.iterdir())

        if had_existing:
            # Step 3: rename live → backup (live now absent)
            if backup_dir.exists():
                _sh.rmtree(backup_dir)
            os.rename(str(live_dir), str(backup_dir))
        else:
            # live doesn't exist or is empty — remove if empty
            if live_dir.exists():
                _sh.rmtree(live_dir)
            backup_dir.mkdir(exist_ok=True)  # empty placeholder for rollback safety

        try:
            # Step 4: rename candidate → live (candidate now at live path)
            os.rename(str(candidate_dir), str(live_dir))
        except Exception:
            # Step 5: restore backup → live
            if backup_dir.exists() and backup_dir != live_dir:
                try:
                    if live_dir.exists():
                        _sh.rmtree(live_dir)
                    os.rename(str(backup_dir), str(live_dir))
                except Exception:
                    pass  # best-effort; caller sees the original exception
            raise

        # Step 6: success — clean backup
        if backup_dir.exists():
            try:
                _sh.rmtree(backup_dir)
            except Exception:
                pass  # non-critical; leftover is detected on next run

    @staticmethod
    def _recover_promotion_transaction(live_dir: Path, backup_dir: Path,
                                       marker_path: Path) -> None:
        """Conservatively recover a directory promotion interrupted by a kill.

        A marker denotes an incomplete transaction.  When both the old backup
        and a candidate at the live path exist, the old live bytes win: the
        candidate is moved aside for inspection and is never published by
        inference.  This is intentionally stricter than treating both paths
        as a successful promotion.
        """
        backup_exists = backup_dir.exists()
        live_exists = live_dir.exists()
        if not marker_path.exists():
            if backup_exists:
                raise RuntimeError(
                    f"refusing ambiguous promotion state without marker: {backup_dir}"
                )
            return
        if backup_exists and not live_exists:
            os.rename(str(backup_dir), str(live_dir))
        elif backup_exists and live_exists:
            preserved = live_dir.with_name(live_dir.name + ".interrupted_candidate")
            if preserved.exists():
                raise RuntimeError(f"refusing to overwrite {preserved}")
            os.rename(str(live_dir), str(preserved))
            os.rename(str(backup_dir), str(live_dir))
        # A marker without a backup is a first-write transaction.  The live
        # directory is its only complete copy; leave it untouched.
        if live_dir.exists():
            try:
                marker_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _recover_interrupted_promote(live_dir: Path, backup_dir: Path) -> None:
        """Detect and recover from a crash during a previous _atomic_promote.

        Invariants after a clean run: live exists, backup does NOT exist.
        Crash states:
          - backup exists, live absent: interrupted between step 3 and 4
            → restore backup → live.
          - both exist: interrupted after step 4 but before cleanup
            → promotion succeeded; delete backup.
        """
        import shutil as _sh
        b_exists = backup_dir.exists()
        l_exists = live_dir.exists()
        if b_exists and not l_exists:
            # crash between rename(live→backup) and rename(candidate→live)
            try:
                os.rename(str(backup_dir), str(live_dir))
            except OSError:
                pass  # best-effort
        elif b_exists and l_exists:
            # crash after successful promotion, before cleanup
            try:
                _sh.rmtree(backup_dir)
            except Exception:
                pass

    @staticmethod
    def _cleanup_staging(staging_root: Path, run_id: str) -> None:
        """Clean any leftover staging entries for this run_id."""
        import shutil as _sh
        candidate = staging_root / run_id
        if candidate.exists():
            try:
                _sh.rmtree(candidate)
            except Exception:
                pass
        backup = staging_root / (run_id + ".backup_promote")
        if backup.exists():
            try:
                _sh.rmtree(backup)
            except Exception:
                pass
