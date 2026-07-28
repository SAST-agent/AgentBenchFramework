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

    # ---- manifest ---- #
    def record_manifest(self, *, opponent_hashes: Dict[int, str], build_hashes: Dict[int, str],
                        ifelse_sha: str, judge_sha: str, code_hashes: Dict[str, str],
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

    # ---- Run-compatible output for Results pipeline ---- #
    def write_run_compatible_output(self) -> Path:
        agg = self.aggregate_from_events()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # events.jsonl = matrix events (the per-game records ARE the events)
        run_events = self.run_dir / "events.jsonl"
        if self.events_path.exists():
            run_events.write_text(self.events_path.read_text(encoding="utf-8"), encoding="utf-8")
        # summary.json
        summary = {
            "run_id": self.run_id, "game": "24_miracle", "agent": self.evaluated_agent,
            "run_type": "eval", "created": "", "git_commit": "",
            "wall_hours": None,
            "total_episodes": agg["valid_games"], "total_steps": agg["steps_stats"]["total"],
            "win_rate": agg["win_rate"], "best_elo": None, "final_elo": None,
            "elo_history": [], "h2h": {}, "resource_summary": {},
            "config": {"matrix": "plan_a_32", "total_attempts": agg["total_attempts"]},
            "matrix_aggregate": agg,
        }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        # run.toml (escape-safe via simple build)
        q = lambda s: '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'
        toml = "[run]\n"
        toml += f"run_id = {q(self.run_id)}\ngame = {q('24_miracle')}\nagent = {q(self.evaluated_agent)}\n"
        toml += f"type = {q('eval')}\ntotal_steps = {agg['steps_stats']['total']}\n"
        toml += f"total_episodes = {agg['valid_games']}\n"
        (self.run_dir / "run.toml").write_text(toml, encoding="utf-8")
        return self.run_dir
