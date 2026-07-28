"""Evidence-safety helpers for the 24_miracle smoke driver.

Pure + unit-testable: session identity (never delete/overwrite), a STRICT
Group-1 gate, an INDEPENDENT residual-process check (exact PID + psutil
create_time — never trusts ``cleanup_succeeded``, never kills by name, never
touches unrelated processes), and manifest building. No subprocess execution
lives here; that stays in ``tools/miracle_smoke.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "Miracle smoke residual check requires psutil.\n"
        "Install with: uv sync --extra miracle   (or: pip install psutil)"
    ) from _exc

IDENTITY_TOL_S = 1.0


@dataclass
class ManagedProc:
    pid: int
    started_at: float            # psutil create_time captured when the process was spawned
    role: str


# --------------------------------------------------------------------------- #
# session identity
# --------------------------------------------------------------------------- #
def make_session_id() -> str:
    """A fresh, effectively-unique session id (timestamp + token)."""
    import datetime
    import secrets
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + secrets.token_hex(3)


def session_exists(root, session_id: str) -> bool:
    return (Path(root) / session_id).exists()


def ensure_fresh_session(root, session_id: str) -> Path:
    """Create and return a brand-new session dir. REFUSE (raise) if it already
    exists — this module NEVER deletes or overwrites a prior session."""
    root = Path(root)
    sd = root / session_id
    if sd.exists():
        raise FileExistsError(f"session already exists; refusing to overwrite: {sd}")
    root.mkdir(parents=True, exist_ok=True)
    sd.mkdir(parents=True)
    return sd


# --------------------------------------------------------------------------- #
# hashing + manifest
# --------------------------------------------------------------------------- #
def sha256_file(path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def build_manifest(*, session_id: str, auth_cap: int, python_executable: str,
                   python_version: str, code_files, asset_files,
                   groups_planned: List[Dict[str, Any]],
                   notes: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "created_unix": time.time(),
        "auth_cap_games": auth_cap,
        "python_executable": python_executable,
        "python_version": python_version,
        "code_hashes": {str(p): sha256_file(p) for p in code_files},
        "asset_hashes": {str(p): sha256_file(p) for p in asset_files},
        "groups_planned": groups_planned,
        "notes": notes or [],
    }


def write_manifest_atomic(session_dir, manifest: Dict[str, Any]) -> Path:
    p = Path(session_dir) / "manifest.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return p


# --------------------------------------------------------------------------- #
# INDEPENDENT residual-process check (exact PID + create_time identity)
# --------------------------------------------------------------------------- #
def load_managed_procs_from_result_json(path) -> List[ManagedProc]:
    """Independently read a result-json and extract Judge/AI0/AI1 with their
    PID + create_time identity. Returns [] if the file is missing/corrupt or has
    no usable identity (callers must treat [] as 'cannot verify', never 'clean')."""
    try:
        rj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    out: List[ManagedProc] = []
    for role in ("judge", "ai0", "ai1"):
        p = rj.get(role) or {}
        pid = p.get("pid")
        started = p.get("started_at")
        if isinstance(pid, int) and pid > 0 and isinstance(started, (int, float)):
            out.append(ManagedProc(pid=pid, started_at=float(started), role=role))
    return out


def check_residual_procs(procs: List[ManagedProc],
                         tol: float = IDENTITY_TOL_S) -> Dict[str, List[ManagedProc]]:
    """For each recorded proc:
      pid gone                          -> clean
      pid exists, create_time matches   -> RESIDUAL (real leftover)
      pid exists, create_time differs   -> reused (PID reuse; DO NOT kill)
    No name matching, no batch kill, unrelated processes are never touched."""
    clean, residual, reused = [], [], []
    for mp in procs:
        if not psutil.pid_exists(mp.pid):
            clean.append(mp)
            continue
        try:
            ct = psutil.Process(mp.pid).create_time()
        except psutil.NoSuchProcess:
            clean.append(mp)
            continue
        if abs(ct - mp.started_at) < tol:
            residual.append(mp)
        else:
            reused.append(mp)
    return {"clean": clean, "residual": residual, "reused": reused}


# --------------------------------------------------------------------------- #
# STRICT Group-1 gate
# --------------------------------------------------------------------------- #
def _att_reasons(att) -> List[str]:
    r = []
    gid = getattr(att, "game_id", "?")
    if not getattr(att, "valid", False):
        r.append(f"{gid}: valid!=true (normalized={getattr(att,'normalized_result',None)})")
    if getattr(att, "error_type", None):
        r.append(f"{gid}: error_type={att.error_type} reason={getattr(att,'reason','')}")
    if getattr(att, "wrapper_timeout", False):
        r.append(f"{gid}: wrapper_timeout")
    if getattr(att, "result_json_status", None) != "ok":
        r.append(f"{gid}: result_json_status={att.result_json_status}")
    if getattr(att, "discrepancies", None):
        r.append(f"{gid}: evidence discrepancies={att.discrepancies}")
    if not getattr(att, "realized_randomization", None):
        r.append(f"{gid}: realized_randomization missing (replay not parseable)")
    if getattr(att, "raw_winner", None) not in (0, 1):
        r.append(f"{gid}: raw_winner not decisive ({att.raw_winner})")
    return r


def group1_strict_clean(attempts, summary) -> Tuple[bool, List[str]]:
    """Strict Group-1 gate. ANY anomaly blocks Group 2. An empty/missing invalid
    set or missing process-identity can NEVER pass this gate."""
    reasons: List[str] = []
    if len(attempts) != 2:
        reasons.append(f"attempt_count={len(attempts)} != 2")
    for att in attempts:
        reasons.extend(_att_reasons(att))
        # independent residual check — requires non-empty identity, never an empty free-pass
        rj_path = getattr(att, "evidence_paths", {}).get("result_json") if hasattr(att, "evidence_paths") else None
        procs = load_managed_procs_from_result_json(rj_path) if rj_path else []
        if not procs:
            reasons.append(f"{att.game_id}: no managed-proc identity in result-json (cannot verify by empty set)")
        else:
            res = check_residual_procs(procs)
            if res["residual"]:
                reasons.append(f"{att.game_id}: RESIDUAL pids={[(p.pid, p.role) for p in res['residual']]}")
    s = summary or {}
    for k, want in (("attempted_games", 2), ("valid_games", 2), ("invalid_games", 0)):
        if s.get(k) != want:
            reasons.append(f"summary.{k}={s.get(k)} != {want}")
    # events/summary recompute consistency (summary carries the runner's own check)
    if s.get("evaluation_status") != "COMPLETE":
        reasons.append(f"summary.evaluation_status={s.get('evaluation_status')} != COMPLETE")
    return (len(reasons) == 0, reasons)


def should_run_group2(g1_ok: bool) -> bool:
    """Group 2 is run ONLY when Group 1 is fully clean."""
    return bool(g1_ok)
