"""Cross-platform process-tree management for the Miracle subprocess wrapper.

Safety contract (阶段4b spec):
  * Only exact PIDs that this manager registered are ever signalled.
  * Before any signal, the PID's psutil create_time must match the value
    recorded at registration. A reused / stale PID is NEVER killed.
  * Graceful terminate first, then a short grace window (polled, not slept),
    then force-kill the tree (descendants included) only if still alive.
  * No name-based / fuzzy kill: a process is never selected by its image
    name, and signalling is never issued in bulk by interpreter or script
    name. All signalling targets exact registered PIDs via psutil; descendant
    discovery uses psutil's own process parentage, which is identity-confirming.
  * Idempotent: cleaning an already-dead PID is a silent no-op.

Per-process bookkeeping distinguishes a natural exit from a runner-requested
termination, so the adapter never mistakes a post-end_info cleanup-kill of an
idling AI client for a strategy crash (see match_runner.py classification).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import psutil
except ImportError as _exc:  # pragma: no cover - exercised via subprocess in tests
    raise ImportError(
        "Miracle process-tree cleanup requires psutil.\n"
        "Install with: uv sync --extra miracle   (or: pip install psutil)"
    ) from _exc

#: tolerance (seconds) for create_time comparison when confirming PID identity.
IDENTITY_TOL_S = 1.0


@dataclass
class ManagedProcess:
    pid: int
    role: str
    started_at: float                     # psutil create_time captured at registration
    popen: Optional[object] = None        # subprocess.Popen when we own the process
    natural_exit: bool = False
    natural_returncode: Optional[int] = None
    termination_requested: bool = False
    termination_reason: Optional[str] = None
    final_returncode: Optional[int] = None
    forced_kill: bool = False
    cleanup_succeeded: bool = False
    identity_confirmed: bool = True       # create_time matched at the moment we acted


# --------------------------------------------------------------------------- #
# low-level helpers (all exact-PID, psutil-based)
# --------------------------------------------------------------------------- #
def _create_time(pid: int) -> Optional[float]:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _identity_ok(pid: int, started_at: float) -> bool:
    """True if `pid` currently belongs to the same process recorded at `started_at`."""
    ct = _create_time(pid)
    if ct is None:
        return False
    return abs(ct - started_at) < IDENTITY_TOL_S


def _exists(pid: int) -> bool:
    return psutil.pid_exists(pid)


def _wait_for_exit(pid: int, timeout: float, step: float = 0.03) -> bool:
    """Poll until `pid` is gone or `timeout` elapses. Returns True if it died."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _exists(pid):
            return True
        time.sleep(step)
    return not _exists(pid)


def _graceful_terminate(pid: int) -> None:
    try:
        psutil.Process(pid).terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def _collect_tree(root_pid: int):
    """Return [root_pid, *descendant_pids] while the root is alive, so cleanup can
    reach orphaned descendants even after the root exits. Descendants come from
    psutil parentage (identity-confirming); selection is never by image name.
    If the root is already gone, returns [root_pid] only."""
    try:
        root = psutil.Process(root_pid)
        kids = root.children(recursive=True)
        return [root_pid] + [k.pid for k in kids]
    except psutil.NoSuchProcess:
        return [root_pid]


def _force_kill_one(pid: int) -> None:
    """Force-kill a single EXACT pid (no descendant walk, no name match)."""
    try:
        psutil.Process(pid).kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #
class ProcessTreeManager:
    """Tracks exact PIDs and cleans them up with graceful -> force semantics.

    Use :meth:`register_popen` for processes you spawned directly and
    :meth:`register_pid` for descendant PIDs you only learned about (e.g. read
    from a child's stdout). Only registered PIDs are ever touched.
    """

    def __init__(self, grace_s: float = 2.0):
        self.grace_s = grace_s
        self._procs: Dict[int, ManagedProcess] = {}

    # ---- registration ----
    def register_popen(self, popen, role: str) -> ManagedProcess:
        mp = ManagedProcess(
            pid=popen.pid, role=role,
            started_at=_create_time(popen.pid) or 0.0, popen=popen,
        )
        self._procs[mp.pid] = mp
        return mp

    def register_pid(self, pid: int, role: str) -> ManagedProcess:
        mp = ManagedProcess(pid=pid, role=role, started_at=_create_time(pid) or 0.0)
        self._procs[mp.pid] = mp
        return mp

    # ---- liveness / refresh ----
    def is_alive(self, mp: ManagedProcess) -> bool:
        return _identity_ok(mp.pid, mp.started_at)

    def poll(self) -> None:
        for mp in self._procs.values():
            self._refresh(mp)

    def _refresh(self, mp: ManagedProcess) -> None:
        if mp.natural_exit:
            return
        if mp.popen is not None:
            rc = mp.popen.poll()
            if rc is not None:
                mp.natural_exit = True
                mp.natural_returncode = rc
                if mp.final_returncode is None:
                    mp.final_returncode = rc
        elif not _identity_ok(mp.pid, mp.started_at):
            # descendant we don't own has disappeared
            mp.natural_exit = True
            mp.natural_returncode = None

    # ---- cleanup ----
    def cleanup_one(self, mp: ManagedProcess, reason: str) -> ManagedProcess:
        self._refresh(mp)
        if mp.natural_exit:
            mp.cleanup_succeeded = True
            return mp
        if not _identity_ok(mp.pid, mp.started_at):
            # PID gone OR reused (create_time mismatch): never kill an unconfirmed PID
            mp.identity_confirmed = False
            mp.cleanup_succeeded = True
            return mp
        # live + identity-confirmed: request termination of the WHOLE tree.
        # Collect descendants now, while the root is alive, so we can still reach
        # them after the root exits (orphaned descendants would otherwise survive).
        mp.identity_confirmed = True
        mp.termination_requested = True
        mp.termination_reason = reason
        tree_pids = _collect_tree(mp.pid)
        for p in tree_pids:
            _graceful_terminate(p)
        _wait_for_exit(mp.pid, self.grace_s)
        survivors = [p for p in tree_pids if _exists(p)]
        if survivors:
            for p in survivors:
                _force_kill_one(p)
            mp.forced_kill = True
            _wait_for_exit(mp.pid, max(self.grace_s, 1.0))
        mp.cleanup_succeeded = not _identity_ok(mp.pid, mp.started_at)
        if mp.cleanup_succeeded and mp.popen is not None and mp.final_returncode is None:
            rc = mp.popen.poll()
            if rc is not None:
                mp.final_returncode = rc
        return mp

    def cleanup_all(self, reason: str) -> List[ManagedProcess]:
        return [self.cleanup_one(mp, reason) for mp in list(self._procs.values())]

    def status(self) -> List[dict]:
        self.poll()
        return [
            {
                "pid": mp.pid, "role": mp.role, "started_at": mp.started_at,
                "natural_exit": mp.natural_exit,
                "natural_returncode": mp.natural_returncode,
                "termination_requested": mp.termination_requested,
                "termination_reason": mp.termination_reason,
                "final_returncode": mp.final_returncode,
                "forced_kill": mp.forced_kill,
                "cleanup_succeeded": mp.cleanup_succeeded,
                "identity_confirmed": mp.identity_confirmed,
            }
            for mp in self._procs.values()
        ]
