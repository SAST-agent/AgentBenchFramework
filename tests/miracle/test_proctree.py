"""Process-tree manager contract tests (SKILL.md 测试门槛 item 17 + the 8
process-safety requirements from the 阶段4b spec).

These run FIRST as a RED LIGHT (the implementation proctree.py does not exist
yet), the failure is saved as evidence, THEN the implementation is written to
turn them green.

Validated behaviours:
  1. normal child exits naturally (natural_exit, no kill)
  2. parent exits but child still survives -> child is reachable + cleanable
  3. timeout -> the whole recorded tree is cleaned
  4. cleanup is idempotent
  5. an already-exited PID is skipped, never a false kill of others
  6. only this run's recorded exact PIDs are cleaned
  7. after cleanup neither parent nor child PID remains
  8. NO name-based / fuzzy kill (structural scan of the implementation)
  +  PID identity (create_time) must match before any kill; a stale/reused PID
     is never killed.

    py -3.13 -m pytest tests/miracle/test_proctree.py -v
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import time

import psutil
import pytest

from agentbench_frame.games.miracle import proctree  # noqa: F401  (red-light import)
from agentbench_frame.games.miracle.proctree import ManagedProcess, ProcessTreeManager

HELPER = pathlib.Path(__file__).parent / "_proc_helper.py"
PY = sys.executable


# ---- spawn helpers ------------------------------------------------------- #
def _spawn(args, **kw):
    return subprocess.Popen(
        [PY, str(HELPER), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw,
    )


def _ready(proc, timeout=5.0):
    """Read the single READY line the helper prints on startup."""
    line = proc.stdout.readline()
    assert line.startswith("READY"), f"unexpected helper output: {line!r}"
    return line


def _child_pid_from(line: str) -> int:
    return int(line.split("child=")[1].strip())


def _alive(pid: int, started_at: float, tol: float = 1.0) -> bool:
    """Identity-checked liveness: PID exists AND its create_time matches."""
    if not psutil.pid_exists(pid):
        return False
    try:
        return abs(psutil.Process(pid).create_time() - started_at) < tol
    except psutil.NoSuchProcess:
        return False


# ---- 1: natural exit ----------------------------------------------------- #
def test_normal_child_exits_naturally():
    mgr = ProcessTreeManager()
    p = _spawn(["stay", "0.3"])
    try:
        _ready(p)
        mp = mgr.register_popen(p, "child")
        p.wait(timeout=5)
        mgr.poll()
        assert mp.natural_exit is True
        assert mp.natural_returncode == 0
        assert mp.termination_requested is False
        mgr.cleanup_all("test-done")
        assert mp.cleanup_succeeded is True
    finally:
        if p.poll() is None:
            p.kill()


# ---- 2: parent exits, child survives, then cleaned ----------------------- #
def test_parent_exits_child_survives_then_cleaned():
    mgr = ProcessTreeManager()
    p = _spawn(["spawn_child", "30", "0.2"])  # child lives 30s, parent 0.2s
    line = _ready(p)
    child_pid = _child_pid_from(line)
    p.wait(timeout=5)  # parent exits naturally
    cmp = mgr.register_pid(child_pid, "grandchild")
    assert _alive(cmp.pid, cmp.started_at) is True      # child still alive
    mgr.cleanup_all("parent-gone")
    assert _alive(cmp.pid, cmp.started_at) is False      # child now cleaned
    assert cmp.cleanup_succeeded is True


# ---- 3: timeout cleans the whole recorded tree --------------------------- #
def test_timeout_kills_whole_tree():
    mgr = ProcessTreeManager()
    p = _spawn(["spawn_child", "30", "30"])  # both long-lived
    line = _ready(p)
    child_pid = _child_pid_from(line)
    mp = mgr.register_popen(p, "parent")
    cmp = mgr.register_pid(child_pid, "child")
    assert _alive(mp.pid, mp.started_at) and _alive(cmp.pid, cmp.started_at)
    mgr.cleanup_all("timeout")
    assert not _alive(mp.pid, mp.started_at)
    assert not _alive(cmp.pid, cmp.started_at)
    assert mp.cleanup_succeeded and cmp.cleanup_succeeded


# ---- 4: idempotent ------------------------------------------------------- #
def test_cleanup_is_idempotent():
    mgr = ProcessTreeManager()
    p = _spawn(["stay", "30"])
    _ready(p)
    mp = mgr.register_popen(p, "x")
    mgr.cleanup_all("first")
    assert not _alive(mp.pid, mp.started_at)
    # second cleanup must be a no-op without raising
    mgr.cleanup_all("second")
    mgr.cleanup_one(mp, "third")
    assert not _alive(mp.pid, mp.started_at)


# ---- 5: already-exited PID is skipped, no false kill --------------------- #
def test_exited_pid_does_not_cause_false_kill():
    mgr = ProcessTreeManager()
    dead = _spawn(["stay", "0.2"])
    _ready(dead)
    dead.wait(timeout=5)
    dmp = mgr.register_popen(dead, "dead")          # already exited
    other = _spawn(["stay", "30"])
    _ready(other)
    other_mp = mgr.register_popen(other, "other")
    mgr.cleanup_one(dmp, "noop")                    # must skip the dead one
    assert _alive(other_mp.pid, other_mp.started_at) is True   # other untouched
    mgr.cleanup_all("final")
    assert not _alive(other_mp.pid, other_mp.started_at)


# ---- 6: only recorded exact PIDs are cleaned ----------------------------- #
def test_only_recorded_pids_are_cleaned():
    mgr = ProcessTreeManager()
    a = _spawn(["stay", "30"]); _ready(a)
    b = _spawn(["stay", "30"]); _ready(b)
    b_ct = psutil.Process(b.pid).create_time()
    ma = mgr.register_popen(a, "a")                 # register ONLY a
    mgr.cleanup_all("partial")                      # should clean a only
    assert not _alive(ma.pid, ma.started_at)
    # b was never registered -> must still be alive
    assert _alive(b.pid, b_ct) is True
    b.terminate(); b.wait(timeout=5)


# ---- 7: no residual parent nor child PID --------------------------------- #
def test_cleanup_leaves_neither_parent_nor_child_pid():
    mgr = ProcessTreeManager()
    p = _spawn(["spawn_child", "30", "30"])
    line = _ready(p)
    child_pid = _child_pid_from(line)
    mp = mgr.register_popen(p, "parent")
    cmp = mgr.register_pid(child_pid, "child")
    mgr.cleanup_all("done")
    assert not _alive(mp.pid, mp.started_at)
    assert not _alive(cmp.pid, cmp.started_at)


# ---- 8: no name-based / fuzzy kill (structural) -------------------------- #
def test_no_name_based_or_fuzzy_kill():
    src = pathlib.Path(proctree.__file__).read_text(encoding="utf-8")
    forbidden = ["/im ", "/im\"", "/IM ", "/IM\"", "tskill ", "imagename",
                 "Get-Process", "wmic process where name"]
    hits = [f for f in forbidden if f in src]
    assert not hits, f"fuzzy/name-based kill patterns found in proctree.py: {hits}"
    # every taskkill must be /PID-based (exact PID), never by image name
    for m in re.finditer(r"taskkill", src, re.IGNORECASE):
        ctx = src[m.start(): m.start() + 160]
        assert "/PID" in ctx or "/pid" in ctx, f"non-PID taskkill: {ctx!r}"


# ---- +: PID identity (create_time) guards against PID reuse -------------- #
def test_identity_mismatch_does_not_kill():
    mgr = ProcessTreeManager()
    p = _spawn(["stay", "30"])
    _ready(p)
    mp = mgr.register_popen(p, "x")
    mp.started_at = 1.0  # forge a stale start time -> simulates PID reuse
    mgr.cleanup_one(mp, "stale")
    assert mp.identity_confirmed is False
    assert _alive(p.pid, psutil.Process(p.pid).create_time()) is True  # NOT killed
    # restore real identity and clean up properly
    mp.started_at = psutil.Process(p.pid).create_time()
    mgr.cleanup_one(mp, "real")
    assert mp.identity_confirmed is True
    assert _alive(mp.pid, mp.started_at) is False
