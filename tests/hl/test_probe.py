"""Tests for hl/probe.py — the ReferenceProbe.

Contract (plan §probe; Q2 decision = re-run both versions over ν):
- The probe drives a staged candidate through each reference decision point
  and records the primitive it emits. Same ν for both versions -> comparable
  policy-KL traces.
- A reference sample carries (observation, legal_actions, inventory, status).
  The probe synthesizes the minimal Saiblo frame sequence to present that
  decision point to the candidate and captures its emitted action.
- Decision points with status in {Died, Escaped, Skip, Error} produce no
  emitted primitive (no decision point) — consistent with distribution.py.
- Out-of-support / illegal emissions are flagged, not coerced.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.adapter import candidate_command
from agentbench_frame.hl.reference import ReferenceSample
from agentbench_frame.hl.distribution import (
    enumerate_legal_actions, FINISH,
)
from agentbench_frame.hl.probe import ReferenceProbe, EmittedAction

# A minimal echo candidate speaking the REAL LostSpace wire format: reads
# 4 ASCII digits + JSON (judger->AI, ``convert_byte_str_for_ai`` — same
# ``int(str(read(4), "utf-8"))`` decode the bundled candidates use) and writes
# 4-byte big-endian length + JSON (AI->judger, ``convert_to_bytes``). On a
# roundbegin for player 0 who is Alive it emits a finish action. NOTE: keep the
# input framing as ASCII digits — a binary header here would mask the exact bug
# this suite guards against (the probe must send ASCII-digit headers).
ECHO_CANDIDATE = r'''
import json, sys

def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        return None
    n = int(hdr.decode("utf-8"))   # 4 ASCII digits (judger->AI framing)
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))

def send(frame):
    s = json.dumps(frame)
    sys.stdout.buffer.write(len(s).to_bytes(4, "big", signed=True) + s.encode("utf-8"))
    sys.stdout.buffer.flush()

first = True
while True:
    msg = read_frame()
    if msg is None:
        break
    t = msg.get("type")
    if t == "id":
        send({"type": "id", "player_num": 1, "player_list": [1,1,1,1]})
    elif t == "roundbegin":
        # always emit a finish on our turn
        send({"type": "action", "action": ["finish"]})
    elif t == "action":
        send({"type": "action", "action": ["finish"]})
    else:
        # offround / notification: no reply expected, but stay alive
        pass
'''


def _make_echo_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(ECHO_CANDIDATE, encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    return ws


def _alive_sample():
    return ReferenceSample(
        observation={"round": 1, "inturn": 0},
        legal_actions={"attack": [], "move": [False]*8, "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
    )


def test_probe_records_finish_from_echo_candidate(tmp_path):
    ws = _make_echo_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=10.0)
    sample = _alive_sample()
    emitted = probe.probe_one(sample)
    probe.close()

    assert isinstance(emitted, EmittedAction)
    assert emitted.primitive == FINISH
    assert emitted.out_of_support is False


def test_probe_no_decision_point_for_dead_status(tmp_path):
    ws = _make_echo_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=10.0)
    dead = ReferenceSample(
        observation={"round": 1, "inturn": 0},
        legal_actions={}, inventory={"LandMine": 0, "Sticky": 0,
                                     "Transport": 0, "Kit": 0},
        status=1,  # Died
        seat=0, opponent="rank01",
    )
    emitted = probe.probe_one(dead)
    probe.close()
    assert emitted is None  # no decision point


def test_probe_flags_out_of_support_emission(tmp_path):
    """If the candidate emits an illegal action (not in A(s)), the probe
    records it with out_of_support=True rather than coercing it."""
    bad_candidate = ECHO_CANDIDATE.replace(
        'send({"type": "action", "action": ["finish"]})',
        'send({"type": "action", "action": ["move", [99, 99, 99]]})',  # illegal
    )
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(bad_candidate, encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=10.0)
    emitted = probe.probe_one(_alive_sample())
    probe.close()
    assert emitted is not None
    assert emitted.out_of_support is True
    # the raw illegal action is preserved, not coerced
    assert emitted.primitive[0] == "move"


def test_probe_full_set_records_one_per_decision_point(tmp_path):
    """probe_set over ν yields one EmittedAction per non-empty decision point,
    in the same order as the reference set."""
    ws = _make_echo_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    samples = [_alive_sample(), _alive_sample(),
               ReferenceSample(observation={}, legal_actions={},
                                inventory={"LandMine":0,"Sticky":0,
                                           "Transport":0,"Kit":0},
                                status=2, seat=0, opponent="rank01")]  # Escaped
    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=10.0)
    emitted = probe.probe_set(samples)
    probe.close()
    # one entry per sample, preserving ν alignment: 2 Alive (emitted) + 1
    # Escaped (None, no decision point).
    assert len(emitted) == 3
    assert emitted[0] is not None and emitted[0].primitive == FINISH
    assert emitted[1] is not None and emitted[1].primitive == FINISH
    assert emitted[2] is None  # Escaped -> no decision point


def test_probe_handles_unresponsive_candidate(tmp_path):
    """A candidate that never replies within timeout yields a missing emission
    (None), recorded as such — not a crash, not coerced to a default."""
    silent = r'''
import json, sys, time
def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4: return None
    n = int(hdr.decode("utf-8"))   # 4 ASCII digits (judger->AI framing)
    sys.stdin.buffer.read(n)
    return {}
while True:
    if read_frame() is None: break
    # never reply
    time.sleep(0.01)
'''
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(silent, encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=2.0)
    emitted = probe.probe_one(_alive_sample())
    probe.close()
    assert emitted is None  # missing, not coerced


def test_probe_drives_real_bundled_baseline():
    """Regression for three real-protocol bugs the echo mock could not catch:

    1. probe -> AI framing must be 4 ASCII digits (``convert_byte_str_for_ai``),
       not a binary length header — the bundled agents decode it with
       ``int(str(read(4), "utf-8"))`` and ValueError/exit on a binary header.
    2. the roundbegin turn fields (``inturn`` etc.) must be TOP-LEVEL, not
       nested under ``"state"`` — the candidate reads ``msg["inturn"]``.
    3. a candidate that does NOT reply to the ``id`` frame must not starve the
       later action read — the single owning reader (no per-read thread leak)
       ensures the move frame reaches ``probe_one``.

    Drives the actual bundled ``random_agent`` baseline (reads the same ASCII
    decode as ``candidates/v1`` and does not ack ``id``) and asserts it emits a
    move on its turn. This is the production code path that crashed at act 2.
    """
    import agentbench_frame.lostspace.baselines.random_agent as _ra
    ra_path = Path(_ra.__file__)
    probe = ReferenceProbe(
        cmd=[sys.executable, str(ra_path)], cwd=str(ra_path.parent),
        timeout=2.0)
    sample = ReferenceSample(
        observation={"round": 1, "inturn": 0},
        legal_actions={"attack": [], "move": [True] * 8, "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
    )
    emitted = probe.probe_one(sample)
    probe.close()
    assert isinstance(emitted, EmittedAction)
    assert emitted.primitive[0] == "move"   # random_agent's on-turn action


def test_probe_hard_timeout_on_write_blocked_candidate(tmp_path):
    """Regression for the act-2 production hang (run hl-run-0730).

    A candidate that spawns but NEVER drains stdin blocks ``_write_frame``'s
    ``flush()`` once the frame exceeds the OS pipe buffer (~4 KB). That block
    sits BEFORE the read-deadline loop in ``probe_one``, so the existing
    per-read deadline never fires and the probe hangs forever (observed ~11 min
    of zero progress at act 2). ``probe_one`` must instead return ``None``
    within a hard wall-clock bound (worker thread joined with a timeout +
    force-kill), not infinity.
    """
    hog = r'''
import time
while True:
    time.sleep(0.5)   # spawn, but never read stdin, never exit
'''
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(hog, encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=1.5)
    # Observation padded past the pipe buffer so the roundbegin frame's flush
    # blocks (the candidate never reads it). The pad is an extra field the
    # probe forwards verbatim — enumerate_legal_actions ignores it.
    big = ReferenceSample(
        observation={"round": 1, "inturn": 0, "pad": "x" * 20000},
        legal_actions={"attack": [], "move": [False] * 8, "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
    )
    t0 = time.monotonic()
    emitted = probe.probe_one(big)
    elapsed = time.monotonic() - t0
    probe.close()
    assert emitted is None          # missing, not coerced
    assert elapsed < 10.0           # hard wall-clock bound, not infinite
