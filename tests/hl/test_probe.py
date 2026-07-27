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
from pathlib import Path

import pytest

from agentbench_frame.hl.codebase import HLCodebase
from agentbench_frame.hl.adapter import candidate_command
from agentbench_frame.hl.reference import ReferenceSample
from agentbench_frame.hl.distribution import (
    enumerate_legal_actions, FINISH,
)
from agentbench_frame.hl.probe import ReferenceProbe, EmittedAction

# A minimal echo candidate: reads one length-prefixed frame, and if it's a
# roundbegin for player 0 who is Alive, emits a finish action. This exercises
# the real wire protocol (4-byte big-endian length header + JSON body).
ECHO_CANDIDATE = r'''
import json, sys

def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4:
        return None
    n = int.from_bytes(hdr, "big", signed=True)
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
    n = int.from_bytes(hdr, "big", signed=True)
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
