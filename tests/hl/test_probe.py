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
from agentbench_frame.hl.probe import ReferenceProbe, EmittedAction, \
    ReferenceSampleError

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


def _make_transcript(observation, *, status=0, move_mask=None,
                     extra_roundbegins=()):
    """Build a minimal valid transcript: ``[id, (earlier roundbegins...),
    roundbegin]``. The last frame is the decision-point roundbegin, built from
    ``observation`` with the same setdefault defaults the old single-frame
    probe path used (so the candidate reads top-level ``inturn``/``status``/
    ``state``/``tools``/``others``). ``extra_roundbegines`` prepends additional
    roundbegin frames before the decision point (to exercise prefix feed /
    drain)."""
    rb = dict(observation)
    rb.setdefault("type", "roundbegin")
    rb.setdefault("inturn", 0)
    rb.setdefault("state", rb.get("round", 1))  # round number
    rb.setdefault("status", status)
    rb.setdefault("hp", 200)
    rb.setdefault("keys", [0])
    rb.setdefault("pos", [0, 0, 1])
    rb.setdefault("tools", {"LandMine": [0, 0], "Sticky": [0, 0],
                            "Kit": 0, "Transport": 0})
    rb.setdefault("others", [
        {"player_id": 1, "status": 0, "keys": [0], "hp": 200},
        {"player_id": 2, "status": 0, "keys": [0], "hp": 200},
        {"player_id": 3, "status": 0, "keys": [0], "hp": 200},
    ])
    id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
    return (id_frame,) + tuple(extra_roundbegins) + (rb,)


def _alive_sample():
    obs = {"round": 1, "inturn": 0}
    return ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": [False]*8, "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(obs),
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
    obs = {"round": 1, "inturn": 0}
    sample = ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": [True] * 8, "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(obs, move_mask=[True] * 8),
    )
    emitted = probe.probe_one(sample)
    probe.close()
    assert isinstance(emitted, EmittedAction)
    assert emitted.primitive[0] == "move"   # random_agent's on-turn action


def test_probe_returns_promptly_when_candidate_blocks_after_first_action(tmp_path):
    """Regression for the ~8 min/act probe slowdown in run hl-run-0730-fix.

    The real candidate protocol (see ``candidates/v1/agent.py``) sends ONE
    action then reads the judger's per-action reply before continuing — it
    never sends ``finish`` on its own. The probe doesn't synthesize that
    reply, so the candidate blocks after the first action. The probe's read
    loop must NOT then drain the full read timeout waiting for a ``finish``
    that will never come: it should return the captured first action
    promptly. (The measurement already records only the first action; each
    sample is a fresh process, so there is no turn state to preserve by
    draining.) Otherwise the probe burns ~timeout seconds per sample —
    8 samples x 2 versions = minutes of dead time per act.
    """
    blocker = r'''
import json, sys
def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4: return None
    n = int(hdr.decode("utf-8"))
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))
def send(frame):
    s = json.dumps(frame)
    sys.stdout.buffer.write(len(s).to_bytes(4, "big", signed=True) + s.encode("utf-8"))
    sys.stdout.buffer.flush()
while True:
    msg = read_frame()
    if msg is None: break
    t = msg.get("type")
    if t == "id":
        send({"type": "id", "player_num": 1, "player_list": [1, 1, 1, 1]})  # ack
    elif t == "roundbegin":
        send({"type": "action", "action": ["move", [1, 2, 3]]})  # one action
        # then block waiting for the judger's per-action reply (never comes)
        msg2 = read_frame()
        if msg2 is None: break
'''
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(blocker, encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")
    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=10.0)
    obs = {"round": 1, "inturn": 0}
    sample = ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": [True] * 8, "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(obs, move_mask=[True] * 8),
    )
    t0 = time.monotonic()
    emitted = probe.probe_one(sample)
    elapsed = time.monotonic() - t0
    probe.close()
    assert emitted is not None
    assert emitted.primitive[0] == "move"
    assert elapsed < 3.0   # prompt, NOT the full 10 s read timeout


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
    big_obs = {"round": 1, "inturn": 0, "pad": "x" * 20000}
    big = ReferenceSample(
        observation=big_obs,
        legal_actions={"attack": [], "move": [False] * 8, "detect": False,
                       "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(big_obs),
    )
    t0 = time.monotonic()
    emitted = probe.probe_one(big)
    elapsed = time.monotonic() - t0
    probe.close()
    assert emitted is None          # missing, not coerced
    assert elapsed < 10.0           # hard wall-clock bound, not infinite


# ---- §3: transcript replay tests ----

# A candidate that emits a specific move on its turn (for in-support replay).
MOVE_CANDIDATE = r'''
import json, sys

def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4: return None
    n = int(hdr.decode("utf-8"))
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))

def send(frame):
    s = json.dumps(frame)
    sys.stdout.buffer.write(len(s).to_bytes(4, "big", signed=True) + s.encode("utf-8"))
    sys.stdout.buffer.flush()

while True:
    msg = read_frame()
    if msg is None: break
    t = msg.get("type")
    if t == "id":
        send({"type": "id", "player_num": 1, "player_list": [1,1,1,1]})
    elif t == "roundbegin":
        if msg.get("inturn") == 0:
            send({"type": "action", "action": ["move", 0]})
        # off-turn roundbegin: no reply
    elif t == "action":
        send({"type": "action", "action": ["finish"]})
    # offround / notifications: no reply
'''


def _make_move_workspace(tmp_path: Path, candidate_src: str = MOVE_CANDIDATE) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "agent.py").write_text(candidate_src, encoding="utf-8")
    (ws / "manifest.toml").write_text(
        'shape = "single_file"\nentrypoint = "agent.py"\n', encoding="utf-8")
    return ws


def test_probe_replay_in_support_emission(tmp_path):
    """§3.3 headline: replaying a recorded transcript yields an in-support
    primitive. The candidate reaches the decision point with its world model
    faithfully reconstructed and emits a legal ``["move", 0]``."""
    ws = _make_move_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=5.0)
    obs = {"round": 1, "inturn": 0}
    sample = ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": [True] + [False]*7,
                       "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(obs, move_mask=[True] + [False]*7),
    )
    emitted = probe.probe_one(sample)
    probe.close()
    assert isinstance(emitted, EmittedAction)
    assert emitted.out_of_support is False
    assert emitted.primitive == ("move", 0)


def test_probe_replay_no_deadlock_on_many_prefix_outframes(tmp_path):
    """§3.4: a candidate that emits an action after every input frame must
    not deadlock the probe's writer (the reader thread drains stdout). The
    probe must complete within the hard bound, not hang."""
    # Candidate that echoes a finish action after EVERY input frame (id,
    # roundbegin, off-turn notifications, everything).
    echo_all = ECHO_CANDIDATE.replace(
        'elif t == "roundbegin":',
        'elif t == "roundbegin" or t == "see" or t == "offround" or t == "action":')
    ws = _make_move_workspace(tmp_path, candidate_src=echo_all)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    # Build a transcript with several prefix frames (off-turn notifications)
    # before the decision-point roundbegin, so the candidate emits on each.
    obs = {"round": 3, "inturn": 0}
    extra = (
        {"type": "see", "round": 1, "inturn": 0},
        {"type": "see", "round": 2, "inturn": 0},
        {"type": "offround", "round": 2, "inturn": 0},
    )
    sample = ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": [True] + [False]*7,
                       "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(obs, extra_roundbegins=extra),
    )
    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=5.0)
    t0 = time.monotonic()
    emitted = probe.probe_one(sample)
    elapsed = time.monotonic() - t0
    probe.close()
    # completes within the hard bound, no hang
    assert elapsed < 15.0
    # may or may not capture an action (the echo candidate emits finish), but
    # the key assertion is that it returns at all rather than hanging.
    assert elapsed < 15.0  # no deadlock


def test_probe_replay_stall_returns_none_in_time(tmp_path):
    """§3.5: a candidate that spawns, reads but never writes yields a missing
    emission (None) within the hard wall-clock timeout — not a hang."""
    silent = r'''
import json, sys
def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4: return None
    n = int(hdr.decode("utf-8"))
    sys.stdin.buffer.read(n)
    return {}
while True:
    if read_frame() is None: break
    # never reply, just keep reading
'''
    ws = _make_move_workspace(tmp_path, candidate_src=silent)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    obs = {"round": 1, "inturn": 0}
    sample = ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": [True] + [False]*7,
                       "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(obs),
    )
    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=2.0)
    t0 = time.monotonic()
    emitted = probe.probe_one(sample)
    elapsed = time.monotonic() - t0
    probe.close()
    assert emitted is None  # missing, not coerced
    # returns within the hard bound (timeout + ack_drain + prefix_slack + grace)
    assert elapsed < 10.0


def test_probe_replay_preserves_hard_timeout(tmp_path):
    """§3.2: a candidate that sleeps forever after the decision-point
    roundbegin yields None within the hard bound and the process is reaped."""
    sleeper = r'''
import json, sys, time
def read_frame():
    hdr = sys.stdin.buffer.read(4)
    if len(hdr) < 4: return None
    n = int(hdr.decode("utf-8"))
    return json.loads(sys.stdin.buffer.read(n).decode("utf-8"))
def send(frame):
    s = json.dumps(frame)
    sys.stdout.buffer.write(len(s).to_bytes(4, "big", signed=True) + s.encode("utf-8"))
    sys.stdout.buffer.flush()
while True:
    msg = read_frame()
    if msg is None: break
    t = msg.get("type")
    if t == "id":
        send({"type": "id", "player_num": 1, "player_list": [1,1,1,1]})
    elif t == "roundbegin":
        # sleep forever — never emit at the decision point
        time.sleep(1000)
'''
    ws = _make_move_workspace(tmp_path, candidate_src=sleeper)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    obs = {"round": 1, "inturn": 0}
    sample = ReferenceSample(
        observation=obs,
        legal_actions={"attack": [], "move": [True] + [False]*7,
                       "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        transcript=_make_transcript(obs),
    )
    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=2.0)
    t0 = time.monotonic()
    emitted = probe.probe_one(sample)
    elapsed = time.monotonic() - t0
    probe.close()
    assert emitted is None  # missing, hard timeout
    assert elapsed < 10.0   # within the hard wall-clock bound


def test_probe_rejects_missing_transcript(tmp_path):
    """§1.3 fail-fast at the probe boundary: a reference sample with
    ``transcript=()`` raises ``ReferenceSampleError`` (not silently coerced
    to None / uniform, which would mask ``policy_kl = 0``)."""
    ws = _make_move_workspace(tmp_path)
    cb = HLCodebase(root=ws, store=tmp_path / "store")
    h = cb.snapshot(parent_version_id=None)
    cmd, cwd = candidate_command(h, store=cb.store, dest=tmp_path / "stage")

    probe = ReferenceProbe(cmd=cmd, cwd=cwd, timeout=5.0)
    sample = ReferenceSample(
        observation={"round": 1, "inturn": 0},
        legal_actions={"attack": [], "move": [True] + [False]*7,
                       "detect": False, "interprops": []},
        inventory={"LandMine": 0, "Sticky": 0, "Transport": 0, "Kit": 0},
        status=0, seat=0, opponent="rank01",
        # NO transcript — legacy single-frame ν
    )
    with pytest.raises(ReferenceSampleError, match="re-record"):
        probe.probe_one(sample)
    probe.close()
