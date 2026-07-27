"""
ReferenceProbe — drive a staged candidate through reference decision points.

Decision (Q2): re-run BOTH versions over the frozen ReferenceStateSet (ν) so
policy KL is a pure measurement (same z for both versions, no occupancy drift
confound). This probe is what obtains a version's emitted primitive at each
reference state.

Wire protocol — asymmetric (see ``candidates/v1/agent.py``):
* probe -> AI (judger->AI, ``convert_byte_str_for_ai``): 4 ASCII decimal
  digits (zero-padded) + UTF-8 JSON; the candidate decodes the length with
  ``int(str(read(4), "utf-8"))``.
* AI -> probe (AI->judger, ``convert_to_bytes``): 4-byte big-endian length
  + UTF-8 JSON (binary, signed).
The probe sends the candidate an ``id`` frame once, then for each reference
decision point sends a synthesized ``roundbegin`` frame and reads action
frames until the candidate emits ``finish`` (or times out / disconnects).

Fidelity: the probe presents *one* decision point per sample using the
sample's observation as the ``roundbegin`` state. It does not replay a full
game; the reference set was captured (or constructed) to be self-contained per
decision point. Decision points with status in {Died, Escaped, Skip, Error}
produce no emitted primitive (no decision point) — consistent with
``distribution.py``.

Missing / unresponsive candidates yield ``None`` (missing), never coerced to a
default. Illegal / out-of-support emissions are flagged, not coerced.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from agentbench_frame.hl.distribution import (
    enumerate_legal_actions,
    LegalActionSet,
    FINISH,
)
from agentbench_frame.hl.reference import ReferenceSample


@dataclass(frozen=True)
class EmittedAction:
    """The primitive a version emitted at one reference decision point."""
    primitive: Tuple[Any, ...]    # canonical action token, or raw illegal action
    out_of_support: bool          # True if the emission was not in A(s)
    sample_index: int              # position in the reference set


def _write_frame(stream, frame: Dict[str, Any]) -> None:
    # probe -> AI direction uses the Saiblo ``convert_byte_str_for_ai`` framing:
    # 4 ASCII decimal digits (zero-padded) + UTF-8 JSON. The candidate decodes
    # the length with ``int(str(read(4), "utf-8"))`` (see
    # ``candidates/v1/agent.py::receive_data``). The reverse direction (AI ->
    # probe) is binary 4-byte big-endian, decoded by ``_read_frame`` — do NOT
    # use a binary header here: the candidate would raise ValueError on read,
    # exit, and the next flush would surface a broken-pipe OSError (Errno 22 on
    # Windows, Errno 32 elsewhere).
    data = json.dumps(frame).encode("utf-8")
    header = f"{len(data):04d}".encode("ascii")
    stream.write(header)
    stream.write(data)
    stream.flush()


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes from ``stream`` or None on EOF.

    Blocking — call only from the dedicated reader thread that owns the
    stream (see ``ReferenceProbe._read_loop``).
    """
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class ReferenceProbe:
    """Drive one staged candidate through reference decision points.

    One probe = one candidate process. ``probe_one(sample)`` presents a single
    decision point; ``probe_set(samples)`` presents all of them (reusing the
    same process). Call ``close()`` when done.
    """

    def __init__(self, *, cmd, cwd, timeout: float = 5.0):
        self.cmd = list(cmd)
        self.cwd = str(cwd)
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._initialized = False
        self._frame_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    def _start_reader(self) -> None:
        """Spawn ONE thread that owns the candidate's stdout and pushes parsed
        frames onto ``_frame_q``. A single owner means a timed-out read leaves
        no second reader blocked on the pipe to steal subsequent frames — the
        bug that silently dropped emissions from candidates that don't reply to
        the ``id`` frame (e.g. ``random_agent``)."""
        if self._reader_thread is not None:
            return
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()
        self._reader_thread = t

    def _read_loop(self) -> None:
        assert self._proc is not None
        stream = self._proc.stdout
        while True:
            try:
                hdr = _read_exact(stream, 4)
                if hdr is None:
                    self._frame_q.put(None)
                    return
                n = int.from_bytes(hdr, "big", signed=True)
                body = _read_exact(stream, n)
                if body is None:
                    self._frame_q.put(None)
                    return
                self._frame_q.put(json.loads(body.decode("utf-8")))
            except (OSError, json.JSONDecodeError):
                self._frame_q.put(None)
                return

    def _read_frame(self, timeout: float) -> Optional[Dict[str, Any]]:
        """Pop one frame the reader thread produced, or None on timeout."""
        try:
            return self._frame_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _ensure_started(self) -> None:
        if self._proc is not None:
            return
        env = dict(os.environ)
        # scrub uv-poisoning env vars (CLAUDE.md gotcha)
        for k in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(k, None)
        self._proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._start_reader()
        self._send_init()

    def _send_init(self) -> None:
        # Send the id frame so the candidate sets its player id. seat 0.
        _write_frame(self._proc.stdin, {"type": "id", "id": 0,
                                        "birth_pos": [0, 0]})
        # The candidate replies with an id-ack; drain it (best-effort). A
        # timeout here (candidate doesn't ack id) no longer leaks a reader —
        # the single reader thread simply waits for the next real frame.
        self._read_frame(self.timeout)
        self._initialized = True

    def probe_one(self, sample: ReferenceSample) -> Optional[EmittedAction]:
        las = enumerate_legal_actions(
            sample.legal_actions, status=sample.status,
            inventory=sample.inventory,
        )
        if len(las) == 0:
            return None  # no decision point

        self._ensure_started()
        if self._proc is None or self._proc.poll() is not None:
            return None  # process died -> missing

        # Present the decision point as a roundbegin for player 0. The sample's
        # observation IS the roundbegin frame: the judger carries the turn
        # fields at the TOP LEVEL (the candidate reads root["inturn"],
        # root["status"], root["state"]=round number, root["tools"],
        # root["others"]; see candidates/v1/agent.py::start_turn). Send it
        # top-level — do NOT nest under "state" or the candidate never sees its
        # turn (inturn missing -> skips play -> no emission). Defaults fill any
        # field a partial observation omits without clobbering provided ones.
        frame = dict(sample.observation)
        frame.setdefault("type", "roundbegin")
        frame.setdefault("inturn", 0)
        frame.setdefault("state", frame.get("round", 1))  # round number
        frame.setdefault("status", sample.status)
        frame.setdefault("hp", 200)
        frame.setdefault("keys", [0])
        frame.setdefault("pos", [0, 0, 1])
        frame.setdefault("tools", {"LandMine": [0, 0], "Sticky": [0, 0],
                                    "Kit": 0, "Transport": 0})
        frame.setdefault("others", [
            {"player_id": 1, "status": 0, "keys": [0], "hp": 200},
            {"player_id": 2, "status": 0, "keys": [0], "hp": 200},
            {"player_id": 3, "status": 0, "keys": [0], "hp": 200},
        ])
        _write_frame(self._proc.stdin, frame)

        # Read action frames until finish or timeout. The FIRST non-finish
        # action is the behavioral choice; we keep draining to finish so the
        # candidate's turn state stays consistent, but if it goes silent we
        # still return the recorded action (not None — None is reserved for
        # "no emission at all", i.e. a truly unresponsive candidate).
        emitted: Optional[Tuple[Any, ...]] = None
        out_of_support = False
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            frame = self._read_frame(remaining)
            if frame is None:
                break  # timeout / disconnect — fall through to return emitted
            if frame.get("type") != "action":
                continue
            action = frame.get("action")
            if not isinstance(action, list) or not action:
                continue
            token = tuple(action)
            if token == FINISH:
                if emitted is None:
                    emitted = FINISH
                break
            if emitted is None:
                emitted = token
                if token not in las.tokens:
                    out_of_support = True
                # keep reading until finish to drain the turn
        if emitted is None:
            return None  # truly no emission
        return EmittedAction(primitive=emitted, out_of_support=out_of_support,
                              sample_index=-1)

    def probe_set(self, samples: Sequence[ReferenceSample]
                  ) -> List[Optional[EmittedAction]]:
        out: List[Optional[EmittedAction]] = []
        for i, s in enumerate(samples):
            ea = self.probe_one(s)
            if ea is not None:
                ea = EmittedAction(primitive=ea.primitive,
                                   out_of_support=ea.out_of_support,
                                   sample_index=i)
            out.append(ea)
        return out

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
