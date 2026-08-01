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
decision point replays the recorded transcript prefix — feeding the judger→AI
frames in order (the ``id`` frame, off-turn notifications, earlier
roundbegins) and finally the decision-point ``roundbegin``, then reads the
first action frame the candidate emits (or times out / disconnects).

Fidelity: the probe replays the recorded transcript so the candidate's world
model is faithfully reconstructed at the decision point. A reference sample
without a transcript is rejected (``ReferenceSampleError``) — a legacy
single-frame ν cannot build the candidate's world model. Decision points with
status in {Died, Escaped, Skip, Error} produce no emitted primitive (no
decision point) — consistent with ``distribution.py``.

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


class ReferenceSampleError(ValueError):
    """Raised when a reference sample cannot be probed — e.g. it carries no
    recorded transcript (a legacy hand-authored single-frame ν) and so cannot
    faithfully reconstruct the candidate's world model. Fail-fast: never
    silently coerce to a uniform distribution that masks ``policy_kl = 0``."""


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

    **One fresh candidate process per sample.** Several ranked/sample
    candidates emit their action and then CRASH on the same turn (a bad
    ``end_turn`` / next-read path). With a single shared process the first
    crash poisons every subsequent sample (``poll() != None`` → every later
    emission ``None``), so both versions measure ``uniform-vs-uniform`` and
    ``policy_kl`` collapses to 0 — the exact "no policy update" symptom that
    stalled prior iterations. Restarting the candidate per sample isolates the
    crash to that one decision point; the judger itself TLEs-and-continues the
    same way, so this mirrors real play.

    ``probe_one(sample)`` starts a fresh process, replays the sample's
    transcript prefix (the recorded ``id`` frame + off-turn notifications +
    earlier roundbegins, ending at the decision-point ``roundbegin``), reads
    the emitted action, and closes the process. ``probe_set(samples)`` loops
    ``probe_one`` (so each sample gets its own process). ``close()`` is a
    no-op kept for API compatibility (each ``probe_one`` already cleans up
    its own process).

    **Hard wall-clock timeout.** The read loop inside ``_probe_one_impl`` is
    deadline-bounded, but the ``_write_frame`` calls (the ``id`` frame in
    ``_start`` and each transcript prefix frame in ``probe_one``) do a blocking
    ``flush()`` with NO deadline guard. A candidate that spawns but never
    drains stdin blocks that ``flush()`` forever once the frame exceeds the OS
    pipe buffer — *before* the read deadline is ever reached — which is exactly
    the act-2 production hang (run ``hl-run-0730``: ~11 min of zero progress).
    So ``probe_one`` runs the blocking body on a daemon worker and
    ``join(timeout)``s it; if the worker is still alive at the deadline it
    force-kills the candidate process (closing the pipes → the worker's stuck
    ``flush`` raises ``BrokenPipeError`` / its read returns EOF) and returns
    ``None`` (missing). The worker is fully joined before ``probe_one``
    returns, so there is no race on ``self._proc`` across the sequential
    samples of ``probe_set``. The join deadline accounts for the prefix feed:
    ``self.timeout + _ACK_DRAIN + prefix_slack + 0.5`` where
    ``prefix_slack = min(self.timeout, 0.05 * len(transcript))``.
    """

    # Grace given to the worker thread to finish after a hard-timeout
    # force-kill: it only needs to surface from the unblocked I/O and run
    # ``_close_proc`` (≤ its 2 s wait). 5 s comfortably bounds the worst case.
    _KILL_GRACE: float = 5.0
    # Best-effort drain of the candidate's ``id`` ack in ``_start``. The ack is
    # not used (only drained so it doesn't sit in the pipe); candidates that
    # don't ack (e.g. ``random_agent``) would otherwise burn the whole
    # ``self.timeout`` here, leaving no budget for the actual decision-point
    # read. Kept small and fixed so the read loop owns the real timeout budget.
    _ACK_DRAIN: float = 1.0
    # Per-prefix-frame drain grace: after feeding each non-last transcript
    # frame, drain the candidate's interleaved out-frames for up to this long
    # so stale prefix actions don't mask the decision-point emission. Small
    # and fixed: a synchronous candidate's response is already in the queue by
    # the time we drain (returns immediately); a non-responding frame (e.g. an
    # off-turn ``see``) burns at most this much per prefix frame.
    _PREFIX_DRAIN: float = 0.02

    def __init__(self, *, cmd, cwd, timeout: float = 5.0):
        self.cmd = list(cmd)
        self.cwd = str(cwd)
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
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

    def _start(self, id_frame: Optional[Dict[str, Any]] = None) -> None:
        """Spawn a fresh candidate process for one sample and send the ``id``
        frame. If ``id_frame`` is None, a default ``{"type":"id","id":0,
        "birth_pos":[0,0]}`` is sent (the legacy behaviour); otherwise the
        caller-supplied frame is sent verbatim (used by transcript replay to
        feed the recorded ``id`` frame so the candidate's birth position
        matches the reference roll)."""
        env = dict(os.environ)
        # scrub uv-poisoning env vars (CLAUDE.md gotcha)
        for k in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(k, None)
        self._proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._frame_q = queue.Queue()
        self._reader_thread = None
        self._start_reader()
        # Send the id frame so the candidate sets its player id. seat 0.
        if id_frame is None:
            id_frame = {"type": "id", "id": 0, "birth_pos": [0, 0]}
        _write_frame(self._proc.stdin, id_frame)
        # The candidate replies with an id-ack; drain it (best-effort, small —
        # candidates that don't ack must not burn the whole timeout here).
        self._read_frame(self._ACK_DRAIN)

    def probe_one(self, sample: ReferenceSample) -> Optional[EmittedAction]:
        las = enumerate_legal_actions(
            sample.legal_actions, status=sample.status,
            inventory=sample.inventory,
        )
        if len(las) == 0:
            return None  # no decision point

        # Hard wall-clock timeout (see class docstring): run the blocking body
        # on a daemon worker so a stuck write-flush / read can't hang the loop.
        holder: Dict[str, Any] = {"result": None}
        worker = threading.Thread(
            target=self._probe_one_impl_safe,
            args=(sample, las, holder), daemon=True)
        worker.start()
        # The body budget = ack-drain (``_ACK_DRAIN``) + the read loop
        # (``self.timeout``) + a small slack for the roundbegin write / close.
        # The transcript prefix grows the body: each prefix frame is a write
        # (and a small drain grace for stale out-frames). Bound the slack by
        # ``min(self.timeout, 0.05 * len(transcript))`` — 50 ms per prefix
        # frame, capped at the read timeout so a very long prefix cannot
        # double the wall-clock bound. A normal sample always finishes before
        # the join; only a genuinely stuck sample trips the force-kill.
        prefix_slack = min(self.timeout, 0.05 * len(sample.transcript))
        worker.join(self.timeout + self._ACK_DRAIN + prefix_slack + 0.5)
        if worker.is_alive():
            # Force-kill the candidate so its pipes close and the worker's
            # blocked flush()/read() unblock, letting the worker exit.
            self._force_kill()
            worker.join(self._KILL_GRACE)
            return None  # missing — same contract as any unresponsive candidate
        if holder.get("error") is not None:
            # Fail-fast: a ReferenceSampleError (e.g. missing transcript) must
            # propagate, not be swallowed to None — silently returning None
            # would mask policy_kl = 0.
            raise holder["error"]
        return holder["result"]

    def _probe_one_impl_safe(self, sample: ReferenceSample, las: LegalActionSet,
                             holder: Dict[str, Any]) -> None:
        """Worker entry: run ``_probe_one_impl`` and capture its result (or
        None on any non-fatal exception). ``ReferenceSampleError`` is captured
        into ``holder["error"]`` so ``probe_one`` can re-raise it — the fail-
        fast contract must not be coerced to a silent None."""
        try:
            holder["result"] = self._probe_one_impl(sample, las)
        except ReferenceSampleError as e:
            holder["error"] = e
        except Exception:
            holder["result"] = None

    def _drain_queue(self, grace: float) -> None:
        """Best-effort drain of any frames the candidate has emitted, so the
        candidate's stdout never fills and stale prefix actions don't mask the
        decision-point emission. Drains for up to ``grace`` seconds, returning
        as soon as a read times out (no more frames immediately available)."""
        deadline = time.monotonic() + grace
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if self._read_frame(remaining) is None:
                break  # timeout or disconnect — nothing more to drain now

    def _probe_one_impl(self, sample: ReferenceSample, las: LegalActionSet
                        ) -> Optional[EmittedAction]:
        # Fail-fast: a reference sample without a transcript cannot faithfully
        # reconstruct the candidate's world model (a legacy single-frame ν).
        # Never silently coerce to a uniform distribution that masks policy_kl=0.
        if len(sample.transcript) == 0:
            raise ReferenceSampleError(
                "reference sample has no transcript — re-record this ν "
                "(legacy single-frame ν cannot build the candidate's "
                "world model)")

        # The transcript's first frame is normally the recorded ``id`` frame
        # (the recorder always emits it first). If so, send it via ``_start``
        # so the candidate's birth position matches the reference roll, and
        # replay the rest as the prefix. If the transcript doesn't begin with
        # an ``id`` frame (defensive — hand-built transcripts), send the
        # default id and replay the whole transcript as the prefix.
        if sample.transcript[0].get("type") == "id":
            id_frame = sample.transcript[0]
            prefix = sample.transcript[1:]
        else:
            id_frame = None
            prefix = sample.transcript

        # One fresh process per sample (see class docstring): a candidate that
        # crashes after its action must not poison the next sample.
        try:
            self._start(id_frame=id_frame)
        except OSError:
            return None  # couldn't spawn -> missing
        if self._proc is None or self._proc.poll() is not None:
            return None

        # Feed the prefix frames in order. The reader thread (already running
        # from ``_start``) continuously drains the candidate's interleaved
        # out-frames so the candidate's stdout never fills and blocks the
        # write. We do NOT reply to the candidate's prefix actions — only
        # drain. For all but the last prefix frame, also drain the queue with
        # a small grace so stale prefix actions don't get captured as the
        # decision-point emission. The LAST frame in the transcript is the
        # decision-point ``roundbegin``; after feeding it we capture the first
        # emitted action.
        last_idx = len(prefix) - 1
        for i, frame in enumerate(prefix):
            try:
                _write_frame(self._proc.stdin, frame)
            except OSError:
                self._close_proc()
                return None
            if i < last_idx:
                self._drain_queue(self._PREFIX_DRAIN)

        # Capture the first primitive the candidate emits at the decision
        # point. The real candidate protocol sends ONE action then reads the
        # judger's per-action reply before continuing — it never sends
        # ``finish`` on its own, and the probe doesn't synthesize that reply,
        # so the candidate blocks after the first action. We therefore return
        # as soon as we capture the first action: draining to ``finish`` would
        # burn the whole read timeout per sample. ``None`` is reserved for a
        # truly unresponsive candidate (no emission at all).
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
            emitted = token
            if token not in las.tokens:
                out_of_support = True
            break  # captured the first action — return promptly
        self._close_proc()
        if emitted is None:
            return None  # truly no emission
        return EmittedAction(primitive=emitted, out_of_support=out_of_support,
                              sample_index=-1)

    def probe_set(self, samples: Sequence[ReferenceSample]
                  ) -> List[Optional[EmittedAction]]:
        out: List[Optional[EmittedAction]] = []
        for i, s in enumerate(samples):
            ea = self.probe_one(s)  # fresh process per sample
            if ea is not None:
                ea = EmittedAction(primitive=ea.primitive,
                                   out_of_support=ea.out_of_support,
                                   sample_index=i)
            out.append(ea)
        return out

    def _close_proc(self) -> None:
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

    def _force_kill(self) -> None:
        """Unconditional kill of the live candidate process — used by the
        hard-timeout path in ``probe_one`` to unblock a worker stuck in
        ``flush()`` / ``read()``. Does NOT clear ``self._proc``; the worker
        is responsible for its own ``_close_proc`` cleanup once unblocked.
        Safe to call when no process is live (no-op)."""
        if self._proc is None:
            return
        try:
            self._proc.kill()
        except Exception:
            pass

    def close(self) -> None:
        # Each probe_one cleans up its own process; kept for API compatibility.
        self._close_proc()
