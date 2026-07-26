"""
HL iteration event layer — public-schema, append-only events for the
Heuristic-Learning measurement contract.

Every event written through ``HLEventWriter`` carries the public fields:

    schema_version, event_id, run_id, created_at, event_type

plus an event-specific payload. Events are appended to ``events.jsonl`` and
never rewritten or deleted (the measurement contract's finalized-only rule).

Forward compatibility: ``read_events`` ignores records whose ``event_type``
it does not recognize (skipping with a count) rather than raising, so a newer
writer's events never break an older reader. Unknown *extra* fields on a
recognized event are preserved, not dropped.

HL event types (see the measurement contract / plan.md):
    agent_act       — one coding-agent invocation lifecycle
    version         — a codebase version (snapshot/restore), content-hash keyed
    eval            — a frozen BenchmarkSpec evaluation result
    policy_kl       — local_policy_kl_trace between two versions
    occupancy_shift — state-occupancy distribution change between two versions
    budget          — learning/evaluation/total resource accounting

Missing values are recorded as ``None`` (unknown), never coerced to 0.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

SCHEMA_VERSION = 1
PUBLIC_FIELDS = ("schema_version", "event_id", "run_id", "created_at", "event_type")

#: Event types this writer/reader pair knows about. New types may be added
#: incrementally; the reader silently skips anything not in this set.
KNOWN_EVENT_TYPES = frozenset({
    "agent_act",
    "version",
    "eval",
    "policy_kl",
    "occupancy_shift",
    "budget",
    # Legacy framework event types are also recognized so a mixed log is
    # readable; they simply lack some HL payload fields.
    "step", "episode", "eval_result", "log", "resource", "lostspace_match",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _event_id() -> str:
    # uuid4 is fine here: this runs in the controller process, not inside a
    # Workflow script (where Math.random/uuid would be unavailable).
    return uuid.uuid4().hex


class HLEventWriter:
    """Append-only writer that stamps every record with the public schema fields.

    Args:
        run_id:  the run this event stream belongs to.
        path:    path to the ``events.jsonl`` file.
        append:  if True, open in append mode (do not truncate an existing log).
    """

    def __init__(self, run_id: str, path: Union[str, os.PathLike],
                 append: bool = False):
        self.run_id = run_id
        self.path = os.fspath(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        mode = "a" if append else "w"
        self._fh = open(self.path, mode, encoding="utf-8")

    def write(self, event_type: str, **payload: Any) -> Dict[str, Any]:
        """Append one event. Returns the full record (public fields + payload)."""
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": _event_id(),
            "run_id": self.run_id,
            "created_at": _now_iso(),
            "event_type": event_type,
        }
        record.update(payload)
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self._fh.flush()
        return record

    def close(self):
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_events(
    path: Union[str, os.PathLike],
    *,
    return_skipped: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
    """Read an HL events.jsonl.

    Returns the list of recognized events in file order. Records whose
    ``event_type`` is unknown or absent are skipped (and counted) rather than
    raising — the forward-compat rule.

    If ``return_skipped=True``, also returns the count of skipped records.
    """
    events: List[Dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            et = rec.get("event_type")
            if et is None:
                # Legacy line keyed on "event" instead of "event_type".
                legacy = rec.get("event")
                if legacy in KNOWN_EVENT_TYPES:
                    rec["event_type"] = legacy
                    events.append(rec)
                    continue
                skipped += 1
                continue
            if et not in KNOWN_EVENT_TYPES:
                skipped += 1
                continue
            events.append(rec)
    if return_skipped:
        return events, skipped
    return events
