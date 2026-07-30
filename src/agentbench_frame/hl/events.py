"""Finalized-only append-only event storage for HL runs."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1.0"
KNOWN_EVENT_TYPES = frozenset(
    {
        "run_started",
        "run_resumed",
        "act_completed",
        "version_created",
        "candidate_selected",
        "match_completed",
        "evaluation_completed",
        "certification_completed",
        "policy_kl_measured",
        "occupancy_measured",
        "elo_updated",
        "champion_promoted",
        "rollback_selected",
        "experience_updated",
        "checkpoint_created",
        "run_completed",
    }
)
_SECRET_KEYS = frozenset({"api_key", "authorization", "access_token", "secret"})
_SECRET_VALUE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def _validate_safe(value: Any, path: str = "event") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ValueError(f"{path} contains secret-like material")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _SECRET_KEYS:
                raise ValueError(f"{path}.{key} is a forbidden secret field")
            _validate_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe(item, f"{path}[{index}]")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class HLEventWriter:
    """Append complete immutable JSON records and reject duplicate IDs."""

    def __init__(self, path: str | os.PathLike[str], run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._event_ids = {
            record.get("event_id")
            for record in read_events(self.path)
            if record.get("event_id")
        }

    def write(self, event_type: str, **fields: Any) -> str:
        if event_type not in KNOWN_EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")
        event_id = str(fields.pop("event_id", uuid.uuid4().hex))
        if event_id in self._event_ids:
            raise ValueError(f"duplicate event_id: {event_id}")
        record = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type,
            "run_id": self.run_id,
            "created_at": str(fields.pop("created_at", _timestamp())),
            **fields,
        }
        _validate_safe(record)
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._event_ids.add(event_id)
        return event_id


def read_events(
    path: str | os.PathLike[str],
    *,
    warn: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"event line {line_number} is not an object")
            event_type = value.get("event_type")
            if event_type not in KNOWN_EVENT_TYPES and warn is not None:
                warn(f"unknown event type: {event_type}")
            records.append(value)
    return records
