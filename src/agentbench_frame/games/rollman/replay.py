"""Strict parsing for frozen Rollman JSONL replays."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any


class ReplayError(ValueError):
    """A replay is malformed, truncated, or scientifically unusable."""


@dataclasses.dataclass(frozen=True)
class RollmanReplay:
    path: Path
    records: tuple[dict[str, Any], ...]
    initializations: tuple[dict[str, Any], ...]
    rounds: tuple[dict[str, Any], ...]
    terminal: dict[str, Any]
    raw_sha256: str
    normalized_sha256: str

    @property
    def final_score(self) -> tuple[int, int]:
        raw = self.terminal.get("score")
        if not isinstance(raw, list) or len(raw) != 2:
            raise ReplayError("terminal score must contain Rollman and Ghosts values")
        return int(raw[0]), int(raw[1])


def load_replay(path: str | Path) -> RollmanReplay:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ReplayError(str(exc)) from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ReplayError(f"line {line_number}: record must be an object")
        records.append(value)
    if not records:
        raise ReplayError("replay is empty")

    initializations = tuple(record for record in records if "board" in record)
    terminals = tuple(
        record for record in records if record.get("StopReason") is not None
    )
    rounds = tuple(
        record
        for record in records
        if "board" not in record and record.get("StopReason") is None
    )
    if not initializations:
        raise ReplayError("replay has no initialization frame")
    if len(terminals) != 1:
        raise ReplayError(
            f"replay must contain exactly one terminal frame, found {len(terminals)}"
        )
    for label, values in (
        ("initialization", initializations),
        ("round", rounds),
        ("terminal", terminals),
    ):
        for index, record in enumerate(values):
            if "level" not in record or "round" not in record:
                raise ReplayError(f"{label} {index}: missing level or round")

    normalized = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ).encode("utf-8")
    return RollmanReplay(
        path=source.resolve(),
        records=tuple(records),
        initializations=initializations,
        rounds=rounds,
        terminal=terminals[0],
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        normalized_sha256=hashlib.sha256(normalized).hexdigest(),
    )

