"""Safe parsing and aggregation for historical DOTO replay ZIPs."""

from __future__ import annotations

import ast
import json
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


MAX_REPLAY_JSON_BYTES = 64 * 1024 * 1024


class DotoReplayError(ValueError):
    pass


@dataclass(frozen=True)
class ReplayFrame:
    frame: int
    humans: list
    fireballs: list
    meteors: list
    balls: list
    scores: list[float]
    bonus: list
    events: list
    raw: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplaySummary:
    final_scores: tuple[float, float]
    winner: int | None
    last_frame: int
    event_counts: dict[str, int]
    by_faction: dict[str, dict[str, float | int]]

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["final_scores"] = list(self.final_scores)
        return value


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise DotoReplayError(f"unsafe ZIP member: {member.filename}")
        if member.file_size > MAX_REPLAY_JSON_BYTES:
            raise DotoReplayError(f"oversized ZIP member: {member.filename}")
    return members


def _list_field(raw: Any, name: str) -> list:
    value = raw
    if isinstance(raw, str):
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError) as exc:
            raise DotoReplayError(f"invalid {name} list") from exc
    if not isinstance(value, list):
        raise DotoReplayError(f"invalid {name} list")
    return value


def iter_replay(path: Path) -> Iterator[ReplayFrame]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = _safe_members(archive)
            json_members = [item for item in members if item.filename.endswith(".json")]
            if len(json_members) != 1:
                raise DotoReplayError("replay ZIP must contain exactly one JSON member")
            raw_json = archive.read(json_members[0])
    except (OSError, zipfile.BadZipFile) as exc:
        raise DotoReplayError(f"invalid replay ZIP: {exc}") from exc
    if len(raw_json) > MAX_REPLAY_JSON_BYTES:
        raise DotoReplayError("replay JSON exceeds size limit")
    try:
        rows = json.loads(raw_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DotoReplayError(f"invalid replay JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise DotoReplayError("replay JSON root must be a list")
    for raw in rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("frame"), int):
            raise DotoReplayError("replay frame must be an object with integer frame")
        scores = _list_field(raw.get("scores", []), "scores")
        try:
            numeric_scores = [float(item) for item in scores]
        except (TypeError, ValueError) as exc:
            raise DotoReplayError("invalid scores list") from exc
        yield ReplayFrame(
            frame=raw["frame"],
            humans=_list_field(raw.get("humans", []), "humans"),
            fireballs=_list_field(raw.get("fireballs", []), "fireballs"),
            meteors=_list_field(raw.get("meteors", []), "meteors"),
            balls=_list_field(raw.get("balls", []), "balls"),
            scores=numeric_scores,
            bonus=_list_field(raw.get("bonus", []), "bonus"),
            events=_list_field(raw.get("events", []), "events"),
            raw=raw,
        )


def summarize_replay(path: Path) -> ReplaySummary:
    counts: Counter[int] = Counter()
    by_faction: dict[str, dict[str, float | int]] = {
        str(faction): {
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "deaths": 0,
            "opponent_deaths": 0,
            "goals": 0,
            "bonuses": 0,
        }
        for faction in (0, 1)
    }
    final_scores: tuple[float, float] | None = None
    last_frame = 0
    for frame in iter_replay(path):
        if frame.frame >= 0:
            last_frame = max(last_frame, frame.frame)
        if frame.frame == -1:
            if len(frame.scores) != 2:
                raise DotoReplayError("final frame must contain two scores")
            final_scores = (frame.scores[0], frame.scores[1])
        for event in frame.events:
            if not isinstance(event, list) or not event or not isinstance(event[0], int):
                raise DotoReplayError("invalid replay event")
            event_type = event[0]
            counts[event_type] += 1
            if event_type == 2 and len(event) >= 4:
                victim, damage, source = int(event[1]), float(event[2]), int(event[3])
                source_faction, victim_faction = source % 2, victim % 2
                by_faction[str(source_faction)]["damage_dealt"] += damage
                by_faction[str(victim_faction)]["damage_taken"] += damage
            elif event_type == 3 and len(event) >= 2:
                dead_faction = int(event[1]) % 2
                by_faction[str(dead_faction)]["deaths"] += 1
                by_faction[str(1 - dead_faction)]["opponent_deaths"] += 1
            elif event_type == 10 and len(event) >= 2:
                by_faction[str(int(event[1]) % 2)]["goals"] += 1
            elif event_type == 12 and len(event) >= 2:
                by_faction[str(int(event[1]) % 2)]["bonuses"] += 1
    if final_scores is None:
        raise DotoReplayError("replay has no final frame")
    winner = 0 if final_scores[0] > final_scores[1] else 1 if final_scores[1] > final_scores[0] else None
    return ReplaySummary(
        final_scores=final_scores,
        winner=winner,
        last_frame=last_frame,
        event_counts={str(key): counts[key] for key in sorted(counts)},
        by_faction=by_faction,
    )
