"""Strict game-neutral match records for HL evaluation and selection."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal


MatchStatus = Literal["complete", "failed", "timeout", "incomplete"]
MatchResult = Literal["win", "draw", "loss"]

_STATUSES = frozenset({"complete", "failed", "timeout", "incomplete"})
_RESULT_POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field=field)


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


@dataclasses.dataclass(frozen=True)
class MatchRecord:
    """One role-specific live or diagnostic game result."""

    schema_version: str
    game: str
    candidate: str
    opponent: str
    candidate_role: str
    seed: int
    status: MatchStatus
    result: MatchResult | None
    points: float | None
    candidate_score: float | None
    opponent_score: float | None
    dense_margin: float | None
    terminal_metrics: Mapping[str, float]
    rounds: int | None
    replay: str | None
    trace: str | None
    faults: tuple[str, ...]
    live_opponent: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MatchRecord":
        if not isinstance(raw, Mapping):
            raise ValueError("match record must be a mapping")
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - fields)
        missing = sorted(fields - set(raw))
        if unknown:
            raise ValueError(f"unknown match record fields: {unknown}")
        if missing:
            raise ValueError(f"missing match record fields: {missing}")

        schema_version = _required_text(
            raw["schema_version"], field="schema_version"
        )
        if schema_version != "1.0":
            raise ValueError("match record schema_version must be 1.0")
        status = str(raw["status"])
        if status not in _STATUSES:
            raise ValueError(f"unknown match status: {status}")
        seed = raw["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        live_opponent = raw["live_opponent"]
        if not isinstance(live_opponent, bool):
            raise ValueError("live_opponent must be boolean")

        terminal = raw["terminal_metrics"]
        if not isinstance(terminal, Mapping):
            raise ValueError("terminal_metrics must be a mapping")
        terminal_metrics: dict[str, float] = {}
        for key, value in terminal.items():
            name = _required_text(key, field="terminal metric name")
            terminal_metrics[name] = _finite_number(
                value,
                field=f"terminal metric {name}",
            )

        raw_faults = raw["faults"]
        if not isinstance(raw_faults, (list, tuple)):
            raise ValueError("faults must be a list")
        faults = tuple(
            _required_text(value, field="fault") for value in raw_faults
        )

        result_value = raw["result"]
        result: MatchResult | None
        if result_value is None:
            result = None
        elif result_value in _RESULT_POINTS:
            result = result_value
        else:
            raise ValueError(f"unknown match result: {result_value}")

        def optional_number(field: str) -> float | None:
            value = raw[field]
            if value is None:
                return None
            return _finite_number(value, field=field)

        points = optional_number("points")
        candidate_score = optional_number("candidate_score")
        opponent_score = optional_number("opponent_score")
        dense_margin = optional_number("dense_margin")
        rounds = raw["rounds"]
        if rounds is not None and (
            isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0
        ):
            raise ValueError("rounds must be a non-negative integer or null")

        if status == "complete":
            if result is None:
                raise ValueError("complete match requires result")
            for field, value in (
                ("points", points),
                ("candidate_score", candidate_score),
                ("opponent_score", opponent_score),
                ("dense_margin", dense_margin),
            ):
                if value is None:
                    raise ValueError(f"complete match requires {field}")
            if rounds is None:
                raise ValueError("complete match requires rounds")
            expected_points = _RESULT_POINTS[result]
            if points != expected_points:
                raise ValueError(
                    f"{result} result requires points {expected_points}"
                )
            if not 0.0 <= points <= 1.0:
                raise ValueError("points must be in [0, 1]")
        else:
            if not faults:
                raise ValueError("non-complete match requires a fault")
            if any(
                value is not None
                for value in (
                    result,
                    points,
                    candidate_score,
                    opponent_score,
                    dense_margin,
                    rounds,
                )
            ):
                raise ValueError(
                    "non-complete match cannot contain strategy outcome fields"
                )

        return cls(
            schema_version=schema_version,
            game=_required_text(raw["game"], field="game"),
            candidate=_required_text(raw["candidate"], field="candidate"),
            opponent=_required_text(raw["opponent"], field="opponent"),
            candidate_role=_required_text(
                raw["candidate_role"], field="candidate_role"
            ),
            seed=seed,
            status=status,
            result=result,
            points=points,
            candidate_score=candidate_score,
            opponent_score=opponent_score,
            dense_margin=dense_margin,
            terminal_metrics=terminal_metrics,
            rounds=rounds,
            replay=_optional_text(raw["replay"], field="replay"),
            trace=_optional_text(raw["trace"], field="trace"),
            faults=faults,
            live_opponent=live_opponent,
        )

    @property
    def comparison_key(self) -> tuple[str, str, int]:
        return self.opponent, self.candidate_role, self.seed

    @property
    def promotable(self) -> bool:
        return (
            self.status == "complete"
            and not self.faults
            and self.live_opponent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "game": self.game,
            "candidate": self.candidate,
            "opponent": self.opponent,
            "candidate_role": self.candidate_role,
            "seed": self.seed,
            "status": self.status,
            "result": self.result,
            "points": self.points,
            "candidate_score": self.candidate_score,
            "opponent_score": self.opponent_score,
            "dense_margin": self.dense_margin,
            "terminal_metrics": dict(self.terminal_metrics),
            "rounds": self.rounds,
            "replay": self.replay,
            "trace": self.trace,
            "faults": list(self.faults),
            "live_opponent": self.live_opponent,
        }


def completed_match_records(
    values: Iterable[MatchRecord | Mapping[str, Any]],
) -> tuple[MatchRecord, ...]:
    """Return only fault-free complete records from live adaptive opponents."""

    records = (
        value if isinstance(value, MatchRecord) else MatchRecord.from_mapping(value)
        for value in values
    )
    return tuple(record for record in records if record.promotable)
