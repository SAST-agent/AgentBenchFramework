"""Immutable records shared by the Generals HL runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ProcessLimits:
    startup_timeout_s: float
    decision_timeout_s: float
    match_timeout_s: float
    max_packet_bytes: int
    max_artifact_bytes: int


@dataclass(frozen=True)
class OpponentSpec:
    opponent_id: str
    tier: str
    language: str
    source: Path
    argv: tuple[str, ...]
    build_argv: tuple[str, ...] = ()
    learning: bool = False


@dataclass(frozen=True)
class PilotConfig:
    benchmark_id: str
    engine_path: Path
    baseline_path: Path
    evaluation_seeds: tuple[int, ...]
    learning_seeds: tuple[int, ...]
    opponents: tuple[OpponentSpec, ...]
    limits: ProcessLimits

    @property
    def learning_opponents(self) -> tuple[OpponentSpec, ...]:
        return tuple(item for item in self.opponents if item.learning)


@dataclass(frozen=True)
class AssetLayout:
    root: Path
    engine_root: Path
    baseline_root: Path
    opponents: tuple[OpponentSpec, ...]
    engine_hash: str


@dataclass(frozen=True)
class AgentProcessSpec:
    agent_id: str
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]


@dataclass(frozen=True)
class MatchCase:
    case_id: str
    seed: int
    evaluated_seat: int
    opponent_id: str = "fake"
    opponent_tier: str = "unknown"


@dataclass(frozen=True)
class TurnRecord:
    step: int
    round_number: int
    player: int
    state_id_before: str
    state_before: Mapping[str, Any]
    commands: tuple[tuple[int, ...], ...]
    state_id_after: str
    state_after: Mapping[str, Any]


@dataclass(frozen=True)
class MatchResult:
    case_id: str
    valid: bool
    winner: int | None
    termination_type: str
    seed: int
    evaluated_seat: int
    turns: tuple[TurnRecord, ...]
    elapsed_time_s: float
    engine_hash: str
    error: str | None = None
