"""Immutable records shared by the Generals HL runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


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
