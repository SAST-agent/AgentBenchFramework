"""Strict, path-aware configuration for the DOTO iteration loop."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key_env: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    reasoning_effort: str | None = None
    stream: bool = True
    max_context_tokens: int = 1_000_000


@dataclass(frozen=True)
class OpponentConfig:
    name: str
    executable: Path | None = None
    source: Path | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (11,)
    seats: tuple[int, ...] = (0, 1)
    realtime_scale: float = 1.0


@dataclass(frozen=True)
class BudgetConfig:
    max_iterations: int
    max_builds: int
    max_rollouts: int
    max_episode_reads: int
    max_frame_reads: int
    max_total_tokens: int
    max_wall_seconds: float


@dataclass(frozen=True)
class LoopConfig:
    agent: str
    initial_player_ai: Path
    opponents: tuple[OpponentConfig, ...]
    llm: LLMConfig
    evaluation: EvaluationConfig
    budget: BudgetConfig

    @classmethod
    def load(cls, path: Path | str) -> "LoopConfig":
        config_path = Path(path).resolve()
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
        base = config_path.parent
        agent = _nonempty(raw.get("agent"), "agent")
        initial = _existing(base, raw.get("initial_player_ai"), "initial_player_ai")

        opponent_rows = raw.get("opponents")
        if not isinstance(opponent_rows, list) or not opponent_rows:
            raise ValueError("opponents must be a nonempty array of tables")
        opponents = []
        names = set()
        for row in opponent_rows:
            if not isinstance(row, dict):
                raise ValueError("each opponent must be a table")
            name = _nonempty(row.get("name"), "opponents.name")
            if name in names:
                raise ValueError(f"duplicate opponent name: {name}")
            names.add(name)
            executable = _optional_existing(base, row.get("executable"), "opponents.executable")
            source = _optional_existing(base, row.get("source"), "opponents.source")
            if executable is None and source is None:
                raise ValueError(f"opponent {name!r} needs executable or source")
            opponents.append(OpponentConfig(name, executable, source))

        llm_raw = _table(raw.get("llm"), "llm")
        stream_value = llm_raw.get("stream", True)
        if not isinstance(stream_value, bool):
            raise ValueError("llm.stream must be true or false")
        reasoning = llm_raw.get("reasoning_effort")
        if reasoning is not None and reasoning not in ("low", "medium", "high"):
            raise ValueError("llm.reasoning_effort must be low, medium, or high")
        max_tokens = llm_raw.get("max_tokens")
        llm = LLMConfig(
            base_url=_nonempty(llm_raw.get("base_url"), "llm.base_url"),
            api_key_env=str(llm_raw.get("api_key_env", "")).strip(),
            model=_nonempty(llm_raw.get("model"), "llm.model"),
            temperature=float(llm_raw.get("temperature", 0.0)),
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            timeout_seconds=float(llm_raw.get("timeout_seconds", 120.0)),
            reasoning_effort=reasoning,
            stream=stream_value,
            max_context_tokens=int(llm_raw.get("max_context_tokens", 1_000_000)),
        )
        for name in ("max_tokens", "timeout_seconds", "max_context_tokens"):
            value = getattr(llm, name)
            if value is not None and value <= 0:
                raise ValueError(f"llm.{name} must be positive")

        eval_raw = _table(raw.get("evaluation", {}), "evaluation")
        seeds = tuple(int(value) for value in eval_raw.get("seeds", [11]))
        seats = tuple(int(value) for value in eval_raw.get("seats", [0, 1]))
        realtime_scale = float(eval_raw.get("realtime_scale", 1.0))
        if not seeds:
            raise ValueError("evaluation.seeds must be nonempty")
        if not seats or len(set(seats)) != len(seats) or any(seat not in (0, 1) for seat in seats):
            raise ValueError("evaluation.seats must contain unique values from 0 and 1")
        if realtime_scale != 1.0:
            raise ValueError("evaluation.realtime_scale must equal 1.0 for formal runs")
        evaluation = EvaluationConfig(seeds, seats, realtime_scale)

        budget_raw = _table(raw.get("budget"), "budget")
        fields = (
            "max_iterations", "max_builds", "max_rollouts", "max_episode_reads",
            "max_frame_reads", "max_total_tokens", "max_wall_seconds",
        )
        missing = [name for name in fields if name not in budget_raw]
        if missing:
            raise ValueError(f"budget missing fields: {', '.join(missing)}")
        budget = BudgetConfig(
            *(float(budget_raw[name]) if name == "max_wall_seconds" else int(budget_raw[name])
              for name in fields)
        )
        for name, value in asdict(budget).items():
            if value <= 0:
                raise ValueError(f"budget.{name} must be positive")
        return cls(agent, initial, tuple(opponents), llm, evaluation, budget)

    def public_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "initial_player_ai": str(self.initial_player_ai),
            "opponents": [
                {"name": item.name,
                 "executable": str(item.executable) if item.executable else None,
                 "source": str(item.source) if item.source else None}
                for item in self.opponents
            ],
            "llm": asdict(self.llm),
            "evaluation": asdict(self.evaluation),
            "budget": asdict(self.budget),
        }


def _table(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table")
    return value


def _nonempty(value: Any, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _existing(base: Path, value: Any, name: str) -> Path:
    path = (base / _nonempty(value, name)).resolve()
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    return path


def _optional_existing(base: Path, value: Any, name: str) -> Path | None:
    if value is None:
        return None
    return _existing(base, value, name)
