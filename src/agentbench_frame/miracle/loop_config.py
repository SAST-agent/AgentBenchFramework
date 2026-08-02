"""Miracle 高层迭代 Loop 的 TOML 配置。"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent_bridge import AGENTS


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key_env: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_seconds: float = 120.0
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...] = (11,)
    seats: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class BudgetConfig:
    max_iterations: int
    max_rollouts: int
    max_episode_reads: int
    max_decision_reads: int
    max_total_tokens: int
    max_wall_seconds: float


@dataclass(frozen=True)
class LoopConfig:
    agent: str
    initial_strategy: Path
    opponent: str
    llm: LLMConfig
    evaluation: EvaluationConfig
    budget: BudgetConfig

    @classmethod
    def from_toml(cls, path: Path | str) -> "LoopConfig":
        path = Path(path).resolve()
        with path.open("rb") as stream:
            raw = tomllib.load(stream)

        agent = str(raw.get("agent", "")).strip()
        if not agent:
            raise ValueError("agent must be nonempty")
        initial_value = str(raw.get("initial_strategy", "")).strip()
        initial = (path.parent / initial_value).resolve() if initial_value else Path()
        if not initial_value or not initial.is_file():
            raise ValueError(f"initial_strategy does not exist: {initial_value!r}")
        opponent = str(raw.get("opponent", "")).strip()
        if opponent not in AGENTS:
            raise ValueError(f"opponent is not registered: {opponent!r}")

        llm_raw = raw.get("llm", {})
        llm = LLMConfig(
            base_url=str(llm_raw.get("base_url", "")).strip(),
            api_key_env=str(llm_raw.get("api_key_env", "")).strip(),
            model=str(llm_raw.get("model", "")).strip(),
            temperature=float(llm_raw.get("temperature", 0.0)),
            max_tokens=int(llm_raw.get("max_tokens", 8192)),
            timeout_seconds=float(llm_raw.get("timeout_seconds", 120.0)),
            reasoning_effort=(
                str(llm_raw["reasoning_effort"]).strip()
                if llm_raw.get("reasoning_effort") is not None else None
            ),
        )
        if not llm.base_url or not llm.model:
            raise ValueError("llm.base_url and llm.model must be nonempty")
        if llm.max_tokens <= 0 or llm.timeout_seconds <= 0:
            raise ValueError("llm.max_tokens and llm.timeout_seconds must be positive")
        if llm.reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("llm.reasoning_effort must be low, medium, or high")

        eval_raw = raw.get("evaluation", {})
        evaluation = EvaluationConfig(
            seeds=tuple(int(value) for value in eval_raw.get("seeds", [11])),
            seats=tuple(int(value) for value in eval_raw.get("seats", [0, 1])),
        )
        if not evaluation.seeds:
            raise ValueError("evaluation.seeds must be nonempty")
        if not evaluation.seats or any(seat not in (0, 1) for seat in evaluation.seats):
            raise ValueError("evaluation.seats must be a nonempty subset of [0, 1]")

        budget_raw = raw.get("budget", {})
        required = (
            "max_iterations", "max_rollouts", "max_episode_reads",
            "max_decision_reads", "max_total_tokens", "max_wall_seconds",
        )
        missing = [name for name in required if name not in budget_raw]
        if missing:
            raise ValueError(f"budget missing fields: {', '.join(missing)}")
        budget = BudgetConfig(
            max_iterations=int(budget_raw["max_iterations"]),
            max_rollouts=int(budget_raw["max_rollouts"]),
            max_episode_reads=int(budget_raw["max_episode_reads"]),
            max_decision_reads=int(budget_raw["max_decision_reads"]),
            max_total_tokens=int(budget_raw["max_total_tokens"]),
            max_wall_seconds=float(budget_raw["max_wall_seconds"]),
        )
        for name, value in asdict(budget).items():
            if value <= 0:
                raise ValueError(f"budget.{name} must be positive")
        return cls(agent, initial, opponent, llm, evaluation, budget)

    def public_dict(self) -> dict:
        return {
            "agent": self.agent,
            "initial_strategy": str(self.initial_strategy),
            "opponent": self.opponent,
            "llm": asdict(self.llm),
            "evaluation": {
                "seeds": list(self.evaluation.seeds),
                "seats": list(self.evaluation.seats),
            },
            "budget": asdict(self.budget),
        }
