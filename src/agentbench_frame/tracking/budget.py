"""Calibration/learning/evaluation/total resource budget accounting."""

from dataclasses import dataclass
from typing import Dict, Optional


_MISSING = object()
PHASES = ("calibration", "learning", "evaluation")


@dataclass
class _PhaseBudget:
    coding_agent_acts: int = 0
    episodes: int = 0
    env_steps: int = 0
    game_agent_decision_steps: int = 0
    primitive_commands: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    time_s: Optional[float] = None
    usage_seen: bool = False
    time_seen: bool = False

    def add(
        self,
        episodes: int,
        env_steps: int,
        game_agent_decision_steps: int = 0,
        primitive_commands: int = 0,
        coding_agent_acts: int = 0,
        prompt_tokens=_MISSING,
        completion_tokens=_MISSING,
        time_s=_MISSING,
    ) -> None:
        counts = (
            episodes,
            env_steps,
            game_agent_decision_steps,
            primitive_commands,
            coding_agent_acts,
        )
        if any(value < 0 for value in counts):
            raise ValueError("budget increments must be non-negative")
        self.coding_agent_acts += int(coding_agent_acts)
        self.episodes += int(episodes)
        self.env_steps += int(env_steps)
        self.game_agent_decision_steps += int(game_agent_decision_steps)
        self.primitive_commands += int(primitive_commands)
        if prompt_tokens is not _MISSING or completion_tokens is not _MISSING:
            prompt_value = None if prompt_tokens is _MISSING else prompt_tokens
            completion_value = None if completion_tokens is _MISSING else completion_tokens
            if not self.usage_seen:
                self.prompt_tokens = prompt_value
                self.completion_tokens = completion_value
            else:
                self.prompt_tokens = _add_optional(self.prompt_tokens, prompt_value)
                self.completion_tokens = _add_optional(self.completion_tokens, completion_value)
            self.usage_seen = True
        if time_s is not _MISSING:
            if not self.time_seen:
                self.time_s = time_s
            else:
                self.time_s = _add_optional(self.time_s, time_s)
            self.time_seen = True

    @property
    def total_tokens(self) -> Optional[int]:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


def _add_optional(current, increment):
    if current is None or increment is None:
        return None
    return current + increment


def _phase_value(phases: tuple[_PhaseBudget, ...], field: str):
    seen = [
        phase
        for phase in phases
        if (phase.usage_seen if field.endswith("tokens") else phase.time_seen)
    ]
    if not seen:
        return None
    value = getattr(seen[0], field)
    for phase in seen[1:]:
        value = _add_optional(value, getattr(phase, field))
    return value


class BudgetLedger:
    """Accumulate budgets without converting unknown provider usage to zero."""

    def __init__(self) -> None:
        self._phases: Dict[str, _PhaseBudget] = {
            phase: _PhaseBudget() for phase in PHASES
        }

    def add(
        self,
        phase: str,
        episodes: int = 0,
        env_steps: int = 0,
        game_agent_decision_steps: int = 0,
        primitive_commands: int = 0,
        coding_agent_acts: int = 0,
        prompt_tokens=_MISSING,
        completion_tokens=_MISSING,
        time_s=_MISSING,
    ) -> None:
        if phase not in self._phases:
            raise ValueError(
                "phase must be 'calibration', 'learning', or 'evaluation'"
            )
        if prompt_tokens is not _MISSING and prompt_tokens is not None and prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if completion_tokens is not _MISSING and completion_tokens is not None and completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if time_s is not _MISSING and time_s is not None and time_s < 0:
            raise ValueError("time_s must be non-negative")
        self._phases[phase].add(
            episodes,
            env_steps,
            game_agent_decision_steps,
            primitive_commands,
            coding_agent_acts,
            prompt_tokens,
            completion_tokens,
            time_s,
        )

    def snapshot(self) -> Dict[str, Optional[float]]:
        phases = tuple(self._phases[name] for name in PHASES)
        result: Dict[str, Optional[float]] = {}
        for name, phase in zip(PHASES, phases):
            result.update(
                {
                    f"{name}_coding_agent_acts": phase.coding_agent_acts,
                    f"{name}_episodes": phase.episodes,
                    f"{name}_env_steps": phase.env_steps,
                    f"{name}_game_agent_decision_steps": phase.game_agent_decision_steps,
                    f"{name}_primitive_commands": phase.primitive_commands,
                    f"{name}_prompt_tokens": phase.prompt_tokens,
                    f"{name}_completion_tokens": phase.completion_tokens,
                    f"{name}_total_tokens": phase.total_tokens,
                    f"{name}_time_s": phase.time_s,
                }
            )
        for field in (
            "coding_agent_acts",
            "episodes",
            "env_steps",
            "game_agent_decision_steps",
            "primitive_commands",
        ):
            result[f"total_{field}"] = sum(getattr(phase, field) for phase in phases)
        result.update(
            {
                "total_prompt_tokens": _phase_value(phases, "prompt_tokens"),
                "total_completion_tokens": _phase_value(phases, "completion_tokens"),
                "total_tokens": _phase_value(phases, "total_tokens"),
                "total_time_s": _phase_value(phases, "time_s"),
            }
        )
        return result
