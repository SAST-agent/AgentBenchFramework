"""Learning/evaluation/total resource budget accounting."""

from dataclasses import dataclass
from typing import Dict, Optional


_MISSING = object()


@dataclass
class _PhaseBudget:
    coding_agent_acts: int = 0
    episodes: int = 0
    env_steps: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    time_s: Optional[float] = None
    usage_seen: bool = False
    time_seen: bool = False

    def add(self, episodes: int, env_steps: int, coding_agent_acts: int = 0,
            prompt_tokens=_MISSING, completion_tokens=_MISSING,
            time_s=_MISSING) -> None:
        if episodes < 0 or env_steps < 0 or coding_agent_acts < 0:
            raise ValueError("budget increments must be non-negative")
        self.coding_agent_acts += int(coding_agent_acts)
        self.episodes += int(episodes)
        self.env_steps += int(env_steps)
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


def _phase_value(learning: _PhaseBudget, evaluation: _PhaseBudget, field: str):
    learning_seen = learning.usage_seen if field.endswith("tokens") else learning.time_seen
    evaluation_seen = evaluation.usage_seen if field.endswith("tokens") else evaluation.time_seen
    learning_value = getattr(learning, field)
    evaluation_value = getattr(evaluation, field)
    if learning_seen and evaluation_seen:
        return _add_optional(learning_value, evaluation_value)
    if learning_seen:
        return learning_value
    if evaluation_seen:
        return evaluation_value
    return None


class BudgetLedger:
    """Accumulate budgets without converting unknown provider usage to zero."""

    def __init__(self) -> None:
        self._phases: Dict[str, _PhaseBudget] = {
            "learning": _PhaseBudget(),
            "evaluation": _PhaseBudget(),
        }

    def add(
        self,
        phase: str,
        episodes: int = 0,
        env_steps: int = 0,
        coding_agent_acts: int = 0,
        prompt_tokens=_MISSING,
        completion_tokens=_MISSING,
        time_s=_MISSING,
    ) -> None:
        if phase not in self._phases:
            raise ValueError("phase must be 'learning' or 'evaluation'")
        if prompt_tokens is not _MISSING and prompt_tokens is not None and prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if completion_tokens is not _MISSING and completion_tokens is not None and completion_tokens < 0:
            raise ValueError("completion_tokens must be non-negative")
        if time_s is not _MISSING and time_s is not None and time_s < 0:
            raise ValueError("time_s must be non-negative")
        self._phases[phase].add(
            episodes, env_steps, coding_agent_acts,
            prompt_tokens, completion_tokens, time_s
        )

    def snapshot(self) -> Dict[str, Optional[float]]:
        learning = self._phases["learning"]
        evaluation = self._phases["evaluation"]
        return {
            "learning_coding_agent_acts": learning.coding_agent_acts,
            "learning_episodes": learning.episodes,
            "learning_env_steps": learning.env_steps,
            "evaluation_coding_agent_acts": evaluation.coding_agent_acts,
            "evaluation_episodes": evaluation.episodes,
            "evaluation_env_steps": evaluation.env_steps,
            "total_coding_agent_acts": learning.coding_agent_acts + evaluation.coding_agent_acts,
            "total_episodes": learning.episodes + evaluation.episodes,
            "total_env_steps": learning.env_steps + evaluation.env_steps,
            "learning_prompt_tokens": learning.prompt_tokens,
            "learning_completion_tokens": learning.completion_tokens,
            "learning_total_tokens": learning.total_tokens,
            "evaluation_prompt_tokens": evaluation.prompt_tokens,
            "evaluation_completion_tokens": evaluation.completion_tokens,
            "evaluation_total_tokens": evaluation.total_tokens,
            "total_prompt_tokens": _phase_value(learning, evaluation, "prompt_tokens"),
            "total_completion_tokens": _phase_value(learning, evaluation, "completion_tokens"),
            "total_tokens": _phase_value(learning, evaluation, "total_tokens"),
            "learning_time_s": learning.time_s,
            "evaluation_time_s": evaluation.time_s,
            "total_time_s": _phase_value(learning, evaluation, "time_s"),
        }
