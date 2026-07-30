"""Strict, secret-free configuration for HL research runs."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional


def _strict_values(
    cls: type,
    raw: Mapping[str, Any] | None,
    *,
    section: str,
) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{section} must be a mapping")
    allowed = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} fields: {unknown}")
    return dict(raw)


@dataclasses.dataclass(frozen=True)
class ProviderConfig:
    kind: str = "codex"
    model: Optional[str] = None
    review_model: Optional[str] = None
    reasoning_effort: str = "xhigh"
    env_key: str = "AGENTBENCH_API_KEY"
    base_url: Optional[str] = None
    wire_api: str = "responses"
    requires_openai_auth: bool = True
    disable_response_storage: bool = True
    network_access: str = "enabled"
    context_mode: str = "resumable"
    executable: str = "codex"

    def __post_init__(self) -> None:
        if self.kind != "codex":
            raise ValueError("only the codex provider is supported by this HL harness")
        if not self.env_key or "=" in self.env_key:
            raise ValueError("provider.env_key must name one environment variable")
        if self.context_mode not in {"resumable", "fresh"}:
            raise ValueError("provider.context_mode must be resumable or fresh")
        if self.wire_api not in {"responses", "chat"}:
            raise ValueError("provider.wire_api must be responses or chat")


@dataclasses.dataclass(frozen=True)
class IterationConfig:
    max_acts: Optional[int] = None
    candidates_per_act: int = 1

    def __post_init__(self) -> None:
        if self.max_acts is not None and self.max_acts < 1:
            raise ValueError("iteration.max_acts must be null or >= 1")
        if self.candidates_per_act < 1:
            raise ValueError("iteration.candidates_per_act must be >= 1")


@dataclasses.dataclass(frozen=True)
class RollbackConfig:
    enabled: bool = True
    policy: str = "champion_on_sustained_degradation"
    patience: int = 3
    score_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.policy != "champion_on_sustained_degradation":
            raise ValueError("unsupported rollback policy")
        if self.patience < 1:
            raise ValueError("rollback.patience must be >= 1")
        if not 0.0 <= self.score_margin <= 1.0:
            raise ValueError("rollback.score_margin must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class ExperienceConfig:
    enabled: bool = True
    compress_every_acts: int = 5

    def __post_init__(self) -> None:
        if self.compress_every_acts < 1:
            raise ValueError("experience.compress_every_acts must be >= 1")


@dataclasses.dataclass(frozen=True)
class MeasurementConfig:
    epsilon: float = 0.05
    local_policy_kl: bool = True
    occupancy_shift: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("measurement.epsilon must be in (0, 1)")


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    learning_opponent: str = "rank01"
    fixed_gate_seeds: tuple[int, ...] = ()
    certification_opponents: str = "all"
    certification_seeds: tuple[int, ...] = ()
    required_human_opponents: int = 15
    required_win_rate: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_gate_seeds", tuple(self.fixed_gate_seeds))
        object.__setattr__(self, "certification_seeds", tuple(self.certification_seeds))
        if self.required_human_opponents < 1:
            raise ValueError("evaluation.required_human_opponents must be >= 1")
        if not 0.0 <= self.required_win_rate <= 1.0:
            raise ValueError("evaluation.required_win_rate must be in [0, 1]")


@dataclasses.dataclass(frozen=True)
class HLRunConfig:
    game: str
    provider: ProviderConfig
    iteration: IterationConfig = dataclasses.field(default_factory=IterationConfig)
    rollback: RollbackConfig = dataclasses.field(default_factory=RollbackConfig)
    experience: ExperienceConfig = dataclasses.field(default_factory=ExperienceConfig)
    measurement: MeasurementConfig = dataclasses.field(default_factory=MeasurementConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if not self.game:
            raise ValueError("game is required")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HLRunConfig":
        values = _strict_values(cls, raw, section="run")
        if "provider" not in values:
            raise ValueError("provider is required")
        values["provider"] = ProviderConfig(
            **_strict_values(ProviderConfig, values["provider"], section="provider")
        )
        for name, section_cls in (
            ("iteration", IterationConfig),
            ("rollback", RollbackConfig),
            ("experience", ExperienceConfig),
            ("measurement", MeasurementConfig),
            ("evaluation", EvaluationConfig),
        ):
            values[name] = section_cls(
                **_strict_values(section_cls, values.get(name), section=name)
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable config containing no credential value."""

        return dataclasses.asdict(self)
