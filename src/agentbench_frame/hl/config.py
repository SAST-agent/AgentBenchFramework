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
    timeout_seconds: int = 600
    transport_retry_attempts: int = 3
    transport_retry_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.kind != "codex":
            raise ValueError("only the codex provider is supported by this HL harness")
        if not self.env_key or "=" in self.env_key:
            raise ValueError("provider.env_key must name one environment variable")
        if self.context_mode not in {"resumable", "fresh"}:
            raise ValueError("provider.context_mode must be resumable or fresh")
        if self.wire_api not in {"responses", "chat"}:
            raise ValueError("provider.wire_api must be responses or chat")
        if self.timeout_seconds < 1:
            raise ValueError("provider.timeout_seconds must be >= 1")
        if self.transport_retry_attempts < 0:
            raise ValueError("provider.transport_retry_attempts must be >= 0")
        if self.transport_retry_backoff_seconds < 0:
            raise ValueError(
                "provider.transport_retry_backoff_seconds must be >= 0"
            )


@dataclasses.dataclass(frozen=True)
class IterationConfig:
    max_acts: Optional[int] = None
    candidates_per_act: int = 1
    candidates_per_cycle: Optional[int] = None
    planner_enabled: bool = False
    reducer_enabled: bool = False
    quick_screen_seeds: int = 1
    finalist_count: int = 1
    finalist_seeds: int = 1

    def __post_init__(self) -> None:
        resolved_candidates = (
            self.candidates_per_act
            if self.candidates_per_cycle is None
            else self.candidates_per_cycle
        )
        if (
            self.candidates_per_cycle is not None
            and self.candidates_per_act != 1
            and self.candidates_per_act != self.candidates_per_cycle
        ):
            raise ValueError(
                "iteration candidates_per_act and candidates_per_cycle disagree"
            )
        object.__setattr__(self, "candidates_per_act", resolved_candidates)
        object.__setattr__(self, "candidates_per_cycle", resolved_candidates)
        if self.max_acts is not None and self.max_acts < 1:
            raise ValueError("iteration.max_acts must be null or >= 1")
        if resolved_candidates < 1:
            raise ValueError("iteration.candidates_per_cycle must be >= 1")
        if self.quick_screen_seeds < 1:
            raise ValueError("iteration.quick_screen_seeds must be >= 1")
        if not 1 <= self.finalist_count <= resolved_candidates:
            raise ValueError(
                "iteration.finalist_count must be within candidate count"
            )
        if self.finalist_seeds < 1:
            raise ValueError("iteration.finalist_seeds must be >= 1")


@dataclasses.dataclass(frozen=True)
class SelectionConfig:
    mode: str = "linear_lexicographic"
    exploration_debt_cycles: int = 3
    source_size_penalty: bool = False

    def __post_init__(self) -> None:
        if self.mode != "linear_lexicographic":
            raise ValueError("selection.mode must be linear_lexicographic")
        if self.exploration_debt_cycles < 1:
            raise ValueError("selection.exploration_debt_cycles must be >= 1")
        if self.source_size_penalty:
            raise ValueError("selection.source_size_penalty must be false")


@dataclasses.dataclass(frozen=True)
class ContextConfig:
    use_game_digest: bool = False
    research_state_max_bytes: int = 16384
    reduction_token_threshold: int = 250000

    def __post_init__(self) -> None:
        if self.research_state_max_bytes < 1024:
            raise ValueError("context.research_state_max_bytes must be >= 1024")
        if self.reduction_token_threshold < 1:
            raise ValueError("context.reduction_token_threshold must be >= 1")


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
class OriginConfig:
    mode: str = "model_bootstrap"
    source_run: Optional[str] = None
    source_version: Optional[str] = None
    reset_session: bool = True
    reset_experience: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"model_bootstrap", "imported_version"}:
            raise ValueError(
                "origin.mode must be model_bootstrap or imported_version"
            )
        has_run = bool(self.source_run)
        has_version = bool(self.source_version)
        if self.mode == "imported_version" and not (has_run and has_version):
            raise ValueError(
                "imported_version origin requires source_run and source_version"
            )
        if self.mode == "model_bootstrap" and (has_run or has_version):
            raise ValueError("model_bootstrap origin cannot define a source")


@dataclasses.dataclass(frozen=True)
class CurriculumConfig:
    mode: str = "fixed"
    target_order: str = "lowest_rank_first"
    preserve_passed_opponents: bool = True
    required_human_opponents: int = 16
    stagnation_patience: int = 4

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "weakest_failed"}:
            raise ValueError("curriculum.mode must be fixed or weakest_failed")
        if self.target_order != "lowest_rank_first":
            raise ValueError(
                "curriculum.target_order must be lowest_rank_first"
            )
        if not 1 <= self.required_human_opponents <= 16:
            raise ValueError(
                "curriculum.required_human_opponents must be in [1, 16]"
            )
        if self.stagnation_patience < 1:
            raise ValueError("curriculum.stagnation_patience must be >= 1")
        if self.mode == "weakest_failed" and not self.preserve_passed_opponents:
            raise ValueError(
                "weakest_failed curriculum requires preserve_passed_opponents"
            )


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    learning_opponent: str = "rank01"
    fixed_gate_seeds: tuple[int, ...] = ()
    certification_opponents: str = "all"
    certification_seeds: tuple[int, ...] = ()
    required_human_opponents: int = 15
    required_win_rate: float = 0.5
    full_pool_every_iteration: bool = True
    reporting_panel_every_cycle: bool = False
    reporting_seeds_per_opponent: int = 1
    max_parallel_matches: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed_gate_seeds", tuple(self.fixed_gate_seeds))
        object.__setattr__(self, "certification_seeds", tuple(self.certification_seeds))
        if self.required_human_opponents < 1:
            raise ValueError("evaluation.required_human_opponents must be >= 1")
        if not 0.0 <= self.required_win_rate <= 1.0:
            raise ValueError("evaluation.required_win_rate must be in [0, 1]")
        if self.reporting_seeds_per_opponent < 1:
            raise ValueError(
                "evaluation.reporting_seeds_per_opponent must be >= 1"
            )
        if self.max_parallel_matches < 1:
            raise ValueError("evaluation.max_parallel_matches must be >= 1")


@dataclasses.dataclass(frozen=True)
class HLRunConfig:
    game: str
    provider: ProviderConfig
    origin: OriginConfig = dataclasses.field(default_factory=OriginConfig)
    curriculum: CurriculumConfig = dataclasses.field(
        default_factory=CurriculumConfig
    )
    iteration: IterationConfig = dataclasses.field(default_factory=IterationConfig)
    selection: SelectionConfig = dataclasses.field(default_factory=SelectionConfig)
    context: ContextConfig = dataclasses.field(default_factory=ContextConfig)
    rollback: RollbackConfig = dataclasses.field(default_factory=RollbackConfig)
    experience: ExperienceConfig = dataclasses.field(default_factory=ExperienceConfig)
    measurement: MeasurementConfig = dataclasses.field(default_factory=MeasurementConfig)
    evaluation: EvaluationConfig = dataclasses.field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if not self.game:
            raise ValueError("game is required")
        if (
            self.curriculum.mode == "weakest_failed"
            and self.origin.mode == "imported_version"
        ):
            if not self.origin.reset_session:
                raise ValueError(
                    "weakest_failed curriculum requires origin.reset_session"
                )
            if not self.origin.reset_experience:
                raise ValueError(
                    "weakest_failed curriculum requires origin.reset_experience"
                )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HLRunConfig":
        values = _strict_values(cls, raw, section="run")
        if "provider" not in values:
            raise ValueError("provider is required")
        values["provider"] = ProviderConfig(
            **_strict_values(ProviderConfig, values["provider"], section="provider")
        )
        for name, section_cls in (
            ("origin", OriginConfig),
            ("curriculum", CurriculumConfig),
            ("iteration", IterationConfig),
            ("selection", SelectionConfig),
            ("context", ContextConfig),
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
