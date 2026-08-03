"""Strict, secret-free configuration for HL research runs."""

from __future__ import annotations

import dataclasses
import math
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
class RolloutBudgetConfig:
    enabled: bool = False
    limit_tokens: Optional[int] = None
    reminder_at_remaining_tokens: tuple[int, ...] = ()
    sampling_token_weight: float = 1.0
    prefill_token_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reminder_at_remaining_tokens",
            tuple(self.reminder_at_remaining_tokens),
        )
        if self.enabled and (self.limit_tokens is None or self.limit_tokens < 1):
            raise ValueError(
                "provider.rollout_budget.limit_tokens must be >= 1 when enabled"
            )
        if self.limit_tokens is not None and self.limit_tokens < 1:
            raise ValueError("provider.rollout_budget.limit_tokens must be >= 1")
        reminders = self.reminder_at_remaining_tokens
        if any(value < 1 for value in reminders):
            raise ValueError(
                "provider.rollout_budget reminders must be positive"
            )
        if self.limit_tokens is not None and any(
            value >= self.limit_tokens for value in reminders
        ):
            raise ValueError(
                "provider.rollout_budget reminders must be below limit_tokens"
            )
        if any(left <= right for left, right in zip(reminders, reminders[1:])):
            raise ValueError(
                "provider.rollout_budget reminders must be strictly descending"
            )
        for name in ("sampling_token_weight", "prefill_token_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"provider.rollout_budget.{name} must be finite and > 0")


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
    expected_cli_version: Optional[str] = None
    rollout_budget: RolloutBudgetConfig = dataclasses.field(
        default_factory=RolloutBudgetConfig
    )
    timeout_seconds: int = 600
    idle_timeout_seconds: int = 360
    transport_retry_attempts: int = 3
    transport_retry_backoff_seconds: float = 2.0
    rate_limit_cooldown_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.kind != "codex":
            raise ValueError("only the codex provider is supported by this HL harness")
        if self.expected_cli_version is not None and not self.expected_cli_version:
            raise ValueError("provider.expected_cli_version cannot be empty")
        if not self.env_key or "=" in self.env_key:
            raise ValueError("provider.env_key must name one environment variable")
        if self.context_mode not in {"resumable", "fresh"}:
            raise ValueError("provider.context_mode must be resumable or fresh")
        if self.wire_api not in {"responses", "chat"}:
            raise ValueError("provider.wire_api must be responses or chat")
        if self.timeout_seconds < 1:
            raise ValueError("provider.timeout_seconds must be >= 1")
        if self.idle_timeout_seconds < 1:
            raise ValueError("provider.idle_timeout_seconds must be >= 1")
        if self.idle_timeout_seconds > self.timeout_seconds:
            raise ValueError(
                "provider.idle_timeout_seconds cannot exceed timeout_seconds"
            )
        if self.transport_retry_attempts < 0:
            raise ValueError("provider.transport_retry_attempts must be >= 0")
        if self.transport_retry_backoff_seconds < 0:
            raise ValueError(
                "provider.transport_retry_backoff_seconds must be >= 0"
            )
        if self.rate_limit_cooldown_seconds < 0:
            raise ValueError(
                "provider.rate_limit_cooldown_seconds must be >= 0"
            )


@dataclasses.dataclass(frozen=True)
class IterationConfig:
    max_acts: Optional[int] = None
    candidates_per_act: int = 1
    candidates_per_cycle: Optional[int] = None
    planner_enabled: bool = False
    reducer_enabled: bool = False
    scope_contract_required: bool = True
    repair_enabled: bool = False
    repair_top_k: int = 2
    repair_rounds: int = 1
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
        if self.repair_rounds not in {0, 1}:
            raise ValueError("iteration.repair_rounds must be 0 or 1")
        if self.repair_enabled:
            if not self.planner_enabled:
                raise ValueError("iteration repair requires planner_enabled")
            if not 1 <= self.repair_top_k <= resolved_candidates:
                raise ValueError(
                    "iteration.repair_top_k must be within candidate count"
                )
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
    reset_research_state: bool = True

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
    hard_opponents: tuple[str, ...] = ()
    training_seeds: tuple[int, ...] = ()
    validation_seeds: tuple[int, ...] = ()
    training_rotation_stride: int = 1
    certification_opponents: str = "all"
    certification_seeds: tuple[int, ...] = ()
    certification_wins_required: int = 1
    required_human_opponents: int = 15
    required_win_rate: float = 0.5
    full_pool_every_iteration: bool = True
    reporting_panel_every_cycle: bool = False
    reporting_seeds_per_opponent: int = 1
    max_parallel_matches: int = 1

    def __post_init__(self) -> None:
        for field in (
            "fixed_gate_seeds",
            "hard_opponents",
            "training_seeds",
            "validation_seeds",
            "certification_seeds",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not self.fixed_gate_seeds and (
            self.training_seeds or self.validation_seeds
        ):
            object.__setattr__(
                self,
                "fixed_gate_seeds",
                (*self.training_seeds, *self.validation_seeds),
            )
        if self.training_rotation_stride < 1:
            raise ValueError(
                "evaluation.training_rotation_stride must be >= 1"
            )
        if self.hard_opponents:
            if (
                len(self.hard_opponents) != 2
                or len(set(self.hard_opponents)) != 2
            ):
                raise ValueError(
                    "evaluation.hard_opponents must contain two distinct opponents"
                )
            if not self.training_seeds or not self.validation_seeds:
                raise ValueError(
                    "generalizable evaluation requires training and validation seeds"
                )
            if (
                len(self.certification_seeds) != 5
                or len(set(self.certification_seeds)) != 5
            ):
                raise ValueError(
                    "generalizable evaluation requires five unique certification seeds"
                )
            seed_sets = (
                set(self.training_seeds),
                set(self.validation_seeds),
                set(self.certification_seeds),
            )
            if any(
                left & right
                for index, left in enumerate(seed_sets)
                for right in seed_sets[index + 1 :]
            ):
                raise ValueError(
                    "training, validation, and certification seeds must be disjoint"
                )
            if not 1 <= self.certification_wins_required <= 5:
                raise ValueError(
                    "evaluation.certification_wins_required must be in [1, 5]"
                )
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

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HLRunConfig":
        values = _strict_values(cls, raw, section="run")
        if "provider" not in values:
            raise ValueError("provider is required")
        provider_values = _strict_values(
            ProviderConfig,
            values["provider"],
            section="provider",
        )
        provider_values["rollout_budget"] = RolloutBudgetConfig(
            **_strict_values(
                RolloutBudgetConfig,
                provider_values.get("rollout_budget"),
                section="provider.rollout_budget",
            )
        )
        values["provider"] = ProviderConfig(**provider_values)
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
