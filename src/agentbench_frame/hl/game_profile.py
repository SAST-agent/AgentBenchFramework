"""Registered game-semantic bindings for the reproducible HL controller."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from agentbench_frame.hl.match_record import MatchRecord


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _texts(
    values: Sequence[str],
    *,
    field: str,
    unique: bool = False,
) -> tuple[str, ...]:
    result = tuple(_text(value, field=field) for value in values)
    if not result:
        raise ValueError(f"{field} cannot be empty")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


@dataclasses.dataclass(frozen=True)
class PromptProfile:
    """Game vocabulary and policy contracts interpolated into HL prompts."""

    candidate_label: str
    opponent_label: str
    roles: tuple[str, ...]
    policy_input: str
    output_contract: str
    planner_diversity: tuple[str, ...]
    prohibited_information: tuple[str, ...]
    policy_entry_symbol: str = "ai_func"
    candidate_source_relative: str = "ai.py"

    def __post_init__(self) -> None:
        for field in (
            "candidate_label",
            "opponent_label",
            "policy_input",
            "output_contract",
            "policy_entry_symbol",
            "candidate_source_relative",
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field=field))
        source = PurePosixPath(self.candidate_source_relative)
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("candidate_source_relative must stay inside the candidate")
        object.__setattr__(
            self,
            "roles",
            _texts(self.roles, field="roles", unique=True),
        )
        object.__setattr__(
            self,
            "planner_diversity",
            _texts(self.planner_diversity, field="planner_diversity"),
        )
        object.__setattr__(
            self,
            "prohibited_information",
            _texts(self.prohibited_information, field="prohibited_information"),
        )


@dataclasses.dataclass(frozen=True)
class MetricSchema:
    """Display names and role support for game-neutral research reports."""

    points_label: str
    dense_margin_label: str
    elo_label: str
    role_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("points_label", "dense_margin_label", "elo_label"):
            object.__setattr__(self, field, _text(getattr(self, field), field=field))
        object.__setattr__(
            self,
            "role_labels",
            _texts(self.role_labels, field="role_labels", unique=True),
        )


@dataclasses.dataclass(frozen=True)
class SmokeResult:
    status: str
    error: str | None
    artifacts: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"complete", "failed"}:
            raise ValueError(f"unknown smoke status: {self.status}")
        if self.status == "complete" and self.error is not None:
            raise ValueError("complete smoke result cannot contain an error")
        if self.status == "failed":
            _text(self.error, field="smoke error")
        if not isinstance(self.artifacts, Mapping):
            raise ValueError("smoke artifacts must be a mapping")
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(self.artifacts)),
        )


@dataclasses.dataclass(frozen=True)
class BehaviorComparison:
    status: str
    decision_count: int
    changed_action_count: int
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"complete", "failed"}:
            raise ValueError(f"unknown behavior comparison status: {self.status}")
        for field in ("decision_count", "changed_action_count"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.changed_action_count > self.decision_count:
            raise ValueError(
                "changed_action_count cannot exceed decision_count"
            )
        if self.status == "failed" and (
            self.decision_count != 0 or self.changed_action_count != 0
        ):
            raise ValueError(
                "failed behavior comparison cannot report decisions"
            )
        if not isinstance(self.details, Mapping):
            raise ValueError("behavior comparison details must be a mapping")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


SmokeVerifier = Callable[..., SmokeResult]
ReplayEvidenceBuilder = Callable[
    [Sequence[MatchRecord]],
    Sequence[Mapping[str, Any]],
]
BehaviorComparator = Callable[..., BehaviorComparison]
ActivationContractBuilder = Callable[..., Sequence[str]]


@dataclasses.dataclass(frozen=True)
class HLGameBindings:
    """Complete game-owned dependencies consumed by one local HL run."""

    prompt_profile: PromptProfile
    metric_schema: MetricSchema
    context_sources: Mapping[str, Path]
    candidate_template: Path
    evaluator: Any
    smoke_verifier: SmokeVerifier
    replay_evidence_builder: ReplayEvidenceBuilder
    behavior_comparator: BehaviorComparator
    activation_contract_builder: ActivationContractBuilder | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context_sources, Mapping) or not self.context_sources:
            raise ValueError("context_sources must be a non-empty mapping")
        sources: dict[str, Path] = {}
        for name, path in self.context_sources.items():
            sources[_text(name, field="context source name")] = Path(path)
        object.__setattr__(self, "context_sources", MappingProxyType(sources))
        object.__setattr__(self, "candidate_template", Path(self.candidate_template))
        if not callable(self.smoke_verifier):
            raise ValueError("smoke_verifier must be callable")
        if not callable(self.replay_evidence_builder):
            raise ValueError("replay_evidence_builder must be callable")
        if not callable(self.behavior_comparator):
            raise ValueError("behavior_comparator must be callable")
        if self.activation_contract_builder is not None and not callable(
            self.activation_contract_builder
        ):
            raise ValueError("activation_contract_builder must be callable")
        if not callable(getattr(self.evaluator, "evaluate", None)):
            raise ValueError("evaluator must define evaluate(version)")


@runtime_checkable
class GameProfile(Protocol):
    game_id: str
    required_local_paths: tuple[str, ...]
    optional_local_paths: tuple[str, ...]

    def prompt_profile(self) -> PromptProfile:
        ...

    def build_bindings(
        self,
        *,
        config: Any,
        run_root: Path,
    ) -> HLGameBindings:
        ...


_GAME_PROFILES: dict[str, GameProfile] = {}


def _load_builtin_profile(game_id: str) -> GameProfile | None:
    if game_id == "29_rollman":
        from agentbench_frame.games.rollman.hl_profile import RollmanHLProfile

        return RollmanHLProfile()
    if game_id == "30_antwar2":
        from agentbench_frame.games.antwar2.hl_profile import AntWar2HLProfile

        return AntWar2HLProfile()
    return None


def register_game_profile(profile: GameProfile) -> None:
    """Register exactly one profile for a stable game identifier."""

    game_id = _text(getattr(profile, "game_id", None), field="game_id")
    required_paths = getattr(profile, "required_local_paths", None)
    if not isinstance(required_paths, tuple):
        raise ValueError("required_local_paths must be a tuple")
    _texts(required_paths, field="required_local_paths", unique=True)
    optional_paths = getattr(profile, "optional_local_paths", ())
    if not isinstance(optional_paths, tuple):
        raise ValueError("optional_local_paths must be a tuple")
    if optional_paths:
        _texts(optional_paths, field="optional_local_paths", unique=True)
    overlap = sorted(set(required_paths) & set(optional_paths))
    if overlap:
        raise ValueError(f"local paths cannot be both required and optional: {overlap}")
    if not callable(getattr(profile, "prompt_profile", None)):
        raise ValueError("game profile must define prompt_profile()")
    if not callable(getattr(profile, "build_bindings", None)):
        raise ValueError("game profile must define build_bindings()")
    if game_id in _GAME_PROFILES:
        raise ValueError(f"game profile already registered: {game_id}")
    _GAME_PROFILES[game_id] = profile


def get_game_profile(game_id: str) -> GameProfile:
    """Resolve a registered profile or list the exact available identifiers."""

    key = _text(game_id, field="game_id")
    if key not in _GAME_PROFILES:
        builtin = _load_builtin_profile(key)
        if builtin is not None:
            register_game_profile(builtin)
    try:
        return _GAME_PROFILES[key]
    except KeyError as exc:
        available = ", ".join(sorted(_GAME_PROFILES)) or "none"
        raise KeyError(
            f"unknown game profile {key}; registered game profiles: {available}"
        ) from exc


def reset_game_profiles_for_testing() -> None:
    """Clear process-local registrations for isolated contract tests."""

    _GAME_PROFILES.clear()
