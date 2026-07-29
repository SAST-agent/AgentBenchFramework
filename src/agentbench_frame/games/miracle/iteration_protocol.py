"""Fail-closed, fake-testable iteration contracts for ``24_miracle``.

This module validates immutable inputs and storage operations only.  It never
starts a Judge, opponent, provider, match, or session.  Production approval
tables intentionally remain empty until separately reviewed human assets and
replay evidence exist.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from agentbench_frame.games.miracle.research_protocol import (
    BENCHMARK_VERSION,
    PROTOCOL_VERSION,
    SEED_MAX,
    SEED_MIN,
    TEST_OPPONENTS,
    TRAJECTORY_KL_EPSILON,
    TRAIN_OPPONENTS,
    VALIDATION_OPPONENTS,
    research_manifest_sha256,
)

ITERATION_PROTOCOL_VERSION = "24-miracle-iteration-v3"
BOOTSTRAP_TEMPLATE_VERSION = "24m-minimal-bootstrap-v1"
HUMAN_CHAMPION_SCHEMA_VERSION = "24-miracle-human-champion-v1"
HUMAN_REPLAY_SKILL_SCHEMA_VERSION = "24-miracle-human-replay-skill-v1"
EXPERIENCE_SKILL_SCHEMA_VERSION = "24-miracle-experience-skill-v2"
MATCH_PLAN_SCHEMA_VERSION = "24-miracle-match-plan-v1"
REPLAY_EVIDENCE_SCHEMA_VERSION = "24-miracle-replay-evidence-v1"
CANDIDATE_EVALUATION_PLAN_SCHEMA_VERSION = "24-miracle-candidate-evaluation-plan-v1"
EVALUATION_EVIDENCE_SCHEMA_VERSION = "24-miracle-evaluation-evidence-v1"
ITERATION_ACCEPTANCE_SCHEMA_VERSION = "24-miracle-iteration-acceptance-v1"
STRATEGY_SCHEMA_VERSION = "24-miracle-strategy-v3"
ROLLBACK_SCHEMA_VERSION = "24-miracle-rollback-v1"
LEGAL_ACTION_UNIT = "one_legal_atomic_judge_command"
KL_DIRECTION = "new||old"
KL_ROLLOUT_SOURCE = "new_policy"
DECISION_CHANGE_RATE = "not_collected"
HUMAN_AUTHORED_CONTENT_REQUIRED = "HUMAN_AUTHORED_CONTENT_REQUIRED"

# This identity was independently calculated from the checked-in template.
APPROVED_BOOTSTRAP_SHA256 = (
    "5907e5135c9174e36f8b44ae90be4e953feca6a6f33773102b50d699ef7b010e"
)

# Production approvals are code-reviewed control-plane state.  Run files and
# public APIs cannot override them.  Tests may monkeypatch them temporarily.
CURRENT_APPROVED_HUMAN_CHAMPION_SHA256: str | None = None
APPROVED_HUMAN_REPLAY_SKILL_SHA256: frozenset[str] = frozenset()
APPROVED_MATCH_PLAN_MANIFEST_SHA256: frozenset[str] = frozenset()
APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256: frozenset[str] = frozenset()
APPROVED_CANDIDATE_EVALUATION_PLAN_SHA256: frozenset[str] = frozenset()
APPROVED_EVALUATION_EVIDENCE_MANIFEST_SHA256: frozenset[str] = frozenset()
APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256: frozenset[str] = frozenset()

_CHANGE_OPERATIONS = frozenset({"add", "replace", "merge", "delete", "simplify"})
_REPLAY_ROLES = frozenset({"train", "validation", "test"})
_EVALUATION_ROLES = frozenset({"evaluation"})
_INTERPRETABLE_CATEGORIES = frozenset(
    {
        "rule_table",
        "finite_state_machine",
        "deterministic_heuristic_planner",
        "scoring_function",
        "structured_combination",
    }
)
_ALLOWED_IMPORT_ROOTS = frozenset(
    {"collections", "dataclasses", "functools", "itertools", "math", "operator", "statistics", "typing"}
)
_FORBIDDEN_CALLS = frozenset(
    {"__import__", "breakpoint", "compile", "delattr", "eval", "exec", "getattr", "globals", "input", "locals", "open", "setattr", "vars"}
)
_FORBIDDEN_ATTRIBUTES = frozenset(
    {"Popen", "connect", "open", "popen", "read", "read_bytes", "read_text", "run", "spawn", "system", "urlopen", "write", "write_bytes", "write_text"}
)
_ALLOWED_SOURCE_SUFFIXES = frozenset({".json", ".py", ".txt"})
_LOGICAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LifecycleState(str, Enum):
    PLANNED = "PLANNED"
    MATCH_READY = "MATCH_READY"
    BASELINE_STORED = "BASELINE_STORED"
    REPLAY_CAPTURED_UNAPPROVED = "REPLAY_CAPTURED_UNAPPROVED"
    REPLAY_APPROVED = "REPLAY_APPROVED"
    LEARNING_READY = "LEARNING_READY"
    CANDIDATE_STORED = "CANDIDATE_STORED"
    EVALUATION_READY = "EVALUATION_READY"
    COMPLETED = "COMPLETED"
    INCOMPLETE = "INCOMPLETE"


class IterationBlockedError(ValueError):
    """Raised before any mutable run resource is created."""


class IterationPreflightError(IterationBlockedError):
    """Raised when an immutable iteration input fails closed."""


class HumanAuthoredContentRequired(IterationPreflightError):
    """Raised when an approved human-authored asset is unavailable."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_object_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise IterationPreflightError(f"{label} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise IterationPreflightError(f"{label} must contain a top-level object")
    return value


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
            raise OSError("missing or symbolic-link asset")
        return path.read_bytes()
    except OSError as exc:
        raise IterationPreflightError(f"{label} is missing or unreadable") from exc


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _logical_id(value: Any, label: str) -> str:
    value = _required_text(value, label)
    if not _LOGICAL_ID.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be a path-safe logical ID")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text_tuple(values: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{label} must be a list of strings")
    result = tuple(_required_text(value, label) for value in values)
    if nonempty and not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _strict_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _strict_score(value: Any, label: str) -> float:
    result = _strict_finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _relative_asset(value: Any, label: str) -> str:
    value = _required_text(value, label).replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a safe relative path")
    if ":" in value:
        raise ValueError(f"{label} must be a safe relative path")
    return path.as_posix()


def _contained(root: Path, relative: str, label: str) -> Path:
    root = root.resolve()
    target = (root / _relative_asset(relative, label)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its approved root") from exc
    return target


@dataclass(frozen=True)
class BootstrapTemplate:
    version: str
    path: Path
    sha256: str

    def validate(self) -> None:
        if self.version != BOOTSTRAP_TEMPLATE_VERSION:
            raise IterationPreflightError("bootstrap template version is not approved")
        if self.sha256 != APPROVED_BOOTSTRAP_SHA256:
            raise IterationPreflightError("bootstrap template approved SHA mismatch")
        try:
            actual = _file_sha256(self.path)
        except (OSError, TypeError) as exc:
            raise IterationPreflightError("bootstrap template is missing or unreadable") from exc
        if actual != APPROVED_BOOTSTRAP_SHA256:
            raise IterationPreflightError("bootstrap template does not match approved SHA")


def default_bootstrap_template() -> BootstrapTemplate:
    return BootstrapTemplate(
        version=BOOTSTRAP_TEMPLATE_VERSION,
        path=Path(__file__).with_name("minimal_bootstrap_strategy.py"),
        sha256=APPROVED_BOOTSTRAP_SHA256,
    )


def validate_bootstrap_template(template: BootstrapTemplate) -> None:
    if not isinstance(template, BootstrapTemplate):
        raise IterationPreflightError("bootstrap template is required")
    template.validate()


def bootstrap_strategy_source_sha256(template: BootstrapTemplate) -> str:
    """Hash the canonical v0 source snapshot after fixed-template validation."""
    validate_bootstrap_template(template)
    try:
        source = template.path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IterationPreflightError("bootstrap template is not readable UTF-8") from exc
    return _digest({"main.py": source})


@dataclass(frozen=True)
class HumanChampion:
    logical_id: str
    version: str
    artifact_path: str
    artifact_sha256: str
    provenance: str
    qualification_status: str
    qualification_evidence: tuple[str, ...]
    protocol_version: str
    benchmark_version: str
    frozen: bool
    schema_version: str = HUMAN_CHAMPION_SCHEMA_VERSION
    sha256: str = ""
    _bundle_root: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def create(cls, **values: Any) -> "HumanChampion":
        champion = cls(**values)
        champion.validate(check_artifact=False)
        return replace(champion, sha256=_digest(champion._unsigned_dict()))

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "version": self.version,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "provenance": self.provenance,
            "qualification_status": self.qualification_status,
            "qualification_evidence": list(self.qualification_evidence),
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "frozen": self.frozen,
        }

    def validate(self, *, check_artifact: bool = True) -> None:
        if self.schema_version != HUMAN_CHAMPION_SCHEMA_VERSION:
            raise IterationPreflightError("human champion schema version mismatch")
        _logical_id(self.logical_id, "champion logical ID")
        _logical_id(self.version, "champion version")
        _relative_asset(self.artifact_path, "champion artifact path")
        _sha256(self.artifact_sha256, "champion artifact SHA")
        _required_text(self.provenance, "champion provenance")
        if self.qualification_status != "qualified":
            raise IterationPreflightError("human champion is not qualified")
        _text_tuple(self.qualification_evidence, "champion qualification evidence", nonempty=True)
        if self.protocol_version != PROTOCOL_VERSION or self.benchmark_version != BENCHMARK_VERSION:
            raise IterationPreflightError("human champion protocol/benchmark version mismatch")
        if self.frozen is not True:
            raise IterationPreflightError("human champion must be frozen")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("human champion descriptor SHA mismatch")
        if check_artifact:
            if self._bundle_root is None:
                raise IterationPreflightError("human champion bundle root is unavailable")
            artifact = _contained(self._bundle_root, self.artifact_path, "champion artifact path")
            if _file_sha256(Path(_readable_file(artifact, "human champion artifact"))) != self.artifact_sha256:
                raise IterationPreflightError("human champion artifact SHA mismatch")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate(check_artifact=False)
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    def public_identity(self) -> dict[str, Any]:
        return {**self._unsigned_dict(), "sha256": self.sha256}

    @classmethod
    def from_bytes(cls, payload: bytes, bundle_root: Path) -> "HumanChampion":
        value = _strict_object_bytes(payload, "human champion descriptor")
        champion = cls(
            logical_id=value.get("logical_id"), version=value.get("version"),
            artifact_path=value.get("artifact_path"), artifact_sha256=value.get("artifact_sha256"),
            provenance=value.get("provenance"), qualification_status=value.get("qualification_status"),
            qualification_evidence=_text_tuple(value.get("qualification_evidence", []), "champion qualification evidence"),
            protocol_version=value.get("protocol_version"), benchmark_version=value.get("benchmark_version"),
            frozen=value.get("frozen"), schema_version=value.get("schema_version"),
            sha256=value.get("sha256", ""), _bundle_root=bundle_root,
        )
        champion.validate()
        if payload != champion.canonical_bytes():
            raise IterationPreflightError("human champion descriptor is not canonical")
        return champion


def _readable_file(path: Path, label: str) -> Path:
    _read_bytes(path, label)
    return path


def load_human_champion(path: Path) -> HumanChampion:
    champion = HumanChampion.from_bytes(_read_bytes(path, "human champion descriptor"), path.parent)
    if CURRENT_APPROVED_HUMAN_CHAMPION_SHA256 is None or champion.sha256 != CURRENT_APPROVED_HUMAN_CHAMPION_SHA256:
        raise IterationPreflightError("human champion descriptor is not the current approved champion")
    return champion


@dataclass(frozen=True)
class HumanReplaySkill:
    logical_id: str
    version: str
    content: Mapping[str, Any]
    provenance: str
    created_by: str
    schema_version: str = HUMAN_REPLAY_SKILL_SCHEMA_VERSION
    sha256: str = ""

    @classmethod
    def create(cls, **values: Any) -> "HumanReplaySkill":
        skill = cls(**values)
        skill.validate()
        return replace(skill, sha256=_digest(skill._unsigned_dict()))

    def _unsigned_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "logical_id": self.logical_id, "version": self.version,
                "content": dict(self.content), "provenance": self.provenance, "created_by": self.created_by}

    def validate(self) -> None:
        if self.schema_version != HUMAN_REPLAY_SKILL_SCHEMA_VERSION:
            raise IterationPreflightError("human replay Skill schema mismatch")
        _logical_id(self.logical_id, "human replay Skill logical ID")
        _logical_id(self.version, "human replay Skill version")
        if not isinstance(self.content, Mapping) or not self.content:
            raise IterationPreflightError("human replay Skill content must be a non-empty object")
        _required_text(self.provenance, "human replay Skill provenance")
        _logical_id(self.created_by, "human replay Skill creator")
        try:
            _canonical_bytes(dict(self.content))
        except (TypeError, ValueError) as exc:
            raise IterationPreflightError("human replay Skill content is not canonical JSON") from exc
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("human replay Skill SHA mismatch")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate()
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    @classmethod
    def from_bytes(cls, payload: bytes) -> "HumanReplaySkill":
        value = _strict_object_bytes(payload, "human replay Skill")
        skill = cls(value.get("logical_id"), value.get("version"), value.get("content"), value.get("provenance"),
                    value.get("created_by"), value.get("schema_version"), value.get("sha256", ""))
        skill.validate()
        if payload != skill.canonical_bytes():
            raise IterationPreflightError("human replay Skill is not canonical")
        return skill


def _strict_seed(value: Any, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not SEED_MIN <= value <= SEED_MAX
    ):
        raise ValueError(f"{label} seed must be an integer in [{SEED_MIN}, {SEED_MAX}]")
    return value


@dataclass(frozen=True)
class MatchPlanCase:
    case_id: str
    role: str
    evaluated_agent_camp: int
    map_type: int
    day_time: int
    repeat: int
    logic_seed: int
    evaluated_agent_seed: int
    opponent_seed: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _logical_id(self.case_id, "match-plan case ID")
        if self.role not in _REPLAY_ROLES:
            raise ValueError("match-plan case role is invalid")
        for label, value in (
            ("camp", self.evaluated_agent_camp),
            ("map type", self.map_type),
            ("day time", self.day_time),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
                raise ValueError(f"match-plan {label} must be strict integer 0 or 1")
        if not isinstance(self.repeat, int) or isinstance(self.repeat, bool) or self.repeat not in (1, 2, 3):
            raise ValueError("match-plan repeat must be strict integer 1, 2, or 3")
        _strict_seed(self.logic_seed, "logic")
        _strict_seed(self.evaluated_agent_seed, "evaluated-agent")
        _strict_seed(self.opponent_seed, "opponent")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MatchPlanCase":
        if not isinstance(value, Mapping):
            raise IterationPreflightError("match-plan case must be an object")
        try:
            return cls(**{name: value.get(name) for name in cls.__dataclass_fields__})
        except ValueError as exc:
            raise IterationPreflightError(str(exc)) from exc


@dataclass(frozen=True)
class MatchPlanManifest:
    champion_descriptor_sha256: str
    human_replay_skill_sha256: str
    evaluated_policy_version: str
    evaluated_policy_source_sha256: str
    cases: tuple[MatchPlanCase, ...]
    bootstrap_sha256: str
    protocol_version: str
    benchmark_version: str
    iteration_protocol_version: str
    research_manifest_sha256: str
    iteration_manifest_sha256: str
    lifecycle_state: str = LifecycleState.PLANNED.value
    schema_version: str = MATCH_PLAN_SCHEMA_VERSION
    sha256: str = ""

    @classmethod
    def create(cls, **values: Any) -> "MatchPlanManifest":
        manifest = cls(**values)
        manifest.validate()
        return replace(manifest, sha256=_digest(manifest._unsigned_dict()))

    @property
    def state(self) -> LifecycleState:
        return LifecycleState(self.lifecycle_state)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lifecycle_state": self.lifecycle_state,
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "iteration_protocol_version": self.iteration_protocol_version,
            "research_manifest_sha256": self.research_manifest_sha256,
            "iteration_manifest_sha256": self.iteration_manifest_sha256,
            "champion_descriptor_sha256": self.champion_descriptor_sha256,
            "human_replay_skill_sha256": self.human_replay_skill_sha256,
            "evaluated_policy_version": self.evaluated_policy_version,
            "evaluated_policy_source_sha256": self.evaluated_policy_source_sha256,
            "bootstrap_sha256": self.bootstrap_sha256,
            "cases": [case.to_dict() for case in self.cases],
        }

    def validate(self) -> None:
        if self.schema_version != MATCH_PLAN_SCHEMA_VERSION:
            raise IterationPreflightError("match-plan schema mismatch")
        if self.lifecycle_state != LifecycleState.PLANNED.value:
            raise IterationPreflightError("match-plan lifecycle state must be PLANNED")
        if self.protocol_version != PROTOCOL_VERSION or self.benchmark_version != BENCHMARK_VERSION:
            raise IterationPreflightError("match-plan protocol/benchmark version mismatch")
        if self.iteration_protocol_version != ITERATION_PROTOCOL_VERSION:
            raise IterationPreflightError("match-plan iteration protocol version mismatch")
        if self.research_manifest_sha256 != research_manifest_sha256():
            raise IterationPreflightError("match-plan research manifest SHA mismatch")
        if self.iteration_manifest_sha256 != hashlib.sha256(canonical_iteration_protocol_manifest_bytes()).hexdigest():
            raise IterationPreflightError("match-plan iteration manifest SHA mismatch")
        _sha256(self.champion_descriptor_sha256, "match-plan champion descriptor SHA")
        _sha256(self.human_replay_skill_sha256, "match-plan human Skill SHA")
        _logical_id(self.evaluated_policy_version, "match-plan policy version")
        _sha256(self.evaluated_policy_source_sha256, "match-plan policy source SHA")
        if self.bootstrap_sha256 != APPROVED_BOOTSTRAP_SHA256:
            raise IterationPreflightError("match-plan bootstrap SHA mismatch")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise IterationPreflightError("match-plan cases must be non-empty")
        for case in self.cases:
            if not isinstance(case, MatchPlanCase):
                raise IterationPreflightError("match-plan case has invalid type")
            case.validate()
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise IterationPreflightError("match-plan has duplicate case identity")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("match-plan SHA mismatch")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate()
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    @classmethod
    def from_bytes(cls, payload: bytes) -> "MatchPlanManifest":
        value = _strict_object_bytes(payload, "match-plan manifest")
        manifest = cls(
            champion_descriptor_sha256=value.get("champion_descriptor_sha256"),
            human_replay_skill_sha256=value.get("human_replay_skill_sha256"),
            evaluated_policy_version=value.get("evaluated_policy_version"),
            evaluated_policy_source_sha256=value.get("evaluated_policy_source_sha256"),
            cases=tuple(MatchPlanCase.from_dict(item) for item in value.get("cases", [])),
            bootstrap_sha256=value.get("bootstrap_sha256"),
            protocol_version=value.get("protocol_version"),
            benchmark_version=value.get("benchmark_version"),
            iteration_protocol_version=value.get("iteration_protocol_version"),
            research_manifest_sha256=value.get("research_manifest_sha256"),
            iteration_manifest_sha256=value.get("iteration_manifest_sha256"),
            lifecycle_state=value.get("lifecycle_state"),
            schema_version=value.get("schema_version"),
            sha256=value.get("sha256", ""),
        )
        manifest.validate()
        if payload != manifest.canonical_bytes():
            raise IterationPreflightError("match-plan manifest is not canonical")
        return manifest


@dataclass(frozen=True)
class CapturedReplay:
    evidence_id: str
    artifact_path: str
    artifact_sha256: str
    game: str
    case: MatchPlanCase
    match_plan_sha256: str
    champion_descriptor_sha256: str
    human_replay_skill_sha256: str
    evaluated_policy_version: str
    evaluated_policy_source_sha256: str
    sha256: str = ""

    @classmethod
    def from_plan_case(
        cls,
        *,
        evidence_id: str,
        artifact_path: str,
        artifact_sha256: str,
        game: str,
        case: MatchPlanCase,
        plan: MatchPlanManifest,
    ) -> "CapturedReplay":
        plan.validate()
        evidence = cls(
            evidence_id,
            artifact_path,
            artifact_sha256,
            game,
            case,
            plan.sha256,
            plan.champion_descriptor_sha256,
            plan.human_replay_skill_sha256,
            plan.evaluated_policy_version,
            plan.evaluated_policy_source_sha256,
        )
        evidence.validate()
        return replace(evidence, sha256=_digest(evidence._unsigned_dict()))

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "game": self.game,
            "case": self.case.to_dict(),
            "match_plan_sha256": self.match_plan_sha256,
            "champion_descriptor_sha256": self.champion_descriptor_sha256,
            "human_replay_skill_sha256": self.human_replay_skill_sha256,
            "evaluated_policy_version": self.evaluated_policy_version,
            "evaluated_policy_source_sha256": self.evaluated_policy_source_sha256,
        }

    def validate(self) -> None:
        _logical_id(self.evidence_id, "replay evidence ID")
        _relative_asset(self.artifact_path, "replay artifact path")
        _sha256(self.artifact_sha256, "replay artifact SHA")
        if self.game != "24_miracle":
            raise IterationPreflightError("replay evidence game must be 24_miracle")
        if not isinstance(self.case, MatchPlanCase):
            raise IterationPreflightError("replay evidence case is invalid")
        self.case.validate()
        for label, value in (
            ("match plan", self.match_plan_sha256),
            ("champion descriptor", self.champion_descriptor_sha256),
            ("human replay Skill", self.human_replay_skill_sha256),
            ("policy source", self.evaluated_policy_source_sha256),
        ):
            _sha256(value, f"replay {label} SHA")
        _logical_id(self.evaluated_policy_version, "replay policy version")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("replay evidence SHA mismatch")

    def validate_against(self, plan: MatchPlanManifest) -> None:
        self.validate()
        case_by_id = {case.case_id: case for case in plan.cases}
        if case_by_id.get(self.case.case_id) != self.case:
            raise IterationPreflightError("replay evidence case/role/seed identity drift")
        expected = (
            plan.sha256,
            plan.champion_descriptor_sha256,
            plan.human_replay_skill_sha256,
            plan.evaluated_policy_version,
            plan.evaluated_policy_source_sha256,
        )
        actual = (
            self.match_plan_sha256,
            self.champion_descriptor_sha256,
            self.human_replay_skill_sha256,
            self.evaluated_policy_version,
            self.evaluated_policy_source_sha256,
        )
        if actual != expected:
            raise IterationPreflightError("replay evidence control identity drift")

    def to_dict(self) -> dict[str, Any]:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate()
        return {**current._unsigned_dict(), "sha256": current.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapturedReplay":
        if not isinstance(value, Mapping):
            raise IterationPreflightError("replay evidence must be an object")
        evidence = cls(
            value.get("evidence_id"),
            value.get("artifact_path"),
            value.get("artifact_sha256"),
            value.get("game"),
            MatchPlanCase.from_dict(value.get("case")),
            value.get("match_plan_sha256"),
            value.get("champion_descriptor_sha256"),
            value.get("human_replay_skill_sha256"),
            value.get("evaluated_policy_version"),
            value.get("evaluated_policy_source_sha256"),
            value.get("sha256", ""),
        )
        evidence.validate()
        return evidence


@dataclass(frozen=True)
class ReplayEvidenceManifest:
    match_plan_sha256: str
    evidence: tuple[CapturedReplay, ...]
    protocol_version: str
    benchmark_version: str
    iteration_protocol_version: str
    lifecycle_state: str = LifecycleState.REPLAY_CAPTURED_UNAPPROVED.value
    schema_version: str = REPLAY_EVIDENCE_SCHEMA_VERSION
    sha256: str = ""

    @classmethod
    def create(cls, **values: Any) -> "ReplayEvidenceManifest":
        manifest = cls(**values)
        manifest.validate()
        return replace(manifest, sha256=_digest(manifest._unsigned_dict()))

    @property
    def state(self) -> LifecycleState:
        return LifecycleState(self.lifecycle_state)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lifecycle_state": self.lifecycle_state,
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "iteration_protocol_version": self.iteration_protocol_version,
            "match_plan_sha256": self.match_plan_sha256,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def validate(self) -> None:
        if self.schema_version != REPLAY_EVIDENCE_SCHEMA_VERSION:
            raise IterationPreflightError("replay-evidence manifest schema mismatch")
        if self.lifecycle_state != LifecycleState.REPLAY_CAPTURED_UNAPPROVED.value:
            raise IterationPreflightError("replay-evidence lifecycle state mismatch")
        if self.protocol_version != PROTOCOL_VERSION or self.benchmark_version != BENCHMARK_VERSION:
            raise IterationPreflightError("replay-evidence protocol/benchmark version mismatch")
        if self.iteration_protocol_version != ITERATION_PROTOCOL_VERSION:
            raise IterationPreflightError("replay-evidence iteration protocol version mismatch")
        _sha256(self.match_plan_sha256, "replay-evidence match-plan SHA")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise IterationPreflightError("replay-evidence entries must be non-empty")
        for item in self.evidence:
            if not isinstance(item, CapturedReplay):
                raise IterationPreflightError("replay-evidence entry has invalid type")
            item.validate()
            if item.match_plan_sha256 != self.match_plan_sha256:
                raise IterationPreflightError("replay evidence references another match plan")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise IterationPreflightError("replay-evidence has duplicate evidence identity")
        if len({item.case.case_id for item in self.evidence}) != len(self.evidence):
            raise IterationPreflightError("replay-evidence has duplicate case identity")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("replay-evidence manifest SHA mismatch")

    def validate_against(self, plan: MatchPlanManifest) -> None:
        self.validate()
        if self.match_plan_sha256 != plan.sha256:
            raise IterationPreflightError("replay-evidence references an unapproved match plan")
        for item in self.evidence:
            item.validate_against(plan)
        planned = {case.case_id for case in plan.cases}
        captured = {item.case.case_id for item in self.evidence}
        if captured != planned or len(self.evidence) != len(plan.cases):
            raise IterationPreflightError("replay-evidence must cover every planned case exactly once")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate()
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ReplayEvidenceManifest":
        value = _strict_object_bytes(payload, "replay-evidence manifest")
        manifest = cls(
            match_plan_sha256=value.get("match_plan_sha256"),
            evidence=tuple(CapturedReplay.from_dict(item) for item in value.get("evidence", [])),
            protocol_version=value.get("protocol_version"),
            benchmark_version=value.get("benchmark_version"),
            iteration_protocol_version=value.get("iteration_protocol_version"),
            lifecycle_state=value.get("lifecycle_state"),
            schema_version=value.get("schema_version"),
            sha256=value.get("sha256", ""),
        )
        manifest.validate()
        if payload != manifest.canonical_bytes():
            raise IterationPreflightError("replay-evidence manifest is not canonical")
        return manifest


@dataclass(frozen=True)
class EvaluationPlanCase:
    case_id: str
    role: str
    evaluated_agent_camp: int
    map_type: int
    day_time: int
    repeat: int
    logic_seed: int
    evaluated_agent_seed: int
    opponent_seed: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _logical_id(self.case_id, "evaluation case ID")
        if self.role not in _EVALUATION_ROLES:
            raise ValueError("evaluation case role must be evaluation")
        for label, value in (
            ("camp", self.evaluated_agent_camp),
            ("map type", self.map_type),
            ("day time", self.day_time),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
                raise ValueError(f"evaluation {label} must be strict integer 0 or 1")
        if not isinstance(self.repeat, int) or isinstance(self.repeat, bool) or self.repeat not in (1, 2, 3):
            raise ValueError("evaluation repeat must be strict integer 1, 2, or 3")
        _strict_seed(self.logic_seed, "evaluation logic")
        _strict_seed(self.evaluated_agent_seed, "evaluation evaluated-agent")
        _strict_seed(self.opponent_seed, "evaluation opponent")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationPlanCase":
        if not isinstance(value, Mapping):
            raise IterationPreflightError("evaluation case must be an object")
        try:
            return cls(**{name: value.get(name) for name in cls.__dataclass_fields__})
        except ValueError as exc:
            raise IterationPreflightError(str(exc)) from exc


@dataclass(frozen=True)
class CandidateEvaluationPlan:
    baseline_strategy_version: str
    baseline_strategy_sha256: str
    baseline_source_sha256: str
    candidate_strategy_version: str
    candidate_strategy_sha256: str
    candidate_source_sha256: str
    experience_skill_version: str
    experience_skill_sha256: str
    champion_descriptor_sha256: str
    human_replay_skill_sha256: str
    training_match_plan_sha256: str
    training_replay_evidence_sha256: str
    research_manifest_sha256: str
    iteration_manifest_sha256: str
    purpose: str
    cases: tuple[EvaluationPlanCase, ...]
    protocol_version: str
    benchmark_version: str
    iteration_protocol_version: str
    schema_version: str = CANDIDATE_EVALUATION_PLAN_SCHEMA_VERSION
    sha256: str = ""

    @classmethod
    def create(cls, **values: Any) -> "CandidateEvaluationPlan":
        plan = cls(**values)
        plan.validate()
        return replace(plan, sha256=_digest(plan._unsigned_dict()))

    @classmethod
    def create_from_candidate(
        cls,
        candidate: "CandidateStoredContext",
        *,
        purpose: str,
        cases: tuple[EvaluationPlanCase, ...],
    ) -> "CandidateEvaluationPlan":
        current = _revalidate_candidate_context(candidate)
        store = ImmutableIterationStore(current.store_root)
        baseline = store.load_strategy("strategy-v0")
        strategy = store.load_strategy(current.version)
        skill = store.load_skill(current.experience_skill_version)
        return cls.create(
            baseline_strategy_version=baseline.version,
            baseline_strategy_sha256=baseline.sha256,
            baseline_source_sha256=baseline.source_sha256,
            candidate_strategy_version=strategy.version,
            candidate_strategy_sha256=strategy.sha256,
            candidate_source_sha256=strategy.source_sha256,
            experience_skill_version=skill.version,
            experience_skill_sha256=skill.sha256,
            champion_descriptor_sha256=current.champion_descriptor_sha256,
            human_replay_skill_sha256=current.human_replay_skill_sha256,
            training_match_plan_sha256=current.match_plan_sha256,
            training_replay_evidence_sha256=current.replay_evidence_manifest_sha256,
            research_manifest_sha256=current.research_manifest_sha256,
            iteration_manifest_sha256=current.iteration_manifest_sha256,
            purpose=purpose,
            cases=cases,
            protocol_version=PROTOCOL_VERSION,
            benchmark_version=BENCHMARK_VERSION,
            iteration_protocol_version=ITERATION_PROTOCOL_VERSION,
        )

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "iteration_protocol_version": self.iteration_protocol_version,
            "research_manifest_sha256": self.research_manifest_sha256,
            "iteration_manifest_sha256": self.iteration_manifest_sha256,
            "baseline_strategy_version": self.baseline_strategy_version,
            "baseline_strategy_sha256": self.baseline_strategy_sha256,
            "baseline_source_sha256": self.baseline_source_sha256,
            "candidate_strategy_version": self.candidate_strategy_version,
            "candidate_strategy_sha256": self.candidate_strategy_sha256,
            "candidate_source_sha256": self.candidate_source_sha256,
            "experience_skill_version": self.experience_skill_version,
            "experience_skill_sha256": self.experience_skill_sha256,
            "champion_descriptor_sha256": self.champion_descriptor_sha256,
            "human_replay_skill_sha256": self.human_replay_skill_sha256,
            "training_match_plan_sha256": self.training_match_plan_sha256,
            "training_replay_evidence_sha256": self.training_replay_evidence_sha256,
            "purpose": self.purpose,
            "cases": [case.to_dict() for case in self.cases],
        }

    def validate(self) -> None:
        if self.schema_version != CANDIDATE_EVALUATION_PLAN_SCHEMA_VERSION:
            raise IterationPreflightError("candidate evaluation plan schema mismatch")
        if (
            self.protocol_version != PROTOCOL_VERSION
            or self.benchmark_version != BENCHMARK_VERSION
            or self.iteration_protocol_version != ITERATION_PROTOCOL_VERSION
        ):
            raise IterationPreflightError("candidate evaluation plan protocol identity mismatch")
        if self.research_manifest_sha256 != research_manifest_sha256():
            raise IterationPreflightError("candidate evaluation plan research identity mismatch")
        if self.iteration_manifest_sha256 != hashlib.sha256(canonical_iteration_protocol_manifest_bytes()).hexdigest():
            raise IterationPreflightError("candidate evaluation plan iteration identity mismatch")
        for label, value in (
            ("baseline strategy", self.baseline_strategy_sha256),
            ("baseline source", self.baseline_source_sha256),
            ("candidate strategy", self.candidate_strategy_sha256),
            ("candidate source", self.candidate_source_sha256),
            ("experience Skill", self.experience_skill_sha256),
            ("champion", self.champion_descriptor_sha256),
            ("human replay Skill", self.human_replay_skill_sha256),
            ("training match plan", self.training_match_plan_sha256),
            ("training replay evidence", self.training_replay_evidence_sha256),
        ):
            _sha256(value, f"evaluation plan {label} SHA")
        for label, value in (
            ("baseline strategy version", self.baseline_strategy_version),
            ("candidate strategy version", self.candidate_strategy_version),
            ("experience Skill version", self.experience_skill_version),
        ):
            _logical_id(value, label)
        if self.baseline_strategy_version != "strategy-v0":
            raise IterationPreflightError("evaluation baseline must be strategy-v0")
        if self.candidate_strategy_version == "strategy-v0":
            raise IterationPreflightError("evaluation candidate must be non-v0")
        _required_text(self.purpose, "evaluation purpose")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise IterationPreflightError("evaluation plan cases must be non-empty")
        for case in self.cases:
            if not isinstance(case, EvaluationPlanCase):
                raise IterationPreflightError("evaluation plan case type is invalid")
            case.validate()
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise IterationPreflightError("evaluation plan has duplicate case identity")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("candidate evaluation plan SHA mismatch")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate()
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    @classmethod
    def from_bytes(cls, payload: bytes) -> "CandidateEvaluationPlan":
        value = _strict_object_bytes(payload, "candidate evaluation plan")
        plan = cls(
            baseline_strategy_version=value.get("baseline_strategy_version"),
            baseline_strategy_sha256=value.get("baseline_strategy_sha256"),
            baseline_source_sha256=value.get("baseline_source_sha256"),
            candidate_strategy_version=value.get("candidate_strategy_version"),
            candidate_strategy_sha256=value.get("candidate_strategy_sha256"),
            candidate_source_sha256=value.get("candidate_source_sha256"),
            experience_skill_version=value.get("experience_skill_version"),
            experience_skill_sha256=value.get("experience_skill_sha256"),
            champion_descriptor_sha256=value.get("champion_descriptor_sha256"),
            human_replay_skill_sha256=value.get("human_replay_skill_sha256"),
            training_match_plan_sha256=value.get("training_match_plan_sha256"),
            training_replay_evidence_sha256=value.get("training_replay_evidence_sha256"),
            research_manifest_sha256=value.get("research_manifest_sha256"),
            iteration_manifest_sha256=value.get("iteration_manifest_sha256"),
            purpose=value.get("purpose"),
            cases=tuple(EvaluationPlanCase.from_dict(item) for item in value.get("cases", [])),
            protocol_version=value.get("protocol_version"),
            benchmark_version=value.get("benchmark_version"),
            iteration_protocol_version=value.get("iteration_protocol_version"),
            schema_version=value.get("schema_version"),
            sha256=value.get("sha256", ""),
        )
        plan.validate()
        if payload != plan.canonical_bytes():
            raise IterationPreflightError("candidate evaluation plan is not canonical")
        return plan


def _validate_trajectory_kl(
    value: Mapping[str, Any], plan: CandidateEvaluationPlan
) -> bool:
    if not isinstance(value, Mapping):
        raise IterationPreflightError("trajectory KL evidence must be an object")
    episode = value.get("episode")
    _strict_positive_int(episode, "trajectory KL episode")
    if value.get("version_before") != plan.baseline_strategy_version:
        raise IterationPreflightError("trajectory KL old-policy identity mismatch")
    if value.get("version_after") != plan.candidate_strategy_version:
        raise IterationPreflightError("trajectory KL new-policy identity mismatch")
    epsilon = value.get("epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon != TRAJECTORY_KL_EPSILON:
        raise IterationPreflightError("trajectory KL epsilon must be 0.01")
    if value.get("direction") != KL_DIRECTION or value.get("rollout_source") != KL_ROLLOUT_SOURCE:
        raise IterationPreflightError("trajectory KL direction/rollout identity mismatch")
    if value.get("log_base") != "e" or value.get("estimand") != "epsilon_regularized_local_kl_sum_under_new_policy_occupancy":
        raise IterationPreflightError("trajectory KL estimand mismatch")
    status = value.get("status")
    if status not in {"complete", "incomplete", "failed"} or value.get("measurement_status") != status:
        raise IterationPreflightError("trajectory KL status is invalid or inconsistent")
    decisions = value.get("decisions")
    if not isinstance(decisions, list):
        raise IterationPreflightError("trajectory KL decisions must be a list")
    if value.get("decision_steps") != len(decisions):
        raise IterationPreflightError("trajectory KL decision count mismatch")
    trace = value.get("trace")
    if not isinstance(trace, list) or len(trace) != len(decisions):
        raise IterationPreflightError("trajectory KL trace length mismatch")
    errors = value.get("errors")
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        raise IterationPreflightError("trajectory KL errors must be strings")
    complete = status == "complete"
    if complete and (not decisions or errors):
        raise IterationPreflightError("complete trajectory KL requires decisions and no errors")
    total = 0.0
    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, Mapping) or decision.get("decision_step") != index:
            raise IterationPreflightError("trajectory KL decision steps must be strict and continuous")
        legal = decision.get("legal_action_ids")
        if not isinstance(legal, list) or not legal or any(not isinstance(item, str) or not item for item in legal):
            raise IterationPreflightError("trajectory KL ActionSupport is incomplete")
        if len(set(legal)) != len(legal) or decision.get("selected_action_id") not in legal:
            raise IterationPreflightError("trajectory KL ActionSupport identity is invalid")
        for label in ("context_ref", "action_schema_version", "support_id"):
            _required_text(decision.get(label), f"trajectory KL {label}")
        local = decision.get("local_policy_kl")
        if complete:
            local = _strict_finite_number(local, "local policy KL")
            if local < 0.0:
                raise IterationPreflightError("local policy KL cannot be negative")
            if trace[index - 1] != local:
                raise IterationPreflightError("trajectory KL trace disagrees with decisions")
            total += local
            for label in ("new_distribution", "old_distribution"):
                distribution = decision.get(label)
                if not isinstance(distribution, Mapping) or set(distribution) != set(legal):
                    raise IterationPreflightError("trajectory KL policy distribution support mismatch")
                values = [_strict_score(distribution[action], f"{label} probability") for action in legal]
                if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise IterationPreflightError("trajectory KL policy distribution is not normalized")
            for label in ("new_probabilities", "old_probabilities"):
                probabilities = decision.get(label)
                if not isinstance(probabilities, list) or len(probabilities) != len(legal):
                    raise IterationPreflightError("trajectory KL probability vector is incomplete")
                values = [_strict_score(item, f"{label} probability") for item in probabilities]
                if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
                    raise IterationPreflightError("trajectory KL probability vector is not normalized")
            if decision.get("errors") != []:
                raise IterationPreflightError("complete trajectory KL decision cannot contain errors")
    if complete:
        uploaded_total = _strict_finite_number(value.get("trajectory_kl_episode"), "trajectory KL episode")
        uploaded_mean = _strict_finite_number(value.get("mean_local_policy_kl"), "mean local policy KL")
        if not math.isclose(uploaded_total, total, rel_tol=0.0, abs_tol=1e-12):
            raise IterationPreflightError("trajectory KL episode total mismatch")
        if not math.isclose(uploaded_mean, total / len(decisions), rel_tol=0.0, abs_tol=1e-12):
            raise IterationPreflightError("trajectory KL episode mean mismatch")
    return complete


@dataclass(frozen=True)
class EvaluationCaseEvidence:
    evidence_id: str
    case: EvaluationPlanCase
    evaluation_plan_sha256: str
    baseline_strategy_version: str
    baseline_strategy_sha256: str
    candidate_strategy_version: str
    candidate_strategy_sha256: str
    experience_skill_version: str
    experience_skill_sha256: str
    baseline_artifact_path: str
    baseline_artifact_sha256: str
    candidate_artifact_path: str
    candidate_artifact_sha256: str
    baseline_outcome: str
    candidate_outcome: str
    baseline_score: float
    candidate_score: float
    terminal_status: str
    trajectory_kl: Mapping[str, Any]
    information_gain: float | None
    failure_reason: str | None
    sha256: str = ""

    @classmethod
    def from_plan_case(cls, *, plan: CandidateEvaluationPlan, case: EvaluationPlanCase, **values: Any) -> "EvaluationCaseEvidence":
        plan.validate()
        evidence = cls(
            case=case,
            evaluation_plan_sha256=plan.sha256,
            baseline_strategy_version=plan.baseline_strategy_version,
            baseline_strategy_sha256=plan.baseline_strategy_sha256,
            candidate_strategy_version=plan.candidate_strategy_version,
            candidate_strategy_sha256=plan.candidate_strategy_sha256,
            experience_skill_version=plan.experience_skill_version,
            experience_skill_sha256=plan.experience_skill_sha256,
            **values,
        )
        evidence.validate(plan)
        return replace(evidence, sha256=_digest(evidence._unsigned_dict()))

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "case": self.case.to_dict(),
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "baseline_strategy_version": self.baseline_strategy_version,
            "baseline_strategy_sha256": self.baseline_strategy_sha256,
            "candidate_strategy_version": self.candidate_strategy_version,
            "candidate_strategy_sha256": self.candidate_strategy_sha256,
            "experience_skill_version": self.experience_skill_version,
            "experience_skill_sha256": self.experience_skill_sha256,
            "baseline_artifact_path": self.baseline_artifact_path,
            "baseline_artifact_sha256": self.baseline_artifact_sha256,
            "candidate_artifact_path": self.candidate_artifact_path,
            "candidate_artifact_sha256": self.candidate_artifact_sha256,
            "baseline_outcome": self.baseline_outcome,
            "candidate_outcome": self.candidate_outcome,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "terminal_status": self.terminal_status,
            "trajectory_kl": dict(self.trajectory_kl),
            "information_gain": self.information_gain,
            "failure_reason": self.failure_reason,
        }

    def validate(self, plan: CandidateEvaluationPlan) -> None:
        _logical_id(self.evidence_id, "evaluation evidence ID")
        if not isinstance(self.case, EvaluationPlanCase):
            raise IterationPreflightError("evaluation evidence case is invalid")
        self.case.validate()
        planned = {case.case_id: case for case in plan.cases}
        if planned.get(self.case.case_id) != self.case:
            raise IterationPreflightError("evaluation evidence case/seed identity drift")
        expected = (
            plan.sha256,
            plan.baseline_strategy_version,
            plan.baseline_strategy_sha256,
            plan.candidate_strategy_version,
            plan.candidate_strategy_sha256,
            plan.experience_skill_version,
            plan.experience_skill_sha256,
        )
        actual = (
            self.evaluation_plan_sha256,
            self.baseline_strategy_version,
            self.baseline_strategy_sha256,
            self.candidate_strategy_version,
            self.candidate_strategy_sha256,
            self.experience_skill_version,
            self.experience_skill_sha256,
        )
        if actual != expected:
            raise IterationPreflightError("evaluation evidence policy/Skill identity drift")
        for label, path, digest in (
            ("baseline", self.baseline_artifact_path, self.baseline_artifact_sha256),
            ("candidate", self.candidate_artifact_path, self.candidate_artifact_sha256),
        ):
            _relative_asset(path, f"{label} evaluation artifact path")
            _sha256(digest, f"{label} evaluation artifact SHA")
        if self.baseline_outcome not in {"win", "loss", "draw"} or self.candidate_outcome not in {"win", "loss", "draw"}:
            raise IterationPreflightError("evaluation outcome is invalid")
        _strict_score(self.baseline_score, "baseline score")
        _strict_score(self.candidate_score, "candidate score")
        if self.terminal_status not in {"complete", "incomplete", "failed"}:
            raise IterationPreflightError("evaluation terminal status is invalid")
        kl_complete = _validate_trajectory_kl(self.trajectory_kl, plan)
        if self.information_gain is not None:
            _strict_finite_number(self.information_gain, "information gain")
        if self.failure_reason is not None:
            _required_text(self.failure_reason, "evaluation failure reason")
        if self.terminal_status == "complete" and (not kl_complete or self.information_gain is None):
            if self.failure_reason is None:
                raise IterationPreflightError("incomplete KL/IG requires a failure reason")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("evaluation evidence SHA mismatch")

    def to_dict(self) -> dict[str, Any]:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        return {**current._unsigned_dict(), "sha256": current.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], plan: CandidateEvaluationPlan) -> "EvaluationCaseEvidence":
        if not isinstance(value, Mapping):
            raise IterationPreflightError("evaluation evidence must be an object")
        evidence = cls(
            evidence_id=value.get("evidence_id"),
            case=EvaluationPlanCase.from_dict(value.get("case")),
            evaluation_plan_sha256=value.get("evaluation_plan_sha256"),
            baseline_strategy_version=value.get("baseline_strategy_version"),
            baseline_strategy_sha256=value.get("baseline_strategy_sha256"),
            candidate_strategy_version=value.get("candidate_strategy_version"),
            candidate_strategy_sha256=value.get("candidate_strategy_sha256"),
            experience_skill_version=value.get("experience_skill_version"),
            experience_skill_sha256=value.get("experience_skill_sha256"),
            baseline_artifact_path=value.get("baseline_artifact_path"),
            baseline_artifact_sha256=value.get("baseline_artifact_sha256"),
            candidate_artifact_path=value.get("candidate_artifact_path"),
            candidate_artifact_sha256=value.get("candidate_artifact_sha256"),
            baseline_outcome=value.get("baseline_outcome"),
            candidate_outcome=value.get("candidate_outcome"),
            baseline_score=value.get("baseline_score"),
            candidate_score=value.get("candidate_score"),
            terminal_status=value.get("terminal_status"),
            trajectory_kl=value.get("trajectory_kl"),
            information_gain=value.get("information_gain"),
            failure_reason=value.get("failure_reason"),
            sha256=value.get("sha256", ""),
        )
        evidence.validate(plan)
        return evidence


@dataclass(frozen=True)
class EvaluationEvidenceManifest:
    evaluation_plan_sha256: str
    evidence: tuple[EvaluationCaseEvidence, ...]
    protocol_version: str
    benchmark_version: str
    iteration_protocol_version: str
    schema_version: str = EVALUATION_EVIDENCE_SCHEMA_VERSION
    sha256: str = ""

    @classmethod
    def create(cls, **values: Any) -> "EvaluationEvidenceManifest":
        manifest = cls(**values)
        manifest.validate()
        return replace(manifest, sha256=_digest(manifest._unsigned_dict()))

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "iteration_protocol_version": self.iteration_protocol_version,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def validate(self, plan: CandidateEvaluationPlan | None = None) -> None:
        if self.schema_version != EVALUATION_EVIDENCE_SCHEMA_VERSION:
            raise IterationPreflightError("evaluation evidence manifest schema mismatch")
        if (
            self.protocol_version != PROTOCOL_VERSION
            or self.benchmark_version != BENCHMARK_VERSION
            or self.iteration_protocol_version != ITERATION_PROTOCOL_VERSION
        ):
            raise IterationPreflightError("evaluation evidence protocol identity mismatch")
        _sha256(self.evaluation_plan_sha256, "evaluation plan SHA")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise IterationPreflightError("evaluation evidence entries must be non-empty")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise IterationPreflightError("evaluation evidence has duplicate identity")
        if len({item.case.case_id for item in self.evidence}) != len(self.evidence):
            raise IterationPreflightError("evaluation evidence has duplicate case identity")
        if plan is not None:
            if self.evaluation_plan_sha256 != plan.sha256:
                raise IterationPreflightError("evaluation evidence references another plan")
            for item in self.evidence:
                item.validate(plan)
            if {item.case.case_id for item in self.evidence} != {case.case_id for case in plan.cases} or len(self.evidence) != len(plan.cases):
                raise IterationPreflightError("evaluation evidence must cover every planned case exactly once")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("evaluation evidence manifest SHA mismatch")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate()
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    @classmethod
    def from_bytes(cls, payload: bytes, plan: CandidateEvaluationPlan) -> "EvaluationEvidenceManifest":
        value = _strict_object_bytes(payload, "evaluation evidence manifest")
        manifest = cls(
            evaluation_plan_sha256=value.get("evaluation_plan_sha256"),
            evidence=tuple(EvaluationCaseEvidence.from_dict(item, plan) for item in value.get("evidence", [])),
            protocol_version=value.get("protocol_version"),
            benchmark_version=value.get("benchmark_version"),
            iteration_protocol_version=value.get("iteration_protocol_version"),
            schema_version=value.get("schema_version"),
            sha256=value.get("sha256", ""),
        )
        manifest.validate(plan)
        if payload != manifest.canonical_bytes():
            raise IterationPreflightError("evaluation evidence manifest is not canonical")
        return manifest


@dataclass(frozen=True)
class IterationAcceptanceManifest:
    baseline_strategy_version: str
    baseline_strategy_sha256: str
    candidate_strategy_version: str
    candidate_strategy_sha256: str
    experience_skill_version: str
    experience_skill_sha256: str
    training_match_plan_sha256: str
    training_replay_evidence_sha256: str
    evaluation_plan_sha256: str
    evaluation_evidence_sha256: str
    research_manifest_sha256: str
    iteration_manifest_sha256: str
    trajectory_kl_evidence_sha256: tuple[str, ...]
    raw_score: float
    evo_score: float
    gain: float
    information_gain: float | None
    evaluation_status: str
    blockers: tuple[str, ...]
    fake_only: bool
    protocol_version: str
    benchmark_version: str
    iteration_protocol_version: str
    schema_version: str = ITERATION_ACCEPTANCE_SCHEMA_VERSION
    sha256: str = ""

    @staticmethod
    def _derived(
        plan: CandidateEvaluationPlan,
        evidence: EvaluationEvidenceManifest,
        blockers: tuple[str, ...],
    ) -> dict[str, Any]:
        evidence.validate(plan)
        raw_score = sum(item.baseline_score for item in evidence.evidence) / len(evidence.evidence)
        evo_score = sum(item.candidate_score for item in evidence.evidence) / len(evidence.evidence)
        gain = evo_score - raw_score
        kl_complete = all(_validate_trajectory_kl(item.trajectory_kl, plan) for item in evidence.evidence)
        ig_values = [item.information_gain for item in evidence.evidence]
        information_gain = (
            sum(value for value in ig_values if value is not None) / len(ig_values)
            if all(value is not None for value in ig_values)
            else None
        )
        reasons = list(_text_tuple(blockers, "acceptance blocker"))
        if any(item.terminal_status != "complete" for item in evidence.evidence):
            reasons.append("evaluation case status is incomplete")
        if not kl_complete:
            reasons.append("trajectory KL evidence is incomplete")
        if information_gain is None:
            reasons.append("information gain evidence is missing")
        if not gain > 0.0:
            reasons.append("candidate has no valid score improvement")
        return {
            "raw_score": raw_score,
            "evo_score": evo_score,
            "gain": gain,
            "information_gain": information_gain,
            "evaluation_status": "complete" if not reasons else "incomplete",
            "blockers": tuple(reasons),
            "trajectory_kl_evidence_sha256": tuple(
                _digest(dict(item.trajectory_kl)) for item in evidence.evidence
            ),
        }

    @classmethod
    def create_from_evidence(
        cls,
        evaluation: "EvaluationReadyContext",
        evidence: EvaluationEvidenceManifest,
        *,
        blockers: tuple[str, ...],
        fake_only: bool,
    ) -> "IterationAcceptanceManifest":
        current = _revalidate_evaluation_context(evaluation)
        if not isinstance(fake_only, bool):
            raise IterationPreflightError("acceptance fake_only must be boolean")
        derived = cls._derived(current.plan, evidence, blockers)
        manifest = cls(
            baseline_strategy_version=current.plan.baseline_strategy_version,
            baseline_strategy_sha256=current.plan.baseline_strategy_sha256,
            candidate_strategy_version=current.plan.candidate_strategy_version,
            candidate_strategy_sha256=current.plan.candidate_strategy_sha256,
            experience_skill_version=current.plan.experience_skill_version,
            experience_skill_sha256=current.plan.experience_skill_sha256,
            training_match_plan_sha256=current.plan.training_match_plan_sha256,
            training_replay_evidence_sha256=current.plan.training_replay_evidence_sha256,
            evaluation_plan_sha256=current.plan.sha256,
            evaluation_evidence_sha256=evidence.sha256,
            research_manifest_sha256=current.plan.research_manifest_sha256,
            iteration_manifest_sha256=current.plan.iteration_manifest_sha256,
            fake_only=fake_only,
            protocol_version=PROTOCOL_VERSION,
            benchmark_version=BENCHMARK_VERSION,
            iteration_protocol_version=ITERATION_PROTOCOL_VERSION,
            **derived,
        )
        manifest.validate()
        return replace(manifest, sha256=_digest(manifest._unsigned_dict()))

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "iteration_protocol_version": self.iteration_protocol_version,
            "baseline_strategy_version": self.baseline_strategy_version,
            "baseline_strategy_sha256": self.baseline_strategy_sha256,
            "candidate_strategy_version": self.candidate_strategy_version,
            "candidate_strategy_sha256": self.candidate_strategy_sha256,
            "experience_skill_version": self.experience_skill_version,
            "experience_skill_sha256": self.experience_skill_sha256,
            "training_match_plan_sha256": self.training_match_plan_sha256,
            "training_replay_evidence_sha256": self.training_replay_evidence_sha256,
            "evaluation_plan_sha256": self.evaluation_plan_sha256,
            "evaluation_evidence_sha256": self.evaluation_evidence_sha256,
            "research_manifest_sha256": self.research_manifest_sha256,
            "iteration_manifest_sha256": self.iteration_manifest_sha256,
            "trajectory_kl_evidence_sha256": list(self.trajectory_kl_evidence_sha256),
            "raw_score": self.raw_score,
            "evo_score": self.evo_score,
            "gain": self.gain,
            "information_gain": self.information_gain,
            "evaluation_status": self.evaluation_status,
            "blockers": list(self.blockers),
            "fake_only": self.fake_only,
        }

    def validate(self) -> None:
        if self.schema_version != ITERATION_ACCEPTANCE_SCHEMA_VERSION:
            raise IterationPreflightError("iteration acceptance schema mismatch")
        if (
            self.protocol_version != PROTOCOL_VERSION
            or self.benchmark_version != BENCHMARK_VERSION
            or self.iteration_protocol_version != ITERATION_PROTOCOL_VERSION
        ):
            raise IterationPreflightError("iteration acceptance protocol identity mismatch")
        for label, value in (
            ("baseline strategy", self.baseline_strategy_sha256),
            ("candidate strategy", self.candidate_strategy_sha256),
            ("experience Skill", self.experience_skill_sha256),
            ("training match plan", self.training_match_plan_sha256),
            ("training replay evidence", self.training_replay_evidence_sha256),
            ("evaluation plan", self.evaluation_plan_sha256),
            ("evaluation evidence", self.evaluation_evidence_sha256),
            ("research manifest", self.research_manifest_sha256),
            ("iteration manifest", self.iteration_manifest_sha256),
        ):
            _sha256(value, f"acceptance {label} SHA")
        for value in self.trajectory_kl_evidence_sha256:
            _sha256(value, "acceptance trajectory KL evidence SHA")
        if not self.trajectory_kl_evidence_sha256:
            raise IterationPreflightError("acceptance requires trajectory KL evidence identities")
        for label, value in (
            ("baseline strategy version", self.baseline_strategy_version),
            ("candidate strategy version", self.candidate_strategy_version),
            ("experience Skill version", self.experience_skill_version),
        ):
            _logical_id(value, label)
        raw = _strict_score(self.raw_score, "acceptance raw score")
        evo = _strict_score(self.evo_score, "acceptance evo score")
        gain = _strict_finite_number(self.gain, "acceptance gain")
        if not math.isclose(gain, evo - raw, rel_tol=0.0, abs_tol=1e-12):
            raise IterationPreflightError("acceptance gain does not match scores")
        if self.information_gain is not None:
            _strict_finite_number(self.information_gain, "acceptance information gain")
        if self.evaluation_status not in {"complete", "incomplete"}:
            raise IterationPreflightError("acceptance evaluation status is invalid")
        _text_tuple(self.blockers, "acceptance blockers")
        if self.evaluation_status == "complete" and self.blockers:
            raise IterationPreflightError("complete acceptance cannot have blockers")
        if self.fake_only not in {True, False} or not isinstance(self.fake_only, bool):
            raise IterationPreflightError("acceptance fake_only must be boolean")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise IterationPreflightError("iteration acceptance SHA mismatch")

    def validate_against(
        self,
        plan: CandidateEvaluationPlan,
        evidence: EvaluationEvidenceManifest,
    ) -> None:
        self.validate()
        derived = self._derived(plan, evidence, self.blockers)
        # self.blockers already includes derived reasons, so recompute from raw
        # evidence using only external blockers that are not standard reasons.
        standard = {
            "evaluation case status is incomplete",
            "trajectory KL evidence is incomplete",
            "information gain evidence is missing",
            "candidate has no valid score improvement",
        }
        external = tuple(item for item in self.blockers if item not in standard)
        derived = self._derived(plan, evidence, external)
        expected = (
            plan.baseline_strategy_version,
            plan.baseline_strategy_sha256,
            plan.candidate_strategy_version,
            plan.candidate_strategy_sha256,
            plan.experience_skill_version,
            plan.experience_skill_sha256,
            plan.training_match_plan_sha256,
            plan.training_replay_evidence_sha256,
            plan.sha256,
            evidence.sha256,
            plan.research_manifest_sha256,
            plan.iteration_manifest_sha256,
            derived["trajectory_kl_evidence_sha256"],
            derived["raw_score"],
            derived["evo_score"],
            derived["gain"],
            derived["information_gain"],
            derived["evaluation_status"],
            derived["blockers"],
        )
        actual = (
            self.baseline_strategy_version,
            self.baseline_strategy_sha256,
            self.candidate_strategy_version,
            self.candidate_strategy_sha256,
            self.experience_skill_version,
            self.experience_skill_sha256,
            self.training_match_plan_sha256,
            self.training_replay_evidence_sha256,
            self.evaluation_plan_sha256,
            self.evaluation_evidence_sha256,
            self.research_manifest_sha256,
            self.iteration_manifest_sha256,
            self.trajectory_kl_evidence_sha256,
            self.raw_score,
            self.evo_score,
            self.gain,
            self.information_gain,
            self.evaluation_status,
            self.blockers,
        )
        if actual != expected:
            raise IterationPreflightError("iteration acceptance does not match first-hand evidence")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate()
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    def to_results_summary(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "benchmark_version": self.benchmark_version,
            "iteration_protocol_version": self.iteration_protocol_version,
            "research_manifest_sha256": self.research_manifest_sha256,
            "population": {
                "train": list(TRAIN_OPPONENTS),
                "validation": list(VALIDATION_OPPONENTS),
                "test": list(TEST_OPPONENTS),
            },
            "benchmark_results": [],
            "iteration_manifest_sha256": self.iteration_manifest_sha256,
            "iteration_acceptance_sha256": self.sha256,
            "candidate_evaluation_plan_sha256": self.evaluation_plan_sha256,
            "evaluation_evidence_sha256": self.evaluation_evidence_sha256,
            "training_match_plan_sha256": self.training_match_plan_sha256,
            "training_replay_evidence_sha256": self.training_replay_evidence_sha256,
            "trajectory_kl_evidence_sha256": list(
                self.trajectory_kl_evidence_sha256
            ),
            "baseline_policy_version": self.baseline_strategy_version,
            "baseline_policy_sha256": self.baseline_strategy_sha256,
            "candidate_policy_version": self.candidate_strategy_version,
            "candidate_policy_sha256": self.candidate_strategy_sha256,
            "experience_skill_version": self.experience_skill_version,
            "experience_skill_sha256": self.experience_skill_sha256,
            "raw_score": self.raw_score,
            "evo_score": self.evo_score,
            "benchmark_score": self.evo_score,
            "gain": self.gain,
            "information_gain": self.information_gain,
            "evaluation_status": self.evaluation_status,
            "incomplete_reasons": list(self.blockers),
            "fake_only": self.fake_only,
        }

    @classmethod
    def from_bytes(cls, payload: bytes) -> "IterationAcceptanceManifest":
        value = _strict_object_bytes(payload, "iteration acceptance manifest")
        manifest = cls(
            baseline_strategy_version=value.get("baseline_strategy_version"),
            baseline_strategy_sha256=value.get("baseline_strategy_sha256"),
            candidate_strategy_version=value.get("candidate_strategy_version"),
            candidate_strategy_sha256=value.get("candidate_strategy_sha256"),
            experience_skill_version=value.get("experience_skill_version"),
            experience_skill_sha256=value.get("experience_skill_sha256"),
            training_match_plan_sha256=value.get("training_match_plan_sha256"),
            training_replay_evidence_sha256=value.get("training_replay_evidence_sha256"),
            evaluation_plan_sha256=value.get("evaluation_plan_sha256"),
            evaluation_evidence_sha256=value.get("evaluation_evidence_sha256"),
            research_manifest_sha256=value.get("research_manifest_sha256"),
            iteration_manifest_sha256=value.get("iteration_manifest_sha256"),
            trajectory_kl_evidence_sha256=_text_tuple(value.get("trajectory_kl_evidence_sha256", []), "trajectory KL evidence SHA"),
            raw_score=value.get("raw_score"),
            evo_score=value.get("evo_score"),
            gain=value.get("gain"),
            information_gain=value.get("information_gain"),
            evaluation_status=value.get("evaluation_status"),
            blockers=_text_tuple(value.get("blockers", []), "acceptance blockers"),
            fake_only=value.get("fake_only"),
            protocol_version=value.get("protocol_version"),
            benchmark_version=value.get("benchmark_version"),
            iteration_protocol_version=value.get("iteration_protocol_version"),
            schema_version=value.get("schema_version"),
            sha256=value.get("sha256", ""),
        )
        manifest.validate()
        if payload != manifest.canonical_bytes():
            raise IterationPreflightError("iteration acceptance manifest is not canonical")
        return manifest


@dataclass(frozen=True)
class ChangeOperation:
    operation: str
    target: str

    def validate(self) -> None:
        if self.operation not in _CHANGE_OPERATIONS:
            raise ValueError(f"unsupported change operation: {self.operation!r}")
        _required_text(self.target, "change target")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return {"operation": self.operation, "target": self.target}


@dataclass(frozen=True)
class ChangePlan:
    operations: tuple[ChangeOperation, ...]
    complexity_growth_reason: str | None = None
    training_evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        if not isinstance(self.operations, tuple) or not self.operations:
            raise ValueError("change plan must contain at least one operation")
        for operation in self.operations:
            if not isinstance(operation, ChangeOperation):
                raise ValueError("change plan operations must be structured")
            operation.validate()
        if self.complexity_growth_reason is not None:
            _required_text(self.complexity_growth_reason, "complexity growth reason")
        for evidence in _text_tuple(self.training_evidence, "training evidence"):
            _sha256(evidence, "training evidence digest")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"operations": [item.to_dict() for item in self.operations],
                "complexity_growth_reason": self.complexity_growth_reason,
                "training_evidence": list(self.training_evidence)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangePlan":
        if not isinstance(value, Mapping) or not isinstance(value.get("operations"), list):
            raise ValueError("change plan operations must be a list")
        plan = cls(tuple(ChangeOperation(item.get("operation"), item.get("target")) for item in value["operations"] if isinstance(item, Mapping)),
                   value.get("complexity_growth_reason"), _text_tuple(value.get("training_evidence", []), "training evidence"))
        plan.validate()
        return plan


@dataclass(frozen=True)
class ComplexityMetrics:
    effective_nodes: int
    source_lines: int
    duplicate_or_shadowed_rules: int
    max_decision_depth: int

    def validate(self) -> None:
        for label, value in (("effective nodes", self.effective_nodes), ("source lines", self.source_lines),
                             ("duplicate or shadowed rules", self.duplicate_or_shadowed_rules), ("maximum decision depth", self.max_decision_depth)):
            _strict_nonnegative_int(value, label)

    def to_dict(self) -> dict[str, int]:
        self.validate()
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComplexityMetrics":
        metrics = cls(*(value.get(name) for name in cls.__dataclass_fields__))
        metrics.validate()
        return metrics


def _decision_depth(node: ast.AST, depth: int = 0) -> int:
    is_decision = isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match, ast.BoolOp, ast.IfExp))
    next_depth = depth + int(is_decision)
    return max([next_depth, *(_decision_depth(child, next_depth) for child in ast.iter_child_nodes(node))])


def compute_complexity_metrics(source_files: Mapping[str, str]) -> ComplexityMetrics:
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("complexity requires a complete source snapshot")
    nodes = lines = duplicates = depth = 0
    seen_statements: set[str] = set()
    for path, content in sorted(source_files.items()):
        _relative_asset(path, "strategy source path")
        if not isinstance(content, str):
            raise ValueError("strategy source must be UTF-8 text")
        lines += sum(1 for line in content.splitlines() if line.strip())
        if path.endswith(".py"):
            try:
                tree = ast.parse(content, filename=path)
            except SyntaxError as exc:
                raise ValueError("strategy Python source must parse") from exc
            statement_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
            nodes += len(statement_nodes)
            depth = max(depth, _decision_depth(tree))
            for node in statement_nodes:
                signature = ast.dump(node, annotate_fields=True, include_attributes=False)
                if signature in seen_statements:
                    duplicates += 1
                seen_statements.add(signature)
    result = ComplexityMetrics(nodes, lines, duplicates, depth)
    result.validate()
    return result


def validate_complexity_update(before: ComplexityMetrics, after: ComplexityMetrics, plan: ChangePlan) -> None:
    before.validate(); after.validate(); plan.validate()
    grew = any(a > b for a, b in zip(after.to_dict().values(), before.to_dict().values()))
    if grew and (not plan.complexity_growth_reason or not plan.training_evidence):
        raise ValueError("complexity growth requires a reason and training-only evidence")


def _validate_python_source(path: str, content: str, dependencies: tuple[str, ...]) -> None:
    tree = ast.parse(content, filename=path)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"__builtins__", "__loader__", "__spec__"}:
            raise ValueError("strategy source contains forbidden dynamic runtime access")
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                raise ValueError("strategy source contains forbidden dynamic/file call")
            if isinstance(node.func, ast.Attribute) and node.func.attr in _FORBIDDEN_ATTRIBUTES:
                raise ValueError("strategy source contains forbidden system/file call")
    if not imports.issubset(_ALLOWED_IMPORT_ROOTS):
        raise ValueError("strategy source imports a module outside the dependency allowlist")
    registered = {item[7:] for item in dependencies if item.startswith("module:")}
    if not imports.issubset(registered):
        raise ValueError("strategy import dependency is not individually registered")


@dataclass(frozen=True)
class StrategyVersion:
    version: str
    parent_version: str | None
    entrypoint: str
    dependencies: tuple[str, ...]
    source_files: Mapping[str, str]
    change_plan: ChangePlan
    interpretability_category: str
    probability_query: str
    complexity: ComplexityMetrics
    bootstrap_version: str = BOOTSTRAP_TEMPLATE_VERSION
    bootstrap_sha256: str = APPROVED_BOOTSTRAP_SHA256
    champion_descriptor_sha256: str = ""
    match_plan_sha256: str = ""
    replay_evidence_manifest_sha256: str | None = None
    human_replay_skill_sha256: str = ""
    iteration_manifest_sha256: str = ""
    research_manifest_sha256: str = ""
    network_dependencies: tuple[str, ...] = ()
    binary_models: tuple[str, ...] = ()
    external_state: tuple[str, ...] = ()
    legal_action_unit: str = LEGAL_ACTION_UNIT
    schema_version: str = STRATEGY_SCHEMA_VERSION
    source_sha256: str = ""
    sha256: str = ""

    @classmethod
    def from_directory(cls, root: Path, *, version: str, parent_version: str | None, entrypoint: str,
                       dependencies: tuple[str, ...], change_plan: ChangePlan, interpretability_category: str,
                       probability_query: str, complexity: ComplexityMetrics | None = None,
                       bootstrap_version: str = BOOTSTRAP_TEMPLATE_VERSION,
                       bootstrap_sha256: str = APPROVED_BOOTSTRAP_SHA256,
                       control_context: "LearningReadyContext | None" = None) -> "StrategyVersion":
        if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
            raise ValueError("strategy source directory is missing or symbolic")
        source_files: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("strategy source snapshots cannot contain symlinks")
            if path.is_file():
                relative = _relative_asset(path.relative_to(root).as_posix(), "strategy source path")
                if Path(relative).suffix not in _ALLOWED_SOURCE_SUFFIXES:
                    raise ValueError("strategy source contains an opaque or unsupported file type")
                try:
                    source_files[relative] = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError) as exc:
                    raise ValueError("strategy source must be traceable UTF-8 text") from exc
        computed = compute_complexity_metrics(source_files)
        if complexity is not None and complexity != computed:
            raise ValueError("candidate-reported complexity does not match computed complexity")
        control: dict[str, Any] = {}
        if control_context is not None:
            current = _revalidate_learning_context(control_context)
            control = {
                "champion_descriptor_sha256": current.champion.sha256,
                "match_plan_sha256": current.match_plan.sha256,
                "replay_evidence_manifest_sha256": current.replay_manifest.sha256,
                "human_replay_skill_sha256": current.human_replay_skill.sha256,
                "iteration_manifest_sha256": current.match_plan.iteration_manifest_sha256,
                "research_manifest_sha256": current.match_plan.research_manifest_sha256,
            }
        strategy = cls(
            version=version,
            parent_version=parent_version,
            entrypoint=entrypoint,
            dependencies=tuple(dependencies),
            source_files=source_files,
            change_plan=change_plan,
            interpretability_category=interpretability_category,
            probability_query=probability_query,
            complexity=computed,
            bootstrap_version=bootstrap_version,
            bootstrap_sha256=bootstrap_sha256,
            **control,
        )
        strategy.validate()
        return strategy._with_computed_hashes()

    def _unsigned_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "version": self.version, "parent_version": self.parent_version,
                "entrypoint": self.entrypoint, "dependencies": list(self.dependencies), "source_files": dict(self.source_files),
                "source_sha256": self.source_sha256, "change_plan": self.change_plan.to_dict(),
                "interpretability_category": self.interpretability_category, "probability_query": self.probability_query,
                "complexity": self.complexity.to_dict(), "bootstrap_version": self.bootstrap_version,
                "bootstrap_sha256": self.bootstrap_sha256,
                "champion_descriptor_sha256": self.champion_descriptor_sha256,
                "match_plan_sha256": self.match_plan_sha256,
                "replay_evidence_manifest_sha256": self.replay_evidence_manifest_sha256,
                "human_replay_skill_sha256": self.human_replay_skill_sha256,
                "iteration_manifest_sha256": self.iteration_manifest_sha256,
                "research_manifest_sha256": self.research_manifest_sha256,
                "network_dependencies": list(self.network_dependencies),
                "binary_models": list(self.binary_models), "external_state": list(self.external_state),
                "legal_action_unit": self.legal_action_unit}

    def _with_computed_hashes(self) -> "StrategyVersion":
        current = replace(self, source_sha256=_digest(dict(self.source_files)), sha256="")
        return replace(current, sha256=_digest(current._unsigned_dict()))

    def validate(self) -> None:
        if self.schema_version != STRATEGY_SCHEMA_VERSION:
            raise ValueError("strategy schema version mismatch")
        _logical_id(self.version, "strategy version")
        if self.parent_version is not None:
            _logical_id(self.parent_version, "strategy parent version")
        if not isinstance(self.entrypoint, str) or ":" not in self.entrypoint:
            raise ValueError("strategy entrypoint must identify a source function")
        entry_file = _relative_asset(self.entrypoint.split(":", 1)[0], "strategy entrypoint path")
        if entry_file not in self.source_files:
            raise ValueError("strategy entrypoint source is absent")
        dependencies = _text_tuple(self.dependencies, "strategy dependencies")
        for dependency in dependencies:
            if dependency.startswith("source:"):
                if _relative_asset(dependency[7:], "strategy source dependency") not in self.source_files:
                    raise ValueError("strategy source dependency is not in the snapshot")
            elif dependency.startswith("module:") and dependency[7:] in _ALLOWED_IMPORT_ROOTS:
                pass
            else:
                raise ValueError("strategy dependency is not individually allowed")
        for path, content in self.source_files.items():
            _relative_asset(path, "strategy source path")
            if Path(path).suffix not in _ALLOWED_SOURCE_SUFFIXES:
                raise ValueError("strategy source contains an opaque or unsupported file type")
            if not isinstance(content, str):
                raise ValueError("strategy source files must be UTF-8 text")
            if path.endswith(".py"):
                _validate_python_source(path, content, dependencies)
        self.change_plan.validate()
        if self.interpretability_category not in _INTERPRETABLE_CATEGORIES:
            raise ValueError("strategy interpretability category is not allowed")
        if self.probability_query != "read-only-complete-action-support":
            raise ValueError("strategy probability query capability is incomplete")
        computed = compute_complexity_metrics(self.source_files)
        if self.complexity != computed:
            raise ValueError("strategy complexity does not match complete source snapshot")
        if self.bootstrap_version != BOOTSTRAP_TEMPLATE_VERSION or self.bootstrap_sha256 != APPROVED_BOOTSTRAP_SHA256:
            raise ValueError("strategy bootstrap identity mismatch")
        for label, value in (
            ("champion descriptor", self.champion_descriptor_sha256),
            ("match plan", self.match_plan_sha256),
            ("human replay Skill", self.human_replay_skill_sha256),
            ("iteration manifest", self.iteration_manifest_sha256),
            ("research manifest", self.research_manifest_sha256),
        ):
            if value:
                _sha256(value, f"strategy {label} SHA")
        if self.replay_evidence_manifest_sha256 is not None:
            _sha256(self.replay_evidence_manifest_sha256, "strategy replay-evidence SHA")
        if self.network_dependencies or self.binary_models or self.external_state:
            raise ValueError("network, opaque binary, and external state dependencies are forbidden")
        if self.legal_action_unit != LEGAL_ACTION_UNIT:
            raise ValueError("strategy legal action unit mismatch")
        if self.source_sha256 and self.source_sha256 != _digest(dict(self.source_files)):
            raise ValueError("strategy source SHA mismatch")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise ValueError("strategy SHA mismatch")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 and self.source_sha256 else self._with_computed_hashes()
        current.validate()
        return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    @classmethod
    def from_bytes(cls, payload: bytes) -> "StrategyVersion":
        value = _strict_object_bytes(payload, "strategy version")
        strategy = cls(
            version=value.get("version"),
            parent_version=value.get("parent_version"),
            entrypoint=value.get("entrypoint"),
            dependencies=_text_tuple(value.get("dependencies", []), "strategy dependencies"),
            source_files=value.get("source_files"),
            change_plan=ChangePlan.from_dict(value.get("change_plan", {})),
            interpretability_category=value.get("interpretability_category"),
            probability_query=value.get("probability_query"),
            complexity=ComplexityMetrics.from_dict(value.get("complexity", {})),
            bootstrap_version=value.get("bootstrap_version"),
            bootstrap_sha256=value.get("bootstrap_sha256"),
            champion_descriptor_sha256=value.get("champion_descriptor_sha256", ""),
            match_plan_sha256=value.get("match_plan_sha256", ""),
            replay_evidence_manifest_sha256=value.get("replay_evidence_manifest_sha256"),
            human_replay_skill_sha256=value.get("human_replay_skill_sha256", ""),
            iteration_manifest_sha256=value.get("iteration_manifest_sha256", ""),
            research_manifest_sha256=value.get("research_manifest_sha256", ""),
            network_dependencies=_text_tuple(value.get("network_dependencies", []), "network dependencies"),
            binary_models=_text_tuple(value.get("binary_models", []), "binary models"),
            external_state=_text_tuple(value.get("external_state", []), "external state"),
            legal_action_unit=value.get("legal_action_unit"),
            schema_version=value.get("schema_version"),
            source_sha256=value.get("source_sha256", ""),
            sha256=value.get("sha256", ""),
        )
        strategy.validate()
        if payload != strategy.canonical_bytes():
            raise ValueError("strategy SHA/canonical bytes mismatch")
        return strategy


@dataclass(frozen=True)
class MiracleExperienceSkill:
    version: str
    parent_version: str | None
    frozen_facts: tuple[str, ...]
    training_observations: tuple[str, ...]
    supported_heuristics: tuple[str, ...]
    failed_heuristics: tuple[str, ...]
    unverified_hypotheses: tuple[str, ...]
    training_evidence_sha256: tuple[str, ...]
    strategy_versions: tuple[str, ...]
    case_identities: tuple[str, ...]
    human_replay_skill_sha256: str
    created_by: str
    modified_by: str
    champion_descriptor_sha256: str = ""
    match_plan_sha256: str = ""
    replay_evidence_manifest_sha256: str = ""
    iteration_manifest_sha256: str = ""
    research_manifest_sha256: str = ""
    bootstrap_sha256: str = APPROVED_BOOTSTRAP_SHA256
    schema_version: str = EXPERIENCE_SKILL_SCHEMA_VERSION
    sha256: str = ""

    @classmethod
    def create(cls, **values: Any) -> "MiracleExperienceSkill":
        skill = cls(**values); skill.validate()
        return replace(skill, sha256=_digest(skill._unsigned_dict()))

    @classmethod
    def create_from_learning(
        cls, context: "LearningReadyContext", **values: Any
    ) -> "MiracleExperienceSkill":
        current = _revalidate_learning_context(context)
        return cls.create(
            **values,
            human_replay_skill_sha256=current.human_replay_skill.sha256,
            champion_descriptor_sha256=current.champion.sha256,
            match_plan_sha256=current.match_plan.sha256,
            replay_evidence_manifest_sha256=current.replay_manifest.sha256,
            iteration_manifest_sha256=current.match_plan.iteration_manifest_sha256,
            research_manifest_sha256=current.match_plan.research_manifest_sha256,
            bootstrap_sha256=current.bootstrap.sha256,
        )

    def _unsigned_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "version": self.version, "parent_version": self.parent_version,
                "frozen_facts": list(self.frozen_facts), "training_observations": list(self.training_observations),
                "supported_heuristics": list(self.supported_heuristics), "failed_heuristics": list(self.failed_heuristics),
                "unverified_hypotheses": list(self.unverified_hypotheses),
                "training_evidence_sha256": list(self.training_evidence_sha256), "strategy_versions": list(self.strategy_versions),
                "case_identities": list(self.case_identities), "human_replay_skill_sha256": self.human_replay_skill_sha256,
                "created_by": self.created_by, "modified_by": self.modified_by,
                "champion_descriptor_sha256": self.champion_descriptor_sha256,
                "match_plan_sha256": self.match_plan_sha256,
                "replay_evidence_manifest_sha256": self.replay_evidence_manifest_sha256,
                "iteration_manifest_sha256": self.iteration_manifest_sha256,
                "research_manifest_sha256": self.research_manifest_sha256,
                "bootstrap_sha256": self.bootstrap_sha256}

    def validate(self) -> None:
        if self.schema_version != EXPERIENCE_SKILL_SCHEMA_VERSION:
            raise ValueError("Miracle experience Skill schema mismatch")
        _logical_id(self.version, "experience Skill version")
        if self.parent_version is not None: _logical_id(self.parent_version, "experience Skill parent version")
        for label, values in (("frozen facts", self.frozen_facts), ("training observations", self.training_observations),
                              ("supported heuristics", self.supported_heuristics), ("failed heuristics", self.failed_heuristics),
                              ("unverified hypotheses", self.unverified_hypotheses), ("strategy versions", self.strategy_versions),
                              ("case identities", self.case_identities)):
            _text_tuple(values, label)
        for evidence in _text_tuple(
            self.training_evidence_sha256,
            "training evidence SHA",
            nonempty=self.version != "experience-v0",
        ):
            _sha256(evidence, "training evidence SHA")
        _sha256(self.human_replay_skill_sha256, "human replay Skill SHA")
        _logical_id(self.created_by, "experience Skill creator"); _logical_id(self.modified_by, "experience Skill modifier")
        for label, value in (
            ("champion descriptor", self.champion_descriptor_sha256),
            ("match plan", self.match_plan_sha256),
            ("replay evidence", self.replay_evidence_manifest_sha256),
            ("iteration manifest", self.iteration_manifest_sha256),
            ("research manifest", self.research_manifest_sha256),
        ):
            if value:
                _sha256(value, f"experience Skill {label} SHA")
        if self.bootstrap_sha256 != APPROVED_BOOTSTRAP_SHA256:
            raise ValueError("experience Skill bootstrap identity mismatch")
        if self.version == "experience-v0":
            if self.parent_version is not None:
                raise ValueError("experience-v0 parent must be null")
            if any(
                (
                    self.frozen_facts,
                    self.training_observations,
                    self.supported_heuristics,
                    self.failed_heuristics,
                    self.unverified_hypotheses,
                    self.training_evidence_sha256,
                    self.case_identities,
                )
            ) or self.strategy_versions != ("strategy-v0",):
                raise ValueError("experience-v0 must be the fixed empty baseline")
        elif self.parent_version is None:
            raise ValueError("non-v0 experience Skill requires parent")
        if self.sha256 and self.sha256 != _digest(self._unsigned_dict()):
            raise ValueError("Miracle experience Skill SHA mismatch")

    def canonical_bytes(self) -> bytes:
        current = self if self.sha256 else replace(self, sha256=_digest(self._unsigned_dict()))
        current.validate(); return _canonical_bytes({**current._unsigned_dict(), "sha256": current.sha256})

    @classmethod
    def from_bytes(cls, payload: bytes) -> "MiracleExperienceSkill":
        value = _strict_object_bytes(payload, "Miracle experience Skill")
        skill = cls(value.get("version"), value.get("parent_version"),
                    _text_tuple(value.get("frozen_facts", []), "frozen facts"),
                    _text_tuple(value.get("training_observations", []), "training observations"),
                    _text_tuple(value.get("supported_heuristics", []), "supported heuristics"),
                    _text_tuple(value.get("failed_heuristics", []), "failed heuristics"),
                    _text_tuple(value.get("unverified_hypotheses", []), "unverified hypotheses"),
                    _text_tuple(value.get("training_evidence_sha256", []), "training evidence SHA"),
                    _text_tuple(value.get("strategy_versions", []), "strategy versions"),
                    _text_tuple(value.get("case_identities", []), "case identities"), value.get("human_replay_skill_sha256"),
                    value.get("created_by"), value.get("modified_by"),
                    value.get("champion_descriptor_sha256", ""), value.get("match_plan_sha256", ""),
                    value.get("replay_evidence_manifest_sha256", ""), value.get("iteration_manifest_sha256", ""),
                    value.get("research_manifest_sha256", ""), value.get("bootstrap_sha256"),
                    value.get("schema_version"), value.get("sha256", ""))
        skill.validate()
        if payload != skill.canonical_bytes(): raise ValueError("Miracle experience Skill is not canonical")
        return skill


@dataclass(frozen=True)
class MatchConfig:
    game: str
    bootstrap: BootstrapTemplate
    champion_manifest_path: Path
    human_replay_skill_path: Path
    match_plan_path: Path


@dataclass(frozen=True)
class LearningConfig:
    match: MatchConfig
    replay_evidence_manifest_path: Path


@dataclass(frozen=True)
class EvaluationConfig:
    learning_config: LearningConfig
    store_root: Path
    candidate_strategy_version: str
    experience_skill_version: str
    evaluation_plan_path: Path


@dataclass(frozen=True)
class AcceptanceConfig:
    evaluation: EvaluationConfig
    evaluation_evidence_path: Path
    acceptance_manifest_path: Path


_CONTEXT_ISSUER = object()


@dataclass(frozen=True, init=False)
class MatchReadyContext:
    config: MatchConfig
    bootstrap: BootstrapTemplate
    champion: HumanChampion
    human_replay_skill: HumanReplaySkill
    match_plan: MatchPlanManifest
    fingerprint: str
    state: LifecycleState
    issuer: object = field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class BaselineStoredContext:
    config: MatchConfig
    store_root: Path
    version: str
    strategy_sha256: str
    source_sha256: str
    cases: tuple[MatchPlanCase, ...]
    match: MatchReadyContext
    fingerprint: str
    state: LifecycleState
    issuer: object = field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class ReplayApprovedContext:
    config: LearningConfig
    match: MatchReadyContext
    replay_manifest: ReplayEvidenceManifest
    fingerprint: str
    state: LifecycleState
    issuer: object = field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class LearningReadyContext:
    config: LearningConfig
    bootstrap: BootstrapTemplate
    champion: HumanChampion
    human_replay_skill: HumanReplaySkill
    match_plan: MatchPlanManifest
    replay_manifest: ReplayEvidenceManifest
    training_evidence: tuple[CapturedReplay, ...]
    training_prompt_payload: Mapping[str, Any]
    fingerprint: str
    state: LifecycleState
    issuer: object = field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class CandidateStoredContext:
    config: LearningConfig
    store_root: Path
    version: str
    strategy_sha256: str
    source_sha256: str
    experience_skill_version: str
    experience_skill_sha256: str
    champion_descriptor_sha256: str
    human_replay_skill_sha256: str
    match_plan_sha256: str
    replay_evidence_manifest_sha256: str
    research_manifest_sha256: str
    iteration_manifest_sha256: str
    fingerprint: str
    state: LifecycleState
    issuer: object = field(repr=False, compare=False)


@dataclass(frozen=True, init=False)
class EvaluationReadyContext:
    config: EvaluationConfig
    candidate: CandidateStoredContext
    plan: CandidateEvaluationPlan
    fingerprint: str
    state: LifecycleState
    issuer: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class IterationOutcome:
    candidate_version: str
    candidate_sha256: str
    state: LifecycleState
    acceptance_sha256: str
    fake_only: bool


def _issue_context(context_type: type, **values: Any):
    context = object.__new__(context_type)
    for name, value in values.items():
        object.__setattr__(context, name, value)
    object.__setattr__(context, "issuer", _CONTEXT_ISSUER)
    return context


def _load_approved_human_skill(path: Path) -> HumanReplaySkill:
    skill = HumanReplaySkill.from_bytes(_read_bytes(path, "human replay Skill"))
    if skill.sha256 not in APPROVED_HUMAN_REPLAY_SKILL_SHA256:
        raise HumanAuthoredContentRequired(
            f"{HUMAN_AUTHORED_CONTENT_REQUIRED}: human replay Skill is not approved"
        )
    return skill


def _load_approved_match_plan(path: Path) -> MatchPlanManifest:
    plan = MatchPlanManifest.from_bytes(_read_bytes(path, "match-plan manifest"))
    if plan.sha256 not in APPROVED_MATCH_PLAN_MANIFEST_SHA256:
        raise IterationPreflightError("match-plan manifest is not independently approved")
    return plan


def _match_fingerprint(
    config: MatchConfig,
    champion: HumanChampion,
    human_skill: HumanReplaySkill,
    plan: MatchPlanManifest,
) -> str:
    return _digest(
        {
            "bootstrap_path": str(config.bootstrap.path.resolve()),
            "bootstrap_sha256": config.bootstrap.sha256,
            "champion_path": str(config.champion_manifest_path.resolve()),
            "champion_sha256": champion.sha256,
            "human_skill_path": str(config.human_replay_skill_path.resolve()),
            "human_skill_sha256": human_skill.sha256,
            "match_plan_path": str(config.match_plan_path.resolve()),
            "match_plan_sha256": plan.sha256,
        }
    )


def preflight_match(config: MatchConfig) -> MatchReadyContext:
    try:
        if not isinstance(config, MatchConfig):
            raise IterationPreflightError("validated match config is required")
        if config.game != "24_miracle":
            raise ValueError("Miracle match schema only applies to 24_miracle")
        validate_bootstrap_template(config.bootstrap)
        champion = load_human_champion(config.champion_manifest_path)
        human_skill = _load_approved_human_skill(config.human_replay_skill_path)
        plan = _load_approved_match_plan(config.match_plan_path)
        if plan.champion_descriptor_sha256 != champion.sha256:
            raise IterationPreflightError("match-plan champion identity mismatch")
        if plan.human_replay_skill_sha256 != human_skill.sha256:
            raise IterationPreflightError("match-plan human replay Skill identity mismatch")
        if plan.bootstrap_sha256 != config.bootstrap.sha256:
            raise IterationPreflightError("match-plan bootstrap identity mismatch")
        if plan.evaluated_policy_version != "strategy-v0":
            raise IterationPreflightError("from-scratch match plan must evaluate strategy-v0")
        if plan.evaluated_policy_source_sha256 != bootstrap_strategy_source_sha256(config.bootstrap):
            raise IterationPreflightError("match-plan evaluated policy source mismatch")
        fingerprint = _match_fingerprint(config, champion, human_skill, plan)
        return _issue_context(
            MatchReadyContext,
            config=config,
            bootstrap=config.bootstrap,
            champion=champion,
            human_replay_skill=human_skill,
            match_plan=plan,
            fingerprint=fingerprint,
            state=LifecycleState.MATCH_READY,
        )
    except IterationPreflightError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise IterationPreflightError(str(exc)) from exc


def _revalidate_match_context(context: MatchReadyContext) -> MatchReadyContext:
    if (
        not isinstance(context, MatchReadyContext)
        or getattr(context, "issuer", None) is not _CONTEXT_ISSUER
        or not hasattr(context, "config")
    ):
        raise IterationPreflightError("internally validated context is required")
    current = preflight_match(context.config)
    if current.fingerprint != context.fingerprint:
        raise IterationPreflightError("validated match context identity changed")
    return current


def open_match_after_preflight(
    config: MatchConfig,
    *,
    store: "ImmutableIterationStore",
    runner_factory: Callable[[BaselineStoredContext], Any],
) -> Any:
    context = preflight_baseline_stored(config, store)
    return runner_factory(context)


def build_replay_evidence_manifest(
    context: MatchReadyContext,
    evidence: tuple[CapturedReplay, ...],
) -> ReplayEvidenceManifest:
    current = _revalidate_match_context(context)
    manifest = ReplayEvidenceManifest.create(
        match_plan_sha256=current.match_plan.sha256,
        evidence=tuple(evidence),
        protocol_version=PROTOCOL_VERSION,
        benchmark_version=BENCHMARK_VERSION,
        iteration_protocol_version=ITERATION_PROTOCOL_VERSION,
    )
    manifest.validate_against(current.match_plan)
    return manifest


def _load_approved_replay_manifest(
    path: Path, plan: MatchPlanManifest
) -> ReplayEvidenceManifest:
    manifest = ReplayEvidenceManifest.from_bytes(
        _read_bytes(path, "replay-evidence manifest")
    )
    manifest.validate_against(plan)
    if manifest.sha256 not in APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256:
        raise IterationPreflightError("replay-evidence manifest is not independently approved")
    for item in manifest.evidence:
        artifact = _contained(path.parent, item.artifact_path, "replay artifact path")
        if artifact.is_symlink():
            raise IterationPreflightError("replay artifact cannot be a symlink")
        try:
            actual = _file_sha256(artifact)
        except OSError as exc:
            raise IterationPreflightError("replay artifact is missing or unreadable") from exc
        if actual != item.artifact_sha256:
            raise IterationPreflightError("replay artifact SHA mismatch")
    return manifest


def _replay_fingerprint(
    config: LearningConfig,
    match: MatchReadyContext,
    replay: ReplayEvidenceManifest,
) -> str:
    return _digest(
        {
            "match_fingerprint": match.fingerprint,
            "replay_path": str(config.replay_evidence_manifest_path.resolve()),
            "replay_sha256": replay.sha256,
        }
    )


def preflight_replay(config: LearningConfig) -> ReplayApprovedContext:
    try:
        if not isinstance(config, LearningConfig):
            raise IterationPreflightError("validated learning config is required")
        match = preflight_match(config.match)
        replay = _load_approved_replay_manifest(
            config.replay_evidence_manifest_path, match.match_plan
        )
        return _issue_context(
            ReplayApprovedContext,
            config=config,
            match=match,
            replay_manifest=replay,
            fingerprint=_replay_fingerprint(config, match, replay),
            state=LifecycleState.REPLAY_APPROVED,
        )
    except IterationPreflightError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise IterationPreflightError(str(exc)) from exc


def preflight_learning(config: LearningConfig) -> LearningReadyContext:
    replay_context = preflight_replay(config)
    training = tuple(
        item
        for item in replay_context.replay_manifest.evidence
        if item.case.role == "train"
    )
    if not training:
        raise IterationPreflightError("approved replay has no evidence with role train")
    match = replay_context.match
    payload = {
        "protocol": {
            "iteration_protocol_version": ITERATION_PROTOCOL_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_version": BENCHMARK_VERSION,
            "legal_action_unit": LEGAL_ACTION_UNIT,
            "trajectory_kl": {
                "epsilon": TRAJECTORY_KL_EPSILON,
                "direction": KL_DIRECTION,
                "rollout_source": KL_ROLLOUT_SOURCE,
                "decision_change_rate": DECISION_CHANGE_RATE,
            },
        },
        "bootstrap": {"version": match.bootstrap.version, "sha256": match.bootstrap.sha256},
        "training_opponent": match.champion.public_identity(),
        "match_plan_sha256": match.match_plan.sha256,
        "replay_evidence_manifest_sha256": replay_context.replay_manifest.sha256,
        "human_replay_skill": {
            "version": match.human_replay_skill.version,
            "sha256": match.human_replay_skill.sha256,
        },
        "training_evidence": [item.to_dict() for item in training],
    }
    return _issue_context(
        LearningReadyContext,
        config=config,
        bootstrap=match.bootstrap,
        champion=match.champion,
        human_replay_skill=match.human_replay_skill,
        match_plan=match.match_plan,
        replay_manifest=replay_context.replay_manifest,
        training_evidence=training,
        training_prompt_payload=payload,
        fingerprint=replay_context.fingerprint,
        state=LifecycleState.LEARNING_READY,
    )


def _revalidate_learning_context(
    context: LearningReadyContext,
) -> LearningReadyContext:
    if (
        not isinstance(context, LearningReadyContext)
        or getattr(context, "issuer", None) is not _CONTEXT_ISSUER
        or not hasattr(context, "config")
    ):
        raise IterationPreflightError("internally validated context is required")
    current = preflight_learning(context.config)
    if current.fingerprint != context.fingerprint:
        raise IterationPreflightError("validated learning context identity changed")
    return current


def open_learning_after_preflight(
    config: LearningConfig,
    *,
    controller_factory: Callable[[LearningReadyContext], Any],
) -> Any:
    context = preflight_learning(config)
    return controller_factory(context)


def preflight_iteration(config: LearningConfig) -> LearningReadyContext:
    return preflight_learning(config)


def preflight_for_game(game: str, config: MatchConfig | LearningConfig | None):
    if game != "24_miracle":
        return None
    if isinstance(config, MatchConfig):
        return preflight_match(config)
    if isinstance(config, LearningConfig):
        return preflight_learning(config)
    raise IterationPreflightError("24_miracle phase config is required")


def open_iteration_after_preflight(
    config: LearningConfig,
    *,
    runner_factory: Callable[[LearningReadyContext], Any],
    session_factory: Callable[[LearningReadyContext], Any],
    log_factory: Callable[[LearningReadyContext], Any],
) -> tuple[Any, Any, Any]:
    context = preflight_learning(config)
    return runner_factory(context), session_factory(context), log_factory(context)


def materialize_bootstrap_strategy(context: MatchReadyContext) -> StrategyVersion:
    current = _revalidate_match_context(context)
    try:
        source = {"main.py": current.bootstrap.path.read_text(encoding="utf-8")}
    except (OSError, UnicodeDecodeError) as exc:
        raise IterationPreflightError("bootstrap source changed after preflight") from exc
    strategy = StrategyVersion(
        version="strategy-v0",
        parent_version=None,
        entrypoint="main.py:choose_action",
        dependencies=(),
        source_files=source,
        change_plan=ChangePlan((ChangeOperation("add", "approved minimal bootstrap"),)),
        interpretability_category="deterministic_heuristic_planner",
        probability_query="read-only-complete-action-support",
        complexity=compute_complexity_metrics(source),
        champion_descriptor_sha256=current.champion.sha256,
        match_plan_sha256=current.match_plan.sha256,
        replay_evidence_manifest_sha256=None,
        human_replay_skill_sha256=current.human_replay_skill.sha256,
        iteration_manifest_sha256=current.match_plan.iteration_manifest_sha256,
        research_manifest_sha256=current.match_plan.research_manifest_sha256,
    )
    strategy.validate()
    strategy = strategy._with_computed_hashes()
    if strategy.source_sha256 != current.match_plan.evaluated_policy_source_sha256:
        raise IterationPreflightError("materialized bootstrap source does not match match plan")
    return strategy


def _control_identity(context: LearningReadyContext) -> dict[str, str]:
    return {
        "champion_descriptor_sha256": context.champion.sha256,
        "match_plan_sha256": context.match_plan.sha256,
        "replay_evidence_manifest_sha256": context.replay_manifest.sha256,
        "human_replay_skill_sha256": context.human_replay_skill.sha256,
        "iteration_manifest_sha256": context.match_plan.iteration_manifest_sha256,
        "research_manifest_sha256": context.match_plan.research_manifest_sha256,
    }


def _strategy_identity(strategy: StrategyVersion) -> dict[str, str | None]:
    return {
        "champion_descriptor_sha256": strategy.champion_descriptor_sha256,
        "match_plan_sha256": strategy.match_plan_sha256,
        "replay_evidence_manifest_sha256": strategy.replay_evidence_manifest_sha256,
        "human_replay_skill_sha256": strategy.human_replay_skill_sha256,
        "iteration_manifest_sha256": strategy.iteration_manifest_sha256,
        "research_manifest_sha256": strategy.research_manifest_sha256,
    }


def _skill_identity(skill: MiracleExperienceSkill) -> dict[str, str]:
    return {
        "champion_descriptor_sha256": skill.champion_descriptor_sha256,
        "match_plan_sha256": skill.match_plan_sha256,
        "replay_evidence_manifest_sha256": skill.replay_evidence_manifest_sha256,
        "human_replay_skill_sha256": skill.human_replay_skill_sha256,
        "iteration_manifest_sha256": skill.iteration_manifest_sha256,
        "research_manifest_sha256": skill.research_manifest_sha256,
    }


def materialize_empty_experience(
    context: LearningReadyContext,
) -> MiracleExperienceSkill:
    current = _revalidate_learning_context(context)
    return MiracleExperienceSkill.create_from_learning(
        current,
        version="experience-v0",
        parent_version=None,
        frozen_facts=(),
        training_observations=(),
        supported_heuristics=(),
        failed_heuristics=(),
        unverified_hypotheses=(),
        training_evidence_sha256=(),
        strategy_versions=("strategy-v0",),
        case_identities=(),
        created_by="protocol-bootstrap",
        modified_by="protocol-bootstrap",
    )


class ImmutableIterationStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, group: str, logical_id: str) -> Path:
        _logical_id(logical_id, f"{group} logical ID")
        root = self.root.resolve()
        path = (root / group / f"{logical_id}.json").resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("store path escapes root") from exc
        return path

    @staticmethod
    def _write_once(path: Path, payload: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
        return path

    def store_baseline(
        self, strategy: StrategyVersion, context: MatchReadyContext
    ) -> Path:
        current = _revalidate_match_context(context)
        if not isinstance(strategy, StrategyVersion):
            raise ValueError("baseline storage requires a canonical strategy")
        expected = materialize_bootstrap_strategy(current)
        if strategy.canonical_bytes() != expected.canonical_bytes():
            raise ValueError("strategy-v0 must exactly match the approved bootstrap")
        path = self._path("strategies", "strategy-v0")
        if path.exists():
            stored = self.load_strategy("strategy-v0")
            if stored.canonical_bytes() != expected.canonical_bytes():
                raise ValueError("stored strategy-v0 differs from the approved baseline")
            return path
        return self._write_once(path, expected.canonical_bytes())

    def store_strategy(
        self, strategy: StrategyVersion, context: LearningReadyContext
    ) -> Path:
        current_context = _revalidate_learning_context(context)
        if not isinstance(strategy, StrategyVersion):
            raise ValueError("strategy storage requires a canonical strategy")
        strategy.validate()
        candidate = (
            strategy
            if strategy.sha256 and strategy.source_sha256
            else strategy._with_computed_hashes()
        )
        expected_identity = _control_identity(current_context)
        if candidate.version == "strategy-v0":
            raise ValueError("strategy-v0 must be frozen through the pre-match baseline gate")
        else:
            if candidate.parent_version is None:
                raise ValueError("non-v0 strategy requires parent")
            if not candidate.change_plan.training_evidence:
                raise ValueError("non-v0 strategy requires training evidence")
            if _strategy_identity(candidate) != expected_identity:
                raise ValueError("strategy control identity mismatch")
            try:
                parent = self.load_strategy(candidate.parent_version)
            except ValueError as exc:
                raise ValueError("strategy parent does not exist or is invalid") from exc
            parent_identity = _strategy_identity(parent)
            expected_parent_identity = dict(expected_identity)
            if parent.version == "strategy-v0":
                expected_parent_identity["replay_evidence_manifest_sha256"] = None
            if parent_identity != expected_parent_identity:
                raise ValueError("strategy parent control identity mismatch")
            training_digests = {
                item.sha256 for item in current_context.training_evidence
            }
            if not set(candidate.change_plan.training_evidence).issubset(training_digests):
                raise ValueError("change plan training evidence is not validated train evidence")
            validate_complexity_update(
                parent.complexity, candidate.complexity, candidate.change_plan
            )
        if _strategy_identity(candidate) != expected_identity:
            raise ValueError("strategy control identity mismatch")
        path = self._path("strategies", candidate.version)
        if path.exists():
            raise FileExistsError(path)
        return self._write_once(path, candidate.canonical_bytes())

    def load_strategy(self, version: str) -> StrategyVersion:
        path = self._path("strategies", version)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"strategy version does not exist: {version}")
        return StrategyVersion.from_bytes(path.read_bytes())

    def store_skill(
        self, skill: MiracleExperienceSkill, context: LearningReadyContext
    ) -> Path:
        current_context = _revalidate_learning_context(context)
        if not isinstance(skill, MiracleExperienceSkill):
            raise ValueError("Skill storage requires a canonical experience Skill")
        skill.validate()
        expected_identity = _control_identity(current_context)
        if _skill_identity(skill) != expected_identity:
            raise ValueError("experience Skill control identity mismatch")
        if skill.version == "experience-v0":
            expected = materialize_empty_experience(current_context)
            if skill.canonical_bytes() != expected.canonical_bytes():
                raise ValueError("experience-v0 must exactly match the empty baseline")
        else:
            if skill.parent_version is None:
                raise ValueError("non-v0 experience Skill requires parent")
            try:
                parent = self.load_skill(skill.parent_version)
            except ValueError as exc:
                raise ValueError("experience Skill parent does not exist or is invalid") from exc
            if _skill_identity(parent) != expected_identity:
                raise ValueError("experience Skill parent control identity mismatch")
            training = {item.sha256 for item in current_context.training_evidence}
            cases = {item.case.case_id for item in current_context.training_evidence}
            if not skill.training_evidence_sha256 or not set(
                skill.training_evidence_sha256
            ).issubset(training):
                raise ValueError("experience Skill evidence is not approved train evidence")
            if not skill.case_identities or not set(skill.case_identities).issubset(cases):
                raise ValueError("experience Skill case is not an approved train case")
        path = self._path("skills", skill.version)
        if path.exists():
            raise FileExistsError(path)
        return self._write_once(path, skill.canonical_bytes())

    def load_skill(self, version: str) -> MiracleExperienceSkill:
        path = self._path("skills", version)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Skill version does not exist: {version}")
        return MiracleExperienceSkill.from_bytes(path.read_bytes())

    def _is_ancestor(self, source: StrategyVersion, target: StrategyVersion) -> bool:
        current = source
        seen: set[str] = set()
        while current.parent_version is not None:
            if current.parent_version == target.version:
                return True
            if current.parent_version in seen:
                raise ValueError("strategy parent cycle detected")
            seen.add(current.parent_version)
            current = self.load_strategy(current.parent_version)
        return False

    def rollback(
        self,
        *,
        source_version: str,
        target_version: str,
        reason: str,
        operator: str,
        context: LearningReadyContext,
    ) -> Path:
        current_context = _revalidate_learning_context(context)
        _required_text(reason, "rollback reason")
        _logical_id(operator, "rollback operator")
        source = self.load_strategy(source_version)
        target = self.load_strategy(target_version)
        expected_identity = _control_identity(current_context)
        expected_target_identity = dict(expected_identity)
        if target.version == "strategy-v0":
            expected_target_identity["replay_evidence_manifest_sha256"] = None
        if (
            _strategy_identity(source) != expected_identity
            or _strategy_identity(target) != expected_target_identity
        ):
            raise ValueError("rollback strategy control identity mismatch")
        if not self._is_ancestor(source, target):
            raise ValueError("rollback target must be an ancestor of source")
        iteration_id = f"rollback-{uuid.uuid4().hex}"
        record = {
            "schema_version": ROLLBACK_SCHEMA_VERSION,
            "operation": "rollback",
            "new_iteration_id": iteration_id,
            "source_version": source.version,
            "source_sha256": source.sha256,
            "target_version": target.version,
            "target_sha256": target.sha256,
            "reason": reason,
            "operator": operator,
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_version": BENCHMARK_VERSION,
            "iteration_protocol_version": ITERATION_PROTOCOL_VERSION,
            "bootstrap_sha256": current_context.bootstrap.sha256,
            **expected_identity,
            "immutable": True,
        }
        return self._write_once(
            self._path("iterations", iteration_id), _canonical_bytes(record)
        )


def preflight_baseline_stored(
    config: MatchConfig, store: ImmutableIterationStore
) -> BaselineStoredContext:
    match = preflight_match(config)
    if not isinstance(store, ImmutableIterationStore):
        raise IterationPreflightError("validated immutable store is required")
    draft = materialize_bootstrap_strategy(match)
    store.store_baseline(draft, match)
    stored = store.load_strategy("strategy-v0")
    current_match = preflight_match(config)
    expected = materialize_bootstrap_strategy(current_match)
    if stored.canonical_bytes() != expected.canonical_bytes():
        raise IterationPreflightError("stored baseline canonical bytes changed")
    if (
        stored.version != current_match.match_plan.evaluated_policy_version
        or stored.source_sha256
        != current_match.match_plan.evaluated_policy_source_sha256
    ):
        raise IterationPreflightError("stored baseline does not match the match plan")
    fingerprint = _digest(
        {
            "match": current_match.fingerprint,
            "store_root": str(store.root.resolve()),
            "version": stored.version,
            "strategy_sha256": stored.sha256,
            "source_sha256": stored.source_sha256,
        }
    )
    return _issue_context(
        BaselineStoredContext,
        config=config,
        store_root=store.root,
        version=stored.version,
        strategy_sha256=stored.sha256,
        source_sha256=stored.source_sha256,
        cases=current_match.match_plan.cases,
        match=current_match,
        fingerprint=fingerprint,
        state=LifecycleState.BASELINE_STORED,
    )


def _revalidate_baseline_context(
    context: BaselineStoredContext,
) -> BaselineStoredContext:
    if (
        not isinstance(context, BaselineStoredContext)
        or getattr(context, "issuer", None) is not _CONTEXT_ISSUER
        or not hasattr(context, "config")
    ):
        raise IterationPreflightError("internally validated baseline context is required")
    current = preflight_baseline_stored(
        context.config, ImmutableIterationStore(context.store_root)
    )
    if current.fingerprint != context.fingerprint:
        raise IterationPreflightError("stored baseline identity changed")
    return current


def preflight_candidate_stored(
    config: LearningConfig,
    store: ImmutableIterationStore,
    version: str,
    experience_skill_version: str,
) -> CandidateStoredContext:
    context = preflight_learning(config)
    if not isinstance(store, ImmutableIterationStore):
        raise IterationPreflightError("validated immutable store is required")
    strategy = store.load_strategy(version)
    try:
        skill = store.load_skill(experience_skill_version)
    except ValueError as exc:
        raise IterationPreflightError("candidate Experience Skill is missing or invalid") from exc
    if strategy.version == "strategy-v0" or skill.version == "experience-v0":
        raise IterationPreflightError("candidate strategy and Experience Skill must be non-v0")
    if _strategy_identity(strategy) != _control_identity(context):
        raise IterationPreflightError("stored candidate control identity mismatch")
    if _skill_identity(skill) != _control_identity(context):
        raise IterationPreflightError("stored candidate Skill control identity mismatch")
    if strategy.version not in skill.strategy_versions:
        raise IterationPreflightError("Experience Skill does not reference the candidate strategy")
    strategy_generation = re.fullmatch(r"strategy-v([1-9][0-9]*)", strategy.version)
    skill_generation = re.fullmatch(r"experience-v([1-9][0-9]*)", skill.version)
    if not strategy_generation or not skill_generation or strategy_generation.group(1) != skill_generation.group(1):
        raise IterationPreflightError("candidate strategy and Skill generation mismatch")
    generation = int(strategy_generation.group(1))
    expected_strategy_parent = f"strategy-v{generation - 1}"
    expected_skill_parent = f"experience-v{generation - 1}"
    if strategy.parent_version != expected_strategy_parent or skill.parent_version != expected_skill_parent:
        raise IterationPreflightError("candidate strategy and Skill parent generation mismatch")
    training = {item.sha256 for item in context.training_evidence}
    cases = {item.case.case_id for item in context.training_evidence}
    if not set(strategy.change_plan.training_evidence).issubset(training):
        raise IterationPreflightError("candidate strategy contains non-train evidence")
    if not set(skill.training_evidence_sha256).issubset(training) or not set(skill.case_identities).issubset(cases):
        raise IterationPreflightError("candidate Skill contains non-train evidence")
    fingerprint = _digest(
        {
            "learning": context.fingerprint,
            "store_root": str(store.root.resolve()),
            "version": strategy.version,
            "sha256": strategy.sha256,
            "source_sha256": strategy.source_sha256,
            "experience_skill_version": skill.version,
            "experience_skill_sha256": skill.sha256,
        }
    )
    return _issue_context(
        CandidateStoredContext,
        config=config,
        store_root=store.root,
        version=strategy.version,
        strategy_sha256=strategy.sha256,
        source_sha256=strategy.source_sha256,
        experience_skill_version=skill.version,
        experience_skill_sha256=skill.sha256,
        champion_descriptor_sha256=context.champion.sha256,
        human_replay_skill_sha256=context.human_replay_skill.sha256,
        match_plan_sha256=context.match_plan.sha256,
        replay_evidence_manifest_sha256=context.replay_manifest.sha256,
        research_manifest_sha256=context.match_plan.research_manifest_sha256,
        iteration_manifest_sha256=context.match_plan.iteration_manifest_sha256,
        fingerprint=fingerprint,
        state=LifecycleState.CANDIDATE_STORED,
    )


def _revalidate_candidate_context(
    context: CandidateStoredContext,
) -> CandidateStoredContext:
    if (
        not isinstance(context, CandidateStoredContext)
        or getattr(context, "issuer", None) is not _CONTEXT_ISSUER
        or not hasattr(context, "config")
    ):
        raise IterationPreflightError("internally validated candidate context is required")
    current = preflight_candidate_stored(
        context.config,
        ImmutableIterationStore(context.store_root),
        context.version,
        context.experience_skill_version,
    )
    if current.fingerprint != context.fingerprint:
        raise IterationPreflightError("stored candidate identity changed")
    return current


def _load_approved_evaluation_plan(
    path: Path, candidate: CandidateStoredContext
) -> CandidateEvaluationPlan:
    plan = CandidateEvaluationPlan.from_bytes(
        _read_bytes(path, "candidate evaluation plan")
    )
    if plan.sha256 not in APPROVED_CANDIDATE_EVALUATION_PLAN_SHA256:
        raise IterationPreflightError("candidate evaluation plan is not independently approved")
    store = ImmutableIterationStore(candidate.store_root)
    baseline = store.load_strategy("strategy-v0")
    strategy = store.load_strategy(candidate.version)
    skill = store.load_skill(candidate.experience_skill_version)
    expected = (
        baseline.version,
        baseline.sha256,
        baseline.source_sha256,
        strategy.version,
        strategy.sha256,
        strategy.source_sha256,
        skill.version,
        skill.sha256,
        candidate.champion_descriptor_sha256,
        candidate.human_replay_skill_sha256,
        candidate.match_plan_sha256,
        candidate.replay_evidence_manifest_sha256,
        candidate.research_manifest_sha256,
        candidate.iteration_manifest_sha256,
    )
    actual = (
        plan.baseline_strategy_version,
        plan.baseline_strategy_sha256,
        plan.baseline_source_sha256,
        plan.candidate_strategy_version,
        plan.candidate_strategy_sha256,
        plan.candidate_source_sha256,
        plan.experience_skill_version,
        plan.experience_skill_sha256,
        plan.champion_descriptor_sha256,
        plan.human_replay_skill_sha256,
        plan.training_match_plan_sha256,
        plan.training_replay_evidence_sha256,
        plan.research_manifest_sha256,
        plan.iteration_manifest_sha256,
    )
    if actual != expected:
        raise IterationPreflightError("candidate evaluation plan identity mismatch")
    return plan


def preflight_evaluation(config: EvaluationConfig) -> EvaluationReadyContext:
    if not isinstance(config, EvaluationConfig):
        raise IterationPreflightError("evaluation config is required")
    store = ImmutableIterationStore(config.store_root)
    candidate = preflight_candidate_stored(
        config.learning_config,
        store,
        config.candidate_strategy_version,
        config.experience_skill_version,
    )
    plan = _load_approved_evaluation_plan(config.evaluation_plan_path, candidate)
    fingerprint = _digest(
        {
            "candidate": candidate.fingerprint,
            "evaluation_plan_path": str(config.evaluation_plan_path.resolve()),
            "evaluation_plan_sha256": plan.sha256,
        }
    )
    return _issue_context(
        EvaluationReadyContext,
        config=config,
        candidate=candidate,
        plan=plan,
        fingerprint=fingerprint,
        state=LifecycleState.EVALUATION_READY,
    )


def _revalidate_evaluation_context(
    context: EvaluationReadyContext,
) -> EvaluationReadyContext:
    if (
        not isinstance(context, EvaluationReadyContext)
        or getattr(context, "issuer", None) is not _CONTEXT_ISSUER
        or not hasattr(context, "config")
    ):
        raise IterationPreflightError("internally validated evaluation context is required")
    current = preflight_evaluation(context.config)
    if current.fingerprint != context.fingerprint:
        raise IterationPreflightError("evaluation plan or candidate identity changed")
    return current


def open_evaluation_after_preflight(
    config: EvaluationConfig,
    *,
    runner_factory: Callable[[EvaluationReadyContext], Any],
) -> Any:
    context = preflight_evaluation(config)
    return runner_factory(context)


def _load_approved_evaluation_evidence(
    path: Path, evaluation: EvaluationReadyContext
) -> EvaluationEvidenceManifest:
    manifest = EvaluationEvidenceManifest.from_bytes(
        _read_bytes(path, "evaluation evidence manifest"), evaluation.plan
    )
    if manifest.sha256 not in APPROVED_EVALUATION_EVIDENCE_MANIFEST_SHA256:
        raise IterationPreflightError("evaluation evidence manifest is not independently approved")
    for item in manifest.evidence:
        for label, relative, expected in (
            ("baseline", item.baseline_artifact_path, item.baseline_artifact_sha256),
            ("candidate", item.candidate_artifact_path, item.candidate_artifact_sha256),
        ):
            artifact = _contained(path.parent, relative, f"{label} evaluation artifact path")
            if artifact.is_symlink() or _file_sha256(Path(_readable_file(artifact, f"{label} evaluation artifact"))) != expected:
                raise IterationPreflightError(f"{label} evaluation artifact SHA mismatch")
    return manifest


def derive_iteration_outcome(config: AcceptanceConfig) -> IterationOutcome:
    if not isinstance(config, AcceptanceConfig):
        raise IterationPreflightError("acceptance config is required")
    evaluation = preflight_evaluation(config.evaluation)
    evidence = _load_approved_evaluation_evidence(
        config.evaluation_evidence_path, evaluation
    )
    acceptance = IterationAcceptanceManifest.from_bytes(
        _read_bytes(config.acceptance_manifest_path, "iteration acceptance manifest")
    )
    if acceptance.sha256 not in APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256:
        raise IterationPreflightError("iteration acceptance manifest is not independently approved")
    acceptance.validate_against(evaluation.plan, evidence)
    current = _revalidate_evaluation_context(evaluation)
    return IterationOutcome(
        current.candidate.version,
        current.candidate.strategy_sha256,
        LifecycleState.COMPLETED
        if acceptance.evaluation_status == "complete"
        else LifecycleState.INCOMPLETE,
        acceptance.sha256,
        acceptance.fake_only,
    )


def iteration_protocol_manifest() -> dict[str, Any]:
    bootstrap = default_bootstrap_template()
    bootstrap.validate()
    return {
        "iteration_protocol_version": ITERATION_PROTOCOL_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "bootstrap_template": {
            "version": bootstrap.version,
            "sha256": bootstrap.sha256,
        },
        "legal_action_unit": LEGAL_ACTION_UNIT,
        "allowed_change_operations": sorted(_CHANGE_OPERATIONS),
        "interpretable_strategy_categories": sorted(_INTERPRETABLE_CATEGORIES),
        "human_champion_schema_version": HUMAN_CHAMPION_SCHEMA_VERSION,
        "human_replay_skill_schema_version": HUMAN_REPLAY_SKILL_SCHEMA_VERSION,
        "experience_skill_schema_version": EXPERIENCE_SKILL_SCHEMA_VERSION,
        "match_plan_schema_version": MATCH_PLAN_SCHEMA_VERSION,
        "replay_evidence_schema_version": REPLAY_EVIDENCE_SCHEMA_VERSION,
        "candidate_evaluation_plan_schema_version": CANDIDATE_EVALUATION_PLAN_SCHEMA_VERSION,
        "evaluation_evidence_schema_version": EVALUATION_EVIDENCE_SCHEMA_VERSION,
        "iteration_acceptance_schema_version": ITERATION_ACCEPTANCE_SCHEMA_VERSION,
        "lifecycle_states": [state.value for state in LifecycleState],
        "lifecycle_transitions": [
            "PLANNED->MATCH_READY",
            "MATCH_READY->BASELINE_STORED",
            "BASELINE_STORED->MATCH_RUNNER",
            "MATCH_READY->REPLAY_CAPTURED_UNAPPROVED",
            "REPLAY_CAPTURED_UNAPPROVED->REPLAY_APPROVED",
            "REPLAY_APPROVED->LEARNING_READY",
            "LEARNING_READY->CANDIDATE_STORED",
            "CANDIDATE_STORED->EVALUATION_READY",
            "EVALUATION_READY->COMPLETED|INCOMPLETE",
        ],
        "independent_approval_boundaries": [
            "human_champion",
            "human_replay_skill",
            "match_plan_manifest",
            "replay_evidence_manifest",
            "candidate_evaluation_plan",
            "evaluation_evidence_manifest",
            "iteration_acceptance_manifest",
        ],
        "baseline_storage_gate": "store_strategy_v0_then_reload_and_verify_before_runner",
        "candidate_bundle": "same_generation_strategy_and_experience_skill",
        "evaluation_isolation": "independent_candidate_plan_and_evidence_not_training_input",
        "terminal_status": "derived_from_independently_approved_first_hand_evidence",
        "fake_completed_label_required": True,
        "replay_case_coverage": "exactly_once_against_approved_match_plan",
        "mutable_paths_revalidate_control_plane": True,
        "immutable_history": True,
        "rollback": "append_new_ancestor_audit_record",
        "training_validation_isolation": "approved_manifest_role_train_only",
        "static_validation_is_not_runtime_sandbox": True,
        # Approval table contents are deliberately not serialized here. Their
        # state must not change the frozen protocol identity, including during
        # test-only monkeypatching.
        "production_approval_source": "version-controlled-module-constants",
        "authoritative_execution": "blocked",
    }


def canonical_iteration_protocol_manifest_bytes() -> bytes:
    return _canonical_bytes(iteration_protocol_manifest())
