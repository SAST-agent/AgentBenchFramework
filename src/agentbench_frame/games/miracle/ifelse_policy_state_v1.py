"""Explicit state and deterministic distribution providers for Miracle HL.

The legacy ``miracle_ifelse`` process emits several atomic Judge operations
from one ``play()`` call.  Every operation is followed by a fresh Judge
observation while Python continues inside the active phase.  This module makes
that continuation explicit and serializable so formal policy KL can compare
old and new policies on the same ``(observation, memory)`` context.

No epsilon smoothing is performed here.  Providers return strict one-hot base
distributions; the formal information-gain layer owns the shared measurement
channel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from agentbench_frame.eval.measurement import (
    ActionSupport,
    canonical_state_id,
)
from agentbench_frame.games.miracle.decision_kl_v1 import (
    build_trusted_action_support,
)
from agentbench_frame.games.miracle.research_protocol import (
    canonical_command,
    command_action_id,
    one_hot,
)


POLICY_CONFIG_SCHEMA_VERSION = "24-miracle-ifelse-policy-config-v1"
POLICY_MEMORY_SCHEMA_VERSION = "24-miracle-ifelse-policy-memory-v1"
STATE_MACHINE_VERSION = "24-miracle-ifelse-explicit-state-machine-v1"
PROVIDER_SCHEMA_VERSION = "24-miracle-ifelse-distribution-provider-v1"
POLICY_COMPARISON_SCHEMA_VERSION = "24-miracle-ifelse-policy-comparison-v1"
POLICY_PAIR_DECISION_SCHEMA_VERSION = "24-miracle-ifelse-policy-pair-decision-v1"
UNKNOWN_CONFIG_VALUE = "unknown"
LEGACY_SOURCE_FILES = (
    "Data.json",
    "ai_client.py",
    "calculator.py",
    "card.py",
    "gameunit.py",
    "main.py",
)
MAX_POLICY_SOURCE_FILE_BYTES = 2_000_000
MAX_REPLAY_TRACE_BYTES = 64_000_000
VALID_ARTIFACTS = frozenset(
    {"HolyLight", "SalamanderShield", "InfernoFlame", "WindBlessing"}
)
VALID_CREATURES = frozenset(
    {
        "Archer",
        "Swordsman",
        "BlackBat",
        "Priest",
        "VolcanoDragon",
        "Inferno",
        "FrostDragon",
    }
)


class IncompletePolicyEvidenceError(ValueError):
    """A formal policy identity or decision context cannot be proved."""


class PolicySourceError(ValueError):
    """A legacy policy source tree is missing, replaced, or unapproved."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} contains invalid Unicode") from exc
    return value


def _strict_sha256(value: Any, label: str) -> str:
    value = _strict_text(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if type(key) is not str:
            raise TypeError("mapping keys must be strings")
        if isinstance(item, Mapping):
            frozen[key] = _freeze_mapping(item)
        elif type(item) in {list, tuple}:
            frozen[key] = tuple(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


def _freeze_sequence(value: Any) -> Any:
    if type(value) in {list, tuple}:
        return tuple(_freeze_sequence(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class MiracleConfigSpec:
    name: str
    attribute: str | None
    kind: str
    default_v0: Any
    default_v1: Any


_BOOLEAN_CONFIG = (
    ("MIRACLE_GATE_DEFENSE", "gate_defense_enabled"),
    ("MIRACLE_ANTIDECOY", "antidecoy_enabled"),
    ("MIRACLE_RACE_LANE", "race_lane_enabled"),
    ("MIRACLE_FRONT_SWORD3", "front_sword3_enabled"),
    ("MIRACLE_FRONT_ARCHER3", "front_archer3_enabled"),
    ("MIRACLE_PAIRED_CLOCK", "paired_clock_enabled"),
    ("MIRACLE_FRONT_ARCHER_LANE", "front_archer_lane_enabled"),
    ("MIRACLE_INFERNO_CLOCK", "inferno_clock_enabled"),
    ("MIRACLE_LANE_MELEE_GUARD", "lane_melee_guard_enabled"),
    ("MIRACLE_LANE_RANGED_GUARD", "lane_ranged_guard_enabled"),
    ("MIRACLE_CLOCK_CONDITIONAL_GUARD", "clock_conditional_guard_enabled"),
    ("MIRACLE_CLOCK_STATE_TABLE", "clock_state_table_enabled"),
    ("MIRACLE_SECOND_ARCHER_LANE", "second_archer_lane_enabled"),
    ("MIRACLE_CONDITIONAL_SECOND_LANE", "conditional_second_lane_enabled"),
    ("MIRACLE_CONDITIONAL_SECOND_LANE_STRICT", "conditional_second_lane_strict"),
    ("MIRACLE_SWORD_CLOCK_SUPPORT", "sword_clock_support_enabled"),
    ("MIRACLE_PREGATE_SWORD_INTERCEPT", "pregate_sword_intercept_enabled"),
    ("MIRACLE_EARLY_SWORD_INTERCEPT", "early_sword_intercept_enabled"),
    ("MIRACLE_MELEE_ARCHER_DISRUPT", "melee_archer_disrupt_enabled"),
    ("MIRACLE_LANE_BODYGUARD", "lane_bodyguard_enabled"),
    ("MIRACLE_LANE_KILL_GUARD", "lane_kill_guard_enabled"),
    ("MIRACLE_LANE_ARTIFACT_GUARD", "lane_artifact_guard_enabled"),
    ("MIRACLE_LATE_SWORD_KILL_GUARD", "late_sword_kill_guard_enabled"),
    ("MIRACLE_LATE_SWORD_ARTIFACT_GUARD", "late_sword_artifact_guard_enabled"),
    ("MIRACLE_LATE_PREGATE_SWORD_FOCUS", "late_pregate_sword_focus_enabled"),
    ("MIRACLE_LATE_INFERNO_GUARD", "late_inferno_guard_enabled"),
    ("MIRACLE_LATE_INFERNO_ARTIFACT_GUARD", "late_inferno_artifact_guard_enabled"),
    ("MIRACLE_LATE_CLOCK_ARTIFACT_TABLE", "late_clock_artifact_table_enabled"),
    ("MIRACLE_EARLY_MIXED_GATE_GUARD", "early_mixed_gate_guard_enabled"),
    ("MIRACLE_EARLY_MIXED_ARTIFACT_GUARD", "early_mixed_artifact_guard_enabled"),
    ("MIRACLE_EARLY_GATE_ANCHOR", "early_gate_anchor_enabled"),
    ("MIRACLE_EARLY_NONLANE_GATE_BLOCKER", "early_nonlane_gate_blocker_enabled"),
    ("MIRACLE_EARLY_GATE_KILL_ONLY", "early_gate_kill_only_enabled"),
    ("MIRACLE_PREGATE_ARCHER_SUPPORT", "pregate_archer_support_enabled"),
    ("MIRACLE_ARCHER_BURST_GUARD", "archer_burst_guard_enabled"),
    ("MIRACLE_PREARCHER_REPLACEMENT_GUARD", "prearcher_replacement_guard_enabled"),
    ("MIRACLE_PREARCHER_ID_GUARD", "prearcher_id_guard_enabled"),
    ("MIRACLE_DIRECT_ARCHER_LOCK", "direct_archer_lock_enabled"),
    ("MIRACLE_OFFENSIVE_LANE_LOCK", "offensive_lane_lock_enabled"),
    ("MIRACLE_SAFE_ARCHER_ROUTE", "safe_archer_route_enabled"),
    ("MIRACLE_RELAY_ARCHER_ROUTE", "relay_archer_route_enabled"),
    ("MIRACLE_SECOND_WAVE_ARCHER_ROUTE", "second_wave_archer_route_enabled"),
    ("MIRACLE_FRONT_SUMMON_VACANCY", "front_summon_vacancy_enabled"),
    ("MIRACLE_ANTI_BLACKBAT", "anti_blackbat_enabled"),
    ("MIRACLE_AIR_OPENING_ARCHER_RUSH", "air_opening_archer_rush_enabled"),
    ("MIRACLE_AIR_POSTRUSH_BLACKBAT_GUARD", "air_postrush_blackbat_guard_enabled"),
    ("MIRACLE_LATE_AIR_RACE_RELEASE", "late_air_race_release_enabled"),
    ("MIRACLE_LATE_FROST_ARTIFACT", "late_frost_artifact_enabled"),
    ("MIRACLE_SAVE_ARTIFACT_FOR_FROST", "save_artifact_for_frost_enabled"),
    ("MIRACLE_LATE_SHELL_DRAGON_PRESSURE", "late_shell_dragon_pressure_enabled"),
    ("MIRACLE_LATE_SHELL_DRAGON_PRESSURE_SHORT", "late_shell_dragon_pressure_short"),
    ("MIRACLE_LATE_BLACKBAT_GATE_GUARD", "late_blackbat_gate_guard_enabled"),
    ("MIRACLE_SWORD_OFFSET_DECOY", "sword_offset_decoy_enabled"),
    ("MIRACLE_SECOND_OFFSET_DECOY", "second_offset_decoy_enabled"),
    ("MIRACLE_PRIEST_BAT_DECOY", "priest_bat_decoy_enabled"),
    ("MIRACLE_SOUTH_BAT_SCREEN", "south_bat_screen_enabled"),
    ("MIRACLE_MID_ARCHER_LATTICE_BLOCK", "mid_archer_lattice_block_enabled"),
    ("MIRACLE_RANK13_ARCHER90_FOCUS", "rank13_archer90_focus_enabled"),
    ("MIRACLE_RANK13_SWORD103_PRIEST_AURA", "rank13_sword103_priest_aura_enabled"),
)

MIRACLE_CONFIG_SPECS: tuple[MiracleConfigSpec, ...] = tuple(
    MiracleConfigSpec(name, attribute, "bool", False, False)
    for name, attribute in _BOOLEAN_CONFIG
) + (
    MiracleConfigSpec("MIRACLE_CAMP1_OPENING", "camp1_opening", "opening", "FF", "SF"),
    MiracleConfigSpec("MIRACLE_ARTIFACT", None, "artifact", "InfernoFlame", "InfernoFlame"),
    MiracleConfigSpec(
        "MIRACLE_DECK",
        None,
        "deck",
        ("Priest", "Archer", "Swordsman"),
        ("Priest", "Archer", "Swordsman"),
    ),
)
_CONFIG_BY_NAME = {spec.name: spec for spec in MIRACLE_CONFIG_SPECS}
if len(MIRACLE_CONFIG_SPECS) != 62 or len(_CONFIG_BY_NAME) != 62:
    raise RuntimeError("Miracle policy configuration inventory must contain 62 unique inputs")


def _validate_config_value(spec: MiracleConfigSpec, value: Any) -> Any:
    if value == UNKNOWN_CONFIG_VALUE:
        return value
    if spec.kind == "bool":
        if type(value) is not bool:
            raise TypeError(f"{spec.name} must be a strict boolean")
        return value
    if spec.kind == "opening":
        if type(value) is not str:
            raise TypeError(f"{spec.name} must be a string")
        if value not in {"FF", "SF", "IF"}:
            raise ValueError(f"{spec.name} must be FF, SF, or IF")
        return value
    if spec.kind == "artifact":
        normalized = _strict_text(value, spec.name)
        if normalized not in VALID_ARTIFACTS:
            raise ValueError(f"{spec.name} is not a known Judge artifact")
        return normalized
    if spec.kind == "deck":
        if type(value) not in {list, tuple} or len(value) != 3:
            raise TypeError(f"{spec.name} must contain exactly three entries")
        normalized = tuple(_strict_text(item, spec.name) for item in value)
        if len(set(normalized)) != 3:
            raise ValueError(f"{spec.name} must contain three distinct creatures")
        if any(item not in VALID_CREATURES for item in normalized):
            raise ValueError(f"{spec.name} contains an unknown Judge creature")
        return normalized
    raise RuntimeError(f"unknown config kind: {spec.kind}")


@dataclass(frozen=True, slots=True)
class PolicyConfigV1:
    values: Mapping[str, Any]
    evidence: str

    def __post_init__(self) -> None:
        if set(self.values) != set(_CONFIG_BY_NAME):
            raise ValueError("policy config must contain exactly all 62 inputs")
        normalized = {
            name: _validate_config_value(_CONFIG_BY_NAME[name], self.values[name])
            for name in sorted(self.values)
        }
        object.__setattr__(self, "values", _freeze_mapping(normalized))
        _strict_text(self.evidence, "config evidence")

    @classmethod
    def from_explicit(cls, values: Mapping[str, Any]) -> "PolicyConfigV1":
        if not isinstance(values, Mapping):
            raise TypeError("explicit policy config must be a mapping")
        return cls(dict(values), "explicit_complete")

    @classmethod
    def historical_unknown(
        cls, *, observed: Mapping[str, Any] | None = None
    ) -> "PolicyConfigV1":
        observed = {} if observed is None else dict(observed)
        extra = set(observed) - set(_CONFIG_BY_NAME)
        if extra:
            raise ValueError(f"unknown historical config inputs: {sorted(extra)}")
        values = {
            name: observed.get(name, UNKNOWN_CONFIG_VALUE)
            for name in _CONFIG_BY_NAME
        }
        return cls(values, "historical_partial_unknown")

    @property
    def complete(self) -> bool:
        return all(value != UNKNOWN_CONFIG_VALUE for value in self.values.values())

    def require_complete(self) -> None:
        if not self.complete:
            missing = sorted(
                name
                for name, value in self.values.items()
                if value == UNKNOWN_CONFIG_VALUE
            )
            raise IncompletePolicyEvidenceError(
                f"policy config contains unknown values: {missing}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_CONFIG_SCHEMA_VERSION,
            "evidence": self.evidence,
            "complete": self.complete,
            "values": {
                name: list(value) if type(value) is tuple else value
                for name, value in self.values.items()
            },
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderIdentityV1:
    version: str
    source_identity: str
    config_identity: str
    state_machine_identity: str = STATE_MACHINE_VERSION
    schema_version: str = PROVIDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _strict_text(self.version, "policy version")
        _strict_sha256(self.source_identity, "policy source identity")
        _strict_sha256(self.config_identity, "policy config identity")
        if self.state_machine_identity != STATE_MACHINE_VERSION:
            raise ValueError("policy state-machine identity mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "source_identity": self.source_identity,
            "config_identity": self.config_identity,
            "state_machine_identity": self.state_machine_identity,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class LegacyPolicySourceV1:
    """Verified six-file source identity for the legacy if-else bot."""

    root: Path
    version: str
    main_sha256: str
    canonical_tree_sha256: str
    legacy_tree_sha256: str
    files: tuple[tuple[str, int, str], ...]

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        version: str,
        expected_main_sha256: str | None = None,
        expected_canonical_tree_sha256: str | None = None,
        expected_legacy_tree_sha256: str | None = None,
    ) -> "LegacyPolicySourceV1":
        path = Path(root).resolve(strict=True)
        if not path.is_dir() or path.is_symlink():
            raise PolicySourceError("policy source root must be a real directory")
        _strict_text(version, "policy version")
        entries: list[tuple[str, int, str]] = []
        payloads: dict[str, bytes] = {}
        for name in LEGACY_SOURCE_FILES:
            candidate = path / name
            if (
                not candidate.is_file()
                or candidate.is_symlink()
                or candidate.parent.resolve(strict=True) != path
            ):
                raise PolicySourceError(f"policy source file is unavailable: {name}")
            payload = candidate.read_bytes()
            if not payload or len(payload) > MAX_POLICY_SOURCE_FILE_BYTES:
                raise PolicySourceError(f"policy source file size is invalid: {name}")
            if payload.startswith(b"\xef\xbb\xbf"):
                raise PolicySourceError(f"policy source has a BOM: {name}")
            try:
                payload.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise PolicySourceError(
                    f"policy source is not strict UTF-8: {name}"
                ) from exc
            payloads[name] = payload
            entries.append((name, len(payload), hashlib.sha256(payload).hexdigest()))
        # The captured H08 and historical v0/v1 manifests were produced on
        # Windows with case-insensitive path ordering.  Bind that ordering
        # explicitly so the identity is portable to POSIX.
        entries.sort(key=lambda item: (item[0].casefold(), item[0]))
        main_sha = next(item[2] for item in entries if item[0] == "main.py")
        canonical_payload = "".join(
            f"{name}\t{digest}\n" for name, _size, digest in entries
        ).encode("utf-8")
        canonical_tree = hashlib.sha256(canonical_payload).hexdigest()
        legacy_hasher = hashlib.sha256()
        for name, _size, _digest_value in entries:
            legacy_hasher.update(name.encode("utf-8"))
            legacy_hasher.update(payloads[name])
        legacy_tree = legacy_hasher.hexdigest()
        for actual, expected, label in (
            (main_sha, expected_main_sha256, "main.py"),
            (canonical_tree, expected_canonical_tree_sha256, "canonical tree"),
            (legacy_tree, expected_legacy_tree_sha256, "legacy tree"),
        ):
            if expected is not None:
                _strict_sha256(expected, f"expected {label} SHA")
                if actual != expected:
                    raise PolicySourceError(f"policy {label} identity mismatch")
        return cls(path, version, main_sha, canonical_tree, legacy_tree, tuple(entries))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "24-miracle-ifelse-policy-source-v1",
            "version": self.version,
            "main_sha256": self.main_sha256,
            "canonical_tree_sha256": self.canonical_tree_sha256,
            "legacy_tree_sha256": self.legacy_tree_sha256,
            "files": [
                {"path": name, "bytes": size, "sha256": digest}
                for name, size, digest in self.files
            ],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def revalidate(self) -> None:
        current = LegacyPolicySourceV1.open(
            self.root,
            version=self.version,
            expected_main_sha256=self.main_sha256,
            expected_canonical_tree_sha256=self.canonical_tree_sha256,
            expected_legacy_tree_sha256=self.legacy_tree_sha256,
        )
        if current.to_dict() != self.to_dict():
            raise PolicySourceError("policy source identity changed")


@dataclass(frozen=True, slots=True)
class PolicyComparisonIdentityV1:
    old_provider: ProviderIdentityV1
    new_provider: ProviderIdentityV1
    old_config: PolicyConfigV1
    new_config: PolicyConfigV1
    schema_version: str = POLICY_COMPARISON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.old_provider.config_identity != self.old_config.sha256:
            raise ValueError("old provider config identity mismatch")
        if self.new_provider.config_identity != self.new_config.sha256:
            raise ValueError("new provider config identity mismatch")

    @classmethod
    def for_test(
        cls,
        *,
        old_version: str,
        new_version: str,
        config: PolicyConfigV1,
    ) -> "PolicyComparisonIdentityV1":
        old_source = hashlib.sha256(("test:" + old_version).encode()).hexdigest()
        new_source = hashlib.sha256(("test:" + new_version).encode()).hexdigest()
        return cls(
            ProviderIdentityV1(old_version, old_source, config.sha256),
            ProviderIdentityV1(new_version, new_source, config.sha256),
            config,
            config,
        )

    @classmethod
    def from_sources(
        cls,
        *,
        old_source: LegacyPolicySourceV1,
        new_source: LegacyPolicySourceV1,
        old_config: PolicyConfigV1,
        new_config: PolicyConfigV1,
    ) -> "PolicyComparisonIdentityV1":
        old_source.revalidate()
        new_source.revalidate()
        old_config.require_complete()
        new_config.require_complete()
        return cls(
            ProviderIdentityV1(
                old_source.version, old_source.sha256, old_config.sha256
            ),
            ProviderIdentityV1(
                new_source.version, new_source.sha256, new_config.sha256
            ),
            old_config,
            new_config,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "old_provider": self.old_provider.to_dict(),
            "new_provider": self.new_provider.to_dict(),
            "old_config_sha256": self.old_config.sha256,
            "new_config_sha256": self.new_config.sha256,
            "state_machine_identity": STATE_MACHINE_VERSION,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    @property
    def config_pair_sha256(self) -> str:
        return _digest(
            {
                "old": self.old_config.sha256,
                "new": self.new_config.sha256,
            }
        )


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class PolicyMemoryV1:
    schema_version: str
    state_machine_version: str
    policy_identity: str
    policy_config_identity: str
    episode_id: str
    decision_step: int
    lifecycle: str
    phase: str
    instruction_label: str
    attack_pass: str | None
    preserve_for_move: bool
    acted_iteration: int
    ordered_unit_ids: tuple[int, ...]
    ordered_unit_snapshots: tuple[tuple[Any, ...], ...]
    current_unit_cursor: int
    ordered_target_ids: tuple[int, ...]
    current_target_cursor: int
    ordered_positions: tuple[tuple[int, int, int], ...]
    position_cursor: int
    remaining_capacities: tuple[tuple[str, int], ...]
    local_mana: int | None
    local_unit_counts: tuple[tuple[str, int], ...]
    camp: int
    rng_mode: str
    rng_state: None
    previous_transition_sha256: str | None

    def __new__(cls, *_args: Any, **_kwargs: Any):
        raise TypeError("PolicyMemoryV1 can only be issued by the explicit state machine")

    def __copy__(self) -> "PolicyMemoryV1":
        copied = object.__new__(PolicyMemoryV1)
        for name in self.__slots__:
            if name != "__weakref__":
                object.__setattr__(copied, name, getattr(self, name))
        return copied

    def __deepcopy__(self, _memo: dict[int, Any]) -> "PolicyMemoryV1":
        return self.__copy__()

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            field: (
                [list(item) if type(item) is tuple else item for item in value]
                if field in {
                    "ordered_positions",
                    "ordered_unit_snapshots",
                    "remaining_capacities",
                    "local_unit_counts",
                }
                else list(value)
                if type(value) is tuple
                else value
            )
            for field, value in (
                ("schema_version", self.schema_version),
                ("state_machine_version", self.state_machine_version),
                ("policy_identity", self.policy_identity),
                ("policy_config_identity", self.policy_config_identity),
                ("episode_id", self.episode_id),
                ("decision_step", self.decision_step),
                ("lifecycle", self.lifecycle),
                ("phase", self.phase),
                ("instruction_label", self.instruction_label),
                ("attack_pass", self.attack_pass),
                ("preserve_for_move", self.preserve_for_move),
                ("acted_iteration", self.acted_iteration),
                ("ordered_unit_ids", self.ordered_unit_ids),
                ("ordered_unit_snapshots", self.ordered_unit_snapshots),
                ("current_unit_cursor", self.current_unit_cursor),
                ("ordered_target_ids", self.ordered_target_ids),
                ("current_target_cursor", self.current_target_cursor),
                ("ordered_positions", self.ordered_positions),
                ("position_cursor", self.position_cursor),
                ("remaining_capacities", self.remaining_capacities),
                ("local_mana", self.local_mana),
                ("local_unit_counts", self.local_unit_counts),
                ("camp", self.camp),
                ("rng_mode", self.rng_mode),
                ("rng_state", self.rng_state),
                ("previous_transition_sha256", self.previous_transition_sha256),
            )
        }

    def to_dict(self) -> dict[str, Any]:
        _validate_issued_memory(self)
        return self._unsigned_dict()

    @property
    def sha256(self) -> str:
        _validate_issued_memory(self)
        return _digest(self._unsigned_dict())


_MEMORY_SNAPSHOTS: "weakref.WeakKeyDictionary[PolicyMemoryV1, str]" = (
    weakref.WeakKeyDictionary()
)


def _issue_memory(**values: Any) -> PolicyMemoryV1:
    memory = object.__new__(PolicyMemoryV1)
    for name, value in values.items():
        object.__setattr__(memory, name, value)
    _validate_memory_schema(memory)
    snapshot = _digest(memory._unsigned_dict())
    _MEMORY_SNAPSHOTS[memory] = snapshot
    return memory


def _validate_issued_memory(memory: PolicyMemoryV1) -> None:
    if type(memory) is not PolicyMemoryV1:
        raise TypeError("memory must be an issued PolicyMemoryV1")
    expected = _MEMORY_SNAPSHOTS.get(memory)
    try:
        _validate_memory_schema(memory)
        actual = _digest(memory._unsigned_dict())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("PolicyMemoryV1 is not an intact issued memory snapshot") from exc
    if expected is None or expected != actual:
        raise ValueError("PolicyMemoryV1 is not an intact issued memory snapshot")


def _validate_nonnegative_int(value: Any, label: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative strict integer")


def _validate_id_sequence(value: Any, label: str) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{label} must be an immutable tuple")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} must contain non-negative strict integer IDs")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicate IDs")


def _validate_memory_schema(memory: PolicyMemoryV1) -> None:
    if memory.schema_version != POLICY_MEMORY_SCHEMA_VERSION:
        raise ValueError("memory schema identity mismatch")
    if memory.state_machine_version != STATE_MACHINE_VERSION:
        raise ValueError("memory state-machine identity mismatch")
    _strict_sha256(memory.policy_identity, "memory policy identity")
    _strict_sha256(memory.policy_config_identity, "memory policy config identity")
    _strict_text(memory.episode_id, "memory episode ID")
    if type(memory.decision_step) is not int or memory.decision_step <= 0:
        raise ValueError("memory decision step must be a positive strict integer")
    if memory.lifecycle != "active":
        raise ValueError("memory lifecycle must be active")
    phases = {
        "turn_start",
        "opening",
        "opening_endround",
        "artifact",
        "attack_pre_move",
        "move",
        "attack_post_move",
        "summon",
        "endround",
    }
    if memory.phase not in phases:
        raise ValueError("memory phase is not defined by the state machine")
    _strict_text(memory.instruction_label, "memory instruction label")
    if memory.attack_pass not in {None, "pre_move", "post_move"}:
        raise ValueError("memory attack_pass is invalid")
    if type(memory.preserve_for_move) is not bool:
        raise ValueError("memory preserve_for_move must be a strict boolean")
    _validate_nonnegative_int(memory.acted_iteration, "acted_iteration")
    _validate_id_sequence(memory.ordered_unit_ids, "ordered_unit_ids")
    if type(memory.ordered_unit_snapshots) is not tuple:
        raise ValueError("ordered_unit_snapshots must be an immutable tuple")
    snapshot_ids: list[int] = []
    for snapshot in memory.ordered_unit_snapshots:
        if type(snapshot) is not tuple or not snapshot or type(snapshot[0]) is not int:
            raise ValueError("ordered_unit_snapshots contains an invalid unit snapshot")
        snapshot_ids.append(snapshot[0])
    if memory.ordered_unit_snapshots and tuple(snapshot_ids) != memory.ordered_unit_ids:
        raise ValueError("ordered_unit_snapshots disagree with ordered_unit_ids")
    _validate_nonnegative_int(memory.current_unit_cursor, "current_unit_cursor")
    if memory.current_unit_cursor > len(memory.ordered_unit_ids):
        raise ValueError("current_unit_cursor exceeds ordered_unit_ids")
    _validate_id_sequence(memory.ordered_target_ids, "ordered_target_ids")
    _validate_nonnegative_int(memory.current_target_cursor, "current_target_cursor")
    if memory.current_target_cursor > len(memory.ordered_target_ids):
        raise ValueError("current_target_cursor exceeds ordered_target_ids")
    if type(memory.ordered_positions) is not tuple:
        raise ValueError("ordered_positions must be an immutable tuple")
    for position in memory.ordered_positions:
        if (
            type(position) is not tuple
            or len(position) != 3
            or any(type(axis) is not int for axis in position)
            or sum(position) != 0
        ):
            raise ValueError("ordered_positions contains an invalid cube coordinate")
    _validate_nonnegative_int(memory.position_cursor, "position_cursor")
    if memory.position_cursor > len(memory.ordered_positions):
        raise ValueError("position_cursor exceeds ordered_positions")
    for field, entries in (
        ("remaining_capacities", memory.remaining_capacities),
        ("local_unit_counts", memory.local_unit_counts),
    ):
        if type(entries) is not tuple:
            raise ValueError(f"{field} must be an immutable tuple")
        names: list[str] = []
        for entry in entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError(f"{field} contains an invalid entry")
            name, count = entry
            _strict_text(name, f"{field} name")
            _validate_nonnegative_int(count, f"{field} count")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(f"{field} contains duplicate names")
    if memory.local_mana is not None:
        _validate_nonnegative_int(memory.local_mana, "local_mana")
    if type(memory.camp) is not int or memory.camp not in {0, 1}:
        raise ValueError("memory camp must be 0 or 1")
    if memory.rng_mode != "none" or memory.rng_state is not None:
        raise ValueError("deterministic provider cannot carry RNG state")
    if memory.previous_transition_sha256 is not None:
        _strict_sha256(
            memory.previous_transition_sha256, "previous transition identity"
        )


@dataclass(frozen=True, slots=True)
class SelectedCommandV1:
    command: Mapping[str, Any]
    next_phase: str
    instruction_label: str = "selected"
    attack_pass: str | None = None
    preserve_for_move: bool = False
    acted_increment: int = 0
    memory_updates: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _freeze_mapping(canonical_command(self.command)))
        _strict_text(self.next_phase, "next phase")
        _strict_text(self.instruction_label, "instruction label")
        updates = {} if self.memory_updates is None else dict(self.memory_updates)
        allowed = {
            "ordered_unit_ids",
            "ordered_unit_snapshots",
            "current_unit_cursor",
            "ordered_target_ids",
            "current_target_cursor",
            "ordered_positions",
            "position_cursor",
            "remaining_capacities",
            "local_mana",
            "local_unit_counts",
        }
        if set(updates) - allowed:
            raise ValueError("selected command contains unknown memory updates")
        object.__setattr__(self, "memory_updates", _freeze_mapping(updates))


@dataclass(frozen=True, slots=True, init=False, eq=False, weakref_slot=True)
class PolicyPairDecisionV1:
    schema_version: str
    policy_identity: str
    episode_id: str
    decision_step: int
    observation_id: str
    old_provider_identity: str
    new_provider_identity: str
    old_action_id: str
    new_action_id: str
    old_distribution: Mapping[str, float]
    new_distribution: Mapping[str, float]
    memory_before_sha256: str
    m_after: PolicyMemoryV1
    transition_sha256: str
    support_id: str

    def __new__(cls, *_args: Any, **_kwargs: Any):
        raise TypeError(
            "PolicyPairDecisionV1 can only be issued by the explicit state machine"
        )

    def __copy__(self) -> "PolicyPairDecisionV1":
        copied = object.__new__(PolicyPairDecisionV1)
        for name in self.__slots__:
            if name != "__weakref__":
                object.__setattr__(copied, name, getattr(self, name))
        return copied

    def __deepcopy__(self, _memo: dict[int, Any]) -> "PolicyPairDecisionV1":
        return self.__copy__()

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_identity": self.policy_identity,
            "episode_id": self.episode_id,
            "decision_step": self.decision_step,
            "observation_id": self.observation_id,
            "old_provider_identity": self.old_provider_identity,
            "new_provider_identity": self.new_provider_identity,
            "old_action_id": self.old_action_id,
            "new_action_id": self.new_action_id,
            "old_distribution": dict(self.old_distribution),
            "new_distribution": dict(self.new_distribution),
            "memory_before_sha256": self.memory_before_sha256,
            "m_after": self.m_after._unsigned_dict(),
            "transition_sha256": self.transition_sha256,
            "support_id": self.support_id,
        }

    def to_dict(self) -> dict[str, Any]:
        _validate_issued_decision(self)
        return self._unsigned_dict()

    @property
    def sha256(self) -> str:
        _validate_issued_decision(self)
        return _digest(self._unsigned_dict())


_DECISION_SNAPSHOTS: "weakref.WeakKeyDictionary[PolicyPairDecisionV1, str]" = (
    weakref.WeakKeyDictionary()
)


def _issue_decision(**values: Any) -> PolicyPairDecisionV1:
    decision = object.__new__(PolicyPairDecisionV1)
    for name, value in values.items():
        if name in {"old_distribution", "new_distribution"}:
            value = MappingProxyType(dict(value))
        object.__setattr__(decision, name, value)
    for distribution in (decision.old_distribution, decision.new_distribution):
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in distribution.values()
        ):
            raise ValueError("provider distribution must contain finite floats")
        if math.fsum(distribution.values()) != 1.0:
            raise ValueError("provider distribution must have exact unit mass")
    _validate_issued_memory(decision.m_after)
    _DECISION_SNAPSHOTS[decision] = _digest(decision._unsigned_dict())
    return decision


def _validate_issued_decision(decision: PolicyPairDecisionV1) -> None:
    if type(decision) is not PolicyPairDecisionV1:
        raise TypeError("decision must be an issued PolicyPairDecisionV1")
    expected = _DECISION_SNAPSHOTS.get(decision)
    try:
        _validate_issued_memory(decision.m_after)
        actual = _digest(decision._unsigned_dict())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("PolicyPairDecisionV1 is not an intact issued decision") from exc
    if expected is None or expected != actual:
        raise ValueError("PolicyPairDecisionV1 is not an intact issued decision")


class _SelectedLegacyCommand(RuntimeError):
    def __init__(self, command: Mapping[str, Any]) -> None:
        super().__init__("legacy policy selected one atomic command")
        self.command = canonical_command(command)


class _LegacyIfElseRuntime:
    """Read-only loader queried through fresh objects with no retained frame."""

    def __init__(self, source: LegacyPolicySourceV1) -> None:
        source.revalidate()
        self.source = source
        self.module = self._load_module(source)
        if not hasattr(self.module, "IfElseAI"):
            raise PolicySourceError("legacy main.py does not define IfElseAI")

    @staticmethod
    def _load_module(source: LegacyPolicySourceV1) -> Any:
        root = source.root
        module_name = f"_agentbench_miracle_ifelse_{source.sha256[:20]}"
        saved_modules = {
            name: sys.modules.get(name)
            for name in ("calculator", "gameunit", "ai_client", "card")
        }
        original_cwd = Path.cwd()
        original_path = list(sys.path)
        original_dont_write_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            for name in saved_modules:
                sys.modules.pop(name, None)
            os.chdir(root)
            sys.path.insert(0, str(root))
            spec = importlib.util.spec_from_file_location(module_name, root / "main.py")
            if spec is None or spec.loader is None:
                raise PolicySourceError("cannot load legacy main.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        finally:
            sys.dont_write_bytecode = original_dont_write_bytecode
            os.chdir(original_cwd)
            sys.path[:] = original_path
            for name, saved in saved_modules.items():
                sys.modules.pop(name, None)
                if saved is not None:
                    sys.modules[name] = saved

    def _new_policy(
        self, observation: Mapping[str, Any], config: PolicyConfigV1
    ) -> Any:
        config.require_complete()
        module = self.module
        base = module.IfElseAI

        class CapturingIfElseAI(base):
            def _emit(self, operation_type: str, **parameters: Any) -> None:
                raise _SelectedLegacyCommand(
                    {
                        "player": self.my_camp,
                        "round": self.round,
                        "operation_type": operation_type,
                        "operation_parameters": parameters,
                    }
                )

            def attack(self, attacker: int, target: int) -> None:
                self._emit("attack", attacker=attacker, target=target)

            def move(self, mover: int, position: Sequence[int]) -> None:
                self._emit("move", mover=mover, position=list(position))

            def summon(
                self, unit_type: str, level: int, position: Sequence[int]
            ) -> None:
                self._emit(
                    "summon", type=unit_type, level=level, position=list(position)
                )

            def use(self, artifact: int, target: Any) -> None:
                normalized = list(target) if type(target) is tuple else target
                self._emit("use", card=artifact, target=normalized)

            def end_round(self) -> None:
                self._emit("endround")

        policy = object.__new__(CapturingIfElseAI)
        camp = observation.get("camp")
        if type(camp) is not int or camp not in {0, 1}:
            raise ValueError("policy observation camp must be 0 or 1")
        policy.my_camp = camp
        policy.round = observation.get("round", 0)
        if type(policy.round) is not int:
            raise ValueError("policy observation round must be an integer")
        policy.map = module.gameunit.Map()
        policy.players = [module.gameunit.Player(0), module.gameunit.Player(1)]
        if "map" in observation or "players" in observation:
            if set(observation) < {"camp", "round", "map", "players"}:
                raise ValueError("policy observation is incomplete")
            policy.map.update(observation["map"])
            policy.players = [
                module.gameunit.Player(0, observation["players"][0]),
                module.gameunit.Player(1, observation["players"][1]),
            ]
        policy.my_miracle = module.MY_MIRACLES[camp]
        policy.enemy_miracle = module.MY_MIRACLES[camp ^ 1]
        for spec in MIRACLE_CONFIG_SPECS:
            if spec.attribute is not None:
                setattr(policy, spec.attribute, config.values[spec.name])
        policy.artifacts = [config.values["MIRACLE_ARTIFACT"]]
        policy.creatures = list(config.values["MIRACLE_DECK"])
        return policy

    @staticmethod
    def _direct_command(
        observation: Mapping[str, Any], operation_type: str
    ) -> Mapping[str, Any]:
        return canonical_command(
            {
                "player": observation["camp"],
                "round": observation.get("round", 0),
                "operation_type": operation_type,
                "operation_parameters": {},
            }
        )

    def select(
        self,
        observation: Mapping[str, Any],
        memory: PolicyMemoryV1,
        config: PolicyConfigV1,
    ) -> SelectedCommandV1:
        self.source.revalidate()
        if set(observation) == {"camp"}:
            command = canonical_command(
                {
                    "player": observation["camp"],
                    "round": 0,
                    "operation_type": "init",
                    "operation_parameters": {
                        "artifacts": [config.values["MIRACLE_ARTIFACT"]],
                        "creatures": list(config.values["MIRACLE_DECK"]),
                    },
                }
            )
            return SelectedCommandV1(command, "turn_start", "init")

        policy = self._new_policy(observation, config)
        policy.refresh_static_positions()
        phase = memory.phase
        phase_memory_updates: dict[str, Any] = {}
        for _advance in range(8):
            try:
                if phase == "turn_start":
                    phase = "opening" if policy.round in {0, 1} else "artifact"
                    continue
                if phase == "opening":
                    policy.play()
                    raise RuntimeError("legacy opening did not emit an action")
                if phase == "opening_endround":
                    return SelectedCommandV1(
                        self._direct_command(observation, "endround"),
                        "turn_start",
                        "opening.endround",
                    )
                if phase == "artifact":
                    policy.use_artifact()
                    phase = "attack_pre_move"
                    continue
                if phase == "attack_pre_move":
                    policy.attack_phase(preserve_for_move=True)
                    phase = "move"
                    continue
                if phase == "move":
                    snapshots = memory.ordered_unit_snapshots
                    cursor = memory.current_unit_cursor
                    if not snapshots:
                        ordered = sorted(
                            policy.allies(),
                            key=lambda unit: (unit.type != "Swordsman", unit.id),
                        )
                        raw_by_id = {
                            raw[0]: raw
                            for raw in observation["map"]["units"]
                            if raw[1] == policy.my_camp
                        }
                        snapshots = tuple(
                            tuple(raw_by_id[unit.id]) for unit in ordered
                        )
                        cursor = 0
                    while cursor < len(snapshots):
                        unit = self.module.gameunit.Unit(list(snapshots[cursor]))
                        cursor += 1
                        if not unit.can_move:
                            continue
                        position = policy.best_move_for(unit)
                        if position and position != unit.pos:
                            command = canonical_command(
                                {
                                    "player": policy.my_camp,
                                    "round": policy.round,
                                    "operation_type": "move",
                                    "operation_parameters": {
                                        "mover": unit.id,
                                        "position": list(position),
                                    },
                                }
                            )
                            return SelectedCommandV1(
                                command,
                                "move",
                                "move.move",
                                memory_updates={
                                    "ordered_unit_ids": tuple(
                                        snapshot[0] for snapshot in snapshots
                                    ),
                                    "ordered_unit_snapshots": snapshots,
                                    "current_unit_cursor": cursor,
                                },
                            )
                    phase_memory_updates = {
                        "ordered_unit_ids": (),
                        "ordered_unit_snapshots": (),
                        "current_unit_cursor": 0,
                    }
                    phase = "attack_post_move"
                    continue
                if phase == "attack_post_move":
                    policy.attack_phase()
                    phase = "summon"
                    continue
                if phase == "summon":
                    policy.summon_phase()
                    phase = "endround"
                    continue
                if phase == "endround":
                    return SelectedCommandV1(
                        self._direct_command(observation, "endround"),
                        "turn_start",
                        "play.endround",
                        memory_updates=phase_memory_updates,
                    )
                raise ValueError(f"unknown policy memory phase: {phase}")
            except _SelectedLegacyCommand as selected:
                operation = selected.command["operation_type"]
                if phase == "opening":
                    next_phase = (
                        "opening_endround"
                        if operation == "summon"
                        else "turn_start"
                    )
                elif phase == "artifact":
                    next_phase = "attack_pre_move"
                else:
                    next_phase = phase
                attack_pass = (
                    "pre_move"
                    if phase == "attack_pre_move"
                    else "post_move"
                    if phase == "attack_post_move"
                    else None
                )
                return SelectedCommandV1(
                    selected.command,
                    next_phase,
                    f"{phase}.{operation}",
                    attack_pass=attack_pass,
                    preserve_for_move=phase == "attack_pre_move",
                    acted_increment=int(operation == "attack"),
                    memory_updates=phase_memory_updates,
                )
        raise RuntimeError("policy phase advancement did not produce an action")


_RUNTIMES: dict[str, _LegacyIfElseRuntime] = {}


def _select_legacy_command(
    provider_identity: str,
    observation: Mapping[str, Any],
    memory: PolicyMemoryV1,
    config: PolicyConfigV1,
) -> SelectedCommandV1:
    runtime = _RUNTIMES.get(provider_identity)
    if runtime is None:
        raise IncompletePolicyEvidenceError(
            f"no verified legacy runtime is registered for provider {provider_identity}"
        )
    return runtime.select(observation, memory, config)


class ExplicitIfElseStateMachineV1:
    """Issuer for one comparison's explicit, replayable policy continuation."""

    def __init__(self, identity: PolicyComparisonIdentityV1) -> None:
        if not isinstance(identity, PolicyComparisonIdentityV1):
            raise TypeError("identity must be a PolicyComparisonIdentityV1")
        self.identity = identity
        self._episode_tips: dict[str, PolicyMemoryV1] = {}

    @classmethod
    def from_sources(
        cls,
        *,
        old_source: LegacyPolicySourceV1,
        new_source: LegacyPolicySourceV1,
        old_config: PolicyConfigV1,
        new_config: PolicyConfigV1,
    ) -> "ExplicitIfElseStateMachineV1":
        identity = PolicyComparisonIdentityV1.from_sources(
            old_source=old_source,
            new_source=new_source,
            old_config=old_config,
            new_config=new_config,
        )
        machine = cls(identity)
        _RUNTIMES[identity.old_provider.sha256] = _LegacyIfElseRuntime(old_source)
        _RUNTIMES[identity.new_provider.sha256] = _LegacyIfElseRuntime(new_source)
        return machine

    def reset(self, *, camp: int, episode_id: str) -> PolicyMemoryV1:
        self.identity.old_config.require_complete()
        self.identity.new_config.require_complete()
        if type(camp) is not int or camp not in {0, 1}:
            raise ValueError("camp must be 0 or 1")
        _strict_text(episode_id, "episode ID")
        memory = _issue_memory(
            schema_version=POLICY_MEMORY_SCHEMA_VERSION,
            state_machine_version=STATE_MACHINE_VERSION,
            policy_identity=self.identity.sha256,
            policy_config_identity=self.identity.config_pair_sha256,
            episode_id=episode_id,
            decision_step=1,
            lifecycle="active",
            phase="turn_start",
            instruction_label="reset",
            attack_pass=None,
            preserve_for_move=False,
            acted_iteration=0,
            ordered_unit_ids=(),
            ordered_unit_snapshots=(),
            current_unit_cursor=0,
            ordered_target_ids=(),
            current_target_cursor=0,
            ordered_positions=(),
            position_cursor=0,
            remaining_capacities=(),
            local_mana=None,
            local_unit_counts=(),
            camp=camp,
            rng_mode="none",
            rng_state=None,
            previous_transition_sha256=None,
        )
        self._episode_tips[episode_id] = memory
        return memory

    def validate_memory(
        self, memory: PolicyMemoryV1, *, episode_id: str | None = None
    ) -> None:
        _validate_issued_memory(memory)
        if memory.schema_version != POLICY_MEMORY_SCHEMA_VERSION:
            raise ValueError("memory schema identity mismatch")
        if memory.state_machine_version != STATE_MACHINE_VERSION:
            raise ValueError("memory state-machine identity mismatch")
        if memory.policy_identity != self.identity.sha256:
            raise ValueError("memory policy identity mismatch")
        if memory.policy_config_identity != self.identity.config_pair_sha256:
            raise ValueError("memory policy config identity mismatch")
        if episode_id is not None and memory.episode_id != episode_id:
            raise ValueError("memory episode identity mismatch")
        if type(memory.decision_step) is not int or memory.decision_step <= 0:
            raise ValueError("memory decision step is invalid")
        if memory.rng_mode != "none" or memory.rng_state is not None:
            raise ValueError("deterministic provider cannot carry RNG state")

    def _validate_current_tip(self, memory: PolicyMemoryV1) -> None:
        current = self._episode_tips.get(memory.episode_id)
        if current is not memory:
            raise ValueError(
                "memory is not the current transition tip for its episode"
            )

    @staticmethod
    def _validate_support(
        observation: Mapping[str, Any], support: ActionSupport
    ) -> None:
        if not isinstance(support, ActionSupport):
            raise TypeError("support must be an ActionSupport")
        trusted = build_trusted_action_support(dict(observation))
        if (
            support.schema_version != trusted.schema_version
            or support.support_id != trusted.support_id
            or support.action_ids != trusted.action_ids
            or tuple(candidate.action for candidate in support.actions)
            != tuple(candidate.action for candidate in trusted.actions)
        ):
            raise ValueError("supplied support does not match trusted complete support")

    def evaluate_pair(
        self,
        observation: Mapping[str, Any],
        memory: PolicyMemoryV1,
        complete_action_support: ActionSupport,
    ) -> PolicyPairDecisionV1:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        self.validate_memory(memory)
        self._validate_current_tip(memory)
        if observation.get("camp") != memory.camp:
            raise ValueError("observation camp disagrees with policy memory")
        self._validate_support(observation, complete_action_support)

        old_selected = _select_legacy_command(
            self.identity.old_provider.sha256,
            dict(observation),
            memory,
            self.identity.old_config,
        )
        new_selected = _select_legacy_command(
            self.identity.new_provider.sha256,
            dict(observation),
            memory,
            self.identity.new_config,
        )
        old_action_id = command_action_id(old_selected.command)
        new_action_id = command_action_id(new_selected.command)
        old_distribution = one_hot(old_action_id, complete_action_support)
        new_distribution = one_hot(new_action_id, complete_action_support)

        transition_payload = {
            "schema_version": POLICY_MEMORY_SCHEMA_VERSION,
            "state_machine_version": STATE_MACHINE_VERSION,
            "policy_identity": self.identity.sha256,
            "episode_id": memory.episode_id,
            "decision_step": memory.decision_step,
            "memory_before_sha256": memory.sha256,
            "observation_id": canonical_state_id(observation),
            "support_id": complete_action_support.support_id,
            "old_action_id": old_action_id,
            "new_action_id": new_action_id,
            "next_phase": new_selected.next_phase,
            "instruction_label": new_selected.instruction_label,
        }
        transition_sha256 = _digest(transition_payload)
        updates = new_selected.memory_updates
        after = _issue_memory(
            schema_version=POLICY_MEMORY_SCHEMA_VERSION,
            state_machine_version=STATE_MACHINE_VERSION,
            policy_identity=self.identity.sha256,
            policy_config_identity=self.identity.config_pair_sha256,
            episode_id=memory.episode_id,
            decision_step=memory.decision_step + 1,
            lifecycle="active",
            phase=new_selected.next_phase,
            instruction_label=new_selected.instruction_label,
            attack_pass=new_selected.attack_pass,
            preserve_for_move=new_selected.preserve_for_move,
            acted_iteration=memory.acted_iteration + new_selected.acted_increment,
            ordered_unit_ids=tuple(
                updates.get("ordered_unit_ids", memory.ordered_unit_ids)
            ),
            ordered_unit_snapshots=tuple(
                _freeze_sequence(item)
                for item in updates.get(
                    "ordered_unit_snapshots", memory.ordered_unit_snapshots
                )
            ),
            current_unit_cursor=updates.get(
                "current_unit_cursor", memory.current_unit_cursor
            ),
            ordered_target_ids=tuple(
                updates.get("ordered_target_ids", memory.ordered_target_ids)
            ),
            current_target_cursor=updates.get(
                "current_target_cursor", memory.current_target_cursor
            ),
            ordered_positions=tuple(
                tuple(item)
                for item in updates.get(
                    "ordered_positions", memory.ordered_positions
                )
            ),
            position_cursor=updates.get("position_cursor", memory.position_cursor),
            remaining_capacities=tuple(
                tuple(item)
                for item in updates.get(
                    "remaining_capacities", memory.remaining_capacities
                )
            ),
            local_mana=updates.get("local_mana", memory.local_mana),
            local_unit_counts=tuple(
                tuple(item)
                for item in updates.get(
                    "local_unit_counts", memory.local_unit_counts
                )
            ),
            camp=memory.camp,
            rng_mode="none",
            rng_state=None,
            previous_transition_sha256=transition_sha256,
        )
        decision = _issue_decision(
            schema_version=POLICY_PAIR_DECISION_SCHEMA_VERSION,
            policy_identity=self.identity.sha256,
            episode_id=memory.episode_id,
            decision_step=memory.decision_step,
            observation_id=transition_payload["observation_id"],
            old_provider_identity=self.identity.old_provider.sha256,
            new_provider_identity=self.identity.new_provider.sha256,
            old_action_id=old_action_id,
            new_action_id=new_action_id,
            old_distribution=old_distribution,
            new_distribution=new_distribution,
            memory_before_sha256=memory.sha256,
            m_after=after,
            transition_sha256=transition_sha256,
            support_id=complete_action_support.support_id,
        )
        self.validate_decision(
            decision,
            episode_id=memory.episode_id,
            decision_step=memory.decision_step,
            observation=observation,
            complete_action_support=complete_action_support,
        )
        self._episode_tips[memory.episode_id] = after
        return decision

    def validate_decision(
        self,
        decision: PolicyPairDecisionV1,
        *,
        episode_id: str,
        decision_step: int,
        observation: Mapping[str, Any],
        complete_action_support: ActionSupport,
    ) -> None:
        _validate_issued_decision(decision)
        _strict_text(episode_id, "episode ID")
        if type(decision_step) is not int or decision_step <= 0:
            raise ValueError("decision step must be a positive integer")
        if decision.schema_version != POLICY_PAIR_DECISION_SCHEMA_VERSION:
            raise ValueError("decision schema identity mismatch")
        if decision.policy_identity != self.identity.sha256:
            raise ValueError("decision policy identity mismatch")
        if decision.episode_id != episode_id:
            raise ValueError("decision episode identity mismatch")
        if decision.decision_step != decision_step:
            raise ValueError("decision step identity mismatch")
        observation_id = canonical_state_id(observation)
        if decision.observation_id != observation_id:
            raise ValueError("decision observation identity mismatch")
        self._validate_support(observation, complete_action_support)
        if decision.support_id != complete_action_support.support_id:
            raise ValueError("decision support identity mismatch")
        if decision.old_provider_identity != self.identity.old_provider.sha256:
            raise ValueError("old provider identity mismatch")
        if decision.new_provider_identity != self.identity.new_provider.sha256:
            raise ValueError("new provider identity mismatch")
        expected_actions = set(complete_action_support.action_ids)
        for action_id, distribution, label in (
            (decision.old_action_id, decision.old_distribution, "old"),
            (decision.new_action_id, decision.new_distribution, "new"),
        ):
            if set(distribution) != expected_actions:
                raise ValueError(f"{label} distribution support mismatch")
            if action_id not in expected_actions or distribution[action_id] != 1.0:
                raise ValueError(f"{label} chosen action is not the one-hot mass")
        after = decision.m_after
        self.validate_memory(after, episode_id=episode_id)
        if after.decision_step != decision_step + 1:
            raise ValueError("decision output memory step mismatch")
        if after.previous_transition_sha256 != decision.transition_sha256:
            raise ValueError("decision transition chain mismatch")
        transition_payload = {
            "schema_version": POLICY_MEMORY_SCHEMA_VERSION,
            "state_machine_version": STATE_MACHINE_VERSION,
            "policy_identity": self.identity.sha256,
            "episode_id": episode_id,
            "decision_step": decision_step,
            "memory_before_sha256": decision.memory_before_sha256,
            "observation_id": observation_id,
            "support_id": complete_action_support.support_id,
            "old_action_id": decision.old_action_id,
            "new_action_id": decision.new_action_id,
            "next_phase": after.phase,
            "instruction_label": after.instruction_label,
        }
        if _digest(transition_payload) != decision.transition_sha256:
            raise ValueError("decision transition evidence mismatch")


@dataclass(frozen=True, slots=True)
class SequentialReplayAuditV1:
    trace_sha256: str
    episode_id: str
    evaluated_camp: int
    decision_frames: int
    chosen_reproduced: int
    action_support_complete: int
    chosen_in_support: int
    first_mismatch_step: int | None
    final_memory_sha256: str | None
    old_new_same_context: int
    memory_before: tuple[PolicyMemoryV1, ...]
    decisions: tuple[PolicyPairDecisionV1, ...]
    recorded_action_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            self.decision_frames > 0
            and self.chosen_reproduced == self.decision_frames
            and self.action_support_complete == self.decision_frames
            and self.chosen_in_support == self.decision_frames
            and self.old_new_same_context == self.decision_frames
            and self.first_mismatch_step is None
            and len(self.memory_before) == self.decision_frames
            and len(self.decisions) == self.decision_frames
            and len(self.recorded_action_ids) == self.decision_frames
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "24-miracle-ifelse-sequential-replay-audit-v1",
            "trace_sha256": self.trace_sha256,
            "episode_id": self.episode_id,
            "evaluated_camp": self.evaluated_camp,
            "decision_frames": self.decision_frames,
            "chosen_reproduced": self.chosen_reproduced,
            "action_support_complete": self.action_support_complete,
            "chosen_in_support": self.chosen_in_support,
            "old_new_same_context": self.old_new_same_context,
            "first_mismatch_step": self.first_mismatch_step,
            "final_memory_sha256": self.final_memory_sha256,
            "decision_records": [
                {
                    "decision_step": step,
                    "recorded_action_id": recorded_action_id,
                    "m_before": memory.to_dict(),
                    "pair_decision": decision.to_dict(),
                }
                for step, (recorded_action_id, memory, decision) in enumerate(
                    zip(
                        self.recorded_action_ids,
                        self.memory_before,
                        self.decisions,
                        strict=True,
                    ),
                    start=1,
                )
            ],
            "complete": self.complete,
        }


def _trace_decisions(
    payload: bytes, *, evaluated_camp: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("trace must be strict UTF-8") from exc
    pending: dict[str, Any] | None = None
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trace line {line_number} is invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"trace line {line_number} must be an object")
        if record.get("kind") == "judge_frame":
            frame = record.get("frame")
            if not isinstance(frame, dict):
                continue
            if (
                evaluated_camp in frame.get("listen", [])
                and evaluated_camp in frame.get("player", [])
                and isinstance(frame.get("content"), list)
                and frame["content"]
            ):
                framed = frame["content"][0]
                if type(framed) is not str or len(framed) < 6:
                    raise ValueError("evaluated Judge observation framing is invalid")
                try:
                    declared = int(framed[:6])
                except ValueError as exc:
                    raise ValueError("evaluated Judge observation length is invalid") from exc
                encoded = framed[6:].encode("utf-8")
                if declared != len(encoded):
                    raise ValueError("evaluated Judge observation length mismatch")
                observation = json.loads(encoded)
                if not isinstance(observation, dict):
                    raise ValueError("evaluated Judge observation must be an object")
                pending = observation
        elif record.get("kind") == "ai_operation" and record.get("player") == evaluated_camp:
            if pending is None:
                raise ValueError("evaluated operation has no preceding observation")
            operation = record.get("operation")
            if not isinstance(operation, dict):
                raise ValueError("evaluated operation must be an object")
            pairs.append((pending, canonical_command(operation)))
            pending = None
    return pairs


def audit_sequential_trace(
    trace_path: str | os.PathLike[str],
    *,
    machine: ExplicitIfElseStateMachineV1,
    evaluated_camp: int,
    episode_id: str,
) -> SequentialReplayAuditV1:
    """Replay one immutable trace from reset without per-frame reinitialization."""

    supplied_path = Path(trace_path)
    if supplied_path.is_symlink():
        raise ValueError("trace path must not be a symlink")
    path = supplied_path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise ValueError("trace path must be a real file")
    if type(evaluated_camp) is not int or evaluated_camp not in {0, 1}:
        raise ValueError("evaluated camp must be 0 or 1")
    lexical_before = path.lstat()
    if not stat.S_ISREG(lexical_before.st_mode):
        raise ValueError("trace path must remain a regular file")
    with path.open("rb") as stream:
        opened_before = os.fstat(stream.fileno())
        payload = stream.read()
        opened_after = os.fstat(stream.fileno())
    identity_before = (
        opened_before.st_dev,
        opened_before.st_ino,
        opened_before.st_ctime_ns,
    )
    if identity_before != (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_ctime_ns,
    ):
        raise ValueError("trace changed while its frozen snapshot was read")
    size_before = opened_before.st_size
    if size_before <= 0 or size_before > MAX_REPLAY_TRACE_BYTES:
        raise ValueError("trace size is outside the approved bound")
    if len(payload) != size_before:
        raise ValueError("trace size changed while its frozen snapshot was read")
    digest = hashlib.sha256(payload).hexdigest()
    pairs = _trace_decisions(payload, evaluated_camp=evaluated_camp)
    memory = machine.reset(camp=evaluated_camp, episode_id=episode_id)
    reproduced = 0
    complete_support = 0
    chosen_in_support = 0
    same_context = 0
    first_mismatch: int | None = None
    memory_records: list[PolicyMemoryV1] = []
    decisions: list[PolicyPairDecisionV1] = []
    recorded_action_ids: list[str] = []
    for step, (observation, recorded) in enumerate(pairs, start=1):
        if memory.decision_step != step:
            raise ValueError("sequential memory decision step is discontinuous")
        try:
            support = build_trusted_action_support(observation)
        except Exception:
            if first_mismatch is None:
                first_mismatch = step
            break
        complete_support += 1
        recorded_id = command_action_id(recorded)
        if recorded_id in support.action_ids:
            chosen_in_support += 1
        decision = machine.evaluate_pair(observation, memory, support)
        memory_records.append(memory)
        decisions.append(decision)
        recorded_action_ids.append(recorded_id)
        if decision.memory_before_sha256 == memory.sha256:
            same_context += 1
        if decision.new_action_id == recorded_id:
            reproduced += 1
        elif first_mismatch is None:
            first_mismatch = step
            break
        memory = decision.m_after
    if path.is_symlink():
        raise ValueError("trace changed during sequential replay audit")
    lexical_after = path.lstat()
    with path.open("rb") as stream:
        reopened = os.fstat(stream.fileno())
        final_payload = stream.read()
    identity_after = (reopened.st_dev, reopened.st_ino, reopened.st_ctime_ns)
    if (
        not stat.S_ISREG(lexical_after.st_mode)
        or identity_after != identity_before
        or reopened.st_size != size_before
        or hashlib.sha256(final_payload).hexdigest() != digest
    ):
        raise ValueError("trace changed during sequential replay audit")
    return SequentialReplayAuditV1(
        trace_sha256=digest,
        episode_id=episode_id,
        evaluated_camp=evaluated_camp,
        decision_frames=len(pairs),
        chosen_reproduced=reproduced,
        action_support_complete=complete_support,
        chosen_in_support=chosen_in_support,
        first_mismatch_step=first_mismatch,
        final_memory_sha256=memory.sha256 if reproduced else None,
        old_new_same_context=same_context,
        memory_before=tuple(memory_records),
        decisions=tuple(decisions),
        recorded_action_ids=tuple(recorded_action_ids),
    )


__all__ = [
    "IncompletePolicyEvidenceError",
    "LEGACY_SOURCE_FILES",
    "LegacyPolicySourceV1",
    "MIRACLE_CONFIG_SPECS",
    "MiracleConfigSpec",
    "POLICY_COMPARISON_SCHEMA_VERSION",
    "POLICY_CONFIG_SCHEMA_VERSION",
    "POLICY_MEMORY_SCHEMA_VERSION",
    "POLICY_PAIR_DECISION_SCHEMA_VERSION",
    "PROVIDER_SCHEMA_VERSION",
    "PolicyComparisonIdentityV1",
    "PolicyConfigV1",
    "PolicyMemoryV1",
    "PolicyPairDecisionV1",
    "ProviderIdentityV1",
    "PolicySourceError",
    "SequentialReplayAuditV1",
    "STATE_MACHINE_VERSION",
    "VALID_ARTIFACTS",
    "VALID_CREATURES",
    "SelectedCommandV1",
    "ExplicitIfElseStateMachineV1",
    "audit_sequential_trace",
]
