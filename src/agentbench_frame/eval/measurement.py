"""Canonical state identifiers and adapter protocols for measurements."""

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable


def _canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical_value(item) for item in value), key=repr)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("state cannot contain non-finite float values")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _canonical_value(value.to_dict())
    return repr(value)


def canonical_state_payload(value: Any) -> str:
    """Serialize a visible state deterministically for hashing and storage."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_state_id(value: Any) -> str:
    """Return a stable content ID for an adapter-provided visible state."""
    return hashlib.sha256(canonical_state_payload(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActionCandidate:
    """One runtime action with a stable identity in a versioned schema."""

    action_id: str
    action: Any

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("action_id must be a non-empty string")


@dataclass(frozen=True)
class ActionSupport:
    """Complete ordered legal support supplied by a game runtime."""

    actions: Sequence[ActionCandidate]
    schema_version: str

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("action support cannot be empty")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        action_ids = [candidate.action_id for candidate in actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action support contains duplicate action IDs")
        object.__setattr__(self, "actions", actions)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(candidate.action_id for candidate in self.actions)

    @property
    def support_id(self) -> str:
        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "action_ids": self.action_ids,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def resolve(self, action_id: str) -> Any:
        for candidate in self.actions:
            if candidate.action_id == action_id:
                return candidate.action
        raise ValueError(f"selected action ID is not legal: {action_id!r}")


@dataclass(frozen=True)
class PolicyDecision:
    """One coherent active-policy choice and its distribution."""

    action_id: str
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError("action_id must be a non-empty string")
        if not isinstance(self.probabilities, Mapping):
            raise TypeError("probabilities must be an action-ID mapping")


@runtime_checkable
class PolicyDistributionProvider(Protocol):
    """Optional game-agent hook for a distribution on a supplied legal support."""

    def get_action_distribution(
        self, observation: Any, legal_actions: Sequence[Any]
    ) -> Sequence[float]:
        ...


@runtime_checkable
class StateIdProvider(Protocol):
    """Optional environment hook for canonical visible state IDs."""

    def canonical_state_id(self, observation: Any) -> str:
        ...
