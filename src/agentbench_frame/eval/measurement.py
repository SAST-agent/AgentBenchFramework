"""Canonical state identifiers and adapter protocols for measurements."""

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable


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
