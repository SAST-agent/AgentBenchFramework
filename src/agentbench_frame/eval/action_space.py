"""Game-adapter boundary for exact canonical action supports."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


CanonicalMacro = tuple[tuple[int, ...], ...]


@runtime_checkable
class CanonicalActionSpace(Protocol):
    """A canonical action support that can be validated and counted exactly."""

    spec_id: str

    def canonicalize(
        self,
        state: Mapping[str, Any],
        action: Sequence[Sequence[int]],
    ) -> CanonicalMacro: ...

    def contains(
        self,
        state: Mapping[str, Any],
        action: Sequence[Sequence[int]],
    ) -> bool: ...

    def cardinality(self, state: Mapping[str, Any]) -> int: ...


@runtime_checkable
class EnumerableCanonicalActionSpace(CanonicalActionSpace, Protocol):
    """Optional refinement for supports small enough to materialize."""

    def iter_actions(
        self,
        state: Mapping[str, Any],
    ) -> Iterable[CanonicalMacro]: ...
