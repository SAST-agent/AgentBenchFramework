"""Bounded, explicit semantic checkpoint for reproducible HL cycles."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional


_SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_REDUCER_FIELDS = {
    "stable_knowledge",
    "failed_hypotheses",
    "open_questions",
    "recent_comparisons",
}


@dataclasses.dataclass(frozen=True)
class ResearchState:
    schema_version: str = "1.0"
    proposal_cycle: int = 0
    search_parent_version_id: Optional[str] = None
    official_champion_version_id: Optional[str] = None
    active_target: Optional[str] = None
    locked_opponents: tuple[str, ...] = ()
    stable_knowledge: tuple[str, ...] = ()
    failed_hypotheses: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    recent_comparisons: tuple[dict[str, Any], ...] = ()
    exploration_debt: int = 0
    max_bytes: int = dataclasses.field(default=16384, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported research state schema")
        if self.proposal_cycle < 0:
            raise ValueError("proposal_cycle must be non-negative")
        if self.exploration_debt < 0:
            raise ValueError("exploration_debt must be non-negative")
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        for field in (
            "locked_opponents",
            "stable_knowledge",
            "failed_hypotheses",
            "open_questions",
        ):
            object.__setattr__(
                self,
                field,
                tuple(str(value) for value in getattr(self, field)),
            )
        object.__setattr__(
            self,
            "recent_comparisons",
            tuple(dict(value) for value in self.recent_comparisons),
        )
        self._validate()

    @classmethod
    def empty(cls, *, max_bytes: int = 16384) -> "ResearchState":
        return cls(max_bytes=max_bytes)

    @classmethod
    def load_or_create(
        cls,
        path: str | Path,
        *,
        max_bytes: int,
    ) -> "ResearchState":
        source = Path(path)
        if not source.is_file():
            return cls.empty(max_bytes=max_bytes)
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("research state must be an object")
        allowed = {
            field.name for field in dataclasses.fields(cls) if field.name != "max_bytes"
        }
        unknown = sorted(set(value) - allowed)
        missing = sorted(allowed - set(value))
        if unknown or missing:
            raise ValueError(
                f"invalid research state fields: missing={missing}, unknown={unknown}"
            )
        return cls(**dict(value), max_bytes=max_bytes)

    def advance(self, **changes: Any) -> "ResearchState":
        unknown = sorted(
            set(changes)
            - {field.name for field in dataclasses.fields(self) if field.name != "max_bytes"}
        )
        if unknown:
            raise ValueError(f"unknown research state fields: {unknown}")
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_cycle": self.proposal_cycle,
            "search_parent_version_id": self.search_parent_version_id,
            "official_champion_version_id": self.official_champion_version_id,
            "active_target": self.active_target,
            "locked_opponents": list(self.locked_opponents),
            "stable_knowledge": list(self.stable_knowledge),
            "failed_hypotheses": list(self.failed_hypotheses),
            "open_questions": list(self.open_questions),
            "recent_comparisons": [dict(value) for value in self.recent_comparisons],
            "exploration_debt": self.exploration_debt,
        }

    def write(self, path: str | Path) -> Path:
        self._validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._canonical_bytes())
        return destination

    def _canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")

    def _validate(self) -> None:
        payload = self._canonical_bytes()
        text = payload.decode("utf-8")
        if _SECRET.search(text):
            raise ValueError("research state cannot contain credential material")
        if len(payload) > self.max_bytes:
            raise ValueError(
                f"research state exceeds max_bytes={self.max_bytes}: {len(payload)}"
            )


def apply_reducer_update(
    current: ResearchState,
    update_path: str | Path,
    *,
    proposal_cycle: int,
    search_parent_version_id: str,
    official_champion_version_id: Optional[str],
    exploration_debt: int,
) -> ResearchState:
    """Apply model-authored findings while keeping factual pointers framework-owned."""

    source = Path(update_path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("reducer update must be an object")
    if set(value) != _REDUCER_FIELDS:
        raise ValueError(
            "invalid reducer update fields: "
            f"expected={sorted(_REDUCER_FIELDS)}, actual={sorted(value)}"
        )
    text_fields: dict[str, tuple[str, ...]] = {}
    for field in ("stable_knowledge", "failed_hypotheses", "open_questions"):
        raw = value[field]
        if not isinstance(raw, list) or not all(
            isinstance(item, str) and item.strip() for item in raw
        ):
            raise ValueError(f"reducer update {field} must be non-empty strings")
        text_fields[field] = tuple(item.strip() for item in raw)
    comparisons = value["recent_comparisons"]
    if not isinstance(comparisons, list) or not all(
        isinstance(item, Mapping) for item in comparisons
    ):
        raise ValueError("reducer update recent_comparisons must be objects")
    return current.advance(
        proposal_cycle=proposal_cycle,
        search_parent_version_id=search_parent_version_id,
        official_champion_version_id=official_champion_version_id,
        stable_knowledge=text_fields["stable_knowledge"],
        failed_hypotheses=text_fields["failed_hypotheses"],
        open_questions=text_fields["open_questions"],
        recent_comparisons=tuple(dict(item) for item in comparisons),
        exploration_debt=exploration_debt,
    )
