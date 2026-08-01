"""Structured planner output for mechanism-diverse K-candidate cycles."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Mapping


@dataclasses.dataclass(frozen=True)
class BranchBrief:
    branch_index: int
    diagnosis: str
    mechanism: str
    expected_change: str
    falsifier: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _text(value: Any, field: str) -> str:
    normalized = " ".join(str(value).split())
    if not normalized:
        raise ValueError(f"branch brief {field} cannot be empty")
    if len(normalized) > 2000:
        raise ValueError(f"branch brief {field} is too long")
    return normalized


def _mechanism_key(value: str) -> str:
    normalized = re.sub(r"\d+(?:\.\d+)?", "#", value.lower())
    normalized = re.sub(r"\b(?:threshold|parameter|value|change|set|tune)\b", " ", normalized)
    return " ".join(normalized.split())


def load_branch_briefs(
    path: str | Path,
    *,
    expected_count: int,
) -> tuple[BranchBrief, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, list):
        raw_branches = value
    elif isinstance(value, Mapping) and set(value) == {"branches"}:
        raw_branches = value["branches"]
    else:
        raise ValueError("planner output must be a branch array")
    if not isinstance(raw_branches, list) or len(raw_branches) != expected_count:
        raise ValueError(f"planner must produce exactly {expected_count} branches")
    briefs: list[BranchBrief] = []
    allowed = {
        "branch_index",
        "diagnosis",
        "mechanism",
        "expected_change",
        "falsifier",
    }
    for raw in raw_branches:
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            raise ValueError("branch brief fields are invalid")
        index = raw["branch_index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("branch_index must be an integer")
        briefs.append(
            BranchBrief(
                branch_index=index,
                diagnosis=_text(raw["diagnosis"], "diagnosis"),
                mechanism=_text(raw["mechanism"], "mechanism"),
                expected_change=_text(raw["expected_change"], "expected_change"),
                falsifier=_text(raw["falsifier"], "falsifier"),
            )
        )
    briefs.sort(key=lambda brief: brief.branch_index)
    if [brief.branch_index for brief in briefs] != list(range(expected_count)):
        raise ValueError("branch indices must be contiguous from zero")
    keys = {_mechanism_key(brief.mechanism) for brief in briefs}
    if "" in keys or len(keys) != expected_count:
        raise ValueError("planner must produce mechanism-distinct mechanisms")
    return tuple(briefs)
