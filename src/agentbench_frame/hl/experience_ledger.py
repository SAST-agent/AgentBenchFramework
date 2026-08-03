"""Framework-owned, append-only empirical outcomes for HL iterations."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal


_SECRET = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_VERDICTS = {
    "verified_good",
    "verified_bad",
    "mixed",
    "inconclusive",
    "invalid",
}


@dataclasses.dataclass(frozen=True)
class ExperienceComparison:
    opponent: str
    seed: int
    parent_result: str
    candidate_result: str
    parent_rollman_score: float
    parent_ghosts_score: float
    candidate_rollman_score: float
    candidate_ghosts_score: float
    parent_margin: float
    candidate_margin: float
    margin_delta: float
    replay: str | None = None
    trace: str | None = None


@dataclasses.dataclass(frozen=True)
class ExperienceRecord:
    schema_version: str
    iteration_id: str
    act_id: str
    branch_index: int
    parent_version_id: str
    candidate_version_id: str
    selected: bool
    activation_condition: str
    mechanism: str
    preservation_contract: str
    activation_status: str
    decision_count: int
    changed_action_count: int
    verdict: Literal[
        "verified_good",
        "verified_bad",
        "mixed",
        "inconclusive",
        "invalid",
    ]
    comparisons: tuple[ExperienceComparison, ...] = ()

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.iteration_id, self.branch_index, self.candidate_version_id


def _text(value: Any, *, field: str) -> str:
    normalized = " ".join(str(value).split())
    if not normalized:
        raise ValueError(f"experience {field} cannot be empty")
    if len(normalized) > 1600:
        raise ValueError(f"experience {field} is too long")
    if _SECRET.search(normalized):
        raise ValueError("experience ledger cannot contain credential material")
    return normalized


def _margin(match: Mapping[str, Any]) -> float | None:
    rollman = match.get("rollman_score")
    ghosts = match.get("ghosts_score")
    if not isinstance(rollman, (int, float)) or not isinstance(
        ghosts, (int, float)
    ):
        return None
    return float(rollman) - float(ghosts)


def _complete_matches(
    matches: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for match in matches:
        if (
            match.get("status", "complete") != "complete"
            or match.get("result") not in {"win", "draw", "loss"}
            or _margin(match) is None
            or not isinstance(match.get("seed"), int)
        ):
            continue
        result[(str(match.get("opponent")), int(match["seed"]))] = match
    return result


def derive_experience_record(
    *,
    iteration_id: str,
    act_id: str,
    branch_index: int,
    parent_version_id: str,
    candidate_version_id: str,
    selected: bool,
    brief: Mapping[str, Any],
    parent_matches: Iterable[Mapping[str, Any]],
    candidate_matches: Iterable[Mapping[str, Any]],
    activation: Mapping[str, Any] | None,
) -> ExperienceRecord:
    """Join comparable Framework matches and derive a deterministic verdict."""

    parent_by_case = _complete_matches(parent_matches)
    candidate_by_case = _complete_matches(candidate_matches)
    comparisons: list[ExperienceComparison] = []
    for opponent, seed in sorted(set(parent_by_case) & set(candidate_by_case)):
        parent = parent_by_case[(opponent, seed)]
        candidate = candidate_by_case[(opponent, seed)]
        parent_margin = _margin(parent)
        candidate_margin = _margin(candidate)
        assert parent_margin is not None and candidate_margin is not None
        comparisons.append(
            ExperienceComparison(
                opponent=opponent,
                seed=seed,
                parent_result=str(parent["result"]),
                candidate_result=str(candidate["result"]),
                parent_rollman_score=float(parent["rollman_score"]),
                parent_ghosts_score=float(parent["ghosts_score"]),
                candidate_rollman_score=float(candidate["rollman_score"]),
                candidate_ghosts_score=float(candidate["ghosts_score"]),
                parent_margin=parent_margin,
                candidate_margin=candidate_margin,
                margin_delta=candidate_margin - parent_margin,
                replay=(
                    None
                    if candidate.get("replay") is None
                    else str(candidate["replay"])
                ),
                trace=(
                    None
                    if candidate.get("trace") is None
                    else str(candidate["trace"])
                ),
            )
        )

    activation_status = (
        "missing" if activation is None else str(activation.get("status", "missing"))
    )
    decision_count = 0 if activation is None else int(activation.get("decision_count", 0))
    changed_action_count = (
        0 if activation is None else int(activation.get("changed_action_count", 0))
    )
    if (
        activation_status != "complete"
        or decision_count < 1
        or changed_action_count < 1
        or changed_action_count > decision_count
    ):
        verdict = "invalid"
    elif not comparisons:
        verdict = "inconclusive"
    else:
        has_gain = any(row.margin_delta > 0 for row in comparisons)
        has_regression = any(row.margin_delta < 0 for row in comparisons)
        if has_gain and has_regression:
            verdict = "mixed"
        elif has_gain:
            verdict = "verified_good"
        elif has_regression:
            verdict = "verified_bad"
        else:
            verdict = "inconclusive"

    return ExperienceRecord(
        schema_version="1.0",
        iteration_id=_text(iteration_id, field="iteration_id"),
        act_id=_text(act_id, field="act_id"),
        branch_index=int(branch_index),
        parent_version_id=_text(parent_version_id, field="parent_version_id"),
        candidate_version_id=_text(
            candidate_version_id, field="candidate_version_id"
        ),
        selected=bool(selected),
        activation_condition=_text(
            brief.get("activation_condition"), field="activation_condition"
        ),
        mechanism=_text(brief.get("mechanism"), field="mechanism"),
        preservation_contract=_text(
            brief.get("preservation_contract"), field="preservation_contract"
        ),
        activation_status=activation_status,
        decision_count=decision_count,
        changed_action_count=changed_action_count,
        verdict=verdict,
        comparisons=tuple(comparisons),
    )


def _record_to_dict(record: ExperienceRecord) -> dict[str, Any]:
    return dataclasses.asdict(record)


def _record_from_dict(value: Mapping[str, Any]) -> ExperienceRecord:
    fields = {field.name for field in dataclasses.fields(ExperienceRecord)}
    if set(value) != fields:
        raise ValueError("invalid experience ledger record fields")
    raw_comparisons = value["comparisons"]
    if not isinstance(raw_comparisons, list):
        raise ValueError("experience comparisons must be a list")
    comparison_fields = {
        field.name for field in dataclasses.fields(ExperienceComparison)
    }
    comparisons = []
    for raw in raw_comparisons:
        if not isinstance(raw, Mapping) or set(raw) != comparison_fields:
            raise ValueError("invalid experience comparison fields")
        comparisons.append(ExperienceComparison(**dict(raw)))
    record = ExperienceRecord(
        **{key: value[key] for key in fields - {"comparisons"}},
        comparisons=tuple(comparisons),
    )
    _validate_record(record)
    return record


def _validate_record(record: ExperienceRecord) -> None:
    if record.schema_version != "1.0":
        raise ValueError("unsupported experience ledger schema")
    if record.verdict not in _VERDICTS:
        raise ValueError("invalid experience verdict")
    if record.branch_index < 0:
        raise ValueError("experience branch_index must be non-negative")
    if record.decision_count < 0 or not 0 <= record.changed_action_count <= record.decision_count:
        raise ValueError("invalid experience activation counts")
    for field in (
        "iteration_id",
        "act_id",
        "parent_version_id",
        "candidate_version_id",
        "activation_condition",
        "mechanism",
        "preservation_contract",
        "activation_status",
    ):
        _text(getattr(record, field), field=field)
    payload = json.dumps(
        _record_to_dict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if _SECRET.search(payload):
        raise ValueError("experience ledger cannot contain credential material")


class ExperienceLedger:
    """Persist exact branch outcomes without model-authored rewriting."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[ExperienceRecord, ...]:
        if not self.path.is_file():
            return ()
        records: list[ExperienceRecord] = []
        identities: set[tuple[str, int, str]] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("experience ledger lines must be objects")
            record = _record_from_dict(value)
            if record.identity in identities:
                raise ValueError("duplicate experience ledger identity")
            identities.add(record.identity)
            records.append(record)
        return tuple(records)

    def append(self, record: ExperienceRecord) -> bool:
        _validate_record(record)
        records = self.load()
        existing = {item.identity: item for item in records}.get(record.identity)
        if existing is not None:
            if existing != record:
                raise ValueError("conflicting experience ledger identity")
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            _record_to_dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        return True
