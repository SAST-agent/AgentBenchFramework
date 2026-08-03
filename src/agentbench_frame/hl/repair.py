"""Bounded same-seed feedback and conservative within-branch repair selection."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentbench_frame.hl.proposal import (
    BranchBrief,
    build_candidate_code_index,
    build_candidate_code_slices,
)
from agentbench_frame.hl.selection import CandidateDiagnostics


_SUMMARY_TEXT_LIMIT = 12_000
_INLINE_TEXT_LIMIT = 32_768

if TYPE_CHECKING:
    from agentbench_frame.hl.controller import CandidateResult


@dataclasses.dataclass(frozen=True)
class RepairSelection:
    """The initial, repaired, and retained result for one planner branch."""

    branch_index: int
    initial: CandidateResult
    repaired: CandidateResult | None
    representative: CandidateResult
    repair_input_path: Path


def _complete_matches(result: CandidateResult) -> dict[int, Mapping[str, Any]]:
    matches: dict[int, Mapping[str, Any]] = {}
    for match in result.evaluation.matches:
        seed = match.get("seed")
        if match.get("status", "complete") != "complete":
            continue
        if isinstance(seed, int) and not isinstance(seed, bool):
            matches[seed] = match
    return matches


def _bounded_match(
    match: Mapping[str, Any],
    summary_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    allowed = {
        "status",
        "seed",
        "opponent",
        "result",
        "rollman_score",
        "ghosts_score",
        "max_level",
        "game_agent_decisions",
        "captures",
        "replay",
        "trace",
    }
    value = {key: match[key] for key in allowed if key in match}
    resolved = summary_resolver(match)
    if not isinstance(resolved, Mapping):
        raise TypeError("summary_resolver must return a mapping")
    for key in ("summary", "replay", "trace"):
        if key in resolved:
            value[key] = resolved[key]
    summary = resolved.get("summary")
    if isinstance(summary, str) and Path(summary).is_file():
        text = Path(summary).read_text(encoding="utf-8")
        if len(text) > _SUMMARY_TEXT_LIMIT:
            text = text[:_SUMMARY_TEXT_LIMIT] + "\n[summary truncated]"
        value["summary_text"] = text
    return value


def _result_packet(
    result: CandidateResult,
    matches: Mapping[int, Mapping[str, Any]],
    shared_seeds: tuple[int, ...],
    summary_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version_id": result.version.version_id,
        "status": result.evaluation.status,
        "score": result.evaluation.score,
        "error": result.evaluation.error,
        "activation": (
            None if result.activation is None else dict(result.activation)
        ),
        "matches": [
            _bounded_match(matches[seed], summary_resolver) for seed in shared_seeds
        ],
    }


def build_repair_packet(
    *,
    output_path: str | Path,
    iteration_id: str,
    branch_brief: BranchBrief,
    parent: CandidateResult,
    candidate: CandidateResult,
    summary_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Path:
    """Write compact parent/candidate evidence aligned on identical match seeds."""

    if candidate.branch_index != branch_brief.branch_index:
        raise ValueError("candidate and repair brief must belong to the same branch")
    parent_matches = _complete_matches(parent)
    candidate_matches = _complete_matches(candidate)
    shared_seeds = tuple(sorted(parent_matches.keys() & candidate_matches.keys()))
    if not shared_seeds:
        raise ValueError("repair feedback requires at least one shared seed")

    packet = {
        "schema_version": "1.0",
        "iteration_id": iteration_id,
        "branch_index": branch_brief.branch_index,
        "scope": branch_brief.to_dict(),
        "parent": _result_packet(
            parent, parent_matches, shared_seeds, summary_resolver
        ),
        "candidate": _result_packet(
            candidate, candidate_matches, shared_seeds, summary_resolver
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def build_activation_repair_packet(
    *,
    output_path: str | Path,
    iteration_id: str,
    branch_brief: BranchBrief,
    parent: CandidateResult,
    candidate: CandidateResult,
    minimum_changed_actions: int,
) -> Path:
    """Write bounded evidence for a candidate that did not reach policy output."""

    if candidate.branch_index != branch_brief.branch_index:
        raise ValueError("candidate and repair brief must belong to the same branch")
    if minimum_changed_actions < 1:
        raise ValueError("minimum_changed_actions must be >= 1")
    if candidate.activation is None:
        raise ValueError("activation repair requires activation evidence")
    packet = {
        "schema_version": "1.0",
        "repair_kind": "activation_integration",
        "iteration_id": iteration_id,
        "branch_index": branch_brief.branch_index,
        "minimum_changed_actions": minimum_changed_actions,
        "scope": branch_brief.to_dict(),
        "parent": {
            "version_id": parent.version.version_id,
            "status": parent.evaluation.status,
            "score": parent.evaluation.score,
            "matches": [],
        },
        "candidate": {
            "version_id": candidate.version.version_id,
            "status": candidate.evaluation.status,
            "score": candidate.evaluation.score,
            "error": candidate.evaluation.error,
            "activation": dict(candidate.activation),
            "matches": [],
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _bounded_text(path: str | Path, *, limit: int = _INLINE_TEXT_LIMIT) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[text truncated]"


def enrich_activation_repair_packet(
    path: str | Path,
    *,
    game_digest_path: str | Path,
    research_state_path: str | Path,
    experience_path: str | Path,
    candidate_source_path: str | Path,
    policy_entry_symbol: str,
    smoke_command: Sequence[str],
) -> Path:
    """Inline all bounded edit inputs so repair needs one read before editing."""

    destination = Path(path)
    value = json.loads(destination.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("repair_kind") != (
        "activation_integration"
    ):
        raise ValueError("activation repair packet has an invalid repair_kind")
    scope = value.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("activation repair packet requires scope")
    raw_symbols = scope.get("code_symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise ValueError("activation repair scope requires code_symbols")
    command = [str(item) for item in smoke_command]
    if not command or any(not item for item in command):
        raise ValueError("smoke command must contain non-empty strings")
    source = Path(candidate_source_path)
    value.update(
        {
            "game_digest": json.loads(
                Path(game_digest_path).read_text(encoding="utf-8")
            ),
            "research_state": json.loads(
                Path(research_state_path).read_text(encoding="utf-8")
            ),
            "experience_skill": _bounded_text(experience_path),
            "candidate_code_index": build_candidate_code_index(source),
            "candidate_code_slices": build_candidate_code_slices(
                source,
                code_symbols=tuple(str(item) for item in raw_symbols),
                entry_symbol=policy_entry_symbol,
            ),
            "smoke_contract": {"command": command},
            "experience_update_contract": {
                "path": str(
                    (
                        source.parent
                        / ".agentbench"
                        / "experience_update.json"
                    ).resolve()
                ),
                "required_arrays": [
                    "positive_patterns",
                    "negative_patterns",
                    "open_questions",
                    "compression_notes",
                ],
            },
        }
    )
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _diagnostics(result: CandidateResult) -> CandidateDiagnostics:
    return CandidateDiagnostics.from_matches(
        version_id=result.version.version_id,
        branch_index=result.branch_index,
        matches=result.evaluation.matches,
    )


def select_branch_representative(
    initial: CandidateResult,
    repaired: CandidateResult,
) -> CandidateResult:
    """Retain a repair only when it strictly improves its own initial branch."""

    if initial.branch_index != repaired.branch_index:
        raise ValueError("initial candidate and repair must belong to the same branch")
    initial_complete = initial.evaluation.status == "complete"
    repaired_complete = repaired.evaluation.status == "complete"
    if not repaired_complete:
        return initial
    if not initial_complete:
        return repaired

    try:
        return repaired if _diagnostics(repaired).key() > _diagnostics(initial).key() else initial
    except ValueError:
        initial_score = initial.evaluation.score
        repaired_score = repaired.evaluation.score
        if (
            initial_score is not None
            and repaired_score is not None
            and repaired_score > initial_score
        ):
            return repaired
        return initial
