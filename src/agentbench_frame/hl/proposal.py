"""Structured planner output for mechanism-diverse K-candidate cycles."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


_INLINE_SUMMARY_LIMIT = 12_000
_INLINE_TEXT_LIMIT = 24_000
_CANDIDATE_SOURCE_LIMIT = 24_000


def stratify_rollout_evidence(
    replay_evidence: list[Mapping[str, Any]],
    *,
    hard_opponents: tuple[str, str] = ("rank15", "rank16"),
) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Assign bounded, complementary failure evidence to four sibling acts."""

    if len(hard_opponents) != 2 or len(set(hard_opponents)) != 2:
        raise ValueError("k4 evidence routing requires two distinct hard opponents")

    def severity(item: Mapping[str, Any]) -> tuple[Any, ...]:
        rollman = item.get("rollman_score")
        ghosts = item.get("ghosts_score")
        margin = (
            float(rollman) - float(ghosts)
            if isinstance(rollman, (int, float))
            and isinstance(ghosts, (int, float))
            else float("inf")
        )
        result_order = {"loss": 0, "draw": 1, "win": 2}
        return (
            result_order.get(str(item.get("result")), 3),
            margin,
            int(item.get("seed", 0)),
            str(item.get("summary") or ""),
        )

    eligible = [
        dict(item)
        for item in replay_evidence
        if item.get("phase") != "certification"
        and item.get("candidate_fault") is None
        and str(item.get("opponent")) in hard_opponents
    ]
    by_opponent = {
        opponent: sorted(
            (
                item
                for item in eligible
                if str(item.get("opponent")) == opponent
            ),
            key=severity,
        )
        for opponent in hard_opponents
    }
    first, second = hard_opponents
    branch0 = by_opponent[first][:2]
    branch1 = by_opponent[second][:2]
    branch2 = [
        *by_opponent[first][:1],
        *by_opponent[second][:1],
    ][:2]
    branch3 = [
        *by_opponent[first][2:3],
        *by_opponent[second][2:3],
    ]
    if not branch3:
        branch3 = sorted(eligible, key=severity, reverse=True)[:2]

    packets = (branch0, branch1, branch2, branch3)
    fallback = sorted(eligible, key=severity)[:2]
    return tuple(tuple(packet or fallback) for packet in packets)


@dataclasses.dataclass(frozen=True)
class BranchBrief:
    branch_index: int
    diagnosis: str
    mechanism: str
    activation_condition: str
    preservation_contract: str
    expected_change: str
    falsifier: str
    code_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _bounded_text(path: str | Path, *, limit: int) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if len(text) > limit:
        return text[:limit] + "\n[summary truncated]"
    return text


def build_candidate_code_index(
    source_path: str | Path,
) -> list[dict[str, Any]]:
    """Index exact module-level policy functions without embedding source."""

    source = Path(source_path).read_text(encoding="utf-8")
    module = ast.parse(source)
    return [
        {
            "name": node.name,
            "signature": f"{node.name}({ast.unparse(node.args)})",
            "start_line": node.lineno,
            "end_line": node.end_lineno,
        }
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _truncate_source(source: str, *, limit: int) -> str:
    marker = "\n[source truncated]\n"
    if limit < len(marker):
        return marker[:limit]
    body_limit = limit - len(marker)
    head = (body_limit + 1) // 2
    tail = body_limit - head
    return source[:head] + marker + (source[-tail:] if tail else "")


def build_candidate_code_slices(
    source_path: str | Path,
    *,
    code_symbols: tuple[str, ...],
    source_limit: int = _CANDIDATE_SOURCE_LIMIT,
) -> list[dict[str, Any]]:
    """Resolve selected module functions into deterministic bounded source."""

    if source_limit < 1:
        raise ValueError("source_limit must be positive")
    source = Path(source_path).read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    module = ast.parse(source)
    nodes = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    unknown = set(code_symbols) - set(nodes)
    if unknown:
        raise ValueError(
            "unknown code symbol: " + ", ".join(sorted(unknown))
        )
    if "ai_func" not in code_symbols:
        raise ValueError("code_symbols must include ai_func")

    def item_for(name: str, body: str, completeness: str) -> dict[str, Any]:
        node = nodes[name]
        return {
            "name": name,
            "signature": f"{name}({ast.unparse(node.args)})",
            "start_line": node.lineno,
            "end_line": node.end_lineno,
            "completeness": completeness,
            "source": body,
        }

    bodies = {
        name: "".join(lines[nodes[name].lineno - 1 : nodes[name].end_lineno])
        for name in code_symbols
    }
    entry = bodies["ai_func"]
    if len(entry) > source_limit:
        raise ValueError(
            f"ai_func exceeds candidate source limit {source_limit}"
        )
    remaining = source_limit - len(entry)
    rendered: dict[str, dict[str, Any]] = {
        "ai_func": item_for("ai_func", entry, "complete")
    }
    helpers = [name for name in code_symbols if name != "ai_func"]
    for position, name in enumerate(helpers):
        helpers_left = len(helpers) - position
        allowance = remaining // helpers_left
        body = bodies[name]
        if len(body) <= allowance:
            rendered[name] = item_for(name, body, "complete")
        else:
            rendered[name] = item_for(
                name,
                _truncate_source(body, limit=allowance),
                "truncated",
            )
        remaining -= len(rendered[name]["source"])
    return [rendered[name] for name in code_symbols]


def write_planner_input_packet(
    *,
    output_path: str | Path,
    iteration_id: str,
    parent_version_id: str,
    game_digest_path: str | Path,
    context_manifest_path: str | Path,
    research_state_path: str | Path,
    replay_evidence: list[Mapping[str, Any]],
    previous_measurements: Mapping[str, Any],
    active_target: str | None,
    candidate_source_path: str | Path,
) -> Path:
    """Collapse bounded planner evidence into one read-only artifact."""

    evidence_fields = (
        "opponent",
        "seed",
        "result",
        "phase",
        "rollman_score",
        "ghosts_score",
    )
    evidence = []
    for item in replay_evidence:
        bounded = {
            field: item[field]
            for field in evidence_fields
            if item.get(field) is not None
        }
        summary = item.get("summary")
        if isinstance(summary, str) and Path(summary).is_file():
            bounded["summary_text"] = _bounded_text(
                summary,
                limit=_INLINE_SUMMARY_LIMIT,
            )
        evidence.append(bounded)

    measurements = {
        key: value
        for key, value in previous_measurements.items()
        if key != "opponent_distillation_path"
    }
    distillation = None
    distillation_path = previous_measurements.get("opponent_distillation_path")
    if isinstance(distillation_path, str) and Path(distillation_path).is_file():
        raw = _bounded_text(distillation_path, limit=_INLINE_TEXT_LIMIT)
        try:
            distillation = json.loads(raw)
        except json.JSONDecodeError:
            distillation = raw

    value = {
        "schema_version": "1.0",
        "iteration_id": iteration_id,
        "parent_version_id": parent_version_id,
        "active_target": active_target,
        "game_digest": json.loads(
            Path(game_digest_path).read_text(encoding="utf-8")
        ),
        "context_manifest": json.loads(
            Path(context_manifest_path).read_text(encoding="utf-8")
        ),
        "research_state": json.loads(
            Path(research_state_path).read_text(encoding="utf-8")
        ),
        "replay_evidence": evidence,
        "previous_measurements": measurements,
        "opponent_distillation": distillation,
        "candidate_code_index": build_candidate_code_index(
            candidate_source_path
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def write_candidate_input_packet(
    *,
    output_path: str | Path,
    iteration_id: str,
    branch_brief: Mapping[str, Any],
    game_digest_path: str | Path,
    research_state_path: str | Path,
    experience_path: str | Path,
    replay_evidence: list[Mapping[str, Any]],
    previous_measurements: Mapping[str, Any],
    active_target: str | None,
    locked_opponents: tuple[str, ...],
    candidate_source_path: str | Path | None = None,
    smoke_fixture_path: str | Path | None = None,
    candidate_workspace: str | Path | None = None,
) -> Path:
    """Write one bounded candidate context artifact for a single first read."""

    evidence = []
    for item in replay_evidence:
        bounded = dict(item)
        summary = item.get("summary")
        if isinstance(summary, str) and Path(summary).is_file():
            bounded["summary_text"] = _bounded_text(
                summary,
                limit=_INLINE_SUMMARY_LIMIT,
            )
        evidence.append(bounded)
    measurements = dict(previous_measurements)
    distillation = None
    distillation_path = measurements.get("opponent_distillation_path")
    if isinstance(distillation_path, str) and Path(distillation_path).is_file():
        raw = _bounded_text(distillation_path, limit=_INLINE_TEXT_LIMIT)
        try:
            distillation = json.loads(raw)
        except json.JSONDecodeError:
            distillation = raw
    code_index = (
        []
        if candidate_source_path is None
        else build_candidate_code_index(candidate_source_path)
    )
    raw_symbols = branch_brief.get("code_symbols", ())
    code_slices = (
        []
        if candidate_source_path is None or not raw_symbols
        else build_candidate_code_slices(
            candidate_source_path,
            code_symbols=tuple(str(item) for item in raw_symbols),
        )
    )
    if (smoke_fixture_path is None) != (candidate_workspace is None):
        raise ValueError(
            "smoke fixture and candidate workspace must be provided together"
        )
    smoke_contract = None
    if smoke_fixture_path is not None and candidate_workspace is not None:
        fixture = Path(smoke_fixture_path).resolve()
        workspace = Path(candidate_workspace).resolve()
        if not fixture.is_file():
            raise FileNotFoundError(fixture)
        scenario = workspace / ".agentbench" / "smoke_scenario.json"
        result = workspace / ".agentbench" / "candidate_smoke_result.json"
        smoke_contract = {
            "fixture_path": str(fixture),
            "scenario_path": str(scenario),
            "result_path": str(result),
            "command": [
                sys.executable,
                str(fixture),
                "--workspace",
                str(workspace),
                "--scenario",
                str(scenario),
                "--output",
                str(result),
            ],
        }
    value = {
        "schema_version": "1.0",
        "iteration_id": iteration_id,
        "branch_brief": dict(branch_brief),
        "active_target": active_target,
        "locked_opponents": list(locked_opponents),
        "game_digest": json.loads(Path(game_digest_path).read_text(encoding="utf-8")),
        "research_state": json.loads(
            Path(research_state_path).read_text(encoding="utf-8")
        ),
        "experience_skill": _bounded_text(
            experience_path,
            limit=_INLINE_TEXT_LIMIT,
        ),
        "replay_evidence": evidence,
        "previous_measurements": measurements,
        "opponent_distillation": distillation,
        "candidate_code_index": code_index,
        "candidate_code_slices": code_slices,
        "smoke_contract": smoke_contract,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def branch_briefs_json_schema(*, expected_count: int) -> dict[str, Any]:
    """Return the strict Codex final-output schema for one planner cycle."""

    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    text_field = {"type": "string", "minLength": 1, "maxLength": 2000}
    branches = {
        "type": "array",
        "minItems": expected_count,
        "maxItems": expected_count,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "branch_index",
                "diagnosis",
                "mechanism",
                "activation_condition",
                "preservation_contract",
                "expected_change",
                "falsifier",
                "code_symbols",
            ],
            "properties": {
                "branch_index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": expected_count - 1,
                },
                "diagnosis": dict(text_field),
                "mechanism": dict(text_field),
                "activation_condition": dict(text_field),
                "preservation_contract": dict(text_field),
                "expected_change": dict(text_field),
                "falsifier": dict(text_field),
                "code_symbols": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "uniqueItems": True,
                    "contains": {"const": "ai_func"},
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                },
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["branches"],
        "properties": {"branches": branches},
    }


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
    known_code_symbols: set[str] | None = None,
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
        "activation_condition",
        "preservation_contract",
        "expected_change",
        "falsifier",
        "code_symbols",
    }
    for raw in raw_branches:
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            raise ValueError("branch brief fields are invalid")
        index = raw["branch_index"]
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("branch_index must be an integer")
        raw_symbols = raw["code_symbols"]
        if not isinstance(raw_symbols, list) or not 2 <= len(raw_symbols) <= 8:
            raise ValueError("code_symbols must contain 2-8 names")
        if any(not isinstance(item, str) or not item for item in raw_symbols):
            raise ValueError("code_symbols must contain non-empty names")
        symbols = tuple(raw_symbols)
        if len(set(symbols)) != len(symbols):
            raise ValueError("code_symbols must be unique")
        if "ai_func" not in symbols:
            raise ValueError("code_symbols must include ai_func")
        if known_code_symbols is not None and set(symbols) - known_code_symbols:
            raise ValueError("code_symbols contain unknown module functions")
        briefs.append(
            BranchBrief(
                branch_index=index,
                diagnosis=_text(raw["diagnosis"], "diagnosis"),
                mechanism=_text(raw["mechanism"], "mechanism"),
                activation_condition=_text(
                    raw["activation_condition"], "activation_condition"
                ),
                preservation_contract=_text(
                    raw["preservation_contract"], "preservation_contract"
                ),
                expected_change=_text(raw["expected_change"], "expected_change"),
                falsifier=_text(raw["falsifier"], "falsifier"),
                code_symbols=symbols,
            )
        )
    briefs.sort(key=lambda brief: brief.branch_index)
    if [brief.branch_index for brief in briefs] != list(range(expected_count)):
        raise ValueError("branch indices must be contiguous from zero")
    keys = {_mechanism_key(brief.mechanism) for brief in briefs}
    if "" in keys or len(keys) != expected_count:
        raise ValueError("planner must produce mechanism-distinct mechanisms")
    return tuple(briefs)
