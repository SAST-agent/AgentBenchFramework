"""Ludemic Rule Description Complexity for the AB-Rule/1 corpus."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

from .agentbench_catalog import AGENTBENCH_SOURCE_COMMIT
from .rule_language import RuleDocument, RuleNode, parse_rule_description

SCHEMA_VERSION = "agentbench.rule-complexity.v1"
LANGUAGE_ID = "AB-Rule/1"
METRIC_NAME = "Ludemic Rule Description Complexity"
METRIC_UNIT = "rule atoms (RA)"
EXPECTED_RULE_GAMES: dict[str, str] = {
    "23_doto": "DOTO",
    "24_miracle": "Miracle",
    "25_aquawar": "AquaWar",
    "25_lostspace": "LostSpace",
    "26_snakego": "SnakeGo",
    "27_antwar": "AntWar",
    "28_generals": "Generals",
    "29_rollman": "Rollman",
    "30_antwar2": "AntWar2",
}
_REQUIRED_METADATA = frozenset(
    {"provenance", "source-root", "reviewed", "includes", "excludes"}
)
_REQUIRED_KINDS = frozenset(
    {
        "players",
        "entity",
        "action",
        "observation",
        "setup",
        "rule",
        "terminal",
    }
)
_ATOM_CATEGORIES = (
    "state",
    "action",
    "observation",
    "setup",
    "condition",
    "transition",
    "outcome",
)
_CONDITION_KINDS = frozenset({"require", "when", "otherwise", "for", "choose"})
_TRANSITION_KINDS = frozenset(
    {
        "let",
        "update",
        "apply",
        "create",
        "delete",
        "emit",
        "return",
        "pass",
    }
)
_GROUP_KINDS = frozenset({"game", "setup", "rule", "terminal"})


class RuleCorpusError(ValueError):
    """Raised when the frozen formal-rule corpus is incomplete or ambiguous."""


def _collect_propositions(
    node: RuleNode,
    counts: Counter[str],
    *,
    section: str | None = None,
) -> None:
    """Place each semantic proposition in exactly one public partition."""

    child_section = section
    if node.kind == "setup":
        child_section = "setup"
    elif node.kind == "terminal":
        child_section = "outcome"
    elif node.kind in {"entity", "enum"}:
        child_section = "state"
    elif node.kind == "action":
        child_section = "action"
    elif node.kind == "observation":
        child_section = "observation"

    category: str | None = None
    if section in {"setup", "outcome"} and node.kind not in _GROUP_KINDS:
        category = section
    elif node.kind in {"players", "constant", "entity", "enum"}:
        category = "state"
    elif node.kind in {"value", "field"} and section == "state":
        category = "state"
    elif node.kind == "action":
        category = "action"
    elif node.kind == "field" and section == "action":
        category = "action"
    elif node.kind == "observation":
        category = "observation"
    elif (
        node.kind == "field" and section == "observation"
    ) or node.kind in {"reveal", "hide"}:
        category = "observation"
    elif node.kind in _CONDITION_KINDS:
        category = "condition"
    elif node.kind in _TRANSITION_KINDS:
        category = "transition"
    elif node.kind not in _GROUP_KINDS:
        raise RuleCorpusError(
            f"cannot classify AB-Rule proposition {node.kind!r} "
            f"at line {node.line}"
        )

    if category is not None:
        counts[category] += 1
    for child in node.children:
        _collect_propositions(child, counts, section=child_section)


def _proposition_breakdown(document: RuleDocument) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for node in document.nodes:
        _collect_propositions(node, counts)
    return {category: counts[category] for category in _ATOM_CATEGORIES}


def measure_rule_document(
    document: RuleDocument,
    source: str,
) -> dict[str, Any]:
    """Measure one parsed formal rule description without compiling it."""

    breakdown = _proposition_breakdown(document)

    return {
        "game_id": document.game_id,
        "language": LANGUAGE_ID,
        "rule_atoms": sum(breakdown.values()),
        "atom_breakdown": breakdown,
        "description_bytes": len(source.encode("utf-8")),
        "description_sha256": sha256(source.encode("utf-8")).hexdigest(),
    }


def validate_rule_document(
    document: RuleDocument,
    *,
    expected_game_id: str,
) -> None:
    """Reject an incomplete or provenance-free corpus description."""

    if document.game_id != expected_game_id:
        raise RuleCorpusError(
            f"{document.path}: expected game ID {expected_game_id!r}, "
            f"got {document.game_id!r}"
        )
    missing_metadata = sorted(_REQUIRED_METADATA - document.metadata.keys())
    if missing_metadata:
        raise RuleCorpusError(
            f"{document.path}: missing metadata: {','.join(missing_metadata)}"
        )
    if document.metadata["provenance"] != AGENTBENCH_SOURCE_COMMIT:
        raise RuleCorpusError(
            f"{document.path}: provenance must be {AGENTBENCH_SOURCE_COMMIT}"
        )
    kinds = {node.kind for node in document.walk()}
    missing_kinds = sorted(_REQUIRED_KINDS - kinds)
    if missing_kinds:
        raise RuleCorpusError(
            f"{document.path}: missing semantic sections: "
            f"{','.join(missing_kinds)}"
        )


def _rule_resource_root():
    return files(__package__).joinpath("rules", "ab_rule_v1")


def measure_rule_corpus() -> dict[str, Any]:
    """Load, validate, and measure the frozen packaged nine-game corpus."""

    root = _rule_resource_root()
    try:
        resources = tuple(
            sorted(
                (
                    resource
                    for resource in root.iterdir()
                    if resource.name.endswith(".abrule")
                ),
                key=lambda resource: resource.name,
            )
        )
    except FileNotFoundError as exc:
        raise RuleCorpusError("AB-Rule/1 corpus resource is missing") from exc

    actual_ids = {resource.name.removesuffix(".abrule") for resource in resources}
    expected_ids = set(EXPECTED_RULE_GAMES)
    if actual_ids != expected_ids:
        missing = ",".join(sorted(expected_ids - actual_ids)) or "-"
        extra = ",".join(sorted(actual_ids - expected_ids)) or "-"
        raise RuleCorpusError(
            f"AB-Rule/1 corpus mismatch: missing={missing}; extra={extra}"
        )

    games_by_id = []
    for resource in resources:
        game_id = resource.name.removesuffix(".abrule")
        source = resource.read_text(encoding="utf-8")
        document = parse_rule_description(source, path=resource.name)
        validate_rule_document(document, expected_game_id=game_id)
        result = measure_rule_document(document, source)
        result.update(
            {
                "title": EXPECTED_RULE_GAMES[game_id],
                "source_root": document.metadata["source-root"],
                "reviewed": document.metadata["reviewed"],
                "includes": document.metadata["includes"],
                "excludes": document.metadata["excludes"],
            }
        )
        games_by_id.append(result)

    games_by_id.sort(key=lambda game: game["game_id"])
    ranked = sorted(
        games_by_id,
        key=lambda game: (game["rule_atoms"], game["game_id"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": {
            "name": METRIC_NAME,
            "language": LANGUAGE_ID,
            "primary": "rule_atoms",
            "unit": METRIC_UNIT,
            "compiled": False,
        },
        "source": {
            "repository": "https://github.com/Aoraku/AgentBench",
            "commit": AGENTBENCH_SOURCE_COMMIT,
            "deepclue_included": False,
        },
        "games": ranked,
        "games_by_id": games_by_id,
    }


def render_rule_markdown(report: dict[str, Any]) -> str:
    """Render a stable human-readable rule-complexity report."""

    lines = [
        "# AgentBench Ludemic Rule Description Complexity",
        "",
        (
            f"Source: [Aoraku/AgentBench]"
            f"(https://github.com/Aoraku/AgentBench) at "
            f"`{report['source']['commit']}`."
        ),
        "",
        (
            "The primary value is the cardinality of the disjoint set of "
            "atomic rule-proposition occurrences in each formal AB-Rule/1 "
            "description. Descriptions are parsed but not compiled, and "
            "expression AST shape is not counted."
        ),
        "",
        (
            "These values are language-relative, best-known formal "
            "description lengths. They are not exact Kolmogorov complexity, "
            "implementation size, state-space size, strategic depth, "
            "learning difficulty, or information gain."
        ),
        "",
        "| Rank | Game ID | Game | Rule atoms | State | Actions | "
        "Observations | Setup | Conditions | Transitions | Outcomes |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, game in enumerate(report["games"], start=1):
        atoms = game["atom_breakdown"]
        lines.append(
            f"| {rank} | `{game['game_id']}` | {game['title']} | "
            f"{game['rule_atoms']} | {atoms['state']} | {atoms['action']} | "
            f"{atoms['observation']} | {atoms['setup']} | "
            f"{atoms['condition']} | {atoms['transition']} | "
            f"{atoms['outcome']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "DeepClue is excluded. The descriptions include abstract "
                "state, setup, legal action families, transitions, chance, "
                "observations, terminal conditions, and outcomes. They "
                "exclude communication, rendering, replay serialization, "
                "logging, and implementation optimizations."
            ),
            "",
            (
                "AB-Ludi/1 `k_upper_bits` remains a separate implementation-"
                "description metric and must not be compared numerically with "
                "AB-Rule/1 rule atoms."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_bytes(path: str | Path, content: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def write_rule_reports(
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    """Measure the packaged corpus and write canonical UTF-8 reports."""

    json_target = Path(json_path).resolve()
    markdown_target = Path(markdown_path).resolve()
    same_output = json_target == markdown_target
    if json_target.exists() and markdown_target.exists():
        same_output = same_output or json_target.samefile(markdown_target)
    if same_output:
        raise ValueError(
            "JSON and Markdown outputs must resolve to different paths"
        )

    report = measure_rule_corpus()
    json_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    markdown_bytes = render_rule_markdown(report).encode("utf-8")
    _write_bytes(json_target, json_bytes)
    _write_bytes(markdown_target, markdown_bytes)
    return report


__all__ = [
    "EXPECTED_RULE_GAMES",
    "LANGUAGE_ID",
    "METRIC_NAME",
    "METRIC_UNIT",
    "RuleCorpusError",
    "measure_rule_corpus",
    "measure_rule_document",
    "render_rule_markdown",
    "validate_rule_document",
    "write_rule_reports",
]
