"""Ludemic Rule Description Complexity for the AB-Rule/1 corpus."""

from __future__ import annotations

import io
import json
import token
import tokenize
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
_TOKEN_TYPES = frozenset(
    {token.NAME, token.NUMBER, token.STRING, token.OP}
)


class RuleCorpusError(ValueError):
    """Raised when the frozen formal-rule corpus is incomplete or ambiguous."""


def _lexical_token_count(fragment: str) -> int:
    if not fragment:
        return 0
    try:
        stream = tokenize.generate_tokens(io.StringIO(fragment).readline)
        return sum(item.type in _TOKEN_TYPES for item in stream)
    except (IndentationError, tokenize.TokenError):
        # Type annotations and parameter lists can be fragments rather than
        # complete Python expressions. A stable punctuation-aware fallback is
        # sufficient because identifier spelling never affects token count.
        return sum(
            1
            for item in tokenize.generate_tokens(
                io.StringIO(f"f({fragment})").readline
            )
            if item.type in _TOKEN_TYPES
        ) - 3


def _node_depth(node: RuleNode, depth: int) -> int:
    own_depth = depth
    if node.expression is not None:
        own_depth = max(own_depth, depth + node.expression.depth)
    for child in node.children:
        own_depth = max(own_depth, _node_depth(child, depth + 1))
    return own_depth


def _canonical_token_count(document: RuleDocument) -> int:
    count = 0
    for node in document.walk():
        count += 1  # frozen AB-Rule keyword
        if node.name is not None:
            count += 1  # alpha-normalized identifier
        if node.detail:
            count += _lexical_token_count(node.detail)
        if node.expression is not None:
            count += _lexical_token_count(node.expression.source)
    return count


def measure_rule_document(
    document: RuleDocument,
    source: str,
) -> dict[str, int | str]:
    """Measure one parsed formal rule description without compiling it."""

    nodes = tuple(document.walk())
    semantic_nodes = tuple(node for node in nodes if node.kind != "game")
    expression_atoms = sum(
        node.expression.calls + node.expression.operators
        for node in semantic_nodes
        if node.expression is not None
    )
    parameters = sum(
        node.expression.literals
        for node in semantic_nodes
        if node.expression is not None
    )
    rule_lines = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in source.splitlines()
    )

    return {
        "game_id": document.game_id,
        "language": LANGUAGE_ID,
        "rule_atoms": len(semantic_nodes) + expression_atoms,
        "composition_depth": max(
            (_node_depth(node, 1) for node in document.nodes),
            default=0,
        ),
        "branch_count": sum(
            node.kind in {"when", "otherwise", "choose"}
            for node in semantic_nodes
        ),
        "parameter_count": parameters,
        "canonical_tokens": _canonical_token_count(document),
        "rule_lines": rule_lines,
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
            "The primary value is the number of semantic rule-atom "
            "occurrences in each formal AB-Rule/1 pseudocode description. "
            "Descriptions are parsed but not compiled."
        ),
        "",
        (
            "These values are language-relative, best-known formal "
            "description lengths. They are not exact Kolmogorov complexity, "
            "implementation size, state-space size, strategic depth, "
            "learning difficulty, or information gain."
        ),
        "",
        "| Rank | Game ID | Game | Rule atoms | Depth | Branches | "
        "Parameters | Canonical tokens | Rule lines |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, game in enumerate(report["games"], start=1):
        lines.append(
            f"| {rank} | `{game['game_id']}` | {game['title']} | "
            f"{game['rule_atoms']} | {game['composition_depth']} | "
            f"{game['branch_count']} | {game['parameter_count']} | "
            f"{game['canonical_tokens']} | {game['rule_lines']} |"
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
