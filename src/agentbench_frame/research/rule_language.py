"""AB-Rule/1 formal pseudocode parser.

AB-Rule/1 is measured as a semantic rule description.  This module parses the
description into a stable tree for validation and counting; it deliberately
does not compile or execute game rules.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Iterator


class RuleSyntaxError(ValueError):
    """Raised when a description is outside the frozen AB-Rule/1 grammar."""


@dataclass(frozen=True)
class ExpressionStats:
    """Structural facts about one restricted mathematical expression."""

    source: str
    calls: int
    operators: int
    literals: int
    depth: int


@dataclass(frozen=True)
class RuleNode:
    """One semantic declaration or statement in a rule description."""

    kind: str
    name: str | None
    line: int
    expression: ExpressionStats | None = None
    detail: str | None = None
    children: tuple["RuleNode", ...] = ()

    def walk(self) -> Iterator["RuleNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(frozen=True)
class RuleDocument:
    """A complete parsed AB-Rule/1 description."""

    game_id: str
    path: str
    metadata: dict[str, str]
    nodes: tuple[RuleNode, ...]

    def walk(self) -> Iterator[RuleNode]:
        for node in self.nodes:
            yield from node.walk()


@dataclass
class _MutableNode:
    kind: str
    name: str | None
    line: int
    expression: ExpressionStats | None = None
    detail: str | None = None
    children: list["_MutableNode"] = field(default_factory=list)

    def freeze(self) -> RuleNode:
        return RuleNode(
            kind=self.kind,
            name=self.name,
            line=self.line,
            expression=self.expression,
            detail=self.detail,
            children=tuple(child.freeze() for child in self.children),
        )


_NAME = r"[A-Za-z_][A-Za-z0-9_.-]*"
_GAME_ID = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"
_METADATA = re.compile(
    r"^#\s*(provenance|source-root|reviewed|includes|excludes):\s*(.+?)\s*$"
)
_BLOCK_KINDS = frozenset(
    {
        "enum",
        "entity",
        "action",
        "observation",
        "setup",
        "rule",
        "terminal",
        "when",
        "otherwise",
        "for",
        "choose",
    }
)
_ALLOWED_EXPRESSION_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Call,
    ast.keyword,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
)


def _error(path: str, line: int, message: str) -> RuleSyntaxError:
    return RuleSyntaxError(f"{path}:{line}: {message}")


def _expression_depth(node: ast.AST) -> int:
    children = [
        child
        for child in ast.iter_child_nodes(node)
        if not isinstance(
            child,
            (
                ast.Load,
                ast.operator,
                ast.unaryop,
                ast.boolop,
                ast.cmpop,
            ),
        )
    ]
    if not children:
        return 1
    return 1 + max(_expression_depth(child) for child in children)


def _analyze_expression(source: str, *, path: str, line: int) -> ExpressionStats:
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise _error(path, line, f"invalid expression: {source!r}") from exc

    unsupported = [
        node
        for node in ast.walk(tree)
        if not isinstance(node, _ALLOWED_EXPRESSION_NODES)
    ]
    if unsupported:
        node_name = type(unsupported[0]).__name__
        raise _error(path, line, f"unsupported expression node: {node_name}")

    calls = sum(isinstance(node, ast.Call) for node in ast.walk(tree))
    operators = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.IfExp)):
            operators += 1
        elif isinstance(node, ast.BoolOp):
            operators += max(1, len(node.values) - 1)
        elif isinstance(node, ast.Compare):
            operators += len(node.ops)
    literals = sum(isinstance(node, ast.Constant) for node in ast.walk(tree))
    return ExpressionStats(
        source=source,
        calls=calls,
        operators=operators,
        literals=literals,
        depth=_expression_depth(tree.body),
    )


def _node(
    kind: str,
    *,
    line: int,
    name: str | None = None,
    expression: str | None = None,
    detail: str | None = None,
    path: str,
) -> _MutableNode:
    return _MutableNode(
        kind=kind,
        name=name,
        line=line,
        expression=(
            _analyze_expression(expression, path=path, line=line)
            if expression is not None
            else None
        ),
        detail=detail,
    )


def _parse_content(content: str, *, path: str, line: int) -> _MutableNode:
    match = re.fullmatch(rf"game\s+({_GAME_ID})", content)
    if match:
        return _node("game", line=line, name=match.group(1), path=path)

    match = re.fullmatch(r"players\s+(.+)", content)
    if match:
        return _node(
            "players", line=line, expression=match.group(1), path=path
        )

    match = re.fullmatch(rf"constant\s+({_NAME})\s*=\s*(.+)", content)
    if match:
        return _node(
            "constant",
            line=line,
            name=match.group(1),
            expression=match.group(2),
            path=path,
        )

    match = re.fullmatch(
        rf"(enum|entity|action|observation|setup|terminal)\s+({_NAME}):",
        content,
    )
    if match:
        return _node(
            match.group(1), line=line, name=match.group(2), path=path
        )

    match = re.fullmatch(rf"rule\s+({_NAME})\((.*?)\):", content)
    if match:
        return _node(
            "rule",
            line=line,
            name=match.group(1),
            detail=match.group(2).strip(),
            path=path,
        )

    match = re.fullmatch(rf"value\s+({_NAME})(?:\s*=\s*(.+))?", content)
    if match:
        return _node(
            "value",
            line=line,
            name=match.group(1),
            expression=match.group(2),
            path=path,
        )

    match = re.fullmatch(
        rf"field\s+({_NAME})\s*:\s*(.+?)(?:\s*=\s*(.+))?", content
    )
    if match:
        return _node(
            "field",
            line=line,
            name=match.group(1),
            detail=match.group(2).strip(),
            expression=match.group(3),
            path=path,
        )

    match = re.fullmatch(rf"let\s+({_NAME})\s*=\s*(.+)", content)
    if match:
        return _node(
            "let",
            line=line,
            name=match.group(1),
            expression=match.group(2),
            path=path,
        )

    match = re.fullmatch(r"require\s+(.+)", content)
    if match:
        return _node(
            "require", line=line, expression=match.group(1), path=path
        )

    match = re.fullmatch(r"when\s+(.+):", content)
    if match:
        return _node(
            "when", line=line, expression=match.group(1), path=path
        )

    if content == "otherwise:":
        return _node("otherwise", line=line, path=path)

    match = re.fullmatch(rf"for\s+({_NAME})\s+in\s+(.+):", content)
    if match:
        return _node(
            "for",
            line=line,
            name=match.group(1),
            expression=match.group(2),
            path=path,
        )

    match = re.fullmatch(rf"choose\s+({_NAME})\s+from\s+(.+):", content)
    if match:
        return _node(
            "choose",
            line=line,
            name=match.group(1),
            expression=match.group(2),
            path=path,
        )

    match = re.fullmatch(r"update\s+(.+?)\s*=\s*(.+)", content)
    if match:
        return _node(
            "update",
            line=line,
            detail=match.group(1).strip(),
            expression=match.group(2),
            path=path,
        )

    for keyword in (
        "apply",
        "create",
        "delete",
        "emit",
        "reveal",
        "hide",
        "return",
    ):
        prefix = f"{keyword} "
        if content.startswith(prefix):
            return _node(
                keyword,
                line=line,
                expression=content[len(prefix) :].strip(),
                path=path,
            )

    if content == "pass":
        return _node("pass", line=line, path=path)

    raise _error(path, line, f"unsupported AB-Rule/1 syntax: {content!r}")


def parse_rule_description(
    source: str,
    *,
    path: str = "<memory>",
) -> RuleDocument:
    """Parse one AB-Rule/1 document without compiling or executing it."""

    roots: list[_MutableNode] = []
    open_blocks: list[_MutableNode] = []
    metadata: dict[str, str] = {}

    for line_number, raw_line in enumerate(source.splitlines(), start=1):
        if "\t" in raw_line:
            raise _error(path, line_number, "indentation must use two spaces")
        stripped = raw_line.strip()
        if not stripped:
            continue
        metadata_match = _METADATA.fullmatch(stripped)
        if metadata_match:
            metadata[metadata_match.group(1)] = metadata_match.group(2)
            continue
        if stripped.startswith("#"):
            continue

        indent_width = len(raw_line) - len(raw_line.lstrip(" "))
        if indent_width % 2:
            raise _error(path, line_number, "indentation must use two spaces")
        level = indent_width // 2
        if level > len(open_blocks):
            if not open_blocks:
                raise _error(path, line_number, "unexpected indentation")
            raise _error(
                path,
                line_number,
                "indentation must increase by exactly two spaces",
            )
        del open_blocks[level:]

        parsed = _parse_content(stripped, path=path, line=line_number)
        if level == 0:
            roots.append(parsed)
        else:
            if not open_blocks:
                raise _error(path, line_number, "unexpected indentation")
            open_blocks[level - 1].children.append(parsed)

        if parsed.kind in _BLOCK_KINDS:
            open_blocks.append(parsed)

    game_nodes = [node for node in roots if node.kind == "game"]
    if len(game_nodes) != 1:
        raise _error(
            path,
            1,
            "AB-Rule/1 requires exactly one game declaration",
        )

    return RuleDocument(
        game_id=game_nodes[0].name or "",
        path=path,
        metadata=metadata,
        nodes=tuple(node.freeze() for node in roots),
    )


__all__ = [
    "ExpressionStats",
    "RuleDocument",
    "RuleNode",
    "RuleSyntaxError",
    "parse_rule_description",
]
