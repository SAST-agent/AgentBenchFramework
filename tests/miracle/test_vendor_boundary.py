"""Pin the historical vendor and current Results aggregate boundary.

The vendor remains historical verification evidence.  The current Results
aggregate supersedes it with UTF-8 output and eight research-summary fields.
Both file identities are pinned, and function ASTs must match after removing
only those declared responsibility changes.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from agentbench_frame.games.miracle.research_protocol import (
    BENCHMARK_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    canonical_research_manifest_bytes,
    research_protocol_manifest,
)

REPO = Path(__file__).resolve().parents[2]
from agentbench_frame.games.miracle.paths import results_repo as _results_repo
UPSTREAM = (_results_repo() or Path("")) / "scripts" / "aggregate.py"
UPSTREAM_REPORT_BUILDER = UPSTREAM.parent / "report_builder.py"
VENDOR = REPO / "vendor" / "results_local" / "aggregate.py"
HISTORICAL_VENDOR_SHA256 = (
    "1c1435a4159c7c93e06e30092a7299b57f38435499bddeed9fa0db2023d8aba9"
)
CURRENT_UPSTREAM_SHA256 = (
    "19691335f7325375d443453f4066de5d843b42f2d3535c1e5edbb5c8f8337e66"
)
CURRENT_REPORT_BUILDER_SHA256 = (
    "922bb12f86f702d0f83f942ff831e295dafa58c3227aa1fc9ab952a2a9ccc90e"
)
_RESEARCH_SUMMARY_FIELDS = {
    "benchmark_score", "raw_score", "evo_score", "gain",
    "evaluation_status", "protocol_version", "benchmark_version",
    "research_manifest_sha256",
}

pytestmark = pytest.mark.skipif(
    not (UPSTREAM.exists() and UPSTREAM_REPORT_BUILDER.exists()),
    reason="upstream aggregate + report_builder not present",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_constant(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


class _DeclaredResultsEvolution(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"read_text", "write_text"}
        ):
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "encoding"]
        return node

    def visit_Try(self, node):
        node = self.generic_visit(node)
        assigned = {
            target.id
            for statement in node.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        if assigned & {"raw", "s"}:
            node.handlers = []
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        test = node.test
        if (
            isinstance(test, ast.UnaryOp)
            and isinstance(test.op, ast.Not)
            and isinstance(test.operand, ast.Call)
            and isinstance(test.operand.func, ast.Name)
            and test.operand.func.id == "isinstance"
            and test.operand.args
            and isinstance(test.operand.args[0], ast.Name)
            and test.operand.args[0].id == "s"
        ):
            return None
        return node

    def visit_Dict(self, node):
        node = self.generic_visit(node)
        pairs = [
            (key, value)
            for key, value in zip(node.keys, node.values)
            if not (
                isinstance(key, ast.Constant)
                and key.value in _RESEARCH_SUMMARY_FIELDS
            )
        ]
        node.keys = [key for key, _value in pairs]
        node.values = [value for _key, value in pairs]
        return node


def _normalized_function_asts(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    normalizer = _DeclaredResultsEvolution()
    return {
        name: ast.dump(ast.fix_missing_locations(normalizer.visit(node)))
        for name, node in functions.items()
    }


def test_current_results_aggregate_supersedes_the_historical_vendor_copy():
    assert VENDOR.exists()
    up_text = UPSTREAM.read_text(encoding="utf-8")
    vn_text = VENDOR.read_text(encoding="utf-8")
    assert 'run_dir.relative_to(data_dir).as_posix()' in up_text
    assert 'write_text("\\n".join(lines), encoding="utf-8")' in up_text
    for field in (
        "benchmark_score", "raw_score", "evo_score", "gain",
        "evaluation_status", "protocol_version", "benchmark_version",
        "research_manifest_sha256",
    ):
        assert field in up_text
    assert 'run_dir.relative_to(data_dir).as_posix()' in vn_text


def test_vendor_and_current_results_both_use_posix_paths():
    up_text = UPSTREAM.read_text(encoding="utf-8")
    vn_text = VENDOR.read_text(encoding="utf-8")
    assert '"path": run_dir.relative_to(data_dir).as_posix()' in up_text
    assert '"path": run_dir.relative_to(data_dir).as_posix()' in vn_text


def test_historical_vendor_and_current_upstream_identities_are_pinned():
    assert _sha256(VENDOR) == HISTORICAL_VENDOR_SHA256
    assert _sha256(UPSTREAM) == CURRENT_UPSTREAM_SHA256
    assert _sha256(UPSTREAM_REPORT_BUILDER) == CURRENT_REPORT_BUILDER_SHA256


def test_framework_and_results_freeze_the_same_research_versions_and_schema():
    assert _literal_constant(UPSTREAM_REPORT_BUILDER, "PROTOCOL_VERSION") == (
        PROTOCOL_VERSION
    )
    assert _literal_constant(UPSTREAM_REPORT_BUILDER, "BENCHMARK_VERSION") == (
        BENCHMARK_VERSION
    )
    assert _literal_constant(
        UPSTREAM_REPORT_BUILDER, "MANIFEST_SCHEMA_VERSION"
    ) == MANIFEST_SCHEMA_VERSION
    manifest = research_protocol_manifest()
    assert manifest["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["protocol_version"] == PROTOCOL_VERSION
    assert manifest["benchmark_version"] == BENCHMARK_VERSION
    assert canonical_research_manifest_bytes().endswith(b"\n")


def test_current_upstream_diff_is_limited_to_declared_results_responsibilities():
    assert _normalized_function_asts(UPSTREAM) == _normalized_function_asts(VENDOR)
