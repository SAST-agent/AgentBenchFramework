"""vendor aggregate boundary check (阶段4b-8 / spec #11).

Proves the vendored ``vendor/results_local/aggregate.py`` differs from the
upstream AgentBenchResults ``aggregate.py`` ONLY by the Windows path-separator
patch (and the explanatory header). Specifically: after reverting the single
``as_posix()`` change, the AST of every function (find_runs / _toml_val / main)
is byte-identical to upstream — so metrics, h2h, and run-discovery logic are
unchanged. Also pins the upstream SHA256 so drift is detected.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
from agentbench_frame.games.miracle.paths import results_repo as _results_repo
UPSTREAM = (_results_repo() or Path("")) / "scripts" / "aggregate.py"
VENDOR = REPO / "vendor" / "results_local" / "aggregate.py"
UPSTREAM_SHA256 = "126796cd3626deb814fff3efef82f64844954d11a7817ae43c17e58dd3667411"

pytestmark = pytest.mark.skipif(not UPSTREAM.exists(), reason="upstream aggregate not present")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _funcs_from_text(text: str):
    tree = ast.parse(text)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_upstream_aggregate_sha_pinned():
    assert _sha256(UPSTREAM) == UPSTREAM_SHA256


def test_vendor_only_differs_by_path_separator():
    assert VENDOR.exists()
    up_text = UPSTREAM.read_text(encoding="utf-8")
    vn_text = VENDOR.read_text(encoding="utf-8")
    # revert the one patched line at the SOURCE level, then every function's AST
    # must match upstream byte-for-byte (metrics / h2h / discovery unchanged)
    vn_text_norm = vn_text.replace(
        '"path": run_dir.relative_to(data_dir).as_posix(),',
        '"path": str(run_dir.relative_to(data_dir)),',
    )
    assert vn_text_norm != vn_text, "vendor path line not found / already native"
    up = _funcs_from_text(up_text)
    vn = _funcs_from_text(vn_text_norm)
    assert set(up) == set(vn), "function set differs"
    for name, dump in up.items():
        assert dump == vn[name], f"function {name} differs beyond the path patch"


def test_vendor_path_uses_posix_and_upstream_uses_native():
    up_text = UPSTREAM.read_text(encoding="utf-8")
    vn_text = VENDOR.read_text(encoding="utf-8")
    assert '"path": str(run_dir.relative_to(data_dir))' in up_text
    assert '"path": run_dir.relative_to(data_dir).as_posix()' in vn_text
