"""迭代编排/版本快照/CLI 测试（要求 1 后半）。

不重跑对局：版本快照与 CLI 的 replay/curves 子命令用已有产物验证。
"""

import glob
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentbench_frame.miracle.versions import VERSIONS_ROOT, list_versions, snapshot_version

MRCS = sorted(glob.glob("agentbench_data/replays/24_miracle/*.mrc"))


# ---------------------------------------------------------------------------
# versions.py
# ---------------------------------------------------------------------------

def test_snapshot_version_copies_source_and_meta(tmp_path):
    vdir = snapshot_version(
        "sample_v2", iteration=1, seed=11,
        match_summary={"score": 0, "winner": 1},
        versions_root=tmp_path,
    )
    assert (vdir / "agent_bridge.py").exists()
    assert "SampleV2Agent" in (vdir / "agent_bridge.py").read_text(encoding="utf-8")
    meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))
    assert meta["agent"] == "sample_v2"
    assert meta["iteration"] == 1
    assert meta["seed"] == 11
    assert meta["match_summary"]["score"] == 0
    assert meta["git_commit"]
    assert meta["created"]


def test_list_versions_roundtrip(tmp_path):
    snapshot_version("sample", iteration=0, versions_root=tmp_path)
    snapshot_version("sample_v2", iteration=1, seed=11, versions_root=tmp_path)
    versions = list_versions(game="24_miracle", versions_root=tmp_path)
    assert len(versions) == 2
    assert [v["iteration"] for v in versions] == [0, 1]
    assert versions[1]["agent"] == "sample_v2"


# ---------------------------------------------------------------------------
# loop.py
# ---------------------------------------------------------------------------

def test_event_append(tmp_path):
    from agentbench_frame.miracle.loop import IterationEvent, _append
    path = tmp_path / "iterations.jsonl"
    _append(path, IterationEvent(ts=1.0, kind="evaluate", iteration=0, agent="sample",
                                 detail={"score": 30000}))
    _append(path, IterationEvent(ts=2.0, kind="snapshot", iteration=0, agent="sample",
                                 detail={"version_dir": "/tmp/x"}))
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert len(rows) == 2
    assert rows[0]["kind"] == "evaluate" and rows[0]["score"] == 30000
    assert rows[1]["kind"] == "snapshot"


# ---------------------------------------------------------------------------
# cli.py（子命令可用性；有产物才跑）
# ---------------------------------------------------------------------------

def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "agentbench_frame.miracle.cli", *args],
        capture_output=True, text=True,
    )


def test_cli_replay_subcommand():
    if not MRCS:
        pytest.skip("无本地 .mrc")
    r = _cli("replay", "--path", MRCS[0])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert "event_counts" in out


def test_cli_curves_subcommand():
    runs = glob.glob("agentbench_data/runs/24_miracle/sample/*/summary.json")
    if not runs:
        pytest.skip("无导出数据，先跑 demo()")
    r = _cli("curves", "--agent", "sample")
    assert r.returncode == 0, r.stderr
    assert "score" in r.stdout and "IG" in r.stdout


def test_cli_versions_subcommand(tmp_path):
    vdir = snapshot_version("sample", iteration=0, versions_root=tmp_path)
    r = _cli("versions")  # 默认读 VERSIONS_ROOT（项目数据），不依赖 tmp_path
    assert r.returncode == 0, r.stderr
    # 至少不崩溃即可；真实版本列表由 demo/iterate 产生
    assert isinstance(r.stdout, str)
