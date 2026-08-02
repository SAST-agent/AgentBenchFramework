"""版本快照（要求 1 后半）：策略源码 + git commit + 评测摘要 + 时间戳。

每次策略更新（新版本保存）时调用 ``snapshot_version``，把当前策略源码复制到
``agentbench_data/versions/24_miracle/<iter>/`` 并写 ``meta.json``，保证
"Agent 修改 → 新版本保存"有可追溯的版本归档（与评测 run 目录互相对应）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .iterate import RUNS_ROOT

__all__ = ["VERSIONS_ROOT", "snapshot_version", "list_versions"]

VERSIONS_ROOT = RUNS_ROOT.parent / "versions"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def snapshot_version(
    agent_name: str,
    *,
    iteration: int,
    seed: Optional[int] = None,
    match_summary: Optional[dict] = None,
    source_path: Optional[Path] = None,
    versions_root: Path = VERSIONS_ROOT,
) -> Path:
    """把当前策略源码快照到 versions 目录，并写 meta.json。

    ``source_path``：策略源码文件（默认 ``agent_bridge.py``）。
    """
    src = Path(source_path or Path(__file__).resolve().parent / "agent_bridge.py")
    assert src.exists(), f"策略源码不存在: {src}"

    version_dir = versions_root / "24_miracle" / f"iter{iteration}_{agent_name}"
    version_dir.mkdir(parents=True, exist_ok=True)

    dst = version_dir / "agent_bridge.py"
    shutil.copy2(src, dst)

    meta = {
        "agent": agent_name,
        "iteration": iteration,
        "seed": seed,
        "git_commit": _git_commit(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(dst),
        "match_summary": match_summary or {},
    }
    (version_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return version_dir


def list_versions(game: str = "24_miracle",
                  versions_root: Path = VERSIONS_ROOT) -> list:
    """列出该 game 的全部版本快照（按 iteration 排序）。"""
    out = []
    base = versions_root / game
    if not base.is_dir():
        return out
    for d in sorted(base.iterdir()):
        meta = d / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["version_dir"] = str(d)
        out.append(m)
    out.sort(key=lambda m: (m.get("iteration", 0), m.get("agent", "")))
    return out
