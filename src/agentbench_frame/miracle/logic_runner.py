"""官方 logic 子进程的启动与路径解析（自己实现）。

官方逻辑（``official_logic/``，从 `AgentBench/backend_sources/corpus/24_miracle/
logic/gamecode_logic/` 原样移植，零改动）以独立子进程运行：进程内
``import main; main.Game().start()``，I/O 走 stdin/stdout 管道。

为了可复现评测，允许通过 ``-c`` 方式在子进程内 ``random.seed(...)``，
这只影响子进程运行时的随机源（地图类型/昼夜），不修改官方源码。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "OFFICIAL_LOGIC_DIR",
    "resolve_official_dir",
    "build_logic_command",
    "start_logic",
]

#: 框架内官方逻辑的默认位置
OFFICIAL_LOGIC_DIR = Path(__file__).resolve().parent / "official_logic"


def resolve_official_dir(override: Optional[str | os.PathLike] = None) -> Path:
    """返回官方逻辑目录；可通过 ``override`` 指向别的 24_miracle 逻辑包。"""
    if override is not None:
        path = Path(override).resolve()
    else:
        path = OFFICIAL_LOGIC_DIR
    if not (path / "main.py").is_file():
        raise FileNotFoundError(f"official logic not found under {path}")
    return path


def build_logic_command(
    official_dir: Path,
    seed: Optional[int] = None,
) -> list[str]:
    """构造运行官方逻辑的 argv。

    无 seed 时直接 ``python main.py``（保持最原样）；有 seed 时用
    ``python -c`` 注入 ``random.seed(seed)``（官方源码不变）。
    """
    if seed is None:
        return [sys.executable, str(official_dir / "main.py")]
    code = (
        "import random; random.seed(%d); "
        "from main import Game; Game().start()" % int(seed)
    )
    return [sys.executable, "-c", code]


def start_logic(
    official_dir: Path,
    seed: Optional[int] = None,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """以子进程启动官方 logic，返回 Popen（stdin/stdout/stderr 均为管道）。"""
    cmd = build_logic_command(official_dir, seed)
    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)
    return subprocess.Popen(
        cmd,
        cwd=str(official_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=proc_env,
    )
