"""高层对局运行器（自己实现）：拼装 logic 子进程 + host + 两个 Agent。

``run_match`` 负责：
- 创建 replay 输出路径（官方 logic 会在 init 消息指定的路径写入 replay 二进制）
- 以子进程启动官方逻辑（可注入 random seed 保证可复现）
- 运行 ``MiracleHost`` 驱动整场对局，返回 ``MatchResult``
- 回收子进程、收集 stderr 尾部便于排障
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from .agent_bridge import MiracleAgent
from .host import DEFAULT_DECISION_TIMEOUT, DEFAULT_IDLE_TIMEOUT, MatchResult, MiracleHost
from .logic_runner import resolve_official_dir, start_logic

__all__ = ["run_match", "DEFAULT_REPLAY_DIR"]

#: 默认 replay 输出目录（调用方可覆盖）
DEFAULT_REPLAY_DIR = Path("agentbench_data") / "replays" / "24_miracle"


def run_match(
    agent0: MiracleAgent,
    agent1: MiracleAgent,
    *,
    replay_dir: Optional[str | os.PathLike] = None,
    trace_dir: Optional[str | os.PathLike] = None,
    seed: Optional[int] = None,
    official_dir: Optional[str | os.PathLike] = None,
    decision_timeout: float = DEFAULT_DECISION_TIMEOUT,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    tag: str = "",
) -> MatchResult:
    """跑一场完整对局。返回 MatchResult（含 replay/trace 路径）。

    参数：
    - ``replay_dir``：replay 落盘目录（默认 ``agentbench_data/replays/24_miracle``）
    - ``trace_dir``：trace jsonl 落盘目录（默认与 replay 同目录）
    - ``seed``：官方逻辑随机种子（影响地图类型/昼夜），None 则随机
    - ``official_dir``：官方逻辑目录（默认包内 official_logic/）
    """
    official = resolve_official_dir(official_dir)
    replay_dir = Path(replay_dir) if replay_dir else DEFAULT_REPLAY_DIR
    replay_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(trace_dir) if trace_dir else replay_dir

    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    fname = f"match{tag_part}_{stamp}{'_seed' + str(seed) if seed is not None else ''}.mrc"
    # 绝对路径：官方逻辑子进程的 cwd 是 official_logic，相对路径会解析错
    replay_path = str((replay_dir / fname).resolve())
    trace_path = str((trace_dir / f"{fname}.trace.jsonl").resolve())

    proc = start_logic(official, seed=seed)
    try:
        host = MiracleHost(
            proc,
            (agent0, agent1),
            decision_timeout=decision_timeout,
            idle_timeout=idle_timeout,
            trace_path=trace_path,
        )
        result = host.run(replay_path)
    finally:
        # 兜底回收子进程
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
        stderr_tail = b""
        try:
            if proc.stderr:
                stderr_tail = proc.stderr.read(4096)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        result.stderr_tail = stderr_tail.decode("utf-8", errors="replace")[-2000:]
    return result
