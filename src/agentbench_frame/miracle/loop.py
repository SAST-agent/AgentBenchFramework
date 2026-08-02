"""迭代编排（要求 1 后半）：改策略 → 存版本 → 再评测，全程 events.jsonl 可追溯。

``run_loop`` 接收策略版本序列（AGENTS 中的名字），逐个：
1. 真实评测（``run_iteration``，落 replay/trace）；
2. 版本快照（``snapshot_version``，源码 + git commit + 评测摘要）；
3. 导出（``export_run``，AgentBenchResults 契约）；
4. 事件写入 ``events.jsonl``（时间戳 + 各步产物路径）。

所有步骤的产物（replay/trace/version/run/ig）与事件日志一一对应，可追溯。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .iterate import DATA_ROOT, AGENTS, Evaluation, export_run, run_iteration
from .versions import snapshot_version

__all__ = ["IterationEvent", "run_loop"]

EVENTS_PATH = DATA_ROOT / "events" / "24_miracle" / "iterations.jsonl"


@dataclass
class IterationEvent:
    """迭代时间线的一条事件（append-only）。"""

    ts: float
    kind: str          # evaluate | snapshot | export | ig | error
    iteration: int
    agent: str
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "ts": round(self.ts, 4),
            "kind": self.kind,
            "iteration": self.iteration,
            "agent": self.agent,
            **self.detail,
        }


def run_loop(
    agent_names: list,
    *,
    seed: int = 11,
    events_path: Optional[Path] = None,
    save: bool = True,
    agent_dir_name: str = "sample",
) -> list:
    """编排一次（可多轮）迭代：对每个版本评测 + 快照 + 导出 + 记事件。

    返回 ``list[Evaluation]``。``agent_names`` 按迭代顺序给出（如
    ``["sample", "sample_v2"]`），首个版本无 IG（基线），后续版本与前一版本
    对比计算 IG（新‖旧）。
    """
    events_path = Path(events_path or EVENTS_PATH)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    results: list = []
    prev_ig_source = None  # 上一个版本的 IG 数据（供下一轮对比）

    for i, name in enumerate(agent_names):
        try:
            ev = run_iteration(name, seed=seed, iteration=i)
        except Exception as exc:
            _append(events_path, IterationEvent(
                ts=time.time(), kind="error", iteration=i, agent=name,
                detail={"error": repr(exc)}))
            raise

        version_dir = None
        if save:
            version_dir = snapshot_version(
                name, iteration=i, seed=seed,
                match_summary={
                    "winner": ev.match.winner, "score": ev.score,
                    "rounds": ev.match.rounds, "terminated_by": ev.match.terminated_by,
                    "ig": ev.ig.get("episode_kl") if ev.ig else None,
                },
            )
            export_run(ev, agent_dir_name=agent_dir_name)

        _append(events_path, IterationEvent(
            ts=time.time(), kind="evaluate", iteration=i, agent=name,
            detail={
                "score": ev.score, "winner": ev.match.winner,
                "rounds": ev.match.rounds, "terminated_by": ev.match.terminated_by,
                "replay": ev.match.replay_path, "trace": ev.match.trace_path,
            }))
        if ev.ig:
            _append(events_path, IterationEvent(
                ts=time.time(), kind="ig", iteration=i, agent=name,
                detail={
                    "episode_kl": ev.ig.get("episode_kl"),
                    "n_decisions": ev.ig.get("n_decisions"),
                    "coverable_ratio": ev.ig.get("coverable_ratio"),
                    "missing": ev.ig.get("missing"),
                    "vs": prev_ig_source or "baseline",
                }))
        if version_dir:
            _append(events_path, IterationEvent(
                ts=time.time(), kind="snapshot", iteration=i, agent=name,
                detail={"version_dir": str(version_dir)}))
        prev_ig_source = name
        results.append(ev)
    return results


def _append(path: Path, event: IterationEvent) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_json(), ensure_ascii=False) + "\n")
