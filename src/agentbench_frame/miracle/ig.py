"""信息增益（要求 4）：按统一口径计算并落盘逐 episode/iteration 的 IG 数据。

统一口径
--------
- 决策点 = Agent 在某回合 obs 下的一次决策。每个决策点记录：
  支撑集（``action_mask`` 候选）、新策略分布 ``p_new``、旧策略分布 ``p_old``。
- 严格 KL：``KL(new ‖ old) = Σ_a p_new(a) * log(p_new(a) / p_old(a))``。
- **缺失原因显式记录，绝不冒充**：
  - ``support_expansion``：存在 ``p_old(a) == 0 且 p_new(a) > 0`` 的动作
    （支撑集外扩），严格 KL 发散（= +∞），不折算为有限值；
  - ``state_mismatch``：新旧 episode 的决策点状态无法对齐（不同 obs），
    该决策点不参与聚合；
  - ``degenerate``：支撑集为空或分布未定义（概率和不等于 1 等）。
- episode 级：``episode_kl`` = 可比决策点 KL 的算术平均；
  ``missing`` = 各缺失原因计数；``coverable_ratio`` = 可比占比。
- iteration 级：汇总全部 episode 的均值与缺失统计（供 score–iteration 曲线对照）。

数据契约（对接 AgentBenchResults）
----------------------------------
``agentbench_data/ig/24_miracle/<iteration>/<episode_id>.json``：单 episode。
``agentbench_data/ig/24_miracle/ig_index.json``：iteration 级汇总。
字段名/语义与 [[docs/miracle/decision_space.md]] 的动作签名保持一致。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = [
    "DecisionPoint",
    "compute_kl",
    "distribution_from_policy",
    "episode_ig",
    "save_episode_ig",
    "save_iteration_index",
]


@dataclass(frozen=True)
class DecisionPoint:
    """一个决策点的观测与新旧策略分布。"""

    obs_hash: str                      # 观测指纹（state_mismatch 判定用）
    round: int                         # 回合号
    support: tuple[str, ...]           # 动作类型支持集（decision_space.support_set）
    p_new: dict[str, float] = field(default_factory=dict)  # action_signature -> prob
    p_old: dict[str, float] = field(default_factory=dict)


def _normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in dist.items() if v > 0}


def compute_kl(p_new: dict[str, float], p_old: dict[str, float]):
    """严格 KL(new ‖ old)。

    返回 ``(kl, missing_reason)``：
    - 有限 KL：``(float, None)``；
    - 支撑集外扩（p_old 为 0 而 p_new 为正）：``(None, "support_expansion")``；
    - 分布退化（空/非法）：``(None, "degenerate")``。
    """
    q = _normalize(p_old)
    p = _normalize(p_new)
    if not p or not q:
        return None, "degenerate"
    if not (math.isclose(sum(q.values()), 1.0) and math.isclose(sum(p.values()), 1.0)):
        return None, "degenerate"
    kl = 0.0
    for a, prob in p.items():
        qa = q.get(a, 0.0)
        if qa == 0.0:
            return None, "support_expansion"  # 严格 KL 发散，不折算
        kl += prob * math.log(prob / qa)
    if kl < 0 and math.isclose(kl, 0.0, abs_tol=1e-12):
        kl = 0.0
    return kl, None


def distribution_from_policy(
    policy: Callable[[dict, list], dict],
    obs: dict,
    support_actions: list,
) -> dict[str, float]:
    """策略函数 (obs, mask_actions) -> {signature: prob}，归一化并只保留支撑集内动作。"""
    dist = policy(obs, support_actions)
    return _normalize({k: v for k, v in dist.items()})


def episode_ig(
    decisions: list[DecisionPoint],
    *,
    episode_id: str,
    iteration: int,
    mismatched: int = 0,
) -> dict:
    """聚合一个 episode 的逐决策点 IG 数据。

    ``mismatched``：调用方判定为"新旧状态无法对齐"（不同 obs）的决策点数，
    计入 ``missing["state_mismatch"]``，不参与 KL 均值（不冒充可比数据）。
    """
    rows = []
    kl_sum = 0.0
    n_covered = 0
    missing = {"support_expansion": 0, "state_mismatch": int(mismatched), "degenerate": 0}

    for d in decisions:
        kl, reason = compute_kl(d.p_new, d.p_old)
        row = {
            "obs_hash": d.obs_hash,
            "round": d.round,
            "support": sorted(d.support),
            "p_new": {k: round(v, 6) for k, v in d.p_new.items()},
            "p_old": {k: round(v, 6) for k, v in d.p_old.items()},
            "kl_new_vs_old": None if kl is None else round(kl, 6),
            "missing_reason": reason,
        }
        rows.append(row)
        if reason is None:
            kl_sum += kl
            n_covered += 1
        else:
            missing[reason] = missing.get(reason, 0) + 1

    return {
        "episode_id": episode_id,
        "iteration": iteration,
        "n_decisions": len(decisions),
        "episode_kl": round(kl_sum / n_covered, 6) if n_covered else None,
        "coverable_ratio": round(n_covered / len(decisions), 6) if decisions else 0.0,
        "missing": missing,
        "decisions": rows,
    }


def save_episode_ig(data: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_iteration_index(index: dict, path) -> None:
    """iteration 级汇总（对接 AgentBenchResults：版本对齐的 IG–iteration 数据）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
        f.write("\n")
