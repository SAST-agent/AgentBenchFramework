"""迭代闭环（要求 1 后半 + 要求 5）：评测 → IG → 版本保存 → 曲线。

管线
----
1. ``evaluate``：用 ``run_match`` 跑一局真实对局，得 ``MatchResult`` 与 trace；
2. ``collect_decision_points``：从 trace 提取逐决策点 obs，用新旧两个策略版本
   离线决策，构造 ε-soft 分布（1-ε 最优 + ε 均匀），得 ``DecisionPoint`` 列表；
3. ``episode_ig``（ig.py）：同 obs 对齐的严格 KL(new‖old)，缺失如实记录；
4. ``export_run``：按 AgentBenchResults 契约写
   ``runs/24_miracle/<agent>/<iter_id>/run.toml + summary.json``；
5. ``build_curves``：汇总各 iteration 的 score / IG → 版本对齐的曲线数据
   （JSON + ASCII），供 AgentBenchResults 可视化对接。

口径说明
--------
- score–iteration 的 score = 本策略作为 camp0 的官方 score（神迹 HP 或 30000）。
- IG–iteration 的 IG = episode 级 KL(新‖旧)（可比决策点均值）。
- 本轮为**受控演示**：v1=SampleAgent、v2=SampleV2Agent（召唤优先级改动），
  各真实评测 1 局；完整多轮闭环由后续迭代填充（数据契约已就绪）。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .agent_bridge import SampleAgent, SampleV2Agent
from .decision_space import Action, action_mask
from .ig import DecisionPoint, episode_ig
from .match import run_match

__all__ = [
    "AGENTS", "evaluate", "collect_decision_points", "epsilon_soft",
    "run_iteration", "export_run", "build_curves", "demo",
]

AGENTS = {
    "sample": SampleAgent,
    "sample_v2": SampleV2Agent,
}

DATA_ROOT = Path(__file__).resolve().parents[3] / "agentbench_data"
RUNS_ROOT = DATA_ROOT / "runs"


@dataclass
class Evaluation:
    """一局真实评测的产物。"""

    iteration: int
    version: str
    agent_name: str
    seed: int
    match: object          # MatchResult
    decisions: list = field(default_factory=list)
    ig: dict = field(default_factory=dict)

    @property
    def score(self) -> int:
        return int(self.match.scores[0])  # camp0（本策略）官方 score


def evaluate(agent, *, seed: int, opponent_name: str = "endround",
             iteration: int = 0, version: str = "", tag: str = "") -> Evaluation:
    """跑一局真实对局（要求 1 后半：再评测的原子单元）。"""
    from .agent_bridge import EndRoundAgent
    opponents = {"endround": EndRoundAgent}
    opp = opponents[opponent_name]()
    match = run_match(
        agent, opp, seed=seed,
        tag=tag or f"iter{iteration}_{version or agent.name}",
    )
    return Evaluation(
        iteration=iteration, version=version or agent.name,
        agent_name=agent.name, seed=seed, match=match,
    )


def _obs_from_trace(trace_path) -> list:
    """从 trace 提取逐决策点 obs。

    官方 from_logic 帧的 state 字段是 **round 号**（state=0 为心跳/超时帧、
    1/2 为 camp 通知），完整 obs 帧的 content 解析后含 ``map`` 键——按内容过滤。
    """
    obs_list = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") != "from_logic":
                continue
            content = row.get("payload", {}).get("content")
            if not content:
                continue
            body = content[0]
            if len(body) < 6:
                continue
            try:
                obs = json.loads(body[6:])
            except Exception:
                continue
            if not isinstance(obs, dict) or "map" not in obs:
                continue
            obs["camp"] = int(obs.get("camp", 0))
            obs_list.append(obs)
    return obs_list


def epsilon_soft(chosen: dict, mask: list, eps: float = 0.1) -> dict:
    """把一次确定性选择平滑为 ε-soft 分布：1-ε 给所选动作，ε 均匀分给 mask 其余。

    确定性策略直接 one-hot 会使跨策略 KL 因支撑集不相交而全部发散；
    ε-soft 是 RL 常见的策略平滑口径，保证 KL 有限可算（缺失仍如实记录）。
    """
    chosen_sig = Action(chosen["operation_type"], chosen.get("operation_parameters", {})).signature()
    sigs = [Action(a.type, a.params).signature() for a in mask]
    if not sigs:
        return {}
    n = len(sigs)
    if n == 1:
        return {sigs[0]: 1.0}
    if chosen_sig not in sigs:
        # 所选动作不在 mask（被拒/异常）→ 退化为均匀
        return {s: 1.0 / n for s in sigs}
    dist = {}
    for s in sigs:
        dist[s] = (1.0 - eps) if s == chosen_sig else eps / (n - 1)
    return dist


def collect_decision_points(trace_path, v1, v2, *, eps: float = 0.1) -> list:
    """在同一 obs 流上分别用 v1/v2 决策，构造新旧分布对齐的决策点。"""
    decisions = []
    for obs in _obs_from_trace(trace_path):
        mask = action_mask(obs, camp=obs.get("camp", 0))
        if not mask:
            continue
        a1 = v1.act(obs)
        a2 = v2.act(obs)
        decisions.append(DecisionPoint(
            obs_hash=str(hash(json.dumps(obs, sort_keys=True)) % (2 ** 31)),
            round=int(obs.get("round", 0)),
            support=tuple(sorted({a.type for a in mask})),
            p_new=epsilon_soft(a2, mask, eps=eps),   # 新版本
            p_old=epsilon_soft(a1, mask, eps=eps),   # 旧版本
        ))
    return decisions


def run_iteration(agent_name: str, *, seed: int, iteration: int,
                  opponent_name: str = "endround") -> Evaluation:
    """一次迭代：真实评测 + （如有新旧版本）IG 计算。

    返回 Evaluation（match/decisions/ig 就绪）。"""
    agent_cls = AGENTS[agent_name]
    v1_name, v2_name = "sample", agent_name  # v1 基线；IG 是新版本 vs 基线
    ev = evaluate(agent_cls(), seed=seed, iteration=iteration,
                  version=agent_name, opponent_name=opponent_name)

    if agent_name != "sample":
        trace_path = ev.match.trace_path
        v1 = AGENTS[v1_name](seed=seed)
        v2 = AGENTS[v2_name](seed=seed)
        decisions = collect_decision_points(trace_path, v1, v2)
        ev.decisions = decisions
        ev.ig = episode_ig(
            decisions, episode_id=Path(trace_path).stem, iteration=iteration,
        )
    return ev


def export_run(ev: Evaluation, *, runs_root: Path = RUNS_ROOT,
               agent_dir_name: Optional[str] = None) -> Path:
    """按 AgentBenchResults 契约导出 run.toml + summary.json。

    ``agent_dir_name``：策略族目录名（默认 ``ev.agent_name``）。版本对齐的
    score/IG 曲线按"同一策略族 + 不同 version"组织，demo 中 v1/v2 都归入
    ``sample`` 族。"""
    agent_dir = agent_dir_name or ev.agent_name
    run_id = f"iter{ev.iteration}_{ev.version}_seed{ev.seed}"
    run_dir = runs_root / "24_miracle" / agent_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        commit = "unknown"
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    run_toml = {
        "run": {
            "type": "eval",
            "created": created,
            "git_commit": commit,
            "agent_version": ev.version,
            "iteration": ev.iteration,
            "seed": ev.seed,
            "winner": ev.match.winner,
            "rounds": ev.match.rounds,
            "terminated_by": ev.match.terminated_by,
        },
    }
    summary = {
        "run_type": "eval",
        "created": created,
        "git_commit": commit,
        "game": "24_miracle",
        "agent": ev.agent_name,
        "agent_version": ev.version,
        "iteration": ev.iteration,
        "seed": ev.seed,
        "winner": ev.match.winner,
        "win_rate": 1.0 if ev.match.winner == 0 else 0.0,
        "score": ev.score,
        "scores": list(ev.match.scores),
        "rounds": ev.match.rounds,
        "terminated_by": ev.match.terminated_by,
        "errors": list(ev.match.errors),
        "episode_ig": ev.ig.get("episode_kl") if ev.ig else None,
        "ig_coverable_ratio": ev.ig.get("coverable_ratio") if ev.ig else None,
        "ig_missing": ev.ig.get("missing") if ev.ig else {},
        "replay": str(ev.match.replay_path),
        "trace": str(ev.match.trace_path),
    }
    (run_dir / "run.toml").write_text(
        "type = \"eval\"\n"
        f"created = \"{created}\"\n"
        f"git_commit = \"{commit}\"\n"
        f"agent_version = \"{ev.version}\"\n"
        f"iteration = {ev.iteration}\n"
        f"seed = {ev.seed}\n"
        f"winner = {ev.match.winner}\n"
        f"rounds = {ev.match.rounds}\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if ev.ig:
        (run_dir / "ig.json").write_text(
            json.dumps(ev.ig, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return run_dir


def build_curves(*, runs_root: Path = RUNS_ROOT, game: str = "24_miracle",
                 agent: str = "sample") -> dict:
    """汇总该 agent 各 iteration 的 score / IG → 版本对齐曲线数据。"""
    points = []
    base = runs_root / game / agent
    if base.is_dir():
        for run_dir in sorted(base.iterdir()):
            sj = run_dir / "summary.json"
            if not sj.exists():
                continue
            s = json.loads(sj.read_text(encoding="utf-8"))
            points.append({
                "iteration": s.get("iteration", 0),
                "version": s.get("agent_version", ""),
                "seed": s.get("seed"),
                "score": s.get("score"),
                "winner": s.get("winner"),
                "episode_ig": s.get("episode_ig"),
                "ig_coverable_ratio": s.get("ig_coverable_ratio"),
                "ig_missing": s.get("ig_missing", {}),
            })
    points.sort(key=lambda p: (p["iteration"], str(p["seed"])))
    return {"game": game, "agent": agent, "points": points}


def curves_ascii(curves: dict) -> str:
    """把曲线数据画成 ASCII 表（无需 matplotlib 依赖）。"""
    lines = [f"# {curves['game']} / {curves['agent']} — score & IG by iteration"]
    lines.append("iteration | version   | seed | score | ig(episode) | coverable | missing")
    lines.append("-" * 78)
    for p in curves["points"]:
        ig = p["episode_ig"]
        ig_s = f"{ig:.4f}" if isinstance(ig, (int, float)) else "n/a"
        cov = p["ig_coverable_ratio"]
        cov_s = f"{cov:.2f}" if isinstance(cov, (int, float)) else "n/a"
        lines.append(
            f"{p['iteration']:<9} | {p['version']:<10} | {str(p['seed']):<4} | "
            f"{str(p['score']):<5} | {ig_s:<11} | {cov_s:<7} | {p.get('ig_missing', {})}"
        )
    return "\n".join(lines)


def demo(*, seed: int = 11, save: bool = True) -> dict:
    """受控演示：v1(sample) 与 v2(sample_v2) 各真实评测 1 局 + IG + 导出 + 曲线。

    注意：这是**单 seed 单局演示**（非完整多轮迭代）；数据如实标注。
    """
    ev1 = run_iteration("sample", seed=seed, iteration=0)
    ev2 = run_iteration("sample_v2", seed=seed, iteration=1)
    if save:
        export_run(ev1, agent_dir_name="sample")
        export_run(ev2, agent_dir_name="sample")
    curves = build_curves(agent="sample")
    return {
        "ev1": {"version": ev1.version, "score": ev1.score, "winner": ev1.match.winner,
                "rounds": ev1.match.rounds},
        "ev2": {"version": ev2.version, "score": ev2.score, "winner": ev2.match.winner,
                "rounds": ev2.match.rounds,
                "ig": ev2.ig.get("episode_kl"),
                "ig_missing": ev2.ig.get("missing"),
                "ig_coverable": ev2.ig.get("coverable_ratio"),
                "n_decisions": ev2.ig.get("n_decisions")},
        "curves": curves,
    }
