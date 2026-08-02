"""曲线（要求 5）：score–iteration / IG–iteration，版本对齐，SVG + 数据 JSON。

- ``build_curves``：从 AgentBenchResults 契约的 ``runs/`` 目录汇总各 iteration
  的 score / IG（版本对齐）；
- ``curves_to_svg``：渲染双曲线 SVG（无第三方依赖）；
- ``save_curves``：同时写数据 JSON 与 SVG；
- 无真实迭代数据时如实报告（``"no_data": true``），不画假曲线。

数据来源：``agentbench_data/runs/24_miracle/<agent>/<run>/summary.json``
（由 ``iterate.export_run`` 生成；``aggregate.py`` 可直接扫描同一目录）。
"""

from __future__ import annotations

import json
from pathlib import Path

from .iterate import RUNS_ROOT

__all__ = ["build_curves", "curves_to_svg", "save_curves", "curves_ascii"]


def build_curves(*, runs_root: Path = RUNS_ROOT, game: str = "24_miracle",
                 agent: str = "sample") -> dict:
    """汇总该 agent（策略族）各 iteration 的 score / IG → 版本对齐曲线数据。"""
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
    no_data = len(points) == 0
    return {
        "game": game, "agent": agent,
        "no_data": no_data,
        "note": "无真实迭代数据" if no_data else "受控演示数据（单 seed 单局）",
        "points": points,
    }


def curves_ascii(curves: dict) -> str:
    """ASCII 表（终端友好）。"""
    lines = [f"# {curves['game']} / {curves['agent']} — score & IG by iteration"]
    if curves.get("no_data"):
        lines.append("(无数据：该 agent 尚无 runs/ 导出)")
        return "\n".join(lines)
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


def curves_to_svg(curves: dict, *, width: int = 720, height: int = 320) -> str:
    """双曲线 SVG：score（蓝）与 IG（红）随 iteration 变化；无数据时如实标注。"""
    points = [p for p in curves.get("points", []) if p.get("iteration") is not None]
    if curves.get("no_data") or not points:
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"
            f"<text x='{width//2}' y='{height//2}' text-anchor='middle' "
            f"fill='#888' font-size='16'>无真实迭代数据：{curves.get('game')}/{curves.get('agent')}</text>"
            f"</svg>"
        )

    iters = sorted({p["iteration"] for p in points})
    xs = {it: 60 + i * (width - 100) / max(len(iters) - 1, 1)
          for i, it in enumerate(iters)}
    max_score = max(p["score"] or 0 for p in points) or 1
    igs = [p["episode_ig"] for p in points if isinstance(p["episode_ig"], (int, float))]
    max_ig = max(igs) if igs else 1.0

    def y_score(v):
        return height - 40 - (v / max_score) * (height - 80)

    def y_ig(v):
        return height - 40 - (v / max_ig) * (height - 80)

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<text x='20' y='25' font-size='14' fill='#333'>{curves['game']} / {curves['agent']} — score(蓝) & IG(红) by iteration</text>",
        f"<line x1='60' y1='{height-40}' x2='{width-30}' y2='{height-40}' stroke='#ccc'/>",
    ]
    # 每个 iteration 的点（同 iteration 多 seed 取首个，多 seed 在 JSON 里保留）
    by_iter = {}
    for p in points:
        by_iter.setdefault(p["iteration"], p)
    for it in iters:
        p = by_iter[it]
        x = xs[it]
        score = p.get("score")
        ig = p.get("episode_ig")
        if isinstance(score, (int, float)):
            parts.append(f"<circle cx='{x:.1f}' cy='{y_score(score):.1f}' r='4' fill='#1f77b4'/>")
            parts.append(f"<text x='{x:.1f}' y='{y_score(score)-8:.1f}' font-size='10' fill='#1f77b4'>{score}</text>")
        if isinstance(ig, (int, float)):
            parts.append(f"<circle cx='{x:.1f}' cy='{y_ig(ig):.1f}' r='4' fill='#d62728'/>")
            parts.append(f"<text x='{x:.1f}' y='{y_ig(ig)+14:.1f}' font-size='10' fill='#d62728'>{ig:.3f}</text>")
        parts.append(f"<text x='{x:.1f}' y='{height-20}' font-size='10' fill='#666' text-anchor='middle'>iter{it}</text>")
        if p.get("version"):
            parts.append(f"<text x='{x:.1f}' y='{height-8}' font-size='9' fill='#999' text-anchor='middle'>{p['version']}</text>")
    parts.append("</svg>")
    return "".join(parts)


def save_curves(curves: dict, out_json: Path, out_svg: Path) -> None:
    """曲线数据 JSON + SVG 一并落盘。"""
    out_json = Path(out_json)
    out_svg = Path(out_svg)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(curves, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    out_svg.write_text(curves_to_svg(curves), encoding="utf-8")
