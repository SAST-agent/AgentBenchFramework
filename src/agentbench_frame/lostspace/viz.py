"""Auto-generated visualizations for LostSpace evaluation runs.

Triggered from ``LostSpaceEvaluator.evaluate()`` right after the run's
``summary.json`` is final, so the "对战 → 回放 → 看结果" loop always produces a
visual without a separate manual step.

Best-effort by design (mirrors the ``[hl] figures`` guard in ``hl/cli.py``):
never raises, never breaks a run. Missing matplotlib → silently skipped;
set ``AGENTBENCH_NO_AUTO_VIS=1`` to disable entirely.

Writes::

    <run_dir>/figures/winrate_by_opponent.png   always, when by_opponent present
    <run_dir>/figures/score_iteration.png       HL runs only (events.jsonl has act_id)
    <run_dir>/figures/ig_iteration.png          HL runs only
    <data_dir>/_site/                           report site rebuilt (embeds figures)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")  # headless — never open a window
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - environment guard
    HAS_MATPLOTLIB = False


def _enabled() -> bool:
    return os.environ.get("AGENTBENCH_NO_AUTO_VIS", "") not in ("1", "true", "yes")


def plot_winrate_by_opponent(summary: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Bar chart of the candidate's win_rate vs each ranked opponent.

    Data source: ``summary["lostspace"]["by_opponent"][name]["win_rate"]``.
    Skips opponents whose win_rate is None (no valid games). Returns the
    written paths (empty when nothing to plot).
    """
    if not HAS_MATPLOTLIB:
        return []
    by_opp = (summary.get("lostspace") or {}).get("by_opponent")
    if not by_opp:
        return []
    names = [n for n in by_opp if by_opp[n].get("win_rate") is not None]
    if not names:
        return []
    vals = [by_opp[n]["win_rate"] for n in names]

    fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * len(names) + 2.0), 3.5))
    ax.bar(names, vals, color="tab:blue")
    ax.axhline(0.5, color="tab:gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("win_rate")
    ax.set_title(
        f"win_rate vs opponent | {summary.get('agent', '?')}",
        fontsize=9, loc="left",
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "winrate_by_opponent.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return [out_path]


def _plot_iteration_curves(run_dir: Path, out_dir: Path) -> List[Path]:
    """HL research streams only: events.jsonl rows carrying ``act_id``.

    Reuses ``hl.plot_curves`` (the existing reader/renderer). A plain eval run
    has no ``act_id`` events, so this returns nothing — that is correct.
    """
    if not HAS_MATPLOTLIB:
        return []
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    try:
        from agentbench_frame.hl import plot_curves
        data = plot_curves.read_iteration_curves(events_path)
        if not data.points:
            return []
        return [Path(p) for p in plot_curves.plot(data, out_dir).values()]
    except Exception:
        return []


def auto_visualize(run_dir: Path, data_dir: Path) -> None:
    """Generate per-run figures and rebuild the static report site.

    Args:
        run_dir: the finished run's directory (contains summary.json).
        data_dir: the AGENTBENCH_DATA root holding ``runs/`` and the site.
    """
    if not _enabled():
        return
    run_dir = Path(run_dir)
    data_dir = Path(data_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. per-run figures from the final summary + events stream
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            plot_winrate_by_opponent(summary, figures_dir)
        except Exception:
            pass
    _plot_iteration_curves(run_dir, figures_dir)

    # 2. rebuild the static site so the HTML reflects this run and embeds
    #    the figures (ReportBuilder._collect_figures copies them in).
    try:
        from agentbench_frame.report.builder import build_report
        build_report(data_dir=str(data_dir), output_dir=str(data_dir / "_site"))
    except Exception:
        pass
