"""
Render the HL iteration curves (score–iteration and IG–iteration) from an
``events.jsonl`` research stream.

This is the plotting module the README §22 follow-up #1 called out as missing.
It is a **pure reader** of the research stream — it runs no games, probes no
agents, and never mutates the events file. Every value comes straight from the
recorded events; nothing is imputed.

Honesty rules (the measurement contract, doc §12):
- An incomplete eval records ``win_rate=None``. That iteration gets **no score
  point** (a gap on the curve), never a 0 and never a line interpolation that
  pretends the measurement existed.
- A failed act (``budget.failure_reason`` set, or a ``version`` event with
  ``failure_reason``) is drawn as a red 'x' at its iteration so it is visible,
  not hidden.
- ``policy_kl`` is only emitted for acts that have a prior version (act ≥ 2 with
  a runnable parent). Act 1 has no IG point — that is correct, not a bug.

Version alignment: every event carries ``act_id`` of the form
``<run_id>-00000<m>`` (see ``hl/naming.py``). We join ``eval`` / ``policy_kl`` /
``budget`` / ``version`` per ``act_id`` and order by the act's iteration index
``<m>``. The x-axis is the 1-based iteration number within the round.

Usage::

    python -m agentbench_frame.hl.plot_curves \\
        --events .hl_codebase/hl-v1/events.jsonl --out-dir ./figures
    # or by round name:
    python -m agentbench_frame.hl.plot_curves --name hl-v1 --out-dir ./figures
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentbench_frame.hl.events import read_events


# act_id is <run_id>-00000<m>; the trailing 6-digit field is the 1-based
# iteration index within the round (see hl/naming.py::act_name). We sort by it.
_ACT_INDEX_RE = re.compile(r"(\d+)$")


def _act_sort_key(act_id: Optional[str]) -> int:
    """Sort acts by their trailing numeric index. Acts without an index sort
    first (in encounter order, stabilized by a secondary counter)."""
    if not act_id:
        return -1
    m = _ACT_INDEX_RE.search(act_id)
    return int(m.group(1)) if m else 0


@dataclass
class IterationPoint:
    """One row of the iteration curve table — one per act."""
    iteration: int                       # 1-based act index within the round
    act_id: str
    # score axis (win_rate). None = incomplete eval (gap, never 0).
    win_rate: Optional[float] = None
    evaluation_status: Optional[str] = None
    # IG axis (policy_kl). None = no policy_kl event for this act (act 1, or
    # the eval/parent was missing). mean of the per-decision trace.
    policy_kl_mean: Optional[float] = None
    policy_kl_n: Optional[int] = None        # number of decision points in the trace
    # action-frequency KL (--action-freq) vs the previous version's full-match
    # action mix — the channel that stays live after the top branch stabilizes.
    action_kl: Optional[float] = None
    # honesty markers
    failed: bool = False                     # failure_reason on version/budget
    failure_reason: Optional[str] = None
    edit_type: Optional[str] = None


@dataclass
class CurveData:
    """The joined, ordered iteration table plus a few honest summary stats."""
    points: List[IterationPoint] = field(default_factory=list)
    n_incomplete: int = 0        # evals with win_rate None (gaps)
    n_failed: int = 0            # acts with a failure_reason

    @property
    def iterations(self) -> List[int]:
        return [p.iteration for p in self.points]


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def read_iteration_curves(events_path) -> CurveData:
    """Walk an ``events.jsonl`` and join events per ``act_id`` into iteration
    rows, ordered by the act index. Pure read — never raises on missing/None
    fields; they stay None."""
    events = read_events(events_path)

    # Buckets per act_id. The first time we see an act_id, register it; the
    # iteration index is parsed from the act_id suffix.
    by_act: Dict[str, IterationPoint] = {}
    order: List[str] = []

    def _point(act_id: str) -> IterationPoint:
        if act_id not in by_act:
            by_act[act_id] = IterationPoint(
                iteration=_act_sort_key(act_id), act_id=act_id,
            )
            order.append(act_id)
        return by_act[act_id]

    for e in events:
        et = e.get("event_type")
        act_id = e.get("act_id")
        if not act_id:
            continue
        p = _point(act_id)
        if et == "eval":
            p.win_rate = e.get("win_rate")
            p.evaluation_status = e.get("evaluation_status")
        elif et == "policy_kl":
            trace = e.get("local_policy_kl_trace") or []
            # Structured entries {kl, status, reason}: only status=="ok" values
            # count (missing/out-of-support stay out of the mean). Legacy flat
            # float entries (pre-Fix-A events.jsonl) are read as ok values.
            kl_vals = []
            for t in trace:
                if isinstance(t, dict):
                    if t.get("status") == "ok" and t.get("kl") is not None:
                        kl_vals.append(t["kl"])
                elif t is not None:
                    kl_vals.append(t)
            p.policy_kl_mean = _mean(kl_vals)
            p.policy_kl_n = len(kl_vals)
        elif et == "action_freq":
            p.action_kl = e.get("kl")
        elif et == "version":
            p.edit_type = e.get("edit_type")
            fr = e.get("failure_reason")
            if fr:
                p.failed = True
                p.failure_reason = fr
        elif et == "budget":
            fr = e.get("failure_reason")
            if fr and not p.failure_reason:
                p.failed = True
                p.failure_reason = fr

    points = sorted(by_act.values(), key=lambda q: (q.iteration, order.index(q.act_id) if q.act_id in order else 0))
    data = CurveData(points=points)
    for p in points:
        if p.win_rate is None:
            data.n_incomplete += 1
        if p.failed:
            data.n_failed += 1
    return data


# ---- rendering -----------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — never open a window
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover - environment guard
        raise SystemExit(
            "[plot_curves] matplotlib is required for rendering. "
            "Install it with:  uv sync --dev   (or: pip install matplotlib)"
        ) from e


def _annotate_honesty(ax, data: CurveData, *, what: str) -> None:
    """Stamp a caption so the figure self-documents its own honesty."""
    cap = (f"{what} | {len(data.points)} acts · "
           f"{data.n_incomplete} incomplete (gap) · "
           f"{data.n_failed} failed (✗)")
    ax.set_title(cap, fontsize=9, loc="left")


def plot_score_iteration(data: CurveData, out_path: Path) -> Path:
    """win_rate vs iteration. Incomplete evals = gap; failed acts = red ✗."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    xs, ys = [], []
    fx, fy = [], []
    for p in data.points:
        if p.win_rate is not None:
            xs.append(p.iteration)
            ys.append(p.win_rate)
        if p.failed:
            # mark the failure at y=0 (visible, but not a data point)
            fx.append(p.iteration)
            fy.append(0.0)
    if xs:
        ax.plot(xs, ys, "-o", color="tab:blue", label="win_rate")
    if fx:
        ax.scatter(fx, fy, marker="x", color="tab:red", s=60, zorder=5,
                   label="failed act")
    ax.set_xlabel("iteration (act index)")
    ax.set_ylabel("win_rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([p.iteration for p in data.points])
    ax.grid(True, alpha=0.3)
    _annotate_honesty(ax, data, what="score–iteration (win_rate)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_ig_iteration(data: CurveData, out_path: Path) -> Path:
    """policy_kl (mean over reference decisions) vs iteration. No policy_kl
    event (act 1, or missing parent) = gap."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    xs, ys = [], []
    axs, ays = [], []
    fx, fy = [], []
    for p in data.points:
        if p.policy_kl_mean is not None:
            xs.append(p.iteration)
            ys.append(p.policy_kl_mean)
        if p.action_kl is not None:
            axs.append(p.iteration)
            ays.append(p.action_kl)
        if p.failed:
            fx.append(p.iteration)
            fy.append(0.0)
    if xs:
        ax.plot(xs, ys, "-o", color="tab:green", label="policy_kl (mean)")
    if axs:
        ax.plot(axs, ays, "-s", color="tab:orange", label="action_freq KL")
    if fx:
        ax.scatter(fx, fy, marker="x", color="tab:red", s=60, zorder=5,
                   label="failed act")
    ax.set_xlabel("iteration (act index)")
    ax.set_ylabel("KL (nats)")
    # leave y auto-scaled; KL can spike to +inf on out-of-support emissions.
    ax.set_xticks([p.iteration for p in data.points])
    ax.grid(True, alpha=0.3)
    _annotate_honesty(ax, data, what="IG–iteration (KL)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot(data: CurveData, out_dir: Path) -> Dict[str, str]:
    """Render both curves into ``out_dir``. Returns {name: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "score_iteration": str(plot_score_iteration(data, out_dir / "score_iteration.png")),
        "ig_iteration": str(plot_ig_iteration(data, out_dir / "ig_iteration.png")),
    }


# ---- CLI -----------------------------------------------------------------

def _resolve_events_path(args) -> Path:
    if args.events:
        return Path(args.events)
    if args.name:
        # default layout: ./.hl_codebase/<name>/events.jsonl
        p = Path.cwd() / ".hl_codebase" / args.name / "events.jsonl"
        if p.exists():
            return p
        raise SystemExit(f"[plot_curves] no events.jsonl at {p} "
                         f"(pass --events PATH explicitly)")
    raise SystemExit("[plot_curves] pass --events PATH or --name ROUND")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m agentbench_frame.hl.plot_curves",
        description="Render score–iteration and IG–iteration curves from an "
                    "HL events.jsonl. Pure reader; honest about gaps/failures.",
    )
    p.add_argument("--events", type=Path, default=None,
                   help="path to an HL events.jsonl")
    p.add_argument("--name", default=None,
                   help="round name; resolves ./.hl_codebase/<name>/events.jsonl")
    p.add_argument("--out-dir", type=Path, default=Path("figures"),
                   help="where to write the PNGs (default: ./figures)")
    args = p.parse_args(argv)

    events_path = _resolve_events_path(args)
    if not events_path.exists():
        raise SystemExit(f"[plot_curves] events file not found: {events_path}")

    data = read_iteration_curves(events_path)
    if not data.points:
        raise SystemExit(f"[plot_curves] no acts found in {events_path}")

    out = plot(data, args.out_dir)
    # Print a compact honest table so the terminal mirrors the figure.
    print(f"[plot_curves] {len(data.points)} acts · "
          f"{data.n_incomplete} incomplete · {data.n_failed} failed")
    print(f"{'#':>3}  {'act_id':<32}  {'win_rate':>9}  {'kl_mean':>9}  "
          f"{'act_kl':>9}  {'edit':>11}  status")
    for pt in data.points:
        wr = "-" if pt.win_rate is None else f"{pt.win_rate:.3f}"
        kl = "-" if pt.policy_kl_mean is None else f"{pt.policy_kl_mean:.4f}"
        ak = "-" if pt.action_kl is None else f"{pt.action_kl:.4f}"
        et = pt.edit_type or "-"
        st = pt.evaluation_status or "-"
        if pt.failed:
            st = f"FAIL({pt.failure_reason[:16]})" if pt.failure_reason else "FAIL"
        print(f"{pt.iteration:>3}  {pt.act_id:<32}  {wr:>9}  {kl:>9}  "
              f"{ak:>9}  {et:>11}  {st}")
    print(f"\n[plot_curves] wrote:")
    for k, v in out.items():
        print(f"  {k:16} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
