"""
KL(step_{n+1} || step_n) overlay: one curve per model, aligned on step index.

Pure reader of each model's events.jsonl under an experiment dir. Reuses
``plot_curves.read_iteration_curves`` (and its honesty rules: a missing
policy_kl event = gap, never 0; a failed act = red x). Act 1 has no point
(no prior version to diverge from).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from agentbench_frame.hl.plot_curves import IterationPoint, read_iteration_curves


def load_experiment(experiment_dir: Path) -> Dict[str, List[IterationPoint]]:
    """Glob <experiment_dir>/*/events.jsonl -> {model_label: [IterationPoint]}."""
    experiment_dir = Path(experiment_dir)
    out: Dict[str, List[IterationPoint]] = {}
    for events_path in sorted(experiment_dir.glob("*/events.jsonl")):
        label = events_path.parent.name
        data = read_iteration_curves(events_path)
        out[label] = data.points
    return out


def _require_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_overlay(series_by_model: Dict[str, List[IterationPoint]],
                 out_path: Path) -> Path:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for i, (label, points) in enumerate(sorted(series_by_model.items())):
        color = colors[i % len(colors)]
        xs, ys = [], []
        fx = []
        for p in points:
            if p.iteration < 2:
                continue  # act 1 has no prior -> no KL point
            if p.policy_kl_mean is not None:
                xs.append(p.iteration)
                ys.append(p.policy_kl_mean)
            if p.failed:
                fx.append(p.iteration)
        if xs:
            ax.plot(xs, ys, "-o", color=color, label=label)
        if fx:
            ax.scatter(fx, [0.0] * len(fx), marker="x", color=color, s=60, zorder=5)
    ax.set_xlabel("step (act index)")
    ax.set_ylabel("KL(step_n+1 || step_n)  [nats, mean over reference decisions]")
    ax.set_title("per-step policy KL by model", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def render(*, experiment_dir: Path, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = load_experiment(experiment_dir)
    out = out_dir / "kl_overlay.png"
    plot_overlay(series, out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m agentbench_frame.hl.plot_kl_compare")
    p.add_argument("--experiment-dir", type=Path, default=None,
                   help="dir containing <model>/events.jsonl entries")
    p.add_argument("--experiment", default=None,
                   help="experiment name; resolves ./.hl_codebase/<name>")
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = p.parse_args(argv)
    exp_dir = args.experiment_dir
    if exp_dir is None:
        if not args.experiment:
            raise SystemExit("pass --experiment-dir or --experiment NAME")
        exp_dir = Path.cwd() / ".hl_codebase" / args.experiment
    if not exp_dir.is_dir():
        raise SystemExit(f"[plot_kl_compare] not a dir: {exp_dir}")
    out = render(experiment_dir=exp_dir, out_dir=args.out_dir)
    print(f"[plot_kl_compare] overlay -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
