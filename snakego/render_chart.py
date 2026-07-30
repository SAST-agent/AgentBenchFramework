"""Render the HL structural iteration chart from runs/curves.json.
Categories: hl_structural (main), weight_tuning_control (faded), hl_rejected (X).
"""
import json
import os
import sys


def render(curves_path, out_path):
    data = json.load(open(curves_path, encoding="utf-8"))
    pts = data["points"]
    hl_pts = [p for p in pts if p.get("category") == "hl_structural"]
    ctrl_pts = [p for p in pts if p.get("category") == "weight_tuning_control"]
    rej_pts = [p for p in pts if p.get("category") == "hl_rejected"]

    n_hl = len(hl_pts)
    x_hl = list(range(n_hl))
    x_ctrl = list(range(len(ctrl_pts)))
    x_rej = list(range(len(ctrl_pts), len(ctrl_pts) + len(rej_pts)))
    tick_labels_hl = ["HL-%d" % (i + 1) for i in range(n_hl)]

    ratios_hl = [p["territory_ratio"] for p in hl_pts]
    ratios_ctrl = [p["territory_ratio"] for p in ctrl_pts]
    ratios_rej = [p["territory_ratio"] for p in rej_pts]
    labels_hl = [p.get("label", "v%d" % p["iter"]) for p in hl_pts]
    labels_ctrl = [p.get("label", "v%d" % p["iter"]) for p in ctrl_pts]
    labels_rej = [p.get("label", "v%d" % p["iter"]) for p in rej_pts]
    ig_hl = [p.get("ig_kl") for p in hl_pts]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_ig) = plt.subplots(2, 1, figsize=(11, 9))
    fig.suptitle("snakego  HL structural iteration  (control = weight-tuning ablation)",
                 fontsize=14, fontweight="bold")

    all_r = ratios_hl + ratios_ctrl + ratios_rej
    lo = max(0, min(all_r) - 0.06)
    hi = max(max(all_r), 0.55) + 0.08
    ax_r.set_ylim(lo, hi)
    ax_r.axhline(0.5, color="#888", ls=":", lw=1.4)
    ax_r.text(max(n_hl - 1, 1) - 0.2, 0.5, "  tie", fontsize=8, color="#888", va="bottom")

    if ratios_ctrl:
        ax_r.plot(x_ctrl, ratios_ctrl, "-o", color="#cccccc", lw=1.2, ms=6,
                  alpha=0.7, zorder=2, label="control: weight-tuning (v0-v7)")
        for xi, lv in zip(x_ctrl, labels_ctrl):
            ax_r.annotate(lv, (xi, lo + 0.015), fontsize=6.5, color="#aaa",
                          ha="center", va="bottom", rotation=25)
    if ratios_rej:
        ax_r.scatter(x_rej, ratios_rej, marker="x", color="#d62728", s=90,
                     lw=2.5, zorder=5, label="rejected HL attempt")
        for xi, lv, rv in zip(x_rej, labels_rej, ratios_rej):
            ax_r.annotate(lv, (xi, rv + 0.015), fontsize=7, color="#d62728",
                          ha="center", fontweight="bold")
    if ratios_hl:
        base_r = ratios_hl[0]
        ax_r.axhline(base_r, color="#1f77b4", ls="--", lw=1.5, alpha=0.4)
        ax_r.plot(x_hl, ratios_hl, "-o", color="#1f77b4", lw=3, ms=12,
                  zorder=4, label="HL structural (main)")
        for xi, rv, lv in zip(x_hl, ratios_hl, labels_hl):
            col = "#2ca02c" if rv >= base_r else "#d62728"
            ax_r.annotate("%s\n%.3f" % (lv, rv), (xi, rv),
                          textcoords="offset points", xytext=(0, 15),
                          ha="center", fontsize=9.5, fontweight="bold", color=col)
        ax_r.scatter([x_hl[-1]], [ratios_hl[-1]], s=380, marker="*",
                     color="#ff7f0e", edgecolors="white", lw=0.8, zorder=6)
    ax_r.set_xticks(x_hl)
    ax_r.set_xticklabels(tick_labels_hl, fontsize=11, fontweight="bold")
    ax_r.set_ylabel("territory ratio  my/(my+opp)", fontsize=10)
    ax_r.set_title("territory ratio vs HL iteration   (>0.5 = winning)", fontsize=12)
    ax_r.legend(fontsize=8, loc="lower right")
    ax_r.grid(axis="y", ls=":", alpha=0.3)

    valid_ig = [(xi, v) for xi, v in zip(x_hl, ig_hl) if v is not None]
    if valid_ig:
        vx, vy = zip(*valid_ig)
        ax_ig.bar(vx, vy, width=0.5, color="#9467bd", alpha=0.85, zorder=3)
        for xi, v in zip(vx, vy):
            ax_ig.annotate("%.3f" % v, (xi, v), textcoords="offset points",
                           xytext=(0, 6), ha="center", fontsize=10,
                           fontweight="bold", color="#9467bd")
        ax_ig.set_ylim(0, max(vy) * 1.35)
    ax_ig.set_xticks(x_hl)
    ax_ig.set_xticklabels(tick_labels_hl, fontsize=11, fontweight="bold")
    ax_ig.set_ylabel("per-state eps-reg KL  (nats/decision)", fontsize=10)
    ax_ig.set_title("information gain: KL(pi_{k-1} || pi_k)  -- structural only", fontsize=12)
    ax_ig.grid(axis="y", ls=":", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    d = os.path.dirname(os.path.abspath(__file__))
    runs = os.path.join(d, "runs")
    out = os.path.join(runs, "iteration_chart.png")
    render(os.path.join(runs, "curves.json"), out)
