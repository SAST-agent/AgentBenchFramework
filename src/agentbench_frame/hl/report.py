"""Three-group HL report classification (doc Fix-E item 3).

Groups, mirroring the zhongkaiyu ``snakego/reports/README.md`` taxonomy:

- ``control_weight_tuning`` — pure parametrize/weight edits with IG≈0. This is
  "manual RL", a control, not HL main-line structure work.
- ``hl_main`` — at least one structural edit (add_rule/reorder/refactor/
  replace/utility/search/planner/consolidate) AND an eval ever showed
  ``win_rate > 0``.
- ``rejected_hl`` — structured work landed but no eval ever showed a score
  gain (``win_rate`` pinned at 0 / None), or nothing usable (all noop / no
  versions). Kept honestly visible, never hidden.

The heuristics are documented approximations: ``edit_type`` is a byte-diff
label, not semantics; ``win_rate`` pinned at 0 makes "rejected" = "no observed
score gain". The report makes flat-``win_rate`` visible rather than pretending
an increase (doc §5 non-goal: no run has raised win_rate yet — that is exactly
the honest signal this file surfaces).

``generate()`` scans ``<experiment>/*/events.jsonl`` (one per model/run),
classifies each, reuses ``plot_curves.read_iteration_curves`` for the series,
and writes ``reports/README.md`` + ``reports/curves.json`` with the three
groups separated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentbench_frame.hl.events import read_events
from agentbench_frame.hl.plot_curves import read_iteration_curves

GROUPS = ("control_weight_tuning", "hl_main", "rejected_hl")

#: edit_type labels that count as structural HL work (vs pure weight tuning).
_STRUCTURAL = frozenset({
    "add_rule", "reorder", "refactor", "replace", "utility", "search",
    "planner", "consolidate",
})
_SKIP_LABELS = frozenset({"initial", "noop", None})


def classify_run(events_path) -> str:
    """Classify one run's events.jsonl into a group (honest heuristics)."""
    events = read_events(events_path)
    edits = [
        e.get("edit_type") for e in events
        if e.get("event_type") == "version"
    ]
    edits = [et for et in edits if et not in _SKIP_LABELS]
    wins = [e.get("win_rate") for e in events if e.get("event_type") == "eval"]
    kl_means = [e.get("kl_mean") for e in events
                if e.get("event_type") == "policy_kl"]

    if not edits:
        return "rejected_hl"  # nothing usable: all noop / no versions
    gained = any(w is not None and w > 0 for w in wins)
    if any(et in _STRUCTURAL for et in edits):
        return "hl_main" if gained else "rejected_hl"
    # parametrize-only edits: a control iff the edits moved no behavior (IG≈0).
    if not any(k is not None and k > 0 for k in kl_means):
        return "control_weight_tuning"
    return "hl_main" if gained else "rejected_hl"


def _point_to_dict(p) -> Dict[str, Any]:
    return {
        "iteration": p.iteration,
        "act_id": p.act_id,
        "win_rate": p.win_rate,
        "evaluation_status": p.evaluation_status,
        "policy_kl_mean": p.policy_kl_mean,
        "policy_kl_n": p.policy_kl_n,
        "edit_type": p.edit_type,
        "failed": p.failed,
        "failure_reason": p.failure_reason,
    }


def _group_curves(events_path) -> Dict[str, Any]:
    data = read_iteration_curves(events_path)
    return {
        "points": [_point_to_dict(p) for p in data.points],
        "n_incomplete": data.n_incomplete,
        "n_failed": data.n_failed,
    }


def generate(experiment_root) -> Path:
    """Scan ``<experiment>/*/events.jsonl`` and write the three-group report.

    Returns the ``reports/`` directory under ``experiment_root``.
    """
    exp = Path(experiment_root)
    by_group: Dict[str, Dict[str, Any]] = {g: {} for g in GROUPS}
    run_info: Dict[str, Dict[str, Any]] = {}
    for events_path in sorted(exp.glob("*/events.jsonl")):
        label = events_path.parent.name
        g = classify_run(events_path)
        by_group[g][label] = _group_curves(events_path)
        run_info[label] = _run_summary(events_path, g)

    reports = exp / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "curves.json").write_text(
        json.dumps(by_group, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (reports / "README.md").write_text(
        _render_readme(exp, run_info, by_group), encoding="utf-8")
    return reports


def _run_summary(events_path, group: str) -> Dict[str, Any]:
    events = read_events(events_path)
    wins = [e.get("win_rate") for e in events if e.get("event_type") == "eval"]
    kls = [e.get("kl_mean") for e in events if e.get("event_type") == "policy_kl"]
    edits = [e.get("edit_type") for e in events if e.get("event_type") == "version"]
    edits = [et for et in edits if et not in _SKIP_LABELS]
    n_acts = len([e for e in events if e.get("event_type") == "agent_act"])
    return {
        "group": group,
        "acts": n_acts,
        "final_win_rate": wins[-1] if wins else None,
        "kl_mean": _mean(kls),
        "edit_types": edits,
    }


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _render_readme(exp: Path, run_info: Dict[str, Dict[str, Any]],
                   by_group: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        f"# HL three-group report — {exp.name}",
        "",
        "Groups (honest heuristics, see `hl/report.py`):",
        "- **control_weight_tuning**: pure parametrize/weight edits, IG≈0 "
        "(manual-RL control, NOT HL main line).",
        "- **hl_main**: structural decision rewrite with an observed score gain.",
        "- **rejected_hl**: structured edit landed but no eval ever showed "
        "win_rate>0, or nothing usable. **Flat win_rate=0 is real and visible — "
        "it is not faked into an improvement.**",
        "",
        "| run | group | acts | final win_rate | kl_mean | edit types |",
        "|-----|-------|------|----------------|---------|------------|",
    ]
    for label, info in sorted(run_info.items()):
        fr = "-" if info["final_win_rate"] is None else f"{info['final_win_rate']:.0%}"
        km = "-" if info["kl_mean"] is None else f"{info['kl_mean']:.3g}"
        lines.append(
            f"| {label} | {info['group']} | {info['acts']} | {fr} | {km} | "
            f"{', '.join(info['edit_types']) or '-'} |")
    for g in GROUPS:
        lines.append(f"\n## {g} — {len(by_group[g])} runs")
        for label in by_group[g]:
            lines.append(f"- {label}: `{exp}/{label}/events.jsonl`")
    gained = any(i.get("final_win_rate") is not None and i["final_win_rate"] > 0
                 for i in run_info.values())
    if gained:
        lines.append(
            "\nScore gain observed in at least one run; the report keeps it "
            "visible alongside the flat-0 runs.")
    else:
        lines.append(
            "\nHonesty note: no run has yet produced a win_rate > 0 across "
            "these runs; the report keeps the flat line visible rather than "
            "hiding it. Diagnosis lives in DIAGNOSIS_AND_FIX.md §5.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m agentbench_frame.hl.report",
        description="Three-group HL report (control / hl_main / rejected_hl).")
    p.add_argument("--experiment", default=None,
                   help="experiment name under .hl_codebase/")
    p.add_argument("--root", default=None,
                   help="explicit experiment root dir (overrides "
                        ".hl_codebase/<experiment>)")
    args = p.parse_args(argv)
    if args.root:
        root = Path(args.root)
    elif args.experiment:
        root = Path.cwd() / ".hl_codebase" / args.experiment
    else:
        p.error("one of --experiment or --root is required")
    out = generate(root)
    print(f"[report] {out}")
    print(f"[report] README   : {out / 'README.md'}")
    print(f"[report] curves   : {out / 'curves.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
