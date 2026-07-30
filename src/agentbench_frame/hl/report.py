"""Derive reproducible HL tables and multi-panel research curves."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


CURVE_FIELDS = (
    "iteration",
    "iteration_id",
    "coding_agent_act",
    "act_id",
    "version_id",
    "evaluation_status",
    "raw_score",
    "benchmark_score",
    "gain",
    "best_score_so_far",
    "win_rate",
    "rollman_elo",
    "mean_local_policy_kl",
    "occupancy_shift",
    "cumulative_prompt_tokens",
    "cumulative_completion_tokens",
    "cumulative_total_tokens",
)


def _mean(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    converted = [float(value) for value in values]
    if any(not math.isfinite(value) for value in converted):
        raise ValueError("KL trace contains non-finite values")
    return sum(converted) / len(converted)


def derive_curve_rows(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = [dict(event) for event in events]
    versions: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    policy_kl: dict[str, float | None] = {}
    occupancy: dict[str, float | None] = {}
    elo: dict[str, float | None] = {}
    prompt_tokens = completion_tokens = total_tokens = act_count = 0

    for event in records:
        event_type = event.get("event_type")
        if event_type == "act_completed":
            act_count += 1
            prompt_tokens += int(event.get("prompt_tokens") or 0)
            completion_tokens += int(event.get("completion_tokens") or 0)
            total_tokens += int(event.get("total_tokens") or 0)
        elif event_type == "version_created":
            versions[str(event["version_id"])] = event
        elif event_type == "candidate_selected":
            selected.append(
                {
                    **event,
                    "_budget": {
                        "coding_agent_act": act_count,
                        "prompt": prompt_tokens,
                        "completion": completion_tokens,
                        "total": total_tokens,
                    },
                }
            )
        elif event_type == "evaluation_completed":
            evaluations[str(event["version_id"])] = event
        elif event_type == "policy_kl_measured":
            policy_kl[str(event["version_id"])] = _mean(
                event.get("local_policy_kl_trace")
            )
        elif event_type == "occupancy_measured":
            value = event.get("occupancy_shift")
            occupancy[str(event["version_id"])] = (
                None if value is None else float(value)
            )
        elif event_type == "elo_updated":
            value = event.get("rating")
            elo[str(event["version_id"])] = None if value is None else float(value)

    raw_score = None
    for selection in selected:
        version = versions.get(str(selection.get("version_id")), {})
        if version.get("benchmark_score") is not None:
            raw_score = float(version["benchmark_score"])
            break

    rows: list[dict[str, Any]] = []
    best = None
    for iteration, selection in enumerate(selected):
        version_id = str(selection.get("version_id"))
        version = versions.get(version_id, {})
        act_id = str(selection.get("act_id", version.get("act_id", "")))
        budget = selection["_budget"]
        score_value = version.get("benchmark_score")
        score = None if score_value is None else float(score_value)
        if score is not None:
            best = score if best is None else max(best, score)
        evaluation = evaluations.get(version_id, {})
        wins = evaluation.get("wins")
        losses = evaluation.get("losses")
        draws = evaluation.get("draws")
        if not all(isinstance(value, int) for value in (wins, losses, draws)):
            matches = evaluation.get("matches")
            if isinstance(matches, list):
                valid_results = [
                    match.get("result")
                    for match in matches
                    if isinstance(match, Mapping)
                    and match.get("status", "complete") == "complete"
                    and match.get("result") in {"win", "draw", "loss"}
                ]
                wins = sum(result == "win" for result in valid_results)
                draws = sum(result == "draw" for result in valid_results)
                losses = sum(result == "loss" for result in valid_results)
        win_rate = None
        if all(isinstance(value, int) for value in (wins, losses, draws)):
            games = wins + losses + draws
            if games:
                win_rate = (wins + 0.5 * draws) / games
        rows.append(
            {
                "iteration": iteration,
                "iteration_id": selection.get("iteration_id"),
                "coding_agent_act": budget["coding_agent_act"],
                "act_id": act_id,
                "version_id": version_id,
                "evaluation_status": version.get("evaluation_status"),
                "raw_score": raw_score,
                "benchmark_score": score,
                "gain": (
                    score - raw_score
                    if score is not None and raw_score is not None
                    else None
                ),
                "best_score_so_far": best,
                "win_rate": win_rate,
                "rollman_elo": elo.get(version_id),
                "mean_local_policy_kl": policy_kl.get(version_id),
                "occupancy_shift": occupancy.get(version_id),
                "cumulative_prompt_tokens": budget["prompt"],
                "cumulative_completion_tokens": budget["completion"],
                "cumulative_total_tokens": budget["total"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[Mapping[str, Any]], png: Path, svg: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)

    def line(axis, field, label, **kwargs):
        axis.plot(x, [row.get(field) for row in rows], marker="o", label=label, **kwargs)

    line(axes[0, 0], "benchmark_score", "benchmark score")
    line(axes[0, 0], "best_score_so_far", "best so far", linestyle="--")
    line(axes[0, 0], "gain", "gain", alpha=0.7)
    axes[0, 0].set_title("Performance")
    axes[0, 0].legend()

    line(axes[0, 1], "rollman_elo", "Rollman Elo", color="#7b2cbf")
    axes[0, 1].set_title("Role-scoped Elo")

    line(axes[1, 0], "win_rate", "win rate", color="#00897b")
    axes[1, 0].set_title("Frozen-evaluation win rate")
    axes[1, 0].set_ylim(-0.02, 1.02)

    line(
        axes[1, 1],
        "mean_local_policy_kl",
        "mean local policy KL",
        color="#1565c0",
    )
    axes[1, 1].set_title("Behavioral information gain")
    axes[1, 1].set_ylabel("nats / decision")

    line(axes[2, 0], "occupancy_shift", "occupancy shift", color="#ef6c00")
    axes[2, 0].set_title("Occupancy shift")

    line(axes[2, 1], "cumulative_prompt_tokens", "prompt tokens")
    line(axes[2, 1], "cumulative_completion_tokens", "completion tokens")
    line(axes[2, 1], "cumulative_total_tokens", "total tokens", linestyle="--")
    axes[2, 1].set_title("Model budget")
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("HL iteration")
        axis.grid(alpha=0.25)
    figure.suptitle("AgentBench HL Research Curves", fontsize=16)
    figure.savefig(png, dpi=180)
    figure.savefig(svg)
    plt.close(figure)


def write_hl_report(
    events: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    records = [dict(event) for event in events]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = derive_curve_rows(records)
    curves_csv = output / "curves.csv"
    matches_csv = output / "matches.csv"
    png = output / "curves.png"
    svg = output / "curves.svg"
    _write_csv(curves_csv, rows, CURVE_FIELDS)
    matches = [
        event for event in records if event.get("event_type") == "match_completed"
    ]
    match_fields = sorted({key for row in matches for key in row}) or [
        "match_id",
        "version_id",
        "opponent",
        "seed",
        "result",
        "valid",
        "error",
    ]
    _write_csv(matches_csv, matches, match_fields)
    _plot(rows, png, svg)
    return {
        "curves_csv": curves_csv,
        "matches_csv": matches_csv,
        "curves_png": png,
        "curves_svg": svg,
    }
