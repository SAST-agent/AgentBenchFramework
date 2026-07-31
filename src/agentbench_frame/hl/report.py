"""Derive reproducible HL tables and the three primary research curves."""

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
    "active_target",
    "target_gate_score",
    "passing_human_opponents",
    "full_pool_win_rate",
    "curriculum_event",
    "cumulative_prompt_tokens",
    "cumulative_completion_tokens",
    "cumulative_total_tokens",
)

CURRICULUM_FIELDS = (
    "event_index",
    "coding_agent_act",
    "act_id",
    "version_id",
    "active_target",
    "target_gate_score",
    "locked_opponents",
    "passing_human_opponents",
    "event",
)


def _mean(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    converted = [float(value) for value in values]
    if any(not math.isfinite(value) for value in converted):
        raise ValueError("KL trace contains non-finite values")
    return sum(converted) / len(converted)


def _fixed_pool_elo(matches: Any) -> float | None:
    """Rate one version from a fresh 1500 against fixed 1500 anchors."""

    if not isinstance(matches, list) or not matches:
        return None
    scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    rating = 1500.0
    games = 0
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        if match.get("status", "complete") != "complete":
            continue
        result = match.get("result")
        if result not in scores:
            continue
        expected = 1.0 / (1.0 + 10.0 ** ((1500.0 - rating) / 400.0))
        rating += 32.0 * (scores[str(result)] - expected)
        games += 1
    return rating if games else None


def derive_curve_rows(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = [dict(event) for event in events]
    versions: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    policy_kl: dict[str, list[float]] = {}
    occupancy: dict[str, float | None] = {}
    reported_elo: dict[str, float | None] = {}
    target_gates: dict[str, dict[str, Any]] = {}
    certifications: dict[str, dict[str, Any]] = {}
    curriculum_events: dict[str, str] = {}
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
            trace = event.get("local_policy_kl_trace")
            if isinstance(trace, list):
                converted = [float(value) for value in trace]
                if any(not math.isfinite(value) for value in converted):
                    raise ValueError("KL trace contains non-finite values")
                policy_kl[str(event["version_id"])] = converted
        elif event_type == "occupancy_measured":
            value = event.get("occupancy_shift")
            occupancy[str(event["version_id"])] = (
                None if value is None else float(value)
            )
        elif event_type == "elo_updated":
            value = event.get("rating")
            reported_elo[str(event["version_id"])] = (
                None if value is None else float(value)
            )
        elif event_type == "curriculum_gate_completed":
            version_id = str(event["version_id"])
            existing = target_gates.get(version_id)
            if existing is None or (
                bool(existing.get("baseline")) and not bool(event.get("baseline"))
            ):
                target_gates[version_id] = event
            curriculum_events[version_id] = "gate"
        elif event_type == "certification_completed":
            certifications[str(event["version_id"])] = event
        elif isinstance(event_type, str) and event_type.startswith("curriculum_"):
            version_id = event.get("version_id")
            if version_id is not None:
                curriculum_events[str(version_id)] = event_type.removeprefix(
                    "curriculum_"
                )

    raw_score = None
    for selection in selected:
        version_id = str(selection.get("version_id"))
        version = versions.get(version_id, {})
        evaluation = evaluations.get(version_id, {})
        value = evaluation.get("benchmark_score", version.get("benchmark_score"))
        if value is not None:
            raw_score = float(value)
            break

    rows: list[dict[str, Any]] = []
    best = None
    for iteration, selection in enumerate(selected):
        version_id = str(selection.get("version_id"))
        version = versions.get(version_id, {})
        act_id = str(selection.get("act_id", version.get("act_id", "")))
        budget = selection["_budget"]
        evaluation = evaluations.get(version_id, {})
        score_value = evaluation.get(
            "benchmark_score", version.get("benchmark_score")
        )
        score = None if score_value is None else float(score_value)
        if score is not None:
            best = score if best is None else max(best, score)
        target_gate = target_gates.get(version_id, {})
        certification = certifications.get(version_id, {})
        kl_trace = policy_kl.get(version_id, [])
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
                "evaluation_status": evaluation.get(
                    "status", version.get("evaluation_status")
                ),
                "raw_score": raw_score,
                "benchmark_score": score,
                "gain": (
                    score - raw_score
                    if score is not None and raw_score is not None
                    else None
                ),
                "best_score_so_far": best,
                "win_rate": win_rate,
                "rollman_elo": (
                    _fixed_pool_elo(certification.get("matches"))
                    if certification.get("matches")
                    else reported_elo.get(version_id)
                ),
                "mean_local_policy_kl": _mean(kl_trace),
                "_local_policy_kl_trace": kl_trace,
                "occupancy_shift": occupancy.get(version_id),
                "active_target": target_gate.get("active_target"),
                "target_gate_score": target_gate.get("score"),
                "passing_human_opponents": certification.get(
                    "passing_human_opponents"
                ),
                "full_pool_win_rate": certification.get("score"),
                "curriculum_event": curriculum_events.get(version_id),
                "cumulative_prompt_tokens": budget["prompt"],
                "cumulative_completion_tokens": budget["completion"],
                "cumulative_total_tokens": budget["total"],
            }
        )
    return rows


def derive_curriculum_rows(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    act_count = 0
    active_target: str | None = None
    locked_opponents: set[str] = set()
    relevant_events = {
        "curriculum_started": "started",
        "curriculum_target_selected": "target_selected",
        "curriculum_gate_completed": "gate",
        "curriculum_stage_promoted": "stage_promoted",
        "curriculum_candidate_rejected": "candidate_rejected",
        "curriculum_rollback_selected": "rollback",
        "curriculum_stagnated": "stagnated",
        "curriculum_resumed": "resumed",
    }

    for event_index, source in enumerate(events):
        event = dict(source)
        event_type = event.get("event_type")
        if event_type == "act_completed":
            act_count += 1
            continue
        label = relevant_events.get(str(event_type))
        if label is None:
            continue
        if event.get("active_target") is not None:
            active_target = str(event["active_target"])
        if event_type == "curriculum_stage_promoted":
            if event.get("completed_target") is not None:
                active_target = str(event["completed_target"])
        supplied_locked = event.get("locked_opponents")
        if isinstance(supplied_locked, list):
            locked_opponents = {str(item) for item in supplied_locked}
        rows.append(
            {
                "event_index": event_index,
                "coding_agent_act": act_count,
                "act_id": event.get("act_id"),
                "version_id": event.get("version_id"),
                "active_target": active_target,
                "target_gate_score": event.get("score")
                if event_type == "curriculum_gate_completed"
                else None,
                "locked_opponents": len(locked_opponents),
                "passing_human_opponents": event.get(
                    "passing_human_opponents"
                ),
                "event": label,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _set_iteration_axis(axis: Any, iterations: list[int]) -> None:
    """Keep the learning-step axis discrete, including a one-point origin."""

    maximum = max(iterations, default=0)
    axis.set_xticks(list(range(maximum + 1)))
    axis.set_xlim(-0.5, maximum + 0.5)
    axis.set_xlabel("HL iteration")


def _plot(rows: list[Mapping[str, Any]], png: Path, svg: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)

    def line(axis, field, label, **kwargs):
        axis.plot(x, [row.get(field) for row in rows], marker="o", label=label, **kwargs)

    for iteration, row in zip(x, rows):
        trace = row.get("_local_policy_kl_trace") or []
        if trace:
            axes[0].scatter(
                [iteration] * len(trace), trace, color="#1565c0", alpha=0.18,
                s=18,
            )
    line(
        axes[0],
        "mean_local_policy_kl",
        "mean local policy KL",
        color="#1565c0",
    )
    axes[0].set_title("Information Gain vs HL Iteration")
    axes[0].set_ylabel("local policy KL (nats / decision)")

    line(axes[1], "rollman_elo", "Rollman Elo", color="#7b2cbf")
    axes[1].set_title("Elo vs HL Iteration")
    axes[1].set_ylabel("fixed-pool Elo")

    line(axes[2], "full_pool_win_rate", "full human pool", color="#00897b")
    axes[2].set_title("Full-pool Win Rate vs HL Iteration")
    axes[2].set_ylabel("win rate")
    axes[2].set_ylim(-0.02, 1.02)

    for axis in axes:
        _set_iteration_axis(axis, x)
        axis.grid(alpha=0.25)
    figure.suptitle("AgentBench HL Learning Curves", fontsize=15)
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
    curriculum_csv = output / "curriculum.csv"
    png = output / "curves.png"
    svg = output / "curves.svg"
    _write_csv(curves_csv, rows, CURVE_FIELDS)
    _write_csv(
        curriculum_csv,
        derive_curriculum_rows(records),
        CURRICULUM_FIELDS,
    )
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
        "curriculum_csv": curriculum_csv,
        "curves_png": png,
        "curves_svg": svg,
    }
