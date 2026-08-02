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
    "mean_score_margin",
    "curriculum_event",
    "cumulative_prompt_tokens",
    "cumulative_completion_tokens",
    "cumulative_total_tokens",
)

BRANCH_FIELDS = (
    "iteration",
    "iteration_id",
    "branch_index",
    "stage",
    "version_id",
    "parent_version_id",
    "representative",
    "selected",
    "evaluation_status",
    "target_win_rate",
    "mean_score_margin",
    "mean_local_policy_kl",
)

AGGREGATE_FIELDS = (
    "global_iteration",
    "phase_iteration",
    "source_run",
    *CURVE_FIELDS,
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
    """Order-invariant Elo estimate against fixed 1500 anchors.

    The empirical score is transformed through the standard Elo logistic
    inverse.  Only exact 0/1 panels are clipped by half a game so finite values
    remain available without shrinking every smaller panel toward 1500.
    """

    if not isinstance(matches, list) or not matches:
        return None
    scores = {"win": 1.0, "draw": 0.5, "loss": 0.0}
    score = 0.0
    games = 0
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        if match.get("status", "complete") != "complete":
            continue
        result = match.get("result")
        if result not in scores:
            continue
        score += scores[str(result)]
        games += 1
    if not games:
        return None
    probability = score / games
    if probability <= 0.0:
        probability = 0.5 / (games + 1.0)
    elif probability >= 1.0:
        probability = 1.0 - 0.5 / (games + 1.0)
    return 1500.0 + 400.0 * math.log10(
        probability / (1.0 - probability)
    )


def _score_margin(matches: Any) -> float | None:
    if not isinstance(matches, list):
        return None
    margins = [
        float(match["rollman_score"]) - float(match["ghosts_score"])
        for match in matches
        if isinstance(match, Mapping)
        and match.get("status", "complete") == "complete"
        and isinstance(match.get("rollman_score"), (int, float))
        and isinstance(match.get("ghosts_score"), (int, float))
    ]
    return sum(margins) / len(margins) if margins else None


def _win_rate(evaluation: Mapping[str, Any]) -> float | None:
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
    if all(isinstance(value, int) for value in (wins, losses, draws)):
        games = wins + losses + draws
        if games:
            return (wins + 0.5 * draws) / games
    return None


def _cycle_number(value: Any, fallback: int) -> int:
    text = str(value or "")
    suffix = text.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


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
    reporting: dict[str, dict[str, Any]] = {}
    reporting_by_iteration: dict[str, dict[str, Any]] = {}
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
        elif event_type == "proposal_cycle_completed":
            selected.append(
                {
                    **event,
                    "version_id": event["selected_version_id"],
                    "act_id": versions.get(
                        str(event["selected_version_id"]), {}
                    ).get("act_id", ""),
                    "_proposal": True,
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
        elif event_type == "reporting_panel_completed":
            reporting[str(event["version_id"])] = event
            iteration_id = event.get("iteration_id")
            if iteration_id is not None:
                reporting_by_iteration[str(iteration_id)] = event
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

    proposal_iteration_ids = {
        str(item.get("iteration_id"))
        for item in selected
        if item.get("_proposal")
    }
    selected = [
        item
        for item in selected
        if item.get("_proposal")
        or str(item.get("iteration_id")) not in proposal_iteration_ids
    ]
    selected.sort(
        key=lambda item: _cycle_number(item.get("iteration_id"), 0)
    )

    rows: list[dict[str, Any]] = []
    best = None
    for ordinal, selection in enumerate(selected):
        iteration = _cycle_number(selection.get("iteration_id"), ordinal)
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
        panel = reporting_by_iteration.get(
            str(selection.get("iteration_id")),
            reporting.get(version_id, {}),
        )
        kl_trace = policy_kl.get(version_id, [])
        win_rate = _win_rate(evaluation)
        no_change = bool(
            selection.get("_proposal")
            and str(selection.get("parent_version_id")) == version_id
        )
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
                    _fixed_pool_elo(panel.get("matches"))
                    if panel.get("matches")
                    else _fixed_pool_elo(certification.get("matches"))
                    if certification.get("matches")
                    else reported_elo.get(version_id)
                ),
                "mean_local_policy_kl": (
                    0.0
                    if no_change or (iteration == 0 and not kl_trace)
                    else _mean(kl_trace)
                ),
                "_local_policy_kl_trace": kl_trace,
                "occupancy_shift": (
                    0.0 if no_change else occupancy.get(version_id)
                ),
                "active_target": target_gate.get("active_target"),
                "target_gate_score": target_gate.get("score"),
                "passing_human_opponents": certification.get(
                    "passing_human_opponents"
                ),
                "full_pool_win_rate": certification.get("score"),
                "mean_score_margin": (
                    panel.get("mean_score_margin")
                    if panel.get("mean_score_margin") is not None
                    else _score_margin(certification.get("matches"))
                ),
                "curriculum_event": curriculum_events.get(version_id),
                "cumulative_prompt_tokens": budget["prompt"],
                "cumulative_completion_tokens": budget["completion"],
                "cumulative_total_tokens": budget["total"],
            }
        )
        if panel.get("score") is not None:
            rows[-1]["full_pool_win_rate"] = float(panel["score"])
    return rows


def derive_branch_rows(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = [dict(event) for event in events]
    versions = {
        str(event["version_id"]): event
        for event in records
        if event.get("event_type") == "version_created"
    }
    evaluations = {
        str(event["version_id"]): event
        for event in records
        if event.get("event_type") == "evaluation_completed"
    }
    policy_kl = {
        str(event["version_id"]): event.get("local_policy_kl_trace")
        for event in records
        if event.get("event_type") == "policy_kl_measured"
    }
    representative_facts = {
        (str(event["iteration_id"]), int(event["branch_index"])): event
        for event in records
        if event.get("event_type") == "branch_representative_selected"
    }
    rows: list[dict[str, Any]] = []
    for fallback, event in enumerate(
        record
        for record in records
        if record.get("event_type") == "proposal_cycle_completed"
    ):
        iteration = _cycle_number(event.get("iteration_id"), fallback + 1)
        selected_id = str(event["selected_version_id"])
        for branch_index, raw_version_id in enumerate(
            event.get("candidate_version_ids", ())
        ):
            version_id = str(raw_version_id)
            fact = representative_facts.get(
                (str(event.get("iteration_id")), branch_index), {}
            )

            def append_stage(stage: str, stage_version_id: str) -> None:
                version = versions.get(stage_version_id, {})
                evaluation = evaluations.get(stage_version_id, {})
                matches = evaluation.get("matches")
                rows.append({
                    "iteration": iteration,
                    "iteration_id": event.get("iteration_id"),
                    "branch_index": branch_index,
                    "stage": stage,
                    "version_id": stage_version_id,
                    "parent_version_id": version.get("parent_version_id"),
                    "representative": (
                        str(fact.get("representative_version_id"))
                        == stage_version_id
                        if fact
                        else stage == "initial"
                    ),
                    "selected": stage_version_id == selected_id,
                    "evaluation_status": evaluation.get(
                        "status", version.get("evaluation_status")
                    ),
                    "target_win_rate": _win_rate(evaluation),
                    "mean_score_margin": _score_margin(matches),
                    "mean_local_policy_kl": _mean(
                        policy_kl.get(stage_version_id)
                    ),
                })

            append_stage("initial", version_id)
            repaired_version_id = fact.get("repaired_version_id")
            if repaired_version_id is not None:
                append_stage("repair-1", str(repaired_version_id))
    return rows


def _load_run_events(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if isinstance(source, (str, Path)):
        from agentbench_frame.hl.events import read_events

        path = Path(source).resolve()
        events_path = path / "events.jsonl" if path.is_dir() else path
        return read_events(events_path), (
            path.name if path.is_dir() else path.parent.name
        )
    records = [dict(event) for event in source]
    run_id = next(
        (
            str(event["run_id"])
            for event in records
            if event.get("run_id") is not None
        ),
        "events",
    )
    return records, run_id


def build_aggregate_curves(
    source_run: str | Path | Iterable[Mapping[str, Any]],
    phase_run: str | Path | Iterable[Mapping[str, Any]],
    *,
    global_origin_iteration: int,
) -> list[dict[str, Any]]:
    """Join a source run and imported-origin phase on one integer axis."""

    if global_origin_iteration < 0:
        raise ValueError("global_origin_iteration must be non-negative")
    source_events, source_label = _load_run_events(source_run)
    phase_events, phase_label = _load_run_events(phase_run)
    source_rows = [
        row
        for row in derive_curve_rows(source_events)
        if int(row["iteration"]) <= global_origin_iteration
    ]
    if not any(
        int(row["iteration"]) == global_origin_iteration
        for row in source_rows
    ):
        raise ValueError("source run does not contain the global origin iteration")
    rows = [
        {
            **row,
            "global_iteration": int(row["iteration"]),
            "phase_iteration": None,
            "source_run": source_label,
        }
        for row in source_rows
    ]
    for row in derive_curve_rows(phase_events):
        phase_iteration = int(row["iteration"])
        if phase_iteration == 0:
            continue
        rows.append(
            {
                **row,
                "iteration": global_origin_iteration + phase_iteration,
                "global_iteration": global_origin_iteration + phase_iteration,
                "phase_iteration": phase_iteration,
                "source_run": phase_label,
            }
        )
    rows.sort(key=lambda row: int(row["global_iteration"]))
    if len({int(row["global_iteration"]) for row in rows}) != len(rows):
        raise ValueError("aggregate curve contains duplicate global iterations")
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


def _plot(
    rows: list[Mapping[str, Any]],
    branches: list[Mapping[str, Any]],
    png: Path,
    svg: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["iteration"] for row in rows]
    figure, axes_grid = plt.subplots(
        2, 2, figsize=(14, 9), constrained_layout=True
    )
    axes = axes_grid.ravel()

    def line(axis, field, label, **kwargs):
        axis.plot(x, [row.get(field) for row in rows], marker="o", label=label, **kwargs)

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

    line(
        axes[3],
        "mean_score_margin",
        "reporting-panel mean margin",
        color="#e76f51",
    )
    axes[3].axhline(0.0, color="#555555", linewidth=1, alpha=0.5)
    axes[3].set_title("Score Margin vs HL Iteration")
    axes[3].set_ylabel("Rollman score − Ghosts score")

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
    branches = derive_branch_rows(records)
    curves_csv = output / "curves.csv"
    matches_csv = output / "matches.csv"
    curriculum_csv = output / "curriculum.csv"
    branches_csv = output / "branches.csv"
    png = output / "curves.png"
    svg = output / "curves.svg"
    _write_csv(curves_csv, rows, CURVE_FIELDS)
    _write_csv(branches_csv, branches, BRANCH_FIELDS)
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
    _plot(rows, branches, png, svg)
    return {
        "curves_csv": curves_csv,
        "matches_csv": matches_csv,
        "curriculum_csv": curriculum_csv,
        "branches_csv": branches_csv,
        "curves_png": png,
        "curves_svg": svg,
    }


def write_aggregate_report(
    source_run: str | Path | Iterable[Mapping[str, Any]],
    phase_run: str | Path | Iterable[Mapping[str, Any]],
    *,
    global_origin_iteration: int,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the joined four-panel curve in CSV, PNG, and SVG formats."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = build_aggregate_curves(
        source_run,
        phase_run,
        global_origin_iteration=global_origin_iteration,
    )
    csv_path = output / "aggregate-curves.csv"
    png = output / "aggregate-curves.png"
    svg = output / "aggregate-curves.svg"
    _write_csv(csv_path, rows, AGGREGATE_FIELDS)
    _plot(rows, [], png, svg)
    return {
        "aggregate_curves_csv": csv_path,
        "aggregate_curves_png": png,
        "aggregate_curves_svg": svg,
    }
