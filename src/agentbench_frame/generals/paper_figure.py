"""Publication figure projection for controlled-reference Generals policy KL."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


EXPECTED_VERSION_PAIRS_BY_MEASUREMENT = {
    "generals-policy-kl-reference-v1": (
        ("v0", "v1"),
        ("v1", "v2"),
        ("v2", "v3"),
        ("v3", "v4"),
        ("v4", "v5"),
        ("v5", "v6"),
    ),
    "generals-policy-kl-reference-v2": (
        ("v0", "v1"),
        ("v1", "v2"),
        ("v2", "v3"),
        ("v3", "v4"),
        ("v4", "v5"),
        ("v5", "v6"),
        ("v6", "v7"),
    ),
}
EXPECTED_EPSILONS = ("0.001", "0.01", "0.05", "0.1")
REFERENCE_STATE_COUNT = 12


@dataclass(frozen=True)
class SupportStatePoint:
    state_id: str
    seed: int
    seat: int
    decision_number: int
    support_size: int | None
    status: str


@dataclass(frozen=True)
class PolicyKLFigureData:
    run_id: str
    transitions: tuple[str, ...]
    primary_epsilon: str
    primary_kl: tuple[float | None, ...]
    sensitivity: dict[str, tuple[float | None, ...]]
    support_states: tuple[SupportStatePoint, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _coverage_value(
    record: dict[str, Any],
    *,
    label: str,
) -> float | None:
    coverage = record.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"{label} coverage must be an object")
    complete = coverage.get("complete")
    total = coverage.get("total")
    if (
        type(complete) is not int
        or type(total) is not int
        or total != REFERENCE_STATE_COUNT
        or not 0 <= complete <= total
    ):
        raise ValueError(f"{label} coverage is invalid")
    value = record.get("mean_kl_nats")
    if complete == total:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{label} complete coverage requires a numeric aggregate"
            )
        return float(value)
    if value is not None:
        raise ValueError(
            f"{label} incomplete coverage requires a null aggregate"
        )
    return None


def _load_support_states(events_path: Path) -> tuple[SupportStatePoint, ...]:
    records: dict[str, SupportStatePoint] = {}
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read event artifact {events_path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed event at line {line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise ValueError(f"event at line {line_number} must be an object")
        event_type = event.get("event_type", event.get("event"))
        if event_type != "action_space_count":
            continue
        state_id = event.get("measurement_state_id")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError("action_space_count is missing measurement_state_id")
        if state_id in records:
            raise ValueError(
                f"duplicate measurement_state_id: {state_id}"
            )
        seed = event.get("seed")
        seat = event.get("seat")
        decision = event.get("decision_number")
        if (
            type(seed) is not int
            or type(seat) is not int
            or seat not in (0, 1)
            or type(decision) is not int
        ):
            raise ValueError(
                f"invalid reference coordinates for state {state_id}"
            )
        status = event.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"missing count status for state {state_id}")
        raw_support = event.get("support_size")
        if status == "complete":
            if (
                not isinstance(raw_support, str)
                or not raw_support.isdecimal()
                or int(raw_support) < 1
            ):
                raise ValueError(
                    f"complete count requires positive support for {state_id}"
                )
            support_size = int(raw_support)
        else:
            if raw_support is not None:
                raise ValueError(
                    f"incomplete count requires null support for {state_id}"
                )
            support_size = None
        records[state_id] = SupportStatePoint(
            state_id=state_id,
            seed=seed,
            seat=seat,
            decision_number=decision,
            support_size=support_size,
            status=status,
        )
    if len(records) != REFERENCE_STATE_COUNT:
        raise ValueError(
            "expected exactly 12 unique action_space_count events"
        )
    return tuple(
        sorted(
            records.values(),
            key=lambda item: (
                item.decision_number,
                item.seed,
                item.seat,
            ),
        )
    )


def load_policy_kl_figure_data(run_dir: Path) -> PolicyKLFigureData:
    """Load and validate the first-hand inputs for the paper figure."""

    run_dir = Path(run_dir)
    summary = _read_json(run_dir / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError("figure source run status must be complete")
    measurement_id = summary.get("measurement_id")
    expected_pairs = EXPECTED_VERSION_PAIRS_BY_MEASUREMENT.get(
        measurement_id
    )
    if expected_pairs is None:
        raise ValueError("unsupported policy KL measurement_id")
    metric = summary.get("controlled_reference_policy_kl")
    if not isinstance(metric, dict):
        raise ValueError("controlled_reference_policy_kl metric is missing")
    if metric.get("metric") != "controlled_reference_policy_kl":
        raise ValueError("metric must be controlled_reference_policy_kl")
    if metric.get("primary_epsilon") != "0.01":
        raise ValueError("primary epsilon must be 0.01")
    if tuple(metric.get("epsilons") or ()) != EXPECTED_EPSILONS:
        raise ValueError("epsilon order must match the frozen specification")
    if metric.get("reference_state_count") != REFERENCE_STATE_COUNT:
        raise ValueError("reference state count must be 12")

    transitions = metric.get("transitions")
    if not isinstance(transitions, list):
        raise ValueError("transitions must be an array")
    actual_pairs = tuple(
        (
            item.get("version_before"),
            item.get("version_after"),
        )
        for item in transitions
        if isinstance(item, dict)
    )
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"transition order does not match {measurement_id}"
        )

    primary_values = []
    sensitivity_values = {
        epsilon: [] for epsilon in EXPECTED_EPSILONS
    }
    for transition, (before, after) in zip(
        transitions,
        expected_pairs,
    ):
        if not isinstance(transition, dict):
            raise ValueError("each transition must be an object")
        label = f"{before}→{after}"
        primary = _coverage_value(transition, label=label)
        sensitivity = transition.get("sensitivity")
        if not isinstance(sensitivity, dict):
            raise ValueError(f"{label} sensitivity must be an object")
        if tuple(sensitivity) != EXPECTED_EPSILONS:
            raise ValueError(f"{label} sensitivity epsilon order changed")
        for epsilon in EXPECTED_EPSILONS:
            item = sensitivity.get(epsilon)
            if not isinstance(item, dict):
                raise ValueError(
                    f"{label} epsilon {epsilon} must be an object"
                )
            sensitivity_values[epsilon].append(
                _coverage_value(
                    item,
                    label=f"{label} epsilon {epsilon}",
                )
            )
        if sensitivity_values["0.01"][-1] != primary:
            raise ValueError(
                f"{label} primary aggregate differs from epsilon 0.01"
            )
        primary_values.append(primary)

    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    return PolicyKLFigureData(
        run_id=run_id,
        transitions=tuple(
            f"{before}→{after}" for before, after in expected_pairs
        ),
        primary_epsilon="0.01",
        primary_kl=tuple(primary_values),
        sensitivity={
            epsilon: tuple(values)
            for epsilon, values in sensitivity_values.items()
        },
        support_states=_load_support_states(run_dir / "events.jsonl"),
    )


def render_policy_kl_three_panel(
    data: PolicyKLFigureData,
    output_prefix: Path,
) -> tuple[Path, Path]:
    """Render the publication figure as editable SVG and 300 DPI PNG."""

    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_config = (
            Path(tempfile.gettempdir()) / "agentbench-matplotlib"
        )
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_config)

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import LogLocator, NullFormatter
    except ImportError as exc:
        raise RuntimeError(
            "paper figure rendering requires the 'figures' extra"
        ) from exc

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_prefix.with_suffix(".svg")
    png_path = output_prefix.with_suffix(".png")

    colors = {
        "blue": "#2A6FBB",
        "navy": "#174A7E",
        "orange": "#F4A261",
        "deep_orange": "#E76F51",
        "green": "#2A9D8F",
        "purple": "#8F5DA2",
        "grey": "#59636E",
        "grid": "#D8DEE5",
        "text": "#20262E",
    }
    epsilon_styles = {
        "0.001": (colors["navy"], "o"),
        "0.01": (colors["deep_orange"], "s"),
        "0.05": (colors["green"], "^"),
        "0.1": (colors["purple"], "D"),
    }
    rc = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": colors["grey"],
        "axes.labelcolor": colors["text"],
        "axes.titlecolor": colors["text"],
        "text.color": colors["text"],
        "xtick.color": colors["text"],
        "ytick.color": colors["text"],
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "svg.fonttype": "none",
    }

    with matplotlib.rc_context(rc):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(24, 7.2),
            gridspec_kw={"width_ratios": (1.0, 1.0, 1.18)},
            layout="constrained",
        )
        x_values = list(range(len(data.transitions)))

        primary_axis = axes[0]
        primary_axis.plot(
            x_values,
            [
                math.nan if value is None else value
                for value in data.primary_kl
            ],
            color=colors["blue"],
            linewidth=2.8,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=2.2,
            zorder=3,
        )
        primary_axis.set_title(
            "Controlled-reference Policy KL over Iterations",
            pad=14,
        )
        primary_axis.set_ylabel("Mean KL (nats / state)")
        primary_axis.set_xlabel("Policy transition")
        primary_axis.set_xticks(x_values, data.transitions, rotation=28, ha="right")
        primary_complete = [
            value for value in data.primary_kl if value is not None
        ]
        primary_max = max(primary_complete, default=1.0)
        primary_axis.set_ylim(0, max(1.0, primary_max * 1.24))
        for x_value, value in zip(x_values, data.primary_kl):
            if value is None:
                continue
            primary_axis.annotate(
                f"{value:.2f}",
                (x_value, value),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=colors["navy"],
                fontweight="bold",
            )
        primary_axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=colors["blue"],
                    marker="o",
                    markerfacecolor="white",
                    markeredgewidth=1.8,
                    linewidth=2.4,
                    label=f"epsilon = {data.primary_epsilon}",
                )
            ],
            loc="upper left",
            frameon=False,
        )

        sensitivity_axis = axes[1]
        all_sensitivity_values = []
        for epsilon in EXPECTED_EPSILONS:
            values = data.sensitivity[epsilon]
            color, marker = epsilon_styles[epsilon]
            sensitivity_axis.plot(
                x_values,
                [
                    math.nan if value is None else value
                    for value in values
                ],
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=6,
                label=f"epsilon = {epsilon}",
            )
            all_sensitivity_values.extend(
                value for value in values if value is not None
            )
        sensitivity_axis.set_title("Epsilon Sensitivity", pad=14)
        sensitivity_axis.set_ylabel("Mean KL (nats / state)")
        sensitivity_axis.set_xlabel("Policy transition")
        sensitivity_axis.set_xticks(
            x_values,
            data.transitions,
            rotation=28,
            ha="right",
        )
        sensitivity_axis.set_ylim(
            0,
            max(1.0, max(all_sensitivity_values, default=1.0) * 1.18),
        )
        sensitivity_axis.legend(
            loc="upper left",
            frameon=False,
            ncols=2,
            columnspacing=1.2,
            handlelength=2.4,
        )

        support_axis = axes[2]
        support_x = list(range(len(data.support_states)))
        support_values = [
            point.support_size if point.support_size is not None else 1
            for point in data.support_states
        ]
        support_colors = [
            (
                colors["orange"]
                if point.decision_number == 2
                else colors["deep_orange"]
            )
            for point in data.support_states
        ]
        bars = support_axis.bar(
            support_x,
            support_values,
            color=support_colors,
            edgecolor="white",
            linewidth=0.8,
            width=0.76,
            zorder=3,
        )
        for bar, point in zip(bars, data.support_states):
            if point.support_size is None:
                bar.set_height(1)
                bar.set_facecolor("white")
                bar.set_edgecolor(colors["grey"])
                bar.set_hatch("///")
                support_axis.annotate(
                    "missing",
                    (bar.get_x() + bar.get_width() / 2, 1.35),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=colors["grey"],
                )
                continue
            support_axis.annotate(
                f"{point.support_size:,}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    point.support_size * 1.12,
                ),
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=colors["text"],
            )
        support_axis.set_yscale("log")
        complete_support = [
            point.support_size
            for point in data.support_states
            if point.support_size is not None
        ]
        support_axis.set_ylim(
            1 if len(complete_support) < len(data.support_states) else 5,
            max(complete_support, default=10) * 3.2,
        )
        support_axis.yaxis.set_major_locator(LogLocator(base=10))
        support_axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=(2, 5))
        )
        support_axis.yaxis.set_minor_formatter(NullFormatter())
        support_axis.set_title("Exact Canonical Support Size", pad=14)
        support_axis.set_ylabel("Exact |A(s)| (log scale)")
        support_axis.set_xlabel("Reference state")
        support_axis.set_xticks(
            support_x,
            [
                (
                    f"d{point.decision_number}\n"
                    f"s{point.seed % 1000:03d} · p{point.seat}"
                )
                for point in data.support_states
            ],
            fontsize=8.5,
        )
        support_axis.legend(
            handles=[
                Patch(
                    facecolor=colors["orange"],
                    edgecolor="none",
                    label="Decision 2",
                ),
                Patch(
                    facecolor=colors["deep_orange"],
                    edgecolor="none",
                    label="Decision 10",
                ),
            ],
            loc="upper left",
            frameon=False,
            ncols=2,
        )

        for panel, axis in zip(("a", "b", "c"), axes):
            axis.text(
                -0.10,
                1.06,
                f"({panel})",
                transform=axis.transAxes,
                fontsize=14,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            axis.grid(
                axis="y",
                color=colors["grid"],
                linestyle="--",
                linewidth=0.8,
                alpha=0.8,
                zorder=0,
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(axis="both", which="major", length=4)

        figure.suptitle(
            "Generals Heuristic-Learning Policy Change",
            fontsize=17,
            fontweight="bold",
        )
        figure.savefig(
            svg_path,
            format="svg",
            facecolor="white",
        )
        figure.savefig(
            png_path,
            format="png",
            dpi=300,
            facecolor="white",
        )
        plt.close(figure)
    return svg_path, png_path
