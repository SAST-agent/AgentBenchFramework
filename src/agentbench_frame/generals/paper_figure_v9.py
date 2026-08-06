"""Validated, deterministic English publication figures for Generals v9."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


EPSILONS = ("0.001", "0.01", "0.05", "0.1")
TRANSITIONS = tuple(f"v{index}→v{index + 1}" for index in range(9))


@dataclass(frozen=True)
class V9KLDomain:
    domain_id: str
    reference_state_count: int
    transitions: tuple[str, ...]
    primary: tuple[float | None, ...]
    sensitivity: Mapping[str, tuple[float | None, ...]]
    support_sizes: tuple[int, ...]


@dataclass(frozen=True)
class V9FigureData:
    attribution_run_id: str
    v9_run_id: str
    attribution: Mapping[str, Any]
    policy_cells: tuple[str, ...]
    score_history: tuple[float | None, ...]
    legacy: V9KLDomain
    expanded: V9KLDomain


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read figure source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"figure source must be an object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric or missing")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _domain(metric: object, domain_id: str, count: int) -> V9KLDomain:
    if not isinstance(metric, dict):
        raise ValueError(f"{domain_id} reference domain is missing")
    if (
        metric.get("domain_id") != domain_id
        or metric.get("reference_state_count") != count
        or tuple(metric.get("epsilons") or ()) != EPSILONS
        or metric.get("primary_epsilon") != "0.01"
    ):
        raise ValueError(f"{domain_id} reference domain contract changed")
    records = metric.get("transitions")
    if not isinstance(records, list) or len(records) != 9:
        raise ValueError(f"{domain_id} reference domain transition count changed")
    labels = tuple(
        f"{item.get('version_before')}→{item.get('version_after')}"
        for item in records
        if isinstance(item, dict)
    )
    if labels != TRANSITIONS:
        raise ValueError(f"{domain_id} reference domain transition order changed")
    primary = []
    sensitivity = {epsilon: [] for epsilon in EPSILONS}
    for label, item in zip(labels, records):
        coverage = item.get("coverage")
        if not isinstance(coverage, dict) or coverage.get("total") != count:
            raise ValueError(f"{domain_id} {label} reference domain coverage changed")
        primary.append(_number(item.get("mean_kl_nats"), f"{domain_id} {label}"))
        raw_sensitivity = item.get("sensitivity")
        if not isinstance(raw_sensitivity, dict) or tuple(raw_sensitivity) != EPSILONS:
            raise ValueError(f"{domain_id} {label} epsilon order changed")
        for epsilon in EPSILONS:
            epsilon_item = raw_sensitivity[epsilon]
            epsilon_coverage = epsilon_item.get("coverage")
            if (
                not isinstance(epsilon_coverage, dict)
                or epsilon_coverage.get("total") != count
            ):
                raise ValueError(f"{domain_id} {label} epsilon coverage changed")
            sensitivity[epsilon].append(
                _number(
                    epsilon_item.get("mean_kl_nats"),
                    f"{domain_id} {label} epsilon {epsilon}",
                )
            )
    return V9KLDomain(
        domain_id=domain_id,
        reference_state_count=count,
        transitions=labels,
        primary=tuple(primary),
        sensitivity={key: tuple(value) for key, value in sensitivity.items()},
        support_sizes=(),
    )


def _supports(run_dir: Path) -> tuple[int, ...]:
    values: dict[str, int] = {}
    path = run_dir / "events.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read expanded support events: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if (
            event.get("event_type") != "action_space_count"
            or event.get("domain_id") != "expanded-24"
        ):
            continue
        state_id = event.get("measurement_state_id")
        support = event.get("support_size")
        if (
            not isinstance(state_id, str)
            or state_id in values
            or not isinstance(support, str)
            or not support.isdecimal()
            or int(support) < 1
            or event.get("status") != "complete"
        ):
            raise ValueError("expanded exact support labels changed")
        values[state_id] = int(support)
    if len(values) != 24:
        raise ValueError("expanded reference domain requires 24 exact support labels")
    return tuple(values[key] for key in sorted(values))


def load_v9_figure_data(
    attribution_run: Path,
    v9_run: Path,
    legacy_kl_run: Path,
    expanded_kl_run: Path,
) -> V9FigureData:
    """Validate the four first-hand authorities and reject domain splicing."""

    attribution_run = Path(attribution_run)
    v9_run = Path(v9_run)
    legacy_kl_run = Path(legacy_kl_run)
    expanded_kl_run = Path(expanded_kl_run)
    attribution_summary = _read(attribution_run / "summary.json")
    v9_summary = _read(v9_run / "summary.json")
    legacy_summary = _read(legacy_kl_run / "summary.json")
    expanded_summary = _read(expanded_kl_run / "summary.json")
    report_path = attribution_run / "diagnosis/report.json"
    report = _read(report_path)
    if (
        attribution_summary.get("status") != "complete"
        or attribution_summary.get("run_id") != attribution_run.name
        or attribution_summary.get("report_hash") != _sha(report_path)
        or report.get("status") != "complete"
    ):
        raise ValueError("attribution figure authority changed")
    policies = report.get("policies")
    cells = tuple(
        str(item.get("cell")) for item in policies or [] if isinstance(item, dict)
    )
    if cells != ("A", "B", "C", "D"):
        raise ValueError("attribution requires the 2x2 A/B/C/D policy cells")
    if (
        v9_summary.get("status") != "complete"
        or v9_summary.get("runnable") is not True
        or v9_summary.get("run_id") != v9_run.name
        or v9_summary.get("diagnosis_report_hash") != _sha(report_path)
    ):
        raise ValueError("v9 figure authority changed")
    scores = v9_summary.get("score_history")
    if (
        not isinstance(scores, list)
        or len(scores) != 11
        or scores[2] is not None
    ):
        raise ValueError("v9 score history must preserve its missing point")
    score_history = tuple(_number(value, "score history") for value in scores)

    legacy_metric = legacy_summary.get("controlled_reference_policy_kl")
    if (
        legacy_summary.get("status") != "complete"
        or legacy_summary.get("run_id") != legacy_kl_run.name
        or not isinstance(legacy_metric, dict)
        or legacy_metric.get("reference_state_count") != 12
        or len(legacy_metric.get("transitions") or ()) != 8
    ):
        raise ValueError("legacy reference domain authority changed")
    domains = expanded_summary.get("domains")
    if (
        expanded_summary.get("status") != "complete"
        or expanded_summary.get("run_id") != expanded_kl_run.name
        or expanded_summary.get("source_run_id") != legacy_kl_run.name
        or expanded_summary.get("target_run_id") != v9_run.name
        or expanded_summary.get("target_content_hash") != v9_summary.get("candidate_hash")
        or not isinstance(domains, dict)
        or set(domains) != {"legacy-12", "expanded-24"}
    ):
        raise ValueError("expanded reference domain authority changed")
    legacy = _domain(domains["legacy-12"], "legacy-12", 12)
    expanded = _domain(domains["expanded-24"], "expanded-24", 24)
    if legacy_metric.get("transitions") != domains["legacy-12"].get("transitions")[:8]:
        raise ValueError("legacy reference domain was spliced or recomputed")
    supports = _supports(expanded_kl_run)
    legacy = V9KLDomain(**{**legacy.__dict__, "support_sizes": supports[:12]})
    expanded = V9KLDomain(**{**expanded.__dict__, "support_sizes": supports})
    attribution = report.get("attribution")
    if not isinstance(attribution, dict):
        raise ValueError("attribution effects are missing")
    return V9FigureData(
        attribution_run_id=attribution_run.name,
        v9_run_id=v9_run.name,
        attribution=attribution,
        policy_cells=cells,
        score_history=score_history,
        legacy=legacy,
        expanded=expanded,
    )


def _segments(values: tuple[float | None, ...]):
    segment_x: list[int] = []
    segment_y: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            if segment_x:
                yield segment_x, segment_y
                segment_x, segment_y = [], []
            continue
        segment_x.append(index)
        segment_y.append(value)
    if segment_x:
        yield segment_x, segment_y


def _save(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    temporary_png = output_dir / f".{stem}.tmp.png"
    temporary_svg = output_dir / f".{stem}.tmp.svg"
    fig.savefig(
        temporary_png,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "AgentBenchFrame"},
    )
    fig.savefig(
        temporary_svg,
        bbox_inches="tight",
        metadata={"Creator": "AgentBenchFrame", "Date": "2026-08-06"},
    )
    temporary_png.replace(png)
    temporary_svg.replace(svg)
    return png, svg


def render_v9_paper_figures(
    data: V9FigureData,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render three English figures; each KL domain stays in its own axes."""

    if "MPLCONFIGDIR" not in os.environ:
        path = Path(tempfile.gettempdir()) / "agentbench-matplotlib"
        path.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(path)
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["svg.hashsalt"] = "agentbench-generals-v9"
    from matplotlib import pyplot as plt

    outputs: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].set_title("2×2 Attribution Policy Cells")
    colors = ("#f2f2f2", "#9ecae1", "#fdae6b", "#74c476")
    for index, (cell, color) in enumerate(zip(data.policy_cells, colors)):
        row, column = divmod(index, 2)
        axes[0].add_patch(plt.Rectangle((column, 1 - row), 0.9, 0.9, color=color))
        axes[0].text(column + 0.45, 1.45 - row, f"Cell {cell}", ha="center", va="center", weight="bold")
    axes[0].set(xlim=(-0.1, 2), ylim=(-0.1, 2), xticks=[], yticks=[])
    axes[1].set_title("Paired Factorial Effects")
    metrics = data.attribution.get("metrics") or {}
    outcome = metrics.get("outcome_score") or next(iter(metrics.values()), {})
    names = ("Large-stack\nB−A", "Contact\nC−A", "Interaction\nD−B−C+A")
    values = [outcome.get(key) for key in ("large_stack", "contact", "interaction")]
    axes[1].bar(names, [0 if value is None else value for value in values], color="#4c78a8")
    for index, value in enumerate(values):
        axes[1].text(index, 0 if value is None else value, "missing" if value is None else f"{value:.3f}", ha="center", va="bottom")
    axes[1].set_ylabel("Mean paired effect")
    fig.suptitle("Generals v8 Regression: Scientific Attribution")
    outputs.extend(_save(fig, Path(output_dir), "generals-v9-scientific-attribution"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].set_title("Formal Score over Heuristic-Learning Iterations")
    for x, y in _segments(data.score_history):
        axes[0].plot(x, y, marker="o", color="#2f6fbb")
    axes[0].set(xlabel="Policy version", ylabel="Formal benchmark score")
    axes[0].set_xticks(range(len(data.score_history)), [f"v{i}" for i in range(len(data.score_history))], rotation=35)
    axes[0].text(2, 0.02, "missing", rotation=90, color="#b36b00")
    axes[1].set_title("Behavioral Information Gain (Exact Policy KL)")
    axes[1].plot(data.expanded.transitions, data.expanded.primary, marker="o", color="#e45756")
    axes[1].set(xlabel="Policy transition", ylabel="Mean KL (nats/state)")
    axes[1].tick_params(axis="x", rotation=35)
    fig.suptitle("Generals v0–v9 Score and Policy Change")
    outputs.extend(_save(fig, Path(output_dir), "generals-v0-v9-score-and-policy-change"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, domain in zip(axes, (data.legacy, data.expanded)):
        for epsilon in EPSILONS:
            axis.plot(domain.transitions, domain.sensitivity[epsilon], marker="o", label=f"epsilon = {epsilon}")
        axis.set_title(f"{domain.domain_id}: {domain.reference_state_count} Reference States")
        axis.set_xlabel("Policy transition")
        axis.tick_params(axis="x", rotation=40)
        axis.text(
            0.02,
            0.98,
            f"Exact support |A(s)|: {min(domain.support_sizes):,}–{max(domain.support_sizes):,}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel("Mean KL (nats/state)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Generals Exact Policy KL: Separate Reference Domains")
    outputs.extend(_save(fig, Path(output_dir), "generals-legacy12-expanded24-kl"))
    plt.close(fig)
    return tuple(outputs)
