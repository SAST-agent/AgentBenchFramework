"""Publication figure projection for controlled-reference Generals policy KL."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


EXPECTED_VERSION_PAIRS = (
    ("v0", "v1"),
    ("v1", "v2"),
    ("v2", "v3"),
    ("v3", "v4"),
    ("v4", "v5"),
    ("v5", "v6"),
)
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
    if actual_pairs != EXPECTED_VERSION_PAIRS:
        raise ValueError("transition order must be v0→v1 through v5→v6")

    primary_values = []
    sensitivity_values = {
        epsilon: [] for epsilon in EXPECTED_EPSILONS
    }
    for transition, (before, after) in zip(
        transitions,
        EXPECTED_VERSION_PAIRS,
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
            f"{before}→{after}" for before, after in EXPECTED_VERSION_PAIRS
        ),
        primary_epsilon="0.01",
        primary_kl=tuple(primary_values),
        sensitivity={
            epsilon: tuple(values)
            for epsilon, values in sensitivity_values.items()
        },
        support_states=_load_support_states(run_dir / "events.jsonl"),
    )
