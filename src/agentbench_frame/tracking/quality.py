"""Non-destructive diagnostics for first-hand JSONL event streams."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


KNOWN_EVENT_TYPES = frozenset({
    "log", "episode", "step", "act", "coding_agent_act", "act_evaluation",
    "benchmark_game_result", "benchmark_evaluation", "evaluation", "budget",
    "policy_kl_trace", "occupancy", "elo", "h2h", "resource", "provider_event",
    "provider_invocation",
    "benchmark_spec", "version", "game_result", "behavior_change",
    "pipeline_error", "pipeline_resumed",
    "calibration_spec", "calibration_result", "dense_trajectory",
    "dense_episode_summary", "dense_metric_error", "lineage_import",
    "behavior_change_episode",
    "provider_retry",
    "learning_validation",
    "decision_class_summary",
    "behavior_gate",
    "behavior_measurement_error",
    "lineage_budget_audit",
})


@dataclass
class EventQualityReport:
    total_lines: int = 0
    valid_events: int = 0
    malformed_lines: int = 0
    invalid_events: int = 0
    unknown_event_types: int = 0
    duplicate_event_ids: int = 0
    missing_event_ids: int = 0
    missing_run_ids: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "valid_events": self.valid_events,
            "malformed_lines": self.malformed_lines,
            "invalid_events": self.invalid_events,
            "unknown_event_types": self.unknown_event_types,
            "duplicate_event_ids": self.duplicate_event_ids,
            "missing_event_ids": self.missing_event_ids,
            "missing_run_ids": self.missing_run_ids,
            "warnings": list(self.warnings),
        }


def inspect_event_lines(
    lines: Iterable[str], known_event_types: Optional[set[str]] = None
) -> EventQualityReport:
    """Inspect lines without rejecting unknown future event types."""

    known = KNOWN_EVENT_TYPES if known_event_types is None else set(known_event_types)
    report = EventQualityReport()
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        report.total_lines += 1
        try:
            event = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            report.malformed_lines += 1
            report.warnings.append(f"line {line_number}: malformed JSON")
            continue
        if not isinstance(event, dict):
            report.invalid_events += 1
            report.warnings.append(f"line {line_number}: event is not an object")
            continue
        report.valid_events += 1
        event_id = event.get("event_id")
        if not event_id:
            report.missing_event_ids += 1
            report.warnings.append(f"line {line_number}: missing event_id")
        elif str(event_id) in seen_ids:
            report.duplicate_event_ids += 1
            report.warnings.append(f"line {line_number}: duplicate event_id {event_id}")
        else:
            seen_ids.add(str(event_id))
        if not event.get("run_id"):
            report.missing_run_ids += 1
            report.warnings.append(f"line {line_number}: missing run_id")
        event_type = event.get("event_type", event.get("event"))
        if event_type not in known:
            report.unknown_event_types += 1
            report.warnings.append(f"line {line_number}: unknown event_type {event_type!r}")
    return report


def inspect_event_file(path: str | Path, known_event_types: Optional[set[str]] = None) -> EventQualityReport:
    target = Path(path)
    try:
        return inspect_event_lines(target.read_text(encoding="utf-8").splitlines(), known_event_types)
    except OSError as exc:
        report = EventQualityReport()
        report.warnings.append(f"cannot read events: {exc}")
        return report
