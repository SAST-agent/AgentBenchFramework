"""Finalized-only append-only event storage for HL runs."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1.0"
KNOWN_EVENT_TYPES = frozenset(
    {
        "run_started",
        "origin_imported",
        "curriculum_started",
        "curriculum_target_selected",
        "curriculum_gate_completed",
        "curriculum_stage_promoted",
        "curriculum_candidate_rejected",
        "curriculum_stagnated",
        "curriculum_resumed",
        "run_resumed",
        "act_completed",
        "version_created",
        "candidate_selected",
        "search_parent_selected",
        "match_completed",
        "evaluation_completed",
        "certification_completed",
        "policy_kl_measured",
        "behavior_measured",
        "occupancy_measured",
        "measurement_failed",
        "elo_updated",
        "champion_promoted",
        "rollback_selected",
        "experience_updated",
        "experience_rebuilt",
        "experience_cycle_consolidated",
        "checkpoint_created",
        "run_completed",
        "proposal_cycle_started",
        "planner_completed",
        "finalists_selected",
        "reducer_completed",
        "proposal_cycle_completed",
        "reporting_panel_completed",
        "bootstrap_recovered",
        "repairs_selected",
        "repair_started",
        "repair_completed",
        "branch_representative_selected",
        "candidate_activation_measured",
        "provider_compatibility_selected",
    }
)
_SECRET_KEYS = frozenset({"api_key", "authorization", "access_token", "secret"})
_SECRET_VALUE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_COMMON_FIELDS = {
    "schema_version", "event_id", "event_type", "run_id", "created_at",
}
_EVENT_FIELDS = {
    "run_started": ({"iteration_config"}, {"game"}),
    "provider_compatibility_selected": (
        {"field", "frozen_value", "active_value", "reason"},
        set(),
    ),
    "origin_imported": (
        {
            "source_run_id",
            "source_version_id",
            "source_content_hash",
            "version_id",
        },
        set(),
    ),
    "curriculum_started": (
        {
            "version_id",
            "active_target",
            "active_target_rank",
            "locked_opponents",
            "required_human_opponents",
            "stage_origin_version_id",
        },
        set(),
    ),
    "curriculum_target_selected": (
        {
            "version_id",
            "active_target",
            "active_target_rank",
            "locked_opponents",
            "stage_origin_version_id",
        },
        set(),
    ),
    "curriculum_gate_completed": (
        {
            "version_id",
            "active_target",
            "status",
            "score",
            "matches",
            "baseline",
            "improved",
            "stagnation_count",
            "stage_best_version_id",
            "stage_best_score",
        },
        set(),
    ),
    "curriculum_stage_promoted": (
        {
            "version_id",
            "completed_target",
            "next_target",
            "locked_opponents",
            "passing_human_opponents",
        },
        set(),
    ),
    "curriculum_candidate_rejected": (
        {
            "version_id",
            "stage_origin_version_id",
            "active_target",
            "lost_locked_opponents",
            "failed_active_target",
        },
        set(),
    ),
    "curriculum_stagnated": (
        {
            "version_id",
            "active_target",
            "stage_best_version_id",
            "stage_best_score",
            "stagnation_count",
        },
        set(),
    ),
    "curriculum_resumed": (
        {
            "version_id",
            "active_target",
            "stage_best_version_id",
            "stage_best_score",
        },
        set(),
    ),
    "run_resumed": ({"coding_agent_acts", "iterations", "lineage_head_version_id"}, {"champion_version_id", "provider_attempts"}),
    "act_completed": (
        {"act_id", "iteration_id", "status"},
        {
            "branch_index",
            "prompt_tokens",
            "cached_input_tokens",
            "completion_tokens",
            "reasoning_output_tokens",
            "total_tokens",
            "tool_call_count",
            "elapsed_time_s",
            "raw_output_ref",
            "thread_id",
            "termination_reason",
            "timeout_kind",
            "weighted_tokens",
            "rollout_budget_limit_tokens",
            "rollout_budget_exhausted",
            "accepted_after_budget_exhaustion",
        },
    ),
    "version_created": ({"version_id", "parent_version_id", "act_id", "content_hash", "edit_type", "evaluation_status", "benchmark_score", "selected"}, set()),
    "candidate_selected": ({"iteration_id", "version_id", "act_id"}, set()),
    "search_parent_selected": ({"iteration_id", "version_id", "act_id"}, set()),
    "match_completed": (
        {
            "match_id",
            "version_id",
            "act_id",
            "phase",
            "role",
            "opponent",
            "seed",
            "result",
            "valid",
        },
        {
            "game",
            "candidate_role",
            "points",
            "candidate_score",
            "opponent_score",
            "dense_margin",
            "terminal_metrics",
            "rounds",
            "faults",
            "live_opponent",
            "rollman_score",
            "ghosts_score",
            "replay",
            "trace",
            "error",
        },
    ),
    "evaluation_completed": (
        {"version_id", "status", "benchmark_score", "wins", "draws", "losses", "matches"},
        {"error"},
    ),
    "certification_completed": (
        {
            "version_id",
            "act_id",
            "status",
            "score",
            "passing_human_opponents",
            "required_human_opponents",
            "matches",
        },
        {"hard_opponents", "hard_opponent_gate_passed"},
    ),
    "policy_kl_measured": ({"version_id", "parent_version_id", "epsilon", "action_support", "local_policy_kl_trace", "episode_local_policy_kl", "reference_manifest"}, set()),
    "behavior_measured": (
        {
            "iteration",
            "version_id",
            "reference_version_id",
            "comparison",
            "epsilon",
            "decision_count",
            "changed_action_count",
            "mean_kl_nats_per_decision",
            "role_mean_kl",
            "reference_replays",
        },
        set(),
    ),
    "occupancy_measured": ({"version_id", "parent_version_id", "occupancy_shift"}, set()),
    "measurement_failed": (
        {
            "version_id",
            "parent_version_id",
            "error_type",
            "error_message",
        },
        set(),
    ),
    "elo_updated": ({"match_id", "version_id", "act_id", "phase", "candidate", "opponent", "seed", "result", "rating_before", "rating", "opponent_rating", "role"}, set()),
    "champion_promoted": ({"version_id", "score"}, set()),
    "rollback_selected": ({"from_version_id", "to_version_id", "reason"}, set()),
    "experience_updated": ({"act_id", "version_id", "experience_path"}, set()),
    "experience_rebuilt": (
        {"rejected_version_id", "accepted_updates", "experience_path"},
        set(),
    ),
    "experience_cycle_consolidated": (
        {
            "iteration_id",
            "selected_version_id",
            "record_count",
            "experience_path",
            "ledger_path",
        },
        set(),
    ),
    "checkpoint_created": ({"act_id", "iteration_id", "path", "parent_version_id"}, {"thread_id"}),
    "run_completed": (
        {"reason", "version_id"},
        {"passing_human_opponents", "hard_opponents", "wins_required"},
    ),
    "proposal_cycle_started": (
        {"iteration_id", "parent_version_id", "candidate_count"},
        set(),
    ),
    "planner_completed": (
        {"act_id", "iteration_id", "status", "branch_briefs"},
        set(),
    ),
    "finalists_selected": ({"iteration_id", "version_ids"}, set()),
    "reducer_completed": (
        {"act_id", "iteration_id", "status", "input_path", "output_path"},
        set(),
    ),
    "proposal_cycle_completed": (
        {
            "iteration_id",
            "parent_version_id",
            "selected_version_id",
            "candidate_version_ids",
        },
        set(),
    ),
    "reporting_panel_completed": (
        {
            "iteration_id",
            "proposal_cycle",
            "version_id",
            "status",
            "score",
            "mean_score_margin",
            "matches",
        },
        set(),
    ),
    "bootstrap_recovered": (
        {
            "failed_act_id",
            "raw_output_ref",
            "failure_reason",
            "version_id",
            "content_hash",
        },
        set(),
    ),
    "repairs_selected": (
        {"iteration_id", "branch_indices", "version_ids"},
        set(),
    ),
    "repair_started": (
        {
            "iteration_id",
            "act_id",
            "branch_index",
            "initial_version_id",
            "repair_input_path",
        },
        set(),
    ),
    "repair_completed": (
        {
            "iteration_id",
            "act_id",
            "branch_index",
            "initial_version_id",
            "repaired_version_id",
            "status",
        },
        set(),
    ),
    "branch_representative_selected": (
        {
            "iteration_id",
            "branch_index",
            "initial_version_id",
            "representative_version_id",
            "reason",
        },
        {"repaired_version_id"},
    ),
    "candidate_activation_measured": (
        {
            "iteration_id",
            "act_id",
            "branch_index",
            "version_id",
            "parent_version_id",
            "status",
            "decision_count",
            "changed_action_count",
            "changed_fraction",
            "episodes",
        },
        {"error"},
    ),
}


def _validate_record(record: Mapping[str, Any]) -> None:
    missing_common = sorted(_COMMON_FIELDS - set(record))
    if missing_common:
        raise ValueError(f"event missing common fields: {missing_common}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported event schema_version: {record['schema_version']}")
    for field in ("event_id", "event_type", "run_id", "created_at"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"event.{field} must be a non-empty string")
    event_type = record["event_type"]
    if event_type not in KNOWN_EVENT_TYPES:
        return
    required, optional = _EVENT_FIELDS[event_type]
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{event_type} missing required fields: {missing}")
    unknown = sorted(set(record) - _COMMON_FIELDS - required - optional)
    if unknown:
        raise ValueError(f"{event_type} has unknown fields: {unknown}")
    for field in (
        "act_id", "iteration_id", "version_id", "parent_version_id",
        "from_version_id", "to_version_id", "match_id", "phase", "role",
        "opponent", "candidate", "reason", "status", "content_hash",
        "edit_type", "evaluation_status", "path", "experience_path",
        "raw_output_ref", "thread_id", "replay", "trace",
        "reference_manifest", "source_run_id", "source_version_id",
        "source_content_hash", "active_target", "stage_origin_version_id",
        "stage_best_version_id", "completed_target", "next_target",
        "error_type", "error_message", "rejected_version_id",
        "branch_briefs", "input_path", "output_path", "selected_version_id",
        "initial_version_id", "repaired_version_id",
        "representative_version_id", "repair_input_path",
        "error", "ledger_path",
    ):
        if field in record and record[field] is not None and not isinstance(record[field], str):
            raise ValueError(f"{event_type}.{field} must be a string or null")
    for field in (
        "branch_index", "seed", "wins", "draws", "losses",
        "passing_human_opponents", "required_human_opponents", "wins_required",
        "coding_agent_acts", "iterations", "prompt_tokens",
        "cached_input_tokens", "completion_tokens",
        "reasoning_output_tokens", "total_tokens", "rollman_score",
        "ghosts_score",
        "active_target_rank", "stagnation_count",
        "accepted_updates",
        "candidate_count",
        "decision_count", "changed_action_count", "record_count",
    ):
        value = record.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{event_type}.{field} must be an integer or null")
    for field in (
        "benchmark_score", "score", "epsilon", "occupancy_shift",
        "rating_before", "rating", "opponent_rating", "elapsed_time_s",
        "stage_best_score",
        "changed_fraction",
    ):
        value = record.get(field)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            raise ValueError(f"{event_type}.{field} must be numeric or null")
    for field in (
        "selected", "valid", "baseline", "improved", "failed_active_target",
        "hard_opponent_gate_passed",
    ):
        if field in record and not isinstance(record[field], bool):
            raise ValueError(f"{event_type}.{field} must be boolean")
    for field in (
        "matches", "action_support", "local_policy_kl_trace",
        "episode_local_policy_kl",
        "locked_opponents", "lost_locked_opponents", "hard_opponents",
        "version_ids", "candidate_version_ids",
        "branch_indices",
        "episodes",
    ):
        if field in record and not isinstance(record[field], list):
            raise ValueError(f"{event_type}.{field} must be a list")
    if "iteration_config" in record and not isinstance(
        record["iteration_config"], Mapping
    ):
        raise ValueError(f"{event_type}.iteration_config must be an object")
    if "branch_indices" in record and any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in record["branch_indices"]
    ):
        raise ValueError(f"{event_type}.branch_indices must contain integers")
    for field in ("version_ids", "candidate_version_ids"):
        if field in record and any(
            not isinstance(value, str) or not value for value in record[field]
        ):
            raise ValueError(f"{event_type}.{field} must contain version ids")


def _validate_safe(value: Any, path: str = "event") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ValueError(f"{path} contains secret-like material")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _SECRET_KEYS:
                raise ValueError(f"{path}.{key} is a forbidden secret field")
            _validate_safe(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe(item, f"{path}[{index}]")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class HLEventWriter:
    """Append complete immutable JSON records and reject duplicate IDs."""

    def __init__(self, path: str | os.PathLike[str], run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._event_ids = {
            record.get("event_id")
            for record in read_events(self.path)
            if record.get("event_id")
        }

    def write(self, event_type: str, **fields: Any) -> str:
        if event_type not in KNOWN_EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")
        event_id = str(fields.pop("event_id", uuid.uuid4().hex))
        if event_id in self._event_ids:
            raise ValueError(f"duplicate event_id: {event_id}")
        record = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "event_type": event_type,
            "run_id": self.run_id,
            "created_at": str(fields.pop("created_at", _timestamp())),
            **fields,
        }
        _validate_record(record)
        _validate_safe(record)
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._event_ids.add(event_id)
        return event_id


def read_events(
    path: str | os.PathLike[str],
    *,
    warn: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"event line {line_number} is not an object")
            _validate_record(value)
            _validate_safe(value)
            event_id = str(value["event_id"])
            if event_id in event_ids:
                raise ValueError(f"duplicate event_id at line {line_number}: {event_id}")
            event_ids.add(event_id)
            event_type = value.get("event_type")
            if event_type not in KNOWN_EVENT_TYPES and warn is not None:
                warn(f"unknown event type: {event_type}")
            records.append(value)
    return records
