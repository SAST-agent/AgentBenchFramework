"""Local commands for reproducible HL runs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentbench_frame.games.rollman.contract import (
    RollmanContract,
    asset_path,
    audit_sources,
)
from agentbench_frame.games.rollman.evaluator import load_human_pool
from agentbench_frame.hl.context import ContextBundle
from agentbench_frame.hl.config import HLRunConfig
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.experience import ExperienceManager
from agentbench_frame.hl.local_config import LocalHLConfig


_SECRET_LIKE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _ensure_replay_summary(
    *,
    replay: str | Path,
    summarizer: str | Path,
) -> Path:
    replay_path = Path(replay).resolve()
    summary_path = replay_path.with_name("summary.md")
    if summary_path.is_file():
        return summary_path
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(summarizer).resolve()),
            str(replay_path),
            "--format",
            "markdown",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "failed to create replay summary: "
            + (completed.stderr.strip() or "unknown summarizer failure")
        )
    if len(completed.stdout.encode("utf-8")) > 64 * 1024:
        raise ValueError("replay summary exceeds 64 KiB")
    summary_path.write_text(completed.stdout, encoding="utf-8")
    return summary_path


def _ensure_opponent_distillation(
    *,
    traces: list[str | Path],
    distillation_tool: str | Path,
    output_root: str | Path,
) -> Path:
    """Create one content-addressed, bounded Ghost distillation artifact."""

    trace_paths = sorted({Path(value).resolve() for value in traces})
    if not trace_paths:
        raise ValueError("opponent distillation requires at least one trace")
    digest = hashlib.sha256()
    digest.update(b"agentbench-modal-distillation-v1\0")
    for path in trace_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path).encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    destination_root = Path(output_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"ghost-{digest.hexdigest()[:20]}.json"

    def compact(value: dict[str, Any]) -> dict[str, Any]:
        compacted = {
            key: value[key]
            for key in (
                "schema_version",
                "trace_count",
                "ghost_decision_samples",
                "frozen_samples_omitted",
                "action_counts",
                "chase_step_rate",
                "same_action_as_previous_rate",
                "feature_contract",
            )
            if key in value
        }
        compacted["representation"] = "modal_patterns_v1"
        for table in ("coarse_backoff_patterns", "fine_patterns"):
            compacted[table] = [
                {
                    key: pattern[key]
                    for key in (
                        "context",
                        "count",
                        "modal_action",
                        "modal_confidence",
                    )
                }
                for pattern in value.get(table, [])
            ]
        return compacted

    def validate(raw: str) -> dict[str, Any]:
        if len(raw.encode("utf-8")) > 24 * 1024:
            raise ValueError("opponent distillation exceeds 24 KiB")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("schema_version") != "1.0":
            raise ValueError("invalid opponent distillation schema")
        if int(value.get("trace_count") or 0) != len(trace_paths):
            raise ValueError("opponent distillation trace count mismatch")
        if int(value.get("ghost_decision_samples") or 0) < 1:
            raise ValueError("opponent distillation has no Ghost decisions")
        for key in ("coarse_backoff_patterns", "fine_patterns"):
            if not isinstance(value.get(key), list):
                raise ValueError(f"opponent distillation lacks {key}")
        if value.get("representation") != "modal_patterns_v1":
            raise ValueError("opponent distillation is not compact modal form")
        if any(
            "action_counts" in pattern
            for key in ("coarse_backoff_patterns", "fine_patterns")
            for pattern in value[key]
        ):
            raise ValueError("opponent distillation contains verbose patterns")
        return value

    if destination.is_file():
        validate(destination.read_text(encoding="utf-8"))
        return destination
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(distillation_tool).resolve()),
            *(str(path) for path in trace_paths),
            "--max-patterns",
            "40",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "failed to distill opponent policy: "
            + (completed.stderr.strip() or "unknown distillation failure")
        )
    raw_value = json.loads(completed.stdout)
    value = compact(raw_value)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    validate(encoded)
    destination.write_text(encoded, encoding="utf-8")
    return destination


def _trace_fault_summary(trace: str | Path | None) -> dict[str, str] | None:
    """Return the latest bounded Rollman fault for the next coding act."""

    if trace is None:
        return None
    path = Path(trace)
    if not path.is_file():
        return None
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(record, dict)
            and record.get("type") == "ai_fault"
            and record.get("player") == 0
        ):
            latest = record
    if latest is None:
        return None
    result: dict[str, str] = {}
    for key in ("error", "detail", "stderr_tail"):
        value = latest.get(key)
        if value is None:
            continue
        text = _SECRET_LIKE.sub("[REDACTED]", str(value))
        if len(text.encode("utf-8")) > 4096:
            text = text[-4096:]
        result[key] = text
    return result or None


def _measure_candidate(
    *,
    measurement_runner: Any,
    version_store: Any,
    writer: Any,
    candidate_version: Any,
    parent_version: Any,
    candidate_evaluation: CandidateEvaluation,
    parent_evaluation: CandidateEvaluation,
) -> bool:
    """Measure one complete candidate, recording policy-probe failure safely."""

    try:
        metrics = measurement_runner.measure(
            new_version=candidate_version,
            old_version=parent_version,
            version_store=version_store,
            new_evaluation=candidate_evaluation,
            old_evaluation=parent_evaluation,
        )
    except Exception as error:
        message = " ".join(str(error).split()) or "measurement failed"
        message = _SECRET_LIKE.sub("[REDACTED]", message)
        if len(message) > 800:
            message = message[:200] + " ... " + message[-595:]
        writer.write(
            "measurement_failed",
            version_id=candidate_version.version_id,
            parent_version_id=parent_version.version_id,
            error_type=type(error).__name__,
            error_message=message,
        )
        return False
    writer.write(
        "policy_kl_measured",
        version_id=candidate_version.version_id,
        parent_version_id=parent_version.version_id,
        epsilon=metrics["epsilon"],
        action_support=metrics["action_support"],
        local_policy_kl_trace=metrics["local_policy_kl_trace"],
        episode_local_policy_kl=metrics["episode_local_policy_kl"],
        reference_manifest=metrics["reference_manifest"],
    )
    writer.write(
        "occupancy_measured",
        version_id=candidate_version.version_id,
        parent_version_id=parent_version.version_id,
        occupancy_shift=metrics["occupancy_shift"],
    )
    return True


def _pending_measurement_candidate(
    events: list[dict[str, Any]],
) -> tuple[str, str] | None:
    """Return the latest selected evaluated candidate left mid-transaction."""

    selected_index = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index].get("event_type")
            in {"candidate_selected", "search_parent_selected"}
            and events[index].get("iteration_id") != "iter-000000"
        ),
        None,
    )
    if selected_index is None:
        return None
    version_id = str(events[selected_index]["version_id"])
    iteration_id = str(events[selected_index].get("iteration_id") or "")
    created = next(
        (
            event
            for event in reversed(events[: selected_index + 1])
            if event.get("event_type") == "version_created"
            and str(event.get("version_id")) == version_id
        ),
        None,
    )
    if (
        created is None
        or created.get("evaluation_status") != "complete"
        or created.get("parent_version_id") is None
    ):
        return None
    later = events[selected_index + 1 :]
    completed_cycle = next(
        (
            event
            for event in later
            if event.get("event_type") == "proposal_cycle_completed"
            and str(event.get("iteration_id") or "") == iteration_id
        ),
        None,
    )
    if (
        completed_cycle is not None
        and str(completed_cycle.get("selected_version_id"))
        == str(completed_cycle.get("parent_version_id"))
    ):
        return None
    if any(
        event.get("event_type") == "measurement_failed"
        and str(event.get("version_id")) == version_id
        for event in later
    ):
        return None
    if any(
        event.get("event_type") == "rollback_selected"
        and str(event.get("from_version_id")) == version_id
        for event in later
    ):
        return None
    completed_types = {
        str(event.get("event_type"))
        for event in later
        if str(event.get("version_id")) == version_id
    }
    if {"policy_kl_measured", "occupancy_measured"} <= completed_types:
        return None
    return version_id, str(created["parent_version_id"])


def _pending_proposal_finalization(
    events: list[dict[str, Any]],
) -> tuple[str, str, str] | None:
    """Find a measured proposal cycle interrupted before its curriculum gate."""

    cycle_index = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index].get("event_type") == "proposal_cycle_completed"
        ),
        None,
    )
    if cycle_index is None:
        return None
    cycle = events[cycle_index]
    iteration_id = str(cycle.get("iteration_id") or "")
    version_id = str(cycle.get("selected_version_id") or "")
    parent_id = str(cycle.get("parent_version_id") or "")
    if not iteration_id or not version_id or not parent_id:
        return None
    later = events[cycle_index + 1 :]
    if any(
        event.get("event_type") == "curriculum_gate_completed"
        and str(event.get("version_id")) == version_id
        for event in later
    ):
        return None
    measured = {
        str(event.get("event_type"))
        for event in later
        if str(event.get("version_id")) == version_id
    }
    no_change = version_id == parent_id
    if (
        not no_change
        and not {"policy_kl_measured", "occupancy_measured"} <= measured
    ):
        return None
    return iteration_id, version_id, parent_id


def _measurement_candidates(iteration_result: Any) -> tuple[Any, ...]:
    """Measure the single linear successor used by the main iteration curves."""

    selected_id = iteration_result.selected.version.version_id
    search_parent_id = getattr(
        iteration_result,
        "search_parent_version_id",
        selected_id,
    )
    if search_parent_id != selected_id:
        return ()
    return (iteration_result.selected,)


def _pending_failed_candidate(
    events: list[dict[str, Any]],
) -> tuple[str, str, str] | None:
    """Return a selected failed candidate not yet restored to its parent."""

    selected_index = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index].get("event_type") == "candidate_selected"
            and events[index].get("iteration_id") != "iter-000000"
        ),
        None,
    )
    if selected_index is None:
        return None
    selected = events[selected_index]
    version_id = str(selected["version_id"])
    later = events[selected_index + 1 :]
    if any(
        event.get("event_type") == "rollback_selected"
        and str(event.get("from_version_id")) == version_id
        for event in later
    ):
        return None
    created = next(
        (
            event
            for event in reversed(events[: selected_index + 1])
            if event.get("event_type") == "version_created"
            and str(event.get("version_id")) == version_id
        ),
        None,
    )
    if (
        created is None
        or created.get("evaluation_status") == "complete"
        or created.get("parent_version_id") is None
    ):
        return None
    act_id = str(created["act_id"])
    act = next(
        (
            event
            for event in reversed(events[: selected_index + 1])
            if event.get("event_type") == "act_completed"
            and str(event.get("act_id")) == act_id
        ),
        None,
    )
    provider_status = None if act is None else str(act.get("status"))
    reason = (
        f"provider_{provider_status}"
        if provider_status not in {None, "completed"}
        else f"evaluation_{created['evaluation_status']}"
    )
    return version_id, str(created["parent_version_id"]), reason


def _validated_experience_updates(
    events: list[dict[str, Any]],
    run_root: str | Path,
) -> tuple[tuple[str, Path], ...]:
    """Select staged Experience updates whose policy measurements completed."""

    event_types_by_version: dict[str, set[str]] = {}
    for event in events:
        version_id = event.get("version_id")
        if version_id is None:
            continue
        event_types_by_version.setdefault(str(version_id), set()).add(
            str(event.get("event_type"))
        )
    updates: list[tuple[str, Path]] = []
    for event in events:
        if event.get("event_type") != "experience_updated":
            continue
        version_id = str(event["version_id"])
        types = event_types_by_version.get(version_id, set())
        if "measurement_failed" in types or not {
            "policy_kl_measured",
            "occupancy_measured",
        } <= types:
            continue
        act_id = str(event["act_id"])
        pending = (
            Path(run_root) / "experience" / "pending" / f"{act_id}.json"
        )
        if not pending.is_file():
            raise FileNotFoundError(
                f"validated experience update is missing: {pending}"
            )
        updates.append((act_id, pending))
    return tuple(updates)


def _load(path: str) -> LocalHLConfig:
    return LocalHLConfig.load(path)


def _validate(config: LocalHLConfig) -> dict[str, Any]:
    from agentbench_frame.hl.codebase import _load_version

    contract = RollmanContract.from_agentbench(config.paths.agentbench_root)
    pool = load_human_pool(config.paths.human_manifest)
    if not config.paths.official_logic_root.is_dir():
        raise FileNotFoundError(config.paths.official_logic_root)
    if not config.paths.pacman_sdk_root.is_dir():
        raise FileNotFoundError(config.paths.pacman_sdk_root)
    source_audit = audit_sources(
        config.paths.agentbench_root,
        config.paths.official_logic_root,
        config.paths.pacman_sdk_root,
    )
    if not source_audit["valid"]:
        raise ValueError("Rollman frozen source audit failed")
    result = {
        "valid": True,
        "game": config.run.game,
        "max_acts": config.run.iteration.max_acts,
        "candidates_per_act": config.run.iteration.candidates_per_act,
        "rollback_enabled": config.run.rollback.enabled,
        "rollback_patience": config.run.rollback.patience,
        "context_mode": config.run.provider.context_mode,
        "measurement_actions": list(contract.directions.values()),
        "human_opponents": len(pool),
        "learning_opponent": config.run.evaluation.learning_opponent,
        "required_human_opponents": config.run.evaluation.required_human_opponents,
        "origin_mode": config.run.origin.mode,
        "curriculum_mode": config.run.curriculum.mode,
    }
    if config.run.origin.mode == "imported_version":
        assert config.run.origin.source_run is not None
        assert config.run.origin.source_version is not None
        source = _load_version(
            Path(config.run.origin.source_run) / "versions",
            config.run.origin.source_version,
        )
        result.update(
            source_run=str(Path(config.run.origin.source_run).resolve()),
            source_version=source.version_id,
            source_content_hash=source.content_hash,
            required_human_opponents=(
                config.run.curriculum.required_human_opponents
            ),
        )
    return result


def _ensure_candidate(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    candidate = workspace / "ai.py"
    if not candidate.exists():
        template = asset_path("candidate-template/ai.py")
        shutil.copy2(template, candidate)


def _dry_run(
    config: LocalHLConfig,
    *,
    run_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    validation = _validate(config)
    _ensure_candidate(workspace)
    run_dir.mkdir(parents=True, exist_ok=False)
    bundle = ContextBundle.create(
        run_dir / "context",
        {
            "rules": asset_path("rules.md"),
            "decision_space": asset_path("decision_space.yaml"),
            "replay_skill": asset_path(
                "replay-skill/rollman-replay"
            ),
        },
    )
    experience = ExperienceManager(
        run_dir / "experience",
        compress_every_acts=config.run.experience.compress_every_acts,
    )
    return {
        **validation,
        "dry_run": True,
        "would_call_model": False,
        "run_dir": str(run_dir.resolve()),
        "workspace": str(workspace.resolve()),
        "context_manifest": str(bundle.manifest_path),
        "context_bundle_hash": bundle.bundle_hash,
        "experience_skill": str(experience.path),
        "context_mode": config.run.provider.context_mode,
    }


def _provider_environment(config: LocalHLConfig) -> dict[str, str] | None:
    key = os.environ.get(config.run.provider.env_key)
    dotenv = config.source_path.parents[2] / ".env"
    if not key and dotenv.is_file():
        from dotenv import dotenv_values

        value = dotenv_values(dotenv).get(config.run.provider.env_key)
        key = None if value is None else str(value)
    if not key:
        return None
    return {**os.environ, config.run.provider.env_key: key}


def _default_run_dir(config: LocalHLConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.paths.runs_root / f"run-{timestamp}"


def _frozen_run_config(config: LocalHLConfig) -> dict[str, Any]:
    """Return the JSON-native run snapshot used for run and resume."""

    value = {
        "schema_version": "1.0",
        "source_config": str(config.source_path),
        "source_config_sha256": hashlib.sha256(
            config.source_path.read_bytes()
        ).hexdigest(),
        "run": config.run.to_dict(),
        "paths": {
            field.name: str(getattr(config.paths, field.name))
            for field in dataclasses.fields(config.paths)
        },
    }
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _normalize_frozen_run_config(value: dict[str, Any]) -> dict[str, Any]:
    """Fill schema defaults when comparing runs frozen by an older harness."""

    normalized = json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    normalized["run"] = HLRunConfig.from_mapping(normalized["run"]).to_dict()
    return json.loads(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    )


def _eligible_certification_version_id(
    *,
    lineage_head_version_id: str | None,
    evaluations_by_version: dict[str, Any],
    completed_certifications: set[str],
    required_score: float,
) -> str | None:
    version_id = lineage_head_version_id
    if version_id is None or version_id in completed_certifications:
        return None
    evaluation = evaluations_by_version.get(version_id)
    if (
        evaluation is None
        or evaluation.status != "complete"
        or evaluation.score is None
        or evaluation.score < required_score
    ):
        return None
    return version_id


def _iteration_stop_reason(iteration_result: Any) -> str | None:
    selected = iteration_result.selected
    if selected.provider.status != "completed":
        return f"provider_{selected.provider.status}"
    if selected.evaluation.status != "complete":
        return f"evaluation_{selected.evaluation.status}"
    return None


def _pending_planner_recovery(
    historical: list[dict[str, Any]],
    *,
    provider: Any,
    workspace: str | Path,
) -> Any | None:
    """Recover a validated planner artifact from an interrupted proposal cycle."""

    from agentbench_frame.hl.proposal import load_branch_briefs

    completed_cycles = {
        str(event.get("iteration_id"))
        for event in historical
        if event.get("event_type") == "proposal_cycle_completed"
    }
    pending = next(
        (
            event
            for event in reversed(historical)
            if event.get("event_type") == "proposal_cycle_started"
            and str(event.get("iteration_id")) not in completed_cycles
        ),
        None,
    )
    if pending is None:
        return None
    iteration_id = str(pending["iteration_id"])
    completed_planner = next(
        (
            event
            for event in reversed(historical)
            if event.get("event_type") == "planner_completed"
            and str(event.get("iteration_id")) == iteration_id
        ),
        None,
    )
    if completed_planner is not None:
        from agentbench_frame.tracking.provider import ProviderInvocation

        persisted = Path(str(completed_planner["branch_briefs"]))
        if not persisted.is_file():
            return None
        load_branch_briefs(persisted, expected_count=4)
        workspace_briefs = Path(workspace) / ".agentbench" / "branch_briefs.json"
        workspace_briefs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(persisted, workspace_briefs)
        return ProviderInvocation(
            status="completed",
            metadata={
                "act_id": str(completed_planner["act_id"]),
                "iteration_id": iteration_id,
                "recovered_from_persisted_output": True,
            },
        )
    failed = next(
        (
            event
            for event in reversed(historical)
            if event.get("event_type") == "act_completed"
            and str(event.get("iteration_id")) == iteration_id
            and str(event.get("act_id", "")).endswith("-planner")
            and event.get("status") == "failed"
        ),
        None,
    )
    if failed is None or not hasattr(provider, "recover_completed_output"):
        return None
    raw_ref = failed.get("raw_output_ref")
    raw_path = None if raw_ref is None else Path(str(raw_ref))
    briefs_path = Path(workspace) / ".agentbench" / "branch_briefs.json"
    if raw_path is None or not raw_path.is_file() or not briefs_path.is_file():
        return None
    load_branch_briefs(briefs_path, expected_count=4)
    recovered = provider.recover_completed_output(
        raw_output_path=raw_path,
        workspace=workspace,
    )
    if recovered.status != "completed":
        return None
    recovered.metadata.update(
        {
            "act_id": str(failed["act_id"]),
            "iteration_id": iteration_id,
        }
    )
    return recovered


def _pending_repair_recoveries(
    historical: list[dict[str, Any]],
    *,
    version_store: Any,
    evaluations_by_version: dict[str, CandidateEvaluation],
) -> dict[int, Any]:
    """Rebuild verified completed repairs from one interrupted proposal cycle."""

    from agentbench_frame.hl.controller import CandidateResult
    from agentbench_frame.tracking.provider import ProviderInvocation

    completed_cycles = {
        str(event.get("iteration_id"))
        for event in historical
        if event.get("event_type") == "proposal_cycle_completed"
    }
    pending = next(
        (
            event
            for event in reversed(historical)
            if event.get("event_type") == "proposal_cycle_started"
            and str(event.get("iteration_id")) not in completed_cycles
        ),
        None,
    )
    if pending is None:
        return {}
    iteration_id = str(pending["iteration_id"])
    versions = {
        str(event["version_id"]): event
        for event in historical
        if event.get("event_type") == "version_created"
    }
    checkpoints = {
        str(event["act_id"]): event
        for event in historical
        if event.get("event_type") == "checkpoint_created"
        and str(event.get("iteration_id")) == iteration_id
    }
    recovered: dict[int, CandidateResult] = {}
    for event in historical:
        if (
            event.get("event_type") != "repair_completed"
            or str(event.get("iteration_id")) != iteration_id
            or event.get("status") != "completed"
        ):
            continue
        branch_index = int(event["branch_index"])
        act_id = str(event["act_id"])
        version_id = str(event["repaired_version_id"])
        version_event = versions.get(version_id)
        checkpoint_event = checkpoints.get(act_id)
        evaluation = evaluations_by_version.get(version_id)
        if version_event is None or checkpoint_event is None or evaluation is None:
            continue
        checkpoint_path = Path(str(checkpoint_event["path"]))
        if not checkpoint_path.is_file():
            continue
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("act_id") != act_id
            or checkpoint.get("iteration_id") != iteration_id
            or int(checkpoint.get("branch_index", -1)) != branch_index
            or checkpoint.get("provider_status") != "completed"
            or version_event.get("act_id") != act_id
        ):
            continue
        version = version_store.get(version_id)
        if (
            version.content_hash != version_event.get("content_hash")
            or version.parent_version_id
            != str(event["initial_version_id"])
        ):
            continue
        recovered[branch_index] = CandidateResult(
            act_id=act_id,
            branch_index=branch_index,
            version=version,
            evaluation=evaluation,
            provider=ProviderInvocation(
                status="completed",
                raw_output_ref=checkpoint.get("raw_output_ref"),
                metadata={
                    "act_id": act_id,
                    "iteration_id": iteration_id,
                    "recovered_from_persisted_output": True,
                },
            ),
        )
    return recovered


def _pending_candidate_recoveries(
    historical: list[dict[str, Any]],
    *,
    version_store: Any,
    evaluations_by_version: dict[str, CandidateEvaluation],
) -> dict[int, Any]:
    """Rebuild completed initial siblings in an interrupted proposal cycle."""

    from agentbench_frame.hl.controller import CandidateResult
    from agentbench_frame.tracking.provider import ProviderInvocation

    completed_cycles = {
        str(event.get("iteration_id"))
        for event in historical
        if event.get("event_type") == "proposal_cycle_completed"
    }
    pending = next(
        (
            event
            for event in reversed(historical)
            if event.get("event_type") == "proposal_cycle_started"
            and str(event.get("iteration_id")) not in completed_cycles
        ),
        None,
    )
    if pending is None:
        return {}
    iteration_id = str(pending["iteration_id"])
    parent_id = str(pending["parent_version_id"])
    acts = {
        str(event["act_id"]): event
        for event in historical
        if event.get("event_type") == "act_completed"
        and str(event.get("iteration_id")) == iteration_id
        and event.get("status") == "completed"
        and isinstance(event.get("branch_index"), int)
    }
    checkpoints = {
        str(event["act_id"]): event
        for event in historical
        if event.get("event_type") == "checkpoint_created"
        and str(event.get("iteration_id")) == iteration_id
    }
    recovered: dict[int, CandidateResult] = {}
    for version_event in historical:
        if (
            version_event.get("event_type") != "version_created"
            or version_event.get("edit_type") != "candidate"
            or version_event.get("parent_version_id") != parent_id
        ):
            continue
        act_id = str(version_event["act_id"])
        act = acts.get(act_id)
        checkpoint_event = checkpoints.get(act_id)
        version_id = str(version_event["version_id"])
        evaluation = evaluations_by_version.get(version_id)
        if act is None or checkpoint_event is None or evaluation is None:
            continue
        branch_index = int(act["branch_index"])
        checkpoint_path = Path(str(checkpoint_event["path"]))
        if not checkpoint_path.is_file():
            continue
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("act_id") != act_id
            or checkpoint.get("iteration_id") != iteration_id
            or checkpoint.get("parent_version_id") != parent_id
            or int(checkpoint.get("branch_index", -1)) != branch_index
            or checkpoint.get("provider_status") != "completed"
        ):
            continue
        version = version_store.get(version_id)
        if (
            version.content_hash != version_event.get("content_hash")
            or version.parent_version_id != parent_id
        ):
            continue
        recovered[branch_index] = CandidateResult(
            act_id=act_id,
            branch_index=branch_index,
            version=version,
            evaluation=evaluation,
            provider=ProviderInvocation(
                status="completed",
                raw_output_ref=checkpoint.get("raw_output_ref"),
                metadata={
                    "act_id": act_id,
                    "iteration_id": iteration_id,
                    "recovered_from_persisted_output": True,
                },
            ),
        )
    return recovered


def _bootstrap_recovery_candidate(
    historical: list[dict[str, Any]],
    *,
    workspace: str | Path,
    provider: Any,
) -> dict[str, Any] | None:
    """Validate an interrupted model-bootstrap workspace before evaluation.

    A bootstrap may finish editing ``ai.py`` before its provider stream reaches a
    terminal event.  Recovery is deliberately conservative: it is allowed only
    for the single bootstrap act, before any version exists, after the provider's
    normal access audit reports no violations, and when the raw stream proves
    that a file edit completed (or records the legacy disconnect marker).
    """

    if any(event.get("event_type") == "version_created" for event in historical):
        return None
    failed = next(
        (
            event
            for event in reversed(historical)
            if event.get("event_type") == "act_completed"
            and event.get("act_id") == "act-000001-b00"
            and event.get("status") == "failed"
        ),
        None,
    )
    if failed is None or not hasattr(provider, "recover_completed_output"):
        return None
    raw_ref = failed.get("raw_output_ref")
    raw_path = None if raw_ref is None else Path(str(raw_ref))
    workspace_path = Path(workspace)
    if (
        raw_path is None
        or not raw_path.is_file()
        or not (workspace_path / "ai.py").is_file()
    ):
        return None

    try:
        recovered = provider.recover_completed_output(
            raw_output_path=raw_path,
            workspace=workspace_path,
        )
    except Exception:
        return None
    metadata = recovered.metadata if isinstance(recovered.metadata, dict) else {}
    if metadata.get("access_policy_violations"):
        return None

    raw_text = raw_path.read_text(encoding="utf-8")
    legacy_disconnect = "stream disconnected before completion" in raw_text
    completed_file_change = False
    for line in raw_text.splitlines():
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            payload.get("type") == "item.completed"
            and isinstance(payload.get("item"), dict)
            and payload["item"].get("type") == "file_change"
        ):
            completed_file_change = True
            break
    if not legacy_disconnect and not completed_file_change:
        return None

    return {
        "failed_act_id": str(failed["act_id"]),
        "raw_output_ref": str(raw_path),
        "failure_reason": (
            "provider_stream_disconnected_after_workspace_edit"
            if legacy_disconnect
            else "provider_interrupted_after_workspace_edit"
        ),
    }


def _curriculum_evaluation_from_events(
    events: list[dict[str, Any]],
    *,
    version_id: str,
    active_target: str,
) -> CandidateEvaluation | None:
    """Recover the latest complete gate for one version and target."""

    for event in reversed(events):
        if str(event.get("version_id")) != version_id:
            continue
        event_type = event.get("event_type")
        if event_type not in {
            "evaluation_completed",
            "curriculum_gate_completed",
        }:
            continue
        matches = tuple(event.get("matches") or ())
        if not matches or any(
            match.get("opponent") != active_target for match in matches
        ):
            continue
        status = str(event.get("status"))
        score_value = (
            event.get("score")
            if event_type == "curriculum_gate_completed"
            else event.get("benchmark_score")
        )
        return CandidateEvaluation(
            status=status,
            score=(
                float(score_value)
                if status == "complete" and score_value is not None
                else None
            ),
            error=(
                None
                if status == "complete"
                else "historical curriculum gate is incomplete"
            ),
            matches=matches,
        )
    return None


def _curriculum_resume_parent(
    events: list[dict[str, Any]],
    *,
    state: Any,
    lineage_head_version_id: str,
    rollback_patience: int | None = None,
) -> str:
    """Choose the safe parent implied by the latest curriculum decision."""

    boundary_types = {
        "candidate_selected",
        "curriculum_candidate_rejected",
        "curriculum_stagnated",
        "curriculum_resumed",
        "curriculum_stage_promoted",
        "curriculum_gate_completed",
    }
    latest = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") in boundary_types
        ),
        None,
    )
    if latest is not None:
        if latest.get("event_type") == "curriculum_candidate_rejected":
            return str(state.stage_origin_version_id)
        if latest.get("event_type") == "curriculum_stagnated":
            return str(state.stage_best_version_id)
        if latest.get("event_type") == "curriculum_resumed":
            return str(latest["stage_best_version_id"])
        if (
            latest.get("event_type") == "curriculum_gate_completed"
            and rollback_patience is not None
            and int(latest.get("stagnation_count", 0))
            >= rollback_patience
        ):
            return str(state.stage_best_version_id)
    return lineage_head_version_id


def _cmd_validate(args: argparse.Namespace) -> int:
    _json(_validate(_load(args.config)))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    config = _load(args.config)
    result = audit_sources(
        config.paths.agentbench_root,
        config.paths.official_logic_root,
        config.paths.pacman_sdk_root,
    )
    if not result["valid"]:
        _json(result)
        return 2
    _json(result)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load(args.config)
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _default_run_dir(config)
    workspace = (
        Path(args.workspace).resolve()
        if args.workspace
        else config.paths.workspace
    )
    if args.dry_run:
        _json(_dry_run(config, run_dir=run_dir, workspace=workspace))
        return 0
    needs_provider = (
        args.acts != 0 or config.run.origin.mode == "model_bootstrap"
    )
    provider_environment = (
        _provider_environment(config) if needs_provider else None
    )
    if needs_provider and provider_environment is None:
        print(
            f"missing provider credential: {config.run.provider.env_key}",
            file=sys.stderr,
        )
        return 2
    return _run_real(
        config,
        run_dir=run_dir,
        workspace=workspace,
        acts=args.acts,
        resume=False,
        provider_environment=provider_environment,
    )


def _cmd_resume(args: argparse.Namespace) -> int:
    config = _load(args.config)
    needs_provider = (
        args.acts != 0 or config.run.origin.mode == "model_bootstrap"
    )
    provider_environment = (
        _provider_environment(config) if needs_provider else None
    )
    if needs_provider and provider_environment is None:
        print(
            f"missing provider credential: {config.run.provider.env_key}",
            file=sys.stderr,
        )
        return 2
    return _run_real(
        config,
        run_dir=Path(args.run_dir).resolve(),
        workspace=(
            Path(args.workspace).resolve()
            if args.workspace
            else config.paths.workspace
        ),
        acts=args.acts,
        resume=True,
        provider_environment=provider_environment,
    )


def _cmd_prepare(args: argparse.Namespace) -> int:
    from agentbench_frame.games.rollman.opponents import (
        prepare_human_pool,
        verify_opponent_start,
    )

    config = _load(args.config)
    pool = load_human_pool(config.paths.human_manifest)
    requested = {
        int(value)
        for value in (args.ranks.split(",") if args.ranks else [])
    }
    selected = (
        tuple(opponent for opponent in pool if opponent.rank in requested)
        if requested
        else pool
    )
    if requested and len(selected) != len(requested):
        raise ValueError("requested rank is outside 1..16")
    prepared = prepare_human_pool(
        selected,
        build_root=config.paths.opponent_build_root,
    )
    for opponent in prepared:
        verify_opponent_start(opponent)
    _json(
        {
            "prepared": [
                {
                    "opponent": opponent.opponent_id,
                    "rank": opponent.rank,
                    "command": list(opponent.process.argv),
                }
                for opponent in prepared
                if opponent.process is not None
            ]
        }
    )
    return 0


def _run_real(
    config: LocalHLConfig,
    *,
    run_dir: Path,
    workspace: Path,
    acts: int | None,
    resume: bool,
    provider_environment: dict[str, str] | None,
) -> int:
    # Imported lazily so validate and dry-run never initialize model/runtime state.
    from agentbench_frame.games.rollman.candidate_runner import __file__ as candidate_runner
    from agentbench_frame.games.rollman.diagnostics import (
        calibrate_opponent_difficulty,
    )
    from agentbench_frame.games.rollman.evaluator import RollmanEvaluator
    from agentbench_frame.games.rollman.logic_runner import __file__ as logic_runner
    from agentbench_frame.games.rollman.match import ProcessSpec, run_match
    from agentbench_frame.games.rollman.opponents import (
        prepare_human_pool,
        prepare_opponent,
    )
    from agentbench_frame.games.rollman.research_measurement import (
        RollmanMeasurementRunner,
    )
    from agentbench_frame.games.rollman.state_tracker import FrozenStateTracker
    from agentbench_frame.hl.codebase import Version, VersionStore
    from agentbench_frame.hl.context import IterationContext, compile_game_digest
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.curriculum import (
        CurriculumManager,
        rerank_certification,
        summarize_certification,
    )
    from agentbench_frame.hl.events import HLEventWriter, read_events
    from agentbench_frame.hl.lineage import LineageManager
    from agentbench_frame.hl.provider import CodexSessionProvider
    from agentbench_frame.hl.research_state import ResearchState
    _validate(config)
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    config_snapshot = _frozen_run_config(config)
    snapshot_path = run_dir / "run-config.json"
    if resume:
        persisted = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if _normalize_frozen_run_config(persisted) != _normalize_frozen_run_config(
            config_snapshot
        ):
            raise ValueError("resume config differs from the frozen run config")
    else:
        snapshot_path.write_text(
            json.dumps(
                config_snapshot,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        source_audit = audit_sources(
            config.paths.agentbench_root,
            config.paths.official_logic_root,
            config.paths.pacman_sdk_root,
        )
        (run_dir / "source-audit.json").write_text(
            json.dumps(
                source_audit,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    _ensure_candidate(workspace)
    pool = load_human_pool(config.paths.human_manifest)
    curriculum_mode = config.run.curriculum.mode == "weakest_failed"
    prepared_pool = (
        prepare_human_pool(
            pool,
            build_root=config.paths.opponent_build_root,
        )
        if curriculum_mode
        else ()
    )
    if curriculum_mode:
        initial_learning_opponent = prepared_pool[0]
    else:
        rank1_raw = next(
            opponent
            for opponent in pool
            if opponent.opponent_id
            == config.run.evaluation.learning_opponent
        )
        initial_learning_opponent = prepare_opponent(
            rank1_raw,
            build_root=config.paths.opponent_build_root,
        )
    logic_root = (
        config.paths.agentbench_root
        / "backend_sources/corpus/29_rollman/logic/gamecode_logic/PacmanLogic"
    )
    logic = ProcessSpec(
        argv=(
            sys.executable,
            str(Path(logic_runner).resolve()),
            "--logic-root",
            str(logic_root),
        ),
        cwd=logic_root,
    )
    def candidate_factory(version: Version) -> ProcessSpec:
        snapshot_workspace = (
            run_dir / "versions" / "objects" / version.content_hash
        )
        if not snapshot_workspace.is_dir():
            raise FileNotFoundError(
                f"missing candidate snapshot object: {version.content_hash}"
            )
        return ProcessSpec(
            argv=(
                sys.executable,
                str(Path(candidate_runner).resolve()),
                "--workspace",
                str(snapshot_workspace),
                "--sdk-root",
                str(config.paths.pacman_sdk_root),
            ),
            cwd=snapshot_workspace,
            untrusted=True,
            read_roots=(
                snapshot_workspace,
                config.paths.pacman_sdk_root,
                Path(candidate_runner).resolve().parents[3],
            ),
            denied_paths=(config.source_path.parents[2] / ".env",),
        )
    evaluator = RollmanEvaluator(
        logic=logic,
        candidate_factory=candidate_factory,
        learning_opponent=initial_learning_opponent,
        human_pool=prepared_pool,
        fixed_gate_seeds=config.run.evaluation.fixed_gate_seeds,
        certification_seeds=config.run.evaluation.certification_seeds,
        artifact_root=run_dir / "matches",
        match_runner=run_match,
        state_tracker_factory=lambda: FrozenStateTracker(logic_root),
        max_parallel_matches=(
            config.run.evaluation.max_parallel_matches
        ),
    )
    bundle = ContextBundle.create(
        run_dir / "context",
        {
            "rules": asset_path("rules.md"),
            "decision_space": asset_path("decision_space.yaml"),
            "replay_skill": asset_path(
                "replay-skill/rollman-replay"
            ),
        },
    )
    iteration_context = IterationContext(bundle)
    game_digest_path = compile_game_digest(
        bundle,
        run_dir / "context" / "game_digest.json",
    )
    research_state_path = run_dir / "research_state.json"
    if not resume and not research_state_path.is_file():
        ResearchState.empty(
            max_bytes=config.run.context.research_state_max_bytes
        ).write(research_state_path)
    experience = ExperienceManager(
        run_dir / "experience",
        compress_every_acts=config.run.experience.compress_every_acts,
    )
    provider = (
        None
        if provider_environment is None
        else CodexSessionProvider(
            config.run.provider,
            run_root=run_dir,
            environ=provider_environment,
            timeout_s=config.run.provider.timeout_seconds,
            idle_timeout_s=config.run.provider.idle_timeout_seconds,
        )
    )
    events_path = run_dir / "events.jsonl"
    historical = read_events(events_path)
    pending_planner_recovery = (
        _pending_planner_recovery(
            historical,
            provider=provider,
            workspace=workspace,
        )
        if resume and provider is not None
        else None
    )
    bootstrap_recovery = (
        _bootstrap_recovery_candidate(
            historical,
            workspace=workspace,
            provider=provider,
        )
        if resume and provider is not None
        else None
    )
    lineage = (
        LineageManager.from_events(
            historical,
            rollback_patience=config.run.rollback.patience,
            rollback_margin=config.run.rollback.score_margin,
            rollback_enabled=config.run.rollback.enabled,
        )
        if resume
        else LineageManager(
            rollback_patience=config.run.rollback.patience,
            rollback_margin=config.run.rollback.score_margin,
            rollback_enabled=config.run.rollback.enabled,
        )
    )
    writer = HLEventWriter(events_path, run_id=run_dir.name)
    version_store = VersionStore(workspace, run_dir / "versions")
    measurement_runner = RollmanMeasurementRunner(
        root=run_dir / "measurement",
        sdk_root=config.paths.pacman_sdk_root,
        epsilon=config.run.measurement.epsilon,
    )
    evaluations_by_version: dict[str, CandidateEvaluation] = {}
    if resume:
        evaluation_events = {
            str(event["version_id"]): event
            for event in historical
            if event.get("event_type") == "evaluation_completed"
        }
        for event in historical:
            if event.get("event_type") != "version_created":
                continue
            version_id = str(event["version_id"])
            finalized = evaluation_events.get(version_id)
            status = str(
                finalized["status"]
                if finalized is not None
                else event["evaluation_status"]
            )
            evaluations_by_version[version_id] = CandidateEvaluation(
                status=status,
                score=(
                    None
                    if (
                        finalized.get("benchmark_score")
                        if finalized is not None
                        else event.get("benchmark_score")
                    )
                    is None
                    else float(
                        finalized["benchmark_score"]
                        if finalized is not None
                        else event["benchmark_score"]
                    )
                ),
                error=None if status == "complete" else "historical incomplete evaluation",
                matches=(
                    tuple(finalized.get("matches", ()))
                    if finalized is not None
                    else ()
                ),
            )
    pending_repair_recoveries = (
        _pending_repair_recoveries(
            historical,
            version_store=version_store,
            evaluations_by_version=evaluations_by_version,
        )
        if resume
        else {}
    )
    pending_candidate_recoveries = (
        _pending_candidate_recoveries(
            historical,
            version_store=version_store,
            evaluations_by_version=evaluations_by_version,
        )
        if resume
        else {}
    )

    curriculum_manager: CurriculumManager | None = None
    shared_distillation_path: Path | None = None

    def prompt_factory(**values):
        nonlocal shared_distillation_path
        if values.get("bootstrap"):
            return iteration_context.build_bootstrap_prompt(
                act_id=values["act_id"],
                workspace=workspace,
                experience_path=experience.path,
            )
        evaluation = evaluator.last_evaluation
        matches = list(evaluation.matches) if evaluation is not None else []
        summarizer = (
            bundle.files["replay_skill"].parent
            / "scripts"
            / "summarize_replay.py"
        )
        evidence = []
        for match in matches:
            replay = match.get("replay")
            summary_path = (
                None
                if replay is None
                else _ensure_replay_summary(
                    replay=str(replay),
                    summarizer=summarizer,
                )
            )
            evidence.append(
                {
                "opponent": match.get("opponent"),
                "seed": match.get("seed"),
                "result": match.get("result"),
                "summary": (
                    None
                    if summary_path is None
                    else str(summary_path)
                ),
                "replay": replay,
                "trace": match.get("trace"),
                "candidate_fault": _trace_fault_summary(match.get("trace")),
                }
            )
        previous_measurements = {
            "benchmark_score": (
                None if evaluation is None else evaluation.score
            ),
            "evaluation_status": (
                None if evaluation is None else evaluation.status
            ),
            "curriculum_stagnation_count": (
                0
                if curriculum_manager is None
                else curriculum_manager.state.stagnation_count
            ),
        }
        active_target = (
            None
            if curriculum_manager is None
            else curriculum_manager.state.active_target
        )
        locked_opponents = (
            ()
            if curriculum_manager is None
            else curriculum_manager.state.locked_opponents
        )
        phase = values.get("phase", "candidate")
        stagnation_count = int(
            previous_measurements["curriculum_stagnation_count"] or 0
        )
        if phase == "planner":
            shared_distillation_path = None
        if stagnation_count >= 3:
            traces = [
                str(item["trace"])
                for item in evidence
                if item.get("trace") is not None
                and item.get("candidate_fault") is None
            ]
            if shared_distillation_path is None and traces:
                shared_distillation_path = _ensure_opponent_distillation(
                    traces=traces,
                    distillation_tool=(
                        bundle.files["replay_skill"].parent
                        / "scripts"
                        / "distill_opponent_policy.py"
                    ),
                    output_root=(
                        run_dir
                        / "distillation"
                        / str(active_target or "unknown")
                    ),
                )
            if shared_distillation_path is not None:
                previous_measurements["opponent_distillation_path"] = str(
                    shared_distillation_path
                )
        if phase == "planner":
            return iteration_context.build_planner_prompt(
                act_id=values["act_id"],
                iteration_id=values["iteration_id"],
                parent_version_id=values["parent_version_id"],
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                replay_evidence=evidence,
                previous_measurements=previous_measurements,
                active_target=active_target,
                scope_contract_required=(
                    config.run.iteration.scope_contract_required
                ),
            )
        if phase == "repair":
            return iteration_context.build_repair_prompt(
                act_id=values["act_id"],
                iteration_id=values["iteration_id"],
                branch_index=values["branch_index"],
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                repair_input_path=Path(values["repair_input"]),
                experience_path=experience.path,
            )
        if phase == "reducer":
            reducer_input = Path(values["reducer_input"])
            reducer_value = json.loads(
                reducer_input.read_text(encoding="utf-8")
            )
            return iteration_context.build_reducer_prompt(
                act_id=values["act_id"],
                iteration_id=values["iteration_id"],
                selected_version_id=str(
                    reducer_value["selected_version_id"]
                ),
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                reducer_input_path=reducer_input,
            )
        if phase == "candidate" and values.get("branch_brief") is not None:
            return iteration_context.build_candidate_prompt(
                act_id=values["act_id"],
                branch_index=values["branch_index"],
                branch_count=values["branch_count"],
                parent_version_id=values["parent_version_id"],
                workspace=workspace,
                game_digest_path=game_digest_path,
                research_state_path=research_state_path,
                replay_evidence=evidence,
                previous_measurements=previous_measurements,
                experience_path=experience.path,
                branch_brief=values["branch_brief"],
                active_target=active_target,
                locked_opponents=locked_opponents,
                scope_contract_required=(
                    config.run.iteration.scope_contract_required
                ),
            )
        return iteration_context.build_prompt(
            act_id=values["act_id"],
            branch_index=values["branch_index"],
            branch_count=values["branch_count"],
            parent_version_id=values["parent_version_id"],
            workspace=workspace,
            replay_evidence=evidence,
            previous_measurements=previous_measurements,
            experience_path=experience.path,
            active_target=active_target,
            locked_opponents=locked_opponents,
        )

    def repair_summary_resolver(match: dict[str, Any]) -> dict[str, Any]:
        replay = match.get("replay")
        summary_path = (
            None
            if replay is None
            else _ensure_replay_summary(
                replay=str(replay),
                summarizer=(
                    bundle.files["replay_skill"].parent
                    / "scripts"
                    / "summarize_replay.py"
                ),
            )
        )
        return {
            "summary": None if summary_path is None else str(summary_path),
            "replay": replay,
            "trace": match.get("trace"),
            "candidate_fault": _trace_fault_summary(match.get("trace")),
        }

    controller = HLController(
        workspace=workspace,
        run_root=run_dir,
        provider=provider,
        evaluator=evaluator,
        version_store=version_store,
        lineage=lineage,
        events=writer,
        iteration=config.run.iteration,
        rollback=config.run.rollback,
        prompt_factory=prompt_factory,
        experience_manager=experience,
        research_state_path=research_state_path,
        research_state_max_bytes=(
            config.run.context.research_state_max_bytes
        ),
        summary_resolver=repair_summary_resolver,
    )

    def sync_research_state() -> None:
        state = ResearchState.load_or_create(
            research_state_path,
            max_bytes=config.run.context.research_state_max_bytes,
        )
        state.advance(
            search_parent_version_id=(
                controller.lineage.lineage_head_version_id
            ),
            official_champion_version_id=(
                controller.lineage.champion_version_id
            ),
            active_target=(
                None
                if curriculum_manager is None
                else curriculum_manager.state.active_target
            ),
            locked_opponents=(
                ()
                if curriculum_manager is None
                else curriculum_manager.state.locked_opponents
            ),
        ).write(research_state_path)

    def measure_iteration(iteration_result: Any) -> set[str]:
        parent_evaluation = evaluations_by_version.get(
            iteration_result.parent_version_id
        )
        parent_version = version_store.get(
            iteration_result.parent_version_id
        )
        failed: set[str] = set()
        for candidate in _measurement_candidates(iteration_result):
            evaluations_by_version[
                candidate.version.version_id
            ] = candidate.evaluation
            if (
                parent_evaluation is None
                or parent_evaluation.status != "complete"
                or candidate.evaluation.status != "complete"
            ):
                continue
            if not _measure_candidate(
                measurement_runner=measurement_runner,
                version_store=version_store,
                writer=writer,
                candidate_version=candidate.version,
                parent_version=parent_version,
                candidate_evaluation=candidate.evaluation,
                parent_evaluation=parent_evaluation,
            ):
                failed.add(candidate.version.version_id)
        return failed

    def rollback_invalid_candidate(
        *,
        version_id: str,
        parent_version_id: str,
        reason: str,
    ) -> None:
        rollback = controller.lineage.force_parent(
            parent_version_id,
            reason=reason,
        )
        version_store.checkout(parent_version_id)
        if rollback.rollback:
            writer.write(
                "rollback_selected",
                from_version_id=rollback.from_version_id,
                to_version_id=rollback.to_version_id,
                reason=rollback.reason,
            )
        repaired_history = read_events(events_path)
        accepted_updates = _validated_experience_updates(
            repaired_history,
            run_dir,
        )
        experience.rebuild(accepted_updates)
        writer.write(
            "experience_rebuilt",
            rejected_version_id=version_id,
            accepted_updates=len(accepted_updates),
            experience_path=str(experience.path),
        )

    if curriculum_mode:
        opponent_by_id = {
            opponent.opponent_id: opponent
            for opponent in prepared_pool
        }
        difficulty_path = run_dir / "opponent-difficulty.json"
        empirical_order: tuple[str, ...] | None = None
        if difficulty_path.is_file():
            difficulty_value = json.loads(
                difficulty_path.read_text(encoding="utf-8")
            )
            empirical_order = tuple(difficulty_value["hardest_to_easiest"])

        def empirical_rank(opponent_id: str) -> int:
            if empirical_order is None:
                return opponent_by_id[opponent_id].rank
            return empirical_order.index(opponent_id) + 1

        def certify_curriculum_version(version: Any):
            nonlocal empirical_order
            version_store.checkout(version.version_id)
            certification = evaluator.certify(version)
            controller.record_matches(
                version=version,
                act_id=version.act_id,
                phase="certification",
                matches=certification.matches,
            )
            summary = (
                summarize_certification(
                    certification.matches,
                    required_win_rate=(
                        config.run.evaluation.required_win_rate
                    ),
                    expected_opponents=len(pool),
                )
                if certification.status == "complete"
                else None
            )
            if summary is not None:
                if empirical_order is None:
                    empirical_order = calibrate_opponent_difficulty(
                        certification.matches
                    )
                    if len(empirical_order) != len(pool):
                        raise ValueError(
                            "opponent calibration requires every valid human Ghost"
                        )
                    difficulty_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "1.0",
                                "reference_version_id": version.version_id,
                                "hardest_to_easiest": list(empirical_order),
                                "criterion": [
                                    "ghost_points",
                                    "mean_ghost_minus_rollman_margin",
                                    "worst_margin",
                                    "opponent_id",
                                ],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                summary = rerank_certification(
                    summary,
                    hardest_to_easiest=empirical_order,
                )
            writer.write(
                "certification_completed",
                version_id=version.version_id,
                act_id=version.act_id,
                status=certification.status,
                score=certification.score,
                passing_human_opponents=(
                    0 if summary is None else summary.passing_opponents
                ),
                required_human_opponents=(
                    config.run.curriculum.required_human_opponents
                ),
                matches=list(certification.matches),
            )
            return certification, summary

        def write_curriculum_gate(
            *,
            version: Any,
            evaluation: CandidateEvaluation,
            baseline: bool,
            improved: bool,
        ) -> None:
            assert curriculum_manager is not None
            assert curriculum_manager.state.active_target is not None
            assert curriculum_manager.state.stage_best_score is not None
            writer.write(
                "curriculum_gate_completed",
                version_id=version.version_id,
                active_target=curriculum_manager.state.active_target,
                status=evaluation.status,
                score=evaluation.score,
                matches=list(evaluation.matches),
                baseline=baseline,
                improved=improved,
                stagnation_count=(
                    curriculum_manager.state.stagnation_count
                ),
                stage_best_version_id=(
                    curriculum_manager.state.stage_best_version_id
                ),
                stage_best_score=(
                    curriculum_manager.state.stage_best_score
                ),
            )

        certified = any(
            event.get("event_type") == "run_completed"
            and event.get("reason") == "all_human_opponents_defeated"
            for event in historical
        )
        curriculum_has_started = any(
            event.get("event_type") == "curriculum_started"
            for event in historical
        )
        if (
            resume
            and bootstrap_recovery is None
            and curriculum_has_started
        ):
            controller.resume(historical)
            curriculum_manager = CurriculumManager.from_events(
                historical,
                required_human_opponents=(
                    config.run.curriculum.required_human_opponents
                ),
                stagnation_patience=(
                    config.run.curriculum.stagnation_patience
                ),
                rollback_patience=(
                    config.run.rollback.patience
                    if config.run.rollback.enabled
                    else None
                ),
            )
            if certified:
                _json(
                    {
                        "run_dir": str(run_dir),
                        "certified": True,
                        "status": "all_human_opponents_defeated",
                        **controller.summary(),
                    }
                )
                return 0
            pending_failure = _pending_failed_candidate(historical)
            if pending_failure is not None:
                (
                    failed_version_id,
                    failed_parent_id,
                    failure_reason,
                ) = pending_failure
                rollback_invalid_candidate(
                    version_id=failed_version_id,
                    parent_version_id=failed_parent_id,
                    reason=failure_reason,
                )
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": failure_reason,
                        "version_id": failed_version_id,
                        "next_parent_version_id": failed_parent_id,
                        **controller.summary(),
                    }
                )
                return 2
            pending_measurement = _pending_measurement_candidate(
                historical
            )
            if pending_measurement is not None:
                pending_version_id, pending_parent_id = (
                    pending_measurement
                )
                pending_evaluation = evaluations_by_version.get(
                    pending_version_id
                )
                pending_parent_evaluation = evaluations_by_version.get(
                    pending_parent_id
                )
                if (
                    pending_evaluation is None
                    or pending_parent_evaluation is None
                    or pending_evaluation.status != "complete"
                    or pending_parent_evaluation.status != "complete"
                ):
                    raise ValueError(
                        "pending measurement lacks complete evaluations"
                    )
                measured = _measure_candidate(
                    measurement_runner=measurement_runner,
                    version_store=version_store,
                    writer=writer,
                    candidate_version=version_store.get(
                        pending_version_id
                    ),
                    parent_version=version_store.get(
                        pending_parent_id
                    ),
                    candidate_evaluation=pending_evaluation,
                    parent_evaluation=pending_parent_evaluation,
                )
                if not measured:
                    rollback_invalid_candidate(
                        version_id=pending_version_id,
                        parent_version_id=pending_parent_id,
                        reason="measurement_failed",
                    )
                    _json(
                        {
                            "run_dir": str(run_dir),
                            "status": "measurement_failed",
                            "version_id": pending_version_id,
                            "next_parent_version_id": pending_parent_id,
                            **controller.summary(),
                        }
                    )
                    return 2
                historical = read_events(events_path)
            pending_finalization = _pending_proposal_finalization(
                historical
            )
            if pending_finalization is not None:
                (
                    pending_iteration_id,
                    pending_version_id,
                    pending_parent_id,
                ) = pending_finalization
                pending_evaluation = evaluations_by_version.get(
                    pending_version_id
                )
                if (
                    pending_evaluation is None
                    or pending_evaluation.status != "complete"
                    or pending_evaluation.score is None
                ):
                    raise ValueError(
                        "pending proposal finalization lacks a complete evaluation"
                    )
                selected_version = version_store.get(pending_version_id)
                version_store.checkout(pending_version_id)
                active_target = curriculum_manager.state.active_target
                if active_target is None:
                    raise ValueError(
                        "pending proposal finalization has no active target"
                    )
                evaluator.set_learning_opponent(opponent_by_id[active_target])

                reporting_done = any(
                    event.get("event_type") == "reporting_panel_completed"
                    and str(event.get("iteration_id")) == pending_iteration_id
                    and str(event.get("version_id")) == pending_version_id
                    for event in historical
                )
                if (
                    config.run.evaluation.reporting_panel_every_cycle
                    and not reporting_done
                ):
                    reporting = evaluator.evaluate_reporting_panel(
                        selected_version,
                        seeds=tuple(
                            900_001 + offset
                            for offset in range(
                                config.run.evaluation.reporting_seeds_per_opponent
                            )
                        ),
                    )
                    controller.record_matches(
                        version=selected_version,
                        act_id=selected_version.act_id,
                        phase="reporting",
                        matches=reporting.matches,
                    )
                    reporting_margins = [
                        float(match["rollman_score"])
                        - float(match["ghosts_score"])
                        for match in reporting.matches
                        if match.get("status") == "complete"
                        and isinstance(match.get("rollman_score"), (int, float))
                        and isinstance(match.get("ghosts_score"), (int, float))
                    ]
                    writer.write(
                        "reporting_panel_completed",
                        iteration_id=pending_iteration_id,
                        proposal_cycle=int(
                            pending_iteration_id.rsplit("-", 1)[-1]
                        ),
                        version_id=pending_version_id,
                        status=reporting.status,
                        score=reporting.score,
                        mean_score_margin=(
                            sum(reporting_margins) / len(reporting_margins)
                            if reporting_margins
                            else None
                        ),
                        matches=list(reporting.matches),
                    )
                    if reporting.status != "complete":
                        _json(
                            {
                                "run_dir": str(run_dir),
                                "status": "incomplete_reporting_panel",
                                "version_id": pending_version_id,
                                **controller.summary(),
                            }
                        )
                        return 2

                act_id = selected_version.act_id
                pending_experience = (
                    run_dir / "experience" / "pending" / f"{act_id}.json"
                )
                experience_done = any(
                    event.get("event_type") == "experience_updated"
                    and str(event.get("act_id")) == act_id
                    for event in historical
                )
                if pending_experience.is_file() and not experience_done:
                    experience_path = experience.apply_file(
                        act_id,
                        pending_experience,
                    )
                    writer.write(
                        "experience_updated",
                        act_id=act_id,
                        version_id=pending_version_id,
                        experience_path=str(experience_path),
                    )

                evaluator.last_evaluation = pending_evaluation
                gate_decision = curriculum_manager.observe_gate(
                    version_id=pending_version_id,
                    score=pending_evaluation.score,
                    tie_break_improved=(
                        pending_version_id != pending_parent_id
                    ),
                )
                write_curriculum_gate(
                    version=selected_version,
                    evaluation=pending_evaluation,
                    baseline=False,
                    improved=gate_decision.kind == "improved",
                )
                state = ResearchState.load_or_create(
                    research_state_path,
                    max_bytes=config.run.context.research_state_max_bytes,
                )
                state.advance(
                    search_parent_version_id=pending_version_id,
                    exploration_debt=(
                        0
                        if pending_version_id != pending_parent_id
                        else state.exploration_debt
                    ),
                ).write(research_state_path)
                sync_research_state()
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": "proposal_cycle_recovered",
                        "version_id": pending_version_id,
                        "active_target": active_target,
                        **controller.summary(),
                    }
                )
                return 0
            active_target = curriculum_manager.state.active_target
            if active_target is None:
                raise ValueError(
                    "active curriculum run is missing a target"
                )
            latest_curriculum_boundary = next(
                (
                    event
                    for event in reversed(historical)
                    if event.get("event_type")
                    in {
                        "curriculum_stagnated",
                        "curriculum_resumed",
                        "curriculum_stage_promoted",
                        "curriculum_target_selected",
                    }
                ),
                None,
            )
            if (
                latest_curriculum_boundary is not None
                and latest_curriculum_boundary.get("event_type")
                == "curriculum_stagnated"
                and (acts is None or acts > 0)
            ):
                resumed_parent = (
                    curriculum_manager.resume_after_stagnation()
                )
                writer.write(
                    "curriculum_resumed",
                    version_id=resumed_parent,
                    active_target=active_target,
                    stage_best_version_id=resumed_parent,
                    stage_best_score=(
                        curriculum_manager.state.stage_best_score
                    ),
                )
            evaluator.set_learning_opponent(
                opponent_by_id[active_target]
            )
            head = controller.lineage.lineage_head_version_id
            if head is None:
                raise ValueError("curriculum resume has no lineage head")
            next_parent = _curriculum_resume_parent(
                historical,
                state=curriculum_manager.state,
                lineage_head_version_id=head,
                rollback_patience=(
                    config.run.rollback.patience
                    if config.run.rollback.enabled
                    else None
                ),
            )
            evaluator.last_evaluation = (
                _curriculum_evaluation_from_events(
                    historical,
                    version_id=next_parent,
                    active_target=active_target,
                )
            )
            if (
                evaluator.last_evaluation is None
                or evaluator.last_evaluation.status != "complete"
            ):
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": "incomplete_curriculum_state",
                        "version_id": next_parent,
                        **controller.summary(),
                    }
                )
                return 2
            evaluations_by_version[
                next_parent
            ] = evaluator.last_evaluation
            if not measurement_runner.manifest_path.is_file():
                measurement_runner.freeze_reference(
                    evaluator.last_evaluation
                )
        else:
            if bootstrap_recovery is not None:
                origin = controller.recover_bootstrap(**bootstrap_recovery)
            elif resume:
                controller.resume(historical)
                origin_id = controller.lineage.lineage_head_version_id
                if origin_id is None:
                    raise ValueError("pre-curriculum resume has no origin")
                origin = version_store.get(origin_id)
                if (
                    controller.lineage.versions[origin_id].status
                    != "complete"
                ):
                    evaluator.last_evaluation = (
                        controller.retry_head_evaluation()
                    )
            elif config.run.origin.mode == "imported_version":
                assert config.run.origin.source_run is not None
                assert config.run.origin.source_version is not None
                origin = controller.initialize_imported(
                    source_run=config.run.origin.source_run,
                    source_version_id=config.run.origin.source_version,
                )
            else:
                if provider is None:
                    raise RuntimeError(
                        "provider credential is required for model bootstrap"
                    )
                origin = controller.bootstrap().version
            if (
                evaluator.last_evaluation is None
                or evaluator.last_evaluation.status != "complete"
            ):
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": "incomplete_origin_evaluation",
                        "version_id": origin.version_id,
                        **controller.summary(),
                    }
                )
                return 2
            certification, origin_summary = certify_curriculum_version(
                origin
            )
            if (
                certification.status != "complete"
                or origin_summary is None
            ):
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": "incomplete_certification",
                        "version_id": origin.version_id,
                        **controller.summary(),
                    }
                )
                return 2
            curriculum_manager = CurriculumManager.start(
                version_id=origin.version_id,
                summary=origin_summary,
                required_human_opponents=(
                    config.run.curriculum.required_human_opponents
                ),
                stagnation_patience=(
                    config.run.curriculum.stagnation_patience
                ),
                rollback_patience=(
                    config.run.rollback.patience
                    if config.run.rollback.enabled
                    else None
                ),
            )
            active_target = curriculum_manager.state.active_target
            writer.write(
                "curriculum_started",
                version_id=origin.version_id,
                active_target=active_target,
                active_target_rank=(
                    None
                    if active_target is None
                    else empirical_rank(active_target)
                ),
                locked_opponents=list(
                    curriculum_manager.state.locked_opponents
                ),
                required_human_opponents=(
                    config.run.curriculum.required_human_opponents
                ),
                stage_origin_version_id=origin.version_id,
            )
            if curriculum_manager.state.completed:
                writer.write(
                    "run_completed",
                    reason="all_human_opponents_defeated",
                    version_id=origin.version_id,
                    passing_human_opponents=(
                        origin_summary.passing_opponents
                    ),
                )
                _json(
                    {
                        "run_dir": str(run_dir),
                        "certified": True,
                        "status": "all_human_opponents_defeated",
                        **controller.summary(),
                    }
                )
                return 0
            assert active_target is not None
            writer.write(
                "curriculum_target_selected",
                version_id=origin.version_id,
                active_target=active_target,
                active_target_rank=empirical_rank(active_target),
                locked_opponents=list(
                    curriculum_manager.state.locked_opponents
                ),
                stage_origin_version_id=origin.version_id,
            )
            evaluator.set_learning_opponent(
                opponent_by_id[active_target]
            )
            if (
                controller.lineage.versions[origin.version_id].status
                == "complete"
            ):
                evaluator.last_evaluation = evaluator.evaluate(origin)
                controller.record_matches(
                    version=origin,
                    act_id=origin.act_id,
                    phase="learning",
                    matches=evaluator.last_evaluation.matches,
                )
            else:
                evaluator.last_evaluation = controller.retry_head_evaluation()
            if evaluator.last_evaluation.status != "complete":
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": "incomplete_evaluation",
                        "version_id": origin.version_id,
                        **controller.summary(),
                    }
                )
                return 2
            evaluations_by_version[
                origin.version_id
            ] = evaluator.last_evaluation
            assert evaluator.last_evaluation.score is not None
            curriculum_manager.begin_stage_gate(
                version_id=origin.version_id,
                score=evaluator.last_evaluation.score,
            )
            write_curriculum_gate(
                version=origin,
                evaluation=evaluator.last_evaluation,
                baseline=True,
                improved=True,
            )
            measurement_runner.freeze_reference(
                evaluator.last_evaluation
            )
            next_parent = origin.version_id
            sync_research_state()

        completed_here = 0
        while not certified and (
            acts is None or completed_here < acts
        ):
            if controller.reached_iteration_limit():
                break
            if provider is None:
                raise RuntimeError(
                    "provider credential is required for a model act"
                )
            iteration_result = (
                controller.run_proposal_cycle(
                    parent_version_id=next_parent,
                    defer_experience=True,
                    parent_evaluation=evaluations_by_version.get(
                        next_parent
                    ),
                    planner_recovery=pending_planner_recovery,
                    candidate_recoveries=pending_candidate_recoveries,
                    repair_recoveries=pending_repair_recoveries,
                )
                if config.run.iteration.planner_enabled
                else controller.run_act(
                    parent_version_id=next_parent,
                    defer_experience=True,
                )
            )
            pending_planner_recovery = None
            pending_candidate_recoveries = {}
            pending_repair_recoveries = {}
            completed_here += 1
            stop_reason = _iteration_stop_reason(iteration_result)
            if stop_reason is not None:
                rollback_invalid_candidate(
                    version_id=(
                        iteration_result.selected.version.version_id
                    ),
                    parent_version_id=(
                        iteration_result.parent_version_id
                    ),
                    reason=stop_reason,
                )
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": stop_reason,
                        "next_parent_version_id": (
                            iteration_result.parent_version_id
                        ),
                        **controller.summary(),
                    }
                )
                return 2
            failed_measurements = measure_iteration(iteration_result)
            best_candidate = iteration_result.selected
            if best_candidate.version.version_id in failed_measurements:
                rollback_invalid_candidate(
                    version_id=best_candidate.version.version_id,
                    parent_version_id=iteration_result.parent_version_id,
                    reason="measurement_failed",
                )
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": "measurement_failed",
                        "version_id": best_candidate.version.version_id,
                        "next_parent_version_id": (
                            iteration_result.parent_version_id
                        ),
                        **controller.summary(),
                    }
                )
                return 2
            search_parent_id = iteration_result.search_parent_version_id
            if search_parent_id == best_candidate.version.version_id:
                selected = best_candidate
                controller.commit_experience(selected)
            else:
                parent_evaluation = evaluations_by_version.get(
                    search_parent_id
                )
                if (
                    parent_evaluation is None
                    or parent_evaluation.status != "complete"
                ):
                    raise ValueError(
                        "retained search parent lacks a complete evaluation"
                    )
                parent_version = version_store.get(search_parent_id)
                selected = dataclasses.replace(
                    best_candidate,
                    act_id=parent_version.act_id,
                    branch_index=-1,
                    version=parent_version,
                    evaluation=parent_evaluation,
                    pending_experience_path=None,
                )
            if config.run.evaluation.reporting_panel_every_cycle:
                reporting = evaluator.evaluate_reporting_panel(
                    selected.version,
                    seeds=tuple(
                        900_001 + offset
                        for offset in range(
                            config.run.evaluation.reporting_seeds_per_opponent
                        )
                    ),
                )
                controller.record_matches(
                    version=selected.version,
                    act_id=selected.act_id,
                    phase="reporting",
                    matches=reporting.matches,
                )
                reporting_margins = [
                    float(match["rollman_score"])
                    - float(match["ghosts_score"])
                    for match in reporting.matches
                    if match.get("status") == "complete"
                    and isinstance(match.get("rollman_score"), (int, float))
                    and isinstance(match.get("ghosts_score"), (int, float))
                ]
                writer.write(
                    "reporting_panel_completed",
                    iteration_id=iteration_result.iteration_id,
                    proposal_cycle=int(
                        iteration_result.iteration_id.rsplit("-", 1)[-1]
                    ),
                    version_id=selected.version.version_id,
                    status=reporting.status,
                    score=reporting.score,
                    mean_score_margin=(
                        sum(reporting_margins) / len(reporting_margins)
                        if reporting_margins
                        else None
                    ),
                    matches=list(reporting.matches),
                )
                if reporting.status != "complete":
                    _json(
                        {
                            "run_dir": str(run_dir),
                            "status": "incomplete_reporting_panel",
                            "version_id": selected.version.version_id,
                            **controller.summary(),
                        }
                    )
                    return 2
            evaluator.last_evaluation = selected.evaluation
            assert selected.evaluation.score is not None
            gate_decision = curriculum_manager.observe_gate(
                version_id=selected.version.version_id,
                score=selected.evaluation.score,
                tie_break_improved=(
                    selected.version.version_id
                    != iteration_result.parent_version_id
                ),
            )
            write_curriculum_gate(
                version=selected.version,
                evaluation=selected.evaluation,
                baseline=False,
                improved=gate_decision.kind == "improved",
            )
            certification = None
            summary = None
            if config.run.evaluation.full_pool_every_iteration:
                certification, summary = certify_curriculum_version(
                    selected.version
                )
                if certification.status != "complete" or summary is None:
                    _json(
                        {
                            "run_dir": str(run_dir),
                            "status": "incomplete_certification",
                            "version_id": selected.version.version_id,
                            **controller.summary(),
                        }
                    )
                    return 2
            next_parent = (
                selected.version.version_id
                if config.run.iteration.planner_enabled
                else gate_decision.parent_version_id
            )
            if config.run.iteration.planner_enabled:
                state = ResearchState.load_or_create(
                    research_state_path,
                    max_bytes=config.run.context.research_state_max_bytes,
                )
                state.advance(
                    search_parent_version_id=next_parent,
                    exploration_debt=(
                        0
                        if next_parent != iteration_result.parent_version_id
                        else state.exploration_debt
                    ),
                ).write(research_state_path)
            sync_research_state()
            if gate_decision.kind == "stagnated":
                writer.write(
                    "curriculum_stagnated",
                    version_id=selected.version.version_id,
                    active_target=(
                        curriculum_manager.state.active_target
                    ),
                    stage_best_version_id=(
                        curriculum_manager.state.stage_best_version_id
                    ),
                    stage_best_score=(
                        curriculum_manager.state.stage_best_score
                    ),
                    stagnation_count=(
                        curriculum_manager.state.stagnation_count
                    ),
                )
                if config.run.iteration.planner_enabled:
                    resumed_parent = curriculum_manager.resume_after_stagnation()
                    writer.write(
                        "curriculum_resumed",
                        version_id=next_parent,
                        active_target=(
                            curriculum_manager.state.active_target
                        ),
                        stage_best_version_id=resumed_parent,
                        stage_best_score=(
                            curriculum_manager.state.stage_best_score
                        ),
                    )
                else:
                    _json(
                        {
                            "run_dir": str(run_dir),
                            "certified": False,
                            "status": "stagnated",
                            "active_target": (
                                curriculum_manager.state.active_target
                            ),
                            **controller.summary(),
                        }
                    )
                    return 0
            if (
                selected.evaluation.score
                < config.run.evaluation.required_win_rate
            ):
                continue
            if certification is None:
                certification, summary = certify_curriculum_version(
                    selected.version
                )
                if certification.status != "complete" or summary is None:
                    _json(
                        {
                            "run_dir": str(run_dir),
                            "status": "incomplete_certification",
                            "version_id": selected.version.version_id,
                            **controller.summary(),
                        }
                    )
                    return 2
            assert summary is not None
            completed_target = (
                curriculum_manager.state.active_target
            )
            certification_decision = (
                curriculum_manager.observe_certification(
                    version_id=selected.version.version_id,
                    summary=summary,
                )
            )
            if certification_decision.kind == "reject":
                writer.write(
                    "curriculum_candidate_rejected",
                    version_id=selected.version.version_id,
                    stage_origin_version_id=(
                        curriculum_manager.state.stage_origin_version_id
                    ),
                    active_target=completed_target,
                    lost_locked_opponents=list(
                        certification_decision.lost_locked_opponents
                    ),
                    failed_active_target=(
                        completed_target
                        not in summary.passed_opponents
                    ),
                )
                next_parent = (
                    certification_decision.parent_version_id
                )
                parent_evaluation = (
                    _curriculum_evaluation_from_events(
                        read_events(events_path),
                        version_id=next_parent,
                        active_target=str(completed_target),
                    )
                )
                if parent_evaluation is None:
                    raise ValueError(
                        "rejected curriculum candidate has no safe-parent gate"
                    )
                evaluator.last_evaluation = parent_evaluation
                evaluations_by_version[next_parent] = parent_evaluation
                sync_research_state()
                continue
            if controller.lineage.promote_champion(
                selected.version.version_id
            ):
                writer.write(
                    "champion_promoted",
                    version_id=selected.version.version_id,
                    score=selected.evaluation.score,
                )
            writer.write(
                "curriculum_stage_promoted",
                version_id=selected.version.version_id,
                completed_target=completed_target,
                next_target=certification_decision.next_target,
                locked_opponents=list(
                    curriculum_manager.state.locked_opponents
                ),
                passing_human_opponents=summary.passing_opponents,
            )
            if certification_decision.kind == "complete":
                certified = True
                writer.write(
                    "run_completed",
                    reason="all_human_opponents_defeated",
                    version_id=selected.version.version_id,
                    passing_human_opponents=summary.passing_opponents,
                )
                sync_research_state()
                break
            active_target = certification_decision.next_target
            assert active_target is not None
            writer.write(
                "curriculum_target_selected",
                version_id=selected.version.version_id,
                active_target=active_target,
                active_target_rank=empirical_rank(active_target),
                locked_opponents=list(
                    curriculum_manager.state.locked_opponents
                ),
                stage_origin_version_id=selected.version.version_id,
            )
            evaluator.set_learning_opponent(
                opponent_by_id[active_target]
            )
            version_store.checkout(selected.version.version_id)
            baseline = evaluator.evaluate(selected.version)
            controller.record_matches(
                version=selected.version,
                act_id=selected.version.act_id,
                phase="learning",
                matches=baseline.matches,
            )
            if baseline.status != "complete" or baseline.score is None:
                _json(
                    {
                        "run_dir": str(run_dir),
                        "status": "incomplete_evaluation",
                        "version_id": selected.version.version_id,
                        **controller.summary(),
                    }
                )
                return 2
            curriculum_manager.begin_stage_gate(
                version_id=selected.version.version_id,
                score=baseline.score,
            )
            controller.lineage.begin_stage(
                selected.version.version_id,
                score=baseline.score,
            )
            evaluations_by_version[
                selected.version.version_id
            ] = baseline
            evaluator.last_evaluation = baseline
            write_curriculum_gate(
                version=selected.version,
                evaluation=baseline,
                baseline=True,
                improved=True,
            )
            next_parent = selected.version.version_id
            sync_research_state()

        _json(
            {
                "run_dir": str(run_dir),
                "certified": certified,
                "status": (
                    "all_human_opponents_defeated"
                    if certified
                    else "running"
                ),
                "active_target": (
                    None
                    if curriculum_manager is None
                    else curriculum_manager.state.active_target
                ),
                **controller.summary(),
            }
        )
        return 0

    if resume:
        controller.resume(historical)
        head = controller.lineage.lineage_head_version_id
        evaluator.last_evaluation = evaluations_by_version.get(str(head))
        if (
            evaluator.last_evaluation is None
            or evaluator.last_evaluation.status != "complete"
        ):
            evaluator.last_evaluation = controller.retry_head_evaluation()
            evaluations_by_version[str(head)] = evaluator.last_evaluation
        if evaluator.last_evaluation.status != "complete":
            _json(
                {
                    "run_dir": str(run_dir),
                    "status": "incomplete_evaluation",
                    "version_id": str(head),
                    "coding_agent_acts": controller.summary()["coding_agent_acts"],
                }
            )
            return 2
        if not measurement_runner.manifest_path.is_file():
            measurement_runner.freeze_reference(evaluator.last_evaluation)
    else:
        origin_result = controller.bootstrap()
        origin = origin_result.version
        assert evaluator.last_evaluation is not None
        evaluations_by_version[origin.version_id] = evaluator.last_evaluation
        if evaluator.last_evaluation.status != "complete":
            _json(
                {
                    "run_dir": str(run_dir),
                    "status": "incomplete_evaluation",
                    "version_id": origin.version_id,
                    "coding_agent_acts": controller.summary()["coding_agent_acts"],
                }
            )
            return 2
        measurement_runner.freeze_reference(evaluator.last_evaluation)
    completed_certifications = {
        str(event["version_id"])
        for event in historical
        if event.get("event_type") == "certification_completed"
        and event.get("status") == "complete"
    }
    certified = any(
        event.get("event_type") == "run_completed"
        and event.get("reason") == "human_pool_target_reached"
        for event in historical
    )
    certification_pool_prepared = False

    def certify_eligible_head() -> bool:
        nonlocal certification_pool_prepared
        version_id = _eligible_certification_version_id(
            lineage_head_version_id=controller.lineage.lineage_head_version_id,
            evaluations_by_version=evaluations_by_version,
            completed_certifications=completed_certifications,
            required_score=config.run.evaluation.required_win_rate,
        )
        if version_id is None:
            return False
        if not certification_pool_prepared:
            evaluator.human_pool = prepare_human_pool(
                pool,
                build_root=config.paths.opponent_build_root,
            )
            certification_pool_prepared = True
        selected_version = version_store.get(version_id)
        certification = evaluator.certify(selected_version)
        controller.record_matches(
            version=selected_version,
            act_id=selected_version.act_id,
            phase="certification",
            matches=certification.matches,
        )
        per_opponent: dict[str, list[str]] = {}
        for match in certification.matches:
            if match.get("status") == "complete":
                per_opponent.setdefault(str(match["opponent"]), []).append(
                    str(match["result"])
                )
        passing = 0
        for results in per_opponent.values():
            rate = (
                sum(result == "win" for result in results)
                + 0.5 * sum(result == "draw" for result in results)
            ) / len(results)
            passing += rate >= config.run.evaluation.required_win_rate
        writer.write(
            "certification_completed",
            version_id=selected_version.version_id,
            act_id=selected_version.act_id,
            status=certification.status,
            score=certification.score,
            passing_human_opponents=passing,
            required_human_opponents=(
                config.run.evaluation.required_human_opponents
            ),
            matches=list(certification.matches),
        )
        if certification.status == "complete":
            completed_certifications.add(selected_version.version_id)
        if (
            certification.status == "complete"
            and passing >= config.run.evaluation.required_human_opponents
        ):
            writer.write(
                "run_completed",
                reason="human_pool_target_reached",
                version_id=selected_version.version_id,
                passing_human_opponents=passing,
            )
            return True
        return False

    if not certified:
        certified = certify_eligible_head()
    completed_here = 0 if resume else 1
    while not certified and (acts is None or completed_here < acts):
        if controller.reached_iteration_limit():
            break
        iteration_result = controller.run_act()
        completed_here += 1
        stop_reason = _iteration_stop_reason(iteration_result)
        if stop_reason is not None:
            _json(
                {
                    "run_dir": str(run_dir),
                    "status": stop_reason,
                    **controller.summary(),
                }
            )
            return 2
        parent_evaluation = evaluations_by_version.get(
            iteration_result.parent_version_id
        )
        parent_version = version_store.get(iteration_result.parent_version_id)
        for candidate in iteration_result.candidates:
            evaluations_by_version[candidate.version.version_id] = candidate.evaluation
            if (
                parent_evaluation is None
                or parent_evaluation.status != "complete"
                or candidate.evaluation.status != "complete"
            ):
                continue
            metrics = measurement_runner.measure(
                new_version=candidate.version,
                old_version=parent_version,
                version_store=version_store,
                new_evaluation=candidate.evaluation,
                old_evaluation=parent_evaluation,
            )
            writer.write(
                "policy_kl_measured",
                version_id=candidate.version.version_id,
                parent_version_id=parent_version.version_id,
                epsilon=metrics["epsilon"],
                action_support=metrics["action_support"],
                local_policy_kl_trace=metrics["local_policy_kl_trace"],
                episode_local_policy_kl=metrics["episode_local_policy_kl"],
                reference_manifest=metrics["reference_manifest"],
            )
            writer.write(
                "occupancy_measured",
                version_id=candidate.version.version_id,
                parent_version_id=parent_version.version_id,
                occupancy_shift=metrics["occupancy_shift"],
            )
        certified = certify_eligible_head()
    _json(
        {
            "run_dir": str(run_dir),
            "certified": certified,
            **controller.summary(),
        }
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from agentbench_frame.hl.events import read_events
    from agentbench_frame.hl.report import write_hl_report

    run_dir = Path(args.run_dir).resolve()
    outputs = write_hl_report(
        read_events(run_dir / "events.jsonl"),
        run_dir / "report",
    )
    _json({key: str(value) for key, value in outputs.items()})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentbench hl")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.set_defaults(handler=_cmd_validate)
    audit = sub.add_parser("audit")
    audit.add_argument("--config", required=True)
    audit.set_defaults(handler=_cmd_audit)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--run-dir")
    run.add_argument("--workspace")
    run.add_argument("--acts", type=int)
    run.set_defaults(handler=_cmd_run)
    resume = sub.add_parser("resume")
    resume.add_argument("--config", required=True)
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--workspace")
    resume.add_argument("--acts", type=int)
    resume.set_defaults(handler=_cmd_resume)
    prepare = sub.add_parser("prepare-opponents")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--ranks")
    prepare.set_defaults(handler=_cmd_prepare)
    report = sub.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.set_defaults(handler=_cmd_report)
    args = parser.parse_args(argv)
    if getattr(args, "acts", None) is not None:
        minimum = 0
        if args.acts < minimum:
            parser.error(f"--acts must be >= {minimum} for {args.command}")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
