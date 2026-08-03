"""Profile-driven local HL run loop shared by non-legacy games."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agentbench_frame.hl.codebase import Version, VersionStore
from agentbench_frame.hl.context import ContextBundle, IterationContext, compile_game_digest
from agentbench_frame.hl.controller import HLController, ProposalCycleResult
from agentbench_frame.hl.evaluator import CandidateEvaluation
from agentbench_frame.hl.events import HLEventWriter, read_events
from agentbench_frame.hl.experience import ExperienceManager
from agentbench_frame.hl.experience_ledger import ExperienceLedger
from agentbench_frame.hl.game_profile import HLGameBindings, get_game_profile
from agentbench_frame.hl.lineage import LineageManager
from agentbench_frame.hl.config import HLRunConfig
from agentbench_frame.hl.local_config import LocalHLConfig
from agentbench_frame.hl.match_record import MatchRecord
from agentbench_frame.hl.proposal import (
    write_bootstrap_input_packet,
    write_candidate_input_packet,
    write_planner_input_packet,
)
from agentbench_frame.hl.provider import CodexSessionProvider
from agentbench_frame.hl.research_state import ResearchState


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _behavior_comparison_payload(comparison: Any) -> dict[str, Any]:
    return _json_native(
        {
            "status": comparison.status,
            "decision_count": comparison.decision_count,
            "changed_action_count": comparison.changed_action_count,
            "details": dict(comparison.details),
        }
    )


def frozen_config(config: LocalHLConfig) -> dict[str, Any]:
    return _json_native(
        {
            "schema_version": "1.0",
            "source_config": str(config.source_path),
            "source_config_sha256": hashlib.sha256(
                config.source_path.read_bytes()
            ).hexdigest(),
            "run": config.run.to_dict(),
            "paths": {
                name: str(path)
                for name, path in sorted(config.paths.values.items())
            },
        }
    )


def _normalized_frozen_config(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_native(value)
    normalized["run"] = _json_native(
        HLRunConfig.from_mapping(normalized["run"]).to_dict()
    )
    return normalized


def _resume_compatibility_transition(
    frozen: Mapping[str, Any],
    active: Mapping[str, Any],
    *,
    allow_provider_compatibility_change: bool,
) -> dict[str, str] | None:
    frozen_normalized = _normalized_frozen_config(frozen)
    active_normalized = _normalized_frozen_config(active)
    if frozen_normalized == active_normalized:
        return None
    if not allow_provider_compatibility_change:
        raise ValueError("resume config differs from frozen run config")

    frozen_mode = str(
        frozen_normalized["run"]["provider"]["structured_output_mode"]
    )
    active_mode = str(
        active_normalized["run"]["provider"]["structured_output_mode"]
    )
    if frozen_mode == active_mode:
        raise ValueError(
            "provider compatibility override only permits "
            "run.provider.structured_output_mode"
        )
    comparable = _json_native(frozen_normalized)
    comparable["run"]["provider"]["structured_output_mode"] = active_mode
    comparable["source_config_sha256"] = active_normalized[
        "source_config_sha256"
    ]
    if comparable != active_normalized:
        raise ValueError(
            "provider compatibility override only permits "
            "run.provider.structured_output_mode"
        )
    return {
        "field": "run.provider.structured_output_mode",
        "frozen_value": frozen_mode,
        "active_value": active_mode,
        "reason": "provider_native_schema_incompatible",
    }


def _pending_cycle_recoveries(
    historical: list[dict[str, Any]],
    *,
    provider: Any,
    workspace: str | Path,
    version_store: Any,
    evaluations_by_version: dict[str, CandidateEvaluation],
    expected_candidate_count: int,
    policy_entry_symbol: str,
) -> dict[str, Any]:
    from agentbench_frame.hl.cli import (
        _pending_candidate_recoveries,
        _pending_planner_recovery,
        _pending_repair_recoveries,
    )

    completed = {
        str(event.get("iteration_id"))
        for event in historical
        if event.get("event_type") == "proposal_cycle_completed"
    }
    start_index = next(
        (
            index
            for index in range(len(historical) - 1, -1, -1)
            if historical[index].get("event_type")
            == "proposal_cycle_started"
            and str(historical[index].get("iteration_id")) not in completed
        ),
        None,
    )
    if start_index is None:
        return {
            "planner_recovery": None,
            "candidate_recoveries": {},
            "repair_recoveries": {},
        }
    pending = historical[start_index:]
    return {
        "planner_recovery": _pending_planner_recovery(
            pending,
            provider=provider,
            workspace=workspace,
            expected_candidate_count=expected_candidate_count,
            policy_entry_symbol=policy_entry_symbol,
        ),
        "candidate_recoveries": _pending_candidate_recoveries(
            pending,
            version_store=version_store,
            evaluations_by_version=evaluations_by_version,
        ),
        "repair_recoveries": _pending_repair_recoveries(
            pending,
            version_store=version_store,
            evaluations_by_version=evaluations_by_version,
        ),
    }


def _seed_candidate(workspace: Path, template: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if any(path for path in workspace.iterdir() if path.name != ".agentbench"):
        return
    if not template.is_dir():
        raise FileNotFoundError(template)
    for source in sorted(template.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(template)
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _records(values: Sequence[Mapping[str, Any]]) -> tuple[MatchRecord, ...]:
    return tuple(MatchRecord.from_mapping(value) for value in values)


def _references(evaluation: CandidateEvaluation) -> tuple[tuple[Path, str], ...]:
    result = []
    for record in _records(evaluation.matches):
        if record.promotable and record.replay is not None:
            result.append((Path(record.replay), record.candidate_role))
    if not result:
        raise ValueError("behavior measurement requires complete replay references")
    return tuple(result)


def _route_evidence(
    evidence: Sequence[Mapping[str, Any]],
    branch_index: int,
) -> list[Mapping[str, Any]]:
    """Give k=4 siblings small role-balanced views without assuming tactics."""

    complete = [
        item
        for item in evidence
        if item.get("status") == "complete" and item.get("summary")
    ]

    def severity(item: Mapping[str, Any]) -> tuple[float, str, int]:
        margin = item.get("dense_margin")
        return (
            float(margin) if isinstance(margin, (int, float)) else 0.0,
            str(item.get("candidate_role") or ""),
            int(item.get("seed") or 0),
        )

    ordered = sorted(complete, key=severity)
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    for item in ordered:
        by_role.setdefault(str(item.get("candidate_role") or "unknown"), []).append(item)
    roles = sorted(by_role)
    if branch_index < len(roles):
        selected = by_role[roles[branch_index]][:2]
    elif branch_index == 2:
        selected = [items[0] for _, items in sorted(by_role.items()) if items][:2]
    else:
        selected = ordered[-2:]
    return list(selected or ordered[:2])


def _smoke_dict(bindings: HLGameBindings, workspace: Path) -> dict[str, Any]:
    result = bindings.smoke_verifier(workspace=workspace)
    return {
        "status": result.status,
        "error": result.error,
        **dict(result.artifacts),
    }


def _materialize_activation_parent(
    *,
    source: Path,
    run_dir: Path,
    content_hash: str,
) -> Path:
    """Expose one immutable parent copy under the provider-audited measurement root."""

    destination = run_dir / "measurement" / "activation-parents" / content_hash
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def _inherit_profile_semantic_state(
    *,
    run_dir: Path,
    origin: Any,
    context: Any,
) -> None:
    """Copy bounded Experience/Research checkpoints for an imported origin."""

    if origin.mode != "imported_version":
        return
    assert origin.source_run is not None
    source_run = Path(origin.source_run).resolve()
    if not origin.reset_experience:
        source = source_run / "experience" / "state.json"
        if not source.is_file():
            raise FileNotFoundError(
                "imported profile requested experience continuation but source "
                f"state is missing: {source}"
            )
        if source.stat().st_size > 1024 * 1024:
            raise ValueError("source experience state exceeds 1 MiB")
        destination = run_dir / "experience" / "state.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_ledger = source_run / "experience" / "ledger.jsonl"
        if source_ledger.is_file():
            if source_ledger.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("source experience ledger exceeds 8 MiB")
            ExperienceLedger(source_ledger).load()
            shutil.copy2(source_ledger, destination.with_name("ledger.jsonl"))
    if not origin.reset_research_state:
        source = source_run / "research_state.json"
        if not source.is_file():
            raise FileNotFoundError(
                f"imported profile research state is missing: {source}"
            )
        limit = int(context.research_state_max_bytes)
        if source.stat().st_size > limit:
            raise ValueError(f"source research state exceeds max_bytes={limit}")
        ResearchState.load_or_create(source, max_bytes=limit)
        shutil.copy2(source, run_dir / "research_state.json")


def _load_imported_certification(
    *,
    source_run: str | Path,
    source_version_id: str,
    origin_version: Version,
    game: str,
    evaluation_config: Any,
) -> dict[str, Any] | None:
    """Load an exact-content, exact-evaluation certification as a safe cache."""

    root = Path(source_run).resolve()
    snapshot = root / "run-config.json"
    manifest = root / "versions" / "manifests" / f"{source_version_id}.json"
    events_path = root / "events.jsonl"
    if not snapshot.is_file() or not manifest.is_file() or not events_path.is_file():
        return None
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    source_run_config = frozen.get("run")
    if not isinstance(source_run_config, Mapping):
        return None
    if source_run_config.get("game") != game:
        return None
    expected_evaluation = _json_native(dataclasses.asdict(evaluation_config))
    if source_run_config.get("evaluation") != expected_evaluation:
        return None
    source_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if source_manifest.get("content_hash") != origin_version.content_hash:
        return None
    certification = next(
        (
            event
            for event in reversed(read_events(events_path))
            if event.get("event_type") == "certification_completed"
            and str(event.get("version_id")) == source_version_id
        ),
        None,
    )
    if certification is None:
        return None
    status = str(certification.get("status"))
    score = certification.get("score")
    matches = certification.get("matches")
    if not isinstance(matches, list):
        return None
    try:
        evaluation = CandidateEvaluation(
            status=status,
            score=(
                float(score)
                if isinstance(score, (int, float)) and not isinstance(score, bool)
                else None
            ),
            matches=tuple(matches),
        )
    except (TypeError, ValueError):
        return None
    return {
        "evaluation": evaluation,
        "passing_human_opponents": int(
            certification.get("passing_human_opponents") or 0
        ),
        "source_run_id": root.name,
        "source_version_id": source_version_id,
        "source_event_id": str(certification["event_id"]),
        "content_hash": origin_version.content_hash,
    }


def validate_profile(config: LocalHLConfig) -> dict[str, Any]:
    profile = get_game_profile(config.run.game)
    missing = [
        name
        for name in profile.required_local_paths
        if name not in config.paths.values
    ]
    if missing:
        raise ValueError(f"missing profile paths: {missing}")
    layout_factory = getattr(profile, "layout", None)
    if callable(layout_factory):
        layout = layout_factory(config)
        validator = getattr(layout, "validate", None)
        if callable(validator):
            validator(
                require_positive_control=(
                    config.run.origin.mode == "imported_version"
                )
            )
    prompt = profile.prompt_profile()
    return {
        "valid": True,
        "game": config.run.game,
        "profile": type(profile).__name__,
        "roles": list(prompt.roles),
        "policy_entry_symbol": prompt.policy_entry_symbol,
        "candidate_source_relative": prompt.candidate_source_relative,
        "candidates_per_cycle": config.run.iteration.candidates_per_cycle,
        "planner_enabled": config.run.iteration.planner_enabled,
        "reducer_enabled": config.run.iteration.reducer_enabled,
        "rollback_enabled": config.run.rollback.enabled,
        "origin_mode": config.run.origin.mode,
    }


def prepare_profile_run(
    config: LocalHLConfig,
    *,
    run_dir: Path,
    workspace: Path,
    resume: bool,
) -> tuple[HLGameBindings, ContextBundle, Path, ExperienceManager, Path]:
    profile = get_game_profile(config.run.game)
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        _inherit_profile_semantic_state(
            run_dir=run_dir,
            origin=config.run.origin,
            context=config.run.context,
        )
    bindings = profile.build_bindings(config=config, run_root=run_dir)
    _seed_candidate(workspace, bindings.candidate_template)
    bundle = ContextBundle.create(run_dir / "context", bindings.context_sources)
    digest = compile_game_digest(bundle, run_dir / "context/game_digest.json")
    research = run_dir / "research_state.json"
    if not research.is_file():
        ResearchState.empty(
            max_bytes=config.run.context.research_state_max_bytes
        ).write(research)
    experience = ExperienceManager(
        run_dir / "experience",
        compress_every_acts=config.run.experience.compress_every_acts,
    )
    return bindings, bundle, digest, experience, research


def dry_run_profile(
    config: LocalHLConfig,
    *,
    run_dir: Path,
    workspace: Path,
) -> dict[str, Any]:
    validation = validate_profile(config)
    bindings, bundle, digest, experience, research = prepare_profile_run(
        config,
        run_dir=run_dir,
        workspace=workspace,
        resume=False,
    )
    smoke = _smoke_dict(bindings, workspace)
    if smoke["status"] != "complete":
        raise RuntimeError(f"profile candidate scaffold failed smoke: {smoke['error']}")
    return {
        **validation,
        "dry_run": True,
        "would_call_model": False,
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "context_manifest": str(bundle.manifest_path),
        "context_bundle_hash": bundle.bundle_hash,
        "game_digest": str(digest),
        "experience_skill": str(experience.path),
        "research_state": str(research),
        "smoke": smoke,
    }


def seed_named_origin(
    config: LocalHLConfig,
    *,
    name: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Create a content-addressed source run for a profile-owned control policy."""

    profile = get_game_profile(config.run.game)
    materialize = getattr(profile, "materialize_named_origin", None)
    if not callable(materialize):
        raise ValueError(f"game profile {config.run.game} has no named origins")
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    provenance = materialize(
        config=config,
        name=name,
        destination=workspace,
    )
    store = VersionStore(workspace, run_dir / "versions")
    version = store.snapshot(
        parent_version_id=None,
        act_id=f"named-origin-{name}",
        edit_type="named_origin",
    )
    value = {
        "schema_version": "1.0",
        "game": config.run.game,
        "name": name,
        "version_id": version.version_id,
        "content_hash": version.content_hash,
        "source_run": str(run_dir.resolve()),
        "provenance": provenance,
    }
    (run_dir / "named-origin.json").write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return value


def _historical_evaluations(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, CandidateEvaluation]:
    result = {}
    activation_errors = {
        str(event["version_id"]): (
            "activation_probe_failed: " + str(event["error"])
        )
        for event in events
        if event.get("event_type") == "candidate_activation_measured"
        and event.get("status") == "failed"
        and event.get("error")
        and event.get("version_id") is not None
    }
    for event in events:
        if event.get("event_type") != "evaluation_completed":
            continue
        status = str(event["status"])
        result[str(event["version_id"])] = CandidateEvaluation(
            status=status,
            score=(
                float(event["benchmark_score"])
                if status == "complete"
                else None
            ),
            error=(
                None
                if status == "complete"
                else str(
                    event.get("error")
                    or activation_errors.get(str(event["version_id"]))
                    or "historical incomplete evaluation"
                )
            ),
            matches=tuple(event.get("matches", ())),
        )
    return result


def _passing_opponents(
    matches: Sequence[Mapping[str, Any]],
    *,
    required_win_rate: float,
) -> int:
    outcomes: dict[str, list[float]] = {}
    for match in matches:
        if match.get("status") != "complete":
            continue
        points = match.get("points")
        if isinstance(points, (int, float)):
            outcomes.setdefault(str(match["opponent"]), []).append(float(points))
    return sum(
        sum(points) / len(points) >= required_win_rate
        for points in outcomes.values()
        if points
    )


@dataclasses.dataclass(frozen=True)
class _RunProgress:
    completed_cycles: int
    best_version_id: str
    best_learning_score: float
    best_passing: int
    stagnation: int
    certified: bool


def _resume_progress(
    events: Sequence[Mapping[str, Any]],
    *,
    origin_version_id: str,
) -> _RunProgress:
    """Rebuild runner-owned state so process restarts preserve search semantics."""

    scores: dict[str, float] = {}
    best_version_id = origin_version_id
    best_learning_score = 0.0
    best_passing = 0
    stagnation = 0
    completed_cycles = 0
    certified = False
    for event in events:
        event_type = event.get("event_type")
        if event_type == "evaluation_completed" and event.get("status") == "complete":
            value = event.get("benchmark_score")
            if isinstance(value, (int, float)):
                version_id = str(event["version_id"])
                scores[version_id] = float(value)
                if version_id == origin_version_id and completed_cycles == 0:
                    best_learning_score = float(value)
        elif event_type == "proposal_cycle_completed":
            completed_cycles += 1
            selected_id = str(event["selected_version_id"])
            selected_score = scores.get(selected_id)
            if selected_score is not None and selected_score > best_learning_score:
                best_learning_score = selected_score
                best_version_id = selected_id
                stagnation = 0
            else:
                stagnation += 1
        elif event_type == "certification_completed":
            passing = event.get("passing_human_opponents")
            if isinstance(passing, int) and passing > best_passing:
                best_passing = passing
                best_version_id = str(event["version_id"])
                stagnation = 0
        elif (
            event_type == "run_completed"
            and event.get("reason") == "human_pool_target_reached"
        ):
            certified = True
            best_version_id = str(event.get("version_id") or best_version_id)
    return _RunProgress(
        completed_cycles=completed_cycles,
        best_version_id=best_version_id,
        best_learning_score=best_learning_score,
        best_passing=best_passing,
        stagnation=stagnation,
        certified=certified,
    )


def _reporting_panel_payload(
    evaluation: CandidateEvaluation,
    *,
    iteration_id: str,
    proposal_cycle: int,
    version_id: str,
) -> dict[str, Any]:
    margins = [
        float(match["dense_margin"])
        for match in evaluation.matches
        if match.get("status") == "complete"
        and isinstance(match.get("dense_margin"), (int, float))
    ]
    return {
        "iteration_id": iteration_id,
        "proposal_cycle": proposal_cycle,
        "version_id": version_id,
        "status": evaluation.status,
        "score": evaluation.score,
        "mean_score_margin": (
            sum(margins) / len(margins) if margins else None
        ),
        "matches": list(evaluation.matches),
    }


def _reporting_evaluation(
    evaluator: Any,
    version: Any,
    *,
    seed_count: int,
) -> CandidateEvaluation:
    method = getattr(evaluator, "evaluate_reporting_panel", None)
    if callable(method):
        try:
            return method(version, seed_count=seed_count)
        except TypeError as exc:
            if "seed_count" not in str(exc):
                raise
            seeds = tuple(evaluator.certification_seeds[:seed_count])
            return method(version, seeds=seeds)
    return evaluator.certify(version)


def run_profile(
    config: LocalHLConfig,
    *,
    run_dir: Path,
    workspace: Path,
    acts: int | None,
    resume: bool,
    provider_environment: Mapping[str, str] | None,
    allow_provider_compatibility_change: bool = False,
    replan_pending: bool = False,
) -> dict[str, Any]:
    validation = validate_profile(config)
    bindings, bundle, digest, experience, research = prepare_profile_run(
        config,
        run_dir=run_dir,
        workspace=workspace,
        resume=resume,
    )
    snapshot = run_dir / "run-config.json"
    frozen = frozen_config(config)
    compatibility_transition = None
    if resume:
        compatibility_transition = _resume_compatibility_transition(
            json.loads(snapshot.read_text(encoding="utf-8")),
            frozen,
            allow_provider_compatibility_change=(
                allow_provider_compatibility_change
            ),
        )
    else:
        snapshot.write_text(
            json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    profile = get_game_profile(config.run.game)
    prompt_profile = bindings.prompt_profile
    provider = (
        None
        if provider_environment is None
        else CodexSessionProvider(
            config.run.provider,
            run_root=run_dir,
            environ=dict(provider_environment),
            timeout_s=config.run.provider.timeout_seconds,
            idle_timeout_s=config.run.provider.idle_timeout_seconds,
        )
    )
    if provider is not None:
        provider.preflight()
    events_path = run_dir / "events.jsonl"
    historical = read_events(events_path)
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
    store = VersionStore(workspace, run_dir / "versions")
    writer = HLEventWriter(events_path, run_id=run_dir.name)
    if compatibility_transition is not None:
        writer.write(
            "provider_compatibility_selected",
            **compatibility_transition,
        )
    context = IterationContext(bundle, prompt_profile=prompt_profile)
    evaluations = _historical_evaluations(historical)
    current_evaluation: CandidateEvaluation | None = None
    previous_measurements: dict[str, Any] = {}

    def evidence() -> list[Mapping[str, Any]]:
        if current_evaluation is None:
            return []
        return list(bindings.replay_evidence_builder(_records(current_evaluation.matches)))

    def prompt_factory(**values: Any) -> str:
        if values.get("bootstrap"):
            bootstrap_packet = write_bootstrap_input_packet(
                output_path=run_dir / "context/bootstrap_input.json",
                context_manifest_path=bundle.manifest_path,
                context_files=bundle.files,
                candidate_source_path=(
                    workspace / prompt_profile.candidate_source_relative
                ),
            )
            return context.build_bootstrap_prompt(
                act_id=values["act_id"],
                workspace=workspace,
                experience_path=experience.path,
                bootstrap_input_path=bootstrap_packet,
            )
        phase = values.get("phase", "candidate")
        all_evidence = evidence()
        branch_evidence = all_evidence
        if phase == "candidate" and values.get("branch_index") is not None:
            branch_evidence = _route_evidence(
                all_evidence, int(values["branch_index"])
            )
        proposal_root = run_dir / "proposals" / values["iteration_id"]
        source = workspace / prompt_profile.candidate_source_relative
        if phase == "planner":
            parent_occupancy = None
            if bindings.occupancy_summarizer is not None:
                if current_evaluation is None:
                    raise ValueError(
                        "parent occupancy requires a complete parent evaluation"
                    )
                parent_version = store.get(str(values["parent_version_id"]))
                parent_occupancy = bindings.occupancy_summarizer(
                    store.objects / parent_version.content_hash,
                    references=_references(current_evaluation),
                )
            planner_input = proposal_root / "planner_input.json"
            planner_output = workspace / ".agentbench/branch_briefs.json"
            planner_check_output = workspace / ".agentbench/planner_check_result.json"
            packet = write_planner_input_packet(
                output_path=planner_input,
                iteration_id=values["iteration_id"],
                parent_version_id=values["parent_version_id"],
                game_digest_path=digest,
                context_manifest_path=bundle.manifest_path,
                research_state_path=research,
                replay_evidence=all_evidence,
                previous_measurements=previous_measurements,
                active_target=config.run.evaluation.learning_opponent,
                candidate_source_path=source,
                parent_occupancy=parent_occupancy,
                planner_command=(
                    sys.executable,
                    "-m",
                    "agentbench_frame.hl.planner_check",
                    "--briefs",
                    str(planner_output.resolve()),
                    "--planner-input",
                    str(planner_input.resolve()),
                    "--output",
                    str(planner_check_output.resolve()),
                    "--expected-count",
                    str(config.run.iteration.candidates_per_cycle),
                    "--entry-symbol",
                    prompt_profile.policy_entry_symbol,
                ),
            )
            return context.build_planner_prompt(
                act_id=values["act_id"],
                iteration_id=values["iteration_id"],
                parent_version_id=values["parent_version_id"],
                workspace=workspace,
                game_digest_path=digest,
                research_state_path=research,
                replay_evidence=all_evidence,
                previous_measurements=previous_measurements,
                active_target=config.run.evaluation.learning_opponent,
                scope_contract_required=config.run.iteration.scope_contract_required,
                planner_input_path=packet,
            )
        if phase == "candidate":
            smoke_result = workspace / ".agentbench/candidate_smoke_result.json"
            packet = write_candidate_input_packet(
                output_path=(
                    proposal_root
                    / f"candidate_input-b{int(values['branch_index']):02d}.json"
                ),
                iteration_id=values["iteration_id"],
                branch_brief=values["branch_brief"],
                game_digest_path=digest,
                research_state_path=research,
                experience_path=experience.path,
                replay_evidence=branch_evidence,
                previous_measurements=previous_measurements,
                active_target=config.run.evaluation.learning_opponent,
                locked_opponents=(),
                candidate_source_path=source,
                policy_entry_symbol=prompt_profile.policy_entry_symbol,
                smoke_command=(
                    sys.executable,
                    str(bundle.files["smoke_fixture"]),
                    "--workspace",
                    str(workspace),
                    "--output",
                    str(smoke_result),
                ),
            )
            return context.build_candidate_prompt(
                act_id=values["act_id"],
                branch_index=values["branch_index"],
                branch_count=values["branch_count"],
                parent_version_id=values["parent_version_id"],
                workspace=workspace,
                game_digest_path=digest,
                research_state_path=research,
                replay_evidence=branch_evidence,
                previous_measurements=previous_measurements,
                experience_path=experience.path,
                branch_brief=values["branch_brief"],
                active_target=config.run.evaluation.learning_opponent,
                scope_contract_required=config.run.iteration.scope_contract_required,
                candidate_input_path=packet,
            )
        if phase == "repair":
            from agentbench_frame.hl.repair import (
                enrich_activation_repair_packet,
            )

            smoke_result = workspace / ".agentbench/candidate_smoke_result.json"
            activation_command = None
            if bindings.activation_contract_builder is not None:
                if current_evaluation is None:
                    raise ValueError(
                        "activation contract requires a complete parent evaluation"
                    )
                repair_value = json.loads(
                    Path(values["repair_input"]).read_text(encoding="utf-8")
                )
                parent_value = repair_value.get("parent")
                if not isinstance(parent_value, Mapping):
                    raise ValueError("repair packet requires parent metadata")
                parent_version_id = str(parent_value.get("version_id") or "")
                parent_version = store.get(parent_version_id)
                activation_parent = _materialize_activation_parent(
                    source=store.objects / parent_version.content_hash,
                    run_dir=run_dir,
                    content_hash=parent_version.content_hash,
                )
                branch_index = int(values["branch_index"])
                activation_command = bindings.activation_contract_builder(
                    parent_root=activation_parent,
                    candidate_root=workspace,
                    references=_references(current_evaluation),
                    references_path=(
                        proposal_root
                        / f"activation_references-b{branch_index:02d}.json"
                    ),
                    output_path=(
                        workspace / ".agentbench/activation_check_result.json"
                    ),
                    epsilon=config.run.measurement.epsilon,
                    minimum_changed_actions=(
                        config.run.iteration.activation_min_changed_actions
                    ),
                )
            repair_input = enrich_activation_repair_packet(
                values["repair_input"],
                game_digest_path=digest,
                research_state_path=research,
                experience_path=experience.path,
                candidate_source_path=source,
                policy_entry_symbol=prompt_profile.policy_entry_symbol,
                smoke_command=(
                    sys.executable,
                    str(bundle.files["smoke_fixture"]),
                    "--workspace",
                    str(workspace),
                    "--output",
                    str(smoke_result),
                ),
                activation_command=activation_command,
            )
            return context.build_repair_prompt(
                act_id=values["act_id"],
                iteration_id=values["iteration_id"],
                branch_index=values["branch_index"],
                workspace=workspace,
                game_digest_path=digest,
                research_state_path=research,
                repair_input_path=repair_input,
                experience_path=experience.path,
            )
        if phase == "reducer":
            reducer_path = Path(values["reducer_input"])
            reducer = json.loads(reducer_path.read_text(encoding="utf-8"))
            return context.build_reducer_prompt(
                act_id=values["act_id"],
                iteration_id=values["iteration_id"],
                selected_version_id=str(reducer["selected_version_id"]),
                workspace=workspace,
                game_digest_path=digest,
                research_state_path=research,
                reducer_input_path=reducer_path,
            )
        raise ValueError(f"unsupported profile prompt phase: {phase}")

    def activation_probe(**values: Any) -> Mapping[str, Any]:
        parent = values["parent_evaluation"]
        comparison = bindings.behavior_comparator(
            store.objects / values["old_version"].content_hash,
            store.objects / values["new_version"].content_hash,
            references=_references(parent),
            epsilon=config.run.measurement.epsilon,
        )
        return _behavior_comparison_payload(comparison)

    controller = HLController(
        workspace=workspace,
        run_root=run_dir,
        provider=provider,
        evaluator=bindings.evaluator,
        version_store=store,
        lineage=lineage,
        events=writer,
        iteration=config.run.iteration,
        rollback=config.run.rollback,
        prompt_factory=prompt_factory,
        experience_manager=experience,
        research_state_path=research,
        research_state_max_bytes=config.run.context.research_state_max_bytes,
        summary_resolver=lambda match: next(
            iter(bindings.replay_evidence_builder(_records((match,)))),
            {},
        ),
        activation_probe=activation_probe,
        candidate_smoke_verifier=lambda **values: _smoke_dict(
            bindings, Path(values["workspace"])
        ),
        candidate_source_relative=prompt_profile.candidate_source_relative,
        policy_entry_symbol=prompt_profile.policy_entry_symbol,
    )

    if resume:
        controller.resume(historical)
        head = controller.lineage.lineage_head_version_id
        if head is None or head not in evaluations:
            raise ValueError("resumed profile run lacks a complete head evaluation")
        current_evaluation = evaluations[head]
        origin_id = next(iter(evaluations))
        origin_evaluation = evaluations[origin_id]
        origin_version = store.get(origin_id)
    elif config.run.origin.mode == "model_bootstrap":
        if provider is None:
            raise ValueError("model bootstrap requires a provider credential")
        bootstrap = controller.bootstrap()
        current_evaluation = bootstrap.evaluation
        evaluations[bootstrap.version.version_id] = bootstrap.evaluation
        origin_version = bootstrap.version
        origin_evaluation = bootstrap.evaluation
    else:
        assert config.run.origin.source_run is not None
        assert config.run.origin.source_version is not None
        origin_version = controller.initialize_imported(
            source_run=config.run.origin.source_run,
            source_version_id=config.run.origin.source_version,
            evaluate=True,
        )
        current_evaluation = bindings.evaluator.last_evaluation
        if current_evaluation is None:
            raise RuntimeError("imported origin evaluation was not recorded")
        evaluations[origin_version.version_id] = current_evaluation
        origin_evaluation = current_evaluation

    if current_evaluation.status != "complete":
        raise RuntimeError(
            "profile origin evaluation failed: " + str(current_evaluation.error)
        )
    origin_references = _references(origin_evaluation)
    previous_measurements.update(
        live_win_rate=current_evaluation.score,
        iteration=0,
        stagnation_count=0,
    )

    if resume:
        progress = _resume_progress(
            historical,
            origin_version_id=origin_version.version_id,
        )
        best_version = store.get(progress.best_version_id)
        best_learning_score = progress.best_learning_score
        best_passing = progress.best_passing
        stagnation = progress.stagnation
        completed_cycles = progress.completed_cycles
        certified = progress.certified
    else:
        best_version = origin_version
        best_learning_score = float(current_evaluation.score or 0.0)
        best_passing = 0
        stagnation = 0
        completed_cycles = 0
        certified = False
    cycles_this_invocation = 0
    origin_panel_exists = any(
        event.get("event_type") == "reporting_panel_completed"
        and event.get("iteration_id") == "iter-000000"
        for event in historical
    )
    needs_origin_certification = (
        not resume and config.run.origin.mode == "imported_version"
    )
    imported_certification = None
    if needs_origin_certification:
        assert config.run.origin.source_run is not None
        assert config.run.origin.source_version is not None
        imported_certification = _load_imported_certification(
            source_run=config.run.origin.source_run,
            source_version_id=config.run.origin.source_version,
            origin_version=origin_version,
            game=config.run.game,
            evaluation_config=config.run.evaluation,
        )

    def write_certification_reuse(
        cache: Mapping[str, Any],
        *,
        iteration_id: str,
        version: Version,
        reason: str,
    ) -> None:
        writer.write(
            "certification_reused",
            iteration_id=iteration_id,
            version_id=version.version_id,
            content_hash=version.content_hash,
            source_run_id=str(cache["source_run_id"]),
            source_version_id=str(cache["source_version_id"]),
            source_event_id=str(cache["source_event_id"]),
            reason=reason,
        )

    if needs_origin_certification:
        if imported_certification is None:
            certification = bindings.evaluator.certify(origin_version)
        else:
            certification = imported_certification["evaluation"]
            write_certification_reuse(
                imported_certification,
                iteration_id="iter-000000",
                version=origin_version,
                reason="exact_imported_content_and_evaluation_config",
            )
        controller.record_matches(
            version=origin_version,
            act_id=origin_version.act_id,
            phase="certification",
            matches=certification.matches,
        )
        best_passing = _passing_opponents(
            certification.matches,
            required_win_rate=config.run.evaluation.required_win_rate,
        )
        writer.write(
            "certification_completed",
            version_id=origin_version.version_id,
            act_id=origin_version.act_id,
            status=certification.status,
            score=certification.score,
            passing_human_opponents=best_passing,
            required_human_opponents=config.run.evaluation.required_human_opponents,
            matches=list(certification.matches),
        )
        certified = (
            certification.status == "complete"
            and best_passing >= config.run.evaluation.required_human_opponents
        )
        if certified:
            writer.write(
                "run_completed",
                reason="human_pool_target_reached",
                version_id=origin_version.version_id,
                passing_human_opponents=best_passing,
            )
    if (
        config.run.evaluation.reporting_panel_every_cycle
        and not origin_panel_exists
    ):
        reporting = (
            certification
            if needs_origin_certification
            else _reporting_evaluation(
                bindings.evaluator,
                origin_version,
                seed_count=(
                    config.run.evaluation.reporting_seeds_per_opponent
                ),
            )
        )
        controller.record_matches(
            version=origin_version,
            act_id=origin_version.act_id,
            phase="reporting",
            matches=reporting.matches,
        )
        writer.write(
            "reporting_panel_completed",
            **_reporting_panel_payload(
                reporting,
                iteration_id="iter-000000",
                proposal_cycle=0,
                version_id=origin_version.version_id,
            ),
        )
    pending_recoveries = (
        _pending_cycle_recoveries(
            historical,
            provider=provider,
            workspace=workspace,
            version_store=store,
            evaluations_by_version=evaluations,
            expected_candidate_count=config.run.iteration.candidates_per_cycle,
            policy_entry_symbol=prompt_profile.policy_entry_symbol,
        )
        if resume and not replan_pending
        else {
            "planner_recovery": None,
            "candidate_recoveries": {},
            "repair_recoveries": {},
        }
    )
    while not certified and (acts is None or cycles_this_invocation < acts):
        if controller.reached_iteration_limit():
            break
        parent_id = controller.lineage.lineage_head_version_id
        if parent_id is None:
            raise RuntimeError("profile loop has no search parent")
        parent_evaluation = evaluations[parent_id]
        cycle = controller.run_proposal_cycle(
            parent_version_id=parent_id,
            parent_evaluation=parent_evaluation,
            **pending_recoveries,
        )
        pending_recoveries = {
            "planner_recovery": None,
            "candidate_recoveries": {},
            "repair_recoveries": {},
        }
        completed_cycles += 1
        cycles_this_invocation += 1
        for candidate in cycle.representatives:
            evaluations[candidate.version.version_id] = candidate.evaluation
        selected_id = cycle.search_parent_version_id
        current_evaluation = evaluations.get(selected_id, parent_evaluation)
        selected_version = store.get(selected_id)
        if current_evaluation.status != "complete":
            stagnation += 1
            continue

        origin_comparison = bindings.behavior_comparator(
            store.objects / origin_version.content_hash,
            store.objects / selected_version.content_hash,
            references=origin_references,
            epsilon=config.run.measurement.epsilon,
        )
        details = dict(origin_comparison.details)
        writer.write(
            "behavior_measured",
            iteration=completed_cycles,
            version_id=selected_id,
            reference_version_id=origin_version.version_id,
            comparison="origin_to_candidate",
            epsilon=config.run.measurement.epsilon,
            decision_count=origin_comparison.decision_count,
            changed_action_count=origin_comparison.changed_action_count,
            mean_kl_nats_per_decision=float(
                details["mean_kl_nats_per_decision"]
            ),
            role_mean_kl=dict(details.get("role_mean_kl", {})),
            reference_replays=[str(path) for path, _ in origin_references],
        )

        improved = float(current_evaluation.score or 0.0) > best_learning_score
        if improved:
            best_learning_score = float(current_evaluation.score or 0.0)
            best_version = selected_version
            stagnation = 0
        else:
            stagnation += 1

        should_certify = (
            config.run.evaluation.full_pool_every_iteration
            or improved
            or completed_cycles == 1
        )
        certification = None
        if should_certify:
            if (
                imported_certification is not None
                and selected_version.content_hash
                == imported_certification["content_hash"]
            ):
                certification = imported_certification["evaluation"]
                write_certification_reuse(
                    imported_certification,
                    iteration_id=cycle.iteration_id,
                    version=selected_version,
                    reason="selected_content_matches_imported_certification",
                )
            else:
                certification = bindings.evaluator.certify(selected_version)
            controller.record_matches(
                version=selected_version,
                act_id=selected_version.act_id,
                phase="certification",
                matches=certification.matches,
            )
            passing = _passing_opponents(
                certification.matches,
                required_win_rate=config.run.evaluation.required_win_rate,
            )
            writer.write(
                "certification_completed",
                version_id=selected_id,
                act_id=selected_version.act_id,
                status=certification.status,
                score=certification.score,
                passing_human_opponents=passing,
                required_human_opponents=config.run.evaluation.required_human_opponents,
                matches=list(certification.matches),
            )
            if passing > best_passing:
                best_passing = passing
                best_version = selected_version
                stagnation = 0
            certified = (
                certification.status == "complete"
                and passing >= config.run.evaluation.required_human_opponents
            )
            if certified:
                controller.lineage.promote_champion(selected_id)
                writer.write(
                    "run_completed",
                    reason="human_pool_target_reached",
                    version_id=selected_id,
                    passing_human_opponents=passing,
                )
        if config.run.evaluation.reporting_panel_every_cycle:
            reporting = (
                certification
                if certification is not None
                else _reporting_evaluation(
                    bindings.evaluator,
                    selected_version,
                    seed_count=(
                        config.run.evaluation.reporting_seeds_per_opponent
                    ),
                )
            )
            if certification is None:
                controller.record_matches(
                    version=selected_version,
                    act_id=selected_version.act_id,
                    phase="reporting",
                    matches=reporting.matches,
                )
            writer.write(
                "reporting_panel_completed",
                **_reporting_panel_payload(
                    reporting,
                    iteration_id=cycle.iteration_id,
                    proposal_cycle=completed_cycles,
                    version_id=selected_id,
                ),
            )

        if (
            not certified
            and config.run.rollback.enabled
            and stagnation >= config.run.rollback.patience
            and best_version.version_id != selected_id
        ):
            decision = controller.lineage.force_parent(
                best_version.version_id,
                reason="stagnation_to_best_archive",
            )
            store.checkout(best_version.version_id)
            current_evaluation = evaluations[best_version.version_id]
            writer.write(
                "rollback_selected",
                from_version_id=decision.from_version_id,
                to_version_id=decision.to_version_id,
                reason=decision.reason,
            )
            stagnation = 0

        previous_measurements.update(
            live_win_rate=current_evaluation.score,
            iteration=completed_cycles,
            best_learning_score=best_learning_score,
            passing_human_opponents=best_passing,
            stagnation_count=stagnation,
            mean_kl_nats_per_decision=float(
                details["mean_kl_nats_per_decision"]
            ),
        )

    return {
        **validation,
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "certified": certified,
        "completed_cycles": completed_cycles,
        "best_version_id": best_version.version_id,
        "best_learning_score": best_learning_score,
        "passing_human_opponents": best_passing,
        **controller.summary(),
    }
