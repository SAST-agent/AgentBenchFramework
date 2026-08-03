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
from agentbench_frame.hl.game_profile import HLGameBindings, get_game_profile
from agentbench_frame.hl.lineage import LineageManager
from agentbench_frame.hl.local_config import LocalHLConfig
from agentbench_frame.hl.match_record import MatchRecord
from agentbench_frame.hl.proposal import (
    write_candidate_input_packet,
    write_planner_input_packet,
)
from agentbench_frame.hl.provider import CodexSessionProvider
from agentbench_frame.hl.research_state import ResearchState


def _json_native(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


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
            error=None if status == "complete" else "historical incomplete evaluation",
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


def run_profile(
    config: LocalHLConfig,
    *,
    run_dir: Path,
    workspace: Path,
    acts: int | None,
    resume: bool,
    provider_environment: Mapping[str, str],
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
    if resume:
        if json.loads(snapshot.read_text(encoding="utf-8")) != frozen:
            raise ValueError("resume config differs from frozen run config")
    else:
        snapshot.write_text(
            json.dumps(frozen, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    profile = get_game_profile(config.run.game)
    prompt_profile = bindings.prompt_profile
    provider = CodexSessionProvider(
        config.run.provider,
        run_root=run_dir,
        environ=dict(provider_environment),
        timeout_s=config.run.provider.timeout_seconds,
        idle_timeout_s=config.run.provider.idle_timeout_seconds,
    )
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
            return context.build_bootstrap_prompt(
                act_id=values["act_id"],
                workspace=workspace,
                experience_path=experience.path,
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
            packet = write_planner_input_packet(
                output_path=proposal_root / "planner_input.json",
                iteration_id=values["iteration_id"],
                parent_version_id=values["parent_version_id"],
                game_digest_path=digest,
                context_manifest_path=bundle.manifest_path,
                research_state_path=research,
                replay_evidence=all_evidence,
                previous_measurements=previous_measurements,
                active_target=config.run.evaluation.learning_opponent,
                candidate_source_path=source,
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
            return context.build_repair_prompt(
                act_id=values["act_id"],
                iteration_id=values["iteration_id"],
                branch_index=values["branch_index"],
                workspace=workspace,
                game_digest_path=digest,
                research_state_path=research,
                repair_input_path=values["repair_input"],
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
        return dataclasses.asdict(comparison)

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

    best_version = origin_version
    best_learning_score = float(current_evaluation.score or 0.0)
    best_passing = 0
    stagnation = 0
    completed_cycles = 0
    certified = False
    while not certified and (acts is None or completed_cycles < acts):
        if controller.reached_iteration_limit():
            break
        parent_id = controller.lineage.lineage_head_version_id
        if parent_id is None:
            raise RuntimeError("profile loop has no search parent")
        parent_evaluation = evaluations[parent_id]
        cycle = controller.run_proposal_cycle(
            parent_version_id=parent_id,
            parent_evaluation=parent_evaluation,
        )
        completed_cycles += 1
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
        if should_certify:
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
