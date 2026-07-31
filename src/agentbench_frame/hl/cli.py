"""Local commands for reproducible HL runs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
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
from agentbench_frame.hl.experience import ExperienceManager
from agentbench_frame.hl.local_config import LocalHLConfig


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _load(path: str) -> LocalHLConfig:
    return LocalHLConfig.load(path)


def _validate(config: LocalHLConfig) -> dict[str, Any]:
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
    return {
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
    }


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
                "replay-skill/rollman-replay/SKILL.md"
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
    provider_environment = _provider_environment(config)
    if provider_environment is None:
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
    provider_environment = _provider_environment(config)
    if provider_environment is None:
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
    provider_environment: dict[str, str],
) -> int:
    # Imported lazily so validate and dry-run never initialize model/runtime state.
    from agentbench_frame.games.rollman.candidate_runner import __file__ as candidate_runner
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
    from agentbench_frame.hl.codebase import VersionStore
    from agentbench_frame.hl.context import IterationContext
    from agentbench_frame.hl.controller import HLController
    from agentbench_frame.hl.events import HLEventWriter, read_events
    from agentbench_frame.hl.lineage import LineageManager
    from agentbench_frame.hl.provider import CodexSessionProvider
    from agentbench_frame.hl.evaluator import CandidateEvaluation

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
        if persisted != config_snapshot:
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
    rank1_raw = next(
        opponent
        for opponent in pool
        if opponent.opponent_id == config.run.evaluation.learning_opponent
    )
    rank1 = prepare_opponent(
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
    candidate_factory = lambda version: ProcessSpec(
        argv=(
            sys.executable,
            str(Path(candidate_runner).resolve()),
            "--workspace",
            str(workspace),
            "--sdk-root",
            str(config.paths.pacman_sdk_root),
        ),
        cwd=workspace,
        untrusted=True,
        read_roots=(
            workspace,
            config.paths.pacman_sdk_root,
            Path(candidate_runner).resolve().parents[3],
        ),
        denied_paths=(config.source_path.parents[2] / ".env",),
    )
    evaluator = RollmanEvaluator(
        logic=logic,
        candidate_factory=candidate_factory,
        learning_opponent=rank1,
        human_pool=(),
        fixed_gate_seeds=config.run.evaluation.fixed_gate_seeds,
        certification_seeds=config.run.evaluation.certification_seeds,
        artifact_root=run_dir / "matches",
        match_runner=run_match,
        state_tracker_factory=lambda: FrozenStateTracker(logic_root),
    )
    bundle = ContextBundle.create(
        run_dir / "context",
        {
            "rules": asset_path("rules.md"),
            "decision_space": asset_path("decision_space.yaml"),
            "replay_skill": asset_path(
                "replay-skill/rollman-replay/SKILL.md"
            ),
        },
    )
    iteration_context = IterationContext(bundle)
    experience = ExperienceManager(
        run_dir / "experience",
        compress_every_acts=config.run.experience.compress_every_acts,
    )
    provider = CodexSessionProvider(
        config.run.provider,
        run_root=run_dir,
        environ=provider_environment,
    )
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

    def prompt_factory(**values):
        if values.get("bootstrap"):
            return iteration_context.build_bootstrap_prompt(
                act_id=values["act_id"],
                workspace=workspace,
                experience_path=experience.path,
            )
        evaluation = evaluator.last_evaluation
        matches = list(evaluation.matches) if evaluation is not None else []
        evidence = [
            {
                "opponent": match.get("opponent"),
                "seed": match.get("seed"),
                "result": match.get("result"),
                "replay": match.get("replay"),
                "trace": match.get("trace"),
            }
            for match in matches
        ]
        return iteration_context.build_prompt(
            act_id=values["act_id"],
            branch_index=values["branch_index"],
            branch_count=values["branch_count"],
            parent_version_id=values["parent_version_id"],
            workspace=workspace,
            replay_evidence=evidence,
            previous_measurements={
                "benchmark_score": (
                    None if evaluation is None else evaluation.score
                ),
                "evaluation_status": (
                    None if evaluation is None else evaluation.status
                ),
            },
            experience_path=experience.path,
        )

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
    )
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

    def certify_eligible_champion() -> bool:
        nonlocal certification_pool_prepared
        champion_id = controller.lineage.champion_version_id
        if champion_id is None or champion_id in completed_certifications:
            return False
        evaluation = evaluations_by_version.get(champion_id)
        if (
            evaluation is None
            or evaluation.status != "complete"
            or evaluation.score is None
            or evaluation.score < config.run.evaluation.required_win_rate
        ):
            return False
        if not certification_pool_prepared:
            evaluator.human_pool = prepare_human_pool(
                pool,
                build_root=config.paths.opponent_build_root,
            )
            certification_pool_prepared = True
        champion = version_store.get(champion_id)
        certification = evaluator.certify(champion)
        controller.record_matches(
            version=champion,
            act_id=champion.act_id,
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
            version_id=champion.version_id,
            act_id=champion.act_id,
            status=certification.status,
            score=certification.score,
            passing_human_opponents=passing,
            required_human_opponents=(
                config.run.evaluation.required_human_opponents
            ),
            matches=list(certification.matches),
        )
        if certification.status == "complete":
            completed_certifications.add(champion.version_id)
        if (
            certification.status == "complete"
            and passing >= config.run.evaluation.required_human_opponents
        ):
            writer.write(
                "run_completed",
                reason="human_pool_target_reached",
                version_id=champion.version_id,
                passing_human_opponents=passing,
            )
            return True
        return False

    if not certified:
        certified = certify_eligible_champion()
    completed_here = 0 if resume else 1
    while not certified and (acts is None or completed_here < acts):
        if controller.reached_iteration_limit():
            break
        iteration_result = controller.run_act()
        completed_here += 1
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
        certified = certify_eligible_champion()
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
    if getattr(args, "acts", None) is not None and args.acts < 1:
        parser.error("--acts must be >= 1")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
