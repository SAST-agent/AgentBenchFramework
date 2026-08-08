"""CLI surface for the official-engine Generals HL pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib

from agentbench_frame.tracking.provider import ProviderInvocation
from agentbench_frame.tracking.providers import ClaudeCodeProvider, CodexProvider

from .assets import (
    AssetValidationError,
    _stable_tree_hash,
    load_calibration_config,
    load_expanded_kl_config,
    resolve_calibration_candidate_source,
    load_pilot_config,
    require_valid_assets,
    resolve_replay_skill,
    resolve_assets,
)
from .attribution_pipeline import GeneralsAttributionPipeline
from .challenge_v9 import load_round9_challenge_config
from .historical_policy import HistoricalPolicySource
from .leaderboard_qualification import (
    QUALIFICATION_RESULT_SCHEMA,
    evaluate_leaderboard_qualification,
    load_leaderboard_qualification_config,
    load_qualification_receipt,
)
from .leaderboard import (
    build_leaderboard,
    qualification_result_from_dict,
)
from .lineage_v9 import load_round9_lineage
from .pipeline import GeneralsHLPipeline
from .pipeline_v2 import (
    GeneralsHLRound2Pipeline,
    calibrate_development,
)
from .pipeline_v3 import GeneralsHLRound3Pipeline
from .pipeline_v4 import GeneralsHLRound4Pipeline
from .pipeline_v5 import GeneralsHLRound5Pipeline
from .pipeline_v6 import GeneralsHLRound6Pipeline
from .pipeline_v7 import GeneralsHLRound7Pipeline
from .pipeline_v8 import GeneralsHLRound8Pipeline
from .pipeline_v9 import GeneralsHLRound9Pipeline
from .pipeline_campaign import ChampionCampaignPipeline
from .policy_kl_expanded import GeneralsExpandedPolicyKLPipeline
from .policy_kl_extension import GeneralsPolicyKLExtensionPipeline
from .policy_kl_v8_extension import GeneralsPolicyKLV8ExtensionPipeline
from .policy_kl_pipeline import GeneralsPolicyKLPipeline
from .paper_figure_v9 import load_v9_figure_data, render_v9_paper_figures


def register_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generals", help="Official-engine Generals HL benchmark"
    )
    commands = parser.add_subparsers(dest="generals_command", required=True)
    for name, help_text in (
        ("prepare", "Validate frozen assets and population"),
        ("eval", "Evaluate the frozen v0 matrix"),
        ("iterate", "Run v0 → learning → one Codex act → v1"),
        (
            "calibrate-dev",
            "Select and freeze a weak opponent on development seeds",
        ),
        (
            "iterate-v2",
            "Run held-out calibration → v1 learning → one Codex act → v2",
        ),
        (
            "recover-v2",
            "Retry a pre-thread provider failure without reopening held-out",
        ),
        (
            "iterate-v3",
            "Run strongest-human v2 learning → one gated Codex act → v3",
        ),
        (
            "iterate-v4",
            "Run replay-guided v3 learning → one non-gated Codex act → v4",
        ),
        (
            "iterate-v5",
            "Run paired v3/v4 learning → one rollback-guided Codex act → v5",
        ),
        (
            "iterate-v6",
            "Run strongest-human v5 learning → one macro-planner act → v6",
        ),
        (
            "recover-v6",
            "Recover an audited v6 prompt, provider, or evaluation failure",
        ),
        (
            "iterate-v7",
            "Run v6 champion learning → one planner act → v7 challenge",
        ),
        (
            "recover-v7",
            "Recover an audited v7 provider or evaluation failure",
        ),
        (
            "iterate-v8",
            "Run one clean-room v7 learning act and evaluate v8",
        ),
        (
            "recover-v8",
            "Recover a frozen clean-room v8 candidate without another act",
        ),
        (
            "measure-policy-kl",
            "Measure exact controlled-reference v0-v6 policy KL",
        ),
        (
            "recover-policy-kl",
            "Resume an incomplete exact controlled-reference measurement",
        ),
        (
            "extend-policy-kl-v7",
            "Reuse verified v1 policy KL inputs and probe only v7",
        ),
        (
            "recover-policy-kl-v7",
            "Resume an incomplete verified v7 policy KL extension",
        ),
        (
            "extend-policy-kl-v8",
            "Reuse verified v0-v7 policy KL inputs and probe only v8",
        ),
        (
            "recover-policy-kl-v8",
            "Resume an incomplete verified v8 policy KL extension",
        ),
        (
            "attribute-v8-regression",
            "Run paired A/B/C/D scientific attribution of the v8 regression",
        ),
        (
            "iterate-v9",
            "Run one attribution-guided v7-to-v9 Codex act and evaluate it",
        ),
        (
            "recover-v9",
            "Evaluate a frozen runnable v9 candidate without another provider act",
        ),
        (
            "measure-policy-kl-expanded",
            "Measure separate legacy-12 and expanded-24 exact policy KL",
        ),
        (
            "recover-policy-kl-expanded",
            "Resume incomplete expanded exact policy KL in the same run",
        ),
        (
            "plot-v9-paper",
            "Render deterministic English v9 publication figures",
        ),
        (
            "check-leaderboard-qualification",
            "Verify the capability gate before leaderboard publication",
        ),
        (
            "build-leaderboard",
            "Aggregate verified provider results at every budget checkpoint",
        ),
        (
            "champion-campaign-act",
            "Advance one sealed, resumable champion-campaign act",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        if name != "plot-v9-paper":
            command.add_argument("--agentbench-root", type=Path, required=True)
            command.add_argument("--manifest", type=Path, required=True)
        if name not in {
            "prepare",
            "plot-v9-paper",
            "check-leaderboard-qualification",
            "build-leaderboard",
        }:
            command.add_argument("--data-dir", type=Path, required=True)
        if name == "eval":
            command.add_argument("--version", choices=("v0",), default="v0")
        if name in {
            "iterate",
            "iterate-v2",
            "recover-v2",
            "iterate-v3",
            "iterate-v4",
            "iterate-v5",
            "iterate-v6",
            "recover-v6",
            "iterate-v7",
            "recover-v7",
            "iterate-v8",
            "recover-v8",
        }:
            explicit_provider = name in {
                "iterate-v6",
                "recover-v6",
                "iterate-v7",
                "recover-v7",
                "iterate-v8",
                "recover-v8",
            }
            command.add_argument(
                "--codex-executable",
                default="codex",
                required=explicit_provider,
            )
            command.add_argument(
                "--provider-timeout",
                type=float,
                default=1800,
                required=explicit_provider,
            )
        if name in {"calibrate-dev", "iterate-v2", "recover-v2"}:
            command.add_argument(
                "--calibration-manifest", type=Path, required=True
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name in {"iterate-v3", "iterate-v4", "iterate-v5"}:
            command.add_argument(
                "--learning-manifest",
                type=Path,
                required=True,
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name in {"iterate-v6", "recover-v6"}:
            command.add_argument(
                "--learning-manifest",
                type=Path,
                required=True,
            )
            command.add_argument(
                "--replay-skill",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name in {
            "iterate-v7", "recover-v7", "iterate-v8", "recover-v8"
        }:
            command.add_argument(
                "--challenge-manifest",
                type=Path,
                required=True,
            )
            command.add_argument(
                "--replay-skill",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
        if name in {"iterate-v4", "iterate-v5"}:
            command.add_argument(
                "--replay-skill",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument(
                "--prior-attempt-run",
                type=Path,
                help=(
                    "Optional finalized prompt_incomplete v4 run whose "
                    "pre-act learning budget must remain cumulative"
                ),
            )
        if name == "iterate-v5":
            command.add_argument(
                "--expected-rollback-hash",
                required=True,
            )
            command.add_argument(
                "--campaign-budget-receipt",
                type=Path,
                required=True,
            )
        if name == "calibrate-dev":
            command.add_argument("--selection-output", type=Path, required=True)
        if name in {"iterate-v2", "recover-v2"}:
            command.add_argument(
                "--calibration-selection", type=Path, required=True
            )
        if name == "recover-v2":
            command.add_argument("--failed-run", type=Path, required=True)
        if name == "recover-v6":
            command.add_argument("--failed-run", type=Path, required=True)
        if name == "recover-v7":
            command.add_argument("--failed-run", type=Path, required=True)
        if name == "recover-v8":
            command.add_argument("--failed-run", type=Path, required=True)
        if name in {"measure-policy-kl", "recover-policy-kl"}:
            command.add_argument(
                "--reference-manifest",
                type=Path,
                required=True,
            )
            command.add_argument(
                "--count-wall-time",
                type=float,
                default=3600,
            )
            command.add_argument(
                "--count-max-states",
                type=int,
                default=5_000_000,
            )
        if name == "recover-policy-kl":
            command.add_argument("--failed-run", type=Path, required=True)
        if name in {"extend-policy-kl-v7", "recover-policy-kl-v7"}:
            command.add_argument(
                "--reference-manifest",
                type=Path,
                required=True,
            )
            command.add_argument(
                "--source-run",
                type=Path,
                required=True,
            )
        if name == "recover-policy-kl-v7":
            command.add_argument("--failed-run", type=Path, required=True)
        if name in {"extend-policy-kl-v8", "recover-policy-kl-v8"}:
            command.add_argument(
                "--reference-manifest", type=Path, required=True
            )
            command.add_argument("--source-run", type=Path, required=True)
            command.add_argument("--target-run", type=Path, required=True)
            command.add_argument("--expected-target-hash", required=True)
        if name == "recover-policy-kl-v8":
            command.add_argument("--failed-run", type=Path, required=True)
        if name == "attribute-v8-regression":
            command.add_argument(
                "--attribution-manifest", type=Path, required=True
            )
            command.add_argument("--v7-run", type=Path, required=True)
            command.add_argument("--v8-run", type=Path, required=True)
        if name in {"iterate-v9", "recover-v9"}:
            command.add_argument(
                "--challenge-manifest", type=Path, required=True
            )
            command.add_argument("--replay-skill", type=Path, required=True)
            command.add_argument("--parent-run", type=Path, required=True)
            command.add_argument("--predecessor-run", type=Path, required=True)
            command.add_argument("--attribution-run", type=Path, required=True)
            command.add_argument("--expected-parent-hash", required=True)
            command.add_argument("--expected-predecessor-hash", required=True)
        if name == "iterate-v9":
            command.add_argument("--codex-executable", required=True)
            command.add_argument("--provider-timeout", type=float, required=True)
        if name == "recover-v9":
            command.add_argument("--failed-run", type=Path, required=True)
        if name in {
            "measure-policy-kl-expanded",
            "recover-policy-kl-expanded",
        }:
            command.add_argument(
                "--reference-manifest", type=Path, required=True
            )
            command.add_argument("--source-run", type=Path, required=True)
            command.add_argument("--target-run", type=Path, required=True)
            command.add_argument("--expected-target-hash", required=True)
            command.add_argument("--count-wall-time", type=float, default=3600)
            command.add_argument("--count-max-states", type=int, default=5_000_000)
        if name == "recover-policy-kl-expanded":
            command.add_argument("--failed-run", type=Path, required=True)
        if name == "plot-v9-paper":
            command.add_argument("--attribution-run", type=Path, required=True)
            command.add_argument("--v9-run", type=Path, required=True)
            command.add_argument("--legacy-kl-run", type=Path, required=True)
            command.add_argument("--expanded-kl-run", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
        if name == "check-leaderboard-qualification":
            command.add_argument(
                "--qualification-manifest", type=Path, required=True
            )
            command.add_argument(
                "--replay-skill",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument("--receipt", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        if name == "build-leaderboard":
            command.add_argument(
                "--qualification-manifest", type=Path, required=True
            )
            command.add_argument(
                "--qualification-result",
                type=Path,
                action="append",
                required=True,
            )
            command.add_argument("--output", type=Path, required=True)
        if name == "champion-campaign-act":
            command.add_argument(
                "--campaign-manifest", type=Path, required=True
            )
            command.add_argument(
                "--qualification-manifest", type=Path, required=True
            )
            command.add_argument(
                "--replay-skill",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument(
                "--decision-space",
                type=Path,
                required=True,
                help="Path relative to --agentbench-root",
            )
            command.add_argument("--campaign-root", type=Path, required=True)
            command.add_argument(
                "--replicate-id",
                choices=("replicate-1", "replicate-2", "replicate-3"),
                required=True,
            )
            command.add_argument("--model", required=True)
            command.add_argument("--model-revision", required=True)
            command.add_argument(
                "--provider", choices=("codex", "claude-code"), default="codex"
            )
            command.add_argument("--codex-executable", default="codex")
            command.add_argument("--claude-executable", default="claude")
            command.add_argument(
                "--claude-permission-mode", default="acceptEdits"
            )
            command.add_argument("--provider-timeout", type=float, required=True)
    return parser


def _assets(args):
    config = load_pilot_config(args.manifest)
    layout = resolve_assets(config, args.agentbench_root)
    require_valid_assets(layout)
    return config, layout


def _check_leaderboard_qualification(args, config, layout) -> int:
    output = args.output.resolve()
    if output.is_relative_to(args.agentbench_root.resolve()):
        raise AssetValidationError(
            "qualification output cannot modify the frozen AgentBench tree"
        )
    if output in {
        args.manifest.resolve(),
        args.qualification_manifest.resolve(),
        args.receipt.resolve(),
    }:
        raise AssetValidationError(
            "qualification output cannot overwrite an input artifact"
        )
    strongest = next(
        item
        for item in layout.opponents
        if item.opponent_id == config.opponents[0].opponent_id
    )
    replay_skill = resolve_replay_skill(
        args.agentbench_root, args.replay_skill
    )
    qualification = load_leaderboard_qualification_config(
        args.qualification_manifest,
        config,
        engine_sha256=layout.engine_hash,
        opponent_tree_sha256=_stable_tree_hash(strongest.source),
        replay_skill_sha256=replay_skill.sha256,
    )
    receipt = load_qualification_receipt(args.receipt)
    result = evaluate_leaderboard_qualification(qualification, receipt)
    payload = {
        "schema": QUALIFICATION_RESULT_SCHEMA,
        "manifest_sha256": hashlib.sha256(
            args.qualification_manifest.read_bytes()
        ).hexdigest(),
        "receipt_sha256": hashlib.sha256(
            args.receipt.read_bytes()
        ).hexdigest(),
        **result.to_dict(),
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AssetValidationError(
            f"cannot write qualification result: {exc}"
        ) from exc
    print(json.dumps(payload, sort_keys=True))
    if result.status == "invalid":
        return 2
    return 0 if result.qualified else 1


def _build_leaderboard(args, config, layout) -> int:
    del config, layout
    output = args.output.resolve()
    agentbench_root = args.agentbench_root.resolve()
    if output.is_relative_to(agentbench_root):
        raise AssetValidationError(
            "leaderboard output cannot modify the frozen AgentBench tree"
        )
    result_paths = [path.resolve() for path in args.qualification_result]
    if len(result_paths) != len(set(result_paths)):
        raise AssetValidationError(
            "leaderboard qualification results contain duplicate paths"
        )
    if output in {
        args.manifest.resolve(),
        args.qualification_manifest.resolve(),
        *result_paths,
    }:
        raise AssetValidationError(
            "leaderboard output cannot overwrite an input artifact"
        )

    qualification_manifest_sha256 = hashlib.sha256(
        args.qualification_manifest.read_bytes()
    ).hexdigest()
    results = []
    result_hashes = []
    for path in result_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetValidationError(
                f"cannot read qualification result {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema") != (
            QUALIFICATION_RESULT_SCHEMA
        ):
            raise AssetValidationError(
                f"qualification result schema invalid: {path}"
            )
        if payload.get("manifest_sha256") != qualification_manifest_sha256:
            raise AssetValidationError(
                f"qualification manifest hash changed for result: {path}"
            )
        try:
            result = qualification_result_from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise AssetValidationError(
                f"qualification result invalid: {path}: {exc}"
            ) from exc
        if result.status == "invalid":
            raise AssetValidationError(
                f"invalid qualification result cannot enter leaderboard: {path}"
            )
        results.append(result)
        result_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())

    try:
        leaderboard = build_leaderboard(results)
    except ValueError as exc:
        raise AssetValidationError(str(exc)) from exc
    output_payload = {
        **leaderboard,
        "qualification_manifest_sha256": qualification_manifest_sha256,
        "qualification_result_sha256": result_hashes,
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AssetValidationError(f"cannot write leaderboard: {exc}") from exc
    print(json.dumps(output_payload, sort_keys=True))
    return 0


class _RawOnlyProvider:
    provider_name = "codex"

    def invoke(self, context):
        return ProviderInvocation(
            status="failed",
            error="v0-only evaluation intentionally stops before a coding-agent act",
        )


class _RecoveryOnlyProvider:
    provider_name = "codex-recovery-disabled"

    def invoke(self, _context):
        raise RuntimeError("v9 recovery must not invoke a provider")


class _LazyAttributionEvaluator:
    def __init__(self, config, layout, data_dir):
        self.helper = GeneralsHLPipeline(
            config,
            layout,
            data_dir,
            _RawOnlyProvider(),
        )
        self.evaluator = None

    def evaluate(self, *args, **kwargs):
        run = kwargs.get("run")
        if run is None and len(args) >= 4:
            run = args[3]
        if self.evaluator is None:
            self.evaluator = self.helper._production_evaluator(
                Path(run.run_dir),
                capture_measurement_states=True,
            )
        return self.evaluator.evaluate(*args, **kwargs)


def _challenge_replay_hash(path: Path) -> str:
    try:
        value = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        digest = value["replay_skill_sha256"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise AssetValidationError(f"cannot read v9 attribution manifest: {exc}") from exc
    if not isinstance(digest, str) or len(digest) != 64:
        raise AssetValidationError("v9 replay skill digest is invalid")
    return digest


def _build_attribution_pipeline(args, config, layout):
    challenge = load_round9_challenge_config(
        args.attribution_manifest,
        config,
        engine_hash=layout.engine_hash,
        replay_skill_sha256=_challenge_replay_hash(args.attribution_manifest),
    )
    lineage = load_round9_lineage(args.v7_run, args.v8_run)
    v7 = HistoricalPolicySource(
        version="v7",
        run_id=lineage.policy_parent_run_id,
        content_hash=lineage.policy_parent_hash,
        source=lineage.v7_source,
        manifest=lineage.v7_manifest,
    )
    v8 = HistoricalPolicySource(
        version="v8",
        run_id=lineage.iteration_predecessor_run_id,
        content_hash=lineage.iteration_predecessor_hash,
        source=lineage.v8_source,
        manifest=lineage.v8_manifest,
    )
    sdk_root = next(item.source for item in layout.opponents if item.tier == "low")
    return GeneralsAttributionPipeline(
        config=config,
        challenge=challenge,
        v7=v7,
        v8=v8,
        data_dir=args.data_dir,
        evaluator=_LazyAttributionEvaluator(config, layout, args.data_dir),
        engine_root=layout.engine_root,
        sdk_root=sdk_root,
    )


def _build_v9_pipeline(args, config, layout, provider):
    replay_skill = resolve_replay_skill(args.agentbench_root, args.replay_skill)
    challenge = load_round9_challenge_config(
        args.challenge_manifest,
        config,
        engine_hash=layout.engine_hash,
        replay_skill_sha256=replay_skill.sha256,
    )
    if (
        args.expected_parent_hash != challenge.parent_content_hash
        or args.expected_predecessor_hash != challenge.predecessor_content_hash
    ):
        raise AssetValidationError("explicit v9 lineage hashes changed")
    return GeneralsHLRound9Pipeline(
        config=config,
        challenge=challenge,
        assets=layout,
        replay_skill=replay_skill,
        policy_parent_run_dir=args.parent_run,
        iteration_predecessor_run_dir=args.predecessor_run,
        attribution_run_dir=args.attribution_run,
        data_dir=args.data_dir,
        provider=provider,
    )


def _build_expanded_pipeline(args, config, layout):
    del config
    reference = load_expanded_kl_config(args.reference_manifest)
    sdk_root = next(item.source for item in layout.opponents if item.tier == "low")
    return GeneralsExpandedPolicyKLPipeline(
        reference=reference,
        legacy_run_dir=args.source_run,
        target_run_dir=args.target_run,
        expected_target_hash=args.expected_target_hash,
        state_pack_path=args.reference_manifest.parent / reference.state_pack,
        data_dir=args.data_dir,
        engine_root=layout.engine_root,
        engine_hash=layout.engine_hash,
        sdk_root=sdk_root,
        count_wall_time_s=args.count_wall_time,
        count_max_states=args.count_max_states,
    )


def handle(args) -> int:
    try:
        if args.generals_command == "plot-v9-paper":
            data = load_v9_figure_data(
                args.attribution_run,
                args.v9_run,
                args.legacy_kl_run,
                args.expanded_kl_run,
            )
            paths = render_v9_paper_figures(data, args.output_dir)
            print(json.dumps({
                "status": "complete",
                "run_dir": str(args.output_dir),
                "artifacts": [str(path) for path in paths],
            }, sort_keys=True))
            return 0
        config, layout = _assets(args)
        if args.generals_command == "check-leaderboard-qualification":
            return _check_leaderboard_qualification(args, config, layout)
        if args.generals_command == "build-leaderboard":
            return _build_leaderboard(args, config, layout)
        if args.generals_command == "champion-campaign-act":
            campaign_root = args.campaign_root.resolve()
            if campaign_root.is_relative_to(args.agentbench_root.resolve()):
                raise AssetValidationError(
                    "campaign output cannot modify the frozen AgentBench tree"
                )
            decision_space = args.decision_space
            if not decision_space.is_absolute():
                decision_space = args.agentbench_root / decision_space
            provider = (
                CodexProvider(
                    executable=args.codex_executable,
                    timeout_s=args.provider_timeout,
                    sandbox="workspace-write",
                )
                if args.provider == "codex"
                else ClaudeCodeProvider(
                    executable=args.claude_executable,
                    timeout_s=args.provider_timeout,
                    permission_mode=args.claude_permission_mode,
                )
            )
            pipeline = ChampionCampaignPipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                campaign_manifest_path=args.campaign_manifest,
                qualification_manifest_path=args.qualification_manifest,
                replay_skill_path=args.replay_skill,
                decision_space_path=decision_space,
                data_dir=args.data_dir,
                campaign_root=campaign_root,
                provider=provider,
                model=args.model,
                model_revision=args.model_revision,
            )
            result = pipeline.advance(args.replicate_id)
            print(json.dumps({
                "status": result.status,
                "campaign_root": str(result.campaign_root),
                "run_dir": str(result.run_dir),
                "replicate_id": result.replicate_id,
                "act_index": result.act_index,
                "promoted": result.promoted,
                "current_policy_hash": result.current_policy_hash,
                "checkpoint_evaluated": result.checkpoint_evaluated,
                "next_act_index": result.next_act_index,
                "qualification_receipt": (
                    str(campaign_root / "sealed/qualification-receipt.json")
                    if (campaign_root / "sealed/qualification-receipt.json").is_file()
                    else None
                ),
            }, sort_keys=True))
            return 0 if result.status in {
                "promoted", "candidate_rejected", "complete"
            } else 1
        if args.generals_command == "prepare":
            print(
                f"{config.benchmark_id}: {len(config.opponents)} opponents, "
                f"18 evaluation cases, 12 learning cases; engine "
                f"{layout.engine_hash[:12]}"
            )
            return 0
        if args.generals_command == "attribute-v8-regression":
            result = _build_attribution_pipeline(args, config, layout).run()
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "case_count_per_policy": result.case_count_per_policy,
                "diagnostic_state_count": result.diagnostic_state_count,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {"iterate-v9", "recover-v9"}:
            provider = (
                CodexProvider(
                    executable=args.codex_executable,
                    timeout_s=args.provider_timeout,
                    sandbox="workspace-write",
                )
                if args.generals_command == "iterate-v9"
                else _RecoveryOnlyProvider()
            )
            pipeline = _build_v9_pipeline(args, config, layout, provider)
            result = (
                pipeline.run()
                if args.generals_command == "iterate-v9"
                else pipeline.recover(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "evo_score_9": result.evo_score_9,
                "validation_passed": result.validation_passed,
                "formal_attempted": result.formal_attempted,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {
            "measure-policy-kl-expanded", "recover-policy-kl-expanded"
        }:
            pipeline = _build_expanded_pipeline(args, config, layout)
            result = (
                pipeline.run()
                if args.generals_command == "measure-policy-kl-expanded"
                else pipeline.recover(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "domains": result.summary.get("domains"),
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {
            "extend-policy-kl-v7",
            "recover-policy-kl-v7",
        }:
            pipeline = GeneralsPolicyKLExtensionPipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                reference_manifest_path=args.reference_manifest,
                source_run_dir=args.source_run,
                data_dir=args.data_dir,
            )
            result = (
                pipeline.run()
                if args.generals_command == "extend-policy-kl-v7"
                else pipeline.recover(args.failed_run)
            )
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "run_dir": str(result.run_dir),
                        "source_run_id": result.summary.get(
                            "source_run_id"
                        ),
                        "controlled_reference_policy_kl": (
                            result.summary.get(
                                "controlled_reference_policy_kl"
                            )
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0 if result.status == "complete" else 1
        if args.generals_command in {
            "extend-policy-kl-v8", "recover-policy-kl-v8"
        }:
            pipeline = GeneralsPolicyKLV8ExtensionPipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                reference_manifest_path=args.reference_manifest,
                source_run_dir=args.source_run,
                target_run_dir=args.target_run,
                expected_target_hash=args.expected_target_hash,
                data_dir=args.data_dir,
            )
            result = (
                pipeline.run()
                if args.generals_command == "extend-policy-kl-v8"
                else pipeline.recover(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "source_run_id": result.summary.get("source_run_id"),
                "controlled_reference_policy_kl": result.summary.get(
                    "controlled_reference_policy_kl"
                ),
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {
            "measure-policy-kl",
            "recover-policy-kl",
        }:
            pipeline = GeneralsPolicyKLPipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                reference_manifest_path=args.reference_manifest,
                data_dir=args.data_dir,
                count_wall_time_s=args.count_wall_time,
                count_max_states=args.count_max_states,
            )
            result = (
                pipeline.run()
                if args.generals_command == "measure-policy-kl"
                else pipeline.recover(args.failed_run)
            )
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "run_dir": str(result.run_dir),
                        "controlled_reference_policy_kl": (
                            result.summary.get(
                                "controlled_reference_policy_kl"
                            )
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 0 if result.status == "complete" else 1
        if args.generals_command == "calibrate-dev":
            calibration = load_calibration_config(
                args.calibration_manifest
            )
            calibration_source = resolve_calibration_candidate_source(
                calibration, args.agentbench_root
            )
            result = calibrate_development(
                config=config,
                assets=layout,
                calibration_config=calibration,
                calibration_source=calibration_source,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                selection_output=args.selection_output,
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "selection_path": str(result.selection_path),
                "selected_mode": result.selected_mode,
                "development_scores": result.development_scores,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "eval":
            provider = _RawOnlyProvider()
        else:
            provider = CodexProvider(
                executable=args.codex_executable,
                timeout_s=args.provider_timeout,
                sandbox="workspace-write",
            )
        if args.generals_command in {"iterate-v2", "recover-v2"}:
            pipeline = GeneralsHLRound2Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                calibration_manifest_path=args.calibration_manifest,
                calibration_selection_path=args.calibration_selection,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            )
            result = (
                pipeline.run()
                if args.generals_command == "iterate-v2"
                else pipeline.recover_provider_init_failure(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "gain_2": result.gain_2,
                "calibration_score": result.calibration_score,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "iterate-v3":
            result = GeneralsHLRound3Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            ).run()
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "gain_3": result.gain_3,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "behavior_gate_passed": result.gate_passed,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "iterate-v4":
            result = GeneralsHLRound4Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
                prior_attempt_run_dir=args.prior_attempt_run,
            ).run()
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "evo_score_4": result.evo_score_4,
                "gain_4": result.gain_4,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command == "iterate-v5":
            result = GeneralsHLRound5Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                expected_rollback_hash=args.expected_rollback_hash,
                campaign_budget_receipt=args.campaign_budget_receipt,
                data_dir=args.data_dir,
                provider=provider,
                prior_attempt_run_dir=args.prior_attempt_run,
            ).run()
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "evo_score_4": result.evo_score_4,
                "evo_score_5": result.evo_score_5,
                "gain_5": result.gain_5,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {"iterate-v6", "recover-v6"}:
            pipeline = GeneralsHLRound6Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                learning_manifest_path=args.learning_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            )
            result = (
                pipeline.run()
                if args.generals_command == "iterate-v6"
                else pipeline.recover(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_1": result.evo_score_1,
                "evo_score_2": result.evo_score_2,
                "evo_score_3": result.evo_score_3,
                "evo_score_4": result.evo_score_4,
                "evo_score_5": result.evo_score_5,
                "evo_score_6": result.evo_score_6,
                "gain_6": result.gain_6,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {"iterate-v7", "recover-v7"}:
            pipeline = GeneralsHLRound7Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                challenge_manifest_path=args.challenge_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            )
            result = (
                pipeline.run()
                if args.generals_command == "iterate-v7"
                else pipeline.recover(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_7": result.evo_score_7,
                "gain_7": result.gain_7,
                "validation_passed": result.validation_passed,
                "sealed_status": result.sealed_status,
                "champion_claim": result.champion_claim,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        if args.generals_command in {"iterate-v8", "recover-v8"}:
            pipeline = GeneralsHLRound8Pipeline.from_paths(
                agentbench_root=args.agentbench_root,
                manifest_path=args.manifest,
                challenge_manifest_path=args.challenge_manifest,
                replay_skill_path=args.replay_skill,
                parent_run_dir=args.parent_run,
                expected_parent_hash=args.expected_parent_hash,
                data_dir=args.data_dir,
                provider=provider,
            )
            result = (
                pipeline.run()
                if args.generals_command == "iterate-v8"
                else pipeline.recover(args.failed_run)
            )
            print(json.dumps({
                "status": result.status,
                "run_dir": str(result.run_dir),
                "raw_score": result.raw_score,
                "evo_score_8": result.evo_score_8,
                "gain_8": result.gain_8,
                "validation_passed": result.validation_passed,
                "formal_attempted": result.formal_attempted,
                "performance_target_met": result.performance_target_met,
                "global_act_count": result.global_act_count,
                "round_act_count": result.round_act_count,
                "runnable": result.runnable,
            }, sort_keys=True))
            return 0 if result.status == "complete" else 1
        result = GeneralsHLPipeline(
            config=config,
            assets=layout,
            data_dir=args.data_dir,
            provider=provider,
        ).run()
        print(result.run_dir)
        if args.generals_command == "eval":
            return 0 if result.raw_score is not None else 1
        return 0 if result.status == "complete" else 1
    except (AssetValidationError, ValueError) as exc:
        print(f"Generals asset validation failed: {exc}")
        return 2
