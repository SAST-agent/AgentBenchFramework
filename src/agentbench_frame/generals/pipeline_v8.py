"""Audited single-act clean-room v7-to-v8 Generals orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping

from agentbench_frame.tracking.provider import ProviderAdapter
from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.snapshot import WorkspaceManifest

from .action_profile import action_profile_payload, summarize_action_profile
from .assets import (
    load_pilot_config,
    require_valid_assets,
    resolve_assets,
    resolve_replay_skill,
)
from .challenge_v8 import (
    ChampionGate,
    build_round8_learning_cases,
    build_round8_sealed_cases,
    build_round8_validation_cases,
    evaluate_champion_gate,
    load_round8_challenge_config,
)
from .dense import build_dense_trace, summarize_dense_trace
from .evaluator import (
    GeneralsEvaluation,
    GeneralsEvaluator,
    build_evaluation_spec,
)
from .lineage_v8 import import_round8_source, load_round8_parent
from .measurement import ProbeState, measure_action_disagreement
from .models import (
    AssetLayout,
    PilotConfig,
    ReplaySkillAsset,
    Round8ChallengeConfig,
)
from .pipeline import GeneralsHLPipeline
from .pipeline_v7 import GeneralsHLRound7Pipeline
from .prompt import PromptBuildResult
from .prompt_v8 import build_round8_prompt, validate_round8_static_context
from .replay import (
    CriticalLearningEvidence,
    CriticalWindowSelection,
    build_critical_learning_evidence,
    build_learning_replay,
)


V8_SELECTION_REASONS = (
    "first_main_danger",
    "large_stack_inactive",
    "missed_counter_or_reinforcement",
    "economy_defense_conflict",
    "first_dense_divergence",
    "unsafe_or_low_value_macro",
    "final_decision",
)
V8_EDITABLE_FILES = frozenset(
    {
        "strategy.py",
        "state_view.py",
        "STRATEGY.md",
        "EXPERIENCE.md",
    }
)
V8_REQUIRED_FILES = frozenset(
    {
        "main.py",
        "strategy.py",
        "state_view.py",
        "STRATEGY.md",
        "EXPERIENCE.md",
    }
)
_RUNTIME_FORBIDDEN = re.compile(
    r"\b(?:random|socket|requests|urllib|subprocess|replay_id|"
    r"opponent_id|seed)\b|time\.time|os\.environ|__file__"
)


def _combine_learning_budgets(
    inherited: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, float | int | None]:
    keys = {
        str(key)
        for budget in (inherited, current)
        for key in budget
        if str(key).startswith("learning_")
    }
    combined: dict[str, float | int | None] = {}
    for key in sorted(keys):
        if key not in inherited or key not in current:
            combined[key] = None
            continue
        left = inherited[key]
        right = current[key]
        if left is None or right is None:
            combined[key] = None
            continue
        total = float(left) + float(right)
        combined[key] = total if key.endswith("_time_s") else int(total)
    return combined


@dataclass(frozen=True)
class Round8PipelineResult:
    run_dir: Path
    status: str
    runnable: bool
    raw_score: float | None
    evo_score_8: float | None
    gain_8: float | None
    validation_passed: bool
    formal_attempted: bool
    performance_target_met: bool
    sealed_status: str
    champion_claim: bool
    global_act_count: int
    round_act_count: int


class _Round8Abort(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


class GeneralsHLRound8Pipeline(GeneralsHLRound7Pipeline):
    """Run one clean-room learning act followed by held-out evaluation."""

    def __init__(
        self,
        config: PilotConfig,
        challenge: Round8ChallengeConfig,
        assets: AssetLayout,
        replay_skill: ReplaySkillAsset,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
        evaluator: GeneralsEvaluator | Any | None = None,
        rules_path: Path | None = None,
        prompt_max_bytes: int = 131_072,
    ):
        GeneralsHLPipeline.__init__(
            self,
            config,
            assets,
            data_dir,
            provider,
            evaluator=evaluator,
            rules_path=rules_path,
        )
        self.challenge = challenge
        self.replay_skill = replay_skill
        self.parent_run_dir = Path(parent_run_dir)
        self.expected_parent_hash = expected_parent_hash
        self.prompt_max_bytes = int(prompt_max_bytes)

    @classmethod
    def from_paths(
        cls,
        *,
        agentbench_root: Path,
        manifest_path: Path,
        challenge_manifest_path: Path,
        replay_skill_path: Path,
        parent_run_dir: Path,
        expected_parent_hash: str,
        data_dir: Path,
        provider: ProviderAdapter,
    ) -> "GeneralsHLRound8Pipeline":
        config = load_pilot_config(manifest_path)
        assets = resolve_assets(config, agentbench_root)
        require_valid_assets(assets)
        replay_skill = resolve_replay_skill(
            agentbench_root,
            replay_skill_path,
        )
        challenge = load_round8_challenge_config(
            challenge_manifest_path,
            config,
            engine_hash=assets.engine_hash,
            replay_skill_sha256=replay_skill.sha256,
        )
        return cls(
            config=config,
            challenge=challenge,
            assets=assets,
            replay_skill=replay_skill,
            parent_run_dir=parent_run_dir,
            expected_parent_hash=expected_parent_hash,
            data_dir=data_dir,
            provider=provider,
        )

    def _write_benchmark_artifacts(
        self,
        run_dir: Path,
        learning_cases: tuple,
        validation_cases: tuple,
        sealed_cases: tuple,
        formal_cases: tuple,
    ) -> None:
        benchmark_dir = run_dir / "benchmark"
        skill_target = benchmark_dir / "replay-skill.md"
        skill_target.parent.mkdir(parents=True, exist_ok=True)
        skill_target.write_text(self.replay_skill.text, encoding="utf-8")
        common = {
            "challenge_id": self.challenge.challenge_id,
            "opponent_id": self.challenge.opponent_id,
            "engine_hash": self.assets.engine_hash,
        }
        self._write_json(
            benchmark_dir / "v8-learning-spec.json",
            {
                **common,
                "prompt_visible": True,
                "seeds": list(self.challenge.learning_seeds),
                "seats": list(self.challenge.seats),
                "cases": [asdict(case) for case in learning_cases],
            },
        )
        self._write_json(
            benchmark_dir / "v8-validation-spec.json",
            {
                **common,
                "prompt_visible": False,
                "minimum_wins": self.challenge.validation_min_wins,
                "minimum_wins_per_seat": (
                    self.challenge.validation_min_wins_per_seat
                ),
                "cases": [asdict(case) for case in validation_cases],
            },
        )
        self._write_json(
            benchmark_dir / "v8-sealed-spec.json",
            {
                **common,
                "prompt_visible": False,
                "minimum_wins": self.challenge.sealed_min_wins,
                "minimum_wins_per_seat": (
                    self.challenge.sealed_min_wins_per_seat
                ),
                "cases": [asdict(case) for case in sealed_cases],
            },
        )
        self._write_json(
            benchmark_dir / "formal-spec.json",
            {
                "benchmark_id": self.config.benchmark_id,
                "engine_hash": self.assets.engine_hash,
                "prompt_visible": False,
                "evaluation_cases": [
                    asdict(case) for case in formal_cases
                ],
            },
        )
        self._write_json(
            benchmark_dir / "replay-skill.json",
            {
                "source_path": str(self.replay_skill.path),
                "sha256": self.replay_skill.sha256,
                "bytes": len(self.replay_skill.text.encode("utf-8")),
                "artifact_ref": str(skill_target),
            },
        )

    @staticmethod
    def _critical_feedback_v8(
        evaluation: GeneralsEvaluation,
        tier_by_case: Mapping[str, str],
    ) -> tuple[
        tuple[CriticalLearningEvidence, ...],
        tuple[CriticalWindowSelection, ...],
        tuple[ProbeState, ...],
    ]:
        evidence: list[CriticalLearningEvidence] = []
        selections: list[CriticalWindowSelection] = []
        probes: list[ProbeState] = []
        for match in evaluation.matches:
            replay_id = (
                f"learn8-high-s{match.seed}-p{match.evaluated_seat}"
            )
            raw = build_learning_replay(
                match,
                "v7",
                tier_by_case[match.case_id],
            )
            replay = replace(
                raw,
                replay_id=replay_id,
                decisions=tuple(
                    replace(
                        decision,
                        state_id=f"v7:{replay_id}-d{index}",
                    )
                    for index, decision in enumerate(
                        raw.decisions,
                        start=1,
                    )
                ),
            )
            trace = build_dense_trace(match)
            dense = summarize_dense_trace(match, trace)
            compact, selection = build_critical_learning_evidence(
                replay,
                dense,
                trace,
                max_decisions=6,
                selection_reasons=V8_SELECTION_REASONS,
                include_strategic_targets=True,
            )
            selected_ids = set(selection.selected_state_ids)
            evidence.append(compact)
            selections.append(selection)
            probes.extend(
                ProbeState(
                    decision.state_id,
                    {**decision.state, "my_seat": decision.seat},
                    decision.action,
                )
                for decision in replay.decisions
                if decision.state_id in selected_ids
            )
        return tuple(evidence), tuple(selections), tuple(probes)

    def _persist_action_profile_v8(
        self,
        run: Run,
        evaluation: GeneralsEvaluation,
        *,
        phase: str,
        artifact_path: Path,
        required: bool,
    ) -> dict[str, object] | None:
        try:
            payload = action_profile_payload(
                summarize_action_profile(evaluation)
            )
            self._write_json(artifact_path, payload)
            run.write(
                "behavior_diagnostics",
                diagnostic="action_profile",
                phase=phase,
                version=evaluation.version,
                artifact_ref=str(artifact_path),
                **payload,
            )
            return payload
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            run.write(
                "behavior_measurement_error",
                diagnostic="action_profile",
                phase=phase,
                version=evaluation.version,
                error=error,
            )
            if required:
                raise _Round8Abort(
                    "learning_profile_failed",
                    error,
                ) from exc
            return None

    @staticmethod
    def _scope_valid(changed_files: Iterable[str]) -> bool:
        return all(
            path in V8_EDITABLE_FILES
            or path.startswith("tests/")
            or path.startswith("policy/")
            for path in changed_files
        )

    @staticmethod
    def _performance_target_met(
        formal: GeneralsEvaluation | None,
        formal_cases: tuple,
        challenge: Round8ChallengeConfig,
    ) -> bool:
        if formal is None or formal.status != "complete":
            return False
        case_by_id = {case.case_id: case for case in formal_cases}
        valid = [
            result
            for result in formal.results
            if result.valid and result.case_id in case_by_id
        ]
        if len(valid) != len(formal_cases):
            return False
        wins = [result for result in valid if result.outcome == "win"]
        high = [
            result
            for result in wins
            if case_by_id[result.case_id].metadata.get("tier") == "high"
        ]
        per_seat = {
            seat: sum(
                case_by_id[result.case_id].first_player == seat
                for result in high
            )
            for seat in challenge.seats
        }
        return (
            len(high) >= challenge.formal_high_min_wins
            and len(wins) >= challenge.formal_total_min_wins
            and all(
                per_seat[seat] >= challenge.formal_high_min_wins_per_seat
                for seat in challenge.seats
            )
        )

    @staticmethod
    def _required_frozen_files_valid(
        source: Path,
        manifest: WorkspaceManifest,
    ) -> bool:
        return all(
            relative in manifest.files
            and (source / relative).is_file()
            and not (source / relative).is_symlink()
            for relative in V8_REQUIRED_FILES
        )

    @staticmethod
    def _runtime_source_valid(source: Path) -> bool:
        total = 0
        for path in source.rglob("*"):
            if path.is_symlink():
                return False
            if not path.is_file():
                continue
            total += path.stat().st_size
            relative = path.relative_to(source).as_posix()
            if relative.startswith("tests/") or path.suffix != ".py":
                continue
            text = path.read_text(encoding="utf-8")
            if _RUNTIME_FORBIDDEN.search(text):
                return False
        return total <= 10_485_760

    @staticmethod
    def _strategy_documents_valid(source: Path) -> bool:
        strategy = (source / "STRATEGY.md").read_text(
            encoding="utf-8"
        ).casefold()
        experience = (source / "EXPERIENCE.md").read_text(
            encoding="utf-8"
        ).casefold()
        strategy_groups = (
            ("beam",),
            ("top-k", "top k"),
            ("macro",),
            ("phase",),
            ("tie",),
            ("fallback",),
        )
        experience_groups = (
            ("retain", "retained"),
            ("reject", "rejected"),
            ("risk",),
        )
        return all(
            any(token in strategy for token in group)
            for group in strategy_groups
        ) and all(
            any(token in experience for token in group)
            for group in experience_groups
        )

    def _verify_runtime_source_v8(
        self,
        run: Run,
        source: Path,
        manifest: WorkspaceManifest,
        *,
        phase: str,
    ) -> None:
        try:
            actual = self.snapshotter.capture(source)
            verified = (
                actual.content_hash == manifest.content_hash
                and actual.files == manifest.files
            )
            actual_hash = actual.content_hash
            files_match: bool | None = actual.files == manifest.files
            error = None
        except Exception as exc:
            verified = False
            actual_hash = None
            files_match = None
            error = f"{type(exc).__name__}: {exc}"
        run.write(
            "behavior_diagnostics",
            diagnostic="frozen_source_verification",
            phase=phase,
            version="v8",
            source=str(source),
            expected_manifest_hash=manifest.content_hash,
            actual_manifest_hash=actual_hash,
            files_match=files_match,
            verified=verified,
            error=error,
        )
        if not verified:
            raise ValueError(
                f"{phase} source does not match frozen v8 manifest"
            )

    @staticmethod
    def _command_shape_valid(command: tuple[int, ...]) -> bool:
        opcode = command[0] if command else None
        if opcode == 1:
            return len(command) == 5 and command[4] > 0
        if opcode == 2:
            return len(command) == 4
        if opcode == 3:
            return len(command) == 3
        if opcode == 4:
            return (
                len(command) == 5
                if len(command) >= 3 and command[2] in (1, 2)
                else len(command) == 3
                if len(command) >= 3 and command[2] in (3, 4, 5)
                else False
            )
        if opcode == 5:
            return len(command) == 2
        if opcode == 6:
            return (
                len(command) == 6
                if len(command) >= 2 and command[1] == 3
                else len(command) == 4
                if len(command) >= 2 and command[1] in (1, 2, 4)
                else False
            )
        if opcode == 7:
            return len(command) == 3
        return command == (8,)

    def _verify_policy_probe_contract(
        self,
        workspace: Path,
        probes: tuple[ProbeState, ...],
    ) -> dict[str, object]:
        start = time.perf_counter()
        first = self._probe_actions(workspace, probes)
        second = self._probe_actions(workspace, probes)
        elapsed = time.perf_counter() - start
        if first != second or set(first) != {
            probe.state_id for probe in probes
        }:
            raise ValueError("candidate policy probes are nondeterministic")
        for probe in probes:
            commands = first[probe.state_id]
            end_positions = [
                index
                for index, command in enumerate(commands)
                if command == (8,)
            ]
            if end_positions != [len(commands) - 1]:
                raise ValueError(
                    "candidate macro must contain exactly one final [8]"
                )
            if len(commands) - 1 > 8:
                raise ValueError(
                    "candidate macro exceeds eight non-end primitives"
                )
            if not all(
                self._command_shape_valid(command)
                for command in commands
            ):
                raise ValueError("candidate emitted malformed command")
            actor = int(probe.state.get("my_seat", 0))
            cells = probe.state.get("cells", {})
            known_armies = {
                str(key): int(value["army"])
                for key, value in cells.items()
                if isinstance(value, Mapping)
                and int(value.get("player", -1)) == actor
            } if isinstance(cells, Mapping) else {}
            for command in commands[:-1]:
                if command[0] != 1:
                    continue
                source_key = f"{command[1]},{command[2]}"
                if source_key not in known_armies:
                    continue
                source_army = known_armies[source_key]
                requested = command[4]
                executed = (
                    source_army - 1
                    if requested == 1_000_000_000
                    else requested
                )
                if executed <= 0 or executed >= source_army:
                    raise ValueError(
                        "candidate movement violates one-army source reserve"
                    )
                known_armies[source_key] = source_army - executed
        mean_call_s = elapsed / max(1, 2 * len(probes))
        if mean_call_s >= 1.5:
            raise ValueError(
                "candidate lacks headroom below the two-second limit"
            )
        return {
            "decision_count": len(probes),
            "deterministic": True,
            "mean_call_s": mean_call_s,
            "max_non_end_primitives": max(
                (len(commands) - 1 for commands in first.values()),
                default=0,
            ),
            "two_second_limit_s": 2.0,
            "headroom_limit_s": 1.5,
        }

    @staticmethod
    def _gate_payload(
        gate: ChampionGate,
        *,
        source_hash: str,
        engine_hash: str,
        minimum_wins: int,
        minimum_wins_per_seat: int,
        suite_digest: str,
        claim_field: str,
    ) -> dict[str, object]:
        payload = asdict(gate)
        payload.update(
            {
                "source_hash": source_hash,
                "engine_hash": engine_hash,
                "minimum_wins": minimum_wins,
                "minimum_wins_per_seat": minimum_wins_per_seat,
                "suite_digest": suite_digest,
                claim_field: gate.passed,
            }
        )
        return payload

    @staticmethod
    def _suite_digest(cases: tuple) -> str:
        encoded = json.dumps(
            [asdict(case) for case in cases],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def run(self) -> Round8PipelineResult:
        fresh_recovery = getattr(self, "_fresh_recovery", None)
        lineage = load_round8_parent(
            self.parent_run_dir,
            self.expected_parent_hash,
        )
        run = Run.start(
            game="28_generals",
            agent="generals-hl",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "benchmark_id": self.config.benchmark_id,
                "challenge_id": self.challenge.challenge_id,
                "provider": self.provider.provider_name,
                "budget_phase": "learning",
                "engine_hash": self.assets.engine_hash,
                "round": 8,
                "lineage_mode": "v7_to_clean_room_v8",
                **(
                    {
                        "failed_run_id": fresh_recovery["run_id"],
                        "recovery_mode": "fresh_retry",
                    }
                    if fresh_recovery is not None
                    else {}
                ),
            },
        )
        run_dir = Path(run.run_dir)
        raw_score = lineage.raw_score
        evo_score_8: float | None = None
        gain_8: float | None = None
        global_act_count = lineage.global_act_count
        round_act_count = 0
        status = "failed"
        runnable = False
        validation_passed = False
        performance_target_met = False
        sealed_status = "not_opened"
        champion_claim = False
        error: str | None = None
        learning: GeneralsEvaluation | None = None
        validation: GeneralsEvaluation | None = None
        formal: GeneralsEvaluation | None = None
        sealed: GeneralsEvaluation | None = None
        validation_gate: ChampionGate | None = None
        prompt: PromptBuildResult | None = None
        act = None
        v8_manifest: WorkspaceManifest | None = None
        validation_error: str | None = None
        formal_error: str | None = None
        sealed_error: str | None = None
        formal_attempted = False
        validation_payload: dict[str, object] = {
            "status": "not_run",
            "passed": False,
            "score": None,
        }
        sealed_payload: dict[str, object] = {
            "status": "not_opened",
            "champion_claim": False,
            "score": None,
            "reason": "validation_not_run",
        }
        action_profiles: dict[str, dict[str, object] | None] = {
            phase: None
            for phase in ("learning", "validation", "formal", "sealed")
        }
        try:
            imported_v7 = import_round8_source(
                lineage,
                run_dir,
                self.snapshotter,
            )
            workspace = run_dir / "workspace"
            frozen_v7_source = run_dir / "versions" / "v7" / "source"
            v7_strategy = (workspace / "strategy.py").read_text(
                encoding="utf-8"
            )
            v7_experience = (workspace / "EXPERIENCE.md").read_text(
                encoding="utf-8"
            )
            rules_text = self.rules_path.read_text(encoding="utf-8")
            validate_round8_static_context(
                v7_strategy=v7_strategy,
                v7_experience=v7_experience,
                rules_text=rules_text,
                replay_skill_text=self.replay_skill.text,
            )

            learning_cases = build_round8_learning_cases(
                self.config,
                self.challenge,
            )
            validation_cases = build_round8_validation_cases(
                self.config,
                self.challenge,
            )
            sealed_cases = build_round8_sealed_cases(
                self.config,
                self.challenge,
            )
            formal_cases = tuple(
                build_evaluation_spec(self.config).cases
            )
            self._write_benchmark_artifacts(
                run_dir,
                learning_cases,
                validation_cases,
                sealed_cases,
                formal_cases,
            )
            run.write(
                "lineage_import",
                parent_run_id=lineage.parent_run_id,
                parent_version="v7",
                starting_version="v7",
                version="v7",
                manifest_hash=imported_v7.content_hash,
                prior_score_history=list(lineage.prior_score_history),
                global_coding_agent_act=global_act_count,
            )
            run.write(
                "version",
                version="v7",
                status="lineage_parent",
                manifest_hash=imported_v7.content_hash,
                parent_run_id=lineage.parent_run_id,
            )
            run.write(
                "benchmark_spec",
                benchmark_id=self.config.benchmark_id,
                challenge_id=self.challenge.challenge_id,
                learning_cases=[
                    asdict(case) for case in learning_cases
                ],
                validation_cases=[
                    asdict(case) for case in validation_cases
                ],
                sealed_cases=[asdict(case) for case in sealed_cases],
                evaluation_cases=[
                    asdict(case) for case in formal_cases
                ],
                engine_hash=self.assets.engine_hash,
                replay_skill_sha256=self.replay_skill.sha256,
            )
            evaluator = self.evaluator or self._production_evaluator(run_dir)
            learning = evaluator.evaluate(
                frozen_v7_source,
                "v7",
                "learning",
                run,
                cases=learning_cases,
            )
            run.write(
                "learning_validation",
                version="v7",
                phase="learning",
                challenge_id=self.challenge.challenge_id,
                status=learning.status,
                case_count=len(learning_cases),
                valid_case_count=sum(
                    result.valid for result in learning.results
                ),
            )
            if not self._complete_learning(learning, len(learning_cases)):
                raise _Round8Abort(
                    "learning_failed",
                    "v7 champion learning suite is incomplete",
                )

            tiers = {
                case.case_id: str(case.metadata["tier"])
                for case in learning_cases
            }
            evidence, selections, probes = self._critical_feedback_v8(
                learning,
                tiers,
            )
            evidence_dir = run_dir / "learning-evidence"
            for item, selection in zip(evidence, selections):
                evidence_path = evidence_dir / f"{item.replay_id}.json"
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                evidence_path.write_text(
                    item.to_json() + "\n",
                    encoding="utf-8",
                )
                run.write(
                    "critical_window_selection",
                    version="v7",
                    replay_id=selection.replay_id,
                    total_decision_count=selection.total_decision_count,
                    selected_state_ids=list(selection.selected_state_ids),
                    reasons={
                        key: list(value)
                        for key, value in selection.reasons.items()
                    },
                    omitted_decision_count=(
                        selection.omitted_decision_count
                    ),
                    artifact_ref=str(evidence_path),
                )
            learning_profile = self._persist_action_profile_v8(
                run,
                learning,
                phase="learning",
                artifact_path=(
                    run_dir
                    / "diagnostics"
                    / "v7-learning-action-profile.json"
                ),
                required=True,
            )
            action_profiles["learning"] = learning_profile
            assert learning_profile is not None
            try:
                prompt = build_round8_prompt(
                    benchmark_id=self.config.benchmark_id,
                    parent_content_hash=lineage.v7_manifest.content_hash,
                    v7_strategy=v7_strategy,
                    v7_experience=v7_experience,
                    rules_text=rules_text,
                    replay_skill_text=self.replay_skill.text,
                    replay_skill_sha256=self.replay_skill.sha256,
                    evidence=evidence,
                    action_profile=learning_profile,
                    max_bytes=self.prompt_max_bytes,
                )
            except ValueError as exc:
                raise _Round8Abort("prompt_incomplete", str(exc)) from exc

            expected_ids = {
                f"learn8-high-s{seed}-p{seat}"
                for seed in self.challenge.learning_seeds
                for seat in self.challenge.seats
            }
            if (
                prompt.truncated
                or set(prompt.included_episode_ids) != expected_ids
                or prompt.omitted_episode_ids
            ):
                raise _Round8Abort(
                    "prompt_incomplete",
                    "v8 prompt omitted a declared learning episode",
                )
            feedback_receipt = {
                "episodes": prompt.feedback_episodes_read,
                "decision_records": (
                    prompt.feedback_decision_records_read
                ),
                "serialized_bytes": (
                    prompt.feedback_serialized_bytes_read
                ),
                "prompt_bytes": prompt.prompt_bytes,
                "estimated_tokens": prompt.estimated_tokens,
                "included_episode_ids": list(
                    prompt.included_episode_ids
                ),
                "omitted_episode_ids": list(
                    prompt.omitted_episode_ids
                ),
                "selection_policy": prompt.selection_policy,
            }
            run.write("feedback_read", **feedback_receipt)
            provider_dir = run_dir / "provider"
            provider_dir.mkdir(parents=True, exist_ok=True)
            canonical_prompt = provider_dir / "codex-act-v8.prompt.md"
            canonical_prompt.write_text(
                prompt.prompt,
                encoding="utf-8",
            )
            shutil.copy2(canonical_prompt, provider_dir / "prompt.txt")
            prompt_manifest = dict(prompt.manifest)
            prompt_manifest.update({
                "included_episode_ids": list(
                    prompt.included_episode_ids
                ),
                "omitted_episode_ids": [],
                "estimated_tokens": prompt.estimated_tokens,
            })
            self._write_json(
                provider_dir / "codex-act-v8.prompt.json",
                prompt_manifest,
            )
            self._write_json(
                provider_dir / "prompt-manifest.json",
                prompt_manifest,
            )
            self._write_json(
                provider_dir / "feedback-receipt.json",
                feedback_receipt,
            )
            raw_path = provider_dir / "codex-act-v8.raw.jsonl"
            stderr_path = provider_dir / "codex-act-v8.stderr.log"
            raw_path.touch()
            stderr_path.touch()
            controller = run.create_coding_agent_controller(
                self.provider,
                snapshotter=self.snapshotter,
                budget_phase="learning",
            )
            act = controller.run_act(
                {
                    "prompt": prompt.prompt,
                    "workspace_root": str(workspace),
                    "raw_output_path": str(raw_path),
                    "stderr_output_path": str(stderr_path),
                    "max_artifact_bytes": (
                        self.config.limits.max_artifact_bytes
                    ),
                },
                workspace_root=str(workspace),
                version_before="v7",
                previous_manifest=imported_v7,
            )
            round_act_count = 1
            global_act_count = lineage.global_act_count + 1
            self._write_json(
                provider_dir / "provider-budget.json",
                run.budget_snapshot(),
            )

            v8_manifest = self._write_version(
                run_dir,
                "v8",
                workspace,
                previous=imported_v7,
            )
            frozen_v8_source = run_dir / "versions" / "v8" / "source"
            self.snapshotter.write_unified_patch(
                frozen_v7_source,
                frozen_v8_source,
                run_dir / "versions" / "v7-to-v8.patch",
            )
            provider_completed = act.status == "completed"
            scope_valid = self._scope_valid(v8_manifest.changed_files)
            manifest_valid = self._frozen_manifest_matches(
                frozen_v8_source,
                v8_manifest,
            )
            required_valid = self._required_frozen_files_valid(
                frozen_v8_source,
                v8_manifest,
            )
            main_unchanged = (
                "main.py" not in v8_manifest.changed_files
                and v8_manifest.files.get("main.py")
                == imported_v7.files.get("main.py")
                and (frozen_v8_source / "main.py").read_bytes()
                == (frozen_v7_source / "main.py").read_bytes()
            )
            runtime_source_valid = self._runtime_source_valid(
                frozen_v8_source
            )
            documents_valid = (
                required_valid
                and self._strategy_documents_valid(frozen_v8_source)
            )
            tests_ok = False
            test_output = "candidate tests skipped before source validation\n"
            probe_diagnostics: dict[str, object] | None = None
            if all(
                (
                    provider_completed,
                    scope_valid,
                    manifest_valid,
                    required_valid,
                    main_unchanged,
                    runtime_source_valid,
                    documents_valid,
                )
            ):
                try:
                    test_workspace = self._materialize_verified_source(
                        frozen_v8_source,
                        run_dir / "isolated-workspaces/v8-tests",
                        v8_manifest,
                    )
                    self._verify_runtime_source_v8(
                        run,
                        test_workspace,
                        v8_manifest,
                        phase="tests",
                    )
                    tests_ok, test_output = self._run_candidate_tests(
                        test_workspace
                    )
                    if tests_ok:
                        probe_workspace = (
                            self._materialize_verified_source(
                                frozen_v8_source,
                                run_dir
                                / "isolated-workspaces/v8-probes",
                                v8_manifest,
                            )
                        )
                        self._verify_runtime_source_v8(
                            run,
                            probe_workspace,
                            v8_manifest,
                            phase="probes",
                        )
                        probe_diagnostics = (
                            self._verify_policy_probe_contract(
                                probe_workspace,
                                probes,
                            )
                        )
                except Exception as exc:
                    tests_ok = False
                    test_output += (
                        "\nFramework probe failure: "
                        f"{type(exc).__name__}: {exc}\n"
                    )
            tests_path = run_dir / "versions" / "v8" / "tests.log"
            tests_path.write_text(test_output, encoding="utf-8")
            runnable = all(
                (
                    provider_completed,
                    scope_valid,
                    manifest_valid,
                    required_valid,
                    main_unchanged,
                    runtime_source_valid,
                    documents_valid,
                    tests_ok,
                    probe_diagnostics is not None,
                )
            )
            run.write(
                "version",
                version="v8",
                version_before="v7",
                status=(
                    "available"
                    if runnable
                    else "provider_failed"
                    if not provider_completed
                    else "invalid"
                ),
                manifest_hash=v8_manifest.content_hash,
                changed_files=v8_manifest.changed_files,
                provider_status=act.status,
                scope_valid=scope_valid,
                main_unchanged=main_unchanged,
                frozen_manifest_valid=manifest_valid,
                required_source_files_valid=required_valid,
                runtime_source_valid=runtime_source_valid,
                strategy_documents_valid=documents_valid,
                tests_passed=tests_ok,
                probe_diagnostics=probe_diagnostics,
                runnable=runnable,
            )
            if not runnable:
                status = (
                    "provider_failed"
                    if not provider_completed
                    else "invalid_version"
                )
            else:
                old_actions = {
                    probe.state_id: probe.old_action for probe in probes
                }
                probe_workspace = self._materialize_verified_source(
                    frozen_v8_source,
                    run_dir / "isolated-workspaces/v8-behavior",
                    v8_manifest,
                )
                new_actions = self._probe_actions(probe_workspace, probes)
                behavior = measure_action_disagreement(probes, new_actions)
                run.write(
                    "behavior_change",
                    version_before="v7",
                    version_after="v8",
                    probe_set="v8_learning_states",
                    challenge_id=self.challenge.challenge_id,
                    decision_count=len(probes),
                    action_disagreement_trace=list(behavior.trace),
                    action_disagreement=behavior.mean,
                    policy_kl=behavior.policy_kl,
                    policy_kl_status=behavior.policy_kl_status,
                    old_action_count=len(old_actions),
                    occupancy_shift=None,
                )

                try:
                    validation_workspace = (
                        self._materialize_verified_source(
                            frozen_v8_source,
                            run_dir
                            / "isolated-workspaces/v8-validation",
                            v8_manifest,
                        )
                    )
                    self._verify_runtime_source_v8(
                        run,
                        validation_workspace,
                        v8_manifest,
                        phase="validation",
                    )
                    validation = evaluator.evaluate(
                        validation_workspace,
                        "v8",
                        "validation",
                        run,
                        cases=validation_cases,
                    )
                    action_profiles["validation"] = (
                        self._persist_action_profile_v8(
                            run,
                            validation,
                            phase="validation",
                            artifact_path=(
                                run_dir
                                / "diagnostics"
                                / "v8-validation-action-profile.json"
                            ),
                            required=False,
                        )
                    )
                    validation_gate = evaluate_champion_gate(
                        results=validation.results,
                        cases=validation_cases,
                        minimum_wins=(
                            self.challenge.validation_min_wins
                        ),
                        minimum_wins_per_seat=(
                            self.challenge.validation_min_wins_per_seat
                        ),
                    )
                    validation_passed = validation_gate.passed
                    validation_payload = self._gate_payload(
                        validation_gate,
                        source_hash=v8_manifest.content_hash,
                        engine_hash=self.assets.engine_hash,
                        minimum_wins=(
                            self.challenge.validation_min_wins
                        ),
                        minimum_wins_per_seat=(
                            self.challenge.validation_min_wins_per_seat
                        ),
                        suite_digest=self._suite_digest(validation_cases),
                        claim_field="sealed_suite_opened",
                    )
                except Exception as exc:
                    validation_error = f"{type(exc).__name__}: {exc}"
                    validation_passed = False
                    validation_payload = {
                        "status": "error",
                        "passed": False,
                        "sealed_suite_opened": False,
                        "score": None,
                        "reason": "validation_evaluation_error",
                        "error": validation_error,
                        "source_hash": v8_manifest.content_hash,
                        "engine_hash": self.assets.engine_hash,
                        "suite_digest": self._suite_digest(
                            validation_cases
                        ),
                    }
                    run.write(
                        "pipeline_error",
                        phase="validation",
                        status="validation_error_non_blocking",
                        error=validation_error,
                    )
                self._write_json(
                    run_dir / "validation-gate.json",
                    validation_payload,
                )
                run.write("champion_validation_gate", **validation_payload)

                try:
                    formal_workspace = self._materialize_verified_source(
                        frozen_v8_source,
                        run_dir / "isolated-workspaces/v8-formal",
                        v8_manifest,
                    )
                    self._verify_runtime_source_v8(
                        run,
                        formal_workspace,
                        v8_manifest,
                        phase="formal",
                    )
                    formal_attempted = True
                    formal = evaluator.evaluate(
                        formal_workspace,
                        "v8",
                        "formal",
                        run,
                        cases=formal_cases,
                    )
                    action_profiles["formal"] = (
                        self._persist_action_profile_v8(
                            run,
                            formal,
                            phase="formal",
                            artifact_path=(
                                run_dir
                                / "diagnostics"
                                / "v8-formal-action-profile.json"
                            ),
                            required=False,
                        )
                    )
                    evo_score_8 = formal.score
                    gain_8 = (
                        evo_score_8 - raw_score
                        if evo_score_8 is not None
                        else None
                    )
                    performance_target_met = self._performance_target_met(
                        formal,
                        formal_cases,
                        self.challenge,
                    )
                    run.write(
                        "evaluation",
                        phase="evolved_8",
                        version="v8",
                        status=formal.status,
                        score=formal.score,
                        gain=gain_8,
                        global_coding_agent_act=global_act_count,
                        per_tier=dict(formal.per_tier),
                        seat_gap=formal.seat_gap,
                    )
                    run.record_act_evaluation(
                        act.act_id,
                        {
                            "benchmark_id": self.config.benchmark_id,
                            "version": "v8",
                            "score": formal.score,
                            "gain": gain_8,
                            "global_coding_agent_act": global_act_count,
                        },
                    )
                except Exception as exc:
                    formal_error = f"{type(exc).__name__}: {exc}"
                    run.write(
                        "pipeline_error",
                        phase="formal",
                        status="formal_error",
                        error=formal_error,
                    )

                if self.challenge.sealed_seeds and validation_passed:
                    try:
                        sealed_workspace = (
                            self._materialize_verified_source(
                                frozen_v8_source,
                                run_dir
                                / "isolated-workspaces/v8-sealed",
                                v8_manifest,
                            )
                        )
                        self._verify_runtime_source_v8(
                            run,
                            sealed_workspace,
                            v8_manifest,
                            phase="sealed",
                        )
                        sealed = evaluator.evaluate(
                            sealed_workspace,
                            "v8",
                            "sealed",
                            run,
                            cases=sealed_cases,
                        )
                        action_profiles["sealed"] = (
                            self._persist_action_profile_v8(
                                run,
                                sealed,
                                phase="sealed",
                                artifact_path=(
                                    run_dir
                                    / "diagnostics"
                                    / "v8-sealed-action-profile.json"
                                ),
                                required=False,
                            )
                        )
                        sealed_gate = evaluate_champion_gate(
                            results=sealed.results,
                            cases=sealed_cases,
                            minimum_wins=(
                                self.challenge.sealed_min_wins
                            ),
                            minimum_wins_per_seat=(
                                self.challenge.sealed_min_wins_per_seat
                            ),
                        )
                        sealed_status = sealed_gate.status
                        champion_claim = sealed_gate.passed
                        sealed_payload = self._gate_payload(
                            sealed_gate,
                            source_hash=v8_manifest.content_hash,
                            engine_hash=self.assets.engine_hash,
                            minimum_wins=(
                                self.challenge.sealed_min_wins
                            ),
                            minimum_wins_per_seat=(
                                self.challenge.sealed_min_wins_per_seat
                            ),
                            suite_digest=self._suite_digest(sealed_cases),
                            claim_field="champion_claim",
                        )
                    except Exception as exc:
                        sealed_error = f"{type(exc).__name__}: {exc}"
                        sealed_status = "error"
                        champion_claim = False
                        sealed_payload = {
                            "status": "error",
                            "champion_claim": False,
                            "score": None,
                            "reason": "sealed_evaluation_error",
                            "error": sealed_error,
                            "source_hash": v8_manifest.content_hash,
                            "engine_hash": self.assets.engine_hash,
                            "suite_digest": self._suite_digest(
                                sealed_cases
                            ),
                        }
                        run.write(
                            "pipeline_error",
                            phase="sealed",
                            status="sealed_error",
                            error=sealed_error,
                        )
                else:
                    sealed_status = "not_opened"
                    sealed_payload = {
                        "status": "not_opened",
                        "champion_claim": False,
                        "score": None,
                        "reason": (
                            "validation_evaluation_error"
                            if validation_error is not None
                            else "validation_gate_failed"
                        ),
                        "source_hash": v8_manifest.content_hash,
                        "engine_hash": self.assets.engine_hash,
                        "suite_digest": self._suite_digest(sealed_cases),
                    }
                self._write_json(
                    run_dir / "sealed-claim.json",
                    sealed_payload,
                )
                run.write("champion_sealed_claim", **sealed_payload)
                if formal is None:
                    status = "formal_failed"
                elif formal.status != "complete":
                    status = "formal_incomplete"
                elif (
                    validation_gate is None
                    or validation_gate.status
                    in {"invalid", "incomplete"}
                ):
                    status = "validation_incomplete"
                elif sealed_status in {"invalid", "incomplete", "error"}:
                    status = "sealed_incomplete"
                else:
                    status = "complete"
        except _Round8Abort as exc:
            status = exc.status
            error = str(exc)
            run.write("pipeline_error", status=status, error=error)
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            run.write("pipeline_error", status=status, error=error)
        finally:
            run.writer.flush()
            quality = inspect_event_file(
                run_dir / "events.jsonl"
            ).to_dict()
            self._write_json(run_dir / "quality.json", quality)
            budget = run.budget_snapshot()
            local_learning_budget = {
                key: value
                for key, value in budget.items()
                if key.startswith("learning_")
            }
            cumulative_learning_budget = _combine_learning_budgets(
                lineage.learning_budget,
                local_learning_budget,
            )
            formal_scores = {
                tier: (
                    formal.per_tier.get(tier)
                    if formal is not None
                    else None
                )
                for tier in ("high", "medium", "low")
            }
            score_history = [
                *lineage.prior_score_history,
                evo_score_8,
            ]
            feedback_summary = (
                {
                    "episodes": prompt.feedback_episodes_read,
                    "decision_records": (
                        prompt.feedback_decision_records_read
                    ),
                    "serialized_bytes": (
                        prompt.feedback_serialized_bytes_read
                    ),
                    "prompt_bytes": prompt.prompt_bytes,
                    "estimated_tokens": prompt.estimated_tokens,
                    "included_episode_ids": list(
                        prompt.included_episode_ids
                    ),
                    "omitted_episode_ids": list(
                        prompt.omitted_episode_ids
                    ),
                    "selection_policy": prompt.selection_policy,
                }
                if prompt is not None
                else None
            )
            run.finish({
                "total_episodes": None,
                "total_steps": None,
                "total_reward": None,
                "win_rate": None,
                "wins": None,
                "losses": None,
                "draws": None,
                "status": status,
                "benchmark_id": self.config.benchmark_id,
                "challenge_id": self.challenge.challenge_id,
                "raw_score": raw_score,
                "evo_score": evo_score_8,
                "evo_score_1": lineage.evo_score_1,
                "evo_score_2": lineage.evo_score_2,
                "evo_score_3": lineage.evo_score_3,
                "evo_score_4": lineage.evo_score_4,
                "evo_score_5": lineage.evo_score_5,
                "evo_score_6": lineage.evo_score_6,
                "evo_score_8": evo_score_8,
                "gain": gain_8,
                "gain_1": lineage.evo_score_1 - raw_score,
                "gain_2": lineage.evo_score_2 - raw_score,
                "gain_3": lineage.evo_score_3 - raw_score,
                "gain_4": lineage.evo_score_4 - raw_score,
                "gain_5": lineage.evo_score_5 - raw_score,
                "gain_6": lineage.evo_score_6 - raw_score,
                "gain_8": gain_8,
                "benchmark_score": evo_score_8,
                "formal_score": (
                    formal.score if formal is not None else None
                ),
                "formal_scores": formal_scores,
                "formal_high_score": formal_scores["high"],
                "formal_medium_score": formal_scores["medium"],
                "formal_low_score": formal_scores["low"],
                "evaluation_status": (
                    formal.status
                    if formal is not None
                    else "error"
                    if formal_attempted
                    else "not_run"
                ),
                "formal_attempted": formal_attempted,
                "validation_error": validation_error,
                "formal_error": formal_error,
                "sealed_error": sealed_error,
                "champion_validation": validation_payload,
                "champion_sealed": sealed_payload,
                "champion_claim": champion_claim,
                "validation_passed": validation_passed,
                "performance_target_met": performance_target_met,
                "sealed_status": sealed_status,
                "score_history": score_history,
                "AUC_coding_agent_act": None,
                "AUC_episode": None,
                "AUC_env_step": None,
                "AUC_token": None,
                "AUC_time": None,
                "auc_status": "unavailable_missing_score_point",
                "act_count": global_act_count,
                "round_act_count": round_act_count,
                "parent_run_id": lineage.parent_run_id,
                "parent_version": "v7",
                "starting_version": "v7",
                "parent_content_hash": lineage.v7_manifest.content_hash,
                "failed_run_id": (
                    fresh_recovery["run_id"]
                    if fresh_recovery is not None
                    else None
                ),
                "recovery_mode": (
                    "fresh_retry"
                    if fresh_recovery is not None
                    else None
                ),
                "source_learning_budget": (
                    fresh_recovery["local_learning_budget"]
                    if fresh_recovery is not None
                    else None
                ),
                "runnable": runnable,
                "feedback_read": feedback_summary,
                "action_profiles": action_profiles,
                "error": error,
                "local_learning_budget": local_learning_budget,
                "cumulative_learning_budget": (
                    cumulative_learning_budget
                ),
                "learning_results": self._result_rows((learning,)),
                "validation_results": self._result_rows((validation,)),
                "formal_results": self._result_rows((formal,)),
                "sealed_results": self._result_rows((sealed,)),
                "benchmark_results": self._result_rows((formal,)),
            })
        return Round8PipelineResult(
            run_dir=run_dir,
            status=status,
            runnable=runnable,
            raw_score=raw_score,
            evo_score_8=evo_score_8,
            gain_8=gain_8,
            validation_passed=validation_passed,
            formal_attempted=formal_attempted,
            performance_target_met=performance_target_met,
            sealed_status=sealed_status,
            champion_claim=champion_claim,
            global_act_count=global_act_count,
            round_act_count=round_act_count,
        )

    def _validate_recovery_prompt(
        self,
        failed_run: Path,
    ) -> dict[str, object]:
        provider_dir = failed_run / "provider"
        prompt_path = provider_dir / "codex-act-v8.prompt.md"
        manifest = self._read_json_object(
            provider_dir / "prompt-manifest.json",
            "v8 prompt manifest",
        )
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"recovery v8 prompt is unavailable: {exc}"
            ) from exc
        observed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if manifest.get("prompt_sha256") != observed:
            raise ValueError("recovery v8 prompt digest changed")
        if manifest.get("replay_skill_sha256") != self.replay_skill.sha256:
            raise ValueError("recovery v8 prompt replay-skill digest changed")
        expected_ids = [
            f"learn8-high-s{seed}-p{seat}"
            for seed in self.challenge.learning_seeds
            for seat in self.challenge.seats
        ]
        episode_ids = manifest.get(
            "episode_ids",
            manifest.get("included_episode_ids"),
        )
        if episode_ids != expected_ids:
            raise ValueError(
                "recovery v8 prompt learning episode set changed"
            )
        forbidden = (
            *self.challenge.validation_seeds,
            *self.challenge.sealed_seeds,
        )
        if any(str(seed) in prompt for seed in forbidden):
            raise ValueError(
                "recovery v8 prompt contains held-out challenge data"
            )
        return manifest

    def _validate_recovery_specs(self, failed_run: Path) -> None:
        expected = {
            "v8-learning-spec.json": [
                asdict(case)
                for case in build_round8_learning_cases(
                    self.config,
                    self.challenge,
                )
            ],
            "v8-validation-spec.json": [
                asdict(case)
                for case in build_round8_validation_cases(
                    self.config,
                    self.challenge,
                )
            ],
            "v8-sealed-spec.json": [
                asdict(case)
                for case in build_round8_sealed_cases(
                    self.config,
                    self.challenge,
                )
            ],
        }
        for filename, cases in expected.items():
            payload = self._read_json_object(
                failed_run / "benchmark" / filename,
                filename,
            )
            if (
                payload.get("challenge_id")
                != self.challenge.challenge_id
                or payload.get("engine_hash") != self.assets.engine_hash
                or payload.get("cases") != cases
            ):
                raise ValueError(
                    f"recovery {filename} changed from frozen contract"
                )
        formal_payload = self._read_json_object(
            failed_run / "benchmark" / "formal-spec.json",
            "formal spec",
        )
        formal_cases = [
            asdict(case)
            for case in build_evaluation_spec(self.config).cases
        ]
        if (
            formal_payload.get("benchmark_id")
            != self.config.benchmark_id
            or formal_payload.get("engine_hash") != self.assets.engine_hash
            or formal_payload.get("evaluation_cases") != formal_cases
        ):
            raise ValueError("recovery formal spec changed")

    def _load_recovery_summary(
        self,
        failed_run_dir: Path,
    ) -> tuple[Path, dict[str, object]]:
        failed_run = Path(failed_run_dir).resolve()
        summary = self._read_json_object(
            failed_run / "summary.json",
            "v8 summary",
        )
        if summary.get("challenge_id") != self.challenge.challenge_id:
            raise ValueError("recovery challenge ID changed")
        if summary.get("parent_content_hash") != self.expected_parent_hash:
            raise ValueError("recovery parent content hash changed")
        if summary.get("act_count") not in {8, 9}:
            raise ValueError("recovery global act count is invalid")
        self._validate_recovery_specs(failed_run)
        if (failed_run / "provider/prompt-manifest.json").is_file():
            self._validate_recovery_prompt(failed_run)
        return failed_run, summary

    def _load_frozen_recovery_source(
        self,
        failed_run: Path,
        summary: Mapping[str, object],
    ) -> WorkspaceManifest | None:
        if summary.get("runnable") is not True:
            return None
        source = failed_run / "versions" / "v8" / "source"
        manifest_path = failed_run / "versions" / "v8" / "manifest.json"
        if not source.is_dir() or not manifest_path.is_file():
            raise ValueError("recovery runnable v8 source is missing")
        manifest = self._read_workspace_manifest(
            manifest_path,
            "v8 source manifest",
        )
        actual = self.snapshotter.capture(source)
        if (
            actual.content_hash != manifest.content_hash
            or actual.files != manifest.files
        ):
            raise ValueError(
                "recovery v8 source does not match its manifest"
            )
        return manifest

    def _recover_frozen_v8(
        self,
        failed_run: Path,
        failed_summary: Mapping[str, object],
        manifest: WorkspaceManifest,
    ) -> Round8PipelineResult:
        lineage = load_round8_parent(
            self.parent_run_dir,
            self.expected_parent_hash,
        )
        run = Run.start(
            game="28_generals",
            agent="generals-hl",
            run_type="rule_iter",
            data_dir=str(self.data_dir),
            config={
                "benchmark_id": self.config.benchmark_id,
                "challenge_id": self.challenge.challenge_id,
                "provider": self.provider.provider_name,
                "budget_phase": "learning",
                "engine_hash": self.assets.engine_hash,
                "round": 8,
                "failed_run_id": failed_run.name,
                "recovery_mode": "frozen_v8",
            },
        )
        run_dir = Path(run.run_dir)
        imported_v7 = import_round8_source(
            lineage,
            run_dir,
            self.snapshotter,
        )
        source = failed_run / "versions" / "v8" / "source"
        frozen_v8 = run_dir / "versions" / "v8" / "source"
        self.snapshotter.materialize_manifest(
            source,
            frozen_v8,
            manifest,
        )
        self.snapshotter.write_manifest(
            manifest,
            run_dir / "versions" / "v8" / "manifest.json",
        )
        self.snapshotter.write_unified_patch(
            run_dir / "versions" / "v7" / "source",
            frozen_v8,
            run_dir / "versions" / "v7-to-v8.patch",
        )
        for directory in (
            "benchmark",
            "provider",
            "learning-evidence",
        ):
            source_dir = failed_run / directory
            target_dir = run_dir / directory
            if source_dir.is_dir():
                shutil.copytree(
                    source_dir,
                    target_dir,
                    dirs_exist_ok=True,
                )
        run.write(
            "recovery_import",
            failed_run_id=failed_run.name,
            recovery_mode="frozen_v8",
            reused_provider_act=True,
            new_provider_act=False,
            source_manifest_hash=manifest.content_hash,
            source_learning_budget=failed_summary.get(
                "local_learning_budget"
            ),
        )
        run.write(
            "lineage_import",
            parent_run_id=lineage.parent_run_id,
            parent_version="v7",
            starting_version="v7",
            version="v7",
            manifest_hash=imported_v7.content_hash,
            prior_score_history=list(lineage.prior_score_history),
            global_coding_agent_act=9,
        )
        run.write(
            "version",
            version="v8",
            version_before="v7",
            status="recovered_frozen",
            manifest_hash=manifest.content_hash,
            changed_files=manifest.changed_files,
            runnable=True,
        )

        learning_cases = build_round8_learning_cases(
            self.config,
            self.challenge,
        )
        validation_cases = build_round8_validation_cases(
            self.config,
            self.challenge,
        )
        formal_cases = tuple(
            build_evaluation_spec(self.config).cases
        )
        sealed_cases = build_round8_sealed_cases(
            self.config,
            self.challenge,
        )
        evaluator = self.evaluator or self._production_evaluator(run_dir)
        validation: GeneralsEvaluation | None = None
        formal: GeneralsEvaluation | None = None
        sealed: GeneralsEvaluation | None = None
        validation_error: str | None = None
        formal_error: str | None = None
        sealed_error: str | None = None
        formal_attempted = False
        validation_passed = False
        performance_target_met = False
        champion_claim = False
        sealed_status = "not_opened"
        status = "failed"
        runnable = True
        evo_score_8: float | None = None
        gain_8: float | None = None
        validation_payload: dict[str, object] = {
            "status": "not_run",
            "passed": False,
            "score": None,
        }
        sealed_payload: dict[str, object] = {
            "status": "not_opened",
            "champion_claim": False,
            "score": None,
            "reason": "validation_not_run",
        }
        action_profiles: dict[str, object | None] = {
            "learning": (
                (failed_summary.get("action_profiles") or {}).get(
                    "learning"
                )
                if isinstance(
                    failed_summary.get("action_profiles"),
                    Mapping,
                )
                else None
            ),
            "validation": None,
            "formal": None,
            "sealed": None,
        }
        try:
            test_workspace = self._materialize_verified_source(
                frozen_v8,
                run_dir / "isolated-workspaces/v8-recovery-tests",
                manifest,
            )
            tests_ok, test_output = self._run_candidate_tests(
                test_workspace
            )
            tests_path = run_dir / "versions" / "v8" / "tests.log"
            tests_path.write_text(test_output, encoding="utf-8")
            if not tests_ok:
                runnable = False
                raise _Round8Abort(
                    "invalid_version",
                    "recovered frozen v8 candidate tests failed",
                )

            validation_workspace = self._materialize_verified_source(
                frozen_v8,
                run_dir / "isolated-workspaces/v8-validation",
                manifest,
            )
            self._verify_runtime_source_v8(
                run,
                validation_workspace,
                manifest,
                phase="validation",
            )
            try:
                validation = evaluator.evaluate(
                    validation_workspace,
                    "v8",
                    "validation",
                    run,
                    cases=validation_cases,
                )
                action_profiles["validation"] = (
                    self._persist_action_profile_v8(
                        run,
                        validation,
                        phase="validation",
                        artifact_path=(
                            run_dir
                            / "diagnostics"
                            / "v8-validation-action-profile.json"
                        ),
                        required=False,
                    )
                )
                validation_gate = evaluate_champion_gate(
                    results=validation.results,
                    cases=validation_cases,
                    minimum_wins=self.challenge.validation_min_wins,
                    minimum_wins_per_seat=(
                        self.challenge.validation_min_wins_per_seat
                    ),
                )
                validation_passed = validation_gate.passed
                validation_payload = self._gate_payload(
                    validation_gate,
                    source_hash=manifest.content_hash,
                    engine_hash=self.assets.engine_hash,
                    minimum_wins=self.challenge.validation_min_wins,
                    minimum_wins_per_seat=(
                        self.challenge.validation_min_wins_per_seat
                    ),
                    suite_digest=self._suite_digest(validation_cases),
                    claim_field="sealed_suite_opened",
                )
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
                validation_payload = {
                    "status": "error",
                    "passed": False,
                    "sealed_suite_opened": False,
                    "score": None,
                    "reason": "validation_evaluation_error",
                    "error": validation_error,
                    "source_hash": manifest.content_hash,
                    "engine_hash": self.assets.engine_hash,
                    "suite_digest": self._suite_digest(validation_cases),
                }
            self._write_json(
                run_dir / "validation-gate.json",
                validation_payload,
            )
            run.write("champion_validation_gate", **validation_payload)

            try:
                formal_workspace = self._materialize_verified_source(
                    frozen_v8,
                    run_dir / "isolated-workspaces/v8-formal",
                    manifest,
                )
                self._verify_runtime_source_v8(
                    run,
                    formal_workspace,
                    manifest,
                    phase="formal",
                )
                formal_attempted = True
                formal = evaluator.evaluate(
                    formal_workspace,
                    "v8",
                    "formal",
                    run,
                    cases=formal_cases,
                )
                action_profiles["formal"] = (
                    self._persist_action_profile_v8(
                        run,
                        formal,
                        phase="formal",
                        artifact_path=(
                            run_dir
                            / "diagnostics"
                            / "v8-formal-action-profile.json"
                        ),
                        required=False,
                    )
                )
                evo_score_8 = formal.score
                gain_8 = (
                    evo_score_8 - lineage.raw_score
                    if evo_score_8 is not None
                    else None
                )
                performance_target_met = self._performance_target_met(
                    formal,
                    formal_cases,
                    self.challenge,
                )
                run.write(
                    "evaluation",
                    phase="evolved_8",
                    version="v8",
                    status=formal.status,
                    score=formal.score,
                    gain=gain_8,
                    global_coding_agent_act=9,
                    per_tier=dict(formal.per_tier),
                    seat_gap=formal.seat_gap,
                )
            except Exception as exc:
                formal_error = f"{type(exc).__name__}: {exc}"

            if self.challenge.sealed_seeds and validation_passed:
                try:
                    sealed_workspace = self._materialize_verified_source(
                        frozen_v8,
                        run_dir / "isolated-workspaces/v8-sealed",
                        manifest,
                    )
                    self._verify_runtime_source_v8(
                        run,
                        sealed_workspace,
                        manifest,
                        phase="sealed",
                    )
                    sealed = evaluator.evaluate(
                        sealed_workspace,
                        "v8",
                        "sealed",
                        run,
                        cases=sealed_cases,
                    )
                    action_profiles["sealed"] = (
                        self._persist_action_profile_v8(
                            run,
                            sealed,
                            phase="sealed",
                            artifact_path=(
                                run_dir
                                / "diagnostics"
                                / "v8-sealed-action-profile.json"
                            ),
                            required=False,
                        )
                    )
                    sealed_gate = evaluate_champion_gate(
                        results=sealed.results,
                        cases=sealed_cases,
                        minimum_wins=self.challenge.sealed_min_wins,
                        minimum_wins_per_seat=(
                            self.challenge.sealed_min_wins_per_seat
                        ),
                    )
                    sealed_status = sealed_gate.status
                    champion_claim = sealed_gate.passed
                    sealed_payload = self._gate_payload(
                        sealed_gate,
                        source_hash=manifest.content_hash,
                        engine_hash=self.assets.engine_hash,
                        minimum_wins=self.challenge.sealed_min_wins,
                        minimum_wins_per_seat=(
                            self.challenge.sealed_min_wins_per_seat
                        ),
                        suite_digest=self._suite_digest(sealed_cases),
                        claim_field="champion_claim",
                    )
                except Exception as exc:
                    sealed_error = f"{type(exc).__name__}: {exc}"
                    sealed_status = "error"
                    sealed_payload = {
                        "status": "error",
                        "score": None,
                        "champion_claim": False,
                        "error": sealed_error,
                    }
            else:
                sealed_payload = {
                    "status": "not_opened",
                    "score": None,
                    "champion_claim": False,
                    "reason": (
                        "validation_evaluation_error"
                        if validation_error is not None
                        else "validation_gate_failed"
                    ),
                }
            self._write_json(
                run_dir / "sealed-claim.json",
                sealed_payload,
            )
            run.write("champion_sealed_claim", **sealed_payload)
            if formal is None:
                status = "formal_failed"
            elif validation_error is not None:
                status = "validation_incomplete"
            elif sealed_status in {"error", "invalid", "incomplete"}:
                status = "sealed_incomplete"
            else:
                status = "complete"
        except _Round8Abort as exc:
            status = exc.status
            run.write(
                "pipeline_error",
                status=status,
                error=str(exc),
            )
        finally:
            run.writer.flush()
            quality = inspect_event_file(
                run_dir / "events.jsonl"
            ).to_dict()
            self._write_json(run_dir / "quality.json", quality)
            local_budget = {
                key: value
                for key, value in run.budget_snapshot().items()
                if key.startswith("learning_")
            }
            source_learning_budget = failed_summary.get(
                "local_learning_budget"
            )
            inherited_cumulative = failed_summary.get(
                "cumulative_learning_budget"
            )
            formal_scores = {
                tier: (
                    formal.per_tier.get(tier)
                    if formal is not None
                    else None
                )
                for tier in ("high", "medium", "low")
            }
            run.finish({
                "total_episodes": None,
                "total_steps": None,
                "total_reward": None,
                "win_rate": None,
                "wins": None,
                "losses": None,
                "draws": None,
                "status": status,
                "benchmark_id": self.config.benchmark_id,
                "challenge_id": self.challenge.challenge_id,
                "raw_score": lineage.raw_score,
                "evo_score": evo_score_8,
                "evo_score_1": lineage.evo_score_1,
                "evo_score_2": lineage.evo_score_2,
                "evo_score_3": lineage.evo_score_3,
                "evo_score_4": lineage.evo_score_4,
                "evo_score_5": lineage.evo_score_5,
                "evo_score_6": lineage.evo_score_6,
                "evo_score_8": evo_score_8,
                "gain": gain_8,
                "gain_8": gain_8,
                "benchmark_score": evo_score_8,
                "formal_score": (
                    formal.score if formal is not None else None
                ),
                "formal_scores": formal_scores,
                "evaluation_status": (
                    formal.status
                    if formal is not None
                    else "error"
                    if formal_attempted
                    else "not_run"
                ),
                "formal_attempted": formal_attempted,
                "champion_validation": validation_payload,
                "champion_sealed": sealed_payload,
                "champion_claim": champion_claim,
                "validation_passed": validation_passed,
                "performance_target_met": performance_target_met,
                "sealed_status": sealed_status,
                "score_history": [
                    *lineage.prior_score_history,
                    evo_score_8,
                ],
                "act_count": 8,
                "round_act_count": 0,
                "parent_run_id": lineage.parent_run_id,
                "parent_version": "v7",
                "starting_version": "v7",
                "parent_content_hash": lineage.v7_manifest.content_hash,
                "failed_run_id": failed_run.name,
                "recovery_mode": "frozen_v8",
                "source_learning_budget": source_learning_budget,
                "runnable": runnable,
                "action_profiles": action_profiles,
                "validation_error": validation_error,
                "formal_error": formal_error,
                "sealed_error": sealed_error,
                "local_learning_budget": local_budget,
                "cumulative_learning_budget": inherited_cumulative,
                "learning_results": failed_summary.get(
                    "learning_results",
                    [],
                ),
                "validation_results": self._result_rows((validation,)),
                "formal_results": self._result_rows((formal,)),
                "sealed_results": self._result_rows((sealed,)),
                "benchmark_results": self._result_rows((formal,)),
            })
        return Round8PipelineResult(
            run_dir=run_dir,
            status=status,
            runnable=runnable,
            raw_score=lineage.raw_score,
            evo_score_8=evo_score_8,
            gain_8=gain_8,
            validation_passed=validation_passed,
            formal_attempted=formal_attempted,
            performance_target_met=performance_target_met,
            sealed_status=sealed_status,
            champion_claim=champion_claim,
            global_act_count=9,
            round_act_count=0,
        )

    def recover(self, failed_run_dir: Path) -> Round8PipelineResult:
        failed_run, summary = self._load_recovery_summary(
            failed_run_dir
        )
        manifest = self._load_frozen_recovery_source(
            failed_run,
            summary,
        )
        if manifest is not None:
            return self._recover_frozen_v8(
                failed_run,
                summary,
                manifest,
            )
        local_learning_budget = summary.get("local_learning_budget")
        if not isinstance(local_learning_budget, Mapping):
            local_learning_budget = {}
        self._fresh_recovery = {
            "run_id": failed_run.name,
            "local_learning_budget": dict(local_learning_budget),
        }
        try:
            return self.run()
        finally:
            del self._fresh_recovery
