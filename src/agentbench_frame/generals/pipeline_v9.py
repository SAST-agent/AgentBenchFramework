"""Single-act attribution-guided v9 Generals iteration and recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from agentbench_frame.tracking.quality import inspect_event_file
from agentbench_frame.tracking.run import Run
from agentbench_frame.tracking.snapshot import WorkspaceManifest

from .challenge_v8 import evaluate_champion_gate
from .challenge_v9 import build_round9_validation_cases
from .evaluator import GeneralsEvaluation, build_evaluation_spec
from .lineage_v9 import Round9Lineage, import_round9_parent, load_round9_lineage
from .measurement import ProbeState
from .models import AssetLayout, PilotConfig, ReplaySkillAsset, Round9ChallengeConfig
from .pipeline import GeneralsHLPipeline
from .pipeline_v8 import GeneralsHLRound8Pipeline
from .prompt import PromptBuildResult
from .prompt_v9 import build_round9_prompt


_V9_FORBIDDEN_RUNTIME = re.compile(
    r"\b(?:random|socket|requests|urllib|subprocess|replay_id|opponent_id|"
    r"seed|state_id|measurement_state_id)\b|time\.time|time\.sleep|"
    r"os\.environ|__file__"
)
_EDITABLE = frozenset({"strategy.py", "state_view.py", "STRATEGY.md", "EXPERIENCE.md"})
_REQUIRED = frozenset({"main.py", "strategy.py", "state_view.py", "STRATEGY.md", "EXPERIENCE.md"})


@dataclass(frozen=True)
class Round9PipelineResult:
    run_dir: Path
    status: str
    runnable: bool
    raw_score: float | None
    evo_score_9: float | None
    gain_9: float | None
    validation_attempted: bool
    validation_passed: bool
    formal_attempted: bool
    performance_target_met: bool
    champion_claim: bool
    global_act_count: int
    round_act_count: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


class GeneralsHLRound9Pipeline(GeneralsHLRound8Pipeline):
    """Use v7 plus attribution once, then always evaluate validation and formal."""

    def __init__(
        self,
        *,
        config: PilotConfig,
        challenge: Round9ChallengeConfig,
        assets: AssetLayout,
        replay_skill: ReplaySkillAsset,
        policy_parent_run_dir: Path,
        iteration_predecessor_run_dir: Path,
        attribution_run_dir: Path,
        data_dir: Path,
        provider: Any,
        evaluator: Any | None = None,
        rules_path: Path | None = None,
        prompt_max_bytes: int = 196_608,
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
        self.policy_parent_run_dir = Path(policy_parent_run_dir)
        self.iteration_predecessor_run_dir = Path(iteration_predecessor_run_dir)
        self.attribution_run_dir = Path(attribution_run_dir)
        self.prompt_max_bytes = int(prompt_max_bytes)

    def _lineage(self) -> Round9Lineage:
        lineage = load_round9_lineage(
            self.policy_parent_run_dir,
            self.iteration_predecessor_run_dir,
        )
        if (
            lineage.policy_parent_run_id != self.challenge.parent_run_id
            or lineage.policy_parent_hash != self.challenge.parent_content_hash
            or lineage.iteration_predecessor_run_id != self.challenge.predecessor_run_id
            or lineage.iteration_predecessor_hash != self.challenge.predecessor_content_hash
        ):
            raise ValueError("round-9 lineage does not match frozen challenge")
        return lineage

    def _start_run(self, *, recovery: str | None = None) -> Run:
        return Run.start(
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
                "round": 9,
                "lineage_mode": "v7_attribution_to_v9",
                **({"recovery_mode": "frozen_candidate", "recovered_from": recovery} if recovery else {}),
            },
        )

    @staticmethod
    def _source_scope_valid(changed_files: list[str]) -> bool:
        return all(
            item in _EDITABLE or item.startswith("tests/") or item.startswith("policy/")
            for item in changed_files
        )

    @staticmethod
    def _runtime_valid(source: Path) -> bool:
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
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return False
            if _V9_FORBIDDEN_RUNTIME.search(text):
                return False
        return total <= 10_485_760

    @staticmethod
    def _required_valid(source: Path, manifest: WorkspaceManifest) -> bool:
        return all(
            relative in manifest.files
            and (source / relative).is_file()
            and not (source / relative).is_symlink()
            for relative in _REQUIRED
        )

    @staticmethod
    def _candidate_probes() -> tuple[ProbeState, ...]:
        def state(seat: int, contact: bool) -> dict[str, Any]:
            cells = {
                "0,0": {"type": 0, "player": seat, "army": 30, "general_id": 0},
                "0,1": {"type": 0, "player": -1, "army": 1, "general_id": None},
                "0,3": {"type": 0, "player": 1 - seat, "army": 20, "general_id": 1},
            }
            if contact:
                cells["1,1"] = {"type": 0, "player": seat, "army": 7, "general_id": None}
                cells["1,2"] = {"type": 0, "player": 1 - seat, "army": 2, "general_id": None}
            return {
                "round": 80 if contact else 1,
                "my_seat": seat,
                "coins": [40, 40],
                "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
                "cells": cells,
                "generals": {
                    "0": {"id": 0, "type": "main", "player": seat, "position": [0, 0], "produce_level": 1},
                    "1": {"id": 1, "type": "main", "player": 1 - seat, "position": [0, 3], "produce_level": 1},
                },
            }
        return (
            ProbeState("v9-static-p0", state(0, False), ((8,),)),
            ProbeState("v9-contact-p1", state(1, True), ((8,),)),
        )

    def _policy_hashes(self) -> dict[str, str]:
        report = _read_object(
            self.attribution_run_dir / "diagnosis/report.json",
            "attribution report",
        )
        policies = report.get("policies")
        if not isinstance(policies, list):
            raise ValueError("attribution report policies are missing")
        result = {
            str(item["cell"]): str(item["content_hash"])
            for item in policies
            if isinstance(item, Mapping)
        }
        if set(result) != {"A", "B", "C", "D"}:
            raise ValueError("attribution policy hash set changed")
        return result

    def _build_prompt(
        self,
        lineage: Round9Lineage,
        workspace: Path,
    ) -> PromptBuildResult:
        rules_text = self.rules_path.read_text(encoding="utf-8")
        return build_round9_prompt(
            benchmark_id=self.config.benchmark_id,
            attribution_run_dir=self.attribution_run_dir,
            policy_parent_hash=lineage.policy_parent_hash,
            iteration_predecessor_hash=lineage.iteration_predecessor_hash,
            expected_policy_hashes=self._policy_hashes(),
            v7_strategy=(workspace / "strategy.py").read_text(encoding="utf-8"),
            v7_experience=(workspace / "EXPERIENCE.md").read_text(encoding="utf-8"),
            rules_text=rules_text,
            rules_sha256=hashlib.sha256(rules_text.encode("utf-8")).hexdigest(),
            replay_skill_text=self.replay_skill.text,
            replay_skill_sha256=self.replay_skill.sha256,
            max_bytes=self.prompt_max_bytes,
        )

    def _write_specs(self, run_dir: Path) -> tuple[tuple, tuple]:
        validation = build_round9_validation_cases(self.config, self.challenge)
        formal = tuple(build_evaluation_spec(self.config).cases)
        self._write_json(run_dir / "benchmark/v9-validation-spec.json", {
            "challenge_id": self.challenge.challenge_id,
            "engine_hash": self.assets.engine_hash,
            "cases": [asdict(case) for case in validation],
        })
        self._write_json(run_dir / "benchmark/formal-spec.json", {
            "benchmark_id": self.config.benchmark_id,
            "engine_hash": self.assets.engine_hash,
            "evaluation_cases": [asdict(case) for case in formal],
        })
        return validation, formal

    @staticmethod
    def _formal_gate(
        evaluation: GeneralsEvaluation | None,
        cases: tuple,
        challenge: Round9ChallengeConfig,
    ) -> tuple[bool, dict[str, Any]]:
        case_by_id = {case.case_id: case for case in cases}
        if evaluation is None or evaluation.status != "complete":
            return False, {"status": "incomplete", "passed": False}
        valid = [item for item in evaluation.results if item.valid and item.case_id in case_by_id]
        wins = [item for item in valid if item.outcome == "win"]
        high = [
            item for item in wins
            if case_by_id[item.case_id].metadata.get("tier") == "high"
        ]
        high_by_seat = {
            seat: sum(case_by_id[item.case_id].first_player == seat for item in high)
            for seat in challenge.seats
        }
        passed = (
            len(valid) == len(cases)
            and len(wins) >= challenge.formal_total_min_wins
            and len(high) >= challenge.formal_high_min_wins
            and all(
                high_by_seat[seat] >= challenge.formal_high_min_wins_per_seat
                for seat in challenge.seats
            )
        )
        return passed, {
            "status": "complete",
            "passed": passed,
            "valid_games": len(valid),
            "wins": len(wins),
            "high_wins": len(high),
            "high_wins_by_seat": high_by_seat,
            "minimum_total_wins": challenge.formal_total_min_wins,
            "minimum_high_wins": challenge.formal_high_min_wins,
            "minimum_high_wins_per_seat": challenge.formal_high_min_wins_per_seat,
        }

    def _candidate_validation(
        self,
        run: Run,
        run_dir: Path,
        frozen_source: Path,
        manifest: WorkspaceManifest,
        parent: WorkspaceManifest,
        provider_completed: bool,
    ) -> tuple[bool, dict[str, Any]]:
        scope = self._source_scope_valid(manifest.changed_files)
        manifest_ok = self._frozen_manifest_matches(frozen_source, manifest)
        required = self._required_valid(frozen_source, manifest)
        main_unchanged = (
            "main.py" not in manifest.changed_files
            and manifest.files.get("main.py") == parent.files.get("main.py")
        )
        runtime = self._runtime_valid(frozen_source)
        documents = False
        if required:
            try:
                documents = self._strategy_documents_valid(frozen_source)
            except OSError:
                documents = False
        tests_ok = False
        test_output = "candidate tests skipped\n"
        probes = None
        if all((provider_completed, scope, manifest_ok, required, main_unchanged, runtime, documents)):
            try:
                tests_workspace = self._materialize_verified_source(
                    frozen_source, run_dir / "isolated-workspaces/v9-tests", manifest
                )
                tests_ok, test_output = self._run_candidate_tests(tests_workspace)
                if tests_ok:
                    probe_workspace = self._materialize_verified_source(
                        frozen_source, run_dir / "isolated-workspaces/v9-probes", manifest
                    )
                    probes = self._verify_policy_probe_contract(
                        probe_workspace, self._candidate_probes()
                    )
            except Exception as exc:
                test_output += f"\nFramework probe failure: {type(exc).__name__}: {exc}\n"
                tests_ok = False
                probes = None
        tests_path = run_dir / "versions/v9/tests.log"
        tests_path.write_text(test_output, encoding="utf-8")
        checks = {
            "provider_completed": provider_completed,
            "scope_valid": scope,
            "manifest_valid": manifest_ok,
            "required_files_valid": required,
            "main_unchanged": main_unchanged,
            "runtime_source_valid": runtime,
            "strategy_documents_valid": documents,
            "tests_passed": tests_ok,
            "probe_diagnostics": probes,
        }
        return all(value for key, value in checks.items() if key != "probe_diagnostics") and probes is not None, checks

    def _evaluate(
        self,
        *,
        run: Run,
        run_dir: Path,
        source: Path,
        manifest: WorkspaceManifest,
        validation_cases: tuple,
        formal_cases: tuple,
    ) -> tuple[
        GeneralsEvaluation | None,
        GeneralsEvaluation | None,
        bool,
        bool,
        bool,
        dict[str, Any],
        dict[str, Any],
        str | None,
    ]:
        evaluator = self.evaluator or self._production_evaluator(run_dir)
        validation = None
        formal = None
        validation_passed = False
        validation_payload: dict[str, Any] = {"status": "error", "passed": False}
        formal_payload: dict[str, Any] = {"status": "error", "passed": False}
        error = None
        try:
            workspace = self._materialize_verified_source(
                source, run_dir / "isolated-workspaces/v9-validation", manifest
            )
            validation = evaluator.evaluate(
                workspace, "v9", "validation9", run, cases=validation_cases
            )
            gate = evaluate_champion_gate(
                results=validation.results,
                cases=validation_cases,
                minimum_wins=self.challenge.validation_min_wins,
                minimum_wins_per_seat=self.challenge.validation_min_wins_per_seat,
            )
            validation_passed = gate.passed
            validation_payload = asdict(gate)
        except Exception as exc:
            error = f"validation9: {type(exc).__name__}: {exc}"
            run.write("pipeline_error", phase="validation9", error=error)
        run.write("champion_validation_gate", **validation_payload)

        formal_attempted = True
        try:
            workspace = self._materialize_verified_source(
                source, run_dir / "isolated-workspaces/v9-formal", manifest
            )
            formal = evaluator.evaluate(
                workspace, "v9", "formal9", run, cases=formal_cases
            )
            performance, formal_payload = self._formal_gate(
                formal, formal_cases, self.challenge
            )
        except Exception as exc:
            performance = False
            formal_error = f"formal9: {type(exc).__name__}: {exc}"
            error = f"{error}; {formal_error}" if error else formal_error
            run.write("pipeline_error", phase="formal9", error=formal_error)
        champion = validation_passed and performance
        run.write("champion_sealed_claim", **formal_payload, champion_claim=champion)
        return (
            validation,
            formal,
            validation_passed,
            formal_attempted,
            performance,
            validation_payload,
            formal_payload,
            error,
        )

    def _finish(
        self,
        *,
        run: Run,
        lineage: Round9Lineage,
        status: str,
        runnable: bool,
        manifest: WorkspaceManifest | None,
        validation: GeneralsEvaluation | None,
        formal: GeneralsEvaluation | None,
        validation_attempted: bool,
        validation_passed: bool,
        formal_attempted: bool,
        performance: bool,
        champion: bool,
        round_act_count: int,
        error: str | None,
        prompt: PromptBuildResult | None = None,
        raw_hash: str | None = None,
        recovered_from: str | None = None,
    ) -> Round9PipelineResult:
        run_dir = Path(run.run_dir)
        raw_score = lineage.score_history[0]
        evo = formal.score if formal is not None else None
        gain = evo - raw_score if evo is not None and raw_score is not None else None
        champion_payload = {
            "version": "v9" if champion else "v7",
            "content_hash": manifest.content_hash if champion and manifest else lineage.policy_parent_hash,
            "claim": champion,
        }
        self._write_json(run_dir / "champion.json", champion_payload)
        run.writer.flush()
        quality = inspect_event_file(run_dir / "events.jsonl").to_dict()
        self._write_json(run_dir / "quality.json", quality)
        events_hash = _sha256(run_dir / "events.jsonl")
        score_history = [*lineage.score_history, evo]
        summary = {
            "status": status,
            "benchmark_id": self.config.benchmark_id,
            "challenge_id": self.challenge.challenge_id,
            "raw_score": raw_score,
            "evo_score": evo,
            "evo_score_9": evo,
            "gain": gain,
            "gain_9": gain,
            "benchmark_score": evo,
            "formal_score": evo,
            "evaluation_status": formal.status if formal is not None else "error" if formal_attempted else "not_run",
            "validation_attempted": validation_attempted,
            "validation_passed": validation_passed,
            "formal_attempted": formal_attempted,
            "performance_target_met": performance,
            "champion_claim": champion,
            "champion": champion_payload,
            "runnable": runnable,
            "act_count": lineage.next_global_act_count,
            "round_act_count": round_act_count,
            "policy_parent_run_id": lineage.policy_parent_run_id,
            "policy_parent_version": "v7",
            "policy_parent_hash": lineage.policy_parent_hash,
            "iteration_predecessor_run_id": lineage.iteration_predecessor_run_id,
            "iteration_predecessor_version": "v8",
            "iteration_predecessor_hash": lineage.iteration_predecessor_hash,
            "score_history": score_history,
            "AUC_coding_agent_act": None,
            "AUC_episode": None,
            "AUC_env_step": None,
            "AUC_token": None,
            "AUC_time": None,
            "auc_status": "unavailable_missing_score_point",
            "candidate_hash": manifest.content_hash if manifest else None,
            "events_sha256": events_hash,
            "prompt_sha256": prompt.manifest.get("prompt_sha256") if prompt else None,
            "diagnosis_report_hash": prompt.manifest.get("diagnosis_report_sha256") if prompt else None,
            "diagnosis_evidence_hash": prompt.manifest.get("diagnosis_evidence_sha256") if prompt else None,
            "provider_raw_sha256": raw_hash,
            "validation_results": self._result_rows((validation,)),
            "formal_results": self._result_rows((formal,)),
            "benchmark_results": self._result_rows((formal,)),
            "error": error,
            "recovery_mode": "frozen_candidate" if recovered_from else None,
            "recovered_from_run_id": recovered_from,
        }
        run.finish(summary)
        return Round9PipelineResult(
            run_dir=run_dir,
            status=status,
            runnable=runnable,
            raw_score=raw_score,
            evo_score_9=evo,
            gain_9=gain,
            validation_attempted=validation_attempted,
            validation_passed=validation_passed,
            formal_attempted=formal_attempted,
            performance_target_met=performance,
            champion_claim=champion,
            global_act_count=lineage.next_global_act_count,
            round_act_count=round_act_count,
        )

    def run(self) -> Round9PipelineResult:
        lineage = self._lineage()
        run = self._start_run()
        run_dir = Path(run.run_dir)
        validation_cases, formal_cases = self._write_specs(run_dir)
        manifest = None
        prompt = None
        raw_hash = None
        runnable = False
        status = "failed"
        error = None
        validation = formal = None
        validation_attempted = validation_passed = formal_attempted = performance = champion = False
        round_act_count = 0
        try:
            parent = import_round9_parent(lineage, run_dir, self.snapshotter)
            workspace = run_dir / "workspace"
            if (workspace / "versions/v8/source").exists():
                raise ValueError("v8 source leaked into v9 workspace")
            prompt = self._build_prompt(lineage, workspace)
            provider_dir = run_dir / "provider"
            provider_dir.mkdir(parents=True)
            prompt_path = provider_dir / "codex-act-v9.prompt.md"
            prompt_path.write_text(prompt.prompt, encoding="utf-8")
            self._write_json(provider_dir / "prompt-manifest.json", dict(prompt.manifest))
            raw_path = provider_dir / "codex-act-v9.raw.jsonl"
            stderr_path = provider_dir / "codex-act-v9.stderr.log"
            raw_path.touch()
            stderr_path.touch()
            controller = run.create_coding_agent_controller(
                self.provider, snapshotter=self.snapshotter, budget_phase="learning"
            )
            act = controller.run_act(
                {
                    "prompt": prompt.prompt,
                    "workspace_root": str(workspace),
                    "raw_output_path": str(raw_path),
                    "stderr_output_path": str(stderr_path),
                    "max_artifact_bytes": self.config.limits.max_artifact_bytes,
                },
                workspace_root=str(workspace),
                version_before="v7",
                previous_manifest=parent,
            )
            round_act_count = 1
            raw_hash = _sha256(raw_path)
            manifest = self._write_version(run_dir, "v9", workspace, previous=parent)
            frozen = run_dir / "versions/v9/source"
            self.snapshotter.write_unified_patch(
                run_dir / "versions/v7/source", frozen, run_dir / "versions/v7-to-v9.patch"
            )
            runnable, checks = self._candidate_validation(
                run, run_dir, frozen, manifest, parent, act.status == "completed"
            )
            run.write(
                "version",
                version="v9",
                version_before="v7",
                status="available" if runnable else "invalid",
                manifest_hash=manifest.content_hash,
                changed_files=manifest.changed_files,
                runnable=runnable,
                **checks,
            )
            if not runnable:
                status = "invalid_version"
            else:
                validation_attempted = True
                (
                    validation,
                    formal,
                    validation_passed,
                    formal_attempted,
                    performance,
                    validation_payload,
                    formal_payload,
                    error,
                ) = self._evaluate(
                    run=run,
                    run_dir=run_dir,
                    source=frozen,
                    manifest=manifest,
                    validation_cases=validation_cases,
                    formal_cases=formal_cases,
                )
                champion = validation_passed and performance
                self._write_json(run_dir / "validation-gate.json", validation_payload)
                self._write_json(run_dir / "formal-gate.json", formal_payload)
                if formal is None:
                    status = "formal_failed"
                elif formal.status != "complete":
                    status = "formal_incomplete"
                else:
                    status = "complete"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            run.write("pipeline_error", status=status, error=error)
        return self._finish(
            run=run,
            lineage=lineage,
            status=status,
            runnable=runnable,
            manifest=manifest,
            validation=validation,
            formal=formal,
            validation_attempted=validation_attempted,
            validation_passed=validation_passed,
            formal_attempted=formal_attempted,
            performance=performance,
            champion=champion,
            round_act_count=round_act_count,
            error=error,
            prompt=prompt,
            raw_hash=raw_hash,
        )

    def _validate_recovery(self, failed_run_dir: Path, lineage: Round9Lineage) -> tuple[Path, dict[str, Any], WorkspaceManifest]:
        failed = Path(failed_run_dir).resolve()
        summary = _read_object(failed / "summary.json", "failed v9 summary")
        if summary.get("run_id") != failed.name:
            raise ValueError("failed v9 run identity changed")
        if (
            summary.get("runnable") is not True
            or summary.get("round_act_count") != 1
            or summary.get("act_count") != lineage.next_global_act_count
            or summary.get("policy_parent_hash") != lineage.policy_parent_hash
            or summary.get("iteration_predecessor_hash") != lineage.iteration_predecessor_hash
        ):
            raise ValueError("failed v9 recovery authority is invalid")
        prompt_path = failed / "provider/codex-act-v9.prompt.md"
        prompt_manifest = _read_object(failed / "provider/prompt-manifest.json", "v9 prompt manifest")
        if _sha256(prompt_path) != prompt_manifest.get("prompt_sha256") or summary.get("prompt_sha256") != prompt_manifest.get("prompt_sha256"):
            raise ValueError("recovery prompt digest changed")
        raw_path = failed / "provider/codex-act-v9.raw.jsonl"
        if not raw_path.is_file() or not raw_path.read_bytes() or _sha256(raw_path) != summary.get("provider_raw_sha256"):
            raise ValueError("recovery provider raw output changed")
        if _sha256(failed / "events.jsonl") != summary.get("events_sha256"):
            raise ValueError("recovery event stream changed")
        events = [json.loads(line) for line in (failed / "events.jsonl").read_text().splitlines()]
        acts = {item.get("act_id") for item in events if item.get("event_type") == "coding_agent_act"}
        if len(acts) != 1:
            raise ValueError("recovery provider act count changed")
        manifest_raw = _read_object(failed / "versions/v9/manifest.json", "v9 manifest")
        manifest = WorkspaceManifest(
            str(manifest_raw["content_hash"]),
            {str(k): str(v) for k, v in manifest_raw["files"].items()},
            [str(item) for item in manifest_raw.get("changed_files", [])],
        )
        actual = self.snapshotter.capture(failed / "versions/v9/source")
        if actual.content_hash != manifest.content_hash or actual.files != manifest.files or summary.get("candidate_hash") != manifest.content_hash:
            raise ValueError("recovery v9 candidate manifest changed")
        return failed, summary, manifest

    def recover(self, failed_run_dir: Path) -> Round9PipelineResult:
        lineage = self._lineage()
        failed, failed_summary, manifest = self._validate_recovery(failed_run_dir, lineage)
        run = self._start_run(recovery=failed.name)
        run_dir = Path(run.run_dir)
        validation_cases, formal_cases = self._write_specs(run_dir)
        parent = import_round9_parent(lineage, run_dir, self.snapshotter)
        del parent
        target = run_dir / "versions/v9/source"
        self.snapshotter.materialize_manifest(failed / "versions/v9/source", target, manifest)
        self.snapshotter.write_manifest(manifest, run_dir / "versions/v9/manifest.json")
        run.write(
            "recovery_import",
            failed_run_id=failed.name,
            version="v9",
            manifest_hash=manifest.content_hash,
            provider_invoked=False,
        )
        validation = formal = None
        validation_passed = performance = champion = False
        error = None
        (
            validation,
            formal,
            validation_passed,
            formal_attempted,
            performance,
            validation_payload,
            formal_payload,
            error,
        ) = self._evaluate(
            run=run,
            run_dir=run_dir,
            source=target,
            manifest=manifest,
            validation_cases=validation_cases,
            formal_cases=formal_cases,
        )
        champion = validation_passed and performance
        self._write_json(run_dir / "validation-gate.json", validation_payload)
        self._write_json(run_dir / "formal-gate.json", formal_payload)
        status = "complete" if formal is not None and formal.status == "complete" else "formal_failed"
        prompt_manifest = _read_object(failed / "provider/prompt-manifest.json", "v9 prompt manifest")
        prompt = PromptBuildResult(
            prompt="",
            included_episode_ids=(),
            omitted_episode_ids=(),
            prompt_bytes=0,
            estimated_tokens=0,
            truncated=False,
            manifest=prompt_manifest,
        )
        return self._finish(
            run=run,
            lineage=lineage,
            status=status,
            runnable=True,
            manifest=manifest,
            validation=validation,
            formal=formal,
            validation_attempted=True,
            validation_passed=validation_passed,
            formal_attempted=formal_attempted,
            performance=performance,
            champion=champion,
            round_act_count=1,
            error=error,
            prompt=prompt,
            raw_hash=failed_summary.get("provider_raw_sha256"),
            recovered_from=failed.name,
        )
