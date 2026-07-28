import json
from pathlib import Path

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import (
    load_pilot_config,
    load_round4_learning_config,
)
from agentbench_frame.generals.evaluator import (
    GeneralsEvaluation,
    build_evaluation_spec,
)
from agentbench_frame.generals.models import (
    AssetLayout,
    MatchResult,
    ReplaySkillAsset,
    TurnRecord,
)
from agentbench_frame.generals.pipeline_v4 import GeneralsHLRound4Pipeline
from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_pilot_config(FIXTURES / "pilot-v1.toml")
LEARNING = load_round4_learning_config(
    FIXTURES / "v4-strongest-learning-v1.toml",
    CONFIG,
)


def _state(round_number, target, improved=False):
    opponent = 1 - target
    target_army = 20 if improved else 5
    return {
        "round": round_number,
        "my_seat": target,
        "coins": (
            [20, 4] if target == 0 and improved
            else [4, 20] if improved
            else [4, 12] if target == 0
            else [12, 4]
        ),
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
        "cells": {
            "0,0": {
                "type": 0,
                "player": target,
                "army": target_army,
                "general_id": 0,
            },
            "0,1": {
                "type": 0,
                "player": target,
                "army": target_army,
                "general_id": None,
            },
            "0,3": {
                "type": 0,
                "player": opponent,
                "army": 9,
                "general_id": 1,
            },
            "0,4": {
                "type": 0,
                "player": opponent,
                "army": 8,
                "general_id": None,
            },
        },
        "generals": {
            "0": {
                "id": 0,
                "type": "main",
                "player": target,
                "position": [0, 0],
                "produce_level": 1,
            },
            "1": {
                "id": 1,
                "type": "main",
                "player": opponent,
                "position": [0, 3],
                "produce_level": 1,
            },
        },
    }


def _match(case, version, improved):
    before = _state(1, case.first_player, False)
    after = _state(5 if improved else 2, case.first_player, improved)
    return MatchResult(
        case_id=case.case_id,
        valid=True,
        winner=case.first_player if improved else 1 - case.first_player,
        termination_type="normal",
        seed=case.seed,
        evaluated_seat=case.first_player,
        turns=(
            TurnRecord(
                0,
                1,
                case.first_player,
                f"{case.case_id}-{version}-before",
                before,
                ((1, 0, 0, 0, 1), (8,)),
                f"{case.case_id}-{version}-after",
                after,
            ),
            TurnRecord(
                1,
                1,
                1 - case.first_player,
                f"{case.case_id}-{version}-other",
                after,
                ((8,),),
                f"{case.case_id}-{version}-terminal",
                after,
            ),
        ),
        elapsed_time_s=0.01,
        engine_hash="engine-hash",
    )


class FakeEvaluator:
    def __init__(self, improved=False, incomplete_validation=False):
        self.improved = improved
        self.incomplete_validation = incomplete_validation
        self.calls = []

    def evaluate(self, workspace, version, phase, run, cases=None):
        selected = tuple(cases or build_evaluation_spec(CONFIG).cases)
        self.calls.append((version, phase, len(selected)))
        improved = version == "v4" and self.improved
        matches = list(_match(case, version, improved) for case in selected)
        if phase == "validation" and self.incomplete_validation:
            first = matches[0]
            matches[0] = MatchResult(
                **{
                    **first.__dict__,
                    "valid": False,
                    "error": "synthetic invalid validation",
                }
            )
        results = tuple(
            GameResult(
                case.case_id,
                "win" if improved and match.valid else "loss",
                valid=match.valid,
                error=match.error,
                metadata={
                    "tier": case.metadata["tier"],
                    "version": version,
                    "phase": phase,
                    "seed": case.seed,
                    "evaluated_seat": case.first_player,
                },
            )
            for case, match in zip(selected, matches)
        )
        for match in matches:
            run.log_budget(
                phase,
                episodes=1,
                env_steps=len(match.turns),
                game_agent_decision_steps=1,
                primitive_commands=2,
                time_s=match.elapsed_time_s,
            )
        status = (
            "incomplete"
            if any(not match.valid for match in matches)
            else "complete"
        )
        score = (
            0.5
            if phase == "evaluation" and status == "complete"
            else None
        )
        return GeneralsEvaluation(
            version=version,
            status=status,
            score=score,
            wins=len(results) if improved else 0,
            losses=0 if improved else len(results),
            draws=0,
            per_tier={"high": score, "medium": score, "low": score},
            seat_gap=0.0 if score is not None else None,
            results=results,
            matches=tuple(matches),
        )


class RewritingProvider:
    provider_name = "codex"

    def __init__(self, change_behavior=False, fail=False, protected=False):
        self.calls = 0
        self.change_behavior = change_behavior
        self.fail = fail
        self.protected = protected

    def invoke(self, context):
        self.calls += 1
        workspace = Path(context["workspace_root"])
        if self.change_behavior:
            (workspace / "strategy.py").write_text(
                "def choose_actions(round_number, my_seat, view):\n"
                "    return [[1, 0, 1, 0, 2], [8]]\n",
                encoding="utf-8",
            )
        (workspace / "STRATEGY.md").write_text(
            "v4 reserve and concentration policy",
            encoding="utf-8",
        )
        (workspace / "EXPERIENCE.md").write_text(
            "Evidence: learn4 replay state IDs; reserve hypothesis.",
            encoding="utf-8",
        )
        if self.protected:
            (workspace / "main.py").write_text("tampered", encoding="utf-8")
        raw = Path(context["raw_output_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        if self.fail:
            return ProviderInvocation(
                status="failed",
                error="synthetic provider failure",
                raw_output_ref=str(raw),
            )
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(100, 25, 125, "exact"),
            tool_call_count=3,
            elapsed_time_s=0.1,
            raw_output_ref=str(raw),
        )


def _parent(tmp_path):
    parent = tmp_path / "parent"
    source = parent / "versions" / "v3" / "source"
    (source / "tests").mkdir(parents=True)
    (source / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view):\n"
        "    return [[1, 0, 0, 0, 1], [8]]\n",
        encoding="utf-8",
    )
    (source / "STRATEGY.md").write_text(
        "v3 sends army - 1",
        encoding="utf-8",
    )
    (source / "EXPERIENCE.md").write_text(
        "v3 says explicit reserve is missing",
        encoding="utf-8",
    )
    (source / "main.py").write_text("", encoding="utf-8")
    (source / "state_view.py").write_text("", encoding="utf-8")
    (source / "tests" / "test_strategy.py").write_text(
        "from strategy import choose_actions\n"
        "def test_end(): assert choose_actions(1, 0, {})[-1] == [8]\n",
        encoding="utf-8",
    )
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(
        manifest,
        parent / "versions" / "v3" / "manifest.json",
    )
    (parent / "summary.json").write_text(
        json.dumps({
            "run_id": "parent-v3",
            "status": "complete",
            "evaluation_status": "complete",
            "raw_score": 0.0,
            "evo_score_1": 0.0,
            "evo_score_2": 0.0,
            "evo_score_3": 7 / 18,
            "score_history": [0.0, 0.0, None, 0.0, 7 / 18],
            "act_count": 4,
            "cumulative_learning_budget": {
                "learning_coding_agent_acts": 4,
                "learning_episodes": 46,
                "learning_env_steps": 1000,
                "learning_game_agent_decision_steps": 500,
                "learning_primitive_commands": 2000,
                "learning_prompt_tokens": None,
                "learning_completion_tokens": None,
                "learning_total_tokens": None,
                "learning_time_s": 50.0,
            },
        }),
        encoding="utf-8",
    )
    return parent, manifest


def _pipeline(
    tmp_path,
    provider,
    *,
    improved=False,
    incomplete_validation=False,
    prior_attempt=False,
):
    parent, manifest = _parent(tmp_path)
    root = tmp_path / "assets"
    root.mkdir()
    layout = AssetLayout(
        root=root,
        engine_root=root,
        baseline_root=root,
        opponents=CONFIG.opponents,
        engine_hash="engine-hash",
    )
    rules = tmp_path / "rules.md"
    rules.write_text("official rules", encoding="utf-8")
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("human replay skill", encoding="utf-8")
    skill = ReplaySkillAsset(
        path=skill_path,
        text=skill_path.read_text(),
        sha256="a" * 64,
    )
    evaluator = FakeEvaluator(
        improved=improved,
        incomplete_validation=incomplete_validation,
    )
    prior_attempt_run_dir = None
    if prior_attempt:
        prior_attempt_run_dir = tmp_path / "prior-prompt-attempt"
        prior_attempt_run_dir.mkdir()
        (prior_attempt_run_dir / "summary.json").write_text(
            json.dumps({
                "run_id": "prior-prompt-attempt",
                "status": "prompt_incomplete",
                "parent_run_id": "parent-v3",
                "parent_version": "v3",
                "learning_id": LEARNING.learning_id,
                "act_count": 4,
                "round_act_count": 0,
                "budget": {
                    "learning_coding_agent_acts": 0,
                    "learning_episodes": 6,
                    "learning_env_steps": 12,
                    "learning_game_agent_decision_steps": 6,
                    "learning_primitive_commands": 12,
                    "learning_prompt_tokens": None,
                    "learning_completion_tokens": None,
                    "learning_total_tokens": None,
                    "learning_time_s": 1.0,
                },
            }),
            encoding="utf-8",
        )
    pipeline = GeneralsHLRound4Pipeline(
        config=CONFIG,
        learning_config=LEARNING,
        assets=layout,
        replay_skill=skill,
        parent_run_dir=parent,
        expected_parent_hash=manifest.content_hash,
        data_dir=tmp_path / "data",
        provider=provider,
        evaluator=evaluator,
        rules_path=rules,
        prior_attempt_run_dir=prior_attempt_run_dir,
    )
    return pipeline, evaluator


def _events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]


def test_round4_runnable_unchanged_behavior_still_gets_formal_evaluation(
    tmp_path,
):
    provider = RewritingProvider(change_behavior=False)
    pipeline, evaluator = _pipeline(tmp_path, provider, improved=False)

    result = pipeline.run()

    assert result.status == "complete"
    assert result.evo_score_4 == 0.5
    assert result.gain_4 == 0.5
    assert result.global_act_count == 5
    assert result.round_act_count == 1
    assert result.runnable is True
    assert evaluator.calls == [
        ("v3", "learning", 6),
        ("v4", "validation", 6),
        ("v4", "evaluation", 18),
    ]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["score_history"] == [
        0.0,
        0.0,
        None,
        0.0,
        7 / 18,
        0.5,
    ]
    assert summary["feedback_read"]["episodes"] == 6
    assert summary["feedback_read"]["decision_records"] == 6
    assert summary["behavior_diagnostics"]["action_disagreement"] == 0.0
    assert summary["behavior_diagnostics"]["formal_evaluation_blocking"] is False
    assert summary["budget"]["learning_episodes"] == 6
    assert summary["budget"]["validation_episodes"] == 6
    assert summary["budget"]["evaluation_episodes"] == 18
    assert summary["cumulative_learning_budget"]["learning_episodes"] == 52
    version = next(
        event for event in _events(result.run_dir)
        if event["event_type"] == "version"
        and event["version"] == "v4"
    )
    assert version["status"] == "available"
    assert not any(
        event["event_type"] == "behavior_gate"
        for event in _events(result.run_dir)
    )


def test_round4_incomplete_validation_is_diagnostic_not_formal_gate(tmp_path):
    provider = RewritingProvider(change_behavior=True)
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        improved=True,
        incomplete_validation=True,
    )

    result = pipeline.run()

    assert result.status == "complete"
    assert result.evo_score_4 == 0.5
    assert evaluator.calls[-1] == ("v4", "evaluation", 18)
    diagnostics = next(
        event for event in _events(result.run_dir)
        if event["event_type"] == "behavior_diagnostics"
    )
    assert diagnostics["validation_complete"] is False
    assert diagnostics["formal_evaluation_blocking"] is False


def test_round4_provider_or_protected_file_failure_retains_missing_score(
    tmp_path,
):
    provider = RewritingProvider(fail=True, protected=True)
    pipeline, evaluator = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert result.status == "provider_failed"
    assert result.runnable is False
    assert result.evo_score_4 is None
    assert evaluator.calls == [("v3", "learning", 6)]
    assert (result.run_dir / "versions" / "v4" / "manifest.json").is_file()
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["score_history"][-1] is None
    assert summary["evaluation_status"] == "not_run"


def test_round4_cumulative_budget_includes_prior_pre_act_attempt(tmp_path):
    provider = RewritingProvider()
    pipeline, _ = _pipeline(
        tmp_path,
        provider,
        prior_attempt=True,
    )

    result = pipeline.run()

    assert result.status == "complete"
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["prior_attempt_run_id"] == "prior-prompt-attempt"
    assert summary["cumulative_learning_budget"]["learning_episodes"] == 58
    assert summary["cumulative_learning_budget"][
        "learning_coding_agent_acts"
    ] == 5
    receipt = next(
        event for event in _events(result.run_dir)
        if event["event_type"] == "prior_attempt_import"
    )
    assert receipt["learning_episodes"] == 6
