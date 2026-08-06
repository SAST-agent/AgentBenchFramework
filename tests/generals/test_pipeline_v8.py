import json
from pathlib import Path

import pytest

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.challenge_v8 import (
    load_round8_challenge_config,
)
from agentbench_frame.generals.evaluator import GeneralsEvaluation
from agentbench_frame.generals.measurement import ProbeState
from agentbench_frame.generals.models import (
    AssetLayout,
    MatchResult,
    ReplaySkillAsset,
    TurnRecord,
)
from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
from tests.generals.test_lineage_v8 import _parent


FIXTURES = Path(__file__).parent / "fixtures"
ENGINE_SHA256 = (
    "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
)
SKILL_SHA256 = (
    "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
)
CONFIG = load_pilot_config(FIXTURES / "pilot-v1.toml")
CHALLENGE = load_round8_challenge_config(
    FIXTURES / "v8-clean-room-challenge-v1.toml",
    CONFIG,
    engine_hash=ENGINE_SHA256,
    replay_skill_sha256=SKILL_SHA256,
)


def _state(round_number: int, seat: int) -> dict:
    return {
        "round": round_number,
        "my_seat": seat,
        "coins": [80, 80],
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
        "cells": {
            "0,0": {
                "type": 0,
                "player": seat,
                "army": 30,
                "general_id": 0,
            },
            "0,1": {
                "type": 0,
                "player": -1,
                "army": 1,
                "general_id": None,
            },
            "0,3": {
                "type": 0,
                "player": 1 - seat,
                "army": 20,
                "general_id": 1,
            },
        },
        "generals": {
            "0": {
                "id": 0,
                "type": "main",
                "player": seat,
                "position": [0, 0],
                "produce_level": 1,
            },
            "1": {
                "id": 1,
                "type": "main",
                "player": 1 - seat,
                "position": [0, 3],
                "produce_level": 1,
            },
        },
    }


def _match(case, version: str, outcome: str, valid: bool = True):
    before = _state(1, case.first_player)
    after = _state(2, case.first_player)
    winner = (
        case.first_player
        if outcome == "win"
        else 1 - case.first_player
        if outcome == "loss"
        else None
    )
    return MatchResult(
        case_id=case.case_id,
        valid=valid,
        winner=winner,
        termination_type="normal" if valid else "engine_error",
        seed=case.seed,
        evaluated_seat=case.first_player,
        turns=(
            TurnRecord(
                step=0,
                round_number=1,
                player=case.first_player,
                state_id_before=f"{case.case_id}-{version}-before",
                state_before=before,
                commands=((1, 0, 0, 2, 8), (8,)),
                state_id_after=f"{case.case_id}-{version}-after",
                state_after=after,
            ),
        ),
        elapsed_time_s=0.01,
        engine_hash=ENGINE_SHA256,
        error=None if valid else "synthetic invalid match",
    )


class ChampionEvaluator:
    def __init__(
        self,
        *,
        validation_wins=(4, 3),
        sealed_wins=(6, 5),
        formal_wins=(6, 6),
        invalid_learning=False,
        invalid_validation=False,
        invalid_sealed=False,
    ):
        self.validation_wins = validation_wins
        self.sealed_wins = sealed_wins
        self.formal_wins = formal_wins
        self.invalid_learning = invalid_learning
        self.invalid_validation = invalid_validation
        self.invalid_sealed = invalid_sealed
        self.calls = []

    def _outcomes(self, cases, phase):
        if phase == "validation":
            remaining = list(self.validation_wins)
        elif phase == "sealed":
            remaining = list(self.sealed_wins)
        elif phase == "formal":
            remaining = list(self.formal_wins)
        else:
            remaining = [0, 0]
        outcomes = []
        for case in cases:
            seat = case.first_player
            if remaining[seat] > 0:
                remaining[seat] -= 1
                outcomes.append("win")
            else:
                outcomes.append("loss")
        return outcomes

    def evaluate(self, workspace, version, phase, run, cases=None):
        selected = tuple(cases or ())
        self.calls.append((version, phase, len(selected)))
        outcomes = self._outcomes(selected, phase)
        invalid_index = (
            0
            if (
                (phase == "learning" and self.invalid_learning)
                or (phase == "validation" and self.invalid_validation)
                or (phase == "sealed" and self.invalid_sealed)
            )
            else None
        )
        matches = tuple(
            _match(
                case,
                version,
                outcome,
                valid=index != invalid_index,
            )
            for index, (case, outcome) in enumerate(zip(selected, outcomes))
        )
        results = tuple(
            GameResult(
                case_id=case.case_id,
                outcome=outcome,
                valid=index != invalid_index,
                error=(
                    "synthetic invalid match"
                    if index == invalid_index
                    else None
                ),
                metadata={
                    "tier": case.metadata["tier"],
                    "version": version,
                    "phase": phase,
                    "seed": case.seed,
                    "evaluated_seat": case.first_player,
                },
            )
            for index, (case, outcome) in enumerate(zip(selected, outcomes))
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
        complete = all(result.valid for result in results)
        wins = sum(result.outcome == "win" for result in results if result.valid)
        score = wins / len(results) if complete and results else None
        per_tier = {
            tier: (
                sum(
                    result.outcome == "win"
                    for result in results
                    if result.metadata["tier"] == tier
                )
                / sum(
                    result.metadata["tier"] == tier
                    for result in results
                )
            )
            for tier in {result.metadata["tier"] for result in results}
        } if complete else {}
        return GeneralsEvaluation(
            version=version,
            status="complete" if complete else "incomplete",
            score=score,
            wins=wins,
            losses=sum(
                result.outcome == "loss"
                for result in results
                if result.valid
            ),
            draws=0,
            per_tier=per_tier,
            seat_gap=0.0 if complete else None,
            results=results,
            matches=matches,
        )


class Round8Provider:
    provider_name = "codex"

    def __init__(
        self,
        *,
        fail=False,
        forbidden_file=False,
        failing_test=False,
        long_macro=False,
        compressed_docs=False,
    ):
        self.fail = fail
        self.forbidden_file = forbidden_file
        self.failing_test = failing_test
        self.long_macro = long_macro
        self.compressed_docs = compressed_docs
        self.calls = 0
        self.prompt = None

    def invoke(self, context):
        self.calls += 1
        self.prompt = context["prompt"]
        workspace = Path(context["workspace_root"])
        (workspace / "STRATEGY.md").write_text(
            (
                "Deterministic policy with legal commands, main safety, "
                "macro limit, lexicographic ties, and verified prefix.\n"
                if self.compressed_docs else
                "Beam width: 4\nFamily top-k: 2\nMacro limit: 8\n"
                "Phase weights: opening/economy/contact/assault\n"
                "Tie order: command,row,column\n"
                "Fallback: v7 commands 1/3/5\n"
            ),
            encoding="utf-8",
        )
        (workspace / "EXPERIENCE.md").write_text(
            "Integrated learning states; retained fallback; rejected unsafe "
            "unmodeled transitions; remaining risk is champion pressure.\n",
            encoding="utf-8",
        )
        (workspace / "policy").mkdir(exist_ok=True)
        (workspace / "policy" / "planner.py").write_text(
            "BEAM_WIDTH = 4\nMAX_MACRO = 8\n",
            encoding="utf-8",
        )
        if self.failing_test:
            (workspace / "tests" / "test_v8_candidate.py").write_text(
                "def test_candidate():\n"
                "    assert False, 'synthetic candidate failure'\n",
                encoding="utf-8",
            )
        if self.long_macro:
            commands = ", ".join(
                "[1, 0, 0, 2, 1]" for _ in range(9)
            )
            (workspace / "strategy.py").write_text(
                "def choose_actions(round_number, my_seat, view):\n"
                f"    return [{commands}, [8]]\n",
                encoding="utf-8",
            )
        if self.forbidden_file:
            (workspace / "main.py").write_text(
                "FORBIDDEN = True\n",
                encoding="utf-8",
            )
        raw = Path(context["raw_output_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        if self.fail:
            return ProviderInvocation(
                status="failed",
                raw_output_ref=str(raw),
                error="synthetic provider failure",
            )
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(100, 25, 125, "exact"),
            tool_call_count=3,
            elapsed_time_s=0.1,
            raw_output_ref=str(raw),
        )


def _pipeline(tmp_path, provider, evaluator=None):
    from agentbench_frame.generals.pipeline_v8 import (
        GeneralsHLRound8Pipeline,
    )

    parent, manifest = _parent(tmp_path)
    root = tmp_path / "assets"
    root.mkdir()
    layout = AssetLayout(
        root=root,
        engine_root=root,
        baseline_root=root,
        opponents=CONFIG.opponents,
        engine_hash=ENGINE_SHA256,
    )
    rules = tmp_path / "rules.md"
    rules.write_text("official rules", encoding="utf-8")
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("human replay skill", encoding="utf-8")
    skill = ReplaySkillAsset(
        path=skill_path,
        text=skill_path.read_text(encoding="utf-8"),
        sha256=SKILL_SHA256,
    )
    selected_evaluator = evaluator or ChampionEvaluator()
    pipeline = GeneralsHLRound8Pipeline(
        config=CONFIG,
        challenge=CHALLENGE,
        assets=layout,
        replay_skill=skill,
        parent_run_dir=parent,
        expected_parent_hash=manifest.content_hash,
        data_dir=tmp_path / "data",
        provider=provider,
        evaluator=selected_evaluator,
        rules_path=rules,
    )
    return pipeline, selected_evaluator


def _summary(run_dir):
    return json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )


class PhaseExceptionEvaluator(ChampionEvaluator):
    def __init__(self, phase):
        super().__init__()
        self.phase = phase

    def evaluate(self, workspace, version, phase, run, cases=None):
        if phase == self.phase:
            selected = tuple(cases or ())
            self.calls.append((version, phase, len(selected)))
            raise RuntimeError(f"synthetic {phase} crash")
        return super().evaluate(
            workspace,
            version,
            phase,
            run,
            cases=cases,
        )


def test_clean_room_v8_runs_one_act_and_all_declared_suites(tmp_path):
    provider = Round8Provider()
    pipeline, evaluator = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert provider.calls == 1
    assert ("v7", "learning", 6) in evaluator.calls
    assert ("v8", "validation", 12) in evaluator.calls
    assert ("v8", "formal", 18) in evaluator.calls
    assert not any(phase == "sealed" for _, phase, _ in evaluator.calls)
    assert result.global_act_count == 9
    assert result.round_act_count == 1
    assert result.runnable is True
    assert result.formal_attempted is True
    assert result.evo_score_8 == 12 / 18
    assert result.performance_target_met is True
    summary = _summary(result.run_dir)
    assert summary["performance_target_met"] is True
    assert len(summary["learning_results"]) == 6
    assert len(summary["validation_results"]) == 12
    assert len(summary["formal_results"]) == 18
    assert "301101" not in provider.prompt
    assert "280101" not in provider.prompt


def test_clean_room_v8_validation_failure_cannot_hide_formal(tmp_path):
    provider = Round8Provider()
    evaluator = ChampionEvaluator(validation_wins=(0, 0), formal_wins=(0, 0))
    pipeline, evaluator = _pipeline(tmp_path, provider, evaluator)

    result = pipeline.run()

    assert ("v8", "formal", 18) in evaluator.calls
    assert result.validation_passed is False
    assert result.formal_attempted is True
    assert result.performance_target_met is False
    assert len(_summary(result.run_dir)["formal_results"]) == 18


def test_clean_room_v8_provider_or_candidate_failure_has_no_score(tmp_path):
    provider = Round8Provider(fail=True)
    pipeline, _ = _pipeline(tmp_path, provider)
    failed = pipeline.run()
    assert provider.calls == 1
    assert failed.runnable is False
    assert failed.evo_score_8 is None

    provider = Round8Provider(failing_test=True)
    pipeline, _ = _pipeline(tmp_path / "candidate", provider)
    failed = pipeline.run()
    assert provider.calls == 1
    assert failed.runnable is False
    assert failed.evo_score_8 is None


def test_clean_room_v8_protected_file_mutation_is_rejected(tmp_path):
    provider = Round8Provider(forbidden_file=True)
    pipeline, _ = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert provider.calls == 1
    assert result.runnable is False
    assert result.formal_attempted is False

