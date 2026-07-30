from dataclasses import replace
import json
from pathlib import Path

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.evaluator import GeneralsEvaluation
from agentbench_frame.generals.models import AssetLayout, MatchResult, TurnRecord
from agentbench_frame.generals.pipeline import GeneralsHLPipeline, _strategy_view
from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
from agentbench_frame.tracking.run import Run


FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"


class FakeProvider:
    provider_name = "codex"

    def invoke(self, context):
        workspace = Path(context["workspace_root"])
        assert (workspace / ".git").is_dir()
        strategy = workspace / "strategy.py"
        strategy.write_text(strategy.read_text().replace("POLICY = 0", "POLICY = 1"))
        raw = Path(context["raw_output_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(
            '{"type":"thread.started","thread_id":"fake"}\n'
            '{"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":3}}\n'
        )
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(2, 3, 5, "exact"),
            raw_output_ref=str(raw),
            tool_call_count=1,
        )


class FakeEvaluator:
    def evaluate(self, workspace, version, phase, run, cases=None):
        selected = list(cases or ())
        if phase == "evaluation" and not selected:
            from agentbench_frame.generals.evaluator import build_evaluation_spec

            selected = build_evaluation_spec(CONFIG).cases
        matches = []
        results = []
        for case in selected:
            turn = TurnRecord(
                0,
                1,
                case.first_player,
                f"{case.case_id}-before",
                {"round": 1, "my_seat": case.first_player, "cells": {}, "generals": {},
                 "coins": [40, 40], "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]]},
                ((8,),),
                f"{case.case_id}-after",
                {"round": 1},
            )
            match = MatchResult(
                case.case_id, True, case.first_player, "normal", case.seed,
                case.first_player, (turn,), 0.01, "hash"
            )
            matches.append(match)
            results.append(GameResult(case.case_id, "win", metadata={"tier": case.metadata["tier"]}))
            run.log_budget(phase, episodes=1, env_steps=1, time_s=0.01)
            run.log_game_result(phase, version, {"case_id": case.case_id, "valid": True})
        score = 0.25 if version == "v0" and phase == "evaluation" else (
            0.50 if version == "v1" and phase == "evaluation" else None
        )
        return GeneralsEvaluation(
            version, "complete", score, len(results), 0, 0,
            {"high": score, "medium": score, "low": score},
            0.0, tuple(results), tuple(matches)
        )


CONFIG = load_pilot_config(FIXTURE)


def test_probe_adapter_accepts_legacy_engine_normalization():
    state = {
        "round": 1,
        "board": [[{"position": [0, 0], "type": 0, "player": 1, "army": 2,
                    "general_id": 3}]],
        "generals": [{"id": 3, "class": "MainGenerals", "player": 1,
                      "position": [0, 0], "produce_level": 1,
                      "defense_level": 1, "mobility_level": 1}],
        "coins": [40, 40],
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
    }
    view = _strategy_view(state, 1)
    assert view["generals"]["3"]["type"] == "main"
    assert view["cells"]["0,0"]["army"] == 2
    assert view["my_seat"] == 1


def test_probe_adapter_enriches_historical_replay_strategy_fields():
    state = {
        "round": 40,
        "cells": {},
        "generals": {},
        "coins": [80, 90],
        "tech_level": [[3, 0, 0, 0], [5, 0, 0, 0]],
        "weapons": [
            {
                "type": 0,
                "player": 1,
                "rest": 4,
                "position": [3, 4],
                "cd": 0,
            }
        ],
    }

    view = _strategy_view(state, 1)

    assert view["movement_budget"] == [3, 5]
    assert view["active_super_weapons"] == [
        {
            "type": 0,
            "player": 1,
            "rest": 4,
            "position": [3, 4],
            "cd": 0,
        }
    ]
    assert view["my_seat"] == 1


def test_run_resume_restores_phase_budgets(tmp_path):
    run = Run.start("28_generals", "baseline", data_dir=str(tmp_path))
    run.log_budget(
        "calibration",
        episodes=3,
        env_steps=6,
        game_agent_decision_steps=3,
        primitive_commands=9,
        time_s=0.25,
    )
    run.log_budget(
        "learning",
        episodes=2,
        env_steps=5,
        game_agent_decision_steps=2,
        primitive_commands=7,
        coding_agent_acts=1,
        prompt_tokens=7,
        completion_tokens=3,
        time_s=0.5,
    )
    run.finish({"status": "failed"})
    resumed = Run.resume(Path(run.run_dir))
    assert resumed.budget_snapshot()["calibration_episodes"] == 3
    assert resumed.budget_snapshot()["calibration_game_agent_decision_steps"] == 3
    assert resumed.budget_snapshot()["calibration_primitive_commands"] == 9
    assert resumed.budget_snapshot()["learning_episodes"] == 2
    assert resumed.budget_snapshot()["learning_total_tokens"] == 10
    assert resumed.budget_snapshot()["learning_game_agent_decision_steps"] == 2
    assert resumed.budget_snapshot()["learning_primitive_commands"] == 7
    resumed.finish({"status": "complete"})


def test_pipeline_writes_complete_v0_v1_evidence_chain(tmp_path):
    baseline = tmp_path / "baseline"
    (baseline / "tests").mkdir(parents=True)
    (baseline / "strategy.py").write_text(
        "POLICY = 0\n"
        "def choose_actions(round_number, my_seat, view):\n"
        "    return [[8]]\n"
    )
    (baseline / "STRATEGY.md").write_text("Current")
    (baseline / "main.py").write_text("")
    (baseline / "state_view.py").write_text("")
    (baseline / "tests" / "test_strategy.py").write_text(
        "from strategy import choose_actions\n"
        "def test_end(): assert choose_actions(1,0,{})[-1] == [8]\n"
    )
    rules = tmp_path / "rules.md"
    replay = tmp_path / "replay.md"
    rules.write_text("Rules")
    replay.write_text("Replay")
    (tmp_path / "main.py").write_text("")
    (tmp_path / "main_for_player_test.py").write_text("")
    (tmp_path / "makefile").write_text("all:\n\t@true\n")
    layout = AssetLayout(
        root=tmp_path,
        engine_root=tmp_path,
        baseline_root=baseline,
        opponents=tuple(replace(item, source=tmp_path) for item in CONFIG.opponents),
        engine_hash="hash",
    )
    pipeline = GeneralsHLPipeline(
        config=CONFIG,
        assets=layout,
        data_dir=tmp_path / "data",
        provider=FakeProvider(),
        evaluator=FakeEvaluator(),
        rules_path=rules,
        replay_guide_path=replay,
    )
    result = pipeline.run()

    assert result.raw_score == 0.25
    assert result.evo_score == 0.50
    assert result.gain == 0.25
    assert result.act_count == 1
    assert (result.run_dir / "provider" / "codex.raw.jsonl").exists()
    assert (result.run_dir / "versions" / "v0" / "manifest.json").exists()
    assert (result.run_dir / "versions" / "v1" / "manifest.json").exists()
    assert (result.run_dir / "versions" / "v0-to-v1.patch").read_text()
    assert (result.run_dir / "quality.json").exists()
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["budget"]["learning_coding_agent_acts"] == 1
