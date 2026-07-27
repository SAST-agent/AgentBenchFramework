import json
from pathlib import Path

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import (
    load_pilot_config,
    load_round3_learning_config,
)
from agentbench_frame.generals.evaluator import (
    GeneralsEvaluation,
    build_evaluation_spec,
)
from agentbench_frame.generals.models import (
    AssetLayout,
    MatchResult,
    TurnRecord,
)
from agentbench_frame.generals.pipeline_v3 import GeneralsHLRound3Pipeline
from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_pilot_config(FIXTURES / "pilot-v1.toml")
LEARNING = load_round3_learning_config(
    FIXTURES / "v3-strongest-learning-v1.toml",
    CONFIG,
)


def _state(round_number: int, target: int, improved: bool = False):
    opponent = 1 - target
    target_army = 14 if improved else 5
    state = {
        "round": round_number,
        "coins": (
            [18, 4] if target == 0 and improved
            else [4, 18] if improved
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
            },
            "1": {
                "id": 1,
                "type": "main",
                "player": opponent,
                "position": [0, 3],
            },
        },
    }
    if improved:
        state["cells"]["0,2"] = {
            "type": 0,
            "player": target,
            "army": 12,
            "general_id": None,
        }
    return state


def _match(case, version):
    improved = version == "v3"
    initial = _state(1, case.first_player, improved=False)
    middle = _state(1, case.first_player, improved=improved)
    terminal = _state(5 if improved else 2, case.first_player, improved=improved)
    old_action = ((1, 0, 0, 0, 1), (8,))
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
                f"{case.case_id}-before",
                initial,
                old_action,
                f"{case.case_id}-middle",
                middle,
            ),
            TurnRecord(
                1,
                1,
                1 - case.first_player,
                f"{case.case_id}-middle",
                middle,
                ((8,),),
                f"{case.case_id}-after",
                terminal,
            ),
        ),
        elapsed_time_s=0.01,
        engine_hash="engine-hash",
    )


class FakeEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, workspace, version, phase, run, cases=None):
        selected = tuple(cases or build_evaluation_spec(CONFIG).cases)
        self.calls.append((version, phase, len(selected)))
        matches = tuple(_match(case, version) for case in selected)
        results = tuple(
            GameResult(
                case.case_id,
                "win" if version == "v3" else "loss",
                metadata={
                    "tier": case.metadata["tier"],
                    "version": version,
                    "phase": phase,
                    "seed": case.seed,
                    "evaluated_seat": case.first_player,
                },
            )
            for case in selected
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
        score = 0.5 if phase == "evaluation" and version == "v3" else None
        return GeneralsEvaluation(
            version=version,
            status="complete",
            score=score,
            wins=len(results) if version == "v3" else 0,
            losses=0 if version == "v3" else len(results),
            draws=0,
            per_tier={"high": score, "medium": score, "low": score},
            seat_gap=0.0 if score is not None else None,
            results=results,
            matches=matches,
        )


class RewritingProvider:
    provider_name = "codex"

    def __init__(self, change_behavior=True):
        self.calls = 0
        self.change_behavior = change_behavior

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
            "v3 frontier policy",
            encoding="utf-8",
        )
        (workspace / "EXPERIENCE.md").write_text(
            "Evidence: all six learn3 replay IDs; rejected main-only BFS.",
            encoding="utf-8",
        )
        raw = Path(context["raw_output_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(100, 25, 125, "exact"),
            tool_call_count=3,
            elapsed_time_s=0.1,
            raw_output_ref=str(raw),
        )


def _parent(tmp_path):
    parent = tmp_path / "parent"
    source = parent / "versions" / "v2" / "source"
    (source / "tests").mkdir(parents=True)
    (source / "strategy.py").write_text(
        "def choose_actions(round_number, my_seat, view):\n"
        "    return [[1, 0, 0, 0, 1], [8]]\n",
        encoding="utf-8",
    )
    (source / "STRATEGY.md").write_text("v2 main-only BFS", encoding="utf-8")
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
        parent / "versions" / "v2" / "manifest.json",
    )
    (parent / "versions" / "v1-to-v2.patch").write_text(
        "second diff",
        encoding="utf-8",
    )
    (parent / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "parent-v2",
                "status": "complete",
                "raw_score": 0.0,
                "evo_score_1": 0.0,
                "evo_score_2": 0.0,
                "act_count": 2,
                "recovery_from_run_id": "failed-act",
                "budget": {
                    "learning_coding_agent_acts": 2,
                    "learning_episodes": 24,
                    "learning_env_steps": 48,
                    "learning_game_agent_decision_steps": 24,
                    "learning_primitive_commands": 48,
                    "learning_prompt_tokens": 100,
                    "learning_completion_tokens": 50,
                    "learning_total_tokens": 150,
                    "learning_time_s": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return parent, manifest


def _pipeline(tmp_path, provider):
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
    guide = tmp_path / "replay.md"
    rules.write_text("official rules", encoding="utf-8")
    guide.write_text("replay fields", encoding="utf-8")
    evaluator = FakeEvaluator()
    pipeline = GeneralsHLRound3Pipeline(
        config=CONFIG,
        learning_config=LEARNING,
        assets=layout,
        parent_run_dir=parent,
        expected_parent_hash=manifest.content_hash,
        data_dir=tmp_path / "data",
        provider=provider,
        evaluator=evaluator,
        rules_path=rules,
        replay_guide_path=guide,
    )
    return pipeline, evaluator


def _events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]


def test_round3_pipeline_gates_then_adds_one_formal_score_point(tmp_path):
    provider = RewritingProvider()
    pipeline, evaluator = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert result.status == "complete"
    assert result.raw_score == 0.0
    assert result.evo_score_1 == 0.0
    assert result.evo_score_2 == 0.0
    assert result.evo_score_3 == 0.5
    assert result.gain_3 == 0.5
    assert result.global_act_count == 4
    assert result.round_act_count == 1
    assert result.gate_passed is True
    assert provider.calls == 1
    assert evaluator.calls == [
        ("v2", "learning", 6),
        ("v3", "learning", 6),
        ("v3", "evaluation", 18),
    ]
    assert (result.run_dir / "versions" / "v2-to-v3.patch").is_file()
    assert (result.run_dir / "versions" / "v3" / "source" / "EXPERIENCE.md").is_file()
    assert (result.run_dir / "provider" / "codex-act-v3.prompt.md").stat().st_size <= 65_536
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["score_history"] == [0.0, 0.0, None, 0.0, 0.5]
    assert summary["behavior_gate"]["passed"] is True
    gate = next(
        item for item in _events(result.run_dir)
        if item["event_type"] == "behavior_gate"
    )
    assert gate["passed"] is True
    assert gate["dense_deltas"]["terminal_territory_share"] > 0


def test_round3_pipeline_rejects_no_behavior_change_before_formal_eval(tmp_path):
    provider = RewritingProvider(change_behavior=False)
    pipeline, evaluator = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert result.status == "behavior_gate_failed"
    assert result.evo_score_3 is None
    assert result.gate_passed is False
    assert evaluator.calls == [
        ("v2", "learning", 6),
        ("v3", "learning", 6),
    ]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["benchmark_score"] is None
    assert summary["score_history"] == [0.0, 0.0, None, 0.0, None]
    assert summary["behavior_gate"]["conditions"]["behavior_changed"] is False
    assert summary["behavior_gate"]["conditions"]["frontline_move_observed"] is False
