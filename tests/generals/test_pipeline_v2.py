from dataclasses import replace
import json
from pathlib import Path

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import (
    load_calibration_config,
    load_calibration_selection,
    load_pilot_config,
)
from agentbench_frame.generals.calibration import build_calibration_cases
from agentbench_frame.generals.evaluator import (
    GeneralsEvaluation,
    build_evaluation_spec,
    build_round2_learning_cases,
)
from agentbench_frame.generals.models import (
    AssetLayout,
    CalibrationEvaluation,
    MatchResult,
    TurnRecord,
)
from agentbench_frame.generals.pipeline_v2 import (
    GeneralsHLRound2Pipeline,
    calibrate_development,
)
from agentbench_frame.tracking.provider import ProviderInvocation, ProviderUsage
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_pilot_config(FIXTURES / "pilot-v1.toml")
CALIBRATION = load_calibration_config(FIXTURES / "calibration-v1.toml")
SELECTION = load_calibration_selection(
    FIXTURES / "calibration-v1-selection.toml", CALIBRATION
)


def _state(round_number, target):
    opponent = 1 - target
    return {
        "round": round_number,
        "coins": [40 + round_number, 42 + round_number],
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
        "cells": {
            "0,0": {"type": 0, "player": target, "army": 5, "general_id": 0},
            "0,1": {"type": 0, "player": opponent, "army": 4, "general_id": 1},
        },
        "generals": {
            "0": {"id": 0, "type": "main", "player": target, "position": [0, 0]},
            "1": {"id": 1, "type": "main", "player": opponent, "position": [0, 1]},
        },
    }


def _match(case, winner=None):
    winner = case.first_player if winner is None else winner
    before = _state(1, case.first_player)
    middle = _state(1, case.first_player)
    after = _state(2, case.first_player)
    return MatchResult(
        case_id=case.case_id,
        valid=True,
        winner=winner,
        termination_type="normal",
        seed=case.seed,
        evaluated_seat=case.first_player,
        turns=(
            TurnRecord(
                0, 1, 0, f"{case.case_id}-0", before, ((8,),),
                f"{case.case_id}-1", middle,
            ),
            TurnRecord(
                1, 1, 1, f"{case.case_id}-1", middle, ((8,),),
                f"{case.case_id}-2", after,
            ),
        ),
        elapsed_time_s=0.01,
        engine_hash="engine-hash",
    )


class FakeProvider:
    provider_name = "codex"

    def __init__(self):
        self.calls = 0

    def invoke(self, context):
        self.calls += 1
        workspace = Path(context["workspace_root"])
        strategy = workspace / "strategy.py"
        strategy.write_text(strategy.read_text().replace("POLICY = 1", "POLICY = 2"))
        raw = Path(context["raw_output_path"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text('{"type":"turn.completed"}\n')
        return ProviderInvocation(
            status="completed",
            usage=ProviderUsage(11, 7, 18, "exact"),
            tool_call_count=1,
            raw_output_ref=str(raw),
        )


class FakeFormalEvaluator:
    def evaluate(self, workspace, version, phase, run, cases=None):
        selected = tuple(cases or build_evaluation_spec(CONFIG).cases)
        matches = tuple(_match(case) for case in selected)
        results = tuple(
            GameResult(
                case.case_id,
                "win",
                metadata={"tier": case.metadata["tier"], "version": version, "phase": phase},
            )
            for case in selected
        )
        for match in matches:
            run.log_budget(
                phase,
                episodes=1,
                env_steps=2,
                game_agent_decision_steps=1,
                primitive_commands=2,
                time_s=0.01,
            )
        score = None
        if phase == "evaluation":
            score = 0.0 if version == "v1" else 0.5
        return GeneralsEvaluation(
            version=version,
            status="complete",
            score=score,
            wins=len(results),
            losses=0,
            draws=0,
            per_tier={"high": score, "medium": score, "low": score},
            seat_gap=0.0 if score is not None else None,
            results=results,
            matches=matches,
        )


class FakeCalibrationEvaluator:
    def __init__(self, heldout_score=0.4):
        self.heldout_score = heldout_score

    def evaluate(self, workspace, version, split, mode, run, budget_phase):
        cases = build_calibration_cases(CALIBRATION, split, mode)
        matches = tuple(_match(case) for case in cases)
        for _match_result in matches:
            run.log_budget(
                budget_phase,
                episodes=1,
                env_steps=2,
                game_agent_decision_steps=1,
                primitive_commands=2,
                time_s=0.01,
            )
        score = self.heldout_score if split == "heldout" else 0.5
        return CalibrationEvaluation(
            mode=mode,
            split=split,
            status="complete",
            score=score,
            wins=4,
            losses=6,
            draws=0,
            per_seat={0: score, 1: score},
            results=tuple(
                GameResult(case.case_id, "win", metadata={"tier": "calibration"})
                for case in cases
            ),
            matches=matches,
            in_target_range=(
                CALIBRATION.target_min <= score <= CALIBRATION.target_max
                if split == "heldout"
                else None
            ),
        )


def _parent(tmp_path):
    parent = tmp_path / "parent"
    snapshotter = LocalWorkspaceSnapshotter()
    manifests = {}
    for version, policy in (("v0", 0), ("v1", 1)):
        source = parent / "versions" / version / "source"
        (source / "tests").mkdir(parents=True)
        (source / "strategy.py").write_text(
            f"POLICY = {policy}\n"
            "def choose_actions(round_number, my_seat, view):\n"
            "    return [[8]] if POLICY == 1 else [[7], [8]]\n"
        )
        (source / "STRATEGY.md").write_text(f"policy {policy}")
        (source / "main.py").write_text("")
        (source / "state_view.py").write_text("")
        (source / "tests" / "test_strategy.py").write_text(
            "from strategy import choose_actions\n"
            "def test_end(): assert choose_actions(1, 0, {})[-1] == [8]\n"
        )
        manifests[version] = snapshotter.capture(
            source, previous=manifests.get("v0")
        )
        snapshotter.write_manifest(
            manifests[version], parent / "versions" / version / "manifest.json"
        )
    (parent / "versions" / "v0-to-v1.patch").write_text("first diff")
    (parent / "summary.json").write_text(json.dumps({
        "run_id": "parent-1",
        "status": "complete",
        "raw_score": 0.0,
        "evo_score": 0.0,
        "act_count": 1,
        "budget": {
            "learning_coding_agent_acts": 1,
            "learning_episodes": 12,
            "learning_env_steps": 24,
            "learning_game_agent_decision_steps": 12,
            "learning_primitive_commands": 24,
            "learning_total_tokens": 100,
            "learning_time_s": 1.0,
        },
    }))
    return parent, manifests["v1"]


def _pipeline(tmp_path, provider, calibration_evaluator):
    parent, manifest = _parent(tmp_path)
    root = tmp_path / "assets"
    root.mkdir()
    layout = AssetLayout(
        root=root,
        engine_root=root,
        baseline_root=root,
        opponents=tuple(replace(item, source=root) for item in CONFIG.opponents),
        engine_hash="engine-hash",
    )
    rules = tmp_path / "rules.md"
    replay = tmp_path / "replay.md"
    rules.write_text("official rules")
    replay.write_text("replay fields")
    return GeneralsHLRound2Pipeline(
        config=CONFIG,
        assets=layout,
        calibration_config=CALIBRATION,
        calibration_selection=SELECTION,
        parent_run_dir=parent,
        expected_parent_hash=manifest.content_hash,
        data_dir=tmp_path / "data",
        provider=provider,
        evaluator=FakeFormalEvaluator(),
        calibration_evaluator=calibration_evaluator,
        rules_path=rules,
        replay_guide_path=replay,
    )


def test_round2_learning_matrix_uses_new_low_and_medium_seeds_only():
    cases = build_round2_learning_cases(CONFIG)
    assert len(cases) == 12
    assert {case.seed for case in cases} == {283101, 283202, 283303}
    assert {case.metadata["tier"] for case in cases} == {"low", "medium"}
    assert {case.first_player for case in cases} == {0, 1}


def test_round2_pipeline_preserves_lineage_and_runs_one_new_act(tmp_path):
    provider = FakeProvider()
    result = _pipeline(tmp_path, provider, FakeCalibrationEvaluator()).run()

    assert result.status == "complete"
    assert result.raw_score == 0.0
    assert result.evo_score_1 == 0.0
    assert result.evo_score_2 == 0.5
    assert result.gain_2 == 0.5
    assert result.calibration_score == 0.4
    assert result.global_act_count == 2
    assert result.round_act_count == 1
    assert provider.calls == 1
    assert (result.run_dir / "versions" / "v1" / "lineage.json").is_file()
    assert (result.run_dir / "versions" / "v2" / "manifest.json").is_file()
    assert (result.run_dir / "versions" / "v1-to-v2.patch").is_file()
    prompt_manifest = json.loads(
        (result.run_dir / "provider" / "prompt-manifest.json").read_text()
    )
    assert prompt_manifest["prompt_bytes"] <= 262_144
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["benchmark_id"] == "generals-hl-pilot-v1"
    assert summary["calibration_benchmark_id"] == "generals-hl-calibration-v1"
    assert summary["benchmark_score"] == 0.5
    assert summary["calibration_score"] == 0.4
    assert summary["budget"]["learning_coding_agent_acts"] == 1
    events = [
        json.loads(line)
        for line in (result.run_dir / "events.jsonl").read_text().splitlines()
    ]
    assert sum(event["event_type"] == "coding_agent_act" for event in events) == 1
    assert any(event["event_type"] == "lineage_import" for event in events)
    assert any(event["event_type"] == "behavior_change_episode" for event in events)


def test_round2_stops_before_codex_when_heldout_calibration_misses_target(tmp_path):
    provider = FakeProvider()
    result = _pipeline(
        tmp_path, provider, FakeCalibrationEvaluator(heldout_score=0.1)
    ).run()

    assert result.status == "calibration_failed"
    assert result.calibration_score == 0.1
    assert result.round_act_count == 0
    assert provider.calls == 0
    assert not (result.run_dir / "versions" / "v2").exists()


def test_development_calibration_freezes_selection_without_heldout_games(tmp_path):
    parent, manifest = _parent(tmp_path)
    root = tmp_path / "assets"
    source = root / "calibration"
    source.mkdir(parents=True)
    (source / "main.py").write_text("print('weak')\n")
    layout = AssetLayout(
        root=root,
        engine_root=root,
        baseline_root=root,
        opponents=tuple(replace(item, source=root) for item in CONFIG.opponents),
        engine_hash="engine-hash",
    )
    output = tmp_path / "calibration-v1-selection.toml"

    result = calibrate_development(
        config=CONFIG,
        assets=layout,
        calibration_config=CALIBRATION,
        calibration_source=source,
        parent_run_dir=parent,
        expected_parent_hash=manifest.content_hash,
        data_dir=tmp_path / "data",
        selection_output=output,
        evaluator=FakeCalibrationEvaluator(),
    )

    assert result.status == "complete"
    assert result.selected_mode == "passive"
    frozen = load_calibration_selection(output, CALIBRATION)
    assert frozen.selected_mode == "passive"
    assert frozen.source_hash != "a" * 64
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["budget"]["calibration_episodes"] == 30
    events = (result.run_dir / "events.jsonl").read_text()
    assert '"split":"heldout"' not in events
