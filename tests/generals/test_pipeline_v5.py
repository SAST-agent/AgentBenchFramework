import json
from pathlib import Path

from agentbench_frame.generals.assets import (
    load_pilot_config,
    load_round5_learning_config,
)
from agentbench_frame.generals.models import (
    AssetLayout,
    ReplaySkillAsset,
)
from agentbench_frame.generals.pipeline_v5 import (
    GeneralsHLRound5Pipeline,
)
from tests.generals.test_lineage_v5 import _parent
from tests.generals.test_pipeline_v4 import (
    FakeEvaluator,
    RewritingProvider,
)


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_pilot_config(FIXTURES / "pilot-v1.toml")
LEARNING = load_round5_learning_config(
    FIXTURES / "v5-rollback-learning-v1.toml",
    CONFIG,
)


def _pipeline(
    tmp_path,
    provider,
    *,
    incomplete_validation=False,
    prior_attempt=False,
    prompt_max_bytes=131_072,
):
    parent, v3_manifest, v4_manifest, receipt = _parent(tmp_path)
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
        improved=False,
        incomplete_validation=incomplete_validation,
    )
    prior_attempt_run_dir = None
    if prior_attempt:
        prior_attempt_run_dir = tmp_path / "prior-v5-prompt"
        prior_attempt_run_dir.mkdir()
        (prior_attempt_run_dir / "summary.json").write_text(
            json.dumps({
                "run_id": "prior-v5-prompt",
                "status": "prompt_incomplete",
                "parent_run_id": "parent-v4",
                "parent_version": "v4",
                "starting_version": "v3",
                "learning_id": LEARNING.learning_id,
                "act_count": 5,
                "round_act_count": 0,
                "budget": {
                    "learning_coding_agent_acts": 0,
                    "learning_episodes": 12,
                    "learning_env_steps": 24,
                    "learning_game_agent_decision_steps": 12,
                    "learning_primitive_commands": 24,
                    "learning_prompt_tokens": None,
                    "learning_completion_tokens": None,
                    "learning_total_tokens": None,
                    "learning_time_s": 2.0,
                },
            }),
            encoding="utf-8",
        )
    pipeline = GeneralsHLRound5Pipeline(
        config=CONFIG,
        learning_config=LEARNING,
        assets=layout,
        replay_skill=skill,
        parent_run_dir=parent,
        expected_parent_hash=v4_manifest.content_hash,
        expected_rollback_hash=v3_manifest.content_hash,
        campaign_budget_receipt=receipt,
        data_dir=tmp_path / "data",
        provider=provider,
        evaluator=evaluator,
        rules_path=rules,
        prior_attempt_run_dir=prior_attempt_run_dir,
        prompt_max_bytes=prompt_max_bytes,
    )
    return pipeline, evaluator


def _events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]


def test_v5_runs_paired_learning_one_act_validation_and_formal(
    tmp_path,
):
    provider = RewritingProvider(change_behavior=False)
    pipeline, evaluator = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert result.status == "complete"
    assert result.evo_score_5 == 0.5
    assert result.gain_5 == 0.5
    assert result.global_act_count == 6
    assert result.round_act_count == 1
    assert result.runnable is True
    assert evaluator.calls == [
        ("v3", "learning", 6),
        ("v4", "learning", 6),
        ("v5", "validation", 6),
        ("v5", "evaluation", 18),
    ]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["parent_version"] == "v4"
    assert summary["starting_version"] == "v3"
    assert summary["rollback_source_version"] == "v3"
    assert summary["score_history"] == [
        0.0,
        0.0,
        None,
        0.0,
        7 / 18,
        4 / 18,
        0.5,
    ]
    assert summary["feedback_read"]["episodes"] == 12
    assert summary["feedback_read"]["decision_records"] == 12
    assert summary["budget"]["learning_episodes"] == 12
    assert summary["budget"]["validation_episodes"] == 6
    assert summary["budget"]["evaluation_episodes"] == 18
    assert summary["cumulative_learning_budget"]["learning_episodes"] == 70
    assert (
        result.run_dir / "versions" / "v3-to-v5.patch"
    ).is_file()
    assert (
        result.run_dir / "versions" / "v4-to-v5.patch"
    ).is_file()
    assert any(
        event["event_type"] == "version_rollback"
        for event in _events(result.run_dir)
    )


def test_v5_incomplete_validation_does_not_block_formal_evaluation(
    tmp_path,
):
    provider = RewritingProvider(change_behavior=True)
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        incomplete_validation=True,
    )

    result = pipeline.run()

    assert result.status == "complete"
    assert evaluator.calls[-1] == ("v5", "evaluation", 18)
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["behavior_diagnostics"]["validation_complete"] is False
    assert (
        summary["behavior_diagnostics"]["formal_evaluation_blocking"]
        is False
    )


def test_v5_provider_failure_retains_candidate_and_missing_score(
    tmp_path,
):
    provider = RewritingProvider(fail=True)
    pipeline, evaluator = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert result.status == "provider_failed"
    assert result.runnable is False
    assert result.evo_score_5 is None
    assert evaluator.calls == [
        ("v3", "learning", 6),
        ("v4", "learning", 6),
    ]
    assert (
        result.run_dir / "versions" / "v5" / "manifest.json"
    ).is_file()
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["score_history"][-1] is None
    assert summary["evaluation_status"] == "not_run"


def test_v5_prompt_omission_finalizes_zero_act_learning_run(tmp_path):
    provider = RewritingProvider()
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        prompt_max_bytes=100,
    )

    result = pipeline.run()

    assert result.status == "prompt_incomplete"
    assert provider.calls == 0
    assert evaluator.calls == [
        ("v3", "learning", 6),
        ("v4", "learning", 6),
    ]
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["round_act_count"] == 0
    assert summary["budget"]["learning_episodes"] == 12
    assert summary["score_history"][-1] is None


def test_v5_cumulative_budget_includes_prior_zero_act_attempt(tmp_path):
    provider = RewritingProvider()
    pipeline, _ = _pipeline(
        tmp_path,
        provider,
        prior_attempt=True,
    )

    result = pipeline.run()

    assert result.status == "complete"
    summary = json.loads((result.run_dir / "summary.json").read_text())
    assert summary["prior_attempt_run_id"] == "prior-v5-prompt"
    assert summary["cumulative_learning_budget"]["learning_episodes"] == 82
    assert (
        summary["cumulative_learning_budget"][
            "learning_coding_agent_acts"
        ]
        == 6
    )
