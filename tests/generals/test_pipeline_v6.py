import json
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

from agentbench_frame.generals.assets import (
    load_pilot_config,
    load_round6_learning_config,
)
from agentbench_frame.generals.models import AssetLayout, ReplaySkillAsset
from agentbench_frame.generals.pipeline_v6 import GeneralsHLRound6Pipeline
from agentbench_frame.tracking.snapshot import LocalWorkspaceSnapshotter
from tests.generals.test_lineage_v6 import _parent
from tests.generals.test_pipeline_v4 import FakeEvaluator, RewritingProvider


FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_pilot_config(FIXTURES / "pilot-v1.toml")
LEARNING = load_round6_learning_config(
    FIXTURES / "v6-strongest-learning-v1.toml",
    CONFIG,
)


class Round6Provider(RewritingProvider):
    def __init__(
        self,
        *,
        fail=False,
        change_main=False,
        change_state_view=False,
        failing_test=False,
        strategy_symlink_target=None,
        document_symlink=None,
        document_symlink_target=None,
        post_test_main_tamper=False,
    ):
        super().__init__(fail=fail)
        self.change_main = change_main
        self.change_state_view = change_state_view
        self.failing_test = failing_test
        self.strategy_symlink_target = strategy_symlink_target
        self.document_symlink = document_symlink
        self.document_symlink_target = document_symlink_target
        self.post_test_main_tamper = post_test_main_tamper

    def invoke(self, context):
        result = super().invoke(context)
        workspace = Path(context["workspace_root"])
        if self.change_main:
            (workspace / "main.py").write_text(
                "V6_MAIN_TAMPER = True\n",
                encoding="utf-8",
            )
        if self.change_state_view:
            (workspace / "state_view.py").write_text(
                "def normalize(value):\n"
                "    return {**value, 'macro_budget': 2}\n",
                encoding="utf-8",
            )
        if self.failing_test:
            (workspace / "tests" / "test_v6_candidate.py").write_text(
                "def test_candidate():\n"
                "    assert False, 'synthetic candidate failure'\n",
                encoding="utf-8",
            )
        if self.strategy_symlink_target is not None:
            strategy = workspace / "strategy.py"
            strategy.unlink()
            strategy.symlink_to(self.strategy_symlink_target)
        if self.document_symlink is not None:
            document = workspace / self.document_symlink
            document.unlink()
            document.symlink_to(self.document_symlink_target)
        if self.post_test_main_tamper:
            (workspace / "tests" / "conftest.py").write_text(
                "from pathlib import Path\n"
                "def pytest_sessionfinish(session, exitstatus):\n"
                "    del session, exitstatus\n"
                "    (Path(__file__).parents[1] / 'main.py').write_text(\n"
                "        'POST_TEST_TAMPER = True\\n', encoding='utf-8'\n"
                "    )\n",
                encoding="utf-8",
            )
        return result


class ValidationExceptionEvaluator(FakeEvaluator):
    def evaluate(self, workspace, version, phase, run, cases=None):
        if version == "v6" and phase == "validation":
            selected = tuple(cases or ())
            self.calls.append((version, phase, len(selected)))
            raise RuntimeError("synthetic validation crash")
        return super().evaluate(
            workspace,
            version,
            phase,
            run,
            cases=cases,
        )


class ZeroFormalScoreEvaluator(FakeEvaluator):
    def evaluate(self, workspace, version, phase, run, cases=None):
        evaluation = super().evaluate(
            workspace,
            version,
            phase,
            run,
            cases=cases,
        )
        if version == "v6" and phase == "evaluation":
            return replace(
                evaluation,
                score=0.0,
                per_tier={"high": 0.0, "medium": 0.0, "low": 0.0},
            )
        return evaluation


class ProvenanceEvaluator(FakeEvaluator):
    def __init__(self):
        super().__init__()
        self.v6_workspaces = []

    def evaluate(self, workspace, version, phase, run, cases=None):
        if version == "v6":
            target = Path(workspace)
            self.v6_workspaces.append({
                "phase": phase,
                "path": target,
                "manifest": LocalWorkspaceSnapshotter().capture(target),
                "main_bytes": (target / "main.py").read_bytes(),
            })
        return super().evaluate(
            workspace,
            version,
            phase,
            run,
            cases=cases,
        )


class FormalExceptionEvaluator(FakeEvaluator):
    def evaluate(self, workspace, version, phase, run, cases=None):
        if version == "v6" and phase == "evaluation":
            selected = tuple(cases or ())
            self.calls.append((version, phase, len(selected)))
            raise RuntimeError("synthetic formal crash")
        return super().evaluate(
            workspace,
            version,
            phase,
            run,
            cases=cases,
        )


def _pipeline(
    tmp_path,
    provider,
    *,
    evaluator=None,
    incomplete_validation=False,
    prompt_max_bytes=131_072,
):
    parent, v5_manifest = _parent(tmp_path)
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
        text=skill_path.read_text(encoding="utf-8"),
        sha256="a" * 64,
    )
    selected_evaluator = evaluator or FakeEvaluator(
        improved=False,
        incomplete_validation=incomplete_validation,
    )
    pipeline = GeneralsHLRound6Pipeline(
        config=CONFIG,
        learning_config=LEARNING,
        assets=layout,
        replay_skill=skill,
        parent_run_dir=parent,
        expected_parent_hash=v5_manifest.content_hash,
        data_dir=tmp_path / "data",
        provider=provider,
        evaluator=selected_evaluator,
        rules_path=rules,
        prompt_max_bytes=prompt_max_bytes,
    )
    return pipeline, selected_evaluator


def _events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]


def _summary(run_dir):
    return json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )


def _assert_generic_episode_fields_are_missing(summary):
    for key in (
        "total_episodes",
        "total_steps",
        "total_reward",
        "win_rate",
        "wins",
        "losses",
        "draws",
    ):
        assert summary[key] is None


def test_v6_runs_high_learning_one_act_validation_and_formal(tmp_path):
    provider = Round6Provider(change_state_view=True)
    pipeline, evaluator = _pipeline(tmp_path, provider)

    result = pipeline.run()

    assert evaluator.calls == [
        ("v5", "learning", 6),
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    assert result.status == "complete"
    assert result.runnable is True
    assert result.global_act_count == 7
    assert result.round_act_count == 1
    assert result.evo_score_6 == 0.5
    assert result.gain_6 == 0.5
    required = (
        "versions/v5/manifest.json",
        "versions/v6/manifest.json",
        "versions/v5-to-v6.patch",
        "versions/v6/tests.log",
        "benchmark/v6-learning-spec.json",
        "benchmark/v6-validation-spec.json",
        "benchmark/formal-spec.json",
        "provider/codex-act-v6.prompt.md",
        "provider/codex-act-v6.prompt.json",
        "provider/prompt-manifest.json",
        "provider/codex-act-v6.raw.jsonl",
        "provider/codex-act-v6.stderr.log",
        "provider/feedback-receipt.json",
        "provider/provider-budget.json",
        "diagnostics/v5-learning-action-profile.json",
        "diagnostics/v6-validation-action-profile.json",
        "diagnostics/v6-formal-action-profile.json",
        "quality.json",
    )
    assert all((result.run_dir / relative).is_file() for relative in required)
    summary = _summary(result.run_dir)
    assert summary["score_history"] == [
        0.0,
        0.0,
        None,
        0.0,
        7 / 18,
        4 / 18,
        7 / 18,
        0.5,
    ]
    assert len(summary["learning_results"]) == 6
    assert len(summary["validation_results"]) == 12
    assert len(summary["formal_results"]) == 18
    assert summary["formal_scores"] == {
        "high": 0.5,
        "low": 0.5,
        "medium": 0.5,
    }
    assert summary["formal_high_score"] == 0.5
    assert summary["formal_medium_score"] == 0.5
    assert summary["formal_low_score"] == 0.5
    assert summary["success"] == {
        "high_breakthrough_at_least_1_of_6": True,
        "low_retained_6_of_6": False,
        "medium_at_least_2_of_6": True,
    }
    assert summary["medium_at_least_2_of_6"] is True
    assert summary["low_retained_6_of_6"] is False
    assert summary["high_breakthrough_at_least_1_of_6"] is True
    assert summary["budget"]["learning_episodes"] == 6
    assert summary["budget"]["validation_episodes"] == 12
    assert summary["budget"]["evaluation_episodes"] == 18
    assert summary["budget"]["learning_prompt_tokens"] == 100
    assert summary["budget"]["learning_completion_tokens"] == 25
    assert (
        summary["cumulative_learning_budget"]["learning_episodes"]
        == 70
    )
    assert (
        summary["cumulative_learning_budget"]["learning_prompt_tokens"]
        is None
    )
    assert summary["AUC_coding_agent_act"] is None
    _assert_generic_episode_fields_are_missing(summary)


def test_v6_incomplete_validation_does_not_block_formal_evaluation(tmp_path):
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(),
        incomplete_validation=True,
    )

    result = pipeline.run()

    assert result.status == "complete"
    assert evaluator.calls[-1] == ("v6", "evaluation", 18)
    summary = _summary(result.run_dir)
    assert summary["validation_status"] == "incomplete"
    assert summary["formal_evaluation_blocking"] is False


def test_v6_validation_exception_still_runs_formal_evaluation(tmp_path):
    evaluator = ValidationExceptionEvaluator()
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(),
        evaluator=evaluator,
    )

    result = pipeline.run()

    assert result.status == "complete"
    assert evaluator.calls[-2:] == [
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    summary = _summary(result.run_dir)
    assert summary["validation_results"] == []
    assert summary["validation_status"] == "error"
    assert summary["formal_results"]


def test_v6_provider_failure_retains_unrunnable_candidate_and_missing_scores(
    tmp_path,
):
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(fail=True),
    )

    result = pipeline.run()

    assert result.status == "provider_failed"
    assert result.runnable is False
    assert result.evo_score_6 is None
    assert evaluator.calls == [("v5", "learning", 6)]
    manifest = json.loads(
        (
            result.run_dir / "versions" / "v6" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["content_hash"]
    summary = _summary(result.run_dir)
    assert summary["validation_score"] is None
    assert summary["formal_score"] is None
    assert summary["score_history"][-1] is None
    assert summary["round_act_count"] == 1
    assert summary["budget"]["learning_prompt_tokens"] is None
    assert summary["budget"]["learning_completion_tokens"] is None
    assert summary["budget"]["learning_total_tokens"] is None
    _assert_generic_episode_fields_are_missing(summary)


def test_v6_candidate_test_failure_saves_source_and_skips_gameplay(tmp_path):
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(failing_test=True),
    )

    result = pipeline.run()

    assert result.status == "invalid_version"
    assert result.runnable is False
    assert evaluator.calls == [("v5", "learning", 6)]
    assert (
        result.run_dir
        / "versions"
        / "v6"
        / "source"
        / "tests"
        / "test_v6_candidate.py"
    ).is_file()
    assert "synthetic candidate failure" in (
        result.run_dir / "versions" / "v6" / "tests.log"
    ).read_text(encoding="utf-8")


def test_v6_prompt_omission_performs_zero_acts(tmp_path):
    provider = Round6Provider()
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        prompt_max_bytes=100,
    )

    result = pipeline.run()

    assert result.status == "prompt_incomplete"
    assert provider.calls == 0
    assert result.round_act_count == 0
    assert result.global_act_count == 6
    assert evaluator.calls == [("v5", "learning", 6)]
    summary = _summary(result.run_dir)
    assert summary["score_history"][-1] is None
    assert summary["budget"]["learning_prompt_tokens"] is None
    assert summary["budget"]["learning_completion_tokens"] is None
    _assert_generic_episode_fields_are_missing(summary)


def test_v6_main_change_invalidates_candidate(tmp_path):
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(change_main=True),
    )

    result = pipeline.run()

    assert result.status == "invalid_version"
    assert result.runnable is False
    assert evaluator.calls == [("v5", "learning", 6)]
    version = next(
        event
        for event in _events(result.run_dir)
        if event["event_type"] == "version"
        and event["version"] == "v6"
    )
    assert version["main_unchanged"] is False
    assert "main.py" in version["changed_files"]


def test_v6_state_view_change_is_allowed(tmp_path):
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(change_state_view=True),
    )

    result = pipeline.run()

    assert result.status == "complete"
    assert result.runnable is True
    assert evaluator.calls[-1] == ("v6", "evaluation", 18)
    version = next(
        event
        for event in _events(result.run_dir)
        if event["event_type"] == "version"
        and event["version"] == "v6"
    )
    assert version["scope_valid"] is True
    assert "state_view.py" in version["changed_files"]


def test_v6_symlinked_strategy_is_rejected_before_candidate_tests_or_gameplay(
    tmp_path,
):
    external_strategy = tmp_path / "external-strategy.py"
    external_strategy.write_text(
        "def choose_actions(round_number, my_seat, view):\n"
        "    del round_number, my_seat, view\n"
        "    return [[8]]\n",
        encoding="utf-8",
    )
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(strategy_symlink_target=external_strategy),
    )

    result = pipeline.run()

    assert result.status == "invalid_version"
    assert result.runnable is False
    assert evaluator.calls == [("v5", "learning", 6)]
    manifest = json.loads(
        (
            result.run_dir / "versions" / "v6" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert "strategy.py" not in manifest["files"]
    version = next(
        event
        for event in _events(result.run_dir)
        if event["event_type"] == "version"
        and event["version"] == "v6"
    )
    assert version["required_source_files_valid"] is False
    assert version["main_unchanged"] is True


@pytest.mark.parametrize("document", ["STRATEGY.md", "EXPERIENCE.md"])
def test_v6_symlinked_required_document_is_rejected(
    tmp_path,
    document,
):
    external_document = tmp_path / f"external-{document}"
    external_document.write_text("external evidence\n", encoding="utf-8")
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(
            document_symlink=document,
            document_symlink_target=external_document,
        ),
    )

    result = pipeline.run()

    assert result.status == "invalid_version"
    assert result.runnable is False
    assert evaluator.calls == [("v5", "learning", 6)]
    manifest = json.loads(
        (
            result.run_dir / "versions" / "v6" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert document not in manifest["files"]


def test_v6_tests_probes_and_gameplay_use_verified_frozen_copies(
    tmp_path,
    monkeypatch,
):
    evaluator = ProvenanceEvaluator()
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(post_test_main_tamper=True),
        evaluator=evaluator,
    )
    test_workspaces = []
    probe_workspaces = []
    real_baseline_tests = pipeline._baseline_tests
    real_probe_actions = pipeline._probe_actions

    def record_test_workspace(workspace):
        target = Path(workspace)
        test_workspaces.append({
            "path": target,
            "manifest": LocalWorkspaceSnapshotter().capture(target),
        })
        return real_baseline_tests(workspace)

    def record_probe_workspace(workspace, probes):
        target = Path(workspace)
        probe_workspaces.append({
            "path": target,
            "manifest": LocalWorkspaceSnapshotter().capture(target),
        })
        return real_probe_actions(workspace, probes)

    monkeypatch.setattr(
        pipeline,
        "_baseline_tests",
        record_test_workspace,
    )
    monkeypatch.setattr(
        pipeline,
        "_probe_actions",
        record_probe_workspace,
    )

    result = pipeline.run()

    assert result.status == "complete"
    assert result.runnable is True
    frozen_manifest = json.loads(
        (
            result.run_dir / "versions" / "v6" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    expected_hash = frozen_manifest["content_hash"]
    parent_main = (
        result.run_dir / "versions" / "v5" / "source" / "main.py"
    ).read_bytes()
    frozen_main = (
        result.run_dir / "versions" / "v6" / "source" / "main.py"
    ).read_bytes()
    assert frozen_main == parent_main
    assert len(test_workspaces) == 1
    assert test_workspaces[0]["manifest"].content_hash == expected_hash
    assert len(probe_workspaces) == 1
    assert probe_workspaces[0]["manifest"].content_hash == expected_hash
    assert [record["phase"] for record in evaluator.v6_workspaces] == [
        "validation",
        "evaluation",
    ]
    assert all(
        record["manifest"].content_hash == expected_hash
        for record in evaluator.v6_workspaces
    )
    assert all(
        record["main_bytes"] == parent_main
        for record in evaluator.v6_workspaces
    )
    isolated_paths = {
        test_workspaces[0]["path"],
        probe_workspaces[0]["path"],
        *(record["path"] for record in evaluator.v6_workspaces),
    }
    assert len(isolated_paths) == 4
    assert result.run_dir / "workspace" not in isolated_paths
    verification_events = [
        event
        for event in _events(result.run_dir)
        if event["event_type"] == "behavior_diagnostics"
        and event.get("diagnostic") == "frozen_source_verification"
    ]
    assert [event["phase"] for event in verification_events] == [
        "tests",
        "probes",
        "validation",
        "formal",
    ]
    assert all(event["verified"] is True for event in verification_events)
    assert all(
        event["actual_manifest_hash"] == expected_hash
        for event in verification_events
    )


def test_v6_candidate_test_timeout_is_retained_as_invalid(
    tmp_path,
    monkeypatch,
):
    pipeline, evaluator = _pipeline(tmp_path, Round6Provider())

    def timeout(_workspace):
        raise subprocess.TimeoutExpired(
            ["python", "-m", "pytest", "tests", "-q"],
            120,
            output="partial stdout\n",
            stderr="partial stderr\n",
        )

    monkeypatch.setattr(pipeline, "_baseline_tests", timeout)

    result = pipeline.run()

    assert result.status == "invalid_version"
    assert result.runnable is False
    assert evaluator.calls == [("v5", "learning", 6)]
    test_log = (
        result.run_dir / "versions" / "v6" / "tests.log"
    ).read_text(encoding="utf-8")
    assert "timeout" in test_log.lower()
    assert "partial stdout" in test_log
    assert "partial stderr" in test_log


def test_v6_formal_cases_never_enter_prompt_manifest(tmp_path):
    pipeline, _ = _pipeline(tmp_path, Round6Provider())

    result = pipeline.run()

    prompt = (
        result.run_dir / "provider" / "codex-act-v6.prompt.md"
    ).read_text(encoding="utf-8")
    manifest_text = (
        result.run_dir / "provider" / "prompt-manifest.json"
    ).read_text(encoding="utf-8")
    for seed in CONFIG.evaluation_seeds:
        assert str(seed) not in prompt
        assert str(seed) not in manifest_text
    assert "evaluation_cases" not in manifest_text
    assert "formal_cases" not in manifest_text


def test_v6_action_profile_failure_is_visible_without_interpolation(
    tmp_path,
    monkeypatch,
):
    from agentbench_frame.generals import pipeline_v6

    real_summarize = pipeline_v6.summarize_action_profile

    def fail_validation_profile(evaluation):
        if evaluation.version == "v6" and len(evaluation.results) == 12:
            raise ValueError("synthetic profile damage")
        return real_summarize(evaluation)

    monkeypatch.setattr(
        pipeline_v6,
        "summarize_action_profile",
        fail_validation_profile,
    )
    pipeline, evaluator = _pipeline(tmp_path, Round6Provider())

    result = pipeline.run()

    assert result.status == "complete"
    assert evaluator.calls[-1] == ("v6", "evaluation", 18)
    assert not (
        result.run_dir
        / "diagnostics"
        / "v6-validation-action-profile.json"
    ).exists()
    summary = _summary(result.run_dir)
    assert summary["action_profiles"]["validation"] is None
    assert summary["action_profiles"]["formal"] is not None
    profile_error = next(
        event
        for event in _events(result.run_dir)
        if event["event_type"] == "behavior_measurement_error"
        and event.get("diagnostic") == "action_profile"
        and event.get("phase") == "validation"
    )
    assert "synthetic profile damage" in profile_error["error"]
    quality = json.loads(
        (result.run_dir / "quality.json").read_text(encoding="utf-8")
    )
    assert quality["malformed_lines"] == 0


def test_v6_every_runnable_candidate_is_formally_evaluated_even_at_zero_score(
    tmp_path,
):
    evaluator = ZeroFormalScoreEvaluator()
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(),
        evaluator=evaluator,
    )

    result = pipeline.run()

    assert result.runnable is True
    assert result.status == "complete"
    assert result.evo_score_6 == 0.0
    assert evaluator.calls[-1] == ("v6", "evaluation", 18)
    summary = _summary(result.run_dir)
    assert summary["formal_score"] == 0.0
    assert summary["success"] == {
        "high_breakthrough_at_least_1_of_6": False,
        "low_retained_6_of_6": False,
        "medium_at_least_2_of_6": False,
    }


def test_v6_formal_exception_is_recorded_as_attempted_error(tmp_path):
    evaluator = FormalExceptionEvaluator()
    pipeline, evaluator = _pipeline(
        tmp_path,
        Round6Provider(),
        evaluator=evaluator,
    )

    result = pipeline.run()

    assert result.status == "formal_failed"
    assert result.runnable is True
    assert result.evo_score_6 is None
    assert evaluator.calls[-1] == ("v6", "evaluation", 18)
    summary = _summary(result.run_dir)
    assert summary["evaluation_status"] == "error"
    assert summary["formal_attempted"] is True
    assert "synthetic formal crash" in summary["formal_error"]
    assert summary["formal_score"] is None
    assert summary["score_history"][-1] is None
    assert (result.run_dir / "summary.json").is_file()
    assert (result.run_dir / "quality.json").is_file()
