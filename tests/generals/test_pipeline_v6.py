import json
from dataclasses import replace
import hashlib
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


class ProviderOrderingEvaluator(FakeEvaluator):
    def __init__(self, provider):
        super().__init__(improved=False)
        self.provider = provider
        self.provider_calls_by_phase = []

    def evaluate(self, workspace, version, phase, run, cases=None):
        self.provider_calls_by_phase.append(
            (version, phase, self.provider.calls)
        )
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


def _inject_forbidden_static_context(
    pipeline: GeneralsHLRound6Pipeline,
    context_name: str,
) -> None:
    marker = "validation trajectory from a held-out split"
    if context_name == "rules":
        pipeline.rules_path.write_text(marker, encoding="utf-8")
        return
    if context_name == "skill":
        pipeline.replay_skill = replace(
            pipeline.replay_skill,
            text=marker,
            sha256=hashlib.sha256(marker.encode("utf-8")).hexdigest(),
        )
        return

    filename = {
        "strategy": "strategy.py",
        "experience": "EXPERIENCE.md",
    }[context_name]
    source = (
        pipeline.parent_run_dir / "versions" / "v5" / "source"
    )
    (source / filename).write_text(marker, encoding="utf-8")
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(
        manifest,
        pipeline.parent_run_dir / "versions" / "v5" / "manifest.json",
    )
    pipeline.expected_parent_hash = manifest.content_hash


def _inject_round5_experience(
    pipeline: GeneralsHLRound6Pipeline,
) -> None:
    source = (
        pipeline.parent_run_dir / "versions" / "v5" / "source"
    )
    (source / "EXPERIENCE.md").write_text(
        "# Retained V5 experience\n\n"
        "- `learn5-high-s286101-p0` / "
        "`v3:learn5-high-s286101-p0-d64`\n"
        "- `learn5-high-s286202-p1` / "
        "`v3:learn5-high-s286202-p1-d179`\n"
        "- `learn5-high-s286303-p0` / "
        "`v3:learn5-high-s286303-p0-d27`\n",
        encoding="utf-8",
    )
    snapshotter = LocalWorkspaceSnapshotter()
    manifest = snapshotter.capture(source)
    snapshotter.write_manifest(
        manifest,
        pipeline.parent_run_dir / "versions" / "v5" / "manifest.json",
    )
    pipeline.expected_parent_hash = manifest.content_hash


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


def _write_excluded_poison(source: Path) -> None:
    (source / "__pycache__").mkdir(parents=True)
    (source / "__pycache__" / "poison.pyc").write_bytes(b"poison cache")
    (source / ".venv").mkdir(parents=True)
    (source / ".venv" / "poison.py").write_text(
        "raise RuntimeError('poison')\n",
        encoding="utf-8",
    )


def _assert_excluded_poison_absent(target: Path) -> None:
    assert not (target / "__pycache__" / "poison.pyc").exists()
    assert not (target / ".venv" / "poison.py").exists()


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


def test_v6_seed_bearing_parent_reaches_provider_before_post_act_games(
    tmp_path,
):
    provider = Round6Provider()
    evaluator = ProviderOrderingEvaluator(provider)
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        evaluator=evaluator,
    )
    _inject_round5_experience(pipeline)

    result = pipeline.run()

    assert result.status == "complete"
    assert provider.calls == 1
    assert evaluator.provider_calls_by_phase == [
        ("v5", "learning", 0),
        ("v6", "validation", 1),
        ("v6", "evaluation", 1),
    ]
    prompt = (
        result.run_dir / "provider" / "codex-act-v6.prompt.md"
    ).read_text(encoding="utf-8")
    assert "learn5-high-s286101-p0" in prompt
    assert "v3:learn5-high-s286101-p0-d64" in prompt


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


@pytest.mark.parametrize(
    "context_name",
    ("rules", "skill", "strategy", "experience"),
)
def test_v6_rejects_forbidden_static_context_before_learning(
    tmp_path,
    context_name,
):
    provider = Round6Provider()
    pipeline, evaluator = _pipeline(tmp_path, provider)
    _inject_forbidden_static_context(pipeline, context_name)

    result = pipeline.run()

    assert result.status == "failed"
    assert evaluator.calls == []
    assert provider.calls == 0
    assert result.round_act_count == 0
    summary = _summary(result.run_dir)
    assert summary["budget"]["learning_episodes"] == 0
    assert summary["budget"]["learning_coding_agent_acts"] == 0
    assert summary["learning_results"] == []
    assert "forbidden round-6 formal or validation material" in summary["error"]


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


def test_v6_isolation_materializes_only_frozen_manifest_files(tmp_path):
    pipeline, _ = _pipeline(tmp_path, Round6Provider())
    frozen_source = (
        pipeline.parent_run_dir / "versions" / "v5" / "source"
    )
    manifest = pipeline.snapshotter.capture(frozen_source)
    _write_excluded_poison(frozen_source)
    isolated = tmp_path / "isolated"

    materialized = pipeline._materialize_verified_source(
        frozen_source,
        isolated,
        manifest,
    )

    assert materialized == isolated
    assert (isolated / "strategy.py").is_file()
    _assert_excluded_poison_absent(isolated)


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


def test_v6_prompt_incomplete_recovery_rebuilds_and_hashes_prompt_in_new_run(
    tmp_path,
):
    provider = Round6Provider()
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        prompt_max_bytes=100,
    )
    failed = pipeline.run()
    failed_summary_before = (failed.run_dir / "summary.json").read_bytes()
    failed_events_before = (failed.run_dir / "events.jsonl").read_bytes()
    assert failed.status == "prompt_incomplete"
    assert provider.calls == 0

    pipeline.prompt_max_bytes = 131_072
    recovered = pipeline.recover(failed.run_dir)

    assert recovered.run_dir != failed.run_dir
    assert recovered.status == "complete"
    assert recovered.global_act_count == 7
    assert recovered.round_act_count == 1
    assert provider.calls == 1
    assert evaluator.calls == [
        ("v5", "learning", 6),
        ("v5", "learning", 6),
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    assert (failed.run_dir / "summary.json").read_bytes() == (
        failed_summary_before
    )
    assert (failed.run_dir / "events.jsonl").read_bytes() == (
        failed_events_before
    )
    prompt = (
        recovered.run_dir / "provider" / "codex-act-v6.prompt.md"
    ).read_bytes()
    prompt_manifest = json.loads(
        (
            recovered.run_dir / "provider" / "prompt-manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert prompt_manifest["prompt_sha256"] == hashlib.sha256(
        prompt
    ).hexdigest()
    summary = _summary(recovered.run_dir)
    assert summary["failed_run_id"] == _summary(failed.run_dir)["run_id"]
    assert summary["recovery_mode"] == "prompt_rebuild"
    assert summary["source_learning_budget"]["learning_episodes"] == 6
    assert (
        summary["cumulative_learning_budget"]["learning_episodes"]
        == 76
    )
    recovery = next(
        event
        for event in _events(recovered.run_dir)
        if event["event_type"] == "recovery_import"
    )
    assert recovery["recovery_mode"] == "prompt_rebuild"
    assert recovery["reused_prompt"] is False


def test_v6_provider_failure_recovery_reuses_verified_inputs_as_new_act(
    tmp_path,
):
    failed_provider = Round6Provider(fail=True)
    pipeline, evaluator = _pipeline(tmp_path, failed_provider)
    failed = pipeline.run()
    failed_summary_before = (failed.run_dir / "summary.json").read_bytes()
    failed_events_before = (failed.run_dir / "events.jsonl").read_bytes()
    assert failed.status == "provider_failed"
    source_prompt = (
        failed.run_dir / "provider" / "codex-act-v6.prompt.md"
    ).read_bytes()

    retry_provider = Round6Provider(change_state_view=True)
    pipeline.provider = retry_provider
    recovered = pipeline.recover(failed.run_dir)

    assert recovered.run_dir != failed.run_dir
    assert recovered.status == "complete"
    assert recovered.runnable is True
    assert recovered.global_act_count == 8
    assert recovered.round_act_count == 1
    assert retry_provider.calls == 1
    assert evaluator.calls == [
        ("v5", "learning", 6),
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    assert (
        recovered.run_dir / "provider" / "codex-act-v6.prompt.md"
    ).read_bytes() == source_prompt
    assert (failed.run_dir / "summary.json").read_bytes() == (
        failed_summary_before
    )
    assert (failed.run_dir / "events.jsonl").read_bytes() == (
        failed_events_before
    )
    summary = _summary(recovered.run_dir)
    assert summary["failed_run_id"] == _summary(failed.run_dir)["run_id"]
    assert summary["recovery_mode"] == "provider_retry"
    assert summary["act_count"] == 8
    assert summary["round_act_count"] == 1
    assert (
        summary["cumulative_learning_budget"][
            "learning_coding_agent_acts"
        ]
        == 8
    )
    events = _events(recovered.run_dir)
    assert sum(
        event["event_type"] == "coding_agent_act" for event in events
    ) == 1
    retry = next(
        event for event in events
        if event["event_type"] == "provider_retry"
    )
    assert retry["reused_prompt"] is True
    assert retry["reused_learning_evidence"] is True


def test_v6_post_act_recovery_uses_verified_frozen_source_and_zero_acts(
    tmp_path,
):
    provider = Round6Provider(change_state_view=True)
    evaluator = FormalExceptionEvaluator()
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        evaluator=evaluator,
    )
    failed = pipeline.run()
    failed_summary_before = (failed.run_dir / "summary.json").read_bytes()
    failed_events_before = (failed.run_dir / "events.jsonl").read_bytes()
    assert failed.status == "formal_failed"
    assert failed.runnable is True
    provider_calls = provider.calls
    frozen_manifest = json.loads(
        (
            failed.run_dir / "versions" / "v6" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    _write_excluded_poison(
        failed.run_dir / "versions" / "v6" / "source"
    )

    recovered_evaluator = ProvenanceEvaluator()
    pipeline.evaluator = recovered_evaluator
    recovered = pipeline.recover(failed.run_dir)

    assert recovered.run_dir != failed.run_dir
    assert recovered.status == "complete"
    assert recovered.runnable is True
    assert recovered.global_act_count == 7
    assert recovered.round_act_count == 0
    assert provider.calls == provider_calls
    assert recovered_evaluator.calls == [
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    assert (failed.run_dir / "summary.json").read_bytes() == (
        failed_summary_before
    )
    assert (failed.run_dir / "events.jsonl").read_bytes() == (
        failed_events_before
    )
    summary = _summary(recovered.run_dir)
    assert summary["failed_run_id"] == _summary(failed.run_dir)["run_id"]
    assert summary["recovery_mode"] == "post_act_evaluation"
    assert summary["act_count"] == 7
    assert summary["round_act_count"] == 0
    assert len(summary["validation_results"]) == 12
    assert len(summary["formal_results"]) == 18
    assert not any(
        event["event_type"] == "coding_agent_act"
        for event in _events(recovered.run_dir)
    )
    expected_hash = frozen_manifest["content_hash"]
    verification_events = [
        event
        for event in _events(recovered.run_dir)
        if event["event_type"] == "behavior_diagnostics"
        and event.get("diagnostic") == "frozen_source_verification"
        and event.get("phase") in {"validation", "formal"}
    ]
    assert [event["phase"] for event in verification_events] == [
        "validation",
        "formal",
    ]
    assert all(
        event["actual_manifest_hash"] == expected_hash
        and event["verified"] is True
        for event in verification_events
    )
    assert len({
        record["path"] for record in recovered_evaluator.v6_workspaces
    }) == 2
    _assert_excluded_poison_absent(recovered.run_dir / "workspace")
    _assert_excluded_poison_absent(
        recovered.run_dir / "versions" / "v6" / "source"
    )


@pytest.mark.parametrize(
    "mismatch",
    [
        "parent",
        "v6",
        "prompt",
        "learning_id",
        "learning",
        "terminal",
    ],
)
def test_v6_recovery_rejects_mismatched_audited_source_before_side_effects(
    tmp_path,
    mismatch,
):
    provider = Round6Provider(change_state_view=True)
    pipeline, _ = _pipeline(
        tmp_path,
        provider,
        evaluator=FormalExceptionEvaluator(),
    )
    failed = pipeline.run()
    assert failed.status == "formal_failed"
    summary_path = failed.run_dir / "summary.json"
    summary = _summary(failed.run_dir)
    if mismatch == "parent":
        summary["parent_run_id"] = "different-parent"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    elif mismatch == "v6":
        (
            failed.run_dir
            / "versions"
            / "v6"
            / "source"
            / "strategy.py"
        ).write_text("tampered recovery source\n", encoding="utf-8")
    elif mismatch == "prompt":
        prompt_path = (
            failed.run_dir / "provider" / "codex-act-v6.prompt.md"
        )
        prompt = prompt_path.read_text(encoding="utf-8")
        prompt_path.write_text(
            ("X" if prompt[0] != "X" else "Y") + prompt[1:],
            encoding="utf-8",
        )
    elif mismatch == "learning_id":
        summary["learning_id"] = "different-learning-suite"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    elif mismatch == "learning":
        summary["learning_results"][0]["valid"] = False
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        summary["status"] = "incomplete"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    provider_calls = provider.calls
    pipeline.evaluator = ProvenanceEvaluator()

    with pytest.raises(ValueError, match=mismatch.replace("_", " ")):
        pipeline.recover(failed.run_dir)

    assert provider.calls == provider_calls
    assert pipeline.evaluator.calls == []


def test_v6_provider_retry_failure_can_be_recovered_as_another_visible_act(
    tmp_path,
):
    first_provider = Round6Provider(fail=True)
    pipeline, evaluator = _pipeline(tmp_path, first_provider)
    first = pipeline.run()
    assert first.status == "provider_failed"
    assert first.global_act_count == 7

    second_provider = Round6Provider(fail=True)
    pipeline.provider = second_provider
    second = pipeline.recover(first.run_dir)
    assert second.status == "provider_failed"
    assert second.global_act_count == 8
    assert second.round_act_count == 1

    third_provider = Round6Provider()
    pipeline.provider = third_provider
    third = pipeline.recover(second.run_dir)

    assert third.status == "complete"
    assert third.global_act_count == 9
    assert third.round_act_count == 1
    assert first_provider.calls == 1
    assert second_provider.calls == 1
    assert third_provider.calls == 1
    assert evaluator.calls == [
        ("v5", "learning", 6),
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    summary = _summary(third.run_dir)
    assert summary["failed_run_id"] == _summary(second.run_dir)["run_id"]
    assert (
        summary["cumulative_learning_budget"][
            "learning_coding_agent_acts"
        ]
        == 9
    )


def test_v6_post_act_failure_can_be_recovered_again_without_provider(
    tmp_path,
):
    provider = Round6Provider()
    pipeline, _ = _pipeline(
        tmp_path,
        provider,
        evaluator=FormalExceptionEvaluator(),
    )
    first = pipeline.run()
    assert first.status == "formal_failed"
    provider_calls = provider.calls

    pipeline.evaluator = FormalExceptionEvaluator()
    second = pipeline.recover(first.run_dir)
    assert second.status == "formal_failed"
    assert second.global_act_count == 7
    assert second.round_act_count == 0

    final_evaluator = ProvenanceEvaluator()
    pipeline.evaluator = final_evaluator
    third = pipeline.recover(second.run_dir)

    assert third.status == "complete"
    assert third.global_act_count == 7
    assert third.round_act_count == 0
    assert provider.calls == provider_calls
    assert final_evaluator.calls == [
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    assert not any(
        event["event_type"] == "coding_agent_act"
        for event in _events(second.run_dir) + _events(third.run_dir)
    )


def test_v6_repeated_prompt_incomplete_records_zero_new_acts_then_recovers(
    tmp_path,
):
    provider = Round6Provider()
    pipeline, evaluator = _pipeline(
        tmp_path,
        provider,
        prompt_max_bytes=100,
    )
    first = pipeline.run()
    assert first.status == "prompt_incomplete"

    second = pipeline.recover(first.run_dir)
    assert second.status == "prompt_incomplete"
    assert second.global_act_count == 6
    assert second.round_act_count == 0
    recovery = next(
        event
        for event in _events(second.run_dir)
        if event["event_type"] == "recovery_import"
    )
    assert recovery["new_provider_act"] is False

    pipeline.prompt_max_bytes = 131_072
    third = pipeline.recover(second.run_dir)

    assert third.status == "complete"
    assert third.global_act_count == 7
    assert third.round_act_count == 1
    assert provider.calls == 1
    assert evaluator.calls == [
        ("v5", "learning", 6),
        ("v5", "learning", 6),
        ("v5", "learning", 6),
        ("v6", "validation", 12),
        ("v6", "evaluation", 18),
    ]
    assert (
        _summary(third.run_dir)["cumulative_learning_budget"][
            "learning_episodes"
        ]
        == 82
    )


def test_v6_recovery_chain_binds_declared_mode_to_failed_run_kind(
    tmp_path,
):
    pipeline, _ = _pipeline(tmp_path, Round6Provider(fail=True))
    first = pipeline.run()
    pipeline.provider = Round6Provider(fail=True)
    second = pipeline.recover(first.run_dir)
    assert second.status == "provider_failed"

    summary_path = second.run_dir / "summary.json"
    summary = _summary(second.run_dir)
    summary["recovery_mode"] = "prompt_rebuild"
    summary["config"]["recovery_mode"] = "prompt_rebuild"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    events = _events(second.run_dir)
    for event in events:
        if event["event_type"] == "recovery_import":
            event["recovery_mode"] = "prompt_rebuild"
    (second.run_dir / "events.jsonl").write_text(
        "".join(
            json.dumps(event, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    pipeline.provider = Round6Provider()

    with pytest.raises(ValueError, match="recovery mode"):
        pipeline.recover(second.run_dir)
