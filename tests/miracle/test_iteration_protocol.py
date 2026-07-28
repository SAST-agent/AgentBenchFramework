import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


def _protocol():
    from agentbench_frame.games.miracle import iteration_protocol

    return iteration_protocol


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, monkeypatch, *, role="train"):
    protocol = _protocol()
    bootstrap_path = tmp_path / "bootstrap.py"
    bootstrap_path.write_bytes(protocol.default_bootstrap_template().path.read_bytes())
    bootstrap = protocol.BootstrapTemplate(
        protocol.BOOTSTRAP_TEMPLATE_VERSION,
        bootstrap_path,
        protocol.APPROVED_BOOTSTRAP_SHA256,
    )
    champion_asset = tmp_path / "champion.py"
    champion_asset.write_text("# fake-only champion\n", encoding="utf-8")
    champion = protocol.HumanChampion.create(
        logical_id="fake-champion",
        version="champion-v1",
        artifact_path=champion_asset.name,
        artifact_sha256=_sha(champion_asset),
        provenance="fake-only fixture",
        qualification_status="qualified",
        qualification_evidence=("fake-review",),
        protocol_version=protocol.PROTOCOL_VERSION,
        benchmark_version=protocol.BENCHMARK_VERSION,
        frozen=True,
    )
    champion_path = tmp_path / "champion.json"
    champion_path.write_bytes(champion.canonical_bytes())

    human_skill = protocol.HumanReplaySkill.create(
        logical_id="human-reader",
        version="reader-v1",
        content={"instructions": ["read fake evidence"]},
        provenance="fake-only fixture",
        created_by="fake-reviewer",
    )
    human_skill_path = tmp_path / "human-skill.json"
    human_skill_path.write_bytes(human_skill.canonical_bytes())

    case = protocol.MatchPlanCase(
        "case-1", role, 0, 1, 0, 1, 11, 12, 13
    )
    plan = protocol.MatchPlanManifest.create(
        champion_descriptor_sha256=champion.sha256,
        human_replay_skill_sha256=human_skill.sha256,
        evaluated_policy_version="strategy-v0",
        evaluated_policy_source_sha256=protocol.bootstrap_strategy_source_sha256(bootstrap),
        cases=(case,),
        bootstrap_sha256=protocol.APPROVED_BOOTSTRAP_SHA256,
        protocol_version=protocol.PROTOCOL_VERSION,
        benchmark_version=protocol.BENCHMARK_VERSION,
        iteration_protocol_version=protocol.ITERATION_PROTOCOL_VERSION,
        research_manifest_sha256=protocol.research_manifest_sha256(),
        iteration_manifest_sha256=hashlib.sha256(
            protocol.canonical_iteration_protocol_manifest_bytes()
        ).hexdigest(),
    )
    plan_path = tmp_path / "match-plan.json"
    plan_path.write_bytes(plan.canonical_bytes())

    replay = tmp_path / "replay.bin"
    replay.write_bytes(b"fake replay")
    evidence = protocol.CapturedReplay.from_plan_case(
        evidence_id="evidence-1",
        artifact_path=replay.name,
        artifact_sha256=_sha(replay),
        game="24_miracle",
        case=case,
        plan=plan,
    )
    replay_manifest = protocol.ReplayEvidenceManifest.create(
        match_plan_sha256=plan.sha256,
        evidence=(evidence,),
        protocol_version=protocol.PROTOCOL_VERSION,
        benchmark_version=protocol.BENCHMARK_VERSION,
        iteration_protocol_version=protocol.ITERATION_PROTOCOL_VERSION,
    )
    replay_manifest_path = tmp_path / "replay-evidence.json"
    replay_manifest_path.write_bytes(replay_manifest.canonical_bytes())

    monkeypatch.setattr(protocol, "CURRENT_APPROVED_HUMAN_CHAMPION_SHA256", champion.sha256)
    monkeypatch.setattr(protocol, "APPROVED_HUMAN_REPLAY_SKILL_SHA256", frozenset({human_skill.sha256}))
    monkeypatch.setattr(protocol, "APPROVED_MATCH_PLAN_MANIFEST_SHA256", frozenset({plan.sha256}))
    monkeypatch.setattr(
        protocol,
        "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256",
        frozenset({replay_manifest.sha256}),
    )
    match_config = protocol.MatchConfig(
        "24_miracle", bootstrap, champion_path, human_skill_path, plan_path
    )
    config = protocol.LearningConfig(
        match_config, replay_manifest_path
    )
    return {
        "config": config,
        "match_config": match_config,
        "bootstrap_path": bootstrap_path,
        "champion": champion,
        "human_skill": human_skill,
        "human_skill_path": human_skill_path,
        "case": case,
        "evidence": evidence,
        "plan": plan,
        "manifest": replay_manifest,
    }


def _strategy(
    tmp_path: Path,
    *,
    version="strategy-v1",
    parent="strategy-v0",
    category="finite_state_machine",
    context=None,
    evidence=(),
):
    protocol = _protocol()
    source = tmp_path / version
    source.mkdir()
    (source / "main.py").write_text(
        "STATE = 'safe'\ndef act(observation, support):\n    return support.actions[0].action\n",
        encoding="utf-8",
    )
    return protocol.StrategyVersion.from_directory(
        source,
        version=version,
        parent_version=parent,
        entrypoint="main.py:act",
        dependencies=(),
        change_plan=protocol.ChangePlan(
            (protocol.ChangeOperation("simplify", "fallback"),),
            training_evidence=evidence,
        ),
        interpretability_category=category,
        probability_query="read-only-complete-action-support",
        control_context=context,
    )


def test_default_bootstrap_is_fixed_minimal_template_not_legacy_ifelse():
    protocol = _protocol()
    template = protocol.default_bootstrap_template()
    protocol.validate_bootstrap_template(template)
    assert template.sha256 == protocol.APPROVED_BOOTSTRAP_SHA256 == _sha(template.path)
    text = template.path.read_text(encoding="utf-8")
    assert "minimal legal fallback" in text
    assert "rank04" not in text and "rank09" not in text and "ifelse" not in template.path.name.lower()
    assert protocol.canonical_iteration_protocol_manifest_bytes() == protocol.canonical_iteration_protocol_manifest_bytes()


def test_iteration_tool_defaults_to_manifest_preflight_not_legacy_ifelse():
    tool = Path(__file__).resolve().parents[2] / "tools" / "miracle_iteration.py"
    spec = importlib.util.spec_from_file_location("miracle_iteration_tool", tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parser = module.build_research_parser()
    subparsers = next(action for action in parser._actions if action.dest == "stage")
    assert set(subparsers.choices) == {"plan", "replay", "learning"}
    assert "legacy-smoke" not in subparsers.choices


def test_iteration_cli_preflight_failure_is_fatal_without_traceback(tmp_path, capsys):
    tool = Path(__file__).resolve().parents[2] / "tools" / "miracle_iteration.py"
    spec = importlib.util.spec_from_file_location("miracle_iteration_cli", tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    missing = tmp_path / "missing.json"
    result = module.main(
        [
            "plan",
            "--champion-manifest", str(missing),
            "--human-replay-skill", str(missing),
            "--match-plan", str(missing),
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err.startswith("FATAL: ")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("stage", ["plan", "replay", "learning"])
def test_iteration_cli_stages_fail_closed_with_empty_production_approvals(
    tmp_path, monkeypatch, capsys, stage
):
    tool = Path(__file__).resolve().parents[2] / "tools" / "miracle_iteration.py"
    spec = importlib.util.spec_from_file_location(f"miracle_iteration_cli_{stage}", tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with monkeypatch.context() as temporary_approvals:
        bundle = _bundle(tmp_path, temporary_approvals)
    args = [
        stage,
        "--champion-manifest",
        str(bundle["match_config"].champion_manifest_path),
        "--human-replay-skill",
        str(bundle["human_skill_path"]),
        "--match-plan",
        str(bundle["match_config"].match_plan_path),
    ]
    if stage != "plan":
        args.extend(
            [
                "--replay-evidence-manifest",
                str(bundle["config"].replay_evidence_manifest_path),
            ]
        )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert module.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("FATAL: ")
    assert "Traceback" not in captured.err
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    ("stage", "expected_state"),
    [
        ("plan", "MATCH_READY"),
        ("replay", "REPLAY_APPROVED"),
        ("learning", "LEARNING_READY"),
    ],
)
def test_iteration_cli_stages_accept_test_only_approved_fake_inputs(
    tmp_path, monkeypatch, capsys, stage, expected_state
):
    bundle = _bundle(tmp_path, monkeypatch)
    tool = Path(__file__).resolve().parents[2] / "tools" / "miracle_iteration.py"
    spec = importlib.util.spec_from_file_location(f"miracle_iteration_ok_{stage}", tool)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = [
        stage,
        "--champion-manifest",
        str(bundle["match_config"].champion_manifest_path),
        "--human-replay-skill",
        str(bundle["human_skill_path"]),
        "--match-plan",
        str(bundle["match_config"].match_plan_path),
    ]
    if stage != "plan":
        args.extend(
            [
                "--replay-evidence-manifest",
                str(bundle["config"].replay_evidence_manifest_path),
            ]
        )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert module.main(args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload == {
        "authoritative_execution": "blocked",
        "lifecycle_state": expected_state,
        "stage": stage,
        "status": "fake-only preflight complete",
    }
    assert captured.err == ""
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_unapproved_champion_failure_creates_no_runner_session_or_log(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(protocol, "CURRENT_APPROVED_HUMAN_CHAMPION_SHA256", None)
    effects = []
    with pytest.raises(protocol.IterationPreflightError, match="champion.*approved"):
        protocol.open_iteration_after_preflight(
            bundle["config"],
            runner_factory=lambda _: effects.append("runner"),
            session_factory=lambda _: effects.append("session"),
            log_factory=lambda _: effects.append("log"),
        )
    assert effects == []


@pytest.mark.parametrize("kind", ["missing", "wrong-version", "wrong-sha"])
def test_bootstrap_identity_failure_creates_no_runner_session_or_log(tmp_path, monkeypatch, kind):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch)
    template = bundle["match_config"].bootstrap
    if kind == "missing":
        template = replace(template, path=tmp_path / "missing.py")
    elif kind == "wrong-version":
        template = replace(template, version="wrong")
    else:
        template = replace(template, sha256="0" * 64)
    effects = []
    with pytest.raises(protocol.IterationPreflightError, match="bootstrap"):
        protocol.open_match_after_preflight(
            replace(bundle["match_config"], bootstrap=template),
            store=protocol.ImmutableIterationStore(tmp_path / "store"),
            runner_factory=lambda _: effects.append("runner"),
        )
    assert effects == []


def test_no_rank04_or_rank09_fallback_is_in_default_config(tmp_path, monkeypatch):
    config = _bundle(tmp_path, monkeypatch)["match_config"]
    assert not hasattr(config, "training_opponent_fallback")
    assert not hasattr(config, "validation_opponent_fallback")


@pytest.mark.parametrize("category", ["finite_state_machine", "scoring_function"])
def test_interpretable_fsm_and_scoring_strategies_are_accepted(tmp_path, category):
    strategy = _strategy(tmp_path, category=category)
    strategy.validate()
    assert strategy.complexity == _protocol().compute_complexity_metrics(strategy.source_files)


@pytest.mark.parametrize(
    "changes",
    [
        {"network_dependencies": ("https://example.invalid/model",)},
        {"binary_models": ("opaque.bin",)},
        {"external_state": ("unregistered-cache",)},
    ],
)
def test_network_opaque_binary_and_unregistered_state_are_rejected(tmp_path, changes):
    strategy = replace(_strategy(tmp_path), **changes)
    with pytest.raises(ValueError):
        strategy.validate()


def test_network_import_hidden_in_source_is_rejected(tmp_path):
    strategy = _strategy(tmp_path)
    contaminated = replace(
        strategy,
        source_files={"main.py": "import socket\ndef act(o, s): return s.actions[0].action\n"},
        source_sha256="",
        sha256="",
    )
    with pytest.raises(ValueError, match="allowlist"):
        contaminated.validate()


@pytest.mark.parametrize("operation", ["replace", "merge", "delete", "simplify"])
def test_change_plan_supports_non_additive_operations(operation):
    protocol = _protocol()
    protocol.ChangePlan((protocol.ChangeOperation(operation, "fake target"),)).validate()


def test_unjustified_complexity_growth_is_rejected():
    protocol = _protocol()
    before = protocol.ComplexityMetrics(2, 10, 0, 2)
    after = protocol.ComplexityMetrics(3, 20, 0, 3)
    plan = protocol.ChangePlan((protocol.ChangeOperation("add", "new rule"),))
    with pytest.raises(ValueError, match="complexity"):
        protocol.validate_complexity_update(before, after, plan)
    justified = replace(plan, complexity_growth_reason="training-only gap", training_evidence=("a" * 64,))
    protocol.validate_complexity_update(before, after, justified)


def test_miracle_experience_skill_versions_are_immutable(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch)
    learning = protocol.preflight_learning(bundle["config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    experience = protocol.materialize_empty_experience(learning)
    stored = store.store_skill(experience, learning)
    before = stored.read_bytes()
    with pytest.raises(FileExistsError):
        store.store_skill(experience, learning)
    assert stored.read_bytes() == before
    assert store.load_skill("experience-v0").sha256 == experience.sha256


def test_validation_evidence_is_excluded_from_prompt_and_experience(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch)
    prepared = protocol.preflight_iteration(bundle["config"])
    payload = json.dumps(prepared.training_prompt_payload, sort_keys=True)
    assert bundle["evidence"].sha256 in payload
    assert all(item.case.role == "train" for item in prepared.training_evidence)
    assert not hasattr(prepared, "validation_score")


def test_validation_role_evidence_is_rejected_even_with_training_like_name(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch, role="validation")
    with pytest.raises(protocol.IterationPreflightError, match="role.*train"):
        protocol.preflight_iteration(bundle["config"])


def test_tampered_strategy_rollback_fails_before_any_write(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch)
    prepared = protocol.preflight_iteration(bundle["config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    path = store.root / "strategies" / "strategy-v0.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    before = sorted(p.relative_to(store.root) for p in store.root.rglob("*"))
    with pytest.raises(ValueError, match="canonical"):
        store.rollback(
            source_version="strategy-v0", target_version="strategy-v0", reason="fake rollback",
            operator="test-operator", context=prepared,
        )
    assert sorted(p.relative_to(store.root) for p in store.root.rglob("*")) == before


def test_valid_rollback_creates_new_record_without_mutating_history(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch)
    prepared = protocol.preflight_iteration(bundle["config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    first_path = store.root / "strategies" / "strategy-v0.json"
    second = _strategy(
        tmp_path,
        version="strategy-v1",
        context=prepared,
        evidence=(bundle["evidence"].sha256,),
    )
    second_path = store.store_strategy(second, prepared)
    before = {path: path.read_bytes() for path in (first_path, second_path)}
    record_path = store.rollback(
        source_version="strategy-v1", target_version="strategy-v0", reason="fake regression",
        operator="test-operator", context=prepared,
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["operation"] == "rollback"
    assert {path: path.read_bytes() for path in (first_path, second_path)} == before


def test_missing_human_replay_skill_is_explicitly_blocked(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _bundle(tmp_path, monkeypatch)
    config = replace(
        bundle["config"],
        match=replace(
            bundle["match_config"],
            human_replay_skill_path=tmp_path / "missing.json",
        ),
    )
    with pytest.raises(protocol.IterationPreflightError, match="human replay Skill"):
        protocol.preflight_iteration(config)


def test_non_miracle_game_is_not_subject_to_miracle_iteration_schema():
    assert _protocol().preflight_for_game("chess", None) is None


def test_decision_unit_and_kl_contract_remain_frozen():
    protocol = _protocol()
    assert protocol.LEGAL_ACTION_UNIT == "one_legal_atomic_judge_command"
    assert protocol.TRAJECTORY_KL_EPSILON == 0.01
    assert protocol.KL_DIRECTION == "new||old"
    assert protocol.KL_ROLLOUT_SOURCE == "new_policy"
    assert protocol.DECISION_CHANGE_RATE == "not_collected"
