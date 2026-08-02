import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest


def _protocol():
    from agentbench_frame.games.miracle import iteration_protocol

    return iteration_protocol


def _canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _build_bundle(tmp_path: Path, monkeypatch=None, *, role="train", replay_name="replay.bin"):
    protocol = _protocol()
    bootstrap_path = tmp_path / "bootstrap.py"
    bootstrap_path.write_bytes(protocol.default_bootstrap_template().path.read_bytes())
    bootstrap = protocol.BootstrapTemplate(
        protocol.BOOTSTRAP_TEMPLATE_VERSION,
        bootstrap_path,
        protocol.APPROVED_BOOTSTRAP_SHA256,
    )

    champion_artifact = tmp_path / "champion.py"
    champion_artifact.write_text("# fake human champion asset\n", encoding="utf-8")
    champion = protocol.HumanChampion.create(
        logical_id="human-champion",
        version="champion-v1",
        artifact_path="champion.py",
        artifact_sha256=_sha_bytes(champion_artifact.read_bytes()),
        provenance="fake-only reviewed fixture",
        qualification_status="qualified",
        qualification_evidence=("fake-qualification-record",),
        protocol_version=protocol.PROTOCOL_VERSION,
        benchmark_version=protocol.BENCHMARK_VERSION,
        frozen=True,
    )
    champion_path = tmp_path / "champion.json"
    champion_path.write_bytes(champion.canonical_bytes())

    human_skill = protocol.HumanReplaySkill.create(
        logical_id="human-replay-reader",
        version="human-skill-v1",
        content={"instructions": ["read the fake replay deterministically"]},
        provenance="fake-only human review fixture",
        created_by="fake-human-reviewer",
    )
    human_skill_path = tmp_path / "human-replay-skill.json"
    human_skill_path.write_bytes(human_skill.canonical_bytes())

    case = protocol.MatchPlanCase(
        case_id="training-case-1",
        role=role,
        evaluated_agent_camp=0,
        map_type=1,
        day_time=0,
        repeat=1,
        logic_seed=101,
        evaluated_agent_seed=202,
        opponent_seed=303,
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
        iteration_manifest_sha256=_sha_bytes(
            protocol.canonical_iteration_protocol_manifest_bytes()
        ),
    )
    plan_path = tmp_path / "match-plan.json"
    plan_path.write_bytes(plan.canonical_bytes())

    replay_path = tmp_path / replay_name
    replay_path.write_bytes(b"fake replay bytes")
    evidence = protocol.CapturedReplay.from_plan_case(
        evidence_id="replay-evidence-1",
        artifact_path=replay_name,
        artifact_sha256=_sha_bytes(replay_path.read_bytes()),
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

    if monkeypatch is not None:
        monkeypatch.setattr(
            protocol,
            "CURRENT_APPROVED_HUMAN_CHAMPION_SHA256",
            champion.sha256,
        )
        monkeypatch.setattr(
            protocol,
            "APPROVED_HUMAN_REPLAY_SKILL_SHA256",
            frozenset({human_skill.sha256}),
        )
        monkeypatch.setattr(
            protocol,
            "APPROVED_MATCH_PLAN_MANIFEST_SHA256",
            frozenset({plan.sha256}),
        )
        monkeypatch.setattr(
            protocol,
            "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256",
            frozenset({replay_manifest.sha256}),
        )

    match_config = protocol.MatchConfig(
        game="24_miracle",
        bootstrap=bootstrap,
        champion_manifest_path=champion_path,
        human_replay_skill_path=human_skill_path,
        match_plan_path=plan_path,
    )
    config = protocol.LearningConfig(
        match=match_config,
        replay_evidence_manifest_path=replay_manifest_path,
    )
    return {
        "config": config,
        "match_config": match_config,
        "bootstrap_path": bootstrap_path,
        "champion": champion,
        "champion_path": champion_path,
        "human_skill": human_skill,
        "human_skill_path": human_skill_path,
        "case": case,
        "evidence": evidence,
        "plan": plan,
        "plan_path": plan_path,
        "replay_path": replay_path,
        "replay_manifest": replay_manifest,
        "replay_manifest_path": replay_manifest_path,
    }


def test_production_approval_tables_are_exact_and_self_signed_assets_are_rejected(tmp_path):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path)
    assert protocol.CURRENT_APPROVED_HUMAN_CHAMPION_SHA256 is None
    assert protocol.APPROVED_HUMAN_REPLAY_SKILL_SHA256 == frozenset(
        {
            "cd16e9eec4c9549a8384debad8f5e6ab8bd7865dcdc83c1cf00dfdc061657e37"
        }
    )
    assert protocol.APPROVED_MATCH_PLAN_MANIFEST_SHA256 == frozenset()
    assert protocol.APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256 == frozenset()
    with pytest.raises(protocol.IterationPreflightError, match="approved"):
        protocol.preflight_iteration(bundle["config"])


def test_test_only_monkeypatch_enables_fake_preflight_and_restores_production_tables(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    prepared = protocol.preflight_iteration(bundle["config"])
    assert prepared.champion.sha256 == bundle["champion"].sha256
    assert prepared.human_replay_skill.sha256 == bundle["human_skill"].sha256
    assert [item.sha256 for item in prepared.training_evidence] == [
        bundle["evidence"].sha256
    ]


def test_production_api_exposes_no_approval_override():
    protocol = _protocol()
    for name in (
        "load_human_champion",
        "preflight_match",
        "preflight_replay",
        "preflight_learning",
        "preflight_iteration",
        "preflight_for_game",
        "open_iteration_after_preflight",
    ):
        assert not any(
            "approved" in parameter
            for parameter in inspect.signature(getattr(protocol, name)).parameters
        )


def test_protocol_manifest_identity_excludes_dynamic_approval_contents(monkeypatch):
    protocol = _protocol()
    before = protocol.canonical_iteration_protocol_manifest_bytes()
    fake = "f" * 64
    monkeypatch.setattr(protocol, "CURRENT_APPROVED_HUMAN_CHAMPION_SHA256", fake)
    monkeypatch.setattr(
        protocol, "APPROVED_HUMAN_REPLAY_SKILL_SHA256", frozenset({fake})
    )
    monkeypatch.setattr(
        protocol, "APPROVED_MATCH_PLAN_MANIFEST_SHA256", frozenset({fake})
    )
    monkeypatch.setattr(
        protocol, "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256", frozenset({fake})
    )
    after = protocol.canonical_iteration_protocol_manifest_bytes()
    assert after == before
    assert fake.encode() not in after


def test_self_signed_human_skill_is_rejected_even_when_champion_and_replays_are_approved(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path)
    monkeypatch.setattr(
        protocol,
        "CURRENT_APPROVED_HUMAN_CHAMPION_SHA256",
        bundle["champion"].sha256,
    )
    monkeypatch.setattr(
        protocol,
        "APPROVED_MATCH_PLAN_MANIFEST_SHA256",
        frozenset({bundle["plan"].sha256}),
    )
    monkeypatch.setattr(
        protocol,
        "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256",
        frozenset({bundle["replay_manifest"].sha256}),
    )
    with pytest.raises(protocol.IterationPreflightError, match="human replay.*approved"):
        protocol.preflight_iteration(bundle["config"])


def test_validation_replay_renamed_like_training_is_still_rejected(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _build_bundle(
        tmp_path, monkeypatch, role="validation", replay_name="train-1.bin"
    )
    with pytest.raises(protocol.IterationPreflightError, match="role.*train"):
        protocol.preflight_iteration(bundle["config"])


def test_replay_sha_mismatch_fails_before_any_mutable_factory(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    bundle["replay_path"].write_bytes(b"tampered replay")
    effects = []
    with pytest.raises(protocol.IterationPreflightError, match="replay.*SHA"):
        protocol.open_learning_after_preflight(
            bundle["config"],
            controller_factory=lambda _: effects.append("controller"),
        )
    assert effects == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("role", "validation"), ("case_id", "renamed-case"), ("logic_seed", 999)],
)
def test_replay_role_case_and_seed_mismatch_fail_before_mutable_factories(
    tmp_path, monkeypatch, field, value
):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    document = json.loads(bundle["replay_manifest_path"].read_text(encoding="utf-8"))
    document["evidence"][0]["case"][field] = value
    evidence_unsigned = dict(document["evidence"][0])
    evidence_unsigned.pop("sha256")
    document["evidence"][0]["sha256"] = _sha_bytes(_canonical(evidence_unsigned))
    manifest_unsigned = dict(document)
    manifest_unsigned.pop("sha256")
    document["sha256"] = _sha_bytes(_canonical(manifest_unsigned))
    bundle["replay_manifest_path"].write_bytes(_canonical(document))
    monkeypatch.setattr(
        protocol,
        "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256",
        frozenset({document["sha256"]}),
    )
    effects = []
    with pytest.raises(protocol.IterationPreflightError, match="case/role/seed"):
        protocol.open_learning_after_preflight(
            bundle["config"],
            controller_factory=lambda _: effects.append("controller"),
        )
    assert effects == []


def test_legacy_self_declared_human_authored_boolean_is_not_an_approval(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    legacy = {
        "schema_version": "24-miracle-experience-skill-v1",
        "version": "legacy-skill",
        "human_authored_replay_reader": True,
    }
    bundle["human_skill_path"].write_bytes(_canonical(legacy))
    monkeypatch.setattr(
        protocol,
        "APPROVED_HUMAN_REPLAY_SKILL_SHA256",
        frozenset({_sha_bytes(_canonical(legacy))}),
    )
    with pytest.raises(protocol.IterationPreflightError, match="human replay Skill"):
        protocol.preflight_iteration(bundle["config"])


def test_replay_champion_digest_mismatch_fails_before_mutable_factories(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    document = json.loads(bundle["replay_manifest_path"].read_text(encoding="utf-8"))
    wrong_digest = "f" * 64
    document["evidence"][0]["champion_descriptor_sha256"] = wrong_digest
    evidence_unsigned = dict(document["evidence"][0])
    evidence_unsigned.pop("sha256")
    document["evidence"][0]["sha256"] = _sha_bytes(_canonical(evidence_unsigned))
    manifest_unsigned = dict(document)
    manifest_unsigned.pop("sha256")
    document["sha256"] = _sha_bytes(_canonical(manifest_unsigned))
    bundle["replay_manifest_path"].write_bytes(_canonical(document))
    monkeypatch.setattr(
        protocol,
        "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256",
        frozenset({document["sha256"]}),
    )
    effects = []
    with pytest.raises(protocol.IterationPreflightError, match="control identity"):
        protocol.open_learning_after_preflight(
            bundle["config"],
            controller_factory=lambda _: effects.append("controller"),
        )
    assert effects == []


def test_bootstrap_runtime_self_hash_cannot_approve_changed_template(tmp_path):
    protocol = _protocol()
    changed = tmp_path / "minimal_bootstrap_strategy.py"
    changed.write_text("# changed bootstrap\n", encoding="utf-8")
    self_signed = protocol.BootstrapTemplate(
        version=protocol.BOOTSTRAP_TEMPLATE_VERSION,
        path=changed,
        sha256=_sha_bytes(changed.read_bytes()),
    )
    with pytest.raises(protocol.IterationPreflightError, match="bootstrap.*approved SHA"):
        protocol.validate_bootstrap_template(self_signed)


def _write_strategy_source(tmp_path, name, text):
    root = tmp_path / name
    root.mkdir()
    (root / "main.py").write_text(text, encoding="utf-8")
    return root


def _strategy(protocol, root, *, version, parent, plan, complexity=None, context=None):
    return protocol.StrategyVersion.from_directory(
        root,
        version=version,
        parent_version=parent,
        entrypoint="main.py:choose_action",
        dependencies=(),
        change_plan=plan,
        interpretability_category="finite_state_machine",
        probability_query="read-only-complete-action-support",
        complexity=complexity,
        bootstrap_version=protocol.BOOTSTRAP_TEMPLATE_VERSION,
        bootstrap_sha256=protocol.APPROVED_BOOTSTRAP_SHA256,
        control_context=context,
    )


def test_falsely_reported_zero_complexity_is_rejected(tmp_path):
    protocol = _protocol()
    root = _write_strategy_source(
        tmp_path,
        "candidate",
        "def choose_action(o, s):\n    if o:\n        return s.actions[0].action\n    return s.actions[-1].action\n",
    )
    with pytest.raises(ValueError, match="complexity"):
        _strategy(
            protocol,
            root,
            version="strategy-v1",
            parent="strategy-v0",
            plan=protocol.ChangePlan((protocol.ChangeOperation("add", "branch"),)),
            complexity=protocol.ComplexityMetrics(0, 0, 0, 0),
        )


def test_parent_missing_and_complexity_failure_leave_store_tree_unchanged(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    prepared = protocol.preflight_iteration(bundle["config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    source = _write_strategy_source(
        tmp_path,
        "missing-parent-candidate",
        "def choose_action(o, s):\n    return s.actions[0].action\n",
    )
    candidate = _strategy(
        protocol,
        source,
        version="strategy-v1",
        parent="missing-parent",
        plan=protocol.ChangePlan(
            (protocol.ChangeOperation("replace", "fallback"),),
            training_evidence=(bundle["evidence"].sha256,),
        ),
        context=prepared,
    )
    before = list((tmp_path / "store").rglob("*"))
    with pytest.raises(ValueError, match="parent"):
        store.store_strategy(candidate, prepared)
    assert list((tmp_path / "store").rglob("*")) == before == []


def test_from_scratch_v0_matches_fixed_bootstrap_and_compression_appends(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    prepared = protocol.preflight_iteration(bundle["config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    v0_path = store.root / "strategies" / "strategy-v0.json"
    v0 = store.load_strategy("strategy-v0")
    assert v0.source_files == {
        "main.py": protocol.default_bootstrap_template().path.read_text(encoding="utf-8")
    }
    before = v0_path.read_bytes()

    source = _write_strategy_source(
        tmp_path,
        "simplified",
        "def choose_action(o, s):\n    return s.actions[0].action\n",
    )
    v1 = _strategy(
        protocol,
        source,
        version="strategy-v1",
        parent="strategy-v0",
        plan=protocol.ChangePlan(
            (protocol.ChangeOperation("simplify", "bootstrap fallback"),),
            training_evidence=(bundle["evidence"].sha256,),
        ),
        context=prepared,
    )
    v1_path = store.store_strategy(v1, prepared)
    assert v1_path.is_file()
    assert v0_path.read_bytes() == before


@pytest.mark.parametrize(
    "source",
    [
        "def choose_action(o, s): return __import__('os').system('x')\n",
        "import importlib\ndef choose_action(o, s): return importlib.import_module('os')\n",
        "def choose_action(o, s): return eval('1')\n",
        "def choose_action(o, s): return open('secret').read()\n",
        "def choose_action(o, s): return getattr(__import__('os'), 'system')('x')\n",
    ],
)
def test_dynamic_import_and_indirect_system_or_file_calls_are_rejected(tmp_path, source):
    protocol = _protocol()
    root = _write_strategy_source(tmp_path, "unsafe", source)
    with pytest.raises(ValueError, match="forbidden|dependency|file|dynamic"):
        _strategy(
            protocol,
            root,
            version="strategy-v1",
            parent="strategy-v0",
            plan=protocol.ChangePlan((protocol.ChangeOperation("replace", "unsafe"),)),
        )


@pytest.mark.parametrize("bad_id", ["../escape", "C:/escape", "a/b", "a\\b", ".."])
def test_store_and_load_reject_path_like_logical_ids(tmp_path, bad_id):
    protocol = _protocol()
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    with pytest.raises(ValueError, match="logical ID"):
        store.load_strategy(bad_id)


def test_unrelated_branch_cannot_be_rollback_target(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _build_bundle(tmp_path, monkeypatch)
    prepared = protocol.preflight_iteration(bundle["config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    for version in ("strategy-left", "strategy-right"):
        source = _write_strategy_source(
            tmp_path,
            version,
            "def choose_action(o, s):\n    return s.actions[0].action\n",
        )
        candidate = _strategy(
            protocol,
            source,
            version=version,
            parent="strategy-v0",
            plan=protocol.ChangePlan(
                (protocol.ChangeOperation("simplify", "fallback"),),
                training_evidence=(bundle["evidence"].sha256,),
            ),
            context=prepared,
        )
        store.store_strategy(candidate, prepared)
    before = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
    with pytest.raises(ValueError, match="ancestor"):
        store.rollback(
            source_version="strategy-left",
            target_version="strategy-right",
            reason="invalid cross-branch rollback",
            operator="fake-reviewer",
            context=prepared,
        )
    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before


def test_non_miracle_game_remains_outside_iteration_schema():
    protocol = _protocol()
    assert protocol.preflight_for_game("chess", None) is None
