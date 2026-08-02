import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest


def _protocol():
    from agentbench_frame.games.miracle import iteration_protocol

    return iteration_protocol


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _make_control_bundle(
    tmp_path: Path,
    monkeypatch,
    *,
    label="one",
    role="train",
    seeds=(0, 1, 2),
    approve_replay=False,
):
    protocol = _protocol()
    root = tmp_path / label
    root.mkdir()

    bootstrap_path = root / "bootstrap.py"
    bootstrap_path.write_bytes(protocol.default_bootstrap_template().path.read_bytes())
    bootstrap = protocol.BootstrapTemplate(
        protocol.BOOTSTRAP_TEMPLATE_VERSION,
        bootstrap_path,
        protocol.APPROVED_BOOTSTRAP_SHA256,
    )

    champion_asset = root / "champion.py"
    champion_asset.write_text(f"# fake champion {label}\n", encoding="utf-8")
    champion = protocol.HumanChampion.create(
        logical_id=f"champion-{label}",
        version="champion-v1",
        artifact_path=champion_asset.name,
        artifact_sha256=_sha(champion_asset),
        provenance="fake-only lifecycle fixture",
        qualification_status="qualified",
        qualification_evidence=("fake-review",),
        protocol_version=protocol.PROTOCOL_VERSION,
        benchmark_version=protocol.BENCHMARK_VERSION,
        frozen=True,
    )
    champion_path = root / "champion.json"
    champion_path.write_bytes(champion.canonical_bytes())

    human_skill = protocol.HumanReplaySkill.create(
        logical_id=f"human-reader-{label}",
        version="reader-v1",
        content={"instructions": ["read only approved fake train replay"]},
        provenance="fake-only lifecycle fixture",
        created_by="fake-reviewer",
    )
    human_skill_path = root / "human-skill.json"
    human_skill_path.write_bytes(human_skill.canonical_bytes())

    case = protocol.MatchPlanCase(
        case_id=f"case-{label}",
        role=role,
        evaluated_agent_camp=0,
        map_type=1,
        day_time=0,
        repeat=1,
        logic_seed=seeds[0],
        evaluated_agent_seed=seeds[1],
        opponent_seed=seeds[2],
    )
    plan = protocol.MatchPlanManifest.create(
        champion_descriptor_sha256=champion.sha256,
        human_replay_skill_sha256=human_skill.sha256,
        evaluated_policy_version="strategy-v0",
        evaluated_policy_source_sha256=protocol.bootstrap_strategy_source_sha256(
            bootstrap
        ),
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
    plan_path = root / "match-plan.json"
    plan_path.write_bytes(plan.canonical_bytes())

    monkeypatch.setattr(
        protocol, "CURRENT_APPROVED_HUMAN_CHAMPION_SHA256", champion.sha256
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
    match_config = protocol.MatchConfig(
        game="24_miracle",
        bootstrap=bootstrap,
        champion_manifest_path=champion_path,
        human_replay_skill_path=human_skill_path,
        match_plan_path=plan_path,
    )

    replay_path = root / "replay.bin"
    replay_path.write_bytes(f"fake replay {label}".encode())
    evidence = protocol.CapturedReplay.from_plan_case(
        evidence_id=f"evidence-{label}",
        artifact_path=replay_path.name,
        artifact_sha256=_sha(replay_path),
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
    replay_manifest_path = root / "replay-evidence.json"
    replay_manifest_path.write_bytes(replay_manifest.canonical_bytes())
    if approve_replay:
        monkeypatch.setattr(
            protocol,
            "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256",
            frozenset({replay_manifest.sha256}),
        )
    learning_config = protocol.LearningConfig(
        match=match_config,
        replay_evidence_manifest_path=replay_manifest_path,
    )
    return {
        "root": root,
        "bootstrap": bootstrap,
        "bootstrap_path": bootstrap_path,
        "champion": champion,
        "champion_path": champion_path,
        "human_skill": human_skill,
        "human_skill_path": human_skill_path,
        "case": case,
        "plan": plan,
        "plan_path": plan_path,
        "match_config": match_config,
        "replay_path": replay_path,
        "evidence": evidence,
        "replay_manifest": replay_manifest,
        "replay_manifest_path": replay_manifest_path,
        "learning_config": learning_config,
    }


def _candidate_strategy(tmp_path, context, *, version="strategy-v1", parent="strategy-v0", evidence=True):
    protocol = _protocol()
    source = tmp_path / f"source-{version}"
    source.mkdir()
    (source / "main.py").write_text(
        "def choose_action(o, s):\n    return s.actions[0].action\n",
        encoding="utf-8",
    )
    training = (
        (context.training_evidence[0].sha256,) if evidence else ()
    )
    return protocol.StrategyVersion.from_directory(
        source,
        version=version,
        parent_version=parent,
        entrypoint="main.py:choose_action",
        dependencies=(),
        change_plan=protocol.ChangePlan(
            (protocol.ChangeOperation("simplify", "fake fallback"),),
            training_evidence=training,
        ),
        interpretability_category="finite_state_machine",
        probability_query="read-only-complete-action-support",
        control_context=context,
    )


def _experience_v1(context):
    protocol = _protocol()
    evidence = context.training_evidence[0]
    return protocol.MiracleExperienceSkill.create_from_learning(
        context,
        version="experience-v1",
        parent_version="experience-v0",
        frozen_facts=(),
        training_observations=("fake train observation",),
        supported_heuristics=("fake heuristic",),
        failed_heuristics=(),
        unverified_hypotheses=(),
        training_evidence_sha256=(evidence.sha256,),
        strategy_versions=("strategy-v1",),
        case_identities=(evidence.case.case_id,),
        created_by="fake-agent",
        modified_by="fake-agent",
    )


def test_match_preflight_does_not_require_future_replay(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch)
    context = protocol.preflight_match(bundle["match_config"])
    assert context.state is protocol.LifecycleState.MATCH_READY
    calls = []
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    result = protocol.open_match_after_preflight(
        bundle["match_config"],
        store=store,
        runner_factory=lambda current: calls.append(current) or "fake-runner",
    )
    assert result == "fake-runner"
    assert len(calls) == 1


def test_unapproved_replay_cannot_reach_agent_or_store(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch)
    match = protocol.preflight_match(bundle["match_config"])
    captured = protocol.build_replay_evidence_manifest(
        match, (bundle["evidence"],)
    )
    assert captured.state is protocol.LifecycleState.REPLAY_CAPTURED_UNAPPROVED
    effects = []
    with pytest.raises(protocol.IterationPreflightError, match="replay.*approved"):
        protocol.open_learning_after_preflight(
            bundle["learning_config"],
            controller_factory=lambda _: effects.append("controller"),
        )
    assert effects == []
    assert not (tmp_path / "store").exists()


def test_full_fake_lifecycle_reaches_candidate_stored_without_real_runner(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    states = [protocol.LifecycleState.PLANNED]
    match = protocol.preflight_match(bundle["match_config"])
    states.append(match.state)
    draft = protocol.materialize_bootstrap_strategy(match)
    assert draft.source_sha256 == bundle["plan"].evaluated_policy_source_sha256
    runner_calls = []
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"],
        store=store,
        runner_factory=lambda current: runner_calls.append(current) or object(),
    )
    assert len(runner_calls) == 1
    states.append(protocol.LifecycleState.REPLAY_CAPTURED_UNAPPROVED)
    replay = protocol.preflight_replay(bundle["learning_config"])
    states.append(replay.state)
    learning = protocol.preflight_learning(bundle["learning_config"])
    states.append(learning.state)

    v0_path = store.root / "strategies" / "strategy-v0.json"
    baseline = protocol.materialize_empty_experience(learning)
    baseline_path = store.store_skill(baseline, learning)
    learned = _experience_v1(learning)
    learned_path = store.store_skill(learned, learning)
    candidate = _candidate_strategy(tmp_path, learning)
    candidate_path = store.store_strategy(candidate, learning)
    assert all(path.is_file() for path in (v0_path, baseline_path, learned_path, candidate_path))
    assert v0_path.read_bytes() == store.load_strategy("strategy-v0").canonical_bytes()
    assert baseline.training_observations == baseline.supported_heuristics == ()
    assert learned.training_evidence_sha256 == (bundle["evidence"].sha256,)

    stored = protocol.preflight_candidate_stored(
        bundle["learning_config"], store, "strategy-v1", "experience-v1"
    )
    states.append(stored.state)
    assert states == [
        protocol.LifecycleState.PLANNED,
        protocol.LifecycleState.MATCH_READY,
        protocol.LifecycleState.REPLAY_CAPTURED_UNAPPROVED,
        protocol.LifecycleState.REPLAY_APPROVED,
        protocol.LifecycleState.LEARNING_READY,
        protocol.LifecycleState.CANDIDATE_STORED,
    ]


def test_prepared_contexts_cannot_be_constructed_or_forged(tmp_path):
    protocol = _protocol()
    for context_type in (
        protocol.MatchReadyContext,
        protocol.BaselineStoredContext,
        protocol.ReplayApprovedContext,
        protocol.LearningReadyContext,
        protocol.CandidateStoredContext,
        protocol.EvaluationReadyContext,
    ):
        with pytest.raises(TypeError):
            context_type("self-declared-approved")
    forged = object.__new__(protocol.LearningReadyContext)
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    with pytest.raises(protocol.IterationPreflightError, match="validated context"):
        store.store_skill(object(), forged)
    assert not store.root.exists()


def test_forged_context_cannot_start_factory_materialize_store_or_rollback(tmp_path):
    protocol = _protocol()
    forged = object.__new__(protocol.MatchReadyContext)
    effects = []
    with pytest.raises(protocol.IterationPreflightError):
        protocol.open_match_after_preflight(
            forged,
            store=protocol.ImmutableIterationStore(tmp_path / "store"),
            runner_factory=lambda _: effects.append("runner"),
        )
    with pytest.raises(protocol.IterationPreflightError):
        protocol.materialize_bootstrap_strategy(forged)
    assert effects == []
    assert not (tmp_path / "store").exists()


def test_forged_context_cannot_store_valid_objects_or_rollback(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    match = protocol.preflight_match(bundle["match_config"])
    learning = protocol.preflight_learning(bundle["learning_config"])
    draft = protocol.materialize_bootstrap_strategy(match)
    baseline = protocol.materialize_empty_experience(learning)
    forged = object.__new__(protocol.LearningReadyContext)
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    with pytest.raises(protocol.IterationPreflightError, match="validated context"):
        store.store_strategy(draft, forged)
    with pytest.raises(protocol.IterationPreflightError, match="validated context"):
        store.store_skill(baseline, forged)
    with pytest.raises(protocol.IterationPreflightError, match="validated context"):
        store.rollback(
            source_version="strategy-v1",
            target_version="strategy-v0",
            reason="forged",
            operator="fake-reviewer",
            context=forged,
        )
    assert not store.root.exists()


def test_production_empty_approvals_block_all_self_declared_objects(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(protocol, "CURRENT_APPROVED_HUMAN_CHAMPION_SHA256", None)
    monkeypatch.setattr(protocol, "APPROVED_HUMAN_REPLAY_SKILL_SHA256", frozenset())
    monkeypatch.setattr(protocol, "APPROVED_MATCH_PLAN_MANIFEST_SHA256", frozenset())
    monkeypatch.setattr(protocol, "APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256", frozenset())
    effects = []
    with pytest.raises(protocol.IterationPreflightError, match="approved"):
        protocol.open_match_after_preflight(
            bundle["match_config"],
            store=protocol.ImmutableIterationStore(tmp_path / "store"),
            runner_factory=lambda _: effects.append("runner"),
        )
    assert effects == []


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("champion_path", b"{}\n"),
        ("human_skill_path", b"{}\n"),
        ("plan_path", b"{}\n"),
        ("bootstrap_path", b"# changed bootstrap\n"),
        ("replay_manifest_path", b"{}\n"),
        ("replay_path", b"changed replay bytes"),
    ],
)
def test_post_preflight_asset_replacement_is_zero_write(
    tmp_path, monkeypatch, attribute, replacement
):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    match = protocol.preflight_match(bundle["match_config"])
    learning = protocol.preflight_learning(bundle["learning_config"])
    draft = protocol.materialize_bootstrap_strategy(match)
    target = bundle[attribute]
    target.write_bytes(replacement)
    effects = []
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    with pytest.raises(protocol.IterationPreflightError):
        protocol.open_learning_after_preflight(
            bundle["learning_config"],
            controller_factory=lambda _: effects.append("controller"),
        )
    with pytest.raises(protocol.IterationPreflightError):
        store.store_strategy(draft, learning)
    assert effects == []
    assert not store.root.exists()


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("champion_path", b"{}\n"),
        ("human_skill_path", b"{}\n"),
        ("plan_path", b"{}\n"),
        ("bootstrap_path", b"# changed bootstrap\n"),
        ("replay_manifest_path", b"{}\n"),
        ("replay_path", b"changed replay bytes"),
    ],
)
def test_post_preflight_replacement_blocks_skill_store_and_rollback_without_write(
    tmp_path, monkeypatch, attribute, replacement
):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    learning = protocol.preflight_learning(bundle["learning_config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    candidate = _candidate_strategy(tmp_path, learning)
    store.store_strategy(candidate, learning)
    baseline = protocol.materialize_empty_experience(learning)
    store.store_skill(baseline, learning)
    learned = _experience_v1(learning)
    before = {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }

    bundle[attribute].write_bytes(replacement)
    with pytest.raises(protocol.IterationPreflightError):
        store.store_skill(learned, learning)
    with pytest.raises(protocol.IterationPreflightError):
        store.rollback(
            source_version="strategy-v1",
            target_version="strategy-v0",
            reason="asset changed",
            operator="fake-reviewer",
            context=learning,
        )
    after = {
        path.relative_to(store.root): path.read_bytes()
        for path in store.root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_non_v0_strategy_requires_nonempty_validated_train_evidence(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    learning = protocol.preflight_learning(bundle["learning_config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    before = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
    candidate = _candidate_strategy(tmp_path, learning, evidence=False)
    with pytest.raises(ValueError, match="training evidence"):
        store.store_strategy(candidate, learning)
    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before


def test_experience_parent_and_train_evidence_are_enforced_before_write(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    learning = protocol.preflight_learning(bundle["learning_config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    learned = _experience_v1(learning)
    with pytest.raises(ValueError, match="parent"):
        store.store_skill(learned, learning)
    assert not store.root.exists()
    baseline = protocol.materialize_empty_experience(learning)
    store.store_skill(baseline, learning)
    contaminated = replace(
        learned,
        training_evidence_sha256=("f" * 64,),
        sha256="",
    )
    before = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
    with pytest.raises(ValueError, match="train evidence"):
        store.store_skill(contaminated, learning)
    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before


def test_different_control_identity_cannot_form_strategy_parent_chain(tmp_path, monkeypatch):
    protocol = _protocol()
    first = _make_control_bundle(tmp_path, monkeypatch, label="first", approve_replay=True)
    first_learning = protocol.preflight_learning(first["learning_config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        first["match_config"], store=store, runner_factory=lambda current: current
    )
    second = _make_control_bundle(tmp_path, monkeypatch, label="second", approve_replay=True)
    second_learning = protocol.preflight_learning(second["learning_config"])
    candidate = _candidate_strategy(tmp_path, second_learning)
    before = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
    with pytest.raises(ValueError, match="identity"):
        store.store_strategy(candidate, second_learning)
    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before


def test_different_control_identity_cannot_form_experience_parent_chain(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    first = _make_control_bundle(tmp_path, monkeypatch, label="skill-first", approve_replay=True)
    first_learning = protocol.preflight_learning(first["learning_config"])
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    store.store_skill(protocol.materialize_empty_experience(first_learning), first_learning)

    second = _make_control_bundle(tmp_path, monkeypatch, label="skill-second", approve_replay=True)
    second_learning = protocol.preflight_learning(second["learning_config"])
    before = sorted(path.relative_to(store.root) for path in store.root.rglob("*"))
    with pytest.raises(ValueError, match="identity"):
        store.store_skill(_experience_v1(second_learning), second_learning)
    assert sorted(path.relative_to(store.root) for path in store.root.rglob("*")) == before


def test_replay_evidence_requires_exact_planned_case_coverage(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch)
    first = bundle["case"]
    second = replace(first, case_id="case-two", repeat=2)
    plan = protocol.MatchPlanManifest.create(
        champion_descriptor_sha256=bundle["champion"].sha256,
        human_replay_skill_sha256=bundle["human_skill"].sha256,
        evaluated_policy_version="strategy-v0",
        evaluated_policy_source_sha256=protocol.bootstrap_strategy_source_sha256(
            bundle["bootstrap"]
        ),
        cases=(first, second),
        bootstrap_sha256=protocol.APPROVED_BOOTSTRAP_SHA256,
        protocol_version=protocol.PROTOCOL_VERSION,
        benchmark_version=protocol.BENCHMARK_VERSION,
        iteration_protocol_version=protocol.ITERATION_PROTOCOL_VERSION,
        research_manifest_sha256=protocol.research_manifest_sha256(),
        iteration_manifest_sha256=hashlib.sha256(
            protocol.canonical_iteration_protocol_manifest_bytes()
        ).hexdigest(),
    )
    bundle["plan_path"].write_bytes(plan.canonical_bytes())
    monkeypatch.setattr(
        protocol, "APPROVED_MATCH_PLAN_MANIFEST_SHA256", frozenset({plan.sha256})
    )
    match_config = replace(bundle["match_config"], match_plan_path=bundle["plan_path"])
    match = protocol.preflight_match(match_config)
    first_evidence = protocol.CapturedReplay.from_plan_case(
        evidence_id="evidence-first",
        artifact_path=bundle["replay_path"].name,
        artifact_sha256=_sha(bundle["replay_path"]),
        game="24_miracle",
        case=first,
        plan=plan,
    )
    with pytest.raises(protocol.IterationPreflightError, match="every planned case"):
        protocol.build_replay_evidence_manifest(match, (first_evidence,))

    duplicate = replace(first_evidence, evidence_id="evidence-duplicate", sha256="")
    with pytest.raises(protocol.IterationPreflightError, match="duplicate case"):
        protocol.ReplayEvidenceManifest.create(
            match_plan_sha256=plan.sha256,
            evidence=(first_evidence, duplicate),
            protocol_version=protocol.PROTOCOL_VERSION,
            benchmark_version=protocol.BENCHMARK_VERSION,
            iteration_protocol_version=protocol.ITERATION_PROTOCOL_VERSION,
        )

    unknown = replace(second, case_id="case-outside")
    outside = protocol.CapturedReplay.from_plan_case(
        evidence_id="evidence-outside",
        artifact_path=bundle["replay_path"].name,
        artifact_sha256=_sha(bundle["replay_path"]),
        game="24_miracle",
        case=unknown,
        plan=plan,
    )
    with pytest.raises(protocol.IterationPreflightError, match="identity drift"):
        protocol.build_replay_evidence_manifest(match, (first_evidence, outside))


def test_production_apis_expose_no_approval_override_and_control_tables_are_exact():
    protocol = _protocol()
    for name in (
        "preflight_match",
        "preflight_replay",
        "preflight_learning",
        "preflight_iteration",
        "open_match_after_preflight",
        "open_learning_after_preflight",
        "open_iteration_after_preflight",
    ):
        assert not any(
            "approved" in parameter
            for parameter in inspect.signature(getattr(protocol, name)).parameters
        )
    assert protocol.CURRENT_APPROVED_HUMAN_CHAMPION_SHA256 is None
    assert protocol.APPROVED_HUMAN_REPLAY_SKILL_SHA256 == frozenset(
        {
            "cd16e9eec4c9549a8384debad8f5e6ab8bd7865dcdc83c1cf00dfdc061657e37"
        }
    )
    assert protocol.APPROVED_MATCH_PLAN_MANIFEST_SHA256 == frozenset()
    assert protocol.APPROVED_REPLAY_EVIDENCE_MANIFEST_SHA256 == frozenset()


@pytest.mark.parametrize(
    "seed",
    [-1, 2147483648, False, True, 1.0, "1"],
)
def test_match_plan_seed_rejects_noncanonical_or_out_of_range_values(seed):
    protocol = _protocol()
    with pytest.raises(ValueError, match="seed"):
        protocol.MatchPlanCase("case-seed", "train", 0, 0, 0, 1, seed, 0, 0)


@pytest.mark.parametrize("seed", [0, 2147483647])
def test_match_plan_seed_accepts_exact_boundaries(seed):
    protocol = _protocol()
    case = protocol.MatchPlanCase("case-seed", "train", 0, 0, 0, 1, seed, seed, seed)
    case.validate()


def test_non_miracle_isolation_remains_intact():
    protocol = _protocol()
    assert protocol.preflight_for_game("chess", None) is None
