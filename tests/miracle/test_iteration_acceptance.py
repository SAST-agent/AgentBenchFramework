import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from test_iteration_lifecycle import _candidate_strategy, _make_control_bundle


def _protocol():
    from agentbench_frame.games.miracle import iteration_protocol

    return iteration_protocol


def _experience_v1(context, *, strategy_version="strategy-v1"):
    protocol = _protocol()
    evidence = context.training_evidence[0]
    return protocol.MiracleExperienceSkill.create_from_learning(
        context,
        version="experience-v1",
        parent_version="experience-v0",
        frozen_facts=(),
        training_observations=("fake approved train observation",),
        supported_heuristics=("fake heuristic",),
        failed_heuristics=(),
        unverified_hypotheses=(),
        training_evidence_sha256=(evidence.sha256,),
        strategy_versions=(strategy_version,),
        case_identities=(evidence.case.case_id,),
        created_by="fake-agent",
        modified_by="fake-agent",
    )


def _store_candidate(tmp_path, monkeypatch, *, audit=None):
    protocol = _protocol()
    bundle = _make_control_bundle(
        tmp_path, monkeypatch, label="acceptance", approve_replay=True
    )

    class AuditStore(protocol.ImmutableIterationStore):
        def store_baseline(self, strategy, context):
            if audit is not None:
                audit.append("store-v0")
            return super().store_baseline(strategy, context)

        def load_strategy(self, version):
            result = super().load_strategy(version)
            if audit is not None and version == "strategy-v0":
                audit.append("verify-v0")
            return result

    store = AuditStore(tmp_path / "store")
    runner_calls = []

    def fake_runner(current):
        if audit is not None:
            audit.append("runner")
        runner_calls.append(current)
        return current

    baseline_context = protocol.open_match_after_preflight(
        bundle["match_config"],
        store=store,
        runner_factory=fake_runner,
    )
    learning = protocol.preflight_learning(bundle["learning_config"])
    store.store_skill(protocol.materialize_empty_experience(learning), learning)
    candidate = _candidate_strategy(tmp_path, learning)
    store.store_strategy(candidate, learning)
    store.store_skill(_experience_v1(learning), learning)
    candidate_context = protocol.preflight_candidate_stored(
        bundle["learning_config"],
        store,
        "strategy-v1",
        "experience-v1",
    )
    return bundle, store, baseline_context, learning, candidate_context, runner_calls


def _kl_result(protocol, *, episode=1, complete=True):
    local = 0.02 if complete else None
    return {
        "episode": episode,
        "version_before": "strategy-v0",
        "version_after": "strategy-v1",
        "epsilon": protocol.TRAJECTORY_KL_EPSILON,
        "status": "complete" if complete else "incomplete",
        "measurement_status": "complete" if complete else "incomplete",
        "direction": protocol.KL_DIRECTION,
        "log_base": "e",
        "rollout_source": protocol.KL_ROLLOUT_SOURCE,
        "estimand": "epsilon_regularized_local_kl_sum_under_new_policy_occupancy",
        "decision_steps": 1,
        "trace": [local],
        "trajectory_kl_episode": local,
        "mean_local_policy_kl": local,
        "errors": [] if complete else ["fake missing old distribution"],
        "metadata": {"fake_only": True},
        "decisions": [
            {
                "decision_step": 1,
                "context_ref": "fake-context",
                "action_schema_version": "24-miracle-action-support-v1",
                "support_id": "fake-support",
                "legal_action_ids": ["endround"],
                "selected_action_id": "endround",
                "new_distribution": {"endround": 1.0},
                "old_distribution": {"endround": 1.0} if complete else None,
                "new_probabilities": [1.0],
                "old_probabilities": [1.0] if complete else None,
                "local_policy_kl": local,
                "errors": [] if complete else ["old distribution missing"],
            }
        ],
    }


def _evaluation_bundle(tmp_path, monkeypatch, *, candidate_score=0.75, kl_complete=True):
    protocol = _protocol()
    bundle, store, baseline, learning, candidate, runner_calls = _store_candidate(
        tmp_path, monkeypatch
    )
    case = protocol.EvaluationPlanCase(
        "eval-case-1", "evaluation", 0, 1, 0, 1, 101, 202, 303
    )
    plan = protocol.CandidateEvaluationPlan.create_from_candidate(
        candidate,
        purpose="candidate-versus-frozen-champion",
        cases=(case,),
    )
    plan_path = bundle["root"] / "candidate-evaluation-plan.json"
    plan_path.write_bytes(plan.canonical_bytes())
    monkeypatch.setattr(
        protocol,
        "APPROVED_CANDIDATE_EVALUATION_PLAN_SHA256",
        frozenset({plan.sha256}),
    )
    config = protocol.EvaluationConfig(
        learning_config=bundle["learning_config"],
        store_root=store.root,
        candidate_strategy_version="strategy-v1",
        experience_skill_version="experience-v1",
        evaluation_plan_path=plan_path,
    )
    evaluation = protocol.preflight_evaluation(config)
    old_replay = bundle["root"] / "eval-old.replay"
    new_replay = bundle["root"] / "eval-new.replay"
    old_replay.write_bytes(b"fake baseline evaluation replay")
    new_replay.write_bytes(b"fake candidate evaluation replay")
    evidence = protocol.EvaluationCaseEvidence.from_plan_case(
        evidence_id="eval-evidence-1",
        case=case,
        plan=plan,
        baseline_artifact_path=old_replay.name,
        baseline_artifact_sha256=hashlib.sha256(old_replay.read_bytes()).hexdigest(),
        candidate_artifact_path=new_replay.name,
        candidate_artifact_sha256=hashlib.sha256(new_replay.read_bytes()).hexdigest(),
        baseline_outcome="loss",
        candidate_outcome="win" if candidate_score > 0.5 else "loss",
        baseline_score=0.5,
        candidate_score=candidate_score,
        terminal_status="complete",
        trajectory_kl=_kl_result(protocol, complete=kl_complete),
        information_gain=0.1,
        failure_reason=None if kl_complete else "KL incomplete",
    )
    evidence_manifest = protocol.EvaluationEvidenceManifest.create(
        evaluation_plan_sha256=plan.sha256,
        evidence=(evidence,),
        protocol_version=protocol.PROTOCOL_VERSION,
        benchmark_version=protocol.BENCHMARK_VERSION,
        iteration_protocol_version=protocol.ITERATION_PROTOCOL_VERSION,
    )
    evidence_path = bundle["root"] / "evaluation-evidence.json"
    evidence_path.write_bytes(evidence_manifest.canonical_bytes())
    monkeypatch.setattr(
        protocol,
        "APPROVED_EVALUATION_EVIDENCE_MANIFEST_SHA256",
        frozenset({evidence_manifest.sha256}),
    )
    acceptance = protocol.IterationAcceptanceManifest.create_from_evidence(
        evaluation,
        evidence_manifest,
        blockers=(),
        fake_only=True,
    )
    acceptance_path = bundle["root"] / "iteration-acceptance.json"
    acceptance_path.write_bytes(acceptance.canonical_bytes())
    monkeypatch.setattr(
        protocol,
        "APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256",
        frozenset({acceptance.sha256}),
    )
    acceptance_config = protocol.AcceptanceConfig(
        evaluation=config,
        evaluation_evidence_path=evidence_path,
        acceptance_manifest_path=acceptance_path,
    )
    return locals()


def test_baseline_is_stored_verified_then_runner_receives_frozen_identity(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    audit = []
    bundle, store, baseline, _, _, calls = _store_candidate(
        tmp_path, monkeypatch, audit=audit
    )
    assert audit[:3] == ["store-v0", "verify-v0", "runner"]
    assert len(calls) == 1
    assert baseline.version == "strategy-v0"
    assert baseline.state is protocol.LifecycleState.BASELINE_STORED
    assert baseline.source_sha256 == bundle["plan"].evaluated_policy_source_sha256
    assert baseline.store_root == store.root
    assert baseline.cases == bundle["plan"].cases
    assert store.load_strategy("strategy-v0").sha256 == baseline.strategy_sha256


def test_changed_existing_baseline_blocks_runner(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    path = store.root / "strategies" / "strategy-v0.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{}\n")
    effects = []
    with pytest.raises((ValueError, protocol.IterationPreflightError)):
        protocol.open_match_after_preflight(
            bundle["match_config"],
            store=store,
            runner_factory=lambda _: effects.append("runner"),
        )
    assert effects == []
    assert path.read_bytes() == b"{}\n"


def test_candidate_ready_requires_non_v0_skill_bound_to_candidate(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    learning = protocol.preflight_learning(bundle["learning_config"])
    store.store_skill(protocol.materialize_empty_experience(learning), learning)
    store.store_strategy(_candidate_strategy(tmp_path, learning), learning)
    with pytest.raises(protocol.IterationPreflightError, match="Skill"):
        protocol.preflight_candidate_stored(
            bundle["learning_config"], store, "strategy-v1", "experience-v1"
        )
    with pytest.raises(protocol.IterationPreflightError, match="non-v0"):
        protocol.preflight_candidate_stored(
            bundle["learning_config"], store, "strategy-v1", "experience-v0"
        )


def test_candidate_ready_rejects_skill_that_references_old_strategy(tmp_path, monkeypatch):
    protocol = _protocol()
    bundle = _make_control_bundle(tmp_path, monkeypatch, approve_replay=True)
    store = protocol.ImmutableIterationStore(tmp_path / "store")
    protocol.open_match_after_preflight(
        bundle["match_config"], store=store, runner_factory=lambda current: current
    )
    learning = protocol.preflight_learning(bundle["learning_config"])
    store.store_skill(protocol.materialize_empty_experience(learning), learning)
    store.store_strategy(_candidate_strategy(tmp_path, learning), learning)
    store.store_skill(_experience_v1(learning, strategy_version="strategy-v0"), learning)
    with pytest.raises(protocol.IterationPreflightError, match="candidate strategy"):
        protocol.preflight_candidate_stored(
            bundle["learning_config"], store, "strategy-v1", "experience-v1"
        )


def test_unapproved_evaluation_plan_cannot_open_factory(tmp_path, monkeypatch):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        protocol, "APPROVED_CANDIDATE_EVALUATION_PLAN_SHA256", frozenset()
    )
    calls = []
    with pytest.raises(protocol.IterationPreflightError, match="evaluation plan.*approved"):
        protocol.open_evaluation_after_preflight(
            data["config"], runner_factory=lambda _: calls.append("evaluation")
        )
    assert calls == []


def test_evaluation_plan_identity_or_seed_drift_blocks_factory(tmp_path, monkeypatch):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    document = json.loads(data["plan_path"].read_text(encoding="utf-8"))
    document["cases"][0]["logic_seed"] += 1
    unsigned = dict(document)
    unsigned.pop("sha256")
    document["sha256"] = hashlib.sha256(
        protocol._canonical_bytes(unsigned)
    ).hexdigest()
    data["plan_path"].write_bytes(protocol._canonical_bytes(document))
    calls = []
    with pytest.raises(protocol.IterationPreflightError):
        protocol.open_evaluation_after_preflight(
            data["config"], runner_factory=lambda _: calls.append("evaluation")
        )
    assert calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "non-empty|every planned case"),
        ("duplicate", "duplicate case"),
        ("unknown", "case/seed identity drift"),
        ("seed-drift", "case/seed identity drift"),
    ],
)
def test_evaluation_evidence_requires_exact_planned_case_and_seed_identity(
    tmp_path, monkeypatch, mutation, message
):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    original = data["evidence"]
    if mutation == "missing":
        changed = ()
    elif mutation == "duplicate":
        changed = (
            original,
            replace(original, evidence_id="eval-evidence-duplicate", sha256=""),
        )
    else:
        changed_case = replace(
            original.case,
            case_id="eval-case-outside" if mutation == "unknown" else original.case.case_id,
            logic_seed=original.case.logic_seed
            if mutation == "unknown"
            else original.case.logic_seed + 1,
        )
        changed = (replace(original, case=changed_case, sha256=""),)
    forged = replace(data["evidence_manifest"], evidence=changed, sha256="")
    with pytest.raises(protocol.IterationPreflightError, match=message):
        forged.validate(data["plan"])


def test_unapproved_evaluation_evidence_cannot_enter_terminal_derivation(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        protocol, "APPROVED_EVALUATION_EVIDENCE_MANIFEST_SHA256", frozenset()
    )
    with pytest.raises(protocol.IterationPreflightError, match="evidence.*approved"):
        protocol.derive_iteration_outcome(data["acceptance_config"])


def test_terminal_status_cannot_be_self_declared_complete(tmp_path, monkeypatch):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch, kl_complete=False)
    document = json.loads(data["acceptance_path"].read_text(encoding="utf-8"))
    document["evaluation_status"] = "complete"
    document["blockers"] = []
    unsigned = dict(document)
    unsigned.pop("sha256")
    document["sha256"] = hashlib.sha256(
        protocol._canonical_bytes(unsigned)
    ).hexdigest()
    data["acceptance_path"].write_bytes(protocol._canonical_bytes(document))
    monkeypatch.setattr(
        protocol,
        "APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256",
        frozenset({document["sha256"]}),
    )
    with pytest.raises(
        protocol.IterationPreflightError, match="first-hand evidence"
    ):
        protocol.derive_iteration_outcome(data["acceptance_config"])


@pytest.mark.parametrize(
    "asset",
    [
        "champion_path",
        "human_skill_path",
        "bootstrap_path",
        "plan_path",
        "replay_manifest_path",
        "replay_path",
        "candidate_strategy",
        "candidate_skill",
        "evaluation_plan",
    ],
)
def test_post_preflight_control_or_candidate_change_blocks_evaluation_factory(
    tmp_path, monkeypatch, asset
):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    paths = {
        "candidate_strategy": data["store"].root
        / "strategies"
        / "strategy-v1.json",
        "candidate_skill": data["store"].root / "skills" / "experience-v1.json",
        "evaluation_plan": data["plan_path"],
    }
    target = paths.get(asset, data["bundle"].get(asset))
    target.write_bytes(b"{}\n")
    calls = []
    with pytest.raises((ValueError, protocol.IterationPreflightError)):
        protocol.open_evaluation_after_preflight(
            data["config"], runner_factory=lambda _: calls.append("evaluation")
        )
    assert calls == []


@pytest.mark.parametrize(
    "asset",
    [
        "champion_path",
        "human_skill_path",
        "bootstrap_path",
        "plan_path",
        "replay_manifest_path",
        "replay_path",
        "candidate_strategy",
        "candidate_skill",
        "evaluation_plan",
        "baseline_evaluation_artifact",
        "candidate_evaluation_artifact",
    ],
)
def test_post_preflight_change_blocks_terminal_derivation(
    tmp_path, monkeypatch, asset
):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    paths = {
        "candidate_strategy": data["store"].root
        / "strategies"
        / "strategy-v1.json",
        "candidate_skill": data["store"].root / "skills" / "experience-v1.json",
        "evaluation_plan": data["plan_path"],
        "baseline_evaluation_artifact": data["old_replay"],
        "candidate_evaluation_artifact": data["new_replay"],
    }
    target = paths.get(asset, data["bundle"].get(asset))
    target.write_bytes(b"tampered after preflight\n")
    with pytest.raises((ValueError, protocol.IterationPreflightError)):
        protocol.derive_iteration_outcome(data["acceptance_config"])


def test_evaluation_evidence_never_enters_learning_payload_or_experience(
    tmp_path, monkeypatch
):
    data = _evaluation_bundle(tmp_path, monkeypatch)
    serialized = json.dumps(data["learning"].training_prompt_payload, sort_keys=True)
    assert data["evidence_manifest"].sha256 not in serialized
    assert data["evidence"].sha256 not in data["store"].load_skill(
        "experience-v1"
    ).training_evidence_sha256


def test_score_improvement_with_incomplete_kl_derives_incomplete(tmp_path, monkeypatch):
    protocol = _protocol()
    data = _evaluation_bundle(
        tmp_path, monkeypatch, candidate_score=0.75, kl_complete=False
    )
    outcome = protocol.derive_iteration_outcome(data["acceptance_config"])
    assert outcome.state is protocol.LifecycleState.INCOMPLETE


def test_complete_kl_without_score_improvement_derives_incomplete(tmp_path, monkeypatch):
    protocol = _protocol()
    data = _evaluation_bundle(
        tmp_path, monkeypatch, candidate_score=0.5, kl_complete=True
    )
    outcome = protocol.derive_iteration_outcome(data["acceptance_config"])
    assert outcome.state is protocol.LifecycleState.INCOMPLETE


def test_fake_complete_evidence_derives_fake_only_completed(tmp_path, monkeypatch):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    outcome = protocol.derive_iteration_outcome(data["acceptance_config"])
    assert outcome.state is protocol.LifecycleState.COMPLETED
    assert outcome.fake_only is True
    summary = data["acceptance"].to_results_summary()
    assert summary["evaluation_status"] == "complete"
    assert summary["raw_score"] == 0.5
    assert summary["evo_score"] == 0.75
    assert summary["gain"] == 0.25
    assert summary["research_manifest_sha256"] == protocol.research_manifest_sha256()
    assert summary["iteration_acceptance_sha256"] == data["acceptance"].sha256
    assert summary["iteration_protocol_version"] == protocol.ITERATION_PROTOCOL_VERSION
    assert summary["baseline_policy_version"] == "strategy-v0"
    assert summary["candidate_policy_version"] == "strategy-v1"
    assert summary["training_match_plan_sha256"] == data["plan"].training_match_plan_sha256
    assert summary["training_replay_evidence_sha256"] == data["plan"].training_replay_evidence_sha256
    assert summary["trajectory_kl_evidence_sha256"] == list(
        data["acceptance"].trajectory_kl_evidence_sha256
    )


def test_no_boolean_finalizer_and_production_terminal_approvals_are_empty():
    protocol = _protocol()
    assert not hasattr(protocol, "finalize_iteration")
    assert protocol.APPROVED_CANDIDATE_EVALUATION_PLAN_SHA256 == frozenset()
    assert protocol.APPROVED_EVALUATION_EVIDENCE_MANIFEST_SHA256 == frozenset()
    assert protocol.APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256 == frozenset()
    assert "completed" not in inspect.signature(
        protocol.derive_iteration_outcome
    ).parameters


def test_test_only_terminal_approval_monkeypatch_restores_empty_production_tables(
    monkeypatch,
):
    protocol = _protocol()
    tables = (
        "APPROVED_CANDIDATE_EVALUATION_PLAN_SHA256",
        "APPROVED_EVALUATION_EVIDENCE_MANIFEST_SHA256",
        "APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256",
    )
    assert all(getattr(protocol, name) == frozenset() for name in tables)
    with monkeypatch.context() as local:
        for index, name in enumerate(tables):
            local.setattr(protocol, name, frozenset({f"{index + 1:064x}"}))
        assert all(getattr(protocol, name) for name in tables)
    assert all(getattr(protocol, name) == frozenset() for name in tables)


def test_production_empty_acceptance_approval_makes_completed_unreachable(
    tmp_path, monkeypatch
):
    protocol = _protocol()
    data = _evaluation_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        protocol, "APPROVED_ITERATION_ACCEPTANCE_MANIFEST_SHA256", frozenset()
    )
    with pytest.raises(protocol.IterationPreflightError, match="acceptance.*approved"):
        protocol.derive_iteration_outcome(data["acceptance_config"])


def test_non_miracle_isolation_remains_intact():
    assert _protocol().preflight_for_game("chess", None) is None
