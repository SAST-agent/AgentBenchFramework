import pytest
import json


def _frozen_config(*, mode=None, hash_value="hash-a", max_acts=None):
    provider = {"kind": "codex"}
    if mode is not None:
        provider["structured_output_mode"] = mode
    return {
        "schema_version": "1.0",
        "source_config": "/experiment/config.yaml",
        "source_config_sha256": hash_value,
        "run": {
            "game": "30_antwar2",
            "provider": provider,
            "iteration": {"max_acts": max_acts},
        },
        "paths": {"workspace": "/experiment/candidate"},
    }


def test_resume_provider_compatibility_allows_only_explicit_mode_change():
    from agentbench_frame.hl.profile_runner import _resume_compatibility_transition

    transition = _resume_compatibility_transition(
        _frozen_config(mode=None, hash_value="hash-old"),
        _frozen_config(mode="validated_file", hash_value="hash-new"),
        allow_provider_compatibility_change=True,
    )

    assert transition == {
        "field": "run.provider.structured_output_mode",
        "frozen_value": "native_schema",
        "active_value": "validated_file",
        "reason": "provider_native_schema_incompatible",
    }


def test_resume_provider_compatibility_rejects_implicit_or_broader_change():
    from agentbench_frame.hl.profile_runner import _resume_compatibility_transition

    frozen = _frozen_config(mode=None, hash_value="hash-old")
    current = _frozen_config(mode="validated_file", hash_value="hash-new")
    with pytest.raises(ValueError, match="differs from frozen"):
        _resume_compatibility_transition(
            frozen,
            current,
            allow_provider_compatibility_change=False,
        )

    broader = _frozen_config(
        mode="validated_file",
        hash_value="hash-new",
        max_acts=9,
    )
    with pytest.raises(ValueError, match="only permits"):
        _resume_compatibility_transition(
            frozen,
            broader,
            allow_provider_compatibility_change=True,
        )


def test_profile_pending_cycle_recovery_uses_latest_attempt_boundary(tmp_path):
    from agentbench_frame.hl.profile_runner import _pending_cycle_recoveries

    workspace = tmp_path / "candidate"
    workspace.mkdir()
    persisted = tmp_path / "run" / "proposals" / "iter-000001" / "new.json"
    persisted.parent.mkdir(parents=True)
    persisted.write_text(
        json.dumps(
            [
                {
                    "branch_index": index,
                    "diagnosis": f"diagnosis-{index}",
                    "mechanism": (
                        "tower recycling",
                        "opponent spell counter",
                        "proactive attack lane",
                        "resource conversion",
                    )[index],
                    "activation_condition": f"condition-{index}",
                    "preservation_contract": f"preserve-{index}",
                    "expected_change": f"expected-{index}",
                    "falsifier": f"falsifier-{index}",
                    "code_symbols": ["AI.choose_operations", "AI.choose_bundle"],
                }
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )
    historical = [
        {
            "event_type": "proposal_cycle_started",
            "iteration_id": "iter-000001",
            "parent_version_id": "v000000",
        },
        {
            "event_type": "planner_completed",
            "iteration_id": "iter-000001",
            "act_id": "act-old-planner",
            "branch_briefs": str(tmp_path / "old.json"),
        },
        {
            "event_type": "act_completed",
            "iteration_id": "iter-000001",
            "act_id": "act-old-b00",
            "status": "failed",
            "branch_index": 0,
        },
        {
            "event_type": "proposal_cycle_started",
            "iteration_id": "iter-000001",
            "parent_version_id": "v000000",
        },
        {
            "event_type": "planner_completed",
            "iteration_id": "iter-000001",
            "act_id": "act-new-planner",
            "branch_briefs": str(persisted),
        },
    ]

    recoveries = _pending_cycle_recoveries(
        historical,
        provider=object(),
        workspace=workspace,
        version_store=object(),
        evaluations_by_version={},
        expected_candidate_count=4,
        policy_entry_symbol="AI.choose_operations",
    )

    assert recoveries["planner_recovery"].metadata["act_id"] == "act-new-planner"
    assert recoveries["candidate_recoveries"] == {}
    assert recoveries["repair_recoveries"] == {}


def test_historical_evaluation_preserves_and_backfills_failure_classification():
    from agentbench_frame.hl.profile_runner import _historical_evaluations

    evaluations = _historical_evaluations(
        [
            {
                "event_type": "candidate_activation_measured",
                "version_id": "v-old",
                "status": "failed",
                "error": "tower delta was invalid",
            },
            {
                "event_type": "evaluation_completed",
                "version_id": "v-old",
                "status": "failed",
                "benchmark_score": None,
                "matches": [],
            },
            {
                "event_type": "evaluation_completed",
                "version_id": "v-new",
                "status": "failed",
                "benchmark_score": None,
                "error": "candidate_smoke_failed: fixture mismatch",
                "matches": [],
            },
        ]
    )

    assert evaluations["v-old"].error == (
        "activation_probe_failed: tower delta was invalid"
    )
    assert evaluations["v-new"].error == (
        "candidate_smoke_failed: fixture mismatch"
    )


def test_behavior_comparison_payload_serializes_immutable_details():
    from agentbench_frame.hl.game_profile import BehaviorComparison
    from agentbench_frame.hl.profile_runner import _behavior_comparison_payload

    payload = _behavior_comparison_payload(
        BehaviorComparison(
            status="complete",
            decision_count=16,
            changed_action_count=4,
            details={"role_mean_kl": {"P0": 0.25}},
        )
    )

    assert payload == {
        "status": "complete",
        "decision_count": 16,
        "changed_action_count": 4,
        "details": {"role_mean_kl": {"P0": 0.25}},
    }


def test_resume_progress_reconstructs_best_archive_and_stagnation():
    from agentbench_frame.hl.profile_runner import _resume_progress

    events = [
        {
            "event_type": "evaluation_completed",
            "version_id": "v0",
            "status": "complete",
            "benchmark_score": 0.0,
        },
        {
            "event_type": "certification_completed",
            "version_id": "v0",
            "passing_human_opponents": 0,
        },
        {
            "event_type": "evaluation_completed",
            "version_id": "v1",
            "status": "complete",
            "benchmark_score": 0.25,
        },
        {
            "event_type": "proposal_cycle_completed",
            "iteration_id": "iter-000001",
            "selected_version_id": "v1",
        },
        {
            "event_type": "certification_completed",
            "version_id": "v1",
            "passing_human_opponents": 3,
        },
        {
            "event_type": "evaluation_completed",
            "version_id": "v2",
            "status": "complete",
            "benchmark_score": 0.10,
        },
        {
            "event_type": "proposal_cycle_completed",
            "iteration_id": "iter-000002",
            "selected_version_id": "v2",
        },
        {
            "event_type": "evaluation_completed",
            "version_id": "v3",
            "status": "complete",
            "benchmark_score": 0.20,
        },
        {
            "event_type": "proposal_cycle_completed",
            "iteration_id": "iter-000003",
            "selected_version_id": "v3",
        },
    ]

    progress = _resume_progress(events, origin_version_id="v0")

    assert progress.completed_cycles == 3
    assert progress.best_version_id == "v1"
    assert progress.best_learning_score == 0.25
    assert progress.best_passing == 3
    assert progress.stagnation == 2
    assert progress.certified is False


def test_resume_progress_preserves_certified_terminal_state():
    from agentbench_frame.hl.profile_runner import _resume_progress

    progress = _resume_progress(
        [
            {
                "event_type": "evaluation_completed",
                "version_id": "v0",
                "status": "complete",
                "benchmark_score": 1.0,
            },
            {
                "event_type": "run_completed",
                "reason": "human_pool_target_reached",
                "version_id": "v0",
            },
        ],
        origin_version_id="v0",
    )

    assert progress.certified is True
    assert progress.best_version_id == "v0"


def test_reporting_panel_payload_uses_generic_dense_margin():
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.profile_runner import _reporting_panel_payload

    evaluation = CandidateEvaluation(
        status="complete",
        score=0.5,
        error=None,
        matches=(
            {
                "status": "complete",
                "result": "win",
                "dense_margin": 35.0,
            },
            {
                "status": "complete",
                "result": "loss",
                "dense_margin": -25.0,
            },
            {
                "status": "incomplete",
                "result": "loss",
                "dense_margin": -100.0,
            },
        ),
    )

    payload = _reporting_panel_payload(
        evaluation,
        iteration_id="iter-000004",
        proposal_cycle=4,
        version_id="v17",
    )

    assert payload["iteration_id"] == "iter-000004"
    assert payload["proposal_cycle"] == 4
    assert payload["version_id"] == "v17"
    assert payload["score"] == 0.5
    assert payload["mean_score_margin"] == 5.0


def test_reporting_evaluation_prefers_profile_evaluator_subset():
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.profile_runner import _reporting_evaluation

    calls = []

    class Evaluator:
        def evaluate_reporting_panel(self, version, *, seed_count):
            calls.append((version, seed_count))
            return CandidateEvaluation(status="complete", score=0.75, matches=())

        def certify(self, _version):
            raise AssertionError("full certification must not be used for reporting")

    result = _reporting_evaluation(Evaluator(), "v1", seed_count=1)

    assert result.score == 0.75
    assert calls == [("v1", 1)]
