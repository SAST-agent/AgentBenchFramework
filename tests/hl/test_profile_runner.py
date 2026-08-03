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
