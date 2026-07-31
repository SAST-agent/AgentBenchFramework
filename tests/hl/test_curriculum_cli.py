def test_resume_rebuilds_latest_gate_for_active_target():
    from agentbench_frame.hl.cli import _curriculum_evaluation_from_events

    events = [
        {
            "event_type": "evaluation_completed",
            "version_id": "v000001",
            "status": "complete",
            "benchmark_score": 1.0,
            "matches": [
                {
                    "status": "complete",
                    "opponent": "rank15",
                    "seed": 101,
                    "result": "win",
                }
            ],
        },
        {
            "event_type": "curriculum_gate_completed",
            "version_id": "v000001",
            "active_target": "rank14",
            "status": "complete",
            "score": 0.0,
            "matches": [
                {
                    "status": "complete",
                    "opponent": "rank14",
                    "seed": 101,
                    "result": "loss",
                }
            ],
        },
    ]

    evaluation = _curriculum_evaluation_from_events(
        events,
        version_id="v000001",
        active_target="rank14",
    )

    assert evaluation is not None
    assert evaluation.status == "complete"
    assert evaluation.score == 0.0
    assert {match["opponent"] for match in evaluation.matches} == {"rank14"}


def test_resume_ignores_gate_from_a_different_target():
    from agentbench_frame.hl.cli import _curriculum_evaluation_from_events

    evaluation = _curriculum_evaluation_from_events(
        [
            {
                "event_type": "evaluation_completed",
                "version_id": "v000001",
                "status": "complete",
                "benchmark_score": 1.0,
                "matches": [
                    {
                        "status": "complete",
                        "opponent": "rank15",
                        "seed": 101,
                        "result": "win",
                    }
                ],
            }
        ],
        version_id="v000001",
        active_target="rank14",
    )

    assert evaluation is None


def test_resume_parent_uses_stage_origin_after_rejection_and_best_after_stagnation():
    from types import SimpleNamespace

    from agentbench_frame.hl.cli import _curriculum_resume_parent

    state = SimpleNamespace(
        stage_origin_version_id="v000000",
        stage_best_version_id="v000002",
    )

    assert (
        _curriculum_resume_parent(
            [
                {
                    "event_type": "curriculum_candidate_rejected",
                    "version_id": "v000003",
                }
            ],
            state=state,
            lineage_head_version_id="v000003",
        )
        == "v000000"
    )
    assert (
        _curriculum_resume_parent(
            [
                {
                    "event_type": "curriculum_stagnated",
                    "version_id": "v000004",
                }
            ],
            state=state,
            lineage_head_version_id="v000004",
        )
        == "v000002"
    )


def test_resume_parent_uses_recorded_best_after_curriculum_resume():
    from types import SimpleNamespace

    from agentbench_frame.hl.cli import _curriculum_resume_parent

    state = SimpleNamespace(
        stage_origin_version_id="v000000",
        stage_best_version_id="v000002",
    )

    assert (
        _curriculum_resume_parent(
            [
                {
                    "event_type": "curriculum_resumed",
                    "version_id": "v000004",
                    "active_target": "rank15",
                    "stage_best_version_id": "v000002",
                    "stage_best_score": 1 / 3,
                }
            ],
            state=state,
            lineage_head_version_id="v000004",
        )
        == "v000002"
    )
