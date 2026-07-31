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


def test_replay_summary_is_generated_once_and_reused(tmp_path):
    from agentbench_frame.hl.cli import _ensure_replay_summary

    replay = tmp_path / "replay.jsonl"
    replay.write_text("{}\n", encoding="utf-8")
    script = tmp_path / "summarize.py"
    script.write_text(
        "import pathlib,sys\n"
        "print('# summary for ' + pathlib.Path(sys.argv[1]).name)\n",
        encoding="utf-8",
    )

    summary = _ensure_replay_summary(
        replay=replay,
        summarizer=script,
    )
    script.write_text("raise RuntimeError('must not rerun')\n", encoding="utf-8")
    reused = _ensure_replay_summary(
        replay=replay,
        summarizer=script,
    )

    assert summary == replay.with_name("summary.md")
    assert reused == summary
    assert summary.read_text(encoding="utf-8") == "# summary for replay.jsonl\n"


def test_measurement_failure_is_recorded_and_returns_false(tmp_path):
    from types import SimpleNamespace

    from agentbench_frame.hl.cli import _measure_candidate
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import HLEventWriter, read_events

    class FailingMeasurementRunner:
        def measure(self, **kwargs):
            raise TypeError("candidate policy failed on reference state 17")

    writer = HLEventWriter(tmp_path / "events.jsonl", run_id="run-test")
    completed = CandidateEvaluation(status="complete", score=0.0, matches=())

    measured = _measure_candidate(
        measurement_runner=FailingMeasurementRunner(),
        version_store=object(),
        writer=writer,
        candidate_version=SimpleNamespace(version_id="v000001"),
        parent_version=SimpleNamespace(version_id="v000000"),
        candidate_evaluation=completed,
        parent_evaluation=completed,
    )

    assert measured is False
    event = read_events(tmp_path / "events.jsonl")[0]
    assert event["event_type"] == "measurement_failed"
    assert event["version_id"] == "v000001"
    assert event["parent_version_id"] == "v000000"
    assert event["error_type"] == "TypeError"
    assert "reference state 17" in event["error_message"]


def test_measurement_failure_summary_retains_root_cause_at_end(tmp_path):
    from types import SimpleNamespace

    from agentbench_frame.hl.cli import _measure_candidate
    from agentbench_frame.hl.evaluator import CandidateEvaluation
    from agentbench_frame.hl.events import HLEventWriter, read_events

    class NoisyFailingRunner:
        def measure(self, **kwargs):
            raise RuntimeError(
                ("compatibility warning " * 100)
                + "TypeError: cannot compare int and tuple"
            )

    completed = CandidateEvaluation(status="complete", score=0.0, matches=())
    writer = HLEventWriter(tmp_path / "events.jsonl", run_id="run-test")

    _measure_candidate(
        measurement_runner=NoisyFailingRunner(),
        version_store=object(),
        writer=writer,
        candidate_version=SimpleNamespace(version_id="v000001"),
        parent_version=SimpleNamespace(version_id="v000000"),
        candidate_evaluation=completed,
        parent_evaluation=completed,
    )

    message = read_events(tmp_path / "events.jsonl")[0]["error_message"]
    assert len(message) <= 800
    assert message.startswith("compatibility warning")
    assert message.endswith("TypeError: cannot compare int and tuple")


def test_resume_detects_selected_candidate_with_unfinished_measurement():
    from agentbench_frame.hl.cli import _pending_measurement_candidate

    events = [
        {
            "event_type": "version_created",
            "version_id": "v000001",
            "parent_version_id": "v000000",
            "evaluation_status": "complete",
        },
        {
            "event_type": "candidate_selected",
            "iteration_id": "iter-000001",
            "version_id": "v000001",
        },
    ]

    assert _pending_measurement_candidate(events) == ("v000001", "v000000")
    assert (
        _pending_measurement_candidate(
            events
            + [
                {
                    "event_type": "measurement_failed",
                    "version_id": "v000001",
                    "parent_version_id": "v000000",
                }
            ]
        )
        is None
    )


def test_experience_rebuild_uses_only_candidates_with_complete_measurement(
    tmp_path,
):
    from agentbench_frame.hl.cli import _validated_experience_updates

    pending = tmp_path / "experience" / "pending"
    pending.mkdir(parents=True)
    valid = pending / "act-valid.json"
    invalid = pending / "act-invalid.json"
    valid.write_text("{}\n", encoding="utf-8")
    invalid.write_text("{}\n", encoding="utf-8")
    events = [
        {
            "event_type": "experience_updated",
            "act_id": "act-valid",
            "version_id": "v1",
        },
        {
            "event_type": "policy_kl_measured",
            "version_id": "v1",
        },
        {
            "event_type": "occupancy_measured",
            "version_id": "v1",
        },
        {
            "event_type": "experience_updated",
            "act_id": "act-invalid",
            "version_id": "v2",
        },
        {
            "event_type": "measurement_failed",
            "version_id": "v2",
        },
    ]

    assert _validated_experience_updates(events, tmp_path) == (
        ("act-valid", valid),
    )
