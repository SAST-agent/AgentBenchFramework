import pytest


def _certification_matches(
    *,
    failed: tuple[int, ...],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rank in range(1, 17):
        for seed in (201, 202, 203):
            records.append(
                {
                    "status": "complete",
                    "opponent": f"rank{rank:02d}",
                    "opponent_rank": rank,
                    "seed": seed,
                    "result": "loss" if rank in failed else "win",
                }
            )
    return records


def _summary(*, failed: tuple[int, ...]):
    from agentbench_frame.hl.curriculum import summarize_certification

    return summarize_certification(
        _certification_matches(failed=failed),
        required_win_rate=0.5,
        expected_opponents=16,
    )


def _manager():
    from agentbench_frame.hl.curriculum import CurriculumManager

    return CurriculumManager.start(
        version_id="v000000",
        summary=_summary(failed=(14, 15)),
        required_human_opponents=16,
        stagnation_patience=4,
    )


def test_weakest_failed_selects_largest_rank_from_complete_certification():
    from agentbench_frame.hl.curriculum import select_weakest_failed

    summary = _summary(failed=(14, 15))

    assert summary.passing_opponents == 14
    assert summary.failed_opponents == ("rank14", "rank15")
    assert select_weakest_failed(summary) == "rank15"


def test_certification_summary_rejects_incomplete_or_missing_opponents():
    from agentbench_frame.hl.curriculum import summarize_certification

    incomplete = _certification_matches(failed=(14, 15))
    incomplete[0] = {**incomplete[0], "status": "incomplete"}
    with pytest.raises(ValueError, match="complete"):
        summarize_certification(
            incomplete,
            required_win_rate=0.5,
            expected_opponents=16,
        )

    missing = _certification_matches(failed=(14, 15))
    missing = [
        match for match in missing if match["opponent"] != "rank16"
    ]
    with pytest.raises(ValueError, match="16 opponents"):
        summarize_certification(
            missing,
            required_win_rate=0.5,
            expected_opponents=16,
        )


def test_rank15_promotion_selects_rank14_and_resets_stage_gate():
    manager = _manager()
    manager.begin_stage_gate(version_id="v000000", score=0.0)

    decision = manager.observe_certification(
        version_id="v000003",
        summary=_summary(failed=(14,)),
    )

    assert decision.kind == "promote"
    assert decision.next_target == "rank14"
    assert decision.parent_version_id == "v000003"
    assert manager.state.active_target == "rank14"
    assert manager.state.stage_origin_version_id == "v000003"
    assert manager.state.stage_best_score is None
    assert "rank15" in manager.state.locked_opponents


def test_target_win_is_rejected_when_a_locked_opponent_regresses():
    manager = _manager()
    manager.begin_stage_gate(version_id="v000000", score=0.0)

    decision = manager.observe_certification(
        version_id="v000004",
        summary=_summary(failed=(3, 14)),
    )

    assert decision.kind == "reject"
    assert decision.parent_version_id == "v000000"
    assert decision.lost_locked_opponents == ("rank03",)
    assert manager.state.active_target == "rank15"
    assert manager.state.stage_origin_version_id == "v000000"


def test_four_non_improving_gates_pause_at_stage_best():
    manager = _manager()
    manager.begin_stage_gate(version_id="v000000", score=0.0)

    decisions = [
        manager.observe_gate(version_id=f"v{index:06d}", score=0.0)
        for index in range(1, 5)
    ]

    assert [decision.kind for decision in decisions] == [
        "continue",
        "continue",
        "continue",
        "stagnated",
    ]
    assert decisions[-1].parent_version_id == "v000000"
    assert manager.state.stagnation_count == 4
    assert manager.state.stage_best_version_id == "v000000"


def test_strict_gate_improvement_resets_stagnation_and_stage_best():
    manager = _manager()
    manager.begin_stage_gate(version_id="v000000", score=0.0)
    manager.observe_gate(version_id="v000001", score=0.0)

    decision = manager.observe_gate(version_id="v000002", score=1 / 3)

    assert decision.kind == "improved"
    assert decision.parent_version_id == "v000002"
    assert manager.state.stagnation_count == 0
    assert manager.state.stage_best_version_id == "v000002"
    assert manager.state.stage_best_score == pytest.approx(1 / 3)


def test_all_sixteen_passed_completes_curriculum():
    manager = _manager()
    manager.begin_stage_gate(version_id="v000000", score=0.0)

    decision = manager.observe_certification(
        version_id="v000005",
        summary=_summary(failed=()),
    )

    assert decision.kind == "complete"
    assert decision.next_target is None
    assert manager.state.completed is True
    assert manager.state.active_target is None
    assert len(manager.state.locked_opponents) == 16


def test_curriculum_state_rebuilds_from_lifecycle_events(tmp_path):
    from agentbench_frame.hl.curriculum import CurriculumManager
    from agentbench_frame.hl.events import HLEventWriter, read_events

    path = tmp_path / "events.jsonl"
    writer = HLEventWriter(path, run_id="run-curriculum")
    locked = tuple(
        f"rank{rank:02d}"
        for rank in range(1, 17)
        if rank not in {14, 15}
    )
    writer.write(
        "curriculum_started",
        version_id="v000000",
        active_target="rank15",
        active_target_rank=15,
        locked_opponents=list(locked),
        required_human_opponents=16,
        stage_origin_version_id="v000000",
    )
    writer.write(
        "curriculum_gate_completed",
        version_id="v000000",
        active_target="rank15",
        status="complete",
        score=0.0,
        matches=[],
        baseline=True,
        improved=True,
        stagnation_count=0,
        stage_best_version_id="v000000",
        stage_best_score=0.0,
    )
    for index in range(1, 5):
        writer.write(
            "curriculum_gate_completed",
            version_id=f"v{index:06d}",
            active_target="rank15",
            status="complete",
            score=0.0,
            matches=[],
            baseline=False,
            improved=False,
            stagnation_count=index,
            stage_best_version_id="v000000",
            stage_best_score=0.0,
        )
    writer.write(
        "curriculum_stagnated",
        version_id="v000004",
        active_target="rank15",
        stage_best_version_id="v000000",
        stage_best_score=0.0,
        stagnation_count=4,
    )

    rebuilt = CurriculumManager.from_events(
        read_events(path),
        required_human_opponents=16,
        stagnation_patience=4,
    )

    assert rebuilt.state.active_target == "rank15"
    assert rebuilt.state.locked_opponents == locked
    assert rebuilt.state.stage_origin_version_id == "v000000"
    assert rebuilt.state.stage_best_version_id == "v000000"
    assert rebuilt.state.stage_best_score == 0.0
    assert rebuilt.state.stagnation_count == 4
    assert rebuilt.state.completed is False
