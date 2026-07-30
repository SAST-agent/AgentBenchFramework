def test_champion_and_latest_attempt_are_distinct():
    from agentbench_frame.hl.lineage import LineageManager

    lineage = LineageManager(rollback_patience=3, rollback_margin=0.05)
    lineage.record_evaluation("v0", parent_version_id=None, status="complete", score=0.60)
    lineage.record_evaluation("v1", parent_version_id="v0", status="complete", score=0.72)
    lineage.record_evaluation("v2", parent_version_id="v1", status="complete", score=0.65)

    assert lineage.champion_version_id == "v1"
    assert lineage.latest_attempt_version_id == "v2"
    assert lineage.versions["v2"].score == 0.65


def test_sustained_degradation_rolls_next_parent_back_without_deleting_attempts():
    from agentbench_frame.hl.lineage import LineageManager

    lineage = LineageManager(rollback_patience=3, rollback_margin=0.05)
    scores = [("v0", None, 0.70), ("v1", "v0", 0.60), ("v2", "v1", 0.62), ("v3", "v2", 0.64)]
    for version_id, parent, score in scores:
        lineage.record_evaluation(version_id, parent_version_id=parent, status="complete", score=score)

    decision = lineage.select_next_parent()

    assert decision.rollback is True
    assert decision.from_version_id == "v3"
    assert decision.to_version_id == "v0"
    assert decision.reason == "sustained_degradation"
    assert set(lineage.versions) == {"v0", "v1", "v2", "v3"}


def test_incomplete_evaluation_is_missing_not_zero_and_does_not_trigger_rollback():
    from agentbench_frame.hl.lineage import LineageManager

    lineage = LineageManager(rollback_patience=2, rollback_margin=0.05)
    lineage.record_evaluation("v0", parent_version_id=None, status="complete", score=0.80)
    lineage.record_evaluation("v1", parent_version_id="v0", status="complete", score=0.60)
    lineage.record_evaluation("v2", parent_version_id="v1", status="incomplete", score=None)

    decision = lineage.select_next_parent()

    assert decision.rollback is False
    assert decision.to_version_id == "v2"
    assert lineage.versions["v2"].score is None


def test_improvement_resets_degradation_and_promotes_new_champion():
    from agentbench_frame.hl.lineage import LineageManager

    lineage = LineageManager(rollback_patience=2, rollback_margin=0.05)
    lineage.record_evaluation("v0", parent_version_id=None, status="complete", score=0.50)
    lineage.record_evaluation("v1", parent_version_id="v0", status="complete", score=0.40)
    lineage.record_evaluation("v2", parent_version_id="v1", status="complete", score=0.75)

    decision = lineage.select_next_parent()

    assert lineage.champion_version_id == "v2"
    assert decision.rollback is False
    assert decision.to_version_id == "v2"


def test_k_siblings_are_retained_but_only_selected_candidate_advances_lineage():
    from agentbench_frame.hl.lineage import LineageManager

    lineage = LineageManager(rollback_patience=2, rollback_margin=0.05)
    lineage.register_candidate("v1", parent_version_id="v0", status="complete", score=0.40)
    lineage.register_candidate("v2", parent_version_id="v0", status="complete", score=0.70)
    lineage.register_candidate("v3", parent_version_id="v0", status="complete", score=0.50)
    promoted = lineage.select_version("v2")

    assert promoted is True
    assert lineage.lineage_head_version_id == "v2"
    assert lineage.latest_attempt_version_id == "v3"
    assert set(lineage.versions) == {"v1", "v2", "v3"}
    assert lineage.select_next_parent().to_version_id == "v2"
