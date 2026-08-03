import json

import pytest


def _ledger_record(*, verdict, version, delta):
    from agentbench_frame.hl.experience_ledger import (
        ExperienceComparison,
        ExperienceRecord,
    )

    return ExperienceRecord(
        schema_version="1.0",
        iteration_id=f"iter-{version}",
        act_id=f"act-{version}",
        branch_index=int(version),
        parent_version_id="v-parent",
        candidate_version_id=f"v-{version}",
        selected=version == "1",
        activation_condition=f"observable condition {version}",
        mechanism=f"mechanism {version}",
        preservation_contract="otherwise preserve parent",
        activation_status="complete",
        decision_count=100,
        changed_action_count=5,
        verdict=verdict,
        comparisons=(
            ExperienceComparison(
                opponent="rank15" if version != "3" else "rank16",
                candidate_role="P0" if version != "3" else "P1",
                seed=100 + int(version),
                parent_result="loss",
                candidate_result="loss",
                parent_points=0.0,
                candidate_points=0.0,
                parent_candidate_score=0.0,
                parent_opponent_score=100.0,
                candidate_candidate_score=delta,
                candidate_opponent_score=100.0,
                parent_dense_margin=-100.0,
                candidate_dense_margin=-100.0 + delta,
                dense_margin_delta=delta,
            ),
        ),
    )


def test_experience_skill_persists_structured_knowledge_and_history(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager, ExperienceUpdate

    manager = ExperienceManager(tmp_path / "experience", compress_every_acts=2)
    path = manager.update(
        "act-0001",
        ExperienceUpdate(
            stable_knowledge=("Open portal must be considered after its activation round.",),
            failed_hypotheses=("Always chase the nearest bean caused repeated captures.",),
            replay_evidence=("replay-7 level 1 round 61: portal open but action moved away",),
            active_questions=("When is a detour for shield worth the delay?",),
        ),
    )

    text = path.read_text(encoding="utf-8")
    assert "# HL Experience Skill" in text
    assert "## Stable knowledge" in text
    assert "## Failed hypotheses" in text
    assert "replay-7 level 1 round 61" in text
    assert (tmp_path / "experience" / "history" / "act-0001.md").exists()


def test_experience_compression_deduplicates_without_erasing_failed_lessons(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager, ExperienceUpdate

    manager = ExperienceManager(tmp_path / "experience", compress_every_acts=2)
    repeated = "A shield changes the next collision outcome."
    manager.update("act-0001", ExperienceUpdate(stable_knowledge=(repeated,)))
    manager.update(
        "act-0002",
        ExperienceUpdate(
            stable_knowledge=(repeated, repeated),
            failed_hypotheses=("A blind threshold sweep gave no causal insight.",),
        ),
    )

    text = manager.path.read_text(encoding="utf-8")
    assert text.count(repeated) == 1
    assert "blind threshold sweep" in text


def test_experience_rejects_secrets_and_source_code_blobs(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager, ExperienceUpdate

    manager = ExperienceManager(tmp_path / "experience")
    with pytest.raises(ValueError):
        manager.update(
            "act-secret",
            ExperienceUpdate(stable_knowledge=("use sk-super-secret-key",)),
        )
    with pytest.raises(ValueError):
        manager.update(
            "act-code",
            ExperienceUpdate(stable_knowledge=("```python\ndef act(s):\n return 1\n```",)),
        )


def test_experience_update_file_is_strict_and_survives_resume(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager

    root = tmp_path / "experience"
    manager = ExperienceManager(root)
    update_path = tmp_path / "experience_update.json"
    update_path.write_text(
        json.dumps(
            {
                "stable_knowledge": ["A complete replay is required for causal diagnosis."],
                "failed_hypotheses": [],
                "replay_evidence": ["replay-1 level 1 round 3: EATEN_BY_GHOST"],
                "active_questions": [],
            }
        ),
        encoding="utf-8",
    )
    manager.apply_file("act-0001", update_path)

    resumed = ExperienceManager(root)
    text = resumed.path.read_text(encoding="utf-8")
    assert "complete replay" in text
    assert "level 1 round 3" in text

    update_path.write_text('{"unknown":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown experience"):
        resumed.apply_file("act-0002", update_path)


def test_experience_can_rebuild_from_only_validated_updates(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager

    manager = ExperienceManager(tmp_path / "experience")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "stable_knowledge": ["validated lesson"],
                "failed_hypotheses": [],
                "replay_evidence": [],
                "active_questions": [],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "stable_knowledge": ["invalid candidate lesson"],
                "failed_hypotheses": [],
                "replay_evidence": [],
                "active_questions": [],
            }
        ),
        encoding="utf-8",
    )
    manager.apply_file("act-1", first)
    manager.apply_file("act-2", second)

    manager.rebuild((("act-1", first),))

    skill = manager.path.read_text(encoding="utf-8")
    assert "validated lesson" in skill
    assert "invalid candidate lesson" not in skill


def test_skill_projects_verified_good_bad_and_mixed_outcomes(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager

    manager = ExperienceManager(tmp_path / "experience")
    manager.consolidate_cycle(
        "iter-000003",
        records=(
            _ledger_record(verdict="mixed", version="3", delta=-5.0),
            _ledger_record(verdict="verified_bad", version="2", delta=-20.0),
            _ledger_record(verdict="verified_good", version="1", delta=60.0),
        ),
    )

    text = manager.path.read_text(encoding="utf-8")
    assert "## Verified good conditions" in text
    assert "v-1" in text and "dense_margin_delta=+60" in text
    assert "## Verified bad conditions" in text
    assert "v-2" in text and "dense_margin_delta=-20" in text
    assert "## Mixed or scope-sensitive findings" in text
    assert "v-3" in text
    assert "## Current hard-opponent failure profile" in text
    assert "rank15" in text and "rank16" in text
    assert "rank16/P1" in text


def test_skill_projection_is_deterministic_across_record_order(tmp_path):
    from agentbench_frame.hl.experience import ExperienceManager

    records = (
        _ledger_record(verdict="verified_good", version="1", delta=60.0),
        _ledger_record(verdict="verified_bad", version="2", delta=-20.0),
        _ledger_record(verdict="mixed", version="3", delta=-5.0),
    )
    first = ExperienceManager(tmp_path / "first")
    second = ExperienceManager(tmp_path / "second")

    first.consolidate_cycle("iter-000003", records=records)
    second.consolidate_cycle("iter-000003", records=tuple(reversed(records)))

    assert first.path.read_bytes() == second.path.read_bytes()
