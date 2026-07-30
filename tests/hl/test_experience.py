import json

import pytest


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

