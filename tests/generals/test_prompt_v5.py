from dataclasses import replace

import pytest

from agentbench_frame.generals.measurement import DecisionClassSummary
from agentbench_frame.generals.prompt_v5 import (
    VersionedCriticalEvidence,
    build_round5_prompt,
)
from agentbench_frame.generals.replay import (
    CriticalDecision,
    CriticalLearningEvidence,
)


def _evidence(seed=286101, seat=0):
    return CriticalLearningEvidence(
        replay_id=f"learn5-high-s{seed}-p{seat}",
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        outcome="loss",
        dense={
            "completed_rounds_survived": 10,
            "army_margin": {"terminal": -5.0},
        },
        total_decision_count=2,
        omitted_decision_count=1,
        decisions=(
            CriticalDecision(
                state_id=f"learn5-high-s{seed}-p{seat}-d1",
                round_number=1,
                seat=seat,
                selection_reasons=("first_decision",),
                action=((8,),),
                decision_class="end_only",
                features={
                    "own_main_army": 1,
                    "adjacent_enemy_pressure": 0,
                },
            ),
        ),
    )


def _classes():
    return DecisionClassSummary(
        total=6,
        counts={
            "main_army_move": 2,
            "non_main_army_move": 3,
            "general_upgrade": 0,
            "end_only": 1,
        },
        rates={
            "main_army_move": 2 / 6,
            "non_main_army_move": 3 / 6,
            "general_upgrade": 0.0,
            "end_only": 1 / 6,
        },
    )


def _build(records, max_bytes=131_072):
    return build_round5_prompt(
        benchmark_id="generals-hl-pilot-v1",
        v3_strategy="global stack planner",
        v3_experience="retained v3 experience",
        v4_strategy="reserve planner with a conservative danger branch",
        v4_experience="retained v4 learning experience",
        rules_text="official rules",
        replay_skill_text="# Human replay skill\nInspect reserves.",
        replay_skill_sha256="a" * 64,
        evidence=records,
        decision_classes={"v3": _classes(), "v4": _classes()},
        dense_deltas={"terminal_army_margin": -10.0},
        max_bytes=max_bytes,
    )


def test_v5_prompt_groups_all_twelve_versioned_episodes():
    records = tuple(
        VersionedCriticalEvidence(
            version=version,
            evidence=_evidence(seed, seat),
        )
        for seed in (286101, 286202, 286303)
        for seat in (0, 1)
        for version in ("v3", "v4")
    )

    result = _build(records)

    assert result.feedback_episodes_read == 12
    assert result.feedback_decision_records_read == 12
    assert result.omitted_episode_ids == ()
    assert result.included_episode_ids[0] == (
        "v3:learn5-high-s286101-p0"
    )
    assert result.included_episode_ids[-1] == (
        "v4:learn5-high-s286303-p1"
    )
    assert "CURRENT ROLLBACK V3 STRATEGY" in result.prompt
    assert "FAILED V4 CANDIDATE STRATEGY" in result.prompt
    assert "Human replay skill" in result.prompt
    assert "formal score" not in result.prompt.lower()


@pytest.mark.parametrize(
    "seed",
    [280101, 281101, 284101, 285101],
)
def test_v5_prompt_rejects_formal_or_historical_episode_records(seed):
    record = VersionedCriticalEvidence(
        version="v3",
        evidence=_evidence(seed, 0),
    )

    with pytest.raises(
        ValueError,
        match="forbidden round-5 evidence seed",
    ):
        _build((record,))


def test_v5_prompt_rejects_non_high_or_unknown_version():
    medium = VersionedCriticalEvidence(
        version="v3",
        evidence=replace(_evidence(), opponent_tier="medium"),
    )
    with pytest.raises(ValueError, match="high-tier"):
        _build((medium,))

    unknown = VersionedCriticalEvidence(
        version="v2",
        evidence=_evidence(),
    )
    with pytest.raises(ValueError, match="version"):
        _build((unknown,))


def test_v5_prompt_truncates_only_at_versioned_episode_boundaries():
    records = tuple(
        VersionedCriticalEvidence(
            version=version,
            evidence=_evidence(286101 + index, index % 2),
        )
        for index in range(20)
        for version in ("v3", "v4")
    )

    result = _build(records, max_bytes=6_000)

    assert result.truncated is True
    assert result.included_episode_ids
    assert result.omitted_episode_ids
    assert result.prompt_bytes <= 6_000
    assert all(
        episode_id.startswith(("v3:", "v4:"))
        for episode_id in (
            *result.included_episode_ids,
            *result.omitted_episode_ids,
        )
    )
