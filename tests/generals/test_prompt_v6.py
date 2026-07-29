from dataclasses import replace

import pytest

from agentbench_frame.generals.prompt import FORBIDDEN_EVALUATION_SEEDS
from agentbench_frame.generals.prompt_v6 import build_round6_prompt
from agentbench_frame.generals.replay import (
    CriticalDecision,
    CriticalLearningEvidence,
)


V6_SEEDS = (287101, 287202, 287303)
HISTORICAL_SEEDS = (
    281101,
    281202,
    281303,
    282101,
    282202,
    282303,
    282404,
    282505,
    282601,
    282702,
    282803,
    282904,
    283005,
    283101,
    283202,
    283303,
    284101,
    284202,
    284303,
    285101,
    285202,
    285303,
    286101,
    286202,
    286303,
)
VALIDATION_SEEDS = (288101, 288202, 288303)


def _evidence(seed: int, seat: int) -> CriticalLearningEvidence:
    return CriticalLearningEvidence(
        replay_id=f"learn6-high-s{seed}-p{seat}",
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        outcome="loss",
        dense={"army_margin": {"terminal": -5.0}},
        total_decision_count=1,
        omitted_decision_count=0,
        decisions=(
            CriticalDecision(
                state_id=f"learn6-high-s{seed}-p{seat}-d1",
                round_number=1,
                seat=seat,
                selection_reasons=("first_decision",),
                action=((8,),),
                decision_class="end_only",
                features={"own_main_army": 1},
            ),
        ),
    )


def _records() -> tuple[CriticalLearningEvidence, ...]:
    return tuple(_evidence(seed, seat) for seed in V6_SEEDS for seat in (0, 1))


def _build(
    records: tuple[CriticalLearningEvidence, ...],
    max_bytes: int = 131_072,
):
    return build_round6_prompt(
        benchmark_id="generals-hl-pilot-v1",
        v5_strategy="exact editable v5 strategy",
        v5_experience="v5 experience",
        rules_text="official rules",
        replay_skill_text="frozen human replay skill",
        replay_skill_sha256="a" * 64,
        evidence=records,
        action_profile={"mean_primitives_per_turn": 1.0},
        max_bytes=max_bytes,
    )


def test_v6_prompt_isolates_exact_high_only_learning_episodes():
    result = _build(_records())

    assert result.feedback_episodes_read == 6
    assert result.feedback_decision_records_read == 6
    assert result.omitted_episode_ids == ()
    assert result.included_episode_ids == tuple(
        f"learn6-high-s{seed}-p{seat}"
        for seed in V6_SEEDS
        for seat in (0, 1)
    )
    assert "bounded macro-action planner" in result.prompt
    assert "You may edit strategy.py, state_view.py" in result.prompt
    assert "formal score" not in result.prompt.lower()
    assert "ACTION-PROFILE DIAGNOSTICS" in result.prompt
    assert "AUTHORITATIVE FROZEN HUMAN REPLAY SKILL" in result.prompt
    assert "sha256=" + "a" * 64 in result.prompt
    assert "frozen human replay skill" in result.prompt


@pytest.mark.parametrize(
    "seed",
    sorted(
        set(FORBIDDEN_EVALUATION_SEEDS)
        | set(HISTORICAL_SEEDS)
        | set(VALIDATION_SEEDS)
    ),
)
def test_v6_prompt_rejects_formal_historical_and_validation_seeds(seed: int):
    records = list(_records())
    records[-1] = _evidence(seed, 1)

    with pytest.raises(ValueError, match="forbidden round-6 evidence seed"):
        _build(tuple(records))


def test_v6_prompt_rejects_non_high_evidence():
    records = list(_records())
    records[-1] = replace(records[-1], opponent_tier="medium")

    with pytest.raises(ValueError, match="high-tier"):
        _build(tuple(records))


def test_v6_prompt_rejects_duplicate_episode_ids():
    records = list(_records())
    records[-1] = replace(records[-1], replay_id=records[0].replay_id)

    with pytest.raises(ValueError, match="duplicate round-6 evidence episode"):
        _build(tuple(records))


def test_v6_prompt_rejects_incomplete_or_unexpected_episode_sets():
    with pytest.raises(ValueError, match="exactly six"):
        _build(_records()[:-1])

    records = list(_records())
    records[-1] = _evidence(299999, 1)
    with pytest.raises(ValueError, match="exactly six"):
        _build(tuple(records))


def test_v6_prompt_rejects_complete_prompt_over_the_byte_cap():
    with pytest.raises(ValueError, match="exceeds max_bytes"):
        _build(_records(), max_bytes=1)
