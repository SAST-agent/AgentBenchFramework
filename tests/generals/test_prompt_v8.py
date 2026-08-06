from dataclasses import replace
import hashlib

import pytest

from agentbench_frame.generals.replay import CriticalDecision, CriticalLearningEvidence


LEARNING_SEEDS = (300101, 300202, 300303)
SKILL_HASH = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
PARENT_HASH = "c1eb1e4eae5fb1afa393e15370d744743fe0af206cc05a329fa36bd62ca3c4c4"


def _evidence(seed: int, seat: int) -> CriticalLearningEvidence:
    replay_id = f"learn8-high-s{seed}-p{seat}"
    return CriticalLearningEvidence(
        replay_id=replay_id,
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        outcome="loss",
        dense={"army_margin": {"terminal": -100}},
        total_decision_count=10,
        omitted_decision_count=9,
        decisions=(CriticalDecision(
            state_id=f"v7:{replay_id}-d5",
            round_number=5,
            seat=seat,
            selection_reasons=("first_main_danger",),
            action=((1, 1, 1, 1, 2, 8), (8,)),
            decision_class="main_army_move",
            features={"adjacent_enemy_pressure": 5},
        ),),
    )


def _records():
    return tuple(_evidence(seed, seat) for seed in LEARNING_SEEDS for seat in (0, 1))


def _build(records=None, **overrides):
    from agentbench_frame.generals.prompt_v8 import build_round8_prompt

    values = {
        "benchmark_id": "generals-hl-pilot-v1",
        "parent_content_hash": PARENT_HASH,
        "v7_strategy": "deterministic v7 strategy",
        "v7_experience": "v7 learning experience without held-out data",
        "rules_text": "official rules",
        "replay_skill_text": "frozen replay skill",
        "replay_skill_sha256": SKILL_HASH,
        "evidence": _records() if records is None else records,
        "action_profile": {"command_counts": {"1": 20, "3": 2, "5": 1}},
        "max_bytes": 131072,
    }
    values.update(overrides)
    return build_round8_prompt(**values)


def test_round8_prompt_reads_exactly_six_learning_episodes():
    result = _build()

    assert result.feedback_episodes_read == 6
    assert result.included_episode_ids == tuple(
        f"learn8-high-s{seed}-p{seat}" for seed in LEARNING_SEEDS for seat in (0, 1)
    )
    assert result.manifest["learning_seeds"] == list(LEARNING_SEEDS)
    assert result.manifest["provider_act_limit"] == 1
    assert result.manifest["parent_version"] == "v7"
    assert result.manifest["parent_content_hash"] == PARENT_HASH
    assert result.manifest["prompt_sha256"] == hashlib.sha256(result.prompt.encode()).hexdigest()
    assert "exactly one coding-agent act" in result.prompt
    assert "compress" in result.prompt.lower()


@pytest.mark.parametrize("forbidden", [
    "301101", "280101", "controlled_policy_kl", "formal_score",
    "validation", "top_algorithms/", "transition-parity act",
])
def test_round8_prompt_rejects_nonlearning_material(forbidden):
    with pytest.raises(ValueError, match="forbidden"):
        _build(v7_strategy=forbidden)


def test_round8_prompt_rejects_duplicate_incomplete_or_wrong_skill():
    records = list(_records())
    records[-1] = replace(records[-1], replay_id=records[0].replay_id)
    with pytest.raises(ValueError, match="duplicate"):
        _build(tuple(records))
    with pytest.raises(ValueError, match="exactly six"):
        _build(_records()[:-1])
    with pytest.raises(ValueError, match="replay skill"):
        _build(replay_skill_sha256="0" * 64)


def test_round8_prompt_rejects_wrong_episode_pair_or_oversize():
    records = list(_records())
    records[-1] = _evidence(301101, 1)
    with pytest.raises(ValueError, match="forbidden"):
        _build(tuple(records))
    with pytest.raises(ValueError, match="max_bytes"):
        _build(max_bytes=128)
