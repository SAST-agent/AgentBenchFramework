from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.champion_campaign import (
    CAMPAIGN_ACTION_SPACE_SPEC_ID,
    CAMPAIGN_DECISION_SPACE_SHA256,
    CAMPAIGN_QUALIFICATION_MANIFEST_SHA256,
    CAMPAIGN_REPLAY_SKILL_SHA256,
    CAMPAIGN_RULES_SHA256,
    campaign_learning_seeds,
    load_champion_campaign_config,
)
from agentbench_frame.generals.prompt_campaign import (
    build_champion_campaign_prompt,
)
from agentbench_frame.generals.replay import (
    CriticalDecision,
    CriticalLearningEvidence,
)


FIXTURES = Path(__file__).parent / "fixtures"
ASSETS = (
    Path(__file__).resolve().parents[5]
    / "AgentBench/.worktrees/generals-assets"
    / "backend_sources/corpus/28_generals"
)


def _config():
    return load_champion_campaign_config(
        FIXTURES / "champion-campaign-v1.toml",
        load_pilot_config(FIXTURES / "pilot-v1.toml"),
        engine_sha256=(
            "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
        ),
        action_space_spec_id=CAMPAIGN_ACTION_SPACE_SPEC_ID,
        decision_space_sha256=CAMPAIGN_DECISION_SPACE_SHA256,
        rules_sha256=CAMPAIGN_RULES_SHA256,
        replay_skill_sha256=CAMPAIGN_REPLAY_SKILL_SHA256,
        qualification_manifest_sha256=(
            CAMPAIGN_QUALIFICATION_MANIFEST_SHA256
        ),
    )


def _authorities():
    paths = {
        "rules_text": ASSETS / "benchmark/rules.md",
        "decision_space_text": ASSETS / "benchmark/decision-space-v1.md",
        "replay_skill_text": ASSETS / "skills/replay-analysis-v2/SKILL.md",
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("production Generals authorities are unavailable")
    return {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
    }


def _evidence(seed: int, seat: int, act: int = 3):
    replay_id = f"campaign-replicate-1-a{act}-high-s{seed}-p{seat}"
    return CriticalLearningEvidence(
        replay_id=replay_id,
        seed=seed,
        evaluated_seat=seat,
        opponent_tier="high",
        termination_type="normal",
        outcome="loss",
        dense={"army_margin": {"terminal": -100.0}},
        total_decision_count=1,
        omitted_decision_count=0,
        decisions=(
            CriticalDecision(
                state_id=f"policy:{replay_id}-d1",
                round_number=1,
                seat=seat,
                selection_reasons=("first_decision",),
                action=((8,),),
                decision_class="end_only",
                features={"coin": 0},
            ),
        ),
    )


def _records(act: int = 3):
    return tuple(
        _evidence(seed, seat, act)
        for seed in campaign_learning_seeds(_config(), act)
        for seat in (0, 1)
    )


def _build(records=None, act=3, **overrides):
    args = {
        "config": _config(),
        "benchmark_id": "generals-hl-pilot-v1",
        "replicate_id": "replicate-1",
        "act_index": act,
        "current_policy_version": f"campaign-r1-a{act - 1}",
        "current_policy_hash": "b" * 64,
        **_authorities(),
        "evidence": _records(act) if records is None else records,
        "action_profile": {
            "command_counts": {str(index): 0 for index in range(1, 8)}
        },
    }
    args.update(overrides)
    return build_champion_campaign_prompt(**args)


def test_campaign_prompt_is_exact_deterministic_and_full_action():
    first = _build()
    second = _build()

    assert first == second
    assert first.feedback_episodes_read == 6
    assert first.omitted_episode_ids == ()
    assert first.truncated is False
    assert first.manifest["prompt_sha256"] == hashlib.sha256(
        first.prompt.encode("utf-8")
    ).hexdigest()
    for opcode in range(1, 8):
        assert f"{opcode}. `[" in first.prompt
    for field in (
        "skills_cd",
        "rest_move",
        "rest_move_step",
        "super_weapon_unlocked",
        "super_weapon_cd",
        "next general ID",
    ):
        assert field in first.prompt


def test_campaign_prompt_rejects_missing_wrong_or_duplicate_evidence():
    with pytest.raises(ValueError, match="exactly the six"):
        _build(_records()[:-1])

    wrong = list(_records())
    wrong[-1] = replace(wrong[-1], replay_id="wrong")
    with pytest.raises(ValueError, match="canonical identity"):
        _build(tuple(wrong))

    duplicate = list(_records())
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate"):
        _build(tuple(duplicate))


def test_campaign_prompt_rejects_qualification_material_everywhere():
    leaked = list(_records())
    leaked[-1] = replace(
        leaked[-1],
        dense={"note": "qualification receipt says score 0.75"},
    )
    with pytest.raises(ValueError, match="qualification artifact"):
        _build(tuple(leaked))

    leaked[-1] = _evidence(304101, 1)
    with pytest.raises(ValueError, match="qualification seed"):
        _build(tuple(leaked))

    with pytest.raises(ValueError, match="qualification artifact"):
        _build(action_profile={"leaderboard_qualification_score": 0.75})


def test_campaign_prompt_rejects_opponent_source_and_mutated_authority():
    with pytest.raises(ValueError, match="opponent source"):
        _build(action_profile={"path": "top_algorithms/private.cpp"})

    authorities = _authorities()
    with pytest.raises(ValueError, match="rules digest"):
        _build(rules_text=authorities["rules_text"] + "\nmutated")


def test_campaign_prompt_rejects_oversized_context_without_truncation():
    config = replace(_config(), prompt_max_bytes=100)
    with pytest.raises(ValueError, match="prompt_max_bytes"):
        _build(config=config)
