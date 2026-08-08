from dataclasses import replace
from pathlib import Path

import pytest

from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals.champion_campaign import (
    CAMPAIGN_ACTION_SPACE_SPEC_ID,
    CAMPAIGN_DECISION_SPACE_SHA256,
    CAMPAIGN_QUALIFICATION_MANIFEST_SHA256,
    CAMPAIGN_REPLAY_SKILL_SHA256,
    CAMPAIGN_RULES_SHA256,
    build_campaign_learning_cases,
    campaign_learning_seeds,
    load_champion_campaign_config,
)


FIXTURES = Path(__file__).parent / "fixtures"


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


def test_campaign_freezes_thirty_two_fresh_learning_acts():
    config = _config()

    assert config.max_acts == 32
    assert config.checkpoint_acts == (1, 2, 4, 8, 16, 32)
    assert config.replicates == 3
    assert campaign_learning_seeds(config, 1) == (306011, 306012, 306013)
    assert campaign_learning_seeds(config, 32) == (306321, 306322, 306323)
    assert len(
        {
            seed
            for act in range(1, 33)
            for seed in campaign_learning_seeds(config, act)
        }
    ) == 96


def test_campaign_learning_cases_are_high_only_and_dual_seat():
    pilot = load_pilot_config(FIXTURES / "pilot-v1.toml")
    cases = build_campaign_learning_cases(
        pilot,
        _config(),
        replicate_id="replicate-2",
        act_index=4,
    )

    assert len(cases) == 6
    assert {case.seed for case in cases} == {306041, 306042, 306043}
    assert {case.first_player for case in cases} == {0, 1}
    assert {case.metadata["tier"] for case in cases} == {"high"}
    assert all("replicate-2-a4" in case.case_id for case in cases)


def test_campaign_rejects_mutated_authority_or_act_identity():
    config = _config()
    with pytest.raises(ValueError, match="outside the frozen range"):
        campaign_learning_seeds(config, 0)
    with pytest.raises(ValueError, match="replicate ID"):
        build_campaign_learning_cases(
            load_pilot_config(FIXTURES / "pilot-v1.toml"),
            config,
            replicate_id="replicate-4",
            act_index=1,
        )

    changed = replace(config, action_space_spec_id="a" * 64)
    assert changed != config
