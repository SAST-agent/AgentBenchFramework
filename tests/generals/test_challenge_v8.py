from pathlib import Path

import pytest

from agentbench_frame.generals.assets import AssetValidationError, load_pilot_config


FIXTURES = Path(__file__).parent / "fixtures"
CHALLENGE = FIXTURES / "v8-clean-room-challenge-v1.toml"
PILOT = load_pilot_config(FIXTURES / "pilot-v1.toml")
ENGINE_HASH = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
SKILL_HASH = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"


def _load(path: Path = CHALLENGE):
    from agentbench_frame.generals.challenge_v8 import load_round8_challenge_config

    return load_round8_challenge_config(
        path,
        PILOT,
        engine_hash=ENGINE_HASH,
        replay_skill_sha256=SKILL_HASH,
    )


def test_round8_contract_loads_exact_clean_room_partitions():
    config = _load()

    assert config.challenge_id == "generals-hl-v8-clean-room-v1"
    assert config.opponent_id == "advanced-rank02-robinliu-v18"
    assert config.learning_seeds == (300101, 300202, 300303)
    assert config.validation_seeds == (
        301101, 301202, 301303, 301404, 301505, 301606,
    )
    assert config.seats == (0, 1)
    assert config.validation_threshold == (2, 1)
    assert config.formal_success_threshold == (2, 12, 1)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("generals-hl-v8-clean-room-v1", "changed"),
        ("300101", "290101"),
        ("301606", "300101"),
        ("seats = [0, 1]", "seats = [1, 0]"),
        ("formal_high_min_wins = 2", "formal_high_min_wins = 1"),
        (ENGINE_HASH, "0" * 64),
        (SKILL_HASH, "1" * 64),
    ],
)
def test_round8_contract_rejects_changed_values(tmp_path, old, new):
    candidate = tmp_path / "challenge.toml"
    candidate.write_text(
        CHALLENGE.read_text(encoding="utf-8").replace(old, new, 1),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError):
        _load(candidate)


def test_round8_cases_are_seed_major_and_dual_seat():
    from agentbench_frame.generals.challenge_v8 import (
        build_round8_learning_cases,
        build_round8_validation_cases,
    )

    config = _load()
    learning = build_round8_learning_cases(PILOT, config)
    validation = build_round8_validation_cases(PILOT, config)

    assert len(learning) == 6
    assert len(validation) == 12
    assert [(case.seed, case.first_player) for case in learning] == [
        (seed, seat) for seed in config.learning_seeds for seat in (0, 1)
    ]
    assert all(case.opponent == config.opponent_id for case in (*learning, *validation))
