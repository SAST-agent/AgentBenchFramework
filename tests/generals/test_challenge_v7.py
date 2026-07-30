from pathlib import Path

import pytest

from agentbench_frame.generals.assets import (
    AssetValidationError,
    load_pilot_config,
)


FIXTURES = Path(__file__).parent / "fixtures"
PILOT_PATH = FIXTURES / "pilot-v1.toml"
CHALLENGE_PATH = FIXTURES / "v7-champion-challenge-v1.toml"
ENGINE_SHA256 = (
    "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
)
SKILL_SHA256 = (
    "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
)


def _load(path: Path = CHALLENGE_PATH):
    from agentbench_frame.generals.challenge_v7 import (
        load_round7_challenge_config,
    )

    return load_round7_challenge_config(
        path,
        load_pilot_config(PILOT_PATH),
        engine_hash=ENGINE_SHA256,
        replay_skill_sha256=SKILL_SHA256,
    )


def test_round7_challenge_loads_exact_frozen_contract():
    challenge = _load()

    assert challenge.challenge_id == "generals-hl-v7-champion-v1"
    assert challenge.opponent_id == "advanced-rank02-robinliu-v18"
    assert challenge.learning_seeds == (290101, 290202, 290303)
    assert challenge.validation_seeds == (
        291101,
        291202,
        291303,
        291404,
        291505,
        291606,
    )
    assert challenge.sealed_seeds == (
        292101,
        292202,
        292303,
        292404,
        292505,
        292606,
        292707,
        292808,
        292909,
        292999,
    )
    assert challenge.seats == (0, 1)
    assert challenge.validation_threshold == (7, 3)
    assert challenge.sealed_threshold == (11, 5)
    assert challenge.engine_sha256 == ENGINE_SHA256
    assert challenge.replay_skill_sha256 == SKILL_SHA256


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("generals-hl-v7-champion-v1", "changed"),
        (
            "advanced-rank02-robinliu-v18",
            "advanced-rank08-nashjunheng-v20",
        ),
        ("learning_seeds = [290101, 290202, 290303]", "learning_seeds = [289101, 290202, 290303]"),
        ("validation_seeds = [291101, 291202, 291303, 291404, 291505, 291606]", "validation_seeds = [291101, 291202, 291303, 291404, 291505, 290101]"),
        ("sealed_seeds = [292101, 292202, 292303, 292404, 292505, 292606, 292707, 292808, 292909, 292999]", "sealed_seeds = [292101, 292202, 292303, 292404, 292505, 292606, 292707, 292808, 292909, 291101]"),
        ("seats = [0, 1]", "seats = [1, 0]"),
        ("validation_min_wins = 7", "validation_min_wins = 6"),
        ("validation_min_wins_per_seat = 3", "validation_min_wins_per_seat = 2"),
        ("sealed_min_wins = 11", "sealed_min_wins = 10"),
        ("sealed_min_wins_per_seat = 5", "sealed_min_wins_per_seat = 4"),
        (ENGINE_SHA256, "a" * 64),
        (SKILL_SHA256, "b" * 64),
    ],
)
def test_round7_challenge_rejects_changed_contract(tmp_path, old, new):
    candidate = tmp_path / "challenge.toml"
    candidate.write_text(
        CHALLENGE_PATH.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError):
        _load(candidate)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("290303", "290303.5"),
        ("seats = [0, 1]", "seats = [false, 1]"),
        ("sealed_min_wins = 11", "sealed_min_wins = 11.0"),
    ],
)
def test_round7_challenge_rejects_numeric_coercion(tmp_path, old, new):
    candidate = tmp_path / "challenge.toml"
    candidate.write_text(
        CHALLENGE_PATH.read_text(encoding="utf-8").replace(old, new, 1),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError):
        _load(candidate)


def test_round7_challenge_rejects_observed_runtime_digests():
    from agentbench_frame.generals.challenge_v7 import (
        load_round7_challenge_config,
    )

    pilot = load_pilot_config(PILOT_PATH)
    with pytest.raises(AssetValidationError, match="engine"):
        load_round7_challenge_config(
            CHALLENGE_PATH,
            pilot,
            engine_hash="0" * 64,
            replay_skill_sha256=SKILL_SHA256,
        )
    with pytest.raises(AssetValidationError, match="replay"):
        load_round7_challenge_config(
            CHALLENGE_PATH,
            pilot,
            engine_hash=ENGINE_SHA256,
            replay_skill_sha256="0" * 64,
        )
