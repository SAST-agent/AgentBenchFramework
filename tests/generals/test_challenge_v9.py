from pathlib import Path

import pytest

from agentbench_frame.generals.assets import AssetValidationError, load_pilot_config


FIXTURES = Path(__file__).parent / "fixtures"
CHALLENGE = FIXTURES / "v9-scientific-attribution-v1.toml"
EXPANDED = FIXTURES / "policy-kl-expanded-v1.toml"
PILOT = load_pilot_config(FIXTURES / "pilot-v1.toml")
ENGINE_HASH = "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
SKILL_HASH = "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"


def _load(path: Path = CHALLENGE):
    from agentbench_frame.generals.challenge_v9 import (
        load_round9_challenge_config,
    )

    return load_round9_challenge_config(
        path,
        PILOT,
        engine_hash=ENGINE_HASH,
        replay_skill_sha256=SKILL_HASH,
    )


def test_round9_contract_loads_exact_attribution_and_validation_partitions():
    config = _load()

    assert config.challenge_id == "generals-hl-v9-scientific-attribution-v1"
    assert config.attribution_seeds == (
        302101, 302202, 302303, 302404, 302505, 302606,
    )
    assert config.validation_seeds == (
        303101, 303202, 303303, 303404, 303505, 303606,
    )
    assert config.seats == (0, 1)
    assert config.diagnostic_max_states == 48
    assert config.validation_threshold == (2, 1)
    assert config.formal_success_threshold == (13, 2, 1)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("generals-hl-v9-scientific-attribution-v1", "changed"),
        ("302101", "301101"),
        ("303606", "302101"),
        ("seats = [0, 1]", "seats = [1, 0]"),
        ("diagnostic_max_states = 48", "diagnostic_max_states = true"),
        ("formal_total_min_wins = 13", "formal_total_min_wins = 12"),
        (ENGINE_HASH, "0" * 64),
        (SKILL_HASH, "1" * 64),
    ],
)
def test_round9_contract_rejects_changed_values(tmp_path, old, new):
    candidate = tmp_path / "challenge.toml"
    candidate.write_text(
        CHALLENGE.read_text(encoding="utf-8").replace(old, new, 1),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError):
        _load(candidate)


def test_round9_cases_are_seed_major_dual_seat_and_pair_stable():
    from agentbench_frame.generals.challenge_v9 import (
        build_round9_attribution_cases,
        build_round9_validation_cases,
    )

    config = _load()
    attribution = build_round9_attribution_cases(PILOT, config)
    validation = build_round9_validation_cases(PILOT, config)

    assert len(attribution) == len(validation) == 12
    assert [(case.seed, case.first_player) for case in attribution] == [
        (seed, seat) for seed in config.attribution_seeds for seat in (0, 1)
    ]
    assert [case.metadata["pair_id"] for case in attribution] == [
        f"s{seed}-p{seat}"
        for seed in config.attribution_seeds
        for seat in (0, 1)
    ]
    assert all(case.opponent == config.opponent_id for case in (*attribution, *validation))


def test_expanded_kl_loader_preserves_exact_recipe_order():
    from agentbench_frame.generals.assets import load_expanded_kl_config

    config = load_expanded_kl_config(EXPANDED)

    assert config.measurement_id == "generals-policy-kl-expanded-v1"
    assert config.source_run_id == "20260806_1129_6085e1a6"
    assert config.epsilons == ("0.001", "0.01", "0.05", "0.1")
    assert config.primary_epsilon == "0.01"
    assert len(config.intervention_states) == 12
    assert config.intervention_states[0].state_key == "contact-p0-a"
    assert config.intervention_states[-1].state_key == "consolidation-p1-b"


def test_expanded_kl_loader_rejects_bool_actor(tmp_path):
    from agentbench_frame.generals.assets import load_expanded_kl_config

    candidate = tmp_path / "expanded.toml"
    candidate.write_text(
        EXPANDED.read_text(encoding="utf-8").replace("actor = 0", "actor = true", 1),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError, match="actor"):
        load_expanded_kl_config(candidate)
