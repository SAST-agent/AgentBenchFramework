from pathlib import Path

import pytest

from agentbench_frame.generals.assets import (
    AssetValidationError,
    load_pilot_config,
    load_round6_learning_config,
)
from agentbench_frame.generals.evaluator import (
    build_round6_learning_cases,
    build_round6_validation_cases,
)


FIXTURE = Path(__file__).parent / "fixtures" / "pilot-v1.toml"
LEARNING_FIXTURE = (
    Path(__file__).parent / "fixtures" / "v6-strongest-learning-v1.toml"
)


def test_v6_learning_manifest_is_exact_fresh_strongest_matrix():
    pilot = load_pilot_config(FIXTURE)
    config = load_round6_learning_config(LEARNING_FIXTURE, pilot)

    assert config.learning_id == "generals-hl-v6-macro-strongest-v1"
    assert config.opponent_id == "advanced-rank02-robinliu-v18"
    assert config.seeds == (287101, 287202, 287303)
    assert config.seats == (0, 1)


def test_v6_validation_is_high_medium_fresh_and_disjoint():
    pilot = load_pilot_config(FIXTURE)
    config = load_round6_learning_config(LEARNING_FIXTURE, pilot)
    learning = build_round6_learning_cases(pilot, config)
    validation = build_round6_validation_cases(pilot)

    assert len(learning) == 6
    assert len({case.case_id for case in learning}) == 6
    assert {case.opponent for case in learning} == {
        "advanced-rank02-robinliu-v18"
    }
    assert {case.metadata["tier"] for case in learning} == {"high"}
    assert {case.metadata["phase"] for case in learning} == {"learn6"}
    assert {case.first_player for case in learning} == {0, 1}
    assert len(validation) == 12
    assert {case.metadata["tier"] for case in validation} == {"high", "medium"}
    assert {case.metadata["phase"] for case in validation} == {"validate6"}
    assert {case.seed for case in validation} == {288101, 288202, 288303}
    assert {case.seed for case in learning}.isdisjoint(
        case.seed for case in validation
    )


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("287101", "286101"),
        ("seats = [0, 1]", "seats = [1, 0]"),
        (
            "advanced-rank02-robinliu-v18",
            "advanced-rank08-nashjunheng-v20",
        ),
    ],
)
def test_v6_learning_manifest_rejects_non_frozen_matrix(
    tmp_path,
    original,
    replacement,
):
    pilot = load_pilot_config(FIXTURE)
    manifest = tmp_path / "learning.toml"
    manifest.write_text(
        LEARNING_FIXTURE.read_text(encoding="utf-8").replace(
            original,
            replacement,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError):
        load_round6_learning_config(manifest, pilot)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        (
            "seeds = [287101, 287202, 287303]",
            "seeds = [287101, 287202, 287303.9]",
        ),
        ("seats = [0, 1]", "seats = [false, 1]"),
    ],
)
def test_v6_learning_manifest_rejects_coerced_numeric_types(
    tmp_path,
    original,
    replacement,
):
    pilot = load_pilot_config(FIXTURE)
    manifest = tmp_path / "learning.toml"
    manifest.write_text(
        LEARNING_FIXTURE.read_text(encoding="utf-8").replace(
            original,
            replacement,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssetValidationError):
        load_round6_learning_config(manifest, pilot)
