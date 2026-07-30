from pathlib import Path

import pytest

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.assets import load_pilot_config
from agentbench_frame.generals import challenge_v7


FIXTURES = Path(__file__).parent / "fixtures"
ENGINE_SHA256 = (
    "4ca898365a6fe9a0cca39c3c8be0e49f449334814b106882c0b78b3aabdb1d97"
)
SKILL_SHA256 = (
    "0e8f01f8825ad3f351a330d324f2a558613bc98a61ca6adc80d69d3c90690b48"
)


@pytest.fixture
def pilot():
    return load_pilot_config(FIXTURES / "pilot-v1.toml")


@pytest.fixture
def challenge(pilot):
    return challenge_v7.load_round7_challenge_config(
        FIXTURES / "v7-champion-challenge-v1.toml",
        pilot,
        engine_hash=ENGINE_SHA256,
        replay_skill_sha256=SKILL_SHA256,
    )


def _results(cases, wins_by_seat, *, invalid_index=None):
    remaining = dict(wins_by_seat)
    results = []
    for index, case in enumerate(cases):
        win = remaining[case.first_player] > 0
        if win:
            remaining[case.first_player] -= 1
        results.append(
            GameResult(
                case_id=case.case_id,
                outcome="win" if win else "loss",
                valid=index != invalid_index,
                error="engine failed" if index == invalid_index else None,
            )
        )
    return results


def test_round7_partitions_have_exact_seed_then_seat_order(pilot, challenge):
    learning = challenge_v7.build_round7_learning_cases(pilot, challenge)
    validation = challenge_v7.build_round7_validation_cases(pilot, challenge)
    sealed = challenge_v7.build_round7_sealed_cases(pilot, challenge)

    assert len(learning) == 6
    assert len(validation) == 12
    assert len(sealed) == 20
    assert [(case.seed, case.first_player) for case in learning] == [
        (seed, seat)
        for seed in (290101, 290202, 290303)
        for seat in (0, 1)
    ]
    assert [(case.seed, case.first_player) for case in validation] == [
        (seed, seat)
        for seed in (291101, 291202, 291303, 291404, 291505, 291606)
        for seat in (0, 1)
    ]
    assert [(case.seed, case.first_player) for case in sealed] == [
        (seed, seat)
        for seed in (
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
        for seat in (0, 1)
    ]
    assert {case.opponent for case in learning + validation + sealed} == {
        "advanced-rank02-robinliu-v18"
    }
    assert {case.metadata["tier"] for case in learning + validation + sealed} == {
        "high"
    }
    assert {case.metadata["phase"] for case in learning} == {"learn7"}
    assert {case.metadata["phase"] for case in validation} == {"validate7"}
    assert {case.metadata["phase"] for case in sealed} == {"sealed7"}


@pytest.mark.parametrize(
    ("wins_by_seat", "minimum_wins", "minimum_per_seat", "passed"),
    [
        ({0: 6, 1: 5}, 11, 5, True),
        ({0: 7, 1: 4}, 11, 5, False),
        ({0: 5, 1: 5}, 11, 5, False),
    ],
)
def test_sealed_gate_requires_overall_and_each_seat(
    pilot,
    challenge,
    wins_by_seat,
    minimum_wins,
    minimum_per_seat,
    passed,
):
    cases = challenge_v7.build_round7_sealed_cases(pilot, challenge)

    gate = challenge_v7.evaluate_champion_gate(
        results=_results(cases, wins_by_seat),
        cases=cases,
        minimum_wins=minimum_wins,
        minimum_wins_per_seat=minimum_per_seat,
    )

    assert gate.status == ("passed" if passed else "failed")
    assert gate.passed is passed
    assert gate.score == sum(wins_by_seat.values()) / 20
    assert gate.wins == sum(wins_by_seat.values())
    assert gate.per_seat_wins == wins_by_seat
    assert gate.expected_games == 20
    assert gate.valid_games == 20


@pytest.mark.parametrize(
    ("wins_by_seat", "passed"),
    [
        ({0: 4, 1: 3}, True),
        ({0: 5, 1: 2}, False),
        ({0: 3, 1: 3}, False),
    ],
)
def test_validation_gate_uses_frozen_boundary(
    pilot,
    challenge,
    wins_by_seat,
    passed,
):
    cases = challenge_v7.build_round7_validation_cases(pilot, challenge)

    gate = challenge_v7.evaluate_champion_gate(
        results=_results(cases, wins_by_seat),
        cases=cases,
        minimum_wins=7,
        minimum_wins_per_seat=3,
    )

    assert gate.passed is passed
    assert gate.status == ("passed" if passed else "failed")


def test_gate_preserves_missingness_for_incomplete_suite(pilot, challenge):
    cases = challenge_v7.build_round7_sealed_cases(pilot, challenge)
    results = _results(cases, {0: 6, 1: 5})[:-1]

    gate = challenge_v7.evaluate_champion_gate(
        results=results,
        cases=cases,
        minimum_wins=11,
        minimum_wins_per_seat=5,
    )

    assert gate.status == "incomplete"
    assert gate.passed is False
    assert gate.score is None
    assert gate.valid_games == 19
    assert gate.reasons == ("missing_expected_case",)


def test_gate_preserves_missingness_for_invalid_suite(pilot, challenge):
    cases = challenge_v7.build_round7_sealed_cases(pilot, challenge)

    gate = challenge_v7.evaluate_champion_gate(
        results=_results(cases, {0: 6, 1: 5}, invalid_index=3),
        cases=cases,
        minimum_wins=11,
        minimum_wins_per_seat=5,
    )

    assert gate.status == "invalid"
    assert gate.passed is False
    assert gate.score is None
    assert gate.valid_games == 19
    assert gate.reasons == ("invalid_result",)


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected"])
def test_gate_rejects_non_exact_result_identity(pilot, challenge, mutation):
    cases = challenge_v7.build_round7_sealed_cases(pilot, challenge)
    results = _results(cases, {0: 6, 1: 5})
    if mutation == "duplicate":
        results[-1] = results[0]
    else:
        results[-1] = GameResult("foreign-case", "win")

    gate = challenge_v7.evaluate_champion_gate(
        results=results,
        cases=cases,
        minimum_wins=11,
        minimum_wins_per_seat=5,
    )

    assert gate.status == "invalid"
    assert gate.passed is False
    assert gate.score is None
    assert gate.reasons == (
        "duplicate_result" if mutation == "duplicate" else "unexpected_case",
    )
