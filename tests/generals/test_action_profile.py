import pytest

from agentbench_frame.eval.benchmark import GameResult
from agentbench_frame.generals.action_profile import (
    action_profile_payload,
    summarize_action_profile,
)
from agentbench_frame.generals.evaluator import GeneralsEvaluation
from agentbench_frame.generals.models import MatchResult, TurnRecord


def _state(*, seat=0, destination_player=-1, destination_general=None):
    return {
        "cells": {
            "0,0": {"player": seat, "general_id": 0},
            "0,1": {
                "player": destination_player,
                "general_id": destination_general,
            },
        }
    }


def _turn(player, commands, state=None, state_after=None):
    return TurnRecord(
        step=0,
        round_number=1,
        player=player,
        state_id_before="before",
        state_before=state or _state(seat=player),
        commands=commands,
        state_id_after="after",
        state_after=state_after or {},
    )


def _evaluation(*turns, evaluated_seat=0):
    match = MatchResult(
        case_id="case",
        valid=True,
        winner=evaluated_seat,
        termination_type="normal",
        seed=7,
        evaluated_seat=evaluated_seat,
        turns=tuple(turns),
        elapsed_time_s=0.1,
        engine_hash="hash",
    )
    return GeneralsEvaluation(
        version="v6",
        status="complete",
        score=1.0,
        wins=1,
        losses=0,
        draws=0,
        per_tier={},
        seat_gap=0.0,
        results=(GameResult("case", "win"),),
        matches=(match,),
    )


def test_action_profile_summarizes_evaluated_seat_macro_actions():
    evaluation = _evaluation(
        _turn(0, ((8,),)),
        _turn(0, ((3, 0, 1), (8,))),
        _turn(0, ((5, 1), (8,))),
        _turn(0, ((1, 0, 0, 4, 2), (8,))),
        _turn(
            0,
            ((1, 0, 0, 4, 2), (1, 0, 1, 3, 1), (8,)),
            _state(seat=0, destination_player=1),
        ),
        _turn(1, ((1, 0, 0, 4, 2), (8,))),
    )

    profile = summarize_action_profile(evaluation)

    assert profile.turn_count == 5
    assert profile.primitive_command_count == 5
    assert profile.mean_primitives_per_turn == 1.0
    assert profile.max_primitives_per_turn == 2
    assert profile.multi_command_turn_count == 1
    assert profile.command_counts == {1: 3, 3: 1, 5: 1}
    assert profile.end_only_turn_count == 1
    assert profile.general_upgrade_count == 1
    assert profile.technology_upgrade_count == 1
    assert profile.move_destination_counts["neutral_plain"] == 1
    assert profile.neutral_plain_move_ratio == pytest.approx(1 / 3)


def test_action_profile_classifies_malformed_moves_as_unknown():
    profile = summarize_action_profile(
        _evaluation(_turn(0, ((1, 0, 0, 9), (8,))))
    )

    assert profile.move_destination_counts == {"unknown": 1}
    assert profile.neutral_plain_move_ratio is None


def test_action_profile_excludes_unknown_moves_from_neutral_plain_ratio():
    profile = summarize_action_profile(
        _evaluation(
            _turn(0, ((1, 0, 0, 9), (1, 0, 0, 4, 2), (8,)))
        )
    )

    assert profile.move_destination_counts == {
        "neutral_plain": 1,
        "unknown": 1,
    }
    assert profile.neutral_plain_move_ratio == 1.0


def test_action_profile_classifies_move_from_state_before_not_state_after():
    profile = summarize_action_profile(
        _evaluation(
            _turn(
                0,
                ((1, 0, 0, 4, 2), (8,)),
                _state(seat=0, destination_player=-1),
                _state(seat=0, destination_player=0),
            )
        )
    )

    assert profile.move_destination_counts == {"neutral_plain": 1}


def test_action_profile_has_no_move_ratio_without_classifiable_moves():
    profile = summarize_action_profile(
        _evaluation(_turn(0, ((3, 0, 1), (8,))))
    )

    assert profile.neutral_plain_move_ratio is None


def test_action_profile_payload_orders_diagnostic_maps_stably():
    profile = summarize_action_profile(
        _evaluation(
            _turn(0, ((5, 1), (3, 0, 1), (8,))),
            _turn(0, ((1, 0, 0, 4, 2), (8,))),
            _turn(
                0,
                ((1, 0, 0, 4, 2), (8,)),
                _state(seat=0, destination_player=1),
            ),
            _turn(
                0,
                ((1, 0, 0, 4, 2), (8,)),
                _state(seat=0, destination_player=-1, destination_general=9),
            ),
        )
    )

    payload = action_profile_payload(profile)

    assert list(payload["command_counts"]) == [1, 3, 5]
    assert list(payload["move_destination_counts"]) == [
        "enemy",
        "neutral_general",
        "neutral_plain",
    ]
