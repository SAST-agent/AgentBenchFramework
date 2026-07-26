import json

import pytest

from agentbench_frame.generals.dense import (
    build_dense_trace,
    dense_sample,
    persist_dense_diagnostics,
    summarize_dense_trace,
)
from agentbench_frame.generals.models import MatchResult, TurnRecord


def state(round_number=1):
    return {
        "round": round_number,
        "cells": {
            "0,0": {"type": 0, "player": 0, "army": 5, "general_id": 0},
            "0,1": {"type": 0, "player": 0, "army": 4, "general_id": None},
            "1,0": {"type": 2, "player": -1, "army": 0, "general_id": None},
            "1,1": {"type": 0, "player": 1, "army": 4, "general_id": 1},
        },
        "generals": {
            "0": {
                "id": 0,
                "type": "main",
                "player": 0,
                "position": [0, 0],
            },
            "1": {
                "id": 1,
                "type": "main",
                "player": 1,
                "position": [1, 1],
            },
        },
        "coins": [30, 10],
    }


def pressure_state(mountain=True):
    obstacle_type = 2 if mountain else 0
    return {
        "round": 1,
        "cells": {
            "0,0": {"type": 0, "player": 0, "army": 3, "general_id": 0},
            "0,1": {
                "type": obstacle_type,
                "player": -1,
                "army": 0,
                "general_id": None,
            },
            "0,2": {"type": 0, "player": 1, "army": 4, "general_id": 1},
            "1,0": {"type": 0, "player": 0, "army": 6, "general_id": None},
            "1,1": {"type": 0, "player": -1, "army": 0, "general_id": None},
            "1,2": {"type": 0, "player": 1, "army": 2, "general_id": None},
        },
        "generals": {
            "0": {
                "id": 0,
                "type": "main",
                "player": 0,
                "position": [0, 0],
            },
            "1": {
                "id": 1,
                "type": "main",
                "player": 1,
                "position": [0, 2],
            },
        },
        "coins": [0, 0],
    }


def ratio_state(round_number, target_cells):
    cells = {}
    generals = {}
    for column in range(4):
        player = 0 if column < target_cells else 1
        general_id = None
        if column == 0:
            general_id = 0
            generals["0"] = {
                "id": 0,
                "type": "main",
                "player": 0,
                "position": [0, 0],
            }
        if column == target_cells:
            general_id = 1
            generals["1"] = {
                "id": 1,
                "type": "main",
                "player": 1,
                "position": [0, column],
            }
        cells[f"0,{column}"] = {
            "type": 0,
            "player": player,
            "army": 1,
            "general_id": general_id,
        }
    return {
        "round": round_number,
        "cells": cells,
        "generals": generals,
        "coins": [round_number, 4 - round_number],
    }


def match(turns=(), valid=True, winner=1, termination_type="normal"):
    return MatchResult(
        case_id="case",
        valid=valid,
        winner=winner,
        termination_type=termination_type,
        seed=7,
        evaluated_seat=0,
        turns=tuple(turns),
        elapsed_time_s=0.1,
        engine_hash="hash",
        error=None if valid else "external failure",
    )


def test_dense_sample_reports_both_seats_without_mixing_perspective():
    seat_zero = dense_sample(state(), 0, 0, "initial")
    seat_one = dense_sample(state(), 1, 0, "initial")

    assert seat_zero.target_territory == 2
    assert seat_zero.opponent_territory == 1
    assert seat_zero.territory_margin == 1
    assert seat_zero.territory_share == pytest.approx(2 / 3)
    assert seat_zero.target_army == 9
    assert seat_zero.opponent_army == 4
    assert seat_zero.army_margin == 5
    assert seat_zero.target_coins == 30
    assert seat_zero.opponent_coins == 10
    assert seat_zero.coin_share == 0.75
    assert seat_one.target_territory == 1
    assert seat_one.territory_margin == -1
    assert seat_one.army_margin == -5
    assert seat_one.coin_share == 0.25


def test_zero_share_denominators_remain_missing():
    empty = state()
    for cell in empty["cells"].values():
        cell["player"] = -1
        cell["army"] = 0
    empty["coins"] = [0, 0]

    sample = dense_sample(empty, 0, 0, "terminal")

    assert sample.territory_share is None
    assert sample.army_share is None
    assert sample.coin_share is None


def test_main_pressure_uses_graph_distance_around_mountains():
    blocked = dense_sample(pressure_state(mountain=True), 0, 0, "initial")
    open_path = dense_sample(pressure_state(mountain=False), 0, 0, "initial")

    assert blocked.target_attack_mass_near_opponent_main == 0
    assert open_path.target_attack_mass_near_opponent_main == 2
    assert blocked.opponent_defense_mass_near_opponent_main == 6
    assert blocked.pressure_for == -6
    assert blocked.net_main_pressure == 3


def test_missing_main_marks_pressure_missing():
    missing = pressure_state()
    missing["generals"].pop("1")

    sample = dense_sample(missing, 0, 0, "terminal")

    assert sample.opponent_alive is False
    assert sample.pressure_for is None
    assert sample.pressure_against is None
    assert sample.net_main_pressure is None


def test_trace_samples_initial_completed_round_and_player_zero_terminal():
    before_one = state(1)
    after_zero = state(1)
    after_one = state(2)
    terminal = state(2)
    turns = (
        TurnRecord(0, 1, 0, "s0", before_one, ((8,),), "s1", after_zero),
        TurnRecord(1, 1, 1, "s1", after_zero, ((8,),), "s2", after_one),
        TurnRecord(2, 2, 0, "s2", after_one, ((8,),), "s3", terminal),
    )

    trace = build_dense_trace(match(turns))

    assert [sample.sample_kind for sample in trace] == [
        "initial",
        "completed_round",
        "terminal",
    ]
    assert [sample.official_round for sample in trace] == [1, 2, 2]
    summary = summarize_dense_trace(match(turns), trace)
    assert summary.completed_rounds_survived == 1
    assert summary.terminated is True
    assert summary.truncated is False
    assert summary.outcome == "loss"


def test_external_failure_is_truncated_not_terminated():
    turn = TurnRecord(
        0, 1, 0, "s0", state(1), ((8,),), "s1", state(1)
    )
    failed = match((turn,), valid=False, winner=None, termination_type="process_error")

    summary = summarize_dense_trace(failed, build_dense_trace(failed))

    assert summary.terminated is False
    assert summary.truncated is True
    assert summary.outcome == "invalid"


def test_episode_summary_keeps_terminal_mean_and_round_auc():
    trace = (
        dense_sample(ratio_state(1, 1), 0, 0, "initial"),
        dense_sample(ratio_state(2, 2), 0, 1, "completed_round"),
        dense_sample(ratio_state(3, 3), 0, 2, "terminal"),
    )

    summary = summarize_dense_trace(match(), trace)

    assert summary.territory_share.terminal == 0.75
    assert summary.territory_share.time_average == pytest.approx(0.5)
    assert summary.territory_share.auc == pytest.approx(1.0)


def test_episode_summary_does_not_interpolate_missing_values():
    zero_coins = ratio_state(2, 2)
    zero_coins["coins"] = [0, 0]
    trace = (
        dense_sample(ratio_state(1, 1), 0, 0, "initial"),
        dense_sample(zero_coins, 0, 1, "terminal"),
    )

    summary = summarize_dense_trace(match(), trace)

    assert summary.coin_share.terminal is None
    assert summary.coin_share.minimum is None
    assert summary.coin_share.maximum is None
    assert summary.coin_share.time_average is None
    assert summary.coin_share.auc is None


def test_dense_artifacts_are_complete_json_records(tmp_path):
    turn = TurnRecord(
        0, 1, 0, "s0", state(1), ((8,),), "s1", state(1)
    )
    trace, summary = persist_dense_diagnostics(match((turn,)), tmp_path)

    records = [
        json.loads(line)
        for line in (tmp_path / "dense-trace.jsonl").read_text().splitlines()
    ]
    saved_summary = json.loads((tmp_path / "dense-summary.json").read_text())
    assert len(records) == len(trace) == 2
    assert saved_summary["case_id"] == summary.case_id == "case"

