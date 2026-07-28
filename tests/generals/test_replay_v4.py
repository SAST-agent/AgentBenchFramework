from agentbench_frame.generals.dense import (
    DenseEpisodeSummary,
    DenseMetricSummary,
)
from agentbench_frame.generals.replay import (
    DecisionRecord,
    LearningReplay,
    build_critical_learning_evidence,
)


def _metric(value):
    return DenseMetricSummary(value, value, value, value, value)


def _summary():
    fields = {
        name: _metric(0)
        for name in (
            "target_territory",
            "opponent_territory",
            "territory_margin",
            "territory_share",
            "target_army",
            "opponent_army",
            "army_margin",
            "army_share",
            "target_coins",
            "opponent_coins",
            "coin_margin",
            "coin_share",
            "pressure_for",
            "pressure_against",
            "net_main_pressure",
        )
    }
    return DenseEpisodeSummary(
        case_id="learn4-high-s285101-p0",
        evaluated_seat=0,
        outcome="loss",
        terminal_round=10,
        completed_rounds_survived=9,
        termination_type="normal",
        terminated=True,
        truncated=False,
        sample_count=10,
        **fields,
    )


def _state(
    index,
    *,
    own_territory=4,
    own_army=40,
    pressure=False,
    opportunity=False,
):
    cells = {
        "0,0": {
            "type": 0,
            "player": 0,
            "army": 5 if pressure else own_army - (own_territory - 1),
            "general_id": 0,
        },
        "0,4": {
            "type": 0,
            "player": 1,
            "army": 20,
            "general_id": 1,
        },
    }
    for offset in range(1, own_territory):
        cells[f"1,{offset}"] = {
            "type": 0,
            "player": 0,
            "army": 1,
            "general_id": None,
        }
    if pressure:
        cells["0,1"] = {
            "type": 0,
            "player": 1,
            "army": 20,
            "general_id": None,
        }
    generals = {
        "0": {
            "id": 0,
            "type": "main",
            "player": 0,
            "position": [0, 0],
            "produce_level": 1,
            "defense_level": 1,
            "mobility_level": 1,
        },
        "1": {
            "id": 1,
            "type": "main",
            "player": 1,
            "position": [0, 4],
            "produce_level": 1,
            "defense_level": 1,
            "mobility_level": 1,
        },
    }
    if opportunity:
        cells["2,2"] = {
            "type": 0,
            "player": -1,
            "army": 3,
            "general_id": 2,
        }
        generals["2"] = {
            "id": 2,
            "type": "resource",
            "player": -1,
            "position": [2, 2],
            "produce_level": 1,
            "defense_level": 1,
            "mobility_level": 1,
        }
    return {
        "round": index + 1,
        "my_seat": 0,
        "coins": [5, 10],
        "tech_level": [[2, 0, 0, 0], [2, 0, 0, 0]],
        "cells": cells,
        "generals": generals,
    }


def _replay():
    states = [
        _state(0),
        _state(1),
        _state(2, pressure=True),
        _state(3, own_territory=6, own_army=60),
        _state(4, own_territory=1, own_army=55),
        _state(5, own_territory=4, own_army=100),
        _state(6, own_territory=4, own_army=10),
        _state(7, opportunity=True),
        _state(8),
        _state(9),
    ]
    decisions = tuple(
        DecisionRecord(
            state_id=f"state-{index}",
            round_number=index + 1,
            seat=0,
            state=state,
            action=(
                ((1, 0, 0, 0, 1), (8,))
                if index == 1
                else ((8,),)
            ),
            outcome="loss",
        )
        for index, state in enumerate(states)
    )
    return LearningReplay(
        replay_id="learn4-high-s285101-p0",
        seed=285101,
        evaluated_seat=0,
        opponent_tier="high",
        termination_type="normal",
        decisions=decisions,
    )


def test_critical_windows_select_declared_reasons_and_order_them():
    evidence, selection = build_critical_learning_evidence(
        _replay(),
        _summary(),
        (),
    )

    assert selection.selected_state_ids == (
        "state-0",
        "state-1",
        "state-2",
        "state-3",
        "state-5",
        "state-7",
        "state-8",
        "state-9",
    )
    assert selection.reasons["state-0"] == ("first_decision",)
    assert "first_non_end_action" in selection.reasons["state-1"]
    assert "first_main_pressure" in selection.reasons["state-2"]
    assert "before_steepest_territory_drop" in selection.reasons["state-3"]
    assert "before_steepest_army_drop" in selection.reasons["state-5"]
    assert "first_strategic_opportunity" in selection.reasons["state-7"]
    assert selection.omitted_decision_count == 2
    assert tuple(item.state_id for item in evidence.decisions) == (
        selection.selected_state_ids
    )


def test_critical_window_features_are_state_derived_and_explainable():
    evidence, _ = build_critical_learning_evidence(
        _replay(),
        _summary(),
        (),
    )

    pressure = next(
        item for item in evidence.decisions
        if item.state_id == "state-2"
    )
    assert pressure.decision_class == "end_only"
    assert pressure.features["own_main_army"] == 5
    assert pressure.features["adjacent_enemy_pressure"] == 19
    assert pressure.features["owned_territory"] == 4
    assert pressure.features["movable_stack_count"] >= 1
    assert pressure.features["largest_movable_stack"] >= 4
    assert pressure.features["largest_movable_stacks"][0]["position"] == [0, 0]
    assert {
        item["type"] for item in pressure.features["visible_generals"]
    } == {"main"}


def test_critical_windows_deduplicate_when_criteria_select_same_state():
    replay = LearningReplay(
        replay_id="learn4-high-s285101-p0",
        seed=285101,
        evaluated_seat=0,
        opponent_tier="high",
        termination_type="normal",
        decisions=_replay().decisions[:2],
    )

    evidence, selection = build_critical_learning_evidence(
        replay,
        _summary(),
        (),
    )

    assert selection.selected_state_ids == ("state-0", "state-1")
    assert len(evidence.decisions) == 2
    assert selection.omitted_decision_count == 0
