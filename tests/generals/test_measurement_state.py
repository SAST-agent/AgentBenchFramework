from copy import deepcopy
import os
from pathlib import Path

import pytest

from agentbench_frame.generals.engine import OfficialGeneralsEngine


@pytest.fixture
def engine_root():
    value = os.environ.get("AGENTBENCH_ASSET_ROOT")
    if not value:
        pytest.skip("AGENTBENCH_ASSET_ROOT is not set")
    return (
        Path(value)
        / "backend_sources/corpus/28_generals/logic/gamecode_logic"
    )


def test_measurement_state_round_trips_every_semantic_field(
    engine_root,
    tmp_path,
):
    engine = OfficialGeneralsEngine(
        engine_root,
        289101,
        tmp_path / "a.jsonl",
    )
    engine.state.round = 17
    engine.state.coin = [123, 77]
    engine.state.rest_move_step = [1, 2]
    engine.state.super_weapon_unlocked = [True, False]
    engine.state.super_weapon_cd = [0, 4]
    engine.state.tech_level = [[3, 1, 0, 1], [2, 0, 1, 0]]
    engine.state.generals[0].rest_move = 3
    farmer = next(
        general
        for general in engine.state.generals
        if type(general).__name__ == "Farmer"
    )
    farmer.defense_level = 1.5
    farmer.rest_move = 4

    weapon_outcome = engine.apply_primitive(0, (6, 2, 0, 0))
    assert weapon_outcome.valid is True
    engine.state.board[0][0].weapon_activate = [
        engine.state.active_super_weapon[0]
    ]

    before = engine.measurement_state(actor=0)
    restored = OfficialGeneralsEngine.from_measurement_state(
        engine_root,
        before,
        tmp_path / "b.jsonl",
    )

    assert restored.measurement_state(actor=0) == before
    assert (
        restored.measurement_state_id(actor=0)
        == engine.measurement_state_id(actor=0)
    )
    assert before["state"]["rest_move_step"] == [1, 2]
    assert all(
        "rest_move" in item
        for item in before["state"]["generals"]
    )
    restored_farmer = next(
        general
        for general in before["state"]["generals"]
        if general["type"] == "Farmer"
    )
    assert restored_farmer["defense_level"] == 1.5
    assert before["state"]["board"][0]["weapon_indices"] == [0]


def test_measurement_state_rejects_missing_movement_budget(
    engine_root,
    tmp_path,
):
    from agentbench_frame.generals.measurement_state import (
        MeasurementStateError,
    )

    engine = OfficialGeneralsEngine(engine_root, 289101, tmp_path / "a.jsonl")
    payload = deepcopy(engine.measurement_state(actor=1))
    del payload["state"]["rest_move_step"]

    with pytest.raises(
        MeasurementStateError,
        match="missing state field: rest_move_step",
    ):
        OfficialGeneralsEngine.from_measurement_state(
            engine_root,
            payload,
            tmp_path / "b.jsonl",
        )


def test_apply_primitive_does_not_turn_illegality_into_a_loss(
    engine_root,
    tmp_path,
):
    engine = OfficialGeneralsEngine(engine_root, 289101, tmp_path / "a.jsonl")
    winner_before = engine.state.winner

    outcome = engine.apply_primitive(0, (1, -1, -1, 1, 10))

    assert outcome.valid is False
    assert outcome.terminal is False
    assert outcome.winner is None
    assert engine.state.winner == winner_before


def test_measurement_state_id_includes_actor(engine_root, tmp_path):
    engine = OfficialGeneralsEngine(engine_root, 289101, tmp_path / "a.jsonl")

    assert engine.measurement_state_id(0) != engine.measurement_state_id(1)
