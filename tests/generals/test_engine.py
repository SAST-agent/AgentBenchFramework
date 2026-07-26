from pathlib import Path
import os

import pytest

from agentbench_frame.generals.engine import OfficialGeneralsEngine


@pytest.fixture
def engine_root():
    value = os.environ.get("AGENTBENCH_ASSET_ROOT")
    if not value:
        pytest.skip("AGENTBENCH_ASSET_ROOT is not set")
    return Path(value) / "backend_sources/corpus/28_generals/logic/gamecode_logic"


def test_same_seed_produces_same_initial_state(engine_root):
    one = OfficialGeneralsEngine(engine_root, seed=280101)
    two = OfficialGeneralsEngine(engine_root, seed=280101)
    assert one.initial_observation(0) == two.initial_observation(0)
    assert one.state_id() == two.state_id()


def test_player_one_end_turn_updates_round(engine_root):
    engine = OfficialGeneralsEngine(engine_root, seed=280101)
    assert engine.round_number == 1
    engine.apply_turn(0, ((8,),))
    engine.apply_turn(1, ((8,),))
    assert engine.round_number == 2


def test_illegal_official_command_is_valid_ia_loss(engine_root):
    engine = OfficialGeneralsEngine(engine_root, seed=280101)
    outcome = engine.apply_turn(0, ((1, -1, -1, 1, 10), (8,)))
    assert outcome.done is True
    assert outcome.winner == 1
    assert outcome.termination_type == "illegal_action"
