import pytest

from agentbench_frame.miracle.strategy_loader import (
    StrategyValidationError,
    load_candidate,
    save_source,
    validate_candidate,
)


VALID_SOURCE = '''
from agentbench_frame.miracle.agent_bridge import MiracleAgent

class CandidateAgent(MiracleAgent):
    def choose_cards(self, camp):
        return {"artifacts": ["HolyLight"], "creatures": ["Archer", "Swordsman", "BlackBat"]}

    def act(self, obs):
        return {"operation_type": "endround", "operation_parameters": {}}
'''


def test_saves_and_loads_separate_strategy_snapshots(tmp_path):
    first = tmp_path / "v0.py"
    second = tmp_path / "v1.py"
    save_source(first, VALID_SOURCE)
    save_source(second, VALID_SOURCE.replace("endround", "surrender"))

    agent0 = load_candidate(first, "candidate_v0")
    agent1 = load_candidate(second, "candidate_v1")

    validate_candidate(agent0)
    validate_candidate(agent1)
    assert type(agent0).__module__ != type(agent1).__module__
    assert agent0.act({})["operation_type"] == "endround"
    assert agent1.act({})["operation_type"] == "surrender"


def test_snapshot_refuses_different_overwrite(tmp_path):
    path = tmp_path / "strategy.py"
    save_source(path, VALID_SOURCE)
    save_source(path, VALID_SOURCE)

    with pytest.raises(StrategyValidationError) as raised:
        save_source(path, VALID_SOURCE + "\n# changed")

    assert raised.value.stage == "snapshot"


@pytest.mark.parametrize(
    ("source", "stage"),
    [
        ("class CandidateAgent(:\n", "compile"),
        ("VALUE = 1\n", "class"),
    ],
)
def test_reports_source_validation_stage(tmp_path, source, stage):
    path = tmp_path / f"{stage}.py"
    save_source(path, source)

    with pytest.raises(StrategyValidationError) as raised:
        load_candidate(path, f"candidate_{stage}")

    assert raised.value.stage == stage


def test_rejects_bad_action_shape(tmp_path):
    path = tmp_path / "bad.py"
    save_source(path, VALID_SOURCE.replace(
        'return {"operation_type": "endround", "operation_parameters": {}}',
        'return {"wrong": "shape"}',
    ))
    agent = load_candidate(path, "candidate_bad_action")

    with pytest.raises(StrategyValidationError) as raised:
        validate_candidate(agent)

    assert raised.value.stage == "action_shape"
