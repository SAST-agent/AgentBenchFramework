import json
import os
from pathlib import Path
import sys

import pytest

from agentbench_frame.generals.assets import load_pilot_config, resolve_assets
from agentbench_frame.generals.match import GeneralsMatchRunner
from agentbench_frame.generals.models import AgentProcessSpec, MatchCase


FAKE = Path(__file__).parent / "fixtures" / "fake_agent.py"


@pytest.fixture
def real_assets():
    value = os.environ.get("AGENTBENCH_ASSET_ROOT")
    if not value:
        pytest.skip("AGENTBENCH_ASSET_ROOT is not set")
    root = Path(value)
    config = load_pilot_config(
        root / "backend_sources/corpus/28_generals/benchmark/pilot-v1.toml"
    )
    return resolve_assets(config, root)


def fake(mode, tmp_path, player):
    return AgentProcessSpec(
        agent_id=f"fake-{player}",
        argv=(sys.executable, str(FAKE), mode),
        cwd=tmp_path,
        env={},
    )


def test_match_sends_player_specific_seats_and_records_every_turn(real_assets, tmp_path):
    runner = GeneralsMatchRunner(real_assets)
    result = runner.run(
        case=MatchCase(case_id="case", seed=280101, evaluated_seat=1),
        players=(fake("valid", tmp_path, 0), fake("valid", tmp_path, 1)),
        artifact_dir=tmp_path / "match",
    )
    assert result.seed == 280101
    assert result.turns[0].player == 0
    assert result.turns[1].player == 1
    metadata = json.loads((tmp_path / "match" / "metadata.json").read_text())
    assert metadata["evaluated_seat"] == 1
    assert (tmp_path / "match" / "replay.jsonl").exists()


def test_agent_decision_timeout_is_valid_tle_loss(real_assets, tmp_path):
    runner = GeneralsMatchRunner(real_assets)
    result = runner.run(
        MatchCase("timeout", 280101, 0),
        (fake("hang", tmp_path, 0), fake("valid", tmp_path, 1)),
        tmp_path / "match",
    )
    assert result.valid is True
    assert result.winner == 1
    assert result.termination_type == "time_limit"


def test_process_exit_is_invalid(real_assets, tmp_path):
    runner = GeneralsMatchRunner(real_assets)
    result = runner.run(
        MatchCase("exit", 280101, 0),
        (fake("exit", tmp_path, 0), fake("valid", tmp_path, 1)),
        tmp_path / "match",
    )
    assert result.valid is False
    assert result.termination_type == "process_error"
