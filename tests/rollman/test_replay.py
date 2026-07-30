from pathlib import Path

import pytest

from agentbench_frame.games.rollman.replay import ReplayError, load_replay


FIXTURES = Path(__file__).with_name("fixtures")


def test_replay_preserves_raw_records_and_distinguishes_terminal_frame():
    replay = load_replay(FIXTURES / "replay_complete.jsonl")

    assert replay.initializations[0]["level"] == 1
    assert len(replay.rounds) == 5
    assert replay.terminal["StopReason"] == "time is up"
    assert replay.final_score == (-58, 135)
    assert replay.raw_sha256
    assert replay.normalized_sha256


def test_replay_rejects_truncated_json_instead_of_scoring_it():
    with pytest.raises(ReplayError, match="line 2"):
        load_replay(FIXTURES / "replay_malformed.jsonl")

