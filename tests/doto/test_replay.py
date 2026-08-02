import json
import zipfile
from pathlib import Path

import pytest

from agentbench_frame.doto.replay import DotoReplayError, iter_replay, summarize_replay


FIXTURES = Path(__file__).parent / "fixtures"


def test_real_replay_matches_expected_summary():
    expected = json.loads((FIXTURES / "real_short_replay.expected.json").read_text())

    summary = summarize_replay(FIXTURES / "real_short_replay.zip")

    assert list(summary.final_scores) == expected["final_scores"]
    assert summary.last_frame == expected["last_frame"]
    assert summary.event_counts == expected["event_counts"]


def test_parser_never_executes_stringified_lists(tmp_path):
    replay = tmp_path / "malicious.zip"
    frames = [{
        "frame": 1,
        "humans": "__import__('os').system('false')",
        "fireballs": "[]",
        "meteors": "[]",
        "balls": "[]",
        "scores": "[0, 0]",
        "bonus": "[]",
        "events": "[]",
    }]
    with zipfile.ZipFile(replay, "w") as archive:
        archive.writestr("replay.json", json.dumps(frames))

    with pytest.raises(DotoReplayError, match="invalid humans"):
        list(iter_replay(replay))


def test_rejects_parent_path_member(tmp_path):
    replay = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(replay, "w") as archive:
        archive.writestr("../replay.json", "[]")

    with pytest.raises(DotoReplayError, match="unsafe ZIP member"):
        list(iter_replay(replay))
