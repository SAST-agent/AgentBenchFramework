import json
from pathlib import Path

from agentbench_frame.doto.build import build_candidate


FIXTURES = Path(__file__).parent / "fixtures"


def test_builds_complete_player_ai_in_isolated_sdk(tmp_path):
    output = tmp_path / "build"
    result = build_candidate(FIXTURES / "valid_playerAI.cpp", output)

    assert result.exit_code == 0
    assert result.executable is not None
    assert result.executable.is_file()
    metadata = json.loads((output / "build.json").read_text())
    assert metadata["source_hash"] == result.source_hash


def test_failed_rebuild_removes_stale_executable(tmp_path):
    output = tmp_path / "build"
    assert build_candidate(FIXTURES / "valid_playerAI.cpp", output).exit_code == 0

    result = build_candidate(FIXTURES / "invalid_playerAI.cpp", output)

    assert result.exit_code != 0
    assert result.executable is None
    assert not (output / "main.out").exists()
