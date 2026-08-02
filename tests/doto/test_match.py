import json
from pathlib import Path

import pytest

from agentbench_frame.doto.match import DotoMatchError, run_match


FIXTURES = Path(__file__).parent / "fixtures"


def test_routes_two_factions_and_preserves_trace(tmp_path):
    result = run_match(
        FIXTURES / "fake_ai.py",
        FIXTURES / "fake_ai.py",
        seed=11,
        output_dir=tmp_path,
        tag="pair",
        server_dir=FIXTURES / "fake_server",
        test_only=True,
    )

    assert result.winner == 0
    assert result.scores == (12.0, 7.0)
    assert result.terminated_by == "normal"
    assert result.replay_path.is_file()
    rows = [json.loads(line) for line in result.trace_path.read_text().splitlines()]
    assert {row["kind"] for row in rows} == {"observation", "action", "final"}
    assert [row["seq"] for row in rows] == list(range(len(rows)))


def test_server_crash_is_not_a_loss(tmp_path):
    with pytest.raises(DotoMatchError, match="process_exit"):
        run_match(
            FIXTURES / "fake_ai.py",
            FIXTURES / "fake_ai.py",
            seed=11,
            output_dir=tmp_path,
            tag="crash",
            server_dir=FIXTURES / "crashing_server",
            test_only=True,
        )


def test_test_only_match_cannot_enter_results_tree(tmp_path):
    output = tmp_path / "runs" / "23_doto" / "agent" / "run"
    with pytest.raises(ValueError, match="test-only"):
        run_match(
            FIXTURES / "fake_ai.py",
            FIXTURES / "fake_ai.py",
            seed=11,
            output_dir=output,
            tag="forbidden",
            server_dir=FIXTURES / "fake_server",
            test_only=True,
        )
