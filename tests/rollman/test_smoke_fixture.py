import json
import subprocess
import sys

import numpy as np
import pytest


def test_make_state_matches_frozen_sdk_types_and_walkability():
    """Catch smoke fixtures that pass list-only policies or all-wall boards."""
    from agentbench_frame.games.rollman.rollman_smoke_fixture import make_state

    state = make_state(
        level=1,
        round_id=7,
        pacman=(5, 5),
        ghosts=[(6, 5), (7, 5), (8, 5)],
        board_size=12,
        board_overrides=[(5, 6, 6)],
    )
    value = state.gamestate_to_statedict()

    assert state.board.dtype.kind in "iu"
    assert np.all(state.board[0, :] == 0)
    assert np.all(state.board[:, 0] == 0)
    assert state.board[5, 5] != 0
    assert state.board[6, 5] != 0
    assert state.board[5, 6] == 6
    assert isinstance(state.pacman_pos, np.ndarray)
    assert all(isinstance(item, np.ndarray) for item in state.ghosts_pos)
    assert isinstance(state.pacman_skill_status, list)
    assert isinstance(value["pacman_skill_status"], np.ndarray)
    assert value["pacman_coord"].tolist() == [5, 5]
    assert [item.tolist() for item in value["ghosts_coord"]] == [
        [6, 5],
        [7, 5],
        [8, 5],
    ]
    assert value["score"] == [0, 0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ghosts": [(2, 2)]}, "exactly three"),
        ({"pacman": (0, 2)}, "walkable interior"),
        ({"pacman": (20, 20)}, "inside board"),
        ({"board_overrides": [(2, 2, 99)]}, "tile"),
    ],
)
def test_make_state_rejects_invalid_frozen_game_geometry(kwargs, message):
    """Catch invalid scenarios before a candidate can claim activation."""
    from agentbench_frame.games.rollman.rollman_smoke_fixture import make_state

    values = {
        "level": 1,
        "round_id": 1,
        "pacman": (2, 2),
        "ghosts": [(3, 3), (4, 4), (5, 5)],
        "board_size": 8,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        make_state(**values)


def test_verify_scenario_calls_public_entry_in_order_and_writes_hashes(tmp_path):
    """Catch internal-helper-only smoke and stale unbound result artifacts."""
    from agentbench_frame.games.rollman.rollman_smoke_fixture import (
        sha256_file,
        verify_scenario,
    )

    workspace = tmp_path / "candidate"
    workspace.mkdir()
    policy = workspace / "ai.py"
    policy.write_text(
        "calls = 0\n"
        "def ai_func(state):\n"
        "    global calls\n"
        "    calls += 1\n"
        "    prefix = 'parent:' if calls == 1 else 'new-mechanism:'\n"
        "    return {'action': 4, 'memory_id': prefix + str(state.round)}\n",
        encoding="utf-8",
    )
    scenario = workspace / ".agentbench" / "smoke_scenario.json"
    scenario.parent.mkdir()
    scenario.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "states": [
                    {
                        "level": 1,
                        "round_id": 4,
                        "pacman": [3, 3],
                        "ghosts": [[4, 4], [5, 5], [6, 6]],
                        "board_size": 9,
                    },
                    {
                        "level": 1,
                        "round_id": 5,
                        "pacman": [3, 4],
                        "ghosts": [[4, 4], [5, 5], [6, 6]],
                        "board_size": 9,
                    },
                ],
                "activation_state_index": 1,
                "preservation_state_index": 0,
                "activation_memory_prefix": "new-mechanism:",
                "preservation_forbidden_prefix": "new-mechanism:",
                "preservation_memory_prefix": "parent:",
            }
        ),
        encoding="utf-8",
    )
    output = workspace / ".agentbench" / "candidate_smoke_result.json"
    output.write_text('{"status":"forged"}\n', encoding="utf-8")

    result = verify_scenario(
        workspace=workspace,
        scenario_path=scenario,
        output_path=output,
    )

    assert result["status"] == "complete"
    assert result["policy_sha256"] == sha256_file(policy)
    assert result["scenario_sha256"] == sha256_file(scenario)
    assert result["decisions"] == [
        {"state_index": 0, "action": 4, "memory_id": "parent:4"},
        {"state_index": 1, "action": 4, "memory_id": "new-mechanism:5"},
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize(
    ("policy_body", "activation_prefix", "forbidden_prefix", "message"),
    [
        ("return {'action': 9, 'memory_id': 'new:'}", "new:", "new:", "0..4"),
        ("return {'action': 1, 'memory_id': 'parent:'}", "new:", "new:", "activation"),
        ("return {'action': 1, 'memory_id': 'new:'}", "new:", "new:", "preservation"),
    ],
)
def test_verify_scenario_fails_closed_on_invalid_public_decision(
    tmp_path,
    policy_body,
    activation_prefix,
    forbidden_prefix,
    message,
):
    """Catch policy results that do not prove both scope-contract branches."""
    from agentbench_frame.games.rollman.rollman_smoke_fixture import verify_scenario

    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text(
        f"def ai_func(state):\n    {policy_body}\n",
        encoding="utf-8",
    )
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "states": [
                    {
                        "level": 1,
                        "round_id": 1,
                        "pacman": [2, 2],
                        "ghosts": [[3, 3], [4, 4], [5, 5]],
                        "board_size": 8,
                    }
                ],
                "activation_state_index": 0,
                "preservation_state_index": 0,
                "activation_memory_prefix": activation_prefix,
                "preservation_forbidden_prefix": forbidden_prefix,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        verify_scenario(
            workspace=workspace,
            scenario_path=scenario,
            output_path=tmp_path / "result.json",
        )


def test_smoke_fixture_cli_returns_nonzero_without_public_entry(tmp_path):
    """Catch executable verifier paths that silently accept missing ai_func."""
    from agentbench_frame.games.rollman import rollman_smoke_fixture

    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "ai.py").write_text("value = 1\n", encoding="utf-8")
    scenario = tmp_path / "scenario.json"
    scenario.write_text('{"schema_version":"1.0","states":[]}\n', encoding="utf-8")

    completed = subprocess.run(
        (
            sys.executable,
            str(rollman_smoke_fixture.__file__),
            "--workspace",
            str(workspace),
            "--scenario",
            str(scenario),
            "--output",
            str(tmp_path / "result.json"),
        ),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "smoke verification failed" in completed.stderr
