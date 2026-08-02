import json
from pathlib import Path

from agentbench_frame.doto.cli import build_parser, main


FIXTURES = Path(__file__).parent / "fixtures"


def test_only_five_public_subcommands_exist():
    parser = build_parser()
    action = next(item for item in parser._actions if item.dest == "cmd")

    assert set(action.choices) == {"build", "match", "replay", "ig", "loop"}


def test_build_command_compiles_and_prints_json(tmp_path, capsys):
    output = tmp_path / "candidate"

    exit_code = main([
        "build",
        "--player-ai", str(FIXTURES / "valid_playerAI.cpp"),
        "--output-dir", str(output),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["executable"] == str(output.resolve() / "main.out")


def test_match_command_runs_test_server(tmp_path, capsys):
    fake_ai = FIXTURES / "fake_ai.py"

    exit_code = main([
        "match",
        "--agent0", str(fake_ai),
        "--agent1", str(fake_ai),
        "--seed", "11",
        "--output-dir", str(tmp_path),
        "--tag", "cli-pair",
        "--server-dir", str(FIXTURES / "fake_server"),
        "--test-only",
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["winner"] == 0
    assert payload["metadata"]["test_only"] is True
