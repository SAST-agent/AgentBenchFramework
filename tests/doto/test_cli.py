import json
from pathlib import Path

from agentbench_frame.doto.cli import build_parser, main
from tests.doto.test_run_store import make_inputs


FIXTURES = Path(__file__).parent / "fixtures"


def test_population_commands_are_atomic_and_candidate_build_remains():
    parser = build_parser()
    action = next(item for item in parser._actions if item.dest == "cmd")

    assert set(action.choices) == {
        "build", "population", "evaluate", "match", "replay", "ig",
        "run", "iteration",
    }
    population = action.choices["population"]
    population_action = next(item for item in population._actions if item.dest == "population_cmd")
    assert set(population_action.choices) == {"verify", "build-train", "verify-sealed"}
    iteration = action.choices["iteration"]
    iteration_action = next(item for item in iteration._actions if item.dest == "iteration_cmd")
    assert set(iteration_action.choices) == {"begin", "build", "evaluate", "compare", "close"}
    run = action.choices["run"]
    run_action = next(item for item in run._actions if item.dest == "run_cmd")
    assert set(run_action.choices) == {"init", "status", "finalize", "export"}


def test_doto_has_no_framework_llm_surface():
    package = Path(__file__).parents[2] / "src/agentbench_frame/doto"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.rglob("*.py"))
    for forbidden in (
        "ChatCompletionsClient", "OPENAI_API_KEY", "api_key_env",
        "reasoning_effort", "max_total_tokens", "def _loop(",
    ):
        assert forbidden not in text
    assert "loop" not in build_parser()._subparsers._group_actions[0].choices


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


def test_replay_command_writes_event_jsonl(capsys, tmp_path):
    events = tmp_path / "events.jsonl"

    exit_code = main([
        "replay",
        "--path", str(FIXTURES / "real_short_replay.zip"),
        "--jsonl", str(events),
    ])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["final_scores"] == [12.0, 7.0]
    assert events.is_file()


def test_run_init_status_and_iteration_build_print_json(tmp_path, capsys):
    source, train, test, _ = make_inputs(tmp_path)
    skills_root = tmp_path / "skills"
    results = tmp_path / "DotoResults"
    assert main([
        "run", "init", "--agent", "codex", "--initial-player-ai", str(source),
        "--doto-results", str(results), "--run-id", "r1",
        "--train-bundle", str(train), "--test-bundle", str(test),
        "--skills-root", str(skills_root),
    ]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["state"] == "created"
    run_dir = Path(initialized["run_dir"])

    assert main(["run", "status", "--run-dir", str(run_dir)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["iterations"][0]["state"] == "iteration_open"

    assert main(["iteration", "build", "--run-dir", str(run_dir), "--iteration", "0"]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["state"] == "candidate_built"
    assert (run_dir / "iterations/iteration-0000/build/Maps/0.json").is_file()
