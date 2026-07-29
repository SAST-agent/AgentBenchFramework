import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser
import pytest


def _arguments(command: str) -> list[str]:
    arguments = [
        "generals",
        command,
        "--agentbench-root",
        "/assets",
        "--manifest",
        "/benchmark/pilot.toml",
        "--learning-manifest",
        "/benchmark/v6-learning.toml",
        "--replay-skill",
        "skills/replay/SKILL.md",
        "--data-dir",
        "/data",
        "--parent-run",
        "/runs/v5",
        "--expected-parent-hash",
        "5" * 64,
        "--codex-executable",
        "/tools/codex",
        "--provider-timeout",
        "900",
    ]
    if command == "recover-v6":
        arguments.extend(["--failed-run", "/runs/failed-v6"])
    return arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    return parser


def _result():
    return SimpleNamespace(
        status="complete",
        run_dir=Path("/runs/v6"),
        raw_score=0.0,
        evo_score_1=0.0,
        evo_score_2=0.0,
        evo_score_3=7 / 18,
        evo_score_4=4 / 18,
        evo_score_5=7 / 18,
        evo_score_6=0.5,
        gain_6=0.5,
        global_act_count=7,
        round_act_count=1,
        runnable=True,
    )


def test_generals_cli_registers_complete_iterate_and_recover_v6_surface():
    parser = _parser()

    iterate = parser.parse_args(_arguments("iterate-v6"))
    recover = parser.parse_args(_arguments("recover-v6"))

    for args in (iterate, recover):
        assert str(args.agentbench_root) == "/assets"
        assert str(args.manifest) == "/benchmark/pilot.toml"
        assert str(args.learning_manifest) == "/benchmark/v6-learning.toml"
        assert str(args.replay_skill) == "skills/replay/SKILL.md"
        assert str(args.data_dir) == "/data"
        assert str(args.parent_run) == "/runs/v5"
        assert args.expected_parent_hash == "5" * 64
        assert args.codex_executable == "/tools/codex"
        assert args.provider_timeout == 900
    assert str(recover.failed_run) == "/runs/failed-v6"


@pytest.mark.parametrize(
    "flag",
    ["--codex-executable", "--provider-timeout"],
)
def test_generals_cli_requires_explicit_v6_provider_configuration(flag):
    arguments = _arguments("iterate-v6")
    index = arguments.index(flag)
    del arguments[index:index + 2]

    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(arguments)

    assert exc.value.code == 2


def test_generals_cli_routes_all_iterate_v6_inputs(monkeypatch, capsys):
    args = _parser().parse_args(_arguments("iterate-v6"))
    captured = {}
    provider_config = {}

    class FakePipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def run(self):
            return _result()

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(
        cli,
        "CodexProvider",
        lambda **kwargs: provider_config.update(kwargs) or object(),
    )
    monkeypatch.setattr(cli, "GeneralsHLRound6Pipeline", FakePipeline)

    assert cli.handle(args) == 0
    assert captured == {
        "agentbench_root": Path("/assets"),
        "manifest_path": Path("/benchmark/pilot.toml"),
        "learning_manifest_path": Path("/benchmark/v6-learning.toml"),
        "replay_skill_path": Path("skills/replay/SKILL.md"),
        "parent_run_dir": Path("/runs/v5"),
        "expected_parent_hash": "5" * 64,
        "data_dir": Path("/data"),
        "provider": captured["provider"],
    }
    assert provider_config == {
        "executable": "/tools/codex",
        "timeout_s": 900,
        "sandbox": "workspace-write",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["evo_score_6"] == 0.5
    assert payload["gain_6"] == 0.5
    assert payload["global_act_count"] == 7
    assert payload["round_act_count"] == 1
    assert payload["runnable"] is True


def test_generals_cli_routes_failed_run_to_recover_v6(monkeypatch, capsys):
    args = _parser().parse_args(_arguments("recover-v6"))
    captured = {}

    class FakePipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def recover(self, failed_run_dir):
            captured["failed_run_dir"] = failed_run_dir
            return _result()

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(cli, "CodexProvider", lambda **kwargs: object())
    monkeypatch.setattr(cli, "GeneralsHLRound6Pipeline", FakePipeline)

    assert cli.handle(args) == 0
    assert captured["failed_run_dir"] == Path("/runs/failed-v6")
    assert captured["expected_parent_hash"] == "5" * 64
    assert captured["replay_skill_path"] == Path(
        "skills/replay/SKILL.md"
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["evo_score_6"] == 0.5
    assert payload["gain_6"] == 0.5
    assert payload["global_act_count"] == 7
    assert payload["round_act_count"] == 1
    assert payload["runnable"] is True
