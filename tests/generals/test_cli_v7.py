import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser


def _arguments(command: str) -> list[str]:
    arguments = [
        "generals",
        command,
        "--agentbench-root",
        "/assets",
        "--manifest",
        "/benchmark/pilot.toml",
        "--challenge-manifest",
        "/benchmark/v7-champion.toml",
        "--replay-skill",
        "skills/replay/SKILL.md",
        "--data-dir",
        "/data",
        "--parent-run",
        "/runs/v6",
        "--expected-parent-hash",
        "6" * 64,
        "--codex-executable",
        "/tools/codex",
        "--provider-timeout",
        "1800",
    ]
    if command == "recover-v7":
        arguments.extend(["--failed-run", "/runs/failed-v7"])
    return arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    return parser


def _result(*, claim=True):
    return SimpleNamespace(
        status="complete",
        run_dir=Path("/runs/v7"),
        raw_score=0.0,
        evo_score_7=0.5,
        gain_7=0.5,
        validation_passed=True,
        sealed_status="passed" if claim else "failed",
        champion_claim=claim,
        global_act_count=8,
        round_act_count=1,
        runnable=True,
    )


def test_generals_cli_registers_iterate_and_recover_v7_contract():
    iterate = _parser().parse_args(_arguments("iterate-v7"))
    recover = _parser().parse_args(_arguments("recover-v7"))

    for args in (iterate, recover):
        assert str(args.agentbench_root) == "/assets"
        assert str(args.manifest) == "/benchmark/pilot.toml"
        assert str(args.challenge_manifest) == (
            "/benchmark/v7-champion.toml"
        )
        assert str(args.replay_skill) == "skills/replay/SKILL.md"
        assert str(args.data_dir) == "/data"
        assert str(args.parent_run) == "/runs/v6"
        assert args.expected_parent_hash == "6" * 64
        assert args.codex_executable == "/tools/codex"
        assert args.provider_timeout == 1800
    assert str(recover.failed_run) == "/runs/failed-v7"


@pytest.mark.parametrize(
    "flag",
    [
        "--challenge-manifest",
        "--replay-skill",
        "--parent-run",
        "--expected-parent-hash",
        "--codex-executable",
        "--provider-timeout",
    ],
)
def test_generals_cli_requires_explicit_v7_inputs(flag):
    arguments = _arguments("iterate-v7")
    index = arguments.index(flag)
    del arguments[index:index + 2]

    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(arguments)

    assert exc.value.code == 2


def test_generals_cli_routes_iterate_v7_and_prints_separate_claim(
    monkeypatch,
    capsys,
):
    args = _parser().parse_args(_arguments("iterate-v7"))
    captured = {}
    provider_config = {}

    class FakePipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def run(self):
            return _result(claim=False)

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(
        cli,
        "CodexProvider",
        lambda **kwargs: provider_config.update(kwargs) or object(),
    )
    monkeypatch.setattr(cli, "GeneralsHLRound7Pipeline", FakePipeline)

    assert cli.handle(args) == 0
    assert captured == {
        "agentbench_root": Path("/assets"),
        "manifest_path": Path("/benchmark/pilot.toml"),
        "challenge_manifest_path": Path("/benchmark/v7-champion.toml"),
        "replay_skill_path": Path("skills/replay/SKILL.md"),
        "parent_run_dir": Path("/runs/v6"),
        "expected_parent_hash": "6" * 64,
        "data_dir": Path("/data"),
        "provider": captured["provider"],
    }
    assert provider_config == {
        "executable": "/tools/codex",
        "timeout_s": 1800,
        "sandbox": "workspace-write",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "champion_claim": False,
        "evo_score_7": 0.5,
        "gain_7": 0.5,
        "global_act_count": 8,
        "raw_score": 0.0,
        "round_act_count": 1,
        "run_dir": "/runs/v7",
        "runnable": True,
        "sealed_status": "failed",
        "status": "complete",
        "validation_passed": True,
    }


def test_generals_cli_routes_recover_v7(monkeypatch, capsys):
    args = _parser().parse_args(_arguments("recover-v7"))
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
    monkeypatch.setattr(cli, "GeneralsHLRound7Pipeline", FakePipeline)

    assert cli.handle(args) == 0
    assert captured["failed_run_dir"] == Path("/runs/failed-v7")
    payload = json.loads(capsys.readouterr().out)
    assert payload["champion_claim"] is True
    assert payload["sealed_status"] == "passed"
