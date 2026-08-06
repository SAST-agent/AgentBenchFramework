import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser


def _parser():
    parser = argparse.ArgumentParser()
    register_parser(parser.add_subparsers(dest="command"))
    return parser


def _arguments(command):
    values = [
        "generals", command,
        "--agentbench-root", "/assets",
        "--manifest", "/pilot.toml",
        "--challenge-manifest", "/v8-clean.toml",
        "--replay-skill", "skills/replay-analysis-v2/SKILL.md",
        "--data-dir", "/data",
        "--parent-run", "/runs/v7",
        "--expected-parent-hash", "7" * 64,
        "--codex-executable", "codex",
        "--provider-timeout", "1800",
    ]
    if command == "recover-v8":
        values.extend(["--failed-run", "/runs/failed-v8"])
    return values


def test_cli_registers_clean_room_v8_without_retry():
    iterate = _parser().parse_args(_arguments("iterate-v8"))
    recover = _parser().parse_args(_arguments("recover-v8"))
    assert str(iterate.challenge_manifest) == "/v8-clean.toml"
    assert str(iterate.parent_run) == "/runs/v7"
    assert str(recover.failed_run) == "/runs/failed-v8"
    with pytest.raises(SystemExit):
        _parser().parse_args(["generals", "retry-v8"])


def test_cli_routes_iterate_v8(monkeypatch, capsys):
    args = _parser().parse_args(_arguments("iterate-v8"))
    captured = {}

    class FakePipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def run(self):
            return SimpleNamespace(
                status="complete", run_dir=Path("/runs/v8"), raw_score=0.0,
                evo_score_8=2 / 3, gain_8=2 / 3,
                validation_passed=True, formal_attempted=True,
                performance_target_met=True, global_act_count=9,
                round_act_count=1, runnable=True,
            )

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(cli, "CodexProvider", lambda **kwargs: object())
    monkeypatch.setattr(cli, "GeneralsHLRound8Pipeline", FakePipeline)
    assert cli.handle(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evo_score_8"] == 2 / 3
    assert payload["performance_target_met"] is True
    assert captured["parent_run_dir"] == Path("/runs/v7")


def test_cli_registers_v8_kl_extension_and_recovery():
    common = [
        "generals", "extend-policy-kl-v8",
        "--agentbench-root", "/assets", "--manifest", "/pilot.toml",
        "--reference-manifest", "/reference-v2.toml",
        "--source-run", "/runs/kl-v7", "--target-run", "/runs/v8",
        "--expected-target-hash", "8" * 64, "--data-dir", "/data",
    ]
    extend = _parser().parse_args(common)
    recover = _parser().parse_args(
        ["generals", "recover-policy-kl-v8", *common[2:],
         "--failed-run", "/runs/failed-kl-v8"]
    )
    assert extend.target_run == Path("/runs/v8")
    assert recover.failed_run == Path("/runs/failed-kl-v8")
