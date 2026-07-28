import argparse
from types import SimpleNamespace

import agentbench_frame.generals.cli as cli
from agentbench_frame.generals.cli import register_parser


def _arguments() -> list[str]:
    return [
        "generals",
        "iterate-v5",
        "--agentbench-root",
        "/assets",
        "--manifest",
        "/benchmark/pilot.toml",
        "--learning-manifest",
        "/benchmark/v5-learning.toml",
        "--replay-skill",
        "skills/replay/SKILL.md",
        "--data-dir",
        "/data",
        "--parent-run",
        "/runs/v4",
        "--expected-parent-hash",
        "4" * 64,
        "--expected-rollback-hash",
        "3" * 64,
        "--campaign-budget-receipt",
        "/derived/v4-budget.json",
        "--codex-executable",
        "codex",
        "--provider-timeout",
        "900",
    ]


def test_generals_cli_registers_complete_iterate_v5_surface():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)

    args = parser.parse_args(_arguments())

    assert args.generals_command == "iterate-v5"
    assert str(args.learning_manifest) == "/benchmark/v5-learning.toml"
    assert str(args.replay_skill) == "skills/replay/SKILL.md"
    assert str(args.parent_run) == "/runs/v4"
    assert args.expected_parent_hash == "4" * 64
    assert args.expected_rollback_hash == "3" * 64
    assert str(args.campaign_budget_receipt) == (
        "/derived/v4-budget.json"
    )
    assert args.provider_timeout == 900


def test_generals_cli_routes_v5_rollback_inputs(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args(
        _arguments()
        + ["--prior-attempt-run", "/runs/failed-v5-prompt"]
    )
    captured = {}

    class FakePipeline:
        @classmethod
        def from_paths(cls, **kwargs):
            captured.update(kwargs)
            return cls()

        def run(self):
            return SimpleNamespace(
                status="complete",
                run_dir="/runs/v5",
                raw_score=0.0,
                evo_score_1=0.0,
                evo_score_2=None,
                evo_score_3=0.0,
                evo_score_4=7 / 18,
                evo_score_5=0.5,
                gain_5=0.5,
                global_act_count=6,
                round_act_count=1,
                runnable=True,
            )

    monkeypatch.setattr(cli, "_assets", lambda args: (object(), object()))
    monkeypatch.setattr(
        cli,
        "CodexProvider",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(cli, "GeneralsHLRound5Pipeline", FakePipeline)

    assert cli.handle(args) == 0
    assert str(captured["campaign_budget_receipt"]) == (
        "/derived/v4-budget.json"
    )
    assert captured["expected_rollback_hash"] == "3" * 64
    assert str(captured["prior_attempt_run_dir"]) == (
        "/runs/failed-v5-prompt"
    )
    output = capsys.readouterr().out
    assert '"evo_score_5": 0.5' in output
    assert '"gain_5": 0.5' in output
